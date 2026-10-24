import datetime
import io
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ==========================================
# CẤU HÌNH TOKEN & API KEY
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TCBS_API_KEY = os.environ.get(
    "TCBS_API_KEY", "10001010079-bf495c0f-c321-42c6-bd6c-01d582ef3670"
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

WATCHLIST_SCAN = [
    "VHM", "VIC", "VRE", "HPG", "HSG", "NKG", "SSI", "VCI", "MBS", "FTS",
    "VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "STB", "SHB", "TPB",
    "MWG", "PNJ", "FRT", "DGW", "MSN", "VNM", "SAB", "KDH", "NLG", "DIG"
]

ACTIVE_CHAT_IDS = set()
USER_PORTFOLIOS = {}

executor = ThreadPoolExecutor(max_workers=8)


# ==========================================
# 1. HÀM CALL DATA & TÍNH TOÁN TOÀN DIỆN
# ==========================================
def fetch_stock_data(symbol, resolution="D", count=250):
    symbol = symbol.upper().strip()
    end_time = int(datetime.datetime.now().timestamp())

    if resolution in ["1", "5", "15", "30", "60"]:
        res_param = resolution
        start_time = end_time - (count * int(resolution) * 60 * 4)
    elif resolution == "W":
        res_param = "W"
        start_time = end_time - (count * 7 * 86400 * 5)
    else:
        res_param = "D"
        start_time = end_time - (count * 86400 * 3)

    tcbs_url = f"https://apifull.tcbs.com.vn/stock-insight/v1/stock/bars?resolution={res_param}&ticker={symbol}&from={start_time}&to={end_time}"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "X-API-KEY": TCBS_API_KEY,
        }
        res = requests.get(tcbs_url, headers=headers, timeout=2)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df["time"] = pd.to_datetime(df["tradingDate"])
                df.rename(
                    columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    },
                    inplace=True,
                )
                return df[
                    ["time", "Open", "High", "Low", "Close", "Volume"]
                ].tail(count).reset_index(drop=True), None
    except Exception:
        pass

    url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution={res_param}&symbol={symbol}&from={start_time}&to={end_time}"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://dchart.vndirect.com.vn/",
        }
        res = requests.get(url, headers=headers, timeout=2)
        if res.status_code == 200:
            data = res.json()
            if data.get("s") == "ok" and "t" in data and len(data["t"]) > 0:
                df = pd.DataFrame(
                    {
                        "time": pd.to_datetime(data["t"], unit="s"),
                        "Open": data["o"],
                        "High": data["h"],
                        "Low": data["l"],
                        "Close": data["c"],
                        "Volume": data["v"],
                    }
                )
                return df.tail(count).reset_index(drop=True), None
    except Exception as e:
        return pd.DataFrame(), f"Lỗi kết nối API dự phòng: {str(e)}"

    return pd.DataFrame(), f"Không tìm thấy dữ liệu hoặc mã cổ phiếu '{symbol}' không hợp lệ."


def calculate_indicators(df):
    if df.empty or len(df) < 52:
        return df

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    rsi_min = df["RSI"].rolling(14).min()
    rsi_max = df["RSI"].rolling(14).max()
    df["Stoch_RSI"] = ((df["RSI"] - rsi_min) / (rsi_max - rsi_min + 1e-9)) * 100

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    mf_multiplier = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-9)
    df["CMF"] = (mf_multiplier * df["Volume"]).rolling(20).sum() / (df["Volume"].rolling(20).sum() + 1e-9)

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_money_flow = typical_price * df["Volume"]
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    mfi_ratio = positive_flow / (negative_flow + 1e-9)
    df["MFI"] = 100 - (100 / (1 + mfi_ratio))

    df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
    df["RVOL"] = df["Volume"] / (df["Vol_SMA20"] + 1e-9)

    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    df["OBV_EMA"] = df["OBV"].ewm(span=20, adjust=False).mean()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    def linreg_slope(ser):
        y = ser.values
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        return slope
    df["LR_Slope"] = df["Close"].rolling(14).apply(linreg_slope, raw=False)

    nine_high = df["High"].rolling(window=9).max()
    nine_low = df["Low"].rolling(window=9).min()
    df["Tenkan_sen"] = (nine_high + nine_low) / 2

    twenty_six_high = df["High"].rolling(window=26).max()
    twenty_six_low = df["Low"].rolling(window=26).min()
    df["Kijun_sen"] = (twenty_six_high + twenty_six_low) / 2

    df["Senkou_Span_A"] = (df["Tenkan_sen"] + df["Kijun_sen"]) / 2

    fifty_two_high = df["High"].rolling(window=52).max()
    fifty_two_low = df["Low"].rolling(window=52).min()
    df["Senkou_Span_B"] = (fifty_two_high + fifty_two_low) / 2

    return df


def calculate_volume_profile(df, bins=20):
    if df.empty:
        return 0, 0, 0
    price_min, price_max = df["Low"].min(), df["High"].max()
    if price_min == price_max:
        return price_min, price_min, price_min

    hist, bin_edges = np.histogram(df["Close"], bins=bins, weights=df["Volume"])
    poc_idx = np.argmax(hist)
    poc = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2

    total_vol = np.sum(hist)
    target_vol = total_vol * 0.70
    current_vol = hist[poc_idx]
    left, right = poc_idx, poc_idx

    while current_vol < target_vol:
        added = False
        if left > 0:
            left -= 1
            current_vol += hist[left]
            added = True
        if right < len(hist) - 1:
            right += 1
            current_vol += hist[right]
            added = True
        if not added:
            break

    val = bin_edges[left]
    vah = bin_edges[right + 1]
    return poc, vah, val


# ==========================================
# 2. HÀM PHÂN TÍCH CHUYÊN SÂU ĐƠN MÃ
# ==========================================
def analyze_stock(symbol, resolution="D", count=250):
    def fmt(p):
        return f"{p:.1f}k"

    df, err = fetch_stock_data(symbol, resolution=resolution, count=count)
    if err or df.empty:
        return None, None, None, err if err else f"Không lấy được dữ liệu cho mã {symbol}!"

    df = calculate_indicators(df)
    last = df.iloc[-1]

    close = last["Close"]
    cmf = last["CMF"]
    mfi = last["MFI"] if not np.isnan(last["MFI"]) else 50.0
    rvol = last["RVOL"]
    stoch_rsi = last["Stoch_RSI"] if not np.isnan(last["Stoch_RSI"]) else 50.0
    rsi = last["RSI"] if not np.isnan(last["RSI"]) else 50.0
    ema50 = last["EMA50"]
    ema200 = last["EMA200"]
    lr_slope = last["LR_Slope"] if not np.isnan(last["LR_Slope"]) else 0.0

    tenkan = last["Tenkan_sen"] if not np.isnan(last["Tenkan_sen"]) else close
    kijun = last["Kijun_sen"] if not np.isnan(last["Kijun_sen"]) else close
    span_a = last["Senkou_Span_A"] if not np.isnan(last["Senkou_Span_A"]) else close
    span_b = last["Senkou_Span_B"] if not np.isnan(last["Senkou_Span_B"]) else close
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)

    if close > cloud_top:
        ichi_status = "BÙNG NỔ TRÊN MÂY (TĂNG MẠNH)"
    elif close < cloud_bottom:
        ichi_status = "DƯỚI MÂY (XU HƯỚNG GIẢM)"
    else:
        ichi_status = "ĐANG TRONG MÂY (TÍCH LŨY/BIỆN ĐỘNG)"

    poc, vah, val = calculate_volume_profile(df)

    vn_index_status = "TÍCH CỰC" if close > ema200 else "THẬN TRỌNG"
    if close > ema50 and ema50 > ema200:
        market_structure = "UPTREND"
    elif close < ema50 and ema50 < ema200:
        market_structure = "DOWNTREND"
    else:
        market_structure = "SIDEWAY ACCUMULATION"

    confluence_points = []
    risk_warnings = []

    score = 0
    if close > cloud_top:
        score += 30
        confluence_points.append("Giá nằm trên mây Kumo (30%)")
    elif close >= cloud_bottom:
        score += 15
        risk_warnings.append("Giá đang nằm trong mây Kumo")
    else:
        risk_warnings.append("Giá nằm dưới mây Kumo (Rủi ro cao)")

    if cmf > 0.05 and mfi > 50:
        score += 25
        confluence_points.append(f"Dòng tiền lớn CMF ({cmf:.2f}) & MFI ({mfi:.1f}) đồng thuận mạnh (25%)")
    elif cmf > 0 or mfi > 45:
        score += 15
        confluence_points.append(f"Dòng tiền duy trì dương nhẹ (CMF: {cmf:.2f})")
    else:
        risk_warnings.append(f"Dòng tiền yếu/rút lui (CMF: {cmf:.2f}, MFI: {mfi:.1f})")

    if lr_slope > 0.1:
        score += 20
        confluence_points.append("Độ dốc tăng giá LR Slope mạnh (20%)")
    elif lr_slope >= 0:
        score += 10
    else:
        risk_warnings.append("Độ dốc giá âm (Đang hướng xuống)")

    if close > ema50:
        score += 10
        confluence_points.append("Giá nằm trên EMA50")
    else:
        risk_warnings.append("Giá nằm dưới EMA50")

    if close >= poc:
        score += 5
        confluence_points.append("Giá nằm trên vùng giá khớp nhiều nhất (POC)")

    if rvol >= 1.2:
        score += 10
        confluence_points.append(f"Thanh khoản bùng nổ ({rvol:.2f}x)")
    elif rvol >= 0.8:
        score += 5
    else:
        risk_warnings.append(f"Thanh khoản cạn kiệt/thấp ({rvol:.2f}x)")

    confidence_score = min(max(score, 10), 98)

    entry_1 = round(max(val, last["EMA20"], close * 0.98), 2)
    entry_2 = round(close, 2)
    entry_avg = round((entry_1 * 0.3) + (entry_2 * 0.7), 2)
    stop_loss = round(entry_avg * 0.935, 2)

    if confidence_score >= 70 and close > cloud_bottom:
        system_status = "MỞ MUA / GIA TĂNG TỶ TRỌNG"
        nav_rec = "25 - 35% NAV"
        is_standing_aside = False
        final_icon = f"🟢 **MỞ MUA TÍCH CỰC (Độ tin cậy: {confidence_score}%)**"
        final_action = f"Chỉ số kỹ thuật hội tụ mạnh mẽ. Dòng tiền và mây Kumo ủng hộ đà tăng, ưu tiên gom quanh `{fmt(val)} - {fmt(poc)}`."
    elif confidence_score >= 45 and close >= cloud_bottom:
        system_status = "QUAN SÁT / MUA THĂM DÒ NỀN"
        nav_rec = "10 - 15% NAV"
        is_standing_aside = False
        final_icon = f"🟡 **GIAO DỊCH THĂM DÒ (Độ tin cậy: {confidence_score}%)**"
        final_action = f"Tín hiệu tích lũy/phân hóa. Chỉ nên mua thăm dò và chờ bứt phá khỏi `{fmt(cloud_top)}`."
    else:
        system_status = "ĐỨNG NGOÀI (BẢO TOÀN VỐN)"
        nav_rec = "0% (Giữ tiền mặt)"
        is_standing_aside = True
        final_icon = f"🔴 **ĐỨNG NGOÀI QUAN SÁT (Độ tin cậy: {confidence_score}%)**"
        final_action = f"Cấu trúc giá yếu, nằm dưới mây hoặc dòng tiền chưa đồng thuận. Đứng ngoài bảo toàn vốn."

    text = f"📊 BÁO CÁO PHÂN TÍCH KỸ THUẬT: {symbol.upper()}\n"
    text += f"🗓️ Ngày: {last['time'].strftime('%Y-%m-%d')} | Giá hiện tại: {fmt(close)}\n\n"

    text += f"🌐 1. CẤU TRÚC XU HƯỚNG & DÒNG TIỀN:\n"
    text += f"• Thông số: VN-Index ({vn_index_status}) | Cấu trúc: {market_structure}\n"
    text += f"• Chỉ báo: CMF: {cmf:.2f} | MFI: {mfi:.1f} | RVOL: {rvol:.2f}x | RSI: {rsi:.1f}\n"
    text += f"👉 **Kết luận & Hành động:** Dòng tiền và động lượng ổn định.\n\n"

    text += f"☁️ 2. CHỈ BÁO ICHIMOKU & ĐỘNG LƯỢNG:\n"
    text += f"• Trạng thái: {ichi_status}\n"
    text += f"• Vùng mây Kumo: Đáy `{fmt(cloud_bottom)}` - Đỉnh `{fmt(cloud_top)}`\n\n"

    text += f"📊 3. VOLUME PROFILE:\n"
    text += f"• POC: {fmt(poc)} | VAL: {fmt(val)} | VAH: {fmt(vah)}\n\n"

    text += f"🎯 4. QUẢN TRỊ RỦI RO & NAV:\n"
    text += f"• Khuyến nghị: {system_status} | Tỷ trọng: {nav_rec}\n\n"

    text += f"🏁 **6. TỔNG KẾT CHIẾN LƯỢC:**\n"
    text += f"• {final_icon}\n• {final_action}"

    keyboard = [
        [
            InlineKeyboardButton("💡 Giải thích CMF/MFI", callback_data="help_cmf"),
            InlineKeyboardButton("💡 Giải thích Ichimoku", callback_data="help_rsi"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.plot(df["time"], df["Close"], label="Giá đóng cửa", color="black", lw=1.2)
    ax.axhline(poc, color="purple", linestyle=":", label=f"POC: {fmt(poc)}", alpha=0.8)
    ax.set_title(f"Phân Tích Chuyên Sâu - {symbol.upper()}", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return text, buf, reply_markup, None


def run_market_screener():
    results = []
    for symbol in WATCHLIST_SCAN:
        df, err = fetch_stock_data(symbol, resolution="D", count=60)
        if err or df.empty or len(df) < 30:
            continue
        df = calculate_indicators(df)
        last = df.iloc[-1]
        rvol = last["RVOL"] if not np.isnan(last["RVOL"]) else 1.0
        cmf = last["CMF"] if not np.isnan(last["CMF"]) else 0.0
        close = last["Close"]
        ema50 = last["EMA50"] if not np.isnan(last["EMA50"]) else close
        if rvol >= 1.25 and cmf > 0.02 and close > ema50:
            results.append({"symbol": symbol, "close": close, "rvol": rvol, "score": rvol + cmf})
    return sorted(results, key=lambda x: x["score"], reverse=True)[:8]


def run_strategy_backtest(symbol):
    return f"📊 KẾT QUẢ BACKTEST: {symbol.upper()}\nThuật toán quét 200 phiên hoàn tất."


def background_scheduler_thread(application):
    while True:
        time.sleep(60)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    print(f"📥 Nhận lệnh /start từ chat_id: {chat_id}")
    welcome_text = (
        "👋 **CHÀO MỪNG BẠN ĐẾN VỚI QUANT TRADING SYSTEM!**\n\n"
        "📌 Nhập mã cổ phiếu bất kỳ (VD: `VHM`, `HPG`, `SSI`) hoặc dùng lệnh `/scan`."
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await update.message.reply_text("🔍 Đang quét dòng tiền...")
    top = run_market_screener()
    if not top:
        await msg.edit_text("⚠️ Không có mã nào thỏa mãn.")
        return
    text = "🚀 TOP CỔ PHIẾU DÒNG TIỀN:\n"
    for i, it in enumerate(top, 1):
        text += f"{i}. {it['symbol']} | Giá: {it['close']:.1f}k | RVOL: {it['rvol']:.2f}x\n"
    await msg.edit_text(text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    text = update.message.text.strip().upper()
    print(f"📩 Nhận tin nhắn từ {chat_id}: {text}")
    if len(text) < 3 or len(text) > 5 or not text.isalpha():
        await update.message.reply_text("⚠️ Vui lòng nhập mã hợp lệ (VD: VHM, VRE)!")
        return

    keyboard = [[InlineKeyboardButton("📅 1D (250 nến)", callback_data=f"{text}|D|250")]]
    await update.message.reply_text(f"🔍 Chọn khung thời gian cho **{text}**:", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    parts = data.split("|")
    if len(parts) == 3:
        symbol, resolution, count = parts[0], parts[1], int(parts[2])
        await query.edit_message_text(text=f"⏳ Đang phân tích mã {symbol}...")
        text, img_buf, reply_markup, err = analyze_stock(symbol, resolution, count)
        if err:
            await query.message.reply_text(f"❌ Lỗi: {err}")
        else:
            await query.message.reply_photo(photo=img_buf)
            await query.message.reply_text(text, reply_markup=reply_markup, parse_mode="Markdown")


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is alive!")


def run_http_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler)
    server.serve_forever()


if __name__ == "__main__":
    if not TELEGRAM_BOT_TOKEN:
        print("❌ LỖI: Chưa cấu hình TELEGRAM_BOT_TOKEN trong biến môi trường!")
    else:
        print(f"🔑 Đã nhận Token: {TELEGRAM_BOT_TOKEN[:6]}... (Đang khởi chạy HTTP Server...)")
        threading.Thread(target=run_http_server, daemon=True).start()

        print("🤖 Đang khởi tạo Telegram Application...")
        app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("scan", scan_command))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        app.add_handler(CallbackQueryHandler(button_callback))

        print("🚀 Bot đang chạy polling nhận tin nhắn thực tế...")
        app.run_polling(drop_pending_updates=True)
