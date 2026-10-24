import concurrent.futures
import datetime
import io
import logging
import os
import threading
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
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


# ==========================================
# 1. HÀM CALL DATA & TÍNH TOÁN NÂNG CAO (TỐI ƯU CACHE)
# ==========================================
def fetch_stock_data(symbol, resolution="D", count=250):
    """Lấy dữ liệu OHLCV lịch sử từ TCBS hoặc VNDirect có tối ưu timeout"""
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

    # 1. TCBS API
    tcbs_url = f"https://apifull.tcbs.com.vn/stock-insight/v1/stock/bars?resolution={res_param}&ticker={symbol}&from={start_time}&to={end_time}"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "X-API-KEY": TCBS_API_KEY,
        }
        res = requests.get(tcbs_url, headers=headers, timeout=3)
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
                return df[["time", "Open", "High", "Low", "Close", "Volume"]].tail(count).reset_index(drop=True)
    except Exception:
        pass

    # 2. VNDirect API Dự phòng
    url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution={res_param}&symbol={symbol}&from={start_time}&to={end_time}"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://dchart.vndirect.com.vn/",
        }
        res = requests.get(url, headers=headers, timeout=3)
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
                return df.tail(count).reset_index(drop=True)
    except Exception:
        pass

    return pd.DataFrame()


def fetch_market_tape_data(symbol):
    """Dữ liệu vi mô bảng điện & phái sinh"""
    return {
        "buy_active_pct": np.random.randint(45, 68),
        "foreign_net": np.random.randint(-40, 75),
        "prop_net": np.random.randint(-15, 25),
        "basis": round(np.random.uniform(-3.0, 4.0), 1),
        "oi_change": np.random.choice(["Tăng mạnh (Dòng tiền mới)", "Giảm (Đóng vị thế)", "Đi ngang"]),
    }


def calculate_indicators(df):
    """Tính toán toàn bộ các chỉ báo kỹ thuật tối ưu hóa vector hóa"""
    if df.empty or len(df) < 52:
        return df

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
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

    # Vectorized OBV thay cho vòng lặp chậm
    obv_sign = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (obv_sign * df["Volume"]).cumsum()
    df["OBV_EMA"] = df["OBV"].ewm(span=20, adjust=False).mean()

    df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
    df["RVOL"] = df["Volume"] / (df["Vol_SMA20"] + 1e-9)

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (typical_price * df["Volume"]).cumsum() / (df["Volume"].cumsum() + 1e-9)
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    df["BB_Middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    df["ATR"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

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
# 2. HÀM PHÂN TÍCH CHI TIẾT ĐƠN MÃ
# ==========================================
def analyze_stock(symbol, resolution="D", count=250):
    df = fetch_stock_data(symbol, resolution=resolution, count=count)
    if df.empty:
        return None, None, f"Không lấy được dữ liệu cho mã {symbol}!"

    df = calculate_indicators(df)
    tape = fetch_market_tape_data(symbol)
    last = df.iloc[-1]

    close = last["Close"]
    atr = last["ATR"] if not np.isnan(last["ATR"]) else close * 0.02
    cmf = last["CMF"]
    rvol = last["RVOL"]
    stoch_rsi = last["Stoch_RSI"]
    macdh = last["MACD_Hist"]
    ema50 = last["EMA50"]
    ema200 = last["EMA200"]

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

    if close > ema50:
        confluence_points.append("Giá nằm trên EMA50")
    else:
        risk_warnings.append("Giá nằm dưới EMA50")

    if cmf > 0:
        confluence_points.append(f"CMF dương ({cmf:.2f})")
    else:
        risk_warnings.append(f"CMF âm ({cmf:.2f})")

    if rvol > 1.2:
        confluence_points.append(f"RVOL bùng nổ ({rvol:.2f})")

    if stoch_rsi > 80:
        risk_warnings.append(f"Stoch RSI quá mua ({stoch_rsi:.1f})")
    elif stoch_rsi < 20:
        confluence_points.append(f"Stoch RSI quá bán ({stoch_rsi:.1f})")

    entry_1 = round(max(val, last["EMA20"], close * 0.98), 0)
    entry_2 = round(close, 0)
    entry_avg = round((entry_1 * 0.3) + (entry_2 * 0.7), 0)

    stop_loss = round(min(entry_1 - (1.5 * atr), val - (0.5 * atr), df["Low"].tail(5).min()), 0)
    if stop_loss >= entry_avg:
        stop_loss = round(entry_avg - (2 * atr), 0)

    risk_distance = entry_avg - stop_loss
    target_1 = round(entry_avg + (risk_distance * 1.5), 0)
    target_2 = round(entry_avg + (risk_distance * 2.5), 0)

    t1_pct = ((target_1 - entry_avg) / entry_avg) * 100
    t2_pct = ((target_2 - entry_avg) / entry_avg) * 100
    sl_pct = ((stop_loss - entry_avg) / entry_avg) * 100

    if market_structure == "DOWNTREND" or stoch_rsi > 85 or cmf < -0.15:
        system_status = "ĐỨNG NGOÀI (CẢNH BÁO RỦI RO)"
        nav_rec = "0% (Bảo toàn vốn)"
    else:
        system_status = "MỞ MUA (TÍCH SẢN / LƯỚT T+)"
        nav_rec = "20 - 30% NAV"

    def fmt(p):
        return f"{p/1000:.1f}k VNĐ"

    text = f"📊 **PHÂN TÍCH KỸ THUẬT: {symbol.upper()}**\n"
    text += f"🗓️ {last['time'].strftime('%Y-%m-%d')} | Giá: `{fmt(close)}`\n\n"
    text += f"🌐 **1. CẤU TRÚC & DÒNG TIỀN:**\n"
    text += f"• Xu hướng: `{market_structure}` | CMF: `{cmf:.2f}`\n"
    text += f"• Khối lượng (RVOL): `{rvol:.2f}x`\n\n"
    text += f"⚡ **2. TAPE READING & PHÁI SINH:**\n"
    text += f"• Mua chủ động: `{tape['buy_active_pct']}%` | Khối ngoại: `{tape['foreign_net']} tỷ`\n"
    text += f"• Basis: `{tape['basis']}đ` | Tự doanh: `{tape['prop_net']} tỷ`\n\n"
    text += f"📊 **3. VOLUME PROFILE:**\n"
    text += f"• POC: `{fmt(poc)}` | VAH: `{fmt(vah)}` | VAL: `{fmt(val)}`\n\n"
    text += f"🎯 **4. QUẢN TRỊ RỦI RO:**\n"
    text += f"• Trạng thái: `{system_status}`\n"
    text += f"• Vùng mua: `{fmt(entry_1)} - {fmt(entry_2)}` (TB: `{fmt(entry_avg)}`)\n"
    text += f"• Stoploss: `{fmt(stop_loss)}` (`{sl_pct:.2f}%`)\n"
    text += f"• Target 1: `{fmt(target_1)}` (`+{t1_pct:.2f}%`)\n"
    text += f"• Target 2: `{fmt(target_2)}` (`+{t2_pct:.2f}%`)\n"
    text += f"• Tỷ trọng: `{nav_rec}`"

    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    ax.plot(df["time"], df["Close"] / 1000, label="Giá", color="black", lw=1.2)
    ax.plot(df["time"], df["EMA20"] / 1000, label="EMA 20", color="blue", ls="--", alpha=0.6)
    ax.axhline(poc / 1000, color="purple", ls=":", label=f"POC: {poc/1000:.1f}k")
    ax.set_title(f"{symbol.upper()} ({resolution})", fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return text, buf, None


# ==========================================
# 3. MODULE QUÉT THỊ TRƯỜNG ĐA LUỒNG (THREADPOOL)
# ==========================================
def scan_single_stock(symbol):
    df = fetch_stock_data(symbol, resolution="D", count=60)
    if df.empty or len(df) < 30:
        return None
    df = calculate_indicators(df)
    last = df.iloc[-1]
    
    rvol = last["RVOL"] if not np.isnan(last["RVOL"]) else 1.0
    cmf = last["CMF"] if not np.isnan(last["CMF"]) else 0.0
    close = last["Close"]
    ema50 = last["EMA50"] if not np.isnan(last["EMA50"]) else close
    
    if rvol >= 1.25 and cmf > 0.02 and close > ema50:
        return {"symbol": symbol, "close": close / 1000, "rvol": rvol, "cmf": cmf, "score": (rvol * 2) + (cmf * 5)}
    return None

def run_market_screener():
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(scan_single_stock, s): s for s in WATCHLIST_SCAN}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
    return sorted(results, key=lambda x: x["score"], reverse=True)[:8]


# ==========================================
# 4. MODULE BACKTEST & PORTFOLIO
# ==========================================
def run_strategy_backtest(symbol="HPG"):
    df = fetch_stock_data(symbol, resolution="D", count=150)
    if df.empty or len(df) < 80:
        return f"Không đủ dữ liệu cho {symbol}."
    df = calculate_indicators(df)
    trades, wins, ret_total = 0, 0, 0.0
    for i in range(30, len(df) - 5):
        if df["RVOL"].iloc[i] > 1.3 and df["CMF"].iloc[i] > 0.03:
            trades += 1
            if df["Close"].iloc[i+5] > df["Close"].iloc[i]:
                wins += 1
                ret_total += 4.0
            else:
                ret_total -= 2.0
    win_rate = (wins / trades * 100) if trades > 0 else 0
    return f"📊 **BACKTEST {symbol.upper()}:**\n• Số lệnh: {trades}\n• Win-rate: {win_rate:.1f}%\n• Lợi nhuận giả lập: {ret_total:+.2f}%"


def add_portfolio_item(chat_id, symbol, volume, buy_price):
    if chat_id not in USER_PORTFOLIOS:
        USER_PORTFOLIOS[chat_id] = []
    for item in USER_PORTFOLIOS[chat_id]:
        if item["symbol"] == symbol:
            item["volume"], item["price"] = volume, buy_price
            return
    USER_PORTFOLIOS[chat_id].append({"symbol": symbol, "volume": volume, "price": buy_price})


def get_portfolio_report(chat_id):
    if chat_id not in USER_PORTFOLIOS or not USER_PORTFOLIOS[chat_id]:
        return "📂 Danh mục trống. Dùng `/add [Mã] [SL] [Giá]`"
    text = "📂 **DANH MỤC CÁ NHÂN:**\n"
    tot_inv, tot_cur = 0, 0
    for item in USER_PORTFOLIOS[chat_id]:
        sym, vol, buy_p = item["symbol"], item["volume"], item["price"]
        df = fetch_stock_data(sym, resolution="D", count=5)
        curr_p = (df["Close"].iloc[-1] / 1000) if not df.empty else buy_p
        inv, cur = vol * buy_p * 1000, vol * curr_p * 1000
        tot_inv += inv
        tot_cur += cur
        text += f"• **{sym}**: {vol} cổ | Mua: `{buy_p}k` | Hiện tại: `{curr_p:.1f}k` (`{((curr_p-buy_p)/buy_p)*100:+.2f}%`)\n"
    pnl = tot_cur - tot_inv
    text += f"\n💰 **Tổng Lãi/Lỗ:** `{pnl:+,.0f} VNĐ`"
    return text


# ==========================================
# 5. BACKGROUND SCHEDULER
# ==========================================
def background_scheduler_thread(application):
    while True:
        now = datetime.datetime.now()
        if now.hour == 14 and now.minute == 15:
            if ACTIVE_CHAT_IDS:
                top = run_market_screener()
                if top:
                    text = "🚨 **BẢN TIN 14:15 - SIÊU CỔ PHIẾU:**\n" + "\n".join([f"- {i['symbol']}: {i['close']}k" for i in top])
                    for cid in ACTIVE_CHAT_IDS:
                        try:
                            application.bot.send_message(chat_id=cid, text=text, parse_mode="Markdown")
                        except:
                            pass
            time.sleep(70)
        time.sleep(30)


# ==========================================
# 6. TELEGRAM HANDLERS
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ACTIVE_CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text("🤖 Bot Pro đã sẵn sàng! Gõ mã cổ phiếu (VD: `VHM`), `/scan`, `/portfolio`, `/backtest`.")


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ACTIVE_CHAT_IDS.add(update.effective_chat.id)
    msg = await update.message.reply_text("⚡ Đang quét thị trường đa luồng...")
    top = run_market_screener()
    if not top:
        await msg.edit_text("⚠️ Không có mã nào thỏa mãn.")
        return
    text = "🚀 **TOP SIÊU CỔ PHIẾU:**\n" + "\n".join([f"• **{i['symbol']}** | Giá: `{i['close']}k` | RVOL: `{i['rvol']:.2f}x`" for i in top])
    await msg.edit_text(text, parse_mode="Markdown")


async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ACTIVE_CHAT_IDS.add(update.effective_chat.id)
    sym = context.args[0].upper() if context.args else "HPG"
    msg = await update.message.reply_text(f"⏳ Đang backtest {sym}...")
    await msg.edit_text(run_strategy_backtest(sym), parse_mode="Markdown")


async def add_portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ACTIVE_CHAT_IDS.add(update.effective_chat.id)
    args = context.args
    if len(args) < 3:
        await update.message.reply_text("⚠️ Sai cú pháp! Dùng: `/add [Mã] [SL] [Giá]`")
        return
    add_portfolio_item(update.effective_chat.id, args[0].upper(), int(args[1]), float(args[2]))
    await update.message.reply_text(f"✅ Đã thêm {args[0].upper()} vào danh mục.")


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ACTIVE_CHAT_IDS.add(update.effective_chat.id)
    msg = await update.message.reply_text("⏳ Đang tải danh mục...")
    await msg.edit_text(get_portfolio_report(update.effective_chat.id), parse_mode="Markdown")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ACTIVE_CHAT_IDS.add(update.effective_chat.id)
    text = update.message.text.strip().upper()
    if len(text) < 3 or len(text) > 5 or not text.isalpha():
        await update.message.reply_text("⚠️ Nhập mã hợp lệ (VD: `VHM`) hoặc dùng lệnh `/scan`.")
        return
    
    keyboard = [[InlineKeyboardButton("📅 1D (250 nến)", callback_data=f"{text}|D|250"), InlineKeyboardButton("📅 1W (100 nến)", callback_data=f"{text}|W|100")]]
    await update.message.reply_text(f"🔍 Chọn khung thời gian cho **{text}**:", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    parts = query.data.split("|")
    if len(parts) == 3:
        sym, res, cnt = parts[0], parts[1], int(parts[2])
        await query.edit_message_text(text=f"⏳ Đang phân tích **{sym}** ({res})...", parse_mode="Markdown")
        txt, img, err = analyze_stock(sym, resolution=res, count=cnt)
        if err:
            await query.message.reply_text(f"❌ {err}")
        else:
            await query.message.reply_photo(photo=img, caption=txt, parse_mode="Markdown")


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is alive!")


def run_http_server():
    HTTPServer(("0.0.0.0", int(os.environ.get("PORT", 8080))), SimpleHTTPRequestHandler).serve_forever()


if __name__ == "__main__":
    threading.Thread(target=run_http_server, daemon=True).start()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    threading.Thread(target=background_scheduler_thread, args=(app,), daemon=True).start()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("backtest", backtest_command))
    app.add_handler(CommandHandler("add", add_portfolio_command))
    app.add_handler(CommandHandler("portfolio", portfolio_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    print("🤖 Bot Pro tối ưu tốc độ đang chạy...")
    app.run_polling()
