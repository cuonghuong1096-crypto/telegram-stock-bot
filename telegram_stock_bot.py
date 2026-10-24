import datetime
import io
import logging
import os
import threading
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

# Configuration Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


# ==========================================
# 1. HÀM CALL DATA & TÍNH FULL CHỈ BÁO KỸ THUẬT
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
        res = requests.get(tcbs_url, headers=headers, timeout=5)
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
                ].tail(count).reset_index(drop=True)
    except Exception:
        pass

    url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution={res_param}&symbol={symbol}&from={start_time}&to={end_time}"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://dchart.vndirect.com.vn/",
        }
        res = requests.get(url, headers=headers, timeout=5)
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


def calculate_indicators(df):
    if df.empty or len(df) < 52:
        return df

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    rsi_min = df["RSI"].rolling(14).min()
    rsi_max = df["RSI"].rolling(14).max()
    df["Stoch_RSI"] = (
        (df["RSI"] - rsi_min) / (rsi_max - rsi_min + 1e-9)
    ) * 100

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_mf = typical_price * df["Volume"]
    pos_mf = (
        raw_mf.where(typical_price > typical_price.shift(1), 0)
        .rolling(14)
        .sum()
    )
    neg_mf = (
        raw_mf.where(typical_price < typical_price.shift(1), 0)
        .rolling(14)
        .sum()
    )
    mfr = pos_mf / neg_mf
    df["MFI"] = 100 - (100 / (1 + mfr))

    mf_multiplier = (
        (df["Close"] - df["Low"]) - (df["High"] - df["Close"])
    ) / (df["High"] - df["Low"] + 1e-9)
    mf_volume = mf_multiplier * df["Volume"]
    df["CMF"] = mf_volume.rolling(20).sum() / (
        df["Volume"].rolling(20).sum() + 1e-9
    )

    tp_vol = typical_price * df["Volume"]
    df["VWAP"] = tp_vol.cumsum() / (df["Volume"].cumsum() + 1e-9)

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

    df["BB_Middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(
        axis=1
    )
    df["ATR"] = true_range.rolling(14).mean()

    df["KC_Middle"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["KC_Upper"] = df["KC_Middle"] + (df["ATR"] * 1.5)
    df["KC_Lower"] = df["KC_Middle"] - (df["ATR"] * 1.5)
    df["Squeeze_On"] = (df["BB_Upper"] < df["KC_Upper"]) & (
        df["BB_Lower"] > df["KC_Lower"]
    )

    plus_dm = df["High"].diff()
    minus_dm = df["Low"].diff()
    plus_dm = np.where(
        (plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0
    )
    minus_dm = np.where(
        (minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0
    )
    tr14 = true_range.rolling(14).sum()
    plus_di = 100 * (
        pd.Series(plus_dm).rolling(14).sum() / (tr14 + 1e-9)
    )
    minus_di = 100 * (
        pd.Series(minus_dm).rolling(14).sum() / (tr14 + 1e-9)
    )
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df["ADX"] = dx.rolling(14).mean()

    slopes = [np.nan] * 19
    for i in range(19, len(df)):
        y = df["Close"].iloc[i - 19 : i + 1].values
        x = np.arange(20)
        slope, _ = np.polyfit(x, y, 1)
        slopes.append(slope)
    df["Slope"] = slopes

    df["Tenkan_sen"] = (
        df["High"].rolling(9).max() + df["Low"].rolling(9).min()
    ) / 2
    df["Kijun_sen"] = (
        df["High"].rolling(26).max() + df["Low"].rolling(26).min()
    ) / 2
    df["Senkou_Span_A"] = (
        (df["Tenkan_sen"] + df["Kijun_sen"]) / 2
    ).shift(26)
    df["Senkou_Span_B"] = (
        (df["High"].rolling(52).max() + df["Low"].rolling(52).min()) / 2
    ).shift(26)
    df["Chikou_Span"] = df["Close"].shift(-26)

    return df


def calculate_volume_profile(df, bins=20):
    if df.empty:
        return 0, 0, 0
    price_min = df["Low"].min()
    price_max = df["High"].max()
    if price_min == price_max:
        return price_min, price_min, price_min

    hist, bin_edges = np.histogram(
        df["Close"], bins=bins, weights=df["Volume"]
    )
    poc_idx = np.argmax(hist)
    poc = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2

    total_vol = np.sum(hist)
    target_vol = total_vol * 0.70
    current_vol = hist[poc_idx]
    left = poc_idx
    right = poc_idx

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


def analyze_stock(symbol, resolution="D", count=250):
    df = fetch_stock_data(symbol, resolution=resolution, count=count)
    if df.empty:
        return None, None, f"Không lấy được dữ liệu cho mã {symbol}!"

    df = calculate_indicators(df)
    last_row = df.iloc[-1]

    close = last_row["Close"]
    rsi = last_row["RSI"]
    stoch_rsi = last_row["Stoch_RSI"]
    cmf = last_row["CMF"]
    vwap = last_row["VWAP"]
    obv = last_row["OBV"]
    obv_ema = last_row["OBV_EMA"]
    adx = last_row["ADX"] if not np.isnan(last_row["ADX"]) else 20
    slope = last_row["Slope"]
    bb_lower = last_row["BB_Lower"]
    bb_upper = last_row["BB_Upper"]
    squeeze_on = last_row["Squeeze_On"]
    atr = last_row["ATR"] if not np.isnan(last_row["ATR"]) else close * 0.02
    kijun = last_row["Kijun_sen"]
    chikou = last_row["Chikou_Span"]

    ma50 = df["Close"].rolling(50).mean().iloc[-1]
    ma200 = (
        df["Close"].rolling(200).mean().iloc[-1]
        if len(df) >= 200
        else df["Close"].mean()
    )
    poc, vah, val = calculate_volume_profile(df)

    support = min(df["Low"].tail(20).min(), val, bb_lower)
    resistance = max(df["High"].tail(20).max(), vah, bb_upper)
    stop_loss = max(support - (atr * 1.5), close * 0.93)
    target_price = close + ((close - stop_loss) * 2.0)

    macro = (
        "🟢 TÍCH CỰC (UpTrend dài hạn)" if close > ma200 else "🔴 THẬN TRỌNG"
    )
    if close > ma200 and cmf > 0.05 and obv > obv_ema:
        macro += "\n   ⭐ [CHÉN THÁNH A+]: Dòng tiền lớn gom mạnh!"

    st_reasons = []
    st_score = 0
    if adx > 25:
        st_score += 1
        st_reasons.append(f"Xu hướng mạnh (ADX: {adx:.1f} > 25)")
    else:
        st_reasons.append(f"Thị trường Sideway (ADX: {adx:.1f})")

    if stoch_rsi < 20:
        st_score += 2
        st_reasons.append(
            f"Stoch RSI Quá bán ({stoch_rsi:.1f}) -> Sắp có nhịp hồi T+"
        )
    elif stoch_rsi > 80:
        st_score -= 2
        st_reasons.append(f"Stoch RSI Quá mua ({stoch_rsi:.1f}) -> Rủi ro")

    if close >= vwap:
        st_score += 1.5
        st_reasons.append(f"Giá nằm trên VWAP ({vwap:,.0f}): Phe mua kiểm soát")
    else:
        st_score -= 1.5
        st_reasons.append(f"Giá dưới VWAP ({vwap:,.0f}): Lực bán ngắn hạn")

    st_action = (
        "🟢 MUA GOM / LƯỚT T+ TÍCH CỰC"
        if st_score >= 2.5
        else (
            "🔴 CANH BÁN / HẠ TỶ TRỌNG"
            if st_score <= -1.5
            else "🟡 QUAN SÁT THÊM"
        )
    )

    mt_score = 0
    mt_reasons = []
    if close > kijun and (not np.isnan(chikou) and close > chikou):
        mt_score += 3
        mt_reasons.append(
            "Ichimoku xác nhận: Giá vượt Kijun VÀ trên Chikou (Xác suất tăng 80%)"
        )
    else:
        mt_score += 1
        mt_reasons.append("Đang dao động tích lũy trung hạn")

    if slope > 0:
        mt_score += 1
        mt_reasons.append(f"Độ dốc hồi quy tuyến tính Dương (+{slope:.2f})")
    else:
        mt_score -= 1
        mt_reasons.append(f"Độ dốc hồi quy tuyến tính Âm ({slope:.2f})")

    mt_action = (
        "🟢 NẮM GIỮ / TÍCH SẢN TRUNG HẠN"
        if mt_score >= 2.5
        else (
            "🔴 THOÁT HÀNG / BẢO TOÀN VỐN"
            if mt_score <= -1.5
            else "🟡 CHỜ BREAKOUT"
        )
    )

    lt_action = (
        "🟢 ĐẦU TƯ GIÁ TRỊ / MUA TÍCH SẢN LỚN"
        if ma50 > ma200 and close > ma200
        else "🟡 THEO DÕI THÊM VĨ MÔ"
    )

    resolution_labels = {
        "D": "1D (Ngày)",
        "W": "1W (Tuần)",
        "15": "15m",
        "30": "30m",
        "60": "1H",
    }
    res_display = resolution_labels.get(resolution, resolution)

    text = f"📊 **BÁO CÁO PHÂN TÍCH CHUYÊN SÂU: {symbol}**\n"
    text += f"🗓️ Ngày: {last_row['time'].strftime('%Y-%m-%d')} | Khung: {res_display} | Số nến: {count}\n"
    text += f"💵 Giá đóng cửa: `{close:,.0f} VNĐ`\n\n"

    text += f"🌐 **1. ĐÁNH GIÁ VĨ MÔ & DÒNG TIỀN:**\n"
    text += f"• Trạng thái: {macro}\n"
    text += (
        f"• Dòng tiền Chaikin (CMF): `{cmf:.2f}`"
        f" ({'Tiền vào' if cmf > 0 else 'Tiền ra'})\n"
    )
    text += (
        f"• Dòng tiền thông minh (OBV):"
        f" `{'Tích cực' if obv > obv_ema else 'Thận trọng'}`\n"
    )
    if squeeze_on:
        text += f"⚡ **[CẢNH BÁO SQUEEZE]:** Biến động nén, sắp bứt phá mạnh!\n"

    text += f"\n📊 **2. HỒ SƠ KHỐI LƯỢNG (VOLUME PROFILE):**\n"
    text += f"• POC (Tâm dòng tiền hút giá): `{poc:,.0f} VNĐ`\n"
    text += f"• VAH (Kháng cự Volume): `{vah:,.0f} VNĐ`\n"
    text += f"• VAL (Hỗ trợ Volume): `{val:,.0f} VNĐ`\n"

    text += f"\n🎯 **3. QUẢN TRỊ RỦI RO & CHIẾN LƯỢC GIAO DỊCH:**\n"
    text += (
        f"• 🟢 **Vùng mua lý tưởng (Entry):**"
        f" `{support * 1.01:,.0f} - {close:,.0f} VNĐ`\n"
    )
    text += f"• 🔴 **Điểm cắt lỗ kỷ luật (Stoploss):** `{stop_loss:,.0f} VNĐ`\n"
    text += (
        f"• 🎯 **Mục tiêu chốt lời (Target R:R 1:2):**"
        f" `{target_price:,.0f} VNĐ`\n"
    )

    text += f"\n⚡ **4. KHUYẾN NGHỊ CHI TIẾT ĐA KHUNG:**\n"
    text += f"• **Ngắn hạn (T+):** *{st_action}*\n"
    for r in st_reasons:
        text += f"   - {r}\n"
    text += f"• **Trung hạn (1-3 tháng):** *{mt_action}*\n"
    for r in mt_reasons:
        text += f"   - {r}\n"
    text += f"• **Dài hạn (Tích sản):** *{lt_action}*\n"

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot(
        df["time"], df["Close"], label="Giá đóng cửa", color="black", lw=1.2
    )
    ax.plot(
        df["time"],
        df["VWAP"],
        label="VWAP",
        color="purple",
        linestyle="-.",
        alpha=0.7,
    )
    ax.axhline(
        poc,
        color="orange",
        linestyle=":",
        label=f"POC: {poc:,.0f}",
        alpha=0.8,
    )
    ax.plot(
        df["time"],
        df["Tenkan_sen"],
        label="Tenkan",
        color="red",
        alpha=0.6,
    )
    ax.plot(
        df["time"],
        df["Kijun_sen"],
        label="Kijun",
        color="blue",
        alpha=0.6,
    )
    ax.fill_between(
        df["time"],
        df["Senkou_Span_A"],
        df["Senkou_Span_B"],
        where=df["Senkou_Span_A"] >= df["Senkou_Span_B"],
        facecolor="green",
        alpha=0.15,
    )
    ax.fill_between(
        df["time"],
        df["Senkou_Span_A"],
        df["Senkou_Span_B"],
        where=df["Senkou_Span_A"] < df["Senkou_Span_B"],
        facecolor="red",
        alpha=0.15,
    )
    ax.set_title(
        f"Phân Tích Kỹ Thuật - {symbol} ({res_display} - {count} nến)",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=7)
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return text, buf, None


# ==========================================
# 2. XỬ LÝ LỆNH & GIAO DIỆN NÚT BẤM TELEGRAM
# ==========================================
async def start_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "👋 Chào mừng bạn đến với Bot Phân Tích Chứng Khoán!\n\n"
        "Hãy gõ mã cổ phiếu bạn muốn kiểm tra (Ví dụ: `HPG`, `VHM`, `SSI`...)"
    )


async def handle_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    text = update.message.text.strip().upper()
    if len(text) < 3 or len(text) > 5 or not text.isalpha():
        await update.message.reply_text(
            "⚠️ Vui lòng nhập mã cổ phiếu hợp lệ (VD: HPG, VHM, SSI)!"
        )
        return

    symbol = text
    # Tạo bảng nút bấm chọn Khung thời gian & Số nến giống PC
    keyboard = [
        [
            InlineKeyboardButton(
                "📅 1D (250 nến)", callback_data=f"{symbol}|D|250"
            ),
            InlineKeyboardButton(
                "📅 1W (100 nến)", callback_data=f"{symbol}|W|100"
            ),
        ],
        [
            InlineKeyboardButton(
                "⏱️ 15m (200 nến)", callback_data=f"{symbol}|15|200"
            ),
            InlineKeyboardButton(
                "⏱️ 30m (200 nến)", callback_data=f"{symbol}|30|200"
            ),
            InlineKeyboardButton(
                "⏱️ 1H (250 nến)", callback_data=f"{symbol}|60|250"
            ),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"🔍 Đã nhận mã **{symbol}**.\n👉 Hãy chọn khung thời gian và số nến bạn muốn phân tích:",
        reply_markup=reply_markup,
        parse_mode="Markdown",
    )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data  # Định dạng: SYMBOL|RESOLUTION|COUNT
    parts = data.split("|")
    if len(parts) == 3:
        symbol, resolution, count = parts[0], parts[1], int(parts[2])

        await query.edit_message_text(
            text=f"⏳ Đang phân tích mã **{symbol}** (Khung: {resolution}, Số nến: {count})...",
            parse_mode="Markdown",
        )

        text, img_buf, err = analyze_stock(
            symbol, resolution=resolution, count=count
        )
        if err:
            await query.message.reply_text(f"❌ {err}")
        else:
            await query.message.reply_photo(
                photo=img_buf, caption=text, parse_mode="Markdown"
            )


# Tạo web server giả lập để Render duy trì service Free
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot Telegram is running live!")


def run_http_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler)
    server.serve_forever()


if __name__ == "__main__":
    threading.Thread(target=run_http_server, daemon=True).start()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(
        MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    )
    app.add_handler(CallbackQueryHandler(button_callback))

    print("🤖 Bot Telegram đang hoạt động...")
    app.run_polling()
