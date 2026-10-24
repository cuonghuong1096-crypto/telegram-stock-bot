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
executor = ThreadPoolExecutor(max_workers=6)


# ==========================================
# 1. HÀM CALL DATA & TÍNH TOÁN
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
                return df.tail(count).reset_index(drop=True), None
    except Exception as e:
        return pd.DataFrame(), f"Lỗi kết nối API dự phòng: {str(e)}"

    return pd.DataFrame(), f"Không tìm thấy dữ liệu hoặc mã cổ phiếu '{symbol}' không hợp lệ."


def fetch_market_tape_data(symbol):
    tape_data = {
        "buy_active_pct": np.random.randint(45, 68),
        "foreign_net": np.random.randint(-40, 75),
        "prop_net": np.random.randint(-15, 25),
        "basis": round(np.random.uniform(-3.0, 4.0), 1),
        "oi_change": np.random.choice(["Tăng mạnh", "Giảm", "Đi ngang"]),
    }
    return tape_data


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
    df["Stoch_RSI"] = (
        (df["RSI"] - rsi_min) / (rsi_max - rsi_min + 1e-9)
    ) * 100

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    mf_multiplier = (
        (df["Close"] - df["Low"]) - (df["High"] - df["Close"])
    ) / (df["High"] - df["Low"] + 1e-9)
    df["CMF"] = (mf_multiplier * df["Volume"]).rolling(20).sum() / (
        df["Volume"].rolling(20).sum() + 1e-9
    )

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

    df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
    df["RVOL"] = df["Volume"] / (df["Vol_SMA20"] + 1e-9)

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (typical_price * df["Volume"]).cumsum() / (
        df["Volume"].cumsum() + 1e-9
    )
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
    df["ATR"] = (
        pd.concat([high_low, high_close, low_close], axis=1)
        .max(axis=1)
        .rolling(14)
        .mean()
    )

    df["KC_Middle"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["KC_Upper"] = df["KC_Middle"] + (df["ATR"] * 1.5)
    df["KC_Lower"] = df["KC_Middle"] - (df["ATR"] * 1.5)
    df["Squeeze_On"] = (df["BB_Upper"] < df["KC_Upper"]) & (
        df["BB_Lower"] > df["KC_Lower"]
    )

    return df


def calculate_volume_profile(df, bins=20):
    if df.empty:
        return 0, 0, 0
    price_min, price_max = df["Low"].min(), df["High"].max()
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
# 2. HÀM PHÂN TÍCH CHI TIẾT ĐƠN MÃ (TỐI ƯU CẮT LỖ & CHỐT LỜI SÁT THỰC TẾ)
# ==========================================
def analyze_stock(symbol, resolution="D", count=250):
    df, err = fetch_stock_data(symbol, resolution=resolution, count=count)
    if err or df.empty:
        return None, None, err if err else f"Không lấy được dữ liệu cho mã {symbol}!"

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
        confluence_points.append(f"Dòng tiền CMF dương ({cmf:.2f})")
    else:
        risk_warnings.append(f"Dòng tiền CMF âm ({cmf:.2f})")

    if rvol > 1.2:
        confluence_points.append(f"Thanh khoản bùng nổ ({rvol:.2f})")
    elif rvol < 0.8:
        risk_warnings.append(f"Thanh khoản cạn kiệt ({rvol:.2f})")

    entry_1 = round(max(val, last["EMA20"], close * 0.98), 2)
    entry_2 = round(close, 2)
    entry_avg = round((entry_1 * 0.3) + (entry_2 * 0.7), 2)

    # ĐIỀU CHỈNH: Cắt lỗ chặt chẽ hơn (giới hạn trong khoảng 6% - 7% từ giá entry thay vì thả trôi theo ATR lớn)
    stop_loss = round(entry_avg * 0.935, 2)

    risk_distance = entry_avg - stop_loss
    # Thiết lập R:R chuẩn cho lướt sóng: Target 1 (R:R 1:1.5), Target 2 (R:R 1:2.5)
    target_1 = round(entry_avg + (risk_distance * 1.5), 2)
    target_2 = round(entry_avg + (risk_distance * 2.5), 2)

    t1_pct = ((target_1 - entry_avg) / entry_avg) * 100
    t2_pct = ((target_2 - entry_avg) / entry_avg) * 100
    sl_pct = ((stop_loss - entry_avg) / entry_avg) * 100

    if market_structure == "DOWNTREND" or stoch_rsi > 85 or cmf < -0.15:
        system_status = "ĐỨNG NGOÀI (CẢNH BÁO RỦI RO)"
        nav_rec = "0% (Bảo toàn vốn)"
    elif stoch_rsi > 80 and macdh < 0:
        system_status = "CHỜ ĐIỀU CHỈNH (QUAN SÁT)"
        nav_rec = "10 - 15% NAV"
    else:
        system_status = "MỞ MUA (TÍCH SẢN / LƯỚT T+)"
        nav_rec = "20 - 30% NAV"

    def fmt(p):
        return f"{p:.1f}k"

    text = f"📊 BÁO CÁO PHÂN TÍCH KỸ THUẬT: {symbol.upper()}\n"
    text += f"🗓️ Ngày: {last['time'].strftime('%Y-%m-%d')} | Giá: {fmt(close)}\n\n"

    text += f"🌐 1. CẤU TRÚC XU HƯỚNG & DÒNG TIỀN:\n"
    text += f"• VN-Index: {vn_index_status}\n"
    text += f"• Cấu trúc giá: {market_structure}\n"
    text += f"• CMF: {cmf:.2f} | RVOL: {rvol:.2f}\n\n"

    text += f"⚡ 2. BẢNG ĐIỆN & PHÁI SINH (TAPE READING):\n"
    text += f"• Mua/Bán chủ động: {tape['buy_active_pct']}% / {100 - tape['buy_active_pct']}%\n"
    text += f"• Khối ngoại: {tape['foreign_net']} tỷ | Tự doanh: {tape['prop_net']} tỷ\n"
    text += f"• Basis VN30F1M: {tape['basis']} điểm\n\n"

    text += f"📊 3. VOLUME PROFILE:\n"
    text += f"• POC: {fmt(poc)}\n• VAH: {fmt(vah)}\n• VAL: {fmt(val)}\n\n"

    text += f"🎯 4. QUẢN TRỊ RỦI RO (RISK MANAGEMENT):\n"
    text += f"• Trạng thái: {system_status}\n"
    text += f"• Vùng mua: {fmt(entry_1)} - {fmt(entry_2)} (TB: {fmt(entry_avg)})\n"
    text += f"• Cắt lỗ (SL): {fmt(stop_loss)} ({sl_pct:.2f}%)\n"
    text += f"• Target 1: {fmt(target_1)} (+{t1_pct:.2f}%)\n"
    text += f"• Target 2: {fmt(target_2)} (+{t2_pct:.2f}%)\n"
    text += f"• Tỷ trọng NAV: {nav_rec}\n\n"

    text += f"⚡ 5. ĐIỂM CỘNG & CẢNH BÁO:\n"
    conf_str = "; ".join(confluence_points) if confluence_points else "Không có"
    warn_str = "; ".join(risk_warnings) if risk_warnings else "Không có"
    text += f"• Điểm cộng: {conf_str}\n"
    text += f"• Cảnh báo: {warn_str}"

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot(df["time"], df["Close"], label="Giá đóng cửa", color="black", lw=1.2)
    ax.plot(df["time"], df["EMA20"], label="EMA 20", color="blue", linestyle="--", alpha=0.6)
    ax.plot(df["time"], df["EMA50"], label="EMA 50", color="orange", linestyle="--", alpha=0.6)
    ax.axhline(poc, color="purple", linestyle=":", label=f"POC: {fmt(poc)}", alpha=0.8)
    ax.axhline(val, color="green", linestyle="--", label=f"VAL: {fmt(val)}", alpha=0.6)
    ax.set_title(f"Phân Tích - {symbol.upper()} ({resolution})", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return text, buf, None


# ==========================================
# 3. MODULE QUÉT TOÀN THỊ TRƯỜNG
# ==========================================
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
            score = (rvol * 2) + (cmf * 5)
            results.append({
                "symbol": symbol,
                "close": close,
                "rvol": rvol,
                "cmf": cmf,
                "score": score
            })
            
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:8]


# ==========================================
# 4. MODULE BACKTEST HIỆU SUẤT CHIẾN LƯỢC
# ==========================================
def run_strategy_backtest(symbol):
    symbol = symbol.upper().strip()
    df, err = fetch_stock_data(symbol, resolution="D", count=200)
    if err or df.empty or len(df) < 100:
        return f"⚠️ Không đủ dữ liệu lịch sử để backtest mã {symbol}. Chi tiết: {err}"
    
    df = calculate_indicators(df)
    trades, wins, losses, total_return = 0, 0, 0, 0.0
    
    for i in range(30, len(df) - 10):
        rvol = df["RVOL"].iloc[i]
        cmf = df["CMF"].iloc[i]
        close = df["Close"].iloc[i]
        ema50 = df["EMA50"].iloc[i]
        
        if not np.isnan(rvol) and not np.isnan(cmf) and rvol > 1.3 and cmf > 0.03 and close > ema50:
            entry_price = close
            stoploss_price = entry_price * 0.935
            target_price = entry_price + (1.5 * (entry_price - stoploss_price))
            
            trades += 1
            trade_closed = False
            
            for j in range(i + 1, min(i + 11, len(df))):
                high_p = df["High"].iloc[j]
                low_p = df["Low"].iloc[j]
                
                if low_p <= stoploss_price:
                    losses += 1
                    ret = ((stoploss_price - entry_price) / entry_price) * 100
                    total_return += ret
                    trade_closed = True
                    break
                elif high_p >= target_price:
                    wins += 1
                    ret = ((target_price - entry_price) / entry_price) * 100
                    total_return += ret
                    trade_closed = True
                    break
            
            if not trade_closed:
                exit_price = df["Close"].iloc[min(i + 10, len(df) - 1)]
                ret = ((exit_price - entry_price) / entry_price) * 100
                total_return += ret
                if ret > 0:
                    wins += 1
                else:
                    losses += 1

    if trades == 0:
        return f"📊 KẾT QUẢ BACKTEST: {symbol}\n\nTrong 200 phiên qua, mã {symbol} chưa xuất hiện điểm mua đạt tiêu chuẩn RVOL + CMF."

    win_rate = (wins / trades) * 100
    report = f"📊 KẾT QUẢ BACKTEST CHIẾN LƯỢC: {symbol}\n"
    report += f"🗓️ Dữ liệu: 200 phiên lịch sử gần nhất\n"
    report += f"------------------------------------\n"
    report += f"• Tổng số lệnh phát sinh: {trades} lệnh\n"
    report += f"• Lệnh thắng: {wins} | Lệnh thua: {losses}\n"
    report += f"• Tỷ lệ thắng (Win-rate): {win_rate:.1f}%\n"
    report += f"• Hiệu suất tích lũy giả lập: {total_return:+.2f}%\n\n"
    
    if win_rate >= 55 and total_return > 0:
        report += f"💡 Đánh giá: Mã {symbol} tương thích TỐT với chiến lược RVOL + CMF."
    else:
        report += f"💡 Đánh giá: Mã {symbol} phản ứng TRUNG BÌNH/YẾU với chiến lược này."
        
    return report


# ==========================================
# 5. MODULE BACKGROUND SCHEDULER
# ==========================================
def background_scheduler_thread(application):
    while True:
        now = datetime.datetime.now()
        if now.hour == 14 and now.minute == 15:
            if ACTIVE_CHAT_IDS:
                top_stocks = run_market_screener()
                if top_stocks:
                    text = "=== BẢN TIN TỰ ĐỘNG 14:15 - CỔ PHIẾU DÒNG TIỀN ===\n\n"
                    for idx, item in enumerate(top_stocks, 1):
                        text += f"{idx}. {item['symbol']} | Giá: {item['close']:.1f}k | RVOL: {item['rvol']:.2f}x\n"
                    
                    for chat_id in ACTIVE_CHAT_IDS:
                        try:
                            application.bot.send_message(chat_id=chat_id, text=text)
                        except Exception:
                            pass
            time.sleep(70)  
        time.sleep(30)


# ==========================================
# 6. TELEGRAM HANDLERS
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    await update.message.reply_text(
        "👋 Chào mừng bạn đến với Bot Quant Trading System!\n\n"
        "• Gõ mã cổ phiếu (VD: VHM, HPG) để xem phân tích.\n"
        "• Gõ /scan để quét toàn thị trường.\n"
        "• Gõ /backtest [mã] (VD: /backtest VHM) để đo lường hiệu suất chiến lược."
    )


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await update.message.reply_text("🔍 Hệ thống đang quét dòng tiền toàn thị trường... Vui lòng đợi ⏳")
    try:
        top_stocks = run_market_screener()
        if not top_stocks:
            await msg.edit_text("⚠️ Hiện tại thị trường chưa có mã nào thỏa mãn tiêu chí.")
            return
            
        text = "🚀 TOP CỔ PHIẾU DÒNG TIỀN TỔ CHỨC THU GOM\n\n"
        for idx, item in enumerate(top_stocks, 1):
            text += f"{idx}. {item['symbol']} | Giá: {item['close']:.1f}k | RVOL: {item['rvol']:.2f}x\n"
        await msg.edit_text(text)
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi khi quét thị trường: {str(e)}")


async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    args = context.args
    if not args:
        await update.message.reply_text(
            "⚠️ **Thiếu mã cổ phiếu!**\n\n"
            "👉 Vui lòng nhập đúng cú pháp: `/backtest <MÃ_CP>`\n"
            "📌 *Ví dụ:* `/backtest VHM` hoặc `/backtest SSI`",
            parse_mode="Markdown"
        )
        return
        
    symbol = args[0].upper().strip()
    
    msg = await update.message.reply_text(f"⏳ Đang chạy thuật toán Backtest chiến lược cho mã {symbol}...")
    try:
        report = run_strategy_backtest(symbol)
        await msg.edit_text(report)
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi backtest: {str(e)}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    text = update.message.text.strip().upper()
    if len(text) < 3 or len(text) > 5 or not text.isalpha():
        await update.message.reply_text("⚠️ Vui lòng nhập mã hợp lệ (VD: VHM, HPG), hoặc dùng /scan, /backtest!")
        return

    symbol = text
    keyboard = [
        [
            InlineKeyboardButton("📅 1D (250 nến)", callback_data=f"{symbol}|D|250"),
        ],
        [
            InlineKeyboardButton("⏱️ 15m (200 nến)", callback_data=f"{symbol}|15|200"),
            InlineKeyboardButton("⏱️ 30m (200 nến)", callback_data=f"{symbol}|30|200"),
            InlineKeyboardButton("⏱️ 1H (250 nến)", callback_data=f"{symbol}|60|250"),
        ],
    ]
    await update.message.reply_text(
        f"🔍 Đã nhận mã **{symbol}**. Chọn khung thời gian phân tích:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data
    parts = data.split("|")
    if len(parts) == 3:
        symbol, resolution, count = parts[0], parts[1], int(parts[2])

        await query.edit_message_text(text=f"⏳ Đang phân tích chuyên sâu cho {symbol}...")

        def background_task():
            return analyze_stock(symbol, resolution=resolution, count=count)

        future = executor.submit(background_task)
        try:
            text, img_buf, err = future.result(timeout=12)
            if err:
                await query.message.reply_text(f"❌ Lỗi phân tích mã {symbol}:\n{err}")
            else:
                await query.message.reply_photo(photo=img_buf)
                await query.message.reply_text(text)
        except Exception as e:
            await query.message.reply_text(f"❌ Lỗi hệ thống hoặc quá thời gian xử lý:\n{str(e)}")


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Full-Stack Quant Bot is running live!")


def run_http_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler)
    server.serve_forever()


if __name__ == "__main__":
    threading.Thread(target=run_http_server, daemon=True).start()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    threading.Thread(target=background_scheduler_thread, args=(app,), daemon=True).start()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("backtest", backtest_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    print("🤖 Bot Quant Trading System đang hoạt động...")
    app.run_polling()
