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


def fetch_market_tape_data(symbol, df_last=None):
    symbol = symbol.upper().strip()
    buy_active = 50
    if df_last is not None and not df_last.empty:
        high = df_last["High"]
        low = df_last["Low"]
        close = df_last["Close"]
        if high > low:
            buy_active = int(((close - low) / (high - low)) * 100)
            buy_active = max(10, min(90, buy_active))

    tape_data = {
        "buy_active_pct": buy_active,
        "foreign_net": "N/A",
        "prop_net": "N/A",
        "put_through_val": "N/A",
        "foreign_room_pct": "N/A",
    }

    try:
        url = f"https://f247.com/api/v1/stocks/{symbol}"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=2)
        if res.status_code == 200:
            d = res.json()
            tape_data["foreign_net"] = f"{round(d.get('foreign_net_val', 0) / 1e9, 1)} tỷ"
            tape_data["foreign_room_pct"] = f"{round(d.get('foreign_room', 0), 1)}%"
    except Exception:
        pass

    try:
        url_tcbs = f"https://apifull.tcbs.com.vn/stock-insight/v1/intraday/{symbol}/his/paging?page=0&size=1"
        res_t = requests.get(url_tcbs, headers={"User-Agent": "Mozilla/5.0"}, timeout=2)
        if res_t.status_code == 200:
            dt = res_t.json()
            if "data" in dt and len(dt["data"]) > 0:
                cp = dt["data"][0]
                tape_data["put_through_val"] = f"{round(cp.get('ptValue', 0) / 1e9, 1)} tỷ"
    except Exception:
        pass

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
# 2. HÀM PHÂN TÍCH CHI TIẾT ĐƠN MÃ
# ==========================================
def analyze_stock(symbol, resolution="D", count=250):
    df, err = fetch_stock_data(symbol, resolution=resolution, count=count)
    if err or df.empty:
        return None, None, None, err if err else f"Không lấy được dữ liệu cho mã {symbol}!"

    df = calculate_indicators(df)
    last = df.iloc[-1]
    tape = fetch_market_tape_data(symbol, df_last=last)

    close = last["Close"]
    cmf = last["CMF"]
    rvol = last["RVOL"]
    stoch_rsi = last["Stoch_RSI"] if not np.isnan(last["Stoch_RSI"]) else 50.0
    rsi = last["RSI"] if not np.isnan(last["RSI"]) else 50.0
    macdh = last["MACD_Hist"]
    ema50 = last["EMA50"]
    ema200 = last["EMA200"]
    squeeze_on = last["Squeeze_On"]

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

    if squeeze_on:
        risk_warnings.append("Đang tích nén chặt (Squeeze On)")
    else:
        confluence_points.append("Biến động bình thường")

    entry_1 = round(max(val, last["EMA20"], close * 0.98), 2)
    entry_2 = round(close, 2)
    entry_avg = round((entry_1 * 0.3) + (entry_2 * 0.7), 2)
    stop_loss = round(entry_avg * 0.935, 2)

    risk_distance = entry_avg - stop_loss
    target_1 = round(entry_avg + (risk_distance * 1.5), 2)
    target_2 = round(entry_avg + (risk_distance * 2.5), 2)

    t1_pct = ((target_1 - entry_avg) / entry_avg) * 100
    t2_pct = ((target_2 - entry_avg) / entry_avg) * 100
    sl_pct = ((stop_loss - entry_avg) / entry_avg) * 100

    if market_structure == "DOWNTREND" or stoch_rsi > 85 or cmf < -0.15:
        system_status = "ĐỨNG NGOÀI (CẢNH BÁO RỦI RO)"
        nav_rec = "0% (Bảo toàn vốn)"
        is_standing_aside = True
    elif stoch_rsi > 80 and macdh < 0:
        system_status = "CHỜ ĐIỀU CHỈNH (QUAN SÁT)"
        nav_rec = "10 - 15% NAV"
        is_standing_aside = False
    else:
        system_status = "MỞ MUA (TÍCH SẢN / LƯỚT T+)"
        nav_rec = "20 - 30% NAV"
        is_standing_aside = False

    def fmt(p):
        return f"{p:.1f}k"

    # Xây dựng báo cáo tối ưu hành động từng phần
    text = f"📊 BÁO CÁO PHÂN TÍCH KỸ THUẬT: {symbol.upper()}\n"
    text += f"🗓️ Ngày: {last['time'].strftime('%Y-%m-%d')} | Giá hiện tại: {fmt(close)}\n\n"

    # MỤC 1
    text += f"🌐 1. CẤU TRÚC XU HƯỚNG & DÒNG TIỀN:\n"
    text += f"• Chỉ số: VN-Index ({vn_index_status}) | Cấu trúc: {market_structure}\n"
    text += f"• Chỉ báo: CMF: {cmf:.2f} | RVOL: {rvol:.2f} | RSI: {rsi:.1f} | StochRSI: {stoch_rsi:.1f}\n"
    if cmf > 0 and rvol >= 1.0:
        text += f"👉 **Hành động Mục 1:** Dòng tiền tích cực, ưu tiên giữ vị thế hoặc canh gia tăng tỷ trọng khi rung lắc.\n\n"
    else:
        text += f"👉 **Hành động Mục 1:** Dòng tiền yếu/âm, hạn chế mua mới, ưu tiên quản trị và quan sát.\n\n"

    # MỤC 2
    text += f"⚡ 2. BẢNG ĐIỆN & ÁP LỰC CUNG CẦU (TAPE READING):\n"
    text += f"• Mua/Bán chủ động: {tape['buy_active_pct']}% / {100 - tape['buy_active_pct']}%\n"
    text += f"• Khối ngoại Net: `{tape['foreign_net']}` | Room còn lại: `{tape['foreign_room_pct']}`\n"
    text += f"• Giao dịch Thỏa thuận: `{tape['put_through_val']}`\n"
    if tape['buy_active_pct'] >= 55:
        text += f"👉 **Hành động Mục 2:** Lực mua chủ động lấn át, có thể canh nhịp test hỗ trợ ngắn hạn để vào lệnh.\n\n"
    else:
        text += f"👉 **Hành động Mục 2:** Áp lực bán chiếm ưu thế, kiên nhẫn chờ giá về vùng chiết khấu sâu hơn.\n\n"

    # MỤC 3
    text += f"📊 3. VOLUME PROFILE:\n"
    text += f"• POC (Giá khớp nhiều nhất): {fmt(poc)}\n"
    text += f"• VAL (Hỗ trợ dưới): {fmt(val)} | VAH (Kháng cự trên): {fmt(vah)}\n"
    text += f"👉 **Hành động Mục 3:** Đặt vùng mua gom quanh **{fmt(val)} - {fmt(poc)}** và chuẩn bị chốt lời khi tiệm cận **{fmt(vah)}**.\n\n"

    # MỤC 4
    text += f"🎯 4. QUẢN TRỊ RỦI RO (RISK MANAGEMENT):\n"
    text += f"• Trạng thái hệ thống: {system_status}\n"
    text += f"• Khuyến nghị NAV phân bổ: {nav_rec}\n"
    if is_standing_aside:
        text += f"• 🛡️ *Watchzone an toàn:* {fmt(entry_1)} - {fmt(entry_2)}\n"
        text += f"👉 **Hành động Mục 4:** Đứng ngoài quan sát, chưa giải ngân cho đến khi cấu trúc giá cải thiện.\n\n"
    else:
        text += f"• Vùng mua khuyến nghị: {fmt(entry_1)} - {fmt(entry_2)} (TB: {fmt(entry_avg)})\n"
        text += f"• Cắt lỗ (SL mẫu): {fmt(stop_loss)} ({sl_pct:.2f}%)\n"
        text += f"• Target 1: {fmt(target_1)} (+{t1_pct:.2f}%)\n"
        text += f"👉 **Hành động Mục 4:** Thực hiện giải ngân theo phân bổ NAV, tuân thủ tuyệt đối điểm cắt lỗ `(SL: {fmt(stop_loss)})`.\n\n"

    # MỤC 5
    text += f"⚡ 5. ĐIỂM CỘNG & CẢNH BÁO TỔNG HỢP:\n"
    conf_str = "; ".join(confluence_points) if confluence_points else "Không có"
    warn_str = "; ".join(risk_warnings) if risk_warnings else "Không có"
    text += f"• Điểm cộng: {conf_str}\n"
    text += f"• Cảnh báo: {warn_str}\n"
    text += f"👉 **Hành động Tổng kết:** Cân đối tỷ trọng danh mục cá nhân dựa trên các tín hiệu trên trước khi đưa ra quyết định giao dịch cuối cùng."

    keyboard = [
        [
            InlineKeyboardButton("💡 Giải thích CMF/RVOL", callback_data="help_cmf"),
            InlineKeyboardButton("💡 Giải thích RSI", callback_data="help_rsi"),
        ],
        [
            InlineKeyboardButton("💡 Giải thích Volume Profile", callback_data="help_vp"),
            InlineKeyboardButton("💡 Hướng dẫn Lệnh Cá Nhân", callback_data="help_portfolio"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

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

    return text, buf, reply_markup, None


# ==========================================
# 3. MODULE QUÉT TOÀN THỊ TRƯỜNG & GIÁM SÁT SL
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
    report += f"• Hiệu suất tích lũy giả lập: {total_return:+.2f}%\n"
    return report


# ==========================================
# 4. BACKGROUND SCHEDULER & GIÁM SÁT VI PHẠM SL
# ==========================================
def background_scheduler_thread(application):
    while True:
        now = datetime.datetime.now()
        
        if USER_PORTFOLIOS:
            for chat_id, portfolio in list(USER_PORTFOLIOS.items()):
                for symbol, pos in list(portfolio.items()):
                    df, err = fetch_stock_data(symbol, resolution="D", count=5)
                    if not err and not df.empty:
                        current_price = df.iloc[-1]["Close"]
                        sl_price = pos["sl"]
                        if current_price <= sl_price:
                            alert_text = (
                                f"🚨 **CẢNH BÁO VI PHẠM CẮT LỖ (STOPLOSS)!** 🚨\n\n"
                                f"• Mã cổ phiếu: **{symbol}**\n"
                                f"• Giá mua của bạn: `{pos['entry']}`\n"
                                f"• Ngưỡng cắt lỗ cài đặt: `{sl_price}`\n"
                                f"• **Giá hiện tại:** `{current_price}` 📉\n\n"
                                f"⚠️ Đã chạm hoặc thủng điểm cắt lỗ! Hãy cân nhắc xử lý ngay để bảo toàn vốn."
                            )
                            try:
                                application.bot.send_message(chat_id=chat_id, text=alert_text, parse_mode="Markdown")
                            except Exception:
                                pass

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
            
        time.sleep(60)


# ==========================================
# 5. TELEGRAM HANDLERS
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    welcome_text = (
        "👋 **CHÀO MỪNG BẠN ĐẾN VỚI QUANT TRADING SYSTEM!**\n\n"
        "📌 *Bấm trực tiếp vào các ô bên dưới để copy cú pháp:* \n\n"
        "1️⃣ **Phân tích chi tiết mã cổ phiếu:**\n"
        "`VHM`  *(hoặc nhập mã bất kỳ như HPG, SSI, VPB)*\n\n"
        "2️⃣ **Lưu giá mua, cắt lỗ & khối lượng:**\n"
        "`/setbuy VHM 63.0 62.0 1500`  *(Mã, Giá mua, SL, Khối lượng)*\n\n"
        "3️⃣ **Xem danh mục đầu tư & Phương án xử lý Lợi nhuận / SL:**\n"
        "`/portfolio` \n\n"
        "4️⃣ **Quét dòng tiền bùng nổ toàn thị trường:**\n"
        "`/scan` \n\n"
        "5️⃣ **Backtest hiệu suất chiến lược 200 phiên:**\n"
        "`/backtest VHM` \n\n"
        "6️⃣ **Xóa mã khỏi danh mục theo dõi:**\n"
        "`/reset VHM`  *(hoặc `/reset ALL` để xóa sạch danh mục)*\n\n"
        "------------------------------------\n"
        "💡 *Lưu ý:* Hệ thống báo cáo kỹ thuật đã được tối ưu hóa hành động chi tiết ở từng mục."
    )
    
    try:
        await update.message.reply_text(welcome_text, parse_mode="Markdown")
    except Exception as e:
        print(f"Lỗi gửi /start: {e}")


async def setbuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 3:
        try:
            await update.message.reply_text(
                "Sai cú pháp!\n\n"
                "Vui lòng nhập đúng định dạng:\n"
                "/setbuy <MÃ> <giá_mua> <giá_cắt_lỗ> [khối_lượng]\n"
                "Ví dụ: /setbuy VHM 63.0 62.0 1500"
            )
        except Exception as e:
            print(f"Lỗi gửi tin nhắn: {e}")
        return

    symbol = args[0].upper().strip()
    try:
        entry_price = float(args[1])
        sl_price = float(args[2])
        qty = int(args[3]) if len(args) >= 4 else 1000
    except ValueError:
        await update.message.reply_text("Giá mua, giá cắt lỗ và khối lượng phải là số hợp lệ!")
        return

    if chat_id not in USER_PORTFOLIOS:
        USER_PORTFOLIOS[chat_id] = {}

    USER_PORTFOLIOS[chat_id][symbol] = {
        "entry": entry_price,
        "sl": sl_price,
        "qty": qty
    }

    await update.message.reply_text(
        f"Đã lưu thành công vào danh mục:\n"
        f"- Mã: {symbol}\n"
        f"- Khối lượng: {qty} cổ phiếu\n"
        f"- Giá mua: {entry_price}\n"
        f"- Cắt lỗ (SL): {sl_price}\n\n"
        f"Bot sẽ tự động giám sát và cảnh báo ngay lập tức nếu giá thị trường thủng mức SL này!"
    )


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in USER_PORTFOLIOS or not USER_PORTFOLIOS[chat_id]:
        await update.message.reply_text(
            "📂 Danh mục của bạn đang trống.\n"
            "👉 Dùng lệnh `/setbuy VHM 63.0 62.0 1500` để thêm mã vào danh mục theo dõi.",
            parse_mode="Markdown"
        )
        return

    text = "📂 **DANH MỤC ĐẦU TƯ CÁ NHÂN & CHIẾN LƯỢC XỬ LÝ (MỤC 1 & 2)**\n\n"
    for symbol, pos in USER_PORTFOLIOS[chat_id].items():
        df, err = fetch_stock_data(symbol, resolution="D", count=5)
        if not err and not df.empty:
            current_price = df.iloc[-1]["Close"]
            qty = pos.get("qty", 1000)
            entry = pos["entry"]
            sl = pos["sl"]
            
            pnl_pct = ((current_price - entry) / entry) * 100
            pnl_vnd = (current_price - entry) * qty
            status = "🟢 LÃI" if pnl_pct >= 0 else "🔴 LỖ"
            
            text += f"🔹 **Mã Cổ Phiếu: {symbol}** ({qty
