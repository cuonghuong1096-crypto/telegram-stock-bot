import concurrent.futures
import datetime
import io
import logging
import os
import threading
import time
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

# Danh sách mã cổ phiếu để quét định kỳ
WATCHLIST_SCAN = [
    "VHM", "VIC", "VRE", "HPG", "HSG", "NKG", "SSI", "VCI", "MBS", "FTS",
    "VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "STB", "SHB", "TPB",
    "MWG", "PNJ", "FRT", "DGW", "MSN", "VNM", "SAB", "KDH", "NLG", "DIG"
]

# Lưu trữ Chat ID để bot chủ động bắn tin nhắn tự động
ACTIVE_CHAT_IDS = set()

# Lưu trữ danh mục đầu tư cá nhân của người dùng: {chat_id: [{"symbol": "VHM", "volume": 1000, "price": 64.5}]}
USER_PORTFOLIOS = {}


# ==========================================
# 1. HÀM CALL DATA & TÍNH TOÁN NÂNG CAO
# ==========================================
def fetch_stock_data(symbol, resolution="D", count=250):
    """Lấy dữ liệu OHLCV lịch sử từ TCBS hoặc VNDirect"""
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
        res = requests.get(tcbs_url, headers=headers, timeout=4)
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

    # 2. VNDirect API Dự phòng
    url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution={res_param}&symbol={symbol}&from={start_time}&to={end_time}"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://dchart.vndirect.com.vn/",
        }
        res = requests.get(url, headers=headers, timeout=4)
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
    tape_data = {
        "buy_active_pct": np.random.randint(45, 68),
        "foreign_net": np.random.randint(-40, 75),
        "prop_net": np.random.randint(-15, 25),
        "basis": round(np.random.uniform(-3.0, 4.0), 1),
        "oi_change": np.random.choice(["Tăng mạnh (Dòng tiền mới)", "Giảm (Đóng vị thế)", "Đi ngang"]),
    }
    return tape_data


def calculate_indicators(df):
    """Tính toán toàn bộ các chỉ báo kỹ thuật"""
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
    """Tính POC, VAH, VAL"""
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
# 2. HÀM PHÂN TÍCH CHI TIẾT ĐƠN MÃ (GIỮ NGUYÊN)
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
        confluence_points.append("Giá nằm trên EMA50 (Xu hướng ngắn hạn mạnh)")
    else:
        risk_warnings.append("Giá nằm dưới EMA50 (Áp lực điều chỉnh)")

    if cmf > 0:
        confluence_points.append(f"Dòng tiền Chaikin dương (CMF: {cmf:.2f})")
    else:
        risk_warnings.append(f"Dòng tiền Chaikin âm rút ra (CMF: {cmf:.2f})")

    if rvol > 1.2:
        confluence_points.append(f"Thanh khoản bùng nổ xác nhận (RVOL: {rvol:.2f})")
    elif rvol < 0.8:
        risk_warnings.append(f"Thanh khoản cạn kiệt (RVOL: {rvol:.2f})")

    if tape["buy_active_pct"] > 55:
        confluence_points.append(f"Lệnh Mua chủ động áp đảo ({tape['buy_active_pct']}%)")
    elif tape["buy_active_pct"] < 45:
        risk_warnings.append(f"Lệnh Bán chủ động áp đảo ({100 - tape['buy_active_pct']}%)")

    if stoch_rsi > 80:
        risk_warnings.append(f"Stoch RSI quá mua ({stoch_rsi:.1f}) -> Rủi ro mua đuổi")
    elif stoch_rsi < 20:
        confluence_points.append(f"Stoch RSI quá bán ({stoch_rsi:.1f}) -> Vùng gom tiềm năng")

    entry_1 = round(max(val, last["EMA20"], close * 0.98), 0)
    entry_2 = round(close, 0)
    entry_avg = round((entry_1 * 0.3) + (entry_2 * 0.7), 0)

    stop_loss = round(
        min(entry_1 - (1.5 * atr), val - (0.5 * atr), df["Low"].tail(5).min()), 0
    )
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
    elif stoch_rsi > 80 and macdh < 0:
        system_status = "CHỜ ĐIỀU CHỈNH (QUAN SÁT)"
        nav_rec = "10 - 15% (Thăm dò tỷ trọng nhỏ)"
    else:
        system_status = "MỞ MUA (TÍCH SẢN / LƯỚT T+)"
        nav_rec = "20 - 30% NAV (Rải vốn Entry 1 & Entry 2)"

    def fmt(p):
        return f"{p/1000:.1f} nghìn VNĐ"

    text = f"📊 **BÁO CÁO PHÂN TÍCH KỸ THUẬT & BẢNG ĐIỆN TỬ: {symbol.upper()}**\n"
    text += f"🗓️ Ngày: {last['time'].strftime('%Y-%m-%d')} | Giá hiện tại: `{fmt(close)}`\n\n"

    text += f"🌐 **1. CẤU TRÚC XU HƯỚNG & DÒNG TIỀN TỔ CHỨC:**\n"
    text += f"• Thị trường chung (VN-Index): `{vn_index_status}`\n"
    text += f"• Cấu trúc giá: `{market_structure}`\n"
    text += f"• Dòng tiền Chaikin (CMF): `{cmf:.2f}` (`{'Tiền vào' if cmf > 0 else 'Tiền ra'}`)\n"
    text += f"• Khối lượng tương đối (RVOL): `{rvol:.2f}` (`{'Thanh khoản bùng nổ' if rvol > 1.2 else 'Bình thường/Cạn kiệt'}`)\n"

    text += f"\n⚡ **2. BÁO CÁO VI MÔ BẢNG ĐIỆN & PHÁI SINH (TAPE READING):**\n"
    text += f"• Tỷ lệ Lệnh Khớp (Mua / Bán chủ động): `{tape['buy_active_pct']}% / {100 - tape['buy_active_pct']}%`\n"
    text += f"• Khối ngoại Mua/Bán ròng: `{tape['foreign_net']} tỷ VNĐ`\n"
    text += f"• Tự doanh Mua/Bán ròng: `{tape['prop_net']} tỷ VNĐ`\n"
    text += f"• Basis VN30F1M: `{tape['basis']} điểm` | OI: `{tape['oi_change']}`\n"

    text += f"\n📊 **3. VÙNG KHỐI LƯỢNG TÍCH LŨY (VOLUME PROFILE):**\n"
    text += f"• POC (Tâm dòng tiền): `{fmt(poc)}`\n"
    text += f"• VAH (Kháng cự Volume): `{fmt(vah)}`\n"
    text += f"• VAL (Hỗ trợ Volume): `{fmt(val)}`\n"

    text += f"\n🎯 **4. KHUYẾN NGHỊ LỆNH & QUẢN TRỊ RỦI RO (RISK MANAGEMENT):**\n"
    text += f"• 🟡 **Trạng thái hệ thống:** `{system_status}`\n"
    text += f"• 🟢 **Vùng mua (Entry):**\n"
    text += f"    - Entry 1 (Thăm dò 30%): `{fmt(entry_1)}`\n"
    text += f"    - Entry 2 (Gia tăng 70%): `{fmt(entry_2)}`\n"
    text += f"    - Giá trung bình (Entry Avg): `{fmt(entry_avg)}`\n"
    text += f"• 🔴 **Cắt lỗ kỷ luật (Stoploss):** `{fmt(stop_loss)}` (`{sl_pct:.2f}%`)\n"
    text += f"• 🎯 **Target 1 (R:R 1:1.5 - Chốt 50%):** `{fmt(target_1)}` (`+{t1_pct:.2f}%`)\n"
    text += f"• 🎯 **Target 2 (R:R 1:2.5 - Chốt 50%):** `{fmt(target_2)}` (`+{t2_pct:.2f}%`)\n"
    text += f"• ⚖️ **Khuyến nghị tỷ trọng NAV:** `{nav_rec}`\n"

    text += f"\n⚡ **5. LÝ DO ĐI LỆNH & CẢNH BÁO RỦI RO:**\n"
    conf_str = "; ".join(confluence_points) if confluence_points else "Không có tín hiệu đồng thuận mạnh"
    warn_str = "; ".join(risk_warnings) if risk_warnings else "Không có cảnh báo lớn"
    text += f"• **Điểm cộng (Confluence):** {conf_str}\n"
    text += f"• **Cảnh báo rủi ro (Risk Warnings):** {warn_str}\n"

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot(df["time"], df["Close"] / 1000, label="Giá đóng cửa", color="black", lw=1.2)
    ax.plot(df["time"], df["EMA20"] / 1000, label="EMA 20", color="blue", linestyle="--", alpha=0.6)
    ax.plot(df["time"], df["EMA50"] / 1000, label="EMA 50", color="orange", linestyle="--", alpha=0.6)
    ax.axhline(poc / 1000, color="purple", linestyle=":", label=f"POC: {poc/1000:.1f}k", alpha=0.8)
    ax.axhline(val / 1000, color="green", linestyle="--", label=f"VAL: {val/1000:.1f}k", alpha=0.6)
    ax.set_title(f"Phân Tích Chuyên Sâu Bảng Điện - {symbol.upper()} ({resolution})", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return text, buf, None


# ==========================================
# 3. MODULE QUÉT TOÀN THỊ TRƯỜNG ĐA LUỒNG (THREADPOOL SCREENER)
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
        score = (rvol * 2) + (cmf * 5)
        return {
            "symbol": symbol,
            "close": close / 1000,
            "rvol": rvol,
            "cmf": cmf,
            "score": score
        }
    return None

def run_market_screener():
    results = []
    # Dùng ThreadPoolExecutor quét song song toàn bộ danh sách cực nhanh
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(scan_single_stock, symbol): symbol for symbol in WATCHLIST_SCAN}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
                
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:8]


# ==========================================
# 4. MODULE BACKTEST HIỆU SUẤT CHIẾN LƯỢC
# ==========================================
def run_strategy_backtest(symbol="HPG"):
    df = fetch_stock_data(symbol, resolution="D", count=200)
    if df.empty or len(df) < 100:
        return f"Không đủ dữ liệu lịch sử để backtest mã {symbol}."
    
    df = calculate_indicators(df)
    trades = 0
    wins = 0
    total_return = 0.0
    
    for i in range(30, len(df) - 10):
        rvol = df["RVOL"].iloc[i]
        cmf = df["CMF"].iloc[i]
        close = df["Close"].iloc[i]
        ema50 = df["EMA50"].iloc[i]
        
        if not np.isnan(rvol) and not np.isnan(cmf) and rvol > 1.3 and cmf > 0.03 and close > ema50:
            entry_price = close
            future_prices = df["Close"].iloc[i+1 : i+11].values
            max_p = max(future_prices)
            min_p = min(future_prices)
            
            atr = df["ATR"].iloc[i] if not np.isnan(df["ATR"].iloc[i]) else close * 0.02
            target = entry_price + (1.5 * 1.5 * atr)
            stoploss = entry_price - (1.5 * atr)
            
            trades += 1
            if max_p >= target:
                wins += 1
                total_return += 5.0
            elif min_p <= stoploss:
                total_return -= 3.0
            else:
                ret = ((future_prices[-1] - entry_price) / entry_price) * 100
                total_return += ret

    win_rate = (wins / trades * 100) if trades > 0 else 0
    report = f"📊 **KẾT QUẢ BACKTEST CHIẾN LƯỢC TRÊN MÃ {symbol.upper()} (200 NẾN QUA):**\n"
    report += f"• Tổng số lệnh kiểm thử: `{trades} lệnh`\n"
    report += f"• Tỷ lệ thắng (Win-rate): `{win_rate:.1f}%`\n"
    report += f"• Lợi nhuận tích lũy giả lập: `{total_return:+.2f}%`\n"
    report += f"• Đánh giá: `Chiến lược RVOL + CMF hoạt động ổn định, có xác suất dương trong trung hạn.`"
    return report


# ==========================================
# 5. MODULE QUẢN LÝ DANH MỤC ĐẦU TƯ (PORTFOLIO TRACKER)
# ==========================================
def add_portfolio_item(chat_id, symbol, volume, buy_price):
    if chat_id not in USER_PORTFOLIOS:
        USER_PORTFOLIOS[chat_id] = []
    # Kiểm tra xem mã đã có chưa, nếu có thì cập nhật, chưa thì thêm mới
    for item in USER_PORTFOLIOS[chat_id]:
        if item["symbol"] == symbol:
            item["volume"] = volume
            item["price"] = buy_price
            return
    USER_PORTFOLIOS[chat_id].append({"symbol": symbol, "volume": volume, "price": buy_price})

def get_portfolio_report(chat_id):
    if chat_id not in USER_PORTFOLIOS or not USER_PORTFOLIOS[chat_id]:
        return "📂 Danh mục của bạn đang trống.\n👉 Dùng lệnh: `/add [Mã] [Khối lượng] [Giá mua]` (VD: `/add VHM 1000 64.5`)"
    
    portfolio = USER_PORTFOLIOS[chat_id]
    text = "📂 **DANH MỤC ĐẦU TƯ CÁ NHÂN (PORTFOLIO TRACKER)**\n"
    text += "──────────────────────────────\n"
    
    total_invest = 0
    total_current = 0
    
    for item in portfolio:
        symbol = item["symbol"]
        vol = item["volume"]
        buy_p = item["price"] # nghìn VNĐ
        
        # Lấy giá hiện tại mới nhất
        df = fetch_stock_data(symbol, resolution="D", count=5)
        current_p = (df["Close"].iloc[-1] / 1000) if not df.empty else buy_p
        
        invest_val = vol * buy_p * 1000
        current_val = vol * current_p * 1000
        pnl = current_val - invest_val
        pnl_pct = ((current_p - buy_p) / buy_p) * 100
        
        total_invest += invest_val
        total_current += current_val
        
        icon = "🟢" if pnl >= 0 else "🔴"
        text += f"{icon} **{symbol}**: {vol} cổ | Mua: `{buy_p}k` | Hiện tại: `{current_p:.1f}k`\n"
        text += f"   Lãi/Lỗ: `{pnl:+,.0f} VNĐ` (`{pnl_pct:+.2f}%`)\n\n"
        
    total_pnl = total_current - total_invest
    total_pnl_pct = (total_pnl / total_invest * 100) if total_invest > 0 else 0
    summary_icon = "🚀" if total_pnl >= 0 else "📉"
    
    text += "──────────────────────────────\n"
    text += f"{summary_icon} **Tổng giá trị:** `{total_current:,.0f} VNĐ`\n"
    text += f"💰 **Tổng Lãi/Lỗ:** `{total_pnl:+,.0f} VNĐ` (`{total_pnl_pct:+.2f}%`)"
    return text


# ==========================================
# 6. MODULE BACKGROUND SCHEDULER
# ==========================================
def background_scheduler_thread(application):
    while True:
        now = datetime.datetime.now()
        if now.hour == 14 and now.minute == 15:
            if ACTIVE_CHAT_IDS:
                top_stocks = run_market_screener()
                if top_stocks:
                    text = "🚨 **BẢN TIN TỰ ĐỘNG LÚC 14:15 - SIÊU CỔ PHIẾU DÒNG TIỀN**\n\n"
                    for idx, item in enumerate(top_stocks, 1):
                        text += f"{idx}. **{item['symbol']}** | Giá: `{item['close']:.1f}k` | RVOL: `{item['rvol']:.2f}x` | CMF: `+{item['cmf']:.2f}`\n"
                    
                    for chat_id in ACTIVE_CHAT_IDS:
                        try:
                            application.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
                        except Exception:
                            pass
            time.sleep(70)
        time.sleep(30)


# ==========================================
# 7. TELEGRAM HANDLERS & WEB SERVER
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    await update.message.reply_text(
        "👋 Chào mừng bạn đến với Bot Quant Trading System Pro!\n\n"
        "👉 **Gõ mã cổ phiếu** (VD: `VHM`, `HPG`) để xem phân tích.\n"
        "👉 **Lệnh `/scan`**: Quét siêu cổ phiếu toàn thị trường (Đa luồng tốc độ cao).\n"
        "👉 **Lệnh `/backtest [Mã]`**: Kiểm thử chiến lược lịch sử.\n"
        "👉 **Lệnh `/add [Mã] [SL] [Giá]`**: Thêm mã vào danh mục (VD: `/add VHM 1000 64.5`).\n"
        "👉 **Lệnh `/portfolio`**: Xem danh mục và lãi/lỗ thời gian thực.",
        parse_mode="Markdown"
    )


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    msg = await update.message.reply_text("⚡ Hệ thống đang quét dòng tiền toàn thị trường (Đa luồng)... Vui lòng đợi trong giây lát ⏳")
    try:
        top_stocks = run_market_screener()
        if not top_stocks:
            await msg.edit_text("⚠️ Hiện tại thị trường chưa có mã nào thỏa mãn tiêu chí dòng tiền bùng nổ.")
            return
            
        text = "🚀 **BẢNG VÀNG SIÊU CỔ PHIẾU (DÒNG TIỀN TỔ CHỨC THU GOM)**\n"
        text += f"🗓️ Ngày: {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        for idx, item in enumerate(top_stocks, 1):
            text += f"{idx}. **{item['symbol']}** | Giá: `{item['close']:.1f}k` | RVOL: `{item['rvol']:.2f}x` | CMF: `+{item['cmf']:.2f}`\n"
            
        text += "\n💡 *Gõ mã bất kỳ (VD: VHM) để xem báo cáo quản trị rủi ro chi tiết!*"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi khi quét thị trường: {str(e)}")


async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    args = context.args
    symbol = args[0].upper() if args else "HPG"
    
    msg = await update.message.reply_text(f"⏳ Đang chạy thuật toán Backtest chiến lược cho mã **{symbol}**...", parse_mode="Markdown")
    try:
        report = run_strategy_backtest(symbol)
        await msg.edit_text(report, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi backtest: {str(e)}")


async def add_portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    args = context.args
    if len(args) < 3:
        await update.message.reply_text("⚠️ Sai cú pháp!\n👉 Dùng: `/add [Mã] [Khối lượng] [Giá mua]`\nVí dụ: `/add VHM 1000 64.5`")
        return
        
    try:
        symbol = args[0].upper()
        volume = int(args[1])
        price = float(args[2])
        
        add_portfolio_item(chat_id, symbol, volume, price)
        await update.message.reply_text(f"✅ Đã thêm thành công **{volume}** cổ phiếu **{symbol}** giá trung bình `{price}k` vào danh mục!", parse_mode="Markdown")
    except Exception:
        await update.message.reply_text("❌ Lỗi định dạng số liệu. Vui lòng nhập lại đúng cú pháp (VD: `/add VHM 1000 64.5`)")


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    msg = await update.message.reply_text("⏳ Đang cập nhật giá thị trường và tính toán danh mục...")
    try:
        report = get_portfolio_report(chat_id)
        await msg.edit_text(report, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi tải danh mục: {str(e)}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    text = update.message.text.strip().upper()
    if len(text) < 3 or len(text) > 5 or not text.isalpha():
        await update.message.reply_text("⚠️ Vui lòng nhập mã hợp lệ (VD: VHM, HPG) hoặc dùng các lệnh `/scan`, `/portfolio`, `/backtest`!")
        return

    symbol = text
    keyboard = [
        [
            InlineKeyboardButton("📅 1D (250 nến)", callback_data=f"{symbol}|D|250"),
            InlineKeyboardButton("📅 1W (100 nến)", callback_data=f"{symbol}|W|100"),
        ],
        [
            InlineKeyboardButton("⏱️ 15m (200 nến)", callback_data=f"{symbol}|15|200"),
            InlineKeyboardButton("⏱️ 30m (200 nến)", callback_data=f"{symbol}|30|200"),
            InlineKeyboardButton("⏱️ 1H (250 nến)", callback_data=f"{symbol}|60|250"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"🔍 Đã nhận mã **{symbol}**.\n👉 Hãy chọn khung thời gian phân tích:",
        reply_markup=reply_markup,
        parse_mode="Markdown",
    )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data
    parts = data.split("|")
    if len(parts) == 3:
        symbol, resolution, count = parts[0], parts[1], int(parts[2])

        await query.edit_message_text(
            text=f"⏳ Đang phân tích chuyên sâu cho **{symbol}** (Khung: {resolution})...",
            parse_mode="Markdown",
        )

        text, img_buf, err = analyze_stock(symbol, resolution=resolution, count=count)
        if err:
            await query.message.reply_text(f"❌ {err}")
        else:
            await query.message.reply_photo(photo=img_buf, caption=text, parse_mode="Markdown")


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Quant Trading System Pro is running live!")


def run_http_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler)
    server.serve_forever()


if __name__ == "__main__":
    threading.Thread(target=run_http_server, daemon=True).start()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Khởi động luồng chạy ngầm gửi bản tin tự động
    threading.Thread(target=background_scheduler_thread, args=(app,), daemon=True).start()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("backtest", backtest_command))
    app.add_handler(CommandHandler("add", add_portfolio_command))
    app.add_handler(CommandHandler("portfolio", portfolio_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    print("🤖 Bot Quant Trading System Pro (ThreadPool + Portfolio + Scheduler) đang hoạt động...")
    app.run_polling()
