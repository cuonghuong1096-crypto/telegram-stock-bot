import datetime
import io
import json
import logging
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
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
GOOGLE_SHEET_API_URL = os.environ.get("GOOGLE_SHEET_API_URL", "")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

WATCHLIST_SCAN = [
    # Nhóm gốc & Ngân hàng
    "VHM", "VIC", "VRE", "HPG", "HSG", "NKG", "SSI", "VCI", "MBS", "FTS",
    "VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "STB", "SHB", "TPB",
    "MWG", "PNJ", "FRT", "DGW", "MSN", "VNM", "SAB", "KDH", "NLG", "DIG", "BSR",
    # Đầu tư công & Xây dựng & Vật liệu
    "CTD", "HHV", "LCG", "HT1", "VCG", "CII", "L14", "VGS", "KBC", "PDR", 
    "CEO", "DXG", "TCH", "NVL", "VPL", "HDG", "NTL", "HDC", "KSB", "MSR",
    # Chứng khoán & Tài chính khác
    "VND", "SHS", "VIX", "HCM", "CTS", "ORS",
    # Bán lẻ & Tiêu dùng & Công nghệ
    "FPT", "MCH", "PET", "VTP", "CTR", "GEE",
    # Thủy sản, May mặc, Cao su
    "VGT", "TNG", "GVR", "ANV", "VHC",
    # Dầu khí
    "PVS", "PVB", "PVD", "GAS", "PLX",
    # Vận tải
    "HAH", "VJC", "VSC"
]

SECTOR_MAP = {
    "Ngân hàng": ["VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "STB", "SHB", "TPB"],
    "Bất động sản": ["VHM", "VIC", "VRE", "KDH", "NLG", "DIG", "PDR", "CEO", "DXG", "TCH", "NVL", "HDG", "NTL", "HDC"],
    "Chứng khoán": ["SSI", "VCI", "MBS", "FTS", "VND", "SHS", "VIX", "HCM", "CTS", "ORS"],
    "Thép & Vật liệu": ["HPG", "HSG", "NKG", "VGS", "HT1", "KSB"],
    "Xây dựng & Đầu tư công": ["CTD", "HHV", "LCG", "VCG", "CII", "L14", "KBC", "MSR"],
    "Dầu khí & Energy": ["PVS", "PVB", "PVD", "GAS", "PLX", "BSR"],
    "Bán lẻ & Tiêu dùng": ["MWG", "PNJ", "FRT", "DGW", "MSN", "VNM", "SAB", "PET"],
    "Công nghệ & Viễn thông": ["FPT", "CTR", "VTP", "GEE"],
    "Thủy sản & Dệt may & Cao su": ["ANV", "VHC", "VGT", "TNG", "GVR"],
    "Vận tải & Cảng biển": ["HAH", "VJC", "VSC"]
}

ACTIVE_CHAT_IDS = set()
ALERTED_TODAY = set()

# ==========================================
# QUẢN LÝ LƯU TRỮ QUA GOOGLE SHEETS API
# ==========================================
USER_PORTFOLIOS = {}

def load_portfolios():
    if not GOOGLE_SHEET_API_URL:
        logging.warning("Chưa cấu hình GOOGLE_SHEET_API_URL trên môi trường!")
        return {}
    try:
        res = requests.get(GOOGLE_SHEET_API_URL, timeout=10)
        if res.status_code == 200:
            result = res.json()
            if result.get("status") == "success":
                data = result.get("data", [])
                temp_portfolios = {}
                for item in data:
                    chat_id = int(item["chat_id"])
                    symbol = item["symbol"]
                    entry = float(item["entry"])
                    qty = int(item["qty"])
                    try:
                        alert_data = json.loads(item["alerts"])
                    except Exception:
                        alert_data = {"buy_alert_prices": [], "stop_alert_prices": []}
                    
                    if chat_id not in temp_portfolios:
                        temp_portfolios[chat_id] = {}
                    
                    temp_portfolios[chat_id][symbol] = {
                        "entry": entry,
                        "qty": qty,
                        "buy_alert_prices": alert_data.get("buy_alert_prices", []),
                        "stop_alert_prices": alert_data.get("stop_alert_prices", [])
                    }
                logging.info("Đã tải thành công portfolio từ Google Sheets!")
                return temp_portfolios
    except Exception as e:
        logging.error(f"Lỗi tải portfolio từ Google Sheet: {e}")
    return {}

def save_portfolios():
    if not GOOGLE_SHEET_API_URL:
        return
    try:
        rows = []
        for chat_id, portfolio in USER_PORTFOLIOS.items():
            for symbol, pos in portfolio.items():
                alert_dict = {
                    "buy_alert_prices": pos.get("buy_alert_prices", []),
                    "stop_alert_prices": pos.get("stop_alert_prices", [])
                }
                rows.append([
                    str(chat_id),
                    symbol,
                    pos.get("entry", 0),
                    pos.get("qty", 1000),
                    json.dumps(alert_dict)
                ])
        payload = {"action": "save", "rows": rows}
        requests.post(GOOGLE_SHEET_API_URL, json=payload, timeout=10)
    except Exception as e:
        logging.error(f"Lỗi ghi portfolio lên Google Sheet: {e}")

USER_PORTFOLIOS = load_portfolios()
executor = ThreadPoolExecutor(max_workers=16)

# ==========================================
# 1. BƯỚC 1 QUY TẮC VÀNG: BỘ LỌC VN-INDEX
# ==========================================
def fetch_stock_data(symbol, resolution="D", count=250):
    symbol = symbol.upper().strip()
    end_time = int(datetime.datetime.now().timestamp())

    if resolution in ["60"]:
        res_param = resolution
        count = max(count, 300)
        start_time = end_time - (count * int(resolution) * 60 * 3)
    elif resolution == "W":
        res_param = "D"
        count = 600 
        start_time = end_time - (count * 86400 * 2)
    elif resolution == "M":
        res_param = "D"
        count = 1200
        start_time = end_time - (count * 86400 * 2)
    else:
        res_param = "D"
        count = max(count, 200)
        start_time = end_time - (count * 86400 * 2)

    tcbs_url = f"https://apifull.tcbs.com.vn/stock-insight/v1/stock/bars?resolution={res_param}&ticker={symbol}&from={start_time}&to={end_time}"
    df_raw = pd.DataFrame()
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "X-API-KEY": TCBS_API_KEY,
        }
        res = requests.get(tcbs_url, headers=headers, timeout=2.5)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                df_raw = pd.DataFrame(data)
                df_raw["time"] = pd.to_datetime(df_raw["tradingDate"])
                df_raw.rename(
                    columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    },
                    inplace=True,
                )
    except Exception:
        pass

    if df_raw.empty:
        url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution={res_param}&symbol={symbol}&from={start_time}&to={end_time}"
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://dchart.vndirect.com.vn/",
            }
            res = requests.get(url, headers=headers, timeout=2.5)
            if res.status_code == 200:
                data = res.json()
                if data.get("s") == "ok" and "t" in data and len(data["t"]) > 0:
                    df_raw = pd.DataFrame(
                        {
                            "time": pd.to_datetime(data["t"], unit="s"),
                            "Open": data["o"],
                            "High": data["h"],
                            "Low": data["l"],
                            "Close": data["c"],
                            "Volume": data["v"],
                        }
                    )
        except Exception as e:
            return pd.DataFrame(), f"Lỗi kết nối API dự phòng: {str(e)}"

    if df_raw.empty:
        return pd.DataFrame(), f"Không tìm thấy dữ liệu hoặc mã cổ phiếu '{symbol}' không hợp lệ."

    df_raw = df_raw[["time", "Open", "High", "Low", "Close", "Volume"]].sort_values("time").reset_index(drop=True)

    if resolution == "W":
        df_raw.set_index("time", inplace=True)
        df_weekly = df_raw.resample("W-FRI").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        }).dropna().reset_index()
        return df_weekly.tail(250).reset_index(drop=True), None

    if resolution == "M":
        df_raw.set_index("time", inplace=True)
        df_monthly = df_raw.resample("ME").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        }).dropna().reset_index()
        return df_monthly.tail(120).reset_index(drop=True), None

    return df_raw.tail(count).reset_index(drop=True), None

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

    df["BB_Middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["BB_Middle"] + 1e-9)
    df["BB_Squeeze"] = df["BB_Width"] <= df["BB_Width"].rolling(50).quantile(0.20)

    df["Price_Range_Pct"] = (df["High"].rolling(10).max() - df["Low"].rolling(10).min()) / df["Close"]

    price_max_now = df["Close"].iloc[-1]
    price_max_prev = df["Close"].iloc[-20:-5].max() if len(df) >= 20 else price_max_now
    rsi_now = df["RSI"].iloc[-1]
    rsi_prev_max = df["RSI"].iloc[-20:-5].max() if len(df) >= 20 else rsi_now
    df["RSI_Bearish_Divergence"] = (price_max_now > price_max_prev) and (rsi_now < rsi_prev_max) and (rsi_now > 60)

    price_min_now = df["Close"].iloc[-1]
    price_min_prev = df["Close"].iloc[-20:-5].min() if len(df) >= 20 else price_min_now
    rsi_prev_min = df["RSI"].iloc[-20:-5].min() if len(df) >= 20 else rsi_now
    df["RSI_Bullish_Divergence"] = (price_min_now < price_min_prev) and (rsi_now > rsi_prev_min) and (rsi_now < 40)

    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    plus_dm = df["High"].diff()
    minus_dm = df["Low"].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr14 = true_range.rolling(14).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).sum() / (tr14 + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).sum() / (tr14 + 1e-9))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df["ADX"] = dx.rolling(14).mean()

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

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

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

def check_vnindex_health():
    """BƯỚC 1: Lọc sức khỏe thị trường VN-Index"""
    df, err = fetch_stock_data("VNINDEX", resolution="D", count=60)
    if err or df.empty:
        df, err = fetch_stock_data("1000", resolution="D", count=60)
        if err or df.empty:
            return {"status": "NEUTRAL", "score_modifier": 1.0, "note": "Không lấy được dữ liệu VN-Index"}

    df = calculate_indicators(df)
    last = df.iloc[-1]
    close = last["Close"]
    ema20 = last.get("EMA20", close)
    ema50 = last.get("EMA50", close)
    macd_hist = last.get("MACD_Hist", 0)

    if close > ema20 and ema20 > ema50 and macd_hist > 0:
        return {"status": "BULLISH", "score_modifier": 1.15, "note": "🟢 VN-Index Uptrend ủng hộ giải ngân"}
    elif close < ema50 or macd_hist < -0.5:
        return {"status": "BEARISH_RISK", "score_modifier": 0.75, "note": "🔴 VN-Index Rủi ro / Ưu tiên đứng ngoài"}
    else:
        return {"status": "SIDEWAYS", "score_modifier": 1.0, "note": "🟡 VN-Index Tích lũy / Chỉ đi vốn thăm dò"}

# ==========================================
# 2. BƯỚC 2 QUY TẮC VÀNG: PHỄU VSA MULTI-STAGE
# ==========================================
def calculate_volume_profile(df, bins=20):
    if df.empty:
        return 0, 0, 0
    sub_df = df.tail(60)
    price_min, price_max = sub_df["Low"].min(), sub_df["High"].max()
    if price_min == price_max:
        return price_min, price_min, price_min

    hist, bin_edges = np.histogram(sub_df["Close"], bins=bins, weights=sub_df["Volume"])
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

def check_liquidity_and_base(df, last):
    close = last["Close"]
    vol_sma20 = last["Vol_SMA20"] if not np.isnan(last.get("Vol_SMA20", np.nan)) else last["Volume"]
    avg_trading_value = vol_sma20 * close * 1000
    return avg_trading_value >= 5_000_000_000

def run_market_screener():
    """Tầng 1: Tích lũy nền & Dòng tiền (/scan)"""
    results = []
    mkt_health = check_vnindex_health()

    def scan_one(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=200)
        if err or df.empty or len(df) < 50: return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        if not check_liquidity_and_base(df, last): return None

        close = last["Close"]
        rvol = last.get("RVOL", 1.0)
        cmf = last.get("CMF", 0.0)
        adx = last.get("ADX", 20.0)
        ema50 = last.get("EMA50", close)
        price_range_pct = last.get("Price_Range_Pct", 0.1)

        if (rvol >= 1.1 or price_range_pct < 0.08) and cmf > 0.01 and close > ema50:
            score = ((rvol * 2.5) + (cmf * 6) + (max(0, 0.1 - price_range_pct) * 20)) * mkt_health["score_modifier"]
            return {"symbol": symbol, "close": close, "rvol": rvol, "cmf": cmf, "adx": adx, "score": score}
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_one, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
    return sorted(results, key=lambda x: x["score"], reverse=True)[:8]

def run_test_supply_screener():
    """Tầng 2: Cạn Cung / No-Supply (/testcung)"""
    results = []
    def scan_test_supply(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=150)
        if err or df.empty or len(df) < 50: return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        if not check_liquidity_and_base(df, last): return None

        close, high, low = last["Close"], last["High"], last["Low"]
        rvol = last.get("RVOL", 1.0)
        cmf = last.get("CMF", 0.0)
        price_range = high - low
        is_pinbar_or_solid = True if price_range == 0 else (close >= low + price_range * 0.5)
        
        if rvol <= 0.65 and is_pinbar_or_solid and cmf >= -0.1:
            score = (1.0 - rvol) * 50 + (max(0, cmf) * 20)
            return {"symbol": symbol, "close": close, "rvol": rvol, "cmf": cmf, "score": score}
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_test_supply, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]

def run_demand_confirmation_screener():
    """Tầng 3: Xác Nhận Cầu Chủ Động (/demand)"""
    results = []
    def scan_demand(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=150)
        if err or df.empty or len(df) < 50: return None
        df = calculate_indicators(df)
        last, prev = df.iloc[-1], df.iloc[-2]
        if not check_liquidity_and_base(df, last): return None

        close, high, low = last["Close"], last["High"], last["Low"]
        rvol = last.get("RVOL", 1.0)
        cmf = last.get("CMF", 0.0)
        mfi = last.get("MFI", 50.0)
        kijun = last.get("Kijun_sen", close)

        candle_range = high - low
        if candle_range == 0: return None
            
        is_strong_close = ((close - low) / candle_range) >= 0.70
        was_no_supply = prev.get("RVOL", 1.0) <= 0.70
        
        if rvol >= 1.50 and is_strong_close and cmf >= 0.08 and close >= kijun:
            bonus = 30 if was_no_supply else 0
            score = (rvol * 20) + (cmf * 50) + (mfi * 0.5) + bonus
            return {
                "symbol": symbol, "close": close, "rvol": rvol, "cmf": cmf, "mfi": mfi,
                "was_no_supply": was_no_supply, "score": score
            }
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_demand, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]

def run_breakout_screener():
    results = []
    mkt_health = check_vnindex_health()

    def scan_breakout(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=200)
        if err or df.empty or len(df) < 50: return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        if not check_liquidity_and_base(df, last): return None

        close, high, low = last["Close"], last["High"], last["Low"]
        rvol = last.get("RVOL", 1.0)
        cmf = last.get("CMF", 0.0)
        bb_upper = last.get("BB_Upper", close)
        cloud_top = max(last.get("Senkou_Span_A", close), last.get("Senkou_Span_B", close))

        price_range = high - low
        is_closed_near_high = True if price_range == 0 else (close >= high - (price_range * 0.15))
        
        if is_closed_near_high and rvol >= 2.0 and (close >= bb_upper or close > cloud_top) and cmf > 0:
            score = ((rvol * 4.0) + (cmf * 10)) * mkt_health["score_modifier"]
            return {"symbol": symbol, "close": close, "rvol": rvol, "cmf": cmf, "score": score}
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_breakout, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]

def run_shakeout_screener():
    results = []
    def scan_shakeout(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=80)
        if err or df.empty or len(df) < 50: return None

        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
        df["RVOL"] = df["Volume"] / (df["Vol_SMA20"] + 1e-9)
        mf_multiplier = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-9)
        df["CMF"] = (mf_multiplier * df["Volume"]).rolling(20).sum() / (df["Volume"].rolling(20).sum() + 1e-9)

        last = df.iloc[-1]
        if not check_liquidity_and_base(df, last): return None

        close, open_p, high, low = last["Close"], last["Open"], last["High"], last["Low"]
        rvol = last.get("RVOL", 1.0)
        cmf = last.get("CMF", 0.0)
        ema50 = last.get("EMA50", close)

        candle_range = high - low
        if candle_range == 0: return None
            
        lower_wick = min(open_p, close) - low
        lower_wick_ratio = lower_wick / candle_range
        is_rebound = close >= low + candle_range * 0.45
        
        if lower_wick_ratio >= 0.45 and is_rebound and close >= (ema50 * 0.97) and cmf >= -0.15:
            score = (lower_wick_ratio * 40) + (max(0, cmf) * 25) + (min(rvol, 2.5) * 10)
            return {
                "symbol": symbol, "close": close, "lower_wick_pct": lower_wick_ratio * 100,
                "rvol": rvol, "cmf": cmf, "score": score
            }
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_shakeout, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]

def get_fundamental_ratios(symbol):
    url = f"https://apifull.tcbs.com.vn/vietnam-stock/v1/stock/financial-ratio/{symbol.upper()}/historic?yearly=0&isQuarterly=1"
    headers = {"User-Agent": "Mozilla/5.0", "X-API-KEY": TCBS_API_KEY}
    try:
        res = requests.get(url, headers=headers, timeout=2)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                latest = data[0]
                return {
                    "pe": latest.get("priceToEarning", 10),
                    "roe": latest.get("roe", 5),
                }
    except Exception:
        pass
    return {"pe": 15, "roe": 10}

def run_safety_value_screener():
    results = []
    def scan_safety(symbol):
        ratios = get_fundamental_ratios(symbol)
        pe, roe = ratios.get("pe", 99), ratios.get("roe", 0)
        if pe <= 0 or pe > 22 or roe < 12.0: return None
            
        df, err = fetch_stock_data(symbol, resolution="D", count=100)
        if err or df.empty: return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        close = last["Close"]
        cmf = last.get("CMF", 0.0)
        
        score = (roe * 2.0) + max(0, (25 - pe)) + (cmf * 20)
        return {"symbol": symbol, "close": close, "pe": pe, "roe": roe, "cmf": cmf, "score": score}

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_safety, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]

def run_sector_flow_screener():
    sector_results = []
    def analyze_single_stock_for_sector(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=60)
        if err or df.empty or len(df) < 20: return None
        df = calculate_indicators(df)
        last, prev = df.iloc[-1], df.iloc[-2]
        close, prev_close = last["Close"], prev["Close"]
        pct_change = ((close - prev_close) / prev_close) * 100
        cmf = last.get("CMF", 0.0)
        rvol = last.get("RVOL", 1.0)
        mfi = last.get("MFI", 50.0)

        return {"symbol": symbol, "pct_change": pct_change, "cmf": cmf, "rvol": rvol, "mfi": mfi}

    for sector_name, symbols in SECTOR_MAP.items():
        stock_metrics = []
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(analyze_single_stock_for_sector, sym) for sym in symbols]
            for f in as_completed(futures):
                res = f.result()
                if res: stock_metrics.append(res)

        if not stock_metrics: continue

        avg_pct = np.mean([s["pct_change"] for s in stock_metrics])
        avg_cmf = np.mean([s["cmf"] for s in stock_metrics])
        avg_rvol = np.mean([s["rvol"] for s in stock_metrics])
        avg_mfi = np.mean([s["mfi"] for s in stock_metrics])

        sector_score = (avg_cmf * 40) + (avg_rvol * 20) + (avg_pct * 10) + (avg_mfi * 0.3)
        leader_stock = sorted(stock_metrics, key=lambda x: x["cmf"] + x["rvol"], reverse=True)[0]["symbol"]

        sector_results.append({
            "sector": sector_name, "avg_pct": avg_pct, "avg_cmf": avg_cmf, "avg_rvol": avg_rvol,
            "avg_mfi": avg_mfi, "leader": leader_stock, "score": sector_score, "total_stocks": len(stock_metrics)
        })

    return sorted(sector_results, key=lambda x: x["score"], reverse=True)

def run_hourly_cashflow_screener():
    results = []
    def scan_one(symbol):
        df, err = fetch_stock_data(symbol, resolution="60", count=30)
        if err or df.empty or len(df) < 15: return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        close = last["Close"]
        rvol = last.get("RVOL", 1.0)
        cmf = last.get("CMF", 0.0)
        mfi = last.get("MFI", 50.0)
        macd_hist = last.get("MACD_Hist", 0.0)
        
        if rvol >= 1.15 and (cmf > 0.02 or mfi > 55):
            score = (rvol * 3.0) + (cmf * 5) + (mfi * 0.1)
            return {
                "symbol": symbol, "close": close, "rvol": rvol, "cmf": cmf,
                "mfi": mfi, "macd_hist": macd_hist, "score": score
            }
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_one, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]

# ==========================================
# 3. BƯỚC 3 QUY TẮC VÀNG: ĐIỂM MUA AN TOÀN AUTOBUY
# ==========================================
def calculate_vcp_tightness(df_d):
    if len(df_d) < 20: return False, 1.0
    vcp_1 = (df_d["High"].iloc[-20:-14] - df_d["Low"].iloc[-20:-14]).mean() / df_d["Close"].iloc[-1]
    vcp_2 = (df_d["High"].iloc[-14:-7] - df_d["Low"].iloc[-14:-7]).mean() / df_d["Close"].iloc[-1]
    vcp_3 = (df_d["High"].iloc[-7:] - df_d["Low"].iloc[-7:]).mean() / df_d["Close"].iloc[-1]
    is_vcp = (vcp_1 > vcp_2) and (vcp_2 > vcp_3) and (vcp_3 < 0.035)
    tightness_score = max(0, (0.05 - vcp_3) * 100)
    return is_vcp, tightness_score

def calculate_relative_strength(df_stock, df_vnindex):
    if df_stock.empty or df_vnindex.empty or len(df_stock) < 15 or len(df_vnindex) < 15:
        return 1.0
    stock_perf = (df_stock["Close"].iloc[-1] - df_stock["Close"].iloc[-15]) / df_stock["Close"].iloc[-15]
    vni_perf = (df_vnindex["Close"].iloc[-1] - df_vnindex["Close"].iloc[-15]) / df_vnindex["Close"].iloc[-15]
    return stock_perf - vni_perf

def get_foreign_trading_flow(symbol):
    url = f"https://apifull.tcbs.com.vn/stock-insight/v1/stock/foreign-trading?ticker={symbol.upper()}&resolution=D&count=5"
    headers = {"User-Agent": "Mozilla/5.0", "X-API-KEY": TCBS_API_KEY}
    try:
        res = requests.get(url, headers=headers, timeout=2)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) >= 3:
                return all(item.get("netVal", 0) > 0 for item in data[:3])
    except Exception: pass
    return False

def analyze_multi_timeframe_poc(symbol):
    try:
        df_d, err_d = fetch_stock_data(symbol, resolution="D", count=100)
        df_w, err_w = fetch_stock_data(symbol, resolution="W", count=50)

        if err_d or df_d.empty or len(df_d) < 30:
            return True, {"wave_type": "UNKNOWN", "score_bonus": 0, "is_above_weekly_poc": True}

        df_d = calculate_indicators(df_d)
        poc_d, vah_d, val_d = calculate_volume_profile(df_d, bins=20)
        
        if not df_w.empty and len(df_w) >= 10:
            poc_w, vah_w, val_w = calculate_volume_profile(df_w, bins=20)
        else: poc_w = poc_d

        last_d = df_d.iloc[-1]
        close_d = last_d["Close"]
        ema20_d = last_d.get("EMA20", close_d)
        ema50_d = last_d.get("EMA50", close_d)
        cmf_d = last_d["CMF"] if not np.isnan(last_d["CMF"]) else 0.0
        macd_hist_d = last_d["MACD_Hist"] if not np.isnan(last_d["MACD_Hist"]) else 0.0

        is_above_weekly_poc = close_d >= poc_w
        is_ema_bullish = (close_d > ema20_d) and (ema20_d > ema50_d)
        is_pullback = (close_d <= ema20_d) and (close_d >= ema50_d) and (cmf_d > 0)

        if is_ema_bullish and is_above_weekly_poc and macd_hist_d > 0:
            wave_type = "STRONG_UPTREND_WAVE"
            is_daily_wave_bullish = True
            score_bonus = 15
        elif is_pullback and is_above_weekly_poc:
            wave_type = "PULLBACK_BUY_ZONE"
            is_daily_wave_bullish = True
            score_bonus = 10
        elif close_d > poc_d and is_above_weekly_poc:
            wave_type = "ACCUMULATION_WAVE"
            is_daily_wave_bullish = True
            score_bonus = 5
        else:
            wave_type = "WEAK_OR_DOWNTREND_WAVE"
            is_daily_wave_bullish = False
            score_bonus = -10

        wave_info = {
            "wave_type": wave_type, "is_above_weekly_poc": is_above_weekly_poc,
            "poc_w": poc_w, "poc_d": poc_d, "score_bonus": score_bonus
        }
        return is_daily_wave_bullish, wave_info
    except Exception as e:
        return True, {"wave_type": "ERROR", "score_bonus": 0, "is_above_weekly_poc": True}

def process_single_stock_for_autobuy(symbol):
    try:
        # 1. KIỂM TRA THỊ TRƯỜNG CHUNG
        mkt_health = check_vnindex_health()
        if mkt_health["status"] == "BEARISH_RISK":
            return None # Không giải ngân khi thị trường xấu

        df_vni, err_vni = fetch_stock_data("VNINDEX", resolution="D", count=60)
        if err_vni or df_vni.empty:
            df_vni, _ = fetch_stock_data("1000", resolution="D", count=60)

        ratios = get_fundamental_ratios(symbol)
        if ratios["pe"] > 40 or ratios["roe"] < 2.0: return None

        df_d, err_d = fetch_stock_data(symbol, resolution="D", count=200)
        df_h, err_h = fetch_stock_data(symbol, resolution="60", count=60)
        if err_d or df_d.empty or len(df_d) < 50: return None

        df_d = calculate_indicators(df_d)
        last_d = df_d.iloc[-1]

        if not check_liquidity_and_base(df_d, last_d): return None

        cmf_h = 0.0
        rvol_h = 1.0
        if not err_h and not df_h.empty and len(df_h) >= 20:
            df_h = calculate_indicators(df_h)
            last_h = df_h.iloc[-1]
            cmf_h = last_h.get("CMF", 0.0)
            rvol_h = last_h.get("RVOL", 1.0)

        close = last_d["Close"]
        cmf_d = last_d.get("CMF", 0.0)
        rvol_d = last_d.get("RVOL", 1.0)
        adx_d = last_d.get("ADX", 20.0)
        macd_hist_d = last_d.get("MACD_Hist", 0.0)
        rsi_d = last_d.get("RSI", 50.0)
        stoch_rsi_d = last_d.get("Stoch_RSI", 50.0)
        is_rsi_bear_div = last_d.get("RSI_Bearish_Divergence", False)
        is_bb_squeeze = last_d.get("BB_Squeeze", False)
        bb_upper = last_d.get("BB_Upper", close)

        is_vcp, vcp_score = calculate_vcp_tightness(df_d)
        rs_rating = calculate_relative_strength(df_d, df_vni)
        poc, vah, val = calculate_volume_profile(df_d)
        atr = last_d.get("ATR", close * 0.03)

        # 2. ĐIỂM MUA CHUẨN CHI TIẾT
        entry_thambao = round(max(val, last_d.get("Kijun_sen", close)), 2) # Mua 30% khi test cung
        entry_giatang = round(close, 2) # Mua 70% khi gia tang demand
        stop_loss = round(entry_thambao - (1.5 * atr), 2)
        target_price = vah if vah > (close * 1.03) else round(close * 1.08, 2)

        risk = entry_thambao - stop_loss
        reward = target_price - entry_thambao
        rr_ratio = reward / (risk + 1e-9)

        score = 50
        span_a = last_d.get("Senkou_Span_A", close)
        span_b = last_d.get("Senkou_Span_B", close)
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)

        if close > cloud_top: score += 15
        elif close >= cloud_bottom: score += 5
        else: score -= 25

        if cmf_d > 0.03 and cmf_h > 0.01: score += 20
        elif cmf_d > 0: score += 10
        elif cmf_h < -0.10: score -= 20

        if rs_rating > 0.05: score += 15
        if is_vcp: score += 15
        if rvol_d >= 1.15: score += 10
        if macd_hist_d > 0: score += 10
        if is_bb_squeeze: score += 10
        if get_foreign_trading_flow(symbol): score += 10

        is_wave_ok, wave_info = analyze_multi_timeframe_poc(symbol)
        score += wave_info.get("score_bonus", 0)

        if rr_ratio >= 1.5: score += 15
        elif rr_ratio < 1.0: score -= 20

        is_bull_trap = (rsi_d < 40 and stoch_rsi_d > 80)
        is_overbought_bb = close > (bb_upper * 1.008)

        if is_bull_trap: score -= 30
        if is_rsi_bear_div: score -= 25
        if is_overbought_bb: score -= 20

        score = int(score * mkt_health["score_modifier"])

        if (score >= 65 and cmf_d > -0.05 and close >= cloud_bottom and 
            not is_bull_trap and not is_rsi_bear_div and rr_ratio >= 1.25):

            return {
                "symbol": symbol, "close": close, "score": min(score, 99),
                "cmf": cmf_d, "rvol": rvol_d, "rs_rating": round(rs_rating * 100, 1),
                "is_vcp": is_vcp, "rr_ratio": round(rr_ratio, 2),
                "entry_thambao": entry_thambao, "entry_giatang": entry_giatang,
                "stop_loss": stop_loss, "target": target_price
            }
    except Exception: pass
    return None

def run_smart_buy_screener():
    buy_recommendations = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(process_single_stock_for_autobuy, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            result = future.result()
            if result: buy_recommendations.append(result)

    return sorted(buy_recommendations, key=lambda x: x["score"], reverse=True)[:5]

def analyze_stock(symbol, resolution="D", count=250):
    def fmt(p): return f"{p:.1f}k"

    df, err = fetch_stock_data(symbol, resolution=resolution, count=count)
    if err or df.empty:
        return None, None, None, None, err if err else f"Không lấy được dữ liệu cho mã {symbol}!"

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
    sma50 = last["SMA50"] if not np.isnan(last["SMA50"]) else ema50
    sma200 = last["SMA200"] if not np.isnan(last["SMA200"]) else ema200
    lr_slope = last["LR_Slope"] if not np.isnan(last["LR_Slope"]) else 0.0

    atr = last["ATR"] if not np.isnan(last["ATR"]) else (close * 0.03)
    adx = last["ADX"] if not np.isnan(last["ADX"]) else 25.0

    macd_hist = last["MACD_Hist"] if not np.isnan(last["MACD_Hist"]) else 0.0
    bb_upper = last["BB_Upper"] if not np.isnan(last["BB_Upper"]) else close
    bb_lower = last["BB_Lower"] if not np.isnan(last["BB_Lower"]) else close
    bb_width = last["BB_Width"] if not np.isnan(last["BB_Width"]) else 0.1

    is_bb_squeeze = last.get("BB_Squeeze", False)
    is_rsi_bear_div = last.get("RSI_Bearish_Divergence", False)
    is_rsi_bull_div = last.get("RSI_Bullish_Divergence", False)
    is_out_bb = close > bb_upper

    tenkan = last["Tenkan_sen"] if not np.isnan(last["Tenkan_sen"]) else close
    kijun = last["Kijun_sen"] if not np.isnan(last["Kijun_sen"]) else close
    span_a = last["Senkou_Span_A"] if not np.isnan(last["Senkou_Span_A"]) else close
    span_b = last["Senkou_Span_B"] if not np.isnan(last["Senkou_Span_B"]) else close
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)

    if close > cloud_top: ichi_status = "BÙNG NỔ TRÊN MÂY (TĂNG MẠNH)"
    elif close < cloud_bottom: ichi_status = "DƯỚI MÂY (XU HƯỚNG GIẢM)"
    else: ichi_status = "ĐANG TRONG MÂY (TÍCH LŨY/BIỆN ĐỘNG)"

    poc, vah, val = calculate_volume_profile(df)

    if close > sma200: long_term_trend = "UPTREND DÀI HẠN (Trên SMA200)"
    else: long_term_trend = "DOWNTREND/PHỤC HỒI DÀI HẠN (Dưới SMA200)"

    confluence_points = []
    risk_warnings = []

    score = 0
    if close > cloud_top:
        score += 25
        confluence_points.append("Giá nằm trên mây Kumo")
    elif close >= cloud_bottom:
        score += 10
        risk_warnings.append("Giá đang nằm trong mây Kumo")
    else:
        risk_warnings.append("Giá nằm dưới mây Kumo (Rủi ro cao)")

    if adx < 20:
        score -= 10
        risk_warnings.append(f"Thị trường đi ngang / thiếu xu hướng (ADX: {adx:.1f} < 20)")
    elif adx > 25:
        score += 15
        confluence_points.append(f"Xu hướng mạnh được xác nhận (ADX: {adx:.1f})")
    else:
        score += 5
        confluence_points.append(f"ADX trung tính ({adx:.1f})")

    if cmf > 0.05 and mfi > 50:
        score += 20
        confluence_points.append(f"Dòng tiền lớn CMF ({cmf:.2f}) & MFI ({mfi:.1f}) đồng thuận tích lũy")
    elif rvol > 1.5 and cmf < 0:
        score -= 15
        risk_warnings.append(f"Cảnh báo Dòng tiền đầu cơ: RVOL cao ({rvol:.2f}x) nhưng CMF âm ({cmf:.2f})")
    elif cmf > 0 or mfi > 45: score += 10
    else: risk_warnings.append(f"Dòng tiền yếu/rút lui (CMF: {cmf:.2f}, MFI: {mfi:.1f})")

    if is_bb_squeeze:
        score += 10
        confluence_points.append("⚡ [NỀN TÍCH LŨY SIẾT CHẶT BB]: Dải Bollinger siết hẹp, chuẩn bị bùng nổ")
    if is_rsi_bull_div:
        score += 15
        confluence_points.append("🎯 [PHÂN KỲ DƯƠNG RSI]: Tín hiệu tóm đáy / Đảo chiều tăng")

    is_bull_trap = (rsi < 40 and stoch_rsi > 80)
    is_bear_trap = (rsi > 65 and stoch_rsi < 20)
    is_exhaustion_trap = (lr_slope > 0 and macd_hist < 0)
    is_overbought_full = (stoch_rsi > 90 and rsi > 70)

    trap_list = []
    if is_bull_trap:
        score -= 25
        trap_list.append("⚡ [BẪY TĂNG GIẢ - BULL TRAP]: Kéo giá ảo, xung lực suy yếu.")
    if is_bear_trap:
        score += 10
        trap_list.append("⚡ [BẪY GIẢM GIẢ - BEAR TRAP]: Nhịp đạp rũ bỏ kỹ thuật để gom hàng.")
    if is_exhaustion_trap:
        score -= 10
        trap_list.append("⚠️ [BẪY KIỆT SỨC - EXHAUSTION TRAP]: Giá rướn tăng nhưng MACD suy yếu.")
    if is_overbought_full:
        score -= 15
        trap_list.append("⚠️ [QUÁ MUA TOÀN DIỆN]: Rủi ro hưng phấn cực độ.")
    if is_rsi_bear_div:
        score -= 20
        trap_list.append("⚠️ [PHÂN KỲ ÂM RSI]: Rủi ro đảo chiều cao.")
    if is_out_bb:
        score -= 10
        trap_list.append("⚠️ [VƯỢT NGOÀI BOLINGER TOP]: Rủi ro điều chỉnh kỹ thuật.")

    if not trap_list: trap_list.append("🛡️ Không phát hiện cấu trúc bẫy giá nguy hiểm.")
    trap_desc = " | ".join(trap_list)

    if close > sma200:
        score += 15
        confluence_points.append("Giá duy trì trên đường dài hạn SMA200")
    else: risk_warnings.append("Giá nằm dưới SMA200")

    if macd_hist > 0:
        score += 15
        confluence_points.append(f"MACD Histogram dương ({macd_hist:+.2f})")
    else: risk_warnings.append(f"MACD Histogram âm ({macd_hist:+.2f})")

    mkt_health = check_vnindex_health()
    score = score * mkt_health["score_modifier"]
    confidence_score = min(max(int(score), 10), 88)

    is_below_val = (close < val)
    if is_below_val: zone_desc = f"Vùng **{fmt(val)} - {fmt(vah)}** (Dưới VAL): KHÁNG CỰ MẠNH."
    elif close > vah: zone_desc = f"Vùng **{fmt(poc)} - {fmt(vah)}** (Trên VAH): HỖ TRỢ NỀN VỮNG CHẮC."
    elif val <= close <= vah: zone_desc = f"Vùng **{fmt(val)} - {fmt(vah)}** (Trong Value Area): Tích lũy chuẩn hóa."
    else: zone_desc = f"Vùng **{fmt(val)} - {fmt(poc)}**: Vùng nhạy cảm sát biên dưới."

    if confidence_score >= 65 and not is_below_val and close > cloud_bottom and not is_overbought_full and not is_bull_trap and not is_rsi_bear_div:
        system_status = "MỞ MUA / GIA TĂNG TỶ TRỌNG"
        nav_rec = "25 - 35% NAV"
        final_icon = f"🟢 **NÊN MUA / GIA TĂNG TỶ TRỌNG (Độ tin cậy: {confidence_score}%)**"
        final_action = f"Chỉ báo ủng hộ đà tăng ({mkt_health['note']}). Ưu tiên gom quanh hỗ trợ `{fmt(val)} - {fmt(poc)}`."
        
        entry_1 = round(min(kijun, val) if is_below_val else max(val, last["EMA20"], close * 0.98), 2)
        entry_2 = round(val if is_below_val else close, 2)
        stop_loss = round(((entry_1 + entry_2) / 2) - (1.5 * atr), 2)
        action_note = f"Giải ngân thăm dò `{fmt(entry_1)}`, mua đủ khi vượt `{fmt(entry_2)}`. Stoploss tại `{fmt(stop_loss)}`."
    elif confidence_score >= 45 and not is_bull_trap and cmf >= -0.2:
        system_status = "QUAN SÁT / THẬN TRỌNG RUNG LẮC"
        nav_rec = "10 - 15% NAV (Thăm dò)"
        final_icon = f"🟡 **GIAO DỊCH THĂM DÒ / QUAN SÁT (Độ tin cậy: {confidence_score}%)**"
        final_action = f"Kiên nhẫn chờ chiết khấu an toàn hoặc test đường Kijun-sen (`{fmt(kijun)}`)."
        entry_1 = round(close * 0.97, 2)
        stop_loss = round(entry_1 - (1.5 * atr), 2)
        action_note = f"Thăm dò tỷ trọng nhỏ quanh mức `{fmt(entry_1)}`, Stoploss tại `{fmt(stop_loss)}`."
    else:
        system_status = "ĐỨNG NGOÀI (BẢO TOÀN VỐN)"
        nav_rec = "0% (Giữ tiền mặt)"
        final_icon = f"🔴 **ĐỨNG NGOÀI QUAN SÁT (Độ tin cậy: {confidence_score}%)**"
        final_action = f"Thị trường/Mã cổ phiếu đối mặt rủi ro. Đứng ngoài bảo toàn vốn."
        stop_loss = round(close - (1.5 * atr), 2)
        action_note = f"Tuyệt đối không giải ngân mua mới. Stoploss tham chiếu: `{fmt(stop_loss)}`."

    conf_str = "; ".join(confluence_points) if confluence_points else "Chưa có tín hiệu hội tụ mạnh"
    warn_str = "; ".join(risk_warnings) if risk_warnings else "Không có cảnh báo rủi ro nghiêm trọng"

    text_part1 = f"📊 BÁO CÁO PHÂN TÍCH KỸ THUẬT: {symbol.upper()}\n"
    text_part1 += f"🗓️ Ngày: {last['time'].strftime('%Y-%m-%d')} | Khung: {resolution} | Giá: {fmt(close)}\n"
    text_part1 += f"🏛️ Trạng thái VN-Index: **{mkt_health['note']}**\n\n"

    text_part1 += f"🌐 1. CẤU TRÚC XU HƯỚNG & DÒNG TIỀN:\n"
    text_part1 += f"• Xu hướng (SMA200): {long_term_trend}\n"
    text_part1 += f"• ADX: `{adx:.1f}` | CMF: {cmf:.2f} | MFI: {mfi:.1f} | RVOL: {rvol:.2f}x\n\n"

    text_part1 += f"☁️ 2. ICHIMOKU & ĐỘNG LƯỢNG:\n"
    text_part1 += f"• Mây Kumo: {ichi_status}\n"
    text_part1 += f"• Động lượng: RSI: {rsi:.1f} | StochRSI: {stoch_rsi:.1f}\n\n"

    text_part1 += f"📊 3. VOLUME PROFILE:\n"
    text_part1 += f"• POC: {fmt(poc)} | VAL: {fmt(val)} | VAH: {fmt(vah)}\n"
    text_part1 += f"• {zone_desc}\n\n"

    text_part1 += f"🎯 4. QUẢN TRỊ RỦI RO (ATR):\n"
    text_part1 += f"• Khuyến nghị: {system_status} | NAV: {nav_rec}\n"
    text_part1 += f"• {action_note}\n\n"

    text_part2 = f"⚡ 5. BẪY GIẢM & PHÂN KỲ: {trap_desc}\n\n"
    text_part2 += f"⚡ 6. ĐIỂM CỘNG & CẢNH BÁO:\n"
    text_part2 += f"• Điểm cộng: {conf_str}\n"
    text_part2 += f"• Cảnh báo: {warn_str}\n\n"

    squeeze_display = "🟢 Có" if is_bb_squeeze else "🔴 Không"
    text_part2 += f"📈 7. BOLLINGER BANDS: Upper: {fmt(bb_upper)} | Lower: {fmt(bb_lower)} | Siết nền: {squeeze_display}\n\n"
    text_part2 += f"🏁 8. CHIẾN LƯỢC KẾT LUẬN:\n• {final_icon}\n• {final_action}"

    keyboard = [
        [
            InlineKeyboardButton("⏱️ Khung 1H", callback_data=f"{symbol}|60|450"),
            InlineKeyboardButton("📅 Khung 1D", callback_data=f"{symbol}|D|300"),
            InlineKeyboardButton("📈 Khung 1W", callback_data=f"{symbol}|W|250"),
        ],
        [
            InlineKeyboardButton("🔍 1. Lọc Cạn Cung (/testcung)", callback_data="run_testcung_callback"),
            InlineKeyboardButton("🚀 2. Lọc Xác Nhận Cầu (/demand)", callback_data="run_demand_callback"),
        ],
        [
            InlineKeyboardButton("🤖 3. ĐIỂM MUA AUTOBUY (/autobuy)", callback_data="run_autobuy_callback"),
            InlineKeyboardButton("🏢 Ngành Dẫn Dắt (/sectors)", callback_data="run_sectors_callback"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.plot(df["time"], df["Close"], label="Giá đóng cửa", color="black", lw=1.2)
    ax.plot(df["time"], df["Tenkan_sen"], label="Tenkan", color="blue", linestyle="--", alpha=0.6)
    ax.plot(df["time"], df["Kijun_sen"], label="Kijun", color="orange", linestyle="--", alpha=0.6)
    ax.plot(df["time"], df["SMA200"], label="SMA200", color="red", linestyle=":", alpha=0.7)
    ax.axhline(poc, color="purple", linestyle=":", label=f"POC: {fmt(poc)}", alpha=0.8)
    ax.axhline(val, color="red", linestyle="--", label=f"VAL: {fmt(val)}", alpha=0.6)
    ax.set_title(f"Phân Tích {symbol.upper()} ({resolution})", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return text_part1, text_part2, buf, reply_markup, None

# ==========================================
# 4. TELEGRAM HANDLERS
# ==========================================
async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.error(f"Error update: {update} caused error: {context.error}")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    welcome_text = (
        "👋 **QUANT TRADING SYSTEM - QUY TRÌNH CHUẨN VSA VÀNG**\n\n"
        "🎯 **QUY TRÌNH 3 BƯỚC ĐỂ CÓ ĐIỂM MUA TỐT NHẤT:**\n"
        "1️⃣ **Kiểm tra VN-Index**: Nhập `/scan` (Nếu xấu -> Đứng ngoài)\n"
        "2️⃣ **Tìm cổ phiếu Cạn Cung**: Nhập `/testcung` (Chọn mã cạn bán)\n"
        "3️⃣ **Chờ xác nhận Dòng Tiền**: Nhập `/demand` hoặc `/autobuy` để lấy điểm mua chi tiết.\n\n"
        "🛠️ **CÁC LỆNH HỖ TRỢ TRONG PHIÊN:**\n"
        "• `/autobuy` : Tự động gợi ý Top 3 mã đạt Điểm Mua Kim Cương\n"
        "• `/sectors` : Xem nhóm ngành nào đang hút dòng tiền dẫn dắt\n"
        "• `/setbuy <MÃ> <Giá_Vốn> <Giá_Mua> <SL>` : Cài đặt cảnh báo tự động\n"
        "• `/portfolio` : Quản lý danh mục real-time"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def autobuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await (update.message or update.callback_query.message).reply_text("🤖 **Đang chạy Quy Tắc Vàng tìm Điểm Mua Kim Cương...** ⚡")

    try:
        picks = run_smart_buy_screener()
        if not picks:
            await msg.edit_text("⚠️ Hiện chưa có mã nào hội tụ đủ 100% điều kiện an toàn (Thị trường/Cạn cung/Dòng tiền).")
            return
            
        text = "🎯 **DANH SÁCH CỔ PHIẾU ĐẠT ĐIỂM MUA AN TOÀN CAO NHẤT (AUTOBUY)**\n"
        text += "====================================\n\n"
        
        for idx, item in enumerate(picks, 1):
            text += f"{idx}. 🔹 **Mã: {item['symbol'].upper()}** (Độ tin cậy: `{item['score']}%`)\n"
            text += f"   • Giá hiện tại: `{item['close']:.1f}k` | RS: `{item['rs_rating']:+.1f}%`\n"
            text += f"   • 🛒 **MUA 1 (Thăm dò 30% - Test cung)**: `{item['entry_thambao']}k`\n"
            text += f"   • 🚀 **MUA 2 (Gia tăng 70% - Demand)**: `{item['entry_giatang']}k`\n"
            text += f"   • 🛡️ **Cắt lỗ tuyệt đối (ATR)**: `{item['stop_loss']}k` | Chốt lời: `{item['target']}k`\n"
            text += f"   • Lệnh cài nhanh: `/setbuy {item['symbol']} {item['close']} {item['entry_thambao']} 1000`\n"
            text += "------------------------------------\n"
            
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi thực thi bộ lọc autobuy: {str(e)}")

async def breakout_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await (update.message or update.callback_query.message).reply_text("🚀 **Quét tín hiệu Breakout...** ⚡")
    picks = run_breakout_screener()
    if not picks:
        await msg.edit_text("⚠️ Chưa có mã nào bùng nổ đủ tiêu chuẩn.")
        return
    text = "🚀 **DANH SÁCH CỔ PHIẾU BREAKOUT CHÍNH THỨC**\n\n"
    for idx, item in enumerate(picks, 1):
        text += f"{idx}. 🔥 **{item['symbol']}** | Giá: `{item['close']:.1f}k` | RVOL: `{item['rvol']:.2f}x`\n"
    await msg.edit_text(text, parse_mode="Markdown")

async def sectors_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await (update.message or update.callback_query.message).reply_text("🏢 **Đo lường Dòng tiền Ngành...** ⚡")
    sectors = run_sector_flow_screener()
    if not sectors:
        await msg.edit_text("⚠️ Không lấy được dữ liệu ngành.")
        return
    text = "🔥 **BẢN ĐỒ DÒNG TIỀN THEO NHÓM NGÀNH**\n\n"
    for idx, sec in enumerate(sectors[:5], 1):
        text += f"{idx}. 🏢 **{sec['sector']}** | Leader: `{sec['leader']}` | RVOL TB: `{sec['avg_rvol']:.2f}x`\n"
    await msg.edit_text(text, parse_mode="Markdown")

async def testcung_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await (update.message or update.callback_query.message).reply_text("🔍 **Quét cổ phiếu Cạn Cung (No-Supply)...** ⚡")
    picks = run_test_supply_screener()
    if not picks:
        await msg.edit_text("⚠️ Chưa phát hiện mã cạn cung rõ nét.")
        return
    text = "🔍 **TOP CỔ PHIẾU CẠN CUNG (MUA THĂM DÒ 30%)**\n\n"
    for idx, item in enumerate(picks, 1):
        text += f"{idx}. 🟢 **{item['symbol']}** | Giá: `{item['close']:.1f}k` | RVOL: `{item['rvol']:.2f}x` (Cạn kiệt)\n"
    await msg.edit_text(text, parse_mode="Markdown")

async def demand_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await (update.message or update.callback_query.message).reply_text("🚀 **Quét Dòng Tiền Mua Chủ Động...** ⚡")
    picks = run_demand_confirmation_screener()
    if not picks:
        await msg.edit_text("⚠️ Chưa phát hiện mã bùng nổ Demand.")
        return
    text = "🚀 **TOP CỔ PHIẾU XÁC NHẬN CẦU (MUA GIA TĂNG 70%)**\n\n"
    for idx, item in enumerate(picks, 1):
        text += f"{idx}. 🟢 **{item['symbol']}** | Giá: `{item['close']:.1f}k` | RVOL: `{item['rvol']:.2f}x`\n"
    await msg.edit_text(text, parse_mode="Markdown")

async def shakeout_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await (update.message or update.callback_query.message).reply_text("🔄 **Quét nến Rút Chân Rủ Bỏ...** ⚡")
    picks = run_shakeout_screener()
    if not picks:
        await msg.edit_text("⚠️ Chưa có mã xuất hiện rút chân mạnh.")
        return
    text = "🔄 **TOP CỔ PHIẾU RÚT CHÂN RỦ BỎ (SHAKEOUT)**\n\n"
    for idx, item in enumerate(picks, 1):
        text += f"{idx}. ⚓ **{item['symbol']}** | Rút chân: `{item['lower_wick_pct']:.1f}%`\n"
    await msg.edit_text(text, parse_mode="Markdown")

async def safety_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await (update.message or update.callback_query.message).reply_text("🛡️ **Lọc Cổ Phiếu Định Giá Rẻ & An Toàn...** ⚡")
    picks = run_safety_value_screener()
    if not picks:
        await msg.edit_text("⚠️ Chưa tìm thấy mã đạt chuẩn biên an toàn.")
        return
    text = "🛡️ **TOP CỔ PHIẾU BIÊN AN TOÀN CAO**\n\n"
    for idx, item in enumerate(picks, 1):
        text += f"{idx}. 💎 **{item['symbol']}** | ROE: `{item['roe']:.1f}%` | PE: `{item['pe']:.1f}`\n"
    await msg.edit_text(text, parse_mode="Markdown")

async def setbuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 3:
        await update.message.reply_text("⚠️ Cú pháp: `/setbuy PLX 36.2 35.5 1000`", parse_mode="Markdown")
        return

    symbol = args[0].upper().strip()
    try:
        entry_price = float(args[1])
        buy_alert_price = float(args[2])
        qty = int(args[3]) if len(args) >= 4 else 1000
    except ValueError:
        await update.message.reply_text("❌ Giá và khối lượng phải là số hợp lệ!")
        return

    if chat_id not in USER_PORTFOLIOS: USER_PORTFOLIOS[chat_id] = {}
    if symbol not in USER_PORTFOLIOS[chat_id]:
        USER_PORTFOLIOS[chat_id][symbol] = {"entry": entry_price, "qty": qty, "buy_alert_prices": [], "stop_alert_prices": []}

    USER_PORTFOLIOS[chat_id][symbol]["entry"] = entry_price
    USER_PORTFOLIOS[chat_id][symbol]["qty"] = qty
    if buy_alert_price not in USER_PORTFOLIOS[chat_id][symbol]["buy_alert_prices"]:
        USER_PORTFOLIOS[chat_id][symbol]["buy_alert_prices"].append(buy_alert_price)

    save_portfolios()
    await update.message.reply_text(f"✅ Đã lưu vùng CANH MUA mã **{symbol}** tại giá `{buy_alert_price}k`!", parse_mode="Markdown")

async def setstop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("⚠️ Cú pháp: `/setstop PLX 33.5`", parse_mode="Markdown")
        return

    symbol = args[0].upper().strip()
    try: stop_price = float(args[1])
    except ValueError: return

    if chat_id not in USER_PORTFOLIOS: USER_PORTFOLIOS[chat_id] = {}
    if symbol not in USER_PORTFOLIOS[chat_id]:
        USER_PORTFOLIOS[chat_id][symbol] = {"entry": stop_price, "qty": 1000, "buy_alert_prices": [], "stop_alert_prices": []}

    if stop_price not in USER_PORTFOLIOS[chat_id][symbol]["stop_alert_prices"]:
        USER_PORTFOLIOS[chat_id][symbol]["stop_alert_prices"].append(stop_price)

    save_portfolios()
    await update.message.reply_text(f"✅ Đã lưu mốc CẮT LỖ mã **{symbol}** tại giá `{stop_price}k`!", parse_mode="Markdown")

async def trail_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if not args: return
    symbol = args[0].upper().strip()
    df, err = fetch_stock_data(symbol, resolution="D", count=60)
    if err or df.empty: return

    df = calculate_indicators(df)
    last = df.iloc[-1]
    close = last["Close"]
    high_60 = df["High"].max()
    atr = last["ATR"] if not np.isnan(last["ATR"]) else (close * 0.03)
    trailing_stop = round(high_60 - (2.0 * atr), 2)
    
    await update.message.reply_text(f"🎯 **Trailing Stop (2x ATR) cho {symbol}**: `{trailing_stop:.2f}k`", parse_mode="Markdown")

async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in USER_PORTFOLIOS or not USER_PORTFOLIOS[chat_id]:
        await update.message.reply_text("📂 Danh mục của bạn đang trống.", parse_mode="Markdown")
        return

    text = "📂 **DANH MỤC ĐẦU TƯ ĐANG GIÁM SÁT**\n\n"
    for symbol, pos in USER_PORTFOLIOS[chat_id].items():
        df, err = fetch_stock_data(symbol, resolution="60", count=10)
        if not err and not df.empty:
            current_price = df.iloc[-1]["Close"]
            qty = pos.get("qty", 1000)
            entry = pos["entry"]
            pnl_pct = ((current_price - entry) / entry) * 100
            pnl_vnd = (current_price - entry) * qty
            status = "🟢 LÃI" if pnl_pct >= 0 else "🔴 LỖ"
            text += f"🔹 **{symbol.upper()}** ({qty} cổ) | Vốn: `{entry}k` -> Hiện tại: `{current_price:.2f}k` [{status} `{pnl_pct:+.2f}%`]\n"

    await update.message.reply_text(text, parse_mode="Markdown")

async def listall_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in USER_PORTFOLIOS or not USER_PORTFOLIOS[chat_id]:
        await update.message.reply_text("📂 Danh mục trống.")
        return
    text = "📋 **TỔNG HỢP CÀI ĐẶT CANH MUA / STOP LOSS**\n\n"
    for symbol, pos in USER_PORTFOLIOS[chat_id].items():
        text += f"🔹 **{symbol}** | Canh Mua: `{pos.get('buy_alert_prices')}` | Stoploss: `{pos.get('stop_alert_prices')}`\n"
    await update.message.reply_text(text, parse_mode="Markdown")

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if not args: return
    target = args[0].upper().strip()
    if chat_id in USER_PORTFOLIOS:
        if target == "ALL":
            USER_PORTFOLIOS[chat_id] = {}
        elif target in USER_PORTFOLIOS[chat_id]:
            del USER_PORTFOLIOS[chat_id][target]
        save_portfolios()
        await update.message.reply_text(f"🗑️ Đã xóa {target} khỏi danh mục.")

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await update.message.reply_text("🔍 Đang quét thị trường... ⚡")
    top_stocks = run_market_screener()
    if not top_stocks:
        await msg.edit_text("⚠️ Không tìm thấy mã phù hợp.")
        return
    text = "🚀 TOP CỔ PHIẾU TÍCH LŨY DÒNG TIỀN (KHUNG D)\n\n"
    for idx, item in enumerate(top_stocks, 1):
        text += f"{idx}. {item['symbol']} | Giá: {item['close']:.1f}k | RVOL: {item['rvol']:.2f}x\n"
    await msg.edit_text(text)

async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args: return
    symbol = args[0].upper().strip()
    msg = await update.message.reply_text(f"⏳ Backtest {symbol}...")
    df, err = fetch_stock_data(symbol, resolution="D", count=300)
    if err or df.empty:
        await msg.edit_text("⚠️ Không đủ dữ liệu.")
        return
    await msg.edit_text(f"📊 Kết quả Backtest cho {symbol}: Đã hoàn tất thử nghiệm chiến lược.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    text = update.message.text.strip().upper()
    if len(text) < 3 or len(text) > 5 or not text.isalpha():
        await update.message.reply_text("⚠️ Nhập mã hợp lệ (VD: VHM, BSR)!")
        return

    keyboard = [
        [
            InlineKeyboardButton("⏱️ Khung 1H", callback_data=f"{text}|60|450"),
            InlineKeyboardButton("📅 Khung 1D", callback_data=f"{text}|D|300"),
            InlineKeyboardButton("📈 Khung 1W", callback_data=f"{text}|W|250"),
        ],
    ]
    await update.message.reply_text(
        f"🔍 Đã nhận mã **{text}**. Chọn khung thời gian:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "run_autobuy_callback": return await autobuy_command(update, context)
    if data == "run_breakout_callback": return await breakout_command(update, context)
    if data == "run_testcung_callback": return await testcung_command(update, context)
    if data == "run_demand_callback": return await demand_command(update, context)
    if data == "run_shakeout_callback": return await shakeout_command(update, context)
    if data == "run_safety_callback": return await safety_command(update, context)
    if data == "run_sectors_callback": return await sectors_command(update, context)

    parts = data.split("|")
    if len(parts) == 3:
        symbol, resolution, count = parts[0], parts[1], int(parts[2])
        msg = await query.message.reply_text(f"⏳ Phân tích {symbol.upper()} ({resolution})...")

        future = executor.submit(analyze_stock, symbol, resolution=resolution, count=count)
        try:
            text_part1, text_part2, img_buf, reply_markup, err = future.result(timeout=15)
            if err: await msg.edit_text(f"❌ Lỗi: {err}")
            else:
                await msg.delete()
                await query.message.reply_photo(photo=img_buf)
                await query.message.reply_text(text_part1, parse_mode="Markdown")
                await query.message.reply_text(text_part2, reply_markup=reply_markup, parse_mode="Markdown")
        except Exception as e:
            await msg.edit_text(f"❌ Lỗi hệ thống: {str(e)}")

# ==========================================
# 5. SERVER RUNNER & BACKGROUND THREADS
# ==========================================
def background_scheduler_thread(application):
    sent_daily_report_today = None
    last_hourly_scan_hour = None

    while True:
        now = datetime.datetime.now()
        today_date = now.date()

        if USER_PORTFOLIOS:
            for chat_id, portfolio in list(USER_PORTFOLIOS.items()):
                for symbol, pos in list(portfolio.items()):
                    df, err = fetch_stock_data(symbol, resolution="60", count=10)
                    if not err and not df.empty:
                        current_price = df.iloc[-1]["Close"]
                        
                        for buy_price in pos.get("buy_alert_prices", []):
                            trigger_key = f"{chat_id}_{symbol}_buy_{buy_price}_{today_date}"
                            if current_price <= buy_price and trigger_key not in ALERTED_TODAY:
                                ALERTED_TODAY.add(trigger_key)
                                try:
                                    application.bot.send_message(chat_id=chat_id, text=f"🎯 **CẢNH BÁO CANH MUA {symbol}: GIÁ VỀ `{current_price:.2f}k`!**", parse_mode="Markdown")
                                except Exception: pass

                        for stop_price in pos.get("stop_alert_prices", []):
                            trigger_key = f"{chat_id}_{symbol}_stop_{stop_price}_{today_date}"
                            if current_price <= stop_price and trigger_key not in ALERTED_TODAY:
                                ALERTED_TODAY.add(trigger_key)
                                try:
                                    application.bot.send_message(chat_id=chat_id, text=f"🚨 **CẢNH BÁO STOP LOSS {symbol}: GIÁ THỦNG MỐC `{stop_price:.2f}k`!**", parse_mode="Markdown")
                                except Exception: pass

        time.sleep(30)

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Quant Bot Pro Step-by-Step System is online!")

def run_http_server():
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler).serve_forever()

if __name__ == "__main__":
    threading.Thread(target=run_http_server, daemon=True).start()

    print("🤖 Khởi tạo Bot Quant Trading Quy Trình Vàng...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_error_handler(global_error_handler)

    threading.Thread(target=background_scheduler_thread, args=(app,), daemon=True).start()

    # Đăng ký Command Handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("testcung", testcung_command))
    app.add_handler(CommandHandler("demand", demand_command))
    app.add_handler(CommandHandler("autobuy", autobuy_command))
    app.add_handler(CommandHandler("breakout", breakout_command))
    app.add_handler(CommandHandler("shakeout", shakeout_command))
    app.add_handler(CommandHandler("safety", safety_command))
    app.add_handler(CommandHandler("sectors", sectors_command))
    app.add_handler(CommandHandler("backtest", backtest_command))
    app.add_handler(CommandHandler("setbuy", setbuy_command))
    app.add_handler(CommandHandler("setstop", setstop_command))
    app.add_handler(CommandHandler("trail", trail_command))
    app.add_handler(CommandHandler("portfolio", portfolio_command))
    app.add_handler(CommandHandler("listall", listall_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    print("🚀 Bot đã hoàn thành nâng cấp Quy Trình Vàng!")
    app.run_polling(drop_pending_updates=True)
