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
# QUẢN LÝ LƯU TRỮ GOOGLE SHEETS API
# ==========================================
USER_PORTFOLIOS = {}

def load_portfolios():
    if not GOOGLE_SHEET_API_URL:
        logging.warning("Chưa cấu hình GOOGLE_SHEET_API_URL!")
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
# 1. TÍNH TOÁN CHỈ BÁO & DỮ LIỆU
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
            return pd.DataFrame(), f"Lỗi API: {str(e)}"

    if df_raw.empty:
        return pd.DataFrame(), f"Không tìm thấy dữ liệu cho mã '{symbol}'."

    df_raw = df_raw[["time", "Open", "High", "Low", "Close", "Volume"]].sort_values("time").reset_index(drop=True)

    if resolution == "W":
        df_raw.set_index("time", inplace=True)
        df_weekly = df_raw.resample("W-FRI").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna().reset_index()
        return df_weekly.tail(250).reset_index(drop=True), None

    if resolution == "M":
        df_raw.set_index("time", inplace=True)
        df_monthly = df_raw.resample("ME").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
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

def check_vnindex_health():
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
        return {"status": "BULLISH", "score_modifier": 1.15, "note": "VN-Index Uptrend ủng hộ"}
    elif close < ema50 or macd_hist < -0.5:
        return {"status": "BEARISH_RISK", "score_modifier": 0.75, "note": "VN-Index rủi ro / Xu hướng yếu"}
    else:
        return {"status": "SIDEWAYS", "score_modifier": 1.0, "note": "VN-Index tích lũy/giằng co"}

def get_foreign_trading_flow(symbol):
    url = f"https://apifull.tcbs.com.vn/stock-insight/v1/stock/foreign-trading?ticker={symbol.upper()}&resolution=D&count=5"
    headers = {"User-Agent": "Mozilla/5.0", "X-API-KEY": TCBS_API_KEY}
    try:
        res = requests.get(url, headers=headers, timeout=2)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) >= 3:
                return all(item.get("netVal", 0) > 0 for item in data[:3])
    except Exception:
        pass
    return False

def get_fundamental_ratios(symbol):
    url = f"https://apifull.tcbs.com.vn/vietnam-stock/v1/stock/financial-ratio/{symbol.upper()}/historic?yearly=0&isQuarterly=1"
    headers = {"User-Agent": "Mozilla/5.0", "X-API-KEY": TCBS_API_KEY}
    try:
        res = requests.get(url, headers=headers, timeout=2)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                return {"pe": data[0].get("priceToEarning", 10), "roe": data[0].get("roe", 5)}
    except Exception:
        pass
    return {"pe": 15, "roe": 10}

def calculate_vcp_tightness(df_d):
    if len(df_d) < 20:
        return False, 1.0
    vcp_1 = (df_d["High"].iloc[-20:-14] - df_d["Low"].iloc[-20:-14]).mean() / df_d["Close"].iloc[-1]
    vcp_2 = (df_d["High"].iloc[-14:-7] - df_d["Low"].iloc[-14:-7]).mean() / df_d["Close"].iloc[-1]
    vcp_3 = (df_d["High"].iloc[-7:] - df_d["Low"].iloc[-7:]).mean() / df_d["Close"].iloc[-1]
    is_vcp = (vcp_1 > vcp_2) and (vcp_2 > vcp_3) and (vcp_3 < 0.035)
    return is_vcp, max(0, (0.05 - vcp_3) * 100)

def calculate_relative_strength(df_stock, df_vnindex):
    if df_stock.empty or df_vnindex.empty or len(df_stock) < 15 or len(df_vnindex) < 15:
        return 1.0
    stock_perf = (df_stock["Close"].iloc[-1] - df_stock["Close"].iloc[-15]) / df_stock["Close"].iloc[-15]
    vni_perf = (df_vnindex["Close"].iloc[-1] - df_vnindex["Close"].iloc[-15]) / df_vnindex["Close"].iloc[-15]
    return stock_perf - vni_perf

def analyze_multi_timeframe_poc(symbol):
    try:
        df_d, err_d = fetch_stock_data(symbol, resolution="D", count=100)
        df_w, err_w = fetch_stock_data(symbol, resolution="W", count=50)

        if err_d or df_d.empty or len(df_d) < 30:
            return True, {"wave_type": "UNKNOWN", "score_bonus": 0, "is_above_weekly_poc": True}

        df_d = calculate_indicators(df_d)
        poc_d, vah_d, val_d = calculate_volume_profile(df_d, bins=20)
        poc_w = calculate_volume_profile(df_w, bins=20)[0] if not df_w.empty and len(df_w) >= 10 else poc_d

        last_d = df_d.iloc[-1]
        close_d = last_d["Close"]
        ema20_d, ema50_d = last_d.get("EMA20", close_d), last_d.get("EMA50", close_d)
        cmf_d = last_d["CMF"] if not np.isnan(last_d["CMF"]) else 0.0
        macd_hist_d = last_d["MACD_Hist"] if not np.isnan(last_d["MACD_Hist"]) else 0.0

        is_above_weekly_poc = close_d >= poc_w
        is_ema_bullish = (close_d > ema20_d) and (ema20_d > ema50_d)
        is_pullback = (close_d <= ema20_d) and (close_d >= ema50_d) and (cmf_d > 0)

        if is_ema_bullish and is_above_weekly_poc and macd_hist_d > 0:
            return True, {"wave_type": "STRONG_UPTREND_WAVE", "score_bonus": 15}
        elif is_pullback and is_above_weekly_poc:
            return True, {"wave_type": "PULLBACK_BUY_ZONE", "score_bonus": 10}
        elif close_d > poc_d and is_above_weekly_poc:
            return True, {"wave_type": "ACCUMULATION_WAVE", "score_bonus": 5}
        else:
            return False, {"wave_type": "WEAK_OR_DOWNTREND_WAVE", "score_bonus": -10}
    except Exception as e:
        return True, {"wave_type": "ERROR", "score_bonus": 0}

# ==========================================
# 2. BỘ LỌC PHỄU THEO TRÌNH TỰ BỘ QUY TẮC VÀNG
# ==========================================
def check_liquidity_and_base(df, last):
    close = last["Close"]
    vol_sma20 = last["Vol_SMA20"] if not np.isnan(last.get("Vol_SMA20", np.nan)) else last["Volume"]
    return (vol_sma20 * close * 1000) >= 5_000_000_000

# Step 1: Quét Dòng Tiền Theo Ngành (/sectors)
def run_sector_flow_screener():
    sector_results = []
    def analyze_single(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=60)
        if err or df.empty or len(df) < 20: return None
        df = calculate_indicators(df)
        last, prev = df.iloc[-1], df.iloc[-2]
        return {
            "symbol": symbol,
            "pct_change": ((last["Close"] - prev["Close"]) / prev["Close"]) * 100,
            "cmf": last.get("CMF", 0.0), "rvol": last.get("RVOL", 1.0), "mfi": last.get("MFI", 50.0)
        }

    for sector_name, symbols in SECTOR_MAP.items():
        stock_metrics = []
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(analyze_single, sym) for sym in symbols]
            for f in as_completed(futures):
                if f.result(): stock_metrics.append(f.result())
        if not stock_metrics: continue

        avg_pct = np.mean([s["pct_change"] for s in stock_metrics])
        avg_cmf = np.mean([s["cmf"] for s in stock_metrics])
        avg_rvol = np.mean([s["rvol"] for s in stock_metrics])
        avg_mfi = np.mean([s["mfi"] for s in stock_metrics])
        leader_stock = sorted(stock_metrics, key=lambda x: x["cmf"] + x["rvol"], reverse=True)[0]["symbol"]

        sector_results.append({
            "sector": sector_name, "avg_pct": avg_pct, "avg_cmf": avg_cmf,
            "avg_rvol": avg_rvol, "avg_mfi": avg_mfi, "leader": leader_stock,
            "score": (avg_cmf * 40) + (avg_rvol * 20) + (avg_pct * 10) + (avg_mfi * 0.3)
        })
    return sorted(sector_results, key=lambda x: x["score"], reverse=True)

# Step 2: Quét Tích Lũy Nền & Dòng Tiền (/scan)
def run_market_screener():
    results = []
    mkt_health = check_vnindex_health()
    def scan_one(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=200)
        if err or df.empty or len(df) < 50: return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        if not check_liquidity_and_base(df, last): return None

        close, rvol, cmf = last["Close"], last.get("RVOL", 1.0), last.get("CMF", 0.0)
        ema50, price_range_pct = last.get("EMA50", close), last.get("Price_Range_Pct", 0.1)

        if (rvol >= 1.1 or price_range_pct < 0.08) and cmf > 0.01 and close > ema50:
            score = ((rvol * 2.5) + (cmf * 6) + (max(0, 0.1 - price_range_pct) * 20)) * mkt_health["score_modifier"]
            return {"symbol": symbol, "close": close, "rvol": rvol, "cmf": cmf, "score": score}
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_one, sym) for sym in WATCHLIST_SCAN]
        for f in as_completed(futures):
            if f.result(): results.append(f.result())
    return sorted(results, key=lambda x: x["score"], reverse=True)[:8]

# Step 3: Quét Cạn Cung / No-Supply (/testcung) -> VÀO LỆNH 1 (MUA THĂM DÒ 30%)
def run_test_supply_screener():
    results = []
    def scan_test_supply(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=150)
        if err or df.empty or len(df) < 50: return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        if not check_liquidity_and_base(df, last): return None

        close, high, low = last["Close"], last["High"], last["Low"]
        rvol, cmf = last.get("RVOL", 1.0), last.get("CMF", 0.0)
        price_range = high - low
        is_pinbar = True if price_range == 0 else (close >= low + price_range * 0.5)

        if rvol <= 0.65 and is_pinbar and cmf >= -0.1:
            return {"symbol": symbol, "close": close, "rvol": rvol, "cmf": cmf, "score": (1.0 - rvol) * 50 + (max(0, cmf) * 20)}
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_test_supply, sym) for sym in WATCHLIST_SCAN]
        for f in as_completed(futures):
            if f.result(): results.append(f.result())
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]

# Step 4: Quét Cầu Mua Chủ Động (/demand) -> VÀO LỆNH 2 (MUA GIA TĂNG 70%)
def run_demand_confirmation_screener():
    results = []
    def scan_demand(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=150)
        if err or df.empty or len(df) < 50: return None
        df = calculate_indicators(df)
        last, prev = df.iloc[-1], df.iloc[-2]
        if not check_liquidity_and_base(df, last): return None

        close, high, low = last["Close"], last["High"], last["Low"]
        rvol, cmf, mfi = last.get("RVOL", 1.0), last.get("CMF", 0.0), last.get("MFI", 50.0)
        kijun = last.get("Kijun_sen", close)
        candle_range = high - low
        if candle_range == 0: return None

        is_strong_close = ((close - low) / candle_range) >= 0.70
        was_no_supply = prev.get("RVOL", 1.0) <= 0.70

        if rvol >= 1.50 and is_strong_close and cmf >= 0.08 and close >= kijun:
            score = (rvol * 20) + (cmf * 50) + (mfi * 0.5) + (30 if was_no_supply else 0)
            return {"symbol": symbol, "close": close, "rvol": rvol, "cmf": cmf, "mfi": mfi, "was_no_supply": was_no_supply, "score": score}
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_demand, sym) for sym in WATCHLIST_SCAN]
        for f in as_completed(futures):
            if f.result(): results.append(f.result())
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]

# Step 5: Quét Bùng Nổ Xu Hướng (/breakout)
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
        rvol, cmf = last.get("RVOL", 1.0), last.get("CMF", 0.0)
        bb_upper = last.get("BB_Upper", close)
        cloud_top = max(last.get("Senkou_Span_A", close), last.get("Senkou_Span_B", close))
        price_range = high - low
        is_closed_near_high = True if price_range == 0 else (close >= high - (price_range * 0.15))

        if is_closed_near_high and rvol >= 2.0 and (close >= bb_upper or close > cloud_top) and cmf > 0:
            return {"symbol": symbol, "close": close, "rvol": rvol, "cmf": cmf, "score": ((rvol * 4.0) + (cmf * 10)) * mkt_health["score_modifier"]}
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_breakout, sym) for sym in WATCHLIST_SCAN]
        for f in as_completed(futures):
            if f.result(): results.append(f.result())
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]

# Step 6: Lọc Điểm Mua Chuẩn VCP & Bộ Quy Tắc Vàng (/autobuy)
def process_single_stock_for_autobuy(symbol):
    try:
        mkt_health = check_vnindex_health()
        if mkt_health["status"] == "BEARISH_RISK":
            return None # Quy tắc 1: Bỏ qua khi VN-Index rủi ro cao

        df_vni, _ = fetch_stock_data("VNINDEX", resolution="D", count=60)
        if df_vni.empty: df_vni, _ = fetch_stock_data("1000", resolution="D", count=60)

        ratios = get_fundamental_ratios(symbol)
        if ratios["pe"] > 40 or ratios["roe"] < 2.0: return None

        df_d, err_d = fetch_stock_data(symbol, resolution="D", count=200)
        df_h, err_h = fetch_stock_data(symbol, resolution="60", count=60)
        if err_d or df_d.empty or len(df_d) < 50: return None

        df_d = calculate_indicators(df_d)
        last_d = df_d.iloc[-1]
        if not check_liquidity_and_base(df_d, last_d): return None

        close = last_d["Close"]
        span_a, span_b = last_d.get("Senkou_Span_A", close), last_d.get("Senkou_Span_B", close)
        cloud_top, cloud_bottom = max(span_a, span_b), min(span_a, span_b)
        poc, vah, val = calculate_volume_profile(df_d)

        # Quy tắc 1: Giá phải TRÊN MÂY KUMO và KHÔNG DƯỚI VAL
        if close < cloud_bottom or close < val:
            return None

        cmf_d = last_d.get("CMF", 0.0)
        cmf_h = 0.0
        if not err_h and not df_h.empty and len(df_h) >= 20:
            df_h = calculate_indicators(df_h)
            cmf_h = df_h.iloc[-1].get("CMF", 0.0)

        is_vcp, vcp_score = calculate_vcp_tightness(df_d)
        rs_rating = calculate_relative_strength(df_d, df_vni)
        atr = last_d.get("ATR", close * 0.03)

        entry_price = round(max(val, close * 0.98), 2)
        stop_loss = round(entry_price - (1.5 * atr), 2) # Quy tắc 3: Cutloss 1.5x ATR
        target_price = vah if vah > (close * 1.03) else round(close * 1.08, 2)
        rr_ratio = (target_price - entry_price) / ((entry_price - stop_loss) + 1e-9)

        if rr_ratio < 1.25: # Quy tắc 3: Tỷ lệ Reward/Risk phải >= 1.25
            return None

        score = 50 + (15 if close > cloud_top else 5) + (20 if cmf_d > 0.03 else 10) + (15 if is_vcp else 0)
        is_bull_trap = (last_d.get("RSI", 50) < 40 and last_d.get("Stoch_RSI", 50) > 80)
        if is_bull_trap or last_d.get("RSI_Bearish_Divergence", False): return None

        return {
            "symbol": symbol, "close": close, "score": min(int(score * mkt_health["score_modifier"]), 99),
            "cmf": cmf_d, "cmf_h": cmf_h, "rvol": last_d.get("RVOL", 1.0),
            "rs_rating": round(rs_rating * 100, 1), "is_vcp": is_vcp,
            "rr_ratio": round(rr_ratio, 2), "entry": entry_price,
            "stop_loss": stop_loss, "target": target_price,
            "suggested_nav": "30% Thăm dò -> 70% Gia tăng"
        }
    except Exception:
        return None

def run_smart_buy_screener():
    buy_recommendations = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(process_single_stock_for_autobuy, sym) for sym in WATCHLIST_SCAN]
        for f in as_completed(futures):
            if f.result(): buy_recommendations.append(f.result())
    return sorted(buy_recommendations, key=lambda x: x["score"], reverse=True)[:5]

# Bổ trợ: Quét Bear Trap & Biên An Toàn
def run_shakeout_screener():
    results = []
    def scan_shakeout(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=80)
        if err or df.empty or len(df) < 50: return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        if not check_liquidity_and_base(df, last): return None
        close, open_p, high, low = last["Close"], last["Open"], last["High"], last["Low"]
        candle_range = high - low
        if candle_range == 0: return None
        lower_wick_ratio = (min(open_p, close) - low) / candle_range
        if lower_wick_ratio >= 0.45 and close >= low + candle_range * 0.45:
            return {"symbol": symbol, "close": close, "lower_wick_pct": lower_wick_ratio * 100, "score": lower_wick_ratio * 100}
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_shakeout, sym) for sym in WATCHLIST_SCAN]
        for f in as_completed(futures):
            if f.result(): results.append(f.result())
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]

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
        return {"symbol": symbol, "close": last["Close"], "pe": pe, "roe": roe, "score": (roe * 2.0) + max(0, (25 - pe))}

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_safety, sym) for sym in WATCHLIST_SCAN]
        for f in as_completed(futures):
            if f.result(): results.append(f.result())
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]

def run_hourly_cashflow_screener():
    results = []
    def scan_one(symbol):
        df, err = fetch_stock_data(symbol, resolution="60", count=30)
        if err or df.empty or len(df) < 15: return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        if last.get("RVOL", 1.0) >= 1.15 and (last.get("CMF", 0.0) > 0.02 or last.get("MFI", 50.0) > 55):
            return {"symbol": symbol, "close": last["Close"], "rvol": last.get("RVOL", 1.0), "cmf": last.get("CMF", 0.0), "mfi": last.get("MFI", 50.0), "macd_hist": last.get("MACD_Hist", 0.0), "score": last.get("RVOL", 1.0) * 3.0}
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_one, sym) for sym in WATCHLIST_SCAN]
        for f in as_completed(futures):
            if f.result(): results.append(f.result())
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]

# ==========================================
# 3. PHÂN TÍCH CHUYÊN SÂU ĐƠN MÃ
# ==========================================
def analyze_stock(symbol, resolution="D", count=250):
    fmt = lambda p: f"{p:.1f}k"
    df, err = fetch_stock_data(symbol, resolution=resolution, count=count)
    if err or df.empty: return None, None, None, None, err if err else "Không lấy được dữ liệu!"

    df = calculate_indicators(df)
    last = df.iloc[-1]
    close, cmf, mfi, rvol = last["Close"], last["CMF"], last["MFI"], last["RVOL"]
    rsi, stoch_rsi, atr = last["RSI"], last["Stoch_RSI"], last["ATR"]
    sma200, macd_hist = last["SMA200"], last["MACD_Hist"]
    poc, vah, val = calculate_volume_profile(df)

    cloud_top = max(last["Senkou_Span_A"], last["Senkou_Span_B"])
    cloud_bottom = min(last["Senkou_Span_A"], last["Senkou_Span_B"])

    text_part1 = f"📊 **PHÂN TÍCH KỸ THUẬT: {symbol.upper()}** ({resolution})\n"
    text_part1 += f"• Giá hiện tại: `{fmt(close)}` | VN-Index: **{check_vnindex_health()['note']}**\n\n"
    text_part1 += f"1️⃣ **Mây Kumo & Trend:** Giá {'TRÊN MÂY (Tốt)' if close > cloud_top else 'DƯỚI MÂY (Rủi ro)'}\n"
    text_part1 += f"2️⃣ **Dòng tiền:** CMF `{cmf:.2f}` | RVOL `{rvol:.2f}x` | MFI `{mfi:.1f}`\n"
    text_part1 += f"3️⃣ **Volume Profile:** POC `{fmt(poc)}` | VAL `{fmt(val)}` | VAH `{fmt(vah)}`\n"

    stop_loss = round(close - (1.5 * atr), 2)
    text_part2 = f"🎯 **ĐIỂM MUA CHUẨN THỰC CHIẾN:**\n"
    text_part2 += f"• Mua thăm dò (30%): Quanh mốc VAL/Kijun `{fmt(val)}`\n"
    text_part2 += f"• Mua gia tăng (70%): Khi có nến /demand bùng nổ RVOL > 1.5x\n"
    text_part2 += f"• Cắt lỗ 1.5x ATR: `{fmt(stop_loss)}`\n"

    keyboard = [[
        InlineKeyboardButton("⏱️ 1H", callback_data=f"{symbol}|60|450"),
        InlineKeyboardButton("📅 1D", callback_data=f"{symbol}|D|300"),
        InlineKeyboardButton("📈 1W", callback_data=f"{symbol}|W|250"),
    ]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(df["time"], df["Close"], label="Close", color="black")
    ax.axhline(poc, color="purple", linestyle=":", label=f"POC: {fmt(poc)}")
    ax.axhline(val, color="red", linestyle="--", label=f"VAL: {fmt(val)}")
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return text_part1, text_part2, buf, reply_markup, None

# ==========================================
# 4. CHẠY NGẦM VÀ LẬP LỊCH TỰ ĐỘNG
# ==========================================
def background_scheduler_thread(application):
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
                            if current_price <= buy_price and f"{chat_id}_{symbol}_buy_{buy_price}_{today_date}" not in ALERTED_TODAY:
                                ALERTED_TODAY.add(f"{chat_id}_{symbol}_buy_{buy_price}_{today_date}")
                                application.bot.send_message(chat_id=chat_id, text=f"🎯 **CẢNH BÁO CANH MUA {symbol}: Giá đã về `{current_price:.2f}k`!**", parse_mode="Markdown")
                        for stop_price in pos.get("stop_alert_prices", []):
                            if current_price <= stop_price and f"{chat_id}_{symbol}_stop_{stop_price}_{today_date}" not in ALERTED_TODAY:
                                ALERTED_TODAY.add(f"{chat_id}_{symbol}_stop_{stop_price}_{today_date}")
                                application.bot.send_message(chat_id=chat_id, text=f"🚨 **CẢNH BÁO CẮT LỖ {symbol}: Giá chạm mốc `{current_price:.2f}k`!**", parse_mode="Markdown")
        time.sleep(30)

def run_strategy_backtest(symbol):
    df, err = fetch_stock_data(symbol, resolution="D", count=300)
    if err or df.empty or len(df) < 100: return f"⚠️ Không đủ dữ liệu backtest {symbol}."
    df = calculate_indicators(df)
    trades, wins, total_return = 0, 0, 0.0
    for i in range(50, len(df) - 10):
        if df["RVOL"].iloc[i] > 1.2 and df["CMF"].iloc[i] > 0.02 and df["Close"].iloc[i] > df["EMA50"].iloc[i]:
            trades += 1
            ret = ((df["Close"].iloc[i+5] - df["Close"].iloc[i]) / df["Close"].iloc[i]) * 100
            total_return += ret
            if ret > 0: wins += 1
    return f"📊 **KẾT QUẢ BACKTEST {symbol}:**\n• Số lệnh: {trades} | Thắng: {wins}\n• Winrate: {(wins/trades*100) if trades else 0:.1f}% | Hiệu suất: {total_return:+.2f}%"

# ==========================================
# 5. LỆNH TELEGRAM BOT
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ACTIVE_CHAT_IDS.add(update.effective_chat.id)
    text = (
        "👋 **BỘ QUY TẮC VÀNG VÀO LỆNH (QUANT TRADING PRO)**\n\n"
        "1️⃣ **Bước 1: Tìm Ngành & Mã Tích Lũy**\n"
        "   ├ `/sectors` : Tìm Dòng Tiền Ngành Dẫn Dắt (Sector Flow)\n"
        "   └ `/scan`    : Tìm Cổ Phiếu Tích Lũy Nền\n\n"
        "2️⃣ **Bước 2: Săn Tín Hiệu Vào Lệnh Chuẩn**\n"
        "   ├ `/testcung`: **Lệnh 1 (Mua Thăm Dò 30%)** - Cạn Cung / No-Supply\n"
        "   ├ `/demand`  : **Lệnh 2 (Mua Gia Tăng 70%)** - Cầu Mua Chủ Động\n"
        "   ├ `/breakout`: Điểm Bùng Nổ Giá & Khối Lượng\n"
        "   └ `/autobuy` : **Lọc Điểm Mua Chuẩn Bộ Quy Tắc Vàng**\n\n"
        "3️⃣ **Bước 3: Quản Trị Lệnh & Danh Mục**\n"
        "   ├ `/setbuy`  : Cài đặt giá canh mua tự động\n"
        "   ├ `/setstop` : Cài đặt cắt lỗ 1.5x ATR\n"
        "   └ `/portfolio`: Quản lý danh mục real-time"
    )
    await update.message.reply_text(text, parse_mode="Markdown")

async def autobuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🤖 **Đang lọc điểm mua theo Bộ Quy Tắc Vàng...** ⚡")
    picks = run_smart_buy_screener()
    if not picks:
        await msg.edit_text("⚠️ Không có mã nào hội đủ 100% Bộ Quy Tắc Vàng (VNIndex + Trên Kumo + R:R chuẩn).")
        return
    text = "🎯 **TOP ĐIỂM MUA CHUẨN BỘ QUY TẮC VÀNG (AUTOBUY)**\n\n"
    for idx, item in enumerate(picks, 1):
        text += f"{idx}. 🔹 **{item['symbol']}** (Độ tin cậy: `{item['score']}%`)\n"
        text += f"   • Giá: `{item['close']:.1f}k` | RS: `{item['rs_rating']:+.1f}%` | R:R: `{item['rr_ratio']}x`\n"
        text += f"   • 🎯 **Mua**: `{item['entry']}k` | **Cắt lỗ ATR**: `{item['stop_loss']}k`\n"
        text += f"   • 💡 **Chiến lược đi tiền**: `{item['suggested_nav']}`\n------------------------------------\n"
    await msg.edit_text(text, parse_mode="Markdown")

async def sectors_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🏢 **Đang đo lường Dòng Tiền Ngành...** ⚡")
    sectors = run_sector_flow_screener()
    text = "🔥 **DÒNG TIỀN THEO NHÓM NGÀNH (STEP 1)**\n\n"
    for idx, sec in enumerate(sectors, 1):
        icon = "🟢" if sec['avg_cmf'] > 0.05 else ("🟡" if sec['avg_cmf'] > 0 else "🔴")
        text += f"{idx}. {icon} **{sec['sector']}** | Leader: `{sec['leader']}` | RVOL: `{sec['avg_rvol']:.2f}x`\n"
    await msg.edit_text(text, parse_mode="Markdown")

async def testcung_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔍 **Quét mã Cạn Cung (Vào Lệnh 1 - Mua Thăm Dò 30%)...** ⚡")
    picks = run_test_supply_screener()
    text = "🔍 **CỔ PHIẾU CẠN CUNG (MUA THĂM DÒ 30%)**\n\n"
    for idx, item in enumerate(picks, 1):
        text += f"{idx}. 🟢 **{item['symbol']}** | Giá: `{item['close']:.1f}k` | RVOL cạn kiệt: `{item['rvol']:.2f}x`\n"
    await msg.edit_text(text, parse_mode="Markdown")

async def demand_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🚀 **Quét Cầu Mua Chủ Động (Vào Lệnh 2 - Gia Tăng 70%)...** ⚡")
    picks = run_demand_confirmation_screener()
    text = "🚀 **CẦU MUA CHỦ ĐỘNG (MUA GIA TĂNG 70%)**\n\n"
    for idx, item in enumerate(picks, 1):
        text += f"{idx}. 🔥 **{item['symbol']}** | Giá: `{item['close']:.1f}k` | RVOL bùng nổ: `{item['rvol']:.2f}x`\n"
    await msg.edit_text(text, parse_mode="Markdown")

async def breakout_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🚀 **Quét tín hiệu Bùng Nổ...** ⚡")
    picks = run_breakout_screener()
    text = "🚀 **CỔ PHIẾU BÙNG NỔ (BREAKOUT)**\n\n"
    for idx, item in enumerate(picks, 1):
        text += f"{idx}. 🔥 **{item['symbol']}** | Giá: `{item['close']:.1f}k` | RVOL: `{item['rvol']:.2f}x`\n"
    await msg.edit_text(text, parse_mode="Markdown")

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔍 **Quét cổ phiếu tích lũy nền...** ⚡")
    picks = run_market_screener()
    text = "🚀 **CỔ PHIẾU TÍCH LŨY NỀN & DÒNG TIỀN**\n\n"
    for idx, item in enumerate(picks, 1):
        text += f"{idx}. {item['symbol']} | Giá: {item['close']:.1f}k | RVOL: {item['rvol']:.2f}x\n"
    await msg.edit_text(text)

async def shakeout_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔄 **Quét nến rút chân Rũ Bỏ...** ⚡")
    picks = run_shakeout_screener()
    text = "🔄 **MÃ RỦ BỎ RÚT CHÂN (SHAKEOUT)**\n\n"
    for idx, item in enumerate(picks, 1):
        text += f"{idx}. ⚓ **{item['symbol']}** | Rút chân: `{item['lower_wick_pct']:.1f}%`\n"
    await msg.edit_text(text, parse_mode="Markdown")

async def safety_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🛡️ **Quét mã Biên An Toàn Cơ Bản...** ⚡")
    picks = run_safety_value_screener()
    text = "🛡️ **MÃ BIÊN AN TOÀN (ROE > 12%, P/E HỢP LÝ)**\n\n"
    for idx, item in enumerate(picks, 1):
        text += f"{idx}. 💎 **{item['symbol']}** | ROE: `{item['roe']:.1f}%` | P/E: `{item['pe']:.1f}`\n"
    await msg.edit_text(text, parse_mode="Markdown")

async def setbuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 3:
        await update.message.reply_text("⚠️ Cú pháp: `/setbuy PLX 36.2 35.5 1000`", parse_mode="Markdown")
        return
    symbol, entry, buy_alert = args[0].upper(), float(args[1]), float(args[2])
    qty = int(args[3]) if len(args) >= 4 else 1000
    
    if chat_id not in USER_PORTFOLIOS: USER_PORTFOLIOS[chat_id] = {}
    USER_PORTFOLIOS[chat_id][symbol] = {"entry": entry, "qty": qty, "buy_alert_prices": [buy_alert], "stop_alert_prices": []}
    save_portfolios()
    await update.message.reply_text(f"✅ Đã lưu cài đặt CANH MUA mã **{symbol}** tại mốc `{buy_alert}k`!", parse_mode="Markdown")

async def setstop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("⚠️ Cú pháp: `/setstop PLX 33.5`", parse_mode="Markdown")
        return
    symbol, stop_price = args[0].upper(), float(args[1])
    if chat_id not in USER_PORTFOLIOS: USER_PORTFOLIOS[chat_id] = {}
    if symbol not in USER_PORTFOLIOS[chat_id]: USER_PORTFOLIOS[chat_id][symbol] = {"entry": stop_price, "qty": 1000, "buy_alert_prices": [], "stop_alert_prices": []}
    USER_PORTFOLIOS[chat_id][symbol]["stop_alert_prices"] = [stop_price]
    save_portfolios()
    await update.message.reply_text(f"✅ Đã lưu CẮT LỖ mã **{symbol}** tại mốc `{stop_price}k`!", parse_mode="Markdown")

async def trail_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args: return
    symbol = context.args[0].upper()
    df, _ = fetch_stock_data(symbol, resolution="D", count=60)
    if df.empty: return
    df = calculate_indicators(df)
    last = df.iloc[-1]
    trailing_stop = round(df["High"].max() - (2.0 * last["ATR"]), 2)
    await update.message.reply_text(f"🛡️ **Trailing Stop đề xuất cho {symbol}**: `{trailing_stop}k`", parse_mode="Markdown")

async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in USER_PORTFOLIOS or not USER_PORTFOLIOS[chat_id]:
        await update.message.reply_text("📂 Danh mục trống.")
        return
    text = "📂 **DANH MỤC ĐẦU TƯ CỦA BẠN**\n\n"
    for symbol, pos in USER_PORTFOLIOS[chat_id].items():
        text += f"🔹 **{symbol}** ({pos.get('qty', 1000)} cổ) | Vốn: `{pos['entry']}k`\n"
    await update.message.reply_text(text, parse_mode="Markdown")

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args: return
    target = context.args[0].upper()
    if chat_id in USER_PORTFOLIOS:
        if target == "ALL": USER_PORTFOLIOS[chat_id] = {}
        elif target in USER_PORTFOLIOS[chat_id]: del USER_PORTFOLIOS[chat_id][target]
        save_portfolios()
        await update.message.reply_text(f"🗑️ Đã xóa {target}.")

async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args: return
    report = run_strategy_backtest(context.args[0].upper())
    await update.message.reply_text(report, parse_mode="Markdown")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().upper()
    if len(text) < 3 or len(text) > 5 or not text.isalpha(): return
    keyboard = [[
        InlineKeyboardButton("⏱️ 1H", callback_data=f"{text}|60|450"),
        InlineKeyboardButton("📅 1D", callback_data=f"{text}|D|300"),
        InlineKeyboardButton("📈 1W", callback_data=f"{text}|W|250"),
    ]]
    await update.message.reply_text(f"🔍 Chọn khung thời gian cho **{text}**:", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data == "run_autobuy_callback": await autobuy_command(update, context); return
    if data == "run_breakout_callback": await breakout_command(update, context); return
    if data == "run_testcung_callback": await testcung_command(update, context); return
    if data == "run_demand_callback": await demand_command(update, context); return
    if data == "run_sectors_callback": await sectors_command(update, context); return

    parts = data.split("|")
    if len(parts) == 3:
        symbol, resolution, count = parts[0], parts[1], int(parts[2])
        text_part1, text_part2, img_buf, reply_markup, err = analyze_stock(symbol, resolution, count)
        if not err:
            await query.message.reply_photo(photo=img_buf)
            await query.message.reply_text(text_part1, parse_mode="Markdown")
            await query.message.reply_text(text_part2, reply_markup=reply_markup, parse_mode="Markdown")

# ==========================================
# 6. KHỞI CHẠY BOT & HTTP SERVER
# ==========================================
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Quant Pro Bot with Golden Rules is running!")

def run_http_server():
    server = HTTPServer(("0.0.0.0", int(os.environ.get("PORT", 8080))), SimpleHTTPRequestHandler)
    server.serve_forever()

if __name__ == "__main__":
    threading.Thread(target=run_http_server, daemon=True).start()
    print("🤖 Đang khởi tạo Telegram Bot Pro...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    threading.Thread(target=background_scheduler_thread, args=(app,), daemon=True).start()

    # Đăng ký handler theo đúng thứ tự Bộ Quy Tắc Vàng
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("sectors", sectors_command))   # Bước 1: Ngành
    app.add_handler(CommandHandler("scan", scan_command))         # Bước 1: Nền
    app.add_handler(CommandHandler("testcung", testcung_command))   # Bước 2: Lệnh 1 (30%)
    app.add_handler(CommandHandler("demand", demand_command))     # Bước 2: Lệnh 2 (70%)
    app.add_handler(CommandHandler("breakout", breakout_command)) 
    app.add_handler(CommandHandler("autobuy", autobuy_command))   # Lọc Điểm Mua Chuẩn
    app.add_handler(CommandHandler("shakeout", shakeout_command))
    app.add_handler(CommandHandler("safety", safety_command))
    app.add_handler(CommandHandler("backtest", backtest_command))
    app.add_handler(CommandHandler("setbuy", setbuy_command))
    app.add_handler(CommandHandler("setstop", setstop_command))
    app.add_handler(CommandHandler("trail", trail_command))
    app.add_handler(CommandHandler("portfolio", portfolio_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    print("🚀 Bot Quant Trading Pro đã cập nhật thành công!")
    app.run_polling(drop_pending_updates=True)

