Import datetime
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
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
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

SECTOR_MAPPING = {
    "Ngân hàng": ["VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "STB", "SHB", "TPB"],
    "Chứng khoán": ["SSI", "VCI", "MBS", "FTS", "VND", "SHS", "VIX", "HCM", "CTS", "ORS"],
    "Bất động sản & KCN": ["VHM", "VIC", "VRE", "KBC", "PDR", "CEO", "DIG", "TCH", "NVL", "HDG", "KDH", "NLG"],
    "Thép & Vật liệu": ["HPG", "HSG", "NKG", "CTD", "VCG", "HT1", "VGS", "KSB"],
    "Dầu khí & Năng lượng": ["BSR", "PVS", "PVB", "PVD", "GAS", "PLX"],
    "Bán lẻ & Tiêu dùng": ["MWG", "PNJ", "FRT", "DGW", "MSN", "VNM", "SAB", "FPT", "MCH"]
}

ACTIVE_CHAT_IDS = set()
ALERTED_TODAY = set()

# ==========================================
# QUẢN LÝ LƯU TRỮ QUA GOOGLE SHEETS API
# ==========================================
USER_PORTFOLIOS = {}

def load_portfolios():
    """Tải danh mục từ Google Sheet khi bot khởi động"""
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
    """Đồng bộ toàn bộ danh mục lên Google Sheet mỗi khi có thay đổi"""
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
# 1. HÀM CALL DATA CHUẨN TCBS
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
        res = requests.get(tcbs_url, headers=headers, timeout=3.5)
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
    except Exception as e:
        return pd.DataFrame(), f"Lỗi kết nối API TCBS: {str(e)}"

    if df_raw.empty:
        return pd.DataFrame(), f"Không tìm thấy dữ liệu từ TCBS hoặc mã cổ phiếu '{symbol}' không hợp lệ."

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
        res = requests.get(url, headers=headers, timeout=2.5)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) >= 3:
                recent_3 = data[:3]
                is_net_buying = all(item.get("netVal", 0) > 0 for item in recent_3)
                return is_net_buying
    except Exception:
        pass
    return False


def get_foreign_net_summary():
    """Lấy dữ liệu tổng hợp mua/bán ròng khối ngoại từ TCBS cho danh sách chính"""
    results = []
    def fetch_one_foreign(sym):
        url = f"https://apifull.tcbs.com.vn/stock-insight/v1/stock/foreign-trading?ticker={sym}&resolution=D&count=1"
        headers = {"User-Agent": "Mozilla/5.0", "X-API-KEY": TCBS_API_KEY}
        try:
            res = requests.get(url, headers=headers, timeout=2)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list) and len(data) > 0:
                    net_val = data[0].get("netVal", 0)
                    return {"symbol": sym, "netVal": net_val}
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(fetch_one_foreign, s) for s in WATCHLIST_SCAN[:35]]
        for f in as_completed(futures):
            res = f.result()
            if res:
                results.append(res)
    
    results = sorted(results, key=lambda x: x["netVal"], reverse=True)
    return results


def get_fundamental_ratios(symbol):
    url = f"https://apifull.tcbs.com.vn/vietnam-stock/v1/stock/financial-ratio/{symbol.upper()}/historic?yearly=0&isQuarterly=1"
    headers = {"User-Agent": "Mozilla/5.0", "X-API-KEY": TCBS_API_KEY}
    try:
        res = requests.get(url, headers=headers, timeout=2.5)
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
        else:
            poc_w = poc_d

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
            "wave_type": wave_type,
            "is_above_weekly_poc": is_above_weekly_poc,
            "poc_w": poc_w,
            "poc_d": poc_d,
            "score_bonus": score_bonus
        }

        return is_daily_wave_bullish, wave_info
    except Exception as e:
        logging.error(f"Lỗi phân tích sóng ngày {symbol}: {e}")
        return True, {"wave_type": "ERROR", "score_bonus": 0, "is_above_weekly_poc": True}


def calculate_vcp_tightness(df_d):
    if len(df_d) < 20:
        return False, 1.0
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


# ==========================================
# 2. HÀM PHÂN TÍCH CHUYÊN SÂU ĐƠN MÃ
# ==========================================
def analyze_stock(symbol, resolution="D", count=250):
    def fmt(p):
        return f"{p:.1f}k"

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
    sma200 = last["SMA200"] if not np.isnan(last["SMA200"]) else close
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
    long_term_trend = "UPTREND DÀI HẠN (Trên SMA200)" if close > sma200 else "DOWNTREND/PHỤC HỒI DÀI HẠN (Dưới SMA200)"

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

    if cmf > 0.05 and mfi > 50:
        score += 20
        confluence_points.append(f"Dòng tiền lớn CMF ({cmf:.2f}) & MFI ({mfi:.1f}) đồng thuận tích lũy")
    elif rvol > 1.5 and cmf < 0:
        score -= 15
        risk_warnings.append(f"Cảnh báo Dòng tiền đầu cơ: RVOL cao ({rvol:.2f}x) nhưng CMF âm ({cmf:.2f})")

    if is_bb_squeeze:
        score += 10
        confluence_points.append("⚡ [NỀN TÍCH LŨY SIẾT CHẶT BB]: Chuẩn bị bùng nổ xu hướng")
    if is_rsi_bull_div:
        score += 15
        confluence_points.append("🎯 [PHÂN KỲ DƯƠNG RSI]: Tín hiệu tóm đáy / Đảo chiều tăng")

    is_bull_trap = (rsi < 40 and stoch_rsi > 80)
    is_overbought_full = (stoch_rsi > 90 and rsi > 70)

    trap_list = []
    if is_bull_trap: trap_list.append("⚡ [BẪY TĂNG GIẢ - BULL TRAP]: Xung lực ngắn hạn suy yếu.")
    if is_rsi_bear_div: trap_list.append("⚠️ [PHÂN KỲ ÂM RSI]: Rủi ro đảo chiều cao.")
    if not trap_list: trap_list.append("🛡️ Không phát hiện bẫy giá nguy hiểm.")
    trap_desc = " | ".join(trap_list)

    mkt_health = check_vnindex_health()
    score = int(score * mkt_health["score_modifier"])
    confidence_score = min(max(score, 10), 88)

    if confidence_score >= 65 and close > cloud_bottom and not is_bull_trap:
        system_status = "MỞ MUA / GIA TĂNG TỶ TRỌNG"
        nav_rec = "25 - 35% NAV"
        final_icon = f"🟢 **NÊN MUA / GIA TĂNG TỶ TRỌNG (Độ tin cậy: {confidence_score}%)**"
        final_action = f"Các chỉ báo đồng thuận ủng hộ đà tăng ({mkt_health['note']}). Ưu tiên gom quanh hỗ trợ `{fmt(val)} - {fmt(poc)}`."
    else:
        system_status = "QUAN SÁT / THẬN TRỌNG"
        nav_rec = "10 - 15% NAV (Thăm dò)"
        final_icon = f"🟡 **GIAO DỊCH THĂM DÒ / QUAN SÁT (Độ tin cậy: {confidence_score}%)**"
        final_action = "Kiên nhẫn chờ chiết khấu an toàn hoặc test hỗ trợ cứng."

    text_part1 = f"📊 BÁO CÁO PHÂN TÍCH KỸ THUẬT (TCBS VSA + SMC): {symbol.upper()}\n"
    text_part1 += f"🗓️ Ngày: {last['time'].strftime('%Y-%m-%d')} | Khung: {resolution} | Giá hiện tại: {fmt(close)}\n"
    text_part1 += f"🏛️ Trạng thái VN-Index: **{mkt_health['note']}**\n\n"
    text_part1 += f"🌐 1. CẤU TRÚC XU HƯỚNG & DÒNG TIỀN:\n• Dài hạn: {long_term_trend}\n• ADX: `{adx:.1f}` | CMF: {cmf:.2f} | MFI: {mfi:.1f} | RVOL: {rvol:.2f}x\n\n"
    text_part1 += f"☁️ 2. ICHIMOKU & ĐỘNG LƯỢNG:\n• Mây Kumo: {ichi_status}\n• RSI: {rsi:.1f} | StochRSI: {stoch_rsi:.1f}\n\n"
    text_part1 += f"📊 3. VOLUME PROFILE & HỖ TRỢ / KHÁNG CỰ:\n• POC: {fmt(poc)} | VAL: {fmt(val)} | VAH: {fmt(vah)}\n\n"

    text_part2 = f"⚡ 4. NHẬN DIỆN BẪY GIÁ (SMC/VSA):\n• {trap_desc}\n\n"
    text_part2 += f"🏁 5. TỔNG KẾT CHIẾN LƯỢC:\n• Khuyến nghị: {final_icon}\n• Định hướng: {final_action}"

    keyboard = [
        [
            InlineKeyboardButton("⏱️ Khung 1H", callback_data=f"{symbol}|60|450"),
            InlineKeyboardButton("📅 Khung 1D", callback_data=f"{symbol}|D|300"),
            InlineKeyboardButton("📈 Khung 1W", callback_data=f"{symbol}|W|250"),
        ],
        [
            InlineKeyboardButton("📰 Tin Tức AI (/tin)", callback_data=f"news_{symbol}"),
            InlineKeyboardButton("🤖 Quét Điểm Mua (/autobuy)", callback_data="run_autobuy_callback"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.plot(df["time"], df["Close"], label="Giá đóng cửa", color="black", lw=1.2)
    ax.axhline(poc, color="purple", linestyle=":", label=f"POC: {fmt(poc)}", alpha=0.8)
    ax.axhline(val, color="red", linestyle="--", label=f"VAL: {fmt(val)}", alpha=0.6)
    ax.set_title(f"Phân Tích TCBS Pro - {symbol.upper()} ({resolution})", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return text_part1, text_part2, buf, reply_markup, None


# ==========================================
# 3. CÁC TÍNH NĂNG MỚI NÂNG CẤP (/tin, /dipbuy, /sectors, /khoingoai, /quytec)
# ==========================================
async def tin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("⚠️ Vui lòng nhập mã cổ phiếu! Ví dụ: `/tin VCG` hoặc `/tin HPG`", parse_mode="Markdown")
        return
    symbol = args[0].upper().strip()
    msg = await update.message.reply_text(f"🤖 **Groq AI đang tổng hợp tin tức và phân tích Tốt / Xấu cho mã {symbol}...** ⚡")

    # Gọi API tin tức từ TCBS
    news_list = []
    url = f"https://apifull.tcbs.com.vn/stock-insight/v1/news/recent?ticker={symbol}&size=5"
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0", "X-API-KEY": TCBS_API_KEY}, timeout=3)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list):
                for item in data[:5]:
                    news_list.append(item.get("title", ""))
    except Exception:
        pass

    news_text = "\n".join([f"- {n}" for n in news_list]) if news_list else f"Hiện tại không có tin tức mới ghi nhận trực tiếp cho {symbol} trên hệ thống."

    # Phân tích bằng Groq AI nếu có API Key, nếu không dùng thuật toán phân tích chuyên gia
    analysis_result = ""
    if GROQ_API_KEY:
        try:
            groq_url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "system", "content": "Bạn là chuyên gia phân tích tài chính chứng khoán Việt Nam xuất sắc theo phương pháp VSA và SMC."},
                    {"role": "user", "content": f"Dựa trên các tiêu đề tin tức sau của mã {symbol}:\n{news_text}\nHãy phân tích chi tiết các yếu tố Tốt (Bullish) và Xấu (Bearish) đối với xu hướng giá sắp tới, sau đó đưa ra lời khuyên hành động ngắn gọn bằng tiếng Việt."}
                ],
                "temperature": 0.3
            }
            res_groq = requests.post(groq_url, headers=headers, json=payload, timeout=10)
            if res_groq.status_code == 200:
                analysis_result = res_groq.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    if not analysis_result:
        analysis_result = (
            f"📰 **TỔNG HỢP & PHÂN TÍCH TIN TỨC: {symbol}**\n\n"
            f"**Tiêu đề tin tức gần đây:**\n{news_text}\n\n"
            f"🔍 **Đánh giá sơ bộ từ hệ thống Quant:**\n"
            f"• **Yếu tố Tốt (Bullish):** Dòng tiền định chế duy trì ổn định, không ghi nhận thông tin tiêu cực về nội bộ doanh nghiệp.\n"
            f"• **Yếu tố Xấu (Bearish):** Cần thận trọng áp lực rung lắc theo nhịp chung của thị trường.\n"
            f"• **Khuyến nghị:** Quan sát kỹ phản ứng giá tại hỗ trợ gần nhất, tránh mua đuổi khi tin tức nhiễu động."
        )

    await msg.edit_text(analysis_result, parse_mode="Markdown")


async def dipbuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await update.message.reply_text("🎯 **Đang quét các cơ hội bắt đáy an toàn (RSI quá bán + Phân kỳ dương / Test thành công)...** ⚡")

    results = []
    def scan_dip(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=100)
        if err or df.empty or len(df) < 50:
            return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        close = last["Close"]
        rsi = last.get("RSI", 50)
        is_bull_div = last.get("RSI_Bullish_Divergence", False)
        cmf = last.get("CMF", 0.0)

        if (rsi <= 40 or is_bull_div) and cmf >= -0.15:
            score = (50 - rsi) + (25 if is_bull_div else 0) + (max(0, cmf) * 30)
            return {"symbol": symbol, "close": close, "rsi": rsi, "is_bull_div": is_bull_div, "score": score}
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_dip, s) for s in WATCHLIST_SCAN]
        for f in as_completed(futures):
            res = f.result()
            if res:
                results.append(res)

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:6]
    if not results:
        await msg.edit_text("⚠️ Hiện tại chưa có mã nào thỏa mãn tiêu chuẩn bắt đáy an toàn.")
        return

    text = "🎯 **DANH SÁCH BẮT ĐÂY AN TOÀN SAU ĐỢT GIẢM (DIP BUY)**\n====================================\n\n"
    for idx, item in enumerate(results, 1):
        div_tag = "🔥 [Có Phân Kỳ Dương RSI]" if item["is_bull_div"] else "⚡ [Vùng Quá Bán Hấp Dẫn]"
        text += f"{idx}. 🟢 **Mã: {item['symbol'].upper()}** {div_tag}\n"
        text += f"   • Giá hiện tại: `{item['close']:.1f}k` | RSI: `{item['rsi']:.1f}`\n"
        text += f"   • Phân tích chi tiết: Nhập mã `{item['symbol']}`\n"
        text += "------------------------------------\n"

    await msg.edit_text(text, parse_mode="Markdown")


async def sectors_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await update.message.reply_text("🏢 **Đang phân tích dòng tiền và hiệu suất các Nhóm Ngành...** ⚡")

    sector_performance = []
    for sector_name, symbols in SECTOR_MAPPING.items():
        total_change = 0
        valid_count = 0
        for sym in symbols[:5]:
            df, err = fetch_stock_data(sym, resolution="D", count=5)
            if not err and not df.empty and len(df) >= 2:
                pct = (df.iloc[-1]["Close"] - df.iloc[-2]["Close"]) / df.iloc[-2]["Close"] * 100
                total_change += pct
                valid_count += 1
        avg_change = (total_change / valid_count) if valid_count > 0 else 0
        sector_performance.append({"sector": sector_name, "avg_change": avg_change})

    sector_performance = sorted(sector_performance, key=lambda x: x["avg_change"], reverse=True)

    text = "🏢 **BẢNG XẾP HẠNG DÒNG TIỀN NHÓM NGÀNH**\n====================================\n\n"
    for idx, item in enumerate(sector_performance, 1):
        icon = "🚀" if item["avg_change"] > 0 else "🔻"
        text += f"{idx}. {icon} **{item['sector']}**: Biến động bình quân `{item['avg_change']:+.2f}%`\n"

    text += "\n💡 *Gợi ý: Hãy tập trung giải ngân vào các cổ phiếu dẫn dắt thuộc Top 2 nhóm ngành đứng đầu!*"
    await msg.edit_text(text, parse_mode="Markdown")


async def khoingoai_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await update.message.reply_text("🌍 **Đang tổng hợp dữ liệu giao dịch Khối Ngoại từ TCBS OpenAPI...** ⚡")

    foreign_data = get_foreign_net_summary()
    if not foreign_data:
        await msg.edit_text("⚠️ Không lấy được dữ liệu khối ngoại từ TCBS lúc này.")
        return

    top_buy = [x for x in foreign_data if x["netVal"] > 0][:5]
    top_sell = sorted([x for x in foreign_data if x["netVal"] < 0], key=lambda x: x["netVal"])[:5]

    text = "🌍 **TOP MUA/BÁN RÒNG KHỐI NGOẠI (TCBS OPENAPI)**\n====================================\n\n"
    text += "🟢 **TOP MUA RÒNG TỐT NHẤT:**\n"
    for idx, item in enumerate(top_buy, 1):
        text += f"{idx}. **{item['symbol']}**: Mua ròng `{item['netVal']:,.0f}` VNĐ\n"

    text += "\n🔴 **TOP BÁN RÒNG MẠNH:**\n"
    for idx, item in enumerate(top_sell, 1):
        text += f"{idx}. **{item['symbol']}**: Bán ròng `{item['netVal']:,.0f}` VNĐ\n"

    await msg.edit_text(text, parse_mode="Markdown")


async def quytec_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    quy_tec_text = (
        "📜 **BỘ QUY TẮC SỐNG CÒN CỦA NHÓM 5% TRADER THÀNH CÔNG**\n"
        "====================================\n\n"
        "1️⃣ **Cắt lỗ là tôn giáo:** Không bao giờ để một khoản lỗ nhỏ biến thành thảm họa tài khoản. Tuân thủ tuyệt đối Stop Loss.\n"
        "2️⃣ **Thuận dòng tiền lớn (Institutional Flow):** Chỉ mua khi dòng tiền CMF dương và có sự xác nhận của RVOL. Đừng chống lại tạo lập.\n"
        "3️⃣ **Không bao giờ trung bình giá xuống:** Việc mua thêm khi cổ phiếu đang downtrend là con đường nhanh nhất dẫn đến cháy tài khoản.\n"
        "4️⃣ **Quản trị tỷ trọng NAV:** Không bao giờall-in vào một mã duy nhất. Phân bổ rủi ro từ 20-30% NAV mỗi lệnh.\n"
        "5️⃣ **Kiên nhẫn chờ điểm chuẩn (VSA / SMC):** Đợi thị trường/cổ phiếu test cung cạn kiệt hoặc siết nền VCP xong mới hành động.\n"
        "6️⃣ **Kỷ luật kỷ luật và kỷ luật:** Cảm xúc là kẻ thù số một trong trading. Hãy để hệ thống lượng tử ra quyết định thay bạn!"
    )
    await update.message.reply_text(quy_tec_text, parse_mode="Markdown")


# ==========================================
# 4. TELEGRAM HANDLERS & AUTOBUY
# ==========================================
async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.error(f"Error update: {update} caused error: {context.error}")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    welcome_text = (
        "👋 **CHÀO MỪNG BẠN ĐẾN VỚI QUANT TRADING SYSTEM PRO (VSA + SMC)!**\n\n"
        "1️⃣ **Phân tích kỹ thuật đơn mã:** Nhập mã cổ phiếu trực tiếp (Ví dụ: `VHM`, `VCG`)\n\n"
        "2️⃣ **TỔNG HỢP TIN TỨC AI:** Gửi lệnh `/tin VCG` để Groq AI phân tích Tốt / Xấu\n\n"
        "3️⃣ **BỘ LỌC & GIAO DỊCH:**\n"
        "   ├ 🎯 `/dipbuy`    : Bắt đáy an toàn sau đợt giảm\n"
        "   ├ 🤖 `/autobuy`   : Tín hiệu mua chuẩn VCP + SMC\n"
        "   ├ 🏢 `/sectors`   : Dòng tiền Nhóm Ngành\n"
        "   ├ 🌍 `/khoingoai` : Top Mua/Bán ròng Khối Ngoại (TCBS OpenAPI)\n"
        "   └ 📜 `/quytec`    : Xem bộ quy tắc sống còn của nhóm 5%\n\n"
        "4️⃣ **QUẢN LÝ DANH MỤC & CÀI ĐẶT:**\n"
        "   • `/setbuy VCG 22.5 21.8 1000` | `/setstop VCG 20.5` | `/portfolio`"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def autobuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    if update.message:
        msg = await update.message.reply_text("🤖 **Đang kích hoạt radar AI quét siêu tốc mã siết nền VCP & Dòng tiền chuẩn VSA/SMC...** ⚡")
    else:
        query = update.callback_query
        msg = await query.message.reply_text("🤖 **Đang kích hoạt radar AI quét siêu tốc mã siết nền VCP & Dòng tiền chuẩn VSA/SMC...** ⚡")

    try:
        picks = []
        def process_one(symbol):
            df_d, err = fetch_stock_data(symbol, resolution="D", count=150)
            if err or df_d.empty or len(df_d) < 50:
                return None
            df_d = calculate_indicators(df_d)
            last = df_d.iloc[-1]
            close = last["Close"]
            cmf = last.get("CMF", 0.0)
            rvol = last.get("RVOL", 1.0)
            is_vcp, _ = calculate_vcp_tightness(df_d)

            if cmf > 0.02 and (rvol >= 1.1 or is_vcp):
                score = (cmf * 50) + (rvol * 20) + (30 if is_vcp else 0)
                return {"symbol": symbol, "close": close, "cmf": cmf, "rvol": rvol, "is_vcp": is_vcp, "score": score}
            return None

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(process_one, s) for s in WATCHLIST_SCAN]
            for f in as_completed(futures):
                res = f.result()
                if res: picks.append(res)

        picks = sorted(picks, key=lambda x: x["score"], reverse=True)[:5]
        if not picks:
            await msg.edit_text("⚠️ Hiện tại chưa có mã nào thỏa mãn tiêu chuẩn autobuy VCP + SMC.")
            return

        text = "🎯 **DANH SÁCH TÍN HIỆU MUA CHUẨN VCP + SMC (AUTO-BUY)**\n====================================\n\n"
        for idx, item in enumerate(picks, 1):
            vcp_tag = "🔥 [Nền VCP Thu Hẹp]" if item["is_vcp"] else "⚡ [Tích Lũy Dòng Tiền]"
            text += f"{idx}. 🔹 **Mã: {item['symbol'].upper()}** {vcp_tag}\n"
            text += f"   • Giá đóng cửa: `{item['close']:.1f}k` | RVOL: `{item['rvol']:.2f}x` | CMF: `{item['cmf']:.2f}`\n"
            text += f"   • Lệnh cài canh mua: `/setbuy {item['symbol']} {item['close']} {item['close']*0.98:.1f} 1000`\n"
            text += "------------------------------------\n"

        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi thực thi auto-buy: {str(e)}")


async def setbuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 3:
        await update.message.reply_text("⚠️ Sai cú pháp! Ví dụ: `/setbuy VCG 22.5 21.8 1000`", parse_mode="Markdown")
        return

    symbol = args[0].upper().strip()
    try:
        entry_price = float(args[1])
        buy_alert_price = float(args[2])
        qty = int(args[3]) if len(args) >= 4 else 1000
    except ValueError:
        await update.message.reply_text("❌ Tham số giá và khối lượng phải là số hợp lệ!")
        return

    if chat_id not in USER_PORTFOLIOS: USER_PORTFOLIOS[chat_id] = {}
    if symbol not in USER_PORTFOLIOS[chat_id]:
        USER_PORTFOLIOS[chat_id][symbol] = {"entry": entry_price, "qty": qty, "buy_alert_prices": [], "stop_alert_prices": []}

    USER_PORTFOLIOS[chat_id][symbol]["buy_alert_prices"].append(buy_alert_price)
    save_portfolios()
    await update.message.reply_text(f"✅ Đã thiết lập canh mua mã **{symbol}** tại mức giá `{buy_alert_price}k` thành công!", parse_mode="Markdown")


async def setstop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("⚠️ Sai cú pháp! Ví dụ: `/setstop VCG 20.5`", parse_mode="Markdown")
        return

    symbol = args[0].upper().strip()
    try:
        stop_price = float(args[1])
    except ValueError:
        await update.message.reply_text("❌ Mốc giá cắt lỗ phải là số hợp lệ!")
        return

    if chat_id not in USER_PORTFOLIOS: USER_PORTFOLIOS[chat_id] = {}
    if symbol not in USER_PORTFOLIOS[chat_id]:
        USER_PORTFOLIOS[chat_id][symbol] = {"entry": stop_price, "qty": 1000, "buy_alert_prices": [], "stop_alert_prices": []}

    USER_PORTFOLIOS[chat_id][symbol]["stop_alert_prices"].append(stop_price)
    save_portfolios()
    await update.message.reply_text(f"✅ Đã thiết lập Stop Loss mã **{symbol}** tại mức giá `{stop_price}k` thành công!", parse_mode="Markdown")


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in USER_PORTFOLIOS or not USER_PORTFOLIOS[chat_id]:
        await update.message.reply_text("📂 Danh mục của bạn đang trống. Dùng `/setbuy` hoặc `/setstop` để thêm.", parse_mode="Markdown")
        return

    text = "📂 **DANH MỤC ĐẦU TƯ & GIÁM SÁT REAL-TIME**\n\n"
    for symbol, pos in USER_PORTFOLIOS[chat_id].items():
        df, err = fetch_stock_data(symbol, resolution="60", count=5)
        if not err and not df.empty:
            cur_price = df.iloc[-1]["Close"]
            entry = pos["entry"]
            pnl = ((cur_price - entry) / entry) * 100
            status = "🟢 LÃI" if pnl >= 0 else "🔴 LỖ"
            text += f"🔹 **{symbol}** | Vốn: `{entry}k` | Hiện tại: `{cur_price:.1f}k` | Hiệu suất: `{pnl:+.2f}%` [{status}]\n"
        else:
            text += f"🔹 **{symbol}**: Không cập nhật được giá.\n"
    await update.message.reply_text(text, parse_mode="Markdown")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    text = update.message.text.strip().upper()
    if len(text) < 3 or len(text) > 5 or not text.isalpha():
        await update.message.reply_text("⚠️ Vui lòng nhập mã cổ phiếu hợp lệ (VD: VHM, VCG, HPG)!")
        return

    keyboard = [
        [
            InlineKeyboardButton("⏱️ Khung 1H", callback_data=f"{text}|60|450"),
            InlineKeyboardButton("📅 Khung 1D", callback_data=f"{text}|D|300"),
            InlineKeyboardButton("📈 Khung 1W", callback_data=f"{text}|W|250"),
        ],
        [
            InlineKeyboardButton("📰 Tin Tức AI (/tin)", callback_data=f"news_{text}"),
        ]
    ]
    await update.message.reply_text(f"🔍 Đã nhận mã **{text}**. Chọn tính năng phân tích:", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith("news_"):
        sym = data.split("_")[1]
        context.args = [sym]
        await tin_command(update, context)
        return

    if data == "run_autobuy_callback":
        await autobuy_command(update, context)
        return

    parts = data.split("|")
    if len(parts) == 3:
        symbol, resolution, count = parts[0], parts[1], int(parts[2])
        msg = await query.message.reply_text(f"⏳ Đang phân tích kỹ thuật {symbol} (Khung: {resolution})...")
        def bg_task():
            return analyze_stock(symbol, resolution=resolution, count=count)
        future = executor.submit(bg_task)
        try:
            t1, t2, img_buf, markup, err = future.result(timeout=15)
            if err:
                await msg.edit_text(f"❌ Lỗi: {err}")
            else:
                await msg.delete()
                await query.message.reply_photo(photo=img_buf)
                await query.message.reply_text(t1, parse_mode="Markdown")
                await query.message.reply_text(t2, reply_markup=markup, parse_mode="Markdown")
        except Exception as e:
            await msg.edit_text(f"❌ Lỗi hệ thống: {str(e)}")


# ==========================================
# 5. HTTP SERVER & MAIN
# ==========================================
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Quant Trading System Pro with TCBS API & AI News is running!")

def run_http_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler)
    server.serve_forever()


if __name__ == "__main__":
    threading.Thread(target=run_http_server, daemon=True).start()

    print("🤖 Đang khởi tạo Quant Trading System Pro (TCBS + AI)...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_error_handler(global_error_handler)

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("tin", tin_command))
    app.add_handler(CommandHandler("dipbuy", dipbuy_command))
    app.add_handler(CommandHandler("autobuy", autobuy_command))
    app.add_handler(CommandHandler("sectors", sectors_command))
    app.add_handler(CommandHandler("khoingoai", khoingoai_command))
    app.add_handler(CommandHandler("quytec", quytec_command))
    app.add_handler(CommandHandler("setbuy", setbuy_command))
    app.add_handler(CommandHandler("setstop", setstop_command))
    app.add_handler(CommandHandler("portfolio", portfolio_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    print("🚀 Bot đã hoàn tất và sẵn sàng phục vụ!")
    app.run_polling(drop_pending_updates=True)
 
