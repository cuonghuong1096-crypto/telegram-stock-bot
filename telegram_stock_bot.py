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
                rows.append([str(chat_id), symbol, pos.get("entry", 0), pos.get("qty", 1000), json.dumps(alert_dict)])
        payload = {"action": "save", "rows": rows}
        requests.post(GOOGLE_SHEET_API_URL, json=payload, timeout=10)
    except Exception as e:
        logging.error(f"Lỗi ghi portfolio lên Google Sheet: {e}")

USER_PORTFOLIOS = load_portfolios()
executor = ThreadPoolExecutor(max_workers=16)

# ==========================================
# 1. HÀM CALL DATA & TÍNH TOÁN TỐI ƯU
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
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "X-API-KEY": TCBS_API_KEY}
        res = requests.get(tcbs_url, headers=headers, timeout=2.5)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                df_raw = pd.DataFrame(data)
                df_raw["time"] = pd.to_datetime(df_raw["tradingDate"])
                df_raw.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    except Exception:
        pass

    if df_raw.empty:
        url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution={res_param}&symbol={symbol}&from={start_time}&to={end_time}"
        try:
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://dchart.vndirect.com.vn/"}
            res = requests.get(url, headers=headers, timeout=2.5)
            if res.status_code == 200:
                data = res.json()
                if data.get("s") == "ok" and "t" in data and len(data["t"]) > 0:
                    df_raw = pd.DataFrame({
                        "time": pd.to_datetime(data["t"], unit="s"),
                        "Open": data["o"], "High": data["h"], "Low": data["l"],
                        "Close": data["c"], "Volume": data["v"]
                    })
        except Exception as e:
            return pd.DataFrame(), f"Lỗi kết nối API dự phòng: {str(e)}"

    if df_raw.empty:
        return pd.DataFrame(), f"Không tìm thấy dữ liệu hoặc mã cổ phiếu '{symbol}' không hợp lệ."

    df_raw = df_raw[["time", "Open", "High", "Low", "Close", "Volume"]].sort_values("time").reset_index(drop=True)

    if resolution == "W":
        df_raw.set_index("time", inplace=True)
        df_weekly = df_raw.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna().reset_index()
        return df_weekly.tail(250).reset_index(drop=True), None

    if resolution == "M":
        df_raw.set_index("time", inplace=True)
        df_monthly = df_raw.resample("ME").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna().reset_index()
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

    # Phân kỳ RSI cải tiến an toàn
    price_max_now = df["Close"].iloc[-1]
    price_max_prev = df["Close"].iloc[-20:-5].max() if len(df) >= 20 else price_max_now
    rsi_now = df["RSI"].iloc[-1]
    rsi_prev_max = df["RSI"].iloc[-20:-5].max() if len(df) >= 20 else rsi_now
    df["RSI_Bearish_Divergence"] = (price_max_now > price_max_prev) and (rsi_now < rsi_prev_max) and (rsi_now > 58)

    price_min_now = df["Close"].iloc[-1]
    price_min_prev = df["Close"].iloc[-20:-5].min() if len(df) >= 20 else price_min_now
    rsi_prev_min = df["RSI"].iloc[-20:-5].min() if len(df) >= 20 else rsi_now
    df["RSI_Bullish_Divergence"] = (price_min_now < price_min_prev) and (rsi_now > rsi_prev_min) and (rsi_now < 42)

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
        return {"status": "BEARISH_RISK", "score_modifier": 0.70, "note": "VN-Index rủi ro / Xu hướng yếu"}
    else:
        return {"status": "SIDEWAYS", "score_modifier": 0.95, "note": "VN-Index tích lũy/giằng co"}


def get_foreign_trading_flow(symbol):
    url = f"https://apifull.tcbs.com.vn/stock-insight/v1/stock/foreign-trading?ticker={symbol.upper()}&resolution=D&count=5"
    headers = {"User-Agent": "Mozilla/5.0", "X-API-KEY": TCBS_API_KEY}
    try:
        res = requests.get(url, headers=headers, timeout=2)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) >= 3:
                recent_3 = data[:3]
                return all(item.get("netVal", 0) > 0 for item in recent_3)
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
                latest = data[0]
                return {"pe": latest.get("priceToEarning", 10), "roe": latest.get("roe", 5)}
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
    tightness_score = max(0, (0.05 - vcp_3) * 100)
    return is_vcp, tightness_score


def calculate_relative_strength(df_stock, df_vnindex):
    if df_stock.empty or df_vnindex.empty or len(df_stock) < 15 or len(df_vnindex) < 15:
        return 1.0
    stock_perf = (df_stock["Close"].iloc[-1] - df_stock["Close"].iloc[-15]) / df_stock["Close"].iloc[-15]
    vni_perf = (df_vnindex["Close"].iloc[-1] - df_vnindex["Close"].iloc[-15]) / df_vnindex["Close"].iloc[-15]
    return stock_perf - vni_perf


def check_liquidity_and_base(df, last):
    """Bộ lọc thanh khoản chuẩn xác: Lọc bỏ mã < 5 tỷ/phiên và giá trị thanh khoản rác"""
    close = last["Close"]
    vol_sma20 = last["Vol_SMA20"] if not np.isnan(last.get("Vol_SMA20", np.nan)) else last["Volume"]
    avg_trading_value = vol_sma20 * close * 1000
    if avg_trading_value < 5_000_000_000:
        return False
    return True


# ==========================================
# 🎯 TẦNG 5 (ĐỈNH PHỄU): ĐIỂM MUA AUTOBUY TỐI ƯU HOÁ CHÍNH XÁC CAO
# ==========================================
def process_single_stock_for_autobuy(symbol):
    try:
        mkt_health = check_vnindex_health()
        df_vni, err_vni = fetch_stock_data("VNINDEX", resolution="D", count=60)
        if err_vni or df_vni.empty:
            df_vni, _ = fetch_stock_data("1000", resolution="D", count=60)

        ratios = get_fundamental_ratios(symbol)
        if ratios["pe"] > 35 or ratios["roe"] < 8.0:  # Chặn cổ phiếu cơ bản quá kém/định giá quá cao
            return None

        df_d, err_d = fetch_stock_data(symbol, resolution="D", count=200)
        df_h, err_h = fetch_stock_data(symbol, resolution="60", count=60)
        
        if err_d or df_d.empty or len(df_d) < 50:
            return None

        df_d = calculate_indicators(df_d)
        last_d = df_d.iloc[-1]

        if not check_liquidity_and_base(df_d, last_d):
            return None

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

        # Tính toán điểm vào lệnh thông minh bám sát hỗ trợ kỹ thuật VAL / Kijun
        entry_price = round(max(val, close * 0.98), 2)
        stop_loss = round(entry_price - (1.5 * atr), 2)
        target_price = vah if vah > (close * 1.03) else round(close * 1.08, 2)

        risk = entry_price - stop_loss
        reward = target_price - entry_price
        rr_ratio = reward / (risk + 1e-9)

        score = 50

        span_a = last_d.get("Senkou_Span_A", close)
        span_b = last_d.get("Senkou_Span_B", close)
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)

        # Bộ khung tính điểm xác suất tin cậy
        if close > cloud_top:
            score += 20
        elif close >= cloud_bottom:
            score += 8
        else:
            score -= 30

        if cmf_d > 0.03 and cmf_h > 0.01:
            score += 20
        elif cmf_d > 0:
            score += 10
        elif cmf_h < -0.10:
            score -= 20

        if rs_rating > 0.03:
            score += 15
        if is_vcp:
            score += 15

        if rvol_d >= 1.15:
            score += 10
        if macd_hist_d > 0:
            score += 10
        if is_bb_squeeze:
            score += 10

        if get_foreign_trading_flow(symbol):
            score += 10

        if rr_ratio >= 2.0:
            score += 15
        elif rr_ratio < 1.2:
            score -= 25

        is_bull_trap = (rsi_d < 40 and stoch_rsi_d > 80)
        is_overbought_bb = close > (bb_upper * 1.008)

        if is_bull_trap: score -= 35
        if is_rsi_bear_div: score -= 30
        if is_overbought_bb: score -= 20

        score = int(score * mkt_health["score_modifier"])

        atr_pct = (atr / close) * 100
        if atr_pct <= 2.5 and is_vcp:
            suggested_nav = "30 - 35% NAV (Nền siết rất chặt - An toàn cao)"
        elif atr_pct <= 4.0:
            suggested_nav = "20 - 25% NAV (Tỷ trọng chuẩn)"
        else:
            suggested_nav = "10 - 15% NAV (Biến động rộng - Thăm dò)"

        # Điều kiện khắt khe để lọt vào danh sách Autobuy chuẩn xác nhất
        if (score >= 70 and 
            cmf_d > -0.03 and 
            cmf_h > -0.03 and 
            close >= cloud_bottom and 
            not is_bull_trap and 
            not is_rsi_bear_div and 
            not is_overbought_bb and 
            rr_ratio >= 1.5):

            return {
                "symbol": symbol,
                "close": close,
                "score": min(score, 99),
                "cmf": cmf_d,
                "cmf_h": cmf_h,
                "rvol": rvol_d,
                "rs_rating": round(rs_rating * 100, 1),
                "is_vcp": is_vcp,
                "rr_ratio": round(rr_ratio, 2),
                "entry": entry_price,
                "stop_loss": stop_loss,
                "target": target_price,
                "suggested_nav": suggested_nav
            }
    except Exception as e:
        pass
    return None


def run_smart_buy_screener():
    buy_recommendations = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(process_single_stock_for_autobuy, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            result = future.result()
            if result:
                buy_recommendations.append(result)

    buy_recommendations = sorted(buy_recommendations, key=lambda x: x["score"], reverse=True)
    return buy_recommendations[:5]


# --- CÁC HÀM SCREENER BỔ TRỢ KHÁC ---
def run_market_screener():
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
        for f in as_completed(futures):
            res = f.result()
            if res: results.append(res)
    return sorted(results, key=lambda x: x["score"], reverse=True)[:8]


def run_test_supply_screener():
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
        for f in as_completed(futures):
            res = f.result()
            if res: results.append(res)
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]


def run_demand_confirmation_screener():
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
            return {"symbol": symbol, "close": close, "rvol": rvol, "cmf": cmf, "mfi": mfi, "score": score}
        return None
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_demand, symbol) for symbol in WATCHLIST_SCAN]
        for f in as_completed(futures):
            res = f.result()
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
        return {"symbol": symbol, "pct_change": pct_change, "cmf": last.get("CMF", 0.0), "rvol": last.get("RVOL", 1.0)}

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
        sector_score = (avg_cmf * 40) + (avg_rvol * 20) + (avg_pct * 10)
        leader_stock = sorted(stock_metrics, key=lambda x: x["cmf"] + x["rvol"], reverse=True)[0]["symbol"]
        sector_results.append({
            "sector": sector_name, "avg_pct": avg_pct, "avg_cmf": avg_cmf,
            "avg_rvol": avg_rvol, "leader": leader_stock, "score": sector_score
        })
    return sorted(sector_results, key=lambda x: x["score"], reverse=True)


# ==========================================
# 2. TELEGRAM HANDLERS & BOT CONTROLLERS
# ==========================================
async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.error(f"Error update: {update} caused error: {context.error}")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    welcome_text = (
        "👋 **CHÀO MỪNG ĐẾN VỚI QUANT TRADING SYSTEM (PRO OPTIMIZED)!**\n\n"
        "1️⃣ **Phân tích kỹ thuật đơn mã:** Nhập trực tiếp mã cổ phiếu (VD: `HPG`, `VHM`)\n\n"
        "2️⃣ **🎯 CÁC LỆNH PHỄU DÒNG TIỀN CHIẾN LƯỢC:**\n"
        "   ├ 🟢 `/scan` : Quét tích lũy nền giá & dòng tiền tổ chức\n"
        "   ├ 🔍 `/testcung`: Săn cổ phiếu cạn kiệt thanh khoản (No-Supply)\n"
        "   ├ 🚀 `/demand` : Xác nhận dòng tiền mua chủ động bùng nổ\n"
        "   ├ 🏢 `/sectors`: Bản đồ dòng tiền luân chuyển nhóm ngành & Leader\n"
        "   └ 🤖 `/autobuy`: **Phễu đỉnh cao (Tầng 5)** - Điểm giải ngân an toàn tuyệt đối\n\n"
        "3️⃣ **📂 QUẢN LÝ DANH MỤC & REAL-TIME:**\n"
        "   • `/setbuy <Mã> <Giá_vốn> <Giá_canh> <KL>`\n"
        "   • `/setstop <Mã> <Giá_cắt_lỗ>` | `/portfolio` | `/trail <Mã>`"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def autobuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await update.message.reply_text("🤖 **Hệ thống AI đang quét qua Phễu Tầng 5 (VCP + RS + R:R chuẩn chỉnh)...** ⚡")
    try:
        picks = run_smart_buy_screener()
        if not picks:
            await msg.edit_text("⚠️ Hiện tại thị trường không có mã nào đạt tiêu chuẩn khắt khe an toàn tối đa (R:R >= 1.5). Hãy kiên nhẫn chờ điểm mua đẹp.")
            return
            
        text = "🎯 **DANH SÁCH CỔ PHIẾU ĐẠT ĐIỂM MUA AN TOÀN CAO NHẤT (AUTO-BUY)**\n"
        text += "====================================\n\n"
        for idx, item in enumerate(picks, 1):
            vcp_tag = "🔥 [Nền VCP Thu Hẹp]" if item['is_vcp'] else "⚡ [Tích Lũy Chuẩn]"
            text += f"{idx}. 🔹 **Mã: {item['symbol'].upper()}** {vcp_tag} (Độ tin cậy: `{item['score']}%`)\n"
            text += f"   • Giá đóng cửa: `{item['close']:.1f}k` | Sức mạnh RS: `{item['rs_rating']:+.1f}%`\n"
            text += f"   • Tỷ lệ Reward/Risk: `{item['rr_ratio']}x` | Tỷ trọng NAV: `{item['suggested_nav']}`\n"
            text += f"   • 🎯 **Vùng giải ngân chuẩn**: `{item['entry']}k` | **Chốt lời mục tiêu**: `{item['target']}k`\n"
            text += f"   • 🛡️ **Cắt lỗ an toàn (ATR)**: `{item['stop_loss']}k`\n"
            text += f"   • Lệnh cài nhanh: `/setbuy {item['symbol']} {item['close']} {item['entry']} 1000`\n"
            text += "------------------------------------\n"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi thực thi bộ lọc: {str(e)}")


async def sectors_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await update.message.reply_text("🏢 **Đang đo lường dòng tiền thông minh theo từng nhóm ngành...** ⚡")
    try:
        sectors = run_sector_flow_screener()
        if not sectors:
            await msg.edit_text("⚠️ Không lấy được dữ liệu dòng tiền ngành.")
            return
        text = "🔥 **BẢN ĐỒ DÒNG TIỀN NHÓM NGÀNH (SECTOR FLOW)**\n"
        text += "====================================\n\n"
        for idx, sec in enumerate(sectors, 1):
            status = "🟢 [HÚT TIỀN MẠNH]" if sec['avg_cmf'] > 0.05 else ("🟡 [TÍCH LŨY]" if sec['avg_cmf'] > 0 else "🔴 [RÚT LUI]")
            text += f"{idx}. 🏢 **{sec['sector'].upper()}** {status}\n"
            text += f"   • Thay đổi giá TB: `{sec['avg_pct']:+.2f}%` | CMF: `{sec['avg_cmf']:.2f}`\n"
            text += f"   • 👑 **Cổ phiếu Dẫn dắt (Leader)**: `{sec['leader']}`\n"
            text += "------------------------------------\n"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi phân tích ngành: {str(e)}")


async def setbuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 3:
        await update.message.reply_text("⚠️ Sai cú pháp! Ví dụ: `/setbuy HPG 28.5 28.0 1000`", parse_mode="Markdown")
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
    await update.message.reply_text(f"✅ Đã thiết lập canh mua mã **{symbol}** tại mốc `{buy_alert_price}k` thành công!", parse_mode="Markdown")


async def setstop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("⚠️ Sai cú pháp! Ví dụ: `/setstop HPG 26.5`", parse_mode="Markdown")
        return
    symbol = args[0].upper().strip()
    try:
        stop_price = float(args[1])
    except ValueError:
        await update.message.reply_text("❌ Mốc cắt lỗ phải là số hợp lệ!")
        return

    if chat_id not in USER_PORTFOLIOS: USER_PORTFOLIOS[chat_id] = {}
    if symbol not in USER_PORTFOLIOS[chat_id]:
        USER_PORTFOLIOS[chat_id][symbol] = {"entry": stop_price, "qty": 1000, "buy_alert_prices": [], "stop_alert_prices": []}
    if stop_price not in USER_PORTFOLIOS[chat_id][symbol]["stop_alert_prices"]:
        USER_PORTFOLIOS[chat_id][symbol]["stop_alert_prices"].append(stop_price)
    save_portfolios()
    await update.message.reply_text(f"✅ Đã thiết lập mốc **Stop Loss** cho **{symbol}** tại `{stop_price}k` thành công!", parse_mode="Markdown")


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in USER_PORTFOLIOS or not USER_PORTFOLIOS[chat_id]:
        await update.message.reply_text("📂 Danh mục của bạn đang trống. Dùng `/setbuy` hoặc `/setstop` để bắt đầu.")
        return
    text = "📂 **DANH MỤC ĐẦU TƯ & GIÁM SÁT REAL-TIME**\n\n"
    for symbol, pos in USER_PORTFOLIOS[chat_id].items():
        df, err = fetch_stock_data(symbol, resolution="60", count=10)
        if not err and not df.empty:
            current_price = df.iloc[-1]["Close"]
            entry, qty = pos["entry"], pos.get("qty", 1000)
            pnl_pct = ((current_price - entry) / entry) * 100
            pnl_vnd = (current_price - entry) * qty
            status = "🟢 LÃI" if pnl_pct >= 0 else "🔴 LỖ"
            text += f"🔹 **{symbol.upper()}** ({qty} cổ) | Vốn: `{entry}k` | Hiện tại: `{current_price:.2f}k`\n"
            text += f"  • Hiệu suất: `{pnl_pct:+.2f}%` (`{pnl_vnd:+,.0f} VNĐ`) [{status}]\n\n"
    await update.message.reply_text(text, parse_mode="Markdown")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    text = update.message.text.strip().upper()
    if len(text) < 3 or len(text) > 5 or not text.isalpha():
        await update.message.reply_text("⚠️ Vui lòng nhập mã cổ phiếu hợp lệ (VD: `HPG`, `VHM`, `BSR`)!")
        return

    keyboard = [
        [
            InlineKeyboardButton("⏱️ Khung 1H (Giờ)", callback_data=f"{text}|60|450"),
            InlineKeyboardButton("📅 Khung 1D (Ngày)", callback_data=f"{text}|D|300"),
            InlineKeyboardButton("📈 Khung 1W (Tuần)", callback_data=f"{text}|W|250"),
        ],
    ]
    await update.message.reply_text(
        f"🔍 Đã nhận mã **{text}**. Chọn khung thời gian để phân tích:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "run_autobuy_callback":
        await autobuy_command(update, context)
        return
    if data == "run_sectors_callback":
        await sectors_command(update, context)
        return

    parts = data.split("|")
    if len(parts) == 3:
        symbol, resolution, count = parts[0], parts[1], int(parts[2])
        msg = await query.message.reply_text(f"⏳ Đang phân tích chuyên sâu {symbol.upper()} (Khung: {resolution})...")
        
        # Gọi lại hàm analyze_stock tiêu chuẩn của bạn để render ảnh và báo cáo
        from __main__ import analyze_stock
        def background_task():
            return analyze_stock(symbol, resolution=resolution, count=count)

        future = executor.submit(background_task)
        try:
            text_part1, text_part2, img_buf, reply_markup, err = future.result(timeout=15)
            if err:
                await msg.edit_text(f"❌ Lỗi: {err}")
            else:
                await msg.delete()
                await query.message.reply_photo(photo=img_buf)
                await query.message.reply_text(text_part1, parse_mode="Markdown")
                await query.message.reply_text(text_part2, reply_markup=reply_markup, parse_mode="Markdown")
        except Exception as e:
            await msg.edit_text(f"❌ Lỗi hệ thống: {str(e)}")


# ==========================================
# 3. HTTP SERVER & KHỞI CHẠY BOT
# ==========================================
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Quant Pro Bot Optimized is running successfully!")

def run_http_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler)
    server.serve_forever()

if __name__ == "__main__":
    threading.Thread(target=run_http_server, daemon=True).start()

    print("🤖 Đang khởi tạo Telegram Bot Pro (Phiên bản Tối ưu)...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_error_handler(global_error_handler)

    # Đăng ký Command Handler
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("scan", lambda u, c: scan_command(u, c)))
    app.add_handler(CommandHandler("testcung", lambda u, c: testcung_command(u, c)))
    app.add_handler(CommandHandler("demand", lambda u, c: demand_command(u, c)))
    app.add_handler(CommandHandler("autobuy", autobuy_command))
    app.add_handler(CommandHandler("sectors", sectors_command))
    app.add_handler(CommandHandler("nganh", sectors_command))
    app.add_handler(CommandHandler("setbuy", setbuy_command))
    app.add_handler(CommandHandler("setstop", setstop_command))
    app.add_handler(CommandHandler("portfolio", portfolio_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    print("🚀 Bot Quant Trading Pro đã khởi chạy thành công và sẵn sàng phục vụ lệnh!")
    app.run_polling(drop_pending_updates=True)
