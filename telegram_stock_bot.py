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
# 1. HÀM CALL DATA & TÍNH TOÁN TOÀN DIỆN
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
    """Bộ lọc sức khỏe Thị trường Chung (VN-Index)"""
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
                recent_3 = data[:3]
                is_net_buying = all(item.get("netVal", 0) > 0 for item in recent_3)
                return is_net_buying
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

    if close > cloud_top:
        ichi_status = "BÙNG NỔ TRÊN MÂY (TĂNG MẠNH)"
    elif close < cloud_bottom:
        ichi_status = "DƯỚI MÂY (XU HƯỚNG GIẢM)"
    else:
        ichi_status = "ĐANG TRONG MÂY (TÍCH LŨY/BIỆN ĐỘNG)"

    poc, vah, val = calculate_volume_profile(df)

    if close > sma200:
        long_term_trend = "UPTREND DÀI HẠN (Trên SMA200)"
    else:
        long_term_trend = "DOWNTREND/PHỤC HỒI DÀI HẠN (Dưới SMA200)"

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
        risk_warnings.append(f"Cảnh báo Dòng tiền đầu cơ (Hot Money): RVOL cao ({rvol:.2f}x) nhưng CMF âm ({cmf:.2f})")
    elif cmf > 0 or mfi > 45:
        score += 10
    else:
        risk_warnings.append(f"Dòng tiền yếu/rút lui (CMF: {cmf:.2f}, MFI: {mfi:.1f})")

    if is_bb_squeeze:
        score += 10
        confluence_points.append("⚡ [NỀN TÍCH LŨY SIẾT CHẶT BB]: Dải Bollinger siết hẹp, chuẩn bị bùng nổ xu hướng")
    if is_rsi_bull_div:
        score += 15
        confluence_points.append("🎯 [PHÂN KỲ DƯƠNG RSI]: Tín hiệu tóm đáy / Đảo chiều tăng giá tiềm năng")

    is_bull_trap = (rsi < 40 and stoch_rsi > 80)
    is_bear_trap = (rsi > 65 and stoch_rsi < 20)
    is_exhaustion_trap = (lr_slope > 0 and macd_hist < 0)
    is_overbought_full = (stoch_rsi > 90 and rsi > 70)

    trap_list = []
    if is_bull_trap:
        score -= 25
        trap_list.append("⚡ [BẪY TĂNG GIẢ - BULL TRAP]: Kéo giá ảo, xung lực ngắn hạn suy yếu.")
    if is_bear_trap:
        score += 10
        trap_list.append("⚡ [BẪY GIẢM GIẢ - BEAR TRAP / SHAKEOUT]: Nhịp đạp rũ bỏ kỹ thuật để gom hàng.")
    if is_exhaustion_trap:
        score -= 10
        trap_list.append("⚠️ [BẪY KIỆT SỨC - EXHAUSTION TRAP]: Giá rướn tăng nhưng Histogram MACD suy yếu.")
    if is_overbought_full:
        score -= 15
        trap_list.append("⚠️ [QUÁ MUA TOÀN DIỆN]: Rủi ro hưng phấn cực độ.")
    if is_rsi_bear_div:
        score -= 20
        trap_list.append("⚠️ [PHÂN KỲ ÂM RSI]: Giá lập đỉnh mới nhưng xung lực suy giảm, rủi ro đảo chiều cao.")
    if is_out_bb:
        score -= 10
        trap_list.append("⚠️ [VƯỢT NGOÀI BOLINGER TOP]: Giá bứt ra khỏi biên trên, rủi ro điều chỉnh kỹ thuật (Mean Reversion).")

    if not trap_list:
        trap_list.append("🛡️ Không phát hiện cấu trúc bẫy giá nguy hiểm đáng kể.")
    trap_desc = " | ".join(trap_list)

    if close > sma200:
        score += 15
        confluence_points.append("Giá duy trì trên đường dài hạn SMA200")
    else:
        risk_warnings.append("Giá nằm dưới SMA200")

    if macd_hist > 0:
        score += 15
        confluence_points.append(f"MACD Histogram dương ({macd_hist:+.2f})")
    else:
        risk_warnings.append(f"MACD Histogram âm ({macd_hist:+.2f})")

    mkt_health = check_vnindex_health()
    score = score * mkt_health["score_modifier"]

    confidence_score = min(max(int(score), 10), 88)

    is_below_val = (close < val)
    if is_below_val:
        zone_desc = f"Vùng **{fmt(val)} - {fmt(vah)}** (Dưới VAL): Trở thành **KHÁNG CỰ MẠNH**."
    elif close > vah:
        zone_desc = f"Vùng **{fmt(poc)} - {fmt(vah)}** (Trên VAH): Trở thành **HỖ TRỢ NỀN VỮNG CHẮC**."
    elif val <= close <= vah:
        zone_desc = f"Vùng **{fmt(val)} - {fmt(vah)}** (Trong Value Area): Tích lũy chuẩn hóa."
    else:
        zone_desc = f"Vùng **{fmt(val)} - {fmt(poc)}**: Vùng nhạy cảm sát biên dưới."

    if confidence_score >= 65 and not is_below_val and close > cloud_bottom and not is_overbought_full and not is_bull_trap and not is_rsi_bear_div:
        system_status = "MỞ MUA / GIA TĂNG TỶ TRỌNG"
        nav_rec = "25 - 35% NAV"
        final_icon = f"🟢 **NÊN MUA / GIA TĂNG TỶ TRỌNG (Độ tin cậy: {confidence_score}%)**"
        final_action = f"Các chỉ báo đồng thuận ủng hộ đà tăng ({mkt_health['note']}). Ưu tiên gom quanh hỗ trợ `{fmt(val)} - {fmt(poc)}`."
        
        entry_1 = round(min(kijun, val) if is_below_val else max(val, last["EMA20"], close * 0.98), 2)
        entry_2 = round(val if is_below_val else close, 2)
        entry_avg = round((entry_1 + entry_2) / 2, 2)
        stop_loss = round(entry_avg - (1.5 * atr), 2)
        action_note = f"Giải ngân quanh mức `{fmt(entry_1)} - {fmt(entry_2)}`, Stoploss ATR tại `{fmt(stop_loss)}`."
        
    elif confidence_score >= 45 and not is_bull_trap and cmf >= -0.2:
        system_status = "QUAN SÁT / THẬN TRỌNG RUNG LẮC"
        nav_rec = "10 - 15% NAV (Thăm dò)"
        final_icon = f"🟡 **GIAO DỊCH THĂM DÒ / QUAN SÁT (Độ tin cậy: {confidence_score}%)**"
        final_action = f"Kiên nhẫn chờ chiết khấu an toàn hoặc test đường Kijun-sen (`{fmt(kijun)}`)."
        
        entry_1 = round(close * 0.97, 2)
        entry_2 = round(close, 2)
        entry_avg = round((entry_1 + entry_2) / 2, 2)
        stop_loss = round(entry_avg - (1.5 * atr), 2)
        action_note = f"Thăm dò tỷ trọng nhỏ quanh mức `{fmt(entry_1)} - {fmt(entry_2)}`, Stoploss tại `{fmt(stop_loss)}`."
    else:
        system_status = "ĐỨNG NGOÀI (BẢO TOÀN VỐN)"
        nav_rec = "0% (Giữ tiền mặt)"
        final_icon = f"🔴 **ĐỨNG NGOÀI QUAN SÁT (Độ tin cậy: {confidence_score}%)**"
        final_action = f"Thị trường/Mã cổ phiếu đối mặt rủi ro. Đứng ngoài bảo toàn vốn."
        stop_loss = round(close - (1.5 * atr), 2)
        action_note = f"Tuyệt đối không giải ngân mua mới. Stoploss tham chiếu: `{fmt(stop_loss)}`."

    conf_str = "; ".join(confluence_points) if confluence_points else "Chưa có tín hiệu hội tụ mạnh"
    warn_str = "; ".join(risk_warnings) if risk_warnings else "Không có cảnh báo rủi ro nghiêm trọng"

    if (is_below_val and cmf < 0) or cmf < -0.20 or close < cloud_bottom:
        sec5_action = "⚠️ **Kịch bản Rủi ro dòng tiền rút:** Cấu trúc xu hướng suy yếu. Tuyệt đối không bắt đáy sớm."
    elif is_bull_trap or is_rsi_bear_div:
        sec5_action = "⚡ **Kịch bản Cảnh báo Phân kỳ / Bẫy giá:** Nhịp tăng rướn rủi ro cao, không mua đuổi."
    elif is_overbought_full or is_out_bb:
        sec5_action = "⚠️ **Kịch bản Quá mua / Vượt biên:** Ưu tiên chốt lời hiện thực hóa lợi nhuận."
    elif confidence_score >= 65 and not is_below_val:
        sec5_action = "🚀 **Kịch bản Thuận lợi:** Tiếp diễn xu hướng tăng."
    else:
        sec5_action = "🛡️ **Kịch bản Tích lũy/Giằng co:** Tín hiệu đan xen, tuân thủ kỷ luật."

    text_part1 = f"📊 BÁO CÁO PHÂN TÍCH KỸ THUẬT: {symbol.upper()}\n"
    text_part1 += f"🗓️ Ngày: {last['time'].strftime('%Y-%m-%d')} | Khung: {resolution} | Giá hiện tại: {fmt(close)}\n"
    text_part1 += f"🏛️ Trạng thái VN-Index: **{mkt_health['note']}**\n\n"

    text_part1 += f"🌐 1. CẤU TRÚC XU HƯỚNG & DÒNG TIỀN (MA & MACD):\n"
    text_part1 += f"• Xu hướng dài hạn (SMA200): {long_term_trend} (Giá: {fmt(close)} vs SMA200: {fmt(sma200)})\n"
    text_part1 += f"• Chỉ số ADX: `{adx:.1f}` | CMF: {cmf:.2f} | MFI: {mfi:.1f} | RVOL: {rvol:.2f}x\n"
    text_part1 += f"👉 **Kết luận & Hành động:** Xu hướng & động lượng vận động ổn định.\n\n"

    text_part1 += f"☁️ 2. CHỈ BÁO ICHIMOKU & ĐỘNG LƯỢNG (RSI & StochRSI):\n"
    text_part1 += f"• Thông số mây: {ichi_status}\n"
    text_part1 += f"• Động lượng ngắn/trung hạn: RSI: {rsi:.1f} | StochRSI: {stoch_rsi:.1f}\n"
    text_part1 += f"👉 **Kết luận & Hành động:** Canh giải ngân kiểm định vùng hỗ trợ an toàn.\n\n"

    text_part1 += f"📊 3. VOLUME PROFILE & ĐÁNH GIÁ KHÁNG CỰ / HỖ TRỢ:\n"
    text_part1 += f"• POC: {fmt(poc)} | VAL: {fmt(val)} | VAH: {fmt(vah)}\n"
    text_part1 += f"👉 **Kết luận & Hành động:** {zone_desc}\n\n"

    text_part1 += f"🎯 4. QUẢN TRỊ RỦI RO & PHÂN BỔ NAV (ATR):\n"
    text_part1 += f"• Khuyến nghị: {system_status} | NAV: {nav_rec}\n"
    text_part1 += f"👉 **Kết luận & Hành động:** {action_note}\n\n"

    text_part2 = f"⚡ 5. NHẬN DIỆN BẪY GIÁ & PHÂN KỲ (MARKET TRAPS & DIVERGENCE):\n"
    text_part2 += f"• Đánh giá: {trap_desc}\n"
    text_part2 += f"👉 **Lời khuyên chuyên gia:** Luôn kiểm chứng khối lượng và phân kỳ để tránh sập bẫy tạo lập.\n\n"

    text_part2 += f"⚡ 6. TỔNG HỢP ĐIỂM CỘNG & CẢNH BÁO RỦI RO:\n"
    text_part2 += f"• Điểm cộng: {conf_str}\n"
    text_part2 += f"• Cảnh báo: {warn_str}\n"
    text_part2 += f"👉 **Kết luận:** {sec5_action}\n\n"

    squeeze_display = "🟢 Có" if is_bb_squeeze else "🔴 Không"
    text_part2 += f"📈 7. DÃI BOLINGER BANDS & TÍCH LŨY SQUEEZE:\n"
    text_part2 += f"• Trên: {fmt(bb_upper)} | Dưới: {fmt(bb_lower)} | Độ rộng: {bb_width:.3f} | Squeeze Nền: {squeeze_display}\n\n"

    text_part2 += f"🏁 8. TỔNG KẾT CHIẾN LƯỢC:\n"
    text_part2 += f"• Khuyến nghị: {final_icon}\n"
    text_part2 += f"• Định hướng: {final_action}"

    keyboard = [
        [
            InlineKeyboardButton("⏱️ Khung 1H (Giờ)", callback_data=f"{symbol}|60|450"),
            InlineKeyboardButton("📅 Khung 1D (Ngày)", callback_data=f"{symbol}|D|300"),
            InlineKeyboardButton("📈 Khung 1W (Tuần)", callback_data=f"{symbol}|W|250"),
        ],
        [
            InlineKeyboardButton("🔍 Test Cung (/testcung)", callback_data="run_testcung_callback"),
            InlineKeyboardButton("🚀 Cầu Chủ Động (/demand)", callback_data="run_demand_callback"),
        ],
        [
            InlineKeyboardButton("🔄 Shakeout (/shakeout)", callback_data="run_shakeout_callback"),
            InlineKeyboardButton("🛡️ Biên An Toàn (/safety)", callback_data="run_safety_callback"),
        ],
        [
            InlineKeyboardButton("🚀 Quét Bùng Nổ (/breakout)", callback_data="run_breakout_callback"),
            InlineKeyboardButton("🤖 Quét Điểm Mua (/autobuy)", callback_data="run_autobuy_callback"),
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
    ax.set_title(f"Phân Tích Chuyên Sâu - {symbol.upper()} ({resolution})", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return text_part1, text_part2, buf, reply_markup, None


# ==========================================
# 3. HỆ THỐNG SCREENER PHỄU 5 TẦNG (MULTI-STAGE FUNNEL)
# ==========================================
def check_liquidity_and_base(df, last):
    """Bộ lọc thanh khoản nền: Lọc bỏ mã có giá trị giao dịch < 5 tỷ VNĐ/phiên"""
    close = last["Close"]
    vol_sma20 = last["Vol_SMA20"] if not np.isnan(last.get("Vol_SMA20", np.nan)) else last["Volume"]
    avg_trading_value = vol_sma20 * close * 1000
    if avg_trading_value < 5_000_000_000:
        return False
    return True


# --- TẦNG 1: QUÉT TÍCH LŨY NỀN & DÒNG TIỀN NHA NHÓM (/scan) ---
def run_market_screener():
    results = []
    mkt_health = check_vnindex_health()

    def scan_one(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=200)
        if err or df.empty or len(df) < 50:
            return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        
        if not check_liquidity_and_base(df, last):
            return None

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
            if res:
                results.append(res)
            
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:8]


# --- TẦNG 2: QUÉT CẠN CUNG / NO-SUPPLY (/testcung) ---
def run_test_supply_screener():
    results = []
    def scan_test_supply(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=150)
        if err or df.empty or len(df) < 50:
            return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        
        if not check_liquidity_and_base(df, last):
            return None

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
            if res:
                results.append(res)
            
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:6]


# --- TẦNG 3: QUÉT RỦ BỎ RÚT CHÂN / BEAR TRAP (/shakeout) ---
def run_shakeout_screener():
    results = []
    def scan_shakeout(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=150)
        if err or df.empty or len(df) < 50:
            return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        
        if not check_liquidity_and_base(df, last):
            return None

        close, open_p, high, low = last["Close"], last["Open"], last["High"], last["Low"]
        rvol = last.get("RVOL", 1.0)
        cmf = last.get("CMF", 0.0)
        ema50 = last.get("EMA50", close)

        candle_range = high - low
        if candle_range == 0:
            return None
            
        lower_wick = min(open_p, close) - low
        lower_wick_ratio = lower_wick / candle_range
        is_rebound = close >= low + candle_range * 0.45
        
        if lower_wick_ratio >= 0.45 and is_rebound and close >= (ema50 * 0.97) and cmf >= -0.15:
            score = (lower_wick_ratio * 40) + (max(0, cmf) * 25) + (min(rvol, 2.5) * 10)
            return {
                "symbol": symbol,
                "close": close,
                "lower_wick_pct": lower_wick_ratio * 100,
                "rvol": rvol,
                "cmf": cmf,
                "score": score
            }
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_shakeout, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
            
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:6]


# --- TẦNG 4A: QUÉT CẦU CHỦ ĐỘNG XÁC NHẬN (/demand) ---
def run_demand_confirmation_screener():
    results = []
    def scan_demand(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=150)
        if err or df.empty or len(df) < 50:
            return None
        df = calculate_indicators(df)
        last, prev = df.iloc[-1], df.iloc[-2]
        
        if not check_liquidity_and_base(df, last):
            return None

        close, high, low = last["Close"], last["High"], last["Low"]
        rvol = last.get("RVOL", 1.0)
        cmf = last.get("CMF", 0.0)
        mfi = last.get("MFI", 50.0)
        kijun = last.get("Kijun_sen", close)

        candle_range = high - low
        if candle_range == 0:
            return None
            
        is_strong_close = ((close - low) / candle_range) >= 0.70
        was_no_supply = prev.get("RVOL", 1.0) <= 0.70
        
        if rvol >= 1.50 and is_strong_close and cmf >= 0.08 and close >= kijun:
            bonus = 30 if was_no_supply else 0
            score = (rvol * 20) + (cmf * 50) + (mfi * 0.5) + bonus
            return {
                "symbol": symbol,
                "close": close,
                "rvol": rvol,
                "cmf": cmf,
                "mfi": mfi,
                "was_no_supply": was_no_supply,
                "score": score
            }
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_demand, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
            
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:6]


# --- TẦNG 4B: QUÉT BÙNG NỔ XU HƯỚNG (/breakout) ---
def run_breakout_screener():
    results = []
    mkt_health = check_vnindex_health()

    def scan_breakout(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=200)
        if err or df.empty or len(df) < 50:
            return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        
        if not check_liquidity_and_base(df, last):
            return None

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
            if res:
                results.append(res)
            
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:6]


# --- TẦNG BỔ TRỢ: QUÉT BIÊN AN TOÀN CƠ BẢN (/safety) ---
def run_safety_value_screener():
    results = []
    def scan_safety(symbol):
        ratios = get_fundamental_ratios(symbol)
        pe, roe = ratios.get("pe", 99), ratios.get("roe", 0)
        
        if pe <= 0 or pe > 22 or roe < 12.0:
            return None
            
        df, err = fetch_stock_data(symbol, resolution="D", count=100)
        if err or df.empty:
            return None
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
            if res:
                results.append(res)
            
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:6]


def run_hourly_cashflow_screener():
    results = []
    def scan_one(symbol):
        df, err = fetch_stock_data(symbol, resolution="60", count=30)
        if err or df.empty or len(df) < 15:
            return None
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
                "symbol": symbol,
                "close": close,
                "rvol": rvol,
                "cmf": cmf,
                "mfi": mfi,
                "macd_hist": macd_hist,
                "score": score
            }
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_one, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
            
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:6]


# --- TẦNG 5 (ĐỈNH PHỄU): ĐIỂM MUA TỔNG HỢP AN TOÀN (/autobuy) ---
def process_single_stock_for_autobuy(symbol):
    try:
        mkt_health = check_vnindex_health()

        ratios = get_fundamental_ratios(symbol)
        if ratios["pe"] > 45 or ratios["roe"] < 0:
            return None

        df, err = fetch_stock_data(symbol, resolution="D", count=200)
        if err or df.empty or len(df) < 50:
            return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        
        if not check_liquidity_and_base(df, last):
            return None

        close = last["Close"]
        cmf = last.get("CMF", 0.0)
        rvol = last.get("RVOL", 1.0)
        adx = last.get("ADX", 20.0)
        macd_hist = last.get("MACD_Hist", 0.0)
        rsi = last.get("RSI", 50.0)
        stoch_rsi = last.get("Stoch_RSI", 50.0)
        is_rsi_bear_div = last.get("RSI_Bearish_Divergence", False)
        is_bb_squeeze = last.get("BB_Squeeze", False)
        price_range_pct = last.get("Price_Range_Pct", 0.1)
        
        span_a = last.get("Senkou_Span_A", close)
        span_b = last.get("Senkou_Span_B", close)
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)

        score = 50
        if close > cloud_top:
            score += 20
        elif close >= cloud_bottom:
            score += 10
        else:
            score -= 15

        if cmf > 0.02:
            score += 15
        if rvol > 1.1 or price_range_pct < 0.07:
            score += 15
        if macd_hist > 0:
            score += 10
        if is_bb_squeeze:
            score += 10

        is_foreign_buying = get_foreign_trading_flow(symbol)
        if is_foreign_buying:
            score += 15

        is_wave_ok, wave_info = analyze_multi_timeframe_poc(symbol)
        score += wave_info.get("score_bonus", 0)

        is_bull_trap = (rsi < 40 and stoch_rsi > 80)
        if is_bull_trap:
            score -= 30
        if is_rsi_bear_div:
            score -= 25

        score = int(score * mkt_health["score_modifier"])

        if score >= 60 and cmf > -0.15 and close >= cloud_bottom and not is_bull_trap and not is_rsi_bear_div and is_wave_ok:
            atr = last.get("ATR", close * 0.03)
            poc, vah, val = calculate_volume_profile(df)
            
            entry_price = round(max(val, close * 0.98), 2)
            stop_loss = round(entry_price - (1.5 * atr), 2)
            
            return {
                "symbol": symbol,
                "close": close,
                "score": min(score, 99),
                "cmf": cmf,
                "rvol": rvol,
                "entry": entry_price,
                "stop_loss": stop_loss
            }
    except Exception:
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
                        
                        buy_prices = pos.get("buy_alert_prices", [])
                        for buy_price in buy_prices:
                            buy_trigger_key = f"{chat_id}_{symbol}_buy_{buy_price}_{today_date}"
                            if current_price <= buy_price and buy_trigger_key not in ALERTED_TODAY:
                                ALERTED_TODAY.add(buy_trigger_key)
                                alert_text = (
                                    f"🎯 **CẢNH BÁO CANH MUA: ĐÃ CHẠM MỐC `{buy_price}k`!** 🎯\n\n"
                                    f"• Mã cổ phiếu: **{symbol.upper()}**\n"
                                    f"• **Giá hiện tại**: `{current_price:.2f}k` 📉\n\n"
                                    f"👉 Giá đã về vùng canh mua đã thiết lập!"
                                )
                                try:
                                    application.bot.send_message(chat_id=chat_id, text=alert_text, parse_mode="Markdown")
                                except Exception:
                                    pass

                        stop_prices = pos.get("stop_alert_prices", [])
                        for stop_price in stop_prices:
                            stop_trigger_key = f"{chat_id}_{symbol}_stop_{stop_price}_{today_date}"
                            if current_price <= stop_price and stop_trigger_key not in ALERTED_TODAY:
                                ALERTED_TODAY.add(stop_trigger_key)
                                alert_text = (
                                    f"🚨 **CẢNH BÁO STOP LOSS: GIÁ ĐÃ RƠI XUỐNG DƯỚI `{stop_price}k`!** 🚨\n\n"
                                    f"• Mã cổ phiếu: **{symbol.upper()}**\n"
                                    f"• **Giá hiện tại**: `{current_price:.2f}k` 📉\n\n"
                                    f"👉 Đã chạm hoặc thủng mốc cắt lỗ an toàn. Hãy xem xét xử lý!"
                                )
                                try:
                                    application.bot.send_message(chat_id=chat_id, text=alert_text, parse_mode="Markdown")
                                except Exception:
                                    pass

        if now.weekday() < 5 and now.hour in [9, 10, 11, 13, 14] and now.minute <= 2:
            if last_hourly_scan_hour != now.hour and ACTIVE_CHAT_IDS:
                last_hourly_scan_hour = now.hour
                top_hourly = run_hourly_cashflow_screener()
                if top_hourly:
                    report_msg = f"⏱️ **BẢN TIN DÒNG TIỀN CÔ ĐỌNG HÀNG GIỜ ({now.strftime('%H:00')} - {now.strftime('%d/%m/%Y')})**\n"
                    report_msg += "------------------------------------\n"
                    report_msg += "💡 *Top cổ phiếu hút dòng tiền mạnh trong 1 giờ qua:*\n\n"
                    
                    for idx, item in enumerate(top_hourly, 1):
                        trend_icon = "🟢" if item['macd_hist'] >= 0 else "🟡"
                        report_msg += f"{idx}. {trend_icon} **{item['symbol']}** | Giá: `{item['close']:.1f}k`\n"
                        report_msg += f"   • RVOL: `{item['rvol']:.2f}x` | CMF: `{item['cmf']:.2f}` | MFI: `{item['mfi']:.1f}`\n"
                    
                    report_msg += "\n------------------------------------\n"
                    report_msg += "🤖 *Hệ thống giám sát real-time liên tục.*"

                    for chat_id in ACTIVE_CHAT_IDS:
                        try:
                            application.bot.send_message(chat_id=chat_id, text=report_msg, parse_mode="Markdown")
                        except Exception:
                            pass

        if now.weekday() < 5 and now.hour == 14 and now.minute == 15:
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

        if now.weekday() < 5 and now.hour == 15 and now.minute == 0:
            if sent_daily_report_today != today_date and USER_PORTFOLIOS:
                sent_daily_report_today = today_date
                
                for chat_id, portfolio in USER_PORTFOLIOS.items():
                    if not portfolio:
                        continue
                        
                    report_text = f"📊 **BÁO CÁO TỔNG KẾT CUỐI PHIÊN GIAO DỊCH** ({now.strftime('%d/%m/%Y')})\n"
                    report_text += "------------------------------------\n"
                    total_pnl_vnd = 0
                    has_data = False
                    
                    for symbol, pos in portfolio.items():
                        df, err = fetch_stock_data(symbol, resolution="D", count=5)
                        if not err and not df.empty:
                            has_data = True
                            current_price = df.iloc[-1]["Close"]
                            qty = pos.get("qty", 1000)
                            entry = pos["entry"]
                            
                            pnl_pct = ((current_price - entry) / entry) * 100
                            pnl_vnd = (current_price - entry) * qty
                            total_pnl_vnd += pnl_vnd
                            status = "🟢 LÃI" if pnl_pct >= 0 else "🔴 LỖ"

                            report_text += f"🔹 **{symbol.upper()}** ({qty} cổ)\n"
                            report_text += f"  • Giá vốn: `{entry}k` | Đóng cửa: `{current_price:.2f}k`\n"
                            report_text += f"  • Hiệu suất: `{pnl_pct:+.2f}%` (`{pnl_vnd:+,.0f} VNĐ`) [{status}]\n\n"
                        else:
                            report_text += f"🔹 **{symbol.upper()}**: Không lấy được dữ liệu chốt phiên.\n\n"
                            
                    if has_data:
                        overall_status = "🟢 LÃI TỔNG" if total_pnl_vnd >= 0 else "🔴 LỖ TỔNG"
                        report_text += "------------------------------------\n"
                        report_text += f"💰 **LỢI NHUẬN TỔNG DANH MỤC**: `{total_pnl_vnd:+,.0f} VNĐ` [{overall_status}]\n"
                        report_text += "🤖 Hệ thống đã lưu trữ trạng thái. Chúc bạn một buổi chiều an lành!"
                        
                        try:
                            application.bot.send_message(chat_id=chat_id, text=report_text, parse_mode="Markdown")
                        except Exception:
                            pass
                
                time.sleep(70)
        
        time.sleep(30)


def run_strategy_backtest(symbol):
    symbol = symbol.upper().strip()
    df, err = fetch_stock_data(symbol, resolution="D", count=300)
    if err or df.empty or len(df) < 100:
        return f"⚠️ Không đủ dữ liệu lịch sử để backtest mã {symbol}."
    
    df = calculate_indicators(df)
    trades, wins, losses, total_return = 0, 0, 0, 0.0
    
    for i in range(50, len(df) - 10):
        rvol = df["RVOL"].iloc[i]
        cmf = df["CMF"].iloc[i]
        adx = df["ADX"].iloc[i]
        close = df["Close"].iloc[i]
        ema50 = df["EMA50"].iloc[i]
        atr = df["ATR"].iloc[i]
        
        if not np.isnan(rvol) and not np.isnan(cmf) and not np.isnan(adx) and (rvol > 1.2 or df["Price_Range_Pct"].iloc[i] < 0.08) and cmf > 0.02 and close > ema50:
            entry_price = close
            stoploss_price = entry_price - (1.5 * atr) if not np.isnan(atr) else entry_price * 0.935
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
        return f"📊 KẾT QUẢ BACKTEST: {symbol}\n\nChưa xuất hiện điểm mua đạt tiêu chuẩn."

    win_rate = (wins / trades) * 100
    report = f"📊 KẾT QUẢ BACKTEST CHIẾN LƯỢC (TÍCH LŨY NỀN & DÒNG TIỀN): {symbol}\n"
    report += f"• Tổng số lệnh: {trades} | Thắng: {wins} | Thua: {losses}\n"
    report += f"• Tỷ lệ thắng: {win_rate:.1f}% | Hiệu suất: {total_return:+.2f}%\n"
    return report


# ==========================================
# 4. TELEGRAM HANDLERS
# ==========================================
async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.error(f"Error update: {update} caused error: {context.error}")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    welcome_text = (
        "👋 **CHÀO MỪNG BẠN ĐẾN VỚI QUANT TRADING SYSTEM (PRO 5%)!**\n\n"
        "1️⃣ **Phân tích kỹ thuật đơn mã:** Nhập mã cổ phiếu trực tiếp (Ví dụ: `VHM`, `BSR`)\n\n"
        "2️⃣ **🎯 BỘ LỌC PHỄU DÒNG TIỀN (VSA & QUANT SCREENER):**\n"
        "   ├ 🟢 `/scan` : **Tầng 1** - Dò cổ phiếu tích lũy nền & dòng tiền nha nhóm\n"
        "   ├ 🔍 `/testcung`: **Tầng 2** - Săn mã cạn kiệt thanh khoản (No-Supply/Rũ xong)\n"
        "   ├ 🔄 `/shakeout`: **Tầng 3** - Quét nến rút chân rũ nhỏ lẻ (Bear Trap)\n"
        "   ├ 🚀 `/demand`  : **Tầng 4A** - Xác nhận dòng tiền mua chủ động (Demand)\n"
        "   ├ 🔥 `/breakout`: **Tầng 4B** - Bùng nổ giá & khối lượng (RVOL >= 2.0x)\n"
        "   └ 🤖 `/autobuy` : **Tầng 5 (Phễu cuối)** - Lọc điểm giải ngân an toàn khắt khe nhất\n\n"
        "3️⃣ **🛡️ BỘ LỌC CƠ BẢN & ĐỊNH GIÁ:**\n"
        "   └ `/safety`: Lọc cổ phiếu Biên An Toàn (P/E hợp lý, ROE > 12%)\n\n"
        "4️⃣ **CÀI ĐẶT CANH MUA / STOP LOSS (REAL-TIME):**\n"
        "   • `/setbuy BSR 29.4 28.5 1000` | `/setstop BSR 27.9` | `/trail BSR`\n\n"
        "5️⃣ **QUẢN LÝ DANH MỤC & BACKTEST:**\n"
        "   • `/portfolio` | `/listall` | `/backtest VHM` | `/reset <MÃ>`"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def autobuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    if update.message:
        msg = await update.message.reply_text("🤖 **Đang quét siêu tốc các mã siết nền tích lũy & bùng nổ dòng tiền...** ⚡")
    else:
        query = update.callback_query
        msg = await query.message.reply_text("🤖 **Đang quét siêu tốc các mã siết nền tích lũy & bùng nổ dòng tiền...** ⚡")

    try:
        picks = run_smart_buy_screener()
        if not picks:
            await msg.edit_text("⚠️ Hiện tại thị trường chưa có mã nào đạt đủ tiêu chuẩn khắt khe để giải ngân an toàn.")
            return
            
        text = "🎯 **DANH SÁCH CỔ PHIẾU ĐẠT ĐIỂM MUA AN TOÀN (AUTO-BUY)**\n"
        text += "====================================\n\n"
        
        for idx, item in enumerate(picks, 1):
            text += f"{idx}. 🔹 **Mã: {item['symbol'].upper()}** (Độ tin cậy: `{item['score']}%`)\n"
            text += f"   • Giá hiện tại: `{item['close']:.1f}k`\n"
            text += f"   • Dòng tiền CMF: `{item['cmf']:.2f}` | RVOL: `{item['rvol']:.2f}x`\n"
            text += f"   • 🎯 **Vùng giải ngân chuẩn**: Khoảng `{item['entry']}k`\n"
            text += f"   • 🛡️ **Stoploss tham chiếu**: `{item['stop_loss']}k`\n"
            text += f"   • Lệnh cài canh mua nhanh: `/setbuy {item['symbol']} {item['close']} {item['entry']} 1000`\n"
            text += "------------------------------------\n"
            
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi thực thi bộ lọc auto-buy: {str(e)}")


async def breakout_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    if update.message:
        msg = await update.message.reply_text("🚀 **Đang kích hoạt radar quét tín hiệu BÙNG NỔ (Breakout chuẩn mực)...** ⚡")
    else:
        query = update.callback_query
        msg = await query.message.reply_text("🚀 **Đang kích hoạt radar quét tín hiệu BÙNG NỔ (Breakout chuẩn mực)...** ⚡")

    try:
        picks = run_breakout_screener()
        if not picks:
            await msg.edit_text("⚠️ Hiện tại chưa có cổ phiếu nào bùng nổ đủ tiêu chuẩn khắt khe (RVOL >= 2.0x, đóng cửa sát đỉnh & vượt biên/mây).")
            return
            
        text = "🚀 **DANH SÁCH CỔ PHIẾU CHÍNH THỨC BÙNG NỔ (BREAKOUT)**\n"
        text += "====================================\n\n"
        
        for idx, item in enumerate(picks, 1):
            text += f"{idx}. 🔥 **Mã: {item['symbol'].upper()}** (Điểm số: `{item['score']:.1f}`)\n"
            text += f"   • Giá hiện tại: `{item['close']:.1f}k`\n"
            text += f"   • RVOL bùng nổ: `{item['rvol']:.2f}x` | CMF: `{item['cmf']:.2f}`\n"
            text += f"   • Phân tích chi tiết: Nhập mã `{item['symbol']}`\n"
            text += "------------------------------------\n"
            
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi thực thi bộ lọc breakout: {str(e)}")


async def testcung_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    if update.message:
        msg = await update.message.reply_text("🔍 **Đang quét các mã cạn cung / rũ bỏ (Test Cung / No-Supply)...** ⚡")
    else:
        query = update.callback_query
        msg = await query.message.reply_text("🔍 **Đang quét các mã cạn cung / rũ bỏ (Test Cung / No-Supply)...** ⚡")

    try:
        picks = run_test_supply_screener()
        if not picks:
            await msg.edit_text("⚠️ Hiện tại chưa phát hiện mã nào có tín hiệu cạn kiệt thanh khoản kiểm định cung.")
            return
            
        text = "🔍 **TOP CỔ PHIẾU CẠN CUNG / KIỂM ĐỊNH CUNG (TEST CUNG)**\n"
        text += "*(Tiêu chí: RVOL rất thấp <= 0.65, giá rút chân/giữ nền, không áp lực bán tháo)*\n"
        text += "====================================\n\n"
        
        for idx, item in enumerate(picks, 1):
            text += f"{idx}. 🟢 **Mã: {item['symbol'].upper()}** (Điểm: `{item['score']:.1f}`)\n"
            text += f"   • Giá hiện tại: `{item['close']:.1f}k`\n"
            text += f"   • Khối lượng cạn kiệt RVOL: `{item['rvol']:.2f}x` (Rất thấp)\n"
            text += f"   • Dòng tiền CMF: `{item['cmf']:.2f}`\n"
            text += f"   • Phân tích chi tiết: Nhập mã `{item['symbol']}`\n"
            text += "------------------------------------\n"
            
        text += "\n💡 *Gợi ý chuyên gia: Đây là thời điểm tuyệt vời để gom mua nhặt dần trước khi dòng tiền lớn đánh thốc!*"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi thực thi bộ lọc test cung: {str(e)}")


async def demand_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    if update.message:
        msg = await update.message.reply_text("🚀 **Đang quét tín hiệu DÒNG TIỀN MUA CHỦ ĐỘNG (Demand Confirmation)...** ⚡")
    else:
        query = update.callback_query
        msg = await query.message.reply_text("🚀 **Đang quét tín hiệu DÒNG TIỀN MUA CHỦ ĐỘNG (Demand Confirmation)...** ⚡")

    try:
        picks = run_demand_confirmation_screener()
        if not picks:
            await msg.edit_text("⚠️ Chưa phát hiện mã nào bùng nổ Demand rõ nét (RVOL >= 1.5x & CMF dâng cao).")
            return
            
        text = "🚀 **TOP CỔ PHIẾU XÁC NHẬN DÒNG TIỀN MUA CHỦ ĐỘNG (DEMAND)**\n"
        text += "*(Tiêu chí: Vừa cạn cung/siết nền + RVOL bùng nổ >= 1.5x, đóng cửa sát đỉnh)*\n"
        text += "====================================\n\n"
        
        for idx, item in enumerate(picks, 1):
            setup_tag = "🔥 [Mô hình Cạn Cung -> Đánh thốc]" if item['was_no_supply'] else "⚡ [Xác nhận Cầu mạnh]"
            text += f"{idx}. 🟢 **Mã: {item['symbol'].upper()}** {setup_tag}\n"
            text += f"   • Giá hiện tại: `{item['close']:.1f}k`\n"
            text += f"   • RVOL bùng nổ: `{item['rvol']:.2f}x` | CMF: `{item['cmf']:.2f}` | MFI: `{item['mfi']:.1f}`\n"
            text += f"   • Phân tích chi tiết: Nhập mã `{item['symbol']}`\n"
            text += "------------------------------------\n"
            
        text += "\n💡 *Gợi ý chiến thuật: Đây là phiên xác nhận mua gia tăng/mua đuổi an toàn nhất theo chuẩn Wyckoff/VSA!*"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi thực thi bộ lọc demand: {str(e)}")


async def shakeout_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    if update.message:
        msg = await update.message.reply_text("🔄 **Đang quét các mã xuất hiện nến rút chân RỦ BỎ (Shakeout / Bear Trap)...** ⚡")
    else:
        query = update.callback_query
        msg = await query.message.reply_text("🔄 **Đang quét các mã xuất hiện nến rút chân RỦ BỎ (Shakeout / Bear Trap)...** ⚡")

    try:
        picks = run_shakeout_screener()
        if not picks:
            await msg.edit_text("⚠️ Hiện tại chưa phát hiện mã nào có cấu trúc rũ bỏ rút chân mạnh mẽ.")
            return
            
        text = "🔄 **TOP CỔ PHIẾU XUẤT HIỆN TÍN HIỆU RỦ BỎ RÚT CHÂN (SHAKEOUT)**\n"
        text += "*(Tiêu chí: Bóng nến dưới >= 45% chiều cao nến, rút chân mạnh giữ nền giá)*\n"
        text += "====================================\n\n"
        
        for idx, item in enumerate(picks, 1):
            text += f"{idx}. ⚓ **Mã: {item['symbol'].upper()}** (Điểm rũ bỏ: `{item['score']:.1f}`)\n"
            text += f"   • Giá hiện tại: `{item['close']:.1f}k`\n"
            text += f"   • Tỷ lệ rút chân nến: `{item['lower_wick_pct']:.1f}%`\n"
            text += f"   • RVOL: `{item['rvol']:.2f}x` | CMF: `{item['cmf']:.2f}`\n"
            text += f"   • Phân tích chi tiết: Nhập mã `{item['symbol']}`\n"
            text += "------------------------------------\n"
            
        text += "\n💡 *Gợi ý chuyên gia: Nến rút chân dài thể hiện Tạo lập đã gom hết hàng hoảng loạn. Có thể canh nhặt nền giá!*"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi thực thi bộ lọc shakeout: {str(e)}")


async def safety_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    if update.message:
        msg = await update.message.reply_text("🛡️ **Đang lọc danh mục cổ phiếu có định giá hấp dẫn & Biên An Toàn cao (P/E & ROE)...** ⚡")
    else:
        query = update.callback_query
        msg = await query.message.reply_text("🛡️ **Đang lọc danh mục cổ phiếu có định giá hấp dẫn & Biên An Toàn cao (P/E & ROE)...** ⚡")

    try:
        picks = run_safety_value_screener()
        if not picks:
            await msg.edit_text("⚠️ Hiện tại chưa tìm thấy mã đạt chuẩn biên an toàn (ROE > 12% và P/E hợp lý).")
            return
            
        text = "🛡️ **TOP CỔ PHIẾU CÓ BIÊN AN TOÀN & CƠ BẢN VỮNG CHẮC (SAFETY VALUE)**\n"
        text += "*(Tiêu chí: ROE > 12%, P/E từ 0 đến 22, dòng tiền ủng hộ)*\n"
        text += "====================================\n\n"
        
        for idx, item in enumerate(picks, 1):
            text += f"{idx}. 💎 **Mã: {item['symbol'].upper()}** (Điểm cơ bản: `{item['score']:.1f}`)\n"
            text += f"   • Giá hiện tại: `{item['close']:.1f}k`\n"
            text += f"   • Hiệu suất sinh lời ROE: `{item['roe']:.1f}%` | P/E: `{item['pe']:.1f}`\n"
            text += f"   • Dòng tiền CMF: `{item['cmf']:.2f}`\n"
            text += f"   • Phân tích chi tiết: Nhập mã `{item['symbol']}`\n"
            text += "------------------------------------\n"
            
        text += "\n💡 *Đây là các doanh nghiệp chất lượng cao, giữ tỷ trọng lớn an tâm nắm giữ dài hạn!*"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi thực thi bộ lọc safety: {str(e)}")


async def setbuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 3:
        await update.message.reply_text(
            "⚠️ Sai cú pháp!\n\n"
            "Ví dụ canh mua: `/setbuy PLX 36.2 35.5 1000`\n"
            "*(Định dạng: /setbuy <Mã> <Giá_vốn> <Giá_canh_mua> <Khối_lượng>)*",
            parse_mode="Markdown"
        )
        return

    symbol = args[0].upper().strip()
    try:
        entry_price = float(args[1])
        buy_alert_price = float(args[2])
        qty = int(args[3]) if len(args) >= 4 else 1000
    except ValueError:
        await update.message.reply_text("❌ Các tham số giá và khối lượng phải là số hợp lệ!")
        return

    if chat_id not in USER_PORTFOLIOS:
        USER_PORTFOLIOS[chat_id] = {}

    if symbol not in USER_PORTFOLIOS[chat_id]:
        USER_PORTFOLIOS[chat_id][symbol] = {
            "entry": entry_price,
            "qty": qty,
            "buy_alert_prices": [],
            "stop_alert_prices": []
        }
    
    USER_PORTFOLIOS[chat_id][symbol]["entry"] = entry_price
    USER_PORTFOLIOS[chat_id][symbol]["qty"] = qty

    if "buy_alert_prices" not in USER_PORTFOLIOS[chat_id][symbol]:
        USER_PORTFOLIOS[chat_id][symbol]["buy_alert_prices"] = []

    if buy_alert_price not in USER_PORTFOLIOS[chat_id][symbol]["buy_alert_prices"]:
        USER_PORTFOLIOS[chat_id][symbol]["buy_alert_prices"].append(buy_alert_price)

    save_portfolios()

    list_buys_str = ", ".join([f"{p}k" for p in USER_PORTFOLIOS[chat_id][symbol]["buy_alert_prices"]])

    await update.message.reply_text(
        f"✅ Đã thiết lập vùng **CANH MUA** thành công:\n"
        f"• Mã cổ phiếu: **{symbol}**\n"
        f"• Giá vốn: `{entry_price}k` | Khối lượng: {qty} cổ\n"
        f"• 🎯 **Các mốc giá canh mua**: `{list_buys_str}`\n\n"
        f"🤖 Bot sẽ thông báo ngay khi giá thị trường chạm/giảm về các mốc này!",
        parse_mode="Markdown"
    )


async def setstop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "⚠️ Sai cú pháp!\n\n"
            "Ví dụ cắt lỗ: `/setstop PLX 33.5`\n"
            "*(Định dạng: /setstop <Mã> <Giá_cắt_lỗ>)*",
            parse_mode="Markdown"
        )
        return

    symbol = args[0].upper().strip()
    try:
        stop_price = float(args[1])
    except ValueError:
        await update.message.reply_text("❌ Mốc giá cắt lỗ phải là số hợp lệ!")
        return

    if chat_id not in USER_PORTFOLIOS:
        USER_PORTFOLIOS[chat_id] = {}

    if symbol not in USER_PORTFOLIOS[chat_id]:
        USER_PORTFOLIOS[chat_id][symbol] = {
            "entry": stop_price,
            "qty": 1000,
            "buy_alert_prices": [],
            "stop_alert_prices": []
        }

    if "stop_alert_prices" not in USER_PORTFOLIOS[chat_id][symbol]:
        USER_PORTFOLIOS[chat_id][symbol]["stop_alert_prices"] = []

    if stop_price not in USER_PORTFOLIOS[chat_id][symbol]["stop_alert_prices"]:
        USER_PORTFOLIOS[chat_id][symbol]["stop_alert_prices"].append(stop_price)

    save_portfolios()

    list_stops_str = ", ".join([f"{p}k" for p in USER_PORTFOLIOS[chat_id][symbol]["stop_alert_prices"]])

    await update.message.reply_text(
        f"✅ Đã thiết lập vùng **STOP LOSS (CẮT LỖ)** thành công:\n"
        f"• Mã cổ phiếu: **{symbol}**\n"
        f"• 🛡️ **Các mốc giá cảnh báo cắt lỗ**: `{list_stops_str}`\n\n"
        f"🤖 Bot sẽ hú còi cảnh báo ngay khi giá rủi ro chạm hoặc thủng mốc này!",
        parse_mode="Markdown"
    )


async def trail_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if not args:
        await update.message.reply_text("⚠️ Cú pháp: `/trail <MÃ_CP>` (VD: `/trail BSR`)", parse_mode="Markdown")
        return

    symbol = args[0].upper().strip()
    df, err = fetch_stock_data(symbol, resolution="D", count=60)
    if err or df.empty:
        await update.message.reply_text(f"❌ Không lấy được dữ liệu tính Trailing Stop cho mã {symbol}.")
        return

    df = calculate_indicators(df)
    last = df.iloc[-1]
    close = last["Close"]
    high_60 = df["High"].max()
    atr = last["ATR"] if not np.isnan(last["ATR"]) else (close * 0.03)

    trailing_stop = round(high_60 - (2.0 * atr), 2)
    
    text = f"🎯 **TÍNH TOÁN MỐC TRAILING STOP (KHÓA LÃI ATR): {symbol}**\n\n"
    text += f"• Giá hiện tại: `{close:.2f}k` | Đỉnh cao nhất 60 phiên: `{high_60:.2f}k`\n"
    text += f"• Độ biến động ATR (14): `{atr:.2f}k`\n"
    text += f"🛡️ **Mốc Trailing Stop đề xuất (2x ATR)**: `{trailing_stop:.2f}k`\n\n"
    text += f"💡 *Gợi ý thực chiến: Nếu giá rơi gãy qua `{trailing_stop:.2f}k`, hãy tiến hành chốt lời hiện thực hóa lợi nhuận! Lệnh cài nhanh: `/setstop {symbol} {trailing_stop}`*"

    await update.message.reply_text(text, parse_mode="Markdown")


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in USER_PORTFOLIOS or not USER_PORTFOLIOS[chat_id]:
        await update.message.reply_text(
            "📂 Danh mục của bạn đang trống.\n"
            "👉 Dùng `/setbuy` để canh mua hoặc `/setstop` để cài đặt cắt lỗ.",
            parse_mode="Markdown"
        )
        return

    text = "📂 **DANH MỤC ĐẦU TƯ & GIÁM SÁT REAL-TIME (TÁCH BIỆT MUA & CẮT LỖ)**\n\n"
    for symbol, pos in USER_PORTFOLIOS[chat_id].items():
        df, err = fetch_stock_data(symbol, resolution="60", count=10)
        if not err and not df.empty:
            current_price = df.iloc[-1]["Close"]
            qty = pos.get("qty", 1000)
            entry = pos["entry"]
            
            buy_list = pos.get("buy_alert_prices", [])
            stop_list = pos.get("stop_alert_prices", [])
            
            buy_str = ", ".join([f"{p}k" for p in buy_list]) if buy_list else "Chưa cài"
            stop_str = ", ".join([f"{p}k" for p in stop_list]) if stop_list else "Chưa cài"
            
            pnl_pct = ((current_price - entry) / entry) * 100
            pnl_vnd = (current_price - entry) * qty
            status = "🟢 LÃI" if pnl_pct >= 0 else "🔴 LỖ"

            text += f"🔹 **Mã: {symbol.upper()}** ({qty} cổ)\n"
            text += f"  • Giá vốn: `{entry}k` | Giá hiện tại: `{current_price:.2f}k`\n"
            text += f"  • Hiệu suất: `{pnl_pct:+.2f}%` (`{pnl_vnd:+,.0f} VNĐ`) [{status}]\n"
            text += f"  • 🎯 **Canh Mua**: `{buy_str}`\n"
            text += f"  • 🛡️ **Stop Loss**: `{stop_str}`\n"
            text += f"  • Lệnh gỡ bỏ: `/reset {symbol}`\n"
            text += "------------------------------------\n"
        else:
            text += f"🔹 **{symbol.upper()}**: Không lấy được dữ liệu.\n"

    text += "\n💡 Dùng lệnh `/listall` để xem tổng quan nhanh, `/trail <MÃ>` để tính Trailing Stop hoặc `/reset <MÃ>` để xóa."
    await update.message.reply_text(text, parse_mode="Markdown")


async def show_all_settings_func(update_or_query, context, chat_id):
    if chat_id not in USER_PORTFOLIOS or not USER_PORTFOLIOS[chat_id]:
        text = "📂 Danh mục theo dõi của bạn hiện đang trống.\n👉 Sử dụng `/setbuy` hoặc `/setstop` để bắt đầu cài đặt."
    else:
        text = "📋 **TỔNG HỢP TOÀN BỘ MÃ & MỐC CÀI ĐẶT CANH MUA / STOP LOSS**\n\n"
        for idx, (symbol, pos) in enumerate(USER_PORTFOLIOS[chat_id].items(), 1):
            entry = pos.get("entry", 0)
            qty = pos.get("qty", 1000)
            buy_list = pos.get("buy_alert_prices", [])
            stop_list = pos.get("stop_alert_prices", [])
            
            buy_str = ", ".join([f"{p}k" for p in buy_list]) if buy_list else "Chưa cài"
            stop_str = ", ".join([f"{p}k" for p in stop_list]) if stop_list else "Chưa cài"
            
            text += f"{idx}. 🔹 **Mã: {symbol.upper()}**\n"
            text += f"   • Giá vốn: `{entry}k` | Khối lượng: `{qty}`\n"
            text += f"   • 🎯 **Mốc Canh Mua**: `{buy_str}`\n"
            text += f"   • 🛡️ **Mốc Stop Loss**: `{stop_str}`\n"
            text += f"   • 🗑️ Lệnh xóa: `/reset {symbol}`\n"
            text += "------------------------------------\n"
        
        text += "\n💡 *Bot đang giám sát real-time 24/7 các mốc này!*"

    if hasattr(update_or_query, "message") and update_or_query.message:
        await update_or_query.message.reply_text(text, parse_mode="Markdown")
    else:
        await update_or_query.edit_message_text(text, parse_mode="Markdown")


async def listall_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await show_all_settings_func(update, context, chat_id)


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if not args:
        await update.message.reply_text("⚠️ Vui lòng nhập mã cần xóa! Ví dụ: `/reset PLX` hoặc `/reset ALL`.", parse_mode="Markdown")
        return

    target = args[0].upper().strip()
    if chat_id in USER_PORTFOLIOS:
        if target == "ALL":
            USER_PORTFOLIOS[chat_id] = {}
            save_portfolios()
            await update.message.reply_text("🗑️ Đã xóa toàn bộ danh mục theo dõi của bạn.")
        elif target in USER_PORTFOLIOS[chat_id]:
            del USER_PORTFOLIOS[chat_id][target]
            save_portfolios()
            await update.message.reply_text(f"🗑️ Đã gỡ bỏ mã **{target}** khỏi danh mục theo dõi.", parse_mode="Markdown")
        else:
            await update.message.reply_text(f"⚠️ Không tìm thấy mã **{target}** trong danh mục.", parse_mode="Markdown")
    else:
        await update.message.reply_text("📂 Danh mục của bạn đang trống.")


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await update.message.reply_text("🔍 Đang quét các mã siết nền giá & dòng tiền tích lũy... ⚡")
    try:
        top_stocks = run_market_screener()
        if not top_stocks:
            await msg.edit_text("⚠️ Hiện tại không có mã nào thỏa mãn tiêu chí.")
            return
            
        text = "🚀 TOP CỔ PHIẾU SIẾT NỀN TÍCH LŨY & DÒNG TIỀN TỔ CHỨC (KHUNG D)\n\n"
        for idx, item in enumerate(top_stocks, 1):
            text += f"{idx}. {item['symbol']} | Giá: {item['close']:.1f}k | RVOL: {item['rvol']:.2f}x\n"
        await msg.edit_text(text)
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi khi quét: {str(e)}")


async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    args = context.args
    if not args:
        await update.message.reply_text("⚠️ Cú pháp: `/backtest <MÃ_CP>` (VD: `/backtest VHM`)", parse_mode="Markdown")
        return
    symbol = args[0].upper().strip()
    msg = await update.message.reply_text(f"⏳ Đang chạy Backtest cho mã {symbol}...")
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
        await update.message.reply_text("⚠️ Vui lòng nhập mã hợp lệ (VD: VHM, BSR, HPG)!")
        return

    keyboard = [
        [
            InlineKeyboardButton("⏱️ Khung 1H (Giờ)", callback_data=f"{text}|60|450"),
            InlineKeyboardButton("📅 Khung 1D (Ngày)", callback_data=f"{text}|D|300"),
            InlineKeyboardButton("📈 Khung 1W (Tuần)", callback_data=f"{text}|W|250"),
        ],
    ]
    await update.message.reply_text(
        f"🔍 Đã nhận mã **{text}**. Chọn khung thời gian phân tích:",
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

    if data == "run_breakout_callback":
        await breakout_command(update, context)
        return

    if data == "run_testcung_callback":
        await testcung_command(update, context)
        return

    if data == "run_demand_callback":
        await demand_command(update, context)
        return

    if data == "run_shakeout_callback":
        await shakeout_command(update, context)
        return

    if data == "run_safety_callback":
        await safety_command(update, context)
        return

    if data == "show_all_settings":
        chat_id = query.message.chat_id
        await show_all_settings_func(query, context, chat_id)
        return

    parts = data.split("|")
    if len(parts) == 3:
        symbol, resolution, count = parts[0], parts[1], int(parts[2])
        
        msg = await query.message.reply_text(f"⏳ Đang phân tích {symbol.upper()} (Khung: {resolution})...")

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
# 5. HTTP SERVER & SELF-PING
# ==========================================
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Quant Pro Bot with Google Sheets Database is running!")

def run_http_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler)
    server.serve_forever()

def self_ping_thread():
    app_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not app_url:
        return
    while True:
        time.sleep(240)
        try:
            requests.get(app_url, timeout=10)
        except Exception:
            pass


if __name__ == "__main__":
    threading.Thread(target=run_http_server, daemon=True).start()
    threading.Thread(target=self_ping_thread, daemon=True).start()

    print("🤖 Đang khởi tạo Telegram Bot Pro...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_error_handler(global_error_handler)

    threading.Thread(target=background_scheduler_thread, args=(app,), daemon=True).start()

    # Đăng ký CommandHandler theo đúng thứ tự phễu lọc
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("testcung", testcung_command))
    app.add_handler(CommandHandler("shakeout", shakeout_command))
    app.add_handler(CommandHandler("demand", demand_command))
    app.add_handler(CommandHandler("breakout", breakout_command))
    app.add_handler(CommandHandler("autobuy", autobuy_command))
    app.add_handler(CommandHandler("safety", safety_command))
    app.add_handler(CommandHandler("backtest", backtest_command))
    app.add_handler(CommandHandler("setbuy", setbuy_command))
    app.add_handler(CommandHandler("setstop", setstop_command))
    app.add_handler(CommandHandler("trail", trail_command))
    app.add_handler(CommandHandler("portfolio", portfolio_command))
    app.add_handler(CommandHandler("listall", listall_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    print("🚀 Bot Quant Trading Pro đã hoàn tất nâng cấp chuẩn Top 5%!")
    app.run_polling(drop_pending_updates=True)
