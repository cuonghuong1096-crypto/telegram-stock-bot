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
DEFAULT_NAV_VND = 500_000_000  # Vốn tài khoản tham chiếu (500 triệu VNĐ) cho quản trị rủi ro 1% NAV

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

# ==========================================
# QUẢN LÝ LƯU TRỮ CHUẨN HOÁ
# ==========================================
def load_portfolios():
    global ACTIVE_CHAT_IDS
    if not GOOGLE_SHEET_API_URL:
        logging.warning("Chưa cấu hình GOOGLE_SHEET_API_URL!")
        return {}
    
    try:
        res = requests.get(GOOGLE_SHEET_API_URL, timeout=5)
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
                    
                    ACTIVE_CHAT_IDS.add(chat_id)
                    
                    try:
                        alert_data = json.loads(item["alerts"])
                    except Exception:
                        alert_data = {"buy_alert_prices": [], "stop_alert_prices": []}
                    
                    if chat_id not in temp_portfolios:
                        temp_portfolios[chat_id] = {}
                    
                    if symbol != "SYSTEM_INIT":
                        temp_portfolios[chat_id][symbol] = {
                            "entry": entry,
                            "qty": qty,
                            "buy_alert_prices": alert_data.get("buy_alert_prices", []),
                            "stop_alert_prices": alert_data.get("stop_alert_prices", [])
                        }
                logging.info(f"Đã tải portfolio & {len(ACTIVE_CHAT_IDS)} Chat IDs từ Google Sheets!")
                return temp_portfolios
    except Exception as e:
        logging.error(f"Lỗi tải portfolio từ Google Sheet: {e}")
    return {}

def save_portfolios_async():
    def _save():
        if not GOOGLE_SHEET_API_URL:
            return
        try:
            rows = []
            for chat_id, portfolio in USER_PORTFOLIOS.items():
                if not portfolio:
                    rows.append([str(chat_id), "SYSTEM_INIT", 0, 0, json.dumps({})])
                else:
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
            requests.post(GOOGLE_SHEET_API_URL, json=payload, timeout=5)
        except Exception as e:
            logging.error(f"Lỗi ghi portfolio lên Google Sheet: {e}")

    threading.Thread(target=_save, daemon=True).start()

USER_PORTFOLIOS = load_portfolios()
executor = ThreadPoolExecutor(max_workers=20)

# ==========================================
# 1. HÀM CALL DATA & TÍNH TOÁN CẮT LỚP KỸ THUẬT
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
        res = requests.get(tcbs_url, headers=headers, timeout=1.5)
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
            res = requests.get(url, headers=headers, timeout=1.5)
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

    lookback = 15
    curr_close = df["Close"].iloc[-1]
    curr_rsi = df["RSI"].iloc[-1]
    prev_close_max = df["Close"].iloc[-lookback:-2].max()
    prev_rsi_max = df["RSI"].iloc[-lookback:-2].max()
    prev_close_min = df["Close"].iloc[-lookback:-2].min()
    prev_rsi_min = df["RSI"].iloc[-lookback:-2].min()

    df["RSI_Bearish_Divergence"] = (curr_close > prev_close_max) and (curr_rsi < prev_rsi_max) and (curr_rsi > 58)
    df["RSI_Bullish_Divergence"] = (curr_close < prev_close_min) and (curr_rsi > prev_rsi_min) and (curr_rsi < 42)

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

    df["Tenkan_sen"] = (df["High"].rolling(window=9).max() + df["Low"].rolling(window=9).min()) / 2
    df["Kijun_sen"] = (df["High"].rolling(window=26).max() + df["Low"].rolling(window=26).min()) / 2
    df["Senkou_Span_A"] = (df["Tenkan_sen"] + df["Kijun_sen"]) / 2
    df["Senkou_Span_B"] = (df["High"].rolling(window=52).max() + df["Low"].rolling(window=52).min()) / 2

    return df


# ==========================================
# SMART MONEY CONCEPTS (SMC) & QUẢN TRỊ 1% NAV
# ==========================================
def calculate_smc_structures(df):
    if df.empty or len(df) < 30:
        return {"has_ob": False, "ob_price": 0, "has_fvg": False, "fvg_low": 0, "fvg_high": 0, "has_liquidity_sweep": False}

    last_idx = len(df) - 1
    curr_close = df["Close"].iloc[-1]

    has_fvg = False
    fvg_low, fvg_high = 0.0, 0.0
    for i in range(last_idx - 4, last_idx):
        if i >= 2:
            high_i_minus_2 = df["High"].iloc[i - 2]
            low_i = df["Low"].iloc[i]
            if low_i > high_i_minus_2 * 1.005:
                has_fvg = True
                fvg_low, fvg_high = high_i_minus_2, low_i
                break

    has_ob = False
    ob_price = 0.0
    for i in range(last_idx - 10, last_idx - 2):
        if df["Close"].iloc[i] < df["Open"].iloc[i]:
            if df["Close"].iloc[i + 1] > df["High"].iloc[i] and df["Volume"].iloc[i + 1] > df["Vol_SMA20"].iloc[i + 1]:
                has_ob = True
                ob_price = df["Low"].iloc[i]
                break

    recent_low_10 = df["Low"].iloc[-15:-3].min()
    curr_low = df["Low"].iloc[-1]
    has_liquidity_sweep = (curr_low < recent_low_10) and (curr_close > recent_low_10)

    return {
        "has_ob": has_ob,
        "ob_price": round(ob_price, 2),
        "has_fvg": has_fvg,
        "fvg_low": round(fvg_low, 2),
        "fvg_high": round(fvg_high, 2),
        "has_liquidity_sweep": has_liquidity_sweep
    }


def calculate_position_size_1pct(entry_price, stop_loss_price, total_nav_vnd=DEFAULT_NAV_VND, risk_pct=0.01):
    if entry_price <= stop_loss_price or entry_price <= 0:
        return 1000, 0.0

    risk_amount_vnd = total_nav_vnd * risk_pct
    risk_per_share_vnd = (entry_price - stop_loss_price) * 1000

    if risk_per_share_vnd <= 0:
        return 1000, 0.0

    raw_qty = risk_amount_vnd / risk_per_share_vnd
    suggested_qty = int(raw_qty // 100) * 100

    max_capital_per_trade = total_nav_vnd * 0.30
    if (suggested_qty * entry_price * 1000) > max_capital_per_trade:
        suggested_qty = int((max_capital_per_trade / (entry_price * 1000)) // 100) * 100

    suggested_qty = max(suggested_qty, 100)
    actual_risk_pct_nav = ((suggested_qty * (entry_price - stop_loss_price) * 1000) / total_nav_vnd) * 100

    return suggested_qty, round(actual_risk_pct_nav, 2)


def calculate_volume_profile(df, bins=25):
    if df.empty or len(df) < 30:
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
    sample_watchlist = WATCHLIST_SCAN[:20]
    advancing_count, total_checked = 0, 0
    
    def check_single_breadth(sym):
        df, err = fetch_stock_data(sym, resolution="D", count=5)
        if not err and not df.empty and len(df) >= 2:
            return df.iloc[-1]["Close"] > df.iloc[-2]["Close"]
        return None

    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = [pool.submit(check_single_breadth, sym) for sym in sample_watchlist]
        for f in as_completed(futures):
            res = f.result()
            if res is not None:
                total_checked += 1
                if res:
                    advancing_count += 1

    breadth_ratio = (advancing_count / total_checked) if total_checked > 0 else 0.5

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
        if breadth_ratio < 0.35:
            return {
                "status": "FAKE_BULLISH_TRAP",
                "score_modifier": 0.65,
                "note": f"⚠️ CẢNH BÁO TRỤ KÉO (Xanh vỏ đỏ lòng)! Độ rộng chỉ {breadth_ratio*100:.0f}% mã tăng."
            }
        return {"status": "BULLISH", "score_modifier": 1.15, "note": f"VN-Index Uptrend ủng hộ ({breadth_ratio*100:.0f}% mã tăng)"}
    elif close < ema50 or macd_hist < -0.5:
        return {"status": "BEARISH_RISK", "score_modifier": 0.50, "note": "VN-Index rủi ro / Xu hướng suy yếu mạnh"}
    else:
        return {"status": "SIDEWAYS", "score_modifier": 1.0, "note": f"VN-Index tích lũy ({breadth_ratio*100:.0f}% mã tăng)"}


def get_fundamental_ratios(symbol):
    url = f"https://apifull.tcbs.com.vn/vietnam-stock/v1/stock/financial-ratio/{symbol.upper()}/historic?yearly=0&isQuarterly=1"
    headers = {"User-Agent": "Mozilla/5.0", "X-API-KEY": TCBS_API_KEY}
    try:
        res = requests.get(url, headers=headers, timeout=1.5)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                latest = data[0]
                return {
                    "pe": latest.get("priceToEarning", 15),
                    "roe": latest.get("roe", 10),
                }
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


def check_advanced_filters(df_d, last, symbol):
    if len(df_d) < 30:
        return False, 0

    close = last["Close"]
    open_price = last["Open"]
    prev_close = df_d["Close"].iloc[-2]
    
    gap_pct = ((open_price - prev_close) / prev_close) * 100
    if gap_pct > 2.5:
        return False, -50

    recent_ranges = (df_d["High"].tail(5) - df_d["Low"].tail(5)) / df_d["Close"].tail(5)
    avg_compression = recent_ranges.mean()
    
    tightness_bonus = 20 if avg_compression < 0.02 else (10 if avg_compression < 0.035 else 0)
    return True, tightness_bonus


# ==========================================
# 2. HÀM PHÂN TÍCH CHUYÊN SÂU ĐƠN MÃ
# ==========================================
def analyze_stock(symbol, resolution="D", count=250):
    def fmt(p): return f"{p:.1f}k"

    df, err = fetch_stock_data(symbol, resolution=resolution, count=count)
    if err or df.empty:
        return None, None, None, None, err if err else f"Không lấy được dữ liệu cho mã {symbol}!"

    with ThreadPoolExecutor(max_workers=3) as local_pool:
        f_ind = local_pool.submit(calculate_indicators, df)
        f_smc = local_pool.submit(calculate_smc_structures, df)
        f_mkt = local_pool.submit(check_vnindex_health)
        
        df = f_ind.result()
        smc = f_smc.result()
        mkt_health = f_mkt.result()

    last = df.iloc[-1]

    close = last["Close"]
    cmf = last.get("CMF", 0.0)
    mfi = last.get("MFI", 50.0)
    rvol = last.get("RVOL", 1.0)
    stoch_rsi = last.get("Stoch_RSI", 50.0)
    rsi = last.get("RSI", 50.0)
    ema50 = last.get("EMA50", close)
    sma200 = last.get("SMA200", close)
    atr = last.get("ATR", close * 0.03)
    adx = last.get("ADX", 25.0)
    bb_upper = last.get("BB_Upper", close)

    is_bb_squeeze = last.get("BB_Squeeze", False)
    is_rsi_bear_div = last.get("RSI_Bearish_Divergence", False)
    is_out_bb = close > bb_upper

    tenkan = last.get("Tenkan_sen", close)
    kijun = last.get("Kijun_sen", close)
    span_a = last.get("Senkou_Span_A", close)
    span_b = last.get("Senkou_Span_B", close)
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)

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

    if smc["has_ob"]:
        score += 10
        confluence_points.append(f"🔥 [SMC - Order Block]: Vùng OB hỗ trợ tại `{fmt(smc['ob_price'])}`")
    if smc["has_fvg"]:
        score += 10
        confluence_points.append(f"⚡ [SMC - Fair Value Gap]: FVG `{fmt(smc['fvg_low'])} - {fmt(smc['fvg_high'])}`")
    if smc["has_liquidity_sweep"]:
        score += 15
        confluence_points.append("🎯 [SMC - Liquidity Sweep]: Quét thanh khoản đáy cũ rút chân đảo chiều")

    if adx > 25:
        score += 15
        confluence_points.append(f"Xu hướng mạnh (ADX: {adx:.1f})")

    if cmf > 0.05 and mfi > 50:
        score += 20
        confluence_points.append(f"Dòng tiền lớn CMF ({cmf:.2f}) & MFI ({mfi:.1f}) đồng thuận")

    if is_bb_squeeze:
        score += 10
        confluence_points.append("⚡ [SIẾT CHẶT BB]: Chuẩn bị bùng nổ biến động")

    is_bull_trap = (rsi < 40 and stoch_rsi > 80)
    trap_list = []
    if is_bull_trap: score -= 25; trap_list.append("⚡ [BẪY TĂNG GIẢ - BULL TRAP]")
    if is_rsi_bear_div: score -= 20; trap_list.append("⚠️ [PHÂN KỲ ÂM RSI]")
    if is_out_bb: score -= 10; trap_list.append("⚠️ [VƯỢT BOLUNGER TOP]")

    if not trap_list: trap_list.append("🛡️ Không phát hiện bẫy giá nguy hiểm")

    score = score * mkt_health["score_modifier"]
    confidence_score = min(max(int(score), 10), 88)

    entry_1 = round(max(val, last.get("EMA20", close), close * 0.98), 2)
    entry_2 = round(close, 2)
    entry_avg = (entry_1 + entry_2) / 2
    stop_loss = round(entry_avg - (1.5 * atr), 2)

    sug_qty, risk_nav_pct = calculate_position_size_1pct(entry_avg, stop_loss)

    # ĐÁNH GIÁ ĐIỂM TỔNG HỢP (THỊ TRƯỜNG -> NGÀNH -> MÃ)
    if mkt_health["status"] in ["BEARISH_RISK", "FAKE_BULLISH_TRAP"]:
        system_status = "ĐỨNG NGOÀI (THỊ TRƯỜNG CHUNG RỦI RO)"
        final_icon = f"🔴 **ĐỨNG NGOÀI QUAN SÁT (Thị trường chung xấu)**"
        action_note = f"⚠️ Thị trường chung rủi ro. Khuyên dùng: Giữ 100% tiền mặt hoặc hạ tỷ trọng."
    elif confidence_score >= 65 and close > cloud_bottom and not is_bull_trap and not is_rsi_bear_div:
        system_status = "MỞ MUA / GIA TĂNG TỶ TRỌNG"
        final_icon = f"🟢 **NÊN MUA (Độ tin cậy: {confidence_score}%)**"
        action_note = f"Mở mua quanh `{fmt(entry_1)} - {fmt(entry_2)}`, Khối lượng: `{sug_qty:,} cổ` (Rủi ro: `{risk_nav_pct}% NAV`), Stoploss tại `{fmt(stop_loss)}`."
    elif confidence_score >= 45 and not is_bull_trap:
        system_status = "THĂM DÒ / QUAN SÁT"
        final_icon = f"🟡 **THĂM DÒ TÍCH LŨY (Độ tin cậy: {confidence_score}%)**"
        action_note = f"Thăm dò nhỏ quanh `{fmt(entry_1)} - {fmt(entry_2)}`, Mua test: `{int(sug_qty*0.5):,} cổ`, Stoploss tại `{fmt(stop_loss)}`."
    else:
        system_status = "ĐỨNG NGOÀI"
        final_icon = f"🔴 **ĐỨNG NGOÀI (Độ tin cậy: {confidence_score}%)**"
        action_note = f"Tuyệt đối không giải ngân mua mới. Stoploss tham chiếu: `{fmt(stop_loss)}`."

    conf_str = "; ".join(confluence_points) if confluence_points else "Chưa có tín hiệu hội tụ"
    warn_str = "; ".join(risk_warnings) if risk_warnings else "Không có cảnh báo nghiêm trọng"

    text_part1 = f"📊 BÁO CÁO PHÂN TÍCH KỸ THUẬT & SMC: {symbol.upper()}\n"
    text_part1 += f"🗓️ Ngày: {last['time'].strftime('%Y-%m-%d')} | Khung: {resolution} | Giá hiện tại: {fmt(close)}\n"
    text_part1 += f"🏛️ Trạng thái VN-Index: **{mkt_health['note']}**\n\n"
    text_part1 += f"🌐 1. CẤU TRÚC XU HƯỚNG & DÒNG TIỀN:\n• Xu hướng: {long_term_trend}\n• ADX: `{adx:.1f}` | CMF: {cmf:.2f} | RVOL: {rvol:.2f}x\n\n"
    text_part1 += f"🏛️ 2. CẤU TRÚC NỀN GIÁ DÒNG TIỀN LỚN (SMC):\n• Order Block (OB): `{'🟢 Có (' + fmt(smc['ob_price']) + ')' if smc['has_ob'] else '🔴 Chưa có'}`\n• FVG Imbalance: `{'🟢 Có' if smc['has_fvg'] else '🔴 Không'}` | Liquidity Sweep: `{'🔥 Có Rút Chân' if smc['has_liquidity_sweep'] else '🔴 Không'}`\n\n"
    text_part1 += f"📊 3. VOLUME PROFILE & PIVOT:\n• POC: {fmt(poc)} | VAL: {fmt(val)} | VAH: {fmt(vah)}\n\n"
    text_part1 += f"🎯 4. QUẢN TRỊ RỦI RO & QUẢN LÝ VỐN 1% NAV:\n• Khuyến nghị: {system_status}\n• {action_note}\n\n"

    text_part2 = f"⚡ 5. BẪY GIÁ & ĐIỂM HỘI TỤ:\n• Bẫy giá: {' | '.join(trap_list)}\n• Điểm cộng SMC/VSA: {conf_str}\n• Cảnh báo: {warn_str}\n\n"
    text_part2 += f"🏁 6. TỔNG KẾT CHIẾN LƯỢC:\n• {final_icon}"

    keyboard = [
        [
            InlineKeyboardButton("⏱️ Khung 1H", callback_data=f"{symbol}|60|450"),
            InlineKeyboardButton("📅 Khung 1D", callback_data=f"{symbol}|D|300"),
            InlineKeyboardButton("📈 Khung 1W", callback_data=f"{symbol}|W|250"),
        ],
        [
            InlineKeyboardButton("🎯 Lọc Đảo Chiều Đáy (/dipbuy)", callback_data="run_dipbuy_callback"),
            InlineKeyboardButton("🤖 Tín Hiệu AutoBuy (/autobuy)", callback_data="run_autobuy_callback"),
        ],
        [InlineKeyboardButton("🏢 Dòng Tiền Ngành (/sectors)", callback_data="run_sectors_callback")]
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
    ax.plot(df["time"], df["Close"], label="Giá đóng cửa", color="black", lw=1.2)
    ax.plot(df["time"], df["Tenkan_sen"], label="Tenkan", color="blue", linestyle="--", alpha=0.5)
    ax.plot(df["time"], df["Kijun_sen"], label="Kijun", color="orange", linestyle="--", alpha=0.5)
    ax.axhline(poc, color="purple", linestyle=":", label=f"POC: {fmt(poc)}", alpha=0.8)
    ax.set_title(f"Phân Tích Kỹ Thuật & SMC - {symbol.upper()} ({resolution})", fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return text_part1, text_part2, buf, InlineKeyboardMarkup(keyboard), None


# ==========================================
# 3. HÀM QUÉT ĐIỂM MUA ĐẢO CHUYỀN SAU ĐỢT GIẢM (DIP BUY / ACCUMULATION)
# ==========================================
def run_dip_buy_screener():
    """
    Sàng lọc cổ phiếu đã điều chỉnh rũ bỏ, quay về nền tích lũy / Order Block 
    và xuất hiện tín hiệu dòng tiền bắt đáy quay lại.
    """
    results = []
    mkt_health = check_vnindex_health()

    def scan_dip(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=120)
        if err or df.empty or len(df) < 40: return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        
        close, low, high = last["Close"], last["Low"], last["High"]
        cmf, rvol = last.get("CMF", 0.0), last.get("RVOL", 1.0)
        rsi, stoch_rsi = last.get("RSI", 50.0), last.get("Stoch_RSI", 50.0)
        
        # SMC & Volume Profile
        smc = calculate_smc_structures(df)
        poc, vah, val = calculate_volume_profile(df)
        
        # Mức giảm từ đỉnh gần nhất trong 20 phiên
        recent_high = df["High"].tail(20).max()
        drop_pct = ((recent_high - close) / recent_high) * 100

        # Điều kiện: Đã giảm từ đỉnh >= 5%, giá về sát hỗ trợ (POC/VAL/Order Block) và RSI không quá nóng (< 60)
        is_near_support = (close <= poc * 1.02) or smc["has_ob"] or smc["has_liquidity_sweep"]
        has_cashflow_rebound = (cmf > -0.05) and (rvol >= 0.8)

        if drop_pct >= 5.0 and is_near_support and has_cashflow_rebound and rsi < 60:
            score = (drop_pct * 2.0) + (cmf * 30) + (15 if smc["has_liquidity_sweep"] else 0) + (10 if smc["has_ob"] else 0)
            score = score * mkt_health["score_modifier"]
            return {
                "symbol": symbol, "close": close, "drop_pct": round(drop_pct, 1),
                "cmf": cmf, "rvol": rvol, "rsi": round(rsi, 1),
                "poc": poc, "smc": smc, "score": score
            }
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_dip, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]


def check_liquidity_and_base(df, last):
    close = last["Close"]
    vol_sma20 = last.get("Vol_SMA20", last["Volume"])
    return (vol_sma20 * close * 1000) >= 4_500_000_000


def run_market_screener():
    results = []
    mkt_health = check_vnindex_health()
    def scan_one(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=200)
        if err or df.empty or len(df) < 50: return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        if not check_liquidity_and_base(df, last): return None
        close, rvol, cmf, adx = last["Close"], last.get("RVOL", 1.0), last.get("CMF", 0.0), last.get("ADX", 20.0)
        ema50, price_range_pct = last.get("EMA50", close), last.get("Price_Range_Pct", 0.1)
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


def run_sector_flow_screener():
    sector_results = []
    def analyze_single_stock_for_sector(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=60)
        if err or df.empty or len(df) < 20: return None
        df = calculate_indicators(df)
        last, prev = df.iloc[-1], df.iloc[-2]
        close, prev_close = last["Close"], prev["Close"]
        return {
            "symbol": symbol,
            "pct_change": ((close - prev_close) / prev_close) * 100,
            "cmf": last.get("CMF", 0.0),
            "rvol": last.get("RVOL", 1.0),
            "mfi": last.get("MFI", 50.0)
        }

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


# ==========================================
# 4. TẦNG AUTOBUY PHỄU LỌC TỐI ƯU
# ==========================================
def process_single_stock_for_autobuy(symbol):
    try:
        mkt_health = check_vnindex_health()
        if mkt_health["status"] in ["BEARISH_RISK", "FAKE_BULLISH_TRAP"]:
            return None

        df_vni, err_vni = fetch_stock_data("VNINDEX", resolution="D", count=60)
        if err_vni or df_vni.empty: df_vni, _ = fetch_stock_data("1000", resolution="D", count=60)

        ratios = get_fundamental_ratios(symbol)
        if ratios["pe"] > 40 or ratios["roe"] < 2.0: return None

        df_d, err_d = fetch_stock_data(symbol, resolution="D", count=200)
        df_h, err_h = fetch_stock_data(symbol, resolution="60", count=60)
        if err_d or df_d.empty or len(df_d) < 50: return None

        df_d = calculate_indicators(df_d)
        last_d = df_d.iloc[-1]
        if not check_liquidity_and_base(df_d, last_d): return None

        is_pass_adv, adv_bonus = check_advanced_filters(df_d, last_d, symbol)
        if not is_pass_adv: return None

        smc = calculate_smc_structures(df_d)

        cmf_h, rvol_h = 0.0, 1.0
        if not err_h and not df_h.empty and len(df_h) >= 20:
            df_h = calculate_indicators(df_h)
            last_h = df_h.iloc[-1]
            cmf_h, rvol_h = last_h.get("CMF", 0.0), last_h.get("RVOL", 1.0)

        close, cmf_d, rvol_d = last_d["Close"], last_d.get("CMF", 0.0), last_d.get("RVOL", 1.0)
        rsi_d, stoch_rsi_d = last_d.get("RSI", 50.0), last_d.get("Stoch_RSI", 50.0)
        is_rsi_bear_div = last_d.get("RSI_Bearish_Divergence", False)
        bb_upper = last_d.get("BB_Upper", close)

        is_vcp, vcp_score = calculate_vcp_tightness(df_d)
        rs_rating = calculate_relative_strength(df_d, df_vni)
        poc, vah, val = calculate_volume_profile(df_d)
        atr = last_d.get("ATR", close * 0.03)

        entry_price = round(max(val, close * 0.98), 2)
        stop_loss = round(entry_price - (1.5 * atr), 2)
        target_price = vah if vah > (close * 1.03) else round(close * 1.08, 2)

        risk, reward = entry_price - stop_loss, target_price - entry_price
        rr_ratio = reward / (risk + 1e-9)

        suggested_qty, actual_risk_pct = calculate_position_size_1pct(entry_price, stop_loss)

        score = 50 + adv_bonus
        cloud_top = max(last_d.get("Senkou_Span_A", close), last_d.get("Senkou_Span_B", close))
        cloud_bottom = min(last_d.get("Senkou_Span_A", close), last_d.get("Senkou_Span_B", close))

        if close > cloud_top: score += 15
        elif close >= cloud_bottom: score += 5
        else: score -= 25

        if smc["has_ob"]: score += 10
        if smc["has_fvg"]: score += 10
        if smc["has_liquidity_sweep"]: score += 15

        if cmf_d > 0.03 and cmf_h > 0.01: score += 15
        elif cmf_d > 0: score += 5

        if rs_rating > 0.05: score += 15
        if is_vcp: score += 15
        if rvol_d >= 1.15: score += 10

        if rr_ratio >= 2.0: score += 15
        elif rr_ratio < 1.0: score -= 20

        is_bull_trap = (rsi_d < 40 and stoch_rsi_d > 80)
        is_overbought_bb = close > (bb_upper * 1.008)

        if is_bull_trap: score -= 30
        if is_rsi_bear_div: score -= 25
        if is_overbought_bb: score -= 20

        score = int(score * mkt_health["score_modifier"])

        if (score >= 68 and cmf_d > -0.05 and cmf_h > -0.05 and close >= cloud_bottom and not is_bull_trap and not is_rsi_bear_div and rr_ratio >= 1.25):
            return {
                "symbol": symbol, "close": close, "score": min(score, 99), "cmf": cmf_d, "cmf_h": cmf_h,
                "rvol": rvol_d, "rs_rating": round(rs_rating * 100, 1), "is_vcp": is_vcp, "rr_ratio": round(rr_ratio, 2),
                "entry": entry_price, "stop_loss": stop_loss, "target": target_price,
                "suggested_qty": suggested_qty, "actual_risk_pct": actual_risk_pct, "smc": smc
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
            if result: buy_recommendations.append(result)
    return sorted(buy_recommendations, key=lambda x: x["score"], reverse=True)[:5]


# ==========================================
# 5. TELEGRAM COMMAND HANDLERS
# ==========================================
async def dipbuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await (update.message.reply_text("🎯 **Đang lọc cổ phiếu đảo chiều từ đáy & tích lũy tốt...** ⚡") if update.message else update.callback_query.message.reply_text("🎯 **Đang lọc cổ phiếu đảo chiều từ đáy & tích lũy tốt...** ⚡"))
    try:
        picks = run_dip_buy_screener()
        if not picks:
            await msg.edit_text("⚠️ Chưa phát hiện cổ phiếu nào đảo chiều cạn cung đạt chuẩn mua hoảng loạn/tích lũy.")
            return
        text = "🎯 **DANH SÁCH CỔ PHIẾU ĐẢO CHUYỀN TỪ ĐÁY & TÍCH LŨY TỐT**\n====================================\n\n"
        for idx, item in enumerate(picks, 1):
            smc_tag = "🔥 [Quét Đáy Sweep]" if item["smc"]["has_liquidity_sweep"] else ("⚡ [Vùng OrderBlock]" if item["smc"]["has_ob"] else "📊 [Chạm Hỗ Trợ POC]")
            text += f"{idx}. 🔹 **Mã: {item['symbol'].upper()}** {smc_tag}\n"
            text += f"   • Giá: `{item['close']:.1f}k` | Đã chỉnh từ đỉnh: `-{item['drop_pct']}%`\n"
            text += f"   • CMF: `{item['cmf']:.2f}` | RVOL: `{item['rvol']:.2f}x` | RSI: `{item['rsi']}`\n"
            text += f"   • Hỗ trợ POC: `{item['poc']:.1f}k` | Tra cứu chi tiết: gõ `{item['symbol']}`\n------------------------------------\n"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi: {str(e)}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    text = update.message.text.strip().upper()
    
    # Cho phép mã nhận diện chữ và số (như HT1, BCM, VC2)
    if len(text) < 3 or len(text) > 5 or not text.isalnum():
        await update.message.reply_text("⚠️ Vui lòng nhập mã hợp lệ (VD: VHM, BSR, HT1)!")
        return

    keyboard = [
        [
            InlineKeyboardButton("⏱️ Khung 1H", callback_data=f"{text}|60|450"),
            InlineKeyboardButton("📅 Khung 1D", callback_data=f"{text}|D|300"),
            InlineKeyboardButton("📈 Khung 1W", callback_data=f"{text}|W|250"),
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

    if data == "run_autobuy_callback": await autobuy_command(update, context); return
    if data == "run_dipbuy_callback": await dipbuy_command(update, context); return
    if data == "run_sectors_callback": await sectors_command(update, context); return

    parts = data.split("|")
    if len(parts) == 3:
        symbol, resolution, count = parts[0], parts[1], int(parts[2])
        msg = await query.message.reply_text(f"⏳ Đang phân tích {symbol.upper()} (Khung: {resolution})...")

        def background_task(): return analyze_stock(symbol, resolution=resolution, count=count)
        future = executor.submit(background_task)
        try:
            text_part1, text_part2, img_buf, reply_markup, err = future.result(timeout=10)
            if err:
                await msg.edit_text(f"❌ Lỗi: {err}")
            else:
                await msg.delete()
                await query.message.reply_photo(photo=img_buf)
                await query.message.reply_text(text_part1, parse_mode="Markdown")
                await query.message.reply_text(text_part2, reply_markup=reply_markup, parse_mode="Markdown")
        except Exception as e:
            await msg.edit_text(f"❌ Lỗi hệ thống: {str(e)}")


# Các Command Standard (start, scan, autobuy, sectors...)
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    welcome_text = (
        "👋 **CHÀO MỪNG BẠN ĐẾN VỚI QUANT TRADING SYSTEM PRO (VSA + SMC)!**\n\n"
        "1️⃣ **TRA CỨU NHANH:** Gửi mã cổ phiếu bất kỳ (VD: `VHM`, `HT1`)\n"
        "2️⃣ **BỘ LỌC CHIẾN LƯỢC:**\n"
        "   ├ 🎯 `/dipbuy`  : Bắt đáy & Tích lũy sau đợt giảm\n"
        "   ├ 🤖 `/autobuy` : Tín hiệu mua chuẩn VCP + SMC\n"
        "   └ 🏢 `/sectors` : Dòng tiền Nhóm Ngành\n"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def autobuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    msg = await update.message.reply_text("🤖 **Đang quét tín hiệu Mua tối ưu (Auto-Buy)...** ⚡")
    picks = run_smart_buy_screener()
    if not picks:
        await msg.edit_text("⚠️ Thị trường chung rủi ro hoặc chưa có mã thỏa mãn 100% tiêu chí khắt khe.")
        return
    text = "🎯 **DANH SÁCH CỔ PHIẾU AUTO-BUY**\n====================================\n\n"
    for idx, item in enumerate(picks, 1):
        text += f"{idx}. 🔹 **Mã: {item['symbol'].upper()}** | Giá: `{item['close']:.1f}k` | Score: `{item['score']}%`\n"
        text += f"   • Vùng mua: `{item['entry']}k` | Stoploss: `{item['stop_loss']}k` | Qty: `{item['suggested_qty']:,} cổ`\n"
    await msg.edit_text(text, parse_mode="Markdown")

async def sectors_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🏢 **Đang phân tích dòng tiền ngành...** ⚡")
    sectors = run_sector_flow_screener()
    if not sectors: await msg.edit_text("⚠️ Không lấy được dữ liệu ngành."); return
    text = "🔥 **BẢN ĐỒ DÒNG TIỀN THEO NHÓM NGÀNH**\n====================================\n\n"
    for idx, sec in enumerate(sectors, 1):
        icon = "🟢 [HÚT DÒNG TIỀN]" if (sec['avg_cmf'] > 0.05 and sec['avg_rvol'] >= 1.0) else ("🟡 [TÍCH LŨY]" if sec['avg_cmf'] > 0 else "🔴 [RÚT LUI]")
        text += f"{idx}. 🏢 **{sec['sector'].upper()}** {icon}\n   • RVOL: `{sec['avg_rvol']:.2f}x` | CMF: `{sec['avg_cmf']:.2f}` | Leader: `{sec['leader']}`\n"
    await msg.edit_text(text, parse_mode="Markdown")

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Quant Pro Bot Running!")

def run_http_server():
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler).serve_forever()

if __name__ == "__main__":
    threading.Thread(target=run_http_server, daemon=True).start()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("autobuy", autobuy_command))
    app.add_handler(CommandHandler("dipbuy", dipbuy_command))
    app.add_handler(CommandHandler("sectors", sectors_command))
    app.add_handler(CommandHandler("nganh", sectors_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    print("🚀 Bot Quant Trading Pro đã sẵn sàng!")
    app.run_polling(drop_pending_updates=True)
