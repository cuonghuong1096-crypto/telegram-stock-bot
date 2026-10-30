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
            requests.post(GOOGLE_SHEET_API_URL, json=payload, timeout=10)
        except Exception as e:
            logging.error(f"Lỗi ghi portfolio lên Google Sheet: {e}")

    threading.Thread(target=_save, daemon=True).start()

USER_PORTFOLIOS = load_portfolios()
executor = ThreadPoolExecutor(max_workers=12)

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
        res = requests.get(tcbs_url, headers=headers, timeout=3.0)
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
            res = requests.get(url, headers=headers, timeout=3.0)
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
# THUẬT TOÁN BỔ SUNG: SMART MONEY CONCEPTS (SMC) & QUẢN TRỊ 1% NAV
# ==========================================
def calculate_smc_structures(df):
    """
    TÍNH TOÁN CẤU TRÚC PHẦN 2: SMART MONEY CONCEPTS (SMC)
    - Order Block (OB): Vùng cản / hỗ trợ của cá mập
    - Fair Value Gap (FVG): Khoảng trống thanh khoản chưa lấp
    - Liquidity Sweep: Quét sạch Stop-loss đáy cũ
    """
    if df.empty or len(df) < 30:
        return {"has_ob": False, "ob_price": 0, "has_fvg": False, "fvg_low": 0, "fvg_high": 0, "has_liquidity_sweep": False}

    last_idx = len(df) - 1
    curr_close = df["Close"].iloc[-1]

    # 1. Quét Fair Value Gap (FVG Tăng giá) trong 5 nến gần nhất
    has_fvg = False
    fvg_low, fvg_high = 0.0, 0.0
    for i in range(last_idx - 4, last_idx):
        if i >= 2:
            high_i_minus_2 = df["High"].iloc[i - 2]
            low_i = df["Low"].iloc[i]
            if low_i > high_i_minus_2 * 1.005:  # Khoảng trống giá rõ rệt > 0.5%
                has_fvg = True
                fvg_low, fvg_high = high_i_minus_2, low_i
                break

    # 2. Tìm Bullish Order Block (OB) gần nhất (Nến giảm cuối cùng trước đà bùng nổ)
    has_ob = False
    ob_price = 0.0
    for i in range(last_idx - 10, last_idx - 2):
        if df["Close"].iloc[i] < df["Open"].iloc[i]:  # Nến đỏ
            if df["Close"].iloc[i + 1] > df["High"].iloc[i] and df["Volume"].iloc[i + 1] > df["Vol_SMA20"].iloc[i + 1]:
                has_ob = True
                ob_price = df["Low"].iloc[i]
                break

    # 3. Liquidity Sweep (Bẫy quét Stop-loss đáy cũ - Quét Thanh khoản)
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
    """
    TÍNH TOÁN PHẦN 3: QUẢN TRỊ RỦI RO & KHỐI LƯỢNG MUA DỰA TRÊN 1% NAV
    Công thức: Qty = (NAV * 1%) / (Entry - StopLoss)
    Làm tròn xuống bội số 100 cổ phiếu chuẩn HOSE/HNX.
    """
    if entry_price <= stop_loss_price or entry_price <= 0:
        return 1000, 0.0

    risk_amount_vnd = total_nav_vnd * risk_pct  # Tiền rủi ro tối đa cho phép mất
    risk_per_share_vnd = (entry_price - stop_loss_price) * 1000  # Rủi ro trên 1 cổ phiếu (đơn vị nghìn đồng -> VNĐ)

    if risk_per_share_vnd <= 0:
        return 1000, 0.0

    raw_qty = risk_amount_vnd / risk_per_share_vnd
    suggested_qty = int(raw_qty // 100) * 100  # Làm tròn xuống lô 100 cổ

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

    with ThreadPoolExecutor(max_workers=10) as pool:
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
        return {"status": "BEARISH_RISK", "score_modifier": 0.75, "note": "VN-Index rủi ro / Xu hướng suy yếu"}
    else:
        return {"status": "SIDEWAYS", "score_modifier": 1.0, "note": f"VN-Index tích lũy ({breadth_ratio*100:.0f}% mã tăng)"}


def get_foreign_trading_flow(symbol):
    url = f"https://apifull.tcbs.com.vn/stock-insight/v1/stock/foreign-trading?ticker={symbol.upper()}&resolution=D&count=5"
    headers = {"User-Agent": "Mozilla/5.0", "X-API-KEY": TCBS_API_KEY}
    try:
        res = requests.get(url, headers=headers, timeout=2.5)
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
        res = requests.get(url, headers=headers, timeout=2.5)
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


def analyze_multi_timeframe_poc(symbol):
    try:
        df_d, err_d = fetch_stock_data(symbol, resolution="D", count=100)
        df_w, err_w = fetch_stock_data(symbol, resolution="W", count=50)

        if err_d or df_d.empty or len(df_d) < 30:
            return True, {"wave_type": "UNKNOWN", "score_bonus": 0, "is_above_weekly_poc": True}

        df_d = calculate_indicators(df_d)
        poc_d, vah_d, val_d = calculate_volume_profile(df_d)
        
        if not df_w.empty and len(df_w) >= 10:
            poc_w, vah_w, val_w = calculate_volume_profile(df_w)
        else:
            poc_w = poc_d

        last_d = df_d.iloc[-1]
        close_d = last_d["Close"]
        ema20_d = last_d.get("EMA20", close_d)
        ema50_d = last_d.get("EMA50", close_d)
        cmf_d = last_d.get("CMF", 0.0)
        macd_hist_d = last_d.get("MACD_Hist", 0.0)

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

        return is_daily_wave_bullish, {
            "wave_type": wave_type,
            "is_above_weekly_poc": is_above_weekly_poc,
            "poc_w": poc_w,
            "poc_d": poc_d,
            "score_bonus": score_bonus
        }
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

    df = calculate_indicators(df)
    last = df.iloc[-1]

    close = last["Close"]
    cmf = last.get("CMF", 0.0)
    mfi = last.get("MFI", 50.0)
    rvol = last.get("RVOL", 1.0)
    stoch_rsi = last.get("Stoch_RSI", 50.0)
    rsi = last.get("RSI", 50.0)
    ema50 = last.get("EMA50", close)
    sma200 = last.get("SMA200", close)
    lr_slope = last.get("LR_Slope", 0.0)
    atr = last.get("ATR", close * 0.03)
    adx = last.get("ADX", 25.0)
    macd_hist = last.get("MACD_Hist", 0.0)
    bb_upper = last.get("BB_Upper", close)
    bb_lower = last.get("BB_Lower", close)

    is_bb_squeeze = last.get("BB_Squeeze", False)
    is_rsi_bear_div = last.get("RSI_Bearish_Divergence", False)
    is_rsi_bull_div = last.get("RSI_Bullish_Divergence", False)
    is_out_bb = close > bb_upper

    # Phân tích SMC & Quản trị rủi ro 1% NAV
    smc = calculate_smc_structures(df)

    tenkan = last.get("Tenkan_sen", close)
    kijun = last.get("Kijun_sen", close)
    span_a = last.get("Senkou_Span_A", close)
    span_b = last.get("Senkou_Span_B", close)
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

    # Bổ sung điểm cộng từ SMC
    if smc["has_ob"]:
        score += 10
        confluence_points.append(f"🔥 [SMC - Order Block]: Đã thiết lập vùng OB hỗ trợ tại `{fmt(smc['ob_price'])}`")
    if smc["has_fvg"]:
        score += 10
        confluence_points.append(f"⚡ [SMC - Fair Value Gap]: Xuất hiện khoảng trống FVG `{fmt(smc['fvg_low'])} - {fmt(smc['fvg_high'])}`")
    if smc["has_liquidity_sweep"]:
        score += 15
        confluence_points.append("🎯 [SMC - Liquidity Sweep]: Vừa quét thanh khoản rút chân đảo chiều mạnh mẽ")

    if adx > 25:
        score += 15
        confluence_points.append(f"Xu hướng mạnh được xác nhận (ADX: {adx:.1f})")

    if cmf > 0.05 and mfi > 50:
        score += 20
        confluence_points.append(f"Dòng tiền lớn CMF ({cmf:.2f}) & MFI ({mfi:.1f}) đồng thuận")

    if is_bb_squeeze:
        score += 10
        confluence_points.append("⚡ [SIẾT CHẶT BB]: Dải Bollinger siết hẹp, chuẩn bị bùng nổ")

    is_bull_trap = (rsi < 40 and stoch_rsi > 80)
    is_overbought_full = (stoch_rsi > 90 and rsi > 70)

    trap_list = []
    if is_bull_trap: score -= 25; trap_list.append("⚡ [BẪY TĂNG GIẢ - BULL TRAP]: Kéo giá ảo.")
    if is_rsi_bear_div: score -= 20; trap_list.append("⚠️ [PHÂN KỲ ÂM RSI]: Rủi ro đảo chiều cao.")
    if is_out_bb: score -= 10; trap_list.append("⚠️ [VƯỢT BOLUNGER TOP]: Rủi ro điều chỉnh kỹ thuật.")

    if not trap_list: trap_list.append("🛡️ Không phát hiện bẫy giá nguy hiểm.")

    mkt_health = check_vnindex_health()
    score = score * mkt_health["score_modifier"]
    confidence_score = min(max(int(score), 10), 88)

    entry_1 = round(max(val, last.get("EMA20", close), close * 0.98), 2)
    entry_2 = round(close, 2)
    entry_avg = (entry_1 + entry_2) / 2
    stop_loss = round(entry_avg - (1.5 * atr), 2)

    # Tính toán Quản trị vốn 1% NAV
    sug_qty, risk_nav_pct = calculate_position_size_1pct(entry_avg, stop_loss)

    if confidence_score >= 65 and close > cloud_bottom and not is_bull_trap and not is_rsi_bear_div:
        system_status = "MỞ MUA / GIA TĂNG TỶ TRỌNG"
        final_icon = f"🟢 **NÊN MUA (Độ tin cậy: {confidence_score}%)**"
        action_note = f"Giải ngân quanh mức `{fmt(entry_1)} - {fmt(entry_2)}`, Khuyến nghị mua: `{sug_qty:,} cổ` (Rủi ro: `{risk_nav_pct}% NAV`), Stoploss tại `{fmt(stop_loss)}`."
    elif confidence_score >= 45 and not is_bull_trap:
        system_status = "THĂM DÒ / QUAN SÁT"
        final_icon = f"🟡 **THĂM DÒ / QUAN SÁT (Độ tin cậy: {confidence_score}%)**"
        action_note = f"Thăm dò nhỏ quanh mức `{fmt(entry_1)} - {fmt(entry_2)}`, Mua test: `{int(sug_qty*0.5):,} cổ`, Stoploss tại `{fmt(stop_loss)}`."
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
            InlineKeyboardButton("🔍 Test Cung (/testcung)", callback_data="run_testcung_callback"),
            InlineKeyboardButton("🚀 Cầu Chủ Động (/demand)", callback_data="run_demand_callback"),
        ],
        [
            InlineKeyboardButton("🔄 Shakeout (/shakeout)", callback_data="run_shakeout_callback"),
            InlineKeyboardButton("🛡️ Biên An Toàn (/safety)", callback_data="run_safety_callback"),
        ],
        [
            InlineKeyboardButton("🚀 Bùng Nổ (/breakout)", callback_data="run_breakout_callback"),
            InlineKeyboardButton("🤖 Điểm Mua (/autobuy)", callback_data="run_autobuy_callback"),
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
# 3. SCREENER PHỄU 5 TẦNG
# ==========================================
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

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(scan_one, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            res = future.result()
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
        close, high, low, rvol, cmf = last["Close"], last["High"], last["Low"], last.get("RVOL", 1.0), last.get("CMF", 0.0)
        price_range = high - low
        is_pinbar_or_solid = True if price_range == 0 else (close >= low + price_range * 0.5)
        if rvol <= 0.65 and is_pinbar_or_solid and cmf >= -0.1:
            score = (1.0 - rvol) * 50 + (max(0, cmf) * 20)
            return {"symbol": symbol, "close": close, "rvol": rvol, "cmf": cmf, "score": score}
        return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(scan_test_supply, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]


def run_shakeout_screener():
    results = []
    def scan_shakeout(symbol):
        df, err = fetch_stock_data(symbol, resolution="D", count=80)
        if err or df.empty or len(df) < 50: return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        if not check_liquidity_and_base(df, last): return None
        close, open_p, high, low = last["Close"], last["Open"], last["High"], last["Low"]
        rvol, cmf, ema50 = last.get("RVOL", 1.0), last.get("CMF", 0.0), last.get("EMA50", close)
        candle_range = high - low
        if candle_range == 0: return None
        lower_wick_ratio = (min(open_p, close) - low) / candle_range
        if lower_wick_ratio >= 0.45 and close >= low + candle_range * 0.45 and close >= (ema50 * 0.97) and cmf >= -0.15:
            score = (lower_wick_ratio * 40) + (max(0, cmf) * 25) + (min(rvol, 2.5) * 10)
            return {"symbol": symbol, "close": close, "lower_wick_pct": lower_wick_ratio * 100, "rvol": rvol, "cmf": cmf, "score": score}
        return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(scan_shakeout, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            res = future.result()
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
        rvol, cmf, mfi, kijun = last.get("RVOL", 1.0), last.get("CMF", 0.0), last.get("MFI", 50.0), last.get("Kijun_sen", close)
        candle_range = high - low
        if candle_range == 0: return None
        is_strong_close = ((close - low) / candle_range) >= 0.70
        was_no_supply = prev.get("RVOL", 1.0) <= 0.70
        if rvol >= 1.50 and is_strong_close and cmf >= 0.08 and close >= kijun:
            score = (rvol * 20) + (cmf * 50) + (mfi * 0.5) + (30 if was_no_supply else 0)
            return {"symbol": symbol, "close": close, "rvol": rvol, "cmf": cmf, "mfi": mfi, "was_no_supply": was_no_supply, "score": score}
        return None

    with ThreadPoolExecutor(max_workers=12) as pool:
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
        rvol, cmf, bb_upper = last.get("RVOL", 1.0), last.get("CMF", 0.0), last.get("BB_Upper", close)
        cloud_top = max(last.get("Senkou_Span_A", close), last.get("Senkou_Span_B", close))
        price_range = high - low
        is_closed_near_high = True if price_range == 0 else (close >= high - (price_range * 0.15))
        if is_closed_near_high and rvol >= 2.0 and (close >= bb_upper or close > cloud_top) and cmf > 0:
            score = ((rvol * 4.0) + (cmf * 10)) * mkt_health["score_modifier"]
            return {"symbol": symbol, "close": close, "rvol": rvol, "cmf": cmf, "score": score}
        return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(scan_breakout, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
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
        close, cmf = last["Close"], last.get("CMF", 0.0)
        score = (roe * 2.0) + max(0, (25 - pe)) + (cmf * 20)
        return {"symbol": symbol, "close": close, "pe": pe, "roe": roe, "cmf": cmf, "score": score}

    with ThreadPoolExecutor(max_workers=12) as pool:
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
        return {
            "symbol": symbol,
            "pct_change": ((close - prev_close) / prev_close) * 100,
            "cmf": last.get("CMF", 0.0),
            "rvol": last.get("RVOL", 1.0),
            "mfi": last.get("MFI", 50.0)
        }

    for sector_name, symbols in SECTOR_MAP.items():
        stock_metrics = []
        with ThreadPoolExecutor(max_workers=8) as pool:
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
        close, rvol, cmf, mfi = last["Close"], last.get("RVOL", 1.0), last.get("CMF", 0.0), last.get("MFI", 50.0)
        if rvol >= 1.15 and (cmf > 0.02 or mfi > 55):
            score = (rvol * 3.0) + (cmf * 5) + (mfi * 0.1)
            return {"symbol": symbol, "close": close, "rvol": rvol, "cmf": cmf, "mfi": mfi, "macd_hist": last.get("MACD_Hist", 0.0), "score": score}
        return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(scan_one, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]


# ==========================================
# 4. TẦNG 5: AUTOBUY PHỄU LỌC TỐI ƯU SỬ DỤNG VSA + SMC + 1% NAV
# ==========================================
def process_single_stock_for_autobuy(symbol):
    try:
        mkt_health = check_vnindex_health()
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

        # BỘ LỌC CHUYÊN GIA (CHỐNG GAP-UP & TIGHTNESS)
        is_pass_adv, adv_bonus = check_advanced_filters(df_d, last_d, symbol)
        if not is_pass_adv:
            return None

        # TÍNH TOÁN SMC (PART 2)
        smc = calculate_smc_structures(df_d)

        cmf_h, rvol_h = 0.0, 1.0
        if not err_h and not df_h.empty and len(df_h) >= 20:
            df_h = calculate_indicators(df_h)
            last_h = df_h.iloc[-1]
            cmf_h, rvol_h = last_h.get("CMF", 0.0), last_h.get("RVOL", 1.0)

        close, cmf_d, rvol_d = last_d["Close"], last_d.get("CMF", 0.0), last_d.get("RVOL", 1.0)
        macd_hist_d, rsi_d, stoch_rsi_d = last_d.get("MACD_Hist", 0.0), last_d.get("RSI", 50.0), last_d.get("Stoch_RSI", 50.0)
        is_rsi_bear_div, is_bb_squeeze = last_d.get("RSI_Bearish_Divergence", False), last_d.get("BB_Squeeze", False)
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

        # TÍNH KHỐI LƯỢNG MUA THEO 1% NAV (PART 3)
        suggested_qty, actual_risk_pct = calculate_position_size_1pct(entry_price, stop_loss)

        score = 50 + adv_bonus
        cloud_top = max(last_d.get("Senkou_Span_A", close), last_d.get("Senkou_Span_B", close))
        cloud_bottom = min(last_d.get("Senkou_Span_A", close), last_d.get("Senkou_Span_B", close))

        if close > cloud_top: score += 15
        elif close >= cloud_bottom: score += 5
        else: score -= 25

        # Bổ sung điểm cộng từ cấu trúc SMC
        if smc["has_ob"]: score += 10
        if smc["has_fvg"]: score += 10
        if smc["has_liquidity_sweep"]: score += 15

        if cmf_d > 0.03 and cmf_h > 0.01: score += 15
        elif cmf_d > 0: score += 5

        if rs_rating > 0.05: score += 15
        if is_vcp: score += 15
        if rvol_d >= 1.15: score += 10
        if macd_hist_d > 0: score += 10

        is_wave_ok, wave_info = analyze_multi_timeframe_poc(symbol)
        score += wave_info.get("score_bonus", 0)

        if rr_ratio >= 2.0: score += 15
        elif rr_ratio < 1.0: score -= 20

        is_bull_trap = (rsi_d < 40 and stoch_rsi_d > 80)
        is_overbought_bb = close > (bb_upper * 1.008)

        if is_bull_trap: score -= 30
        if is_rsi_bear_div: score -= 25
        if is_overbought_bb: score -= 20

        score = int(score * mkt_health["score_modifier"])

        if (score >= 68 and cmf_d > -0.05 and cmf_h > -0.05 and close >= cloud_bottom and not is_bull_trap and not is_rsi_bear_div and not is_overbought_bb and is_wave_ok and rr_ratio >= 1.25):
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
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(process_single_stock_for_autobuy, symbol) for symbol in WATCHLIST_SCAN]
        for future in as_completed(futures):
            result = future.result()
            if result: buy_recommendations.append(result)
    return sorted(buy_recommendations, key=lambda x: x["score"], reverse=True)[:5]


# ==========================================
# 5. BACKGROUND SCHEDULER
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
                            buy_trigger_key = f"{chat_id}_{symbol}_buy_{buy_price}_{today_date}"
                            if current_price <= buy_price and buy_trigger_key not in ALERTED_TODAY:
                                ALERTED_TODAY.add(buy_trigger_key)
                                alert_text = f"🎯 **CẢNH BÁO CANH MUA: ĐÃ CHẠM MỐC `{buy_price}k`!**\n\n• Mã cổ phiếu: **{symbol.upper()}**\n• Giá hiện tại: `{current_price:.2f}k` 📉"
                                try: application.bot.send_message(chat_id=chat_id, text=alert_text, parse_mode="Markdown")
                                except Exception: pass

                        for stop_price in pos.get("stop_alert_prices", []):
                            stop_trigger_key = f"{chat_id}_{symbol}_stop_{stop_price}_{today_date}"
                            if current_price <= stop_price and stop_trigger_key not in ALERTED_TODAY:
                                ALERTED_TODAY.add(stop_trigger_key)
                                alert_text = f"🚨 **CẢNH BÁO STOP LOSS: RƠI DƯỚI `{stop_price}k`!**\n\n• Mã cổ phiếu: **{symbol.upper()}**\n• Giá hiện tại: `{current_price:.2f}k` 📉"
                                try: application.bot.send_message(chat_id=chat_id, text=alert_text, parse_mode="Markdown")
                                except Exception: pass

        if now.weekday() < 5 and now.hour in [9, 10, 11, 13, 14] and now.minute <= 2:
            if last_hourly_scan_hour != now.hour and ACTIVE_CHAT_IDS:
                last_hourly_scan_hour = now.hour
                top_hourly = run_hourly_cashflow_screener()
                if top_hourly:
                    report_msg = f"⏱️ **BẢN TIN DÒNG TIỀN HÀNG GIỜ ({now.strftime('%H:00')} - {now.strftime('%d/%m/%Y')})**\n------------------------------------\n"
                    for idx, item in enumerate(top_hourly, 1):
                        report_msg += f"{idx}. **{item['symbol']}** | Giá: `{item['close']:.1f}k` | RVOL: `{item['rvol']:.2f}x` | CMF: `{item['cmf']:.2f}`\n"
                    for chat_id in ACTIVE_CHAT_IDS:
                        try: application.bot.send_message(chat_id=chat_id, text=report_msg, parse_mode="Markdown")
                        except Exception: pass

        if now.weekday() < 5 and now.hour == 14 and now.minute == 15:
            if ACTIVE_CHAT_IDS:
                top_stocks = run_market_screener()
                if top_stocks:
                    text = "=== BẢN TIN TỰ ĐỘNG 14:15 - CỔ PHIẾU DÒNG TIỀN ===\n\n"
                    for idx, item in enumerate(top_stocks, 1):
                        text += f"{idx}. {item['symbol']} | Giá: {item['close']:.1f}k | RVOL: {item['rvol']:.2f}x\n"
                    for chat_id in ACTIVE_CHAT_IDS:
                        try: application.bot.send_message(chat_id=chat_id, text=text)
                        except Exception: pass
            time.sleep(70)

        if now.weekday() < 5 and now.hour == 15 and now.minute == 0:
            if sent_daily_report_today != today_date and USER_PORTFOLIOS:
                sent_daily_report_today = today_date
                for chat_id, portfolio in USER_PORTFOLIOS.items():
                    if not portfolio: continue
                    report_text = f"📊 **BÁO CÁO CUỐI PHIÊN GIAO DỊCH** ({now.strftime('%d/%m/%Y')})\n------------------------------------\n"
                    total_pnl_vnd = 0
                    for symbol, pos in portfolio.items():
                        df, err = fetch_stock_data(symbol, resolution="D", count=5)
                        if not err and not df.empty:
                            current_price, qty, entry = df.iloc[-1]["Close"], pos.get("qty", 1000), pos["entry"]
                            pnl_pct = ((current_price - entry) / entry) * 100
                            pnl_vnd = (current_price - entry) * qty
                            total_pnl_vnd += pnl_vnd
                            report_text += f"🔹 **{symbol.upper()}** ({qty} cổ) | Giá vốn: `{entry}k` -> `{current_price:.2f}k` ({pnl_pct:+.2f}%)\n"
                    report_text += f"------------------------------------\n💰 **LỢI NHUẬN TỔNG**: `{total_pnl_vnd:+,.0f} VNĐ`"
                    try: application.bot.send_message(chat_id=chat_id, text=report_text, parse_mode="Markdown")
                    except Exception: pass
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
        rvol, cmf, close, ema50, atr = df["RVOL"].iloc[i], df["CMF"].iloc[i], df["Close"].iloc[i], df["EMA50"].iloc[i], df["ATR"].iloc[i]
        if not np.isnan(rvol) and not np.isnan(cmf) and (rvol > 1.2 or df["Price_Range_Pct"].iloc[i] < 0.08) and cmf > 0.02 and close > ema50:
            entry_price = close
            stoploss_price = entry_price - (1.5 * atr) if not np.isnan(atr) else entry_price * 0.935
            target_price = entry_price + (1.5 * (entry_price - stoploss_price))
            trades += 1
            trade_closed = False
            for j in range(i + 1, min(i + 11, len(df))):
                if df["Low"].iloc[j] <= stoploss_price:
                    losses += 1
                    total_return += ((stoploss_price - entry_price) / entry_price) * 100
                    trade_closed = True; break
                elif df["High"].iloc[j] >= target_price:
                    wins += 1
                    total_return += ((target_price - entry_price) / entry_price) * 100
                    trade_closed = True; break
            if not trade_closed:
                ret = ((df["Close"].iloc[min(i + 10, len(df) - 1)] - entry_price) / entry_price) * 100
                total_return += ret
                if ret > 0: wins += 1
                else: losses += 1
    if trades == 0: return f"📊 KẾT QUẢ BACKTEST: {symbol}\n\nChưa xuất hiện điểm mua đạt tiêu chuẩn."
    return f"📊 KẾT QUẢ BACKTEST CHIẾN LƯỢC: {symbol}\n• Tổng lệnh: {trades} | Thắng: {wins} | Thua: {losses}\n• Tỷ lệ thắng: {(wins / trades) * 100:.1f}% | Hiệu suất: {total_return:+.2f}%"


# ==========================================
# 6. TELEGRAM HANDLERS
# ==========================================
async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.error(f"Error update: {update} caused error: {context.error}")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    save_portfolios_async()
    welcome_text = (
        "👋 **CHÀO MỪNG BẠN ĐẾN VỚI QUANT TRADING SYSTEM PRO (VSA + SMC)!**\n\n"
        "1️⃣ **TRA CỨU NHANH:**\n"
        "   └ Gửi mã cổ phiếu bất kỳ (VD: `VHM`, `BSR`, `HPG`)\n\n"
        "2️⃣ **🎯 BỘ LỌC CHIẾN LƯỢC (VSA + SMC + 1% NAV):**\n"
        "   ├ 🔥 `/breakout`: Lướt sóng Breakout - RVOL >= 2.0x\n"
        "   ├ 🚀 `/demand`  : Bùng nổ Dòng tiền Demand\n"
        "   ├ 🔍 `/testcung`: Gom Nền Cạn Cung (No-Supply)\n"
        "   ├ 🔄 `/shakeout`: Săn Nến Rút Chân (Liquidity Sweep)\n"
        "   └ 🤖 `/autobuy` : Mua Tối Ưu Tầng 5 - Điểm mua VCP & SMC (Quản trị 1% NAV)\n\n"
        "3️⃣ **📊 BỘ LỌC DÒNG TIỀN & CƠ BẢN:**\n"
        "   ├ 🟢 `/scan`    : Quét tích lũy toàn thị trường\n"
        "   ├ 🏢 `/sectors` : Dòng tiền Nhóm Ngành\n"
        "   └ 🛡️ `/safety`  : Đầu tư Giá trị (P/E & ROE)\n\n"
        "4️⃣ **⚙️ CÀI ĐẶT CANH MUA / STOP LOSS:**\n"
        "   ├ `/setbuy BSR 29.4 28.5 1000` : Cài mốc canh mua\n"
        "   ├ `/setstop BSR 27.9`          : Cài mốc Stoploss\n"
        "   └ `/trail BSR`                 : Trailing Stop khóa lãi\n\n"
        "5️⃣ **📂 QUẢN LÝ DANH MỤC & BACKTEST:**\n"
        "   ├ `/portfolio` : Xem danh mục Real-time\n"
        "   ├ `/listall`   : Cài đặt toàn bộ mốc theo dõi\n"
        "   ├ `/reset <MÃ>`: Xóa mốc theo dõi (`/reset ALL` để xóa hết)\n"
        "   └ `/backtest <MÃ>`: Backtest hiệu suất chiến lược"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def autobuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await (update.message.reply_text("🤖 **Đang kích hoạt AI quét điểm mua VSA + SMC & Quản trị 1% NAV...** ⚡") if update.message else update.callback_query.message.reply_text("🤖 **Đang kích hoạt AI quét điểm mua VSA + SMC & Quản trị 1% NAV...** ⚡"))
    try:
        picks = run_smart_buy_screener()
        if not picks:
            await msg.edit_text("⚠️ Hiện tại chưa có mã nào thỏa mãn 100% tiêu chuẩn khắt khe (Cơ bản + VCP + SMC + RS + Quản trị R:R >= 1.25).")
            return
        text = "🎯 **DANH SÁCH CỔ PHIẾU ĐẠT ĐIỂM MUA AN TOÀN CAO NHẤT (AUTO-BUY)**\n====================================\n\n"
        for idx, item in enumerate(picks, 1):
            smc_info = item["smc"]
            smc_tag = "🔥 [SMC Liquidity Sweep]" if smc_info["has_liquidity_sweep"] else ("⚡ [SMC OB/FVG]" if smc_info["has_ob"] or smc_info["has_fvg"] else "")
            
            text += f"{idx}. 🔹 **Mã: {item['symbol'].upper()}** {smc_tag} (Độ tin cậy: `{item['score']}%`)\n"
            text += f"   • Giá: `{item['close']:.1f}k` | RS: `{item['rs_rating']:+.1f}%` | R:R: `{item['rr_ratio']}x`\n"
            text += f"   • CMF (D): `{item['cmf']:.2f}` | CMF (60m): `{item['cmf_h']:.2f}`\n"
            text += f"   • 🎯 **Vùng mua**: `{item['entry']}k` | **Chốt lời**: `{item['target']}k` | 🛡️ **Stoploss**: `{item['stop_loss']}k`\n"
            text += f"   • 💰 **Khối lượng chuẩn 1% NAV**: `{item['suggested_qty']:,} cổ` (Rủi ro: `{item['actual_risk_pct']}% NAV`)\n"
            text += f"   • Lệnh cài: `/setbuy {item['symbol']} {item['entry']} {item['entry']} {item['suggested_qty']}`\n------------------------------------\n"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e: await msg.edit_text(f"❌ Lỗi: {str(e)}")


async def breakout_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await (update.message.reply_text("🚀 **Quét BÙNG NỔ Breakout...** ⚡") if update.message else update.callback_query.message.reply_text("🚀 **Quét BÙNG NỔ Breakout...** ⚡"))
    try:
        picks = run_breakout_screener()
        if not picks: await msg.edit_text("⚠️ Chưa có cổ phiếu nào bùng nổ đủ tiêu chuẩn (RVOL >= 2.0x, đóng cửa đỉnh)."); return
        text = "🚀 **DANH SÁCH CỔ PHIẾU BÙNG NỔ (BREAKOUT)**\n====================================\n\n"
        for idx, item in enumerate(picks, 1):
            text += f"{idx}. 🔥 **Mã: {item['symbol'].upper()}** | Giá: `{item['close']:.1f}k` | RVOL: `{item['rvol']:.2f}x` | CMF: `{item['cmf']:.2f}`\n"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e: await msg.edit_text(f"❌ Lỗi: {str(e)}")


async def sectors_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await (update.message.reply_text("🏢 **Phân tích dòng tiền Nhóm Ngành...** ⚡") if update.message else update.callback_query.message.reply_text("🏢 **Phân tích dòng tiền Nhóm Ngành...** ⚡"))
    try:
        sectors = run_sector_flow_screener()
        if not sectors: await msg.edit_text("⚠️ Không lấy được dữ liệu ngành."); return
        text = "🔥 **BẢN ĐỒ DÒNG TIỀN THEO NHÓM NGÀNH**\n====================================\n\n"
        for idx, sec in enumerate(sectors, 1):
            icon = "🟢 [HÚT DÒNG TIỀN]" if (sec['avg_cmf'] > 0.05 and sec['avg_rvol'] >= 1.0) else ("🟡 [TÍCH LŨY]" if sec['avg_cmf'] > 0 else "🔴 [RÚT LUI]")
            text += f"{idx}. 🏢 **{sec['sector'].upper()}** {icon}\n   • Giá TB: `{sec['avg_pct']:+.2f}%` | RVOL: `{sec['avg_rvol']:.2f}x` | CMF: `{sec['avg_cmf']:.2f}`\n   • 👑 Leader: `{sec['leader']}`\n------------------------------------\n"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e: await msg.edit_text(f"❌ Lỗi: {str(e)}")


async def testcung_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await (update.message.reply_text("🔍 **Quét cạn cung / No-Supply...** ⚡") if update.message else update.callback_query.message.reply_text("🔍 **Quét cạn cung / No-Supply...** ⚡"))
    try:
        picks = run_test_supply_screener()
        if not picks: await msg.edit_text("⚠️ Chưa phát hiện mã cạn kiệt thanh khoản."); return
        text = "🔍 **TOP CỔ PHIẾU CẠN CUNG (TEST CUNG)**\n====================================\n\n"
        for idx, item in enumerate(picks, 1):
            text += f"{idx}. 🟢 **{item['symbol'].upper()}** | Giá: `{item['close']:.1f}k` | RVOL: `{item['rvol']:.2f}x` | CMF: `{item['cmf']:.2f}`\n"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e: await msg.edit_text(f"❌ Lỗi: {str(e)}")


async def demand_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await (update.message.reply_text("🚀 **Quét DÒNG TIỀN MUA CHỦ ĐỘNG...** ⚡") if update.message else update.callback_query.message.reply_text("🚀 **Quét DÒNG TIỀN MUA CHỦ ĐỘNG...** ⚡"))
    try:
        picks = run_demand_confirmation_screener()
        if not picks: await msg.edit_text("⚠️ Chưa phát hiện mã bùng nổ Demand rõ nét."); return
        text = "🚀 **TOP CỔ PHIẾU XÁC NHẬN CẦU CHỦ ĐỘNG (DEMAND)**\n====================================\n\n"
        for idx, item in enumerate(picks, 1):
            text += f"{idx}. 🟢 **{item['symbol'].upper()}** {'🔥 [Cạn Cung -> Bùng Nổ]' if item['was_no_supply'] else '⚡ [Cầu Mạnh]'}\n   • Giá: `{item['close']:.1f}k` | RVOL: `{item['rvol']:.2f}x` | CMF: `{item['cmf']:.2f}`\n"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e: await msg.edit_text(f"❌ Lỗi: {str(e)}")


async def shakeout_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await (update.message.reply_text("🔄 **Quét nến rút chân RỦ BỎ...** ⚡") if update.message else update.callback_query.message.reply_text("🔄 **Quét nến rút chân RỦ BỎ...** ⚡"))
    try:
        picks = run_shakeout_screener()
        if not picks: await msg.edit_text("⚠️ Chưa phát hiện mã rũ bỏ rút chân mạnh mẽ."); return
        text = "🔄 **TOP CỔ PHIẾU RỦ BỎ RÚT CHÂN (SHAKEOUT)**\n====================================\n\n"
        for idx, item in enumerate(picks, 1):
            text += f"{idx}. ⚓ **{item['symbol'].upper()}** | Giá: `{item['close']:.1f}k` | Rút chân: `{item['lower_wick_pct']:.1f}%` | RVOL: `{item['rvol']:.2f}x`\n"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e: await msg.edit_text(f"❌ Lỗi: {str(e)}")


async def safety_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await (update.message.reply_text("🛡️ **Lọc Biên An Toàn (P/E & ROE)...** ⚡") if update.message else update.callback_query.message.reply_text("🛡️ **Lọc Biên An Toàn (P/E & ROE)...** ⚡"))
    try:
        picks = run_safety_value_screener()
        if not picks: await msg.edit_text("⚠️ Chưa tìm thấy mã đạt chuẩn biên an toàn."); return
        text = "🛡️ **TOP CỔ PHIẾU CÓ BIÊN AN TOÀN VỮNG CHẮC**\n====================================\n\n"
        for idx, item in enumerate(picks, 1):
            text += f"{idx}. 💎 **{item['symbol'].upper()}** | Giá: `{item['close']:.1f}k` | ROE: `{item['roe']:.1f}%` | P/E: `{item['pe']:.1f}`\n"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e: await msg.edit_text(f"❌ Lỗi: {str(e)}")


async def setbuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 3:
        await update.message.reply_text("⚠️ Sai cú pháp! Ví dụ: `/setbuy PLX 36.2 35.5 1000`", parse_mode="Markdown")
        return
    symbol = args[0].upper().strip()
    try:
        entry_price, buy_alert_price = float(args[1]), float(args[2])
        qty = int(args[3]) if len(args) >= 4 else 1000
    except ValueError:
        await update.message.reply_text("❌ Các tham số giá/khối lượng phải là số hợp lệ!")
        return

    if chat_id not in USER_PORTFOLIOS: USER_PORTFOLIOS[chat_id] = {}
    if symbol not in USER_PORTFOLIOS[chat_id]:
        USER_PORTFOLIOS[chat_id][symbol] = {"entry": entry_price, "qty": qty, "buy_alert_prices": [], "stop_alert_prices": []}
    
    USER_PORTFOLIOS[chat_id][symbol]["entry"] = entry_price
    USER_PORTFOLIOS[chat_id][symbol]["qty"] = qty
    if buy_alert_price not in USER_PORTFOLIOS[chat_id][symbol]["buy_alert_prices"]:
        USER_PORTFOLIOS[chat_id][symbol]["buy_alert_prices"].append(buy_alert_price)

    save_portfolios_async()
    list_buys_str = ", ".join([f"{p}k" for p in USER_PORTFOLIOS[chat_id][symbol]["buy_alert_prices"]])
    await update.message.reply_text(f"✅ Đã cài CANH MUA **{symbol}**\n• Giá vốn: `{entry_price}k` | Qty: {qty:,}\n• Mốc canh mua: `{list_buys_str}`", parse_mode="Markdown")


async def setstop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("⚠️ Sai cú pháp! Ví dụ: `/setstop PLX 33.5`", parse_mode="Markdown")
        return
    symbol = args[0].upper().strip()
    try: stop_price = float(args[1])
    except ValueError: await update.message.reply_text("❌ Mốc giá cắt lỗ phải là số hợp lệ!"); return

    if chat_id not in USER_PORTFOLIOS: USER_PORTFOLIOS[chat_id] = {}
    if symbol not in USER_PORTFOLIOS[chat_id]:
        USER_PORTFOLIOS[chat_id][symbol] = {"entry": stop_price, "qty": 1000, "buy_alert_prices": [], "stop_alert_prices": []}

    if stop_price not in USER_PORTFOLIOS[chat_id][symbol]["stop_alert_prices"]:
        USER_PORTFOLIOS[chat_id][symbol]["stop_alert_prices"].append(stop_price)

    save_portfolios_async()
    list_stops_str = ", ".join([f"{p}k" for p in USER_PORTFOLIOS[chat_id][symbol]["stop_alert_prices"]])
    await update.message.reply_text(f"✅ Đã cài STOP LOSS **{symbol}**\n• Mốc cắt lỗ: `{list_stops_str}`", parse_mode="Markdown")


async def trail_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if not args: await update.message.reply_text("⚠️ Cú pháp: `/trail <MÃ_CP>`", parse_mode="Markdown"); return
    symbol = args[0].upper().strip()
    df, err = fetch_stock_data(symbol, resolution="D", count=60)
    if err or df.empty: await update.message.reply_text(f"❌ Không lấy được dữ liệu cho {symbol}."); return
    df = calculate_indicators(df)
    last = df.iloc[-1]
    close, high_60 = last["Close"], df["High"].max()
    atr = last.get("ATR", close * 0.03)
    trailing_stop = round(high_60 - (2.0 * atr), 2)
    await update.message.reply_text(f"🎯 **TRAILING STOP (KHÓA LÃI): {symbol}**\n• Giá hiện tại: `{close:.2f}k` | Đỉnh 60 phiên: `{high_60:.2f}k`\n🛡️ **Mốc Trailing Stop (2x ATR)**: `{trailing_stop:.2f}k`", parse_mode="Markdown")


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in USER_PORTFOLIOS or not USER_PORTFOLIOS[chat_id]:
        await update.message.reply_text("📂 Danh mục của bạn đang trống.", parse_mode="Markdown"); return
    text = "📂 **DANH MỤC ĐẦU TƯ REAL-TIME**\n\n"
    for symbol, pos in USER_PORTFOLIOS[chat_id].items():
        df, err = fetch_stock_data(symbol, resolution="60", count=10)
        if not err and not df.empty:
            current_price = df.iloc[-1]["Close"]
            qty, entry = pos.get("qty", 1000), pos["entry"]
            pnl_pct = ((current_price - entry) / entry) * 100
            pnl_vnd = (current_price - entry) * qty
            text += f"🔹 **{symbol.upper()}** ({qty:,} cổ) | Vốn: `{entry}k` -> Giá HT: `{current_price:.2f}k` ({pnl_pct:+.2f}% / `{pnl_vnd:+,.0f} VNĐ`)\n"
    await update.message.reply_text(text, parse_mode="Markdown")


async def show_all_settings_func(update_or_query, context, chat_id):
    if chat_id not in USER_PORTFOLIOS or not USER_PORTFOLIOS[chat_id]:
        text = "📂 Danh mục theo dõi của bạn hiện đang trống."
    else:
        text = "📋 **TOÀN BỘ MÃ & MỐC CÀI ĐẶT**\n\n"
        for idx, (symbol, pos) in enumerate(USER_PORTFOLIOS[chat_id].items(), 1):
            text += f"{idx}. 🔹 **Mã: {symbol.upper()}** | Vốn: `{pos.get('entry', 0)}k` | Qty: `{pos.get('qty', 1000):,}`\n   • Mốc Mua: `{pos.get('buy_alert_prices', [])}` | Stoploss: `{pos.get('stop_alert_prices', [])}`\n"
    if hasattr(update_or_query, "message") and update_or_query.message: await update_or_query.message.reply_text(text, parse_mode="Markdown")
    else: await update_or_query.edit_message_text(text, parse_mode="Markdown")


async def listall_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await show_all_settings_func(update, context, update.effective_chat.id)


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if not args: await update.message.reply_text("⚠️ Nhập mã cần xóa! Ví dụ: `/reset PLX` hoặc `/reset ALL`.", parse_mode="Markdown"); return
    target = args[0].upper().strip()
    if chat_id in USER_PORTFOLIOS:
        if target == "ALL":
            USER_PORTFOLIOS[chat_id] = {}
            save_portfolios_async()
            await update.message.reply_text("🗑️ Đã xóa toàn bộ danh mục theo dõi.")
        elif target in USER_PORTFOLIOS[chat_id]:
            del USER_PORTFOLIOS[chat_id][target]
            save_portfolios_async()
            await update.message.reply_text(f"🗑️ Đã gỡ bỏ mã **{target}**.", parse_mode="Markdown")


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await update.message.reply_text("🔍 Đang quét các mã siết nền giá & dòng tiền... ⚡")
    try:
        top_stocks = run_market_screener()
        if not top_stocks: await msg.edit_text("⚠️ Không có mã nào thỏa mãn tiêu chí."); return
        text = "🚀 TOP CỔ PHIẾU SIẾT NỀN TÍCH LŨY & DÒNG TIỀN (KHUNG D)\n\n"
        for idx, item in enumerate(top_stocks, 1): text += f"{idx}. {item['symbol']} | Giá: {item['close']:.1f}k | RVOL: {item['rvol']:.2f}x\n"
        await msg.edit_text(text)
    except Exception as e: await msg.edit_text(f"❌ Lỗi khi quét: {str(e)}")


async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    args = context.args
    if not args: await update.message.reply_text("⚠️ Cú pháp: `/backtest <MÃ_CP>`", parse_mode="Markdown"); return
    symbol = args[0].upper().strip()
    msg = await update.message.reply_text(f"⏳ Đang chạy Backtest cho mã {symbol}...")
    try:
        report = run_strategy_backtest(symbol)
        await msg.edit_text(report)
    except Exception as e: await msg.edit_text(f"❌ Lỗi backtest: {str(e)}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    text = update.message.text.strip().upper()
    if len(text) < 3 or len(text) > 5 or not text.isalpha():
        await update.message.reply_text("⚠️ Vui lòng nhập mã hợp lệ (VD: VHM, BSR, HPG)!")
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
    if data == "run_breakout_callback": await breakout_command(update, context); return
    if data == "run_testcung_callback": await testcung_command(update, context); return
    if data == "run_demand_callback": await demand_command(update, context); return
    if data == "run_shakeout_callback": await shakeout_command(update, context); return
    if data == "run_safety_callback": await safety_command(update, context); return
    if data == "run_sectors_callback": await sectors_command(update, context); return
    if data == "show_all_settings": await show_all_settings_func(query, context, query.message.chat_id); return

    parts = data.split("|")
    if len(parts) == 3:
        symbol, resolution, count = parts[0], parts[1], int(parts[2])
        msg = await query.message.reply_text(f"⏳ Đang phân tích {symbol.upper()} (Khung: {resolution})...")

        def background_task(): return analyze_stock(symbol, resolution=resolution, count=count)
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
# 7. SERVER & MAIN ENTRY
# ==========================================
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Quant Pro Bot is running reliably!")


def run_http_server():
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler).serve_forever()


def self_ping_thread():
    app_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not app_url: return
    while True:
        time.sleep(240)
        try: requests.get(app_url, timeout=10)
        except Exception: pass


if __name__ == "__main__":
    threading.Thread(target=run_http_server, daemon=True).start()
    threading.Thread(target=self_ping_thread, daemon=True).start()

    print("🤖 Đang khởi tạo Telegram Bot Pro (VSA + SMC)...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_error_handler(global_error_handler)

    threading.Thread(target=background_scheduler_thread, args=(app,), daemon=True).start()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("testcung", testcung_command))
    app.add_handler(CommandHandler("shakeout", shakeout_command))
    app.add_handler(CommandHandler("demand", demand_command))
    app.add_handler(CommandHandler("breakout", breakout_command))
    app.add_handler(CommandHandler("autobuy", autobuy_command))
    app.add_handler(CommandHandler("safety", safety_command))
    app.add_handler(CommandHandler("sectors", sectors_command))
    app.add_handler(CommandHandler("nganh", sectors_command))
    app.add_handler(CommandHandler("backtest", backtest_command))
    app.add_handler(CommandHandler("setbuy", setbuy_command))
    app.add_handler(CommandHandler("setstop", setstop_command))
    app.add_handler(CommandHandler("trail", trail_command))
    app.add_handler(CommandHandler("portfolio", portfolio_command))
    app.add_handler(CommandHandler("listall", listall_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    print("🚀 Bot Quant Trading Pro (VSA + SMC + 1% NAV) đã khởi chạy!")
    app.run_polling(drop_pending_updates=True)
