import datetime
import io
import json
import logging
import os
import html
import threading
import time
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, HTTPServer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import feedparser  # Thư viện phục vụ cào tin tức RSS
from groq import Groq  # Thư viện gọi Groq AI phân tích tin tức
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
GOOGLE_SHEET_API_URL = os.environ.get("GOOGLE_SHEET_API_URL", "")
DEFAULT_NAV_VND = 500_000_000

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# ==========================================
# DANH SÁCH WATCHLIST ĐÃ TINH GỌN ("THÙNG ĐẶC")
# ==========================================
WATCHLIST_SCAN = [
    # Ngân hàng (Trụ cột tín dụng)
    "VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "STB", "TPB",
    
    # Bất động sản & KCN chất lượng cao ("Thùng đặc")
    "VHM", "VIC", "VRE", "KDH", "NLG", "KBC", "VGC", "HDG",
    
    # Chứng khoán đầu ngành
    "SSI", "VCI", "MBS", "FTS", "HCM", "VND", "SHS",
    
    # Thép & Vật liệu xây dựng cốt lõi
    "HPG", "HSG", "VCG", "HT1", "KSB", "CTD", "HHV",
    
    # Bán lẻ & Tiêu dùng thiết yếu
    "MWG", "PNJ", "FRT", "DGW", "MSN", "VNM", "SAB", "QNS",
    
    # Công nghệ & Viễn thông & Hạ tầng điện
    "FPT", "CTR", "VTP", "GEE", "REE", "POW",
    
    # Dầu khí & Năng lượng lớn
    "GAS", "PLX", "PVS", "PVD", "BSR",
    
    # Vận tải & Cảng biển & Hàng không
    "HAH", "VJC", "VSC", "GMD"
]

SECTOR_MAP = {
    "Ngân hàng": ["VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "STB", "TPB"],
    "Bất động sản & KCN": ["VHM", "VIC", "VRE", "KDH", "NLG", "KBC", "VGC", "HDG"],
    "Chứng khoán": ["SSI", "VCI", "MBS", "FTS", "HCM", "VND", "SHS"],
    "Thép & Xây dựng": ["HPG", "HSG", "VCG", "HT1", "KSB", "CTD", "HHV"],
    "Dầu khí & Hạ tầng": ["PVS", "PVD", "GAS", "PLX", "BSR", "REE", "POW"],
    "Bán lẻ & Tiêu dùng": ["MWG", "PNJ", "FRT", "DGW", "MSN", "VNM", "SAB", "QNS"],
    "Công nghệ & Viễn thông": ["FPT", "CTR", "VTP", "GEE"],
    "Vận tải & Cảng biển": ["HAH", "VJC", "VSC", "GMD"]
}

ACTIVE_CHAT_IDS = set()
USER_PORTFOLIOS = {}

VNINDEX_HEALTH_CACHE = {"data": None, "timestamp": 0}
STOCK_DATA_CACHE = {} 
executor = ThreadPoolExecutor(max_workers=30)

def load_portfolios():
    global ACTIVE_CHAT_IDS
    if not GOOGLE_SHEET_API_URL:
        return {}
    try:
        res = requests.get(GOOGLE_SHEET_API_URL, timeout=3)
        if res.status_code == 200:
            result = res.json()
            if result.get("status") == "success":
                data = result.get("data", [])
                for item in data:
                    chat_id = int(item["chat_id"])
                    ACTIVE_CHAT_IDS.add(chat_id)
                return {}
    except Exception:
        pass
    return {}

USER_PORTFOLIOS = load_portfolios()

# ==========================================
# 1. HÀM CALL DATA & CHỈ BÁO NÂNG CAO (ICHIMOKU, MACD, BB, ATR, RS)
# ==========================================
def fetch_stock_data(symbol, resolution="D", count=150):
    symbol = symbol.upper().strip()
    cache_key = f"{symbol}_{resolution}_{count}"
    now_ts = time.time()

    if cache_key in STOCK_DATA_CACHE:
        cached_df, cached_err, cached_time = STOCK_DATA_CACHE[cache_key]
        if now_ts - cached_time < 60:
            return cached_df, cached_err

    if resolution == "60":
        api_res = "60"
        lookback_days = max(count * 2, 30)
    elif resolution in ["W", "M"]:
        api_res = "D"
        lookback_days = 3650 if resolution == "M" else 1825
    else:
        api_res = "D"
        lookback_days = max(count * 2, 365)

    end_time = int(now_ts)
    start_time = int(end_time - (lookback_days * 86400))

    url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution={api_res}&symbol={symbol}&from={start_time}&to={end_time}"
    
    df_raw = pd.DataFrame()
    try:
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://dchart.vndirect.com.vn/"}
        res = requests.get(url, headers=headers, timeout=5.0)
        if res.status_code == 200:
            data = res.json()
            if data.get("s") == "ok" and "t" in data and len(data["t"]) > 0:
                df_raw = pd.DataFrame({
                    "time": pd.to_datetime(data["t"], unit="s"),
                    "Open": data["o"], "High": data["h"], "Low": data["l"], "Close": data["c"], "Volume": data["v"]
                })
    except Exception as e:
        res_err = str(e)
        STOCK_DATA_CACHE[cache_key] = (pd.DataFrame(), res_err, now_ts)
        return pd.DataFrame(), res_err

    if df_raw.empty:
        res_err = "Không có dữ liệu từ máy chủ"
        STOCK_DATA_CACHE[cache_key] = (pd.DataFrame(), res_err, now_ts)
        return pd.DataFrame(), res_err

    df_raw = df_raw[["time", "Open", "High", "Low", "Close", "Volume"]].sort_values("time").reset_index(drop=True)
    
    if resolution in ["W", "M"]:
        df_raw.set_index("time", inplace=True)
        rule = "W-FRI" if resolution == "W" else "ME"
        agg_dict = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        df_resampled = df_raw.resample(rule).agg(agg_dict).dropna().reset_index()
        ret_df = df_resampled.tail(count).reset_index(drop=True)
    else:
        ret_df = df_raw.tail(count).reset_index(drop=True)

    STOCK_DATA_CACHE[cache_key] = (ret_df, None, now_ts)
    return ret_df, None

def calculate_indicators(df):
    if df.empty or len(df) < 55:
        return df

    # 1. RSI (14)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # 2. Bollinger Bands (20, 2)
    df["BB_Middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    # 3. MACD (12, 26, 9)
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_Line"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD_Line"] - df["MACD_Signal"]

    # 4. ATR (Average True Range - 14)
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    # 5. CMF & RVOL
    mf_multiplier = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-9)
    df["CMF"] = (mf_multiplier * df["Volume"]).rolling(20).sum() / (df["Volume"].rolling(20).sum() + 1e-9)
    df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
    df["RVOL"] = df["Volume"] / (df["Vol_SMA20"] + 1e-9)

    # 6. Đường trung bình
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    # 7. ICHIMOKU KINKO HYO CHUYÊN SÂU
    df["Tenkan_sen"] = (df["High"].rolling(window=9).max() + df["Low"].rolling(window=9).min()) / 2
    df["Kijun_sen"] = (df["High"].rolling(window=26).max() + df["Low"].rolling(window=26).min()) / 2
    df["Senkou_Span_A"] = (df["Tenkan_sen"] + df["Kijun_sen"]) / 2
    df["Senkou_Span_B"] = (df["High"].rolling(window=52).max() + df["Low"].rolling(window=52).min()) / 2

    return df

def calculate_relative_strength(df, symbol):
    """Tính điểm Sức mạnh Tương đối (RS) so với VNINDEX trong 60 phiên gần nhất"""
    try:
        df_index, err = fetch_stock_data("VNINDEX", resolution="D", count=len(df))
        if err or df_index.empty or len(df_index) < len(df):
            return 50 # Mặc định trung bình
        
        # Hiệu suất cổ phiếu % trong khoảng thời gian
        stock_perf = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100
        index_perf = (df_index["Close"].iloc[-1] - df_index["Close"].iloc[0]) / df_index["Close"].iloc[0] * 100
        
        diff = stock_perf - index_perf
        # Quy đổi ra điểm thang 100
        rs_score = int(min(max(50 + (diff * 2), 10), 99))
        return rs_score
    except:
        return 50

def check_bull_trap(df):
    """Phát hiện bẫy tăng giá giả (Bull Trap)"""
    if len(df) < 5: return False
    last = df.iloc[-1]
    prev = df.iloc[-2]
    # Giá vượt đỉnh 10 phiên trước nhưng CMF âm và khối lượng không tương xứng
    recent_high = df["High"].tail(10).max()
    is_breakout = last["Close"] >= recent_high * 0.99
    is_fakeout = is_breakout and (last.get("CMF", 0) < -0.02 or last.get("RVOL", 1) < 0.9)
    return is_fakeout

def calculate_volume_profile_detailed(df, bins=25):
    if df.empty or len(df) < 20:
        return 0, 0, 0, 0
    sub_df = df.tail(50)
    hist, bin_edges = np.histogram(sub_df["Close"], bins=bins, weights=sub_df["Volume"])
    poc_idx = np.argmax(hist)
    poc = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2
    
    total_vol = np.sum(hist)
    target_vol = total_vol * 0.70
    current_vol = hist[poc_idx]
    left = poc_idx
    right = poc_idx
    
    while current_vol < target_vol:
        added = False
        if right < len(hist) - 1:
            right += 1
            current_vol += hist[right]
            added = True
        if left > 0:
            left -= 1
            current_vol += hist[left]
            added = True
        if not added:
            break
            
    val = bin_edges[left]
    vah = bin_edges[right + 1]
    return round(poc, 2), round(vah, 2), round(val, 2), round(bin_edges[0], 2)

def calculate_smc_structures(df):
    if df.empty or len(df) < 20:
        return {"has_ob": False, "ob_price": 0, "has_liquidity_sweep": False}

    last_idx = len(df) - 1
    curr_close = df["Close"].iloc[-1]

    has_ob = False
    ob_price = 0.0
    for i in range(last_idx - 8, last_idx - 2):
        if df["Close"].iloc[i] < df["Open"].iloc[i]:
            if df["Close"].iloc[i + 1] > df["High"].iloc[i]:
                has_ob = True
                ob_price = df["Low"].iloc[i]
                break

    recent_low_10 = df["Low"].iloc[-12:-3].min()
    curr_low = df["Low"].iloc[-1]
    has_liquidity_sweep = (curr_low < recent_low_10) and (curr_close > recent_low_10)

    return {"has_ob": has_ob, "ob_price": round(ob_price, 2), "has_liquidity_sweep": has_liquidity_sweep}

def check_ob_confirmation_type(df, smc):
    if not smc["has_ob"]:
        return "CHƯA VỀ VÙNG OB", 0, "Giá chưa tiếp cận Order Block của tạo lập."

    last = df.iloc[-1]
    close = last["Close"]
    ob_price = smc["ob_price"]
    cmf = last.get("CMF", 0.0)
    rvol = last.get("RVOL", 1.0)
    
    is_near_ob = abs(close - ob_price) / ob_price <= 0.03

    if not is_near_ob and not smc["has_liquidity_sweep"]:
        return "QUAN SÁT THÊM", 1, f"Giá đang cách xa vùng Order Block ({ob_price:.1f}k)."

    if smc["has_liquidity_sweep"] or (last["Low"] < ob_price and close >= ob_price):
        return ("🎯 KIỂU 1: QUÉT THANH KHOẢN (LIQUIDITY SWEEP)", 3, f"Giá đã nhúng thủng/chạm Order Block ({ob_price:.1f}k) rồi rút chân mạnh mẽ.")
    elif is_near_ob and rvol < 0.9 and cmf > 0.02:
        return ("🛡️ KIỂU 2: CẠN CUNG TÍCH LŨY (NO-SUPPLY)", 2, f"Giá đang lùi về sát Order Block với khối lượng cạn kiệt (RVOL={rvol:.2f}x).")
    else:
        return ("⏳ ĐANG CHỜ TÍN HIỆU XÁC NHẬN", 1, f"Giá đã về vùng Order Block ({ob_price:.1f}k) nhưng cần thêm phiên xác nhận.")

def calculate_position_size_1pct(entry_price, stop_loss_price, total_nav_vnd=DEFAULT_NAV_VND, risk_pct=0.01):
    if entry_price <= stop_loss_price or entry_price <= 0:
        return 1000, 0.0
    risk_amount_vnd = total_nav_vnd * risk_pct
    risk_per_share_vnd = (entry_price - stop_loss_price) * 1000
    if risk_per_share_vnd <= 0: return 1000, 0.0
    raw_qty = risk_amount_vnd / risk_per_share_vnd
    suggested_qty = int(raw_qty // 100) * 100
    suggested_qty = max(suggested_qty, 100)
    actual_risk_pct_nav = ((suggested_qty * (entry_price - stop_loss_price) * 1000) / total_nav_vnd) * 100
    return suggested_qty, round(actual_risk_pct_nav, 2)

def check_vnindex_health():
    now_ts = time.time()
    if VNINDEX_HEALTH_CACHE["data"] and (now_ts - VNINDEX_HEALTH_CACHE["timestamp"] < 60):
        return VNINDEX_HEALTH_CACHE["data"]

    df, err = fetch_stock_data("VNINDEX", resolution="D", count=40)
    if err or df.empty:
        res = {"status": "NEUTRAL", "score_modifier": 1.0, "note": "VN-Index ổn định"}
        VNINDEX_HEALTH_CACHE["data"], VNINDEX_HEALTH_CACHE["timestamp"] = res, now_ts
        return res

    df = calculate_indicators(df)
    last = df.iloc[-1]
    close, ema20 = last["Close"], last.get("EMA20", last["Close"])

    if close >= ema20:
        res = {"status": "BULLISH", "score_modifier": 1.1, "note": "VN-Index ủng hộ tăng"}
    else:
        res = {"status": "SIDEWAYS", "score_modifier": 0.9, "note": "VN-Index phân hóa tích lũy"}

    VNINDEX_HEALTH_CACHE["data"], VNINDEX_HEALTH_CACHE["timestamp"] = res, now_ts
    return res

def get_foreign_trading_data():
    url = "https://api.24hmoney.vn/v1/web/index/foreign-trading?symbol=VNINDEX"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://24hmoney.vn/indices/vn-index/giao-dich-khoi-ngoai",
            "Accept": "application/json, text/plain, */*"
        }
        res = requests.get(url, headers=headers, timeout=6)
        if res.status_code == 200:
            data = res.json()
            items = []
            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], list):
                    items = data["data"]
                elif "data" in data and isinstance(data["data"], dict):
                    items = data["data"].get("list", []) or data["data"].get("stocks", [])
            elif isinstance(data, list):
                items = data

            if items:
                parsed_list = []
                for item in items:
                    ticker = item.get("ticker") or item.get("symbol") or item.get("stockCode") or ""
                    if not ticker: continue
                    buy_val = item.get("buyVal", 0) or item.get("buyValue", 0)
                    sell_val = item.get("sellVal", 0) or item.get("sellValue", 0)
                    net_val_raw = item.get("netValue", buy_val - sell_val)
                    net_val = net_val_raw / 1e9 if abs(net_val_raw) > 1e6 else net_val_raw
                    parsed_list.append({"ticker": ticker.upper(), "netValue": round(net_val, 2)})
                parsed_list.sort(key=lambda x: abs(x["netValue"]), reverse=True)
                return parsed_list, None
        return None, f"Không lấy được dữ liệu từ 24hmoney"
    except Exception as e:
        return None, str(e)

def get_latest_stock_news(symbol):
    symbol = symbol.upper().strip()
    url = f"https://news.google.com/rss/search?q=cổ+phiếu+{symbol}+when:7d&hl=vi&gl=VN&ceid=VN:vi"
    try:
        feed = feedparser.parse(url)
        if not feed.entries:
            return f"⚠️ Không tìm thấy tin tức mới về <b>{symbol}</b>."
        news_items = [f"- Tiêu đề: {entry.title}" for entry in feed.entries[:5]]
        raw_news_text = "\n".join(news_items)

        groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not groq_api_key:
            return f"📰 <b>TIN TỨC THÔ {symbol}:</b>\n\n" + raw_news_text

        client = Groq(api_key=groq_api_key)
        prompt = f"Tổng hợp tin tức mã {symbol} ngắn gọn thành Tin tốt, Tin xấu và Đánh giá chung dạng HTML Telegram:\n{raw_news_text}"
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ Lỗi Groq AI tin tức: {html.escape(str(e))}"

# ==========================================
# HÀM PHÂN TÍCH ĐA KHUNG VÀ ĐỌC VỊ BẰNG GROQ AI
# ==========================================
def get_groq_comprehensive_analysis(symbol):
    symbol = symbol.upper().strip()
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            df_d, err_d = fetch_stock_data(symbol, "D", 60)
            df_w, err_w = fetch_stock_data(symbol, "W", 30)
            df_m, err_m = fetch_stock_data(symbol, "M", 24)
            
            if err_d or df_d.empty:
                return f"❌ Không lấy được dữ liệu phân tích kỹ thuật cho mã {symbol}."

            df_d = calculate_indicators(df_d)
            last_d = df_d.iloc[-1]
            rs_score = calculate_relative_strength(df_d, symbol)
            
            data_summary = f"""
Mã cổ phiếu: {symbol}
- Giá hiện tại: {last_d['Close']:.1f}k
- Sức mạnh tương đối (RS Rating): {rs_score}/100
- Khung Ngày (D): CMF={last_d.get('CMF', 0):.2f}, RVOL={last_d.get('RVOL', 1):.2f}x, RSI={last_d.get('RSI', 50):.1f}
- Ichimoku (D): Tenkan={last_d.get('Tenkan_sen', 0):.1f}, Kijun={last_d.get('Kijun_sen', 0):.1f}, SpanA={last_d.get('Senkou_Span_A', 0):.1f}, SpanB={last_d.get('Senkou_Span_B', 0):.1f}
- Biên độ dao động gần nhất (D): Cao nhất={df_d['High'].tail(10).max():.1f}k, Thấp nhất={df_d['Low'].tail(10).min():.1f}k
- Khung Tuần (W): {df_w['Close'].tail(12).tolist() if not df_w.empty else 'N/A'}
- Khung Tháng (M): Đỉnh={df_m['High'].max() if not df_m.empty else 0:.1f}k, Đáy={df_m['Low'].min() if not df_m.empty else 0:.1f}k
"""

            groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
            if not groq_api_key:
                return f"⚠️ Lỗi hệ thống: Biến môi trường GROQ_API_KEY chưa được cấu hình.\n\n{data_summary}"

            client = Groq(api_key=groq_api_key)
            prompt = f"""
Bạn là chuyên gia phân tích kỹ thuật chứng khoán. Dựa vào dữ liệu và hệ thống bộ lọc nâng cao của mã {symbol}:
{data_summary}

Hãy viết một bản phân tích cực kỳ chi tiết, sắc bén gồm các phần:
1. ĐỐI CHIẾU VÒNG LẶP & MÂY ICHIMOKU (KHUNG TUẦN & NGÀY)
2. BẢN ĐỐI CHIẾU ĐA KHUNG (THÁNG vs NGÀY) & CHỈ SỐ RS
3. ĐỌC VỊ MÁNH KHÓE TẠO LẬP & HÀNH VI TRONG PHIÊN (BẪY GIÁ / TRAP)
4. KỊCH BẢN THỰC CHIẾN & QUẢN TRỊ RỦI RO

YÊU CẦU BẮT BUỘC: 
- KHÔNG sử dụng bất kỳ thẻ HTML nào ngoại trừ cặp thẻ <b> dan </b> cho tiêu đề hoặc từ khóa quan trọng. 
- TUYỆT ĐỐI KHÔNG dùng ký tự < hoặc > đơn lẻ trong văn bản.
- Không dùng dấu sao (**). Viết trọn vẹn, không cắt cụt câu.
"""

            completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            return completion.choices[0].message.content

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                return f"❌ Lỗi kết nối AI sau {max_retries} lần thử: {str(e)}"

# ==========================================
# 2. HÀM PHÂN TÍCH KỸ THUẬT & VẼ BIỂU ĐỒ 2 TẦNG (TÍCH HỢP ICHIMOKU & RS & TRAP DETECTOR)
# ==========================================
def analyze_stock(symbol, resolution="D", count=200):
    def fmt(p): return f"{p:.1f}k"

    df, err = fetch_stock_data(symbol, resolution=resolution, count=count)
    if err or df.empty:
        return None, None, None, None, err if err else f"Không lấy được dữ liệu cho mã {symbol}!"

    df = calculate_indicators(df)
    smc = calculate_smc_structures(df)
    mkt_health = check_vnindex_health()
    rs_score = calculate_relative_strength(df, symbol)
    is_trap = check_bull_trap(df)

    last = df.iloc[-1]
    close = last["Close"]
    cmf = last.get("CMF", 0.0)
    rvol = last.get("RVOL", 1.0)
    rsi = last.get("RSI", 50.0)
    atr = last.get("ATR", close * 0.03)
    macd = last.get("MACD_Line", 0.0)
    macd_signal = last.get("MACD_Signal", 0.0)
    sma200 = last.get("SMA200", close)

    tenkan = last.get("Tenkan_sen", close)
    kijun = last.get("Kijun_sen", close)
    span_a = last.get("Senkou_Span_A", close)
    span_b = last.get("Senkou_Span_B", close)
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)

    poc, vah, val, _ = calculate_volume_profile_detailed(df)
    
    is_near_sma200 = (close >= sma200 * 0.97) and (close < sma200)
    is_downtrend_heavy = (close < sma200) and not (is_near_sma200 and cmf > 0.03)

    confluence_points = []
    score = 50

    # Chấm điểm dựa trên các bộ lọc nâng cao
    if rs_score >= 75: score += 15; confluence_points.append(f"⭐ Sức mạnh RS vượt trội ({rs_score}/100)")
    elif rs_score >= 50: score += 5; confluence_points.append(f"RS trung bình ({rs_score}/100)")
    else: score -= 10; confluence_points.append(f"⚠️ Sức mạnh RS yếu ({rs_score}/100)")

    if is_trap: 
        score -= 25; confluence_points.append("🚨 CẢNH BÁO BẪY TĂNG GIÁ (BULL TRAP)")

    if close > cloud_top: 
        score += 15; confluence_points.append("☁️ Giá vượt trên mây Kumo")
    elif close >= cloud_bottom: 
        score += 8; confluence_points.append("☁️ Giá nằm trong mây Kumo")
    else: 
        score -= 10; confluence_points.append("⚠️ Giá nằm dưới mây Kumo")
        
    if tenkan > kijun:
        score += 10; confluence_points.append("⚔️ Tenkan cắt lên Kijun")
    
    if macd > macd_signal: score += 10; confluence_points.append(f"MACD cắt lên tín hiệu")
    if smc["has_ob"]: score += 10; confluence_points.append(f"🔥 [Order Block tại {fmt(smc['ob_price'])}]")
    if smc["has_liquidity_sweep"]: score += 15; confluence_points.append("🎯 [Quét thanh khoản rút chân]")
    
    if cmf > 0.03: score += 15; confluence_points.append(f"Dòng tiền lớn CMF mạnh ({cmf:.2f})")
    elif cmf < -0.1: score -= 20; confluence_points.append(f"🚨 Dòng tiền rút ròng ({cmf:.2f})")

    confidence_score = min(max(int(score * mkt_health["score_modifier"]), 10), 90)

    entry_1 = round(min(close, max(val, close * 0.98)), 2)
    entry_2 = round(close, 2)
    entry_avg = (entry_1 + entry_2) / 2
    
    recent_swing_low = df["Low"].tail(10).min()
    stop_loss = round(min(entry_avg - (1.5 * atr), recent_swing_low * 0.98), 2)
    sug_qty, _ = calculate_position_size_1pct(entry_avg, stop_loss)
    ob_title, _, _ = check_ob_confirmation_type(df, smc)

    if confidence_score >= 65 and not is_downtrend_heavy and not is_trap:
        system_status = "ĐÈN XANH: MỞ MUA / TĂNG GA"
        driving_conclusion = "🛣️ <b>Trạng thái lái xe:</b> Cao tốc thông thoáng, CMF > 0. <b>ĐẠP GA MẠNH</b>."
        final_icon = f"🟢 <b>NÊN MUA (Độ tin cậy: {confidence_score}%)</b>"
        action_note = f"Mở mua quanh <code>{fmt(entry_1)} - {fmt(entry_2)}</code>, Qty: <code>{sug_qty:,} cổ</code>, Stoploss: <code>{fmt(stop_loss)}</code>."
    elif confidence_score >= 50:
        system_status = "ĐÈN VÀNG: THĂM DÒ THẬN TRỌNG"
        driving_conclusion = "🛣️ <b>Trạng thái lái xe:</b> Sương mù nhẹ, <b>GIẢM TỐC THĂM DÒ</b>."
        final_icon = f"🟡 <b>THĂM DÒ THẬN TRỌNG (Độ tin cậy: {confidence_score}%)</b>"
        action_note = f"Đi chậm, quan sát hỗ trợ, Stoploss: <code>{fmt(stop_loss)}</code>."
    else:
        system_status = "ĐÈN ĐỎ: DỪNG XE / QUAN SÁT"
        driving_conclusion = "🛣️ <b>Trạng thái lái xe:</b> Trượt dốc / Cảnh báo bẫy. <b>TẤP LỀ NGHỈ NGƠI</b>."
        final_icon = f"🔴 <b>TẤP XE VÀO LỀ (Độ tin cậy: {confidence_score}%)</b>"
        action_note = "Xu hướng rủi ro cao hoặc có bẫy giá. <b>Tuyệt đối đứng ngoài!</b>"

    text_part1 = f"📊 <b>BÁO CÁO PHÂN TÍCH QUANT & FILTERS: {symbol.upper()}</b>\n"
    text_part1 += f"🗓️ Ngày: {last['time'].strftime('%Y-%m-%d')} | Khung: {resolution} | Giá: {fmt(close)}\n"
    text_part1 += f"⭐ RS Rating: <code>{rs_score}/100</code> | Trap Alert: <code>{'🚨 CÓ BẪY' if is_trap else '✅ An toàn'}</code>\n"
    text_part1 += f"🏛️ Thị trường: <b>{html.escape(mkt_health['note'])}</b>\n\n"
    text_part1 += f"☁️ Ichimoku -> Tenkan: <code>{fmt(tenkan)}</code> | Kijun: <code>{fmt(kijun)}</code> | Mây: <code>{fmt(cloud_bottom)}-{fmt(cloud_top)}</code>\n"
    text_part1 += f"⛽ CMF: <code>{cmf:.2f}</code> | RVOL: <code>{rvol:.2f}x</code> | RSI: <code>{rsi:.1f}</code> | ATR: <code>{atr:.2f}k</code>\n"
    text_part1 += f"📦 Volume Profile -> POC: <code>{fmt(poc)}</code> | VAH: <code>{fmt(vah)}</code> | VAL: <code>{fmt(val)}</code>\n"
    text_part1 += f"🚛 Order Block: <code>{'Có (' + fmt(smc['ob_price']) + ')' if smc['has_ob'] else 'Chưa'}</code> | Trạng thái: <b>{ob_title}</b>\n"
    text_part1 += f"🏗️ Khuyến nghị: <b>{system_status}</b>\n• {action_note}\n\n"

    text_part2 = f"⚡ <b>HỘI TỤ BỘ LỌC:</b> {'; '.join(confluence_points)}\n\n{driving_conclusion}\n\n🏁 {final_icon}"

    keyboard = [
        [
            InlineKeyboardButton("⏱️ 1H", callback_data=f"{symbol}|60|300"),
            InlineKeyboardButton("📅 1D", callback_data=f"{symbol}|D|200"),
            InlineKeyboardButton("📈 1W", callback_data=f"{symbol}|W|100"),
            InlineKeyboardButton("🗓️ 1M", callback_data=f"{symbol}|M|50"),
        ],
        [
            InlineKeyboardButton("🛡️ Check Kỷ Luật Mua", callback_data=f"checkmua_cb|{symbol}"),
            InlineKeyboardButton("🤖 AI Phân Tích Toàn Diện", callback_data=f"pt_cb|{symbol}"),
        ],
        [
            InlineKeyboardButton("🎯 Lọc Đáy An Toàn", callback_data="run_dipbuy_callback"),
            InlineKeyboardButton("🏢 Dòng Tiền Ngành", callback_data="run_sectors_callback"),
        ]
    ]

    # VẼ BIỂU ĐỒ 2 TẦNG (TÍCH HỢP ICHIMOKU & CÁC VÙNG LỌC)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.8), dpi=110, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(df["time"], df["Close"], label="Giá đóng cửa", color="#111111", lw=1.3, zorder=4)
    ax1.plot(df["time"], df["Tenkan_sen"], label="Tenkan (9)", color="#00bcd4", lw=1.0, alpha=0.9)
    ax1.plot(df["time"], df["Kijun_sen"], label="Kijun (26)", color="#e91e63", lw=1.1, alpha=0.9)
    
    ax1.plot(df["time"], df["Senkou_Span_A"], color="green", alpha=0.3, lw=0.8, label="Span A")
    ax1.plot(df["time"], df["Senkou_Span_B"], color="red", alpha=0.3, lw=0.8, label="Span B")
    ax1.fill_between(df["time"], df["Senkou_Span_A"], df["Senkou_Span_B"], 
                     where=(df["Senkou_Span_A"] >= df["Senkou_Span_B"]), color="lightgreen", alpha=0.2, interpolate=True)
    ax1.fill_between(df["time"], df["Senkou_Span_A"], df["Senkou_Span_B"], 
                     where=(df["Senkou_Span_A"] < df["Senkou_Span_B"]), color="lightcoral", alpha=0.2, interpolate=True)

    if "BB_Upper" in df.columns:
        ax1.plot(df["time"], df["BB_Upper"], color="gray", linestyle="--", alpha=0.4, lw=0.7)
        ax1.plot(df["time"], df["BB_Lower"], color="gray", linestyle="--", alpha=0.4, lw=0.7)

    ax1.axhline(poc, color="purple", linestyle=":", label=f"POC: {fmt(poc)}", alpha=0.9, lw=1.2)
    ax1.axhline(vah, color="blue", linestyle="--", label=f"VAH: {fmt(vah)}", alpha=0.5, lw=1.0)
    ax1.axhline(val, color="green", linestyle="--", label=f"VAL: {fmt(val)}", alpha=0.5, lw=1.0)
    
    ax1.set_title(f"Bản Đồ GPS Quant & Ichimoku - {symbol.upper()} ({resolution})", fontsize=10, fontweight="bold", pad=8)
    ax1.legend(loc="upper left", fontsize=5.5, ncol=3, framealpha=0.8)
    ax1.grid(True, linestyle=":", alpha=0.4)

    ax2.plot(df["time"], df["MACD_Line"], label="MACD", color="blue", lw=1)
    ax2.plot(df["time"], df["MACD_Signal"], label="Signal", color="orange", lw=1)
    ax2.bar(df["time"], df["MACD_Hist"], label="Histogram", color=np.where(df["MACD_Hist"] >= 0, 'green', 'red'), alpha=0.5, width=0.8)
    ax2.legend(loc="upper left", fontsize=6, framealpha=0.8)
    ax2.grid(True, linestyle=":", alpha=0.4)

    fig.autofmt_xdate()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return text_part1, text_part2, buf, InlineKeyboardMarkup(keyboard), None

# ==========================================
# 3. CÁC HÀM QUÉT BỘ LỌC TỰ ĐỘNG
# ==========================================
def scan_super_stock(symbol):
    try:
        df_d, err_d = fetch_stock_data(symbol, resolution="D", count=200)
        if err_d or df_d.empty or len(df_d) < 50: return None
        df_d = calculate_indicators(df_d)
        last_d = df_d.iloc[-1]
        close, sma200, cmf_d, rvol_d = last_d["Close"], last_d.get("SMA200", close), last_d.get("CMF", 0.0), last_d.get("RVOL", 1.0)
        rs_score = calculate_relative_strength(df_d, symbol)
        if close < sma200 and not ((close >= sma200 * 0.97) and cmf_d > 0.03): return None
        if cmf_d < 0.03 or rvol_d < 1.2 or rs_score < 60: return None
        smc = calculate_smc_structures(df_d)
        entry_price = round(close, 2)
        stop_loss = round(min(entry_price * 0.93, df_d["Low"].tail(10).min() * 0.98), 2)
        sug_qty, _ = calculate_position_size_1pct(entry_price, stop_loss)
        score = 50 + (15 if smc["has_ob"] else 0) + (20 if smc["has_liquidity_sweep"] else 0) + (15 if cmf_d > 0.05 else 0) + (10 if rs_score >= 75 else 0)
        if score >= 70:
            return {"symbol": symbol, "close": close, "score": min(score, 98), "rs": rs_score, "entry": entry_price, "stop_loss": stop_loss, "cmf": round(cmf_d, 2), "rvol": round(rvol_d, 2), "suggested_qty": sug_qty}
    except Exception: pass
    return None

async def run_super_stock_screener_with_progress(msg):
    total = len(WATCHLIST_SCAN)
    completed, super_stocks = 0, []
    with ThreadPoolExecutor(max_workers=30) as pool:
        futures = {pool.submit(scan_super_stock, s): s for s in WATCHLIST_SCAN}
        for f in as_completed(futures):
            completed += 1
            res = f.result()
            if res: super_stocks.append(res)
    return sorted(super_stocks, key=lambda x: x["score"], reverse=True)[:5]

def scan_monthly_higher_lows(symbol):
    df, err = fetch_stock_data(symbol, resolution="M", count=48)
    if err or df.empty or len(df) < 18: return None
    df = calculate_indicators(df)
    lows = df["Low"].values
    if lows[-1] >= lows[-6:-1].min() and lows[-6:-1].min() >= lows[-12:-6].min():
        last = df.iloc[-1]
        return {"symbol": symbol.upper(), "close": round(last["Close"], 1), "cmf": round(last.get("CMF", 0.0), 2), "recent_low": round(lows[-1], 1), "prev_low": round(lows[-6:-1].min(), 1)}
    return None

async def run_monthly_higher_lows_screener_with_progress(msg):
    vn30 = ["VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "STB", "TPB", "VHM", "VIC", "VRE", "SSI", "VCI", "HPG", "HSG", "MSN", "VNM", "FPT", "REE", "GAS"]
    results = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(scan_monthly_higher_lows, s) for s in vn30]
        for f in as_completed(futures):
            res = f.result()
            if res: results.append(res)
    return results[:6]

def scan_single_dip_stock(symbol):
    df, err = fetch_stock_data(symbol, resolution="D", count=100)
    if err or df.empty or len(df) < 30: return None
    df = calculate_indicators(df)
    last = df.iloc[-1]
    close, cmf, sma200 = last["Close"], last.get("CMF", 0.0), last.get("SMA200", last["Close"])
    if close < sma200 * 0.95 or cmf <= 0: return None
    smc = calculate_smc_structures(df)
    poc, _, _, _ = calculate_volume_profile_detailed(df)
    recent_high = df["High"].tail(15).max()
    drop_pct = ((recent_high - close) / recent_high) * 100
    if drop_pct >= 4.0 and (close <= poc * 1.02 or smc["has_ob"] or smc["has_liquidity_sweep"]):
        return {"symbol": symbol, "close": close, "drop_pct": round(drop_pct, 1), "cmf": cmf, "poc": poc, "smc": smc}
    return None

async def run_dip_buy_screener_with_progress(msg):
    results = []
    with ThreadPoolExecutor(max_workers=30) as pool:
        futures = [pool.submit(scan_single_dip_stock, s) for s in WATCHLIST_SCAN]
        for f in as_completed(futures):
            res = f.result()
            if res: results.append(res)
    return results[:6]

def analyze_single_sector_stock(symbol):
    df, err = fetch_stock_data(symbol, resolution="D", count=40)
    if err or df.empty or len(df) < 15: return None
    df = calculate_indicators(df)
    last, prev = df.iloc[-1], df.iloc[-2]
    return {"symbol": symbol, "pct_change": ((last["Close"] - prev["Close"]) / prev["Close"]) * 100, "cmf": last.get("CMF", 0.0), "rvol": last.get("RVOL", 1.0)}

async def run_sector_flow_screener_with_progress(msg):
    sector_results = []
    for sector_name, symbols in SECTOR_MAP.items():
        stock_metrics = []
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(analyze_single_sector_stock, sym) for sym in symbols]
            for f in as_completed(futures):
                res = f.result()
                if res: stock_metrics.append(res)
        if stock_metrics:
            avg_cmf = np.mean([s["cmf"] for s in stock_metrics])
            sector_results.append({"sector": sector_name, "flow_pct": round(avg_cmf * 100, 1), "leader": sorted(stock_metrics, key=lambda x: x["cmf"], reverse=True)[0]["symbol"]})
    return sorted(sector_results, key=lambda x: x["flow_pct"], reverse=True)

# ==========================================
# 4. HANDLERS XỬ LÝ SỰ KIỆN BOT
# ==========================================

async def sieuco_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await (update.message or update.callback_query.message).reply_text("🚀 Đang quét Siêu Cổ Phiếu (Tích hợp RS & Mây)...", parse_mode="HTML")
    try:
        picks = await run_super_stock_screener_with_progress(msg)
        if not picks:
            await msg.edit_text("⚠️ Không tìm thấy mã đạt tiêu chuẩn.")
            return
        text = "🚀 <b>DANH SÁCH SIÊU CỔ PHIẾU BỨT PHÁ</b>\n====================================\n\n"
        for i, it in enumerate(picks, 1):
            text += f"{i}. 🔹 <b>{it['symbol']}</b> | Giá: <code>{it['close']}k</code> | RS: <code>{it['rs']}/100</code>\n   • Mua: <code>{it['entry']}k</code> | SL: <code>{it['stop_loss']}k</code> | Qty: <code>{it['suggested_qty']:,}</code>\n"
        await msg.edit_text(text, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi: {html.escape(str(e))}", parse_mode="HTML")

async def check_buy_discipline_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    target_msg = update.message or update.callback_query.message
    if not args:
        await target_msg.reply_text("⚠️ Vui lòng nhập mã. Ví dụ: <code>/checkmua TCB</code>", parse_mode="HTML")
        return
    symbol = args[0].upper().strip()
    msg = await target_msg.reply_text(f"🛡️ Kiểm tra kỷ luật mua cho <b>{symbol}</b>...", parse_mode="HTML")
    try:
        loop = asyncio.get_running_loop()
        df, err = await loop.run_in_executor(executor, fetch_stock_data, symbol, "D", 150)
        if err or df.empty:
            await msg.edit_text(f"❌ Không có dữ liệu cho <b>{symbol}</b>.", parse_mode="HTML")
            return
        df = calculate_indicators(df)
        smc = calculate_smc_structures(df)
        rs_score = calculate_relative_strength(df, symbol)
        is_trap = check_bull_trap(df)
        close = df.iloc[-1]["Close"]
        cmf = df.iloc[-1].get("CMF", 0.0)
        ob_price = smc["ob_price"] if smc["has_ob"] else 0
        is_near_ob = (ob_price > 0 and abs(close - ob_price) / ob_price <= 0.03) or smc["has_liquidity_sweep"]
        
        report = (
            f"📋 <b>CHECKLIST KỶ LUẬT MUA: {symbol}</b>\n====================================\n"
            f"• Giá: <code>{close:.1f}k</code>\n"
            f"• Sức mạnh RS: <code>{rs_score}/100</code>\n"
            f"• Cảnh báo Bẫy giá (Trap): <code>{'🚨 CÓ' if is_trap else '✅ Không'}</code>\n"
            f"• Thuộc Thùng Đặc: <code>{'✅' if symbol in WATCHLIST_SCAN else '❌'}</code>\n"
            f"• Sát vùng Order Block ({ob_price:.1f}k): <code>{'✅' if is_near_ob else '⚠️ Chưa sát'}</code>\n"
            f"• Dòng tiền CMF dương ({cmf:.2f}): <code>{'✅' if cmf > 0 else '❌'}</code>\n"
        )
        keyboard = [[InlineKeyboardButton("🤖 AI Đọc Vị & Vòng Lặp", callback_data=f"pt_cb|{symbol}")] ]
        await msg.edit_text(report, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi: {html.escape(str(e))}", parse_mode="HTML")

async def quytec_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "🚗 <b>CẨM NANG SINH TỒN: LÁI XE CHỨNG KHOÁN</b>\n====================================\n1. Giữ bình xăng dự phòng (tiền mặt).\n2. Chọn đúng làn 'thùng đặc'.\n3. Chờ tạo lập bẻ lái (rút chân OB).\n4. Cắt lỗ khắt khe 1% NAV."
    await (update.message or update.callback_query.message).reply_text(text, parse_mode="HTML")

async def autobuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await (update.message or update.callback_query.message).reply_text("🤖 Đang quét danh sách Auto-Buy...", parse_mode="HTML")
    try:
        picks = await run_super_stock_screener_with_progress(msg)
        if not picks:
            await msg.edit_text("⚠️ Không có mã đủ chuẩn an toàn.")
            return
        text = "🎯 <b>DANH SÁCH AUTO-BUY AN TOÀN</b>\n====================================\n"
        for i, it in enumerate(picks, 1):
            text += f"{i}. 🔹 <b>{it['symbol']}</b> | Giá: <code>{it['close']}k</code> | Mua: <code>{it['entry']}k</code>\n"
        await msg.edit_text(text, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi: {html.escape(str(e))}", parse_mode="HTML")

async def dipbuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await (update.message or update.callback_query.message).reply_text("🎯 Đang quét bắt đáy an toàn...", parse_mode="HTML")
    try:
        picks = await run_dip_buy_screener_with_progress(msg)
        if not picks:
            await msg.edit_text("⚠️ Chưa phát hiện mã đảo chiều đạt chuẩn.")
            return
        text = "🎯 <b>CỔ PHIẾU TÍCH LŨY / BẮT ĐÁY</b>\n====================================\n"
        for i, it in enumerate(picks, 1):
            text += f"{i}. 🔹 <b>{it['symbol']}</b> | Giảm từ đỉnh: <code>-{it['drop_pct']}%</code> | POC: <code>{it['poc']:.1f}k</code>\n"
        await msg.edit_text(text, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi: {html.escape(str(e))}", parse_mode="HTML")

async def nangday_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await (update.message or update.callback_query.message).reply_text("📈 Đang quét VN30 Nâng Đáy Khung Tháng...", parse_mode="HTML")
    try:
        picks = await run_monthly_higher_lows_screener_with_progress(msg)
        if not picks:
            await msg.edit_text("⚠️ Không có mã thỏa mãn.")
            return
        text = "📈 <b>VN30 NÂNG ĐÁY (KHUNG THÁNG)</b>\n====================================\n"
        for i, it in enumerate(picks, 1):
            text += f"{i}. 🔹 <b>{it['symbol']}</b> | Giá: <code>{it['close']}k</code> | Đáy: <code>{it['prev_low']}k ➔ {it['recent_low']}k</code>\n"
        await msg.edit_text(text, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi: {html.escape(str(e))}", parse_mode="HTML")

async def sectors_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await (update.message or update.callback_query.message).reply_text("🏢 Đang phân tích dòng tiền ngành...", parse_mode="HTML")
    try:
        sectors = await run_sector_flow_screener_with_progress(msg)
        text = "🔥 <b>DÒNG TIỀN NHÓM NGÀNH</b>\n====================================\n"
        for i, sec in enumerate(sectors, 1):
            text += f"{i}. 🏢 <b>{sec['sector']}</b>: <code>{sec['flow_pct']}%</code> | Dẫn dắt: <code>{sec['leader']}</code>\n"
        await msg.edit_text(text, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi: {html.escape(str(e))}", parse_mode="HTML")

async def foreign_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await (update.message or update.callback_query.message).reply_text("⏳ Đang cào dữ liệu khối ngoại...", parse_mode="HTML")
    try:
        loop = asyncio.get_running_loop()
        data, err = await loop.run_in_executor(executor, get_foreign_trading_data)
        if not data:
            await msg.edit_text(f"⚠️ Lỗi lấy dữ liệu khối ngoại.", parse_mode="HTML")
            return
        text = "🌍 <b>GIAO DỊCH KHỐI NGOẠI</b>\n====================================\n"
        for i, item in enumerate(data[:10], 1):
            text += f"{i}. <b>{item['ticker']}</b>: <code>{item['netValue']:+.2f} tỷ</code>\n"
        await msg.edit_text(text, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi: {html.escape(str(e))}", parse_mode="HTML")

async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("⚠️ Vui lòng nhập mã. Ví dụ: <code>/tin VCG</code>", parse_mode="HTML")
        return
    symbol = args[0].upper()
    msg = await update.message.reply_text(f"⏳ Groq AI đang phân tích tin tức <b>{symbol}</b>...", parse_mode="HTML")
    try:
        loop = asyncio.get_running_loop()
        news_text = await loop.run_in_executor(executor, get_latest_stock_news, symbol)
        await msg.edit_text(news_text, parse_mode="HTML", disable_web_page_preview=True)
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi tải tin: {html.escape(str(e))}", parse_mode="HTML")

# ==========================================
# LỆNH GỘP /phantich (hoặc /pt)
# ==========================================
async def phantich_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    target_msg = update.message or update.callback_query.message
    
    if not args:
        await target_msg.reply_text(
            "⚠️ Vui lòng nhập mã cổ phiếu cần phân tích toàn diện.\n"
            "<i>Ví dụ:</i> <code>/phantich TCB</code> hoặc <code>/pt HPG</code>",
            parse_mode="HTML"
        )
        return
        
    symbol = args[0].upper().strip()
    msg = await target_msg.reply_text(f"🤖 <b>Groq AI đang phân tích định lượng (Ichimoku, RS, Trap Detector) cho {symbol}...</b>", parse_mode="HTML")
    
    try:
        loop = asyncio.get_running_loop()
        ai_report = await loop.run_in_executor(executor, get_groq_comprehensive_analysis, symbol)
        
        cleaned_report = ai_report.replace("<br>", "\n").replace("<br/>", "\n").replace("<BR>", "\n")
        full_report = f"🎯 <b>BẢN TIN PHÂN TÍCH CHUYÊN SÂU AI: {symbol}</b>\n====================================\n\n" + cleaned_report
        chat_id = update.effective_chat.id
        
        if len(full_report) > 3900:
            chunks = [full_report[i:i+3900] for i in range(0, len(full_report), 3900)]
            await msg.edit_text(chunks[0], parse_mode="HTML")
            for chunk in chunks[1:]:
                await context.bot.send_message(chat_id=chat_id, text=chunk, parse_mode="HTML")
            
            keyboard = [
                [
                    InlineKeyboardButton("📊 Xem Biểu Đồ Kỹ Thuật", callback_data=f"{symbol}|D|200"),
                    InlineKeyboardButton("🛡️ Check Kỷ Luật Mua", callback_data=f"checkmua_cb|{symbol}")
                ]
            ]
            await context.bot.send_message(chat_id=chat_id, text="📌 <b>Hành động tiếp theo:</b>", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")
        else:
            keyboard = [
                [
                    InlineKeyboardButton("📊 Xem Biểu Đồ Kỹ Thuật", callback_data=f"{symbol}|D|200"),
                    InlineKeyboardButton("🛡️ Check Kỷ Luật Mua", callback_data=f"checkmua_cb|{symbol}")
                ]
            ]
            await msg.edit_text(full_report, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML", disable_web_page_preview=True)

    except Exception as e:
        await msg.edit_text(f"❌ Lỗi phân tích AI: {html.escape(str(e))}", parse_mode="HTML")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().upper()
    if len(text) < 3 or len(text) > 5 or not text.isalnum():
        await update.message.reply_text("⚠️ Vui lòng nhập mã hợp lệ (VD: TCB, HPG)!")
        return

    keyboard = [
        [
            InlineKeyboardButton("⏱️ 1H", callback_data=f"{text}|60|300"),
            InlineKeyboardButton("📅 1D", callback_data=f"{text}|D|200"),
            InlineKeyboardButton("📈 1W", callback_data=f"{text}|W|100"),
            InlineKeyboardButton("🗓️ 1M", callback_data=f"{text}|M|50"),
        ],
        [
            InlineKeyboardButton("🤖 AI Phân Tích Toàn Diện", callback_data=f"pt_cb|{text}"),
            InlineKeyboardButton("🛡️ Check Kỷ Luật Mua", callback_data=f"checkmua_cb|{text}")
        ]
    ]
    await update.message.reply_text(
        f"🔍 Đã nhận mã <b>{text}</b>. Chọn tính năng phân tích:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="HTML"
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "run_sieuco_callback": await sieuco_command(update, context); return
    if data == "run_autobuy_callback": await autobuy_command(update, context); return
    if data == "run_dipbuy_callback": await dipbuy_command(update, context); return
    if data == "run_nangday_callback": await nangday_command(update, context); return
    if data == "run_sectors_callback": await sectors_command(update, context); return
    if data == "run_foreign_callback": await foreign_command(update, context); return

    if data.startswith("checkmua_cb|"):
        symbol = data.split("|")[1]
        context.args = [symbol]
        await check_buy_discipline_command(update, context)
        return

    if data.startswith("pt_cb|"):
        symbol = data.split("|")[1]
        context.args = [symbol]
        await phantich_command(update, context)
        return

    parts = data.split("|")
    if len(parts) == 3:
        symbol, resolution, count = parts[0], parts[1], int(parts[2])
        msg = await query.message.reply_text(f"⚡ Đang tải dữ liệu khung {resolution} của <b>{symbol.upper()}</b>...", parse_mode="HTML")
        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(executor, analyze_stock, symbol, resolution, count),
                timeout=20.0
            )
            if not result or len(result) < 5 or result[4] is not None:
                await msg.edit_text("❌ Lỗi lấy dữ liệu phân tích từ máy chủ.", parse_mode="HTML")
                return
            text_part1, text_part2, img_buf, reply_markup, _ = result
            chat_id = query.message.chat_id
            await msg.delete()
            await context.bot.send_photo(chat_id=chat_id, photo=img_buf, caption=text_part1, parse_mode="HTML")
            await context.bot.send_message(chat_id=chat_id, text=text_part2, reply_markup=reply_markup, parse_mode="HTML")
        except asyncio.TimeoutError:
            await msg.edit_text("⏱️ Quá thời gian xử lý yêu cầu (Timeout). Vui lòng thử lại sau!", parse_mode="HTML")
        except Exception as e:
            await msg.edit_text(f"❌ Lỗi: {html.escape(str(e))}", parse_mode="HTML")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    welcome_text = (
        "👋 <b>CHÀO MỪNG BẠN ĐẾN VỚI QUANT TRADING SYSTEM PRO!</b>\n\n"
        "1️⃣ <b>TRA CỨU NHANH:</b> Gửi mã cổ phiếu bất kỳ (VD: <code>TCB</code>, <code>HPG</code>)\n"
        "2️⃣ <b>KIỂM TRA KỶ LUẬT MUA:</b> Gõ <code>/checkmua TCB</code>\n"
        "3️⃣ 🤖 <b>PHÂN TÍCH AI TOÀN DIỆN (ICHIMOKU, RS, TRAP FILTER):</b>\n"
        "   └ Gõ <code>/phantich TCB</code> hoặc <code>/pt TCB</code>\n"
        "4️⃣ <b>BỘ LỌC TỰ ĐỘNG:</b>\n"
        "   ├ 🚀 /sieuco | 🎯 /dipbuy | 📈 /nangday\n"
        "   ├ 🤖 /autobuy | 🏢 /sectors | 🌍 /khoingoai\n"
        "   └ 📜 /quytec\n"
    )
    await update.message.reply_text(welcome_text, parse_mode="HTML")

# ==========================================
# 5. SERVER & CHẠY BOT (CÓ TÍCH HỢP HÀM ĐÁNH THỨC RENDER MỖI 3 GIÂY)
# ==========================================
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Quant Pro Bot Running!")

def run_http_server():
    server_port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", server_port), SimpleHTTPRequestHandler).serve_forever()

def render_awakener_daemon():
    """Hàm tự động ping HTTP chính mình mỗi 3 giây để đánh thức Render không bị ngủ đông"""
    port = os.environ.get("PORT", "8080")
    render_external_url = os.environ.get("RENDER_EXTERNAL_URL", f"http://127.0.0.1:{port}")
    
    while True:
        try:
            requests.get(render_external_url, timeout=3)
        except Exception:
            pass
        time.sleep(3)

if __name__ == "__main__":
    # Khởi chạy HTTP Server
    threading.Thread(target=run_http_server, daemon=True).start()
    
    # ⚡ KHỞI CHẠY TIẾN TRÌNH ĐÁNH THỨC RENDER MỖI 3 GIÂY
    threading.Thread(target=render_awakener_daemon, daemon=True).start()
    
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("sieuco", sieuco_command))
    app.add_handler(CommandHandler("autobuy", autobuy_command))
    app.add_handler(CommandHandler("dipbuy", dipbuy_command))
    app.add_handler(CommandHandler("nangday", nangday_command))
    app.add_handler(CommandHandler("sectors", sectors_command))
    app.add_handler(CommandHandler("nganh", sectors_command))
    app.add_handler(CommandHandler("khoingoai", foreign_command))  
    app.add_handler(CommandHandler("tin", news_command))
    app.add_handler(CommandHandler("checkmua", check_buy_discipline_command))
    app.add_handler(CommandHandler("checkkyloimua", check_buy_discipline_command))
    app.add_handler(CommandHandler("quytec", quytec_command))
    
    # Đăng ký lệnh gộp AI mới
    app.add_handler(CommandHandler("phantich", phantich_command))
    app.add_handler(CommandHandler("pt", phantich_command))
    
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    print("🚀 Bot Quant Trading Pro đã tích hợp trọn bộ Ichimoku, RS Rating, Trap Detector và Auto-Awakener thành công!")
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)
