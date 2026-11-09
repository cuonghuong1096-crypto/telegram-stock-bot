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
# 1. HÀM CALL DATA & GỘP KHUNG THỜI GIAN CHUẨN (ĐÃ MỞ RỘNG KHOẢNG THỜI GIAN VÀ FIX LỖI W/M)
# ==========================================
def fetch_stock_data(symbol, resolution="D", count=150):
    symbol = symbol.upper().strip()
    
    api_resolution = "60" if resolution == "60" else "D"
    
    # Xác định khoảng thời gian lùi về quá khứ an toàn cho từng khung
    now_ts = time.time()
    if resolution == "M":
        lookback_days = max(count * 40, 3650)  # Kéo tối đa khoảng 10 năm cho khung Tháng
    elif resolution == "W":
        lookback_days = max(count * 15, 1825)  # Kéo tối đa khoảng 5 năm cho khung Tuần
    elif resolution == "60":
        lookback_days = max(count * 2, 60)     # Khung ngắn hạn 1H
    else:
        lookback_days = max(count * 2, 365)    # Khung Ngày (D)

    cache_key = f"{symbol}_{resolution}"
    if cache_key in STOCK_DATA_CACHE:
        cached_df, cached_err, cached_time = STOCK_DATA_CACHE[cache_key]
        if now_ts - cached_time < 60:
            return cached_df, cached_err

    end_time = int(now_ts)
    start_time = int(end_time - (lookback_days * 86400))

    df_raw = pd.DataFrame()
    url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution={api_resolution}&symbol={symbol}&from={start_time}&to={end_time}"
    
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
    
    # Gộp dữ liệu chuẩn xác nếu là khung Tuần (W) hoặc Tháng (M)
    if resolution in ["W", "M"]:
        df_raw.set_index("time", inplace=True)
        rule = "W-FRI" if resolution == "W" else "ME"
        
        agg_dict = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }
        df_resampled = df_raw.resample(rule).agg(agg_dict).dropna().reset_index()
        ret_df = df_resampled.tail(count).reset_index(drop=True)
    else:
        ret_df = df_raw.tail(count).reset_index(drop=True)

    STOCK_DATA_CACHE[cache_key] = (ret_df, None, now_ts)
    return ret_df, None

def calculate_indicators(df):
    if df.empty or len(df) < 30:
        return df

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["BB_Middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    mf_multiplier = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-9)
    df["CMF"] = (mf_multiplier * df["Volume"]).rolling(20).sum() / (df["Volume"].rolling(20).sum() + 1e-9)

    df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
    df["RVOL"] = df["Volume"] / (df["Vol_SMA20"] + 1e-9)

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    df["Tenkan_sen"] = (df["High"].rolling(window=9).max() + df["Low"].rolling(window=9).min()) / 2
    df["Kijun_sen"] = (df["High"].rolling(window=26).max() + df["Low"].rolling(window=26).min()) / 2
    df["Senkou_Span_A"] = (df["Tenkan_sen"] + df["Kijun_sen"]) / 2
    df["Senkou_Span_B"] = (df["High"].rolling(window=52).max() + df["Low"].rolling(window=52).min()) / 2

    return df

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

# ==========================================
# TÍCH HỢP 2 KIỂU XÁC NHẬN MUA TẠI ORDER BLOCK
# ==========================================
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
        return (
            "🎯 KIỂU 1: QUÉT THANH KHOẢN (LIQUIDITY SWEEP)",
            3,
            f"Giá đã nhúng thủng/chạm Order Block ({ob_price:.1f}k) rồi rút chân mạnh mẽ. Điểm mua uy tín!"
        )
    elif is_near_ob and rvol < 0.9 and cmf > 0.02:
        return (
            "🛡️ KIỂU 2: CẠN CUNG TÍCH LŨY (NO-SUPPLY)",
            2,
            f"Giá đang lùi về sát Order Block ({ob_price:.1f}k) với khối lượng cạn kiệt (RVOL={rvol:.2f}x) và dòng tiền CMF dương ({cmf:.2f})."
        )
    else:
        return (
            "⏳ ĐANG CHỜ TÍN HIỆU XÁC NHẬN",
            1,
            f"Giá đã về vùng Order Block ({ob_price:.1f}k) nhưng cần thêm phiên xác nhận rút chân hoặc cạn cung."
        )

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

def calculate_volume_profile(df, bins=20):
    if df.empty or len(df) < 20:
        return 0, 0, 0
    sub_df = df.tail(40)
    hist, bin_edges = np.histogram(sub_df["Close"], bins=bins, weights=sub_df["Volume"])
    poc_idx = np.argmax(hist)
    poc = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2
    return poc, bin_edges[-1], bin_edges[0]

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
        res = {"status": "BULLISH", "score_modifier": 1.1, "note": "VN-Index ủng hộ tăng (Thời tiết đẹp, đường thông thoáng)"}
    else:
        res = {"status": "SIDEWAYS", "score_modifier": 0.9, "note": "VN-Index phân hóa tích lũy (Sương mù tầm nhìn hạn chế)"}

    VNINDEX_HEALTH_CACHE["data"], VNINDEX_HEALTH_CACHE["timestamp"] = res, now_ts
    return res

# ==========================================
# HÀM CÀO DỮ LIỆU KHỐI NGOẠI CHUẨN TỪ 24HMONEY API
# ==========================================
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
                    if not ticker:
                        continue
                    
                    buy_val = item.get("buyVal", 0) or item.get("buyValue", 0)
                    sell_val = item.get("sellVal", 0) or item.get("sellValue", 0)
                    net_val_raw = item.get("netValue", buy_val - sell_val)
                    
                    net_val = net_val_raw / 1e9 if abs(net_val_raw) > 1e6 else net_val_raw
                    
                    parsed_list.append({
                        "ticker": ticker.upper(),
                        "netValue": round(net_val, 2)
                    })
                
                parsed_list.sort(key=lambda x: abs(x["netValue"]), reverse=True)
                return parsed_list, None

        return None, f"Không lấy được dữ liệu từ 24hmoney (HTTP status {res.status_code})"
    except Exception as e:
        logging.error(f"Error fetching 24hmoney data: {e}")
        return None, str(e)

# ==========================================
# HÀM CÀO & PHÂN TÍCH TIN TỨC THÔNG MINH (GROQ AI)
# ==========================================
def get_latest_stock_news(symbol):
    symbol = symbol.upper().strip()
    url = f"https://news.google.com/rss/search?q=cổ+phiếu+{symbol}+when:7d&hl=vi&gl=VN&ceid=VN:vi"
    
    try:
        feed = feedparser.parse(url)
        if not feed.entries:
            return f"⚠️ Hiện không tìm thấy bài báo mới nào gần đây về mã <b>{symbol}</b>."
            
        news_items = []
        for entry in feed.entries[:5]:
            news_items.append(f"- Tiêu đề: {entry.title} (Thời gian: {entry.get('published', 'Mới đây')})")
            
        raw_news_text = "\n".join(news_items)

        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            return f"⚠️ <b>Lỗi cấu hình:</b> Chưa nhận được biến môi trường <code>GROQ_API_KEY</code> trên Server.\n\n📰 <b>TIN TỨC THÔ {symbol}:</b>\n\n" + raw_news_text

        client = Groq(api_key=groq_api_key)
        
        prompt = f"""
Bạn là chuyên gia phân tích chứng khoán Việt Nam. Hãy đọc danh sách tin tức mới nhất về mã {symbol} dưới đây:
{raw_news_text}

Hãy tổng hợp ngắn gọn, rõ ràng thành đúng định dạng HTML Telegram như sau:

🟢 <b>TIN TÍCH CỰC (TIN TỐT):</b>
• (Ý chính 1)
• (Ý chính 2 - nếu có)

🔴 <b>TIN TIÊU CỰC / RỦI RO (TIN XẤU):</b>
• (Ý chính 1)
• (Ý chính 2 - nếu có)

📌 <b>ĐÁNH GIÁ CHUNG:</b> (Tóm tắt 1 câu tác động ngắn hạn tới giá dưới lăng kính đèn pha đi đêm)

Lưu ý: Chỉ dựa vào thông tin được cung cấp, trả ra bằng tiếng Việt, ngắn gọn, súc tích, không dông dài.
"""

        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600
        )
        
        ai_analysis = completion.choices[0].message.content
        return f"📰 <b>TỔNG HỢP TIN TỨC (ĐÈN PHA SOI ĐƯỜNG): {symbol}</b>\n====================================\n\n" + ai_analysis

    except Exception as e:
        err_msg = str(e)
        logging.error(f"Groq API Error for {symbol}: {err_msg}")
        return f"❌ <b>Lỗi khi gọi Groq AI:</b> <code>{html.escape(err_msg)}</code>"

# ==========================================
# HÀM PHÂN TÍCH KỸ THUẬT & SMC (TÍCH HỢP VẼ BIỂU ĐỒ & 4 MỤC QUY TẮC LÁI XE)
# ==========================================
def analyze_stock(symbol, resolution="D", count=200):
    def fmt(p): return f"{p:.1f}k"

    df, err = fetch_stock_data(symbol, resolution=resolution, count=count)
    if err or df.empty:
        return None, None, None, None, err if err else f"Không lấy được dữ liệu cho mã {symbol}!"

    df = calculate_indicators(df)
    smc = calculate_smc_structures(df)
    mkt_health = check_vnindex_health()

    last = df.iloc[-1]
    close = last["Close"]
    cmf = last.get("CMF", 0.0)
    rvol = last.get("RVOL", 1.0)
    rsi = last.get("RSI", 50.0)
    sma200 = last.get("SMA200", close)
    ema20 = last.get("EMA20", close)
    adx = last.get("ADX", 25.0)

    span_a = last.get("Senkou_Span_A", close)
    span_b = last.get("Senkou_Span_B", close)
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)

    poc, vah, val = calculate_volume_profile(df)
    
    is_near_sma200 = (close >= sma200 * 0.97) and (close < sma200)
    is_downtrend_heavy = (close < sma200) and not (is_near_sma200 and cmf > 0.03)

    confluence_points = []
    score = 50

    if close > cloud_top: 
        score += 20; confluence_points.append("Giá nằm trên mây Kumo (Đường thênh thang)")
    elif close >= cloud_bottom: 
        score += 10; confluence_points.append("Giá trong mây Kumo (Đường hẹp)")
    else: 
        confluence_points.append("⚠️ Giá nằm dưới mây Kumo (Đoạn đường sạt lở)")
    
    if smc["has_ob"]: score += 10; confluence_points.append(f"🔥 [Order Block tại {fmt(smc['ob_price'])}]")
    if smc["has_liquidity_sweep"]: score += 15; confluence_points.append("🎯 [Quét thanh khoản rút chân]")
    
    if cmf > 0.03: 
        score += 15
        confluence_points.append(f"Dòng tiền lớn CMF ({cmf:.2f}) mạnh (Tiếng máy khỏe)")
    elif cmf < -0.1:
        score -= 20
        confluence_points.append(f"🚨 Dòng tiền rút ròng mạnh (CMF: {cmf:.2f})")

    if is_downtrend_heavy and cmf < 0:
        score = min(score, 40)
        confluence_points.append("🛡️ [Khiên chống Downtrend: Cấm bắt đáy khi xe mất phanh]")

    confidence_score = min(max(int(score * mkt_health["score_modifier"]), 10), 90)

    entry_1 = round(min(close, max(val, close * 0.98)), 2)
    entry_2 = round(close, 2)
    entry_avg = (entry_1 + entry_2) / 2
    
    recent_swing_low = df["Low"].tail(10).min()
    stop_loss = round(min(entry_avg * 0.93, recent_swing_low * 0.98), 2)

    sug_qty, risk_nav_pct = calculate_position_size_1pct(entry_avg, stop_loss)
    ob_title, _, _ = check_ob_confirmation_type(df, smc)

    if confidence_score >= 65 and not is_downtrend_heavy:
        system_status = "ĐÈN XANH: MỞ MUA / TĂNG GA"
        driving_conclusion = "🛣️ <b>Trạng thái lái xe:</b> Cao tốc thông thoáng, tiếng máy gầm khỏe (CMF > 0, vượt mây). Tài xế <b>ĐẠP GA MẠNH</b> theo đúng làn đường tăng trưởng."
        final_icon = f"🟢 <b>NÊN MUA (Độ tin cậy: {confidence_score}%)</b>"
        action_note = f"Mở mua quanh <code>{fmt(entry_1)} - {fmt(entry_2)}</code>, Khối lượng: <code>{sug_qty:,} cổ</code>, Stoploss tại <code>{fmt(stop_loss)}</code>."
    elif confidence_score >= 50:
        system_status = "ĐÈN VÀNG: THĂM DÒ THẬN TRỌNG"
        driving_conclusion = "🛣️ <b>Trạng thái lái xe:</b> Đường sương mù nhẹ. Tài xế <b>GIẢM TỐC ĐỘ, THĂM DÒ</b>."
        final_icon = f"🟡 <b>THĂM DÒ THẬN TRỌNG (Độ tin cậy: {confidence_score}%)</b>"
        action_note = f"Đi chậm lại, quan sát kỹ vùng hỗ trợ, Stoploss tại <code>{fmt(stop_loss)}</code>."
    else:
        system_status = "ĐÈN ĐỎ: DỪNG XE / QUAN SÁT (NO TRADE)"
        driving_conclusion = "🛣️ <b>Trạng thái lái xe:</b> Cổ phiếu trượt dốc sâu / Mất phanh. Tài xế <b>TẤP XE VÀO LỀ NGHỈ NGƠI</b>."
        final_icon = f"🔴 <b>TẤP XE VÀO LỀ (Độ tin cậy: {confidence_score}%)</b>"
        action_note = f"Cổ phiếu xu hướng giảm hoặc rủi ro cao. <b>Tuyệt đối đứng ngoài!</b>"

    conf_str = "; ".join(confluence_points) if confluence_points else "Tích lũy ổn định"
    long_term_trend = "UPTREND (Xuống dốc thoáng)" if not is_downtrend_heavy else "DOWNTREND (Trượt lùi dốc)"

    text_part1 = f"📊 <b>BÁO CÁO PHÂN TÍCH KĨ THUẬT & SMC: {symbol.upper()}</b>\n"
    text_part1 += f"🗓️ Ngày: {last['time'].strftime('%Y-%m-%d')} | Khung: {resolution} | Giá: {fmt(close)}\n"
    text_part1 += f"🏛️ Thời tiết thị trường: <b>{html.escape(mkt_health['note'])}</b>\n\n"
    
    text_part1 += f"⛽ <b>1. XE VÀ NHIÊN LIỆU (Vốn & Khối lượng):</b>\n"
    text_part1 += f"• Trạng thái dòng tiền (Xăng): CMF = <code>{cmf:.2f}</code> | RVOL = <code>{rvol:.2f}x</code>\n\n"
    
    text_part1 += f"🛣️ <b>2. TUYẾN ĐƯỜNG & LÀN ĐƯỜNG (Xu hướng & Timing):</b>\n"
    text_part1 += f"• Làn đường hiện tại: {long_term_trend} | ADX = <code>{adx:.1f}</code>\n\n"
    
    text_part1 += f"🚛 <b>3. GIAO THƯỜNG XE CONTAINER & SMC (Tạo lập & Vùng giá):</b>\n"
    text_part1 += f"• Order Block: <code>{'Có (' + fmt(smc['ob_price']) + ')' if smc['has_ob'] else 'Chưa'}</code> | Trạng thái: <b>{ob_title}</b>\n"
    text_part1 += f"• Volume Profile POC: <code>{fmt(poc)}</code> | VAH: <code>{fmt(vah)}</code> | VAL: <code>{fmt(val)}</code>\n\n"

    text_part1 += f"🏗️ <b>4. BỘ MÁY QUẢN TRỊ & QUẢN TRỊ RỦI RO (1% NAV):</b>\n"
    text_part1 += f"• Khuyến nghị: <b>{system_status}</b>\n• {action_note}\n\n"

    text_part2 = f"⚡ <b>HỘI TỤ TÍN HIỆU:</b> {conf_str}\n\n"
    text_part2 += f"{driving_conclusion}\n\n"
    text_part2 += f"🏁 <b>KẾT LUẬN CUỐI CÙNG:</b> {final_icon}"

    keyboard = [
        [
            InlineKeyboardButton("⏱️ 1H", callback_data=f"{symbol}|60|300"),
            InlineKeyboardButton("📅 1D", callback_data=f"{symbol}|D|200"),
            InlineKeyboardButton("📈 1W", callback_data=f"{symbol}|W|100"),
            InlineKeyboardButton("🗓️ 1M", callback_data=f"{symbol}|M|50"),
        ],
        [
            InlineKeyboardButton("🛡️ Check Kỷ Luật Mua", callback_data=f"checkmua_cb|{symbol}"),
            InlineKeyboardButton("🎯 Lọc Đáy An Toàn", callback_data="run_dipbuy_callback"),
        ],
        [
            InlineKeyboardButton("🤖 Tín Hiệu AutoBuy", callback_data="run_autobuy_callback"),
            InlineKeyboardButton("🏢 Dòng Tiền Ngành", callback_data="run_sectors_callback"),
        ],
        [
            InlineKeyboardButton("🌍 Khối Ngoại Mua/Bán", callback_data="run_foreign_callback")
        ]
    ]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=110)
    ax.plot(df["time"], df["Close"], label="Giá đóng cửa", color="#111111", lw=1.3, zorder=3)
    
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        ax.plot(df["time"], df["BB_Upper"], label="BB Upper", color="gray", linestyle="--", alpha=0.5, lw=0.8)
        ax.plot(df["time"], df["BB_Lower"], label="BB Lower", color="gray", linestyle="--", alpha=0.5, lw=0.8)
        ax.fill_between(df["time"], df["BB_Lower"], df["BB_Upper"], color="gray", alpha=0.06, label="Bollinger Channel")

    if "EMA20" in df.columns:
        ax.plot(df["time"], df["EMA20"], label="EMA 20", color="#ff7f0e", lw=1.1, alpha=0.85)
    if "SMA200" in df.columns:
        ax.plot(df["time"], df["SMA200"], label="SMA 200", color="#d62728", lw=1.2, linestyle="-.", alpha=0.85)

    ax.axhline(poc, color="purple", linestyle=":", label=f"POC: {fmt(poc)}", alpha=0.9, lw=1.2)
    if smc["has_ob"] and smc["ob_price"] > 0:
        ax.axhline(smc["ob_price"], color="blue", linestyle="--", label=f"Order Block: {fmt(smc['ob_price'])}", alpha=0.8, lw=1.1)

    ax.set_title(f"Bản Đồ GPS Kỹ Thuật & Chỉ Báo - {symbol.upper()} ({resolution})", fontsize=10, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=6.5, ncol=2, framealpha=0.8)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return text_part1, text_part2, buf, InlineKeyboardMarkup(keyboard), None


# ==========================================
# 3. CÁC HÀM QUÉT BỘ LỌC TỰ ĐỘNG & TÌM SIÊU CỔ PHIẾU (/sieuco)
# ==========================================
def scan_super_stock(symbol):
    try:
        df_d, err_d = fetch_stock_data(symbol, resolution="D", count=200)
        if err_d or df_d.empty or len(df_d) < 50: 
            return None

        df_d = calculate_indicators(df_d)
        last_d = df_d.iloc[-1]
        close = last_d["Close"]
        sma200 = last_d.get("SMA200", close)
        cmf_d = last_d.get("CMF", 0.0)
        rvol_d = last_d.get("RVOL", 1.0)
        
        is_near_sma200 = (close >= sma200 * 0.97) and (close < sma200)
        if close < sma200 and not (is_near_sma200 and cmf_d > 0.03):
            return None
            
        if cmf_d < 0.03 or rvol_d < 1.2:
            return None

        smc = calculate_smc_structures(df_d)
        poc, vah, val = calculate_volume_profile(df_d)

        entry_price = round(close, 2)
        recent_swing_low = df_d["Low"].tail(10).min()
        stop_loss = round(min(entry_price * 0.93, recent_swing_low * 0.98), 2)
        suggested_qty, _ = calculate_position_size_1pct(entry_price, stop_loss)

        score = 50
        if smc["has_ob"]: score += 15
        if smc["has_liquidity_sweep"]: score += 20
        if cmf_d > 0.05: score += 15
        if rvol_d >= 1.5: score += 10

        if score >= 70:
            return {
                "symbol": symbol, 
                "close": close, 
                "score": min(score, 98),
                "entry": entry_price, 
                "stop_loss": stop_loss,
                "cmf": round(cmf_d, 2),
                "rvol": round(rvol_d, 2),
                "suggested_qty": suggested_qty
            }
    except Exception:
        pass
    return None

async def run_super_stock_screener_with_progress(msg):
    total_symbols = len(WATCHLIST_SCAN)
    completed = 0
    super_stocks = []

    with ThreadPoolExecutor(max_workers=30) as pool:
        futures = {pool.submit(scan_super_stock, symbol): symbol for symbol in WATCHLIST_SCAN}
        
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            if res:
                super_stocks.append(res)
            
            if completed % 5 == 0 or completed == total_symbols:
                percent = int((completed / total_symbols) * 100)
                bar = "█" * int(percent / 10) + "░" * (10 - int(percent / 10))
                try:
                    await msg.edit_text(
                        f"🚀 <b>Đang quét tìm Siêu Cổ Phiếu bứt phá...</b>\n"
                        f"[{bar}] <b>{percent}%</b> ({completed}/{total_symbols} mã)\n"
                        f"⏳ Đang áp dụng bộ lọc dòng tiền & cấu trúc SMC...",
                        parse_mode="HTML"
                    )
                except Exception:
                    pass

    return sorted(super_stocks, key=lambda x: x["score"], reverse=True)[:5]


def scan_monthly_higher_lows(symbol):
    df, err = fetch_stock_data(symbol, resolution="M", count=48)
    if err or df.empty or len(df) < 18: return None
    df = calculate_indicators(df)
    lows, closes = df["Low"].values, df["Close"].values
    if len(lows) < 12: return None
    recent_low_1 = lows[-1]
    prev_low_2 = lows[-6:-1].min()
    old_low_3 = lows[-12:-6].min()
    is_higher_lows = (recent_low_1 >= prev_low_2) and (prev_low_2 >= old_low_3)
    last = df.iloc[-1]
    close, cmf = closes[-1], last.get("CMF", 0.0)
    span_a, span_b = last.get("Senkou_Span_A", close), last.get("Senkou_Span_B", close)
    cloud_bottom = min(span_a, span_b)
    
    if is_higher_lows and cmf > -0.05 and close >= cloud_bottom * 0.95:
        score = (cmf * 50) + (20 if close > max(span_a, span_b) else 10)
        return {"symbol": symbol.upper(), "close": round(close, 1), "cmf": round(cmf, 2), "recent_low": round(recent_low_1, 1), "prev_low": round(prev_low_2, 1), "score": round(score, 1)}
    return None

async def run_monthly_higher_lows_screener_with_progress(msg):
    vn30_core = ["VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "STB", "TPB", "VHM", "VIC", "VRE", "KDH", "NLG", "KBC", "SSI", "VCI", "HCM", "HPG", "HSG", "MSN", "VNM", "FPT", "REE", "POW", "GAS", "PLX", "GMD"]
    total_symbols = len(vn30_core)
    completed, results = 0, []

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(scan_monthly_higher_lows, symbol): symbol for symbol in vn30_core}
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            if res: results.append(res)
            if completed % 4 == 0 or completed == total_symbols:
                percent = int((completed / total_symbols) * 100)
                bar = "█" * int(percent / 10) + "░" * (10 - int(percent / 10))
                try:
                    await msg.edit_text(f"📈 <b>Đang quét siêu cấu trúc VN30 Nâng Đáy (Khung Tháng)...</b>\n[{bar}] <b>{percent}%</b>", parse_mode="HTML")
                except Exception:
                    pass
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]

def process_single_stock_for_autobuy(symbol):
    try:
        df_d, err_d = fetch_stock_data(symbol, resolution="D", count=200)
        if err_d or df_d.empty or len(df_d) < 50: return None
        df_d = calculate_indicators(df_d)
        last_d = df_d.iloc[-1]
        close, sma200, cmf_d, rvol_d = last_d["Close"], last_d.get("SMA200", last_d["Close"]), last_d.get("CMF", 0.0), last_d.get("RVOL", 1.0)
        
        is_near_sma200 = (close >= sma200 * 0.97) and (close < sma200)
        if close < sma200 and not (is_near_sma200 and cmf_d > 0.03): return None
        if rvol_d < 1.3: return None

        smc = calculate_smc_structures(df_d)
        entry_price = round(close, 2)
        recent_swing_low = df_d["Low"].tail(10).min()
        stop_loss = round(min(entry_price * 0.93, recent_swing_low * 0.98), 2)
        suggested_qty, _ = calculate_position_size_1pct(entry_price, stop_loss)

        score = 50
        if smc["has_ob"]: score += 15
        if smc["has_liquidity_sweep"]: score += 20
        if cmf_d > 0.02: score += 15
        if rvol_d >= 1.5: score += 10

        if score >= 65 and cmf_d > 0:
            return {"symbol": symbol, "close": close, "score": min(score, 95), "entry": entry_price, "stop_loss": stop_loss, "suggested_qty": suggested_qty}
    except Exception:
        pass
    return None

async def run_smart_buy_screener_with_progress(msg):
    total_symbols = len(WATCHLIST_SCAN)
    completed, buy_recommendations = 0, []
    with ThreadPoolExecutor(max_workers=30) as pool:
        futures = {pool.submit(process_single_stock_for_autobuy, symbol): symbol for symbol in WATCHLIST_SCAN}
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            if res: buy_recommendations.append(res)
    return sorted(buy_recommendations, key=lambda x: x["score"], reverse=True)[:5]

def scan_single_dip_stock(symbol):
    df, err = fetch_stock_data(symbol, resolution="D", count=100)
    if err or df.empty or len(df) < 30: return None
    df = calculate_indicators(df)
    last = df.iloc[-1]
    close, cmf, rvol, sma200 = last["Close"], last.get("CMF", 0.0), last.get("RVOL", 1.0), last.get("SMA200", last["Close"])
    if close < sma200 * 0.95: return None
    smc = calculate_smc_structures(df)
    poc, _, _ = calculate_volume_profile(df)
    recent_high = df["High"].tail(15).max()
    drop_pct = ((recent_high - close) / recent_high) * 100

    if drop_pct >= 4.0 and (close <= poc * 1.02 or smc["has_ob"] or smc["has_liquidity_sweep"]) and cmf > 0:
        score = (drop_pct * 2.0) + (cmf * 30) + (15 if smc["has_liquidity_sweep"] else 0)
        return {"symbol": symbol, "close": close, "drop_pct": round(drop_pct, 1), "cmf": cmf, "rvol": rvol, "poc": poc, "smc": smc, "score": score}
    return None

async def run_dip_buy_screener_with_progress(msg):
    total_symbols = len(WATCHLIST_SCAN)
    completed, results = 0, []
    with ThreadPoolExecutor(max_workers=30) as pool:
        futures = {pool.submit(scan_single_dip_stock, symbol): symbol for symbol in WATCHLIST_SCAN}
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            if res: results.append(res)
    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]

def analyze_single_sector_stock(symbol):
    df, err = fetch_stock_data(symbol, resolution="D", count=40)
    if err or df.empty or len(df) < 15: return None
    df = calculate_indicators(df)
    last, prev = df.iloc[-1], df.iloc[-2]
    pct_change = ((last["Close"] - prev["Close"]) / prev["Close"]) * 100
    return {
        "symbol": symbol, 
        "pct_change": pct_change, 
        "cmf": last.get("CMF", 0.0), 
        "rvol": last.get("RVOL", 1.0)
    }

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
            avg_rvol = np.mean([s["rvol"] for s in stock_metrics])
            avg_pct = np.mean([s["pct_change"] for s in stock_metrics])
            
            flow_pct = round(avg_cmf * 100, 1)
            
            leader_stock = sorted(stock_metrics, key=lambda x: x["cmf"] + x["rvol"], reverse=True)[0]["symbol"]
            sector_results.append({
                "sector": sector_name, 
                "flow_pct": flow_pct, 
                "avg_pct": round(avg_pct, 2),
                "avg_rvol": round(avg_rvol, 2),
                "leader": leader_stock, 
                "score": (avg_cmf * 40) + (avg_rvol * 20)
            })
            
    return sorted(sector_results, key=lambda x: x["flow_pct"], reverse=True)


# ==========================================
# 4. HANDLERS XỬ LÝ SỰ KIỆN BOT
# ==========================================

async def sieuco_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    msg = await target_msg.reply_text(
        "🚀 <b>Đang khởi tạo tiến trình quét Siêu Cổ Phiếu...</b>\n"
        "[░░░░░░░░░░] <b>0%</b> (0 mã)\n"
        "⏳ Đang kết nối máy chủ dữ liệu...",
        parse_mode="HTML"
    )
    
    try:
        picks = await run_super_stock_screener_with_progress(msg)
        if not picks:
            await msg.edit_text("⚠️ Hiện tại chưa tìm thấy mã nào đạt tiêu chí Siêu Cổ Phiếu bứt phá.")
            return
            
        text = "🚀 <b>DANH SÁCH SIÊU CỔ PHIẾU BỨT PHÁ (MÃNH HỔ)</b>\n====================================\n\n"
        for idx, item in enumerate(picks, 1):
            text += f"{idx}. 🔹 <b>Mã: {item['symbol'].upper()}</b> | Giá: <code>{item['close']}k</code> | Score: <code>{item['score']}%</code>\n"
            text += f"   • Dòng tiền CMF: <code>{item['cmf']}</code> | Khối lượng RVOL: <code>{item['rvol']}x</code>\n"
            text += f"   • Điểm mua: <code>{item['entry']}k</code> | Stoploss: <code>{item['stop_loss']}k</code> | Qty: <code>{item['suggested_qty']:,} cổ</code>\n------------------------------------\n"
        await msg.edit_text(text, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ <b>Lỗi:</b> {html.escape(str(e))}", parse_mode="HTML")


async def check_buy_discipline_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    target_msg = update.message if update.message else update.callback_query.message
    
    if not args:
        await target_msg.reply_text(
            "⚠️ Vui lòng nhập mã cổ phiếu cần kiểm tra kỷ luật mua.\n"
            "<i>Ví dụ:</i> <code>/checkmua TCB</code> hoặc <code>/checkkyloimua HPG</code>",
            parse_mode="HTML"
        )
        return
        
    symbol = args[0].upper().strip()
    is_solid_stock = symbol in WATCHLIST_SCAN
    
    msg = await target_msg.reply_text(f"🛡️ Đang chạy bộ lọc Checklist kỷ luật cho <b>{symbol}</b>...", parse_mode="HTML")
    
    try:
        loop = asyncio.get_running_loop()
        df, err = await loop.run_in_executor(executor, fetch_stock_data, symbol, "D", 150)
        
        if err or df.empty:
            await msg.edit_text(f"❌ Không lấy được dữ liệu cho mã <b>{symbol}</b>: {html.escape(str(err))}", parse_mode="HTML")
            return
            
        df = calculate_indicators(df)
        smc = calculate_smc_structures(df)
        
        last = df.iloc[-1]
        close = last["Close"]
        cmf = last.get("CMF", 0.0)
        
        ob_price = smc["ob_price"] if smc["has_ob"] else 0
        is_near_ob = (ob_price > 0 and abs(close - ob_price) / ob_price <= 0.03) or smc["has_liquidity_sweep"]
        has_signal_confirmed = smc["has_liquidity_sweep"] or (cmf > 0.02)
        
        passed_criteria_count = sum([is_solid_stock, is_near_ob, has_signal_confirmed])
        
        if passed_criteria_count == 3:
            status_icon = "🟢 ĐÈN XANH TUYỆT ĐỐI: ĐỦ ĐIỀU KIỆN MUA"
            action_advice = "<b>Đạt 3/3 tiêu chuẩn kỷ luật!</b> Tự tin trích quỹ Hũ 3 mở mua gom thăm dò."
        elif passed_criteria_count == 2:
            status_icon = "🟡 ĐÈN VÀNG: THẬN TRỌNG CHỜ THÊM"
            action_advice = "<b>Mới đạt 2/3 tiêu chuẩn.</b> Kiên nhẫn chờ thêm tín hiệu xác nhận ở vùng hỗ trợ."
        else:
            status_icon = "🔴 ĐÈN ĐỎ: TUYỆT ĐỐI KHÔNG MUA"
            action_advice = "<b>Chưa thỏa mãn điều kiện an toàn.</b> Vi phạm checklist kỷ luật, đứng ngoài quan sát!"

        report_text = (
            f"📋 <b>CHECKLIST KỶ LUẬT TRƯỚC KHI ĐẶT LỆNH: {symbol}</b>\n"
            f"====================================\n"
            f"• <b>Mức giá hiện tại:</b> <code>{close:.1f}k</code>\n\n"
            f"<b>CHI TIẾT 3 TIÊU CHÍ KỶ LUẬT:</b>\n"
            f"1️⃣ Thuộc danh mục 'Thùng đặc': <code>{'✅ Đạt (Có)' if is_solid_stock else '❌ Không có'}</code>\n"
            f"2️⃣ Sát Order Block / Hỗ trợ: <code>{'✅ Đạt (OB ' + f'{ob_price:.1f}k' + ')' if is_near_ob else '⚠️ Chưa sát vùng giá'}</code>\n"
            f"3️⃣ Tín hiệu xác nhận (Sweep / CMF>0): <code>{'✅ Đạt (CMF=' + f'{cmf:.2f}' + ')' if has_signal_confirmed else '❌ Chưa có tín hiệu'}</code>\n\n"
            f"🎯 <b>KẾT LUẬN KỶ LUẬT:</b>\n"
            f"<b>{status_icon}</b>\n"
            f"💡 <i>Lời khuyên:</i> {action_advice}"
        )
        
        keyboard = [
            [
                InlineKeyboardButton(f"🔍 Kiểm Tra Chi Tiết OB", callback_data=f"checkob_cb|{symbol}"),
                InlineKeyboardButton(f"📊 Xem Biểu Đồ", callback_data=f"{symbol}|D|200")
            ]
        ]
        
        await msg.edit_text(report_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")
        
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi hệ thống: {html.escape(str(e))}", parse_mode="HTML")


async def quytec_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    rules_text = (
        "🚗 <b>CẨM NANG SINH TỒN BẤT BIẾN: LÁI XE CHỨNG KHOÁN</b>\n"
        "====================================\n\n"
        "⛽ <b>1. XE VÀ NHIÊN LIỆU (Vốn & Margin):</b> Luôn giữ bình xăng dự phòng (tiền mặt).\n"
        "🛣️ <b>2. TUYẾN ĐƯỜNG (Ngành & Cổ phiếu):</b> Chọn đúng làn 'thùng đặc' (TCB, HPG, FPT...).\n"
        "🚛 <b>3. XE CONTAINER (Tạo lập):</b> Chờ xe bẻ lái xong (rút chân Order Block) mới bám theo.\n"
        "🏗️ <b>4. QUẢN TRỊ RỦI RO:</b> Cắt lỗ khắt khe 1% NAV để bảo vệ vốn mua nhà."
    )
    keyboard = [[InlineKeyboardButton("🚀 Tìm Siêu Cổ", callback_data="run_sieuco_callback"), InlineKeyboardButton("🎯 Lọc Đáy", callback_data="run_dipbuy_callback")]]
    await target_msg.reply_text(rules_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")


async def autobuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    msg = await target_msg.reply_text("🤖 Đang quét danh sách Auto-Buy an toàn...", parse_mode="HTML")
    try:
        picks = await run_smart_buy_screener_with_progress(msg)
        if not picks:
            await msg.edit_text("⚠️ Hiện tại không có mã nào đủ chuẩn an toàn.")
            return
        text = "🎯 <b>DANH SÁCH CỔ PHIẾU AUTO-BUY AN TOÀN</b>\n====================================\n\n"
        for idx, item in enumerate(picks, 1):
            text += f"{idx}. 🔹 <b>{item['symbol'].upper()}</b> | Giá: <code>{item['close']:.1f}k</code> | Score: <code>{item['score']}%</code>\n"
            text += f"   • Mua: <code>{item['entry']}k</code> | Stoploss: <code>{item['stop_loss']}k</code> | Qty: <code>{item['suggested_qty']:,} cổ</code>\n"
        await msg.edit_text(text, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi: {html.escape(str(e))}", parse_mode="HTML")


async def dipbuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    msg = await target_msg.reply_text("🎯 Đang quét cổ phiếu tích lũy an toàn...", parse_mode="HTML")
    try:
        picks = await run_dip_buy_screener_with_progress(msg)
        if not picks:
            await msg.edit_text("⚠️ Chưa phát hiện cổ phiếu đảo chiều đạt chuẩn.")
            return
        text = "🎯 <b>DANH SÁCH CỔ PHIẾU TÍCH LŨY AN TOÀN</b>\n====================================\n\n"
        for idx, item in enumerate(picks, 1):
            smc_tag = "🔥 [Sweep]" if item["smc"]["has_liquidity_sweep"] else ("⚡ [OB]" if item["smc"]["has_ob"] else "📊 [POC]")
            text += f"{idx}. 🔹 <b>{item['symbol'].upper()}</b> {smc_tag}\n"
            text += f"   • Giá: <code>{item['close']:.1f}k</code> | Giảm từ đỉnh: <code>-{item['drop_pct']}%</code> | Hỗ trợ: <code>{item['poc']:.1f}k</code>\n"
        await msg.edit_text(text, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi: {html.escape(str(e))}", parse_mode="HTML")


async def nangday_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    msg = await target_msg.reply_text("📈 Đang quét VN30 Nâng Đáy Khung Tháng...", parse_mode="HTML")
    try:
        picks = await run_monthly_higher_lows_screener_with_progress(msg)
        if not picks:
            await msg.edit_text("⚠️ Chưa ghi nhận mã nào thỏa mãn cấu trúc nâng đáy khung tháng.")
            return
        text = "📈 <b>VN30 NÂNG ĐÁY (KHUNG THÁNG - M)</b>\n====================================\n\n"
        for idx, item in enumerate(picks, 1):
            text += f"{idx}. 🔹 <b>{item['symbol']}</b> | Giá: <code>{item['close']}k</code> | CMF: <code>{item['cmf']}</code>\n"
            text += f"   • Đáy tháng: <code>{item['prev_low']}k ➔ {item['recent_low']}k</code>\n"
        await msg.edit_text(text, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi: {html.escape(str(e))}", parse_mode="HTML")


async def sectors_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    msg = await target_msg.reply_text("🏢 Đang phân tích dòng tiền vào/ra các Nhóm Ngành...", parse_mode="HTML")
    try:
        sectors = await run_sector_flow_screener_with_progress(msg)
        if not sectors:
            await msg.edit_text("⚠️ Không lấy được dữ liệu ngành.")
            return
            
        text = "🔥 <b>DÒNG TIỀN VÀO/RA THEO NHÓM NGÀNH</b>\n====================================\n\n"
        for idx, sec in enumerate(sectors, 1):
            flow_val = sec['flow_pct']
            if flow_val > 1.0:
                icon = "🟢 [HÚT MẠNH]"
                flow_str = f"+{flow_val}% (Tiền vào)"
            elif flow_val >= 0:
                icon = "🟡 [CÂN BẰNG]"
                flow_str = f"+{flow_val}% (Ổn định)"
            else:
                icon = "🔴 [RÚT RÒNG]"
                flow_str = f"{flow_val}% (Tiền ra)"
                
            text += f"{idx}. 🏢 <b>{sec['sector'].upper()}</b> {icon}\n"
            text += f"   • Dòng tiền: <code>{flow_str}</code> | Biến động: <code>{sec['avg_pct']:+.2f}%</code>\n"
            text += f"   • Cổ phiếu dẫn dắt (Leader): <code>{sec['leader']}</code>\n------------------------------------\n"
            
        await msg.edit_text(text, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi: {html.escape(str(e))}", parse_mode="HTML")


async def foreign_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    msg = await target_msg.reply_text("⏳ Đang cào dữ liệu khối ngoại từ 24hmoney...", parse_mode="HTML")
    try:
        loop = asyncio.get_running_loop()
        data, err = await loop.run_in_executor(executor, get_foreign_trading_data)
        if not data:
            await msg.edit_text(f"⚠️ Không lấy được dữ liệu khối ngoại: <code>{html.escape(str(err))}</code>", parse_mode="HTML")
            return
        text = "🌍 <b>GIAO DỊCH KHỐI NGOẠI (24HMONEY)</b>\n====================================\n\n"
        for idx, item in enumerate(data[:10], 1):
            action_icon = "🟢 Mua" if item["netValue"] > 0 else "🔴 Bán"
            text += f"{idx}. <b>{item['ticker']}</b>: <code>{item['netValue']:+.2f} tỷ</code> ({action_icon})\n"
        await msg.edit_text(text, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi: {html.escape(str(e))}", parse_mode="HTML")


async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("⚠️ Vui lòng nhập mã cổ phiếu. Ví dụ: <code>/tin VCG</code>", parse_mode="HTML")
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
# 🆕 TÍNH NĂNG 1: /vonglap (ĐỘNG THEO DỮ LIỆU THẬT)
# ==========================================
async def vonglap_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    target_msg = update.message if update.message else update.callback_query.message
    
    if not args:
        await target_msg.reply_text(
            "⚠️ Vui lòng nhập mã cổ phiếu cần đối chiếu vòng lặp.\n"
            "<i>Ví dụ:</i> <code>/vonglap TCB</code>",
            parse_mode="HTML"
        )
        return
        
    symbol = args[0].upper().strip()
    msg = await target_msg.reply_text(f"🔄 Đang quét lịch sử khung Tuần (W) để tìm đoạn nến đồng pha cho <b>{symbol}</b>...", parse_mode="HTML")
    
    try:
        loop = asyncio.get_running_loop()
        df, err = await loop.run_in_executor(executor, fetch_stock_data, symbol, "W", 150)
        
        if err or df.empty or len(df) < 40:
            await msg.edit_text(f"❌ Không đủ dữ liệu khung Weekly để đối chiếu cho mã <b>{symbol}</b>.", parse_mode="HTML")
            return
            
        df = calculate_indicators(df)
        last = df.iloc[-1]
        close = last["Close"]
        rvol = last.get("RVOL", 1.0)
        cmf = last.get("CMF", 0.0)
        
        recent_closes = df["Close"].tail(12)
        price_range_pct = ((recent_closes.max() - recent_closes.min()) / recent_closes.min()) * 100
        
        if price_range_pct < 8.0:
            pattern_status = "Đang co hẹp biên độ tích lũy (Sideways Box / Nén lò xo)"
            historical_match = "giai đoạn tích lũy cạn cung trước các nhịp sóng lớn trung hạn"
            advice = "Cấu trúc siết chặt, phù hợp gom mua thăm dò tỷ trọng nhỏ ở vùng hỗ trợ biên dưới."
        elif close > df["EMA20"].iloc[-1] and cmf > 0:
            pattern_status = "Đang duy trì kênh tăng giá trung hạn (Uptrend channel)"
            historical_match = "chu kỳ đẩy giá mạnh (Markup phase) khi dòng tiền lớn nhập cuộc"
            advice = "Canh các nhịp rung lắc trong tuần để gia tăng tỷ trọng theo xu hướng."
        else:
            pattern_status = "Đang trong nhịp điều chỉnh kiểm định cung cầu (Pullback / Test)"
            historical_match = "giai đoạn rũ bỏ (Washout) test lại vùng hỗ trợ cứng"
            advice = "Kiên nhẫn chờ tín hiệu dòng tiền CMF cải thiện hoặc giá chạm vùng biên hỗ trợ mạnh."

        report_content = (
            f"🔄 <b>ĐỐI CHIẾU CẤU TRÚC NẾN TƯƠNG ĐỒNG: {symbol}</b>\n"
            f"====================================\n"
            f"• <b>Mức giá hiện tại (W):</b> <code>{close:.1f}k</code>\n"
            f"• <b>Độ nén biên độ (12 tuần):</b> <code>{price_range_pct:.1f}%</code> | <b>CMF:</b> <code>{cmf:.2f}</code>\n\n"
            f"🔍 <b>KẾT QUẢ SO KHỚP LỊCH SỬ (KHUNG TUẦN):</b>\n"
            f"• <b>Trạng thái hiện tại:</b> {pattern_status}.\n"
            f"• <b>Mô hình tương đồng:</b> Vận động có độ tương đồng cao với <b>{historical_match}</b> trong dữ liệu lịch sử của mã này.\n\n"
            f"💡 <i>Đánh giá kỹ thuật:</i> {advice}"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("📊 Xem Biểu Đồ Chi Tiết", callback_data=f"{symbol}|W|100"),
                InlineKeyboardButton("⚖️ So Sánh Đa Khung", callback_data=f"sosanh_cb|{symbol}")
            ]
        ]
        
        await msg.edit_text(report_content, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")
        
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi xử lý vòng lặp: {html.escape(str(e))}", parse_mode="HTML")


# ==========================================
# 🆕 TÍNH NĂNG 2: /sosanh_muavumedia (ĐỘNG THEO DỮ LIỆU THẬT)
# ==========================================
async def sosanh_muavumedia_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    target_msg = update.message if update.message else update.callback_query.message
    
    if not args:
        await target_msg.reply_text(
            "⚠️ Vui lòng nhập mã cổ phiếu cần đối chiếu đa khung.\n"
            "<i>Ví dụ:</i> <code>/sosanh_muavumedia TCB</code>",
            parse_mode="HTML"
        )
        return
        
    symbol = args[0].upper().strip()
    msg = await target_msg.reply_text(f"⚖️ Đang bóc tách dữ liệu đa khung (Macro Tháng vs T+ Ngày) cho <b>{symbol}</b>...", parse_mode="HTML")
    
    try:
        loop = asyncio.get_running_loop()
        df_m, err_m = await loop.run_in_executor(executor, fetch_stock_data, symbol, "M", 60)
        df_d, err_d = await loop.run_in_executor(executor, fetch_stock_data, symbol, "D", 100)
        
        if err_m or err_d or df_m.empty or df_d.empty:
            await msg.edit_text(f"❌ Không đủ dữ liệu để đối chiếu đa khung cho mã <b>{symbol}</b>.", parse_mode="HTML")
            return
            
        df_m = calculate_indicators(df_m)
        df_d = calculate_indicators(df_d)
        
        close_price = df_d.iloc[-1]["Close"]
        cmf_d = df_d.iloc[-1].get("CMF", 0.0)
        smc_d = calculate_smc_structures(df_d)
        
        m_high = df_m["High"].max()
        m_low = df_m["Low"].min()
        pos_in_cycle = (close_price - m_low) / (m_high - m_low + 1e-9) * 100
        
        macro_eval = "vùng đỉnh chu kỳ dài hạn, rủi ro phân phối lớn" if pos_in_cycle > 75 else ("vùng lưng chừng xu hướng" if pos_in_cycle > 40 else "vùng tích lũy chân sóng vĩ mô")
        
        comparison_text = (
            f"⚖️ <b>BẢN ĐỐI CHIẾU ĐA KHUNG (MACRO vs T+): {symbol}</b>\n"
            f"====================================\n"
            f"• <b>Mã tài sản:</b> <code>{symbol}</code> | Giá hiện tại: <code>{close_price:.1f}k</code>\n"
            f"• <b>Vị thế chu kỳ Tháng (M):</b> Nằm ở mức <code>{pos_in_cycle:.1f}%</code> biên độ lịch sử ({macro_eval}).\n\n"
            f"🏛️ <b>1. GÓC NHÌN MÙA VỤ / CHU KỲ LỚN (Khung Tháng - M):</b>\n"
            f"• <i>Trạng thái:</i> {'Tiềm ẩn áp lực chốt lời dài hạn do đã ở vùng giá cao' if pos_in_cycle > 75 else 'Dư địa chu kỳ dài còn khá thoáng, cấu trúc dài hạn ủng hộ nắm giữ'}.\n\n"
            f"⏱️ <b>2. GÓC NHÌN TÁC CHIẾN NGẮN HẠN (Khung Ngày - D / SMC):</b>\n"
            f"• <i>Trạng thái:</i> Order Block gần nhất tại <code>{smc_d.get('ob_price', 0):.1f}k</code> | Dòng tiền CMF ngắn hạn: <code>{cmf_d:.2f}</code>.\n"
            f"• <i>Hành động ngắn hạn:</i> {'Đủ điều kiện giải ngân thăm dò T+ tại hỗ trợ' if cmf_d > -0.05 else 'Dòng tiền ngắn hạn chưa thuận lợi, cần quan sát thêm'}.\n\n"
            f"🎯 <b>KẾT LUẬN CHIẾN LƯỢC TỔNG HÒA:</b>\n"
            f"✅ Kết hợp linh hoạt giữa khung Tháng và Ngày để tối ưu hóa biên lợi nhuận và quản trị chặt rủi ro dừng lỗ 1% NAV!"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("🛡️ Check Kỷ Luật Mua", callback_data=f"checkmua_cb|{symbol}"),
                InlineKeyboardButton("📊 Xem Biểu Đồ Ngày", callback_data=f"{symbol}|D|200")
            ]
        ]
        
        await msg.edit_text(comparison_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")
        
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi đối chiếu: {html.escape(str(e))}", parse_mode="HTML")


# ==========================================
# 🆕 TÍNH NĂNG 3: /docvi (Đọc vị mánh khóe tạo lập phiên)
# ==========================================
async def docvi_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    target_msg = update.message if update.message else update.callback_query.message
    
    if not args:
        await target_msg.reply_text(
            "⚠️ Vui lòng nhập mã cổ phiếu cần đọc vị phiên.\n"
            "<i>Ví dụ:</i> <code>/docvi TCB</code>",
            parse_mode="HTML"
        )
        return
        
    symbol = args[0].upper().strip()
    msg = await target_msg.reply_text(f"🔍 Đang bóc tách dữ liệu khung ngắn hạn để bóc trần mánh khóe mã <b>{symbol}</b>...", parse_mode="HTML")
    
    try:
        loop = asyncio.get_running_loop()
        df, err = await loop.run_in_executor(executor, fetch_stock_data, symbol, "60", 100)
        
        if err or df.empty or len(df) < 20:
            await msg.edit_text(f"❌ Không đủ dữ liệu ngắn hạn cho mã <b>{symbol}</b>.", parse_mode="HTML")
            return
            
        df = calculate_indicators(df)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        close = last["Close"]
        vol = last["Volume"]
        rvol = last.get("RVOL", 1.0)
        cmf = last.get("CMF", 0.0)
        price_change = ((close - prev["Close"]) / prev["Close"]) * 100
        
        tricks = []
        if rvol > 2.0 and abs(price_change) < 0.5:
            tricks.append("🛡️ <b>Mánh khóe Trao tay / Kê lệnh ngầm:</b> Volume nổ lớn (RVOL > 2x) nhưng giá đi ngang. Nhà cái đang gom hoặc phân phối kín.")
        if price_change < -1.5 and rvol > 1.5:
            tricks.append("🚨 <b>Bẫy Đè giá rũ bỏ (Washout):</b> Đạp ép giá sâu kèm thanh khoản lớn để quét hàng yếu bóng vía trước khi kéo ngược.")
        if price_change > 1.5 and rvol > 1.5 and cmf > 0:
            tricks.append("🔥 <b>Mánh khóe Kéo thốc (Bull Move):</b> Dòng tiền lớn dồn dập đánh vượt kháng cự, xác nhận phe mua kiểm soát.")
        if rvol < 0.6 and abs(price_change) < 0.4:
            tricks.append("🧊 <b>Mánh khóe Cạn cung (No-Supply):</b> Thanh khoản khô kiệt, tạo lập đang bỏ ngỏ phiên để thăm dò tâm lý.")
        if not tricks:
            tricks.append("⚖️ <b>Trạng thái giằng co:</b> Cung cầu cân bằng, chưa có dấu hiệu bẻ lái thô bạo.")

        report = (
            f"🕵️‍♂️ <b>HỒ SƠ ĐỌC VỊ MÁNH KHÓE TẠO LẬP: {symbol}</b>\n"
            f"====================================\n"
            f"• <b>Giá hiện tại:</b> <code>{close:.1f}k</code> ({price_change:+.2f}%)\n"
            f"• <b>Khối lượng (RVOL):</b> <code>{rvol:.2f}x</code> | <b>CMF:</b> <code>{cmf:.2f}</code>\n\n"
            f"🎯 <b>NHẬN ĐỊNH HÀNH VI NHÀ CÁI:</b>\n"
            f"{'\n'.join(tricks)}\n\n"
            f"💡 <i>Lời khuyên:</i> Luôn giữ cái đầu lạnh, đừng để Volume ảo đánh lừa cảm xúc trong phiên!"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("🛡️ Check Kỷ Luật Mua", callback_data=f"checkmua_cb|{symbol}"),
                InlineKeyboardButton("📊 Xem Biểu Đồ Ngày", callback_data=f"{symbol}|D|200")
            ]
        ]
        
        await msg.edit_text(report, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi đọc vị: {html.escape(str(e))}", parse_mode="HTML")


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
            InlineKeyboardButton("🛡️ Check Kỷ Luật Mua", callback_data=f"checkmua_cb|{text}"),
            InlineKeyboardButton("🔍 Kiểm Tra OB", callback_data=f"checkob_cb|{text}")
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

    if data == "run_sieuco_callback":
        await sieuco_command(update, context)
        return
    if data == "run_autobuy_callback": 
        await autobuy_command(update, context)
        return
    if data == "run_dipbuy_callback": 
        await dipbuy_command(update, context)
        return
    if data == "run_nangday_callback":
        await nangday_command(update, context)
        return
    if data == "run_sectors_callback": 
        await sectors_command(update, context)
        return
    if data == "run_foreign_callback":
        await foreign_command(update, context)
        return
    if data == "run_quytec_callback":
        await quytec_command(update, context)
        return

    if data.startswith("checkmua_cb|"):
        symbol = data.split("|")[1]
        context.args = [symbol]
        await check_buy_discipline_command(update, context)
        return

    if data.startswith("checkob_cb|"):
        symbol = data.split("|")[1]
        context.args = [symbol]
        await check_buy_discipline_command(update, context)
        return

    if data.startswith("sosanh_cb|"):
        symbol = data.split("|")[1]
        context.args = [symbol]
        await sosanh_muavumedia_command(update, context)
        return

    parts = data.split("|")
    if len(parts) == 3:
        symbol, resolution, count = parts[0], parts[1], int(parts[2])
        msg = await query.message.reply_text(f"⏳ Đang phân tích {symbol.upper()} (Khung: {resolution})...")
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
        except Exception as e:
            await msg.edit_text(f"❌ Lỗi: {html.escape(str(e))}", parse_mode="HTML")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    welcome_text = (
        "👋 <b>CHÀO MỪNG BẠN ĐẾN VỚI QUANT TRADING SYSTEM PRO!</b>\n\n"
        "1️⃣ <b>TRA CỨU NHANH:</b> Gửi mã cổ phiếu bất kỳ (VD: <code>TCB</code>, <code>HPG</code>)\n"
        "2️⃣ <b>KIỂM TRA KỶ LUẬT MUA:</b> Gõ <code>/checkmua TCB</code>\n"
        "3️⃣ <b>ĐỐI CHIẾU & ĐỌC VỊ:</b>\n"
        "   ├ 🔄 /vonglap TCB        : So khớp cấu trúc nến với quá khứ\n"
        "   ├ ⚖️ /sosanh_muavumedia TCB : So sánh Mùa vụ (Macro) vs T+ (Ngắn hạn)\n"
        "   └ 🕵️‍♂️ /docvi TCB        : Đọc vị mánh khóe tạo lập phiên\n"
        "4️⃣ <b>BỘ LỌC TỰ ĐỘNG & TÌM SIÊU CỔ:</b>\n"
        "   ├ 🚀 /sieuco    : Quét tìm Siêu Cổ Phiếu bứt phá\n"
        "   ├ 🎯 /dipbuy    : Bắt đáy an toàn\n"
        "   ├ 📈 /nangday   : Quét VN30 Nâng Đáy Khung Tháng\n"
        "   ├ 🤖 /autobuy   : Tín hiệu mua chuẩn VCP + SMC\n"
        "   ├ 🏢 /sectors   : Dòng tiền Nhóm Ngành (% Vào/Ra)\n"
        "   ├ 🌍 /khoingoai : Top Khối Ngoại Mua/Bán ròng\n"
        "   └ 📜 /quytec    : Cẩm nang lái xe chứng khoán\n"
    )
    await update.message.reply_text(welcome_text, parse_mode="HTML")


# ==========================================
# 5. TIẾN TRÌNH AUTO-SCAN CHẠY NGẦM ĐỊNH KỲ
# ==========================================
def run_scheduled_auto_scan(application):
    import pytz
    vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    print("⏰ Hệ thống Auto-Scan Scheduler đã được kích hoạt chạy ngầm thành công!")
    
    while True:
        try:
            now = datetime.datetime.now(vn_tz)
            current_weekday = now.weekday()
            current_hour = now.hour
            current_minute = now.minute

            is_weekday_afternoon = (0 <= current_weekday <= 3) and (current_hour == 15 and current_minute == 30)
            is_sunday_night = (current_weekday == 6) and (current_hour == 20 and current_minute == 0)

            if is_weekday_afternoon or is_sunday_night:
                if ACTIVE_CHAT_IDS:
                    print(f"🚀 Bắt đầu tiến trình Auto-Scan định kỳ lúc {now.strftime('%Y-%m-%d %H:%M:%S')}...")
                    
                    super_stocks = []
                    for symbol in WATCHLIST_SCAN:
                        res = scan_super_stock(symbol)
                        if res:
                            super_stocks.append(res)
                    
                    if super_stocks:
                        sorted_picks = sorted(super_stocks, key=lambda x: x["score"], reverse=True)[:5]
                        
                        text = "🚨 <b>BẢN TIN AUTO-SCAN: TOP SIÊU CỔ PHIẾU BỨT PHÁ</b>\n"
                        text += f"📅 <i>Thời gian quét tự động: {now.strftime('%d/%m/%Y %H:%M')}</i>\n"
                        text += "====================================\n\n"
                        
                        for idx, item in enumerate(sorted_picks, 1):
                            text += f"{idx}. 🔹 <b>Mã: {item['symbol'].upper()}</b> | Giá: <code>{item['close']}k</code> | Score: <code>{item['score']}%</code>\n"
                            text += f"   • Dòng tiền CMF: <code>{item['cmf']}</code> | RVOL: <code>{item['rvol']}x</code>\n"
                            text += f"   • Điểm mua: <code>{item['entry']}k</code> | Stoploss: <code>{item['stop_loss']}k</code>\n------------------------------------\n"

                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        for chat_id in ACTIVE_CHAT_IDS:
                            try:
                                loop.run_until_complete(
                                    application.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
                                )
                            except Exception as e:
                                print(f"Không thể gửi tin tới chat_id {chat_id}: {e}")
                        loop.close()
                
                time.sleep(70)
            else:
                time.sleep(30)
                
        except Exception as e:
            print(f"Lỗi trong tiến trình Auto-Scan Thread: {e}")
            time.sleep(60)


# ==========================================
# 6. CẤU HÌNH SERVER, SELF-PING & CHẠY BOT
# ==========================================
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Quant Pro Bot Running!")

def run_http_server():
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler).serve_forever()

def self_ping_thread():
    app_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not app_url:
        return
    while True:
        time.sleep(3)
        try:
            requests.get(app_url, timeout=5)
        except Exception:
            pass

if __name__ == "__main__":
    threading.Thread(target=run_http_server, daemon=True).start()
    threading.Thread(target=self_ping_thread, daemon=True).start()
    
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Đăng ký các lệnh chính
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
    
    # Đăng ký các lệnh nâng cao mới
    app.add_handler(CommandHandler("vonglap", vonglap_command))
    app.add_handler(CommandHandler("sosanh_muavumedia", sosanh_muavumedia_command))
    app.add_handler(CommandHandler("docvi", docvi_command))
    
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    # Kích hoạt tiến trình Auto-Scan chạy ngầm định kỳ
    threading.Thread(target=run_scheduled_auto_scan, args=(app,), daemon=True).start()

    print("🚀 Bot Quant Trading Pro đã cập nhật tính năng Đọc vị & Vòng lặp động thành công!")
    
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)
