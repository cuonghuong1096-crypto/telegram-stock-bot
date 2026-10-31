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
from bs4 import BeautifulSoup  # Thư viện cào HTML 24hmoney
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
TCBS_API_KEY = os.environ.get(
    "TCBS_API_KEY", "10001010079-bf495c0f-c321-42c6-bd6c-01d582ef3670"
)
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
# 1. HÀM CALL DATA & GỘP KHUNG THỜI GIAN CHUẨN
# ==========================================
def fetch_stock_data(symbol, resolution="D", count=150):
    symbol = symbol.upper().strip()
    
    api_resolution = "60" if resolution == "60" else "D"
    fetch_count = count * 7 if resolution in ["W", "M"] else count

    cache_key = f"{symbol}_{resolution}"
    now_ts = time.time()

    if cache_key in STOCK_DATA_CACHE:
        cached_df, cached_err, cached_time = STOCK_DATA_CACHE[cache_key]
        if now_ts - cached_time < 60:
            return cached_df, cached_err

    end_time = int(now_ts)
    start_time = end_time - (fetch_count * 86400 * 2)

    tcbs_url = f"https://apifull.tcbs.com.vn/stock-insight/v1/stock/bars?resolution={api_resolution}&ticker={symbol}&from={start_time}&to={end_time}"
    df_raw = pd.DataFrame()
    
    try:
        headers = {"User-Agent": "Mozilla/5.0", "X-API-KEY": TCBS_API_KEY}
        res = requests.get(tcbs_url, headers=headers, timeout=4.5)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                df_raw = pd.DataFrame(data)
                df_raw["time"] = pd.to_datetime(df_raw["tradingDate"])
                df_raw.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    except Exception:
        pass

    if df_raw.empty:
        url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution={api_resolution}&symbol={symbol}&from={start_time}&to={end_time}"
        try:
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://dchart.vndirect.com.vn/"}
            res = requests.get(url, headers=headers, timeout=4.5)
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
        res_err = "Không có dữ liệu"
        STOCK_DATA_CACHE[cache_key] = (pd.DataFrame(), res_err, now_ts)
        return pd.DataFrame(), res_err

    df_raw = df_raw[["time", "Open", "High", "Low", "Close", "Volume"]].sort_values("time").reset_index(drop=True)
    
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
        res = {"status": "BULLISH", "score_modifier": 1.1, "note": "VN-Index ủng hộ tăng"}
    else:
        res = {"status": "SIDEWAYS", "score_modifier": 0.9, "note": "VN-Index phân hóa tích lũy"}

    VNINDEX_HEALTH_CACHE["data"], VNINDEX_HEALTH_CACHE["timestamp"] = res, now_ts
    return res

# ==========================================
# HÀM CÀO DỮ LIỆU KHỐI NGOẠI TRỰC TIẾP TỪ 24HMONEY
# ==========================================
def get_foreign_trading_data():
    url = "https://24hmoney.vn/indices/vn-index/giao-dich-khoi-ngoai"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7"
        }
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code != 200:
            return None, f"HTTP Status Code: {res.status_code}"

        soup = BeautifulSoup(res.text, 'html.parser')
        results = []
        
        # Quét các dòng bảng (tr, td) trong HTML của trang 24hmoney
        rows = soup.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 3:
                ticker = cols[0].get_text(strip=True)
                if ticker.isalnum() and 3 <= len(ticker) <= 4:
                    val_text = cols[-1].get_text(strip=True).replace(",", "").replace("+", "")
                    try:
                        net_val = float(val_text)
                        results.append({"ticker": ticker, "netValue": net_val})
                    except ValueError:
                        pass

        if results:
            return results[:10], None

        return None, "Không tìm thấy cấu trúc bảng dữ liệu khối ngoại (trang web có thể dùng JavaScript render động)."

    except Exception as e:
        logging.error(f"Error scraping 24hmoney foreign trading data: {e}")
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

📌 <b>ĐÁNH GIÁ CHUNG:</b> (Tóm tắt 1 câu tác động ngắn hạn tới giá)

Lưu ý: Chỉ dựa vào thông tin được cung cấp, trả ra bằng tiếng Việt, ngắn gọn, súc tích, không dông dài.
"""

        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600
        )
        
        ai_analysis = completion.choices[0].message.content
        return f"📰 <b>TỔNG HỢP TIN TỨC: {symbol}</b>\n====================================\n\n" + ai_analysis

    except Exception as e:
        err_msg = str(e)
        logging.error(f"Groq API Error for {symbol}: {err_msg}")
        return f"❌ <b>Lỗi khi gọi Groq AI:</b> <code>{html.escape(err_msg)}</code>"

# ==========================================
# HÀM PHÂN TÍCH KỸ THUẬT & SMC
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
        score += 20; confluence_points.append("Giá nằm trên mây Kumo")
    elif close >= cloud_bottom: 
        score += 10; confluence_points.append("Giá trong mây Kumo")
    else: 
        confluence_points.append("⚠️ Giá nằm dưới mây Kumo (Downtrend)")
    
    if smc["has_ob"]: score += 10; confluence_points.append(f"🔥 [Order Block tại {fmt(smc['ob_price'])}]")
    if smc["has_liquidity_sweep"]: score += 15; confluence_points.append("🎯 [Quét thanh khoản rút chân]")
    
    if cmf > 0.03: 
        score += 15
        confluence_points.append(f"Dòng tiền lớn CMF ({cmf:.2f}) tốt")
    elif cmf < -0.1:
        score -= 20
        confluence_points.append(f"🚨 Dòng tiền rút ròng mạnh (CMF: {cmf:.2f})")

    if is_downtrend_heavy and cmf < 0:
        score = min(score, 40)
        confluence_points.append("🛡️ [Kích hoạt khiên chống Downtrend: Cấm bắt đáy]")

    confidence_score = min(max(int(score * mkt_health["score_modifier"]), 10), 90)

    entry_1 = round(min(close, max(val, close * 0.98)), 2)
    entry_2 = round(close, 2)
    entry_avg = (entry_1 + entry_2) / 2
    
    recent_swing_low = df["Low"].tail(10).min()
    stop_loss = round(min(entry_avg * 0.93, recent_swing_low * 0.98), 2)

    sug_qty, risk_nav_pct = calculate_position_size_1pct(entry_avg, stop_loss)

    if confidence_score >= 65 and not is_downtrend_heavy:
        system_status = "MỞ MUA / GIA TĂNG TỶ TRỌNG"
        final_icon = f"🟢 <b>NÊN MUA (Độ tin cậy: {confidence_score}%)</b>"
        action_note = f"Mở mua quanh <code>{fmt(entry_1)} - {fmt(entry_2)}</code>, Khối lượng: <code>{sug_qty:,} cổ</code>, Stoploss tại <code>{fmt(stop_loss)}</code>."
    elif confidence_score >= 50:
        system_status = "THĂM DÒ THẬN TRỌNG"
        final_icon = f"🟡 <b>THĂM DÒ THẬN TRỌNG (Độ tin cậy: {confidence_score}%)</b>"
        action_note = f"Tỷ trọng nhỏ, quan sát kỹ vùng hỗ trợ, Stoploss tại <code>{fmt(stop_loss)}</code>."
    else:
        system_status = "ĐỨNG NGOÀI / QUAN SÁT (NO TRADE)"
        final_icon = f"🔴 <b>CHƯA ĐỦ ĐIỀU KIỆN MUA (Độ tin cậy: {confidence_score}%)</b>"
        action_note = f"Cổ phiếu đang xu hướng giảm hoặc rủi ro cao. <b>Tuyệt đối đứng ngoài, không bắt đáy!</b>"

    conf_str = "; ".join(confluence_points) if confluence_points else "Tích lũy ổn định"
    long_term_trend = "UPTREND (Trên SMA200)" if not is_downtrend_heavy else "DOWNTREND / RƠI MẠNH"

    text_part1 = f"📊 <b>BÁO CÁO PHÂN TÍCH KỸ THUẬT & SMC: {symbol.upper()}</b>\n"
    text_part1 += f"🗓️ Ngày: {last['time'].strftime('%Y-%m-%d')} | Khung: {resolution} | Giá: {fmt(close)}\n"
    text_part1 += f"🏛️ Trạng thái thị trường: <b>{html.escape(mkt_health['note'])}</b>\n\n"
    text_part1 += f"🌐 <b>1. CẤU TRÚC XU HƯỚNG & DÒNG TIỀN:</b>\n• Xu hướng: {long_term_trend}\n• ADX: <code>{adx:.1f}</code> | CMF: <code>{cmf:.2f}</code> | RVOL: <code>{rvol:.2f}x</code>\n\n"
    text_part1 += f"🏛️ <b>2. CẤU TRÚC SMC & VOLUME PROFILE:</b>\n• Order Block: <code>{'Có (' + fmt(smc['ob_price']) + ')' if smc['has_ob'] else 'Chưa'}</code> | Sweep: <code>{'Có' if smc['has_liquidity_sweep'] else 'Không'}</code>\n• POC: <code>{fmt(poc)}</code> | VAL: <code>{fmt(val)}</code> | VAH: <code>{fmt(vah)}</code>\n\n"
    text_part1 += f"🎯 <b>3. QUẢN TRỊ RỦI RO & VỐN 1% NAV:</b>\n• Khuyến nghị: <b>{system_status}</b>\n• {action_note}\n\n"

    text_part2 = f"⚡ <b>4. TÍN HIỆU HỘI TỤ:</b> {conf_str}\n🏁 <b>5. KẾT LUẬN:</b> {final_icon}"

    keyboard = [
        [
            InlineKeyboardButton("⏱️ 1H", callback_data=f"{symbol}|60|300"),
            InlineKeyboardButton("📅 1D", callback_data=f"{symbol}|D|200"),
            InlineKeyboardButton("📈 1W", callback_data=f"{symbol}|W|100"),
            InlineKeyboardButton("🗓️ 1M", callback_data=f"{symbol}|M|50"),
        ],
        [
            InlineKeyboardButton("🎯 Lọc Đảo Chiều Đáy", callback_data="run_dipbuy_callback"),
            InlineKeyboardButton("🤖 Tín Hiệu AutoBuy", callback_data="run_autobuy_callback"),
        ],
        [
            InlineKeyboardButton("🏢 Dòng Tiền Ngành", callback_data="run_sectors_callback"),
            InlineKeyboardButton("🌍 Khối Ngoại Mua/Bán", callback_data="run_foreign_callback")
        ]
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
    ax.plot(df["time"], df["Close"], label="Giá đóng cửa", color="black", lw=1.2)
    ax.axhline(poc, color="purple", linestyle=":", label=f"POC: {fmt(poc)}", alpha=0.8)
    ax.set_title(f"Phân Tích Kỹ Thuật - {symbol.upper()} ({resolution})", fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return text_part1, text_part2, buf, InlineKeyboardMarkup(keyboard), None


# ==========================================
# 3. HÀM QUÉT CÓ BỘ LỌC CHẶN RỦI RO CHO AUTO-BUY & DIP-BUY
# ==========================================
def process_single_stock_for_autobuy(symbol):
    try:
        df_d, err_d = fetch_stock_data(symbol, resolution="D", count=200)
        if err_d or df_d.empty or len(df_d) < 50: return None

        df_d = calculate_indicators(df_d)
        last_d = df_d.iloc[-1]
        close = last_d["Close"]
        sma200 = last_d.get("SMA200", close)
        cmf_d = last_d.get("CMF", 0.0)
        rvol_d = last_d.get("RVOL", 1.0)
        
        is_near_sma200 = (close >= sma200 * 0.97) and (close < sma200)
        if close < sma200 and not (is_near_sma200 and cmf_d > 0.03):
            return None
        if rvol_d < 1.3:
            return None

        smc = calculate_smc_structures(df_d)
        poc, vah, val = calculate_volume_profile(df_d)

        entry_price = round(close, 2)
        recent_swing_low = df_d["Low"].tail(10).min()
        stop_loss = round(min(entry_price * 0.93, recent_swing_low * 0.98), 2)
        suggested_qty, actual_risk_pct = calculate_position_size_1pct(entry_price, stop_loss)

        score = 50
        if smc["has_ob"]: score += 15
        if smc["has_liquidity_sweep"]: score += 20
        if cmf_d > 0.02: score += 15
        if rvol_d >= 1.5: score += 10

        if score >= 65 and cmf_d > 0:
            return {
                "symbol": symbol, "close": close, "score": min(score, 95),
                "entry": entry_price, "stop_loss": stop_loss,
                "suggested_qty": suggested_qty
            }
    except Exception:
        pass
    return None

async def run_smart_buy_screener_with_progress(msg):
    total_symbols = len(WATCHLIST_SCAN)
    completed = 0
    buy_recommendations = []

    with ThreadPoolExecutor(max_workers=30) as pool:
        futures = {pool.submit(process_single_stock_for_autobuy, symbol): symbol for symbol in WATCHLIST_SCAN}
        
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            if res:
                buy_recommendations.append(res)
            
            if completed % 5 == 0 or completed == total_symbols:
                percent = int((completed / total_symbols) * 100)
                filled_blocks = int(percent / 10)
                bar = "█" * filled_blocks + "░" * (10 - filled_blocks)
                
                try:
                    await msg.edit_text(
                        f"🤖 <b>Đang quét danh sách Auto-Buy an toàn tuyệt đối...</b>\n"
                        f"[{bar}] <b>{percent}%</b> ({completed}/{total_symbols} mã)\n"
                        f"⏳ Đang áp dụng bộ lọc khối lượng & xu hướng chuyên sâu...",
                        parse_mode="HTML"
                    )
                except Exception:
                    pass

    return sorted(buy_recommendations, key=lambda x: x["score"], reverse=True)[:5]


def scan_single_dip_stock(symbol):
    df, err = fetch_stock_data(symbol, resolution="D", count=100)
    if err or df.empty or len(df) < 30: return None
    df = calculate_indicators(df)
    last = df.iloc[-1]
    close, cmf, rvol = last["Close"], last.get("CMF", 0.0), last.get("RVOL", 1.0)
    sma200 = last.get("SMA200", close)
    
    if close < sma200 * 0.95: return None

    smc = calculate_smc_structures(df)
    poc, _, _ = calculate_volume_profile(df)
    recent_high = df["High"].tail(15).max()
    drop_pct = ((recent_high - close) / recent_high) * 100

    if drop_pct >= 4.0 and (close <= poc * 1.02 or smc["has_ob"] or smc["has_liquidity_sweep"]) and cmf > 0:
        score = (drop_pct * 2.0) + (cmf * 30) + (15 if smc["has_liquidity_sweep"] else 0)
        return {
            "symbol": symbol, "close": close, "drop_pct": round(drop_pct, 1),
            "cmf": cmf, "rvol": rvol, "poc": poc, "smc": smc, "score": score
        }
    return None

async def run_dip_buy_screener_with_progress(msg):
    total_symbols = len(WATCHLIST_SCAN)
    completed = 0
    results = []

    with ThreadPoolExecutor(max_workers=30) as pool:
        futures = {pool.submit(scan_single_dip_stock, symbol): symbol for symbol in WATCHLIST_SCAN}
        
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            if res:
                results.append(res)
            
            if completed % 5 == 0 or completed == total_symbols:
                percent = int((completed / total_symbols) * 100)
                filled_blocks = int(percent / 10)
                bar = "█" * filled_blocks + "░" * (10 - filled_blocks)
                
                try:
                    await msg.edit_text(
                        f"🎯 <b>Đang quét cổ phiếu tích lũy an toàn...</b>\n"
                        f"[{bar}] <b>{percent}%</b> ({completed}/{total_symbols} mã)\n"
                        f"⏳ Đang rà soát vùng hỗ trợ...",
                        parse_mode="HTML"
                    )
                except Exception:
                    pass

    return sorted(results, key=lambda x: x["score"], reverse=True)[:6]


def analyze_single_sector_stock(symbol):
    df, err = fetch_stock_data(symbol, resolution="D", count=40)
    if err or df.empty or len(df) < 15: return None
    df = calculate_indicators(df)
    last, prev = df.iloc[-1], df.iloc[-2]
    return {
        "symbol": symbol,
        "pct_change": ((last["Close"] - prev["Close"]) / prev["Close"]) * 100,
        "cmf": last.get("CMF", 0.0), "rvol": last.get("RVOL", 1.0)
    }

async def run_sector_flow_screener_with_progress(msg):
    total_sectors = len(SECTOR_MAP)
    completed_sectors = 0
    sector_results = []

    for sector_name, symbols in SECTOR_MAP.items():
        stock_metrics = []
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(analyze_single_sector_stock, sym) for sym in symbols]
            for f in as_completed(futures):
                res = f.result()
                if res: stock_metrics.append(res)
        
        completed_sectors += 1
        percent = int((completed_sectors / total_sectors) * 100)
        filled_blocks = int(percent / 10)
        bar = "█" * filled_blocks + "░" * (10 - filled_blocks)
        
        try:
            await msg.edit_text(
                f"🏢 <b>Đang phân tích dòng tiền nhóm ngành...</b>\n"
                f"[{bar}] <b>{percent}%</b> ({completed_sectors}/{total_sectors} ngành: {sector_name})\n"
                f"⏳ Đang đo lường chỉ số CMF & RVOL...",
                parse_mode="HTML"
            )
        except Exception:
            pass

        if stock_metrics:
            avg_pct = np.mean([s["pct_change"] for s in stock_metrics])
            avg_cmf = np.mean([s["cmf"] for s in stock_metrics])
            avg_rvol = np.mean([s["rvol"] for s in stock_metrics])
            leader_stock = sorted(stock_metrics, key=lambda x: x["cmf"] + x["rvol"], reverse=True)[0]["symbol"]

            sector_results.append({
                "sector": sector_name, "avg_cmf": avg_cmf, "avg_rvol": avg_rvol,
                "leader": leader_stock, "score": (avg_cmf * 40) + (avg_rvol * 20)
            })

    return sorted(sector_results, key=lambda x: x["score"], reverse=True)


# ==========================================
# 4. HANDLERS XỬ LÝ SỰ KIỆN BOT
# ==========================================

async def quytec_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    
    rules_text = (
        "📜 <b>HỆ THỐNG QUY TẮC SỐNG CÒN CỦA NHÓM 5% CHIẾN THẮNG</b>\n"
        "====================================\n\n"
        
        "🏛️ <b>1. QUY TẮC TƯ DUY TẠO LẬP & ĐÁM ĐÔNG (SMART MONEY)</b>\n"
        "• <b>Mua ngầm theo Tạo lập:</b> Tạo lập gom hàng âm thầm ở nền giá đi ngang (sideway) thanh khoản cạn kiệt, không mua đuổi giá trần.\n"
        "• <b>Nói KHÔNG với FOMO:</b> Tuyệt đối không mua đuổi khi cổ phiếu đã chạy xa nền tích lũy hoặc khi tin tốt tràn ngập diễn đàn.\n"
        "• <b>Tránh bẫy hoảng loạn:</b> Khi thị trường giảm sâu về hỗ trợ mạnh với thanh khoản cạn kiệt, không bán tháo theo đám đông. Đó là lúc kiểm tra lại nội tại để gom cùng Tạo lập.\n\n"
        
        "🛑 <b>2. QUY TẮC TÀI KHOẢN & CHỐNG TRẢ THÙ THỊ TRƯỜNG</b>\n"
        "• <b>Cấm tuyệt đối gỡ gạc (Anti-Revenge):</b> Không bao giờ đánh lớn hay mở lệnh gấp thếp để gỡ lỗ. Lệnh sau độc lập hoàn toàn với lệnh trước.\n"
        "• <b>Rủi ro tối đa mỗi lệnh:</b> Số tiền lỗ tối đa cho một giao dịch không vượt quá 1% đến 2% tổng NAV. Sai là tự động văng ra.\n"
        "• <b>Cool-down nghiêm ngặt:</b> Thua liên tục 2 lệnh, khóa app giao dịch ngay lập tức trong 2 tuần để đưa não bộ về trạng thái cân bằng.\n\n"
        
        "📊 <b>3. QUY TẮC THỜI ĐIỂM & HIỆU SUẤT VỐN</b>\n"
        "• <b>Điểm mua tối ưu:</b> Tích lũy đủ + Giá tốt + Khả năng bật lên càng sớm càng tốt để tránh chôn vốn.\n"
        "• <b>Thời gian chờ (Time Stop):</b> Sau một tuần mua mà cổ phiếu ì ạch không chạy, kiên quyết cơ cấu thu hồi vốn.\n\n"
        
        "💎 <b>4. QUY TẮC NỘI TAI DOANH NGHIỆP (\"THÙNG ĐẶC\")</b>\n"
        "• <b>Loại bỏ thùng rỗng:</b> Tránh xa cổ phiếu kỳ vọng ảo. Chỉ chọn doanh nghiệp \"thùng đặc\" (Tài sản thật, dòng tiền kinh doanh dương, nợ an toàn).\n"
        "• <b>Phòng thủ Thiên nga đen:</b> Doanh nghiệp tốt đến mấy vẫn phải đề phòng biến cố vĩ mô hệ thống bất ngờ.\n\n"
        
        "🌐 <b>5. QUY TẮC XU HƯỚNG & KỶ LUẬT THÉP</b>\n"
        "• <b>Thuận thị trường chung:</b> Không chống lại xu hướng lớn của VN-Index.\n"
        "• <b>Cắt lỗ là chi phí kinh doanh:</b> Vi phạm âm 7-8% hoặc gãy hỗ trợ kỹ thuật là <b>phải bán ngay lập tức</b>, tuyệt đối không gồng lỗ hay bình quân giá xuống.\n"
        "• <b>Sự im lặng là vàng:</b> Ngồi im, ngắt kết nối bảng điện khi thị trường chưa rõ ràng chính là cách kiếm tiền an toàn nhất.\n"
        "====================================\n"
        "🎯 <i>Ghi nhớ và đọc lại mỗi ngày để bảo vệ huyết mạch tài khoản!</i>"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("🤖 Kiểm tra Auto-Buy", callback_data="run_autobuy_callback"),
            InlineKeyboardButton("🎯 Lọc Đáy An Toàn", callback_data="run_dipbuy_callback")
        ]
    ]
    
    await target_msg.reply_text(rules_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")


async def autobuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    msg = await target_msg.reply_text(
        "🤖 <b>Đang khởi tạo tiến trình quét Auto-Buy...</b>\n"
        "[░░░░░░░░░░] <b>0%</b> (0 mã)\n"
        "⏳ Đang kết nối máy chủ dữ liệu...",
        parse_mode="HTML"
    )
    
    try:
        picks = await run_smart_buy_screener_with_progress(msg)
        if not picks:
            await msg.edit_text("⚠️ Hiện tại không có mã nào đủ chuẩn an toàn.")
            return
            
        text = "🎯 <b>DANH SÁCH CỔ PHIẾU AUTO-BUY AN TOÀN</b>\n====================================\n\n"
        for idx, item in enumerate(picks, 1):
            text += f"{idx}. 🔹 <b>Mã: {item['symbol'].upper()}</b> | Giá: <code>{item['close']:.1f}k</code> | Score: <code>{item['score']}%</code>\n"
            text += f"   • Vùng mua: <code>{item['entry']}k</code> | Stoploss: <code>{item['stop_loss']}k</code> | Qty: <code>{item['suggested_qty']:,} cổ</code>\n"
        await msg.edit_text(text, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ <b>Lỗi:</b> {html.escape(str(e))}", parse_mode="HTML")


async def dipbuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    msg = await target_msg.reply_text(
        "🎯 <b>Đang khởi tạo tiến trình lọc Đáy...</b>\n"
        "[░░░░░░░░░░] <b>0%</b> (0 mã)\n"
        "⏳ Đang kết nối máy chủ dữ liệu...",
        parse_mode="HTML"
    )
    
    try:
        picks = await run_dip_buy_screener_with_progress(msg)
        if not picks:
            await msg.edit_text("⚠️ Chưa phát hiện cổ phiếu nào đảo chiều cạn cung đạt chuẩn.")
            return
            
        text = "🎯 <b>DANH SÁCH CỔ PHIẾU TÍCH LŨY AN TOÀN</b>\n====================================\n\n"
        for idx, item in enumerate(picks, 1):
            smc_tag = "🔥 [Sweep]" if item["smc"]["has_liquidity_sweep"] else ("⚡ [OB]" if item["smc"]["has_ob"] else "📊 [POC]")
            text += f"{idx}. 🔹 <b>Mã: {item['symbol'].upper()}</b> {smc_tag}\n"
            text += f"   • Giá: <code>{item['close']:.1f}k</code> | Giảm từ đỉnh: <code>-{item['drop_pct']}%</code>\n"
            text += f"   • CMF: <code>{item['cmf']:.2f}</code> | RVOL: <code>{item['rvol']:.2f}x</code> | Hỗ trợ: <code>{item['poc']:.1f}k</code>\n------------------------------------\n"
        await msg.edit_text(text, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ <b>Lỗi:</b> {html.escape(str(e))}", parse_mode="HTML")


async def sectors_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    msg = await target_msg.reply_text(
        "🏢 <b>Đang khởi tạo phân tích Nhóm Ngành...</b>\n"
        "[░░░░░░░░░░] <b>0%</b> (0 ngành)\n"
        "⏳ Đang kết nối máy chủ dữ liệu...",
        parse_mode="HTML"
    )
    
    try:
        sectors = await run_sector_flow_screener_with_progress(msg)
        if not sectors: 
            await msg.edit_text("⚠️ Không lấy được dữ liệu ngành.")
            return
            
        text = "🔥 <b>BẢN ĐỒ DÒNG TIỀN THEO NHÓM NGÀNH</b>\n====================================\n\n"
        for idx, sec in enumerate(sectors, 1):
            icon = "🟢 [HÚT DÒNG TIỀN]" if sec['avg_cmf'] > 0.02 else "🔴 [RÚT LUI]"
            text += f"{idx}. 🏢 <b>{sec['sector'].upper()}</b> {icon}\n   • CMF: <code>{sec['avg_cmf']:.2f}</code> | Leader: <code>{sec['leader']}</code>\n"
        await msg.edit_text(text, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ <b>Lỗi:</b> {html.escape(str(e))}", parse_mode="HTML")


async def foreign_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    msg = await target_msg.reply_text("⏳ Đang cào dữ liệu khối ngoại từ 24hmoney...", parse_mode="HTML")
    
    try:
        loop = asyncio.get_running_loop()
        data, err = await loop.run_in_executor(executor, get_foreign_trading_data)
        
        if not data:
            error_detail = html.escape(err) if err else "Không nhận được phản hồi từ nguồn 24hmoney"
            await msg.edit_text(
                f"⚠️ <b>Không lấy được dữ liệu khối ngoại.</b>\n"
                f"🔍 <i>Chi tiết lỗi:</i> <code>{error_detail}</code>", 
                parse_mode="HTML"
            )
            return
            
        text = "🌍 <b>THÔNG TIN GIAO DỊCH KHỐI NGOẠI (24HMONEY)</b>\n====================================\n\n"
        
        if isinstance(data, list):
            for idx, item in enumerate(data[:10], 1):
                ticker = item.get("ticker") or item.get("symbol") or "N/A"
                net_val_raw = item.get("netValue") or item.get("net_value") or item.get("netVal") or 0
                net_val = net_val_raw / 1e9 if abs(net_val_raw) > 1e6 else net_val_raw
                action_icon = "🟢 Mua ròng" if net_val > 0 else "🔴 Bán ròng"
                text += f"{idx}. <b>{ticker}</b>: <code>{net_val:+.2f} tỷ</code> ({action_icon})\n"
        else:
            text += f"<code>{html.escape(str(data))}</code>"
            
        await msg.edit_text(text, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ <b>Lỗi hệ thống:</b> {html.escape(str(e))}", parse_mode="HTML")


async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("⚠️ Vui lòng nhập mã cổ phiếu cần xem tin. Ví dụ: <code>/tin VCG</code>", parse_mode="HTML")
        return
        
    symbol = args[0].upper()
    msg = await update.message.reply_text(f"⏳ Groq AI đang phân tích tin tức mới nhất của <b>{symbol}</b>...", parse_mode="HTML")
    
    try:
        loop = asyncio.get_running_loop()
        news_text = await loop.run_in_executor(executor, get_latest_stock_news, symbol)
        
        try:
            await msg.edit_text(news_text, parse_mode="HTML", disable_web_page_preview=True)
        except Exception:
            clean_text = news_text.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "")
            await msg.edit_text(clean_text, disable_web_page_preview=True)
            
    except Exception as e:
        await msg.edit_text(f"❌ Không thể tải tin tức: {html.escape(str(e))}", parse_mode="HTML")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().upper()
    if len(text) < 3 or len(text) > 5 or not text.isalnum():
        await update.message.reply_text("⚠️ Vui lòng nhập mã hợp lệ (VD: VHM, HPG, TCB)!")
        return

    keyboard = [
        [
            InlineKeyboardButton("⏱️ 1H", callback_data=f"{text}|60|300"),
            InlineKeyboardButton("📅 1D", callback_data=f"{text}|D|200"),
            InlineKeyboardButton("📈 1W", callback_data=f"{text}|W|100"),
            InlineKeyboardButton("🗓️ 1M", callback_data=f"{text}|M|50"),
        ],
    ]
    await update.message.reply_text(
        f"🔍 Đã nhận mã <b>{text}</b>. Chọn khung thời gian phân tích:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="HTML"
    )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "run_autobuy_callback": 
        await autobuy_command(update, context)
        return
    if data == "run_dipbuy_callback": 
        await dipbuy_command(update, context)
        return
    if data == "run_sectors_callback": 
        await sectors_command(update, context)
        return
    if data == "run_foreign_callback":
        await foreign_command(update, context)
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
                err_msg = result[4] if (result and len(result) >= 5) else "Không lấy được dữ liệu từ máy chủ."
                await msg.edit_text(f"❌ <b>Lỗi phân tích:</b> {html.escape(str(err_msg))}", parse_mode="HTML")
                return
                
            text_part1, text_part2, img_buf, reply_markup, err = result
            chat_id = query.message.chat_id
            
            await msg.delete()
            await context.bot.send_photo(chat_id=chat_id, photo=img_buf, caption=text_part1, parse_mode="HTML")
            await context.bot.send_message(chat_id=chat_id, text=text_part2, reply_markup=reply_markup, parse_mode="HTML")
            
        except asyncio.TimeoutError:
            await msg.edit_text("⚠️ <b>Hệ thống quá tải hoặc máy chủ phản hồi chậm.</b> Vui lòng bấm lại!", parse_mode="HTML")
        except Exception as e:
            await msg.edit_text(f"❌ <b>Lỗi:</b> {html.escape(str(e))}", parse_mode="HTML")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "👋 <b>CHÀO MỪNG BẠN ĐẾN VỚI QUANT TRADING SYSTEM PRO (VSA + SMC)!</b>\n\n"
        "1️⃣ <b>TRA CỨU NHANH:</b> Gửi mã cổ phiếu bất kỳ (VD: <code>TCB</code>, <code>HPG</code>)\n"
        "2️⃣ <b>TỔNG HỢP TIN TỨC AI:</b> Gửi lệnh <code>/tin VCG</code> để Groq AI phân tích Tốt / Xấu\n"
        "3️⃣ <b>BỘ LỌC & GIAO DỊCH:</b>\n"
        "   ├ 🎯 /dipbuy    : Bắt đáy an toàn sau đợt giảm\n"
        "   ├ 🤖 /autobuy   : Tín hiệu mua chuẩn VCP + SMC\n"
        "   ├ 🏢 /sectors   : Dòng tiền Nhóm Ngành\n"
        "   ├ 🌍 /khoingoai : Top Mua/Bán ròng Khối Ngoại (24hmoney)\n"
        "   └ 📜 /quytec    : Xem bộ quy tắc sống còn của nhóm 5%\n"
    )
    await update.message.reply_text(welcome_text, parse_mode="HTML")


# ==========================================
# 5. CẤU HÌNH SERVER, SELF-PING & CHẠY BOT
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
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("autobuy", autobuy_command))
    app.add_handler(CommandHandler("dipbuy", dipbuy_command))
    app.add_handler(CommandHandler("sectors", sectors_command))
    app.add_handler(CommandHandler("nganh", sectors_command))
    app.add_handler(CommandHandler("khoingoai", foreign_command))  
    app.add_handler(CommandHandler("tin", news_command))
    app.add_handler(CommandHandler("quytec", quytec_command))
    app.add_handler(CommandHandler("quytrach", quytec_command))
    
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    print("🚀 Bot Quant Trading Pro đã khởi động thành công với hàm cào 24hmoney mới!")
    
    app.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES
    )
