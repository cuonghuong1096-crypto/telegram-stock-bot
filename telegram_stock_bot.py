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
import feedparser  # Thư viện cào tin tức RSS
from groq import Groq  # Thư viện Groq AI
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
# CẤU HÌNH TOKEN & API KEY CHUẨN TCBS OPENAPI
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TCBS_API_KEY = os.environ.get("TCBS_API_KEY", "10001010079-bf495c0f-c321-42c6-bd6c-01d582ef3670")
TCBS_OTP = os.environ.get("TCBS_OTP", "111111") # Mã Smart OTP (IOTP) từ TCInvest
GOOGLE_SHEET_API_URL = os.environ.get("GOOGLE_SHEET_API_URL", "")
DEFAULT_NAV_VND = 500_000_000

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Cache lưu JWT token để tránh gọi API lấy token liên tục
TCBS_TOKEN_CACHE = {"token": None, "time": 0}

def get_tcbs_jwt_token():
    """
    Thực hiện trao đổi API Key lấy JWT Token chuẩn theo tài liệu 1.1 của TCBS OpenAPI.
    URL: POST https://openapi.tcbs.com.vn/gaia/v1/oauth2/openapi/token[span_5](start_span)[span_5](end_span)
    """
    now = time.time()
    if TCBS_TOKEN_CACHE["token"] and (now - TCBS_TOKEN_CACHE["time"] < 3600):
        return TCBS_TOKEN_CACHE["token"]

    token_url = "https://openapi.tcbs.com.vn/gaia/v1/oauth2/openapi/token"
    payload = {
        "apiKey": TCBS_API_KEY,
        "otp": TCBS_OTP
    }
    headers = {
        "Content-Type": "application/json"
    }
    try:
        res = requests.post(token_url, json=payload, headers=headers, timeout=6.0)
        if res.status_code == 200:
            data = res.json()
            jwt_token = data.get("token")
            if jwt_token:
                TCBS_TOKEN_CACHE["token"] = jwt_token
                TCBS_TOKEN_CACHE["time"] = now
                return jwt_token
    except Exception as e:
        logging.error(f"Lỗi lấy TCBS JWT Token: {e}")
    
    # Fallback trả về key cứng nếu không lấy được token động
    return TCBS_API_KEY

# ==========================================
# DANH SÁCH WATCHLIST "THÙNG ĐẶC"
# ==========================================
WATCHLIST_SCAN = [
    "VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "STB", "TPB",
    "VHM", "VIC", "VRE", "KDH", "NLG", "KBC", "VGC", "HDG",
    "SSI", "VCI", "MBS", "FTS", "HCM", "VND", "SHS",
    "HPG", "HSG", "VCG", "HT1", "KSB", "CTD", "HHV",
    "MWG", "PNJ", "FRT", "DGW", "MSN", "VNM", "SAB", "QNS",
    "FPT", "CTR", "VTP", "GEE", "REE", "POW",
    "GAS", "PLX", "PVS", "PVD", "BSR",
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
STOCK_DATA_CACHE = {} 
executor = ThreadPoolExecutor(max_workers=30)

# ==========================================
# 1. HÀM GỌI DỮ LIỆU NẾN TỪ TCBS OPENAPI CHUẨN
# ==========================================
def fetch_stock_data(symbol, resolution="D", count=150):
    symbol = symbol.upper().strip()
    cache_key = f"{symbol}_{resolution}"
    now_ts = time.time()

    if cache_key in STOCK_DATA_CACHE:
        cached_df, cached_err, cached_time = STOCK_DATA_CACHE[cache_key]
        if now_ts - cached_time < 60:
            return cached_df, cached_err

    end_time = int(now_ts)
    if resolution == "60":
        res_param = "60"
        start_time = end_time - (count * 3600 * 4)
    elif resolution == "W":
        res_param = "D"
        count = max(count * 5, 500)
        start_time = end_time - (count * 86400 * 2)
    elif resolution == "M":
        res_param = "D"
        count = max(count * 20, 1000)
        start_time = end_time - (count * 86400 * 2)
    else:
        res_param = "D"
        start_time = end_time - (count * 86400 * 2)

    # Sử dụng đúng domain chính thức openapi.tcbs.com.vn
    tcbs_url = f"https://openapi.tcbs.com.vn/stock-insight/v1/stock/bars?resolution={res_param}&ticker={symbol}&from={start_time}&to={end_time}"
    df_raw = pd.DataFrame()
    try:
        token = get_tcbs_jwt_token()
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        res = requests.get(tcbs_url, headers=headers, timeout=5.0)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                df_raw = pd.DataFrame(data)
                df_raw["time"] = pd.to_datetime(df_raw["tradingDate"])
                df_raw.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    except Exception as e:
        err_msg = str(e)
        STOCK_DATA_CACHE[cache_key] = (pd.DataFrame(), err_msg, now_ts)
        return pd.DataFrame(), err_msg

    if df_raw.empty:
        # Fallback sang domain phụ apifull nếu cần thiết
        try:
            fallback_url = f"https://apifull.tcbs.com.vn/stock-insight/v1/stock/bars?resolution={res_param}&ticker={symbol}&from={start_time}&to={end_time}"
            res_fb = requests.get(fallback_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=4.0)
            if res_fb.status_code == 200:
                data_fb = res_fb.json()
                if isinstance(data_fb, list) and len(data_fb) > 0:
                    df_raw = pd.DataFrame(data_fb)
                    df_raw["time"] = pd.to_datetime(df_raw["tradingDate"])
                    df_raw.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
        except Exception:
            pass

    if df_raw.empty:
        err_msg = "Không có dữ liệu từ TCBS OpenAPI"
        STOCK_DATA_CACHE[cache_key] = (pd.DataFrame(), err_msg, now_ts)
        return pd.DataFrame(), err_msg

    df_raw = df_raw[["time", "Open", "High", "Low", "Close", "Volume"]].sort_values("time").reset_index(drop=True)

    if resolution == "W":
        df_raw.set_index("time", inplace=True)
        df_raw = df_raw.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna().reset_index()
    elif resolution == "M":
        df_raw.set_index("time", inplace=True)
        df_raw = df_raw.resample("ME").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna().reset_index()

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
    if risk_per_share_vnd <= 0: 
        return 1000, 0.0
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
    df, err = fetch_stock_data("VNINDEX", resolution="D", count=40)
    if err or df.empty:
        return {"status": "NEUTRAL", "score_modifier": 1.0, "note": "VN-Index ổn định"}

    df = calculate_indicators(df)
    last = df.iloc[-1]
    close, ema20 = last["Close"], last.get("EMA20", last["Close"])

    if close >= ema20:
        return {"status": "BULLISH", "score_modifier": 1.1, "note": "VN-Index ủng hộ tăng"}
    else:
        return {"status": "SIDEWAYS", "score_modifier": 0.9, "note": "VN-Index phân hóa tích lũy"}


# ==========================================
# 2. LẤY THÔNG TIN ROOM NƯỚC NGOÀI CHUẨN TCBS OPENAPI (`tickerSnaps`)
# ==========================================
def get_foreign_trading_data():
    parsed_list = []
    indexes_to_fetch = [1, 2] # 1 = HOSE, 2 = VN30
    token = get_tcbs_jwt_token()
    
    for idx_val in indexes_to_fetch:
        url = f"https://openapi.tcbs.com.vn/tartarus/v1/tickerSnaps?index={idx_val}"
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }
            res = requests.get(url, headers=headers, timeout=6)
            if res.status_code == 200:
                body = res.json()
                data_items = body.get("data", [])
                for item in data_items:
                    ticker = item.get("symbol", "")
                    if not ticker:
                        continue
                    
                    price = float(item.get("matchPrice", 0) or item.get("refPrice", 0) or 0)
                    buy_qtty = float(item.get("buyForeignQtty", 0) or 0)
                    sell_qtty = float(item.get("sellForeignQtty", 0) or 0)
                    net_qtty = buy_qtty - sell_qtty
                    
                    net_val = (net_qtty * price) / 1000.0 if price > 0 else 0.0
                    
                    if net_val != 0:
                        parsed_list.append({
                            "ticker": ticker.upper(),
                            "netValue": round(net_val, 2)
                        })
        except Exception as e:
            logging.error(f"Error fetching TCBS tickerSnaps for index {idx_val}: {e}")

    if not parsed_list:
        return None, "Không lấy được dữ liệu room nước ngoài từ TCBS OpenAPI."

    unique_dict = {}
    for item in parsed_list:
        ticker = item["ticker"]
        if ticker not in unique_dict or abs(item["netValue"]) > abs(unique_dict[ticker]["netValue"]):
            unique_dict[ticker] = item
            
    final_list = list(unique_dict.values())
    final_list.sort(key=lambda x: abs(x["netValue"]), reverse=True)
    return final_list, None


# ==========================================
# 3. TỔNG HỢP TIN TỨC QUA GROQ AI
# ==========================================
def get_latest_stock_news(symbol):
    symbol = symbol.upper().strip()
    url = f"https://news.google.com/rss/search?q=cổ+phiếu+{symbol}+when:7d&hl=vi&gl=VN&ceid=VN:vi"
    
    try:
        feed = feedparser.parse(url)
        if not feed.entries:
            return f"⚠️ Hiện không tìm thấy bài báo mới nào gần đây về mã <b>{symbol}</b>."
            
        news_items = [f"- Tiêu đề: {entry.title}" for entry in feed.entries[:5]]
        raw_news_text = "\n".join(news_items)

        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            return f"📰 <b>TIN TỨC THÔ {symbol}:</b>\n\n" + raw_news_text

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
"""
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600
        )
        return f"📰 <b>TỔNG HỢP TIN TỨC: {symbol}</b>\n====================================\n\n" + completion.choices[0].message.content
    except Exception as e:
        return f"❌ Lỗi Groq AI: {html.escape(str(e))}"


# ==========================================
# 4. PHÂN TÍCH KỸ THUẬT & SMC
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
# 5. CÁC BỘ LỌC AUTO-BUY, DIP-BUY & SECTORS
# ==========================================
def process_single_stock_for_autobuy(symbol):
    try:
        df_d, err_d = fetch_stock_data(symbol, resolution="D", count=100)
        if err_d or df_d.empty or len(df_d) < 30: return None
        df_d = calculate_indicators(df_d)
        last = df_d.iloc[-1]
        close, cmf, rvol = last["Close"], last.get("CMF", 0.0), last.get("RVOL", 1.0)
        if cmf > 0.02 and rvol >= 1.1:
            return {"symbol": symbol, "close": close, "cmf": cmf, "rvol": rvol, "score": (cmf*50)+(rvol*20)}
    except Exception:
        pass
    return None

async def autobuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    msg = await target_msg.reply_text("🤖 **Đang quét tín hiệu mua chuẩn VCP + SMC từ TCBS OpenAPI...**", parse_mode="Markdown")
    try:
        picks = []
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(process_single_stock_for_autobuy, s) for s in WATCHLIST_SCAN]
            for f in as_completed(futures):
                res = f.result()
                if res: picks.append(res)
        picks = sorted(picks, key=lambda x: x["score"], reverse=True)[:5]
        if not picks:
            await msg.edit_text("⚠️ Hiện không có mã nào thỏa mãn tiêu chuẩn Auto-Buy.")
            return
        text = "🎯 **DANH SÁCH TÍN HIỆU MUA AUTO-BUY**\n====================================\n\n"
        for idx, item in enumerate(picks, 1):
            text += f"{idx}. 🔹 **{item['symbol']}** | Giá: `{item['close']:.1f}k` | CMF: `{item['cmf']:.2f}`\n"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi: {str(e)}")


def scan_single_dip_stock(symbol):
    try:
        df, err = fetch_stock_data(symbol, resolution="D", count=60)
        if err or df.empty or len(df) < 25: return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        rsi = last.get("RSI", 50)
        close = last["Close"]
        if rsi <= 45:
            return {"symbol": symbol, "close": close, "rsi": rsi}
    except Exception:
        pass
    return None

async def dipbuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    msg = await target_msg.reply_text("🎯 **Đang quét cơ hội bắt đáy an toàn...**", parse_mode="Markdown")
    try:
        results = []
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(scan_single_dip_stock, s) for s in WATCHLIST_SCAN]
            for f in as_completed(futures):
                res = f.result()
                if res: results.append(res)
        results = sorted(results, key=lambda x: x["rsi"])[:6]
        if not results:
            await msg.edit_text("⚠️ Hiện không có mã nào đạt chuẩn bắt đáy.")
            return
        text = "🎯 **DANH SÁCH BẮT ĐÂY AN TOÀN (DIP BUY)**\n====================================\n\n"
        for idx, item in enumerate(results, 1):
            text += f"{idx}. 🟢 **{item['symbol']}** | Giá: `{item['close']:.1f}k` | RSI: `{item['rsi']:.1f}`\n"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi: {str(e)}")


async def sectors_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    msg = await target_msg.reply_text("🏢 **Đang phân tích hiệu suất và dòng tiền các Nhóm Ngành từ TCBS...**", parse_mode="Markdown")
    try:
        sector_perf = []
        for sector_name, symbols in SECTOR_MAP.items():
            total_pct, valid = 0, 0
            for sym in symbols[:5]:
                df, err = fetch_stock_data(sym, resolution="D", count=5)
                if not err and not df.empty and len(df) >= 2:
                    pct = ((df.iloc[-1]["Close"] - df.iloc[-2]["Close"]) / df.iloc[-2]["Close"]) * 100
                    total_pct += pct
                    valid += 1
            avg_pct = (total_pct / valid) if valid > 0 else 0.0
            sector_perf.append({"sector": sector_name, "avg_pct": avg_pct})
        
        sector_perf = sorted(sector_perf, key=lambda x: x["avg_pct"], reverse=True)
        text = "🏢 **BẢNG XẾP HẠNG DÒNG TIỀN NHÓM NGÀNH**\n====================================\n\n"
        for idx, item in enumerate(sector_perf, 1):
            icon = "🚀" if item["avg_pct"] >= 0 else "🔻"
            text += f"{idx}. {icon} **{item['sector']}**: Biến động bình quân `{item['avg_pct']:+.2f}%`\n"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi: {str(e)}")


async def foreign_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    msg = await target_msg.reply_text("🌍 Đang gọi thông tin room nước ngoài từ TCBS OpenAPI...", parse_mode="HTML")
    try:
        loop = asyncio.get_running_loop()
        data, err = await loop.run_in_executor(executor, get_foreign_trading_data)
        if not data:
            await msg.edit_text(f"⚠️ Lỗi lấy dữ liệu khối ngoại từ TCBS: {err}", parse_mode="HTML")
            return
            
        text = "🌍 **THÔNG TIN ROOM NƯỚC NGOÀI & GIAO DỊCH KHỐI NGOÀI (TCBS)**\n====================================\n\n"
        for idx, item in enumerate(data[:12], 1):
            act = "🟢 Mua ròng" if item["netValue"] > 0 else "🔴 Bán ròng" if item["netValue"] < 0 else "⚪"
            text += f"{idx}. **{item['ticker']}**: `{item['netValue']:+.2f} tỷ` ({act})\n"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi hệ thống: {html.escape(str(e))}", parse_mode="HTML")


async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("⚠️ Vui lòng nhập mã cổ phiếu! Ví dụ: `/tin VCG`", parse_mode="Markdown")
        return
    symbol = args[0].upper()
    msg = await update.message.reply_text(f"🤖 Groq AI đang tổng hợp tin tức cho mã **{symbol}**...", parse_mode="Markdown")
    try:
        loop = asyncio.get_running_loop()
        res_text = await loop.run_in_executor(executor, get_latest_stock_news, symbol)
        await msg.edit_text(res_text, parse_mode="HTML", disable_web_page_preview=True)
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi: {html.escape(str(e))}", parse_mode="HTML")


async def quytec_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    text = (
        "📜 **BỘ QUY TẮC SỐNG CÒN CỦA NHÓM 5% TRADER THÀNH CÔNG**\n"
        "====================================\n\n"
        "1️⃣ **Cắt lỗ là tôn giáo:** Không bao giờ để khoản lỗ nhỏ biến thành thảm họa.\n"
        "2️⃣ **Thuận dòng tiền lớn:** Chỉ mua khi CMF dương và có sự xác nhận của RVOL.\n"
        "3️⃣ **Không trung bình giá xuống:** Tuyệt đối không mua thêm khi cổ phiếu downtrend.\n"
        "4️⃣ **Quản trị NAV:** Phân bổ rủi ro tối đa 1-2% tổng tài khoản mỗi lệnh.\n"
        "5️⃣ **Kỷ luật thép:** Cảm xúc là kẻ thù số một trong trading!"
    )
    await target_msg.reply_text(text, parse_mode="Markdown")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().upper()
    if len(text) < 3 or len(text) > 5 or not text.isalnum():
        await update.message.reply_text("⚠️ Vui lòng nhập mã cổ phiếu hợp lệ (VD: VHM, HPG, TCB)!")
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
        msg = await query.message.reply_text(f"⏳ Đang phân tích {symbol.upper()} (Khung: {resolution}) từ TCBS OpenAPI...")
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(executor, analyze_stock, symbol, resolution, count)
            if not result or len(result) < 5 or result[4] is not None:
                await msg.edit_text(f"❌ Lỗi: {result[4] if result else 'Không có dữ liệu'}")
                return
            t1, t2, img_buf, markup, _ = result
            chat_id = query.message.chat_id
            await msg.delete()
            await context.bot.send_photo(chat_id=chat_id, photo=img_buf, caption=t1, parse_mode="HTML")
            await context.bot.send_message(chat_id=chat_id, text=t2, reply_markup=markup, parse_mode="HTML")
        except Exception as e:
            await msg.edit_text(f"❌ Lỗi hệ thống: {str(e)}")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "👋 **CHÀO MỪNG BẠN ĐẾN VỚI QUANT TRADING SYSTEM PRO (VSA + SMC)!**\n\n"
        "1️⃣ **TRA CỨU NHANH:** Nhập mã cổ phiếu trực tiếp (VD: `VCB`, `HPG`, `VHM`)\n"
        "2️⃣ **TIN TỨC AI:** Gửi lệnh `/tin VCG` để Groq AI phân tích Tốt / Xấu\n"
        "3️⃣ **BỘ LỌC & GIAO DỊCH:**\n"
        "   ├ 🎯 `/dipbuy`    : Bắt đáy an toàn sau đợt giảm\n"
        "   ├ 🤖 `/autobuy`   : Tín hiệu mua chuẩn VCP + SMC\n"
        "   ├ 🏢 `/sectors`   : Dòng tiền Nhóm Ngành\n"
        "   ├ 🌍 `/khoingoai` : Thông tin Room nước ngoài (TCBS OpenAPI)\n"
        "   └ 📜 `/quytec`    : Xem bộ quy tắc sống còn\n"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


# ==========================================
# 6. HTTP SERVER & CHẠY BOT
# ==========================================
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Quant Pro Bot Running with TCBS OpenAPI!")

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
    app.add_handler(CommandHandler("khoingoai", foreign_command))  
    app.add_handler(CommandHandler("tin", news_command))
    app.add_handler(CommandHandler("quytec", quytec_command))
    
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    print("🚀 Bot Quant Trading Pro đã khởi động thành công với JWT Token chuẩn từ TCBS OpenAPI!")
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)
