import datetime
import io
import json
import logging
import os
import html
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
DEFAULT_NAV_VND = 500_000_000

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

WATCHLIST_SCAN = [
    "VHM", "VIC", "VRE", "HPG", "HSG", "NKG", "SSI", "VCI", "MBS", "FTS",
    "VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "STB", "SHB", "TPB",
    "MWG", "PNJ", "FRT", "DGW", "MSN", "VNM", "SAB", "KDH", "NLG", "DIG", "BSR",
    "CTD", "HHV", "LCG", "HT1", "VCG", "CII", "L14", "VGS", "KBC", "PDR", 
    "CEO", "DXG", "TCH", "NVL", "VPL", "HDG", "NTL", "HDC", "KSB", "MSR",
    "VND", "SHS", "VIX", "HCM", "CTS", "ORS",
    "FPT", "MCH", "PET", "VTP", "CTR", "GEE",
    "VGT", "TNG", "GVR", "ANV", "VHC",
    "PVS", "PVB", "PVD", "GAS", "PLX",
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
# 1. HÀM CALL DATA ĐÃ NỚI LỎNG TIMEOUT AN TOÀN
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
    start_time = end_time - (count * 86400 * 2)

    tcbs_url = f"https://apifull.tcbs.com.vn/stock-insight/v1/stock/bars?resolution={resolution if resolution=='60' else 'D'}&ticker={symbol}&from={start_time}&to={end_time}"
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
        url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution={resolution if resolution=='60' else 'D'}&symbol={symbol}&from={start_time}&to={end_time}"
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
    long_term_trend = "UPTREND (Trên SMA200)" if close > sma200 else "DOWNTREND/PHỤC HỒI"

    confluence_points = []
    score = 50

    if close > cloud_top: score += 20; confluence_points.append("Giá nằm trên mây Kumo")
    elif close >= cloud_bottom: score += 10; confluence_points.append("Giá trong mây Kumo")
    
    if smc["has_ob"]: score += 10; confluence_points.append(f"🔥 [Order Block tại {fmt(smc['ob_price'])}]")
    if smc["has_liquidity_sweep"]: score += 15; confluence_points.append("🎯 [Quét thanh khoản rút chân]")
    if cmf > 0.03: score += 15; confluence_points.append(f"Dòng tiền lớn CMF ({cmf:.2f}) tốt")

    confidence_score = min(max(int(score * mkt_health["score_modifier"]), 15), 90)

    entry_1 = round(max(val, last.get("EMA20", close), close * 0.98), 2)
    entry_2 = round(close, 2)
    entry_avg = (entry_1 + entry_2) / 2
    stop_loss = round(entry_avg * 0.93, 2)

    sug_qty, risk_nav_pct = calculate_position_size_1pct(entry_avg, stop_loss)

    if confidence_score >= 60:
        system_status = "MỞ MUA / GIA TĂNG TỶ TRỌNG"
        final_icon = f"🟢 <b>NÊN MUA (Độ tin cậy: {confidence_score}%)</b>"
        action_note = f"Mở mua quanh <code>{fmt(entry_1)} - {fmt(entry_2)}</code>, Khối lượng: <code>{sug_qty:,} cổ</code>, Stoploss tại <code>{fmt(stop_loss)}</code>."
    else:
        system_status = "THĂM DÒ / QUAN SÁT"
        final_icon = f"🟡 <b>THĂM DÒ TÍCH LŨY (Độ tin cậy: {confidence_score}%)</b>"
        action_note = f"Thăm dò nhỏ quanh <code>{fmt(entry_1)} - {fmt(entry_2)}</code>, Stoploss tại <code>{fmt(stop_loss)}</code>."

    conf_str = "; ".join(confluence_points) if confluence_points else "Tích lũy ổn định"

    text_part1 = f"📊 <b>BÁO CÁO PHÂN TÍCH KỸ THUẬT & SMC: {symbol.upper()}</b>\n"
    text_part1 += f"🗓️ Ngày: {last['time'].strftime('%Y-%m-%d')} | Khung: {resolution} | Giá: {fmt(close)}\n"
    text_part1 += f"🏛️ Trạng thái thị trường: <b>{html.escape(mkt_health['note'])}</b>\n\n"
    text_part1 += f"🌐 <b>1. CẤU TRÚC XU HƯỚNG & DÒNG TIỀN:</b>\n• Xu hướng: {long_term_trend}\n• ADX: <code>{adx:.1f}</code> | CMF: <code>{cmf:.2f}</code> | RVOL: <code>{rvol:.2f}x</code>\n\n"
    text_part1 += f"🏛️ <b>2. CẤU TRÚC SMC & VOLUME PROFILE:</b>\n• Order Block: <code>{'Có (' + fmt(smc['ob_price']) + ')' if smc['has_ob'] else 'Chưa'}</code> | Sweep: <code>{'Có' if smc['has_liquidity_sweep'] else 'Không'}</code>\n• POC: <code>{fmt(poc)}</code> | VAL: <code>{fmt(val)}</code> | VAH: <code>{fmt(vah)}</code>\n\n"
    text_part1 += f"🎯 <b>3. QUẢN TRỊ RỦI RO & VỐN 1% NAV:</b>\n• Khuyến nghị: <b>{system_status}</b>\n• {action_note}\n\n"

    text_part2 = f"⚡ <b>4. TÍN HIỆU HỘI TỤ:</b> {conf_str}\n🏁 <b>5. KẾT LUẬN:</b> {final_icon}"

    keyboard = [
        [
            InlineKeyboardButton("⏱️ Khung 1H", callback_data=f"{symbol}|60|300"),
            InlineKeyboardButton("📅 Khung 1D", callback_data=f"{symbol}|D|200"),
            InlineKeyboardButton("📈 Khung 1W", callback_data=f"{symbol}|W|150"),
        ],
        [
            InlineKeyboardButton("🎯 Lọc Đảo Chiều Đáy (/dipbuy)", callback_data="run_dipbuy_callback"),
            InlineKeyboardButton("🤖 Tín Hiệu AutoBuy (/autobuy)", callback_data="run_autobuy_callback"),
        ],
        [InlineKeyboardButton("🏢 Dòng Tiền Ngành (/sectors)", callback_data="run_sectors_callback")]
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
# 3. HÀM QUÉT CÓ THANH TIẾN TRÌNH % CHO TẤT CẢ TÍNH NĂNG
# ==========================================
def process_single_stock_for_autobuy(symbol):
    try:
        df_d, err_d = fetch_stock_data(symbol, resolution="D", count=100)
        if err_d or df_d.empty or len(df_d) < 30: return None

        df_d = calculate_indicators(df_d)
        last_d = df_d.iloc[-1]
        close = last_d["Close"]
        
        cmf_d = last_d.get("CMF", 0.0)
        rvol_d = last_d.get("RVOL", 1.0)
        smc = calculate_smc_structures(df_d)
        poc, vah, val = calculate_volume_profile(df_d)

        entry_price = round(max(val, close * 0.98), 2)
        stop_loss = round(entry_price * 0.93, 2)
        suggested_qty, actual_risk_pct = calculate_position_size_1pct(entry_price, stop_loss)

        score = 50
        if smc["has_ob"]: score += 15
        if smc["has_liquidity_sweep"]: score += 20
        if cmf_d > 0.02: score += 15
        if rvol_d >= 1.1: score += 10

        if score >= 65:
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
                        f"🤖 <b>Đang quét danh sách Auto-Buy...</b>\n"
                        f"[{bar}] <b>{percent}%</b> ({completed}/{total_symbols} mã)\n"
                        f"⏳ Đang xử lý dữ liệu thị trường...",
                        parse_mode="HTML"
                    )
                except Exception:
                    pass

    return sorted(buy_recommendations, key=lambda x: x["score"], reverse=True)[:5]


def scan_single_dip_stock(symbol):
    df, err = fetch_stock_data(symbol, resolution="D", count=80)
    if err or df.empty or len(df) < 30: return None
    df = calculate_indicators(df)
    last = df.iloc[-1]
    close, cmf, rvol = last["Close"], last.get("CMF", 0.0), last.get("RVOL", 1.0)
    smc = calculate_smc_structures(df)
    poc, _, _ = calculate_volume_profile(df)
    recent_high = df["High"].tail(15).max()
    drop_pct = ((recent_high - close) / recent_high) * 100

    if drop_pct >= 4.0 and (close <= poc * 1.02 or smc["has_ob"] or smc["has_liquidity_sweep"]) and cmf > -0.05:
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
                        f"🎯 <b>Đang quét cổ phiếu đảo chiều từ đáy...</b>\n"
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
# 4. HANDLERS VỚI TIẾN TRÌNH % ĐẦY ĐỦ CHO TẤT CẢ LỆNH
# ==========================================
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
            await msg.edit_text("⚠️ Hiện tại chưa có mã nào thỏa mãn tiêu chí của Auto-Buy.")
            return
            
        text = "🎯 <b>DANH SÁCH CỔ PHIẾU AUTO-BUY</b>\n====================================\n\n"
        for idx, item in enumerate(picks, 1):
            text += f"{idx}. 🔹 <b>Mã: {item['symbol'].upper()}</b> | Giá: <code>{item['close']:.1f}k</code> | Score: <code>{item['score']}%</code>\n"
            text += f"   • Vùng mua: <code>{item['entry']}k</code> | Stoploss: <code>{item['stop_loss']}k</code> | Qty: <code>{item['suggested_qty']:,} cổ</code>\n"
        
        await msg.edit_text(text, parse_mode="HTML")
        
    except Exception as e:
        await msg.edit_text(f"❌ <b>Lỗi tiến trình Auto-Buy:</b> {html.escape(str(e))}", parse_mode="HTML")


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
            
        text = "🎯 <b>DANH SÁCH CỔ PHIẾU ĐẢO CHIỀN TỪ ĐÁY</b>\n====================================\n\n"
        for idx, item in enumerate(picks, 1):
            smc_tag = "🔥 [Sweep]" if item["smc"]["has_liquidity_sweep"] else ("⚡ [OB]" if item["smc"]["has_ob"] else "📊 [POC]")
            text += f"{idx}. 🔹 <b>Mã: {item['symbol'].upper()}</b> {smc_tag}\n"
            text += f"   • Giá: <code>{item['close']:.1f}k</code> | Giảm từ đỉnh: <code>-{item['drop_pct']}%</code>\n"
            text += f"   • CMF: <code>{item['cmf']:.2f}</code> | RVOL: <code>{item['rvol']:.2f}x</code> | Hỗ trợ: <code>{item['poc']:.1f}k</code>\n------------------------------------\n"
        await msg.edit_text(text, parse_mode="HTML")
        
    except Exception as e:
        await msg.edit_text(f"❌ <b>Lỗi tiến trình Dip-Buy:</b> {html.escape(str(e))}", parse_mode="HTML")


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
        await msg.edit_text(f"❌ <b>Lỗi phân tích Nhóm Ngành:</b> {html.escape(str(e))}", parse_mode="HTML")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().upper()
    if len(text) < 3 or len(text) > 5 or not text.isalnum():
        await update.message.reply_text("⚠️ Vui lòng nhập mã hợp lệ (VD: VHM, BSR, HT1)!")
        return

    keyboard = [
        [
            InlineKeyboardButton("⏱️ Khung 1H", callback_data=f"{text}|60|300"),
            InlineKeyboardButton("📅 Khung 1D", callback_data=f"{text}|D|200"),
            InlineKeyboardButton("📈 Khung 1W", callback_data=f"{text}|W|150"),
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
            result = future.result(timeout=15)
            if not result or len(result) < 5:
                await msg.edit_text("❌ Lỗi dữ liệu: Không lấy được thông tin từ máy chủ dữ liệu.")
                return
                
            text_part1, text_part2, img_buf, reply_markup, err = result
            if err:
                await msg.edit_text(f"❌ Lỗi: {html.escape(str(err))}")
            else:
                chat_id = query.message.chat_id
                await msg.delete()
                await context.bot.send_photo(chat_id=chat_id, photo=img_buf, caption=text_part1, parse_mode="HTML")
                await context.bot.send_message(chat_id=chat_id, text=text_part2, reply_markup=reply_markup, parse_mode="HTML")
        except Exception as e:
            await msg.edit_text(f"❌ <b>Máy chủ không phản hồi:</b> Lỗi API dữ liệu ({html.escape(str(e))}). Vui lòng bấm lại!", parse_mode="HTML")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "👋 <b>CHÀO MỪNG BẠN ĐẾN VỚI QUANT TRADING SYSTEM PRO (VSA + SMC)!</b>\n\n"
        "1️⃣ <b>TRA CỨU NHANH:</b> Gửi mã cổ phiếu bất kỳ (VD: <code>VHM</code>, <code>HT1</code>)\n"
        "2️⃣ <b>BỘ LỌC CHIẾN LƯỢC:</b>\n"
        "   ├ 🎯 /dipbuy  : Bắt đáy & Tích lũy sau đợt giảm\n"
        "   ├ 🤖 /autobuy : Tín hiệu mua chuẩn VCP + SMC\n"
        "   └ 🏢 /sectors : Dòng tiền Nhóm Ngành\n"
    )
    await update.message.reply_text(welcome_text, parse_mode="HTML")


# ==========================================
# 5. CẤU HÌNH SERVER & CHẠY BOT (ĐÃ FIX CHO RENDER)
# ==========================================
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Quant Pro Bot Running!")

def run_http_server():
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler).serve_forever()

if __name__ == "__main__":
    # Khởi chạy HTTP Server giữ port cho Render
    threading.Thread(target=run_http_server, daemon=True).start()
    
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("autobuy", autobuy_command))
    app.add_handler(CommandHandler("dipbuy", dipbuy_command))
    app.add_handler(CommandHandler("sectors", sectors_command))
    app.add_handler(CommandHandler("nganh", sectors_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    print("🚀 Bot Quant Trading Pro đã khởi động thành công trên Render!")
    
    # Quan trọng: Thêm allowed_updates để đồng bộ dữ liệu chuẩn xác trên môi trường Cloud
    app.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES
    )
