import datetime
import io
import json
import logging
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
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

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

WATCHLIST_SCAN = [
    "VHM", "VIC", "VRE", "HPG", "HSG", "NKG", "SSI", "VCI", "MBS", "FTS",
    "VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "STB", "SHB", "TPB",
    "MWG", "PNJ", "FRT", "DGW", "MSN", "VNM", "SAB", "KDH", "NLG", "DIG", "BSR"
]

ACTIVE_CHAT_IDS = set()
ALERTED_TODAY = set()

# ==========================================
# QUẢN LÝ LƯU TRỮ DỮ LIỆU BỀN VỮNG (JSON)
# ==========================================
PORTFOLIO_FILE = "user_portfolios.json"

def load_portfolios():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {int(k): v for k, v in data.items()}
        except Exception as e:
            logging.error(f"Lỗi đọc file portfolio: {e}")
    return {}

def save_portfolios():
    try:
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(USER_PORTFOLIOS, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.error(f"Lỗi ghi file portfolio: {e}")

USER_PORTFOLIOS = load_portfolios()

executor = ThreadPoolExecutor(max_workers=8)


# ==========================================
# 1. HÀM CALL DATA & TÍNH TOÁN TOÀN DIỆN
# ==========================================
def fetch_stock_data(symbol, resolution="D", count=250):
    symbol = symbol.upper().strip()
    end_time = int(datetime.datetime.now().timestamp())

    if resolution in ["1", "5", "15", "30", "60"]:
        res_param = resolution
        count = max(count, 450)
        start_time = end_time - (count * int(resolution) * 60 * 4)
    elif resolution == "W":
        res_param = "D"
        count = 1000 
        start_time = end_time - (count * 86400 * 3)
    else:
        res_param = "D"
        count = max(count, 300)
        start_time = end_time - (count * 86400 * 3)

    tcbs_url = f"https://apifull.tcbs.com.vn/stock-insight/v1/stock/bars?resolution={res_param}&ticker={symbol}&from={start_time}&to={end_time}"
    df_raw = pd.DataFrame()
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "X-API-KEY": TCBS_API_KEY,
        }
        res = requests.get(tcbs_url, headers=headers, timeout=3)
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
            res = requests.get(url, headers=headers, timeout=3)
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
        df_weekly = df_raw.resample("W").agg({
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

    confidence_score = min(max(score, 10), 88)

    is_below_val = (close < val)
    if is_below_val:
        zone_desc = f"Vùng **{fmt(val)} - {fmt(vah)}** (Dưới VAL): Trở thành **KHÁNG CỰ MẠNH**."
    elif close > vah:
        zone_desc = f"Vùng **{fmt(poc)} - {fmt(vah)}** (Trên VAH): Trở thành **HỖ TRỢ NỀN VỮNG CHẮC**."
    elif val <= close <= vah:
        zone_desc = f"Vùng **{fmt(val)} - {fmt(vah)}** (Trong Value Area): Tích lũy chuẩn hóa."
    else:
        zone_desc = f"Vùng **{fmt(val)} - {fmt(poc)}**: Vùng nhạy cảm sát biên dưới."

    if confidence_score >= 65 and not is_below_val and close > cloud_bottom and not is_overbought_full and not is_bull_trap:
        system_status = "MỞ MUA / GIA TĂNG TỶ TRỌNG"
        nav_rec = "25 - 35% NAV"
        final_icon = f"🟢 **NÊN MUA / GIA TĂNG TỶ TRỌNG (Độ tin cậy: {confidence_score}%)**"
        final_action = f"Các chỉ báo đồng thuận ủng hộ đà tăng. Ưu tiên gom quanh hỗ trợ `{fmt(val)} - {fmt(poc)}`."
        
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
        final_action = f"Thị trường đối mặt rủi ro dòng tiền rút. Đứng ngoài bảo toàn vốn."
        stop_loss = round(close - (1.5 * atr), 2)
        action_note = f"Tuyệt đối không giải ngân mua mới. Stoploss tham chiếu: `{fmt(stop_loss)}`."

    conf_str = "; ".join(confluence_points) if confluence_points else "Chưa có tín hiệu hội tụ mạnh"
    warn_str = "; ".join(risk_warnings) if risk_warnings else "Không có cảnh báo rủi ro nghiêm trọng"

    if (is_below_val and cmf < 0) or cmf < -0.20 or close < cloud_bottom:
        sec5_action = "⚠️ **Kịch bản Rủi ro dòng tiền rút:** Cấu trúc xu hướng suy yếu. Tuyệt đối không bắt đáy sớm."
    elif is_bull_trap:
        sec5_action = "⚡ **Kịch bản Bẫy giá ngắn hạn (Bull Trap):** Nhịp hồi kỹ thuật rủi ro cao."
    elif is_overbought_full:
        sec5_action = "⚠️ **Kịch bản Quá mua hưng phấn:** Ưu tiên chốt lời hiện thực hóa lợi nhuận."
    elif confidence_score >= 65 and not is_below_val:
        sec5_action = "🚀 **Kịch bản Thuận lợi:** Tiếp diễn xu hướng tăng."
    else:
        sec5_action = "🛡️ **Kịch bản Tích lũy/Giằng co:** Tín hiệu đan xen, tuân thủ kỷ luật."

    text_part1 = f"📊 BÁO CÁO PHÂN TÍCH KỸ THUẬT: {symbol.upper()}\n"
    text_part1 += f"🗓️ Ngày: {last['time'].strftime('%Y-%m-%d')} | Khung: {resolution} | Giá hiện tại: {fmt(close)}\n\n"

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

    text_part2 = f"⚡ 5. NHẬN DIỆN BẪY GIÁ (MARKET TRAPS):\n"
    text_part2 += f"• Đánh giá: {trap_desc}\n"
    text_part2 += f"👉 **Lời khuyên chuyên gia:** Luôn kiểm chứng khối lượng và lực cầu cuối phiên để tránh sập bẫy tạo lập.\n\n"

    text_part2 += f"⚡ 6. TỔNG HỢP ĐIỂM CỘNG & CẢNH BÁO RỦI RO:\n"
    text_part2 += f"• Điểm cộng: {conf_str}\n"
    text_part2 += f"• Cảnh báo: {warn_str}\n"
    text_part2 += f"👉 **Kết luận:** {sec5_action}\n\n"

    text_part2 += f"📈 7. DÃI BOLINGER BANDS:\n"
    text_part2 += f"• Trên: {fmt(bb_upper)} | Dưới: {fmt(bb_lower)} | Độ rộng: {bb_width:.3f}\n\n"

    text_part2 += f"🏁 8. TỔNG KẾT CHIẾN LƯỢC:\n"
    text_part2 += f"• Khuyến nghị: {final_icon}\n"
    text_part2 += f"• Định hướng: {final_action}"

    keyboard = [
        [
            InlineKeyboardButton("📅 1D", callback_data=f"{symbol}|D|300"),
            InlineKeyboardButton("⏱️ 15m", callback_data=f"{symbol}|15|450"),
            InlineKeyboardButton("⏱️ 30m", callback_data=f"{symbol}|30|450"),
            InlineKeyboardButton("⏱️ 1H", callback_data=f"{symbol}|60|450"),
            InlineKeyboardButton("📈 1W", callback_data=f"{symbol}|W|250"),
        ],
        [
            InlineKeyboardButton("🤖 Quét Điểm Mua (/autobuy)", callback_data="run_autobuy_callback"),
            InlineKeyboardButton("📂 Cài Đặt / Portfolio", callback_data="show_all_settings"),
        ],
        [
            InlineKeyboardButton("💡 Giải thích CMF/MFI", callback_data="help_cmf"),
            InlineKeyboardButton("💡 Giải thích Ichimoku", callback_data="help_rsi"),
        ],
        [
            InlineKeyboardButton("💡 Giải thích Volume Profile", callback_data="help_vp"),
            InlineKeyboardButton("💡 Hướng dẫn Lệnh Cá Nhân", callback_data="help_portfolio"),
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
# 3. MODULE SCAN, SMART-BUY & BACKTEST
# ==========================================
def run_market_screener():
    results = []
    for symbol in WATCHLIST_SCAN:
        df, err = fetch_stock_data(symbol, resolution="D", count=300)
        if err or df.empty or len(df) < 50:
            continue
        df = calculate_indicators(df)
        last = df.iloc[-1]
        
        rvol = last["RVOL"] if not np.isnan(last["RVOL"]) else 1.0
        cmf = last["CMF"] if not np.isnan(last["CMF"]) else 0.0
        adx = last["ADX"] if not np.isnan(last["ADX"]) else 20.0
        close = last["Close"]
        high = last["High"]
        low = last["Low"]
        volume = last["Volume"]
        ema50 = last["EMA50"] if not np.isnan(last["EMA50"]) else close
        vol_sma20 = last["Vol_SMA20"] if not np.isnan(last["Vol_SMA20"]) else volume
        
        avg_trading_value = vol_sma20 * close * 1000
        if avg_trading_value < 5_000_000_000:
            continue
            
        price_range = high - low
        is_solid_candle = True if price_range == 0 else (close >= low + price_range * 0.4)

        if rvol >= 1.25 and cmf > 0.03 and close > ema50 and adx >= 20 and is_solid_candle:
            score = (rvol * 2.5) + (cmf * 6) + (adx * 0.15)
            results.append({
                "symbol": symbol,
                "close": close,
                "rvol": rvol,
                "cmf": cmf,
                "adx": adx,
                "score": score
            })
            
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:8]


def run_smart_buy_screener():
    buy_recommendations = []
    for symbol in WATCHLIST_SCAN:
        df, err = fetch_stock_data(symbol, resolution="D", count=300)
        if err or df.empty or len(df) < 50:
            continue
        df = calculate_indicators(df)
        last = df.iloc[-1]
        
        close = last["Close"]
        cmf = last["CMF"] if not np.isnan(last["CMF"]) else 0.0
        mfi = last["MFI"] if not np.isnan(last["MFI"]) else 50.0
        rvol = last["RVOL"] if not np.isnan(last["RVOL"]) else 1.0
        adx = last["ADX"] if not np.isnan(last["ADX"]) else 20.0
        macd_hist = last["MACD_Hist"] if not np.isnan(last["MACD_Hist"]) else 0.0
        rsi = last["RSI"] if not np.isnan(last["RSI"]) else 50.0
        stoch_rsi = last["Stoch_RSI"] if not np.isnan(last["Stoch_RSI"]) else 50.0
        
        span_a = last["Senkou_Span_A"] if not np.isnan(last["Senkou_Span_A"]) else close
        span_b = last["Senkou_Span_B"] if not np.isnan(last["Senkou_Span_B"]) else close
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        vol_sma20 = last["Vol_SMA20"] if not np.isnan(last["Vol_SMA20"]) else last["Volume"]
        avg_trading_value = vol_sma20 * close * 1000
        if avg_trading_value < 5_000_000_000: 
            continue

        # Chấm điểm kỹ thuật chuẩn xác theo hệ thống chiến lược
        score = 50
        if close > cloud_top:
            score += 20
        elif close >= cloud_bottom:
            score += 10
        else:
            score -= 15

        if cmf > 0.03:
            score += 15
        if rvol > 1.1:
            score += 10
        if macd_hist > 0:
            score += 15
        if adx >= 20:
            score += 10

        # Kiểm tra bẫy giá giả
        is_bull_trap = (rsi < 40 and stoch_rsi > 80)
        if is_bull_trap:
            score -= 30

        # Lọc các mã đạt chuẩn giải ngân an toàn (Score >= 65)
        if score >= 65 and cmf > -0.15 and close >= cloud_bottom and not is_bull_trap:
            atr = last["ATR"] if not np.isnan(last["ATR"]) else (close * 0.03)
            poc, vah, val = calculate_volume_profile(df)
            
            entry_price = round(max(val, close * 0.98), 2)
            stop_loss = round(entry_price - (1.5 * atr), 2)
            
            buy_recommendations.append({
                "symbol": symbol,
                "close": close,
                "score": score,
                "cmf": cmf,
                "rvol": rvol,
                "entry": entry_price,
                "stop_loss": stop_loss
            })

    buy_recommendations = sorted(buy_recommendations, key=lambda x: x["score"], reverse=True)
    return buy_recommendations[:5]


def background_scheduler_thread(application):
    sent_daily_report_today = None

    while True:
        now = datetime.datetime.now()
        today_date = now.date()

        if USER_PORTFOLIOS:
            for chat_id, portfolio in list(USER_PORTFOLIOS.items()):
                for symbol, pos in list(portfolio.items()):
                    df, err = fetch_stock_data(symbol, resolution="1", count=10)
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
        
        if not np.isnan(rvol) and not np.isnan(cmf) and not np.isnan(adx) and rvol > 1.3 and cmf > 0.03 and close > ema50 and adx >= 20:
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
    report = f"📊 KẾT QUẢ BACKTEST CHIẾN LƯỢC (ATR & ADX): {symbol}\n"
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
        "👋 **CHÀO MỪNG BẠN ĐẾN VỚI QUANT TRADING SYSTEM!**\n\n"
        "1️⃣ **Phân tích chi tiết mã cổ phiếu:** Nhập trực tiếp mã bất kỳ (Ví dụ: `VHM`, `BSR`)\n\n"
        "2️⃣ **🤖 Lọc mã có điểm mua chuẩn xác (`/autobuy`):** Tự động quét và chọn lọc mã đủ điều kiện giải ngân an toàn.\n\n"
        "3️⃣ **Cài đặt canh MUA (Real-Time):**\n"
        "`/setbuy BSR 29.4 28.5 1000` *(Mã, Giá vốn, Giá canh mua, Khối lượng)*\n\n"
        "4️⃣ **Cài đặt canh BÁN STOP LOSS (Cắt lỗ):**\n"
        "`/setstop BSR 27.9` *(Mã, Giá cắt lỗ cần cảnh báo)*\n\n"
        "5️⃣ **Xem danh mục đầu tư & Các mốc giá đang canh:**\n"
        "`/portfolio` hoặc `/listall`\n\n"
        "6️⃣ **Quét dòng tiền & Backtest:**\n"
        "`/scan` | `/backtest VHM`\n\n"
        "7️⃣ **Gỡ bỏ mã khỏi danh mục theo dõi:**\n"
        "`/reset BSR`  *(Hoặc `/reset ALL` để xóa sạch danh mục)*\n\n"
        "📌 **Tính năng mới:** Đã bổ sung nút bấm và lệnh `/autobuy` giúp chọn lọc điểm mua giải ngân tối ưu!"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def autobuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    
    # Kiểm tra xem gọi từ tin nhắn lệnh hay từ callback query
    if update.message:
        msg = await update.message.reply_text("🤖 **Đang phân tích đa khung & lọc các mã đạt điểm mua chuẩn xác...** ⏳")
    else:
        query = update.callback_query
        msg = await query.message.reply_text("🤖 **Đang phân tích đa khung & lọc các mã đạt điểm mua chuẩn xác...** ⏳")

    try:
        picks = run_smart_buy_screener()
        if not picks:
            await msg.edit_text("⚠️ Hiện tại thị trường chưa có mã nào đạt đủ tiêu chuẩn khắt khe để giải ngân an toàn.\n\n💡 *Khuyên nghị: Đứng ngoài bảo toàn vốn (NAV 0%)!*")
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
            
        text += "\n💡 *Bạn có thể bấm trực tiếp lệnh `/setbuy` được gợi ý bên trên để bot tự động giám sát real-time!*"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi thực thi bộ lọc auto-buy: {str(e)}")


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
        df, err = fetch_stock_data(symbol, resolution="1", count=10)
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

    text += "\n💡 Dùng lệnh `/listall` để xem tổng quan nhanh hoặc `/reset <MÃ>` để xóa."
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
    msg = await update.message.reply_text("🔍 Đang quét dòng tiền toàn thị trường... ⏳")
    try:
        top_stocks = run_market_screener()
        if not top_stocks:
            await msg.edit_text("⚠️ Hiện tại không có mã nào thỏa mãn tiêu chí.")
            return
            
        text = "🚀 TOP CỔ PHIẾU DÒNG TIỀN TỔ CHỨC THU GOM\n\n"
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
            InlineKeyboardButton("📅 1D", callback_data=f"{text}|D|300"),
            InlineKeyboardButton("⏱️ 15m", callback_data=f"{text}|15|450"),
            InlineKeyboardButton("⏱️ 30m", callback_data=f"{text}|30|450"),
            InlineKeyboardButton("⏱️ 1H", callback_data=f"{text}|60|450"),
            InlineKeyboardButton("📈 1W", callback_data=f"{text}|W|250"),
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

    if data == "show_all_settings":
        chat_id = query.message.chat_id
        await show_all_settings_func(query, context, chat_id)
        return

    if data.startswith("help_"):
        explanations = {
            "help_cmf": "💡 **Giải thích CMF & MFI:**\n• **CMF:** Dòng tiền tích lũy/phân phối lớn.\n• **MFI:** Chỉ số dòng tiền kết hợp khối lượng và giá.",
            "help_rsi": "💡 **Giải thích Ichimoku & ADX:**\n• Xác định xu hướng, độ mạnh của dòng tiền.",
            "help_vp": "💡 **Giải thích Volume Profile:**\n• **POC, VAL, VAH:** Các vùng hỗ trợ & kháng cự tổ chức.",
            "help_portfolio": "💡 **Hướng dẫn Danh mục:**\n• Dùng lệnh `/setbuy` để canh mua và `/setstop` để cài đặt stop loss riêng biệt."
        }
        await query.message.reply_text(explanations.get(data, "Đang cập nhật..."), parse_mode="Markdown")
        return

    parts = data.split("|")
    if len(parts) == 3:
        symbol, resolution, count = parts[0], parts[1], int(parts[2])
        await query.edit_message_text(text=f"⏳ Đang phân tích {symbol} (Khung: {resolution})...")

        def background_task():
            return analyze_stock(symbol, resolution=resolution, count=count)

        future = executor.submit(background_task)
        try:
            text_part1, text_part2, img_buf, reply_markup, err = future.result(timeout=15)
            if err:
                await query.message.reply_text(f"❌ Lỗi: {err}")
            else:
                await query.message.reply_photo(photo=img_buf)
                await query.message.reply_text(text_part1, parse_mode="Markdown")
                await query.message.reply_text(text_part2, reply_markup=reply_markup, parse_mode="Markdown")
        except Exception as e:
            await query.message.reply_text(f"❌ Lỗi hệ thống: {str(e)}")


# ==========================================
# 5. HTTP SERVER & SELF-PING
# ==========================================
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Quant Bot with Smart Auto-Buy Screener is running!")

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

    print("🤖 Đang khởi tạo Telegram Bot...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_error_handler(global_error_handler)

    threading.Thread(target=background_scheduler_thread, args=(app,), daemon=True).start()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("autobuy", autobuy_command))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("backtest", backtest_command))
    app.add_handler(CommandHandler("setbuy", setbuy_command))
    app.add_handler(CommandHandler("setstop", setstop_command))
    app.add_handler(CommandHandler("portfolio", portfolio_command))
    app.add_handler(CommandHandler("listall", listall_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    print("🚀 Bot Quant Trading System đã hoàn tất tích hợp bộ lọc thông minh /autobuy!")
    app.run_polling(drop_pending_updates=True)
