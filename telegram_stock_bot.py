import datetime
import io
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
    "MWG", "PNJ", "FRT", "DGW", "MSN", "VNM", "SAB", "KDH", "NLG", "DIG"
]

ACTIVE_CHAT_IDS = set()
USER_PORTFOLIOS = {}
ALERTED_TODAY = set()

executor = ThreadPoolExecutor(max_workers=8)


# ==========================================
# 1. HÀM CALL DATA & TÍNH TOÁN TOÀN DIỆN (TỰ ĐỘNG QUY ĐỔI COUNT & RESAMPLE KHUNG W)
# ==========================================
def fetch_stock_data(symbol, resolution="D", count=250):
    symbol = symbol.upper().strip()
    end_time = int(datetime.datetime.now().timestamp())

    # TỰ ĐỘNG QUY ĐỔI SỐ LƯỢNG NẾN (COUNT) THEO KHUNG THỜI GIAN
    if resolution in ["1", "5", "15", "30", "60"]:
        res_param = resolution
        count = max(count, 450)  # Phủ 450+ nến cho khung phút
        start_time = end_time - (count * int(resolution) * 60 * 4)
    elif resolution == "W":
        # Khung tuần: Gọi khung Ngày (D) với 1000 phiên rồi resample thành tuần để tránh lỗi API
        res_param = "D"
        count = 1000 
        start_time = end_time - (count * 86400 * 3)
    else:
        res_param = "D"
        count = max(count, 300)  # Đảm bảo tối thiểu 300 nến cho khung D (SMA200 & Volume Profile)
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

    # NẾU LÀ KHUNG TUẦN (W): TỰ ĐỘNG GOM NHÓM (RESAMPLE) TỪ DỮ LIỆU NẾN NGÀY (D)
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

    # Tích hợp ATR (14 phiên)
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    # Tích hợp ADX (14 phiên)
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
# 2. HÀM PHÂN TÍCH CHUYÊN SÂU ĐƠN MÃ (GIỮ NGUYÊN MẪU TRÌNH BÀY)
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

    macd = last["MACD"] if not np.isnan(last["MACD"]) else 0.0
    macd_signal = last["MACD_Signal"] if not np.isnan(last["MACD_Signal"]) else 0.0
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
    is_overbought_full = (stoch_rsi > 90 and rsi > 70)

    if is_bull_trap:
        score -= 25
        risk_warnings.append(f"⚡ Phát hiện Bẫy giá (Bull Trap)")
    elif is_bear_trap:
        score += 10
        confluence_points.append(f"⚡ Phát hiện Rũ bỏ (Bear Trap)")
    elif is_overbought_full:
        score -= 15
        risk_warnings.append(f"⚠️ Cảnh báo Quá mua toàn diện")
    else:
        score += 10
        confluence_points.append("Động lượng hài hòa, không quá cực đoan")

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

    text_part2 = f"⚡ 5. TỔNG HỢP ĐIỂM CỘNG & CẢNH BÁO RỦI RO:\n"
    text_part2 += f"• Điểm cộng: {conf_str}\n"
    text_part2 += f"• Cảnh báo: {warn_str}\n"
    text_part2 += f"👉 **Kết luận:** {sec5_action}\n\n"

    text_part2 += f"📈 6. DÃI BOLINGER BANDS:\n"
    text_part2 += f"• Trên: {fmt(bb_upper)} | Dưới: {fmt(bb_lower)} | Độ rộng: {bb_width:.3f}\n\n"

    text_part2 += f"🏁 7. TỔNG KẾT CHIẾN LƯỢC:\n"
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
# 3. MODULE SCAN & BACKTEST & REALTIME ALERTS (GIÁM SÁT REALTIME ĐA MÃ)
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
        ema50 = last["EMA50"] if not np.isnan(last["EMA50"]) else close
        
        if rvol >= 1.25 and cmf > 0.02 and close > ema50 and adx >= 20:
            score = (rvol * 2) + (cmf * 5) + (adx * 0.1)
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


def background_scheduler_thread(application):
    while True:
        now = datetime.datetime.now()
        
        # 1. KIỂM TRA & BÁO ĐỘNG REAL-TIME CHÍNH XÁC MỐC GIÁ NGƯỜI DÙNG ĐẶT (QUÉT KHUNG NẾN 1 PHÚT)
        if USER_PORTFOLIOS:
            for chat_id, portfolio in list(USER_PORTFOLIOS.items()):
                for symbol, pos in list(portfolio.items()):
                    df, err = fetch_stock_data(symbol, resolution="1", count=10)
                    if not err and not df.empty:
                        current_price = df.iloc[-1]["Close"]
                        entry_price = pos["entry"]
                        alert_price = pos.get("alert_price", entry_price)

                        alert_key_trigger = f"{chat_id}_{symbol}_trigger_{alert_price}_{datetime.date.today()}"

                        # Báo ngay lập tức nếu giá hiện tại tụt xuống hoặc chạm mốc giá đặt ra
                        if current_price <= alert_price and alert_key_trigger not in ALERTED_TODAY:
                            ALERTED_TODAY.add(alert_key_trigger)
                            alert_text = (
                                f"🚨 **CẢNH BÁO REAL-TIME: GIÁ CHẠM MỐC ĐẶT RA!** 🚨\n\n"
                                f"• Mã cổ phiếu: **{symbol.upper()}**\n"
                                f"• Mốc giá bạn cài đặt: `{alert_price}k`\n"
                                f"• **Giá hiện tại trên Chart**: `{current_price:.2f}k` 📉\n\n"
                                f"👉 Giá đã biến động đúng mốc bạn yêu cầu, kiểm tra bảng giá ngay!"
                            )
                            try:
                                application.bot.send_message(chat_id=chat_id, text=alert_text, parse_mode="Markdown")
                            except Exception:
                                pass

        # 2. BẢN TIN TỰ ĐỘNG LÚC 14:15 HÀNG NGÀY
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
        
        time.sleep(30)  # Quét liên tục mỗi 30 giây


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
        "1️⃣ **Phân tích chi tiết mã cổ phiếu:** Nhập trực tiếp mã bất kỳ (Ví dụ: `VHM`, `PLX`)\n\n"
        "2️⃣ **Cài đặt cảnh báo giá Real-Time:**\n"
        "`/setbuy PLX 35.9 35.8 1500`  *(Mã, Giá mua, Giá cảnh báo, Khối lượng)*\n"
        "*(Hoặc `/setbuy PLX 35.9 1500` nếu muốn lấy chính giá mua làm mốc cảnh báo)*\n\n"
        "3️⃣ **Xem danh mục đầu tư & Mốc giá đang canh:**\n"
        "`/portfolio`\n\n"
        "4️⃣ **Quét dòng tiền & Backtest:**\n"
        "`/scan` | `/backtest VHM`\n\n"
        "5️⃣ **Gỡ bỏ mã khỏi danh mục theo dõi:**\n"
        "`/reset PLX`  *(Hoặc `/reset ALL` để xóa sạch danh mục)*"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def setbuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "⚠️ Sai cú pháp!\n\n"
            "Vui lòng nhập một trong hai cách:\n"
            "• `/setbuy <MÃ> <giá_mua> <giá_cảnh_báo> [khối_lượng]`\n"
            "• `/setbuy <MÃ> <giá_mua> [khối_lượng]`\n\n"
            "Ví dụ: `/setbuy PLX 35.9 35.8 1500`",
            parse_mode="Markdown"
        )
        return

    symbol = args[0].upper().strip()
    try:
        entry_price = float(args[1])
        
        if len(args) >= 4:
            alert_price = float(args[2])
            qty = int(args[3])
        elif len(args) == 3:
            val_arg3 = float(args[2])
            if val_arg3 < 100:
                alert_price = val_arg3
                qty = 1000
            else:
                alert_price = entry_price
                qty = int(args[2])
        else:
            alert_price = entry_price
            qty = 1000
            
    except ValueError:
        await update.message.reply_text("❌ Các tham số giá và khối lượng phải là số hợp lệ!")
        return

    if chat_id not in USER_PORTFOLIOS:
        USER_PORTFOLIOS[chat_id] = {}

    USER_PORTFOLIOS[chat_id][symbol] = {
        "entry": entry_price,
        "alert_price": alert_price,
        "qty": qty
    }

    await update.message.reply_text(
        f"✅ Đã thiết lập giám sát Real-Time thành công:\n"
        f"• Mã cổ phiếu: **{symbol}**\n"
        f"• Giá mua: `{entry_price}k` | Khối lượng: {qty} cổ\n"
        f"• **Mốc giá cảnh báo đặt ra**: `{alert_price}k`\n\n"
        f"🤖 Bot đang chạy ngầm kiểm tra giá thị trường...",
        parse_mode="Markdown"
    )

    # CHECK NGAY LẬP TỨC KHUNG 1 PHÚT KHI VỪA SET
    df, err = fetch_stock_data(symbol, resolution="1", count=10)
    if not err and not df.empty:
        current_price = df.iloc[-1]["Close"]
        if current_price <= alert_price:
            alert_text = (
                f"🚨 **CẢNH BÁO REAL-TIME: GIÁ HIỆN TẠI ĐÃ DƯỚI MỐC CÀI ĐẶT!** 🚨\n\n"
                f"• Mã cổ phiếu: **{symbol.upper()}**\n"
                f"• Mốc giá bạn đặt: `{alert_price}k`\n"
                f"• **Giá hiện tại thị trường**: `{current_price:.2f}k` 📉\n\n"
                f"👉 Giá hiện tại đã thấp hơn mốc cài đặt, hãy kiểm tra lại vị thế!"
            )
            await update.message.reply_text(alert_text, parse_mode="Markdown")


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in USER_PORTFOLIOS or not USER_PORTFOLIOS[chat_id]:
        await update.message.reply_text(
            "📂 Danh mục của bạn đang trống.\n"
            "👉 Dùng lệnh `/setbuy PLX 35.9 35.8 1500` để thêm mã và mốc giá cần canh.",
            parse_mode="Markdown"
        )
        return

    text = "📂 **DANH MỤC ĐẦU TƯ & GIÁM SÁT REAL-TIME**\n\n"
    for symbol, pos in USER_PORTFOLIOS[chat_id].items():
        df, err = fetch_stock_data(symbol, resolution="1", count=10)
        if not err and not df.empty:
            current_price = df.iloc[-1]["Close"]
            qty = pos.get("qty", 1000)
            entry = pos["entry"]
            alert_price = pos.get("alert_price", entry)
            
            pnl_pct = ((current_price - entry) / entry) * 100
            pnl_vnd = (current_price - entry) * qty
            status = "🟢 LÃI" if pnl_pct >= 0 else "🔴 LỖ"

            text += f"🔹 **Mã: {symbol.upper()}** ({qty} cổ)\n"
            text += f"  • Giá vốn: `{entry}k` | Giá hiện tại: `{current_price:.2f}k`\n"
            text += f"  • Hiệu suất: `{pnl_pct:+.2f}%` (`{pnl_vnd:+,.0f} VNĐ`) [{status}]\n"
            text += f"  • 🎯 **Mốc giá canh báo**: `{alert_price}k`\n"
            text += f"  • Lệnh gỡ bỏ: `/reset {symbol}`\n"
            text += "------------------------------------\n"
        else:
            text += f"🔹 **{symbol.upper()}**: Không lấy được dữ liệu.\n"

    text += "\n💡 Dùng `/reset <MÃ>` để xóa mã cụ thể, hoặc `/reset ALL` để xóa sạch."
    await update.message.reply_text(text, parse_mode="Markdown")


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
            await update.message.reply_text("🗑️ Đã xóa toàn bộ danh mục theo dõi của bạn.")
        elif target in USER_PORTFOLIOS[chat_id]:
            del USER_PORTFOLIOS[chat_id][target]
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
        await update.message.reply_text("⚠️ Vui lòng nhập mã hợp lệ (VD: VHM, PLX, HPG)!")
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

    if data.startswith("help_"):
        explanations = {
            "help_cmf": "💡 **Giải thích CMF & MFI:**\n• **CMF:** Dòng tiền tích lũy/phân phối lớn.\n• **MFI:** Chỉ số dòng tiền kết hợp khối lượng và giá.",
            "help_rsi": "💡 **Giải thích Ichimoku & ADX:**\n• Xác định xu hướng, độ mạnh của dòng tiền.",
            "help_vp": "💡 **Giải thích Volume Profile:**\n• **POC, VAL, VAH:** Các vùng hỗ trợ & kháng cự tổ chức.",
            "help_portfolio": "💡 **Hướng dẫn Danh mục:**\n• Dùng lệnh `/setbuy PLX 35.9 35.8 1500` để theo dõi chuẩn xác mốc giá cảnh báo real-time."
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
        self.wfile.write(b"Quant Bot with Custom Price Alert is running!")

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
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("backtest", backtest_command))
    app.add_handler(CommandHandler("setbuy", setbuy_command))
    app.add_handler(CommandHandler("portfolio", portfolio_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    print("🚀 Bot Quant Trading System đã sẵn sàng hoàn chỉnh!")
    app.run_polling(drop_pending_updates=True)
