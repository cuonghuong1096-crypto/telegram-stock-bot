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
# 1. HÀM CALL DATA & TÍNH TOÁN TOÀN DIỆN
# ==========================================
def fetch_stock_data(symbol, resolution="D", count=250):
    symbol = symbol.upper().strip()
    end_time = int(datetime.datetime.now().timestamp())

    if resolution in ["1", "5", "15", "30", "60"]:
        res_param = resolution
        start_time = end_time - (count * int(resolution) * 60 * 4)
    elif resolution == "W":
        res_param = "W"
        start_time = end_time - (count * 7 * 86400 * 5)
    else:
        res_param = "D"
        start_time = end_time - (count * 86400 * 3)

    tcbs_url = f"https://apifull.tcbs.com.vn/stock-insight/v1/stock/bars?resolution={res_param}&ticker={symbol}&from={start_time}&to={end_time}"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "X-API-KEY": TCBS_API_KEY,
        }
        res = requests.get(tcbs_url, headers=headers, timeout=3)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df["time"] = pd.to_datetime(df["tradingDate"])
                df.rename(
                    columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    },
                    inplace=True,
                )
                return df[
                    ["time", "Open", "High", "Low", "Close", "Volume"]
                ].tail(count).reset_index(drop=True), None
    except Exception:
        pass

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
                df = pd.DataFrame(
                    {
                        "time": pd.to_datetime(data["t"], unit="s"),
                        "Open": data["o"],
                        "High": data["h"],
                        "Low": data["l"],
                        "Close": data["c"],
                        "Volume": data["v"],
                    }
                )
                return df.tail(count).reset_index(drop=True), None
    except Exception as e:
        return pd.DataFrame(), f"Lỗi kết nối API dự phòng: {str(e)}"

    return pd.DataFrame(), f"Không tìm thấy dữ liệu hoặc mã cổ phiếu '{symbol}' không hợp lệ."


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
    
    # Bổ sung các đường MA đơn giản (SMA) phục vụ xác định xu hướng dài hạn theo yêu cầu
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
# 2. HÀM PHÂN TÍCH CHUYÊN SÂU ĐƠN MÃ (TÍCH HỢP LOGIC NÂNG CAO)
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

    # Xác định xu hướng dài hạn qua MA/SMA
    if close > sma200:
        long_term_trend = "UPTREND DÀI HẠN (Trên SMA200)"
        vn_index_status = "TÍCH CỰC"
    else:
        long_term_trend = "DOWNTREND/PHỤC HỒI DÀI HẠN (Dưới SMA200)"
        vn_index_status = "THẬN TRỌNG"

    if close > ema50 and ema50 > ema200:
        market_structure = "UPTREND"
    elif close < ema50 and ema50 < ema200:
        market_structure = "DOWNTREND"
    else:
        market_structure = "SIDEWAY ACCUMULATION"

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

    if cmf > 0.05 and mfi > 50:
        score += 20
        confluence_points.append(f"Dòng tiền lớn CMF ({cmf:.2f}) & MFI ({mfi:.1f}) đồng thuận")
    elif cmf > 0 or mfi > 45:
        score += 10
    else:
        risk_warnings.append(f"Dòng tiền yếu/rút lui (CMF: {cmf:.2f}, MFI: {mfi:.1f})")

    # Kiểm tra đồng nhất động lượng và tín hiệu StochRSI / RSI
    is_rsi_neutral = (45 <= rsi <= 60)
    is_stoch_overbought = (stoch_rsi > 80)
    
    if is_rsi_neutral and is_stoch_overbought:
        score -= 10
        risk_warnings.append(f"Mâu thuẫn động lượng: RSI trung tính ({rsi:.1f}) nhưng StochRSI quá mua ({stoch_rsi:.1f}) ➔ Ngắn hạn dễ rung lắc")
    elif stoch_rsi > 80 and rsi > 65:
        score -= 15
        risk_warnings.append(f"Cảnh báo quá mua toàn diện (RSI: {rsi:.1f}, StochRSI: {stoch_rsi:.1f})")
    else:
        score += 10
        confluence_points.append("Động lượng RSI và StochRSI hài hòa, không quá cực đoan")

    if close > sma200:
        score += 15
        confluence_points.append("Giá duy trì trên đường dài hạn SMA200")
    else:
        risk_warnings.append("Giá nằm dưới SMA200 (Xu hướng dài hạn thận trọng)")

    if macd_hist > 0:
        score += 15
        confluence_points.append(f"MACD Histogram dương ({macd_hist:+.2f}) ủng hộ động lượng tăng")
    else:
        risk_warnings.append(f"MACD Histogram âm ({macd_hist:+.2f}) thể hiện động lượng suy yếu")

    confidence_score = min(max(score, 10), 88)

    # Xử lý vùng giá Volume Profile: Nếu giá nằm dưới dải giá trị thấp (VAL), vùng này là KHÁNG CỰ MẠNH
    if close < val:
        zone_desc = f"Vùng **{fmt(val)} - {fmt(vah)}** (Giá hiện tại nằm dưới VAL): Trở thành **KHÁNG CỰ MẠNH** (vùng kẹp hàng lớn phía trên)."
        is_below_val = True
    elif close > vah:
        zone_desc = f"Vùng **{fmt(poc)} - {fmt(vah)}** (Giá hiện tại nằm trên VAH): Trở thành **HỖ TRỢ NỀN VỮNG CHẮC**."
        is_below_val = False
    else:
        zone_desc = f"Vùng **{fmt(val)} - {fmt(poc)}**: Lực cầu đỡ giá dao động trong biên độ Volume Profile."
        is_below_val = False

    entry_1 = round(min(kijun, val) if is_below_val else max(val, last["EMA20"], close * 0.98), 2)
    entry_2 = round(val if is_below_val else close, 2)
    entry_avg = round((entry_1 + entry_2) / 2, 2)
    stop_loss = round(entry_avg * 0.93, 2)

    if confidence_score >= 65 and not is_below_val and close > cloud_bottom:
        system_status = "MỞ MUA / GIA TĂNG TỶ TRỌNG"
        nav_rec = "25 - 35% NAV"
        is_standing_aside = False
        final_icon = f"🟢 **NÊN MUA / GIA TĂNG TỶ TRỌNG (Độ tin cậy: {confidence_score}%)**"
        final_action = f"Các chỉ báo MA, MACD và Volume Profile đồng thuận ủng hộ đà tăng. Ưu tiên gom quanh hỗ trợ `{fmt(val)} - {fmt(poc)}`."
    elif confidence_score >= 45:
        system_status = "QUAN SÁT / THẬN TRỌNG RUNG LẮC"
        nav_rec = "10 - 15% NAV (Thăm dò tỷ trọng nhỏ)"
        is_standing_aside = False
        final_icon = f"🟡 **GIAO DỊCH THĂM DÒ / QUAN SÁT (Độ tin cậy: {confidence_score}%)**"
        final_action = f"Tồn tại mâu thuẫn chỉ báo (StochRSI quá mua trong khi RSI trung tính) hoặc giá đang tiềm cận kháng cự Volume Profile. **Kiên nhẫn chờ chiết khấu an toàn** hoặc test đường Kijun-sen (`{fmt(kijun)}`)."
    else:
        system_status = "ĐỨNG NGOÀI (BẢO TOÀN VỐN)"
        nav_rec = "0% (Giữ tiền mặt)"
        is_standing_aside = True
        final_icon = f"🔴 **ĐỨNG NGOÀI QUAN SÁT (Độ tin cậy: {confidence_score}%)**"
        final_action = f"Giá nằm dưới vùng giá trị Volume Profile (đối mặt kháng cự dày đặc) hoặc cấu trúc dài hạn yếu. Đứng ngoài bảo toàn vốn."

    # --- PHẦN 1: MỤC 1 ĐẾN MỤC 4 ---
    text_part1 = f"📊 BÁO CÁO PHÂN TÍCH KỸ THUẬT: {symbol.upper()}\n"
    text_part1 += f"🗓️ Ngày: {last['time'].strftime('%Y-%m-%d')} | Giá hiện tại: {fmt(close)}\n\n"

    # MỤC 1
    text_part1 += f"🌐 1. CẤU TRÚC XU HƯỚNG & DÒNG TIỀN (MA & MACD):\n"
    text_part1 += f"• Xu hướng dài hạn (SMA200): {long_term_trend} (Giá: {fmt(close)} vs SMA200: {fmt(sma200)})\n"
    text_part1 += f"• Chỉ báo: CMF: {cmf:.2f} | MFI: {mfi:.1f} | RVOL: {rvol:.2f}x | MACD Hist: {macd_hist:+.2f}\n"
    text_part1 += f"• Độ dốc xu hướng (LR Slope): {lr_slope:+.2f}\n"
    text_part1 += f"👉 **Kết luận & Hành động:** "
    if close > sma200 and macd_hist > 0:
        text_part1 += f"Xu hướng dài hạn tích cực, MACD ủng hộ động lượng tăng trưởng ổn định.\n\n"
    else:
        text_part1 += f"Cấu trúc dài hạn/động lượng phân hóa, cần tuân thủ kỷ luật phòng thủ.\n\n"

    # MỤC 2
    text_part1 += f"☁️ 2. CHỈ BÁO ICHIMOKU & ĐỘNG LƯỢNG (RSI & StochRSI):\n"
    text_part1 += f"• Thông số mây: {ichi_status}\n"
    text_part1 += f"• Động lượng ngắn/trung hạn: RSI: {rsi:.1f} (Trung tính) | StochRSI: {stoch_rsi:.1f} (Quá mua)\n"
    text_part1 += f"• Đường cơ sở: Tenkan-sen: {fmt(tenkan)} | Kijun-sen: {fmt(kijun)}\n"
    text_part1 += f"👉 **Kết luận & Hành động:** "
    if is_rsi_neutral and is_stoch_overbought:
        text_part1 += f"Phát hiện mâu thuẫn động lượng ngắn hạn (StochRSI quá mua trong khi RSI trung tính). Áp lực rung lắc ngắn hạn hiện hữu, canh mua khi giá test đường Kijun-sen (`{fmt(kijun)}`).\n\n"
    else:
        text_part1 += f"Thế nến và động lượng vận động đồng pha, canh giải ngân kiểm định vùng hỗ trợ.\n\n"

    # MỤC 3
    text_part1 += f"📊 3. VOLUME PROFILE & ĐÁNH GIÁ KHÁNG CỰ / HỖ TRỢ:\n"
    text_part1 += f"• Thông số: POC (Khớp nhiều nhất): {fmt(poc)}\n"
    text_part1 += f"• Vùng giá trị: VAL: {fmt(val)} | VAH: {fmt(vah)}\n"
    text_part1 += f"👉 **Kết luận & Hành động:** {zone_desc}\n\n"

    # MỤC 4
    text_part1 += f"🎯 4. QUẢN TRỊ RỦI RO & PHÂN BỔ NAV:\n"
    text_part1 += f"• Trạng thái khuyến nghị: {system_status}\n"
    text_part1 += f"• Tỷ trọng phân bổ NAV: {nav_rec}\n"
    text_part1 += f"👉 **Kết luận & Hành động:** Giải ngân quanh mức `{fmt(entry_1)} - {fmt(entry_2)}`, cài đặt Stoploss bắt buộc tại mốc `{fmt(stop_loss)}`.\n\n"

    # --- PHẦN 2: MỤC 5, 6 & 7 ---
    text_part2 = ""
    text_part2 += f"⚡ 5. TỔNG HỢP ĐIỂM CỘNG & CẢNH BÁO RỦI RO:\n"
    conf_str = "; ".join(confluence_points) if confluence_points else "Chưa xuất hiện tín hiệu hội tụ đủ mạnh"
    warn_str = "; ".join(risk_warnings) if risk_warnings else "Không có cảnh báo rủi ro nghiêm trọng"
    text_part2 += f"• Điểm cộng kỹ thuật: {conf_str}\n"
    text_part2 += f"• Cảnh báo rủi ro: {warn_str}\n"
    text_part2 += f"👉 **Kết luận & Hành động:** Đánh giá kỹ tỷ lệ R:R, tuyệt đối không mua đuổi khi gặp vùng kháng cự Volume Profile.\n\n"

    text_part2 += f"📈 6. DÃI BOLINGER BANDS & BIẾN ĐỘNG GIÁ:\n"
    text_part2 += f"• Bollinger Bands: Cận trên: {fmt(bb_upper)} | Cận dưới: {fmt(bb_lower)} | Trạm giữa: {fmt(last['BB_Middle'])}\n"
    bien_bb = "Thắt cổ chai (Squeeze - Chuẩn bị bứt phá biên độ)" if bb_width < 0.05 else "Biên độ ổn định / Mở rộng"
    text_part2 += f"• Trạng thái biến động: {bien_bb} (Độ rộng: {bb_width:.3f})\n"
    text_part2 += f"👉 **Kết luận & Hành động:** Dựa vào biên độ Bollinger Bands để đo lường dư địa ngắn hạn trước khi mở vị thế.\n\n"

    text_part2 += f"🏁 7. TỔNG KẾT CHIẾN LƯỢC TỔNG HỢP:\n"
    text_part2 += f"• Khuyến nghị: {final_icon}\n"
    text_part2 += f"• Định hướng quyết định: {final_action}"

    keyboard = [
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
    ax.axhline(val, color="red", linestyle="--", label=f"VAL (Kháng cự): {fmt(val)}", alpha=0.6)
    ax.set_title(f"Phân Tích Chuyên Sâu - {symbol.upper()} ({resolution})", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return text_part1, text_part2, buf, reply_markup, None


# ==========================================
# 3. MODULE SCAN & BACKTEST & REALTIME ALERTS
# ==========================================
def run_market_screener():
    results = []
    for symbol in WATCHLIST_SCAN:
        df, err = fetch_stock_data(symbol, resolution="D", count=60)
        if err or df.empty or len(df) < 30:
            continue
        df = calculate_indicators(df)
        last = df.iloc[-1]
        
        rvol = last["RVOL"] if not np.isnan(last["RVOL"]) else 1.0
        cmf = last["CMF"] if not np.isnan(last["CMF"]) else 0.0
        close = last["Close"]
        ema50 = last["EMA50"] if not np.isnan(last["EMA50"]) else close
        
        if rvol >= 1.25 and cmf > 0.02 and close > ema50:
            score = (rvol * 2) + (cmf * 5)
            results.append({
                "symbol": symbol,
                "close": close,
                "rvol": rvol,
                "cmf": cmf,
                "score": score
            })
            
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:8]


def check_realtime_alerts(application):
    now = datetime.datetime.now()
    if now.weekday() >= 5:
        return
    if not (9 <= now.hour < 15 or (now.hour == 14 and now.minute <= 45)):
        return

    today_str = now.strftime("%Y-%m-%d")
    
    for symbol in WATCHLIST_SCAN:
        df, err = fetch_stock_data(symbol, resolution="D", count=50)
        if err or df.empty or len(df) < 30:
            continue
        
        df = calculate_indicators(df)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        close = last["Close"]
        rvol = last["RVOL"] if not np.isnan(last["RVOL"]) else 1.0
        high_20 = df["High"].iloc[-20:-1].max()
        rsi = last["RSI"] if not np.isnan(last["RSI"]) else 50.0
        prev_rsi = prev["RSI"] if not np.isnan(prev["RSI"]) else 50.0
        macd_hist = last["MACD_Hist"] if not np.isnan(last["MACD_Hist"]) else 0.0
        prev_macd_hist = prev["MACD_Hist"] if not np.isnan(prev["MACD_Hist"]) else 0.0

        alert_key_breakout = f"{symbol}_breakout_{today_str}"
        alert_key_div_bull = f"{symbol}_div_bull_{today_str}"

        if close > high_20 and rvol >= 2.0 and alert_key_breakout not in ALERTED_TODAY:
            ALERTED_TODAY.add(alert_key_breakout)
            msg = (
                f"🚨 **CẢNH BÁO BREAKOUT BÙNG NỔ!** 🚨\n\n"
                f"• Mã cổ phiếu: **{symbol.upper()}**\n"
                f"• Giá hiện tại: `{close:.1f}k` *(Đã vượt đỉnh 20 phiên)*\n"
                f"• Thanh khoản đột biến (RVOL): `{rvol:.2f}x` so với trung bình!\n\n"
                f"👉 Dòng tiền lớn đang đánh bứt phá, chú ý quan sát điểm mua gia tăng!"
            )
            for chat_id in ACTIVE_CHAT_IDS:
                try:
                    application.bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
                except Exception:
                    pass

        if close <= prev["Close"] * 1.005 and rsi > prev_rsi + 3 and macd_hist > prev_macd_hist and alert_key_div_bull not in ALERTED_TODAY:
            ALERTED_TODAY.add(alert_key_div_bull)
            msg = (
                f"⚡ **CẢNH BÁO PHÂN KỲ DƯƠNG (BÁO HIỆU ĐÁY)!** ⚡\n\n"
                f"• Mã cổ phiếu: **{symbol.upper()}**\n"
                f"• Giá hiện tại: `{close:.1f}k`\n"
                f"• Chỉ báo RSI phục hồi: `{prev_rsi:.1f} ➔ {rsi:.1f}`\n"
                f"• Động lượng MACD Histogram cải thiện tích cực.\n\n"
                f"👉 Tín hiệu kỹ thuật phát hiện dòng tiền bắt đáy nhen nhóm!"
            )
            for chat_id in ACTIVE_CHAT_IDS:
                try:
                    application.bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
                except Exception:
                    pass


def run_strategy_backtest(symbol):
    symbol = symbol.upper().strip()
    df, err = fetch_stock_data(symbol, resolution="D", count=200)
    if err or df.empty or len(df) < 100:
        return f"⚠️ Không đủ dữ liệu lịch sử để backtest mã {symbol}. Chi tiết: {err}"
    
    df = calculate_indicators(df)
    trades, wins, losses, total_return = 0, 0, 0, 0.0
    
    for i in range(30, len(df) - 10):
        rvol = df["RVOL"].iloc[i]
        cmf = df["CMF"].iloc[i]
        close = df["Close"].iloc[i]
        ema50 = df["EMA50"].iloc[i]
        
        if not np.isnan(rvol) and not np.isnan(cmf) and rvol > 1.3 and cmf > 0.03 and close > ema50:
            entry_price = close
            stoploss_price = entry_price * 0.935
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
        return f"📊 KẾT QUẢ BACKTEST: {symbol}\n\nTrong 200 phiên qua, mã {symbol} chưa xuất hiện điểm mua đạt tiêu chuẩn RVOL + CMF."

    win_rate = (wins / trades) * 100
    report = f"📊 KẾT QUẢ BACKTEST CHIẾN LƯỢC: {symbol}\n"
    report += f"🗓️ Dữ liệu: 200 phiên lịch sử gần nhất\n"
    report += f"------------------------------------\n"
    report += f"• Tổng số lệnh phát sinh: {trades} lệnh\n"
    report += f"• Lệnh thắng: {wins} | Lệnh thua: {losses}\n"
    report += f"• Tỷ lệ thắng (Win-rate): {win_rate:.1f}%\n"
    report += f"• Hiệu suất tích lũy giả lập: {total_return:+.2f}%\n"
    return report


def background_scheduler_thread(application):
    while True:
        now = datetime.datetime.now()
        
        try:
            check_realtime_alerts(application)
        except Exception:
            pass

        if USER_PORTFOLIOS:
            for chat_id, portfolio in list(USER_PORTFOLIOS.items()):
                for symbol, pos in list(portfolio.items()):
                    df, err = fetch_stock_data(symbol, resolution="D", count=10)
                    if not err and not df.empty:
                        current_price = df.iloc[-1]["Close"]
                        sl_price = pos["sl"]
                        entry_price = pos["entry"]
                        poc, vah, val = calculate_volume_profile(df)
                        target_take_profit = max(entry_price * 1.05, vah)

                        if current_price <= sl_price:
                            alert_text = (
                                f"🚨 **CẢNH BÁO KHẨN CẤP: THỦNG MỨC CẮT LỖ (STOPLOSS)!** 🚨\n\n"
                                f"• Mã cổ phiếu: **{symbol}**\n"
                                f"• Giá vốn: `{entry_price}k` | Ngưỡng SL: `{sl_price}k`\n"
                                f"• Giá hiện tại: `{current_price:.1f}k` 📉\n\n"
                                f"👉 BÁN CẮT LỖ NGAY LẬP TỨC để bảo toàn vốn!"
                            )
                            try:
                                application.bot.send_message(chat_id=chat_id, text=alert_text, parse_mode="Markdown")
                            except Exception:
                                pass
                        elif current_price >= target_take_profit:
                            alert_text = (
                                f"🎯 **CẢNH BÁO TÍN HIỆU: ĐẠT MỤC TIÊU CHỐT LỜI!** 🎯\n\n"
                                f"• Mã cổ phiếu: **{symbol}**\n"
                                f"• Giá vốn: `{entry_price}k` | Target: `≥ {target_take_profit:.1f}k`\n"
                                f"• Giá hiện tại: `{current_price:.1f}k` 🚀\n\n"
                                f"👉 Canh BÁN CHỐT LỜI hiện thực hóa thành quả!"
                            )
                            try:
                                application.bot.send_message(chat_id=chat_id, text=alert_text, parse_mode="Markdown")
                            except Exception:
                                pass

        if now.hour == 14 and now.minute == 15:
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
        
        time.sleep(60)


# ==========================================
# 4. TELEGRAM HANDLERS
# ==========================================
async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err_msg = f"❌ **HỆ THỐNG PHÁT HIỆN LỖI RUNTIME:**\n\n`{context.error}`"
    logging.error(f"Error update: {update} caused error: {context.error}")
    
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(err_msg, parse_mode="Markdown")
        except Exception:
            await update.effective_message.reply_text(f"❌ HỆ THỐNG PHÁT HIỆN LỖI:\n{str(context.error)}")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_id = update.effective_chat.id
        ACTIVE_CHAT_IDS.add(chat_id)
        welcome_text = (
            "👋 **CHÀO MỪNG BẠN ĐẾN VỚI QUANT TRADING SYSTEM!**\n\n"
            "📌 *Bấm trực tiếp vào các ô bên dưới để copy cú pháp:* \n\n"
            "1️⃣ **Phân tích chi tiết mã cổ phiếu:**\n"
            "`VHM`  *(hoặc nhập mã bất kỳ như HPG, PNJ, VRE)*\n\n"
            "2️⃣ **Lưu giá mua, cắt lỗ & khối lượng:**\n"
            "`/setbuy VHM 63.0 62.0 1500`  *(Mã, Giá mua, SL, Khối lượng)*\n\n"
            "3️⃣ **Xem danh mục đầu tư & Mức giá cụ thể Chốt lời / Cắt lỗ:**\n"
            "`/portfolio` \n\n"
            "4️⃣ **Quét dòng tiền bùng nổ toàn thị trường:**\n"
            "`/scan` \n\n"
            "5️⃣ **Backtest hiệu suất chiến lược 200 phiên:**\n"
            "`/backtest VHM` \n\n"
            "6️⃣ **Xóa mã khỏi danh mục theo dõi:**\n"
            "`/reset VHM`  *(hoặc `/reset ALL` để xóa sạch danh mục)*"
        )
        await update.message.reply_text(welcome_text, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi thực thi /start: {str(e)}")


async def setbuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_id = update.effective_chat.id
        args = context.args
        if len(args) < 3:
            await update.message.reply_text(
                "Sai cú pháp!\n\n"
                "Vui lòng nhập đúng định dạng:\n"
                "/setbuy <MÃ> <giá_mua> <giá_cắt_lỗ> [khối_lượng]\n"
                "Ví dụ: /setbuy VHM 63.0 62.0 1500"
            )
            return

        symbol = args[0].upper().strip()
        try:
            entry_price = float(args[1])
            sl_price = float(args[2])
            qty = int(args[3]) if len(args) >= 4 else 1000
        except ValueError:
            await update.message.reply_text("Giá mua, giá cắt lỗ và khối lượng phải là số hợp lệ!")
            return

        if chat_id not in USER_PORTFOLIOS:
            USER_PORTFOLIOS[chat_id] = {}

        USER_PORTFOLIOS[chat_id][symbol] = {
            "entry": entry_price,
            "sl": sl_price,
            "qty": qty
        }

        await update.message.reply_text(
            f"Đã lưu thành công vào danh mục:\n"
            f"- Mã: {symbol}\n"
            f"- Khối lượng: {qty} cổ phiếu\n"
            f"- Giá mua: {entry_price}\n"
            f"- Cắt lỗ (SL): {sl_price}\n\n"
            f"Bot sẽ tự động giám sát và cảnh báo ngay lập tức nếu giá thị trường chạm mốc chốt lời hoặc thủng mức SL!"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi thực thi /setbuy: {str(e)}")


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_id = update.effective_chat.id
        if chat_id not in USER_PORTFOLIOS or not USER_PORTFOLIOS[chat_id]:
            await update.message.reply_text(
                "📂 Danh mục của bạn đang trống.\n"
                "👉 Dùng lệnh `/setbuy VHM 63.0 62.0 1500` để thêm mã vào danh mục theo dõi.",
                parse_mode="Markdown"
            )
            return

        text = "📂 **DANH MỤC ĐẦU TƯ CÁ NHÂN & CHIẾN LƯỢC QUẢN TRỊ VỊ THẾ**\n\n"
        for symbol, pos in USER_PORTFOLIOS[chat_id].items():
            df, err = fetch_stock_data(symbol, resolution="D", count=20)
            if not err and not df.empty:
                current_price = df.iloc[-1]["Close"]
                qty = pos.get("qty", 1000)
                entry = pos["entry"]
                sl = pos["sl"]
                
                pnl_pct = ((current_price - entry) / entry) * 100
                pnl_vnd = (current_price - entry) * qty
                status = "🟢 LÃI" if pnl_pct >= 0 else "🔴 LỖ"
                
                poc, vah, val = calculate_volume_profile(df)
                target_1 = max(entry * 1.05, vah)
                target_2 = target_1 * 1.08

                text += f"🔹 **Mã Cổ Phiếu: {symbol}** ({qty} cổ)\n"
                text += f"  • Giá vốn: `{entry}k` | Giá hiện tại: `{current_price:.1f}k`\n"
                text += f"  • Hiệu suất: `{pnl_pct:+.2f}%` (`{pnl_vnd:+,.0f} VNĐ`) [{status}]\n\n"
                text += f"🎯 **CHIẾN LƯỢC CHỐT LỜI & CẮT LỖ CỤ THỂ:**\n"
                if current_price <= sl:
                    text += f"  🔴 **CẮT LỖ KHẨN CẤP:** Đã thủng mốc SL **`{sl}k`**. Hãy BÁN NGAY!\n"
                else:
                    text += f"  🛡️ **Mức Cắt Lỗ (SL):** `{sl}k`\n"
                if current_price >= target_1:
                    text += f"  ✅ **CHỐT LỜI NGAY:** Đã vượt mục tiêu 1 (`{target_1:.1f}k`). Chốt 50% tỷ trọng.\n"
                else:
                    text += f"  ⏳ **Mục tiêu Chốt Lời 1:** `{target_1:.1f}k` | **Mục tiêu 2:** `{target_2:.1f}k`\n"
                text += "------------------------------------\n"
            else:
                text += f"🔹 **{symbol}**: Không lấy được dữ liệu giá hiện tại.\n"

        text += "\n💡 Dùng lệnh `/reset <MÃ>` để xóa mã khỏi danh mục."
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi thực thi /portfolio: {str(e)}")


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_id = update.effective_chat.id
        args = context.args
        if not args:
            await update.message.reply_text("⚠️ Vui lòng nhập mã cần xóa! Ví dụ: `/reset VHM` hoặc `/reset ALL`.", parse_mode="Markdown")
            return

        target = args[0].upper().strip()
        if chat_id in USER_PORTFOLIOS:
            if target == "ALL":
                USER_PORTFOLIOS[chat_id] = {}
                await update.message.reply_text("🗑️ Đã xóa toàn bộ danh mục theo dõi của bạn.")
            elif target in USER_PORTFOLIOS[chat_id]:
                del USER_PORTFOLIOS[chat_id][target]
                await update.message.reply_text(f"🗑️ Đã xóa mã **{target}** khỏi danh mục theo dõi.", parse_mode="Markdown")
            else:
                await update.message.reply_text(f"⚠️ Không tìm thấy mã **{target}** trong danh mục.", parse_mode="Markdown")
        else:
            await update.message.reply_text("📂 Danh mục của bạn đang trống.")
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi thực thi /reset: {str(e)}")


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    msg = await update.message.reply_text("🔍 Hệ thống đang quét dòng tiền toàn thị trường... Vui lòng đợi ⏳")
    try:
        top_stocks = run_market_screener()
        if not top_stocks:
            await msg.edit_text("⚠️ Hiện tại thị trường chưa có mã nào thỏa mãn tiêu chí.")
            return
            
        text = "🚀 TOP CỔ PHIẾU DÒNG TIỀN TỔ CHỨC THU GOM\n\n"
        for idx, item in enumerate(top_stocks, 1):
            text += f"{idx}. {item['symbol']} | Giá: {item['close']:.1f}k | RVOL: {item['rvol']:.2f}x\n"
        await msg.edit_text(text)
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi khi quét thị trường: {str(e)}")


async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ACTIVE_CHAT_IDS.add(chat_id)
    args = context.args
    if not args:
        await update.message.reply_text("⚠️ **Thiếu mã cổ phiếu!**\n👉 Cú pháp: `/backtest <MÃ_CP>` (VD: `/backtest VHM`)", parse_mode="Markdown")
        return
    symbol = args[0].upper().strip()
    msg = await update.message.reply_text(f"⏳ Đang chạy thuật toán Backtest chiến lược cho mã {symbol}...")
    try:
        report = run_strategy_backtest(symbol)
        await msg.edit_text(report)
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi backtest: {str(e)}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_id = update.effective_chat.id
        ACTIVE_CHAT_IDS.add(chat_id)
        text = update.message.text.strip().upper()
        if len(text) < 3 or len(text) > 5 or not text.isalpha():
            await update.message.reply_text("⚠️ Vui lòng nhập mã hợp lệ (VD: VHM, PNJ, HPG), hoặc dùng /scan, /portfolio, /backtest!")
            return

        keyboard = [
            [InlineKeyboardButton("📅 1D (250 nến)", callback_data=f"{text}|D|250")],
            [
                InlineKeyboardButton("⏱️ 15m (200 nến)", callback_data=f"{text}|15|200"),
                InlineKeyboardButton("⏱️ 30m (200 nến)", callback_data=f"{text}|30|200"),
                InlineKeyboardButton("⏱️ 1H (250 nến)", callback_data=f"{text}|60|200"),
            ],
        ]
        await update.message.reply_text(
            f"🔍 Đã nhận mã **{text}**. Chọn khung thời gian phân tích:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi tiếp nhận mã: {str(e)}")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    try:
        if data.startswith("help_"):
            explanations = {
                "help_cmf": "💡 **Giải thích CMF & MFI:**\n• **CMF:** Dòng tiền tích lũy/phân phối lớn của tổ chức.\n• **MFI:** Chỉ số dòng tiền kết hợp khối lượng và giá (0-100).",
                "help_rsi": "💡 **Giải thích Ichimoku:**\n• Gồm mây Kumo, Tenkan, Kijun giúp xác định xu hướng và vùng hỗ trợ/kháng cự cực mạnh.",
                "help_vp": "💡 **Giải thích Volume Profile:**\n• **POC:** Mức giá khớp khối lượng nhiều nhất.\n• **VAL - VAH:** Vùng giá hỗ trợ & kháng cự tổ chức.",
                "help_portfolio": "💡 **Hướng dẫn Danh mục:**\n• Dùng `/setbuy VHM 63.0 62.0 1500` để lưu giá mua, SL & khối lượng quản trị rủi ro tự động."
            }
            await query.message.reply_text(explanations.get(data, "Đang cập nhật hướng dẫn..."), parse_mode="Markdown")
            return

        parts = data.split("|")
        if len(parts) == 3:
            symbol, resolution, count = parts[0], parts[1], int(parts[2])
            await query.edit_message_text(text=f"⏳ Chuyên gia đang phân tích chuyên sâu cho {symbol}...")

            def background_task():
                return analyze_stock(symbol, resolution=resolution, count=count)

            future = executor.submit(background_task)
            try:
                text_part1, text_part2, img_buf, reply_markup, err = future.result(timeout=15)
                if err:
                    await query.message.reply_text(f"❌ Lỗi phân tích mã {symbol}:\n{err}")
                else:
                    await query.message.reply_photo(photo=img_buf)
                    await query.message.reply_text(text_part1, parse_mode="Markdown")
                    await query.message.reply_text(text_part2, reply_markup=reply_markup, parse_mode="Markdown")
            except Exception as e:
                await query.message.reply_text(f"❌ Lỗi hệ thống trong quá trình xử lý mã {symbol}:\n{str(e)}")
    except Exception as e:
        await query.message.reply_text(f"❌ Lỗi Callback nút bấm: {str(e)}")


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Optimized Quant Bot with MA, MACD, Bollinger & Fixed Volume Profile is running live!")


def run_http_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler)
    server.serve_forever()


if __name__ == "__main__":
    threading.Thread(target=run_http_server, daemon=True).start()

    print("🤖 Đang khởi tạo kết nối Telegram Bot...")
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

    print("🚀 Bot Quant Trading System đã được tối ưu hóa toàn diện cấu trúc logic!")
    app.run_polling(drop_pending_updates=True)
