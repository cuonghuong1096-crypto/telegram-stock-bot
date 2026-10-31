import datetime
import io
import json
import logging
import os
import html
import threading
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# ==========================================
# CẤU HÌNH TOKEN & API KEY CHUẨN TCBS OPENAPI
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TCBS_API_KEY = os.environ.get("TCBS_API_KEY", "10001010079-bf495c0f-c321-42c6-bd6c-01d582ef3670")
TCBS_OTP = os.environ.get("TCBS_OTP", "288038") 

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Cache lưu JWT Token
TCBS_TOKEN_CACHE = {"token": None, "time": 0}

def get_tcbs_jwt_token():
    now = time.time()
    if TCBS_TOKEN_CACHE["token"] and (now - TCBS_TOKEN_CACHE["time"] < 3300):
        return TCBS_TOKEN_CACHE["token"]

    token_url = "https://openapi.tcbs.com.vn/gaia/v1/oauth2/openapi/token"
    payload = {"apiKey": TCBS_API_KEY, "otp": TCBS_OTP}
    headers = {"Content-Type": "application/json"}
    try:
        res = requests.post(token_url, json=payload, headers=headers, timeout=5.0)
        if res.status_code == 200:
            data = res.json()
            jwt_token = data.get("token")
            if jwt_token:
                TCBS_TOKEN_CACHE["token"] = jwt_token
                TCBS_TOKEN_CACHE["time"] = now
                return jwt_token
    except Exception as e:
        logging.error(f"Lỗi lấy TCBS JWT Token: {e}")
    
    return TCBS_API_KEY

STOCK_DATA_CACHE = {} 
executor = ThreadPoolExecutor(max_workers=30)

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

    tcbs_url = f"https://openapi.tcbs.com.vn/stock-insight/v1/stock/bars?resolution={resolution}&ticker={symbol}&from={start_time}&to={end_time}"
    df_raw = pd.DataFrame()
    try:
        token = get_tcbs_jwt_token()
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        res = requests.get(tcbs_url, headers=headers, timeout=4.0)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                df_raw = pd.DataFrame(data)
                df_raw["time"] = pd.to_datetime(df_raw["tradingDate"])
                df_raw.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    except Exception as e:
        err_msg = str(e)
        return pd.DataFrame(), err_msg

    if df_raw.empty:
        return pd.DataFrame(), "Không có dữ liệu từ TCBS OpenAPI"

    df_raw = df_raw[["time", "Open", "High", "Low", "Close", "Volume"]].sort_values("time").reset_index(drop=True)
    ret_df = df_raw.tail(count).reset_index(drop=True)
    STOCK_DATA_CACHE[cache_key] = (ret_df, None, now_ts)
    return ret_df, None

def get_foreign_trading_data():
    parsed_list = []
    indexes_to_fetch = [1, 2] 
    token = get_tcbs_jwt_token()
    
    for idx_val in indexes_to_fetch:
        url = f"https://openapi.tcbs.com.vn/tartarus/v1/tickerSnaps?index={idx_val}"
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }
            res = requests.get(url, headers=headers, timeout=4)
            if res.status_code == 200:
                body = res.json()
                data_items = body.get("data", [])
                for item in data_items:
                    ticker = item.get("symbol", "")
                    if not ticker: continue
                    price = float(item.get("matchPrice", 0) or item.get("refPrice", 0) or 0)
                    buy_qtty = float(item.get("buyForeignQtty", 0) or 0)
                    sell_qtty = float(item.get("sellForeignQtty", 0) or 0)
                    net_qtty = buy_qtty - sell_qtty
                    net_val = (net_qtty * price) / 1000.0 if price > 0 else 0.0
                    if net_val != 0:
                        parsed_list.append({"ticker": ticker.upper(), "netValue": round(net_val, 2)})
        except Exception as e:
            logging.error(f"Error fetching tickerSnaps: {e}")

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
# CÁC LỆNH TELEGRAM BOT
# ==========================================
async def foreign_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message if update.message else update.callback_query.message
    msg = await target_msg.reply_text("🌍 Đang gọi thông tin khối ngoại từ TCBS OpenAPI...", parse_mode="HTML")
    try:
        loop = asyncio.get_running_loop()
        data, err = await loop.run_in_executor(executor, get_foreign_trading_data)
        if not data:
            await msg.edit_text(f"⚠️ Lỗi lấy dữ liệu khối ngoại từ TCBS: {err}", parse_mode="HTML")
            return
            
        text = "🌍 **THÔNG TIN KHỐI NGOẠI (TCBS OPENAPI)**\n====================================\n\n"
        for idx, item in enumerate(data[:12], 1):
            act = "🟢 Mua ròng" if item["netValue"] > 0 else "🔴 Bán ròng" if item["netValue"] < 0 else "⚪"
            text += f"{idx}. **{item['ticker']}**: `{item['netValue']:+.2f} tỷ` ({act})\n"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Lỗi hệ thống: {html.escape(str(e))}", parse_mode="HTML")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "👋 **CHÀO MỪNG BẠN ĐẾN VỚI QUANT TRADING SYSTEM PRO!**\n\n"
        "1️⃣ **TRA CỨU NHANH:** Gửi mã cổ phiếu bất kỳ\n"
        "2️⃣ **LỆNH KHỐI NGOẠI:** Gửi `/khoingoai` để xem dữ liệu chuẩn OpenAPI\n"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

# HTTP Server để Render ghi nhận port và phục vụ cơ chế Keep-Alive
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Quant Trading Bot Active & Running!")
    def log_message(self, format, *args):
        return # Tắt log rác HTTP để terminal gọn gàng

def run_http_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler)
    server.serve_forever()

# Hàm tự động ping chính nó mỗi 4 phút để chống ngủ đông trên Render
def self_ping_loop():
    app_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not app_url:
        return
    while True:
        time.sleep(240) # 4 phút
        try:
            urllib.request.urlopen(app_url, timeout=10)
        except Exception:
            pass

if __name__ == "__main__":
    # Khởi chạy HTTP Server nền
    threading.Thread(target=run_http_server, daemon=True).start()
    
    # Khởi chạy luồng tự động giữ sống bot nếu deploy trên Render
    threading.Thread(target=self_ping_loop, daemon=True).start()
    
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("khoingoai", foreign_command))  

    print("🚀 Bot đã khởi động mượt mà, sẵn sàng phản hồi tức thì!")
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)
