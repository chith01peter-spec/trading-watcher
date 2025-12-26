import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
from datetime import datetime, timedelta

# ==========================================
# ⚙️ 設定 & 定数
# ==========================================
try:
    DISCORD_WEBHOOK_URL = st.secrets["DISCORD_URL"]
except:
    DISCORD_WEBHOOK_URL = ""

# 💎 宋スペシャル・パック (50銘柄)
WATCH_LIST = [
    "9984", "6857", "5803", "6920", "3563", "8385", "5020", "8136", "3778", 
    "9107", "7011", "8035", "8306", "7203", "6146", "6526", "7735", "6723", 
    "6758", "6367", "8316", "8411", "8001", "8002", "8058", "7012", "7013",
    "5253", "5032", "5574", "9166", "2160", "2413", "4385", "4483", "9613",
    "9983", "7974", "4661", "3099", "3382", "8267", "9843", "9501", "7267", 
    "6501", "6701", "4502", "4568", "2914", "4911"
]
WATCH_LIST = sorted(list(set(WATCH_LIST)))

TICKER_MAP = {
    "9984": "SBG", "6857": "アドバン", "6920": "レーザー", "8306": "三菱UFJ", 
    "8035": "東エレク", "6146": "ディスコ", "6526": "ソシオ", "7735": "SCREEN",
    "5253": "カバー", "5032": "ANYCOLOR", "9166": "GENDA", "7011": "三菱重", 
    "5803": "フジクラ", "8001": "伊藤忠", "9107": "川崎船", "7203": "トヨタ",
    "8316": "三井住友", "8058": "三菱商", "4661": "OLC", "7974": "任天堂"
}

def get_name(code):
    return TICKER_MAP.get(code, code)

# ==========================================
# 📊 テクニカル計算ロジック
# ==========================================
def calculate_technical_indicators(df):
    if df is None or df.empty: return None
    df = df.copy()
    
    # VWAP
    try:
        v = df['Volume']
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    except: df['VWAP'] = np.nan

    # MACD
    close = df['Close']
    df['MACD'] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))

    # ADX & SuperTrend (Simplified Loop for Batch Speed)
    high = df['High']; low = df['Low']
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    # ADX
    up = high.diff(); down = -low.diff()
    p_dm = np.where((up > down) & (up > 0), up, 0)
    m_dm = np.where((down > up) & (down > 0), down, 0)
    p_di = 100 * (pd.Series(p_dm, index=df.index).ewm(alpha=1/14).mean() / atr)
    m_di = 100 * (pd.Series(m_dm, index=df.index).ewm(alpha=1/14).mean() / atr)
    df['ADX'] = (abs(p_di - m_di) / abs(p_di + m_di) * 100).ewm(alpha=1/14).mean()

    # SuperTrend (Period=10, Mult=3)
    # ループ計算を関数内で高速に行う
    basic_upper = (high + low) / 2 + (3 * tr.rolling(10).mean())
    basic_lower = (high + low) / 2 - (3 * tr.rolling(10).mean())
    
    supertrend = [np.nan] * len(df)
    trend_dir = [True] * len(df)
    curr_trend = True
    curr_st = basic_lower.iloc[0] if not pd.isna(basic_lower.iloc[0]) else 0
    
    for i in range(len(df)):
        if pd.isna(basic_upper.iloc[i]): continue
        c = close.iloc[i]
        
        # Upper/Lower Logic
        u = basic_upper.iloc[i]
        l = basic_lower.iloc[i]
        # (簡易化のため直近値比較のみ実装)
        
        if curr_trend: # UP
            if c < curr_st: curr_trend = False; curr_st = u
            else: curr_st = max(curr_st, l)
        else: # DOWN
            if c > curr_st: curr_trend = True; curr_st = l
            else: curr_st = min(curr_st, u)
            
        supertrend[i] = curr_st
        trend_dir[i] = curr_trend
        
    df['SuperTrend'] = supertrend
    df['SuperTrend_Dir'] = trend_dir
    
    # Signals
    sigs = []
    for i in range(len(df)):
        if i < 30: sigs.append(None); continue
        r = df.iloc[i]; p = df.iloc[i-1]
        s = None
        gc = p['MACD'] < p['Signal'] and r['MACD'] > r['Signal']
        dc = p['MACD'] > p['Signal'] and r['MACD'] < r['Signal']
        if r['SuperTrend_Dir'] and gc: s = "BUY"
        elif not r['SuperTrend_Dir'] and dc: s = "SELL"
        sigs.append(s)
    df['Trade_Signal'] = sigs
    
    return df

# ==========================================
# 📡 データ取得・スキャン・通知
# ==========================================
def send_discord(msg):
    if not DISCORD_WEBHOOK_URL: return False
    try: requests.post(DISCORD_WEBHOOK_URL, json={"content": msg}); return True
    except: return False

@st.cache_data(ttl=30)
def fetch_batch_data(tickers):
    try:
        ts = [f"{t}.T" for t in tickers]
        return yf.download(ts, period="5d", interval="5m", group_by='ticker', auto_adjust=False, progress=False, threads=True)
    except: return None

def scan_market(notified_ids):
    data = fetch_batch_data(WATCH_LIST)
    if data is None: return [], notified_ids
    
    results = []
    now = pd.Timestamp.now(tz='Asia/Tokyo')
    
    for code in WATCH_LIST:
        try:
            key = f"{code}.T"
            if key not in data['Close'].columns: continue
            
            df_t = pd.DataFrame({
                'Open': data['Open'][key], 'High': data['High'][key],
                'Low': data['Low'][key], 'Close': data['Close'][key],
                'Volume': data['Volume'][key]
            }).dropna()
            
            if df_t.empty: continue
            if df_t.index.tz is None: df_t.index = df_t.index.tz_localize('Asia/Tokyo')
            else: df_t.index = df_t.index.tz_convert('Asia/Tokyo')

            df_calc = calculate_technical_indicators(df_t)
            if df_calc is None: continue
            
            row = df_calc.iloc[-1]
            sig = row['Trade_Signal']
            
            if sig:
                note = []
                if row['RSI'] > 75: note.append("⚠️RSI過熱")
                if row['RSI'] < 25: note.append("⚠️RSI底")
                if row['ADX'] < 20: note.append("📉レンジ")
                note_str = " ".join(note)
                
                results.append({
                    "code": code, "name": get_name(code), "time": row.name,
                    "sig": sig, "price": row['Close'], "rsi": row['RSI'],
                    "sl": row['SuperTrend'], "note": note_str
                })
                
                # 通知
                sig_id = f"{row.name}_{code}_{sig}"
                if (now - row.name) < timedelta(minutes=30) and sig_id not in notified_ids:
                    emoji = "🚀" if "BUY" in sig else "🥀"
                    msg = f"**{emoji} {get_name(code)} {sig}**\n値: {row['Close']:,.0f} | RSI:{row['RSI']:.0f}\n損切: {row['SuperTrend']:,.0f}\n{note_str}"
                    if send_discord(msg): notified_ids.add(sig_id)
        except: continue
        
    results.sort(key=lambda x: x['time'], reverse=True)
    return results, notified_ids

# ==========================================
# 🧪 バックテスト
# ==========================================
def run_backtest(df, tp_pct, trade_dir, shares):
    trades = []; active = None
    do_long = "買い" in trade_dir or "両方" in trade_dir
    do_short = "売り" in trade_dir or "両方" in trade_dir
    max_dd = 0; peak = 0; equity = 0
    
    for i in range(len(df)):
        row = df.iloc[i]; sig = row['Trade_Signal']; st_v = row['SuperTrend']
        closed = False; profit = 0
        
        if active:
            ep = active['price']; tp = active['tp']
            if active['type'] == 'buy':
                if row['High'] >= tp: profit=(tp-ep)*shares; closed=True; res="WIN🏆"
                elif row['Close'] < st_v: profit=(st_v-ep)*shares; closed=True; res="Trail"
            elif active['type'] == 'sell':
                if row['Low'] <= tp: profit=(ep-tp)*shares; closed=True; res="WIN🏆"
                elif row['Close'] > st_v: profit=(ep-st_v)*shares; closed=True; res="Trail"
            
            if closed:
                trades.append({'date':row.name, 'type':active['type'], 'res':res, 'profit':profit, 'entry':ep, 'exit':tp if "WIN" in res else st_v})
                active = None; equity += profit
                peak = max(peak, equity); max_dd = max(max_dd, peak-equity)
        
        if not active and sig:
            if do_long and "BUY" in sig: active={'type':'buy', 'price':row['Close'], 'tp':row['Close']*(1+tp_pct/100)}
            elif do_short and "SELL" in sig: active={'type':'sell', 'price':row['Close'], 'tp':row['Close']*(1-tp_pct/100)}
            
    return trades, max_dd