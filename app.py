import sys
import subprocess
import os
import time
from datetime import datetime, timedelta

# --- 0. 全自動起動ランチャー ---
def run_streamlit():
    try:
        import streamlit
        import streamlit_autorefresh
        import requests
        import pandas
        import yfinance
        import plotly
    except ImportError:
        print("初回セットアップ中...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly", "yfinance", "pandas", "streamlit-autorefresh", "requests"])
    
    is_streamlit_running = False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx():
            is_streamlit_running = True
    except:
        pass

    if not is_streamlit_running:
        print("アプリをブラウザで起動しています...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
        sys.exit()

run_streamlit()

# ==========================================
# V21.2 Trading Watcher (3-Layer Strategy: Scalp/Day/Swing)
# ==========================================
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import numpy as np
import requests

st.set_page_config(page_title="Trading Watcher V21.2", layout="wide")

# ==========================================
# ★設定エリア
# ==========================================
DISCORD_WEBHOOK_URL = "ここにコピーしたDiscordのウェブフックURLを貼り付けてください"

TICKER_NAMES = {
    "9984": "ソフトバンクG", "6857": "アドバンテスト", "5803": "フジクラ",
    "6920": "レーザーテック", "3563": "F&L (スシロー)", "8385": "伊予銀行",
    "5020": "ENEOS", "8136": "サンリオ", "3778": "さくらネット",
    "9107": "川崎汽船", "7011": "三菱重工", "8035": "東エレク",
    "8306": "三菱UFJ", "7203": "トヨタ自動車"
}
DEFAULT_FAVORITES = list(TICKER_NAMES.keys())

if 'favorites' not in st.session_state: st.session_state.favorites = DEFAULT_FAVORITES
if 'current_ticker' not in st.session_state: st.session_state.current_ticker = "9984"
if 'notified_signals' not in st.session_state: st.session_state.notified_signals = set()
if 'bt_results' not in st.session_state: st.session_state.bt_results = None
if 'alert_history' not in st.session_state: st.session_state.alert_history = []

def get_name(code): return TICKER_NAMES.get(code, code)

def send_discord_notify(msg):
    if not DISCORD_WEBHOOK_URL: return False
    try: requests.post(DISCORD_WEBHOOK_URL, json={"content": msg}); return True
    except: return False

@st.cache_data(ttl=15)
def get_data(ticker, period, interval):
    try:
        ticker_mod = f"{ticker}.T" if ".T" not in ticker and ticker.isdigit() else ticker
        df = yf.download(ticker_mod, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            if df.columns.nlevels > 1: df.columns = df.columns.droplevel(1)
        if df.empty: return None
        if df.index.tz is None:
            try: df.index = df.index.tz_localize('Asia/Tokyo')
            except: pass
        else:
            df.index = df.index.tz_convert('Asia/Tokyo')
        return df
    except: return None

# --- テクニカル計算 ---
def process_data(df, interval):
    df = df.copy()
    if "m" in interval or "h" in interval: df['DisplayDate'] = df.index.strftime('%m/%d %H:%M')
    else: df['DisplayDate'] = df.index.strftime('%Y/%m/%d')

    v = df['Volume']; tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (tp * v).cumsum() / v.cumsum()

    high = df['High']; low = df['Low']; close = df['Close']
    tr1 = high - low; tr2 = (high - close.shift()).abs(); tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    up = high.diff(); down = low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0); minus_dm = np.where((down > up) & (down > 0), down, 0)
    plus_dm = pd.Series(plus_dm, index=df.index); minus_dm = pd.Series(minus_dm, index=df.index)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    df['ADX'] = dx.ewm(alpha=1/14).mean()

    exp12 = close.ewm(span=12, adjust=False).mean(); exp26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26; df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))

    period=10; multiplier=3.0; atr_st = tr.rolling(period).mean()
    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr_st); basic_lower = hl2 - (multiplier * atr_st)
    final_upper = [np.nan]*len(df); final_lower = [np.nan]*len(df)
    supertrend = [np.nan]*len(df); trend_dir = [True]*len(df)
    
    for i in range(len(df)):
        if i < period:
            final_upper[i] = basic_upper.iloc[i]; final_lower[i] = basic_lower.iloc[i]; continue
        prev_close = close.iloc[i-1]
        if pd.isna(final_upper[i-1]) or pd.isna(final_lower[i-1]):
            final_upper[i] = basic_upper.iloc[i]; final_lower[i] = basic_lower.iloc[i]
        else:
            final_upper[i] = basic_upper.iloc[i] if basic_upper.iloc[i] < final_upper[i-1] or prev_close > final_upper[i-1] else final_upper[i-1]
            final_lower[i] = basic_lower.iloc[i] if basic_lower.iloc[i] > final_lower[i-1] or prev_close < final_lower[i-1] else final_lower[i-1]
        prev_trend = trend_dir[i-1]
        if prev_trend: trend_dir[i] = False if close.iloc[i] < final_lower[i] else True
        else: trend_dir[i] = True if close.iloc[i] > final_upper[i] else False
        supertrend[i] = final_lower[i] if trend_dir[i] else final_upper[i]
    df['SuperTrend'] = supertrend; df['SuperTrend_Dir'] = trend_dir

    # ★ 3層シグナル判定ロジック
    signals = []
    is_daily = "d" in interval # 日足
    is_5m = "5m" in interval   # 5分足

    for i in range(len(df)):
        if i < 30: signals.append(None); continue
        row = df.iloc[i]; prev = df.iloc[i-1]
        sig = None
        
        is_uptrend = row['SuperTrend_Dir'] == True
        is_downtrend = row['SuperTrend_Dir'] == False
        g_cross = prev['MACD'] < prev['Signal'] and row['MACD'] > row['Signal']
        d_cross = prev['MACD'] > prev['Signal'] and row['MACD'] < row['Signal']
        
        if is_daily:
            # 🐢 SWING (日足) - 緩和モード
            if is_uptrend and g_cross: sig = 'SWING_BUY'
            elif is_downtrend and d_cross: sig = 'SWING_SELL'
            elif row['RSI'] < 30 and row['Close'] > prev['Close']: sig = 'SWING_BUY (RSI)'
            elif row['RSI'] > 70 and row['Close'] < prev['Close']: sig = 'SWING_SELL (RSI)'
            elif row['SuperTrend_Dir'] == True and prev['SuperTrend_Dir'] == False: sig = 'SWING_BUY (Trend)'
            elif row['SuperTrend_Dir'] == False and prev['SuperTrend_Dir'] == True: sig = 'SWING_SELL (Trend)'
        
        elif is_5m:
            # 🛡️ DAY-STD (5分足) - 標準モード
            # ここではシンプルにトレンドフォローのみ。スキャン時に「60分足」との一致を確認する
            if is_uptrend and g_cross: sig = 'DAY_BUY'
            elif is_downtrend and d_cross: sig = 'DAY_SELL'
            
        else:
            # ⚡ DAY-FAST (1分足) - 厳格モード
            if pd.isna(row['ADX']) or pd.isna(row['VWAP']): signals.append(None); continue
            adx_ok = row['ADX'] > 25
            buy_vwap_ok = row['Close'] > row['VWAP']; sell_vwap_ok = row['Close'] < row['VWAP']
            if is_uptrend and g_cross and adx_ok and buy_vwap_ok: sig = 'SCALP_BUY'
            elif is_downtrend and d_cross and adx_ok and sell_vwap_ok: sig = 'SCALP_SELL'
            
        signals.append(sig)
    
    df['Trade_Signal'] = signals
    return df

# --- バックテスト ---
def run_backtest(df, tp_pct, trade_dir, shares):
    trades = []; active_trade = None 
    do_long = "買い" in trade_dir
    max_dd = 0; peak_equity = 0; equity = 0

    for i in range(len(df)):
        row = df.iloc[i]; sig = row['Trade_Signal']; st_val = row['SuperTrend']
        trade_closed = False; profit = 0

        if active_trade:
            entry_price = active_trade['entry_price']; entry_tp = active_trade['target_tp']
            if active_trade['type'] == 'buy':
                if row['High'] >= entry_tp:
                    profit = (entry_tp - entry_price) * shares
                    trades.append({'date': row['DisplayDate'], 'type': 'Buy', 'res': 'WIN 🏆', 'entry': entry_price, 'exit': entry_tp, 'profit': profit})
                    active_trade = None; trade_closed = True
                elif row['Close'] < st_val:
                    profit = (st_val - entry_price) * shares
                    res_label = 'WIN (Trail)' if profit > 0 else 'LOSE'
                    trades.append({'date': row['DisplayDate'], 'type': 'Buy', 'res': res_label, 'entry': entry_price, 'exit': st_val, 'profit': profit})
                    active_trade = None; trade_closed = True
            elif active_trade['type'] == 'sell':
                if row['Low'] <= entry_tp:
                    profit = (entry_price - entry_tp) * shares
                    trades.append({'date': row['DisplayDate'], 'type': 'Sell', 'res': 'WIN 🏆', 'entry': entry_price, 'exit': entry_tp, 'profit': profit})
                    active_trade = None; trade_closed = True
                elif row['Close'] > st_val:
                    profit = (entry_price - st_val) * shares
                    res_label = 'WIN (Trail)' if profit > 0 else 'LOSE'
                    trades.append({'date': row['DisplayDate'], 'type': 'Sell', 'res': res_label, 'entry': entry_price, 'exit': st_val, 'profit': profit})
                    active_trade = None; trade_closed = True
        
        if active_trade is None and not trade_closed and sig is not None:
            if do_long and "BUY" in sig:
                active_trade = {'entry_price': row['Close'], 'type': 'buy', 'target_tp': row['Close']*(1+tp_pct/100)}
            elif not do_long and "SELL" in sig:
                active_trade = {'entry_price': row['Close'], 'type': 'sell', 'target_tp': row['Close']*(1-tp_pct/100)}

        if trade_closed:
            equity += profit
            if equity > peak_equity: peak_equity = equity
            dd = peak_equity - equity
            if dd > max_dd: max_dd = dd

    return trades, max_dd

# --- スキャン機能 (3層構造) ---
def scan_signals(tickers):
    history_buffer = []
    scan_bar = st.progress(0, text="スキャン中...")
    total = len(tickers)
    
    for idx, t in enumerate(tickers):
        t_name = get_name(t)
        scan_bar.progress((idx + 1) / total, text=f"Analyzing: {t}...")
        
        # 上位足トレンドチェック用
        df_daily = get_data(t, "3mo", "1d")
        daily_trend = "NEUTRAL"
        if df_daily is not None and not df_daily.empty:
            df_daily = process_data(df_daily, "1d")
            daily_trend = "UP" if df_daily.iloc[-1]['SuperTrend_Dir'] else "DOWN"
            
        df_60m = get_data(t, "1mo", "60m")
        hourly_trend = "NEUTRAL"
        if df_60m is not None and not df_60m.empty:
            df_60m = process_data(df_60m, "60m")
            hourly_trend = "UP" if df_60m.iloc[-1]['SuperTrend_Dir'] else "DOWN"

        # ---------------------------
        # 1. 🐢 SWING (日足)
        # ---------------------------
        if df_daily is not None and not df_daily.empty:
            for i in range(len(df_daily)-1, -1, -1):
                sig = df_daily.iloc[i]['Trade_Signal']
                if sig:
                    row = df_daily.iloc[i]
                    bars_ago = len(df_daily) - 1 - i
                    is_forming = (bars_ago == 0)
                    status = "⚡日足形成" if is_forming else "🔒日足確定"
                    ago_label = "今日" if is_forming else f"{bars_ago}日前"
                    
                    history_buffer.append({
                        "dt": df_daily.index[i], "time_str": df_daily.index[i].strftime("%Y/%m/%d"),
                        "code": t, "name": t_name, "sig": sig, "price": row['Close'], 
                        "status": status, "type": "SWING (日足)", "ago_label": ago_label
                    })
                    
                    sig_id = f"{df_daily.index[i]}_{t}_{sig}_SWING"
                    if bars_ago == 1 and sig_id not in st.session_state.notified_signals:
                        emoji = "🌊" if "BUY" in sig else "📉"
                        send_discord_notify(f"**{emoji} [SWING] {sig} 確定**\n銘柄: {t_name}\n(日足: チャンス感知)")
                        st.session_state.notified_signals.add(sig_id)
                    if bars_ago > 7: break

        # ---------------------------
        # 2. 🛡️ DAY-STD (5分足) ★NEW!
        # ---------------------------
        df_5m = get_data(t, "5d", "5m")
        if df_5m is not None and not df_5m.empty:
            df_5m = process_data(df_5m, "5m")
            for i in range(len(df_5m)-1, -1, -1):
                sig = df_5m.iloc[i]['Trade_Signal']
                if sig:
                    # 60分足とのトレンド一致フィルター
                    if "BUY" in sig and hourly_trend == "DOWN": continue
                    if "SELL" in sig and hourly_trend == "UP": continue
                    
                    row = df_5m.iloc[i]
                    bars_ago = len(df_5m) - 1 - i
                    is_forming = (bars_ago == 0)
                    status = "⚡5分足形成" if is_forming else "🔒5分足確定"
                    
                    history_buffer.append({
                        "dt": df_5m.index[i], "time_str": df_5m.index[i].strftime("%m/%d %H:%M"),
                        "code": t, "name": t_name, "sig": sig, "price": row['Close'], 
                        "status": status, "type": "DAY-STD (5分)", "ago_label": f"{bars_ago*5}分前"
                    })
                    
                    sig_id = f"{df_5m.index[i]}_{t}_{sig}_DAYSTD"
                    if bars_ago == 1 and sig_id not in st.session_state.notified_signals:
                        emoji = "🟢" if "BUY" in sig else "🔴"
                        send_discord_notify(f"**{emoji} [DAY-STD] {sig} 確定**\n銘柄: {t_name}\n価格: {row['Close']:,.0f}円\n(5分足: 中期トレンド一致)")
                        st.session_state.notified_signals.add(sig_id)
                    if bars_ago > 12: break # 直近1時間

        # ---------------------------
        # 3. ⚡ DAY-FAST (1分足)
        # ---------------------------
        df_1m = get_data(t, "1d", "1m") # 高速化のため期間短縮
        if df_1m is not None and not df_1m.empty:
            df_1m = process_data(df_1m, "1m")
            for i in range(len(df_1m)-1, -1, -1):
                sig = df_1m.iloc[i]['Trade_Signal']
                if sig:
                    if "BUY" in sig and daily_trend == "DOWN": continue
                    if "SELL" in sig and daily_trend == "UP": continue
                    
                    row = df_1m.iloc[i]
                    bars_ago = len(df_1m) - 1 - i
                    is_forming = (bars_ago == 0)
                    status = "⚡1分足形成" if is_forming else "🔒1分足確定"
                    
                    history_buffer.append({
                        "dt": df_1m.index[i], "time_str": df_1m.index[i].strftime("%m/%d %H:%M"),
                        "code": t, "name": t_name, "sig": sig, 
                        "price": row['Close'], "status": status,
                        "type": "DAY-FAST (1分)", "ago_label": f"{bars_ago}分前"
                    })
                    
                    sig_id = f"{df_1m.index[i]}_{t}_{sig}_DAYFAST"
                    if bars_ago == 1 and sig_id not in st.session_state.notified_signals:
                        emoji = "🔥" if "BUY" in sig else "❄️"
                        send_discord_notify(f"**{emoji} [DAY-FAST] {sig} 確定**\n銘柄: {t_name}\n価格: {row['Close']:,.0f}円\n(1分足: 厳格条件突破)")
                        st.session_state.notified_signals.add(sig_id)
                    if bars_ago > 5: break

    scan_bar.empty()
    history_buffer.sort(key=lambda x: x['dt'], reverse=True)
    st.session_state.alert_history = history_buffer
    return history_buffer

def display_signal_cards(signal_list, use_cols=4):
    if not signal_list: return
    cols = st.columns(use_cols)
    for i, item in enumerate(signal_list):
        with cols[i % use_cols]:
            color = "#d32f2f" if "BUY" in item['sig'] else "#00796b"
            icon = "🔥" if "BUY" in item['sig'] else "🧊"
            status_text = item.get('status', '履歴'); border = "dashed" if "形成" in status_text else "solid"
            
            src_type = item.get('type', 'Unknown')
            if "SWING" in src_type: badge_style = "background-color:#3F51B5; color:white"
            elif "STD" in src_type: badge_style = "background-color:#4CAF50; color:white" # 5分は緑
            else: badge_style = "background-color:#FF9800; color:white" # 1分はオレンジ

            st.markdown(f"""
            <div style="border:2px {border} {color}; padding:10px; border-radius:8px; margin-bottom:10px; background-color:#262730;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
                    <span style="font-weight:bold; color:#fff; font-size:1.1em;">{item['name']}</span>
                    <span style="{badge_style}; padding:2px 6px; border-radius:4px; font-size:0.75em;">{src_type}</span>
                </div>
                <div style="font-size:0.85em; color:#ddd;">{status_text} | {item.get('ago_label')}</div>
                <div style="color:{color}; font-weight:900; font-size:1.1em; margin: 5px 0;">{icon} {item['sig']}</div>
                <div style="color:#fff; font-size:1.1em;">{item['price']:,.0f} 円</div>
            </div>
            """, unsafe_allow_html=True)

# --- レイアウト ---
st.sidebar.title("💎 Watcher 21.2")
if st.sidebar.button("🔔 通知テスト"):
    if send_discord_notify("🔔 [TEST] System Normal."): st.sidebar.success("OK")
    else: st.sidebar.error("NG")

if st.session_state.current_ticker in st.session_state.favorites: fav_idx = st.session_state.favorites.index(st.session_state.current_ticker)
else: fav_idx = 0
def update_ticker(): st.session_state.current_ticker = st.session_state.ticker_radio
st.sidebar.radio("銘柄選択", st.session_state.favorites, index=fav_idx, key="ticker_radio", on_change=update_ticker, format_func=lambda x: f"{x} {get_name(x)}")

with st.sidebar.expander("リスト編集"):
    new_code = st.text_input("コード追加")
    if st.button("追加"): 
        if new_code and new_code not in st.session_state.favorites: st.session_state.favorites.append(new_code); st.rerun()
    del_targets = st.multiselect("削除", st.session_state.favorites, format_func=lambda x: f"{x} {get_name(x)}")
    if st.button("削除実行"):
        for t in del_targets: 
            if t in st.session_state.favorites: st.session_state.favorites.remove(t)
        if st.session_state.current_ticker not in st.session_state.favorites and st.session_state.favorites: st.session_state.current_ticker = st.session_state.favorites[0]
        st.rerun()

with st.sidebar.expander("🛡 ロット計算機", expanded=True):
    fund = st.number_input("余力 (円)", 100000, 100000000, 3000000, step=100000)
    risk_pct = st.number_input("許容リスク (%)", 0.1, 5.0, 1.0, 0.1)
    max_loss = fund * (risk_pct / 100)
    st.caption(f"1トレード許容損失: {max_loss:,.0f}円")
    stop_range = st.number_input("損切幅 (円)", 0, 5000, 30)
    if stop_range > 0:
        rec_shares = max_loss // stop_range
        st.markdown(f"推奨: **{rec_shares:,.0f} 株**")

c1, c2 = st.sidebar.columns(2)
period = c1.selectbox("期間", ["1d", "5d", "1mo", "3mo"], index=1)
interval = c2.selectbox("時間足", ["1m", "5m", "15m", "1h", "1d"], index=0)
auto_refresh = st.sidebar.checkbox("自動更新 (60s)", True)
if auto_refresh: st_autorefresh(interval=60*1000, key="refresh")

st.sidebar.subheader("💰 バックテスト設定")
tp_pct = st.sidebar.number_input("利確 (%)", 0.1, 100.0, 1.0, 0.1) 
trade_shares = st.sidebar.number_input("株数", 100, 10000, 100, 100)
trade_dir = st.sidebar.radio("売買方向", ["買い", "売り"], horizontal=True)

# --- メイン画面 ---
tab_mon, tab_anl = st.tabs(["🚨 監視パネル", "📈 統合分析"])

with tab_mon:
    if st.button("スキャン更新 (3層分析)", type="primary"): st.rerun()
    all_history = scan_signals(st.session_state.favorites)
    now = pd.Timestamp.now(tz='Asia/Tokyo'); one_day_ago = now - pd.Timedelta(days=1)
    
    recent_list = [h for h in all_history if h['dt'] >= one_day_ago]
    past_list = [h for h in all_history if h['dt'] < one_day_ago]
            
    st.markdown("### 🔔 直近24時間 (全銘柄)")
    if recent_list: display_signal_cards(recent_list)
    else: st.info("直近24時間にシグナルはありません。")
    st.markdown("---")
    st.markdown("### 📜 過去1週間の履歴 (全銘柄)")
    if past_list:
        df_view = pd.DataFrame(past_list)
        st.dataframe(df_view[['time_str', 'type', 'name', 'code', 'sig', 'price', 'status']], use_container_width=True)
    else: st.caption("過去1週間のシグナルはありません。")

with tab_anl:
    t = st.session_state.current_ticker; t_name = get_name(t)
    st.header(f"{t} {t_name}")
    
    df_5m = get_data(t, "5d", "5m"); df_60m = get_data(t, "1mo", "60m"); df_d = get_data(t, "3mo", "1d")
    trend_res = {"5m": "NONE", "60m": "NONE", "1d": "NONE"}
    if df_5m is not None: df_5m = process_data(df_5m, "5m"); trend_res["5m"] = "UP" if df_5m.iloc[-1]['SuperTrend_Dir'] else "DOWN"
    if df_60m is not None: df_60m = process_data(df_60m, "60m"); trend_res["60m"] = "UP" if df_60m.iloc[-1]['SuperTrend_Dir'] else "DOWN"
    if df_d is not None: df_d = process_data(df_d, "1d"); trend_res["1d"] = "UP" if df_d.iloc[-1]['SuperTrend_Dir'] else "DOWN"

    s = trend_res["5m"]; m = trend_res["60m"]; l = trend_res["1d"]
    nav_title = ""; nav_desc = ""; nav_color = ""
    if s=="UP" and m=="UP" and l=="UP": nav_title = "🟢 SWING POSSIBLE"; nav_color = "success"; nav_desc = "全期間上昇。持ち越し可。"
    elif s=="UP" and l=="DOWN": nav_title = "🔴 DAY TRADE ONLY"; nav_color = "error"; nav_desc = "長期下落中。本日中に手仕舞い推奨。"
    elif s=="DOWN" and l=="UP": nav_title = "⚠️ WAITING"; nav_color = "warning"; nav_desc = "押し目待ち。"
    elif s=="DOWN" and m=="DOWN" and l=="DOWN": nav_title = "🟢 SELLING SWING"; nav_color = "success"; nav_desc = "全期間下落。売り持ち越し可。"
    else: nav_title = "👀 OBSERVATION"; nav_color = "info"; nav_desc = "方向感なし。"

    st.subheader("🧭 推奨戦略ナビゲーター")
    with st.container():
        if nav_color == "success": st.success(f"### {nav_title}\n{nav_desc}")
        elif nav_color == "error": st.error(f"### {nav_title}\n{nav_desc}")
        elif nav_color == "warning": st.warning(f"### {nav_title}\n{nav_desc}")
        else: st.info(f"### {nav_title}\n{nav_desc}")
        c1, c2, c3 = st.columns(3)
        c1.metric("短期(5m)", s, delta_color="normal" if s=="UP" else "inverse")
        c2.metric("中期(60m)", m, delta_color="normal" if m=="UP" else "inverse")
        c3.metric("長期(Day)", l, delta_color="normal" if l=="UP" else "inverse")
    
    st.markdown("---")

    df_chart = get_data(t, period, interval)
    if df_chart is not None:
        df_chart = process_data(df_chart, interval)
        last = df_chart.iloc[-1]
        c_price, c_v, c_a = st.columns(3)
        price_delta = 0
        if len(df_chart) > 1: price_delta = last['Close'] - df_chart.iloc[-2]['Close']
        c_price.metric("現在株価", f"{last['Close']:,.0f}", f"{price_delta:,.0f}")
        vwap_color = "normal" if (last['Close'] > last['VWAP'] and last['SuperTrend_Dir']) else "inverse"
        c_v.metric("VWAP乖離", f"{last['Close'] - last['VWAP']:.0f}", delta="価格有利" if vwap_color=="normal" else "価格不利", delta_color=vwap_color)
        c_a.metric("ADX強度", f"{last['ADX']:.1f}", delta="トレンド中" if last['ADX']>25 else "レンジ", delta_color="normal" if last['ADX']>25 else "off")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], subplot_titles=("Price & VWAP", "MACD", "ADX"))
        x = df_chart['DisplayDate']
        fig.add_trace(go.Candlestick(x=x, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df_chart['VWAP'], mode='lines', line=dict(color='purple'), name='VWAP'), row=1, col=1)
        st_g = df_chart['SuperTrend'].copy(); st_g[df_chart['SuperTrend_Dir'] == False] = None
        st_r = df_chart['SuperTrend'].copy(); st_r[df_chart['SuperTrend_Dir'] == True] = None
        fig.add_trace(go.Scatter(x=x, y=st_g, mode='lines', line=dict(color='blue'), name='Support'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=st_r, mode='lines', line=dict(color='orange'), name='Resist'), row=1, col=1)
        
        # マーカープロット (現在表示中の時間足のシグナルのみ)
        buy = df_chart[df_chart['Trade_Signal'].str.contains('BUY', na=False)]
        sell = df_chart[df_chart['Trade_Signal'].str.contains('SELL', na=False)]
        if not buy.empty: fig.add_trace(go.Scatter(x=buy['DisplayDate'], y=buy['Low'], mode='markers', marker=dict(symbol='triangle-up', size=15, color='red'), name='BUY'), row=1, col=1)
        if not sell.empty: fig.add_trace(go.Scatter(x=sell['DisplayDate'], y=sell['High'], mode='markers', marker=dict(symbol='triangle-down', size=15, color='green'), name='SELL'), row=1, col=1)

        fig.add_trace(go.Bar(x=x, y=df_chart['MACD']-df_chart['Signal'], name='Hist'), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=df_chart['MACD'], name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=df_chart['ADX'], name='ADX', line=dict(color='white')), row=3, col=1)
        fig.add_hline(y=25, line_dash="dot", row=3, col=1)
        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🧪 バックテスト検証")
        min_dt = df_chart.index.min().to_pydatetime(); max_dt = df_chart.index.max().to_pydatetime()
        bt_range = st.slider("検証期間", min_value=min_dt, max_value=max_dt, value=(min_dt, max_dt), format="MM/DD")
        
        if st.button("バックテスト実行", type="primary"):
            # 期間補完
            if period == "1d" and interval == "1m": df_calc = get_data(t, "5d", "1m")
            elif period == "1d" and interval == "5m": df_calc = get_data(t, "5d", "5m")
            else: df_calc = get_data(t, period, interval)
            
            if df_calc is not None:
                df_calc = process_data(df_calc, interval)
                df_bt = df_calc[(df_calc.index >= bt_range[0]) & (df_calc.index <= bt_range[1])]
                trades, max_dd = run_backtest(df_bt, tp_pct, trade_dir, trade_shares)
                st.session_state.bt_results = {"trades": trades, "max_dd": max_dd}
            else: st.error("データ取得エラー")

        if st.session_state.bt_results is not None:
            trades = st.session_state.bt_results["trades"]; max_dd = st.session_state.bt_results["max_dd"]
            if len(trades) > 0:
                wins = len([t for t in trades if 'WIN' in t['res']]); pl = sum([t['profit'] for t in trades]); rate = (wins/len(trades))*100
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("回数", len(trades)); c2.metric("勝率", f"{rate:.1f}%"); c3.metric("損益", f"{pl:,.0f}", delta=pl)
                c4.metric("最大DD", f"-{max_dd:,.0f}", delta=-max_dd, delta_color="inverse")
                st.dataframe(pd.DataFrame(trades)[['date','type','res','entry','exit','profit']].round(0), use_container_width=True)
            else: st.warning("期間内にシグナルはありませんでした。")
    
    st.markdown("---")
    st.subheader("📜 シグナル履歴 (過去1週間)")
    # 履歴は1分足で代表表示
    df_hist = get_data(t, "5d", "1m")
    if df_hist is not None and not df_hist.empty:
        df_hist = process_data(df_hist, "1m")
        past_signals = []
        for i in range(len(df_hist)-1, -1, -1):
            row = df_hist.iloc[i]
            if row['Trade_Signal']:
                past_signals.append({
                    "name": t_name, "code": t, "time_str": row['DisplayDate'], "ago": row['DisplayDate'],
                    "sig": row['Trade_Signal'], "price": row['Close'], "status": "🔒確定", "dt": df_hist.index[i], "type": "DAY-FAST"
                })
        now = pd.Timestamp.now(tz='Asia/Tokyo'); one_day_ago = now - pd.Timedelta(days=1)
        recent_list = [h for h in past_signals if h['dt'] >= one_day_ago]
        past_list = [h for h in past_signals if h['dt'] < one_day_ago]
        st.markdown("##### 🔔 直近24時間以内")
        if recent_list: display_signal_cards(recent_list)
        else: st.info("直近24時間のシグナルはありません")
        st.markdown("##### ⏮ それ以前 (1週間以内)")
        if past_list:
            view_data = [{"日時": i['time_str'], "シグナル": i['sig'], "価格": f"{i['price']:,.0f}", "状態": i['status']} for i in past_list]
            st.dataframe(pd.DataFrame(view_data), use_container_width=True)
        else: st.info("過去1週間のシグナルはありません")