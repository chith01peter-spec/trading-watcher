import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import numpy as np
import requests
from datetime import datetime, timedelta

# ==========================================
# System Setup & Configuration
# ==========================================
st.set_page_config(page_title="Trading Watcher V25.1", layout="wide", page_icon="📈")

# ★ Streamlit CloudのSecretsからDiscord URLを取得
# (ローカルテスト用には空文字を入れていますが、本番はSecrets設定必須)
try:
    DISCORD_WEBHOOK_URL = st.secrets["DISCORD_URL"]
except:
    DISCORD_WEBHOOK_URL = ""

# 監視対象銘柄
TICKER_NAMES = {
    "9984": "ソフトバンクG", "6857": "アドバンテスト", "5803": "フジクラ",
    "6920": "レーザーテック", "3563": "F&L (スシロー)", "8385": "伊予銀行",
    "5020": "ENEOS", "8136": "サンリオ", "3778": "さくらネット",
    "9107": "川崎汽船", "7011": "三菱重工", "8035": "東エレク",
    "8306": "三菱UFJ", "7203": "トヨタ自動車"
}
DEFAULT_FAVORITES = list(TICKER_NAMES.keys())

# --- グローバルメモリ (通知済みID管理) ---
@st.cache_resource
def get_global_state():
    return {"notified_ids": set()}

global_state = get_global_state()

# セッション状態の初期化
if 'favorites' not in st.session_state: st.session_state.favorites = DEFAULT_FAVORITES
if 'current_ticker' not in st.session_state: st.session_state.current_ticker = "9984"
if 'bt_results' not in st.session_state: st.session_state.bt_results = None

def get_name(code):
    return TICKER_NAMES.get(code, code)

# Discord通知関数 (損切り目安を追加)
def send_discord_notify(msg):
    if not DISCORD_WEBHOOK_URL:
        return False
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
        return True
    except:
        return False

# データ取得 (キャッシング強化 & 型安全化)
@st.cache_data(ttl=10)
def get_data(ticker, period, interval):
    try:
        ticker_mod = f"{ticker}.T" if ".T" not in ticker and ticker.isdigit() else ticker
        # auto_adjust=False で生の価格を取得し、計算のズレを防ぐ
        df = yf.download(ticker_mod, period=period, interval=interval, progress=False, auto_adjust=False, multi_level_index=False)
        
        if df.empty: return None
        
        # タイムゾーン処理
        if df.index.tz is None:
            df.index = df.index.tz_localize('Asia/Tokyo')
        else:
            df.index = df.index.tz_convert('Asia/Tokyo')
        return df
    except Exception as e:
        return None

# テクニカル指標計算
def process_data(df, interval):
    df = df.copy()
    # 表示用日付フォーマット
    if "m" in interval or "h" in interval:
        df['DisplayDate'] = df.index.strftime('%m/%d %H:%M')
    else:
        df['DisplayDate'] = df.index.strftime('%Y/%m/%d')

    # VWAP
    try:
        v = df['Volume']
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    except:
        df['VWAP'] = np.nan

    # ADX / MACD / RSI / SuperTrend
    # (計算ロジックはV25.0を維持)
    high = df['High']; low = df['Low']; close = df['Close']
    
    # ATR & ADX
    tr1 = high - low; tr2 = (high - close.shift()).abs(); tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    up = high.diff(); down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    df['ADX'] = dx.ewm(alpha=1/14).mean()

    # MACD
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))

    # SuperTrend (損切りラインに使用)
    period_st = 10
    multiplier = 3.0
    atr_st = tr.rolling(period_st).mean()
    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr_st)
    basic_lower = hl2 - (multiplier * atr_st)
    
    final_upper = pd.Series([np.nan]*len(df), index=df.index)
    final_lower = pd.Series([np.nan]*len(df), index=df.index)
    supertrend = pd.Series([np.nan]*len(df), index=df.index)
    trend_dir = pd.Series([True]*len(df), index=df.index) # True=UP, False=DOWN

    # ループ処理の最適化は難しいが、視認性のためこのままとする
    # (実運用上、データ点数が数千でなければ問題ない)
    for i in range(period_st, len(df)):
        prev_close = close.iloc[i-1]
        
        if pd.isna(final_upper.iloc[i-1]) or pd.isna(final_lower.iloc[i-1]):
            final_upper.iloc[i] = basic_upper.iloc[i]
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_upper.iloc[i] = basic_upper.iloc[i] if basic_upper.iloc[i] < final_upper.iloc[i-1] or prev_close > final_upper.iloc[i-1] else final_upper.iloc[i-1]
            final_lower.iloc[i] = basic_lower.iloc[i] if basic_lower.iloc[i] > final_lower.iloc[i-1] or prev_close < final_lower.iloc[i-1] else final_lower.iloc[i-1]

        prev_trend = trend_dir.iloc[i-1]
        if prev_trend:
            trend_dir.iloc[i] = False if close.iloc[i] < final_lower.iloc[i] else True
        else:
            trend_dir.iloc[i] = True if close.iloc[i] > final_upper.iloc[i] else False
            
        supertrend.iloc[i] = final_lower.iloc[i] if trend_dir.iloc[i] else final_upper.iloc[i]

    df['SuperTrend'] = supertrend
    df['SuperTrend_Dir'] = trend_dir

    # --- シグナル判定 ---
    signals = [None] * len(df)
    is_daily = "d" in interval
    is_5m = "5m" in interval
    
    for i in range(30, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        sig = None
        
        is_uptrend = row['SuperTrend_Dir']
        is_downtrend = not row['SuperTrend_Dir']
        g_cross = prev['MACD'] < prev['Signal'] and row['MACD'] > row['Signal']
        d_cross = prev['MACD'] > prev['Signal'] and row['MACD'] < row['Signal']

        if is_daily: # SWING Logic
            if is_uptrend and g_cross: sig = 'SWING_BUY'
            elif is_downtrend and d_cross: sig = 'SWING_SELL'
            elif row['RSI'] < 30 and row['Close'] > prev['Close']: sig = 'SWING_BUY (RSI)'
            elif row['RSI'] > 70 and row['Close'] < prev['Close']: sig = 'SWING_SELL (RSI)'
            elif is_uptrend and not df.iloc[i-1]['SuperTrend_Dir']: sig = 'SWING_BUY (Trend)'
            elif is_downtrend and df.iloc[i-1]['SuperTrend_Dir']: sig = 'SWING_SELL (Trend)'
            
        elif is_5m: # DAY-STD Logic
            if is_uptrend and g_cross: sig = 'DAY_BUY'
            elif is_downtrend and d_cross: sig = 'DAY_SELL'
            
        else: # DAY-FAST (Scalp) Logic
            if pd.isna(row['ADX']) or pd.isna(row['VWAP']): continue
            adx_ok = row['ADX'] > 25
            buy_vwap_ok = row['Close'] > row['VWAP']
            sell_vwap_ok = row['Close'] < row['VWAP']
            
            if is_uptrend and g_cross and adx_ok and buy_vwap_ok: sig = 'SCALP_BUY'
            elif is_downtrend and d_cross and adx_ok and sell_vwap_ok: sig = 'SCALP_SELL'
            
        signals[i] = sig
        
    df['Trade_Signal'] = signals
    return df

# バックテスト関数
def run_backtest(df, tp_pct, trade_dir, shares):
    trades = []
    active_trade = None 
    do_long = "買い" in trade_dir or "両方" in trade_dir
    do_short = "売り" in trade_dir or "両方" in trade_dir
    
    max_drawdown = 0
    cum_profit = 0
    peak_profit = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        sig = row['Trade_Signal']
        st_val = row['SuperTrend']
        
        trade_closed = False
        profit = 0
        
        # 決済判定
        if active_trade:
            entry_price = active_trade['entry_price']
            entry_tp = active_trade['target_tp']
            
            if active_trade['type'] == 'buy':
                # 利確 or 損切り(SuperTrend割れ)
                if row['High'] >= entry_tp:
                    profit = (entry_tp - entry_price) * shares
                    trades.append({'date': row['DisplayDate'], 'type': 'Buy', 'res': 'WIN 🏆', 'entry': entry_price, 'exit': entry_tp, 'profit': profit})
                    active_trade = None; trade_closed = True
                elif row['Close'] < st_val:
                    profit = (st_val - entry_price) * shares
                    res_label = 'WIN (Trail)' if profit > 0 else 'LOSE 💀'
                    trades.append({'date': row['DisplayDate'], 'type': 'Buy', 'res': res_label, 'entry': entry_price, 'exit': st_val, 'profit': profit})
                    active_trade = None; trade_closed = True
                    
            elif active_trade['type'] == 'sell':
                if row['Low'] <= entry_tp:
                    profit = (entry_price - entry_tp) * shares
                    trades.append({'date': row['DisplayDate'], 'type': 'Sell', 'res': 'WIN 🏆', 'entry': entry_price, 'exit': entry_tp, 'profit': profit})
                    active_trade = None; trade_closed = True
                elif row['Close'] > st_val:
                    profit = (entry_price - st_val) * shares
                    res_label = 'WIN (Trail)' if profit > 0 else 'LOSE 💀'
                    trades.append({'date': row['DisplayDate'], 'type': 'Sell', 'res': res_label, 'entry': entry_price, 'exit': st_val, 'profit': profit})
                    active_trade = None; trade_closed = True
        
        if trade_closed:
            cum_profit += profit
            peak_profit = max(peak_profit, cum_profit)
            dd = peak_profit - cum_profit
            if dd > max_drawdown: max_drawdown = dd

        # エントリー判定
        if active_trade is None and not trade_closed and sig is not None:
            if do_long and "BUY" in sig:
                active_trade = {'entry_price': row['Close'], 'type': 'buy', 'target_tp': row['Close']*(1+tp_pct/100)}
            elif do_short and "SELL" in sig:
                active_trade = {'entry_price': row['Close'], 'type': 'sell', 'target_tp': row['Close']*(1-tp_pct/100)}
                
    return trades, max_drawdown

# シグナルスキャン & 通知ロジック
def scan_signals(tickers):
    history_buffer = []
    notified_set = global_state["notified_ids"]
    now_jst = pd.Timestamp.now(tz='Asia/Tokyo')
    today_str = now_jst.strftime("%Y-%m-%d")
    
    scan_bar = st.progress(0, text="市場監視中...")
    total = len(tickers)
    
    for idx, t in enumerate(tickers):
        scan_bar.progress((idx + 1) / total, text=f"Checking: {t} {get_name(t)}...")
        
        # 上位足トレンド判定
        daily_trend = "NEUTRAL"
        hourly_trend = "NEUTRAL"
        
        try:
            # 日足トレンド
            df_daily = get_data(t, "3mo", "1d")
            if df_daily is not None:
                df_daily = process_data(df_daily, "1d")
                daily_trend = "UP" if df_daily.iloc[-1]['SuperTrend_Dir'] else "DOWN"
            
            # 60分足トレンド
            df_60m = get_data(t, "1mo", "60m")
            if df_60m is not None:
                df_60m = process_data(df_60m, "60m")
                hourly_trend = "UP" if df_60m.iloc[-1]['SuperTrend_Dir'] else "DOWN"

            # ---------------------------
            # 1. SWING (日足)
            # ---------------------------
            if df_daily is not None:
                row = df_daily.iloc[-1]
                sig = row['Trade_Signal']
                
                # 確定足のみを対象にする場合、i=-2を見るべきだが、日足は「今日の終わり」まで待てないので形成中も見る
                # ここでは「形成中(今日)」を通知対象とする
                if sig:
                    date_val = row.name.strftime("%Y-%m-%d")
                    stop_loss_price = row['SuperTrend'] # 損切りライン
                    
                    sig_id = f"{date_val}_{t}_{sig}_SWING"
                    
                    # 通知条件: 今日発生 & 未通知
                    if date_val == today_str and sig_id not in notified_set:
                        sl_str = f"{stop_loss_price:,.0f}"
                        msg = f"**🌊 [SWING] {sig} 発生**\n銘柄: {get_name(t)} ({t})\n現在値: {row['Close']:,.0f}円\n🛑 **損切目安: {sl_str}円** (厳守)"
                        if send_discord_notify(msg):
                            notified_set.add(sig_id)
                    
                    history_buffer.append({
                        "dt": row.name, "time_str": date_val, "code": t, "name": get_name(t),
                        "sig": sig, "price": row['Close'], "sl": stop_loss_price,
                        "type": "SWING", "ago_label": "今日"
                    })

            # ---------------------------
            # 2. DAY-STD (5分足)
            # ---------------------------
            df_5m = get_data(t, "5d", "5m")
            if df_5m is not None:
                df_5m = process_data(df_5m, "5m")
                # 最新確定足 (i=-2) を見る
                i = -2 # 直近の確定足
                if len(df_5m) > 2:
                    row = df_5m.iloc[i]
                    sig = row['Trade_Signal']
                    if sig:
                        # フィルタ: 60分足トレンドと一致時のみ
                        valid = False
                        if "BUY" in sig and hourly_trend == "UP": valid = True
                        if "SELL" in sig and hourly_trend == "DOWN": valid = True
                        
                        if valid:
                            time_str = row.name.strftime("%Y-%m-%d %H:%M")
                            stop_loss_price = row['SuperTrend']
                            
                            sig_id = f"{time_str}_{t}_{sig}_DAYSTD"
                            # 通知条件: 発生から20分以内 & 未通知
                            is_fresh = (now_jst - row.name) < timedelta(minutes=20)
                            
                            if is_fresh and sig_id not in notified_set:
                                emoji = "🟢" if "BUY" in sig else "🔴"
                                sl_str = f"{stop_loss_price:,.0f}"
                                msg = f"**{emoji} [DAY-STD] {sig} 確定**\n⏰ {time_str}\n銘柄: {get_name(t)}\n価格: {row['Close']:,.0f}円\n🛑 **損切目安: {sl_str}円**"
                                if send_discord_notify(msg):
                                    notified_set.add(sig_id)
                            elif not is_fresh:
                                notified_set.add(sig_id) # 古いのは通知済みにする

                            history_buffer.append({
                                "dt": row.name, "time_str": time_str, "code": t, "name": get_name(t),
                                "sig": sig, "price": row['Close'], "sl": stop_loss_price,
                                "type": "DAY-STD", "ago_label": "直近"
                            })

            # ---------------------------
            # 3. DAY-FAST (1分足) - スキャル
            # ---------------------------
            df_1m = get_data(t, "3d", "1m")
            if df_1m is not None:
                df_1m = process_data(df_1m, "1m")
                i = -2
                if len(df_1m) > 2:
                    row = df_1m.iloc[i]
                    sig = row['Trade_Signal']
                    if sig:
                        # フィルタ: 日足トレンドと一致時のみ
                        valid = False
                        if "BUY" in sig and daily_trend == "UP": valid = True
                        if "SELL" in sig and daily_trend == "DOWN": valid = True
                        
                        if valid:
                            time_str = row.name.strftime("%Y-%m-%d %H:%M")
                            stop_loss_price = row['SuperTrend']
                            
                            sig_id = f"{time_str}_{t}_{sig}_DAYFAST"
                            is_fresh = (now_jst - row.name) < timedelta(minutes=5)
                            
                            if is_fresh and sig_id not in notified_set:
                                emoji = "🔥" if "BUY" in sig else "❄️"
                                sl_str = f"{stop_loss_price:,.0f}"
                                msg = f"**{emoji} [DAY-FAST] {sig} 確定**\n⏰ {time_str}\n銘柄: {get_name(t)}\n価格: {row['Close']:,.0f}円\n🛑 **損切: {sl_str}円**"
                                if send_discord_notify(msg):
                                    notified_set.add(sig_id)
                            elif not is_fresh:
                                notified_set.add(sig_id)

                            history_buffer.append({
                                "dt": row.name, "time_str": time_str, "code": t, "name": get_name(t),
                                "sig": sig, "price": row['Close'], "sl": stop_loss_price,
                                "type": "DAY-FAST", "ago_label": "直近"
                            })

        except Exception as e:
            # エラー時も止めずに次へ
            print(f"Error scanning {t}: {e}")
            continue

    scan_bar.empty()
    # 新しい順にソート
    history_buffer.sort(key=lambda x: x['dt'], reverse=True)
    return history_buffer

# カード表示用関数
def display_signal_cards(signal_list, use_cols=4):
    if not signal_list: return
    cols = st.columns(use_cols)
    for i, item in enumerate(signal_list):
        with cols[i % use_cols]:
            is_buy = "BUY" in item['sig']
            color = "#ff5252" if is_buy else "#448aff" # 赤:買い、青:売り
            icon = "🔥" if is_buy else "🧊"
            
            src_type = item.get('type', 'Unknown')
            if "SWING" in src_type: badge_col = "#7E57C2" # 紫
            elif "STD" in src_type: badge_col = "#66BB6A" # 緑
            else: badge_col = "#FFA726" # オレンジ
            
            # 損切り価格の表示を追加
            sl_price = item.get('sl', 0)
            
            st.markdown(f"""
            <div style="border-left:5px solid {color}; padding:10px; border-radius:5px; margin-bottom:10px; background-color:#1e1e1e;">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span style="font-weight:bold; font-size:0.9em;">{item['name']}</span>
                    <span style="background-color:{badge_col}; padding:2px 6px; border-radius:3px; font-size:0.6em;">{src_type}</span>
                </div>
                <div style="font-size:0.8em; color:#aaa;">{item.get('time_str')}</div>
                <div style="color:{color}; font-weight:bold; font-size:1.1em; margin: 5px 0;">{icon} {item['sig']}</div>
                <div style="display:flex; justify-content:space-between; align-items:end;">
                    <div style="font-size:1.0em;">{item['price']:,.0f} <span style="font-size:0.7em">円</span></div>
                    <div style="font-size:0.8em; color:#FF5252;">損切: {sl_price:,.0f}</div>
                </div>
            </div>""", unsafe_allow_html=True)

# ==========================================
# Main Layout
# ==========================================
st.sidebar.title("💎 Watcher V25.1")
mode = st.sidebar.radio("モード選択", ["🤖 監視モード (通知用)", "📈 分析モード (人間用)"])

# ----------------------------
# MODE 1: 🤖 監視モード
# ----------------------------
if "監視" in mode:
    st.markdown("## 🤖 自動監視中 (20秒更新)")
    st.info("このタブを開いたままにしてください。Discordへ通知を送信します。")
    st_autorefresh(interval=20*1000, key="monitor_refresh")
    
    # 監視実行
    with st.spinner("市場データをスキャン中..."):
        all_history = scan_signals(st.session_state.favorites)
    
    st.markdown("### 📡 検出されたシグナル (直近24時間)")
    if all_history:
        # 直近のものを表示
        recent_list = [h for h in all_history if (pd.Timestamp.now(tz='Asia/Tokyo') - h['dt']) < timedelta(days=1)]
        display_signal_cards(recent_list)
    else:
        st.write("現在、条件に合致するシグナルはありません。")

# ----------------------------
# MODE 2: 📈 分析モード
# ----------------------------
else:
    st.markdown("## 📈 詳細分析 & 検証")
    
    # サイドバー操作
    if st.session_state.current_ticker in st.session_state.favorites:
        fav_idx = st.session_state.favorites.index(st.session_state.current_ticker)
    else:
        fav_idx = 0
        
    def update_ticker():
        st.session_state.current_ticker = st.session_state.ticker_radio

    st.sidebar.radio("銘柄選択", st.session_state.favorites, index=fav_idx, key="ticker_radio", on_change=update_ticker, format_func=lambda x: f"{x} {get_name(x)}")

    with st.sidebar.expander("銘柄リスト編集"):
        new_code = st.text_input("コード追加 (例: 9984)")
        if st.button("追加"): 
            if new_code and new_code not in st.session_state.favorites:
                st.session_state.favorites.append(new_code)
                st.rerun()
        
        del_targets = st.multiselect("削除", st.session_state.favorites, format_func=lambda x: f"{x} {get_name(x)}")
        if st.button("削除実行"):
            for t in del_targets: 
                if t in st.session_state.favorites: st.session_state.favorites.remove(t)
            if st.session_state.current_ticker not in st.session_state.favorites and st.session_state.favorites:
                st.session_state.current_ticker = st.session_state.favorites[0]
            st.rerun()

    # チャート設定
    c1, c2 = st.sidebar.columns(2)
    period = c1.selectbox("期間", ["1d", "5d", "1mo", "3mo", "6mo"], index=2)
    interval = c2.selectbox("時間足", ["1m", "5m", "15m", "1h", "1d"], index=4)

    # バックテスト設定
    st.sidebar.divider()
    st.sidebar.subheader("💰 バックテスト設定")
    tp_pct = st.sidebar.number_input("利確目標 (%)", 0.1, 50.0, 2.0, 0.1) 
    trade_shares = st.sidebar.number_input("取引株数", 100, 10000, 100, 100)
    trade_dir = st.sidebar.radio("売買方向", ["買い", "売り", "両方"], horizontal=True)

    # --- チャート描画 ---
    t = st.session_state.current_ticker
    t_name = get_name(t)
    st.header(f"{t} {t_name}")

    with st.spinner("チャートデータ取得中..."):
        # 1m, 5mの場合は期間を短く自動調整してロード時間を短縮
        req_period = period
        if interval == "1m" and period not in ["1d", "5d"]: req_period = "5d"
        if interval == "5m" and period not in ["1d", "5d", "1mo"]: req_period = "1mo"
        
        df_chart = get_data(t, req_period, interval)

    if df_chart is not None:
        df_chart = process_data(df_chart, interval)
        last = df_chart.iloc[-1]
        
        # 指標パネル
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("現在値", f"{last['Close']:,.0f}")
        col2.metric("SuperTrend", f"{last['SuperTrend']:,.0f}", delta="上昇トレンド" if last['SuperTrend_Dir'] else "下降トレンド", delta_color="normal" if last['SuperTrend_Dir'] else "inverse")
        
        rsi_val = last['RSI']
        col3.metric("RSI (14)", f"{rsi_val:.1f}", delta="買われすぎ" if rsi_val>70 else ("売られすぎ" if rsi_val<30 else "中立"), delta_color="off")
        
        adx_val = last['ADX']
        col4.metric("ADX (14)", f"{adx_val:.1f}", delta="トレンド発生中" if adx_val>25 else "レンジ", delta_color="normal" if adx_val>25 else "off")

        # Plotly チャート
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
        
        # 1. メインチャート
        fig.add_trace(go.Candlestick(x=df_chart['DisplayDate'], open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart['DisplayDate'], y=df_chart['VWAP'], mode='lines', line=dict(color='purple', width=1), name='VWAP'), row=1, col=1)
        
        # SuperTrendの描画 (色分け)
        st_green = df_chart['SuperTrend'].copy()
        st_green[~df_chart['SuperTrend_Dir']] = np.nan
        st_red = df_chart['SuperTrend'].copy()
        st_red[df_chart['SuperTrend_Dir']] = np.nan
        
        fig.add_trace(go.Scatter(x=df_chart['DisplayDate'], y=st_green, mode='lines', line=dict(color='green', width=1), name='Support'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart['DisplayDate'], y=st_red, mode='lines', line=dict(color='red', width=1), name='Resist'), row=1, col=1)

        # シグナルプロット
        buy_sig = df_chart[df_chart['Trade_Signal'].str.contains('BUY', na=False)]
        sell_sig = df_chart[df_chart['Trade_Signal'].str.contains('SELL', na=False)]
        
        if not buy_sig.empty:
            fig.add_trace(go.Scatter(x=buy_sig['DisplayDate'], y=buy_sig['Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', size=12, color='red'), name='BUY Signal'), row=1, col=1)
        if not sell_sig.empty:
            fig.add_trace(go.Scatter(x=sell_sig['DisplayDate'], y=sell_sig['High']*1.01, mode='markers', marker=dict(symbol='triangle-down', size=12, color='blue'), name='SELL Signal'), row=1, col=1)

        # 2. MACD
        fig.add_trace(go.Bar(x=df_chart['DisplayDate'], y=df_chart['MACD']-df_chart['Signal'], name='MACD Hist', marker_color='gray'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_chart['DisplayDate'], y=df_chart['MACD'], name='MACD', line=dict(color='orange')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_chart['DisplayDate'], y=df_chart['Signal'], name='Signal', line=dict(color='blue')), row=2, col=1)

        # 3. ADX & RSI
        fig.add_trace(go.Scatter(x=df_chart['DisplayDate'], y=df_chart['ADX'], name='ADX', line=dict(color='yellow')), row=3, col=1)
        fig.add_hline(y=25, line_dash="dot", line_color="white", row=3, col=1, annotation_text="Trend Line (25)")

        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # バックテスト実行ボタンエリア
        st.markdown("### 🧪 戦略バックテスト")
        st.caption("表示中のチャート期間・時間足に対してシミュレーションを行います。")
        
        if st.button("バックテスト実行", type="primary"):
            # 計算用データ（少し長めに取るなど調整可能だが今回は表示データを使用）
            bt_df = df_chart.copy()
            trades, max_dd = run_backtest(bt_df, tp_pct, trade_dir, trade_shares)
            st.session_state.bt_results = {"trades": trades, "max_dd": max_dd}
        
        # 結果表示
        if st.session_state.bt_results is not None:
            res = st.session_state.bt_results
            trades = res["trades"]
            max_dd = res["max_dd"]
            
            if len(trades) > 0:
                df_res = pd.DataFrame(trades)
                wins = df_res[df_res['res'].str.contains('WIN')]
                win_rate = (len(wins) / len(trades)) * 100
                total_pl = df_res['profit'].sum()
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("総取引数", f"{len(trades)}回")
                m2.metric("勝率", f"{win_rate:.1f}%")
                m3.metric("純損益", f"{total_pl:,.0f}円", delta=total_pl)
                m4.metric("最大ドローダウン", f"-{max_dd:,.0f}円", delta=-max_dd, delta_color="inverse")
                
                st.dataframe(df_res[['date', 'type', 'res', 'entry', 'exit', 'profit']], use_container_width=True)
            else:
                st.warning("この期間・条件ではシグナルが発生しませんでした。")

    else:
        st.error("データ取得に失敗しました。銘柄コードまたは期間を確認してください。")