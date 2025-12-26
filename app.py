import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import os
import requests  # ← これが必要です
import logic

# --- 設定 ---
CSV_FILE = "signals_history.csv"

# --- Discord設定の読み込み (復活) ---
try:
    DISCORD_WEBHOOK_URL = st.secrets["DISCORD_URL"]
except:
    DISCORD_WEBHOOK_URL = ""

def send_discord_notify(msg):
    """Discordにメッセージを送信する"""
    if not DISCORD_WEBHOOK_URL: return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
    except:
        pass

# --- 履歴管理 & 通知 ---
def update_signal_history(current_results):
    if os.path.exists(CSV_FILE):
        try:
            df_history = pd.read_csv(CSV_FILE, parse_dates=['time'])
        except:
            df_history = pd.DataFrame(columns=['time', 'code', 'name', 'sig', 'price', 'sl', 'rsi', 'note'])
    else:
        df_history = pd.DataFrame(columns=['time', 'code', 'name', 'sig', 'price', 'sl', 'rsi', 'note'])

    if current_results:
        new_items = []
        now = datetime.now()
        
        # 新着チェック
        for item in current_results:
            is_duplicate = False
            
            # 1. 履歴CSV内の重複チェック (60分以内)
            if not df_history.empty:
                recent = df_history[df_history['time'] >= (now - timedelta(minutes=60))]
                matches = recent[(recent['code'].astype(str) == str(item['code'])) & (recent['sig'] == item['sig'])]
                if not matches.empty: is_duplicate = True
            
            # 2. まだ通知していない場合のみ処理
            if not is_duplicate:
                new_items.append(item)
                
                # ▼▼ ここで通知 (復活！) ▼▼
                msg = f"🦅 **{item['name']} ({item['code']})**\nシグナル: {item['sig']}\n価格: {item['price']:,.0f}円\nRSI: {item['rsi']:.1f}"
                send_discord_notify(msg)       # Discordへ
                st.toast(f"🔔 {item['name']}", icon="🦅") # 画面へ

        # 新しいデータがあれば保存
        if new_items:
            df_new = pd.DataFrame(new_items)
            df_history = pd.concat([df_history, df_new], ignore_index=True)
            df_history = df_history[df_history['time'] >= (now - timedelta(days=7))]
            df_history = df_history.sort_values('time', ascending=False)
            df_history.to_csv(CSV_FILE, index=False)

    return df_history

# --- コックピット表示 ---
def display_signal_area(df_signals):
    if df_signals is None or df_signals.empty:
        st.info("現在、履歴にあるシグナルはありません。スキャン中...")
        return

    now = datetime.now()
    threshold_24h = now - timedelta(hours=24)
    threshold_1week = now - timedelta(days=7)

    df_recent = df_signals[df_signals['time'] >= threshold_24h]
    df_past = df_signals[(df_signals['time'] < threshold_24h) & (df_signals['time'] >= threshold_1week)]

    st.subheader("🔔 直近24時間のシグナル")
    if not df_recent.empty:
        cols = st.columns(3) 
        for i, row in df_recent.iterrows():
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{row['code']} {row['name']}**")
                    st.caption(f"日時: {row['time'].strftime('%m/%d %H:%M')}")
                    st.error(f"{row['sig']}")
                    st.info(f"Price: {row['price']:,.0f} / RSI: {row['rsi']:.0f}")
    else:
        st.info("直近24時間のシグナルはありません。")

    st.subheader("📜 過去1週間の履歴")
    if not df_past.empty:
        for i, row in df_past.iterrows():
            d_str = row['time'].strftime('%m/%d %H:%M')
            st.markdown(f"・ {d_str} | **{row['code']} {row['name']}** | `{row['sig']}` (RSI: {row['rsi']:.0f})")
    else:
        st.text("過去の履歴はありません。")

# ==========================================
# メイン処理
# ==========================================
st.set_page_config(page_title="Trading Watcher V26.6", layout="wide", page_icon="🦅")
st_autorefresh(interval=60*1000, key="auto_update")

if 'notified_ids' not in st.session_state: st.session_state.notified_ids = set()

# --- 裏で全銘柄分析 ---
with st.spinner('🦅 全銘柄分析中...'):
    current_results, new_notified = logic.scan_market(st.session_state.notified_ids)
    st.session_state.notified_ids = new_notified
    df_history = update_signal_history(current_results)

# --- サイドバー ---
st.sidebar.title("🦅 Watcher V26.6")
# 通知テストボタン
if st.sidebar.button("🔔 通知テスト"):
    send_discord_notify("🔔 [TEST] 通信テストOKです！")
    st.sidebar.success("送信しました")

mode = st.sidebar.radio("モード", ["🦅 コックピット", "🔍 詳細分析"])

with st.sidebar.expander("🛡 ロット計算"):
    fund = st.number_input("余力", 100000, 100000000, 3000000, 100000)
    loss_pct = st.number_input("許容リスク%", 0.1, 5.0, 1.0)
    stop_yen = st.number_input("損切幅", 0, 5000, 50)
    if stop_yen > 0:
        shares = (fund * loss_pct / 100) // stop_yen
        st.markdown(f"推奨: **{shares:,.0f} 株**")

# --- メイン画面 ---
if mode == "🦅 コックピット":
    st.markdown("### 🦅 Market Cockpit (全銘柄監視中)")
    display_signal_area(df_history)

else: # 詳細分析モード
    st.markdown("### 📊 Market Indices")
    indices = {"日経平均": "^N225", "日経先物(CME)": "NIY=F", "TOPIX": "^TOPX", "USD/JPY": "JPY=X"}
    idx_cols = st.columns(len(indices))
    for i, (label, ticker) in enumerate(indices.items()):
        with idx_cols[i]:
            try:
                d = yf.Ticker(ticker).history(period="2d")
                if not d.empty:
                    last = d.iloc[-1]['Close']
                    prev = d.iloc[-2]['Close']
                    delta = last - prev
                    st.metric(label, f"{last:,.2f}", f"{delta:+.2f}")
                else: st.metric(label, "取得失敗", "-")
            except: st.metric(label, "Error", "-")
    
    st.divider()

    st.markdown("### 🔍 個別詳細分析")
    c1, c2 = st.columns([1, 3])
    with c1:
        target = st.selectbox("銘柄", logic.WATCH_LIST, format_func=lambda x: f"{x} {logic.get_name(x)}")
        period = st.selectbox("期間", ["1d","5d","1mo","3mo"], index=1)
        interval = st.selectbox("足", ["1m","5m","15m","60m"], index=1)
        st.divider()
        tp = st.number_input("利確%", 0.5, 20.0, 2.0)
        sh = st.number_input("株数", 100, 5000, 100)
        run_btn = st.button("分析実行", type="primary")
        
    with c2:
        if run_btn:
            with st.spinner(f"{target} 取得中..."):
                df = yf.download(f"{target}.T", period=period, interval=interval, auto_adjust=False, progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                    if df.index.tz is None: df.index = df.index.tz_localize('Asia/Tokyo')
                    else: df.index = df.index.tz_convert('Asia/Tokyo')
                    
                    df = logic.calculate_technical_indicators(df)
                    
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                    
                    sg = df['SuperTrend'].copy(); sg[~df['SuperTrend_Dir']] = np.nan
                    sr = df['SuperTrend'].copy(); sr[df['SuperTrend_Dir']] = np.nan
                    fig.add_trace(go.Scatter(x=df.index, y=sg, line=dict(color='green'), name='Sup'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=sr, line=dict(color='red'), name='Res'), row=1, col=1)
                    
                    b = df[df['Trade_Signal'].astype(str).str.contains('BUY', na=False)]
                    s = df[df['Trade_Signal'].astype(str).str.contains('SELL', na=False)]
                    if not b.empty: fig.add_trace(go.Scatter(x=b.index, y=b['Low'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='red'), name='BUY'), row=1, col=1)
                    if not s.empty: fig.add_trace(go.Scatter(x=s.index, y=s['High'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='blue'), name='SELL'), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='orange'), name='MACD'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='blue'), name='Sig'), row=2, col=1)
                    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    trades, dd = logic.run_backtest(df, tp, ["買い","売り"], sh)
                    if trades:
                        pl = sum([t['profit'] for t in trades])
                        k1, k2, k3 = st.columns(3)
                        k1.metric("損益", f"{pl:,.0f}", delta=pl)
                        k2.metric("回数", len(trades))
                        k3.metric("最大DD", f"-{dd:,.0f}")
                        st.dataframe(pd.DataFrame(trades)[['date','type','res','profit']], use_container_width=True)