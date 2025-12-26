import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import numpy as np
import logic  # さっき作ったファイル

# ページ設定
st.set_page_config(page_title="Trading Watcher V26.2 (Split)", layout="wide", page_icon="🦅")

# --- Session State ---
if 'monitor_results' not in st.session_state: st.session_state.monitor_results = []
if 'notified_ids' not in st.session_state: st.session_state.notified_ids = set()

# --- 自動更新ループ (20秒) ---
st_autorefresh(interval=20*1000, key="auto_update")

# --- 裏方：スキャン実行 ---
with st.spinner('🦅 全50銘柄 スキャン中...'):
    # logicファイルの関数を呼び出す
    results, new_notified = logic.scan_market(st.session_state.notified_ids)
    st.session_state.monitor_results = results
    st.session_state.notified_ids = new_notified

# --- サイドバー ---
st.sidebar.title("🦅 Watcher V26.2")
st.sidebar.caption("Split Architecture")
mode = st.sidebar.radio("モード", ["🦅 コックピット", "🔍 詳細分析"])

with st.sidebar.expander("🛡 ロット計算"):
    fund = st.number_input("余力", 100000, 100000000, 3000000, 100000)
    loss_pct = st.number_input("許容リスク%", 0.1, 5.0, 1.0)
    stop_yen = st.number_input("損切幅(円)", 0, 5000, 50)
    if stop_yen > 0:
        shares = (fund * loss_pct / 100) // stop_yen
        st.markdown(f"推奨: **{shares:,.0f} 株**")

# --- メイン画面 ---
if mode == "🦅 コックピット":
    st.markdown("### 🦅 Market Cockpit")
    if st.session_state.monitor_results:
        df = pd.DataFrame(st.session_state.monitor_results)
        df['時刻'] = df['time'].dt.strftime('%H:%M')
        df['銘柄'] = df.apply(lambda x: f"{x['name']} ({x['code']})", axis=1)
        df['価格'] = df['price'].apply(lambda x: f"{x:,.0f}")
        df['損切'] = df['sl'].apply(lambda x: f"{x:,.0f}")
        df['RSI'] = df['rsi'].apply(lambda x: f"{x:.0f}")
        
        def highlight(row):
            if 'BUY' in row['sig']: return ['background-color: #3d0000']*len(row)
            if 'SELL' in row['sig']: return ['background-color: #001a3d']*len(row)
            return ['']*len(row)
            
        st.dataframe(df[['時刻','銘柄','sig','価格','損切','RSI','note']].style.apply(highlight, axis=1), use_container_width=True, height=700)
    else:
        st.info("シグナルなし")

else: # 詳細分析
    st.markdown("### 🔍 詳細分析 & バックテスト")
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
            with st.spinner("取得中..."):
                df = yf.download(f"{target}.T", period=period, interval=interval, auto_adjust=False, progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                    if df.index.tz is None: df.index = df.index.tz_localize('Asia/Tokyo')
                    else: df.index = df.index.tz_convert('Asia/Tokyo')
                    
                    # logicファイルを使って計算
                    df = logic.calculate_technical_indicators(df)
                    
                    # チャート描画
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                    
                    # SuperTrend
                    sg = df['SuperTrend'].copy(); sg[~df['SuperTrend_Dir']] = np.nan
                    sr = df['SuperTrend'].copy(); sr[df['SuperTrend_Dir']] = np.nan
                    fig.add_trace(go.Scatter(x=df.index, y=sg, line=dict(color='green'), name='S'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=sr, line=dict(color='red'), name='R'), row=1, col=1)
                    
                    # Signals
                    b = df[df['Trade_Signal']=='BUY']; s = df[df['Trade_Signal']=='SELL']
                    if not b.empty: fig.add_trace(go.Scatter(x=b.index, y=b['Low'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='red')), row=1, col=1)
                    if not s.empty: fig.add_trace(go.Scatter(x=s.index, y=s['High'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='blue')), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='orange')), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='blue')), row=2, col=1)
                    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # バックテスト (logicファイルを使用)
                    trades, dd = logic.run_backtest(df, tp, ["買い","売り"], sh)
                    if trades:
                        pl = sum([t['profit'] for t in trades])
                        k1, k2, k3 = st.columns(3)
                        k1.metric("損益", f"{pl:,.0f}", delta=pl)
                        k2.metric("回数", len(trades))
                        k3.metric("最大DD", f"-{dd:,.0f}")
                        st.dataframe(pd.DataFrame(trades)[['date','type','res','profit']], use_container_width=True)