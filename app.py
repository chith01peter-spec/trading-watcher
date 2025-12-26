from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import numpy as np
import logic  # さっき作ったファイル

# --- シグナル表示専用の関数（変数名をあなたのコードに合わせました） ---
def display_signal_area(df_signals):
    """
    データフレームを受け取り、24時間以内はパネル、それ以前はリストで表示する
    """
    if df_signals.empty:
        st.info("現在シグナルはありません。")
        return

    now = datetime.now()
    threshold_24h = now - timedelta(hours=24)
    threshold_1week = now - timedelta(days=7)

    # データを分ける（'time'カラムを使用）
    df_recent = df_signals[df_signals['time'] >= threshold_24h]
    df_past = df_signals[(df_signals['time'] < threshold_24h) & (df_signals['time'] >= threshold_1week)]

    # --- A. 【24時間以内】パネル表示 ---
    st.subheader("🔔 直近24時間のシグナル")
    
    if not df_recent.empty:
        # 新しい順に並び替え
        df_recent = df_recent.sort_values('time', ascending=False)
        
        # 3列のカラムを作成してカード風に配置
        cols = st.columns(3) 
        for i, row in df_recent.iterrows():
            col = cols[i % 3]
            with col:
                with st.container(border=True):
                    # 銘柄名とコードを連結
                    st.markdown(f"**{row['code']} {row['name']}**")
                    # 日付時間を表示
                    st.caption(f"日時: {row['time'].strftime('%Y-%m-%d %H:%M')}")
                    # シグナル内容（'sig'カラムを使用）
                    st.error(f"{row['sig']}")
                    st.info(f"RSI: {row['rsi']:.1f}")
    else:
        st.info("直近24時間のシグナルはありません。")

    # --- B. 【1週間以内】箇条書きリスト ---
    st.subheader("📜 過去1週間の履歴")
    
    if not df_past.empty:
        # 新しい順に並び替え
        df_past = df_past.sort_values('time', ascending=False)

        for i, row in df_past.iterrows():
            date_str = row['time'].strftime('%Y-%m-%d %H:%M')
            stock_str = f"{row['code']} {row['name']}"
            
            st.markdown(
                f"・ {date_str} | **{stock_str}** | "
                f"シグナル: `{row['sig']}` (RSI: {row['rsi']:.1f})"
            )
    else:
        st.text("過去の履歴はありません。")

# --- ページ設定 ---
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
    
    # データがある場合のみ処理
    if st.session_state.monitor_results:
        # リストをDataFrameに変換
        df = pd.DataFrame(st.session_state.monitor_results)
        
        # 新しい表示関数を呼び出し（ここで表示切替！）
        display_signal_area(df)
        
    else:
        st.info("現在、監視対象のシグナルは検出されていません。")

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