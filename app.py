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
# ⚙️ 1. システム設定 & 銘柄リスト
# ==========================================
st.set_page_config(page_title="Trading Watcher V26.2", layout="wide", page_icon="🦅")

# Discord Webhook (Secretsエラー回避付き)
try:
    DISCORD_WEBHOOK_URL = st.secrets["DISCORD_URL"]
except:
    DISCORD_WEBHOOK_URL = ""

# --- 💎 監視対象: 宋スペシャル・パック (50銘柄) ---
WATCH_LIST = [
    # 主力・構成銘柄
    "9984", "6857", "5803", "6920", "3563", "8385", "5020", "8136", "3778", 
    "9107", "7011", "8035", "8306", "7203", 
    # 半導体・ハイテク
    "6146", "6526", "7735", "6723", "6758", "6367",
    # 金融・商社
    "8316", "8411", "8001", "8002", "8058", "7012", "7013",
    # グロース・AI・ゲーム
    "5253", "5032", "5574", "9166", "2160", "2413", "4385", "4483", "9613",
    # インバウンド・小売り・その他
    "9983", "7974", "4661", "3099", "3382", "8267", "9843", "9501", "7267", 
    "6501", "6701", "4502", "4568", "2914", "4911"
]
# 重複削除とソート
WATCH_LIST = sorted(list(set(WATCH_LIST)))

# 銘柄名マッピング（可読性用）
TICKER_MAP = {
    "9984": "SBG", "6857": "アドバン", "6920": "レーザー", "8306": "三菱UFJ", 
    "8035": "東エレク", "6146": "ディスコ", "6526": "ソシオ", "7735": "SCREEN",
    "5253": "カバー", "5032": "ANYCOLOR", "9166": "GENDA", "7011": "三菱重", 
    "5803": "フジクラ", "8001": "伊藤忠", "9107": "川崎船", "7203": "トヨタ",
    "8316": "三井住友", "8058": "三菱商", "4661": "OLC", "7974": "任天堂"
}

def get_name(code):
    """コードから銘柄名を取得。未定義ならコードを返す"""
    return TICKER_MAP.get(code, code)

# --- Session State 初期化 ---
if 'monitor_results' not in st.session_state:
    st.session_state.monitor_results = []
if 'notified_ids' not in st.session_state:
    st.session_state.notified_ids = set()
if 'bt_results' not in st.session_state:
    st.session_state.bt_results = None

# ==========================================
# 📊 2. データ処理 & テクニカル計算エンジン
# ==========================================

def calculate_technical_indicators(df):
    """
    テクニカル指標を計算してDataFrameに追加する中核関数
    """
    if df is None or df.empty:
        return None
    
    df = df.copy()
    
    # 1. VWAP
    try:
        v = df['Volume']
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    except:
        df['VWAP'] = np.nan

    # 2. MACD (12, 26, 9)
    close = df['Close']
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 3. RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))

    # 4. ADX (14) & ATR
    high = df['High']
    low = df['Low']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14).mean() / atr)
    
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    df['ADX'] = dx.ewm(alpha=1/14).mean()

    # 5. SuperTrend (Period=10, Multiplier=3)
    # ループ処理が必要なため計算コストが高いが、正確性を重視
    period = 10
    multiplier = 3.0
    atr_st = tr.rolling(period).mean()
    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr_st)
    basic_lower = hl2 - (multiplier * atr_st)
    
    # 配列初期化
    final_upper = [np.nan] * len(df)
    final_lower = [np.nan] * len(df)
    supertrend = [np.nan] * len(df)
    trend_dir = [True] * len(df) # True: UP, False: DOWN
    
    # 最初の確定足までスキップしつつ初期値を設定
    # (ここは高速化のため、単純なループで実装)
    prev_upper = basic_upper.iloc[0]
    prev_lower = basic_lower.iloc[0]
    prev_trend = True
    
    for i in range(len(df)):
        if np.isnan(basic_upper.iloc[i]): continue
        
        curr_close = close.iloc[i]
        prev_close = close.iloc[i-1] if i > 0 else curr_close
        
        # Upper Band Logic
        if basic_upper.iloc[i] < prev_upper or prev_close > prev_upper:
            curr_upper = basic_upper.iloc[i]
        else:
            curr_upper = prev_upper
            
        # Lower Band Logic
        if basic_lower.iloc[i] > prev_lower or prev_close < prev_lower:
            curr_lower = basic_lower.iloc[i]
        else:
            curr_lower = prev_lower
            
        # Trend Logic
        if prev_trend: # Currently Up
            if curr_close < curr_lower:
                curr_trend = False # Flip to Down
            else:
                curr_trend = True
        else: # Currently Down
            if curr_close > curr_upper:
                curr_trend = True # Flip to Up
            else:
                curr_trend = False
                
        # Final Value
        if curr_trend:
            st_val = curr_lower
        else:
            st_val = curr_upper
            
        final_upper[i] = curr_upper
        final_lower[i] = curr_lower
        supertrend[i] = st_val
        trend_dir[i] = curr_trend
        
        # 次のループのために保存
        prev_upper = curr_upper
        prev_lower = curr_lower
        prev_trend = curr_trend

    df['SuperTrend'] = supertrend
    df['SuperTrend_Dir'] = trend_dir
    
    # 6. 売買シグナル判定
    signals = []
    for i in range(len(df)):
        if i < 30:
            signals.append(None)
            continue
            
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        sig = None
        
        # MACDクロス
        gold_cross = prev_row['MACD'] < prev_row['Signal'] and row['MACD'] > row['Signal']
        dead_cross = prev_row['MACD'] > prev_row['Signal'] and row['MACD'] < row['Signal']
        
        # トレンドフィルター (SuperTrend)
        is_uptrend = row['SuperTrend_Dir']
        
        if is_uptrend and gold_cross:
            sig = "BUY"
        elif not is_uptrend and dead_cross:
            sig = "SELL"
            
        signals.append(sig)
        
    df['Trade_Signal'] = signals
    return df

# ==========================================
# 📡 3. 監視 & 通知エンジン (非同期風処理)
# ==========================================

def send_discord_notify(msg):
    """Discordへの通知送信（エラーハンドリング付き）"""
    if not DISCORD_WEBHOOK_URL:
        return False
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
        return True
    except Exception as e:
        print(f"Discord Error: {e}")
        return False

@st.cache_data(ttl=30)
def fetch_batch_data_50(tickers):
    """50銘柄を一括ダウンロードする関数"""
    try:
        tickers_t = [f"{t}.T" for t in tickers]
        # 監視用なので直近5日分あれば十分
        data = yf.download(tickers_t, period="5d", interval="5m", group_by='ticker', auto_adjust=False, progress=False, threads=True)
        return data
    except Exception as e:
        print(f"Batch Fetch Error: {e}")
        return None

def scan_market_batch():
    """
    全銘柄をスキャンし、結果リストを返す。
    UI表示用のデータ作成と、Discord通知を同時に行う。
    """
    batch_data = fetch_batch_data_50(WATCH_LIST)
    if batch_data is None:
        return []
        
    results = []
    now_jst = pd.Timestamp.now(tz='Asia/Tokyo')
    
    for code in WATCH_LIST:
        try:
            ticker_key = f"{code}.T"
            
            # データの抽出と整形 (yfinanceの構造変化に強い書き方)
            try:
                # 必要なカラムが存在するか確認
                if ticker_key not in batch_data['Close'].columns:
                    continue
                    
                df_t = pd.DataFrame({
                    'Open': batch_data['Open'][ticker_key],
                    'High': batch_data['High'][ticker_key],
                    'Low': batch_data['Low'][ticker_key],
                    'Close': batch_data['Close'][ticker_key],
                    'Volume': batch_data['Volume'][ticker_key]
                })
                # 全てNaNならスキップ
                if df_t['Close'].isna().all():
                    continue
                
                df_t = df_t.dropna()
                
            except KeyError:
                continue

            # タイムゾーン設定
            if df_t.index.tz is None:
                df_t.index = df_t.index.tz_localize('Asia/Tokyo')
            else:
                df_t.index = df_t.index.tz_convert('Asia/Tokyo')

            # テクニカル計算実行
            df_calc = calculate_technical_indicators(df_t)
            if df_calc is None: continue
            
            # 最新の足を取得
            row = df_calc.iloc[-1]
            sig = row['Trade_Signal']
            
            # シグナルがある場合、結果に追加
            if sig:
                # 注記（Note）の作成
                notes = []
                if row['RSI'] > 75: notes.append("⚠️RSI過熱")
                elif row['RSI'] < 25: notes.append("⚠️RSI底")
                if row['ADX'] < 20: notes.append("📉レンジ気味")
                
                note_str = " ".join(notes)
                
                results.append({
                    "code": code,
                    "name": get_name(code),
                    "time": row.name,
                    "sig": sig,
                    "price": row['Close'],
                    "rsi": row['RSI'],
                    "adx": row['ADX'],
                    "sl": row['SuperTrend'], # 損切りライン
                    "note": note_str
                })
                
                # --- 通知ロジック ---
                sig_id = f"{row.name}_{code}_{sig}"
                # 条件: 直近30分以内のシグナル かつ 未通知
                is_recent = (now_jst - row.name) < timedelta(minutes=30)
                
                if is_recent and sig_id not in st.session_state.notified_ids:
                    emoji = "🚀" if "BUY" in sig else "🥀"
                    sl_fmt = f"{row['SuperTrend']:,.0f}"
                    
                    msg = (f"**{emoji} {get_name(code)} ({code}) {sig}**\n"
                           f"現在値: {row['Close']:,.0f}円\n"
                           f"RSI: {row['RSI']:.0f} | ADX: {row['ADX']:.0f}\n"
                           f"🛑 損切目安: {sl_fmt}円\n"
                           f"{note_str}")
                    
                    if send_discord_notify(msg):
                        st.session_state.notified_ids.add(sig_id)

        except Exception as e:
            # 1銘柄のエラーで全体を止めない
            continue
            
    # 新しい順にソートして返す
    results.sort(key=lambda x: x['time'], reverse=True)
    return results

# ==========================================
# 🧪 4. バックテスト・エンジン
# ==========================================

def run_backtest_engine(df, tp_pct, trade_dir, shares):
    """
    指定されたDataFrameに対してバックテストを実行する
    """
    trades = []
    active_trade = None
    
    do_long = "買い" in trade_dir or "両方" in trade_dir
    do_short = "売り" in trade_dir or "両方" in trade_dir
    
    max_dd = 0
    peak_profit = 0
    cum_profit = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        sig = row['Trade_Signal']
        st_val = row['SuperTrend']
        
        # 決済処理
        if active_trade:
            entry_price = active_trade['price']
            tp_price = active_trade['tp']
            pnl = 0
            closed = False
            res_type = ""
            
            if active_trade['type'] == 'buy':
                # TP到達
                if row['High'] >= tp_price:
                    pnl = (tp_price - entry_price) * shares
                    closed = True
                    res_type = "WIN 🏆"
                # トレーリングストップ (SuperTrend割れ)
                elif row['Close'] < st_val:
                    pnl = (st_val - entry_price) * shares
                    closed = True
                    res_type = "WIN (Trail)" if pnl > 0 else "LOSE 💀"
                    
            elif active_trade['type'] == 'sell':
                # TP到達
                if row['Low'] <= tp_price:
                    pnl = (entry_price - tp_price) * shares
                    closed = True
                    res_type = "WIN 🏆"
                # トレーリングストップ (SuperTrend超え)
                elif row['Close'] > st_val:
                    pnl = (entry_price - st_val) * shares
                    closed = True
                    res_type = "WIN (Trail)" if pnl > 0 else "LOSE 💀"
            
            if closed:
                trades.append({
                    'date': row.name,
                    'type': active_trade['type'],
                    'res': res_type,
                    'profit': pnl,
                    'entry': entry_price,
                    'exit': tp_price if "WIN 🏆" in res_type else st_val
                })
                active_trade = None
                
                # ドローダウン計算
                cum_profit += pnl
                peak_profit = max(peak_profit, cum_profit)
                dd = peak_profit - cum_profit
                max_dd = max(max_dd, dd)

        # 新規エントリー処理
        if active_trade is None and sig:
            if do_long and "BUY" in sig:
                tp = row['Close'] * (1 + tp_pct / 100)
                active_trade = {'type': 'buy', 'price': row['Close'], 'tp': tp}
            elif do_short and "SELL" in sig:
                tp = row['Close'] * (1 - tp_pct / 100)
                active_trade = {'type': 'sell', 'price': row['Close'], 'tp': tp}
                
    return trades, max_dd

# ==========================================
# 🖥️ 5. メイン UI 構築
# ==========================================

# 自動更新 (20秒)
st_autorefresh(interval=20*1000, key="auto_update")

# --- バックグラウンド処理 ---
# 画面がリロードされるたびに、UI描画前に最新データをスキャンする
with st.spinner('🦅 全50銘柄 市場スキャン中... (Batch Processing)'):
    current_results = scan_market_batch()
    st.session_state.monitor_results = current_results

# --- サイドバー設定 ---
st.sidebar.title("🦅 Watcher V26.2")
st.sidebar.caption("Robust Edition")

mode = st.sidebar.radio("モード選択", ["🦅 コックピット (全体監視)", "🔍 詳細分析 (個別検証)"])

# 資金管理ツール (V24からの復活)
with st.sidebar.expander("🛡 ロット計算機", expanded=False):
    fund = st.number_input("余力 (円)", 100000, 100000000, 3000000, step=100000)
    risk_pct = st.number_input("許容リスク (%)", 0.1, 5.0, 1.0, 0.1)
    max_loss = fund * (risk_pct / 100)
    st.caption(f"1トレード許容損失: {max_loss:,.0f}円")
    
    stop_range = st.number_input("想定損切幅 (円)", 0, 5000, 50)
    if stop_range > 0:
        rec_shares = max_loss // stop_range
        st.markdown(f"推奨株数: **{rec_shares:,.0f} 株**")

# --- メインコンテンツ ---

# 【モード1: コックピット】
if mode == "🦅 コックピット (全体監視)":
    st.markdown("### 🦅 Market Cockpit (Real-time)")
    st.markdown(f"**監視対象:** {len(WATCH_LIST)}銘柄 | **最終更新:** {datetime.now().strftime('%H:%M:%S')}")
    
    if st.session_state.monitor_results:
        # 表示用DataFrame作成
        df_res = pd.DataFrame(st.session_state.monitor_results)
        
        # 整形
        df_res['時刻'] = df_res['time'].dt.strftime('%H:%M')
        df_res['銘柄'] = df_res.apply(lambda x: f"{x['name']} ({x['code']})", axis=1)
        df_res['価格'] = df_res['price'].apply(lambda x: f"{x:,.0f}")
        df_res['損切目安'] = df_res['sl'].apply(lambda x: f"{x:,.0f}")
        df_res['RSI'] = df_res['rsi'].apply(lambda x: f"{x:.0f}")
        
        # 色付けスタイリング
        def style_rows(row):
            if 'BUY' in row['sig']:
                return ['background-color: #3d0000; color: #ffcccc'] * len(row)
            if 'SELL' in row['sig']:
                return ['background-color: #001a3d; color: #ccffff'] * len(row)
            return [''] * len(row)

        # カラム選択して表示
        cols_to_show = ['時刻', '銘柄', 'sig', '価格', '損切目安', 'RSI', 'note']
        st.dataframe(
            df_res[cols_to_show].style.apply(style_rows, axis=1),
            use_container_width=True,
            height=700
        )
    else:
        st.info("現在、シグナル発生中の銘柄はありません。監視を継続します...")

# 【モード2: 詳細分析】
else:
    st.markdown("### 🔍 詳細分析 & バックテスト")
    
    # レイアウト分割
    col_ui, col_chart = st.columns([1, 3])
    
    with col_ui:
        st.subheader("設定")
        target_ticker = st.selectbox("分析銘柄", WATCH_LIST, format_func=lambda x: f"{x} {get_name(x)}")
        
        period = st.selectbox("期間", ["1d", "5d", "1mo", "3mo"], index=1)
        interval = st.selectbox("時間足", ["1m", "5m", "15m", "60m", "1d"], index=1)
        
        st.divider()
        st.subheader("シミュレーション")
        tp_pct = st.number_input("利確目標 (%)", 0.5, 20.0, 2.0, 0.5)
        shares = st.number_input("取引株数", 100, 10000, 100, 100)
        
        if st.button("分析実行", type="primary"):
            st.session_state.do_analysis = True
    
    with col_chart:
        if getattr(st.session_state, 'do_analysis', False):
            with st.spinner(f"{target_ticker} の詳細データを取得・計算中..."):
                # 1. 個別データ取得
                df_detail = yf.download(f"{target_ticker}.T", period=period, interval=interval, auto_adjust=False, progress=False)
                
                if df_detail is not None and not df_detail.empty:
                    # MultiIndex解除
                    if isinstance(df_detail.columns, pd.MultiIndex):
                        df_detail.columns = df_detail.columns.get_level_values(0)
                    
                    # タイムゾーン
                    if df_detail.index.tz is None:
                        df_detail.index = df_detail.index.tz_localize('Asia/Tokyo')
                    else:
                        df_detail.index = df_detail.index.tz_convert('Asia/Tokyo')
                        
                    # 2. テクニカル計算（共通関数を使用）
                    df_detail = calculate_technical_indicators(df_detail)
                    
                    # 3. チャート描画
                    last_row = df_detail.iloc[-1]
                    
                    # メトリクス表示
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("現在値", f"{last_row['Close']:,.0f}")
                    m2.metric("RSI", f"{last_row['RSI']:.1f}")
                    m3.metric("ADX", f"{last_row['ADX']:.1f}")
                    st_status = "UP" if last_row['SuperTrend_Dir'] else "DOWN"
                    m4.metric("SuperTrend", st_status, delta_color="normal" if st_status=="UP" else "inverse")
                    
                    # Plotlyグラフ
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                    
                    # Main Chart
                    fig.add_trace(go.Candlestick(x=df_detail.index, open=df_detail['Open'], high=df_detail['High'], low=df_detail['Low'], close=df_detail['Close'], name='Price'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_detail.index, y=df_detail['VWAP'], line=dict(color='purple', width=1), name='VWAP'), row=1, col=1)
                    
                    # SuperTrend Lines
                    st_green = df_detail['SuperTrend'].copy()
                    st_green[~df_detail['SuperTrend_Dir']] = np.nan
                    st_red = df_detail['SuperTrend'].copy()
                    st_red[df_detail['SuperTrend_Dir']] = np.nan
                    
                    fig.add_trace(go.Scatter(x=df_detail.index, y=st_green, line=dict(color='green'), name='Support'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_detail.index, y=st_red, line=dict(color='red'), name='Resist'), row=1, col=1)
                    
                    # Signals
                    buys = df_detail[df_detail['Trade_Signal'] == 'BUY']
                    sells = df_detail[df_detail['Trade_Signal'] == 'SELL']
                    if not buys.empty:
                        fig.add_trace(go.Scatter(x=buys.index, y=buys['Low'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='red'), name='BUY'), row=1, col=1)
                    if not sells.empty:
                        fig.add_trace(go.Scatter(x=sells.index, y=sells['High'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='blue'), name='SELL'), row=1, col=1)

                    # Sub Chart (MACD)
                    fig.add_trace(go.Bar(x=df_detail.index, y=df_detail['MACD']-df_detail['Signal'], marker_color='gray', name='Hist'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df_detail.index, y=df_detail['MACD'], line=dict(color='orange'), name='MACD'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df_detail.index, y=df_detail['Signal'], line=dict(color='blue'), name='Signal'), row=2, col=1)
                    
                    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 4. バックテスト実行
                    st.markdown("#### 🧪 バックテスト結果")
                    trades, max_dd = run_backtest_engine(df_detail, tp_pct, ["買い", "売り"], shares)
                    
                    if trades:
                        df_trades = pd.DataFrame(trades)
                        total_profit = df_trades['profit'].sum()
                        win_count = len(df_trades[df_trades['res'].str.contains("WIN")])
                        win_rate = (win_count / len(trades)) * 100
                        
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("総損益", f"{total_profit:,.0f}円", delta=total_profit)
                        k2.metric("勝率", f"{win_rate:.1f}%")
                        k3.metric("取引回数", f"{len(trades)}回")
                        k4.metric("最大DD", f"-{max_dd:,.0f}円", delta=-max_dd, delta_color="inverse")
                        
                        st.dataframe(df_trades[['date', 'type', 'res', 'profit', 'entry', 'exit']], use_container_width=True)
                    else:
                        st.warning("この期間・条件ではシグナルが発生しませんでした。")
                        
                else:
                    st.error("データ取得に失敗しました。銘柄や通信状況を確認してください。")