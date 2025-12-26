import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# --- 監視対象リスト ---
TICKER_NAMES = {
    "9984": "ソフトバンクG", "6857": "アドバンテスト", "5803": "フジクラ",
    "6920": "レーザーテック", "3563": "F&L (スシロー)", "8385": "伊予銀行",
    "5020": "ENEOS", "8136": "サンリオ", "3778": "さくらネット",
    "9107": "川崎汽船", "7011": "三菱重工", "8035": "東エレク",
    "8306": "三菱UFJ", "7203": "トヨタ自動車", "4755": "楽天G",
    "7974": "任天堂", "6501": "日立製作所", "6758": "ソニーG",
    "6098": "リクルート", "4502": "武田薬品", "9432": "NTT",
    "8058": "三菱商事", "8001": "伊藤忠", "3382": "7&iHD"
}
WATCH_LIST = list(TICKER_NAMES.keys())

def get_name(code):
    return TICKER_NAMES.get(code, code)

# --- データ取得 (時刻修正強化版) ---
def get_data(ticker, period, interval):
    try:
        ticker_mod = f"{ticker}.T" if ".T" not in ticker and ticker.isdigit() else ticker
        if ticker in ["^N225", "NIY=F", "^TOPX"]: ticker_mod = ticker
        
        df = yf.download(ticker_mod, period=period, interval=interval, progress=False, auto_adjust=False)
        if df.empty: return None

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # ★ タイムゾーン強制補正
        # yfinanceの短期足はタイムゾーンなしのUTCで来ることがあるため、JSTに変換
        if df.index.tz is None:
            # 短期足(日足以外)ならUTCとみなしてJST(+9h)へ変換
            if "m" in interval or "h" in interval: 
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Tokyo')
            else:
                # 日足などはローカル日付なのでそのままでよいが念の為JST化
                pass
        else:
            # 既にタイムゾーン情報があればJSTへ変換
            df.index = df.index.tz_convert('Asia/Tokyo')

        # 扱いやすくするためタイムゾーン情報を削除（NaiveなJST時刻にする）
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        return df
    except:
        return None

# --- テクニカル計算 ---
def calculate_technical_indicators(df):
    df = df.copy()
    try:
        v = df['Volume'].squeeze()
        tp = ((df['High'] + df['Low'] + df['Close']) / 3).squeeze()
        df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    except: df['VWAP'] = np.nan

    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    # ADX
    up, down = high.diff(), low.diff()
    p_dm = np.where((up > down) & (up > 0), up, 0)
    m_dm = np.where((down > up) & (down > 0), down, 0)
    p_di = 100 * (pd.Series(p_dm, index=df.index).ewm(alpha=1/14).mean() / atr)
    m_di = 100 * (pd.Series(m_dm, index=df.index).ewm(alpha=1/14).mean() / atr)
    df['ADX'] = (abs(p_di - m_di) / abs(p_di + m_di) * 100).ewm(alpha=1/14).mean()

    # MACD & RSI
    df['MACD'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    delta = close.diff()
    df['RSI'] = 100 - (100 / (1 + delta.where(delta>0,0).ewm(alpha=1/14).mean() / -delta.where(delta<0,0).ewm(alpha=1/14).mean()))

    # SuperTrend
    atr_st = tr.rolling(10).mean()
    hl2 = (high + low) / 2
    b_up = hl2 + 3 * atr_st; b_lo = hl2 - 3 * atr_st
    st_line = [np.nan]*len(df); st_dir = [True]*len(df)
    f_up = [np.nan]*len(df); f_lo = [np.nan]*len(df)

    for i in range(len(df)):
        if i < 10:
            f_up[i], f_lo[i] = b_up.iloc[i], b_lo.iloc[i]
            continue
        prev_c = close.iloc[i-1]
        f_up[i] = b_up.iloc[i] if b_up.iloc[i] < f_up[i-1] or prev_c > f_up[i-1] else f_up[i-1]
        f_lo[i] = b_lo.iloc[i] if b_lo.iloc[i] > f_lo[i-1] or prev_c < f_lo[i-1] else f_lo[i-1]
        
        if st_dir[i-1]: st_dir[i] = False if close.iloc[i] < f_lo[i] else True
        else: st_dir[i] = True if close.iloc[i] > f_up[i] else False
        st_line[i] = f_lo[i] if st_dir[i] else f_up[i]

    df['SuperTrend'] = st_line; df['SuperTrend_Dir'] = st_dir

    # シグナル判定
    signals = []
    is_daily = (df.index[1] - df.index[0]) >= timedelta(hours=20) if len(df)>1 else True

    for i in range(len(df)):
        if i < 30: signals.append(None); continue
        row = df.iloc[i]; prev = df.iloc[i-1]
        sig = None
        
        g_cross = prev['MACD'] < prev['Signal'] and row['MACD'] > row['Signal']
        d_cross = prev['MACD'] > prev['Signal'] and row['MACD'] < row['Signal']
        
        if is_daily:
            if row['SuperTrend_Dir'] and g_cross: sig = 'SWING_BUY'
            elif not row['SuperTrend_Dir'] and d_cross: sig = 'SWING_SELL'
            elif row['RSI'] < 30 and row['Close'] > prev['Close']: sig = 'SWING_BUY (RSI)'
            elif row['RSI'] > 70 and row['Close'] < prev['Close']: sig = 'SWING_SELL (RSI)'
        else:
            if pd.notna(row['ADX']):
                if row['SuperTrend_Dir'] and g_cross and row['ADX']>25: sig = 'DAY_BUY'
                elif not row['SuperTrend_Dir'] and d_cross and row['ADX']>25: sig = 'DAY_SELL'
        
        signals.append(sig)
    df['Trade_Signal'] = signals
    return df

# --- バックテスト ---
def run_backtest(df, tp_pct, trade_dir, shares):
    trades = []
    active = None
    do_long, do_short = "買い" in trade_dir, "売り" in trade_dir
    max_dd, peak, equity = 0, 0, 0

    for i in range(len(df)):
        row = df.iloc[i]; sig = row['Trade_Signal']
        closed = False; profit = 0
        
        if active:
            ep, tp = active['entry'], active['tp']
            if active['type'] == 'B':
                if row['High'] >= tp: profit, res = (tp-ep)*shares, "WIN 🏆"
                elif row['Close'] < row['SuperTrend']: profit, res = (row['SuperTrend']-ep)*shares, "Trail"
                else: res = None
            else: # Sell
                if row['Low'] <= tp: profit, res = (ep-tp)*shares, "WIN 🏆"
                elif row['Close'] > row['SuperTrend']: profit, res = (ep-row['SuperTrend'])*shares, "Trail"
                else: res = None
            
            if res:
                trades.append({'date':row.name, 'type':active['type'], 'res':res, 'profit':profit})
                active = None; closed = True
                equity += profit
                max_dd = max(max_dd, peak - equity)
                peak = max(peak, equity)

        if not active and not closed and sig:
            if do_long and "BUY" in sig: active = {'entry':row['Close'], 'type':'B', 'tp':row['Close']*(1+tp_pct/100)}
            elif do_short and "SELL" in sig: active = {'entry':row['Close'], 'type':'S', 'tp':row['Close']*(1-tp_pct/100)}

    return trades, max_dd

# --- 全銘柄スキャン実行 ---
def scan_market(notified_ids):
    results = []
    new_notified = notified_ids.copy()
    scan_limit_days = 5 
    
    # ★ ここでもJSTの現在時刻を確実に取得する
    now = pd.Timestamp.now(tz='Asia/Tokyo').tz_localize(None)

    for t in WATCH_LIST:
        t_name = get_name(t)
        
        # 1. 日足スキャン
        try:
            df = get_data(t, "3mo", "1d")
            if df is not None and not df.empty:
                df = calculate_technical_indicators(df)
                recent_df = df.iloc[-scan_limit_days:]
                
                for date, row in recent_df.iterrows():
                    sig = row['Trade_Signal']
                    if sig and "SWING" in sig:
                        sig_id = f"{date}_{t}_{sig}"
                        if sig_id not in new_notified:
                            results.append({
                                "time": date, "code": t, "name": t_name, "sig": sig,
                                "price": row['Close'], "sl": row['SuperTrend'], "rsi": row['RSI'], "note": "日足"
                            })
                            new_notified.add(sig_id)
        except: pass

        # 2. 60分足スキャン
        try:
            df = get_data(t, "1mo", "60m")
            if df is not None and not df.empty:
                df = calculate_technical_indicators(df)
                recent_df = df.iloc[-24:] 
                
                for date, row in recent_df.iterrows():
                    sig = row['Trade_Signal']
                    if sig:
                        sig_id = f"{date}_{t}_{sig}"
                        if sig_id not in new_notified:
                            results.append({
                                "time": date, "code": t, "name": t_name, "sig": sig,
                                "price": row['Close'], "sl": row['SuperTrend'], "rsi": row['RSI'], "note": "60分足"
                            })
                            new_notified.add(sig_id)
        except: pass
        
    results.sort(key=lambda x: x['time'], reverse=True)
    return results, new_notified