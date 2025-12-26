import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# --- 銘柄リスト (監視対象) ---
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

# --- データ取得 ---
def get_data(ticker, period, interval):
    try:
        # 日本株の場合は .T をつける補正
        ticker_mod = f"{ticker}.T" if ".T" not in ticker and ticker.isdigit() else ticker
        
        # 指数の場合などの例外処理
        if ticker in ["^N225", "NIY=F", "^TOPX"]: ticker_mod = ticker

        df = yf.download(ticker_mod, period=period, interval=interval, progress=False, auto_adjust=False)
        
        if df.empty: return None

        # MultiIndexカラムの解除
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.loc[:, ~df.columns.duplicated()]
        
        # タイムゾーン処理
        if df.index.tz is None:
            df.index = df.index.tz_localize('Asia/Tokyo')
        else:
            df.index = df.index.tz_convert('Asia/Tokyo')
            
        return df
    except Exception as e:
        return None

# --- テクニカル計算 (V24.1ベース) ---
def calculate_technical_indicators(df):
    df = df.copy()
    
    # 1. 基礎データ整形
    try:
        v = df['Volume'].squeeze()
        tp = ((df['High'] + df['Low'] + df['Close']) / 3).squeeze()
        df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    except:
        df['VWAP'] = np.nan

    # 2. ATR & ADX
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    up = high.diff()
    down = low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    df['ADX'] = dx.ewm(alpha=1/14).mean()

    # 3. MACD & RSI
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))

    # 4. SuperTrend
    period_st = 10
    multiplier = 3.0
    atr_st = tr.rolling(period_st).mean()
    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr_st)
    basic_lower = hl2 - (multiplier * atr_st)
    
    supertrend = [np.nan] * len(df)
    trend_dir = [True] * len(df) # True: UP, False: DOWN
    final_upper = [np.nan] * len(df)
    final_lower = [np.nan] * len(df)

    for i in range(len(df)):
        if i < period_st:
            final_upper[i] = basic_upper.iloc[i]
            final_lower[i] = basic_lower.iloc[i]
            continue

        prev_close = close.iloc[i-1]
        
        # Upper Band Calculation
        if basic_upper.iloc[i] < final_upper[i-1] or prev_close > final_upper[i-1]:
            final_upper[i] = basic_upper.iloc[i]
        else:
            final_upper[i] = final_upper[i-1]
            
        # Lower Band Calculation
        if basic_lower.iloc[i] > final_lower[i-1] or prev_close < final_lower[i-1]:
            final_lower[i] = basic_lower.iloc[i]
        else:
            final_lower[i] = final_lower[i-1]

        # Trend Direction
        if trend_dir[i-1]:
            trend_dir[i] = False if close.iloc[i] < final_lower[i] else True
        else:
            trend_dir[i] = True if close.iloc[i] > final_upper[i] else False
            
        supertrend[i] = final_lower[i] if trend_dir[i] else final_upper[i]

    df['SuperTrend'] = supertrend
    df['SuperTrend_Dir'] = trend_dir
    
    # 5. シグナル判定ロジック (V24.1完全再現)
    signals = []
    # 時間足判定
    time_diff = df.index.to_series().diff().median()
    is_daily = time_diff >= timedelta(hours=20)
    is_5m = timedelta(minutes=4) <= time_diff <= timedelta(minutes=6)
    
    for i in range(len(df)):
        if i < 30:
            signals.append(None)
            continue
            
        row = df.iloc[i]
        prev = df.iloc[i-1]
        sig = None
        
        # MACDクロス
        g_cross = prev['MACD'] < prev['Signal'] and row['MACD'] > row['Signal']
        d_cross = prev['MACD'] > prev['Signal'] and row['MACD'] < row['Signal']
        is_uptrend = row['SuperTrend_Dir']
        
        if is_daily:
            # 日足ロジック (SWING)
            if is_uptrend and g_cross: sig = 'SWING_BUY'
            elif not is_uptrend and d_cross: sig = 'SWING_SELL'
            elif row['RSI'] < 30 and row['Close'] > prev['Close']: sig = 'SWING_BUY (RSI)'
            elif row['RSI'] > 70 and row['Close'] < prev['Close']: sig = 'SWING_SELL (RSI)'
            elif row['SuperTrend_Dir'] and not prev['SuperTrend_Dir']: sig = 'SWING_BUY (Trend)'
            elif not row['SuperTrend_Dir'] and prev['SuperTrend_Dir']: sig = 'SWING_SELL (Trend)'
            
        elif is_5m:
            # 5分足ロジック (DAY-STD)
            if is_uptrend and g_cross: sig = 'DAY_BUY'
            elif not is_uptrend and d_cross: sig = 'DAY_SELL'
            
        else:
            # 1分足など (SCALP / DAY-FAST)
            if pd.isna(row['ADX']) or pd.isna(row['VWAP']):
                signals.append(None)
                continue
            
            adx_ok = row['ADX'] > 25
            buy_vwap = row['Close'] > row['VWAP']
            sell_vwap = row['Close'] < row['VWAP']
            
            if is_uptrend and g_cross and adx_ok and buy_vwap: sig = 'SCALP_BUY'
            elif not is_uptrend and d_cross and adx_ok and sell_vwap: sig = 'SCALP_SELL'
            
        signals.append(sig)
        
    df['Trade_Signal'] = signals
    return df

# --- バックテスト関数 ---
def run_backtest(df, tp_pct, trade_dir, shares):
    trades = []
    active_trade = None 
    do_long = "買い" in trade_dir
    do_short = "売り" in trade_dir
    
    max_dd = 0; peak_equity = 0; equity = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        sig = row['Trade_Signal']
        st_val = row['SuperTrend']
        trade_closed = False; profit = 0
        
        # 決済
        if active_trade:
            entry_price = active_trade['entry_price']
            entry_tp = active_trade['target_tp']
            
            if active_trade['type'] == 'buy':
                if row['High'] >= entry_tp:
                    profit = (entry_tp - entry_price) * shares
                    trades.append({'date': row.name, 'type': 'Buy', 'res': 'WIN 🏆', 'profit': profit})
                    active_trade = None; trade_closed = True
                elif row['Close'] < st_val:
                    profit = (st_val - entry_price) * shares
                    trades.append({'date': row.name, 'type': 'Buy', 'res': 'Trail', 'profit': profit})
                    active_trade = None; trade_closed = True
            elif active_trade['type'] == 'sell':
                if row['Low'] <= entry_tp:
                    profit = (entry_price - entry_tp) * shares
                    trades.append({'date': row.name, 'type': 'Sell', 'res': 'WIN 🏆', 'profit': profit})
                    active_trade = None; trade_closed = True
                elif row['Close'] > st_val:
                    profit = (entry_price - st_val) * shares
                    trades.append({'date': row.name, 'type': 'Sell', 'res': 'Trail', 'profit': profit})
                    active_trade = None; trade_closed = True

        if trade_closed:
            equity += profit
            max_dd = max(max_dd, peak_equity - equity)
            peak_equity = max(peak_equity, equity)

        # エントリー
        if active_trade is None and not trade_closed and sig is not None:
            if do_long and "BUY" in sig:
                active_trade = {'entry_price': row['Close'], 'type': 'buy', 'target_tp': row['Close']*(1+tp_pct/100)}
            elif do_short and "SELL" in sig:
                active_trade = {'entry_price': row['Close'], 'type': 'sell', 'target_tp': row['Close']*(1-tp_pct/100)}
                
    return trades, max_dd

# --- 全銘柄スキャン実行 (裏方) ---
def scan_market(notified_ids):
    results = []
    new_notified = notified_ids.copy()
    now = datetime.now()
    
    # 簡易化のため、時間のかかるスキャンを効率的に行う
    # 全銘柄に対して「日足」と「5分足」をチェックする
    for t in WATCH_LIST:
        t_name = get_name(t)
        
        # --- 1. 日足チェック (SWING) ---
        try:
            df_d = get_data(t, "3mo", "1d")
            if df_d is not None and not df_d.empty:
                df_d = calculate_technical_indicators(df_d)
                # 最新の確定足を確認
                row = df_d.iloc[-1]
                sig = row['Trade_Signal']
                if sig and "SWING" in sig:
                    sig_id = f"{row.name}_{t}_{sig}"
                    if sig_id not in new_notified:
                        results.append({
                            "time": row.name, "code": t, "name": t_name, "sig": sig,
                            "price": row['Close'], "sl": row['SuperTrend'], "rsi": row['RSI'],
                            "note": "日足検知"
                        })
                        new_notified.add(sig_id)
        except: pass

        # --- 2. 5分足チェック (DAY) ---
        try:
            df_5m = get_data(t, "5d", "5m")
            if df_5m is not None and not df_5m.empty:
                df_5m = calculate_technical_indicators(df_5m)
                # 直近2本を見る（形成中含む）
                for i in [-1, -2]:
                    if abs(i) > len(df_5m): break
                    row = df_5m.iloc[i]
                    sig = row['Trade_Signal']
                    if sig:
                        # 6時間以内のシグナルのみ有効とする
                        if (now - row.name).total_seconds() < 3600 * 6:
                            sig_id = f"{row.name}_{t}_{sig}"
                            if sig_id not in new_notified:
                                results.append({
                                    "time": row.name, "code": t, "name": t_name, "sig": sig,
                                    "price": row['Close'], "sl": row['SuperTrend'], "rsi": row['RSI'],
                                    "note": "5分足検知"
                                })
                                new_notified.add(sig_id)
        except: pass

    return results, new_notified