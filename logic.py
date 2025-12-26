import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# --- 定数設定 ---
TICKER_NAMES = {
    "9984": "ソフトバンクG", "6857": "アドバンテスト", "5803": "フジクラ",
    "6920": "レーザーテック", "3563": "F&L (スシロー)", "8385": "伊予銀行",
    "5020": "ENEOS", "8136": "サンリオ", "3778": "さくらネット",
    "9107": "川崎汽船", "7011": "三菱重工", "8035": "東エレク",
    "8306": "三菱UFJ", "7203": "トヨタ自動車"
}
WATCH_LIST = list(TICKER_NAMES.keys())

def get_name(code):
    return TICKER_NAMES.get(code, code)

# --- データ取得 ---
def get_data(ticker, period, interval):
    try:
        ticker_mod = f"{ticker}.T" if ".T" not in ticker and ticker.isdigit() else ticker
        df = yf.download(ticker_mod, period=period, interval=interval, progress=False, auto_adjust=False)
        
        if df.empty: return None

        # MultiIndex解除
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
        print(f"Error: {e}")
        return None

# --- テクニカル計算 ---
def calculate_technical_indicators(df):
    df = df.copy()
    
    # 表示用日付フォーマット
    interval_check = df.index.to_series().diff().min()
    if interval_check < timedelta(days=1):
        df['DisplayDate'] = df.index.strftime('%m/%d %H:%M')
    else:
        df['DisplayDate'] = df.index.strftime('%Y/%m/%d')

    # VWAP
    try:
        v = df['Volume'].squeeze()
        tp = ((df['High'] + df['Low'] + df['Close']) / 3).squeeze()
        df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    except:
        df['VWAP'] = np.nan

    # ATR & ADX
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

    # MACD & RSI
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))

    # SuperTrend
    period_st = 10
    multiplier = 3.0
    atr_st = tr.rolling(period_st).mean()
    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr_st)
    basic_lower = hl2 - (multiplier * atr_st)
    
    supertrend = [np.nan] * len(df)
    trend_dir = [True] * len(df)
    final_upper = [np.nan] * len(df)
    final_lower = [np.nan] * len(df)

    for i in range(len(df)):
        if i < period_st:
            final_upper[i] = basic_upper.iloc[i]
            final_lower[i] = basic_lower.iloc[i]
            continue

        prev_close = close.iloc[i-1]
        
        # Upper Band
        if basic_upper.iloc[i] < final_upper[i-1] or prev_close > final_upper[i-1]:
            final_upper[i] = basic_upper.iloc[i]
        else:
            final_upper[i] = final_upper[i-1]
            
        # Lower Band
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
    
    # シグナル判定
    signals = []
    # データ間隔推定 (日足かどうか)
    is_daily = (df.index[1] - df.index[0]) >= timedelta(hours=23) if len(df) > 1 else True
    
    for i in range(len(df)):
        if i < 30:
            signals.append(None)
            continue
            
        row = df.iloc[i]
        prev = df.iloc[i-1]
        sig = None
        
        # クロス判定
        g_cross = prev['MACD'] < prev['Signal'] and row['MACD'] > row['Signal']
        d_cross = prev['MACD'] > prev['Signal'] and row['MACD'] < row['Signal']
        is_uptrend = row['SuperTrend_Dir']
        
        if is_daily:
            if is_uptrend and g_cross: sig = 'SWING_BUY'
            elif not is_uptrend and d_cross: sig = 'SWING_SELL'
            elif row['RSI'] < 30 and row['Close'] > prev['Close']: sig = 'SWING_BUY (RSI)'
            elif row['RSI'] > 70 and row['Close'] < prev['Close']: sig = 'SWING_SELL (RSI)'
        else:
            # 短期足条件
            if pd.isna(row['ADX']) or pd.isna(row['VWAP']):
                signals.append(None)
                continue
            
            adx_ok = row['ADX'] > 25
            buy_vwap = row['Close'] > row['VWAP']
            sell_vwap = row['Close'] < row['VWAP']
            
            if is_uptrend and g_cross and adx_ok and buy_vwap: sig = 'DAY_BUY'
            elif not is_uptrend and d_cross and adx_ok and sell_vwap: sig = 'DAY_SELL'
            
        signals.append(sig)
        
    df['Trade_Signal'] = signals
    return df

# --- バックテスト ---
def run_backtest(df, tp_pct, trade_dir, shares):
    trades = []
    active_trade = None 
    do_long = "買い" in trade_dir
    do_short = "売り" in trade_dir
    
    max_dd = 0
    peak_equity = 0
    equity = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        sig = row['Trade_Signal']
        st_val = row['SuperTrend']
        trade_closed = False
        profit = 0
        
        # 決済チェック
        if active_trade:
            entry_price = active_trade['entry_price']
            entry_tp = active_trade['target_tp']
            
            if active_trade['type'] == 'buy':
                if row['High'] >= entry_tp:
                    profit = (entry_tp - entry_price) * shares
                    trades.append({'date': row['DisplayDate'], 'type': 'Buy', 'res': 'WIN 🏆', 'profit': profit})
                    active_trade = None; trade_closed = True
                elif row['Close'] < st_val:
                    profit = (st_val - entry_price) * shares
                    res = 'WIN (Trail)' if profit > 0 else 'LOSE'
                    trades.append({'date': row['DisplayDate'], 'type': 'Buy', 'res': res, 'profit': profit})
                    active_trade = None; trade_closed = True
                    
            elif active_trade['type'] == 'sell':
                if row['Low'] <= entry_tp:
                    profit = (entry_price - entry_tp) * shares
                    trades.append({'date': row['DisplayDate'], 'type': 'Sell', 'res': 'WIN 🏆', 'profit': profit})
                    active_trade = None; trade_closed = True
                elif row['Close'] > st_val:
                    profit = (entry_price - st_val) * shares
                    res = 'WIN (Trail)' if profit > 0 else 'LOSE'
                    trades.append({'date': row['DisplayDate'], 'type': 'Sell', 'res': res, 'profit': profit})
                    active_trade = None; trade_closed = True

        if trade_closed:
            equity += profit
            if equity > peak_equity: peak_equity = equity
            dd = peak_equity - equity
            if dd > max_dd: max_dd = dd

        # エントリーチェック
        if active_trade is None and not trade_closed and sig is not None:
            if do_long and "BUY" in sig:
                active_trade = {'entry_price': row['Close'], 'type': 'buy', 'target_tp': row['Close']*(1+tp_pct/100)}
            elif do_short and "SELL" in sig:
                active_trade = {'entry_price': row['Close'], 'type': 'sell', 'target_tp': row['Close']*(1-tp_pct/100)}
                
    return trades, max_dd

# --- スキャンロジック ---
def scan_market(notified_ids):
    results = []
    new_notified = notified_ids.copy()
    now_jst = pd.Timestamp.now(tz='Asia/Tokyo')
    
    # 簡易スキャンループ（UI表示はapp.pyで担当するため、ここではデータ作成に集中）
    for t in WATCH_LIST:
        t_name = get_name(t)
        
        # 直近のデータを取得してシグナルがあるか見る
        # ここでは軽量化のため5分足だけチェックする例
        try:
            df = get_data(t, "5d", "5m")
            if df is None or df.empty: continue
            df = calculate_technical_indicators(df)
            
            # 最新の確定足（最後から2番目）を見る
            row = df.iloc[-2] 
            sig = row['Trade_Signal']
            
            if sig:
                # 重複通知防止
                sig_id = f"{row.name}_{t}_{sig}"
                if sig_id not in new_notified:
                    results.append({
                        "time": row.name,
                        "code": t,
                        "name": t_name,
                        "sig": sig,
                        "price": row['Close'],
                        "sl": row['SuperTrend'],
                        "rsi": row['RSI'],
                        "note": "新規検出"
                    })
                    new_notified.add(sig_id)
        except:
            continue
            
    return results, new_notified