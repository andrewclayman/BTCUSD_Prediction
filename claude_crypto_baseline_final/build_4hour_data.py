import pandas as pd
import numpy as np
import pathlib

DATA_DIR = pathlib.Path("data")

def resample_to_4h(hourly_csv):
    """Resample hourly OHLCV to 4-hour bars"""
    df = pd.read_csv(hourly_csv)
    df['timestamp'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('timestamp').sort_index()

    # Rename columns to match expected format
    df = df.rename(columns={'Volume BTC': 'volume_btc', 'Volume USD': 'volume_usd'})

    # Resample to 4H bars
    resampled = df.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume_btc': 'sum',
        'volume_usd': 'sum'
    }).dropna()

    return resampled.reset_index()

def add_technical_features(df):
    """Add technical indicators to OHLCV data"""
    df = df.copy()

    # Returns
    df['ret_4h_log'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility
    df['sigma24'] = df['ret_4h_log'].rolling(24).std()  # 24 * 4h = 4 days
    df['sigma72'] = df['ret_4h_log'].rolling(72).std()  # 72 * 4h = 12 days

    # ATR
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr24'] = tr.rolling(24).mean()

    # EMAs
    df['ema24'] = df['close'].ewm(span=24, adjust=False).mean()
    df['ema168'] = df['close'].ewm(span=168, adjust=False).mean()  # 168 * 4h = 4 weeks
    df['ema_ratio_24'] = df['close'] / df['ema24']
    df['ema_ratio_168'] = df['close'] / df['ema168']

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi14'] = 100 - (100 / (1 + rs))

    # Candle features
    df['candle_body'] = df['close'] - df['open']
    df['candle_body_pct'] = df['candle_body'] / df['open']
    df['hl_range'] = df['high'] - df['low']
    df['hl_range_pct'] = df['hl_range'] / df['close']
    df['wick_upper'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['wick_lower'] = df[['open', 'close']].min(axis=1) - df['low']

    # Forward returns for labels (will be dropped in make_windows.py)
    for h in [1, 3, 6, 12, 24, 42]:  # horizons in 4-hour bars
        r1 = np.log(df['close'] / df['close'].shift(1))
        df[f'fwd_logret_h{h}'] = r1.rolling(h, min_periods=h).sum().shift(-h)

    return df

if __name__ == "__main__":
    print("Creating 4-hour data from hourly...")

    # Resample
    df_4h = resample_to_4h(DATA_DIR / "BTCUSD_hourly.csv")
    print(f"4-hour bars: {len(df_4h)} rows")

    # Add features
    df_4h = add_technical_features(df_4h)
    print(f"Features added: {len(df_4h.columns)} columns")

    # Save
    out_path = DATA_DIR / "BTCUSD_4hour_feats.parquet"
    df_4h.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Date range: {df_4h['timestamp'].min()} to {df_4h['timestamp'].max()}")
