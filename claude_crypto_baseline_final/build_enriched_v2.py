"""
Build enriched hourly features V2 with:
- Volume-based features (imbalances, momentum, profiles)
- Macro features (SP500, ETH, VIX, DXY, Gold, correlations)
- Funding rate features (perpetual futures sentiment)
"""
import pandas as pd
import numpy as np
import pathlib

DATA_DIR = pathlib.Path("data")

def add_volume_features(df):
    """Add volume-based features"""
    df = df.copy()
    df['is_buy'] = (df['close'] > df['open']).astype(int)
    df['buy_volume'] = df['volume_usd'] * df['is_buy']
    df['sell_volume'] = df['volume_usd'] * (1 - df['is_buy'])

    for w in [6, 12, 24]:
        df[f'vol_imbalance_{w}'] = (
            df['buy_volume'].rolling(w).sum() /
            (df['volume_usd'].rolling(w).sum() + 1e-10)
        )

    df['vol_ret_1h'] = df['volume_usd'].pct_change()
    for w in [6, 12, 24]:
        df[f'vol_mom_{w}'] = df['volume_usd'].pct_change(w)
        df[f'vol_zscore_{w}'] = (
            (df['volume_usd'] - df['volume_usd'].rolling(w).mean()) /
            (df['volume_usd'].rolling(w).std() + 1e-10)
        )

    df['vol_ratio_24'] = df['volume_usd'] / (df['volume_usd'].rolling(24).mean() + 1e-10)
    df['vol_ratio_168'] = df['volume_usd'] / (df['volume_usd'].rolling(168).mean() + 1e-10)

    for w in [12, 24]:
        price_dir = (df['close'] - df['close'].shift(w)) > 0
        vol_dir = df['volume_usd'] > df['volume_usd'].shift(w)
        df[f'vol_price_div_{w}'] = (price_dir != vol_dir).astype(int)

    df = df.drop(columns=['is_buy', 'buy_volume', 'sell_volume'], errors='ignore')
    return df

def resample_to_hourly(daily_df, hourly_timestamps):
    """Forward-fill daily data to hourly frequency"""
    daily_df = daily_df.copy()
    daily_df['timestamp'] = pd.to_datetime(daily_df['timestamp'], utc=True)
    daily_df = daily_df.set_index('timestamp').sort_index()

    hourly_idx = pd.DatetimeIndex(hourly_timestamps)
    if hourly_idx.tz is None:
        hourly_idx = hourly_idx.tz_localize('UTC')
    else:
        hourly_idx = hourly_idx.tz_convert('UTC')

    hourly_df = daily_df.reindex(hourly_idx, method='ffill')
    hourly_df = hourly_df.reset_index()
    hourly_df = hourly_df.rename(columns={'index': 'timestamp'})
    return hourly_df

def add_macro_features(hourly_df):
    """Add macro market features"""
    df = hourly_df.copy()

    sp = pd.read_csv(DATA_DIR / 'sp500_daily.csv')
    eth = pd.read_csv(DATA_DIR / 'ETHUSD_daily.csv')
    vix = pd.read_csv(DATA_DIR / 'vix_daily.csv')
    dxy = pd.read_csv(DATA_DIR / 'dxy_daily.csv')
    gold = pd.read_csv(DATA_DIR / 'gold_daily.csv')

    # Prepare SP500
    sp['timestamp'] = pd.to_datetime(sp['Date'], utc=True, errors='coerce')
    sp = sp.dropna(subset=['timestamp'])
    sp['sp_close'] = pd.to_numeric(sp['Close'], errors='coerce')
    sp = sp[['timestamp', 'sp_close']].dropna()
    sp['sp_ret'] = sp['sp_close'].pct_change()
    for w in [5, 20]:
        sp[f'sp_mom_{w}'] = sp['sp_close'].pct_change(w)
        sp[f'sp_vol_{w}'] = sp['sp_ret'].rolling(w).std()

    # Prepare ETH
    date_col = 'Date' if 'Date' in eth.columns else 'date'
    close_col = 'Close' if 'Close' in eth.columns else 'close'
    eth['timestamp'] = pd.to_datetime(eth[date_col], utc=True, errors='coerce')
    eth = eth.dropna(subset=['timestamp'])
    eth['eth_close'] = pd.to_numeric(eth[close_col], errors='coerce')
    eth = eth[['timestamp', 'eth_close']].dropna()
    eth['eth_ret'] = eth['eth_close'].pct_change()
    for w in [5, 20]:
        eth[f'eth_mom_{w}'] = eth['eth_close'].pct_change(w)
        eth[f'eth_vol_{w}'] = eth['eth_ret'].rolling(w).std()

    # Prepare VIX
    vix['timestamp'] = pd.to_datetime(vix['Date'], utc=True, errors='coerce')
    vix = vix.dropna(subset=['timestamp'])
    vix['vix_close'] = pd.to_numeric(vix['Close'], errors='coerce')
    vix = vix[['timestamp', 'vix_close']].dropna()
    vix['vix_ret'] = vix['vix_close'].pct_change()
    vix['vix_zscore'] = (vix['vix_close'] - vix['vix_close'].rolling(20).mean()) / (vix['vix_close'].rolling(20).std() + 1e-10)

    # Prepare DXY
    dxy['timestamp'] = pd.to_datetime(dxy['Date'], utc=True, errors='coerce')
    dxy = dxy.dropna(subset=['timestamp'])
    dxy['dxy_close'] = pd.to_numeric(dxy['Close'], errors='coerce')
    dxy = dxy[['timestamp', 'dxy_close']].dropna()
    dxy['dxy_ret'] = dxy['dxy_close'].pct_change()

    # Prepare Gold
    gold['timestamp'] = pd.to_datetime(gold['Date'], utc=True, errors='coerce')
    gold = gold.dropna(subset=['timestamp'])
    gold['gold_close'] = pd.to_numeric(gold['Close'], errors='coerce')
    gold = gold[['timestamp', 'gold_close']].dropna()
    gold['gold_ret'] = gold['gold_close'].pct_change()

    # Resample all to hourly
    sp_h = resample_to_hourly(sp, df['timestamp'])
    eth_h = resample_to_hourly(eth, df['timestamp'])
    vix_h = resample_to_hourly(vix, df['timestamp'])
    dxy_h = resample_to_hourly(dxy, df['timestamp'])
    gold_h = resample_to_hourly(gold, df['timestamp'])

    # Merge
    df = df.merge(sp_h, on='timestamp', how='left')
    df = df.merge(eth_h, on='timestamp', how='left')
    df = df.merge(vix_h, on='timestamp', how='left')
    df = df.merge(dxy_h, on='timestamp', how='left')
    df = df.merge(gold_h, on='timestamp', how='left')

    # Add correlations
    btc_ret = df['ret_1h_log'].fillna(0)
    for w in [24, 168]:
        df[f'corr_btc_sp_{w}'] = btc_ret.rolling(w).corr(df['sp_ret'].fillna(0))
        df[f'corr_btc_eth_{w}'] = btc_ret.rolling(w).corr(df['eth_ret'].fillna(0))
        df[f'corr_btc_vix_{w}'] = btc_ret.rolling(w).corr(df['vix_ret'].fillna(0))
        df[f'corr_btc_dxy_{w}'] = btc_ret.rolling(w).corr(df['dxy_ret'].fillna(0))

    # BTC dominance proxies
    df['btc_eth_ratio'] = df['close'] / (df['eth_close'] + 1e-10)
    df['btc_eth_ratio_change'] = df['btc_eth_ratio'].pct_change(24)

    # Risk-on / Risk-off indicators
    df['risk_on'] = (df['sp_ret'] > 0) & (df['vix_ret'] < 0)
    df['risk_off'] = (df['sp_ret'] < 0) & (df['vix_ret'] > 0)

    return df

def add_funding_rate_features(df):
    """Add funding rate features"""
    print("\nAdding funding rate features...")

    # Load funding rate data
    fr = pd.read_csv(DATA_DIR / 'funding_rate_history_BTCUSD.csv')
    fr['timestamp'] = pd.to_datetime(fr['Time(UTC)'], utc=True, errors='coerce')
    fr = fr.dropna(subset=['timestamp'])
    fr['funding_rate'] = pd.to_numeric(fr['Funding Rate'], errors='coerce')
    fr = fr[['timestamp', 'funding_rate']].dropna()
    fr = fr.sort_values('timestamp')

    # Forward fill to hourly frequency
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], utc=True)

    # Resample funding rates to hourly
    fr_h = fr.set_index('timestamp').resample('H').ffill().reset_index()

    # Merge with hourly data
    df_copy = df_copy.merge(fr_h, on='timestamp', how='left')

    # Add funding rate features
    # Rolling means
    for w in [24, 168]:  # 1 day, 1 week
        df_copy[f'fr_mean_{w}'] = df_copy['funding_rate'].rolling(w).mean()
        df_copy[f'fr_std_{w}'] = df_copy['funding_rate'].rolling(w).std()

    # Z-score (extreme funding indicates potential reversal)
    df_copy['fr_zscore'] = (
        (df_copy['funding_rate'] - df_copy['funding_rate'].rolling(168).mean()) /
        (df_copy['funding_rate'].rolling(168).std() + 1e-10)
    )

    # Momentum
    df_copy['fr_mom_24'] = df_copy['funding_rate'].diff(24)
    df_copy['fr_mom_168'] = df_copy['funding_rate'].diff(168)

    # Extreme funding (top/bottom 10% historically)
    q_low = df_copy['funding_rate'].quantile(0.1)
    q_high = df_copy['funding_rate'].quantile(0.9)
    df_copy['fr_extreme_low'] = (df_copy['funding_rate'] < q_low).astype(int)
    df_copy['fr_extreme_high'] = (df_copy['funding_rate'] > q_high).astype(int)

    print(f"Funding rate coverage: {df_copy['funding_rate'].notna().mean():.1%}")
    print(f"Added {8} funding rate features")

    return df_copy

if __name__ == "__main__":
    print("Loading hourly features...")
    df = pd.read_parquet(DATA_DIR / "BTCUSD_hourly_feats.parquet")
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    print("\nAdding volume features...")
    df = add_volume_features(df)
    print(f"After volume features: {len(df.columns)} columns")

    print("\nAdding macro features...")
    df = add_macro_features(df)
    print(f"After macro features: {len(df.columns)} columns")

    print("\nAdding funding rate features...")
    df = add_funding_rate_features(df)
    print(f"After funding rate features: {len(df.columns)} columns")

    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan)

    out_path = DATA_DIR / "BTCUSD_hourly_enriched_v2.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(df.columns)}")

    # Count new features
    orig_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume_btc', 'volume_usd',
                 'ret_1h_log', 'sigma24', 'sigma72', 'atr24', 'ema24', 'ema168',
                 'ema_ratio_24', 'ema_ratio_168', 'rsi14', 'candle_body', 'candle_body_pct',
                 'hl_range', 'hl_range_pct', 'wick_upper', 'wick_lower']
    fwd_cols = [c for c in df.columns if c.startswith('fwd_logret')]
    new_cols = [c for c in df.columns if c not in orig_cols and c not in fwd_cols]
    print(f"\nAdded {len(new_cols)} new features (volume + macro + funding rate)")
