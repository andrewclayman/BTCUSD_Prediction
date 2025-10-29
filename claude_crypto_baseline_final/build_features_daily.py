import argparse
import numpy as np
import pandas as pd

def ema(a, span):
    return a.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(series, window=14):
    diff = series.diff()
    up = diff.clip(lower=0.0)
    down = -diff.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)

    # Map common column variants
    cols = {c.lower(): c for c in df.columns}
    def get_like(name):
        key = name.lower()
        for k, v in cols.items():
            if k.replace(" ", "_") == key:
                return v
        if name in df.columns:
            return name
        raise KeyError(f"Missing column: {name}")

    # Timestamp
    if "timestamp" not in df.columns:
        if "date" in cols:
            df["timestamp"] = pd.to_datetime(df[get_like("date")])
        elif "unix" in cols:
            df["timestamp"] = pd.to_datetime(df[get_like("unix")], unit="ms")
        else:
            raise ValueError("Need 'date' or 'unix' column.")

    # Std names
    df = df.rename(columns={
        get_like("open"):  "open",
        get_like("high"):  "high",
        get_like("low"):   "low",
        get_like("close"): "close",
    })
    if "Volume BTC" in df.columns: df = df.rename(columns={"Volume BTC": "volume_btc"})
    if "volume btc" in df.columns: df = df.rename(columns={"volume btc": "volume_btc"})
    if "Volume USD" in df.columns: df = df.rename(columns={"Volume USD": "volume_usd"})
    if "volume usd" in df.columns: df = df.rename(columns={"volume usd": "volume_usd"})

    df = df.sort_values("timestamp").reset_index(drop=True)

    # DAILY log return (consistent name!)
    df["ret_1d_log"] = np.log(df["close"] / df["close"].shift(1))

    # Candle / range features
    body = (df["close"] - df["open"]).astype(float)
    df["candle_body"] = body
    df["candle_body_pct"] = body / (df["open"] + 1e-12)
    hl_range = (df["high"] - df["low"]).astype(float)
    df["hl_range"] = hl_range
    df["hl_range_pct"] = hl_range / (df["low"] + 1e-12)
    df["wick_upper"] = (df["high"] - df[["open", "close"]].max(axis=1)).clip(lower=0.0)
    df["wick_lower"] = (df[["open", "close"]].min(axis=1) - df["low"]).clip(lower=0.0)

    # Volatility & trend
    df["sigma20"] = df["ret_1d_log"].rolling(20, min_periods=5).std()
    df["atr20"] = (df["high"] - df["low"]).rolling(20, min_periods=5).mean()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema_ratio_20"] = (df["close"] - df["ema20"]) / (df["ema20"] + 1e-12)
    df["ema_ratio_50"] = (df["close"] - df["ema50"]) / (df["ema50"] + 1e-12)
    df["rsi14"] = rsi(df["close"], 14)

    # Forward log returns for daily horizons
    for h in [1, 3, 5, 10, 20]:
        df[f"fwd_logret_h{h}"] = df["ret_1d_log"].rolling(h, min_periods=h).sum().shift(-h)

    keep = [
        "timestamp","open","high","low","close","volume_btc","volume_usd",
        "ret_1d_log","sigma20","atr20","ema20","ema50","ema_ratio_20","ema_ratio_50","rsi14",
        "candle_body","candle_body_pct","hl_range","hl_range_pct","wick_upper","wick_lower",
        "fwd_logret_h1","fwd_logret_h3","fwd_logret_h5","fwd_logret_h10","fwd_logret_h20",
    ]
    keep = [c for c in keep if c in df.columns]
    df[keep].to_parquet(args.out, index=False)
    print(f"Wrote features: {args.out}  | rows={len(df)}  cols={len(keep)}")

if __name__ == "__main__":
    main()
