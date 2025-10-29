# augment_features_market_extra.py
import argparse
import pathlib
import pandas as pd
import numpy as np

def _read_any_sp500(path):
    df = pd.read_csv(path)
    # Normalize likely formats
    if "Date" in df.columns and "Close" in df.columns:
        pass
    elif "Date" in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    # Enforce presence
    if "Date" not in df.columns:
        raise ValueError("S&P CSV must include a 'Date' column.")
    # Prefer Close, fallback to Adj Close or any 'close'
    close_col = None
    for cand in ["Close", "Adj Close"]:
        if cand in df.columns:
            close_col = cand
            break
    if close_col is None:
        for c in df.columns:
            if str(c).strip().lower() == "close":
                close_col = c
                break
    if close_col is None:
        raise ValueError("Could not find a Close column in S&P CSV.")

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["Date"], utc=True, errors="coerce"),
        "sp_close": pd.to_numeric(df[close_col], errors="coerce")
    }).dropna().sort_values("timestamp").reset_index(drop=True)
    return out

def _read_any_eth(path):
    df = pd.read_csv(path)
    if {"Date","Close"}.issubset(df.columns):
        ts = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        close = pd.to_numeric(df["Close"], errors="coerce")
    elif {"date","close"}.issubset(df.columns):
        ts = pd.to_datetime(df["date"], utc=True, errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
    elif {"timestamp","close"}.issubset(df.columns):
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
    elif {"unix","close"}.issubset(df.columns):
        ts = pd.to_datetime(df["unix"], unit="ms", utc=True, errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
    elif {"unix","close","Volume ETH"}.issubset(df.columns):
        ts = pd.to_datetime(df["unix"], unit="ms", utc=True, errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
    else:
        if {"Date","Adj Close"}.issubset(df.columns):
            ts = pd.to_datetime(df["Date"], utc=True, errors="coerce")
            close = pd.to_numeric(df["Adj Close"], errors="coerce")
        else:
            raise ValueError("ETH CSV must include Date/Close or compatible columns.")

    out = pd.DataFrame({"timestamp": ts, "eth_close": close})
    out = out.dropna().sort_values("timestamp").reset_index(drop=True)
    return out

def _roll_feats(s, windows=(20, 60)):
    out = {}
    ret1 = s.pct_change().replace([np.inf, -np.inf], np.nan)
    out["ret1"] = ret1
    for w in windows:
        out[f"ret1_vol{w}"] = ret1.rolling(w).std()
        out[f"z{w}"] = (s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-9)
        out[f"mom{w}"] = s.pct_change(w)
    return pd.DataFrame(out)

def _merge_safe(left, right, on="timestamp"):
    left = left.copy()
    right = right.copy()
    left[on]  = pd.to_datetime(left[on],  utc=True, errors="coerce")
    right[on] = pd.to_datetime(right[on], utc=True, errors="coerce")
    return left.merge(right, on=on, how="inner")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc_feats", required=True, help="Existing BTC feature parquet with 'timestamp' and 'close' or 'ret_1d_log'")
    ap.add_argument("--sp500_csv", required=True)
    ap.add_argument("--eth_csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    btc = pd.read_parquet(args.btc_feats)
    if "timestamp" not in btc.columns:
        raise ValueError("btc_feats parquet must contain 'timestamp'.")
    btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True, errors="coerce")
    btc = btc.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    sp  = _read_any_sp500(args.sp500_csv)
    eth = _read_any_eth(args.eth_csv)

    sp_feat  = _roll_feats(sp["sp_close"], windows=(20, 60))
    eth_feat = _roll_feats(eth["eth_close"], windows=(20, 60))
    sp_all   = pd.concat([sp[["timestamp", "sp_close"]], sp_feat], axis=1)
    eth_all  = pd.concat([eth[["timestamp", "eth_close"]], eth_feat], axis=1)

    # Merge
    df = _merge_safe(btc, sp_all, on="timestamp")
    df = _merge_safe(df,  eth_all, on="timestamp")

    # Use BTC close if available for correlations; otherwise fall back to daily return from ret_1d_log
    if "close" in df.columns:
        btc_ret = df["close"].pct_change()
    elif "ret_1d_log" in df.columns:
        btc_ret = np.exp(df["ret_1d_log"]) - 1.0
    else:
        raise ValueError("btc_feats must contain 'close' or 'ret_1d_log' to compute correlations.")

    for w in (60, 120):
        df[f"corr_btc_sp_{w}"]  = btc_ret.rolling(w).corr(df["sp_close"].pct_change())
        df[f"corr_btc_eth_{w}"] = btc_ret.rolling(w).corr(df["eth_close"].pct_change())

    df = df.replace([np.inf, -np.inf], np.nan)

    essential = [c for c in ["ret_1d_log"] if c in df.columns]
    if essential:
        df = df.dropna(subset=essential)

    df = df.dropna().reset_index(drop=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote augmented features -> {args.out} | rows={len(df)} | cols={len(df.columns)}")
    new_cols = [c for c in df.columns if c.startswith(("sp_", "eth_", "corr_"))]
    print("New columns examples:", new_cols[:10])

if __name__ == "__main__":
    main()
