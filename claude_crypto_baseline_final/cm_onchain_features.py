# cm_onchain_features.py
import argparse
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import requests

# =========================
# Config
# =========================
CM_ROOT = "https://community-api.coinmetrics.io/v4"  # Community API: no key required

# Pragmatic metrics that are broadly available on the community tier
BASE_METRICS = [
    "AdrActCnt",          # active addresses
    "AdrActNewCnt",       # new addresses
    "TxCnt",              # transaction count
    "TxTfrValAdjUSD",     # adjusted transfer value (USD)
    "FeeTotUSD",          # total fees in USD
    "SplyCur",            # circulating supply
    # Intentionally omit NVTAdj in community pulls due to spotty coverage
]

# Engineering plan applied to whatever survives probing
DERIVED_PLAN = {
    "pct_change": ["AdrActCnt", "AdrActNewCnt", "TxCnt", "TxTfrValAdjUSD", "FeeTotUSD"],
    "ema": {"TxCnt": [7, 30], "AdrActCnt": [7, 30], "TxTfrValAdjUSD": [7, 30]},
    "zscore": ["AdrActCnt", "TxCnt", "TxTfrValAdjUSD", "FeeTotUSD"],
}

# Coverage policy
COVERAGE_MIN_CORE = 0.80        # for your existing/core features
COVERAGE_MIN_OC_POST2017 = 0.30 # for on-chain engineered features measured post cutoff
COVERAGE_CUTOFF = pd.Timestamp("2017-01-01", tz="UTC")  # ignore very early sparse years

# =========================
# Utilities
# =========================
def _infer_bar_frequency(ts: pd.Series) -> str:
    ts = pd.to_datetime(ts, utc=True).dropna().sort_values()
    if len(ts) < 3:
        return "1d"
    dsec = (ts.iloc[1:] - ts.iloc[:-1]).dt.total_seconds().median()
    return "1h" if dsec is not None and 1800 <= dsec <= 5400 else "1d"

def _period_end_floor(s: pd.Series, freq: str) -> pd.Series:
    idx = pd.to_datetime(s, utc=True)
    return idx.dt.floor("H" if freq == "1h" else "D")

def _safe_get_json(url: str, params: Optional[Dict] = None, max_retries: int = 5) -> Dict:
    """Request JSON with retries and graceful error shape on failure."""
    import time
    last_err = None
    for i in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            try:
                js = r.json()
            except Exception:
                r.raise_for_status()
                return {"data": []}
            if r.status_code >= 400:
                return {"data": [], "error": js}
            return js
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(0.5 * (2 ** i))  # 0.5,1,2,4,8s
    print(f"[cm_onchain] WARN: network error for {url} → {last_err}")
    return {"data": [], "error": {"network_error": str(last_err)}}

def _cm_timeseries_request(
    assets: str,
    metrics: List[str],
    start_time: Optional[str],
    end_time: Optional[str],
    frequency: str,
) -> pd.DataFrame:
    """
    Pull /timeseries/asset-metrics with pagination.
    Returns standardized long-form dataframe with columns: time, asset, metric, value
    """
    url = f"{CM_ROOT}/timeseries/asset-metrics"
    params = {
        "assets": assets,
        "metrics": ",".join(metrics),
        "frequency": frequency,
        "start_time": start_time,
        "end_time": end_time,
        "page_size": 10000,
    }
    frames = []
    js = _safe_get_json(url, params)
    data = js.get("data", [])
    if data:
        frames.append(pd.DataFrame(data))
    next_url = js.get("next_page_url")

    while next_url:
        js = _safe_get_json(next_url, None)
        data = js.get("data", [])
        if data:
            frames.append(pd.DataFrame(data))
        next_url = js.get("next_page_url")

    if not frames:
        return pd.DataFrame(columns=["time", "asset", "metric", "value"])

    df = pd.concat(frames, ignore_index=True)

    # Some error payloads lack 'value'
    for col in ("time", "asset", "metric", "value"):
        if col not in df.columns:
            return pd.DataFrame(columns=["time", "asset", "metric", "value"])

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["time"])
    return df[["time", "asset", "metric", "value"]]

def _probe_working_metrics(asset: str, candidates: List[str], frequency: str) -> List[str]:
    """
    Return only metric IDs that produce non-empty data on a small recent window.
    Keeps the script robust to Community coverage differences.
    """
    ok = []
    end = pd.Timestamp.utcnow().floor("D") - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=14)
    s = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    e = end.strftime("%Y-%m-%dT%H:%M:%SZ")
    for m in candidates:
        js = _safe_get_json(
            f"{CM_ROOT}/timeseries/asset-metrics",
            dict(assets=asset, metrics=m, frequency=frequency, start_time=s, end_time=e, page_size=1000),
        )
        data = js.get("data", [])
        if data:
            ok.append(m)
    return ok

def _pivot_metrics(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame()
    wide = df_long.pivot_table(index=["time", "asset"], columns="metric", values="value", aggfunc="last")
    wide = wide.sort_index().reset_index()
    wide.columns.name = None
    return wide

def _expanding_z(x: pd.Series, minp: int = 30) -> pd.Series:
    mu = x.expanding(min_periods=minp).mean()
    sd = x.expanding(min_periods=minp).std()
    return (x - mu) / sd.replace(0, np.nan)

def _engineer_derivatives(oc: pd.DataFrame) -> pd.DataFrame:
    out = oc.copy()

    # Percent changes
    for c in DERIVED_PLAN["pct_change"]:
        if c in out:
            out[f"{c}_pct"] = out[c].pct_change().replace([np.inf, -np.inf], np.nan)

    # EMAs
    for c, spans in DERIVED_PLAN["ema"].items():
        if c in out:
            for s in spans:
                out[f"{c}_ema{s}"] = out[c].ewm(span=s, adjust=False, min_periods=s).mean()

    # Z-scores
    for c in DERIVED_PLAN["zscore"]:
        if c in out:
            out[f"{c}_z"] = _expanding_z(out[c])

    # Intensity ratio
    if "TxTfrValAdjUSD" in out and "SplyCur" in out:
        out["TfrPerSupply_USD"] = out["TxTfrValAdjUSD"] / (out["SplyCur"] + 1e-12)

    # Clean numeric columns
    num_cols = [c for c in out.columns if c not in ("time", "asset")]
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out

def _leak_safe_merge(price_df: pd.DataFrame, oc_df: pd.DataFrame, price_freq: str) -> pd.DataFrame:
    """
    Left-join on-chain daily data to price bars, align to bar-end, and lag by one bar to avoid look-ahead.
    """
    if oc_df.empty:
        return price_df.copy()

    p = price_df.copy()
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True, errors="coerce")
    p["timestamp"] = _period_end_floor(p["timestamp"], price_freq)

    oc = oc_df.rename(columns={"time": "timestamp"}).copy()
    oc["timestamp"] = pd.to_datetime(oc["timestamp"], utc=True, errors="coerce")
    # OC fetched daily; align to daily end, upsample if needed
    oc["timestamp"] = oc["timestamp"].dt.floor("D")
    if price_freq == "1h":
        oc = oc.set_index("timestamp").sort_index().resample("H").ffill().reset_index()

    merged = p.merge(oc, on="timestamp", how="left")
    newcols = [c for c in merged.columns if c not in p.columns and c != "asset"]
    merged[newcols] = merged[newcols].shift(1)  # one-bar lag
    return merged

# =========================
# Main pipeline
# =========================
def build_onchain_features(
    price_path: str,
    out_path: str,
    asset: str = "btc",
    metrics: Optional[List[str]] = None,
):
    # Read price/features parquet
    dfp = pd.read_parquet(price_path).sort_values("timestamp").reset_index(drop=True)
    if "timestamp" not in dfp.columns:
        raise ValueError("Input parquet must contain a 'timestamp' column.")
    dfp["timestamp"] = pd.to_datetime(dfp["timestamp"], utc=True, errors="coerce")

    price_freq = _infer_bar_frequency(dfp["timestamp"])

    # Pull a lead window for EMAs/z-scores; OC requests are forced daily
    lead_days = 120
    start = (dfp["timestamp"].min() - pd.Timedelta(days=lead_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (dfp["timestamp"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    want_metrics = metrics if metrics else BASE_METRICS

    # Probe to keep only community-available metrics
    working = _probe_working_metrics(asset, want_metrics, frequency="1d")
    if not working:
        print("[cm_onchain] WARN: no community metrics responded; proceeding with input only.")
        merged = dfp.loc[:, ~dfp.columns.duplicated(keep="first")]
        merged.to_parquet(out_path, index=False)
        return

    # Fetch daily OC
    raw = _cm_timeseries_request(asset, working, start, end, frequency="1d")
    wide = _pivot_metrics(raw)
    oc_feat = _engineer_derivatives(wide)

    merged = _leak_safe_merge(dfp, oc_feat, price_freq)

    # ---------- Smart coverage selection ----------
    # 1) Deduplicate columns first
    merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]

    # 2) Identify on-chain columns = columns not in original dfp and not 'asset'
    oc_cols = [c for c in merged.columns if c not in dfp.columns and c not in ("timestamp", "asset")]

    # 3) Compute coverage on post-cutoff period to avoid penalizing early sparse years
    post_mask = merged["timestamp"] >= COVERAGE_CUTOFF
    post = merged.loc[post_mask]
    cov_post = post.notna().mean()

    keep = ["timestamp"]
    for c in merged.columns:
        if c == "timestamp":
            continue
        if c in oc_cols:
            if cov_post.get(c, 0.0) >= COVERAGE_MIN_OC_POST2017:
                keep.append(c)
        else:
            if cov_post.get(c, 0.0) >= COVERAGE_MIN_CORE:
                keep.append(c)

    merged[keep].to_parquet(out_path, index=False)

    # ---------- Logging ----------
    used = working
    added = [c for c in keep if c not in dfp.columns and c != "timestamp"]
    print(f"[cm_onchain] price_freq={price_freq} | asset={asset}")
    print(f"[cm_onchain] metrics_requested={want_metrics}")
    print(f"[cm_onchain] metrics_used={used}")
    print(f"[cm_onchain] rows={len(merged)} | cols_out={len(keep)} | added_cols={len(added)}")
    for c in added:
        print("   + OC:", c)

# =========================
# CLI
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Add Coin Metrics Community on-chain features to your features parquet.")
    ap.add_argument("--in", dest="inp", required=True, help="Input features parquet (must include 'timestamp')")
    ap.add_argument("--out", required=True, help="Output parquet with on-chain features merged")
    ap.add_argument("--asset", default="btc", help="Asset code, e.g., btc, eth")
    ap.add_argument("--metrics", default="", help="Comma-separated metric IDs to override defaults")
    args = ap.parse_args()

    metric_list = [m.strip() for m in args.metrics.split(",") if m.strip()] if args.metrics else None
    build_onchain_features(args.inp, args.out, asset=args.asset.lower(), metrics=metric_list)
