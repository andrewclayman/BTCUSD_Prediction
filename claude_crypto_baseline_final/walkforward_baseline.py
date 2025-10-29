# walkforward_baseline.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def to_utc_ts(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    return ts


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" not in df or "close" not in df:
        raise ValueError("features parquet must contain 'timestamp' and 'close'.")
    df = df[["timestamp", "close"]].copy()
    df["timestamp"] = to_utc_ts(df["timestamp"])
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    return df.reset_index(drop=True)


def load_preds(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" not in df or "p_up" not in df:
        raise ValueError("preds parquet must contain 'timestamp' and 'p_up'.")
    df = df[["timestamp", "p_up"]].copy()
    df["timestamp"] = to_utc_ts(df["timestamp"])
    df = df.dropna(subset=["timestamp", "p_up"]).sort_values("timestamp")
    return df.reset_index(drop=True)


def compute_simple_returns(prices: pd.Series) -> pd.Series:
    ret = prices.pct_change()
    ret = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return ret


def position_from_probs(p: np.ndarray,
                        thr: float,
                        band_low: float | None,
                        band_high: float | None,
                        long_only: bool) -> np.ndarray:
    """
    Returns position vector y in {-1,0,1} or {0,1} if long_only.
    If band provided, take long above band_high, short below band_low.
    Else use threshold: long if p>=thr, short otherwise (or 0 if long_only).
    """
    if band_low is not None and band_high is not None:
        y = np.zeros_like(p, dtype=float)
        if long_only:
            y[p >= band_high] = 1.0
            # remain flat inside band or below band_low
        else:
            y[p >= band_high] = 1.0
            y[p <= band_low] = -1.0
        return y
    # single threshold
    if long_only:
        return (p >= thr).astype(float)
    # symmetric long/short around thr
    y = np.where(p >= thr, 1.0, -1.0)
    return y.astype(float)


def apply_delay(y: np.ndarray, delay: int) -> np.ndarray:
    if delay <= 0:
        return y
    yd = np.roll(y, delay)
    yd[:delay] = 0.0
    return yd


def transaction_costs(pos: np.ndarray, cost_bps: float) -> np.ndarray:
    """
    Cost charged on position changes. If pos in {-1,0,1}, change magnitude is 1 or 2.
    cost per change = |Δpos| * bps * 1e-4
    """
    if len(pos) == 0:
        return np.zeros(0, dtype=float)
    delta = np.diff(pos, prepend=pos[0])
    trans = np.abs(delta)
    return trans * cost_bps * 1e-4


def volatility_targeting(gross: pd.Series,
                         costs: pd.Series,
                         ret_underlying: pd.Series,
                         target_vol_annual: float,
                         vol_window: int,
                         bars_per_year: float,
                         leverage_cap: float | None) -> tuple[pd.Series, pd.Series]:
    """
    Scale gross returns and costs by k_t = target / realized_vol_t,
    where realized_vol_t is rolling annualized std of underlying returns.
    """
    if target_vol_annual <= 0:
        return gross, costs

    # Realized volatility of underlying returns
    roll_sd = ret_underlying.rolling(vol_window, min_periods=max(5, vol_window // 4)).std(ddof=0)
    realized = roll_sd * np.sqrt(bars_per_year)
    # back/forward fill for warmup
    realized = realized.bfill().fillna(ret_underlying.std(ddof=0) * np.sqrt(bars_per_year) or 1e-6)

    k = target_vol_annual / realized.replace(0.0, np.nan)
    k = k.fillna(0.0)
    if leverage_cap is not None and leverage_cap > 0:
        k = k.clip(lower=0.0, upper=float(leverage_cap))

    gross_adj = gross * k
    costs_adj = costs * k.abs()  # scale costs with abs(leverage)
    return gross_adj, costs_adj


def run_backtest(df_merge: pd.DataFrame,
                 thr: float,
                 band_low: float | None,
                 band_high: float | None,
                 delay: int,
                 cost_bps: float,
                 long_only: bool,
                 invert: bool,
                 target_vol_annual: float,
                 vol_window: int,
                 bars_per_year: float,
                 leverage_cap: float | None) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns:
      net   -> per-bar net simple returns after costs
      pos   -> position series
      eq    -> equity curve (starting at 1.0)
    """
    p = df_merge["p_up"].to_numpy(dtype=float)
    if invert:
        p = 1.0 - p

    pos = position_from_probs(p, thr, band_low, band_high, long_only)
    pos = apply_delay(pos, delay)
    pos_series = pd.Series(pos, index=df_merge.index)

    ret = compute_simple_returns(df_merge["close"])
    gross = pos_series * ret

    cost = pd.Series(transaction_costs(pos, cost_bps), index=df_merge.index)

    # Optional vol targeting
    gross_adj, cost_adj = volatility_targeting(
        gross=gross,
        costs=cost,
        ret_underlying=ret,
        target_vol_annual=target_vol_annual,
        vol_window=vol_window,
        bars_per_year=bars_per_year,
        leverage_cap=leverage_cap,
    )

    net = gross_adj - cost_adj
    net = net.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    eq = (1.0 + net).cumprod()
    return net, pos_series, eq


def summarize_and_write(net: pd.Series,
                        eq: pd.Series,
                        pos: pd.Series,
                        out_path: Path,
                        summary_csv: Path | None,
                        bars_per_year: float,
                        thr: float,
                        band_low: float | None,
                        band_high: float | None,
                        long_only: bool,
                        invert: bool) -> None:
    # Diagnostics
    ppyear = float(bars_per_year)
    test_n = int(len(net))
    span = f"{net.index.min()} .. {net.index.max()}"

    mu = float(net.mean()) * ppyear
    sd = float(net.std(ddof=0)) * (ppyear ** 0.5)
    sharpe = (mu / sd) if sd > 0 else 0.0

    years = test_n / ppyear if ppyear > 0 else 1.0
    cagr = float(eq.iloc[-1]) ** (1.0 / max(years, 1e-12)) - 1.0
    maxdd = float((eq / eq.cummax() - 1.0).min())

    # Write per-bar equity and position
    out_df = pd.DataFrame({
        "timestamp": net.index,
        "ret_net": net.values,
        "equity": eq.values,
        "position": pos.values,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # Monthly P&L (month-end)
    monthly = net.resample("M").sum()
    monthly_df = pd.DataFrame({
        "month": monthly.index,
        "ret_sum": monthly.values,
        "cum_equity": (1.0 + monthly).cumprod()
    })
    monthly_out = out_path.with_suffix(".monthly.csv")
    monthly_df.to_csv(monthly_out, index=False)

    print(
        f"[single] Span={span}  bars={test_n}  | Sharpe={sharpe:.2f}  CAGR={cagr*100:.2f}%  "
        f"MaxDD={maxdd*100:.2f}%  thr={thr:.3f} band=[{(band_low if band_low is not None else 0.0):.2f},"
        f"{(band_high if band_high is not None else 1.0):.2f}] long_only={bool(long_only)} invert={invert}"
    )
    print(f"[single] Wrote per-bar equity -> {out_path}")
    print(f"[single] Monthly table -> {monthly_out}")

    # Optional summary append
    if summary_csv is not None:
        row = {
            "out": str(out_path),
            "span_start": str(net.index.min()),
            "span_end": str(net.index.max()),
            "bars": test_n,
            "sharpe": sharpe,
            "cagr": cagr,
            "maxdd": maxdd,
            "thr": thr,
            "band_low": band_low,
            "band_high": band_high,
            "long_only": bool(long_only),
            "invert": invert,
        }
        sc = Path(summary_csv)
        if sc.exists():
            sd = pd.read_csv(sc)
            sd = pd.concat([sd, pd.DataFrame([row])], ignore_index=True)
        else:
            sd = pd.DataFrame([row])
        sd.to_csv(sc, index=False)
        print(f"[single] Appended summary -> {sc}")


def main():
    ap = argparse.ArgumentParser(description="Simple backtest from probability predictions.")
    ap.add_argument("--features", required=True, help="Parquet with 'timestamp' and 'close'")
    ap.add_argument("--preds", required=True, help="Parquet with 'timestamp' and 'p_up'")
    ap.add_argument("--h", type=int, default=12, help="Horizon label for bookkeeping only")
    # Trading rule
    ap.add_argument("--thr", type=float, default=0.50, help="Threshold for long/short or long-only")
    ap.add_argument("--band_low", type=float, default=None, help="Confidence band low; enables band trading")
    ap.add_argument("--band_high", type=float, default=None, help="Confidence band high; enables band trading")
    ap.add_argument("--long_only", action="store_true", help="Use long-only rules")
    ap.add_argument("--invert", action="store_true", help="Invert probabilities p -> 1-p")
    ap.add_argument("--delay", type=int, default=1, help="Bars to delay signal execution")
    ap.add_argument("--cost_bps", type=float, default=2.0, help="Per-change transaction cost in bps")
    # Vol targeting
    ap.add_argument("--target_vol_annual", type=float, default=0.0, help="Annualized target vol; 0 disables")
    ap.add_argument("--vol_window", type=int, default=48, help="Rolling window (bars) for realized vol")
    ap.add_argument("--leverage_cap", type=float, default=None, help="Cap on leverage multiplier k_t")
    # Annualization
    ap.add_argument("--bars_per_year", type=float, default=8760.0,
                    help="Annualization factor. Use 8760 for hourly crypto; 252 for daily equities; 365 for daily crypto.")
    # IO
    ap.add_argument("--mode", type=str, default="single", choices=["single"], help="Only 'single' mode supported here")
    ap.add_argument("--out", required=True, help="CSV path for per-bar equity output")
    ap.add_argument("--summary_csv", default=None, help="CSV path to append a summary row")
    args = ap.parse_args()

    # Load
    feat = load_features(args.features)
    preds = load_preds(args.preds)

    # Merge strictly on timestamp to avoid any asof leakage
    df = feat.merge(preds, on="timestamp", how="inner")
    if df.empty:
        raise ValueError("Merged features/preds is empty. Time ranges may not overlap.")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.index = pd.to_datetime(df["timestamp"], utc=True)

    # Backtest
    net, pos, eq = run_backtest(
        df_merge=df,
        thr=float(args.thr),
        band_low=(None if args.band_low is None else float(args.band_low)),
        band_high=(None if args.band_high is None else float(args.band_high)),
        delay=int(args.delay),
        cost_bps=float(args.cost_bps),
        long_only=bool(args.long_only),
        invert=bool(args.invert),
        target_vol_annual=float(args.target_vol_annual),
        vol_window=int(args.vol_window),
        bars_per_year=float(args.bars_per_year),
        leverage_cap=(None if args.leverage_cap is None else float(args.leverage_cap)),
    )

    # Write outputs and summary
    out_path = Path(args.out)
    summary_csv = Path(args.summary_csv) if args.summary_csv else None
    summarize_and_write(
        net=net, eq=eq, pos=pos,
        out_path=out_path,
        summary_csv=summary_csv,
        bars_per_year=float(args.bars_per_year),
        thr=float(args.thr),
        band_low=(None if args.band_low is None else float(args.band_low)),
        band_high=(None if args.band_high is None else float(args.band_high)),
        long_only=bool(args.long_only),
        invert=bool(args.invert),
    )


if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    main()
