import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def build_labels(df: pd.DataFrame, h: int) -> pd.Series:
    """
    Return H-ahead forward log return as Series named f"fwd_logret_h{h}".
    If already present, reuse it; otherwise compute from 'close'.
    """
    tcol = f"fwd_logret_h{h}"
    if tcol in df.columns:
        s = df[tcol].copy()
        s.name = tcol
        return s
    if "close" not in df.columns:
        raise ValueError("Input must contain 'close' or precomputed forward returns.")
    r1 = np.log(df["close"] / df["close"].shift(1))
    s = r1.rolling(h, min_periods=h).sum().shift(-h)
    s.name = tcol
    return s

def main():
    ap = argparse.ArgumentParser(description="Build windowed tensors for sequence models.")
    ap.add_argument("--in", dest="inp", required=True, help="Input parquet with features")
    ap.add_argument("--out", required=True, help="Output .pt path")
    ap.add_argument("--h", type=int, default=5, help="Forecast horizon in bars for label")
    ap.add_argument("--window", type=int, default=96, help="Lookback window length")
    ap.add_argument("--train_frac", type=float, default=0.7, help="Train fraction (chronological)")
    ap.add_argument("--val_frac", type=float, default=0.15, help="Validation fraction (chronological)")
    ap.add_argument("--seed", type=int, default=42)

    # Labeling options
    ap.add_argument("--label_mode", type=str, default="sign",
                    choices=["sign", "median", "quantile"],
                    help="Binarization: 'sign' uses >eps, 'median' uses train median, "
                         "'quantile' keeps tails only and drops the middle.")
    ap.add_argument("--eps", type=float, default=0.0, help="For label_mode=sign: 1 if fwd_logret > eps")
    ap.add_argument("--q_low", type=float, default=0.3, help="Quantile lower bound in [0,0.5)")
    ap.add_argument("--q_high", type=float, default=0.7, help="Quantile upper bound in (0.5,1]")

    # Optional date filters
    ap.add_argument("--min_date", type=str, default="", help="ISO start date, e.g., 2017-01-01")
    ap.add_argument("--max_date", type=str, default="", help="ISO end date, e.g., 2023-12-31")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.inp).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if args.min_date:
        df = df[df["timestamp"] >= pd.to_datetime(args.min_date, utc=True)]
    if args.max_date:
        df = df[df["timestamp"] <= pd.to_datetime(args.max_date, utc=True)]
    print(f"[make_windows] rows={len(df)}  date_range=[{df['timestamp'].min()} .. {df['timestamp'].max()}]")

    y_cont = build_labels(df, args.h)
    tcol = y_cont.name
    df[tcol] = y_cont
    df = df.dropna(subset=[tcol]).reset_index(drop=True)

    num = df.select_dtypes(include=[np.number]).copy()
    drop_tgts = [c for c in num.columns if c.startswith("fwd_logret_h")]
    # Drop ALL forward return columns from features to prevent leakage
    Xdf = num.drop(columns=drop_tgts, errors="ignore")

    # Lag predictors by 1 bar to avoid leakage
    Xdf = Xdf.shift(1)
    valid = np.isfinite(Xdf.values).all(axis=1)
    Xdf = Xdf[valid]
    y_cont_valid = df.loc[valid, tcol].values
    ts = df.loc[valid, "timestamp"].values
    print(f"[valid] kept_rows={len(Xdf)}")

    n = len(Xdf)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    idx_train = np.arange(0, n_train)
    idx_val = np.arange(n_train, n_train + n_val)
    idx_test = np.arange(n_train + n_val, n)

    mode = args.label_mode
    if mode == "sign":
        eps = float(args.eps)
        y_bin_all = (y_cont_valid > eps).astype(np.int64)
        thr_desc = f"sign>eps({eps})"
    elif mode == "median":
        thr = float(np.median(y_cont_valid[idx_train])) if len(idx_train) else float(np.median(y_cont_valid))
        y_bin_all = (y_cont_valid > thr).astype(np.int64)
        thr_desc = f"train_median={thr:.6g}"
    else:
        ql = float(args.q_low); qh = float(args.q_high)
        if not (0.0 <= ql < 0.5 and 0.5 < qh <= 1.0 and ql < qh):
            raise ValueError("For quantile mode, need 0 ≤ q_low < 0.5 < q_high ≤ 1 and q_low < q_high.")
        low_thr  = float(np.quantile(y_cont_valid[idx_train], ql)) if len(idx_train) else float(np.quantile(y_cont_valid, ql))
        high_thr = float(np.quantile(y_cont_valid[idx_train], qh)) if len(idx_train) else float(np.quantile(y_cont_valid, qh))
        y_bin_all = np.full(len(y_cont_valid), -1, dtype=np.int64)
        y_bin_all[y_cont_valid <= low_thr]  = 0
        y_bin_all[y_cont_valid >= high_thr] = 1
        thr_desc = f"train_quantiles q_low={ql}={low_thr:.6g}, q_high={qh}={high_thr:.6g}"

    def _ratio(arr, idx):
        if len(idx) == 0:
            return float("nan")
        v = arr[idx]
        if mode == "quantile":
            v = v[v != -1]
        return float(v.mean()) if len(v) else float("nan")

    overall = y_bin_all[y_bin_all != -1] if mode == "quantile" else y_bin_all
    print(f"[labels] mode={mode} ({thr_desc})")
    print(f"[labels] pos_ratio_all={float(overall.mean()):.3f}")
    print(f"[split] train_raw N={len(idx_train)} pos_ratio={_ratio(y_bin_all, idx_train):.3f}")
    print(f"[split] val_raw   N={len(idx_val)}   pos_ratio={_ratio(y_bin_all, idx_val):.3f}")
    print(f"[split] test_raw  N={len(idx_test)}  pos_ratio={_ratio(y_bin_all, idx_test):.3f}")

    mu = Xdf.iloc[idx_train].mean()
    sd = Xdf.iloc[idx_train].std().replace(0, 1.0)
    Xz = (Xdf - mu) / (sd + 1e-8)
    Xz = Xz.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X = Xz.values
    W = args.window

    def roll(idx):
        if len(idx) == 0:
            return np.empty((0, W, X.shape[1])), np.empty((0,), dtype=np.int64), np.array([], dtype="datetime64[ns]")
        start, end = idx.min(), idx.max()
        rows, labels, tend = [], [], []
        for t in range(start + W - 1, end + 1):
            if mode == "quantile" and y_bin_all[t] == -1:
                continue
            sl = slice(t - W + 1, t + 1)
            rows.append(X[sl])
            labels.append(int(y_bin_all[t]))
            tend.append(ts[t])
        if not rows:
            return np.empty((0, W, X.shape[1])), np.empty((0,), dtype=np.int64), np.array([], dtype="datetime64[ns]")
        return np.stack(rows), np.array(labels, dtype=np.int64), np.array(tend)

    Xtr, ytr, ttr = roll(idx_train)
    Xva, yva, tva = roll(idx_val)
    Xte, yte, tte = roll(idx_test)

    pack = {
        "X_train": torch.tensor(Xtr, dtype=torch.float32),
        "y_train": torch.tensor(ytr, dtype=torch.long),
        "X_val": torch.tensor(Xva, dtype=torch.float32),
        "y_val": torch.tensor(yva, dtype=torch.long),
        "X_test": torch.tensor(Xte, dtype=torch.float32),
        "y_test": torch.tensor(yte, dtype=torch.long),
        "t_test": tte,
        "feature_names": list(Xdf.columns),
        "window": W,
        "horizon": args.h,
        "label_mode": mode,
        "eps": float(args.eps),
        "q_low": float(args.q_low),
        "q_high": float(args.q_high),
        "mu": mu.to_dict(),
        "sd": sd.to_dict(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, args.out)
    print(f"Saved {args.out} | features={X.shape[1]} | windows train/val/test = {len(ytr)}/{len(yva)}/{len(yte)}")

if __name__ == "__main__":
    main()
