# calibrate_fix_and_walk.py
import argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import subprocess, sys

def calibrate(preds_path, out_path, val_start, test_start, invert=False):
    df = pd.read_parquet(preds_path).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if invert:
        df["p_up"] = 1.0 - df["p_up"].astype(float)

    val = df[(df["timestamp"] >= pd.to_datetime(val_start, utc=True)) &
             (df["timestamp"] <  pd.to_datetime(test_start, utc=True))].copy()
    test = df[df["timestamp"] >= pd.to_datetime(test_start, utc=True)].copy()

    if len(val)==0 or val["y"].nunique()<2:
        df["p_up_cal"] = df["p_up"].values
        df.to_parquet(out_path, index=False)
        print("[warn] invalid val; passed through.")
        return

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val["p_up"].values.astype(float), val["y"].values.astype(int))
    df["p_up_cal"] = iso.predict(df["p_up"].values.astype(float))

    if len(test) and test["y"].nunique() == 2:
        y = test["y"].astype(int).values
        p0 = test["p_up"].astype(float).values
        p1 = test["p_up_cal"].astype(float).values
        print(f"TEST raw  : AUC={roc_auc_score(y,p0):.3f} Brier={brier_score_loss(y,p0):.4f}")
        print(f"TEST calib: AUC={roc_auc_score(y,p1):.3f} Brier={brier_score_loss(y,p1):.4f}")

    df.to_parquet(out_path, index=False)
    print(f"[ok] wrote calibrated preds -> {out_path}")

def run_walkforward(features, preds_cal, h, cost_bps, delay, out_csv, summary_csv, thr_grid):
    for thr in thr_grid:
        out = Path(out_csv.replace(".csv", f"_thr{thr:.3f}.csv"))
        cmd = [
            sys.executable, "walkforward_baseline.py",
            "--features", features,
            "--preds", preds_cal,
            "--h", str(h),
            "--thr", f"{thr:.3f}",
            "--cost_bps", str(cost_bps),
            "--delay", str(delay),
            "--min_train", "480",
            "--test_len", "120",
            "--n_folds", "6",
            "--out", str(out),
            "--summary_csv", summary_csv
        ]
        print(">", " ".join(cmd))
        subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--val_start", required=True)
    ap.add_argument("--test_start", required=True)
    ap.add_argument("--invert", action="store_true", help="invert scores before calibration")
    ap.add_argument("--cost_bps", type=float, default=5.0)
    ap.add_argument("--delay", type=int, default=1)
    ap.add_argument("--summary_csv", default="runs/wf_summary.csv")
    args = ap.parse_args()

    preds_cal = str(Path(args.preds).with_name("moe_preds_cal.parquet"))
    calibrate(args.preds, preds_cal, args.val_start, args.test_start, invert=args.invert)

    # Sweep a reasonable grid; tighten after you see results
    thr_grid = np.round(np.linspace(0.50, 0.85, 8), 3)
    run_walkforward(
        features=args.features,
        preds_cal=preds_cal,
        h=args.h,
        cost_bps=args.cost_bps,
        delay=args.delay,
        out_csv=f"runs/bt_wf_h{args.h}.csv",
        summary_csv=args.summary_csv,
        thr_grid=thr_grid
    )

if __name__ == "__main__":
    main()
