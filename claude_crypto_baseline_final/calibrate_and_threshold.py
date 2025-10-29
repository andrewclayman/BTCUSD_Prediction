# calibrate_and_threshold.py
import argparse, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True, help="runs\\hXX\\moe_preds.parquet (with timestamp, p_up, y)")
ap.add_argument("--out", required=True, help="output parquet with p_up_cal")
ap.add_argument("--val_start", required=True, help="ISO date delimiting VAL<=date<TEST")
ap.add_argument("--test_start", required=True, help="ISO date delimiting TEST>=date")
args = ap.parse_args()

df = pd.read_parquet(args.preds).sort_values("timestamp").reset_index(drop=True)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

val = df[(df["timestamp"] < pd.to_datetime(args.test_start, utc=True)) &
         (df["timestamp"] >= pd.to_datetime(args.val_start, utc=True))].copy()
test = df[df["timestamp"] >= pd.to_datetime(args.test_start, utc=True)].copy()

# Fallback if any split empty
if len(val)==0 or len(test)==0 or val["y"].nunique()<2:
    print("[warn] invalid splits for calibration; copying through")
    df["p_up_cal"] = df["p_up"].values
    df.to_parquet(args.out, index=False)
    raise SystemExit(0)

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(val["p_up"].values, val["y"].values.astype(int))

df["p_up_cal"] = iso.predict(df["p_up"].values)

# Diagnostics
def _m(y,p): 
    auc = roc_auc_score(y, p) if len(np.unique(y))==2 else np.nan
    brier = brier_score_loss(y, p)
    return auc, brier
a0,b0 = _m(test["y"].values, test["p_up"].values)
a1,b1 = _m(test["y"].values, test["p_up_cal"].values)
print(f"TEST raw  : AUC={a0:.3f} | Brier={b0:.4f}")
print(f"TEST calib: AUC={a1:.3f} | Brier={b1:.4f}")

df.to_parquet(args.out, index=False)
print(f"Wrote -> {args.out}")
