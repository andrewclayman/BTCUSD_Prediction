# diagnose_preds.py
import argparse, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

def fmt3(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    return f"{x:.3f}"

def fmt4(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    return f"{x:.4f}"

def find_best_acc_thr(y, p):
    thrs = np.linspace(0.01, 0.99, 197)
    accs = [((p >= t) == y).mean() for t in thrs]
    i = int(np.argmax(accs))
    return float(thrs[i]), float(accs[i])

def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 and len(y) else np.nan

def summarize(name, part):
    if part.empty or part["y"].nunique() < 2:
        print(f"[{name}] insufficient labels; skipping.")
        return None
    y = part["y"].astype(int).values
    p = part["p_up"].astype(float).values

    auc      = safe_auc(y, p)
    brier    = brier_score_loss(y, p)
    thr_acc, best_acc = find_best_acc_thr(y, p)
    acc_050  = ((p >= 0.5) == y).mean()

    pinv = 1.0 - p
    auc_i      = safe_auc(y, pinv)
    brier_i    = brier_score_loss(y, pinv)
    thr_acc_i, best_acc_i = find_best_acc_thr(y, pinv)

    print(f"\n[{name}] N={len(y)}  pos_ratio={y.mean():.3f}")
    print(f" raw : AUC={fmt3(auc)}  Brier={fmt4(brier)}  Acc@0.50={fmt3(acc_050)}  BestAcc={fmt3(best_acc)}@thr={fmt3(thr_acc)}")
    print(f" inv : AUC={fmt3(auc_i)}  Brier={fmt4(brier_i)}  BestAcc={fmt3(best_acc_i)}@thr={fmt3(thr_acc_i)}")

    # Decision rule: prefer orientation with lower Brier; tie-break by higher best-accuracy
    invert_better = (brier_i + 1e-6 < brier) or (abs(brier_i - brier) <= 1e-6 and best_acc_i > best_acc + 1e-6)
    suggested_thr = thr_acc_i if invert_better else thr_acc
    print(f" RECOMMEND: invert={invert_better}  thr≈{fmt3(suggested_thr)}")
    return invert_better, suggested_thr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="runs\\hXX\\moe_preds.parquet with [timestamp,p_up,y]")
    ap.add_argument("--val_start", required=True)
    ap.add_argument("--test_start", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.preds).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    val  = df[(df["timestamp"] >= pd.to_datetime(args.val_start,  utc=True)) &
              (df["timestamp"] <  pd.to_datetime(args.test_start, utc=True))].copy()
    test = df[df["timestamp"] >= pd.to_datetime(args.test_start, utc=True)].copy()

    summarize("VAL", val)
    summarize("TEST", test)

if __name__ == "__main__":
    main()
