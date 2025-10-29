import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, brier_score_loss

# ------------------------
# Utility
# ------------------------
def batch_iter(X, y, bs, shuffle=True):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, n, bs):
        j = idx[i:i+bs]
        yield X[j], y[j]

def metrics_bin(y_true, p_up, thr=0.5):
    y = y_true.astype(int)
    p = p_up.astype(float)
    acc = np.mean((p >= thr) == y) if len(y) else np.nan
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 and len(y) else np.nan
    brier = brier_score_loss(y, p) if len(y) else np.nan
    return acc, auc, brier

def init_head_bias(module_seq, prior):
    prior = min(max(prior, 1e-4), 1 - 1e-4)
    b = math.log(prior / (1 - prior))
    with torch.no_grad():
        for m in module_seq.modules():
            if isinstance(m, nn.Linear) and m.out_features == 1:
                m.bias.fill_(b)

# ------------------------
# Experts
# ------------------------
class TCNExpert(nn.Module):
    """Dilated temporal convs + attention pooling."""
    def __init__(self, d_in, d_model=128, dropout=0.2):
        super().__init__()
        self.proj = nn.Conv1d(d_in, d_model, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=1, padding=1),
            nn.GELU(), nn.BatchNorm1d(d_model), nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=2, padding=2),
            nn.GELU(), nn.BatchNorm1d(d_model), nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=4, padding=4),
            nn.GELU(), nn.BatchNorm1d(d_model),
        )
        self.attn_q = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):  # [B,T,F]
        y = x.transpose(1, 2)
        y = self.proj(y)
        y = self.tcn(y)
        y = y.transpose(1, 2)
        w = torch.softmax(self.attn_q(y), dim=1)  # [B,T,1]
        z = (y * w).sum(dim=1)
        logit = self.head(z).squeeze(-1)
        return logit, z

class TransformerExpert(nn.Module):
    """Vanilla Transformer encoder over tokenized time steps."""
    def __init__(self, d_in, d_model=128, nhead=4, nlayers=2, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4*d_model, batch_first=True,
            dropout=dropout, activation="gelu"
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):  # [B,T,F]
        y = self.in_proj(x)
        y = self.tr(y)
        z = y[:, -1]
        logit = self.head(z).squeeze(-1)
        return logit, z

class PatchTSTExpert(nn.Module):
    """PatchTST-style: non-overlapping patches + transformer on patches."""
    def __init__(self, d_in, d_model=128, patch_len=8, dropout=0.2, nhead=4, nlayers=2):
        super().__init__()
        self.patch_len = patch_len
        self.embed = nn.Linear(d_in * patch_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4*d_model, batch_first=True,
            dropout=dropout, activation="gelu"
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):  # [B,T,F]
        B, T, F = x.shape
        P = self.patch_len
        t_eff = (T // P) * P
        if t_eff == 0:
            raise ValueError("Sequence shorter than one patch.")
        xt = x[:, T - t_eff:]
        patches = xt.reshape(B, t_eff // P, P * F)
        y = self.embed(patches)
        y = self.tr(y)
        z = y[:, -1]
        logit = self.head(z).squeeze(-1)
        return logit, z

# ------------------------
# Mixture-of-Experts
# ------------------------
class GatingNet(nn.Module):
    """Gating over expert embeddings. Uses pooled raw input + simple MLP."""
    def __init__(self, d_in, d_embed, n_experts, dropout=0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Conv1d(d_in, d_embed, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(d_embed + d_embed*n_experts, 128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_experts)
        )

    def forward(self, x_raw, expert_embeds):
        p = self.proj(x_raw.transpose(1, 2))
        p = self.pool(p).squeeze(-1)
        zcat = torch.cat(expert_embeds + [p], dim=1)
        logits = self.mlp(zcat)
        return torch.softmax(logits, dim=1)

class MoEClassifier(nn.Module):
    def __init__(self, d_in, d_model=128, dropout=0.2, nhead=4, nlayers=2, patch_len=8):
        super().__init__()
        self.exp_cnn  = TCNExpert(d_in, d_model=d_model, dropout=dropout)
        self.exp_tr   = TransformerExpert(d_in, d_model=d_model, nhead=nhead, nlayers=nlayers, dropout=dropout)
        self.exp_ptst = PatchTSTExpert(d_in, d_model=d_model, patch_len=patch_len, dropout=dropout, nhead=nhead, nlayers=nlayers)
        self.experts = [self.exp_cnn, self.exp_tr, self.exp_ptst]
        self.gate = GatingNet(d_in=d_in, d_embed=d_model, n_experts=len(self.experts), dropout=0.1)

    def forward(self, x):
        logits, embeds = [], []
        for exp in self.experts:
            l, z = exp(x)
            logits.append(l)
            embeds.append(z)
        L = torch.stack(logits, dim=1)
        w = self.gate(x, embeds)
        mix = (w * L).sum(dim=1)
        return mix, L, w

# ------------------------
# Train one pack
# ------------------------
def train_one(pack, args, out_dir: Path, tag: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, ytr = pack["X_train"], pack["y_train"]
    Xva, yva = pack["X_val"],   pack["y_val"]
    Xte, yte = pack["X_test"],  pack["y_test"]
    tte = pack["t_test"]

    def _ratio(t): return float(t.float().mean().item()) if len(t) else float("nan")
    print(f"[{tag}] y=1 ratio  train={_ratio(ytr):.3f}  val={_ratio(yva):.3f}  test={_ratio(yte):.3f}")
    for split, yy in [("train", ytr), ("val", yva), ("test", yte)]:
        if len(torch.unique(yy)) < 2:
            print(f"[{tag}] WARNING: {split} labels are single-class; AUC undefined and accuracy uninformative.")

    if len(torch.unique(ytr)) < 2 or len(torch.unique(yva)) < 2 or len(torch.unique(yte)) < 2:
        out_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(out_dir / "metrics.json", "w") as f:
            json.dump({
                "error": "single-class labels in one or more splits",
                "train_pos": _ratio(ytr), "val_pos": _ratio(yva), "test_pos": _ratio(yte)
            }, f, indent=2)
        print(f"[{tag}] Skipped training due to single-class labels.")
        return {"val_loss": np.nan, "acc": np.nan, "auc": np.nan, "brier": np.nan,
                "thr": np.nan, "val_acc_at_thr": np.nan, "acc_thr": np.nan, "brier_thr": np.nan}

    d_in = Xtr.shape[-1]
    model = MoEClassifier(
        d_in=d_in, d_model=args.d_model, dropout=args.dropout,
        nhead=args.nhead, nlayers=args.nlayers, patch_len=args.patch_len
    ).to(device)

    prior = float((ytr == 1).float().mean().item())
    init_head_bias(model.exp_cnn.head, prior)
    init_head_bias(model.exp_tr.head, prior)
    init_head_bias(model.exp_ptst.head, prior)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=6e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_val_probs = None
    patience, bad = 12, 0

    for ep in range(1, args.epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in batch_iter(Xtr.numpy(), ytr.numpy(), args.bs, shuffle=True):
            xb = torch.tensor(xb, dtype=torch.float32, device=device)
            yb = torch.tensor(yb, dtype=torch.float32, device=device)
            opt.zero_grad()
            mix_logit, _, w = model(xb)
            loss = loss_fn(mix_logit, yb)
            ent = -(w * (w.clamp_min(1e-8)).log()).sum(dim=1).mean()
            loss = loss + 0.001 * (-ent)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= max(len(Xtr), 1)

        model.eval()
        with torch.no_grad():
            xv = Xva.to(device)
            v_logit, _, _ = model(xv)
            pva = torch.sigmoid(v_logit).cpu().numpy()

        if not np.isfinite(pva).all():
            n_bad = (~np.isfinite(pva)).sum()
            print(f"[{tag}] WARNING: {n_bad} non-finite val probs; replacing with 0.5 for metrics")
            pva = np.where(np.isfinite(pva), pva, 0.5)

        val_loss = loss_fn(v_logit, yva.to(device).float()).item()
        acc, auc, brier = metrics_bin(yva.numpy(), pva, thr=0.5)
        print(f"[{tag}] Epoch {ep:02d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} "
              f"| val_acc={acc:.3f} | val_auc={auc:.3f} | val_brier={brier:.4f}")

        improved = np.isfinite(val_loss) and (val_loss < best_val - 1e-4)
        if improved:
            best_val = val_loss
            bad = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_val_probs = pva.copy()
        else:
            bad += 1
            if bad >= patience:
                print(f"[{tag}] Early stopping.")
                break

    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        logit, _, _ = model(Xte.to(device))
        pte = torch.sigmoid(logit).cpu().numpy()
        y = yte.numpy()

    if not np.isfinite(pte).all():
        n_bad = (~np.isfinite(pte)).sum()
        print(f"[{tag}] WARNING: {n_bad} non-finite test probs; replacing with 0.5 for metrics")
        pte = np.where(np.isfinite(pte), pte, 0.5)

    acc, auc, brier = metrics_bin(y, pte, thr=0.5)
    print(f"[{tag}] TEST@0.50 | acc={acc:.3f} | auc={auc:.3f} | brier={brier:.4f}")

    thr_best, acc_best, acc_t, brier_t = np.nan, np.nan, np.nan, np.nan
    if best_val_probs is not None and len(best_val_probs) == len(yva):
        yv = yva.numpy().astype(int)
        thrs = np.linspace(0.35, 0.65, 121)
        thr_best, acc_best = 0.5, -1
        for t in thrs:
            a = ((best_val_probs >= t).astype(int) == yv).mean()
            if a > acc_best:
                thr_best, acc_best = t, a
        acc_t, _, brier_t = metrics_bin(y, pte, thr=thr_best)
        print(f"[{tag}] VAL-chosen thr={thr_best:.3f} | VAL_acc={acc_best:.3f}")
        print(f"[{tag}] TEST@{thr_best:.3f} | acc={acc_t:.3f} | auc={auc:.3f} | brier={brier_t:.4f}")
    else:
        print(f"[{tag}] Skipped threshold tuning due to invalid val probabilities.")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": pd.to_datetime(tte), "p_up": pte, "y": y}).to_parquet(out_dir / "moe_preds.parquet", index=False)
    torch.save(best_state, out_dir / "best_state.pt")
    with open(out_dir / "metrics.json", "w") as f:
        import json
        json.dump({
            "val_loss": float(best_val) if np.isfinite(best_val) else None,
            "test_acc@0.50": float(acc) if np.isfinite(acc) else None,
            "test_auc": float(auc) if np.isfinite(auc) else None,
            "test_brier@0.50": float(brier) if np.isfinite(brier) else None,
            "val_thr": float(thr_best) if np.isfinite(thr_best) else None,
            "val_acc_at_thr": float(acc_best) if np.isfinite(acc_best) else None,
            "test_acc@val_thr": float(acc_t) if np.isfinite(acc_t) else None,
            "test_brier@val_thr": float(brier_t) if np.isfinite(brier_t) else None
        }, f, indent=2)

    return {
        "val_loss": best_val,
        "acc": acc, "auc": auc, "brier": brier,
        "thr": thr_best, "val_acc_at_thr": acc_best,
        "acc_thr": acc_t, "brier_thr": brier_t
    }

# ------------------------
# Multi-horizon driver
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="MoE (CNN + Transformer + PatchTST) for crypto direction.")
    ap.add_argument("--data", help=".pt from make_windows.py")
    ap.add_argument("--horizons", type=str, default="", help="Comma-separated horizons, e.g. '1,10,15,20,25,30,50,75'")
    ap.add_argument("--data_tmpl", type=str, default="data/crypto_h{h}.pt", help="Template path with {h} placeholder")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--patch_len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_root", default="runs", help="Root folder for per-horizon outputs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_root = Path(args.out_root)

    if args.horizons.strip():
        horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
        summary_rows = []
        for h in horizons:
            data_path = Path(args.data_tmpl.format(h=h))
            if not data_path.exists():
                raise FileNotFoundError(f"Missing dataset for horizon {h}: {data_path}")
            print(f"\n=== Horizon h={h} | loading {data_path} ===")
            pack = torch.load(str(data_path), weights_only=False, map_location="cpu")
            tag = f"h{h}"
            out_dir = out_root / tag
            metrics = train_one(pack, args, out_dir, tag=tag)
            summary_rows.append({"h": h, **metrics})
        pd.DataFrame(summary_rows).sort_values("h").to_csv(out_root / "summary_by_horizon.csv", index=False)
        print(f"\nWrote summary -> {out_root / 'summary_by_horizon.csv'}")
    else:
        if not args.data:
            raise ValueError("Provide --data for single horizon, or use --horizons with --data_tmpl.")
        pack = torch.load(args.data, weights_only=False, map_location="cpu")
        _ = train_one(pack, args, Path(args.out_root) / "single", tag="single")

if __name__ == "__main__":
    main()
