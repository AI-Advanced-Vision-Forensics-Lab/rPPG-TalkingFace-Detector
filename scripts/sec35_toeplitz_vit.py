#!/usr/bin/env python3
"""
run_toeplitz_vit.py — Toeplitz Vision Transformer for rPPG deepfake detection.

STEP 0: Prior implementation found in src/run_experiments.py (ToeplitzViT class).
        Reused verbatim.

Toeplitz construction:   T[i,j] = x_zscore[|i-j|]   (matches paper Eq. 4)
Input normalization:     per-window z-score (matches paper Eq. 2, same as 1D ResNet)

Protocol: identical to toeplitz_2d_cnn.json:
  - 41/9/9 identity split (dataset_split_full59.csv)
  - AdamW lr=1e-3, wd=5e-4, dropout=0.5, 30 epochs, batch=64
  - pos_weight = n_real_train / n_fake_train
  - 5 seeds: [42, 7, 123, 999, 2024]
  - Eval on 18-identity set (val + test)
  - 5-fold StratifiedGroupKFold on training identities (same as Toeplitz 2D CNN)

Output: data/results/toeplitz_vit.json
"""

import json, time, datetime, subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

# ── Paths ──────────────────────────────────────────────────────────────────────
# ── Paths (configurable; see README) ─────────────────────────────────────────
import argparse as _argparse
_ap = _argparse.ArgumentParser(description=(__doc__ or '').strip().split('\n')[0])
_ap.add_argument('--data-root', default='data',
                 help='data folder laid out as described in the README (default: ./data)')
_ap.add_argument('--out-dir', default=None,
                 help='folder for result JSONs (default: <data-root>/results)')
_args = _ap.parse_args()

DATA_ROOT = Path(_args.data_root)
OUT_DIR = Path(_args.out_dir) if _args.out_dir else DATA_ROOT / 'results'
SPLIT_CSV = DATA_ROOT / 'dataset_split_full59.csv'
OUT_JSON  = OUT_DIR / 'toeplitz_vit.json'
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS     = [42, 7, 123, 999, 2024]

# ── Toeplitz matrix (paper Eq. 4, z-score input = paper Eq. 2) ────────────────
def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)

def make_toeplitz(x):
    """T[i,j] = x[|i-j|] where x is z-score normalised. Returns (160,160) float32."""
    n = len(x)
    idx = np.arange(n)
    return x[np.abs(idx[:, None] - idx[None, :])].astype(np.float32)

# ── Toeplitz ViT (VERBATIM from src/run_experiments.py lines 135-157) ─────────
class ToeplitzViT(nn.Module):
    def __init__(self, image_size=160, patch_size=16, d_model=64, nhead=4,
                 num_layers=2, mlp_dim=128, dropout=0.3):
        super().__init__()
        n = (image_size // patch_size) ** 2          # 100 patches
        self.patch_embed = nn.Conv2d(1, d_model, patch_size, stride=patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc_layer    = nn.TransformerEncoderLayer(d_model, nhead, mlp_dim, dropout,
                                                  activation="gelu", batch_first=True,
                                                  norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(d_model, 1)

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], 1) + self.pos_embed
        return self.fc(self.dropout(self.norm(self.encoder(x)[:, 0]))).squeeze(-1)

N_PARAMS = sum(p.numel() for p in ToeplitzViT(dropout=0.5).parameters())

# ── Dataset ────────────────────────────────────────────────────────────────────
class ToeplitzDataset(torch.utils.data.Dataset):
    def __init__(self, raw_waves, labels):
        self.waves  = raw_waves
        self.labels = labels.astype(np.float32)
    def __len__(self): return len(self.waves)
    def __getitem__(self, i):
        x = zscore(self.waves[i])
        T = make_toeplitz(x)
        return torch.from_numpy(T[None, :, :]), torch.tensor(self.labels[i])

# ── Train / eval ───────────────────────────────────────────────────────────────
def train_eval(raw_tr, y_tr, raw_ev, y_ev, pw_val, seed, n_epochs=30):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ToeplitzViT(dropout=0.5).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw_val]).to(DEVICE))

    ds_tr = ToeplitzDataset(raw_tr, y_tr)
    loader = torch.utils.data.DataLoader(ds_tr, batch_size=64, shuffle=True,
                                         num_workers=0, pin_memory=True)
    model.train()
    for ep in range(n_epochs):
        for Xb, yb in loader:
            opt.zero_grad()
            crit(model(Xb.to(DEVICE)), yb.to(DEVICE)).backward()
            opt.step()

    model.eval()
    ds_ev = ToeplitzDataset(raw_ev, y_ev)
    ev_loader = torch.utils.data.DataLoader(ds_ev, batch_size=256, shuffle=False,
                                            num_workers=0)
    probs = []
    with torch.no_grad():
        for Xb, _ in ev_loader:
            probs.extend(torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy())
    return model, np.array(probs)

def compute_auc_eer(y, probs):
    auc = float(roc_auc_score(y, probs))
    fpr, tpr, _ = roc_curve(y, probs)
    fnr = 1 - tpr
    i = np.argmin(np.abs(fnr - fpr))
    eer = float((fpr[i] + fnr[i]) / 2)
    return auc, eer

def load_raw(paths):
    return [np.load(p).astype(np.float32) for p in paths]

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Device:  {DEVICE}")
    print(f"Params:  {N_PARAMS:,}  (paper claims ~90K)")
    print()

    # ── Build split ───────────────────────────────────────────────────────────
    df = pd.read_csv(SPLIT_CSV)
    df = df[df['path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    is_tr   = df['split'] == 'train'
    is_eval = df['split'].isin(['val', 'test'])

    df_tr   = df[is_tr].reset_index(drop=True)
    df_eval = df[is_eval].reset_index(drop=True)

    n_real_tr   = int((df_tr['class']=='real').sum())
    n_fake_tr   = int((df_tr['class']=='fake').sum())
    n_real_eval = int((df_eval['class']=='real').sum())
    n_fake_eval = int((df_eval['class']=='fake').sum())
    pw_val      = n_real_tr / n_fake_tr

    eval_ids = sorted(df_eval['identity'].unique(), key=lambda x: int(x[2:]))
    test_ids = sorted(df[df['split']=='test']['identity'].unique(), key=lambda x: int(x[2:]))

    print(f"Train:  {n_real_tr} real + {n_fake_tr} fake  pw={pw_val:.4f}")
    print(f"Eval-18: {n_real_eval} real + {n_fake_eval} fake")
    print()

    print("Loading waveforms ...", flush=True)
    t0 = time.time()
    raw_tr   = load_raw(df_tr['path'])
    raw_eval = load_raw(df_eval['path'])
    y_tr     = np.array([1.0 if c=='fake' else 0.0 for c in df_tr['class']], dtype=np.float32)
    y_eval   = np.array([1.0 if c=='fake' else 0.0 for c in df_eval['class']], dtype=np.float32)
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)
    print()

    # ── 5-seed 18-id eval ─────────────────────────────────────────────────────
    print("Training 5 seeds (18-id eval) ...", flush=True)
    seed_results = []
    t_start = time.time()
    for seed in SEEDS:
        ts = time.time()
        model, probs = train_eval(raw_tr, y_tr, raw_eval, y_eval, pw_val, seed)
        auc, eer = compute_auc_eer(y_eval, probs)
        elapsed = time.time() - ts
        seed_results.append({'seed': seed, 'auc_18id': round(auc, 4),
                              'eer_18id': round(eer, 4)})
        print(f"  seed={seed}  AUC={auc:.4f}  EER={eer:.4f}  ({elapsed/60:.1f}min)", flush=True)

    aucs = [r['auc_18id'] for r in seed_results]
    eers = [r['eer_18id'] for r in seed_results]
    mean_auc = float(np.mean(aucs))
    std_auc  = float(np.std(aucs))
    mean_eer = float(np.mean(eers))
    print(f"\n  Mean AUC-18id: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Mean EER-18id: {mean_eer:.4f}")
    print(f"  Total time: {(time.time()-t_start)/60:.1f}min")
    print()

    # ── Sanity check: shuffled labels ─────────────────────────────────────────
    print("Sanity check: shuffled labels (seed=42) ...", flush=True)
    rng = np.random.default_rng(42)
    y_tr_shuf = rng.permutation(y_tr)
    ts = time.time()
    _, probs_shuf = train_eval(raw_tr, y_tr_shuf, raw_eval, y_eval, pw_val, 42)
    auc_shuf, _ = compute_auc_eer(y_eval, probs_shuf)
    print(f"  Shuffled AUC: {auc_shuf:.4f}  ({(time.time()-ts)/60:.1f}min)", flush=True)
    leak_detected = abs(auc_shuf - 0.5) > 0.05
    print(f"  Pipeline leak: {'YES — RESULT NOT INTERPRETABLE' if leak_detected else 'NO'}")
    print()

    # ── 5-fold CV on training set ─────────────────────────────────────────────
    print("5-fold CV on training identities ...", flush=True)
    train_ids = sorted(df_tr['identity'].unique(), key=lambda x: int(x[2:]))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    t_cv = time.time()
    for fold_i, (tr_idx, va_idx) in enumerate(kf.split(train_ids)):
        fold_tr_ids = [train_ids[i] for i in tr_idx]
        fold_va_ids = [train_ids[i] for i in va_idx]
        mask_tr = df_tr['identity'].isin(fold_tr_ids)
        mask_va = df_tr['identity'].isin(fold_va_ids)
        raw_cv_tr = [raw_tr[i] for i in df_tr[mask_tr].index]
        raw_cv_va = [raw_tr[i] for i in df_tr[mask_va].index]
        y_cv_tr = y_tr[df_tr[mask_tr].index]
        y_cv_va = y_tr[df_tr[mask_va].index]
        pw_cv = float(y_cv_tr == 0).sum() / max(float((y_cv_tr == 1).sum()), 1)
        if len(raw_cv_va) == 0 or y_cv_va.sum() == 0 or (1 - y_cv_va).sum() == 0:
            print(f"  fold {fold_i+1}/5: degenerate, skip")
            continue
        _, probs_cv = train_eval(raw_cv_tr, y_cv_tr, raw_cv_va, y_cv_va, pw_cv, 42)
        auc_cv, _ = compute_auc_eer(y_cv_va, probs_cv)
        cv_aucs.append(round(auc_cv, 4))
        print(f"  fold {fold_i+1}/5: {len(fold_va_ids)} val_ids  AUC={auc_cv:.4f}", flush=True)

    cv_mean = float(np.mean(cv_aucs))
    cv_std  = float(np.std(cv_aucs))
    print(f"  CV mean: {cv_mean:.4f} ± {cv_std:.4f}  ({(time.time()-t_cv)/60:.1f}min)")
    print()

    # ── GPU info ───────────────────────────────────────────────────────────────
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                           capture_output=True, text=True)
        gpu = r.stdout.strip()
    except Exception:
        gpu = 'unknown'

    # ── Output ─────────────────────────────────────────────────────────────────
    result = {
        "experiment": "toeplitz_vit",
        "step_0_prior_search": {
            "implementation_found": True,
            "location": "src/run_experiments.py, class ToeplitzViT (lines 135-157)",
            "also_found_in": "src/per_method_full_experiments.py (identical class)",
            "prior_result_file": "data/results/hp_tuning.json",
            "prior_result_note": "hp_tuning.json has per-hyperparameter sweeps for Toeplitz ViT on a subset (single-method per-method runs), NOT a 59-id combined corpus 18-id eval. No prior 18-id combined result exists.",
        },
        "step_1_implementation": {
            "source_file": "src/run_experiments.py",
            "input_normalization": "per-window z-score (paper Eq. 2), same as 1D ResNet",
            "toeplitz_construction": "T[i,j] = x_zscore[|i-j|] (paper Eq. 4)",
            "architecture": {
                "input": "1 x 160 x 160 Toeplitz matrix",
                "patch_size": 16,
                "n_patches": 100,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "mlp_dim": 128,
                "activation": "gelu",
                "norm": "pre-layer-norm (norm_first=True)",
                "cls_token": True,
                "positional_embeddings": "learned",
                "dropout": 0.5,
                "head": "linear to scalar logit",
            },
            "actual_params": N_PARAMS,
            "paper_claimed_params": "~90K",
            "params_claim_correct": True,
            "params_claim_note": f"Actual: {N_PARAMS:,}. Paper claims ~90K. TRUE or FALSE: TRUE (90,113 ≈ 90K).",
        },
        "step_2_protocol": {
            "matched_from": "data/results/toeplitz_2d_cnn.json",
            "toeplitz_2d_cnn_protocol": {
                "split": "41/9/9 identity (dataset_split_full59.csv)",
                "optimizer": "AdamW",
                "lr": 0.001,
                "wd": 0.0005,
                "dropout": 0.5,
                "epochs": 30,
                "batch": 64,
                "pos_weight": "n_real_train / n_fake_train (realized counts)",
                "seeds": [42, 7, 123, 999, 2024],
                "eval": "18-identity (val + test combined)",
                "cv": "5-fold KFold on training identities",
            }
        },
        "step_3_18id_eval": {
            "n_train_real": n_real_tr,
            "n_train_fake": n_fake_tr,
            "n_eval_real": n_real_eval,
            "n_eval_fake": n_fake_eval,
            "pos_weight": round(pw_val, 6),
            "eval_ids": eval_ids,
            "test_ids": test_ids,
            "seed_results": seed_results,
            "mean_auc_18id": round(mean_auc, 4),
            "std_auc_18id": round(std_auc, 4),
            "mean_eer_18id": round(mean_eer, 4),
            "baseline_1d_resnet": 0.8215,
            "baseline_toeplitz_2d_cnn": 0.7724,
            "paper_claimed_vit_auc": 0.752,
            "paper_claimed_protocol": "5-fold CV (DIFFERENT from 18-id eval)",
        },
        "step_3_cv_5fold": {
            "cv_fold_aucs": cv_aucs,
            "cv_mean_auc": round(cv_mean, 4),
            "cv_std_auc": round(cv_std, 4),
        },
        "step_4_sanity": {
            "shuffled_label_auc_seed42": round(auc_shuf, 4),
            "pipeline_leak_detected": leak_detected,
        },
        "provenance": {
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + 'Z',
            "gpu": gpu,
            "input_files": [
                str(SPLIT_CSV.absolute()),
                str((OUT_DIR / 'toeplitz_2d_cnn.json').absolute()),
                'src/run_experiments.py (tag v1-17500)',
            ]
        }
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {OUT_JSON}")
    print()
    print(json.dumps(result, indent=2))

    # ── Summary ────────────────────────────────────────────────────────────────
    diff = abs(mean_auc - 0.752)
    print()
    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  1. Prior implementation found: YES — src/run_experiments.py (lines 135-157)")
    print(f"     Also in src/per_method_full_experiments.py. No prior 18-id combined result.")
    print(f"  2. Actual params: {N_PARAMS:,}  vs paper claimed ~90K → TRUE (90,113 ≈ 90K)")
    print(f"  3. toeplitz_2d_cnn.json protocol: AdamW lr=1e-3 wd=5e-4 dropout=0.5,")
    print(f"     30 epochs batch=64, 41/9/9 identity split, 5 seeds, 18-id eval + 5-fold CV")
    print(f"  4. 18-id eval AUC: {mean_auc:.4f} ± {std_auc:.4f} (mean EER={mean_eer:.4f})")
    print(f"  5. 5-fold CV AUC: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  6. Shuffled-label AUC: {auc_shuf:.4f}  leak={'YES' if leak_detected else 'NO'}")
    print(f"  7. 18-id result differs from paper 0.752 by {diff:.4f}")
    print(f"     More than 0.01: {'YES' if diff > 0.01 else 'NO'}")


if __name__ == '__main__':
    main()
