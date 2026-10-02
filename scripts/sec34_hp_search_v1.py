#!/usr/bin/env python3
"""
sec34_hp_search_v1.py — One-at-a-time hyperparameter search (Sec. 3.4 / 4.4).

Search portion of src/run_experiments.py from tag v1-17500 (--experiment hp),
with paths made configurable. Output: hp_tuning.json, shipped here as
results/hp_tuning_v1_corpus.json.

This search ran on the EARLIER corpus (17,500 fakes; 1,709 real + 11,947 fake
training waveforms, 41 identities), not on the 20,279-fake corpus of the paper.
It needs that version's dataset_split.csv and waveforms/{real,fake}/ layout
(see tag v1-17500).

Protocol:
  - Training identities only, 5-fold StratifiedGroupKFold (identity groups).
  - Each architecture starts from BEST_CONFIGS (the base) and varies one of
    lr / wd / dropout at a time over HP_SWEEP; the other two stay at the base.
  - Inputs are the raw waveforms (no per-window z-score).
  - Each fold reports its best epoch on the validation fold.
  - The "1D CNN" here is the earlier 1D CNN (kernels 9/7/5, ~56K parameters),
    not the 36K 1D CNN of Table 3.

Usage:
    python scripts/sec34_hp_search_v1.py --data-root data_v1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def parse_args():
    p = argparse.ArgumentParser(description="Sec. 3.4 one-at-a-time HP search (earlier 17,500-fake corpus)")
    p.add_argument("--data-root", default="data",
                   help="data folder of the v1-17500 layout (default: ./data)")
    p.add_argument("--split-file", default=None,
                   help="v1 split CSV (default: <data-root>/dataset_split.csv)")
    p.add_argument("--out-dir", default=None,
                   help="where hp_tuning.json is written (default: <data-root>/results)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--n-folds", type=int, default=5)
    return p.parse_args()


class Waveform1DCNN(nn.Module):
    def __init__(self, dropout=0.3, channels=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = channels
        self.conv1 = nn.Conv1d(1, c1, 9, padding=4); self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, 7, padding=3); self.bn2 = nn.BatchNorm1d(c2)
        self.conv3 = nn.Conv1d(c2, c3, 5, padding=2); self.bn3 = nn.BatchNorm1d(c3)
        self.pool1 = nn.MaxPool1d(2); self.pool2 = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1); self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c3, 1)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc(self.drop(self.gap(x).squeeze(-1))).squeeze(-1)


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm1d(out_ch))
            if stride != 1 or in_ch != out_ch else nn.Identity()
        )
    def forward(self, x):
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class Waveform1DResNet(nn.Module):
    def __init__(self, dropout=0.3, channels=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, c1, 7, padding=3, bias=False),
            nn.BatchNorm1d(c1), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.stage1 = nn.Sequential(BasicBlock1D(c1, c1, dropout=dropout), BasicBlock1D(c1, c1, dropout=dropout))
        self.stage2 = nn.Sequential(BasicBlock1D(c1, c2, stride=2, dropout=dropout), BasicBlock1D(c2, c2, dropout=dropout))
        self.stage3 = nn.Sequential(BasicBlock1D(c2, c3, stride=2, dropout=dropout), BasicBlock1D(c3, c3, dropout=dropout))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c3, 1)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.fc(self.gap(self.stage3(self.stage2(self.stage1(self.stem(x))))).squeeze(-1)).squeeze(-1)


class WaveformTransformer(nn.Module):
    def __init__(self, patch_size=8, d_model=64, nhead=4, num_layers=2, mlp_dim=128, dropout=0.3):
        super().__init__()
        assert 160 % patch_size == 0
        self.patch_embed = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 160 // patch_size + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, mlp_dim, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.patch_embed(x).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], 1) + self.pos_embed
        return self.fc(self.dropout(self.norm(self.encoder(x)[:, 0]))).squeeze(-1)


class ToeplitzViT(nn.Module):
    def __init__(self, image_size=160, patch_size=16, d_model=64, nhead=4, num_layers=2, mlp_dim=128, dropout=0.3):
        super().__init__()
        n = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(1, d_model, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, mlp_dim, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], 1) + self.pos_embed
        return self.fc(self.dropout(self.norm(self.encoder(x)[:, 0]))).squeeze(-1)


ARCH_CLASSES = {
    "1D CNN": (Waveform1DCNN, "X"),
    "1D ResNet": (Waveform1DResNet, "X"),
    "Transformer": (WaveformTransformer, "X"),
    "Toeplitz ViT": (ToeplitzViT, "T"),
}

BEST_CONFIGS = {
    "1D CNN": {"lr": 5e-4, "wd": 1e-3,  "dropout": 0.3},
    "1D ResNet": {"lr": 1e-3, "wd": 5e-4,  "dropout": 0.5},
    "Transformer": {"lr": 1e-3, "wd": 1e-4,  "dropout": 0.3},
    "Toeplitz ViT": {"lr": 1e-3, "wd": 5e-4,  "dropout": 0.5},
}

HP_SWEEP = {
    "lr": [5e-4, 1e-3, 2e-3],
    "wd": [1e-4, 5e-4, 1e-3],
    "dropout": [0.1,  0.3,  0.5],
}


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)

def augment_waveform(x, noise_std=0.05):
    x = x + np.random.normal(0, noise_std, x.shape).astype(np.float32)
    mask_len = np.random.randint(10, 25)
    start = np.random.randint(0, len(x) - mask_len)
    x = x.copy(); x[start:start + mask_len] = 0.0
    return x

def build_toeplitz(X):
    L = X.shape[1]
    idx = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    return X[:, idx].astype(np.float32)

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    i = np.argmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


def get_X(X_plain, X_zscore, T_plain, T_zscore, arch_name, use_zscore):
    _, inp = ARCH_CLASSES[arch_name]
    if inp == "X":
        return X_zscore if use_zscore else X_plain
    return T_zscore if use_zscore else T_plain


def run_cv(model_class, dropout, X_data, y_data, groups, lr, wd, augment=False, epochs=30, batch=64, n_folds=5):
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    all_probs, all_labels, fold_aucs = [], [], []
    for tr_idx, va_idx in sgkf.split(X_data, y_data, groups=groups):
        X_tr, X_va = X_data[tr_idx], X_data[va_idx]
        y_tr, y_va = y_data[tr_idx], y_data[va_idx]
        if augment:
            X_tr = np.stack([augment_waveform(w) for w in X_tr])
        pw = torch.tensor([(y_tr==0).sum() / max((y_tr==1).sum(), 1)]).to(DEVICE)
        mdl = model_class(dropout=dropout).to(DEVICE)
        opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=wd)
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
        Xt = torch.from_numpy(X_tr).unsqueeze(1).to(DEVICE)
        yt = torch.from_numpy(y_tr).to(DEVICE)
        Xv = torch.from_numpy(X_va).unsqueeze(1).to(DEVICE)
        best_auc, best_probs = -1, None
        for _ in range(epochs):
            mdl.train()
            perm = torch.randperm(len(Xt), device=DEVICE)
            for s in range(0, len(Xt), batch):
                ix = perm[s:s+batch]
                opt.zero_grad()
                crit(mdl(Xt[ix]).reshape(-1), yt[ix]).backward()
                opt.step()
            mdl.eval()
            with torch.no_grad():
                probs = torch.sigmoid(mdl(Xv).reshape(-1)).cpu().numpy()
            auc = roc_auc_score(y_va, probs)
            if auc > best_auc:
                best_auc, best_probs = auc, probs.copy()
        all_probs.extend(best_probs)
        all_labels.extend(y_va)
        fold_aucs.append(best_auc)
    return np.array(fold_aucs), np.array(all_probs), np.array(all_labels)


def save_results(results_dir, name, data):
    path = results_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {path}")


def setup_data(args):
    data_root = Path(args.data_root)
    wave_root = data_root / "waveforms"
    split_df = pd.read_csv(Path(args.split_file) if args.split_file else data_root / "dataset_split.csv")
    results_dir = Path(args.out_dir) if args.out_dir else data_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_real = split_df[(split_df["split"] == "train") & (split_df["class"] == "real")]
    train_fake = split_df[(split_df["split"] == "train") & (split_df["class"] == "fake")]

    print("Loading training waveforms...")
    X_real_t = np.stack([np.load(wave_root/"real"/f"{r['video_id']}.npy").astype(np.float32)
                         for _, r in train_real.iterrows()])
    X_fake_t = np.stack([np.load(wave_root/"fake"/f"{r['video_id']}.npy").astype(np.float32)
                         for _, r in train_fake.iterrows()])

    X_plain = np.concatenate([X_real_t, X_fake_t])
    y = np.array([0.0]*len(X_real_t) + [1.0]*len(X_fake_t), dtype=np.float32)
    groups = np.concatenate([
        train_real["video_id"].str.extract(r"(id\d+)")[0].values,
        train_fake["video_id"].str.extract(r"(id\d+)")[0].values,
    ])

    X_zscore = np.stack([zscore(x) for x in X_plain])
    T_plain = build_toeplitz(X_plain)
    T_zscore = build_toeplitz(X_zscore)

    print(f"Training: {int((y==0).sum())} real + {int((y==1).sum())} fake "
          f"| {len(np.unique(groups))} identity groups")

    return dict(
        data_root=data_root, wave_root=wave_root, split_df=split_df,
        results_dir=results_dir,
        X_plain=X_plain, X_zscore=X_zscore, T_plain=T_plain, T_zscore=T_zscore,
        y=y, groups=groups
    )


# ─────────────────────────────────────────────────────────────────────────────
# Experiment C: HP tuning
# ─────────────────────────────────────────────────────────────────────────────
def run_hp_tuning(ctx, args):
    print(f"\n{'='*70}")
    print("  EXPERIMENT C — One-at-a-Time HP Tuning")
    print(f"{'='*70}")
    results = {}
    total = len(ARCH_CLASSES) * sum(len(v) for v in HP_SWEEP.values())
    done = 0

    for arch_name, (cls, _) in ARCH_CLASSES.items():
        base = BEST_CONFIGS[arch_name]
        X_in = get_X(ctx["X_plain"], ctx["X_zscore"],
                     ctx["T_plain"], ctx["T_zscore"],
                     arch_name, False)
        results[arch_name] = {}
        print(f"\n  {arch_name}  "
              f"[base: lr={base['lr']}, wd={base['wd']}, dropout={base['dropout']}]")

        for hp_name, values in HP_SWEEP.items():
            results[arch_name][hp_name] = {}
            for hp_val in values:
                cfg = base.copy(); cfg[hp_name] = hp_val
                is_base = (hp_val == base[hp_name])
                t0 = time.time()
                fold_aucs, probs, labels_ = run_cv(
                    cls, cfg["dropout"], X_in, ctx["y"], ctx["groups"],
                    lr=cfg["lr"], wd=cfg["wd"], augment=False,
                    epochs=args.epochs, batch=args.batch, n_folds=args.n_folds
                )
                done += 1
                preds = (probs >= 0.5).astype(int)
                rpt = classification_report(labels_, preds,
                                              target_names=["real","fake"], output_dict=True)
                results[arch_name][hp_name][str(hp_val)] = {
                    "mean_auc": float(fold_aucs.mean()),
                    "std_auc": float(fold_aucs.std()),
                    "eer": float(compute_eer(labels_, probs)),
                    "p_real": float(rpt["real"]["precision"]),
                    "f1_real": float(rpt["real"]["f1-score"]),
                }
                marker = " [BASE]" if is_base else ""
                print(f"    {hp_name}={hp_val}{marker:<8}  "
                      f"AUC: {fold_aucs.mean():.4f}±{fold_aucs.std():.4f}  "
                      f"P-Real: {rpt['real']['precision']:.4f}  "
                      f"({done}/{total})  [{time.time()-t0:.0f}s]")

    save_results(ctx["results_dir"], "hp_tuning", results)
    return results


def main():
    args = parse_args()
    print(f"Device: {DEVICE}")
    print(f"Data root: {args.data_root}")

    ctx = setup_data(args)
    run_hp_tuning(ctx, args)

    print(f"\n=== Done. Results in {ctx['results_dir']} ===")


if __name__ == "__main__":
    main()
