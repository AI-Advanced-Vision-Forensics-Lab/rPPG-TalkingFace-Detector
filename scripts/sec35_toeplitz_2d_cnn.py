"""
sec35_toeplitz_2d_cnn.py — Toeplitz 2D CNN for rPPG deepfake detection.

Each 160-sample z-scored waveform is converted to a 160×160 symmetric Toeplitz matrix:
    T[i,j] = x[|i-j|]   (x is 0-indexed, so T is determined by x[0..159])

A 3-block 2D CNN (~32K params) classifies the matrix.
Trained on combined 7-generator corpus (dataset_split_full59.csv).

Reports:
  - 18-id eval AUC/EER per seed and mean ± std
  - 9-id test AUC/EER per seed and mean ± std
  - 5-fold CV AUC on training set (matching Toeplitz ViT protocol)

Output: data/results/toeplitz_2d_cnn.json
"""

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

# ── Paths ─────────────────────────────────────────────────────────────────────
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
REAL_WF   = DATA_ROOT / 'waveforms/real'
FAKE_WF   = DATA_ROOT / 'waveforms/CelebDF/TalkingFace'
OUT_JSON  = OUT_DIR / 'toeplitz_2d_cnn.json'

SEEDS   = [42, 7, 123, 999, 2024]
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
METHODS = ['AniTalker','EDTalk','EchoMimic','FLOAT','IP_LAP','Real3DPortrait','SadTalker']
N_CV    = 5

# ── Toeplitz matrix ───────────────────────────────────────────────────────────
def make_toeplitz(x):
    """Build 160×160 symmetric Toeplitz matrix. T[i,j] = x[|i-j|]."""
    n = len(x)
    idx = np.arange(n)
    T = x[np.abs(idx[:, None] - idx[None, :])]
    return T.astype(np.float32)

def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)

# ── 3-block 2D CNN (≈32K params) ─────────────────────────────────────────────
class ToeplitzCNN(nn.Module):
    """
    Input: (B, 1, 160, 160)
    Block 1: Conv2d(1→8, 3, pad=1) BN ReLU MaxPool2d(4)   → (B, 8,  40, 40)
    Block 2: Conv2d(8→16, 3, pad=1) BN ReLU MaxPool2d(4)  → (B, 16, 10, 10)
    Block 3: Conv2d(16→32, 3, pad=1) BN ReLU MaxPool2d(2) → (B, 32,  5,  5)
    Flatten → 800 → FC(800,32) ReLU Dropout(0.5) → FC(32,1)
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1,  8,  3, padding=1, bias=False),
            nn.BatchNorm2d(8),  nn.ReLU(True), nn.MaxPool2d(4))
        self.block2 = nn.Sequential(
            nn.Conv2d(8,  16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(True), nn.MaxPool2d(4))
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 32),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(32, 1))

    def forward(self, x):
        # x: (B, 1, 160, 160)
        return self.fc(self.block3(self.block2(self.block1(x)))).squeeze(-1)

N_PARAMS = sum(p.numel() for p in ToeplitzCNN().parameters())

def eer_from_roc(y, probs):
    fpr, tpr, _ = roc_curve(y, probs)
    fnr = 1 - tpr
    i = np.argmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)

# ── Build DataFrame ───────────────────────────────────────────────────────────
def build_df():
    sp = pd.read_csv(SPLIT_CSV)
    rows = []
    for _, r in sp.iterrows():
        p = Path(r['path'])
        if not p.exists():
            continue
        rows.append({**r.to_dict(), 'npy_path': str(p)})
    return pd.DataFrame(rows)

# ── Load waveforms → raw arrays ───────────────────────────────────────────────
def load_raw(paths):
    return [np.load(p).astype(np.float32) for p in paths]

# ── Toeplitz dataset ──────────────────────────────────────────────────────────
class ToeplitzDataset(torch.utils.data.Dataset):
    def __init__(self, raw_waves, labels):
        self.waves  = raw_waves   # list of np arrays length 160
        self.labels = labels      # np float32 array

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, i):
        x = zscore(self.waves[i])
        T = make_toeplitz(x)
        return torch.from_numpy(T[None, :, :]), torch.tensor(self.labels[i])

# ── Train / eval ──────────────────────────────────────────────────────────────
def train_eval(raw_tr, y_tr, raw_ev, y_ev, pw_val, seed, n_epochs=30):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ToeplitzCNN().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw_val]).to(DEVICE))

    ds_tr = ToeplitzDataset(raw_tr, y_tr)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=64, shuffle=True,
                                        num_workers=0, pin_memory=True)
    model.train()
    for ep in range(n_epochs):
        for X, y in dl_tr:
            opt.zero_grad()
            crit(model(X.to(DEVICE)), y.to(DEVICE)).backward()
            opt.step()

    # Detect degenerate: check score variance after epoch 1
    model.eval()
    probs = infer(model, raw_ev)
    probs_arr = np.array(probs)
    auc = float(roc_auc_score(y_ev, probs_arr))
    eer = eer_from_roc(y_ev, probs_arr)
    degenerate = (probs_arr.std() < 0.02) or (abs(auc - 0.5) < 0.03)
    return auc, eer, probs_arr, degenerate

def infer(model, raw_waves):
    model.eval()
    probs = []
    ds = ToeplitzDataset(raw_waves, np.zeros(len(raw_waves)))
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    with torch.no_grad():
        for X, _ in dl:
            probs.extend(torch.sigmoid(model(X.to(DEVICE))).cpu().numpy())
    return probs

# ── Main ──────────────────────────────────────────────────────────────────────
print(f"Device:   {DEVICE}")
print(f"Params:   {N_PARAMS:,}")
print(f"Seeds:    {SEEDS}")
print()

print("Building split ...", flush=True)
df = build_df()
is_tr   = df['split'] == 'train'
is_eval = df['split'].isin(['val', 'test'])
is_test = df['split'] == 'test'

df_tr   = df[is_tr].reset_index(drop=True)
df_eval = df[is_eval].reset_index(drop=True)
df_test = df[is_test].reset_index(drop=True)

n_real_tr   = int((df_tr['class']=='real').sum())
n_fake_tr   = int((df_tr['class']=='fake').sum())
n_real_eval = int((df_eval['class']=='real').sum())
n_fake_eval = int((df_eval['class']=='fake').sum())
n_real_test = int((df_test['class']=='real').sum())
n_fake_test = int((df_test['class']=='fake').sum())
pw_val      = n_real_tr / max(n_fake_tr, 1)

tr_ids   = sorted(df_tr['identity'].unique(), key=lambda x: int(x[2:]))
eval_ids = sorted(df_eval['identity'].unique(), key=lambda x: int(x[2:]))
test_ids = sorted(df_test['identity'].unique(), key=lambda x: int(x[2:]))

print(f"Train: {n_real_tr} real + {n_fake_tr} fake  ({len(tr_ids)} ids)  pw={pw_val:.4f}")
print(f"Eval (18-id): {n_real_eval} real + {n_fake_eval} fake  ({len(eval_ids)} ids)")
print(f"Test  (9-id): {n_real_test} real + {n_fake_test} fake  ({len(test_ids)} ids)")
print(f"Eval IDs: {eval_ids}")
print(f"Test IDs: {test_ids}")

print("\nLoading waveforms (raw) ...", flush=True)
t0 = time.time()
raw_tr   = load_raw(df_tr['npy_path'])
raw_eval = load_raw(df_eval['npy_path'])
raw_test = load_raw(df_test['npy_path'])
y_tr   = np.array([float(c=='fake') for c in df_tr['class']],   dtype=np.float32)
y_eval = np.array([float(c=='fake') for c in df_eval['class']], dtype=np.float32)
y_test = np.array([float(c=='fake') for c in df_test['class']], dtype=np.float32)
print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

# ── Train 5 seeds ─────────────────────────────────────────────────────────────
print("\nTraining 5 seeds ...", flush=True)
t_total = time.time()
seed_results = []
all_probs_eval = []
all_probs_test = []

for seed in SEEDS:
    t1 = time.time()
    auc_eval, eer_eval, probs_eval, degen = train_eval(
        raw_tr, y_tr, raw_eval, y_eval, pw_val, seed)
    # Also infer on test set
    torch.manual_seed(seed); np.random.seed(seed)
    model2 = ToeplitzCNN().to(DEVICE)
    # We need to re-train to get model for test inference
    # Retrain same seed
    opt2 = torch.optim.AdamW(model2.parameters(), lr=0.001, weight_decay=0.0005)
    crit2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw_val]).to(DEVICE))
    ds_tr2 = ToeplitzDataset(raw_tr, y_tr)
    dl_tr2 = torch.utils.data.DataLoader(ds_tr2, batch_size=64, shuffle=True,
                                          num_workers=0, pin_memory=True)
    model2.train()
    for ep in range(30):
        for X, y in dl_tr2:
            opt2.zero_grad()
            crit2(model2(X.to(DEVICE)), y.to(DEVICE)).backward()
            opt2.step()
    probs_test = np.array(infer(model2, raw_test))
    auc_test = float(roc_auc_score(y_test, probs_test))
    eer_test = eer_from_roc(y_test, probs_test)

    all_probs_eval.append(probs_eval)
    all_probs_test.append(probs_test)
    seed_results.append({'seed': seed, 'auc_18id': round(auc_eval,4),
                         'eer_18id': round(eer_eval,4),
                         'auc_9id': round(auc_test,4),
                         'eer_9id': round(eer_test,4),
                         'degenerate': bool(degen)})
    print(f"  seed={seed}  AUC-18id={auc_eval:.4f}  EER-18id={eer_eval*100:.1f}%  "
          f"AUC-9id={auc_test:.4f}  EER-9id={eer_test*100:.1f}%  "
          f"{'⚠ DEGEN' if degen else ''}  ({(time.time()-t1)/60:.1f}min)", flush=True)

# Detect degenerate overall
any_degen = any(r['degenerate'] for r in seed_results)
if any_degen:
    # Report score distribution
    mean_probs_eval = np.mean(all_probs_eval, axis=0)
    print(f"\n⚠  DEGENERATE OUTPUT DETECTED")
    print(f"  Score distribution (eval): mean={mean_probs_eval.mean():.4f} "
          f"std={mean_probs_eval.std():.4f} "
          f"min={mean_probs_eval.min():.4f} max={mean_probs_eval.max():.4f}")

aucs_eval = [r['auc_18id'] for r in seed_results]
aucs_test = [r['auc_9id']  for r in seed_results]
eers_eval = [r['eer_18id'] for r in seed_results]
eers_test = [r['eer_9id']  for r in seed_results]

mean_probs_eval = np.mean(all_probs_eval, axis=0)
mean_probs_test = np.mean(all_probs_test, axis=0)

# Per-generator AUC (eval)
pg_eval = {}
eval_real_mask = (y_eval == 0)
for m in METHODS:
    mmask = (y_eval == 1) & (df_eval['method'].to_numpy() == m)
    if not mmask.any(): continue
    yl = np.concatenate([np.zeros(eval_real_mask.sum()), np.ones(mmask.sum())])
    pl = np.concatenate([mean_probs_eval[eval_real_mask], mean_probs_eval[mmask]])
    pg_eval[m] = round(float(roc_auc_score(yl, pl)), 4)

# ── 5-fold CV on training set ─────────────────────────────────────────────────
print("\n5-fold CV on training set (by identity) ...", flush=True)
t_cv = time.time()
kf = KFold(n_splits=N_CV, shuffle=True, random_state=42)
tr_ids_arr = np.array(tr_ids)
cv_aucs = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(tr_ids_arr)):
    fold_tr_ids  = set(tr_ids_arr[tr_idx])
    fold_val_ids = set(tr_ids_arr[val_idx])

    fold_tr_mask  = df_tr['identity'].isin(fold_tr_ids).to_numpy()
    fold_val_mask = df_tr['identity'].isin(fold_val_ids).to_numpy()

    raw_f_tr  = [raw_tr[i] for i in np.where(fold_tr_mask)[0]]
    raw_f_val = [raw_tr[i] for i in np.where(fold_val_mask)[0]]
    y_f_tr  = y_tr[fold_tr_mask]
    y_f_val = y_tr[fold_val_mask]

    n_r = int((y_f_tr == 0).sum())
    n_f = int((y_f_tr == 1).sum())
    pw_f = n_r / max(n_f, 1)

    # Single seed for CV (seed=42 for stability)
    auc_f, _, _, _ = train_eval(raw_f_tr, y_f_tr, raw_f_val, y_f_val, pw_f, seed=42)
    cv_aucs.append(round(auc_f, 4))
    print(f"  fold {fold+1}/{N_CV}: val_ids={len(fold_val_ids)}  n_val={fold_val_mask.sum()}  "
          f"AUC={auc_f:.4f}", flush=True)

cv_mean = round(float(np.mean(cv_aucs)), 4)
cv_std  = round(float(np.std(cv_aucs)),  4)
print(f"  CV mean: {cv_mean:.4f} ± {cv_std:.4f}  ({(time.time()-t_cv)/60:.1f}min)")

# ── Report ─────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("TASK 1 — Toeplitz 2D CNN")
print("=" * 65)
print(f"  Params: {N_PARAMS:,}  (target: ~32K)")
print(f"  Train: {n_real_tr}r + {n_fake_tr}f  |  Eval-18id: {n_real_eval}r + {n_fake_eval}f  |"
      f"  Test-9id: {n_real_test}r + {n_fake_test}f")
print(f"  pw={pw_val:.4f}  seeds={SEEDS}")
print()
print(f"  Per-seed:")
for r in seed_results:
    print(f"    seed={r['seed']}  AUC-18id={r['auc_18id']:.4f}  EER-18id={r['eer_18id']*100:.1f}%"
          f"  AUC-9id={r['auc_9id']:.4f}  EER-9id={r['eer_9id']*100:.1f}%")
print(f"  Mean AUC-18id: {np.mean(aucs_eval):.4f} ± {np.std(aucs_eval):.4f}")
print(f"  Mean EER-18id: {np.mean(eers_eval)*100:.1f}%")
print(f"  Mean AUC-9id:  {np.mean(aucs_test):.4f} ± {np.std(aucs_test):.4f}")
print(f"  Mean EER-9id:  {np.mean(eers_test)*100:.1f}%")
print(f"  5-fold CV AUC: {cv_mean:.4f} ± {cv_std:.4f}")
print(f"  Degenerate: {any_degen}")
print()
print(f"  Per-generator AUC (18-id, mean-prob ensemble):")
for m, v in pg_eval.items():
    print(f"    {m:<20}  {v:.4f}")
print(f"\n  Runtime: {(time.time()-t_total)/60:.1f} min")

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'experiment': 'toeplitz_2d_cnn',
    'architecture': 'ToeplitzCNN_3block',
    'params': N_PARAMS,
    'target_params': 32000,
    'lr': 0.001, 'wd': 0.0005, 'dropout': 0.5, 'epochs': 30, 'batch': 64,
    'n_train_real': n_real_tr, 'n_train_fake': n_fake_tr,
    'n_eval_real': n_real_eval, 'n_eval_fake': n_fake_eval,
    'n_test_real': n_real_test, 'n_test_fake': n_fake_test,
    'pw': round(pw_val, 4),
    'train_ids': tr_ids, 'eval_ids': eval_ids, 'test_ids': test_ids,
    'seeds': SEEDS,
    'seed_results': seed_results,
    'mean_auc_18id': round(float(np.mean(aucs_eval)), 4),
    'std_auc_18id':  round(float(np.std(aucs_eval)),  4),
    'mean_eer_18id': round(float(np.mean(eers_eval)), 4),
    'mean_auc_9id':  round(float(np.mean(aucs_test)), 4),
    'std_auc_9id':   round(float(np.std(aucs_test)),  4),
    'mean_eer_9id':  round(float(np.mean(eers_test)),  4),
    'cv_5fold_aucs': cv_aucs,
    'cv_5fold_mean': cv_mean,
    'cv_5fold_std':  cv_std,
    'per_generator_auc_18id': pg_eval,
    'any_degenerate': any_degen,
    'score_stats_if_degen': (
        {'mean': round(float(mean_probs_eval.mean()),4),
         'std':  round(float(mean_probs_eval.std()),4),
         'min':  round(float(mean_probs_eval.min()),4),
         'max':  round(float(mean_probs_eval.max()),4)}
        if any_degen else None),
    'runtime_min': round((time.time()-t_total)/60, 1),
}
with open(OUT_JSON, 'w') as f:
    json.dump(out, f, indent=2)
print(f"  Saved → {OUT_JSON}")
