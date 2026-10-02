"""
Technique isolation — fourth condition: BOTH z-score AND augmentation.

Extends data/results/technique_isolation.json with a fourth condition.
Protocol matched exactly to run_technique_isolation.py:
  - StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on train split
  - Grouped by identity
  - 1D ResNet: lr=0.001, wd=0.0005, dropout=0.5, AdamW, 30 epochs, batch=64
  - Single seed=42 per fold
  - pos_weight recomputed per fold from training portion
  - Augmentation spec (identical to existing run):
      noise:  x += N(0, 0.1 × window_std)  per window, online
      scale:  x *= U(0.8, 1.2)             per window, online
  - Z-score applied offline (before training loop), then augmentation applied online

Fold identity check: folds are re-generated from the same splitter call.
If fold_val_ids match existing JSON exactly, fold_identical=True is reported.

Output: data/results/technique_isolation_both.json
"""

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve

# ── Paths (configurable; see README) ─────────────────────────────────────────
import argparse as _argparse
_ap = _argparse.ArgumentParser(description=(__doc__ or '').strip().split('\n')[0])
_ap.add_argument('--data-root', default='data',
                 help='data folder laid out as described in the README (default: ./data)')
_ap.add_argument('--out-dir', default=None,
                 help='folder for result JSONs (default: <data-root>/results)')
_args = _ap.parse_args()

DATA_ROOT  = Path(_args.data_root)
OUT_DIR = Path(_args.out_dir) if _args.out_dir else DATA_ROOT / 'results'
SPLIT_FILE = DATA_ROOT / 'dataset_split_full59.csv'
EXISTING_JSON = OUT_DIR / 'technique_isolation.json'
OUT_JSON   = OUT_DIR / 'technique_isolation_both.json'

N_FOLDS  = 5
LR, WD, DROPOUT = 0.001, 0.0005, 0.5
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Preprocessing helpers ──────────────────────────────────────────────────────
def apply_zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)

def apply_augment_batch(Xb_raw, rng):
    Xb = Xb_raw.copy()
    for i in range(len(Xb)):
        sig_std = Xb[i].std() + 1e-6
        noise   = rng.normal(0, 0.1 * sig_std, Xb[i].shape).astype(np.float32)
        scale   = rng.uniform(0.8, 1.2)
        Xb[i]   = Xb[i] * scale + noise
    return Xb

# ── ResNet (identical to run_technique_isolation.py) ──────────────────────────
class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv1    = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm1d(out_ch)
        self.conv2    = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm1d(out_ch)
        self.drop     = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = (nn.Sequential(
                            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                            nn.BatchNorm1d(out_ch))
                         if stride != 1 or in_ch != out_ch else nn.Identity())
    def forward(self, x):
        return F.relu(self.bn2(self.conv2(self.drop(F.relu(self.bn1(self.conv1(x)))))) + self.shortcut(x))

class Waveform1DResNet(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        c1, c2, c3 = 32, 64, 128
        self.stem   = nn.Sequential(nn.Conv1d(1,c1,7,padding=3,bias=False),
                                    nn.BatchNorm1d(c1), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.stage1 = nn.Sequential(BasicBlock1D(c1,c1,dropout=dropout), BasicBlock1D(c1,c1,dropout=dropout))
        self.stage2 = nn.Sequential(BasicBlock1D(c1,c2,stride=2,dropout=dropout), BasicBlock1D(c2,c2,dropout=dropout))
        self.stage3 = nn.Sequential(BasicBlock1D(c2,c3,stride=2,dropout=dropout), BasicBlock1D(c3,c3,dropout=dropout))
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Linear(c3, 1)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.fc(self.gap(self.stage3(self.stage2(self.stage1(self.stem(x))))).squeeze(-1)).squeeze(-1)

PARAMS = sum(p.numel() for p in Waveform1DResNet(dropout=DROPOUT).parameters())
assert PARAMS == 240161, f"Unexpected param count: {PARAMS}"

# ── Load training data ─────────────────────────────────────────────────────────
print(f"Device:  {DEVICE}")
print(f"Params:  {PARAMS:,}")
print(f"Loading training waveforms (raw) ...", flush=True)
t0 = time.time()
split_df  = pd.read_csv(SPLIT_FILE)
train_df  = split_df[split_df['split'] == 'train'].reset_index(drop=True)

X_raw  = np.stack([np.load(p).astype(np.float32) for p in train_df['path']])
y      = np.array([float(c == 'fake') for c in train_df['class']], dtype=np.float32)
groups = train_df['identity'].to_numpy()
print(f"  {len(X_raw)} windows in {time.time()-t0:.1f}s")
print(f"  Real: {int((y==0).sum())}  Fake: {int((y==1).sum())}")
print()

# Apply z-score to ALL windows (offline, matching 'zscore' condition exactly)
X_zscored = np.stack([apply_zscore(x) for x in X_raw])

# ── Cross-validation (identical splitter call) ────────────────────────────────
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Load existing results to compare fold assignments
with open(EXISTING_JSON) as f:
    existing = json.load(f)
existing_val_ids = existing['conditions']['baseline']['fold_val_ids']  # reference

fold_aucs, fold_eers, fold_pws, fold_val_ids = [], [], [], []
fold_id_match = []

t_total = time.time()

for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_raw, y, groups)):
    va_ids = sorted(np.unique(groups[va_idx]).tolist(), key=lambda x: int(x[2:]))
    nr_tr  = int((y[tr_idx] == 0).sum()); nf_tr = int((y[tr_idx] == 1).sum())
    nr_va  = int((y[va_idx] == 0).sum()); nf_va = int((y[va_idx] == 1).sum())
    pw_val = nr_tr / max(nf_tr, 1)

    match = (va_ids == existing_val_ids[fold])
    fold_id_match.append(match)

    print(f"{'='*65}")
    print(f"Fold {fold+1}/{N_FOLDS}  (fold_ids_match_existing={match})")
    print(f"  Train: {nr_tr} real + {nf_tr} fake  pos_weight={pw_val:.4f}")
    print(f"  Val:   {nr_va} real + {nf_va} fake  ids={va_ids}")
    print(f"{'='*65}", flush=True)

    # "both": z-scored waveforms + online augmentation
    X_tr, y_tr = X_zscored[tr_idx], y[tr_idx]
    X_va, y_va = X_zscored[va_idx], y[va_idx]

    seed = 42
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = Waveform1DResNet(dropout=DROPOUT).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    pw    = torch.tensor([pw_val]).to(DEVICE)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pw)

    yt = torch.from_numpy(y_tr).to(DEVICE)

    model.train()
    for epoch in range(30):
        perm = torch.randperm(len(X_tr))  # CPU permutation; index into numpy array
        for s in range(0, len(X_tr), 64):
            ix  = perm[s:s+64].numpy()
            Xb  = apply_augment_batch(X_tr[ix], rng)        # augment z-scored batch
            xb_t = torch.from_numpy(Xb).unsqueeze(1).to(DEVICE)
            opt.zero_grad()
            crit(model(xb_t).reshape(-1), yt[perm[s:s+64].to(DEVICE)]).backward()
            opt.step()

    model.eval()
    Xv = torch.from_numpy(X_va).unsqueeze(1)
    probs = []
    with torch.no_grad():
        for s in range(0, len(Xv), 512):
            probs.extend(
                torch.sigmoid(model(Xv[s:s+512].to(DEVICE)).reshape(-1)).cpu().numpy())
    probs = np.array(probs)

    auc = float(roc_auc_score(y_va, probs))
    fpr, tpr, _ = roc_curve(y_va, probs)
    fnr = 1 - tpr
    i   = np.argmin(np.abs(fnr - fpr))
    eer = float((fpr[i] + fnr[i]) / 2)

    fold_aucs.append(auc)
    fold_eers.append(eer)
    fold_pws.append(round(pw_val, 4))
    fold_val_ids.append(va_ids)

    print(f"  [both]  AUC={auc:.4f}  EER={eer*100:.1f}%", flush=True)
    print()

elapsed = (time.time() - t_total) / 60

# ── Summary ────────────────────────────────────────────────────────────────────
mean_auc = float(np.mean(fold_aucs))
std_auc  = float(np.std(fold_aucs))
mean_eer = float(np.mean(fold_eers))

baseline_mean = existing['conditions']['baseline']['mean_auc']
zscore_mean   = existing['conditions']['zscore']['mean_auc']
delta_vs_baseline = mean_auc - baseline_mean
delta_vs_zscore   = mean_auc - zscore_mean
folds_match_all   = all(fold_id_match)

print("=" * 65)
print("  TECHNIQUE ISOLATION — ALL FOUR CONDITIONS")
print("=" * 65)
print(f"  {'Condition':<18}  {'CV AUC':>14}  {'Δ vs baseline':>14}  {'EER':>6}")
print(f"  {'-'*60}")

existing_rows = [
    ('baseline',    existing['conditions']['baseline']['mean_auc'],  existing['conditions']['baseline']['std_auc'],  existing['conditions']['baseline']['mean_eer'],  0.0),
    ('zscore only', existing['conditions']['zscore']['mean_auc'],    existing['conditions']['zscore']['std_auc'],    existing['conditions']['zscore']['mean_eer'],    existing['conditions']['zscore']['delta_vs_baseline']),
    ('augment only',existing['conditions']['augment']['mean_auc'],   existing['conditions']['augment']['std_auc'],   existing['conditions']['augment']['mean_eer'],   existing['conditions']['augment']['delta_vs_baseline']),
]
for name, mu, sd, eer, delta in existing_rows:
    delta_str = "—" if name == 'baseline' else f"{delta:+.4f}"
    print(f"  {name:<18}  {mu:.4f}±{sd:.4f}  {delta_str:>14}  {eer*100:>5.1f}%")
print(f"  {'both (this run)':<18}  {mean_auc:.4f}±{std_auc:.4f}  {delta_vs_baseline:>+14.4f}  {mean_eer*100:>5.1f}%")

print()
print(f"  Fold assignments identical to existing: {folds_match_all}")
print(f"  Augment adds on top of z-score:         {delta_vs_zscore:+.4f}")
print(f"  Runtime: {elapsed:.1f} min")

# ── Save ───────────────────────────────────────────────────────────────────────
out = {
    'experiment': 'technique_isolation_both',
    'description': 'Fourth condition (both z-score and augmentation) added to technique_isolation.json protocol',
    'split_file': str(SPLIT_FILE),
    'existing_json': str(EXISTING_JSON),
    'n_folds': N_FOLDS,
    'seed': 42,
    'arch': '1D ResNet',
    'params': PARAMS,
    'lr': LR, 'wd': WD, 'dropout': DROPOUT, 'epochs': 30,
    'batch_size': 64,
    'n_train_total': len(train_df),
    'augmentation_spec': {
        'noise': 'N(0, 0.1 × window_std) additive per window, applied to z-scored signal',
        'scale': 'U(0.8, 1.2) multiplicative per window',
        'applied': 'online per-batch during training, after offline z-score',
    },
    'protocol_matched': True,
    'fold_assignments_identical': folds_match_all,
    'fold_id_match_per_fold': fold_id_match,
    'conditions_reference': {
        'baseline':     {'source': str(EXISTING_JSON), 'mean_auc': baseline_mean, 'std_auc': existing['conditions']['baseline']['std_auc'], 'mean_eer': existing['conditions']['baseline']['mean_eer'], 'delta_vs_baseline': 0.0},
        'zscore_only':  {'source': str(EXISTING_JSON), 'mean_auc': zscore_mean,   'std_auc': existing['conditions']['zscore']['std_auc'],   'mean_eer': existing['conditions']['zscore']['mean_eer'],   'delta_vs_baseline': round(existing['conditions']['zscore']['delta_vs_baseline'], 4)},
        'augment_only': {'source': str(EXISTING_JSON), 'mean_auc': existing['conditions']['augment']['mean_auc'], 'std_auc': existing['conditions']['augment']['std_auc'], 'mean_eer': existing['conditions']['augment']['mean_eer'], 'delta_vs_baseline': round(existing['conditions']['augment']['delta_vs_baseline'], 4)},
    },
    'condition_both': {
        'fold_aucs': fold_aucs,
        'fold_eers': fold_eers,
        'fold_pw': fold_pws,
        'fold_val_ids': fold_val_ids,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'mean_eer': mean_eer,
        'delta_vs_baseline': round(delta_vs_baseline, 4),
        'delta_vs_zscore_only': round(delta_vs_zscore, 4),
    },
    'augment_adds_on_top_of_zscore': round(delta_vs_zscore, 4),
    'runtime_min': round(elapsed, 1),
}
with open(OUT_JSON, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved → {OUT_JSON}")

print("\n\nFINAL JSON:")
print(json.dumps(out, indent=2))
