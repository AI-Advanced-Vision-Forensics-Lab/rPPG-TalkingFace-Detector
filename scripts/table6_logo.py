"""
Leave-One-Generator-Out (LOGO) cross-method generalization experiment.
Paper 1 — rPPG only, 59-identity full corpus (dataset_split_full59.csv).

For each held-out generator:
  Train:  1,675 real train + other-6-generators train fakes
  Eval:   696 real eval + held-out generator eval fakes
  Arch:   1D ResNet  (lr=0.001, wd=0.0005, dropout=0.5)
  Seeds:  42, 7, 123, 999, 2024

Transfer gap = LOGO AUC − combined-model AUC (from full59_per_generator_breakdown.json)

Output: data/results/logo_generalization.json
"""

import sys, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

# ── Paths ─────────────────────────────────────────────────────────────────────
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
BREAKDOWN  = OUT_DIR / 'full59_per_generator_breakdown.json'
OUT_JSON   = OUT_DIR / 'logo_generalization.json'

METHODS = ['AniTalker','EDTalk','EchoMimic','FLOAT','IP_LAP','Real3DPortrait','SadTalker']
SEEDS   = [42, 7, 123, 999, 2024]
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Canonical BEST_CONFIGS — ResNet only
LR, WD, DROPOUT = 0.001, 0.0005, 0.5

EXPECTED_REAL_TRAIN = 1675
EXPECTED_REAL_EVAL  = 696
EXPECTED_EVAL_IDS   = 18

# ── Helpers ───────────────────────────────────────────────────────────────────
def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    i = np.argmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ── ResNet ────────────────────────────────────────────────────────────────────
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

# ── Load split ────────────────────────────────────────────────────────────────
print(f"Device: {DEVICE}")
print(f"Split:  {SPLIT_FILE}")
print(f"Arch:   1D ResNet  lr={LR} wd={WD} dropout={DROPOUT}")
print()

split_df = pd.read_csv(SPLIT_FILE)
params = count_params(Waveform1DResNet(dropout=DROPOUT))
print(f"Model params: {params:,}")
print()

# Validate global counts
mask_tr = split_df['split'] == 'train'
mask_ev = split_df['split'].isin(['val','test'])
n_real_tr_global = int(((split_df['class']=='real') & mask_tr).sum())
n_real_ev_global = int(((split_df['class']=='real') & mask_ev).sum())
n_ev_ids_global  = split_df.loc[mask_ev, 'identity'].nunique()

if n_real_tr_global != EXPECTED_REAL_TRAIN:
    sys.exit(f"FAIL: expected {EXPECTED_REAL_TRAIN} train real, got {n_real_tr_global}")
if n_real_ev_global != EXPECTED_REAL_EVAL:
    sys.exit(f"FAIL: expected {EXPECTED_REAL_EVAL} eval real, got {n_real_ev_global}")
if n_ev_ids_global != EXPECTED_EVAL_IDS:
    sys.exit(f"FAIL: expected {EXPECTED_EVAL_IDS} eval ids, got {n_ev_ids_global}")

ev_ids_sorted = sorted(split_df.loc[mask_ev, 'identity'].unique().tolist(),
                       key=lambda x: int(x[2:]))
print(f"Global checks: {n_real_tr_global} train real ✓ | {n_real_ev_global} eval real ✓ | "
      f"{n_ev_ids_global} eval ids ✓")
print(f"Eval ids: {ev_ids_sorted}")
print()

# ── Load combined-model breakdown for transfer gap ────────────────────────────
with open(BREAKDOWN) as f:
    breakdown = json.load(f)
combined_aucs = {m: breakdown['1D ResNet'][m]['mean_auc'] for m in METHODS}

# ── Load all waveforms once ────────────────────────────────────────────────────
print("Loading waveforms ...", flush=True)
t0 = time.time()

def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)

paths   = split_df['path'].tolist()
classes = split_df['class'].tolist()
methods = split_df['method'].tolist()   # NaN for real
splits  = split_df['split'].tolist()
identities = split_df['identity'].tolist()

X_all = np.stack([zscore(np.load(p).astype(np.float32)) for p in paths])
y_all = np.array([float(c == 'fake') for c in classes], dtype=np.float32)
m_all = np.array(methods)   # strings, NaN for real → compare with ==
s_all = np.array(splits)
id_all = np.array(identities)

print(f"  Loaded {len(X_all)} windows in {time.time()-t0:.1f}s", flush=True)
print()

# Boolean index arrays
is_train = (s_all == 'train')
is_eval  = np.isin(s_all, ['val', 'test'])
is_real  = (y_all == 0)
is_fake  = (y_all == 1)

# Real train / real eval (constant across all folds)
X_real_tr = X_all[is_train & is_real]
y_real_tr = y_all[is_train & is_real]
X_real_ev = X_all[is_eval  & is_real]
y_real_ev = y_all[is_eval  & is_real]

# ── Training function ─────────────────────────────────────────────────────────
def run_seed(X_tr, y_tr, X_ev, y_ev, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n_real = int((y_tr == 0).sum());  n_fake = int((y_tr == 1).sum())
    pw = torch.tensor([n_real / max(n_fake, 1)]).to(DEVICE)

    model = Waveform1DResNet(dropout=DROPOUT).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pw)

    Xt = torch.from_numpy(X_tr).unsqueeze(1).to(DEVICE)
    yt = torch.from_numpy(y_tr).to(DEVICE)

    model.train()
    for _ in range(30):
        perm = torch.randperm(len(Xt), device=DEVICE)
        for s in range(0, len(Xt), 64):
            ix = perm[s:s+64]; opt.zero_grad()
            crit(model(Xt[ix]).reshape(-1), yt[ix]).backward()
            opt.step()

    model.eval()
    Xe = torch.from_numpy(X_ev).unsqueeze(1)
    probs = []
    with torch.no_grad():
        for s in range(0, len(Xe), 512):
            probs.extend(torch.sigmoid(model(Xe[s:s+512].to(DEVICE)).reshape(-1)).cpu().numpy())
    probs = np.array(probs)
    auc = float(roc_auc_score(y_ev, probs))
    eer = compute_eer(y_ev, probs)
    return auc, eer, n_real, n_fake, float(pw.item())

# ── Resume support ────────────────────────────────────────────────────────────
if OUT_JSON.exists():
    with open(OUT_JSON) as f:
        results = json.load(f)
    print(f"Resuming — {len(results)} folds already done")
else:
    results = {}

# ── Main LOGO loop ────────────────────────────────────────────────────────────
print("=" * 70)
print("  LOGO Generalization — 1D ResNet, 7 folds")
print("=" * 70)

t_total = time.time()
for held_out in METHODS:
    if held_out in results:
        print(f"\n[held-out: {held_out}] already done — skipping")
        continue

    # Training fakes: all generators EXCEPT held_out
    tr_fake_mask = is_train & is_fake & (m_all != held_out)
    X_tr_fake = X_all[tr_fake_mask]
    y_tr_fake = y_all[tr_fake_mask]

    # Eval fakes: ONLY held_out generator
    ev_fake_mask = is_eval & is_fake & (m_all == held_out)
    X_ev_fake = X_all[ev_fake_mask]
    y_ev_fake = y_all[ev_fake_mask]

    # Concatenate
    X_tr = np.concatenate([X_real_tr, X_tr_fake])
    y_tr = np.concatenate([y_real_tr, y_tr_fake])
    X_ev = np.concatenate([X_real_ev, X_ev_fake])
    y_ev = np.concatenate([y_real_ev, y_ev_fake])

    nr_tr = int((y_tr == 0).sum());  nf_tr = int((y_tr == 1).sum())
    nr_ev = int((y_ev == 0).sum());  nf_ev = int((y_ev == 1).sum())
    ev_fake_ids = sorted(id_all[ev_fake_mask & is_eval].tolist(), key=lambda x: int(x[2:]))
    # unique ids in eval (real + fake)
    ev_ids_fold = sorted(np.unique(id_all[is_eval]).tolist(), key=lambda x: int(x[2:]))

    # Expected: train real unchanged, eval real unchanged
    if nr_tr != EXPECTED_REAL_TRAIN:
        sys.exit(f"FAIL [{held_out}]: expected {EXPECTED_REAL_TRAIN} train real, got {nr_tr}")
    if nr_ev != EXPECTED_REAL_EVAL:
        sys.exit(f"FAIL [{held_out}]: expected {EXPECTED_REAL_EVAL} eval real, got {nr_ev}")

    # Expected train fake = total_train_fake - held_out's train fakes
    total_tr_fake = int((is_train & is_fake).sum())
    held_tr_fake  = int((is_train & is_fake & (m_all == held_out)).sum())
    expected_nf_tr = total_tr_fake - held_tr_fake
    if nf_tr != expected_nf_tr:
        sys.exit(f"FAIL [{held_out}]: expected {expected_nf_tr} train fakes, got {nf_tr}")

    pw_val = nr_tr / max(nf_tr, 1)

    print(f"\n{'='*65}")
    print(f"  Held-out: {held_out}")
    print(f"{'='*65}")
    print(f"  Train: {nr_tr} real + {nf_tr} fake  (6 generators)")
    print(f"  Eval:  {nr_ev} real + {nf_ev} fake  ({len(ev_fake_ids)} held-out ids: {ev_fake_ids})")
    print(f"  Eval identities ({len(ev_ids_fold)}): {ev_ids_fold}")
    print(f"  pos_weight = {nr_tr}/{nf_tr} = {pw_val:.6f}")

    seed_aucs, seed_eers = [], []
    for seed in SEEDS:
        auc, eer, _, _, pw_s = run_seed(X_tr, y_tr, X_ev, y_ev, seed)
        seed_aucs.append(auc);  seed_eers.append(eer)
        print(f"    seed={seed:>4}  AUC={auc:.4f}  EER={eer*100:.1f}%", flush=True)

    mean_auc = float(np.mean(seed_aucs));  std_auc = float(np.std(seed_aucs))
    mean_eer = float(np.mean(seed_eers))
    comb_auc = combined_aucs[held_out]
    gap      = round(mean_auc - comb_auc, 4)

    print(f"  → LOGO {mean_auc:.4f}±{std_auc:.4f}  EER={mean_eer*100:.1f}%  "
          f"combined={comb_auc:.4f}  gap={gap:+.4f}", flush=True)

    results[held_out] = {
        'held_out':             held_out,
        'n_train_real':         nr_tr,
        'n_train_fake':         nf_tr,
        'n_eval_real':          nr_ev,
        'n_eval_fake':          nf_ev,
        'n_eval_ids':           len(ev_ids_fold),
        'eval_ids':             ev_ids_fold,
        'pos_weight':           round(pw_val, 6),
        'seed_aucs':            seed_aucs,
        'seed_eers':            seed_eers,
        'mean_auc':             mean_auc,
        'std_auc':              std_auc,
        'mean_eer':             mean_eer,
        'combined_model_auc':   comb_auc,
        'transfer_gap':         gap,
        'lr': LR, 'wd': WD, 'dropout': DROPOUT, 'params': params,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("=" * 80)
print("  LOGO Generalization Summary — 1D ResNet (59-id full corpus)")
print("=" * 80)
print(f"  {'Held-out':<18}  {'LOGO AUC':>10}  {'EER':>6}  {'Combined':>9}  {'Gap':>8}  pos_weight")
print("  " + "-" * 72)

logo_aucs = []
for m in METHODS:
    r = results[m]
    logo_aucs.append(r['mean_auc'])
    print(f"  {m:<18}  {r['mean_auc']:.4f}±{r['std_auc']:.4f}  "
          f"{r['mean_eer']*100:>5.1f}%  "
          f"{r['combined_model_auc']:>8.4f}  "
          f"{r['transfer_gap']:>+8.4f}  "
          f"{r['pos_weight']:.6f}")

print("  " + "-" * 72)
print(f"  {'Mean LOGO AUC':<18}  {np.mean(logo_aucs):.4f}")
print()
print(f"Total runtime: {(time.time()-t_total)/60:.1f} min")
print(f"Saved → {OUT_JSON}")
