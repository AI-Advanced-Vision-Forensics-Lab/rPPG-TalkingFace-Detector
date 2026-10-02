"""
Phase 2 — Three experiments on the 59-identity full Celeb-DF++ TF corpus.

(a) Combined model: 1D ResNet, Transformer, CNN trained on all 7 generators combined.
    Eval: (i) 18-id eval set, (ii) 9-id test-only.
    Per-identity AUC across 18 eval ids at seed 42 for ResNet (reported in paper).
    Seeds: 42, 7, 123, 999, 2024.
    Saved → data/results/full59_combined_model.json

(b) Per-generator breakdown: the ResNet from (a), evaluated per-generator.
    No re-training — computed inline during (a) ResNet runs.
    Saved → data/results/full59_per_generator_breakdown.json

(c) Isolated per-method: ResNet + Transformer per generator.
    Train: real_train + that_generator_train_fakes.
    Eval:  real_eval  + that_generator_eval_fakes (18-id).
    Saved → data/results/full59_per_method_isolated.json

Canonical BEST_CONFIGS:
  ResNet:      lr=0.001,  wd=0.0005, dropout=0.5
  Transformer: lr=0.001,  wd=0.0001, dropout=0.3
  CNN:         lr=0.0005, wd=0.001,  dropout=0.3

All runs: AdamW, 30 epochs, batch=64, BCEWithLogitsLoss(pos_weight=n_real/n_fake).
Fail loudly on unexpected counts or missing configs.
"""

import sys, json, time
from pathlib import Path
from collections import defaultdict

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
RESULTS    = OUT_DIR
RESULTS.mkdir(exist_ok=True)

OUT_COMBINED  = RESULTS / 'full59_combined_model.json'
OUT_BREAKDOWN = RESULTS / 'full59_per_generator_breakdown.json'
OUT_ISOLATED  = RESULTS / 'full59_per_method_isolated.json'

METHODS = ['AniTalker','EDTalk','EchoMimic','FLOAT','IP_LAP','Real3DPortrait','SadTalker']
SEEDS   = [42, 7, 123, 999, 2024]
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BEST_CONFIGS = {
    '1D ResNet':   dict(lr=0.001,  wd=0.0005, dropout=0.5),
    'Transformer': dict(lr=0.001,  wd=0.0001, dropout=0.3),
    '1D CNN':      dict(lr=0.0005, wd=0.001,  dropout=0.3),
}

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

# ── Model definitions ─────────────────────────────────────────────────────────
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

class WaveformTransformer(nn.Module):
    def __init__(self, patch_size=8, d_model=64, nhead=4, num_layers=2,
                 mlp_dim=128, dropout=0.3):
        super().__init__()
        n_patches = 160 // patch_size
        self.patch_embed = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, mlp_dim, dropout,
                                         activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(d_model, 1)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.patch_embed(x).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], 1) + self.pos_embed
        return self.fc(self.dropout(self.norm(self.encoder(x)[:, 0]))).squeeze(-1)

class Waveform1DCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3, bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(2))
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2), nn.Dropout(dropout))
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1, bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(dropout))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(128, 1)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.fc(self.gap(self.block3(self.block2(self.block1(x)))).squeeze(-1)).squeeze(-1)

ARCH_CLS = {
    '1D ResNet':   Waveform1DResNet,
    'Transformer': WaveformTransformer,
    '1D CNN':      Waveform1DCNN,
}

# ── Load all waveforms into memory once ───────────────────────────────────────
def load_all(df):
    """Returns parallel arrays: X (N,160), y (N,), identity list, method list, split list."""
    X, y, ids, methods, splits = [], [], [], [], []
    for _, row in df.iterrows():
        w = np.load(row['path']).astype(np.float32)
        X.append(zscore(w))
        y.append(float(row['class'] == 'fake'))
        ids.append(row['identity'])
        methods.append(row['method'] if pd.notna(row['method']) else None)
        splits.append(row['split'])
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            ids, methods, splits)

# ── Training + evaluation core ────────────────────────────────────────────────
def train_and_eval(model_cls, dropout, lr, wd, X_tr, y_tr, eval_sets, seed,
                   arch_name, context_str):
    """
    eval_sets: dict of name → (X_ev, y_ev, extra_meta)
    Returns dict of name → {'probs': np.array, 'y_true': np.array, 'meta': extra_meta}
    """
    torch.manual_seed(seed); np.random.seed(seed)
    n_real_tr = int((y_tr == 0).sum());  n_fake_tr = int((y_tr == 1).sum())
    pw = torch.tensor([n_real_tr / max(n_fake_tr, 1)]).to(DEVICE)
    print(f"    pos_weight = {n_real_tr}/{n_fake_tr} = {pw.item():.6f}")

    model = model_cls(dropout=dropout).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pw)

    Xt = torch.from_numpy(X_tr).unsqueeze(1).to(DEVICE)
    yt = torch.from_numpy(y_tr).to(DEVICE)

    model.train()
    for epoch in range(30):
        perm = torch.randperm(len(Xt), device=DEVICE)
        for s in range(0, len(Xt), 64):
            ix = perm[s:s+64]; opt.zero_grad()
            crit(model(Xt[ix]).reshape(-1), yt[ix]).backward()
            opt.step()

    model.eval()
    results = {}
    for name, (X_ev, y_ev, meta) in eval_sets.items():
        Xe = torch.from_numpy(X_ev).unsqueeze(1)
        probs = []
        with torch.no_grad():
            for s in range(0, len(Xe), 512):
                probs.extend(torch.sigmoid(model(Xe[s:s+512].to(DEVICE)).reshape(-1)).cpu().numpy())
        results[name] = {'probs': np.array(probs), 'y_true': y_ev, 'meta': meta}
    return results, n_real_tr, n_fake_tr, float(pw.item())

def auc_eer(y_true, probs):
    auc = float(roc_auc_score(y_true, probs))
    eer = compute_eer(y_true, probs)
    return auc, eer

# ─────────────────────────────────────────────────────────────────────────────
print(f"Device: {DEVICE}")
print(f"Split:  {SPLIT_FILE}")
print()

# ── Load split ────────────────────────────────────────────────────────────────
print("Loading split ...", flush=True)
split_df = pd.read_csv(SPLIT_FILE)

print("Loading waveforms into memory ...", flush=True)
t0_load = time.time()
X_all, y_all, ids_all, methods_all, splits_all = load_all(split_df)
ids_all     = np.array(ids_all)
methods_all = np.array(methods_all)
splits_all  = np.array(splits_all)
y_all       = y_all.astype(np.float32)
print(f"  Loaded {len(X_all)} windows in {time.time()-t0_load:.1f}s", flush=True)

# Verify expected counts
mask_tr = (splits_all == 'train')
mask_ev = np.isin(splits_all, ['val', 'test'])
mask_te = (splits_all == 'test')

n_real_tr = int(((y_all == 0) & mask_tr).sum())
n_real_ev = int(((y_all == 0) & mask_ev).sum())
n_ev_ids  = len(np.unique(ids_all[mask_ev]))

if n_real_tr != EXPECTED_REAL_TRAIN:
    sys.exit(f"FAIL: expected {EXPECTED_REAL_TRAIN} train real, got {n_real_tr}")
if n_real_ev != EXPECTED_REAL_EVAL:
    sys.exit(f"FAIL: expected {EXPECTED_REAL_EVAL} eval real, got {n_real_ev}")
if n_ev_ids != EXPECTED_EVAL_IDS:
    sys.exit(f"FAIL: expected {EXPECTED_EVAL_IDS} eval identities, got {n_ev_ids}")

ev_ids_sorted = sorted(np.unique(ids_all[mask_ev]).tolist(), key=lambda x: int(x[2:]))
te_ids_sorted = sorted(np.unique(ids_all[mask_te]).tolist(), key=lambda x: int(x[2:]))

print(f"  Train real: {n_real_tr} ✓  |  Eval real: {n_real_ev} ✓  |  Eval ids: {n_ev_ids} ✓")
print()

# Shared data subsets
X_tr_all = X_all[mask_tr];  y_tr_all = y_all[mask_tr]
X_ev_all = X_all[mask_ev];  y_ev_all = y_all[mask_ev]
X_te_all = X_all[mask_te];  y_te_all = y_all[mask_te]
ids_ev   = ids_all[mask_ev]
ids_te   = ids_all[mask_te]
meth_ev  = methods_all[mask_ev]

n_fake_tr_all = int((y_tr_all == 1).sum())
print(f"  Train: {n_real_tr} real + {n_fake_tr_all} fake  (all generators combined)")
print(f"  Eval (18-id): {n_real_ev} real + {int((y_ev_all==1).sum())} fake")
print(f"  Test (9-id):  {int((y_te_all==0).sum())} real + {int((y_te_all==1).sum())} fake")
print()

# Per-method eval masks (for breakdown)
meth_ev_masks = {m: (meth_ev == m) | (y_ev_all == 0) for m in METHODS}  # real + that method's fakes
# More precisely: real eval OR fake eval from that method
def per_method_eval_mask(method):
    return (y_ev_all == 0) | ((y_ev_all == 1) & (meth_ev == method))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE (a) + (b): Combined model, all architectures
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  PHASE (a)+(b): Combined model — all 7 generators")
print("=" * 70)

# Load existing results if resuming
if OUT_COMBINED.exists():
    with open(OUT_COMBINED) as f:
        combined_results = json.load(f)
    print(f"  Resuming — {len(combined_results)} architectures already done")
else:
    combined_results = {}

if OUT_BREAKDOWN.exists():
    with open(OUT_BREAKDOWN) as f:
        breakdown_results = json.load(f)
else:
    breakdown_results = {}

ARCHS_AB = ['1D ResNet', 'Transformer', '1D CNN']

for arch_name in ARCHS_AB:
    if arch_name in combined_results and arch_name in breakdown_results:
        print(f"\n[{arch_name}] already done — skipping")
        continue

    cfg     = BEST_CONFIGS[arch_name]
    cls     = ARCH_CLS[arch_name]
    dropout = cfg['dropout']; lr = cfg['lr']; wd = cfg['wd']
    params  = count_params(cls(dropout=dropout))

    print(f"\n[{arch_name}]  lr={lr} wd={wd} dropout={dropout}  params={params:,}")
    print(f"  Train: {n_real_tr} real + {n_fake_tr_all} fake")
    print(f"  Eval (18-id): {n_real_ev} real + {int((y_ev_all==1).sum())} fake  "
          f"ids: {ev_ids_sorted}")
    print(f"  Eval (9-id test): {int((y_te_all==0).sum())} real + {int((y_te_all==1).sum())} fake  "
          f"ids: {te_ids_sorted}")

    arch_combined  = {'params': params, 'lr': lr, 'wd': wd, 'dropout': dropout,
                      'n_train_real': n_real_tr, 'n_train_fake': n_fake_tr_all,
                      'n_eval_real': n_real_ev, 'n_eval_fake': int((y_ev_all==1).sum()),
                      'n_test_real': int((y_te_all==0).sum()), 'n_test_fake': int((y_te_all==1).sum()),
                      'eval_ids': ev_ids_sorted, 'test_ids': te_ids_sorted,
                      'seeds': {}}
    arch_breakdown = {'params': params, 'seeds': {}}

    t_arch = time.time()
    aucs_ev, eers_ev, aucs_te, eers_te = [], [], [], []

    for seed in SEEDS:
        print(f"  seed={seed:>4}", flush=True)
        is_resnet_42 = (arch_name == '1D ResNet' and seed == 42)

        # Build eval sets
        eval_sets = {
            '18id': (X_ev_all, y_ev_all, None),
            '9id':  (X_te_all, y_te_all, None),
        }
        for m in METHODS:
            mask = per_method_eval_mask(m)
            eval_sets[f'method_{m}'] = (X_ev_all[mask], y_ev_all[mask], None)
        if is_resnet_42:
            # Per-identity: pass identity array as meta
            eval_sets['per_id'] = (X_ev_all, y_ev_all, ids_ev)

        run_out, nr, nf, pw_val = train_and_eval(
            cls, dropout, lr, wd, X_tr_all, y_tr_all, eval_sets, seed,
            arch_name, f"{arch_name} seed={seed}")

        # 18-id eval
        auc_ev, eer_ev = auc_eer(run_out['18id']['y_true'], run_out['18id']['probs'])
        # 9-id test eval
        auc_te, eer_te = auc_eer(run_out['9id']['y_true'],  run_out['9id']['probs'])
        aucs_ev.append(auc_ev); eers_ev.append(eer_ev)
        aucs_te.append(auc_te); eers_te.append(eer_te)
        print(f"         18-id AUC={auc_ev:.4f} EER={eer_ev*100:.1f}%  |  "
              f"9-id AUC={auc_te:.4f} EER={eer_te*100:.1f}%", flush=True)

        seed_entry = {
            'pos_weight': pw_val,
            'n_real_train': nr, 'n_fake_train': nf,
            'auc_18id': auc_ev, 'eer_18id': eer_ev,
            'auc_9id':  auc_te, 'eer_9id':  eer_te,
        }

        # Per-generator breakdown
        breakdown_seed = {'pos_weight': pw_val}
        for m in METHODS:
            r = run_out[f'method_{m}']
            m_auc, m_eer = auc_eer(r['y_true'], r['probs'])
            nm_real = int((r['y_true'] == 0).sum())
            nm_fake = int((r['y_true'] == 1).sum())
            nm_ids  = len(np.unique(ids_ev[per_method_eval_mask(m)]))
            breakdown_seed[m] = {'auc': m_auc, 'eer': m_eer,
                                 'n_eval_real': nm_real, 'n_eval_fake': nm_fake}
            if seed == 42:
                print(f"         {m:<16} AUC={m_auc:.4f} EER={m_eer*100:.1f}%  "
                      f"({nm_real}r+{nm_fake}f)", flush=True)
        arch_breakdown['seeds'][str(seed)] = breakdown_seed

        # Per-identity AUC (ResNet seed=42 only)
        if is_resnet_42:
            per_id_aucs = {}
            for id_ in ev_ids_sorted:
                id_mask = (ids_ev == id_)
                if id_mask.sum() < 2 or len(np.unique(y_ev_all[id_mask])) < 2:
                    continue
                ia, _ = auc_eer(y_ev_all[id_mask], run_out['per_id']['probs'][id_mask])
                per_id_aucs[id_] = ia
            seed_entry['per_identity_auc_seed42'] = per_id_aucs
            vals = list(per_id_aucs.values())
            print(f"         Per-identity AUC: mean={np.mean(vals):.4f} std={np.std(vals):.4f}  "
                  f"min={min(vals):.4f} max={max(vals):.4f}", flush=True)

        arch_combined['seeds'][str(seed)] = seed_entry

    # Summary
    arch_combined['mean_auc_18id'] = float(np.mean(aucs_ev))
    arch_combined['std_auc_18id']  = float(np.std(aucs_ev))
    arch_combined['mean_eer_18id'] = float(np.mean(eers_ev))
    arch_combined['mean_auc_9id']  = float(np.mean(aucs_te))
    arch_combined['std_auc_9id']   = float(np.std(aucs_te))
    arch_combined['mean_eer_9id']  = float(np.mean(eers_te))

    # Breakdown summary
    for m in METHODS:
        m_aucs = [arch_breakdown['seeds'][str(s)][m]['auc'] for s in SEEDS]
        m_eers = [arch_breakdown['seeds'][str(s)][m]['eer'] for s in SEEDS]
        arch_breakdown[m] = {
            'mean_auc': float(np.mean(m_aucs)), 'std_auc': float(np.std(m_aucs)),
            'mean_eer': float(np.mean(m_eers)),
            'seed_aucs': m_aucs, 'seed_eers': m_eers,
        }

    elapsed = time.time() - t_arch
    arch_combined['runtime_min'] = round(elapsed / 60, 1)
    print(f"  → {arch_name}: 18-id {np.mean(aucs_ev):.4f}±{np.std(aucs_ev):.4f} EER={np.mean(eers_ev)*100:.1f}%  |  "
          f"9-id {np.mean(aucs_te):.4f}±{np.std(aucs_te):.4f} EER={np.mean(eers_te)*100:.1f}%  "
          f"[{elapsed/60:.1f}min]", flush=True)

    combined_results[arch_name] = arch_combined
    breakdown_results[arch_name] = arch_breakdown

    with open(OUT_COMBINED,  'w') as f: json.dump(combined_results,  f, indent=2)
    with open(OUT_BREAKDOWN, 'w') as f: json.dump(breakdown_results, f, indent=2)
    print(f"  Saved → {OUT_COMBINED.name}, {OUT_BREAKDOWN.name}", flush=True)

# Final combined summary
print()
print("=" * 70)
print("  Phase (a) Summary — Combined model (59-id full corpus)")
print("=" * 70)
print(f"  {'Architecture':<14}  {'18-id AUC':>10}  {'18-id EER':>9}  {'9-id AUC':>9}  {'9-id EER':>8}")
print("  " + "-" * 62)
for arch_name in ARCHS_AB:
    r = combined_results[arch_name]
    print(f"  {arch_name:<14}  "
          f"{r['mean_auc_18id']:.4f}±{r['std_auc_18id']:.4f}  "
          f"{r['mean_eer_18id']*100:>7.1f}%  "
          f"{r['mean_auc_9id']:.4f}±{r['std_auc_9id']:.4f}  "
          f"{r['mean_eer_9id']*100:>6.1f}%")

# Print per-id AUC for ResNet
if 'per_identity_auc_seed42' in combined_results['1D ResNet']['seeds']['42']:
    pid = combined_results['1D ResNet']['seeds']['42']['per_identity_auc_seed42']
    vals = list(pid.values())
    print()
    print(f"  ResNet seed=42 per-identity AUC ({len(vals)} ids):")
    for id_, v in sorted(pid.items(), key=lambda x: int(x[0][2:])):
        print(f"    {id_:>5}: {v:.4f}")
    print(f"  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

print()
print("  Phase (b) Summary — ResNet per-generator breakdown (combined model)")
print("  " + "-" * 65)
print(f"  {'Generator':<18}  {'mean AUC':>9}  {'std':>6}  {'mean EER':>8}")
for m in METHODS:
    r = breakdown_results['1D ResNet'][m]
    print(f"  {m:<18}  {r['mean_auc']:.4f}    ±{r['std_auc']:.4f}  {r['mean_eer']*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE (c): Isolated per-method
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("  PHASE (c): Isolated per-method training")
print("=" * 70)

if OUT_ISOLATED.exists():
    with open(OUT_ISOLATED) as f:
        isolated_results = json.load(f)
    print(f"  Resuming — {len(isolated_results)} method/arch combos already done")
else:
    isolated_results = {}

ARCHS_C = ['1D ResNet', 'Transformer']

for method in METHODS:
    # Build per-method training and eval masks
    mask_tr_m_real = (splits_all == 'train') & (y_all == 0)
    mask_tr_m_fake = (splits_all == 'train') & (y_all == 1) & (methods_all == method)
    mask_ev_m      = (mask_ev) & ((y_all == 0) | ((y_all == 1) & (methods_all == method)))

    X_tr_m = np.concatenate([X_all[mask_tr_m_real], X_all[mask_tr_m_fake]])
    y_tr_m = np.concatenate([y_all[mask_tr_m_real], y_all[mask_tr_m_fake]])
    X_ev_m = X_all[mask_ev_m]
    y_ev_m = y_all[mask_ev_m]
    ids_ev_m = ids_all[mask_ev_m]

    nr_tr_m  = int((y_tr_m == 0).sum())
    nf_tr_m  = int((y_tr_m == 1).sum())
    nr_ev_m  = int((y_ev_m == 0).sum())
    nf_ev_m  = int((y_ev_m == 1).sum())
    ev_ids_m = sorted(np.unique(ids_ev_m[y_ev_m == 1]).tolist(), key=lambda x: int(x[2:]))
    n_ev_ids_m = len(ev_ids_m)

    print(f"\n{'='*65}")
    print(f"  METHOD: {method}")
    print(f"{'='*65}")
    print(f"  Train: {nr_tr_m} real + {nf_tr_m} fake")
    print(f"  Eval:  {nr_ev_m} real + {nf_ev_m} fake  ({n_ev_ids_m} ids: {ev_ids_m})")

    for arch_name in ARCHS_C:
        key = f"{method}::{arch_name}"
        if key in isolated_results:
            print(f"  [{arch_name}] already done — skipping")
            continue

        cfg     = BEST_CONFIGS[arch_name]
        cls     = ARCH_CLS[arch_name]
        dropout = cfg['dropout']; lr = cfg['lr']; wd = cfg['wd']
        params  = count_params(cls(dropout=dropout))

        print(f"  [{arch_name}]  lr={lr} wd={wd} dropout={dropout}  params={params:,}")

        seed_aucs, seed_eers, seed_pws = [], [], []
        for seed in SEEDS:
            print(f"    seed={seed:>4}", flush=True)
            run_out, nr, nf, pw_val = train_and_eval(
                cls, dropout, lr, wd, X_tr_m, y_tr_m,
                {'eval': (X_ev_m, y_ev_m, None)},
                seed, arch_name, key)
            auc_s, eer_s = auc_eer(run_out['eval']['y_true'], run_out['eval']['probs'])
            seed_aucs.append(auc_s); seed_eers.append(eer_s); seed_pws.append(pw_val)
            print(f"           AUC={auc_s:.4f}  EER={eer_s*100:.1f}%", flush=True)

        mean_auc = float(np.mean(seed_aucs)); std_auc = float(np.std(seed_aucs))
        mean_eer = float(np.mean(seed_eers))

        # Δ vs breakdown (ResNet combined model)
        delta_vs_combined = None
        if arch_name == '1D ResNet' and method in breakdown_results.get('1D ResNet', {}):
            delta_vs_combined = round(mean_auc - breakdown_results['1D ResNet'][method]['mean_auc'], 4)

        print(f"  → {method} [{arch_name}]:  "
              f"{mean_auc:.4f}±{std_auc:.4f}  EER={mean_eer*100:.1f}%"
              + (f"  Δ(iso-combined)={delta_vs_combined:+.4f}" if delta_vs_combined is not None else ""),
              flush=True)

        isolated_results[key] = {
            'method': method, 'arch': arch_name,
            'params': params, 'lr': lr, 'wd': wd, 'dropout': dropout,
            'n_train_real': nr_tr_m, 'n_train_fake': nf_tr_m,
            'n_eval_real':  nr_ev_m,  'n_eval_fake':  nf_ev_m,
            'n_eval_ids': n_ev_ids_m, 'eval_ids': ev_ids_m,
            'mean_auc': mean_auc, 'std_auc': std_auc, 'mean_eer': mean_eer,
            'seed_aucs': seed_aucs, 'seed_eers': seed_eers,
            'pos_weights': seed_pws,
            'delta_vs_combined_resnet': delta_vs_combined,
        }
        with open(OUT_ISOLATED, 'w') as f: json.dump(isolated_results, f, indent=2)

# Final isolated summary
print()
print("=" * 70)
print("  Phase (c) Summary — Isolated per-method")
print("=" * 70)
print(f"  {'Method':<18}  {'ResNet AUC':>11}  {'R-EER':>5}  {'Tfmr AUC':>10}  {'T-EER':>5}  {'Δ(iso-comb)':>11}")
print("  " + "-" * 72)
for method in METHODS:
    rr = isolated_results.get(f"{method}::1D ResNet", {})
    rt = isolated_results.get(f"{method}::Transformer", {})
    delta = rr.get('delta_vs_combined_resnet')
    delta_str = f"{delta:+.4f}" if delta is not None else "  N/A"
    print(f"  {method:<18}  "
          f"{rr.get('mean_auc', float('nan')):.4f}±{rr.get('std_auc', float('nan')):.4f}  "
          f"{rr.get('mean_eer', float('nan'))*100:>4.1f}%  "
          f"{rt.get('mean_auc', float('nan')):.4f}±{rt.get('std_auc', float('nan')):.4f}  "
          f"{rt.get('mean_eer', float('nan'))*100:>4.1f}%  "
          f"{delta_str}")

total_time = (time.time() - t0_load) / 60
print()
print(f"Total runtime: {total_time:.1f} min")
print(f"Saved → {OUT_COMBINED.name}, {OUT_BREAKDOWN.name}, {OUT_ISOLATED.name}")
