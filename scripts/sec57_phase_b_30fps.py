"""
Phase B (30-fps-normalised corpus) — combined-model per-generator evaluation.

Architecture:  1D ResNet  (lr=0.001, wd=0.0005, dropout=0.5, AdamW, 30 epochs)
Split file:    data/dataset_split_30fps.csv
Seeds:         42, 7, 123, 999, 2024

Reports:
  (1) Per-generator AUC ± std and EER, plus pooled AUC — side-by-side with
      the non-normalised values (0.822 overall) and Δ per generator.
  (2) Train/eval counts, identity count, parameter count, pos_weight.
  (3) Spectral check on the new waveforms: median bpm, IQR, fraction in
      50–100 bpm, median SNR — real vs each generator.

Saved to: data/results/phase_b_30fps.json
"""

import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
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
SPLIT_FILE = DATA_ROOT / 'dataset_split_30fps.csv'
OUT_JSON   = OUT_DIR / 'phase_b_30fps.json'
PREV_JSON  = OUT_DIR / 'full59_per_generator_breakdown.json'

METHODS  = ['AniTalker','EDTalk','EchoMimic','FLOAT','IP_LAP','Real3DPortrait','SadTalker']
SEEDS    = [42, 7, 123, 999, 2024]
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR, WD, DROPOUT = 0.001, 0.0005, 0.5

# Non-normalised reference AUCs (from full59_per_generator_breakdown.json, ResNet mean)
PREV_AUC = {
    'AniTalker': 0.8662, 'EDTalk': 0.9025, 'EchoMimic': 0.7583,
    'FLOAT': 0.7871, 'IP_LAP': 0.6170, 'Real3DPortrait': 0.9368, 'SadTalker': 0.8898,
    'pooled': 0.8215,
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    i = np.argmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)

# ── 1D ResNet ──────────────────────────────────────────────────────────────────
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

params = sum(p.numel() for p in Waveform1DResNet(dropout=DROPOUT).parameters())

# ── Spectral check ─────────────────────────────────────────────────────────────
def spectral_check(waves, fps=30.0, band=(0.7, 4.0)):
    """For each 160-sample waveform: dominant bpm, SNR in 0.7-4.0 Hz."""
    N = 160
    bpms, snrs = [], []
    lo_bin = max(1, int(np.floor(band[0] * N / fps)))
    hi_bin = min(N // 2, int(np.ceil(band[1] * N / fps)))
    for w in waves:
        fft_mag = np.abs(np.fft.rfft(w))[:N // 2 + 1]
        band_mag = fft_mag[lo_bin:hi_bin + 1]
        dom_k    = np.argmax(band_mag) + lo_bin
        dom_hz   = dom_k * fps / N
        bpms.append(dom_hz * 60)
        # SNR: signal = dom_k ± 1 bin; noise = rest of band
        sig_bins = set(range(max(lo_bin, dom_k - 1), min(hi_bin + 1, dom_k + 2)))
        sig_pwr  = sum(band_mag[k - lo_bin] ** 2 for k in sig_bins if lo_bin <= k <= hi_bin)
        noi_pwr  = sum(band_mag[k - lo_bin] ** 2 for k in range(lo_bin, hi_bin + 1) if k not in sig_bins)
        snrs.append(sig_pwr / max(noi_pwr, 1e-12))
    return np.array(bpms), np.array(snrs)

# ── Load data ──────────────────────────────────────────────────────────────────
print(f"Device:  {DEVICE}")
print(f"Params:  {params:,}")
print(f"Split:   {SPLIT_FILE}")
if not SPLIT_FILE.exists():
    sys.exit("ERROR: split file not found — run build_split_30fps.py first")

split_df = pd.read_csv(SPLIT_FILE)

# Validate
n_real_tr  = int(((split_df['class']=='real') & (split_df['split']=='train')).sum())
n_fake_tr  = int(((split_df['class']=='fake') & (split_df['split']=='train')).sum())
n_real_ev  = int(((split_df['class']=='real') & split_df['split'].isin(['val','test'])).sum())
n_fake_ev  = int(((split_df['class']=='fake') & split_df['split'].isin(['val','test'])).sum())
ev_ids     = sorted(split_df[split_df['split'].isin(['val','test'])]['identity'].unique(),
                    key=lambda x: int(x[2:]))
pw_val     = n_real_tr / max(n_fake_tr, 1)

print(f"Train:   {n_real_tr} real + {n_fake_tr} fake  pos_weight={pw_val:.6f}")
print(f"Eval:    {n_real_ev} real + {n_fake_ev} fake  ({len(ev_ids)} ids: {ev_ids})")
print()

print("Loading waveforms ...", flush=True)
t0 = time.time()
X_all  = np.stack([zscore(np.load(p).astype(np.float32)) for p in split_df['path']])
y_all  = np.array([float(c == 'fake') for c in split_df['class']], dtype=np.float32)
s_all  = split_df['split'].to_numpy()
m_all  = split_df['method'].to_numpy()
id_all = split_df['identity'].to_numpy()
print(f"  {len(X_all)} windows in {time.time()-t0:.1f}s")

is_tr = (s_all == 'train')
is_ev = np.isin(s_all, ['val', 'test'])

X_tr = X_all[is_tr]; y_tr = y_all[is_tr]
X_ev = X_all[is_ev]; y_ev = y_all[is_ev]
m_ev = m_all[is_ev]

# ── Spectral check on new waveforms ───────────────────────────────────────────
print()
print("=" * 60)
print("Spectral check on 30-fps-normalised waveforms")
print("=" * 60)

spectral_results = {}
# Real eval
real_ev_X = X_all[is_ev & (y_all == 0)]
bpms_r, snrs_r = spectral_check(real_ev_X, fps=30.0)
q1_r, q3_r     = np.percentile(bpms_r, 25), np.percentile(bpms_r, 75)
frac_r = float(((bpms_r >= 50) & (bpms_r <= 100)).mean())
spectral_results['real'] = {
    'median_bpm': float(np.median(bpms_r)), 'iqr_bpm': float(q3_r - q1_r),
    'frac_50_100': round(frac_r, 4), 'median_snr': float(np.median(snrs_r))
}
print(f"  {'Class':<18}  {'median bpm':>10}  {'IQR':>8}  {'50-100%':>8}  {'median SNR':>10}")
print(f"  {'-'*60}")
print(f"  {'Real (eval)':<18}  {np.median(bpms_r):>10.1f}  "
      f"{q3_r-q1_r:>8.1f}  {frac_r*100:>8.1f}%  {np.median(snrs_r):>10.3f}")

for method in METHODS:
    mask_m = is_ev & (y_all == 1) & (m_all == method)
    if mask_m.sum() == 0:
        print(f"  {method:<18}  (no eval windows)")
        continue
    bpms_m, snrs_m = spectral_check(X_all[mask_m], fps=30.0)  # now all at 30fps
    q1_m, q3_m     = np.percentile(bpms_m, 25), np.percentile(bpms_m, 75)
    frac_m = float(((bpms_m >= 50) & (bpms_m <= 100)).mean())
    spectral_results[method] = {
        'median_bpm': float(np.median(bpms_m)), 'iqr_bpm': float(q3_m - q1_m),
        'frac_50_100': round(frac_m, 4), 'median_snr': float(np.median(snrs_m))
    }
    print(f"  {method:<18}  {np.median(bpms_m):>10.1f}  "
          f"{q3_m-q1_m:>8.1f}  {frac_m*100:>8.1f}%  {np.median(snrs_m):>10.3f}")

# ── ResNet training ────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"1D ResNet  lr={LR} wd={WD} dropout={DROPOUT}  5 seeds")
print("=" * 60)

def method_mask(m):
    return (y_ev == 0) | ((y_ev == 1) & (m_ev == m))

pw = torch.tensor([pw_val]).to(DEVICE)
Xt = torch.from_numpy(X_tr).unsqueeze(1).to(DEVICE)
yt = torch.from_numpy(y_tr).to(DEVICE)
Xe = torch.from_numpy(X_ev).unsqueeze(1)

# seed_probs[m] = list of (y_true, probs) per seed
seed_probs = {m: [] for m in METHODS}
pooled_seed_aucs = []

t_start = time.time()
for si, seed in enumerate(SEEDS):
    print(f"\nseed={seed}", flush=True)
    torch.manual_seed(seed); np.random.seed(seed)
    model = Waveform1DResNet(dropout=DROPOUT).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pw)

    model.train()
    for _ in range(30):
        perm = torch.randperm(len(Xt), device=DEVICE)
        for s in range(0, len(Xt), 64):
            ix = perm[s:s+64]; opt.zero_grad()
            crit(model(Xt[ix]).reshape(-1), yt[ix]).backward()
            opt.step()

    model.eval()
    probs_ev = []
    with torch.no_grad():
        for s in range(0, len(Xe), 512):
            probs_ev.extend(
                torch.sigmoid(model(Xe[s:s+512].to(DEVICE)).reshape(-1)).cpu().numpy())
    probs_ev = np.array(probs_ev)

    pooled_auc = float(roc_auc_score(y_ev, probs_ev))
    pooled_seed_aucs.append(pooled_auc)
    print(f"  pooled AUC={pooled_auc:.4f}", flush=True)

    for m in METHODS:
        mask = method_mask(m)
        if mask.sum() < 10: continue
        auc = roc_auc_score(y_ev[mask], probs_ev[mask])
        seed_probs[m].append((y_ev[mask], probs_ev[mask]))
        print(f"  {m:<18}  AUC={auc:.4f}", flush=True)

elapsed = (time.time() - t_start) / 60

# ── Aggregate ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("  PHASE B RESULTS — 30-fps-normalised (vs non-normalised)")
print("=" * 70)
print(f"  {'Generator':<18}  {'30fps AUC':>10}  {'Prev AUC':>9}  {'Δ':>8}  {'EER':>6}")
print(f"  {'-'*60}")

per_gen = {}
for m in METHODS:
    if not seed_probs[m]:
        print(f"  {m:<18}  (no data)")
        per_gen[m] = {'mean_auc': None, 'std_auc': None, 'mean_eer': None}
        continue
    aucs = [float(roc_auc_score(yt_, ps_)) for yt_, ps_ in seed_probs[m]]
    eers = [compute_eer(yt_, ps_) for yt_, ps_ in seed_probs[m]]
    mean_auc = float(np.mean(aucs)); std_auc = float(np.std(aucs))
    mean_eer = float(np.mean(eers))
    prev     = PREV_AUC.get(m, float('nan'))
    delta    = mean_auc - prev
    print(f"  {m:<18}  {mean_auc:.4f}±{std_auc:.4f}  {prev:>8.4f}  {delta:>+8.4f}  {mean_eer*100:>5.1f}%")
    per_gen[m] = {'seed_aucs': aucs, 'seed_eers': eers,
                  'mean_auc': mean_auc, 'std_auc': std_auc, 'mean_eer': mean_eer,
                  'prev_auc': prev, 'delta': round(delta, 4)}

pool_mean = float(np.mean(pooled_seed_aucs)); pool_std = float(np.std(pooled_seed_aucs))
prev_pool = PREV_AUC['pooled']
print(f"  {'-'*60}")
print(f"  {'Pooled':<18}  {pool_mean:.4f}±{pool_std:.4f}  {prev_pool:>8.4f}  "
      f"{pool_mean - prev_pool:>+8.4f}")
print(f"  Runtime: {elapsed:.1f} min")

# ── Save ───────────────────────────────────────────────────────────────────────
result = {
    'corpus': '30fps-normalised',
    'arch': '1D ResNet', 'params': params,
    'lr': LR, 'wd': WD, 'dropout': DROPOUT,
    'n_train_real': n_real_tr, 'n_train_fake': n_fake_tr,
    'n_eval_real': n_real_ev, 'n_eval_fake': n_fake_ev,
    'pos_weight': round(pw_val, 6),
    'n_eval_ids': len(ev_ids), 'eval_ids': ev_ids,
    'seeds': SEEDS,
    'pooled_seed_aucs': pooled_seed_aucs,
    'pooled_mean_auc': pool_mean, 'pooled_std_auc': pool_std,
    'pooled_prev_auc': prev_pool, 'pooled_delta': round(pool_mean - prev_pool, 4),
    'per_generator': per_gen,
    'spectral': spectral_results,
    'runtime_min': round(elapsed, 1),
}
with open(OUT_JSON, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved → {OUT_JSON}")
