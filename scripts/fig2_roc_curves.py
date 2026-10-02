"""
Collect 5-seed per-generator probabilities from the combined ResNet,
then plot 5-seed mean ROC curves per generator → roc_curves.pdf.

Uses dataset_split_full59.csv (59-id corpus, same as Phase 2(b)).
ResNet: lr=0.001, wd=0.0005, dropout=0.5, 30 epochs, pos_weight=n_real/n_fake.
"""

import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths (configurable; see README) ─────────────────────────────────────────
import argparse as _argparse
_ap = _argparse.ArgumentParser(description=(__doc__ or '').strip().split('\n')[0])
_ap.add_argument('--data-root', default='data',
                 help='data folder laid out as described in the README (default: ./data)')
_ap.add_argument('--out-dir', default=None,
                 help='folder for the figure (default: ./figures)')
_args = _ap.parse_args()

DATA_ROOT  = Path(_args.data_root)
OUT_DIR = Path(_args.out_dir) if _args.out_dir else DATA_ROOT / 'results'
SPLIT_FILE = DATA_ROOT / 'dataset_split_full59.csv'
OUT_PDF    = Path(_args.out_dir or 'figures') / 'roc_curves.pdf'

METHODS = ['AniTalker','EDTalk','EchoMimic','FLOAT','IP_LAP','Real3DPortrait','SadTalker']
SEEDS   = [42, 7, 123, 999, 2024]
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# AUC values for legend (Phase 2(b) means, rounded to 3 dp)
LEGEND_AUCS = {
    'Real3DPortrait': 0.937,
    'EDTalk':         0.903,
    'SadTalker':      0.890,
    'AniTalker':      0.866,
    'FLOAT':          0.787,
    'EchoMimic':      0.758,
    'IP_LAP':         0.617,
}
LEGEND_LABEL = {
    'IP_LAP': 'IP-LAP',   # hyphen as in user spec
}
METHODS_SORTED = ['Real3DPortrait','EDTalk','SadTalker','AniTalker','FLOAT','EchoMimic','IP_LAP']

# ── ResNet ─────────────────────────────────────────────────────────────────────
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

# ── Load data ──────────────────────────────────────────────────────────────────
def zscore(x): return (x - x.mean()) / (x.std() + 1e-6)

print(f"Device: {DEVICE}")
print("Loading waveforms ...", flush=True)
t0 = time.time()
split_df = pd.read_csv(SPLIT_FILE)
X_all = np.stack([zscore(np.load(p).astype(np.float32)) for p in split_df['path']])
y_all = np.array([float(c == 'fake') for c in split_df['class']], dtype=np.float32)
s_all = split_df['split'].to_numpy()
m_all = split_df['method'].to_numpy()
print(f"  {len(X_all)} windows in {time.time()-t0:.1f}s")

is_tr = (s_all == 'train')
is_ev = np.isin(s_all, ['val', 'test'])

X_tr = X_all[is_tr]; y_tr = y_all[is_tr]
X_ev = X_all[is_ev]; y_ev = y_all[is_ev]
m_ev = m_all[is_ev]

nr_tr = int((y_tr==0).sum()); nf_tr = int((y_tr==1).sum())
pw_val = nr_tr / nf_tr
print(f"  Train: {nr_tr} real + {nf_tr} fake  pos_weight={pw_val:.6f}")
print(f"  Eval:  {int((y_ev==0).sum())} real + {int((y_ev==1).sum())} fake")
print()

pw = torch.tensor([pw_val]).to(DEVICE)
Xt = torch.from_numpy(X_tr).unsqueeze(1).to(DEVICE)
yt = torch.from_numpy(y_tr).to(DEVICE)
Xe = torch.from_numpy(X_ev).unsqueeze(1)

# ── Per-method eval masks (real + that method's fakes) ────────────────────────
def method_mask(m):
    return (y_ev == 0) | ((y_ev == 1) & (m_ev == m))

# ── Train 5 seeds, collect per-generator probs ────────────────────────────────
# seed_probs[m][seed_idx] = (y_true, probs)
seed_probs = {m: [] for m in METHODS}

for si, seed in enumerate(SEEDS):
    print(f"seed={seed}", flush=True)
    torch.manual_seed(seed); np.random.seed(seed)
    model = Waveform1DResNet(dropout=0.5).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pw)

    model.train()
    for _ in range(30):
        perm = torch.randperm(len(Xt), device=DEVICE)
        for s in range(0, len(Xt), 64):
            ix = perm[s:s+64]; opt.zero_grad()
            crit(model(Xt[ix]).reshape(-1), yt[ix]).backward()
            opt.step()

    model.eval()
    all_probs = []
    with torch.no_grad():
        for s in range(0, len(Xe), 512):
            all_probs.extend(
                torch.sigmoid(model(Xe[s:s+512].to(DEVICE)).reshape(-1)).cpu().numpy())
    all_probs = np.array(all_probs)

    for m in METHODS:
        mask = method_mask(m)
        seed_probs[m].append((y_ev[mask], all_probs[mask]))
        auc = roc_auc_score(y_ev[mask], all_probs[mask])
        print(f"  {m:<16} AUC={auc:.4f}", flush=True)
    print()

# ── Compute 5-seed mean ROC per generator ─────────────────────────────────────
FPR_GRID = np.linspace(0, 1, 1000)

mean_rocs = {}
for m in METHODS:
    tprs = []
    for y_true, probs in seed_probs[m]:
        fpr, tpr, _ = roc_curve(y_true, probs)
        tprs.append(np.interp(FPR_GRID, fpr, tpr))
    mean_rocs[m] = np.mean(tprs, axis=0)

# ── Plot ───────────────────────────────────────────────────────────────────────
# Single-column width: 3.5 in. Height: 3.3 in.
# Font: 8 pt axes, 7 pt legend, 8 pt tick labels.
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         8,
    'axes.labelsize':    8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'legend.fontsize':   7,
    'legend.handlelength': 2.5,
    'legend.handleheight': 0.9,
    'lines.linewidth':   1.1,
    'pdf.fonttype':      42,   # embed TrueType fonts
    'ps.fonttype':       42,
})

fig, ax = plt.subplots(figsize=(3.5, 3.3))

# Color + linestyle pairs, chosen to be distinguishable in both color and grayscale
STYLES = {
    'Real3DPortrait': dict(color='#1f77b4', linestyle='-',          label_suffix=''),   # solid blue
    'EDTalk':         dict(color='#d62728', linestyle='--',         label_suffix=''),   # dashed red
    'SadTalker':      dict(color='#2ca02c', linestyle=':',          label_suffix=''),   # dotted green
    'AniTalker':      dict(color='#ff7f0e', linestyle='-.',         label_suffix=''),   # dash-dot orange
    'FLOAT':          dict(color='#9467bd', linestyle=(0,(5,2)),    label_suffix=''),   # long-dash purple
    'EchoMimic':      dict(color='#8c564b', linestyle=(0,(3,1,1,1)),label_suffix=''),   # dash-dot-dot brown
    'IP_LAP':         dict(color='#7f7f7f', linestyle=(0,(1,2)),    label_suffix=''),   # dotted gray
}

for m in METHODS_SORTED:
    auc_label = LEGEND_AUCS[m]
    display   = LEGEND_LABEL.get(m, m)
    label     = f'{display} ({auc_label:.3f})'
    st        = STYLES[m]
    ax.plot(FPR_GRID, mean_rocs[m],
            color=st['color'], linestyle=st['linestyle'],
            label=label, linewidth=1.1)

# Chance diagonal
ax.plot([0,1], [0,1], color='black', linestyle='--', linewidth=0.7,
        label='Random (0.500)', dashes=(4,3))

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax.set_aspect('equal')

leg = ax.legend(loc='lower right', frameon=True, framealpha=0.9,
                edgecolor='0.7', borderpad=0.5, labelspacing=0.3,
                handletextpad=0.5)
leg.get_frame().set_linewidth(0.5)

ax.tick_params(direction='in', length=3)
for spine in ax.spines.values():
    spine.set_linewidth(0.6)

fig.tight_layout(pad=0.4)
fig.savefig(OUT_PDF, format='pdf', dpi=300, bbox_inches='tight')
print(f"Saved → {OUT_PDF}")

# Quick sanity: print confirmed AUCs from this run
print("\nConfirmed mean AUCs (this run):")
for m in METHODS_SORTED:
    aucs = [roc_auc_score(*pair) for pair in seed_probs[m]]
    print(f"  {m:<18} {np.mean(aucs):.4f}±{np.std(aucs):.4f}")
