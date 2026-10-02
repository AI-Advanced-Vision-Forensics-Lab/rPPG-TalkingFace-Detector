"""
table3_main_results.py

Retrains 1D ResNet, 1D CNN, and 1D Transformer with the exact configuration
from table4_table5_phase2.py (no augmentation, same HPs, same split).
Saves every checkpoint. Computes window-level and video-level (mean pool) AUC
on the 18-identity eval set for all three architectures from one code path.

Canonical HPs from table4_table5_phase2.py:
  ResNet:      lr=0.001,  wd=0.0005, dropout=0.5
  Transformer: lr=0.001,  wd=0.0001, dropout=0.3
  CNN:         lr=0.0005, wd=0.001,  dropout=0.3   (NOT lr=1e-3 — confirmed from source)

No augmentation: original combined-model run had none.

Output: data/results/combined_retrain_video_level.json
"""

import json, re, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

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
CKPT_ROOT = DATA_ROOT / 'checkpoints/combined_retrain'
OUT_JSON  = OUT_DIR / 'combined_retrain_video_level.json'
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS     = [42, 7, 123, 999, 2024]

# Previously reported window-level AUC (for delta comparison only; not used in training)
PREV_REPORTED = {'1D ResNet': 0.8215, '1D CNN': 0.809, 'Transformer': 0.806}

# ── Exact architecture definitions from run_phase2_full59.py ──────────────────

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

# Exact HPs from run_phase2_full59.py
ARCH_CFG = {
    '1D ResNet':   dict(cls=Waveform1DResNet, lr=0.001,  wd=0.0005, dropout=0.5),
    'Transformer': dict(cls=WaveformTransformer, lr=0.001, wd=0.0001, dropout=0.3),
    '1D CNN':      dict(cls=Waveform1DCNN, lr=0.0005, wd=0.001, dropout=0.3),
}

# ── Utilities ─────────────────────────────────────────────────────────────────

def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)

def eer_from_roc(y, probs):
    fpr, tpr, _ = roc_curve(y, probs)
    fnr = 1 - tpr
    i = np.argmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)

def get_probs(model, X):
    model.eval()
    Xe = torch.from_numpy(X).unsqueeze(1)
    probs = []
    with torch.no_grad():
        for s in range(0, len(Xe), 512):
            probs.extend(torch.sigmoid(model(Xe[s:s+512].to(DEVICE)).reshape(-1)).cpu().numpy())
    return np.array(probs)

def video_level_mean(df_ev, window_probs, y_ev):
    df = df_ev.reset_index(drop=True).copy()
    df['prob'] = window_probs
    df['label'] = y_ev
    groups = df.groupby('src_video').agg(
        label=('label', 'first'),
        cls=('class', 'first'),
        score=('prob', 'mean'),
    ).reset_index()
    y_vid = groups['label'].to_numpy()
    s_vid = groups['score'].to_numpy()
    auc = float(roc_auc_score(y_vid, s_vid))
    eer = eer_from_roc(y_vid, s_vid)
    n_real = int((groups['cls'] == 'real').sum())
    n_fake = int((groups['cls'] == 'fake').sum())
    return round(auc, 4), round(eer, 4), n_real, n_fake

# ── Load data ─────────────────────────────────────────────────────────────────

print(f"Device: {DEVICE}", flush=True)
print("Loading split ...", flush=True)
sp = pd.read_csv(SPLIT_CSV)

rows = []
for _, r in sp.iterrows():
    p = Path(r['path'])
    if not p.exists():
        continue
    vid_id = r['video_id']
    if r['class'] == 'real':
        m = re.match(r'(id\d+_\d+)_w\d+', vid_id)
        src_video = m.group(1) if m else vid_id
    else:
        src_video = vid_id
    rows.append({**r.to_dict(), 'npy_path': str(p), 'src_video': src_video})

df = pd.DataFrame(rows)
is_tr   = df['split'] == 'train'
is_eval = df['split'].isin(['val', 'test'])

df_tr   = df[is_tr].reset_index(drop=True)
df_eval = df[is_eval].reset_index(drop=True)

n_real_tr   = int((df_tr['class'] == 'real').sum())
n_fake_tr   = int((df_tr['class'] == 'fake').sum())
n_real_eval = int((df_eval['class'] == 'real').sum())
n_fake_eval = int((df_eval['class'] == 'fake').sum())
pw_val      = n_real_tr / max(n_fake_tr, 1)

# Source video counts
src_real = df_eval[df_eval['class'] == 'real'].groupby('src_video').size()
src_fake = df_eval[df_eval['class'] == 'fake'].groupby('src_video').size()
n_src_real = len(src_real)
n_src_fake = len(src_fake)

print(f"Train:   {n_real_tr} real + {n_fake_tr} fake  pw={pw_val:.4f}", flush=True)
print(f"Eval-18: {n_real_eval} real + {n_fake_eval} fake", flush=True)
print(f"Source videos: {n_src_real} real, {n_src_fake} fake", flush=True)

print("Loading waveforms ...", flush=True)
t0 = time.time()
X_tr   = np.stack([zscore(np.load(p).astype(np.float32)) for p in df_tr['npy_path']])
X_eval = np.stack([zscore(np.load(p).astype(np.float32)) for p in df_eval['npy_path']])
y_tr   = np.array([float(c == 'fake') for c in df_tr['class']], dtype=np.float32)
y_eval = np.array([float(c == 'fake') for c in df_eval['class']], dtype=np.float32)
print(f"  Loaded {len(X_tr)} train + {len(X_eval)} eval in {time.time()-t0:.1f}s", flush=True)

# ── Train all architectures ───────────────────────────────────────────────────

all_results = {}
t_global = time.time()

for arch_name, cfg in ARCH_CFG.items():
    print(f"\n{'='*65}", flush=True)
    cls = cfg['cls']
    params = sum(p.numel() for p in cls(dropout=cfg['dropout']).parameters())
    print(f"[{arch_name}]  lr={cfg['lr']} wd={cfg['wd']} dropout={cfg['dropout']}  params={params:,}", flush=True)

    ckpt_dir = CKPT_ROOT / arch_name.replace(' ', '_')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_records = []
    ckpt_paths_written = []
    t_arch = time.time()

    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        model = cls(dropout=cfg['dropout']).to(DEVICE)
        opt   = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
        crit  = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pw_val], dtype=torch.float32).to(DEVICE))

        Xt = torch.from_numpy(X_tr).unsqueeze(1).to(DEVICE)
        yt = torch.from_numpy(y_tr).to(DEVICE)

        model.train()
        for epoch in range(30):
            perm = torch.randperm(len(Xt), device=DEVICE)
            for s in range(0, len(Xt), 64):
                ix = perm[s:s+64]
                opt.zero_grad()
                crit(model(Xt[ix]).reshape(-1), yt[ix]).backward()
                opt.step()

        # Save checkpoint
        ckpt_path = ckpt_dir / f'seed{seed}.pt'
        torch.save({
            'state_dict':  model.state_dict(),
            'seed':        seed,
            'arch':        arch_name,
            'lr':          cfg['lr'],
            'wd':          cfg['wd'],
            'dropout':     cfg['dropout'],
            'params':      params,
            'split_csv':   str(SPLIT_CSV),
            'n_real_tr':   n_real_tr,
            'n_fake_tr':   n_fake_tr,
        }, ckpt_path)
        ckpt_paths_written.append(str(ckpt_path))
        print(f"  seed={seed} trained  → saved {ckpt_path.name}", flush=True)

        # Inference
        probs = get_probs(model, X_eval)

        win_auc = float(roc_auc_score(y_eval, probs))
        win_eer = eer_from_roc(y_eval, probs)
        vid_auc, vid_eer, n_rv, n_fv = video_level_mean(df_eval, probs, y_eval)

        print(f"    win-AUC={win_auc:.4f} EER={win_eer:.4f}  vid-AUC={vid_auc:.4f} EER={vid_eer:.4f}"
              f"  ({n_rv} real, {n_fv} fake source vids)", flush=True)

        seed_records.append({
            'seed': seed,
            'ckpt_path': str(ckpt_path),
            'window_auc_18id': round(win_auc, 4),
            'window_eer_18id': round(win_eer, 4),
            'video_auc_mean_18id': vid_auc,
            'video_eer_mean_18id': vid_eer,
            'n_src_real': n_rv,
            'n_src_fake': n_fv,
        })

    win_aucs = [r['window_auc_18id'] for r in seed_records]
    vid_aucs = [r['video_auc_mean_18id'] for r in seed_records]
    vid_eers = [r['video_eer_mean_18id'] for r in seed_records]

    prev = PREV_REPORTED[arch_name]
    new_win_mean = round(float(np.mean(win_aucs)), 4)
    delta = round(new_win_mean - prev, 4)
    within_005 = abs(delta) <= 0.005

    print(f"\n  {arch_name} SUMMARY:", flush=True)
    print(f"    window-AUC: {new_win_mean:.4f} ± {np.std(win_aucs):.4f}  "
          f"prev={prev}  delta={delta:+.4f}  within_0.005={within_005}", flush=True)
    print(f"    video-AUC:  {np.mean(vid_aucs):.4f} ± {np.std(vid_aucs):.4f}  "
          f"EER={np.mean(vid_eers):.4f}", flush=True)

    all_results[arch_name] = {
        'status': 'OK',
        'ckpt_dir': str(ckpt_dir),
        'ckpt_paths_written': ckpt_paths_written,
        'params': params,
        'lr': cfg['lr'], 'wd': cfg['wd'], 'dropout': cfg['dropout'],
        'seeds': seed_records,
        'window_auc_18id_per_seed': win_aucs,
        'window_auc_18id_mean':  new_win_mean,
        'window_auc_18id_std':   round(float(np.std(win_aucs)), 4),
        'video_auc_mean_pool_18id_per_seed': vid_aucs,
        'video_auc_mean_pool_18id_mean': round(float(np.mean(vid_aucs)), 4),
        'video_auc_mean_pool_18id_std':  round(float(np.std(vid_aucs)), 4),
        'video_eer_mean_pool_18id_mean': round(float(np.mean(vid_eers)), 4),
        'comparison_vs_prev_reported': {
            'prev_window_auc': prev,
            'new_window_auc': new_win_mean,
            'delta': delta,
            'within_0.005': within_005,
        },
        'runtime_min': round((time.time() - t_arch) / 60, 2),
    }

# ── Source video count check ──────────────────────────────────────────────────

all_src_counts = set()
for arch_name, res in all_results.items():
    if res['status'] == 'OK':
        for sr in res['seeds']:
            all_src_counts.add((sr['n_src_real'], sr['n_src_fake']))

counts_consistent = len(all_src_counts) == 1
if all_src_counts:
    actual_real, actual_fake = list(all_src_counts)[0]
else:
    actual_real, actual_fake = -1, -1

count_check = {
    'expected_real': 173, 'expected_fake': 6159,
    'actual_real': actual_real, 'actual_fake': actual_fake,
    'match_expected': actual_real == 173 and actual_fake == 6159,
    'identical_across_all_seeds_and_archs': counts_consistent,
}

# ── Assemble output ───────────────────────────────────────────────────────────

out = {
    'experiment': 'combined_retrain_video_level',
    'timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    'device': str(DEVICE),
    'seeds': SEEDS,
    'eval_set': '18-identity (val+test merged)',
    'augmentation': 'none — matched to run_phase2_full59.py which has no augmentation',
    'hp_source': 'run_phase2_full59.py BEST_CONFIGS (verified from source)',
    'note_on_cnn_hp': 'CNN lr=5e-4 wd=1e-3, NOT lr=1e-3 wd=5e-4; task description mixed them up',
    'window_counts': {'eval_real_windows': n_real_eval, 'eval_fake_windows': n_fake_eval},
    'source_video_count_check': count_check,
    'architectures': all_results,
    'provenance': {
        'split_csv': str(SPLIT_CSV),
        'ckpt_root': str(CKPT_ROOT),
        'source_script': 'run_combined_retrain_video_level.py',
        'architecture_source': 'run_phase2_full59.py (class definitions copied verbatim)',
    },
    'total_runtime_min': round((time.time() - t_global) / 60, 2),
}

# Print summary
print(f"\n{'='*65}", flush=True)
print('SUMMARY', flush=True)
print('='*65, flush=True)
print(f"1. Checkpoint paths written per architecture:", flush=True)
for arch, res in all_results.items():
    if res['status'] == 'OK':
        print(f"   {arch}: {len(res['ckpt_paths_written'])} files → {res['ckpt_dir']}", flush=True)

print(f"\n2. Window-level AUC — new vs previously reported:", flush=True)
for arch, res in all_results.items():
    c = res['comparison_vs_prev_reported']
    print(f"   {arch}: new={c['new_window_auc']:.4f}  prev={c['prev_window_auc']}  "
          f"delta={c['delta']:+.4f}  within_0.005={c['within_0.005']}", flush=True)

print(f"\n3. Video-level AUC and EER (mean pool, 18-id):", flush=True)
for arch, res in all_results.items():
    print(f"   {arch}: AUC={res['video_auc_mean_pool_18id_mean']:.4f}±{res['video_auc_mean_pool_18id_std']:.4f}"
          f"  EER={res['video_eer_mean_pool_18id_mean']:.4f}", flush=True)

print(f"\n4. Source-video counts confirmed (173 real / 6159 fake): "
      f"{count_check['match_expected']}", flush=True)
print(f"   Identical across all seeds and architectures: "
      f"{count_check['identical_across_all_seeds_and_archs']}", flush=True)

print(f"\n5. All three architectures from one run, one code path: TRUE", flush=True)
print(f"   Total runtime: {out['total_runtime_min']:.1f} min", flush=True)

# Save
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {OUT_JSON}", flush=True)

print('\nFINAL JSON:', flush=True)
print(json.dumps(out, indent=2), flush=True)
