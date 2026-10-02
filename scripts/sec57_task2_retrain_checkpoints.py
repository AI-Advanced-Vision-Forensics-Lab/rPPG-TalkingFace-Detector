"""
run_task2_video_level_auc.py — Video-level AUC for the 1D ResNet (RhythmFormer features).

The canonical window-level result is AUC=0.822 (18-id) / 0.826 (9-id).
Here we aggregate per-window scores to source-video level and recompute AUC/EER.

Window→video mapping:
  Real  : video_id = "id0_0000_w3"  →  source = "id0_0000"  (multiple windows per video)
  Fake  : video_id = "AniTalker__id0_0000_test_..."  →  source = stem (1 window = 1 video)

Aggregation:
  mean-aggregation : score(video) = mean of window scores
  max-aggregation  : score(video) = max  of window scores

No retraining needed if checkpoints exist; otherwise retrains with same seeds/hyperparams
(original run did not save checkpoints — retraining is necessary to recover scores).

Output: data/results/task2_video_level_auc.json
"""

import json, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
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

DATA_ROOT = Path(_args.data_root)
OUT_DIR = Path(_args.out_dir) if _args.out_dir else DATA_ROOT / 'results'
SPLIT_CSV = DATA_ROOT / 'dataset_split_full59.csv'
REAL_WF   = DATA_ROOT / 'waveforms/real'
FAKE_WF   = DATA_ROOT / 'waveforms/CelebDF/TalkingFace'
CKPT_DIR  = DATA_ROOT / 'checkpoints/task2_resnet'
OUT_JSON  = OUT_DIR / 'task2_video_level_auc.json'

SEEDS   = [42, 7, 123, 999, 2024]
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
METHODS = ['AniTalker','EDTalk','EchoMimic','FLOAT','IP_LAP','Real3DPortrait','SadTalker']

# ── 1D ResNet (canonical 240,161-param model) ─────────────────────────────────
class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm1d(out_ch)
        self.c2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm1d(out_ch)
        self.dr = nn.Dropout(0.5)
        self.sc = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                                 nn.BatchNorm1d(out_ch))
                   if stride != 1 or in_ch != out_ch else nn.Identity())
    def forward(self, x):
        return F.relu(self.b2(self.c2(self.dr(F.relu(self.b1(self.c1(x)))))) + self.sc(x))

class Waveform1DResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1,32,7,padding=3,bias=False),
                                  nn.BatchNorm1d(32), nn.ReLU(True), nn.MaxPool1d(2))
        self.s1   = nn.Sequential(BasicBlock1D(32,32), BasicBlock1D(32,32))
        self.s2   = nn.Sequential(BasicBlock1D(32,64,2), BasicBlock1D(64,64))
        self.s3   = nn.Sequential(BasicBlock1D(64,128,2), BasicBlock1D(128,128))
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(128, 1)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.fc(self.gap(self.s3(self.s2(self.s1(self.stem(x))))).squeeze(-1)).squeeze(-1)

N_PARAMS = sum(p.numel() for p in Waveform1DResNet().parameters())

def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)

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
        vid_id = r['video_id']
        if r['class'] == 'real':
            m = re.match(r'(id\d+_\d+)_w\d+', vid_id)
            src_video = m.group(1) if m else vid_id
        else:
            # fake: keep full video_id (includes method) so each fake video is unique.
            # Stripping the method prefix would merge different generators' outputs
            # of the same (source, driving) pair — wrong.
            src_video = vid_id
        rows.append({**r.to_dict(), 'npy_path': str(p), 'src_video': src_video})
    return pd.DataFrame(rows)

# ── Train / eval ──────────────────────────────────────────────────────────────
def train_eval(X_tr, y_tr, X_ev, y_ev, pw_val, seed, ckpt_path=None):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Waveform1DResNet().to(DEVICE)

    if ckpt_path and Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"  [seed={seed}] Loaded checkpoint", flush=True)
    else:
        opt  = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw_val]).to(DEVICE))
        Xt = torch.from_numpy(X_tr).unsqueeze(1).to(DEVICE)
        yt = torch.from_numpy(y_tr).to(DEVICE)
        model.train()
        for _ in range(30):
            perm = torch.randperm(len(Xt), device=DEVICE)
            for s in range(0, len(Xt), 64):
                ix = perm[s:s+64]; opt.zero_grad()
                crit(model(Xt[ix]).reshape(-1), yt[ix]).backward(); opt.step()
        if ckpt_path:
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

    model.eval()
    return model

def get_probs(model, X):
    Xe = torch.from_numpy(X).unsqueeze(1)
    probs = []
    with torch.no_grad():
        for s in range(0, len(Xe), 512):
            probs.extend(torch.sigmoid(model(Xe[s:s+512].to(DEVICE)).reshape(-1)).cpu().numpy())
    return np.array(probs)

# ── Video-level aggregation ───────────────────────────────────────────────────
def video_level_auc(df_ev, window_probs, y_ev, agg='mean'):
    """
    Aggregate window-level scores to video level, return AUC and EER.
    Returns: auc, eer, n_real_videos, n_fake_videos, win_per_real (mean, max)
    """
    df_ev = df_ev.reset_index(drop=True)
    df_ev = df_ev.assign(prob=window_probs, label=y_ev)

    # Group by src_video
    groups = df_ev.groupby('src_video').agg(
        label=('label', 'first'),
        cls=('class', 'first'),
        n_windows=('prob', 'count'),
        score_mean=('prob', 'mean'),
        score_max=('prob', 'max'),
    ).reset_index()

    score_col = 'score_mean' if agg == 'mean' else 'score_max'
    y_vid = groups['label'].to_numpy()
    s_vid = groups[score_col].to_numpy()

    auc = float(roc_auc_score(y_vid, s_vid))
    eer = eer_from_roc(y_vid, s_vid)

    n_real = int((groups['cls']=='real').sum())
    n_fake = int((groups['cls']=='fake').sum())
    win_real = groups[groups['cls']=='real']['n_windows']
    win_fake = groups[groups['cls']=='fake']['n_windows']

    return {
        'auc': round(auc, 4), 'eer': round(eer, 4),
        'n_real_videos': n_real, 'n_fake_videos': n_fake,
        'win_per_real_mean': round(float(win_real.mean()), 2),
        'win_per_real_max':  int(win_real.max()),
        'win_per_fake_mean': round(float(win_fake.mean()), 2),
        'win_per_fake_max':  int(win_fake.max()),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
print(f"Device:  {DEVICE}")
print(f"Params:  {N_PARAMS:,}")
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

print(f"Train:    {n_real_tr} real + {n_fake_tr} fake  ({len(tr_ids)} ids)  pw={pw_val:.4f}")
print(f"Eval-18:  {n_real_eval} real + {n_fake_eval} fake  ({len(eval_ids)} ids)")
print(f"Test-9:   {n_real_test} real + {n_fake_test} fake  ({len(test_ids)} ids)")

# Window distribution summary
src_real_eval = df_eval[df_eval['class']=='real'].groupby('src_video').size()
src_fake_eval = df_eval[df_eval['class']=='fake'].groupby('src_video').size()
src_real_test = df_test[df_test['class']=='real'].groupby('src_video').size()
src_fake_test = df_test[df_test['class']=='fake'].groupby('src_video').size()
print(f"\nEval-18 source videos: {len(src_real_eval)} real, {len(src_fake_eval)} fake")
print(f"  Windows/video — real: mean={src_real_eval.mean():.1f} max={src_real_eval.max()} "
      f"| fake: mean={src_fake_eval.mean():.1f} max={src_fake_eval.max()}")
print(f"Test-9 source videos:  {len(src_real_test)} real, {len(src_fake_test)} fake")
print(f"  Windows/video — real: mean={src_real_test.mean():.1f} max={src_real_test.max()} "
      f"| fake: mean={src_fake_test.mean():.1f} max={src_fake_test.max()}")

print("\nLoading waveforms ...", flush=True)
t0 = time.time()
X_tr   = np.stack([zscore(np.load(p).astype(np.float32)) for p in df_tr['npy_path']])
X_eval = np.stack([zscore(np.load(p).astype(np.float32)) for p in df_eval['npy_path']])
X_test = np.stack([zscore(np.load(p).astype(np.float32)) for p in df_test['npy_path']])
y_tr   = np.array([float(c=='fake') for c in df_tr['class']],   dtype=np.float32)
y_eval = np.array([float(c=='fake') for c in df_eval['class']], dtype=np.float32)
y_test = np.array([float(c=='fake') for c in df_test['class']], dtype=np.float32)
print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

CKPT_DIR.mkdir(parents=True, exist_ok=True)
seed_results = []
t_total = time.time()

for seed in SEEDS:
    ckpt = str(CKPT_DIR / f"seed_{seed}.pt")
    model = train_eval(X_tr, y_tr, X_eval, y_eval, pw_val, seed, ckpt_path=ckpt)
    probs_eval = get_probs(model, X_eval)
    probs_test = get_probs(model, X_test)

    # Window-level AUC (reference)
    win_auc_eval = float(roc_auc_score(y_eval, probs_eval))
    win_eer_eval = eer_from_roc(y_eval, probs_eval)
    win_auc_test = float(roc_auc_score(y_test, probs_test))
    win_eer_test = eer_from_roc(y_test, probs_test)

    # Video-level
    vid_mean_eval = video_level_auc(df_eval, probs_eval, y_eval, 'mean')
    vid_max_eval  = video_level_auc(df_eval, probs_eval, y_eval, 'max')
    vid_mean_test = video_level_auc(df_test, probs_test, y_test, 'mean')
    vid_max_test  = video_level_auc(df_test, probs_test, y_test, 'max')

    seed_results.append({
        'seed': seed,
        'window_auc_18id': round(win_auc_eval, 4), 'window_eer_18id': round(win_eer_eval, 4),
        'window_auc_9id':  round(win_auc_test, 4), 'window_eer_9id':  round(win_eer_test, 4),
        'vid_mean_auc_18id': vid_mean_eval['auc'], 'vid_mean_eer_18id': vid_mean_eval['eer'],
        'vid_max_auc_18id':  vid_max_eval['auc'],  'vid_max_eer_18id':  vid_max_eval['eer'],
        'vid_mean_auc_9id':  vid_mean_test['auc'], 'vid_mean_eer_9id':  vid_mean_test['eer'],
        'vid_max_auc_9id':   vid_max_test['auc'],  'vid_max_eer_9id':   vid_max_test['eer'],
    })
    print(f"  seed={seed}  win-AUC-18id={win_auc_eval:.4f}  "
          f"vid-mean={vid_mean_eval['auc']:.4f}  vid-max={vid_max_eval['auc']:.4f}  "
          f"| win-AUC-9id={win_auc_test:.4f}  "
          f"vid-mean={vid_mean_test['auc']:.4f}  vid-max={vid_max_test['auc']:.4f}", flush=True)

# Means
def col_mean(col): return round(float(np.mean([r[col] for r in seed_results])), 4)
def col_std(col):  return round(float(np.std( [r[col] for r in seed_results])), 4)

# Video info (use last seed's probs for the structure info)
probs_eval_last = get_probs(model, X_eval)
vid_mean_eval_last = video_level_auc(df_eval, probs_eval_last, y_eval, 'mean')

print()
print("=" * 70)
print("TASK 2 — Video-level AUC (1D ResNet, RhythmFormer features)")
print("=" * 70)
print(f"  Params: {N_PARAMS:,}  pw={pw_val:.4f}  seeds={SEEDS}")
print(f"  Original (window-level) reference AUC: 0.8215 (18-id) / 0.8262 (9-id)")
print()
print(f"  {'':30s}  {'18-id':>20s}  {'9-id':>20s}")
print(f"  {'':30s}  {'AUC':>8s} {'EER':>8s}  {'AUC':>8s} {'EER':>8s}")
print(f"  {'Window-level (this run)':30s}  "
      f"{col_mean('window_auc_18id'):>8.4f} {col_mean('window_eer_18id')*100:>7.1f}%  "
      f"{col_mean('window_auc_9id'):>8.4f} {col_mean('window_eer_9id')*100:>7.1f}%")
print(f"  {'Video-level (mean agg)':30s}  "
      f"{col_mean('vid_mean_auc_18id'):>8.4f} {col_mean('vid_mean_eer_18id')*100:>7.1f}%  "
      f"{col_mean('vid_mean_auc_9id'):>8.4f} {col_mean('vid_mean_eer_9id')*100:>7.1f}%")
print(f"  {'Video-level (max agg)':30s}  "
      f"{col_mean('vid_max_auc_18id'):>8.4f} {col_mean('vid_max_eer_18id')*100:>7.1f}%  "
      f"{col_mean('vid_max_auc_9id'):>8.4f} {col_mean('vid_max_eer_9id')*100:>7.1f}%")
print()
print(f"  Source video counts (18-id eval):")
print(f"    Real: {vid_mean_eval_last['n_real_videos']} videos, "
      f"{vid_mean_eval_last['win_per_real_mean']:.1f} windows/video (max {vid_mean_eval_last['win_per_real_max']})")
print(f"    Fake: {vid_mean_eval_last['n_fake_videos']} videos, "
      f"{vid_mean_eval_last['win_per_fake_mean']:.1f} windows/video (max {vid_mean_eval_last['win_per_fake_max']})")
print(f"\n  Runtime: {(time.time()-t_total)/60:.1f} min")

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'experiment': 'video_level_auc',
    'extractor': 'RhythmFormer',
    'arch': '1D_ResNet',
    'params': N_PARAMS, 'pw': round(pw_val, 4),
    'note': 'original run did not save checkpoints; retrained same seeds/hyperparams',
    'n_train_real': n_real_tr, 'n_train_fake': n_fake_tr,
    'n_eval_real': n_real_eval, 'n_eval_fake': n_fake_eval,
    'n_test_real': n_real_test, 'n_test_fake': n_fake_test,
    'train_ids': tr_ids, 'eval_ids': eval_ids, 'test_ids': test_ids,
    'source_video_info': {
        'eval_real_n_videos': len(src_real_eval),
        'eval_real_wins_mean': round(float(src_real_eval.mean()), 2),
        'eval_real_wins_max':  int(src_real_eval.max()),
        'eval_fake_n_videos': len(src_fake_eval),
        'eval_fake_wins_mean': round(float(src_fake_eval.mean()), 2),
        'eval_fake_wins_max':  int(src_fake_eval.max()),
        'test_real_n_videos': len(src_real_test),
        'test_real_wins_mean': round(float(src_real_test.mean()), 2),
        'test_real_wins_max':  int(src_real_test.max()),
        'test_fake_n_videos': len(src_fake_test),
        'test_fake_wins_mean': round(float(src_fake_test.mean()), 2),
        'test_fake_wins_max':  int(src_fake_test.max()),
    },
    'window_level': {
        'mean_auc_18id': col_mean('window_auc_18id'), 'std_auc_18id': col_std('window_auc_18id'),
        'mean_eer_18id': col_mean('window_eer_18id'),
        'mean_auc_9id':  col_mean('window_auc_9id'),  'std_auc_9id':  col_std('window_auc_9id'),
        'mean_eer_9id':  col_mean('window_eer_9id'),
        'reference_18id': 0.8215, 'reference_9id': 0.8262,
    },
    'video_level_mean_agg': {
        'mean_auc_18id': col_mean('vid_mean_auc_18id'), 'std_auc_18id': col_std('vid_mean_auc_18id'),
        'mean_eer_18id': col_mean('vid_mean_eer_18id'),
        'mean_auc_9id':  col_mean('vid_mean_auc_9id'),  'std_auc_9id':  col_std('vid_mean_auc_9id'),
        'mean_eer_9id':  col_mean('vid_mean_eer_9id'),
    },
    'video_level_max_agg': {
        'mean_auc_18id': col_mean('vid_max_auc_18id'), 'std_auc_18id': col_std('vid_max_auc_18id'),
        'mean_eer_18id': col_mean('vid_max_eer_18id'),
        'mean_auc_9id':  col_mean('vid_max_auc_9id'),  'std_auc_9id':  col_std('vid_max_auc_9id'),
        'mean_eer_9id':  col_mean('vid_max_eer_9id'),
    },
    'seed_results': seed_results,
}
with open(OUT_JSON, 'w') as f:
    json.dump(out, f, indent=2)
print(f"  Saved → {OUT_JSON}")
