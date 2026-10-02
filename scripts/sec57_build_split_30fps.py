"""
build_split_30fps.py — rebuild the identity split on the fps-normalised waveforms.

Real waveforms:  reused from data/waveforms/real/  (30 fps, no change)
Fake waveforms:  taken from data/waveforms/CelebDF_30fps/TalkingFace/{method}/

Identity split preserved:
    TEST_IDS = {id0, id4, id6, id11, id13, id16, id23, id27, id54}
    VAL_IDS  = {id8, id12, id19, id22, id34, id42, id44, id52, id56}
    TRAIN    = remaining 41 identities

Outputs:
    data/dataset_split_30fps.csv    (split, video_id, class, method, identity, path)
    data/results/split_30fps_summary.json
"""

import json, re, sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ── Paths (configurable; see README) ─────────────────────────────────────────
import argparse as _argparse
_ap = _argparse.ArgumentParser(description=(__doc__ or '').strip().split('\n')[0])
_ap.add_argument('--data-root', default='data',
                 help='data folder laid out as described in the README (default: ./data)')
_ap.add_argument('--out-dir', default=None,
                 help='folder for result JSONs (default: <data-root>/results)')
_args = _ap.parse_args()

DATA_ROOT    = Path(_args.data_root)
OUT_DIR = Path(_args.out_dir) if _args.out_dir else DATA_ROOT / 'results'
REAL_WAVE    = DATA_ROOT / 'waveforms/real'
FAKE_WAVE_30 = DATA_ROOT / 'waveforms/CelebDF_30fps/TalkingFace'
SPLIT_OUT    = DATA_ROOT / 'dataset_split_30fps.csv'
SUMMARY_OUT  = OUT_DIR / 'split_30fps_summary.json'

METHODS = ['AniTalker','EDTalk','EchoMimic','FLOAT','IP_LAP','Real3DPortrait','SadTalker']

TEST_IDS = {'id0','id4','id6','id11','id13','id16','id23','id27','id54'}
VAL_IDS  = {'id8','id12','id19','id22','id34','id42','id44','id52','id56'}

def identity_split(ident):
    if ident in TEST_IDS: return 'test'
    if ident in VAL_IDS:  return 'val'
    return 'train'

rows = []

# ── Real waveforms (reuse existing, 30 fps, sliding-window) ───────────────────
print(f"Scanning real waveforms: {REAL_WAVE}")
real_id_re = re.compile(r'^(id\d{1,2})_')
for npy in sorted(REAL_WAVE.glob('*.npy')):
    m = real_id_re.match(npy.stem)
    if m is None:
        print(f"  WARN: cannot parse identity from {npy.name} — skipping")
        continue
    ident = m.group(1)
    id_num = int(ident[2:])
    if id_num > 61:
        continue  # outside 59-id range
    rows.append({
        'split': identity_split(ident),
        'video_id': npy.stem,
        'class': 'real',
        'method': '',
        'identity': ident,
        'path': str(npy),
    })
n_real = len(rows)
print(f"  Real waveforms found: {n_real}")

# ── Fake waveforms (30-fps-normalised) ────────────────────────────────────────
print(f"Scanning fake waveforms: {FAKE_WAVE_30}")
id_re = re.compile(r'^(id\d{1,2})_')
for method in METHODS:
    mdir = FAKE_WAVE_30 / method
    if not mdir.exists():
        print(f"  WARN: {mdir} does not exist — no fakes for {method}")
        continue
    for npy in sorted(mdir.glob('*.npy')):
        m = id_re.match(npy.stem)
        if m is None:
            print(f"  WARN: cannot parse identity from {npy.name} — skipping")
            continue
        ident = m.group(1)
        id_num = int(ident[2:])
        if id_num > 61:
            continue
        rows.append({
            'split': identity_split(ident),
            'video_id': f"{method}__{npy.stem}",
            'class': 'fake',
            'method': method,
            'identity': ident,
            'path': str(npy),
        })
n_fake = len(rows) - n_real
print(f"  Fake waveforms found: {n_fake}")

df = pd.DataFrame(rows)

# ── Integrity checks ──────────────────────────────────────────────────────────
ev_ids = sorted(df[df['split'].isin(['val','test'])]['identity'].unique(), key=lambda x: int(x[2:]))
tr_ids = sorted(df[df['split']=='train']['identity'].unique(), key=lambda x: int(x[2:]))
n_ev_ids = len(ev_ids)

n_real_tr  = int(((df['class']=='real') & (df['split']=='train')).sum())
n_real_val = int(((df['class']=='real') & (df['split']=='val')).sum())
n_real_te  = int(((df['class']=='real') & (df['split']=='test')).sum())
n_real_ev  = n_real_val + n_real_te

n_fake_tr  = int(((df['class']=='fake') & (df['split']=='train')).sum())
n_fake_val = int(((df['class']=='fake') & (df['split']=='val')).sum())
n_fake_te  = int(((df['class']=='fake') & (df['split']=='test')).sum())
n_fake_ev  = n_fake_val + n_fake_te

pw_val = n_real_tr / max(n_fake_tr, 1)

print()
print("=" * 60)
print("  SPLIT SUMMARY (30-fps-normalised corpus)")
print("=" * 60)
print(f"  Total waveforms: {len(df)}")
print(f"  Real: {n_real_tr} train | {n_real_val} val | {n_real_te} test (total {n_real})")
print(f"  Fake: {n_fake_tr} train | {n_fake_val} val | {n_fake_te} test (total {n_fake})")
print(f"  pos_weight = {n_real_tr}/{n_fake_tr} = {pw_val:.6f}")
print(f"  Eval identities ({n_ev_ids}): {ev_ids}")
print(f"  Train identities ({len(tr_ids)}): {tr_ids}")
print()
print(f"  Per-method fake counts:")
for sp in ['train','val','test']:
    sub = df[(df['class']=='fake') & (df['split']==sp)]
    mc = sub['method'].value_counts()
    parts = '  '.join(f"{m}={mc.get(m,0)}" for m in METHODS)
    print(f"    {sp:5s}: {len(sub):5d}   [{parts}]")

# Warn if expected counts differ significantly from 30fps baseline
EXPECTED_REAL_TR = 1675
EXPECTED_REAL_EV = 696
if n_real_tr != EXPECTED_REAL_TR:
    print(f"\n  NOTE: train real = {n_real_tr} (expected {EXPECTED_REAL_TR})")
if n_real_ev != EXPECTED_REAL_EV:
    print(f"  NOTE: eval real = {n_real_ev} (expected {EXPECTED_REAL_EV})")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(SPLIT_OUT, index=False)
print(f"\n  Saved: {SPLIT_OUT}")

summary = {
    'n_total': len(df),
    'n_real_train': n_real_tr, 'n_real_val': n_real_val, 'n_real_test': n_real_te,
    'n_real_eval': n_real_ev,
    'n_fake_train': n_fake_tr, 'n_fake_val': n_fake_val, 'n_fake_test': n_fake_te,
    'n_fake_eval': n_fake_ev,
    'pos_weight': round(pw_val, 6),
    'n_eval_ids': n_ev_ids, 'eval_ids': ev_ids,
    'n_train_ids': len(tr_ids), 'train_ids': tr_ids,
    'per_method_train': {m: int(((df['class']=='fake')&(df['split']=='train')&(df['method']==m)).sum())
                         for m in METHODS},
    'per_method_val':   {m: int(((df['class']=='fake')&(df['split']=='val')&(df['method']==m)).sum())
                         for m in METHODS},
    'per_method_test':  {m: int(((df['class']=='fake')&(df['split']=='test')&(df['method']==m)).sum())
                         for m in METHODS},
}
SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(SUMMARY_OUT, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: {SUMMARY_OUT}")
