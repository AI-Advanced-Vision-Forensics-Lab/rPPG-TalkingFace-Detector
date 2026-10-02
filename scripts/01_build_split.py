"""
Phase 1 — Rebuild manifest + dataset split for full 59-identity Celeb-DF++ TF corpus.

Source:  data/waveforms/CelebDF/TalkingFace/{method}/idXX_....npy
Real:    data/waveforms/real/idXX_....npy  (unchanged)

Output:
  data/manifest_full59.csv      — all fake rows with explicit paths
  data/dataset_split_full59.csv — train/val/test rows (real + fake) with path column

Fail-loud checks:
  - All 7 generators present
  - 59 identities in fake, matching real
  - Same 9 test + 9 val identities as before
  - Real counts match expected (2,371 total)
  - All 7 generators cover the same identity set (intersection == union == 59)
"""

import re, sys, json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

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
WAVE_ROOT = DATA_ROOT / 'waveforms'
TF_ROOT   = WAVE_ROOT / 'CelebDF' / 'TalkingFace'
REAL_ROOT = WAVE_ROOT / 'real'

MANIFEST_PATH   = DATA_ROOT / 'manifest_full59.csv'
SPLIT_PATH      = DATA_ROOT / 'dataset_split_full59.csv'

METHODS = ['AniTalker','EDTalk','EchoMimic','FLOAT','IP_LAP','Real3DPortrait','SadTalker']

VAL_IDS  = {'id8','id12','id19','id22','id34','id42','id44','id52','id56'}
TEST_IDS = {'id0','id4','id6','id11','id13','id16','id23','id27','id54'}
EVAL_IDS = VAL_IDS | TEST_IDS   # 18 identities

EXPECTED_REAL_TOTAL   = 2371
EXPECTED_REAL_TRAIN   = 1675
EXPECTED_IDENTITIES   = 59
EXPECTED_METHODS      = set(METHODS)

# ── Celeb-DF++ identity filter: id0..id61, at most 2 numeric digits ──────────
def is_celebdf(s):
    m = re.fullmatch(r'id(\d{1,2})', s)
    return m and 0 <= int(m.group(1)) <= 61

def id_sortkey(s):
    return int(s[2:])

# ─────────────────────────────────────────────────────────────────────────────
# 1. Scan real waveforms
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  PHASE 1  —  59-identity Celeb-DF++ TF corpus split builder")
print("=" * 65)
print()
print("[1] Scanning real waveforms ...")

real_rows = []
real_ids  = set()
for f in sorted(REAL_ROOT.glob('*.npy')):
    stem = f.stem
    id_  = stem.split('_')[0]
    if not is_celebdf(id_):
        print(f"  WARN: unexpected real identity '{id_}' in {f.name}, skipping")
        continue
    real_ids.add(id_)
    real_rows.append({'video_id': stem, 'class': 'real', 'method': None,
                      'identity': id_, 'path': str(f)})

print(f"  Real files:      {len(real_rows)}")
print(f"  Real identities: {len(real_ids)} → {sorted(real_ids, key=id_sortkey)}")

if len(real_rows) != EXPECTED_REAL_TOTAL:
    sys.exit(f"FAIL: expected {EXPECTED_REAL_TOTAL} real windows, got {len(real_rows)}")
if len(real_ids) != EXPECTED_IDENTITIES:
    sys.exit(f"FAIL: expected {EXPECTED_IDENTITIES} real identities, got {len(real_ids)}")
print(f"  ✓ {EXPECTED_REAL_TOTAL} real windows across {EXPECTED_IDENTITIES} identities confirmed")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Scan fake waveforms
# ─────────────────────────────────────────────────────────────────────────────
print("[2] Scanning fake waveforms (CelebDF/TalkingFace) ...")

missing_methods = EXPECTED_METHODS - {d.name for d in TF_ROOT.iterdir() if d.is_dir()}
if missing_methods:
    sys.exit(f"FAIL: missing generator directories: {sorted(missing_methods)}")

fake_rows = []
per_method_ids = {}
per_method_excl = {}

for method in METHODS:
    mdir = TF_ROOT / method
    kept, excl = [], 0
    for f in sorted(mdir.glob('*.npy')):
        id_ = f.stem.split('_')[0]
        if is_celebdf(id_):
            kept.append({'video_id': f'{method}__{f.stem}', 'class': 'fake',
                         'method': method, 'identity': id_, 'path': str(f)})
        else:
            excl += 1
    per_method_ids[method]  = {r['identity'] for r in kept}
    per_method_excl[method] = excl
    fake_rows.extend(kept)
    print(f"  {method}: {len(kept)} kept, {excl} excluded  |  "
          f"{len(per_method_ids[method])} identities")

# Uniformity check
common = set.intersection(*per_method_ids.values())
union  = set.union(*per_method_ids.values())
if common != union:
    asym = {m: sorted(union - per_method_ids[m], key=id_sortkey) for m in METHODS
            if union - per_method_ids[m]}
    sys.exit(f"FAIL: generator identity sets are not identical. Missing:\n{asym}")
if len(common) != EXPECTED_IDENTITIES:
    sys.exit(f"FAIL: expected {EXPECTED_IDENTITIES} fake identities, got {len(common)}")

# Confirm real and fake identity sets match
if real_ids != common:
    sys.exit(f"FAIL: real and fake identity sets differ.\n"
             f"  real only: {real_ids - common}\n  fake only: {common - real_ids}")

total_excl = sum(per_method_excl.values())
print()
print(f"  Total VoxCeleb excluded: {total_excl}")
print(f"  Total Celeb-DF++ fake:   {len(fake_rows)}")
print(f"  All 7 generators share identical 59-identity set ✓")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Assign splits
# ─────────────────────────────────────────────────────────────────────────────
print("[3] Assigning identity-level splits ...")

all_fake_ids = common
train_ids = all_fake_ids - EVAL_IDS

# Sanity: no eval identity overlaps train
assert not (TEST_IDS & train_ids), "TEST_IDS overlap TRAIN"
assert not (VAL_IDS  & train_ids), "VAL_IDS overlap TRAIN"
assert len(TEST_IDS) == 9 and len(VAL_IDS) == 9
assert TEST_IDS <= all_fake_ids, f"TEST id(s) missing from fake: {TEST_IDS - all_fake_ids}"
assert VAL_IDS  <= all_fake_ids, f"VAL id(s) missing from fake:  {VAL_IDS  - all_fake_ids}"

print(f"  Train identities ({len(train_ids)}): {sorted(train_ids, key=id_sortkey)}")
print(f"  Val   identities ({len(VAL_IDS)}):  {sorted(VAL_IDS, key=id_sortkey)}")
print(f"  Test  identities ({len(TEST_IDS)}):  {sorted(TEST_IDS, key=id_sortkey)}")
print()

def assign(id_):
    if id_ in TEST_IDS: return 'test'
    if id_ in VAL_IDS:  return 'val'
    return 'train'

for r in real_rows + fake_rows:
    r['split'] = assign(r['identity'])

# ─────────────────────────────────────────────────────────────────────────────
# 4. Build DataFrames and validate
# ─────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame(real_rows + fake_rows,
                  columns=['split','video_id','class','method','identity','path'])

real_df = df[df['class'] == 'real']
fake_df = df[df['class'] == 'fake']

real_tr = real_df[real_df['split'] == 'train']
real_va = real_df[real_df['split'] == 'val']
real_te = real_df[real_df['split'] == 'test']

if len(real_tr) != EXPECTED_REAL_TRAIN:
    sys.exit(f"FAIL: expected {EXPECTED_REAL_TRAIN} train real, got {len(real_tr)}")

print("[4] Count summary")
print()
print(f"  REAL")
print(f"    train: {len(real_tr)} windows  ({sorted(real_tr['identity'].unique(), key=id_sortkey)})")
print(f"    val:   {len(real_va)} windows  ({sorted(real_va['identity'].unique(), key=id_sortkey)})")
print(f"    test:  {len(real_te)} windows  ({sorted(real_te['identity'].unique(), key=id_sortkey)})")
print(f"    total: {len(real_df)}")
print()
print(f"  FAKE  (all generators combined)")
for split in ['train','val','test']:
    sub = fake_df[fake_df['split'] == split]
    print(f"    {split}: {len(sub)} windows  ({sorted(sub['identity'].unique(), key=id_sortkey)})")
print(f"    total: {len(fake_df)}")
print()

print(f"  FAKE  per generator")
print(f"  {'Generator':<18} {'train':>6} {'val':>6} {'test':>6} {'total':>6}  eval_ids")
for method in METHODS:
    mdf = fake_df[fake_df['method'] == method]
    tr  = mdf[mdf['split']=='train']
    va  = mdf[mdf['split']=='val']
    te  = mdf[mdf['split']=='test']
    ev_ids = sorted((mdf[mdf['split'].isin(['val','test'])]['identity'].unique()), key=id_sortkey)
    print(f"  {method:<18} {len(tr):>6} {len(va):>6} {len(te):>6} {len(mdf):>6}  ({len(ev_ids)} ids: {ev_ids})")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save
# ─────────────────────────────────────────────────────────────────────────────
print("[5] Saving ...")
fake_df[['video_id','class','method','identity','path']].to_csv(MANIFEST_PATH, index=False)
print(f"  Manifest saved → {MANIFEST_PATH}  ({len(fake_df)} rows)")

df.to_csv(SPLIT_PATH, index=False)
print(f"  Split saved    → {SPLIT_PATH}  ({len(df)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. JSON summary for Phase 2 scripts to read
# ─────────────────────────────────────────────────────────────────────────────
summary = {
    'n_real':          int(len(real_df)),
    'n_fake':          int(len(fake_df)),
    'n_real_train':    int(len(real_tr)),
    'n_real_val':      int(len(real_va)),
    'n_real_test':     int(len(real_te)),
    'n_identities':    EXPECTED_IDENTITIES,
    'train_ids':       sorted(train_ids, key=id_sortkey),
    'val_ids':         sorted(VAL_IDS, key=id_sortkey),
    'test_ids':        sorted(TEST_IDS, key=id_sortkey),
    'per_method': {m: {
        'train': int(len(fake_df[(fake_df['method']==m)&(fake_df['split']=='train')])),
        'val':   int(len(fake_df[(fake_df['method']==m)&(fake_df['split']=='val')])),
        'test':  int(len(fake_df[(fake_df['method']==m)&(fake_df['split']=='test')])),
        'total': int(len(fake_df[fake_df['method']==m])),
    } for m in METHODS},
}
with open(OUT_DIR / 'split_full59_summary.json', 'w') as fh:
    json.dump(summary, fh, indent=2)
print(f"  Summary JSON   → data/results/split_full59_summary.json")
print()
print("Phase 1 complete. Awaiting confirmation before Phase 2.")
