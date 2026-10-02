"""
run_task3_method_label_validation.py — Method-label internal consistency check.

For each of the seven generator buckets (AniTalker, EDTalk, EchoMimic, FLOAT,
IP_LAP, Real3DPortrait, SadTalker), reports per-video technical metadata:
  - Resolution (width × height)
  - Frame rate
  - Codec + encoder tag
  - File size
  - Native frame count

Then assesses whether buckets separate cleanly on these attributes.

Samples up to N_SAMPLE videos per generator; uses ffprobe for metadata.

Output: data/results/task3_method_label_validation.json
"""

import json, os, subprocess, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

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
SPLIT_CSV  = DATA_ROOT / 'dataset_split_full59.csv'
VIDEO_FAKE = DATA_ROOT / 'videos/fake'
VIDEO_REAL = DATA_ROOT / 'videos/real'
OUT_JSON   = OUT_DIR / 'task3_method_label_validation.json'

METHODS   = ['AniTalker','EDTalk','EchoMimic','FLOAT','IP_LAP','Real3DPortrait','SadTalker']
N_SAMPLE  = 150   # max videos per generator to probe

# ── ffprobe metadata ──────────────────────────────────────────────────────────
def probe_video(path):
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
               '-show_streams', '-show_format', str(path)]
        r = subprocess.run(cmd, capture_output=True, timeout=15)
        if r.returncode != 0:
            return None
        d = json.loads(r.stdout)
        streams = d.get('streams', [])
        vid = next((s for s in streams if s.get('codec_type')=='video'), None)
        if vid is None:
            return None
        fmt = d.get('format', {})

        # Frame rate: r_frame_rate is a fraction string like "25/1" or "30000/1001"
        rfps = vid.get('r_frame_rate', '0/1')
        try:
            num, den = rfps.split('/')
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        except:
            fps = 0.0

        # nb_frames: may be absent; fall back to duration × fps
        nb_frames = vid.get('nb_frames')
        if nb_frames:
            try: nb_frames = int(nb_frames)
            except: nb_frames = None
        if nb_frames is None:
            dur = vid.get('duration') or fmt.get('duration')
            if dur:
                try: nb_frames = int(float(dur) * fps)
                except: nb_frames = None

        encoder_tag = vid.get('tags', {}).get('encoder', None) or \
                      vid.get('tags', {}).get('ENCODER', None) or \
                      fmt.get('tags', {}).get('encoder', None) or \
                      fmt.get('tags', {}).get('ENCODER', None)

        return {
            'width':    int(vid.get('width', 0)),
            'height':   int(vid.get('height', 0)),
            'fps':      round(fps, 3),
            'codec':    vid.get('codec_name', 'unknown'),
            'encoder':  encoder_tag,
            'file_bytes': int(fmt.get('size', os.path.getsize(path))),
            'nb_frames': nb_frames,
            'pix_fmt':  vid.get('pix_fmt', 'unknown'),
        }
    except Exception as e:
        return None

# ── Sample videos ─────────────────────────────────────────────────────────────
def sample_method_videos(method, n=N_SAMPLE, seed=42):
    d = VIDEO_FAKE / method
    if not d.exists():
        return []
    vids = sorted(d.glob('*.mp4'))
    rng = np.random.default_rng(seed)
    if len(vids) > n:
        idx = rng.choice(len(vids), n, replace=False)
        vids = [vids[i] for i in sorted(idx)]
    return vids

# ── Summarise metadata list ───────────────────────────────────────────────────
def summarise(records):
    if not records:
        return {}
    def cnt(key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return dict(Counter(vals).most_common(10))
    def stats(key):
        vals = [r[key] for r in records if r.get(key) is not None and isinstance(r[key], (int,float))]
        if not vals: return None
        return {'mean': round(float(np.mean(vals)), 2),
                'std':  round(float(np.std(vals)),  2),
                'min':  round(float(np.min(vals)),  2),
                'max':  round(float(np.max(vals)),  2),
                'p25':  round(float(np.percentile(vals, 25)), 2),
                'p75':  round(float(np.percentile(vals, 75)), 2)}

    resolutions = Counter(f"{r['width']}x{r['height']}" for r in records if r.get('width'))
    return {
        'n_probed':      len(records),
        'resolution':    dict(resolutions.most_common(5)),
        'fps':           cnt('fps'),
        'codec':         cnt('codec'),
        'encoder':       cnt('encoder'),
        'pix_fmt':       cnt('pix_fmt'),
        'file_bytes':    stats('file_bytes'),
        'nb_frames':     stats('nb_frames'),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
print(f"Probing up to {N_SAMPLE} videos per generator ...", flush=True)
t0 = time.time()

results = {}
all_fps = {}
all_res = {}
all_codecs = {}
all_nb_frames = {}

for method in METHODS:
    vids = sample_method_videos(method)
    print(f"  {method}: {len(vids)} videos ...", end='', flush=True)
    records = []
    for v in vids:
        m = probe_video(v)
        if m:
            records.append(m)
    s = summarise(records)
    results[method] = s
    all_fps[method]      = [r['fps'] for r in records]
    all_res[method]      = [f"{r['width']}x{r['height']}" for r in records if r.get('width')]
    all_codecs[method]   = [r['codec'] for r in records]
    all_nb_frames[method]= [r['nb_frames'] for r in records if r.get('nb_frames')]
    dominant_res  = max(Counter(all_res[method]).items(), key=lambda x: x[1])[0] if all_res[method] else 'n/a'
    dominant_fps  = round(float(np.mean(all_fps[method])), 2) if all_fps[method] else 'n/a'
    dominant_codec= max(Counter(all_codecs[method]).items(), key=lambda x: x[1])[0] if all_codecs[method] else 'n/a'
    print(f"  {s['n_probed']} ok | res={dominant_res} fps={dominant_fps} codec={dominant_codec}", flush=True)

# Also probe real videos for reference
real_vids = sorted(VIDEO_REAL.glob('*.mp4'))
real_sample = real_vids[:N_SAMPLE]
print(f"  Real: {len(real_sample)} videos ...", end='', flush=True)
real_records = [probe_video(v) for v in real_sample]
real_records = [r for r in real_records if r]
results['real'] = summarise(real_records)
all_fps['real']       = [r['fps'] for r in real_records]
all_res['real']       = [f"{r['width']}x{r['height']}" for r in real_records if r.get('width')]
all_codecs['real']    = [r['codec'] for r in real_records]
all_nb_frames['real'] = [r['nb_frames'] for r in real_records if r.get('nb_frames')]
print(f"  {len(real_records)} ok", flush=True)

# ── Separability assessment ───────────────────────────────────────────────────
print()
print("=" * 65)
print("TASK 3 — Method Label Validation")
print("=" * 65)

assessment_rows = []
for m in METHODS + ['real']:
    s = results.get(m, {})
    fps_vals = all_fps.get(m, [])
    res_vals = all_res.get(m, [])
    codec_vals = all_codecs.get(m, [])
    nf_vals = all_nb_frames.get(m, [])
    dominant_res   = Counter(res_vals).most_common(1)[0][0] if res_vals else 'n/a'
    dominant_fps   = round(float(np.mean(fps_vals)), 2) if fps_vals else 0
    fps_unique     = sorted(set(round(f, 1) for f in fps_vals))
    dominant_codec = Counter(codec_vals).most_common(1)[0][0] if codec_vals else 'n/a'
    mean_frames    = round(float(np.mean(nf_vals)), 1) if nf_vals else 0
    print(f"  {m:<20}  res={dominant_res:<12}  fps={dominant_fps:<6}  "
          f"codec={dominant_codec:<8}  mean_frames={mean_frames:.0f}")
    assessment_rows.append({'method': m, 'dominant_res': dominant_res,
                             'fps_mean': dominant_fps, 'fps_unique': fps_unique,
                             'codec': dominant_codec, 'mean_frames': mean_frames})

print()

# Check if fps cleanly separates generators
fps_by_method = {m: round(float(np.mean(all_fps[m])), 2) if all_fps[m] else 0
                 for m in METHODS + ['real']}
unique_fps_groups = defaultdict(list)
for m, f in fps_by_method.items():
    unique_fps_groups[round(f, 0)].append(m)
fps_separates = len(unique_fps_groups) > 1
fps_notes = {str(int(k)): v for k, v in unique_fps_groups.items()}

# Check resolution
res_by_method = {m: Counter(all_res[m]).most_common(1)[0][0] if all_res[m] else 'n/a'
                 for m in METHODS + ['real']}
unique_res_groups = defaultdict(list)
for m, r in res_by_method.items():
    unique_res_groups[r].append(m)
res_separates = len(unique_res_groups) > 1

# Check nb_frames
nf_by_method = {m: (round(float(np.mean(all_nb_frames[m])), 1) if all_nb_frames[m] else None)
                for m in METHODS + ['real']}

print("Separability assessment:")
print(f"  FPS separates generators: {fps_separates}")
for fps_val, methods in sorted(fps_notes.items(), key=lambda x: float(x[0])):
    print(f"    {fps_val} fps: {methods}")
print(f"  Resolution separates generators: {res_separates}")
for res_val, methods in unique_res_groups.items():
    print(f"    {res_val}: {methods}")

# Verdict
# Clean separation = each generator has a unique combination of (fps, resolution, codec)
combos = {}
for m in METHODS:
    combo = (fps_by_method[m], res_by_method.get(m, 'n/a'),
             Counter(all_codecs.get(m, [])).most_common(1)[0][0] if all_codecs.get(m) else 'n/a')
    combos[m] = combo

unique_combos = len(set(combos.values()))
cleanly_separable = (unique_combos == len(METHODS))

print(f"\n  Unique (fps, resolution, codec) combos across 7 generators: {unique_combos}/7")
if cleanly_separable:
    print("  → Buckets are CLEANLY SEPARABLE on technical attributes.")
    print("    Supports internal consistency of label assignment.")
else:
    overlapping = [(m1, m2) for i, m1 in enumerate(METHODS)
                   for m2 in METHODS[i+1:]
                   if combos[m1] == combos[m2]]
    print(f"  → Buckets OVERLAP on technical attributes ({len(overlapping)} pairs).")
    print(f"    Overlapping pairs: {overlapping}")
    print("    This weakens the claim of internally consistent label assignment.")

print(f"\nRuntime: {(time.time()-t0)/60:.1f} min")

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'experiment': 'method_label_validation',
    'n_sample_per_method': N_SAMPLE,
    'per_method': results,
    'fps_by_method': fps_by_method,
    'resolution_dominant': res_by_method,
    'mean_frames_by_method': nf_by_method,
    'codec_by_method': {m: Counter(all_codecs.get(m,[])).most_common(3)
                        for m in METHODS + ['real']},
    'separability': {
        'fps_separates_generators': fps_separates,
        'fps_groups': fps_notes,
        'resolution_separates_generators': res_separates,
        'resolution_groups': {k: v for k, v in unique_res_groups.items()},
        'n_unique_combos_fps_res_codec': unique_combos,
        'cleanly_separable': cleanly_separable,
        'combo_per_method': {m: list(v) for m, v in combos.items()},
        'overlapping_pairs': ([] if cleanly_separable else
                              [(m1, m2) for i, m1 in enumerate(METHODS)
                               for m2 in METHODS[i+1:] if combos[m1] == combos[m2]]),
    }
}
with open(OUT_JSON, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"Saved → {OUT_JSON}")
