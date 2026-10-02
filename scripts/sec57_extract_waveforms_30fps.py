#!/usr/bin/env python3
"""
extract_waveforms_30fps.py — fps-normalized waveform extraction.

Resamples every fake video to 30 fps via nearest-neighbour frame selection
against wall-clock time before RhythmFormer inference.  Real videos are already
30 fps; their existing waveforms are reused unchanged.

Resampling method:
    For target frame i (0…159) at 30 fps, select source frame
        k_i = round(i × fps_native / 30.0)
    clipped to [0, n_native−1].  This aligns each target frame to the
    nearest captured frame at the corresponding wall-clock time, NOT a
    linspace stretch over the full clip.

Short-video policy (applied identically to all fakes):
    Require n_native ≥ ceil(160 × fps_native / 30).
    25 fps → need ≥ 134 frames (5.36 s of source).
    24 fps → need ≥ 128 frames (5.33 s of source).
    Videos below threshold are dropped (logged as "too_short").

Missing source videos (id53–id61 on disk) are logged as "missing_source".

Output directory: data/waveforms/CelebDF_30fps/TalkingFace/{method}/
Resumable: already-extracted files are skipped.
"""

import csv, json, math, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch

# ── Paths (configurable; see README) ─────────────────────────────────────────
import argparse as _argparse
_ap = _argparse.ArgumentParser(description=(__doc__ or '').strip().split('\n')[0])
_ap.add_argument('--data-root', default='data',
                 help='data folder laid out as described in the README (default: ./data)')
_ap.add_argument('--out-dir', default=None,
                 help='folder for result JSONs (default: <data-root>/results)')
_ap.add_argument('--rhythmformer-dir', default='RhythmFormer',
                 help='clone of github.com/zizheng-guo/RhythmFormer (default: ./RhythmFormer)')
_args = _ap.parse_args()

DATA_ROOT  = Path(_args.data_root)
OUT_DIR = Path(_args.out_dir) if _args.out_dir else DATA_ROOT / 'results'
REPO_DIR   = Path(_args.rhythmformer_dir)
WAVE_OUT   = DATA_ROOT / 'waveforms/CelebDF_30fps/TalkingFace'
VIDEO_ROOT = DATA_ROOT / 'videos/fake'
LOG_CSV    = DATA_ROOT / 'waveforms/CelebDF_30fps/extraction_log.csv'

METHODS    = ['AniTalker','EDTalk','EchoMimic','FLOAT','IP_LAP','Real3DPortrait','SadTalker']
N_FRAMES   = 160
FPS_TARGET = 30.0
SIZE       = 128
EXPAND     = 1.5
FLUSH_EVERY = 200

# ── Helpers ───────────────────────────────────────────────────────────────────
def min_native_frames(fps_native):
    """Minimum native frames to yield a full 5.33-second window at 30 fps."""
    return math.ceil(N_FRAMES * fps_native / FPS_TARGET)

def fps_normalized_indices(fps_native):
    """Return 160 source-frame indices aligned to a 30-fps timeline."""
    raw = np.arange(N_FRAMES) * (fps_native / FPS_TARGET)
    return np.round(raw).astype(int)

# ── Model loader ──────────────────────────────────────────────────────────────
def load_rhythmformer(device):
    weights = REPO_DIR / 'PreTrainedModels' / 'UBFC_cross_RhythmFormer.pth'
    assert REPO_DIR.exists(), f"RhythmFormer repo not found at {REPO_DIR}"
    assert weights.exists(),  f"Weights not found at {weights}"
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    from neural_methods.model.RhythmFormer import RhythmFormer
    model = RhythmFormer()
    try:
        state = torch.load(weights, map_location='cpu', weights_only=True)
    except Exception:
        state = torch.load(weights, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()

# ── Face detector ─────────────────────────────────────────────────────────────
def build_face_detector():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    model_path = DATA_ROOT / 'blaze_face_short_range.tflite'
    return mp_vision.FaceDetector.create_from_options(
        mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=0.5))

# ── Single-video extraction ───────────────────────────────────────────────────
def extract_one(vp, face_detector, rhythmformer, device):
    """Returns (waveform: np.ndarray, meta: dict) or raises."""
    import mediapipe as mp

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise IOError(f"Cannot open {vp}")
    fps_native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    n_native = len(frames)

    # Short-video check
    thresh = min_native_frames(fps_native)
    if n_native < thresh:
        raise ValueError(f"too_short:{n_native}<{thresh}@{fps_native:.0f}fps")

    # Nearest-neighbour fps-normalised frame selection
    idx = fps_normalized_indices(fps_native)
    idx = np.clip(idx, 0, n_native - 1)
    selected = [frames[i] for i in idx]   # exactly N_FRAMES frames

    # Face detection on first 30 selected frames
    bbox = None
    for f in selected[:30]:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=f)
        result = face_detector.detect(mp_img)
        if not result.detections: continue
        det = max(result.detections, key=lambda d: d.categories[0].score)
        bb  = det.bounding_box
        h, w = f.shape[:2]
        cx   = bb.origin_x + bb.width  / 2
        cy   = bb.origin_y + bb.height / 2
        side = max(bb.width, bb.height) * EXPAND
        x1   = int(max(0, cx - side / 2));  y1 = int(max(0, cy - side / 2))
        x2   = int(min(w, cx + side / 2));  y2 = int(min(h, cy + side / 2))
        bbox = (x1, y1, x2, y2); break
    if bbox is None:
        raise ValueError("no_face")

    x1, y1, x2, y2 = bbox
    crops = np.stack([
        cv2.resize(f[y1:y2, x1:x2], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        for f in selected
    ]).astype(np.float32)
    mean, std = crops.mean(), crops.std()
    if std < 1e-6: raise ValueError("constant_pixels")
    crops = (crops - mean) / std
    tensor = torch.from_numpy(crops).permute(0, 3, 1, 2).unsqueeze(0).contiguous().to(device)

    with torch.no_grad():
        wave = rhythmformer(tensor).squeeze(0).cpu().numpy()

    return wave, {'n_native': n_native, 'fps_native': round(fps_native, 2),
                  'thresh': thresh, 'idx_max': int(idx.max())}

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:      {device}")
    print(f"Output root: {WAVE_OUT}")
    print(f"FPS target:  {FPS_TARGET} fps  →  {N_FRAMES} frames = {N_FRAMES/FPS_TARGET:.2f}s window")
    print(f"Short-video: drop (no linspace, no padding)")
    print()

    # Create output dirs
    for m in METHODS:
        (WAVE_OUT / m).mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading RhythmFormer ...", flush=True)
    rhythmformer = load_rhythmformer(device)
    print("Loading face detector ...", flush=True)
    face_detector = build_face_detector()
    print()

    # Load existing log to support resumption
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    log_rows = []
    if LOG_CSV.exists():
        with open(LOG_CSV) as f:
            for row in csv.DictReader(f):
                log_rows.append(row)
                if row['status'] == 'ok':
                    done.add(row['video_id'])
        print(f"Resuming — {len(done)} already extracted")

    # Build job list: one job per fake source video, 59-id corpus
    import pandas as pd
    split_df = pd.read_csv(DATA_ROOT / 'dataset_split_full59.csv')
    fakes = split_df[split_df['class'] == 'fake'].copy()

    # Build unique (method, stem) jobs from the split
    jobs = []
    seen = set()
    for _, row in fakes.iterrows():
        stem   = Path(row['path']).stem
        method = row['method']
        key    = f"{method}__{stem}"
        if key in seen: continue
        seen.add(key)
        vp      = VIDEO_ROOT / method / f"{stem}.mp4"
        out_npy = WAVE_OUT / method / f"{stem}.npy"
        jobs.append((method, stem, vp, out_npy))

    print(f"Unique fake source videos in 59-id corpus: {len(jobs)}")
    pending = [j for j in jobs if j[1] not in done and f"{j[0]}__{j[1]}" not in done]
    print(f"Pending (not yet extracted): {len(pending)}")
    print()

    # Counters
    n_ok = n_fail_missing = n_fail_short = n_fail_noface = n_fail_other = 0
    new_rows = []

    t0 = time.time()
    for i, (method, stem, vp, out_npy) in enumerate(pending):
        # Check if already done under composite key
        ckey = f"{method}__{stem}"
        if ckey in done:
            continue

        row = {'video_id': stem, 'method': method, 'status': '', 'error': '',
               'n_native': '', 'fps_native': '', 'thresh': ''}

        if not vp.exists():
            row.update(status='missing_source', error='source video not on disk')
            n_fail_missing += 1
        elif out_npy.exists():
            row.update(status='ok')
            n_ok += 1
            done.add(ckey)
        else:
            try:
                wave, meta = extract_one(vp, face_detector, rhythmformer, device)
                np.save(out_npy, wave)
                row.update(status='ok', **{k: str(v) for k, v in meta.items()})
                n_ok += 1
                done.add(ckey)
            except ValueError as e:
                err = str(e)
                if 'too_short' in err:
                    row.update(status='too_short', error=err)
                    n_fail_short += 1
                elif 'no_face' in err:
                    row.update(status='no_face', error=err)
                    n_fail_noface += 1
                else:
                    row.update(status='fail', error=err[:200])
                    n_fail_other += 1
            except Exception as e:
                row.update(status='fail', error=str(e)[:200])
                n_fail_other += 1

        new_rows.append(row)

        # Progress
        processed = i + 1
        if processed % 100 == 0 or processed == 1 or processed == len(pending):
            elapsed = time.time() - t0
            rate    = processed / max(elapsed, 1e-6)
            eta_min = (len(pending) - processed) / rate / 60
            print(f"  [{processed:>6}/{len(pending)}] ok={n_ok} "
                  f"short={n_fail_short} missing={n_fail_missing} "
                  f"noface={n_fail_noface} other={n_fail_other} "
                  f"| {rate:.1f} vid/s | ETA {eta_min:.0f} min", flush=True)

        # Flush log
        if len(new_rows) >= FLUSH_EVERY:
            log_rows.extend(new_rows); new_rows = []
            _write_log(LOG_CSV, log_rows)

    log_rows.extend(new_rows)
    _write_log(LOG_CSV, log_rows)

    elapsed = time.time() - t0
    total_ok      = n_ok
    total_missing = n_fail_missing
    total_short   = n_fail_short
    total_other   = n_fail_noface + n_fail_other

    print()
    print("=" * 60)
    print("  EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  OK (extracted):      {total_ok}")
    print(f"  Dropped — missing source: {total_missing}")
    print(f"  Dropped — too short:      {total_short}")
    print(f"  Dropped — no face/other:  {total_other}")
    print(f"  Runtime: {elapsed/60:.1f} min")

    # Per-method summary from log
    import collections
    m_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    with open(LOG_CSV) as f:
        for row in csv.DictReader(f):
            m_counts[row['method']][row['status']] += 1
    print()
    print(f"  {'Method':<18}  {'ok':>5}  {'short':>6}  {'missing':>7}  {'noface':>6}  {'other':>5}")
    for m in METHODS:
        d = m_counts[m]
        print(f"  {m:<18}  {d.get('ok',0):>5}  {d.get('too_short',0):>6}  "
              f"{d.get('missing_source',0):>7}  {d.get('no_face',0):>6}  "
              f"{d.get('fail',0):>5}")
    print()
    print(f"  Log: {LOG_CSV}")
    print(f"  Waveforms: {WAVE_OUT}")
    print()
    print("Next step: python3 src/build_split_30fps.py")


def _write_log(path, rows):
    fieldnames = ['video_id','method','status','error','n_native','fps_native','thresh','idx_max']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
