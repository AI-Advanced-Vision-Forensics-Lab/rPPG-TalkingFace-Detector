#!/usr/bin/env python3
"""
00_extract_waveforms.py — rPPG waveform extraction for Celeb-DF-v3.

Fake videos (Celeb-synthesis):
  - One waveform per video, early-stop at 160 frames
  - Saved per method: waveforms/CelebDF/{category}/{method}/{stem}.npy
  - video_id = {Method}__{stem}

Real videos (Celeb-real):
  - Stride-60 sliding windows
  - Saved: waveforms/CelebDF/Celeb-real/{stem}_w{k}.npy
  - video_id = {stem}_w{k}

Manifest: data/celebdf_wave_manifest.csv
Failures: data/waveforms/CelebDF/extraction_failures.log
Resumable: skips video_ids already in manifest with status=ok.
"""

import csv
import sys
import time
import urllib.request
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

DATA_ROOT   = Path(_args.data_root)
OUT_DIR = Path(_args.out_dir) if _args.out_dir else DATA_ROOT / 'results'
CELEB_ROOT  = DATA_ROOT / "Celeb-DF-v3"
REPO_DIR    = Path(_args.rhythmformer_dir)
WAVE_ROOT   = DATA_ROOT / "waveforms" / "CelebDF"
N_FRAMES    = 160
SIZE        = 128
EXPAND      = 1.5
MIN_FRAMES  = 60
STRIDE      = 60


# ── RhythmFormer ──────────────────────────────────────────────────────────────

def load_rhythmformer(device):
    weights = REPO_DIR / "PreTrainedModels" / "UBFC_cross_RhythmFormer.pth"
    assert REPO_DIR.exists(), f"Repo not found: {REPO_DIR}"
    assert weights.exists(), f"Weights not found: {weights}"
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    from neural_methods.model.RhythmFormer import RhythmFormer
    model = RhythmFormer()
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, "State dict mismatch"
    model = model.to(device).eval()
    print(f"  RhythmFormer loaded on {device} | "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    return model


def load_face_detector():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    model_path = DATA_ROOT / "blaze_face_short_range.tflite"
    if not model_path.exists():
        url = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
               "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")
        print("  Downloading BlazeFace model...")
        urllib.request.urlretrieve(url, model_path)
    return mp_vision.FaceDetector.create_from_options(
        mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=0.5,
        )
    )


# ── Inference helpers ─────────────────────────────────────────────────────────

def detect_bbox(frames, face_detector):
    import mediapipe as mp
    for frame in frames[:30]:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = face_detector.detect(mp_img)
        if not result.detections:
            continue
        det  = max(result.detections, key=lambda d: d.categories[0].score)
        bb   = det.bounding_box
        h, w = frame.shape[:2]
        cx   = bb.origin_x + bb.width  / 2
        cy   = bb.origin_y + bb.height / 2
        side = max(bb.width, bb.height) * EXPAND
        x1   = int(max(0, cx - side / 2))
        y1   = int(max(0, cy - side / 2))
        x2   = int(min(w,  cx + side / 2))
        y2   = int(min(h,  cy + side / 2))
        return (x1, y1, x2, y2)
    return None


def frames_to_waveform(frames, face_detector, rhythmformer, device):
    bbox = detect_bbox(frames, face_detector)
    if bbox is None:
        raise ValueError("No face detected in first 30 frames")
    x1, y1, x2, y2 = bbox
    crops = np.stack([
        cv2.resize(f[y1:y2, x1:x2], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        for f in frames
    ]).astype(np.float32)
    mean, std = crops.mean(), crops.std()
    if std < 1e-6:
        raise ValueError("Degenerate crop (constant pixels)")
    crops = (crops - mean) / std
    tensor = (torch.from_numpy(crops)
              .permute(0, 3, 1, 2)
              .unsqueeze(0)
              .contiguous()
              .to(device))
    with torch.no_grad():
        waveform = rhythmformer(tensor).squeeze(0).cpu().numpy()
    return waveform.astype(np.float32)


# ── Job builders ──────────────────────────────────────────────────────────────

def build_fake_jobs():
    """One job per fake video. Output: (video_id, out_path, video_path, category, method, None)"""
    jobs = []
    synthesis_root = CELEB_ROOT / "Celeb-synthesis"
    for category in ("TalkingFace",):
        cat_dir = synthesis_root / category
        if not cat_dir.exists():
            continue
        for method_dir in sorted(cat_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method   = method_dir.name
            out_dir  = WAVE_ROOT / category / method
            out_dir.mkdir(parents=True, exist_ok=True)
            for vp in sorted(method_dir.glob("*.mp4")):
                vid_id   = f"{method}__{vp.stem}"
                out_path = out_dir / f"{vp.stem}.npy"
                jobs.append({
                    "video_id": vid_id, "class": "fake",
                    "source": "Celeb-synthesis", "category": category,
                    "method": method, "video_path": vp,
                    "out_path": out_path, "start": None,
                })
    return jobs


def build_real_jobs():
    """Stride-60 windows for Celeb-real."""
    jobs = []
    for source_name in ("Celeb-real",):
        src_dir = CELEB_ROOT / source_name
        if not src_dir.exists():
            continue
        out_dir = WAVE_ROOT / source_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for vp in sorted(src_dir.glob("*.mp4")):
            cap      = cv2.VideoCapture(str(vp))
            n_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n_native < MIN_FRAMES:
                continue
            max_start = max(1, n_native - N_FRAMES + 1)
            starts    = list(range(0, max_start, STRIDE))
            for k, start in enumerate(starts):
                vid_id   = f"{vp.stem}_w{k}"
                out_path = out_dir / f"{vid_id}.npy"
                jobs.append({
                    "video_id": vid_id, "class": "real",
                    "source": source_name, "category": source_name,
                    "method": "real", "video_path": vp,
                    "out_path": out_path, "start": start,
                })
    return jobs


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_fake(job, face_detector, rhythmformer, device):
    vp       = job["video_path"]
    cap      = cv2.VideoCapture(str(vp))
    n_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_native < MIN_FRAMES:
        cap.release()
        raise ValueError(f"Too short: {n_native} frames")
    max_read = N_FRAMES if n_native >= N_FRAMES else n_native
    frames   = []
    for _ in range(max_read + 10):
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < MIN_FRAMES:
        raise ValueError(f"Too short after read: {len(frames)} frames")
    if len(frames) < N_FRAMES:
        idx    = np.linspace(0, len(frames) - 1, N_FRAMES).round().astype(int)
        frames = [frames[i] for i in idx]
    else:
        frames = frames[:N_FRAMES]
    return frames_to_waveform(frames, face_detector, rhythmformer, device)


# Cache for real video frames to avoid re-reading the same file for every window
_real_cache = {}


def extract_real_window(job, face_detector, rhythmformer, device):
    vp    = job["video_path"]
    start = job["start"]
    key   = str(vp)

    if key not in _real_cache:
        cap    = cv2.VideoCapture(str(vp))
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        _real_cache[key] = frames
        # Evict cache entries beyond the 2 most recent videos to cap memory
        if len(_real_cache) > 2:
            oldest = next(iter(_real_cache))
            del _real_cache[oldest]

    all_frames = _real_cache[key]
    window     = all_frames[start:start + N_FRAMES]
    if len(window) < MIN_FRAMES:
        raise ValueError(f"Window too short: {len(window)} frames")
    if len(window) < N_FRAMES:
        idx    = np.linspace(0, len(window) - 1, N_FRAMES).round().astype(int)
        window = [window[i] for i in idx]
    return frames_to_waveform(window, face_detector, rhythmformer, device)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = DATA_ROOT / "celebdf_wave_manifest.csv"
    fail_log_path = WAVE_ROOT / "extraction_failures.log"
    WAVE_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  rPPG WAVEFORM EXTRACTION — Celeb-DF-v3")
    print("=" * 70)
    print(f"  Device: {device}")

    print("\n  Loading models...")
    rhythmformer  = load_rhythmformer(device)
    face_detector = load_face_detector()

    print("\n  Building job list...")
    fake_jobs = build_fake_jobs()
    real_jobs = build_real_jobs()
    all_jobs  = fake_jobs + real_jobs
    print(f"  Fake jobs: {len(fake_jobs)}")
    print(f"  Real jobs: {len(real_jobs)}")
    print(f"  Total:     {len(all_jobs)}")

    # Resume from manifest
    done = set()
    manifest_rows = []
    fieldnames = ["video_id", "class", "source", "category", "method", "status", "error"]
    if manifest_path.exists():
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                manifest_rows.append(row)
                if row["status"] == "ok":
                    done.add(row["video_id"])
        print(f"  Already done: {len(done)} | Remaining: {len(all_jobs) - len(done)}")

    t0     = time.time()
    n_ok   = 0
    n_fail = 0
    new_rows = []

    with open(fail_log_path, "a") as flog:
        for i, job in enumerate(all_jobs):
            vid_id = job["video_id"]
            if vid_id in done:
                n_ok += 1
                continue

            try:
                if not job["out_path"].exists():
                    if job["class"] == "fake":
                        w = extract_fake(job, face_detector, rhythmformer, device)
                    else:
                        w = extract_real_window(job, face_detector, rhythmformer, device)
                    np.save(job["out_path"], w)
                new_rows.append({
                    "video_id": vid_id, "class": job["class"],
                    "source": job["source"], "category": job["category"],
                    "method": job["method"], "status": "ok", "error": "",
                })
                n_ok += 1
                done.add(vid_id)
            except Exception as e:
                msg = str(e)[:200]
                new_rows.append({
                    "video_id": vid_id, "class": job["class"],
                    "source": job["source"], "category": job["category"],
                    "method": job["method"], "status": "fail", "error": msg,
                })
                flog.write(f"{vid_id}\t{msg}\n")
                n_fail += 1

            processed = n_ok + n_fail
            if processed % 200 == 0 or processed == 1:
                elapsed   = time.time() - t0
                rate      = processed / elapsed
                remaining = (len(all_jobs) - len(done)) / max(rate, 1e-6)
                print(f"  [{processed:>6}/{len(all_jobs)}] "
                      f"ok={n_ok} fail={n_fail} | "
                      f"{rate:.1f} v/s | ETA {remaining/3600:.1f} hr")

            if len(new_rows) >= 500:
                manifest_rows.extend(new_rows)
                new_rows = []
                with open(manifest_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader(); w.writerows(manifest_rows)

    manifest_rows.extend(new_rows)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(manifest_rows)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  EXTRACTION COMPLETE  ({elapsed/3600:.2f} hr)")
    print(f"{'='*70}")
    print(f"  OK:       {n_ok}")
    print(f"  Failed:   {n_fail}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Failures: {fail_log_path}")


if __name__ == "__main__":
    main()
