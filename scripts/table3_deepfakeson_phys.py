"""
DeepFakesON-Phys reproduction — Paper 1 baseline
Paper: arXiv:2010.00400 (Hernandez-Ortega et al.)
Model: DeepFakesON-Phys_CelebDF_V2.h5 → converted to ONNX, run with onnxruntime CPU

Evaluation: 9-identity test set (TEST_IDS) from Paper 1's locked split
  Real: <data-root>/videos/real/
  Fake: <data-root>/videos/fake/{method}/

Preprocessing follows vid_to_deepframes_rawframes.py exactly:
  DeepFrames (input_1): temporal difference ratio, per-pixel normalized, uint8 round-trip
  RawFrames  (input_2): raw face, per-pixel normalized, uint8 round-trip
Both channels-first (3, 36, 36), float32
Video-level score: mean of per-frame predictions
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import cv2
import json
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
import onnxruntime as ort

# ── Paths ────────────────────────────────────────────────────────────────────
import argparse as _argparse
_ap = _argparse.ArgumentParser(description='DeepFakesON-Phys (ONNX) baseline on the 9-identity test split')
_ap.add_argument('--data-root', default='data',
                 help='data folder laid out as described in the README (default: ./data)')
_ap.add_argument('--out-dir', default=None,
                 help='folder for result JSONs (default: <data-root>/results)')
_ap.add_argument('--dfp-dir', default='DeepFakesON-Phys',
                 help='folder with model.onnx and haarcascade_frontalface_default.xml')
_args = _ap.parse_args()
ONNX_MODEL = str(Path(_args.dfp_dir) / 'model.onnx')
CASCADE     = str(Path(_args.dfp_dir) / 'haarcascade_frontalface_default.xml')
REAL_DIR    = str(Path(_args.data_root) / 'videos/real')
FAKE_DIR    = str(Path(_args.data_root) / 'videos/fake')
OUT_JSON    = str(Path(_args.out_dir or Path(_args.data_root) / 'results') / 'deepfakeson_phys_baseline.json')

# ── Locked test identities (Paper 1 TEST split) ───────────────────────────
TEST_IDS     = {'id0','id4','id6','id11','id13','id16','id23','id27','id54'}
FAKE_METHODS = ['AniTalker','EchoMimic','EDTalk','FLOAT','IP_LAP','Real3DPortrait','SadTalker']
L = 36   # face resize dimension
BATCH = 256

# ── Load model ───────────────────────────────────────────────────────────────
print('Loading ONNX model...', flush=True)
sess = ort.InferenceSession(ONNX_MODEL, providers=['CPUExecutionProvider'])
inp_names = [i.name for i in sess.get_inputs()]
print(f'  Inputs: {inp_names}', flush=True)

face_cascade = cv2.CascadeClassifier(CASCADE)


def extract_face_frames(video_path):
    """Return (C_R, C_G, C_B) arrays (L, L, n_frames) or None on failure.

    Face detection is run on the first frame only; the detected box is reused
    for all subsequent frames (talking-face videos have stable head position).
    Falls back to center crop if Haar cascade finds no face on the first frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames < 2:
        cap.release()
        return None

    C_R = np.zeros((L, L, n_frames), dtype=np.float32)
    C_G = np.zeros((L, L, n_frames), dtype=np.float32)
    C_B = np.zeros((L, L, n_frames), dtype=np.float32)
    face_box = None   # (x, y, w, h) — set once on first detection
    ka = 0
    while cap.isOpened() and ka < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Detect face only on first frame
        if face_box is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                face_box = tuple(faces[0])
        if face_box is not None:
            x, y, w, h = face_box
            crop = frame[y:y+h, x:x+w]
        else:
            fh, fw = frame.shape[:2]
            s = min(fh, fw)
            crop = frame[(fh-s)//2:(fh-s)//2+s, (fw-s)//2:(fw-s)//2+s]
        face = cv2.resize(crop, (L, L), interpolation=cv2.INTER_AREA)
        C_R[:, :, ka] = face[:, :, 0]
        C_G[:, :, ka] = face[:, :, 1]
        C_B[:, :, ka] = face[:, :, 2]
        ka += 1
    cap.release()
    if ka < 2:
        return None
    return C_R[:, :, :ka], C_G[:, :, :ka], C_B[:, :, :ka]


def build_input_arrays(C_R, C_G, C_B):
    """
    Replicate vid_to_deepframes_rawframes.py preprocessing exactly.
    Returns (deep_frames, raw_frames) each (n-1, 3, L, L) float32.
    """
    n = C_R.shape[2]

    # Temporal difference ratio (DeepFrames)
    D_R = np.zeros((L, L, n), dtype=np.float32)
    D_G = np.zeros((L, L, n), dtype=np.float32)
    D_B = np.zeros((L, L, n), dtype=np.float32)
    for k in range(1, n):
        den_R = C_R[:,:,k] + C_R[:,:,k-1]; den_R[den_R == 0] = 1e-6
        den_G = C_G[:,:,k] + C_G[:,:,k-1]; den_G[den_G == 0] = 1e-6
        den_B = C_B[:,:,k] + C_B[:,:,k-1]; den_B[den_B == 0] = 1e-6
        D_R[:,:,k-1] = (C_R[:,:,k] - C_R[:,:,k-1]) / den_R
        D_G[:,:,k-1] = (C_G[:,:,k] - C_G[:,:,k-1]) / den_G
        D_B[:,:,k-1] = (C_B[:,:,k] - C_B[:,:,k-1]) / den_B

    # Per-pixel temporal statistics
    m_R, m_G, m_B = D_R.mean(2), D_G.mean(2), D_B.mean(2)
    s_R, s_G, s_B = D_R.std(2),  D_G.std(2),  D_B.std(2)
    mC_R, mC_G, mC_B = C_R.mean(2), C_G.mean(2), C_B.mean(2)
    sC_R, sC_G, sC_B = C_R.std(2),  C_G.std(2),  C_B.std(2)

    n_out = n - 1
    deep = np.zeros((n_out, 3, L, L), dtype=np.float32)
    raw  = np.zeros((n_out, 3, L, L), dtype=np.float32)

    for k in range(n_out):
        # DeepFrame k: normalized temporal difference, uint8 round-trip
        img_d = np.stack([
            (D_R[:,:,k] - m_R) / (s_R + 0.1),
            (D_G[:,:,k] - m_G) / (s_G + 0.1),
            (D_B[:,:,k] - m_B) / (s_B + 0.1),
        ], axis=-1).astype(np.uint8)
        deep[k] = img_d.transpose(2, 0, 1).astype(np.float32)

        # RawFrame k+1: normalized appearance, uint8 round-trip
        img_r = np.stack([
            (C_R[:,:,k+1] - mC_R) / (sC_R + 0.1),
            (C_G[:,:,k+1] - mC_G) / (sC_G + 0.1),
            (C_B[:,:,k+1] - mC_B) / (sC_B + 0.1),
        ], axis=-1).astype(np.uint8)
        raw[k] = img_r.transpose(2, 0, 1).astype(np.float32)

    return deep, raw


def predict_video(video_path):
    """Returns mean fake score [0,1] for a video, or None on failure."""
    result = extract_face_frames(video_path)
    if result is None:
        return None
    C_R, C_G, C_B = result
    deep, raw = build_input_arrays(C_R, C_G, C_B)
    if len(deep) == 0:
        return None

    preds = []
    for i in range(0, len(deep), BATCH):
        df = deep[i:i+BATCH]
        rf = raw[i:i+BATCH]
        out = sess.run(None, {inp_names[0]: df, inp_names[1]: rf})
        preds.extend(out[0].flatten().tolist())
    return float(np.mean(preds))


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────
t0 = time.time()
print(f'\nEvaluating DeepFakesON-Phys on 9-identity test set', flush=True)
print(f'Test IDs: {sorted(TEST_IDS)}\n', flush=True)

all_scores = []
all_labels = []
real_scores_list = []   # reused for per-method AUC
per_video = []

# ── Real videos ──────────────────────────────────────────────────────────────
real_videos = sorted(
    v for v in Path(REAL_DIR).glob('*.mp4')
    if v.stem.split('_')[0] in TEST_IDS
)
print(f'Real videos: {len(real_videos)}', flush=True)

for i, vid in enumerate(real_videos):
    score = predict_video(vid)
    if score is None:
        print(f'  SKIP (no face): {vid.name}', flush=True)
        continue
    all_scores.append(score)
    all_labels.append(0)
    real_scores_list.append(score)
    per_video.append({'video': vid.name, 'label': 'real', 'score': score})
    if (i+1) % 20 == 0 or i == len(real_videos) - 1:
        print(f'  Real {i+1}/{len(real_videos)} | elapsed {(time.time()-t0)/60:.1f}min', flush=True)

n_real = len(real_scores_list)
print(f'  Real done: {n_real}/{len(real_videos)} videos\n', flush=True)

# ── Fake videos per method ────────────────────────────────────────────────────
method_results = {}
for method in FAKE_METHODS:
    method_dir = Path(FAKE_DIR) / method
    if not method_dir.exists():
        print(f'  MISSING: {method}', flush=True)
        continue

    fake_videos = sorted(
        v for v in method_dir.glob('*.mp4')
        if v.stem.split('_')[0] in TEST_IDS
    )
    print(f'{method}: {len(fake_videos)} test videos', flush=True)

    method_scores = []
    for i, vid in enumerate(fake_videos):
        score = predict_video(vid)
        if score is None:
            continue
        all_scores.append(score)
        all_labels.append(1)
        method_scores.append(score)
        per_video.append({'video': vid.name, 'label': 'fake', 'method': method, 'score': score})
        if (i+1) % 100 == 0 or i == len(fake_videos) - 1:
            print(f'  {method} {i+1}/{len(fake_videos)} | elapsed {(time.time()-t0)/60:.1f}min', flush=True)

    if method_scores and n_real > 0:
        combined_labels = [0]*n_real + [1]*len(method_scores)
        combined_scores = real_scores_list + method_scores
        try:
            auc = roc_auc_score(combined_labels, combined_scores)
        except Exception:
            auc = float('nan')
        method_results[method] = {
            'n_videos': len(method_scores),
            'mean_score': float(np.mean(method_scores)),
            'auc_vs_real': round(auc, 4),
        }
        print(f'  → {method} AUC vs real: {auc:.4f}\n', flush=True)

# ── Overall AUC ───────────────────────────────────────────────────────────────
print('='*60, flush=True)
if len(set(all_labels)) == 2:
    overall_auc = roc_auc_score(all_labels, all_scores)
    print(f'Overall AUC (all 7 methods vs real): {overall_auc:.4f}', flush=True)
else:
    overall_auc = float('nan')

elapsed = (time.time() - t0) / 60
print(f'Runtime: {elapsed:.1f} min', flush=True)
print(f'n_real={n_real}, n_fake_total={sum(all_labels)}', flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    'experiment':   'deepfakeson_phys_baseline',
    'model':        'DeepFakesON-Phys_CelebDF_V2 (arXiv:2010.00400)',
    'eval_set':     '9-identity test split (TEST_IDS)',
    'test_ids':     sorted(TEST_IDS),
    'n_real_videos': n_real,
    'n_fake_videos': sum(all_labels),
    'overall_auc':  round(overall_auc, 4),
    'per_method':   method_results,
    'runtime_min':  round(elapsed, 1),
}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, 'w') as f:
    json.dump(result, f, indent=2)
print(f'\nSaved → {OUT_JSON}', flush=True)
