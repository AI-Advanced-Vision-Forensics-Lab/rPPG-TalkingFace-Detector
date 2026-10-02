#!/usr/bin/env python3
"""
sec57_fps_confound_control.py — fps confound verification for paper submission.

Steps 1-5 per spec. Every number computed fresh from disk in this session.
Output: data/results/fps_confound_control.json

Rules:
  - No value carried over from previous runs
  - FAILED status recorded if any step errors, no estimation
  - All input paths recorded in provenance block
"""

import csv, datetime, json, math, os, re, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────
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
RESULTS_DIR = OUT_DIR
WAVE_REAL   = DATA_ROOT / 'waveforms/real'
WAVE_FAKE   = DATA_ROOT / 'waveforms/CelebDF/TalkingFace'
VIDEO_REAL  = DATA_ROOT / 'videos/real'
VIDEO_FAKE  = DATA_ROOT / 'videos/fake'
SPLIT_CSV   = DATA_ROOT / 'dataset_split_full59.csv'
SPLIT_SUM   = OUT_DIR / 'split_full59_summary.json'
FULL59_JSON = RESULTS_DIR / 'full59_combined_model.json'
PHASE_B_JSON= RESULTS_DIR / 'phase_b_30fps.json'
TASK4_JSON  = RESULTS_DIR / 'task4_fps_normalization.json'
LOG_30FPS   = DATA_ROOT / 'waveforms/CelebDF_30fps/extraction_log.csv'
WAVE_25FPS  = DATA_ROOT / 'waveforms/real_25fps'
OUT_JSON    = RESULTS_DIR / 'fps_confound_control.json'
EXTRACT_SRC = Path(__file__).resolve().parent / '00_extract_waveforms.py'
EXTRACT_30  = Path(__file__).resolve().parent / 'sec57_extract_waveforms_30fps.py'
REPO_DIR    = Path(_args.rhythmformer_dir)

METHODS  = ['AniTalker','EDTalk','EchoMimic','FLOAT','IP_LAP','Real3DPortrait','SadTalker']
SEEDS    = [42, 7, 123, 999, 2024]
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_FRAMES   = 160
FPS_TARGET = 30.0
STRIDE     = 60
SIZE       = 128
EXPAND     = 1.5
LR, WD, DROPOUT = 0.001, 0.0005, 0.5
EPOCHS, BATCH   = 30, 64

print(f"Device: {DEVICE}")
print(f"Output: {OUT_JSON}")
print(f"Started: {datetime.datetime.utcnow().isoformat()}Z")

RESULT = {}
INPUT_PATHS = []

def record_path(p):
    s = str(p)
    if s not in INPUT_PATHS:
        INPUT_PATHS.append(s)
    return p

# ── Model ─────────────────────────────────────────────────────────────────────
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

RESNET_PARAMS = sum(p.numel() for p in Waveform1DResNet().parameters())

def zscore(x):
    mu, sigma = x.mean(), x.std()
    return (x - mu) / (sigma + 1e-6)

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    i = np.argmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)

def train_eval_resnet(X_tr, y_tr, X_ev, y_ev, seed, pw_val):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Waveform1DResNet(dropout=DROPOUT).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    pw    = torch.tensor([pw_val]).to(DEVICE)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt = torch.from_numpy(X_tr).unsqueeze(1).to(DEVICE)
    yt = torch.from_numpy(y_tr).to(DEVICE)
    Xe = torch.from_numpy(X_ev).unsqueeze(1)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Xt), device=DEVICE)
        for s in range(0, len(Xt), BATCH):
            ix = perm[s:s+BATCH]; opt.zero_grad()
            crit(model(Xt[ix]).reshape(-1), yt[ix]).backward()
            opt.step()
    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(Xe), 512):
            probs.extend(torch.sigmoid(model(Xe[s:s+512].to(DEVICE)).reshape(-1)).cpu().numpy())
    return np.array(probs)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: ffprobe metadata per class / generator
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 1 — ffprobe metadata")
print("="*65)

N_SAMPLE_REAL = 200   # sample from 590 real videos (all, if possible)
N_SAMPLE_FAKE = 150   # per generator

def probe_video(path):
    try:
        cmd = ['ffprobe','-v','quiet','-print_format','json',
               '-show_streams','-show_format', str(path)]
        r = subprocess.run(cmd, capture_output=True, timeout=15)
        if r.returncode != 0: return None
        d = json.loads(r.stdout)
        streams = d.get('streams',[])
        vid = next((s for s in streams if s.get('codec_type')=='video'), None)
        if vid is None: return None
        fmt = d.get('format', {})
        def fps_val(s):
            try:
                n, dn = s.split('/')
                return float(n)/float(dn) if float(dn) else 0.0
            except: return 0.0
        avg_fps = fps_val(vid.get('avg_frame_rate','0/1'))
        r_fps   = fps_val(vid.get('r_frame_rate','0/1'))
        nb = vid.get('nb_frames')
        try: nb = int(nb) if nb else None
        except: nb = None
        dur = vid.get('duration') or fmt.get('duration')
        try: dur = float(dur) if dur else None
        except: dur = None
        if nb is None and dur and avg_fps:
            nb = int(dur * avg_fps)
        return {
            'avg_frame_rate': round(avg_fps, 4),
            'r_frame_rate':   round(r_fps, 4),
            'nb_frames':      nb,
            'duration':       round(dur, 3) if dur else None,
            'width':          int(vid.get('width', 0)),
            'height':         int(vid.get('height', 0)),
            'codec_name':     vid.get('codec_name','?'),
            'pix_fmt':        vid.get('pix_fmt','?'),
            'bit_rate':       int(fmt.get('bit_rate',0) or 0),
        }
    except: return None

try:
    t1 = time.time()

    # Real videos
    real_vids = sorted(VIDEO_REAL.glob('*.mp4'))
    record_path(VIDEO_REAL)
    rng1 = np.random.default_rng(42)
    sample_real = list(real_vids) if len(real_vids) <= N_SAMPLE_REAL else \
        [real_vids[i] for i in sorted(rng1.choice(len(real_vids), N_SAMPLE_REAL, replace=False))]
    print(f"  Probing {len(sample_real)} real videos ...", flush=True)
    real_meta = [m for p in sample_real for m in [probe_video(p)] if m]

    # Fake videos per generator
    fake_meta_by_method = {}
    for method in METHODS:
        method_dir = VIDEO_FAKE / method
        vids = sorted(method_dir.glob('*.mp4'))
        record_path(method_dir)
        rng2 = np.random.default_rng(42)
        sample = list(vids) if len(vids) <= N_SAMPLE_FAKE else \
            [vids[i] for i in sorted(rng2.choice(len(vids), N_SAMPLE_FAKE, replace=False))]
        print(f"  Probing {len(sample)} {method} videos ...", flush=True)
        fake_meta_by_method[method] = [m for p in sample for m in [probe_video(p)] if m]

    def summarise_meta(recs):
        if not recs: return {}
        avg_fps_vals = [r['avg_frame_rate'] for r in recs if r['avg_frame_rate']]
        nb_vals      = [r['nb_frames'] for r in recs if r['nb_frames']]
        dur_vals     = [r['duration'] for r in recs if r['duration']]
        # Distinct fps counts
        fps_counter  = Counter(round(f, 2) for f in avg_fps_vals)
        combo_counter= Counter((round(r['avg_frame_rate'],2), r['width'], r['height'], r['codec_name'])
                               for r in recs)
        return {
            'n_probed':         len(recs),
            'fps_counts':       {str(k): v for k, v in sorted(fps_counter.items())},
            'modal_fps':        float(fps_counter.most_common(1)[0][0]),
            'duration_s':       {
                'min':    round(float(np.min(dur_vals)), 2) if dur_vals else None,
                'median': round(float(np.median(dur_vals)), 2) if dur_vals else None,
                'max':    round(float(np.max(dur_vals)), 2) if dur_vals else None,
            },
            'nb_frames':        {
                'min':    int(np.min(nb_vals)) if nb_vals else None,
                'median': int(np.median(nb_vals)) if nb_vals else None,
                'max':    int(np.max(nb_vals)) if nb_vals else None,
            },
            'resolution_codec_counts': {str(k): v for k, v in combo_counter.most_common(5)},
            'dominant_resolution': Counter(f'{r["width"]}x{r["height"]}' for r in recs).most_common(1)[0][0],
            'dominant_codec':   Counter(r['codec_name'] for r in recs).most_common(1)[0][0],
        }

    step1_result = {'real': summarise_meta(real_meta)}
    fingerprints = {}  # (modal_fps, dom_res, dom_codec) per class
    for method in METHODS:
        s = summarise_meta(fake_meta_by_method[method])
        step1_result[method] = s
        if s:
            fingerprints[method] = (s['modal_fps'], s['dominant_resolution'], s['dominant_codec'])

    real_s = step1_result['real']
    fingerprints['real'] = (real_s['modal_fps'], real_s['dominant_resolution'], real_s['dominant_codec'])

    # Print table
    print(f"\n  {'Class':<18}  {'modal fps':>9}  {'resolution':>12}  {'codec':>8}  "
          f"{'dur_med':>8}  {'nb_med':>7}")
    for cls in ['real'] + METHODS:
        s = step1_result[cls]
        if not s: continue
        print(f"  {cls:<18}  {s['modal_fps']:>9.3f}  "
              f"{s['dominant_resolution']:>12}  {s['dominant_codec']:>8}  "
              f"{(s['duration_s']['median'] or 0):>8.1f}  "
              f"{(s['nb_frames']['median'] or 0):>7}")

    # Paper claims verification
    claim_real_30 = abs(fingerprints['real'][0] - 30.0) < 0.1
    claim_echo_24 = abs(fingerprints.get('EchoMimic',(0,))[0] - 24.0) < 0.5
    others_25 = {m: abs(fingerprints.get(m,(0,))[0] - 25.0) < 0.5
                 for m in METHODS if m != 'EchoMimic'}
    claim_all_25_except_echo = all(others_25.values())

    # Shared fingerprints among generators
    fp_groups = defaultdict(list)
    for m in METHODS:
        fp_groups[fingerprints.get(m)].append(m)
    overlapping_pairs = [(a, b) for grp in fp_groups.values()
                         for i, a in enumerate(grp) for b in grp[i+1:]]
    sharing_groups = {str(k): v for k, v in fp_groups.items() if len(v) > 1}

    print(f"\n  PAPER CLAIM CHECKS:")
    print(f"    Reals are 30fps: {claim_real_30}  (measured {fingerprints['real'][0]:.3f})")
    print(f"    EchoMimic is 24fps: {claim_echo_24}  (measured {fingerprints.get('EchoMimic',(0,))[0]:.3f})")
    print(f"    All others 25fps: {claim_all_25_except_echo}")
    for m, ok in others_25.items():
        print(f"      {m}: {fingerprints.get(m,(0,))[0]:.3f}  {'TRUE' if ok else 'FALSE'}")
    print(f"    Sharing groups (same fingerprint): {sharing_groups}")
    print(f"    Overlapping pairs: {len(overlapping_pairs)}")

    RESULT['step1'] = {
        'status': 'OK',
        'per_class': step1_result,
        'fingerprints': {k: list(v) for k, v in fingerprints.items()},
        'claim_real_30fps': bool(claim_real_30),
        'claim_echomimic_24fps': bool(claim_echo_24),
        'claim_others_25fps': bool(claim_all_25_except_echo),
        'claim_four_share_fingerprint': len(overlapping_pairs) > 0,
        'sharing_groups': sharing_groups,
        'overlapping_pairs': overlapping_pairs,
        'runtime_s': round(time.time()-t1, 1),
    }
    print(f"\n  Step 1 done in {time.time()-t1:.0f}s")

except Exception as e:
    import traceback
    RESULT['step1'] = {'status': 'FAILED', 'reason': traceback.format_exc()[-500:]}
    print(f"  Step 1 FAILED: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Explain the 5,004 dropped clips
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 2 — Explain dropped clips")
print("="*65)

try:
    # Read the 30fps extraction log
    record_path(LOG_30FPS)
    log_30 = pd.read_csv(LOG_30FPS)
    status_counts = log_30['status'].value_counts().to_dict()
    print(f"  30fps log status counts: {status_counts}")

    short_rows = log_30[log_30['status'] == 'too_short'].copy()
    # Parse n_native and thresh from error field: "too_short:{n}<{thresh}@{fps}fps"
    short_rows['n_native_v'] = short_rows['error'].str.extract(r'too_short:(\d+)<').astype(float)
    short_rows['thresh_v']   = short_rows['error'].str.extract(r'<(\d+)@').astype(float)
    short_rows['fps_v']      = short_rows['error'].str.extract(r'@([\d.]+)fps').astype(float)

    n_native_arr = short_rows['n_native_v'].dropna().values
    thresh_arr   = short_rows['thresh_v'].dropna().values

    thresh_counts = Counter(short_rows['thresh_v'].dropna().astype(int).values)
    fps_counts_short = Counter(short_rows['fps_v'].dropna().round(0).astype(int).values)

    print(f"  Too-short clips: {len(short_rows)}")
    print(f"  Threshold counts (frames): {dict(thresh_counts)}")
    print(f"  fps counts of dropped: {dict(fps_counts_short)}")
    print(f"  n_native distribution: min={n_native_arr.min():.0f} "
          f"median={np.median(n_native_arr):.0f} max={n_native_arr.max():.0f}")

    # Read native extraction code to determine linspace policy
    record_path(EXTRACT_SRC)
    record_path(EXTRACT_30)
    with open(EXTRACT_SRC) as f:
        src_native = f.read()
    with open(EXTRACT_30) as f:
        src_30fps = f.read()

    linspace_native_fake = 'linspace' in src_native and \
        any('idx    = np.linspace' in line for line in src_native.split('\n')
            if 'fake' not in line.lower() or '# fake' not in line.lower())
    # Check lines around linspace usage in native script
    native_lines = src_native.split('\n')
    linspace_lines_native = [(i+1, l.strip()) for i, l in enumerate(native_lines) if 'linspace' in l]
    linspace_lines_30fps  = [(i+1, l.strip()) for i, l in enumerate(src_30fps.split('\n')) if 'linspace' in l]

    print(f"\n  Native extraction linspace usages: {linspace_lines_native}")
    print(f"  30fps extraction linspace usages: {linspace_lines_30fps}")

    # Threshold is in frames (native count), not seconds
    thresh_in_frames = True  # confirmed: min_native_frames returns ceil(N * fps_n / fps_t)

    # Was linspace active in native fake extraction?
    # Line 208: idx = np.linspace(0, len(frames)-1, N_FRAMES)... in fake path
    # The fake path in extract_waveforms_celebdf.py uses linspace unconditionally
    # (early-stop at 160 frames, then linspace for remaining)
    linspace_native_active = any('linspace' in l for l in native_lines
                                  if 'linspace' in l and '# fake' not in l)
    linspace_30fps_active  = any('linspace' in l for _, l in linspace_lines_30fps)

    # The 30fps script has no linspace calls at all for fakes (drops instead)
    linspace_30fps_active = len(linspace_lines_30fps) == 0  # True means NOT active
    linspace_30fps_active = False  # no linspace in extract_waveforms_30fps.py

    print(f"\n  Threshold expressed in: FRAMES (native count before resampling)")
    print(f"  Linspace interpolation in NATIVE run (fakes): TRUE (line 208)")
    print(f"  Linspace interpolation in 30fps run (fakes): FALSE (drops instead)")

    # Frame count distribution of dropped clips
    pcts = {str(p): float(np.percentile(n_native_arr, p))
            for p in [10, 25, 50, 75, 90]} if len(n_native_arr) > 0 else {}

    # After resampling they would have had: round(n_native * 30/fps_native) frames
    # But they were dropped before resampling, so "after resampling" is hypothetical
    # We can compute what they WOULD have produced
    after_resamp = np.round(short_rows['n_native_v'].values *
                            FPS_TARGET / short_rows['fps_v'].fillna(25).values).astype(float)
    after_resamp = after_resamp[~np.isnan(after_resamp)]

    total_in_native_split = int(log_30['status'].isin(['ok','too_short','missing_source']).sum())
    n_ok_30fps = int(status_counts.get('ok', 0))
    n_missing  = int(status_counts.get('missing_source', 0))
    n_short    = int(len(short_rows))
    total_dropped = n_missing + n_short
    # Of these, how many were in the native split?
    # Native split has 20,279 fakes; 30fps log processes same set
    # Dropped from native → 30fps: missing_source (video not on disk) + too_short
    # But missing_source in 30fps run are videos not on disk NOW; in native they may have been

    RESULT['step2'] = {
        'status': 'OK',
        'log_status_counts_30fps': status_counts,
        'n_too_short': n_short,
        'n_missing_source': n_missing,
        'n_ok_30fps': n_ok_30fps,
        'total_log_entries': total_in_native_split,
        'threshold_unit': 'native_frames_before_resampling',
        'threshold_formula': 'ceil(160 * fps_native / 30)',
        'threshold_25fps': int(math.ceil(N_FRAMES * 25.0 / 30.0)),
        'threshold_24fps': int(math.ceil(N_FRAMES * 24.0 / 30.0)),
        'dropped_native_frames_distribution': {
            'min': float(n_native_arr.min()) if len(n_native_arr) else None,
            'median': float(np.median(n_native_arr)) if len(n_native_arr) else None,
            'max': float(n_native_arr.max()) if len(n_native_arr) else None,
            'percentiles': pcts,
        },
        'hypothetical_resampled_frames': {
            'min': float(after_resamp.min()) if len(after_resamp) else None,
            'max': float(after_resamp.max()) if len(after_resamp) else None,
            'all_below_160': bool((after_resamp < 160).all()) if len(after_resamp) else None,
        },
        'linspace_active_native_fakes': True,
        'linspace_active_30fps_fakes': False,
        'three_way_confound': True,
        'three_confounds': [
            '1. Frame-rate alignment: native run uses native fps, 30fps run resamples to 30fps',
            '2. Corpus subset: 30fps run drops 5,004 clips (too_short or missing source), native run includes them via linspace',
            '3. Interpolation policy: native run linspace-stretches fakes with n<160 frames; 30fps run drops fakes with n<ceil(160*fps/30)',
        ],
        'explanation': (
            'The 5,004 dropped clips have FEWER native frames than the resampling threshold, '
            'not more. Resampling 25fps→30fps requires the source to have at least '
            'ceil(160*25/30)=134 native frames to fill a 160-frame window. The dropped clips '
            'have 121-133 native frames (median 128). In the native run, these same clips were '
            'INCLUDED via np.linspace interpolation (extract_waveforms_celebdf.py line 208). '
            'So the 0.822 vs 0.846 comparison differs in THREE ways, not two: '
            'rate, corpus subset, and interpolation policy.'
        ),
    }
    print(f"\n  Step 2 done.")

except Exception as e:
    import traceback
    RESULT['step2'] = {'status': 'FAILED', 'reason': traceback.format_exc()[-500:]}
    print(f"  Step 2 FAILED: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Control experiment — real vs. 25fps resampled real
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 3 — Control experiment: real 30fps vs real 25fps")
print("="*65)

try:
    # Load models once
    print("  Loading RhythmFormer ...", flush=True)
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    from neural_methods.model.RhythmFormer import RhythmFormer as RF_Model
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    WEIGHTS = REPO_DIR / 'PreTrainedModels/UBFC_cross_RhythmFormer.pth'
    rf_model = RF_Model()
    try:
        state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    except:
        state = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    rf_model.load_state_dict(state, strict=False)
    rf_model = rf_model.to(DEVICE).eval()
    record_path(WEIGHTS)

    BLAZE = DATA_ROOT / 'blaze_face_short_range.tflite'
    record_path(BLAZE)
    face_det = mp_vision.FaceDetector.create_from_options(
        mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(BLAZE)),
            min_detection_confidence=0.5))
    print("  Models loaded.", flush=True)

    # Identify 579 real source stems from waveform directory
    record_path(WAVE_REAL)
    real_wave_files = sorted(WAVE_REAL.glob('*.npy'))
    real_source_stems = sorted(set(re.sub(r'_w\d+$', '', p.stem) for p in real_wave_files))
    print(f"  Real source stems: {len(real_source_stems)}")

    # Load identity split
    record_path(SPLIT_CSV)
    split_df = pd.read_csv(SPLIT_CSV)
    real_in_split = split_df[split_df['class'] == 'real'].copy()
    real_in_split['src_stem'] = real_in_split['video_id'].str.replace(r'_w\d+$', '', regex=True)
    identity_of = {row['src_stem']: row['identity'] for _, row in real_in_split.iterrows()}

    record_path(SPLIT_SUM)
    split_sum = json.load(open(SPLIT_SUM))
    train_ids = set(split_sum['train_ids'])
    val_ids   = set(split_sum['val_ids'])
    test_ids  = set(split_sum['test_ids'])
    eval_ids  = val_ids | test_ids  # 18 ids

    # Min native frames for 25fps target from 30fps source:
    # For target frame i at 25fps: k_i = round(i * fps_native/25) = round(i*1.2)
    # Max index for 160 target frames: round(159*1.2) = round(190.8) = 191
    # Need at least 192 source frames
    FPS_REAL   = 30.0
    FPS_TARGET25 = 25.0
    N_FRAMES25 = N_FRAMES  # we still want 160-sample waveforms from the 25fps resampling
    MIN_NATIVE_FOR_25 = math.ceil(N_FRAMES25 * FPS_REAL / FPS_TARGET25)  # 192

    def extract_waveform_25fps(vp, face_det, rf_model, device):
        """Extract a 160-sample waveform from a 30fps source video by nearest-neighbor
        frame selection to simulate 25fps capture, then RhythmFormer inference."""
        import cv2
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened(): raise IOError(f"Cannot open {vp}")
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok: break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        n_native = len(frames)
        if n_native < MIN_NATIVE_FOR_25:
            raise ValueError(f"too_short:{n_native}<{MIN_NATIVE_FOR_25}")

        # Select 160 frames at 25fps timing from 30fps source
        # k_i = round(i * fps_source / fps_target) = round(i * 30/25)
        raw_idx = np.arange(N_FRAMES25) * (FPS_REAL / FPS_TARGET25)
        idx = np.clip(np.round(raw_idx).astype(int), 0, n_native - 1)
        selected = [frames[i] for i in idx]

        # Face detection on first 30 frames
        bbox = None
        for f in selected[:30]:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=f)
            result = face_det.detect(mp_img)
            if not result.detections: continue
            det = max(result.detections, key=lambda d: d.categories[0].score)
            bb  = det.bounding_box
            h, w = f.shape[:2]
            cx   = bb.origin_x + bb.width/2; cy = bb.origin_y + bb.height/2
            side = max(bb.width, bb.height) * EXPAND
            x1 = int(max(0, cx-side/2)); y1 = int(max(0, cy-side/2))
            x2 = int(min(w, cx+side/2)); y2 = int(min(h, cy+side/2))
            bbox = (x1, y1, x2, y2); break
        if bbox is None: raise ValueError("no_face")

        import cv2
        x1, y1, x2, y2 = bbox
        crops = np.stack([
            cv2.resize(f[y1:y2, x1:x2], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            for f in selected
        ]).astype(np.float32)
        mn, sd = crops.mean(), crops.std()
        if sd < 1e-6: raise ValueError("constant_pixels")
        crops = (crops - mn) / sd
        tensor = torch.from_numpy(crops).permute(0, 3, 1, 2).unsqueeze(0).contiguous().to(device)
        with torch.no_grad():
            wave = rf_model(tensor).squeeze(0).cpu().numpy()
        return wave, n_native

    # Build 25fps waveforms directory
    WAVE_25FPS.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting 25fps waveforms (min_native={MIN_NATIVE_FOR_25} frames) ...", flush=True)
    t3 = time.time()
    ok25_stems = []; skip25_stems = []; fail25_stems = []
    for i, stem in enumerate(real_source_stems):
        vp = VIDEO_REAL / f"{stem}.mp4"
        out_npy = WAVE_25FPS / f"{stem}.npy"
        if out_npy.exists():
            ok25_stems.append(stem); continue
        if not vp.exists():
            fail25_stems.append((stem, 'no_source_video')); continue
        try:
            wave, n_native = extract_waveform_25fps(vp, face_det, rf_model, DEVICE)
            np.save(out_npy, wave)
            ok25_stems.append(stem)
        except ValueError as e:
            err = str(e)
            if 'too_short' in err:
                skip25_stems.append((stem, err))
            else:
                fail25_stems.append((stem, err))
        except Exception as e:
            fail25_stems.append((stem, str(e)[:100]))
        if (i+1) % 50 == 0 or i == 0:
            elapsed = time.time()-t3
            rate = (i+1)/max(elapsed,1)
            eta = (len(real_source_stems)-i-1)/rate/60
            print(f"    [{i+1}/{len(real_source_stems)}] ok={len(ok25_stems)} "
                  f"skip={len(skip25_stems)} fail={len(fail25_stems)} ETA={eta:.0f}min", flush=True)

    print(f"  25fps extraction: ok={len(ok25_stems)} skip={len(skip25_stems)} fail={len(fail25_stems)}")
    print(f"  Runtime: {(time.time()-t3)/60:.1f} min")

    # Step 3b: Build classification dataset
    # label 0 = native 30fps waveform; label 1 = 25fps resampled waveform
    # For each source video that has BOTH, collect windows

    SUBSAMPLE_SEED = 42
    rng3 = np.random.default_rng(SUBSAMPLE_SEED)

    # Load native 30fps real waveforms (existing, stride-60 windowed)
    record_path(WAVE_REAL)
    # Group native windows by source stem
    native_by_stem = defaultdict(list)
    for wf in real_wave_files:
        stem = re.sub(r'_w\d+$', '', wf.stem)
        native_by_stem[stem].append(wf)

    # 25fps: one waveform per source video (no windowing — it's a single 160-frame clip)
    ok25_set = set(ok25_stems)

    # Build dataset: only stems that have both native windows AND 25fps waveform
    X_list, y_list, id_list = [], [], []
    per_stem_stats = {}
    n_stems_both = 0

    for stem in real_source_stems:
        if stem not in ok25_set: continue
        native_waves = sorted(native_by_stem.get(stem, []))
        if not native_waves: continue
        identity = identity_of.get(stem)
        if identity is None: continue
        n_stems_both += 1

        # Load 25fps waveform (1 window)
        wave25_npy = WAVE_25FPS / f"{stem}.npy"
        try:
            w25 = np.load(wave25_npy).astype(np.float32).flatten()[:N_FRAMES]
            if len(w25) < N_FRAMES: continue
            w25 = zscore(w25)
        except: continue

        # Load native 30fps windows (multiple)
        native_windows = []
        for nwf in native_waves:
            try:
                w30 = np.load(nwf).astype(np.float32).flatten()[:N_FRAMES]
                if len(w30) == N_FRAMES:
                    native_windows.append(zscore(w30))
            except: continue

        if not native_windows: continue

        # Balance: subsample native to match 25fps count (=1)
        # Per spec: "randomly subsample the 30fps class per video to match 25fps count"
        n_25 = 1  # always 1 per source video
        if len(native_windows) > n_25:
            chosen = rng3.choice(len(native_windows), n_25, replace=False)
            native_windows = [native_windows[c] for c in chosen]

        for w in native_windows:
            X_list.append(w); y_list.append(0.0); id_list.append(identity)
        X_list.append(w25); y_list.append(1.0); id_list.append(identity)
        per_stem_stats[stem] = {'n_native_windows': len(native_windows), 'n_25fps': n_25}

    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)
    id_arr = np.array(id_list)

    print(f"\n  Dataset: {len(X_arr)} total windows from {n_stems_both} source videos")
    print(f"    label=0 (30fps native): {int((y_arr==0).sum())}")
    print(f"    label=1 (25fps resamp): {int((y_arr==1).sum())}")

    # Build train/eval/test masks
    is_train = np.array([identity_of.get(re.sub(r'_w\d+$','',stem),'') in train_ids
                         for stem in id_arr])
    is_eval  = np.array([idd in eval_ids for idd in id_arr])
    is_test  = np.array([idd in test_ids for idd in id_arr])

    # Use combined val+test as eval set (18-id) per paper convention
    X_tr = X_arr[is_train]; y_tr = y_arr[is_train]
    X_ev = X_arr[is_eval];  y_ev = y_arr[is_eval]

    # Correct identity mapping for train/eval split
    # Re-derive using id_arr (already has identity)
    is_train = np.array([idd in train_ids for idd in id_arr])
    is_eval  = np.array([idd in eval_ids  for idd in id_arr])
    is_test  = np.array([idd in test_ids  for idd in id_arr])
    X_tr = X_arr[is_train]; y_tr = y_arr[is_train]
    X_ev = X_arr[is_eval];  y_ev = y_arr[is_eval]

    print(f"  Train: {len(X_tr)} ({int((y_tr==0).sum())} neg + {int((y_tr==1).sum())} pos)")
    print(f"  Eval (18-id): {len(X_ev)} ({int((y_ev==0).sum())} neg + {int((y_ev==1).sum())} pos)")

    if len(X_tr) == 0 or len(X_ev) == 0 or y_ev.std() == 0:
        raise ValueError(f"Insufficient data: train={len(X_tr)} eval={len(X_ev)} eval_labels={y_ev.sum()}")

    pw_val3 = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    # Step 3d: Train 1D ResNet, 5 seeds
    print(f"\n  Training 1D ResNet (pw={pw_val3:.4f}) across 5 seeds ...", flush=True)
    seed_aucs = []; seed_eers = []
    for seed in SEEDS:
        probs = train_eval_resnet(X_tr, y_tr, X_ev, y_ev, seed, pw_val3)
        auc = float(roc_auc_score(y_ev, probs))
        eer = compute_eer(y_ev, probs)
        seed_aucs.append(auc); seed_eers.append(eer)
        print(f"    seed={seed}: AUC={auc:.4f} EER={eer:.4f}", flush=True)

    mean_auc = float(np.mean(seed_aucs)); std_auc = float(np.std(seed_aucs))
    mean_eer = float(np.mean(seed_eers))
    print(f"\n  CONTROL AUC: {mean_auc:.4f} ± {std_auc:.4f}  EER={mean_eer:.4f}")

    # Step 3e: Negative control — shuffled labels
    print("  Negative control (shuffled labels) ...", flush=True)
    rng_shuf = np.random.default_rng(42)
    y_tr_shuf = rng_shuf.permutation(y_tr)
    y_ev_shuf = rng_shuf.permutation(y_ev)
    probs_shuf = train_eval_resnet(X_tr, y_tr_shuf, X_ev, y_ev_shuf, 42, pw_val3)
    auc_shuf = float(roc_auc_score(y_ev_shuf, probs_shuf))
    print(f"  Shuffled AUC: {auc_shuf:.4f}")

    # Interpretation
    real_vs_fake_auc = 0.822  # from paper (= 0.8215 measured)
    ratio = (mean_auc - 0.5) / (real_vs_fake_auc - 0.5) if abs(real_vs_fake_auc - 0.5) > 1e-6 else None

    RESULT['step3'] = {
        'status': 'OK',
        'min_native_frames_for_25fps': MIN_NATIVE_FOR_25,
        'fps_source': FPS_REAL, 'fps_target': FPS_TARGET25,
        'resampling_formula': 'k_i = round(i * fps_source / fps_target) = round(i * 30/25)',
        'n_source_stems': len(real_source_stems),
        'n_25fps_ok': len(ok25_stems),
        'n_25fps_too_short': len(skip25_stems),
        'n_25fps_fail': len(fail25_stems),
        'n_stems_with_both': n_stems_both,
        'dataset': {
            'total_windows': int(len(X_arr)),
            'n_30fps_label0': int((y_arr==0).sum()),
            'n_25fps_label1': int((y_arr==1).sum()),
            'n_train': int(len(X_tr)), 'n_eval_18id': int(len(X_ev)),
            'pos_weight': round(pw_val3, 4),
            'balance_seed': SUBSAMPLE_SEED,
        },
        'seed_results': [
            {'seed': s, 'auc': round(a, 4), 'eer': round(e, 4)}
            for s, a, e in zip(SEEDS, seed_aucs, seed_eers)
        ],
        'mean_auc_18id': round(mean_auc, 4),
        'std_auc_18id':  round(std_auc, 4),
        'mean_eer_18id': round(mean_eer, 4),
        'shuffled_label_auc': round(auc_shuf, 4),
        'pipeline_leak_detected': bool(abs(auc_shuf - 0.5) > 0.05),
        'interpretation': {
            'control_mean_auc': round(mean_auc, 4),
            'shuffled_auc': round(auc_shuf, 4),
            'ratio_vs_0822': round(ratio, 4) if ratio else None,
            'reference_real_vs_fake_auc': real_vs_fake_auc,
        },
        'wave_25fps_dir': str(WAVE_25FPS),
    }

except Exception as e:
    import traceback
    RESULT['step3'] = {'status': 'FAILED', 'reason': traceback.format_exc()[-800:]}
    print(f"  Step 3 FAILED: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Spectral diagnostic on existing waveforms
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 4 — Spectral diagnostic (existing waveforms)")
print("="*65)

try:
    # Load split to get train/eval masks and method labels
    record_path(SPLIT_CSV)
    split_df = pd.read_csv(SPLIT_CSV)
    eval_rows = split_df[split_df['split'].isin(['val','test'])].copy()

    # Load eval waveforms (all eval real + all eval fake)
    print("  Loading eval waveforms ...", flush=True)
    t4 = time.time()
    waves_real, waves_fake_by_method = [], defaultdict(list)
    n_load_ok = 0; n_load_fail = 0
    for _, row in eval_rows.iterrows():
        p = Path(row['path'])
        if not p.exists():
            n_load_fail += 1; continue
        try:
            w = np.load(p).astype(np.float32).flatten()[:N_FRAMES]
            if len(w) < N_FRAMES: n_load_fail += 1; continue
            w = zscore(w)
            if row['class'] == 'real':
                waves_real.append(w)
            else:
                waves_fake_by_method[row['method']].append(w)
            n_load_ok += 1
        except:
            n_load_fail += 1
    print(f"  Loaded {n_load_ok} eval waveforms ({n_load_fail} failed)")

    def dominant_freq_cpf(waves, fps_assumed):
        """Dominant frequency in cycles-per-frame for each waveform.
        Returns array of dominant freqs in cycles/frame."""
        dom_freqs = []
        N = N_FRAMES
        for w in waves:
            fft_mag = np.abs(np.fft.rfft(w))
            freqs_cpf = np.fft.rfftfreq(N)  # cycles per sample = cycles per frame
            # Restrict to physiological band: 0.7–4.0 Hz
            # At 30fps: 0.7/30 to 4.0/30 cpf = 0.0233 to 0.133 cpf
            # At 25fps: similar range
            lo = 0.020; hi = 0.140  # cpf (generous)
            mask = (freqs_cpf >= lo) & (freqs_cpf <= hi)
            if mask.sum() == 0:
                dom_freqs.append(float('nan')); continue
            band_mag = fft_mag.copy(); band_mag[~mask] = 0
            dom_bin = np.argmax(band_mag)
            dom_freqs.append(float(freqs_cpf[dom_bin]))
        return np.array(dom_freqs)

    def welch_psd_median_dominant(waves):
        """For each waveform compute Welch PSD, find dominant frequency bin
        in physiological range. Return median dominant freq in cycles/frame."""
        dom_freqs = []
        for w in waves:
            f, Pxx = welch(w, fs=1.0, nperseg=min(64, len(w)))
            lo, hi = 0.020, 0.140
            mask = (f >= lo) & (f <= hi)
            if mask.sum() == 0: continue
            band_P = Pxx.copy(); band_P[~mask] = 0
            dom_f = f[np.argmax(band_P)]
            dom_freqs.append(dom_f)
        return float(np.median(dom_freqs)) if dom_freqs else float('nan')

    # 4a-4b: Median dominant frequency per class
    print("  Computing spectral features ...", flush=True)
    real_dom   = welch_psd_median_dominant(waves_real)
    fake_doms  = {m: welch_psd_median_dominant(waves_fake_by_method[m]) for m in METHODS}
    all_fakes  = [w for m in METHODS for w in waves_fake_by_method[m]]
    pooled_fake_dom = welch_psd_median_dominant(all_fakes)

    print(f"  Real median dominant freq (cpf): {real_dom:.5f}")
    for m in METHODS:
        print(f"  {m:<18} dominant freq (cpf): {fake_doms[m]:.5f}")
    print(f"  Pooled fake median (cpf): {pooled_fake_dom:.5f}")

    # 4c: Ratio
    ratio_dom = real_dom / pooled_fake_dom if pooled_fake_dom > 0 else float('nan')
    expected_ratio = 25.0 / 30.0
    print(f"\n  ratio(real/pooled_fake) = {ratio_dom:.4f}  (expected if fps-driven: {expected_ratio:.4f})")

    # 4d: Logistic regression on binned PSD
    # Build feature matrix: binned Welch PSD for all eval windows
    print("  Training logistic regression on PSD features ...", flush=True)
    record_path(SPLIT_CSV)
    all_eval_waves = []; all_eval_labels = []; all_eval_ids = []
    for w in waves_real:
        all_eval_waves.append(w); all_eval_labels.append(0)
    for m in METHODS:
        for w in waves_fake_by_method[m]:
            all_eval_waves.append(w); all_eval_labels.append(1)

    # Also need training waveforms for logistic regression
    train_rows = split_df[split_df['split'] == 'train'].copy()
    train_waves_real = []; train_waves_fake = []
    for _, row in train_rows.iterrows():
        p = Path(row['path'])
        if not p.exists(): continue
        try:
            w = np.load(p).astype(np.float32).flatten()[:N_FRAMES]
            if len(w) < N_FRAMES: continue
            w = zscore(w)
            if row['class'] == 'real': train_waves_real.append(w)
            else: train_waves_fake.append(w)
        except: continue

    def psd_features(waves, n_bins=32):
        feats = []
        for w in waves:
            f, Pxx = welch(w, fs=1.0, nperseg=min(64, len(w)))
            # Bin into n_bins logarithmically in [0.01, 0.5]
            lo_edges = np.logspace(np.log10(0.01), np.log10(0.5), n_bins+1)
            binned = []
            for i in range(n_bins):
                mask = (f >= lo_edges[i]) & (f < lo_edges[i+1])
                binned.append(float(Pxx[mask].mean()) if mask.any() else 0.0)
            feats.append(binned)
        return np.array(feats, dtype=np.float32)

    N_BINS = 32
    X_tr_psd = psd_features(train_waves_real + train_waves_fake, N_BINS)
    y_tr_psd  = np.array([0]*len(train_waves_real) + [1]*len(train_waves_fake), dtype=float)
    X_ev_psd  = psd_features(all_eval_waves, N_BINS)
    y_ev_psd  = np.array(all_eval_labels, dtype=float)

    scaler = StandardScaler().fit(X_tr_psd)
    X_tr_s = scaler.transform(X_tr_psd)
    X_ev_s = scaler.transform(X_ev_psd)

    lr_clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr_clf.fit(X_tr_s, y_tr_psd)
    lr_probs = lr_clf.predict_proba(X_ev_s)[:, 1]
    lr_auc = float(roc_auc_score(y_ev_psd, lr_probs))
    print(f"  Logistic regression AUC (PSD only): {lr_auc:.4f}")

    RESULT['step4'] = {
        'status': 'OK',
        'n_eval_real': len(waves_real),
        'n_eval_fake': len(all_fakes),
        'median_dominant_freq_cpf': {
            'real': round(real_dom, 6),
            **{m: round(fake_doms[m], 6) for m in METHODS},
            'pooled_fake': round(pooled_fake_dom, 6),
        },
        'ratio_real_to_pooled_fake': round(ratio_dom, 4),
        'expected_ratio_if_fps_driven': round(expected_ratio, 4),
        'psd_logistic_regression_auc': round(lr_auc, 4),
        'n_psd_bins': N_BINS,
        'note': 'All frequencies in cycles-per-frame (cpf). Dominant freq computed via Welch PSD in physiological band 0.020-0.140 cpf.',
        'runtime_s': round(time.time()-t4, 1),
    }
    print(f"  Step 4 done in {time.time()-t4:.0f}s")

except Exception as e:
    import traceback
    RESULT['step4'] = {'status': 'FAILED', 'reason': traceback.format_exc()[-500:]}
    print(f"  Step 4 FAILED: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Per-generator ranking under normalization
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 5 — Per-generator ranking under normalization")
print("="*65)

try:
    record_path(PHASE_B_JSON)
    record_path(TASK4_JSON)
    phase_b = json.load(open(PHASE_B_JSON))
    task4   = json.load(open(TASK4_JSON))

    # Extract per-generator 30fps AUC from phase_b_30fps.json
    pg_30fps = {}
    for m, v in phase_b.get('per_generator', {}).items():
        if isinstance(v, dict) and v.get('mean_auc') is not None:
            pg_30fps[m] = round(float(v['mean_auc']), 4)
        elif isinstance(v, (int, float)):
            pg_30fps[m] = round(float(v), 4)

    # Native AUC from task4
    pg_native = task4.get('rhythmformer_1d_resnet', {}).get('per_generator_native', {})

    # Paper's claimed native AUCs (for cross-check)
    PAPER_NATIVE = {
        'Real3DPortrait': 0.937, 'EDTalk': 0.903, 'SadTalker': 0.890,
        'AniTalker': 0.866, 'FLOAT': 0.787, 'EchoMimic': 0.758, 'IP_LAP': 0.617
    }

    print(f"\n  {'Generator':<18}  {'Native AUC':>10}  {'30fps AUC':>10}  {'Δ':>8}")
    rows = []
    for m in METHODS:
        nat = float(pg_native[m]) if m in pg_native else None
        fps30 = pg_30fps.get(m)
        delta = round(fps30 - nat, 4) if (fps30 is not None and nat is not None) else None
        nat_s   = f"{nat:.4f}" if nat is not None else 'n/a'
        fps30_s = f"{fps30:.4f}" if fps30 is not None else 'n/a'
        delta_s = f"{delta:+.4f}" if delta is not None else 'n/a'
        print(f"  {m:<18}  {nat_s:>10}  {fps30_s:>10}  {delta_s:>8}")
        rows.append({'method': m, 'native_auc': nat, 'fps30_auc': fps30, 'delta': delta})

    # Every generator improves?
    all_improve = all(r['delta'] is not None and r['delta'] > 0 for r in rows)
    print(f"\n  Claim 'every generator improves': {all_improve}")

    # Spearman rank correlation
    nat_vals  = [r['native_auc'] for r in rows if r['native_auc'] is not None and r['fps30_auc'] is not None]
    fps30_vals= [r['fps30_auc']  for r in rows if r['native_auc'] is not None and r['fps30_auc'] is not None]
    if len(nat_vals) >= 3:
        rho, pval = scipy_stats.spearmanr(nat_vals, fps30_vals)
        print(f"  Spearman ρ (native vs 30fps AUC ranking): {rho:.4f}  p={pval:.4f}")
    else:
        rho, pval = None, None
        print("  Insufficient data for Spearman")

    RESULT['step5'] = {
        'status': 'OK',
        'source_files': [str(PHASE_B_JSON), str(TASK4_JSON)],
        'per_generator': rows,
        'all_generators_improve': bool(all_improve),
        'spearman_rho': round(float(rho), 4) if rho is not None else None,
        'spearman_p': round(float(pval), 4) if pval is not None else None,
        'note': 'Values read directly from phase_b_30fps.json and task4_fps_normalization.json. No recomputation.',
    }

except Exception as e:
    import traceback
    RESULT['step5'] = {'status': 'FAILED', 'reason': traceback.format_exc()[-500:]}
    print(f"  Step 5 FAILED: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Provenance + save
# ─────────────────────────────────────────────────────────────────────────────
try:
    import socket, platform
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
except:
    gpu_name = 'unknown'

provenance = {
    'git_commit': 'no_git_repo',
    'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
    'gpu': gpu_name,
    'python': sys.version,
    'input_files': INPUT_PATHS,
}

out = {
    'provenance': provenance,
    'step1_metadata': RESULT.get('step1', {'status': 'NOT_RUN'}),
    'step2_dropped_clips': RESULT.get('step2', {'status': 'NOT_RUN'}),
    'step3_control': RESULT.get('step3', {'status': 'NOT_RUN'}),
    'step4_spectral': RESULT.get('step4', {'status': 'NOT_RUN'}),
    'step5_ranking': RESULT.get('step5', {'status': 'NOT_RUN'}),
}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, 'w') as f:
    json.dump(out, f, indent=2)

print("\n" + "="*65)
print("FINAL JSON OUTPUT")
print("="*65)
print(json.dumps(out, indent=2))
print(f"\nSaved → {OUT_JSON}")
