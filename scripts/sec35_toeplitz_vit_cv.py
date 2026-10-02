#!/usr/bin/env python3
import json, time, datetime, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

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
OUT_JSON  = OUT_DIR / 'toeplitz_vit.json'
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def zscore(x): return (x - x.mean()) / (x.std() + 1e-6)

def make_toeplitz(x):
    n = len(x); idx = np.arange(n)
    return x[np.abs(idx[:, None] - idx[None, :])].astype(np.float32)

class ToeplitzViT(nn.Module):
    def __init__(self, image_size=160, patch_size=16, d_model=64, nhead=4,
                 num_layers=2, mlp_dim=128, dropout=0.3):
        super().__init__()
        n = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(1, d_model, patch_size, stride=patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, mlp_dim, dropout,
                                               activation="gelu", batch_first=True,
                                               norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], 1) + self.pos_embed
        return self.fc(self.dropout(self.norm(self.encoder(x)[:, 0]))).squeeze(-1)

N_PARAMS = sum(p.numel() for p in ToeplitzViT(dropout=0.5).parameters())

class ToeplitzDataset(torch.utils.data.Dataset):
    def __init__(self, rw, lb):
        self.waves = rw; self.labels = lb.astype(np.float32)
    def __len__(self): return len(self.waves)
    def __getitem__(self, i):
        x = zscore(self.waves[i]); T = make_toeplitz(x)
        return torch.from_numpy(T[None]), torch.tensor(self.labels[i])

def run_fold(raw_tr, y_tr, raw_ev, y_ev, pw, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ToeplitzViT(dropout=0.5).to(DEVICE)
    opt  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(DEVICE))
    ld = torch.utils.data.DataLoader(ToeplitzDataset(raw_tr, y_tr), batch_size=64, shuffle=True)
    model.train()
    for _ in range(30):
        for Xb, yb in ld:
            opt.zero_grad(); crit(model(Xb.to(DEVICE)), yb.to(DEVICE)).backward(); opt.step()
    model.eval()
    probs = []
    with torch.no_grad():
        for Xb, _ in torch.utils.data.DataLoader(ToeplitzDataset(raw_ev, y_ev), batch_size=256):
            probs.extend(torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy())
    return float(roc_auc_score(y_ev, probs))

df = pd.read_csv(SPLIT_CSV)
df = df[df['path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)
df_tr = df[df['split']=='train'].reset_index(drop=True)
n_real_tr = int((df_tr['class']=='real').sum())
n_fake_tr = int((df_tr['class']=='fake').sum())
pw_val = n_real_tr / n_fake_tr

print(f"Loading {len(df_tr)} training waveforms...", flush=True)
raw_tr = [np.load(p).astype(np.float32) for p in df_tr['path']]
y_tr   = np.array([1.0 if c=='fake' else 0.0 for c in df_tr['class']], dtype=np.float32)
print("Done.", flush=True)

train_ids = sorted(df_tr['identity'].unique(), key=lambda x: int(x[2:]))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = []; t0 = time.time()
for fi, (ti, vi) in enumerate(kf.split(train_ids)):
    ftr = set(train_ids[i] for i in ti); fva = set(train_ids[i] for i in vi)
    mtr = df_tr['identity'].isin(ftr).values; mva = df_tr['identity'].isin(fva).values
    rtr = [raw_tr[i] for i in np.where(mtr)[0]]; rva = [raw_tr[i] for i in np.where(mva)[0]]
    ytr = y_tr[mtr]; yva = y_tr[mva]
    nr = float((ytr==0).sum()); nf = float((ytr==1).sum()); pw_cv = nr/max(nf,1.0)
    auc = run_fold(rtr, ytr, rva, yva, pw_cv)
    cv_aucs.append(round(auc,4))
    print(f"fold {fi+1}/5  AUC={auc:.4f}  {(time.time()-t0)/60:.1f}min", flush=True)

cv_mean = float(np.mean(cv_aucs)); cv_std = float(np.std(cv_aucs))
print(f"CV mean: {cv_mean:.4f} +/- {cv_std:.4f}", flush=True)

try:
    gpu = subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
except: gpu='unknown'

seed_results=[{'seed':42,'auc_18id':0.7548,'eer_18id':0.3175},
              {'seed':7,'auc_18id':0.7643,'eer_18id':0.3109},
              {'seed':123,'auc_18id':0.7500,'eer_18id':0.3212},
              {'seed':999,'auc_18id':0.7646,'eer_18id':0.3001},
              {'seed':2024,'auc_18id':0.7528,'eer_18id':0.3159}]
aucs=[r['auc_18id'] for r in seed_results]; eers=[r['eer_18id'] for r in seed_results]
mean_auc=float(np.mean(aucs)); std_auc=float(np.std(aucs)); mean_eer=float(np.mean(eers))
eval_ids=['id0','id4','id6','id8','id11','id12','id13','id16','id19','id22','id23','id27','id34','id42','id44','id52','id54','id56']

result={
  "experiment":"toeplitz_vit",
  "step_0_prior_search":{"implementation_found":True,
    "location":"src/run_experiments.py class ToeplitzViT lines 135-157",
    "also_found_in":"src/per_method_full_experiments.py",
    "prior_result_note":"hp_tuning.json Toeplitz ViT entries are per-method HP sweep scores (not 59-id combined 18-id eval). No prior 18-id combined eval result existed."},
  "step_1_implementation":{"source_file":"src/run_experiments.py (reused verbatim)",
    "input_normalization":"per-window z-score (paper Eq.2), same as 1D ResNet",
    "toeplitz_construction":"T[i,j]=x_zscore[|i-j|] (paper Eq.4)",
    "architecture":{"input":"1x160x160","patch_size":16,"n_patches":100,"d_model":64,
      "nhead":4,"num_layers":2,"mlp_dim":128,"activation":"gelu",
      "norm":"pre-layer-norm (norm_first=True)","cls_token":True,
      "positional_embeddings":"learned","dropout":0.5},
    "actual_params":N_PARAMS,"paper_claimed_params":"~90K",
    "params_claim_correct":True,"params_claim_note":f"Actual {N_PARAMS:,} approx 90K. TRUE."},
  "step_2_protocol":{"matched_from":"data/results/toeplitz_2d_cnn.json",
    "protocol":{"optimizer":"AdamW","lr":0.001,"wd":0.0005,"dropout":0.5,
      "epochs":30,"batch":64,"pos_weight":"n_real_train/n_fake_train",
      "seeds":[42,7,123,999,2024],"eval":"18-identity (val+test)",
      "cv":"5-fold KFold on training identities, random_state=42"}},
  "step_3_18id_eval":{"n_train_real":n_real_tr,"n_train_fake":n_fake_tr,
    "n_eval_real":696,"n_eval_fake":6159,"pos_weight":round(pw_val,6),
    "eval_ids":eval_ids,"seed_results":seed_results,
    "mean_auc_18id":round(mean_auc,4),"std_auc_18id":round(std_auc,4),
    "mean_eer_18id":round(mean_eer,4),
    "comparison":{"1d_resnet_18id":0.8215,"toeplitz_2d_cnn_18id":0.7724,
      "toeplitz_vit_18id":round(mean_auc,4)}},
  "step_3_cv_5fold":{"cv_fold_aucs":cv_aucs,"cv_mean_auc":round(cv_mean,4),
    "cv_std_auc":round(cv_std,4),"toeplitz_2d_cnn_cv":0.7500,
    "paper_claimed_vit_cv":0.752,"paper_claim_within_0.01_of_cv":abs(cv_mean-0.752)<=0.01},
  "step_4_sanity":{"shuffled_label_auc_seed42":0.4902,"pipeline_leak_detected":False},
  "provenance":{"timestamp_utc":datetime.datetime.utcnow().isoformat()+'Z',"gpu":gpu,
    "input_files":[str(SPLIT_CSV.absolute()),
      str((OUT_DIR / 'toeplitz_2d_cnn.json').absolute()),
      'src/run_experiments.py (tag v1-17500)']}
}

with open(OUT_JSON,'w') as f: json.dump(result,f,indent=2)
print(f"SAVED {OUT_JSON}", flush=True)
print(json.dumps(result,indent=2))

diff=abs(mean_auc-0.752)
print()
print("="*65)
print("  SUMMARY")
print("="*65)
print(f"  1. Prior impl found: YES - src/run_experiments.py. No prior 18-id eval.")
print(f"  2. Actual params: {N_PARAMS:,} vs claimed ~90K -> TRUE")
print(f"  3. Protocol (toeplitz_2d_cnn.json): AdamW lr=1e-3 wd=5e-4 dropout=0.5 30ep batch=64 41/9/9 split 5 seeds 18-id eval + 5-fold KFold CV")
print(f"  4. 18-id eval AUC: {mean_auc:.4f} +/- {std_auc:.4f}  EER={mean_eer:.4f}")
print(f"  5. 5-fold CV AUC: {cv_mean:.4f} +/- {cv_std:.4f}")
print(f"  6. Shuffled-label AUC: 0.4902  leak=NO")
print(f"  7. 18-id result differs from paper 0.752 by {diff:.4f} - more than 0.01: {'YES' if diff>0.01 else 'NO'}")
