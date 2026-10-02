# PulseGuard: rPPG-Derived Temporal Signals as a Forensic Modality for Talking-Face Deepfake Detection

**Othmane Harraq, Tamer Aldwairi** — Temple University

Code and result files for the revised PulseGuard paper. RhythmFormer extracts a 160-sample rPPG waveform from each face window. Lightweight 1D classifiers then separate real from talking-face (TF) videos in the TF subset of Celeb-DF++, using an identity-disjoint split.

| | Window AUC | Video AUC (mean-pooled) |
|---|---|---|
| 1D ResNet (240K) | 0.826 ± 0.005 | 0.842 ± 0.006 |
| 1D CNN (36K) | 0.810 ± 0.002 | 0.846 ± 0.009 |
| Transformer (69K) | 0.806 ± 0.004 | 0.815 ± 0.012 |
| DeepFakesON-Phys (reproduction) | — | 0.576 |

The numbers are means over 5 seeds on the 18 validation + test identities. The DeepFakesON-Phys row uses only the 9 test identities.

> **Earlier version.** The code for the first version of this work (17,500 fakes, AUC 0.806) is preserved under the git tag [`v1-17500`](../../tree/v1-17500). The current code uses the full 20,279-video TF corpus and supersedes it.

---

## Repository layout

```
scripts/        one script per table / figure / section (see "Reproducing the paper")
results/        the result JSON files behind every number in the paper (absolute paths removed)
results/logs/   phase2_full59.log, source of the per-identity numbers in Sec. 6.2
splits/         identity_split.csv (59 identities -> train/val/test) and corpus_full59.csv
figures/        roc_curves.pdf (Fig. 2)
```

## Setup

Python 3.10 and an NVIDIA GPU are required for extraction and training.

```bash
pip install -r requirements.txt          # see the note in requirements.txt about the CUDA build of torch
git clone https://github.com/zizheng-guo/RhythmFormer.git
git -C RhythmFormer checkout 7c9cea2     # the commit used for all extractions
```

- **RhythmFormer checkpoint.** Download `UBFC_cross_RhythmFormer.pth` from the RhythmFormer authors and place it at `RhythmFormer/PreTrainedModels/UBFC_cross_RhythmFormer.pth`. We do not redistribute it.
- **Face detector.** The MediaPipe BlazeFace short-range model (`blaze_face_short_range.tflite`) is downloaded automatically into `--data-root` on first use.
- **ffprobe.** Required for Sec. 5.7 and Sec. 6.4 (`apt install ffmpeg`).
- **Arguments.** Every script takes `--data-root` (default `./data`) and `--out-dir` (default `<data-root>/results`). Extraction scripts also take `--rhythmformer-dir` (default `./RhythmFormer`).

## Dataset

We use the TalkingFace subset of **Celeb-DF++** (Li et al., 2025): 590 real videos of 59 celebrities and 20,279 TF videos from seven generators. Request access from the authors at <https://github.com/OUC-VAS/Celeb-DF-PP>. The dataset is released for non-commercial academic research under the authors' terms of use.

**This repository contains no videos, frames, face crops, extracted waveforms or model weights.** You need your own copy of the dataset to run anything.

Expected layout under `--data-root`:

```
data/
├── Celeb-DF-v3/                      # the Celeb-DF++ release as unpacked
│   ├── Celeb-real/*.mp4
│   └── Celeb-synthesis/TalkingFace/{AniTalker,EchoMimic,EDTalk,FLOAT,IP_LAP,Real3DPortrait,SadTalker}/*.mp4
├── videos/                           # used by Sec. 5.7, Sec. 6.4 and DeepFakesON-Phys
│   ├── real/*.mp4                    #   copy or symlink of Celeb-DF-v3/Celeb-real
│   └── fake/<method>/*.mp4           #   copy or symlink of Celeb-DF-v3/Celeb-synthesis/TalkingFace/<method>
└── waveforms/                        # written by the scripts
    ├── CelebDF/TalkingFace/<method>/*.npy
    ├── CelebDF/Celeb-real/*.npy
    └── real -> CelebDF/Celeb-real    # 01_build_split.py reads waveforms/real (see note below)
```

> **Note on `waveforms/real`.** The real-video waveforms used in the paper were written to `waveforms/real/` by the original extraction script of the first version (tag `v1-17500`, `src/extract_waveforms.py`). `00_extract_waveforms.py` writes the same stride-60 windows to `waveforms/CelebDF/Celeb-real/`. We checked that the two sets are byte-identical on 400 sampled files. Create the symlink `ln -s CelebDF/Celeb-real data/waveforms/real` before building the split.

> **Real windows.** Of the 590 real source videos, 2 yield no waveform (id8_0008, id27_0005). Lip-DCT extraction, used by the companion fusion paper ([rppg-dct-fusion](https://github.com/AI-Advanced-Vision-Forensics-Lab/rppg-dct-fusion)), found no face in 64 windows: all 19 windows of 9 sources and 45 windows of 23 other sources. These windows were removed from the corpus as well. This is why the corpus has 579 sources and 2,371 windows, while `00_extract_waveforms.py` writes 2,435 windows from 588 sources. `scripts/01a_filter_no_dct.py` reproduces the removal from the companion repository's `dct_features.csv` (written by its `02_extract_lip_dct.py`): `python scripts/01a_filter_no_dct.py --layout celebdf --classes real` prints the counts, and adding `--apply` moves the 64 windows to `waveforms/excluded/real/`. Only real windows are filtered here; the 3 EchoMimic videos without DCT features remain in this corpus. The kept windows are also listed in `splits/corpus_full59.csv`, and `01_build_split.py` checks that it finds exactly 2,371 real windows.

## Identity split

`splits/identity_split.csv` assigns each of the 59 identities to `train` (41), `val` (9) or `test` (9). The test identities are id0, id4, id6, id11, id13, id16, id23, id27 and id54. `splits/corpus_full59.csv` lists every waveform (2,371 real windows and 20,279 fakes) with its method, identity and split, matching Table 1. `01_build_split.py` writes `data/dataset_split_full59.csv`, which the training scripts read.

## Reproducing the paper

Run from the repository root. Each script writes the JSON named in the last column, and that file is shipped in `results/` for comparison. All training scripts use seeds {42, 7, 123, 999, 2024}.

| Paper item | Command | Output JSON (shipped) |
|---|---|---|
| Waveform extraction (Sec. 3.2) | `python scripts/00_extract_waveforms.py` | `data/celebdf_wave_manifest.csv` |
| Drop real windows without lip DCT | `python scripts/01a_filter_no_dct.py --layout celebdf --classes real --apply` (needs the companion repository's `dct_features.csv`; see "Real windows" above) | — |
| Table 1: identity split | `python scripts/01_build_split.py` | `split_full59_summary.json` |
| Sec. 3.4 / 4.4: hyperparameter search (earlier 17,500-fake corpus) | `python scripts/sec34_hp_search_v1.py --data-root <v1-17500 data folder>` | `hp_tuning_v1_corpus.json` (written as `hp_tuning.json`) |
| Table 2: technique isolation (5-fold StratifiedGroupKFold, seed 42) | `python scripts/table2_technique_isolation.py` then `python scripts/table2_technique_isolation_both.py` | `technique_isolation.json`, `technique_isolation_both.json` |
| Table 3: ResNet / CNN / Transformer, window and video AUC | `python scripts/table3_main_results.py` | `combined_retrain_video_level.json` |
| Table 3: DeepFakesON-Phys | `python scripts/table3_deepfakeson_phys.py --dfp-dir <folder with model.onnx>` | `deepfakeson_phys_baseline.json` |
| Sec. 3.5: Toeplitz 2D CNN | `python scripts/sec35_toeplitz_2d_cnn.py` | `toeplitz_2d_cnn.json` |
| Sec. 3.5: Toeplitz ViT (18-identity eval; 5-fold CV) | `python scripts/sec35_toeplitz_vit.py`; `python scripts/sec35_toeplitz_vit_cv.py` | `toeplitz_vit.json` |
| Table 4, Table 5, Sec. 6.2 | `python scripts/table4_table5_phase2.py` | `full59_combined_model.json`, `full59_per_generator_breakdown.json`, `full59_per_method_isolated.json`, per-identity AUCs in `results/logs/phase2_full59.log` |
| Table 6: leave-one-generator-out | `python scripts/table6_logo.py` | `logo_generalization.json` |
| Sec. 5.7: 30 fps extraction and split | `python scripts/sec57_extract_waveforms_30fps.py`; `python scripts/sec57_build_split_30fps.py` | `split_30fps_summary.json` |
| Sec. 5.7: 0.822 → 0.846, per-generator gains | `python scripts/sec57_phase_b_30fps.py` | `phase_b_30fps.json` (also `task4_fps_normalization.json`, see notes) |
| Sec. 5.7: rate control (0.631 / 0.481 shuffled), 2,225 excluded clips | `python scripts/sec57_fps_confound_control.py` | `fps_confound_control.json` |
| Sec. 5.7: excluding stretched clips (+0.006, ρ = 1.00) | see notes | `label_and_corpus_audit.json` → `step_3_no_stretched_reeval` |
| Sec. 6.4: method-label metadata | `python scripts/sec64_method_label_validation.py` (fps, resolution, codec, encoder) | `task3_method_label_validation.json`; level and FLOAT/Real3DPortrait comparison in `label_and_corpus_audit.json` → `step_2_method_label_verification` |
| Fig. 2: per-method ROC | `python scripts/fig2_roc_curves.py` | `figures/roc_curves.pdf` |

### Provenance notes

- **Hyperparameters (Sec. 3.4, 4.4).** The search ran one setting at a time (lr, wd, dropout, each over three values around a base configuration) with 5-fold StratifiedGroupKFold on the training identities of the earlier 17,500-fake corpus (tag `v1-17500`, 1,709 real + 11,947 fake training waveforms), not on the 20,279-fake corpus. The 1D ResNet uses the search optimum. The 1D CNN and Transformer use the base settings, which are within about 0.01 AUC of the optimum (less than the fold standard deviation). The search used raw waveforms without per-window z-score, scored each fold at its best epoch on the validation fold, and its "1D CNN" is the earlier 56K-parameter variant (kernels 9/7/5), not the 36K 1D CNN of Table 3. `sec34_hp_search_v1.py` is the search portion of `src/run_experiments.py` from that tag, with paths made configurable; it needs that version's `dataset_split.csv` (`src/build_split.py` in the tag) and `waveforms/{real,fake}/` layout.
- **Two training runs.** Tables 4 and 5, Fig. 2's legend values and the 0.822 baseline in Sec. 5.7 come from the original training run (`table4_table5_phase2.py`, pooled window AUC 0.8215). Table 3 comes from a later retrain with identical data and hyperparameters (`table3_main_results.py`, pooled window AUC 0.8259). That retrain also saved the checkpoints needed for video-level pooling. The two runs differ only by training stochasticity.
- **Stretched-clip exclusion (Sec. 5.7).** The +0.006 and ρ = 1.00 are recorded in `results/label_and_corpus_audit.json` (`step_3_no_stretched_reeval`). The script that wrote this file is not part of this repository. The analysis evaluated the 5 checkpoints written by `scripts/sec57_task2_retrain_checkpoints.py` (window AUC 0.8256 with all 6,855 eval windows) on the 6,190 eval windows that were not linspace-stretched (0.8314). Its per-generator ρ compares those values with the per-generator AUCs of the original run (Table 4).
- **Metadata check (Sec. 6.4).** `sec64_method_label_validation.py` produced `task3_method_label_validation.json` (150 videos per method: fps, resolution, codec, encoder, pixel format). The H.264 level values and the extended comparison are in `label_and_corpus_audit.json` (50 videos per method), whose generating script is not included. In that file EchoMimic's encoder field is mixed (`Lavf58.29.100` / `Lavc61.3.100 libx264`), and FLOAT and Real3DPortrait match on every measured field.
- **Edited result file.** `task4_fps_normalization.json` is shipped because `sec57_fps_confound_control.py` reads it. One section belonging to unrelated work was removed from it; the remaining fields are unchanged. The script that wrote it is not included.
- **Missing source videos in the 30 fps run.** In our runs, `data/videos/fake` held a working copy in which the source videos of identities id53–id61 were no longer present (2,779 clips, recorded as `missing_source` in `fps_confound_control.json`). Their native-rate waveforms had already been extracted. With the full release these clips will be found, so 30 fps corpus counts will differ.
- **30 fps evaluation set (Sec. 5.7).** The 30 fps run excluded 5,004 of the 20,279 forgeries: 2,225 too short for a 160-frame window at 30 fps and 2,779 whose source videos (id53–id61) were unavailable. Real waveforms were reused as extracted, not resampled. The native and 30 fps evaluations therefore differ in their evaluation set as well as in frame rate. The stretched-clip comparison (+0.006, ρ = 1.00) is separate: it uses the task2 retrain checkpoints, and its rank correlation is computed against Table 4's original run (see "Stretched-clip exclusion" above). `label_and_corpus_audit.json` has no producing script in this repository.
- **DeepFakesON-Phys.** `table3_deepfakeson_phys.py` runs an **ONNX conversion of the released `DeepFakesON-Phys_CelebDF_V2.h5` weights** with onnxruntime on CPU. The preprocessing is **re-implemented** following the official `vid_to_deepframes_rawframes.py`: 36×36 face, DeepFrames from normalised temporal differences, RawFrames from normalised appearance, video score = mean frame score. One difference is that the Haar-cascade face box is detected on the first frame and reused for all frames, with a centre-crop fallback. **The script used for the h5 → ONNX conversion was not found and is not included**; the metadata of the ONNX file we used names tf2onnx 1.17.0 as the converter. Obtain the weights from the official DeepFakesON-Phys repository; we do not redistribute them. The run scored 100 real and 3,171 of the 3,466 fake test-identity videos from the `videos/` working copy. The other 295 test fakes belong to id53–id61, whose source videos were unavailable in that working copy (see "Missing source videos in the 30 fps run" above). Per-video scores were not saved.
- **Fig. 2.** The curves are 5-seed mean ROC curves, computed by retraining inside the script. The AUC values in the legend are fixed constants taken from Table 4.
- **Figs. 1 and 3.** Their sources are not included. Fig. 1 (pipeline diagram) contains a frame from the dataset. The source of the submitted Fig. 3 (example traces) was not found.
- **Path edits only.** Relative to the code that produced the results, the scripts were changed only to make paths configurable (`--data-root`, `--out-dir`, `--rhythmformer-dir`, `--dfp-dir`), to drop the face-swap / face-reenactment / YouTube-real branches from the extraction script, and to rename files. `01a_filter_no_dct.py` is new and replaces the manual removal of the real windows without lip DCT; a dry run on our data identifies exactly the 64 removed windows. The scripts were not re-run end to end after these edits. They were checked statically: they compile, `--help` works, and referenced paths exist.

## Citation

```bibtex
@article{harraq2026pulseguard,
  title   = {PulseGuard: rPPG-Derived Temporal Signals as a Forensic Modality for Talking-Face Deepfake Detection},
  author  = {Harraq, Othmane and Aldwairi, Tamer},
  journal = {arXiv preprint arXiv:2607.21776},
  year    = {2026}
}
```

## Acknowledgements

[RhythmFormer](https://github.com/zizheng-guo/RhythmFormer), [MediaPipe](https://github.com/google-ai-edge/mediapipe), [DeepFakesON-Phys](https://github.com/BiDAlab/DeepFakesON-Phys) and [Celeb-DF++](https://github.com/OUC-VAS/Celeb-DF-PP).
