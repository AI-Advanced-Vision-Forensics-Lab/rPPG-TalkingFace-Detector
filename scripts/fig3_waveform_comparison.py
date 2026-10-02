#!/usr/bin/env python3
"""
generate_waveform_comparison.py — Side-by-side rPPG waveform comparison for paper.
Real vs IP-LAP (hard fake) vs Real3DPortrait (easy fake).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths (configurable; see README) ─────────────────────────────────────────
import argparse as _argparse
_ap = _argparse.ArgumentParser(description='rPPG waveform comparison figure')
_ap.add_argument('--data-root', default='data',
                 help='data folder laid out as described in the README (default: ./data)')
_ap.add_argument('--out-dir', default='figures', help='folder for the figure (default: ./figures)')
_args = _ap.parse_args()
DATA = Path(_args.data_root)
OUT  = Path(_args.out_dir)

PANELS = [
    (
        DATA / "waveforms/real/id4_0000_w0.npy",
        "Real",
        "#000000",
        "",
    ),
    (
        DATA / "waveforms/CelebDF/TalkingFace/IP_LAP/id40_0000_test_id00866_usqvLtEq2qQ.npy",
        "IP-LAP",
        "#c0392b",
        "AUC 0.690",
    ),
    (
        DATA / "waveforms/CelebDF/TalkingFace/Real3DPortrait/id40_0000_test_id00866_usqvLtEq2qQ.npy",
        "Real3DPortrait",
        "#1f77b4",
        "AUC 0.985",
    ),
]

fig, axes = plt.subplots(3, 1, figsize=(5, 4), sharex=True,
                         facecolor="white",
                         gridspec_kw={"hspace": 0.08})

frames = np.arange(160)

for i, (path, ylabel, color, auc_text) in enumerate(PANELS):
    ax = axes[i]
    ax.set_facecolor("white")

    w = np.load(path).astype(np.float32)
    w = (w - w.mean()) / (w.std() + 1e-6)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)
    ax.plot(frames, w, color=color, linewidth=1.2, zorder=2)

    ax.set_ylim(-3.5, 3.5)
    ax.set_yticks([-2.5, 0, 2.5])
    ax.set_yticklabels(['-2.5', '0', '2.5'])
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, color="gray", alpha=0.2, linewidth=0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # AUC annotation top-right
    ax.text(0.98, 0.90, auc_text, transform=ax.transAxes,
            fontsize=8, color="gray", ha="right", va="top")

axes[-1].set_xlabel("Frame index", fontsize=10)
axes[-1].set_xlim(0, 159)

plt.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "waveform_comparison.pdf", format="pdf", bbox_inches="tight")
fig.savefig(OUT / "waveform_comparison.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("Saved")
