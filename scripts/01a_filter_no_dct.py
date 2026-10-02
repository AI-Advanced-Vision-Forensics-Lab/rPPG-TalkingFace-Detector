#!/usr/bin/env python3
"""
01a_filter_no_dct.py — Remove waveforms that have no lip-DCT feature vector.

Lip-DCT extraction (02_extract_lip_dct.py) finds no face in some clips. In our
runs those waveforms were moved out of the waveform folder before the split was
built, so every model (including the rPPG-only ones) sees the same samples:

  real : 64 windows from 32 source videos (all windows of 9 sources)
  fake : 3 EchoMimic videos

A waveform is excluded when its video_id has no row in dct_features.csv.
video_id is the file stem for real windows (idXX_YYYY_wK) and
<method>__<stem> for fakes.

Default is a dry run that only prints counts. --apply moves the files to
<wave-dir>/excluded/{real,fake}/ (nothing is deleted).

Layouts:
  flat    : <wave-dir>/real/<stem>_w<k>.npy, <wave-dir>/fake/<method>__<stem>.npy
  celebdf : <wave-dir>/CelebDF/Celeb-real/<stem>_w<k>.npy,
            <wave-dir>/CelebDF/TalkingFace/<method>/<stem>.npy

Run before building the split (and, for the flat layout, before creating the
CelebDF/ symlinks described in the README).
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="List or move waveforms without lip-DCT features")
    p.add_argument("--data-root", default="data",
                   help="data folder (default: ./data)")
    p.add_argument("--wave-dir", default=None,
                   help="waveform folder (default: <data-root>/waveforms)")
    p.add_argument("--dct-file", default=None,
                   help="lip-DCT feature CSV (default: <data-root>/dct_features.csv)")
    p.add_argument("--layout", default="flat", choices=["flat", "celebdf"],
                   help="waveform folder layout (default: flat)")
    p.add_argument("--classes", default="both", choices=["real", "fake", "both"],
                   help="which classes to filter (default: both)")
    p.add_argument("--apply", action="store_true",
                   help="move the files to <wave-dir>/excluded/ (default: dry run)")
    p.add_argument("--list", action="store_true",
                   help="print every excluded video_id")
    return p.parse_args()


def scan(wave_dir, layout, cls):
    """Yield (video_id, path) for every waveform of one class."""
    if layout == "flat":
        for f in sorted((wave_dir / cls).glob("*.npy")):
            yield f.stem, f
    elif cls == "real":
        for f in sorted((wave_dir / "CelebDF" / "Celeb-real").glob("*.npy")):
            yield f.stem, f
    else:
        for mdir in sorted(p for p in (wave_dir / "CelebDF" / "TalkingFace").iterdir() if p.is_dir()):
            for f in sorted(mdir.glob("*.npy")):
                yield f"{mdir.name}__{f.stem}", f


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    wave_dir = Path(args.wave_dir) if args.wave_dir else data_root / "waveforms"
    dct_file = Path(args.dct_file) if args.dct_file else data_root / "dct_features.csv"

    if not dct_file.exists():
        sys.exit(f"DCT feature file not found: {dct_file}")
    with open(dct_file, newline="") as fh:
        dct_ids = {row["video_id"] for row in csv.DictReader(fh)}
    print(f"DCT vectors: {len(dct_ids)}  ({dct_file})")
    print(f"Waveforms:   {wave_dir}  (layout: {args.layout})")

    classes = ["real", "fake"] if args.classes == "both" else [args.classes]
    total = 0
    for cls in classes:
        rows = list(scan(wave_dir, args.layout, cls))
        missing = [(vid, p) for vid, p in rows if vid not in dct_ids]
        total += len(missing)
        line = f"  {cls:4s}: {len(rows):6d} waveforms, {len(missing):4d} without DCT"
        if cls == "real":
            srcs = {vid.rsplit('_w', 1)[0] for vid, _ in rows}
            gone = {vid.rsplit('_w', 1)[0] for vid, _ in missing}
            kept = {vid.rsplit('_w', 1)[0] for vid, _ in rows if vid in dct_ids}
            line += (f"  (from {len(gone)} sources, {len(gone - kept)} lose every window;"
                     f" {len(kept)} of {len(srcs)} sources remain)")
        else:
            by_method = {}
            for vid, _ in missing:
                m = vid.split("__")[0]
                by_method[m] = by_method.get(m, 0) + 1
            if by_method:
                line += "  " + ", ".join(f"{m}: {n}" for m, n in sorted(by_method.items()))
        print(line)
        if args.list:
            for vid, _ in missing:
                print(f"    {vid}")
        if args.apply:
            dest = wave_dir / "excluded" / cls
            dest.mkdir(parents=True, exist_ok=True)
            for vid, p in missing:
                shutil.move(str(p), str(dest / p.name))
            print(f"    moved {len(missing)} files to {dest}")

    if not args.apply:
        print(f"Dry run: {total} waveforms would be moved. Re-run with --apply to move them.")


if __name__ == "__main__":
    main()
