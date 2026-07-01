#!/usr/bin/env python3
"""Standalone CLI wrapper for pipeline.preprocess.

Prefer `python run.py preprocess ...` from the repo root. This script is kept
so you can invoke preprocessing without going through run.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make repo root importable when invoked as scripts/preprocess_for_droid.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import preprocess  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Preprocess endoscope video for DROID-SLAM.")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path,
                    help="Output root. Will create frames/, calib.txt, preview.png, metadata.json")
    ap.add_argument("--target-width", type=int, default=1280)
    ap.add_argument("--fps", type=float, default=15.0,
                    help="Target fps after ffmpeg fps filter (default 15; use 0 to keep source fps)")
    ap.add_argument("--distortion-margin", type=float, default=0.15,
                    help="Shrink FOV radius by this fraction to trim peripheral distortion")
    ap.add_argument("--assumed-fov-deg", type=float, default=90.0,
                    help="Assumed horizontal FOV of the full circular view (used only for rough calib)")
    ap.add_argument("--fov-threshold", type=int, default=12,
                    help="Pixel intensity threshold for FOV detection")
    ap.add_argument("--keep-raw", action="store_true",
                    help="Keep intermediate raw_frames/ folder")
    args = ap.parse_args()

    fps = args.fps if args.fps and args.fps > 0 else None
    preprocess.run(
        video=args.video,
        out_dir=args.outdir,
        target_width=args.target_width,
        target_fps=fps,
        distortion_margin=args.distortion_margin,
        assumed_fov_deg=args.assumed_fov_deg,
        fov_threshold=args.fov_threshold,
        keep_raw=args.keep_raw,
    )
    frames = args.outdir / "frames"
    calib = args.outdir / "calib.txt"
    print("\nNext (from $DROID_SLAM_ROOT):")
    print(f"  python demo.py --imagedir={frames.resolve()} \\")
    print(f"                 --calib={calib.resolve()} \\")
    print(f"                 --reconstruction_path={(args.outdir/'droid').resolve()}")


if __name__ == "__main__":
    main()
