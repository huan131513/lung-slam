"""Stage 7 — DROID-SLAM wrapper.

DROID-SLAM is not pip-installable. It must be cloned and built separately,
then this module invokes its demo.py via subprocess.

Setup (on Windows + RTX 3080):
  1) git clone https://github.com/princeton-vl/DROID-SLAM
  2) cd DROID-SLAM
  3) Create conda env per their README (PyTorch + CUDA matching driver)
  4) python setup.py install
  5) Download droid.pth to DROID-SLAM/
  6) export DROID_SLAM_ROOT=/path/to/DROID-SLAM

Then this script will call:
    python $DROID_SLAM_ROOT/demo.py --imagedir <frames> --calib <calib.txt> ...
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .common import ensure_dir, log, run_cmd, timed

STAGE = "droid"


def run(frames_dir: Path, calib_txt: Path, out_dir: Path,
        masks_dir: Path | None = None,
        droid_root: str | None = None,
        weights: str | None = None,
        stride: int = 1,
        extra_args: list[str] | None = None) -> None:
    out_dir = ensure_dir(out_dir)
    droid_root = droid_root or os.environ.get("DROID_SLAM_ROOT")
    if not droid_root or not Path(droid_root).exists():
        log(STAGE, "DROID_SLAM_ROOT not set or not found", level="err")
        log(STAGE, "  on Windows+CUDA:")
        log(STAGE, "  git clone https://github.com/princeton-vl/DROID-SLAM")
        log(STAGE, "  cd DROID-SLAM && python setup.py install")
        log(STAGE, "  set DROID_SLAM_ROOT=C:\\path\\to\\DROID-SLAM")
        raise SystemExit(2)

    droid_root = Path(droid_root)
    demo = droid_root / "demo.py"
    if not demo.exists():
        log(STAGE, f"{demo} not found", level="err")
        raise SystemExit(2)

    if weights is None:
        weights = str(droid_root / "droid.pth")
    if not Path(weights).exists():
        log(STAGE, f"weights file {weights} not found — download from DROID-SLAM README", level="err")
        raise SystemExit(2)

    # Build command
    cmd = [
        sys.executable, str(demo),
        "--imagedir", str(frames_dir),
        "--calib", str(calib_txt),
        "--weights", weights,
        "--stride", str(stride),
        "--reconstruction_path", str(out_dir),
    ]
    if masks_dir is not None and Path(masks_dir).exists():
        cmd += ["--mask_dir", str(masks_dir)]
        log(STAGE, f"using masks from {masks_dir}")
        log(STAGE, "NOTE: official DROID-SLAM demo.py may not accept --mask_dir;")
        log(STAGE, "      if it errors, use a forked version or pre-bake masks into frames", level="warn")
    if extra_args:
        cmd += extra_args

    log(STAGE, f"DROID_SLAM_ROOT = {droid_root}")
    with timed(STAGE, "Running DROID-SLAM"):
        rc = run_cmd(cmd, stage=STAGE)
        if rc != 0:
            raise RuntimeError(f"DROID-SLAM exited with code {rc}")

    log(STAGE, f"outputs (if successful): {out_dir}", level="ok")
    log(STAGE, "  expected: poses.npy, disps.npy, tstamps.npy, images.npy")
