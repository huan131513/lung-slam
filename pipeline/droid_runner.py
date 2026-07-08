"""Stage 7 — DROID-SLAM wrapper.

DROID-SLAM is not pip-installable. It must be cloned and built separately,
then this module invokes its demo.py via subprocess.

Setup (Ubuntu + RTX 3090):
  1) git clone https://github.com/princeton-vl/DROID-SLAM
  2) cd DROID-SLAM
  3) Create conda env per their README (PyTorch + CUDA matching driver)
  4) python setup.py install
  5) Download droid.pth to DROID-SLAM/  (see their tools/download_model.sh)
  6) export DROID_SLAM_ROOT=/path/to/DROID-SLAM

This module calls the official demo.py with args that are 100% compatible
with upstream: --imagedir, --calib, --weights, --stride, --reconstruction_path.
The optional --mask_dir arg is NOT supported by the official demo.py; it is
only passed when you explicitly opt in with --use-masks and point --droid-root
at a fork that adds that flag.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .common import ensure_dir, log, run_cmd, timed

STAGE = "droid"


def run(frames_dir: Path, calib_txt: Path, out_dir: Path,
        masks_dir: Path | None = None,
        droid_root: str | None = None,
        weights: str | None = None,
        stride: int = 1,
        filter_thresh: float | None = None,
        keyframe_thresh: float | None = None,
        disable_vis: bool = True,
        extra_args: list[str] | None = None) -> None:
    out_dir = ensure_dir(out_dir)
    droid_root = droid_root or os.environ.get("DROID_SLAM_ROOT")
    if not droid_root or not Path(droid_root).exists():
        log(STAGE, "DROID_SLAM_ROOT not set or not found", level="err")
        log(STAGE, "  git clone https://github.com/princeton-vl/DROID-SLAM")
        log(STAGE, "  cd DROID-SLAM && python setup.py install")
        log(STAGE, "  export DROID_SLAM_ROOT=/path/to/DROID-SLAM")
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

    # current upstream demo.py torch.saves a single dict to --reconstruction_path
    # (must be a file, not a directory) — split it into per-array .npy files below.
    reconstruction_pth = out_dir / "reconstruction.pth"

    # demo.py does `sys.path.append('droid_slam')` (relative) and must be run
    # with droid_root as cwd, so all path args are made absolute here.
    cmd = [
        sys.executable, str(demo.resolve()),
        "--imagedir", str(frames_dir.resolve()),
        "--calib", str(calib_txt.resolve()),
        "--weights", str(Path(weights).resolve()),
        "--stride", str(stride),
        "--reconstruction_path", str(reconstruction_pth.resolve()),
    ]
    if filter_thresh is not None:
        # gates whether an incoming frame is even considered for tracking
        # (demo.py default 2.4) — lower keeps more candidate frames
        cmd += ["--filter_thresh", str(filter_thresh)]
    if keyframe_thresh is not None:
        # gates whether a tracked frame is kept as a permanent keyframe
        # (demo.py default 4.0) — lower keeps more keyframes
        cmd += ["--keyframe_thresh", str(keyframe_thresh)]
    if disable_vis:
        # demo.py's own live moderngl preview window runs as a non-daemon
        # subprocess that Droid.terminate() never actually stops (despite its
        # docstring) -- Python waits for it to be closed by hand before the
        # process can exit, hanging the shell after the real computation is
        # already done. Suppressing it also avoids confusing its different
        # (hardcoded 0.02) filter threshold with this pipeline's own viz-live.
        cmd += ["--disable_vis"]
    if masks_dir is not None and Path(masks_dir).exists():
        cmd += ["--mask_dir", str(Path(masks_dir).resolve())]
        log(STAGE, f"using masks from {masks_dir}")
        log(STAGE, "WARNING: official DROID-SLAM demo.py does NOT accept --mask_dir.", level="warn")
        log(STAGE, "         This will fail unless --droid-root points at a fork with mask support.", level="warn")
        log(STAGE, "         For the standard path, run with --no-masks (default).", level="warn")
    if extra_args:
        cmd += extra_args

    log(STAGE, f"DROID_SLAM_ROOT = {droid_root}")
    with timed(STAGE, "Running DROID-SLAM"):
        rc = run_cmd(cmd, stage=STAGE, cwd=str(droid_root))
        if rc != 0:
            raise RuntimeError(f"DROID-SLAM exited with code {rc}")

    # unpack the single torch.save bundle into the per-array .npy files the
    # rest of this pipeline (viz stage) expects.
    bundle = torch.load(reconstruction_pth, map_location="cpu")

    # bundle["poses"] is DROID-SLAM's internal world-to-camera SE3 buffer, NOT
    # the camera position in world space. Droid.terminate() (the documented
    # public API) and DROID-SLAM's own visualizer/view_reconstruction.py both
    # invert it before use — do the same so poses.npy's translation column is
    # the actual camera position (otherwise the plotted trajectory is wrong).
    from lietorch import SE3
    poses_c2w = SE3(bundle["poses"]).inv().data.numpy()
    np.save(out_dir / "poses.npy", poses_c2w)
    for key in ("tstamps", "images", "disps", "intrinsics"):
        np.save(out_dir / f"{key}.npy", bundle[key].numpy())

    try:
        _save_point_cloud(bundle, out_dir / "points.ply")
    except Exception as e:
        log(STAGE, f"point cloud export skipped: {e}", level="warn")

    log(STAGE, f"outputs: {out_dir}", level="ok")
    log(STAGE, "  wrote: poses.npy, disps.npy, tstamps.npy, images.npy, intrinsics.npy, points.ply")


def _save_point_cloud(bundle: dict, ply_path: Path,
                       filter_thresh: float = 0.005, filter_count: int = 2) -> None:
    """Back-project keyframe depth maps into a colored world-frame point cloud.

    Same math as DROID-SLAM's own view_reconstruction.py: droid_backends.iproj
    unprojects each pixel using its inverse depth + pose, depth_filter keeps
    only points with consistent depth across >= filter_count other views.
    """
    import droid_backends
    import open3d as o3d
    from lietorch import SE3

    images = bundle["images"].cuda()[..., ::2, ::2]
    disps = bundle["disps"].cuda()[..., ::2, ::2].contiguous()
    poses = bundle["poses"].cuda()
    intrinsics = 4 * bundle["intrinsics"].cuda()

    index = torch.arange(len(images), device="cuda")
    thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))

    points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
    colors = images[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.0
    counts = droid_backends.depth_filter(poses, disps, intrinsics[0], index, thresh)

    mask = (counts >= filter_count) & (disps > 0.25 * disps.mean())
    points_np = points[mask].cpu().numpy()
    colors_np = colors[mask].cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)
    o3d.io.write_point_cloud(str(ply_path), pcd)
    log(STAGE, f"points.ply -> {ply_path}  ({len(points_np)} pts)", level="ok")
