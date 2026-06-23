"""Stage 6 — Camera intrinsics estimation.

Heuristic K from image dimensions; optionally refine via COLMAP auto-init
on a small subset of frames (requires pycolmap).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from .common import ensure_dir, list_frames, log, timed

STAGE = "calib"


def heuristic_K(width: int, height: int, fov_deg: float = 70.0) -> dict:
    """Approximate pinhole intrinsics from image size and assumed horizontal FOV."""
    import math
    fx = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height,
            "model": "PINHOLE", "source": f"heuristic (fov={fov_deg}°)"}


def run(frames_dir: Path, out_dir: Path, method: str = "heuristic",
        fov_deg: float = 70.0) -> None:
    out_dir = ensure_dir(out_dir)
    frames = list_frames(Path(frames_dir))
    if not frames:
        raise FileNotFoundError(frames_dir)

    sample = cv2.imread(str(frames[0]))
    h, w = sample.shape[:2]
    log(STAGE, f"frame size: {w}×{h}, total {len(frames)} frames")

    if method == "heuristic":
        with timed(STAGE, f"computing heuristic K (FOV={fov_deg}°)"):
            K = heuristic_K(w, h, fov_deg)
    elif method == "colmap":
        try:
            import pycolmap  # noqa: F401
        except ImportError:
            log(STAGE, "pycolmap not installed; falling back to heuristic", level="warn")
            log(STAGE, "  pip install pycolmap")
            K = heuristic_K(w, h, fov_deg)
        else:
            log(STAGE, "TODO: pycolmap-based auto-init not yet implemented; using heuristic", level="warn")
            K = heuristic_K(w, h, fov_deg)
    else:
        raise ValueError(f"unknown method: {method}")

    out_path = out_dir / "calib.json"
    out_path.write_text(json.dumps(K, indent=2))
    log(STAGE, f"fx={K['fx']:.1f}  fy={K['fy']:.1f}  cx={K['cx']:.1f}  cy={K['cy']:.1f}")
    log(STAGE, f"saved → {out_path}", level="ok")

    # Also write DROID-SLAM compatible calib.txt (single line: fx fy cx cy)
    txt = out_dir / "calib.txt"
    txt.write_text(f"{K['fx']:.4f} {K['fy']:.4f} {K['cx']:.4f} {K['cy']:.4f}\n")
    log(STAGE, f"DROID-SLAM calib.txt → {txt}", level="ok")
