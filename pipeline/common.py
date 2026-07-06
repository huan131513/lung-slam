"""Shared utilities: timestamped logger, progress wrappers, environment probe."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


# ---------- ANSI colors (auto-disable if not a TTY) ----------
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR", "") == ""


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _USE_COLOR else s


def cyan(s):    return _c("36", s)
def green(s):   return _c("32", s)
def yellow(s):  return _c("33", s)
def red(s):     return _c("31", s)
def bold(s):    return _c("1",  s)
def dim(s):     return _c("2",  s)


# ---------- Logger ----------
def log(stage: str, msg: str, *, level: str = "info") -> None:
    """Print a timestamped, color-coded log line.

    Format: [HH:MM:SS | stage] message
    """
    ts = time.strftime("%H:%M:%S")
    tag = f"[{dim(ts)} | {cyan(stage)}]"
    if level == "ok":
        msg = green("✓ ") + msg
    elif level == "warn":
        msg = yellow("⚠ ") + msg
    elif level == "err":
        msg = red("✗ ") + msg
    print(f"{tag} {msg}", flush=True)


def section(title: str) -> None:
    """Print a section divider."""
    bar = "─" * max(8, 70 - len(title) - 2)
    print(f"\n{cyan(bold('▶ ' + title))} {dim(bar)}\n", flush=True)


# ---------- Timing ----------
@contextmanager
def timed(stage: str, label: str):
    """Context manager that logs start and finish (with elapsed seconds)."""
    log(stage, f"{label} ...")
    t0 = time.time()
    try:
        yield
    except Exception as e:
        log(stage, f"{label} FAILED after {time.time()-t0:.1f}s — {type(e).__name__}: {e}", level="err")
        raise
    else:
        log(stage, f"{label} (took {time.time()-t0:.1f}s)", level="ok")


# ---------- Progress ----------
def _progress_manual(iterable, *, total=None, desc=""):
    """Generator fallback used when tqdm is unavailable. Prints every 50 items."""
    count = 0
    total = total or (len(iterable) if hasattr(iterable, "__len__") else None)
    t0 = time.time()
    for item in iterable:
        yield item
        count += 1
        if count % 50 == 0 or count == total:
            rate = count / max(1e-6, time.time() - t0)
            print(f"  {desc}: {count}/{total or '?'} ({rate:.1f} it/s)", flush=True)


def progress(iterable, *, total=None, desc=""):
    """Wrap an iterable with tqdm if available, else manual generator.

    IMPORTANT: keep this function non-generator (no `yield`) so tqdm branch
    returns the iterator object directly.
    """
    if _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc, unit="frm", dynamic_ncols=True)
    return _progress_manual(iterable, total=total, desc=desc)


# ---------- Environment probe ----------
def probe_env(stage: str = "check") -> dict:
    """Probe Python, CUDA, ffmpeg, key packages. Print to stdout, return dict."""
    info: dict = {}

    info["python"] = sys.version.split()[0]
    log(stage, f"Python {info['python']}")

    info["platform"] = sys.platform
    log(stage, f"Platform: {info['platform']}")

    # ffmpeg
    ff = shutil.which("ffmpeg")
    info["ffmpeg"] = ff
    if ff:
        log(stage, f"ffmpeg: {ff}", level="ok")
    else:
        log(stage, "ffmpeg NOT found — Stage 1 extract requires it", level="err")

    # PyTorch + CUDA / MPS
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        msg = f"torch {torch.__version__}"
        if info["cuda_available"]:
            msg += f", CUDA {torch.version.cuda} ({torch.cuda.get_device_name(0)})"
            log(stage, msg, level="ok")
        elif info["mps_available"]:
            msg += ", MPS available (Apple GPU)"
            log(stage, msg, level="ok")
        else:
            log(stage, msg + " — CPU only (heavy stages will be slow / unusable)", level="warn")
    except ImportError:
        info["torch"] = None
        log(stage, "torch NOT installed — needed for SAM 2 / DROID-SLAM", level="warn")

    # SAM 2
    try:
        import sam2  # noqa: F401
        info["sam2"] = True
        log(stage, "sam2 package present", level="ok")
    except ImportError:
        info["sam2"] = False
        log(stage, "sam2 NOT installed (install on Windows 3080 machine)", level="warn")

    # DROID-SLAM is not pip-importable; we check by env var
    droid_root = os.environ.get("DROID_SLAM_ROOT")
    info["droid_slam_root"] = droid_root
    if droid_root and Path(droid_root).exists():
        log(stage, f"DROID_SLAM_ROOT = {droid_root}", level="ok")
    else:
        log(stage, "DROID_SLAM_ROOT not set (export DROID_SLAM_ROOT=/path/to/DROID-SLAM)", level="warn")

    # Open3D
    try:
        import open3d  # noqa: F401
        info["open3d"] = True
        log(stage, f"open3d {open3d.__version__}", level="ok")
    except ImportError:
        info["open3d"] = False
        log(stage, "open3d NOT installed (needed for visualization)", level="warn")

    # OpenCV
    try:
        import cv2
        info["cv2"] = cv2.__version__
        log(stage, f"opencv {cv2.__version__}", level="ok")
    except ImportError:
        info["cv2"] = None
        log(stage, "opencv-python NOT installed", level="err")

    return info


# ---------- Path helpers ----------
def ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_frames(frames_dir, exts=(".png", ".jpg")) -> list[Path]:
    frames_dir = Path(frames_dir)
    files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in exts)
    return files


def run_cmd(cmd: list[str], stage: str = "shell", cwd: str | None = None) -> int:
    """Run a subprocess, stream stdout/stderr, return returncode."""
    log(stage, f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False, cwd=cwd)
    if proc.returncode != 0:
        log(stage, f"command exited with code {proc.returncode}", level="err")
    return proc.returncode
