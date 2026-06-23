"""Stage 2/3/5 — FOV mask, bad-frame filter, mask combination."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .common import ensure_dir, list_frames, log, progress, timed


# =========================================================================
# Stage 2 — Circular FOV mask
# =========================================================================
STAGE_FOV = "fovmask"


def build_fov_mask(frames_dir: Path, out_path: Path, threshold: int = 12) -> np.ndarray:
    """Compute one circular FOV mask from the first frame (foreground = pixels brighter than threshold)."""
    frames = list_frames(frames_dir)
    if not frames:
        raise FileNotFoundError(f"no frames in {frames_dir}")
    log(STAGE_FOV, f"reading {frames[0].name} for FOV detection")
    img = cv2.imread(str(frames[0]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initial threshold
    fov = (gray > threshold).astype(np.uint8) * 255

    # Morphological closing to fill small gaps
    k = np.ones((9, 9), np.uint8)
    fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, k)

    # Take largest connected component
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fov, connectivity=8)
    if n > 1:
        sizes = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(sizes))
        fov = (labels == keep).astype(np.uint8) * 255

    cv2.imwrite(str(out_path), fov)
    coverage = (fov > 0).mean()
    log(STAGE_FOV, f"FOV coverage = {coverage*100:.1f}% of frame area", level="ok")
    log(STAGE_FOV, f"saved → {out_path}", level="ok")
    return fov


def run_fovmask(frames_dir: Path, out_dir: Path) -> None:
    out_dir = ensure_dir(out_dir)
    out_path = out_dir / "fov_mask.png"
    with timed(STAGE_FOV, "Building circular FOV mask"):
        build_fov_mask(Path(frames_dir), out_path)


# =========================================================================
# Stage 3 — Bad-frame filter
# =========================================================================
STAGE_FILT = "filter"


def compute_frame_metrics(img: np.ndarray, fov: np.ndarray) -> dict:
    """Return brightness, specular ratio, Laplacian variance, all evaluated within FOV."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = fov > 0
    if m.sum() == 0:
        return dict(brightness=0, specular=0, lap_var=0)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    spec = (gray > 220) & ((mx - mn) < 25) & m
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return dict(
        brightness=float(gray[m].mean()),
        specular=float(spec.sum() / m.sum()),
        lap_var=float(lap[m].var()),
    )


def run_filter(frames_dir: Path, fov_path: Path, out_dir: Path,
               max_brightness: float = 140.0,
               max_specular: float = 0.05,
               min_brightness: float = 50.0) -> None:
    out_dir = ensure_dir(out_dir)
    frames = list_frames(Path(frames_dir))
    fov = cv2.imread(str(fov_path), cv2.IMREAD_GRAYSCALE)
    if fov is None:
        raise FileNotFoundError(fov_path)

    log(STAGE_FILT, f"filtering {len(frames)} frames")
    log(STAGE_FILT, f"thresholds: {min_brightness} ≤ brightness ≤ {max_brightness}, specular ≤ {max_specular*100:.1f}%")

    metrics_csv = out_dir / "frame_metrics.csv"
    good_txt = out_dir / "good_frames.txt"

    with open(metrics_csv, "w", newline="") as fc, open(good_txt, "w") as fg:
        writer = csv.writer(fc)
        writer.writerow(["frame_idx", "filename", "brightness", "specular", "lap_var", "is_good"])

        n_good = 0
        n_bad_bright = 0
        n_bad_spec = 0
        n_bad_dim = 0

        for i, f in enumerate(progress(frames, total=len(frames), desc="filter")):
            img = cv2.imread(str(f))
            m = compute_frame_metrics(img, fov)
            is_good = True
            if m["brightness"] > max_brightness:
                is_good = False
                n_bad_bright += 1
            elif m["specular"] > max_specular:
                is_good = False
                n_bad_spec += 1
            elif m["brightness"] < min_brightness:
                is_good = False
                n_bad_dim += 1

            writer.writerow([i, f.name, f"{m['brightness']:.2f}", f"{m['specular']:.4f}",
                             f"{m['lap_var']:.1f}", int(is_good)])
            if is_good:
                fg.write(f"{i:06d}\n")
                n_good += 1

    log(STAGE_FILT, f"good frames: {n_good}/{len(frames)} ({100*n_good/len(frames):.1f}%)", level="ok")
    log(STAGE_FILT, f"discarded — washout: {n_bad_bright}, specular: {n_bad_spec}, dim: {n_bad_dim}")
    log(STAGE_FILT, f"metrics → {metrics_csv}", level="ok")
    log(STAGE_FILT, f"good list → {good_txt}", level="ok")


# =========================================================================
# Stage 5 — Combine masks (FOV ∩ ¬Tool, restricted to good frames)
# =========================================================================
STAGE_COMB = "combine-masks"


def run_combine(frames_dir: Path, fov_path: Path, tool_masks_dir: Path,
                good_frames_txt: Path, out_dir: Path) -> None:
    out_dir = ensure_dir(out_dir)
    frames = list_frames(Path(frames_dir))
    fov = cv2.imread(str(fov_path), cv2.IMREAD_GRAYSCALE)
    if fov is None:
        raise FileNotFoundError(fov_path)

    good = set()
    if good_frames_txt and Path(good_frames_txt).exists():
        with open(good_frames_txt) as f:
            good = {int(line.strip()) for line in f if line.strip()}
        log(STAGE_COMB, f"loaded {len(good)} good frame indices")
    else:
        good = set(range(len(frames)))
        log(STAGE_COMB, "no good_frames.txt — using all frames", level="warn")

    tool_masks_dir = Path(tool_masks_dir) if tool_masks_dir else None
    if tool_masks_dir and not tool_masks_dir.exists():
        log(STAGE_COMB, f"tool_masks dir {tool_masks_dir} not found — using only FOV mask", level="warn")
        tool_masks_dir = None

    n_written = 0
    for i, f in enumerate(progress(frames, total=len(frames), desc="combine")):
        if i not in good:
            continue
        m = fov.copy()
        if tool_masks_dir:
            tm_path = tool_masks_dir / f"{i:06d}.png"
            if tm_path.exists():
                tm = cv2.imread(str(tm_path), cv2.IMREAD_GRAYSCALE)
                if tm is not None and tm.shape == m.shape:
                    m = cv2.bitwise_and(m, cv2.bitwise_not(tm))
        cv2.imwrite(str(out_dir / f"{i:06d}.png"), m)
        n_written += 1

    log(STAGE_COMB, f"wrote {n_written} final masks → {out_dir}", level="ok")
