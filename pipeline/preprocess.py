"""Stage 1+2 combined — endoscope-friendly preprocessing for DROID-SLAM.

Produces a directory layout that DROID-SLAM's official demo.py can consume
directly, with no --mask_dir needed:

    <out_dir>/
      frames/           cropped rectangular PNGs (no black border, no distortion edge)
      calib.txt         "fx fy cx cy" one-line file (DROID-SLAM format)
      preview.png       overlay debug image
      metadata.json     provenance

Steps:
  1) ffmpeg extract  (video -> raw PNG frames, optionally resized/fps-limited)
  2) Robust FOV detection  (median circle over 30 sample frames)
  3) Shrink FOV radius by `distortion_margin` to drop peripheral distortion
  4) Crop each frame to the maximal square inscribed in the shrunk circle
  5) Emit calib.txt from the raw circle radius + assumed full-circle FOV

The calibration produced here is a rough default. Replace `calib.txt` with
a real calibration when the vendor / chessboard values are available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .common import ensure_dir, log, timed

STAGE = "preprocess"


# -----------------------------------------------------------------------------
# ffmpeg extraction
# -----------------------------------------------------------------------------

def probe_video(video: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "json", str(video),
    ]
    info = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)["streams"][0]
    num, den = (int(x) for x in info["r_frame_rate"].split("/"))
    info["fps"] = num / den if den else float(num)
    return info


def extract_frames(video: Path, out_dir: Path, target_width: int,
                   target_fps: float | None) -> tuple[int, int]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")

    info = probe_video(video)
    sw, sh = int(info["width"]), int(info["height"])
    src_fps = float(info["fps"])
    scale = target_width / sw
    out_h = int(round(sh * scale / 2) * 2)
    log(STAGE, f"src {sw}x{sh}@{src_fps:.2f}fps  ->  {target_width}x{out_h}")

    vf = f"scale={target_width}:{out_h}"
    if target_fps and target_fps < src_fps:
        vf += f",fps={target_fps}"
        log(STAGE, f"downsampling fps -> {target_fps}")

    cmd = [
        "ffmpeg", "-y", "-i", str(video), "-vf", vf,
        "-vsync", "0", "-q:v", "2", "-start_number", "0",
        str(out_dir / "%06d.png"),
        "-hide_banner", "-loglevel", "warning", "-stats",
    ]
    subprocess.run(cmd, check=True)
    n = len(list(out_dir.glob("*.png")))
    log(STAGE, f"wrote {n} frames -> {out_dir}", level="ok")
    return target_width, out_h


# -----------------------------------------------------------------------------
# Robust circular FOV detection
# -----------------------------------------------------------------------------

def detect_fov_circle(frame_bgr: np.ndarray, threshold: int = 12) -> tuple[int, int, int] | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.where(labels == keep)
    if xs.size < 50:
        return None
    (cx, cy), r = cv2.minEnclosingCircle(np.stack([xs, ys], axis=1).astype(np.float32))
    return int(round(cx)), int(round(cy)), int(round(r))


def robust_fov(frames: list[Path], n_samples: int = 30, threshold: int = 12) -> tuple[int, int, int]:
    idxs = np.linspace(0, len(frames) - 1, min(n_samples, len(frames))).astype(int)
    cxs, cys, rs = [], [], []
    for i in idxs:
        img = cv2.imread(str(frames[i]))
        if img is None:
            continue
        result = detect_fov_circle(img, threshold)
        if result is None:
            continue
        cxs.append(result[0])
        cys.append(result[1])
        rs.append(result[2])
    if not rs:
        raise RuntimeError(
            "failed to detect circular FOV in any sample frame — try adjusting --fov-threshold"
        )
    cx = int(np.median(cxs))
    cy = int(np.median(cys))
    r = int(np.median(rs))
    log(STAGE, f"FOV detected from {len(rs)} samples: center=({cx},{cy}) radius={r}", level="ok")
    return cx, cy, r


# -----------------------------------------------------------------------------
# Crop + calib + preview
# -----------------------------------------------------------------------------

def inscribed_square(cx: int, cy: int, r: int) -> tuple[int, int, int, int]:
    half = int(r / np.sqrt(2))
    return cx - half, cy - half, cx + half, cy + half


def crop_all(raw_dir: Path, out_dir: Path, box: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = box
    frames = sorted(raw_dir.glob("*.png"))
    n = 0
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(w, x1), min(h, y1)
        cv2.imwrite(str(out_dir / f.name), img[ya:yb, xa:xb])
        n += 1
    log(STAGE, f"cropped {n} frames -> {out_dir}  size={x1-x0}x{y1-y0}", level="ok")
    return n


def write_calib(out_path: Path, W: int, H: int, r_outer: int, assumed_fov_deg: float) -> None:
    """DROID-SLAM calib.txt line: 'fx fy cx cy'.

    Focal length derived from the *raw* circle radius (which spans assumed FOV).
    Focal length is a sensor property; cropping to an inscribed square does
    NOT change fx/fy, only cx/cy.
    """
    fov_rad = np.deg2rad(assumed_fov_deg)
    fx = r_outer / np.tan(fov_rad / 2.0)
    fy = fx
    cx, cy = W / 2.0, H / 2.0
    out_path.write_text(f"{fx:.4f} {fy:.4f} {cx:.4f} {cy:.4f}\n")
    log(STAGE, f"assumed FOV={assumed_fov_deg}° -> fx=fy={fx:.2f}  cx={cx:.1f}  cy={cy:.1f}")
    log(STAGE, f"calib.txt -> {out_path}", level="ok")


def save_preview(sample_bgr: np.ndarray, cx: int, cy: int,
                 r_outer: int, r_inner: int, box: tuple[int, int, int, int],
                 out_path: Path) -> None:
    vis = sample_bgr.copy()
    cv2.circle(vis, (cx, cy), r_outer, (0, 255, 255), 2)   # yellow: raw FOV
    cv2.circle(vis, (cx, cy), r_inner, (0, 0, 255), 2)     # red:    after distortion trim
    x0, y0, x1, y1 = box
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2) # green:  final crop
    cv2.imwrite(str(out_path), vis)
    log(STAGE, f"preview -> {out_path}", level="ok")


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def run(video: Path, out_dir: Path,
        target_width: int = 1280,
        target_fps: float | None = 15.0,
        distortion_margin: float = 0.15,
        assumed_fov_deg: float = 90.0,
        fov_threshold: int = 12,
        keep_raw: bool = False) -> None:
    """End-to-end preprocess. Produces <out_dir>/{frames, calib.txt, preview.png}."""
    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(video)

    out_dir = ensure_dir(out_dir)
    raw_dir = ensure_dir(out_dir / "raw_frames")
    crop_dir = ensure_dir(out_dir / "frames")

    # 1) extract
    with timed(STAGE, "ffmpeg extract"):
        W, H = extract_frames(video, raw_dir, target_width, target_fps)

    # 2) detect FOV
    frames = sorted(raw_dir.glob("*.png"))
    if not frames:
        raise RuntimeError("no frames extracted from video")
    with timed(STAGE, "FOV detection"):
        cx, cy, r_outer = robust_fov(frames, n_samples=30, threshold=fov_threshold)

    # 3) shrink circle
    r_inner = int(r_outer * (1.0 - distortion_margin))
    log(STAGE, f"distortion_margin={distortion_margin} -> r_inner={r_inner}")

    # 4) inscribed square
    box = inscribed_square(cx, cy, r_inner)
    box = (max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3]))
    Wc, Hc = box[2] - box[0], box[3] - box[1]

    # preview from a mid-video frame
    mid = frames[len(frames) // 2]
    save_preview(cv2.imread(str(mid)), cx, cy, r_outer, r_inner, box,
                 out_dir / "preview.png")

    # crop all
    with timed(STAGE, "crop all frames"):
        n = crop_all(raw_dir, crop_dir, box)

    # 5) calib
    write_calib(out_dir / "calib.txt", Wc, Hc, r_outer, assumed_fov_deg)

    # metadata
    meta = {
        "video": str(video),
        "extracted_frames": n,
        "raw_size": [W, H],
        "cropped_size": [Wc, Hc],
        "fov_center": [cx, cy],
        "fov_radius_outer": r_outer,
        "fov_radius_inner": r_inner,
        "distortion_margin": distortion_margin,
        "assumed_fov_deg": assumed_fov_deg,
        "notes": "calib.txt is a rough estimate; replace with real calibration when available.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    log(STAGE, f"metadata -> {out_dir/'metadata.json'}", level="ok")

    if not keep_raw:
        shutil.rmtree(raw_dir)
        log(STAGE, f"cleanup {raw_dir}")
