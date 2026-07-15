#!/usr/bin/env python3
"""Extract a frame range from a source video into a DROID-SLAM-ready folder.

Same output layout as `pipeline/preprocess.py` (frames/, calib.txt, preview.png,
metadata.json), but restricted to a [start, end] frame range of the source
video instead of the whole thing -- useful for testing a specific segment
without re-running preprocess on the full clip.

Usage:
  python split_video.py --video data/A129315.mp4 --start 1000 --end 1500 --out ./out/A129315_1000-1500
  python run.py droid --out ./out/A129315_1000-1500
  python run.py viz-live --out ./out/A129315_1000-1500
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2

from pipeline.common import ensure_dir, log, timed
from pipeline.preprocess import (
    STAGE, probe_video, robust_fov, inscribed_square, crop_all, write_calib, save_preview,
    DEFAULT_INTRINSICS_PATH, load_real_intrinsics, real_calib_line,
)


def extract_frame_range(video: Path, out_dir: Path, start: int, end: int,
                        target_width: int, target_fps: float | None) -> tuple[int, int]:
    info = probe_video(video)
    sw, sh = int(info["width"]), int(info["height"])
    src_fps = float(info["fps"])
    scale = target_width / sw
    out_h = int(round(sh * scale / 2) * 2)
    log(STAGE, f"src {sw}x{sh}@{src_fps:.2f}fps  ->  {target_width}x{out_h}, frames [{start},{end}]")

    vf = f"select=between(n\\,{start}\\,{end}),scale={target_width}:{out_h}"
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


def run(video: Path, out_dir: Path, start: int, end: int,
        target_width: int = 1280,
        target_fps: float | None = None,
        distortion_margin: float = 0.15,
        assumed_fov_deg: float = 90.0,
        fov_threshold: int = 12,
        keep_raw: bool = False,
        intrinsics_path: Path | None = DEFAULT_INTRINSICS_PATH,
        calib_model: str = "pinhole_4param") -> None:
    """End-to-end: video[start:end] -> <out_dir>/{frames, calib.txt, preview.png}.

    calib.txt uses the real checkerboard calibration (intrinsics_path, default
    data/intrinsics.json) rescaled to this clip's actual resolution/crop
    whenever the source video's native resolution matches the calibration's
    own; otherwise falls back to the FOV-heuristic estimate.
    """
    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(video)
    if end < start:
        raise ValueError(f"--end ({end}) must be >= --start ({start})")

    out_dir = ensure_dir(out_dir)
    raw_dir = ensure_dir(out_dir / "raw_frames")
    crop_dir = ensure_dir(out_dir / "frames")

    src_info = probe_video(video)
    real_intr = None
    if intrinsics_path and Path(intrinsics_path).exists():
        try:
            real_intr = load_real_intrinsics(intrinsics_path, calib_model)
        except Exception as e:
            log(STAGE, f"could not load {intrinsics_path}: {e}", level="warn")
        else:
            if (int(src_info["width"]), int(src_info["height"])) != (real_intr["W0"], real_intr["H0"]):
                log(STAGE,
                    f"source video {int(src_info['width'])}x{int(src_info['height'])} != "
                    f"calibration {real_intr['W0']}x{real_intr['H0']} -- falling back to "
                    f"FOV-heuristic calib.txt for this clip", level="warn")
                real_intr = None

    with timed(STAGE, f"ffmpeg extract frames [{start},{end}]"):
        W, H = extract_frame_range(video, raw_dir, start, end, target_width, target_fps)

    frames = sorted(raw_dir.glob("*.png"))
    if not frames:
        raise RuntimeError(
            f"no frames extracted for range [{start},{end}] -- check --start/--end "
            f"against the source video's frame count"
        )

    with timed(STAGE, "FOV detection"):
        cx, cy, r_outer = robust_fov(frames, n_samples=min(30, len(frames)), threshold=fov_threshold)

    r_inner = int(r_outer * (1.0 - distortion_margin))
    log(STAGE, f"distortion_margin={distortion_margin} -> r_inner={r_inner}")

    box = inscribed_square(cx, cy, r_inner)
    box = (max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3]))
    Wc, Hc = box[2] - box[0], box[3] - box[1]

    mid = frames[len(frames) // 2]
    save_preview(cv2.imread(str(mid)), cx, cy, r_outer, r_inner, box, out_dir / "preview.png")

    with timed(STAGE, "crop all frames"):
        n = crop_all(raw_dir, crop_dir, box)

    if real_intr is not None:
        (out_dir / "calib.txt").write_text(real_calib_line(real_intr, W, box))
        log(STAGE, f"calib.txt <- REAL calibration ({calib_model}) from {intrinsics_path}", level="ok")
    else:
        write_calib(out_dir / "calib.txt", Wc, Hc, r_outer, assumed_fov_deg)

    meta = {
        "video": str(video),
        "start_frame": start,
        "end_frame": end,
        "extracted_frames": n,
        "raw_size": [W, H],
        "cropped_size": [Wc, Hc],
        "fov_center": [cx, cy],
        "fov_radius_outer": r_outer,
        "fov_radius_inner": r_inner,
        "distortion_margin": distortion_margin,
        "assumed_fov_deg": assumed_fov_deg,
        "calib_source": f"real:{calib_model}:{intrinsics_path}" if real_intr is not None else "heuristic",
        "notes": ("calib.txt is a rough estimate; replace with real calibration when available."
                  if real_intr is None else
                  "calib.txt uses the real checkerboard calibration, rescaled to this crop."),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    log(STAGE, f"metadata -> {out_dir/'metadata.json'}", level="ok")

    if not keep_raw:
        shutil.rmtree(raw_dir)
        log(STAGE, f"cleanup {raw_dir}")


def main():
    p = argparse.ArgumentParser(
        description="Extract a [start,end] frame range from a video into a DROID-SLAM-ready folder.")
    p.add_argument("--video", required=True, help="path to source video")
    p.add_argument("--start", type=int, required=True, help="start frame index (inclusive, 0-based)")
    p.add_argument("--end", type=int, required=True, help="end frame index (inclusive)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--width", type=int, default=1280, help="target frame width (default 1280)")
    p.add_argument("--fps", type=float, default=0,
                   help="target fps after extraction (default 0 = keep source fps)")
    p.add_argument("--distortion-margin", type=float, default=0.15)
    p.add_argument("--assumed-fov-deg", type=float, default=90.0)
    p.add_argument("--fov-threshold", type=int, default=12)
    p.add_argument("--keep-raw", action="store_true", help="keep intermediate raw_frames/ folder")
    p.add_argument("--intrinsics", default=str(DEFAULT_INTRINSICS_PATH),
                   help="real checkerboard calibration JSON (default: data/intrinsics.json); "
                        "used automatically when the source video's resolution matches it, "
                        "otherwise falls back to the FOV-heuristic estimate")
    p.add_argument("--calib-model", choices=["pinhole_4param", "pinhole_5param"], default="pinhole_4param")
    p.add_argument("--no-real-calib", action="store_true",
                   help="always use the FOV-heuristic calib.txt, even if a matching calibration exists")
    args = p.parse_args()

    run(
        Path(args.video), Path(args.out), args.start, args.end,
        target_width=args.width,
        target_fps=(args.fps if args.fps and args.fps > 0 else None),
        distortion_margin=args.distortion_margin,
        assumed_fov_deg=args.assumed_fov_deg,
        fov_threshold=args.fov_threshold,
        keep_raw=args.keep_raw,
        intrinsics_path=(None if args.no_real_calib else Path(args.intrinsics)),
        calib_model=args.calib_model,
    )

    print(f"\nNext steps:")
    print(f"  python run.py droid --out {args.out}")
    print(f"  python run.py viz-live --out {args.out}")


if __name__ == "__main__":
    main()
