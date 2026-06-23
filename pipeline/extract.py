"""Stage 1 — extract frames from video, optionally downsample width."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .common import ensure_dir, log, run_cmd, timed

STAGE = "extract"


def probe_video(video: Path) -> dict:
    """Run ffprobe and return key stream/format info."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name",
        "-of", "json",
        str(video),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    info = json.loads(out)["streams"][0]
    # Parse frame rate fraction
    num, den = (int(x) for x in info["r_frame_rate"].split("/"))
    info["fps"] = num / den if den else float(num)
    return info


def run(video: Path, out_dir: Path, target_width: int = 1280, force: bool = False) -> None:
    """Extract video frames into out_dir/frames/ as PNG, downsampled to target_width."""
    video = Path(video)
    out_dir = ensure_dir(out_dir)
    frames_dir = ensure_dir(out_dir / "frames")
    meta_path = out_dir / "metadata.json"

    if not video.exists():
        raise FileNotFoundError(video)

    # Probe
    with timed(STAGE, "Probing video"):
        info = probe_video(video)
    log(STAGE, f"input: {video.name}  size={info['width']}x{info['height']}  fps={info['fps']:.2f}  codec={info['codec_name']}")
    if "nb_frames" in info:
        log(STAGE, f"declared nb_frames = {info['nb_frames']}, duration = {info.get('duration', '?')}s")

    # Skip if frames already exist and not forced
    existing = list(frames_dir.glob("*.png"))
    if existing and not force:
        log(STAGE, f"{len(existing)} frames already exist in {frames_dir} (use --force to redo)", level="warn")
        if meta_path.exists():
            return
    else:
        # Compute scale: keep width = target_width, height auto, keep even
        sw, sh = info["width"], info["height"]
        scale_factor = target_width / sw
        new_h = int(round(sh * scale_factor / 2) * 2)
        scale_filter = f"scale={target_width}:{new_h}"
        log(STAGE, f"downsampling to {target_width}x{new_h} (factor {scale_factor:.3f})")

        # ffmpeg extract
        cmd = [
            "ffmpeg", "-y", "-i", str(video),
            "-vf", scale_filter,
            "-vsync", "0",
            "-q:v", "2",
            "-start_number", "0",
            str(frames_dir / "%06d.png"),
            "-hide_banner",
            "-loglevel", "warning",
            "-stats",
        ]
        with timed(STAGE, "Extracting frames with ffmpeg"):
            rc = run_cmd(cmd, stage=STAGE)
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed (rc={rc})")
        existing = list(frames_dir.glob("*.png"))
        log(STAGE, f"extracted {len(existing)} frames → {frames_dir}", level="ok")

    # Write metadata.json
    meta = {
        "source_video": str(video),
        "original_width": info["width"],
        "original_height": info["height"],
        "fps": info["fps"],
        "frame_count": len(existing),
        "extracted_width": target_width,
        "scale_factor": target_width / info["width"],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log(STAGE, f"metadata → {meta_path}", level="ok")
