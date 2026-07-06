#!/usr/bin/env python3
"""Endoscopic 3D reconstruction pipeline — main entry point.

Run any single stage from terminal. Each stage prints timestamped progress.

Quick start
-----------
  python run.py check
  python run.py extract --video /path/to/lung_inner_76s.mov --out ./out
  python run.py fovmask --out ./out
  python run.py filter  --out ./out
  python run.py sam2-prompt --out ./out
  python run.py sam2-propagate --out ./out --ckpt sam2.1_hiera_large.pt   # Windows + CUDA
  python run.py combine-masks --out ./out
  python run.py calib --out ./out
  python run.py droid --out ./out                                          # Windows + CUDA
  python run.py viz --out ./out
  python run.py viz-sync --out ./out                                       # interactive, ←→ to step
  python run.py viz-live --out ./out                                       # Open3D animated replay (official style)

Convenience
-----------
  python run.py all-mac     # runs everything that works on Mac
  python run.py all-cuda    # runs the CUDA-only stages (assumes Mac prep done)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline import calib, droid_runner, extract, masks, preprocess, sam2_runner, viz, viz_live, viz_sync
from pipeline.common import bold, cyan, log, probe_env, section


# ----------------------------------------------------------------------------
# Path conventions inside out_dir
# ----------------------------------------------------------------------------
def paths(out: Path) -> dict:
    out = Path(out)
    return {
        "out":            out,
        "frames":         out / "frames",
        "fov_mask":       out / "fov_mask.png",
        "metadata":       out / "metadata.json",
        "metrics":        out / "frame_metrics.csv",
        "good":           out / "good_frames.txt",
        "prompts":        out / "prompts.json",
        "tool_masks":     out / "tool_masks",
        "final_masks":    out / "final_masks",
        "calib_json":     out / "calib.json",
        "calib_txt":      out / "calib.txt",
        "droid":          out / "droid",
        "viz":            out / "viz",
    }


# ----------------------------------------------------------------------------
# Subcommand implementations
# ----------------------------------------------------------------------------
def cmd_check(args):
    section("Environment check")
    probe_env()


def cmd_extract(args):
    section("Stage 1 — Extract frames")
    p = paths(args.out)
    extract.run(Path(args.video), p["out"], target_width=args.width, force=args.force)


def cmd_preprocess(args):
    """Stage 1+2 combined: extract + FOV crop + distortion trim + calib.txt.

    Produces DROID-SLAM-ready layout: out/{frames, calib.txt, preview.png}.
    """
    section("Preprocess — video → cropped frames + calib.txt (DROID-SLAM ready)")
    p = paths(args.out)
    fps = args.fps if args.fps and args.fps > 0 else None
    preprocess.run(
        video=Path(args.video),
        out_dir=p["out"],
        target_width=args.width,
        target_fps=fps,
        distortion_margin=args.distortion_margin,
        assumed_fov_deg=args.assumed_fov_deg,
        fov_threshold=args.fov_threshold,
        keep_raw=args.keep_raw,
    )


def cmd_fovmask(args):
    section("Stage 2 — Circular FOV mask")
    p = paths(args.out)
    masks.run_fovmask(p["frames"], p["out"])


def cmd_filter(args):
    section("Stage 3 — Bad-frame filter")
    p = paths(args.out)
    masks.run_filter(
        p["frames"], p["fov_mask"], p["out"],
        max_brightness=args.max_brightness,
        max_specular=args.max_specular,
        min_brightness=args.min_brightness,
    )


def cmd_sam2_prompt(args):
    section("Stage 4a — SAM 2 prompt selector")
    p = paths(args.out)
    key_seconds = [float(x) for x in args.key_seconds.split(",")] if args.key_seconds else (8, 12, 30, 44)
    sam2_runner.run_prompt(p["frames"], p["prompts"], key_seconds=key_seconds)


def cmd_sam2_propagate(args):
    section("Stage 4b — SAM 2 video propagation")
    p = paths(args.out)
    sam2_runner.run_propagate(
        p["frames"], p["prompts"], p["tool_masks"],
        ckpt=args.ckpt,
        model_cfg=args.model_cfg,
    )


def cmd_combine_masks(args):
    section("Stage 5 — Combine FOV + tool masks")
    p = paths(args.out)
    masks.run_combine(p["frames"], p["fov_mask"], p["tool_masks"], p["good"], p["final_masks"])


def cmd_calib(args):
    section("Stage 6 — Camera intrinsics")
    p = paths(args.out)
    calib.run(p["frames"], p["out"], method=args.method, fov_deg=args.fov_deg)


def cmd_droid(args):
    section("Stage 7 — DROID-SLAM")
    p = paths(args.out)
    droid_runner.run(
        frames_dir=p["frames"],
        calib_txt=p["calib_txt"],
        out_dir=p["droid"],
        masks_dir=p["final_masks"] if args.use_masks else None,
        droid_root=args.droid_root,
        weights=args.weights,
        stride=args.stride,
        filter_thresh=getattr(args, "filter_thresh", None),
        keyframe_thresh=getattr(args, "keyframe_thresh", None),
    )


def cmd_viz(args):
    section("Stage 8 — Visualize")
    p = paths(args.out)
    viz.run(p["droid"], p["viz"], show=not args.no_show)


def cmd_viz_sync(args):
    section("Stage 9 — Synchronized video + trajectory viewer")
    p = paths(args.out)
    viz_sync.run(p["frames"], p["droid"], fps=args.fps)


def cmd_viz_live(args):
    section("Stage 9b — Open3D live-animated reconstruction replay")
    p = paths(args.out)
    viz_live.run(
        p["droid"], frames_dir=None if args.no_frames else p["frames"],
        droid_root=args.droid_root, fps=args.fps,
        filter_thresh=args.filter_thresh, filter_count=args.filter_count,
        cam_scale=args.cam_scale,
    )


# Convenience compounds
def cmd_all_mac(args):
    """Run all Mac-friendly stages in sequence."""
    section("ALL-MAC: extract → fovmask → filter → sam2-prompt → calib")
    cmd_extract(args)
    cmd_fovmask(args)
    cmd_filter(args)
    cmd_sam2_prompt(args)
    cmd_calib(args)
    log("all-mac", "Mac stages complete. Now move out/ to Windows+3080 and run all-cuda.", level="ok")


def cmd_all_cuda(args):
    """Run CUDA-required stages."""
    section("ALL-CUDA: sam2-propagate → combine-masks → droid → viz")
    cmd_sam2_propagate(args)
    cmd_combine_masks(args)
    cmd_droid(args)
    cmd_viz(args)


def cmd_all_droid(args):
    """Direct DROID-SLAM path — bypass filter/sam2/combine-masks entirely.

    preprocess → droid (no masks, official demo.py args) → viz.
    Recommended path for endoscope videos: DROID-SLAM does its own keyframe
    selection internally, so pre-filtering by brightness/specular is not needed.
    """
    section("ALL-DROID: preprocess → droid → viz  (direct, no filter / no sam2)")
    cmd_preprocess(args)
    args.use_masks = False  # ensure official demo.py compatibility (no --mask_dir)
    cmd_droid(args)
    cmd_viz(args)


# ----------------------------------------------------------------------------
# Argument parser
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    P = argparse.ArgumentParser(
        prog="run.py",
        description="Endoscopic 3D reconstruction pipeline (trajectory + sparse points).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Common arg shared by every subcommand via parents=
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="./out", help="output root directory (default: ./out)")

    sub = P.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", parents=[common],
                   help="probe environment (Python, CUDA, ffmpeg, packages)").set_defaults(func=cmd_check)

    s = sub.add_parser("extract", parents=[common], help="Stage 1: ffmpeg extract frames, downsample width")
    s.add_argument("--video", required=True, help="path to source .mov / .mp4")
    s.add_argument("--width", type=int, default=1280, help="target frame width (default 1280)")
    s.add_argument("--force", action="store_true", help="re-extract even if frames exist")
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("preprocess", parents=[common],
                       help="Combined extract + FOV crop + distortion trim + calib.txt (DROID-SLAM ready)")
    s.add_argument("--video", required=True, help="path to source video")
    s.add_argument("--width", type=int, default=1280, help="target frame width (default 1280)")
    s.add_argument("--fps", type=float, default=15.0,
                   help="target fps after ffmpeg fps filter (default 15; use 0 to keep source fps)")
    s.add_argument("--distortion-margin", type=float, default=0.15,
                   help="shrink FOV radius by this fraction (default 0.15)")
    s.add_argument("--assumed-fov-deg", type=float, default=90.0,
                   help="assumed horizontal FOV of the full circular view (used only for rough calib)")
    s.add_argument("--fov-threshold", type=int, default=12,
                   help="pixel intensity threshold for FOV detection")
    s.add_argument("--keep-raw", action="store_true", help="keep intermediate raw_frames/ folder")
    s.set_defaults(func=cmd_preprocess)

    s = sub.add_parser("fovmask", parents=[common], help="Stage 2: circular FOV mask from first frame")
    s.set_defaults(func=cmd_fovmask)

    s = sub.add_parser("filter", parents=[common], help="Stage 3: bad-frame filter (washout / specular)")
    s.add_argument("--max-brightness", type=float, default=140.0)
    s.add_argument("--min-brightness", type=float, default=50.0)
    s.add_argument("--max-specular", type=float, default=0.05, help="0.05 = 5%%")
    s.set_defaults(func=cmd_filter)

    s = sub.add_parser("sam2-prompt", parents=[common], help="Stage 4a: interactive prompt picker (matplotlib GUI)")
    s.add_argument("--key-seconds", default="8,12,30,44",
                   help="comma-separated seconds at which to prompt (default: 8,12,30,44)")
    s.set_defaults(func=cmd_sam2_prompt)

    s = sub.add_parser("sam2-propagate", parents=[common], help="Stage 4b: SAM 2 video propagation (CUDA strongly recommended)")
    s.add_argument("--ckpt", default="sam2.1_hiera_large.pt", help="SAM 2 checkpoint .pt path")
    s.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    s.set_defaults(func=cmd_sam2_propagate)

    s = sub.add_parser("combine-masks", parents=[common], help="Stage 5: FOV ∩ ¬Tool, restricted to good frames")
    s.set_defaults(func=cmd_combine_masks)

    s = sub.add_parser("calib", parents=[common], help="Stage 6: camera intrinsics estimation")
    s.add_argument("--method", choices=["heuristic", "colmap"], default="heuristic")
    s.add_argument("--fov-deg", type=float, default=70.0, help="horizontal FOV in degrees (heuristic)")
    s.set_defaults(func=cmd_calib)

    s = sub.add_parser("droid", parents=[common], help="Stage 7: DROID-SLAM (CUDA required)")
    s.add_argument("--droid-root", default=None, help="DROID-SLAM repo path (or $DROID_SLAM_ROOT)")
    s.add_argument("--weights", default=None, help="droid.pth weights path")
    s.add_argument("--stride", type=int, default=1)
    s.add_argument("--filter-thresh", type=float, default=None,
                   help="motion threshold to even consider a frame for tracking (demo.py default 2.4; "
                        "lower keeps more candidate frames)")
    s.add_argument("--keyframe-thresh", type=float, default=None,
                   help="motion threshold to keep a tracked frame as a permanent keyframe "
                        "(demo.py default 4.0; lower keeps more keyframes)")
    # Official DROID-SLAM demo.py does NOT accept --mask_dir. Default is off for
    # full compatibility; only enable if you point --droid-root at a fork.
    s.add_argument("--use-masks", action="store_true", default=False,
                   help="pass --mask_dir to DROID-SLAM (requires a fork that supports it; off by default)")
    s.add_argument("--no-masks", dest="use_masks", action="store_false")
    s.set_defaults(func=cmd_droid)

    s = sub.add_parser("viz", parents=[common], help="Stage 8: visualize trajectory + sparse points")
    s.add_argument("--no-show", action="store_true", help="do not open Open3D viewer")
    s.set_defaults(func=cmd_viz)

    s = sub.add_parser("viz-sync", parents=[common],
                        help="Stage 9: synchronized video + 3D trajectory viewer (←→ to step)")
    s.add_argument("--fps", type=float, default=5.0, help="auto-play speed (keyframes/sec)")
    s.set_defaults(func=cmd_viz_sync)

    s = sub.add_parser("viz-live", parents=[common],
                        help="Stage 9b: Open3D live-animated replay (growing point cloud + moving camera, official style)")
    s.add_argument("--droid-root", default=None, help="DROID-SLAM repo path (or $DROID_SLAM_ROOT)")
    s.add_argument("--fps", type=float, default=6.0, help="keyframe reveal rate (keyframes/sec)")
    s.add_argument("--filter-thresh", type=float, default=0.005, help="depth-consistency filter threshold")
    s.add_argument("--filter-count", type=int, default=2, help="min agreeing views to keep a point")
    s.add_argument("--cam-scale", type=float, default=0.05, help="camera frustum wireframe size")
    s.add_argument("--no-frames", action="store_true",
                   help="don't open the synchronized real-frame window")
    s.set_defaults(func=cmd_viz_live)

    s = sub.add_parser("all-mac", parents=[common], help="Run all Mac-friendly stages (extract → calib)")
    s.add_argument("--video", required=True)
    s.add_argument("--width", type=int, default=1280)
    s.add_argument("--force", action="store_true")
    s.add_argument("--max-brightness", type=float, default=140.0)
    s.add_argument("--min-brightness", type=float, default=50.0)
    s.add_argument("--max-specular", type=float, default=0.05)
    s.add_argument("--key-seconds", default="8,12,30,44")
    s.add_argument("--method", choices=["heuristic", "colmap"], default="heuristic")
    s.add_argument("--fov-deg", type=float, default=70.0)
    s.set_defaults(func=cmd_all_mac)

    s = sub.add_parser("all-cuda", parents=[common], help="Run CUDA-required stages (sam2-propagate → viz)")
    s.add_argument("--ckpt", default="sam2.1_hiera_large.pt")
    s.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    s.add_argument("--droid-root", default=None)
    s.add_argument("--weights", default=None)
    s.add_argument("--stride", type=int, default=1)
    s.add_argument("--use-masks", action="store_true", default=False)
    s.add_argument("--no-masks", dest="use_masks", action="store_false")
    s.add_argument("--no-show", action="store_true", default=True)
    s.set_defaults(func=cmd_all_cuda)

    s = sub.add_parser("all-droid", parents=[common],
                       help="Direct DROID-SLAM path: preprocess → droid → viz (skip filter/sam2)")
    s.add_argument("--video", required=True, help="path to source video")
    s.add_argument("--width", type=int, default=1280)
    s.add_argument("--fps", type=float, default=15.0)
    s.add_argument("--distortion-margin", type=float, default=0.15)
    s.add_argument("--assumed-fov-deg", type=float, default=90.0)
    s.add_argument("--fov-threshold", type=int, default=12)
    s.add_argument("--keep-raw", action="store_true")
    s.add_argument("--droid-root", default=None, help="DROID-SLAM repo path (or $DROID_SLAM_ROOT)")
    s.add_argument("--weights", default=None, help="droid.pth path (default: $DROID_SLAM_ROOT/droid.pth)")
    s.add_argument("--stride", type=int, default=1)
    s.add_argument("--no-show", action="store_true", default=True)
    s.set_defaults(func=cmd_all_droid)

    return P


def main():
    parser = build_parser()
    args = parser.parse_args()
    print(cyan(bold("\n┌─ Endoscopic 3D pipeline ─────────────────────────────────────┐")))
    print(cyan(bold(f"│ cmd = {args.cmd:<10}  out = {str(args.out):<40} │")))
    print(cyan(bold("└──────────────────────────────────────────────────────────────┘")))
    try:
        args.func(args)
    except KeyboardInterrupt:
        log("main", "interrupted by user", level="warn")
        sys.exit(130)


if __name__ == "__main__":
    main()
