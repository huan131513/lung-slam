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

Convenience
-----------
  python run.py all-mac     # runs everything that works on Mac
  python run.py all-cuda    # runs the CUDA-only stages (assumes Mac prep done)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline import calib, droid_runner, extract, masks, sam2_runner, viz
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
    )


def cmd_viz(args):
    section("Stage 8 — Visualize")
    p = paths(args.out)
    viz.run(p["droid"], p["viz"], show=not args.no_show)


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
    s.add_argument("--use-masks", action="store_true", default=True,
                   help="pass --mask_dir to DROID-SLAM (requires fork that supports it)")
    s.add_argument("--no-masks", dest="use_masks", action="store_false")
    s.set_defaults(func=cmd_droid)

    s = sub.add_parser("viz", parents=[common], help="Stage 8: visualize trajectory + sparse points")
    s.add_argument("--no-show", action="store_true", help="do not open Open3D viewer")
    s.set_defaults(func=cmd_viz)

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
    s.add_argument("--use-masks", action="store_true", default=True)
    s.add_argument("--no-masks", dest="use_masks", action="store_false")
    s.add_argument("--no-show", action="store_true", default=True)
    s.set_defaults(func=cmd_all_cuda)

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
