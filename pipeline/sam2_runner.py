"""Stage 4 — SAM 2 prompt-and-propagate for surgical tool segmentation.

Two subcommands:
  4a) sam2-prompt    : interactive matplotlib GUI for picking foreground/background
                       points on selected key frames (works on Mac).
  4b) sam2-propagate : runs SAM 2 video predictor to fill mask for every frame
                       (CUDA strongly recommended → Windows + 3080).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .common import ensure_dir, list_frames, log, progress, timed

STAGE = "sam2"


# -------------------- 4a: interactive prompt picker --------------------

def run_prompt(frames_dir: Path, out_path: Path,
               key_seconds: list[float] = (8.0, 12.0, 30.0, 44.0),
               fps: float | None = None) -> None:
    """
    Open matplotlib viewer on `key_seconds` × fps frames, let user click prompts.

    Left click  : positive (tool foreground)
    Right click : negative (background)
    Press 'n'   : save current frame's points and go to next key frame
    Press 'd'   : delete the last point
    Press 'q'   : quit early
    """
    import matplotlib
    matplotlib.use("TkAgg")  # interactive backend
    import matplotlib.pyplot as plt

    frames = list_frames(Path(frames_dir))
    if not frames:
        raise FileNotFoundError(frames_dir)

    # Determine fps
    if fps is None:
        meta = Path(frames_dir).parent / "metadata.json"
        if meta.exists():
            fps = json.loads(meta.read_text()).get("fps", 30.0)
        else:
            fps = 30.0
            log(STAGE, "metadata.json not found — assuming fps=30", level="warn")

    # Pick key frame indices (clip to available range)
    candidates = [int(round(t * fps)) for t in key_seconds]
    key_idxs = sorted({i for i in candidates if 0 <= i < len(frames)})
    log(STAGE, f"prompt frames (idx): {key_idxs}")

    prompts: dict[str, dict] = {}

    for idx in key_idxs:
        img = cv2.cvtColor(cv2.imread(str(frames[idx])), cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.imshow(img)
        ax.set_title(f"frame {idx}  —  L=+ (tool)  R=– (bg)  n=next  d=undo  q=quit")
        points: list[tuple[int, int, int]] = []  # (x, y, label)

        def redraw():
            ax.clear()
            ax.imshow(img)
            for (x, y, lbl) in points:
                ax.plot(x, y, "o", color=("lime" if lbl == 1 else "red"),
                        markersize=10, markeredgecolor="black")
            ax.set_title(f"frame {idx}  —  L=+ R=– n=next d=undo q=quit  ({len(points)} pts)")
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes != ax or event.xdata is None:
                return
            label = 1 if event.button == 1 else 0
            points.append((int(event.xdata), int(event.ydata), label))
            redraw()

        state = {"done": False, "quit": False}

        def on_key(event):
            if event.key == "n":
                state["done"] = True
                plt.close(fig)
            elif event.key == "q":
                state["quit"] = True
                state["done"] = True
                plt.close(fig)
            elif event.key == "d" and points:
                points.pop()
                redraw()

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

        if points:
            prompts[str(idx)] = {
                "points": [[p[0], p[1]] for p in points],
                "labels": [p[2] for p in points],
            }
            log(STAGE, f"frame {idx}: collected {len(points)} prompt(s)")
        if state["quit"]:
            log(STAGE, "user quit early", level="warn")
            break

    out_path = Path(out_path)
    out_path.write_text(json.dumps(prompts, indent=2))
    log(STAGE, f"prompts → {out_path}", level="ok")


# -------------------- 4b: SAM 2 video propagation --------------------

def run_propagate(frames_dir: Path, prompts_path: Path, out_dir: Path,
                  ckpt: str | None = None,
                  model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml") -> None:
    """Load SAM 2 video predictor, init from prompts.json, propagate to whole video."""
    out_dir = ensure_dir(out_dir)
    frames = list_frames(Path(frames_dir))
    prompts = json.loads(Path(prompts_path).read_text())
    if not prompts:
        raise ValueError(f"prompts file {prompts_path} is empty")

    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as e:
        log(STAGE, f"sam2 / torch not installed in this environment: {e}", level="err")
        log(STAGE, "install on Windows+CUDA:", level="warn")
        log(STAGE, "  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        log(STAGE, "  pip install git+https://github.com/facebookresearch/sam2.git")
        raise SystemExit(2)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    )
    log(STAGE, f"device = {device}")
    if device != "cuda":
        log(STAGE, "running SAM 2 video predictor on non-CUDA device — will be slow", level="warn")

    if ckpt is None:
        # default name pattern
        ckpt = "sam2.1_hiera_large.pt"
    if not Path(ckpt).exists():
        log(STAGE, f"checkpoint {ckpt} not found", level="err")
        log(STAGE, "download from https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
            level="warn")
        raise SystemExit(2)

    log(STAGE, f"loading SAM 2 ({model_cfg}, ckpt={ckpt}) ...")
    predictor = build_sam2_video_predictor(model_cfg, ckpt, device=device)

    # SAM 2 expects a folder of jpg/png frames
    with timed(STAGE, "initializing predictor state"):
        state = predictor.init_state(video_path=str(frames_dir))

    # Add each prompt set; SAM 2 supports multiple objects via obj_id (we use 1 = tool).
    for frame_str, p in prompts.items():
        f_idx = int(frame_str)
        pts = np.array(p["points"], dtype=np.float32)
        lbls = np.array(p["labels"], dtype=np.int32)
        log(STAGE, f"adding prompt at frame {f_idx}: {len(pts)} points")
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=f_idx,
            obj_id=1,
            points=pts,
            labels=lbls,
        )

    # Propagate forward
    h, w = cv2.imread(str(frames[0])).shape[:2]
    n_written = 0
    with timed(STAGE, "propagating mask through video"):
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
            # masks: (n_obj, 1, H, W) bool / float
            m = masks[0].squeeze().cpu().numpy() if hasattr(masks[0], "cpu") else masks[0]
            m = (m > 0).astype(np.uint8) * 255
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out_dir / f"{frame_idx:06d}.png"), m)
            n_written += 1
            if n_written % 50 == 0:
                log(STAGE, f"  wrote {n_written} masks ...")
    log(STAGE, f"total tool masks written: {n_written} → {out_dir}", level="ok")
