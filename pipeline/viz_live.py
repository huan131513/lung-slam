"""Stage 9b — Open3D live-animated reconstruction replay.

Reproduces the presentation style of DROID-SLAM's original real-time Open3D
visualizer (droid_slam/visualization.py's droid_visualization()): the point
cloud grows keyframe-by-keyframe and a camera frustum is dropped at each pose,
instead of showing everything at once like the official view_reconstruction.py.

Unlike the real-time version (which only exists while DROID-SLAM is actually
running, and — in the current upstream repo — has been replaced by a
moderngl-based visualizer with a different look), this replays an already
saved reconstruction.pth, so you can watch the animation as many times as you
want without re-running SLAM. Math (iproj / depth_filter masking) matches
DROID-SLAM's own view_reconstruction.py exactly.

If frames_dir is given, a second OpenCV window shows the real endoscope frame
for whichever keyframe was just revealed, kept in sync with the 3D animation
(same tstamps.npy keyframe -> original-frame-index mapping viz_sync.py uses).

Controls (standard Open3D navigation, always available):
  left-drag    orbit
  right-drag   pan
  scroll       zoom
Keys:
  P            pause / resume the growth animation
  R            restart the animation from frame 0
  S / A        looser / stricter depth-consistency filter (re-filters + restarts)
  Q / Esc      quit
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from .common import log, timed

STAGE = "viz-live"

_CAM_POINTS = np.array([
    [0, 0, 0],
    [-1, -1, 1.5],
    [1, -1, 1.5],
    [1, 1, 1.5],
    [-1, 1, 1.5],
    [-0.5, 1, 1.5],
    [0.5, 1, 1.5],
    [0, 1.2, 1.5],
])
_CAM_LINES = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 0), (0, 2), (3, 0), (0, 4), (5, 7), (7, 6)]


def _camera_actor(o3d, scale: float, color=(0.0, 0.5, 0.9)):
    actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * _CAM_POINTS),
        lines=o3d.utility.Vector2iVector(_CAM_LINES),
    )
    actor.paint_uniform_color(color)
    return actor


def run(droid_dir: Path, frames_dir: Path | None = None, droid_root: str | None = None,
        fps: float = 6.0, filter_thresh: float = 0.005, filter_count: int = 2,
        cam_scale: float = 0.05) -> None:
    import cv2
    import open3d as o3d
    import droid_backends
    from lietorch import SE3

    droid_dir = Path(droid_dir)
    recon_path = droid_dir / "reconstruction.pth"
    if not recon_path.exists():
        log(STAGE, f"{recon_path} not found — run the droid stage first", level="err")
        raise SystemExit(2)

    droid_root = droid_root or os.environ.get("DROID_SLAM_ROOT")

    tstamps = None
    frame_files: list[Path] = []
    if frames_dir is not None:
        frames_dir = Path(frames_dir)
        tstamps_path = droid_dir / "tstamps.npy"
        if tstamps_path.exists() and frames_dir.exists():
            tstamps = np.load(tstamps_path).astype(int)
            frame_files = sorted(frames_dir.glob("*.png")) or sorted(frames_dir.glob("*.jpg"))
            if not frame_files:
                log(STAGE, f"no frames found in {frames_dir} — skipping real-image window", level="warn")
                tstamps = None
        else:
            log(STAGE, f"{tstamps_path} or {frames_dir} not found — skipping real-image window", level="warn")

    show_frames = tstamps is not None and len(frame_files) > 0
    frame_win = "Endoscope camera view"

    with timed(STAGE, "Loading reconstruction + computing point cloud"):
        bundle = torch.load(recon_path, map_location="cpu")
        images = bundle["images"].cuda()[..., ::2, ::2]
        disps = bundle["disps"].cuda()[..., ::2, ::2].contiguous()
        poses = bundle["poses"].cuda()
        intrinsics = 4 * bundle["intrinsics"].cuda()
        n_frames = len(images)

        colors_all = (images[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.0).cpu()
        cams_c2w = SE3(poses).inv().matrix().cpu().numpy()

        traj_center = cams_c2w[:, :3, 3].mean(axis=0)
        traj_radius = float(np.linalg.norm(cams_c2w[:, :3, 3] - traj_center, axis=1).max())

    log(STAGE, f"{n_frames} keyframes loaded")

    state = {
        "ix": 0, "playing": True, "last_t": None,
        "cam_actors": [], "pt_actors": [],
        "filter_thresh": filter_thresh,
        "frame_points": None, "frame_colors": None,
        # set once, the first time ANY geometry has been added to the
        # (until-then empty) scene, and never reset by restart()/refilter —
        # so R / S / A don't clobber a view the user rotated to by hand.
        "view_initialized": False,
    }

    def initial_endoscope_extrinsic():
        # Frame 0's own rotation (world frame ~= frame 0 for DROID-SLAM, but
        # use the real rotation rather than assume identity) — same
        # convention already validated in viz_sync.py — points the camera
        # to match the source video's left/right/up/down. Back the
        # viewpoint off from frame 0 along its own -Z so the whole
        # trajectory (not just frame 0) fits in view.
        R0 = cams_c2w[0][:3, :3]
        view_dist = max(traj_radius * 2.5, 0.3)
        cam_pos = traj_center - R0[:, 2] * view_dist
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R0.T
        extrinsic[:3, 3] = -R0.T @ cam_pos
        return extrinsic

    def recompute_mask():
        with torch.no_grad():
            index = torch.arange(n_frames, device="cuda")
            thresh = state["filter_thresh"] * torch.ones_like(disps.mean(dim=[1, 2]))
            points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
            counts = droid_backends.depth_filter(poses, disps, intrinsics[0], index, thresh)
            mask = (counts >= filter_count) & (disps > 0.25 * disps.mean())
            state["frame_points"] = [points[i][mask[i]].cpu().numpy() for i in range(n_frames)]
            state["frame_colors"] = [colors_all[i][mask[i].cpu()].numpy() for i in range(n_frames)]
        n_pts = sum(len(p) for p in state["frame_points"])
        log(STAGE, f"filter_thresh={state['filter_thresh']:.4f} -> {n_pts} points")

    def restart(vis):
        for actor in state["cam_actors"] + state["pt_actors"]:
            vis.remove_geometry(actor, reset_bounding_box=False)
        state["cam_actors"].clear()
        state["pt_actors"].clear()
        state["ix"] = 0
        state["last_t"] = None
        state["playing"] = True
        log(STAGE, "restarted")

    def toggle_play(vis):
        state["playing"] = not state["playing"]
        log(STAGE, "playing" if state["playing"] else "paused")

    def looser_filter(vis):
        state["filter_thresh"] *= 2
        recompute_mask()
        restart(vis)

    def stricter_filter(vis):
        state["filter_thresh"] *= 0.5
        recompute_mask()
        restart(vis)

    def animation_callback(vis):
        if not state["playing"] or state["ix"] >= n_frames:
            return
        period = 1.0 / max(fps, 0.1)
        now = time.time()
        if state["last_t"] is not None and now - state["last_t"] < period:
            return
        state["last_t"] = now

        i = state["ix"]

        # add_geometry() resets Open3D's own camera on every single call
        # (leaving reset_bounding_box at its default). Letting that happen
        # and then restoring the view straight after — every tick, not just
        # once — is DROID-SLAM's own droid_visualization() technique ("hack
        # to allow interacting with visualization during inference"). Doing
        # it only once, up front, and passing reset_bounding_box=False for
        # every later add looked fine in quick tests but silently degraded
        # over a real ~20s default-speed playback (clip planes/zoom drift
        # until the point cloud stopped rendering) — because Open3D never
        # got to refit anything to the actual (growing) scene extent again.
        if state["view_initialized"]:
            cam_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        else:
            cam_params = None

        cam = _camera_actor(o3d, cam_scale)
        cam.transform(cams_c2w[i])
        vis.add_geometry(cam)
        state["cam_actors"].append(cam)

        pts, cols = state["frame_points"][i], state["frame_colors"][i]
        if len(pts) > 0:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pts)
            pc.colors = o3d.utility.Vector3dVector(cols)
            vis.add_geometry(pc)
            state["pt_actors"].append(pc)

        if cam_params is None:
            # Nothing meaningful to restore yet (first-ever geometry) — use
            # our own endoscope-aligned view instead. Only happens once:
            # from here on cam_params always reflects whatever the user
            # last rotated to, so R / S / A restarts don't clobber it.
            cam_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
            cam_params.extrinsic = initial_endoscope_extrinsic()
            state["view_initialized"] = True
        vis.get_view_control().convert_from_pinhole_camera_parameters(cam_params, True)

        if show_frames:
            fi = max(0, min(int(tstamps[i]), len(frame_files) - 1))
            frame_bgr = cv2.imread(str(frame_files[fi]))
            if frame_bgr is not None:
                cv2.setWindowTitle(frame_win, f"{frame_win} — frame {fi:04d}  (keyframe {i + 1}/{n_frames})")
                cv2.imshow(frame_win, frame_bgr)
                cv2.waitKey(1)

        state["ix"] += 1
        if state["ix"] >= n_frames:
            log(STAGE, "animation complete — drag to keep exploring, R to restart", level="ok")

        vis.poll_events()
        vis.update_renderer()

    recompute_mask()

    if show_frames:
        cv2.namedWindow(frame_win, cv2.WINDOW_NORMAL)
        log(STAGE, f"real-image window enabled — {len(frame_files)} frames in {frames_dir}")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)
    vis.register_key_callback(ord("R"), restart)
    vis.register_key_callback(ord("P"), toggle_play)
    vis.register_key_callback(ord("S"), looser_filter)
    vis.register_key_callback(ord("A"), stricter_filter)
    vis.create_window(height=960, width=960, window_name="Endoscope reconstruction — live replay")

    render_json = Path(droid_root) / "misc" / "renderoption.json" if droid_root else None
    if render_json and render_json.exists():
        vis.get_render_option().load_from_json(str(render_json))

    log(STAGE, "P=pause/resume  R=restart  S/A=looser/stricter filter  drag=orbit  scroll=zoom  Q=quit")
    vis.run()
    vis.destroy_window()
    if show_frames:
        cv2.destroyWindow(frame_win)
