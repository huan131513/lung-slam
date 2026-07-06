"""Stage 9 — Synchronized video + 3D trajectory viewer.

Shows the original endoscope frame (left) alongside the reconstructed
camera position in 3D space (right), stepping through DROID-SLAM keyframes.

Controls:
  space      pause / resume
  ←  →       step one keyframe backward / forward
  x  y  z    snap view so that axis points into the screen
  q / Esc    quit
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import log

STAGE = "viz-sync"


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = (qw**2 + qx**2 + qy**2 + qz**2) ** 0.5
    if n < 1e-9:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),      2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),      1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),      1 - 2*(qx**2 + qy**2)],
    ])


def _load_point_cloud(ply_path: Path, max_pts: int = 40000):
    """Returns (points [M,3], colors [M,3] in 0..1 or None) or (None, None)."""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors) if pcd.has_colors() else None
        if len(pts) == 0:
            return None, None
        if len(pts) > max_pts:
            idx = np.random.default_rng(0).choice(len(pts), max_pts, replace=False)
            pts = pts[idx]
            if cols is not None:
                cols = cols[idx]
        log(STAGE, f"loaded {len(pts)} points from {ply_path.name}")
        return pts, cols
    except ImportError:
        log(STAGE, "open3d not installed — skipping point cloud", level="warn")
        return None, None
    except Exception as e:
        log(STAGE, f"could not load {ply_path.name}: {e}", level="warn")
        return None, None


# Camera frustum wireframe (apex = camera center, base = image plane at z=1.5)
# — same shape DROID-SLAM's own visualizer uses, so it reads the same way.
_CAM_POINTS = np.array([
    [ 0,    0,   0],
    [-1,   -1, 1.5],
    [ 1,   -1, 1.5],
    [ 1,    1, 1.5],
    [-1,    1, 1.5],
    [-0.5,  1, 1.5],
    [ 0.5,  1, 1.5],
    [ 0,  1.2, 1.5],
])
_CAM_LINES = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 0),
              (0, 2), (3, 0), (0, 4), (5, 7), (7, 6)]


def _frustum_segments(t: np.ndarray, R: np.ndarray, scale: float):
    verts = (_CAM_POINTS * scale) @ R.T + t
    return [(verts[i], verts[j]) for i, j in _CAM_LINES]


def run(frames_dir: Path, droid_dir: Path, fps: float = 5.0) -> None:
    import matplotlib as mpl
    # Remove left/right from all default keymaps so arrow keys don't rotate the 3D view
    for param in list(mpl.rcParams.keys()):
        if param.startswith("keymap.") and isinstance(mpl.rcParams[param], list):
            mpl.rcParams[param] = [k for k in mpl.rcParams[param]
                                   if k not in ("left", "right")]
    mpl.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    frames_dir = Path(frames_dir)
    droid_dir = Path(droid_dir)

    # ── Load DROID outputs ──────────────────────────────────────────────────
    poses = np.load(droid_dir / "poses.npy")              # (N, 7)  tx ty tz qx qy qz qw
    tstamps = np.load(droid_dir / "tstamps.npy").astype(int)  # (N,)

    N = len(poses)
    log(STAGE, f"{N} keyframes · frame indices {tstamps[0]}..{tstamps[-1]}")

    frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("*.jpg"))
    n_frames = len(frame_files)
    log(STAGE, f"{n_frames} original frames in {frames_dir}")

    xyz = poses[:, :3]  # (N, 3) — camera positions

    # Arrow scale: 5% of scene extent, at least 0.01
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    arrow_scale = max(float(extent.max()) * 0.06, 0.01)
    margin = arrow_scale * 2

    # ── Figure layout ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 6.5), facecolor="#12121f")
    fig.suptitle(
        "Endoscope trajectory validation  |  ←→=step  space=auto-play  scroll=zoom  drag=rotate  x/y/z=axis view  q=quit",
        color="#cccccc", fontsize=9, y=0.98,
    )

    ax_vid = fig.add_subplot(1, 2, 1)
    ax_vid.set_facecolor("black")
    ax_vid.axis("off")

    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    ax3d.set_facecolor("#0d1117")
    ax3d.tick_params(colors="#666666", labelsize=7)
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#333344")
    ax3d.set_xlabel("X", color="#888888", fontsize=8)
    ax3d.set_ylabel("Y", color="#888888", fontsize=8)
    ax3d.set_zlabel("Z", color="#888888", fontsize=8)

    # Static: full trajectory line
    ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
              "-", color="#3a7fd5", lw=0.9, alpha=0.5, label="trajectory")
    ax3d.scatter(*xyz[0], c="lime", s=50, zorder=5, label="start", depthshade=False)
    ax3d.scatter(*xyz[-1], c="tomato", s=50, zorder=5, label="end", depthshade=False)

    # Dense colored point cloud (from droid_backends.iproj, world frame)
    pts, cols = _load_point_cloud(droid_dir / "points.ply")
    if pts is not None:
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                     c=cols if cols is not None else "#aaaaaa",
                     s=0.6, alpha=0.5, depthshade=False, zorder=1)

    # Static white camera frustum at every keyframe (matches DROID-SLAM's own
    # wireframe-box look) — the currently-selected one is redrawn in
    # _draw_camera() below, highlighted in yellow instead.
    for i in range(N):
        R_i = _quat_to_rot(poses[i, 3], poses[i, 4], poses[i, 5], poses[i, 6])
        for p0, p1 in _frustum_segments(xyz[i], R_i, arrow_scale):
            ax3d.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                      color="white", lw=1.4, alpha=0.7, zorder=2)

    # Base axis limits (stored for zoom computation)
    base_lims = [
        (xyz[:, 0].min() - margin, xyz[:, 0].max() + margin),
        (xyz[:, 1].min() - margin, xyz[:, 1].max() + margin),
        (xyz[:, 2].min() - margin, xyz[:, 2].max() + margin),
    ]
    ax3d.set_xlim(*base_lims[0])
    ax3d.set_ylim(*base_lims[1])
    ax3d.set_zlim(*base_lims[2])
    ax3d.legend(loc="upper left", fontsize=7, facecolor="#1a1a2e",
                labelcolor="white", framealpha=0.6)

    # View state: elev/azim saved after mouse rotation; zoom from scroll wheel.
    # Default = endoscope's own view: world axes at keyframe 0 equal the
    # camera's local axes (image convention: +X = image right, +Y = image
    # down, +Z = forward/into the scene), so this makes the 3D panel's
    # left/right/up/down match the video panel's for the starting frame.
    # elev=-90         → camera at -Z looking toward +Z (Z axis into screen)
    # azim=-90 (not 90)→ screen right = +X, screen down = +Y (checked by
    #                     rendering plain axis arrows with matplotlib's Agg
    #                     backend: azim=90 put +X on screen-LEFT and +Y UP,
    #                     which is mirrored/upside-down vs. the video frame)
    view = {"elev": -90, "azim": -90, "zoom": 1.0}

    # Press x/y/z to snap the camera so that axis points INTO the screen
    # (camera placed on the negative side, looking toward the positive side).
    AXIS_VIEWS = {
        "x": {"elev": 0, "azim": 180},
        "y": {"elev": 0, "azim": -90},
        "z": {"elev": -90, "azim": -90},
    }

    def _apply_view() -> None:
        """Restore saved view angle and zoom level after every redraw."""
        z = view["zoom"]
        for (lo, hi), setter in zip(base_lims,
                                    [ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim]):
            c = (lo + hi) / 2
            h = (hi - lo) / 2 * z
            setter(c - h, c + h)
        ax3d.view_init(elev=view["elev"], azim=view["azim"])

    # Label: legend for frustum colors
    ax3d.text2D(
        0.02, 0.04,
        "Yellow box = current camera  ·  white boxes = other keyframes  ·  white arrow (on yellow box) = viewing direction",
        transform=ax3d.transAxes,
        color="white", fontsize=8,
        bbox=dict(facecolor="#1a1a2e", edgecolor="#444466",
                  boxstyle="round,pad=0.3", alpha=0.8),
    )

    # ── Initial video frame ─────────────────────────────────────────────────
    def _read_frame(fi: int):
        fi = max(0, min(fi, n_frames - 1))
        bgr = cv2.imread(str(frame_files[fi]))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    im = ax_vid.imshow(_read_frame(tstamps[0]))
    title_vid = ax_vid.set_title("", color="#dddddd", fontsize=9, pad=4)
    title_3d = ax3d.set_title("", color="#dddddd", fontsize=9, pad=4)

    # ── Camera icon (mutable list so update() can replace) ──────────────────
    cam_artists: list = []

    def _draw_camera(i: int) -> None:
        """Draw/refresh camera icon at keyframe i."""
        for a in cam_artists:
            try:
                a.remove()
            except Exception:
                pass
        cam_artists.clear()

        t = poses[i, :3]
        R = _quat_to_rot(poses[i, 3], poses[i, 4], poses[i, 5], poses[i, 6])

        # Highlighted frustum box at the current keyframe (bigger + brighter
        # than the static blue ones drawn once at setup)
        for p0, p1 in _frustum_segments(t, R, arrow_scale * 1.4):
            line, = ax3d.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                               color="yellow", lw=2.2, zorder=10)
            cam_artists.append(line)

        # White arrow = viewing direction (local +Z), drawn on top of the box
        vec = R[:, 2]
        q = ax3d.quiver(
            t[0], t[1], t[2],
            vec[0] * arrow_scale, vec[1] * arrow_scale, vec[2] * arrow_scale,
            color="white", linewidth=2.0, arrow_length_ratio=0.25, zorder=11,
        )
        cam_artists.append(q)

        # Red dot at camera position
        dot = ax3d.scatter(
            [t[0]], [t[1]], [t[2]],
            c="red", s=70, zorder=12, depthshade=False,
        )
        cam_artists.append(dot)

    # ── Playback state — start paused so arrow keys work immediately ────────
    state = {"paused": True, "idx": 0}

    def _refresh(i: int) -> None:
        fi = int(tstamps[i])
        im.set_data(_read_frame(fi))
        title_vid.set_text(
            f"Original frame {fi:04d}  (keyframe {i + 1}/{N}  t={fi/30:.1f}s)"
        )
        title_3d.set_text(f"Camera pose  [{i + 1} / {N}]")
        _draw_camera(i)
        _apply_view()   # restore angle + zoom after quiver/scatter may have rescaled
        fig.canvas.draw_idle()

    def _update(_frame_counter):
        if not state["paused"]:
            _refresh(state["idx"])
            state["idx"] = (state["idx"] + 1) % N

    def _on_key(event):
        key = event.key
        if key == " ":
            state["paused"] = not state["paused"]
            log(STAGE, "paused" if state["paused"] else "resumed")
        elif key == "right":
            state["paused"] = True  # stop auto-play when manually stepping
            state["idx"] = min(state["idx"] + 1, N - 1)
            _refresh(state["idx"])
        elif key == "left":
            state["paused"] = True  # stop auto-play when manually stepping
            state["idx"] = max(state["idx"] - 1, 0)
            _refresh(state["idx"])
        elif key in ("q", "escape"):
            plt.close(fig)
        elif key in AXIS_VIEWS:
            view["elev"] = AXIS_VIEWS[key]["elev"]
            view["azim"] = AXIS_VIEWS[key]["azim"]
            log(STAGE, f"view -> {key.upper()} axis into screen")
            _apply_view()
            fig.canvas.draw_idle()

    def _on_mouse_release(event) -> None:
        # Save view angle after the user manually rotates with the mouse
        if event.inaxes == ax3d:
            view["elev"] = ax3d.elev
            view["azim"] = ax3d.azim

    def _on_scroll(event) -> None:
        if event.inaxes != ax3d:
            return
        view["zoom"] *= 0.85 if event.button == "up" else 1.15
        view["zoom"] = max(0.05, min(view["zoom"], 20.0))
        _apply_view()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig.canvas.mpl_connect("button_release_event", _on_mouse_release)
    fig.canvas.mpl_connect("scroll_event", _on_scroll)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    log(STAGE, "opening viewer — ←→=step  space=auto-play  x/y/z=axis view  q=quit")

    # Draw first keyframe immediately (viewer starts paused)
    _refresh(0)

    ani = FuncAnimation(  # noqa: F841 (kept alive by reference)
        fig, _update,
        interval=int(1000 / fps),
        cache_frame_data=False,
    )
    plt.show()
