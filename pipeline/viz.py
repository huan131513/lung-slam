"""Stage 8 — Visualize trajectory + sparse point cloud."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import ensure_dir, log, timed

STAGE = "viz"


def _quat_to_rot(qw, qx, qy, qz):
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def load_droid_outputs(droid_dir: Path):
    """Load DROID-SLAM outputs. Returns poses [N,4,4], points [M,3] (may be None)."""
    droid_dir = Path(droid_dir)
    poses_p = droid_dir / "poses.npy"
    if not poses_p.exists():
        raise FileNotFoundError(poses_p)
    poses_raw = np.load(poses_p)
    log(STAGE, f"loaded poses: shape={poses_raw.shape} dtype={poses_raw.dtype}")

    # DROID-SLAM stores [tx, ty, tz, qx, qy, qz, qw] per frame
    poses = np.zeros((len(poses_raw), 4, 4), dtype=np.float64)
    poses[:, 3, 3] = 1.0
    for i, p in enumerate(poses_raw):
        if p.shape[-1] == 7:
            tx, ty, tz, qx, qy, qz, qw = p
            R = _quat_to_rot(qw, qx, qy, qz)
            poses[i, :3, :3] = R
            poses[i, :3, 3] = [tx, ty, tz]
        elif p.shape == (4, 4):
            poses[i] = p
        else:
            raise ValueError(f"unrecognized pose shape: {p.shape}")

    # Try to load a sparse point cloud (DROID can dump via reconstruction_path)
    pts = None
    for name in ("points.ply", "pointcloud.ply", "sparse.ply"):
        ply = droid_dir / name
        if ply.exists():
            try:
                import open3d as o3d
                cloud = o3d.io.read_point_cloud(str(ply))
                pts = np.asarray(cloud.points)
                log(STAGE, f"loaded {len(pts)} points from {ply.name}")
                break
            except ImportError:
                log(STAGE, "open3d not installed — skipping point cloud", level="warn")
                break
    return poses, pts


def export_json(poses: np.ndarray, out_path: Path):
    traj = []
    for i, T in enumerate(poses):
        traj.append({"frame": i,
                     "t": T[:3, 3].tolist(),
                     "R": T[:3, :3].tolist()})
    Path(out_path).write_text(json.dumps(traj, indent=2))
    log(STAGE, f"trajectory.json → {out_path}", level="ok")


def plot_2d(poses: np.ndarray, out_png: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    xyz = poses[:, :3, 3]
    for ax, (i, j, lbl) in zip(axes, [(0, 1, "XY (top)"), (0, 2, "XZ (side)"), (1, 2, "YZ (front)")]):
        ax.plot(xyz[:, i], xyz[:, j], "-", lw=1, color="steelblue")
        ax.scatter(xyz[0, i], xyz[0, j], c="green", s=40, label="start", zorder=3)
        ax.scatter(xyz[-1, i], xyz[-1, j], c="red", s=40, label="end", zorder=3)
        ax.set_title(lbl)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    log(STAGE, f"2D trajectory plot → {out_png}", level="ok")


def show_3d(poses: np.ndarray, pts: np.ndarray | None):
    try:
        import open3d as o3d
    except ImportError:
        log(STAGE, "open3d not installed — cannot open 3D viewer", level="warn")
        log(STAGE, "  pip install open3d")
        return

    geoms = []
    # Camera trajectory as line set
    xyz = poses[:, :3, 3]
    lines = [[i, i + 1] for i in range(len(xyz) - 1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(xyz)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([[0.2, 0.6, 1.0]] * len(lines))
    geoms.append(ls)

    # Frusta every N poses (use small coord frames)
    step = max(1, len(poses) // 30)
    for T in poses[::step]:
        f = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        f.transform(T)
        geoms.append(f)

    # Point cloud
    if pts is not None and len(pts) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
        geoms.append(pcd)

    log(STAGE, "opening Open3D viewer (close window to continue) ...")
    o3d.visualization.draw_geometries(geoms,
                                      window_name="Endoscope trajectory + sparse points")


def run(droid_dir: Path, out_dir: Path, show: bool = True) -> None:
    droid_dir = Path(droid_dir)
    out_dir = ensure_dir(out_dir)
    with timed(STAGE, "Loading DROID outputs"):
        poses, pts = load_droid_outputs(droid_dir)
    export_json(poses, out_dir / "trajectory.json")
    plot_2d(poses, out_dir / "trajectory.png")
    if show:
        show_3d(poses, pts)
