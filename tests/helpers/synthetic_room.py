"""Phòng tổng hợp cho kiểm thử trang web: Gaussian tạo thành một căn phòng và camera đi một vòng bên trong.

Ghi bằng chính code xuất của sản phẩm (src/product/export_web.py: splat_records, write_splat, camera_layout,
rebuild_manifest), trong một hệ tọa độ có hướng lên bất kỳ như hệ đã chuẩn hóa của gsplat (không phải +y hay +z).
Phòng: sàn ô cờ, bốn tường bốn màu, một cái bàn, và các điểm mốc màu hồng tím (magenta) ở vị trí biết trước để kiểm
tra camera của trình xem khớp mô hình (điểm mốc phải hiện đúng pixel mà cameras.json tính ra).

Dùng: python tests/helpers/synthetic_room.py <thư mục web> [--gaussians N]  (in JSON: phòng, số Gaussian, điểm mốc)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.product import export_web  # noqa: E402

SIZE = (640, 480)
K = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
HALF_X, HALF_Z, HEIGHT = 2.0, 1.5, 2.6  # phòng 4 x 3 x 2,6 (đơn vị tương đối), y lên trong hệ cục bộ
MAGENTA = (1.0, 0.0, 1.0)
# điểm mốc (hệ cục bộ): trên tường, sát vào trong phòng một chút để không bị tường che; một điểm lơ lửng giữa phòng
MARKERS_LOCAL = [(-1.0, 1.6, -HALF_Z + 0.06), (0.8, 1.0, -HALF_Z + 0.06), (HALF_X - 0.06, 1.4, 0.3),
                 (-HALF_X + 0.06, 1.2, -0.4), (0.2, 2.0, HALF_Z - 0.06), (0.0, 1.2, 0.0)]


def rotation(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def surfaces(rng, count: int):
    """Điểm và màu trên sàn, bốn tường, mặt bàn (hệ cục bộ)."""
    pts, cols = [], []

    def add(points, colors):
        pts.append(points)
        cols.append(colors)

    n_floor = count * 30 // 100
    n_wall = count * 15 // 100
    n_table = count - n_floor - 4 * n_wall
    floor = np.stack([rng.uniform(-HALF_X, HALF_X, n_floor), np.zeros(n_floor), rng.uniform(-HALF_Z, HALF_Z, n_floor)], 1)
    checker = ((np.floor(floor[:, 0] / 0.5) + np.floor(floor[:, 2] / 0.5)) % 2)[:, None]
    add(floor, checker * np.array([0.85, 0.83, 0.78]) + (1 - checker) * np.array([0.22, 0.23, 0.28]))
    walls = [  # (cố định trục, giá trị, màu)
        (0, -HALF_X, (0.78, 0.32, 0.28)), (0, HALF_X, (0.30, 0.66, 0.38)),
        (2, -HALF_Z, (0.30, 0.42, 0.85)), (2, HALF_Z, (0.88, 0.78, 0.30)),
    ]
    for axis, value, color in walls:
        p = np.stack([rng.uniform(-HALF_X, HALF_X, n_wall), rng.uniform(0, HEIGHT, n_wall), rng.uniform(-HALF_Z, HALF_Z, n_wall)], 1)
        p[:, axis] = value
        shade = 0.85 + 0.15 * ((np.floor(p[:, 1] / 0.4)) % 2)[:, None]  # sọc ngang nhạt để thấy được chiều sâu
        add(p, shade * np.array(color))
    table = np.stack([rng.uniform(-0.6, 0.6, n_table), np.full(n_table, 0.75), rng.uniform(-0.35, 0.35, n_table)], 1)
    add(table, np.tile([0.55, 0.36, 0.20], (n_table, 1)))
    return np.concatenate(pts), np.clip(np.concatenate(cols), 0, 1)


def camera_path(frames: int) -> list[np.ndarray]:
    """Camera-to-world (OpenCV, hệ cục bộ): đi một vòng elip ở tầm mắt, nhìn về giữa phòng hơi lệch."""
    poses = []
    for i in range(frames):
        a = 2 * np.pi * i / frames
        position = np.array([1.25 * np.cos(a), 1.5 + 0.1 * np.sin(3 * a), 0.85 * np.sin(a)])
        target = np.array([0.35 * np.cos(a + 1.2), 1.1, 0.25 * np.sin(a + 1.2)])
        forward = target - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)  # trục y của camera OpenCV hướng xuống
        c2w = np.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, forward, position
        poses.append(c2w)
    return poses


def make_room(web_dir: Path, room: str = "phong_tong_hop", name: str = "Phòng tổng hợp", gaussians: int = 60000,
              frames: int = 48, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    # hệ cục bộ -> hệ của mô hình: xoay bất kỳ, dời, co giãn (hướng lên của phòng không trùng trục nào)
    world_rot = rotation([0.3, -0.5, 0.8], 1.1)
    world_t = np.array([0.2, -0.1, 0.3])
    world_s = 0.4

    def to_model(points):
        return (np.asarray(points, dtype=np.float64) @ world_rot.T) * world_s + world_t

    points, colors = surfaces(rng, gaussians)
    marker_pts = np.array(MARKERS_LOCAL)
    all_pts = np.concatenate([points, marker_pts])
    all_cols = np.concatenate([colors, np.tile(MAGENTA, (len(marker_pts), 1))])
    scales = np.full((len(all_pts), 3), np.log(0.035 * world_s))
    scales[len(points):] = np.log(0.06 * world_s)  # điểm mốc to hơn để dễ tìm trong ảnh
    opacity = np.full(len(all_pts), 3.0)  # sigmoid(3) ≈ 0,95
    opacity[len(points):] = 8.0
    dtype = [(n, "<f4") for n in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                                  "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")]
    vertices = np.zeros(len(all_pts), dtype=dtype)
    model_pts = to_model(all_pts)
    for i, axis in enumerate("xyz"):
        vertices[axis] = model_pts[:, i]
    dc = (all_cols - 0.5) / export_web.SH_C0
    for i in range(3):
        vertices[f"f_dc_{i}"] = dc[:, i]
        vertices[f"scale_{i}"] = scales[:, i]
    vertices["opacity"] = opacity
    vertices["rot_0"] = 1.0
    records, stats = export_web.splat_records(vertices)
    room_dir = Path(web_dir) / "rooms" / room
    digest = export_web.write_splat(records, room_dir / "model.splat")
    poses = []
    for c2w in camera_path(frames):
        m = np.eye(4)
        m[:3, :3] = world_rot @ c2w[:3, :3]
        m[:3, 3] = to_model(c2w[:3, 3][None])[0]
        poses.append(m)
    names = [f"{i + 1:05d}.jpg" for i in range(frames)]
    layout = export_web.camera_layout(poses, names, K, SIZE, model_pts)
    export_web.write_json(room_dir / "cameras.json", layout)
    entry = {
        "id": room, "name": name, "version": digest[:12], "built_at": "2026-11-04T08:00:00+00:00", "format": "splat",
        "files": {"model": f"rooms/{room}/model.splat", "cameras": f"rooms/{room}/cameras.json", "thumbnail": None},
        "num_gaussians": stats["kept"], "size_bytes": int(records.nbytes), "viewer_modes": ["orbit", "walk", "tour"],
        "source": {"kind": "video", "input_images": frames, "registered_images": frames, "sfm_verdict": "PASS",
                   "video_duration_s": 60.0},
        "model": {"name": "mcmc", "cap_max": 1000000, "max_steps": 30000, "trainer_commit": "synthetic",
                  "num_gaussians_trained": stats["kept"], "train_seconds": 650.0,
                  "trained_at": "2026-11-04T07:50:00+00:00"},
        "metrics": None, "warnings": [],
    }
    export_web.write_json(room_dir / "room.json", entry)
    export_web.rebuild_manifest(web_dir)
    return {"room": room, "num_gaussians": stats["kept"], "markers": to_model(marker_pts).tolist(),
            "up": layout["up"], "world_up": (world_rot @ np.array([0.0, 1.0, 0.0])).tolist()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("web_dir", type=Path)
    parser.add_argument("--room", default="phong_tong_hop")
    parser.add_argument("--gaussians", type=int, default=60000)
    args = parser.parse_args(argv)
    print(json.dumps(make_room(args.web_dir, args.room, gaussians=args.gaussians), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
