"""Kiểm tra trước khi train một lượt có loss độ sâu (depth_loss của gsplat, vd. biến thể `depth` của mcmc trong
configs/experiments/tune_mcmc.yaml): mỗi ảnh train của cảnh phải có điểm trong track của SfM; ảnh mà mọi điểm đều nằm
ngoài ảnh chỉ bị cảnh báo. Chỉ đọc dữ liệu, không dùng GPU, vài giây tới vài chục giây mỗi cảnh.

    python -m src.training.depth_check --examples-dir $WS/code/gsplat/examples --test-every 8 <cảnh> [<cảnh> ...]
    (chạy trong env gs-gsplat; scripts/pod/run_experiment.sh gọi sau khi tải dữ liệu, trước khi train)

Vì sao cần: với `--depth-loss`, Dataset của gsplat @6e8c837 (examples/datasets/colmap.py) lấy các điểm 3D mà COLMAP ghi
là thấy trong ảnh (track trong points3D.bin), chiếu vào ảnh, chỉ giữ điểm nằm trong ảnh và ở phía trước camera:
  * ảnh không có điểm nào trong track: KeyError khi nạp ảnh đó, lượt train lỗi (không đạt);
  * ảnh có điểm nhưng không điểm nào được giữ: loss độ sâu của ảnh đó là trung bình của tập rỗng, nên giá trị loss ghi
    ra (thanh tiến trình, TensorBoard) là NaN ở những vòng gặp ảnh đó; gradient vẫn hữu hạn (phần độ sâu bằng 0, thử bằng
    PyTorch trên CPU), nên lượt train không hỏng: chỉ cảnh báo, kèm tên ảnh.
Module này dựng Parser thật của gsplat (normalize và test_every như simple_trainer.py) với hệ số ảnh 1, nên không ghi
gì vào thư mục cảnh (hệ số > 1 làm Parser ghi ảnh PNG thu nhỏ), rồi đếm điểm được giữ của từng ảnh train đúng như
Dataset của gsplat, không nạp ảnh (kích thước ảnh lấy từ Parser). Kết quả như nhau với mọi hệ số thu nhỏ ảnh (K và kích
thước ảnh cùng chia cho hệ số), trừ điểm nằm sát mép ảnh.
Mã thoát: 0 khi mọi cảnh đạt (kể cả khi có cảnh báo); 1 khi có cảnh không đạt hoặc không đọc được; 2 khi tham số sai
hoặc không nạp được gsplat.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

SHOW = 10  # số tên ảnh in ra khi không đạt


def valid_point_counts(parser, test_every: int) -> dict[str, int]:
    """Số điểm SfM được giữ cho từng ảnh train, như Dataset(split="train", load_depths=True) của gsplat @6e8c837.

    Ảnh train: chỉ số (theo tên ảnh đã sắp xếp) không chia hết cho test_every. Ảnh không có trong
    `parser.point_indices` (không có điểm nào trong track): -1."""
    counts = {}
    for index, name in enumerate(parser.image_names):
        if index % test_every == 0:  # ảnh test
            continue
        if name not in parser.point_indices:
            counts[name] = -1
            continue
        camera_id = parser.camera_ids[index]
        K = parser.Ks_dict[camera_id]
        width, height = parser.imsize_dict[camera_id]
        worldtocam = np.linalg.inv(parser.camtoworlds[index])
        points_world = parser.points[parser.point_indices[name]]
        points_cam = (worldtocam[:3, :3] @ points_world.T + worldtocam[:3, 3:4]).T
        points_proj = (K @ points_cam.T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = points_proj[:, :2] / points_proj[:, 2:3]
        depths = points_cam[:, 2]
        kept = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height) & (depths > 0)
        counts[name] = int(kept.sum())
    return counts


def load_parser(examples_dir: Path):
    """Lớp Parser của examples/datasets/colmap.py (gsplat), nạp như simple_trainer.py. `datasets` là tên gói rất dễ trùng
    (vd. gói datasets của Hugging Face): module phải nằm trong examples_dir, không thì RuntimeError."""
    examples = Path(examples_dir).resolve()
    sys.path.insert(0, str(examples))
    import datasets.colmap as colmap  # examples/datasets/colmap.py của gsplat

    origin = Path(getattr(colmap, "__file__", "") or "").resolve()
    if examples not in origin.parents:
        raise RuntimeError(f"datasets.colmap was imported from {origin}, not from {examples}")
    return colmap.Parser


def check_scene(parser_class, scene: Path, test_every: int) -> tuple[bool, str]:
    """(đạt?, một dòng kết quả) cho một cảnh."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # Parser in vài dòng thông tin
            parser = parser_class(data_dir=str(scene), factor=1, normalize=True, test_every=test_every)
    except Exception as error:  # noqa: BLE001 - mọi lỗi đọc cảnh đều làm lượt train lỗi: báo và dừng cảnh này
        return False, f"{scene.name}: FAIL: gsplat cannot read the scene ({type(error).__name__}: {error})"
    counts = valid_point_counts(parser, test_every)
    if not counts:
        return False, f"{scene.name}: FAIL: no training images (test_every {test_every})"
    no_track = sorted(name for name, count in counts.items() if count < 0)
    no_point = sorted(name for name, count in counts.items() if count == 0)
    if no_track:
        return False, (f"{scene.name}: FAIL: {len(counts)} training images, {len(no_track)} with no SfM point in their "
                       f"tracks (gsplat: KeyError when loading them) {no_track[:SHOW]}")
    values = np.array(sorted(counts.values()))
    line = (f"{scene.name}: OK, {len(counts)} training images, SfM points per image: "
            f"min {values[0]}, median {int(np.median(values))}")
    if no_point:
        line += (f"; WARN: {len(no_point)} whose SfM points all fall outside the image (no depth supervision there; "
                 f"the logged loss is NaN on those steps, the gradients are not) {no_point[:SHOW]}")
    return True, line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples-dir", type=Path, required=True, help="thư mục examples/ của gsplat (bản ghim)")
    parser.add_argument("--test-every", type=int, default=8, help="như protocol (configs/base.yaml, train_modes)")
    parser.add_argument("scenes", nargs="+", type=Path, help="thư mục cảnh có images/ và sparse/0/")
    args = parser.parse_args(argv)
    if args.test_every < 1:
        print("ERROR: --test-every must be at least 1", file=sys.stderr)
        return 2
    try:
        Parser = load_parser(args.examples_dir)
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: cannot import gsplat's COLMAP parser from {args.examples_dir}: {error}", file=sys.stderr)
        return 2
    failed = 0
    for scene in args.scenes:
        ok, line = check_scene(Parser, scene, args.test_every)
        print(line, flush=True)
        failed += not ok
    if failed:
        print(f"ERROR: {failed} scene(s) cannot be trained with the SfM depth loss (see above): a run with depth_loss "
              "there would fail (KeyError) or the scene cannot be read", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
