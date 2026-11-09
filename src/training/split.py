"""Chia ảnh train/test và ghi lại, để mọi chế độ huấn luyện đánh giá trên đúng cùng một tập ảnh.

Quy tắc mặc định (`every_n`) giống hệt hai bản cài đặt đang dùng (đã đối chiếu mã nguồn):

- Inria @ 54c035f, `scene/dataset_readers.py`: sắp xếp tên ảnh COLMAP, ảnh thứ i là test khi i % 8 == 0.
- gsplat @ 6e8c837, `examples/datasets/colmap.py`: sắp xếp tên ảnh, tập "val" gồm các ảnh i % test_every == 0.

Quy tắc thứ hai (`test_list`): danh sách ảnh test cho trước, mỗi dòng một tên ảnh (cùng định dạng
`sparse/0/test.txt` của Inria). Dùng khi có đoạn quay kiểm tra riêng (docs/data/capture_protocol.md);
áp dụng cho gsplat qua `src.training.gsplat_launcher`. Khi đó đoạn kiểm tra nằm trong cùng model COLMAP,
nên `make_train_view` tạo bản dữ liệu mà points3D.bin chỉ giữ các điểm đủ ảnh train quan sát: điểm chỉ
được dựng nhờ ảnh test không được dùng để khởi tạo Gaussian.

Kèm danh sách test có thể có danh sách ảnh bỏ ra (`excluded`): không train, không chấm. Dùng cho quy tắc
`segments` (khi không có đoạn quay kiểm tra riêng): các đoạn liền của chính video chính làm tập test, vài khung
mỗi bên đoạn bị bỏ ra để khung test không nằm sát khung train (docs/PROJECT_PLAN.md mục 4.2: "tách tập test
theo các đoạn cách xa nhau"). Tạo danh sách:

    python -m src.training.split segments --scene <cảnh có sparse/0> --out <thư mục>   # test.txt, exclude.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.reconstruction.colmap_model import read_images_binary, read_points3d_binary, write_points3d_binary


@dataclass
class Split:
    rule: str  # "every_n" | "test_list"
    test_every: int | None
    train: list[str]
    test: list[str]
    excluded: list[str] = field(default_factory=list)  # không train, không chấm (khung đệm của quy tắc segments)

    def digest(self) -> str:
        """SHA-256 của hai danh sách (đã sắp xếp); hai lượt chạy cùng digest là cùng tập train/test."""
        canonical = json.dumps({"train": sorted(self.train), "test": sorted(self.test)}, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        data = {
            "rule": self.rule,
            "test_every": self.test_every,
            "num_train": len(self.train),
            "num_test": len(self.test),
            "sha256": self.digest(),
            "train": self.train,
            "test": self.test,
        }
        if self.excluded:
            data["num_excluded"] = len(self.excluded)
            data["excluded"] = self.excluded
        return data


def every_nth_split(names: list[str], test_every: int) -> Split:
    if test_every < 2:
        raise ValueError("test_every must be >= 2")
    ordered = sorted(names)
    test = [name for index, name in enumerate(ordered) if index % test_every == 0]
    train = [name for index, name in enumerate(ordered) if index % test_every != 0]
    if not train:
        raise ValueError("no training images left after the split")
    return Split("every_n", test_every, train, test)


def list_split(names: list[str], test_names: list[str], exclude_names: list[str] = ()) -> Split:
    ordered = sorted(names)
    known = set(ordered)
    for label, given in (("test", test_names), ("excluded", exclude_names)):
        unknown = sorted(set(given) - known)
        if unknown:
            raise ValueError(f"{len(unknown)} {label} image(s) are not registered in the COLMAP model, "
                             f"e.g. {unknown[:3]}")
    wanted = set(test_names)
    dropped = set(exclude_names)
    both = sorted(wanted & dropped)
    if both:
        raise ValueError(f"{len(both)} image(s) are both in the test list and in the exclude list, e.g. {both[:3]}")
    test = [name for name in ordered if name in wanted]
    excluded = [name for name in ordered if name in dropped]
    train = [name for name in ordered if name not in wanted and name not in dropped]
    if not test:
        raise ValueError("the test list is empty")
    if not train:
        raise ValueError("no training images left after removing the test list")
    return Split("test_list", None, train, test, excluded)


def read_test_list(path: Path) -> list[str]:
    """Một tên ảnh mỗi dòng; bỏ dòng trống và dòng bắt đầu bằng '#'."""
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate names in {path}")
    return names


def registered_image_names(scene_dir: Path) -> list[str]:
    """Tên các ảnh đã đăng ký trong `<cảnh>/sparse/0/images.bin`."""
    images = read_images_binary(Path(scene_dir) / "sparse" / "0" / "images.bin")
    return sorted(image.name for image in images.values())


def make_split(scene_dir: Path, test_every: int = 8, test_list: Path | None = None,
               exclude_list: Path | None = None) -> Split:
    if exclude_list is not None and test_list is None:
        raise ValueError("an exclude list is only used together with a test list")
    names = registered_image_names(scene_dir)
    if test_list is not None:
        excluded = read_test_list(exclude_list) if exclude_list is not None else []
        return list_split(names, read_test_list(test_list), excluded)
    return every_nth_split(names, test_every)


def segment_split(names: list[str], blocks: int = 6, test_fraction: float = 0.125,
                  gap: int = 3) -> tuple[list[str], list[str], list[dict]]:
    """(test, excluded, các đoạn): `blocks` đoạn test liền nhau đặt đều trên dãy khung đã sắp theo tên (frames.py đặt
    tên 00001.jpg... theo thứ tự thời gian của video), mỗi đoạn round(N × test_fraction / blocks) khung; `gap` khung
    mỗi bên đoạn test bị bỏ ra (không train, không chấm). Giữa hai đoạn luôn còn ít nhất một khung train."""
    if blocks < 1:
        raise ValueError("blocks must be >= 1")
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5")
    if gap < 0:
        raise ValueError("gap must be >= 0")
    ordered = sorted(names)
    count = len(ordered)
    length = max(1, round(count * test_fraction / blocks))
    period = count / blocks
    if length + 2 * gap + 1 > period:
        raise ValueError(f"{count} images are too few for {blocks} test segments of {length} images with a gap of "
                         f"{gap}: use fewer segments (--blocks) or a smaller gap (--gap)")
    test, excluded, segments = [], [], []
    for index in range(blocks):
        start = int(round((index + 0.5) * period - length / 2))
        stop = start + length
        before = ordered[max(0, start - gap):start]
        after = ordered[stop:min(count, stop + gap)]
        test.extend(ordered[start:stop])
        excluded.extend(before + after)
        segments.append({"test": ordered[start:stop], "excluded": before + after})
    if len(set(test)) != len(test) or set(test) & set(excluded) or len(set(excluded)) != len(excluded):
        raise ValueError("test segments overlap: use fewer segments or a smaller gap")  # phòng hờ, không xảy ra
    if count - len(test) - len(excluded) < max(10, count // 2):
        raise ValueError(f"only {count - len(test) - len(excluded)} of {count} images would be left for training: "
                         "lower --test-fraction or --gap")
    return test, excluded, segments


def frame_times(scene_dir: Path) -> dict[str, float]:
    """Tên khung -> thời điểm trong video (giây), từ frames_report.json của frames.py; {} khi nguồn là ảnh chụp."""
    path = Path(scene_dir) / "frames_report.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if report.get("source_type") != "video":
        return {}
    return {item["name"]: float(item["time_s"]) for item in report.get("frames", [])
            if isinstance(item, dict) and "name" in item and item.get("time_s") is not None}


def write_segments(scene_dir: Path, out_dir: Path, blocks: int = 6, test_fraction: float = 0.125,
                   gap: int = 3) -> dict:
    """Ghi `test.txt`, `exclude.txt` (cho src.training.run --test-list / --exclude-list) và `segments.json`."""
    names = registered_image_names(scene_dir)
    test, excluded, segments = segment_split(names, blocks, test_fraction, gap)
    times = frame_times(scene_dir)
    train = sorted(set(names) - set(test) - set(excluded))
    order = {name: index for index, name in enumerate(sorted(names))}
    nearest = []  # khoảng cách từ mỗi khung test tới khung train gần nhất (theo thứ tự khung, và theo giây nếu có)
    for name in test:
        frames = min(abs(order[name] - order[other]) for other in train)
        seconds = (min(abs(times[name] - times[other]) for other in train if other in times)
                   if name in times and any(other in times for other in train) else None)
        nearest.append((frames, seconds))
    seconds = [value for _, value in nearest if value is not None]
    info = {
        "rule": "segments", "scene": str(Path(scene_dir).resolve()), "images": len(names),
        "blocks": blocks, "test_fraction": test_fraction, "gap": gap,
        "num_test": len(test), "num_excluded": len(excluded), "num_train": len(train),
        "min_frames_to_train": min(frames for frames, _ in nearest),
        "min_seconds_to_train": round(min(seconds), 3) if seconds and len(seconds) == len(nearest) else None,
        "source": "video" if times else "ảnh chụp hoặc không có frames_report.json (thứ tự theo tên ảnh)",
        "segments": [{"test": [item["test"][0], item["test"][-1]], "num_test": len(item["test"]),
                      "num_excluded": len(item["excluded"]),
                      "time_s": ([times[item["test"][0]], times[item["test"][-1]]]
                                 if item["test"][0] in times and item["test"][-1] in times else None)}
                     for item in segments],
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    header = (f"# segments: {blocks} doan, test_fraction {test_fraction}, gap {gap}, {len(names)} anh "
              f"(src.training.split segments)\n")
    (out_dir / "test.txt").write_text(header + "".join(f"{name}\n" for name in test), encoding="utf-8")
    (out_dir / "exclude.txt").write_text(header + "".join(f"{name}\n" for name in excluded), encoding="utf-8")
    (out_dir / "segments.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def write_split(split: Split, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_split(path: Path) -> Split:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Split(data["rule"], data.get("test_every"), list(data["train"]), list(data["test"]),
                 list(data.get("excluded", [])))


def _link_or_copy(source: Path, target: Path) -> str:
    try:
        target.symlink_to(source, target_is_directory=source.is_dir())
        return "symlink"
    except OSError:
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        return "copy"


def make_train_view(scene_dir: Path, view_dir: Path, split: Split, min_train_views: int = 2) -> dict:
    """Thư mục dữ liệu cho gsplat khi tập test là danh sách cho trước.

    `images/`, `cameras.bin`, `images.bin` giữ nguyên (liên kết tới cảnh gốc; ảnh test vẫn cần pose để
    render khi đánh giá). `points3D.bin` chỉ giữ các điểm được ít nhất `min_train_views` ảnh train khác
    nhau quan sát, tức là dựng được mà không cần ảnh test.
    """
    model = Path(scene_dir) / "sparse" / "0"
    images = read_images_binary(model / "images.bin")
    train_names = set(split.train)
    train_ids = {image_id for image_id, image in images.items() if image.name in train_names}
    points = read_points3d_binary(model / "points3D.bin")
    kept = [point for point in points
            if len({int(image_id) for image_id in point.track[:, 0]} & train_ids) >= min_train_views]
    if not kept:
        raise ValueError("no SfM point is observed by enough training images")

    out_model = Path(view_dir) / "sparse" / "0"
    out_model.mkdir(parents=True, exist_ok=True)
    methods = {_link_or_copy((Path(scene_dir) / "images").resolve(), Path(view_dir) / "images")}
    for name in ("cameras.bin", "images.bin"):
        methods.add(_link_or_copy((model / name).resolve(), out_model / name))
    write_points3d_binary(out_model / "points3D.bin", kept)
    return {"data_dir": str(view_dir), "min_train_views": min_train_views, "points_total": len(points),
            "points_kept": len(kept), "link_method": sorted(methods)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Danh sách ảnh test / ảnh bỏ ra cho src.training.run "
                                                 "(--test-list, --exclude-list)")
    commands = parser.add_subparsers(dest="command", required=True)
    seg = commands.add_parser("segments", help="các đoạn liền của video làm tập test, bỏ vài khung mỗi bên đoạn")
    seg.add_argument("--scene", required=True, type=Path, help="thư mục cảnh có sparse/0 (outputs/custom_scenes/<cảnh>/colmap)")
    seg.add_argument("--out", required=True, type=Path, help="thư mục ghi test.txt, exclude.txt, segments.json")
    seg.add_argument("--blocks", type=int, default=6, help="số đoạn test (mặc định 6)")
    seg.add_argument("--test-fraction", type=float, default=0.125, help="phần ảnh làm test (mặc định 1/8, như mỗi ảnh thứ 8)")
    seg.add_argument("--gap", type=int, default=3, help="số khung bỏ ra mỗi bên đoạn test (mặc định 3)")
    args = parser.parse_args(argv)
    try:
        info = write_segments(args.scene, args.out, args.blocks, args.test_fraction, args.gap)
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    seconds = info["min_seconds_to_train"]
    print(f"segments: {info['num_test']} test / {info['num_excluded']} bo ra / {info['num_train']} train "
          f"tren {info['images']} anh; khung test cach khung train it nhat {info['min_frames_to_train']} khung"
          + (f" ({seconds} giay)" if seconds is not None else ""))
    if seconds is None:
        print("NOTE: khong co thoi diem cua video (frames_report.json): thu tu lay theo ten anh")
    print(f"-> {args.out / 'test.txt'}, {args.out / 'exclude.txt'}, {args.out / 'segments.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
