"""Depth prior cho biến thể `mono` (src/training/trainer_hooks.py): độ sâu tương đối do Depth Anything V2 Small ước lượng
cho từng ảnh đã đăng ký của một cảnh, tính một lần trước khi train, dùng chung cho mọi chế độ và seed của cảnh đó.
src/training/run.py gọi module này thành bước `mono` (log mono_depth.log) trước bước train của lượt có `sr3d.mono`.

    python -m src.training.mono_depth --scene <cảnh có images/ và sparse/0> --fingerprint <dấu sparse/0> \\
        --cache <thư mục cache> --split <lượt>/split.json --record <lượt>/mono_depth.json \\
        --depth-anything-dir /workspace/code/depth_anything_v2      (env gs-gsplat, cần GPU)

Các bước:
1. Đọc model COLMAP, kèm điểm 2D của từng ảnh. Camera phải không méo (PINHOLE / SIMPLE_PINHOLE): với camera méo, gsplat
   khử méo rồi cắt ảnh, nên độ sâu tính trên ảnh gốc sẽ lệch với ảnh train.
2. Mỗi ảnh (đọc bỏ qua thẻ xoay EXIF, như imageio mà gsplat dùng): Depth Anything V2 Small (code chính thức @a561b84;
   trọng số Apache-2.0, kiểm SHA-256) cho bản đồ độ sâu tương đối (giá trị lớn = gần, như nghịch đảo độ sâu). Bản đồ được thu về cạnh dài 768 px (mạng chạy ở cạnh ngắn 518 px nên
   gần như không mất chi tiết), chia cho giá trị lớn nhất, lưu float16 `<tên ảnh>.npy`. Lúc train, bản đồ được nội suy
   về đúng kích thước ảnh train.
3. Độ tin cậy từng ảnh: hệ số Pearson giữa bản đồ và nghịch đảo độ sâu (1/z) của các điểm SfM mà ảnh đó quan sát được
   (cần ít nhất 30 điểm). Hệ số âm thì thử đảo bản đồ (1/bản đồ), lấy cách cho hệ số cao hơn. Ảnh có hệ số < 0,15 không
   dùng prior; ảnh không đủ điểm SfM vẫn dùng (không kiểm được), như notebook v2 của nhóm.
4. Kiểm cả cảnh, trên các ảnh train của split.json: trung vị các hệ số ≥ 0,3 và có ít nhất một ảnh dùng được. Không đạt
   thì bước `mono` báo lỗi (mã thoát 1) và lượt đó không train: không có chuyện lượt "mono" lặng lẽ train như mcmc thường.
Cache được dùng lại khi manifest.json khớp: cùng mô hình COLMAP (dấu SHA-256 của sparse/0), cùng mô hình ước lượng và
tham số, đủ file. Kết quả kiểm (bước 4) ghi vào --record.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from src.reconstruction.colmap_model import (
    INVALID_POINT3D_ID,
    model_exists,
    qvec_to_rotation,
    read_cameras_binary,
    read_images_binary,
    read_points3d_positions,
)

# Ghim trong configs/environment/gs-gsplat.conf (DA2_*); scripts/setup/setup_depth_anything.sh tải về đúng các bản này.
DA2_COMMIT = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
WEIGHTS_FILE = "checkpoints/depth_anything_v2_vits.pth"  # trong thư mục code, như run.py của Depth Anything V2
WEIGHTS_SHA256 = "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
MODEL = {"name": "Depth Anything V2 Small", "encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384],
         "input_size": 518, "commit": DA2_COMMIT, "weights_sha256": WEIGHTS_SHA256}
PARAMS = {"max_side": 768, "min_points": 30, "min_corr": 0.15, "min_median_corr": 0.3}
UNDISTORTED = ("PINHOLE", "SIMPLE_PINHOLE")
MANIFEST = "manifest.json"
VERSION = 1
SHOW = 5  # số tên ảnh in ra trong thông báo


class MonoDepthError(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, data: dict) -> None:
    """Ghi qua file tạm rồi đổi tên: pod tắt giữa chừng không để lại JSON hỏng."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pearson(a, b) -> float:
    """Hệ số tương quan Pearson; 0 khi một bên không đổi (như notebook v2)."""
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def resize_long_side(depth: np.ndarray, max_side: int) -> np.ndarray:
    """Thu nhỏ (INTER_AREA) để cạnh dài nhất không quá max_side; không phóng to."""
    import cv2

    height, width = depth.shape
    scale = max_side / max(height, width)
    depth = np.asarray(depth, dtype=np.float32)
    if scale >= 1.0:
        return depth
    size = (max(1, round(width * scale)), max(1, round(height * scale)))  # cv2: (rộng, cao)
    return cv2.resize(depth, size, interpolation=cv2.INTER_AREA)


def sfm_check(depth: np.ndarray, image, camera, positions: tuple[np.ndarray, np.ndarray]) -> tuple[dict, np.ndarray]:
    """So bản đồ (đã thu nhỏ) với 1/z của các điểm SfM mà ảnh quan sát được; đảo bản đồ khi cách đó tương quan tốt hơn.

    Trả về ({corr, points, flipped}, bản đồ dùng tiếp). corr None khi ảnh có ít hơn PARAMS["min_points"] điểm."""
    height, width = depth.shape
    points = image.points2d
    ids = points["point3d_id"]
    seen = ids != INVALID_POINT3D_ID
    ids, x, y = ids[seen], points["x"][seen], points["y"][seen]
    known_ids, known_xyz = positions
    where = np.clip(np.searchsorted(known_ids, ids), 0, max(len(known_ids) - 1, 0))
    found = known_ids[where] == ids if len(known_ids) else np.zeros(len(ids), dtype=bool)
    xyz, x, y = known_xyz[where[found]], x[found], y[found]
    z = (xyz @ qvec_to_rotation(image.qvec).T + np.asarray(image.tvec, dtype=float))[:, 2]
    # tọa độ COLMAP: tâm pixel góc trái trên là (0,5; 0,5); pixel của bản đồ thu nhỏ chứa điểm đó
    u = np.clip(np.floor(x * width / camera.width).astype(int), 0, width - 1)
    v = np.clip(np.floor(y * height / camera.height).astype(int), 0, height - 1)
    front = z > 1e-6
    count = int(front.sum())
    if count < PARAMS["min_points"]:
        return {"corr": None, "points": count, "flipped": False}, depth
    disparity = 1.0 / z[front]
    corr = pearson(depth[v[front], u[front]], disparity)
    flipped = False
    if corr < 0:  # mô hình trả "độ sâu" thay vì "nghịch đảo độ sâu"
        inverse = 1.0 / np.maximum(depth, 1e-3 * max(float(depth.max()), 1e-12))
        corr_inverse = pearson(inverse[v[front], u[front]], disparity)
        if corr_inverse > corr:
            depth, corr, flipped = inverse.astype(np.float32), corr_inverse, True
    return {"corr": corr, "points": count, "flipped": flipped}, depth


def _expected(fingerprint: str) -> dict:
    return {"version": VERSION, "fingerprint": fingerprint, "model": MODEL, "params": PARAMS}


def cache_is_complete(cache_dir: Path, fingerprint: str) -> bool:
    manifest = read_json(Path(cache_dir) / MANIFEST)
    if not isinstance(manifest, dict) or not manifest.get("complete"):
        return False
    if any(manifest.get(key) != value for key, value in _expected(fingerprint).items()):
        return False
    return all((Path(cache_dir) / entry["file"]).is_file() for entry in (manifest.get("images") or {}).values())


def _clear_cache(cache_dir: Path) -> None:
    """Xóa cache cũ trước khi tính lại. Chỉ xóa thư mục chứa toàn file của module này (.npy, manifest): --cache trỏ
    nhầm vào một thư mục khác thì dừng, không xóa gì."""
    if cache_dir.exists():
        own = (MANIFEST, f".{MANIFEST}.tmp")
        stray = [path for path in cache_dir.rglob("*")
                 if not path.is_dir() and path.suffix != ".npy" and path.name not in own]
        if stray:
            raise MonoDepthError(f"{cache_dir} holds files that are not a depth prior (e.g. {stray[0]}); "
                                 "not clearing it: choose another --cache")
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)


def prepare(scene_dir: Path, cache_dir: Path, fingerprint: str, load_estimator: Callable[[], Callable],
            log: Callable[[str], None] = print) -> dict:
    """Tính (hoặc dùng lại) depth prior cho mọi ảnh đã đăng ký của cảnh; trả về manifest.

    `load_estimator()` trả về hàm ảnh BGR (H, W, 3) uint8 -> bản đồ (H, W); chỉ được gọi khi phải tính lại."""
    scene_dir, cache_dir = Path(scene_dir), Path(cache_dir)
    model_dir = scene_dir / "sparse" / "0"
    if not model_exists(model_dir):
        raise MonoDepthError(f"{model_dir} must contain cameras.bin, images.bin and points3D.bin")
    cameras = read_cameras_binary(model_dir / "cameras.bin")
    distorted = sorted({camera.model for camera in cameras.values() if camera.model not in UNDISTORTED})
    if distorted:
        raise MonoDepthError(f"camera model(s) {distorted} are not undistorted: gsplat undistorts and crops those images, "
                             "so a depth prior computed on the original images would not line up (use colmap_runner, "
                             "which runs image_undistorter)")
    if cache_is_complete(cache_dir, fingerprint):
        manifest = read_json(cache_dir / MANIFEST)
        log(f"[mono] reusing the depth prior in {cache_dir} ({len(manifest['images'])} images, {manifest['written']})")
        return manifest

    images = sorted(read_images_binary(model_dir / "images.bin", with_points2d=True).values(), key=lambda im: im.name)
    if not images:
        raise MonoDepthError(f"no registered images in {model_dir}")
    positions = read_points3d_positions(model_dir / "points3D.bin")
    estimate = load_estimator()
    import cv2

    _clear_cache(cache_dir)  # cache dở dang hoặc của mô hình / tham số khác
    entries = {}
    for count, image in enumerate(images, start=1):
        camera = cameras[image.camera_id]
        path = scene_dir / "images" / image.name
        # bỏ thẻ xoay EXIF như imageio (gsplat đọc ảnh train bằng imageio, không xoay): prior khớp đúng pixel ảnh train
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if bgr is None:
            raise MonoDepthError(f"cannot read {path}")
        if bgr.shape[:2] != (camera.height, camera.width):
            raise MonoDepthError(f"{path} is {bgr.shape[1]}x{bgr.shape[0]} but its COLMAP camera is "
                                 f"{camera.width}x{camera.height}")
        depth = np.asarray(estimate(bgr), dtype=np.float32)
        if depth.shape != bgr.shape[:2] or not np.isfinite(depth).all():
            raise MonoDepthError(f"the depth estimate of {image.name} has shape {depth.shape} or non-finite values")
        check, depth = sfm_check(resize_long_side(depth, PARAMS["max_side"]), image, camera, positions)
        peak = float(depth.max())
        depth = depth / peak if peak > 0 else depth  # Pearson không đổi khi nhân hệ số dương; tránh tràn float16
        relative = Path(image.name).with_suffix(".npy")
        (cache_dir / relative).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / relative, depth.astype(np.float16))
        use = check["corr"] is None or check["corr"] >= PARAMS["min_corr"]
        entries[image.name] = {"file": relative.as_posix(), "use": use, "shape": list(depth.shape), **check}
        if count % 50 == 0 or count == len(images):
            log(f"[mono] {count}/{len(images)} images")
    manifest = {**_expected(fingerprint), "scene_dir": str(scene_dir), "images": entries, "complete": True,
                "written": _now()}
    _write_json(cache_dir / MANIFEST, manifest)
    return manifest


def check_scene(manifest: dict, train_names: list[str]) -> dict:
    """Kiểm prior trên các ảnh train của lượt: trung vị hệ số ≥ PARAMS["min_median_corr"] và có ảnh dùng được."""
    entries = manifest["images"]
    missing = [name for name in train_names if name not in entries]
    if missing:
        raise MonoDepthError(f"{len(missing)} train image(s) are not registered in the COLMAP model of the depth prior, "
                             f"e.g. {missing[:SHOW]}")
    rows = [entries[name] for name in train_names]
    corrs = [row["corr"] for row in rows if row["corr"] is not None]
    median = float(np.median(corrs)) if corrs else None
    used = sum(1 for row in rows if row["use"])
    low = [name for name in train_names if not entries[name]["use"]]
    ok = used > 0 and (median is None or median >= PARAMS["min_median_corr"])
    return {
        "status": "OK" if ok else "FAILED",
        "train_images": len(train_names),
        "used": used,
        "low_corr": low,
        "unchecked": sum(1 for row in rows if row["corr"] is None),
        "flipped": sum(1 for row in rows if row["flipped"]),
        "median_corr": median,
        "thresholds": {key: PARAMS[key] for key in ("min_points", "min_corr", "min_median_corr")},
    }


def checkout_state(code_dir: Path) -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(["git", "-C", str(code_dir), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(code_dir), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip() != ""
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def load_depth_anything(code_dir: Path) -> Callable[[np.ndarray], np.ndarray]:
    """Depth Anything V2 Small từ code chính thức (đúng commit ghim, không sửa) và trọng số đúng SHA-256."""
    code_dir = Path(code_dir).resolve()
    hint = "run bash scripts/setup/setup_depth_anything.sh"
    sha, dirty = checkout_state(code_dir)
    if sha != DA2_COMMIT or dirty:
        raise MonoDepthError(f"{code_dir} is not an unmodified Depth Anything V2 checkout at {DA2_COMMIT[:7]} "
                             f"(found {sha or 'no git checkout'}{', modified' if dirty else ''}); {hint}")
    weights = code_dir / WEIGHTS_FILE
    if not weights.is_file() or file_sha256(weights) != WEIGHTS_SHA256:
        raise MonoDepthError(f"{weights} is missing or not the pinned Depth Anything V2 Small weights; {hint}")
    import torch

    sys.path.insert(0, str(code_dir))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(encoder=MODEL["encoder"], features=MODEL["features"], out_channels=MODEL["out_channels"])
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    # image2tensor của Depth Anything V2 tự đưa ảnh lên GPU khi có: mô hình phải nằm cùng chỗ
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def estimate(bgr: np.ndarray) -> np.ndarray:
        return model.infer_image(bgr, MODEL["input_size"])

    return estimate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", required=True, type=Path, help="thư mục cảnh có images/ và sparse/0/")
    parser.add_argument("--fingerprint", required=True, help="SHA-256 của sparse/0 (src.training.run.scene_fingerprint)")
    parser.add_argument("--cache", required=True, type=Path, help="thư mục lưu depth prior của cảnh (dùng chung)")
    parser.add_argument("--split", required=True, type=Path, help="split.json của lượt (lấy danh sách ảnh train)")
    parser.add_argument("--record", required=True, type=Path, help="file JSON ghi kết quả kiểm cho lượt này")
    parser.add_argument("--depth-anything-dir", required=True, type=Path, help="code Depth Anything V2 (có checkpoints/)")
    args = parser.parse_args(argv)
    record = {"cache": str(args.cache.resolve()), "model": MODEL, "params": PARAMS, "started": _now()}
    try:
        split = read_json(args.split)
        if not isinstance(split, dict) or not split.get("train"):
            raise MonoDepthError(f"{args.split} has no list of train images")
        manifest = prepare(args.scene.resolve(), args.cache.resolve(), args.fingerprint,
                           lambda: load_depth_anything(args.depth_anything_dir))
        record.update(check_scene(manifest, list(split["train"])),
                      manifest_sha256=file_sha256(args.cache.resolve() / MANIFEST))
    except MonoDepthError as error:
        record.update(status="FAILED", error=str(error), finished=_now())
        _write_json(args.record, record)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    record["finished"] = _now()
    _write_json(args.record, record)
    median = "n/a" if record["median_corr"] is None else f"{record['median_corr']:.3f}"
    print(f"[mono] {record['used']}/{record['train_images']} train images use the depth prior | median correlation "
          f"with SfM {median} (needs >= {PARAMS['min_median_corr']}) | flipped {record['flipped']} | "
          f"unchecked (< {PARAMS['min_points']} SfM points) {record['unchecked']}")
    if record["low_corr"]:
        print(f"[mono] no prior (correlation < {PARAMS['min_corr']}) for {len(record['low_corr'])} image(s), "
              f"e.g. {record['low_corr'][:SHOW]}")
    if record["status"] != "OK":
        print("ERROR: the depth prior does not agree with the SfM points of this scene; the mono run is not trained "
              f"(see {args.record})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
