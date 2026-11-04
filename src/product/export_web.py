"""Xuất mô hình chính của một phòng cho web và đưa phòng vào thư viện (manifest.json).

Đọc <phòng>/model (do train_main.py ghi: train_meta.json, ply/point_cloud_<N>.ply) và <phòng>/colmap (ảnh, COLMAP,
báo cáo frames_report.json / sfm_report.json). Ghi vào thư mục web (web.dir của configs/product.yaml, mặc định
${WS}/outputs/product/web). Đó là một thư mục tĩnh: phục vụ bằng bất kỳ web server tĩnh nào, hoặc chép nguyên sang nơi
đặt trang web. Mọi đường dẫn trong manifest.json tính từ chính thư mục đó.
  manifest.json                  danh sách phòng, phòng train gần nhất trước (dựng lại từ rooms/*/room.json mỗi lần xuất)
  rooms/<phòng>/room.json        mục của phòng trong manifest.json
  rooms/<phòng>/model.splat      mô hình (.splat của antimatter15: 32 byte mỗi Gaussian)
  rooms/<phòng>/cameras.json     vị trí camera lúc quay / chụp, hướng lên, tâm, biên của phòng (cho các chế độ xem)
  rooms/<phòng>/thumbnail.jpg    ảnh đại diện (ảnh đầu vào ở giữa đường quay, cạnh dài 640 px)
Định dạng từng file và hệ tọa độ: docs/design/product.md mục 4.

Các bước của export_room(), theo thứ tự các phần trong file:
  1. mô hình đã train    trained_model                 train_meta.json phải OK; lấy cấu hình đã thực sự train
  2. model.splat         splat_records, write_model    PLY -> 32 byte mỗi Gaussian
  3. cameras.json        read_cameras, camera_layout   Parser của gsplat: cùng hệ tọa độ với mô hình đã train
  4. thumbnail.jpg       write_thumbnail
  5. thư viện            room_entry, rebuild_manifest  rooms/<phòng>/room.json -> manifest.json
Cảnh báo (`warnings` trong room.json) hiện cho người xem trên web nên viết bằng tiếng Việt; log cho người chạy pod viết
tiếng Anh như phần còn lại của repo.

Cách dùng (Python của env gs-gsplat; thường do scripts/build_room.sh gọi):
  python -m src.product.export_web --room <phòng> --model <phòng>/model --scene <phòng>/colmap
                                   [--name "Tên hiển thị"] [--web-dir <thư mục>] [--progress <phòng>/progress.json]
                                   [--config configs/product.yaml] [--no-cameras]
  python -m src.product.export_web --rebuild-manifest [--web-dir <thư mục>]   # vd. sau khi xóa thư mục một phòng
--no-cameras: không đọc camera bằng Parser của gsplat (máy không có env gs-gsplat); web chỉ còn chế độ xoay quanh.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

from src.product import config as product_config
from src.product import ply as ply_io
from src.product import progress as room_progress
from src.product import train_main

SH_C0 = 0.28209479177387814  # hệ số SH bậc 0 (gsplat/exporter.py: sh2rgb)
SPLAT_DTYPE = [("position", "<f4", (3,)), ("scale", "<f4", (3,)), ("rgba", "u1", (4,)), ("rotation", "u1", (4,))]
SPLAT_FIELDS = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3")
FORMAT_VERSION = 1
THUMBNAIL_SIDE = 640
BOUNDS_PERCENTILES = (1.0, 99.0)
DECIMALS = 5
WARN_OTHER_CONFIG = ("Mô hình được train với cấu hình khác configs/product.yaml hiện tại; chạy lại scripts/build_room.sh "
                     "để train lại")
WARN_NO_CAMERAS = "Không đọc được vị trí camera ({error}): chỉ xem xoay quanh, tắt đi bộ và tham quan"
WARN_MIRRORED = ("Hệ tọa độ sau chuẩn hóa của gsplat bị lật: phòng hiện như ảnh gương; dựng lại COLMAP "
                 "(scripts/build_room.sh ... --redo-colmap)")


class ExportError(RuntimeError):
    """Không xuất được; thông điệp chỉ rõ nguyên nhân."""


def now() -> str:
    return room_progress.now()


def say(text: str) -> None:
    print(text, flush=True)


def write_json(path: Path, data) -> None:
    """Ghi file tạm rồi đổi tên: trang web không bao giờ đọc phải file đang ghi dở."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _rounded(values) -> list[float]:
    return [round(float(value), DECIMALS) + 0.0 for value in values]  # + 0.0: không ghi -0.0


# ================================================================================== 1. mô hình đã train
def trained_model(model_dir: Path, cfg: dict) -> tuple[dict, dict, Path, list[str]]:
    """(train_meta.json, cấu hình đã thực sự train, file PLY, cảnh báo). Chưa train xong thì dừng."""
    meta = train_main.read_meta(model_dir)
    if not meta or meta.get("status") != "OK":
        raise ExportError(f"{model_dir}: no finished training (train_meta.json status "
                          f"{(meta or {}).get('status', 'missing')}); run src.product.train_main first")
    trained = meta["config"] if isinstance(meta.get("config"), dict) else train_main.training_key(cfg)
    warnings = [] if trained == train_main.training_key(cfg) else [WARN_OTHER_CONFIG]
    ply_file = Path(model_dir) / ((meta.get("files") or {}).get("ply") or train_main.ply_path(model_dir, cfg))
    return meta, trained, ply_file, warnings


# ========================================================================================= 2. model.splat
def splat_records(vertices):
    """PLY 3DGS (mảng có cấu trúc) -> bản ghi .splat theo thứ tự quan trọng giảm dần, kèm thống kê.

    Mỗi bản ghi 32 byte, như `splat2splat_bytes` của gsplat @6e8c837 (gsplat/exporter.py): vị trí (3 float32),
    exp(scale) (3 float32), RGBA (4 uint8: 0,5 + C0 * f_dc và sigmoid(opacity), nhân 255, kẹp 0-255, cắt phần lẻ),
    quaternion w x y z chuẩn hóa (4 uint8: q * 128 + 128, kẹp, cắt phần lẻ). Khác gsplat ở hai điểm, không đổi hình ảnh:
      - bỏ Gaussian có giá trị không hữu hạn (gsplat cũng bỏ) hoặc quaternion bằng 0, và Gaussian có alpha 0 sau khi
        lượng tử hóa (trong suốt hoàn toàn trong mọi trình xem .splat);
      - xếp theo thể tích × độ đục giảm dần (như convert.py của antimatter15) thay cho thứ tự Morton: trình xem tải dần
        hiện các Gaussian lớn, đục trước. Thứ tự không ảnh hưởng ảnh render (trình xem tự sắp theo độ sâu mỗi khung).
    """
    import numpy as np

    names = vertices.dtype.names or ()
    missing = [name for name in SPLAT_FIELDS if name not in names]
    if missing:
        raise ExportError(f"the PLY file is not a 3DGS model (missing {', '.join(missing)})")

    def column(*fields):
        return np.stack([vertices[field].astype(np.float32) for field in fields], axis=1)

    position = column("x", "y", "z")
    log_scale = column("scale_0", "scale_1", "scale_2")
    dc = column("f_dc_0", "f_dc_1", "f_dc_2")
    opacity = vertices["opacity"].astype(np.float32)
    quat = column("rot_0", "rot_1", "rot_2", "rot_3")
    # công thức của gsplat, tính bằng float32 như gsplat
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scale = np.exp(log_scale)
        alpha = np.float32(1.0) / (np.float32(1.0) + np.exp(-opacity))
        norm = np.linalg.norm(quat, axis=1)
        rgba = np.concatenate([dc * np.float32(SH_C0) + np.float32(0.5), alpha[:, None]], axis=1) * np.float32(255)
        rotation = quat / norm[:, None] * np.float32(128) + np.float32(128)
    # bỏ Gaussian hỏng và Gaussian trong suốt hoàn toàn
    valid = (np.isfinite(position).all(axis=1) & np.isfinite(scale).all(axis=1) & np.isfinite(dc).all(axis=1)
             & np.isfinite(opacity) & np.isfinite(quat).all(axis=1) & (norm > 0))
    rgba_u8 = np.clip(np.nan_to_num(rgba), 0, 255).astype(np.uint8)
    visible = rgba_u8[:, 3] > 0
    keep = valid & visible
    # xếp: log(thể tích × sigmoid(opacity)) giảm dần (tính trong log để không tràn số)
    importance = log_scale.astype(np.float64).sum(axis=1) - np.logaddexp(0.0, -opacity.astype(np.float64))
    order = np.flatnonzero(keep)[np.argsort(-importance[keep], kind="stable")]
    records = np.empty(len(order), dtype=SPLAT_DTYPE)
    records["position"] = position[order]
    records["scale"] = scale[order]
    records["rgba"] = rgba_u8[order]
    records["rotation"] = np.clip(rotation[order], 0, 255).astype(np.uint8)
    stats = {"input": int(len(vertices)), "kept": int(len(order)), "dropped_invalid": int((~valid).sum()),
             "dropped_transparent": int((valid & ~visible).sum())}
    return records, stats


def write_splat(records, path: Path) -> str:
    """Ghi file .splat (file tạm rồi đổi tên); trả về SHA-256 của nội dung."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = records.tobytes()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def write_model(ply_file: Path, target: Path):
    """PLY của train_main.py -> model.splat. Trả về (bản ghi, thống kê, SHA-256)."""
    try:
        records, stats = splat_records(ply_io.read_vertices(ply_file))
    except (OSError, ply_io.PlyError) as error:
        raise ExportError(str(error)) from None
    if not len(records):
        raise ExportError(f"{ply_file}: no visible Gaussian to export")
    digest = write_splat(records, target)
    say(f"[export_web] {stats['kept']} / {stats['input']} Gaussians -> {target} "
        f"({stats['dropped_invalid']} invalid, {stats['dropped_transparent']} fully transparent left out)")
    return records, stats, digest


# ======================================================================================== 3. cameras.json
def load_gsplat_parser(examples: Path):
    """Lớp Parser của examples/datasets/colmap.py (gsplat, commit ghim): chính lớp simple_trainer.py dùng để đọc
    COLMAP và chuẩn hóa hệ tọa độ, nên camera ra đúng hệ tọa độ của mô hình đã train."""
    examples = Path(examples).resolve()
    if not (examples / "datasets" / "colmap.py").is_file():
        raise ExportError(f"{examples}/datasets/colmap.py not found (install gsplat with scripts/setup/setup_gsplat.sh)")
    train_main.check_examples_imports(examples)
    if str(examples) not in sys.path:
        sys.path.insert(0, str(examples))
    try:
        module = importlib.import_module("datasets.colmap")
    except Exception as error:  # thiếu thư viện của env gs-gsplat (torch, pycolmap...): báo rõ, không dừng ở traceback
        raise ExportError(f"cannot import datasets.colmap from {examples}: {type(error).__name__}: {error}") from None
    origin = Path(getattr(module, "__file__", "") or "").resolve()
    if examples not in origin.parents:
        raise ExportError(f"datasets.colmap was imported from {origin}, not from {examples}")
    return module.Parser


def camera_layout(camtoworlds, names: list[str], K, size, positions=None) -> dict:
    """Nội dung cameras.json từ camera-to-world (quy ước OpenCV: x phải, y xuống, z nhìn tới) trong hệ tọa độ của mô
    hình. `positions`: tâm các Gaussian (cho biên của phòng)."""
    import numpy as np

    c2w = np.asarray(camtoworlds, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4) or len(c2w) != len(names) or not len(names):
        raise ExportError("camera poses and image names do not match")

    def unit(vectors):
        return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)

    centers = c2w[:, :3, 3]
    forwards = unit(c2w[:, :3, 2])   # trục z của camera: hướng nhìn
    ups = unit(-c2w[:, :3, 1])       # -y của camera: hướng lên
    # forward + up chỉ đủ dựng lại camera khi hệ tọa độ thuận tay phải. Chuẩn hóa của gsplat có một nhánh phản xạ
    # (similarity_from_cameras khi hướng lên của thế giới COLMAP đúng bằng +y): khi đó mô hình bị lật như ảnh gương.
    right_handed = bool((np.linalg.det(c2w[:, :3, :3]) > 0).all())
    up = unit(ups.mean(axis=0))  # người quay giữ máy thẳng: trung bình hướng lên của các camera
    width, height = (int(value) for value in size)
    fx, fy, cx, cy = (float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2]))
    layout = {
        "format_version": FORMAT_VERSION,
        "coordinate_frame": "model",
        "camera_convention": "opencv",
        "units": "relative",
        "right_handed": right_handed,
        "up": _rounded(up),
        "center": _rounded(np.median(centers, axis=0)),
        "scene_radius": round(float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).max()), DECIMALS),
        "bounds": None,
        "intrinsics": {"width": width, "height": height, "fx": round(fx, 3), "fy": round(fy, 3), "cx": round(cx, 3),
                       "cy": round(cy, 3), "fov_y_deg": round(math.degrees(2 * math.atan(height / (2 * fy))), 3)},
        "default_view": len(names) // 2,
        "frames": [{"name": name, "position": _rounded(center), "forward": _rounded(forward), "up": _rounded(cam_up)}
                   for name, center, forward, cam_up in zip(names, centers, forwards, ups)],
    }
    if positions is not None and len(positions):
        low, high = np.percentile(np.asarray(positions, dtype=np.float64), BOUNDS_PERCENTILES, axis=0)
        layout["bounds"] = {"min": _rounded(low), "max": _rounded(high), "percentiles": list(BOUNDS_PERCENTILES)}
    return layout


def read_cameras(scene: Path, cfg: dict, positions=None) -> dict:
    """Camera của phòng, đọc như simple_trainer.py đọc lúc train (cùng data_factor, cùng chuẩn hóa)."""
    Parser = load_gsplat_parser(train_main.examples_dir(cfg))
    args = {product_config.flag_name(key): value for key, value in cfg["model"]["args"].items()}
    normalize = bool(args.get("normalize-world-space", True))  # mặc định của simple_trainer.py: True
    parser = Parser(data_dir=str(scene), factor=cfg["training"]["data_factor"], normalize=normalize,
                    test_every=cfg["training"]["test_every"])
    camera = parser.camera_ids[0]  # ảnh từ nhiều máy: góc nhìn theo máy của ảnh đầu tiên
    return camera_layout(parser.camtoworlds, list(parser.image_names), parser.Ks_dict[camera],
                         parser.imsize_dict[camera], positions)


def write_cameras(scene: Path, cfg: dict, positions, target: Path) -> tuple[list[str], list[str]]:
    """cameras.json; trả về (tên ảnh theo thứ tự, cảnh báo). Không đọc được camera thì phòng vẫn xem được (xoay
    quanh): trả về danh sách tên rỗng kèm cảnh báo, thay vì bỏ cả lần xuất."""
    try:
        layout = read_cameras(scene, cfg, positions)
    except Exception as error:  # Parser của gsplat có thể báo đủ loại lỗi (assert, ValueError...)
        return [], [WARN_NO_CAMERAS.format(error=f"{type(error).__name__}: {error}")]
    write_json(target, layout)
    return [frame["name"] for frame in layout["frames"]], [] if layout["right_handed"] else [WARN_MIRRORED]


# ======================================================================================= 4. thumbnail.jpg
def write_thumbnail(scene: Path, names: list[str], target: Path) -> bool:
    """Ảnh đầu vào ở giữa đường quay, thu về cạnh dài 640 px (không có Pillow: chép nguyên ảnh JPEG)."""
    images = Path(scene) / "images"
    if not names and images.is_dir():
        names = sorted(path.name for path in images.iterdir() if path.is_file())
    if not names:
        return False
    source = images / names[len(names) // 2]
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        from PIL import Image

        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((THUMBNAIL_SIDE, THUMBNAIL_SIDE))
            image.save(tmp, "JPEG", quality=85)
    except ImportError:
        if source.suffix.lower() not in (".jpg", ".jpeg"):
            return False
        shutil.copyfile(source, tmp)
    except OSError as error:
        tmp.unlink(missing_ok=True)
        say(f"WARN: no thumbnail ({source}: {error})")
        return False
    os.replace(tmp, target)
    return True


# ============================================================================================ 5. thư viện
def source_info(scene: Path) -> dict:
    """Nguồn của phòng (video / ảnh, số ảnh, đánh giá COLMAP) từ các báo cáo của bước chuẩn bị ảnh."""
    frames = read_json(Path(scene) / "frames_report.json")
    sfm = read_json(Path(scene) / "sfm_report.json")
    kind = {"video": "video", "images": "photos"}.get(frames.get("source_type"))
    info = {
        "kind": kind,
        "input_images": sfm.get("input_images", frames.get("selected")),
        "registered_images": (sfm.get("model") or {}).get("registered_images"),
        "sfm_verdict": sfm.get("verdict"),
    }
    if kind == "video":
        info["video_duration_s"] = frames.get("duration_s")
    return info


def room_entry(room: str, name: str, cfg: dict, meta: dict, trained: dict, records, stats: dict, digest: str,
               has_cameras: bool, has_thumbnail: bool, scene: Path, warnings: list[str]) -> dict:
    """Mục của phòng trong manifest.json (docs/design/product.md mục 4.2)."""
    trained_model_cfg, trained_training = trained.get("model") or {}, trained.get("training") or {}
    return {
        "id": room,
        "name": name,
        "version": digest[:12],
        "built_at": now(),
        "format": cfg["web"]["format"],
        "files": {
            "model": f"rooms/{room}/model.splat",
            "cameras": f"rooms/{room}/cameras.json" if has_cameras else None,
            "thumbnail": f"rooms/{room}/thumbnail.jpg" if has_thumbnail else None,
        },
        "num_gaussians": stats["kept"],
        "size_bytes": int(records.nbytes),
        "viewer_modes": ["orbit", "walk", "tour"] if has_cameras else ["orbit"],
        "source": source_info(scene),
        "model": {  # cấu hình mô hình đã thực sự train (train_meta.json)
            "name": trained_model_cfg.get("name"),
            "cap_max": trained_model_cfg.get("cap_max"),
            "max_steps": trained_training.get("max_steps"),
            "trainer_commit": (trained.get("trainer") or {}).get("commit"),
            "num_gaussians_trained": meta.get("num_gaussians"),
            "train_seconds": meta.get("duration_s"),
            "trained_at": meta.get("finished_at"),
        },
        "metrics": None,  # chưa chấm điểm phòng tự dựng (không có lượt quay kiểm tra): docs/design/product.md mục 5
        "warnings": warnings,
    }


def rebuild_manifest(web_dir: Path) -> dict:
    """manifest.json = mọi rooms/<phòng>/room.json đọc được, phòng train gần nhất trước. Thứ tự theo lúc train, không
    theo lúc xuất: xuất lại một phòng (vd. sau khi sửa code xuất) không đưa phòng đó lên đầu."""
    rooms = []
    for path in sorted((Path(web_dir) / "rooms").glob("*/room.json")):
        entry = read_json(path)
        if entry.get("id") != path.parent.name:
            say(f"WARN: {path} is not a valid room entry; left out of the manifest")
            continue
        rooms.append(entry)
    rooms.sort(key=lambda entry: entry["id"])
    rooms.sort(key=lambda entry: (entry.get("model") or {}).get("trained_at") or entry.get("built_at") or "",
               reverse=True)
    manifest = {"format_version": FORMAT_VERSION, "generated_at": now(), "rooms": rooms}
    write_json(Path(web_dir) / "manifest.json", manifest)
    return manifest


# ================================================================================================= xuất
def export_room(room: str, model_dir: Path, scene: Path, web_dir: Path, cfg: dict, name: str | None = None,
                cameras: bool = True, progress_file: Path | None = None) -> dict:
    """Năm bước xuất một phòng (đầu file); trả về mục của phòng trong manifest.json."""
    def step(message: str) -> None:
        say(f"[export_web] {message}")
        if progress_file is not None:
            try:
                room_progress.update(progress_file, "export", message=message)
            except (OSError, ValueError) as error:
                say(f"WARN: progress file not updated: {error}")

    if not product_config.ROOM_ID.fullmatch(room):
        raise ExportError(f"invalid room id {room!r} (letters, digits, '.', '_', '-'; starts with a letter or digit)")
    room_dir = Path(web_dir) / "rooms" / room
    # 1. mô hình đã train
    meta, trained, ply_file, warnings = trained_model(model_dir, cfg)
    # 2. model.splat
    step("Đổi mô hình sang .splat")
    records, stats, digest = write_model(ply_file, room_dir / "model.splat")
    # 3. cameras.json (file cũ không còn khớp mô hình mới: bỏ khi lần này không có)
    names: list[str] = []
    if cameras:
        step("Đọc vị trí camera")
        names, camera_warnings = write_cameras(scene, cfg, records["position"], room_dir / "cameras.json")
        warnings += camera_warnings
    if not names and (room_dir / "cameras.json").exists():
        (room_dir / "cameras.json").unlink()
    # 4. thumbnail.jpg
    step("Ảnh đại diện")
    has_thumbnail = write_thumbnail(scene, names, room_dir / "thumbnail.jpg")
    # 5. thư viện (tên hiển thị: --name, không có thì giữ tên cũ, không có nữa thì id)
    step("Thêm vào thư viện")
    display_name = name or read_json(room_dir / "room.json").get("name") or room
    entry = room_entry(room, display_name, cfg, meta, trained, records, stats, digest, bool(names), has_thumbnail,
                       scene, warnings)
    write_json(room_dir / "room.json", entry)
    rebuild_manifest(web_dir)
    for warning in warnings:
        say(f"WARN: {warning}")
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Xuất mô hình chính của một phòng cho web (configs/product.yaml)")
    parser.add_argument("--room", help="id của phòng (tên phòng của scripts/build_room.sh)")
    parser.add_argument("--model", type=Path, help="thư mục mô hình do train_main.py ghi, vd. <phòng>/model")
    parser.add_argument("--scene", type=Path, help="thư mục phòng đã chuẩn bị: images/, sparse/0/, báo cáo")
    parser.add_argument("--name", help="tên hiển thị trên web (mặc định: giữ tên cũ, hoặc id)")
    parser.add_argument("--web-dir", type=Path, help="thư mục web (mặc định web.dir của configs/product.yaml)")
    parser.add_argument("--config", type=Path, default=product_config.DEFAULT_CONFIG)
    parser.add_argument("--ws", default=None)
    parser.add_argument("--progress", type=Path, default=None)
    parser.add_argument("--no-cameras", action="store_true", help="không xuất cameras.json (web chỉ xoay quanh)")
    parser.add_argument("--rebuild-manifest", action="store_true", help="chỉ dựng lại manifest.json")
    args = parser.parse_args(argv)
    try:
        cfg = product_config.load(args.config, args.ws)
    except product_config.ConfigError as error:
        say(f"ERROR: {args.config}: {error}")
        return 2
    web_dir = (args.web_dir or Path(cfg["web"]["dir"])).resolve()
    if args.rebuild_manifest:
        manifest = rebuild_manifest(web_dir)
        say(f"[export_web] {web_dir / 'manifest.json'}: {len(manifest['rooms'])} room(s)")
        return 0
    if not (args.room and args.model and args.scene):
        parser.error("--room, --model and --scene are required (or --rebuild-manifest)")
    try:
        entry = export_room(args.room, args.model.resolve(), args.scene.resolve(), web_dir, cfg, args.name,
                            cameras=not args.no_cameras, progress_file=args.progress)
    except (ExportError, OSError) as error:
        say(f"ERROR: export failed: {error}")
        return 1
    say(f"[export_web] OK: {entry['name']} ({entry['num_gaussians']} Gaussians, "
        f"{entry['size_bytes'] / 1e6:.1f} MB) -> {web_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
