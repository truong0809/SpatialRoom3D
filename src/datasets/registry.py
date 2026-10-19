"""Đọc cấu hình `configs/*.yaml`: nơi duy nhất ghi dữ liệu và kết quả của từng luồng nằm ở đâu (CHỈ ĐỌC).

`configs/base.yaml` ghi phần chung (thư mục code, danh sách luồng, chế độ huấn luyện); mỗi luồng một file
`configs/<tên>.yaml`: Deep Blending, Mip-NeRF 360 (benchmark) và cảnh tự chụp (`custom_scene`). Đường dẫn viết theo
`${WS}`; WS lấy từ `--ws`, biến môi trường WS, hoặc mặc định `/workspace`. Đổi chỗ dữ liệu trên một pod mà không sửa
file trong Git (theo thứ tự ưu tiên tăng dần):
  * `configs/local.yaml` (không đưa lên Git), cùng cấu trúc, chỉ cần các khóa muốn đổi, vd.
    `datasets: {mipnerf360: {root: "${WS}/data/MipNeRF360"}}`;
  * biến môi trường `SR3D_DATA_<TÊN>` cho thư mục gốc, vd. SR3D_DATA_MIPNERF360=/workspace/data/m360.

Kiểm tra (mặc định): với mỗi benchmark: thư mục gốc, từng cảnh (số ảnh trong `images/`, các thư mục ảnh thu nhỏ khi bộ
dữ liệu có, mô hình COLMAP `sparse/0`). Với cảnh tự chụp: từng cảnh có nguồn gốc (ảnh / video trong
`data/custom_scenes/<cảnh>/`), đã chuẩn bị chưa (`outputs/custom_scenes/<cảnh>/colmap/`), kết luận COLMAP. Với mọi
luồng: thư mục kết quả có bao nhiêu lượt chạy. Không tải, không sửa, không dời gì (tải dữ liệu benchmark là bước riêng:
`src/datasets/fetch.py`, theo mục `download` của `configs/<luồng>.yaml`).
Mã thoát: 0 khi mọi luồng có `required: true` đều đủ; 1 nếu thiếu; 2 khi cấu hình không đọc được.

`--get <luồng> <khóa>` in một trường (vd. `root`, `benchmark.tag`) cho script bash (scripts/lib/config.sh).
`--export-benchmark <luồng> --out <file.json>` ghi phần `benchmark` của một luồng (kèm tên luồng, danh sách cảnh) ra
JSON: code chạy trong env conda không có PyYAML (vd. gs-inria) đọc bản JSON này, và bản đó nằm lại trong
`outputs/reports/configs/` làm bằng chứng cấu hình đã dùng. Cần PyYAML: có trong env gs-gsplat; python3 của
template Runpod thường có (JupyterLab cần PyYAML; ô 1 của notebook kiểm tra), chưa kiểm tra trực tiếp.

Cách dùng:
  python3 -m src.datasets.registry                       # mọi luồng
  python3 -m src.datasets.registry --dataset deep_blending --ws /workspace
  python3 -m src.datasets.registry --get mipnerf360 root
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_DIR / "configs"
BASE_NAME = "base.yaml"
LOCAL_NAME = "local.yaml"
ENV_PREFIX = "SR3D_DATA_"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
MODEL_FILES = ("cameras", "images", "points3D")
PREPARED_SUBDIR = "colmap"  # cảnh tự chụp: <results.gsplat>/<cảnh>/colmap (scripts/pipeline/prepare_scene.sh)


def read_yaml(path: Path) -> dict:
    """Một file YAML có gốc là mapping; lỗi đọc / cú pháp thành ValueError có tên file."""
    try:
        import yaml
    except ImportError as error:  # Python của máy thiếu PyYAML: báo cách sửa thay vì traceback
        raise ValueError(f"PyYAML is required to read {path} (python3 -m pip install pyyaml)") from error
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        merged[key] = _merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) \
            else value
    return merged


def load_registry(config_dir: Path = CONFIG_DIR, ws: str = "/workspace", local: Path | None = None,
                  environ: dict | None = None) -> dict:
    """Đọc configs/base.yaml và configs/<luồng>.yaml, ghép file ghi đè cục bộ và biến môi trường, rồi thay `${WS}`.

    Kết quả: {"code_dir", "train_modes", "datasets": {tên: mục}}. `local` mặc định là `configs/local.yaml`
    (bỏ qua nếu không có).
    """
    config_dir = Path(config_dir)
    base = read_yaml(config_dir / BASE_NAME)
    registry = {key: value for key, value in base.items() if key != "datasets"}
    registry["datasets"] = {}
    for name in base.get("datasets") or []:
        registry["datasets"][name] = read_yaml(config_dir / f"{name}.yaml")
    local = config_dir / LOCAL_NAME if local is None else Path(local)
    if local.is_file():
        try:
            override = read_yaml(local)
        except ValueError as error:
            raise ValueError(f"cannot read the local override {local} (fix or delete it): {error}") from error
        extra = sorted(set(override) - {"datasets"})
        if extra:  # vd. train_modes: protocol và chế độ huấn luyện chỉ nằm ở configs/base.yaml (cùng commit với code)
            raise ValueError(f"the local override {local} may only change 'datasets', not {extra} (fix or delete it)")
        datasets = override.get("datasets") or {}
        if not isinstance(datasets, dict) or not all(isinstance(entry, dict) for entry in datasets.values()):
            raise ValueError(f"the local override {local} must map dataset names to settings, e.g. "
                             "'datasets: {mipnerf360: {root: ...}}' (fix or delete it)")
        unknown = sorted(set(datasets) - set(registry["datasets"]))
        if unknown:
            raise ValueError(f"the local override {local} names unknown dataset(s) {unknown} (fix or delete it)")
        registry = _merge(registry, override)
        registry["local_overrides"] = str(local)
    environ = os.environ if environ is None else environ
    for name, entry in registry["datasets"].items():
        value = environ.get(ENV_PREFIX + name.upper())
        if value:
            entry["root"] = value
            entry["root_from_env"] = ENV_PREFIX + name.upper()

    def expand(value):
        if isinstance(value, str):
            return value.replace("${WS}", ws)
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value
    registry = expand(registry)
    for name, entry in registry["datasets"].items():
        if entry.get("kind") == "custom":  # script đặt COLMAP trong thư mục kết quả của từng cảnh: hai khóa phải khớp
            expected = f"{str((entry.get('results') or {}).get('gsplat', '')).rstrip('/')}/<cảnh>/{PREPARED_SUBDIR}"
            if entry.get("prepared") != expected:
                raise ValueError(f"datasets.{name}.prepared must be <results.gsplat>/<cảnh>/{PREPARED_SUBDIR} "
                                 f"({expected}), not {entry.get('prepared')}: change both together")
    return registry


def benchmark_config(name: str, entry: dict) -> dict:
    """Phần `benchmark` của một luồng, kèm tên luồng và danh sách cảnh: đầu vào của các script benchmark và của
    src.evaluation.compare_reference / compare_modes (`--config`, `--reference-config`)."""
    if not isinstance(entry.get("benchmark"), dict):
        raise KeyError(f"dataset {name} has no benchmark section")
    config = {"dataset": name, "scenes": list(entry.get("scenes") or [])}
    if entry.get("indoor_scenes"):
        config["indoor_scenes"] = list(entry["indoor_scenes"])
    config.update(entry["benchmark"])
    return config


def get_field(entry: dict, key: str):
    """Trường lồng nhau theo khóa có dấu chấm; KeyError nếu không có."""
    value = entry
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(key)
        value = value[part]
    return value


def format_field(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def count_images(folder: Path) -> int | None:
    if not folder.is_dir():
        return None
    return sum(1 for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def has_model(model_dir: Path) -> bool:
    return any(all((model_dir / f"{name}.{ext}").is_file() for name in MODEL_FILES) for ext in ("bin", "txt"))


def check_scene(scene_dir: Path, layout: list[str], name: str | None = None) -> dict:
    """Trạng thái một cảnh theo bố cục mong đợi (vd. images, images_2, sparse/0)."""
    name = name or scene_dir.name
    if not scene_dir.is_dir():
        return {"scene": name, "status": "MISSING", "path": str(scene_dir), "missing": list(layout)}
    found, missing = {}, []
    for part in layout:
        if part.startswith("sparse"):
            if has_model(scene_dir / part):
                found[part] = "ok"
            else:
                missing.append(part)
        else:
            count = count_images(scene_dir / part)
            if count:
                found[part] = count
            else:
                missing.append(part)
    return {"scene": name, "status": "OK" if not missing else "INCOMPLETE", "path": str(scene_dir),
            "found": found, "missing": missing}


def count_runs(pattern: str) -> dict:
    """Số lượt chạy dưới phần cố định của mẫu đường dẫn kết quả (phần trước `<...>` đầu tiên)."""
    base = Path(pattern.split("<", 1)[0].rstrip("/"))
    if not base.is_dir():
        return {"root": str(base), "exists": False, "runs": 0}
    runs = {path.parent for name in ("summary.json", "results.json") for path in base.glob(f"**/{name}")
            if ".prev_" not in str(path.relative_to(base))}
    return {"root": str(base), "exists": True, "runs": len(runs)}


def describe_source(folder: Path) -> str | None:
    """Nguồn của một cảnh tự chụp: số ảnh nằm thẳng trong thư mục, hoặc các file video; None nếu không có thư mục."""
    if not folder.is_dir():
        return None
    files = [path for path in folder.iterdir() if path.is_file() and not path.name.startswith(".")]
    photos = sum(1 for path in files if path.suffix.lower() in IMAGE_EXTS)
    videos = [path.name for path in files if path.suffix.lower() in VIDEO_EXTS]
    parts = ([f"{photos} ảnh"] if photos else []) + ([f"video {', '.join(videos)}"] if videos else [])
    return " + ".join(parts) or "không có ảnh / video"


def sfm_verdict(scene_dir: Path) -> str | None:
    try:
        return json.loads((scene_dir / "sfm_report.json").read_text(encoding="utf-8")).get("verdict")
    except (OSError, ValueError, AttributeError):
        return None


def check_dataset(name: str, entry: dict) -> dict:
    root = Path(entry["root"])
    result = {"dataset": name, "kind": entry.get("kind"), "required": bool(entry.get("required")),
              "root": str(root), "root_exists": root.is_dir(), "root_from_env": entry.get("root_from_env"),
              "status_note": entry.get("status"), "scenes": [],
              "results": {key: count_runs(value) for key, value in (entry.get("results") or {}).items()}}
    if entry.get("kind") == "custom":
        out_root = Path((entry.get("results") or {}).get("gsplat", ""))
        names = set()
        for folder in (root, out_root):
            if folder.is_dir():
                names.update(path.name for path in folder.iterdir() if path.is_dir() and not path.name.startswith("."))
        for name in sorted(names):
            prepared = Path(entry["prepared"].replace("<cảnh>", name))
            info = check_scene(prepared, entry.get("layout", []), name=name)
            if info["status"] == "MISSING":
                info["status"] = "CHƯA CHUẨN BỊ"
            info["source"] = describe_source(root / name)
            info["sfm_verdict"] = sfm_verdict(prepared)
            info["runs"] = count_runs(str(out_root / name))["runs"]
            result["scenes"].append(info)
        result["ok"] = True  # cảnh tự chụp có thể chưa có: không phải lỗi
        return result
    result["scenes"] = [check_scene(root / scene, entry.get("layout", [])) for scene in entry.get("scenes", [])]
    result["ok"] = result["root_exists"] and all(scene["status"] == "OK" for scene in result["scenes"])
    return result


def format_result(result: dict) -> list[str]:
    mark = "OK" if result["ok"] else ("THIẾU" if result["required"] else "chưa có")
    lines = [f"== {result['dataset']} ({result['kind']}{', bắt buộc' if result['required'] else ''}): {mark}",
             f"   gốc: {result['root']}" + ("" if result["root_exists"] else "  (không có)")
             + (f"  (từ biến {result['root_from_env']})" if result.get("root_from_env") else "")]
    if result.get("status_note"):
        lines.append(f"   ghi chú: {result['status_note']}")
    for scene in result["scenes"]:
        parts = []
        if "source" in scene:
            parts.append(f"nguồn: {scene['source'] or 'không có trong data/custom_scenes'}")
        parts += [f"{key}={value}" for key, value in (scene.get("found") or {}).items()]
        if scene.get("missing") and scene["status"] != "CHƯA CHUẨN BỊ":
            parts.append("thiếu " + ", ".join(scene["missing"]))
        if scene.get("sfm_verdict"):
            parts.append(f"COLMAP {scene['sfm_verdict']}")
        if "runs" in scene:
            parts.append(f"{scene['runs']} lượt train")
        lines.append(f"   {scene['scene']:<12} {scene['status']:<13} {' | '.join(parts)}")
    if result["kind"] == "custom" and not result["scenes"]:
        lines.append("   (chưa có cảnh nào)")
    for key, info in result["results"].items():
        lines.append(f"   kết quả {key}: {info['root']} " + (f"({info['runs']} lượt)" if info["exists"] else "(chưa có)"))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cấu hình dữ liệu configs/*.yaml (chỉ đọc)")
    parser.add_argument("--configs", type=Path, default=CONFIG_DIR, help="thư mục cấu hình (mặc định configs/ của repo)")
    parser.add_argument("--ws", default=os.environ.get("WS") or "/workspace", help="thư mục gốc trên pod (WS)")
    parser.add_argument("--dataset", action="append", help="chỉ kiểm tra luồng này (lặp lại được)")
    parser.add_argument("--json", action="store_true", help="in kết quả dạng JSON")
    parser.add_argument("--get", nargs=2, metavar=("DATASET", "KEY"),
                        help="in một trường của luồng (vd. root, benchmark.tag) rồi thoát")
    parser.add_argument("--export-benchmark", metavar="DATASET", help="ghi phần benchmark của luồng ra JSON (--out)")
    parser.add_argument("--out", type=Path, help="file JSON cho --export-benchmark")
    args = parser.parse_args(argv)
    ws = args.ws.rstrip("/") or "/"
    try:
        registry = load_registry(args.configs, ws)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    datasets = registry.get("datasets", {})
    wanted = args.get[0] if args.get else args.export_benchmark
    if wanted is not None and wanted not in datasets:
        print(f"ERROR: unknown dataset {wanted!r}; known: {sorted(datasets)}", file=sys.stderr)
        return 2
    if args.get:
        name, key = args.get
        try:
            print(format_field(get_field(datasets[name], key)))
        except KeyError:
            print(f"ERROR: dataset {name} has no field {key!r}", file=sys.stderr)
            return 1
        return 0
    if args.export_benchmark:
        if args.out is None:
            print("ERROR: --export-benchmark needs --out <file.json>", file=sys.stderr)
            return 2
        try:
            config = benchmark_config(args.export_benchmark, datasets[args.export_benchmark])
        except KeyError as error:
            print(f"ERROR: {error.args[0]}", file=sys.stderr)
            return 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(args.out)
        return 0
    names = args.dataset or list(datasets)
    unknown = [name for name in names if name not in datasets]
    if unknown:
        print(f"ERROR: unknown dataset(s) {unknown}; known: {sorted(datasets)}", file=sys.stderr)
        return 2
    results = [check_dataset(name, datasets[name]) for name in names]
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(f"WS = {ws} | cấu hình: {args.configs}"
              + (f" | ghi đè: {registry['local_overrides']}" if registry.get("local_overrides") else ""))
        for result in results:
            print("\n".join(format_result(result)))
    return 0 if all(result["ok"] for result in results if result["required"]) else 1


if __name__ == "__main__":
    sys.exit(main())
