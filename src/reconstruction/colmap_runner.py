"""Chạy COLMAP cho một cảnh và đưa kết quả về cấu trúc mà 3DGS gốc (Inria) đọc được.

Các bước giống `convert.py` của Inria: trích đặc trưng (camera OPENCV, một camera chung)
-> so khớp -> dựng model (Mapper.ba_global_function_tolerance = 1e-6) -> khử méo về PINHOLE
-> chuyển model vào `<scene>/sparse/0`. Khác biệt có chủ đích:

  * Tự nhận tên tùy chọn GPU theo phiên bản COLMAP: bản mới dùng `FeatureExtraction.*` /
    `FeatureMatching.*`, bản cũ dùng `SiftExtraction.*` / `SiftMatching.*`. `convert.py`
    gốc chỉ biết tên cũ nên không chạy được với COLMAP mới.
  * Nếu COLMAP tách thành nhiều model, khử méo model có nhiều ảnh nhất
    (`convert.py` luôn lấy model 0).
  * Tùy chọn sequential matcher (video dài) và global mapper (COLMAP >= 4.0) cho thực nghiệm.
  * GPU hay CPU: mặc định (`--gpu auto`) đọc dòng đầu của `colmap -h`; chỉ dùng GPU khi bản COLMAP
    có CUDA. Bản CPU (cài khi driver cũ hơn yêu cầu của bản CUDA) mà bị yêu cầu dùng GPU sẽ cố mở
    cửa sổ Qt/OpenGL và lỗi trên pod không có màn hình. Lựa chọn thực tế ghi vào
    `colmap_settings` của báo cáo.
  * Làm lại (`--overwrite`) xóa cả báo cáo và log cũ; lỗi giữa chừng vẫn ghi `sfm_report.json` với
    verdict FAIL, để không còn báo cáo cũ trông như hợp lệ.

Đầu vào:  `<scene>/input/*.jpg`
Đầu ra:   `<scene>/images/`, `<scene>/sparse/0/`, `<scene>/distorted/` (database + model gốc),
          `<scene>/colmap_logs/*.log`, `<scene>/sfm_report.json`

Cách dùng:
  python -m src.reconstruction.colmap_runner --scene /workspace/outputs/custom_scenes/phong/colmap \\
      --colmap /workspace/miniforge3/envs/gs-tools/bin/colmap
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.utils.gpu_lock import inherited_lock_fds
from src.reconstruction.report import analyze_scene, failure_hints, format_report, list_models, read_frames_report


OUTPUTS = ("distorted", "images", "sparse", "stereo", "colmap_logs", "sfm_report.json")


def colmap_env() -> dict:
    """Môi trường cho COLMAP: không có màn hình nên Qt (nếu được dùng) phải chạy offscreen."""
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return env


def detect_gpu_option(colmap: str, command: str, candidates: tuple[str, ...]) -> str:
    """Tìm tên tùy chọn use_gpu mà phiên bản COLMAP này hỗ trợ (dựa trên `colmap <command> -h`)."""
    proc = subprocess.run([colmap, command, "-h"], capture_output=True, text=True, env=colmap_env())
    text = proc.stdout + proc.stderr
    for option in candidates:
        if option in text:
            return option
    raise RuntimeError(f"cannot find any of {candidates} in 'colmap {command} -h' output")


def colmap_has_cuda(colmap: str) -> bool | None:
    """Bản COLMAP này dùng được GPU không, theo phần đầu của `colmap -h`.

    COLMAP 4.x: một dòng "COLMAP 4.2.1 (Commit ... with CUDA)" hoặc "... without GPU support".
    COLMAP 3.x: dòng thứ hai "(Commit ... with CUDA)" hoặc "... without CUDA".
    Không nhận ra thì trả về None.
    """
    try:
        proc = subprocess.run([colmap, "-h"], capture_output=True, text=True, env=colmap_env())
    except OSError:
        return None
    head = "\n".join((proc.stdout + proc.stderr).splitlines()[:5]).lower()
    if "without cuda" in head or "without gpu" in head:
        return False
    if "with cuda" in head or "with hip" in head:
        return True
    return None


def run_step(name: str, cmd: list[str], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    print(f"[colmap] {name}: {' '.join(cmd)}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=colmap_env(),
                              pass_fds=inherited_lock_fds())  # COLMAP giữ khóa GPU của script cha
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        raise RuntimeError(f"COLMAP step '{name}' failed (exit {proc.returncode}); see {log_path}\n"
                           + "\n".join(tail))


def resolve_gpu(gpu: str, has_cuda: bool | None) -> bool:
    """gpu: 'auto' | 'on' | 'off' -> có dùng GPU cho SIFT không."""
    if gpu == "off":
        return False
    if gpu == "on":
        if has_cuda is False:
            raise RuntimeError("this COLMAP build has no GPU support (CPU build); use --gpu auto or --gpu off")
        return True
    if gpu != "auto":
        raise ValueError(f"gpu must be auto, on or off (got {gpu!r})")
    if has_cuda is None:
        print("[colmap] cannot tell from 'colmap -h' whether this build has CUDA: using the CPU", flush=True)
    return has_cuda is True


def run_colmap(scene: Path, colmap: str = "colmap", gpu: str = "auto", matcher: str = "exhaustive",
               sequential_overlap: int = 20, loop_detection: bool = False, mapper: str = "incremental",
               camera_model: str = "OPENCV", overwrite: bool = False) -> dict:
    input_dir = scene / "input"
    if not input_dir.is_dir() or not any(input_dir.iterdir()):
        raise RuntimeError(f"{input_dir} is missing or empty")
    old = [name for name in OUTPUTS if (scene / name).exists()]
    if old:
        if not overwrite:
            raise RuntimeError(f"previous COLMAP outputs exist in {scene}: {', '.join(old)} (use --overwrite)")
        for name in old:
            path = scene / name
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    has_cuda = colmap_has_cuda(colmap)
    use_gpu = resolve_gpu(gpu, has_cuda)
    settings = {
        "colmap": colmap, "gpu_requested": gpu, "colmap_cuda": has_cuda, "use_gpu": use_gpu,
        "matcher": matcher, "mapper": mapper, "camera_model": camera_model,
        "sequential_overlap": sequential_overlap if matcher == "sequential" else None,
        "loop_detection": loop_detection if matcher == "sequential" else None,
    }
    print(f"[colmap] SIFT on {'GPU' if use_gpu else 'CPU'} (COLMAP CUDA: {has_cuda})", flush=True)
    try:
        return _run_colmap_steps(scene, colmap, use_gpu, settings)
    except RuntimeError as exc:
        source_type = read_frames_report(scene).get("source_type")
        report = {"scene": str(scene), "verdict": "FAIL", "error": str(exc).splitlines()[0],
                  "warnings": [str(exc).splitlines()[0]], "source_type": source_type,
                  "hints": failure_hints(str(exc), source_type), "colmap_settings": settings}
        (scene / "sfm_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        raise


def _run_colmap_steps(scene: Path, colmap: str, use_gpu: bool, settings: dict) -> dict:
    input_dir = scene / "input"
    distorted = scene / "distorted"
    sparse_raw = distorted / "sparse"
    sparse_raw.mkdir(parents=True)
    database = distorted / "database.db"
    logs = scene / "colmap_logs"
    matcher, mapper, camera_model = settings["matcher"], settings["mapper"], settings["camera_model"]
    gpu = "1" if use_gpu else "0"

    extract_opt = detect_gpu_option(colmap, "feature_extractor",
                                    ("FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"))
    match_command = "exhaustive_matcher" if matcher == "exhaustive" else "sequential_matcher"
    match_opt = detect_gpu_option(colmap, match_command,
                                  ("FeatureMatching.use_gpu", "SiftMatching.use_gpu"))

    run_step("1_feature_extractor", [
        colmap, "feature_extractor", "--database_path", str(database), "--image_path", str(input_dir),
        "--ImageReader.single_camera", "1", "--ImageReader.camera_model", camera_model,
        f"--{extract_opt}", gpu], logs)

    match_cmd = [colmap, match_command, "--database_path", str(database), f"--{match_opt}", gpu]
    if matcher == "sequential":
        match_cmd += ["--SequentialMatching.overlap", str(settings["sequential_overlap"])]
        if settings["loop_detection"]:
            # COLMAP 4.x tải cây từ vựng (vocab tree) về ~/.cache/colmap ở lần đầu
            match_cmd += ["--SequentialMatching.loop_detection", "1"]
    run_step("2_matcher", match_cmd, logs)

    if mapper == "incremental":
        run_step("3_mapper", [
            colmap, "mapper", "--database_path", str(database), "--image_path", str(input_dir),
            "--output_path", str(sparse_raw), "--Mapper.ba_global_function_tolerance=0.000001"], logs)
    else:
        run_step("3_global_mapper", [
            colmap, "global_mapper", "--database_path", str(database), "--image_path", str(input_dir),
            "--output_path", str(sparse_raw)], logs)

    models = list_models(sparse_raw)
    if not models:
        raise RuntimeError("COLMAP produced no model (no images could be registered)")
    chosen = Path(models[0]["path"])
    run_step("4_image_undistorter", [
        colmap, "image_undistorter", "--image_path", str(input_dir), "--input_path", str(chosen),
        "--output_path", str(scene), "--output_type", "COLMAP"], logs)

    # Same layout as Inria convert.py: move undistorted model files into sparse/0.
    final_dir = scene / "sparse" / "0"
    final_dir.mkdir(parents=True, exist_ok=True)
    for item in (scene / "sparse").iterdir():
        if item.is_file():
            shutil.move(str(item), str(final_dir / item.name))

    report = analyze_scene(scene)
    report["colmap_settings"] = {**settings, "extract_gpu_option": extract_opt, "match_gpu_option": match_opt,
                                 "undistorted_model": str(chosen)}
    (scene / "sfm_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chạy COLMAP cho một cảnh (tương đương convert.py của Inria)")
    parser.add_argument("--scene", type=Path, required=True, help="thư mục cảnh chứa input/")
    parser.add_argument("--colmap", default="colmap", help="đường dẫn chương trình colmap")
    parser.add_argument("--gpu", choices=("auto", "on", "off"), default="auto",
                        help="SIFT trên GPU: auto = chỉ khi bản COLMAP có CUDA (mặc định)")
    parser.add_argument("--no-gpu", action="store_true", help="giống --gpu off")
    parser.add_argument("--matcher", choices=("exhaustive", "sequential"), default="exhaustive")
    parser.add_argument("--sequential-overlap", type=int, default=20)
    parser.add_argument("--loop-detection", action="store_true",
                        help="sequential matcher: bật phát hiện vòng lặp (tải vocab tree ở lần đầu)")
    parser.add_argument("--mapper", choices=("incremental", "global"), default="incremental")
    parser.add_argument("--camera-model", default="OPENCV")
    parser.add_argument("--overwrite", action="store_true", help="xóa kết quả COLMAP cũ của cảnh")
    args = parser.parse_args(argv)

    try:
        report = run_colmap(args.scene, colmap=args.colmap, gpu="off" if args.no_gpu else args.gpu,
                            matcher=args.matcher, sequential_overlap=args.sequential_overlap,
                            loop_detection=args.loop_detection, mapper=args.mapper,
                            camera_model=args.camera_model, overwrite=args.overwrite)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        try:  # COLMAP lỗi giữa chừng: báo cáo FAIL vừa ghi có gợi ý cho người dùng
            failed = json.loads((args.scene / "sfm_report.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            failed = {}
        if failed.get("error") == str(exc).splitlines()[0]:
            for hint in failed.get("hints") or []:
                print(f"HINT: {hint}", file=sys.stderr)
        return 1
    print(format_report(report))
    print(f"report: {args.scene / 'sfm_report.json'}")
    return 0 if report["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
