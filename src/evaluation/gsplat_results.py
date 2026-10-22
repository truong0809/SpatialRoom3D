"""Đọc kết quả một lượt huấn luyện gsplat (một chế độ, một cảnh, một seed).

Thư mục kết quả (`--result-dir` của simple_trainer.py) sau khi `src.training.run` chạy xong:

    run_meta.json                     do src.training.run ghi (cấu hình, lệnh, commit, GPU, thời gian)
    split.json / split_applied.json   tập train/test dự kiến và tập gsplat thực sự dùng
    resolved_config.json              cấu hình simple_trainer sau khi phân giải
    gpu_log.csv                       nhật ký nvidia-smi trong lúc huấn luyện
    stats/train_step<N>_rank0.json    gsplat: mem (GiB, max_memory_allocated), ellipse_time, num_GS
    stats/val_step<N>.json            gsplat: psnr, ssim, lpips, ellipse_time (giây/ảnh), num_GS
    ckpts/ckpt_<N>_rank0.pt, ply/point_cloud_<N>.ply
    filtered/                         (tùy chọn) bản đã lọc Gaussian và lượt chấm lại của nó (src.training.prune):
                                      filter_meta.json, stats/val_step<N>.json, ckpts/, ply/, renders/, videos/

Số liệu trong summary.json chỉ lấy từ các file trên, không ước lượng. Phần `filtered` luôn được đọc lại từ thư mục
`filtered/` (lọc chạy sau khi summary.json đã ghi).
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.evaluation.compare_reference import SceneResult, peak_gpu_memory_mib, read_ply_vertex_count

_VAL = re.compile(r"^val_step(\d+)\.json$")
_TRAIN = re.compile(r"^train_step(\d+)_rank0\.json$")
FILTER_DIR = "filtered"  # <lượt>/filtered/: bản đã lọc Gaussian (src/training/prune.py)
FILTER_META = "filter_meta.json"


def latest_stats(stats_dir: Path, kind: str) -> tuple[int, dict] | None:
    """File stats có bước lớn nhất: kind = 'val' hoặc 'train'."""
    pattern = _VAL if kind == "val" else _TRAIN
    best = None
    if not stats_dir.is_dir():
        return None
    for path in stats_dir.iterdir():
        match = pattern.match(path.name)
        if match and (best is None or int(match.group(1)) > best[0]):
            best = (int(match.group(1)), path)
    if best is None:
        return None
    data = read_json_or_none(best[1])
    return (best[0], data) if data is not None else None


def read_json_or_none(path: Path) -> dict | None:
    """Nội dung JSON (dict) của file; None nếu không có hoặc hỏng (vd. pod tắt khi đang ghi)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_json(path: Path) -> dict:
    return read_json_or_none(path) or {}


def _size_mb(path: Path) -> float | None:
    return round(path.stat().st_size / 2**20, 2) if path.is_file() else None


@dataclass
class RunResult:
    run_dir: str
    scene: str | None = None
    mode: str | None = None
    seed: int | None = None
    status: str = "MISSING"  # OK | MISSING | FAILED | RUNNING
    problem: str | None = None  # lý do không OK khi đã có số liệu (vd. chỉ số không hữu hạn)
    metrics: dict = field(default_factory=dict)  # PSNR, SSIM, LPIPS (lượt đánh giá cuối)
    eval_step: int | None = None
    num_gaussians: int | None = None
    cap_max: int | None = None
    train_seconds: float | None = None  # thời gian thực của tiến trình huấn luyện (gồm nạp dữ liệu, ghi file)
    train_loop_seconds: float | None = None  # ellipse_time của gsplat: chỉ vòng lặp huấn luyện
    eval_seconds: float | None = None
    peak_vram_mib: float | None = None  # nvidia-smi (gồm cả bộ nhớ đệm của PyTorch và CUDA context)
    torch_peak_gib: float | None = None  # torch.cuda.max_memory_allocated() do gsplat ghi
    render_ms_per_image: float | None = None
    ply_size_mb: float | None = None
    ckpt_size_mb: float | None = None
    protocol: dict = field(default_factory=dict)
    deviations: dict = field(default_factory=dict)
    args: dict = field(default_factory=dict)  # tham số riêng của chế độ (--set)
    cap_policy: str | None = None
    split_sha256: str | None = None
    split_rule: str | None = None  # "every_n" | "test_list"
    split_applied_ok: bool | None = None
    gsplat_commit: str | None = None
    gpu: str | None = None
    lpips_net: str | None = None
    scene_fingerprint: str | None = None  # SHA-256 của sparse/0 (src.training.run.scene_fingerprint)
    tag: str | None = None  # biến thể của chế độ trong kế hoạch thực nghiệm (src.training.run --tag)
    filtered: dict | None = None  # bản đã lọc Gaussian, xem load_filtered; None khi chưa lọc

    def to_dict(self) -> dict:
        return asdict(self)

    def to_scene_result(self) -> SceneResult:
        """Chuyển sang SceneResult để dùng lại compare_reference.compare / format_table."""
        return SceneResult(
            scene=self.scene or Path(self.run_dir).name,
            model_dir=self.run_dir,
            status="OK" if self.status == "OK" else "MISSING",
            metrics=dict(self.metrics),
            train_seconds=self.train_seconds,
            peak_vram_mib=self.peak_vram_mib,
            num_gaussians=self.num_gaussians,
            ply_size_mb=self.ply_size_mb,
        )


def _split_applied_ok(run_dir: Path) -> bool | None:
    """So tập ảnh gsplat thực sự dùng với tập dự kiến (split.json).

    Tập train lấy từ lượt huấn luyện (split_applied.json), tập val lấy từ lượt đánh giá
    (split_applied_eval.json) nếu có, vì chỉ số được tính ở lượt đó.
    """
    planned = _read_json(run_dir / "split.json")
    trained = _read_json(run_dir / "split_applied.json").get("splits", {})
    evaluated = _read_json(run_dir / "split_applied_eval.json").get("splits", {}) or trained
    if not planned or "train" not in trained or "val" not in evaluated:
        return None
    return (sorted(trained["train"]) == sorted(planned.get("train", []))
            and sorted(evaluated["val"]) == sorted(planned.get("test", [])))


def load_run(run_dir: Path) -> RunResult:
    run_dir = Path(run_dir)
    meta = _read_json(run_dir / "run_meta.json")
    spec = meta.get("spec", {})
    protocol = spec.get("protocol", {})
    result = RunResult(
        run_dir=str(run_dir),
        scene=meta.get("scene"),
        mode=meta.get("mode"),
        seed=meta.get("seed"),
        cap_max=(meta.get("cap_max") or {}).get("value"),
        cap_policy=(meta.get("cap_max") or {}).get("policy"),
        train_seconds=meta.get("train_seconds"),
        eval_seconds=meta.get("eval_seconds"),
        protocol=protocol,
        deviations=spec.get("deviations", {}),
        args=spec.get("args", {}),
        split_sha256=meta.get("split", {}).get("sha256"),
        split_rule=meta.get("split", {}).get("rule"),
        gsplat_commit=meta.get("gsplat_commit"),
        gpu=meta.get("gpu"),
        lpips_net=protocol.get("lpips_net"),
        scene_fingerprint=(meta.get("scene_info") or {}).get("fingerprint"),
        tag=meta.get("tag"),
        filtered=load_filtered(run_dir),
    )
    final_step = int(protocol["max_steps"]) - 1 if "max_steps" in protocol else None

    val = latest_stats(run_dir / "stats", "val")
    if val is not None:
        step, stats = val
        if all(key in stats for key in ("psnr", "ssim", "lpips")):
            result.metrics = {"PSNR": float(stats["psnr"]), "SSIM": float(stats["ssim"]),
                              "LPIPS": float(stats["lpips"])}
            result.eval_step = step
            if stats.get("num_GS") is not None:
                result.num_gaussians = int(stats["num_GS"])
            if stats.get("ellipse_time") is not None:
                result.render_ms_per_image = round(float(stats["ellipse_time"]) * 1000, 3)

    train = latest_stats(run_dir / "stats", "train")
    if train is not None:
        _, stats = train
        if stats.get("mem") is not None:
            result.torch_peak_gib = round(float(stats["mem"]), 3)
        if stats.get("ellipse_time") is not None:
            result.train_loop_seconds = round(float(stats["ellipse_time"]), 1)
        if result.num_gaussians is None and stats.get("num_GS") is not None:
            result.num_gaussians = int(stats["num_GS"])

    result.peak_vram_mib = peak_gpu_memory_mib(run_dir / "gpu_log.csv")
    if final_step is not None:
        ply = run_dir / "ply" / f"point_cloud_{final_step}.ply"
        result.ply_size_mb = _size_mb(ply)
        if result.num_gaussians is None and ply.is_file():
            result.num_gaussians = read_ply_vertex_count(ply)
        result.ckpt_size_mb = _size_mb(run_dir / "ckpts" / f"ckpt_{final_step}_rank0.pt")
    result.split_applied_ok = _split_applied_ok(run_dir)

    meta_status = meta.get("status")
    if (run_dir / "run_meta.json").exists() and not meta:
        result.status, result.problem = "FAILED", "run_meta.json is unreadable"
    elif meta_status in ("FAILED", "RUNNING"):
        result.status = meta_status
    elif result.metrics and not all(math.isfinite(value) for value in result.metrics.values()):
        result.status, result.problem = "FAILED", "non-finite metrics (training diverged?)"
    elif result.metrics and (final_step is None or result.eval_step == final_step):
        result.status = "OK"
    return result


def load_summary(run_dir: Path) -> RunResult:
    """Đọc summary.json nếu có, nếu không thì tính lại từ các file gốc.

    `run_dir` của kết quả là thư mục đang đọc, không phải đường dẫn ghi trong summary.json: thư mục kết quả có thể đã
    được dời hoặc chép sang chỗ khác kể từ khi lượt đó chạy.
    """
    data = read_json_or_none(Path(run_dir) / "summary.json")
    if data is not None and data.get("run_dir"):
        known = {f for f in RunResult.__dataclass_fields__}
        result = RunResult(**{key: value for key, value in data.items() if key in known})
        result.run_dir = os.path.abspath(run_dir)
        result.filtered = load_filtered(Path(run_dir))
        return result
    return load_run(Path(run_dir))


def load_filtered(run_dir: Path) -> dict | None:
    """Bản đã lọc Gaussian của một lượt (`<lượt>/filtered/`, src.training.prune); None khi lượt chưa được lọc.

    Ngưỡng và số Gaussian bỏ đi lấy từ filter_meta.json; chỉ số, số Gaussian còn lại và thời gian render lấy từ stats của
    lượt chấm lại (gsplat ghi), dung lượng từ file PLY đã lọc. `status` khác OK (đang chạy, lỗi, hoặc số Gaussian trong
    stats không khớp số đã giữ lại) thì không có chỉ số."""
    folder = Path(run_dir) / FILTER_DIR
    meta = read_json_or_none(folder / FILTER_META)
    if meta is None:
        return None
    counts = meta.get("counts") or {}
    info = {"status": meta.get("status") or "MISSING", "thresholds": meta.get("thresholds"),
            "scene_scale": meta.get("scene_scale"), "num_gaussians_input": counts.get("input"),
            "removed": counts.get("removed"), "kept_fraction": counts.get("kept_fraction"), "dir": str(folder)}
    if info["status"] != "OK":
        info["problem"] = meta.get("error")
        return info
    val = latest_stats(folder / "stats", "val")
    stats = val[1] if val is not None else {}
    if not all(key in stats for key in ("psnr", "ssim", "lpips")):
        return {**info, "status": "FAILED", "problem": "no evaluation stats in filtered/stats"}
    metrics = {"PSNR": float(stats["psnr"]), "SSIM": float(stats["ssim"]), "LPIPS": float(stats["lpips"])}
    if not all(math.isfinite(value) for value in metrics.values()):
        return {**info, "status": "FAILED", "problem": "non-finite metrics after filtering"}
    if stats.get("num_GS") is not None and stats["num_GS"] != counts.get("kept"):
        return {**info, "status": "FAILED",
                "problem": f"the evaluation saw {stats['num_GS']} Gaussians, the filter kept {counts.get('kept')}"}
    info.update(metrics=metrics, num_gaussians=counts.get("kept"), eval_step=val[0],
                render_ms_per_image=(round(float(stats["ellipse_time"]) * 1000, 3)
                                     if stats.get("ellipse_time") is not None else None),
                ply_size_mb=_size_mb(folder / "ply" / f"point_cloud_{val[0]}.ply"))
    return info
