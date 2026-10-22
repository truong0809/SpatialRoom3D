"""Lọc Gaussian của một lượt gsplat đã train xong, rồi chấm lại bản đã lọc trên đúng ảnh test (không train lại).

    python -m src.training.prune <lượt> [<lượt> ...] [--min-opacity 0.02] [--max-scale 0.1] [--skip-unfinished]
    (Python của env gs-gsplat, cần GPU để chấm lại; scripts/pod/run_experiment.sh gọi khi kế hoạch có post_filter,
    scripts/run_filter.sh cho một lượt bất kỳ)

Bỏ một Gaussian khi (theo thứ tự; mỗi Gaussian chỉ được đếm ở tiêu chí đầu tiên loại nó):
  * tham số không hữu hạn (NaN / vô cực; export_splats của gsplat cũng bỏ các Gaussian này khi ghi PLY);
  * opacity (sau sigmoid) < --min-opacity: gần như trong suốt (lớp sương mờ, vệt mờ lơ lửng);
  * trục lớn nhất (exp của scale) > --max-scale × scene_scale: quá lớn so với cảnh. scene_scale tính đúng như
    simple_trainer.py @6e8c837 (Runner.__init__: parser.scene_scale × 1,1 × global_scale, với parser.scene_scale là khoảng
    cách lớn nhất từ một camera tới tâm các camera sau khi chuẩn hóa), cùng đại lượng mà DefaultStrategy dùng cho ngưỡng
    prune_scale3d = 0,1 lúc densify; đối chiếu với dòng "Scene scale:" trong train.log khi có.
Ngưỡng 0 tắt tiêu chí đó. Các Gaussian còn lại giữ nguyên tham số (không tối ưu lại).

Ghi vào <lượt>/filtered/ (không sửa gì của lượt gốc):
  ckpts/ckpt_<N>_rank0.pt   checkpoint đã lọc, cùng cấu trúc với checkpoint của gsplat
  ply/point_cloud_<N>.ply   PLY đã lọc, ghi bằng export_splats của gsplat như PLY của lượt gốc
  stats/, renders/, videos/ kết quả chấm lại: lệnh đánh giá của lượt gốc (run_meta.json, commands.eval), chỉ đổi
                            --result-dir, --ckpt và hai file ghi lại của gsplat_launcher
  filter_meta.json          ngưỡng, scene_scale, số Gaussian bỏ theo từng tiêu chí, SHA-256 checkpoint gốc, lệnh, thời gian
Chỉ số sau khi lọc nằm trong stats/ do gsplat ghi; src.evaluation.gsplat_results.load_filtered đọc chúng cho báo cáo.
Chạy lại an toàn: lượt đã lọc xong với cùng ngưỡng và cùng checkpoint gốc, chỉ số còn đọc được và PLY còn, thì bỏ qua;
khác (hoặc lần trước dở dang, hoặc thiếu file) thì thư mục cũ đổi tên filtered.prev_<thời gian> (không xóa). Mô hình COLMAP
của cảnh đổi sau khi train thì dừng (camera không còn khớp checkpoint). Không kiểm được rằng lượt chấm lại dùng đúng tập
ảnh test của lượt (thiếu file ghi lại) cũng là lỗi.
Mã thoát: 0 khi mọi lượt đã lọc xong hoặc được bỏ qua; 1 khi có lượt lỗi (lỗi của một lượt không dừng các lượt sau);
2 khi tham số sai.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import shlex
import sys
import time
from pathlib import Path

from src.evaluation.gsplat_results import (FILTER_DIR, FILTER_META, latest_stats, load_filtered, load_summary,
                                           read_json_or_none)

REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MIN_OPACITY = 0.02
DEFAULT_MAX_SCALE = 0.1
SCENE_SCALE_FACTOR = 1.1  # simple_trainer.py @6e8c837, Runner.__init__: parser.scene_scale * 1.1 * cfg.global_scale
SPLAT_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")  # checkpoint của simple_trainer (không app_opt)
SCENE_SCALE_LOG = re.compile(r"^Scene scale:\s*([0-9.eE+-]+)\s*$", re.M)


class FilterError(ValueError):
    pass


class Unfinished(FilterError):
    """Thư mục không phải một lượt đã train xong (với --skip-unfinished: bỏ qua, không tính là lỗi)."""


def keep_mask(splats: dict, scene_scale: float, min_opacity: float, max_scale: float):
    """(mảng bool các Gaussian giữ lại, số Gaussian bỏ theo từng tiêu chí) từ các mảng numpy của checkpoint gsplat
    (`scales` là log của trục, `opacities` là logit)."""
    import numpy as np

    count = len(splats["means"])
    finite = np.ones(count, dtype=bool)
    for key in SPLAT_KEYS:
        finite &= np.isfinite(np.asarray(splats[key]).reshape(count, -1)).all(axis=1)
    keep = finite.copy()
    removed = {"non_finite": int(count - finite.sum())}
    with np.errstate(over="ignore", invalid="ignore"):
        opacity = 1.0 / (1.0 + np.exp(-np.asarray(splats["opacities"], dtype=np.float64).reshape(count)))
        largest = np.exp(np.asarray(splats["scales"], dtype=np.float64).reshape(count, -1).max(axis=1))
    for name, active, bad in (("opacity", min_opacity > 0, opacity < min_opacity),
                              ("max_scale", max_scale > 0, largest > max_scale * scene_scale)):
        dropped = keep & bad if active else np.zeros(count, dtype=bool)
        removed[name] = int(dropped.sum())
        keep &= ~dropped
    return keep, removed


class TorchIO:
    """Đọc / ghi checkpoint và PLY như simple_trainer.py @6e8c837 (torch.save của {"step", "splats", ...};
    export_splats(format="ply")). `exporter`: thay export_splats của gsplat (test)."""

    def __init__(self, exporter=None):
        self.exporter = exporter

    def load(self, path: Path) -> dict:
        import torch
        return torch.load(path, map_location="cpu", weights_only=True)  # như lệnh đánh giá --ckpt của gsplat

    def arrays(self, ckpt: dict) -> dict:
        return {key: value.detach().cpu().numpy() for key, value in ckpt["splats"].items()}

    def subset(self, ckpt: dict, keep) -> dict:
        import torch
        mask = torch.from_numpy(keep)
        splats = ckpt["splats"]
        return {**ckpt, "splats": type(splats)((key, value[mask].contiguous()) for key, value in splats.items())}

    def save(self, ckpt: dict, path: Path) -> None:
        import torch
        torch.save(ckpt, path)

    def export_ply(self, ckpt: dict, path: Path) -> None:
        exporter = self.exporter
        if exporter is None:
            from gsplat import export_splats as exporter  # cùng hàm simple_trainer.py dùng để ghi PLY
        s = ckpt["splats"]
        exporter(means=s["means"], scales=s["scales"], quats=s["quats"], opacities=s["opacities"], sh0=s["sh0"],
                 shN=s["shN"], format="ply", save_to=str(path))


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def logged_scene_scale(run: Path) -> float | None:
    """scene_scale mà simple_trainer.py in ra lúc train ("Scene scale: <số>" trong train.log), nếu có."""
    try:
        text = (Path(run) / "train.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    found = SCENE_SCALE_LOG.findall(text)
    return float(found[-1]) if found else None


def gsplat_scene_scale(meta: dict, run: Path, parser_class) -> float:
    """scene_scale của Runner (simple_trainer.py @6e8c837) cho lượt này: Parser của gsplat trên đúng thư mục dữ liệu và
    cách chuẩn hóa của lượt. Hệ số ảnh 1 (Parser không ghi ảnh PNG thu nhỏ); scene_scale chỉ phụ thuộc vị trí camera."""
    resolved = read_json_or_none(Path(run) / "resolved_config.json") or {}
    data_dir = meta.get("data_dir")
    if not data_dir or not Path(data_dir).is_dir():
        raise FilterError(f"the data folder of this run is not there any more: {data_dir}")
    test_every = int(((meta.get("spec") or {}).get("protocol") or {}).get("test_every", 8))
    with contextlib.redirect_stdout(io.StringIO()):  # Parser in vài dòng thông tin
        parser = parser_class(data_dir=str(data_dir), factor=1,
                              normalize=bool(resolved.get("normalize_world_space", True)), test_every=test_every)
    return float(parser.scene_scale) * SCENE_SCALE_FACTOR * float(resolved.get("global_scale", 1.0))


def filtered_eval_command(meta: dict, filtered: Path, ckpt: Path, python: str) -> list[str]:
    """Lệnh đánh giá đã ghi của lượt (commands.eval), chỉ đổi --result-dir, --ckpt và các file ghi lại của launcher."""
    from src.rendering.render import value_after, without_option

    recorded = (meta.get("commands") or {}).get("eval")
    if not isinstance(recorded, list) or "--" not in recorded:
        raise FilterError("run_meta.json has no recorded evaluation command (commands.eval)")
    cut = recorded.index("--")
    launcher, trainer = recorded[:cut], list(recorded[cut + 1:])
    examples = value_after(launcher, "--examples-dir")
    data_dir = value_after(trainer, "--data-dir")
    if not examples or not data_dir or trainer.count("--result-dir") != 1 or trainer.count("--ckpt") != 1:
        raise FilterError("the recorded evaluation command lacks --examples-dir, --data-dir, --result-dir or --ckpt")
    for label, path in (("data", data_dir), ("gsplat examples", examples)):
        if not Path(path).exists():
            raise FilterError(f"{label} not found: {path}")
    trainer = without_option(without_option(trainer, "--result-dir"), "--ckpt")
    trainer += ["--result-dir", str(filtered), "--ckpt", str(ckpt)]
    cmd = [python, "-m", "src.training.gsplat_launcher", "--examples-dir", examples]
    seed = value_after(launcher, "--seed")
    if seed:
        cmd += ["--seed", seed]
    cmd += ["--record-split", str(filtered / "split_applied_eval.json"),
            "--resolved-config", str(filtered / "resolved_config_eval.json")]
    split_file = value_after(launcher, "--split-file")
    if split_file:
        cmd += ["--split-file", split_file]
    return cmd + ["--", *trainer]


def same_test_images(run: Path, filtered: Path) -> bool | None:
    """Ảnh gsplat dùng khi chấm lại == tập test của lượt (split.json); None khi thiếu file ghi lại."""
    planned = read_json_or_none(Path(run) / "split.json") or {}
    used = (read_json_or_none(Path(filtered) / "split_applied_eval.json") or {}).get("splits") or {}
    if "test" not in planned or "val" not in used:
        return None
    return sorted(used["val"]) == sorted(planned["test"])


class Filter:
    """Lọc và chấm lại từng lượt; `runner`, `parser_class`, `io` thay được (test)."""

    def __init__(self, min_opacity: float, max_scale: float, *, force: bool = False, python: str = sys.executable,
                 gpu: str = "0", runner=None, parser_class=None, io=None):
        self.thresholds = {"min_opacity": float(min_opacity), "max_scale": float(max_scale)}
        self.force = force
        self.python = python
        self.gpu = str(gpu)
        self.runner = runner
        self.parser_class = parser_class
        self.io = io or TorchIO()

    def _parser(self, examples: str):
        if self.parser_class is None:
            from src.training.depth_check import load_parser
            self.parser_class = load_parser(Path(examples))
        return self.parser_class

    def prepare(self, run: Path) -> dict:
        """Kiểm tra trước khi đụng tới gì: lượt đã xong, checkpoint cuối, lệnh chấm lại, mô hình COLMAP chưa đổi."""
        from src.rendering.render import RenderError, check_scene_model, read_meta

        try:
            meta = read_meta(run)
        except RenderError as error:
            raise Unfinished(str(error)) from error
        summary = load_summary(run)
        if summary.status != "OK":
            raise Unfinished(f"{run} is not a finished run (summary status {summary.status})")
        protocol = (meta.get("spec") or {}).get("protocol") or {}
        if "max_steps" not in protocol:
            raise FilterError("run_meta.json records no protocol max_steps")
        step = int(protocol["max_steps"]) - 1  # gsplat lưu checkpoint cuối với chỉ số max_steps - 1
        source = run / "ckpts" / f"ckpt_{step}_rank0.pt"
        if not source.is_file():
            raise FilterError(f"final checkpoint not found: {source}")
        try:
            check_scene_model(meta)
        except RenderError as error:
            raise FilterError(str(error)) from error
        filtered = run / FILTER_DIR
        ckpt = filtered / "ckpts" / source.name
        cmd = filtered_eval_command(meta, filtered, ckpt, self.python)
        return {"meta": meta, "summary": summary, "step": step, "source": source, "filtered": filtered, "ckpt": ckpt,
                "ply": filtered / "ply" / f"point_cloud_{step}.ply", "cmd": cmd,
                "examples": cmd[cmd.index("--examples-dir") + 1]}

    def run(self, run: Path) -> str:
        """'OK' | 'SKIP' (đã lọc với cùng ngưỡng); lỗi -> FilterError (Unfinished khi không phải lượt đã xong)."""
        run = Path(run).resolve()
        job = self.prepare(run)
        filtered, digest = job["filtered"], sha256_file(job["source"])
        previous = read_json_or_none(filtered / FILTER_META)
        if (not self.force and previous is not None and previous.get("status") == "OK"
                and previous.get("thresholds") == self.thresholds
                and (previous.get("source_checkpoint") or {}).get("sha256") == digest):
            current = load_filtered(run) or {}
            if current.get("status") == "OK" and job["ply"].is_file():
                print(f"[filter] {run}: already filtered with the same thresholds; skipping")
                return "SKIP"
            print(f"[filter] {run}: the previous {FILTER_DIR}/ is incomplete "
                  f"({current.get('problem') or 'no filtered PLY'}); filtering again")
        try:
            scale = gsplat_scene_scale(job["meta"], run, self._parser(job["examples"]))
        except FilterError:
            raise
        except Exception as error:  # Parser của gsplat không đọc được cảnh, hoặc không nạp được gsplat
            raise FilterError(f"cannot compute gsplat's scene_scale for this run: {type(error).__name__}: {error}") \
                from error
        logged = logged_scene_scale(run)
        if logged is not None and abs(logged - scale) > 1e-6 * max(1.0, abs(scale)):
            raise FilterError(f"scene scale {scale} differs from the one training printed ({logged})")

        if filtered.exists():
            from src.training.run import backup_name
            backup = backup_name(filtered)
            filtered.rename(backup)
            print(f"[filter] previous {FILTER_DIR}/ moved to {backup}")
        (filtered / "ckpts").mkdir(parents=True)
        (filtered / "ply").mkdir()
        record = {
            "status": "RUNNING", "failed_step": None, "source_run": str(run),
            "scene": job["meta"].get("scene"), "mode": job["meta"].get("mode"), "seed": job["meta"].get("seed"),
            **({"tag": job["meta"]["tag"]} if job["meta"].get("tag") else {}),
            "thresholds": self.thresholds, "scene_scale": scale, "scene_scale_logged": logged,
            "max_scale_world": self.thresholds["max_scale"] * scale,
            "source_checkpoint": {"path": str(job["source"]), "sha256": digest},
            "before": {"metrics": job["summary"].metrics, "num_gaussians": job["summary"].num_gaussians,
                       "ply_size_mb": job["summary"].ply_size_mb},
            "commands": {"eval": job["cmd"]}, "started": _now(),
        }
        meta_path = filtered / FILTER_META
        _write_json(meta_path, record)
        stage = "filter"
        try:
            ckpt = self.io.load(job["source"])
            splats = ckpt.get("splats") if isinstance(ckpt, dict) else None
            if not isinstance(splats, dict) or set(splats) != set(SPLAT_KEYS):
                raise FilterError(f"unsupported checkpoint: splats {sorted(splats or {})}, expected {list(SPLAT_KEYS)} "
                                  "(appearance optimisation is not supported)")
            if ckpt.get("step") != job["step"]:
                raise FilterError(f"checkpoint step {ckpt.get('step')} is not the final step {job['step']}")
            arrays = self.io.arrays(ckpt)
            if len({len(value) for value in arrays.values()}) != 1:
                raise FilterError("the splat tensors of the checkpoint do not have the same length")
            keep, removed = keep_mask(arrays, scale, **self.thresholds)
            count, kept = len(keep), int(keep.sum())
            if kept == 0:
                raise FilterError("the thresholds remove every Gaussian")
            record["counts"] = {"input": count, "removed": removed, "kept": kept, "kept_fraction": round(kept / count, 6)}
            subset = self.io.subset(ckpt, keep)
            self.io.save(subset, job["ckpt"])
            self.io.export_ply(subset, job["ply"])
            record["files"] = {"checkpoint": str(job["ckpt"]), "ply": str(job["ply"]),
                               "ckpt_size_mb": round(job["ckpt"].stat().st_size / 2**20, 2),
                               "ply_size_mb": round(job["ply"].stat().st_size / 2**20, 2)}
            _write_json(meta_path, record)
            del ckpt, arrays, subset

            stage = "eval"
            runner = self.runner
            if runner is None:
                from src.training.run import run_command as runner
            env = dict(os.environ)
            env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=self.gpu, PYTHONUNBUFFERED="1",
                       MPLBACKEND="Agg")
            env["PYTHONPATH"] = str(REPO_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            print(f"[filter] {run.name}: kept {kept:,}/{count:,} Gaussians ({removed}); re-evaluating "
                  f"-> {filtered}")
            started = time.monotonic()
            code = runner(job["cmd"], Path(job["examples"]), env, filtered / "eval.log")
            record["eval_seconds"] = round(time.monotonic() - started, 1)
            record["exit_code"] = code
            if code != 0:
                raise FilterError(f"evaluation of the filtered model failed with exit code {code}; see "
                                  f"{filtered / 'eval.log'}")

            stage = "collect"
            stats = latest_stats(filtered / "stats", "val")
            if stats is None or stats[0] != job["step"] or not all(k in stats[1] for k in ("psnr", "ssim", "lpips")):
                raise FilterError(f"no evaluation stats for step {job['step']} in {filtered / 'stats'}")
            if stats[1].get("num_GS") != kept:
                raise FilterError(f"the evaluation saw {stats[1].get('num_GS')} Gaussians, the filter kept {kept}")
            record["same_test_images"] = same_test_images(run, filtered)
            if record["same_test_images"] is None:
                raise FilterError("cannot check that the filtered model was evaluated on the run's test images "
                                  f"(missing {run / 'split.json'} or {filtered / 'split_applied_eval.json'})")
            if record["same_test_images"] is False:
                raise FilterError("the filtered model was evaluated on other images than the run's test set")
            after = {"PSNR": float(stats[1]["psnr"]), "SSIM": float(stats[1]["ssim"]), "LPIPS": float(stats[1]["lpips"])}
            if not all(math.isfinite(value) for value in after.values()):
                raise FilterError(f"non-finite metrics after filtering: {after}")
            record["after"] = {"metrics": after, "num_gaussians": kept}
        except Exception as error:
            record.update(status="FAILED", failed_step=stage, error=f"{type(error).__name__}: {error}", finished=_now())
            _write_json(meta_path, record)
            raise FilterError(f"{stage}: {error}") from error
        record.update(status="OK", finished=_now())
        _write_json(meta_path, record)
        before, after = record["before"]["metrics"] or {}, record["after"]["metrics"]
        print(f"[filter] OK {run.name}: PSNR {before.get('PSNR', float('nan')):.3f} -> {after['PSNR']:.3f}, "
              f"SSIM {before.get('SSIM', float('nan')):.4f} -> {after['SSIM']:.4f}, "
              f"LPIPS {before.get('LPIPS', float('nan')):.4f} -> {after['LPIPS']:.4f}, Gaussians {count:,} -> {kept:,}, "
              f"PLY {record['before']['ply_size_mb']} -> {record['files']['ply_size_mb']} MB")
        return "OK"


def main(argv: list[str] | None = None, runner=None, parser_class=None, io=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="thư mục lượt chạy (có run_meta.json, summary.json OK)")
    parser.add_argument("--min-opacity", type=float, default=DEFAULT_MIN_OPACITY,
                        help=f"bỏ Gaussian có opacity nhỏ hơn (mặc định {DEFAULT_MIN_OPACITY}; 0 = tắt)")
    parser.add_argument("--max-scale", type=float, default=DEFAULT_MAX_SCALE,
                        help=f"bỏ Gaussian có trục lớn nhất > giá trị × scene_scale (mặc định {DEFAULT_MAX_SCALE}; 0 = tắt)")
    parser.add_argument("--skip-unfinished", action="store_true",
                        help="thư mục chưa phải lượt đã train xong thì bỏ qua (không tính là lỗi)")
    parser.add_argument("--force", action="store_true", help="lọc lại kể cả khi đã lọc với cùng ngưỡng")
    parser.add_argument("--dry-run", action="store_true", help="chỉ kiểm tra và in lệnh chấm lại, không ghi gì")
    parser.add_argument("--python", default=sys.executable, help="Python của env gs-gsplat (mặc định: python này)")
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    args = parser.parse_args(argv)
    if not 0 <= args.min_opacity < 1 or args.max_scale < 0 or (args.min_opacity == 0 and args.max_scale == 0):
        print("ERROR: need 0 <= --min-opacity < 1, --max-scale >= 0, and at least one of them above 0", file=sys.stderr)
        return 2
    if "," in args.gpu:
        print("ERROR: the evaluation must use exactly one GPU (CUDA_VISIBLE_DEVICES lists several)", file=sys.stderr)
        return 2
    job = Filter(args.min_opacity, args.max_scale, force=args.force, python=args.python, gpu=args.gpu, runner=runner,
                 parser_class=parser_class, io=io)
    failed = done = skipped = 0
    for run in args.runs:
        try:
            if args.dry_run:
                plan = job.prepare(Path(run).resolve())
                print(json.dumps({"run": str(Path(run).resolve()), "thresholds": job.thresholds,
                                  "filtered": str(plan["filtered"]), "eval": shlex.join(plan["cmd"])},
                                 indent=2, ensure_ascii=False))
                continue
            status = job.run(run)
            done += status == "OK"
            skipped += status == "SKIP"
        except Unfinished as error:
            if args.skip_unfinished:
                print(f"[filter] skip {run}: {error}")
                skipped += 1
                continue
            print(f"ERROR: {run}: {error}", file=sys.stderr)
            failed += 1
        except Exception as error:  # noqa: BLE001 - lỗi của một lượt không dừng các lượt sau
            print(f"ERROR: {run}: {type(error).__name__ + ': ' if not isinstance(error, FilterError) else ''}{error}",
                  file=sys.stderr)
            failed += 1
    if not args.dry_run:
        print(f"[filter] {done} filtered, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
