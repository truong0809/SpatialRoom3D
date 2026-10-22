"""Render lại và xuất mô hình của một lượt gsplat đã train xong (bước "Export PLY/Checkpoint" của luồng chính).

Một lượt train OK (`<lượt>/run_meta.json` có status DONE) đã có sẵn, do bước đánh giá của `src.training.run`:
  - ảnh test render được (`renders/val_step<N>_*.png`), chỉ số (`stats/val_step<N>.json`);
  - video quỹ đạo `videos/traj_<N>.mp4` (chế độ `--ckpt` của gsplat simple_trainer, khi protocol không tắt video);
  - file PLY `ply/point_cloud_<N>.ply` và checkpoint `ckpts/ckpt_<N>_rank0.pt`.
Module này KHÔNG train và KHÔNG ghi đè gì của lượt gốc:
  - render: dựng lại lệnh đánh giá đã ghi trong run_meta.json (`commands.eval`), đổi `--result-dir` sang thư mục mới
    `<lượt>/render_<quỹ đạo>_<thời gian>/` và chọn quỹ đạo video (`--render-traj-path interp | ellipse | spiral`).
    gsplat ghi ảnh test (renders/), chỉ số (stats/) và video (videos/) vào thư mục đó; `render_meta.json` ghi lệnh,
    thời gian, mã thoát. Chỉ dùng chức năng có sẵn của gsplat (không có thuật toán render mới);
  - xuất (`--export <thư mục>`): đưa PLY cuối của lượt vào `<thư mục>/<cảnh>/<lượt>.ply` (để tải về / mở bằng trình
    xem 3D), bằng liên kết cứng khi được (không tốn thêm chỗ trên volume; file xuất dùng chung dữ liệu với PLY của
    lượt: chỉ đọc / tải về, không sửa tại chỗ), không thì chép. Không ghi đè: tên đó đã là một PLY khác (vd. lượt cũ
    trước khi train lại) thì xuất thành `<lượt>_<thời điểm train xong>.ply`.
Render dừng nếu mô hình COLMAP của cảnh đã đổi sau khi train (vd. làm lại COLMAP): checkpoint chỉ khớp với camera cũ.
Xuất PLY và `--dry-run` không cần GPU.

Cách dùng (Python của env gs-gsplat; thường qua scripts/run_render.sh):
  python -m src.rendering.render --run <thư mục lượt> [--traj ellipse] [--export <thư mục>] [--no-render] [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import filecmp
import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
TRAJECTORIES = ("interp", "ellipse", "spiral")  # gsplat simple_trainer @6e8c837, Config.render_traj_path


class RenderError(ValueError):
    pass


def read_meta(run: Path) -> dict:
    path = Path(run) / "run_meta.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RenderError(f"{path}: cannot read ({error}); is {run} a gsplat run of src.training.run?") from error
    if not isinstance(meta, dict):
        raise RenderError(f"{path}: not a run description")
    if meta.get("status") != "DONE":
        raise RenderError(f"{run} is not a finished run (status {meta.get('status')!r}): train it first")
    return meta


def value_after(tokens: list[str], flag: str) -> str | None:
    if flag in tokens:
        index = tokens.index(flag)
        if index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def without_option(tokens: list[str], flag: str, has_value: bool = True) -> list[str]:
    result, skip = [], False
    for token in tokens:
        if skip:
            skip = False
            continue
        if token == flag:
            skip = has_value
            continue
        result.append(token)
    return result


def render_command(meta: dict, render_dir: Path, trajectory: str, python: str) -> list[str]:
    """Lệnh render từ checkpoint: phần tham số gsplat của lệnh đánh giá đã ghi, chỉ đổi thư mục ghi và quỹ đạo video."""
    if trajectory not in TRAJECTORIES:
        raise RenderError(f"unknown trajectory {trajectory!r} (known: {', '.join(TRAJECTORIES)})")
    recorded = (meta.get("commands") or {}).get("eval")
    if not isinstance(recorded, list) or "--" not in recorded:
        raise RenderError("run_meta.json has no recorded evaluation command (commands.eval)")
    cut = recorded.index("--")
    launcher, trainer = recorded[:cut], list(recorded[cut + 1:])
    examples = value_after(launcher, "--examples-dir")
    ckpt = value_after(trainer, "--ckpt")
    data_dir = value_after(trainer, "--data-dir")
    if not examples or not ckpt or not data_dir:
        raise RenderError("the recorded evaluation command lacks --examples-dir, --ckpt or --data-dir")
    for label, path in (("checkpoint", ckpt), ("data", data_dir), ("gsplat examples", examples)):
        if not Path(path).exists():
            raise RenderError(f"{label} not found: {path}")
    trainer = without_option(trainer, "--result-dir")
    trainer = without_option(trainer, "--render-traj-path")
    trainer = [token for token in trainer if token not in ("--disable-video", "--no-disable-video")]
    trainer += ["--result-dir", str(render_dir), "--render-traj-path", trajectory, "--no-disable-video"]
    cmd = [python, "-m", "src.training.gsplat_launcher", "--examples-dir", examples, "--seed", str(meta.get("seed", 42)),
           "--record-split", str(render_dir / "split_applied_render.json"),
           "--resolved-config", str(render_dir / "resolved_config_render.json")]
    split_file = value_after(launcher, "--split-file")
    if split_file:
        cmd += ["--split-file", split_file]
    return cmd + ["--", *trainer]


def check_scene_model(meta: dict) -> None:
    """Checkpoint chỉ khớp với mô hình COLMAP đã dùng để train: dừng nếu mô hình của cảnh đã được dựng lại."""
    from src.training.run import scene_fingerprint
    scene_dir = meta.get("scene_dir")
    recorded = (meta.get("scene_info") or {}).get("fingerprint")
    if not scene_dir or not recorded:
        raise RenderError("run_meta.json does not record the COLMAP model used for training (scene_dir, "
                          "scene_info.fingerprint): cannot check that the cameras still match the checkpoint")
    try:
        current = scene_fingerprint(Path(scene_dir))
    except OSError as error:
        raise RenderError(f"COLMAP model used for training not found in {scene_dir} ({error})") from error
    if current != recorded:
        raise RenderError(f"the COLMAP model in {scene_dir} changed after this run was trained (COLMAP redone?): "
                          "its cameras no longer match the checkpoint; train the scene again instead")


def final_ply(run: Path) -> Path:
    """PLY của bước cuối (số bước lớn nhất) trong <lượt>/ply/."""
    candidates = []
    for path in (Path(run) / "ply").glob("point_cloud_*.ply"):
        step = path.stem.rsplit("_", 1)[-1]
        if step.isdigit():
            candidates.append((int(step), path))
    if not candidates:
        raise RenderError(f"no PLY in {Path(run) / 'ply'} (the run was trained without save_ply?)")
    return max(candidates)[1]


def _same_file(source: Path, target: Path) -> bool:
    """Cùng một file hoặc cùng nội dung (so từng khối, không nạp cả file PLY vào bộ nhớ)."""
    return os.path.samefile(source, target) or filecmp.cmp(source, target, shallow=False)


def export_ply(run: Path, meta: dict, export_root: Path) -> tuple[Path, str]:
    """Đưa PLY cuối vào <export_root>/<cảnh>/<lượt>.ply; trả về (đích, "hardlink" | "copy" | "exists").

    Tên đó đã là một PLY khác (lượt cũ trước khi train lại, đã đổi tên .prev_*): xuất thành
    <lượt>_<thời điểm train xong>.ply, không ghi đè.
    """
    source = final_ply(run)
    scene = str(meta.get("scene") or Path(run).parent.name)
    target = Path(export_root) / scene / f"{Path(run).name}.ply"
    if target.exists() and not _same_file(source, target):
        digits = re.sub(r"\D", "", str(meta.get("finished") or ""))[:14]
        stamp = f"{digits[:8]}_{digits[8:]}" if len(digits) == 14 else f"{dt.datetime.now():%Y%m%d_%H%M%S}"
        target = target.with_name(f"{Path(run).name}_{stamp}.ply")
    if target.exists():
        if _same_file(source, target):
            return target, "exists"
        raise RenderError(f"{target} already exists with other content: not overwriting it")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return target, "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return target, "copy"


def summarize(render_dir: Path) -> dict:
    stats = sorted((render_dir / "stats").glob("val_step*.json"))
    metrics = {}
    if stats:
        try:
            data = json.loads(stats[-1].read_text(encoding="utf-8"))
            metrics = {key: data[key] for key in ("psnr", "ssim", "lpips") if key in data}
        except (OSError, ValueError):
            metrics = {}
    return {"images": len(list((render_dir / "renders").glob("*.png"))),
            "videos": sorted(str(path) for path in (render_dir / "videos").glob("*.mp4")),
            "metrics": metrics}


def main(argv: list[str] | None = None, runner=None) -> int:
    parser = argparse.ArgumentParser(description="Render lại / xuất PLY của một lượt gsplat đã train (không train)")
    parser.add_argument("--run", type=Path, required=True, help="thư mục lượt chạy (có run_meta.json)")
    parser.add_argument("--traj", default="interp", choices=TRAJECTORIES, help="quỹ đạo video (mặc định interp)")
    parser.add_argument("--export", type=Path, help="xuất PLY cuối vào <thư mục>/<cảnh>/<lượt>.ply")
    parser.add_argument("--no-render", action="store_true", help="chỉ xuất PLY, không render (không cần GPU)")
    parser.add_argument("--dry-run", action="store_true", help="chỉ in lệnh render, không chạy, không ghi gì")
    parser.add_argument("--python", default=sys.executable, help="Python của env gs-gsplat (mặc định: python này)")
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    try:
        meta = read_meta(run)
        if args.no_render and args.export is None:
            raise RenderError("--no-render needs --export (nothing to do otherwise)")
        if args.export is not None:
            final_ply(run)  # không có PLY thì dừng trước khi render
        render_dir = run / f"render_{args.traj}_{dt.datetime.now():%Y%m%d_%H%M%S}"
        cmd = None
        if not args.no_render:
            cmd = render_command(meta, render_dir, args.traj, args.python)
            check_scene_model(meta)
    except RenderError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps({"run": str(run), "render_dir": None if cmd is None else str(render_dir),
                          "render": None if cmd is None else shlex.join(cmd),
                          "export": None if args.export is None else str(args.export)}, indent=2, ensure_ascii=False))
        return 0
    code = 0
    if cmd is not None:
        if runner is None:
            from src.training.run import run_command as runner
        render_dir.mkdir(parents=True)
        record = {"source_run": str(run), "scene": meta.get("scene"), "mode": meta.get("mode"),
                  "trajectory": args.traj, "command": cmd, "started": dt.datetime.now().isoformat(timespec="seconds")}
        (render_dir / "render_meta.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        env = dict(os.environ)
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1",
                   MPLBACKEND="Agg")
        env["PYTHONPATH"] = str(REPO_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        print(f"[render] {meta.get('scene')} / {run.name}: trajectory {args.traj} -> {render_dir}")
        code = runner(cmd, Path(cmd[cmd.index("--examples-dir") + 1]), env, render_dir / "render.log")
        record.update(finished=dt.datetime.now().isoformat(timespec="seconds"), exit_code=code, **summarize(render_dir))
        (render_dir / "render_meta.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        if code != 0:
            print(f"ERROR: rendering failed with exit code {code}; see {render_dir / 'render.log'}", file=sys.stderr)
        else:
            print(f"[render] {record['images']} test images, videos: {', '.join(record['videos']) or 'none'}; "
                  f"metrics {record['metrics'] or '-'}")
    if args.export is not None and code == 0:
        try:
            target, how = export_ply(run, meta, args.export)
        except RenderError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        print(f"[export] {target} ({how})")
    return 0 if code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
