"""Test src/evaluation/segment_rule.py: độ đầy đủ, điều kiện so sánh, d_r và kết luận theo luật mục 3.8 trên lượt giả."""

import json
import math

import pytest

from src.evaluation import segment_rule
from src.evaluation.segment_rule import CONFIGS, ROOMS, SEEDS, build_report, main, markdown

# default và mcmc: độ lệch chuẩn mỗi chỉ số = 1 bước, nên d_r = √2 · √((1² + 1²) / 2) bước = √2 bước
STEP = {"PSNR": 0.1, "SSIM": 0.001, "LPIPS": 0.001}
BASE = {"default": {"PSNR": 25.0, "SSIM": 0.940, "LPIPS": 0.140}, "mcmc": {"PSNR": 25.5, "SSIM": 0.944, "LPIPS": 0.134}}
# chênh lệch của biến thể so với mcmc cùng seed, theo phòng (bội của STEP; d_r ≈ 1,41 bước)
OFFSETS = {
    "rbg": {"video8": {"PSNR": 3}, "video9": {"PSNR": 3}, "video10": {}},              # tốt hơn ở 2 phòng: ĐẠT
    "mono": {"video8": {"PSNR": 3}, "video9": {"PSNR": 3}, "video10": {"SSIM": -2}},   # kém SSIM ở video10
    "kr": {"video8": {"LPIPS": -3}, "video9": {}, "video10": {}},                       # chỉ 1 phòng
}


def write_summary(root, room, config, seed, metrics, gaussians=1000, status="OK", **extra):
    folder = root / room / f"{config}_s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    summary = {"status": status, "metrics": metrics, "num_gaussians": gaussians, "split_sha256": f"split-{room}",
               "gsplat_commit": "6e8c837", "gpu": "NVIDIA GeForce RTX 4090", **extra}
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def make_runs(root, skip=()):
    for room in ROOMS:
        for index, seed in enumerate(SEEDS):
            values = {}
            for config in ("default", "mcmc"):
                values[config] = {name: value + (index - 1) * STEP[name] for name, value in BASE[config].items()}
            for variant, rooms in OFFSETS.items():
                values[f"mcmc_{variant}"] = {name: values["mcmc"][name] + rooms[room].get(name, 0) * STEP[name]
                                             for name in STEP}
            for config, metrics in values.items():
                if (room, config, seed) not in skip:
                    write_summary(root, room, config, seed, metrics)


def test_a_complete_evaluation_applies_the_three_conditions(tmp_path):
    make_runs(tmp_path)
    report = build_report(tmp_path)
    assert report["ok"] == report["total"] == 45 and report["not_ok"] == []
    room = report["rooms"]["video8"]
    assert room["d"]["PSNR"] == pytest.approx(math.sqrt(2) * 0.1)
    assert room["comparisons"]["mcmc - default"]["delta"]["PSNR"] == pytest.approx(0.5)
    assert room["comparisons"]["rbg - mcmc"]["flags"] == {"PSNR": "better", "SSIM": "within", "LPIPS": "within"}
    verdicts = report["verdicts"]
    assert verdicts["rbg"]["status"] == "ĐẠT" and sorted(verdicts["rbg"]["better"]) == ["video8", "video9"]
    assert verdicts["mono"]["status"] == "KHÔNG ĐẠT" and verdicts["mono"]["worse"] == {"video10": ["SSIM"]}
    assert verdicts["kr"]["status"] == "KHÔNG ĐẠT" and verdicts["kr"]["better"] == {"video8": ["LPIPS"]}
    assert "chỉ tốt hơn quá d_r ở 1 phòng" in verdicts["kr"]["reasons"][0]
    text = markdown(report)
    assert "Tổng: 45/45 lượt OK." in text and "| rbg |" in text and "**ĐẠT**" in text
    for line in text.splitlines():  # bảng Markdown: mọi dòng của một bảng cùng số cột
        if line.startswith("|"):
            assert line.count("|") in {len(CONFIGS) + 2, 6, 5}, line


def test_missing_and_failed_runs_are_listed_and_leave_the_verdict_open(tmp_path):
    failed = {("video9", "mcmc_mono", seed) for seed in SEEDS} | {("video9", "mcmc_kr", seed) for seed in SEEDS}
    make_runs(tmp_path, skip=failed | {("video10", "default", 44)})
    for room, config, seed in sorted(failed):
        folder = tmp_path / room / f"{config}_s{seed}"
        folder.mkdir(parents=True)
        (folder / "run_meta.json").write_text(json.dumps({"status": "FAILED"}), encoding="utf-8")
        prev = tmp_path / room / f"{config}_s{seed}.prev_20261112_120000"  # lượt cũ đã thay: không được đọc
        prev.mkdir()
        (prev / "summary.json").write_text(json.dumps({"status": "OK", "metrics": BASE["mcmc"]}), encoding="utf-8")
    report = build_report(tmp_path)
    assert report["ok"] == 38
    assert "video9/mcmc_mono_s42 (FAILED)" in report["not_ok"] and "video10/default_s44 (MISSING)" in report["not_ok"]
    assert report["rooms"]["video10"]["d"] is None
    assert report["verdicts"]["mono"]["status"] == "CHƯA ĐỦ SỐ"
    assert report["verdicts"]["mono"]["missing"] == ["video9", "video10"]
    assert report["verdicts"]["rbg"]["status"] == "CHƯA ĐỦ SỐ"  # video10 thiếu d_r
    assert main(["--root", str(tmp_path), "--markdown", str(tmp_path / "out" / "r.md"),
                 "--json", str(tmp_path / "out" / "r.json")]) == 1
    assert "Chưa OK:" in (tmp_path / "out" / "r.md").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "out" / "r.json").read_text(encoding="utf-8"))["ok"] == 38


def test_a_different_budget_or_split_fails_the_first_condition(tmp_path):
    make_runs(tmp_path)
    metrics = json.loads((tmp_path / "video8" / "mcmc_rbg_s43" / "summary.json").read_text(encoding="utf-8"))["metrics"]
    write_summary(tmp_path, "video8", "mcmc_rbg", 43, metrics, gaussians=999)
    write_summary(tmp_path, "video9", "mcmc_kr", 42, BASE["mcmc"], split_sha256="another-split")
    report = build_report(tmp_path)
    assert report["rooms"]["video8"]["budget"]["mcmc_rbg"] == [{"seed": 43, "gaussians": 999, "default": 1000}]
    assert report["verdicts"]["rbg"]["status"] == "KHÔNG ĐẠT"
    assert report["verdicts"]["rbg"]["unfair"] == ["video8", "video9"]  # video8: số Gaussian; video9: tập chia
    assert not report["rooms"]["video9"]["same"]["split"]["ok"]
    assert report["verdicts"]["mono"]["unfair"] == ["video9"]  # tập chia khác: mọi biến thể của phòng đó
    assert "KHÁC: 6e8c837" not in markdown(report) and "KHÁC: another-split, split-video9" in markdown(report)


def test_a_run_without_finite_metrics_is_not_counted(tmp_path):
    make_runs(tmp_path)
    write_summary(tmp_path, "video8", "default", 42, {"PSNR": float("nan"), "SSIM": 0.9, "LPIPS": 0.1})
    report = build_report(tmp_path)
    assert "video8/default_s42 (NO_METRICS)" in report["not_ok"]
    assert report["rooms"]["video8"]["d"] is None


def test_the_command_needs_an_existing_folder(tmp_path, capsys):
    assert main(["--root", str(tmp_path / "missing")]) == 2
    assert "not found" in capsys.readouterr().err
    assert segment_rule.MIN_ROOMS_BETTER == 2
