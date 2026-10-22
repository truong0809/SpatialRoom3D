"""Test so sánh các chế độ huấn luyện. Mọi con số là dữ liệu giả để kiểm tra logic, KHÔNG phải kết quả."""

import json

import pytest

from src.evaluation.compare_modes import b0_check, build_report, discover, format_markdown, main

PROTOCOL = {"data_factor": 1, "test_every": 8, "max_steps": 30000, "lpips_net": "vgg",
            "antialiased": False, "save_ply": True, "disable_video": False}
REFERENCE = {"PSNR": 29.690, "SSIM": 0.906, "LPIPS": 0.238}
TOLERANCE = {"PSNR": 0.5, "SSIM": 0.01, "LPIPS": 0.01}


def write_run(root, scene, mode, seed, psnr, *, count=2_000_000, split="s1", deviations=None,
              protocol=None, gpu="RTX A6000", status="OK", args=None, tag="", fingerprint="fp-a", run_tag=None):
    """`tag`: hậu tố thư mục; `run_tag`: tag ghi trong summary (biến thể của kế hoạch thực nghiệm, run.py --tag)."""
    run = root / scene / f"{mode}{tag}_s{seed}"
    run.mkdir(parents=True)
    (run / "run_meta.json").write_text("{}", encoding="utf-8")
    protocol = protocol or PROTOCOL
    summary = {"run_dir": str(run), "scene": scene, "mode": mode, "seed": seed, "status": status,
               "metrics": {"PSNR": psnr, "SSIM": 0.9, "LPIPS": 0.25}, "num_gaussians": count,
               "train_seconds": 1800.0, "peak_vram_mib": 10240.0, "protocol": protocol,
               "deviations": deviations or {}, "split_sha256": split, "split_rule": "every_n",
               "split_applied_ok": True, "gsplat_commit": "6e8c837", "gpu": gpu, "args": args or {},
               "lpips_net": protocol.get("lpips_net"), "scene_fingerprint": fingerprint, "tag": run_tag}
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run


def write_filtered(run, psnr, *, kept=1_500_000, total=2_000_000, status="OK", min_opacity=0.02):
    """Bản đã lọc của một lượt như src.training.prune ghi ra (rút gọn): filter_meta.json, stats, PLY."""
    folder = run / "filtered"
    (folder / "stats").mkdir(parents=True)
    (folder / "ply").mkdir()
    meta = {"status": status, "thresholds": {"min_opacity": min_opacity, "max_scale": 0.1}, "scene_scale": 4.4,
            "counts": {"input": total, "removed": {"non_finite": 0, "opacity": total - kept, "max_scale": 0},
                       "kept": kept, "kept_fraction": kept / total},
            **({"error": "FilterError: eval: exit code 3"} if status != "OK" else {})}
    (folder / "filter_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (folder / "stats" / "val_step29999.json").write_text(
        json.dumps({"psnr": psnr, "ssim": 0.89, "lpips": 0.26, "ellipse_time": 0.003, "num_GS": kept}))
    (folder / "ply" / "point_cloud_29999.ply").write_bytes(b"\0" * (3 * 2**20))


def checks(report, scene, mode):
    return {c["check"]: c["status"] for c in report["fairness"][scene][mode]}


def test_fair_comparison_with_seeds(tmp_path):
    for scene in ("drjohnson", "playroom"):
        for seed, (base, mcmc) in zip((42, 43), ((29.5, 29.7), (29.7, 29.9))):
            write_run(tmp_path, scene, "default", seed, base)
            write_run(tmp_path, scene, "mcmc", seed, mcmc)
    report = build_report(discover([tmp_path]), "default", ["default", "mcmc"])
    assert report["complete_scenes"] == ["drjohnson", "playroom"]
    agg = report["scenes"]["playroom"]["mcmc"]
    assert agg["n"] == 2 and agg["seeds"] == [42, 43]
    assert agg["metrics"]["PSNR"] == pytest.approx(29.8)
    assert agg["metrics_std"]["PSNR"] == pytest.approx(0.1414, abs=1e-3)
    assert report["deltas"]["playroom"]["mcmc"]["PSNR"] == pytest.approx(0.2)
    assert checks(report, "playroom", "mcmc") == {"protocol": "OK", "split": "OK", "gsplat_commit": "OK",
                                                   "gpu": "OK", "split_applied": "OK", "seeds": "FEW",
                                                   "gaussian_budget": "EQUAL"}
    assert report["mean_over_scenes"]["default"]["PSNR"] == pytest.approx(29.6)
    text = format_markdown(report)
    assert "chưa đủ điều kiện: seeds=FEW" in text  # 2 seed < 3: chưa được nói "cải thiện"
    assert "29.800 ± 0.141" in text


def test_unfair_runs_are_flagged(tmp_path):
    write_run(tmp_path, "room", "default", 42, 30.0)
    write_run(tmp_path, "room", "mcmc", 42, 30.5, split="other", gpu="RTX 4090")
    write_run(tmp_path, "room", "mcmc", 43, 30.4)
    deviation = {"max_steps": {"base": 30000, "used": 40000, "source": "mode"}}
    # chế độ tự thêm có protocol riêng (40.000 vòng) và số Gaussian cố định
    write_run(tmp_path, "room", "mcmc_40k", 42, 31.0, count=3_500_000, deviations=deviation,
              protocol={**PROTOCOL, "max_steps": 40000})
    report = build_report(discover([tmp_path]), "default", ["default", "mcmc", "mcmc_40k"])
    mcmc = checks(report, "room", "mcmc")
    assert mcmc["split"] == "DIFF" and mcmc["gpu"] == "DIFF" and mcmc["protocol"] == "OK"
    assert mcmc["seeds"] == "DIFF"  # baseline [42], mcmc [42, 43]
    assert "chưa đủ điều kiện: split=DIFF, gpu=DIFF, seeds=DIFF" in format_markdown(report)
    # lượt sai khác protocol bị loại mặc định, không trộn vào bảng
    assert report["excluded"][0]["variant"] == "mcmc_40k{max_steps=40000}"
    assert "mcmc_40k" not in json.dumps(report["scenes"])
    report = build_report(discover([tmp_path]), "default", ["default", "mcmc_40k"], include_deviating=True)
    custom = checks(report, "room", "mcmc_40k{max_steps=40000}")
    assert custom["protocol"] == "DIFF" and custom["gaussian_budget"] == "UNEQUAL"


def test_mode_arguments_make_separate_variants(tmp_path):
    write_run(tmp_path, "playroom", "default", 42, 30.0)
    write_run(tmp_path, "playroom", "default", 42, 31.0, args={"strategy.absgrad": True}, tag="_abs")
    write_run(tmp_path, "playroom", "mcmc", 42, 30.2, args={"opacity-reg": 0.001})
    report = build_report(discover([tmp_path]), "default", ["default", "mcmc"])
    variants = report["scenes"]["playroom"]
    assert variants["default"]["n"] == 1 and variants["default"]["metrics"]["PSNR"] == 30.0  # không bị trộn
    assert "default[strategy.absgrad=True]" in variants and "mcmc[opacity-reg=0.001]" in variants
    assert report["deltas"]["playroom"]["mcmc[opacity-reg=0.001]"]["PSNR"] == pytest.approx(0.2)


def test_incomplete_and_failed_runs(tmp_path):
    write_run(tmp_path, "a", "default", 42, 30.0)
    write_run(tmp_path, "b", "default", 42, 30.0)
    write_run(tmp_path, "b", "mcmc", 42, 30.0, status="FAILED")
    report = build_report(discover([tmp_path]), "default", ["default", "mcmc"])
    assert report["complete_scenes"] == []
    assert {"scene": "a", "missing_variants": ["mcmc"]} in report["incomplete"]
    assert any(item.get("status") == "FAILED" for item in report["incomplete"])


def test_backup_folders_are_ignored(tmp_path):
    write_run(tmp_path, "a", "default", 42, 30.0)
    old = write_run(tmp_path / "x", "a", "default", 42, 20.0)
    old.rename(old.with_name(old.name + ".prev_20261009_120000"))
    runs = discover([tmp_path])
    assert len(runs) == 1 and runs[0].metrics["PSNR"] == 30.0


def test_reference_check_and_cli(tmp_path):
    write_run(tmp_path, "drjohnson", "default", 42, 29.4)
    write_run(tmp_path, "playroom", "default", 42, 30.0)
    report = build_report(discover([tmp_path]), "default", ["default"], REFERENCE, TOLERANCE,
                          ["drjohnson", "playroom"])
    assert report["reference_check"]["verdict"] == "FAIL"  # LPIPS giả 0.25 lệch 0.012 > ngưỡng 0.01
    assert report["reference_check"]["delta"]["PSNR"] == pytest.approx(0.01)

    config = tmp_path / "ref.json"
    config.write_text(json.dumps({"reference": {"metrics": REFERENCE}, "tolerance": TOLERANCE,
                                  "scenes": ["drjohnson", "playroom", "missing"]}), encoding="utf-8")
    out, markdown = tmp_path / "report.json", tmp_path / "report.md"
    assert main(["--root", str(tmp_path), "--modes", "default", "--reference-config", str(config),
                 "--out", str(out), "--markdown", str(markdown)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["reference_check"]["verdict"] == "INCOMPLETE"  # thiếu cảnh "missing"
    assert "INCOMPLETE" in markdown.read_text(encoding="utf-8")


def write_b0(root, scene, psnr, ssim=0.905, lpips=0.24):
    model = root / scene
    model.mkdir(parents=True)
    (model / "results.json").write_text(json.dumps({"ours_30000": {"PSNR": psnr, "SSIM": ssim, "LPIPS": lpips}}),
                                        encoding="utf-8")


def test_b0_paired_check(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "drjohnson", "default", 42, 29.30)
    write_run(runs, "playroom", "default", 42, 30.10)
    b0 = tmp_path / "b0"
    write_b0(b0, "drjohnson", 29.20, ssim=0.9, lpips=0.25)
    write_b0(b0, "playroom", 30.00, ssim=0.9, lpips=0.25)
    report = build_report(discover([runs]), "default", ["default"])
    result = b0_check(report, b0, ["drjohnson", "playroom"], "ours_30000", 30000, TOLERANCE)
    assert result["verdict"] == "PASS"
    assert result["scenes"][0]["delta"]["PSNR"] == pytest.approx(0.1)
    write_b0(b0, "room", 31.0)
    assert b0_check(report, b0, ["drjohnson", "room"], "ours_30000", 30000, TOLERANCE)["verdict"] == "INCOMPLETE"
    (b0 / "playroom" / "results.json").write_text(json.dumps({"ours_30000": {"PSNR": 31.0, "SSIM": 0.9,
                                                                             "LPIPS": 0.25}}))
    assert b0_check(report, b0, ["drjohnson", "playroom"], "ours_30000", 30000, TOLERANCE)["verdict"] == "FAIL"


def test_quick_report_compares_against_the_quick_baseline(tmp_path):
    quick = {**PROTOCOL, "max_steps": 3000}
    deviation = {"max_steps": {"base": 30000, "used": 3000, "source": "cli"}}
    for scene in ("drjohnson", "playroom"):
        write_run(tmp_path, scene, "default", 42, 25.0, protocol=quick, deviations=deviation)
        write_run(tmp_path, scene, "mcmc", 42, 25.5, protocol=quick, deviations=deviation,
                  args={"opacity-reg": 0.001})
    report = build_report(discover([tmp_path]), "default{max_steps=3000}", ["default", "mcmc"], REFERENCE,
                          TOLERANCE, ["drjohnson", "playroom"], include_deviating=True)
    variant = "mcmc[opacity-reg=0.001]{max_steps=3000}"
    assert report["complete_scenes"] == ["drjohnson", "playroom"]
    assert report["deltas"]["playroom"][variant]["PSNR"] == pytest.approx(0.5)
    assert report["reference_check"]["verdict"] == "N/A"  # 3.000 vòng: không so với số Inria
    assert any("max_steps 3000" in reason for reason in report["reference_check"]["reasons"])
    assert not [item for item in report["incomplete"] if "missing_variants" in item]


def test_duplicate_runs_and_seeds_are_not_counted_twice(tmp_path):
    write_run(tmp_path, "room", "default", 42, 30.0)
    write_run(tmp_path, "room", "default", 42, 30.2, tag="_again")  # cùng cấu hình, cùng seed, TAG khác
    write_run(tmp_path, "room", "mcmc", 42, 30.4)
    runs = discover([tmp_path, tmp_path / "room"])  # hai --root chồng nhau
    assert len(runs) == 3
    report = build_report(runs, "default", ["default", "mcmc"])
    assert report["scenes"]["room"]["default"]["duplicate_seeds"] == [42]
    assert checks(report, "room", "mcmc")["seeds"] == "DUP"


def test_deltas_and_means_use_common_seeds(tmp_path):
    for seed, base, mcmc in ((42, 30.0, 30.3), (43, 30.2, 30.5), (44, 30.4, None)):
        write_run(tmp_path, "room", "default", seed, base)
        if mcmc is not None:
            write_run(tmp_path, "room", "mcmc", seed, mcmc)
    report = build_report(discover([tmp_path]), "default", ["default", "mcmc"])
    delta = report["deltas"]["room"]["mcmc"]
    assert delta["seeds"] == [42, 43] and delta["PSNR"] == pytest.approx(0.3)  # không lẫn seed 44
    assert checks(report, "room", "mcmc")["seeds"] == "DIFF"
    assert report["mean_over_scenes_seeds"]["room"] == [42, 43]
    assert report["mean_over_scenes"]["default"]["PSNR"] == pytest.approx(30.1)


def test_rebuilt_scene_models_are_not_mixed(tmp_path):
    write_run(tmp_path, "room", "default", 42, 30.0, fingerprint="aaaaaaaaaaaa")
    write_run(tmp_path / "old", "room", "default", 43, 25.0, fingerprint="bbbbbbbbbbbb")
    report = build_report(discover([tmp_path]), "default", ["default"])
    assert sorted(report["scenes"]) == ["room@aaaaaaaa", "room@bbbbbbbb"]
    assert report["warnings"]


def test_b0_verdict_and_comparability(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "drjohnson", "default", 42, 31.0)  # lệch B0 1 dB
    b0 = tmp_path / "b0"
    write_b0(b0, "drjohnson", 30.0, ssim=0.9, lpips=0.25)
    discovered = discover([runs])
    report = build_report(discovered, "default", ["default"])
    # drjohnson FAIL + playroom thiếu: kết luận phải là FAIL, không phải INCOMPLETE
    assert b0_check(report, b0, ["drjohnson", "playroom"], "ours_30000", 30000, TOLERANCE,
                    discovered)["verdict"] == "FAIL"
    other = tmp_path / "alex"
    write_run(other, "drjohnson", "default", 42, 30.0, protocol={**PROTOCOL, "lpips_net": "alex"})
    alex_runs = discover([other])
    result = b0_check(build_report(alex_runs, "default", ["default"]), b0, ["drjohnson"], "ours_30000", 30000,
                      TOLERANCE, alex_runs)
    assert result["verdict"] == "N/A" and "LPIPS alex" in result["reasons"][0]


def test_strict_exit_code(tmp_path):
    write_run(tmp_path, "drjohnson", "default", 42, 29.4)
    write_run(tmp_path, "playroom", "default", 42, 30.0)
    config = tmp_path / "ref.json"
    config.write_text(json.dumps({"reference": {"metrics": REFERENCE}, "tolerance": TOLERANCE,
                                  "scenes": ["drjohnson", "playroom"]}), encoding="utf-8")
    argv = ["--root", str(tmp_path), "--modes", "default", "--reference-config", str(config)]
    assert main(argv) == 0
    assert main([*argv, "--strict"]) == 1  # LPIPS giả lệch 0.012 > 0.01: FAIL


def test_modes_default_to_every_mode_found(tmp_path):
    for mode, psnr in (("default", 30.0), ("mcmc", 30.2), ("mcmc_x", 30.3)):  # mcmc_x: một chế độ tự thêm
        write_run(tmp_path, "room", mode, 42, psnr, args={"noise-lr": 250000} if mode == "mcmc_x" else None)
    report = build_report(discover([tmp_path]), "default", None)
    assert report["modes"] == ["default", "mcmc", "mcmc_x"]
    assert report["variants"] == ["default", "mcmc", "mcmc_x[noise-lr=250000]"]
    assert report["deltas"]["room"]["mcmc_x[noise-lr=250000]"]["PSNR"] == pytest.approx(0.3)
    out = tmp_path / "r.json"
    assert main(["--root", str(tmp_path), "--out", str(out)]) == 0  # không có --modes: mọi chế độ
    assert json.loads(out.read_text(encoding="utf-8"))["modes"] == ["default", "mcmc", "mcmc_x"]
    assert main(["--root", str(tmp_path), "--modes", "default", "mcmc", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["variants"] == ["default", "mcmc"]


def test_part_of_the_scenes_skips_the_published_check_and_strict_ignores_it(tmp_path):
    runs, b0 = tmp_path / "runs", tmp_path / "b0"
    for scene in ("room", "counter", "bicycle"):
        write_run(runs, scene, "default", 42, 30.0)
        write_run(runs, scene, "mcmc", 42, 30.4)
    config = tmp_path / "ref.json"
    config.write_text(json.dumps({"reference": {"metrics": REFERENCE}, "tolerance": TOLERANCE,
                                  "scenes": ["room", "counter", "kitchen", "bicycle"]}), encoding="utf-8")
    out, markdown = tmp_path / "r.json", tmp_path / "r.md"
    argv = ["--root", str(runs), "--reference-config", str(config), "--scenes", "room", "counter",
            "--out", str(out), "--markdown", str(markdown)]
    assert main([*argv, "--strict"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert sorted(report["scenes"]) == ["counter", "room"]  # bicycle không thuộc tập cảnh được chọn
    ref = report["reference_check"]
    assert ref["verdict"] == "SKIPPED" and ref["scenes"] == ["room", "counter"]
    assert "số liệu công bố là trung bình của 4 cảnh" in ref["reasons"][0]
    assert "**SKIPPED**" in markdown.read_text(encoding="utf-8") and "- bỏ qua: chỉ chạy 2 cảnh" in \
        markdown.read_text(encoding="utf-8")
    # B0 chỉ đối chiếu trên các cảnh được chọn; vẫn được tính với --strict
    write_b0(b0, "room", 30.0, ssim=0.9, lpips=0.25)
    assert main([*argv, "--b0-root", str(b0), "--strict"]) == 1  # counter thiếu B0: INCOMPLETE
    assert json.loads(out.read_text(encoding="utf-8"))["b0_check"]["verdict"] == "INCOMPLETE"
    write_b0(b0, "counter", 30.0, ssim=0.9, lpips=0.25)
    assert main([*argv, "--b0-root", str(b0), "--strict"]) == 0
    # đủ mọi cảnh của benchmark: đối chiếu với số công bố như bình thường
    full = ["--root", str(runs), "--reference-config", str(config), "--scenes", "room", "counter", "bicycle",
            "kitchen", "--out", str(out)]
    assert main(full) == 0 and json.loads(out.read_text(encoding="utf-8"))["reference_check"]["verdict"] == \
        "INCOMPLETE"  # kitchen chưa có lượt nào


def test_a_mode_name_with_one_variant_can_be_the_baseline(tmp_path, capsys):
    for seed in (42, 43, 44):
        write_run(tmp_path, "playroom", "default", seed, 29.8)
        write_run(tmp_path, "playroom", "mcmc", seed, 29.9, args={"opacity-reg": 0.001})
        write_run(tmp_path, "playroom", "mcmc_x", seed, 30.0, args={"noise-lr": 250000, "opacity-reg": 0.001})
    out = tmp_path / "r.json"
    assert main(["--root", str(tmp_path), "--baseline", "mcmc", "--modes", "mcmc", "mcmc_x", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["baseline"] == "mcmc[opacity-reg=0.001]"
    delta = report["deltas"]["playroom"]["mcmc_x[noise-lr=250000, opacity-reg=0.001]"]
    assert delta["PSNR"] == pytest.approx(0.1) and delta["seeds"] == [42, 43, 44]
    assert "default" not in report["scenes"]["playroom"]
    write_run(tmp_path, "playroom", "mcmc", 42, 29.9, args={"opacity-reg": 0.01}, tag="_or01")  # biến thể thứ hai
    assert main(["--root", str(tmp_path), "--baseline", "mcmc", "--modes", "mcmc", "--out", str(out)]) == 0
    assert "has several variants" in capsys.readouterr().err
    assert json.loads(out.read_text(encoding="utf-8"))["baseline"] == "mcmc"  # không đoán: baseline thiếu


def test_the_plain_runs_of_a_mode_are_its_baseline_beside_plan_variants(tmp_path):
    for seed in (42, 43, 44):
        write_run(tmp_path, "playroom", "default", seed, 29.8)
        write_run(tmp_path, "playroom", "mcmc", seed, 29.9, args={"opacity-reg": 0.001})  # quy ước mcmc của benchmark
        write_run(tmp_path, "playroom", "mcmc", seed, 30.1, args={"opacity-reg": 0.005}, tag="_or0005",
                  run_tag="or0005")  # biến thể của kế hoạch: tham số riêng, có tag
    out = tmp_path / "r.json"
    argv = ["--root", str(tmp_path), "--baseline", "mcmc", "--modes", "mcmc", "--out", str(out)]
    assert main([*argv, "--tags", "or0005"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["baseline"] == "mcmc[opacity-reg=0.001]"
    assert report["deltas"]["playroom"]["mcmc[opacity-reg=0.005]"]["PSNR"] == pytest.approx(0.2)
    assert not any("with a tag" in warning for warning in report["warnings"])


def test_runs_with_a_tag_stay_out_unless_their_tag_is_asked_for(tmp_path):
    # tune_mcmc ghi các biến thể có tag vào cùng thư mục với thực nghiệm chính: bảng của thực nghiệm chính không lẫn chúng
    for scene in ("room", "counter"):
        for seed in (42, 43, 44):
            write_run(tmp_path, scene, "default", seed, 29.0)
            write_run(tmp_path, scene, "mcmc", seed, 29.5)
    write_run(tmp_path, "room", "mcmc", 42, 30.5, args={"opacity-reg": 0.001}, tag="_or0001", run_tag="or0001")
    write_run(tmp_path, "room", "mcmc", 42, 29.0, args={"random-bkgd": True}, tag="_rbg", run_tag="rbg")
    out = tmp_path / "r.json"
    assert main(["--root", str(tmp_path), "--modes", "default", "mcmc", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["variants"] == ["default", "mcmc"] and report["complete_scenes"] == ["counter", "room"]
    assert report["mean_over_scenes_seeds"] == {"counter": [42, 43, 44], "room": [42, 43, 44]}
    assert ("left out 2 run(s) with a tag (or0001, rbg): variants of another experiment plan; pass their tags with "
            "--tags (TAGS=... for scripts/run_evaluate.sh) to include them") in report["warnings"]
    assert main(["--root", str(tmp_path), "--modes", "default", "mcmc", "--tags", "or0001", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["variants"] == ["default", "mcmc", "mcmc[opacity-reg=0.001]"]
    assert report["scenes"]["room"]["mcmc[opacity-reg=0.001]"]["n"] == 1
    assert any("left out 1 run(s) with a tag (rbg)" in warning for warning in report["warnings"])


def test_filtered_runs_get_a_before_after_table(tmp_path):
    for seed, psnr in ((42, 30.0), (43, 30.2)):
        write_run(tmp_path, "room", "default", seed, 29.0)
        write_filtered(write_run(tmp_path, "room", "mcmc", seed, psnr), psnr - 0.1)
    failed = write_run(tmp_path, "room", "default", 44, 29.1)
    write_filtered(failed, 0.0, status="FAILED")
    report = build_report(discover([tmp_path]), "default", ["default", "mcmc"])
    after = report["scenes"]["room"]["mcmc"]["filtered"]
    assert after["n"] == 2 and after["seeds"] == [42, 43]
    assert after["metrics"]["PSNR"] == pytest.approx(30.0) and after["delta"]["PSNR"] == pytest.approx(-0.1)
    assert after["delta"]["LPIPS"] == pytest.approx(0.01) and after["kept_fraction"] == pytest.approx(0.75)
    assert after["num_gaussians_before"] == 2_000_000 and after["ply_size_mb"] == pytest.approx(3.0)
    assert report["scenes"]["room"]["default"]["filtered"] is None  # lượt lọc lỗi không có số
    assert {"run_dir": str(failed), "scene": "room", "variant": "default", "filtered": "FAILED",
            "problem": "FilterError: eval: exit code 3"} in report["incomplete"]
    assert report["deltas"]["room"]["mcmc"]["PSNR"] == pytest.approx(1.1)  # bảng chính: mô hình chưa lọc
    text = format_markdown(report)
    assert "Sau khi lọc Gaussian (src.training.prune: bỏ opacity < 0.02 hoặc trục lớn nhất > 0.1 × scene_scale)" in text
    assert "| room | mcmc | 2 | 30.000 | 0.8900 | 0.2600 | -0.100 | -0.0100 | +0.0100 | 1,500,000 (75.0%) |" in text
    write_filtered(write_run(tmp_path, "room", "mcmc", 44, 30.1), 30.0, min_opacity=0.05)
    report = build_report(discover([tmp_path]), "default", ["default", "mcmc"])
    assert any("filtered with different thresholds" in warning for warning in report["warnings"])
    assert ("| 3.00 | bỏ opacity < 0.02 hoặc trục lớn nhất > 0.1 × scene_scale; bỏ opacity < 0.05 hoặc trục lớn nhất "
            "> 0.1 × scene_scale |") in format_markdown(report)  # một dòng trộn hai bộ ngưỡng: ghi cả hai


def test_filtered_rows_name_their_thresholds_when_rows_differ(tmp_path):
    write_run(tmp_path, "room", "default", 42, 29.0)
    write_filtered(write_run(tmp_path, "room", "mcmc", 42, 30.0), 29.9)
    write_filtered(write_run(tmp_path, "room", "default", 43, 29.1), 29.0, min_opacity=0.0123456789)
    write_run(tmp_path, "room", "mcmc", 43, 30.1)
    text = format_markdown(build_report(discover([tmp_path]), "default", ["default", "mcmc"]))
    assert "| PLY (MB) trước → sau | Render (ms/ảnh) | Ngưỡng |" in text
    rows = [line for line in text.splitlines() if line.startswith("| room |") and "→" in line]
    assert rows[0].endswith("| bỏ opacity < 0.0123456789 hoặc trục lớn nhất > 0.1 × scene_scale |")  # default
    assert rows[1].endswith("| bỏ opacity < 0.02 hoặc trục lớn nhất > 0.1 × scene_scale |")  # mcmc
