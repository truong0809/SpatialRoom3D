"""Test cấu hình chế độ huấn luyện và tham số dòng lệnh sinh cho simple_trainer.py của gsplat.

Dùng file cấu hình thật configs/base.yaml (mục train_modes); các số Gaussian là dữ liệu giả.
"""

from pathlib import Path

import pytest

from src.training.profiles import (
    checkpoint_path,
    eval_args,
    load_config,
    normalize_flag,
    parse_assignments,
    resolve_cap_max,
    resolve_mode,
    to_cli,
    trainer_args,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"  # mục train_modes


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


def flag_value(argv, flag):
    index = argv.index(flag)
    return argv[index + 1]


def baseline_summary(config, count=2_500_000, mode="default", status="OK", seed=42, deviations=None, protocol=None):
    return {"mode": mode, "status": status, "num_gaussians": count, "run_dir": "/x/default", "seed": seed,
            "split_sha256": "abc", "protocol": protocol or dict(config["protocol"]), "deviations": deviations or {}}


CUSTOM_MODE = {  # một chế độ người dùng tự thêm vào configs/base.yaml: protocol riêng, số Gaussian cố định
    "description": "mcmc 40k vòng, antialiased, cap cố định",
    "preset": "mcmc",
    "protocol": {"max_steps": 40000, "antialiased": True},
    "args": {"strategy.refine_stop_iter": 35000},
    "cap_max": {"policy": "fixed", "value": 3_500_000},
}


def with_mode(config, name, entry):
    return {**config, "modes": {**config["modes"], name: entry}}


def test_real_config_defines_both_modes(config):
    assert config["baseline_mode"] == "default"
    assert set(config["modes"]) == {"default", "mcmc"}
    # mcmc không có số Gaussian cố định: luôn lấy đúng số Gaussian cuối của lượt default tương ứng
    assert config["modes"]["mcmc"]["cap_max"] == {"policy": "match_baseline"}
    assert config["protocol"]["lpips_net"] == "vgg"  # cùng mạng LPIPS với Inria
    assert len(config["trainer"]["commit"]) == 40


def test_default_mode_arguments(config):
    spec = resolve_mode(config, "default")
    argv = trainer_args(spec, Path("/data/scene"), Path("/out/run"))
    assert argv[0] == "default"
    assert flag_value(argv, "--data-dir") == "/data/scene"
    assert flag_value(argv, "--result-dir") == "/out/run"
    assert flag_value(argv, "--data-factor") == "1"
    assert flag_value(argv, "--test-every") == "8"
    assert flag_value(argv, "--max-steps") == "30000"
    assert flag_value(argv, "--lpips-net") == "vgg"
    assert flag_value(argv, "--eval-steps") == "-1"
    assert flag_value(argv, "--save-steps") == "30000"
    assert flag_value(argv, "--ply-steps") == "30000"
    assert "--no-antialiased" in argv and "--save-ply" in argv and "--no-disable-video" in argv
    assert flag_value(argv, "--init-type") == "sfm"  # cả hai chế độ khởi tạo từ điểm SfM
    assert "--disable-viewer" in argv
    assert not any(token.startswith("--strategy.") for token in argv)
    assert spec.deviations == {} and spec.final_step == 29999


def test_mcmc_needs_baseline_budget(config):
    spec = resolve_mode(config, "mcmc")
    with pytest.raises(ValueError, match="cap-max-from"):
        resolve_cap_max(spec)
    cap, source = resolve_cap_max(spec, baseline_summary=baseline_summary(config, 2_345_678), seed=42)
    assert cap == 2_345_678 and source["policy"] == "match_baseline" and source["baseline_seed"] == 42
    argv = trainer_args(spec, Path("/s"), Path("/o"), cap)
    assert argv[0] == "mcmc" and flag_value(argv, "--strategy.cap-max") == "2345678"
    with pytest.raises(ValueError, match="expected 'default'"):
        resolve_cap_max(spec, baseline_summary=baseline_summary(config, mode="mcmc"))
    with pytest.raises(ValueError, match="status must be OK"):
        resolve_cap_max(spec, baseline_summary=baseline_summary(config, status="FAILED"))
    with pytest.raises(ValueError, match="different protocol"):  # vd. lượt chạy thử 3000 vòng
        resolve_cap_max(spec, baseline_summary=baseline_summary(config, protocol={**config["protocol"],
                                                                                  "max_steps": 3000}))
    # lượt chạy thử: baseline và mcmc cùng đổi max_steps -> cùng protocol nên dùng được
    quick = resolve_mode(config, "mcmc", protocol_overrides={"max_steps": 3000})
    quick_base = baseline_summary(config, count=150_000, protocol=dict(quick.protocol),
                                  deviations={"max_steps": {"base": 30000, "used": 3000, "source": "cli"}})
    assert resolve_cap_max(quick, baseline_summary=quick_base, seed=42)[0] == 150_000
    with pytest.raises(ValueError, match="different protocol"):  # nhưng lượt chuẩn thì không
        resolve_cap_max(spec, baseline_summary=quick_base, seed=42)
    with pytest.raises(ValueError, match="seed 43"):
        resolve_cap_max(spec, baseline_summary=baseline_summary(config, seed=43), seed=42)
    cap, source = resolve_cap_max(spec, cli_value=1_000_000)
    assert cap == 1_000_000 and source == {"policy": "cli", "value": 1_000_000}


def test_custom_mode_with_its_own_protocol_and_fixed_budget(config):
    """Chế độ tự thêm: protocol riêng bị ghi là sai khác (nguồn "mode"); cap_max cố định không cần lượt default."""
    custom = with_mode(config, "mcmc_custom", CUSTOM_MODE)
    spec = resolve_mode(custom, "mcmc_custom")
    assert set(spec.deviations) == {"max_steps", "antialiased"}
    assert all(deviation["source"] == "mode" for deviation in spec.deviations.values())
    cap, source = resolve_cap_max(spec)
    assert cap == 3_500_000 and source == {"policy": "fixed", "value": 3_500_000}
    argv = trainer_args(spec, Path("/s"), Path("/o"), cap)
    assert flag_value(argv, "--max-steps") == "40000"
    assert flag_value(argv, "--strategy.refine-stop-iter") == "35000"
    assert flag_value(argv, "--strategy.cap-max") == "3500000"
    assert "--antialiased" in argv and flag_value(argv, "--save-steps") == "40000"
    assert checkpoint_path(spec, Path("/o")) == Path("/o/ckpts/ckpt_39999_rank0.pt")
    same = resolve_mode(custom, "mcmc_custom", protocol_overrides={"max_steps": 30000})
    assert set(same.deviations) == {"antialiased"}  # đặt lại đúng giá trị gốc thì không còn là sai khác
    for cap_max, message in (({"policy": "fixed"}, "must be an integer"), ({"policy": "auto"}, "must be one of")):
        with pytest.raises(ValueError, match=message):
            resolve_mode(with_mode(config, "bad", {**CUSTOM_MODE, "cap_max": cap_max}), "bad")
    with pytest.raises(ValueError, match="only applies to the mcmc preset"):
        resolve_mode(with_mode(config, "bad", {**CUSTOM_MODE, "preset": "default"}), "bad")


def test_eval_arguments_reuse_training_configuration(config):
    spec = resolve_mode(config, "mcmc")
    train = trainer_args(spec, Path("/s"), Path("/o"), 123)
    evaluation = eval_args(spec, Path("/s"), Path("/o"), 123)
    assert evaluation[:len(train)] == train
    assert evaluation[len(train):] == ["--ckpt", "/o/ckpts/ckpt_29999_rank0.pt"]


def test_reserved_flags_and_protocol_overrides(config):
    for key in ("max_steps", "strategy.cap_max", "--data-dir", "lpips_net", "batch_size"):
        with pytest.raises(ValueError, match="controlled by the protocol"):
            resolve_mode(config, "default", set_args={key: 1})
    spec = resolve_mode(config, "mcmc", set_args={"opacity_reg": 0.001},
                        protocol_overrides={"max_steps": 3000})
    assert spec.args == {"opacity-reg": 0.001}
    assert spec.deviations == {"max_steps": {"base": 30000, "used": 3000, "source": "cli"}}
    with pytest.raises(ValueError, match="unknown protocol key"):
        resolve_mode(config, "default", protocol_overrides={"steps": 1})
    with pytest.raises(ValueError, match="integer"):
        resolve_mode(config, "default", protocol_overrides={"max_steps": "many"})
    with pytest.raises(ValueError, match="unknown mode"):
        resolve_mode(config, "nope")


def test_cli_helpers():
    assert normalize_flag("--strategy.refine_stop_iter") == "strategy.refine-stop-iter"
    assert to_cli("strategy.verbose", False) == ["--strategy.no-verbose"]
    assert to_cli("antialiased", False) == ["--no-antialiased"]
    assert to_cli("save-ply", True) == ["--save-ply"]
    assert to_cli("eval-steps", [7000, 30000]) == ["--eval-steps", "7000", "30000"]
    assert parse_assignments(["a=1", "b=0.5", "c=true", "d=vgg", "e=-2"]) == {
        "a": 1, "b": 0.5, "c": True, "d": "vgg", "e": -2}
    with pytest.raises(ValueError):
        parse_assignments(["novalue"])
    with pytest.raises(ValueError):
        normalize_flag("bad flag")
