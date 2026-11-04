"""Đọc và kiểm tra configs/product.yaml: cấu hình duy nhất của mô hình chính (sản phẩm web).

`${WS}` trong các đường dẫn (trainer.dir, web.dir, output_root) được thay bằng thư mục gốc trên pod: `--ws`, biến môi
trường WS, hoặc mặc định /workspace (cùng quy ước với configs/base.yaml).

Cách dùng (cần PyYAML: python3 của pod hoặc python của env gs-gsplat):
  python3 -m src.product.config                  # in cấu hình đã kiểm tra (JSON)
  python3 -m src.product.config --get web.dir    # in một giá trị (khóa nối bằng dấu chấm)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_DIR / "configs" / "product.yaml"
DEFAULT_WS = "/workspace"
ROOM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # cùng quy tắc tên cảnh của scripts/pipeline/prepare_scene.sh
PRESETS = ("default", "mcmc")
WEB_FORMATS = ("splat",)

# khóa bắt buộc -> kiểu
SCHEMA = {
    "name": str,
    "model.name": str,
    "model.preset": str,
    "model.args": dict,
    "model.cap_max": int,
    "trainer.commit": str,
    "trainer.dir": str,
    "trainer.env": str,
    "trainer.seed": int,
    "training.data_factor": int,
    "training.test_every": int,
    "training.max_steps": int,
    "training.antialiased": bool,
    "training.init_type": str,
    "training.lpips_net": str,
    "web.format": str,
    "web.dir": str,
    "output_root": str,
}
PATH_KEYS = ("trainer.dir", "web.dir", "output_root")
# Tham số của simple_trainer.py do train_main.py đặt (từ mục training / model.cap_max hoặc cố định): không đặt qua model.args
RESERVED_FLAGS = {
    "data-dir", "result-dir", "data-factor", "test-every", "max-steps", "antialiased", "init-type", "lpips-net",
    "strategy.cap-max", "eval-steps", "save-steps", "ply-steps", "save-ply", "disable-viewer", "disable-video",
    "ckpt", "steps-scaler", "batch-size",
}


class ConfigError(ValueError):
    """configs/product.yaml sai; thông điệp chỉ rõ khóa."""


def flag_name(key: str) -> str:
    """'strategy.noise_lr' -> 'strategy.noise-lr' (tên tham số của simple_trainer.py theo quy ước của tyro)."""
    key = str(key).strip().lstrip("-")
    if not re.fullmatch(r"[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)*", key.replace("-", "_")):
        raise ConfigError(f"invalid trainer argument name in model.args: {key!r}")
    return ".".join(part.replace("_", "-") for part in key.split("."))


def get(config: dict, dotted: str):
    value = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def _type_ok(value, expected) -> bool:
    if expected is bool:
        return isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected)


def validate(config) -> dict:
    if not isinstance(config, dict):
        raise ConfigError("the product config must be a mapping")
    for key, expected in SCHEMA.items():
        try:
            value = get(config, key)
        except KeyError:
            raise ConfigError(f"missing key {key}") from None
        if not _type_ok(value, expected):
            raise ConfigError(f"{key} must be {expected.__name__}, got {value!r}")
    model, training = config["model"], config["training"]
    if model["preset"] not in PRESETS:
        raise ConfigError(f"model.preset must be one of {PRESETS}, got {model['preset']!r}")
    if model["cap_max"] <= 0:
        raise ConfigError("model.cap_max must be a positive integer")
    for key in ("data_factor", "max_steps"):
        if training[key] < 1:
            raise ConfigError(f"training.{key} must be >= 1")
    if training["test_every"] < 2:  # gsplat: ảnh i là ảnh train khi i % test_every != 0; 1 thì không còn ảnh train
        raise ConfigError("training.test_every must be >= 2")
    if not re.fullmatch(r"[0-9a-f]{40}", config["trainer"]["commit"]):
        raise ConfigError("trainer.commit must be a full 40-character git commit hash")
    if config["web"]["format"] not in WEB_FORMATS:
        raise ConfigError(f"web.format must be one of {WEB_FORMATS}, got {config['web']['format']!r}")
    for key, value in model["args"].items():
        flag = flag_name(key)
        if flag in RESERVED_FLAGS:
            raise ConfigError(f"model.args.{key}: --{flag} is set by src/product/train_main.py "
                              "(change the training section or model.cap_max instead)")
        if value is None or isinstance(value, dict):
            raise ConfigError(f"model.args.{key} must be a number, a string, true / false or a list")
    return config


def load(path: Path | str | None = None, ws: str | None = None) -> dict:
    import yaml  # PyYAML: có trong python3 của template Runpod và trong env gs-gsplat

    path = Path(path or DEFAULT_CONFIG)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot read {path}: {error}") from None
    config = validate(data)
    ws = ws or os.environ.get("WS") or DEFAULT_WS
    for key in PATH_KEYS:
        section, _, leaf = key.rpartition(".")
        holder = get(config, section) if section else config
        holder[leaf] = holder[leaf].replace("${WS}", ws)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cấu hình mô hình chính của sản phẩm (configs/product.yaml)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ws", default=None, help="thư mục gốc trên pod (mặc định: biến WS, rồi /workspace)")
    parser.add_argument("--get", metavar="KHÓA", help="in một giá trị, vd. web.dir")
    args = parser.parse_args(argv)
    try:
        config = load(args.config, args.ws)
        value = config if args.get is None else get(config, args.get)
    except ConfigError as error:
        print(f"ERROR: {args.config}: {error}", file=sys.stderr)
        return 1
    except KeyError:
        print(f"ERROR: {args.config} has no key {args.get}", file=sys.stderr)
        return 1
    print(value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
