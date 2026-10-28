"""Kế hoạch một thực nghiệm (configs/experiments/<tên>.yaml): các chế độ huấn luyện, biến thể tham số, seed, benchmark và
cảnh, có chạy B0 không, có lọc Gaussian sau khi train không, ước lượng. scripts/pod/run_experiment.sh đọc kế hoạch qua
module này; module chỉ đọc và kiểm tra cấu hình, không chạy gì.

    python3 -m src.training.experiment --plan main                        # các dòng "khóa|giá trị" cho bash
    python3 -m src.training.experiment --export main --out kế_hoạch.json  # bản chụp kế hoạch đã kiểm tra
    python3 -m src.training.experiment --space main                       # chỗ còn cần cho các lượt chưa xong

Kiểm tra:
  * `modes` có trong train_modes của configs/base.yaml và gồm chế độ baseline (default); `also_against` (tùy chọn):
    các chế độ khác trong `modes` dùng làm mốc cho thêm một bảng so sánh (vd. các biến thể của mcmc so với mcmc);
  * `variants` (tùy chọn): mỗi mục `{mode, tag, set}` là một bộ tham số của simple_trainer.py cho một chế độ trong
    `modes` (không phải baseline), chạy vào thư mục riêng `<chế_độ>_<tag>_s<seed>` như mọi lượt khác (cùng protocol, cùng
    ngân sách Gaussian với default). `set` chỉ đổi tham số của trainer, không đổi protocol (src/training/profiles.py từ
    chối khóa protocol); giá trị là số, true / false hoặc chữ không dấu cách. So cấu hình thật của các lượt trên từng
    benchmark (preset, protocol, ngân sách Gaussian, tham số; 250000 và 250000.0 là một) sau khi ghép quy ước mcmc của
    benchmark (`benchmark.gsplat.mcmc_set`, cho chế độ preset mcmc trừ các khóa chính chế độ đặt): hai lượt CÙNG CHẾ ĐỘ
    mà cùng cấu hình là lỗi (bảng so sánh gom lượt theo chế độ + tham số, hai lượt sẽ thành một); hai chế độ khác tên mà
    cùng cấu hình (vd. một chế độ mới chọn đúng opacity_reg mà Deep Blending đã dùng cho mcmc) chỉ là cảnh báo
    (`warnings`): cả hai vẫn train, mỗi chế độ một dòng trong bảng;
  * chế độ hoặc biến thể có loss độ sâu (`depth_loss: true`) được liệt kê để kiểm tra điểm SfM trước khi train
    (src/training/depth_check.py);
  * `set` có thể bật biến thể có code bằng `sr3d.kr: true` / `sr3d.mono: true` (src/training/trainer_hooks.py; kr chỉ cho
    preset mcmc); lượt có `sr3d.mono` được liệt kê (`mono_runs`) để scripts/pod/run_experiment.sh cài Depth Anything V2;
  * `post_filter` (tùy chọn): `{min_opacity, max_scale}`, lọc Gaussian mỗi lượt đã xong rồi chấm lại
    (src/training/prune.py);
  * `seeds` là số nguyên không âm, không trùng;
  * mỗi `benchmarks[].dataset` là một luồng `kind: benchmark` có mục download (src/datasets/fetch.py), `scenes` thuộc
    danh sách cảnh của luồng, `b0` là true / false;
  * `estimate.gpu_hours`, `estimate.disk_gb`: số dương (chỉ để in và kiểm tra chỗ trống).
Chỗ còn cần (`--space`, xem `space_needs`): mỗi lượt gsplat chưa xong của một seed tính bằng dung lượng thật của lượt cùng
cảnh, cùng chế độ / biến thể đã xong ở seed khác (lớn nhất, gồm cả bản đã lọc trong `filtered/`); chưa có thì bằng phần
chia đều của `estimate.disk_gb` cho mọi lượt. Kế hoạch có `post_filter`: lượt đã xong mà chưa lọc với đúng ngưỡng còn cần
chỗ cho bản đã lọc.
Cần PyYAML (python3 của template Runpod thường có, ô 1 của notebook kiểm tra; hoặc env gs-gsplat).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path

from src.datasets.registry import CONFIG_DIR, load_registry, read_yaml
from src.training.profiles import hook_names, normalize_flag, parse_assignments, parse_value, resolve_mode

EXPERIMENTS = "experiments"  # configs/experiments/<tên>.yaml
NAME = re.compile(r"[a-z0-9_]+")
VARIANT_TAG = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")
SET_KEY = re.compile(r"[A-Za-z0-9_.-]+")
SET_TEXT = re.compile(r"[A-Za-z0-9_.+-]+")  # giá trị trong dòng `variants|` của --plan (không dấu cách, ':' hay ',')
FILTER_KEYS = ("min_opacity", "max_scale")
DEPTH_FLAG = "depth-loss"  # simple_trainer.py @6e8c837: Config.depth_loss


def experiment_path(config_dir: Path, name: str) -> Path:
    if not NAME.fullmatch(name):
        raise ValueError(f"invalid experiment name {name!r} (lowercase letters, digits and '_')")
    return Path(config_dir) / EXPERIMENTS / f"{name}.yaml"


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{where} must be a positive number")
    return value


def set_text(value, where: str) -> str:
    """Giá trị một tham số của biến thể dạng chữ, đọc lại đúng giá trị đó (src.training.profiles.parse_value)."""
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (int, float, str)):
        text = repr(value) if isinstance(value, float) else str(value)
    else:
        raise ValueError(f"{where} must be a number, true / false or a word, not {value!r}")
    back = parse_value(text)
    if not SET_TEXT.fullmatch(text) or type(back) is not type(value) or back != value:
        raise ValueError(f"{where}: value {value!r} cannot be passed on the command line unchanged (use a number, "
                         "true / false, or a word of letters, digits and '_ . + -')")
    return text


def _variants(raw, where: str, modes: list[str], baseline: str, train_modes: dict) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{where}: variants must be a list of {{mode, tag, set}}")
    known = train_modes.get("modes") or {}
    variants, runs = [], set()
    for index, item in enumerate(raw):
        here = f"{where}: variants[{index}]"
        if not isinstance(item, dict) or set(item) - {"mode", "tag", "set", "description"}:
            raise ValueError(f"{here} must be a mapping with mode, tag, set (and an optional description)")
        mode, tag, values = item.get("mode"), item.get("tag"), item.get("set")
        if mode not in modes or mode == baseline:
            raise ValueError(f"{here}: mode must be one of the plan's modes other than {baseline!r}, not {mode!r}")
        if not isinstance(tag, str) or not VARIANT_TAG.fullmatch(tag):
            raise ValueError(f"{here}: tag must be lowercase letters and digits, words joined by '_' (got {tag!r})")
        run = f"{mode}_{tag}"
        if run in known or run in runs:
            raise ValueError(f"{here}: the results folder {run}_s<seed> is already used by "
                             f"{'the mode' if run in known else 'another variant'} {run}; choose another tag")
        if not isinstance(values, dict) or not values:
            raise ValueError(f"{here}: set must be a non-empty mapping of simple_trainer.py arguments")
        texts = {}
        for key, value in values.items():
            if not isinstance(key, str) or not SET_KEY.fullmatch(key):
                raise ValueError(f"{here}: invalid argument name {key!r}")
            texts[key] = set_text(value, f"{here}: set.{key}")
        try:
            resolve_mode(train_modes, mode, dict(values))  # khóa protocol / tham số do wrapper quản lý: ValueError
        except ValueError as error:
            raise ValueError(f"{here}: {error}") from error
        runs.add(run)
        variants.append({"mode": mode, "tag": tag, "run": run, "set": dict(values), "set_text": texts,
                         "description": str(item.get("description") or "")})
    return variants


def _mcmc_set(entry: dict) -> dict:
    """Tham số chung của benchmark cho mọi chế độ preset mcmc (benchmark.gsplat.mcmc_set, vd. opacity_reg=0.001)."""
    text = (((entry.get("benchmark") or {}).get("gsplat") or {}).get("mcmc_set")) or ""
    return parse_assignments([text]) if text else {}


def run_spec(train_modes: dict, mode: str, extra: dict, mcmc_set: dict):
    """Cấu hình mà một lượt dùng thật (như scripts/pipeline/train_gsplat_modes.sh ghép): quy ước mcmc của benchmark
    (chế độ preset mcmc, trừ các khóa chính chế độ đã đặt trong configs/base.yaml), tham số của chế độ, rồi tham số của
    biến thể (ưu tiên tăng dần)."""
    entry = (train_modes.get("modes") or {}).get(mode) or {}
    own = {normalize_flag(key) for key in entry.get("args") or {}}
    convention = ({key: value for key, value in mcmc_set.items() if normalize_flag(key) not in own}
                  if entry.get("preset") == "mcmc" else {})
    return resolve_mode(train_modes, mode, {**convention, **extra})


def run_args(train_modes: dict, mode: str, extra: dict, mcmc_set: dict) -> dict:
    """Tham số của trainer mà một lượt dùng thật (run_spec)."""
    return run_spec(train_modes, mode, extra, mcmc_set).args


def _canonical(value):
    """Giá trị để so cấu hình hai lượt: số nguyên và số thực bằng nhau là một (250000 và 250000.0)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return float(value)


def configuration_key(spec) -> str:
    """Cấu hình huấn luyện của một lượt, không kể tên: preset, protocol, ngân sách Gaussian, tham số."""
    return json.dumps({"preset": spec.preset, "protocol": spec.protocol, "cap_max": spec.cap_max_policy,
                       "args": {key: _canonical(value) for key, value in spec.args.items()}}, sort_keys=True)


def _post_filter(raw, where: str) -> dict | None:
    if raw is None or raw is False:
        return None
    if not isinstance(raw, dict) or set(raw) != set(FILTER_KEYS):
        raise ValueError(f"{where}: post_filter must be a mapping with min_opacity and max_scale")
    opacity, scale = raw["min_opacity"], raw["max_scale"]
    for key, value in (("min_opacity", opacity), ("max_scale", scale)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{where}: post_filter.{key} must be a number >= 0 (0 = this criterion is off)")
    if opacity >= 1:
        raise ValueError(f"{where}: post_filter.min_opacity must be below 1")
    if opacity == 0 and scale == 0:
        raise ValueError(f"{where}: post_filter removes nothing with both thresholds at 0; drop the section instead")
    return {"min_opacity": float(opacity), "max_scale": float(scale)}


def load_experiment(name: str, config_dir: Path = CONFIG_DIR, ws: str = "/workspace") -> dict:
    """Đọc và kiểm tra một kế hoạch; lỗi là ValueError có tên file."""
    path = experiment_path(config_dir, name)
    if not path.is_file():
        available = sorted(p.stem for p in (Path(config_dir) / EXPERIMENTS).glob("*.yaml"))
        raise ValueError(f"no experiment {name!r}: {path} not found (available: {available})")
    raw = read_yaml(path)
    where = f"{path.name}"
    registry = load_registry(config_dir, ws)
    train_modes = registry.get("train_modes") or {}
    known_modes = train_modes.get("modes") or {}
    baseline = train_modes.get("baseline_mode", "default")

    modes = raw.get("modes")
    if not isinstance(modes, list) or not modes or not all(isinstance(m, str) for m in modes):
        raise ValueError(f"{where}: modes must be a non-empty list of mode names")
    unknown = [mode for mode in modes if mode not in known_modes]
    if unknown:
        raise ValueError(f"{where}: unknown mode(s) {unknown}; configs/base.yaml has {sorted(known_modes)}")
    if baseline not in modes:
        raise ValueError(f"{where}: modes must include the baseline mode {baseline!r}")
    if len(set(modes)) != len(modes):
        raise ValueError(f"{where}: modes are listed twice")
    modes = [baseline] + [mode for mode in modes if mode != baseline]  # baseline chạy trước, các chế độ khác lấy ngân sách từ nó
    also = raw.get("also_against") or []
    if (not isinstance(also, list) or any(mode not in modes or mode == baseline for mode in also)
            or len(set(also)) != len(also)):
        raise ValueError(f"{where}: also_against must list distinct modes of the plan other than {baseline!r}")
    variants = _variants(raw.get("variants"), where, modes, baseline, train_modes)

    seeds = raw.get("seeds")
    if (not isinstance(seeds, list) or not seeds
            or not all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds)):
        raise ValueError(f"{where}: seeds must be a non-empty list of non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{where}: a seed is listed twice")

    benches = raw.get("benchmarks")
    if not isinstance(benches, list) or not benches:
        raise ValueError(f"{where}: benchmarks must be a non-empty list")
    datasets = registry.get("datasets") or {}
    plan_benches = []
    depth_runs, mono_runs = set(), set()
    warnings = []
    for index, bench in enumerate(benches):
        here = f"{where}: benchmarks[{index}]"
        if not isinstance(bench, dict):
            raise ValueError(f"{here} must be a mapping with dataset, scenes, b0")
        dataset = bench.get("dataset")
        entry = datasets.get(dataset)
        if entry is None or entry.get("kind") != "benchmark":
            benchmarks = sorted(n for n, e in datasets.items() if e.get("kind") == "benchmark")
            raise ValueError(f"{here}: dataset must be one of the benchmarks {benchmarks}, not {dataset!r}")
        if not entry.get("download"):
            raise ValueError(f"{here}: configs/{dataset}.yaml has no download section")
        known = [str(scene) for scene in entry.get("scenes") or []]
        scenes = bench.get("scenes") or known
        if not isinstance(scenes, list) or not all(isinstance(s, str) for s in scenes):
            raise ValueError(f"{here}: scenes must be a list of scene names")
        if [s for s in scenes if s not in known] or len(set(scenes)) != len(scenes):
            raise ValueError(f"{here}: scenes must be distinct scenes of {dataset} ({known}), got {scenes}")
        b0 = bench.get("b0", False)
        if not isinstance(b0, bool):
            raise ValueError(f"{here}: b0 must be true or false")
        if any(other["dataset"] == dataset for other in plan_benches):
            raise ValueError(f"{here}: {dataset} is listed twice")
        # cấu hình thật của từng lượt trên benchmark này: hai lượt cùng cấu hình là train hai lần cùng một thứ (bảng
        # so sánh còn gộp hai lượt cùng chế độ, cùng tham số làm một)
        mcmc_set = _mcmc_set(entry)
        seen = {}  # cấu hình -> [(lượt, chế độ)]
        for run, mode, extra in [(m, m, {}) for m in modes] + [(v["run"], v["mode"], v["set"]) for v in variants]:
            spec = run_spec(train_modes, mode, extra, mcmc_set)
            key = configuration_key(spec)
            twins = seen.setdefault(key, [])
            same_mode = [other for other, other_mode in twins if other_mode == mode]
            detail = (f"preset {spec.preset}, parameters {spec.args or 'none'}; the benchmark adds "
                      f"{mcmc_set or 'nothing'} to mcmc modes that do not set it themselves")
            if same_mode:
                raise ValueError(f"{where}: {run} and {same_mode[0]} would train {mode} with the same configuration on "
                                 f"{dataset} ({detail}): the comparison table would merge them; change or drop one")
            for other, _ in twins:
                warnings.append(f"{run} and {other} train the same configuration on {dataset} ({detail}): both are "
                                "trained, as separate rows of the comparison table")
            twins.append((run, mode))
            if spec.args.get(DEPTH_FLAG) is True:
                depth_runs.add(run)
            if "mono" in hook_names(spec):
                mono_runs.add(run)
        plan_benches.append({"dataset": dataset, "scenes": scenes, "b0": b0,
                             "all_scenes": sorted(scenes) == sorted(known)})

    estimate = raw.get("estimate") or {}
    if not isinstance(estimate, dict):
        raise ValueError(f"{where}: estimate must be a mapping")
    gpu_hours = _number(estimate.get("gpu_hours"), f"{where}: estimate.gpu_hours")
    disk_gb = _number(estimate.get("disk_gb"), f"{where}: estimate.disk_gb")
    run_mode = {**{mode: mode for mode in modes}, **{variant["run"]: variant["mode"] for variant in variants}}
    runs = list(run_mode)
    depth_runs = [run for run in runs if run in depth_runs]
    mono_runs = [run for run in runs if run in mono_runs]
    test_every = sorted({resolve_mode(train_modes, run_mode[run]).protocol["test_every"] for run in depth_runs})
    if len(test_every) > 1:
        raise ValueError(f"{where}: the depth-loss runs {depth_runs} use different test_every values {test_every}")
    return {
        "name": name,
        "description": str(raw.get("description") or ""),
        "config_file": str(path),
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "modes": modes,
        "also_against": also,
        "variants": [{key: value for key, value in variant.items() if key != "set_text"} for variant in variants],
        "variant_lines": [f"{v['mode']}:{v['tag']}:" + ",".join(f"{k}={t}" for k, t in v["set_text"].items())
                          for v in variants],
        "runs": runs,
        "depth_runs": depth_runs,
        "depth_test_every": test_every[0] if test_every else None,
        "mono_runs": mono_runs,
        "post_filter": _post_filter(raw.get("post_filter"), where),
        "seeds": seeds,
        "benchmarks": plan_benches,
        "estimate": {"gpu_hours": gpu_hours, "disk_gb": disk_gb},
        "warnings": warnings,
    }


def tree_bytes(path: Path) -> int:
    """Tổng kích thước khai báo của một cây thư mục, như `du -sb`: file, thư mục, liên kết (không đi theo liên kết);
    liên kết cứng đếm một lần."""
    total, seen, stack = 0, set(), [str(path)]
    while stack:
        current = stack.pop()
        try:
            info = os.lstat(current)
        except OSError:
            continue
        if info.st_nlink > 1 and not stat.S_ISDIR(info.st_mode):
            if (info.st_dev, info.st_ino) in seen:
                continue
            seen.add((info.st_dev, info.st_ino))
        total += info.st_size
        if stat.S_ISDIR(info.st_mode):
            try:
                with os.scandir(current) as entries:
                    stack.extend(entry.path for entry in entries)
            except OSError:
                pass
    return total


def run_finished(run_dir: Path) -> bool:
    """Lượt gsplat đã xong: summary.json (src/training/run.py ghi sau cùng) có status OK."""
    try:
        return json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8")).get("status") == "OK"
    except (OSError, ValueError, AttributeError):
        return False


FILTER_SOURCES = ("ckpts", "ply", "renders", "videos")  # phần của một lượt mà bản đã lọc (filtered/) lặp lại, nhỏ hơn


def filter_pending_bytes(run_dir: Path, thresholds: dict | None) -> int:
    """Chỗ bước lọc Gaussian (src/training/prune.py) còn cần cho một lượt đã xong: 0 khi không lọc hoặc `filtered/` đã
    lọc xong với đúng các ngưỡng này; nếu không thì tối đa bằng checkpoint, PLY, ảnh render và video của chính lượt đó
    (bản đã lọc giữ không quá chừng ấy Gaussian và chấm lại trên cùng ảnh test)."""
    if not thresholds:
        return 0
    try:
        meta = json.loads((Path(run_dir) / "filtered" / "filter_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = None
    if isinstance(meta, dict) and meta.get("status") == "OK" and meta.get("thresholds") == thresholds:
        return 0
    return sum(tree_bytes(Path(run_dir) / name) for name in FILTER_SOURCES)


def space_needs(plan: dict, config_dir: Path = CONFIG_DIR, ws: str = "/workspace",
                datasets: list[str] | None = None, without: list[str] | None = None) -> dict:
    """Byte còn cần ghi cho từng seed: tổng của các lượt gsplat chưa xong (benchmark × cảnh × chế độ / biến thể) của seed
    đó; `datasets`: chỉ tính các benchmark này (vd. bỏ benchmark không tải được), mặc định mọi benchmark của kế hoạch;
    `without`: các lượt không train, dạng `<benchmark>:<lượt>` (vd. lượt có loss độ sâu khi kiểm tra điểm SfM không đạt).

    Một lượt chưa xong được tính bằng dung lượng thật lớn nhất của lượt cùng benchmark, cùng cảnh, cùng chế độ / biến thể
    đã xong ở seed khác (kích thước gần như không đổi giữa các seed; gồm cả `filtered/` khi kế hoạch lọc Gaussian và
    seed đó đã lọc xong); chưa có lượt nào như vậy thì bằng `estimate.disk_gb` chia đều cho mọi lượt của kế hoạch (phần
    này gồm cả dữ liệu, B0 và ảnh PNG thu nhỏ, chỉ ghi một lần). Lượt đã xong không cần thêm gì, kể cả khi chạy lại sau
    khi pod dừng giữa chừng; riêng khi kế hoạch có `post_filter`, lượt đã xong mà chưa lọc với đúng ngưỡng còn cần chỗ
    cho bản đã lọc (filter_pending_bytes), và kích thước của lượt đo được cũng tính cả phần đó."""
    registry = load_registry(config_dir, ws)
    runs = []
    for bench in plan["benchmarks"]:
        root = Path(registry["datasets"][bench["dataset"]]["results"]["gsplat"])
        for scene in bench["scenes"]:
            for run in plan["runs"]:
                for seed in plan["seeds"]:
                    runs.append(((bench["dataset"], scene, run), seed, root / scene / f"{run}_s{seed}"))
    estimate = round(plan["estimate"]["disk_gb"] * 1e9 / len(runs))
    if datasets is not None:
        unknown = sorted(set(datasets) - {bench["dataset"] for bench in plan["benchmarks"]})
        if unknown:
            raise ValueError(f"benchmark(s) {unknown} are not in the plan {plan['name']}")
        runs = [run for run in runs if run[0][0] in datasets]
    skipped = set()
    for item in without or []:
        dataset, _, run = item.partition(":")
        if run not in plan["runs"] or dataset not in {bench["dataset"] for bench in plan["benchmarks"]}:
            raise ValueError(f"--without {item!r}: expected <benchmark>:<run> of the plan {plan['name']}")
        skipped.add((dataset, run))
    runs = [run for run in runs if (run[0][0], run[0][2]) not in skipped]
    post = plan.get("post_filter")
    pending = {(key, seed): filter_pending_bytes(path, post) for key, seed, path in runs if run_finished(path)}
    sizes = {(key, seed): tree_bytes(path) + pending[key, seed]
             for key, seed, path in runs if (key, seed) in pending}
    seeds = []
    for seed in plan["seeds"]:
        need = finished = measured = total = 0
        for key, run_seed, _ in runs:
            if run_seed != seed:
                continue
            total += 1
            if (key, seed) in sizes:
                finished += 1
                need += pending[key, seed]  # bước lọc của lượt đã xong (0 khi đã lọc hoặc kế hoạch không lọc)
                continue
            others = [size for (other_key, other_seed), size in sizes.items() if other_key == key]
            if others:
                need += max(others)
                measured += 1
            else:
                need += estimate
        seeds.append({"seed": seed, "need_bytes": need, "finished": finished, "runs": total, "measured": measured})
    return {"run_estimate_bytes": estimate, "seeds": seeds}


def plan_lines(plan: dict) -> list[str]:
    """Kế hoạch dạng "khóa|giá trị" (danh sách nối bằng dấu cách) cho scripts/pod/run_experiment.sh.

    `variants|` gồm các mục `<chế_độ>:<tag>:<khóa>=<giá trị>[,<khóa>=<giá trị>...]` (biến VARIANTS của
    scripts/pipeline/train_gsplat_modes.sh); `post_filter|<min_opacity> <max_scale>`, rỗng khi không lọc; mỗi cảnh báo
    của kế hoạch một dòng `warning|<nội dung>`."""
    def words(values) -> str:
        return " ".join(str(value) for value in values)
    post = plan["post_filter"]
    lines = [f"name|{plan['name']}", f"description|{plan['description'].replace('|', '/')}",
             f"modes|{words(plan['modes'])}", f"also_against|{words(plan['also_against'])}",
             f"variants|{words(plan['variant_lines'])}", f"runs|{words(plan['runs'])}",
             f"depth_runs|{words(plan['depth_runs'])}", f"depth_test_every|{plan['depth_test_every'] or ''}",
             f"mono_runs|{words(plan['mono_runs'])}",
             f"post_filter|{'' if post is None else f'{post[FILTER_KEYS[0]]!r} {post[FILTER_KEYS[1]]!r}'}",
             f"seeds|{words(plan['seeds'])}",
             f"gpu_hours|{plan['estimate']['gpu_hours']:g}", f"disk_gb|{plan['estimate']['disk_gb']:g}"]
    for bench in plan["benchmarks"]:
        lines.append(f"bench|{bench['dataset']}|{1 if bench['b0'] else 0}|{words(bench['scenes'])}")
    lines += [f"warning|{text.replace('|', '/')}" for text in plan.get("warnings") or []]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", type=Path, default=CONFIG_DIR, help="thư mục cấu hình (mặc định configs/ của repo)")
    parser.add_argument("--ws", default=os.environ.get("WS") or "/workspace", help="thư mục gốc trên pod (WS)")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", metavar="TÊN", help="in kế hoạch dạng khóa|giá trị")
    action.add_argument("--export", metavar="TÊN", help="ghi kế hoạch đã kiểm tra ra JSON (--out)")
    action.add_argument("--space", metavar="TÊN", help="byte còn cần cho từng seed: dòng seed|<seed>|<byte>|<xong>|<lượt>|<đo>")
    parser.add_argument("--datasets", nargs="*", help="với --space: chỉ tính các benchmark này")
    parser.add_argument("--without", nargs="*", metavar="BENCHMARK:LƯỢT",
                        help="với --space: các lượt không train (vd. bị bỏ vì kiểm tra điểm SfM không đạt)")
    parser.add_argument("--out", type=Path, help="file JSON cho --export")
    args = parser.parse_args(argv)
    ws = args.ws.rstrip("/") or "/"
    try:
        plan = load_experiment(args.plan or args.export or args.space, args.configs, ws)
        space = space_needs(plan, args.configs, ws, args.datasets, args.without) if args.space else None
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if space is not None:
        print(f"run_estimate|{space['run_estimate_bytes']}")
        for row in space["seeds"]:
            print(f"seed|{row['seed']}|{row['need_bytes']}|{row['finished']}|{row['runs']}|{row['measured']}")
        return 0
    if args.plan:
        print("\n".join(plan_lines(plan)))
        return 0
    if args.out is None:
        print("ERROR: --export needs --out <file.json>", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
