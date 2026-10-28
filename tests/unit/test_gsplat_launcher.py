"""Test gsplat_launcher với một bản giả lập tối giản của examples/ và gói gsplat (chạy trên CPU).

Bản giả lập giữ đúng những điểm launcher dựa vào: simple_trainer import `set_random_seed` từ `utils`,
`Dataset`/`Parser` từ `datasets.colmap` (namespace package), `cli` từ `gsplat.distributed`; Runner gọi
`set_random_seed(42 + local_rank)`; Dataset chọn ảnh theo `i % test_every`.
Kiểm tra với gsplat thật được ghi trong docs/design/training_modes.md (mục kiểm chứng).
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]  # gốc repo: package src

FAKE_UTILS = """
LAST = {}
def set_random_seed(seed):
    LAST["seed"] = seed
"""

FAKE_COLMAP = """
import numpy as np

class Parser:
    def __init__(self, data_dir, test_every=8):
        with open(data_dir + "/names.txt") as handle:
            self.image_names = sorted(line.strip() for line in handle if line.strip())
        self.test_every = test_every

class Dataset:
    def __init__(self, parser, split="train", patch_size=None, load_depths=False):
        self.parser = parser
        indices = np.arange(len(parser.image_names))
        if split == "train":
            self.indices = indices[indices % parser.test_every != 0]
        else:
            self.indices = indices[indices % parser.test_every == 0]
"""

FAKE_TRAINER = """
import json
import sys
from dataclasses import dataclass, field

import utils
from utils import set_random_seed
from datasets.colmap import Dataset, Parser
from gsplat.distributed import cli

@dataclass
class MCMCStrategy:
    cap_max: int = 1_000_000

@dataclass
class Config:
    data_dir: str = ""
    result_dir: str = ""
    test_every: int = 8
    strategy: MCMCStrategy = field(default_factory=MCMCStrategy)

def main(local_rank, world_rank, world_size, cfg):
    set_random_seed(42 + local_rank)
    parser = Parser(cfg.data_dir, cfg.test_every)
    train = Dataset(parser, split="train", patch_size=None, load_depths=False)
    val = Dataset(parser, split="val")
    out = {"seed": utils.LAST["seed"],
           "train": [parser.image_names[i] for i in train.indices],
           "val": [parser.image_names[i] for i in val.indices]}
    with open(cfg.result_dir + "/fake_out.json", "w") as handle:
        json.dump(out, handle)

if __name__ == "__main__":
    argv = sys.argv[1:]
    assert argv[0] in ("default", "mcmc")
    values = dict(zip(argv[1::2], argv[2::2]))
    cfg = Config(data_dir=values["--data-dir"], result_dir=values["--result-dir"],
                 test_every=int(values.get("--test-every", 8)))
    if "--strategy.cap-max" in values:
        cfg.strategy.cap_max = int(values["--strategy.cap-max"])
    cli(main, cfg, verbose=True)
"""

FAKE_DISTRIBUTED = """
def cli(fn, args, verbose=False):
    return fn(0, 0, 1, args)
"""

FAKE_TORCH = """
import os

class cuda:
    @staticmethod
    def device_count():
        return int(os.environ.get("FAKE_GPU_COUNT", "1"))
"""


@pytest.fixture()
def fake(tmp_path):
    examples = tmp_path / "gsplat" / "examples"
    (examples / "datasets").mkdir(parents=True)  # không có __init__.py, như gsplat @ 6e8c837
    (examples / "utils.py").write_text(textwrap.dedent(FAKE_UTILS), encoding="utf-8")
    (examples / "datasets" / "colmap.py").write_text(textwrap.dedent(FAKE_COLMAP), encoding="utf-8")
    (examples / "simple_trainer.py").write_text(textwrap.dedent(FAKE_TRAINER), encoding="utf-8")
    package = tmp_path / "site" / "gsplat"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "distributed.py").write_text(textwrap.dedent(FAKE_DISTRIBUTED), encoding="utf-8")
    (tmp_path / "site" / "torch").mkdir()
    (tmp_path / "site" / "torch" / "__init__.py").write_text(textwrap.dedent(FAKE_TORCH), encoding="utf-8")
    data = tmp_path / "scene"
    data.mkdir()
    (data / "names.txt").write_text("\n".join(f"{i:05d}.jpg" for i in range(1, 18)), encoding="utf-8")
    result = tmp_path / "result"
    result.mkdir()
    return {"examples": examples, "site": tmp_path / "site", "data": data, "result": result, "tmp": tmp_path}


def launch(fake, own_args, trainer_args=None, gpus=1):
    env = dict(os.environ, FAKE_GPU_COUNT=str(gpus))
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), str(fake["site"])])
    trainer_args = trainer_args or ["mcmc", "--data-dir", str(fake["data"]), "--result-dir", str(fake["result"]),
                                    "--test-every", "8", "--strategy.cap-max", "1234"]
    cmd = [sys.executable, "-m", "src.training.gsplat_launcher", "--examples-dir", str(fake["examples"]),
           *own_args, "--", *trainer_args]
    return subprocess.run(cmd, cwd=fake["tmp"], env=env, capture_output=True, text=True)


def test_seed_split_record_and_resolved_config(fake):
    record, resolved = fake["tmp"] / "applied.json", fake["tmp"] / "resolved.json"
    proc = launch(fake, ["--seed", "7", "--record-split", str(record), "--resolved-config", str(resolved)])
    assert proc.returncode == 0, proc.stderr
    out = json.loads((fake["result"] / "fake_out.json").read_text())
    assert out["seed"] == 7  # 42 + rank 0 -> 7 + rank 0
    assert out["val"] == ["00001.jpg", "00009.jpg", "00017.jpg"]
    applied = json.loads(record.read_text())
    assert applied["splits"]["val"] == out["val"] and applied["splits"]["train"] == out["train"]
    config = json.loads(resolved.read_text())
    assert config["strategy"] == {"cap_max": 1234} and config["strategy_type"] == "MCMCStrategy"
    assert config["_launcher"]["seed"] == 7
    assert config["_launcher"]["gsplat"]["installed_dir"] == str((fake["site"] / "gsplat").resolve())


def test_split_file_replaces_every_nth_rule(fake):
    split = {"rule": "test_list", "train": ["00002.jpg", "00003.jpg", "00004.jpg"],
             "test": ["00016.jpg", "00017.jpg"]}
    split_path = fake["tmp"] / "split.json"
    split_path.write_text(json.dumps(split))
    record = fake["tmp"] / "applied.json"
    proc = launch(fake, ["--split-file", str(split_path), "--record-split", str(record)])
    assert proc.returncode == 0, proc.stderr
    out = json.loads((fake["result"] / "fake_out.json").read_text())
    assert out["train"] == split["train"] and out["val"] == split["test"]
    assert out["seed"] == 42  # không truyền --seed thì giữ nguyên seed của gsplat


def test_unknown_names_in_split_file_fail(fake):
    split_path = fake["tmp"] / "split.json"
    split_path.write_text(json.dumps({"rule": "test_list", "train": ["00002.jpg"], "test": ["nope.jpg"]}))
    proc = launch(fake, ["--split-file", str(split_path)])
    assert proc.returncode != 0 and "not known to gsplat" in proc.stderr


def test_several_visible_gpus_are_refused(fake):
    proc = launch(fake, [], gpus=2)
    assert proc.returncode != 0 and "more than one GPU" in proc.stderr
    assert not (fake["result"] / "fake_out.json").exists()


def test_resolve_only_does_not_train(fake):
    resolved = fake["tmp"] / "resolved.json"
    proc = launch(fake, ["--resolve-only", "--resolved-config", str(resolved)])
    assert proc.returncode == 0, proc.stderr
    assert resolved.is_file() and not (fake["result"] / "fake_out.json").exists()


def test_requires_trainer_arguments(fake):
    env = dict(os.environ, PYTHONPATH=str(SRC))
    proc = subprocess.run([sys.executable, "-m", "src.training.gsplat_launcher", "--examples-dir",
                           str(fake["examples"])], env=env, capture_output=True, text=True)
    assert proc.returncode != 0 and "trainer arguments" in proc.stderr


def test_installed_gsplat_must_match_checkout(fake):
    checkout = fake["examples"].parent / "gsplat" / "strategy"
    installed = fake["site"] / "gsplat" / "strategy"
    checkout.mkdir(parents=True)
    installed.mkdir(parents=True)
    (checkout / "default.py").write_text("if step % reset_every == 0 and step > 0: pass\n", encoding="utf-8")
    (installed / "default.py").write_text("if step % reset_every == 0 & step > 0: pass\n", encoding="utf-8")
    resolved = fake["tmp"] / "resolved.json"
    proc = launch(fake, ["--resolve-only", "--resolved-config", str(resolved)])
    assert proc.returncode != 0 and "differs from the checkout" in proc.stderr
    (installed / "default.py").write_text((checkout / "default.py").read_text(encoding="utf-8"), encoding="utf-8")
    proc = launch(fake, ["--resolve-only", "--resolved-config", str(resolved)])
    assert proc.returncode == 0, proc.stderr
    assert json.loads(resolved.read_text())["_launcher"]["gsplat"]["matches_checkout"]



# Trainer giả có đúng 4 chỗ chèn của src/training/trainer_hooks.py (cùng thụt lề với Runner.train của gsplat @6e8c837).
# Không dùng textwrap.dedent: chỗ chèn phải giữ nguyên khoảng trắng. Các lời gọi vào hook được ghi lại để biết mọi dòng
# [SR3D] đều chạy.
FAKE_HOOK_TRAINER = """import json
import sys
from dataclasses import dataclass, field

from gsplat.distributed import cli


@dataclass
class MCMCStrategy:
    cap_max: int = 1_000_000
    refine_start_iter: int = 500
    refine_stop_iter: int = 25_000
    refine_every: int = 100

    def step_post_backward(self, step):
        return step


class DefaultStrategy:
    pass


@dataclass
class Config:
    data_dir: str = ""
    result_dir: str = ""
    max_steps: int = 3
    depth_loss: bool = False
    use_bilateral_grid: bool = False
    strategy: MCMCStrategy = field(default_factory=MCMCStrategy)


def rasterize(render_mode):
    return render_mode


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg

    def train(self):
        cfg = self.cfg
        hooks = globals().get("_SR3D_HOOKS")
        calls = []
        if hooks is not None:
            for name in ("wants_depth", "mono_loss", "kill_then_relocate"):
                def wrapper(*args, _original=getattr(hooks, name), _name=name, **kwargs):
                    calls.append(_name)
                    return _original(*args, **kwargs)
                setattr(hooks, name, wrapper)
        modes, losses = [], []
        for step in range(cfg.max_steps):
            data = {"image_id": [0]}
            alphas = None
            renders = rasterize(
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
            )
            modes.append(renders)
            if renders == "RGB+ED":
                colors, depths = renders, "depth map"
            else:
                colors, depths = renders, None
            loss = 1.0
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss
            losses.append(loss)
            if isinstance(self.cfg.strategy, DefaultStrategy):
                pass
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    step,
                )
        out = {"argv0": sys.argv[0], "hooks": getattr(hooks, "names", None), "calls": calls, "modes": modes,
               "losses": losses}
        with open(cfg.result_dir + "/fake_out.json", "w") as handle:
            json.dump(out, handle)


def main(local_rank, world_rank, world_size, cfg):
    Runner(cfg).train()


if __name__ == "__main__":
    argv = sys.argv[1:]
    values = dict(zip(argv[1::2], argv[2::2]))
    cli(main, Config(data_dir=values["--data-dir"], result_dir=values["--result-dir"]), verbose=True)
"""


@pytest.fixture()
def hook_fake(fake):
    (fake["examples"] / "simple_trainer.py").write_text(FAKE_HOOK_TRAINER, encoding="utf-8")
    return fake


def test_hook_runs_a_patched_copy_of_the_trainer(hook_fake):
    resolved = hook_fake["tmp"] / "resolved.json"
    original = (hook_fake["examples"] / "simple_trainer.py").read_text(encoding="utf-8")
    proc = launch(hook_fake, ["--hook", "kr", "--resolved-config", str(resolved)])
    assert proc.returncode == 0, proc.stderr
    patched = hook_fake["result"] / "sr3d_simple_trainer.py"
    out = json.loads((hook_fake["result"] / "fake_out.json").read_text())
    assert out["argv0"] == str(patched) and out["hooks"] == ["kr"]
    # mỗi vòng: dòng render_mode, dòng loss của depth prior, dòng KR trước bước relocate của MCMC
    assert out["calls"] == ["wants_depth", "mono_loss", "kill_then_relocate"] * 3
    assert out["modes"] == ["RGB"] * 3 and out["losses"] == [1.0] * 3  # hook tắt / ngoài vòng KR: không đổi gì
    assert (hook_fake["examples"] / "simple_trainer.py").read_text(encoding="utf-8") == original  # gốc không bị sửa
    assert patched.read_text(encoding="utf-8").count("[SR3D]") >= 4
    hooks = json.loads(resolved.read_text())["_launcher"]["hooks"]
    assert hooks["names"] == ["kr"] and hooks["patched_trainer"] == str(patched)
    assert hooks["trainer_is_pinned"] is False and hooks["parameters"]["kr"]["every"] == 500
    assert hooks["log"] == str(hook_fake["result"] / "sr3d_hooks.jsonl") and hooks["mono_dir"] is None


def test_without_hooks_the_original_trainer_runs(hook_fake):
    proc = launch(hook_fake, [])
    assert proc.returncode == 0, proc.stderr
    out = json.loads((hook_fake["result"] / "fake_out.json").read_text())
    assert out["argv0"] == str(hook_fake["examples"] / "simple_trainer.py") and out["hooks"] is None
    assert not (hook_fake["result"] / "sr3d_simple_trainer.py").exists()


def test_mono_hook_needs_a_complete_depth_prior(hook_fake):
    proc = launch(hook_fake, ["--hook", "mono"])
    assert proc.returncode != 0 and "--hook mono needs --mono-dir" in proc.stderr
    proc = launch(hook_fake, ["--mono-dir", str(hook_fake["tmp"])])
    assert proc.returncode != 0 and "only applies with --hook mono" in proc.stderr
    proc = launch(hook_fake, ["--hook", "mono", "--mono-dir", str(hook_fake["tmp"] / "none")])
    assert proc.returncode != 0 and "no complete depth prior" in proc.stderr
    assert not (hook_fake["result"] / "fake_out.json").exists()
    resolved = hook_fake["tmp"] / "resolved.json"  # chỉ phân giải tham số (smoke test): không cần depth prior
    proc = launch(hook_fake, ["--hook", "mono", "--hook", "kr", "--resolve-only", "--resolved-config", str(resolved)])
    assert proc.returncode == 0, proc.stderr
    assert json.loads(resolved.read_text())["_launcher"]["hooks"]["names"] == ["kr", "mono"]
    assert not (hook_fake["result"] / "fake_out.json").exists()


def test_hook_refuses_a_trainer_without_the_anchors(fake):
    proc = launch(fake, ["--hook", "kr"])  # trainer giả gốc: không có chỗ chèn nào
    assert proc.returncode != 0 and "anchor occurs 0 time" in proc.stderr
    assert not (fake["result"] / "fake_out.json").exists()
