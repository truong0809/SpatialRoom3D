"""Test src/training/trainer_hooks.py: vá simple_trainer.py (chỗ chèn phải có đúng một lần), KR và loss của depth prior.

Phần vá dùng một đoạn mã giả chứa đúng 4 chỗ chèn; với simple_trainer.py thật của gsplat @6e8c837, đặt biến môi trường
SR3D_GSPLAT_TRAINER=<đường dẫn tới examples/simple_trainer.py> để kiểm tra thêm: đúng SHA-256 của bản ghim, bản vá biên
dịch được và các dòng chèn nằm đúng chỗ trong vòng lặp của Runner.train (cây cú pháp). KR và loss Pearson cần PyTorch
(CPU là đủ); không có thì bỏ qua.
"""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.training import mono_depth
from src.training.trainer_hooks import (
    ANCHORS,
    HOOKS_LOG,
    KR,
    MONO,
    PINNED_TRAINER_SHA256,
    Hooks,
    PatchError,
    hook_parameters,
    patch_trainer_source,
    write_patched_trainer,
)

SNIPPET = "".join(old for _name, old, _new in ANCHORS)  # 4 chỗ chèn, mỗi chỗ một lần


def fake_source(body: str = SNIPPET) -> str:
    return "import json\n\nclass Runner:\n    def train(self):\n        for step in range(3):\n" + body + "\n"


def test_patch_inserts_the_four_hook_lines_once():
    patched = patch_trainer_source(fake_source())
    assert patched.startswith("# [SR3D]")
    for _name, old, new in ANCHORS:
        assert patched.count(new) == 1 and new != old
    assert patched.count("_SR3D_HOOKS.") == 3  # wants_depth, mono_loss, kill_then_relocate
    assert "            _sr3d_depths = depths  # [SR3D]" in patched


def test_patch_refuses_missing_duplicated_or_already_patched_anchors():
    for name, old, _new in ANCHORS:
        with pytest.raises(PatchError, match=f"'{name}' anchor occurs 0 time"):
            patch_trainer_source(fake_source(SNIPPET.replace(old, "")))
        with pytest.raises(PatchError, match=f"'{name}' anchor occurs 2 time"):
            patch_trainer_source(fake_source(SNIPPET + old))
    with pytest.raises(PatchError, match="already contains"):
        patch_trainer_source(patch_trainer_source(fake_source()))


def test_write_patched_trainer_records_hashes(tmp_path):
    script = tmp_path / "simple_trainer.py"
    script.write_text(fake_source(), encoding="utf-8")
    info = write_patched_trainer(script, tmp_path / "run" / "sr3d_simple_trainer.py")
    assert Path(info["patched_trainer"]).read_text(encoding="utf-8") == patch_trainer_source(fake_source())
    assert info["trainer_is_pinned"] is False and len(info["trainer_sha256"]) == 64
    assert script.read_text(encoding="utf-8") == fake_source()  # file gốc không đổi


def _loop_body(tree: ast.Module) -> list[ast.stmt]:
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Runner")
    train = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    loop = next(node for node in ast.walk(train) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == "step")
    return loop.body


def _index(body: list[ast.stmt], predicate) -> int:
    matches = [i for i, stmt in enumerate(body) if predicate(stmt)]
    assert len(matches) == 1, [ast.unparse(body[i])[:80] for i in matches]
    return matches[0]


@pytest.mark.skipif(not os.environ.get("SR3D_GSPLAT_TRAINER"), reason="set SR3D_GSPLAT_TRAINER to gsplat's trainer")
def test_real_trainer_patch_lands_in_the_training_loop():
    script = Path(os.environ["SR3D_GSPLAT_TRAINER"])
    original = script.read_text(encoding="utf-8")
    import hashlib

    assert hashlib.sha256(original.encode("utf-8")).hexdigest() == PINNED_TRAINER_SHA256
    patched = patch_trainer_source(original)
    compile(patched, "sr3d_simple_trainer.py", "exec")
    body = _loop_body(ast.parse(patched))
    text = [ast.unparse(stmt) for stmt in body]
    render = _index(body, lambda s: "rasterize_splats" in ast.unparse(s) and "render_mode" in ast.unparse(s))
    assert "cfg.depth_loss or _SR3D_HOOKS.wants_depth()" in text[render]
    split = _index(body, lambda s: isinstance(s, ast.If) and "renders.shape[-1] == 4" in ast.unparse(s.test))
    saved = _index(body, lambda s: ast.unparse(s) == "_sr3d_depths = depths")
    assert saved == split + 1  # ngay sau khi tách màu / độ sâu, trước khi nhánh depth_loss lấy mẫu lại `depths`
    depth_branch = _index(body, lambda s: isinstance(s, ast.If) and ast.unparse(s.test) == "cfg.depth_loss"
                          and "loss += depthloss * cfg.depth_lambda" in ast.unparse(s))
    mono = _index(body, lambda s: "_SR3D_HOOKS.mono_loss(self, step, data, _sr3d_depths, alphas)" in ast.unparse(s))
    tv = _index(body, lambda s: isinstance(s, ast.If) and ast.unparse(s.test) == "cfg.use_bilateral_grid"
                and "loss += tvloss" in ast.unparse(s))
    backward = _index(body, lambda s: ast.unparse(s) == "loss.backward()")
    assert depth_branch < mono < tv < backward
    strategy = _index(body, lambda s: isinstance(s, ast.If) and "DefaultStrategy" in ast.unparse(s.test))
    mcmc = body[strategy].orelse[0]
    assert isinstance(mcmc, ast.If) and "MCMCStrategy" in ast.unparse(mcmc.test)
    assert ast.unparse(mcmc.body[0]) == "_SR3D_HOOKS.kill_then_relocate(self, step)"
    assert "step_post_backward" in ast.unparse(mcmc.body[1])
    assert strategy > backward


def test_hook_parameters_and_names():
    assert hook_parameters([]) is None
    assert hook_parameters(["mono", "kr"])["kr"] == KR
    mono = hook_parameters(["mono"])["mono"]
    assert mono["lambda"] == MONO["lambda"] and mono["prior"]["model"] == mono_depth.MODEL
    assert Hooks(["mono", "kr"]).names == ["kr", "mono"]
    with pytest.raises(ValueError, match="unknown hook"):
        Hooks(["relocate"])
    off = Hooks([])
    assert off.wants_depth() is False and off.kill_then_relocate(None, 3000) == 0 and off.mono_loss(None, 0, {}, None,
                                                                                                       None) == 0.0


# ---------------------------------------------------------------------------------------------------- KR (PyTorch)

def _kr_runner(torch, means, scales, opacities, cameras, train):
    strategy = SimpleNamespace(refine_start_iter=500, refine_stop_iter=25_000, refine_every=100, min_opacity=0.005)
    camtoworlds = np.tile(np.eye(4), (len(cameras), 1, 1))
    camtoworlds[:, :3, 3] = cameras
    splats = {"means": torch.nn.Parameter(torch.tensor(means, dtype=torch.float32)),
              "scales": torch.nn.Parameter(torch.log(torch.tensor(scales, dtype=torch.float32))),
              "opacities": torch.nn.Parameter(torch.logit(torch.tensor(opacities, dtype=torch.float32)))}
    return SimpleNamespace(cfg=SimpleNamespace(strategy=strategy, max_steps=30_000), splats=splats, scene_scale=1.0,
                           parser=SimpleNamespace(camtoworlds=camtoworlds), trainset=SimpleNamespace(indices=np.array(train)))


def test_kill_then_relocate_marks_bad_gaussians_dead_only_on_kr_steps(tmp_path):
    torch = pytest.importorskip("torch")
    cameras = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]  # camera 0: train, camera 1: test
    means = [[0.0, 0.0, 2.0],     # 0 bình thường
             [0.0, 0.0, 2.0],     # 1 khổng lồ: trục lớn nhất 0,2 > 0,15
             [0.0, 0.0, 2.0],     # 2 hình kim: 0,02 > 0,01 và gấp 20 lần trục giữa
             [0.0, 0.0, 0.01],    # 3 sát camera train (0,01 < 0,03)
             [5.0, 0.0, 0.01],    # 4 sát camera TEST: không được xét
             [0.0, 0.0, 2.0],     # 5 khổng lồ nhưng đã "chết" (opacity < min_opacity): MCMC tự dời, KR không đếm
             [0.0, 0.0, 2.0]]     # 6 dẹt nhưng nhỏ: trục lớn nhất 0,005 < 0,01, không phải kim
    scales = [[0.01, 0.01, 0.01], [0.1, 0.1, 0.2], [0.001, 0.001, 0.02], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01],
              [0.1, 0.1, 0.2], [0.0001, 0.0001, 0.005]]
    opacities = [0.5, 0.5, 0.5, 0.5, 0.5, 0.004, 0.5]
    filler = 193  # Gaussian bình thường: tổng 200, giới hạn mỗi lần 2 % = 4 > 3 Gaussian xấu
    runner = _kr_runner(torch, means + [[0.0, 0.0, 2.0]] * filler, scales + [[0.01, 0.01, 0.01]] * filler,
                        opacities + [0.5] * filler, cameras, train=[0])
    hooks = Hooks(["kr"], log_path=tmp_path / HOOKS_LOG)
    before = runner.splats["opacities"].detach().clone()
    for step in (2_500, 3_050, 3_100, 21_500, 25_000):  # trước vòng 3.000, không chia hết cho 500, sau 21.000, hết refine
        assert hooks.kill_then_relocate(runner, step) == 0
    assert torch.equal(runner.splats["opacities"].detach(), before)
    assert hooks.kill_then_relocate(runner, 3_000) == 3
    after = torch.sigmoid(runner.splats["opacities"].detach())
    dead = (after <= 0.005).nonzero().flatten().tolist()
    assert dead == [1, 2, 3, 5]  # 5 đã chết từ trước
    alive = [0, 4, 6, *range(7, 200)]
    assert torch.allclose(after[alive], before.sigmoid()[alive])
    record = json.loads((tmp_path / HOOKS_LOG).read_text(encoding="utf-8").splitlines()[-1])
    assert record == {"hook": "kr", "step": 3000, "gaussians": 200, "candidates": 3, "killed": 3, "huge": 1,
                      "needle": 1, "near_camera": 1}


def test_kill_then_relocate_caps_each_pass_at_two_percent_largest_first():
    torch = pytest.importorskip("torch")
    count = 200
    means = [[0.0, 0.0, 2.0]] * count
    largest = [0.2 + 0.001 * i if i < 10 else 0.01 for i in range(count)]  # 10 khổng lồ, giới hạn 2 % = 4
    scales = [[0.05, 0.05, value] if value > 0.1 else [0.01, 0.01, value] for value in largest]
    runner = _kr_runner(torch, means, scales, [0.5] * count, [[100.0, 0.0, 0.0]], train=[0])
    assert Hooks(["kr"]).kill_then_relocate(runner, 3_000) == int(KR["max_frac"] * count) == 4
    dead = (torch.sigmoid(runner.splats["opacities"].detach()) <= 0.005).nonzero().flatten().tolist()
    assert dead == [6, 7, 8, 9]  # 4 Gaussian có trục lớn nhất lớn nhất


def test_kr_steps_are_refine_steps_of_the_real_mcmc_strategy(monkeypatch):
    """Với MCMCStrategy thật của gsplat (cần gsplat import được, vd. PYTHONPATH tới checkout @6e8c837): mọi vòng KR là
    vòng mà step_post_backward gọi relocate, và Gaussian KR đánh dấu nằm trong mặt nạ "chết" mà relocate nhận."""
    torch = pytest.importorskip("torch")
    mcmc = pytest.importorskip("gsplat.strategy.mcmc")
    current, relocated = {"step": None}, {}  # vòng -> mặt nạ Gaussian "chết" mà MCMC dời
    monkeypatch.setattr(mcmc, "relocate", lambda **kwargs: relocated.__setitem__(current["step"], kwargs["mask"].clone()))
    monkeypatch.setattr(mcmc, "sample_add", lambda **kwargs: None)
    monkeypatch.setattr(mcmc, "inject_noise_to_position", lambda **kwargs: None)
    strategy = mcmc.MCMCStrategy(cap_max=10)  # mặc định @6e8c837: refine 500 → 25.000, mỗi 100 vòng, min_opacity 0,005
    runner = _kr_runner(torch, [[0.0, 0.0, 2.0]] * 100, [[0.1, 0.1, 0.2]] + [[0.01, 0.01, 0.01]] * 99, [0.5] * 100,
                        [[100.0, 0.0, 0.0]], train=[0])  # Gaussian 0 khổng lồ, còn lại bình thường
    runner.cfg.strategy = strategy
    hooks, state, kr_steps = Hooks(["kr"]), strategy.initialize_state(), []
    for step in range(0, 30_000, 50):
        current["step"] = step
        if hooks.kill_then_relocate(runner, step):
            kr_steps.append(step)
        strategy.step_post_backward(runner.splats, {}, state, step, {}, lr=1e-3)
        runner.splats["opacities"].data[:] = torch.logit(torch.tensor(0.5))  # relocate giả không đổi gì: hồi lại
    assert kr_steps == list(range(3_000, 21_001, 500))
    assert sorted(relocated) == kr_steps  # chỉ vòng KR mới có Gaussian chết, và MCMC dời chúng ngay trong vòng đó
    assert all(bool(mask[0]) and not bool(mask[1:].any()) for mask in relocated.values())


# --------------------------------------------------------------------------------------------- depth prior (PyTorch)

def _mono_dir(tmp_path, priors: dict) -> Path:
    folder = tmp_path / "mono"
    folder.mkdir()
    images = {}
    for name, prior in priors.items():
        if prior is None:
            images[name] = {"file": name + ".npy", "use": False, "corr": 0.01, "points": 50, "flipped": False}
            np.save(folder / (name + ".npy"), np.ones((2, 2), dtype=np.float16))
        else:
            images[name] = {"file": name + ".npy", "use": True, "corr": 0.9, "points": 50, "flipped": False}
            np.save(folder / (name + ".npy"), prior.astype(np.float16))
    (folder / mono_depth.MANIFEST).write_text(json.dumps({"version": mono_depth.VERSION, "complete": True,
                                                         "images": images}), encoding="utf-8")
    return folder


def _mono_runner(names, max_steps=30_000, patch_size=None):
    return SimpleNamespace(cfg=SimpleNamespace(max_steps=max_steps, patch_size=patch_size),
                           trainset=SimpleNamespace(indices=np.arange(len(names))),
                           parser=SimpleNamespace(image_names=list(names)))


def test_mono_loss_is_pearson_on_inverse_depth(tmp_path):
    torch = pytest.importorskip("torch")
    height, width = 32, 48
    yy, xx = np.mgrid[0:height, 0:width]
    depth = 1.0 + 0.05 * xx + 0.02 * yy  # độ sâu render
    good = 1.0 / depth  # prior cùng hình dạng với nghịch đảo độ sâu
    folder = _mono_dir(tmp_path, {"a": good, "b": -good, "c": None, "d": good[::2, ::2]})
    runner = _mono_runner(["a", "b", "c", "d"])
    hooks = Hooks(["mono"], mono_dir=folder, log_path=tmp_path / HOOKS_LOG)
    assert hooks.wants_depth() is True
    depths = torch.tensor(depth, dtype=torch.float32)[None, ..., None].requires_grad_(True)
    alphas = torch.ones(1, height, width, 1)

    def loss_for(image_id, step=100, d=depths, a=alphas):
        return hooks.mono_loss(runner, step, {"image_id": torch.tensor([image_id])}, d, a)

    perfect = loss_for(0)
    assert float(perfect) == pytest.approx(0.0, abs=2e-3)  # tương quan ~1 (float16, chặn phân vị 2-98 %)
    opposite = loss_for(1)
    assert float(opposite) == pytest.approx(2 * MONO["lambda"], rel=1e-2)
    opposite.backward()
    assert depths.grad is not None and torch.isfinite(depths.grad).all() and depths.grad.abs().sum() > 0
    assert loss_for(2) == 0.0  # ảnh không dùng prior (tương quan với SfM thấp)
    assert float(loss_for(3)) == pytest.approx(0.0, abs=5e-3)  # prior nhỏ hơn ảnh train: nội suy về đúng kích thước
    stop = int(MONO["stop_frac"] * 30_000)
    assert float(loss_for(1, step=stop)) > 0 and loss_for(1, step=stop + 1) == 0.0  # tắt sau 60 % số vòng
    few = torch.zeros(1, height, width, 1)
    few[0, :10, :10] = 1.0  # 100 pixel có alpha > 0,5 < 256
    assert loss_for(0, a=few) == 0.0
    log = [json.loads(line) for line in (tmp_path / HOOKS_LOG).read_text(encoding="utf-8").splitlines()]
    assert log and log[-1]["hook"] == "mono" and log[-1]["step"] == stop


def test_mono_loss_errors_stop_the_run(tmp_path):
    torch = pytest.importorskip("torch")
    folder = _mono_dir(tmp_path, {"a": np.ones((4, 4))})
    depths, alphas = torch.ones(1, 4, 4, 1), torch.ones(1, 4, 4, 1)
    data = {"image_id": torch.tensor([0])}
    with pytest.raises(RuntimeError, match="rendered depth"):
        Hooks(["mono"], mono_dir=folder).mono_loss(_mono_runner(["a"]), 0, data, None, alphas)
    with pytest.raises(RuntimeError, match="patch_size"):
        Hooks(["mono"], mono_dir=folder).mono_loss(_mono_runner(["a"], patch_size=64), 0, data, depths, alphas)
    with pytest.raises(RuntimeError, match="no entry for the train image"):
        Hooks(["mono"], mono_dir=folder).mono_loss(_mono_runner(["zz"]), 0, data, depths, alphas)
    with pytest.raises(RuntimeError, match="no complete depth prior"):
        Hooks(["mono"], mono_dir=tmp_path / "empty").load_priors()
    with pytest.raises(RuntimeError, match="needs the folder"):
        Hooks(["mono"]).load_priors()


def test_the_gpu_smoke_check_of_the_code_variants_runs_on_cpu(monkeypatch):
    """scripts/setup/smoke_test_gsplat.py mục 6 (hook_smoke), trên CPU và không có bước relocate thật (kernel CUDA): trên
    pod, env_check chạy đúng hàm này trên GPU, kèm relocate của MCMCStrategy."""
    pytest.importorskip("torch")
    pytest.importorskip("gsplat.strategy.mcmc")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts" / "setup"))
    import smoke_test_gsplat

    result = smoke_test_gsplat.hook_smoke("cpu", mcmc_step=False)
    assert result["killed"] == 4 and result["mono_loss"] < 0.01


def test_the_nan_check_stops_a_run_with_hooks(tmp_path):
    torch = pytest.importorskip("torch")
    runner = _kr_runner(torch, [[0.0, 0.0, 2.0]] * 3, [[0.01, 0.01, 0.01]] * 3, [0.5] * 3, [[100.0, 0.0, 0.0]], train=[0])
    hooks = Hooks(["kr"], log_path=tmp_path / HOOKS_LOG)
    hooks.guard(runner, 500)  # hữu hạn: không làm gì
    with torch.no_grad():
        runner.splats["opacities"][1] = float("nan")
    hooks.guard(runner, 499)  # chỉ kiểm mỗi 500 vòng
    with pytest.raises(FloatingPointError, match="non-finite Gaussian parameters"):
        hooks.guard(runner, 1000)
    record = json.loads((tmp_path / HOOKS_LOG).read_text(encoding="utf-8").splitlines()[-1])
    assert record == {"hook": "guard", "step": 1000, "not_finite": ["opacities"]}
    with pytest.raises(FloatingPointError):  # dòng chèn của loss gọi guard mỗi vòng, cả khi chỉ bật kr
        hooks.mono_loss(runner, 1500, {}, None, None)
