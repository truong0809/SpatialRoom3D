"""Smoke test cho env gs-gsplat trên GPU (gọi ở bước cuối của setup_gsplat.sh).

Kiểm tra:
1. gsplat có extension CUDA biên dịch sẵn (gsplat.csrc), không phải biên dịch JIT lúc chạy.
2. rasterization chạy xuôi/ngược; DefaultStrategy và MCMCStrategy chạy vài bước (kể cả nhân relocation).
3. fused_ssim: SSIM(x, x) ~ 1 và có gradient.
4. LPIPS (VGG, như Inria) tính được; lần đầu sẽ tải trọng số.
5. examples/simple_trainer.py import được và phân giải đúng tham số của mọi chế độ trong configs/base.yaml (strategy,
   protocol, cap_max, tham số riêng của chế độ), và của các biến thể tham số (`--variants`, mục variants của một kế
   hoạch thực nghiệm), không train. Biến thể có code (`sr3d.kr`, `sr3d.mono`): bản sao có vá của simple_trainer.py
   (src/training/trainer_hooks.py) phải vá được và phân giải được như file gốc.
6. Khi có biến thể có code (`sr3d.kr`, `sr3d.mono`): trên GPU, với tensor nhỏ, KR đánh dấu đúng Gaussian quá lớn và bước
   relocate thật của MCMCStrategy dời chúng; loss của depth prior có gradient hữu hạn; kiểm NaN dừng lượt
   (src/training/trainer_hooks.py). Khi có `sr3d.mono`: Depth Anything V2 Small (--depth-anything-dir) nạp được với trọng
   số đã ghim và ước lượng được một ảnh trên GPU.

    python scripts/setup/smoke_test_gsplat.py --examples-dir /workspace/code/gsplat/examples
    python scripts/setup/smoke_test_gsplat.py --examples-dir ... --modes default mcmc   # chỉ mục 5, vài giây
    python scripts/setup/smoke_test_gsplat.py --examples-dir ... --modes default mcmc --variants mcmc:rbg:random_bkgd=true
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

# Jupyter đặt MPLBACKEND=module://matplotlib_inline... cho tiến trình con; env này không có gói đó,
# nên `import matplotlib` (torchmetrics, examples/utils.py) báo lỗi. Đặt trước mọi import.
os.environ["MPLBACKEND"] = "Agg"

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[2]  # gốc repo (package src)
RESULTS: list[tuple[str, bool]] = []


def check(name):
    def wrap(fn):
        def run(*args, **kwargs):
            print(f"\n--- {name}")
            try:
                fn(*args, **kwargs)
            except Exception:  # noqa: BLE001 - báo lỗi rồi chạy tiếp các mục khác
                traceback.print_exc()
                RESULTS.append((name, False))
                print(f"FAIL {name}")
            else:
                RESULTS.append((name, True))
                print(f"PASS {name}")
        return run
    return wrap


@check("gsplat compiled extension")
def check_extension():
    import importlib

    import gsplat

    importlib.import_module("gsplat.csrc")  # chỉ có khi extension được build sẵn bằng setup.py
    print("gsplat", gsplat.__version__, "|", gsplat.__file__)


def make_params(n: int, opacity: float, device: str):
    torch.manual_seed(0)
    quats = F.normalize(torch.randn(n, 4, device=device), dim=-1)
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.randn(n, 3, device=device) * 0.5 + torch.tensor([0.0, 0.0, 3.0], device=device)),
        "scales": torch.nn.Parameter(torch.full((n, 3), math.log(0.03), device=device)),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(torch.full((n,), math.log(opacity / (1 - opacity)), device=device)),
        "sh0": torch.nn.Parameter(torch.rand(n, 1, 3, device=device) * 0.5),
        "shN": torch.nn.Parameter(torch.zeros(n, 15, 3, device=device)),
    })


@check("rasterization + DefaultStrategy + MCMCStrategy")
def check_strategies():
    from gsplat import DefaultStrategy, MCMCStrategy, rasterization

    device = "cuda"
    width, height, n = 320, 240, 3000
    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.tensor([[[300.0, 0.0, width / 2], [0.0, 300.0, height / 2], [0.0, 0.0, 1.0]]], device=device)
    target = torch.rand(1, height, width, 3, device=device)
    strategies = {
        # ngưỡng 0 để chắc chắn có nhân bản / tách Gaussian trong vài bước
        "default": DefaultStrategy(refine_start_iter=0, refine_every=2, grow_grad2d=0.0, verbose=True),
        "mcmc": MCMCStrategy(cap_max=int(n * 1.2), refine_start_iter=0, refine_every=2, verbose=True),
    }
    for name, strategy in strategies.items():
        params = make_params(n, 0.1 if name == "default" else 0.5, device)
        optimizers = {key: torch.optim.Adam([value], lr=1e-3) for key, value in params.items()}
        strategy.check_sanity(params, optimizers)
        state = strategy.initialize_state(scene_scale=1.0) if name == "default" else strategy.initialize_state()
        start = len(params["means"])
        for step in range(7):
            colors = torch.cat([params["sh0"], params["shN"]], dim=1)
            render, _, info = rasterization(
                params["means"], params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
                colors, viewmats, Ks, width, height, sh_degree=3, packed=False)
            if name == "default":
                strategy.step_pre_backward(params, optimizers, state, step, info)
            loss = (render - target).abs().mean()
            loss.backward()
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if name == "default":
                strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
            else:
                strategy.step_post_backward(params, optimizers, state, step, info, lr=1e-3)
        end = len(params["means"])
        print(f"{name}: {start} -> {end} Gaussians, last loss {loss.item():.4f}")
        assert torch.isfinite(loss), "loss is not finite"
        assert end > start, f"{name} strategy did not add Gaussians"
        if name == "mcmc":
            assert end <= strategy.cap_max, "MCMC exceeded cap_max"


@check("fused_ssim")
def check_fused_ssim():
    from fused_ssim import fused_ssim

    x = torch.rand(1, 3, 64, 64, device="cuda", requires_grad=True)
    value = fused_ssim(x, x.detach())
    assert abs(value.item() - 1.0) < 1e-3, f"SSIM(x, x) = {value.item()}"
    other = torch.rand(1, 3, 64, 64, device="cuda")
    (1 - fused_ssim(x, other)).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@check("LPIPS (VGG) from torchmetrics")
def check_lpips():
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to("cuda")
    a = torch.rand(1, 3, 64, 64, device="cuda")
    assert lpips(a, a).item() < 1e-6
    print("LPIPS(a, b) =", lpips(a, torch.rand_like(a)).item())


def resolved_value(cfg: dict, flag: str):
    """Giá trị trong cấu hình đã phân giải của một tham số dòng lệnh, vd. 'strategy.refine-stop-iter'."""
    value = cfg
    for part in flag.split("."):
        value = value[part.replace("-", "_")]
    return value


def specs_of(modes: list[str] | None, variants: list[str] | None) -> list:
    """(tên lượt, ModeSpec) của các chế độ (mặc định mọi chế độ trong configs/base.yaml) và biến thể."""
    sys.path.insert(0, str(REPO_DIR))
    from src.training.profiles import load_config, parse_assignments, resolve_mode

    config = load_config(REPO_DIR / "configs" / "base.yaml")
    specs = [(mode, resolve_mode(config, mode)) for mode in modes or list(config["modes"])]
    for entry in variants or []:  # <chế_độ>:<tag>:<khóa>=<giá_trị>[,...], như VARIANTS của train_gsplat_modes.sh
        mode, tag, sets = entry.split(":", 2)
        specs.append((f"{mode}_{tag}", resolve_mode(config, mode, parse_assignments(sets.split(",")))))
    return specs


@check("simple_trainer.py arguments of the training modes")
def check_trainer_cli(examples_dir: Path, modes: list[str] | None = None, variants: list[str] | None = None):
    sys.path.insert(0, str(REPO_DIR))
    from src.training.profiles import HOOK_KEYS, hook_names, resolve_cap_max, trainer_args

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_DIR), env.get("PYTHONPATH")]))
    specs = specs_of(modes, variants)
    with tempfile.TemporaryDirectory() as tmp:
        for mode, spec in specs:
            cap = None
            if spec.preset == "mcmc":
                # 1_234_567 khác giá trị mặc định của MCMCStrategy.cap_max (1_000_000): kiểm tra được tham số có tác dụng
                policy = (spec.cap_max_policy or {}).get("policy")
                cap = resolve_cap_max(spec)[0] if policy == "fixed" else 1_234_567
            out = Path(tmp) / f"{mode}.json"
            hooks = hook_names(spec)
            cmd = [sys.executable, "-m", "src.training.gsplat_launcher", "--examples-dir", str(examples_dir),
                   "--resolve-only", "--resolved-config", str(out), *[f"--hook={name}" for name in hooks], "--",
                   *trainer_args(spec, Path(tmp) / "scene", Path(tmp) / mode, cap)]
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(proc.stdout[-2000:] + proc.stderr[-4000:])
            cfg = json.loads(out.read_text(encoding="utf-8"))
            expected = "DefaultStrategy" if spec.preset == "default" else "MCMCStrategy"
            assert cfg["strategy_type"] == expected, cfg["strategy_type"]
            assert cfg["max_steps"] == spec.max_steps and cfg["lpips_net"] == spec.protocol["lpips_net"]
            # đánh giá tách riêng sau huấn luyện; checkpoint và PLY chỉ ở bước cuối
            assert cfg["eval_steps"] == [-1], cfg["eval_steps"]
            assert cfg["save_steps"] == [spec.max_steps] and cfg["ply_steps"] == [spec.max_steps]
            assert cfg["data_factor"] == spec.protocol["data_factor"] and cfg["init_type"] == "sfm"
            if cap is not None:
                assert cfg["strategy"]["cap_max"] == cap
            for flag, value in spec.args.items():  # tham số riêng của chế độ có tác dụng
                if flag in HOOK_KEYS:  # biến thể có code: không phải tham số của simple_trainer.py
                    continue
                assert resolved_value(cfg, flag) == value, (mode, flag, resolved_value(cfg, flag), value)
            hooks_info = cfg["_launcher"].get("hooks") or {}
            assert hooks_info.get("names", []) == hooks, (mode, hooks_info.get("names"), hooks)
            if hooks:  # bản sao có vá phân giải được như file gốc; ghi rõ file gốc có đúng bản ghim không
                print(f"{mode}: patched trainer for {hooks} (simple_trainer.py "
                      f"{'is' if hooks_info['trainer_is_pinned'] else 'is NOT'} the pinned file)")
            extra = ", ".join(f"{flag}={value}" for flag, value in spec.args.items())
            print(f"{mode}: {expected}, max_steps {cfg['max_steps']}, lpips {cfg['lpips_net']}"
                  + (f", {extra}" if extra else ""))


def hooks_used(modes: list[str] | None, variants: list[str] | None) -> set[str]:
    """Các hook (sr3d.kr, sr3d.mono) mà chế độ / biến thể bật (tham số sai đã được báo ở mục 5)."""
    try:
        specs = specs_of(modes, variants)  # thêm gốc repo vào sys.path
        from src.training.profiles import hook_names

        return {name for _, spec in specs for name in hook_names(spec)}
    except Exception:  # noqa: BLE001
        return set()


def hook_smoke(device: str, mcmc_step: bool = True) -> dict:
    """Biến thể có code chạy được trên `device`, với tensor nhỏ: KR đánh dấu đúng các Gaussian quá lớn; bước relocate
    thật của MCMCStrategy dời chúng ngay trong vòng đó (`mcmc_step`, cần CUDA); loss của depth prior có gradient hữu
    hạn; kiểm NaN dừng lượt."""
    import tempfile as _tempfile
    from types import SimpleNamespace

    import numpy as np

    sys.path.insert(0, str(REPO_DIR))
    from src.training import mono_depth
    from src.training.trainer_hooks import Hooks

    n, huge = 400, 4
    params = make_params(n, 0.5, device)
    with torch.no_grad():
        params["scales"][:huge] = math.log(0.6)  # trục lớn nhất 0,6 > 0,15 × scene_scale
    optimizers = {key: torch.optim.Adam([value], lr=1e-3) for key, value in params.items()}
    from gsplat import MCMCStrategy

    strategy = MCMCStrategy(cap_max=n)
    camtoworlds = np.tile(np.eye(4), (2, 1, 1))
    camtoworlds[1, :3, 3] = (0.0, 0.0, -5.0)
    runner = SimpleNamespace(cfg=SimpleNamespace(strategy=strategy, max_steps=30_000, patch_size=None), splats=params,
                             scene_scale=1.0, parser=SimpleNamespace(camtoworlds=camtoworlds, image_names=["a.png"]),
                             trainset=SimpleNamespace(indices=np.array([0])))
    with _tempfile.TemporaryDirectory() as tmp:
        height, width = 48, 64
        depth = 2.0 + torch.linspace(0.0, 1.0, width, device=device).expand(height, width)
        np.save(Path(tmp) / "a.npy", (1.0 / depth).cpu().numpy().astype(np.float16))
        manifest = {"version": mono_depth.VERSION, "complete": True,
                    "images": {"a.png": {"file": "a.npy", "use": True, "corr": 0.9, "points": 50, "flipped": False}}}
        (Path(tmp) / mono_depth.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
        hooks = Hooks(["kr", "mono"], mono_dir=Path(tmp))
        killed = hooks.kill_then_relocate(runner, 3000)
        assert killed == huge, f"KR marked {killed} Gaussians, expected {huge}"
        dead = torch.sigmoid(params["opacities"]) <= strategy.min_opacity
        assert bool(dead[:huge].all()) and not bool(dead[huge:].any())
        if mcmc_step:  # relocate, thêm Gaussian và nhiễu vị trí của gsplat (kernel CUDA)
            strategy.step_post_backward(params, optimizers, strategy.initialize_state(), 3000, {}, lr=1e-3)
            assert len(params["means"]) == n, len(params["means"])
            assert bool((torch.sigmoid(params["opacities"]) > strategy.min_opacity).all()), "dead Gaussians remain"
        rendered = depth[None, ..., None].clone().requires_grad_(True)
        loss = hooks.mono_loss(runner, 100, {"image_id": torch.tensor([0])}, rendered, torch.ones_like(rendered))
        loss.backward()
        assert torch.isfinite(loss) and rendered.grad is not None and bool(torch.isfinite(rendered.grad).all())
        assert float(loss) < 0.01, f"depth prior loss {float(loss)} for a prior equal to 1 / depth"
        with torch.no_grad():
            params["means"][0, 0] = float("nan")
        try:
            hooks.guard(runner, 500)
        except FloatingPointError:
            pass
        else:
            raise AssertionError("the NaN check did not stop the run")
    return {"killed": killed, "mono_loss": float(loss)}


@check("code variants (Kill-then-Relocate + MCMC relocate, depth prior loss, NaN check) on the GPU")
def check_code_variants():
    result = hook_smoke("cuda")
    print(f"KR marked {result['killed']} Gaussians, MCMC relocated them; depth prior loss {result['mono_loss']:.5f} "
          "with finite gradients; NaN check stops the run")


@check("Depth Anything V2 Small (depth prior of the mono variant)")
def check_depth_anything(code_dir: Path):
    sys.path.insert(0, str(REPO_DIR))
    import numpy as np

    from src.training.mono_depth import load_depth_anything

    estimate = load_depth_anything(code_dir)  # đúng commit, đúng SHA-256 của trọng số
    y, x = np.mgrid[0:240, 0:320]
    image = np.stack([(x * 255 // 319), (y * 255 // 239), np.full_like(x, 128)], axis=-1).astype(np.uint8)
    depth = estimate(image)
    assert depth.shape == (240, 320) and np.isfinite(depth).all(), (depth.shape, np.isfinite(depth).all())
    assert float(depth.max()) > float(depth.min()), "constant depth map"
    print(f"Depth Anything V2 Small on {'GPU' if torch.cuda.is_available() else 'CPU'}: depth map {depth.shape}, "
          f"range {float(depth.min()):.3f} .. {float(depth.max()):.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples-dir", required=True, type=Path)
    parser.add_argument("--modes", nargs="+", help="chỉ kiểm tra tham số của các chế độ này (mục 5, không train)")
    parser.add_argument("--variants", nargs="*", default=[], metavar="CHẾ_ĐỘ:TAG:KHÓA=GIÁ_TRỊ[,...]",
                        help="kiểm tra thêm tham số của các biến thể này (mục 5)")
    parser.add_argument("--depth-anything-dir", type=Path, default=None,
                        help="code Depth Anything V2 cho mục 6 (mặc định: <examples-dir>/../../depth_anything_v2)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("FAIL: PyTorch cannot see a GPU")
        return 1
    print("torch", torch.__version__, "| CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0))
    if not args.modes and not args.variants:
        check_extension()
        check_strategies()
        check_fused_ssim()
        check_lpips()
    examples = args.examples_dir.resolve()
    check_trainer_cli(examples, args.modes, args.variants)
    hooks = hooks_used(args.modes, args.variants)
    if hooks:
        check_code_variants()
    if "mono" in hooks:
        check_depth_anything(args.depth_anything_dir or examples.parent.parent / "depth_anything_v2")
    print("\n===== SUMMARY =====")
    for name, ok in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    passed = all(ok for _, ok in RESULTS)
    print("SMOKE TEST", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
