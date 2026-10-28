"""Biến thể có code cho examples/simple_trainer.py của gsplat @6e8c837: Kill-then-Relocate (`kr`) và depth prior (`mono`).

Không sửa file nào của gsplat. Với `--hook kr` / `--hook mono`, src/training/gsplat_launcher.py đọc simple_trainer.py, chèn
4 dòng tại 4 chỗ cố định (ANCHORS; mỗi chỗ phải xuất hiện đúng một lần, không thì dừng trước khi train), ghi bản sao có vá
vào thư mục kết quả của lượt (sr3d_simple_trainer.py, giữ làm bằng chứng) rồi chạy bản sao đó, với một đối tượng `Hooks`
đặt sẵn trong biến toàn cục `_SR3D_HOOKS`. Bốn dòng chèn chỉ gọi vào `Hooks`; hook nào không bật thì hàm của nó không làm
gì. Không bật hook nào thì launcher chạy file gốc như mọi lượt khác. Bật bằng tham số `sr3d.kr` / `sr3d.mono` của chế độ
hoặc biến thể (src/training/profiles.py, HOOK_KEYS).

- kr (`kill_then_relocate`, ngay trước MCMCStrategy.step_post_backward, chỉ preset mcmc): ở các vòng refine của MCMC
  chia hết cho 500, từ 10 % đến 70 % số vòng (3.000 → 21.000 khi train 30.000 vòng), Gaussian còn sống mà quá lớn (trục
  lớn nhất > 0,15 × scene_scale), hình kim (trục lớn nhất > 0,01 × scene_scale và gấp hơn 15 lần trục giữa) hoặc nằm
  cách một camera TRAIN dưới 0,03 × scene_scale bị hạ opacity xuống dưới strategy.min_opacity; bước relocate của MCMC
  chạy ngay sau đó trong cùng vòng dời chúng tới vị trí các Gaussian khác (chọn theo opacity, như Gaussian "chết" của
  MCMC). Mỗi lần tối đa 2 % số Gaussian, ưu tiên trục lớn nhất. Số Gaussian không đổi.
  Nguồn: notebook demo v4 / v2 của nhóm (không có paper). Khác notebook: chỉ xét camera train (notebook xét cả camera
  test, tức dùng thông tin của tập test khi train). Khác bước relocate gốc của MCMC, vốn chỉ dời Gaussian gần như trong
  suốt nên ảnh render gần như không đổi (3DGS-MCMC, Kheradmand et al., 2024), KR dời cả Gaussian còn nhìn thấy: đây là
  heuristic nằm ngoài công thức MCMC. Như Gaussian "chết" của MCMC (và như notebook), Gaussian bị dời nhận tham số mới
  nhưng giữ moment Adam cũ: relocate của gsplat chỉ đặt lại moment của Gaussian nguồn.
- mono (`mono_loss`, cộng vào loss ngay trước TV loss của bilateral grid): loss Pearson giữa nghịch đảo độ sâu render
  (render_mode "RGB+ED") và depth prior của Depth Anything V2 Small tính trước (src/training/mono_depth.py), trên pixel
  có alpha > 0,5 (cần ít nhất 256 pixel), nghịch đảo độ sâu bị chặn trong khoảng phân vị 2–98 % của ảnh; trọng số 0,05,
  bật đến 60 % số vòng. Loss Pearson cho độ sâu đơn ảnh theo FSGS (Zhu et al., ECCV 2024); hằng số theo notebook v2.
- Kiểm NaN (mọi lượt có hook, `guard`, gọi mỗi vòng qua dòng chèn của loss): mỗi 500 vòng, vị trí / scale / quaternion /
  opacity của mọi Gaussian phải hữu hạn; không thì dừng lượt với lỗi (FloatingPointError). Không tự tắt thành phần nào
  rồi train tiếp như notebook.

Nhật ký của hook: <thư mục kết quả>/sr3d_hooks.jsonl (kr: số Gaussian bị dời mỗi lần; mono: loss trung bình mỗi 500
vòng; guard: tham số không hữu hạn). torch chỉ được import khi hook thật sự tính.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.training import mono_depth

PINNED_TRAINER_SHA256 = "79319e1cd7404e4d1ba0c425634235c39e6054f0643b904a01feea6179462c05"  # simple_trainer.py @6e8c837
HOOK_NAMES = ("kr", "mono")
PATCHED_TRAINER = "sr3d_simple_trainer.py"  # trong thư mục kết quả của lượt
HOOKS_LOG = "sr3d_hooks.jsonl"
MARK = "[SR3D]"
KR = {"start_frac": 0.1, "stop_frac": 0.7, "every": 500, "huge": 0.15, "needle_ratio": 15.0, "needle_min": 0.01,
      "near": 0.03, "max_frac": 0.02, "dead_factor": 0.5}
MONO = {"lambda": 0.05, "stop_frac": 0.6, "min_alpha": 0.5, "min_pixels": 256, "min_depth": 1e-3,
        "clamp": [0.02, 0.98], "quantile_samples": 100_000, "log_every": 500}
CAMERA_CHUNK = 262_144  # số Gaussian mỗi lần tính khoảng cách tới camera
GUARD_EVERY = 500  # kiểm tham số Gaussian hữu hạn mỗi chừng này vòng
GUARDED = ("means", "scales", "quats", "opacities")

HEADER = ("# [SR3D] Bản sao có vá của examples/simple_trainer.py (gsplat), do src/training/trainer_hooks.py của SpatialRoom3D\n"
          "# [SR3D] sinh ra cho lượt này; chỉ chạy được qua src/training/gsplat_launcher.py (--hook). Các dòng chèn có [SR3D].\n")
# (tên, đoạn gốc, đoạn thay); đoạn gốc lấy nguyên văn từ simple_trainer.py @6e8c837 (Runner.train)
ANCHORS = (
    ("render_mode",
     '                render_mode="RGB+ED" if cfg.depth_loss else "RGB",\n',
     '                render_mode="RGB+ED" if (cfg.depth_loss or _SR3D_HOOKS.wants_depth()) else "RGB",  # [SR3D]\n'),
    ("depth_map",
     "                colors, depths = renders, None\n",
     "                colors, depths = renders, None\n"
     "            _sr3d_depths = depths  # [SR3D] bản đồ độ sâu render, trước khi nhánh depth_loss lấy mẫu lại `depths`\n"),
    ("mono_loss",
     "            if cfg.use_bilateral_grid:\n"
     "                tvloss = 10 * total_variation_loss(self.bil_grids.grids)\n",
     "            loss = loss + _SR3D_HOOKS.mono_loss(self, step, data, _sr3d_depths, alphas)  # [SR3D]\n"
     "            if cfg.use_bilateral_grid:\n"
     "                tvloss = 10 * total_variation_loss(self.bil_grids.grids)\n"),
    ("kill_then_relocate",
     "            elif isinstance(self.cfg.strategy, MCMCStrategy):\n"
     "                self.cfg.strategy.step_post_backward(\n",
     "            elif isinstance(self.cfg.strategy, MCMCStrategy):\n"
     "                _SR3D_HOOKS.kill_then_relocate(self, step)  # [SR3D]\n"
     "                self.cfg.strategy.step_post_backward(\n"),
)


class PatchError(RuntimeError):
    pass


def patch_trainer_source(text: str) -> str:
    """Nội dung simple_trainer.py -> bản có vá. Một chỗ chèn không có, hoặc có nhiều hơn một lần: PatchError."""
    if MARK in text:
        raise PatchError("simple_trainer.py already contains [SR3D] lines; patch the unmodified file of gsplat")
    for name, old, _new in ANCHORS:
        count = text.count(old)
        if count != 1:
            raise PatchError(f"simple_trainer.py: the '{name}' anchor occurs {count} time(s), expected exactly once "
                             f"(the code variants are written for gsplat @6e8c837)")
    for _name, old, new in ANCHORS:
        text = text.replace(old, new, 1)
    return HEADER + text


def write_patched_trainer(script: Path, target: Path) -> dict:
    """Ghi bản có vá của `script` vào `target`; trả về thông tin để ghi vào cấu hình đã phân giải (bằng chứng)."""
    original = Path(script).read_bytes()
    patched = patch_trainer_source(original.decode("utf-8")).encode("utf-8")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(patched)
    digest = hashlib.sha256(original).hexdigest()
    return {"trainer": str(script), "trainer_sha256": digest, "trainer_is_pinned": digest == PINNED_TRAINER_SHA256,
            "patched_trainer": str(target), "patched_sha256": hashlib.sha256(patched).hexdigest()}


def hook_parameters(names) -> dict | None:
    """Tham số của các hook đang bật (ghi vào run_meta.json; đổi hằng số thì lượt cũ không còn là cùng cấu hình)."""
    names = [name for name in HOOK_NAMES if name in set(names)]
    if not names:
        return None
    result = {}
    if "kr" in names:
        result["kr"] = dict(KR)
    if "mono" in names:
        result["mono"] = {**MONO, "prior": {"model": mono_depth.MODEL, "params": mono_depth.PARAMS}}
    return result


class Hooks:
    """Đối tượng mà bản sao có vá của simple_trainer.py gọi vào (biến toàn cục `_SR3D_HOOKS`)."""

    def __init__(self, names=(), mono_dir: Path | None = None, log_path: Path | None = None):
        unknown = sorted(set(names) - set(HOOK_NAMES))
        if unknown:
            raise ValueError(f"unknown hook(s) {unknown} (known: {list(HOOK_NAMES)})")
        self.names = [name for name in HOOK_NAMES if name in set(names)]
        self.kr = "kr" in self.names
        self.mono = "mono" in self.names
        self.mono_dir = Path(mono_dir) if mono_dir is not None else None
        self.log_path = Path(log_path) if log_path is not None else None
        self._manifest = None
        self._priors = {}  # tên ảnh -> tensor float16 trên CPU, hoặc None khi ảnh không dùng prior
        self._window = self._empty_window()

    # ------------------------------------------------------------------------------------------------ chung
    def wants_depth(self) -> bool:
        """Có cần render độ sâu (RGB+ED) dù depth_loss của gsplat đang tắt không."""
        return self.mono

    def _log(self, record: dict) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def guard(self, runner, step: int) -> None:
        """Mỗi GUARD_EVERY vòng: tham số Gaussian không hữu hạn thì dừng lượt (FloatingPointError), ghi vào nhật ký."""
        if step % GUARD_EVERY != 0:
            return
        splats = getattr(runner, "splats", None)
        if splats is None:
            return
        import torch

        bad = [key for key in GUARDED if key in splats and not bool(torch.isfinite(splats[key]).all())]
        if bad:
            self._log({"hook": "guard", "step": int(step), "not_finite": bad})
            raise FloatingPointError(f"[SR3D] non-finite Gaussian parameters {bad} at step {step}: the run stops "
                                     "(no component is switched off)")

    # ------------------------------------------------------------------------------------------------ kr
    def kill_then_relocate(self, runner, step: int) -> int:
        """Hạ opacity của các Gaussian xấu xuống dưới min_opacity ngay trước bước relocate của MCMC ở vòng này.

        Trả về số Gaussian bị dời (0 khi hook tắt hoặc vòng này không phải vòng KR)."""
        if not self.kr:
            return 0
        strategy = runner.cfg.strategy
        max_steps = int(runner.cfg.max_steps)
        # cùng điều kiện với MCMCStrategy.step_post_backward @6e8c837, để relocate chạy ngay sau trong đúng vòng này
        refine = (strategy.refine_start_iter < step < strategy.refine_stop_iter) and step % strategy.refine_every == 0
        start, stop = int(KR["start_frac"] * max_steps), int(KR["stop_frac"] * max_steps)
        if not refine or step < start or step > stop or step % KR["every"] != 0:
            return 0
        import torch

        with torch.no_grad():
            splats = runner.splats
            means = splats["means"]
            scales = torch.sort(torch.exp(splats["scales"]), dim=1).values
            middle, largest = scales[:, 1], scales[:, 2]
            scene_scale = float(runner.scene_scale)
            huge = largest > KR["huge"] * scene_scale
            needle = (largest > KR["needle_min"] * scene_scale) & (largest / middle.clamp_min(1e-12) > KR["needle_ratio"])
            train = np.asarray(runner.trainset.indices, dtype=np.int64)  # chỉ camera train
            cameras = torch.as_tensor(np.asarray(runner.parser.camtoworlds)[train, :3, 3], dtype=means.dtype,
                                      device=means.device)
            near = torch.zeros_like(huge)
            limit = KR["near"] * scene_scale
            for begin in range(0, means.shape[0], CAMERA_CHUNK):
                chunk = means[begin:begin + CAMERA_CHUNK]
                near[begin:begin + CAMERA_CHUNK] = torch.cdist(chunk, cameras).min(dim=1).values < limit
            alive = torch.sigmoid(splats["opacities"]) > strategy.min_opacity
            bad = (huge | needle | near) & alive
            total, candidates = int(means.shape[0]), int(bad.sum())
            cap = int(KR["max_frac"] * total)
            if candidates > cap:  # ưu tiên trục lớn nhất
                score = torch.where(bad, largest, torch.full_like(largest, -1.0))
                keep = torch.topk(score, cap).indices
                bad = torch.zeros_like(bad)
                bad[keep] = True
            killed = int(bad.sum())
            if killed:
                dead = torch.logit(torch.tensor(strategy.min_opacity * KR["dead_factor"], dtype=splats["opacities"].dtype,
                                                device=means.device))
                splats["opacities"].data[bad] = dead
        self._log({"hook": "kr", "step": int(step), "gaussians": total, "candidates": candidates, "killed": killed,
                   "huge": int((huge & alive).sum()), "needle": int((needle & alive).sum()),
                   "near_camera": int((near & alive).sum())})
        return killed

    # ------------------------------------------------------------------------------------------------ mono
    @staticmethod
    def _empty_window() -> dict:
        return {"terms": 0, "loss_sum": 0.0, "no_prior": 0, "few_pixels": 0, "not_finite": 0}

    def load_priors(self) -> dict:
        """Đọc manifest.json của depth prior (src/training/mono_depth.py); lỗi thì dừng trước khi train."""
        if self._manifest is None:
            if self.mono_dir is None:
                raise RuntimeError("the mono hook needs the folder of the depth prior (--mono-dir)")
            manifest = mono_depth.read_json(self.mono_dir / mono_depth.MANIFEST)
            if not isinstance(manifest, dict) or not manifest.get("complete") \
                    or manifest.get("version") != mono_depth.VERSION:
                raise RuntimeError(f"{self.mono_dir} holds no complete depth prior (run src.training.mono_depth first)")
            self._manifest = manifest
        return self._manifest

    def _prior(self, name: str):
        if name not in self._priors:
            entry = self.load_priors()["images"].get(name)
            if entry is None:
                raise RuntimeError(f"the depth prior in {self.mono_dir} has no entry for the train image {name!r}")
            if not entry["use"]:
                self._priors[name] = None
            else:
                import torch

                values = np.load(self.mono_dir / entry["file"]).astype(np.float16)
                self._priors[name] = torch.from_numpy(values)
        return self._priors[name]

    def mono_loss(self, runner, step: int, data: dict, depths, alphas):
        """Loss Pearson của depth prior cho batch hiện tại; 0 khi hook tắt, sau 60 % số vòng, hoặc khi không có ảnh
        nào của batch dùng được. Dòng chèn này chạy mỗi vòng của mọi lượt có hook, nên kiểm NaN (`guard`) cũng ở đây."""
        self.guard(runner, step)
        if not self.mono:
            return 0.0
        stop = int(MONO["stop_frac"] * int(runner.cfg.max_steps))
        if step > stop:
            return 0.0
        if depths is None:
            raise RuntimeError("the mono hook needs the rendered depth (render_mode RGB+ED)")
        if getattr(runner.cfg, "patch_size", None) is not None:
            raise RuntimeError("the mono hook does not support patch_size (random crops would not line up with the prior)")
        import torch
        import torch.nn.functional as F

        window = self._window
        terms = []
        height, width = depths.shape[1], depths.shape[2]
        for b in range(depths.shape[0]):
            index = int(runner.trainset.indices[int(data["image_id"][b])])
            prior = self._prior(runner.parser.image_names[index])
            if prior is None:
                window["no_prior"] += 1
                continue
            prior = prior.to(device=depths.device, dtype=torch.float32)
            if tuple(prior.shape) != (height, width):
                prior = F.interpolate(prior[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
            depth = depths[b, ..., 0]
            valid = ((depth > MONO["min_depth"]) & (alphas[b, ..., 0] > MONO["min_alpha"]) & torch.isfinite(depth)
                     & torch.isfinite(prior))
            if int(valid.sum()) < MONO["min_pixels"]:
                window["few_pixels"] += 1
                continue
            x = 1.0 / depth[valid].clamp_min(MONO["min_depth"])
            y = prior[valid]
            # chặn pixel cực đoan (sát ống kính) để gradient của 1/d không bùng nổ
            sample = x.detach()[:: max(1, x.numel() // MONO["quantile_samples"])].float()
            low, high = torch.quantile(sample, torch.tensor(MONO["clamp"], device=sample.device, dtype=sample.dtype))
            x = x.clamp(float(low), float(high))
            x = (x - x.mean()) / (x.std() + 1e-6)
            y = (y - y.mean()) / (y.std() + 1e-6)
            corr = (x * y).mean()
            if bool(torch.isfinite(corr)):
                terms.append(1.0 - corr)
            else:
                window["not_finite"] += 1
        loss = MONO["lambda"] * torch.stack(terms).mean() if terms else 0.0
        if terms:
            window["terms"] += len(terms)
            window["loss_sum"] += float(loss.detach()) * len(terms)
        if step % MONO["log_every"] == 0 or step == stop:
            self._log({"hook": "mono", "step": int(step), **window,
                       "mean_loss": window["loss_sum"] / window["terms"] if window["terms"] else None})
            self._window = self._empty_window()
        return loss
