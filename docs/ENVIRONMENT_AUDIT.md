# Môi trường thực nghiệm

_Cập nhật: 04/11/2026 (bản rút gọn, D21; lịch sử trong Git). Repo này đã cài và chạy trên pod ngày 31/10/2026 (mục 7). Các file `configs/environment/*.conf` vẫn để `provisional` (`scripts/pod/run_all.sh` đặt `ALLOW_PROVISIONAL=1`)._

## 1. Pod và bố cục `/workspace`

| Mục | Đã chạy ngày 31/10/2026 | Yêu cầu |
|---|---|---|
| Cloud | Secure Cloud | Network volume chỉ gắn được với pod Secure Cloud |
| GPU | 1 × RTX 4090, bản 48 GB, compute capability 8.9 | ≥ 24 GB VRAM; compute capability 8.0 / 8.6 / 8.9 / 9.0 (A100, A6000, A40, RTX 4090, L40S, H100); không dùng Blackwell |
| Driver | hỗ trợ CUDA 13.2 | hỗ trợ CUDA ≥ 12.8 (12.6 với `TORCH_CUDA_OVERRIDE=cu126`); COLMAP bản CUDA cần ≥ 12.9, thấp hơn thì tự dùng bản CPU |
| Image | CUDA 12.8.1, Ubuntu 22.04: nvcc 12.8.93, gcc 11 (tên template chính xác chưa ghi lại) | bản **devel** (có `nvcc`), gcc ≤ 12; `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` cũng đã chạy được (17/10/2026) |
| Volume | Network volume 170 GB, EU-CZ-1, `fuse` (MooseFS), `chmod` chạy được | Loại **Network volume**, không dùng Global volume; cho đổi quyền file (ô 1 của notebook kiểm tra) |

`/workspace` là network volume: còn nguyên khi Stop hoặc xóa pod. Mọi thứ ngoài `/workspace` (kể cả `/root`) mất khi Stop pod.

```
/workspace/
├── SpatialRoom3D/            # code (repo Git, clone qua HTTPS)
├── code/                     # gaussian-splatting (Inria), gsplat, depth_anything_v2: ghim theo commit
├── miniforge3/               # conda: env gs-inria, gs-tools, gs-gsplat
├── data/                     # chỉ dữ liệu gốc: custom_scenes/<cảnh>/, benchmarks/{deep_blending,mipnerf360}/
├── outputs/                  # mọi thứ sinh ra: custom_scenes/, benchmarks/, product/ (sản phẩm), reports/, logs/, jobs/
└── ket_qua_*.tar.gz          # gói kết quả nhẹ để tải về (scripts/pod/pack_results.sh)
```

Chi tiết: `docs/design/architecture.md` mục 2. Clone lại repo không làm mất gì: dữ liệu, env và kết quả đều nằm ngoài `SpatialRoom3D/`. Chỉ `configs/local.yaml` nằm trong repo; file này bạn tự tạo nếu cần, không có trong Git.

## 2. Lấy code từ GitHub

Repo `truong0809/SpatialRoom3D` là public (D19). Ô 2 và ô 3 của các notebook tương đương:

```bash
GIT_TERMINAL_PROMPT=0 git ls-remote https://github.com/truong0809/SpatialRoom3D.git refs/heads/main   # ô 2
git clone -b main https://github.com/truong0809/SpatialRoom3D.git /workspace/SpatialRoom3D           # ô 3, lần đầu
git -C /workspace/SpatialRoom3D pull --ff-only origin main                                            # ô 3, các lần sau
```

## 3. Môi trường cho 3DGS gốc (baseline B0)

Env `gs-inria`:
- Miniforge `26.7.2-0`, Python 3.10.22;
- PyTorch 2.7.1 + cu128 (`configs/environment/gs-inria.conf`); các gói khác trong `gs-inria.lock.txt`;
- extension build bằng `nvcc` của image, cho các kiến trúc 8.0, 8.6, 8.9, 9.0.

| Thành phần | Phiên bản |
|---|---|
| graphdeco-inria/gaussian-splatting | commit `54c035f` |
| diff-gaussian-rasterization | `9c5c202` (nhánh `dr_aa`) |
| simple-knn | `86710c2`; gitlab.inria.fr từ chối thì lấy từ mirror GitHub `camenduru/simple-knn` |
| fused-ssim | `1272e21` |

Không dùng nguyên `environment.yml` của Inria (Python 3.7, PyTorch 1.12, CUDA 11.6, không có `nvcc`). Khác môi trường gốc: phiên bản Python / PyTorch / CUDA, và `opencv-python-headless` thay `opencv-python`. Báo cáo phải nêu điều này khi so số liệu. Chọn biến thể CUDA của PyTorch không mới hơn CUDA mà driver hỗ trợ (dòng "CUDA Version" của `nvidia-smi`).

## 4. Cài đặt và kiểm tra

```bash
bash scripts/pod/run_all.sh --background     # env_probe -> setup_inria -> data_tools -> setup_gsplat -> pack; không train
bash scripts/pod/job.sh status               # theo dõi; log: bash scripts/pod/job.sh log run_all
```

Có thể chạy từng bước: `scripts/setup/env_probe.sh`, `setup_3dgs_inria.sh`, `setup_data_tools.sh`, `setup_gsplat.sh`. Mỗi script cài xong thì chạy smoke test.

Script dừng khi gặp một trong các trường hợp:
- GPU đang bận;
- phiên bản còn `provisional` mà không đặt `ALLOW_PROVISIONAL=1`;
- driver không hỗ trợ CUDA đã chọn;
- compute capability của GPU không có trong danh sách build;
- không có `nvcc` cùng major version;
- thiếu dung lượng;
- không truy cập được GitHub, PyPI hoặc conda;
- submodule sai commit.

Chạy lại an toàn: phần đã cài được kiểm tra rồi bỏ qua. Extension được build lại khi đổi commit, danh sách kiến trúc, PyTorch hoặc loại GPU.

### 4.1 Env `gs-tools`

COLMAP 4.2.1 (conda-forge) và ffmpeg, ghim trong `configs/environment/gs-tools.conf`. Bản CUDA của COLMAP chỉ có `cuda_129` và `cuda_134`, nên cần driver hỗ trợ CUDA ≥ 12.9. Driver thấp hơn thì `setup_data_tools.sh` tự cài bản CPU (ép bản CPU: `COLMAP_BUILD_OVERRIDE=cpu`). `src/reconstruction/colmap_runner.py` làm đúng các bước của `convert.py` của Inria, nhưng tự nhận tên tùy chọn GPU theo phiên bản COLMAP (D4).

### 4.2 Lệnh chạy chính

`README.md` mục "Chạy trên Runpod" và `docs/USAGE_GUIDE.md`. Luồng sản phẩm: `docs/design/product.md` mục 3.

### 4.3 Env `gs-gsplat`

Env này chạy `default`, `mcmc` và luồng sản phẩm.

| Thành phần | Phiên bản | Ghi chú |
|---|---|---|
| gsplat | commit `6e8c837` (v1.5.3 + 4 commit) | thư viện và `examples/` cùng commit, không sửa file |
| fused-ssim | `328dc98` | theo `examples/requirements.txt` của gsplat |
| pycolmap | fork rmbrualla `cc7ea4b` | Parser của gsplat cần `SceneManager` của fork này |
| Gói Python khác | `configs/environment/gs-gsplat.lock.txt` | 90 gói, ghim theo ngày 23/08/2025; có PyYAML, matplotlib 3.10.5, Pillow |
| PyTorch | như `gs-inria` | `setup_gsplat.sh` cảnh báo nếu hai file conf khác nhau |

Depth Anything V2 Small (chỉ dùng cho biến thể `mono`): code commit `a561b84`, trọng số `depth_anything_v2_vits.pth` (Apache-2.0, kiểm SHA-256 khi tải). Chạy trong env `gs-gsplat`; cài bằng `scripts/setup/setup_depth_anything.sh`.

### 4.4 Pod mới từ Jupyter

Tạo pod theo mục 1. Tải `app/notebooks/runpod.ipynb` lên `/workspace`, chạy ô 1 → 2 → 3 → 4, rồi chạy tiếp các ô theo dữ liệu (`docs/USAGE_GUIDE.md` mục 2). Pod vẫn tính tiền khi đang có việc chạy nền. Xong thì Stop hoặc Terminate pod; mọi thứ trong `/workspace` vẫn còn.

### 4.5 Chạy nền và khóa GPU (`scripts/pod/job.sh`)

- `job.sh start <tên> -- <lệnh>` chạy lệnh trong phiên riêng (`setsid nohup`): đóng trình duyệt hay tắt kernel Jupyter không làm dừng việc.
- Mỗi lúc chỉ một việc dùng GPU, nhờ khóa `flock` trên `/run/lock/spatialroom3d-gpu.lock`. Khóa mất theo khi tắt pod nên không bao giờ bị treo. Việc nền mới thì xếp hàng; lệnh chạy trực tiếp gặp khóa đang bị giữ thì dừng.
- Mỗi việc có một token riêng (`SR3D_JOB_TOKEN`). `status` và `stop` tìm tiến trình theo token; `stop` gửi SIGTERM, sau 20 giây thì SIGKILL.
- Trạng thái nằm ở `outputs/jobs/<tên>.state`, log ở `outputs/logs/jobs/<tên>.log`. Pod tắt giữa chừng thì chạy lại lệnh là đủ.
- `AUTO_STOP=1`: dừng pod khi việc cuối cùng xong. Chưa thử trên pod.

## 5. Mốc đối chiếu cho B0

Số Inria công bố cho Deep Blending (trung bình 2 cảnh, code hiện hành): PSNR 29,690 / SSIM 0,906 / LPIPS 0,238. Ngưỡng chấp nhận: lệch PSNR ≤ 0,5 dB, SSIM ≤ 0,01, LPIPS ≤ 0,01. Chạy bằng đúng `train.py --eval`, `render.py` và `metrics.py` của Inria.

## 6. Sự cố đã biết

- **gsplat v1.5.3:** `DefaultStrategy` không bao giờ reset opacity (lỗi thứ tự toán tử, đã sửa ở `6e8c837`). v1.3.0 thì reset cả ở bước 0. Vì vậy ghim `6e8c837`.
- **pycolmap:** cài gói `pycolmap` chính thức làm Parser của gsplat lỗi; phải dùng fork rmbrualla.
- **Gói `datasets`:** `examples/datasets` của gsplat là namespace package. Một gói `datasets` khác cài trong env (vd. của Hugging Face) sẽ che mất nó. `gsplat_launcher.py` và `src/product/train_main.py` kiểm tra trường hợp này và dừng.
- **`simple_trainer.py`:**
  - thấy nhiều GPU thì tự train phân tán: code của repo giới hạn một GPU;
  - gán cứng seed 42: launcher của thực nghiệm cho đổi seed.
- **fused-ssim `328dc98`:** chỉ build cho loại GPU lúc cài. Đổi loại GPU thì `setup_gsplat.sh` build lại. Các script train chạy thử fused-ssim trên GPU trước khi train.
- **Jupyter:** đặt `MPLBACKEND=module://matplotlib_inline...`, làm matplotlib báo lỗi trong env `gs-gsplat`. Mọi script đặt `MPLBACKEND=Agg`.
- **COLMAP:**
  - bản CUDA cần driver hỗ trợ CUDA ≥ 12.9 (mục 4.1);
  - bản không có CUDA mà bật GPU thì cần OpenGL: `colmap_runner.py` chỉ bật GPU khi có CUDA, và chạy với `QT_QPA_PLATFORM=offscreen`.
- **`df` trên network volume:** báo dung lượng của cả cụm lưu trữ (hàng PB), không phải quota. Xem dung lượng trên trang Runpod.
- **Volume không cho `chmod`** (31/10/2026, volume mới tạo ở một data center khác EU-RO-1): `chmod` báo `Operation not permitted`, nên `git clone`, conda và pip đều hỏng. Trang tạo volume lúc đó đang chọn Global volume (beta); chưa xác nhận đó đúng là loại của volume lỗi. Network volume thì `chmod` bình thường (EU-RO-1 ngày 17/10, EU-CZ-1 ngày 31/10/2026). Ô 1 của notebook kiểm tra `chmod`.
- **PyTorch:** `pip install torch==2.7.1` coi bản khác biến thể CUDA là đủ, nên script ghi rõ `torch==2.7.1+cu128`.
- **Phiên bản ghim:** `opencv-python-headless` 4.11.0.86 (bản 4.12 cần `numpy>=2`, xung đột với gsplat); `viser` 0.2.23.
- **3DGS gốc:**
  - GCC ≥ 13 có thể lỗi biên dịch (`rasterizer_impl.h` thiếu `#include <cstdint>`);
  - `--eval` luôn lấy mỗi ảnh thứ 8 làm ảnh test (`llffhold` gán cứng);
  - `--train_test_exp` đưa nửa ảnh test vào train: không dùng khi so số liệu.
- **PyYAML:** `configs/*.yaml` được đọc bằng `python3` của máy nếu có PyYAML, không thì bằng python của env `gs-gsplat`. Env `gs-inria` không có PyYAML.
- **`run_panels.sh` với `VIDEO=1`:** có thể bị `Killed` vì hết RAM (31/10/2026, ở `phong_nhom` và `mipnerf360 room`).

## 7. Lượt chạy trên pod

**31/10/2026, repo này (commit `c0a7a92`).** Pod ở mục 1; code clone qua HTTPS bằng tay.

| Việc | Thời gian | Kết quả |
|---|---|---|
| `run_all` (cài 3 env, smoke test) | 37 phút | xong |
| `fetch_mipnerf360` (4 cảnh trong nhà) | 7 phút 54 giây | `360_v2.zip` 12,54 GB; SHA-256 ghi trong `fetch_*.json` |
| `gsplat_mipnerf360` (`default`, `mcmc` × 4 cảnh × 3 seed) | 6 giờ 13 phút | xong |
| `scene_phong_nhom` (video phòng nhóm: khung ảnh, COLMAP, `default`, `mcmc`) | 48 phút | xong |
| `main` (B0, Deep Blending, bảng tổng; dùng lại 24 lượt Mip-NeRF 360) | 4 giờ 9 phút | mọi bước OK |
| `tune_mcmc` (16 biến thể trên `room`, lọc Gaussian) | 5 giờ 50 phút | mọi bước OK |

Số liệu: `docs/PROJECT_STATUS.md` mục "Kết quả".

**17/10/2026, repo trước.** Cùng script cài đặt và cùng lệnh train; RTX 4090 24 GB, EU-RO-1. Đây là số đo tham khảo, `configs/*.yaml` và `experiments/*/README.md` trích các số này.

| Việc | Kết quả |
|---|---|
| B0 trên Deep Blending, 30.000 vòng | Đạt ngưỡng. drjohnson 29,369 / 0,9051 / 0,2361 (23,4 phút); playroom 30,077 / 0,9081 / 0,2408 (17,2 phút); trung bình 29,723 / 0,9066 / 0,2384 (+0,033 dB so với số công bố); cả bước B0 49,5 phút |
| gsplat trên Deep Blending, seed 42 | `default` / `mcmc`: drjohnson 17,5 / 19,4 phút, playroom 13,0 / 13,6 phút; cả lượt 74 phút; `default` PASS so với số công bố (−0,192 dB) và so với B0 |
| Cài đặt | `gs-inria` 29 phút; `gs-gsplat` 32 phút (lần đầu); `gs-tools` 18 phút |
