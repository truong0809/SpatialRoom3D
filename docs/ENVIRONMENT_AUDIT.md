# Môi trường thực nghiệm

_Cập nhật: 13/11/2026 (bản rút gọn, D21; lịch sử trong Git). Repo này đã cài và chạy trên pod ngày 31/10/2026, chạy lại trên pod mới cùng volume ngày 08/11/2026 và 12–13/11/2026 (mục 7). Các file `configs/environment/*.conf` vẫn để `provisional` (`scripts/pod/run_all.sh` đặt `ALLOW_PROVISIONAL=1`)._

## 1. Pod và bố cục `/workspace`

| Mục | Đã chạy ngày 31/10/2026 (và 08/11/2026 khi khác) | Yêu cầu |
|---|---|---|
| Cloud | Secure Cloud | Network volume chỉ gắn được với pod Secure Cloud |
| GPU | 1 × RTX 4090, bản 48 GB VRAM (bản thường gặp là 24 GB), compute capability 8.9. Ngày 08/11/2026: RTX 4090 24 GB (`nvidia-smi`: 24564 MiB) | ≥ 24 GB VRAM; compute capability 8.0 / 8.6 / 8.9 / 9.0 (A100, A6000, A40, RTX 4090, L40S, H100); không dùng Blackwell |
| Driver | hỗ trợ CUDA 13.2 | hỗ trợ CUDA ≥ 12.8 (12.6 với `TORCH_CUDA_OVERRIDE=cu126`); COLMAP bản CUDA cần ≥ 12.9, thấp hơn thì tự dùng bản CPU |
| Image | CUDA 12.8.1, Ubuntu 22.04: nvcc 12.8.93, gcc 11 (tên template chính xác chưa ghi lại). Ngày 08/11/2026: template PyTorch 2.8.0, CUDA 12.9.0, Ubuntu 22.04 (nvcc 12.9.86) | bản **devel** (có `nvcc`), gcc ≤ 12; `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` cũng đã chạy được (17/10/2026) |
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

`README.md` mục 2 ("Cài đặt và chạy") và `docs/USAGE_GUIDE.md`. Luồng sản phẩm: `docs/design/product.md` mục 3.

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

Trang web của sản phẩm (`app/web/`, D22): server chỉ dùng thư viện chuẩn Python (python3 có PyYAML của máy, hoặc của env `gs-gsplat`; `scripts/web_server.sh` tự chọn). Thư viện của trình xem trong trình duyệt: three.js 0.180.0 và Spark 2.4.0 (MIT); phông chữ Be Vietnam Pro 5.3.0 (gói `@fontsource/be-vietnam-pro`, SIL OFL 1.1, chỉ các file latin và vietnamese, 4 độ đậm); ghim phiên bản và integrity SHA-512 của gói npm trong `configs/environment/web_viewer.lock.json`, tải vào `app/web/static/vendor/` (ngoài Git) bằng `python3 -m app.web.vendor`. Trình duyệt cần WebGL2.

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
- **Edit Pod làm pod khởi động lại** (08/11/2026 khi thêm cổng; 12/11/2026 giữa lượt đánh giá theo đoạn): việc đang chạy hoặc xếp hàng của `scripts/pod/job.sh` mất (trạng thái MẤT). Gửi lại cùng lệnh: `run_segments.sh` bỏ qua lượt đã OK, lượt dở được dời sang `*.prev_<thời điểm>` rồi chạy lại từ đầu.
- **`job.sh status` in `binary file matches`:** dòng cuối của log lấy từ 4.000 byte cuối (`tail -c`); khi chỗ cắt rơi vào giữa một ký tự UTF-8 nhiều byte, `grep` 3.11 trong locale UTF-8 coi là file nhị phân. Chỉ ảnh hưởng phần hiển thị; xem log bằng `tail -n`. Sửa đề xuất: `grep -a` (chưa sửa).
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

**12–13/11/2026, repo này (commit `49c208b`, sau đó `52afb1d`, lấy bằng `git pull`).** RTX 4090 (`torch` 2.7.1+cu128, Python 3.10.22 trong env `gs-gsplat`), cùng network volume với các lượt trước, đã tăng dung lượng trong lượt chạy. Giờ ghi theo đồng hồ pod (UTC; giờ Việt Nam cộng 7).

| Việc | Thời gian | Kết quả |
|---|---|---|
| `seg_video8` (`scripts/pod/run_segments.sh video8`, 15 lượt) | 2 giờ 47 phút (07:06–09:53) | 15/15 lượt OK |
| `seg_video9` | 1 giờ 23 phút (09:53–11:16) | FAILED: 9 lượt OK; 6 lượt `mono`, `kr` dừng ở vòng 0 vì bộ kiểm NaN (sửa ở `4251ff2`) |
| `seg_video10` | từ 11:16 tới khi pod khởi động lại (Edit Pod, sau 12:55) | 4 lượt OK; lượt thứ năm (`kr`, seed 42) đang chấm thì mất |
| `danh_gia_theo_doan` (chạy nốt `video9`, `video10`; bảng từng phòng và bảng gộp; luật mục 3.8; đóng gói) | 4 giờ 54 phút (13:09–18:03) | 17 lượt OK, đủ 45/45; gói `ket_qua_danh_gia_theo_doan_20261112_180311.tar.gz` (308 MB) |
| `dong_goi` (ảnh so sánh ba phòng, đóng gói lại) | khoảng 1 phút | `ket_qua_danh_gia_theo_doan_day_du_20261112_180406.tar.gz` (337 MB): như gói trên, thêm ảnh so sánh |

Tổng thời gian train của 45 lượt: 9,1 giờ (`video8` 2,4; `video9` 2,1; `video10` 4,7 giờ), chưa kể bước chấm. PLY cộng checkpoint: 34 GB, mỗi lượt khoảng 0,34 / 0,37 / 1,6 GB ở `video8` / `video9` / `video10` (`video10` có khoảng 3,6 triệu Gaussian). Số liệu: `docs/RESULTS.md` mục 6.2.

**08/11/2026, repo này (commit `b428d95`, lấy bằng `git pull`).** Pod mới (RTX 4090 24 GB, image ở mục 1), cùng network volume với lượt 31/10/2026.

| Việc | Thời gian | Kết quả |
|---|---|---|
| `run_all` (env đã có: kiểm tra lại, smoke test) | 3 phút 34 giây | 5/5 bước OK |
| `bash scripts/web_server.sh start` | | Server chạy; trang chỉ vào được sau khi thêm cổng HTTP 8000 (Edit Pod, pod khởi động lại; trước đó proxy trả 404, cổng hiện "Initializing" cho tới khi server chạy) |
| Dựng phòng `video10` từ trang web (video 2 phút 54 giây) | train 8 phút 23 giây (các bước khác chưa ghi) | Xong: 249 ảnh định vị, `model.splat` 637.031 Gaussian (20,4 MB), xem được 3D (`docs/RESULTS.md` mục 7) |
| Dựng lại cùng video (phòng `1`, quay màn hình demo) | 18 phút 45 giây: tách khung 12 giây, COLMAP 9 phút 40 giây, train 8 phút 32 giây, xuất 18 giây | Xong: 642.142 Gaussian (20,5 MB); cả ngày 7 phòng dựng xong, 2 dự án dừng ở bước COLMAP |

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
