# Môi trường thực nghiệm

_Cập nhật: 19/10/2026 · Trạng thái: **đề xuất**. Hai lượt chạy ngày 17/10/2026 của repo trước (mục 7) cho thấy PyTorch 2.7.1 + cu128 chạy được với driver CUDA 12.8 và 13.0; smoke test của cả hai env đều đã PASSED. Các file `configs/environment/*.conf` chưa được chuyển khỏi `provisional` (`scripts/pod/run_all.sh` đặt `ALLOW_PROVISIONAL=1`). Repo này chưa cài môi trường trên pod nào._

## 1. Bố cục trên pod Runpod

`/workspace` là network volume: còn nguyên khi Stop hoặc xóa pod. Container disk (mọi thứ ngoài `/workspace`, kể cả `/root`) bị xóa khi Stop pod, nên mọi thứ cần giữ phải nằm trong `/workspace`.

Bố cục (chi tiết và ba luồng dữ liệu: `docs/design/architecture.md` mục 2 và 3):

```
/workspace/
├── SpatialRoom3D/            # code của đồ án (repo Git, clone từ GitHub bằng deploy key)
├── code/gaussian-splatting/  # repo 3DGS của Inria, ghim theo commit
├── code/gsplat/              # repo gsplat, ghim theo commit (hai chế độ huấn luyện Default / MCMC)
├── miniforge3/               # conda (Miniforge) + env gs-inria (3DGS) + env gs-tools (COLMAP, ffmpeg) + env gs-gsplat
├── data/                     # chỉ dữ liệu gốc, pipeline không ghi vào (ngoại lệ: docs/design/architecture.md mục 3)
│   ├── custom_scenes/<cảnh>/ # ảnh / video tự chụp
│   ├── benchmarks/           # deep_blending/, mipnerf360/
│   └── downloads/            # file tải về (tandt_db.zip, ...)
├── outputs/                  # mọi thứ sinh ra
│   ├── custom_scenes/<cảnh>/ # colmap/ (ảnh đã chuẩn hóa, COLMAP, báo cáo) + <chế_độ>_s<seed>/ (các lượt train)
│   ├── benchmarks/<benchmark>/   # inria_b0/, gsplat/, gsplat_quick/
│   ├── checkpoints/          # PLY đã xuất để tải về / xem 3D (scripts/run_render.sh)
│   ├── reports/              # báo cáo môi trường, pip freeze, conda env, tóm tắt run_all_*.tsv, bảng so sánh
│   ├── logs/                 # log cài đặt và huấn luyện; logs/jobs/<việc>.log: log của việc chạy nền
│   └── jobs/                 # trạng thái các việc chạy nền (scripts/pod/job.sh), còn lại sau khi tắt pod
├── ket_qua_*.tar.gz          # gói kết quả nhẹ để tải về (scripts/pod/pack_results.sh)
└── .ssh/                     # deploy key (chmod 600): không bao giờ đưa lên Git hay gửi đi
```

## 2. Truy cập repo private từ pod

Repo `truong0809/SpatialRoom3D` là private. Pod dùng **deploy key chỉ đọc**: khóa chỉ có quyền đọc đúng một repo này, không push được. Ô 2 và ô 3 của `app/notebooks/runpod.ipynb` làm các bước dưới đây.

```bash
mkdir -p /workspace/.ssh && chmod 700 /workspace/.ssh
[ -f /workspace/.ssh/github_deploy ] || ssh-keygen -t ed25519 -N "" -C "runpod-spatialroom3d" -f /workspace/.ssh/github_deploy
cat /workspace/.ssh/github_deploy.pub   # chỉ in khóa công khai
```

1. Copy dòng khóa công khai vừa in ra.
2. Trên GitHub: repo `SpatialRoom3D` → **Settings → Deploy keys → Add deploy key**, dán khóa, **không** tick "Allow write access".
3. Clone repo trên pod:

```bash
SSH_CMD="ssh -i /workspace/.ssh/github_deploy -o IdentitiesOnly=yes -o UserKnownHostsFile=/workspace/.ssh/known_hosts -o StrictHostKeyChecking=accept-new"
GIT_SSH_COMMAND="$SSH_CMD" git clone -b main git@github.com:truong0809/SpatialRoom3D.git /workspace/SpatialRoom3D
git -C /workspace/SpatialRoom3D config core.sshCommand "$SSH_CMD"
```

Sau đó cập nhật code bằng `git -C /workspace/SpatialRoom3D pull --ff-only`. Khóa và `known_hosts` nằm trên volume nên pod mới gắn cùng volume không phải làm lại.

## 3. Môi trường cho 3DGS gốc (baseline B0)

### 3.1 Mã nguồn được ghim

| Thành phần | Phiên bản | Ghi chú |
|---|---|---|
| graphdeco-inria/gaussian-splatting | commit `54c035f` (30/10/2024) | Commit mới nhất trên `main` khi kiểm tra ngày 09/10/2026 |
| diff-gaussian-rasterization | `9c5c202` (nhánh `dr_aa`) | Submodule, kèm thư viện glm |
| simple-knn | `86710c2` | Submodule, tải từ gitlab.inria.fr; dự phòng mirror GitHub `camenduru/simple-knn` (có đúng commit này) |
| fused-ssim | `1272e21` | Submodule |
| SIBR_viewers | không cài | Viewer desktop, không cần cho huấn luyện |

### 3.2 Vì sao không dùng nguyên `environment.yml` của Inria

File gốc ghim Python 3.7.13, PyTorch 1.12.1, `cudatoolkit` 11.6. Không dùng nguyên trạng vì:

- Python 3.7 đã hết hỗ trợ từ lâu.
- Gói `cudatoolkit` của conda chỉ có runtime, không có `nvcc` để build CUDA extension. README của Inria cũng cho biết họ build extension bằng CUDA SDK 11.8 cài riêng.
- CUDA 11.x không nhận GCC mới hơn 11 làm trình biên dịch host.

README của Inria ghi nhận thử nghiệm giới hạn cho thấy code chạy được với PyTorch 2.0 và CUDA 12.

### 3.3 Phương án đề xuất

- Miniforge và conda env `gs-inria` cài trong `/workspace`, độc lập với Python của template.
- Miniforge `26.7.2-0` (ghim phiên bản bộ cài, không dùng "latest"), Python 3.10.22; PyTorch và biến thể CUDA ghim trong `configs/environment/gs-inria.conf` (hiện tạm đặt 2.7.1 + CUDA 12.8). Các gói Python còn lại ghim trong `configs/environment/gs-inria.lock.txt` theo đúng lượt cài đã cho B0 đạt (mục 7).
- Build extension bằng `nvcc` của template, cùng major version với CUDA của PyTorch.
- Build sẵn cho các kiến trúc 8.0, 8.6, 8.9, 9.0, để dùng lại env khi đổi giữa A100, A6000/A40, RTX 4090/L40S, H100.

**Quy tắc chọn biến thể CUDA của PyTorch:** không mới hơn phiên bản CUDA mà driver hỗ trợ (dòng "CUDA Version" của `nvidia-smi`). GPU Blackwell cần CUDA ≥ 12.8.

Biến thể tạm chọn `cu128` cần driver hỗ trợ CUDA ≥ 12.8 (dòng driver R570 trở lên). Nếu pod chỉ có driver 12.4 hoặc 12.6, script cài đặt dừng ở bước kiểm tra; khi đó chọn host khác, hoặc đặt `TORCH_CUDA_OVERRIDE=cu126` (PyTorch 2.7.1 có bản này; ô 4 của notebook).

**Khác biệt so với môi trường gốc** (phải nêu trong báo cáo khi so sánh số liệu):

- Phiên bản Python, PyTorch và CUDA.
- `opencv-python-headless` thay cho `opencv-python`, vì server thường không có libGL.

## 4. Cài đặt và kiểm tra

Cách gọn nhất là `bash scripts/pod/run_all.sh --background --setup-only` (ô 4 của notebook, mục 4.2). Từng script riêng:

```bash
cd /workspace/SpatialRoom3D
bash scripts/setup/env_probe.sh                 # GPU, driver, CUDA, mạng; báo cáo: /workspace/outputs/reports
ALLOW_PROVISIONAL=1 bash scripts/setup/setup_3dgs_inria.sh   # log: /workspace/outputs/logs, báo cáo: /workspace/outputs/reports
```

Script dừng ngay nếu:

- đang có việc khác dùng GPU (khóa GPU, mục 4.5);
- phiên bản trong file conf còn ở trạng thái `provisional` mà không đặt `ALLOW_PROVISIONAL=1`;
- driver không hỗ trợ CUDA đã chọn;
- compute capability của GPU không nằm trong `TORCH_CUDA_ARCH_LIST` (extension sẽ không có mã máy cho GPU đó);
- không thấy `nvcc` cùng major version;
- thiếu dung lượng (chỉ kiểm tra được trên ổ thường, xem mục 6 về network volume), hoặc không truy cập được GitHub, PyPI, kho PyTorch hay conda;
- một submodule cần thiết bị thiếu hoặc không đúng commit được ghim.

Chạy lại an toàn: Miniforge hỏng hoặc cài dở được cài đè (`-u`); env chỉ được coi là có khi có `conda-meta/history`; PyTorch được cài theo đúng biến thể (`torch==2.7.1+cu128`) và kiểm tra lại phiên bản lẫn CUDA sau khi cài. Extension được build lại khi đổi commit Inria, danh sách kiến trúc hoặc PyTorch (dấu build `.sr3d_inria_build` trong env). Bước cuối chạy `scripts/setup/smoke_test_3dgs.py` để kiểm tra cả ba extension trên GPU.

Kích hoạt env trong phiên terminal mới:

```bash
. /workspace/miniforge3/etc/profile.d/conda.sh && conda activate gs-inria
```

### 4.1 Công cụ xử lý dữ liệu (env `gs-tools`)

COLMAP và ffmpeg được cài vào conda env riêng `gs-tools`, tách khỏi `gs-inria` để tránh xung đột thư viện. Phiên bản ghim trong `configs/environment/gs-tools.conf`: COLMAP 4.2.1 từ conda-forge (bản mới nhất khi kiểm tra ngày 09/10/2026, có bản build CUDA và bản CPU).

```bash
bash scripts/setup/setup_data_tools.sh                            # tự chọn bản CUDA hoặc CPU theo driver
COLMAP_BUILD_OVERRIDE=cpu bash scripts/setup/setup_data_tools.sh  # ép dùng bản CPU
```

Bản CUDA của colmap 4.2.1 trên conda-forge chỉ có build `cuda_129` và `cuda_134` (kiểm tra 17/10/2026), nên cần driver hỗ trợ CUDA ≥ 12.9 (`COLMAP_CUDA_MIN` trong `configs/environment/gs-tools.conf`). Driver thấp hơn thì script tự cài bản CPU. `src/reconstruction/colmap_runner.py` đọc phần đầu `colmap -h`: nếu bản COLMAP không có CUDA thì tự trích và so khớp đặc trưng trên CPU (`--gpu auto`); lựa chọn được ghi vào `sfm_report.json` (`colmap_settings.colmap_cuda`, `colmap_settings.use_gpu`).

Vì sao không dùng `convert.py` của Inria: script này truyền `--SiftExtraction.use_gpu` và `--SiftMatching.use_gpu`. Mã nguồn COLMAP hiện hành đã đổi thành `--FeatureExtraction.use_gpu` và `--FeatureMatching.use_gpu` (thay đổi nằm trong đợt tái cấu trúc trích đặc trưng của bản 3.13), nên `convert.py` không chạy được với COLMAP 4.x. `src/reconstruction/colmap_runner.py` làm đúng các bước của `convert.py` nhưng tự nhận tên tùy chọn theo phiên bản COLMAP.

### 4.2 Các lệnh chạy chính trên pod

Các lệnh gom cả luồng, nên dùng (chạy nền, xếp hàng GPU, chạy lại an toàn):

```bash
bash scripts/pod/run_all.sh --background --setup-only               # pod mới: chỉ cài / kiểm tra môi trường
bash scripts/pod/run_all.sh --background                            # kiểm tra -> cài đặt -> chạy thử -> B0 -> gsplat đầy đủ
bash scripts/pod/run_scene.sh /workspace/data/custom_scenes/<cảnh>/<video.mp4> <cảnh> --background   # video: frame + COLMAP -> default -> mcmc
bash scripts/pod/run_scene.sh /workspace/data/custom_scenes/<cảnh> <cảnh> --background               # ≥ 10 ảnh: ảnh + COLMAP -> default -> mcmc
bash scripts/pod/job.sh status                                      # trạng thái các việc chạy nền
bash scripts/pod/results.sh [tên_cảnh]                              # xem nhanh kết quả (chỉ đọc)
bash scripts/pod/pack_results.sh                                    # gói kết quả nhẹ để tải về
bash scripts/pod/check_datasets.sh                                  # dữ liệu Deep Blending / Mip-NeRF 360 / cảnh (chỉ đọc)
```

Thứ tự bước của `run_all.sh`: `env_probe` → `setup_inria` → `data_tools` → `setup_gsplat` → `quick_test` (gsplat 3.000 vòng) → `b0_inria` → `modes_full` → `pack`. Ba bước benchmark gọi `scripts/run_train.sh deep_blending gsplat` (`QUICK=1` cho `quick_test`) và `scripts/run_train.sh deep_blending b0`; `--setup-only` bỏ qua cả ba. `env_probe` có `[FAIL]` thì dừng cả lượt; bước lỗi khác chỉ làm bỏ qua các bước cần nó. Tóm tắt từng bước ghi vào `outputs/reports/run_all_<thời gian>.tsv`. Các biến tùy chọn (`TORCH_CUDA_OVERRIDE`, `MAX_JOBS`, `SEEDS`, `SKIP`...) ghi ở đầu script.

`run_scene.sh` dùng lại mô hình COLMAP đã có của cảnh (không đọc lại video hay ảnh); `--redo-colmap` làm lại bước ảnh đầu vào và COLMAP. Đầu vào là ảnh thì không chọn frame; các kiểm tra và gợi ý trước khi train: `docs/USAGE_GUIDE.md` mục 3.4. COLMAP ghi `FAIL` thì không train. Lượt train OK cũ được train lại khi nó làm từ dữ liệu cũ (`src.training.run --replace-stale`): mô hình COLMAP của cảnh đã đổi (dấu SHA-256 của `sparse/0`), hoặc với `mcmc`, chỉ ngân sách Gaussian khác vì lượt `default` đã train lại. Việc so sánh dựa trên `run_meta.json`, nên vẫn đúng khi lượt trước bị dừng giữa chừng. Lượt cũ khác ở cấu hình do người chạy chọn (seed, protocol, tham số chế độ, `--test-list`, `--cap-max`) thì báo lỗi; `run_scene.sh --retrain` train lại mọi chế độ. Thư mục cũ được đổi tên `<thư_mục>.prev_<thời gian>`, không xóa. `scripts/pipeline/train_gsplat_modes.sh` dùng cùng cơ chế cho `mcmc`.

Các bước của luồng chính vẫn chạy riêng được (bốn script ở gốc `scripts/`, phần thực thi trong `scripts/pipeline/`):

```bash
bash scripts/prepare_data.sh custom <video.mp4 | thư_mục_ảnh> <tên_cảnh>   # frame hoặc ảnh -> COLMAP -> báo cáo
bash scripts/run_train.sh custom <tên_cảnh> [default] [mcmc]               # baseline rồi mcmc trên cảnh tự chụp
bash scripts/run_render.sh custom <tên_cảnh> [<lượt>]                      # render lại, xuất PLY
bash scripts/run_evaluate.sh custom <tên_cảnh>                             # bảng so sánh
bash scripts/run_train.sh deep_blending b0                                 # tái lập B0 trên Deep Blending
bash scripts/run_train.sh deep_blending gsplat                             # hai chế độ gsplat trên Deep Blending
```

Benchmark, protocol và nơi lưu kết quả: `docs/design/experiments.md`.

### 4.3 Env `gs-gsplat` (hai chế độ huấn luyện Default / MCMC)

Thiết kế và giao thức so sánh: `docs/design/training_modes.md`. Env tách riêng khỏi `gs-inria`. Lý do: hai bên cần hai commit fused-ssim khác nhau (cùng tên gói `fused_ssim`), và `examples/` của gsplat yêu cầu `numpy<2`.

| Thành phần | Phiên bản | Ghi chú |
|---|---|---|
| gsplat | commit `6e8c837` (22/08/2025) = v1.5.3 + 4 commit | Thư viện (build sẵn extension, kiểm tra bằng `from gsplat import csrc`) và `examples/` lấy từ cùng commit, không sửa file |
| glm | submodule của gsplat | Theo commit gsplat ghim |
| fused-ssim | `328dc98` | Đúng commit trong `examples/requirements.txt` của gsplat |
| pycolmap (fork rmbrualla) | `cc7ea4b` | Parser của gsplat dùng lớp `SceneManager` của fork này, không phải gói `pycolmap` chính thức |
| Phụ thuộc Python khác | `configs/environment/gs-gsplat.lock.txt` | 90 gói (88 từ PyPI, 2 từ git), ghim theo ngày 23/08/2025 bằng `uv pip compile --exclude-newer`; đầu vào `configs/environment/gs-gsplat.in`. Có PyYAML (đọc `configs/*.yaml`) |
| PyTorch | giống `gs-inria` (tạm 2.7.1 + cu128) | `setup_gsplat.sh` cảnh báo nếu hai file conf khác nhau |

```bash
ALLOW_PROVISIONAL=1 bash scripts/setup/setup_gsplat.sh   # 9 bước như setup_3dgs_inria.sh; smoke test: scripts/setup/smoke_test_gsplat.py
```

### 4.4 Chạy trên Runpod từ Jupyter (pod mới, volume trống)

**Tạo pod** (cấu hình đã chạy được ngày 17/10/2026, mục 7)

- Secure Cloud, gắn network volume ≥ 100 GB cùng data center (mount tại `/workspace`).
- GPU có compute capability 8.0 / 8.6 / 8.9 / 9.0: RTX 4090, RTX A6000, A40, L40S, A100, H100.
- Image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (bản **devel**, có `nvcc` 12.4 và gcc 11). Driver phải hỗ trợ CUDA ≥ 12.8 (12.6 với `TORCH_CUDA_OVERRIDE=cu126`).

**Các bước:** tải `app/notebooks/runpod.ipynb` lên `/workspace` trong Jupyter, chạy ô 1 → 2 → 3 → 4 → 6 (chi tiết: `docs/USAGE_GUIDE.md` mục 2). Tương đương bằng lệnh:

```bash
# ô 2: khóa deploy (mục 2), thêm khóa công khai vào GitHub; ô 3: clone
GIT_SSH_COMMAND="$SSH_CMD" git clone -b main git@github.com:truong0809/SpatialRoom3D.git /workspace/SpatialRoom3D
cd /workspace/SpatialRoom3D
bash scripts/pod/run_all.sh --background --setup-only   # ô 4: cài môi trường (không train)
bash scripts/pod/job.sh status                          # ô 5: theo dõi; log: bash scripts/pod/job.sh log run_all
bash scripts/prepare_data.sh deep_blending              # ô 6: tải Deep Blending (không cần GPU)
bash scripts/pod/run_all.sh --background                # ô 7: baseline Deep Blending (tốn GPU, khi đã duyệt)
bash scripts/pod/results.sh                             # kết quả; gói tải về: /workspace/ket_qua_*.tar.gz
```

Thời gian đo được ngày 17/10/2026 trên RTX 4090 (repo trước, cùng script cài đặt và cùng tham số): cài `gs-inria` 29 phút, cài `gs-gsplat` 32 phút (lượt đầu) và 7 phút (lượt thứ hai), cài `gs-tools` 18 phút, B0 49,5 phút, gsplat chạy thử 3.000 vòng 17 phút, gsplat đầy đủ (2 cảnh × `default` + `mcmc`, seed 42) 74 phút (mục 7). Pod vẫn tính tiền khi việc chạy nền; xong thì Terminate (hoặc `AUTO_STOP=1`, chưa thử). Mọi thứ trong `/workspace` vẫn còn.

**Gửi lại:** gói `ket_qua_*.tar.gz` (báo cáo, log, trạng thái việc, JSON/CSV kết quả; không có checkpoint hay PLY, không có `/workspace/.ssh`).

### 4.5 Chạy nền và khóa GPU (`scripts/pod/job.sh`)

- `bash scripts/pod/job.sh start <tên> -- <lệnh>` chạy lệnh trong phiên riêng (`setsid nohup`): đóng trình duyệt hay tắt kernel Jupyter không làm dừng việc.
- Mỗi lúc chỉ một việc dùng GPU: khóa `flock` trên `/run/lock/spatialroom3d-gpu.lock` (ổ của pod, mất theo khi tắt pod nên không bao giờ "treo"). Việc mới xếp hàng tới khi khóa nhả. Script chạy trực tiếp (không qua `job.sh`) mà gặp khóa đang bị giữ thì dừng ngay, không chạy chen.
- Mỗi việc có một token riêng trong biến môi trường `SR3D_JOB_TOKEN`, mọi tiến trình con đều mang theo. `status` và `stop` tìm tiến trình theo token, không theo số pid (pid lặp lại sau khi pod khởi động lại). `stop` gửi SIGTERM rồi SIGKILL sau `STOP_TIMEOUT` giây (mặc định 20) cho mọi tiến trình mang token, kể cả tiến trình đã `setsid` hoặc mồ côi.
- Trạng thái nằm ở `/workspace/outputs/jobs/<tên>.state`, log ở `/workspace/outputs/logs/jobs/<tên>.log`. Sau khi pod tắt, việc dở dang hiện là `MẤT`; chạy lại lệnh là đủ (các bước đã xong tự bỏ qua).
- Lệnh: `status [tên]`, `log <tên> [số dòng]`, `stop <tên> | --all`, `busy` (exit 0 khi đang có việc).
- Tiến trình con kế thừa fd khóa nên khóa chỉ nhả khi cả cây tiến trình kết thúc. `subprocess` của Python đóng mọi fd kế thừa, nên `src.training.run` và `colmap_runner.py` truyền riêng fd khóa (`SR3D_GPU_LOCK_FD`, `src/utils/gpu_lock.py`) cho tiến trình huấn luyện và COLMAP. Tiến trình phụ chạy lâu không dùng GPU (log `nvidia-smi -l`) đóng fd này.
- `AUTO_STOP=1` khi start một việc (kể cả lệnh start lặp lại cho việc đang chạy): ghi yêu cầu `<file khóa>.auto_stop` kèm phiên của pod. Việc **cuối cùng** kết thúc bình thường (kể cả việc xếp hàng sau đó) gọi `runpodctl pod stop $RUNPOD_POD_ID`. Yêu cầu bị hủy khi: start với `AUTO_STOP=0`; `stop` xong mà không còn việc nào; pod khởi động lại (phiên khác). Chưa thử trên pod.

## 5. Mốc đối chiếu cho Bước 4

Bước 4 tái lập kết quả trên **Deep Blending**: 2 cảnh trong nhà (playroom, drjohnson), Inria cung cấp sẵn dữ liệu COLMAP trong `tandt_db.zip`.

Số liệu Inria công bố cho code hiện hành (`results.md`, rasterizer mặc định, cấu hình gốc), trung bình 2 cảnh:

| PSNR | SSIM | LPIPS |
|---|---|---|
| 29.690 | 0.906 | 0.238 |

- **Ngưỡng chấp nhận đề xuất:** lệch PSNR ≤ 0,5 dB, SSIM ≤ 0,01, LPIPS ≤ 0,01.
- **Cách chạy:** dùng đúng `train.py --eval`, `render.py` và `metrics.py` của Inria.
- **Thời gian tham khảo:** Inria báo cáo khoảng 45 phút huấn luyện với rasterizer gốc khi bật đủ tính năng; biểu đồ không ghi rõ GPU.

## 6. Sự cố đã biết

- `cuda_rasterizer/rasterizer_impl.h` dùng `uint32_t` nhưng không `#include <cstdint>`. Với GCC ≥ 13 có thể lỗi biên dịch. Cách xử lý: dùng GCC ≤ 12, hoặc thêm dòng include và ghi nhận đây là thay đổi so với mã gốc.
- `simple-knn` tải từ gitlab.inria.fr. Trang web của host này có lớp chống bot và từ chối một số truy cập tự động. Nếu git không clone được, script tự lấy cùng commit `86710c2` từ mirror GitHub; vì commit được xác định bằng mã SHA nên mã nguồn giống hệt bản gốc. Đường dự phòng này đã chạy thử thành công trong sandbox.
- `metrics.py` tải trọng số VGG và LPIPS từ Internet ở lần chạy đầu.
- Khi bật bù phơi sáng (`--train_test_exp`), Inria đưa nửa trái ảnh test vào huấn luyện và chỉ đánh giá nửa phải. Số liệu khi đó không so được với cấu hình chuẩn.
- Với `--eval`, code Inria luôn lấy mỗi ảnh thứ 8 (theo thứ tự tên) làm ảnh test: giá trị `llffhold=8` được gán cứng ở chỗ gọi hàm. Nhánh đọc danh sách ảnh test từ `sparse/0/test.txt` có trong code nhưng không bật được qua tham số dòng lệnh. Muốn dùng đoạn quay kiểm tra riêng (theo `docs/data/capture_protocol.md`), cần thêm một thay đổi nhỏ hoặc công cụ chia dữ liệu riêng — sẽ quyết định ở mốc M4 và ghi nhận là khác biệt so với mã gốc.
- COLMAP từ bản 3.12 ghi thêm `rigs.bin` và `frames.bin`. Định dạng `cameras.bin`, `images.bin`, `points3D.bin` giữ nguyên (đã đối chiếu mã nguồn COLMAP), nên bộ đọc của Inria vẫn dùng được.
- **gsplat v1.5.3** (bản phát hành mới nhất khi kiểm tra) có lỗi trong `DefaultStrategy`: điều kiện `step % reset_every == 0 & step > 0` luôn sai vì thứ tự ưu tiên toán tử, nên opacity không bao giờ được reset. Lỗi được sửa ở commit `6e8c837` (PR #776). v1.3.0 thì reset cả ở bước 0 (sửa ở PR #735). Vì vậy env ghim `6e8c837`.
- `examples/` của gsplat dùng fork `rmbrualla/pycolmap`. Cài gói `pycolmap` chính thức sẽ làm parser lỗi; đây là nguyên nhân notebook demo phải trộn file từ nhiều phiên bản (`docs/research/demo_review.md`).
- `examples/datasets` của gsplat là namespace package (không có `__init__.py`). Nếu env cài thêm một gói tên `datasets` (vd. của Hugging Face), gói đó sẽ che mất parser của gsplat. `src/training/gsplat_launcher.py` kiểm tra nguồn module và dừng nếu gặp trường hợp này. Gói `datasets` của repo này nằm trong package `src` (`src.datasets`), không trùng tên ở mức cao nhất.
- `simple_trainer.py` tự huấn luyện phân tán khi thấy nhiều GPU, và gán cứng seed 42. `src.training.run` giới hạn còn một GPU; launcher cho đổi seed.
- fused-ssim `328dc98` tự dò kiến trúc của GPU lúc build và thêm cờ `-arch`. Khi đó PyTorch bỏ qua `TORCH_CUDA_ARCH_LIST`. Extension chỉ có mã máy cho GPU lúc cài, kèm PTX: GPU có compute capability cao hơn vẫn chạy được (biên dịch PTX lúc chạy), GPU thấp hơn thì không. `setup_gsplat.sh` ghi compute capability vào dấu build (`.sr3d_gsplat_build`) và tự build lại khi đổi loại GPU; `scripts/pipeline/train_scene.sh` và `scripts/pipeline/train_gsplat_modes.sh` chạy thử fused-ssim trên GPU trước khi huấn luyện. Bản fused-ssim trong repo Inria (`1272e21`) không có hành vi này.
- `opencv-python-headless` được ghim 4.11.0.86 vì bản 4.12 yêu cầu `numpy>=2`, xung đột với `numpy<2` của gsplat. `viser` được ghim 0.2.23 vì `nerfview` 0.1.2 và viewer của gsplat viết cho viser 0.2.x.
- **Chạy script từ Jupyter** (gặp ở lượt chạy 17/10/2026): ipykernel đặt `MPLBACKEND=module://matplotlib_inline.backend_inline` cho mọi tiến trình con. Env `gs-gsplat` không có `matplotlib_inline`, nên `import matplotlib` (torchmetrics và `simple_trainer.py` đều nạp) báo `ValueError: Key backend ... is not a valid value for backend`; smoke test gsplat FAIL dù phần build đã xong. Đã tái hiện với matplotlib 3.10.5 (bản trong file khóa). `setup_gsplat.sh`, `train_gsplat_modes.sh`, `train_scene.sh`, `run_render.sh` và `job.sh` đặt `MPLBACKEND=Agg`.
- **COLMAP bản CUDA không cài được với driver CUDA 12.8** (lượt chạy 17/10/2026): conda báo `nothing provides __cuda >=12.9 needed by colmap-4.2.1-cuda_129...`. Xử lý ở mục 4.1: tự dùng bản CPU.
- **`df` trên network volume** (MooseFS, lượt chạy 17/10/2026) báo dung lượng của cả cụm lưu trữ (2,3 PB), không phải quota của volume. Script cài đặt nhận ra trường hợp này (kích thước > 50.000 GB) và chỉ cảnh báo; dung lượng volume xem trên trang Runpod.
- **COLMAP bản không có CUDA mà gọi với `use_gpu=1`** sẽ trích SIFT bằng OpenGL và phải tạo ngữ cảnh OpenGL qua Qt (mã nguồn COLMAP 4.2.1, `FeatureExtractionOptions::RequiresOpenGL`), không chạy được trên pod không có màn hình. Chưa gặp trên pod; `colmap_runner.py` chỉ bật GPU khi bản COLMAP có CUDA (`--gpu auto`), và mọi lệnh COLMAP chạy với `QT_QPA_PLATFORM=offscreen` để phòng hờ.
- **`pip install torch==2.7.1`** coi bản đã cài là đủ dù khác biến thể CUDA (vd. `+cu126` thay vì `+cu128`). Script cài ghi rõ biến thể (`torch==2.7.1+cu128`).
- **PyYAML:** `configs/*.yaml` được đọc bằng `python3` của máy nếu có PyYAML (template Runpod thường có sẵn qua JupyterLab; ô 1 của notebook kiểm tra), nếu không thì bằng python của env `gs-gsplat`. Env `gs-inria` không có PyYAML: script B0 đọc bản JSON của mục `benchmark` mà `scripts/lib/config.sh` ghi ra `outputs/reports/configs/`.

## 7. Kết quả thực tế (repo trước)

Các lượt dưới đây chạy bằng repo trước (`truong0809/3D-Indoor-Scene-Reconstruction-from-Images`), trên volume cũ, với bố cục và tên file cũ. Repo này dùng cùng script cài đặt, cùng phiên bản ghim và cùng lệnh huấn luyện (chỉ khác đường dẫn), nhưng **chưa chạy lại** trên pod.

### Lượt chạy 17/10/2026 (repo trước, commit `8037a66`)

Nguồn: gói `ket_qua_1017_0518.tar.gz` (log, báo cáo `env_*.txt`, `setup_3dgs_inria_*.txt`, `baseline_db_summary_*.json`).

| Mục | Giá trị |
|---|---|
| Pod | Secure Cloud, data center EU-RO-1, 1 × RTX 4090 24 GB (cc 8.9), On-Demand; `/workspace` là network volume (fuse), container disk 20 GB |
| Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (Ubuntu 22.04.5, nvcc 12.4, gcc 11.4.0) |
| Driver | 570.195.03, hỗ trợ CUDA 12.8 |
| Env `gs-inria` | Python 3.10.22, torch 2.7.1+cu128; extension build cho 8.0/8.6/8.9/9.0; smoke test **PASSED**; cài mất 29 phút |
| Env `gs-gsplat` | gsplat 1.5.3 (`6e8c837`) và fused-ssim build xong; smoke test FAIL do `MPLBACKEND` (mục 6), đã sửa ở lượt sau; 32 phút |
| Env `gs-tools` | Không cài được bản CUDA (mục 6), đã chuyển sang tự chọn bản CPU |

**B0, Inria `54c035f` trên Deep Blending, step 30.000: ĐẠT**

| Cảnh | PSNR | SSIM | LPIPS | Train | Số Gaussian | VRAM đỉnh |
|---|---|---|---|---|---|---|
| drjohnson | 29,369 | 0,9051 | 0,2361 | 23,4 phút | 3.105.559 | 15,1 GB |
| playroom | 30,077 | 0,9081 | 0,2408 | 17,2 phút | 1.843.819 | 9,3 GB |
| Trung bình | 29,723 | 0,9066 | 0,2384 | | | |
| Inria công bố | 29,690 | 0,906 | 0,238 | | | |
| Chênh lệch | +0,033 | +0,0006 | +0,0004 | | | |

Cả ba chỉ số nằm trong ngưỡng mục 5 (0,5 dB / 0,01 / 0,01). Toàn bộ bước B0 (tải dữ liệu, train, render, metrics) mất 49,5 phút.

### Lượt chạy thứ hai 17/10/2026 (repo trước, `8037a66` + bản vá v3, qua `run_all.sh`)

Nguồn: gói kết quả lúc 17:42 (`reports/run_all_20261017_154255.tsv`, `env_20261017_154255.txt`, `setup_3dgs_inria_20261017_154301.txt`, `setup_data_tools_20261017_154556.txt`, `setup_gsplat_20261017_160408.txt`, `modes_db_20261017_162815.{md,json}`).

| Mục | Giá trị |
|---|---|
| Pod | 1 × RTX 4090 24 GB (cc 8.9), cùng image; driver 580.126.20, hỗ trợ CUDA 13.0 (pod khác lượt đầu) |
| `run_all.sh` | Cả 7 bước OK, tổng khoảng 2 giờ |
| Env `gs-inria` | Đã có, chỉ kiểm tra lại (3 phút) |
| Env `gs-tools` | COLMAP 4.2.1 bản CUDA (`cuda_129`, vì driver ≥ 12.9), ffmpeg 9.0.2; 18 phút |
| Env `gs-gsplat` | Smoke test **PASSED**; 7 phút |
| gsplat chạy thử 3.000 vòng | 2 cảnh × 2 chế độ, 17 phút |
| gsplat đầy đủ, seed 42 | 74 phút |

**gsplat `6e8c837` trên Deep Blending, step 30.000, seed 42**

| Cảnh | Chế độ | PSNR | SSIM | LPIPS | Số Gaussian | Train | VRAM đỉnh |
|---|---|---|---|---|---|---|---|
| drjohnson | `default` | 29,159 | 0,9005 | 0,2442 | 3.682.098 | 17,5 phút | 7,6 GiB |
| drjohnson | `mcmc` (`opacity_reg` 0,001) | 29,229 | 0,9036 | 0,2339 | 3.682.098 | 19,4 phút | 7,2 GiB |
| playroom | `default` | 29,836 | 0,9063 | 0,2481 | 2.363.710 | 13,0 phút | 4,9 GiB |
| playroom | `mcmc` (`opacity_reg` 0,001) | 29,839 | 0,9074 | 0,2332 | 2.363.710 | 13,6 phút | 4,6 GiB |

- `default` so với số liệu Inria công bố: **PASS** (trung bình ΔPSNR −0,192 dB, ΔSSIM −0,003, ΔLPIPS +0,008).
- `default` so với B0 tự chạy: **PASS** ở từng cảnh (ΔPSNR −0,209 dB ở drjohnson, −0,241 dB ở playroom).
- `mcmc` dùng `cap_max` bằng số Gaussian cuối của `default` (ngân sách EQUAL). Mới có một seed nên bảng so sánh ghi `seeds=FEW`: chưa kết luận `mcmc` tốt hơn.
