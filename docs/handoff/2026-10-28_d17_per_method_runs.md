# Bàn giao D17: chạy riêng từng phương pháp của thực nghiệm chính và tên bản

| | |
|---|---|
| Ngày | 28/10/2026 |
| Quyết định | D17 ([PROJECT_STATUS.md](../PROJECT_STATUS.md), mục "Quyết định đã chốt") |
| Commit | `c362b9e`, nhánh `main` |
| Trạng thái | Đã viết; đã kiểm thử trong sandbox (CPU, stub thay GPU / conda, mô phỏng notebook); **chưa chạy trên pod / GPU** |

## 1. Mục tiêu và phạm vi

Bàn giao này bổ sung cách chạy riêng từng phương pháp của thực nghiệm chính, cho cùng kết quả như chạy cả kế hoạch, và một hệ thống tên thống nhất cho các đầu ra.

**Hai cách chạy, một bộ kết quả:**
- code tổng: `app/notebooks/runpod.ipynb`, ô 7, `EXP = "main"` (cả kế hoạch);
- từng phương pháp, mỗi phương pháp một notebook:

  | Notebook | Phương pháp |
  |---|---|
  | `app/notebooks/b0_inria.ipynb` | B0, 3DGS gốc của Inria |
  | `app/notebooks/gsplat_default.ipynb` | `default`, 3DGS với DefaultStrategy của gsplat |
  | `app/notebooks/gsplat_mcmc.ipynb` | `mcmc`, 3DGS-MCMC của gsplat |

**Tên bản:** mỗi lần chạy có một tên thống nhất, dùng cho việc nền, báo cáo, log, gói kết quả và file PLY xuất ra (mục 2.3).

**Ngoài phạm vi:**
- không đổi thuật toán, chế độ huấn luyện, protocol hay kế hoạch `main`;
- không chạy GPU.

## 2. Thiết kế

### 2.1 Chạy riêng một phương pháp

`scripts/pod/run_experiment.sh main --only <b0|default|mcmc>` chạy phần của một phương pháp trong `configs/experiments/main.yaml` (`select_method`, `src/training/experiment.py`). Phần đó có cùng benchmark, cảnh, seed, protocol và thư mục lượt train với cả kế hoạch. Vì vậy lượt đã xong ở một cách chạy được cách chạy kia dùng lại, không train hai lần.

| `--only` | Lượt train | Bảng kết quả |
|---|---|---|
| `b0` | B0 trên Deep Blending (2 cảnh) | B0 so với số liệu Inria công bố |
| `default` | `default` trên 6 cảnh × 3 seed | Chỉ số của `default`; đối chiếu số liệu Inria công bố và B0 tự chạy |
| `mcmc` | `default` (dùng lại nếu đã có) rồi `mcmc`, 6 cảnh × 3 seed | `mcmc` so với `default` |

- **Phụ thuộc:** `mcmc` lấy ngân sách Gaussian (`cap_max`) từ lượt `default` cùng cảnh, cùng seed. Nếu lượt đó chưa có thì `default` được train trước.
- **Đối chiếu B0:**
  - cả kế hoạch bắt buộc đối chiếu (`B0_CHECK=required`);
  - khi chạy riêng một chế độ, chỉ đối chiếu nếu mọi cảnh đã có kết quả B0 đủ chỉ số (`B0_CHECK=if_present`), cùng tiêu chí với bảng so sánh;
  - kế hoạch in ra và báo cáo ghi rõ đã đối chiếu hay chưa, kèm các cảnh còn thiếu.
- **Thứ tự khuyến nghị:** `b0_inria` → `gsplat_default` → `gsplat_mcmc`.
- **Ước lượng** (RTX 4090, `estimate.only` trong kế hoạch): `b0` khoảng 1 giờ GPU, 5 GB; `default` khoảng 6 giờ, 30 GB; `mcmc` khoảng 12 giờ, 54 GB khi chưa có `default`, khoảng một nửa khi đã có. Các số này suy ra từ số đo ngày 17/10/2026 trên Deep Blending và từ giả định cho Mip-NeRF 360; **chưa đo**.

### 2.2 Notebook

- **Bộ sinh:** cả bốn notebook sinh từ `app/notebooks/build_notebooks.py`; `--check` xác nhận file trong repo đúng là bản sinh ra.
- **Ô dùng chung:** ô 1–5 (kiểm tra máy, khóa deploy, lấy code, cài môi trường, theo dõi) và ô dừng giống hệt nhau ở mọi notebook.
- **Ô riêng:** ô 6 chạy phương pháp (`CHAY = False` chỉ in kế hoạch); ô "Bảng" in báo cáo mới nhất của bản. Notebook `gsplat_default` và `gsplat_mcmc` có thêm ô cho cảnh tự chụp và ô xuất PLY.

### 2.3 Tên bản

| Bản | Tên | Lệnh |
|---|---|---|
| Code tổng (thực nghiệm chính) | `main` | `run_experiment.sh main` |
| B0 | `b0_inria` | `run_experiment.sh main --only b0` |
| `default` | `gsplat_default` | `run_experiment.sh main --only default` |
| `mcmc` | `gsplat_mcmc` | `run_experiment.sh main --only mcmc` |
| Dò tham số | `tune_mcmc` | `run_experiment.sh tune_mcmc` |

Đầu ra mang tên bản (ví dụ `gsplat_mcmc`):

| Đầu ra | Đường dẫn |
|---|---|
| Việc nền và log | `gsplat_mcmc`, `/workspace/outputs/logs/jobs/gsplat_mcmc.log` |
| Báo cáo, bảng các bước, bảng từng benchmark | `/workspace/outputs/reports/gsplat_mcmc_<thời gian>[_<benchmark>].{md,tsv,json}` |
| Bản chụp kế hoạch | `/workspace/outputs/reports/configs/gsplat_mcmc_<thời gian>.json` |
| Gói kết quả nhẹ | `/workspace/ket_qua_gsplat_mcmc_<thời gian>.tar.gz`, cùng thời gian với báo cáo |
| PLY xuất ra | `/workspace/outputs/checkpoints/<luồng>/<cảnh>/gsplat_mcmc_<cảnh>_s<seed>.ply`; lượt chạy thử thêm `_quick` |

Quy ước:
- **Thư mục lượt train giữ nguyên** (`<cảnh>/<chế độ>_s<seed>`): đây là khóa để hai cách chạy dùng lại lượt của nhau.
- **File của các bước bên trong giữ tên riêng** (`modes_<tag>_*`, `baseline_*`, `configs/<benchmark>_*`); bảng của chúng có trong `<tên bản>_<thời gian>_<benchmark>.md`.
- **Gói kết quả** mang tên bản nhưng chứa mọi kết quả nhẹ trên volume.
- **Kiểm tra tên trùng:** khi đọc kế hoạch, tên bản không được trùng tên bản của lần chạy khác, tên một luồng dữ liệu, hay tên việc / báo cáo loại khác (`run_all`, `scene_*`, `b0_<benchmark>`, `gsplat_<benchmark>`...).

## 3. Thay đổi trong repo

| Thành phần | File |
|---|---|
| Kế hoạch thực nghiệm | `src/training/experiment.py` (`--only`, `select_method`, `estimate.only`, `run_name`, kiểm tra tên trùng); `configs/experiments/main.yaml` (`estimate.only`) |
| Điều phối | `scripts/pod/run_experiment.sh` (`--only`, tên bản, kiểm tra môi trường và chỗ trống theo phần được chọn, bảng B0, dòng đối chiếu B0 trong báo cáo) |
| Huấn luyện, thư viện | `scripts/pipeline/train_gsplat_modes.sh` (`B0_CHECK`); `scripts/lib/config.sh` (`b0_missing_scenes`) |
| Đầu ra | `scripts/pod/pack_results.sh` (`--name`); `src/rendering/render.py`, `scripts/run_render.sh` (tên PLY); `src/evaluation/overview.py` (nhận báo cáo thực nghiệm theo dòng đầu) |
| Notebook | `app/notebooks/build_notebooks.py` (mới); `b0_inria.ipynb`, `gsplat_default.ipynb`, `gsplat_mcmc.ipynb` (mới); `runpod.ipynb` (sinh lại) |
| Giao diện dòng lệnh | `app/cli/main.py` (`--only` trong phần trợ giúp) |
| Kiểm thử | `tests/unit/test_notebooks.py` (mới); `test_experiment.py`, `test_benchmarks.py`, `test_overview.py`, `test_render.py`, `tests/test_dataset.py` |
| Tài liệu | `README.md`; `docs/USAGE_GUIDE.md` (mục 5.1.1 mới); `docs/design/experiments.md`, `architecture.md`, `training_modes.md`; `docs/PROJECT_STATUS.md` (D17); `docs/ENVIRONMENT_AUDIT.md`; `docs/data/capture_protocol.md`; `app/web/README.md` |

Không xóa file nào.

## 4. Kiểm thử

| Hạng mục | Môi trường | Kết quả |
|---|---|---|
| pytest | 5 môi trường: Python 3.13 và 3.10; numpy 2.5 / 2.2 / 1.26; OpenCV 4.11 / 5.0 | 452 test: 443 qua, 9 bỏ qua (cần PyTorch hoặc `simple_trainer.py` thật của gsplat) |
| pytest, nhóm PyTorch | Python 3.10.20, PyTorch 2.7.1 chạy trên CPU; trainer và `MCMCStrategy` thật của gsplat @6e8c837 | 123 qua |
| pytest, nhóm thư viện của env `gs-gsplat` | numpy 1.26.4, opencv-python-headless 4.11.0.86, imageio 2.37.0 | 60 qua |
| shellcheck 0.11.0, pyflakes, liên kết trong file `.md` | — | Sạch |
| Bộ sinh notebook | `build_notebooks.py --check` | 4 notebook trong repo đúng là bản sinh ra |
| Mô phỏng notebook | `/workspace` giả: GitHub thay bằng repo Git cục bộ, `ssh` / `curl` giả, không GPU, không conda | 86/86 kiểm tra đạt (19 kiểm tra mới cho ba notebook theo phương pháp) |

**So sánh đầu cuối** (trong lượt rà soát, bằng script thật, chỉ trainer, GPU, conda, tải dữ liệu và đóng gói là stub):
- ba thứ tự chạy (cả kế hoạch; `b0` → `default` → `mcmc`; `mcmc` → `default` → `b0` → cả kế hoạch) sinh cùng 36 lệnh train gsplat và 2 lệnh B0, giống từng byte;
- lần chạy sau không train thêm lượt nào;
- không dùng `--only` thì lời gọi của `main` và `tune_mcmc` không đổi so với commit trước, ngoài các tên mới.

**Rà soát độc lập** (ba lượt, người rà soát không tham gia viết code):
- không có lỗi chặn;
- các lỗi tìm được (1 lỗi vừa, 7 lỗi nhỏ và 12 ghi chú qua ba lượt) đã sửa kèm test, trừ các mục ở mục 5;
- chi tiết: [PROJECT_STATUS.md](../PROJECT_STATUS.md), mục "Kiểm thử trong sandbox".

**Chưa kiểm thử:**
- chạy trên pod với GPU;
- tải dữ liệu thật;
- thời gian và dung lượng thật của từng phương pháp.

## 5. Hạn chế và việc còn mở

- **Chưa chạy trên pod / GPU.** Mọi kết quả ở mục 4 là kiểm thử trong sandbox.
- **Ước lượng chưa đo:** thời gian và dung lượng của từng phương pháp là ước lượng, chưa có số đo.
- **PLY:**
  - PLY sau khi lọc Gaussian (`<lượt>/filtered/ply/`) chưa có tên theo bản;
  - PLY của B0 chưa có bước xuất;
  - tên cảnh tự chụp nên khác tên cảnh benchmark để file PLY tải về không trùng tên.
- **Kiểm chứng bản sửa:** các bản sửa sau lượt rà soát thứ ba mới được kiểm bằng test và mô phỏng notebook.
- **Quyết định còn mở từ D16:** có đưa gói `v2` vào vòng ghép (2 seed mới) hay không; cần chốt trước khi đọc kết quả `tune_mcmc`.

## 6. Bước tiếp theo

Theo mục "Việc tiếp theo" của [PROJECT_STATUS.md](../PROJECT_STATUS.md):

1. **P1–P4: dựng pod mới gắn network volume.**
   - Chạy ô 1–4 của một trong bốn notebook; ô 3 lấy code từ commit `c362b9e` trở đi.
   - Tiêu chí: ô 1 báo `ĐẠT`; việc `run_all` exit 0, smoke test PASSED.
2. **P5: thực nghiệm chính.** Cần duyệt chi phí GPU. Chọn một trong hai cách:
   - code tổng: `runpod.ipynb`, ô 7, `EXP = "main"`, việc `main`, khoảng 12 giờ GPU;
   - ba notebook theo thứ tự `b0_inria` → `gsplat_default` → `gsplat_mcmc`, khoảng 1 + 6 + 6 giờ GPU.

   Tiêu chí:
   - mọi bước OK trong báo cáo của từng bản;
   - B0 đạt ngưỡng so với số liệu Inria công bố;
   - `default` PASS so với B0 từng cảnh;
   - mỗi cảnh × chế độ đủ 3 seed;
   - ngân sách Gaussian EQUAL.
