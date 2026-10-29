# Bàn giao D18: biểu đồ tổng cho báo cáo so sánh

| | |
|---|---|
| Ngày | 29/10/2026 |
| Quyết định | D18 ([PROJECT_STATUS.md](../PROJECT_STATUS.md), mục "Quyết định đã chốt") |
| Commit | `4429786`, nhánh `main` |
| Trạng thái | Đã viết; đã kiểm thử trong sandbox trên báo cáo lập từ lượt giả, với matplotlib 3.10.5 như env `gs-gsplat`; **chưa vẽ từ kết quả thật** |

## 1. Mục tiêu và phạm vi

Mỗi báo cáo so sánh có vài ảnh tổng, đọc được ngay: mỗi phương pháp hơn hay kém mốc bao nhiêu khi gộp mọi cảnh và seed, và mốc `default` có khớp Inria không. Mọi thanh, mọi điểm có số cụ thể.

**Ngoài phạm vi:**
- không đổi bảng, chỉ số, điều kiện so sánh hay kế hoạch nào;
- không thêm phụ thuộc (matplotlib đã có trong env `gs-gsplat`); không chạy GPU;
- số từng cảnh, chi phí, dung lượng chỉ ở bảng. Bản đầu của D18 có 7 loại ảnh; đã bớt còn ảnh tổng theo yêu cầu "thêm vừa phải", "so sánh chênh lệch tổng tất cả, 1 cái của default 1 cái của mcmc các loại".

## 2. Thiết kế

### 2.1 Các ảnh

| Ảnh | Nội dung |
|---|---|
| `01_doi_chieu_inria.png` | `default` (gsplat) so với B0 tự chạy từng cảnh và với số Inria công bố: chênh lệch, vùng ngưỡng, ✓ / ✗. Chỉ Deep Blending đối chiếu được; Mip-NeRF 360 chạy 4/9 cảnh nên ghi SKIPPED ở đầu ảnh |
| `NN_so_voi_<mốc>.png` | Mỗi mốc một ảnh (`default`; `mcmc` khi có bảng so với `mcmc`). Mỗi phương pháp một thanh: chênh lệch trung bình trên mọi cặp (cảnh, seed) chung với mốc; chấm đen: từng cặp; số ghi: chênh lệch trung bình · số cặp tốt hơn / số cặp. Một báo cáo: đúng số ở bảng số cặp tốt hơn của nó. Nhiều benchmark: dòng `tổng` và một dòng (thanh nhạt) cho mỗi benchmark |
| `01_b0_so_voi_cong_bo.png` | Báo cáo chỉ B0: B0 từng cảnh, trung bình, số công bố với vạch ngưỡng |

Theo kế hoạch: `main` có `01_doi_chieu_inria.png` và `02_so_voi_default.png` (18 cặp: 6 cảnh × 3 seed); `tune_mcmc` có `01_so_voi_default.png` (17 phương pháp) và `02_so_voi_mcmc.png` (16 biến thể); `--only b0` có `01_b0_so_voi_cong_bo.png`.

### 2.2 Nơi vẽ

- `scripts/run_evaluate.sh`, `scripts/pipeline/train_gsplat_modes.sh`, `train_inria_b0.sh`: ngay sau khi lập bảng, vào `<báo cáo>_charts/`.
- `scripts/pod/run_experiment.sh`: một lần ở cuối cho cả lần chạy, từ mọi bảng `<tên bản>_<thời gian>_*.json`, vào `<tên bản>_<thời gian>_charts/`. Các bước bên trong (bảng từng benchmark, B0, gsplat) chạy với `CHARTS=0`.
- Hàm chung `make_charts <tên> <json>...` (`scripts/lib/config.sh`): python có matplotlib (env `gs-gsplat`, không có thì python3 của máy), `MPLBACKEND=Agg`. Không có matplotlib thì chỉ in `NOTE`; vẽ lỗi chỉ in `WARN`, bảng không đổi; `CHARTS=0` để tắt.
- Mỗi thư mục có `index.md`: cách đọc, bảng số kèm độ lệch chuẩn, lý do ảnh nào không vẽ. Báo cáo `.md` có mục "Biểu đồ" dẫn tới các ảnh. Notebook (ô "Bảng"; `runpod.ipynb` ô 7, `WHAT = "report"` / `"evaluate"`) hiện bảng kẻ sẵn rồi các ảnh.

### 2.3 Gộp và quy ước

- Gộp theo nhãn = tên thư mục lượt bỏ `_s<seed>`: `mcmc` của Deep Blending (`opacity_reg` 0,001) và của Mip-NeRF 360 cùng là `mcmc`. Mốc có phần `{…}` (lượt chạy thử) không gộp với mốc đúng protocol; hai báo cáo có chung lượt thì không gộp.
- Mọi khung chênh lệch "phải là tốt hơn" (trục LPIPS đảo chiều); số kiểu Việt (dấu phẩy, "−"), chênh lệch luôn có dấu; màu theo phương pháp; đầu ảnh ghi mốc, cảnh, seed, các mục điều kiện chưa đạt, file nguồn. Biểu đồ không tự kết luận.

## 3. Thay đổi trong repo

| Thành phần | File |
|---|---|
| Biểu đồ | `src/evaluation/charts.py` (mới) |
| Script | `scripts/lib/config.sh` (`make_charts`); `scripts/run_evaluate.sh`; `scripts/pod/run_experiment.sh`; `scripts/pipeline/train_gsplat_modes.sh`, `train_inria_b0.sh`; `scripts/pod/pack_results.sh` (ghi chú) |
| Tổng hợp, notebook | `src/evaluation/overview.py` (đường dẫn thư mục biểu đồ); `app/notebooks/build_notebooks.py` (`show_report`, `WHAT = "report"`) và 4 notebook sinh lại |
| Kiểm thử | `tests/unit/test_charts.py` (mới); `test_benchmarks.py`, `test_experiment.py`, `test_notebooks.py`, `test_overview.py` |
| Tài liệu | `README.md`; `docs/design/comparisons.md` (mục Biểu đồ), `experiments.md`, `architecture.md`; `docs/USAGE_GUIDE.md`; `docs/PROJECT_STATUS.md` (D18) |

Không xóa file nào. Repo không có ảnh, dữ liệu hay số giả: test tự lập báo cáo giả trong thư mục tạm khi chạy.

## 4. Kiểm thử

| Hạng mục | Kết quả |
|---|---|
| pytest, 5 môi trường như các lượt trước | 475 test: 457 qua, 18 bỏ qua (cần PyTorch, `simple_trainer.py` thật của gsplat hoặc matplotlib); môi trường có matplotlib 3.10.9: 466 qua, 9 bỏ qua |
| pytest với matplotlib 3.10.5 và các thư viện ghim của env `gs-gsplat` | 164 qua |
| pytest, nhóm PyTorch / nhóm thư viện của env `gs-gsplat` | 123 / 60 qua |
| shellcheck, pyflakes, liên kết trong file `.md`, `build_notebooks.py --check` | Sạch |
| Mô phỏng notebook (`/workspace` giả) | 89/89 kiểm tra đạt |
| Số trên ảnh | Tính lại độc lập trên báo cáo giả: khớp mục `wins` (một báo cáo) và trung bình, độ lệch chuẩn tính lại từ `pairs` (gộp) |
| Rà soát độc lập (một lượt) | Không có lỗi chặn; 1 lỗi vừa, 5 lỗi nhỏ đã sửa kèm test ([PROJECT_STATUS.md](../PROJECT_STATUS.md), mục "Kiểm thử trong sandbox") |

## 5. Hạn chế và việc còn mở

- **Chưa vẽ từ kết quả thật**, chưa chạy trên pod.
- **Dòng tổng** tính mọi cặp như nhau: Mip-NeRF 360 (4 cảnh) nặng gấp đôi Deep Blending (2 cảnh); đọc cùng các dòng của từng benchmark.
- **`mcmc_indoor` (P8):** ảnh tổng của `main` sẽ gộp cả `room`, cảnh dùng để chọn tham số; kết luận dùng bảng và biểu đồ không có `room` ([experiments.md](../design/experiments.md) mục 3.1).
- **Ảnh B0 so với số công bố** chỉ có ở lần chạy `--only b0`; khi chạy cả kế hoạch, điều kiện đó đọc ở bước `b0_deep_blending` (OK khi B0 đạt ngưỡng).
- **Kiểm chứng bản sửa:** các bản sửa sau lượt rà soát mới được kiểm bằng test, mô phỏng notebook và script của người rà soát, chưa rà soát lại.

## 6. Bước tiếp theo

Theo mục "Việc tiếp theo" của [PROJECT_STATUS.md](../PROJECT_STATUS.md): dựng pod (P1–P4), kiểm tải dữ liệu trước khi train (ô 7 `WHAT = "fetch"` cho từng benchmark, rồi `WHAT = "check"`), chạy `main` (P5, cần duyệt GPU). Xem `main_<thời gian>_charts/` ở ô 7 `WHAT = "report"`; trước khi chọn `mcmc_indoor` chỉ dùng phần kiểm điều kiện (`01_doi_chieu_inria.png`, bước B0, cột n / Seed của bảng).
