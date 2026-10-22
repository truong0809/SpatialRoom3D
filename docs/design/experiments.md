# Thực nghiệm: Deep Blending, Mip-NeRF 360, so sánh các chế độ huấn luyện

_Cập nhật: 19/10/2026_

Mỗi benchmark có một file cấu hình `configs/<benchmark>.yaml` (dữ liệu, nguồn tải, nơi lưu kết quả, mục `benchmark`: protocol, số liệu công bố, ngưỡng) và một thư mục `experiments/<benchmark>/` chỉ chứa `README.md` (protocol, lệnh, kết quả đã ghi nhận). Dữ liệu và kết quả nằm trên volume, không nằm trong `experiments/`.

**Dữ liệu benchmark** có hai cách đưa vào thư mục gốc trong cấu hình:
- Tải thẳng về pod: `bash scripts/pod/fetch_data.sh <benchmark> [cảnh ...]` (`src/datasets/fetch.py`, theo mục `download` của `configs/<benchmark>.yaml`).
  - Chỉ tải cảnh còn thiếu, chỉ giải nén các thư mục cần dùng, xóa file zip sau khi giải nén.
  - Deep Blending: kiểm SHA-256 của `tandt_db.zip` với giá trị đã đo ngày 17/10/2026. Mip-NeRF 360 không có SHA-256 công bố: script ghi lại SHA-256 của file đã tải, CRC của từng file được kiểm khi giải nén.
  - Không sửa, không ghi đè dữ liệu đã có. Nguồn, kích thước, SHA-256 và số file mỗi cảnh ghi vào `<root>/.sr3d_fetch.json` ngay khi cảnh đó vào chỗ, và vào `outputs/reports/fetch_<benchmark>_<thời_gian>.json` (cả khi lỗi, kèm lỗi).
  - curl tự thử lại khi lỗi mạng, và khi tải dưới 100 KB/s suốt 5 phút (kết nối treo).
- Tự đưa lên (`docs/USAGE_GUIDE.md` mục 2.1).

Lệnh train không tự tải: thiếu dữ liệu thì dừng trước khi train và chỉ lệnh tải. Chạy benchmark nào, lúc nào là do bạn chọn.

## 1. Lệnh cho một benchmark

Từ thư mục gốc của repo:

```bash
bash scripts/pod/fetch_data.sh <benchmark> [cảnh ...]  # tải dữ liệu còn thiếu (không dùng GPU)
bash scripts/prepare_data.sh <benchmark>          # kiểm tra dữ liệu: bố cục từng cảnh (không dùng GPU)
bash scripts/run_train.sh <benchmark> b0          # B0: 3DGS gốc của Inria (commit 54c035f), như full_eval.py
bash scripts/run_train.sh <benchmark> gsplat      # gsplat 6e8c837: default, rồi các chế độ khác cùng số Gaussian; bảng so sánh
bash scripts/run_evaluate.sh <benchmark>          # chỉ lập lại bảng so sánh từ kết quả đã có (không dùng GPU)
```

`<benchmark>` là `deep_blending` hoặc `mipnerf360`. Phần thực thi nằm ở `scripts/pipeline/{prepare_benchmark,train_inria_b0,train_gsplat_modes}.sh`; hàm dùng chung ở `scripts/lib/benchmark.sh`.

**Biến tùy chọn:**
- `SEEDS="42 43 44"`: lặp nhiều seed, cần trước khi nói về "cải thiện".
- `MODES="default mcmc <chế độ tự thêm>"`: các chế độ (mặc định `default mcmc`; chế độ tự thêm trong `configs/base.yaml`, `docs/design/training_modes.md` mục 3). `default` luôn chạy trước; mọi chế độ khác lấy `cap_max` từ lượt `default` cùng cảnh, cùng seed.
- `QUICK=1`: chạy thử 3.000 vòng, kết quả vào thư mục `gsplat_quick` riêng, không bao giờ trộn với lượt chuẩn.
- `SCENES="room counter"`: chỉ chạy các cảnh này. Khi chỉ là một phần các cảnh của benchmark, bảng không đối chiếu được với trung bình công bố (ghi SKIPPED), và chỉ đối chiếu B0 khi mọi cảnh được chọn đều có lượt B0.
- `MCMC_SET=...`: thay tham số riêng của mọi chế độ preset `mcmc` (mặc định lấy `benchmark.gsplat.mcmc_set` trong `configs/<benchmark>.yaml`; Deep Blending: `opacity_reg=0.001`).

Chạy trên pod, sau khi môi trường đã sẵn sàng (`bash scripts/pod/run_all.sh --background`, chỉ cài môi trường). Chạy nền: `bash scripts/pod/job.sh start <tên> -- bash scripts/run_train.sh <benchmark> gsplat`, hoặc ô 7 của `app/notebooks/runpod.ipynb`. Mỗi lúc chỉ một việc dùng GPU. Chạy `b0` trước `gsplat`: bước so sánh cuối của `gsplat` (30.000 vòng, mọi cảnh) đối chiếu `default` với B0 từng cảnh, chưa có B0 thì phần đó là INCOMPLETE và việc kết thúc với exit 1 (các lượt train vẫn được giữ).

Mỗi lần chạy, phần `benchmark` của `configs/<benchmark>.yaml` được ghi ra một bản JSON trong `$WS/outputs/reports/configs/` và script đọc bản đó (env `gs-inria` không có PyYAML). Bản JSON ở lại để đối chiếu sau này.

## 2. Thực nghiệm chính: `default` / `mcmc`

Kế hoạch: `configs/experiments/main.yaml` (kiểm tra bởi `src/training/experiment.py`). Chạy: `scripts/pod/run_experiment.sh`. **Trạng thái: đã viết, đã kiểm thử bằng stub và mô phỏng notebook; đã rà soát độc lập một lượt, lỗi tìm được đã sửa kèm test; chưa chạy trên pod.**

### 2.1 Câu hỏi

- `mcmc` (3DGS-MCMC) có tốt hơn `default` ở cùng số Gaussian trên cảnh trong nhà không, với 3 seed trên 6 cảnh? Lượt 17/10/2026 của repo trước mới có một seed (`docs/design/training_modes.md` mục 8).
- Kết quả không cải thiện vẫn được báo cáo đầy đủ (`docs/PROJECT_PLAN.md`, rủi ro R13).

### 2.2 Thiết kế

| Mục | Giá trị |
|---|---|
| Chế độ | `default` (baseline), `mcmc` (`configs/base.yaml`, `docs/design/training_modes.md` mục 3) |
| Dữ liệu | 6 cảnh trong nhà: Deep Blending `drjohnson`, `playroom` (ảnh gốc); Mip-NeRF 360 `room`, `counter`, `kitchen`, `bonsai` (`images_2`, quy ước của benchmark) |
| Seed | 42, 43, 44 cho mọi chế độ và mọi cảnh |
| Protocol | Chung cho mọi chế độ (`docs/design/training_modes.md` mục 4.1): 30.000 vòng, mỗi ảnh thứ 8 làm test, LPIPS-VGG, khởi tạo từ điểm SfM |
| Ngân sách Gaussian | `mcmc`: `cap_max` = số Gaussian cuối của lượt `default` cùng cảnh, cùng seed |
| Quy ước `mcmc` của benchmark | Deep Blending: `opacity_reg` 0,001; Mip-NeRF 360: giá trị mặc định của preset |
| Kiểm chứng baseline | B0 Inria trên Deep Blending (từng cảnh) và số Inria công bố (trung bình 2 cảnh). Mip-NeRF 360: chỉ 4/9 cảnh nên không đối chiếu được với trung bình công bố (ghi SKIPPED); không chạy B0 |
| Chỉ số | PSNR, SSIM, LPIPS (VGG) trên ảnh test; kèm số Gaussian, thời gian train, VRAM đỉnh, thời gian render mỗi ảnh, dung lượng PLY |

Thêm một chế độ vào thực nghiệm: khai báo chế độ trong `configs/base.yaml` (`docs/design/training_modes.md` mục 3), thêm tên vào `modes` của kế hoạch rồi chạy lại. Chỉ các lượt của chế độ đó được train; lượt `default` / `mcmc` đã xong được dùng lại. `also_against` (vd. `[mcmc]`) thêm một bảng so chế độ mới với `mcmc`.

### 2.3 Chạy

```bash
VOLUME_GB=100 bash scripts/pod/run_experiment.sh main --background   # một việc nền: exp_main
bash scripts/pod/run_experiment.sh main --plan                       # chỉ in kế hoạch, ước lượng, dữ liệu đã có
```

Hoặc ô 7 của notebook với `WHAT = "experiment"`. Gửi được ngay sau ô 4: việc xếp hàng và tự chạy khi cài môi trường xong.

Các bước, trong một việc nền (giữ khóa GPU suốt lượt):
1. `env_check`: GPU, env `gs-gsplat` và `gs-inria`; `simple_trainer.py` thật nhận đúng tham số của mọi chế độ trong kế hoạch (`scripts/setup/smoke_test_gsplat.py --modes`, vài giây, không train).
2. `space_check`: đo dung lượng đang dùng (mục 2.4).
3. `fetch_<benchmark>`: tải dữ liệu còn thiếu (`scripts/pod/fetch_data.sh`). Tải lỗi thì bỏ qua các bước của benchmark đó, benchmark khác vẫn chạy.
4. `b0_deep_blending`: B0 Inria trên 2 cảnh.
5. `gsplat_<benchmark>_s<seed>`: xen kẽ theo seed: đủ seed 42 trên cả 6 cảnh, rồi 43, rồi 44. Trên mỗi cảnh: `default` trước, rồi `mcmc`.
6. `report_<benchmark>`: bảng so với `default` (và `report_<benchmark>_vs_<mốc>` cho mỗi mốc trong `also_against`).
7. `pack`: gói kết quả nhẹ để tải về.

Luôn chạy đầy đủ, không có chế độ chạy thử. Biến môi trường của bạn (`QUICK`, `SEEDS`, `MODES`, `MCMC_SET`, `OUT_ROOT`...) không đổi được kế hoạch. Chạy lại an toàn: dữ liệu đã có không tải lại, lượt đã xong tự bỏ qua; lượt dở dang (pod tắt giữa chừng) được chạy lại từ đầu.

### 2.4 Thời gian và dung lượng

Ước lượng trên RTX 4090, chưa phải số đo (căn cứ ghi trong `configs/experiments/main.yaml`):
- Khoảng 12 giờ GPU.
  - Deep Blending từ số đo ngày 17/10/2026 của repo trước: B0 49,5 phút; `default` + `mcmc` 74 phút mỗi seed.
  - Mip-NeRF 360 chưa đo.
- Thêm khoảng 56 GB trên volume, giữ mọi file kết quả của mọi seed (checkpoint, PLY, ảnh render, video):
  - Deep Blending khoảng 19 GB;
  - Mip-NeRF 360 khoảng 22–30 GB (số Gaussian chưa đo);
  - B0 và dữ liệu khoảng 7 GB.
- Môi trường conda cũng nằm trên volume (khoảng 15–20 GB, chưa đo). Theo ước lượng, volume 100 GB đủ. Thêm chế độ vào kế hoạch thì cần thêm chỗ: tăng volume trên trang Runpod (tăng được, không giảm được) rồi đặt `VOLUME_GB` bằng dung lượng mới.

`df` trên network volume của Runpod không đọc được quota, nên script tự đo bằng `du` và so với `VOLUME_GB`. Chỗ cần cho một seed là tổng các lượt gsplat chưa xong của seed đó (`python -m src.training.experiment --space main`):
- lượt đã xong (`summary.json` OK) không cần thêm gì, kể cả khi chạy lại sau khi pod dừng giữa chừng;
- lượt chưa xong tính bằng dung lượng thật lớn nhất của lượt cùng cảnh, cùng chế độ đã xong ở seed khác;
- chưa có lượt như vậy thì tính bằng ước lượng 56 GB của kế hoạch chia đều cho 36 lượt (khoảng 1,6 GB mỗi lượt, gồm cả phần dữ liệu, B0, ảnh PNG thu nhỏ chỉ ghi một lần).

Kiểm tra:
- **Trước khi chạy:** seed đầu tiên còn lượt chưa xong không đủ chỗ thì dừng, chưa tải, chưa train gì. Phần còn lại của kế hoạch không đủ chỗ thì chỉ cảnh báo.
- **Trước mỗi seed:** không đủ chỗ cho các lượt chưa xong của seed đó (chỉ tính benchmark sẽ train) thì dừng train; các lượt đã xong vẫn giữ, bảng so sánh vẫn được lập. Tăng volume, đặt `VOLUME_GB` rồi chạy lại để làm nốt.
- File zip tải về chỉ nằm tạm trên volume trong lúc giải nén, không tính trong kiểm tra (lúc tải, volume còn trống nhiều).
- `SPACE_CHECK=0`: bỏ qua cả hai bước.

### 2.5 Kết quả và cách đọc

| File (trong `$WS/outputs/reports/`) | Nội dung |
|---|---|
| `experiment_main_<thời_gian>.md` | Bảng các bước, rồi với từng benchmark: bảng so với `default` (và với mỗi mốc trong `also_against`) |
| `experiment_main_<thời_gian>.tsv` | Các bước (đọc bởi `scripts/pod/results.sh`) |
| `experiment_main_<thời_gian>_<benchmark>[_vs_<mốc>].{md,json}` | Từng bảng so sánh (`src.evaluation.compare_modes`) |
| `configs/experiment_main_<thời_gian>.json` | Bản chụp kế hoạch đã dùng, kèm SHA-256 của file kế hoạch |
| `fetch_<benchmark>_<thời_gian>.json` | Bản ghi tải dữ liệu |

Các lượt train nằm ở `$WS/outputs/benchmarks/<benchmark>/gsplat/<cảnh>/<chế_độ>_s<seed>/` (mục 3).

Chỉ gọi là "cải thiện" theo mục 4, và thêm:
- chênh lệch so với `default` cùng dấu trên phần lớn cảnh, trên cùng 3 seed;
- chênh lệch lớn hơn độ dao động giữa các seed (độ lệch chuẩn trong bảng).

Nếu chỉ một benchmark cải thiện, hoặc chỉ một chỉ số, thì ghi đúng như vậy.

## 3. Kết quả lưu ở đâu

Thư mục dữ liệu và kết quả lấy từ `configs/<benchmark>.yaml` (`docs/design/architecture.md` mục 2 và 5):

| | Thư mục |
|---|---|
| Dữ liệu | `$WS/data/benchmarks/<benchmark>/<cảnh>/` (file zip tải về nằm tạm ở `$WS/data/downloads/`) |
| B0 Inria | `$WS/outputs/benchmarks/<benchmark>/inria_b0/<cảnh>/` |
| gsplat | `$WS/outputs/benchmarks/<benchmark>/gsplat/<cảnh>/<chế_độ>_s<seed>/` |
| gsplat chạy thử (`QUICK=1`) | `$WS/outputs/benchmarks/<benchmark>/gsplat_quick/<cảnh>/<chế_độ>_s<seed>/` |
| Báo cáo, log | `$WS/outputs/reports/{baseline_<tag>_summary,modes_<tag>,compare_<tag>,experiment_<tên>}_<thời_gian>.*`, `$WS/outputs/logs/` (`benchmark.tag`: `db`, `mipnerf360`) |

Không ghi đè lượt gsplat: lượt đã có kết quả OK với cùng cấu hình thì được bỏ qua; khác cấu hình thì báo lỗi. Khi được phép thay, thư mục cũ được đổi tên `.prev_<thời_gian>`. B0 thì khác: cảnh đã có `results.json` (đủ chỉ số `ours_30000`) thì bỏ qua mà không so cấu hình; `FORCE=1` chạy lại, lượt cũ được đổi tên `<cảnh>.prev_<thời_gian>` (không ghi đè).

## 4. So sánh các chế độ

`scripts/run_evaluate.sh` (`docs/design/comparisons.md`) và bước cuối của `scripts/run_train.sh <benchmark> gsplat` lập cùng một bảng.

**Chỉ gọi là "công bằng" khi:**
- cùng cảnh và cùng mô hình COLMAP;
- cùng tập train/test;
- cùng protocol;
- cùng số Gaussian (`mcmc` lấy `cap_max` bằng số Gaussian cuối của `default`).

Bảng tự kiểm tra các điều kiện này.

**Chỉ gọi là "cải thiện" khi** có ít nhất 3 seed và mọi điều kiện đều đạt (`docs/design/training_modes.md` mục 4.3).

## 5. Đã chạy / mới cấu hình

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| B0 Inria trên Deep Blending | Repo trước đã chạy trên pod ngày 17/10/2026, đạt ngưỡng so với số liệu Inria. **Repo này chưa chạy** | `docs/PROJECT_STATUS.md` (lịch sử), `docs/ENVIRONMENT_AUDIT.md` mục 7 |
| gsplat `default` / `mcmc` trên Deep Blending, seed 42 | Repo trước đã chạy trên pod ngày 17/10/2026. **Repo này chưa chạy** | như trên |
| Thực nghiệm chính (mục 2): `default` / `mcmc`, 3 seed, Deep Blending + 4 cảnh Mip-NeRF 360 | **Đã viết, chưa chạy trên pod** | Kiểm thử bằng stub: `tests/unit/test_experiment.py`, `test_fetch.py`, `test_benchmarks.py`, `test_compare_modes.py`; mô phỏng notebook và rà soát độc lập (`docs/PROJECT_STATUS.md`) |
| Tải dữ liệu benchmark trên pod | Đã viết; kiểm thử với curl giả và file zip tổng hợp; **chưa tải thật** (sandbox không ra được Internet) | `tests/unit/test_fetch.py` |
| Mip-NeRF 360 (B0 và gsplat) | **Đã cấu hình, chưa chạy; chưa có dữ liệu trên volume** | `experiments/mipnerf360/README.md` |
| Lệnh huấn luyện mà repo này sinh ra cho Deep Blending | Kiểm thử bằng stub: `default` / `mcmc` giống lệnh repo trước đã chạy ngày 17/10/2026, chỉ khác đường dẫn | `tests/unit/test_benchmarks.py` |
