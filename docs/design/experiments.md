# Thực nghiệm: Deep Blending, Mip-NeRF 360, so sánh Default / MCMC

_Cập nhật: 19/10/2026_

Mỗi benchmark có một file cấu hình `configs/<benchmark>.yaml` (dữ liệu, nơi lưu kết quả, mục `benchmark`: protocol, số liệu công bố, ngưỡng) và một thư mục `experiments/<benchmark>/` chỉ chứa `README.md` (protocol, lệnh, kết quả đã ghi nhận). Dữ liệu và kết quả nằm trên volume, không nằm trong `experiments/`.

Bạn tự đưa các cảnh lên thư mục gốc trong cấu hình (`docs/USAGE_GUIDE.md` mục 2.1). Thiếu thì mọi lệnh dưới đây dừng trước khi train. Chạy benchmark nào, lúc nào là do bạn chọn.

Các lệnh dùng chung cho mọi benchmark, từ thư mục gốc của repo:

```bash
bash scripts/prepare_data.sh <benchmark>          # kiểm tra dữ liệu đã đưa lên: bố cục từng cảnh (không dùng GPU)
bash scripts/run_train.sh <benchmark> b0          # B0: 3DGS gốc của Inria (commit 54c035f), như full_eval.py
bash scripts/run_train.sh <benchmark> gsplat      # gsplat 6e8c837: default, rồi mcmc cùng số Gaussian; bảng so sánh
bash scripts/run_evaluate.sh <benchmark>          # chỉ lập lại bảng so sánh từ kết quả đã có (không dùng GPU)
```

`<benchmark>` là `deep_blending` hoặc `mipnerf360`. Phần thực thi nằm ở `scripts/pipeline/{prepare_benchmark,train_inria_b0,train_gsplat_modes}.sh`; hàm dùng chung ở `scripts/lib/benchmark.sh`.

**Biến tùy chọn:**
- `SEEDS="42 43 44"`: lặp nhiều seed, cần trước khi nói về "cải thiện".
- `QUICK=1`: chạy thử 3.000 vòng, kết quả vào thư mục `gsplat_quick` riêng, không bao giờ trộn với lượt chuẩn.
- `SCENES="room counter"`: chỉ chạy các cảnh này.
- `MCMC_SET=...`: thay tham số riêng của mcmc (mặc định lấy `benchmark.gsplat.mcmc_set` trong `configs/<benchmark>.yaml`).

Chạy trên pod, sau khi môi trường đã sẵn sàng (`bash scripts/pod/run_all.sh --background`, chỉ cài môi trường). Chạy nền: `bash scripts/pod/job.sh start <tên> -- bash scripts/run_train.sh <benchmark> gsplat`, hoặc ô 7 của `app/notebooks/runpod.ipynb`. Mỗi lúc chỉ một việc dùng GPU. Chạy `b0` trước `gsplat`: bước so sánh cuối của `gsplat` (30.000 vòng) đối chiếu `default` với B0 từng cảnh, chưa có B0 thì phần đó là INCOMPLETE và việc kết thúc với exit 1 (các lượt train vẫn được giữ).

Mỗi lần chạy, phần `benchmark` của `configs/<benchmark>.yaml` được ghi ra một bản JSON trong `$WS/outputs/reports/configs/` và script đọc bản đó (env `gs-inria` không có PyYAML). Bản JSON ở lại để đối chiếu sau này.

## Kết quả lưu ở đâu

Thư mục dữ liệu và kết quả lấy từ `configs/<benchmark>.yaml` (`docs/design/architecture.md` mục 2 và 5):

| | Thư mục |
|---|---|
| Dữ liệu | `$WS/data/benchmarks/<benchmark>/<cảnh>/` |
| B0 Inria | `$WS/outputs/benchmarks/<benchmark>/inria_b0/<cảnh>/` |
| gsplat | `$WS/outputs/benchmarks/<benchmark>/gsplat/<cảnh>/<chế_độ>_s<seed>/` |
| gsplat chạy thử (`QUICK=1`) | `$WS/outputs/benchmarks/<benchmark>/gsplat_quick/<cảnh>/<chế_độ>_s<seed>/` |
| Báo cáo, log | `$WS/outputs/reports/{baseline_<tag>_summary,modes_<tag>,compare_<tag>}_<thời_gian>.*`, `$WS/outputs/logs/` (`benchmark.tag`: `db`, `mipnerf360`) |

Không ghi đè lượt gsplat: lượt đã có kết quả OK với cùng cấu hình thì được bỏ qua; khác cấu hình thì báo lỗi. Khi được phép thay, thư mục cũ được đổi tên `.prev_<thời_gian>`. B0 thì khác: cảnh đã có `results.json` (đủ chỉ số `ours_30000`) thì bỏ qua mà không so cấu hình; `FORCE=1` chạy lại, lượt cũ được đổi tên `<cảnh>.prev_<thời_gian>` (không ghi đè).

## So sánh DefaultStrategy với MCMC

`scripts/run_evaluate.sh` (`docs/design/comparisons.md`) và bước cuối của `scripts/run_train.sh <benchmark> gsplat` lập cùng một bảng.

**Chỉ gọi là "công bằng" khi:**
- cùng cảnh và cùng mô hình COLMAP;
- cùng tập train/test;
- cùng protocol;
- cùng số Gaussian (MCMC lấy `cap_max` bằng số Gaussian cuối của `default`).

Bảng tự kiểm tra các điều kiện này.

**Chỉ gọi là "cải thiện" khi** có ít nhất 3 seed và mọi điều kiện đều đạt.

## Đã chạy / mới cấu hình

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| B0 Inria trên Deep Blending | Repo trước đã chạy trên pod ngày 17/10/2026, đạt ngưỡng so với số liệu Inria. **Repo này chưa chạy** | `docs/PROJECT_STATUS.md` (lịch sử), `docs/ENVIRONMENT_AUDIT.md` mục 7 |
| gsplat `default` / `mcmc` trên Deep Blending, seed 42 | Repo trước đã chạy trên pod ngày 17/10/2026. **Repo này chưa chạy** | như trên |
| Deep Blending 3 seed | Mới cấu hình (`SEEDS="42 43 44"`) | — |
| Mip-NeRF 360 (B0 và gsplat) | **Đã cấu hình, chưa chạy; chưa có dữ liệu trên volume** | `experiments/mipnerf360/README.md` |
| Lệnh huấn luyện mà repo này sinh ra cho Deep Blending | Kiểm thử bằng stub: giống lệnh repo trước đã chạy ngày 17/10/2026, chỉ khác đường dẫn | `tests/unit/test_benchmarks.py` |
