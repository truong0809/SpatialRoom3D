# Thực nghiệm: Deep Blending, Mip-NeRF 360, so sánh các chế độ huấn luyện, dò tham số mcmc

_Cập nhật: 24/10/2026_

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
- `MCMC_SET=...`: một cặp `<khóa>=<giá_trị>` thêm cho mọi chế độ preset `mcmc` (mặc định lấy `benchmark.gsplat.mcmc_set` trong `configs/<benchmark>.yaml`; Deep Blending: `opacity_reg=0.001`), trừ chế độ đã tự đặt khóa đó trong `configs/base.yaml`: chế độ giữ giá trị của nó (mục 3.2).
- `VARIANTS="mcmc:or0001:opacity_reg=0.001 ..."`: biến thể tham số của một chế độ (mục 3.2), thường do `scripts/pod/run_experiment.sh` đặt theo kế hoạch. Mỗi biến thể chạy sau các chế độ, cùng ngân sách Gaussian, vào thư mục `<chế_độ>_<tag>_s<seed>`; tham số của nó thêm sau `MCMC_SET`.

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

Thêm một chế độ vào thực nghiệm: khai báo chế độ trong `configs/base.yaml` (`docs/design/training_modes.md` mục 3), thêm tên vào `modes` của kế hoạch rồi chạy lại. Chỉ các lượt của chế độ đó được train; lượt `default` / `mcmc` đã xong được dùng lại. `also_against` (vd. `[mcmc]`) thêm một bảng so chế độ mới với `mcmc`. Các khóa khác của kế hoạch (`variants`, `post_filter`) ở mục 3.2.

### 2.3 Chạy

```bash
VOLUME_GB=100 bash scripts/pod/run_experiment.sh main --background   # một việc nền: exp_main
bash scripts/pod/run_experiment.sh main --plan                       # chỉ in kế hoạch, ước lượng, dữ liệu đã có
```

Hoặc ô 7 của notebook với `WHAT = "experiment"`. Gửi được ngay sau ô 4: việc xếp hàng và tự chạy khi cài môi trường xong.

Các bước, trong một việc nền (giữ khóa GPU suốt lượt):
1. `env_check`: GPU, env `gs-gsplat` và `gs-inria`; `simple_trainer.py` thật nhận đúng tham số của mọi chế độ và biến thể trong kế hoạch (`scripts/setup/smoke_test_gsplat.py --modes ... --variants ...`, vài giây, không train).
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

Các lượt train nằm ở `$WS/outputs/benchmarks/<benchmark>/gsplat/<cảnh>/<chế_độ>_s<seed>/` (mục 5).

Ngoài PSNR / SSIM / LPIPS, mỗi bảng so sánh có số cặp (cảnh, seed) và số cảnh tốt hơn, chất lượng theo dung lượng (byte mỗi Gaussian, bản đã lọc) và cấu trúc mô hình đọc từ PLY (tỉ lệ Gaussian gần trong suốt / rất to); ảnh so sánh và video xếp chồng của các lượt: `scripts/run_panels.sh` (`docs/design/comparisons.md`). Cột `Render (ms/ảnh)` gồm cả thời gian khởi động của lượt chấm, không phải tốc độ render. `main.yaml` chưa có `post_filter`, nên `main` không tự lọc. Bảng sau khi lọc và các dòng bản đã lọc của bảng chất lượng theo dung lượng chỉ có cho các lượt đã lọc ở kế hoạch khác: vd. `default`, `mcmc` của `room` sau `tune_mcmc` và vòng ghép, không có cho `mcmc_indoor` hay 5 cảnh còn lại; các dòng đó không dùng để so các chế độ. Thêm `post_filter` vào kế hoạch thì mỗi lượt tốn thêm khoảng 4 phút GPU để lọc và chấm lại (ước lượng như `tune_mcmc`).

Chỉ gọi là "cải thiện" theo mục 6, và thêm:
- chênh lệch so với `default` cùng dấu trên phần lớn cảnh, trên cùng 3 seed (cột "Cảnh tốt hơn" của bảng số cặp thắng);
- chênh lệch lớn hơn độ dao động giữa các seed (độ lệch chuẩn trong bảng).

Nếu chỉ một benchmark cải thiện, hoặc chỉ một chỉ số, thì ghi đúng như vậy.

## 3. Dò tham số `mcmc`: `tune_mcmc`

Kế hoạch: `configs/experiments/tune_mcmc.yaml`. Chạy bằng cùng script với thực nghiệm chính. **Trạng thái: đã viết, đã kiểm thử bằng stub trong sandbox; chưa chạy trên pod, chưa có số liệu.** Bốn biến thể lấy từ notebook v2 (mục 3.7): đã viết, đã kiểm thử trên CPU (kể cả `simple_trainer.py` thật của gsplat @6e8c837 ở chế độ chỉ phân giải tham số); **chưa chạy một vòng train nào trên GPU**.

### 3.1 Câu hỏi và cách dùng kết quả

- Với cảnh trong nhà, tham số nào của `mcmc` đáng đổi: độ mạnh của regularization opacity / scale, nhiễu của MCMC, ngưỡng opacity mà MCMC dời Gaussian, nền ngẫu nhiên khi train, loss độ sâu theo điểm SfM, dời Gaussian đến vòng cuối, bậc SH thấp hơn (file nhẹ hơn)?
- Ba thành phần mà notebook v2 của nhóm thêm vào `mcmc` (bilateral grid, Kill-then-Relocate, depth prior) có ích không khi đo theo protocol chung, từng thành phần một; và cả gói v2 so với `mcmc` (mục 3.7)?
- Đây là bước **sàng lọc**: một cảnh (`room`), một seed (42). Bảng ghi `seeds=FEW` (hoặc `seeds=DIFF` khi `room` đã có 3 seed của thực nghiệm chính) và không đủ để nói "cải thiện". Một biến thể chỉ được coi là đáng giữ khi hơn `mcmc` rõ ràng hơn độ dao động thường gặp giữa các seed; độ dao động này trên `room` chỉ đo được khi có 3 seed của thực nghiệm chính (chạy `main` trước thì đã có: độ lệch chuẩn của `mcmc` trên `room`), nên kết luận của bước này là giả thuyết, được kiểm lại ở vòng ghép (mục 3.6).
- Sau khi có kết quả: chọn ứng viên, ghép thành vài cấu hình rồi chạy 2 seed mới (43, 44) trên `room` (vòng ghép, mục 3.6). Cấu hình được chọn thành chế độ mới `mcmc_indoor` trong `configs/base.yaml` (cùng preset `mcmc`, cùng `cap_max: {policy: match_baseline}`), rồi được đưa vào `main.yaml` với `also_against: [mcmc]` để so trên 6 cảnh × 3 seed.
  - Thư mục của `mcmc_indoor` là `mcmc_indoor_s<seed>`, khác thư mục của cấu hình trong vòng ghép, nên `main` train lại 3 lượt của nó trên `room` dù cùng cấu hình: thêm một lần lặp trên cảnh dùng để chọn, không đổi kết luận.
  - Trên một benchmark, `mcmc_indoor` có thể trùng cấu hình với `mcmc` (vd. chọn `opacity_reg` 0,001, đúng giá trị Deep Blending đã dùng cho `mcmc`): kế hoạch in cảnh báo, cả hai vẫn train và có dòng riêng trong bảng; chênh lệch ở đó chỉ là dao động giữa hai lần train.
- **Thiên lệch khi chọn tham số** (quy tắc ở `docs/design/training_modes.md` mục 4.2): tham số được chọn bằng ảnh test của `room` (cả lượt dò lẫn vòng ghép), nên `room` là cảnh dùng để dò. Kết luận về `mcmc_indoor` dựa trên 5 cảnh còn lại của thực nghiệm chính (Deep Blending 2 cảnh, `counter`, `kitchen`, `bonsai`) và cảnh tự quay. Bảng tự động của thực nghiệm chính vẫn có `room` (cả trong trung bình); bảng không có `room` lập lại từ kết quả đã có, không dùng GPU:
  ```bash
  BASELINE=mcmc MODES="mcmc mcmc_indoor" bash scripts/run_evaluate.sh deep_blending
  BASELINE=mcmc MODES="mcmc mcmc_indoor" SCENES="counter kitchen bonsai" bash scripts/run_evaluate.sh mipnerf360
  ```
  Khi báo cáo, dòng của `room` vẫn ghi nhưng nói rõ là cảnh đã dùng để chọn tham số. Cách chặt hơn (chưa làm): dò trên một phần ảnh train của `room` tách ra làm tập kiểm định.
- 12 biến thể tham số là tinh chỉnh tham số của một phương pháp có sẵn (3DGS-MCMC trên gsplat) cộng một bước hậu xử lý, không phải thuật toán mới. Ba thành phần của notebook v2 là thay đổi phương pháp ở mức tích hợp: bilateral grid và loss Pearson với độ sâu đơn ảnh là kỹ thuật đã công bố (mục 3.7), Kill-then-Relocate là heuristic của nhóm, không có paper. Không gọi phần nào là thuật toán mới.

### 3.2 Thiết kế

| Mục | Giá trị |
|---|---|
| Chế độ | `default`, `mcmc` |
| Biến thể của `mcmc` (`variants`) | 16 biến thể, theo thứ tự chạy. Ba biến thể thêm từng thành phần của notebook v2: `mono` (`sr3d.mono`), `kr` (`sr3d.kr`), `bilagrid` (`use_bilateral_grid`); rồi gói `v2` (mục 3.7). Rồi 12 biến thể tham số, mỗi biến thể đổi đúng một tham số của `simple_trainer.py` so với preset `mcmc` của gsplat @6e8c837 (`opacity_reg` 0,01, `scale_reg` 0,01, `strategy.noise_lr` 5e5, `strategy.min_opacity` 0,005, `random_bkgd` tắt, `depth_loss` tắt, `strategy.refine_stop_iter` 25.000, `sh_degree` 3): `or0001` (`opacity_reg` 0,001), `or0005` (0,005), `or005` (0,05), `sr0001` (`scale_reg` 0,001), `sr005` (0,05), `nlr250k` (`strategy.noise_lr` 2,5e5), `nlr1m` (1e6), `mo002` (`strategy.min_opacity` 0,02), `rbg` (`random_bkgd`), `depth` (`depth_loss`, `depth_lambda` mặc định 0,01), `rs30k` (`strategy.refine_stop_iter` 30.000), `sh2` (`sh_degree` 2). Bốn biến thể của notebook chạy trước để code mới (bước `mono`, KR, loss của prior trên GPU) chạy sớm: lỗi thì thấy sau khoảng một giờ, không phải sau năm giờ; thứ tự chỉ đổi lịch chạy, không đổi kết quả |
| Dữ liệu | Mip-NeRF 360 `room` (`images_2`) |
| Seed | 42 |
| Protocol, ngân sách Gaussian | như thực nghiệm chính (mục 2.2): mọi biến thể lấy `cap_max` từ lượt `default` cùng cảnh, cùng seed |
| Bảng so sánh | với `default`, và thêm một bảng so với `mcmc` (`also_against: [mcmc]`) |
| Lọc sau khi train (`post_filter`) | mọi lượt: `min_opacity` 0,02, `max_scale` 0,1 (mục 4); bảng trước / sau khi lọc |

`or005` và `mo002` nhắm vào lớp mờ và floater (giả thuyết, chưa có số liệu). `or005` phạt opacity mạnh hơn preset. Với `mo002`, MCMC dời cả Gaussian có opacity đến 0,02 (bằng ngưỡng lọc ở mục 4) sang chỗ có Gaussian đậm ngay trong lúc train, thay vì để bộ lọc xóa sau khi train; trong gsplat @6e8c837, `strategy.min_opacity` cũng là opacity nhỏ nhất của Gaussian vừa được dời hoặc thêm (`relocate`, `sample_add` trong `gsplat/strategy/ops.py`).

`rs30k` và `sh2` (giả thuyết, chưa có số liệu):
- `rs30k`: mỗi 100 vòng, sau vòng 500 và trước `strategy.refine_stop_iter` (mặc định 25.000), MCMC của gsplat dời các Gaussian gần trong suốt sang chỗ có Gaussian đậm, và thêm Gaussian cho tới khi đạt `cap_max` (việc thêm thường dừng sau vài nghìn vòng). Notebook demo của nhóm đặt 35.000 khi chạy 40.000 vòng (`docs/research/demo_review.md`); với protocol 30.000 vòng, 35.000 và 30.000 như nhau: lần dời cuối ở vòng 29.900 thay vì 24.900. Biến thể kiểm tra lựa chọn đó: dời thêm 5.000 vòng để đặt lại Gaussian, nhưng Gaussian dời ở cuối có ít vòng để tối ưu.
- `sh2`: SH bậc 2 thay vì 3 (gsplat tăng bậc SH mỗi 1.000 vòng đến `sh_degree`): PLY và checkpoint 152 thay vì 236 byte mỗi Gaussian, cùng số Gaussian, tức nhẹ hơn khoảng 36%; đổi lại có thể mất một phần màu thay đổi theo góc nhìn (bề mặt bóng loáng, phản chiếu). Lệnh chấm lại sau khi lọc và lệnh render dùng đúng lệnh đánh giá đã ghi của lượt, nên cùng `sh_degree` với lúc train.

Các khóa của kế hoạch, ngoài các khóa của mục 2 (`src/training/experiment.py` kiểm tra):
- `variants`: danh sách `{mode, tag, set, description}`.
  - `mode` là một chế độ trong `modes`, không phải `default`.
  - `tag`: chữ thường và số, các từ nối bằng `_`.
  - `set`: tham số của `simple_trainer.py`, giá trị là số, true / false hoặc chữ không dấu cách. Khóa protocol và tham số do wrapper quản lý (`max_steps`, `data_factor`, `strategy.cap_max`...) bị từ chối, nên biến thể không đổi được protocol. Khóa `sr3d.kr`, `sr3d.mono` bật biến thể có code (mục 3.7), chỉ nhận `true`; khóa `sr3d.*` khác bị từ chối.
  - Thư mục của biến thể `<chế_độ>_<tag>_s<seed>` không được trùng một chế độ khác.
  - Cấu hình của mỗi lượt (preset, protocol, ngân sách Gaussian, tham số; 250000 và 250000.0 là một) được so sau khi ghép quy ước mcmc của benchmark. Hai lượt cùng chế độ mà cùng cấu hình là lỗi, vì bảng so sánh gom lượt theo chế độ + tham số (vd. trên Deep Blending, một biến thể `opacity_reg: 0.001` của `mcmc` trùng với chính `mcmc`). Hai chế độ khác tên mà cùng cấu hình (vd. một chế độ mới không có tham số riêng) chỉ là cảnh báo, in cùng kế hoạch: cả hai vẫn train, mỗi chế độ một dòng trong bảng. Biến thể bằng đúng giá trị mặc định của gsplat (vd. `opacity_reg: 0.01`) thì chưa bắt được: không có giá trị mặc định của gsplat lúc kiểm tra kế hoạch.
  - Thứ tự ghép tham số (ưu tiên tăng dần): preset của gsplat → quy ước mcmc của benchmark (`benchmark.gsplat.mcmc_set`, vd. `opacity_reg=0.001` trên Deep Blending; chỉ cho chế độ preset mcmc) → tham số của chế độ trong `configs/base.yaml` → tham số của biến thể. Chế độ tự đặt khóa của quy ước (vd. `mcmc_indoor` với `opacity_reg` riêng) thì giữ giá trị của nó trên mọi benchmark; khóa khác của quy ước vẫn được thêm.
  - Tag được ghi vào `run_meta.json` (`src.training.run --tag`): bảng so với `mcmc` lấy các lượt không có tag làm mốc. Bảng chỉ đưa vào lượt có tag khi được yêu cầu (`TAGS` của `scripts/run_evaluate.sh`): bảng của kế hoạch này có các biến thể của nó, bảng của thực nghiệm chính (chạy trên cùng thư mục) thì không, kèm một dòng cảnh báo ghi số lượt để ngoài.
- `post_filter: {min_opacity, max_scale}`: sau mỗi seed, lọc mọi lượt đã train xong của seed đó rồi chấm lại (mục 4). 0 là tắt một tiêu chí, không được tắt cả hai.
- Chế độ hoặc biến thể có `depth_loss: true` thêm bước kiểm tra điểm SfM (mục 3.3, bước 3b). Có `sr3d.mono: true` thì thêm bước `depth_anything` trước `env_check` (mục 3.3, bước 0).

### 3.3 Chạy

```bash
VOLUME_GB=<dung lượng volume> bash scripts/pod/run_experiment.sh tune_mcmc --background   # một việc nền: exp_tune_mcmc
bash scripts/pod/run_experiment.sh tune_mcmc --plan                       # chỉ in kế hoạch, các bước, dữ liệu đã có
```

Hoặc ô 7 của notebook với `WHAT = "experiment"`, `EXP = "tune_mcmc"`. Các bước như mục 2.3, thêm:
- 0. `depth_anything`: tải code và trọng số Depth Anything V2 Small nếu chưa có (`scripts/setup/setup_depth_anything.sh`; bản ghim, kiểm SHA-256). Không tải được thì bước này lỗi và chỉ bỏ các lượt dùng prior (`mono`, `v2`) trên mọi benchmark: kiểm tra môi trường và chỗ trống không tính hai lượt đó, các lượt khác vẫn train, báo cáo ghi một dòng, việc nền kết thúc với exit 1.
- 1. `env_check` kiểm tra tham số của mọi lượt sẽ train với `simple_trainer.py` thật, kể cả bản sao có vá của `kr`, `mono`, `v2`; chạy thử trên GPU với tensor nhỏ KR (cùng bước relocate thật của MCMC), loss của prior và kiểm NaN; khi bước 0 đạt, chạy thử Depth Anything V2 trên ảnh tổng hợp. Sai thì dừng trước khi tải, chưa tốn GPU cho train: lỗi ở đây là lỗi code hoặc môi trường, dừng cả việc như mọi lỗi của `env_check`, sửa rồi chạy lại. Lỗi ở bước 0 (thường do mạng: GitHub, Hugging Face, không sửa ngay được) thì chỉ bỏ riêng các lượt dùng prior.
- 3b. `depth_check_<benchmark>`: mỗi ảnh train của mỗi cảnh có điểm SfM (`src/training/depth_check.py`; Parser của gsplat, không GPU, vài giây). Với `depth_loss`, gsplat @6e8c837 báo `KeyError` khi nạp một ảnh không có điểm nào trong track của SfM: không đạt, và chỉ các lượt có loss độ sâu của benchmark đó bị bỏ (B0 và các lượt khác vẫn chạy; baseline có loss độ sâu thì bỏ mọi lượt gsplat của benchmark đó, vì các lượt khác lấy ngân sách Gaussian từ nó). Ảnh có track nhưng mọi điểm nằm ngoài ảnh thì chỉ cảnh báo: ảnh đó không có giám sát độ sâu, loss ghi ra là NaN ở những vòng gặp ảnh đó, gradient vẫn hữu hạn (thử lại đoạn loss của simple_trainer.py bằng PyTorch trên CPU).
- 5. `gsplat_<benchmark>_s<seed>`: trên mỗi cảnh: `default`, `mcmc`, rồi 16 biến thể theo thứ tự trong kế hoạch (mục 3.2). Lượt `mono`, `v2` có thêm bước `mono` trước khi train (mục 3.7): lần đầu tính depth prior của cảnh, các lượt sau dùng lại; prior không khớp điểm SfM thì lượt đó lỗi ở bước này, các lượt khác vẫn train, lọc và báo cáo vẫn chạy, nhưng bước `gsplat_<benchmark>_s<seed>` ghi lỗi và việc nền kết thúc với exit 1.
- 5b. `filter_<benchmark>_s<seed>`: lọc và chấm lại mọi lượt đã xong của seed đó (`src/training/prune.py --skip-unfinished`; lượt train lỗi thì bỏ qua, bước gsplat đã báo lỗi). Lọc lỗi không dừng việc train.
- Lượt `room/default_s42` và `room/mcmc_s42` dùng chung thư mục với thực nghiệm chính: hai lượt này chỉ train một lần, dù `main` chạy trước hay sau.

### 3.4 Thời gian và dung lượng

Ước lượng trên RTX 4090, chưa phải số đo (`configs/experiments/tune_mcmc.yaml`):
- Khoảng 7,3 giờ GPU (kế hoạch ghi 8 cho dư): 18 lượt × khoảng 19 phút (train, chấm, video; cùng giả định với Mip-NeRF 360 trong `main.yaml`), cộng lọc và chấm lại khoảng 4 phút mỗi lượt (khoảng 6,9 giờ), cộng phần chậm hơn chưa đo của `bilagrid`, `mono`, `v2` (giả định +30 % mỗi lượt: render thêm độ sâu, loss thêm) và depth prior của `room` (vài phút, một lần). Chạy sau `main` thì `default_s42`, `mcmc_s42` đã có: 16 lượt train, 18 lượt lọc, khoảng 6,7 giờ.
- Thêm khoảng 35 GB trên volume: khoảng 1,8 GB mỗi lượt (checkpoint và PLY, ảnh render, video, bản đã lọc; `sh2` nhỏ hơn), cộng khoảng 2 GB dữ liệu `room` và ảnh PNG thu nhỏ, khoảng 0,3 GB depth prior, khoảng 0,2 GB Depth Anything V2 Small (code khoảng 114 MB kể cả lịch sử git, trọng số 99 MB). Lúc tải, `360_v2.zip` (khoảng 12,5 GB) nằm tạm trên volume.
- Chạy `main`, `tune_mcmc`, vòng ghép (mục 3.6) rồi `mcmc_indoor` trên cùng volume: khoảng 140 GB kết quả (56 + 35 + tối đa 20 + khoảng 25 GB cho 18 lượt `mcmc_indoor`) cộng 15–20 GB môi trường, nên dùng volume 170 GB. Volume tăng được bất cứ lúc nào, không giảm được: có thể bắt đầu với 100 GB cho `main` rồi tăng trước các bước sau.

### 3.5 Kết quả và cách đọc

`experiment_tune_mcmc_<thời_gian>.md`: bảng các bước, bảng so với `default`, bảng so với `mcmc` (mỗi biến thể là một dòng `mcmc[<tham số>]`, kèm ΔPSNR / ΔSSIM / ΔLPIPS so với `mcmc` và tỉ lệ thời gian train), bảng sau khi lọc (mục 4), chất lượng theo dung lượng và cấu trúc mô hình (`docs/design/comparisons.md`). Đọc theo mục 3.1: một seed chỉ để chọn ứng viên cho vòng ghép (mục 3.6).

Với các biến thể nhắm vào floater (`or005`, `mo002`, `rbg`, `depth`), chỉ số trên ảnh test chưa đủ: xem thêm bảng cấu trúc mô hình (tỉ lệ Gaussian gần trong suốt, rất to) và ảnh so sánh với `mcmc`, nhất là nửa độ sâu của video: `VIDEO=1 bash scripts/run_panels.sh mipnerf360 room mcmc_s42 mcmc_or005_s42 mcmc_mo002_s42` (ô 7: `WHAT = "panels"`, `BENCH = "mipnerf360"`, `SCENES = "room"`, `PANEL_RUNS`).

Với `sh2`: bảng chất lượng theo dung lượng (cột byte mỗi Gaussian) và bề mặt bóng loáng, phản chiếu trên ảnh so sánh, vì SH bậc thấp hơn giữ ít màu thay đổi theo góc nhìn hơn.

Với các biến thể của mục 3.7:
- `bilagrid`, `v2`: PSNR / SSIM / LPIPS tính trên ảnh render **không qua** bilateral grid (ảnh test không có grid riêng), như mọi lượt. gsplat còn ghi `cc_psnr` / `cc_ssim` / `cc_lpips` vào `stats/val_step*.json`, nhưng các số đó chỉnh màu ảnh render theo chính ảnh test (`color_correct`), nên không dùng để so. Bilateral grid hấp thụ chênh lệch màu giữa các ảnh train; nếu ảnh test có phơi sáng khác ảnh train, PSNR có thể giảm dù hình học tốt hơn: xem thêm ảnh so sánh.
- `kr`, `v2`: `sr3d_hooks.jsonl` của lượt ghi mỗi lần KR bao nhiêu Gaussian bị dời và vì sao (quá lớn, hình kim, sát camera). Xem ảnh so sánh và nửa độ sâu của video như các biến thể nhắm vào floater.
- Lượt có hook (`kr`, `mono`, `v2`) dừng ngay khi một Gaussian có tham số không hữu hạn (mục 3.7); lượt khác train đến hết, PLY của gsplat lặng lẽ bỏ các Gaussian đó (`export_splats`) còn checkpoint vẫn giữ. Số Gaussian không hữu hạn trong checkpoint cuối của mọi lượt có ở `filtered/filter_meta.json` (`counts.removed.non_finite`). Một lượt có hook dừng vì NaN thì xem số đó của `mcmc` và các lượt khác trước khi kết luận biến thể làm train kém ổn định: tiêu chí dừng chặt hơn tiêu chí của các lượt được so.
- `mono`, `v2`: `mono_depth.json` ghi số ảnh train dùng prior và trung vị tương quan với điểm SfM; `sr3d_hooks.jsonl` ghi loss Pearson trung bình mỗi 500 vòng. Tường, sàn phẳng hơn phải thấy ở nửa độ sâu của video; độ sâu render đẹp hơn không có nghĩa là hình học đúng hơn khi chưa có độ sâu thật để so.

### 3.6 Vòng ghép: chọn `mcmc_indoor` trên seed mới

**Trạng thái: mới thiết kế; file kế hoạch viết khi có kết quả của `tune_mcmc`; chưa chạy.**

Mục đích: chọn bộ tham số cho `mcmc_indoor` trên những lượt chưa dùng để sàng lọc. Một seed không tách được cải thiện thật khỏi dao động giữa hai lần train; hai tham số tốt khi đổi riêng chưa chắc còn tốt khi ghép chung; và lượt đã dùng để chọn (seed 42, kể cả mốc `mcmc_s42`) cho kết quả lạc quan nếu dùng lại để quyết định. Các luật dưới đây cố định trước khi có số liệu.

**Mức dao động.** Với mỗi chỉ số (PSNR, SSIM, LPIPS), σ là độ lệch chuẩn gộp giữa các seed của `default` và `mcmc` trên `room` trong bảng của `main`: σ = √((σ_default² + σ_mcmc²) / 2), từ 6 lượt (một σ ước từ 3 lượt quá bất định). Chênh lệch giữa hai lượt có độ lệch khoảng √2·σ, nên đơn vị dao động là d = √2·σ. σ gộp vẫn còn bất định: d chỉ là ngưỡng thô. Điểm chuẩn hóa của một cấu hình là S = ΔPSNR / d_PSNR + ΔSSIM / d_SSIM − ΔLPIPS / d_LPIPS (Δ = cấu hình − `mcmc`; LPIPS thấp hơn là tốt hơn).

1. **Chọn ứng viên** từ bảng so với `mcmc` của `tune_mcmc` (`room`, seed 42), trong 15 biến thể đổi một thứ (12 tham số cùng `bilagrid`, `kr`, `mono`); gói `v2` không phải ứng viên (mục 3.7). Một biến thể là ứng viên khi không kém `mcmc` quá d ở chỉ số nào, và có thêm một trong ba điều:
   - hơn `mcmc` quá d ở ít nhất một chỉ số;
   - ảnh so sánh và nửa độ sâu của video có ít floater hơn rõ (bằng chứng chính, đánh giá bằng mắt). Bảng cấu trúc mô hình chỉ để tham khảo: `mo002` (MCMC dời mọi Gaussian có opacity ≤ 0,02, đúng ngưỡng của bảng và của bộ lọc) và phần nào `rs30k` tự làm giảm tỉ lệ Gaussian gần trong suốt;
   - file nhẹ hơn rõ (`sh2`; bảng chất lượng theo dung lượng).

   Mỗi ứng viên ghi lý do theo số liệu. Cùng một tham số chỉ giữ giá trị có S lớn nhất (vd. một trong `or0001`, `or0005`, `or005`). Nhiều hơn 3 ứng viên thì giữ 3 ứng viên có S lớn nhất; nếu 3 ứng viên đó không có ứng viên nào vào nhờ floater mà vẫn còn ứng viên như vậy, ứng viên floater có S lớn nhất thay cho ứng viên thứ ba. Không có ứng viên nào thì dừng: ghi kết quả âm tính, không có vòng ghép, không định nghĩa `mcmc_indoor`.
2. **Cấu hình.** Gọi A là ứng viên có S lớn nhất, B và C là các ứng viên còn lại: 1 ứng viên → {A}; 2 ứng viên → {A, A+B}; 3 ứng viên → {A, A+B, A+C, A+B+C}. Mỗi cấu hình là một biến thể của `mcmc`, có thể nhiều tham số trong `set`, vd. `{mode: mcmc, tag: c2, set: {opacity_reg: 0.05, strategy.min_opacity: 0.02}}`; ứng viên `kr`, `mono` vào cấu hình bằng `sr3d.kr: true`, `sr3d.mono: true`. Mọi cấu hình dùng tag mới (`c1`…`c4`), kể cả A: bảng lấy mọi seed có trên đĩa của các tag được yêu cầu, nên tag cũ sẽ đưa lượt seed 42 của lượt dò vào bảng.
3. **Kế hoạch** `configs/experiments/tune_mcmc_combo.yaml`, cùng các khóa của mục 3.2: `modes: [default, mcmc]`, `also_against: [mcmc]`, `variants` là các cấu hình ở bước 2, `seeds: [43, 44]`, `room` của Mip-NeRF 360, `b0: false`, cùng `post_filter` với `tune_mcmc`.
   - `estimate.disk_gb` khoảng 1,8 GB × số lượt của kế hoạch, kể cả các lượt `default`, `mcmc`: khi chưa có lượt cùng cấu hình đã xong, kiểm tra chỗ trống tính mỗi lượt bằng `disk_gb` chia đều cho mọi lượt.
   - Chạy như mục 3.3 với tên kế hoạch mới (ô 7: `EXP = "tune_mcmc_combo"`), sau `main` (cần σ và các lượt seed 43, 44 của nó) và trên cùng loại GPU với `main`: lượt `default`, `mcmc` seed 43, 44 của `room` dùng chung thư mục với `main` nên không train lại, và bảng so sánh kiểm tra tên GPU của mọi lượt.
   - Bảng ghi `seeds=DIFF`, vì `default` và `mcmc` còn lượt seed 42; chênh lệch và số cặp tốt hơn tính trên seed chung 43, 44. Một cấu hình không có `seeds=DIFF` hoặc số cặp không phải x/2 là đã lẫn lượt seed 42 (tag cũ): dừng, sửa tag rồi chạy lại.
4. **Chọn `mcmc_indoor`** theo bảng so với `mcmc` của vòng này (chỉ seed 43, 44; lượt seed 42 không dùng). Δ và S lấy từ bảng chênh lệch so với `mcmc` (tính trên seed chung) và bảng số cặp, không lấy hiệu hai dòng của bảng đầu (bảng đầu tính `default`, `mcmc` trên cả seed 42):
   - một cấu hình đạt khi có ít nhất hai chỉ số tốt hơn `mcmc` ở cả hai seed (cột "Cặp tốt hơn" là 2/2), chỉ số còn lại không kém quá d (trung bình hai seed), mọi mục điều kiện khác `seeds` đều OK và ngân sách EQUAL;
   - trong các cấu hình đạt, cấu hình có S (trung bình hai seed) cách S lớn nhất dưới 2 đơn vị coi là ngang nhau; trong nhóm đó chọn cấu hình ít floater hơn rõ trên ảnh so sánh và video độ sâu, rồi cấu hình cho file nhẹ hơn (bảng chất lượng theo dung lượng), rồi cấu hình ít tham số hơn. Ngoài luật hòa này, bằng chứng floater chỉ dùng để vào vòng ghép và để báo cáo;
   - không cấu hình nào đạt: ghi kết quả âm tính, không định nghĩa `mcmc_indoor`.

   "Đạt" ở bước này là bộ lọc thô, không phải bằng chứng cải thiện. Mô phỏng Monte Carlo (chỉ số theo phân phối chuẩn, tương quan giữa ba chỉ số từ 0 đến 0,8, σ gộp như trên): khi không có hiệu ứng thật, khoảng 13–23 % lần vẫn có cấu hình đạt nếu chỉ có 1 cấu hình, 36–52 % nếu có 4 cấu hình; một hiệu ứng thật cỡ 1σ ở mọi chỉ số đạt khoảng 59–60 %, cỡ 2σ khoảng 87–94 %. Bằng chứng là thực nghiệm chính trên 5 cảnh còn lại.

   `mcmc_indoor` trong `configs/base.yaml` mang đúng `set` của cấu hình được chọn (mục 3.1). Bước lọc không thuộc chế độ: lọc là hậu xử lý, báo cáo riêng (mục 4).
5. **Không dùng 5 cảnh kết luận để chọn.** Bước 1–4 chỉ dùng bảng và ảnh của `room`. Bảng của `main` trên 5 cảnh còn lại đã có trước khi chọn; khi chọn, chúng chỉ dùng để kiểm điều kiện (B0 so với số Inria công bố, `default` so với B0, đủ seed). Việc đối chiếu `default` / `mcmc` với lượt 17/10/2026 của repo trước làm sau khi chọn (P9 của `docs/PROJECT_STATUS.md`). Ghi quyết định và lý do trước khi đọc các bảng đó để kết luận; báo cáo ghi rõ kết quả `default` / `mcmc` trên các cảnh đó đã có trước khi chọn.
6. **Báo cáo công sức dò:** số cấu hình đã thử (16 biến thể của `tune_mcmc` và các cấu hình của vòng này), tất cả trên `room`; `default` và `mcmc` dùng tham số gốc, không dò. Kết luận về `mcmc_indoor` vẫn dựa trên 5 cảnh còn lại (mục 3.1).

**Ước lượng** trên RTX 4090, chưa phải số đo: 1–4 cấu hình × 2 seed, tức 2–8 lượt, mỗi lượt khoảng 23 phút kể cả lọc và chấm lại, cộng lọc và chấm lại 4 lượt `default`, `mcmc` seed 43, 44 (khoảng 16 phút): khoảng 1–3,5 giờ GPU. Thêm khoảng 10–20 GB: khoảng 1,8 GB mỗi lượt mới, cộng bản đã lọc của 4 lượt kia (tối đa khoảng 6 GB).

### 3.7 Biến thể từ notebook v2: `bilagrid`, `kr`, `mono` và gói `v2`

**Trạng thái: đã viết, đã kiểm thử trên CPU (mục cuối); chưa train trên GPU, chưa có số liệu.** Quyết định D16 (`docs/PROJECT_STATUS.md`).

Notebook demo v2 của nhóm (`3D_demo_v2.ipynb`, giống v4 `3D_demo_v4_improved_30k.ipynb` nhưng 7.000 vòng) thêm vào `mcmc`: `global_scale` tính lại cho bền với camera lỗi, tâm cảnh theo vị trí camera, Kill-then-Relocate, bilateral grid, depth prior; cùng `depth_loss` (điểm SfM), `random_bkgd` và `strategy.refine_stop_iter` = 0,8 × số vòng. `depth_loss` và `random_bkgd` đã có ở `depth`, `rbg`. Ba thành phần còn lại thành ba biến thể, mỗi biến thể thêm đúng một thành phần; thêm một gói bật cùng lúc mọi thứ.

| Biến thể | `set` | Cách làm trong repo | Nguồn |
|---|---|---|---|
| `bilagrid` | `use_bilateral_grid: true` | Cờ có sẵn của `simple_trainer.py` (`examples/lib_bilagrid.py`, cần `tensorly`: có trong env `gs-gsplat`). Mỗi ảnh train có một grid 16 × 16 × 8 học phép biến đổi màu; ảnh test render không qua grid. Grid không nằm trong checkpoint | Wang et al., *Bilateral Guided Radiance Field Processing*, ACM TOG (SIGGRAPH) 2024 |
| `kr` | `sr3d.kr: true` | Hook trên bản sao có vá của trainer (`docs/design/training_modes.md` mục 3.1): ở các vòng refine của MCMC chia hết cho 500, từ 3.000 đến 21.000 (10–70 % số vòng), Gaussian còn sống mà trục lớn nhất > 0,15 × `scene_scale`, hình kim (trục lớn nhất > 0,01 × `scene_scale` và gấp hơn 15 lần trục giữa) hoặc cách một camera train dưới 0,03 × `scene_scale` bị hạ opacity xuống dưới `strategy.min_opacity`; bước relocate của MCMC ngay sau đó trong cùng vòng dời chúng tới vị trí các Gaussian khác. Mỗi lần tối đa 2 % số Gaussian, ưu tiên trục lớn nhất; số Gaussian không đổi. Như Gaussian "chết" của MCMC (và như notebook), Gaussian bị dời nhận tham số mới nhưng giữ moment Adam cũ: relocate của gsplat chỉ đặt lại moment của Gaussian nguồn | Notebook của nhóm, không có paper. Khác bước relocate gốc của 3DGS-MCMC (chỉ dời Gaussian gần trong suốt, ảnh render gần như không đổi), KR dời cả Gaussian còn nhìn thấy: heuristic nằm ngoài công thức MCMC |
| `mono` | `sr3d.mono: true` | Bước `mono` trước khi train tính depth prior của cảnh bằng Depth Anything V2 Small (`src/training/mono_depth.py`, mục dưới); hook cộng vào loss: λ · (1 − Pearson(1/độ sâu render, prior)) trên pixel có alpha > 0,5 (ít nhất 256 pixel), nghịch đảo độ sâu bị chặn trong khoảng phân vị 2–98 %, λ = 0,05, đến vòng 18.000 (60 % số vòng). Render thêm độ sâu (`RGB+ED`) dù `depth_loss` tắt | Depth Anything V2 (Yang et al., NeurIPS 2024); loss Pearson cho độ sâu đơn ảnh như FSGS (Zhu et al., ECCV 2024); hằng số theo notebook |
| `v2` | `depth_loss`, `random_bkgd`, `use_bilateral_grid`: true; `strategy.refine_stop_iter`: 24000; `sr3d.kr`, `sr3d.mono`: true | Mọi thứ trên cùng lúc | Gói, theo notebook |

**Depth prior** (`src/training/mono_depth.py`): Depth Anything V2 Small từ code chính thức (commit `a561b84`, Apache-2.0) với trọng số bản Small (Apache-2.0; các bản lớn hơn là CC-BY-NC-4.0), kiểm SHA-256; tải bằng `scripts/setup/setup_depth_anything.sh` vào `$WS/code/depth_anything_v2`, chạy trong env `gs-gsplat` (không thêm gói Python). Với mỗi ảnh đã đăng ký: bản đồ độ sâu tương đối (lớn = gần), thu về cạnh dài 768 px (mạng chạy ở cạnh ngắn 518 px), lưu float16 trong `<thư mục kết quả của cảnh>/.mono_depth_<dấu mô hình COLMAP>/`, dùng chung cho mọi lượt của cảnh. Độ tin cậy từng ảnh: hệ số Pearson với nghịch đảo độ sâu của các điểm SfM ảnh đó thấy (≥ 30 điểm); hệ số âm thì thử đảo bản đồ; dưới 0,15 thì ảnh đó không dùng prior. Cả cảnh: trung vị trên ảnh train phải ≥ 0,3, không thì bước `mono` lỗi và lượt không train. Camera phải không méo (PINHOLE / SIMPLE_PINHOLE): benchmark và cảnh qua `colmap_runner` đều vậy.

**Khác notebook** (theo protocol chung, để so được với `mcmc`):
- 30.000 vòng (notebook v2: 7.000); `cap_max` bằng số Gaussian cuối của `default` (notebook: cố định 1.000.000); `antialiased` tắt (notebook bật); LPIPS VGG (notebook AlexNet).
- KR chỉ xét camera train. Notebook xét mọi camera, kể cả camera test, tức dùng vị trí camera test khi train.
- Không tự tắt thành phần khi gặp NaN. Notebook tự tắt dần depth prior, bilateral grid, fused-ssim, antialiased rồi train lại, nên không chắc lượt cuối dùng phương pháp nào; bộ dò NaN của notebook cũng không tắt được `depth_loss` (đã kiểm tra bằng mô phỏng). Ở repo, lượt có hook (`kr`, `mono`, `v2`) kiểm tham số Gaussian mỗi 500 vòng: không hữu hạn thì lượt dừng và lỗi (`FloatingPointError`, ghi vào `sr3d_hooks.jsonl`). Lượt không có hook (`bilagrid` và mọi lượt khác) như mọi lượt của gsplat: không kiểm giữa chừng, chỉ số cuối không hữu hạn thì bảng so sánh ghi lượt là lỗi.
- Depth prior: code chính thức thay cho pipeline `transformers` (bản `-hf`) của notebook; cạnh dài 768 px thay cho một nửa độ phân giải ảnh; prior không đạt thì lượt lỗi, notebook thì tự tắt prior rồi train tiếp như `mcmc` thường. Bộ ước lượng được kiểm bằng ảnh tổng hợp trong `env_check`.
- Bỏ `global_scale` tính lại và tâm cảnh theo camera: cảnh benchmark không có camera đăng ký sai, còn đổi tâm cảnh chỉ dời và co giãn cả cảnh. Hai bước này thuộc phần chuẩn bị dữ liệu của cảnh tự quay; xét lại khi chạy video của nhóm.
- Chỉ số: PSNR trên ảnh render không qua grid, không dùng `cc_psnr` (mục 3.5).

**Gói `v2`** chỉ trả lời "notebook v2, đưa về protocol chung, so với `mcmc`" trên `room`, seed 42. Nó không phải ứng viên của vòng ghép: gộp sáu thay đổi nên không biết phần nào có ích, và một phần có lợi có thể bị một phần có hại che đi; câu hỏi đó do ba biến thể tách riêng trả lời. **Chưa quyết định:** có đưa `v2` vào vòng ghép như một cấu hình thêm (2 seed mới) để có số chắc hơn một seed hay không. Quyết định đó phải ghi lại trước khi đọc kết quả của `tune_mcmc` (mục 3.6: luật cố định trước khi có số liệu); không ghi thì `v2` chỉ là một dòng sàng lọc.

**Đã kiểm thử trên CPU** (chưa train trên GPU):
- vá `simple_trainer.py` thật của gsplat @6e8c837: đúng SHA-256 của bản ghim, 4 dòng chèn nằm đúng chỗ trong vòng lặp của `Runner.train` (cây cú pháp: sau khi tách màu / độ sâu; loss của prior sau loss độ sâu SfM, trước TV loss và `loss.backward()`; KR là lệnh đầu của nhánh MCMC, trước `step_post_backward`); `smoke_test_gsplat.py` mục 5 với trainer thật (PyTorch 2.7.1 chạy trên CPU, thư viện theo file khóa của env, fused-ssim thay bằng stub) phân giải đúng tham số của cả 18 lượt của `tune_mcmc`, kể cả bản sao có vá của `kr`, `mono`, `v2` (kiểm tra tay, ngoài bộ test);
- KR với `MCMCStrategy` thật của gsplat: vòng KR trùng vòng relocate, Gaussian bị đánh dấu nằm trong mặt nạ "chết" mà relocate nhận; đúng các trường hợp quá lớn / hình kim / sát camera train, bỏ qua camera test, giới hạn 2 %;
- loss Pearson bằng PyTorch: prior đúng hình dạng cho loss gần 0, prior ngược cho 2λ, có gradient, tắt sau 60 % số vòng, ảnh không dùng prior và ảnh ít pixel cho 0, prior nhỏ hơn được nội suy;
- kiểm NaN: tham số Gaussian không hữu hạn ở vòng kiểm thì lượt dừng (`FloatingPointError`) và nhật ký có dòng `guard`; phần chạy thử mà `env_check` làm trên GPU (`hook_smoke` của `smoke_test_gsplat.py`) chạy được trên CPU, trừ bước relocate thật của MCMC (kernel CUDA);
- bước `mono` trên cảnh COLMAP tổng hợp với bộ ước lượng giả (tương quan, đảo bản đồ, ảnh ít điểm SfM, kiểm cả cảnh, dùng lại cache, không xóa thư mục cache có file lạ, từ chối camera méo, sai kích thước ảnh, ảnh JPEG có cờ xoay EXIF đọc theo đúng hướng điểm ảnh như gsplat) và với code Depth Anything V2 thật nhưng trọng số ngẫu nhiên (kiểm tra tay, ngoài bộ test: đúng kích thước, float16, manifest, dùng lại cache; trọng số thật chưa tải được trong sandbox);
- `setup_depth_anything.sh` với một kho git cục bộ và `file://` thay cho GitHub, Hugging Face: cài đúng commit, chạy lại an toàn, trọng số sai SHA-256 bị xóa, không đụng thư mục không phải bản checkout, từ chối bản checkout đã sửa;
- launcher với trainer giả có đủ 4 chỗ chèn: chạy bản sao có vá, file gốc không đổi, mọi dòng chèn đều được gọi; `run.py`: bước `mono` → train (có `--hook`) → đánh giá (không hook), lỗi ở bước `mono` thì không train, đổi hằng số của hook thì không bỏ qua lượt cũ;
- `run_experiment.sh` với stub: bước `depth_anything` chạy trước `env_check`; lỗi thì chỉ bỏ lượt dùng prior (kiểm tra môi trường và chỗ trống không tính lượt đó), các lượt khác vẫn train, báo cáo ghi một dòng; baseline dùng prior thì không train gì.

Môi trường và cách làm các kiểm tra tay: `docs/PROJECT_STATUS.md`, mục "Kiểm thử trong sandbox", lượt 28/10/2026.

Chưa biết: thời gian và VRAM thêm của `bilagrid`, `mono`, `v2` (ước lượng ở mục 3.4 giả định +30 %); trọng số thật của Depth Anything V2 cho tương quan bao nhiêu với điểm SfM của `room`.

## 4. Lọc Gaussian sau khi train

`src/training/prune.py`; trong kế hoạch: khóa `post_filter`; cho một lượt bất kỳ: `bash scripts/run_filter.sh <luồng> <cảnh> [<lượt> ...]` (vd. `custom phong1 mcmc_s42`). **Trạng thái: đã viết, đã kiểm thử trong sandbox (logic lọc bằng numpy; đọc / ghi checkpoint bằng PyTorch 2.7.1 chạy trên CPU; ghi PLY bằng `export_splats` thật của gsplat @6e8c837; lệnh chấm lại bằng runner giả); chưa chạy trên GPU.**

- **Bỏ** một Gaussian khi tham số không hữu hạn, khi opacity (sau sigmoid) < `min_opacity` (lớp sương mờ, vệt mờ lơ lửng), hoặc khi trục lớn nhất > `max_scale` × `scene_scale`. Mỗi Gaussian chỉ được đếm ở tiêu chí đầu tiên loại nó. Các Gaussian còn lại giữ nguyên tham số, không tối ưu lại.
- **`scene_scale`** đúng như `simple_trainer.py` tính (Parser của gsplat trên thư mục dữ liệu của lượt: khoảng cách lớn nhất từ một camera tới tâm các camera sau khi chuẩn hóa, × 1,1 × `global_scale`), cùng đại lượng mà DefaultStrategy dùng cho ngưỡng `prune_scale3d` = 0,1 lúc densify. Script đối chiếu với dòng `Scene scale:` trong `train.log`; lệch thì dừng. Ngưỡng tính theo camera, nên mọi chế độ trên cùng cảnh dùng cùng một ngưỡng tuyệt đối.
- **Chấm lại** bằng đúng lệnh đánh giá đã ghi của lượt (`run_meta.json`, `commands.eval`), chỉ đổi `--result-dir`, `--ckpt` và hai file ghi lại của `gsplat_launcher`: cùng ảnh test, cùng protocol. Script kiểm tra số Gaussian gsplat thấy bằng số đã giữ lại, và ảnh chấm lại đúng là tập test của lượt.
- **Kết quả** vào `<lượt>/filtered/`: checkpoint và PLY đã lọc, `stats/`, `renders/`, `videos/` của lượt chấm lại, `filter_meta.json` (ngưỡng, `scene_scale`, số Gaussian bỏ theo từng tiêu chí, SHA-256 checkpoint gốc, lệnh). Lượt gốc không bị sửa.
- **Chạy lại:** cùng ngưỡng và cùng checkpoint gốc thì bỏ qua; ngưỡng khác (hoặc lần trước lỗi) thì thư mục cũ đổi tên `filtered.prev_<thời_gian>`. Mô hình COLMAP của cảnh đổi sau khi train thì dừng.
- **Báo cáo:** `src.evaluation.compare_modes` thêm bảng sau khi lọc: chỉ số của bản đã lọc, chênh lệch so với chính lượt đó trước khi lọc (cùng ảnh test), số Gaussian giữ lại, PLY trước → sau; có thêm cột ngưỡng khi các dòng lọc với ngưỡng khác nhau. Mọi bảng khác là mô hình chưa lọc. Bảng sau khi lọc chỉ có các lượt đã lọc: vd. chạy lại `main` sau `tune_mcmc` và vòng ghép thì bảng của Mip-NeRF 360 có thêm phần này cho riêng `default`, `mcmc` của `room` (seed 42 lọc trong `tune_mcmc`, seed 43, 44 trong vòng ghép).
- Lọc là hậu xử lý cho mô hình gọn hơn khi xem, không phải cải tiến huấn luyện. Bỏ Gaussian lớn có thể làm thủng tường hay trần ở xa; vì vậy luôn báo cáo chỉ số sau khi lọc cạnh chỉ số gốc.

## 5. Kết quả lưu ở đâu

Thư mục dữ liệu và kết quả lấy từ `configs/<benchmark>.yaml` (`docs/design/architecture.md` mục 2 và 5):

| | Thư mục |
|---|---|
| Dữ liệu | `$WS/data/benchmarks/<benchmark>/<cảnh>/` (file zip tải về nằm tạm ở `$WS/data/downloads/`) |
| B0 Inria | `$WS/outputs/benchmarks/<benchmark>/inria_b0/<cảnh>/` |
| gsplat | `$WS/outputs/benchmarks/<benchmark>/gsplat/<cảnh>/<chế_độ>[_<tag>]_s<seed>/` (biến thể: `<tag>`; bản đã lọc: `filtered/` bên trong lượt) |
| gsplat chạy thử (`QUICK=1`) | `$WS/outputs/benchmarks/<benchmark>/gsplat_quick/<cảnh>/<chế_độ>_s<seed>/` |
| Báo cáo, log | `$WS/outputs/reports/{baseline_<tag>_summary,modes_<tag>,compare_<tag>,experiment_<tên>}_<thời_gian>.*`, `$WS/outputs/logs/` (`benchmark.tag`: `db`, `mipnerf360`) |

Không ghi đè lượt gsplat: lượt đã có kết quả OK với cùng cấu hình thì được bỏ qua; khác cấu hình thì báo lỗi. Khi được phép thay, thư mục cũ được đổi tên `.prev_<thời_gian>`. B0 thì khác: cảnh đã có `results.json` (đủ chỉ số `ours_30000`) thì bỏ qua mà không so cấu hình; `FORCE=1` chạy lại, lượt cũ được đổi tên `<cảnh>.prev_<thời_gian>` (không ghi đè).

## 6. So sánh các chế độ

`scripts/run_evaluate.sh` (`docs/design/comparisons.md`) và bước cuối của `scripts/run_train.sh <benchmark> gsplat` lập cùng một bảng.

**Chỉ gọi là "công bằng" khi:**
- cùng cảnh và cùng mô hình COLMAP;
- cùng tập train/test;
- cùng protocol;
- cùng số Gaussian (`mcmc` lấy `cap_max` bằng số Gaussian cuối của `default`).

Bảng tự kiểm tra các điều kiện này.

**Chỉ gọi là "cải thiện" khi** có ít nhất 3 seed và mọi điều kiện đều đạt (`docs/design/training_modes.md` mục 4.3).

Biến thể của một kế hoạch (mục 3.2) là một dòng riêng `<chế_độ>[<tham số>]`, cùng các kiểm tra công bằng như một chế độ. Bảng sau khi lọc (mục 4) chỉ có khi có lượt đã lọc. Số cặp thắng, chất lượng theo dung lượng, cấu trúc mô hình và ảnh so sánh: `docs/design/comparisons.md`.

## 7. Đã chạy / mới cấu hình

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| B0 Inria trên Deep Blending | Repo trước đã chạy trên pod ngày 17/10/2026, đạt ngưỡng so với số liệu Inria. **Repo này chưa chạy** | `docs/PROJECT_STATUS.md` (lịch sử), `docs/ENVIRONMENT_AUDIT.md` mục 7 |
| gsplat `default` / `mcmc` trên Deep Blending, seed 42 | Repo trước đã chạy trên pod ngày 17/10/2026. **Repo này chưa chạy** | như trên |
| Thực nghiệm chính (mục 2): `default` / `mcmc`, 3 seed, Deep Blending + 4 cảnh Mip-NeRF 360 | **Đã viết, chưa chạy trên pod** | Kiểm thử bằng stub: `tests/unit/test_experiment.py`, `test_fetch.py`, `test_benchmarks.py`, `test_compare_modes.py`; mô phỏng notebook và rà soát độc lập (`docs/PROJECT_STATUS.md`) |
| Tải dữ liệu benchmark trên pod | Đã viết; kiểm thử với curl giả và file zip tổng hợp; **chưa tải thật** (sandbox không ra được Internet) | `tests/unit/test_fetch.py` |
| Mip-NeRF 360 (B0 và gsplat) | **Đã cấu hình, chưa chạy; chưa có dữ liệu trên volume** | `experiments/mipnerf360/README.md` |
| Lệnh huấn luyện mà repo này sinh ra cho Deep Blending | Kiểm thử bằng stub: `default` / `mcmc` giống lệnh repo trước đã chạy ngày 17/10/2026, chỉ khác đường dẫn | `tests/unit/test_benchmarks.py` |
| Dò tham số `mcmc` (`tune_mcmc`, mục 3): biến thể, kiểm tra điểm SfM, lọc theo seed | **Đã viết, chưa chạy trên pod** | Kiểm thử bằng stub: `tests/unit/test_experiment.py`, `test_benchmarks.py`, `test_depth_check.py` |
| Biến thể từ notebook v2 (mục 3.7): `bilagrid`, `kr`, `mono`, gói `v2`; bản sao có vá của trainer; depth prior; bước `depth_anything` | **Đã viết, đã kiểm thử trên CPU (kể cả trainer và `MCMCStrategy` thật của gsplat), chưa train trên GPU, chưa chạy smoke test trên GPU, chưa tải Depth Anything V2 thật** | `tests/unit/test_trainer_hooks.py`, `test_mono_depth.py`, `test_setup_depth_anything.py`, `test_gsplat_launcher.py`, `test_train_run.py`, `test_train_profiles.py`, `test_experiment.py`; các kiểm tra ngoài bộ test ghi ở mục 3.7 |
| Vòng ghép (mục 3.6) | **Mới thiết kế**: luật chọn đã cố định; file kế hoạch viết khi có kết quả `tune_mcmc`; chưa chạy | Cơ chế đã có: biến thể nhiều tham số được kiểm tra trong kế hoạch (`test_experiment.py::test_a_combination_round_plan_has_several_parameters_per_variant`) và thành nhiều `--set` khi train (`test_benchmarks.py::test_variants_run_after_the_modes_into_their_own_folders`) |
| Lọc Gaussian và chấm lại (mục 4) | **Đã viết, chưa chạy trên GPU**; logic lọc, checkpoint PyTorch và PLY của gsplat kiểm tra trên CPU | `tests/unit/test_prune.py`, `test_compare_modes.py`, `test_gsplat_results.py` |
