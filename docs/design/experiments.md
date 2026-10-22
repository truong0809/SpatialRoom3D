# Thực nghiệm: Deep Blending, Mip-NeRF 360, so sánh các chế độ huấn luyện, dò tham số mcmc

_Cập nhật: 21/10/2026_

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

Chỉ gọi là "cải thiện" theo mục 6, và thêm:
- chênh lệch so với `default` cùng dấu trên phần lớn cảnh, trên cùng 3 seed;
- chênh lệch lớn hơn độ dao động giữa các seed (độ lệch chuẩn trong bảng).

Nếu chỉ một benchmark cải thiện, hoặc chỉ một chỉ số, thì ghi đúng như vậy.

## 3. Dò tham số `mcmc`: `tune_mcmc`

Kế hoạch: `configs/experiments/tune_mcmc.yaml`. Chạy bằng cùng script với thực nghiệm chính. **Trạng thái: đã viết, đã kiểm thử bằng stub trong sandbox; chưa chạy trên pod, chưa có số liệu.**

### 3.1 Câu hỏi và cách dùng kết quả

- Với cảnh trong nhà, tham số nào của `mcmc` đáng đổi: độ mạnh của regularization opacity / scale, nhiễu của MCMC, ngưỡng opacity mà MCMC dời Gaussian, nền ngẫu nhiên khi train, loss độ sâu theo điểm SfM?
- Đây là bước **sàng lọc**: một cảnh (`room`), một seed (42). Bảng ghi `seeds=FEW` (hoặc `seeds=DIFF` khi `room` đã có 3 seed của thực nghiệm chính) và không đủ để nói "cải thiện". Một biến thể chỉ được coi là đáng giữ khi hơn `mcmc` rõ ràng hơn độ dao động thường gặp giữa các seed; chưa đo được độ dao động này trên `room` (cần 3 seed của thực nghiệm chính), nên kết luận của bước này là giả thuyết.
- Sau khi có kết quả: ghép các tham số đáng giữ thành một chế độ mới `mcmc_indoor` trong `configs/base.yaml` (cùng preset `mcmc`, cùng `cap_max: {policy: match_baseline}`), chạy xác nhận một lượt trên `room` (thêm `mcmc_indoor` vào `modes` của `tune_mcmc.yaml` rồi chạy lại: chỉ lượt mới được train), rồi đưa vào `main.yaml` với `also_against: [mcmc]` để so trên 6 cảnh × 3 seed.
  - `mcmc_indoor` chỉ đổi đúng một tham số thì trùng cấu hình với biến thể tương ứng: lượt của biến thể đó đã là lượt xác nhận trên `room`, không cần thêm vào `tune_mcmc.yaml` (thêm vào thì kế hoạch in cảnh báo và vẫn train lại).
  - Trên một benchmark, `mcmc_indoor` có thể trùng cấu hình với `mcmc` (vd. chọn `opacity_reg` 0,001, đúng giá trị Deep Blending đã dùng cho `mcmc`): kế hoạch in cảnh báo, cả hai vẫn train và có dòng riêng trong bảng; chênh lệch ở đó chỉ là dao động giữa hai lần train.
- **Thiên lệch khi chọn tham số** (quy tắc ở `docs/design/training_modes.md` mục 4.2): tham số được chọn bằng ảnh test của `room`, nên `room` là cảnh dùng để dò. Kết luận về `mcmc_indoor` dựa trên 5 cảnh còn lại của thực nghiệm chính (Deep Blending 2 cảnh, `counter`, `kitchen`, `bonsai`) và cảnh tự quay. Bảng tự động của thực nghiệm chính vẫn có `room` (cả trong trung bình); bảng không có `room` lập lại từ kết quả đã có, không dùng GPU:
  ```bash
  BASELINE=mcmc MODES="mcmc mcmc_indoor" bash scripts/run_evaluate.sh deep_blending
  BASELINE=mcmc MODES="mcmc mcmc_indoor" SCENES="counter kitchen bonsai" bash scripts/run_evaluate.sh mipnerf360
  ```
  Khi báo cáo, dòng của `room` vẫn ghi nhưng nói rõ là cảnh đã dùng để chọn tham số. Cách chặt hơn (chưa làm): dò trên một phần ảnh train của `room` tách ra làm tập kiểm định.
- Đây là tinh chỉnh tham số của một phương pháp có sẵn (3DGS-MCMC trên gsplat) cộng một bước hậu xử lý, không phải thuật toán mới.

### 3.2 Thiết kế

| Mục | Giá trị |
|---|---|
| Chế độ | `default`, `mcmc` |
| Biến thể của `mcmc` (`variants`) | mỗi biến thể đổi đúng một tham số của `simple_trainer.py` so với preset `mcmc` của gsplat @6e8c837 (`opacity_reg` 0,01, `scale_reg` 0,01, `strategy.noise_lr` 5e5, `strategy.min_opacity` 0,005, `random_bkgd` tắt, `depth_loss` tắt): `or0001` (`opacity_reg` 0,001), `or0005` (0,005), `or005` (0,05), `sr0001` (`scale_reg` 0,001), `sr005` (0,05), `nlr250k` (`strategy.noise_lr` 2,5e5), `nlr1m` (1e6), `mo002` (`strategy.min_opacity` 0,02), `rbg` (`random_bkgd`), `depth` (`depth_loss`, `depth_lambda` mặc định 0,01) |
| Dữ liệu | Mip-NeRF 360 `room` (`images_2`) |
| Seed | 42 |
| Protocol, ngân sách Gaussian | như thực nghiệm chính (mục 2.2): mọi biến thể lấy `cap_max` từ lượt `default` cùng cảnh, cùng seed |
| Bảng so sánh | với `default`, và thêm một bảng so với `mcmc` (`also_against: [mcmc]`) |
| Lọc sau khi train (`post_filter`) | mọi lượt: `min_opacity` 0,02, `max_scale` 0,1 (mục 4); bảng trước / sau khi lọc |

`or005` và `mo002` nhắm vào lớp mờ và floater (giả thuyết, chưa có số liệu). `or005` phạt opacity mạnh hơn preset. Với `mo002`, MCMC dời cả Gaussian có opacity đến 0,02 (bằng ngưỡng lọc ở mục 4) sang chỗ có Gaussian đậm ngay trong lúc train, thay vì để bộ lọc xóa sau khi train; trong gsplat @6e8c837, `strategy.min_opacity` cũng là opacity nhỏ nhất của Gaussian vừa được dời hoặc thêm (`relocate`, `sample_add` trong `gsplat/strategy/ops.py`).

Các khóa của kế hoạch, ngoài các khóa của mục 2 (`src/training/experiment.py` kiểm tra):
- `variants`: danh sách `{mode, tag, set, description}`.
  - `mode` là một chế độ trong `modes`, không phải `default`.
  - `tag`: chữ thường và số, các từ nối bằng `_`.
  - `set`: tham số của `simple_trainer.py`, giá trị là số, true / false hoặc chữ không dấu cách. Khóa protocol và tham số do wrapper quản lý (`max_steps`, `data_factor`, `strategy.cap_max`...) bị từ chối, nên biến thể không đổi được protocol.
  - Thư mục của biến thể `<chế_độ>_<tag>_s<seed>` không được trùng một chế độ khác.
  - Cấu hình của mỗi lượt (preset, protocol, ngân sách Gaussian, tham số; 250000 và 250000.0 là một) được so sau khi ghép quy ước mcmc của benchmark. Hai lượt cùng chế độ mà cùng cấu hình là lỗi, vì bảng so sánh gom lượt theo chế độ + tham số (vd. trên Deep Blending, một biến thể `opacity_reg: 0.001` của `mcmc` trùng với chính `mcmc`). Hai chế độ khác tên mà cùng cấu hình (vd. một chế độ mới không có tham số riêng) chỉ là cảnh báo, in cùng kế hoạch: cả hai vẫn train, mỗi chế độ một dòng trong bảng. Biến thể bằng đúng giá trị mặc định của gsplat (vd. `opacity_reg: 0.01`) thì chưa bắt được: không có giá trị mặc định của gsplat lúc kiểm tra kế hoạch.
  - Thứ tự ghép tham số (ưu tiên tăng dần): preset của gsplat → quy ước mcmc của benchmark (`benchmark.gsplat.mcmc_set`, vd. `opacity_reg=0.001` trên Deep Blending; chỉ cho chế độ preset mcmc) → tham số của chế độ trong `configs/base.yaml` → tham số của biến thể. Chế độ tự đặt khóa của quy ước (vd. `mcmc_indoor` với `opacity_reg` riêng) thì giữ giá trị của nó trên mọi benchmark; khóa khác của quy ước vẫn được thêm.
  - Tag được ghi vào `run_meta.json` (`src.training.run --tag`): bảng so với `mcmc` lấy các lượt không có tag làm mốc. Bảng chỉ đưa vào lượt có tag khi được yêu cầu (`TAGS` của `scripts/run_evaluate.sh`): bảng của kế hoạch này có các biến thể của nó, bảng của thực nghiệm chính (chạy trên cùng thư mục) thì không, kèm một dòng cảnh báo ghi số lượt để ngoài.
- `post_filter: {min_opacity, max_scale}`: sau mỗi seed, lọc mọi lượt đã train xong của seed đó rồi chấm lại (mục 4). 0 là tắt một tiêu chí, không được tắt cả hai.
- Chế độ hoặc biến thể có `depth_loss: true` thêm bước kiểm tra điểm SfM (mục 3.3, bước 3b).

### 3.3 Chạy

```bash
VOLUME_GB=100 bash scripts/pod/run_experiment.sh tune_mcmc --background   # một việc nền: exp_tune_mcmc
bash scripts/pod/run_experiment.sh tune_mcmc --plan                       # chỉ in kế hoạch, các bước, dữ liệu đã có
```

Hoặc ô 7 của notebook với `WHAT = "experiment"`, `EXP = "tune_mcmc"`. Các bước như mục 2.3, thêm:
- 1. `env_check` kiểm tra cả tham số của 10 biến thể với `simple_trainer.py` thật (sai tên tham số thì dừng trước khi tải, chưa tốn GPU).
- 3b. `depth_check_<benchmark>`: mỗi ảnh train của mỗi cảnh có điểm SfM (`src/training/depth_check.py`; Parser của gsplat, không GPU, vài giây). Với `depth_loss`, gsplat @6e8c837 báo `KeyError` khi nạp một ảnh không có điểm nào trong track của SfM: không đạt, và chỉ các lượt có loss độ sâu của benchmark đó bị bỏ (B0 và các lượt khác vẫn chạy; baseline có loss độ sâu thì bỏ mọi lượt gsplat của benchmark đó, vì các lượt khác lấy ngân sách Gaussian từ nó). Ảnh có track nhưng mọi điểm nằm ngoài ảnh thì chỉ cảnh báo: ảnh đó không có giám sát độ sâu, loss ghi ra là NaN ở những vòng gặp ảnh đó, gradient vẫn hữu hạn (thử lại đoạn loss của simple_trainer.py bằng PyTorch trên CPU).
- 5. `gsplat_<benchmark>_s<seed>`: trên mỗi cảnh: `default`, `mcmc`, rồi 10 biến thể theo thứ tự trong kế hoạch.
- 5b. `filter_<benchmark>_s<seed>`: lọc và chấm lại mọi lượt đã xong của seed đó (`src/training/prune.py --skip-unfinished`; lượt train lỗi thì bỏ qua, bước gsplat đã báo lỗi). Lọc lỗi không dừng việc train.
- Lượt `room/default_s42` và `room/mcmc_s42` dùng chung thư mục với thực nghiệm chính: chạy `main` sau đó thì hai lượt này không train lại.

### 3.4 Thời gian và dung lượng

Ước lượng trên RTX 4090, chưa phải số đo (`configs/experiments/tune_mcmc.yaml`):
- Khoảng 5 giờ GPU: 12 lượt × khoảng 19 phút (train, chấm, video; cùng giả định với Mip-NeRF 360 trong `main.yaml`), cộng lọc và chấm lại khoảng 4 phút mỗi lượt (khoảng 4,6 giờ).
- Thêm khoảng 24 GB trên volume: khoảng 1,8 GB mỗi lượt (checkpoint và PLY, ảnh render, video, bản đã lọc), cộng khoảng 2 GB dữ liệu `room` và ảnh PNG thu nhỏ. Lúc tải, `360_v2.zip` (khoảng 12,5 GB) nằm tạm trên volume.
- Chạy cả `tune_mcmc` lẫn `main` trên cùng volume (khoảng 80 GB kết quả cộng 15–20 GB môi trường), rồi thêm `mcmc_indoor`: nên dùng volume 150 GB.

### 3.5 Kết quả và cách đọc

`experiment_tune_mcmc_<thời_gian>.md`: bảng các bước, bảng so với `default`, bảng so với `mcmc` (mỗi biến thể là một dòng `mcmc[<tham số>]`, kèm ΔPSNR / ΔSSIM / ΔLPIPS so với `mcmc` và tỉ lệ thời gian train), và bảng sau khi lọc (mục 4). Đọc theo mục 3.1: một seed chỉ để chọn ứng viên cho `mcmc_indoor`.

## 4. Lọc Gaussian sau khi train

`src/training/prune.py`; trong kế hoạch: khóa `post_filter`; cho một lượt bất kỳ: `bash scripts/run_filter.sh <luồng> <cảnh> [<lượt> ...]` (vd. `custom phong1 mcmc_s42`). **Trạng thái: đã viết, đã kiểm thử trong sandbox (logic lọc bằng numpy; đọc / ghi checkpoint bằng PyTorch 2.7.1 CPU; ghi PLY bằng `export_splats` thật của gsplat @6e8c837; lệnh chấm lại bằng runner giả); chưa chạy trên GPU.**

- **Bỏ** một Gaussian khi tham số không hữu hạn, khi opacity (sau sigmoid) < `min_opacity` (lớp sương mờ, vệt mờ lơ lửng), hoặc khi trục lớn nhất > `max_scale` × `scene_scale`. Mỗi Gaussian chỉ được đếm ở tiêu chí đầu tiên loại nó. Các Gaussian còn lại giữ nguyên tham số, không tối ưu lại.
- **`scene_scale`** đúng như `simple_trainer.py` tính (Parser của gsplat trên thư mục dữ liệu của lượt: khoảng cách lớn nhất từ một camera tới tâm các camera sau khi chuẩn hóa, × 1,1 × `global_scale`), cùng đại lượng mà DefaultStrategy dùng cho ngưỡng `prune_scale3d` = 0,1 lúc densify. Script đối chiếu với dòng `Scene scale:` trong `train.log`; lệch thì dừng. Ngưỡng tính theo camera, nên mọi chế độ trên cùng cảnh dùng cùng một ngưỡng tuyệt đối.
- **Chấm lại** bằng đúng lệnh đánh giá đã ghi của lượt (`run_meta.json`, `commands.eval`), chỉ đổi `--result-dir`, `--ckpt` và hai file ghi lại của `gsplat_launcher`: cùng ảnh test, cùng protocol. Script kiểm tra số Gaussian gsplat thấy bằng số đã giữ lại, và ảnh chấm lại đúng là tập test của lượt.
- **Kết quả** vào `<lượt>/filtered/`: checkpoint và PLY đã lọc, `stats/`, `renders/`, `videos/` của lượt chấm lại, `filter_meta.json` (ngưỡng, `scene_scale`, số Gaussian bỏ theo từng tiêu chí, SHA-256 checkpoint gốc, lệnh). Lượt gốc không bị sửa.
- **Chạy lại:** cùng ngưỡng và cùng checkpoint gốc thì bỏ qua; ngưỡng khác (hoặc lần trước lỗi) thì thư mục cũ đổi tên `filtered.prev_<thời_gian>`. Mô hình COLMAP của cảnh đổi sau khi train thì dừng.
- **Báo cáo:** `src.evaluation.compare_modes` thêm bảng sau khi lọc: chỉ số của bản đã lọc, chênh lệch so với chính lượt đó trước khi lọc (cùng ảnh test), số Gaussian giữ lại, PLY trước → sau; có thêm cột ngưỡng khi các dòng lọc với ngưỡng khác nhau. Mọi bảng khác là mô hình chưa lọc. Bảng sau khi lọc chỉ có các lượt đã lọc: vd. chạy `main` sau `tune_mcmc` thì bảng của Mip-NeRF 360 có thêm phần này cho riêng `room` (`default_s42`, `mcmc_s42` đã lọc trong `tune_mcmc`).
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

Biến thể của một kế hoạch (mục 3.2) là một dòng riêng `<chế_độ>[<tham số>]`, cùng các kiểm tra công bằng như một chế độ. Bảng sau khi lọc (mục 4) chỉ có khi có lượt đã lọc.

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
| Lọc Gaussian và chấm lại (mục 4) | **Đã viết, chưa chạy trên GPU**; logic lọc, checkpoint PyTorch và PLY của gsplat kiểm tra trên CPU | `tests/unit/test_prune.py`, `test_compare_modes.py`, `test_gsplat_results.py` |
