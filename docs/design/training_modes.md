# Thiết kế: hai chế độ huấn luyện 3DGS — Default (baseline) và MCMC

_09/10/2026, cập nhật 24/10/2026 · Trạng thái: `default` và `mcmc` **đã chạy trên GPU** với Deep Blending bằng repo trước (17/10/2026, RTX 4090, seed 42; số liệu ở `docs/ENVIRONMENT_AUDIT.md` mục 7); repo này dùng cùng mã huấn luyện, chưa chạy trên pod. Mới một seed nên chưa kết luận về `mcmc`. Mip-NeRF 360 đã cấu hình, chưa chạy. Thực nghiệm chính (3 seed, 6 cảnh trong nhà): `docs/design/experiments.md` mục 2._

## 1. Mục tiêu và ranh giới

- **Baseline:** 3DGS với `DefaultStrategy` của gsplat. Đây là cài đặt lại heuristic densification của 3DGS gốc (Kerbl et al., 2023), tương ứng B1 trong `docs/PROJECT_PLAN.md` (mục 3.3).
- **Chế độ thứ hai:** 3DGS-MCMC (Kheradmand et al., NeurIPS 2024) qua `MCMCStrategy` của gsplat, lấy từ notebook demo của nhóm. Kết quả kiểm tra demo: `docs/research/demo_review.md`.
- **Ranh giới trung thực:** MCMC là phương pháp đã công bố và đã có sẵn trong gsplat.
  - Đưa nó vào pipeline và đánh giá trên dữ liệu nội thất quay bằng điện thoại thuộc loại đóng góp (b) ở `docs/PROJECT_PLAN.md` mục 3.4.
  - Đây không phải thuật toán mới. Đổi `cap_max` hay hệ số regularization cũng không phải đóng góp thuật toán.
- **Quan hệ với B0:** bản Inria (B0, `scripts/run_train.sh deep_blending b0`) vẫn là mốc đối chiếu.
  - Default trên gsplat chỉ được dùng làm baseline sau khi chứng minh số liệu tương đương B0 trên Deep Blending, theo điều kiện đã ghi ở mục 3.3 của kế hoạch.
  - Bước cuối của `scripts/run_train.sh deep_blending gsplat` (`scripts/pipeline/train_gsplat_modes.sh`) làm việc đối chiếu này, theo hai cách:
    - đối chiếu từng cảnh với lượt B0 tự chạy (`--b0-root`, cùng tập test), là điều kiện chính;
    - đối chiếu trung bình với số liệu Inria công bố.

## 2. Vị trí trong pipeline

```
ảnh / video ─ frames.py ─ colmap_runner.py (COLMAP 4.2.1, undistort) ─▶ <cảnh>/images + sparse/0   (dùng chung)
                                                                   │
              ┌────────────────────────────────────────────────────┤
              ▼                                                    ▼
  B0: Inria train.py/render.py/metrics.py            gsplat simple_trainer.py @ 6e8c837
      (đối chiếu số liệu công bố)                    qua src.training.run + gsplat_launcher
                                                       ├─ default    baseline
                                                       └─ mcmc       cùng protocol, cùng số Gaussian cuối với default
                                                                   │  summary.json mỗi lượt chạy
                                                                   ▼
                                                     src.evaluation.compare_modes → bảng + kiểm tra điều kiện
```

Mọi chế độ dùng cùng một thư mục cảnh: cùng frame, cùng pose camera và cùng điểm SfM để khởi tạo. Khác biệt giữa các chế độ chỉ nằm ở chiến lược quản lý Gaussian và tham số đi kèm.

## 3. Các chế độ (`configs/base.yaml`, mục `train_modes`)

Giá trị dưới đây đọc từ cấu hình mà `simple_trainer.py` thật (gsplat `6e8c837`) phân giải ra, xem mục 8.

| Chế độ | Preset gsplat | Tham số chính | Số Gaussian | Dùng cho |
|---|---|---|---|---|
| `default` | `default` | `DefaultStrategy`: densify mỗi 100 vòng, từ vòng 600 đến 14.900; ngưỡng gradient 0,0002; reset opacity ở các vòng 3.000, 6.000, 9.000, 12.000; opacity khởi tạo 0,1 | Do heuristic quyết định | Baseline |
| `mcmc` | `mcmc` | `MCMCStrategy`: relocate và thêm Gaussian mỗi 100 vòng, từ vòng 600 đến 24.900 (mỗi lần thêm 5% cho tới `cap_max`); `noise_lr` 5e5; ngưỡng "chết" opacity 0,005. Opacity khởi tạo 0,5; hệ số scale khởi tạo 0,1 (nhân với căn trung bình bình phương khoảng cách tới 3 điểm SfM gần nhất); `opacity_reg` = `scale_reg` = 0,01 | `cap_max` = số Gaussian cuối của lượt `default` cùng cảnh, cùng seed, cùng protocol, cùng tập test. Không có số cố định: chưa có lượt `default` thì `mcmc` không chạy (giá trị mặc định 1.000.000 của `MCMCStrategy` không bao giờ được dùng) | So sánh công bằng với baseline |

**Đổi tham số** (mọi thay đổi được ghi vào `run_meta.json`):
- Cho một lượt, không sửa cấu hình, luôn kèm `TAG` để có thư mục riêng `<chế_độ>_<TAG>_s<seed>`:
  - `--set khóa=giá_trị`: tham số riêng của chế độ, ví dụ `--set opacity_reg=0.001` (giá trị paper dùng cho Deep Blending). Không phải sai khác protocol, nên `mcmc` vẫn so được với `default` chuẩn, thành một biến thể riêng (`mcmc[opacity-reg=0.001]`);
  - `--protocol khóa=giá_trị`: đổi protocol (`max_steps`, `antialiased`...). Bị ghi là sai khác; muốn so thì chạy cả `default` và `mcmc` với cùng `TAG` và cùng protocol (`mcmc` lấy ngân sách từ `default_<TAG>`);
  - `--cap-max N`: đặt tay số Gaussian cho `mcmc`.
- Nhiều bộ tham số trong một thực nghiệm (vd. dò tham số `mcmc`): mục `variants` của kế hoạch (`configs/experiments/<tên>.yaml`, `docs/design/experiments.md` mục 3.2). Mỗi biến thể chạy như một lượt `--set` vào thư mục `<chế_độ>_<tag>_s<seed>`, cùng ngân sách Gaussian, không cần thêm chế độ vào `configs/base.yaml`.
- Dùng lại nhiều lần: thêm một chế độ có tên vào `train_modes.modes` (mẫu ghi trong chú thích của `configs/base.yaml`): `preset`, `args`, tùy chọn `protocol`, và `cap_max` theo `match_baseline` hoặc `fixed`. Sửa trong repo rồi commit, không sửa trên pod (ô 3 của notebook sẽ không cập nhật được code). Chế độ `mcmc_indoor` dự kiến được thêm theo cách này, sau khi có kết quả của `tune_mcmc`. Trên benchmark có quy ước mcmc (`benchmark.gsplat.mcmc_set`, vd. `opacity_reg=0.001` trên Deep Blending), tham số mà chế độ tự đặt được giữ, quy ước chỉ thêm các khóa chế độ không đặt.
- Lọc Gaussian sau khi train (`src/training/prune.py`, `docs/design/experiments.md` mục 4) là hậu xử lý trên checkpoint đã có, không phải một chế độ: không đổi lượt train, bản đã lọc nằm trong `<lượt>/filtered/` và có bảng riêng.
- Thay đổi không có sẵn dưới dạng tham số của `simple_trainer.py` (Kill-then-Relocate, depth prior): biến thể có code, bật bằng khóa `sr3d.kr` / `sr3d.mono` như một tham số (mục 3.1).

### 3.1 Biến thể có code (`sr3d.kr`, `sr3d.mono`)

Hai thành phần của notebook v2 không có trong gsplat: Kill-then-Relocate (`kr`) và depth prior (`mono`); thiết kế, hằng số, nguồn và khác biệt với notebook: `docs/design/experiments.md` mục 3.7. Cách gắn vào trainer, không sửa file nào của gsplat:

- **Bật:** khóa `sr3d.kr: true` / `sr3d.mono: true` trong `args` của chế độ, `set` của biến thể, hoặc `--set sr3d.kr=true` (`src/training/profiles.py`, `HOOK_KEYS`). Khóa nằm trong `args` như mọi tham số, nên đi cùng đường: `run_meta.json`, so cấu hình giữa các lượt, tên dòng trong bảng (`mcmc[sr3d.kr=True]`). Chỉ nhận `true`; khóa `sr3d.*` khác bị từ chối; `kr` chỉ cho preset `mcmc`. `trainer_args` không đưa các khóa này sang `simple_trainer.py`.
- **Chạy:** `src.training.run` thêm `--hook kr|mono` cho launcher ở bước train (bước đánh giá vẫn chạy file gốc: hook không đổi gì khi đánh giá). `gsplat_launcher` đọc `examples/simple_trainer.py`, chèn 4 dòng tại 4 chỗ cố định (`src/training/trainer_hooks.py`, `ANCHORS`): `render_mode` (thêm độ sâu khi cần prior), giữ bản đồ độ sâu render, cộng loss của prior trước TV loss của bilateral grid, gọi KR ngay trước `MCMCStrategy.step_post_backward`. Mỗi chỗ phải xuất hiện đúng một lần, không thì dừng trước khi train (gsplat khác bản ghim). Bản có vá ghi vào `<lượt>/sr3d_simple_trainer.py` rồi được chạy với một đối tượng `Hooks` đặt sẵn trong biến toàn cục `_SR3D_HOOKS`; các module của gsplat vẫn import từ `examples/` như với file gốc. `resolved_config.json` ghi thêm `_launcher.hooks`: tên hook, hằng số, SHA-256 của file gốc (có phải bản ghim không) và của bản vá.
- **Depth prior:** lượt có `sr3d.mono` có thêm bước `mono` trước bước train (`src/training/mono_depth.py`, log `mono_depth.log`), cần Depth Anything V2 Small ở `$WS/code/depth_anything_v2` (`scripts/setup/setup_depth_anything.sh`). `scripts/pod/run_experiment.sh` gọi script đó ở bước riêng `depth_anything`, trước `env_check`, khi kế hoạch có lượt `sr3d.mono`; bước đó lỗi thì chỉ bỏ các lượt dùng prior, các lượt khác vẫn train (baseline dùng prior thì không train lượt gsplat nào). Thiếu trọng số thì `run.py` dừng trước khi ghi gì; code không đúng commit ghim hoặc đã bị sửa thì bước `mono` lỗi khi phải tính prior (cache của cảnh đã đủ thì được dùng lại, không nạp mạng: dấu của cache gồm commit và SHA-256 của trọng số); prior không khớp điểm SfM thì bước `mono` lỗi, lượt không train.
- **Kiểm NaN:** lượt có hook kiểm vị trí, scale, quaternion, opacity của mọi Gaussian mỗi 500 vòng (`Hooks.guard`, gọi qua dòng chèn của loss): có giá trị không hữu hạn thì lượt dừng với `FloatingPointError` và ghi vào nhật ký. Không tự tắt thành phần rồi train lại như notebook. Lượt không có hook (kể cả `bilagrid`) không được kiểm giữa chừng, như mọi lượt của gsplat: chỉ số cuối không hữu hạn thì bảng so sánh ghi lượt là lỗi.
- **Kiểm tra trước khi train:** `env_check` của `run_experiment.sh` (`scripts/setup/smoke_test_gsplat.py`) phân giải tham số của mọi lượt với bản sao có vá, rồi chạy thử trên GPU với tensor nhỏ: KR đánh dấu đúng Gaussian quá lớn và bước relocate thật của `MCMCStrategy` dời chúng, loss của prior có gradient hữu hạn, kiểm NaN dừng lượt; kế hoạch có `sr3d.mono` thì chạy thử Depth Anything V2 trên ảnh tổng hợp.
- **Định danh lượt:** hằng số của hook được ghi vào `run_meta.json` (`hooks`) và so khi quyết định bỏ qua lượt đã xong: đổi hằng số trong code thì lượt cũ không còn là cùng cấu hình.
- **Nhật ký:** `<lượt>/sr3d_hooks.jsonl` (KR: số Gaussian bị dời mỗi lần và lý do; mono: loss trung bình mỗi 500 vòng; kiểm NaN: tham số không hữu hạn, nếu có).

## 4. Giao thức so sánh

### 4.1 Protocol chung

| Khóa | Giá trị | Lý do |
|---|---|---|
| `data_factor` | 1 | Giữ độ phân giải gốc. Inria tự thu ảnh rộng hơn 1.600 px về 1.600 px, gsplat thì không. Vì vậy cảnh tự quay nên trích frame với `--max-side 1600`; `src.training.run` cảnh báo nếu ảnh rộng hơn |
| `test_every` | 8 | Cùng quy tắc với Inria (`llffhold=8`) và gsplat: sắp xếp tên ảnh, ảnh thứ i là test khi i % 8 == 0. Đã đối chiếu mã nguồn cả hai, nên hai bản dùng **cùng tập test** |
| `max_steps` | 30.000 | Như Inria và code chính chủ 3DGS-MCMC |
| `lpips_net` | `vgg` | Như Inria; gsplat mặc định AlexNet nên phải đặt rõ |
| `antialiased` | `false` | Inria mặc định tắt; gsplat ghi chú antialiased có thể làm giảm nhẹ chỉ số |
| `save_ply` | `true` | Đo dung lượng mô hình; dùng cho viewer |
| `disable_video` | `false` | Video quỹ đạo để đánh giá định tính |
| `init_type` | `sfm` | Mọi chế độ khởi tạo từ điểm SfM, như B0. Code chính chủ 3DGS-MCMC mặc định khởi tạo ngẫu nhiên; paper báo cả hai kiểu (Bảng 1, Bảng 5), nên khi đối chiếu với paper phải lấy các dòng khởi tạo SfM |

Đổi một khóa protocol (trong file cấu hình hoặc bằng `--protocol`) sẽ được ghi thành `deviations`. Bảng so sánh đánh dấu lượt chạy đó là khác điều kiện.

**Quy ước của benchmark** (`--dataset-protocol`, hiện chỉ nhận `data_factor`):
- Mip-NeRF 360 dùng ảnh thu nhỏ có sẵn: `data_factor 2` cho 4 cảnh trong nhà, `4` cho 5 cảnh ngoài trời. Đây là quy ước chính chủ của cả Inria (`full_eval.py`) lẫn gsplat (`benchmarks/basic.sh`).
- Giá trị này thay protocol gốc của lượt chạy, nên không bị ghi là sai khác. Mọi chế độ trên cùng benchmark dùng chung quy ước, nên vẫn so được với nhau và với số liệu công bố.
- Lượt `data_factor 1` (Deep Blending, cảnh của app) không đổi gì; `run_meta.json` của chúng giữ nguyên như trước.
- Với `data_factor` > 1, gsplat đọc một bản xem dữ liệu (liên kết tới cảnh gốc) để không ghi `images_N_png/` vào bộ dữ liệu (`experiments/mipnerf360/README.md`).

### 4.2 Các quy tắc khác

- **Cùng ngân sách Gaussian.** Chất lượng của MCMC phụ thuộc trực tiếp vào `cap_max`. Paper so sánh ở cùng số Gaussian với 3DGS (mục 4.1, Bảng 1 và Bảng 5), nên chế độ `mcmc` lấy `cap_max` bằng số Gaussian cuối của lượt `default` tương ứng.
  - `run.py` từ chối nếu lượt baseline chưa xong, sai chế độ, khác protocol, có sai khác protocol, khác seed hoặc dùng tập test khác.
  - `compare_modes` ghi `gaussian_budget: EQUAL/UNEQUAL`.
- **Đánh giá tách khỏi huấn luyện**, giống script benchmark của gsplat.
  - Khi huấn luyện: `--eval-steps -1`. Sau đó một lệnh riêng nạp checkpoint cuối (`--ckpt`) với đúng cấu hình đã huấn luyện.
  - Nhờ vậy thời gian huấn luyện không gồm thời gian đánh giá, và mọi chế độ dùng cùng một đoạn mã tính chỉ số.
- **Một GPU, cùng loại GPU.**
  - Nếu thấy nhiều GPU, gsplat tự chia sang huấn luyện phân tán (tăng batch thực tế). Vì vậy `run.py` đặt `CUDA_VISIBLE_DEVICES` về một GPU, còn launcher từ chối khi thấy nhiều hơn một.
  - Tên GPU được ghi vào kết quả và có trong mục kiểm tra điều kiện.
- **Seed.** `simple_trainer.py` gán cứng seed 42. Launcher cho phép đổi seed để chạy lặp. Trước khi nói "cải thiện" cần ít nhất 3 seed trên một tập cảnh (`docs/PROJECT_PLAN.md` mục 4.2).
  - Seed cố định thứ tự ảnh huấn luyện (DataLoader xáo trộn), cùng bộ sinh số ngẫu nhiên dùng cho nhiễu và relocate của MCMC và cho bước tách Gaussian của Default.
  - Một số kernel CUDA không tất định (cộng dồn gradient song song), nên hai lần chạy cùng seed vẫn có thể lệch nhau một chút.
- **Tập test của cảnh tự quay.** Frame test xen kẽ trong cùng một video chỉ đo khả năng nội suy (xem demo_review mục 3.3).
  - `--test-list <file>` (mỗi dòng một tên ảnh, cùng định dạng `sparse/0/test.txt` của Inria) cho phép đánh giá trên đoạn quay kiểm tra riêng.
  - Đoạn quay kiểm tra phải được đăng ký trong cùng model COLMAP với video chính để có pose. Thêm bước chuẩn bị dữ liệu này vào `prepare_scene.sh` thuộc mốc M4.
  - Khi đó điểm SfM có thể chỉ được dựng nhờ ảnh test. `run.py` tạo bản dữ liệu `<lượt chạy>/data/`: ảnh và pose giữ nguyên, nhưng `points3D.bin` chỉ giữ điểm được ít nhất 2 ảnh train khác nhau quan sát, nên Gaussian không được khởi tạo từ thông tin riêng của ảnh test. Số điểm giữ lại được ghi vào `run_meta.json`.
  - gsplat vẫn chuẩn hóa hệ tọa độ theo mọi camera, kể cả camera test. Việc này chỉ dùng pose, không dùng nội dung ảnh test; các benchmark công khai cũng làm như vậy.
  - Với B0, code Inria chưa đọc được danh sách này; quyết định ở M4 (xem `docs/ENVIRONMENT_AUDIT.md` mục 6).
- **Không chỉnh tham số trên ảnh test của dữ liệu dùng để kết luận.** Chọn `opacity_reg`, `cap_max`... theo một trong hai cách:
  - trên một cảnh hay bộ dữ liệu không dùng để kết luận. Ví dụ: `tune_mcmc` dò trên `room` (`docs/design/experiments.md` mục 3). Kết luận về chế độ chọn ra dựa trên các cảnh còn lại và cảnh tự quay (bảng không có `room`: lệnh ở mục 3.1 của tài liệu đó); dòng của `room` vẫn báo cáo nhưng ghi rõ là cảnh đã dùng để chọn tham số;
  - trên một phần ảnh train tách ra làm tập kiểm định.

  Không chọn theo chỉ số trên ảnh test rồi báo cáo chính chỉ số đó như kết quả.
- **Một thư mục kết quả cho một cấu hình.**
  - `run.py` chỉ bỏ qua lượt đã xong khi preset, protocol, tham số (kể cả hằng số của biến thể có code, mục 3.1), seed, `cap_max` và tập test đều giống yêu cầu mới; khác thì dừng và yêu cầu `--force` (thư mục cũ được đổi tên, không xóa).
  - `train_scene.sh` bắt buộc đặt `TAG` khi dùng `--protocol` hoặc `--set`, để lượt chạy thử không ghi vào thư mục của lượt chuẩn.
  - Lượt dở dang (pod bị dừng giữa chừng) được chạy lại với `--rerun-incomplete`; hai script trên pod luôn bật tùy chọn này.
  - `--replace-stale`: lượt OK làm từ dữ liệu cũ thì được thay thay vì báo lỗi. Dữ liệu cũ nghĩa là mô hình COLMAP của cảnh đã đổi (cả lượt lỗi thời), hoặc chỉ ngân sách Gaussian khác khi cả lượt cũ lẫn yêu cầu mới đều theo chính sách `match_baseline` (lượt baseline đã train lại). Khác ở preset, protocol, tham số, seed, `--test-list` hay `--cap-max` vẫn cần `--force`. `scripts/pod/run_scene.sh` dùng tùy chọn này cho mọi chế độ, `scripts/pipeline/train_gsplat_modes.sh` dùng cho `mcmc`, để lượt `mcmc` luôn đi cùng lượt `default` hiện tại.

### 4.3 Khi nào được nói "cải thiện"

Chỉ khi đủ tất cả các điều kiện sau:

- mọi mục kiểm tra trong báo cáo `compare_modes` là OK, và ngân sách Gaussian là EQUAL;
- các chế độ được so dùng cùng một bộ seed, ít nhất 3 seed;
- báo cáo trung bình ± độ lệch chuẩn theo từng cảnh, kèm chênh lệch theo từng cảnh, không chỉ trung bình gộp;
- chênh lệch nhất quán trên các cảnh. Nếu không nhất quán, nêu rõ cảnh nào tốt hơn, cảnh nào kém hơn.

Kết quả không cải thiện vẫn được báo cáo đầy đủ (`docs/PROJECT_PLAN.md`, rủi ro R13).

## 5. Thành phần

| File | Vai trò |
|---|---|
| `configs/base.yaml` (mục `train_modes`) | Protocol chung, hai chế độ (kèm mẫu thêm chế độ riêng), commit gsplat mong đợi, seed mặc định |
| `src/training/profiles.py` | Đọc cấu hình, ghép protocol và tham số; chặn tham số do wrapper quản lý; sinh dòng lệnh cho `simple_trainer.py` (huấn luyện và đánh giá); chính sách `cap_max` |
| `src/training/split.py` | Chia train/test theo đúng quy tắc của Inria và gsplat, hoặc theo danh sách; ghi `split.json` kèm SHA-256 |
| `src/training/gsplat_launcher.py` | Chạy `simple_trainer.py` **không sửa file nào của gsplat**: đổi seed, áp tập test cho trước, ghi tập ảnh thực sự dùng và cấu hình đã phân giải; `--resolve-only` để kiểm tra tham số không cần GPU; `--hook` chạy bản sao có vá của trainer (mục 3.1) |
| `src/training/trainer_hooks.py` | Biến thể có code: vá bản sao của `simple_trainer.py` tại 4 chỗ cố định; Kill-then-Relocate; loss Pearson của depth prior (mục 3.1) |
| `src/training/mono_depth.py`, `scripts/setup/setup_depth_anything.sh` | Depth prior của biến thể `mono`: Depth Anything V2 Small (code và trọng số ghim), kiểm với điểm SfM, cache theo cảnh (`docs/design/experiments.md` mục 3.7) |
| `src/training/run.py` | Điều phối một lượt chạy: kiểm tra cảnh → split → huấn luyện → đánh giá → `summary.json`; ghi lệnh, commit, GPU, thời gian vào `run_meta.json`; `--dry-run` |
| `src/evaluation/gsplat_results.py` | Đọc kết quả gsplat (`stats/*.json`, checkpoint, PLY, nhật ký GPU) thành một bản tóm tắt |
| `src/evaluation/compare_modes.py` | Nhóm các lượt chạy theo (cảnh, biến thể). Biến thể = chế độ + tham số riêng; lượt sai khác protocol bị loại. Lượt đã lọc Gaussian: thêm bảng trước / sau khi lọc.<br>Tính trung bình ± độ lệch chuẩn theo seed và chênh lệch so với baseline (`default`, hoặc một chế độ khác, vd. `--baseline mcmc` để so một chế độ tự thêm với `mcmc`).<br>Kiểm tra điều kiện: protocol, tập test, commit, GPU, bộ seed, ngân sách.<br>Đối chiếu baseline `default` với số liệu công bố (chỉ khi đủ mọi cảnh của benchmark; một phần cảnh thì ghi SKIPPED) và với lượt B0 tự chạy, từng cảnh |
| `scripts/setup/setup_gsplat.sh`, `configs/environment/gs-gsplat.conf`, `configs/environment/gs-gsplat.in`, `configs/environment/gs-gsplat.lock.txt` | Env `gs-gsplat`: gsplat `6e8c837` (build sẵn extension), fused-ssim `328dc98`, phụ thuộc ghim theo ngày |
| `scripts/setup/smoke_test_gsplat.py` | Kiểm tra trên GPU: extension, rasterization, hai strategy, fused-ssim, LPIPS-VGG; `simple_trainer.py` thật nhận đúng tham số của mọi chế độ. `--modes ... [--variants ...]`: chỉ phần tham số của các chế độ và biến thể, vài giây (bước đầu của `scripts/pod/run_experiment.sh`) |
| `scripts/run_train.sh custom` → `scripts/pipeline/train_scene.sh` | Một cảnh tự chụp, một chế độ mỗi lượt; tự tìm lượt `default` cùng seed cho `mcmc` |
| `scripts/run_train.sh <benchmark> gsplat` → `scripts/pipeline/train_gsplat_modes.sh` | Deep Blending hoặc Mip-NeRF 360: `default` → các chế độ khác (`MODES`, mặc định `mcmc`; cùng ngân sách) → các biến thể (`VARIANTS`) → bảng so sánh; đối chiếu `default` với lượt B0 tự chạy (từng cảnh) và với số liệu Inria công bố |
| `configs/experiments/main.yaml`, `src/training/experiment.py`, `scripts/pod/run_experiment.sh` | Thực nghiệm chính trong một việc nền: tải dữ liệu còn thiếu, B0, gsplat theo từng seed, bảng so sánh (`docs/design/experiments.md` mục 2) |
| `configs/experiments/tune_mcmc.yaml` | Dò tham số `mcmc` trên `room`: 16 biến thể (12 biến thể đổi một tham số; `bilagrid`, `kr`, `mono` thêm từng thành phần của notebook v2; gói `v2`); lọc Gaussian sau khi train; các ứng viên được ghép ở vòng ghép, 2 seed mới (`docs/design/experiments.md` mục 3, 3.6, 3.7) |
| `src/training/depth_check.py` | Trước khi train lượt có `depth_loss`: mọi ảnh train có điểm SfM nằm trong ảnh (Parser của gsplat, không GPU) |
| `src/training/prune.py`, `scripts/run_filter.sh` | Lọc Gaussian của một lượt đã xong (opacity thấp, trục quá lớn so với `scene_scale` của gsplat) rồi chấm lại bằng lệnh đánh giá đã ghi; kết quả trong `<lượt>/filtered/` (`docs/design/experiments.md` mục 4) |
| `scripts/run_evaluate.sh` | Lập lại bảng so sánh từ kết quả đã có (benchmark hoặc cảnh tự chụp), mọi chế độ đã chạy (`MODES`, `SCENES`, `BASELINE` để chọn), không train |
| `src/rendering/render.py`, `scripts/run_render.sh` | Render lại ảnh test và video quỹ đạo từ checkpoint của một lượt đã xong (thư mục mới, không train, không ghi đè); xuất PLY của bước cuối vào `outputs/checkpoints/` |

## 6. Cách chạy trên pod

```bash
cd /workspace/SpatialRoom3D
bash scripts/setup/setup_gsplat.sh                         # một lần: env gs-gsplat (scripts/pod/run_all.sh đã gồm bước này)
bash scripts/run_train.sh deep_blending gsplat             # kiểm chứng trên Deep Blending (SEEDS="42 43 44" để chạy lặp)
VOLUME_GB=100 bash scripts/pod/run_experiment.sh main --background   # thực nghiệm chính, một việc nền
VOLUME_GB=<dung lượng volume> bash scripts/pod/run_experiment.sh tune_mcmc --background   # dò tham số mcmc trên room, một việc nền
bash scripts/run_train.sh custom phong01 default           # cảnh tự chụp, đã qua scripts/prepare_data.sh custom
bash scripts/run_train.sh custom phong01 mcmc              # cap_max = số Gaussian cuối của default_s42
TAG=or001 BASELINE_TAG= bash scripts/run_train.sh custom phong01 mcmc -- --set opacity_reg=0.001   # đổi tham số mcmc
TAG=thu bash scripts/run_train.sh custom phong01 default -- --protocol max_steps=3000   # chạy thử, thư mục riêng
bash scripts/run_filter.sh custom phong01 mcmc_s42         # lọc Gaussian rồi chấm lại: mcmc_s42/filtered/
. /workspace/miniforge3/etc/profile.d/conda.sh && conda activate gs-gsplat
python -m src.evaluation.compare_modes --root /workspace/outputs/custom_scenes \
    --modes default mcmc --include-deviating --markdown /workspace/outputs/reports/modes_scenes.md
```

Lượt đổi protocol (vd. `TAG=thu ... --protocol max_steps=3000`) chỉ hiện trong bảng khi có `--include-deviating`, và luôn là biến thể riêng.

Mỗi lượt chạy ghi vào `$OUT_ROOT/<cảnh>/<chế_độ>[_<TAG>]_s<seed>/`:

| File | Nội dung |
|---|---|
| `run_meta.json` | Chế độ, protocol, sai khác, lệnh đã chạy, `cap_max` và nguồn của nó, commit repo/gsplat, GPU, thời gian huấn luyện và đánh giá, cảnh báo |
| `split.json`, `split_applied.json`, `split_applied_eval.json` | Tập train/test dự kiến và tập gsplat thực sự dùng khi huấn luyện / đánh giá |
| `resolved_config.json`, `resolved_config_eval.json` | Toàn bộ cấu hình `simple_trainer.py` sau khi phân giải, kể cả tham số strategy |
| `train.log`, `eval.log`, `gpu_log.csv` | Log và nhật ký `nvidia-smi` mỗi 2 giây |
| `stats/`, `ckpts/`, `ply/`, `renders/`, `videos/`, `tb/` | Output gốc của gsplat |
| `data/` | Chỉ có khi dùng `--test-list`: bản dữ liệu đã lọc điểm SfM (mục 4.2) |
| `summary.json` | PSNR/SSIM/LPIPS, số Gaussian, thời gian (cả tiến trình và riêng vòng lặp huấn luyện), VRAM đỉnh (nvidia-smi và PyTorch), thời gian render mỗi ảnh, dung lượng PLY/checkpoint, kết quả kiểm tra tập ảnh |
| `filtered/` | Chỉ có sau khi lọc Gaussian (`src/training/prune.py`): checkpoint và PLY đã lọc, `stats/`, `renders/`, `videos/` của lượt chấm lại, `filter_meta.json` |
| `sr3d_simple_trainer.py`, `sr3d_hooks.jsonl` | Chỉ có ở lượt có biến thể có code (mục 3.1): bản sao có vá của trainer đã chạy, nhật ký của hook |
| `mono_depth.json`, `mono_depth.log` | Chỉ có ở lượt có `sr3d.mono`: kết quả kiểm depth prior trên ảnh train (số ảnh dùng prior, trung vị tương quan với điểm SfM) và log của bước `mono`; bản thân prior nằm ở `<thư mục của cảnh>/.mono_depth_<dấu mô hình COLMAP>/`, dùng chung cho mọi lượt của cảnh |

Dung lượng: checkpoint và PLY mỗi file khoảng 236 byte mỗi Gaussian (59 số thực). Một lượt 2,5 triệu Gaussian tốn khoảng 1,2 GB trên network volume.

## 7. Khác biệt đã biết giữa Default (gsplat) và B0 (Inria)

Đây là lý do phải đối chiếu bằng số liệu, không mặc định hai bản tương đương:

- **Hàm mất SSIM:** gsplat gọi `fused_ssim(..., padding="valid")`, Inria dùng padding mặc định ("same"). Commit fused-ssim cũng khác nhau (`328dc98` và `1272e21`).
- **Chuẩn hóa cảnh:** gsplat chuẩn hóa hệ tọa độ theo toàn bộ camera và trục chính của đám điểm (`normalize_world_space`). Inria tính bán kính cảnh từ camera huấn luyện.
- **Mã tính chỉ số:** gsplat dùng torchmetrics (PSNR, SSIM, LPIPS-VGG với `normalize=False`; gsplat ghi đây là cách tương đương code Inria). B0 dùng `metrics.py` của Inria.
  - So sánh `default` với `mcmc` dùng cùng một đoạn mã nên không bị ảnh hưởng.
  - So sánh B1 với B0 có thể lệch nhỏ do mã tính, nên dùng ngưỡng chấp nhận như ở `docs/ENVIRONMENT_AUDIT.md` mục 5.

## 8. Đã kiểm chứng và chưa

Các dòng ghi ngày 09/10/2026 là kiểm tra của repo trước, trên cùng mã huấn luyện (nay ở `src/training/`).

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Logic cấu hình, split, điều phối, tổng hợp, so sánh | Đã kiểm thử (CPU) | Repo trước, 09/10/2026: 41 test mới (tổng 66). Repo này, 19/10/2026: `tests/unit/test_train_*.py`, `test_gsplat_launcher.py`, `test_gsplat_results.py`, `test_compare_modes.py` (số test và kết quả lượt chạy gần nhất: `docs/PROJECT_STATUS.md`, mục "Kiểm thử trong sandbox"). Dùng gsplat giả lập và runner giả |
| File khóa phụ thuộc | Đã chạy (CPU), repo trước | Trong sandbox ngày 09/10/2026: cài vào Python 3.10 cùng torch 2.7.1 (PyPI). `examples/simple_trainer.py` của gsplat `6e8c837` import được |
| Tham số sinh ra so với `simple_trainer.py` thật | Đã chạy (CPU), repo trước, 09/10/2026 | Launcher `--resolve-only` cho các chế độ (huấn luyện và đánh giá). Strategy, `cap_max`, `max_steps`, `eval_steps [-1]`, `lpips_net`, antialiased ra đúng như mục 3 |
| Áp tập test và seed lên `Parser`/`Dataset` thật | Đã chạy (CPU), repo trước, 09/10/2026 | Cảnh COLMAP tổng hợp 17 ảnh, đọc bằng fork pycolmap thật. Quy tắc mỗi ảnh thứ 8 và danh sách cho trước đều cho đúng tập ảnh; seed 42 → 7 đúng. Bản dữ liệu đã lọc điểm (`data/`) đọc được: 185/272 điểm |
| Gói gsplat đang cài trùng mã nguồn checkout | Đã chạy (CPU), repo trước, 09/10/2026 | Launcher so SHA-256 của `strategy/{default,mcmc,ops}.py` và `rendering.py`; bản cài từ checkout `6e8c837` khớp. Test giả lập cho trường hợp lệch |
| Rà soát độc lập | Đã làm (repo trước, 09/10/2026) | Một lượt rà soát riêng (không tham gia viết code) đã chỉ ra 4 vấn đề lớn: bỏ qua lượt chạy khác cấu hình; trộn các biến thể khi so sánh; điểm SfM chỉ có nhờ ảnh test; đối chiếu với số công bố thay vì với B0 tự chạy. Cả 4 vấn đề đã được sửa, kèm test; các góp ý nhỏ đã được sửa hoặc ghi thành quy tắc ở mục 4 |
| Script shell | Đã kiểm tra tĩnh; đã chạy giả lập | `shellcheck` sạch. `scripts/pipeline/train_scene.sh` (`tests/unit/test_train_scene.py`) và `scripts/pipeline/train_gsplat_modes.sh` (`tests/unit/test_benchmarks.py`) chạy được với conda, `nvidia-smi` và `src.training.run` giả |
| Biến thể tham số trong kế hoạch, kiểm tra điểm SfM, lọc Gaussian và chấm lại | Đã viết, đã kiểm thử bằng stub và runner giả (21/10/2026); checkpoint PyTorch và PLY của `export_splats` thật kiểm tra trên CPU; **chưa chạy trên GPU** | `tests/unit/test_experiment.py`, `test_benchmarks.py`, `test_depth_check.py`, `test_prune.py` |
| Biến thể có code (mục 3.1): bản sao có vá, KR, depth prior, kiểm NaN | Đã viết, đã kiểm thử trên CPU (28/10/2026): bản vá của `simple_trainer.py` thật đúng chỗ (cây cú pháp) và phân giải đúng tham số của 18 lượt `tune_mcmc` (PyTorch 2.7.1 chạy trên CPU, thư viện theo file khóa, fused-ssim là stub; kiểm tra tay, ngoài bộ test); KR với `MCMCStrategy` thật; loss Pearson bằng PyTorch; kiểm NaN dừng lượt; phần chạy thử của smoke test (trên CPU, không có bước relocate cần CUDA); depth prior với code Depth Anything V2 thật, trọng số ngẫu nhiên (kiểm tra tay, ngoài bộ test); `setup_depth_anything.sh` với kho git cục bộ và `file://`; **chưa train trên GPU, chưa chạy smoke test trên GPU, chưa tải trọng số thật** | `tests/unit/test_trainer_hooks.py`, `test_mono_depth.py`, `test_setup_depth_anything.py`, `test_gsplat_launcher.py`, `test_train_run.py`, `test_train_profiles.py`, `test_experiment.py`; `docs/design/experiments.md` mục 3.7 |
| Build extension, smoke test GPU, huấn luyện, số liệu | Repo trước đã chạy trên Runpod (17/10/2026) với Deep Blending, seed 42; **repo này chưa chạy trên pod** | `reports/modes_db_20261017_162815.md` của repo trước; `docs/ENVIRONMENT_AUDIT.md` mục 7. Mip-NeRF 360 và cảnh tự thu: chưa chạy |
