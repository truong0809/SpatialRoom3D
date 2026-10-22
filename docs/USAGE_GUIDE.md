# Hướng dẫn sử dụng: ảnh chụp hoặc video → mô hình 3DGS

_Cập nhật: 21/10/2026_

Giao diện hiện có:
- notebook `app/notebooks/runpod.ipynb`, chạy trong Jupyter của pod Runpod;
- giao diện dòng lệnh `python -m app.cli` (mục 6), cùng các script trong `scripts/`.

**Chưa có web app** (`app/web/README.md`; `docs/PROJECT_PLAN.md` mục 1.5: upload qua web là mở rộng có điều kiện).

## 1. Chuẩn bị một lần: volume, pod, khóa GitHub

### 1.1 Network volume

Runpod → **Storage → New Network Volume**: ≥ 100 GB (thực nghiệm chính ở mục 5.1 giữ mọi file kết quả: khoảng 56 GB cộng môi trường; chạy cả dò tham số ở mục 5.2 rồi thêm chế độ mới vào thực nghiệm chính thì nên 150 GB; volume tăng được, không giảm được), ở một data center Secure Cloud có loại GPU ở mục 1.2. Volume giữ mọi thứ trong `/workspace` (code, env conda, dữ liệu, kết quả) khi Stop hoặc Terminate pod. Pod sau này phải ở **cùng data center** mới gắn được volume này.

### 1.2 Pod

| Mục | Chọn | Lý do |
|---|---|---|
| Cloud | Secure Cloud, gắn network volume ở 1.1 (mount tại `/workspace`) | Network volume chỉ gắn được vào pod Secure Cloud |
| GPU | Compute capability 8.0 / 8.6 / 8.9 / 9.0: RTX 4090, RTX A6000, A40, L40S, A100, H100; VRAM ≥ 24 GB | Extension CUDA build sẵn cho đúng các kiến trúc này (`configs/environment/*.conf`). T4, RTX 50xx... không chạy được |
| Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | Bản **devel** có `nvcc` 12.4 và gcc 11 để build extension; đã chạy được ngày 17/10/2026 (`docs/ENVIRONMENT_AUDIT.md` mục 7) |
| Driver | Hỗ trợ CUDA ≥ 12.8 (dòng "CUDA Version" của `nvidia-smi`) | PyTorch 2.7.1 + cu128. Driver 12.6: dùng được với `TORCH_CUDA_OVERRIDE="cu126"` (ô 4); thấp hơn: đổi máy |
| Container disk | 20 GB là đủ | Mọi thứ cần giữ nằm trên volume |

Ô 1 của notebook kiểm tra các điều kiện này và báo dòng nào chưa đạt.

### 1.3 Khóa deploy cho repo private

Repo `truong0809/SpatialRoom3D` là private. Pod đọc repo bằng một **deploy key chỉ đọc**, tạo ở ô 2 của notebook:
1. Chạy ô 2: tạo khóa `/workspace/.ssh/github_deploy` (nếu chưa có) và in **khóa công khai**.
2. Trên GitHub: repo `SpatialRoom3D` → **Settings → Deploy keys → Add deploy key**, dán dòng khóa công khai, **không** tick "Allow write access".
3. Chạy lại ô 2: dòng cuối phải là `✓ GitHub nhận khóa`.

Khóa nằm trên volume: mỗi volume chỉ cần làm một lần. Không bao giờ gửi thư mục `/workspace/.ssh` cho ai; gói kết quả (mục 7) không chứa nó.

## 2. Notebook `runpod.ipynb`: chạy ô nào, khi nào

Tải file `app/notebooks/runpod.ipynb` lên `/workspace` (kéo thả vào thanh file của Jupyter), mở bản ở `/workspace`, chạy từng ô bằng Shift+Enter. Không mở thẳng bản trong repo: Jupyter ghi kết quả chạy vào file, làm repo trên pod có thay đổi và ô 3 sẽ không cập nhật code.

| Lúc nào | Chạy các ô |
|---|---|
| Lần đầu trên pod mới, volume trống: chỉ cài môi trường | 1 → 2 → 3 → 4 (theo dõi ở ô 5) |
| Thực nghiệm chính (mục 5.1): tự tải dữ liệu, so sánh `default` / `mcmc`, 3 seed (tốn GPU nhiều giờ; gửi được ngay sau ô 4) | 7 (`WHAT = "experiment"`, `EXP = "main"`, `CHAY = True`), theo dõi ở ô 5 |
| Dò tham số mcmc trên `room` (mục 5.2; tốn GPU khoảng 5 giờ) | 7 (`WHAT = "experiment"`, `EXP = "tune_mcmc"`, `CHAY = True`), theo dõi ở ô 5 |
| Ảnh so sánh các lượt trên cùng ảnh test và video xếp chồng (mục 4; không cần GPU) | 7 (`WHAT = "panels"`) hoặc 9 (`SO_SANH = True`) |
| Đưa dữ liệu của bạn lên (mục 2.1; trong lúc ô 4 chạy cũng được), rồi kiểm tra | 6 |
| Benchmark bạn chọn: tải dữ liệu, Deep Blending / Mip-NeRF 360 (tốn GPU) | 7 (`CHAY = True`), theo dõi ở ô 5 |
| Cảnh tự chụp: thư mục ảnh hoặc video (tốn GPU) | 8 (`CHAY = True`), theo dõi ở ô 5, kết quả ở ô 9 |
| Xem mọi kết quả, đóng gói để tải về | 10 |
| Dừng mọi việc đang chạy | 11 (`DUNG=1`) |

| Ô | Việc | GPU |
|---|---|---|
| 1 | Kiểm tra máy: GPU, driver, `nvcc`, gcc, VRAM, `/workspace` là network volume, PyYAML, git | Không |
| 2 | Khóa deploy (mục 1.3) | Không |
| 3 | Lấy code: chưa có thì clone nhánh `main` vào `/workspace/SpatialRoom3D`; đã có thì `git pull --ff-only`. Code trên pod có thay đổi chưa commit thì **không cập nhật** và in danh sách (không xóa, không cất gì). Không chạy khi đang có việc nền | Không |
| 4 | Cài môi trường ở nền: `scripts/pod/run_all.sh` (env `gs-inria`, `gs-tools`, `gs-gsplat`, kèm smoke test). **Không train.** Lần đầu khoảng 80 phút (đo ngày 17/10/2026 trên RTX 4090) | Có (smoke test) |
| 5 | Theo dõi việc nền, cập nhật mỗi 15 giây; tự dừng khi hết việc | Không |
| 6 | Kiểm tra dữ liệu (chỉ đọc): thư mục gốc của từng bộ dữ liệu, từng cảnh có `images/` và `sparse/0/` không | Không |
| 7 | `WHAT` = `experiment` (kế hoạch `EXP`: `main` là thực nghiệm chính, mục 5.1; `tune_mcmc` là dò tham số mcmc, mục 5.2; `CHAY = False` chỉ in kế hoạch), hoặc với benchmark `BENCH`: `check` (kiểm tra dữ liệu), `fetch` (tải dữ liệu về volume), `gsplat` (`default` / `mcmc`), `b0` (3DGS gốc của Inria), `evaluate` (lập lại bảng). Việc chạy nền chỉ chạy khi `CHAY = True`; `gsplat` và `b0` chỉ được xếp hàng khi dữ liệu qua bước kiểm tra (mục 5) | Tùy |
| 8 | Cảnh tự chụp (`CHAY = True`): ảnh / video → COLMAP → `default` → `mcmc` (mục 3) | Có |
| 9 | Kết quả một cảnh (kèm dòng "+ đã lọc" dưới lượt đã lọc Gaussian); `XUAT_PLY` (xuất PLY, không cần GPU), `RENDER_LAI` (render lại video quỹ đạo, chạy nền), `LOC` (lọc Gaussian của lượt `RUN` rồi chấm lại, chạy nền; mục 4) | Tùy |
| 10 | Mọi kết quả và gói kết quả nhẹ `ket_qua_*.tar.gz` | Không |
| 11 | Dừng mọi việc nền, kể cả việc đang xếp hàng (`DUNG=1`) | — |

- Notebook không tự chạy train hay tự tải dữ liệu: bạn chọn ở ô 7, ô 8. Dữ liệu benchmark tải được thẳng về pod (ô 7); ảnh / video của bạn thì bạn tự đưa lên (mục 2.1).
- Việc dài đều chạy nền: đóng trình duyệt không sao, nhưng **pod vẫn tính tiền** tới khi Stop / Terminate.
- Mỗi lúc chỉ một việc dùng GPU; việc gửi sau tự xếp hàng.
- Chạy lại ô nào cũng an toàn: phần đã xong tự bỏ qua.
- Xong việc: tải gói `ket_qua_*.tar.gz` (ô 10 in đường dẫn), rồi **Terminate pod**. Dữ liệu trên volume vẫn còn.

### 2.1 Đưa dữ liệu lên

Bạn tự đưa dữ liệu lên volume (kéo thả vào thanh file của Jupyter, hoặc cách khác), đúng chỗ dưới đây. Chỗ đặt lấy từ `configs/*.yaml`. Hai benchmark không cần đưa lên: pod tự tải về đúng chỗ (`bash scripts/pod/fetch_data.sh <benchmark> [cảnh ...]`, ô 7 `WHAT = "fetch"`, hoặc thực nghiệm chính tự tải; mục 5).

| Dữ liệu | Đặt ở |
|---|---|
| Ảnh tự chụp | `/workspace/data/custom_scenes/<cảnh>/`: các ảnh JPG/JPEG/PNG nằm thẳng trong thư mục (mục 3) |
| Video tự quay | `/workspace/data/custom_scenes/<cảnh>/<video>.mp4` (mục 3) |
| Deep Blending | `/workspace/data/benchmarks/deep_blending/drjohnson/`, `.../playroom/`: mỗi cảnh có `images/` và `sparse/0/` (`experiments/deep_blending/README.md`) |
| Mip-NeRF 360 | `/workspace/data/benchmarks/mipnerf360/<cảnh>/`: `images/`, `images_2/`, `images_4/`, `sparse/0/` (`experiments/mipnerf360/README.md`) |

- Tên cảnh: chữ không dấu, số và `. _ -`, không dấu cách. Không đặt dữ liệu trong `/workspace/SpatialRoom3D` (thư mục code).
- Dữ liệu nằm chỗ khác: tạo `configs/local.yaml` (chỉ nhận mục `datasets`), vd. `datasets: {mipnerf360: {root: "${WS}/data/MipNeRF360"}}`, hoặc đặt biến `SR3D_DATA_MIPNERF360`.
- Benchmark chưa có dữ liệu thì mọi lệnh train của nó dừng trước khi train, báo `dataset not found` kèm chỗ cần đặt và lệnh tải.

## 3. Cảnh tự chụp

### 3.1 Luồng sử dụng

1. Tải nguồn lên `/workspace/data/custom_scenes/<tên_cảnh>/` (mỗi cảnh một thư mục):
   - **nhiều ảnh** JPG/JPEG/PNG chụp cùng một căn phòng, nằm thẳng trong thư mục: ít nhất 10, nên 30–100. Trong Jupyter: tạo thư mục, mở nó, Upload và chọn nhiều ảnh cùng lúc;
   - hoặc **một file video**.
2. Chạy ô 8 của notebook với `CHAY = True` (`SCENE` để trống thì lấy tên thư mục), hoặc:
   ```bash
   cd /workspace/SpatialRoom3D
   bash scripts/pod/run_scene.sh /workspace/data/custom_scenes/phong1 phong1 --background             # thư mục ảnh
   bash scripts/pod/run_scene.sh /workspace/data/custom_scenes/phong2/phong2.mp4 phong2 --background  # video
   bash scripts/pod/run_scene.sh - phong1 --background                                                # đã có COLMAP: chỉ train
   ```
   Từng bước riêng: `bash scripts/prepare_data.sh custom <nguồn> <cảnh>` → `bash scripts/run_train.sh custom <cảnh>` → `bash scripts/run_render.sh custom <cảnh> [lượt]` → `bash scripts/run_evaluate.sh custom <cảnh>`.
3. Theo dõi: ô 5, hoặc `bash scripts/pod/job.sh status`.
4. Xem kết quả: ô 9, hoặc `bash scripts/pod/results.sh phong1`. Mô hình 3D: `outputs/custom_scenes/phong1/<chế_độ>_s42/ply/`; bản xuất để tải về: mục 4.
5. So sánh hai chế độ của cảnh: `bash scripts/run_evaluate.sh custom phong1`.

Cách chụp và quay: `docs/data/capture_protocol.md` (mục 8 cho ảnh chụp).

### 3.2 Mỗi job một thư mục

Một job là một tên cảnh. Dữ liệu gốc và kết quả nằm ở hai chỗ tách biệt:
- `data/custom_scenes/<cảnh>/`: ảnh / video gốc. Pipeline chỉ đọc, không ghi gì vào đây.
- `outputs/custom_scenes/<cảnh>/`: mọi thứ sinh ra:
  - `colmap/`:
    - `input/`: ảnh đã chuẩn hóa;
    - `images/`, `sparse/0/`: sau COLMAP;
    - `frames_report.json`: nguồn, ảnh nào được dùng, cảnh báo;
    - `sfm_report.json`: kết luận COLMAP, gợi ý, độ phủ góc nhìn;
  - `<chế_độ>_s<seed>/`: các lượt train (`default_s42`, `mcmc_s42`), mỗi lượt có ảnh test (`renders/`), chỉ số (`stats/`, `summary.json`), video quỹ đạo (`videos/`), checkpoint (`ckpts/`) và PLY (`ply/`).

Tên cảnh đã dùng cho một nguồn khác thì `run_scene.sh` dừng và yêu cầu đặt tên mới. Lượt train cũ không bị ghi đè: nó được đổi tên thành `.prev_<thời_gian>`. `--redo-colmap` chỉ làm lại `colmap/`. Luồng này không chạy benchmark nào.

### 3.3 Ảnh và video: khác nhau ở bước đầu, chung phần sau

| | Video | Thư mục ảnh |
|---|---|---|
| Bước đầu | ffmpeg trích frame, chọn frame nét nhất mỗi đoạn | **Không** trích hay chọn frame: mọi ảnh dùng được đều vào `input/`, theo thứ tự tên tự nhiên |
| Chuẩn hóa | Cạnh dài ≤ 1600 px, JPEG | Xoay theo EXIF, cạnh dài ≤ 1600 px, JPEG chất lượng 95. **Giữ tiêu cự quy đổi 35 mm trong EXIF** để COLMAP có tiêu cự ban đầu |
| Bị bỏ | Frame mờ, frame trùng | Chỉ bỏ file không đọc được và ảnh trùng y hệt. Lý do bỏ ảnh trùng: nếu một bản rơi vào tập test, điểm đánh giá bị thổi phồng |
| Chỉ cảnh báo | — | Ảnh có độ nét thấp hơn hẳn các ảnh khác; dưới 30 ảnh; file không phải JPG/JPEG/PNG (HEIC cần xuất sang JPG); ảnh không có tiêu cự trong EXIF; ảnh có nhiều tiêu cự khác nhau (lẫn ống kính hoặc zoom) |
| Dừng ngay, trước khi xếp hàng GPU | — | Một ảnh; ít hơn 10 ảnh (`-- --min-images N` để thử với ít hơn); thư mục không có ảnh; ảnh khác kích thước sau chuẩn hóa (vd. lẫn ảnh dọc và ngang, vì COLMAP dùng một camera chung) |
| COLMAP | Một camera chung (OPENCV), so khớp mọi cặp ảnh, mapper incremental, khử méo | Giống hệt |
| Huấn luyện | `default` (DefaultStrategy), rồi `mcmc` (MCMCStrategy, cùng số Gaussian) | Giống hệt |

**Vì sao giữ tiêu cự EXIF:** ảnh được ghi lại bằng OpenCV nên mất EXIF. Không có tiêu cự ban đầu, COLMAP đoán 1,2 × cạnh dài. Thử trên cùng bộ 36 ảnh tổng hợp (COLMAP 3.9.1, cùng đặc trưng và cặp khớp):
- không có tiêu cự ban đầu: model lớn nhất chỉ có 23/36 ảnh (FAIL);
- có tiêu cự đúng: 36/36 ảnh.

Thẻ tiêu cự mà bước này ghi lại đọc được bằng:
- COLMAP 3.9.1 (`tests/unit/test_frames.py`, chạy khi máy có `colmap`);
- pycolmap 4.2.1, cùng phiên bản COLMAP với bản trên pod: kiểm tra tay ngày 18/10/2026, tiêu cự 480,7 px cho ảnh 640×480, tiêu cự quy đổi 26 mm.

### 3.4 Kiểm tra trước khi train (`sfm_report.json`)

- **Kết luận `verdict`** quyết định có train hay không:
  - PASS khi ≥ 95 % ảnh có vị trí camera;
  - WARN khi ≥ 80 %, hoặc COLMAP tách ảnh thành nhiều nhóm, hoặc sai số chiếu lại > 1,5 px. Model phụ chỉ gồm ảnh đã có trong model lớn nhất (COLMAP đôi khi để lại một model 2–3 ảnh như vậy) không tính là bị tách;
  - FAIL khi dưới 80 %, khi COLMAP không dựng được, hoặc khi mô hình không có điểm 3D nào (3DGS khởi tạo từ điểm SfM). **FAIL thì không train.**
- **Gợi ý `hints`:**
  - ảnh không ghép được, kèm tên file gốc;
  - khi bị tách nhóm: số ảnh mỗi nhóm. Thường do ảnh không cùng một căn phòng hoặc thiếu ảnh nối giữa các khu vực;
  - mô hình không có điểm 3D: thường do đứng một chỗ chỉ xoay máy;
  - với lỗi COLMAP: nguyên nhân thường gặp và cách chụp / quay lại (lời khuyên khác nhau cho ảnh và cho video).
- **Độ phủ `coverage`** (chỉ cảnh báo, vì chụp một phần căn phòng vẫn hợp lệ):
  - phần hướng ngang mà các ảnh nhìn tới (hợp các góc nhìn ngang của camera);
  - mức camera di chuyển so với khoảng cách tới cảnh;
  - `LOW` khi có khoảng hướng ≥ 90 độ không ảnh nào nhìn tới (kèm hai ảnh ở hai mép khoảng trống), hoặc khi camera gần như đứng một chỗ (dịch chuyển < 5 %).
- Các ngưỡng là điểm xuất phát, sẽ hiệu chỉnh sau lần chụp thử.

**Giới hạn:** các kiểm tra này chỉ thấy được điều COLMAP thấy.
- Ảnh của phòng khác chỉ bị phát hiện khi COLMAP không nối được chúng với phần còn lại.
- Hai nơi có họa tiết giống hệt nhau (cùng một bức tranh, gạch lát lặp lại...) có thể bị COLMAP ghép nhầm thành một mô hình, và báo cáo vẫn PASS. Đã gặp với ảnh tổng hợp của hai phòng dán cùng các ảnh nguồn: COLMAP 3.9.1 ghép cả 40 ảnh của hai phòng vào một mô hình.
- Độ phủ tính trên vị trí camera mà COLMAP ước lượng, nên cũng sai theo khi COLMAP ghép nhầm.
- Vẫn cần xem lại mô hình trên viewer trước khi dùng kết quả.

### 3.5 Một ảnh

Bị từ chối, kèm hướng dẫn chụp thêm. Pipeline 3DGS nhiều góc nhìn ước lượng camera và hình học từ các ảnh chồng lấp, nên một ảnh không đủ để dựng lại căn phòng. Repo không có chế độ dựng từ một ảnh: muốn làm thì cần mô hình độ sâu hoặc mô hình sinh, và vẫn không khôi phục đúng phần không thấy trong ảnh.

### 3.6 Lưu ý khi đánh giá

- Cảnh tự quay hoặc tự chụp dùng cùng quy tắc tập test: cứ 8 ảnh lấy 1, theo thứ tự tên.
- Với video, frame test nằm xen giữa các frame train nên điểm số lạc quan (`docs/PROJECT_PLAN.md` mục 4.2).
- Với ảnh chụp, các ảnh cách nhau xa hơn, nhưng vẫn là ảnh của cùng lượt chụp.

### 3.7 Đổi tham số train

Ô 8 và `run_scene.sh` (mặc định) chạy hai lượt chuẩn: `default`, rồi `mcmc` với `cap_max` bằng số Gaussian cuối của `default` (không có số cố định). Muốn thử tham số khác, sau khi cảnh đã có COLMAP, chạy trong terminal của Jupyter. `TAG` cho mỗi lượt một thư mục riêng `<chế_độ>_<TAG>_s<seed>`, lượt chuẩn không bị ghi đè:

```bash
cd /workspace/SpatialRoom3D
TAG=or001 BASELINE_TAG= bash scripts/run_train.sh custom phong1 mcmc -- --set opacity_reg=0.001       # chỉ đổi mcmc
TAG=40k bash scripts/run_train.sh custom phong1 default mcmc -- --protocol max_steps=40000            # đổi protocol cho cả hai
TAG=cap2m bash scripts/run_train.sh custom phong1 mcmc -- --cap-max 2000000                          # tự đặt số Gaussian
```

- `--set`: tham số của chế độ (vd. `opacity_reg`, `strategy.refine_stop_iter`). `--protocol`: `max_steps`, `antialiased`... (bị ghi là sai khác protocol). Mọi thay đổi được ghi vào `run_meta.json`.
- Chạy nền: thêm `bash scripts/pod/job.sh start <tên_việc> --` trước `bash scripts/run_train.sh ...`.
- Bảng `bash scripts/run_evaluate.sh custom phong1` có cả biến thể `--set` (vd. `mcmc[opacity-reg=0.001]`); lượt đổi protocol chỉ hiện khi lập bảng với `--include-deviating` (`docs/design/training_modes.md` mục 6).
- Dùng lại nhiều lần: thêm một chế độ có tên vào `configs/base.yaml` (mẫu trong chú thích của file), sửa trong repo rồi commit, không sửa trên pod.
- Không chọn tham số theo điểm trên ảnh test (`docs/design/training_modes.md` mục 4.2).

## 4. Mô hình 3D: xuất PLY, render lại

Mỗi lượt train đã có sẵn ảnh test, video quỹ đạo (`videos/traj_<bước>.mp4`) và file PLY (`ply/point_cloud_<bước>.ply`, định dạng PLY của gsplat cho 3DGS) do bước đánh giá của gsplat ghi ra.

```bash
RENDER=0 bash scripts/run_render.sh custom phong1 default_s42        # chỉ xuất PLY (không cần GPU)
bash scripts/run_render.sh custom phong1 mcmc_s42                    # render lại ảnh test + video, rồi xuất PLY (cần GPU)
TRAJ=ellipse bash scripts/run_render.sh custom phong1 default_s42    # quỹ đạo khác: interp (mặc định), ellipse, spiral
bash scripts/run_render.sh deep_blending playroom default_s42        # benchmark cũng được
```

- **Xuất PLY:** file PLY của bước cuối được đưa vào `/workspace/outputs/checkpoints/<luồng>/<cảnh>/<lượt>.ply` (liên kết cứng nếu được, không tốn thêm chỗ: file xuất dùng chung dữ liệu với PLY của lượt, nên chỉ đọc / tải về, không sửa tại chỗ). Không ghi đè: tên đó đã là một PLY khác (vd. lượt cũ trước khi train lại) thì xuất thành `<lượt>_<thời điểm train xong>.ply`. Lượt chạy thử của benchmark (`QUICK=1`) xuất vào `checkpoints/<benchmark>_quick/`. Tải file về rồi mở bằng một trình xem 3D Gaussian Splatting đọc được định dạng PLY của 3DGS. **Chưa kiểm tra với trình xem cụ thể nào** (giai đoạn 5).
- **Render lại:** dùng chế độ `--ckpt` có sẵn của `simple_trainer.py`: đọc checkpoint, render ảnh test và video quỹ đạo vào thư mục mới `<lượt>/render_<quỹ_đạo>_<thời_gian>/` (kèm `render_meta.json`), cùng tập test với lượt gốc. Không train, không sửa file nào của lượt gốc. Bị từ chối khi lượt chưa train xong, hoặc khi mô hình COLMAP của cảnh đã đổi sau khi train (vd. `--redo-colmap`): checkpoint chỉ khớp với camera cũ, render sẽ sai.
- Trong notebook: ô 9 (`XUAT_PLY`, `RENDER_LAI`, `TRAJ`).
- Gói kết quả nhẹ (mục 7) không chứa PLY, checkpoint hay video: tải các file này riêng qua thanh file của Jupyter.

**Lọc Gaussian** (bỏ lớp sương mờ và Gaussian quá lớn, rồi chấm lại trên đúng ảnh test; `docs/design/experiments.md` mục 4). Cần GPU, chạy trong terminal của Jupyter:

```bash
bash scripts/run_filter.sh custom phong1 mcmc_s42              # một lượt: kết quả trong .../phong1/mcmc_s42/filtered/
bash scripts/run_filter.sh custom phong1                       # mọi lượt đã train xong của cảnh
MIN_OPACITY=0.05 MAX_SCALE=0.2 bash scripts/run_filter.sh mipnerf360 room mcmc_s42   # ngưỡng khác (mặc định 0,02 / 0,1)
bash scripts/run_evaluate.sh custom phong1                     # bảng có thêm phần sau khi lọc
```

- PLY đã lọc: `<lượt>/filtered/ply/point_cloud_<bước>.ply`, mở bằng cùng trình xem như PLY gốc. Ảnh test và video của bản đã lọc: `<lượt>/filtered/renders/`, `<lượt>/filtered/videos/`.
- Lượt gốc không bị sửa. Lọc lại với cùng ngưỡng thì bỏ qua; ngưỡng khác (hoặc `FORCE=1`) thì thư mục `filtered/` cũ đổi tên `filtered.prev_<thời gian>`.
- Trong notebook: ô 9 với `LOC = True` lọc lượt `RUN` (chạy nền). Khi việc xong, chạy lại ô 9: dòng `+ đã lọc` ngay dưới lượt đó có chỉ số sau khi lọc và tỉ lệ Gaussian giữ lại.
- Bỏ Gaussian lớn có thể làm thủng tường hay trần ở xa: xem cột Δ của bảng (chỉ số sau so với trước khi lọc) và video trước khi dùng bản đã lọc cho demo.

**Ảnh so sánh** các lượt trên cùng ảnh test (ảnh gốc, render, vùng cắt phóng to, bản đồ sai số) và video quỹ đạo của các lượt xếp chồng (`docs/design/comparisons.md`). Không cần GPU:

```bash
bash scripts/run_panels.sh custom phong1                                   # các lượt seed 42 của cảnh
VIDEO=1 bash scripts/run_panels.sh mipnerf360 counter default_s42 mcmc_s42   # chọn lượt, thêm video xếp chồng
```

- Kết quả: `/workspace/outputs/panels/<luồng>[_quick]_<cảnh>_<thời gian>/`; `panels.md` liệt kê panel, vùng cắt và PSNR (tính lại từ PNG) của mọi ảnh test. Thư mục này không nằm trong gói kết quả nhẹ: tải ảnh và video qua thanh file của Jupyter.
- So một lượt với bản đã lọc của nó: `bash scripts/run_panels.sh custom phong1 mcmc_s42 mcmc_s42/filtered`.
- Trong notebook: ô 9 với `SO_SANH = True` (cảnh tự chụp), ô 7 với `WHAT = "panels"`, `SCENES`, `PANEL_RUNS` (benchmark).

## 5. Benchmark: Deep Blending, Mip-NeRF 360

Chạy benchmark nào, lúc nào là do bạn quyết định; chỉ chạy khi đã duyệt chi phí GPU. Lệnh train cần dữ liệu có sẵn: thiếu thì dừng trước khi train. Protocol, số liệu đối chiếu và lệnh từng benchmark: `experiments/deep_blending/README.md`, `experiments/mipnerf360/README.md`; quy tắc so sánh: `docs/design/experiments.md`.

```bash
bash scripts/pod/fetch_data.sh deep_blending           # tải drjohnson, playroom (tandt_db.zip của Inria, kiểm SHA-256)
bash scripts/pod/fetch_data.sh mipnerf360 room counter kitchen bonsai   # 4 cảnh trong nhà (tải cả 360_v2.zip)
bash scripts/prepare_data.sh deep_blending             # kiểm tra dữ liệu (không dùng GPU)
QUICK=1 bash scripts/pod/job.sh start gsplat_db_quick -- bash scripts/run_train.sh deep_blending gsplat  # chạy thử 3.000 vòng
bash scripts/pod/job.sh start b0_db -- bash scripts/run_train.sh deep_blending b0          # B0: 3DGS gốc của Inria, chạy nền
bash scripts/pod/job.sh start gsplat_db -- bash scripts/run_train.sh deep_blending gsplat  # default / mcmc, chạy nền
SEEDS="42 43 44" bash scripts/pod/job.sh start gsplat_db3 -- bash scripts/run_train.sh deep_blending gsplat
bash scripts/run_evaluate.sh deep_blending             # lập lại bảng từ kết quả đã có, mọi chế độ (không dùng GPU)
```

- Mip-NeRF 360: thay `deep_blending` bằng `mipnerf360`; chạy một phần cảnh: `SCENES="room counter"`.
- Muốn bảng đối chiếu đủ thì chạy `b0` trước `gsplat`. Chưa có B0, bước so sánh cuối của `gsplat` (30.000 vòng) báo phần đối chiếu B0 là INCOMPLETE và việc kết thúc với exit 1; các lượt train vẫn được giữ. Chạy `b0` sau đó rồi `bash scripts/run_evaluate.sh <benchmark>` để lập lại bảng.
- Thời gian đo trên RTX 4090 ngày 17/10/2026 (repo trước, Deep Blending, seed 42): chạy thử 3.000 vòng 17 phút, B0 khoảng 50 phút, gsplat đầy đủ (`default` + `mcmc`) 74 phút.
- Tải dữ liệu: chỉ cảnh còn thiếu, chỉ thư mục cần dùng; file zip nằm tạm ở `/workspace/data/downloads/` và bị xóa sau khi giải nén (`KEEP_ARCHIVE=1` để giữ). Không bao giờ sửa hay ghi đè thư mục cảnh đã có; thư mục cảnh có nhưng thiếu phần thì dừng và báo.

Trong notebook: ô 7 (`WHAT`, `CHAY`, `BENCH`, `QUICK`, `SEEDS`, `SCENES`).

### 5.1 Thực nghiệm chính: `default` / `mcmc`

Một việc nền làm trọn thực nghiệm chính (thiết kế: `docs/design/experiments.md` mục 2): tải dữ liệu còn thiếu → B0 trên Deep Blending → `default`, `mcmc` với seed 42, 43, 44 trên Deep Blending (2 cảnh) và 4 cảnh trong nhà của Mip-NeRF 360 → bảng so sánh → gói kết quả. Luôn chạy đầy đủ 30.000 vòng; giữ mọi file kết quả. Thêm một chế độ vào `modes` của `configs/experiments/main.yaml` rồi chạy lại thì chỉ các lượt của chế độ đó được train.

```bash
bash scripts/pod/run_experiment.sh main --plan                       # kế hoạch, ước lượng, dữ liệu đã có
VOLUME_GB=100 bash scripts/pod/run_experiment.sh main --background   # chạy nền (việc exp_main)
```

Trong notebook: ô 7 với `WHAT = "experiment"`, `EXP = "main"`, `VOLUME_GB` = dung lượng thật của volume, rồi `CHAY = True`. Gửi được ngay sau ô 4: việc xếp hàng và tự chạy khi cài xong.

- Ước lượng (chưa đo trên Mip-NeRF 360): khoảng 12 giờ GPU trên RTX 4090; thêm khoảng 56 GB trên volume, cộng môi trường conda khoảng 15–20 GB, nên volume 100 GB đủ. Thêm chế độ vào kế hoạch thì cần thêm chỗ: tăng volume rồi đặt `VOLUME_GB` bằng dung lượng mới.
- Chỗ trống được kiểm tra trước khi chạy và trước mỗi seed, chỉ cho các lượt chưa xong: mỗi lượt tính theo dung lượng thật của lượt cùng cảnh, cùng chế độ đã xong ở seed khác (chưa có thì theo ước lượng). Không đủ thì dừng train, các lượt đã xong vẫn giữ; tăng volume rồi chạy lại để làm nốt. `SPACE_CHECK=0` bỏ qua.
- Pod vẫn tính tiền suốt lượt chạy. `AUTO_STOP = True` ở ô 7 (`AUTO_STOP=1` khi chạy lệnh) dừng pod khi việc cuối cùng xong; chưa thử trên pod.
- Kết quả: `/workspace/outputs/reports/experiment_main_<thời_gian>.md` (các bước, bảng so với `default` của từng benchmark); ô 10 cũng in các bước của lượt gần nhất.

### 5.2 Dò tham số `mcmc`: `tune_mcmc`

Một việc nền trên `room` của Mip-NeRF 360, seed 42 (thiết kế: `docs/design/experiments.md` mục 3): `default`, `mcmc` và 10 biến thể của `mcmc`, mỗi biến thể đổi một tham số (`opacity_reg`, `scale_reg`, `strategy.noise_lr`, `strategy.min_opacity`, `random_bkgd`, `depth_loss`); kiểm tra điểm SfM trước khi train (vì có biến thể `depth_loss`); sau khi train, lọc Gaussian mọi lượt rồi chấm lại; bảng so với `default`, bảng so với `mcmc`, bảng sau khi lọc.

```bash
bash scripts/pod/run_experiment.sh tune_mcmc --plan                       # kế hoạch, các bước, dữ liệu đã có
VOLUME_GB=100 bash scripts/pod/run_experiment.sh tune_mcmc --background   # chạy nền (việc exp_tune_mcmc)
```

Trong notebook: ô 7 với `WHAT = "experiment"`, `EXP = "tune_mcmc"`, `VOLUME_GB`, rồi `CHAY = True`.

- Ước lượng (chưa đo): khoảng 5 giờ GPU trên RTX 4090; thêm khoảng 24 GB trên volume. Lúc tải, `360_v2.zip` (khoảng 12,5 GB) nằm tạm trên volume.
- Một cảnh, một seed: chỉ để chọn ứng viên cho chế độ `mcmc_indoor`, chưa phải kết quả "cải thiện". `room` là cảnh dùng để chọn tham số: khi so `mcmc_indoor` trong thực nghiệm chính, kết luận dựa trên 5 cảnh còn lại. Bảng tự động vẫn có `room`; bảng không có `room`: `docs/design/experiments.md` mục 3.1.
- Lượt `room/default_s42`, `room/mcmc_s42` dùng chung với thực nghiệm chính: chạy `main` sau đó không train lại hai lượt này. 10 biến thể có tag nên bảng của `main` không lẫn chúng (có một dòng cảnh báo ghi số lượt để ngoài); xem lại bảng của các biến thể: `TAGS="or0001 or0005 or005 sr0001 sr005 nlr250k nlr1m mo002 rbg depth" SCENES=room bash scripts/run_evaluate.sh mipnerf360` (ô 7: `WHAT = "evaluate"`, `BENCH = "mipnerf360"`, `SCENES = "room"`, `TAGS`).
- Kết quả: `/workspace/outputs/reports/experiment_tune_mcmc_<thời_gian>.md`.

## 6. Giao diện dòng lệnh `python -m app.cli`

Một lệnh cho mọi bước; mỗi lệnh gọi đúng một script trong `scripts/` với tham số đi nguyên, không có logic riêng. Chạy từ `/workspace/SpatialRoom3D`:

| Lệnh | Gọi | Việc |
|---|---|---|
| `check` | `scripts/pod/check_datasets.sh` | Kiểm tra dữ liệu và kết quả của mọi luồng (chỉ đọc) |
| `prepare` | `scripts/prepare_data.sh` | `custom <ảnh\|video> <cảnh>` hoặc `<benchmark>` |
| `train` | `scripts/run_train.sh` | `custom <cảnh> [chế độ ...]` hoặc `<benchmark> [gsplat\|b0]` |
| `render` | `scripts/run_render.sh` | `custom <cảnh> [lượt]` hoặc `<benchmark> <cảnh> [lượt]` |
| `filter` | `scripts/run_filter.sh` | Lọc Gaussian rồi chấm lại: `custom <cảnh> [lượt ...]` hoặc `<benchmark> <cảnh> [lượt ...]` (mục 4) |
| `evaluate` | `scripts/run_evaluate.sh` | `<benchmark>` hoặc `custom [cảnh ...]` |
| `panels` | `scripts/run_panels.sh` | Ảnh so sánh các lượt, video xếp chồng: `<benchmark> <cảnh> [lượt ...]` hoặc `custom <cảnh> [lượt ...]` (mục 4) |
| `fetch` | `scripts/pod/fetch_data.sh` | Tải dữ liệu benchmark: `<benchmark> [cảnh ...] [--dry-run]` |
| `experiment` | `scripts/pod/run_experiment.sh` | Thực nghiệm trọn vẹn: `<tên> --plan` hoặc `<tên> --background` (`main`: mục 5.1, `tune_mcmc`: mục 5.2) |
| `scene` | `scripts/pod/run_scene.sh` | Cả luồng cho một cảnh tự chụp: `<ảnh\|video\|-> <cảnh> [--background]` |
| `setup` | `scripts/pod/run_all.sh --background` | Cài / kiểm tra môi trường (chạy nền, không train) |
| `status`, `log`, `stop` | `scripts/pod/job.sh` | Việc chạy nền: trạng thái, cuối log, dừng |
| `results` | `scripts/pod/results.sh` | Xem nhanh mọi kết quả |
| `pack` | `scripts/pod/pack_results.sh` | Gói kết quả nhẹ để tải về |

```bash
python3 -m app.cli --help
python3 -m app.cli --dry-run train deep_blending gsplat    # chỉ in lệnh bash sẽ chạy
python3 -m app.cli scene /workspace/data/custom_scenes/phong1 phong1 --background
```

Biến môi trường (`WS`, `SEEDS`, `QUICK`, `TRAJ`, ...) được truyền nguyên cho script.

## 7. Theo dõi, dừng, tải kết quả

```bash
bash scripts/pod/job.sh status                 # các việc nền: ĐANG CHẠY, ĐANG XẾP HÀNG, XONG, LỖI, ĐÃ DỪNG, MẤT
bash scripts/pod/job.sh log <tên> 50           # cuối log của một việc (log đầy đủ: /workspace/outputs/logs/jobs/<tên>.log)
bash scripts/pod/job.sh stop <tên>             # hoặc stop --all
bash scripts/pod/pack_results.sh               # /workspace/ket_qua_<tháng><ngày>_<giờ><phút><giây>.tar.gz
```

Gói kết quả gồm báo cáo, log, trạng thái việc nền, JSON / CSV kết quả và một ảnh render mỗi lượt; không có checkpoint, PLY, video, ảnh gốc hay dataset.

## 8. Khi gặp lỗi

| Thấy trong log / ô | Cách xử lý |
|---|---|
| `✗ GitHub chưa nhận khóa`, `Permission denied (publickey)` (ô 2, 3) | Khóa công khai chưa được thêm vào Deploy keys của **đúng** repo `SpatialRoom3D`: thêm rồi chạy lại ô 2 |
| `Code trên pod có thay đổi chưa commit` (ô 3) | Có file trong `/workspace/SpatialRoom3D` bị sửa (vd. mở notebook trong repo): xem `git -C /workspace/SpatialRoom3D status`, tự quyết định giữ hay bỏ |
| `no Python with PyYAML` | `python3 -m pip install pyyaml`, hoặc chờ ô 4 cài xong env `gs-gsplat` |
| `driver supports CUDA 12.6 < 12.8` | Ô 4: đặt `TORCH_CUDA_OVERRIDE="cu126"`, chạy lại ô 4 |
| `Killed` khi build gsplat | Ô 4: đặt `MAX_JOBS="4"`, chạy lại ô 4 |
| `compute capability ... is not in TORCH_CUDA_ARCH_LIST` | GPU không được hỗ trợ (vd. T4, RTX 50xx): deploy GPU khác |
| `another job is using the GPU` | Đang có việc khác chạy: xem ô 5; muốn dừng thì ô 11 |
| `MẤT` ở ô 5 | Pod tắt hoặc việc bị giết giữa chừng: chạy lại ô vừa dùng |
| `dataset not found` (ô 7) | Dữ liệu benchmark chưa có ở chỗ `configs/<benchmark>.yaml` ghi: tải bằng ô 7 (`WHAT = "fetch"`) hoặc lệnh in trong thông báo, tự đưa lên đúng chỗ (mục 2.1), hoặc ghi đường dẫn vào `configs/local.yaml` |
| `seed ... would not fit in a ... GB volume`, `may not fit`, `không đủ chỗ trống cho seed` (thực nghiệm chính) | Volume không đủ chỗ: tăng dung lượng network volume trên trang Runpod, đặt `VOLUME_GB` bằng dung lượng mới (ô 7) rồi chạy lại; phần đã xong tự bỏ qua |
| `SHA-256 of ... expected`, `is damaged`, `curl exited with code` (tải dữ liệu) | File tải về hỏng, khác bản gốc hoặc mạng lỗi: file dở đã bị xóa, chạy lại để tải lại |
| `already exist(s) ... but are incomplete` (tải dữ liệu) | Thư mục cảnh đã có nhưng thiếu phần (vd. tự đưa lên dở dang): xem lại hoặc dời thư mục đó đi rồi chạy lại; bước tải không sửa dữ liệu đã có |
| `prepared must be <results.gsplat>/<cảnh>/colmap` | `configs/local.yaml` chỉ đổi một trong hai khóa của `custom_scene`: đổi cả hai |
| `Chỉ có 1 ảnh`, `Chỉ có N ảnh dùng được` | Chụp thêm (ít nhất 10, nên 30–100 ảnh), tải vào cùng thư mục, chạy lại ô 8 |
| `ảnh khác kích thước` | Lẫn ảnh dọc và ngang, hoặc khác ống kính: bỏ các ảnh ở nhóm ít hơn (tên in trong thông báo) hoặc chụp lại cùng chiều máy |
| `không ảnh nào có tiêu cự trong EXIF` | Ảnh đã mất EXIF (gửi qua ứng dụng nhắn tin, chụp màn hình, chỉnh sửa): dùng file gốc của máy nếu có |
| `đã được dựng từ ..., không phải ...` | Tên cảnh đã dùng cho nguồn khác: đặt `SCENE` mới |
| `COLMAP verdict FAIL` | Xem các dòng `gợi ý` ở ô 9, chụp / quay lại theo `docs/data/capture_protocol.md` và chạy ô 8 với tên cảnh mới |
| `góc nhìn: THIẾU` (ô 9), `VIEW COVERAGE: LOW` (log) | Chỉ là cảnh báo: có hướng chưa ảnh nào nhìn tới, hoặc camera gần như đứng yên |
| `not a finished run` (ô 9) | Lượt `RUN` chưa train xong hoặc tên sai: xem tên các lượt trong bảng của ô 9 |
| `cannot be trained with the SfM depth loss` (bước `depth_check_<benchmark>` của thực nghiệm) | Có ảnh train không có điểm nào trong track của SfM (tên ảnh in ở dòng `FAIL` phía trên): lượt có `depth_loss` sẽ lỗi. Chỉ các lượt đó của benchmark đó bị bỏ, các lượt khác vẫn train. Dòng `WARN` (mọi điểm của ảnh nằm ngoài ảnh) chỉ là cảnh báo |
| `the thresholds remove every Gaussian`, `evaluation of the filtered model failed` (lọc Gaussian) | Ngưỡng quá chặt: giảm `MIN_OPACITY` hoặc tăng `MAX_SCALE`. Lỗi chấm lại: xem `<lượt>/filtered/eval.log`; lượt gốc không bị ảnh hưởng |
| `COLMAP model ... changed after this run was trained` (render lại) | Cảnh đã làm lại COLMAP sau lượt này: chạy ô 8 với `SOURCE = "-"` và `SCENE` của cảnh (lượt làm từ mô hình cũ tự được train lại), rồi render lượt mới; xuất PLY của lượt cũ vẫn được (`XUAT_PLY`) |
| Lỗi khác | Chạy ô 10 và gửi gói `ket_qua_*.tar.gz` (không gửi `/workspace/.ssh`) |

Sự cố môi trường đã biết: `docs/ENVIRONMENT_AUDIT.md` mục 6.

## 9. Đã kiểm chứng tới đâu

| Phần | Trạng thái | Bằng chứng |
|---|---|---|
| Luồng ảnh và video, kiểm tra trước khi train | Đã kiểm thử trên CPU | `tests/unit/test_sources.py`, `tests/unit/test_frames.py`, `tests/unit/test_sfm.py`, `tests/test_pipeline.py`, `tests/unit/test_overview.py` |
| Cùng luồng với COLMAP thật | Bước chuẩn bị (ảnh / video → COLMAP → kiểm tra) đã chạy với COLMAP 3.9.1 bản CPU trên ảnh và video tổng hợp (không phải dữ liệu thực nghiệm); bước train thay bằng stub | `docs/PROJECT_STATUS.md`, mục "Kiểm thử trong sandbox" (log nằm trong sandbox, không vào repo) |
| Render lại, xuất PLY, CLI | Đã kiểm thử trên CPU với runner giả / stub (không có gsplat thật) | `tests/unit/test_render.py`, `tests/unit/test_cli.py` |
| Notebook `runpod.ipynb` | Đã mô phỏng trên `/workspace` giả: GitHub thay bằng repo Git cục bộ, `ssh` / `curl` giả, không GPU, không conda (các việc cần GPU dừng sớm) | `docs/PROJECT_STATUS.md`, mục "Kiểm thử trong sandbox" |
| Tải dữ liệu benchmark, thực nghiệm chính | Đã kiểm thử với curl giả, file zip tổng hợp và các bước giả (stub); **chưa tải thật, chưa chạy trên pod** | `tests/unit/test_fetch.py`, `tests/unit/test_experiment.py` |
| Dò tham số `tune_mcmc` (biến thể, kiểm tra điểm SfM), lọc Gaussian | Đã kiểm thử với các bước giả, runner giả; checkpoint PyTorch và PLY của gsplat kiểm tra trên CPU; **chưa chạy trên GPU** | `tests/unit/test_experiment.py`, `test_benchmarks.py`, `test_depth_check.py`, `test_prune.py` |
| Bảng so sánh mở rộng (số cặp và số cảnh tốt hơn, chất lượng theo dung lượng, cấu trúc mô hình), ảnh so sánh và video xếp chồng | Đã kiểm thử trên CPU với PLY, ảnh render và video tổng hợp; **chưa chạy trên kết quả thật** | `tests/unit/test_compare_modes.py`, `tests/unit/test_splat_stats.py`, `tests/unit/test_panels.py` |
| Trên pod (COLMAP 4.2.1, GPU) bằng repo này | **Chưa chạy** | — |

Tóm tắt các ca với COLMAP thật (sandbox):
- cùng một phòng, có tiêu cự EXIF: 36/36 ảnh, PASS; cùng bộ ảnh nhưng không có tiêu cự: 23/36, FAIL;
- video đi vòng quanh phòng: 68/68 frame, PASS;
- ảnh hai phòng lẫn nhau (24 + 16 ảnh, có EXIF): mô hình chỉ có 24 ảnh của phòng thứ nhất, 16 ảnh còn lại được liệt kê kèm tên file gốc, FAIL nên không train;
- đứng một chỗ chỉ xoay máy: FAIL, kèm gợi ý chụp lại có di chuyển;
- 1 ảnh, 1 file ảnh, 5 ảnh: bị từ chối trước khi xếp hàng GPU.
