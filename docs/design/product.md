# Sản phẩm: dựng phòng cho web bằng mô hình chính

Luồng sản phẩm nhận **một video hoặc một thư mục ảnh** của căn phòng, dựng mô hình 3D bằng **mô hình chính** (một cấu hình duy nhất, chọn qua thực nghiệm), rồi xuất các file mà trang web đọc để xem phòng. Tài liệu này là hợp đồng giữa phần xử lý trên pod và phần web: file nào, nằm ở đâu, định dạng ra sao, hệ tọa độ nào, tiến độ đọc ở đâu.

Trạng thái: **đã chạy trên pod RTX 4090 ngày 08/11/2026**: 7 phòng dựng từ video tải lên trang web; một video 2 phút 54 giây mất 18 phút 45 giây (`docs/RESULTS.md` mục 7). Đã kiểm thử bằng stub và đối chiếu với code thật của gsplat trên CPU (mục 7). Trang web đọc các file này (tải ảnh / video lên, tiến độ, xem 3D: mục 8) cũng đã chạy trên pod ngày đó.

## 1. Mô hình chính

| | |
|---|---|
| Phương pháp | 3DGS-MCMC (Kheradmand et al., 2024): `MCMCStrategy` của [gsplat](https://github.com/nerfstudio-project/gsplat) commit `6e8c837`, preset `mcmc` của `examples/simple_trainer.py` |
| Tham số | giữ nguyên preset (`opacity_reg` 0,01, `scale_reg` 0,01, `init_opa` 0,5, `init_scale` 0,1), 30.000 vòng, ảnh gốc (`data_factor` 1), khởi tạo từ điểm SfM của COLMAP, seed 42 (simple_trainer gán cứng) |
| Cấu hình | [`configs/product.yaml`](../../configs/product.yaml), chỉ luồng sản phẩm đọc file này |
| Code train | [`src/product/train_main.py`](../../src/product/train_main.py): gọi thẳng `simple_trainer.py`, không dùng code thực nghiệm (`src/training/`) |

Lý do chọn (thực nghiệm chính chạy ngày 31/10/2026, cùng số Gaussian, 6 cảnh trong nhà, 3 seed): `mcmc` tốt hơn `default` ở 17/18 cặp (cảnh, seed) về PSNR, 18/18 về SSIM và LPIPS. ΔPSNR trung bình: Mip-NeRF 360 +0,590 ± 0,181 dB (4 cảnh), Deep Blending +0,338 ± 0,239 dB (2 cảnh). Thời gian train bằng ×0,98–1,10 so với `default`, VRAM đỉnh thấp hơn. Trên Deep Blending, `mcmc` dùng `opacity_reg` 0,001 theo quy ước của benchmark. Sản phẩm giữ 0,01 như Mip-NeRF 360 và phòng nhóm. Phòng nhóm (1 seed) cho kết quả trái chiều: PSNR thấp hơn `default` (39,07 so với 39,32 dB) nhưng SSIM, LPIPS tốt hơn. Ảnh test là các khung kề khung train của cùng video, nên số này không dùng để so với benchmark.

Lượt dò tham số (`tune_mcmc`, `room`, 1 seed) chưa cho biến thể nào tốt hơn rõ ràng. Chỉ `random_bkgd` hơn 0,156 dB, sát ngưỡng dao động. Vì vậy sản phẩm giữ nguyên preset; vòng ghép để kiểm lại ứng viên này đã bỏ (D25). Muốn đổi cấu hình sau này, chỉ cần sửa `configs/product.yaml`: các phòng đã dựng sẽ được train lại ở lần chạy sau (mục 3).

## 2. Khác lượt `mcmc` của thực nghiệm, có chủ ý

| | Thực nghiệm | Sản phẩm | Lý do |
|---|---|---|---|
| Số Gaussian tối đa (`cap_max`) | bằng số Gaussian cuối của lượt `default` cùng cảnh (0,97–3,55 triệu) | cố định 1.000.000 (mặc định của `MCMCStrategy`) | không phải train `default` trước; biết trước dung lượng file web (≤ 32 MB) |
| Ảnh dùng để train | 7/8 số ảnh (`test_every` 8) | mọi ảnh trừ ảnh số 0 (gsplat luôn giữ ảnh số 0 làm ảnh kiểm tra) | nhiều góc nhìn hơn cho mô hình xem trên web |
| Chấm điểm, render video | có | không | sản phẩm không có ảnh test riêng (mục 5) |

Ngoài hai điểm trên, lệnh train giống hệt: phân giải tham số bằng chính `simple_trainer.py` @6e8c837 cho 70 khóa cấu hình trùng nhau, trừ `test_every` (mục 7). Phòng lớn có thể cần hơn 1 triệu Gaussian: với giới hạn này, chất lượng có thể thấp hơn số đo thực nghiệm một chút. Điều này **chưa đo**.

## 3. Cách chạy

Đầu vào:
- **video**: một file MP4, MOV...; tách khung, giữ các khung nét;
- **ảnh**: một thư mục ảnh JPG/JPEG/PNG chụp cùng một phòng, các ảnh chồng lấp nhau, tối thiểu 10 ảnh, nên từ 30 ảnh; dùng thẳng, không tách khung.

Cách quay / chụp: [docs/data/capture_protocol.md](../data/capture_protocol.md). Lệnh (trên pod, từ thư mục repo, sau `scripts/pod/run_all.sh`):

```bash
bash scripts/build_room.sh /workspace/data/custom_scenes/phong1 phong1 --name "Phòng khách" --background   # thư mục ảnh
bash scripts/build_room.sh /workspace/data/custom_scenes/video8/video8.mp4 phong2 --background              # video
bash scripts/build_room.sh - phong1                    # phòng đã có COLMAP: train khi cần, xuất lại file web
python3 -m app.cli build-room ...                      # cùng lệnh qua giao diện dòng lệnh
DRY_RUN=1 bash scripts/build_room.sh <nguồn> <phòng>   # chỉ kiểm tra đầu vào và in các lệnh sẽ chạy
```

| Bước | Tự động | Code |
|---|---|---|
| Kiểm tra nguồn (video / thư mục ảnh, số ảnh), trước khi xếp hàng | có | `src/input/sources.py` |
| Ảnh đầu vào (tách khung hoặc chuẩn hóa ảnh), COLMAP, báo cáo chất lượng; COLMAP FAIL thì dừng | có | `scripts/pipeline/prepare_scene.sh`, dùng chung với luồng thực nghiệm |
| Train mô hình chính | có | `src/product/train_main.py` |
| Xuất file web, thêm vào thư viện | có | `src/product/export_web.py` |
| Đưa video / ảnh lên volume và gọi lệnh dựng phòng | có, qua trang web (mục 8); hoặc chép tay rồi chạy lệnh trên | `app/web/` |
| Cài môi trường trên pod, chạy server trang web, mở cổng HTTP của pod trên trang Runpod | **thủ công** | `scripts/pod/run_all.sh`, `scripts/web_server.sh`; mục 8 |

- `--background`: chạy nền qua `scripts/pod/job.sh` (tên việc `room_<phòng>`); đang có việc khác dùng GPU thì xếp hàng.
- Chạy lại an toàn. Phòng đã có COLMAP thì dùng lại (`--redo-colmap`: làm lại). Mô hình đã train từ đúng COLMAP đó với đúng cấu hình thì không train lại (`--retrain`: train lại), và không cần GPU: chỉ xuất lại file web. Kết quả cũ không bị xóa mà được đổi tên `model.prev_<thời gian>`; xóa tay khi cần chỗ trên volume (mỗi bản khoảng 0,5 GB với 1 triệu Gaussian).
- Dữ liệu đã có trên volume không bị đụng tới. Luồng sản phẩm chỉ ghi vào `$WS/outputs/product/`, log `build_room_*` và việc nền `room_*`, kể cả khi tên phòng trùng tên một cảnh thực nghiệm. Có test kiểm tra từng byte (mục 7). Clone lại repo cũng không mất gì: mọi dữ liệu và kết quả nằm ngoài thư mục repo.
- Mỗi phòng một tên (chữ, số, `.`, `_`, `-`). Tên đã dùng cho một nguồn khác thì dừng và báo.
- Thời gian: đo ngày 08/11/2026 trên RTX 4090, một video 2 phút 54 giây mất 18 phút 45 giây cho cả luồng (tách khung 12 giây, COLMAP 9 phút 40 giây, train 8 phút 32 giây, xuất 18 giây; `docs/RESULTS.md` mục 7).

## 4. Hợp đồng với web

### 4.1 Thư mục

```
$WS/outputs/product/                 output_root của configs/product.yaml ($WS = /workspace trên Runpod)
  <phòng>/colmap/                    ảnh, COLMAP, frames_report.json, sfm_report.json
  <phòng>/model/                     ply/point_cloud_29999.ply, ckpts/, stats/, train_meta.json
  <phòng>/progress.json              tiến độ dựng phòng (4.5)
  web/                               THƯ MỤC TĨNH CHO WEB (web.dir)
    manifest.json                    thư viện phòng (4.2)
    rooms/<phòng>/room.json          mục của phòng trong manifest.json
    rooms/<phòng>/model.splat        mô hình (4.3)
    rooms/<phòng>/cameras.json       camera và khung của phòng (4.4)
    rooms/<phòng>/thumbnail.jpg      ảnh đại diện, cạnh dài 640 px
```

Trang web chỉ cần đọc `web/`, và `<phòng>/progress.json` khi đang dựng. Mọi đường dẫn trong `manifest.json` tính từ thư mục `web/`. Các file được ghi trọn vẹn (file tạm rồi đổi tên), nên bên đọc không thấy file viết dở. `manifest.json` được dựng lại từ các `rooms/*/room.json` mỗi lần xuất. Xóa một phòng khỏi thư viện: xóa thư mục `rooms/<phòng>/`, rồi chạy `python -m src.product.export_web --rebuild-manifest`.

### 4.2 `manifest.json`

```json
{
  "format_version": 1,
  "generated_at": "2026-11-04T02:00:00+00:00",
  "rooms": [
    {
      "id": "phong1",
      "name": "Phòng khách",
      "version": "8d9d549af42d",
      "built_at": "2026-11-04T02:00:00+00:00",
      "format": "splat",
      "files": {"model": "rooms/phong1/model.splat", "cameras": "rooms/phong1/cameras.json",
                "thumbnail": "rooms/phong1/thumbnail.jpg"},
      "num_gaussians": 998123,
      "size_bytes": 31939936,
      "viewer_modes": ["orbit", "walk", "tour"],
      "source": {"kind": "video", "input_images": 243, "registered_images": 240, "sfm_verdict": "PASS",
                 "video_duration_s": 95.2},
      "model": {"name": "mcmc", "cap_max": 1000000, "max_steps": 30000, "trainer_commit": "6e8c837f…",
                "num_gaussians_trained": 1000000, "train_seconds": 650.2, "trained_at": "2026-11-04T01:50:00+00:00"},
      "metrics": null,
      "warnings": []
    }
  ]
}
```

Số trong ví dụ là minh họa. Các trường:

| Trường | Ý nghĩa |
|---|---|
| `rooms` | phòng train gần nhất đứng trước; xuất lại một phòng không đổi thứ tự |
| `id` | tên phòng của lệnh `build_room.sh`, cũng là tên thư mục |
| `name` | tên hiển thị (`--name`; mặc định giữ tên cũ, hoặc `id`) |
| `version` | 12 ký tự đầu SHA-256 của `model.splat`; thêm `?v=<version>` vào URL để trình duyệt không dùng bản cũ trong cache |
| `files` | `cameras` / `thumbnail` là `null` khi không có |
| `num_gaussians`, `size_bytes` | số Gaussian và dung lượng của `model.splat` (32 byte mỗi Gaussian) |
| `viewer_modes` | `["orbit"]` khi không có `cameras.json`: khi đó tắt chế độ đi bộ và tham quan |
| `source.kind` | `"video"` hoặc `"photos"` |
| `source.sfm_verdict` | `PASS` / `WARN` của COLMAP; `FAIL` thì không có phòng |
| `model` | cấu hình đã thực sự train (từ `train_meta.json`) |
| `metrics` | luôn `null`: phòng tự dựng chưa được chấm điểm (mục 5) |
| `warnings` | cảnh báo của lần xuất, viết bằng tiếng Việt (vd. không đọc được camera, hệ tọa độ bị lật); nên hiện cho người xem |

### 4.3 `model.splat`

Định dạng `.splat` của [antimatter15/splat](https://github.com/antimatter15/splat), cùng bố cục với `export_splats(format="splat")` của gsplat. Thư viện xem 3DGS trên web đọc được trực tiếp, vd. [Spark](https://sparkjs.dev/docs/loading-splats/) và [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) (theo tài liệu của hai thư viện; **chưa thử** với file của luồng này). Mỗi Gaussian 32 byte, little-endian, không có header:

| Byte | Kiểu | Nội dung |
|---|---|---|
| 0–11 | 3 × float32 | vị trí x, y, z |
| 12–23 | 3 × float32 | độ lớn theo 3 trục (đã lấy `exp`, không phải log) |
| 24–27 | 4 × uint8 | R, G, B = (0,5 + 0,2821 × hệ số SH bậc 0) × 255; A = sigmoid(opacity) × 255 |
| 28–31 | 4 × uint8 | quaternion w, x, y, z đã chuẩn hóa, lưu dạng q × 128 + 128 |

Gaussian xếp theo thể tích × độ đục giảm dần, nên trình xem tải dần sẽ hiện các Gaussian lớn trước. Gaussian trong suốt hoàn toàn (A = 0) và Gaussian có giá trị không hữu hạn bị bỏ, không làm đổi ảnh render. `.splat` chỉ giữ màu cơ bản: hiệu ứng đổi theo góc nhìn (SH bậc 1–3, vd. phản chiếu) mất khi xem trên web. Mô hình đầy đủ nằm ở `<phòng>/model/ply/point_cloud_29999.ply` (khoảng 236 MB với 1 triệu Gaussian), mở được bằng SuperSplat và các trình xem đọc PLY 3DGS.

### 4.4 `cameras.json`

```json
{
  "format_version": 1, "coordinate_frame": "model", "camera_convention": "opencv", "units": "relative",
  "right_handed": true,
  "up": [0.045, 0.033, -0.998], "center": [-0.014, -0.001, 0.032], "scene_radius": 0.975,
  "bounds": {"min": [-0.5, -0.39, -0.16], "max": [0.48, 0.4, 0.17], "percentiles": [1.0, 99.0]},
  "intrinsics": {"width": 1600, "height": 900, "fx": 1310.2, "fy": 1310.2, "cx": 800.0, "cy": 450.0,
                 "fov_y_deg": 37.89},
  "default_view": 120,
  "frames": [{"name": "00001.jpg", "position": [-0.982, 0.102, -0.008], "forward": [0.991, -0.107, 0.074],
              "up": [0.078, 0.029, -0.997]}]
}
```

- **Hệ tọa độ**: cùng hệ với `model.splat`. Đây là hệ đã chuẩn hóa của gsplat, do lớp `Parser` của `examples/datasets/colmap.py` tạo ra khi train, nên camera khớp mô hình. **Hướng lên không cố định** (không phải luôn +y hay +z): luôn dùng trường `up`.
- **Đơn vị tương đối**, không phải mét. Không có LiDAR nên tỉ lệ thật của phòng không được biết. Tốc độ di chuyển, khoảng cách nên tính theo `scene_radius` (khoảng cách xa nhất từ tâm các vị trí camera, như `scene_scale` của gsplat).
- **Mỗi khung** là một ảnh đầu vào đã định vị: `position`, hướng nhìn `forward` và hướng lên của chính camera `up` (vector đơn vị). Khung xếp theo tên ảnh. Với video, đó là thứ tự quay (đường tham quan).
- Dựng camera của three.js từ một khung: `camera.position.set(...position)`, `camera.up.set(...up)`, `camera.lookAt(position + forward)`; `camera.fov = intrinsics.fov_y_deg`, tỉ lệ khung = `width / height`. (Quy ước OpenCV: x phải, y xuống, z nhìn tới; `forward` = z, `up` = −y của camera.)
- Gợi ý cho ba chế độ xem của bản thiết kế:
  - xoay quanh: tâm `center`, bán kính khởi đầu cỡ `scene_radius`, mở ở khung `default_view`;
  - đi bộ: giữ mắt ở độ cao trung bình của các camera theo `up`, giới hạn trong `bounds`;
  - tham quan: đi lần lượt qua `frames`, nội suy vị trí và hướng giữa các khung kề nhau.
- `bounds`: phân vị 1 % và 99 % tâm các Gaussian theo từng trục, bỏ các Gaussian lạc xa.
- `right_handed`: `false` khi phép chuẩn hóa của gsplat rơi vào nhánh phản xạ. Nhánh này chỉ xảy ra khi hướng lên của thế giới COLMAP đúng bằng +y, hiếm với dữ liệu thật. Khi đó phòng hiện như ảnh gương, và `warnings` của phòng có ghi. Cách xử lý: `--redo-colmap`.

### 4.5 `progress.json` (khi đang dựng)

```json
{
  "format_version": 1, "room": "phong1", "source_kind": "photos", "status": "running", "stage": "train",
  "stages": [
    {"id": "images", "label": "Chuẩn hóa ảnh", "status": "done", "percent": null, "message": null,
     "started_at": "…", "finished_at": "…"},
    {"id": "colmap", "label": "COLMAP: ước lượng camera", "status": "done", "percent": null, "message": null,
     "started_at": "…", "finished_at": "…"},
    {"id": "train", "label": "Train mô hình chính (mcmc)", "status": "running", "percent": 41.3,
     "message": "Vòng 12.390/30.000 · 0,62 triệu Gaussian", "started_at": "…", "finished_at": null},
    {"id": "export", "label": "Xuất mô hình web và thêm vào thư viện", "status": "waiting", "percent": null,
     "message": null, "started_at": null, "finished_at": null}
  ],
  "message": "Vòng 12.390/30.000 · 0,62 triệu Gaussian", "error": null, "job": "room_phong1",
  "log": "/workspace/outputs/logs/build_room_phong1_20261104_013000.log",
  "created_at": "…", "started_at": "…", "updated_at": "…", "finished_at": null
}
```

- `status`: `queued` (chờ GPU) → `running` → `done` | `failed` | `stopped`. Khi `failed`, `error` là câu báo lỗi cho người xem, `stage` là bước bị lỗi.
- Trạng thái của từng bước: `waiting` | `running` | `done` | `skipped` (dùng lại COLMAP đã có) | `failed` | `stopped`.
- Nhãn bước 1: `Tách khung ảnh từ video` hoặc `Chuẩn hóa ảnh`, theo loại nguồn.
- `percent` chỉ có ở bước train, ghi tối đa mỗi 5 giây. Nên đọc lại file mỗi 2–5 giây.
- `started_at` / `finished_at` của từng bước: thời gian thật của mỗi bước (tách khung, COLMAP, train, xuất), dùng cho câu hỏi "chạy mất bao lâu". Bước bỏ qua (dùng lại COLMAP) hoặc chưa chạy thì là `null`.
- Việc nền bị dừng khi còn xếp hàng thì không ai ghi file này. Trạng thái thật của việc nằm ở `$WS/outputs/jobs/room_<phòng>.state` (dòng `state=` cuối cùng; `scripts/pod/job.sh status room_<phòng>`).

### 4.6 Phục vụ file

`web/` là thư mục tĩnh: bất kỳ web server tĩnh nào cũng phục vụ được, hoặc có thể chép nguyên sang nơi đặt trang web. Server của trang web (mục 8) phục vụ thư mục này ở `/web/...`, cùng các API tải lên và tiến độ.

Lưu ý với Runpod: [tài liệu của Runpod](https://docs.runpod.io/pods/configuration/expose-ports) ghi proxy HTTP (Cloudflare) giới hạn mỗi kết nối 100 giây, quá thì lỗi 524. Tài liệu không nói rõ giới hạn tính đến lúc máy chủ bắt đầu trả lời hay đến khi tải xong. Để chắc, nên tải xong file 32 MB trong 100 giây, tức đường truyền từ khoảng 0,5 MB/s. Vì vậy trang web tải video / ảnh lên theo từng phần 8 MiB, và việc dựng phòng luôn chạy nền: trang chỉ đọc `progress.json`. Địa chỉ proxy công khai, nên server cần mật khẩu (mục 8) và chỉ dùng cho dữ liệu demo. Tự dừng pod khi xong việc **chưa làm** trên trang web (`AUTO_STOP=1` của `job.sh` chưa thử trên pod).

## 5. Giới hạn, việc chưa làm

- **Chưa chấm điểm phòng tự dựng** (`metrics` = `null`). Muốn PSNR / SSIM / LPIPS phản ánh góc nhìn mới cần một lượt quay kiểm tra riêng, định vị trong cùng mô hình COLMAP nhưng không dùng để train ("Lượt quay kiểm tra" trong bản thiết kế trang "Dựng phòng mới"). Chưa có code.
- `.splat` bỏ SH bậc 1–3 (mục 4.3). Đơn vị tương đối (mục 4.4).
- `cap_max` cố định 1 triệu: ảnh hưởng tới chất lượng trên phòng lớn chưa đo (mục 2).
- Thời gian và dung lượng đã đo ngày 08/11/2026 trên 7 phòng, mỗi phòng dựng một lần (`docs/RESULTS.md` mục 7); chưa lặp lại.
- Trang web (mục 8): đã chạy trên pod ngày 08/11/2026, tải lên qua proxy của Runpod; FPS trên máy thật mới đo một lần, trong lúc đang quay màn hình (đang đo lại). Chưa có trang Kết quả thực nghiệm của bản thiết kế, chưa tự dừng pod.

## 6. Đọc code

Thuật toán (rasterization, `MCMCStrategy`) là của thư viện gsplat, ở `gsplat/strategy/mcmc.py` của bản checkout. Code của luồng sản phẩm dựng lệnh, kiểm tra điều kiện, chạy và đổi định dạng cho web.

| File | Đọc ở đâu trước |
|---|---|
| `scripts/build_room.sh` | cuối file: trình tự `read_args` → `load_config` → `check_source` → (`queue_job`) → `start_run` → `step_images_colmap` → `check_colmap` → `step_train` → `step_export` → `finish_run`; mỗi bước một hàm |
| `src/product/train_main.py` | `main()`: 5 bước có đánh số, mỗi bước một phần của file. Toàn bộ cấu hình train nằm trong `trainer_command()` |
| `src/product/export_web.py` | `export_room()`: 5 bước có đánh số; công thức đổi sang `.splat` trong `splat_records()`; camera trong `camera_layout()` |
| `src/product/progress.py` | các hàm `queue` / `start` / `set_stage` / `update` / `fail` / `done`, mỗi hàm một lần đổi trạng thái của `progress.json` |
| `configs/product.yaml` | mọi tham số của mô hình chính |

Ngôn ngữ: lỗi về đầu vào của người dùng và chữ hiện trên web (`progress.json`, `warnings`) viết tiếng Việt. Log cho người chạy pod viết tiếng Anh như phần còn lại của repo.

## 7. Kiểm thử

Trong repo (không cần GPU, PyTorch hay gsplat; `python -m pytest tests/unit/test_product.py tests/unit/test_product_export.py tests/unit/test_build_room.py`):
- `configs/product.yaml` khớp chế độ `mcmc` của `configs/base.yaml` (preset, tham số, điều kiện train, commit), trừ hai điểm của mục 2. Lệnh train bằng lệnh của thực nghiệm (cùng `cap_max`) ở mọi tham số trừ `test_every`. Cấu hình sai bị từ chối, chỉ rõ khóa.
- `train_main.py` với `simple_trainer.py` giả:
  - ghi `train_meta.json`, đọc số Gaussian từ PLY, cập nhật % train;
  - không train lại mô hình khớp; train lại khi COLMAP hoặc cấu hình đổi, kết quả cũ được giữ;
  - ghi lỗi của trainer; từ chối nhiều GPU;
  - kiểm tra commit và gói gsplat đã cài (kho git giả);
  - dừng bằng SIGTERM.
- `export_web.py`:
  - bố cục 32 byte của `.splat`; bỏ Gaussian hỏng, trong suốt; thứ tự;
  - `cameras.json` (khung, hướng lên, góc nhìn, biên, phát hiện hệ bị lật) với Parser giả;
  - thư viện: thứ tự, giữ tên, dựng lại manifest; khi không có camera;
  - mô hình chưa train xong hoặc khác cấu hình.
- `build_room.sh` chạy bash thật:
  - dữ liệu cũ trên volume giữ nguyên từng byte, chỉ có file mới trong `outputs/product/`, log và việc nền của phòng;
  - cả hai loại nguồn; dùng lại COLMAP;
  - dừng khi COLMAP FAIL / train lỗi / xuất lỗi, `progress.json` ghi đúng bước;
  - đầu vào sai bị từ chối trước khi tạo gì;
  - `DRY_RUN`; chạy nền và xếp hàng chờ GPU;
  - một lượt với `train_main.py` và `export_web.py` thật.

Ngoài repo, ngày 04/11/2026, trên CPU với PyTorch 2.7.1 và mã nguồn gsplat @6e8c837:
- `simple_trainer.py` thật phân giải lệnh train của sản phẩm thành đúng cấu hình mong muốn: `MCMCStrategy`, `cap_max` 1.000.000, `opacity_reg` 0,01, không đánh giá, PLY và checkpoint ở vòng cuối, tắt viewer. So với lệnh `mcmc` của thực nghiệm cùng `cap_max`: 70 khóa, chỉ khác `test_every`.
- `.splat`: 20.000 Gaussian ngẫu nhiên, PLY do `export_splats` của gsplat ghi. 19.110 bản ghi trùng từng byte RGBA và quaternion với `export_splats(format="splat")` của gsplat. Độ lớn lệch trong mức làm tròn float32. Chỉ khác thứ tự, và 890 Gaussian có A = 0 bị bỏ.
- `cameras.json`: cảnh COLMAP tổng hợp, thế giới COLMAP xoay / co giãn / dời ngẫu nhiên, `Parser` thật của gsplat. Chiếu các điểm khởi tạo Gaussian (đã chuẩn hóa) qua camera trong `cameras.json` ra đúng pixel quan sát gốc, sai số ≤ 0,001 px. `up` lệch hướng lên thật 0,06°. Lần đầu, cảnh tổng hợp có hướng lên đúng bằng +y đã làm lộ nhánh phản xạ trong phép chuẩn hóa của gsplat. Trường `right_handed` và cảnh báo của mục 4.4 được thêm từ đó.

Trên GPU: đã chạy trên pod ngày 08/11/2026, dựng 7 phòng từ video tải lên trang web (`docs/RESULTS.md` mục 7).

## 8. Trang web

Code: `app/web/` (cách chạy, bảng file: `app/web/README.md`). Server HTTP chỉ dùng thư viện chuẩn Python, chạy trên pod cạnh luồng sản phẩm; trang tĩnh dùng three.js 0.180.0 và [Spark](https://github.com/sparkjsdev/spark) 2.4.0 (MIT, WebGL2) để vẽ `model.splat`, phông Be Vietnam Pro 5.3.0 (SIL OFL 1.1, đủ dấu tiếng Việt); ghim phiên bản và mã kiểm tra ở `configs/environment/web_viewer.lock.json`.

Giao diện (làm lại ngày 05/11/2026): thanh bên thu gọn được (Tổng quan, Dự án mới, các dự án gần đây, trạng thái pod); Tổng quan → Dự án mới → Tiến độ → Xem 3D. Trang không tự đặt số: số Gaussian, dung lượng, số ảnh định vị được lấy từ `manifest.json`; thông số dựng ở trang Dự án mới lấy từ `configs/product.yaml` qua trường `pipeline` của `/api/status` (số khung tối đa 250 và cạnh dài 1.600 px là mặc định của `src/input/frames.py`, có test giữ hai nơi khớp nhau); tiến độ tổng là ước lượng theo trọng số đặt tay cho từng bước (`app/web/static/js/ui.js`), chỉ % của bước train là số thật.

```bash
bash scripts/web_server.sh start     # trên pod; rồi Edit Pod -> Expose HTTP Ports: 8000; mở https://<POD_ID>-8000.proxy.runpod.net
```

| Bước của người dùng | Trang | Phía server |
|---|---|---|
| Chọn video hoặc nhiều ảnh, đặt tên | Dự án mới | kiểm tra như `build_room.sh` (loại file, ≥ 10 ảnh, tên phòng chưa dùng) trước khi nhận dữ liệu |
| Tải lên | Dự án mới | nhận từng phần ≤ 8 MiB đúng thứ tự (`offset`), phần lỗi gửi lại; đủ thì chuyển sang `$WS/data/uploads/<phòng>/` |
| Bắt đầu dựng | → Tiến độ | `scripts/build_room.sh <nguồn> <phòng> --name <tên> --background` (việc nền `room_<phòng>`, xếp hàng chờ GPU); lỗi đầu vào của script trả về đúng câu báo lỗi |
| Theo dõi | Tiến độ | đọc `progress.json` (bước, %, thời gian từng bước), cuối log, trạng thái việc nền (pod tắt giữa chừng: "bị gián đoạn"); hủy = `job.sh stop`; dựng lại = `build_room.sh` với nguồn đã tải lên (dùng lại ảnh và COLMAP đã xong) |
| Xem | Xem 3D | `manifest.json`, `model.splat`, `cameras.json` (mục 4); tải `.splat`, PLY đầy đủ |

Trình xem không xoay mô hình: camera three.js đặt theo `cameras.json` (khung OpenCV: phải = forward × up, lên = up, nhìn theo forward). Ba chế độ: xoay quanh (mở đúng góc `default_view`), đi bộ (hướng lên là `up` của phòng, giới hạn trong hộp bao các camera nới 0,25 × `scene_radius` giao với `bounds`), tham quan (đường Catmull-Rom qua vị trí các khung, hướng nhìn làm mượt). Mức chất lượng Vừa / Nhẹ vẽ 50 % / 20 % Gaussian đầu file, tức các Gaussian lớn và đục nhất (thứ tự của mục 4.3).

Bảo mật: link proxy công khai nên mọi thứ trừ khung trang cần mật khẩu. Đăng nhập cấp một phiên ngẫu nhiên giữ trong bộ nhớ của server (cookie HttpOnly, SameSite=Strict, tối đa 200 phiên; đăng xuất hoặc khởi động lại server là hết hiệu lực); request ghi phải có header `X-SR3D: 1`; file chỉ đọc trong thư mục của nó, không đọc file ẩn; tên file tải lên được làm sạch, tên phòng theo quy tắc của `build_room.sh` (không dùng được `web`, `uploads`, `server`); tối đa 3 lượt tải lên dở cùng lúc. Không mật khẩu chỉ cho chế độ chỉ xem trên `127.0.0.1`. Request lạ (mật khẩu ngoài ASCII, `Content-Length` sai, đường dẫn có ký tự NUL) được trả 4xx, không làm lỗi server.

Câu trả lời của `build_room.sh --background` không phải lúc nào cũng là "đã chạy": `job.sh` trả mã 0 khi việc cũ cùng tên còn sống ("không chạy thêm bản thứ hai"), và lệnh có thể chậm. Server coi câu đó là lỗi, chỉ chờ 80 giây (dưới giới hạn 100 giây của proxy), hết giờ thì xem việc nền `room_<phòng>` đã có chưa: có thì coi như đã xếp hàng, chưa thì báo lỗi và xóa nguồn vừa tải lên. Dựng lại bị từ chối khi tiến trình của việc cũ còn sống; `progress.json` còn ghi "đang chạy" của một lần dựng đã mất (pod tắt) thì được ghi "đã dừng" trước, để `progress.py queue` ghi được lần xếp hàng mới.

Kiểm thử (CPU, `app/web/README.md` mục Kiểm thử): API qua HTTP thật; trọn luồng tải video lên → `build_room.sh` thật → `train_main.py` / `export_web.py` thật (gsplat giả) → phòng trong thư viện; script `web_server.sh`; trong Chromium headless (WebGL2 bằng SwiftShader): tải 12 ảnh lên → tiến độ → xem 3D, và điểm mốc của phòng tổng hợp hiện đúng pixel mà `cameras.json` tính ra (trung vị sai lệch ≤ 3 px). Lỗi tìm được nhờ test: lệnh `web_server.sh start` không trả về khi đầu ra là pipe (vd. ô `!` của Jupyter), đã sửa. Một lượt rà soát bảo mật độc lập phần server (04/11/2026) không thấy lỗi mức cao; các lỗi mức trung bình / thấp đã sửa, mỗi lỗi có test: request lạ gây lỗi 500 (mật khẩu ngoài ASCII, `Content-Length` như "²", đường dẫn có NUL hoặc tên quá dài), dựng lại báo "đã bắt đầu" khi `job.sh` không chạy gì, cookie là giá trị cố định suy ra từ mật khẩu (đăng xuất không thu hồi được), tên hiển thị bắt đầu bằng "-" làm `export_web.py` lỗi, `web_server.sh run` mang token của việc nền đang gọi nó, `status` in sai cổng khi `PORT` khác lúc `start`. Tìm thêm khi viết test cho các lỗi đó: lỗi 411 / 413 của body JSON bị báo thành "JSON không hợp lệ".
