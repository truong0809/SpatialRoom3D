# app/web — trang web của sản phẩm

Luồng người dùng: **tải ảnh hoặc video của căn phòng lên → pod GPU tự dựng phòng → xem tiến độ → xem mô hình 3D**.

Trạng thái (05/11/2026): **đã viết, đã kiểm thử trên CPU** (unit test qua HTTP thật, trọn luồng với `build_room.sh` thật và gsplat giả, trình duyệt Chromium headless với phòng tổng hợp). **Chưa chạy trên pod**, chưa có phòng thật nào dựng bằng trang này. Giao diện làm lại ngày 05/11/2026 (bảng điều khiển nền tối, thanh bên thu gọn được); API giữ nguyên, chỉ thêm hai trường chỉ đọc (`pipeline` trong `/api/status`, `last_build` trong `/api/rooms`).

| Màn | Địa chỉ | Làm gì |
|---|---|---|
| Tổng quan | `#/` | Cảnh 3D mới nhất (ảnh lớn, số Gaussian, dung lượng, số ảnh định vị được), ô kéo thả để tạo dự án, dự án đang dựng / lỗi (tiến độ tổng), các cảnh khác, lịch sử xử lý |
| Dự án mới | `#/new` | Chọn video hoặc nhiều ảnh (kéo thả cả thư mục được), xem trước, bỏ bớt ảnh, đặt tên; thông số dựng đọc từ `configs/product.yaml`; kiểm tra trước khi dựng; tải lên theo từng phần 8 MiB, rồi server gọi `scripts/build_room.sh` |
| Tiến độ | `#/room/<phòng>` | Tiến độ tổng (ước lượng: trọng số đặt tay cho từng bước trong `js/ui.js`, chưa đo trên pod), 4 bước với thời gian, nhật ký, hủy việc, dựng lại; lỗi thì hiện bước đã dừng |
| Xem 3D | `#/view/<phòng>` | Xoay quanh, đi bộ (W A S D, bản đồ nhỏ, giữ trong vùng đã quay), tham quan theo đường quay; camera COLMAP; 12 góc nhìn lấy đều trên đường quay; chụp ảnh, quay video, toàn màn hình; bảng thông tin ẩn / hiện (phím I); mức chất lượng; tải `.splat` / PLY; đo FPS và thời gian tải |

Chưa làm: trang Kết quả thực nghiệm, và "lượt quay kiểm tra" để chấm điểm phòng tự dựng (`docs/design/product.md` mục 5). Trang không tự đặt số liệu: mọi con số lấy từ `manifest.json`, `progress.json`, `configs/product.yaml` hoặc đo trên máy đang xem.

## Chạy trên pod

```bash
cd /workspace/SpatialRoom3D
bash scripts/web_server.sh start     # lần đầu tải thư viện xem 3D và tạo mật khẩu; in địa chỉ và mật khẩu
bash scripts/web_server.sh status    # xem lại địa chỉ, mật khẩu
bash scripts/web_server.sh stop      # việc dựng phòng đang chạy vẫn chạy tiếp
```

Trên trang Runpod, một lần cho mỗi pod: **Edit Pod → Expose HTTP Ports → thêm `8000`**. Sau đó mở `https://<POD_ID>-8000.proxy.runpod.net` (lệnh `start` in sẵn địa chỉ này) và nhập mật khẩu.

- Cần cài môi trường trước (`scripts/pod/run_all.sh`), vì trang web gọi đúng luồng sản phẩm (`build_room.sh` → env `gs-gsplat`, GPU).
- Video / ảnh tải lên nằm ở `/workspace/data/uploads/<phòng>/`; phòng dựng xong vào `/workspace/outputs/product/` như khi chạy `build_room.sh` bằng tay (`docs/design/product.md` mục 4).
- Mỗi lúc chỉ một việc dùng GPU: phòng mới xếp hàng sau việc đang chạy (`scripts/pod/job.sh`, tên việc `room_<phòng>`).
- Link proxy của Runpod là công khai. Mọi thứ trừ khung trang cần mật khẩu (`/workspace/outputs/product/server/token`, quyền 600; đổi: xóa file rồi `stop` / `start`). Đăng nhập xong là một phiên ngẫu nhiên trong bộ nhớ của server: đăng xuất hoặc khởi động lại server thì phải nhập lại mật khẩu. Chỉ dùng cho dữ liệu demo.
- Tối đa 3 lượt tải lên dở cùng lúc (lượt ngừng nhận quá 30 phút nhường chỗ cho lượt mới; bỏ dở quá 2 ngày thì bị xóa). Tên hiển thị đặt lúc tải lên nằm cạnh nguồn, trong file ẩn `data/uploads/<phòng>/.sr3d_upload.json`.

## Xem trên máy khác (laptop, không có pod)

Chép thư mục `/workspace/outputs/product/web/` về máy, rồi từ thư mục repo:

```bash
python3 -m app.web.vendor          # một lần, cần Internet: tải three.js, Spark, phông chữ vào app/web/static/vendor/
python3 -m app.web.server --web-dir <thư mục web đã chép> --view-only --host 127.0.0.1 --no-auth --port 8000
```

Mở `http://127.0.0.1:8000`. Chế độ này chỉ xem (không tải lên, không dựng), không cần PyYAML hay GPU, chạy được khi không có mạng: dùng làm phương án dự phòng lúc bảo vệ. `--no-auth` chỉ được dùng cùng `--view-only --host 127.0.0.1`, và khi đó server chỉ nhận địa chỉ `127.0.0.1` / `localhost` (trang web lạ không gọi được API qua tên miền trỏ về máy này).

## Code

| File | Vai trò |
|---|---|
| `server.py` | Server HTTP (thư viện chuẩn Python): trang tĩnh, thư mục web, API tải lên / phòng / log / hủy / dựng lại, mật khẩu |
| `uploads.py` | Phiên tải lên: kiểm tra đầu vào (cùng quy tắc với `build_room.sh`), nhận từng phần đúng thứ tự, chuyển thành nguồn của phòng |
| `rooms.py` | Đọc `manifest.json`, `progress.json`, trạng thái việc nền của `job.sh`, cuối log, file PLY |
| `vendor.py` | Tải three.js 0.180.0, Spark 2.4.0 và phông Be Vietnam Pro 5.3.0 (SIL OFL 1.1) đúng phiên bản và mã kiểm tra trong `configs/environment/web_viewer.lock.json` |
| `static/` | Trang web: `index.html` (khung: thanh bên, hộp đăng nhập), `app.css`, `js/` (`main` điều hướng và thanh bên, `dashboard` tổng quan, `upload` dự án mới, `job` tiến độ, `viewer` xem 3D, `ui` biểu tượng / trạng thái / tiến độ tổng, `camera` toán camera, `api`, `fmt`) |
| `../../scripts/web_server.sh` | Chạy nền / dừng / xem trạng thái server trên pod |

Server không giữ khóa GPU và không train gì: nó gọi `scripts/build_room.sh <nguồn> <phòng> --name <tên> --background`, rồi chỉ đọc các file luồng sản phẩm ghi ra. Hệ tọa độ: `model.splat` và `cameras.json` cùng hệ đã chuẩn hóa của gsplat, trình xem không xoay mô hình, hướng lên lấy từ trường `up` (`docs/design/product.md` mục 4.4).

## Kiểm thử

```bash
python -m pytest tests/unit/test_web_server.py tests/unit/test_web_uploads.py -q       # không cần trình duyệt
SR3D_BROWSER=1 python -m pytest tests/browser -q    # cần Playwright + Chromium; vài phút (WebGL bằng CPU)
```

Test trình duyệt kiểm tra trọn luồng (đăng nhập → tải 12 ảnh lên → tiến độ đi qua 4 bước → xem 3D) và camera của trình xem khớp mô hình: các điểm mốc của phòng tổng hợp (`tests/helpers/synthetic_room.py`) hiện đúng pixel mà `cameras.json` tính ra. FPS đo trong sandbox không có ý nghĩa (WebGL chạy bằng CPU): đo trên máy thật bằng nút "Đo trên máy này" của trang xem 3D.
