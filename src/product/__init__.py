"""Luồng sản phẩm: dựng một phòng cho web bằng mô hình chính (configs/product.yaml).

Đầu vào: một video hoặc một thư mục ảnh của căn phòng. Các bước (scripts/build_room.sh gọi lần lượt):
  1. chuẩn bị ảnh + COLMAP: dùng chung bước chuẩn bị dữ liệu với luồng thực nghiệm (scripts/pipeline/prepare_scene.sh);
  2. train mô hình chính: train_main.py (gọi thẳng simple_trainer.py của gsplat, không qua code thực nghiệm src/training/);
  3. xuất cho web: export_web.py (model.splat, cameras.json, thumbnail.jpg, manifest.json).
Tiến độ cho web: progress.py (progress.json của từng phòng). Hợp đồng với web: docs/design/product.md.
"""
