"""Ảnh so sánh nhiều lượt trên cùng ảnh test (ảnh gốc, ảnh render, vùng cắt phóng to, bản đồ sai số) và video quỹ đạo
của các lượt xếp chồng. Không train, không dùng GPU: chỉ đọc ảnh và video mà simple_trainer.py đã ghi khi chấm.

    python -m src.evaluation.panels --runs <lượt 1> <lượt 2> ... --out <thư mục> [--images 0 12 24]
        [--crop x,y,w,h] [--labels a b c] [--video]
    python -m src.evaluation.panels --scene-dir <thư mục kết quả của cảnh> [--seed 42] --out <thư mục> ...

- `--scene-dir`: các lượt đã xong (status OK), đúng protocol chung (như bảng so sánh mặc định), không có tag, của một
  seed trong thư mục kết quả của cảnh (`default`, `mcmc` trước, rồi các chế độ khác theo tên); bỏ thư mục cũ `.prev_*`.
  Lượt có tag (biến thể của một kế hoạch dò tham số), lượt khác protocol hay bản đã lọc (`<lượt>/filtered`) chỉ vào
  khi chỉ rõ bằng `--runs`.
- `<lượt>/renders/val_step<N>_<i>.png` là [ảnh gốc | ảnh render] của ảnh test thứ i (lấy bước N lớn nhất). Các lượt
  phải có cùng các ảnh test với cùng ảnh gốc (cùng cảnh, cùng tập test, cùng độ phân giải), nếu không thì dừng. Chấm ở
  bước khác nhau (vd. một chế độ 40.000 vòng) thì chỉ cảnh báo; cột "Bước" của panels.md ghi bước của từng lượt.
- PSNR từng ảnh tính lại từ PNG 8 bit, chỉ để chọn và chú thích ảnh; chỉ số của báo cáo là của gsplat (ảnh float).
- Vùng cắt: `--crop` (tỉ lệ theo chiều rộng / cao, vd. 0.5,0.4,0.25,0.25), hoặc tự chọn theo một quy tắc cố định: ô
  1/4 × 1/4 khung hình có tổng sai số (cộng trên mọi lượt) lớn nhất. Ô đó thường là chỗ hỏng nặng nhất của lượt kém
  nhất, và thêm hay bớt lượt có thể làm đổi ô. Quy tắc đã dùng ghi trong panels.json.
- Ảnh mặc định: ảnh test đầu, giữa và cuối. PSNR (PNG) của mọi ảnh test luôn được ghi vào panels.json và panels.md.
- Bản đồ sai số: |render - gốc| trung bình 3 kênh, nhân 4 rồi cắt ở 1 (sai số 0,25 trở lên là màu sáng nhất), cùng
  thang cho mọi lượt.
- `--video`: ghép `videos/traj_<N>.mp4` (nửa trái màu, nửa phải độ sâu) của các lượt thành một video, mỗi lượt một
  hàng có nhãn, dừng ở video ngắn nhất. Nửa độ sâu do gsplat chuẩn hóa min–max theo từng khung của từng lượt: so hình
  dạng, không so giá trị giữa các hàng. Video ghi bằng H.264 (imageio-ffmpeg, phát được trong trình duyệt) khi có, không
  thì mp4v của OpenCV. Lượt thiếu video hay không ghi được video thì chỉ cảnh báo: ảnh so sánh vẫn được giữ.

Kết quả: <out>/panel_<i>.png, <out>/panels.json, <out>/panels.md, <out>/traj_compare.mp4 (khi --video).
Cần numpy và opencv (env gs-gsplat có sẵn, kèm imageio-ffmpeg).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

RENDER = re.compile(r"^val_step(\d+)_(\d{4})\.png$")
VIDEO = re.compile(r"^traj_(\d+)\.mp4$")
THUMB_WIDTH = 480  # chiều rộng ảnh toàn khung trong panel
ERROR_GAIN = 4.0  # bản đồ sai số: sai số 1 / ERROR_GAIN trở lên là màu sáng nhất
LABEL_HEIGHT = 26
MAX_PSNR = 100.0  # hai ảnh PNG giống hệt nhau (PSNR vô hạn) ghi là 100 dB
MAX_VIDEO_HEIGHT = 4096  # video ghép cao hơn thì thu nhỏ thêm (bộ mã hóa giới hạn kích thước khung)
AUTO_CROP_RULE = ("auto: the 1/4 x 1/4 window with the largest error summed over all runs (usually where the worst run "
                  "fails most; adding or removing runs can move it)")


class PanelError(ValueError):
    pass


def run_label(run: Path) -> str:
    """Nhãn mặc định: tên thư mục lượt; bản đã lọc `<lượt>/filtered` là `<lượt>+filtered`."""
    run = Path(run)
    return f"{run.parent.name}+filtered" if run.name == "filtered" else run.name


def scene_runs(scene_dir: Path, seed: int) -> list[Path]:
    """Các lượt đã xong, không có tag, của `seed` trong thư mục kết quả của cảnh, có ảnh render: default, mcmc trước."""
    from src.evaluation.gsplat_results import load_summary

    found = []
    for meta_path in sorted(Path(scene_dir).glob("*/run_meta.json")):
        folder = meta_path.parent
        if ".prev_" in folder.name or not (folder / "renders").is_dir():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("seed") != seed or meta.get("tag"):
            continue
        summary = load_summary(folder)
        if summary.status != "OK" or summary.deviations:  # chưa xong, hoặc khác protocol chung
            continue
        found.append((meta.get("mode") or folder.name, folder))
    order = {"default": 0, "mcmc": 1}
    return [run for _, run in sorted(found, key=lambda item: (order.get(item[0], 2), item[0], item[1].name))]


def latest_renders(run: Path) -> tuple[int, dict[int, Path]]:
    """(bước, {chỉ số ảnh test: file}) của bước chấm lớn nhất trong <lượt>/renders/."""
    found: dict[int, dict[int, Path]] = {}
    for path in sorted((Path(run) / "renders").glob("val_step*_*.png")):
        match = RENDER.match(path.name)
        if match:
            found.setdefault(int(match.group(1)), {})[int(match.group(2))] = path
    if not found:
        raise PanelError(f"no renders/val_step<N>_<i>.png in {run}")
    step = max(found)
    return step, found[step]


def latest_video(run: Path) -> Path | None:
    videos = [(int(m.group(1)), p) for p in (Path(run) / "videos").glob("traj_*.mp4") if (m := VIDEO.match(p.name))]
    return max(videos)[1] if videos else None


def split_render(path: Path):
    """(ảnh gốc, ảnh render), mỗi ảnh BGR uint8, từ một PNG [gốc | render] của gsplat."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise PanelError(f"cannot read {path}")
    width = image.shape[1]
    if width % 2:
        raise PanelError(f"{path}: odd width {width}, not a [ground truth | render] image")
    return image[:, :width // 2], image[:, width // 2:]


def psnr(render, truth) -> float:
    import numpy as np

    mse = float(np.mean((render.astype(np.float64) - truth.astype(np.float64)) ** 2)) / 255.0 ** 2
    return MAX_PSNR if mse == 0 else min(MAX_PSNR, 10 * math.log10(1.0 / mse))


def error_map(render, truth):
    """Sai số |render - gốc| trung bình 3 kênh, trong [0, 1]."""
    import numpy as np

    return np.abs(render.astype(np.float32) - truth.astype(np.float32)).mean(axis=2) / 255.0


def auto_crop(errors: list, fraction: float = 0.25) -> tuple[int, int, int, int]:
    """Ô fraction × fraction khung hình có tổng sai số (cộng trên mọi lượt) lớn nhất: (x, y, w, h) theo pixel."""
    import numpy as np

    total = np.sum(np.stack(errors), axis=0)
    height, width = total.shape
    w, h = max(1, int(round(width * fraction))), max(1, int(round(height * fraction)))
    integral = np.pad(total.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    step_x, step_y = max(1, width // 32), max(1, height // 32)
    best, where = -1.0, (0, 0)
    for y in range(0, height - h + 1, step_y):
        for x in range(0, width - w + 1, step_x):
            value = integral[y + h, x + w] - integral[y, x + w] - integral[y + h, x] + integral[y, x]
            if value > best:
                best, where = value, (x, y)
    return where[0], where[1], w, h


def parse_crop(text: str, width: int, height: int) -> tuple[int, int, int, int]:
    try:
        x, y, w, h = (float(part) for part in text.split(","))
    except ValueError as error:
        raise PanelError(f"--crop must be x,y,w,h fractions, e.g. 0.5,0.4,0.25,0.25: {text}") from error
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 - x and 0 < h <= 1 - y):
        raise PanelError(f"--crop outside the image: {text}")
    return int(x * width), int(y * height), max(1, int(w * width)), max(1, int(h * height))


def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def _labelled(image, text: str):
    """Ảnh có một dải nhãn ở trên (chữ ASCII: font của OpenCV không có dấu tiếng Việt)."""
    import cv2
    import numpy as np

    strip = np.full((LABEL_HEIGHT, image.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(strip, _ascii(text), (6, LABEL_HEIGHT - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return np.vstack([strip, image])


def _resize(image, width: int | None = None, height: int | None = None):
    import cv2

    h, w = image.shape[:2]
    if width is not None:
        size = (width, max(1, int(round(h * width / w))))
    else:
        size = (max(1, int(round(w * height / h))), height)
    interpolation = cv2.INTER_AREA if size[0] < w else cv2.INTER_NEAREST  # phóng to giữ nguyên điểm ảnh
    return cv2.resize(image, size, interpolation=interpolation)


def _colorize(error):
    import cv2
    import numpy as np

    scaled = np.clip(error * ERROR_GAIN, 0, 1)
    return cv2.applyColorMap((scaled * 255).astype(np.uint8), getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET))


def build_panel(truth, renders: list, labels: list[str], crop: tuple[int, int, int, int], index: int):
    """Hàng đầu: ảnh gốc (khung đỏ là vùng cắt) và vùng cắt; mỗi lượt một hàng: ảnh render, vùng cắt, bản đồ sai số."""
    import cv2
    import numpy as np

    x, y, w, h = crop
    thumb_scale = THUMB_WIDTH / truth.shape[1]

    def thumb(image):
        small = _resize(image, width=THUMB_WIDTH)
        p0 = (int(x * thumb_scale), int(y * thumb_scale))
        p1 = (int((x + w) * thumb_scale), int((y + h) * thumb_scale))
        cv2.rectangle(small, p0, p1, (0, 0, 255), 2)
        return small

    row_height = thumb(truth).shape[0]

    def zoom(image):
        return _resize(image[y:y + h, x:x + w], height=row_height)

    blank = np.zeros_like(zoom(truth))
    rows = [np.hstack([_labelled(thumb(truth), f"ground truth (test image {index})"),
                       _labelled(zoom(truth), "crop"), _labelled(blank, "")])]
    for render, label in zip(renders, labels):
        error = error_map(render, truth)
        rows.append(np.hstack([_labelled(thumb(render), f"{label}  PSNR {psnr(render, truth):.2f} dB (PNG)"),
                               _labelled(zoom(render), "crop"),
                               _labelled(_resize(_colorize(error)[y:y + h, x:x + w], height=row_height),
                                         f"|error| x{ERROR_GAIN:g}")]))
    return np.vstack(rows)


class _VideoOut:
    """H.264 qua imageio-ffmpeg khi có (phát được trong trình duyệt), không thì mp4v của OpenCV."""

    def __init__(self, path: Path, fps: float, width: int, height: int):
        self.codec = None
        if importlib.util.find_spec("imageio") and importlib.util.find_spec("imageio_ffmpeg"):
            import imageio.v2 as imageio
            import numpy as np

            self._writer = imageio.get_writer(str(path), fps=fps, codec="libx264", pixelformat="yuv420p",
                                              macro_block_size=2, quality=8, ffmpeg_log_level="error")
            self._write = lambda frame: self._writer.append_data(np.ascontiguousarray(frame[:, :, ::-1]))  # BGR -> RGB
            self._close = self._writer.close
            self.codec = "h264"
        else:
            import cv2

            self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not self._writer.isOpened():
                raise PanelError(f"cannot write {path}")
            self._write, self._close = self._writer.write, self._writer.release
            self.codec = "mp4v"

    def write(self, frame) -> None:
        self._write(frame)

    def close(self) -> None:
        self._close()


def stack_videos(videos: list[Path], labels: list[str], out: Path, scale: float = 0.5) -> tuple[int, str]:
    """Ghép các video quỹ đạo theo chiều dọc, mỗi video một hàng có nhãn; trả về (số khung đã ghi, bộ mã hóa)."""
    import cv2
    import numpy as np

    readers = [cv2.VideoCapture(str(path)) for path in videos]
    writer = None
    try:
        if not all(reader.isOpened() for reader in readers):
            raise PanelError("cannot open " + ", ".join(str(p) for p, r in zip(videos, readers) if not r.isOpened()))
        fps = readers[0].get(cv2.CAP_PROP_FPS) or 30.0
        frames, width = 0, None
        while True:
            images = []
            for reader in readers:
                ok, frame = reader.read()
                if not ok:
                    break
                images.append(frame)
            if len(images) < len(readers):
                break
            if width is None:  # khung đầu: chọn cỡ sao cho video ghép không quá MAX_VIDEO_HEIGHT
                width = max(2, int(images[0].shape[1] * scale) // 2 * 2)
                per_width = sum(image.shape[0] / image.shape[1] for image in images)  # chiều cao / chiều rộng
                available = MAX_VIDEO_HEIGHT - (LABEL_HEIGHT + 1) * len(images)  # dải nhãn, làm tròn mỗi hàng
                if available < 2 * per_width:
                    raise PanelError(f"too many runs ({len(images)}) for one video of at most {MAX_VIDEO_HEIGHT} px")
                width = min(width, max(2, int(available / per_width) // 2 * 2))
            canvas = np.vstack([_labelled(_resize(image, width=width), label) for image, label in zip(images, labels)])
            canvas = canvas[:canvas.shape[0] // 2 * 2]  # bộ mã hóa cần kích thước chẵn
            if writer is None:
                writer = _VideoOut(out, fps, canvas.shape[1], canvas.shape[0])
            writer.write(canvas)
            frames += 1
        if not frames:
            raise PanelError("no frame could be read from " + ", ".join(str(path) for path in videos))
    except PanelError:
        raise
    except Exception as error:  # lỗi của bộ mã hóa / ffmpeg: báo như lỗi của bước video
        raise PanelError(f"cannot write {out}: {type(error).__name__}: {error}") from error
    finally:
        for reader in readers:
            reader.release()
        if writer is not None:
            try:
                writer.close()
            except Exception as error:  # ffmpeg / OpenCV báo lỗi lúc đóng file
                raise PanelError(f"cannot write {out}: {type(error).__name__}: {error}") from error
    verify_video(out, frames)
    return frames, writer.codec


def verify_video(path: Path, frames: int) -> None:
    """Mở lại video vừa ghi: bộ mã hóa có thể lỗi mà không báo (imageio-ffmpeg không kiểm mã thoát của ffmpeg)."""
    import cv2

    if not path.is_file() or path.stat().st_size == 0:
        raise PanelError(f"cannot write {path}")
    reader = cv2.VideoCapture(str(path))
    try:
        ok = reader.isOpened() and reader.read()[0]
        count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        reader.release()
    if not ok or (count and count != frames):
        raise PanelError(f"{path} is not a readable video of {frames} frames (read back: {count})")


def _write_record(out: Path, record: dict) -> None:
    (out / "panels.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "panels.md").write_text(markdown(record), encoding="utf-8")


def make_panels(runs: list[Path], out: Path, labels: list[str] | None = None, images: list[int] | None = None,
                crop: str | None = None, video: bool = False) -> dict:
    import cv2

    runs = [Path(run) for run in runs]
    labels = labels or [run_label(run) for run in runs]
    if len(labels) != len(runs):
        raise PanelError(f"{len(labels)} labels for {len(runs)} runs")
    if len(set(labels)) != len(labels):
        raise PanelError(f"labels must be distinct: {labels}")
    found = [latest_renders(run) for run in runs]
    steps = {label: step for label, (step, _) in zip(labels, found)}
    warnings = []
    if len(set(steps.values())) > 1:
        warnings.append(f"the runs were evaluated at different steps: {steps}")
    index_sets = [set(files) for _, files in found]
    if any(indices != index_sets[0] for indices in index_sets[1:]):
        sizes = {label: len(indices) for label, indices in zip(labels, index_sets)}
        raise PanelError(f"the runs have different test images (number of renders: {sizes})")
    common = sorted(index_sets[0])
    wanted = sorted(set(images)) if images else sorted({common[0], common[len(common) // 2], common[-1]})
    missing = [index for index in wanted if index not in index_sets[0]]
    if missing:
        raise PanelError(f"test images {missing} are not in the runs (available: {common[0]}..{common[-1]})")

    psnrs: dict[str, dict[int, float]] = {label: {} for label in labels}
    loaded = {}
    for index in common:  # đọc từng ảnh: chỉ giữ trong bộ nhớ các ảnh sẽ vẽ panel
        pairs = [split_render(files[index]) for _, files in found]
        truth = pairs[0][0]
        for (other, _), run in zip(pairs[1:], runs[1:]):
            if other.shape != truth.shape or not (other == truth).all():
                raise PanelError(f"test image {index}: {run} has a different ground truth than {runs[0]} "
                                 "(not the same scene, test split or resolution)")
        for (_, render), label in zip(pairs, labels):
            psnrs[label][index] = psnr(render, truth)
        if index in wanted:
            loaded[index] = (truth, [render for _, render in pairs])

    out.mkdir(parents=True, exist_ok=True)
    record = {"runs": [str(run) for run in runs], "labels": labels, "steps": steps, "test_images": common,
              "images_rule": "given (--images)" if images else "default: first, middle and last test image",
              "panels": [], "psnr_png": {label: {str(i): v for i, v in values.items()} for label, values in psnrs.items()},
              "mean_psnr_png": {label: sum(v.values()) / len(v) for label, v in psnrs.items()},
              "error_gain": ERROR_GAIN, "warnings": warnings}
    for index in wanted:
        truth, renders = loaded[index]
        height, width = truth.shape[:2]
        if crop:
            box, rule = parse_crop(crop, width, height), "given (--crop)"
        else:
            box, rule = auto_crop([error_map(render, truth) for render in renders]), AUTO_CROP_RULE
        path = out / f"panel_{index:04d}.png"
        if not cv2.imwrite(str(path), build_panel(truth, renders, labels, box, index)):
            raise PanelError(f"cannot write {path}")
        record["panels"].append({"image": index, "file": path.name, "crop": dict(zip("xywh", box)), "crop_rule": rule})
    _write_record(out, record)  # ghi trước bước video: lỗi ở video không làm mất ảnh so sánh
    if video:
        videos = [latest_video(run) for run in runs]
        lacking = [label for label, path in zip(labels, videos) if path is None]
        if lacking:
            record["video"] = {"error": "no videos/traj_<N>.mp4 in " + ", ".join(lacking)}
        else:
            try:
                frames, codec = stack_videos(videos, labels, out / "traj_compare.mp4")
                record["video"] = {"file": "traj_compare.mp4", "frames": frames, "codec": codec,
                                   "sources": [str(v) for v in videos]}
            except PanelError as error:
                record["video"] = {"error": str(error)}
        _write_record(out, record)
    return record


def markdown(record: dict) -> str:
    labels = record["labels"]
    lines = ["# Ảnh so sánh", "",
             "PSNR tính lại từ ảnh PNG 8 bit của lượt chấm (gần đúng, chỉ để chọn và chú thích ảnh); chỉ số chính thức là "
             "bảng so sánh của src.evaluation.compare_modes.", "",
             "| Lượt | PSNR trung bình (PNG) | Bước |", "|---|---|---|"]
    lines += [f"| {label} | {record['mean_psnr_png'][label]:.3f} | {record['steps'][label]} |" for label in labels]
    lines += ["", "| Ảnh test | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for index in record["test_images"]:
        lines.append(f"| {index} | " + " | ".join(f"{record['psnr_png'][label][str(index)]:.2f}" for label in labels)
                     + " |")
    lines += ["", f"Panel ({record['images_rule']}; hàng đầu: ảnh gốc; mỗi lượt một hàng: ảnh render, vùng cắt phóng to, "
                  f"bản đồ sai số |render - gốc| × {record['error_gain']:g}):", ""]
    for panel in record["panels"]:
        crop = panel["crop"]
        lines.append(f"- `{panel['file']}`: ảnh test {panel['image']}, vùng cắt x={crop['x']} y={crop['y']} "
                     f"w={crop['w']} h={crop['h']} ({panel['crop_rule']})")
    lines += [f"- cảnh báo: {warning}" for warning in record.get("warnings", [])]
    video = record.get("video")
    if video and video.get("error"):
        lines.append(f"- video: không ghép được ({video['error']})")
    elif video:
        lines.append(f"- `{video['file']}`: {video['frames']} khung ({video['codec']}), mỗi lượt một hàng theo thứ tự "
                     "của bảng trên; nửa trái màu, nửa phải độ sâu. Độ sâu được gsplat chuẩn hóa theo từng khung của "
                     "từng lượt: so hình dạng, không so giá trị giữa các hàng")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs", nargs="+", type=Path, help="thư mục các lượt, cùng cảnh, cùng tập test")
    source.add_argument("--scene-dir", type=Path, help="thư mục kết quả của cảnh: các lượt không có tag của --seed")
    parser.add_argument("--seed", type=int, default=42, help="với --scene-dir (mặc định 42)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--labels", nargs="+", help="nhãn của từng lượt (mặc định: tên thư mục)")
    parser.add_argument("--images", nargs="+", type=int, help="chỉ số ảnh test (mặc định: đầu, giữa, cuối)")
    parser.add_argument("--crop", help="vùng cắt x,y,w,h theo tỉ lệ (mặc định: tự chọn, xem đầu file)")
    parser.add_argument("--video", action="store_true", help="ghép video quỹ đạo của các lượt")
    args = parser.parse_args(argv)
    runs = args.runs or scene_runs(args.scene_dir, args.seed)
    if not runs:
        print(f"ERROR: no finished run of seed {args.seed} without a tag and with renders/ in {args.scene_dir}",
              file=sys.stderr)
        return 1
    try:
        record = make_panels(runs, args.out, args.labels, args.images, args.crop, args.video)
    except PanelError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    video = record.get("video") or {}
    print(f"{len(record['panels'])} panel(s) in {args.out}" + (" + traj_compare.mp4" if video.get("file") else ""))
    for label in record["labels"]:
        print(f"  {label}: PSNR (PNG) trung bình {record['mean_psnr_png'][label]:.3f} trên {len(record['test_images'])} "
              "ảnh test")
    for warning in record["warnings"]:
        print(f"WARN: {warning}", file=sys.stderr)
    if video.get("error"):
        print(f"WARN: video skipped: {video['error']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
