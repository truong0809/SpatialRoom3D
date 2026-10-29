"""Biểu đồ tổng cho báo cáo so sánh của src.evaluation.compare_modes (và của src.evaluation.compare_reference: B0 so
với số liệu Inria công bố): mỗi phương pháp hơn hay kém mốc bao nhiêu khi gộp mọi cảnh và seed, và mốc có khớp Inria
không.

    python -m src.evaluation.charts --report <báo cáo>.json [<báo cáo>.json ...] [--out <thư mục>] [--append-to <.md>]

- Các ảnh, theo thứ tự đọc (`01_...`, `02_...`):
  1. Đối chiếu mốc với Inria: default (gsplat) so với B0 tự chạy từng cảnh và với số công bố, khi báo cáo có phần đối
     chiếu (nhiều báo cáo: mỗi báo cáo một khung). Báo cáo chỉ B0 (compare_reference): B0 so với số công bố.
  2. Chênh lệch tổng so với <mốc>, một ảnh cho mỗi mốc (vd. default, mcmc): mỗi phương pháp một thanh, là chênh lệch
     trung bình trên mọi cặp (cảnh, seed) chung với mốc, gộp các báo cáo cùng mốc (vd. các benchmark của một thực
     nghiệm; khi đó mỗi phương pháp có thêm một dòng cho mỗi benchmark); chấm đen: từng cặp; số ghi: chênh lệch trung
     bình và số cặp tốt hơn / số cặp.
  Số từng cảnh, chi phí, dung lượng: ở các bảng của báo cáo.
- Chỉ đọc file JSON của báo cáo (không đọc lại các lượt chạy): số trên ảnh là số của báo cáo (đối chiếu Inria, B0),
  hoặc tính thẳng từ các số đó (trung bình, độ lệch chuẩn và số cặp tốt hơn trên các cặp của mục `pairs`: với một báo
  cáo là đúng các số ở bảng số cặp tốt hơn của nó). index.md ghi đủ bảng số.
- Ghi vào --out (mặc định với một báo cáo: `<báo cáo>_charts/` cạnh file JSON) các ảnh PNG và `index.md`: mỗi biểu đồ
  có cách đọc và bảng số đúng như trên ảnh. Biểu đồ thiếu dữ liệu, hoặc phần báo cáo cần cho nó hỏng, thì không vẽ,
  index.md ghi lý do; các biểu đồ khác vẫn vẽ. Chạy lại thì ảnh và index.md do module này ghi (file `NN_<tên biểu
  đồ>.png`) được thay, file khác trong thư mục giữ nguyên; thư mục có index.md không do module này ghi thì dừng.
- --append-to: thêm mục "Biểu đồ" (ảnh dẫn bằng đường dẫn tương đối) vào cuối báo cáo Markdown; chạy lại thì thay mục cũ.
- Trên ảnh: số thập phân dùng dấu phẩy, số âm dùng dấu trừ "−"; mọi khung chênh lệch đều "phải là tốt hơn" (trục LPIPS
  đảo chiều vì LPIPS thấp hơn là tốt hơn). Nhãn của một phương pháp là tên thư mục lượt bỏ `_s<seed>` (vd. mcmc_or0001);
  các báo cáo được gộp theo nhãn này (mcmc của Deep Blending, chạy với opacity_reg 0.001, và mcmc của Mip-NeRF 360 cùng
  là `mcmc`); index.md ghi nhãn ứng với biến thể nào trong bảng của từng báo cáo.
- Màu theo phương pháp, giữ nguyên giữa các báo cáo: default xanh dương, mcmc cam, mọi biến thể khác xanh ngọc, B0
  (Inria) xám đậm; dòng của một benchmark nhạt hơn dòng tổng. Ba màu này đã qua bộ kiểm tra bảng màu (phân biệt được
  cả khi mù màu); xanh ngọc tương phản thấp với nền nên mọi giá trị đều có nhãn số và bảng.
- Đầu mỗi ảnh ghi cảnh, seed, mốc, các mục điều kiện so với mốc chưa đạt và file báo cáo nguồn: ảnh tách khỏi báo cáo
  (vd. chèn vào báo cáo đồ án) vẫn đọc đúng.
- Vẽ cần matplotlib (env gs-gsplat có sẵn, phiên bản ghim trong configs/environment/gs-gsplat.lock.txt); phần lập dữ
  liệu (`build`) không cần.

Biểu đồ chỉ trình bày số liệu, không tự kết luận phương pháp nào tốt hơn: các mục điều kiện so sánh ở báo cáo.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.evaluation.compare_modes import GOOD_STATUSES
from src.evaluation.compare_reference import METRICS

DIGITS = {"PSNR": 3, "SSIM": 4, "LPIPS": 4}  # như bảng của compare_modes
HEAD = {"PSNR": "PSNR ↑ (dB)", "SSIM": "SSIM ↑", "LPIPS": "LPIPS ↓"}
LOWER = {"LPIPS"}  # thấp hơn là tốt hơn: trục đảo chiều để mọi khung đều "phải là tốt hơn"

# Bảng màu tham chiếu của skill dataviz (nền sáng). validate_palette (--pairs all) với 3 màu loạt: đạt mọi mục, chỉ cảnh
# báo xanh ngọc dưới 3:1 so với nền, bù bằng nhãn số trên mọi điểm và bảng trong index.md.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"  # màu trạng thái: chỉ cho trong / ngoài ngưỡng, luôn kèm ✓ / ✗
B0_COLOR = INK2
B0_LABEL, PUBLISHED_LABEL, OTHERS_LABEL = "B0 (Inria)", "Inria công bố", "biến thể khác"
PART_TINT = 0.55  # dòng của một benchmark: cùng màu với dòng tổng, nhạt hơn
MINUS = "−"

DPI = 150
ROW_IN = 0.3  # chiều cao một dòng (inch)
BAR = 0.56  # bề dày thanh (phần của một dòng)
WIDTHS = {1: 6.6, 2: 8.8, 3: 10.8, 4: 12.6}
IMAGE_LINK = re.compile(r"\]\((\d\d_[a-z0-9_]+\.png)\)")  # ảnh do module này ghi, như index.md dẫn
OWN_IMAGE = re.compile(r"\d\d_(?:doi_chieu_inria|b0_so_voi_cong_bo|so_voi_[a-z0-9_]+)\.png")  # tên ảnh của module
INDEX_HEAD = "# Biểu đồ của "  # dòng đầu index.md của module này
SECTION = "**Biểu đồ**"
OLD_REPORT = ("báo cáo lập bằng bản cũ của src.evaluation.compare_modes (chưa có mục `{}`): lập lại bằng "
              "scripts/run_evaluate.sh")
BROKEN = (KeyError, TypeError, ValueError, AttributeError, IndexError, ZeroDivisionError)  # phần báo cáo hỏng / thiếu


@dataclass
class Panel:
    kind: str  # dots | delta | tolerance
    title: str
    items: list  # một mục cho mỗi dòng của nhóm; None: dòng đó không có số
    invert: bool = False  # trục x đảo chiều (LPIPS)
    hint: str = ""  # dòng chú thích dưới trục x


@dataclass
class Group:
    labels: list[str]  # nhãn các dòng, từ trên xuống
    panels: list[Panel]
    title: str = ""


@dataclass
class Chart:
    slug: str
    title: str
    subtitle: list[str]
    read: str
    groups: list[Group]
    header: list[str]
    rows: list[list[str]]
    legend: list[tuple[str, str, str]] = field(default_factory=list)  # (nhãn, màu, kiểu: dot | bar | ref | ...)
    sources: list[str] = field(default_factory=list)  # file báo cáo dùng cho biểu đồ (dòng "Nguồn" trên ảnh)
    key: str = ""  # tên file không đuôi: số thứ tự + slug


@dataclass
class Source:
    """Một báo cáo đầu vào."""
    file: str  # tên file JSON
    report: dict
    kind: str  # modes (compare_modes) | reference (compare_reference)
    name: str  # tên ngắn khi gộp nhiều báo cáo: benchmark của các lượt, hoặc tên file
    labels: dict = field(default_factory=dict)  # biến thể -> nhãn (báo cáo compare_modes)
    base: str | None = None  # nhãn của mốc (báo cáo compare_modes); None: mục baseline hỏng


# ------------------------------------------------------------------------------------------------ số và nhãn
def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value, digits: int) -> str:
    """Số theo cách viết tiếng Việt: dấu phẩy thập phân, dấu trừ "−"; không có số thì "–"."""
    if not _finite(value):
        return "–"
    text = f"{value:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace("-", MINUS).replace(".", ",")


def fmt_signed(value, digits: int) -> str:
    """Chênh lệch, luôn có dấu như bảng của compare_modes (kể cả +0,0000 / −0,0000: dấu quyết định "tốt hơn")."""
    if not _finite(value):
        return "–"
    return f"{value:+.{digits}f}".replace("-", MINUS).replace(".", ",")


def fmt_compact(value) -> str:
    """Ngưỡng, không thêm số 0 thừa: 0,5; 0,01."""
    return "–" if not _finite(value) else f"{value:g}".replace("-", MINUS).replace(".", ",")


def _seeds(seeds) -> str:
    return "seed " + ", ".join(str(seed) for seed in seeds) if seeds else "không có seed chung"


def _shorten(name: str, width: int = 32) -> str:
    return name if len(name) <= width else name[:width - 1] + "…"


def short_labels(report: dict) -> dict[str, str]:
    """Nhãn của mỗi biến thể: tên thư mục lượt bỏ `_s<seed>` (vd. mcmc_or0001) khi mọi lượt của biến thể có cùng tên đó
    và không biến thể nào khác có tên đó; không thì tên biến thể, cắt còn 32 ký tự (trùng nhau sau khi cắt thì thêm số
    thứ tự). Nhãn luôn khác nhau và không quá 32 ký tự."""
    stems: dict[str, set] = {}
    for by_variant in (report.get("scenes") or {}).values():
        for variant, agg in (by_variant or {}).items():
            for run_dir in (agg or {}).get("run_dirs") or []:
                stems.setdefault(variant, set()).add(re.sub(r"_s\d+$", "", Path(str(run_dir)).name))
    variants = [str(variant) for variant in report.get("variants") or []]
    first = {variant: (next(iter(stems[variant])) if len(stems.get(variant, ())) == 1 else None)
             for variant in variants}
    taken = Counter(name for name in first.values() if name)
    labels = {variant: (name if name and taken[name] == 1 and len(name) <= 32 and name not in (B0_LABEL, PUBLISHED_LABEL)
                        else _shorten(variant)) for variant, name in first.items()}
    clash, seen = Counter(labels.values()), Counter()
    for variant in variants:
        label = labels[variant]
        if clash[label] > 1:
            seen[label] += 1
            labels[variant] = f"{_shorten(variant, 27)} ({seen[label]})"
    return labels


def report_name(report: dict, stem: str) -> str:
    """Tên ngắn của báo cáo: benchmark của các lượt (thư mục ngay sau `benchmarks/` trong đường dẫn, bố cục
    outputs/benchmarks/<benchmark>/...) khi mọi lượt cùng một benchmark; không thì tên file báo cáo (`stem`)."""
    if isinstance(report.get("scenes"), list):  # compare_reference: thư mục mô hình B0 của từng cảnh
        paths = [row.get("model_dir") for row in report["scenes"] if isinstance(row, dict)]
    else:
        paths = [run_dir for by_variant in (report.get("scenes") or {}).values() for agg in (by_variant or {}).values()
                 for run_dir in (agg or {}).get("run_dirs") or []]
    names = set()
    for path in paths:
        parts = Path(str(path or "")).parts
        names.add(parts[parts.index("benchmarks") + 1] if "benchmarks" in parts[:-1] else None)
    return next(iter(names)) if len(names) == 1 and None not in names else stem


def color_of(variant: str, label: str) -> str:
    """Màu theo phương pháp, không theo thứ hạng: lượt default xanh dương, lượt mcmc cam, mọi biến thể khác xanh ngọc."""
    if "default" in (label, variant):
        return BLUE
    if "mcmc" in (label, variant):
        return ORANGE
    return AQUA


def _plain(label: str, variant: str) -> bool:
    """Lượt chính của một chế độ: nhãn là tên chế độ (vd. default, mcmc), không phải một biến thể có tag."""
    return label == re.match(r"[A-Za-z0-9_]*", variant).group(0)


def label_order(pairs: list[tuple[str, str]]) -> list[str]:
    """Thứ tự dòng của các nhãn [(nhãn, biến thể)]: lượt chính của mỗi chế độ theo thứ tự của báo cáo, rồi các biến thể
    khác theo nhãn (các giá trị của cùng một tham số đứng cạnh nhau)."""
    labels = list(dict.fromkeys(label for label, _ in pairs))
    plain = [label for label in labels if any(_plain(label, variant) for name, variant in pairs if name == label)]
    return plain + sorted(label for label in labels if label not in plain)


COLOR_RANK = {B0_COLOR: 0, BLUE: 1, ORANGE: 2, AQUA: 3}


def _legend(entries: list[tuple[str, str]], style: str = "bar") -> list[tuple[str, str, str]]:
    """Mỗi màu một mục, thứ tự cố định (B0, default, mcmc, biến thể khác): tên phương pháp khi màu đó chỉ có một phương
    pháp, không thì tên nhóm (tên từng dòng đã ghi ở trục)."""
    names = {BLUE: "default", ORANGE: "mcmc", AQUA: OTHERS_LABEL, B0_COLOR: B0_LABEL}
    by_color: dict[str, list[str]] = {}
    for label, color in entries:
        labels = by_color.setdefault(color, [])
        if label not in labels:
            labels.append(label)
    return [(labels[0] if len(labels) == 1 else names.get(color, labels[0]), color, style)
            for color, labels in sorted(by_color.items(), key=lambda item: COLOR_RANK.get(item[0], 9))]


def tint(color: str, amount: float) -> str:
    """Màu nhạt hơn cùng sắc: `amount` phần màu, phần còn lại là màu nền."""
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    base = [int(SURFACE[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a * amount + b * (1 - amount)):02x}" for a, b in zip(rgb, base))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "moc"


def _complete(metrics) -> bool:
    return isinstance(metrics, dict) and all(_finite(metrics.get(name)) for name in METRICS)


def _metric_title(name: str, delta: bool = False) -> str:
    return ("Δ" if delta else "") + HEAD[name] + (" · trục đảo" if name in LOWER else "")


def _broken(error: Exception) -> str:
    return f"không đọc được phần của báo cáo cần cho biểu đồ này ({type(error).__name__}: {error})"


# --------------------------------------------------------------------------------------------- báo cáo đầu vào
def report_kind(report) -> str | None:
    if not isinstance(report, dict):
        return None
    if isinstance(report.get("scenes"), dict) and isinstance(report.get("variants"), list) and "baseline" in report:
        return "modes"
    if isinstance(report.get("scenes"), list) and "reference" in report and "verdict" in report:
        return "reference"
    return None


def make_source(report: dict, file: str) -> Source:
    """Báo cáo đã đọc (report_kind khác None), kèm tên ngắn, nhãn các biến thể và nhãn của mốc."""
    path = Path(file)
    kind = report_kind(report)
    try:
        name = report_name(report, path.stem)
    except BROKEN:
        name = path.stem
    labels, base = {}, None
    if kind == "modes":
        try:
            labels = short_labels(report)
        except BROKEN:
            labels = {}
        try:
            base = labels.get(report["baseline"], str(report["baseline"]))
            deviation = re.search(r"\{.*\}$", str(report["baseline"]))  # sai khác protocol (vd. lượt chạy thử)
            if deviation and deviation.group(0) not in base:
                base += deviation.group(0)  # không gộp với mốc đúng protocol cùng tên thư mục
        except BROKEN:
            base = None
    return Source(path.name, report, kind or "", name, labels, base)


def _names(sources: list[Source]) -> dict[int, str]:
    """Tên hiện cho mỗi báo cáo khi gộp: tên benchmark, hoặc tên file khi hai báo cáo cùng tên benchmark."""
    names = [source.name for source in sources]
    if len(set(names)) < len(names):
        names = [Path(source.file).stem for source in sources]
    return {id(source): name for source, name in zip(sources, names)}


# --------------------------------------------------------------------------- đối chiếu mốc với Inria, B0
def _inria_parts(report: dict, base: str):
    """(các dòng chênh lệch, trạng thái, ghi chú) của phần đối chiếu trong một báo cáo compare_modes."""
    b0 = report.get("b0_check") or {}
    ref = report.get("reference_check") or {}
    entries = []  # (nhãn, chênh lệch, trong ngưỡng, ngưỡng, giá trị Inria, giá trị mốc, kết luận)
    for row in b0.get("scenes") or []:
        if row.get("status") in ("OK", "FAIL") and isinstance(row.get("delta"), dict):
            entries.append((f"{row['scene']}: {base} − B0", row["delta"], row.get("within_tolerance") or {},
                            b0.get("tolerance") or {}, row.get("b0") or {}, row.get("b1") or {}, row["status"]))
    if ref.get("verdict") in ("PASS", "FAIL") and isinstance(ref.get("delta"), dict) and ref["delta"]:
        entries.append((f"trung bình: {base} − công bố", ref["delta"], ref.get("within_tolerance") or {},
                        ref.get("tolerance") or {}, ref.get("reference") or {}, ref.get("mean") or {}, ref["verdict"]))
    status = ([f"B0 tự chạy, từng cảnh: {b0.get('verdict', '?')}"] if b0 else []) + \
             ([f"số Inria công bố, trung bình các cảnh: {ref.get('verdict', '?')}"] if ref else [])
    missing = [row.get("scene") for row in b0.get("scenes") or [] if row.get("status") == "MISSING"]
    notes = ([f"thiếu B0 hoặc mốc ở: {', '.join(missing)}"] if missing else []) + \
        [*(b0.get("reasons") or []), *(ref.get("reasons") or [])]
    return entries, status, notes


def inria_chart(sources: list[Source]):
    """Mốc gsplat (default) so với B0 tự chạy từng cảnh và với số công bố; nhiều báo cáo có phần đối chiếu thì mỗi báo
    cáo một khung."""
    checked = [s for s in sources if s.report.get("b0_check") or s.report.get("reference_check")]
    base = (checked[0].base if checked else None) or "default"
    title = f"Đối chiếu {base} với Inria"
    if not checked:
        return title, ("báo cáo không có phần đối chiếu (mốc không phải default, hoặc chưa có B0 tự chạy và cấu hình "
                       "benchmark)")
    multi = len(checked) > 1
    names = _names(checked)
    groups, subtitle, rows, reasons = [], [], [], []
    for source in checked:
        entries, status, notes = _inria_parts(source.report, base)
        prefix = f"{names[id(source)]}: " if multi else ""
        subtitle += [prefix + "; ".join(status), *(prefix + note for note in notes[:2])]
        reasons.append(prefix + "; ".join(status) + (": " + "; ".join(notes) if notes else ""))
        if not entries:
            continue
        panels = []
        for name in METRICS:
            tols = {fmt_compact(entry[3].get(name)) for entry in entries}
            unit = " (dB)" if name == "PSNR" else ""
            panels.append(Panel("tolerance", f"Δ{name}{unit}" + (f" · ngưỡng ±{tols.pop()}" if len(tols) == 1 else ""), [
                {"value": delta.get(name), "tol": tol.get(name), "ok": bool(within.get(name)),
                 "text": fmt_signed(delta.get(name), DIGITS[name]) + (" ✓" if within.get(name) else " ✗")}
                for _, delta, within, tol, *_ in entries], hint="gần 0 là khớp"))
        groups.append(Group([entry[0] for entry in entries], panels, title=names[id(source)] if multi else ""))
        rows += [[*([names[id(source)]] if multi else []), label,
                  *(f"{fmt(inria.get(n), DIGITS[n])} → {fmt(own.get(n), DIGITS[n])} ({fmt_signed(delta.get(n), DIGITS[n])} "
                    f"{'✓' if within.get(n) else '✗'}; ±{fmt_compact(tol.get(n))})" for n in METRICS), verdict]
                 for label, delta, within, tol, inria, own, verdict in entries]
    if not groups:
        return title, "chưa đối chiếu được (" + " | ".join(reasons) + ")"
    read = (f"Chấm: chênh lệch của mốc gsplat ({base}) so với B0 tự chạy trên từng cảnh (cùng ảnh test) và so với số "
            "Inria công bố (trung bình các cảnh); vùng xám: ngưỡng chấp nhận (±) của cấu hình benchmark. Xanh lá ✓: trong "
            "ngưỡng; đỏ ✗: ngoài ngưỡng. Trục không đảo chiều: ở đây cần gần 0, không phải lớn hơn. Kết luận PASS / FAIL "
            "như mục đối chiếu của báo cáo. Bảng: Inria → mốc (chênh lệch; ngưỡng).")
    legend = [("trong ngưỡng ✓", GOOD, "dot"), ("ngoài ngưỡng ✗", CRITICAL, "dot"), ("ngưỡng chấp nhận (±)", GRID, "band")]
    return Chart("doi_chieu_inria", title, subtitle, read, groups,
                 [*(["Báo cáo"] if multi else []), "Đối chiếu", *(HEAD[n] for n in METRICS), "Kết luận"], rows, legend,
                 sources=[s.file for s in checked])


def _dots_item(value, color, digits, err=None, band=False, text=None) -> dict:
    return {"value": value, "err": err, "band": band, "color": color,
            "text": text if text is not None else fmt(value, digits) + (f" ± {fmt_compact(err)}" if band else "")}


def reference_chart(source: Source, multi: bool = False):
    """Báo cáo chỉ B0 (compare_reference): B0 từng cảnh, trung bình và số Inria công bố với ngưỡng."""
    report = source.report
    title = "B0 so với số liệu Inria công bố" + (f" · {source.name}" if multi else "")
    scenes = [row for row in report.get("scenes") or [] if isinstance(row, dict)]
    done = [row for row in scenes if row.get("status") == "OK" and _complete(row.get("metrics"))]
    if not done:
        return title, "chưa cảnh nào có kết quả B0"
    missing = [str(row.get("scene")) for row in scenes if row not in done]
    mean, reference = report.get("mean") or {}, report.get("reference") or {}
    tolerance, delta, within = report.get("tolerance") or {}, report.get("delta") or {}, report.get("within_tolerance") or {}
    status = (f"{report.get('config') or 'benchmark'}, {report.get('method') or '?'}: kết luận {report.get('verdict', '?')} "
              "(ngưỡng " + ", ".join(f"{n} ±{fmt_compact(tolerance.get(n))}" for n in METRICS) + ")")
    entries = [(str(row["scene"]), row["metrics"], None, "scene") for row in done]
    if _complete(mean):
        entries.append(("trung bình B0", mean, None, "mean"))
    if _complete(reference):
        entries.append((PUBLISHED_LABEL, reference, tolerance, "ref"))
    panels = []
    for name in METRICS:
        items = []
        for _, metrics, tol, role in entries:
            text = None
            if role == "mean" and _finite(delta.get(name)):
                text = (f"{fmt(metrics[name], DIGITS[name])} (Δ {fmt_signed(delta[name], DIGITS[name])} "
                        f"{'✓' if within.get(name) else '✗'})")
            items.append(_dots_item(metrics.get(name), INK2 if role == "ref" else B0_COLOR, DIGITS[name],
                                    err=(tol or {}).get(name), band=role == "ref", text=text))
        panels.append(Panel("dots", _metric_title(name), items, invert=name in LOWER))
    rows = [[label, *(fmt(metrics.get(n), DIGITS[n]) + (f" ± {fmt_compact(tol.get(n))}" if tol else "")
                      for n in METRICS)] for label, metrics, tol, _ in entries]
    if delta:
        rows.append(["chênh lệch (trung bình − công bố)",
                     *(f"{fmt_signed(delta.get(n), DIGITS[n])} {'✓' if within.get(n) else '✗'}" for n in METRICS)])
    legend = [(B0_LABEL, B0_COLOR, "dot")] + ([(f"{PUBLISHED_LABEL} (± ngưỡng)", INK2, "ref")] if reference else [])
    subtitle = [status] + ([f"Thiếu kết quả: {', '.join(missing)} (chưa có trung bình)"] if missing else [])
    if report.get("reference_source"):
        subtitle.append(f"Số công bố: {_shorten(str(report['reference_source']), 150)}")
    read = ("Chấm xám: B0 (3DGS gốc của Inria, tự chạy) từng cảnh và trung bình các cảnh; dòng Inria công bố: số trong "
            "kết quả chính thức, vạch xám là ngưỡng chấp nhận (±): trung bình B0 nằm trong vạch thì đạt. Mọi khung: "
            "phải là tốt hơn (trục LPIPS đảo chiều); trục không bắt đầu từ 0: so bằng các con số.")
    return Chart("b0_so_voi_cong_bo", title, subtitle, read, [Group([e[0] for e in entries], panels)],
                 ["Dòng", *(HEAD[n] for n in METRICS)], rows, legend, sources=[source.file])


# ------------------------------------------------------------------------------------- chênh lệch tổng
def _better(name: str, value) -> bool:
    """Như compare_modes: PSNR, SSIM cao hơn, LPIPS thấp hơn; bằng nhau không tính."""
    return _finite(value) and (value < 0 if name in LOWER else value > 0)


def pair_stats(pairs: list[dict]) -> dict:
    """Trên các cặp (cảnh, seed) của mục `pairs`: trung bình, độ lệch chuẩn (mẫu) và số cặp tốt hơn của mỗi chỉ số, như
    mục wins của compare_modes."""
    values = {name: [pair[name] for pair in pairs if _finite(pair.get(name))] for name in METRICS}
    return {"n": len(pairs), "points": values,
            "mean": {name: statistics.fmean(found) if found else None for name, found in values.items()},
            "std": {name: statistics.stdev(found) if len(found) > 1 else None for name, found in values.items()},
            "better": {name: sum(_better(name, value) for value in found) for name, found in values.items()},
            "scenes": list(dict.fromkeys(str(pair.get("scene")) for pair in pairs)),
            "seeds": sorted({pair["seed"] for pair in pairs if _finite(pair.get("seed"))})}


def _conditions(sources: list[Source], base: str) -> list[str]:
    """Một dòng tóm tắt các mục điều kiện so với mốc chưa đạt (cột Điều kiện của báo cáo), đếm trên các cặp (cảnh, biến
    thể) có chênh lệch so với mốc; không có cặp nào như vậy thì không có dòng."""
    counts, total = Counter(), 0
    for source in sources:
        report = source.report
        for scene, by_variant in (report.get("deltas") or {}).items():
            for variant in by_variant:
                total += 1
                for check in (report.get("fairness") or {}).get(scene, {}).get(variant, []):
                    if check.get("status") not in GOOD_STATUSES:
                        counts[f"{check.get('check')}={check.get('status')}"] += 1
    if not total:
        return []
    if not counts:
        return [f"Mọi mục điều kiện so với mốc {base} đạt (cột Điều kiện của báo cáo)."]
    return [f"Điều kiện so với mốc {base} chưa đạt: " + ", ".join(f"{name} ở {count}/{total}" for name, count in counts.items())
            + " cặp (cảnh, biến thể) (cột Điều kiện của báo cáo): chưa được nói \"cải thiện\"."]


def total_chart(sources: list[Source], base: str):
    """Chênh lệch tổng so với mốc `base`: mỗi phương pháp, trên mọi cặp (cảnh, seed) chung với mốc của mọi báo cáo
    trong `sources` (cùng mốc); nhiều báo cáo thì thêm một dòng cho mỗi báo cáo (benchmark)."""
    title = f"Chênh lệch tổng so với {base}"
    old = [s.file for s in sources if "pairs" not in s.report]
    if old:
        return title, OLD_REPORT.format("pairs") + (f" ({', '.join(old)})" if len(sources) > 1 else "")
    owner = {}
    for index, source in enumerate(sources):  # một lượt chỉ được tính một lần (vd. cùng một báo cáo đưa hai lần)
        for by_variant in source.report["scenes"].values():
            for agg in by_variant.values():
                for run_dir in agg.get("run_dirs") or []:
                    if owner.setdefault(str(run_dir), index) != index:
                        return title, (f"{sources[owner[str(run_dir)]].file} và {source.file} có chung lượt {run_dir}: "
                                       "không gộp (mỗi cặp (cảnh, seed) chỉ được tính một lần)")
    multi = len(sources) > 1
    names = _names(sources)
    found: dict[str, list] = {}  # nhãn -> [(báo cáo, biến thể, các cặp)]
    for source in sources:
        for variant in source.report["variants"]:
            if variant != source.report["baseline"]:
                pairs = [pair for pair in (source.report["pairs"].get(variant) or []) if isinstance(pair, dict)]
                found.setdefault(source.labels.get(variant, variant), []).append((source, variant, pairs))
    rows, left_out = [], []  # dòng: (nhãn dòng, nhãn, màu, thống kê, [(báo cáo, thống kê)], là dòng một benchmark)
    for label in label_order([(label, parts[0][1]) for label, parts in found.items()]):
        parts = [(source, pair_stats(pairs)) for source, _, pairs in found[label] if pairs]
        if not parts:
            left_out.append(label)
            continue
        color = color_of(found[label][0][1], label)
        if multi and len(parts) > 1:
            pooled = pair_stats([pair for _, _, pairs in found[label] for pair in pairs])
            rows.append((f"{label}: tổng", label, color, pooled, parts, False))
            rows += [(f"{label}: {names[id(source)]}", label, tint(color, PART_TINT), stats, [(source, stats)], True)
                     for source, stats in parts]
        else:
            rows.append(((f"{label}: {names[id(parts[0][0])]}" if multi else label), label, color, parts[0][1], parts,
                         False))
    if not rows:
        if not found:
            return title, f"báo cáo chỉ có mốc {base}: không có phương pháp nào khác để so"
        return title, "không phương pháp nào so được với mốc (mốc thiếu ở mọi cảnh, hoặc không có seed chung)"

    def scope(parts) -> str:
        if not multi:
            return ", ".join(parts[0][1]["scenes"])
        return "; ".join(f"{names[id(source)]}: {', '.join(stats['scenes'])}" for source, stats in parts)

    panels = []
    for name in METRICS:
        items = []
        for _, _, color, stats, _, _ in rows:
            count = stats["n"]
            items.append({"value": stats["mean"][name], "points": stats["points"][name] if count > 1 else [],
                          "color": color, "text": fmt_signed(stats["mean"][name], DIGITS[name])
                          + (f" · {stats['better'][name]}/{count}" if count > 1 else "")})
        panels.append(Panel("delta", _metric_title(name, delta=True), items, invert=name in LOWER,
                            hint="← kém hơn · tốt hơn →"))
    covered = {}  # báo cáo -> các cảnh có ít nhất một cặp, theo thứ tự của báo cáo
    for _, _, _, _, parts, _ in rows:
        for source, stats in parts:
            covered.setdefault(id(source), (source, set()))[1].update(stats["scenes"])
    scenes = [(source, [scene for scene in source.report["scenes"] if scene in seen]) for source, seen in covered.values()]
    where = ("; ".join(f"{names[id(source)]}: {', '.join(found_scenes)}" for source, found_scenes in scenes) if multi
             else ", ".join(scenes[0][1]))
    seeds = sorted({seed for row in rows for seed in row[3]["seeds"]})
    many = max(row[3]["n"] for row in rows) > 1
    subtitle = [f"Mốc {base}; {sum(len(found_scenes) for _, found_scenes in scenes)} cảnh ({where}); {_seeds(seeds)}",
                ("Thanh: chênh lệch trung bình trên mọi cặp (cảnh, seed) chung với mốc; chấm đen: từng cặp; số ghi: chênh "
                 "lệch trung bình · số cặp tốt hơn / số cặp." if many else
                 "Mỗi phương pháp một cặp (cảnh, seed) chung với mốc: thanh và số ghi là chênh lệch của cặp đó."),
                *_conditions(sources, base)]
    if left_out:
        subtitle.append(f"Không có cặp (cảnh, seed) chung với mốc, không vẽ: {', '.join(left_out)}.")
    if any("{" in str(variant) for source in sources for variant in source.report["variants"]):
        subtitle.append("Có biến thể sai khác protocol (phần {…} của tên biến thể, vd. lượt chạy thử): số liệu không dùng "
                        "để kết luận.")
    read = (f"Thanh: chênh lệch so với {base}, trung bình trên mọi cặp (cùng cảnh, cùng seed) mà phương pháp và {base} "
            "đều có (các seed chung của từng cảnh, như bảng chênh lệch của báo cáo); chấm đen: từng cặp. Số ghi: chênh "
            "lệch trung bình, rồi số cặp tốt hơn / số cặp (PSNR, SSIM cao hơn; LPIPS thấp hơn; bằng nhau không tính). "
            f"Mọi khung: phải là tốt hơn {base}, trái là kém hơn (trục LPIPS đảo chiều vì LPIPS thấp hơn là tốt hơn); số "
            "ghi là chênh lệch thật (ΔLPIPS âm là tốt hơn). Các chấm nằm cả hai phía của 0, hoặc trải rộng so với độ dài "
            "thanh: chênh lệch chưa nhất quán giữa các cảnh và seed, trung bình chưa đủ để nói \"tốt hơn\". Số từng cảnh: "
            "bảng chênh lệch của báo cáo.")
    if multi:
        read += (" Dòng \"tổng\" gộp mọi cặp của các benchmark (benchmark nhiều cảnh hơn có nhiều cặp hơn); dòng của một "
                 "benchmark (thanh nhạt): chỉ các cặp của benchmark đó, đúng số ở bảng số cặp tốt hơn của báo cáo "
                 "benchmark đó.")
    else:
        read += " Số trên ảnh đúng số ở bảng số cặp tốt hơn của báo cáo (Δ: trung bình ± độ lệch chuẩn theo cặp)."

    def cell(stats, name) -> str:
        std = stats["std"][name]
        return fmt_signed(stats["mean"][name], DIGITS[name]) + (f" ± {fmt(std, DIGITS[name])}" if _finite(std) else "")
    table = [[row_label, scope(parts), ", ".join(str(seed) for seed in stats["seeds"]), str(stats["n"]),
              *(cell(stats, name) for name in METRICS), " · ".join(f"{stats['better'][n]}/{stats['n']}" for n in METRICS)]
             for row_label, _, _, stats, parts, _ in rows]
    legend = _legend([(label, color) for _, label, color, _, _, part in rows if not part], style="bar")
    if many:
        legend.append(("từng cặp (cảnh, seed)", INK, "small"))
    return Chart("so_voi_" + _slug(base), title, subtitle, read, [Group([row[0] for row in rows], panels)],
                 ["Phương pháp", "Cảnh", "Seed", "Cặp", *(f"Δ{HEAD[n]}" for n in METRICS),
                  "Cặp tốt hơn: PSNR · SSIM · LPIPS"], table, legend, sources=[s.file for s in sources])


# ------------------------------------------------------------------------------------------------- lập dữ liệu
def _guard(title: str, maker, *args):
    """Một phần báo cáo hỏng chỉ bỏ biểu đồ dùng nó."""
    try:
        return maker(*args)
    except BROKEN as error:
        return title, _broken(error)


def build(sources: list[Source]) -> tuple[list[Chart], list[tuple[str, str]]]:
    """(biểu đồ sẽ vẽ, theo thứ tự đọc; [(biểu đồ không vẽ, lý do)]). Thứ tự: B0 so với số công bố (báo cáo chỉ B0), mốc
    so với Inria (điều kiện để dùng nó làm mốc), rồi chênh lệch tổng so với từng mốc (default trước). Không cần
    matplotlib."""
    results = []
    references = [source for source in sources if source.kind == "reference"]
    for source in references:
        results.append(_guard("B0 so với số liệu Inria công bố", reference_chart, source, len(references) > 1))
    modes = [source for source in sources if source.kind == "modes"]
    if modes:
        results.append(_guard("Đối chiếu mốc với Inria", inria_chart, modes))
        results += [("Chênh lệch tổng so với mốc", f"mục baseline của {source.file} hỏng")
                    for source in modes if source.base is None]
        bases = list(dict.fromkeys(source.base for source in modes if source.base is not None))
        for base in sorted(bases, key=lambda name: name != "default"):
            results.append(_guard(f"Chênh lệch tổng so với {base}", total_chart,
                                  [source for source in modes if source.base == base], base))
    charts = [item for item in results if isinstance(item, Chart)]
    for number, chart in enumerate(charts, 1):
        chart.key = f"{number:02d}_{chart.slug}"
    return charts, [item for item in results if not isinstance(item, Chart)]


# ------------------------------------------------------------------------------------------------------- vẽ
RC = {"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": AXIS, "text.color": INK,
      "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.titlesize": 9.5}


def _wrap(line: str, width_in: float) -> list[str]:
    return textwrap.wrap(line, width=max(int(width_in * 13), 40)) or [""]


def _source_line(chart: Chart) -> str:
    return "Nguồn: " + ", ".join(chart.sources) if chart.sources else ""


def _size(chart: Chart) -> tuple[float, float, list[float]]:
    columns = max(len(group.panels) for group in chart.groups)
    longest = max((len(label) for group in chart.groups for label in group.labels), default=0)
    width = WIDTHS.get(columns, WIDTHS[4]) + max(0, longest - 20) * 0.075  # nhãn dòng dài: khung không bị hẹp lại
    lines = sum(len(_wrap(line, width)) for line in [*chart.subtitle, _source_line(chart)] if line)
    header = 0.5 + 0.2 * lines + (0.34 if chart.legend else 0.0)
    groups = [(0.32 if group.title else 0.0) + 0.3 + 0.52 + ROW_IN * max(len(group.labels), 1)
              for group in chart.groups]
    return width, header, groups


def _comma_formatter():
    from matplotlib.ticker import Formatter

    class Comma(Formatter):
        """Nhãn trục kiểu Việt; số chữ số thập phân theo khoảng cách giữa các vạch (tính trong format_ticks, không dùng
        Formatter.locs: thuộc tính này bị bỏ từ matplotlib 3.11)."""
        digits = 0

        def format_ticks(self, values):
            ticks = sorted(value for value in values if math.isfinite(value))
            steps = [b - a for a, b in zip(ticks, ticks[1:]) if b - a > 0]
            self.digits = max(0, -math.floor(math.log10(min(steps)) + 1e-9)) if steps else 0
            return [fmt(value, self.digits) for value in values]

        def __call__(self, x, pos=None):
            return fmt(x, self.digits)

    return Comma()


def _style(ax, labels: list[str], first: bool) -> None:
    from matplotlib.ticker import MaxNLocator
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(max(len(labels), 1) - 0.5, -0.5)
    ax.set_yticks(range(len(labels)))
    if first:
        ax.set_yticklabels(labels, fontsize=8.8, color=INK)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", colors=AXIS, labelcolor=MUTED, labelsize=7.8, length=3, width=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(_comma_formatter())


def _label(ax, text: str, x: float, y: float, right: bool = True):
    return ax.annotate(text, (x, y), xytext=(6 if right else -6, 0), textcoords="offset points",
                       ha="left" if right else "right", va="center", fontsize=8, color=INK2, zorder=5,
                       annotation_clip=False)


def _set_xlim(ax, lo: float, hi: float, invert: bool) -> None:
    ax.set_xlim((hi, lo) if invert else (lo, hi))


def _row_lines(ax, panel: Panel) -> None:
    if len(panel.items) >= 6:  # nhiều dòng: đường mảnh qua từng dòng để mắt đi từ tên tới số
        ax.grid(axis="y", color=GRID, linewidth=0.5)


def _dots(ax, panel: Panel) -> list:
    texts, xs = [], []
    direction = -1 if panel.invert else 1
    for i, item in enumerate(panel.items):
        if not item or not _finite(item.get("value")):
            continue
        value, err, color = item["value"], item.get("err"), item.get("color", INK2)
        reach = err if _finite(err) and err > 0 else 0.0
        if item.get("band"):
            ax.plot([value - reach, value + reach], [i, i], color=GRID, lw=9, solid_capstyle="butt", zorder=1)
            ax.plot([value], [i], ls="", marker="|", ms=13, mew=2.2, color=INK2, zorder=3)
        else:
            if reach:
                ax.plot([value - reach, value + reach], [i, i], color=color, lw=1.6, solid_capstyle="butt", zorder=2)
            ax.plot([value], [i], ls="", marker="o", ms=7.5, mfc=color, mec=SURFACE, mew=1.2, zorder=3)
        xs += [value - reach, value + reach]
        texts.append(_label(ax, item["text"], value + direction * reach, i))
    if xs:
        lo, hi = min(xs), max(xs)
        pad = (hi - lo) * 0.06 or max(abs(hi) * 0.002, 1e-6)
        _set_xlim(ax, lo - pad, hi + pad, panel.invert)
    _row_lines(ax, panel)
    return texts


def _delta(ax, panel: Panel) -> list:
    ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
    direction = -1 if panel.invert else 1
    texts, extent = [], []
    for i, item in enumerate(panel.items):
        if not item:
            continue
        value = item.get("value")
        points = [point for point in item.get("points") or [] if _finite(point)]
        reach = []
        if _finite(value):
            ax.barh(i, value, height=BAR, color=item["color"], linewidth=0, zorder=2)
            reach.append(value)
        if len(points) > 1:
            ax.plot(points, [i] * len(points), ls="", marker="o", ms=4.6, mfc=INK, mec=SURFACE, mew=0.9, zorder=4)
            reach += points
        if not reach:
            continue
        extent += reach
        if _finite(value) and direction * value < 0:
            texts.append(_label(ax, item["text"], min(reach, key=lambda x: direction * x), i, right=False))
        else:
            texts.append(_label(ax, item["text"], max(reach, key=lambda x: direction * x), i))
    limit = max((abs(x) for x in extent), default=0.0) or 1e-3
    _set_xlim(ax, -limit * 1.1, limit * 1.1, panel.invert)
    _row_lines(ax, panel)
    return texts


def _tolerance(ax, panel: Panel) -> list:
    from matplotlib.patches import Rectangle
    ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
    texts, extent = [], []
    for i, item in enumerate(panel.items):
        if not item:
            continue
        tol, value = item.get("tol"), item.get("value")
        if _finite(tol):
            ax.add_patch(Rectangle((-tol, i - 0.36), 2 * tol, 0.72, facecolor=GRID, edgecolor="none", zorder=0.2))
            extent.append(tol)
        if _finite(value):
            ax.plot([value], [i], ls="", marker="o", ms=7.5, mfc=GOOD if item.get("ok") else CRITICAL, mec=SURFACE,
                    mew=1.2, zorder=3)
            extent.append(abs(value))
            texts.append(_label(ax, item["text"], max(value, tol) if _finite(tol) else value, i))
    limit = max(extent, default=1.0) * 1.15
    ax.set_xlim(-limit, limit)
    return texts


DRAWERS = {"dots": _dots, "delta": _delta, "tolerance": _tolerance}


def _handle(label: str, color: str, style: str):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    if style == "bar":
        return Patch(facecolor=color, edgecolor="none", label=label)
    if style == "band":
        return Line2D([], [], color=color, lw=8, solid_capstyle="butt", label=label)
    if style == "ref":
        return Line2D([], [], color=GRID, lw=8, solid_capstyle="butt", marker="|", ms=11, mew=2, mec=color, label=label)
    return Line2D([], [], ls="", marker="o", ms=5 if style == "small" else 7.5, mfc=color, mec=SURFACE, mew=1.0,
                  label=label)


def _header(sub, chart: Chart, width: float, height: float) -> None:
    sub.set_facecolor(SURFACE)
    y = 1 - 0.12 / height
    sub.text(0.008, y, chart.title, ha="left", va="top", fontsize=13.5, fontweight="semibold", color=INK)
    y -= 0.38 / height
    source = _source_line(chart)
    for line in [part for line in chart.subtitle for part in _wrap(line, width)]:
        sub.text(0.008, y, line, ha="left", va="top", fontsize=8.8, color=INK2)
        y -= 0.2 / height
    for line in _wrap(source, width) if source else []:
        sub.text(0.008, y, line, ha="left", va="top", fontsize=8.8, color=MUTED)
        y -= 0.2 / height
    if chart.legend:
        handles = [_handle(*entry) for entry in chart.legend]
        sub.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncols=len(handles), frameon=False,
                   fontsize=8.5, labelcolor=INK2, handlelength=1.3, handletextpad=0.4, columnspacing=1.5,
                   borderaxespad=0.3)


def _widen(ax, texts: list, renderer) -> bool:
    """Nới trục x khi nhãn số tràn ra ngoài khung: chữ không bị cắt, không đè lên khung bên cạnh."""
    box = ax.get_window_extent(renderer)
    boxes = [text.get_window_extent(renderer) for text in texts if text.get_text()]
    if box.width <= 0 or not boxes:
        return False
    over_right = max(b.x1 for b in boxes) - (box.x1 - 2)
    over_left = (box.x0 + 2) - min(b.x0 for b in boxes)
    left, right = ax.get_xlim()
    span = right - left
    changed = False
    if over_right > 0.5:
        right += span * over_right / box.width * 1.25
        changed = True
    if over_left > 0.5:
        left -= span * over_left / box.width * 1.25
        changed = True
    if changed:
        ax.set_xlim(left, right)
    return changed


def figure(chart: Chart):
    """Hình của một biểu đồ (matplotlib Figure, không qua pyplot), đã căn để nhãn số nằm trong khung."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    width, header, groups = _size(chart)
    with matplotlib.rc_context(RC):
        fig = Figure(figsize=(width, header + sum(groups)), dpi=DPI, facecolor=SURFACE, layout="constrained")
        FigureCanvasAgg(fig)
        fig.get_layout_engine().set(w_pad=0.06, h_pad=0.04, hspace=0.0, wspace=0.04)
        subs = fig.subfigures(len(chart.groups) + 1, 1, height_ratios=[header, *groups], hspace=0.0)
        _header(subs[0], chart, width, header)
        labeled = []
        for sub, group in zip(subs[1:], chart.groups):
            sub.set_facecolor(SURFACE)
            if group.title:
                sub.suptitle(group.title, x=0.008, ha="left", fontsize=10, fontweight="semibold", color=INK)
            axes = sub.subplots(1, len(group.panels), sharey=True, squeeze=False)[0]
            for column, (ax, panel) in enumerate(zip(axes, group.panels)):
                _style(ax, group.labels, column == 0)
                ax.set_title(panel.title, loc="left", fontsize=9.5, color=INK, pad=5)
                texts = DRAWERS[panel.kind](ax, panel)
                if panel.hint:
                    ax.set_xlabel(panel.hint, fontsize=7.8, color=MUTED, labelpad=2)
                labeled.append((ax, texts))
        for _ in range(4):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            if not [ax for ax, texts in labeled if _widen(ax, texts, renderer)]:
                break
        else:
            fig.canvas.draw()
    return fig


def save(chart: Chart, path: Path) -> None:
    import matplotlib
    with matplotlib.rc_context(RC):
        fig = figure(chart)
        fig.savefig(path, dpi=DPI, facecolor=SURFACE, metadata={"Software": None})


# ------------------------------------------------------------------------------------------------- file ghi ra
def _cell(text) -> str:
    return str(text).replace("|", "\\|")


def table_lines(header: list[str], rows: list[list[str]]) -> list[str]:
    return (["| " + " | ".join(_cell(head) for head in header) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows])


def index_markdown(charts: list[Chart], skipped: list, sources: list[Source]) -> str:
    files = ", ".join(f"`{source.file}`" for source in sources)
    tools = ", ".join(dict.fromkeys("src.evaluation.compare_modes" if source.kind == "modes"
                                    else "src.evaluation.compare_reference" for source in sources))
    lines = [INDEX_HEAD + files, "",
             f"Lập từ {files} ({tools}), không đọc lại lượt chạy: số trên biểu đồ và trong các bảng dưới đây là số của "
             "báo cáo, hoặc tính thẳng từ các số đó (trung bình, độ lệch chuẩn và số cặp tốt hơn trên các cặp (cảnh, "
             "seed) của mục `pairs`, gộp các báo cáo cùng mốc). Số thập phân dùng dấu phẩy, số âm dùng dấu trừ (−). ↑: "
             "cao hơn là tốt hơn; ↓: thấp hơn là tốt hơn; trên biểu đồ chênh lệch, mọi khung đều \"phải là tốt hơn\" "
             "(trục LPIPS đảo chiều). Số từng cảnh, chi phí và dung lượng: các bảng của báo cáo. Biểu đồ chỉ trình bày "
             "số liệu, không tự kết luận phương pháp nào tốt hơn: điều kiện so sánh ở báo cáo "
             "(`docs/design/comparisons.md`).", ""]
    modes = [source for source in sources if source.kind == "modes"]
    named = len({source.name for source in modes}) > 1  # nhiều benchmark: ghi nhãn ứng với biến thể nào ở báo cáo nào
    renamed = sorted(dict.fromkeys((label, f"`{variant}`", *((source.name,) if named else ())) for source in modes
                                   for variant, label in source.labels.items() if label != variant),
                     key=lambda row: row[0])
    if renamed:
        lines += ["Nhãn trong biểu đồ (tên thư mục lượt, bỏ `_s<seed>`; các báo cáo được gộp theo nhãn) và biến thể tương "
                  "ứng trong bảng của báo cáo:", "",
                  *table_lines(["Nhãn", "Biến thể", *(["Báo cáo"] if named else [])], [list(row) for row in renamed]), ""]
    for chart in charts:  # số mục = số của file ảnh (một biểu đồ vẽ lỗi thì để trống số đó)
        lines += [f"## {int(chart.key[:2])}. {chart.title}", "", f"![{chart.title}]({chart.key}.png)", ""]
        lines += [f"- {line}" for line in chart.subtitle] + [""]
        lines += [f"**Cách đọc.** {chart.read}", "", *table_lines(chart.header, chart.rows), ""]
    if skipped:
        lines += ["## Không vẽ", ""] + [f"- {title}: {reason}" for title, reason in skipped] + [""]
    return "\n".join(lines).rstrip("\n") + "\n"


def append_section(markdown: Path, out: Path, charts: list[Chart]) -> None:
    """Mục "Biểu đồ" ở cuối báo cáo Markdown, ảnh dẫn tương đối từ thư mục của báo cáo; chạy lại thì thay mục cũ."""
    rel = Path(os.path.relpath(out, markdown.parent)).as_posix()
    marker = f"{SECTION} (`{rel}/`"
    text = markdown.read_text(encoding="utf-8") if markdown.is_file() else ""
    at = text.rfind(f"\n{SECTION} (`")  # mục cũ (luôn ở cuối), kể cả khi lần trước ghi ảnh vào thư mục khác
    if at >= 0:
        text = text[:at]
    lines = ["", f"{marker}; cách đọc và bảng số của từng biểu đồ: `{rel}/index.md`):", ""]
    for chart in charts:
        lines += [f"![{chart.title}]({rel}/{chart.key}.png)", ""]
    if not charts:
        lines += ["Không vẽ được biểu đồ nào (lý do trong index.md).", ""]
    markdown.write_text(text.rstrip("\n") + "\n" + "\n".join(lines), encoding="utf-8")


def owned_files(out: Path) -> set[str] | None:
    """Ảnh của module này trong `out`: các ảnh index.md của nó dẫn tới, và mọi file mang tên biểu đồ của nó (lần chạy
    trước bị ngắt trước khi ghi index.md); None khi `out` có index.md không do module này ghi."""
    index = out / "index.md"
    named = {path.name for path in out.iterdir() if OWN_IMAGE.fullmatch(path.name)}
    if not index.is_file():
        return named
    text = index.read_text(encoding="utf-8", errors="replace")
    return set(IMAGE_LINK.findall(text)) | named if text.startswith(INDEX_HEAD) else None


def clean(out: Path, owned: set[str]) -> None:
    """Xóa ảnh và index.md do lần chạy trước của module này ghi; không đụng tới file khác trong thư mục."""
    for name in [*owned, "index.md"]:
        (out / name).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", required=True, type=Path, nargs="+",
                        help="báo cáo JSON của src.evaluation.compare_modes hoặc compare_reference; nhiều báo cáo (vd. các "
                             "benchmark của một thực nghiệm): các báo cáo cùng mốc được gộp")
    parser.add_argument("--out", type=Path,
                        help="thư mục ghi ảnh và index.md (mặc định với một báo cáo: <báo cáo>_charts cạnh file JSON; "
                             "bắt buộc khi có nhiều báo cáo)")
    parser.add_argument("--append-to", type=Path, metavar="MD",
                        help="báo cáo Markdown: thêm mục Biểu đồ (ảnh, đường dẫn tương đối) vào cuối")
    args = parser.parse_args(argv)
    if len(args.report) > 1 and args.out is None:
        parser.error("--out is required with several reports")
    sources = []
    for path in args.report:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            print(f"ERROR: cannot read {path}: {error}", file=sys.stderr)
            return 2
        if report_kind(report) is None:
            print(f"ERROR: {path} is not a report of src.evaluation.compare_modes or compare_reference", file=sys.stderr)
            return 2
        sources.append(make_source(report, path.name))
    charts, skipped = build(sources)
    if importlib.util.find_spec("matplotlib") is None:
        print("ERROR: matplotlib is not installed (env gs-gsplat has it): no charts", file=sys.stderr)
        return 3
    out = args.out or args.report[0].with_name(args.report[0].stem + "_charts")
    owned = owned_files(out) if out.is_dir() else set()
    if owned is None:
        print(f"ERROR: {out}/index.md was not written by src.evaluation.charts: choose another --out", file=sys.stderr)
        return 2
    os.environ["MPLBACKEND"] = "Agg"  # vẽ không cần màn hình; MPLBACKEND của Jupyter làm matplotlib báo lỗi khi import
    out.mkdir(parents=True, exist_ok=True)
    clean(out, owned)
    drawn = []
    for chart in charts:
        try:
            save(chart, out / f"{chart.key}.png")
            drawn.append(chart)
        except Exception as error:  # vẽ lỗi (matplotlib có nhiều loại lỗi): các biểu đồ khác và index.md vẫn được ghi
            print(f"WARN: chart {chart.key} could not be drawn: {type(error).__name__}: {error}", file=sys.stderr)
            skipped.append((chart.title, f"vẽ lỗi ({type(error).__name__}: {error})"))
    (out / "index.md").write_text(index_markdown(drawn, skipped, sources), encoding="utf-8")
    if args.append_to:
        append_section(args.append_to, out, drawn)
    print(f"Biểu đồ: {len(drawn)} ảnh trong {out} (cách đọc, bảng số: {out / 'index.md'})"
          + (f"; không vẽ: {', '.join(title for title, _ in skipped)}" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
