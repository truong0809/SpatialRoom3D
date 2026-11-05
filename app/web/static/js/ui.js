// Thành phần giao diện dùng chung: biểu tượng, nhãn trạng thái, thanh trên của trang, hình minh họa.
import * as fmt from "./fmt.js";

const PATHS = {
  menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  upload: '<path d="M12 15V4"/><path d="m7 9 5-5 5 5"/><path d="M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15"/>',
  video: '<rect x="3" y="6" width="13" height="12" rx="2.5"/><path d="m16 10.5 5-3v9l-5-3"/>',
  image: '<rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><circle cx="9" cy="10" r="1.8"/><path d="m4 17 5-4.5 4 3.5 2.5-2 4.5 4"/>',
  folder: '<path d="M3.5 7.5A2 2 0 0 1 5.5 5.5h4l2 2.5h7a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  check: '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
  chevDown: '<path d="m6 9 6 6 6-6"/>',
  chevLeft: '<path d="m15 6-6 6 6 6"/>',
  chevRight: '<path d="m9 6 6 6-6 6"/>',
  play: '<path d="M8 5.5v13l11-6.5z"/>',
  open: '<path d="M14 4h6v6"/><path d="M20 4 11 13"/><path d="M19 14v4.5A1.5 1.5 0 0 1 17.5 20h-12A1.5 1.5 0 0 1 4 18.5v-12A1.5 1.5 0 0 1 5.5 5H10"/>',
  cube: '<path d="M12 3 4 7.5v9L12 21l8-4.5v-9z"/><path d="M4 7.5 12 12l8-4.5M12 12v9"/>',
  spark: '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M18 6l-2.5 2.5M8.5 15.5 6 18"/>',
  camera: '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
  record: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3" fill="currentColor"/>',
  full: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
  panel: '<rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><path d="M14.5 4.5v15"/>',
  reset: '<path d="M4 12a8 8 0 1 0 2.4-5.7"/><path d="M4 4v4.5h4.5"/>',
  zoomIn: '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2M11 8.5v5M8.5 11h5"/>',
  zoomOut: '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2M8.5 11h5"/>',
  spin: '<path d="M12 4a8 8 0 1 1-7.4 5"/><path d="M4 4.5v4.5h4.5"/><circle cx="12" cy="12" r="1.6" fill="currentColor"/>',
  download: '<path d="M12 4v11"/><path d="m7 10 5 5 5-5"/><path d="M4 19h16"/>',
  gauge: '<path d="M4 18a8 8 0 1 1 16 0"/><path d="m12 18 4-6"/>',
  orbit: '<path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 4v5h-5"/>',
  walk: '<circle cx="13" cy="4" r="2"/><path d="M9 21l2-6 3 3v3"/><path d="M7 12l3-4 4 1 3 3"/><path d="M10 8l1 7"/>',
  tour: '<circle cx="5" cy="18" r="2"/><circle cx="19" cy="6" r="2"/><path d="M7 18h6a4 4 0 0 0 0-8h-2a4 4 0 0 1 0-8h6"/>',
  clock: '<circle cx="12" cy="12" r="8"/><path d="M12 8v4l2.5 2.5"/>',
  alert: '<path d="M12 4 2.8 19.5h18.4z"/><path d="M12 10v4M12 17v.5"/>',
};

export function icon(name, size = 18, stroke = 1.8) {
  return `<svg aria-hidden="true" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${stroke}" stroke-linecap="round" stroke-linejoin="round">${PATHS[name] || ""}</svg>`;
}

// Trạng thái dự án (rooms.py: queued / running / done / failed / stopped / lost)
export const STATUS = {
  queued: { label: "Đang chờ GPU", tone: "warn" },
  running: { label: "Đang dựng", tone: "run" },
  done: { label: "Hoàn thành", tone: "ok" },
  failed: { label: "Lỗi", tone: "err" },
  stopped: { label: "Đã dừng", tone: "err" },
  lost: { label: "Bị gián đoạn", tone: "err" },
};

export function tone(status) {
  return STATUS[status]?.tone || "";
}

export function badge(status) {
  const s = STATUS[status] || { label: status || "Không rõ", tone: "" };
  return `<span class="badge tone-${s.tone}"><span class="dot" aria-hidden="true"></span>${fmt.escapeHtml(s.label)}</span>`;
}

export function sourceKind(kind) {
  return kind === "video" ? "Video" : kind === "photos" ? "Ảnh chụp" : "Không rõ nguồn";
}

// Thanh trên của trang: nút mở thanh bên (điện thoại), đường dẫn, thao tác bên phải
export function topbar(crumbs, actions = "") {
  const items = crumbs.map(([label, href], i) => {
    const last = i === crumbs.length - 1;
    const text = fmt.escapeHtml(label);
    const node = last || !href ? `<span class="${last ? "current" : ""}"${last ? ' aria-current="page"' : ""}>${text}</span>` : `<a href="${href}">${text}</a>`;
    return i ? `<span class="sep" aria-hidden="true">/</span>${node}` : node;
  }).join("");
  return `<header class="topbar">
    <button type="button" class="icon-btn menu-btn" data-action="menu" aria-label="Mở thanh bên">${icon("menu")}</button>
    <nav class="crumbs" aria-label="Vị trí">${items}</nav>
    <div class="actions">${actions}</div></header>`;
}

// Hình minh họa: các Gaussian (elip mờ) tạo thành góc một căn phòng; dùng cho trạng thái rỗng và ô kéo thả
let artCount = 0;
export function splatArt(className = "art") {
  const id = `splat-blur-${++artCount}`;
  const blobs = [
    [62, 70, 46, 20, -24, "#3D6BD6", 0.55], [112, 58, 38, 15, 18, "#6B9BFF", 0.5], [150, 84, 44, 17, -8, "#3D6BD6", 0.42],
    [84, 108, 52, 18, 6, "#6B9BFF", 0.38], [134, 118, 40, 14, -20, "#B7CCFF", 0.32], [46, 128, 30, 12, 30, "#6B9BFF", 0.35],
    [178, 120, 26, 10, 12, "#3D6BD6", 0.4], [100, 142, 58, 13, -4, "#3D6BD6", 0.3], [70, 38, 24, 9, 40, "#B7CCFF", 0.4],
    [160, 46, 22, 8, -36, "#B7CCFF", 0.35],
  ].map(([cx, cy, rx, ry, rot, color, opacity]) =>
    `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" transform="rotate(${rot} ${cx} ${cy})" fill="${color}" opacity="${opacity}"/>`).join("");
  return `<svg class="${className}" aria-hidden="true" viewBox="0 0 220 170" fill="none"><defs><filter id="${id}" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="2.2"/></filter></defs>
    <path d="M30 120 110 80l80 40-80 40z" stroke="#2A3140" stroke-width="1.2"/><path d="M30 120V44l80-34v70M110 10l80 34v76" stroke="#2A3140" stroke-width="1.2"/>
    <g filter="url(#${id})">${blobs}</g></svg>`;
}

// Phần thời gian ước lượng của mỗi bước dựng (để tính % tổng): trọng số đặt tay, chưa đo trên pod, nên trang luôn ghi
// "ước lượng". Bước train có % thật từ train_main.py; bước đang chạy mà không có % thì tính là đã xong 40 %.
export const STAGE_WEIGHTS = { images: 0.1, colmap: 0.25, train: 0.6, export: 0.05 };
const STAGE_ORDER = ["images", "colmap", "train", "export"];

// % tổng từ danh sách bước của progress.json (null khi chưa có tiến độ)
export function overallPercent(stages) {
  if (!stages?.length) return null;
  let value = 0;
  for (const stage of stages) {
    const w = STAGE_WEIGHTS[stage.id] ?? 0;
    if (stage.status === "done" || stage.status === "skipped") value += w;
    else if (stage.status === "running") value += w * (stage.percent != null ? stage.percent / 100 : 0.4);
  }
  return Math.min(100, value * 100);
}

// % tổng từ bản tóm tắt của /api/rooms (chỉ biết bước hiện tại và % train)
export function summaryPercent(build) {
  const index = STAGE_ORDER.indexOf(build?.stage);
  if (index < 0) return null;
  let value = STAGE_ORDER.slice(0, index).reduce((sum, id) => sum + STAGE_WEIGHTS[id], 0);
  const current = STAGE_WEIGHTS[build.stage];
  value += current * (Number.isFinite(build.percent) ? build.percent / 100 : 0.4);
  return Math.min(100, value * 100);
}
