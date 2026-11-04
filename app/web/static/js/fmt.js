// Định dạng số và thời gian kiểu Việt Nam (dấu phẩy thập phân, dấu chấm hàng nghìn).

const nf = (digits) => new Intl.NumberFormat("vi-VN", { minimumFractionDigits: digits, maximumFractionDigits: digits });

export function int(value) {
  return value == null ? "–" : nf(0).format(value);
}

export function decimal(value, digits = 2) {
  return value == null || Number.isNaN(value) ? "–" : nf(digits).format(value);
}

export function millions(count) {
  if (count == null) return "–";
  return count >= 1e6 ? `${decimal(count / 1e6, 2)} triệu` : int(count);
}

export function bytes(size) {
  if (size == null) return "–";
  if (size >= 1e9) return `${decimal(size / 1e9, 2)} GB`;
  if (size >= 1e6) return `${decimal(size / 1e6, 1)} MB`;
  if (size >= 1e3) return `${decimal(size / 1e3, 0)} kB`;
  return `${int(size)} B`;
}

export function duration(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return "–";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h) return `${h} giờ ${m} phút`;
  if (m) return `${m} phút ${r} giây`;
  return `${r} giây`;
}

export function clock(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return "–:–";
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = String(Math.floor((s % 3600) / 60)).padStart(h ? 2 : 1, "0");
  const r = String(s % 60).padStart(2, "0");
  return h ? `${h}:${m}:${r}` : `${m}:${r}`;
}

export function dateTime(iso) {
  if (!iso) return "–";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString("vi-VN", { day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function secondsSince(iso, until) {
  if (!iso) return null;
  const start = new Date(iso).getTime();
  const end = until ? new Date(until).getTime() : Date.now();
  return Number.isNaN(start) || Number.isNaN(end) ? null : (end - start) / 1000;
}

// "Phòng khách số 2" -> "phong_khach_so_2" (tên phòng hợp lệ của build_room.sh)
export function slug(text) {
  return String(text || "")
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/đ/g, "d")
    .replace(/Đ/g, "D")
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "_")
    .replace(/^[._-]+|[._-]+$/g, "")
    .slice(0, 64);
}

export function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
}
