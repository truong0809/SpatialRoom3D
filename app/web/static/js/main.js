// Khung ứng dụng: điều hướng theo #, thanh bên (thu gọn, mục gần đây), trạng thái server, đăng nhập.
//   #/              Tổng quan: dự án, cảnh 3D, lịch sử (dashboard.js)
//   #/new           Dự án mới: tải video / ảnh lên, cấu hình dựng (upload.js)
//   #/room/<id>     Tiến độ dựng một dự án (job.js)
//   #/view/<id>     Trình xem 3D (viewer.js; ?mode=orbit|walk|tour)
import * as api from "./api.js";
import * as dashboard from "./dashboard.js";
import * as upload from "./upload.js";
import * as job from "./job.js";
import * as ui from "./ui.js";
import * as fmt from "./fmt.js";

const view = document.getElementById("view");
const podStatus = document.getElementById("pod-status");
const loginDialog = document.getElementById("login-dialog");
const COLLAPSE_KEY = "sr3d.sidebar.collapsed";
let cleanup = null;
let serverStatus = null;
let renderId = 0;
let handoff = null;
let recentRooms = [];

const ctx = {
  status: () => serverStatus,
  refreshStatus,
  refreshRecent,
  navigate: (hash) => {
    if (location.hash === hash) render();
    else location.hash = hash;
  },
  // tổng quan -> dự án mới: file kéo thả ở tổng quan được chọn sẵn ở trang dự án mới
  setHandoff: (value) => {
    handoff = value;
  },
  takeHandoff: () => {
    const value = handoff;
    handoff = null;
    return value;
  },
};

function parseHash() {
  const hash = location.hash.replace(/^#/, "") || "/";
  const [path, query = ""] = hash.split("?");
  const parts = path.split("/").filter(Boolean).map((part) => {
    try {
      return decodeURIComponent(part);
    } catch {
      return part;
    }
  });
  return { parts, query: new URLSearchParams(query) };
}

function setNav(section, roomId) {
  for (const link of document.querySelectorAll("[data-nav]")) {
    if (link.dataset.nav === section) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  for (const link of document.querySelectorAll("#sb-recent a")) {
    if (roomId && link.dataset.room === roomId) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
}

function showError(error) {
  const box = document.createElement("div");
  box.innerHTML = `${ui.topbar([["Tổng quan", "#/"], ["Lỗi"]])}<div class="page"><div class="notice error"></div></div>`;
  box.querySelector(".notice").textContent = error?.message || String(error);
  view.replaceChildren(...box.children);
}

async function render() {
  const id = ++renderId;
  if (cleanup) {
    try {
      cleanup();
    } catch (error) {
      console.error(error);
    }
    cleanup = null;
  }
  view.replaceChildren();
  document.body.classList.remove("viewer-open", "sb-open");
  const { parts, query } = parseHash();
  // alive(): trang này còn là trang đang hiện; các bước async của trang cũ dừng lại thay vì vẽ vào trang mới
  const pageCtx = { ...ctx, alive: () => id === renderId };
  const section = parts[0] === "new" ? "new" : parts.length === 0 ? "dashboard" : "";
  setNav(section, parts[0] === "room" || parts[0] === "view" ? parts[1] : null);
  try {
    let result = null;
    if (parts.length === 0) result = await dashboard.mount(view, pageCtx);
    else if (parts[0] === "new") result = await upload.mount(view, pageCtx);
    else if (parts[0] === "room" && parts[1]) result = await job.mount(view, pageCtx, parts[1]);
    else if (parts[0] === "view" && parts[1]) {
      const viewer = await import("./viewer.js");  // three.js + Spark chỉ tải khi mở trình xem
      if (id !== renderId) return;
      result = await viewer.mount(view, pageCtx, parts[1], query);
    } else {
      showError(new Error("Không có trang này."));
    }
    if (id !== renderId) {
      if (typeof result === "function") result();  // đã chuyển trang trong lúc chờ
      return;
    }
    cleanup = typeof result === "function" ? result : null;
  } catch (error) {
    if (id === renderId && error?.status !== 401) showError(error);
  }
}

// ------------------------------------------------------------------------------------------ thanh bên
function readCollapsed() {
  try {
    return localStorage.getItem(COLLAPSE_KEY) === "1";
  } catch {
    return false;
  }
}

function setCollapsed(collapsed) {
  document.body.classList.toggle("sb-collapsed", collapsed);
  const button = document.getElementById("sb-toggle");
  button.setAttribute("aria-expanded", String(!collapsed));
  button.setAttribute("aria-label", collapsed ? "Mở rộng thanh bên" : "Thu gọn thanh bên");
  button.title = button.getAttribute("aria-label");
  try {
    localStorage.setItem(COLLAPSE_KEY, collapsed ? "1" : "0");
  } catch {
    /* trình duyệt chặn lưu: chỉ không nhớ lựa chọn */
  }
}

function closeDrawer() {
  document.body.classList.remove("sb-open");
  document.getElementById("scrim").hidden = true;
}

document.getElementById("sb-toggle").addEventListener("click", () => setCollapsed(!document.body.classList.contains("sb-collapsed")));
// liên kết "Bỏ qua tới nội dung": dấu # của địa chỉ dùng để điều hướng, nên chỉ chuyển focus, không đổi địa chỉ
document.querySelector(".skip").addEventListener("click", (event) => {
  event.preventDefault();
  view.focus();
});
document.getElementById("scrim").addEventListener("click", closeDrawer);
document.addEventListener("click", (event) => {
  if (event.target.closest('[data-action="menu"]')) {
    document.body.classList.add("sb-open");
    document.getElementById("scrim").hidden = false;
  } else if (event.target.closest("#sidebar a")) {
    closeDrawer();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && document.body.classList.contains("sb-open")) closeDrawer();
});

// mục "Gần đây": tối đa 6 dự án, mở trình xem nếu đã dựng xong, không thì trang tiến độ
async function refreshRecent() {
  if (!serverStatus?.auth?.ok) return;
  try {
    recentRooms = (await api.rooms()).rooms;
  } catch {
    return;
  }
  const list = document.getElementById("sb-recent");
  const items = recentRooms.slice(0, 6).map((room) => {
    const href = room.status === "done" && room.entry ? `#/view/${encodeURIComponent(room.id)}` : `#/room/${encodeURIComponent(room.id)}`;
    const status = room.build ? room.build.status : room.status;
    return `<li><a href="${href}" data-room="${fmt.escapeHtml(room.id)}" title="${fmt.escapeHtml(room.name)}">
      <span class="sdot tone-${ui.tone(status)}" aria-hidden="true"></span><span class="name">${fmt.escapeHtml(room.name)}</span>
      <span class="sr-only">${fmt.escapeHtml(ui.STATUS[status]?.label || "")}</span></a></li>`;
  });
  list.innerHTML = items.join("");
  document.getElementById("sb-recent-section").classList.toggle("hidden", !items.length);
  const { parts } = parseHash();
  setNav(parts[0] === "new" ? "new" : parts.length === 0 ? "dashboard" : "", parts[0] === "room" || parts[0] === "view" ? parts[1] : null);
}

// ------------------------------------------------------------------------------------- trạng thái server
function paintStatus(state, label, detail = "") {
  podStatus.className = `podstatus ${state}`;
  podStatus.querySelector(".label").textContent = label;
  document.getElementById("pod-detail").textContent = detail;
  podStatus.title = detail ? `${label}: ${detail}` : label;
}

async function refreshStatus() {
  try {
    serverStatus = await api.status();
  } catch (error) {
    serverStatus = null;
    paintStatus("off", "Mất kết nối server", "Pod tắt hoặc mạng ngắt");
    return null;
  }
  const s = serverStatus;
  if (s.mode === "view") paintStatus("", "Chế độ chỉ xem", "Không dựng phòng ở đây");
  else if (!s.auth.ok) paintStatus("", "Cần đăng nhập");
  else if (s.gpu?.busy) paintStatus("busy", "Pod GPU đang dựng", s.gpu.name || "");
  else if (s.gpu?.name) paintStatus("ok", "Pod GPU sẵn sàng", s.gpu.name);
  else paintStatus("", "Server đang chạy", "Không thấy GPU");
  return s;
}

// --------------------------------------------------------------------------------------------- đăng nhập
function askLogin() {
  if (loginDialog.open) return;
  document.getElementById("login-error").textContent = "";
  loginDialog.showModal();
  document.getElementById("login-token").focus();
}

document.getElementById("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.getElementById("login-token");
  const errorBox = document.getElementById("login-error");
  errorBox.textContent = "";
  try {
    await api.login(input.value);
    input.value = "";
    loginDialog.close();
    await refreshStatus();
    refreshRecent();
    render();
  } catch (error) {
    errorBox.textContent = error.message;
  }
});
loginDialog.addEventListener("cancel", (event) => event.preventDefault());  // phải đăng nhập mới dùng được

api.onUnauthorized(askLogin);
window.addEventListener("hashchange", render);
setCollapsed(readCollapsed());

(async () => {
  const s = await refreshStatus();
  if (s && s.auth.required && !s.auth.ok) askLogin();
  else {
    refreshRecent();
    render();
  }
  setInterval(refreshStatus, 8000);
  setInterval(refreshRecent, 10000);
})();
