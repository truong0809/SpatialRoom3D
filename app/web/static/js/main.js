// Khung trang: điều hướng theo #, trạng thái pod ở thanh trên, đăng nhập.
//   #/              Thư viện phòng (library.js)
//   #/new           Dựng phòng mới: tải video / ảnh lên (upload.js)
//   #/room/<id>     Tiến độ dựng phòng (job.js)
//   #/view/<id>     Xem 3D (viewer.js; ?mode=orbit|walk|tour)
import * as api from "./api.js";
import * as library from "./library.js";
import * as upload from "./upload.js";
import * as job from "./job.js";

const view = document.getElementById("view");
const podStatus = document.getElementById("pod-status");
const loginDialog = document.getElementById("login-dialog");
let cleanup = null;
let serverStatus = null;
let renderId = 0;

const ctx = {
  status: () => serverStatus,
  refreshStatus,
  navigate: (hash) => {
    if (location.hash === hash) render();
    else location.hash = hash;
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

function setNav(section) {
  for (const link of document.querySelectorAll("[data-nav]")) {
    if (link.dataset.nav === section) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
}

function showError(error) {
  const box = document.createElement("div");
  box.className = "page";
  const notice = document.createElement("div");
  notice.className = "notice error";
  notice.textContent = error?.message || String(error);
  box.append(notice);
  view.replaceChildren(box);
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
  document.body.classList.remove("viewer-open");  // trình xem 3D đặt lại khi mở
  const { parts, query } = parseHash();
  const section = parts[0] === "new" ? "new" : parts.length === 0 ? "library" : "";
  setNav(section);
  try {
    let result = null;
    if (parts.length === 0) result = await library.mount(view, ctx);
    else if (parts[0] === "new") result = await upload.mount(view, ctx);
    else if (parts[0] === "room" && parts[1]) result = await job.mount(view, ctx, parts[1]);
    else if (parts[0] === "view" && parts[1]) {
      const viewer = await import("./viewer.js");  // three.js + Spark chỉ tải khi mở trình xem
      result = await viewer.mount(view, ctx, parts[1], query);
    } else {
      showError(new Error("Không có trang này"));
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

// ------------------------------------------------------------------------------------------ trạng thái pod
function paintStatus(state, text) {
  podStatus.className = `podstatus ${state}`;
  podStatus.querySelector(".label").textContent = text;
}

async function refreshStatus() {
  try {
    serverStatus = await api.status();
  } catch (error) {
    serverStatus = null;
    paintStatus("off", "Mất kết nối tới server");
    return null;
  }
  const s = serverStatus;
  if (s.mode === "view") paintStatus("", "Chỉ xem (không dựng phòng ở đây)");
  else if (!s.auth.ok) paintStatus("", "Cần đăng nhập");
  else if (s.gpu?.busy) paintStatus("busy", `Pod GPU: đang dựng phòng${s.gpu.name ? ` · ${s.gpu.name}` : ""}`);
  else if (s.gpu?.name) paintStatus("ok", `Pod GPU: rảnh · ${s.gpu.name}`);
  else paintStatus("", "Server chạy · không thấy GPU");
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
    render();
  } catch (error) {
    errorBox.textContent = error.message;
  }
});
loginDialog.addEventListener("cancel", (event) => event.preventDefault());  // phải đăng nhập mới dùng được

api.onUnauthorized(askLogin);
window.addEventListener("hashchange", render);

(async () => {
  const s = await refreshStatus();
  if (s && s.auth.required && !s.auth.ok) askLogin();
  else render();
  setInterval(refreshStatus, 8000);
})();
