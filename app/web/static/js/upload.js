// Dựng phòng mới: chọn một video hoặc nhiều ảnh, tải lên theo từng phần, rồi server gọi build_room.sh.
// Quy tắc đầu vào giống server (app/web/uploads.py, src/input/sources.py): video MP4/MOV...; ảnh JPG/PNG, tối thiểu 10.
import * as api from "./api.js";
import * as fmt from "./fmt.js";

const PHOTO_RE = /\.(jpe?g|png)$/i;
const OTHER_IMAGE_RE = /\.(heic|heif|webp|bmp|tiff?|gif|dng|raw)$/i;
const VIDEO_RE = /\.(mp4|mov|m4v|avi|mkv|webm)$/i;
const ROOM_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const naturalCompare = new Intl.Collator("en", { numeric: true, sensitivity: "base" }).compare;

const TIPS = {
  video: [
    "Khóa phơi sáng và lấy nét; tắt chống rung điện tử, HDR, bộ lọc.",
    "Đi chậm và đều, luôn dịch chuyển; không đứng một chỗ xoay tròn.",
    "Đi vòng ngoài dọc tường, vòng trong giữa phòng, thêm một vòng thấp; khép vòng ở chỗ bắt đầu.",
    "1–3 phút cho một phòng; không có người hay vật di chuyển trong khung hình.",
  ],
  photos: [
    "Nên 30–100 ảnh cho một phòng (tối thiểu 10); mỗi ảnh trùng 60–80 % khung hình với ảnh trước.",
    "Bước 30–50 cm giữa hai lần chụp; không đứng một chỗ xoay máy.",
    "Cùng chiều máy, cùng ống kính 1x, không zoom, không chân dung / HDR.",
    "Dùng file gốc của máy (còn EXIF tiêu cự), không dùng ảnh đã gửi qua ứng dụng nhắn tin.",
  ],
};

// Thư mục kéo thả vào: đọc các file nằm trực tiếp trong thư mục (giống server, không đọc thư mục con)
async function filesFromDrop(dataTransfer) {
  const items = [...(dataTransfer.items || [])];
  const entries = items.map((item) => item.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.some((entry) => entry.isDirectory)) return [...dataTransfer.files];
  const out = [];
  for (const entry of entries) {
    if (entry.isFile) out.push(await new Promise((resolve, reject) => entry.file(resolve, reject)));
    else if (entry.isDirectory) {
      const reader = entry.createReader();
      for (;;) {
        const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
        if (!batch.length) break;
        for (const child of batch) {
          if (child.isFile) out.push(await new Promise((resolve, reject) => child.file(resolve, reject)));
        }
      }
    }
  }
  return out;
}

export function checkSelection(kind, files, limits) {
  // -> {files, errors[], warnings[], ignored[]}: các file sẽ tải lên và lý do không tải được
  const errors = [];
  const warnings = [];
  const ignored = [];
  let chosen = [...files].filter((f) => !f.name.startsWith("."));
  if (kind === "video") {
    if (chosen.length > 1) errors.push("Chỉ chọn một video cho một phòng.");
    const file = chosen[0];
    if (file && PHOTO_RE.test(file.name)) errors.push("File đã chọn là ảnh: chọn mục Ảnh chụp để tải nhiều ảnh lên.");
    else if (file && !VIDEO_RE.test(file.name)) errors.push("Video phải là MP4, MOV, M4V, AVI, MKV hoặc WEBM.");
  } else {
    const other = chosen.filter((f) => !PHOTO_RE.test(f.name));
    if (other.some((f) => OTHER_IMAGE_RE.test(f.name))) {
      errors.push(`Có ${other.filter((f) => OTHER_IMAGE_RE.test(f.name)).length} ảnh không phải JPG/PNG (vd. HEIC của iPhone): xuất sang JPG trước.`);
    }
    ignored.push(...other.filter((f) => !OTHER_IMAGE_RE.test(f.name)).map((f) => f.name));
    chosen = chosen.filter((f) => PHOTO_RE.test(f.name)).sort((a, b) => naturalCompare(a.name, b.name));
    const min = limits?.min_photos ?? 10;
    const rec = limits?.recommended_photos ?? 30;
    if (chosen.length === 1) errors.push("Chỉ có 1 ảnh: không dựng được cả căn phòng từ một ảnh. Chụp thêm nhiều ảnh của cùng căn phòng ở các vị trí khác nhau.");
    else if (chosen.length > 0 && chosen.length < min) errors.push(`Chỉ có ${chosen.length} ảnh, cần ít nhất ${min} ảnh chụp cùng một căn phòng từ nhiều vị trí.`);
    else if (chosen.length >= min && chosen.length < rec) warnings.push(`${chosen.length} ảnh: dựng được nhưng nên từ ${rec} ảnh trở lên để phủ đủ căn phòng.`);
    const sizes = new Set(chosen.map((f) => f.type));
    if (sizes.size > 1) warnings.push("Ảnh có nhiều định dạng khác nhau: nên dùng cùng một máy, cùng chiều chụp.");
  }
  const total = chosen.reduce((sum, f) => sum + f.size, 0);
  if (limits?.max_upload_bytes && total > limits.max_upload_bytes) {
    errors.push(`Tổng dung lượng ${fmt.bytes(total)} vượt giới hạn ${fmt.bytes(limits.max_upload_bytes)} của server.`);
  }
  if (chosen.some((f) => f.size === 0)) errors.push("Có file rỗng (0 byte).");
  return { files: chosen, errors, warnings, ignored, total };
}

async function putChunk(uploadId, index, offset, blob, signal) {
  return api.api(`/api/uploads/${uploadId}/files/${index}?offset=${offset}`, {
    method: "PUT", body: blob, signal, headers: { "Content-Type": "application/octet-stream" },
  });
}

// Tải mọi file lên theo từng phần; lỗi mạng / 5xx thì thử lại (tối đa 6 lần mỗi phần), 409 thì đi tiếp từ chỗ server đã nhận.
export async function uploadFiles(session, files, { signal, onProgress } = {}) {
  const chunk = session.chunk_bytes;
  const total = files.reduce((sum, f) => sum + f.size, 0);
  let doneBefore = 0;
  for (let index = 0; index < files.length; index++) {
    const file = files[index];
    let offset = session.files[index].received || 0;
    while (offset < file.size) {
      const end = Math.min(offset + chunk, file.size);
      for (let attempt = 0; ; attempt++) {
        try {
          const result = await putChunk(session.id, index, offset, file.slice(offset, end), signal);
          offset = result.received;
          break;
        } catch (error) {
          if (error.name === "AbortError") throw error;
          if (error.status === 409 && typeof error.data?.received === "number") {
            offset = error.data.received;
            break;
          }
          const retryable = !error.status || error.status >= 500 || error.status === 408 || error.status === 429;
          if (!retryable || attempt >= 5) throw error;
          onProgress?.({ sent: doneBefore + offset, total, file: file.name, index, retrying: attempt + 1 });
          await sleep(1000 * 2 ** attempt);
          try {
            offset = (await api.api(`/api/uploads/${session.id}`, { signal })).files[index].received;
          } catch (inner) {
            if (inner.name === "AbortError") throw inner;
          }
        }
      }
      onProgress?.({ sent: doneBefore + offset, total, file: file.name, index });
    }
    doneBefore += file.size;
  }
}

export async function mount(container, ctx) {
  const status = ctx.status() || (await ctx.refreshStatus());
  const limits = status?.limits;
  const page = document.createElement("div");
  page.className = "page";
  container.append(page);
  if (status?.mode === "view") {
    page.innerHTML = `<div class="page-head"><div><h1>Dựng phòng mới</h1></div></div>
      <div class="notice">Server này chỉ để xem thư viện. Dựng phòng trên trang web chạy ở pod GPU (<code>bash scripts/web_server.sh start</code>).</div>`;
    return () => {};
  }
  page.innerHTML = `<div class="page-head"><div><h1>Dựng phòng mới</h1>
      <p class="lead">Tải video hoặc ảnh chụp căn phòng lên. Pod GPU tự tách khung hoặc chuẩn hóa ảnh, chạy COLMAP để ước lượng camera, train mô hình chính rồi đưa phòng vào thư viện.</p></div></div>
    <div class="columns">
      <form class="panel main-col" id="up-form" novalidate>
        <div class="field"><span class="label" id="kind-label">1. Nguồn</span>
          <div class="seg" role="group" aria-labelledby="kind-label">
            <button type="button" data-kind="video" aria-pressed="true">Video</button>
            <button type="button" data-kind="photos" aria-pressed="false">Ảnh chụp</button>
          </div></div>
        <div class="field">
          <label class="dropzone" id="up-drop" for="up-input">
            <svg aria-hidden="true" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#4C9AFF" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4"/><path d="m6 10 6-6 6 6"/><path d="M4 20h16"/></svg>
            <strong id="up-drop-title"></strong><span class="muted" id="up-drop-hint"></span>
          </label>
          <input id="up-input" type="file" class="sr-only">
          <div class="row"><button type="button" class="btn small hidden" id="up-folder">Chọn cả thư mục ảnh</button></div>
          <input id="up-folder-input" type="file" class="sr-only" webkitdirectory multiple>
          <div class="filelist" id="up-files" aria-live="polite"></div>
        </div>
        <div class="field"><label for="up-name">2. Tên hiển thị</label>
          <input id="up-name" type="text" maxlength="80" placeholder="vd. Phòng khách" autocomplete="off"></div>
        <div class="field"><label for="up-room">3. Mã phòng</label>
          <input id="up-room" type="text" maxlength="64" placeholder="vd. phong_khach" autocomplete="off" spellcheck="false">
          <span class="hint">Chữ không dấu, số, '.', '_', '-'; dùng làm tên thư mục trên pod. Mỗi phòng một mã.</span></div>
        <div class="notice">Phòng được dựng bằng <strong>3DGS-MCMC</strong> (gsplat), mô hình chính chọn qua thực nghiệm ngày 31/10/2026: hơn 3DGS mặc định ở 17/18 cặp (cảnh, seed) về PSNR, 18/18 về SSIM và LPIPS. Tối đa 1 triệu Gaussian, nên file xem trên web ≤ 32 MB.</div>
        <div class="notice error hidden" id="up-error" role="alert"></div>
        <div class="upload-progress hidden" id="up-progress">
          <div class="row" style="justify-content:space-between"><span id="up-progress-text">Đang tải lên…</span><span class="mono" id="up-progress-pct">0 %</span></div>
          <div class="progressbar"><span id="up-progress-bar"></span></div>
          <span class="muted" id="up-progress-file"></span>
        </div>
        <div class="row">
          <button type="submit" class="btn primary" id="up-start" disabled>Bắt đầu dựng</button>
          <button type="button" class="btn danger hidden" id="up-cancel">Hủy tải lên</button>
          <span class="muted" style="font-size:13px">Train trên RTX 4090 mất khoảng 11 phút (phòng nhóm), chưa tính tách khung và COLMAP.</span>
        </div>
      </form>
      <div class="side-col">
        <section class="panel quiet"><h2 id="tips-title">Mẹo quay để dựng đẹp</h2><ul class="tips" id="tips"></ul>
          <p class="muted" style="font-size:13px">Đầy đủ: <code>docs/data/capture_protocol.md</code>.</p></section>
      </div>
    </div>`;

  const $ = (sel) => page.querySelector(sel);
  const input = $("#up-input");
  const folderInput = $("#up-folder-input");
  const drop = $("#up-drop");
  const nameInput = $("#up-name");
  const roomInput = $("#up-room");
  let kind = "video";
  let selection = { files: [], errors: [], warnings: [], ignored: [], total: 0 };
  let roomEdited = false;
  let controller = null;
  let uploading = false;

  function setKind(next) {
    kind = next;
    for (const button of page.querySelectorAll("[data-kind]")) button.setAttribute("aria-pressed", String(button.dataset.kind === kind));
    input.multiple = kind === "photos";
    input.accept = kind === "photos" ? "image/jpeg,image/png,.jpg,.jpeg,.png" : "video/*,.mp4,.mov,.m4v,.avi,.mkv,.webm";
    $("#up-drop-title").textContent = kind === "photos" ? "Kéo thả ảnh hoặc cả thư mục ảnh vào đây, hoặc bấm để chọn" : "Kéo thả video vào đây hoặc bấm để chọn";
    $("#up-drop-hint").textContent = kind === "photos"
      ? `JPG hoặc PNG · tối thiểu ${limits?.min_photos ?? 10}, nên từ ${limits?.recommended_photos ?? 30} ảnh`
      : "MP4 hoặc MOV · 1–3 phút cho một phòng · 4K hoặc 1080p, 30 fps";
    $("#up-folder").classList.toggle("hidden", kind !== "photos");
    $("#tips-title").textContent = kind === "photos" ? "Mẹo chụp để dựng đẹp" : "Mẹo quay để dựng đẹp";
    $("#tips").innerHTML = TIPS[kind].map((tip) => `<li>${fmt.escapeHtml(tip)}</li>`).join("");
    select([]);
  }

  function select(files) {
    selection = checkSelection(kind, files, limits);
    const box = $("#up-files");
    const parts = [];
    if (selection.files.length) {
      const names = selection.files.slice(0, 60).map((f) => fmt.escapeHtml(f.name)).join(", ");
      parts.push(`<span><strong>${selection.files.length === 1 ? fmt.escapeHtml(selection.files[0].name) : `${selection.files.length} ảnh`}</strong> · ${fmt.bytes(selection.total)}</span>`);
      if (selection.files.length > 1) parts.push(`<span class="names">${names}${selection.files.length > 60 ? " …" : ""}</span>`);
    }
    if (selection.ignored.length) parts.push(`<span class="muted">Bỏ qua ${selection.ignored.length} file không phải ảnh JPG/PNG.</span>`);
    for (const warning of selection.warnings) parts.push(`<span class="notice warn">${fmt.escapeHtml(warning)}</span>`);
    for (const error of selection.errors) parts.push(`<span class="notice error">${fmt.escapeHtml(error)}</span>`);
    box.innerHTML = parts.join("");
    if (!nameInput.value && selection.files.length === 1 && kind === "video") {
      nameInput.value = selection.files[0].name.replace(/\.[^.]+$/, "").slice(0, 80);
      syncRoom();
    }
    validate();
  }

  function syncRoom() {
    if (!roomEdited) roomInput.value = fmt.slug(nameInput.value);
    validate();
  }

  function validate() {
    const room = roomInput.value.trim();
    const roomOk = ROOM_RE.test(room) && !["web", "uploads", "server"].includes(room.toLowerCase());
    roomInput.classList.toggle("invalid", Boolean(room) && !roomOk);
    $("#up-start").disabled = uploading || !roomOk || !selection.files.length || selection.errors.length > 0;
  }

  function showError(message) {
    const box = $("#up-error");
    box.textContent = message || "";
    box.classList.toggle("hidden", !message);
  }

  function setUploading(state) {
    uploading = state;
    $("#up-cancel").classList.toggle("hidden", !state);
    $("#up-progress").classList.toggle("hidden", !state);
    for (const el of page.querySelectorAll("#up-form input, [data-kind], #up-folder")) el.disabled = state;
    validate();
  }

  let lastSample = null;
  let speed = null;
  function progress({ sent, total, file, retrying }) {
    const now = performance.now();
    if (lastSample && now - lastSample.t > 800) {
      const instant = (sent - lastSample.sent) / ((now - lastSample.t) / 1000);
      speed = speed == null ? instant : 0.7 * speed + 0.3 * instant;
      lastSample = { t: now, sent };
    } else if (!lastSample) lastSample = { t: now, sent };
    const pct = total ? (100 * sent) / total : 0;
    $("#up-progress-bar").style.width = `${pct.toFixed(1)}%`;
    $("#up-progress-pct").textContent = `${fmt.decimal(pct, 0)} %`;
    const eta = speed > 0 ? (total - sent) / speed : null;
    $("#up-progress-text").textContent = retrying
      ? `Mạng lỗi, đang thử lại (lần ${retrying})…`
      : `Đang tải lên ${fmt.bytes(sent)} / ${fmt.bytes(total)}${speed > 0 ? ` · ${fmt.bytes(speed)}/s · còn khoảng ${fmt.duration(eta)}` : ""}`;
    $("#up-progress-file").textContent = file ? `File: ${file}` : "";
  }

  async function start(event) {
    event.preventDefault();
    if ($("#up-start").disabled) return;
    showError("");
    setUploading(true);
    controller = new AbortController();
    lastSample = null;
    speed = null;
    const room = roomInput.value.trim();
    let session = null;
    try {
      session = await api.api("/api/uploads", {
        method: "POST", signal: controller.signal,
        json: { room, name: nameInput.value.trim() || room, kind, files: selection.files.map((f) => ({ name: f.name, size: f.size })) },
      });
      await uploadFiles(session, selection.files, { signal: controller.signal, onProgress: progress });
      $("#up-progress-text").textContent = "Đã tải lên đủ. Đang kiểm tra nguồn và xếp việc dựng phòng…";
      const result = await api.api(`/api/uploads/${session.id}/finish`, { method: "POST", json: {}, signal: controller.signal });
      setUploading(false);
      ctx.refreshStatus();
      ctx.navigate(`#/room/${encodeURIComponent(result.room)}`);
    } catch (error) {
      setUploading(false);
      if (error.name === "AbortError") {
        showError("Đã hủy tải lên.");
        if (session) api.api(`/api/uploads/${session.id}`, { method: "DELETE" }).catch(() => {});
      } else {
        showError(error.message);
      }
    }
  }

  page.querySelectorAll("[data-kind]").forEach((button) => button.addEventListener("click", () => setKind(button.dataset.kind)));
  input.addEventListener("change", () => select(input.files));
  folderInput.addEventListener("change", () => select(folderInput.files));
  $("#up-folder").addEventListener("click", () => folderInput.click());
  drop.addEventListener("dragover", (event) => {
    event.preventDefault();
    drop.classList.add("drag");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
  drop.addEventListener("drop", async (event) => {
    event.preventDefault();
    drop.classList.remove("drag");
    if (uploading) return;
    const files = await filesFromDrop(event.dataTransfer);
    if (kind === "video" && files.length > 1 && files.every((f) => PHOTO_RE.test(f.name))) setKind("photos");
    select(files);
  });
  nameInput.addEventListener("input", syncRoom);
  roomInput.addEventListener("input", () => {
    roomEdited = Boolean(roomInput.value);
    validate();
  });
  $("#up-form").addEventListener("submit", start);
  $("#up-cancel").addEventListener("click", () => controller?.abort());
  const beforeUnload = (event) => {
    if (uploading) {
      event.preventDefault();
      event.returnValue = "";
    }
  };
  window.addEventListener("beforeunload", beforeUnload);
  setKind("video");
  return () => {
    window.removeEventListener("beforeunload", beforeUnload);
    controller?.abort();
  };
}
