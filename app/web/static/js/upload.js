// Dự án mới: chọn một video hoặc nhiều ảnh (xem trước, bỏ bớt ảnh), đặt tên, xem cấu hình dựng, tải lên theo từng
// phần, rồi server gọi build_room.sh. Quy tắc đầu vào giống server (app/web/uploads.py, src/input/sources.py):
// video MP4/MOV...; ảnh JPG/PNG, tối thiểu 10.
import * as api from "./api.js";
import * as fmt from "./fmt.js";
import * as ui from "./ui.js";

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
export async function filesFromDrop(dataTransfer) {
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
    errors.push(`Tổng dung lượng ${fmt.gib(total)} vượt giới hạn ${fmt.gib(limits.max_upload_bytes)} của server.`);
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

const MAX_THUMBS = 47;  // ảnh xem trước; còn lại hiện số lượng

function specList(pipeline) {
  if (!pipeline) return '<p class="hint">Cấu hình chi tiết hiện khi trang chạy trên pod (server đọc configs/product.yaml).</p>';
  const init = pipeline.init === "sfm" ? "Điểm 3D từ COLMAP" : pipeline.init || "–";
  const rows = [
    ["Số Gaussian tối đa", fmt.int(pipeline.cap_max)],
    ["Số vòng huấn luyện", fmt.int(pipeline.max_steps)],
    ["Khởi tạo", init],
    ["Khung lấy từ video", pipeline.frames?.video_max_frames ? `Tối đa ${fmt.int(pipeline.frames.video_max_frames)} khung nét nhất` : "–"],
    ["Cạnh dài của ảnh", pipeline.frames?.max_side ? `≤ ${fmt.int(pipeline.frames.max_side)} px` : "–"],
    ["Thư viện huấn luyện", `${pipeline.trainer || "gsplat"}${pipeline.trainer_commit ? ` ${pipeline.trainer_commit}` : ""}`],
  ];
  return `<dl class="spec">${rows.map(([k, v]) => `<dt>${fmt.escapeHtml(k)}</dt><dd>${fmt.escapeHtml(v)}</dd>`).join("")}</dl>`;
}

export async function mount(container, ctx) {
  const status = ctx.status() || (await ctx.refreshStatus());
  const limits = status?.limits;
  const wrap = document.createElement("div");
  if (status?.mode === "view") {
    wrap.innerHTML = `${ui.topbar([["Tổng quan", "#/"], ["Dự án mới"]])}
      <div class="page"><div class="ws-head"><h1>Dự án mới</h1></div>
      <div class="notice">Server này chỉ để xem các cảnh đã dựng. Dựng cảnh mới trên trang web chạy ở pod GPU (<code>bash scripts/web_server.sh start</code>).</div></div>`;
    container.append(...wrap.children);
    return () => {};
  }
  const minPhotos = limits?.min_photos ?? 10;
  wrap.innerHTML = `${ui.topbar([["Tổng quan", "#/"], ["Dự án mới"]])}
    <div class="page">
      <div class="ws-head"><h1>Dự án mới</h1>
        <p>Tải video hoặc ảnh chụp căn phòng lên. Pod GPU tách khung hoặc chuẩn hóa ảnh, ước lượng camera bằng COLMAP, huấn luyện 3DGS-MCMC rồi đưa cảnh vào trang tổng quan.</p></div>
      <form class="ws" id="up-form" novalidate>
        <div class="ws-main">
          <section class="panel" aria-labelledby="src-title">
            <div class="panel-head"><div><h2 id="src-title">Dữ liệu đầu vào</h2><p class="sub" id="src-sub"></p></div>
              <div class="seg" role="group" aria-label="Loại nguồn">
                <button type="button" data-kind="video" aria-pressed="true">${ui.icon("video", 16)}Video</button>
                <button type="button" data-kind="photos" aria-pressed="false">${ui.icon("image", 16)}Ảnh chụp</button></div></div>
            <label class="dropzone" id="up-drop" for="up-input">${ui.splatArt("art")}<span class="dz-icon" aria-hidden="true">${ui.icon("upload", 20)}</span>
              <span class="dz-text"><strong id="up-drop-title"></strong><br><span class="hint" id="up-drop-hint"></span></span>
              <span class="pick"><span class="btn" aria-hidden="true">${ui.icon("upload", 16)}<span id="up-pick-label">Chọn file</span></span></span></label>
            <input id="up-input" type="file" class="sr-only">
            <input id="up-folder-input" type="file" class="sr-only" webkitdirectory multiple>
            <div class="row-folder hidden" id="up-folder-row"><button type="button" class="btn sm quiet" id="up-folder">${ui.icon("folder", 16)}Chọn cả thư mục ảnh</button></div>
            <div class="preview hidden" id="up-preview" aria-live="polite"></div>
            <div class="filelist" id="up-files" aria-live="polite"></div>
          </section>
          <details class="guide" id="up-guide">
            <summary>${ui.icon("camera", 18)}<span id="tips-title"></span><span class="chev">${ui.icon("chevDown", 16)}</span></summary>
            <div class="guide-body"><ul class="tips" id="tips"></ul><p class="hint">Hướng dẫn đầy đủ: <code>docs/data/capture_protocol.md</code>.</p></div>
          </details>
        </div>
        <aside class="ws-side" aria-label="Thiết lập và bắt đầu">
          <section class="panel" aria-labelledby="set-title">
            <h2 id="set-title">Thiết lập dự án</h2>
            <div class="field"><label for="up-name">Tên dự án</label>
              <input id="up-name" class="input" type="text" maxlength="80" placeholder="Ví dụ: Phòng khách" autocomplete="off"></div>
            <div class="field"><label for="up-room">Mã dự án</label>
              <input id="up-room" class="input" type="text" maxlength="64" placeholder="phong_khach" autocomplete="off" spellcheck="false">
              <span class="hint">Tự tạo từ tên. Chữ không dấu, số, dấu chấm, gạch dưới, gạch ngang; là tên thư mục trên pod nên mỗi dự án một mã.</span></div>
          </section>
          <section class="panel" aria-labelledby="method-title">
            <div class="method"><span class="chipmark">${ui.icon("spark", 18)}</span>
              <div><b id="method-title">3DGS-MCMC</b><span>Gaussian Splatting huấn luyện kiểu MCMC (Kheradmand và cộng sự, 2024)</span></div></div>
            ${specList(status?.pipeline)}
            <p class="hint">Cấu hình cố định của sản phẩm (<code>configs/product.yaml</code>), chọn qua thực nghiệm 31/10/2026: hơn 3DGS mặc định ở 17/18 cặp (cảnh, seed) về PSNR, 18/18 về SSIM và LPIPS.</p>
          </section>
          <section class="panel" aria-labelledby="go-title">
            <h2 id="go-title">Sẵn sàng dựng</h2>
            <ul class="checklist" id="up-check">
              <li data-check="source"><span class="ic">${ui.icon("check", 12, 2.6)}</span><span id="check-source"></span></li>
              <li data-check="room"><span class="ic">${ui.icon("check", 12, 2.6)}</span><span>Tên và mã dự án hợp lệ</span></li>
              <li data-check="size"><span class="ic">${ui.icon("check", 12, 2.6)}</span><span>Dung lượng không quá ${fmt.gib(limits?.max_upload_bytes ?? 0)}</span></li>
            </ul>
            <div class="notice error hidden" id="up-error" role="alert"></div>
            <div class="upload-progress hidden" id="up-progress">
              <div class="row"><span id="up-progress-text">Đang tải lên…</span><span class="num" id="up-progress-pct">0 %</span></div>
              <div class="progressbar"><span id="up-progress-bar"></span></div>
              <span class="muted" id="up-progress-file"></span>
            </div>
            <button type="submit" class="btn primary lg block" id="up-start" disabled>${ui.icon("spark", 17)}Bắt đầu dựng</button>
            <button type="button" class="btn danger block hidden" id="up-cancel">Hủy tải lên</button>
            <p class="hint">Sau khi tải lên, pod tách khung, chạy COLMAP, huấn luyện rồi xuất cảnh cho web; trang chuyển sang tiến độ. Huấn luyện trên RTX 4090 mất khoảng 11 phút (phòng nhóm), chưa tính tách khung và COLMAP.</p>
          </section>
        </aside>
      </form>
    </div>`;
  container.append(...wrap.children);

  const $ = (sel) => container.querySelector(sel);
  const input = $("#up-input");
  const folderInput = $("#up-folder-input");
  const drop = $("#up-drop");
  const nameInput = $("#up-name");
  const roomInput = $("#up-room");
  let kind = "video";
  let picked = [];  // file người dùng đã chọn (trước khi lọc)
  let selection = { files: [], errors: [], warnings: [], ignored: [], total: 0 };
  let roomEdited = false;
  let controller = null;
  let uploading = false;
  let unmounted = false;  // đã rời trang: các bước async còn dở không được đụng vào trang mới
  const urls = new Map();  // File -> object URL của ảnh / video xem trước
  const urlOf = (file) => {
    if (!urls.has(file)) urls.set(file, URL.createObjectURL(file));
    return urls.get(file);
  };
  const forget = (keep) => {
    for (const [file, url] of urls) {
      if (!keep.has(file)) {
        URL.revokeObjectURL(url);
        urls.delete(file);
      }
    }
  };

  function setKind(next, files = []) {
    kind = next;
    for (const button of container.querySelectorAll("[data-kind]")) button.setAttribute("aria-pressed", String(button.dataset.kind === kind));
    input.multiple = kind === "photos";
    input.accept = kind === "photos" ? "image/jpeg,image/png,.jpg,.jpeg,.png" : "video/*,.mp4,.mov,.m4v,.avi,.mkv,.webm";
    $("#src-sub").textContent = kind === "photos"
      ? `Nhiều ảnh chụp cùng căn phòng từ các vị trí khác nhau: tối thiểu ${minPhotos}, nên từ ${limits?.recommended_photos ?? 30} ảnh.`
      : "Một video đi chậm quanh căn phòng, 1–3 phút.";
    $("#check-source").textContent = kind === "photos" ? `Ít nhất ${minPhotos} ảnh JPG hoặc PNG` : "Một video MP4, MOV, M4V, AVI, MKV hoặc WEBM";
    $("#up-folder-row").classList.toggle("hidden", kind !== "photos");
    $("#tips-title").textContent = kind === "photos" ? "Cách chụp ảnh để dựng đẹp" : "Cách quay video để dựng đẹp";
    $("#tips").innerHTML = TIPS[kind].map((tip) => `<li>${fmt.escapeHtml(tip)}</li>`).join("");
    select(files);
  }

  function addFiles(files) {
    if (kind !== "photos") return select([...files]);
    const seen = new Set(picked.map((f) => `${f.name}:${f.size}`));
    return select([...picked, ...[...files].filter((f) => !seen.has(`${f.name}:${f.size}`))]);
  }

  function select(files) {
    picked = [...files];
    selection = checkSelection(kind, picked, limits);
    forget(new Set(selection.files));
    const has = selection.files.length > 0;
    drop.classList.toggle("compact", has);
    $("#up-drop-title").textContent = has
      ? (kind === "photos" ? "Thêm ảnh hoặc kéo thả thêm vào đây" : "Kéo thả video khác vào đây để đổi")
      : (kind === "photos" ? "Kéo thả ảnh hoặc cả thư mục ảnh vào đây" : "Kéo thả video căn phòng vào đây");
    $("#up-drop-hint").textContent = kind === "photos"
      ? `JPG hoặc PNG, tối thiểu ${minPhotos} ảnh`
      : "MP4 hoặc MOV, 4K hoặc 1080p, 30 fps";
    $("#up-pick-label").textContent = has ? (kind === "photos" ? "Thêm ảnh" : "Đổi video") : "Chọn file";
    paintPreview();
    const notes = [];
    if (selection.ignored.length) notes.push(`<span class="notice">Bỏ qua ${selection.ignored.length} file không phải ảnh JPG/PNG.</span>`);
    for (const warning of selection.warnings) notes.push(`<span class="notice warn">${fmt.escapeHtml(warning)}</span>`);
    for (const error of selection.errors) notes.push(`<span class="notice error">${fmt.escapeHtml(error)}</span>`);
    $("#up-files").innerHTML = notes.join("");
    if (!nameInput.value && selection.files.length === 1 && kind === "video") {
      nameInput.value = selection.files[0].name.replace(/\.[^.]+$/, "").slice(0, 80);
      syncRoom();
    }
    validate();
  }

  function paintPreview() {
    const box = $("#up-preview");
    const files = selection.files;
    box.classList.toggle("hidden", !files.length);
    if (!files.length) {
      box.innerHTML = "";
      return;
    }
    if (kind === "video") {
      const file = files[0];
      box.innerHTML = `<div class="video-preview"><video controls muted playsinline preload="metadata"></video></div>
        <dl class="facts-grid"><div><dt>Tên file</dt><dd>${fmt.escapeHtml(file.name)}</dd></div><div><dt>Dung lượng</dt><dd>${fmt.bytes(file.size)}</dd></div>
          <div><dt>Thời lượng</dt><dd id="vid-duration">–</dd></div><div><dt>Độ phân giải</dt><dd id="vid-size">–</dd></div></dl>`;
      const video = box.querySelector("video");
      video.addEventListener("loadedmetadata", () => {
        if (Number.isFinite(video.duration)) box.querySelector("#vid-duration").textContent = fmt.clock(video.duration);
        if (video.videoWidth) box.querySelector("#vid-size").textContent = `${video.videoWidth} × ${video.videoHeight}`;
      });
      video.addEventListener("error", () => {
        box.querySelector(".video-preview").innerHTML = '<p class="notice" style="margin:auto">Trình duyệt không xem trước được video này; vẫn tải lên và dựng được.</p>';
      }, { once: true });
      video.src = urlOf(file);
      return;
    }
    const shown = files.slice(0, MAX_THUMBS);
    const more = files.length - shown.length;
    box.innerHTML = `<div class="preview-head"><div><b>${fmt.int(files.length)} ảnh</b><span class="muted">${fmt.bytes(selection.total)}</span></div>
        <button type="button" class="btn sm quiet" id="up-clear">${ui.icon("x", 15)}Bỏ tất cả</button></div>
      <div class="thumbs">${shown.map((file, i) => `<figure><img alt="" decoding="async" data-i="${i}">
          <figcaption>${fmt.escapeHtml(file.name)}</figcaption>
          <button type="button" class="rm" data-rm="${i}" aria-label="Bỏ ${fmt.escapeHtml(file.name)}">${ui.icon("x", 14, 2.2)}</button></figure>`).join("")}
        ${more > 0 ? `<figure class="more">và ${fmt.int(more)} ảnh nữa</figure>` : ""}</div>`;
    box.querySelectorAll("img[data-i]").forEach((img) => {
      img.src = urlOf(shown[Number(img.dataset.i)]);
    });
  }

  function syncRoom() {
    if (!roomEdited) roomInput.value = fmt.slug(nameInput.value);
    validate();
  }

  function validate() {
    const room = roomInput.value.trim();
    const roomOk = ROOM_RE.test(room) && !["web", "uploads", "server"].includes(room.toLowerCase());
    roomInput.classList.toggle("invalid", Boolean(room) && !roomOk);
    const sourceOk = selection.files.length > 0 && !selection.errors.some((e) => !/dung lượng/i.test(e));
    const sizeOk = !limits?.max_upload_bytes || selection.total <= limits.max_upload_bytes;
    const checks = { source: sourceOk, room: roomOk, size: selection.files.length > 0 && sizeOk };
    for (const li of container.querySelectorAll("#up-check li")) li.classList.toggle("ok", Boolean(checks[li.dataset.check]));
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
    $("#up-start").classList.toggle("hidden", state);
    $("#up-progress").classList.toggle("hidden", !state);
    for (const el of container.querySelectorAll("#up-form input, [data-kind], #up-folder, #up-clear, .rm")) el.disabled = state;
    drop.classList.toggle("disabled", state);
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
      : `Đang tải lên ${fmt.bytes(sent)} / ${fmt.bytes(total)}${speed > 0 ? `, ${fmt.bytes(speed)}/s, còn khoảng ${fmt.duration(eta)}` : ""}`;
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
      $("#up-progress-text").textContent = "Đã tải lên đủ. Đang kiểm tra nguồn và xếp việc dựng…";
      const result = await api.api(`/api/uploads/${session.id}/finish`, { method: "POST", json: {}, signal: controller.signal });
      setUploading(false);
      ctx.refreshStatus();
      ctx.refreshRecent?.();
      ctx.navigate(`#/room/${encodeURIComponent(result.room)}`);
    } catch (error) {
      if (unmounted) {  // rời trang giữa lúc tải lên: chỉ báo server bỏ phiên tải lên dở
        if (session) api.api(`/api/uploads/${session.id}`, { method: "DELETE" }).catch(() => {});
        return;
      }
      setUploading(false);
      if (error.name === "AbortError") {
        showError("Đã hủy tải lên.");
        if (session) api.api(`/api/uploads/${session.id}`, { method: "DELETE" }).catch(() => {});
      } else {
        showError(error.message);
      }
    }
  }

  container.querySelectorAll("[data-kind]").forEach((button) => button.addEventListener("click", () => {
    if (button.dataset.kind !== kind) setKind(button.dataset.kind);
  }));
  input.addEventListener("change", () => {
    addFiles(input.files);
    input.value = "";  // chọn lại đúng file đó vẫn nhận
  });
  folderInput.addEventListener("change", () => {
    addFiles(folderInput.files);
    folderInput.value = "";
  });
  $("#up-folder").addEventListener("click", () => folderInput.click());
  $("#up-preview").addEventListener("click", (event) => {
    if (uploading) return;
    const remove = event.target.closest("[data-rm]");
    if (remove) {
      const file = selection.files[Number(remove.dataset.rm)];
      select(picked.filter((f) => f !== file));
    } else if (event.target.closest("#up-clear")) {
      select([]);
    }
  });
  drop.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (!uploading) drop.classList.add("drag");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
  drop.addEventListener("drop", async (event) => {
    event.preventDefault();
    drop.classList.remove("drag");
    if (uploading) return;
    const files = await filesFromDrop(event.dataTransfer);
    if (kind === "video" && files.length > 1 && files.every((f) => PHOTO_RE.test(f.name))) setKind("photos", files);
    else addFiles(files);
  });
  drop.addEventListener("click", (event) => {
    if (uploading) event.preventDefault();
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

  // file kéo thả ở trang tổng quan: chọn sẵn ở đây
  const handed = ctx.takeHandoff?.();
  const files = handed?.files || [];
  const startKind = files.length && files.every((f) => PHOTO_RE.test(f.name) || f.name.startsWith(".")) ? "photos" : "video";
  setKind(startKind, files);
  if (!files.length) $("#up-guide").open = true;  // lần đầu: hiện cách quay / chụp
  return () => {
    unmounted = true;
    window.removeEventListener("beforeunload", beforeUnload);
    controller?.abort();
    forget(new Set());
  };
}
