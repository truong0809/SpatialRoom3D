// Xem 3D một phòng: model.splat (Spark trên three.js, WebGL2) và cameras.json, ba chế độ xoay quanh / đi bộ / tham quan.
// Hệ tọa độ: model.splat và cameras.json cùng hệ đã chuẩn hóa của gsplat (docs/design/product.md mục 4.4): không xoay
// mô hình; hướng lên của phòng là trường `up` (không cố định +y hay +z).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as api from "./api.js";
import * as fmt from "./fmt.js";
import * as cam from "./camera.js";
import * as ui from "./ui.js";

const QUALITY = { high: 1, medium: 0.5, light: 0.2 };  // phần Gaussian được vẽ; file xếp theo độ quan trọng giảm dần
const RECORD_FPS = 30;
const MAX_VIEWS = 12;  // số góc nhìn (lấy đều trên đường quay) trong mục Góc nhìn
const MODE_LABEL = { orbit: "Xoay quanh", walk: "Đi bộ", tour: "Tham quan" };
const TOUCH = window.matchMedia?.("(pointer: coarse)").matches;  // điện thoại, máy tính bảng
// gợi ý thao tác ở góc khung xem: [thao tác, kết quả]; ẩn dần sau HINT_MS hoặc ngay sau lần thao tác đầu
const HINTS = TOUCH ? {
  orbit: [["Một ngón", "xoay"], ["Hai ngón", "phóng to, thu nhỏ, dời khung nhìn"]],
  walk: [["Kéo", "nhìn quanh"], ["Đi lại", "cần bàn phím; trên điện thoại dùng chế độ Tham quan"]],
  tour: [["Tự động", "đi lại đúng đường đã quay"], ["Kéo", "nhìn quanh, thả ra để đi tiếp"]],
} : {
  orbit: [["Kéo chuột trái", "xoay"], ["Cuộn chuột", "phóng to, thu nhỏ"], ["Kéo chuột phải", "dời khung nhìn"], ["R", "về góc ban đầu"]],
  walk: [["Nhấp khung xem", "điều khiển bằng chuột"], ["Esc", "trả lại con trỏ chuột"]],
  tour: [["Tự động", "đi lại đúng đường đã quay"], ["Kéo chuột", "nhìn quanh, thả ra để đi tiếp"]],
};
const HINT_MS = 12000;
const SFM_VERDICT = { PASS: "Đạt", WARN: "Đạt, có cảnh báo", FAIL: "Không đạt" };  // src/reconstruction/report.py

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

function stamp() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

async function fetchBytes(url, expected, onProgress, signal) {
  const response = await fetch(url, { credentials: "same-origin", signal });
  if (response.status === 401) throw new api.ApiError("Cần đăng nhập", 401);
  if (!response.ok) throw new api.ApiError(`Không tải được mô hình (${response.status})`, response.status);
  const total = Number(response.headers.get("Content-Length")) || expected || 0;
  if (!response.body) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const out = total ? new Uint8Array(total) : null;
  const parts = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (out && received + value.length <= out.length) out.set(value, received);
    else parts.push(value);
    received += value.length;
    onProgress?.(received, total);
  }
  if (out && received === out.length && !parts.length) return out;
  const merged = new Uint8Array(received);
  let offset = 0;
  if (out) {
    merged.set(out.subarray(0, Math.min(received, out.length)), 0);
    offset = Math.min(received, out.length);
  }
  for (const part of parts) {
    merged.set(part, offset);
    offset += part.length;
  }
  return merged;
}

export async function mount(container, ctx, roomId, query) {
  const detail = await api.room(roomId);
  if (ctx.alive && !ctx.alive()) return () => {};  // đã sang trang khác trong lúc chờ server
  const entry = detail.entry;
  if (!entry) {
    const wrap = document.createElement("div");
    wrap.innerHTML = `${ui.topbar([["Tổng quan", "#/"], [detail.name || roomId]])}<div class="page"><div class="notice">Dự án <strong>${fmt.escapeHtml(detail.name || roomId)}</strong> chưa dựng xong.
      <a href="#/room/${encodeURIComponent(roomId)}">Xem tiến độ</a>.</div></div>`;
    container.append(...wrap.children);
    return () => {};
  }
  const page = document.createElement("div");
  page.className = "viewer";
  container.append(page);
  document.body.classList.add("viewer-open");
  if (window.matchMedia?.("(max-width: 720px)").matches) document.body.classList.add("panel-hidden");
  const modesAllowed = new Set(entry.viewer_modes || ["orbit"]);
  const src = entry.source || {};
  const model = entry.model || {};
  const kindText = src.kind === "video" ? `Video${src.video_duration_s ? ` ${fmt.clock(src.video_duration_s)}` : ""}` : src.kind === "photos" ? "Ảnh chụp" : "Không rõ nguồn";
  const tool = (id, iconName, label, extra = "") => `<button type="button" class="icon-btn" id="${id}" aria-label="${label}" title="${label}" ${extra}>${ui.icon(iconName)}</button>`;
  const modeButton = (m) => `<button type="button" data-mode="${m}" aria-pressed="false" ${m === "orbit" || modesAllowed.has(m) ? "" : 'disabled title="Không có vị trí camera"'}>${ui.icon(m, 16)}<span class="lbl">${MODE_LABEL[m]}</span></button>`;
  page.innerHTML = `<header class="vbar">
      <a class="back" href="#/" aria-label="Về tổng quan">${ui.icon("chevLeft", 18)}<span class="lbl">Tổng quan</span></a>
      <span class="divider" aria-hidden="true"></span>
      <div class="title"><h1>${fmt.escapeHtml(detail.name)}</h1><span class="sub">${fmt.escapeHtml(kindText)}, ${fmt.escapeHtml(fmt.millions(entry.num_gaussians))} Gaussian</span></div>
      <div class="seg modes" role="group" aria-label="Chế độ xem">${["orbit", "walk", "tour"].map(modeButton).join("")}</div>
      <div class="tools" role="toolbar" aria-label="Công cụ">
        ${tool("tb-shot", "camera", "Chụp ảnh")}${tool("tb-rec", "record", "Quay video")}${tool("tb-full", "full", "Toàn màn hình (F)")}
        <span class="divider" aria-hidden="true"></span>${tool("tb-panel", "panel", "Bảng thông tin (I)", 'aria-pressed="true" aria-controls="vp-panel"')}
      </div></header>
    <div class="vbody">
      <section class="viewport" id="vp" aria-label="Khung xem 3D">
        <div class="overlay tl"><span class="chip float" id="vp-mode"></span><dl class="hintbox float" id="vp-hint"></dl></div>
        <div class="overlay tr"><span class="fps float" id="vp-fps">– FPS</span></div>
        <div class="overlay center" id="vp-loading"><div class="loading float"><span id="vp-loading-text">Đang tải mô hình…</span>
          <div class="progressbar"><span id="vp-loading-bar"></span></div></div></div>
        <div class="overlay bl hidden" id="vp-walkhelp"><div class="walkhelp float"><div class="keys" aria-hidden="true"><span></span><span>W</span><span></span><span>A</span><span>S</span><span>D</span></div>
          <ul><li>W A S D: đi tới, sang trái, lùi, sang phải</li><li>Chuột: nhìn quanh</li><li>Q / E: hạ thấp, nâng cao</li><li>Shift: đi nhanh</li></ul></div></div>
        <div class="overlay br hidden" id="vp-minimap"><div class="minimap float"><div>Bản đồ nhìn từ trên</div><canvas id="vp-minimap-canvas" width="400" height="300" role="img" aria-label="Bản đồ nhỏ: đường quay và vị trí hiện tại"></canvas><div>Nét đứt là đường đã quay</div></div></div>
        <div class="overlay bc" id="vp-bottom">
          <div class="camdock float" id="vp-toolbar" role="toolbar" aria-label="Điều khiển camera">
            ${tool("tb-reset", "reset", "Về góc ban đầu (R)")}
            <span class="sep" aria-hidden="true"></span>
            ${tool("tb-prev", "chevLeft", "Góc nhìn trước")}<span class="view-label" id="view-label">–</span>${tool("tb-next", "chevRight", "Góc nhìn sau")}
            <span class="sep orbit-only" aria-hidden="true"></span>
            <span class="orbit-only" style="display:contents">${tool("tb-zoom-out", "zoomOut", "Thu nhỏ")}${tool("tb-zoom-in", "zoomIn", "Phóng to")}${tool("tb-spin", "spin", "Tự xoay quanh", 'aria-pressed="false"')}</span>
          </div>
          <div class="tourbar float hidden" id="vp-tourbar">
            <button type="button" class="btn sm" id="tour-play">Tạm dừng</button>
            <label class="sr-only" for="tour-slider">Vị trí trên đường tham quan</label>
            <input type="range" id="tour-slider" min="0" max="1" step="0.001" value="0">
            <span class="time" id="tour-time">0:00 / 0:00</span>
            <div class="seg" role="group" aria-label="Tốc độ"><button type="button" data-speed="0.5" aria-pressed="false">0,5×</button><button type="button" data-speed="1" aria-pressed="true">1×</button><button type="button" data-speed="2" aria-pressed="false">2×</button></div>
            <label class="loop"><input type="checkbox" id="tour-loop" checked> Lặp lại</label>
          </div>
        </div>
      </section>
      <aside class="vpanel" id="vp-panel" aria-label="Thông tin và tùy chỉnh">
        <section><h2>Cảnh</h2><dl class="kv">
          <dt>Phương pháp</dt><dd>3DGS-MCMC (gsplat)</dd>
          <dt>Số Gaussian</dt><dd>${fmt.int(entry.num_gaussians)}</dd>
          <dt>File xem web</dt><dd>${fmt.bytes(entry.size_bytes)} (.splat)</dd>
          <dt>Thời gian train</dt><dd>${model.train_seconds ? fmt.duration(model.train_seconds) : "–"}</dd>
          <dt>Dựng xong</dt><dd>${fmt.escapeHtml(fmt.dateTime(model.trained_at || entry.built_at))}</dd></dl></section>
        <section><h2>Nguồn và camera</h2><dl class="kv">
          <dt>Nguồn</dt><dd>${fmt.escapeHtml(kindText)}</dd>
          <dt>Ảnh đầu vào</dt><dd>${fmt.int(src.input_images)}</dd>
          <dt>Ảnh định vị được</dt><dd>${fmt.int(src.registered_images)}</dd>
          <dt>Kiểm tra COLMAP</dt><dd>${fmt.escapeHtml(SFM_VERDICT[src.sfm_verdict] || src.sfm_verdict || "–")}</dd></dl></section>
        <section><h2>Góc nhìn</h2>
          <p class="note" id="views-note">Các vị trí camera lúc quay, lấy đều trên đường đi. Bấm để đưa camera về đó.</p>
          <div class="views" id="views" role="group" aria-label="Góc nhìn"></div></section>
        <section><h2>Hiển thị</h2>
          <div class="seg" role="group" aria-label="Chất lượng render" id="q-group">
            <button type="button" data-q="high" aria-pressed="true">Cao</button><button type="button" data-q="medium" aria-pressed="false">Vừa</button><button type="button" data-q="light" aria-pressed="false">Nhẹ</button></div>
          <p class="note" id="q-note"></p>
          <div class="slider"><div class="lbl"><label for="res-scale">Độ phân giải render</label><span id="res-value">100%</span></div>
            <input id="res-scale" type="range" min="50" max="100" step="5" value="100"></div>
          <label class="switch">Hiện vị trí camera lúc quay <input type="checkbox" id="show-cams" ${modesAllowed.has("walk") ? "" : "disabled"}></label>
          <label class="switch">Nền sáng <input type="checkbox" id="light-bg"></label>
          <div class="hidden" id="walk-settings">
            <div class="slider"><div class="lbl"><label for="walk-speed">Tốc độ đi</label><span id="walk-speed-v">1×</span></div>
              <input id="walk-speed" type="range" min="0.25" max="4" step="0.25" value="1"></div>
            <div class="slider"><div class="lbl"><label for="walk-height">Độ cao tầm mắt</label><span id="walk-height-v">theo camera lúc quay</span></div>
              <input id="walk-height" type="range" min="-100" max="100" step="5" value="0"></div>
            <label class="switch">Giữ trong vùng đã quay <input type="checkbox" id="walk-keep" checked></label>
            <p class="note">Mô hình không có tỉ lệ mét thật, nên tốc độ và độ cao tính theo kích thước cảnh. Ra xa vùng đã quay thì ảnh nhòe hoặc có mảng lơ lửng.</p>
          </div>
          <div class="hidden" id="tour-settings">
            <label class="switch">Luôn nhìn vào giữa phòng <input type="checkbox" id="tour-center"></label>
            <label class="switch">Làm mượt hướng nhìn <input type="checkbox" id="tour-smooth" checked></label>
          </div>
        </section>
        <section><h2>Chất lượng</h2><p class="note">Chưa chấm điểm: đo PSNR, SSIM, LPIPS đúng ở góc nhìn mới cần một lượt quay kiểm tra riêng, không dùng để train. Số đo của phương pháp trên benchmark nằm trong báo cáo thực nghiệm.</p>
          ${(entry.warnings || []).map((w) => `<p class="notice warn">${fmt.escapeHtml(w)}</p>`).join("")}</section>
        <section><h2>Tải xuống</h2><div class="row-dl" style="display:flex;flex-wrap:wrap;gap:8px">
          <a class="btn sm" href="/web/${encodeURI(entry.files.model)}?v=${encodeURIComponent(entry.version || "")}" download="${fmt.escapeHtml(roomId)}.splat">${ui.icon("download", 15)}File .splat</a>
          ${detail.has_ply ? `<a class="btn sm" href="/download/${encodeURIComponent(roomId)}/model.ply" download>${ui.icon("download", 15)}PLY gốc</a>` : ""}</div>
          <p class="note">PLY gốc giữ đủ màu theo góc nhìn (SH bậc 3), mở được bằng SuperSplat hoặc phần mềm 3DGS khác; .splat chỉ giữ màu cơ bản.</p></section>
        <section><h2>Hiệu năng</h2><button type="button" class="btn sm" id="perf-run">${ui.icon("gauge", 15)}Đo trên máy này (10 giây)</button>
          <pre class="perf hidden" id="perf-out"></pre><button type="button" class="btn sm hidden" id="perf-copy">Sao chép số liệu</button></section>
        <section><h2>Phím tắt</h2><dl class="shortcuts">
          <dt><kbd>1</kbd><kbd>2</kbd><kbd>3</kbd></dt><dd>Xoay quanh, đi bộ, tham quan</dd>
          <dt><kbd>R</kbd></dt><dd>Về góc ban đầu</dd><dt><kbd>F</kbd></dt><dd>Toàn màn hình</dd><dt><kbd>I</kbd></dt><dd>Ẩn hoặc hiện bảng này</dd>
          <dt><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd></dt><dd>Đi lại (chế độ đi bộ)</dd></dl></section>
      </aside>
    </div>`;

  const $ = (sel) => page.querySelector(sel);
  const viewport = $("#vp");
  const cleanups = [];
  const listen = (target, type, fn, opts) => {
    target.addEventListener(type, fn, opts);
    cleanups.push(() => target.removeEventListener(type, fn, opts));
  };

  // gợi ý thao tác: hiện khi đổi chế độ, mờ dần sau HINT_MS hoặc 3 giây sau lần thao tác đầu trên khung xem
  let hintTimer = null;
  const fadeHint = (ms) => {
    clearTimeout(hintTimer);
    hintTimer = setTimeout(() => $("#vp-hint").classList.add("faded"), ms);
  };
  function showHint(lines) {
    const box = $("#vp-hint");
    box.innerHTML = lines.map(([what, does]) => `<dt>${fmt.escapeHtml(what)}</dt><dd>${fmt.escapeHtml(does)}</dd>`).join("");
    box.classList.remove("faded");
    fadeHint(HINT_MS);
  }
  listen(viewport, "pointerdown", () => fadeHint(3000));
  listen(viewport, "wheel", () => fadeHint(3000), { passive: true });
  cleanups.push(() => clearTimeout(hintTimer));

  // ---------------------------------------------------------------------------------------- three + Spark
  const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false, powerPreference: "high-performance" });
  renderer.setClearColor(0x0a0c10, 1);
  viewport.prepend(renderer.domElement);
  renderer.domElement.tabIndex = 0;
  renderer.domElement.setAttribute("aria-label", `Mô hình 3D của ${detail.name}`);
  const scene = new THREE.Scene();
  const spark = new SparkRenderer({ renderer });
  scene.add(spark);
  const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
  let pixelScale = 1;

  function resize() {
    const w = Math.max(1, viewport.clientWidth);
    const h = Math.max(1, viewport.clientHeight);
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1) * pixelScale);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  const observer = new ResizeObserver(resize);
  observer.observe(viewport);
  cleanups.push(() => observer.disconnect());
  resize();

  // ------------------------------------------------------------------------------------- tải dữ liệu
  const abort = new AbortController();
  cleanups.push(() => abort.abort());
  let mesh = null;
  let orbit = null;
  // rời trang khi còn đang tải (main.js đã dựng trang khác, khung xem không còn trong trang): dọn ngay, không đặt lại
  // địa chỉ hay chế độ cho trang mới
  const gone = () => abort.signal.aborted || !page.isConnected || (ctx.alive ? !ctx.alive() : false);
  let disposed = false;
  function dispose() {
    if (disposed) return;
    disposed = true;
    if (orbit) orbit.dispose();
    for (const fn of cleanups.splice(0).reverse()) {
      try {
        fn();
      } catch (error) {
        console.error(error);
      }
    }
    if (mesh) {
      scene.remove(mesh);
      mesh.dispose?.();
    }
    renderer.dispose();
    renderer.forceContextLoss?.();
  }
  const loadStart = performance.now();
  let cameras = null;
  if (entry.files.cameras) {
    try {
      const response = await fetch(`/web/${encodeURI(entry.files.cameras)}?v=${encodeURIComponent(entry.version || "")}`, { credentials: "same-origin", signal: abort.signal });
      if (response.ok) cameras = await response.json();
    } catch (error) {
      if (error.name === "AbortError") {
        dispose();
        return () => {};
      }
    }
  }
  if (gone()) {
    dispose();
    return () => {};
  }
  const radius = cameras?.scene_radius || 1;
  camera.near = radius * 0.005;
  camera.far = radius * 200;
  if (cameras?.intrinsics?.fov_y_deg) camera.fov = cameras.intrinsics.fov_y_deg;
  camera.updateProjectionMatrix();
  if (cameras?.up) camera.up.copy(cam.vec(cameras.up)).normalize();

  let bytes;
  try {
    bytes = await fetchBytes(`/web/${encodeURI(entry.files.model)}?v=${encodeURIComponent(entry.version || "")}`, entry.size_bytes,
      (received, total) => {
        if (gone()) abort.abort();  // đã rời trang: thôi tải phần còn lại
        $("#vp-loading-bar").style.width = total ? `${((100 * received) / total).toFixed(1)}%` : "50%";
        $("#vp-loading-text").textContent = `Đang tải mô hình ${fmt.bytes(received)}${total ? ` / ${fmt.bytes(total)}` : ""}…`;
      }, abort.signal);
  } catch (error) {
    if (error.name === "AbortError" || gone()) {
      dispose();
      return () => {};
    }
    $("#vp-loading-text").textContent = error.message;
    return dispose;
  }
  if (gone()) {
    dispose();
    return () => {};
  }
  const downloadMs = performance.now() - loadStart;
  const totalSplats = Math.floor(bytes.length / 32);
  let decodeMs = null;
  let shownSplats = 0;

  async function showQuality(level) {
    const count = Math.max(1, Math.round(totalSplats * (QUALITY[level] ?? 1)));
    $("#vp-loading").classList.remove("hidden");
    $("#vp-loading-text").textContent = "Đang dựng mô hình trên GPU…";
    $("#vp-loading-bar").style.width = "100%";
    const started = performance.now();
    // bản sao: Spark có thể chuyển (transfer) bộ đệm sang worker, bản gốc giữ lại để đổi mức chất lượng
    const next = new SplatMesh({ fileBytes: bytes.slice(0, count * 32), fileType: "splat" });
    await next.initialized;
    if (gone()) {
      next.dispose?.();
      return;
    }
    if (mesh) {
      scene.remove(mesh);
      mesh.dispose?.();
    }
    mesh = next;
    scene.add(mesh);
    shownSplats = count;
    decodeMs ??= performance.now() - started;
    $("#vp-loading").classList.add("hidden");
    $("#q-note").textContent = `Đang vẽ ${fmt.int(count)} / ${fmt.int(totalSplats)} Gaussian${count < totalSplats ? " (bỏ các Gaussian nhỏ, mờ nhất trước)" : ""}.`;
    for (const button of page.querySelectorAll("[data-q]")) button.setAttribute("aria-pressed", String(button.dataset.q === level));
  }
  await showQuality("high");
  if (gone()) {
    dispose();
    return () => {};
  }
  const firstFrameMs = performance.now() - loadStart;

  // ------------------------------------------------------------------------- camera ban đầu, đường quay
  const center = cameras?.center ? cam.vec(cameras.center) : new THREE.Vector3();
  if (!cameras) {
    try {
      mesh.getBoundingBox(true).getCenter(center);
    } catch {
      /* chưa có hộp bao: giữ gốc tọa độ */
    }
  }
  const startIndex = cameras ? Math.min(Math.max(0, cameras.default_view ?? 0), cameras.frames.length - 1) : 0;
  // góc nhìn để chọn nhanh: các khung lấy đều trên đường quay, luôn có góc ban đầu
  const views = [];
  if (cameras?.frames?.length) {
    const n = cameras.frames.length;
    const count = Math.min(n, MAX_VIEWS);
    for (let k = 0; k < count; k++) views.push(Math.round((k * (n - 1)) / Math.max(1, count - 1)));
    if (!views.includes(startIndex)) {
      // thay góc lấy đều gần nhất bằng góc ban đầu: vẫn đúng MAX_VIEWS ô, không thêm ô lẻ
      let near = 0;
      for (let k = 1; k < views.length; k++) if (Math.abs(views[k] - startIndex) < Math.abs(views[near] - startIndex)) near = k;
      views[near] = startIndex;
      views.sort((a, b) => a - b);
    }
  }
  let viewPos = 0;

  const helpers = new THREE.Group();
  helpers.visible = false;
  scene.add(helpers);
  let marker = null;
  let frusta = null;
  if (cameras?.frames?.length) {
    const aspect = cameras.intrinsics ? cameras.intrinsics.width / cameras.intrinsics.height : 4 / 3;
    const h = Math.tan(THREE.MathUtils.degToRad((cameras.intrinsics?.fov_y_deg || 50) / 2));
    const size = radius * 0.06;
    const step = Math.max(1, Math.ceil(cameras.frames.length / 200));
    const pts = [];
    for (let i = 0; i < cameras.frames.length; i += step) {
      const f = cam.frameOf(cameras, i);
      const right = new THREE.Vector3().crossVectors(f.forward, f.up).normalize();
      const c = f.position.clone().add(f.forward.clone().multiplyScalar(size));
      const dx = right.clone().multiplyScalar(size * h * aspect);
      const dy = f.up.clone().multiplyScalar(size * h);
      const corners = [c.clone().add(dx).add(dy), c.clone().sub(dx).add(dy), c.clone().sub(dx).sub(dy), c.clone().add(dx).sub(dy)];
      for (const corner of corners) pts.push(f.position, corner);
      for (let k = 0; k < 4; k++) pts.push(corners[k], corners[(k + 1) % 4]);
    }
    frusta = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0x4c9aff, depthTest: false, transparent: true, opacity: 0.85 }));
    const path = new THREE.Line(new THREE.BufferGeometry().setFromPoints(cameras.frames.map((f) => cam.vec(f.position))),
      new THREE.LineBasicMaterial({ color: 0xf5a524, depthTest: false, transparent: true, opacity: 0.9 }));
    frusta.renderOrder = path.renderOrder = 10;
    helpers.add(frusta, path);
    marker = new THREE.Mesh(new THREE.SphereGeometry(radius * 0.02, 12, 8), new THREE.MeshBasicMaterial({ color: 0xff5470, depthTest: false }));
    marker.renderOrder = 11;
    marker.visible = false;
    scene.add(marker);
  }

  // ------------------------------------------------------------------------------------- chế độ xem
  let mode = null;
  const walk = { yaw: 0, pitch: 0, keys: new Set(), base: null, height: 0, dragging: false, last: null };
  const tour = { t: 0, playing: true, speed: 1, frames: null, smooth: null, curve: null, look: new THREE.Euler(), drag: null };
  const ground = cameras?.frames?.length ? cam.groundBasis(cameras) : null;
  const box = cameras?.frames?.length ? cam.walkBox(cameras) : null;

  function orbitFromFrame(index) {
    if (cameras?.frames?.length) {
      const f = cam.placeAtFrame(camera, cameras, index);
      const toCenter = center.clone().sub(f.position).dot(f.forward);
      const d = THREE.MathUtils.clamp(toCenter, radius * 0.2, radius * 2);
      orbit.target.copy(f.position).add(f.forward.clone().multiplyScalar(d));
    } else {
      camera.position.copy(center).add(new THREE.Vector3(0, 0, radius * 2));
      orbit.target.copy(center);
    }
    orbit.update();
  }

  function setMode(next) {
    if (next !== "orbit" && !modesAllowed.has(next)) next = "orbit";
    if (mode === next) return;
    if (orbit) {
      orbit.dispose();
      orbit = null;
    }
    if (document.pointerLockElement === renderer.domElement) document.exitPointerLock();
    mode = next;
    for (const button of page.querySelectorAll("[data-mode]")) button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
    $("#vp-mode").innerHTML = `${ui.icon(mode, 15)}${MODE_LABEL[mode]}`;
    showHint(HINTS[mode]);
    for (const el of page.querySelectorAll(".orbit-only")) el.classList.toggle("hidden", mode !== "orbit");
    setSpin(false);
    $("#vp-walkhelp").classList.toggle("hidden", mode !== "walk");
    $("#vp-minimap").classList.toggle("hidden", mode !== "walk" || !ground);
    $("#walk-settings").classList.toggle("hidden", mode !== "walk");
    $("#tour-settings").classList.toggle("hidden", mode !== "tour");
    $("#vp-tourbar").classList.toggle("hidden", mode !== "tour");
    $("#vp-toolbar").classList.toggle("hidden", mode === "tour");
    if (marker) marker.visible = false;
    if (frusta) frusta.visible = mode !== "tour";  // tham quan: camera ở sát người xem che khung nhìn, chỉ vẽ đường đi
    if (mode === "orbit") {
      if (cameras?.up) camera.up.copy(cam.vec(cameras.up)).normalize();
      orbit = new OrbitControls(camera, renderer.domElement);  // dùng camera.up lúc tạo làm trục xoay
      orbit.enableDamping = true;
      orbit.dampingFactor = 0.08;
      orbit.minDistance = radius * 0.03;
      orbit.maxDistance = radius * 6;
      orbitFromFrame(startIndex);
    } else if (mode === "walk") {
      startWalk(startIndex);
    } else if (mode === "tour") {
      startTour();
    }
    // ghi chế độ vào địa chỉ (gửi link mở đúng chế độ); replaceState không gây hashchange nên không dựng lại trang
    history.replaceState(null, "", `#/view/${encodeURIComponent(roomId)}${mode === "orbit" ? "" : `?mode=${mode}`}`);
  }

  // ---------------------------------------------------------------------------------------- đi bộ
  function startWalk(index) {
    const f = cam.frameOf(cameras, index);
    const up = ground.up;
    const flat = f.forward.clone().sub(up.clone().multiplyScalar(f.forward.dot(up)));
    walk.base = flat.lengthSq() > 1e-6 ? flat.normalize() : ground.e1.clone();
    walk.yaw = 0;
    walk.pitch = Math.asin(THREE.MathUtils.clamp(f.forward.dot(up), -0.99, 0.99));
    walk.height = 0;
    $("#walk-height").value = "0";
    $("#walk-height-v").textContent = "theo camera lúc quay";
    camera.position.copy(f.position);
    walk.eye = f.position.dot(up);
    camera.up.copy(up);
    applyWalk();
  }

  function walkVectors() {
    const up = ground.up;
    const heading = walk.base.clone().applyAxisAngle(up, walk.yaw);
    const right = new THREE.Vector3().crossVectors(heading, up).normalize();
    const look = heading.clone().multiplyScalar(Math.cos(walk.pitch)).add(up.clone().multiplyScalar(Math.sin(walk.pitch)));
    return { up, heading, right, look };
  }

  function applyWalk() {
    const { look } = walkVectors();
    camera.lookAt(camera.position.clone().add(look));
  }

  function stepWalk(dt) {
    const { up, heading, right } = walkVectors();
    const speed = radius * 0.35 * Number($("#walk-speed").value) * (walk.keys.has("ShiftLeft") || walk.keys.has("ShiftRight") ? 3 : 1);
    const move = new THREE.Vector3();
    if (walk.keys.has("KeyW") || walk.keys.has("ArrowUp")) move.add(heading);
    if (walk.keys.has("KeyS") || walk.keys.has("ArrowDown")) move.sub(heading);
    if (walk.keys.has("KeyD") || walk.keys.has("ArrowRight")) move.add(right);
    if (walk.keys.has("KeyA") || walk.keys.has("ArrowLeft")) move.sub(right);
    if (walk.keys.has("KeyE")) move.add(up);
    if (walk.keys.has("KeyQ")) move.sub(up);
    if (move.lengthSq() > 0) camera.position.add(move.normalize().multiplyScalar(speed * dt));
    // độ cao tầm mắt: thanh trượt dời theo `up` so với độ cao khi bắt đầu
    const targetEye = walk.eye + (Number($("#walk-height").value) / 100) * radius * 0.5;
    if (!walk.keys.has("KeyE") && !walk.keys.has("KeyQ")) {
      const current = camera.position.dot(up);
      camera.position.add(up.clone().multiplyScalar((targetEye - current) * Math.min(1, dt * 6)));
    } else {
      walk.eye = camera.position.dot(up) - (Number($("#walk-height").value) / 100) * radius * 0.5;
    }
    if ($("#walk-keep").checked && box) box.clampPoint(camera.position, camera.position);
    applyWalk();
  }

  function drawMinimap() {
    const canvas = $("#vp-minimap-canvas");
    const g = canvas.getContext("2d");
    const { e1, e2, mean } = ground;
    const pts = cameras.frames.map((f) => {
      const d = cam.vec(f.position).sub(mean);
      return [d.dot(e1), d.dot(e2)];
    });
    const here = camera.position.clone().sub(mean);
    const span = Math.max(1e-6, ...pts.map(([x, y]) => Math.max(Math.abs(x), Math.abs(y))), Math.abs(here.dot(e1)), Math.abs(here.dot(e2))) * 1.15;
    const sx = (x) => canvas.width / 2 + (x / span) * (canvas.width / 2 - 10);
    const sy = (y) => canvas.height / 2 - (y / span) * (canvas.height / 2 - 10);
    g.clearRect(0, 0, canvas.width, canvas.height);
    g.strokeStyle = "#2A2F3A";
    g.lineWidth = 3;
    g.strokeRect(4, 4, canvas.width - 8, canvas.height - 8);
    g.setLineDash([8, 10]);
    g.strokeStyle = "#5B6474";
    g.lineWidth = 4;
    g.beginPath();
    pts.forEach(([x, y], i) => (i ? g.lineTo(sx(x), sy(y)) : g.moveTo(sx(x), sy(y))));
    g.stroke();
    g.setLineDash([]);
    const { look } = walkVectors();
    const px = sx(here.dot(e1));
    const py = sy(here.dot(e2));
    const angle = Math.atan2(-look.dot(e2), look.dot(e1));
    g.fillStyle = "rgba(76,154,255,0.25)";
    g.beginPath();
    g.moveTo(px, py);
    g.arc(px, py, 46, angle - 0.6, angle + 0.6);
    g.closePath();
    g.fill();
    g.fillStyle = "#4C9AFF";
    g.beginPath();
    g.arc(px, py, 9, 0, Math.PI * 2);
    g.fill();
  }

  // ----------------------------------------------------------------------------------------- tham quan
  function startTour() {
    tour.frames = cameras.frames.map((_, i) => cam.frameOf(cameras, i));
    tour.smooth = cam.smoothedFrames(cameras, 7);
    tour.curve = tour.frames.length > 1
      ? new THREE.CatmullRomCurve3(tour.frames.map((f) => f.position), false, "centripetal") : null;
    tour.duration = Math.max(20, tour.frames.length / 2);  // giây ở tốc độ 1×
    tour.playing = true;
    tour.look.set(0, 0, 0);
    $("#tour-play").textContent = "Tạm dừng";
    applyTour();
  }

  function tourPose(u) {
    const n = tour.frames.length;
    const x = u * (n - 1);
    const i = Math.min(n - 2, Math.max(0, Math.floor(x)));
    const a = n > 1 ? x - i : 0;
    const src = $("#tour-smooth").checked ? tour.smooth : tour.frames;
    const position = tour.curve ? tour.curve.getPoint(u) : tour.frames[0].position.clone();
    let forward;
    let up;
    if (n === 1) {
      forward = src[0].forward.clone();
      up = src[0].up.clone();
    } else {
      forward = src[i].forward.clone().lerp(src[i + 1].forward, a).normalize();
      up = src[i].up.clone().lerp(src[i + 1].up, a).normalize();
    }
    if ($("#tour-center").checked) {
      forward = center.clone().sub(position).normalize();
      up = cam.vec(cameras.up).normalize();
    }
    return { position, forward, up, index: Math.round(x) };
  }

  function applyTour() {
    const pose = tourPose(tour.t);
    camera.position.copy(pose.position);
    const q = cam.frameQuaternion(pose.forward, pose.up);
    const look = new THREE.Quaternion().setFromEuler(tour.look);
    camera.quaternion.copy(q.multiply(look));
    $("#tour-slider").value = String(tour.t);
    $("#tour-time").textContent = `${fmt.clock(tour.t * tour.duration / tour.speed)} / ${fmt.clock(tour.duration / tour.speed)} · khung ${pose.index + 1}/${tour.frames.length}`;
    if (marker && helpers.visible) {
      marker.visible = true;
      marker.position.copy(pose.position);
    }
  }

  function stepTour(dt) {
    if (tour.playing && !tour.drag) {
      tour.t += (dt * tour.speed) / tour.duration;
      if (tour.t >= 1) {
        if ($("#tour-loop").checked) tour.t -= 1;
        else {
          tour.t = 1;
          tour.playing = false;
          $("#tour-play").textContent = "Chạy tiếp";
        }
      }
    }
    if (!tour.drag) {  // thả chuột: hướng nhìn trở về theo đường quay
      tour.look.x *= Math.max(0, 1 - dt * 2.5);
      tour.look.y *= Math.max(0, 1 - dt * 2.5);
    }
    applyTour();
  }

  // ------------------------------------------------------------------------------------- vòng vẽ, FPS
  let frames = 0;
  let fpsStart = performance.now();
  let fps = null;
  let last = performance.now();
  let screenshotPending = false;
  const perfSamples = [];
  let perfUntil = 0;

  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (mode === "orbit" && orbit) orbit.update();
    else if (mode === "walk") {
      stepWalk(dt);
      if (ground) drawMinimap();
    } else if (mode === "tour") stepTour(dt);
    renderer.render(scene, camera);
    if (screenshotPending) {
      screenshotPending = false;
      renderer.domElement.toBlob((blob) => blob && download(blob, `${roomId}_${stamp()}.png`), "image/png");
    }
    frames += 1;
    if (now - fpsStart >= 1000) {
      fps = (frames * 1000) / (now - fpsStart);
      frames = 0;
      fpsStart = now;
      $("#vp-fps").textContent = fps < 1 ? "<1 FPS" : `${fmt.decimal(fps, 0)} FPS`;
      if (now < perfUntil) perfSamples.push(fps);
    }
  });
  cleanups.push(() => renderer.setAnimationLoop(null));

  // ------------------------------------------------------------------------------------------ sự kiện
  for (const button of page.querySelectorAll("[data-mode]")) listen(button, "click", () => setMode(button.dataset.mode));
  for (const button of page.querySelectorAll("[data-q]")) listen(button, "click", () => showQuality(button.dataset.q));
  listen($("#res-scale"), "input", (event) => {
    pixelScale = Number(event.target.value) / 100;
    $("#res-value").textContent = `${event.target.value}%`;
    resize();
  });
  listen($("#show-cams"), "change", (event) => {
    helpers.visible = event.target.checked;
    if (marker) marker.visible = event.target.checked && mode === "tour";
  });
  listen($("#light-bg"), "change", (event) => {
    renderer.setClearColor(event.target.checked ? 0xe9ecf1 : 0x0a0c10, 1);
    viewport.classList.toggle("light", event.target.checked);
  });
  listen($("#walk-speed"), "input", (event) => {
    $("#walk-speed-v").textContent = `${fmt.decimal(Number(event.target.value), 2).replace(/,?0+$/, "") || "0"}×`;
  });
  listen($("#walk-height"), "input", (event) => {
    const v = Number(event.target.value);
    $("#walk-height-v").textContent = v === 0 ? "theo camera lúc quay" : `${v > 0 ? "cao" : "thấp"} hơn ${Math.abs(v)}%`;
  });
  listen($("#tb-reset"), "click", () => reset());
  listen($("#tb-shot"), "click", () => {
    screenshotPending = true;
  });
  listen($("#tb-full"), "click", () => (document.fullscreenElement ? document.exitFullscreen() : viewport.requestFullscreen?.()));

  function reset() {
    viewPos = nearestView(startIndex);
    paintViews();
    if (mode === "orbit" && orbit) orbitFromFrame(startIndex);
    else if (mode === "walk") startWalk(startIndex);
    else if (mode === "tour") {
      tour.t = 0;
      tour.look.set(0, 0, 0);
    }
  }

  // quay video khung xem (MediaRecorder, WebM)
  let recorder = null;
  listen($("#tb-rec"), "click", () => {
    const button = $("#tb-rec");
    if (recorder) {
      recorder.stop();
      return;
    }
    if (!renderer.domElement.captureStream || typeof MediaRecorder === "undefined") {
      showHint([["Quay video", "trình duyệt này không hỗ trợ quay khung xem"]]);
      return;
    }
    const chunks = [];
    const type = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t));
    recorder = new MediaRecorder(renderer.domElement.captureStream(RECORD_FPS), type ? { mimeType: type, videoBitsPerSecond: 8e6 } : undefined);
    recorder.ondataavailable = (event) => event.data.size && chunks.push(event.data);
    recorder.onstop = () => {
      download(new Blob(chunks, { type: "video/webm" }), `${roomId}_${stamp()}.webm`);
      recorder = null;
      button.classList.remove("rec");
      button.setAttribute("aria-label", "Quay video");
      button.title = "Quay video";
    };
    recorder.start(1000);
    button.classList.add("rec");
    button.setAttribute("aria-label", "Dừng quay");
    button.title = "Dừng quay";
  });
  cleanups.push(() => recorder?.state === "recording" && recorder.stop());

  // đi bộ: khóa chuột hoặc kéo chuột để nhìn; phím di chuyển
  const canvas = renderer.domElement;
  listen(canvas, "click", () => {
    if (mode === "walk" && document.pointerLockElement !== canvas) canvas.requestPointerLock?.();
  });
  function look(dx, dy) {
    if (mode === "walk") {
      walk.yaw -= dx * 0.0025;
      walk.pitch = THREE.MathUtils.clamp(walk.pitch - dy * 0.0025, -1.45, 1.45);
    } else if (mode === "tour") {
      tour.look.y = THREE.MathUtils.clamp(tour.look.y - dx * 0.003, -1.2, 1.2);
      tour.look.x = THREE.MathUtils.clamp(tour.look.x - dy * 0.003, -0.9, 0.9);
    }
  }
  listen(document, "mousemove", (event) => {
    if (document.pointerLockElement === canvas) look(event.movementX, event.movementY);
  });
  listen(canvas, "pointerdown", (event) => {
    if (mode === "tour" || (mode === "walk" && document.pointerLockElement !== canvas)) {
      walk.last = [event.clientX, event.clientY];
      tour.drag = mode === "tour";
      canvas.setPointerCapture?.(event.pointerId);
    }
  });
  listen(canvas, "pointermove", (event) => {
    if (!walk.last || document.pointerLockElement === canvas) return;
    look(event.clientX - walk.last[0], event.clientY - walk.last[1]);
    walk.last = [event.clientX, event.clientY];
  });
  const endDrag = () => {
    walk.last = null;
    tour.drag = null;
  };
  listen(canvas, "pointerup", endDrag);
  listen(canvas, "pointercancel", endDrag);
  listen(window, "keydown", (event) => {
    if (event.target.closest?.("input, textarea, select") && event.target.type !== "range" && event.target.type !== "checkbox") return;
    if (mode === "walk") {
      walk.keys.add(event.code);
      if (event.code.startsWith("Arrow") || event.code === "Space") event.preventDefault();
    }
    if (event.code === "KeyR" && !event.ctrlKey && !event.metaKey) reset();
    if (event.code === "KeyF" && !event.ctrlKey && !event.metaKey) $("#tb-full").click();
    if (event.code === "KeyI" && !event.ctrlKey && !event.metaKey) togglePanel();
    if (event.code === "Digit1") setMode("orbit");
    if (event.code === "Digit2") setMode("walk");
    if (event.code === "Digit3") setMode("tour");
  });
  listen(window, "keyup", (event) => walk.keys.delete(event.code));
  listen(window, "blur", () => walk.keys.clear());

  // tham quan: chạy / dừng, tua, tốc độ
  listen($("#tour-play"), "click", () => {
    if (tour.t >= 1) tour.t = 0;
    tour.playing = !tour.playing;
    $("#tour-play").textContent = tour.playing ? "Tạm dừng" : "Chạy tiếp";
  });
  listen($("#tour-slider"), "input", (event) => {
    tour.t = Number(event.target.value);
    tour.playing = false;
    $("#tour-play").textContent = "Chạy tiếp";
  });
  for (const button of page.querySelectorAll("[data-speed]")) {
    listen(button, "click", () => {
      tour.speed = Number(button.dataset.speed);
      for (const other of page.querySelectorAll("[data-speed]")) other.setAttribute("aria-pressed", String(other === button));
    });
  }

  // ---------------------------------------------------------------- góc nhìn, phóng to, tự xoay, bảng thông tin
  function nearestView(index) {
    if (!views.length) return 0;
    let best = 0;
    views.forEach((v, k) => {
      if (Math.abs(v - index) < Math.abs(views[best] - index)) best = k;
    });
    return best;
  }

  function paintViews() {
    $("#view-label").textContent = views.length ? `Góc ${viewPos + 1}/${views.length}` : "–";
    for (const button of page.querySelectorAll("[data-view]")) button.setAttribute("aria-pressed", String(Number(button.dataset.view) === viewPos));
  }

  function goView(k) {
    if (!views.length) return;
    viewPos = (k + views.length) % views.length;
    const index = views[viewPos];
    if (mode === "tour") {
      tour.t = index / Math.max(1, cameras.frames.length - 1);
      tour.look.set(0, 0, 0);
    } else if (mode === "walk") {
      startWalk(index);
    } else {
      if (!orbit) setMode("orbit");
      setSpin(false);
      orbitFromFrame(index);
    }
    paintViews();
  }

  function dolly(factor) {
    if (mode !== "orbit" || !orbit) return;
    const offset = camera.position.clone().sub(orbit.target);
    const distance = THREE.MathUtils.clamp(offset.length() * factor, orbit.minDistance, orbit.maxDistance);
    camera.position.copy(orbit.target).add(offset.setLength(distance));
    orbit.update();
  }

  function setSpin(on) {
    if (orbit) {
      orbit.autoRotate = on;
      orbit.autoRotateSpeed = 0.8;
    }
    $("#tb-spin").setAttribute("aria-pressed", String(Boolean(on && orbit)));
  }

  function togglePanel(show = document.body.classList.contains("panel-hidden")) {
    document.body.classList.toggle("panel-hidden", !show);
    $("#tb-panel").setAttribute("aria-pressed", String(show));
  }

  $("#views").innerHTML = views.map((index, k) => `<button type="button" data-view="${k}" aria-pressed="false" title="Khung ${index + 1} của đường quay">${k + 1}</button>`).join("");
  if (!views.length) {
    $("#views-note").textContent = "Mô hình này không có vị trí camera lúc quay, nên chỉ xoay quanh được.";
    for (const id of ["#tb-prev", "#tb-next"]) $(id).disabled = true;
  }
  for (const button of page.querySelectorAll("[data-view]")) listen(button, "click", () => goView(Number(button.dataset.view)));
  listen($("#tb-prev"), "click", () => goView(viewPos - 1));
  listen($("#tb-next"), "click", () => goView(viewPos + 1));
  listen($("#tb-zoom-in"), "click", () => dolly(0.8));
  listen($("#tb-zoom-out"), "click", () => dolly(1.25));
  listen($("#tb-spin"), "click", () => setSpin(!(orbit && orbit.autoRotate)));
  listen($("#tb-panel"), "click", () => togglePanel());
  $("#tb-panel").setAttribute("aria-pressed", String(!document.body.classList.contains("panel-hidden")));
  cleanups.push(() => document.body.classList.remove("panel-hidden"));
  viewPos = nearestView(startIndex);
  paintViews();

  // đo hiệu năng trên máy đang xem (cho báo cáo: thời gian tải, FPS)
  listen($("#perf-run"), "click", () => {
    const button = $("#perf-run");
    button.disabled = true;
    perfSamples.length = 0;
    perfUntil = performance.now() + 10500;
    if (mode === "orbit" && orbit) setSpin(true);
    setTimeout(() => {
      setSpin(false);
      button.disabled = false;
      const gl = renderer.getContext();
      const info = gl.getExtension("WEBGL_debug_renderer_info");
      const avg = perfSamples.length ? perfSamples.reduce((a, b) => a + b, 0) / perfSamples.length : null;
      const result = {
        room: roomId, measured_at: new Date().toISOString(), mode,
        splats_drawn: shownSplats, splats_total: totalSplats, file_bytes: bytes.length,
        download_ms: Math.round(downloadMs), decode_ms: decodeMs == null ? null : Math.round(decodeMs), first_frame_ms: Math.round(firstFrameMs),
        fps_mean: avg == null ? null : Number(avg.toFixed(1)), fps_min: perfSamples.length ? Number(Math.min(...perfSamples).toFixed(1)) : null,
        canvas_px: [renderer.domElement.width, renderer.domElement.height], pixel_ratio: renderer.getPixelRatio(),
        gpu: info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : null, user_agent: navigator.userAgent,
      };
      const out = $("#perf-out");
      out.textContent = JSON.stringify(result, null, 2);
      out.classList.remove("hidden");
      $("#perf-copy").classList.remove("hidden");
    }, 10600);
  });
  listen($("#perf-copy"), "click", () => navigator.clipboard?.writeText($("#perf-out").textContent));

  // dùng cho kiểm thử tự động (tests/browser): đặt camera đúng một khung ảnh rồi vẽ ngay
  window.__sr3dViewer = {
    ready: true, totalSplats, roomId,
    setMode,
    viewFrame(index) {  // camera đứng yên đúng như khung ảnh (không điều khiển nào ghi đè)
      setMode("orbit");
      orbit.dispose();
      orbit = null;
      mode = "still";
      cam.placeAtFrame(camera, cameras, index);
      renderer.render(scene, camera);
      return { width: renderer.domElement.width, height: renderer.domElement.height, fov: camera.fov };
    },
    capture() {  // vẽ một khung rồi lấy ảnh ngay trong cùng tác vụ (bộ đệm WebGL còn nguyên)
      renderer.render(scene, camera);
      return renderer.domElement.toDataURL("image/png");
    },
    cameraState: () => ({ position: camera.position.toArray(), quaternion: camera.quaternion.toArray(), fov: camera.fov, mode }),
    fps: () => fps,
  };
  cleanups.push(() => delete window.__sr3dViewer);

  setMode(query?.get("mode") || "orbit");
  return dispose;
}
