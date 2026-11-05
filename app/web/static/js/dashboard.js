// Tổng quan: cảnh 3D mới nhất (lớn), tạo dự án (kéo thả), dự án chưa xong, các cảnh khác, lịch sử xử lý.
// Dữ liệu: /api/rooms (rooms.py: manifest.json + progress.json + trạng thái việc nền). Không có số liệu tự đặt.
import * as api from "./api.js";
import * as fmt from "./fmt.js";
import * as ui from "./ui.js";
import { filesFromDrop } from "./upload.js";

const ACTIVE = new Set(["queued", "running"]);

const thumbUrl = (entry) => `/web/${encodeURI(entry.files.thumbnail)}?v=${encodeURIComponent(entry.version || "")}`;

function sourceLine(entry) {
  const src = entry.source || {};
  if (src.kind === "video") return src.video_duration_s ? `Video ${fmt.clock(src.video_duration_s)}` : "Video";
  if (src.kind === "photos") return src.input_images ? `${fmt.int(src.input_images)} ảnh chụp` : "Ảnh chụp";
  return "Không rõ nguồn";
}

function builtAt(entry) {
  return entry.model?.trained_at || entry.built_at;
}

// ------------------------------------------------------------------------------------------- các khối
function featureBlock(room) {
  const e = room.entry;
  const src = e.source || {};
  const facts = [
    [fmt.millions(e.num_gaussians), "Gaussian"],
    [fmt.bytes(e.size_bytes), "file xem web"],
    src.registered_images != null ? [fmt.int(src.registered_images), "ảnh định vị được"] : null,
    e.model?.train_seconds ? [fmt.duration(e.model.train_seconds), "train"] : null,
  ].filter(Boolean).map(([value, label]) => `<span><b>${fmt.escapeHtml(value)}</b>${fmt.escapeHtml(label)}</span>`).join("");
  const media = e.files?.thumbnail ? `<img src="${thumbUrl(e)}" alt="">` : ui.splatArt("art");
  return `<article class="feature${e.files?.thumbnail ? "" : " empty-art"}" data-room="${fmt.escapeHtml(room.id)}" aria-labelledby="feat-title">
    ${media}
    <div class="info">
      <div>
        <div class="kicker">${ui.badge("done")}<span>${fmt.escapeHtml(sourceLine(e))}, dựng ${fmt.escapeHtml(fmt.ago(builtAt(e)))}</span></div>
        <h2 id="feat-title">${fmt.escapeHtml(room.name)}</h2>
        <div class="facts">${facts}</div>
      </div>
      <div class="cta">
        <a class="btn primary lg" href="#/view/${encodeURIComponent(room.id)}">${ui.icon("cube", 17)}Mở trình xem 3D</a>
        <a class="btn lg" href="#/room/${encodeURIComponent(room.id)}">Chi tiết</a>
      </div>
    </div></article>`;
}

function emptyFeature(viewOnly) {
  return `<article class="feature empty-art" aria-labelledby="feat-title">${ui.splatArt("art")}
    <h2 id="feat-title">Dựng căn phòng đầu tiên thành không gian 3D</h2>
    <p>Quay một video đi quanh phòng hoặc chụp 30–100 ảnh. Pod GPU tách khung, ước lượng camera bằng COLMAP, huấn luyện 3DGS-MCMC rồi đưa cảnh vào đây để xem.</p>
    ${viewOnly ? "" : `<div class="cta"><a class="btn primary lg" href="#/new">${ui.icon("plus", 17)}Tạo dự án</a></div>`}</article>`;
}

function createBlock() {
  return `<section class="create" aria-labelledby="create-title">
    <h2 id="create-title">Tạo dự án mới</h2>
    <p>Kéo video hoặc ảnh chụp căn phòng vào đây để bắt đầu.</p>
    <a class="drop" id="dash-drop" href="#/new">${ui.icon("upload", 26, 1.6)}
      <strong>Thả video hoặc ảnh vào đây</strong><span>hoặc bấm để chọn file</span></a>
    <ol class="flow" aria-label="Các bước"><li><span>Tải video hoặc ảnh lên</span></li><li><span>Dựng <span class="nowrap">3DGS-MCMC</span> trên GPU</span></li><li><span>Xem và đi trong cảnh 3D</span></li></ol>
  </section>`;
}

function pendingRow(room) {
  const b = room.build || room.last_build || {};
  const status = room.build ? b.status : room.status;
  const active = ACTIVE.has(status);
  const pct = active ? ui.summaryPercent(b) : null;
  const what = status === "failed" || status === "stopped" || status === "lost"
    ? (b.error || (status === "lost" ? "Việc dựng không còn chạy (pod tắt hoặc khởi động lại giữa chừng)" : b.message || ""))
    : [b.stage_label, b.message && b.message !== b.stage_label ? b.message : ""].filter(Boolean).join(": ") || "Đang chờ GPU";
  const bar = active
    ? `<div class="progressbar${pct == null || status === "queued" ? " indeterminate" : ""}" aria-hidden="true"><span style="width:${pct ?? 30}%"></span></div>` : "";
  return `<div class="proc" data-room="${fmt.escapeHtml(room.id)}">
    <div class="who"><b>${fmt.escapeHtml(room.name)}</b><span>${ui.badge(status)}</span></div>
    <div class="what"><div class="line"><span>${fmt.escapeHtml(what)}</span>${pct != null && status === "running" ? `<span class="num" title="Tiến độ tổng, ước lượng theo bước">${fmt.decimal(pct, 0)} % tổng</span>` : ""}</div>${bar}
      <div class="line"><span class="muted">${ui.sourceKind(b.source_kind)}, bắt đầu ${fmt.escapeHtml(fmt.ago(b.started_at || b.created_at))}</span></div></div>
    <a class="btn sm" href="#/room/${encodeURIComponent(room.id)}">${active ? "Xem tiến độ" : "Xem chi tiết"}</a></div>`;
}

function sceneCard(room) {
  const e = room.entry;
  const media = e.files?.thumbnail ? `<img src="${thumbUrl(e)}" alt="" loading="lazy">` : ui.splatArt("placeholder");
  const rebuilding = room.build ? ui.badge(room.build.status) : "";
  return `<article class="card" data-room="${fmt.escapeHtml(room.id)}">
    <div class="media">${media}${rebuilding}</div>
    <div class="body"><h3>${fmt.escapeHtml(room.name)}</h3>
      <p class="meta">${fmt.escapeHtml(sourceLine(e))}, ${fmt.escapeHtml(fmt.millions(e.num_gaussians))} Gaussian</p>
      <p class="meta">Dựng ${fmt.escapeHtml(fmt.dateTime(builtAt(e)))}</p></div>
    <a class="cover" href="#/view/${encodeURIComponent(room.id)}" aria-label="Mở ${fmt.escapeHtml(room.name)} trong trình xem 3D"></a></article>`;
}

// lần cập nhật cuối: bản dựng gần nhất, không có thì lúc mô hình dựng xong (phòng có sẵn trong thư viện)
function lastActivity(room) {
  const b = room.last_build || room.build;
  return b?.updated_at || b?.finished_at || (room.entry ? builtAt(room.entry) : "") || "";
}

function historyRow(room) {
  const b = room.last_build || room.build;
  const status = room.build ? room.build.status : room.status;
  let took = "–";
  if (b?.started_at) {
    const seconds = fmt.secondsSince(b.started_at, b.finished_at);
    if (seconds != null) took = ACTIVE.has(status) ? `${fmt.duration(seconds)} (đang chạy)` : fmt.duration(seconds);
  } else if (room.entry?.model?.train_seconds) took = `${fmt.duration(room.entry.model.train_seconds)} (chỉ train)`;
  const kind = room.entry?.source?.kind || b?.source_kind;
  const open = room.entry && status === "done"
    ? `<a class="btn sm" href="#/view/${encodeURIComponent(room.id)}">Mở 3D</a>`
    : `<a class="btn sm" href="#/room/${encodeURIComponent(room.id)}">Chi tiết</a>`;
  return `<tr data-room="${fmt.escapeHtml(room.id)}"><td class="name">${fmt.escapeHtml(room.name)}<small>${fmt.escapeHtml(room.id)}</small></td>
    <td>${ui.sourceKind(kind)}</td><td>${ui.badge(status)}</td><td class="num">${fmt.escapeHtml(fmt.dateTime(lastActivity(room) || null))}</td>
    <td class="num">${fmt.escapeHtml(took)}</td><td class="go">${open}</td></tr>`;
}

// ------------------------------------------------------------------------------------------------ trang
export async function mount(container, ctx) {
  const viewOnly = ctx.status()?.mode === "view";
  const wrap = document.createElement("div");
  wrap.innerHTML = `${ui.topbar([["Tổng quan"]], viewOnly ? "" : `<a class="btn primary" href="#/new">${ui.icon("plus", 16)}Dự án mới</a>`)}
    <div class="page" id="lib-body">
      <div class="welcome"><div><h1 id="dash-title">Không gian 3D của bạn</h1><p id="dash-lead">Đang tải dự án…</p></div></div>
      <div class="hero${viewOnly ? " solo" : ""}" id="dash-hero"></div>
      <section class="section hidden" id="dash-pending" aria-labelledby="pending-title">
        <div class="section-head"><h2 id="pending-title">Dự án chưa hoàn thành <span class="count" id="pending-count"></span></h2></div>
        <div class="proc-list" id="pending-list"></div></section>
      <section class="section hidden" id="dash-scenes" aria-labelledby="scenes-title">
        <div class="section-head"><h2 id="scenes-title">Các cảnh 3D khác <span class="count" id="scenes-count"></span></h2></div>
        <div class="scene-grid" id="scene-grid"></div></section>
      <section class="section hidden" id="dash-history" aria-labelledby="history-title">
        <div class="section-head"><h2 id="history-title">Lịch sử xử lý <span class="count" id="history-count"></span></h2>
          <label class="sr-only" for="lib-search">Tìm dự án</label><input id="lib-search" class="search hidden" type="search" placeholder="Tìm dự án"></div>
        <div class="table-wrap"><table class="history"><thead><tr><th scope="col">Dự án</th><th scope="col">Nguồn</th><th scope="col">Trạng thái</th>
          <th scope="col">Cập nhật</th><th scope="col">Thời gian xử lý</th><th scope="col"><span class="sr-only">Mở</span></th></tr></thead>
          <tbody id="history-body"></tbody></table></div></section>
    </div>`;
  container.append(...wrap.children);
  const $ = (sel) => container.querySelector(sel);
  if (viewOnly) $("#dash-hero").style.gridTemplateColumns = "minmax(0, 1fr)";
  let rooms = [];
  let timer = null;
  let stopped = false;
  const alive = () => !stopped && (ctx.alive ? ctx.alive() : true);
  let heroKey = null;

  function paintHero(featured) {
    const key = featured ? `${featured.id}:${featured.entry.version}:${featured.build?.status || ""}` : "none";
    if (key === heroKey) return;  // không vẽ lại ảnh lớn mỗi lần hỏi lại server
    heroKey = key;
    $("#dash-hero").innerHTML = (featured ? featureBlock(featured) : emptyFeature(viewOnly)) + (viewOnly ? "" : createBlock());
    bindDrop();
  }

  function paint() {
    const scenes = rooms.filter((r) => r.status === "done" && r.entry);
    const pending = rooms.filter((r) => !(r.status === "done" && r.entry) || (r.build && ACTIVE.has(r.build.status)));
    const running = rooms.filter((r) => ACTIVE.has(r.build ? r.build.status : r.status)).length;
    $("#dash-title").textContent = scenes.length ? "Không gian 3D của bạn" : "Bắt đầu với SpatialRoom3D";
    $("#dash-lead").textContent = rooms.length
      ? `${scenes.length} cảnh 3D đã dựng${running ? `, ${running} dự án đang xử lý` : ""}. Mở một cảnh để xoay quanh, đi bộ hoặc tham quan theo đường quay.`
      : "Tải video hoặc ảnh chụp một căn phòng. Pod GPU dựng mô hình 3D Gaussian Splatting bằng 3DGS-MCMC, rồi bạn xem ngay trên trình duyệt.";
    paintHero(scenes[0] || null);

    $("#dash-pending").classList.toggle("hidden", !pending.length);
    $("#pending-count").textContent = pending.length || "";
    $("#pending-list").innerHTML = pending.map(pendingRow).join("");

    const others = scenes.slice(1);
    const q = fmt.slug($("#lib-search").value);
    const match = (r) => !q || fmt.slug(`${r.name} ${r.id}`).includes(q);
    $("#dash-scenes").classList.toggle("hidden", !others.length);
    $("#scenes-count").textContent = others.length || "";
    $("#scene-grid").innerHTML = others.filter(match).map(sceneCard).join("")
      || '<div class="empty-state"><strong>Không có cảnh nào khớp</strong></div>';

    const history = [...rooms].sort((a, b) => lastActivity(b).localeCompare(lastActivity(a)));
    $("#dash-history").classList.toggle("hidden", !rooms.length);
    $("#history-count").textContent = rooms.length || "";
    $("#lib-search").classList.toggle("hidden", rooms.length < 6);
    $("#history-body").innerHTML = history.filter(match).map(historyRow).join("")
      || '<tr><td colspan="6" class="muted">Không có dự án nào khớp.</td></tr>';
  }

  function bindDrop() {
    const drop = $("#dash-drop");
    if (!drop) return;
    drop.addEventListener("dragover", (event) => {
      event.preventDefault();
      drop.classList.add("drag");
    });
    drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
    drop.addEventListener("drop", async (event) => {
      event.preventDefault();
      drop.classList.remove("drag");
      const files = await filesFromDrop(event.dataTransfer);
      if (!files.length) return;
      ctx.setHandoff({ files });
      ctx.navigate("#/new");
    });
  }

  async function load() {
    try {
      rooms = (await api.rooms()).rooms;
      if (!alive()) return;  // đã rời trang trong lúc chờ server
      paint();
    } catch (error) {
      if (!alive()) return;
      if (error.status !== 401) {
        $("#dash-lead").textContent = "";
        $("#dash-hero").innerHTML = `<div class="notice error">${fmt.escapeHtml(error.message)}</div>`;
        heroKey = null;
      }
    }
    if (!alive()) return;
    const active = rooms.some((r) => ACTIVE.has(r.build ? r.build.status : r.status));
    timer = setTimeout(load, active ? 4000 : 20000);
  }

  $("#lib-search").addEventListener("input", paint);
  await load();
  return () => {
    stopped = true;
    clearTimeout(timer);
  };
}
