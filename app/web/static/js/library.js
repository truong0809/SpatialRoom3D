// Thư viện phòng: mọi phòng đã dựng xong (manifest.json) và các phòng đang dựng / lỗi (progress.json).
import * as api from "./api.js";
import * as fmt from "./fmt.js";

const STATUS = {
  queued: ["Đang chờ GPU", "warn"],
  running: ["Đang dựng", "accent"],
  failed: ["Lỗi", "error"],
  stopped: ["Đã dừng", "error"],
  lost: ["Bị gián đoạn", "error"],
  done: ["Xong", "accent"],
};
const ROOM_ICON = `<svg class="placeholder" aria-hidden="true" width="132" height="88" viewBox="0 0 132 88" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><path d="M16 56 66 32l50 24-50 24z"/><path d="M16 56V22L66 2v30"/><path d="M66 2l50 20v34"/><path d="M50 58l14-7 14 7-14 7z" stroke="#4C9AFF"/></svg>`;

function sourceLine(entry) {
  const src = entry.source || {};
  const parts = [];
  if (src.kind === "video") parts.push(src.video_duration_s ? `Video ${fmt.clock(src.video_duration_s)}` : "Video");
  else if (src.kind === "photos") parts.push(src.input_images ? `${fmt.int(src.input_images)} ảnh` : "Ảnh chụp");
  if (src.registered_images != null) parts.push(`${fmt.int(src.registered_images)} ảnh định vị`);
  if (src.sfm_verdict) parts.push(`COLMAP ${src.sfm_verdict}`);
  return parts.join(" · ");
}

function doneCard(room) {
  const e = room.entry;
  const thumb = e.files?.thumbnail
    ? `<img src="/web/${encodeURI(e.files.thumbnail)}?v=${encodeURIComponent(e.version || "")}" alt="" loading="lazy">`
    : ROOM_ICON;
  const kind = e.source?.kind === "video" ? "Video" : e.source?.kind === "photos" ? "Ảnh" : "Phòng";
  const warn = (e.warnings || []).length
    ? `<p class="notice warn">${(e.warnings || []).map(fmt.escapeHtml).join("<br>")}</p>` : "";
  const rebuilding = room.build ? `<p class="sub">Đang dựng lại: ${fmt.escapeHtml(room.build.stage_label || room.build.message || "")}</p>` : "";
  return `<article class="card" data-room="${fmt.escapeHtml(room.id)}" data-kind="${fmt.escapeHtml(e.source?.kind || "")}">
    <div class="thumb">${thumb}
      <span class="badge left">${kind}</span><span class="badge accent right">${fmt.escapeHtml(e.model?.name || "mcmc")}</span></div>
    <div class="body">
      <div><h2>${fmt.escapeHtml(room.name)}</h2>
        <p class="sub">${fmt.escapeHtml(sourceLine(e))}</p>
        <p class="sub">Dựng xong ${fmt.escapeHtml(fmt.dateTime(e.model?.trained_at || e.built_at))}</p>${rebuilding}</div>
      <dl class="stats">
        <div><dt>Gaussian</dt><dd>${fmt.millions(e.num_gaussians)}</dd></div>
        <div><dt>File web</dt><dd>${fmt.bytes(e.size_bytes)}</dd></div>
        <div><dt>Train</dt><dd>${e.model?.train_seconds ? fmt.decimal(e.model.train_seconds / 60, 1) + " ph" : "–"}</dd></div>
      </dl>${warn}
      <div class="actions"><a class="btn primary" href="#/view/${encodeURIComponent(room.id)}">Xem 3D</a></div>
    </div></article>`;
}

function buildCard(room) {
  const b = room.build || {};
  const [label, tone] = STATUS[room.status] || [room.status, ""];
  const active = room.status === "queued" || room.status === "running";
  const percent = b.stage === "train" && Number.isFinite(b.percent) ? Math.max(0, Math.min(100, b.percent)) : null;
  const bar = active
    ? `<div class="progressbar${percent == null ? " indeterminate" : ""}" aria-hidden="true"><span style="width:${percent ?? 30}%"></span></div>` : "";
  const detail = room.status === "failed" || room.status === "lost" || room.status === "stopped"
    ? fmt.escapeHtml(b.error || (room.status === "lost" ? "Việc dựng phòng không còn chạy (pod tắt hoặc khởi động lại giữa chừng)" : b.message || ""))
    : fmt.escapeHtml([b.stage_label, b.message].filter(Boolean).join(" · "));
  return `<article class="card" data-room="${fmt.escapeHtml(room.id)}" data-status="${fmt.escapeHtml(room.status)}" data-kind="${fmt.escapeHtml(b.source_kind || "")}">
    <div class="thumb">${ROOM_ICON}<span class="badge ${tone} left">${fmt.escapeHtml(label)}</span></div>
    <div class="body">
      <div><h2>${fmt.escapeHtml(room.name)}</h2><p class="sub">${detail}</p>
        <p class="sub">Bắt đầu ${fmt.escapeHtml(fmt.dateTime(b.started_at || b.created_at))}</p></div>
      ${bar}
      <div class="actions"><a class="btn" href="#/room/${encodeURIComponent(room.id)}">Xem tiến độ</a></div>
    </div></article>`;
}

export async function mount(container, ctx) {
  const page = document.createElement("div");
  page.className = "page";
  page.innerHTML = `<div class="page-head">
      <div><h1>Thư viện phòng</h1><p class="lead" id="lib-lead">Đang tải…</p></div>
      <div class="row"><label class="sr-only" for="lib-search">Tìm phòng</label>
        <input id="lib-search" class="search" type="search" placeholder="Tìm phòng…">
        <a class="btn primary" id="lib-new" href="#/new">Dựng phòng mới</a></div>
    </div>
    <div class="chips" role="group" aria-label="Lọc phòng" id="lib-filters"></div>
    <div id="lib-body"></div>`;
  container.append(page);
  if (ctx.status()?.mode === "view") page.querySelector("#lib-new").classList.add("hidden");
  let filter = "all";
  let rooms = [];
  let timer = null;
  let stopped = false;

  const body = page.querySelector("#lib-body");
  const search = page.querySelector("#lib-search");

  function paint() {
    const counts = {
      all: rooms.length,
      active: rooms.filter((r) => r.status !== "done").length,
      video: rooms.filter((r) => (r.entry?.source?.kind || r.build?.source_kind) === "video").length,
      photos: rooms.filter((r) => (r.entry?.source?.kind || r.build?.source_kind) === "photos").length,
    };
    const chips = [["all", "Tất cả"], ["active", "Đang dựng / lỗi"], ["video", "Từ video"], ["photos", "Từ ảnh"]]
      .filter(([key]) => key === "all" || counts[key] > 0)
      .map(([key, label]) => `<button type="button" class="chip" data-filter="${key}" aria-pressed="${filter === key}">${label} · ${counts[key]}</button>`);
    page.querySelector("#lib-filters").innerHTML = chips.join("");
    const done = rooms.filter((r) => r.status === "done").length;
    page.querySelector("#lib-lead").textContent = rooms.length
      ? `${done} phòng đã dựng bằng mô hình chính (3DGS-MCMC)${counts.active ? `, ${counts.active} phòng đang dựng hoặc lỗi` : ""}. Mở một phòng để xoay quanh, đi bộ hoặc tham quan theo đường quay.`
      : "Chưa có phòng nào.";
    const q = fmt.slug(search.value);
    const shown = rooms.filter((r) => {
      const kind = r.entry?.source?.kind || r.build?.source_kind;
      if (filter === "active" && r.status === "done") return false;
      if ((filter === "video" || filter === "photos") && kind !== filter) return false;
      return !q || fmt.slug(`${r.name} ${r.id}`).includes(q);
    });
    if (!rooms.length) {
      body.innerHTML = `<div class="empty"><p>Chưa có phòng nào trong thư viện.</p>${ctx.status()?.mode === "view" ? "" : '<a class="btn primary" href="#/new">Tải video hoặc ảnh để dựng phòng đầu tiên</a>'}</div>`;
    } else if (!shown.length) {
      body.innerHTML = `<div class="empty"><p>Không có phòng nào khớp.</p></div>`;
    } else {
      body.innerHTML = `<div class="grid">${shown.map((r) => (r.status === "done" && r.entry ? doneCard(r) : buildCard(r))).join("")}</div>`;
    }
  }

  async function load() {
    try {
      rooms = (await api.rooms()).rooms;
      paint();
    } catch (error) {
      if (error.status !== 401) body.innerHTML = `<div class="notice error">${fmt.escapeHtml(error.message)}</div>`;
    }
    if (stopped) return;
    const active = rooms.some((r) => r.status === "queued" || r.status === "running" || r.build);
    timer = setTimeout(load, active ? 4000 : 20000);
  }

  page.querySelector("#lib-filters").addEventListener("click", (event) => {
    const button = event.target.closest("[data-filter]");
    if (!button) return;
    filter = button.dataset.filter;
    paint();
  });
  search.addEventListener("input", paint);
  await load();
  return () => {
    stopped = true;
    clearTimeout(timer);
  };
}
