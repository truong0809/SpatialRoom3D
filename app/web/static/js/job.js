// Tiến độ dựng một phòng: đọc progress.json qua /api/rooms/<id> (src/product/progress.py ghi), cuối log, hủy việc.
import * as api from "./api.js";
import * as fmt from "./fmt.js";

const WEIGHTS = { images: 0.1, colmap: 0.25, train: 0.6, export: 0.05 };  // ước lượng phần thời gian của mỗi bước
const STATE_TEXT = { done: "xong", running: "đang chạy", waiting: "chờ", skipped: "bỏ qua", failed: "lỗi", stopped: "đã dừng" };
const MARK = { done: "✓", running: "•", waiting: "", skipped: "–", failed: "✕", stopped: "✕" };
const TITLE = {
  queued: "đang chờ GPU", running: "đang dựng", done: "đã xong", failed: "lỗi", stopped: "đã dừng", lost: "bị gián đoạn",
};

function overall(progress) {
  if (!progress) return null;
  let value = 0;
  for (const stage of progress.stages || []) {
    const w = WEIGHTS[stage.id] ?? 0;
    if (stage.status === "done" || stage.status === "skipped") value += w;
    else if (stage.status === "running") value += w * (stage.percent != null ? stage.percent / 100 : 0.4);
  }
  return Math.min(100, value * 100);
}

function stageDuration(stage) {
  if (!stage.started_at) return "";
  const seconds = fmt.secondsSince(stage.started_at, stage.finished_at);
  return seconds == null ? "" : ` · ${fmt.duration(seconds)}`;
}

export async function mount(container, ctx, roomId) {
  const page = document.createElement("div");
  page.className = "page";
  page.innerHTML = `<div class="page-head"><div><h1 id="job-title">${fmt.escapeHtml(roomId)}</h1>
      <p class="lead" id="job-lead">Đang tải…</p></div>
      <div class="row"><a class="btn" href="#/">Thư viện</a></div></div>
    <div class="columns">
      <div class="main-col" style="display:flex;flex-direction:column;gap:20px">
        <section class="panel" aria-label="Tiến độ">
          <div class="row" style="justify-content:space-between"><h2 id="job-status">…</h2>
            <span class="mono muted" id="job-elapsed"></span></div>
          <div class="progressbar" id="job-bar"><span></span></div>
          <ol class="steps" id="job-steps"></ol>
          <div class="notice hidden" id="job-notice" role="status"></div>
          <div class="row" id="job-actions"></div>
        </section>
        <section class="panel" aria-label="Nhật ký">
          <div class="row" style="justify-content:space-between"><h2>Nhật ký</h2>
            <button type="button" class="btn small" id="job-log-more" aria-expanded="false">Xem nhật ký đầy đủ</button></div>
          <pre class="log" id="job-log" tabindex="0">…</pre>
        </section>
      </div>
      <aside class="side-col"><section class="panel quiet"><h2>Thông tin</h2><dl class="kv" id="job-info"></dl></section></aside>
    </div>`;
  container.append(page);
  const $ = (sel) => page.querySelector(sel);
  let detail = null;
  let stopped = false;
  let timer = null;
  let logTimer = null;
  let logLines = 14;
  let cancelArmed = false;

  function actions() {
    const box = $("#job-actions");
    const status = detail?.status;
    const parts = [];
    if (detail?.entry) parts.push(`<a class="btn primary" href="#/view/${encodeURIComponent(roomId)}">Xem 3D</a>`);
    if ((status === "queued" || status === "running") && ctx.status()?.mode !== "view") {
      parts.push(`<button type="button" class="btn danger" id="job-cancel">${cancelArmed ? "Bấm lần nữa để hủy" : "Hủy việc"}</button>`);
    }
    if ((status === "failed" || status === "stopped" || status === "lost") && ctx.status()?.mode !== "view") {
      parts.push(`<button type="button" class="btn" id="job-rebuild">Dựng lại</button>`);
      parts.push(`<a class="btn" href="#/new">Dựng phòng khác</a>`);
    }
    box.innerHTML = parts.join("");
    $("#job-cancel")?.addEventListener("click", cancel);
    $("#job-rebuild")?.addEventListener("click", rebuild);
  }

  async function cancel() {
    if (!cancelArmed) {
      cancelArmed = true;
      actions();
      setTimeout(() => {
        cancelArmed = false;
        if (!stopped) actions();
      }, 5000);
      return;
    }
    cancelArmed = false;
    try {
      const result = await api.cancelRoom(roomId);
      notice(result.message, "");
    } catch (error) {
      notice(error.message, "error");
    }
    actions();
  }

  async function rebuild() {
    try {
      const result = await api.api(`/api/rooms/${encodeURIComponent(roomId)}/rebuild`, { method: "POST", json: {} });
      notice(result.message, "");
      ctx.refreshStatus();
      load();
    } catch (error) {
      notice(error.message, "error");
    }
  }

  function notice(text, tone) {
    const box = $("#job-notice");
    box.className = `notice${tone ? ` ${tone}` : ""}`;
    box.textContent = text || "";
    box.classList.toggle("hidden", !text);
  }

  function paint() {
    const p = detail.progress;
    const status = detail.status;
    $("#job-title").textContent = detail.name || roomId;
    const kind = p?.source_kind === "video" ? "Từ video" : p?.source_kind === "photos" ? "Từ ảnh chụp" : "";
    $("#job-lead").textContent = [kind, p?.created_at ? `gửi lúc ${fmt.dateTime(p.created_at)}` : ""].filter(Boolean).join(" · ") || "Phòng trong thư viện";
    $("#job-status").textContent = `Phòng ${TITLE[status] || status || ""}`;
    const start = p?.started_at || p?.created_at;
    const seconds = start ? fmt.secondsSince(start, p?.finished_at) : null;
    $("#job-elapsed").textContent = seconds != null ? `${p?.finished_at ? "tổng" : "đã chạy"} ${fmt.clock(seconds)}` : "";
    const bar = $("#job-bar");
    const pct = status === "done" ? 100 : overall(p);
    bar.classList.toggle("indeterminate", status === "queued");
    bar.firstElementChild.style.width = status === "queued" ? "30%" : `${(pct ?? 0).toFixed(1)}%`;
    $("#job-steps").innerHTML = (p?.stages || []).map((stage) => {
      let state = STATE_TEXT[stage.status] || stage.status;
      if (stage.status === "running" && stage.percent != null) state += ` · ${fmt.decimal(stage.percent, 0)} %`;
      if (stage.status === "done" || stage.status === "failed") state += stageDuration(stage);
      const detailLine = stage.message && (stage.status === "running" || stage.status === "failed" || stage.status === "skipped")
        ? `<span class="detail">${fmt.escapeHtml(stage.message)}</span>` : "";
      return `<li class="${fmt.escapeHtml(stage.status)}" data-stage="${fmt.escapeHtml(stage.id)}"><span class="mark" aria-hidden="true">${MARK[stage.status] ?? ""}</span>
        <span class="name">${fmt.escapeHtml(stage.label)}${detailLine}</span><span class="state">${fmt.escapeHtml(state)}</span></li>`;
    }).join("") || `<li class="waiting"><span class="mark"></span><span class="name">Chưa có tiến độ cho phòng này</span></li>`;
    if (status === "failed" || status === "stopped") notice(p?.error || p?.message || "Việc dựng phòng không xong", "error");
    else if (status === "lost") notice("Việc dựng phòng không còn chạy (pod tắt hoặc khởi động lại giữa chừng). Bấm Dựng lại: ảnh và COLMAP đã xong thì được dùng lại.", "error");
    else if (status === "queued") notice("Đang có việc khác dùng GPU: phòng này sẽ tự dựng khi việc đó xong.", "");
    else if (status === "done") notice("Xong: phòng đã có trong thư viện.", "");
    else notice(p?.message || "", "");
    const info = [
      ["Mã phòng", roomId],
      ["Nguồn", kind || "–"],
      ["Việc nền", p?.job || "–"],
      ["Gửi lúc", fmt.dateTime(p?.created_at)],
      ["Bắt đầu", fmt.dateTime(p?.started_at)],
      ["Kết thúc", fmt.dateTime(p?.finished_at)],
    ];
    if (detail.entry) {
      info.push(["Số Gaussian", fmt.int(detail.entry.num_gaussians)], ["File web", fmt.bytes(detail.entry.size_bytes)]);
    }
    $("#job-info").innerHTML = info.map(([k, v]) => `<dt>${fmt.escapeHtml(k)}</dt><dd>${fmt.escapeHtml(v)}</dd>`).join("");
    if (!cancelArmed) actions();
  }

  async function load() {
    clearTimeout(timer);
    try {
      detail = await api.room(roomId);
      paint();
    } catch (error) {
      if (error.status !== 401) {
        $("#job-status").textContent = "";
        notice(error.message, "error");
      }
    }
    if (stopped) return;
    const active = detail && (detail.status === "queued" || detail.status === "running");
    timer = setTimeout(load, active ? 2000 : 15000);
  }

  async function loadLog() {
    clearTimeout(logTimer);
    const pre = $("#job-log");
    try {
      const data = await api.roomLog(roomId, logLines);
      const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;
      pre.textContent = data.lines.length ? data.lines.join("\n") : "(chưa có nhật ký)";
      if (atBottom) pre.scrollTop = pre.scrollHeight;
    } catch (error) {
      if (error.status !== 401) pre.textContent = error.message;
    }
    if (stopped) return;
    const active = detail && (detail.status === "queued" || detail.status === "running");
    logTimer = setTimeout(loadLog, active ? 4000 : 20000);
  }

  $("#job-log-more").addEventListener("click", (event) => {
    const expanded = logLines > 14;
    logLines = expanded ? 14 : 400;
    event.currentTarget.setAttribute("aria-expanded", String(!expanded));
    event.currentTarget.textContent = expanded ? "Xem nhật ký đầy đủ" : "Thu gọn";
    loadLog();
  });
  await load();
  loadLog();
  return () => {
    stopped = true;
    clearTimeout(timer);
    clearTimeout(logTimer);
  };
}
