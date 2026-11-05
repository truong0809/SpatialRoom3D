// Tiến độ dựng một dự án: đọc progress.json qua /api/rooms/<id> (src/product/progress.py ghi), cuối log, hủy việc,
// dựng lại; khi xong: thẻ kết quả mở trình xem 3D.
import * as api from "./api.js";
import * as fmt from "./fmt.js";
import * as ui from "./ui.js";

const STATE_TEXT = { done: "Xong", running: "Đang chạy", waiting: "Chờ", skipped: "Bỏ qua", failed: "Lỗi", stopped: "Đã dừng" };
const TITLE = {
  queued: "đang chờ GPU", running: "đang dựng", done: "đã xong", failed: "bị lỗi", stopped: "đã dừng", lost: "bị gián đoạn",
};

function stageDuration(stage) {
  if (!stage.started_at) return "";
  const seconds = fmt.secondsSince(stage.started_at, stage.finished_at);
  return seconds == null ? "" : fmt.duration(seconds);
}

function stageMark(stage, index) {
  if (stage.status === "done") return ui.icon("check", 15, 2.4);
  if (stage.status === "failed" || stage.status === "stopped") return ui.icon("x", 14, 2.4);
  if (stage.status === "skipped") return "–";
  if (stage.status === "running") return '<span aria-hidden="true" style="width:8px;height:8px;border-radius:50%;background:currentColor"></span>';
  return `<span style="font-size:12px;font-weight:600">${index + 1}</span>`;
}

export async function mount(container, ctx, roomId) {
  const viewOnly = ctx.status()?.mode === "view";
  const wrap = document.createElement("div");
  wrap.innerHTML = `${ui.topbar([["Tổng quan", "#/"], [roomId]], '<div class="actions" id="job-actions"></div>')}
    <div class="page">
      <div class="job-head"><div>
        <div class="title"><h1 id="job-title">${fmt.escapeHtml(roomId)}</h1><span id="job-badge"></span></div>
        <p class="lead" id="job-lead">Đang tải…</p></div></div>
      <div class="job-grid">
        <div class="job-main">
          <section class="panel" aria-label="Tiến trình">
            <div class="overall">
              <div><h2 id="job-status">…</h2><div class="pct" id="job-pct" aria-live="polite"></div></div>
              <div class="elapsed" id="job-elapsed"></div></div>
            <div class="progressbar" id="job-bar"><span></span></div>
            <ol class="pipeline" id="job-steps"></ol>
            <div class="notice hidden" id="job-notice" role="status"></div>
          </section>
          <section class="panel" aria-labelledby="log-title">
            <div class="panel-head"><h2 id="log-title">Nhật ký</h2>
              <button type="button" class="btn sm" id="job-log-more" aria-expanded="false">Xem nhật ký đầy đủ</button></div>
            <pre class="log" id="job-log" tabindex="0">…</pre>
          </section>
        </div>
        <div class="job-side">
          <section class="result" id="job-result" aria-labelledby="result-title"></section>
          <section class="panel" aria-labelledby="info-title"><h2 id="info-title">Thông tin</h2><dl class="kv" id="job-info"></dl></section>
        </div>
      </div>
    </div>`;
  container.append(...wrap.children);
  const $ = (sel) => container.querySelector(sel);
  // thanh trên: phần "actions" đầu tiên là của khung chung, phần trong cùng là nơi đặt nút của trang
  const actionsBox = $("#job-actions");
  let detail = null;
  let stopped = false;
  const alive = () => !stopped && (ctx.alive ? ctx.alive() : true);
  let timer = null;
  let logTimer = null;
  let logLines = 14;
  let cancelArmed = false;
  let resultKey = null;

  function actions() {
    const status = detail?.status;
    const parts = [];
    // nút chính "Mở trình xem 3D" nằm ở thẻ kết quả; trên thanh trên là lối tắt (không trùng hai nút chính)
    if (detail?.entry) parts.push(`<a class="btn" href="#/view/${encodeURIComponent(roomId)}">${ui.icon("cube", 16)}Mở trình xem 3D</a>`);
    if ((status === "queued" || status === "running") && !viewOnly) {
      parts.push(`<button type="button" class="btn danger" id="job-cancel">${cancelArmed ? "Bấm lần nữa để hủy" : "Hủy việc"}</button>`);
    }
    if ((status === "failed" || status === "stopped" || status === "lost") && !viewOnly) {
      parts.push(`<button type="button" class="btn" id="job-rebuild">${ui.icon("reset", 16)}Dựng lại</button>`);
      parts.push(`<a class="btn quiet" href="#/new">Dự án khác</a>`);
    }
    actionsBox.innerHTML = parts.join("");
    $("#job-cancel")?.addEventListener("click", cancel);
    $("#job-rebuild")?.addEventListener("click", rebuild);
  }

  async function cancel() {
    if (!cancelArmed) {
      cancelArmed = true;
      actions();
      setTimeout(() => {
        cancelArmed = false;
        if (alive()) actions();
      }, 5000);
      return;
    }
    cancelArmed = false;
    try {
      const result = await api.cancelRoom(roomId);
      if (!alive()) return;
      notice(result.message, "");
    } catch (error) {
      if (!alive()) return;
      notice(error.message, "error");
    }
    actions();
  }

  async function rebuild() {
    try {
      const result = await api.api(`/api/rooms/${encodeURIComponent(roomId)}/rebuild`, { method: "POST", json: {} });
      if (!alive()) return;
      notice(result.message, "");
      ctx.refreshStatus();
      ctx.refreshRecent?.();
      load();
    } catch (error) {
      if (alive()) notice(error.message, "error");
    }
  }

  function notice(text, tone) {
    const box = $("#job-notice");
    box.className = `notice${tone ? ` ${tone}` : ""}`;
    box.textContent = text || "";
    box.classList.toggle("hidden", !text);
  }

  function paintResult() {
    const e = detail.entry;
    const status = detail.status;
    const key = e ? `done:${e.version}` : `pending:${status}`;
    if (key === resultKey) return;
    resultKey = key;
    const box = $("#job-result");
    if (e) {
      const media = e.files?.thumbnail
        ? `<img src="/web/${encodeURI(e.files.thumbnail)}?v=${encodeURIComponent(e.version || "")}" alt="">` : ui.splatArt("placeholder");
      box.innerHTML = `<div class="media">${media}</div><div class="body">
        <div><h2 id="result-title">Cảnh 3D đã sẵn sàng</h2><p>${fmt.escapeHtml(fmt.millions(e.num_gaussians))} Gaussian, file xem web ${fmt.escapeHtml(fmt.bytes(e.size_bytes))}.</p></div>
        <a class="btn primary block" href="#/view/${encodeURIComponent(roomId)}">${ui.icon("cube", 16)}Mở trình xem 3D</a></div>`;
      return;
    }
    const failed = status === "failed" || status === "stopped" || status === "lost";
    box.innerHTML = `<div class="media">${ui.splatArt("placeholder")}</div><div class="body">
      <div><h2 id="result-title">Kết quả</h2><p>${failed ? "Dựng chưa xong. Xem lỗi ở phần tiến trình, sửa nguồn nếu cần rồi bấm Dựng lại."
        : "Cảnh 3D xuất hiện ở đây khi pod dựng xong; trang tự cập nhật."}</p></div></div>`;
  }

  function paint() {
    const p = detail.progress;
    const status = detail.status;
    const name = detail.name || roomId;
    $("#job-title").textContent = name;
    container.querySelector(".crumbs .current").textContent = name;
    $("#job-badge").innerHTML = ui.badge(status);
    const kind = p?.source_kind || detail.entry?.source?.kind;
    $("#job-lead").textContent = [ui.sourceKind(kind), p?.created_at ? `gửi lúc ${fmt.dateTime(p.created_at)}` : ""].filter(Boolean).join(", ");
    $("#job-status").textContent = `Dự án ${TITLE[status] || status || ""}`;
    const start = p?.started_at || p?.created_at;
    const seconds = start ? fmt.secondsSince(start, p?.finished_at) : null;
    $("#job-elapsed").textContent = seconds != null ? `${p?.finished_at ? "Tổng thời gian" : "Đã chạy"} ${fmt.clock(seconds)}` : "";
    const bar = $("#job-bar");
    const pct = status === "done" ? 100 : ui.overallPercent(p?.stages);
    const ended = status === "failed" || status === "stopped" || status === "lost";
    bar.classList.toggle("indeterminate", status === "queued");
    bar.classList.toggle("failed", ended);
    bar.firstElementChild.style.width = status === "queued" ? "30%" : `${(pct ?? 0).toFixed(1)}%`;
    // lỗi, đã dừng, bị gián đoạn: phần trăm không còn ý nghĩa, hiện bước đã dừng lại
    const stages = p?.stages || [];
    const stopAt = ended ? stages.findIndex((stage) => ["failed", "stopped", "running"].includes(stage.status)) : -1;
    if (stopAt >= 0) {
      $("#job-pct").innerHTML = `<small class="lead">Bước</small>${stopAt + 1}<small>/${stages.length}</small>`
        + `<span class="est">dừng ở ${fmt.escapeHtml(stages[stopAt].label)}</span>`;
    } else {
      $("#job-pct").innerHTML = status === "queued" || pct == null ? "–"
        : `${fmt.decimal(pct, 0)}<small>%</small>${status === "done" ? "" : '<span class="est">tổng, ước lượng theo bước</span>'}`;
    }
    $("#job-steps").innerHTML = (p?.stages || []).map((stage, index) => {
      let state = STATE_TEXT[stage.status] || stage.status;
      if (stage.status === "running" && stage.percent != null) state += `, ${fmt.decimal(stage.percent, 0)} %`;
      const took = stageDuration(stage);
      if (took && stage.status !== "waiting") state += `, ${took}`;
      const detailLine = stage.message && (stage.status === "running" || stage.status === "failed" || stage.status === "skipped")
        ? `<span class="detail">${fmt.escapeHtml(stage.message)}</span>` : "";
      return `<li class="${fmt.escapeHtml(stage.status)}" data-stage="${fmt.escapeHtml(stage.id)}"><span class="mark" aria-hidden="true">${stageMark(stage, index)}</span>
        <span class="name">${fmt.escapeHtml(stage.label)}</span><span class="state">${fmt.escapeHtml(state)}</span>${detailLine}</li>`;
    }).join("") || `<li class="waiting"><span class="mark" aria-hidden="true"></span><span class="name">Chưa có tiến độ cho dự án này</span></li>`;
    if (status === "failed" || status === "stopped") notice(p?.error || p?.message || "Việc dựng không xong.", "error");
    else if (status === "lost") notice("Việc dựng không còn chạy (pod tắt hoặc khởi động lại giữa chừng). Bấm Dựng lại: ảnh và COLMAP đã xong thì được dùng lại.", "error");
    else if (status === "queued") notice("Đang có việc khác dùng GPU: dự án này tự bắt đầu khi việc đó xong.", "");
    else if (status === "done") notice("Xong: cảnh 3D đã có ở trang tổng quan.", "ok");
    else notice("", "");
    const info = [
      ["Mã dự án", roomId],
      ["Nguồn", ui.sourceKind(kind)],
      ["Việc nền", p?.job || "–"],
      ["Gửi lúc", fmt.dateTime(p?.created_at)],
      ["Bắt đầu", fmt.dateTime(p?.started_at)],
      ["Kết thúc", fmt.dateTime(p?.finished_at)],
    ];
    if (detail.entry) {
      info.push(["Số Gaussian", fmt.int(detail.entry.num_gaussians)], ["File xem web", fmt.bytes(detail.entry.size_bytes)]);
    }
    $("#job-info").innerHTML = info.map(([k, v]) => `<dt>${fmt.escapeHtml(k)}</dt><dd>${fmt.escapeHtml(v)}</dd>`).join("");
    paintResult();
    if (!cancelArmed) actions();
  }

  async function load() {
    clearTimeout(timer);
    try {
      detail = await api.room(roomId);
      if (!alive()) return;  // đã rời trang trong lúc chờ server
      paint();
    } catch (error) {
      if (!alive()) return;
      if (error.status !== 401) {
        $("#job-status").textContent = "";
        $("#job-lead").textContent = "";
        notice(error.message, "error");
      }
    }
    if (!alive()) return;
    const active = detail && (detail.status === "queued" || detail.status === "running");
    timer = setTimeout(load, active ? 2000 : 15000);
  }

  async function loadLog() {
    clearTimeout(logTimer);
    if (!alive()) return;
    const pre = $("#job-log");
    try {
      const data = await api.roomLog(roomId, logLines);
      if (!alive()) return;
      const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;
      pre.textContent = data.lines.length ? data.lines.join("\n") : "Chưa có nhật ký cho dự án này.";
      pre.classList.toggle("empty", !data.lines.length);
      if (atBottom) pre.scrollTop = pre.scrollHeight;
    } catch (error) {
      if (!alive()) return;
      if (error.status !== 401) pre.textContent = error.message;
    }
    if (!alive()) return;
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
