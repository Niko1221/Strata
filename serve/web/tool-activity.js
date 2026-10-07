// Readable tool activity; the app owns streaming, disclosure state and I/O.
(function (root) {
  "use strict";
  const STATES = {writing: ["st-badge--reading", "Preparing"], running: ["st-badge--generating", "Running"],
    done: ["", "Done"], error: ["st-badge--error", "Failed"], skipped: ["st-badge--queued", "Skipped"]};
  const escape = value => String(value ?? "").replace(/[&<>"']/g, char =>
    ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[char]);
  const text = (value, limit = 180) => typeof value === "string" ? value.replace(/\s+/g, " ").trim().slice(0, limit) : "";
  function object(value) {
    if (typeof value === "string" && value.length <= 65536) {
      try { value = JSON.parse(value); } catch (_) { return {}; }
    }
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  }
  function url(value) {
    try {
      const parsed = new URL(value);
      return ["https:", "http:"].includes(parsed.protocol) && !parsed.username && !parsed.password ? parsed : null;
    } catch (_) { return null; }
  }
  function describe(call) {
    const kind = text(call.tool || call.name).split("__").pop(), args = object(call.arguments), result = object(call.result);
    const page = url(result.source_url) || url(args.url);
    let label = kind.replace(/_/g, " ") || "Tool", subject = text(args.query || args.path), note = "";
    if (kind === "web_search") {
      label = "Search web";
      if (Array.isArray(result.results)) note = `${result.results.length} result${result.results.length === 1 ? "" : "s"}`;
    } else if (kind === "read_webpage") {
      label = "Read page";
      subject = text(result.title) || (page ? page.hostname : text(args.url));
      note = page ? page.hostname : "";
    } else label = label[0].toUpperCase() + label.slice(1);
    if (call.state === "error") {
      const status = String(call.result).match(/HTTP(?: Error)? (\d{3})/);
      note = status ? `HTTP ${status[1]}` : "Tool failed";
    }
    if (call.state === "skipped") note = "Not executed";
    return {label, subject, note, page, web: kind === "web_search" || kind === "read_webpage"};
  }
  function renderCall(call, index, options = {}) {
    const esc = options.escape || escape, icon = options.icon || (() => "");
    const [cls, label] = STATES[call.state] || ["", "Pending"], info = describe(call);
    let body = "";
    if (call.open) {
      const args = call.arguments == null ? "" : JSON.stringify(call.arguments, null, 2);
      body = `<div class="tool-call__label">${esc(call.name || call.tool || "Tool")}</div>` +
        `<div class="tool-call__label">Arguments</div><pre class="tool-call__pre">${esc(args || "(being written)")}</pre>`;
      if (call.result != null) {
        body += `<div class="tool-call__label">${call.ok ? "Result" : "Error"}${call.chars ? ` · ${esc(call.chars)} characters` : ""}` +
          `${call.truncated ? " · shortened for the model" : ""}</div><pre class="tool-call__pre">${esc(call.result)}</pre>`;
      }
      if (info.page) body += `<a class="tool-call__source" href="${esc(info.page.href)}" target="_blank" rel="noopener noreferrer">Open source ↗</a>`;
    }
    const ms = typeof call.ms === "number" && Number.isFinite(call.ms) && call.ms >= 0 ? call.ms : null;
    return `<details class="st-collapse tool-call" data-tool="${index}" data-state="${esc(call.state || "writing")}"${call.open ? " open" : ""}>` +
      `<summary>${icon("tool", "st-icon st-icon--sm")}<span class="tool-call__copy">` +
      `<span class="tool-call__name">${esc(info.label)}</span>` +
      `<span class="tool-call__preview muted" title="${esc(info.subject)}">${esc(info.subject || info.note || "Preparing tool call…")}</span></span>` +
      (info.note && info.note !== info.subject ? `<span class="tool-call__note muted">${esc(info.note)}</span>` : "") +
      `<span class="st-badge ${cls}">${label}</span>` +
      (ms !== null && call.state !== "skipped" ? `<span class="tool-call__time muted">${Math.ceil(ms / 1000)}s</span>` : "") +
      `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body">${body}</div></details>`;
  }
  function renderGroup(calls, offset, options = {}) {
    if (!calls.length) return "";
    if (calls.length === 1) return renderCall(calls[0], offset, options);
    const esc = options.escape || escape, icon = options.icon || (() => ""), descriptions = calls.map(describe);
    const activeIndex = calls.findLastIndex(call => call.state === "writing" || call.state === "running");
    const failures = calls.filter(call => call.state === "error").length;
    const skipped = calls.filter(call => call.state === "skipped").length;
    const pages = new Set(calls.flatMap((call, index) => call.state === "done" && descriptions[index].page
      ? [descriptions[index].page.href] : []));
    const active = activeIndex >= 0, title = descriptions.every(info => info.web) ? "Web research" : "Tool activity";
    const detail = active ? `${descriptions[activeIndex].label} · ${descriptions[activeIndex].subject || "Preparing…"}` :
      `${calls.length} steps${pages.size ? ` · ${pages.size} page${pages.size === 1 ? "" : "s"}` : ""}${failures ? ` · ${failures} failed` : ""}${skipped ? ` · ${skipped} skipped` : ""}`;
    const label = active ? "Working" : failures || skipped ? "Partial" : "Done";
    const cls = active ? "st-badge--generating" : failures || skipped ? "st-badge--queued" : "";
    const open = options.open === undefined ? calls.some(call => call.open) : options.open;
    return `<details class="st-collapse tool-group" data-tool-group="${offset}" data-state="${active ? "running" : "done"}"${open ? " open" : ""}>` +
      `<summary>${icon("tool", "st-icon st-icon--sm")}<span class="tool-call__copy"><span class="tool-group__title">${title}</span>` +
      `<span class="tool-call__preview muted">${esc(detail)}</span></span><span class="st-badge ${cls}">${label}</span>` +
      `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary>` +
      `<div class="tool-group__body">${open ? calls.map((call, index) => renderCall(call, offset + index, options)).join("") : ""}</div></details>`;
  }
  const api = {describe, renderCall, renderGroup};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.StrataToolActivity = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
