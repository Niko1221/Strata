"use strict";
const $ = id => document.getElementById(id);
let selected = null, detail = null, pane = "input", records = [], state = null, operating = false, stamp = "", actionError = "";
let apiKey = sessionStorage.getItem("strata.monitor.key") || "";
document.documentElement.dataset.theme = localStorage.getItem("strata.theme") || "dark";
const seconds = n => typeof n === "number" ? `${n.toFixed(2)} s` : "—";
const speed = n => typeof n === "number" ? n.toFixed(1) : "—";
const active = r => !["completed", "error", "disconnected"].includes(r.state);
function text(id, value) { const el = $(id); if (el.textContent !== String(value)) el.textContent = value; }
async function api(path, body) {
  const response = await fetch(path, {cache:"no-store", headers:{...(apiKey ? {Authorization:`Bearer ${apiKey}`} : {}), ...(body !== undefined ? {"Content-Type":"application/json"} : {})}, ...(body !== undefined ? {method:"POST", body:JSON.stringify(body)} : {})});
  const value = await response.json();
  if (!response.ok) {
    if (response.status === 401) $("auth").hidden = false;
    throw new Error(value.error?.message || `HTTP ${response.status}`);
  }
  return value;
}
function notice(message) { $("error").hidden = !message; text("error", message || ""); }
function list() {
  const query = $("search").value.toLowerCase(), filter = $("filter").value;
  const visible = records.filter(r => `${r.id} ${r.path} ${r.model}`.toLowerCase().includes(query) && (filter === "all" || (filter === "active" ? active(r) : r.state === filter)));
  const nextStamp = JSON.stringify([visible, selected]);
  if (stamp === nextStamp) return;
  stamp = nextStamp;
  const focusId = document.activeElement?.dataset.request;
  $("requests").replaceChildren();
  if (!visible.length) {
    const empty = document.createElement("div"); empty.className = "empty";
    empty.textContent = records.length ? "No matching requests." : "No requests yet. API calls will appear here automatically.";
    $("requests").append(empty);
  }
  for (const r of visible) {
    const button = document.createElement("button"); button.className = `request${selected === r.id ? " selected" : ""}`;
    button.dataset.request = r.id; button.setAttribute("aria-label", `Request ${r.id}`); button.setAttribute("aria-pressed", String(selected === r.id));
    const row = document.createElement("div"); row.className = "row";
    const time = document.createElement("span"); time.textContent = new Date(r.started_at * 1000).toLocaleTimeString();
    const tag = document.createElement("span"); tag.className = `tag ${r.state}`; tag.textContent = r.state;
    row.append(time, tag);
    const endpoint = document.createElement("div"); endpoint.className = "endpoint"; endpoint.textContent = `POST ${r.path}`;
    const meta = document.createElement("div"); meta.className = "small";
    meta.textContent = `${r.id} · ${seconds(r.wallclock_s)} · ${r.stream ? "stream" : "JSON response"}${r.http_status ? ` · HTTP ${r.http_status}` : ""}`;
    button.append(row, endpoint, meta);
    button.onclick = async () => { selected = r.id; detail = null; list(); await showDetail(); };
    $("requests").append(button);
    if (focusId === r.id) button.focus({preventScroll:true});
  }
}
function content() {
  if (!detail) return;
  let value = detail[pane] || (pane === "response" && detail.stream ? "Streaming response: see Output and the request error, if any." : "No content.");
  if (pane === "output") { try { value = JSON.stringify(JSON.parse(value), null, 2); } catch (_) {} }
  text("content", value);
  const note = pane === "output" && detail.error ? "Raw model output retained for diagnosis. The API returned an error." : pane === "input" ? "Original request body." : pane === "output" ? "Model answer. JSON is formatted here for readability." : pane === "reasoning" ? "Separate reasoning content, when enabled." : "API response body.";
  text("content-note", note + (detail[`${pane}_truncated`] ? " Monitor capture truncated at 256K characters; the API response was not shortened." : ""));
}
async function showDetail() {
  if (!selected) return;
  const id = selected;
  const value = await api(`/api/requests?id=${encodeURIComponent(id)}`);
  if (selected !== id) return;
  detail = value; $("selection").hidden = false; $("no-selection").hidden = true;
  text("detail-title", `${value.id} · ${value.path}`);
  text("detail-status", value.state); $("detail-status").className = `tag ${value.state}`;
  text("detail-time", new Date(value.started_at * 1000).toLocaleString());
  text("detail-wall", seconds(value.wallclock_s ?? ((Date.now() / 1000) - value.started_at)));
  text("detail-load", seconds(value.load_s)); text("detail-queue", seconds(value.queue_s)); text("detail-first", seconds(value.first_token_s));
  text("detail-prompt", value.usage?.prompt_tokens ?? value.usage?.input_tokens ?? "—");
  text("detail-tokens", value.usage?.completion_tokens ?? value.usage?.output_tokens ?? "—");
  text("detail-speed", speed(value.timings?.predicted_per_second)); text("detail-format", value.response_format || "text");
  $("detail-error").hidden = !value.error; text("detail-error", value.error?.message || ""); content();
}
// M6: the parallel view - the batch slots as the engine holds them, and the requests in flight over four minutes.  It
// hides itself when the server serves one request at a time, so the page looks as it always did there.
const SERIES_MS = 240000;
let series = [], peak = 0;
function slotCells(slots) {
  const el = $("slots"), stamp = JSON.stringify(slots);
  if (el.dataset.stamp === stamp) return;
  el.dataset.stamp = stamp;
  el.replaceChildren();
  for (const s of slots) {
    const cell = document.createElement("div");
    cell.className = "slot" + (s.busy ? " busy" : "");
    cell.setAttribute("role", "status");
    cell.setAttribute("aria-label", "Slot " + s.slot + (s.busy ? " busy" : " idle"));
    const n = document.createElement("b"); n.textContent = s.slot;
    const label = document.createElement("span"); label.textContent = s.busy ? "busy" : "idle";
    cell.append(n, label);
    el.append(cell);
  }
}
function paintChart() {
  const c = $("chart");
  if (!c) return;
  const ctx = c.getContext("2d"), ratio = window.devicePixelRatio || 1, w = c.clientWidth || 600, h = 120;
  if (c.width !== Math.round(w * ratio) || c.height !== Math.round(h * ratio)) {
    c.width = Math.round(w * ratio); c.height = Math.round(h * ratio);
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const cs = getComputedStyle(document.documentElement);
  const line = cs.getPropertyValue("--st-line-soft").trim() || "#2a2f39";
  const accent = cs.getPropertyValue("--st-accent").trim() || "#4f8cff";
  const muted = cs.getPropertyValue("--st-ink-muted").trim() || "#8b93a3";
  const mono = cs.getPropertyValue("--st-font-mono").trim() || "monospace";
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = line; ctx.lineWidth = 1;
  for (let i = 1; i < 4; ++i) {
    const y = Math.round((h - 20) * i / 4) + 10;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  if (!series.length) {
    ctx.fillStyle = muted; ctx.font = "12px " + mono;
    ctx.fillText("Waiting for traffic…", 12, h / 2);
    return;
  }
  const now = Date.now(), t0 = now - SERIES_MS, top = Math.max(1, peak);
  const x = t => Math.max(0, Math.min(w, (t - t0) / SERIES_MS * w));
  const y = n => h - 10 - (n / top) * (h - 20);
  ctx.beginPath(); ctx.moveTo(x(series[0].t), y(0));
  for (const p of series) ctx.lineTo(x(p.t), y(p.inflight));
  ctx.lineTo(x(series[series.length - 1].t), y(0)); ctx.closePath();
  ctx.globalAlpha = 0.18; ctx.fillStyle = accent; ctx.fill(); ctx.globalAlpha = 1;
  ctx.beginPath(); ctx.moveTo(x(series[0].t), y(series[0].inflight));
  for (const p of series) ctx.lineTo(x(p.t), y(p.inflight));
  ctx.strokeStyle = accent; ctx.lineWidth = 2; ctx.stroke();
  ctx.fillStyle = muted; ctx.font = "11px " + mono;
  ctx.fillText("in flight · peak " + top + " · 4 min", 12, 18);
}
function parallel(status, records) {
  const conc = status.concurrency || {}, slots = conc.slots || [];
  const serving = Math.max(1, conc.serving || 1);
  $("parallel").hidden = !(slots.length > 0 || serving > 1);
  if ($("parallel").hidden) return;
  const inflight = typeof status.activity?.in_flight === "number" ? status.activity.in_flight : records.filter(active).length;
  const busy = slots.filter(s => s.busy).length;
  peak = Math.max(peak, inflight, busy);
  series.push({t: Date.now(), inflight: Math.max(inflight, busy)});
  while (series.length && series[0].t < Date.now() - SERIES_MS) series.shift();
  slotCells(slots.length ? slots : Array.from({length: serving}, (_, i) => ({slot: i, busy: false})));
  text("p-inflight", inflight);
  text("p-busy", busy + " / " + (slots.length || serving));
  text("p-waiting", conc.waiting || Math.max(0, inflight - busy));   // (the engine's number, or what the slots do not hold)
  text("p-peak", peak);
  text("parallel-note", serving + " slot(s) · " + (conc.requested ?? serving) + " asked");
  paintChart();
}
window.addEventListener("resize", paintChart);
async function refresh() {
  try {
    const [status, history] = await Promise.all([api("/v1/status"), api("/api/requests")]);
    state = status; records = history.requests;
    status.loaded = history.loaded; status.auto_load = history.auto_load;
    if (selected && !records.some(r => r.id === selected)) {
      selected = detail = null; $("selection").hidden = true; $("no-selection").hidden = false;
    }
    const inflight = Math.max(status.activity.in_flight, records.filter(active).length);
    const loading = records.some(r => r.state === "loading");
    text("loaded", loading ? "Loading…" : inflight ? "In use" : status.loaded ? "Loaded" : "Unloaded");
    $("model-state").className = `value ${inflight || loading ? "busy" : status.loaded ? "loaded" : ""}`;
    text("model", status.model); text("count", records.length);
    text("active", inflight ? `${inflight} active / queued` : status.auto_load ? "Loads automatically on the next request" : "No active requests");
    parallel(status, records);
    const last = records.find(r => !active(r));
    text("wall", last ? seconds(last.wallclock_s) : "—"); text("speed", speed(last?.timings?.predicted_per_second));
    $("load").disabled = operating || Boolean(inflight) || status.loaded; $("unload").disabled = operating || Boolean(inflight) || !status.loaded;
    list(); if (selected) await showDetail();
    text("updated", `Live · updated ${new Date().toLocaleTimeString()}`);
    if (!operating) notice(actionError);
  } catch (error) { notice(error.message); text("updated", "Disconnected · retrying"); $("load").disabled = $("unload").disabled = true; }
}
async function control(load) {
  actionError = "";
  operating = true; $("load").disabled = $("unload").disabled = true;
  text("loaded", load ? "Loading…" : "Unloading…");
  try { await api(load ? "/load" : "/unload", {}); notice(""); }
  catch (error) { actionError = error.message; notice(actionError); }
  finally { operating = false; await refresh(); }
}
$("load").onclick = () => control(true); $("unload").onclick = () => control(false);
$("key").onclick = () => { $("auth").hidden = !$("auth").hidden; if (!$("auth").hidden) $("api-key").focus(); };
$("auth").onsubmit = async event => { event.preventDefault(); apiKey = $("api-key").value.trim(); sessionStorage.setItem("strata.monitor.key", apiKey); $("auth").hidden = true; await refresh(); };
$("theme").onclick = () => { const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = theme; localStorage.setItem("strata.theme", theme); };
$("search").oninput = list; $("filter").onchange = list;
document.querySelectorAll("[data-pane]").forEach(button => { button.onclick = () => { pane = button.dataset.pane; document.querySelectorAll("[data-pane]").forEach(b => b.setAttribute("aria-selected", String(b === button))); content(); }; });
$("copy").onclick = async () => { try { await navigator.clipboard.writeText($("content").textContent); text("copy", "Copied"); setTimeout(() => text("copy", "Copy"), 1500); } catch (_) { notice("Clipboard unavailable; select the text to copy it."); } };
async function poll() { await refresh(); setTimeout(poll, 2000); }
poll();
