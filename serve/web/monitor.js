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
// Context in VRAM: the tokens each engine cache keeps, read on request (it can be 100K+ tokens, so it is not polled)
let ctx = null, ctxView = "plain", ctxPage = -1, ctxPicked = false;   // ctxPage -1: the last page (the newest tokens)
const UNKNOWN = {busy: "In use: its tokens are shown when its request ends.",
  stopped: "Not known: the last request was stopped while its prompt was read.",
  restored: "Not known: the cache was restored from a disk session. The next request shows it again.",
  dropped: "Empty: the engine dropped its cache after non-finite logits.",
  ended: "Not known: the engine did not finish the last request."};
// The colour comes from the id, so the same token always has the same colour: a hash of the id picks one of a fixed
// palette per theme.  The palette samples every allowed colour uniformly by volume in OKLab (roughly perceptual), so
// muted and very light or dark colours, which are a smaller part of that space, get fewer entries.  Allowed: inside
// sRGB, a WCAG contrast of 7:1 or more against the theme's text, and clear of the page background (the lightness
// range).  A seeded generator makes the palette the same on every load.
const PALETTE_SIZE = 4096;
const THEMES = {light: {text: [0x11, 0x11, 0x11], L: [0.66, 0.93]}, dark: {text: [0xee, 0xf1, 0xf3], L: [0.25, 0.50]}};
const toLinear = c => (c /= 255) <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
const toSrgb = c => Math.round(255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055));
function oklabToLinear(L, a, b) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3,
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3;
  return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s];
}
function makePalette({text, L: [lo, hi]}) {
  const lum = ([r, g, b]) => 0.2126 * r + 0.7152 * g + 0.0722 * b, ink = lum(text.map(toLinear));
  let seed = 1;
  const random = () => {                              // mulberry32
    let t = seed = (seed + 0x6D2B79F5) | 0;
    t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
  const palette = [];
  while (palette.length < PALETTE_SIZE) {
    const rgb = oklabToLinear(lo + (hi - lo) * random(), 0.5 * random() - 0.25, 0.5 * random() - 0.25);
    if (rgb.some(c => c < 0 || c > 1)) continue;
    const srgb = rgb.map(toSrgb), y = lum(srgb.map(toLinear));   // the contrast of the colour as it is drawn
    if ((Math.max(y, ink) + 0.05) / (Math.min(y, ink) + 0.05) < 7) continue;
    palette.push(`rgb(${srgb.join(" ")})`);
  }
  return palette;
}
let palettes = null;                                  // made when the colored view is first shown
const tokenColour = id => (Math.imul(id, 2654435761) >>> 0) >>> 20;   // the hash's top 12 bits: an entry of 4096
const PAGE = 8000;                                    // tokens per page of the colored view: the page stays light
function ctxSlot() { return ctx?.slots?.[Number($("ctx-slot").value) || 0] || null; }
function showContext() {
  const view = $("ctx-view"), slot = ctxSlot();
  $("ctx-save").disabled = !slot?.tokens?.length; $("ctx-pager").hidden = true;
  if (!ctx) return;
  view.replaceChildren(); text("ctx-note", "");
  if (!ctx.available) { text("ctx-sum", ctx.loaded ? "This engine does not report its cache." : "The model is not loaded."); return; }
  if (!slot) { text("ctx-sum", "No cache."); return; }
  if (slot.tokens === null) { text("ctx-sum", `${slot.name} · ${UNKNOWN[slot.unknown] || "Not known."}`); return; }
  const n = slot.tokens.length;
  text("ctx-sum", `${n.toLocaleString()} of ${ctx.max_context.toLocaleString()} tokens · ${slot.name} · ${slot.state === "idle" ? "idle" : "in use, still growing"}`);
  text("ctx-note", (ctxView === "colored" ? "Each token's colour comes from its id, so the same token always has the same colour. Hover a token for its position and id. " : "") +
    "The last prompt and the answer to it, as the server sent and read them. The last token of the answer is not in the cache; with draft tokens (MTP) a few more can be.");
  if (ctxView === "plain") { view.textContent = slot.pieces.join(""); return; }
  const pages = Math.max(1, Math.ceil(n / PAGE));
  ctxPage = Math.min(Math.max(ctxPage < 0 ? pages - 1 : ctxPage, 0), pages - 1);
  const start = ctxPage * PAGE, end = Math.min(n, start + PAGE);
  $("ctx-pager").hidden = pages < 2; $("ctx-prev").disabled = ctxPage === 0; $("ctx-next").disabled = ctxPage === pages - 1;
  text("ctx-range", `Tokens ${start.toLocaleString()}–${Math.max(start, end - 1).toLocaleString()} (page ${ctxPage + 1} of ${pages})`);
  palettes ??= {light: makePalette(THEMES.light), dark: makePalette(THEMES.dark)};
  const frag = document.createDocumentFragment();     // one append: one layout
  for (let i = start; i < end; i++) {
    const id = slot.tokens[i], k = tokenColour(id), span = document.createElement("span"); span.className = "tok";
    span.style.setProperty("--bl", palettes.light[k]); span.style.setProperty("--bd", palettes.dark[k]);   // both themes
    span.textContent = slot.pieces[i]; span.title = `token #${i} · id ${id}`;
    frag.append(span);
  }
  view.append(frag);
}
async function loadContext() {
  $("ctx-refresh").disabled = true; text("ctx-sum", "Reading…");
  try {
    ctx = await api("/api/context");
    const select = $("ctx-slot"), keep = select.value;
    select.replaceChildren(...ctx.slots.map((s, i) => { const o = document.createElement("option"); o.value = String(i); o.textContent = `${s.name} · ${s.tokens ? `${s.tokens.length.toLocaleString()} tokens` : s.state}`; return o; }));
    // the cache a request used last (with --batch a request alone runs on the solo session, others in slots)
    const latest = ctx.slots.reduce((best, s, i) => (s.used || 0) > (ctx.slots[best].used || 0) ? i : best, 0);
    select.value = ctxPicked && Number(keep) < ctx.slots.length ? keep : String(latest);
    select.hidden = ctx.slots.length < 2;
    ctxPage = -1; showContext();
  } catch (error) { text("ctx-sum", error.message); }
  finally { $("ctx-refresh").disabled = false; }
}
function saveContext() {
  const slot = ctxSlot(); if (!slot?.tokens) return;
  const binary = $("ctx-binary").checked, parts = [slot.pieces.join("")];
  if (binary) {                                       // the ids are the file's last 4 x N bytes
    const ids = new DataView(new ArrayBuffer(slot.tokens.length * 4));
    slot.tokens.forEach((id, i) => ids.setUint32(i * 4, id, true));
    parts.push(`\n\n----- ${slot.tokens.length} token ids, uint32 little-endian, to the end of the file -----\n`, ids.buffer);
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(parts, {type: binary ? "application/octet-stream" : "text/plain;charset=utf-8"}));
  a.download = `strata-context-${slot.slot === null ? "session" : `slot${slot.slot}`}-${new Date().toISOString().replace(/[:.]/g, "-")}.txt`;
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
$("ctx-refresh").onclick = loadContext; $("ctx-save").onclick = saveContext;
$("ctx-slot").onchange = () => { ctxPicked = true; ctxPage = -1; showContext(); };
$("ctx-prev").onclick = () => { ctxPage--; showContext(); }; $("ctx-next").onclick = () => { ctxPage++; showContext(); };
document.querySelectorAll("[data-view]").forEach(button => { button.onclick = () => { ctxView = button.dataset.view; document.querySelectorAll("[data-view]").forEach(b => b.setAttribute("aria-selected", String(b === button))); showContext(); }; });
