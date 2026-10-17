// ------------------------------------------------------------------ language (#2686)
// The same packs the main page uses (serve/web/i18n/<lang>.json, UTF-8).  The chosen language is the one the main
// page saved, else the browser's own; this page has no switch of its own, so it follows.
const LANGS = [{code: "en", name: "English", dir: "ltr"}];
let lang = "en", pack = {};
const BASE = new Map(Object.entries({}));
function t(key, vars) {
  const s = pack[key] !== undefined ? pack[key] : BASE.get(key);
  if (s === undefined || s === null) return key;
  if (!vars) return String(s);
  return String(s).replace(/\{(\w+)\}/g, (m, k) => (vars[k] == null ? "" : vars[k]));
}
function setWords(el, value) {
  if (!el) return;
  if (el.dataset.words === String(value) && el.textContent === String(value)) return;
  el.textContent = value; el.dataset.words = value;
}
function applyStatic(root) {
  for (const el of root.querySelectorAll("[data-i18n]")) setWords(el, t(el.dataset.i18n));
  for (const el of root.querySelectorAll("[data-i18n-aria]")) el.setAttribute("aria-label", t(el.dataset.i18nAria));
  for (const el of root.querySelectorAll("[data-i18n-title]")) el.title = t(el.dataset.i18nTitle);
  for (const el of root.querySelectorAll("[data-i18n-placeholder]")) el.placeholder = t(el.dataset.i18nPlaceholder);
}
function browserLang() {
  const wanted = [...(navigator.languages || [navigator.language]).filter(Boolean)].map(String);
  for (const w of wanted) {
    const lower = w.toLowerCase();
    for (const l of LANGS) {
      if (l.code.toLowerCase() === lower || l.code.toLowerCase() === lower.split("-")[0]) return l.code;
    }
  }
  return "en";
}
async function loadLangs() {
  try {
    const r = await fetch("/i18n");
    if (!r.ok) return;
    const list = (await r.json()).languages || [];
    if (list.length) { LANGS.length = 0; LANGS.push(...list); }
  } catch (e) { /* an older server: English only */ }
}
async function setLang(code) {
  code = code || "en";
  try {
    const r = await fetch("/web/i18n/" + encodeURIComponent(code) + ".json");
    pack = r.ok ? JSON.parse(new TextDecoder("utf-8").decode(await r.arrayBuffer())) : {};
  } catch (e) { pack = {}; code = "en"; }
  lang = code;
  document.documentElement.lang = lang;
  applyStatic(document);
  stamp = "";                               // the wording changed: the list's own cache must not hide it
  list();                                   // the empty state is translated text too
  if (selected) showDetail();
}
let langReady = null;
function startLanguage() {
  const saved = localStorage.getItem("strata.lang");
  const wanted = saved ? saved.replace(/^"|"$/g, "") : null;
  langReady = loadLangs().then(() => setLang(wanted || browserLang()), () => setLang("en"));
  return langReady;
}

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
    empty.textContent = records.length ? t("apimon.noMatch") : t("apimon.noRequests");
    $("requests").append(empty);
  }
  for (const r of visible) {
    const button = document.createElement("button"); button.className = `request${selected === r.id ? " selected" : ""}`;
    button.dataset.request = r.id; button.setAttribute("aria-label", `${t("apimon.request")} ${r.id}`); button.setAttribute("aria-pressed", String(selected === r.id));
    const row = document.createElement("div"); row.className = "row";
    const time = document.createElement("span"); time.textContent = new Date(r.started_at * 1000).toLocaleTimeString();
    const tag = document.createElement("span"); tag.className = `tag ${r.state}`; tag.textContent = r.state;
    row.append(time, tag);
    const endpoint = document.createElement("div"); endpoint.className = "endpoint"; endpoint.textContent = `POST ${r.path}`;
    const meta = document.createElement("div"); meta.className = "small";
    meta.textContent = `${r.id} · ${seconds(r.wallclock_s)} · ${r.stream ? t("apimon.stream") : t("apimon.jsonResponse")}${r.http_status ? ` · HTTP ${r.http_status}` : ""}`;
    button.append(row, endpoint, meta);
    button.onclick = async () => { selected = r.id; detail = null; list(); await showDetail(); };
    $("requests").append(button);
    if (focusId === r.id) button.focus({preventScroll:true});
  }
}
function content() {
  if (!detail) return;
  let value = detail[pane] || (pane === "response" && detail.stream ? t("apimon.streamingResponse") : t("apimon.noContent"));
  if (pane === "output") { try { value = JSON.stringify(JSON.parse(value), null, 2); } catch (_) {} }
  text("content", value);
  const note = pane === "output" && detail.error ? t("apimon.noteRaw") : pane === "input" ? t("apimon.noteInput") : pane === "output" ? t("apimon.noteOutput") : pane === "reasoning" ? t("apimon.noteReasoning") : t("apimon.noteResponse");
  text("content-note", note + (detail[`${pane}_truncated`] ? " " + t("apimon.noteTruncated") : ""));
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
  text("detail-speed", speed(value.timings?.predicted_per_second)); text("detail-format", value.response_format || t("apimon.textFormat"));
  $("detail-error").hidden = !value.error; text("detail-error", value.error?.message || ""); content();
}
async function refresh() {
  try {
    const [status, history] = await Promise.all([api("/v1/status"), api("/api/requests")]);
    return await apply(status, history);
  } catch (error) { notice(error.message); text("updated", t("apimon.disconnected")); $("load").disabled = $("unload").disabled = true; }
}
async function apply(status, history) {
  try {
    state = status; records = history.requests;
    status.loaded = history.loaded; status.auto_load = history.auto_load;
    if (selected && !records.some(r => r.id === selected)) {
      selected = detail = null; $("selection").hidden = true; $("no-selection").hidden = false;
    }
    const inflight = Math.max(status.activity.in_flight, records.filter(active).length);
    const loading = records.some(r => r.state === "loading");
    text("loaded", loading ? t("apimon.loading") : inflight ? t("apimon.inUse") : status.loaded ? t("apimon.loaded") : t("apimon.unloaded"));
    $("model-state").className = `value ${inflight || loading ? "busy" : status.loaded ? "loaded" : ""}`;
    text("model", status.model); text("count", records.length);
    text("active", inflight ? t("apimon.activeQueued", {n: inflight}) : status.auto_load ? t("apimon.autoLoad") : t("apimon.noActive"));
    const last = records.find(r => !active(r));
    text("wall", last ? seconds(last.wallclock_s) : "—"); text("speed", speed(last?.timings?.predicted_per_second));
    $("load").disabled = operating || Boolean(inflight) || status.loaded; $("unload").disabled = operating || Boolean(inflight) || !status.loaded;
    list(); if (selected) await showDetail();
    text("updated", t("apimon.live", {time: new Date().toLocaleTimeString()}));
    if (!operating) notice(actionError);
  } catch (error) { notice(error.message); text("updated", t("apimon.disconnected")); $("load").disabled = $("unload").disabled = true; }
}
async function control(load) {
  actionError = "";
  operating = true; $("load").disabled = $("unload").disabled = true;
  text("loaded", load ? t("apimon.loading") : t("apimon.unloading"));
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
$("copy").onclick = async () => { try { await navigator.clipboard.writeText($("content").textContent); text("copy", t("apimon.copied")); setTimeout(() => text("copy", t("chat.copy")), 1500); } catch (_) { notice(t("apimon.clipboard")); } };
// #2687: one long-lived GET /api/requests/stream instead of asking every 2 s.  A page that rewrote its own text
// twice a second fought Chrome's page translation (the UI flipped back to English); the server now sends an
// event only when what this page draws from changed, and nothing while nothing does.  A build without the route
// (an older server) never delivers an event, so the page keeps asking every 2 s - what it did before.
let stream = null, streamGot = false, pollTimer = null;
function openStream() {
  if (stream || pollTimer) return;
  let es;
  try { es = new EventSource("/api/requests/stream"); } catch (error) { startPolling(); return; }
  stream = es;
  es.addEventListener("requests", event => {
    streamGot = true;
    try { apply(JSON.parse(event.data).status, {requests: JSON.parse(event.data).requests || [], loaded: state?.loaded, auto_load: state?.auto_load}); } catch (error) {}
  });
  es.onerror = () => {
    if (!streamGot && es.readyState === EventSource.CLOSED) { stream = null; startPolling(); return; }
    notice("Disconnected · retrying");
  };
}
function startPolling() {
  if (pollTimer) return;
  if (stream) { stream.close(); stream = null; }
  const tick = async () => {
    await refresh();
    if (pollTimer) pollTimer = setTimeout(tick, 2000);
  };
  pollTimer = setTimeout(tick, 0);
}
// the pack first (so the first paint is already translated), then the data
startLanguage().then(() => refresh());
openStream();
