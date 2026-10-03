/* gui/web/app.js - the unified shell of the Strata Manager page.  It runs AFTER serve/web/app.js
   (loaded as web/serve-app.js), which already implements Chat, Monitor, About, Markdown, the sampling
   drawer and the theme - untouched, against the SAME origin (the Manager's port proxies the Strata
   server's endpoints, so no CORS and no copied code).  This file adds:

     - the fourth tab (Manager): the model editor, lifecycle buttons, models list, logs, system info
     - the hash-based navigation over manager|chat|monitor|about (default #manager); Start/Restart never
       change the current page
     - the offline awareness: while Strata is stopped/starting/restarting, Chat/Monitor/About show a
       clean offline state and the model APIs are not polled; when Strata answers again (also when it
       was started externally), everything resumes by itself
     - the lifecycle pill: "External" detection disables Start/Stop/Restart so the Manager never owns
       a process it did not start

   It deliberately does NOT re-declare anything serve/web/app.js already provides ($, store, tab, busy,
   toast, poll, showTab, setPill, setBusy, ...): those live in the shared global scope.  The few places
   where this shell must change their behavior are patched through window.* after capturing the originals.
*/
"use strict";

/* ---------------------------------------------------------------- lifecycle bookkeeping */
const MG_POLL_MS = 2000;
const mgState = { models: [], editor: null, status: { server: { state: "unknown" } }, tool: null,
                  busy: false };
// the running operation (Stop / Force Stop / Restart): shown immediately on click, updated by polling.
const mgOp = { state: null, phase: null, since: 0, ticker: null };
let mgOnline = false;                        // is Strata's server answering? (running or external)

const MG_PILL_TEXT = { running: "Running", starting: "Starting…", stopping: "Stopping…",
                       restarting: "Restarting…", external: "Running · External",
                       error: "Error", stopped: "Stopped", unknown: "…" };
const MG_PILL_DOT = { running: "generating", external: "generating", starting: "queued",
                      stopping: "queued", restarting: "queued", error: "queued",
                      stopped: "idle", unknown: "queued" };

/* ---------------------------------------------------------------- offline gating
   The Strata APIs (health, metrics, mcp, settings, v1/*, ...) run on THIS origin through the Manager
   gateway.  While Strata is down they are not worth polling: reject them before they hit the wire, so
   the serve/web/app.js poll/send loops just see a failed request and this shell paints the lifecycle
   pill instead.  The Manager's own /api/* stays reachable always - that is what keeps the page alive. */
function mgIsManagerUrl(url) {
  return typeof url === "string" && (url.startsWith("/api/") || url.startsWith("/web/") ||
         url.startsWith("/fonts/"));
}
const _baseFetch = window.fetch;
window.fetch = function (url, opts) {
  if (!mgOnline && !mgIsManagerUrl(url)) return Promise.reject(new TypeError("Strata is not running"));
  return _baseFetch(url, opts);
};

const _baseSetPill = window.setPill;
window.setPill = function (state, text) {
  if (!mgOnline) {                               // the model state is meaningless: show the lifecycle
    const st = mgState.status.server ? mgState.status.server.state : "unknown";
    $("pill").dataset.state = MG_PILL_DOT[st] || "idle";
    $("pill-text").textContent = MG_PILL_TEXT[st] || st;
    return;
  }
  _baseSetPill(state, text);
};

const _basePoll = window.poll;
window.poll = function () {
  if (!mgOnline) { setTimeout(window.poll, 1000); return; }
  _basePoll();
};

const _baseSetBusy = window.setBusy;
window.setBusy = function (on) {
  _baseSetBusy(on);
  if (!mgOnline && $("send-btn")) $("send-btn").disabled = true;
};

/* ---------------------------------------------------------------- navigation
   The shell's own tab logic (serve/app.js's is Chat/Monitor/About only).  The selected page is kept in
   the hash (#manager default) and remembered, and nothing here ever navigates away - Start / Restart
   leave the user exactly on the page they are viewing. */
const _baseShowTab = window.showTab;
window.showTab = function (name) {
  tab = ["manager", "chat", "monitor", "about"].includes(name) ? name : "manager";
  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of ["manager", "chat", "monitor", "about"]) { const el = $(`view-${v}`); if (el) el.hidden = v !== tab; }
  if (location.hash.slice(1) !== tab) history.replaceState(null, "", `#${tab}`);
  store.set("lasttab", tab);
  if (tab === "chat") $("input").focus();
  if (tab === "monitor") loadMcp();
  if (lastMetrics) render(lastMetrics);
};
// serve/web/app.js already registered window's hashchange -> showTab(hash); it now resolves to this shell.

/* ---------------------------------------------------------------- API */
async function api(path, opts) {
  const r = await fetch(path, Object.assign({ method: "GET", headers: { "Accept": "application/json" } }, opts));
  let j = {};
  try { j = await r.json(); } catch (e) { /* not json */ }
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  if (j.ok === false) throw new Error(j.error || "request failed");
  return j.data !== undefined ? j.data : j;
}

function post(path, body) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" },
                     body: JSON.stringify(body) });
}

/* ---------------------------------------------------------------- toasts (the serve app's style) */
function mgToast(msg, kind) {
  toast(kind === "ok" ? "success" : kind === "error" ? "error" : "info", msg);
}

/* ---------------------------------------------------------------- system + lifecycle display */
async function mgRefreshSystem() {
  try {
    const s = await api("/api/system");
    const g = (s.gpus || []).map(g =>
      `${g.name} · ${Math.round(g.vram_used_gb || 0)}/${g.vram_gb.toFixed(0)} GB VRAM`).join(" · ");
    const ram = s.ram_gb ? ` · RAM ${s.ram_used_gb != null ? Math.round(s.ram_used_gb) + "/" : ""}${Math.round(s.ram_gb)} GB` : "";
    $("sys-line").textContent = `Strata v${s.strata_version} · ${s.cpu || "?"} · ${g || "no NVIDIA GPU"}${ram}`;
  } catch (e) { $("sys-line").textContent = "GPU / RAM unavailable: " + e.message; }
}

function mgSetPill() {
  const srv = mgState.status.server || { state: "unknown" };
  const p = $("state-pill");
  if (!p) return;
  p.dataset.state = srv.state || "unknown";
  const note = srv.state === "running" ? `port ${srv.port}`
              : srv.state === "external" ? "started outside the Manager" : "";
  p.textContent = (MG_PILL_TEXT[srv.state] || srv.state) + (note ? ` — ${note}` : "");
}

function applyOnline() {
  const st = mgState.status.server ? mgState.status.server.state : "unknown";
  mgOnline = st === "running" || st === "external";
  for (const id of ["chat-offline", "monitor-offline", "about-offline"]) {
    const el = $(id); if (el) el.hidden = mgOnline;
  }
  if ($("input")) $("input").disabled = !mgOnline;
  if ($("send-btn")) $("send-btn").disabled = !mgOnline || !!busy;
  for (const id of ["attach-btn", "new-btn", "export-btn", "sampling-btn"]) {
    const el = $(id); if (el) el.disabled = !mgOnline;
  }
  const ex = $("external-note");
  if (ex) ex.hidden = st !== "external";
  window.setPill("idle", "");                        // repaint the header pill now, not on the next tick
}

/* ---------------------------------------------------------------- the running operation display */
const MG_OP_PHASE = {
  stopping: { stopping: "Stopping Strata…", graceful: "Gracefully shutting down the engine", forcing: "Force-stopping…" },
  restarting: { stopping: "Restarting… (stopping)", starting: "Restarting… (starting)", running: "Restarting… (running)" },
};
const MG_OP_TITLE = { stopping: "Stopping Strata", restarting: "Restarting Strata" };

function setOp(stateName, phase, elapsed) {
  if (mgOp.state !== stateName || mgOp.phase !== phase) {
    mgOp.state = stateName; mgOp.phase = phase;
    mgOp.since = Date.now() - (elapsed != null ? elapsed * 1000 : 0);
  } else if (elapsed != null) {
    mgOp.since = Date.now() - elapsed * 1000;        // keep in step with the backend's clock
  }
  $("op").hidden = false;
  if (mgOp.ticker == null) mgOp.ticker = setInterval(opTick, 250);
  opTick();
}

function opTick() {
  const table = MG_OP_PHASE[mgOp.state] || {};
  const base = table[mgOp.phase] || MG_OP_TITLE[mgOp.state] || "Working…";
  const s = Math.max(0, (Date.now() - mgOp.since) / 1000);
  $("op-text").textContent = `${base}  ${s.toFixed(1)}s`;
}

function clearOp() {
  mgOp.state = null; mgOp.phase = null;
  const op = $("op"); if (op) op.hidden = true;
  if (mgOp.ticker != null) { clearInterval(mgOp.ticker); mgOp.ticker = null; }
}

function fmtElapsed(secs) { return secs != null ? `${secs.toFixed(1)}s` : ""; }

/* ---------------------------------------------------------------- status + buttons */
function refreshButtons() {
  const s = mgState.status.server || { state: "unknown" };
  const st = s.state;
  const inOp = st === "stopping" || st === "restarting";
  const external = st === "external";
  $("start").disabled = mgState.busy || inOp || external || st === "running" || st === "starting" || !mgState.models.length;
  $("stop").disabled = mgState.busy || inOp || external || (st !== "running" && st !== "starting");
  $("restart").disabled = mgState.busy || inOp || external || st !== "running";
  $("force-stop").hidden = !inOp;
  $("save").disabled = mgState.busy || !mgState.editor;
}

async function refreshStatus() {
  try {
    const st = await api("/api/status");
    const prev = mgState.status.server ? mgState.status.server.state : "unknown";
    mgState.status = st;
    mgState.tool = st.tool;
    const srv = st.server || { state: "unknown" };

    if (srv.state === "stopping" || srv.state === "restarting") {
      setOp(srv.state, srv.phase, srv.elapsed);
    } else {
      if (mgOp.state === "restarting" && srv.state === "running") {
        const secs = Math.max(0, (Date.now() - mgOp.since) / 1000);
        clearOp();
        mgToast(`Restarted in ${secs.toFixed(1)}s — the model is serving again (port ${srv.port})`, "ok");
      } else if (mgOp.state === "restarting" || mgOp.state === "stopping") {
        clearOp();                                    // stopped (or crashed) - the response already said why
      }
      if (srv.state === "error") {
        clearOp();
        mgToast((srv.error || "the server exited unexpectedly") + " — see the Server log", "error");
      }
    }

    if (prev !== "external" && srv.state === "external") {
      mgToast("Strata is running, started outside the Manager. Close its console to stop it.", "info");
    }
    if (prev !== "running" && srv.state === "running" && mgOp.state == null) {
      mgToast(`Strata is running (port ${srv.port}).`, "ok");
    }
    if (prev === "external" && srv.state === "stopped") {
      mgToast("The external Strata instance stopped.", "info");
    }

    mgSetPill();
    applyOnline();
    refreshButtons();
    handleLog();
    handleTool();
  } catch (e) {
    mgSetPill();
    refreshButtons();
  }
}

/* ---------------------------------------------------------------- models list */
function renderModels(defaultName) {
  const ul = $("models");
  ul.textContent = "";
  const tpl = $("tpl-model");
  for (const m of mgState.models) {
    const li = tpl.content.firstElementChild.cloneNode(true);
    li.className = "model";
    li.querySelector(".model__title").textContent = m.title;
    li.querySelector(".badge.quant").textContent = m.quant || m.family || "?";
    if (m.error) {
      li.querySelector(".model__meta").textContent = "unreadable: " + m.error;
      ul.appendChild(li);
      continue;
    }
    const cur = li.querySelector(".badge.current");
    cur.hidden = !(defaultName && m.config === defaultName);
    const meta = [];
    meta.push(`family ${m.family || "?"}`);
    if (m.variant) meta.push(`variant ${m.variant}`);
    meta.push(`context ${m.context ? m.context.toLocaleString() : "?"}`);
    if (m.kv) meta.push(`KV ${m.kv}`);
    meta.push(`vision ${m.vision || "off"}`);
    meta.push(m.low_ram !== "off" ? `low-RAM ${m.low_ram}` : "RAM-resident");
    meta.push(`port ${m.port}`);
    if (m.layer_split) meta.push(`GPUs ${Array.isArray(m.gpu) ? m.gpu.join("+") : m.gpu}`);
    if (!m.gguf_ok) meta.push("⚠ missing files");
    li.querySelector(".model__meta").textContent = meta.join("  ·  ");
    const gg = li.querySelector(".model__gguf");
    gg.textContent = m.gguf && m.gguf[0] ? m.gguf[0] : (m.pack ? "pack: " + m.pack : "");
    li.querySelector("button.select").addEventListener("click", () => selectModel(m.config));
    li.querySelector("button.edit").addEventListener("click", () => {
      selectModel(m.config); window.scrollTo({ top: 0, behavior: "smooth" });
    });
    ul.appendChild(li);
  }
  ul.hidden = !mgState.models.length;
  if (!mgState.models.length) {
    const li = document.createElement("li");
    li.className = "model";
    li.innerHTML = `<div class="model__title">None yet</div>
      <div class="model__meta">Install a model with SETUP.bat / setup.sh first — the Manager manages, it does not install.</div>`;
    ul.appendChild(li);
    ul.hidden = false;
  }
}

function renderModelSelect(defaultName) {
  const sel = $("model-sel");
  sel.textContent = "";
  for (const m of mgState.models) {
    const o = document.createElement("option");
    o.value = m.config;
    o.textContent = `${m.title}  (${m.quant || m.family || "?"} · ${m.context ? m.context.toLocaleString() : "?"} ctx)`;
    sel.appendChild(o);
  }
  sel.value = defaultName && mgState.models.length ? defaultName : (mgState.models[0] || {}).config || "";
}

/* ---------------------------------------------------------------- editor */
function fillEditor() {
  const e = mgState.editor, s = e.summary;
  $("ctx-num").value = e.context || 32768;
  renderChips(e);
  updateCtxHint();
  const vis = e.vision || "off";
  for (const r of document.querySelectorAll('input[name="vision"]')) r.checked = r.value === vis;
  $("lowram").checked = e.low_ram;
  $("lowram-hint").textContent = e.low_ram
    ? (e.low_ram_resident ? "resident: the experts the GPU does not hold stay in RAM"
                         : "mapped: the experts are read from the model folder through the OS file cache")
    : "";
  $("kv-sel").value = (e.kv_options && e.kv_options.length && (e.kv === "fp16" ? "int8" : e.kv)) || "int8";
  $("kv-hint").textContent = e.kv_options && e.kv_options.length
    ? (e.kv === "fp16" ? "" : "above 8K context; the engine streams it from 64K up")
    : "below 8K context the engine uses fp16 (no flag)";
  $("kv-sel").disabled = !(e.kv_options && e.kv_options.length);
  fillGpu(e);
  $("port-num").value = e.port || 8080;
  $("host-sel").value = (e.host === "0.0.0.0") ? "0.0.0.0" : "127.0.0.1";
  $("mg-api-key").value = e.api_key || "";
  $("vision-hint").textContent = e.vision_available ? "" : "Vision not installed for this model — run SETUP.bat --vision to add it (the Manager never downloads).";
  for (const r of document.querySelectorAll('input[name="vision"]')) r.disabled = !e.vision_available;
  const firstGguf = s.gguf && s.gguf[0];
  if (firstGguf && !$("gguf-path").value) {
    $("gguf-path").value = firstGguf.replace(new RegExp("/?[^/\\\\]+$"), "");
    ggufPathChanged();   // the field is the source of truth: detect what it was just filled with
  }
}

function fillGpu(e) {
  const sel = $("gpu-sel");
  const layerSplit = Array.isArray(e.gpu);
  sel.textContent = "";
  const auto = document.createElement("option"); auto.value = "auto"; auto.textContent = "Auto (Strata's choice)";
  sel.appendChild(auto);
  for (const g of e.gpus || []) {
    const o = document.createElement("option");
    o.value = String(g.index);
    o.textContent = `GPU ${g.index} — ${g.name} (${g.vram_gb.toFixed(0)} GB)`;
    sel.appendChild(o);
  }
  if (layerSplit) {
    sel.disabled = true;
    $("gpu-hint").textContent = `layer split across ${e.gpu.length} GPUs — change with SETUP.bat --gpus`;
  } else {
    sel.disabled = false;
    sel.value = e.gpu !== undefined && e.gpu !== null ? String(e.gpu) : "auto";
    $("gpu-hint").textContent = "which card runs the model";
  }
}

function renderChips(e) {
  const box = $("ctx-chips");
  box.textContent = "";
  for (const c of e.presets) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.dataset.ctx = String(c);
    const k = c / 1024;
    b.textContent = (k >= 1024 ? k / 1024 : k) + (k >= 1024 ? "M" : "K") + (c > e.trained_context ? " ⚠" : "");
    b.title = c.toLocaleString() + (c > e.trained_context ? " tokens — past the trained 262144; the setup adds yarn rope scaling" : " tokens");
    b.addEventListener("click", () => {
      $("ctx-num").value = String(c);
      syncChips();
      updateCtxHint();
    });
    box.appendChild(b);
  }
  syncChips();
}

function syncChips() {
  const v = $("ctx-num").value;
  for (const b of $("ctx-chips").querySelectorAll("button.chip")) b.setAttribute("aria-checked", b.dataset.ctx === v ? "true" : "false");
}

function updateCtxHint() {
  const v = parseInt($("ctx-num").value, 10) || 0;
  $("ctx-hint").textContent = v > mgState.editor.trained_context
    ? "past the trained 262144: the setup adds yarn rope scaling (factor " + (v / 262144).toFixed(2) + ")"
    : v > 0 ? "tokens the model can see at once" : "";
  if (mgState.editor) {
    $("kv-hint").textContent = v > 8192 ? "above 8K context; the engine streams it from 64K up" : "below 8K context the engine uses fp16 (no flag)";
    $("kv-sel").disabled = !(v > 8192);
  }
}

async function selectModel(name) {
  if (!name) return;
  mgState.busy = true;
  try {
    const e = await api("/api/config?config=" + encodeURIComponent(name));
    mgState.editor = e;
    fillEditor();
    $("model-sel").value = name;
  } catch (err) { mgToast(err.message, "error"); }
  mgState.busy = false;
  refreshButtons();
}

function editorChanges() {
  const e = mgState.editor;
  const board = {
    config: e.summary.config,
    context: parseInt($("ctx-num").value, 10) || e.context,
    kv: $("kv-sel").value,
    vision: [...document.querySelectorAll('input[name="vision"]')].find(r => r.checked)?.value || "off",
    low_ram: $("lowram").checked,
    port: parseInt($("port-num").value, 10) || e.port,
    host: $("host-sel").value,
    api_key: $("mg-api-key").value.trim(),
    gpu: $("gpu-sel").value,
  };
  if (Array.isArray(e.gpu)) delete board.gpu;      // a layer split is not edited here
  return board;
}

async function doSave() {
  if (!mgState.editor) return;
  mgState.busy = true; refreshButtons();
  try {
    await post("/api/save", editorChanges());
    mgToast("saved — Strata reads the same config file format", "ok");
    await reloadModels();
    await selectModel(mgState.editor.summary.config);
  } catch (err) { mgToast(err.message, "error"); }
  mgState.busy = false; refreshButtons();
}

/* ---------------------------------------------------------------- start / stop / restart
   All of these only talk to the Manager API on this origin - nothing ever opens another URL, another
   tab, or navigates away: the polls below walk Stopping/Restarting/Starting/Running on the SAME page. */
async function doStart() {
  if (!mgState.editor) return;
  mgState.busy = true; refreshButtons();
  try {
    const r = await post("/api/start", { config: mgState.editor.summary.config,
                                         port: parseInt($("port-num").value, 10) || undefined,
                                         open_chat: false });
    if (r.error) { mgToast(r.error, "error"); }
    else mgToast("starting Strata — the model loads in the background", "ok");
    await refreshStatus();
  } catch (err) { mgToast(err.message, "error"); }
  mgState.busy = false; refreshButtons();
}

async function doStop() {
  if (!mgState.editor) return;
  setOp("stopping", "graceful");                       // react instantly: don't wait for the API call
  refreshButtons();
  try {
    const r = await post("/api/stop");
    const t = r.elapsed_s != null ? `Stopped in ${fmtElapsed(r.elapsed_s)}` : "Stopped";
    mgToast(r.forced ? `${t} — the graceful shutdown timed out, so the process tree was force-closed`
                     : `${t} — graceful shutdown`, "ok");
  } catch (err) { mgToast(err.message, "error"); }
  clearOp();                                           // the next status poll confirms "stopped"
  await refreshStatus();
}

async function doForceStop() {
  if (!mgState.editor) return;
  setOp("stopping", "forcing");
  refreshButtons();
  try {
    const r = await post("/api/force-stop");
    mgToast(`Force-stopped in ${fmtElapsed(r.elapsed_s)} — the process tree was terminated immediately`, "ok");
  } catch (err) { mgToast(err.message, "error"); }
  clearOp();
  await refreshStatus();
}

async function doRestart() {
  if (!mgState.editor) return;
  setOp("restarting", "stopping");
  refreshButtons();
  try {
    const r = await post("/api/restart", {
      config: mgState.editor.summary.config,
      port: parseInt($("port-num").value, 10) || undefined,
      open_chat: false,
    });
    if (r.error) { mgToast(r.error, "error"); clearOp(); }
    // otherwise the operation indicator stays up; polls walk it through stopping → starting → running.
  } catch (err) { mgToast(err.message, "error"); clearOp(); }
  refreshButtons();
}

/* ---------------------------------------------------------------- log + tool */
let mgLogRefreshAt = 0;
async function handleLog() {
  const s = mgState.status.server;
  if (!s || !s.log) return;
  const showState0 = s.state === "starting" || s.state === "running" || s.state === "error";
  const showState = mgOp.state != null || showState0;  // mid-operation: leave the last log visible
  if (!showState) { $("logcard").hidden = true; return; }
  const now = Date.now();
  if (now < mgLogRefreshAt) return;
  mgLogRefreshAt = now + 3000;
  $("logcard").hidden = false;
  $("log-where").textContent = s.config || "";
  try {
    const d = await api("/api/log?path=" + encodeURIComponent(s.log) + "&tail=140");
    const box = $("logbox");
    box.textContent = d.text || "(no output yet — the model is loading, this can take a minute or two)";
    box.scrollTop = box.scrollHeight;
  } catch (e) { /* transient */ }
}

/* the running setup.py preparation (custom GGUF): its own state + a settle-once transition so a finished
   run (also one that failed in seconds, before the first poll) shows its outcome and never just flashes. */
let mgPrepare = null;                       // { config, title, alive, settled }
let mgPrepareLogAt = 0;

function mgPrepareResetCard(clear) {
  const card = $("prepare-card");
  if (card) card.hidden = true;
  if (clear) mgPrepare = null;
}

async function pollPrepareLog(path, force) {
  const now = Date.now();
  if (!force && now < mgPrepareLogAt) return;
  mgPrepareLogAt = now + 1500;
  try {
    const d = await api("/api/log?path=" + encodeURIComponent(path) + "&tail=200");
    const box = $("prepare-log");
    box.textContent = d.text || "(no output yet — setup.py is starting)";
    box.scrollTop = box.scrollHeight;
  } catch (e) { /* transient */ }
}

function setUpPrepareButton(btn) {
  $("gguf-prepare").disabled = !!btn;
  $("gguf-prepare").textContent = btn ? "Preparing…" : "Prepare with Strata";
}

function handleTool() {
  const t = mgState.tool;
  if (!t) {
    mgPrepareResetCard(true);
    setUpPrepareButton(false);
    return;
  }
  const alive = !!t.alive;
  if (!mgPrepare || mgPrepare.config !== t.config) {
    mgPrepare = { config: t.config, title: t.title || "", alive: null, settled: false };
  }
  const el = Math.round((Date.now() / 1000) - (t.started || 0));
  const spin = $("prepare-card").querySelector(".spinner");
  if (alive) {
    mgPrepare.alive = true;
    mgPrepare.settled = false;                      // another run started: watch it again
    setUpPrepareButton(true);
    $("prepare-card").hidden = false;
    if (spin) spin.style.visibility = "visible";
    $("prepare-text").textContent =
      `${t.what || mgPrepare.title || t.config} — running (${el}s) · log: ${t.log}`;
    pollPrepareLog(t.log);
  } else if (!mgPrepare.settled) {
    // finished: either just now (alive->dead) or already gone before the first poll (fast fail)
    mgPrepare.settled = true;
    mgPrepare.alive = false;
    setUpPrepareButton(false);
    $("prepare-card").hidden = false;
    if (spin) spin.style.visibility = "hidden";
    const settleToken = mgPrepare.config;
    const failed = t.failed || !t.ready;
    if (failed) {
      $("prepare-text").textContent =
        `✗ ${mgPrepare.title || mgPrepare.config} FAILED — setup.py's log (${t.log}) is shown below`;
      mgToast(`✗ ${mgPrepare.title || "preparation"} failed — see the log`, "error");
    } else {
      $("prepare-text").textContent = `✓ ${mgPrepare.title} installed as ${mgPrepare.config}`;
      mgToast(`✓ ${mgPrepare.title} installed as ${mgPrepare.config}`, "ok");
      reloadModels().then(() => selectModel(mgPrepare.config));   // refresh + select the NEW config
    }
    pollPrepareLog(t.log, true);
    setTimeout(() => { if (mgPrepare && mgPrepare.settled && mgPrepare.config === settleToken) {
      mgPrepareResetCard(true);
    } }, 15000);
  } else {
    setUpPrepareButton(false);
  }
}

/* ---------------------------------------------------------------- browse + custom GGUF
   The Custom GGUF text field is the SOURCE OF TRUTH.  The only path that is detected is the
   field's value, and a detection is only ever usable (and Prepare only enabled) when it was
   made for exactly the path the field shows now:
     - changing the field (typed, pasted, autofilled, Browse's "Use this folder") invalidates
       the previous detection immediately and hides Prepare
     - every detection carries the generation token it was started under; a response that
       arrives after the field changed again (or for an older path) is ignored
     - Prepare refuses to run with any detection whose directory does not exactly match the
       field (ggufDet.prepareAllowed) - visible path A + internal detection B cannot happen
   Browse is only a picker with a preview: "Use this folder" writes the chosen path into the
   field and the SAME detect-on-change path runs, so there is no second, hidden state to go
   stale.  The field's note (gguf-note) always describes the detection of the CURRENT value. */
const ggufDet = new GgufState.GgufDetection();
let mgBrowsePath = "";
let mgBrowseToken = 0;
let mgGgufPathTimer = null;

function setGgufNote(text, ok) {
  const n = $("gguf-note");
  if (!n) return;
  n.textContent = text || "";
  n.dataset.ok = ok || "";
}

function setGgufPrepareEnabled(on) {
  const b = $("gguf-prepare");
  if (!b) return;
  b.hidden = !on;
  b.disabled = false;
  b.textContent = "Prepare with Strata";
}

function paintGgufDetect(det) {
  // Paint the COMMITTED detection for the current field value (ggufDet.detect).
  if (!det || det.ok === false) {
    setGgufNote(det && det.error ? det.error : "", det && det.ok === false ? "0" : "");
    setGgufPrepareEnabled(false);
    return;
  }
  const shown = (det.variant ? det.variant.replace(/-/g, " ").replace(/\b\w/g, c => c.toUpperCase()) : "");
  const what = [det.family_title, det.quant, shown].filter(Boolean).join(" ");
  setGgufNote(`✓ ${det.title || what} — ${det.shards.length} shards`
    + (det.variant ? ` · custom build “${det.variant}”` : ""), "1");
  setGgufPrepareEnabled(true);
  $("gguf-prepare").title = det.variant
    ? `prepares strata-${det.tag}-${det.variant}.json`
    : `prepares strata-${det.tag}.json (an existing config pointing at these same files is reused)`;
}

// The field changed (typed / pasted / autofilled / "Use this folder"): the old detection dies
// NOW, Prepare disappears, and the current value is (re)detected - always the same code path.
function ggufPathChanged() {
  clearTimeout(mgGgufPathTimer);
  const token = ggufDet.invalidate();            // stale detection is gone immediately
  const path = $("gguf-path").value.trim();
  if (!path) { setGgufNote(""); setGgufPrepareEnabled(false); return; }
  setGgufNote("detecting …", "");
  setGgufPrepareEnabled(false);
  mgGgufPathTimer = setTimeout(() => detectGgufPath(token, path), 200);   // debounce keystrokes
}

async function detectGgufPath(token, path) {
  let det;
  try {
    det = await api("/api/gguf-detect", { method: "POST", headers: { "Content-Type": "application/json" },
                                           body: JSON.stringify({ path }) });
  } catch (e) {
    if (ggufDet.token === token && GgufState.pathsEqual($("gguf-path").value, path))
      setGgufNote("✗ " + e.message, "0");
    return;
  }
  // accept() refuses the response unless it is the current generation AND (on success) its
  // directory is exactly the path the field shows now - the late/stale responses never land.
  if (!ggufDet.accept($("gguf-path").value, det, token)) return;
  paintGgufDetect(ggufDet.detect);
}

function openBrowse(startPath) {
  api("/api/browse", { method: "POST", headers: { "Content-Type": "application/json" },
                       body: JSON.stringify({ path: startPath || mgBrowsePath }) })
    .then(d => {
      mgBrowsePath = d.path;
      $("browse-path").value = d.path;
      $("browse-modal").hidden = false;
      renderDirs(d);
    })
    .catch(e => mgToast(e.message, "error"));
}

function renderDirs(d) {
  const ul = $("browse-dirs");
  ul.textContent = "";
  if (d.parent && d.parent !== d.path) {
    const up = document.createElement("li");
    up.appendChild(Object.assign(document.createElement("span"), { textContent: "↑ .." }));
    ul.appendChild(up);
    up.addEventListener("click", () => setBrowseDir(d.parent));
  }
  for (const dir of d.dirs) {
    const li = document.createElement("li");
    li.dataset.path = dir;
    const t = document.createElement("span");
    t.className = "dir";
    t.textContent = dir.replace(/^.*[\\/]/, "");
    li.appendChild(t);
    li.addEventListener("click", () => browsePreview(dir));
    ul.appendChild(li);
  }
  browsePreview(d.path);
}

function setBrowseDir(path) {
  api("/api/browse", { method: "POST", headers: { "Content-Type": "application/json" },
                       body: JSON.stringify({ path }) })
    .then(d => {
      mgBrowsePath = d.path;
      $("browse-path").value = d.path;
      renderDirs(d);
    })
    .catch(e => mgToast(e.message, "error"));
}

// The picker's own preview panel only - it never touches the committed detection.  "Use this
// folder" is the only thing that writes into the field, and that runs ggufPathChanged() like
// any other change.  A late preview response (the user clicked another folder meanwhile) is
// ignored by the preview's own generation token.
async function browsePreview(path) {
  mgBrowsePath = path;
  const token = ++mgBrowseToken;
  const dd = $("browse-detect");
  dd.textContent = "detecting …";
  dd.dataset.ok = "0";
  let det;
  try {
    det = await api("/api/gguf-detect", { method: "POST", headers: { "Content-Type": "application/json" },
                                           body: JSON.stringify({ path }) });
  } catch (e) {
    if (token === mgBrowseToken) dd.textContent = e.message;
    return;
  }
  if (token !== mgBrowseToken) return;                       // a later preview superseded this one
  if (det.ok) {
    dd.textContent = `✓ ${det.title || det.family_title + " " + det.quant} — shards: ${det.shards.join(", ")}`
      + (det.variant ? ` · custom build “${det.variant}”` : "");
    dd.title = det.variant ? `prepares strata-${det.tag}-${det.variant}.json`
      : `prepares strata-${det.tag}.json (an existing config pointing at these same files is reused)`;
    dd.dataset.ok = "1";
  } else {
    dd.textContent = det.error;
    dd.dataset.ok = "0";
  }
}

function useBrowseFolder() {
  $("browse-modal").hidden = true;
  $("gguf-path").value = mgBrowsePath;       // write the choice into the field, then the usual change
  ggufPathChanged();
}

async function doPrepareGguf() {
  const path = $("gguf-path").value.trim();
  // The field is the source of truth: never prepare a detection whose directory is not exactly
  // the path the field shows now.  If they disagree (or nothing was detected), re-detect and
  // refuse instead of running the stale path.
  if (!ggufDet.prepareAllowed(path)) {
    mgToast("detect this exact folder first — the path in the field is what gets prepared", "error");
    ggufPathChanged();
    return;
  }
  const det = ggufDet.detect;
  const btn = $("gguf-prepare");
  setUpPrepareButton(true);
  try {
    const r = await post("/api/prepare-gguf", { path: det.dir, context: parseInt($("ctx-num").value, 10) || 32768 });
    if (r.already) {
      // the exact same GGUF files are already a config: select it, nothing runs
      setUpPrepareButton(false);
      mgToast(`already installed as ${r.config}${r.title ? " — " + r.title : ""}`, "ok");
      await reloadModels();
      await selectModel(r.config);
      return;
    }
    // setup.py runs detached; /api/status polls its log and settles the result (success or failure)
    $("prepare-card").hidden = false;
    $("prepare-text").textContent =
      `⏳ ${r.title || det.title} is being prepared as ${r.config} — setup.py runs below`;
    mgPrepare = { config: r.config, title: r.title || det.title || "", alive: null, settled: false };
    setTimeout(refreshStatus, 400);
  } catch (e) {
    setUpPrepareButton(false);
    $("prepare-card").hidden = false;
    $("prepare-text").textContent = `✗ ${e.message}`;
    mgToast(e.message, "error");
  }
}

async function reloadModels(keep) {
  const d = await api("/api/models");
  mgState.models = d.models;
  const defaultName = keep && mgState.editor ? mgState.editor.summary.config : d.default;
  renderModels(defaultName);
  renderModelSelect(defaultName);
  if (keep && mgState.editor) $("model-sel").value = mgState.editor.summary.config;
  return defaultName;
}

/* ---------------------------------------------------------------- boot */
async function mgBoot() {
  try { await mgRefreshSystem(); } catch (e) { /* non-fatal */ }
  try {
    const defaultName = await reloadModels();
    if (defaultName) await selectModel(defaultName);
    else mgToast("No models installed yet — SETUP.bat first.", "error");
  } catch (e) { mgToast(e.message, "error"); }
  await refreshStatus();
  setInterval(refreshStatus, MG_POLL_MS);
  setInterval(() => { if (!mgState.busy) reloadModels(true).catch(() => {}); }, MG_POLL_MS * 8);
  showTab(location.hash.slice(1) || store.get("lasttab", "manager") || "manager");
}
mgBoot();

/* ---------------------------------------------------------------- wire up */
$("model-sel").addEventListener("change", (ev) => selectModel(ev.target.value));
$("ctx-num").addEventListener("input", () => { syncChips(); updateCtxHint(); });
$("save").addEventListener("click", doSave);
$("start").addEventListener("click", doStart);
$("stop").addEventListener("click", doStop);
$("force-stop").addEventListener("click", doForceStop);
$("restart").addEventListener("click", doRestart);
$("gguf-browse").addEventListener("click", () => openBrowse($("gguf-path").value || ""));
$("gguf-path").addEventListener("input", ggufPathChanged);        // typed / pasted: detect on change
$("gguf-path").addEventListener("change", ggufPathChanged);       // plus the committed (blur) value
$("browse-close").addEventListener("click", () => { $("browse-modal").hidden = true; });
$("browse-use").addEventListener("click", useBrowseFolder);
$("browse-go").addEventListener("click", () => setBrowseDir($("browse-path").value.trim()));
$("browse-path").addEventListener("keydown", (ev) => { if (ev.key === "Enter") setBrowseDir($("browse-path").value.trim()); });
$("gguf-prepare").addEventListener("click", doPrepareGguf);
