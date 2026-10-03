// serve/web/app.js - the Strata web app: Chat, Monitor, About. No framework, no network beyond this server.
// The Monitor tab rebuilds PR #22's dashboard idea (code-martin) on the server's own /metrics.
"use strict";

const $ = (id) => document.getElementById(id);
const SPRITE = "web/sprite.svg";
const icon = (name, cls = "st-icon") => `<svg class="${cls}" aria-hidden="true"><use href="${SPRITE}#i-${name}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
// a context size: 32768 -> "32K" (powers of two), else like kfmt
const ctxfmt = (n) => (n && n % 1024 === 0 ? `${fmt(n / 1024)}K` : kfmt(n));
const gb = (b, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d));   // memory: binary GB, as Windows shows it

const store = {
  get(k, d) { try { const v = localStorage.getItem("strata." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  // false when it was not kept: private mode, or the browser's storage for this page is full
  set(k, v) { try { localStorage.setItem("strata." + k, JSON.stringify(v)); return true; } catch (e) { return false; } },
  del(k) { try { localStorage.removeItem("strata." + k); } catch (e) { /* ignore */ } },
};

// ------------------------------------------------------------------ toasts
function toast(kind, title, text = "", ms = 3500, action = null) {
  const names = {info: "info", success: "check", warn: "warning", error: "error"};
  const el = document.createElement("div");
  el.className = `st-toast st-toast--${kind}`;
  el.innerHTML = `${icon(names[kind] || "info")}<div><div class="st-toast__title"></div><div class="t-text"></div></div>`;
  el.querySelector(".st-toast__title").textContent = title;
  el.querySelector(".t-text").textContent = text;
  if (action) {
    const b = document.createElement("button");
    b.className = "st-btn st-btn--secondary";
    b.style.height = "32px";
    b.style.marginLeft = "auto";
    b.textContent = action.label;
    b.onclick = () => { action.run(); el.remove(); };
    el.appendChild(b);
  }
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {                                   // http on another host: no async clipboard
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  if (btn) {
    const use = btn.querySelector("use");
    use.setAttribute("href", `${SPRITE}#i-check`);
    setTimeout(() => use.setAttribute("href", `${SPRITE}#i-copy`), 1500);
  }
  toast("success", "Copied to clipboard", "", 1800);
}

// ------------------------------------------------------------------ theme and tabs
// the system's theme until the user picks one (only a click is saved)
function setTheme(t, save) {
  document.documentElement.dataset.theme = t;
  if (save) try { localStorage.setItem("strata.theme", t); } catch (e) { /* ignore */ }
  $("theme-icon").setAttribute("href", `${SPRITE}#i-${t === "dark" ? "sun" : "moon"}`);
  $("dark-toggle").setAttribute("aria-checked", String(t === "dark"));
}
const flipTheme = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
$("theme-btn").onclick = flipTheme;
$("dark-toggle").onclick = flipTheme;
setTheme(document.documentElement.dataset.theme || "light", false);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  let saved = null;
  try { saved = localStorage.getItem("strata.theme"); } catch (err) { /* ignore */ }
  if (!saved) setTheme(e.matches ? "dark" : "light", false);
});

let tab = "chat";
const TABS = ["chat", "files", "monitor", "settings", "about"];
function showTab(name) {
  tab = TABS.includes(name) ? name : "chat";
  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of TABS) $(`view-${v}`).hidden = v !== tab;
  if (location.hash.slice(1) !== tab) history.replaceState(null, "", tab === "chat" ? location.pathname : `#${tab}`);
  if (tab === "chat") $("input").focus();
  if (tab === "monitor") loadMcp();
  if (tab === "files" && !fs.loaded && wsState === "ready") fsOpen("");
  if (tab === "settings") renderRoots();
  if (lastMetrics) render(lastMetrics);
}
for (const b of document.querySelectorAll(".st-tab")) b.onclick = () => showTab(b.dataset.tab);
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

// ------------------------------------------------------------------ server access
function headers(json = false) {
  const h = {};
  const key = store.get("apikey", "");
  if (key) h.Authorization = "Bearer " + key;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
$("api-key").value = store.get("apikey", "");
$("api-key").onchange = () => {
  store.set("apikey", $("api-key").value.trim());
  toast("success", "API key saved", "Kept in this browser only.");
  loadWorkspace();                                 // the projects and chats need it too
};

let health = {model: "strata", images: false, max_context: 0};
async function loadHealth() {
  try {
    health = await (await fetch("health")).json();
    $("attach-btn").title = health.images ? "Attach a text file or a picture (or drop it here)"
                                          : "Attach a text file (or drop it here)";
    $("chat-empty-sub").textContent = `${health.model} runs on this PC. Nothing leaves it.`;
  } catch (e) {
    setTimeout(loadHealth, 2000);
  }
}

// ------------------------------------------------------------------ Monitor
const METRICS = [
  {key: "speed", label: "Speed", icon: "gauge", unit: "t/s", series: "tok_s"},
  {key: "gpu", label: "GPU load", icon: "gpu", unit: "%", series: "gpu_util", max: 100},
  {key: "vram", label: "VRAM", icon: "layers", unit: "GB", series: "gpu_mem_used"},
  {key: "temp", label: "GPU temp", icon: "thermometer", unit: "°C", series: "gpu_temp", tone: "warn"},
  {key: "power", label: "Power", icon: "bolt", unit: "W", series: "gpu_power"},
  {key: "pcie", label: "PCIe", icon: "link", unit: "", series: "gpu_pcie_rx_mb", tone: "info"},
  {key: "cpu", label: "CPU", icon: "cpu", unit: "%", series: "cpu", max: 100},
  {key: "disk", label: "Disk read", icon: "disk", unit: "MB/s", series: "disk_read_mb", tone: "info"},
];
$("metrics").innerHTML = METRICS.map((m) => `
  <div class="st-card metric-card"><div class="st-metric">
    <span class="st-metric__label">${icon(m.icon, "st-icon st-icon--sm")}${esc(m.label)}</span>
    ${m.key === "speed" ? `<div class="speed-values">
      <div><span class="st-metric__value" id="mv-speed">-</span><span class="st-metric__sub" id="ms-speed">Decode</span></div>
      <div class="speed-prefill"><span class="st-metric__value" id="mv-prefill">-</span><span class="st-metric__sub" id="ms-prefill">Prefill</span></div>
    </div>` : `<span class="st-metric__value" id="mv-${m.key}">–</span>
    <span class="st-metric__sub" id="ms-${m.key}"></span>`}
    <svg class="st-metric__spark" id="sp-${m.key}" viewBox="0 0 100 32" preserveAspectRatio="none"${m.tone ? ` data-tone="${m.tone}"` : ""}>
      <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
      stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
      ${m.key === "speed" ? `<g id="sp-prefill" class="speed-prefill"><path class="area" fill="currentColor" opacity=".12"/>
        <path class="line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"
        stroke-linecap="round" vector-effect="non-scaling-stroke"/></g>` : ""}</svg>
  </div></div>`).join("");

function spark(id, values, max) {
  const svg = $(id);
  const v = (values || []).map((x) => (x == null ? 0 : x));
  if (v.length < 2) { svg.querySelector(".line").setAttribute("d", ""); svg.querySelector(".area").setAttribute("d", ""); return; }
  const top = Math.max(max || 0, ...v, 1e-9);
  const pts = v.map((x, i) => [(i / (v.length - 1)) * 100, 30 - (x / top) * 26]);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");
  svg.querySelector(".line").setAttribute("d", line);
  svg.querySelector(".area").setAttribute("d", `${line}L100,32L0,32Z`);
}
function setMetric(key, value, unit, sub) {
  $(`mv-${key}`).innerHTML = value == null ? "–" : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
  $(`ms-${key}`).textContent = sub || "";
}

let lastMetrics = null, metricsFailures = 0, keyWarned = false, mcpTick = 0;
let reqShowAll = false;   // the Monitor's request table: the last 12, or every one the server keeps (issue #35)
async function poll() {
  try {
    const r = await fetch(reqShowAll ? "metrics?requests=all" : "metrics", {headers: headers()});
    if (r.status === 401) {
      setPill("error", "API key needed");
      if (!keyWarned) { keyWarned = true; toast("warn", "API key needed", "This server needs a key: add it under Settings.", 6000); }
    } else if (r.ok) {
      lastMetrics = await r.json();
      metricsFailures = 0;
      render(lastMetrics);
    } else {
      throw new Error(`HTTP ${r.status}`);
    }
  } catch (e) {
    if (++metricsFailures === 3) setPill("error", "Server not reachable");
  }
  if (tab === "monitor" && ++mcpTick % 10 === 0) loadMcp();       // server states change rarely: every 10 s
  setTimeout(poll, 1000);
}

function setPill(state, text) {
  $("pill").dataset.state = state === "error" ? "queued" : state;
  $("pill-text").textContent = text;
}

function render(m) {
  const live = m.live || {}, hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {}, h = m.history || {};
  const last = (m.requests || [])[0];
  // the header pill
  if (live.state === "reading") {
    const pct = live.prompt_total ? Math.round((100 * live.prompt_read) / live.prompt_total) : null;
    setPill("reading", pct != null ? `Reading prompt · ${pct}%` : "Reading prompt");
  } else if (live.state === "generating") {
    setPill("generating", `Generating · ${fmt(live.tok_s, 1)} tok/s`);
  } else {
    setPill("idle", "Idle");
  }
  if (live.queued > 0) setPill("queued", `${live.queued} queued`);
  if (engine && engine.state !== "running" && live.state !== "reading" && live.state !== "generating") {
    setPill("idle", engine.state === "starting" ? "Loading" : "Engine off");
  }
  if (tab === "monitor") renderMonitor(live, hw, st, eng, h, last, m.requests || [], m.totals, m.requests_kept);
  if (tab === "about") renderAbout(eng, hw, st);
}

function renderTotals(t) {
  if (!t || !t.requests) return "";
  const since = new Date(t.since * 1000).toLocaleString([], {weekday: "short", hour: "2-digit", minute: "2-digit"});
  const read = t.prompt_tokens - t.reused;
  const pSpeed = t.prompt_ms > 0 && read > 0 ? ` at ${fmt(read / (t.prompt_ms / 1000))} tok/s` : "";
  const oSpeed = t.decode_ms > 0 && t.output_tokens > 0 ? ` at ${fmt(t.output_tokens / (t.decode_ms / 1000), 1)} tok/s` : "";
  return `Since ${since}: ${fmt(t.requests)} requests · ${fmt(read)} prompt tokens read${pSpeed} (${fmt(t.reused)} reused) · ` +
         `${fmt(t.output_tokens)} written${oSpeed}`;
}
function renderMonitor(live, hw, st, eng, h, last, requests, totals, kept) {
  // model state
  const on = live.queued > 0 ? "queued" : live.state;
  for (const b of document.querySelectorAll("#state-badges .st-badge")) b.classList.toggle("on", b.dataset.s === on || b.dataset.s === live.state);
  const prog = $("state-progress");
  let label = "Waiting for a request", detail = "", pct = 0;
  if (live.state === "reading") {
    label = "Reading prompt";
    prog.dataset.tone = "info";
    if (live.prompt_total) {
      pct = (100 * live.prompt_read) / live.prompt_total;
      detail = `${fmt(live.prompt_read)} / ${fmt(live.prompt_total)} tokens · ${fmt(pct)}%`;
    } else {
      detail = `${fmt(live.prompt_tokens)} tokens`;
    }
  } else if (live.state === "generating") {
    label = live.phase ? live.phase[0].toUpperCase() + live.phase.slice(1) : "Generating";
    delete prog.dataset.tone;
    pct = live.max_tokens ? Math.min(100, (100 * live.generated) / live.max_tokens) : 0;
    detail = `${fmt(live.generated)} tokens · ${fmt(live.tok_s, 1)} tok/s`;
  } else if (last) {
    delete prog.dataset.tone;
    detail = `last: ${fmt(last.output_tokens)} tokens${last.decode_tok_s ? ` at ${fmt(last.decode_tok_s, 1)} tok/s` : ""}`;
  }
  $("state-label").textContent = label;
  $("state-detail").textContent = detail;
  $("state-bar").style.width = `${pct}%`;

  // the eight cards
  const speed = live.state === "generating" ? live.tok_s : last ? last.decode_tok_s : null;
  setMetric("speed", speed == null ? null : fmt(speed, 1), "t/s",
            live.state === "generating" ? "Decode now" : last ? "Decode last request" : "Decode");
  const prefill = live.state !== "idle" ? live.prefill_tok_s_mean
                : last && last.prompt_ms > 0 ? Math.max(0, last.prompt_tokens - (last.reused || 0)) / (last.prompt_ms / 1000) : null;
  setMetric("prefill", prefill == null ? null : fmt(prefill), "t/s",
            live.state === "reading" ? "Prefill now" : live.state === "generating" ? "Prefill this request" : last ? "Prefill last request" : "Prefill");
  spark("sp-speed", h.tok_s);
  spark("sp-prefill", h.prefill_tok_s_mean);
  // a model split across several cards (issue #112): the cards show their total / mean / hottest, and each card's own
  const per = (f) => (hw.gpus || []).map((g) => `GPU ${g.index} ${f(g)}`).join(" · ");
  const multi = (hw.gpus || []).length > 1;
  setMetric("gpu", hw.gpu_util == null ? null : fmt(hw.gpu_util), "%",
            multi ? per((g) => (g.util == null ? "–" : `${fmt(g.util)}%`)) : st.gpu_name || "");
  spark("sp-gpu", h.gpu_util, 100);
  setMetric("vram", hw.gpu_mem_used == null ? null : gb(hw.gpu_mem_used), hw.gpu_mem_total ? `/ ${gb(hw.gpu_mem_total, 0)} GB` : "GB",
            multi ? per((g) => (g.mem_used == null ? "–" : `${gb(g.mem_used)} GB`))
                  : eng.expert_slots ? `${fmt(eng.expert_slots)} experts cached` : "");
  spark("sp-vram", h.gpu_mem_used, hw.gpu_mem_total);
  setMetric("temp", hw.gpu_temp == null ? null : fmt(hw.gpu_temp), "°C",
            multi ? per((g) => (g.temp == null ? "–" : `${fmt(g.temp)}°`)) : "");
  spark("sp-temp", h.gpu_temp, 90);
  setMetric("power", hw.gpu_power == null ? null : fmt(hw.gpu_power), "W", hw.gpu_power_limit ? `of ${fmt(hw.gpu_power_limit)} W limit` : "");
  spark("sp-power", h.gpu_power, hw.gpu_power_limit);
  const gen = hw.gpu_pcie_gen_max || hw.gpu_pcie_gen;
  setMetric("pcie", gen ? `Gen${gen}` : null, hw.gpu_pcie_width ? `x${hw.gpu_pcie_width}` : "",
            hw.gpu_pcie_rx_mb == null ? "" : `to GPU ${fmt(hw.gpu_pcie_rx_mb, hw.gpu_pcie_rx_mb < 10 ? 1 : 0)} MB/s` +
            (hw.gpu_pcie_gen && gen && hw.gpu_pcie_gen < gen ? ` · idle Gen${hw.gpu_pcie_gen}` : ""));
  spark("sp-pcie", h.gpu_pcie_rx_mb);
  setMetric("cpu", hw.cpu == null ? null : fmt(hw.cpu), "%", st.threads ? `${st.cores ? `${st.cores} cores · ` : ""}${st.threads} threads` : "");
  spark("sp-cpu", h.cpu, 100);
  if (hw.disk_read_mb == null) {
    setMetric("disk", null, "", st.psutil ? "" : "needs psutil (setup installs it)");
  } else {
    const big = hw.disk_read_mb >= 1000;
    setMetric("disk", big ? fmt(hw.disk_read_mb / 1024, 2) : fmt(hw.disk_read_mb, hw.disk_read_mb < 10 ? 1 : 0), big ? "GB/s" : "MB/s",
              hw.disk_write_mb == null ? "" : `write ${fmt(hw.disk_write_mb, 1)} MB/s`);
  }
  spark("sp-disk", h.disk_read_mb);

  // context fill: the running request, else the last one
  const ctx = eng.max_context || 0;
  let used = 0;
  if (live.state !== "idle") used = (live.prompt_tokens || 0) + (live.generated || 0);
  else if (last) used = (last.prompt_tokens || 0) + (last.output_tokens || 0);
  const frac = ctx ? Math.min(1, used / ctx) : 0;
  $("ctx-fill").setAttribute("stroke-dasharray", `${(235.6 * frac).toFixed(1)} 314.2`);
  $("ctx-fill").style.opacity = 235.6 * frac >= 3 ? "1" : "0";         // a near-zero arc would draw just its round cap
  $("ctx-pct").textContent = `${Math.round(frac * 100)}%`;
  $("ctx-sub").textContent = ctx ? `${kfmt(used)} / ${ctxfmt(ctx)}` : "–";
  const cacheBytes = (eng.expert_cache_mib || 0) * 1048576;
  $("slots-text").textContent = eng.expert_slots ? `${fmt(eng.expert_slots)} · ${gb(cacheBytes)} GB` : "–";
  $("slots-bar").style.width = hw.gpu_mem_total ? `${Math.min(100, (100 * cacheBytes) / hw.gpu_mem_total)}%` : "0%";
  $("ram-text").textContent = hw.ram_total ? `${gb(hw.ram_used)} / ${gb(hw.ram_total, 0)} GB` : "–";
  const ramPct = hw.ram_total ? (100 * hw.ram_used) / hw.ram_total : 0;
  $("ram-bar").style.width = `${ramPct}%`;
  if (ramPct > 92) $("ram-progress").dataset.tone = "danger"; else delete $("ram-progress").dataset.tone;
  $("temp-text").textContent = hw.gpu_temp == null ? "–" : `${fmt(hw.gpu_temp)} °C`;
  $("temp-bar").style.width = hw.gpu_temp == null ? "0%" : `${Math.min(100, hw.gpu_temp)}%`;

  // recent requests
  const body = $("req-body");
  if (!requests.length) {
    body.innerHTML = `<tr><td colspan="8" class="muted">No requests yet</td></tr>`;
  } else {
    const badge = {stop: ["", "Done"], length: ["", "Max tokens"], cancel: ["st-badge--queued", "Stopped"],
                   disconnect: ["st-badge--queued", "Closed"], error: ["st-badge--error", "Error"]};
    body.innerHTML = requests.slice(0, reqShowAll ? requests.length : 12).map((r) => {
      const [cls, text] = badge[r.finish] || ["", r.finish || "–"];
      const t = new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      const proj = r.projection == null ? "" : ` <span class="st-badge${r.projection ? " st-badge--reading" : ""}" title="experimental speed projection ${r.projection ? "on" : "off"}">${r.projection ? "ESP" : "stock"}</span>`;
      const hit = r.hit_rate == null ? "–" : `${(r.hit_rate * 100).toFixed(1)}%`;
      return `<tr><td>${esc(t)}</td><td><span class="st-badge ${cls}">${esc(text)}</span>${proj}</td><td class="num">${fmt(r.prompt_tokens)}</td>
        <td class="num">${fmt(r.reused)}</td><td class="num">${fmt(r.output_tokens)}</td><td class="num">${fmt(r.decode_tok_s, 1)}</td>
        <td class="num">${hit}</td><td class="num">${fmt(r.duration_s, 1)} s</td></tr>`;
    }).join("");
  }
  const all = $("req-all");
  kept = kept == null ? requests.length : kept;
  all.hidden = kept <= 12;
  all.textContent = reqShowAll ? "Show fewer" : `Show all (${kept})`;
  $("req-wrap").classList.toggle("all", reqShowAll);
  $("req-totals").textContent = renderTotals(totals);
}

function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v, copy]) =>
    `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code><button class="st-btn st-btn--icon" data-copy="${esc(v)}" aria-label="Copy">${icon("copy")}</button>` : esc(v)}</dd>`).join("");
}
// INFO cvec=project:4-44[:singleL] | add:A-B | 0
function projectionText(c) {
  if (!c || c === "0" || c === 0) return null;
  const [mode, range, single] = String(c).split(":");
  const [a, b] = (range || "").split("-");
  return `${mode === "project" ? "Projection" : "Additive"} control vector on layers ${a}–${b}` +
         `${single ? ` (layer ${single.replace("single", "")}'s direction)` : ""}. Per chat in Sampling. Its package ` +
         "describes the vector as a refusal-direction projection; measure the speed yourself";
}
function renderAbout(eng, hw, st) {
  const kv = {int8: "8-bit", q4_0: "4-bit (Hadamard-rotated)", fp16: "16-bit"}[eng.kv] || eng.kv;
  facts($("facts-engine"), [
    ["Model", eng.model],
    ["Engine", eng.version ? `v${eng.version}` : "built from source"],
    ["Context", eng.max_context ? `${fmt(eng.max_context)} tokens` : null],
    ["KV cache", kv ? `${kv}${eng.kv_resident ? `, streamed: ${fmt(eng.kv_resident)} positions per layer in VRAM, the rest in RAM` : ", all in VRAM"}` : null],
    ["Experts in VRAM", eng.expert_slots ? `${fmt(eng.expert_slots)} (${gb((eng.expert_cache_mib || 0) * 1048576)} GB)` : null],
    ["Speculation", eng.spec ? `MTP drafts up to ${Math.max(0, (eng.mtp_max || eng.spec) - 1)} tokens${eng.lookup ? ", prompt lookup on" : ""}` : null],
    ["Images", eng.images ? "on" : "off"],
    ["Experimental speed projection", projectionText(eng.cvec)],
  ]);
  facts($("facts-hw"), [
    ["GPU", st.gpu_name ? `${st.gpu_name}${hw.gpu_mem_total ? `, ${gb(hw.gpu_mem_total, 0)} GB` : ""}` : "not readable (NVML)"],
    ["CPU", st.cpu_name ? `${st.cpu_name}${st.threads ? `, ${st.threads} threads` : ""}` : null],
    ["RAM", hw.ram_total ? `${gb(hw.ram_total, 0)} GB` : null],
  ]);
  const base = location.origin;
  facts($("facts-api"), [
    ["OpenAI base URL", `${base}/v1`, true],
    ["Anthropic base URL", base, true],
    ["Model name", eng.model, true],
  ]);
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (b) copyText(b.dataset.copy, b);
});
$("req-all").addEventListener("click", () => { reqShowAll = !reqShowAll; if (lastMetrics) render(lastMetrics); });

// ------------------------------------------------------------------ MCP servers (GET /mcp)
// Tools from the MCP servers in the run config: the chat offers them to the model (opt-in per request,
// "strata_mcp": true, which only this page sends); the Monitor lists the servers and what they offer.
let mcpInfo = {servers: [], tools: 0}, mcpRetry = null;
async function loadMcp() {
  try {
    const r = await fetch("mcp", {headers: headers()});
    if (!r.ok) return;
    mcpInfo = await r.json();
  } catch (e) { return; /* an older server: no MCP */ }
  renderMcp();
  clearTimeout(mcpRetry);                          // right after the start, servers may still be starting (npx downloads)
  if ((mcpInfo.servers || []).some((s) => s.status === "starting")) mcpRetry = setTimeout(loadMcp, 3000);
}
const MCP_STATE = {ready: ["st-badge--generating", "Connected"], starting: ["st-badge--reading", "Starting"],
                   failed: ["st-badge--error", "Failed"], stopped: ["st-badge--queued", "Stopped"], idle: ["", "Waiting"]};
function renderMcp() {
  const servers = mcpInfo.servers || [];
  $("mcp-card").hidden = !servers.length;
  $("mcp-row").hidden = !servers.length;
  const ready = servers.filter((s) => s.status === "ready" || s.status === "stopped");
  $("mcp-sum").textContent = servers.length ? `${fmt(mcpInfo.tools)} tools · ${ready.length} of ${servers.length} servers connected` : "";
  $("mcp-row-sub").textContent = mcpInfo.tools ? `${fmt(mcpInfo.tools)} tools from ${ready.map((s) => s.name).join(", ")}; the model calls them when it decides to`
                                               : "no server is connected yet (see the Monitor)";
  $("mcp-list").innerHTML = servers.map((s) => {
    const [cls, text] = MCP_STATE[s.status] || ["", s.status];
    const info = s.info && s.info.name ? ` · ${s.info.name}${s.info.version ? ` ${s.info.version}` : ""}` : "";
    return `<div class="mcp-server"><div class="mcp-server__head"><span class="st-badge ${cls}">${esc(text)}</span>` +
      `<strong>${esc(s.name)}</strong><span class="muted small">${esc(s.transport)} · ${fmt(s.tools.length)} tools${esc(info)}</span></div>` +
      (s.error ? `<div class="msg-error">${esc(s.error)}</div>` : "") +
      (s.tools.length ? `<div class="mcp-server__tools">${s.tools.map((t) => `<span class="chip" title="${esc(t.description || "")}">${esc(t.tool)}</span>`).join("")}</div>` : "") +
      `</div>`;
  }).join("");
}

// ------------------------------------------------------------------ Markdown (escaped first, then formatted)
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code class="inline">${esc(codes[+i])}</code>`);
}
function codeBlock(lang, code) {
  return `<div class="st-code"><div class="st-code__head"><span>${esc(lang || "code")}</span>` +
    `<button class="st-btn st-btn--icon" data-code-copy aria-label="Copy code">${icon("copy")}</button></div>` +
    `<pre><code>${esc(code)}</code></pre></div>`;
}
function blocks(text) {
  const out = [], lines = text.split("\n");
  let para = [], list = null;
  const flushPara = () => { if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; };
  const flushList = () => { if (list) out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; };
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    let m;
    if (!l.trim()) { flushPara(); flushList(); continue; }
    if ((m = l.match(/^(#{1,6})\s+(.*)$/))) { flushPara(); flushList(); out.push(`<${m[1].length <= 2 ? "h3" : "h4"}>${inline(m[2])}</${m[1].length <= 2 ? "h3" : "h4"}>`); continue; }
    if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(l)) { flushPara(); flushList(); out.push("<hr>"); continue; }
    if ((m = l.match(/^>\s?(.*)$/))) { flushPara(); flushList(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); continue; }
    if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])) {
      flushPara(); flushList();
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      let html = `<table><thead><tr>${cells(l).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) html += `<tr>${cells(lines[i++]).map((c) => `<td>${c}</td>`).join("")}</tr>`;
      i--;
      out.push(html + "</tbody></table>");
      continue;
    }
    if ((m = l.match(/^\s*(?:[-*+]|(\d+)[.)])\s+(.*)$/))) {
      flushPara();
      const tag = m[1] ? "ol" : "ul";
      if (!list || list.tag !== tag) { flushList(); list = {tag, items: []}; }
      list.items.push(m[2]);
      continue;
    }
    if (list && /^\s{2,}\S/.test(l)) { list.items[list.items.length - 1] += " " + l.trim(); continue; }
    flushList();
    para.push(l);
  }
  flushPara(); flushList();
  return out.join("");
}
function markdown(text) {
  let html = "", rest = text;
  for (;;) {
    const m = rest.match(/(^|\n)```([^\n`]*)\n/);
    if (!m) { html += blocks(rest); break; }
    html += blocks(rest.slice(0, m.index));
    rest = rest.slice(m.index + m[0].length);
    const end = rest.match(/(^|\n)```[ \t]*(\n|$)/);
    if (!end) { html += codeBlock(m[2].trim(), rest); break; }         // still streaming
    html += codeBlock(m[2].trim(), rest.slice(0, end.index));
    rest = rest.slice(end.index + end[0].length);
  }
  return html;
}

// ------------------------------------------------------------------ Chat
const DEFAULTS = {thinking: "high", temperature: 0.6, top_p: 0.95, top_k: 20, max: "", seed: "", show: true, esp: true, mcp: true};
let settings = {...DEFAULTS, ...store.get("sampling", {})};
let messages = [];                    // the open chat (set by showChat)
let attachments = [];                 // {name, url}
let busy = null;                      // {controller, msg}

// ------------------------------------------------------------------ workspace (serve/workspace.py)
// Projects and chats live on this Strata server (POST /workspace/<op>), so every device that opens it sees the same
// ones. This browser only remembers which project and chat it shows (strata.project, strata.chat_open). Chats kept
// by the browser-only sidebar (strata.chats, strata.chat.<id>) and the single chat of earlier versions (strata.chat)
// move to the server once.
let ws = {projects: [], chats: [], roots: [], limits: {}};
let wsState = "loading";                       // loading | ready | key (API key needed) | off | error
let chatId = null, chatProject = null;         // the open chat and its project
let viewProject = store.get("project", null);  // the project the sidebar shows (null: all chats)
let searchHits = null, wsLoaded = false, unsavedWarned = false;
const projCtx = new Map();                     // project id -> {time, text}: its system message, cached
const newId = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
const projectOf = (id) => ws.projects.find((p) => p.id === id) || null;
// what is kept of a message: pictures and files by name (their data is in the request, not the store)
const keepable = (msgs) => msgs.map((m) => ({...m, images: (m.images || []).map((i) => ({name: i.name})),
                                              files: (m.files || []).map((f) => ({name: f.name}))}));
function chatTitle(msgs) {
  const u = msgs.find((m) => m.role === "user");
  const t = u ? u.text || (u.files && u.files.length ? u.files[0].name : "") || (u.images && u.images.length ? "Picture" : "") : "";
  const s = t.replace(/\s+/g, " ").trim();
  return s ? (s.length > 60 ? `${s.slice(0, 57)}…` : s) : "New chat";
}
async function wsCall(op, body = {}) {
  const r = await fetch(`workspace/${op}`, {method: "POST", headers: headers(true), body: JSON.stringify(body)});
  let j = {};
  try { j = await r.json(); } catch (e) { /* not json */ }
  if (!r.ok) {
    const err = new Error((j.error && j.error.message) || `HTTP ${r.status}`);
    err.status = r.status;
    throw err;
  }
  return j;
}
async function loadWorkspace() {
  try {
    ws = await wsCall("state");
    wsState = "ready";
  } catch (e) {
    wsState = e.status === 401 ? "key" : e.status === 404 ? "off" : "error";
    renderSide();
    renderRoots();
    return;
  }
  if (viewProject && !projectOf(viewProject)) { viewProject = null; store.set("project", null); }
  if (!wsLoaded) await importLocalChats();
  renderSide();
  renderProjectBar();
  if (tab === "files" && !fs.loaded) fsOpen("");
  renderRoots();
  if (!wsLoaded) {                               // the first load: the chat this browser showed last
    wsLoaded = true;
    const open = store.get("chat_open", null);
    if (open && !messages.length && ws.chats.some((c) => c.id === open)) await openChat(open, true);
    else if (!messages.length) chatProject = viewProject;
  }
}
async function importLocalChats() {
  const idx = store.get("chats", null), old = store.get("chat", null), list = [];
  if (Array.isArray(idx)) for (const c of idx) list.push({...c, messages: store.get(`chat.${c.id}`, [])});
  if (Array.isArray(old) && old.length) list.push({id: newId(), title: chatTitle(old), time: old[old.length - 1].time || Date.now(), messages: old});
  if (!list.length) return;
  let moved = 0;
  for (const c of list) {
    if (!Array.isArray(c.messages) || !c.messages.length || ws.chats.some((x) => x.id === c.id)) continue;
    try {
      await wsCall("chat.save", {id: c.id, project: null, title: c.title || chatTitle(c.messages), named: !!c.named,
                                 time: c.time || Date.now(), messages: c.messages});
      moved++;
    } catch (e) {
      toast("error", "This browser's chats were not moved to the server", e.message, 8000);   // kept for the next try
      return;
    }
  }
  for (const c of idx || []) store.del(`chat.${c.id}`);
  store.del("chats");
  store.del("chat");
  if (moved) {
    ws = await wsCall("state");
    toast("success", `${moved} chat${moved > 1 ? "s" : ""} moved to this server`, "Every device that opens Strata now sees them.", 6000);
  }
}
function saveChat() {
  if (!messages.length) return;
  if (wsState !== "ready") {
    if (!unsavedWarned) { unsavedWarned = true; toast("warn", "This chat is not saved", wsState === "key" ? "Add the API key under Settings." : "The server's workspace is not reachable.", 8000); }
    return;
  }
  if (!chatId) chatId = newId();
  let c = ws.chats.find((x) => x.id === chatId);
  if (!c) { c = {id: chatId, project: chatProject, title: "", named: false, time: 0}; ws.chats.unshift(c); }
  if (!c.named) c.title = chatTitle(messages);
  c.time = Date.now();
  c.project = chatProject;
  ws.chats.sort((a, b) => b.time - a.time);
  store.set("chat_open", chatId);
  wsCall("chat.save", {id: c.id, project: c.project, title: c.title, named: !!c.named, time: c.time, messages: keepable(messages)})
    .catch((e) => toast("error", "This chat was not saved", e.message, 8000));
  renderSide();
}
function showChat(id, msgs, project) {
  chatId = id;
  messages = msgs;
  chatProject = project || null;
  store.set("chat_open", id);
  renderChat();
  renderSide();
  renderProjectBar();
  refreshChanges();
  if (narrowMq.matches) setSide(false);
  $("input").focus();
}
async function openChat(id, quiet) {
  if (busy) { toast("warn", "Still writing", "Stop the answer first."); return; }
  if (id === chatId) { if (narrowMq.matches) setSide(false); return; }
  try {
    const {chat} = await wsCall("chat.get", {id});
    showChat(id, chat.messages || [], chat.project);
  } catch (e) {
    if (!quiet) toast("error", "Could not open the chat", e.message);
  }
}
function newChat() {
  if (busy) { toast("warn", "Still writing", "Stop the answer first."); return; }
  showChat(null, [], viewProject);
}
async function deleteChat(id) {
  if (busy && id === chatId) { toast("warn", "Still writing", "Stop the answer first."); return; }
  const c = ws.chats.find((x) => x.id === id);
  if (!c) return;
  const wasOpen = id === chatId;
  let data;
  try {
    data = wasOpen ? {project: chatProject, messages: keepable(messages)} : (await wsCall("chat.get", {id})).chat;
    await wsCall("chat.delete", {id});
  } catch (e) {
    toast("error", "Could not delete the chat", e.message);
    return;
  }
  ws.chats = ws.chats.filter((x) => x.id !== id);
  if (searchHits) searchHits = searchHits.filter((x) => x.id !== id);
  if (wasOpen) showChat(null, [], chatProject); else renderSide();
  toast("info", "Chat deleted", c.title, 6000, {label: "Undo", run: async () => {
    try {
      await wsCall("chat.save", {id, project: data.project || null, title: c.title, named: !!c.named, time: c.time,
                                 messages: data.messages || []});
    } catch (e) { toast("error", "Could not restore the chat", e.message); return; }
    ws.chats.push({...c, project: data.project || null});
    ws.chats.sort((a, b) => b.time - a.time);
    if (wasOpen && !messages.length && !busy) showChat(id, data.messages || [], data.project); else renderSide();
  }});
}
// an inline editor in a list row: Enter or leaving it saves, Escape cancels
function inlineEdit(li, el, onSave) {
  li.replaceChildren(el);
  el.focus();
  if (el.select) el.select();
  let done = false;
  const finish = async (save) => {
    if (done) return;
    done = true;
    if (save) await onSave(el.value);
    renderSide();
  };
  el.onkeydown = (e) => {
    if (e.key === "Enter") { e.preventDefault(); finish(true); }
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); finish(false); }
  };
  el.onblur = () => finish(true);
  return finish;
}
function renameChat(id, li) {
  const c = ws.chats.find((x) => x.id === id);
  if (!c) return;
  const input = document.createElement("input");
  Object.assign(input, {className: "st-input chat-item__edit", value: c.title, maxLength: 120});
  input.setAttribute("aria-label", "Chat name");
  inlineEdit(li, input, async (v) => {
    v = v.replace(/\s+/g, " ").trim();
    if (!v || v === c.title) return;
    try { await wsCall("chat.meta", {id, title: v}); Object.assign(c, {title: v, named: true}); }
    catch (e) { toast("error", "Could not rename the chat", e.message); }
  });
}
function moveChat(id, li) {
  const c = ws.chats.find((x) => x.id === id);
  if (!c) return;
  const sel = document.createElement("select");
  sel.className = "st-input chat-item__edit";
  sel.setAttribute("aria-label", "Move to project");
  sel.innerHTML = `<option value="">No project</option>` + ws.projects.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join("");
  sel.value = c.project || "";
  const finish = inlineEdit(li, sel, async (v) => {
    const pid = v || null;
    if (pid === (c.project || null)) return;
    try { await wsCall("chat.meta", {id, project: pid}); c.project = pid; }
    catch (e) { toast("error", "Could not move the chat", e.message); return; }
    if (id === chatId) { chatProject = pid; renderProjectBar(); }
    toast("success", "Moved", pid ? `to ${projectOf(pid).name}` : "out of its project", 2500);
  });
  sel.onchange = () => finish(true);
}
function whenStr(t) {
  const d = new Date(t);
  return d.toDateString() === new Date().toDateString() ? timeStr(t) : d.toLocaleDateString([], {day: "numeric", month: "short"});
}
const svgUse = (id) => `<svg class="st-icon st-icon--sm" aria-hidden="true"><use href="#${id}"/></svg>`;
function renderSide() {
  // the projects
  const pl = $("project-list");
  const count = (pid) => ws.chats.filter((c) => (c.project || null) === pid).length;
  pl.innerHTML = wsState !== "ready" ? "" :
    `<li class="chat-item" data-project=""${viewProject ? "" : ' aria-current="true"'}><button type="button" class="chat-item__open" data-project-open>` +
    `<span class="chat-item__title">All chats</span></button><span class="muted small side-count">${fmt(ws.chats.length)}</span></li>` +
    ws.projects.map((p) => `<li class="chat-item" data-project="${esc(p.id)}"${p.id === viewProject ? ' aria-current="true"' : ""}>` +
      `<button type="button" class="chat-item__open" data-project-open title="${esc(p.name)}"><span class="chat-item__title">` +
      `${svgUse("i-folder")} ${esc(p.name)}</span></button><span class="muted small side-count">${fmt(count(p.id))}</span>` +
      `<button type="button" class="st-btn st-btn--icon" data-project-edit aria-label="Instructions and files" title="Instructions and files">` +
      `${icon("settings", "st-icon st-icon--sm")}</button></li>`).join("");
  // the chats: the search's hits, else the selected project's chats
  const list = $("chat-list");
  const p = projectOf(viewProject);
  $("chats-heading").textContent = searchHits ? "Found" : p ? p.name : "Chats";
  if (wsState === "key") { list.innerHTML = `<li class="chat-list__empty muted small">Add the API key under Settings to see your projects and chats.</li>`; return; }
  if (wsState === "off") { list.innerHTML = `<li class="chat-list__empty muted small">This server has no workspace.</li>`; return; }
  if (wsState === "error") { list.innerHTML = `<li class="chat-list__empty muted small">The workspace is not reachable right now.</li>`; return; }
  const rows = searchHits || (viewProject ? ws.chats.filter((c) => c.project === viewProject) : ws.chats);
  if (!rows.length) {
    list.innerHTML = `<li class="chat-list__empty muted small">${searchHits ? "Nothing found." : wsState === "loading" ? "Loading…" : "Your chats appear here."}</li>`;
    return;
  }
  list.innerHTML = rows.map((c) => {
    const tag = !viewProject || searchHits ? (projectOf(c.project) ? ` · ${esc(projectOf(c.project).name)}` : "") : "";
    const snip = searchHits && c.snippet ? `<span class="chat-item__snip muted small">${esc(c.snippet)}</span>` : "";
    return `<li class="chat-item" data-id="${esc(c.id)}"${c.id === chatId ? ' aria-current="true"' : ""}>` +
      `<button type="button" class="chat-item__open" data-chat-open title="${esc(c.title)}"><span class="chat-item__title">${esc(c.title)}</span>` +
      `${snip}<span class="muted small">${esc(whenStr(c.time))}${tag}</span></button>` +
      `<button type="button" class="st-btn st-btn--icon" data-chat-move aria-label="Move to project" title="Move to project">${svgUse("i-folder")}</button>` +
      `<button type="button" class="st-btn st-btn--icon" data-chat-rename aria-label="Rename" title="Rename">${svgUse("i-rename")}</button>` +
      `<button type="button" class="st-btn st-btn--icon" data-chat-delete aria-label="Delete" title="Delete">${icon("trash", "st-icon st-icon--sm")}</button></li>`;
  }).join("");
}
const renderChatList = renderSide;
$("chat-list").addEventListener("click", (e) => {
  const li = e.target.closest(".chat-item");
  if (!li || !li.dataset.id) return;
  if (e.target.closest("[data-chat-rename]")) renameChat(li.dataset.id, li);
  else if (e.target.closest("[data-chat-move]")) moveChat(li.dataset.id, li);
  else if (e.target.closest("[data-chat-delete]")) deleteChat(li.dataset.id);
  else if (e.target.closest("[data-chat-open]")) openChat(li.dataset.id);
});
$("project-list").addEventListener("click", (e) => {
  const li = e.target.closest(".chat-item");
  if (!li || li.dataset.project === undefined) return;
  const pid = li.dataset.project || null;
  if (e.target.closest("[data-project-edit]")) { openProjectDrawer(pid); return; }
  viewProject = pid;
  store.set("project", pid);
  clearSearch();
  if (!messages.length && !busy) { chatProject = pid; renderProjectBar(); }   // a new chat starts in this project
  renderSide();
});
$("proj-new").onclick = () => {
  if (wsState !== "ready") { toast("warn", "No workspace", "Add the API key under Settings."); return; }
  const li = document.createElement("li");
  li.className = "chat-item";
  $("project-list").appendChild(li);
  const input = document.createElement("input");
  Object.assign(input, {className: "st-input chat-item__edit", placeholder: "Project name", maxLength: 120});
  inlineEdit(li, input, async (v) => {
    v = v.trim();
    if (!v) return;
    try {
      const {project} = await wsCall("project.save", {name: v});
      ws.projects.push(project);
      viewProject = project.id;
      store.set("project", project.id);
      if (!messages.length && !busy) chatProject = project.id;
      renderProjectBar();
      openProjectDrawer(project.id);
    } catch (e) { toast("error", "Could not create the project", e.message); }
  });
};
// search across every chat's title and messages (on the server)
let searchTimer = null;
function clearSearch() { $("chat-search").value = ""; searchHits = null; }
$("chat-search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  const q = $("chat-search").value.trim();
  if (!q) { searchHits = null; renderSide(); return; }
  searchTimer = setTimeout(async () => {
    try { searchHits = (await wsCall("search", {q})).hits; } catch (e) { searchHits = []; }
    if ($("chat-search").value.trim() === q) renderSide();
  }, 250);
});
$("chat-search").addEventListener("keydown", (e) => { if (e.key === "Escape" && $("chat-search").value) { e.stopPropagation(); clearSearch(); renderSide(); } });
// other devices may have changed things: look again when this page comes back
document.addEventListener("visibilitychange", () => { if (!document.hidden && wsLoaded && !busy) loadWorkspace(); });

// ------------------------------------------------------------------ the project bar and drawer
function renderProjectBar() {
  const p = projectOf(chatProject);
  $("project-bar").hidden = !p;
  $("view-chat").classList.toggle("has-project", !!p);
  if (!p) return;
  $("project-bar-name").textContent = p.name;
  $("project-bar-folder").textContent = p.folder || "";
  $("project-bar-folder").title = p.folder || "";
  $("project-mode").hidden = !p.folder;
  $("project-mode").value = p.mode || "ask";
  $("project-mode").dataset.mode = p.mode || "ask";
  renderChangesButton();
}
$("project-mode").onchange = async () => {
  const mode = $("project-mode").value;
  try {
    const {project} = await wsCall("project.save", {id: chatProject, mode});
    updateProject(project);
    renderProjectBar();
    for (const [id, a] of approvals) {           // what waits for you and this mode allows by itself: go on
      if (!needsApproval(mode, a.name, a.m, a.args, project.allow)) { approvals.delete(id); a.resolve(true); }
    }
    toast(mode === "auto" ? "warn" : "info", MODES[mode],
          {read: "The model can look, not change.", ask: "You approve every change and command.",
           edit: "Edits run right away; commands wait for you.", auto: "Changes and commands run without asking: watch it."}[mode], 4000);
  } catch (e) { toast("error", "Could not switch the mode", e.message); renderProjectBar(); }
};
$("project-bar-edit").onclick = () => openProjectDrawer(chatProject);
let drawerProject = null, deleteArmed = null;
function openProjectDrawer(pid) {
  const p = projectOf(pid);
  if (!p) return;
  drawerProject = pid;
  $("pd-name").value = p.name;
  $("pd-head-name").textContent = p.folder ? `${p.name} · ${p.folder}` : p.name;
  $("pd-instructions").value = p.instructions || "";
  $("pd-folder").value = p.folder || "";
  $("pd-rounds").value = p.max_rounds == null ? 200 : p.max_rounds;
  $("pd-compact").value = p.compact_at == null ? 100000 : p.compact_at;
  renderProjectFiles(p);
  renderAllowRules(p);
  disarmDelete();
  $("proj-drawer").dataset.open = "true";
  $("proj-drawer").setAttribute("aria-hidden", "false");
  $("proj-scrim").hidden = false;
  $("pd-instructions").focus({preventScroll: true});
  $("pd-instructions").setSelectionRange(0, 0);           // the text from its start, not scrolled to its end
  $("pd-instructions").scrollTop = 0;
}
function closeProjectDrawer() {
  $("proj-drawer").dataset.open = "false";
  $("proj-drawer").setAttribute("aria-hidden", "true");
  $("proj-scrim").hidden = true;
  drawerProject = null;
}
function renderProjectFiles(p) {
  const total = p.files.reduce((s, f) => s + f.size, 0);
  $("pd-files-sum").textContent = p.files.length ? `${p.files.length} file${p.files.length > 1 ? "s" : ""} · ${kfmt(total)} characters, sent with every message of this project` : "";
  $("pd-files").innerHTML = p.files.length ? p.files.map((f) => `<li class="pd-row" data-file="${esc(f.id)}">` +
    `${icon("attach", "st-icon st-icon--sm")}<span class="pd-row__text" title="${esc(f.source || f.name)}">${esc(f.name)}</span>` +
    `<span class="pd-row__meta">${kfmt(f.size)}</span>` +
    `<button type="button" class="st-btn st-btn--icon" data-file-delete aria-label="Remove ${esc(f.name)}" title="Remove">${icon("trash", "st-icon st-icon--sm")}</button></li>`).join("")
    : `<li class="pd-empty">No files yet: upload text files here, or add them from the Files tab.</li>`;
}
function updateProject(p) {
  const i = ws.projects.findIndex((x) => x.id === p.id);
  if (i >= 0) ws.projects[i] = p; else ws.projects.push(p);
  projCtx.delete(p.id);
}
$("pd-save").onclick = async () => {
  const name = $("pd-name").value.trim();
  if (!name) { toast("warn", "A project needs a name"); return; }
  try {
    const {project} = await wsCall("project.save", {id: drawerProject, name, instructions: $("pd-instructions").value,
                                                     folder: $("pd-folder").value.trim() || null,
                                                     max_rounds: Math.max(0, parseInt($("pd-rounds").value, 10) || 0),
                                                     compact_at: Math.max(0, parseInt($("pd-compact").value, 10) || 0)});
    updateProject(project);
    closeProjectDrawer();
    renderSide();
    renderProjectBar();
    toast("success", "Project saved", "Its instructions and files go with every message in it.");
  } catch (e) { toast("error", "Could not save the project", e.message); }
};
$("pd-files").addEventListener("click", async (e) => {
  const li = e.target.closest("[data-file]");
  if (!li || !e.target.closest("[data-file-delete]")) return;
  try {
    const {project} = await wsCall("project.file.delete", {project: drawerProject, id: li.dataset.file});
    updateProject(project);
    renderProjectFiles(project);
  } catch (err) { toast("error", "Could not remove the file", err.message); }
});
$("pd-upload").onclick = () => $("pd-file").click();
$("pd-file").onchange = async () => {
  for (const f of $("pd-file").files) {
    if (!isTextFile(f)) { toast("warn", "Not a text file", f.name); continue; }
    if (f.size > MAX_TEXT_FILE) { toast("warn", "File too large", `${f.name} is over 512 KB.`); continue; }
    const text = await f.text();
    if (text.includes("\u0000")) { toast("warn", "Not a text file", `${f.name} looks like a binary file.`); continue; }
    try {
      const {project} = await wsCall("project.file.add", {project: drawerProject, name: f.name, text});
      updateProject(project);
      renderProjectFiles(project);
    } catch (e) { toast("error", `Could not add ${f.name}`, e.message); }
  }
  $("pd-file").value = "";
};
function disarmDelete() { clearTimeout(deleteArmed); deleteArmed = null; $("pd-delete").textContent = "Delete project"; }
$("pd-delete").onclick = async () => {
  if (!deleteArmed) {                           // two clicks: the first only asks
    $("pd-delete").textContent = "Click again: delete (its chats stay)";
    deleteArmed = setTimeout(disarmDelete, 4000);
    return;
  }
  disarmDelete();
  const pid = drawerProject;
  try { await wsCall("project.delete", {id: pid}); } catch (e) { toast("error", "Could not delete the project", e.message); return; }
  ws.projects = ws.projects.filter((p) => p.id !== pid);
  for (const c of ws.chats) if (c.project === pid) c.project = null;
  if (viewProject === pid) { viewProject = null; store.set("project", null); }
  if (chatProject === pid) chatProject = null;
  closeProjectDrawer();
  renderSide();
  renderProjectBar();
  toast("info", "Project deleted", "Its chats are under All chats now.");
};
$("pd-close").onclick = closeProjectDrawer;
$("proj-scrim").onclick = closeProjectDrawer;
// the system message of a chat in a project: its instructions and its files
async function projectSystem(pid) {
  const p = projectOf(pid);
  if (!p) return "";
  const hit = projCtx.get(pid);
  if (hit && hit.time === p.time) return hit.text;
  const ctx = await wsCall("project.context", {project: pid});
  const parts = [`You are working in the project "${ctx.name}".`];
  if (ctx.folder) {
    const modeText = {read: "read only: you can look at everything but change nothing; suggest changes as diffs in your answer",
                      ask: "the user approves every change and command before it runs",
                      edit: "file changes run right away; the user approves every command",
                      auto: "every change and command runs right away"}[ctx.mode] || ctx.mode;
    parts.push(`You are a coding agent. The project's folder is ${ctx.folder}; your tools work in it and take paths relative ` +
      `to it. Look at the code with the tools before you answer or change anything, and read a file before you edit it ` +
      `(edit_file needs the exact text). Prefer small, focused edits; run the tests or a build with run_command when it helps ` +
      `to check your work. For a task with several steps, first write a short plan with update_plan and keep it ` +
      `current. When you are done, say briefly what you changed and why. Permission mode: ${modeText}.`);
    if (ctx.guide) parts.push(`The repository's ${ctx.guide.name}:\n\n${ctx.guide.text.trim()}`);
    if (ctx.skills && ctx.skills.length) {          // its agent skills: the model reads the one a task needs
      parts.push("The project has skills: instructions for certain kinds of work. When a task matches one, read its file " +
        "with read_file before you start and follow it; say which skill you use.\n\n" +
        ctx.skills.map((k) => `- ${k.name} (${k.path}): ${k.description}`).join("\n"));
    }
  }
  if ((ctx.instructions || "").trim()) parts.push(ctx.instructions.trim());
  if (ctx.files.length) parts.push("The project's files:\n\n" + ctx.files.map(fileBlock).join("\n\n"));
  const text = parts.join("\n\n");
  projCtx.set(pid, {time: p.time, text});
  return text;
}

// ------------------------------------------------------------------ Files: the server's shared folders, read only
const fs = {loaded: false, path: "", parent: null, file: null};
function sizeStr(n) { return n == null ? "" : n >= 1048576 ? `${fmt(n / 1048576, 1)} MB` : n >= 1024 ? `${fmt(n / 1024)} KB` : `${fmt(n)} B`; }
async function fsOpen(path) {
  fs.loaded = true;
  const list = $("fs-list");
  if (wsState !== "ready") {
    list.innerHTML = `<li class="chat-list__empty muted small">${wsState === "key" ? "Add the API key under Settings." : "The workspace is not reachable."}</li>`;
    return;
  }
  if (!ws.roots.length) {
    list.innerHTML = `<li class="chat-list__empty muted small">No folder is shared yet. ` +
      `<a href="#settings">Share one under Settings</a> to browse it here.</li>`;
    $("fs-where").textContent = "";
    return;
  }
  let d;
  try { d = await wsCall("fs.list", {path}); } catch (e) { toast("error", "Could not open the folder", e.message); return; }
  Object.assign(fs, {path: d.path, parent: d.parent});
  $("fs-where").textContent = d.path ? `\u200e${d.path}\u200e` : "Shared folders";   // marks: the "/" stays in front (rtl box)
  $("fs-up").disabled = !d.path;
  $("fs-as-project").hidden = !d.path;
  list.innerHTML = d.entries.map((x) => `<li class="chat-item" aria-current="${!!fs.file && fs.file.path === x.path}" data-path="${esc(x.path)}" data-dir="${x.dir ? 1 : ""}">` +
    `<button type="button" class="chat-item__open fs-row">${x.dir ? svgUse("i-folder") : icon("attach", "st-icon st-icon--sm")}` +
    `<span class="chat-item__title">${esc(x.name)}</span><span class="muted small">${x.dir ? "" : esc(sizeStr(x.size))}</span></button></li>`).join("") +
    (d.more ? `<li class="chat-list__empty muted small">and ${fmt(d.more)} more not shown</li>` : "") +
    (!d.entries.length ? `<li class="chat-list__empty muted small">Empty folder</li>` : "");
}
$("fs-up").onclick = () => fsOpen(fs.parent || "");
// this folder as a project: the model works in it with the coding tools (the mode starts at "Ask before changes")
$("fs-as-project").onclick = async () => {
  const have = ws.projects.find((p) => p.folder === fs.path);
  let project = have;
  if (!have) {
    try {
      project = (await wsCall("project.save", {name: fs.path.split(/[\\/]/).filter(Boolean).pop() || fs.path, folder: fs.path})).project;
      ws.projects.push(project);
    } catch (e) { toast("error", "Could not make the project", e.message); return; }
  }
  viewProject = project.id;
  store.set("project", project.id);
  showTab("chat");
  if (busy) { renderSide(); return; }
  showChat(null, [], project.id);
  toast("success", have ? `Project ${project.name}` : `New project ${project.name}`, `The model works in ${fs.path} (${MODES[project.mode || "ask"]}).`, 5000);
};
$("fs-list").addEventListener("click", async (e) => {
  const li = e.target.closest("[data-path]");
  if (!li) return;
  if (li.dataset.dir) { fsOpen(li.dataset.path); return; }
  openPath(li.dataset.path);
});
// a file from the list or from a link in a Markdown file; a link to a folder opens the folder
async function openPath(path, anchor = "") {
  let f;
  try { f = await wsCall("fs.read", {path}); }
  catch (err) {
    if (err.status === 400 && /not a file/.test(err.message)) fsOpen(path);
    else toast("warn", "Cannot show this file", err.message);
    return;
  }
  fs.file = f;
  const dir = parentOf(f.path);
  if (dir && dir !== fs.path && !narrowMq.matches) await fsOpen(dir);    // the list shows where the file is
  for (const li of $("fs-list").querySelectorAll("[data-path]")) li.setAttribute("aria-current", String(li.dataset.path === f.path));
  $("fs-name").textContent = f.name;
  $("fs-meta").textContent = `${f.path} · ${sizeStr(f.size)}`;
  $("fs-actions").hidden = false;
  await showFile(f);
  if (anchor) scrollToAnchor(anchor);
  if (narrowMq.matches) $("fs-name").scrollIntoView({block: "start", behavior: "smooth"});
  $("fs-project").innerHTML = ws.projects.length ? ws.projects.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join("")
                                                  : `<option value="">No project yet</option>`;
  $("fs-project").value = chatProject || viewProject || (ws.projects[0] && ws.projects[0].id) || "";
  $("fs-add-project").disabled = !ws.projects.length;
}
$("fs-attach").onclick = () => {
  if (!fs.file) return;
  attachments.push({kind: "file", name: fs.file.name, text: fs.file.text});
  renderAttachments();
  showTab("chat");
  toast("success", "Attached", `${fs.file.name} goes with your next message.`, 2500);
};
$("fs-add-project").onclick = async () => {
  const pid = $("fs-project").value;
  if (!fs.file || !pid) return;
  try {
    const {project} = await wsCall("project.file.add", {project: pid, path: fs.file.path});
    updateProject(project);
    toast("success", "Added to the project", `${fs.file.name} → ${project.name}`, 3000);
  } catch (e) { toast("error", "Could not add the file", e.message); }
};

// ------------------------------------------------------------------ the file viewer: Markdown and highlighted code
// marked (Markdown), DOMPurify (cleans what marked makes: a file cannot run script in this page or load anything from
// the internet) and highlight.js are served from serve/web like the app's own files (VENDOR.md), loaded on first use
let viewerLibs = null;
function loadViewer() {
  viewerLibs ||= ["vendor-marked.js", "vendor-purify.js", "vendor-hljs.js"].reduce((done, name) => done.then(() => new Promise((ok, fail) => {
    const el = document.createElement("script");
    el.src = `web/${name}`;
    el.onload = ok;
    el.onerror = () => fail(new Error(`web/${name} did not load`));
    document.head.appendChild(el);
  })), Promise.resolve()).then(() => {
    DOMPurify.addHook("afterSanitizeAttributes", (node) => {
      if (node.tagName === "IMG") {                  // shown by wireMarkdown: from the shared folders, never the internet
        node.setAttribute("data-src", node.getAttribute("src") || "");
        node.removeAttribute("src");
        node.removeAttribute("srcset");
      }
    });
  }).catch((e) => { viewerLibs = null; throw e; });
  return viewerLibs;
}
const LANG_BY_EXT = {cu: "cpp", cuh: "cpp", jl: "julia", vue: "xml", svelte: "xml", gradle: "groovy", ipynb: "json",
  jsonl: "json", cfg: "ini", conf: "ini", env: "bash", mjs: "javascript", cjs: "javascript", mdx: "markdown"};
const LANG_BY_NAME = {dockerfile: "dockerfile", containerfile: "dockerfile", makefile: "makefile", gnumakefile: "makefile",
  "cmakelists.txt": "cmake", ".bashrc": "bash", ".zshrc": "bash", ".profile": "bash", ".env": "bash", "nginx.conf": "nginx",
  ".htaccess": "apache", "jenkinsfile": "groovy", "vagrantfile": "ruby", "gemfile": "ruby", "rakefile": "ruby"};
function fileLang(name) {
  const n = name.toLowerCase();
  if (LANG_BY_NAME[n]) return LANG_BY_NAME[n];
  if (/^dockerfile\.|\.dockerfile$/.test(n)) return "dockerfile";
  if (/^\.env\./.test(n)) return "bash";
  const ext = n.includes(".") ? n.split(".").pop() : "";
  const lang = LANG_BY_EXT[ext] || ext;
  return lang && hljs.getLanguage(lang) ? lang : "plaintext";
}
const isMarkdown = (name) => /\.(md|markdown|mdx)$/i.test(name);
const HL_MAX = 300 * 1024;                       // larger files are shown without colours (colouring them stalls the page)
const fsView = {mode: store.get("fs.view", "preview"), wrap: store.get("fs.wrap", false)};

// highlighted code, one element per line (the spans that cross a line break are closed and opened again)
function codeLines(text, lang) {
  let html = esc(text);
  if (lang !== "plaintext" && text.length <= HL_MAX) {
    try { html = hljs.highlight(text, {language: lang, ignoreIllegals: true}).value; } catch (e) { /* shown plain */ }
  }
  const lines = [], open = [];
  let line = "";
  for (const part of html.split(/(<span[^>]*>|<\/span>|\n)/)) {
    if (part === "\n") { lines.push(line + "</span>".repeat(open.length)); line = open.join(""); }
    else if (part.startsWith("<span")) { open.push(part); line += part; }
    else if (part === "</span>") { open.pop(); line += part; }
    else line += part;
  }
  lines.push(line + "</span>".repeat(open.length));
  if (lines.length > 1 && text.endsWith("\n")) lines.pop();
  return {count: lines.length, html: lines.map((l) => `<span class="cl"><span class="cl__t">${l}</span></span>`).join("")};
}
function markdownHtml(text) {
  const fm = text.match(/^---\r?\n([\s\S]{0,4000}?)\r?\n---\r?\n/);   // front matter: shown as YAML, not as a rule and text
  const body = fm ? "```yaml\n" + fm[1] + "\n```\n\n" + text.slice(fm[0].length) : text;
  return DOMPurify.sanitize(marked.parse(body, {gfm: true, async: false}),
    {FORBID_TAGS: ["style", "form", "video", "audio", "source", "track", "picture", "object", "embed"], FORBID_ATTR: ["style"]});
}
async function showFile(f) {
  const box = $("fs-text"), md = isMarkdown(f.name);
  box.scrollTop = 0;
  try { await loadViewer(); } catch (e) {
    box.className = "files-text files-text--plain";
    box.textContent = f.text;
    toast("warn", "Shown as plain text", e.message);
    return;
  }
  if (fs.file !== f) return;                         // another file was opened meanwhile
  const lang = md ? "markdown" : fileLang(f.name), preview = md && fsView.mode === "preview";
  $("fs-lang").hidden = false;
  $("fs-lang").textContent = lang === "plaintext" ? "Text" : (hljs.getLanguage(lang).name || lang);
  $("fs-modes").hidden = !md;
  for (const b of $("fs-modes").querySelectorAll("[data-view]")) b.setAttribute("aria-pressed", String((b.dataset.view === "preview") === preview));
  $("fs-wrap").hidden = preview;
  $("fs-wrap").setAttribute("aria-pressed", String(fsView.wrap));
  $("fs-copy").hidden = false;
  if (preview) {
    box.className = "files-text files-text--doc";
    box.innerHTML = `<article class="md-doc">${markdownHtml(f.text)}</article>`;
    wireMarkdown(box, f);
  } else {
    const code = codeLines(f.text, lang);
    box.className = `files-text files-text--code${fsView.wrap ? " is-wrap" : ""}`;
    box.innerHTML = `<pre class="code-view hljs" style="--ln:${String(code.count).length}ch">${code.html}</pre>`;
  }
}
// a path relative to a file (a link or a picture in it); "/x" starts at the shared folder the file is in
function parentOf(path) {
  const i = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return i > 0 ? path.slice(0, i) : i === 0 ? "/" : "";
}
function resolvePath(file, rel) {
  try { rel = decodeURIComponent(rel); } catch (e) { /* kept as written */ }
  const sep = file.includes("\\") && !file.includes("/") ? "\\" : "/";
  const root = ws.roots.filter((r) => file === r || file.startsWith(r.endsWith(sep) ? r : r + sep)).sort((a, b) => b.length - a.length)[0];
  const parts = (rel.startsWith("/") && root ? root : parentOf(file)).split(/[\\/]/);
  for (const seg of rel.split(/[\\/]/)) {
    if (!seg || seg === ".") continue;
    if (seg === "..") { if (parts.length > 1) parts.pop(); } else parts.push(seg);
  }
  return parts.join(sep) || sep;
}
const slug = (t) => t.toLowerCase().trim().replace(/[^\p{L}\p{N}\s_-]/gu, "").replace(/\s+/g, "-");
function wireMarkdown(box, f) {
  const seen = new Map();
  for (const h of box.querySelectorAll("h1, h2, h3, h4, h5, h6")) {   // anchors like GitHub's, kept apart from the page's ids
    const base = slug(h.textContent), n = seen.get(base) || 0;
    seen.set(base, n + 1);
    h.id = `md-${n ? `${base}-${n}` : base}`;
  }
  for (const el of box.querySelectorAll("pre code")) {
    const lang = ((el.className.match(/language-([\w+#-]+)/) || [])[1] || "").toLowerCase();
    const known = lang && hljs.getLanguage(LANG_BY_EXT[lang] || lang);
    if (known) el.innerHTML = hljs.highlight(el.textContent, {language: LANG_BY_EXT[lang] || lang, ignoreIllegals: true}).value;
    el.classList.add("hljs");
  }
  for (const a of box.querySelectorAll("a[href]")) {
    const href = a.getAttribute("href");
    if (href.startsWith("#")) a.dataset.anchor = href.slice(1);
    else if (/^[a-z][a-z0-9+.-]*:/i.test(href)) { a.target = "_blank"; a.rel = "noopener noreferrer"; }
    else {
      const [path, hash = ""] = href.split("#");
      a.dataset.local = path ? resolvePath(f.path, path.split("?")[0]) : f.path;
      a.dataset.anchor = hash;
      if (/[\\/]$/.test(path)) a.dataset.folder = "1";              // "src/": a folder, opened in the list
      a.title = a.dataset.local;
    }
  }
  for (const img of box.querySelectorAll("img[data-src]")) {
    const src = img.dataset.src, alt = img.getAttribute("alt") || "";
    if (/^data:image\//i.test(src)) { img.src = src; continue; }
    if (!src || /^[a-z][a-z0-9+.-]*:|^\/\//i.test(src)) {          // from the internet: not loaded, a link instead
      const a = document.createElement("a");
      a.className = "md-remote";
      a.href = src; a.target = "_blank"; a.rel = "noopener noreferrer";
      a.textContent = alt || "picture";
      a.title = `${src} (not loaded: from the internet)`;
      img.replaceWith(a);
      continue;
    }
    const path = resolvePath(f.path, src.split(/[?#]/)[0]);
    if (!/\.(png|jpe?g|gif|webp|avif|bmp|ico|svg)$/i.test(path)) {   // not a picture: no request for it
      img.replaceWith(Object.assign(document.createElement("span"), {className: "md-missing", textContent: alt || src, title: "not a picture"}));
      continue;
    }
    wsCall("fs.image", {path})
      .then((d) => { img.src = `data:${d.type};base64,${d.data}`; })
      .catch((e) => { img.replaceWith(Object.assign(document.createElement("span"), {className: "md-missing", textContent: alt || src, title: e.message})); });
  }
}
function scrollToAnchor(name) {
  const t = $("fs-text").querySelector(`[id="md-${CSS.escape(name.toLowerCase())}"]`);
  if (t) t.scrollIntoView({block: "start"});
}
$("fs-text").addEventListener("click", (e) => {
  const a = e.target.closest("a[data-local], a[data-anchor]");
  if (!a) return;
  e.preventDefault();
  if (a.dataset.folder) fsOpen(a.dataset.local);
  else if (a.dataset.local && fs.file && a.dataset.local !== fs.file.path) openPath(a.dataset.local, a.dataset.anchor);
  else if (a.dataset.anchor) scrollToAnchor(a.dataset.anchor);
});
$("fs-modes").addEventListener("click", (e) => {
  const b = e.target.closest("[data-view]");
  if (!b || !fs.file) return;
  fsView.mode = b.dataset.view;
  store.set("fs.view", fsView.mode);
  showFile(fs.file);
});
$("fs-wrap").onclick = () => {
  fsView.wrap = !fsView.wrap;
  store.set("fs.wrap", fsView.wrap);
  $("fs-wrap").setAttribute("aria-pressed", String(fsView.wrap));
  $("fs-text").classList.toggle("is-wrap", fsView.wrap);
};
$("fs-copy").onclick = async () => {
  if (!fs.file) return;
  try { await navigator.clipboard.writeText(fs.file.text); toast("success", "Copied", fs.file.name, 2000); }
  catch (e) { toast("warn", "Could not copy", "The browser allows copying only on https or on this PC (127.0.0.1)."); }
};

// ------------------------------------------------------------------ Settings: the shared folders (serve/workspace.py)
function renderRoots() {
  const list = $("roots-list"), rs = ws.root_settings, empty = (t) => `<li class="chat-list__empty muted small">${t}</li>`;
  const usable = wsState === "ready" && rs && rs.editable;
  $("roots-input").disabled = $("roots-add").disabled = !usable;
  if (wsState === "key") { list.innerHTML = empty("Add the API key above to see and change the shared folders."); return; }
  if (wsState !== "ready") { list.innerHTML = empty("The workspace of this server is not reachable."); return; }
  if (!rs) { list.innerHTML = empty("This server runs an older version than this page: restart Strata to change the folders here."); return; }
  const inside = (root) => ws.projects.filter((p) => p.folder && (p.folder === root || p.folder.startsWith(root.replace(/[\\/]$/, "") + (root.includes("\\") ? "\\" : "/")))).length;
  const row = (path, extra, ok = true) => `<li class="roots__row${ok ? "" : " is-missing"}">${svgUse("i-folder")}` +
    `<span class="roots__path" title="${esc(path)}">&lrm;${esc(path)}&lrm;</span>` +
    (inside(path) ? `<span class="muted small">${fmt(inside(path))} project${inside(path) === 1 ? "" : "s"}</span>` : "") + extra + "</li>";
  list.innerHTML = rs.fixed.map((p) => row(p, `<span class="st-badge" title="Set with --workspace-root when Strata starts: change it there">start command</span>`))
    .concat(rs.saved.map((r) => row(r.path, (r.ok ? "" : `<span class="st-badge st-badge--error" title="The folder is not there now; it is shared again once it is">missing</span>`) +
      `<button type="button" class="st-btn st-btn--icon" data-unshare="${esc(r.path)}" title="Stop sharing" aria-label="Stop sharing ${esc(r.path)}">${icon("trash")}</button>`, r.ok)))
    .join("") || empty(rs.editable ? "No folder is shared yet: the Files tab and projects with a folder stay off until you share one."
                                   : "Folders cannot be shared: other devices reach this server and it has no API key. Start it with --api-key.");
}
async function saveRoots(roots, done) {
  let r;
  try { r = await wsCall("roots.save", {roots}); } catch (e) { toast("error", "Not changed", e.message, 9000); return false; }
  ws.roots = r.roots;
  ws.root_settings = r.root_settings;
  fs.loaded = false;                                 // the Files tab starts again at the shared folders
  renderRoots();
  toast("success", done, "", 2500);
  if (r.outside.length) toast("warn", "Projects outside the shared folders", `${r.outside.join(", ")}: the agent does not work in them until their folder is shared again.`, 9000);
  return true;
}
$("roots-form").onsubmit = async (e) => {
  e.preventDefault();
  const v = $("roots-input").value.trim();
  if (!v || !ws.root_settings) return;
  if (await saveRoots([...ws.root_settings.saved.map((r) => r.path), v], "Folder shared")) $("roots-input").value = "";
};
$("roots-list").addEventListener("click", (e) => {
  const b = e.target.closest("[data-unshare]");
  if (b && ws.root_settings) saveRoots(ws.root_settings.saved.map((r) => r.path).filter((p) => p !== b.dataset.unshare), "No longer shared");
});

// the list: beside the chat on wide screens (open or closed as last chosen), over it on narrow ones
const narrowMq = matchMedia("(max-width: 1000px)");
let sideOpen = false;
function setSide(open, save) {
  sideOpen = open;
  $("chat-side").dataset.open = String(open);
  $("chats-btn").setAttribute("aria-expanded", String(open));
  $("chats-btn").hidden = open;
  $("view-chat").classList.toggle("side-shut", !open);
  $("side-scrim").hidden = !(open && narrowMq.matches);
  if (save && !narrowMq.matches) store.set("sidebar", open);
}
narrowMq.addEventListener("change", () => setSide(!narrowMq.matches && store.get("sidebar", true)));
$("chats-btn").onclick = () => setSide(!sideOpen, true);
$("side-close").onclick = () => setSide(false, true);
$("side-scrim").onclick = () => setSide(false);
$("side-new").onclick = newChat;

function timeStr(t) { return new Date(t).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}); }

function msgEl(m, i) {
  if (m.role === "compact") {                       // a slim line; the summary on demand
    const el = document.createElement("div");
    el.className = "st-msg st-msg--compact";
    el.dataset.i = i;
    el.innerHTML = m.working ? `<div class="compact-note muted small">${icon("layers", "st-icon st-icon--sm")} Compacting the chat …</div>` :
      `<details class="st-collapse compact-note"><summary>${icon("layers", "st-icon st-icon--sm")}<span>Context compacted` +
      `${m.before ? ` · ${kfmt(m.before)} tokens → the summary below` : ""}</span><span class="muted small">${esc(timeStr(m.time))}</span>` +
      `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body compact-note__body">${markdown(m.summary || "")}</div></details>`;
    return el;
  }
  const el = document.createElement("div");
  el.className = `st-msg st-msg--${m.role}`;
  el.dataset.i = i;
  if (m.role === "user") {
    if (m.files && m.files.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const f of m.files) {
        const c = document.createElement("span");
        c.className = "chip";
        c.innerHTML = icon("attach", "st-icon st-icon--sm");
        c.append(f.name);
        wrap.appendChild(c);
      }
      el.appendChild(wrap);
    }
    if (m.images && m.images.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const im of m.images) {
        if (im.url) { const img = document.createElement("img"); img.src = im.url; img.alt = im.name || "image"; wrap.appendChild(img); }
        else { const c = document.createElement("span"); c.className = "chip"; c.innerHTML = icon("image", "st-icon st-icon--sm"); c.append(im.name || "image"); wrap.appendChild(c); }
      }
      el.appendChild(wrap);
    }
    const b = document.createElement("div");
    b.className = "st-bubble";
    b.textContent = m.text;
    el.appendChild(b);
    const meta = document.createElement("div");
    meta.className = "st-msg__meta";
    meta.textContent = `You · ${timeStr(m.time)}`;
    el.appendChild(meta);
  } else {
    el.innerHTML = `<details class="st-collapse think" hidden><summary>${icon("thinking", "st-icon st-icon--sm")}<span class="think-title"></span>` +
      `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body thinking"></div></details>` +
      `<div class="st-bubble"></div><div class="st-msg__meta"><span class="meta-text"></span>` +
      `<button type="button" class="st-btn st-btn--secondary meta-continue" data-continue hidden>Continue</button>` +
      `<button class="st-btn st-btn--icon" data-msg-copy aria-label="Copy the answer" title="Copy">${icon("copy")}</button></div>`;
    updateAssistant(el, m, false);
  }
  return el;
}
// One MCP tool call in the answer: a compact block (name, state, a one-line preview) that opens to the arguments and
// the result as the model read it.  Its body is built only while open: a result can be 20,000 characters.
const TOOL_STATE = {writing: ["st-badge--reading", "Writing"], running: ["st-badge--generating", "Running"], done: ["", "Done"],
                    error: ["st-badge--error", "Error"], skipped: ["st-badge--queued", "Not run"],
                    approve: ["st-badge--reading", "Waiting for you"], declined: ["st-badge--queued", "Declined"]};
// what a change or a command will do, readable, for the approval
function toolPreview(t) {
  const a = t.arguments && typeof t.arguments === "object" ? t.arguments : {};
  if (t.name === "run_command") return `<div class="tool-call__label">Command in the project folder</div><pre class="tool-call__pre">$ ${esc(a.command || "")}</pre>`;
  if (t.name === "edit_file") {
    const lines = (x, sign) => String(x || "").split("\n").map((l) => `${sign} ${l}`).join("\n");
    return `<div class="tool-call__label">Change in ${esc(a.path || "")}${a.replace_all ? " (every occurrence)" : ""}</div>` +
           `<pre class="tool-call__pre tool-diff"><span class="del">${esc(lines(a.old_string, "-"))}</span>\n<span class="add">${esc(lines(a.new_string, "+"))}</span></pre>`;
  }
  if (t.name === "write_file") {
    const c = String(a.content || ""), shown = c.split("\n").slice(0, 80).join("\n");
    return `<div class="tool-call__label">Write ${esc(a.path || "")} (${fmt(c.length)} characters)</div>` +
           `<pre class="tool-call__pre">${esc(shown)}${shown.length < c.length ? "\n…" : ""}</pre>`;
  }
  return "";
}
function toolHtml(t, k) {
  const [cls, label] = TOOL_STATE[t.state] || ["", t.state];
  const args = t.arguments == null ? "" : typeof t.arguments === "string" ? t.arguments : JSON.stringify(t.arguments, null, 2);
  const preview = t.result != null ? t.result : t.state === "writing" ? "being written…" : args.replace(/\s+/g, " ");
  let body = "";
  if (t.state === "approve") {
    body = toolPreview(t) + `<div class="tool-approve"><button type="button" class="st-btn st-btn--primary" data-approve="yes">Allow</button>` +
      `<button type="button" class="st-btn st-btn--secondary" data-approve="all">Allow all in this answer</button>` +
      (t.name === "run_command" && suggestRule(t.arguments && t.arguments.command) ?
        `<button type="button" class="st-btn st-btn--secondary" data-approve="rule">Always allow <code>${esc(suggestRule(t.arguments.command))}</code></button>` : "") +
      `<button type="button" class="st-btn st-btn--secondary" data-approve="no">Deny</button></div>`;
  } else if (t.state === "running" && t.jobId) {
    body = toolPreview(t) + `<pre class="tool-call__pre tool-live" data-live="${esc(t.id)}">${esc(t.live || "")}</pre>` +
      `<div class="tool-approve"><button type="button" class="st-btn st-btn--secondary" data-job-stop="${esc(t.jobId)}">Stop command</button></div>`;
  } else if (t.open) {
    body = (toolPreview(t) || `<div class="tool-call__label">Arguments</div><pre class="tool-call__pre">${esc(args || "(being written)")}</pre>`);
    if (t.result != null) {
      body += `<div class="tool-call__label">${t.ok ? "Result" : "Error"}${t.chars ? ` · ${fmt(t.chars)} characters` : ""}` +
              `${t.truncated ? ", cut for the model" : ""}</div><pre class="tool-call__pre">${esc(t.result)}</pre>`;
    }
  }
  return `<details class="st-collapse tool-call" data-tool="${k}" data-id="${esc(t.id || "")}" data-state="${esc(t.state)}"${t.open || t.state === "approve" || (t.state === "running" && t.jobId) ? " open" : ""}>` +
    `<summary>${icon("tool", "st-icon st-icon--sm")}<span class="tool-call__name" title="${esc(t.name || "")}">${esc(t.tool || t.name || "tool")}</span>` +
    (t.server ? `<span class="muted small">${esc(t.server)}</span>` : "") +
    `<span class="tool-call__preview muted">${esc(preview.slice(0, 200))}</span>` +
    `<span class="st-badge ${cls}">${esc(label)}</span>${t.ms != null && t.state !== "skipped" ? `<span class="muted small">${fmt(t.ms / 1000, 1)} s</span>` : ""}` +
    `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body">${body}</div></details>`;
}
// The answer: its text and its tool steps, in order. Consecutive steps collapse into one line ("7 steps · read 4
// files · ran 2 commands"), expandable; only what needs you stays out in the open (an approval, a running command).
// It is rendered piece by piece, and a piece's DOM is only replaced when its HTML changed: while the answer streams,
// the tool blocks and their icons are left alone (rebuilding them every frame made the icons flicker).
const STEP_KIND = {read_file: "read", list_dir: "look", find_files: "look", search: "search", edit_file: "change",
                   write_file: "change", run_command: "run", job_output: "run", job_stop: "run", update_plan: "plan",
                   compact_context: "compact"};
const LIVE = ["approve", "running", "writing"];
function stepSummary(tools) {
  const c = {read: 0, look: 0, search: 0, change: 0, run: 0, plan: 0, compact: 0, other: 0};
  for (const t of tools) c[STEP_KIND[t.name] || "other"]++;
  const pl = (n, one, many) => `${n} ${n === 1 ? one : many}`;
  const parts = [];
  if (c.read) parts.push(`read ${pl(c.read, "file", "files")}`);
  if (c.look) parts.push(`looked in ${pl(c.look, "folder", "folders")}`);
  if (c.search) parts.push(pl(c.search, "search", "searches"));
  if (c.change) parts.push(`changed ${pl(c.change, "file", "files")}`);
  if (c.run) parts.push(`ran ${pl(c.run, "command", "commands")}`);
  if (c.plan) parts.push("updated the plan");
  if (c.compact) parts.push("compacted the context");
  if (c.other) parts.push(pl(c.other, "other tool", "other tools"));
  return parts.join(" · ");
}
function groupHtml(m, g) {
  const open = !!(m.groups && m.groups[g.key]);
  const live = g.items.filter(([t]) => LIVE.includes(t.state));
  const tools = g.items.map(([t]) => t).filter((t) => !LIVE.includes(t.state));   // the line counts what is done
  const errors = tools.filter((t) => t.state === "error").length, declined = tools.filter((t) => t.state === "declined").length;
  const ms = tools.reduce((a, t) => a + (t.ms || 0), 0);
  let head = "";
  if (tools.length) {
    head = `<details class="st-collapse tool-group" data-group="${esc(g.key)}"${open ? " open" : ""}><summary>` +
      `${icon("tool", "st-icon st-icon--sm")}<span class="tool-group__count">${tools.length} step${tools.length === 1 ? "" : "s"}</span>` +
      `<span class="tool-group__what muted">${esc(stepSummary(tools))}</span>` +
      (errors ? `<span class="st-badge st-badge--error">${errors} error${errors > 1 ? "s" : ""}</span>` : "") +
      (declined ? `<span class="st-badge st-badge--queued">${declined} declined</span>` : "") +
      (ms >= 100 ? `<span class="muted small">${fmt(ms / 1000, 1)} s</span>` : "") +
      `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary>` +
      `<div class="st-collapse__body tool-group__body">${open ? g.items.filter(([t]) => !LIVE.includes(t.state)).map(([t, k]) => toolHtml(t, k)).join("") : ""}</div></details>`;
  }
  return head + live.map(([t, k]) => toolHtml(t, k)).join("");
}
function planHtml(plan) {
  const mark = {done: "✓", in_progress: "◐", pending: "○"};
  const doneN = plan.filter((x) => x.status === "done").length;
  return `<div class="plan"><div class="plan__head"><strong>Plan</strong><span class="muted small">${doneN} of ${plan.length} done</span></div>` +
    `<ol class="plan__items">${plan.map((x) => `<li class="plan__item" data-status="${x.status}"><span class="plan__mark">${mark[x.status]}</span>` +
    `<span class="plan__text">${esc(x.text)}</span></li>`).join("")}</ol></div>`;
}
function answerSegments(m) {
  const text = m.text || "";
  const plan = m.plan && m.plan.length ? [{key: "plan", html: planHtml(m.plan)}] : [];
  return [...plan, ...answerParts(m, text)];
}
// More than FOLD steps: the work (the steps and the text between them) folds into one block that follows the run - the
// last steps go by under its line while it works - and the answer's last text stays below it. What needs you (an
// approval, a running command) stays out in the open. Opened, the block shows everything, grouped as before.
const FOLD = 3, TICK = 3;
function answerParts(m, text) {
  if (!m.tools || !m.tools.length) return [{key: "t0", html: markdown(text)}];
  if (m.tools.length > FOLD) return workParts(m, text);
  const segs = [];
  let pos = 0, group = null;
  m.tools.forEach((t, k) => {
    const at = Math.min(Math.max(t.at || 0, pos), text.length);
    if (at > pos && text.slice(pos, at).trim()) { segs.push({key: `t${pos}`, html: markdown(text.slice(pos, at))}); group = null; }
    pos = Math.max(pos, at);
    if (!group) { group = {key: `g${k}`, items: []}; segs.push(group); }
    group.items.push([t, k]);
  });
  if (text.slice(pos).trim()) segs.push({key: `t${pos}`, html: markdown(text.slice(pos))});
  return segs.map((x) => x.items ? {key: x.key, html: groupHtml(m, x)} : x);
}
function workParts(m, text) {
  const live = [], segs = [];
  let pos = 0, group = null;
  m.tools.forEach((t, k) => {
    const at = Math.min(Math.max(t.at || 0, pos), text.length);
    if (at > pos && text.slice(pos, at).trim()) { segs.push({text: text.slice(pos, at)}); group = null; }
    pos = Math.max(pos, at);
    if (LIVE.includes(t.state)) { live.push([t, k]); return; }
    if (!group) { group = {key: `g${k}`, items: []}; segs.push(group); }
    group.items.push([t, k]);
  });
  const running = !!busy && busy.msg === m, done = m.tools.filter((t) => !LIVE.includes(t.state));
  const errors = done.filter((t) => t.state === "error").length, open = !!m.workOpen;
  const ms = done.reduce((a, t) => a + (t.ms || 0), 0);
  const notes = segs.filter((x) => x.text);
  let tick = "";
  if (!open && running) {                            // following the run: the last steps, one line each, no icons
    const last = done.slice(-TICK);
    const note = notes.length && notes[notes.length - 1].text.trim().split("\n").pop();
    const noteAt = notes.length ? text.lastIndexOf(notes[notes.length - 1].text) : -1;   // after the steps it followed
    const row = (t) => `<div class="work__step" data-state="${esc(t.state)}"><span class="work__name">${esc(t.tool || t.name || "tool")}</span>` +
      `<span class="muted">${esc(stepLine(t))}</span></div>`;
    tick = `<div class="work__tick">` + last.filter((t) => (t.at || 0) <= noteAt).map(row).join("") +
      (note ? `<div class="work__note">${esc(note.slice(0, 200))}</div>` : "") + last.filter((t) => (t.at || 0) > noteAt).map(row).join("") + `</div>`;
  }
  const body = open ? segs.map((x) => x.text ? `<div class="work__text">${markdown(x.text)}</div>` : groupHtml(m, x)).join("") : "";
  const head = `<details class="st-collapse work" data-work${open ? " open" : ""}><summary>` +
    `<span class="work__dot${running ? " is-running" : ""}"></span>` +
    `<span class="tool-group__count">${running ? "Working" : "Worked"} · ${done.length} step${done.length === 1 ? "" : "s"}</span>` +
    `<span class="tool-group__what muted">${esc(stepSummary(done))}</span>` +
    (errors ? `<span class="st-badge st-badge--error">${errors} error${errors > 1 ? "s" : ""}</span>` : "") +
    (ms >= 100 ? `<span class="muted small">${fmt(ms / 1000, 1)} s</span>` : "") +
    `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body work__body">${body}</div></details>`;
  const out = [{key: "work", html: head + tick}];
  for (const [t, k] of live) out.push({key: `live${k}`, html: toolHtml(t, k)});
  if (text.slice(pos).trim()) out.push({key: `t${pos}`, html: markdown(text.slice(pos))});
  return out;
}
// one line for a step in the folded block: what it worked on
function stepLine(t) {
  const a = t.arguments && typeof t.arguments === "object" ? t.arguments : {};
  const what = a.path || a.command || a.pattern || a.query || a.id || "";
  const res = t.state === "error" ? `failed: ${t.result || ""}` : t.state === "declined" ? "declined" : "";
  return String(what || res || (t.result || "").split("\n")[0]).replace(/\s+/g, " ").slice(0, 160) + (what && res ? ` - ${res}` : "");
}
function renderAnswer(bubble, m) {
  const segs = answerSegments(m);
  if ([...bubble.children].some((c) => !c.dataset.seg)) bubble.replaceChildren();   // an error or a cursor was here
  segs.forEach((sg, i) => {
    let el = bubble.children[i];
    if (!el || el.dataset.seg !== sg.key) {
      const fresh = document.createElement("div");
      fresh.dataset.seg = sg.key;
      if (el) bubble.insertBefore(fresh, el); else bubble.appendChild(fresh);
      el = fresh;
    }
    if (el._html !== sg.html) { el.innerHTML = sg.html; el._html = sg.html; }
  });
  while (bubble.children.length > segs.length) bubble.lastElementChild.remove();
}
// a tool event from the stream (the `strata_mcp` field of a chunk)
function onTool(m, x) {
  if (x.event === "limit") { m.limit = x.max_rounds; return; }
  m.tools = m.tools || [];
  let t = m.tools.find((y) => y.id === x.id);
  if (!t) { t = {id: x.id, name: x.name, at: m.text.length, rat: m.reasoning.length, state: "writing"}; m.tools.push(t); }
  if (x.event === "call") {
    Object.assign(t, {name: x.name, server: x.server, tool: x.tool, arguments: x.arguments, round: x.round, state: "running"});
  } else if (x.event === "result") {
    Object.assign(t, {result: x.text, ok: x.ok, chars: x.chars, truncated: x.truncated, ms: x.ms,
                      state: x.skipped ? "skipped" : x.ok ? "done" : "error"});
  }
}
function updateAssistant(el, m, streaming) {
  const det = el.querySelector("details.think");
  if (m.reasoning) {
    det.hidden = false;
    const thinkingNow = streaming && !m.text;
    el.querySelector(".think-title").textContent = thinkingNow ? "Thinking…" :
      m.thinkSecs != null ? `Thought for ${fmt(m.thinkSecs, 1)} s` : "Thoughts";
    const body = el.querySelector(".thinking");
    if (det.open || thinkingNow) body.textContent = m.reasoning;
    else body.dataset.pending = "1";
    // open while it streams (if wanted), closed once the answer starts - unless the user toggled it themselves
    if (thinkingNow && settings.show && !det.dataset.touched && !det.open) { det._auto = true; det.open = true; }
    if (!thinkingNow && det.open && !det.dataset.touched) { det._auto = true; det.open = false; }
  }
  const bubble = el.querySelector(".st-bubble");
  if (m.error) {
    bubble.innerHTML = `<div class="msg-error"></div>`;
    bubble.firstChild.textContent = m.error;
  } else if (!m.text && streaming && !(m.tools && m.tools.length)) {
    bubble.innerHTML = m.reasoning ? `<span class="muted cursor">Writing</span>` : `<span class="cursor"></span>`;
  } else {
    renderAnswer(bubble, m);
    if (streaming) bubble.classList.add("cursor"); else bubble.classList.remove("cursor");
  }
  el.querySelector(".meta-text").textContent = m.meta || (streaming ? "" : m.stopped ? "Stopped" : "");
  el.querySelector("[data-msg-copy]").hidden = streaming || !m.text;
  el.querySelector("[data-continue]").hidden = streaming || !m.limit;
}
function renderChat() {
  const chat = $("chat");
  chat.querySelectorAll(".st-msg").forEach((e) => e.remove());
  $("chat-empty").hidden = messages.length > 0;
  messages.forEach((m, i) => chat.appendChild(msgEl(m, i)));
  scrollDown(true);
}
function nearBottom() { const s = $("chat-scroll"); return s.scrollHeight - s.scrollTop - s.clientHeight < 120; }
function scrollDown(force) { const s = $("chat-scroll"); if (force || nearBottom()) s.scrollTop = s.scrollHeight; }

$("chat").addEventListener("click", (e) => {
  const cc = e.target.closest("[data-code-copy]");
  if (cc) { copyText(cc.closest(".st-code").querySelector("pre").textContent, cc); return; }
  const js = e.target.closest("[data-job-stop]");
  if (js) { wsCall("job.stop", {id: js.dataset.jobStop}).catch((err) => toast("error", "Could not stop it", err.message)); js.disabled = true; return; }
  const ap = e.target.closest("[data-approve]");
  if (ap) {
    const id = ap.closest(".tool-call").dataset.id, a = approvals.get(id);
    if (a) { approvals.delete(id); a.resolve({no: false, all: "all", rule: "rule"}[ap.dataset.approve] ?? true); }
    return;
  }
  const cont = e.target.closest("[data-continue]");
  if (cont) { continueAnswer(+cont.closest(".st-msg").dataset.i); return; }
  const mc = e.target.closest("[data-msg-copy]");
  if (mc) { const i = +mc.closest(".st-msg").dataset.i; copyText(messages[i].text, mc); return; }
  // the folded work, and a group of steps: their open state lives in the message too
  const ws_ = e.target.closest(".work > summary");
  if (ws_) {
    e.preventDefault();
    const el = ws_.closest(".st-msg"), m = messages[+el.dataset.i];
    if (!m) return;
    m.workOpen = !m.workOpen;
    updateAssistant(el, m, !!busy && busy.msg === m);
    return;
  }
  const gs = e.target.closest(".tool-group > summary");
  if (gs) {
    e.preventDefault();
    const el = gs.closest(".st-msg"), m = messages[+el.dataset.i], key = gs.parentElement.dataset.group;
    if (!m) return;
    m.groups = m.groups || {};
    m.groups[key] = !m.groups[key];
    updateAssistant(el, m, !!busy && busy.msg === m);
    return;
  }
  // a tool block: its open state lives in the message (the answer is rebuilt while it streams), so the click sets it
  const sum = e.target.closest(".tool-call > summary");
  if (sum) {
    e.preventDefault();
    const el = sum.closest(".st-msg"), m = messages[+el.dataset.i], t = m && m.tools && m.tools[+sum.parentElement.dataset.tool];
    if (!t) return;
    t.open = !t.open;
    updateAssistant(el, m, !!busy && busy.msg === m);
  }
});
$("chat").addEventListener("toggle", (e) => {
  const d = e.target;
  if (d.tagName !== "DETAILS" || !d.classList.contains("think")) return;
  if (d._auto) { d._auto = false; return; }          // our own open/close, not the user's
  d.dataset.touched = "1";
  const body = d.querySelector(".thinking");
  if (d.open && body.dataset.pending) { body.textContent = messages[+d.closest(".st-msg").dataset.i].reasoning; delete body.dataset.pending; }
}, true);

// The history the model gets. A compaction stands in for what came before it: a chat-level marker (role "compact",
// the Compact button) replaces every earlier message; one inside an answer (m.compact, during a long agent session)
// replaces the earlier messages and the answer's rounds up to it - the task itself (the last question) is kept word
// for word. The chat on the page keeps everything.
const compactLead = (summary) => `Summary of the work so far (the earlier messages and steps were compacted to save context):\n\n${summary}`;
function apiMessages() {
  let start = 0, lead = null;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "compact" && !messages[i].working) { start = i + 1; lead = compactLead(messages[i].summary); break; }
  }
  let out = [];
  const userMsg = (m, prefix) => {
    const imgs = (m.images || []).filter((i) => i.url);
    const text = prefix ? `${prefix}\n\n---\n\n${userText(m)}` : userText(m);
    return {role: "user", content: imgs.length ? [{type: "text", text}, ...imgs.map((i) => ({type: "image_url", image_url: {url: i.url}}))] : text};
  };
  let task = null;
  for (const m of messages.slice(start)) {
    if (m.role === "user") {
      out.push(userMsg(m, lead));
      lead = null;
      task = m;
    } else if (m.role === "assistant" && !m.error) {
      if (m.compact && task) {
        out = [userMsg(task, null)];
        out[0] = typeof out[0].content === "string" ? {role: "user", content: `${out[0].content}\n\n---\n\n${compactLead(m.compact.summary)}`}
                                                   : {role: "user", content: [...out[0].content.slice(0, 1).map((x) => ({...x, text: `${x.text}\n\n---\n\n${compactLead(m.compact.summary)}`})), ...out[0].content.slice(1)]};
        out.push(...assistantMessages(m, m.compact.round));
      } else {
        out.push(...assistantMessages(m));
      }
    }
  }
  if (lead) out.push({role: "user", content: lead});   // compacted, nothing asked since
  return out;
}
// An answer that used tools goes back as the model wrote it: per round the text before the calls, the calls and their
// results (as the model read them), then the rest - so the next question can build on what the tools found. Rounds up
// to `after` are left out (a compaction summarized them).
function assistantMessages(m, after = -1) {
  const tools = (m.tools || []).filter((t) => !t.compact);
  const ran = tools.filter((t) => t.round != null && t.round > after && t.result != null && t.state !== "skipped");
  let pos = 0;
  if (after >= 0) for (const t of tools) if (t.round != null && t.round <= after) pos = Math.max(pos, t.at || 0);
  if (!ran.length) { const rest = m.text.slice(pos).trim(); return rest ? [{role: "assistant", content: rest}] : []; }
  const out = [];
  for (const r of [...new Set(ran.map((t) => t.round))]) {
    const calls = ran.filter((t) => t.round === r);
    const at = Math.min(Math.max(pos, calls[0].at || 0), m.text.length);
    out.push({role: "assistant", content: m.text.slice(pos, at).trim(),
              tool_calls: calls.map((t) => ({id: t.id, type: "function", function: {name: t.name, arguments: JSON.stringify(t.arguments || {})}}))});
    for (const t of calls) out.push({role: "tool", tool_call_id: t.id, content: t.result});
    pos = at;
  }
  const rest = m.text.slice(pos).trim();
  if (rest) out.push({role: "assistant", content: rest});
  return out;
}

// ------------------------------------------------------------------ compaction
const COMPACT_PROMPT = "Compact the conversation so far into a summary that lets you continue the work seamlessly " +
  "without the earlier messages. Do NOT call any tool now; answer with the summary only, in this order:\n" +
  "1. The user's goal and constraints (keep their exact requirements).\n" +
  "2. What has been done: files read and changed (exact paths), commands run and their outcome, key findings and decisions.\n" +
  "3. The current state, including errors still open (exact messages) and anything running in the background (job ids).\n" +
  "4. What remains, as a short to-do list, and the next step.\n" +
  "Keep exact file paths, function and variable names, numbers and error messages. Be concise: at most about 1,500 words.";
// one summary request. It starts with exactly the session's own tokens (system prompt, tools, history), so the server
// reuses what it holds and only reads the request at the end; if the model calls a tool anyway, once more without tools
async function summarizeContext(system, tools, signal) {
  const ask = async (withTools) => {
    const body = {model: health.model, stream: false, max_tokens: 3000, temperature: 0.3, reasoning_effort: "low",
                  messages: [...(system ? [{role: "system", content: system}] : []), ...apiMessages(), {role: "user", content: COMPACT_PROMPT}]};
    if (withTools && tools) body.tools = tools;
    const r = await fetch("v1/chat/completions", {method: "POST", headers: headers(true), body: JSON.stringify(body), signal});
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const j = await r.json();
    const msg = j.choices && j.choices[0] && j.choices[0].message;
    return {text: msg && typeof msg.content === "string" ? msg.content.trim() : "", prompt: j.usage ? j.usage.prompt_tokens : null};
  };
  let got = await ask(true);
  if (!got.text) got = await ask(false);
  if (!got.text) throw new Error("the model wrote no summary");
  return got;
}
const compactAt = (p) => { const v = p ? (projectOf(p.id) || p).compact_at : null; return v === 0 ? Infinity : v || 100000; };
// during a long answer: compact before the next round once the prompt passed the project's threshold
async function compactInAnswer(m, round, el, system, tools, before, signal) {
  const t = {id: `compact_${round}_${Date.now()}`, name: "compact_context", tool: "compact_context", compact: true, state: "running",
             at: m.text.length, rat: m.reasoning.length, round, server: "context", arguments: {tokens: before}};
  m.tools = m.tools || [];
  m.tools.push(t);
  updateAssistant(el, m, true);
  const t0 = performance.now();
  try {
    const {text} = await summarizeContext(system, tools, signal);
    m.compact = {round, summary: text, before, time: Date.now()};
    Object.assign(t, {state: "done", ok: true, result: text, chars: text.length, ms: performance.now() - t0});
  } catch (e) {
    if (signal.aborted) throw e;
    Object.assign(t, {state: "error", ok: false, result: `not compacted: ${e.message}`, ms: performance.now() - t0});
    toast("warn", "The context was not compacted", e.message, 6000);
  }
  updateAssistant(el, m, true);
}
// the Compact button: the whole chat so far into one summary (a marker in the chat)
async function compactChat() {
  if (busy) { toast("warn", "Still writing", "Stop the answer first."); return; }
  if (!messages.some((m) => m.role === "assistant" && !m.error)) { toast("info", "Nothing to compact yet"); return; }
  const controller = new AbortController();
  busy = {controller, msg: null};
  setBusy(true);
  const marker = {role: "compact", summary: "", time: Date.now(), working: true};
  messages.push(marker);
  renderChat();
  try {
    const proj = projectOf(chatProject);
    const system = await projectSystem(chatProject).catch(() => "");
    const {text, prompt} = await summarizeContext(system, proj && proj.folder ? agentTools(proj.mode) : null, controller.signal);
    Object.assign(marker, {summary: text, before: prompt, working: false});
    toast("success", "Chat compacted", "The model continues from the summary; the chat above stays as it is.", 4000);
  } catch (e) {
    messages.splice(messages.indexOf(marker), 1);
    if (e.name !== "AbortError") toast("error", "Could not compact the chat", e.message, 6000);
  }
  busy = null;
  setBusy(false);
  renderChat();
  saveChat();
}
$("compact-btn").onclick = compactChat;

function setBusy(on) {
  $("stop-btn").hidden = !on;
  $("send-btn").disabled = on;
  $("composer-hint").textContent = on ? "" : "Shift+Enter: new line";
}

// ------------------------------------------------------------------ the coding agent (a project with a folder)
// The page runs the loop: the model asks for a tool, the page asks you when the project's mode says so, the server
// runs it in the project folder (POST /workspace/tool.run), and the result goes back to the model.
const AGENT_READ = ["list_dir", "read_file", "search", "find_files"];
const AGENT_TOOLS = [
  {name: "list_dir", description: "List a folder of the project (default: its top folder).",
   parameters: {type: "object", properties: {path: {type: "string", description: "a folder, relative to the project folder"}}}},
  {name: "read_file", description: "Read a text file, with line numbers. For a long file use offset and limit (lines).",
   parameters: {type: "object", properties: {path: {type: "string"}, offset: {type: "integer"}, limit: {type: "integer"}}, required: ["path"]}},
  {name: "search", description: "Search the files' contents for a regular expression. Returns path:line:text.",
   parameters: {type: "object", properties: {pattern: {type: "string"}, path: {type: "string", description: "a folder or file to search in"},
                glob: {type: "string", description: "only files like this, e.g. *.rs"}}, required: ["pattern"]}},
  {name: "find_files", description: "Find files and folders by a glob pattern, e.g. **/*.rs or src/**/mod.rs (skips .git, node_modules, target, ...).",
   parameters: {type: "object", properties: {pattern: {type: "string"}}, required: ["pattern"]}},
  {name: "write_file", description: "Create a file, or replace a file's whole content.",
   parameters: {type: "object", properties: {path: {type: "string"}, content: {type: "string"}}, required: ["path", "content"]}},
  {name: "edit_file", description: "Replace an exact piece of text in a file. Read the file first: old_string must match exactly (spaces and line breaks too) and only once, unless replace_all is true.",
   parameters: {type: "object", properties: {path: {type: "string"}, old_string: {type: "string"}, new_string: {type: "string"},
                replace_all: {type: "boolean"}}, required: ["path", "old_string", "new_string"]}},
  {name: "run_command", description: "Run a shell command (bash) in the project folder, e.g. tests, a build, git. Returns its output and exit code. No input can be typed into it. For something that keeps running (a dev server, a watcher, a long build) set background: true, go on with other work, and read its output later with job_output.",
   parameters: {type: "object", properties: {command: {type: "string"}, timeout_s: {type: "integer", description: "default 120 (3600 in the background)"},
                background: {type: "boolean", description: "start it and return at once with a job id"}}, required: ["command"]}},
  {name: "job_output", description: "The new output of a command started in the background, and whether it still runs. wait_s waits up to that many seconds for it to finish first.",
   parameters: {type: "object", properties: {job_id: {type: "string"}, wait_s: {type: "number"}}, required: ["job_id"]}},
  {name: "job_stop", description: "Stop a command started in the background (and everything it started).",
   parameters: {type: "object", properties: {job_id: {type: "string"}}, required: ["job_id"]}},
];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// a tool the page answers itself: the model's visible plan (never asks, also in Read only)
const PLAN_TOOL = {name: "update_plan", description: "Keep a short, visible plan for a task with several steps: send the whole list each time. Exactly one item is in_progress while you work; mark items done as you finish them.",
  parameters: {type: "object", properties: {items: {type: "array", items: {type: "object", properties: {
    text: {type: "string"}, status: {type: "string", enum: ["pending", "in_progress", "done"]}}, required: ["text", "status"]}}},
    required: ["items"]}};
const MODES = {read: "Read only", ask: "Ask before changes", edit: "Auto-edit, ask before commands", auto: "Full auto"};
const agentTools = (mode) => [...AGENT_TOOLS.filter((t) => mode !== "read" || AGENT_READ.includes(t.name)), PLAN_TOOL]
                                        .map((f) => ({type: "function", function: f}));
// allow rules: command prefixes that run without asking, per project. Only for a plain command: anything that
// chains, pipes, redirects or substitutes (; & | < > ` $( or a line break) always asks.
const SHELL_META = /[;&|<>`\n]|\$\(/;
function allowMatch(cmd, rules) {
  const c = String(cmd || "").trim();
  return !!c && !SHELL_META.test(c) && (rules || []).some((r) => c === r || c.startsWith(r + " "));
}
function suggestRule(cmd) {
  const c = String(cmd || "").trim();
  if (!c || SHELL_META.test(c)) return null;
  return c.split(/\s+/).slice(0, 2).join(" ");
}
function needsApproval(mode, name, m, args, rules) {
  if (AGENT_READ.includes(name) || ["job_output", "job_stop", "update_plan"].includes(name) || m.allowAll) return false;
  if (name === "run_command") return mode !== "auto" && !(mode !== "read" && allowMatch(args && args.command, rules));
  return mode === "ask";
}
const approvals = new Map();                     // tool id -> {resolve(true | false | "all"), name, m}
function askApproval(id, name, m, args, signal) {
  return new Promise((resolve) => {
    approvals.set(id, {resolve, name, m, args});
    signal.addEventListener("abort", () => { if (approvals.delete(id)) resolve(false); }, {once: true});
  });
}
// the project's mode right now (it can be switched while an answer runs)
const liveMode = (p) => (projectOf(p.id) || p).mode || "ask";
function finishTool(t, ok, text, ms) {
  Object.assign(t, {ok, result: text, chars: text.length, ms: ms == null ? null : ms, open: false,
                    state: t.declined ? "declined" : ok ? "done" : "error"});
}
async function runAgentCalls(p, m, calls, round, el, signal) {
  for (const c of calls) {
    const t = m.tools.find((x) => x.id === c.id);
    let args = null;
    try { args = JSON.parse(c.args || "{}"); } catch (e) { /* the model wrote broken JSON */ }
    Object.assign(t, {arguments: args, round, state: "running", server: "project"});
    if (!args || typeof args !== "object") { finishTool(t, false, "error: the arguments are not valid JSON"); continue; }
    if (c.name === "update_plan") {
      const items = (Array.isArray(args.items) ? args.items : []).filter((x) => x && typeof x.text === "string")
        .slice(0, 30).map((x) => ({text: x.text.slice(0, 300), status: ["pending", "in_progress", "done"].includes(x.status) ? x.status : "pending"}));
      m.plan = items;
      const doneN = items.filter((x) => x.status === "done").length;
      finishTool(t, items.length > 0, items.length ? `plan updated: ${doneN} of ${items.length} done` : "error: items is empty", 0);
      updateAssistant(el, m, true);
      continue;
    }
    const rules = () => (projectOf(p.id) || p).allow || [];
    if (needsApproval(liveMode(p), c.name, m, args, rules())) {
      Object.assign(t, {state: "approve", open: true});
      updateAssistant(el, m, true);
      scrollDown(true);
      const answer = await askApproval(t.id, c.name, m, args, signal);
      if (answer === "all") m.allowAll = true;
      if (answer === "rule") {                    // "Always allow ..." for this project
        const rule = suggestRule(args.command);
        try {
          const {project} = await wsCall("project.save", {id: p.id, allow: [...rules(), rule]});
          updateProject(project);
          toast("success", "Always allowed", `${rule} … runs without asking in ${project.name}.`, 3500);
        } catch (e) { toast("error", "The rule was not saved", e.message); }
      }
      if (!answer) {
        t.declined = true;
        finishTool(t, false, signal.aborted ? "Stopped by the user." : "The user declined this action. Do not try it again unless they ask for it; say what you wanted to do, or suggest another way.");
        updateAssistant(el, m, true);
        if (signal.aborted) return;
        continue;
      }
      Object.assign(t, {state: "running", open: false});
    }
    const t0 = performance.now();
    let following = false;
    if (c.name === "run_command" && !args.background) {   // its output, live, while it runs
      Object.assign(t, {jobId: "job" + newId(), live: ""});
      following = true;
      const onAbort = () => wsCall("job.stop", {id: t.jobId}).catch(() => {});
      signal.addEventListener("abort", onAbort, {once: true});
      (async () => {
        let offset = 0;
        while (following) {
          await sleep(700);
          if (!following) break;
          try {
            const o = await wsCall("job.output", {id: t.jobId, since: offset});
            offset = o.offset;
            if (o.text) {
              t.live = (t.live + o.text).slice(-20000);
              const pre = el.querySelector(`[data-live="${CSS.escape(t.id)}"]`);
              if (pre) { pre.textContent = t.live; pre.scrollTop = pre.scrollHeight; }
            }
          } catch (e) { /* not started yet */ }
        }
        signal.removeEventListener("abort", onAbort);
      })();
    }
    updateAssistant(el, m, true);
    try {
      const r = await wsCall("tool.run", {project: p.id, name: c.name, arguments: args, job: t.jobId, chat: chatId});
      finishTool(t, r.ok, r.text, performance.now() - t0);
      if (r.ok && (c.name === "write_file" || c.name === "edit_file")) refreshChanges();
    } catch (e) {
      finishTool(t, false, `error: ${e.message}`, performance.now() - t0);
    } finally {
      following = false;
    }
    updateAssistant(el, m, true);
    scrollDown();
    if (signal.aborted) return;
  }
}

async function send() {
  const text = $("input").value.trim();
  if ((!text && !attachments.length) || busy) return;
  if (engine && engine.held && engine.state !== "running" && engine.state !== "starting") {   // stopped on purpose
    toast("warn", "The engine is stopped", "Start it, then send again.", 8000, {label: "Start engine", run: () => engineDo("start", true)});
    return;
  }
  messages.push({role: "user", text, images: attachments.filter((a) => a.kind !== "file"),
                 files: attachments.filter((a) => a.kind === "file"), time: Date.now()});
  attachments = [];
  renderAttachments();
  $("input").value = "";
  autosize();
  const m = {role: "assistant", text: "", reasoning: "", time: Date.now()};
  messages.push(m);
  renderChat();
  await runAnswer(m, $("chat").lastElementChild);
}
// an answer that stopped at the project's round limit goes on where it stopped (no new message in the chat)
async function continueAnswer(i) {
  const m = messages[i];
  if (!m || busy || !m.limit) return;
  m.limit = null;
  await runAnswer(m, $("chat").querySelector(`.st-msg[data-i="${i}"]`));
}
// the rounds an answer may take before it asks to continue: the project's setting (0 = no limit), default 200
const maxRounds = (p) => { const v = (projectOf(p.id) || p).max_rounds; return v === 0 ? Infinity : v || 200; };

async function runAnswer(m, el) {
  const controller = new AbortController();
  busy = {controller, msg: m};
  setBusy(true);
  updateAssistant(el, m, true);

  let system = "";                                  // a chat in a project: its instructions and files first
  try { system = await projectSystem(chatProject); } catch (e) { toast("warn", "The project's instructions and files were not added", e.message, 6000); }
  const proj = projectOf(chatProject);
  const agent = proj && proj.folder ? proj : null;  // the coding tools work in its folder
  const makeBody = () => {
    const body = {model: health.model, messages: system ? [{role: "system", content: system}, ...apiMessages()] : apiMessages(),
                  stream: true, reasoning_effort: settings.thinking};
    if (settings.temperature > 0) {
      Object.assign(body, {temperature: +settings.temperature, top_p: +settings.top_p, top_k: +settings.top_k});
    } else {
      body.temperature = 0;
    }
    if (settings.seed) body.seed = +settings.seed;
    if (settings.max) body.max_tokens = +settings.max;
    if (projectionLoaded()) body.experimental_speed_projection = !!settings.esp;
    if (agent) body.tools = agentTools(agent.mode);
    else if (settings.mcp !== false && mcpInfo.tools > 0) body.strata_mcp = true;   // this server may run MCP tools for it
    return body;
  };

  // the numbers across every round of this answer (also across Continue): tokens written, the time spent writing
  // them (the speed), and the answer's whole time (commands, approvals and prompt reading included)
  const st = m.stats = m.stats || {tokens: 0, writeMs: 0, wallMs: 0, rounds: 0};
  const began = performance.now();
  let thinkStart = null, frame = 0;
  const paint = () => { frame = 0; updateAssistant(el, m, true); scrollDown(); };
  // one request: streams text and thinking into m, returns the tool calls the model asked for
  const round = async (n) => {
    const r = await fetch("v1/chat/completions", {method: "POST", headers: headers(true), body: JSON.stringify(makeBody()),
                                                   signal: controller.signal});
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
      if (r.status === 401) msg = "This server needs an API key: add it under Settings.";
      throw new Error(msg);
    }
    const reader = r.body.getReader(), dec = new TextDecoder(), calls = [];
    let buf = "", finish = null, firstAt = null;
    try {
      for (;;) {
        const {value, done} = await reader.read();
        if (done) break;
        buf += dec.decode(value, {stream: true});
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line.startsWith("data:")) continue;              // ": keep-alive" comments while a long prompt is read
          const data = line.slice(5).trim();
          if (data === "[DONE]") continue;
          let j;
          try { j = JSON.parse(data); } catch (e) { continue; }
          if (j.error) throw new Error(j.error.message || "the engine reported an error");
          if (j.usage) {
            st.tokens += j.usage.completion_tokens || 0;
            if (j.usage.prompt_tokens) {
              st.lastPrompt = j.usage.prompt_tokens;
              if (st.justCompacted) { st.base = j.usage.prompt_tokens; st.justCompacted = false; }   // the size right after
            }
          }
          if (j.strata_mcp) onTool(m, j.strata_mcp);
          const ch = (j.choices && j.choices[0]) || {};
          if (ch.finish_reason) finish = ch.finish_reason;
          const d = ch.delta || {};
          if ((d.reasoning_content || d.content || d.tool_calls) && !firstAt) firstAt = performance.now();
          const lastTool = m.tools && m.tools.length ? m.tools[m.tools.length - 1] : null;   // a new round after a tool
          if (d.reasoning_content) {
            if (!thinkStart) thinkStart = performance.now();
            if (lastTool && m.reasoning && lastTool.rat === m.reasoning.length) m.reasoning += "\n\n";
            m.reasoning += d.reasoning_content;
          }
          if (d.content) {
            if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
            if (lastTool && m.text && lastTool.at === m.text.length) m.text += "\n\n";
            m.text += d.content;
          }
          for (const tc of d.tool_calls || []) {               // pieces of a call, merged by index
            let c = calls[tc.index];
            if (!c) {
              c = calls[tc.index] = {id: tc.id || `call_${n}_${tc.index}`, name: "", args: ""};
              m.tools = m.tools || [];
              m.tools.push({id: c.id, name: "", tool: "", at: m.text.length, rat: m.reasoning.length, state: "writing", server: "project"});
            }
            const t = m.tools.find((x) => x.id === c.id);
            if (tc.function && tc.function.name) { c.name = tc.function.name; t.name = t.tool = c.name; }
            if (tc.function && tc.function.arguments) { c.args += tc.function.arguments; t.arguments = c.args; }
          }
          if (!frame) frame = requestAnimationFrame(paint);
        }
      }
    } finally {
      if (firstAt) st.writeMs += performance.now() - firstAt;   // only while the model was writing
    }
    return {calls: calls.filter(Boolean), finish};
  };
  try {
    let inThisRun = 0;
    for (;;) {
      const n = st.rounds++;                       // round numbers go on across Continue (the history needs them unique)
      const {calls, finish} = await round(n);
      if (!agent || finish !== "tool_calls" || !calls.length) break;
      if (!chatId) saveChat();                     // the chat's id, before a tool changes a file in its name
      await runAgentCalls(agent, m, calls, n, el, controller.signal);
      saveChat();                                  // after every round: in the list, and kept across a reload
      if (controller.signal.aborted) { m.stopped = true; break; }
      // a long session: summarize before the next round - once it passed the threshold, and grew by half the
      // threshold since the last compaction (a prompt that stays big after one must not compact every round)
      const limit = compactAt(agent);
      if (st.lastPrompt > limit && st.lastPrompt - (st.base || 0) > limit / 2) {
        await compactInAnswer(m, n, el, system, agentTools(liveMode(agent)), st.lastPrompt, controller.signal);
        st.justCompacted = true;
        saveChat();
      }
      if (++inThisRun >= maxRounds(agent)) { m.limit = inThisRun; break; }
    }
  } catch (e) {
    if (e.name === "AbortError") m.stopped = true;
    else { m.error = e.message || String(e); toast("error", "The request failed", m.error, 6000); }
  }
  st.wallMs += performance.now() - began;
  if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
  const parts = [];
  if (st.tokens) parts.push(`${fmt(st.tokens)} tokens` + (st.writeMs > 250 ? ` · ${fmt(st.tokens / (st.writeMs / 1000), 1)} tok/s` : ""));
  const ran = (m.tools || []).filter((t) => !t.compact && (t.state === "done" || t.state === "error")).length;
  for (const t of m.tools || []) if (["writing", "running", "approve"].includes(t.state)) { t.state = "skipped"; t.ms = null; t.open = false; }
  if (ran) parts.push(`${ran} tool call${ran > 1 ? "s" : ""}`);
  if (ran && st.wallMs >= 60000) parts.push(st.wallMs >= 3600000 ? `${fmt(st.wallMs / 3600000, 1)} h` : `${fmt(st.wallMs / 60000)} min`);
  if (m.stopped) parts.push("stopped");
  if (projectionLoaded()) parts.push(settings.esp ? "projection on" : "projection off");
  if (m.limit) parts.push(`paused after ${m.limit} tool rounds (the project's limit)`);
  m.meta = parts.join(" · ") || (m.stopped ? "Stopped" : "");
  delete m.allowAll;
  busy = null;
  setBusy(false);
  if (frame) cancelAnimationFrame(frame);
  updateAssistant(el, m, false);
  saveChat();
  scrollDown();
}

$("composer").onsubmit = (e) => { e.preventDefault(); send(); };
$("stop-btn").onclick = () => { if (busy) busy.controller.abort(); };
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});
function autosize() { const t = $("input"); t.style.height = "auto"; t.style.height = `${Math.min(t.scrollHeight, innerHeight * 0.4)}px`; }
$("input").addEventListener("input", autosize);

$("new-btn").onclick = newChat;            // the last chat stays in the list
$("export-btn").onclick = () => {
  if (!messages.length) { toast("info", "Nothing to save yet"); return; }
  const tools = (m) => (m.tools || []).filter((t) => t.result != null).map((t) =>
    `<details><summary>Tool ${t.server ? `${t.server} / ` : ""}${t.tool || t.name}${t.ok ? "" : " (error)"}</summary>\n\n` +
    `\`\`\`json\n${JSON.stringify(t.arguments || {}, null, 2)}\n\`\`\`\n\n\`\`\`\n${t.result}\n\`\`\`\n\n</details>\n\n`).join("");
  const md = messages.map((m) => m.role === "compact" ? `## Compacted\n\n${m.summary || ""}\n` : m.role === "user" ? `## You\n\n${m.text}\n` :
    `## ${health.model}\n\n${m.reasoning ? `<details><summary>Thinking</summary>\n\n${m.reasoning}\n\n</details>\n\n` : ""}${tools(m)}${m.text || m.error || ""}\n`).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([md], {type: "text/markdown"}));
  a.download = `strata-chat-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};

// pictures and text files: the attach button, dropping them on the chat, or pasting a picture (issue #30)
const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql)$/i;
const MAX_TEXT_FILE = 512 * 1024;
function isTextFile(f) {
  return f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) ||
         TEXT_EXT.test(f.name) || /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name);
}
function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("image/")) {
      if (!health.images) { toast("warn", "Pictures are off", "This model was set up for text only."); continue; }
      if (f.size > 20e6) { toast("warn", "Picture too large", `${f.name} is over 20 MB.`); continue; }
      const r = new FileReader();
      r.onload = () => { attachments.push({kind: "image", name: f.name || "pasted image", url: r.result}); renderAttachments(); };
      r.readAsDataURL(f);
      continue;
    }
    if (!isTextFile(f)) { toast("warn", "Not a text file", `${f.name}: attach text files (code, notes, logs, data)${health.images ? " or pictures" : ""}.`); continue; }
    if (f.size > MAX_TEXT_FILE) { toast("warn", "File too large", `${f.name} is over 512 KB.`); continue; }
    const r = new FileReader();
    r.onload = () => {
      const text = String(r.result);
      if (text.includes("\u0000")) { toast("warn", "Not a text file", `${f.name} looks like a binary file.`); return; }
      attachments.push({kind: "file", name: f.name, text});
      renderAttachments();
    };
    r.readAsText(f);
  }
}
// a file's text in the message, fenced with more backticks than it contains itself
function fileBlock(f) {
  const longest = Math.max(2, ...(f.text.match(/`+/g) || []).map((s) => s.length));
  const fence = "`".repeat(longest + 1);
  return `File: ${f.name}\n${fence}\n${f.text}\n${fence}`;
}
function userText(m) {
  const files = (m.files || []).filter((f) => f.text != null);
  return [m.text, ...files.map(fileBlock)].filter((s) => s).join("\n\n");
}
function renderAttachments() {
  const box = $("attachments");
  box.hidden = !attachments.length;
  box.innerHTML = "";
  attachments.forEach((a, i) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.innerHTML = icon(a.kind === "file" ? "attach" : "image", "st-icon st-icon--sm");
    c.append(a.name);
    const x = document.createElement("button");
    x.type = "button"; x.className = "st-btn st-btn--icon"; x.setAttribute("aria-label", "Remove");
    x.innerHTML = icon("trash");
    x.onclick = () => { attachments.splice(i, 1); renderAttachments(); };
    c.appendChild(x);
    box.appendChild(c);
  });
}
$("attach-btn").onclick = () => $("file").click();
// drop files on the chat or the message box
for (const id of ["chat", "composer"]) {
  const el = $(id);
  el.addEventListener("dragover", (e) => {
    if (![...(e.dataTransfer || {}).types || []].includes("Files")) return;
    e.preventDefault();
    $("composer").classList.add("dragging");
  });
  el.addEventListener("dragleave", () => $("composer").classList.remove("dragging"));
  el.addEventListener("drop", (e) => {
    $("composer").classList.remove("dragging");
    if (!e.dataTransfer || !e.dataTransfer.files.length) return;
    e.preventDefault();
    addFiles(e.dataTransfer.files);
    $("input").focus();
  });
}
$("file").onchange = () => { addFiles($("file").files); $("file").value = ""; };
$("input").addEventListener("paste", (e) => {
  if (!health.images) return;
  const files = [...(e.clipboardData || {}).files || []].filter((f) => f.type.startsWith("image/"));
  if (files.length) { e.preventDefault(); addFiles(files); }
});

// ------------------------------------------------------------------ the sampling drawer
function openDrawer(open) {
  $("drawer").dataset.open = String(open);
  $("drawer").setAttribute("aria-hidden", String(!open));
  $("scrim").hidden = !open;
  if (open) { loadDrawer(); loadShared(); loadMcp(); }
}
function loadDrawer(s = settings) {
  for (const b of $("s-thinking").children) b.setAttribute("aria-checked", String(b.dataset.v === s.thinking));
  $("s-temp").value = s.temperature; $("s-topp").value = s.top_p; $("s-topk").value = s.top_k;
  $("s-max").value = s.max; $("s-seed").value = s.seed;
  $("s-show").setAttribute("aria-checked", String(!!s.show));
  $("s-esp").setAttribute("aria-checked", String(s.esp !== false));
  $("esp-row").hidden = !projectionLoaded();
  $("s-mcp").setAttribute("aria-checked", String(s.mcp !== false));
  $("s-share").setAttribute("aria-checked", String(sharedOn));
  outputs();
}
// "Use for other apps too": the server keeps these settings as every client's defaults (GET/POST /settings)
let sharedOn = false;
async function loadShared() {
  try {
    const r = await fetch("settings", {headers: headers()});
    if (r.ok) sharedOn = !!(await r.json()).shared;
  } catch (e) { /* an older server: the switch just stays off */ }
  $("s-share").setAttribute("aria-checked", String(sharedOn));
}
function sharedDefaults(s) {
  const d = {reasoning_effort: s.thinking, temperature: +s.temperature};
  if (+s.temperature > 0) Object.assign(d, {top_p: +s.top_p, top_k: +s.top_k});
  if (s.seed) d.seed = +s.seed;
  if (s.max) d.max_tokens = +s.max;
  if (projectionLoaded()) d.experimental_speed_projection = s.esp !== false;
  return d;
}
async function saveShared(on, s) {
  const r = await fetch("settings", {method: "POST", headers: headers(true),
                                      body: JSON.stringify({defaults: on ? sharedDefaults(s) : null})});
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  sharedOn = !!(await r.json()).shared;
}
// the engine was started with the experimental-speed-projection control vector (INFO cvec=...)
function projectionLoaded() {
  const c = lastMetrics && lastMetrics.engine ? lastMetrics.engine.cvec : 0;
  return !!c && c !== "0";
}
function outputs() {
  const t = +$("s-temp").value;
  $("o-temp").textContent = t === 0 ? "0 · greedy" : t.toFixed(2);
  $("o-topp").textContent = (+$("s-topp").value).toFixed(2);
  $("o-topk").textContent = $("s-topk").value;
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  $("o-thinking").textContent = sel ? {none: "answers right away", low: "short", medium: "medium", high: "thorough (default)"}[sel.dataset.v] : "";
  for (const id of ["s-topp", "s-topk"]) $(id).disabled = t === 0;
}
for (const b of $("s-thinking").children) b.onclick = () => { for (const x of $("s-thinking").children) x.setAttribute("aria-checked", String(x === b)); outputs(); };
for (const id of ["s-temp", "s-topp", "s-topk"]) $(id).oninput = outputs;
$("s-show").onclick = () => $("s-show").setAttribute("aria-checked", String($("s-show").getAttribute("aria-checked") !== "true"));
$("s-esp").onclick = () => $("s-esp").setAttribute("aria-checked", String($("s-esp").getAttribute("aria-checked") !== "true"));
$("s-mcp").onclick = () => $("s-mcp").setAttribute("aria-checked", String($("s-mcp").getAttribute("aria-checked") !== "true"));
$("s-share").onclick = () => $("s-share").setAttribute("aria-checked", String($("s-share").getAttribute("aria-checked") !== "true"));
$("s-reset").onclick = () => loadDrawer(DEFAULTS);
$("s-apply").onclick = async () => {
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  settings = {thinking: sel ? sel.dataset.v : "high", temperature: +$("s-temp").value, top_p: +$("s-topp").value,
              top_k: +$("s-topk").value, max: $("s-max").value.trim(), seed: $("s-seed").value.trim(),
              show: $("s-show").getAttribute("aria-checked") === "true",
              esp: $("s-esp").getAttribute("aria-checked") === "true",
              mcp: $("s-mcp").getAttribute("aria-checked") === "true"};
  store.set("sampling", settings);
  const share = $("s-share").getAttribute("aria-checked") === "true";
  openDrawer(false);
  if (share || sharedOn) {
    try {
      await saveShared(share, settings);
      toast("success", "Sampling saved", share ? "Other apps (omp, API clients) use these settings from their next request."
                                               : "Other apps use their own settings again.");
    } catch (e) {
      toast("error", "Saved here, but not for other apps", e.message, 6000);
    }
    return;
  }
  toast("success", "Sampling saved", settings.temperature === 0 ? "Greedy: the same question gives the same answer." : "");
};
$("sampling-btn").onclick = () => openDrawer(true);
$("drawer-close").onclick = () => openDrawer(false);
$("scrim").onclick = () => openDrawer(false);
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if ($("drawer").dataset.open === "true") openDrawer(false);
  else if ($("proj-drawer").dataset.open === "true") closeProjectDrawer();
  else if ($("changes-drawer").dataset.open === "true") closeChanges();
  else if (sideOpen && narrowMq.matches) setSide(false);
});

// ------------------------------------------------------------------ start
setBusy(false);
setSide(!narrowMq.matches && store.get("sidebar", true));
renderChat();
renderSide();
const startQuestion = new URLSearchParams(location.search).get("q");   // /?q=... starts a chat (a shortcut)
if (startQuestion) history.replaceState(null, "", location.pathname + location.hash);
loadHealth().then(loadWorkspace).then(loadMcp).then(() => {
  if (!startQuestion) return;
  if (messages.length) showChat(null, [], viewProject);   // a new chat, not the end of the last one
  $("input").value = startQuestion;
  send();
});
showTab(location.hash.slice(1) || "chat");
poll();

// ------------------------------------------------------------------ the engine (serve/engine_control.py, --engine-background)
// The web server runs without the model: the header's button starts and stops it, its card shows how far a start has
// got (from the engine's log, and its memory while it reads the experts).  Hidden on a server without it.
let engine = null, engineTimer = null, engineAutoCard = false;
const ENGINE_BADGE = {stopped: ["", "Stopped"], starting: ["st-badge--reading", "Starting"], stopping: ["st-badge--queued", "Stopping"],
                      running: ["st-badge--generating", "Running"], error: ["st-badge--error", "Did not start"]};
const dur = (sec) => { const t = Math.max(0, Math.round(sec || 0)); return t >= 60 ? `${Math.floor(t / 60)} min ${t % 60} s` : `${t} s`; };
async function engineLoop() {
  clearTimeout(engineTimer);
  let next = 4000;
  try {
    const r = await fetch("engine", {headers: headers()});
    if (r.ok) {
      const d = await r.json();
      if (!d.can && !engine) { $("engine").hidden = true; return; }     // started without --engine-background
      setEngine(d);
      next = engine.state === "starting" || engine.state === "stopping" ? 700 : 4000;
    }
  } catch (e) { /* the server is restarting: try again */ }
  engineTimer = setTimeout(engineLoop, next);
}
function engineCard(open) {
  const show = open == null ? $("engine-card").hidden : open;
  $("engine-card").hidden = !show;
  $("engine-btn").setAttribute("aria-expanded", String(show));
  if (!show) engineAutoCard = false;
}
function setEngine(d) {
  const was = engine && engine.state;
  engine = d;
  $("engine").hidden = !d.can;                                          // an engine that cannot stop: nothing to switch
  const st = d.state, p = d.progress || {};
  const pct = st === "starting" ? Math.max(1, Math.round(p.percent || 0)) : st === "running" ? 100 : 0;
  const btn = $("engine-btn");
  btn.dataset.state = st;
  $("engine-label").textContent = st === "running" ? "Stop engine" : st === "starting" ? `Starting · ${pct}%` :
                                  st === "stopping" ? "Stopping…" : "Start engine";
  btn.title = st === "running" ? "Unload the model: the GPU and RAM are free again" : st === "starting" ? "Show the start" :
              "Load the model";
  $("engine-btn-bar").style.width = `${pct}%`;
  const [cls, label] = ENGINE_BADGE[st] || ENGINE_BADGE.stopped;
  $("engine-badge").className = `st-badge ${cls}`;
  $("engine-badge").textContent = label;
  let msg = "";
  const stats = [];                                                     // [label, value]: never cut, one column each
  if (st === "starting") {
    msg = p.message || "Starting the engine…";
    stats.push(["Elapsed", dur(p.elapsed_s)]);
    if (p.eta_s != null) stats.push(["Left", `about ${dur(p.eta_s)}`]);
    if (p.read_gib != null) stats.push(["Experts in RAM", p.total_gib ? `${fmt(p.read_gib, 1)} / ${fmt(p.total_gib, 1)} GiB` : `${fmt(p.read_gib, 1)} GiB`]);
    if (p.rate_gib_s) stats.push(["Speed", `${fmt(p.rate_gib_s, 2)} GiB/s`]);
  } else if (st === "running") {
    msg = `Running${d.version ? ` · engine ${d.version}` : ""}`;
    if (d.ready_s) stats.push(["Started in", dur(d.ready_s)]);
  } else if (st === "stopping") {
    msg = "Unloading the model…";
  } else if (st === "error") {
    msg = d.error || "The engine did not start.";
  } else {
    msg = d.held ? "Stopped. Requests get an error until you start it." : "Not loaded. The next request loads it.";
  }
  $("engine-msg").textContent = msg;
  $("engine-pct").textContent = st === "starting" ? `${pct}%` : "";
  $("engine-stats").innerHTML = stats.map(([k, v]) => `<div class="engine-stat"><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("");
  const prog = $("engine-progress");
  if (st === "error") prog.dataset.tone = "danger"; else if (st === "running") delete prog.dataset.tone; else prog.dataset.tone = "info";
  $("engine-bar").style.width = `${pct}%`;
  $("engine-line").textContent = st === "starting" ? p.line || "" : "";
  $("engine-line").title = $("engine-line").textContent;
  const act = $("engine-act");
  act.className = `st-btn ${st === "running" || st === "starting" ? "st-btn--danger" : "st-btn--primary"}`;
  act.textContent = st === "running" ? "Stop engine" : st === "starting" ? "Cancel start" : "Start engine";
  act.disabled = st === "stopping";
  if (was === "starting" && st === "running") {
    toast("success", "Engine running", d.ready_s ? `Started in ${dur(d.ready_s)}.` : "", 4000);
    if (engineAutoCard) setTimeout(() => { if (engine.state === "running") engineCard(false); }, 2500);
  }
  if (was === "starting" && st === "error") toast("error", "The engine did not start", d.error || "", 10000);
  if (lastMetrics) render(lastMetrics);                                   // the pill follows
}
async function engineDo(what, showCard) {
  if (showCard) { engineCard(true); engineAutoCard = true; }
  if (what === "stop" && engine && engine.state === "running") setEngine({...engine, state: "stopping"});
  try {
    const r = await fetch(`engine/${what}`, {method: "POST", headers: headers(true), body: "{}"});
    const j = await r.json().catch(() => ({}));
    if (r.status === 409) toast("warn", "A request is running", "Stop it, or wait until it is done; then stop the engine.", 6000);
    else if (!r.ok) throw new Error((j.error && j.error.message) || `HTTP ${r.status}`);
    if (j.state) setEngine(j);
    if (what === "stop" && j.result) toast("success", j.result === "cancelled" ? "Start cancelled" : "Engine stopped",
                                           "The GPU and RAM are free. Requests get an error until you start it again.", 4000);
  } catch (e) {
    toast("error", what === "start" ? "Could not start the engine" : "Could not stop the engine", e.message);
  }
  engineLoop();
}
$("engine-btn").onclick = () => {
  if (!engine) return;
  if (engine.state === "running") engineDo("stop");
  else if (engine.state === "starting" || engine.state === "stopping") engineCard();
  else engineDo("start", true);
};
$("engine-act").onclick = () => engineDo(engine && (engine.state === "running" || engine.state === "starting") ? "stop" : "start");
$("engine-close").onclick = () => engineCard(false);
document.addEventListener("click", (e) => { if (!$("engine-card").hidden && !e.target.closest("#engine, .st-toast")) engineCard(false); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("engine-card").hidden) engineCard(false); });
engineLoop();

// ------------------------------------------------------------------ what this chat changed, and undo
let changes = [];
async function refreshChanges() {
  const p = projectOf(chatProject);
  if (!p || !p.folder || !chatId || wsState !== "ready") { changes = []; renderChangesButton(); return; }
  try { changes = (await wsCall("changes.list", {project: p.id, chat: chatId})).files; } catch (e) { changes = []; }
  renderChangesButton();
  if ($("changes-drawer").dataset.open === "true") renderChanges();
}
function renderChangesButton() {
  const p = projectOf(chatProject);
  $("changes-btn").hidden = !(p && p.folder);
  $("changes-count").textContent = changes.length ? String(changes.length) : "";
  $("changes-btn").classList.toggle("has-changes", changes.length > 0);
}
// the review: one tab per changed file, its diff with the old and new line numbers, the code coloured like the viewer's
let changesSel = null;
function diffRows(diff, lang) {
  const hl = (t) => lang && lang !== "plaintext" && window.hljs ? hljs.highlight(t, {language: lang, ignoreIllegals: true}).value : esc(t);
  let o = 0, n = 0;
  const rows = [];
  for (const l of diff.replace(/\n$/, "").split("\n")) {
    if (l.startsWith("+++") || l.startsWith("---")) continue;
    const h = l.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$/);
    if (h) { o = +h[1]; n = +h[2]; rows.push(`<div class="dl dl--hunk"><span class="dl__t">${esc(l)}</span></div>`); continue; }
    if (l.startsWith("\\")) { rows.push(`<div class="dl dl--meta"><span class="dl__t">${esc(l)}</span></div>`); continue; }
    const kind = l[0] === "+" ? "add" : l[0] === "-" ? "del" : "ctx", t = l.slice(1);
    if (kind === "ctx" && !l.length && !rows.length) continue;
    rows.push(`<div class="dl dl--${kind}"><span class="dl__o">${kind === "add" ? "" : o++}</span><span class="dl__n">${kind === "del" ? "" : n++}</span>` +
              `<span class="dl__s">${kind === "add" ? "+" : kind === "del" ? "−" : ""}</span><span class="dl__t">${hl(t)}</span></div>`);
  }
  return {html: rows.join(""), digits: String(Math.max(o, n)).length};
}
function renderChanges() {
  const label = {added: "new", modified: "changed", deleted: "deleted"};
  if (!changes.some((f) => f.path === changesSel)) changesSel = changes.length ? changes[0].path : null;
  const add = changes.reduce((x, f) => x + f.added, 0), del = changes.reduce((x, f) => x + f.removed, 0);
  $("changes-sum").innerHTML = changes.length ? `${fmt(changes.length)} file${changes.length === 1 ? "" : "s"} · <span class="add">+${fmt(add)}</span> <span class="del">−${fmt(del)}</span>` : "";
  $("changes-tabs").hidden = !changes.length;
  $("changes-tabs").innerHTML = changes.map((f) => `<button type="button" class="changes-tab" role="tab" data-path="${esc(f.path)}" ` +
      `aria-selected="${f.path === changesSel}" tabindex="${f.path === changesSel ? 0 : -1}" title="${esc(f.path)}">` +
      `<span class="changes-tab__dot" data-status="${esc(f.status)}"></span><span class="changes-tab__name">${esc(f.path.split(/[\\/]/).pop())}</span>` +
      (f.added ? `<span class="add">+${f.added}</span>` : "") + (f.removed ? `<span class="del">−${f.removed}</span>` : "") + `</button>`).join("");
  const f = changes.find((x) => x.path === changesSel);
  $("changes-undo-file").disabled = !f;
  $("changes-undo-all").disabled = !changes.length;
  if (!f) { $("changes-view").innerHTML = `<div class="changes-empty muted">This chat has not changed any file (yet).</div>`; return; }
  const lang = window.hljs ? fileLang(f.path.split(/[\\/]/).pop()) : "";
  $("changes-view").innerHTML = `<div class="changes-file"><span class="changes-file__path">&lrm;${esc(f.path)}&lrm;</span>` +
    `<span class="st-badge" data-status="${esc(f.status)}">${label[f.status] || esc(f.status)}</span></div>` +
    (f.diff.trim() ? ((d) => `<div class="diff-view" style="--ln:${d.digits}ch"><div class="diff-rows">${d.html}</div></div>`)(diffRows(f.diff, lang))
                   : `<div class="changes-empty muted">No text to compare (an empty or binary file).</div>`);
  if (!window.hljs) loadViewer().then(() => { if ($("changes-drawer").dataset.open === "true") renderChanges(); }).catch(() => {});
}
function selectChange(path, focus) {
  changesSel = path;
  renderChanges();
  $("changes-view").scrollTop = 0;
  if (focus) { const t = $("changes-tabs").querySelector(`[data-path="${CSS.escape(path)}"]`); if (t) { t.focus(); t.scrollIntoView({inline: "nearest", block: "nearest"}); } }
}
$("changes-tabs").addEventListener("click", (e) => { const t = e.target.closest("[data-path]"); if (t) selectChange(t.dataset.path); });
$("changes-tabs").addEventListener("keydown", (e) => {            // arrow keys go from tab to tab
  const i = changes.findIndex((f) => f.path === changesSel), step = {ArrowRight: 1, ArrowLeft: -1}[e.key];
  if (step == null || i < 0) return;
  e.preventDefault();
  selectChange(changes[(i + step + changes.length) % changes.length].path, true);
});
$("changes-undo-file").onclick = () => { if (changesSel) undoChanges(changesSel); };
async function undoChanges(path) {
  try {
    const r = await wsCall("changes.undo", {project: chatProject, chat: chatId, ...(path ? {path} : {})});
    toast("success", "Undone", r.undone.length === 1 ? r.undone[0] : `${r.undone.length} files are as they were before this chat`, 3500);
  } catch (e) { toast("error", "Could not undo", e.message); }
  await refreshChanges();
  renderChanges();
}
$("changes-btn").onclick = async () => {
  await refreshChanges();
  renderChanges();
  $("changes-drawer").dataset.open = "true";
  $("changes-drawer").setAttribute("aria-hidden", "false");
  $("changes-scrim").hidden = false;
};
const closeChanges = () => {
  $("changes-drawer").dataset.open = "false";
  $("changes-drawer").setAttribute("aria-hidden", "true");
  $("changes-scrim").hidden = true;
};
$("changes-close").onclick = closeChanges;
$("changes-scrim").onclick = closeChanges;
let undoAllArmed = null;
$("changes-undo-all").onclick = () => {
  if (!undoAllArmed) {
    $("changes-undo-all").textContent = "Click again: undo all";
    undoAllArmed = setTimeout(() => { undoAllArmed = null; $("changes-undo-all").textContent = "Undo all"; }, 4000);
    return;
  }
  clearTimeout(undoAllArmed);
  undoAllArmed = null;
  $("changes-undo-all").textContent = "Undo all";
  undoChanges(null);
};

// ------------------------------------------------------------------ allow rules in the project drawer
function renderAllowRules(p) {
  const rules = p.allow || [];
  $("pd-allow").innerHTML = rules.length ? rules.map((r, i) => `<li class="pd-row" data-rule="${i}">` +
    `<code class="pd-row__text" title="${esc(r)} …">${esc(r)} <span class="muted">…</span></code>` +
    `<button type="button" class="st-btn st-btn--icon" data-rule-delete aria-label="Remove ${esc(r)}" title="Remove">` +
    `${icon("trash", "st-icon st-icon--sm")}</button></li>`).join("")
    : `<li class="pd-empty">None yet. Add one here, or with "Always allow" when a command asks.</li>`;
}
async function saveAllowRules(rules) {
  try {
    const {project} = await wsCall("project.save", {id: drawerProject, allow: rules});
    updateProject(project);
    renderAllowRules(project);
  } catch (e) { toast("error", "Could not save the rules", e.message); }
}
$("pd-allow").addEventListener("click", (e) => {
  const li = e.target.closest("[data-rule]");
  if (!li || !e.target.closest("[data-rule-delete]")) return;
  const rules = [...(projectOf(drawerProject).allow || [])];
  rules.splice(+li.dataset.rule, 1);
  saveAllowRules(rules);
});
$("pd-allow-add").onclick = () => {
  const v = $("pd-allow-new").value.replace(/\s+/g, " ").trim();
  if (!v) return;
  if (SHELL_META.test(v)) { toast("warn", "Only a plain command", "A rule cannot contain ; & | < > ` or $( - those always ask."); return; }
  $("pd-allow-new").value = "";
  saveAllowRules([...(projectOf(drawerProject).allow || []), v]);
};
$("pd-allow-new").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); $("pd-allow-add").click(); } });
