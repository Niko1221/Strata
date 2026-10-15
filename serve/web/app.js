// serve/web/app.js - the Strata web app: Chat, Monitor, About (and Profile, opt-in). No framework, no network beyond this server.
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
  set(k, v) { try { localStorage.setItem("strata." + k, JSON.stringify(v)); } catch (e) { /* private mode: in memory only */ } },
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
function showTab(name) {
  tab = ["chat", "images", "monitor", "about", "profile"].includes(name) ? name : "chat";
  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of ["chat", "images", "monitor", "about", "profile"]) $(`view-${v}`).hidden = v !== tab;
  if (tab === "images") $("img-prompt").focus();
  if (location.hash.slice(1) !== tab) history.replaceState(null, "", tab === "chat" ? location.pathname : `#${tab}`);
  if (tab === "chat") $("input").focus();
  if (tab === "monitor") loadMcp();
  if (tab === "about") loadConfig();
  if (tab === "profile") profileShown();
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
  profileRefresh();
};

let health = {model: "strata", images: false, max_context: 0}, healthKnown = false;
async function loadHealth() {
  try {
    health = await (await fetch("health")).json();
    $("attach-btn").title = health.images ? "Attach a text file or a picture (or drop it here)"
                                          : "Attach a text file (or drop it here)";
    $("chat-empty-sub").textContent = `${health.model} runs on this PC. Nothing leaves it.`;
    $("tab-btn-images").hidden = !health.image_generation;
    if (location.hash === "#images" && health.image_generation) showTab("images");
    healthKnown = true;
    $("tab-btn-profile").hidden = !health.profile;           // the key is there only when "profile" is on
    if (location.hash === "#profile" && health.profile) showTab("profile");
    else if (tab === "profile") profileShown();              // off: its card says how to turn it on
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
  const top = Math.max(max || 0, ...v, 1);
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
      if (!keyWarned) { keyWarned = true; toast("warn", "API key needed", "This server needs a key: add it under About > Settings.", 6000); }
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
  if (tab === "monitor") renderMonitor(live, hw, st, eng, h, last, m.requests || [], m.totals, m.requests_kept);
  if (tab === "monitor") renderConvCache(m.conversation_cache);
  if (tab === "about") renderAbout(eng, hw, st);
  if (tab === "profile") renderInflight(live);
}

// #596: the conversation cache - the prompt's state the engine keeps between requests (always), and the whole
// conversations it parks in RAM when "--conversation-cache-mib N" is in the run config's args (opt-in)
function since(t) {
  if (!t) return "";
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? "just now" : s < 3600 ? `${fmt(s / 60)} min ago` : `${fmt(s / 3600, 1)} h ago`;
}
function renderConvCache(c) {
  $("cc-card").hidden = !c;
  if (!c) return;                                  // an older server
  const pct = (a, b) => (b ? `${Math.min(100, (100 * a) / b)}%` : "0%");
  $("cc-bars").hidden = !c.enabled;
  if (c.enabled) {
    $("cc-slots-text").textContent = `${fmt(c.parked)} / ${fmt(c.slots)}`;
    $("cc-slots-bar").style.width = pct(c.parked, c.slots);
    const budget = c.budget_mib * 1048576;
    $("cc-mem-text").textContent = `${gb(c.bytes)} / ${gb(budget)} GB`;
    $("cc-mem-bar").style.width = pct(c.bytes, budget);
  }
  $("cc-sum").textContent = c.requests ? `${fmt(c.requests_reused)} of ${fmt(c.requests)} requests reused part of their prompt` : "";
  const share = c.prompt_tokens ? ` (${fmt((100 * c.reused_tokens) / c.prompt_tokens)}% of all prompt tokens)` : "";
  const event = c.last_event ? `${c.last_event === "parked" ? "Parked" : "Restored"} ${fmt(c.last_tokens)} tokens, ${since(c.last_at)}` : null;
  facts($("cc-facts"), [
    ["Last request", c.last_prompt != null ? `${fmt(c.last_reused || 0)} of ${fmt(c.last_prompt)} prompt tokens reused` : null],
    ["Reused since start", c.requests ? `${fmt(c.reused_tokens)} tokens${share}` : null],
    ["Parked / restored", c.enabled ? `${fmt(c.parks)} / ${fmt(c.restores)}${c.evictions ? ` · ${fmt(c.evictions)} evicted` : ""}` : null],
    ["Last switch", c.enabled ? event : null],
  ]);
  $("cc-note").textContent = c.enabled
    ? "A request that continues a parked conversation gets its state back instead of reading it again; the oldest goes when the slots or the memory are full."
    : "The engine keeps the last conversation's state, so a follow-up reads only what is new. To keep several conversations (agents taking turns), add \"--conversation-cache-mib\", \"8192\" to the run config's args (docs/DETAILS.md).";
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
            multi ? per((g) => (g.util == null ? "–" : `${fmt(g.util)}%`)) : st.gpu_name || (st.gpu_note ? "not available" : ""));
  spark("sp-gpu", h.gpu_util, 100);
  setMetric("vram", hw.gpu_mem_used == null ? null : gb(hw.gpu_mem_used), hw.gpu_mem_total ? `/ ${gb(hw.gpu_mem_total, 0)} GB` : "GB",
            multi ? per((g) => (g.mem_used == null ? "–" : `${gb(g.mem_used)} GB`))
                  : eng.expert_slots ? `${fmt(eng.expert_slots)} experts cached` : (st.gpu_note ? "not available on Windows AMD yet" : ""));
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
    body.innerHTML = `<tr><td colspan="9" class="muted">No requests yet</td></tr>`;
  } else {
    const badge = {stop: ["", "Done"], length: ["", "Max tokens"], cancel: ["st-badge--queued", "Stopped"],
                   disconnect: ["st-badge--queued", "Closed"], error: ["st-badge--error", "Error"]};
    body.innerHTML = requests.slice(0, reqShowAll ? requests.length : 12).map((r) => {
      const [cls, text] = badge[r.finish] || ["", r.finish || "–"];
      const t = new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      const proj = r.projection == null ? "" : ` <span class="st-badge${r.projection ? " st-badge--reading" : ""}" title="experimental speed projection ${r.projection ? "on" : "off"}">${r.projection ? "ESP" : "stock"}</span>`;
      // #588: the VRAM share; the PCIe share (--pcie-frac) beside it when there is one
      const hit = r.hit_rate == null ? "–" : `${(r.hit_rate * 100).toFixed(1)}%` +
        (r.pcie_share ? ` <span class="muted" title="routed experts the GPU read over PCIe (--pcie-frac) or another GPU computed">+${(r.pcie_share * 100).toFixed(1)}% PCIe</span>` : "");
      // Reused tokens are not prefetched again. Match the Monitor's existing Prefill metric.
      const fresh = r.prompt_tokens == null ? null : Math.max(0, r.prompt_tokens - (r.reused || 0));
      const prefillRate = fresh > 0 && r.prompt_ms > 0 ? fresh / (r.prompt_ms / 1000) : null;
      return `<tr><td>${esc(t)}</td><td><span class="st-badge ${cls}">${esc(text)}</span>${proj}</td><td class="num">${fmt(r.prompt_tokens)}</td>
        <td class="num">${fmt(r.reused)}</td><td class="num">${fmt(r.output_tokens)}</td><td class="num">${fmt(prefillRate)}</td><td class="num">${fmt(r.decode_tok_s, 1)}</td>
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
    ["GPU", st.gpu_name ? `${st.gpu_name}${hw.gpu_mem_total ? `, ${gb(hw.gpu_mem_total, 0)} GB` : ""}` : (st.gpu_note || "not readable (NVML)")],
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

// ------------------------------------------------------------------ Model settings (GET / POST /config, #564)
// A few documented keys of the run config (strata-<model>.json), for every client, from the next start on.  The
// server lists them, checks every value and keeps every other key of the file as it is.
let cfgKeys = [], cfgLoad = null, cfgStatus = 0;           // cfgStatus: what the last GET /config answered (0: no answer)
function loadConfig() {                                     // one load at a time: openModelSetting() waits for the one showTab started
  if (!cfgLoad) cfgLoad = loadConfigNow().finally(() => { cfgLoad = null; });
  return cfgLoad;
}
async function loadConfigNow() {
  let r;
  cfgStatus = 0;
  try { r = await fetch("config", {headers: headers()}); } catch (e) { return; }
  cfgStatus = r.status;
  if (!r.ok) { $("cfg-card").hidden = true; return; }       // no run config, an older server, or no key yet
  const c = await r.json();
  cfgKeys = c.keys || [];
  $("cfg-file").textContent = c.file || "";
  $("cfg-form").innerHTML = cfgKeys.map((k, i) => {
    const id = `cfg-${i}`, v = k.value;
    let input;
    if (k.kind === "bool" || k.kind === "enum") {
      const opts = k.kind === "bool" ? [["true", "on"], ["false", "off"]] : k.choices.map((x) => [x, x]);
      const cur = v == null ? "" : String(v);
      input = `<select class="st-input" id="${id}" data-key="${esc(k.key)}"><option value=""${cur === "" ? " selected" : ""}>default</option>` +
        opts.map(([val, text]) => `<option value="${esc(val)}"${cur === val ? " selected" : ""}>${esc(text)}</option>`).join("") + `</select>`;
    } else {
      const text = v == null ? "" : Array.isArray(v) ? v.join(", ") : String(v);
      input = `<input class="st-input" id="${id}" data-key="${esc(k.key)}" ${k.kind === "number" ? 'type="number" step="any" min="0"' : 'type="text"'} ` +
        `value="${esc(text)}" placeholder="default" autocomplete="off">`;
    }
    return `<label for="${id}" title="${esc(k.help)}">${esc(k.help)}<code>${esc(k.key)}</code></label>${input}`;
  }).join("");
  $("cfg-card").hidden = false;
}
function configValue(k, el) {
  const s = el.value.trim();
  if (s === "") return null;
  if (k.kind === "bool") return s === "true";
  if (k.kind === "number") return Number(s);
  return s;                                         // enum, or names (the server splits them at commas)
}
$("cfg-save").addEventListener("click", async () => {
  const set = {};
  cfgKeys.forEach((k, i) => {
    const v = configValue(k, $(`cfg-${i}`));
    const old = Array.isArray(k.value) ? k.value.join(", ") : k.value;
    if (JSON.stringify(v) !== JSON.stringify(old ?? null)) set[k.key] = v;
  });
  if (!Object.keys(set).length) { $("cfg-msg").textContent = "Nothing changed."; return; }
  try {
    const r = await fetch("config", {method: "POST", headers: headers(true), body: JSON.stringify({set})});
    const b = await r.json();
    if (!r.ok) throw new Error((b.error || {}).message || `HTTP ${r.status}`);
    $("cfg-msg").textContent = b.changed.length
      ? `Saved (${b.changed.join(", ")}); the earlier file is ${b.file}.bak. Start the model again to use it.` : "Nothing changed.";
    loadConfig();
  } catch (e) {
    $("cfg-msg").textContent = "";
    toast("error", "Not saved", String(e.message || e), 6000);
  }
});

// ------------------------------------------------------------------ Profile (GET /profile/*)
// The Profile tab, shown when /health says "profile": true (the run config's "profile" key): the last request's
// breakdown, the last 50, a measured run of tools/strata_profile.py, setup's calibration and the speed settings.
// Everything the server sends goes into the page as text (textContent, setAttribute with numbers, createElementNS).
const PROMPT_GROUPS = ["experts over PCIe", "dequant", "expert GEMM", "attention", "GDN", "other"];
const DECODE_GROUPS = ["GPU", "CPU experts", "draft", "adapt wait", "other"];
const SVG_NS = "http://www.w3.org/2000/svg";
const pf = {timer: null, busy: false, again: false, keyGen: 0, ticks: 0, stamps: {}, key401: false, down: false, status: null,
            settings: null, settingsDue: true, seen: {run: false, tune: false}, unloaded: null, runState: null, tuneState: null};

const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
};
const pct = (x) => (typeof x === "number" ? `${Math.round(x * 100)}%` : "–");
const msfmt = (s) => {                                       // 681 -> "11 min 21 s"
  const t = Math.max(0, Math.round(s || 0)), m = Math.floor(t / 60);
  return m ? `${m} min ${t % 60} s` : `${t} s`;
};
function pfText(id, value) { const e = $(id), v = String(value); if (e.textContent !== v) e.textContent = v; }
function logText(pre, lines) {
  const v = lines.join("\n");
  if (pre.textContent !== v) { pre.textContent = v; pre.scrollTop = pre.scrollHeight; }
}

async function pget(path) {
  const r = await fetch(path, {headers: headers(), cache: "no-store"});
  if (!r.ok) throw {status: r.status};
  return r.json();
}
function ppost(path) {
  return fetch(path, {method: "POST", headers: headers(true), body: "{}"});
}
async function pstart(path) {                                // POST a run; an error answer becomes the Error's message
  const r = await ppost(path);
  let b = {};
  try { b = await r.json(); } catch (e) { /* no body */ }
  if (!r.ok) throw new Error((b.error && b.error.message) || `HTTP ${r.status}`);
}

// a stacked bar with its legend: values[name] in `unit` ("%": shares of 1, or "ms"); colour never carries a number alone
function stackBar(barId, legendId, names, values, unit) {
  const bar = $(barId), legend = $(legendId);
  bar.replaceChildren();
  legend.replaceChildren();
  const total = names.reduce((s, n) => s + (values[n] || 0), 0) || 1;
  names.forEach((name, i) => {
    const v = values[name] || 0, shown = unit === "%" ? pct(v) : `${fmt(v, 1)} ms`;
    const seg = el("span", `pf-c${i}`);
    seg.style.width = `${(v / total) * 100}%`;
    seg.title = `${name}: ${shown}`;
    bar.append(seg);
    const item = el("span");
    item.append(el("i", `pf-c${i}`), `${name} ${shown}`);
    legend.append(item);
  });
}
function table(tbl, header, rows) {
  const t = typeof tbl === "string" ? $(tbl) : tbl;
  t.replaceChildren();
  const head = el("tr"), thead = el("thead");
  header.forEach((h, i) => head.append(el("th", i ? "num" : "", h)));
  thead.append(head);
  t.append(thead);
  const body = el("tbody");
  rows.forEach((row) => {
    const tr = el("tr");
    row.forEach((c, i) => tr.append(el("td", i ? "num" : "", c)));
    body.append(tr);
  });
  t.append(body);
}
// one bar per request, oldest left; value(e) is a number, an array (stacked, bottom first) or null; a dashed line
// marks each time the engine started again
function sparkBars(svgId, entries, value, label, classes) {
  const svg = $(svgId);
  svg.replaceChildren();
  const w = 100 / Math.max(entries.length, 10);
  const parts = entries.map((e) => { const v = value(e); return v == null ? null : Array.isArray(v) ? v : [v]; });
  const highest = Math.max(0, ...parts.map((p) => (p ? p.reduce((a, b) => a + b, 0) : 0)));
  const top = highest || 1;
  entries.forEach((e, i) => {
    if (i && e.engine_run !== entries[i - 1].engine_run) {
      const mark = document.createElementNS(SVG_NS, "line");
      mark.setAttribute("class", "pf-mark");
      for (const [k, v] of [["x1", i * w], ["x2", i * w], ["y1", 0], ["y2", 40]]) mark.setAttribute(k, v.toFixed(2));
      const t = document.createElementNS(SVG_NS, "title");
      t.textContent = "The model loaded again here";
      mark.append(t);
      svg.append(mark);
    }
    let y = 40;
    (parts[i] || []).forEach((v, k) => {
      const h = (v / top) * 38;
      if (!(h > 0)) return;
      y -= h;
      const rect = document.createElementNS(SVG_NS, "rect");
      rect.setAttribute("class", classes[k]);
      for (const [a, b] of [["x", i * w + w * 0.1], ["width", w * 0.8], ["y", y], ["height", h]]) rect.setAttribute(a, b.toFixed(2));
      const t = document.createElementNS(SVG_NS, "title");
      t.textContent = label(e);
      rect.append(t);
      svg.append(rect);
    });
  });
  return highest;
}

// ---- Last request
function renderLast(data) {
  const r = data.request;
  $("pf-last-empty").hidden = Boolean(r);
  $("pf-last").hidden = !r;
  if (!r) {
    pfText("pf-last-empty", data.log === false
      ? 'The engine writes no log file (no "log" in the run config), so there is nothing to read.'
      : "No request with its breakdown yet. Send one from Chat or any API client; it appears here when it ends.");
    pfText("pf-last-note", "");
    return;
  }
  const p = r.prompt, d = r.decode, h = r.hit;
  pfText("pf-decode-tps", fmt(d.tok_s, 1));
  pfText("pf-hit", pct(h && h.hit_rate));
  pfText("pf-pcie", h && h.pcie_share ? `plus ${pct(h.pcie_share)} of all routed experts read by the card over PCIe`
                                      : "of the experts looked up while decoding");
  pfText("pf-prompt-tps", p.tok_s ? fmt(p.tok_s) : "–");
  pfText("pf-window", d.groups ? fmt(d.groups.ms_per_window, 1) : "–");
  pfText("pf-last-note", `prompt ${fmt(p.tokens)} tokens · ${fmt(d.generated)} generated`);
  if (p.groups) {
    stackBar("pf-prompt-bar", "pf-prompt-legend", PROMPT_GROUPS, p.groups.share, "%");
  } else {
    $("pf-prompt-bar").replaceChildren();
    pfText("pf-prompt-legend", p.read ? "No prefill timing line for this request." : "Nothing was read: the whole prompt was reused.");
  }
  if (d.groups) {
    stackBar("pf-decode-bar", "pf-decode-legend", DECODE_GROUPS, d.groups.ms, "ms");
  } else {
    $("pf-decode-bar").replaceChildren();
    pfText("pf-decode-legend", "No decode timing line for this request.");
  }
  const g = d.groups;
  const dl = $("pf-facts");
  dl.replaceChildren();
  for (const [k, v] of [
    ["Prompt tokens", fmt(p.tokens)],
    ["Reused / read", `${fmt(p.reused)} / ${fmt(p.read)}`],
    ["Generated", fmt(d.generated)],
    ["Tokens per window", g ? fmt(g.tokens_per_window, 2) : "–"],
    ["CPU experts per layer-window", g && typeof g.cpu_experts === "number" ? `${fmt(g.cpu_experts, 1)} (${fmt(g.cpu_entries, 1)} entries)` : "–"],
    ["Drafts accepted", d.drafts_offered ? `${fmt(d.drafts_accepted)} of ${fmt(d.drafts_offered)}` : "–"],
  ]) dl.append(el("dt", "", k), el("dd", "", v));
  pfText("pf-kv", r.kv ? `KV streaming: ${pct(r.kv.kv_hit)} of this request's ${fmt(r.kv.block_reads)} block reads from VRAM, ${fmt(r.kv.ram_mib)} MiB from RAM` : "");
}
// the request that is running now, from the 1 s /metrics poll (the same text as the header pill)
function renderInflight(live) {
  if (!live || !health.profile) return;
  const busy = live.state === "reading" || live.state === "generating";
  $("pf-inflight").hidden = !busy;
  if (busy) pfText("pf-inflight", `A request is running (${$("pill-text").textContent}); its breakdown appears when it ends`);
  const unloaded = live.state === "unloaded";
  if (pf.unloaded !== null && pf.unloaded !== unloaded) pf.settingsDue = true;   // loaded or unloaded: the settings it reports change
  pf.unloaded = unloaded;
}

// ---- Recent requests
function renderHistory(h) {
  const rows = h.requests || [];
  $("pf-hist-empty").hidden = rows.length > 0;
  $("pf-hist").hidden = rows.length === 0;
  if (!rows.length) return;
  const what = (e, text) => `#${e.n} · ${fmt(e.prompt_tokens)} tokens of context · ${text}`;
  const tps = sparkBars("pf-spark-tps", rows, (e) => e.decode_tok_s, (e) => what(e, `${fmt(e.decode_tok_s, 1)} tok/s`), ["pf-c2"]);
  const hit = sparkBars("pf-spark-hit", rows, (e) => e.hit_rate, (e) => what(e, `hit rate ${pct(e.hit_rate)}`), ["pf-c0"]);
  pfText("pf-max-tps", tps ? `highest ${fmt(tps, 1)}` : "");     // the bars start at zero: this is the scale
  pfText("pf-max-hit", hit ? `highest ${pct(hit)}` : "");
  const ms = sparkBars("pf-spark-ms", rows, (e) => {
    if (!e.ms) return null;
    const card = e.ms.GPU || 0, cpu = e.ms["CPU experts"] || 0;
    return [card, cpu, Math.max(0, (e.ms_per_window || 0) - card - cpu)];
  }, (e) => what(e, `${fmt(e.ms_per_window, 1)} ms per window (card ${fmt(e.ms && e.ms.GPU, 1)}, CPU experts ${fmt(e.ms && e.ms["CPU experts"], 1)})`),
  ["pf-c0", "pf-c1", "pf-c5"]);
  pfText("pf-max-ms", ms ? `highest ${fmt(ms, 1)} ms` : "");
  const body = $("pf-hist-body");
  body.replaceChildren();
  rows.slice().reverse().forEach((e) => {
    const i = rows.indexOf(e), again = i > 0 && e.engine_run !== rows[i - 1].engine_run;
    const t = new Date(e.seen_at * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
    const tr = el("tr");
    const cells = [`${e.n}`, again ? `${t} · model loaded again` : t, ctxfmt(e.prompt_tokens), fmt(e.prompt_tok_s), fmt(e.decode_tok_s, 1),
                   pct(e.hit_rate), fmt(e.ms_per_window, 1), fmt(e.ms && e.ms.GPU, 1), fmt(e.ms && e.ms["CPU experts"], 1), e.prompt_top || "–"];
    cells.forEach((c, k) => tr.append(el("td", k > 1 && k < 9 ? "num" : "", c)));
    body.append(tr);
  });
}

// ---- Measured run
const RUN_LABEL = {idle: () => "No run yet", running: (s) => `Running · ${msfmt(s.elapsed_s)}`,
                   done: (s) => `Done in ${msfmt(s.elapsed_s)}`, failed: (s) => `Failed (exit ${s.returncode})`};
function setOutput(detailsId, logId, lines, state) {          // closed while it runs and when it is done, open when it failed
  const out = $(detailsId);
  out.hidden = !lines.length;
  logText($(logId), lines);
  return state;
}
function renderRun(st) {                                      // the state, the newest line and the output: every tick while it runs
  pfText("pf-run-state", (RUN_LABEL[st.state] || (() => st.state))(st));
  $("pf-run-help").hidden = st.state !== "idle";
  $("pf-run-wait").hidden = st.state !== "running";
  pfText("pf-run-step", st.state === "running" && st.lines.length ? st.lines[st.lines.length - 1] : "");
  setOutput("pf-run-out", "pf-run-log", st.lines, st.state);
  if (pf.runState !== st.state) { pf.runState = st.state; $("pf-run-out").open = st.state === "failed"; }
}
function renderRunReport(st) {                                // the tables and findings: only when the run's state or report changed
  const rep = st.report, res = rep && rep.results && rep.results.results;
  const note = $("pf-run-note");
  note.hidden = !(st.state === "failed" || (rep && !res && st.state !== "running"));
  if (st.state === "failed") pfText("pf-run-note", "The run failed; nothing was changed. Its output says why." +
    (rep && !res ? " The report folder could not be read." : ""));
  else if (rep && !res) pfText("pf-run-note", "The report was written but its files could not be read.");
  $("pf-report").hidden = !(res && st.state === "done");
  pf.report = res && st.state === "done" ? rep : null;
  if (pf.settings) renderSettings(pf.settings);               // "what the profile said" follows the report
  if (!pf.report) return;
  table("pf-t-prompts", ["Read", "Tokens", "Prompt tok/s", "GPU ms", ...PROMPT_GROUPS],
        Object.entries(res.prompts || {}).map(([k, v]) => [k, fmt(v.prompt_tokens), fmt(v.summary && v.summary.prompt_tok_s != null ? v.summary.prompt_tok_s : v.prompt_tps),
                                                           v.groups ? fmt(v.groups.gpu_ms) : "–", ...PROMPT_GROUPS.map((g) => (v.groups ? pct(v.groups.share[g]) : "–"))]));
  table("pf-t-decode", ["Request", "Context", "Decode tok/s", "Hit rate", "PCIe share", "ms/window", ...DECODE_GROUPS, "Tokens/window", "CPU experts per layer-window"],
        Object.entries(res.decode || {}).map(([k, v]) => [k, ctxfmt(v.context), fmt(v.decode_tps && v.decode_tps.median, 1), pct(v.hit && v.hit.hit_rate), pct(v.hit && v.hit.pcie_share),
                                                           v.groups ? fmt(v.groups.ms_per_window, 1) : "–", ...DECODE_GROUPS.map((g) => (v.groups ? fmt(v.groups.ms[g], 1) : "–")),
                                                           v.groups ? fmt(v.groups.tokens_per_window, 2) : "–", v.groups ? fmt(v.groups.cpu_experts, 1) : "–"]));
  const turns = (res.agent && res.agent.turns) || [];
  table("pf-t-agent", ["Turn", "Prompt tokens", "Reused", "Read", "Prompt tok/s", "Decode tok/s", "Hit rate", "ms/window", "Card ms", "CPU experts ms"],
        turns.map((t) => [t.turn, fmt(t.prompt_tokens), fmt(t.reused_tokens), fmt(t.read_tokens), fmt(t.prompt_tps), fmt(t.decode_tps), pct(t.hit && t.hit.hit_rate),
                          t.groups ? fmt(t.groups.ms_per_window, 1) : "–", t.groups ? fmt(t.groups.ms.GPU, 1) : "–", t.groups ? fmt(t.groups.ms["CPU experts"], 1) : "–"]));
  const shown = String(rep.path).match(/(bench[\\/]results[\\/].*)$/);
  pfText("pf-report-path", `Report: ${shown ? shown[1] : rep.path} (README.md, results.json)`);
  $("pf-report-copy").onclick = () => copyText(String(rep.path), $("pf-report-copy"));
  if (!pf.settings) renderFindings();
}
// each finding is a sentence and one chip per setting it names, with the value loaded now and at the next load
function renderFindings() {
  const ul = $("pf-findings");
  ul.replaceChildren();
  const byId = {};
  for (const row of (pf.settings && pf.settings.settings) || []) byId[row.id] = row;
  for (const f of (pf.report && pf.report.findings) || []) {
    const li = el("li", "", f.text);
    if (f.settings.length) {
      const chips = el("div", "pf-chips");
      for (const id of f.settings) {
        const row = byId[id];
        const next = row && settingValue(row, row.next_load, "default");
        const b = el("button", "chip pf-chip", !row ? id : row.loaded_now == null ? `${row.name}: ${next} at the next load`
                     : `${row.name} ${settingValue(row, row.loaded_now, "—")} loaded · ${next} next load`);
        b.type = "button";
        b.onclick = () => {
          const tr = $(`pf-set-${id}`);
          if (tr) { tr.scrollIntoView({block: "center"}); tr.classList.add("pf-flash"); setTimeout(() => tr.classList.remove("pf-flash"), 1500); }
        };
        chips.append(b);
      }
      li.append(chips);
    }
    ul.append(li);
  }
}

// ---- Tune this PC
function renderTune(t, why) {                                 // the state, the newest line and the output: every tick while it runs
  const states = {idle: "Not run", running: t && `Running · ${msfmt(t.elapsed_s)}`, done: t && `Done in ${msfmt(t.elapsed_s)}`,
                  failed: t && `Failed (exit ${t.returncode})`};
  pfText("pf-tune-state", t ? states[t.state] || t.state : why || "Not available");
  $("pf-tune-help").hidden = !(!t || t.state === "idle");
  $("pf-tune-warn").hidden = !t || t.state !== "running";
  $("pf-tune-failed").hidden = !t || t.state !== "failed";
  pfText("pf-tune-step", t && t.state === "running" && t.lines.length ? t.lines[t.lines.length - 1] : "");
  setOutput("pf-tune-out", "pf-tune-log", t ? t.lines : [], t && t.state);
  if (t && pf.tuneState !== t.state) { pf.tuneState = t.state; $("pf-tune-out").open = t.state === "failed"; }
}
function renderTuneResult(t) {                                // the before and after: only when the tuning's state or report changed
  const done = !!t && t.state === "done";
  $("pf-tune-result").hidden = !done;
  if (!done) return;
  const rep = t.report;
  const changes = (rep && rep.changes) || [];
  const anyChanged = changes.some((c) => c.changed);
  const kept = !!rep && /already the fastest/.test(rep.summary || "");
  const text = kept && anyChanged ? "The default settings were the fastest here: the values an earlier tuning wrote went back to the defaults."
    : kept ? "The default settings are already the fastest on this PC; no setting changed."
    : changes.length && !anyChanged ? "The calibration kept the settings the model started with; no setting changed." : "";
  $("pf-tune-kept").hidden = !text;
  pfText("pf-tune-kept", text);
  $("pf-tune-table-wrap").hidden = !changes.length;
  table("pf-t-tune", ["Setting", "Before", "After"], changes.map((c) => {
    const loaded = c.before_loaded != null ? ` (${c.before_loaded})` : "";
    return [c.flag, c.before == null ? `engine default${loaded}` : c.before,
            c.after == null ? `engine default${c.changed ? "" : loaded}` : c.after];
  }));
  [...$("pf-t-tune").tBodies[0].rows].forEach((tr, i) => tr.classList.toggle("pf-changed", changes[i].changed));
  pfText("pf-tune-toks", rep && rep.tok_s != null ? `Calibration's measured speed with these: ${fmt(rep.tok_s, 1)} tok/s` : "");
  $("pf-tune-toks").hidden = !(rep && rep.tok_s != null);
  pfText("pf-tune-note", t.note || "");
  $("pf-tune-note").hidden = !t.note;
}

// ---- Settings that matter for speed
function settingValue(row, v, fallback) {
  if (v == null) return fallback;
  return row.id === "kv_resident" && v === "0" ? "0 (all in VRAM)" : v;
}
function linkButton(text, run) {
  const b = el("button", "pf-link", text);
  b.type = "button";
  b.onclick = run;
  return b;
}
// About > Model settings, scrolled to and focused on the input of a run config key
async function openModelSetting(key) {
  showTab("about");
  await loadConfig();
  for (const input of document.querySelectorAll("#cfg-form [data-key]")) {
    if (input.dataset.key === key) { input.scrollIntoView({block: "center"}); input.focus(); return; }
  }
  const why = cfgStatus === 401 ? "This server needs an API key: add it under About > Settings."
    : cfgStatus === 404 ? "This server was started without a run config, so there is nothing to change here."
    : cfgStatus === 0 ? "The server did not answer."
    : "This setting is not in this server's Model settings list.";
  toast("info", "Not in Model settings", why, 6000);
}
function renderSettings(data) {
  pf.settings = data;
  pfText("pf-settings-file", data.file ? `Run config: ${data.file}` : "No run config");
  const findings = pf.report ? pf.report.findings || [] : null;
  const body = $("pf-settings-body");
  body.replaceChildren();
  for (const row of data.settings || []) {
    const tr = el("tr");
    tr.id = `pf-set-${row.id}`;
    const name = el("td");
    name.append(el("code", "", row.name));
    const next = el("td", "", settingValue(row, row.next_load, "default"));
    if (data.file && row.run_config !== row.next_load) {      // the file and the start command differ: a hand edit, or a flag the server adds itself
      next.append(el("div", "muted small", `the run config file has ${settingValue(row, row.run_config, "default")}; the server reads it when Strata starts`));
    }
    const said = el("td");
    said.append(el("div", "", row.about));
    if (findings === null) {
      said.append(el("div", "muted small", "No run yet"));
    } else {
      const named = findings.filter((f) => f.settings.includes(row.id));
      if (!named.length) {
        said.append(el("div", "muted small", "Not named by the last run"));
      } else {
        const d = el("details", "st-collapse pf-named");
        d.append(el("summary", "", `Named by the last run in ${named.length} finding${named.length > 1 ? "s" : ""}`));
        const inner = el("div", "st-collapse__body");
        named.forEach((f) => inner.append(el("p", "", f.text)));
        d.append(inner);
        said.append(d);
      }
    }
    const change = el("td");
    if (row.edit === "about") change.append(linkButton("Model settings", () => openModelSetting(row.id)));
    else if (row.calibrated) change.append(linkButton("Tune this PC", () => { $("pf-tune").scrollIntoView({block: "center"}); $("pf-tune").focus(); }));
    else change.append(el("span", "muted small", row.kind === "env" ? "In the run config's env (setup writes it)"
                         : row.kind === "key" ? "A key of the run config" : "In the run config's args (setup writes it)"));
    tr.append(name, el("td", "", settingValue(row, row.loaded_now, "—")), next, said, change);
    body.append(tr);
  }
  renderFindings();
}

// ---- the poll: one timer; every 2 s while the tab is shown, every 10 s for a run seen running, else nothing
function setNotice(kind, text) {
  $("pf-notice").hidden = !kind;
  $("pf-notice-btn").hidden = kind !== "key";
  if (kind === "key") pfText("pf-notice-text", "This server needs an API key: add it under About > Settings");
  if (kind === "down") pfText("pf-notice-text", "Server not reachable · retrying");
  if (kind === "answered") pfText("pf-notice-text", `${text} · retrying`);
}
function syncButtons() {
  const st = pf.status, tune = st && st.tune;
  const running = !!st && (st.state === "running" || (!!tune && tune.state === "running"));
  const blocked = pf.down || pf.key401 || !st || running;
  $("pf-run").disabled = blocked;
  $("pf-tune").disabled = blocked || !tune;
}
function stamped(key, data, render) {
  const s = JSON.stringify(data);
  if (pf.stamps[key] !== s) { pf.stamps[key] = s; render(data); }
}
function finished(kind, state) {
  if (state !== "done" && state !== "failed") return;
  const ok = state === "done", here = tab === "profile" && !document.hidden;
  const title = kind === "tune" ? (ok ? "Tuning done" : "Tuning failed") : (ok ? "Profile run done" : "Profile run failed");
  if (here && !(kind === "tune" && ok)) return;                 // the state on the page says it; a finished tuning also toasts
  toast(ok ? "success" : "error", title, "", 8000, here ? null : {label: "Open", run: () => showTab("profile")});
}
function onStatus(st) {
  const run = st.state === "running", tune = !!st.tune && st.tune.state === "running";
  const ended = [];
  if (pf.seen.run && !run) ended.push(["run", st.state]);
  if (pf.seen.tune && !tune) ended.push(["tune", st.tune ? st.tune.state : "failed"]);
  pf.seen = {run, tune};
  pf.status = st;
  // four stamps, so a part is rebuilt only when it changed: the tables, findings and settings rows do not follow the clock
  const tail = (l) => ({n: l.length, last: l.length ? l[l.length - 1] : null});
  const t = st.tune;
  stamped("run", {state: st.state, e: st.elapsed_s, rc: st.returncode, ...tail(st.lines)}, () => renderRun(st));
  stamped("run-report", {state: st.state, rc: st.returncode, path: st.report && st.report.path,
                         res: !!(st.report && st.report.results), found: st.report && st.report.findings ? st.report.findings.length : 0},
          () => renderRunReport(st));
  stamped("tune", t ? {state: t.state, e: t.elapsed_s, rc: t.returncode, ...tail(t.lines)} : {why: st.tune_unavailable},
          () => renderTune(t, st.tune_unavailable));
  stamped("tune-result", t ? {state: t.state, rc: t.returncode, note: t.note, report: t.report} : null, () => renderTuneResult(t));
  syncButtons();
  if (ended.length) pf.settingsDue = true;
  ended.forEach(([kind, state]) => finished(kind, state));
}
async function profileTick() {
  const [last, st, hist] = await Promise.all([pget("profile/last"), pget("profile/status"), pget("profile/history")]);
  pf.down = false;
  setNotice(null);
  onStatus(st);
  stamped("last", last, renderLast);
  stamped("hist", hist, renderHistory);
  pfText("pf-updated", `Updated ${new Date().toLocaleTimeString()}`);
  if (pf.settingsDue || ++pf.ticks % 15 === 0) {                // when the tab opens, a run ends, the model (un)loads, else every 30 s
    pf.settingsDue = false;
    try {
      stamped("settings", await pget("profile/settings"), renderSettings);
    } catch (e) {                                               // its own failure leaves the rest of the tab alone
      if (e && e.status === 401) throw e;
      pf.settingsDue = true;
    }
  }
}
async function profilePoll() {
  clearTimeout(pf.timer);
  if (!health.profile || pf.key401) return;
  if (pf.busy) { pf.again = true; return; }
  const here = tab === "profile" && !document.hidden;
  if (!here && !pf.seen.run && !pf.seen.tune) return;           // nothing this page saw running: no profile fetch at all
  const gen = pf.keyGen;
  pf.busy = true;
  try {
    if (here) await profileTick(); else onStatus(await pget("profile/status"));
  } catch (e) {
    const status = e && e.status;
    if (status === 401) {
      if (gen === pf.keyGen) {                                   // a key entered while this fetch was in flight is tried again
        pf.key401 = true;
        setNotice("key");
      }
    } else if (status === 404) {                                 // the server no longer has the tab: /health says so
      await loadHealth();
      if (health.profile) {                                      // still there: some other 404 (a proxy)
        pf.down = true;
        setNotice("answered", "The server answered 404");
      }
    } else {
      pf.down = true;
      pf.settingsDue = true;
      setNotice(status ? "answered" : "down", `The server answered ${status}`);   // no status: fetch itself failed
    }
    syncButtons();
  }
  pf.busy = false;
  if (pf.key401) return;                                         // stops until the key changes (profileRefresh)
  pf.timer = setTimeout(profilePoll, pf.again ? 0 : here ? 2000 : 10000);
  pf.again = false;
}
function profileRefresh() {                                      // at once: the key changed, the tab opened, the page is visible
  pf.key401 = false;
  pf.keyGen++;
  if (!health.profile) return;
  if (!pf.down) setNotice(null);
  profilePoll();
}
function profileShown() {
  if (!healthKnown) return;                                      // loadHealth() calls this again when it knows
  const on = health.profile === true;
  $("pf-off").hidden = on;
  $("pf-cards").hidden = !on;
  if (!on) return;
  pf.settingsDue = true;
  profileRefresh();
}
document.addEventListener("visibilitychange", () => { if (!document.hidden && tab === "profile") profileRefresh(); });

$("pf-run").onclick = async () => {
  $("pf-run").disabled = true;
  try { await pstart("profile/run"); } catch (e) { toast("error", "Not started", String(e.message || e), 6000); }
  profileRefresh();
};
$("pf-tune").onclick = async () => {
  if (!confirm("Tune this PC now? The model is unloaded while setup's calibration measures the PCIe share, the draft floor, " +
               "the CPU workers and the adaptive expert tier. It took 681 s on an RTX 5070 Ti + Ryzen 7 5800X3D; setup's " +
               "estimate is 15-30 minutes, longer on a slow card. Requests get 503 meanwhile, and the model loads again " +
               "with the tuned settings on the next request.")) return;
  $("pf-tune").disabled = true;
  try { await pstart("profile/tune"); } catch (e) { toast("error", "Not started", String(e.message || e), 6000); }
  profileRefresh();
};
$("pf-notice-btn").onclick = () => { showTab("about"); $("api-key").focus(); };
$("pf-off-link").onclick = () => openModelSetting("profile");
$("pf-profile-link").onclick = () => openModelSetting("profile");

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
let messages = store.get("chat", []);
let attachments = [];                 // {name, url}
let busy = null;                      // {controller, msg}

function saveChat() {
  store.set("chat", messages.map((m) => ({...m, images: (m.images || []).map((i) => ({name: i.name})),
                                           files: (m.files || []).map((f) => ({name: f.name}))})));
}
function timeStr(t) { return new Date(t).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}); }

function msgEl(m, i) {
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
      `<button class="st-btn st-btn--icon" data-msg-copy aria-label="Copy the answer" title="Copy">${icon("copy")}</button></div>`;
    updateAssistant(el, m, false);
  }
  return el;
}
// One MCP tool call in the answer: a compact block (name, state, a one-line preview) that opens to the arguments and
// the result as the model read it.  Its body is built only while open: a result can be 20,000 characters.
const TOOL_STATE = {writing: ["st-badge--reading", "Writing"], running: ["st-badge--generating", "Running"], done: ["", "Done"],
                    error: ["st-badge--error", "Error"], skipped: ["st-badge--queued", "Not run"]};
function toolHtml(t, k) {
  const [cls, label] = TOOL_STATE[t.state] || ["", t.state];
  const args = t.arguments == null ? "" : JSON.stringify(t.arguments, null, 2);
  const preview = t.result != null ? t.result : args.replace(/\s+/g, " ");
  let body = "";
  if (t.open) {
    body = `<div class="tool-call__label">Arguments</div><pre class="tool-call__pre">${esc(args || "(being written)")}</pre>`;
    if (t.result != null) {
      body += `<div class="tool-call__label">${t.ok ? "Result" : "Error"}${t.chars ? ` · ${fmt(t.chars)} characters` : ""}` +
              `${t.truncated ? ", cut for the model" : ""}</div><pre class="tool-call__pre">${esc(t.result)}</pre>`;
    }
  }
  return `<details class="st-collapse tool-call" data-tool="${k}" data-state="${esc(t.state)}"${t.open ? " open" : ""}>` +
    `<summary>${icon("tool", "st-icon st-icon--sm")}<span class="tool-call__name" title="${esc(t.name || "")}">${esc(t.tool || t.name || "tool")}</span>` +
    (t.server ? `<span class="muted small">${esc(t.server)}</span>` : "") +
    `<span class="tool-call__preview muted">${esc(preview.slice(0, 200))}</span>` +
    `<span class="st-badge ${cls}">${esc(label)}</span>${t.ms != null && t.state !== "skipped" ? `<span class="muted small">${fmt(t.ms / 1000, 1)} s</span>` : ""}` +
    `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body">${body}</div></details>`;
}
// the answer's text with the tool blocks where the model called them
function answerHtml(m) {
  if (!m.tools || !m.tools.length) return markdown(m.text || "");
  let html = "", pos = 0;
  m.tools.forEach((t, k) => {
    const at = Math.min(Math.max(t.at || 0, pos), m.text.length);
    if (at > pos) html += markdown(m.text.slice(pos, at));
    pos = at;
    html += toolHtml(t, k);
  });
  return html + markdown(m.text.slice(pos));
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
    bubble.innerHTML = answerHtml(m);
    if (streaming) bubble.classList.add("cursor"); else bubble.classList.remove("cursor");
  }
  el.querySelector(".meta-text").textContent = m.meta || (streaming ? "" : m.stopped ? "Stopped" : "");
  el.querySelector("[data-msg-copy]").hidden = streaming || !m.text;
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
  const mc = e.target.closest("[data-msg-copy]");
  if (mc) { const i = +mc.closest(".st-msg").dataset.i; copyText(messages[i].text, mc); return; }
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

function apiMessages() {
  const out = [];
  for (const m of messages) {
    if (m.role === "user") {
      const imgs = (m.images || []).filter((i) => i.url);
      const text = userText(m);
      out.push({role: "user", content: imgs.length ? [{type: "text", text},
        ...imgs.map((i) => ({type: "image_url", image_url: {url: i.url}}))] : text});
    } else if (!(busy && busy.msg === m)) {            // the answer being asked for now is not history yet
      out.push(...assistantMessages(m));
    }
  }
  return out;
}
// An answer that used MCP tools goes back as the model wrote it: per round the text before the calls, the calls and
// their results (as the model read them), then the rest - so the next question can build on what the tools found.
function assistantMessages(m) {
  const ran = (m.tools || []).filter((t) => t.round != null && t.result != null && t.state !== "skipped");
  // #1392: a turn with no answer text (only reasoning, a stop before the first content token, or an error) still goes
  // back as an assistant turn, so the history keeps alternating and a reasoning model sees that it already answered.
  // The server leaves such a turn out of the prompt itself (serve/frontend.py, #843); reasoning_content rides along.
  if (!ran.length) {
    const msg = {role: "assistant", content: m.text || ""};
    if (!m.text && m.reasoning) msg.reasoning_content = m.reasoning;
    return [msg];
  }
  const out = [];
  let pos = 0;
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

function setBusy(on) {
  $("stop-btn").hidden = !on;
  $("send-btn").disabled = on;
  $("composer-hint").textContent = on ? "" : "Shift+Enter: new line";
}

async function send() {
  const text = $("input").value.trim();
  if ((!text && !attachments.length) || busy) return;
  messages.push({role: "user", text, images: attachments.filter((a) => a.kind !== "file"),
                 files: attachments.filter((a) => a.kind === "file"), time: Date.now()});
  attachments = [];
  renderAttachments();
  $("input").value = "";
  autosize();
  const m = {role: "assistant", text: "", reasoning: "", time: Date.now()};
  messages.push(m);
  renderChat();
  const el = $("chat").lastElementChild;
  const controller = new AbortController();
  busy = {controller, msg: m};
  setBusy(true);

  const body = {model: health.model, messages: apiMessages(), stream: true,
                reasoning_effort: settings.thinking};
  if (settings.temperature > 0) {
    Object.assign(body, {temperature: +settings.temperature, top_p: +settings.top_p, top_k: +settings.top_k});
  } else {
    body.temperature = 0;
  }
  if (settings.seed) body.seed = +settings.seed;
  if (settings.max) body.max_tokens = +settings.max;
  if (projectionLoaded()) body.experimental_speed_projection = !!settings.esp;
  if (settings.mcp !== false && mcpInfo.tools > 0) body.strata_mcp = true;   // this server may run MCP tools for it

  let firstAt = null, thinkStart = null, usage = null, frame = 0;
  const paint = () => { frame = 0; updateAssistant(el, m, true); scrollDown(); };
  try {
    const r = await fetch("v1/chat/completions", {method: "POST", headers: headers(true), body: JSON.stringify(body),
                                                   signal: controller.signal});
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
      if (r.status === 401) msg = "This server needs an API key: add it under About > Settings.";
      throw new Error(msg);
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
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
        if (j.usage) usage = j.usage;
        if (j.strata_mcp) onTool(m, j.strata_mcp);
        const d = (j.choices && j.choices[0] && j.choices[0].delta) || {};
        const lastTool = m.tools && m.tools.length ? m.tools[m.tools.length - 1] : null;   // a new round after a tool
        if (d.reasoning_content) {
          if (!firstAt) firstAt = performance.now();
          if (!thinkStart) thinkStart = performance.now();
          if (lastTool && m.reasoning && lastTool.rat === m.reasoning.length) m.reasoning += "\n\n";
          m.reasoning += d.reasoning_content;
        }
        if (d.content) {
          if (!firstAt) firstAt = performance.now();
          if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
          if (lastTool && m.text && lastTool.at === m.text.length) m.text += "\n\n";
          m.text += d.content;
        }
        if (!frame) frame = requestAnimationFrame(paint);
      }
    }
  } catch (e) {
    if (e.name === "AbortError") m.stopped = true;
    else { m.error = e.message || String(e); toast("error", "The request failed", m.error, 6000); }
  }
  if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
  const n = usage ? usage.completion_tokens : null;
  if (n && firstAt) {
    const secs = (performance.now() - firstAt) / 1000;
    m.meta = `${fmt(n)} tokens${secs > 0.25 ? ` · ${fmt(n / secs, 1)} tok/s` : ""}${m.stopped ? " · stopped" : ""}` +
             (projectionLoaded() ? (settings.esp ? " · projection on" : " · projection off") : "");
  } else if (m.stopped) {
    m.meta = "Stopped";
  }
  for (const t of m.tools || []) if (t.state === "writing" || t.state === "running") { t.state = "skipped"; t.ms = null; }
  const ran = (m.tools || []).filter((t) => t.state === "done" || t.state === "error").length;
  if (ran) m.meta = `${m.meta ? `${m.meta} · ` : ""}${ran} tool call${ran > 1 ? "s" : ""}`;
  if (m.limit) m.meta = `${m.meta || ""} · stopped at the limit of ${m.limit} tool rounds (mcp.max_rounds)`;
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

$("new-btn").onclick = () => {
  if (busy) { toast("warn", "Still writing", "Stop the answer first."); return; }
  if (!messages.length) return;
  const backup = messages;
  messages = [];
  saveChat();
  renderChat();
  toast("info", "New chat", "The last one was cleared.", 6000, {label: "Undo", run: () => { messages = backup; saveChat(); renderChat(); }});
};
$("export-btn").onclick = () => {
  if (!messages.length) { toast("info", "Nothing to save yet"); return; }
  const tools = (m) => (m.tools || []).filter((t) => t.result != null).map((t) =>
    `<details><summary>Tool ${t.server ? `${t.server} / ` : ""}${t.tool || t.name}${t.ok ? "" : " (error)"}</summary>\n\n` +
    `\`\`\`json\n${JSON.stringify(t.arguments || {}, null, 2)}\n\`\`\`\n\n\`\`\`\n${t.result}\n\`\`\`\n\n</details>\n\n`).join("");
  const md = messages.map((m) => m.role === "user" ? `## You\n\n${m.text}\n` :
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
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("drawer").dataset.open === "true") openDrawer(false); });

// ------------------------------------------------------------------ start
setBusy(false);
renderChat();
const startQuestion = new URLSearchParams(location.search).get("q");   // /?q=... starts a chat (a shortcut)
if (startQuestion) history.replaceState(null, "", location.pathname + location.hash);
loadHealth().then(loadMcp).then(() => { if (startQuestion) { $("input").value = startQuestion; send(); } });
showTab(location.hash.slice(1) || "chat");
poll();

// ------------------------------------------------------------------ images (POST /v1/images/generations)
$("img-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const prompt = $("img-prompt").value.trim();
  if (!prompt) return;
  const body = {prompt, size: $("img-size").value, response_format: "b64_json"};
  if ($("img-seed").value !== "") body.seed = Number($("img-seed").value);
  $("img-go").disabled = true;
  const t0 = performance.now();
  $("img-hint").textContent = "Generating\u2026 (a minute or more on a small GPU)";
  try {
    const r = await fetch("v1/images/generations", {method: "POST", headers: headers(true), body: JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) throw new Error((j.error && j.error.message) || r.statusText);
    for (const d of j.data) {
      const fig = document.createElement("figure");
      const img = document.createElement("img");
      img.src = "data:image/png;base64," + d.b64_json;
      img.alt = prompt;
      const cap = document.createElement("figcaption");
      cap.textContent = `${j.size}, seed ${d.seed}, ${((performance.now() - t0) / 1000).toFixed(0)} s: ${prompt}`;
      fig.append(img, cap);
      $("img-results").prepend(fig);
    }
    $("img-hint").textContent = "Done. Right-click an image to save it.";
  } catch (e) {
    $("img-hint").textContent = "";
    toast("error", "No image", String(e.message || e), 8000);
  } finally {
    $("img-go").disabled = false;
  }
};
