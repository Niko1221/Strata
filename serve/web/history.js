// /metrics-history: the metrics log (--metrics-log) as line charts over time, updated every second.
// It reads GET /metrics-log: the whole log on open, then only the new lines from the last byte offset.
"use strict";
const $ = (id) => document.getElementById(id);

let apiKey = "";
try { apiKey = sessionStorage.getItem("strata.monitor.key") || ""; } catch (e) { /* storage blocked */ }

const S = {
  samples: [], requests: [], offset: 0, started: false, everyS: 1, enabled: null, path: "",
  range: 900, live: true, frozenEnd: null, hover: null, error: null,
};
try {
  const r = localStorage.getItem("strata.history.range");
  if (r !== null && [300, 900, 3600, 21600, 86400].includes(+r)) S.range = +r;
} catch (e) { /* ignore */ }

// ------------------------------------------------------------------ the charts
// one y-axis per chart; two series at most (palette slots 1 and 2); a series' `when` hides it outside a state
const CHARTS = [
  { id: "state", title: "Model state", strip: true, wide: true },
  { id: "decode", title: "Decode speed", unit: "tok/s", digits: 1, wide: true,
    series: [{ key: "decode_tok_s", label: "While generating", slot: 1, when: (s) => s.state === "generating" }],
    points: { key: "decode_tok_s", label: "Each finished request", slot: 2 } },
  { id: "prefill", title: "Prefill speed", unit: "tok/s", digits: 0,
    series: [{ key: "prefill_tok_s", label: "While reading", slot: 1, when: (s) => s.state === "reading" }],
    points: { key: "prefill_tok_s", label: "Each finished request", slot: 2 } },
  { id: "ctx", title: "Context fill", unit: "%", max: 100, digits: 1,
    series: [{ key: "context_pct", label: "Context", slot: 1 }] },
  { id: "load", title: "GPU and CPU load", unit: "%", max: 100, digits: 0,
    series: [{ key: "gpu_util_pct", label: "GPU", slot: 1 }, { key: "cpu_pct", label: "CPU", slot: 2 }] },
  { id: "mem", title: "Memory", unit: "GB", digits: 1,
    series: [{ key: "vram_used_gb", label: "VRAM used", slot: 1 }, { key: "ram_used_gb", label: "System RAM used", slot: 2 }] },
  { id: "power", title: "GPU power", unit: "W", digits: 0,
    series: [{ key: "power_w", label: "Power", slot: 1 }], ref: { key: "power_limit_w", label: "Limit" } },
  { id: "temp", title: "GPU temperature", unit: "°C", digits: 0,
    series: [{ key: "gpu_temp_c", label: "Temperature", slot: 1 }] },
  { id: "pcie", title: "PCIe traffic", unit: "MB/s", digits: 0,
    series: [{ key: "pcie_to_gpu_mb_s", label: "To GPU", slot: 1 }, { key: "pcie_from_gpu_mb_s", label: "From GPU", slot: 2 }] },
  { id: "disk", title: "Disk", unit: "MB/s", digits: 1,
    series: [{ key: "disk_read_mb_s", label: "Read", slot: 1 }, { key: "disk_write_mb_s", label: "Write", slot: 2 }] },
];
const STATES = [["idle", "Idle", "--st-line"], ["reading", "Reading", "--st-info"],
                ["generating", "Generating", "--st-accent"], ["unloaded", "Unloaded", "--st-ink-muted"]];
const PAD = { l: 48, r: 12, t: 8, b: 22 };

function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "style") e.style.cssText = v; else if (k in e) e[k] = v; else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
}
const swatch = (cls, color) => el("i", { className: cls || "", style: color ? `background:${color}` : "" });

function buildCharts() {
  const root = $("charts");
  for (const c of CHARTS) {
    const legend = el("div", { className: "legend" });
    if (c.strip) {
      for (const [, label, tok] of STATES) legend.append(el("span", {}, swatch("", `var(${tok})`), label));
    } else if (c.series.length > 1 || c.points || c.ref) {    // one series alone: the title names it
      for (const s of c.series) legend.append(el("span", {}, swatch("", `var(--series-${s.slot})`), s.label));
      if (c.points) legend.append(el("span", {}, swatch("pt", `var(--series-${c.points.slot})`), c.points.label));
      if (c.ref) legend.append(el("span", {}, swatch("dash"), c.ref.label));
    }
    c.nowEl = el("span", { className: "now" });
    c.canvas = el("canvas", { role: "img", "aria-label": `${c.title} over time` });
    const head = el("div", { className: "card-head" }, el("h2", {}, c.title + (c.unit ? ` (${c.unit})` : "")), legend, c.nowEl);
    root.append(el("div", { className: `card${c.wide ? " wide" : ""}${c.strip ? " strip" : ""}` }, head, c.canvas));
    c.canvas.addEventListener("pointermove", (ev) => hoverAt(c, ev));
    c.canvas.addEventListener("pointerleave", () => { S.hover = null; $("tip").hidden = true; schedule(); });
  }
  new ResizeObserver(schedule).observe(root);
}

// ------------------------------------------------------------------ data
async function api(url) {
  const r = await fetch(url, { cache: "no-store", headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {} });
  if (r.status === 401) { $("auth").hidden = false; throw new Error("This server needs its API key."); }
  if (!r.ok) throw new Error(`The server answered HTTP ${r.status}.`);
  return r.json();
}

function ingest(lines) {
  for (const x of lines) {
    if (x.type === "sample") {
      const last = S.samples[S.samples.length - 1];
      if (!last || x.t > last.t) S.samples.push(x);
    } else if (x.type === "request") {
      S.requests.push(x);
    }
  }
}

async function poll() {
  try {
    let d;
    do {
      // the first read: the last day only (the log runs all the time); then the new lines from the byte offset
      d = await api(S.started ? `/metrics-log?offset=${S.offset}` : `/metrics-log?since=${Date.now() / 1000 - 86400}`);
      S.started = true;
      if (d.reset) { S.samples = []; S.requests = []; }
      S.enabled = d.enabled;
      S.path = d.path || "";
      S.everyS = d.every_s || 1;
      ingest(d.lines);
      S.offset = d.offset;
    } while (d.more);
    S.error = null;
    $("auth").hidden = true;
  } catch (e) {
    S.error = e.message || String(e);
  }
  renderStatus();
  renderTiles();
  renderTable();
  schedule();
  setTimeout(poll, S.error ? 3000 : 1000);
}

// ------------------------------------------------------------------ helpers
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (v, d = 0) => (v == null || Number.isNaN(v) ? "–" :
  Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }));
const clock = (t, secs = true) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit",
  second: secs ? "2-digit" : undefined, hour12: false });
const lastSample = () => S.samples[S.samples.length - 1];

function lowerBound(arr, t) {          // first index with arr[i].t >= t
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m].t < t) lo = m + 1; else hi = m; }
  return lo;
}

function view() {
  const now = Date.now() / 1000;
  const end = S.live ? Math.max(now, lastSample()?.t || 0) : (S.frozenEnd ?? now);
  let start = S.range ? end - S.range : (S.samples[0]?.t ?? end - 60);
  if (end - start < 10) start = end - 10;
  return [start, end];
}

function niceStep(span, n) {
  const raw = span / n, p = 10 ** Math.floor(Math.log10(raw)), f = raw / p;
  return p * (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10);
}
const TIME_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400];

// ------------------------------------------------------------------ drawing
let pending = false;
function schedule() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; for (const c of CHARTS) draw(c); });
}

function setup(c) {
  const cv = c.canvas, dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  return { g, W, H };
}

function draw(c) {
  const { g, W, H } = setup(c);
  if (W < 50) return;
  const [t0, t1] = view();
  const pw = W - PAD.l - PAD.r;
  const X = (t) => PAD.l + ((t - t0) / (t1 - t0)) * pw;
  const i0 = Math.max(0, lowerBound(S.samples, t0) - 1), i1 = Math.min(S.samples.length, lowerBound(S.samples, t1) + 1);
  const maxGap = Math.max(3, S.everyS * 3 + 1);       // a longer silence (server stopped) breaks the line
  const muted = css("--st-ink-muted"), soft = css("--st-line-soft"), surface = css("--st-surface");
  g.font = `11px ${css("--st-font") || "system-ui"}`;

  if (c.strip) return drawStrip(c, g, H, X, i0, i1, maxGap);

  const ph = H - PAD.t - PAD.b;
  // the y range: 0 to the visible maximum (or the chart's fixed maximum)
  let ymax = 0;
  for (let i = i0; i < i1; i++) {
    const s = S.samples[i];
    for (const se of c.series) if (s[se.key] != null && (!se.when || se.when(s))) ymax = Math.max(ymax, s[se.key]);
    if (c.ref && s[c.ref.key] != null) ymax = Math.max(ymax, s[c.ref.key]);
  }
  const reqs = c.points ? S.requests.filter((r) => r.t >= t0 && r.t <= t1 && r[c.points.key] != null) : [];
  for (const r of reqs) ymax = Math.max(ymax, r[c.points.key]);
  if (c.max) ymax = c.max;
  const step = niceStep(ymax || 1, 3);
  ymax = Math.max(step, Math.ceil((ymax || 1) / step) * step);
  const Y = (v) => PAD.t + ph - (v / ymax) * ph;

  // grid and y labels (recessive)
  g.strokeStyle = soft; g.fillStyle = muted; g.lineWidth = 1; g.textAlign = "right"; g.textBaseline = "middle";
  for (let v = 0; v <= ymax + step / 2; v += step) {
    const y = Math.round(Y(v)) + 0.5;
    g.beginPath(); g.moveTo(PAD.l, y); g.lineTo(W - PAD.r, y); g.stroke();
    g.fillText(fmt(v, step < 1 ? 1 : 0), PAD.l - 6, y);
  }
  drawTimeAxis(g, H, X, t0, t1, pw, muted);

  // the reference line (power limit)
  if (c.ref) {
    const s = [...S.samples.slice(i0, i1)].reverse().find((x) => x[c.ref.key] != null);
    if (s) {
      g.save(); g.setLineDash([5, 4]); g.strokeStyle = css("--ref"); g.lineWidth = 1.5;
      const y = Math.round(Y(s[c.ref.key])) + 0.5;
      g.beginPath(); g.moveTo(PAD.l, y); g.lineTo(W - PAD.r, y); g.stroke(); g.restore();
    }
  }

  // the lines: one column per pixel when there are more samples than pixels (first, min, max, last of each)
  g.save();
  g.beginPath(); g.rect(PAD.l, 0, pw, H); g.clip();
  for (const se of c.series) {
    const color = css(`--series-${se.slot}`);
    g.strokeStyle = color; g.fillStyle = color; g.lineWidth = 2; g.lineJoin = "round"; g.lineCap = "round";
    g.beginPath();
    let pen = false, b = null, buckets = 0, prev = null;
    const lone = [];
    const flush = () => {
      if (!b) return;
      const px = b.px;
      if (!pen) { g.moveTo(px, Y(b.first)); pen = true; } else g.lineTo(px, Y(b.first));
      if (b.n > 1) { g.lineTo(px, Y(b.min)); g.lineTo(px, Y(b.max)); g.lineTo(px, Y(b.last)); }
      buckets++; b = null;
    };
    const breakLine = () => {
      if (b && buckets === 0 && b.n === 1) lone.push([b.px, Y(b.first)]);   // a single point: drawn as a dot
      flush(); pen = false; buckets = 0;
    };
    for (let i = i0; i < i1; i++) {
      const s = S.samples[i];
      const v = (!se.when || se.when(s)) ? s[se.key] : null;
      if (v == null || (prev && s.t - prev.t > maxGap)) breakLine();
      prev = s;
      if (v == null) continue;
      const px = Math.round(X(s.t));
      if (b && b.px === px) { b.n++; b.min = Math.min(b.min, v); b.max = Math.max(b.max, v); b.last = v; }
      else { flush(); b = { px, first: v, last: v, min: v, max: v, n: 1 }; }
    }
    breakLine();
    g.stroke();
    for (const [x, y] of lone) { g.beginPath(); g.arc(x, y, 2, 0, 2 * Math.PI); g.fill(); }
  }
  // finished requests: dots with a surface ring
  if (c.points) {
    g.fillStyle = css(`--series-${c.points.slot}`); g.strokeStyle = surface; g.lineWidth = 2;
    for (const r of reqs) {
      g.beginPath(); g.arc(X(r.t), Y(r[c.points.key]), 4, 0, 2 * Math.PI); g.stroke(); g.fill();
    }
  }
  g.restore();

  // the latest value in the card's head
  const last = lastSample();
  const se0 = c.series[0];
  if (c.points && last && !(se0.when && se0.when(last))) {
    const r = S.requests[S.requests.length - 1];
    c.nowEl.textContent = r && r[c.points.key] != null ? `last request ${fmt(r[c.points.key], c.digits)} ${c.unit}` : "";
  } else {
    c.nowEl.textContent = last ? c.series.map((s) => `${c.series.length > 1 ? s.label + " " : "now "}${fmt(last[s.key], c.digits)}`)
      .join(" · ") + ` ${c.unit}` : "";
  }

  // the crosshair: the sample nearest the pointer, in every chart at once
  if (S.hover) {
    const s = nearest(S.hover.t);
    if (s && s.t >= t0 && s.t <= t1) {
      const x = Math.round(X(s.t)) + 0.5;
      g.strokeStyle = muted; g.lineWidth = 1;
      g.beginPath(); g.moveTo(x, PAD.t); g.lineTo(x, H - PAD.b); g.stroke();
      for (const se of c.series) {
        const v = (!se.when || se.when(s)) ? s[se.key] : null;
        if (v == null) continue;
        g.fillStyle = css(`--series-${se.slot}`); g.strokeStyle = surface; g.lineWidth = 2;
        g.beginPath(); g.arc(x, Y(v), 4.5, 0, 2 * Math.PI); g.stroke(); g.fill();
      }
    }
  }
}

function drawTimeAxis(g, H, X, t0, t1, pw, muted) {
  const step = TIME_STEPS.find((s) => (t1 - t0) / s <= pw / 90) || 86400;
  const tz = -new Date(t0 * 1000).getTimezoneOffset() * 60;
  g.fillStyle = muted; g.textAlign = "center"; g.textBaseline = "alphabetic";
  for (let t = Math.ceil((t0 + tz) / step) * step - tz; t <= t1; t += step) {
    g.fillText(clock(t, step < 60), X(t), H - 6);
  }
}

function drawStrip(c, g, H, X, i0, i1, maxGap) {
  const colors = Object.fromEntries(STATES.map(([k, , tok]) => [k, css(tok)]));
  const top = 6, h = H - 12;
  let run = null;
  const end = (tEnd) => {
    if (!run) return;
    const x0 = Math.max(PAD.l, X(run.t)), x1 = Math.min(c.canvas.clientWidth - PAD.r, X(tEnd));
    if (x1 > x0) { g.fillStyle = colors[run.state] || colors.idle; g.fillRect(x0, top, Math.max(1, x1 - x0), h); }
    run = null;
  };
  let prev = null;
  for (let i = i0; i < i1; i++) {
    const s = S.samples[i];
    if (prev && s.t - prev.t > maxGap) end(prev.t + S.everyS);
    if (run && run.state !== s.state) end(s.t);
    if (!run) run = { t: s.t, state: s.state };
    prev = s;
  }
  if (prev) end(prev.t + S.everyS);
  const last = lastSample();
  c.nowEl.textContent = last ? STATES.find(([k]) => k === last.state)?.[1] || last.state : "";
  if (S.hover) {
    const s = nearest(S.hover.t);
    if (s) {
      const x = Math.round(X(s.t)) + 0.5;
      g.strokeStyle = css("--st-ink"); g.lineWidth = 1;
      g.beginPath(); g.moveTo(x, 2); g.lineTo(x, H - 2); g.stroke();
    }
  }
}

function nearest(t) {
  const a = S.samples;
  if (!a.length) return null;
  const i = lowerBound(a, t);
  const cand = [a[i - 1], a[i]].filter(Boolean).sort((p, q) => Math.abs(p.t - t) - Math.abs(q.t - t));
  return cand[0];
}

// ------------------------------------------------------------------ hover tooltip
function hoverAt(c, ev) {
  const r = c.canvas.getBoundingClientRect();
  const [t0, t1] = view();
  const x = ev.clientX - r.left;
  if (x < PAD.l || x > r.width - PAD.r) { S.hover = null; $("tip").hidden = true; schedule(); return; }
  const t = t0 + ((x - PAD.l) / (r.width - PAD.l - PAD.r)) * (t1 - t0);
  S.hover = { t };
  const s = nearest(t);
  const tip = $("tip");
  if (!s) { tip.hidden = true; schedule(); return; }
  const rows = [];
  const row = (color, label, value) => el("div", { className: "row" },
    el("span", {}, color ? swatch("", color) : "", label), el("span", {}, value));
  if (c.strip) {
    rows.push(row(`var(${(STATES.find(([k]) => k === s.state) || STATES[0])[2]})`, "State",
      STATES.find(([k]) => k === s.state)?.[1] || s.state));
    if (s.queued) rows.push(row(null, "Queued", fmt(s.queued)));
    if (s.state === "reading" && s.prompt_total) rows.push(row(null, "Prompt read", `${fmt(s.prompt_read)} / ${fmt(s.prompt_total)}`));
    if (s.state === "generating") rows.push(row(null, "Generated", fmt(s.generated)));
  } else {
    for (const se of c.series) {
      const v = (!se.when || se.when(s)) ? s[se.key] : null;
      rows.push(row(`var(--series-${se.slot})`, se.label, v == null ? "–" : `${fmt(v, c.digits)} ${c.unit}`));
    }
    if (c.ref && s[c.ref.key] != null) rows.push(row(null, c.ref.label, `${fmt(s[c.ref.key], 0)} ${c.unit}`));
    if (c.points) {     // a finished request within 6 pixels of the pointer
      const win = ((t1 - t0) / (r.width - PAD.l - PAD.r)) * 6;
      const q = S.requests.filter((q) => Math.abs(q.t - t) <= win && q[c.points.key] != null).pop();
      if (q) rows.push(row(`var(--series-${c.points.slot})`, "Request", `${fmt(q[c.points.key], c.digits)} ${c.unit}`));
    }
  }
  tip.replaceChildren(el("div", { className: "t" }, new Date(s.t * 1000).toLocaleString([], { hour12: false })), ...rows);
  tip.hidden = false;
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let left = ev.clientX + 14, top = ev.clientY + 14;
  if (left + tw > window.innerWidth - 8) left = ev.clientX - tw - 14;
  if (top + th > window.innerHeight - 8) top = ev.clientY - th - 14;
  tip.style.left = `${Math.max(8, left)}px`; tip.style.top = `${Math.max(8, top)}px`;
  schedule();
}

// ------------------------------------------------------------------ tiles, status, table
function setTile(id, value, unit, sub) {
  const v = $(`v-${id}`);
  v.replaceChildren(value == null ? "–" : value, unit && value != null ? el("small", {}, unit) : "");
  $(`s-${id}`).textContent = sub || "";
}

function renderTiles() {
  const s = lastSample();
  if (!s) return;
  const lastReq = S.requests[S.requests.length - 1];
  const label = STATES.find(([k]) => k === s.state)?.[1] || s.state;
  setTile("state", label, "", s.state === "reading" && s.prompt_total ? `prompt ${fmt(s.prompt_read)} / ${fmt(s.prompt_total)}`
    : s.state === "generating" ? `${fmt(s.generated)} tokens · ${fmt(s.elapsed_s, 1)} s` : s.queued ? `${s.queued} queued` : "");
  const gen = s.state === "generating", rd = s.state === "reading";
  setTile("decode", s.decode_tok_s == null ? null : fmt(s.decode_tok_s, 1), "t/s", gen ? "now" : lastReq ? "last request" : "");
  setTile("prefill", s.prefill_tok_s == null ? null : fmt(s.prefill_tok_s), "t/s", rd ? "now" : s.state === "idle" ? "last request" : "this request");
  setTile("gpu", s.gpu_util_pct == null ? null : fmt(s.gpu_util_pct), "%", s.cpu_pct == null ? "" : `CPU ${fmt(s.cpu_pct)}%`);
  setTile("vram", s.vram_used_gb == null ? null : fmt(s.vram_used_gb, 1), s.vram_total_gb ? `/ ${fmt(s.vram_total_gb)} GB` : "GB",
    s.ram_total_gb ? `RAM ${fmt(s.ram_used_gb, 1)} / ${fmt(s.ram_total_gb)} GB` : "");
  setTile("power", s.power_w == null ? null : fmt(s.power_w), "W", s.power_limit_w ? `of ${fmt(s.power_limit_w)} W limit` : "");
  setTile("temp", s.gpu_temp_c == null ? null : fmt(s.gpu_temp_c), "°C",
    s.pcie_gen_max || s.pcie_gen ? `PCIe Gen${s.pcie_gen_max || s.pcie_gen} x${s.pcie_width ?? "?"}` : "");
  setTile("ctx", s.context_pct == null ? null : fmt(s.context_pct, 0), "%",
    s.context_max ? `${fmt(s.context_used)} / ${fmt(s.context_max)}` : "");
}

function renderStatus() {
  $("off").hidden = S.enabled !== false;
  $("error").hidden = !S.error;
  $("error").textContent = S.error ? `${S.error} Retrying…` : "";
  const s = lastSample();
  const fresh = s && Date.now() / 1000 - s.t < Math.max(5, S.everyS * 4);
  $("dot").classList.toggle("on", !!fresh && !S.error);
  $("state-text").textContent = S.error ? "Not connected" : !s ? (S.enabled ? "Waiting for the first sample…" : "")
    : fresh ? `Live · last sample ${clock(s.t)}` : `No new samples since ${new Date(s.t * 1000).toLocaleString([], { hour12: false })}`;
  $("count").textContent = s ? `· ${fmt(S.samples.length)} samples · ${fmt(S.requests.length)} requests · every ${S.everyS} s` : "";
  $("path").textContent = S.path ? `· ${S.path}` : "";
}

function renderTable() {
  const [t0, t1] = view();
  const rows = S.requests.filter((r) => r.t >= t0 && r.t <= t1).slice(-200).reverse();
  $("req-count").textContent = rows.length ? `(${fmt(rows.length)} in this range)` : "";
  $("req-empty").hidden = rows.length > 0;
  const pct = (v) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);
  $("req-body").replaceChildren(...rows.map((r) => el("tr", {},
    el("td", {}, clock(r.t)), el("td", {}, fmt(r.prompt_tokens)), el("td", {}, fmt(r.reused)), el("td", {}, fmt(r.output_tokens)),
    el("td", {}, fmt(r.decode_tok_s, 1)), el("td", {}, fmt(r.prefill_tok_s)), el("td", {}, pct(r.hit_rate)),
    el("td", {}, pct(r.pcie_share)), el("td", {}, r.duration_s == null ? "–" : `${fmt(r.duration_s, 1)} s`),
    el("td", {}, r.finish || "–"))));
}

// ------------------------------------------------------------------ controls
function setRange(sec) {
  S.range = sec;
  try { localStorage.setItem("strata.history.range", String(sec)); } catch (e) { /* ignore */ }
  for (const b of $("ranges").querySelectorAll("button")) b.setAttribute("aria-pressed", String(+b.dataset.s === sec));
  renderTable(); schedule();
}
$("ranges").addEventListener("click", (ev) => { const b = ev.target.closest("button"); if (b) setRange(+b.dataset.s); });
$("live").onclick = () => {
  if (S.live) S.frozenEnd = view()[1];
  S.live = !S.live;
  $("live").setAttribute("aria-pressed", String(S.live));
  $("live").textContent = S.live ? "Live" : "Paused";
  schedule();
};
$("theme").onclick = () => {
  const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("strata.theme", t); } catch (e) { /* ignore */ }
  schedule();
};
$("auth").onsubmit = (ev) => {
  ev.preventDefault();
  apiKey = $("api-key").value.trim();
  try { sessionStorage.setItem("strata.monitor.key", apiKey); } catch (e) { /* ignore */ }
  $("auth").hidden = true;
};

buildCharts();
setRange(S.range);
poll();
setInterval(() => { if (S.live) { renderStatus(); schedule(); } }, 1000);   // the window moves on between samples
