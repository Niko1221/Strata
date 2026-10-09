// serve/web/app.js - the Strata web app: Chat, Monitor, About. No framework, no network beyond this server.
// The Monitor tab rebuilds PR #22's dashboard idea (code-martin) on the server's own /metrics.
"use strict";

const $ = (id) => document.getElementById(id);
const SPRITE = "web/sprite.svg";
const icon = (name, cls = "st-icon") => name === "forge"
  ? `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor"/><rect x="9" y="4" width="1.5" height="3" rx="0.5" fill="currentColor"/><rect x="13.5" y="4" width="1.5" height="3" rx="0.5" fill="currentColor"/><rect x="9" y="17" width="1.5" height="3" rx="0.5" fill="currentColor"/><rect x="13.5" y="17" width="1.5" height="3" rx="0.5" fill="currentColor"/><rect x="4" y="9" width="3" height="1.5" rx="0.5" fill="currentColor"/><rect x="4" y="13.5" width="3" height="1.5" rx="0.5" fill="currentColor"/><rect x="17" y="9" width="3" height="1.5" rx="0.5" fill="currentColor"/><rect x="17" y="13.5" width="3" height="1.5" rx="0.5" fill="currentColor"/></svg>`
  : name === "close" ? `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`
  : `<svg class="${cls}" aria-hidden="true"><use href="${SPRITE}#i-${name}"/></svg>`;
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
  tab = ["chat", "monitor", "about"].includes(name) ? name : "chat";
  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of ["chat", "monitor", "about"]) $(`view-${v}`).hidden = v !== tab;
  if (location.hash.slice(1) !== tab) history.replaceState(null, "", tab === "chat" ? location.pathname : `#${tab}`);
  if (tab === "chat") $("input").focus();
  if (tab === "monitor") loadMcp();
  if (tab === "about") loadConfig();
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
$("api-key").onchange = () => { store.set("apikey", $("api-key").value.trim()); toast("success", "API key saved", "Kept in this browser only."); };

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
  {key: "cpu", label: "CPU load", icon: "cpu", unit: "%", series: "cpu", max: 100},
  {key: "disk", label: "Disk read", icon: "disk", unit: "MB/s", series: "disk_read_mb", tone: "info"},
];
const metricMarkup = Object.fromEntries(METRICS.map((m) => [m.key, `
  <div class="st-card metric-card" id="${m.key}" data-card-key="${m.key}"><button class="metric-card__hide st-btn st-btn--icon" data-hide-card="${m.key}" aria-label="Hide ${esc(m.label)}" title="Hide ${esc(m.label)}">${icon("close", "st-icon st-icon--sm")}</button><div class="st-metric">
    <span class="st-metric__label" id="ml-${m.key}">${icon(m.icon, "st-icon st-icon--sm")}${esc(m.label)}</span>
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
  </div></div>`]));

const STAT_CARDS = [
  {key: "speculation", label: "Speculation", icon: "bolt", spark: true},
  {key: "reuse", label: "Prompt reuse", icon: "layers", spark: true},
  {key: "prefill-time", label: "Prefill time", icon: "clock", spark: true},
  {key: "vram-free", label: "VRAM free", icon: "layers", spark: true},
  {key: "cpu-temp", label: "CPU temp", icon: "thermometer", spark: true},
  {key: "all-power", label: "All GPU power", icon: "bolt", spark: true},
  {key: "session", label: "Session", icon: "clock", spark: false},
  {key: "forge", label: "Forge", icon: "forge", spark: false},
];
const statMarkup = Object.fromEntries(STAT_CARDS.map((card) => [card.key, `
  <div class="st-card metric-card stats-card" id="${card.key}" data-card-key="${card.key}"${["all-power", "cpu-temp", "forge"].includes(card.key) ? " hidden" : ""}>
    <button class="metric-card__hide st-btn st-btn--icon" data-hide-card="${card.key}" aria-label="Hide ${esc(card.label)}" title="Hide ${esc(card.label)}">${icon("close", "st-icon st-icon--sm")}</button>
    <div class="st-metric">
      <span class="st-metric__label" id="sl-${card.key}">
        ${icon(card.icon, "st-icon st-icon--sm")}<span id="slt-${card.key}">${esc(card.label)}</span></span>
      <span class="st-metric__value" id="sv-${card.key}">–</span>
      <span class="st-metric__sub" id="ss-${card.key}"></span>
      ${card.key === "vram-free" ? `<div class="st-progress stats-vram-bar" id="vram-free-progress">
        <div class="st-progress__bar" id="vram-free-bar" style="width:0%"></div></div>` : ""}
      ${card.spark ? `<svg class="st-metric__spark" id="sp-${card.key}" viewBox="0 0 100 32"
        preserveAspectRatio="none">
        <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
        stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"
        vector-effect="non-scaling-stroke"/></svg>` : ""}
      ${["reuse", "prefill-time"].includes(card.key)
        ? `<div class="stats-graph-note" id="gn-${card.key}" hidden>graph after 2 requests</div>` : ""}
    </div>
  </div>`]));

// One grid for every Monitor card, so a card can be dragged to any position; the rows come from wrapping.
const METRIC_CARDS = ["speed", "prefill-time", "reuse", "speculation", "gpu", "vram", "vram-free", "temp",
  "power", "all-power", "pcie", "cpu-temp", "cpu", "disk", "session", "forge"];
const metricGrid = $("metric-cards");
metricGrid.innerHTML = METRIC_CARDS.map((key) => metricMarkup[key] || statMarkup[key]).join("");
const CARD_ORDER_KEY = "monitor.cardOrder";
// Cards hidden with the x stay hidden across reloads; Reset layout brings them back.
const HIDDEN_CARDS_KEY = "monitor.hiddenCards";
const savedHidden = store.get(HIDDEN_CARDS_KEY, []);
const userHiddenCards = new Set(Array.isArray(savedHidden) ? savedHidden.filter((key) => METRIC_CARDS.includes(key)) : []);
for (const key of userHiddenCards) $(key).hidden = true;
const cardKeys = () => [...metricGrid.children].map((card) => card.dataset.cardKey);
function updateResetCardOrder() {
  const hidden = userHiddenCards.size;
  const button = $("reset-card-order");
  button.textContent = hidden ? `Reset layout · ${hidden} icon${hidden === 1 ? "" : "s"} hidden` : "Reset layout";
  button.parentElement.hidden = !hidden && cardKeys().join("\0") === METRIC_CARDS.join("\0");
}
function applySavedCardOrder() {
  const saved = store.get(CARD_ORDER_KEY, null);
  const known = Array.isArray(saved)
    ? saved.filter((key, index, list) => METRIC_CARDS.includes(key) && list.indexOf(key) === index) : [];
  const cards = new Map([...metricGrid.children].map((card) => [card.dataset.cardKey, card]));
  for (const key of [...known, ...METRIC_CARDS.filter((key) => !known.includes(key))]) metricGrid.appendChild(cards.get(key));
  updateResetCardOrder();
}
function saveCardOrder() {
  store.set(CARD_ORDER_KEY, cardKeys());
  updateResetCardOrder();
}
applySavedCardOrder();
$("reset-card-order").addEventListener("click", () => {
  try {
    localStorage.removeItem("strata." + CARD_ORDER_KEY);
    localStorage.removeItem("strata." + HIDDEN_CARDS_KEY);
  } catch (e) { /* storage unavailable */ }
  userHiddenCards.clear();
  for (const key of METRIC_CARDS) {
    const card = metricGrid.querySelector(`[data-card-key="${key}"]`);
    metricGrid.appendChild(card);
    card.hidden = card.dataset.metricDataHidden === "true";
  }
  updateResetCardOrder();
});
const cardPositionLive = $("card-position-live");
let cardDrag = null;
let pendingCardDrag = null;
function startCardDrag(card, event) {
  if (cardDrag) return;
  const rect = card.getBoundingClientRect();
  const placeholder = document.createElement("div");
  placeholder.className = "metric-card-placeholder";
  placeholder.style.width = `${rect.width}px`;
  placeholder.style.height = `${rect.height}px`;
  card.parentNode.insertBefore(placeholder, card);
  cardDrag = {card, placeholder, grid: card.parentNode,
    next: placeholder.nextSibling, x: event.clientX, y: event.clientY, style: card.getAttribute("style")};
  card.classList.add("is-dragging");
  card.style.position = "fixed";
  card.style.left = `${rect.left}px`;
  card.style.top = `${rect.top}px`;
  card.style.width = `${rect.width}px`;
  card.style.height = `${rect.height}px`;
  card.style.zIndex = "10";
  card.setPointerCapture(event.pointerId);
}
function placeCardDrag(event) {
  if (!cardDrag) return;
  const {card, placeholder, grid, x, y} = cardDrag;
  card.style.transform = `translate3d(${event.clientX - x}px, ${event.clientY - y}px, 0)`;
  const target = document.elementFromPoint(event.clientX, event.clientY)?.closest(".metric-card");
  if (!target || target === card || target.parentNode !== grid || target.hidden) return;
  const rect = target.getBoundingClientRect();
  const before = event.clientY < rect.top + rect.height / 2 ||
    (event.clientY <= rect.bottom && event.clientX < rect.left + rect.width / 2);
  grid.insertBefore(placeholder, before ? target : target.nextSibling);
}
function finishCardDrag(event, cancel = false) {
  if (!cardDrag) return;
  const drag = cardDrag;
  cardDrag = null;
  // Released outside the card area (over the state card, say): the card goes back where it was.
  const valid = !cancel && Boolean(document.elementFromPoint(event.clientX, event.clientY)?.closest("#metrics"));
  drag.card.classList.remove("is-dragging");
  if (drag.style === null) drag.card.removeAttribute("style");
  else drag.card.setAttribute("style", drag.style);
  if (valid) {
    drag.grid.insertBefore(drag.card, drag.placeholder);
    saveCardOrder();
  } else if (drag.next?.parentNode === drag.grid) drag.grid.insertBefore(drag.card, drag.next);
  else drag.grid.appendChild(drag.card);
  drag.placeholder.remove();
}
document.addEventListener("pointerdown", (event) => {
  if (event.button !== 0 || !(event.target instanceof Element)) return;
  const card = event.target.closest(".metric-card[data-card-key]");
  if (!card || card.hidden || event.target.closest("button,a,select,input,textarea,[contenteditable='true']")) return;
  const touch = event.pointerType === "touch";
  const pending = {card, event, x: event.clientX, y: event.clientY, timer: null, started: false};
  pendingCardDrag = pending;
  if (touch) pending.timer = setTimeout(() => {
    pending.started = true;
    startCardDrag(card, event);
  }, 250);
});
document.addEventListener("pointermove", (event) => {
  const pending = pendingCardDrag;
  if (pending && !pending.started && Math.hypot(event.clientX - pending.x, event.clientY - pending.y) >= 6) {
    clearTimeout(pending.timer);
    if (event.pointerType !== "touch") { pending.started = true; startCardDrag(pending.card, event); }
    else pendingCardDrag = null;
  }
  if (cardDrag) placeCardDrag(event);
});
document.addEventListener("pointerup", (event) => {
  if (pendingCardDrag) { clearTimeout(pendingCardDrag.timer); pendingCardDrag = null; }
  if (cardDrag) finishCardDrag(event);
});
// Touch: once the press-and-hold has started a drag, the finger must move the card, not scroll the page (the
// browser would take the gesture for a pan and cancel the pointer). Needs a non-passive listener.
document.addEventListener("touchmove", (event) => { if (cardDrag) event.preventDefault(); }, {passive: false});
document.addEventListener("contextmenu", (event) => { if (cardDrag || pendingCardDrag?.started) event.preventDefault(); });
document.addEventListener("pointercancel", (event) => {
  if (pendingCardDrag) { clearTimeout(pendingCardDrag.timer); pendingCardDrag = null; }
  if (cardDrag) finishCardDrag(event, true);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && cardDrag) { finishCardDrag(event, true); return; }
  if (!event.altKey || !["ArrowLeft", "ArrowRight"].includes(event.key) || !(event.target instanceof Element)) return;
  if (!event.target.closest("[data-hide-card]")) return;
  const card = event.target.closest(".metric-card[data-card-key]");
  if (!card) return;
  const cards = [...card.parentNode.children].filter((candidate) => !candidate.hidden);
  const index = cards.indexOf(card), next = index + (event.key === "ArrowLeft" ? -1 : 1);
  if (next < 0 || next >= cards.length) return;
  event.preventDefault();
  card.parentNode.insertBefore(card, event.key === "ArrowLeft" ? cards[next] : cards[next].nextSibling);
  saveCardOrder();
  const label = card.querySelector(".st-metric__label, [id^='slt-']")?.textContent.trim() || card.dataset.cardKey;
  cardPositionLive.textContent = `Moved ${label} to position ${next + 1} of ${cards.length}`;
});
function setCardHidden(card, dataHidden) {
  card.dataset.metricDataHidden = String(Boolean(dataHidden));
  card.hidden = Boolean(dataHidden) || userHiddenCards.has(card.dataset.cardKey || card.id);
}
document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-hide-card]");
  if (!button) return;
  const card = $(button.dataset.hideCard);
  if (!card) return;
  const order = [...document.querySelectorAll(".metric-card")];
  const index = order.indexOf(card);
  if (!("metricDataHidden" in card.dataset)) card.dataset.metricDataHidden = String(card.hidden);
  userHiddenCards.add(button.dataset.hideCard);
  store.set(HIDDEN_CARDS_KEY, [...userHiddenCards]);
  card.hidden = true;
  updateResetCardOrder();
  const next = order.slice(index + 1).find((candidate) => !candidate.hidden)
    || order.slice(0, index).reverse().find((candidate) => !candidate.hidden);
  if (next) next.querySelector("[data-hide-card]").focus();
  else document.querySelector('.st-tab[data-tab="monitor"]')?.focus();
});

// `fit`: scale to the series' own range (at least 10 units) instead of from 0, so a temperature moving between
// 25 and 30 degrees is visible rather than a flat line near the top.
function spark(id, values, max, fit) {
  const svg = $(id);
  const v = (values || []).map((x) => (x == null ? 0 : x));
  if (v.length < 2) { svg.querySelector(".line").setAttribute("d", ""); svg.querySelector(".area").setAttribute("d", ""); return; }
  let bottom = 0, top = Math.max(max || 0, ...v, 1e-9);
  const known = fit ? (values || []).filter((x) => x != null) : [];
  if (known.length) {
    bottom = Math.max(0, Math.min(...known) - 2);
    top = Math.max(Math.max(...known) + 2, bottom + 10);
  }
  const pts = v.map((x, i) => [(i / (v.length - 1)) * 100, 30 - (Math.max(0, x - bottom) / (top - bottom)) * 26]);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");
  svg.querySelector(".line").setAttribute("d", line);
  svg.querySelector(".area").setAttribute("d", `${line}L100,32L0,32Z`);
}
function setMetric(key, value, unit, sub, options = {}) {
  $(`mv-${key}`).innerHTML = value == null ? "–" : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
  $(`ms-${key}`).textContent = sub || "";
  $(`ms-${key}`).title = "";
  const card = $(key);   // "prefill" is a value inside the Speed card, not a card of its own
  if (card) setCardHidden(card, options.hidden);
}

let gpuSelection = String(store.get("monitor.gpu", "all"));
let gpuSelectorSignature = "";
function updateGpuSelector(hw, st) {
  const engineCards = (Array.isArray(hw.gpus) ? hw.gpus : []).map((g) => ({...g, inModel: true}));
  const otherCards = (Array.isArray(hw.other_gpus) ? hw.other_gpus : []).map((g) => ({...g, inModel: false}));
  const cards = [...engineCards, ...otherCards];
  const indexes = cards.map((g) => String(g.index));
  if (gpuSelection !== "all" && !indexes.includes(gpuSelection)) {
    gpuSelection = "all";
    store.set("monitor.gpu", gpuSelection);
  }
  const multiple = cards.length > 1;
  const buttonsMode = cards.length <= 6;
  const seg = $("gpu-seg"), selectWrap = $("gpu-select-wrap"), select = $("gpu-select"), selector = $("gpu-selector");
  selector.hidden = !multiple;
  $("gpu-seg-label").hidden = !multiple || !buttonsMode;
  seg.hidden = !multiple || !buttonsMode;
  selectWrap.hidden = !multiple || buttonsMode;
  const splitNames = typeof st.gpu_name === "string" ? st.gpu_name.split(" + ") : [];
  const engineNames = splitNames.length === engineCards.length ? splitNames : [];
  const otherNames = Array.isArray(st.other_gpu_names) ? st.other_gpu_names : [];
  const names = [...engineCards.map((_, i) => engineNames[i] || ""),
    ...otherCards.map((_, i) => otherNames[i] || "")];
  const signature = JSON.stringify([buttonsMode, indexes, names, engineCards.length]);
  if (signature !== gpuSelectorSignature) {
    gpuSelectorSignature = signature;
    const options = [{value: "all", label: "All", card: null},
      ...cards.map((card, i) => ({value: String(card.index), label: `GPU ${card.index}`, card, name: names[i]}))];
    const optionTitle = (option) => !option.card ? "" : option.card.inModel ? option.name || ""
      : `${option.label}${option.name ? ` (${option.name})` : ""} is not in use by the model`;
    if (buttonsMode) {
      seg.replaceChildren();
      for (const option of options) {
        const button = document.createElement("button");
        button.type = "button";
        button.setAttribute("role", "radio");
        button.dataset.value = option.value;
        button.appendChild(document.createTextNode(option.label));
        if (option.card && !option.card.inModel) {
          const marker = document.createElement("span");
          marker.className = "gpu-seg__other";
          marker.setAttribute("aria-hidden", "true");
          marker.textContent = " (not in use)";
          button.appendChild(marker);
        }
        button.title = optionTitle(option);
        seg.appendChild(button);
      }
    } else {
      select.replaceChildren();
      for (const option of options) {
        const el = document.createElement("option");
        el.value = option.value;
        el.textContent = `${option.label}${option.card && !option.card.inModel ? " (not in use)" : ""}`;
        el.title = optionTitle(option);
        select.appendChild(el);
      }
    }
  }
  for (const button of seg.querySelectorAll('[role="radio"]')) {
    button.setAttribute("aria-checked", String(button.dataset.value === gpuSelection));
  }
  select.value = gpuSelection;
  return {cards, engineCards, otherCards, indexes, names};
}
function setGpuSelection(value, refocus = false) {
  gpuSelection = value;
  store.set("monitor.gpu", gpuSelection);
  if (lastMetrics) render(lastMetrics);
  if (refocus) {
    const button = [...$("gpu-seg").querySelectorAll('[role="radio"]')]
      .find((el) => el.dataset.value === gpuSelection);
    if (button) button.focus();
  }
}
$("gpu-seg").addEventListener("click", (event) => {
  const button = event.target.closest('[role="radio"]');
  if (button) setGpuSelection(button.dataset.value, true);
});
$("gpu-seg").addEventListener("keydown", (event) => {
  if (!["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp"].includes(event.key)) return;
  const buttons = [...$("gpu-seg").querySelectorAll('[role="radio"]')];
  if (!buttons.length) return;
  event.preventDefault();
  const current = buttons.findIndex((button) => button === document.activeElement);
  const from = current < 0 ? Math.max(0, buttons.findIndex((button) => button.dataset.value === gpuSelection)) : current;
  const step = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : -1;
  const next = buttons[(from + step + buttons.length) % buttons.length];
  setGpuSelection(next.dataset.value, true);
});
$("gpu-select").addEventListener("change", (event) => setGpuSelection(event.target.value));

function gpuMetricData(hw, h, selector) {
  const cards = selector.cards;
  const selected = cards.length > 1 && gpuSelection !== "all"
    ? cards.find((g) => String(g.index) === gpuSelection) : null;
  if (!selected) return {hw, history: h, cards, engineCards: selector.engineCards,
    otherCards: selector.otherCards, selected: null, names: selector.names};
  const history = h.gpus && h.gpus[gpuSelection] ? h.gpus[gpuSelection] : {};
  return {
    hw: {...hw, gpu_util: selected.util, gpu_mem_used: selected.mem_used, gpu_mem_total: selected.mem_total,
      gpu_temp: selected.temp, gpu_power: selected.power, gpu_power_limit: selected.power_limit,
      gpu_pcie_rx_mb: selected.pcie_rx_mb, gpu_pcie_tx_mb: selected.pcie_tx_mb,
      gpu_pcie_gen: selected.pcie_gen, gpu_pcie_gen_max: selected.pcie_gen_max,
      gpu_pcie_width: selected.pcie_width},
    history, cards, engineCards: selector.engineCards, otherCards: selector.otherCards,
    selected, names: selector.names,
  };
}

// A one-line summary of a reading over several cards (the tightest or the highest); `title` lists every card.
function perCardLine(cards, value, format, options = {}) {
  const rows = cards.map((card) => ({card, score: value(card), text: `GPU ${card.index} ${format(card)}`}));
  const full = rows.map((row) => row.text).join(" · ");
  if (cards.length === 1) return {text: full, title: full};
  if (options.summary === "tightest") {
    const tightest = rows.filter((row) => Number.isFinite(row.score)).reduce(
      (best, row) => !best || row.score < best.score ? row : best, null);
    return {text: tightest ? `${cards.length} cards · GPU ${tightest.card.index} tightest` : `${cards.length} cards`,
      title: full};
  }
  const highest = rows.filter((row) => Number.isFinite(row.score)).reduce(
    (best, row) => !best || row.score > best.score ? row : best, null);
  const count = `${cards.length} ${options.countLabel || "cards"}`;
  return {text: highest ? `${count} · max ${format(highest.card)} on GPU ${highest.card.index}` : count, title: full};
}

function allGpuLine(engineCards, otherCards, value, format) {
  const rows = [...engineCards, ...otherCards].map((card) => ({card,
    text: `GPU ${card.index} ${format(card)}${card.inModel ? "" : " (not in use)"}`}));
  const details = rows.map((row) => row.text).join(" · ");
  if (engineCards.length + otherCards.length <= 3) return {text: details, title: ""};
  const summary = perCardLine(engineCards, value, format, {countLabel: "cards in use"});
  return {text: `${summary.text}${otherCards.length ? ` · ${otherCards.length} not in use` : ""}`,
    title: details};
}

let lastMetrics = null, metricsFailures = 0, keyWarned = false, mcpTick = 0;
let reqShowAll = false;   // the Monitor's request table: the last 12, or every one the server keeps (issue #35)
let reqClearedAfter = null;
async function poll() {
  try {
    const r = await fetch(reqShowAll || reqClearedAfter != null ? "metrics?requests=all" : "metrics", {headers: headers()});
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
  if (tab === "monitor") renderMonitor(live, hw, st, eng, h, last, m.requests || [], m.totals, m.requests_kept, m.forge);
  if (tab === "monitor") renderConvCache(m.conversation_cache);
  if (tab === "about") renderAbout(eng, hw, st);
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
function setStat(key, value, unit, sub, values, max, options = {}) {
  $(`sv-${key}`).innerHTML = options.mutedValue
    ? `<span class="stats-empty-value">${esc(options.mutedValue)}</span>`
    : value == null ? "–" : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
  $(`ss-${key}`).textContent = sub || "";
  $(`ss-${key}`).title = options.subTitle || "";
  $(`sl-${key}`).title = options.title || "";
  const card = $(key);
  setCardHidden(card, options.hidden);
  card.title = options.cardTitle || "";
  if (options.tone) card.dataset.tone = options.tone; else delete card.dataset.tone;
  if (key === "vram-free") {
    const progress = $("vram-free-progress");
    const bar = $("vram-free-bar");
    if (options.tone) progress.dataset.tone = options.tone; else delete progress.dataset.tone;
    bar.style.width = `${options.ratio == null ? 0 : Math.max(0, Math.min(100, options.ratio))}%`;
  }
  if (values !== undefined) {
    spark(`sp-${key}`, values, max, key === "cpu-temp");
    if (key === "reuse" || key === "prefill-time") {
      const graph = $(`sp-${key}`), note = $(`gn-${key}`);
      if (options.graphState === "unreported") {
        graph.hidden = true;
        note.hidden = true;
      } else {
        const waiting = options.graphState === "waiting" || values.length < 2;
        graph.hidden = waiting;
        note.hidden = !waiting;
      }
    }
  }
}
const pct = (part, whole) => Number.isFinite(part) && Number.isFinite(whole) && whole > 0 ? 100 * part / whole : null;
function renderStats(eng, hw, st, h, requests, totals, forge) {
  const cpuTempMissing = hw.cpu_temp == null;
  const cpuTempNotice = cpuTempMissing && st.os === "windows";
  const cpuTempHint = "Run LibreHardwareMonitor with its web server on (Options > Remote Web Server, port 8085), or set STRATA_LHM_URL.";
  setStat("cpu-temp", cpuTempMissing ? null : fmt(hw.cpu_temp), "°C",
    cpuTempMissing ? "" : st.cpu_name || "", h.cpu_temp, 100,
    {tone: "warn", hidden: cpuTempMissing && !cpuTempNotice,
      mutedValue: cpuTempNotice ? "Needs LibreHardwareMonitor" : "", title: cpuTempNotice ? cpuTempHint : "",
      cardTitle: cpuTempNotice ? cpuTempHint : ""});
  const rows = Array.isArray(requests) ? requests : [];
  const newest = rows[0] || null;
  const noRequests = !rows.length || Number(totals && totals.requests) === 0;
  const offeredTotal = Number(totals && totals.drafts_offered);
  const acceptedTotal = Number(totals && totals.drafts_accepted);
  const specEnabled = Number(eng && eng.spec) > 0;
  const specSeries = rows.slice().reverse().filter((r) => Number.isFinite(r.drafts_offered) && r.drafts_offered > 0 &&
    Number.isFinite(r.drafts_accepted)).map((r) => pct(r.drafts_accepted, r.drafts_offered));
  const specValue = newest && pct(newest.drafts_accepted, newest.drafts_offered);
  const specSince = pct(acceptedTotal, offeredTotal);
  setStat("speculation", noRequests || specValue == null ? null : fmt(specValue, 1), "%",
    noRequests ? "" : specSince == null ? "" : `since start ${fmt(specSince, 1)}%`, specSeries, 100,
    {hidden: !specEnabled, mutedValue: noRequests ? "No requests yet" : !(offeredTotal > 0) ? "No drafts yet" : ""});

  const reuseSeries = rows.slice().reverse().filter((r) => Number.isFinite(r.prompt_tokens) && r.prompt_tokens > 0 &&
    Number.isFinite(r.reused)).map((r) => pct(r.reused, r.prompt_tokens));
  const reuseReported = rows.some((r) => Number.isFinite(r.prompt_tokens) && Number.isFinite(r.reused));
  const reuseValue = newest && pct(newest.reused, newest.prompt_tokens);
  const reuseSince = pct(Number(totals && totals.reused), Number(totals && totals.prompt_tokens));
  const reuseEmpty = noRequests ? "No requests yet" : !reuseReported ? "Not reported by this engine" : "";
  setStat("reuse", reuseEmpty || reuseValue == null ? null : fmt(reuseValue, 1), "%",
    reuseEmpty ? "" : `since start ${reuseSince == null ? "–" : `${fmt(reuseSince, 1)}%`} · ` +
      `${fmt(totals && totals.reused || 0)} tokens`, reuseSeries, 100,
    {mutedValue: reuseEmpty, graphState: noRequests ? "waiting" : !reuseReported ? "unreported" : ""});

  const newTokens = newest && Number.isFinite(newest.prompt_tokens)
    ? Math.max(0, newest.prompt_tokens - (Number.isFinite(newest.reused) ? newest.reused : 0)) : null;
  const promptMs = newest && Number.isFinite(newest.prompt_ms) ? newest.prompt_ms : null;
  const promptSpeed = promptMs > 0 && newTokens != null ? newTokens / (promptMs / 1000) : null;
  const prefillSeries = rows.slice().reverse().filter((r) => Number.isFinite(r.prompt_ms))
    .map((r) => r.prompt_ms / 1000);
  const prefillReported = rows.some((r) => Number.isFinite(r.prompt_ms));
  const prefillEmpty = noRequests ? "No requests yet" : !prefillReported ? "Not reported by this engine" : "";
  setStat("prefill-time", prefillEmpty || promptMs == null ? null : fmt(promptMs / 1000, 1), "s",
    prefillEmpty ? "" : newTokens == null ? "" : `${fmt(newTokens)} new tokens${promptSpeed == null ? "" : ` at ${fmt(promptSpeed)} t/s`}`,
    prefillSeries, undefined,
    {mutedValue: prefillEmpty, graphState: noRequests ? "waiting" : !prefillReported ? "unreported" : ""});

  const engineCards = Array.isArray(hw.gpus) && hw.gpus.length ? hw.gpus :
    hw.gpu_mem_total != null && hw.gpu_mem_used != null
      ? [{index: null, mem_total: hw.gpu_mem_total, mem_used: hw.gpu_mem_used}] : [];
  const otherGpuCards = Array.isArray(hw.other_gpus) ? hw.other_gpus : [];
  const selectedEngineCard = gpuSelection !== "all"
    ? engineCards.find((g) => String(g.index) === gpuSelection) : null;
  const selectedOtherCard = gpuSelection !== "all"
    ? otherGpuCards.find((g) => String(g.index) === gpuSelection) : null;
  const selectedCard = selectedEngineCard || selectedOtherCard;
  const freeCards = selectedCard ? [selectedCard] : engineCards;
  const freeInfo = freeCards.map((g) => ({...g,
    free: Number.isFinite(g.mem_total) && Number.isFinite(g.mem_used) ? g.mem_total - g.mem_used : null}));
  const readableFree = freeInfo.filter((x) => Number.isFinite(x.free));
  const tightest = readableFree.reduce((best, x) => !best || x.free < best.free ? x : best, null);
  const freeGiB = tightest ? tightest.free / 1073741824 : null;
  const totalGiB = tightest ? tightest.mem_total / 1073741824 : null;
  const engineCount = Number.isFinite(Number(st.gpu_count)) ? Number(st.gpu_count) : engineCards.length;
  const freeLine = !selectedCard && engineCards.length > 1
    ? perCardLine(freeInfo, (g) => g.free, (g) => g.free == null ? "–" : `${fmt(g.free / 1073741824, 2)} GiB free`,
      {summary: "tightest"}) : null;
  let freeSub = "";
  if (tightest && selectedOtherCard) freeSub = `GPU ${tightest.index} · not in use · of ${fmt(totalGiB, 1)} GiB`;
  else if (tightest && engineCards.length === 1) freeSub = `of ${fmt(totalGiB, 1)} GiB`;
  else if (tightest && selectedCard) freeSub = `GPU ${tightest.index} selected · ${engineCount} cards · ` +
    `of ${fmt(totalGiB, 1)} GiB`;
  else if (tightest) freeSub = `${freeLine.text} · of ${fmt(totalGiB, 1)} GiB`;
  else if (engineCount > 1) freeSub = `${engineCount} cards · free VRAM unavailable`;
  const freeTitle = freeLine ? freeLine.title : freeInfo.map((gpu) => {
    const label = gpu.index == null ? "GPU" : `GPU ${gpu.index}`;
    return `${label}: ${gpu.free == null ? "–" : `${fmt(gpu.free / 1073741824, 2)} GiB free`}`;
  }).join(" · ");
  let freeSeries = [];
  if (tightest && engineCards.length === 1 && tightest.index == null) {
    freeSeries = (h.gpu_mem_used || []).map((used) => used == null ? null : hw.gpu_mem_total - used);
    freeSeries = freeSeries.map((free) => free == null ? null : free / 1073741824);
  } else if (tightest && h.gpus && h.gpus[String(tightest.index)]) {
    const total = tightest.mem_total;
    freeSeries = (h.gpus[String(tightest.index)].mem_used || []).map((used) =>
      used == null ? null : (total - used) / 1073741824);
  }
  const tone = freeGiB == null ? "" : freeGiB < 0.5 ? "danger" : freeGiB < 1 ? "warn" : "";
  setStat("vram-free", freeGiB == null ? null : fmt(freeGiB, 2), "GiB", freeSub, freeSeries,
    totalGiB, {tone, ratio: freeGiB == null || !totalGiB ? 0 : 100 * freeGiB / totalGiB,
      title: tone ? "little VRAM left: a growing context or the vision encoder may not fit"
        : selectedCard ? "free VRAM on the selected GPU" : "the smallest free VRAM over the cards that run the model",
      subTitle: freeTitle});

  const measuredPower = hw.measured_power != null;
  $("slt-all-power").textContent = measuredPower ? "GPUs + CPU" : "All GPU power";
  const power = measuredPower ? hw.measured_power : hw.all_gpu_power;
  const powerHistory = measuredPower ? h.measured_power : h.all_gpu_power;
  setStat("all-power", power == null ? null : fmt(power), "W",
    measuredPower && hw.cpu_power != null
      ? `GPUs ${fmt(hw.measured_power - hw.cpu_power)} W · CPU ${fmt(hw.cpu_power)} W`
      : `${engineCount + otherGpuCards.length} NVIDIA cards · ${engineCount} in use`, powerHistory,
    undefined, {hidden: otherGpuCards.length === 0 && !measuredPower});

  const since = Number(totals && totals.since);
  const elapsed = Number.isFinite(since) ? Math.max(0, Math.floor(Date.now() / 1000 - since)) : null;
  const minutes = elapsed == null ? null : Math.floor(elapsed / 60);
  const uptime = minutes == null ? null : minutes >= 60
    ? `${Math.floor(minutes / 60)} h${minutes % 60 ? ` ${minutes % 60} min` : ""}` : `${minutes} min`;
  const sessionSub = `${fmt(totals && totals.requests || 0)} requests · ` +
    `${fmt(totals && totals.output_tokens || 0)} tokens written`;
  setStat("session", uptime, "", sessionSub);

  // Forge's `session` block is the chat open in its sidebar (null: none open). A Forge without the field
  // predates it; its day-wide totals are shown, labelled as such.
  const hasSession = Boolean(forge) && "session" in forge;
  const forgeChat = hasSession ? forge.session : null;
  const forgeToday = (hasSession ? forgeChat : forge && forge.today) || {};
  const forgeLast = hasSession ? forgeChat && forgeChat.last_request : forge && forge.last_request;
  $("slt-forge").textContent = hasSession ? "Forge · this chat" : forge ? "Forge · today" : "Forge";
  const compact = (n) => {
    if (!Number.isFinite(Number(n))) return "?";
    const value = Number(n), abs = Math.abs(value);
    if (abs >= 1e6) return `${fmt(value / 1e6, 1)}M`;
    if (abs >= 1e3) return `${fmt(value / 1e3, abs >= 10000 ? 0 : 1)}K`;
    return fmt(value);
  };
  const tools = Number(forgeToday.tool_calls) || 0;
  const failures = Number(forgeToday.tool_failures) || 0;
  const failedPct = tools > 0 ? fmt(100 * failures / tools, 1) : "0";
  let forgeSub = `${fmt(forgeToday.compactions || 0)} compactions` +
    (Number(forgeToday.compaction_attempts_failed) > 0 ? ` (${fmt(forgeToday.compaction_attempts_failed)} failed)` : "") +
    ` · ${fmt(tools)} tools, ${failedPct}% failed · in ${compact(forgeToday.input_tokens)} / out ${compact(forgeToday.output_tokens)} tokens`;
  if (Number(forgeToday.turn_errors) > 0) forgeSub += ` · ${fmt(forgeToday.turn_errors)} errors`;
  let forgeTitle = "";
  if (forgeLast) {
    const input = fmt(forgeLast.input_tokens);
    const limit = Number(forgeLast.context_limit);
    const context = forgeLast.context_limit == null ? (input === "?" ? "" : " tokens")
      : ` / ${fmt(limit)} tokens${limit > 0 ? ` (${fmt(100 * forgeLast.input_tokens / limit, 1)}%)` : ""}`;
    forgeTitle = `${forgeLast.model || "Forge"} · last request ${input}${context}`;
  }
  if (forge && forge.stale) forgeTitle += `${forgeTitle ? " · " : ""}stale`;
  if (hasSession && !forgeChat) {
    setStat("forge", null, "", "", undefined, undefined,
      {hidden: false, cardTitle: forgeTitle, mutedValue: "No active Forge chat"});
    return;
  }
  setStat("forge", forge ? fmt(forgeToday.turns || 0) : null, "turns", forgeSub,
    undefined, undefined, {hidden: !forge, cardTitle: forgeTitle});
}
function renderMonitor(live, hw, st, eng, h, last, requests, totals, kept, forge) {
  const gpu = gpuMetricData(hw, h, updateGpuSelector(hw, st));
  const ghw = gpu.hw, gh = gpu.history, cards = gpu.cards, selected = gpu.selected;
  const engineCards = gpu.engineCards, otherCards = gpu.otherCards;
  const multi = cards.length > 1 && !selected;
  const multiEngine = engineCards.length > 1 && !selected;
  const cardName = selected ? gpu.names[cards.indexOf(selected)] || `GPU ${selected.index}` : "";
  const selectedSub = selected && !selected.inModel ? `${cardName} · not in use` : cardName;
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
  const loadLine = multi ? allGpuLine(engineCards, otherCards, (g) => g.util,
    (g) => g.util == null ? "–" : `${fmt(g.util)}%`) : null;
  setMetric("gpu", ghw.gpu_util == null ? null : fmt(ghw.gpu_util), "%",
            selected ? selectedSub : multi ? loadLine.text : st.gpu_name || (st.gpu_note ? "not available" : ""));
  if (loadLine) $("ms-gpu").title = loadLine.title;
  spark("sp-gpu", selected ? gh.util : h.gpu_util, 100);

  const vramLine = multi ? allGpuLine(engineCards, otherCards, (g) => g.mem_used,
    (g) => g.mem_used == null ? "–" : `${gb(g.mem_used)} GB`) : null;
  setMetric("vram", ghw.gpu_mem_used == null ? null : gb(ghw.gpu_mem_used),
            ghw.gpu_mem_total ? `/ ${gb(ghw.gpu_mem_total, 0)} GB` : "GB",
            selected ? selectedSub : multi ? vramLine.text
              : eng.expert_slots ? `${fmt(eng.expert_slots)} experts cached` : (st.gpu_note ? "not available on Windows AMD yet" : ""));
  if (vramLine) $("ms-vram").title = vramLine.title;
  spark("sp-vram", selected ? gh.mem_used : h.gpu_mem_used, ghw.gpu_mem_total);

  const tempLine = multi ? allGpuLine(engineCards, otherCards, (g) => g.temp,
    (g) => g.temp == null ? "–" : `${fmt(g.temp)}°`) : null;
  setMetric("temp", ghw.gpu_temp == null ? null : fmt(ghw.gpu_temp), "°C",
            selected ? selectedSub : multi ? tempLine.text : "");
  if (tempLine) $("ms-temp").title = tempLine.title;
  spark("sp-temp", selected ? gh.temp : h.gpu_temp, 90, true);

  const powerLine = multi ? allGpuLine(engineCards, otherCards, (g) => g.power,
    (g) => g.power == null ? "–" : `${fmt(g.power)} W`) : null;
  setMetric("power", ghw.gpu_power == null ? null : fmt(ghw.gpu_power), "W",
            selected ? !selected.inModel ? selectedSub : ghw.gpu_power_limit == null ? "" : `of ${fmt(ghw.gpu_power_limit)} W limit`
              : multi ? powerLine.text : ghw.gpu_power_limit ? `of ${fmt(ghw.gpu_power_limit)} W limit` : "");
  if (powerLine) $("ms-power").title = powerLine.title;
  const limitLine = multi ? allGpuLine(engineCards, otherCards, (g) => g.power_limit,
    (g) => g.power_limit == null ? "–" : `${fmt(g.power_limit)} W limit`) : null;
  const limits = selected
    ? (ghw.gpu_power_limit == null ? selectedSub : `${selectedSub}: ${fmt(ghw.gpu_power_limit)} W limit`)
    : multi ? limitLine.title || limitLine.text : "";
  $("ml-power").title = limits;
  spark("sp-power", selected ? gh.power : h.gpu_power, ghw.gpu_power_limit);

  const gen = ghw.gpu_pcie_gen_max || ghw.gpu_pcie_gen;
  let pcieSub = "", pcieTitle = "";
  const idle = ghw.gpu_pcie_gen && gen && ghw.gpu_pcie_gen < gen ? `idle Gen${ghw.gpu_pcie_gen}` : "";
  const hasCardPcie = engineCards.some((g) => g.pcie_rx_mb != null || g.pcie_tx_mb != null);
  if (selected) {
    pcieSub = `in ${ghw.gpu_pcie_rx_mb == null ? "–" : fmt(ghw.gpu_pcie_rx_mb, ghw.gpu_pcie_rx_mb < 10 ? 1 : 0)} MB/s` +
      ` · out ${ghw.gpu_pcie_tx_mb == null ? "–" : fmt(ghw.gpu_pcie_tx_mb, ghw.gpu_pcie_tx_mb < 10 ? 1 : 0)} MB/s`;
    if (!selected.inModel) pcieSub += " · not in use";
    pcieTitle = "in = host to GPU, out = GPU to host";
  } else if (multiEngine && hasCardPcie) {
    const rates = (g) => `${g.pcie_rx_mb == null ? "–" : fmt(g.pcie_rx_mb, g.pcie_rx_mb < 10 ? 1 : 0)} in / ` +
      `${g.pcie_tx_mb == null ? "–" : fmt(g.pcie_tx_mb, g.pcie_tx_mb < 10 ? 1 : 0)} out`;
    const rows = engineCards.map((g) => ({gpu: g, score: g.pcie_rx_mb == null && g.pcie_tx_mb == null
      ? null : (g.pcie_rx_mb || 0) + (g.pcie_tx_mb || 0), text: `GPU ${g.index}: ${rates(g)}`}));
    const full = rows.map((row) => `${row.text} MB/s`).join(" · ");
    if (engineCards.length <= 3) {
      pcieSub = `${rows.map((row) => row.text).join(" · ")} MB/s`;
      pcieTitle = "in = host to GPU, out = GPU to host";
    } else {
      const busiest = rows.filter((row) => Number.isFinite(row.score)).reduce(
        (best, row) => !best || row.score > best.score ? row : best, null);
      pcieSub = busiest ? `${engineCards.length} cards · busiest GPU ${busiest.gpu.index}: ${rates(busiest.gpu)} MB/s`
                         : `${engineCards.length} cards`;
      pcieTitle = `${full} · in = host to GPU, out = GPU to host`;
    }
  } else {
    // One card, or several without per-card PCIe readings: the aggregate wording.
    pcieSub = ghw.gpu_pcie_rx_mb == null ? ""
      : `to GPU ${fmt(ghw.gpu_pcie_rx_mb, ghw.gpu_pcie_rx_mb < 10 ? 1 : 0)} MB/s`;
  }
  if (idle) pcieSub += `${pcieSub ? " · " : ""}${idle}`;
  setMetric("pcie", gen ? `Gen${gen}` : null, ghw.gpu_pcie_width ? `x${ghw.gpu_pcie_width}` : "", pcieSub);
  $("ms-pcie").title = pcieTitle;
  spark("sp-pcie", selected ? gh.pcie_rx_mb : h.gpu_pcie_rx_mb);
  const cpuSensors = hw.cpu_power == null ? "" : `${fmt(hw.cpu_power, 1)} W package`;
  const cpuThreads = st.threads ? `${st.cores ? `${st.cores} cores · ` : ""}${st.threads} threads` : "";
  setMetric("cpu", hw.cpu == null ? null : fmt(hw.cpu), "%", cpuSensors || cpuThreads);
  $("ms-cpu").title = cpuSensors && cpuThreads ? cpuThreads : "";
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
  $("cpu-temp-text").textContent = hw.cpu_temp == null ? "–" : `${fmt(hw.cpu_temp)} °C`;
  $("cpu-temp-bar").style.width = hw.cpu_temp == null ? "0%" : `${Math.min(100, hw.cpu_temp)}%`;

  // recent requests
  const body = $("req-body");
  const visibleRequests = reqClearedAfter == null ? requests : requests.filter((r) => r.time > reqClearedAfter);
  if (!visibleRequests.length) {
    body.innerHTML = `<tr><td colspan="10" class="muted">${reqClearedAfter == null ? "No requests yet" : "Cleared. New requests will appear here (refresh to see all)"}</td></tr>`;
  } else {
    const badge = {stop: ["", "Done"], length: ["", "Max tokens"], cancel: ["st-badge--queued", "Stopped"],
                   disconnect: ["st-badge--queued", "Closed"], error: ["st-badge--error", "Error"]};
    body.innerHTML = visibleRequests.slice(0, reqShowAll ? visibleRequests.length : 12).map((r) => {
      const [cls, text] = badge[r.finish] || ["", r.finish || "–"];
      const t = new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      const proj = r.projection == null ? "" : ` <span class="st-badge${r.projection ? " st-badge--reading" : ""}" title="experimental speed projection ${r.projection ? "on" : "off"}">${r.projection ? "ESP" : "stock"}</span>`;
      // #588: the VRAM share; the PCIe share (--pcie-frac) beside it when there is one
      const hit = r.hit_rate == null ? "–" : `${(r.hit_rate * 100).toFixed(1)}%` +
        (r.pcie_share ? ` <span class="muted" title="routed experts the GPU read over PCIe (--pcie-frac) or another GPU computed">+${(r.pcie_share * 100).toFixed(1)}% PCIe</span>` : "");
      const prefill = r.prompt_ms == null ? "–" : `${fmt(r.prompt_ms / 1000, 1)} s`;
      const drafts = Number.isFinite(r.drafts_offered) && r.drafts_offered > 0 && Number.isFinite(r.drafts_accepted)
        ? `${fmt(100 * r.drafts_accepted / r.drafts_offered, 1)}%` : "–";
      const draftTitle = drafts === "–" ? "No draft tokens offered"
        : `${fmt(r.drafts_accepted)} accepted / ${fmt(r.drafts_offered)} offered`;
      return `<tr><td>${esc(t)}</td><td><span class="st-badge ${cls}">${esc(text)}</span>${proj}</td><td class="num">${fmt(r.prompt_tokens)}</td>
        <td class="num">${fmt(r.reused)}</td>
        <td class="num" title="time spent reading the new prompt tokens">${prefill}</td>
        <td class="num">${fmt(r.output_tokens)}</td><td class="num">${fmt(r.decode_tok_s, 1)}</td>
        <td class="num" title="${draftTitle}">${drafts}</td>
        <td class="num">${hit}</td><td class="num">${fmt(r.duration_s, 1)} s</td></tr>`;
    }).join("");
  }
  const all = $("req-all");
  kept = reqClearedAfter == null ? (kept == null ? requests.length : kept) : visibleRequests.length;
  all.hidden = kept <= 12;
  all.textContent = reqShowAll ? "Show fewer" : `Show all (${kept})`;
  $("req-wrap").classList.toggle("all", reqShowAll);
  $("req-totals").textContent = renderTotals(totals);
  renderStats(eng, hw, st, h, requests, totals, forge);
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
$("req-clear").addEventListener("click", () => {
  const rows = lastMetrics && Array.isArray(lastMetrics.requests) ? lastMetrics.requests : [];
  reqClearedAfter = rows.length ? Math.max(...rows.map((r) => r.time)) : Date.now() / 1000;
  if (lastMetrics) render(lastMetrics);
  poll();
});

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
let cfgKeys = [];
async function loadConfig() {
  let r;
  try { r = await fetch("config", {headers: headers()}); } catch (e) { return; }
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
      input = `<select class="st-input" id="${id}"><option value=""${cur === "" ? " selected" : ""}>default</option>` +
        opts.map(([val, text]) => `<option value="${esc(val)}"${cur === val ? " selected" : ""}>${esc(text)}</option>`).join("") + `</select>`;
    } else {
      const text = v == null ? "" : Array.isArray(v) ? v.join(", ") : String(v);
      input = `<input class="st-input" id="${id}" ${k.kind === "number" ? 'type="number" step="any" min="0"' : 'type="text"'} ` +
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
