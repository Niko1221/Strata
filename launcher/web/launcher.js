// launcher.js — the model and preset picker.
//
// The page holds no rules of its own: what fits this PC, what is installed, what an install would download, how
// fast a model is — all of it comes from /api/*, which answers with setup.py's own tables and the running server's
// own numbers.  What this file does is draw that, keep a preset being edited in memory, and ask before anything
// big happens (a 60-110 GB download, a 5-10 minute measurement, starting a model).
"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = String(text);
  return n;
};
const clear = (n) => { while (n.firstChild) n.removeChild(n.firstChild); };

let SCHEMA = null, MODELS = null, S = null;      // the tables, the catalog, the state
let sel = null;                                  // {id, preset, builtin, draft, dirty}
let FACTS = null, PLAN = null, JOBS = {}, DIFF = null;
let CONNECT = null, CONNECT_PORT = null;        // the base URLs of the model that is running, fetched once per start
let logTab = "install", timer = null, busy = false;

async function api(path, opts) {
  const r = await fetch("/api/" + path, opts);
  let body = {};
  try { body = await r.json(); } catch (e) { /* an empty body: the message below is enough */ }
  if (!r.ok) throw new Error((body && body.error && body.error.message) || `the launcher got HTTP ${r.status}`);
  return body;
}
const post = (path, data) => api(path, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data || {}),
});

let toastTimer = null;
function toast(msg, bad) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast" + (bad ? " bad" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, bad ? 12000 : 6000);
}
function fail(e) { toast(e.message || String(e), true); }

// ---------------------------------------------------------------- the chat page
// starting a model on the page opens its chat page in a tab, as run-<model>.bat does with --open.  The tab is
// opened inside the click (a popup blocker allows that) and filled with the chat page once the model answers.
let CHAT_TAB = null, CHAT_WAIT = false;
let JOB_WAS_RUNNING = false;

function openChatWhenReady(withTab) {
  CHAT_WAIT = true;
  if (!withTab) return;
  CHAT_TAB = window.open("about:blank", "strata-chat");
  if (!CHAT_TAB) return;                     // blocked: showChatPage falls back to a message
  CHAT_TAB.document.write('<!doctype html><meta charset="utf-8"><title>Strata</title>'
    + '<body style="font:15px/1.6 system-ui,sans-serif;max-width:34em;margin:8vh auto;padding:0 22px">'
    + '<h1 style="font-size:19px;font-weight:600">Strata is loading the model</h1>'
    + '<p>This page becomes the chat page as soon as the model is in memory - 1-3 minutes, and the PC may be slow '
    + 'meanwhile. Keep this window open; the launcher shows the server log.</p></body>');
  CHAT_TAB.document.close();
}

function showChatPage(port) {
  if (!CHAT_WAIT) return;
  CHAT_WAIT = false;
  const url = `http://127.0.0.1:${port}/`;
  if (CHAT_TAB && !CHAT_TAB.closed) {
    try { CHAT_TAB.location = url; CHAT_TAB.focus(); } catch (e) { /* the user closed it */ }
  } else {
    toast(`the model is ready - the chat page is at ${url}; the top bar has an Open button too`);
  }
  CHAT_TAB = null;
}

function dropChatTab() {                     // the start stopped: no empty tab left behind
  if (CHAT_TAB && !CHAT_TAB.closed) { try { CHAT_TAB.close(); } catch (e) {} }
  CHAT_TAB = null;
  CHAT_WAIT = false;
}

// ---------------------------------------------------------------- the preset being shown
function allPresets() {
  return [...(S.presets || []), ...(S.builtin || [])];
}
function presetById(id) {
  return allPresets().find((p) => p.id === id) || null;
}
function familyOf(id) {
  return (MODELS.families || []).find((f) => f.id === id) || null;
}

async function select(id) {
  const p = presetById(id);
  if (!p) return;
  sel = { id, listId: id, preset: JSON.parse(JSON.stringify(p)), builtin: !!p.builtin, draft: false, dirty: false };
  PLAN = null; FACTS = null;
  $("plan").hidden = true;
  renderPreset();
  checkDiff();
  try { FACTS = await api(`model?id=${encodeURIComponent(id)}`); renderInfo(); } catch (e) { fail(e); }
  markList();
}

// a draft preset: from an installed model, or a new one. Not stored until Save.
async function openDraft(preset, title, listId) {
  sel = { id: preset.id, listId: listId || preset.id, preset, builtin: false, draft: true, dirty: true };
  PLAN = null; FACTS = null;
  $("plan").hidden = true;
  renderPreset(title);
  checkDiff();
  renderInfo();
  markList();
}

function collect() {
  if (!sel) return;
  const out = {};
  for (const f of SCHEMA.fields) {
    const n = $(`f-${f.key}`);
    if (!n) continue;
    out[f.key] = f.type === "number" ? (n.value.trim() === "" ? "" : n.value.trim()) : n.value;
  }
  out.name = $("f-name").value.trim();
  out.note = $("f-note").value.trim();
  out.id = sel.preset.id;
  Object.assign(sel.preset, out);
  const dirty = JSON.stringify(sel.preset) !== JSON.stringify(presetById(sel.id) || {});
  if (dirty !== sel.dirty) { sel.dirty = dirty; renderActions(); }
}

// ---------------------------------------------------------------- drawing: the preset
function renderPreset(title) {
  const p = sel.preset;
  $("preset-name").textContent = title || p.name || "New preset";
  $("preset-summary").textContent = summaryLine(p);
  const note = $("preset-note");
  note.textContent = p.note || "";
  note.hidden = !p.note;
  renderForm();
  renderActions();
  const warn = $("preset-warn");
  warn.hidden = true;
  clear(warn);
}

function summaryLine(p) {
  const fam = familyOf(p.family);
  const bits = [`${fam ? fam.title : p.family} ${p.model}`, `${Math.round((p.context || 0) / 1024)}K context`];
  if (p.kv) bits.push(`KV ${p.kv}`);
  bits.push(p.vision === "no" || !p.vision ? "text only" : `images (${p.vision})`);
  if (p.gpus) bits.push(`cards ${p.gpus}`);
  else if (p.gpu) bits.push(`card ${p.gpu}`);
  if (p.low_ram && p.low_ram !== "auto") bits.push(`low-RAM ${p.low_ram}`);
  if (p.calibrate && p.calibrate !== "ask") bits.push(`measure this PC: ${p.calibrate}`);
  return bits.join(" · ");
}

function renderForm() {
  const form = $("form");
  clear(form);
  form.hidden = false;
  form.appendChild(field("name", "Preset name", sel.preset.name, { wide: true, type: "text" }));
  form.appendChild(field("note", "Note", sel.preset.note, { wide: true, type: "textarea" }));
  for (const f of SCHEMA.fields) form.appendChild(field(f.key, f.label, sel.preset[f.key], f));
  fillSizes();
}

function field(key, label, value, f) {
  const wrap = el("div", "field" + (f.wide ? " wide" : ""));
  const id = `f-${key}`;
  const lab = el("label", null, label);
  lab.htmlFor = id;
  wrap.appendChild(lab);
  let input, ownInput = null;
  if (f.type === "textarea") input = el("textarea");
  else if (f.type === "select" || f.type === "family" || f.type === "size" || f.type === "context") input = el("select");
  else input = el("input");
  input.id = id;
  input.name = key;
  if (f.type === "number") input.type = "number";
  if (f.type === "text") input.type = "text";
  if (f.secret) input.type = "text";
  if (f.type === "family") {
    for (const fam of MODELS.families) input.appendChild(new Option(fam.title, fam.id));
  } else if (f.type === "context") {
    const list = MODELS.contexts || [];
    for (const c of list) {
      input.appendChild(new Option(`${Math.round(c / 1024)}K` + (c > SCHEMA.trained_context ? " (scaled)" : ""), c));
    }
    if (value && !list.includes(Number(value))) {
      input.appendChild(new Option(`${Math.round(Number(value) / 1024)}K (your own value)`, value));
    }
    input.appendChild(new Option("own value…", "__own"));
    ownInput = el("input");
    ownInput.id = "f-context-own";
    ownInput.type = "number"; ownInput.min = 1024; ownInput.max = 1048576; ownInput.step = 1024;
    ownInput.hidden = true;
    ownInput.oninput = ownInput.onchange = () => {
      const n = parseInt(ownInput.value, 10);
      if (!n) return;
      if (![...input.options].some((o) => o.value === String(n))) {
        input.insertBefore(new Option(`${Math.round(n / 1024)}K (your own value)`, n), input.lastElementChild);
      }
      input.value = String(n);            // the field's value stays the token count the preset stores
    };
    input.onchange = () => {
      if (input.value !== "__own") return;
      ownInput.hidden = false;
      ownInput.value = value && !list.includes(Number(value)) ? value : 196608;
      ownInput.focus();
    };
  } else if (f.type === "select") {
    for (const c of f.choices) input.appendChild(new Option(c.label, c.value));
  }
  if (f.type === "number" || f.type === "text" || f.type === "textarea") input.value = value === null ? "" : (value ?? "");
  else input.value = value ?? "";
  wrap.appendChild(input);
  if (ownInput) wrap.appendChild(ownInput);
  const help = [];
  if (f.help) help.push(f.help);
  if (f.flag) help.push(`setup: ${f.flag}`);
  if (f.type === "context") help.push("1,024 to 1,048,576 tokens: setup takes any --context, its menu lists the ones above");
  if (help.length) wrap.appendChild(el("div", "help", help.join(" · ")));
  return wrap;
}

function fillSizes() {
  const selEl = $("f-model");
  if (!selEl) return;
  const fam = familyOf(sel.preset.family);
  if (!fam) return;
  clear(selEl);
  const known = (MODELS.sizeInfo || {});
  for (const m of fam.sizes) {
    const info = known[m] || {};
    const fit = info.on_this_pc && info.on_this_pc !== "fits" ? ` — ${info.on_this_pc}` : "";
    selEl.appendChild(new Option(`${m} (${info.download_gb || "?"} GB download)${fit}`, m));
  }
  // the preset's own size, not what the select happens to hold: a form that was just built has an empty one, and
  // that used to leave the first size of the family (Q2_0) showing for every preset
  const want = sel.preset.model || selEl.value;
  selEl.value = fam.sizes.includes(want) ? want : fam.sizes[0];
  sel.preset.model = selEl.value;
}

function renderActions() {
  const box = $("preset-actions");
  clear(box);
  if (!sel) return;
  const installed = installedById(modelId(sel.preset));  const running = (S.running || [])[0];

  const add = (label, cls, fn, disabled) => {
    const b = el("button", cls, label);
    b.disabled = !!disabled;
    b.onclick = fn;
    box.appendChild(b);
    return b;
  };
  if (sel.draft || sel.dirty) {
    add(sel.draft ? "Save preset" : "Save changes", "primary", savePreset);
  }
  if (!sel.draft && sel.dirty) add("Discard", "ghost", () => select(sel.id));
  add("Duplicate", "", duplicatePreset);
  if (sel.draft) add("Delete", "ghost", () => deletePreset(), true).title = "not stored yet: nothing to delete";
  else if (sel.builtin) add("Delete", "ghost", null, true).title =
    "this preset is not stored: it is made from this PC or from the model's own config";
  else add("Delete", "danger", () => deletePreset());
  add("What installing it would do", "", showPlan);
  if (installed) add("Apply these settings", "", applyNow);
  else add("Download", "primary", installNow);
  if (installed) {
    const runningThis = running && FACTS && running.model === FACTS.model_name;
    const other = running && !runningThis;
    const start = add(runningThis ? "Running" : "Start", runningThis ? "" : "primary", startNow, runningThis || other);
    if (other) start.title = `stop ${running.model} first - one model runs at a time on this PC`;
    if (installed && !other && DIFF && DIFF.changes && DIFF.changes.length) {
      start.title = "Start runs the settings in the model's own config; this preset differs from them";
    }
    add("Measure this PC", "", () => calibrateNow(false), !!running || !!(JOBS.calibrate && JOBS.calibrate.running));
  }
}

function modelId(p) {
  const fam = familyOf(p.family) || {};
  return `${fam.tag || ""}${p.model || ""}`.toLowerCase();
}
function installedById(id) {
  return (S.installed?.models || []).find((m) => m.model === id) || null;
}

// ---------------------------------------------------------------- actions
async function savePreset() {
  collect();
  try {
    const r = await post("preset/save", sel.preset);
    toast(`saved the preset "${r.preset.name}"`);
    await refresh(false);
    sel = { id: r.preset.id, preset: r.preset, builtin: false, draft: false, dirty: false };
    renderPreset();
    renderActions();
    markList();
  } catch (e) { showErrors(e); }
}

function showErrors(e) {
  const box = $("preset-warn");
  box.hidden = false;
  box.textContent = e.message || String(e);
}

async function duplicatePreset() {
  collect();
  const copy = { ...sel.preset, id: "", name: `${sel.preset.name} (copy)`.slice(0, 60) };
  try {
    const r = await post("preset/save", copy);
    await refresh(false);
    select(r.preset.id);
  } catch (e) { showErrors(e); }
}

async function deletePreset(p) {
  p = p || sel.preset;
  if (!confirm(`Remove the preset "${p.name}"? The model it installed stays on this PC.`)) return;
  try {
    await post("preset/delete", { id: p.id });
    if (sel && sel.id === p.id) {
      sel = null;
      $("preset-name").textContent = "No preset selected";
      $("preset-summary").textContent = "";
      $("preset-note").hidden = true;
      $("form").hidden = true;
      $("plan").hidden = true;
    }
    await refresh(false);
    const first = allPresets()[0];
    if (first) select(first.id);
  } catch (e) { fail(e); }
}

async function showPlan() {
  if (sel.draft) { toast("save the preset first — the plan is made from a saved preset"); return; }
  try {
    PLAN = await api(`plan?id=${encodeURIComponent(sel.id)}`);
    drawPlan();
  } catch (e) { fail(e); }
}

// what the preset differs from on the model's own config: Start runs the config, not the preset
let diffTimer = null;
function queueDiff() { clearTimeout(diffTimer); diffTimer = setTimeout(checkDiff, 400); }

async function checkDiff() {
  if (!sel) { DIFF = null; renderDiff(); return; }
  try { DIFF = await post("preset/diff", sel.preset); } catch (e) { DIFF = null; }
  renderDiff();
}

function renderDiff() {
  const box = $("preset-diff");
  clear(box);
  box.hidden = true;
  if (!DIFF || !DIFF.installed || !(DIFF.changes || []).length) return;
  box.appendChild(el("b", null, `Starting ${DIFF.model_id} runs the settings in its own config, not this preset:`));
  const ul = el("ul");
  for (const c of DIFF.changes) {
    ul.appendChild(el("li", null, `${c.label}: the model runs with ${c.config}, this preset says ${c.preset}`));
  }
  box.appendChild(ul);
  box.appendChild(el("div", "sub", "Start applies them first: setup re-writes the model's own config - a few "
    + "seconds to a minute here, the files already downloaded are only checked again. 'Apply these settings' does "
    + "the same without starting it."));
  box.hidden = false;
}

function drawPlan() {
  const box = $("plan");
  clear(box);
  box.hidden = false;
  const p = PLAN;
  box.appendChild(el("h3", null, p.already_installed ? `Applying ${p.title} to this model`
    : `Installing ${p.title}`));
  const dl = el("dl");
  const facts = [
    ["Download", p.already_installed ? "already here: nothing is fetched again, setup only re-checks it"
      : `${p.download_gb} GB of model files, plus ~8 GB of the draft layer`],
    ["Disk", `${p.disk_needed_gb} GB needed · ${p.disk_free_gb ?? "?"} GB free in ${p.data_folder}`],
    ["RAM", p.ram || "not checked"],
    ["Context", `${Math.round(p.context / 1024)}K · images ${p.images}` +
      (p.own_context ? " · a length of your own: setup takes any --context, and the published numbers mark the "
        + "nearest length they measured" : "")],
    ["Takes", p.already_installed ? "a few seconds to a minute" : p.takes],
    ["Already here", p.already_installed ? "this model is installed: setup will re-check it and apply these settings" : "no"],
  ];
  for (const [k, v] of facts) {
    const row = el("div", "kv");
    row.appendChild(el("span", null, k));
    row.appendChild(el("b", null, v));
    dl.appendChild(row);
  }
  box.appendChild(dl);
  if (p.changes && p.changes.length) {
    const w = el("div", "notice warn");
    w.appendChild(el("b", null, "This model is installed. Installing this preset would change how it runs:"));
    const ul = el("ul");
    for (const c of p.changes) ul.appendChild(el("li", null, `${c.label}: ${c.config} becomes ${c.preset}`));
    w.appendChild(ul);
    w.appendChild(el("div", "sub", "the model keeps these settings after the install; the running server needs a "
      + "stop and a start to use them"));
    box.appendChild(w);
  }
  if (p.note) box.appendChild(el("p", "sub", p.note));
  if (p.license) box.appendChild(el("p", "sub", `Licence: ${p.license}`));
  box.appendChild(el("pre", "cmd", p.setup_command));
  const go = el("button", "primary", p.already_installed ? "Apply these settings to the model"
    : `Download and prepare it (${p.download_gb} GB)`);
  go.onclick = p.already_installed ? applyNow : installNow;
  const row = el("div", "controls");
  row.style.marginTop = "12px";
  row.appendChild(go);
  if (p.calibrate === "always") row.appendChild(el("span", "sub", "this preset measures this PC after installing (5-10 minutes)"));
  box.appendChild(row);
}

async function applyNow() {
  if (sel.draft) { toast("save the preset first — the settings are applied from a saved preset"); return; }
  try {
    const r = await post("apply", { id: sel.id });
    toast(r.summary);
    await refresh(false);
    checkDiff();
  } catch (e) { fail(e); }
}

async function installNow() {
  if (sel.draft) { toast("save the preset first"); return; }
  if (!PLAN) await showPlan();
  try {
    await post("install", { id: sel.id });
    toast("the download started — its progress is in the Download panel");
    logTab = "install";
    renderTabs();
    await refresh(false);
  } catch (e) { fail(e); }
}

async function startNow() {
  // a preset that says "ask": offer the measurement once, the first time this model runs on this PC
  const neverCalibrated = !FACTS || !(FACTS.calibrations || []).length;
  if (sel.preset.calibrate === "ask" && neverCalibrated) {
    const first = confirm("This model has never been measured on this PC.\n\n"
      + "Measure it first (5-10 minutes, the PC is busy) and start it with the settings that were fastest,\n"
      + "or start now with the settings the engine ships with.\n\n"
      + "OK = measure first, then start. Cancel = start now.");
    if (first) { await calibrateNow(true); return; }
  }
  try {
    openChatWhenReady(true);                                  // the tab is opened in the click, filled when ready
    const r = await post("start", { id: sel.id, wait_seconds: 10 });
    toast(r.summary);
    await refresh(false);
  } catch (e) { dropChatTab(); fail(e); }
}

async function stopNow() {
  dropChatTab();
  try {
    const r = await post("stop", {});
    toast(r.summary);
    await refresh(false);
  } catch (e) { fail(e); }
}

async function calibrateNow(thenStart) {
  const m = modelId(sel.preset);
  if (!thenStart && !confirm("Measuring this PC takes 5-10 minutes and keeps it busy: the model is started a few " +
    "times with different settings, and the fastest ones are kept in the model's own config. Continue?")) return;
  try {
    await post("calibrate", { model: m, then_start: !!thenStart });
    if (thenStart) openChatWhenReady(false);        // the launcher starts the model when the measurement ends
    logTab = "calibrate";
    renderTabs();
    toast("measuring this PC — " + (thenStart ? "the model starts when it is done" : "watch the Measuring panel"));
    await refresh(false);
  } catch (e) { fail(e); }
}

// ---------------------------------------------------------------- drawing: the lists
function markList() {
  const want = sel ? (sel.listId || sel.id) : null;
  for (const n of document.querySelectorAll(".item")) n.classList.toggle("selected", n.dataset.id === want);
}

function renderInstalled() {
  const box = $("installed");
  clear(box);
  const models = S.installed?.models || [];
  if (!models.length) {
    box.appendChild(el("p", "empty", "No model is installed here yet. Pick a preset and download one."));
  }
  for (const m of models) {
    const b = el("button", "item");
    b.dataset.id = `model:${m.model}`;
    const row = el("div", "row");
    row.appendChild(el("strong", null, m.model));
    const running = (S.running || []).some((r) => r.model === m.model_name);
    row.appendChild(el("span", "tag " + (running ? "ok" : m.ready ? "" : "bad"),
      running ? "running" : m.ready ? "installed" : "files missing"));
    b.appendChild(row);
    b.appendChild(el("div", "sub", [
      m.context ? `${Math.round(m.context / 1024)}K` : "", m.kv ? `KV ${m.kv}` : "",
      m.images && m.images !== "off" ? `images ${m.images}` : "", m.port ? `:${m.port}` : "",
      m.last_used ? `used ${m.last_used}` : "",
    ].filter(Boolean).join(" · ")));
    b.onclick = () => fromModel(m.model);
    box.appendChild(b);
  }
  $("data-dir").textContent = `Strata folder: ${S.strata_folder || "?"} · model files: ${S.installed?.data_folder || "?"}` +
    ` · engine ${S.installed?.engine?.version || "?"}`;
}

async function fromModel(id) {
  try {
    const r = await post("preset/from-model", { model: id });
    await openDraft(r.preset, `${id} — installed model`, `model:${id}`);
    FACTS = await api(`model?id=${encodeURIComponent(id)}`).catch(() => null);   // its missing files, if any
    renderInfo();
  } catch (e) { fail(e); }
}

function renderPresets() {
  const box = $("presets");
  clear(box);
  const all = S.builtin || [];
  const here = all.filter((p) => p.id.startsWith("builtin-installed-"));
  const groups = [["Your presets", S.presets || [], "none yet"],
                  ["As it is set up here", here, ""],
                  ["For this PC", all.filter((p) => !p.id.startsWith("builtin-installed-")), "none yet"]];
  for (const [title, items, ifEmpty] of groups) {
    if (!items.length && !ifEmpty) continue;
    box.appendChild(el("div", "group", title));
    if (!items.length) { box.appendChild(el("p", "empty", ifEmpty)); continue; }
    for (const p of items) {
      const b = el("button", "item");
      b.dataset.id = p.id;
      const row = el("div", "row");
      row.appendChild(el("strong", null, p.name));
      const inst = installedById(modelId(p));
      if (inst) row.appendChild(el("span", "tag ok", "on this PC"));
      else if (p.builtin) row.appendChild(el("span", "tag builtin", "for this PC"));
      if (!p.builtin) {
        const x = el("span", "del", "\u2715");
        x.title = "remove this preset";
        x.onclick = (e) => { e.stopPropagation(); deletePreset(p); };
        row.appendChild(x);
      }
      b.appendChild(row);
      b.appendChild(el("div", "sub", summaryLine(p)));
      b.onclick = () => select(p.id);
      box.appendChild(b);
    }
  }
  markList();
}

function renderStatus() {
  const box = $("status");
  const text = $("status-text");
  const running = (S.running || [])[0];
  const job = S.install;
  box.className = "status";
  if (running) {
    box.classList.add(running.busy ? "busy" : "running");
    text.textContent = running.busy ? `answering a request — ${running.model}` :
      `${running.model} is running on http://127.0.0.1:${running.port}/ (idle)`;
  } else if (S.loading) {
    box.classList.add("loading");
    text.textContent = `loading ${S.loading.model}… (1-3 minutes; the PC can be slow meanwhile)`;
  } else if (job && job.running) {
    box.classList.add("installing");
    text.textContent = `installing ${job.model || ""}: ${job.step || "starting"}` +
      (job.download ? ` — ${job.download.file} ${job.download.percent}%` : "");
  } else if (job && job.error) {
    box.classList.add("failed");
    text.textContent = `the install stopped: ${job.error}`;
  } else if (JOBS.calibrate && JOBS.calibrate.running) {
    box.classList.add("loading");
    text.textContent = "measuring this PC (5-10 minutes, the PC is busy)";
  } else {
    text.textContent = S.summary || "Strata is not running";
  }
  const actions = $("status-actions");
  clear(actions);
  if (running) {
    const open = el("a", null);
    const b = el("button", "primary", `Open the chat page (:${running.port})`);
    open.href = `http://127.0.0.1:${running.port}/`;
    open.target = "_blank";
    open.appendChild(b);
    actions.appendChild(open);
    const stop = el("button", "", "Stop");
    stop.onclick = stopNow;
    actions.appendChild(stop);
  }
  const pc = $("pc");
  const hw = S.hardware;
  if (hw) {
    clear(pc);
    const g = (hw.gpus || []).map((x) => `${x.name.replace(/^NVIDIA |^AMD /, "")} ${x.vram_gb} GB`).join(" + ");
    pc.appendChild(el("span", null, `${hw.os} · `));
    pc.appendChild(el("b", null, g || "no usable GPU"));
    pc.appendChild(el("span", null, ` · ${hw.ram_gb} GB RAM · engine ${S.installed?.engine?.version || "?"}`));
  }
}

function renderInfo() {
  const box = $("info");
  clear(box);
  if (!sel) { box.appendChild(el("p", "empty", "Nothing selected.")); return; }
  const p = sel.preset;
  const fam = familyOf(p.family) || {};
  const info = (MODELS.sizeInfo || {})[p.model] || {};
  const inst = installedById(modelId(p));
  // one column: two .kv rows side by side in this card left a base URL about 110px, and it broke mid-host
  const dl = el("dl", "one");
  const rows = [
    ["Model", `${fam.title || p.family} ${p.model}`],
    ["By", fam.by || ""],
    ["About", info.about || ""],
    ["Download", `${info.download_gb ?? "?"} GB`],
    ["RAM it wants", `${info.ram_needed_gb ?? "?"} GB`],
    ["Experts", `${info.experts_gb ?? "?"} GB`],
    ["On this PC", info.on_this_pc || "not checked"],
    ["Context / KV / images", `${Math.round((p.context || 0) / 1024)}K · ${p.kv || "setup's choice"} · ${p.vision === "gpu" ? "on the GPU" : p.vision === "cpu" ? "on the CPU" : "off"}`],
    ["Address", `http://127.0.0.1:${p.port || 8080}/`],
  ];
  if (inst) {
    rows.push(["Config", inst.config]);
    rows.push(["Start script", inst.start_script || `run-${inst.model}.bat`]);
    rows.push(["Last used", inst.last_used || ""]);
  }
  if (CONNECT) {
    // the same answer the strata_connect_info MCP tool gives, for pointing the user's apps at Strata
    rows.push(["OpenAI base URL", CONNECT.openai_base_url]);
    rows.push(["Anthropic base URL", CONNECT.anthropic_base_url]);
    rows.push(["API key", CONNECT.api_key ? "set" : "none: this PC only"]);
  }
  if (FACTS?.model_settings?.ready === false) rows.push(["Missing files", (FACTS.model_settings.missing_files || []).join(", ")]);
  for (const [k, v] of rows) {
    if (!v) continue;
    const r = el("div", "kv");
    r.appendChild(el("span", null, k));
    r.appendChild(el("b", null, v));
    dl.appendChild(r);
  }
  box.appendChild(dl);
  const rec = MODELS.recommendation;
  if (rec && rec.model && p.model !== rec.model) {
    box.appendChild(el("p", "sub", `setup would pick ${rec.title} for this PC: ${rec.why}`));
  }
}

// ---------------------------------------------------------------- logs and jobs
function renderTabs() {
  const live = {
    install: !!(S.install && S.install.running),
    calibrate: !!(JOBS.calibrate && JOBS.calibrate.running),
    server: !!(S.running || []).length,
    engine: !!(S.running || []).length,
  };
  for (const b of document.querySelectorAll(".tabs button")) {
    b.setAttribute("aria-selected", b.dataset.pane === logTab ? "true" : "false");
    b.classList.toggle("live", !!live[b.dataset.pane]);
    b.title = live[b.dataset.pane] ? `${b.textContent}: something is writing here now` : `${b.textContent}: not writing`;
  }
  const running = (logTab === "install" && live.install) || (logTab === "calibrate" && live.calibrate);
  $("log-cancel").hidden = !running;
}

// the log pane follows the file: it is re-read on every poll, and only scrolls when you were already at the bottom
let LOG_TEXT = "";
async function showLog() {
  const box = $("log");
  renderTabs();
  let text = "", foot = "";
  try {
    if (logTab === "install" || logTab === "calibrate") {
      const job = logTab === "install" ? S.install : JOBS.calibrate;
      const lines = job && (job.lines || job.last_lines);      // the install job names them last_lines
      text = (lines && lines.length) ? (job.running ? "running — " : "") + lines.join("\n")
                                     : ((job && job.result_note) || "Nothing running.");
      foot = job && job.log ? `${job.log} · ${lines ? lines.length : 0} lines · re-read every 3 s` : "";
    } else {
      const r = await api(`logs?source=${logTab}&lines=120`);
      text = (r.lines || []).join("\n") || r.summary || "no log yet";
      foot = `${r.log || ""} · ${(r.lines || []).length} lines · re-read every 3 s`;
    }
  } catch (e) {
    text = e.message || String(e);
  }
  if (text === LOG_TEXT) return;                                // nothing new: leave the scroll where it is
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  LOG_TEXT = text;
  box.textContent = text;
  $("log-foot").textContent = foot;
  if (atBottom) box.scrollTop = box.scrollHeight;
}

// ---------------------------------------------------------------- the loop
async function refresh(hardware) {
  S = await api("state" + (hardware ? "?hardware=1" : ""));
  JOBS = await api("jobs").catch(() => JOBS);
  if (JOBS.calibrate && JOBS.calibrate.then_start) toast(JOBS.calibrate.then_start.summary);
  const port = (S.running[0] || {}).port || null;      // the connection settings change only when a model starts
  if (port !== CONNECT_PORT) {
    CONNECT = port ? await api("connect").catch(() => null) : null;
    CONNECT_PORT = port;
  }
  renderStatus();
  if (CHAT_WAIT && (S.running || []).length) showChatPage(S.running[0].port);
  const jobRunning = !!(S.install && S.install.running) || !!(JOBS.calibrate && JOBS.calibrate.running);
  if (JOB_WAS_RUNNING && !jobRunning) checkDiff();      // an install or a measurement rewrote the model's config
  JOB_WAS_RUNNING = jobRunning;
  renderInstalled();
  renderPresets();
  renderActions();
  renderTabs();
  showLog();                       // every tab, not only the jobs: the Server and Engine panes follow their files
}

async function tick() {
  if (busy) return;
  busy = true;
  try {
    await refresh(false);

  } catch (e) {
    toast(`the launcher lost its own server: ${e.message}`, true);
  } finally {
    busy = false;
  }
}

// ---------------------------------------------------------------- start
async function boot() {
  $("theme").onclick = () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("strata-theme", next);
  };
  $("refresh").onclick = () => refresh(false).catch(fail);
  $("new-preset").onclick = async () => {
    try {
      const r = await api("preset/new");        // the defaults are setup's own answers for this PC
      await openDraft(r.preset, "New preset");
      $("f-name").focus();
    } catch (e) { fail(e); }
  };
  const form = $("form");                        // one listener for the whole form, kept across re-renders
  form.addEventListener("input", () => { collect(); queueDiff(); });
  form.addEventListener("change", () => { collect(); if (sel && sel.preset.family) fillSizes(); queueDiff(); });
  for (const b of document.querySelectorAll(".tabs button")) {
    b.onclick = () => { logTab = b.dataset.pane; showLog(); };
  }
  $("log-cancel").onclick = async () => {
    try {
      if (logTab === "install") await post("install/cancel", {});
      else await post("calibrate/cancel", {});
      await refresh(false);
    } catch (e) { fail(e); }
  };
  try {
    [SCHEMA, MODELS] = await Promise.all([api("schema"), api("models")]);
    await refresh(true);
    const first = (S.presets || [])[0] || (S.builtin || [])[0];
    if (first) await select(first.id);
    else toast("no preset is available: this PC's hardware may not fit any model — see the note in the header");
    timer = setInterval(tick, 3000);
  } catch (e) {
    fail(e);
  }
}
boot();
