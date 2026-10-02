// serve/web/pool.js - the Pool tab: this PC's role, the pool's workers, the secret, and the layer map.
// Uses app.js's helpers ($, esc, icon, toast, copyText, headers, fmt). Everything a peer or a beacon says about itself
// is untrusted text from the network: it only ever reaches the page through esc() / textContent.
"use strict";

(() => {
  const ROLE_HELP = {
    off: "This PC runs the model alone, as Strata does.",
    router: "Every PC runs the whole model on its own; a request that reaches any of them runs on the PC that holds its " +
      "conversation, else on an idle one. Two chats at once run on two PCs. One chat is as fast as one PC.",
    coordinator: "This PC runs the first layers, the output and the draft layer, and chats; the workers below run the " +
      "rest of the layers on their GPUs. Their GPUs and RAM add to this PC's: more experts on GPUs, the context " +
      "divided between the PCs.",
    worker: "This PC lends its GPU and RAM to a coordinator: it runs the layers the coordinator assigns. Its chat is " +
      "off meanwhile. Give the coordinator this PC's address and the pool secret.",
  };
  const STATE_TEXT = {
    starting: "Starting", "loading weights": "Starting", waiting: "Waiting for a coordinator", loading: "Loading its layers",
    connected: "Coordinator connected", serving: "Serving", idle: "Idle (layers loaded)", reloading: "Reloading",
    error: "Error", stopped: "Stopped", running: "Running",
  };
  let state = null;          // GET /pool
  let draft = null;          // the form
  let dirty = false;
  let timer = null;
  let showSecret = false;
  let busy = false;

  const clone = (o) => JSON.parse(JSON.stringify(o));
  const ROLE_NAME = {off: "Alone", router: "Sharing", coordinator: "Coordinator", worker: "Worker"};
  const gib = (mib) => (mib == null ? "–" : `${fmt(mib / 1024, mib >= 10240 ? 0 : 1)} GB`);

  async function load() {
    try {
      const r = await fetch("pool", {headers: headers()});
      if (!r.ok) throw new Error(r.status === 401 ? "the API key is missing or wrong (About tab)" : `HTTP ${r.status}`);
      state = await r.json();
      if (!dirty || !draft) {
        draft = clone(state.config);
        dirty = false;
      }
      render();
    } catch (e) {
      notice("error", `Cannot read the pool's state: ${e.message}`);
    }
  }

  function start() {
    if (timer) return;
    load();
    timer = setInterval(load, 2500);
  }
  function stop() {
    clearInterval(timer);
    timer = null;
  }
  window.addEventListener("strata:tab", (e) => (e.detail === "pool" ? start() : stop()));
  if (location.hash === "#pool") start();

  // ---------------------------------------------------------------- rendering
  function notice(tone, text) {
    const n = $("pool-notice");
    n.hidden = !text;
    n.dataset.tone = tone;
    n.textContent = text || "";
  }

  function setSeg(id, value) {
    for (const b of $(id).querySelectorAll("button")) b.setAttribute("aria-checked", String(b.dataset.v === value));
  }

  function nodesForMap() {
    // [{name, lb, le, slots}] in layer order: what is running when the engine says so, else what the form implies
    const s = state, d = draft, L = 48;
    const me = (s.node && s.node.name) || "This PC";
    const pool = s.engine && s.engine.pool;
    if (d.role === "coordinator" && pool && s.config.role === "coordinator" && !dirty) {
      const out = [{name: `${me} (coordinator)`, lb: 0, le: pool.le, slots: pool.slots_here}];
      for (const w of pool.workers) {
        const peer = (s.peers || []).find((p) => p.addr === w.addr);
        const nm = (peer && peer.facts && peer.facts.name) || peer?.label || w.addr;
        out.push({name: nm, lb: w.lb, le: w.le, slots: w.slots});
      }
      return {nodes: out, live: true};
    }
    if (d.role === "coordinator") {
      const peers = d.peers.filter((p) => p.enabled);
      const names = [`${me} (coordinator)`, ...peers.map((p) => p.label || p.addr)];
      if (d.split !== "auto" && /^\d+(,\d+)*$/.test(d.split)) {
        const ks = d.split.split(",").map(Number);
        if (ks.length === peers.length) {
          const b = [0, ...ks, L];
          return {nodes: names.map((n, i) => ({name: n, lb: b[i], le: b[i + 1]})), live: false};
        }
      }
      return {nodes: names.map((n) => ({name: n})), live: false, auto: true};
    }
    if (d.role === "worker") {
      const w = s.worker || {};
      if (w.lb != null && w.le) {
        return {nodes: [{name: w.coordinator ? `coordinator ${w.coordinator}` : "the coordinator", lb: 0, le: w.lb},
                        {name: `${me} (this worker)`, lb: w.lb, le: w.le, slots: w.ready && w.ready.slots}], live: true, me: 1};
      }
      return {nodes: [{name: `${me} (this worker)`}], live: false, auto: true};
    }
    return {nodes: [{name: me, lb: 0, le: L, slots: s.engine && s.engine.expert_slots}], live: true};
  }

  function renderMap() {
    const {nodes, live, auto, me} = nodesForMap();
    const cells = [];
    for (let l = 0; l < 48; ++l) {
      const i = nodes.findIndex((n) => n.lb != null && l >= n.lb && l < n.le);
      const n = nodes[i];
      cells.push(`<span data-node="${i < 0 ? "" : me && i === 0 ? "" : i}" title="layer ${l}${n ? " · " + esc(n.name) : ""}"></span>`);
    }
    $("pool-map").innerHTML = cells.join("");
    $("pool-legend").innerHTML = nodes.map((n, i) => {
      const range = n.lb != null ? `layers ${n.lb}–${n.le - 1}` : "layers: decided when it starts";
      const slots = n.slots != null ? ` · ${fmt(n.slots)} experts in VRAM` : "";
      const color = ["var(--st-accent)", "var(--st-info)", "var(--st-warn)", "var(--st-danger)"][me && i === 0 ? 4 : i] || "var(--st-ink-muted)";
      return `<span><i style="background:${color}"></i>${esc(n.name)} · ${range}${slots}</span>`;
    }).join("");
    const total = nodes.reduce((a, n) => a + (n.slots || 0), 0);
    $("pool-map-sum").textContent = auto ? "Automatic: placed when the model starts" :
      live && total ? `${fmt(total)} experts on GPUs across the pool` : live ? "" : "as configured (not running yet)";
  }

  function factsHTML(rows) {
    return rows.filter(([, v]) => v != null && v !== "").map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("");
  }

  function renderNode() {
    const n = state.node || {};
    const addrs = (state.addresses || []).map((a) => `<code>${esc(a)}</code>`).join(", ");
    $("pool-node").innerHTML = factsHTML([
      ["Name", esc(n.name || "")],
      ["GPU", esc(n.gpu || "–")],
      ["VRAM / RAM", `${gib(n.vram_mib)} / ${gib(n.ram_mib)}`],
      ["Model", esc(n.model || "–")],
      ["Address", addrs || '<span class="muted">this PC only (start with host 0.0.0.0 to reach it from others)</span>'],
      ["Worker port", `<code>${esc(String(n.worker_port))}</code>`],
      ["Version", esc(n.version || "–")],
    ]);
  }

  function peerState(p) {
    const f = p.facts;
    if (!p.enabled) return '<span class="st-badge">Not used</span>';
    if (!f) return p.reachable ? '<span class="st-badge">Seen</span>' :
      '<span class="st-badge st-badge--error" title="Its Strata page does not answer; the engine may still reach it">No answer</span>';
    if (f.role !== "worker") return `<span class="st-badge st-badge--queued" title="Set its role to Worker in its Pool tab">Not a worker (${esc(f.role || "?")})</span>`;
    const t = STATE_TEXT[f.state] || f.state || "?";
    const tone = f.state === "serving" || f.state === "connected" ? "generating" : f.state === "error" ? "error" :
      f.state === "loading" || f.state === "reloading" ? "reading" : "";
    return `<span class="st-badge ${tone ? "st-badge--" + tone : ""}">${esc(t)}</span>`;
  }

  function renderPeers() {
    const rows = draft.peers.map((p, i) => {
      const live = (state.peers || []).find((x) => x.addr === p.addr) || {};
      const f = live.facts || null;
      const gpu = f ? `${esc(f.gpu || "?")}${f.vram_mib ? ` · ${gib(f.vram_mib)}` : ""}${f.ram_mib ? ` · ${gib(f.ram_mib)} RAM` : ""}` : "–";
      const pool = state.engine && state.engine.pool;
      const w = pool && pool.workers.find((x) => x.addr === p.addr);
      const layers = w ? `${w.lb}–${w.le - 1}` : f && f.layers ? esc(f.layers) : "–";
      return `<tr data-i="${i}">
        <td class="addr">${esc(p.addr)}</td>
        <td>${esc((f && f.name) || p.label || "–")}</td>
        <td>${gpu}</td>
        <td>${peerState({...p, ...live})}</td>
        <td class="num">${layers}</td>
        <td><button class="st-toggle" data-act="toggle" role="switch" aria-checked="${p.enabled}" aria-label="Use this worker"></button></td>
        <td class="acts">
          <button class="st-btn st-btn--icon" data-act="up" title="Earlier layers" aria-label="Move up" ${i === 0 ? "disabled" : ""}>${icon("chevron", "st-icon up")}</button>
          <button class="st-btn st-btn--icon" data-act="down" title="Later layers" aria-label="Move down" ${i === draft.peers.length - 1 ? "disabled" : ""}>${icon("chevron")}</button>
          <button class="st-btn st-btn--icon" data-act="remove" title="Remove" aria-label="Remove">${icon("trash")}</button>
        </td></tr>`;
    });
    $("pool-peers").innerHTML = rows.length ? rows.join("") :
      '<tr><td colspan="7" class="muted">No workers yet: add one below, or pick one the network shows.</td></tr>';
    // what the network shows, minus the ones already listed
    const have = new Set(draft.peers.map((p) => p.addr));
    const found = (state.discovered || []).filter((d) => d.fresh && !have.has(`${d.ip}:${d.worker_port}`));
    const disc = state.discovery || {};
    $("pool-found").innerHTML = found.length ? found.map((d, i) =>
      `<span class="found-chip"><span><b>${esc(d.name || d.ip)}</b> <span class="muted">${esc(d.ip)} · ${esc(d.gpu || "?")} · ` +
      `${esc(STATE_TEXT[d.state] || d.role || "")}${d.role !== "worker" ? ` (${esc(d.role || "?")})` : ""}</span></span>` +
      `<button class="st-btn st-btn--secondary" data-found="${i}" type="button">Add</button></span>`).join("") :
      `<span class="muted small">${disc.error ? esc(disc.error) : disc.on ? "No other Strata PC with a pool heard yet (their Pool tab sends a beacon every few seconds; the firewall must let UDP 7702 through)." : "Discovery is off."}</span>`;
    for (const b of $("pool-found").querySelectorAll("button[data-found]")) {
      const d = found[Number(b.dataset.found)];
      b.onclick = () => {
        draft.peers.push({addr: `${d.ip}:${d.worker_port || 7701}`, enabled: true, label: d.name || "", http: `${d.ip}:${d.http_port}`});
        touch();
      };
    }
  }

  function routeState(p, live, f) {
    if (!p.enabled) return '<span class="st-badge">Not used</span>';
    if (!live || !live.checked) return '<span class="st-badge">Checking</span>';
    if (!live.reachable) return `<span class="st-badge st-badge--error" title="${esc(live.why || "")}">No answer</span>`;
    if (!live.usable) return `<span class="st-badge st-badge--queued" title="${esc(live.why || "")}">${esc(live.why || "Not usable")}</span>`;
    if (live.inflight) return '<span class="st-badge st-badge--generating">Running a request from here</span>';
    if (f && f.busy) return '<span class="st-badge st-badge--reading">Busy</span>';
    if (f && !f.loaded) return '<span class="st-badge">Model unloaded</span>';
    return '<span class="st-badge st-badge--generating">Ready</span>';
  }

  function renderRoute() {
    const rt = state.router || {peers: {}, counts: {}, recent: []};
    const me = state.node || {};
    const rows = draft.route_peers.map((p, i) => {
      const live = rt.peers[p.addr];
      const f = live && live.facts;
      const gpu = f ? `${esc(f.gpu || "?")}${f.vram_mib ? ` · ${gib(f.vram_mib)}` : ""}` : "–";
      let warn = "";
      if (f && me.settings && f.settings && f.settings !== me.settings)
        warn = ' <span class="st-badge st-badge--queued" title="Its sampling / thinking defaults differ from this PC\'s: a request that runs there gets its defaults">other defaults</span>';
      if (f && f.primary) warn += ' <span class="st-badge">primary</span>';
      return `<tr data-i="${i}">
        <td class="addr">${esc(p.addr)}</td>
        <td>${esc((f && f.name) || p.label || "–")}</td>
        <td>${gpu}</td>
        <td>${routeState(p, live, f)}${warn}</td>
        <td class="num">${f && f.max_context ? fmt(f.max_context) : "–"}</td>
        <td><button class="st-toggle" data-act="toggle" role="switch" aria-checked="${p.enabled}" aria-label="Use this PC"></button></td>
        <td class="acts"><button class="st-btn st-btn--icon" data-act="remove" title="Remove" aria-label="Remove">${icon("trash")}</button></td></tr>`;
    });
    $("pool-route-peers").innerHTML = rows.length ? rows.join("") :
      '<tr><td colspan="7" class="muted">No other PC yet: add one below, or pick one the network shows.</td></tr>';
    const have = new Set(draft.route_peers.map((p) => p.addr));
    const found = (state.discovered || []).filter((d) => d.fresh && d.http_port && !have.has(`${d.ip}:${d.http_port}`));
    const disc = state.discovery || {};
    $("pool-route-found").innerHTML = found.length ? found.map((d, i) =>
      `<span class="found-chip"><span><b>${esc(d.name || d.ip)}</b> <span class="muted">${esc(d.ip)}:${esc(String(d.http_port))} · ` +
      `${esc(d.gpu || "?")} · ${esc(ROLE_NAME[d.role] || d.role || "")}</span></span>` +
      `<button class="st-btn st-btn--secondary" data-found="${i}" type="button">Add</button></span>`).join("") :
      `<span class="muted small">${disc.error ? esc(disc.error) : disc.on ? "No other Strata PC with a pool heard yet (the firewall must let UDP 7702 through)." : "Discovery is off."}</span>`;
    for (const b of $("pool-route-found").querySelectorAll("button[data-found]")) {
      const d = found[Number(b.dataset.found)];
      b.onclick = () => { draft.route_peers.push({addr: `${d.ip}:${d.http_port}`, enabled: true, label: d.name || ""}); touch(); };
    }
    $("pool-primary").setAttribute("aria-checked", String(!!draft.primary));
    if (document.activeElement !== $("pool-move")) $("pool-move").value = draft.move_tokens;
    const c = rt.counts || {};
    const recent = (rt.recent || []).slice(-5).reverse();
    $("pool-route-sum").textContent = state.config.role === "router" ?
      (rt.islanded ? "on its own: no other PC answers" : "sharing") : "";
    $("pool-route-stats").innerHTML = state.config.role !== "router" ? "" : factsHTML([
      ["Requests here", `${fmt(c.local || 0)} of its own, ${fmt(c.routed_in || 0)} from other PCs`],
      ["Sent to others", `${fmt(c.routed_out || 0)}${c.fallback ? ` (${fmt(c.fallback)} came back: the other PC was busy or gone)` : ""}`],
      ["Context here", me.max_context ? `${fmt(me.max_context)} tokens` : "–"],
      ["Last decisions", recent.length ? `<ol class="route-recent">${recent.map((r) =>
        `<li>${esc(r.to)} · ~${fmt(r.tokens)} tokens · ${esc(r.why)}</li>`).join("")}</ol>` : null],
    ]);
  }

  function renderWorker() {
    const w = state.worker || {};
    const r = w.ready || {};
    const last = w.last_request;
    $("pool-worker").innerHTML = factsHTML([
      ["State", esc(STATE_TEXT[w.state] || w.state || "–")],
      ["Coordinator", w.coordinator ? `<code>${esc(w.coordinator)}</code>` : '<span class="muted">none connected</span>'],
      ["Layers", w.lb != null && w.le ? `${w.lb}–${w.le - 1}` : "–"],
      ["Experts", r.slots != null ? `${fmt(r.slots)} in VRAM (${gib(r.cache_mib)}), ${gib(r.arena_mib)} in RAM` : "–"],
      ["Free VRAM", r.vram_free_mib != null ? gib(r.vram_free_mib) : null],
      ["Requests", fmt(w.requests || 0)],
      ["Last request", last ? `${fmt(last.windows)} windows, ${fmt(last.ms_verify / Math.max(1, last.windows), 1)} ms each here; ` +
        `${fmt(last.prompt)} prompt tokens; hit rate ${last.lookups ? fmt(100 * last.hits / last.lookups, 1) : "–"}%` : null],
      ["Refused", w.refused ? `<code>${esc(w.refused.from)}</code> (another secret)` : null],
      ["Error", w.error ? esc(w.error) : null],
    ]);
    $("pool-worker-sum").textContent = w.port ? `listening on port ${w.port}` : "";
  }

  function render() {
    if (!state || !draft) return;
    setSeg("pool-role", draft.role);
    $("pool-role-help").textContent = ROLE_HELP[draft.role] || "";
    // the badge: what this PC is doing now
    const n = state.node || {};
    const badge = $("pool-state-badge");
    const rt = state.router;
    badge.textContent = `${ROLE_NAME[state.config.role] || state.config.role}${rt && rt.islanded ? " (on its own)" : ""} · ${STATE_TEXT[n.state] || n.state || "–"}`;
    badge.className = "st-badge" + (n.state === "serving" || n.state === "running" ? " st-badge--generating" :
      n.state === "error" ? " st-badge--error" : n.state === "loading" || n.state === "starting" ? " st-badge--reading" : "");
    // notices: an apply in progress, the last error, what the run config cannot do in a pool
    if (state.applying) notice("info", `Applying: ${state.applying} …`);
    else if (state.error) notice("error", state.error);
    else if (draft.role !== "off" && (state.problems || []).length) notice("warn", state.problems.join("; "));
    else if (draft.role === "coordinator" && !draft.peers.some((p) => p.enabled)) notice("warn", "Add at least one worker.");
    else if (draft.role === "router" && !draft.route_peers.some((p) => p.enabled)) notice("warn", "Add at least one other PC.");
    else if (draft.role === "router" && rt && rt.islanded && state.config.role === "router")
      notice("info", "No other PC answers right now: this PC serves its own requests and keeps looking for them.");
    else notice("", "");
    $("pool-map-card").hidden = draft.role === "router";
    if (draft.role !== "router") renderMap();
    renderNode();
    // secret and identity (not while the user types in them)
    const sec = $("pool-secret");
    if (document.activeElement !== sec) sec.value = draft.secret || "";
    sec.type = showSecret ? "text" : "password";
    if (document.activeElement !== $("pool-name")) $("pool-name").value = draft.name || "";
    if (document.activeElement !== $("pool-port")) $("pool-port").value = draft.worker_port;
    $("pool-coord-card").hidden = draft.role !== "coordinator";
    $("pool-route-card").hidden = draft.role !== "router";
    if (draft.role === "router") renderRoute();
    $("pool-worker-card").hidden = draft.role !== "worker" || state.config.role !== "worker";
    if (draft.role === "coordinator") {
      renderPeers();
      const manual = draft.split !== "auto";
      setSeg("pool-split-mode", manual ? "manual" : "auto");
      $("pool-split").hidden = !manual;
      if (document.activeElement !== $("pool-split")) $("pool-split").value = manual ? draft.split : "";
      setSeg("pool-wire", draft.wire);
      const pool = state.engine && state.engine.pool;
      $("pool-split-out").textContent = pool && pool.workers.length ? `now ${pool.workers.map((w) => w.lb).join(",")}` : "";
    }
    if (state.config.role === "worker") renderWorker();
    $("pool-dirty").textContent = dirty ? "Not applied yet: Apply restarts the model with these settings." : "";
    $("pool-apply").disabled = busy || !!state.applying;
    $("pool-restart").disabled = busy || !!state.applying || state.config.role === "off" || state.config.role === "router";
  }

  // ---------------------------------------------------------------- editing
  function touch() {
    dirty = true;
    render();
  }

  for (const b of $("pool-role").querySelectorAll("button")) b.onclick = () => { if (draft) { draft.role = b.dataset.v; touch(); } };
  for (const b of $("pool-split-mode").querySelectorAll("button")) b.onclick = () => {
    if (!draft) return;
    if (b.dataset.v === "auto") draft.split = "auto";
    else {
      const n = Math.max(1, draft.peers.filter((p) => p.enabled).length);
      draft.split = draft.split !== "auto" ? draft.split : Array.from({length: n}, (_, i) => Math.round(48 * (i + 1) / (n + 1))).join(",");
    }
    touch();
  };
  for (const b of $("pool-wire").querySelectorAll("button")) b.onclick = () => { if (draft) { draft.wire = b.dataset.v; touch(); } };
  $("pool-split").oninput = () => { draft.split = $("pool-split").value.replace(/\s/g, "") || "auto"; dirty = true; renderMap(); };
  $("pool-name").oninput = () => { draft.name = $("pool-name").value; dirty = true; };
  $("pool-port").oninput = () => { draft.worker_port = Number($("pool-port").value) || 7701; dirty = true; };
  $("pool-secret").oninput = () => { draft.secret = $("pool-secret").value; dirty = true; };
  $("pool-secret-show").onclick = () => { showSecret = !showSecret; render(); };
  $("pool-secret-copy").onclick = () => {
    if (!draft.secret) { toast("warn", "The secret is hidden here", "Open this page on the PC itself to copy it."); return; }
    copyText(draft.secret, $("pool-secret-copy"));
  };
  $("pool-secret-new").onclick = () => {
    const a = new Uint8Array(18);
    crypto.getRandomValues(a);
    draft.secret = btoa(String.fromCharCode(...a)).replace(/[+/=]/g, (c) => ({"+": "-", "/": "_", "=": ""}[c]));
    showSecret = true;
    touch();
    toast("info", "New secret", "Apply, then paste it into every other PC of the pool.");
  };
  $("pool-add").onsubmit = (e) => {
    e.preventDefault();
    const v = $("pool-add-addr").value.trim();
    if (!v) return;
    if (!/^(\[[0-9a-fA-F:.]+\]|[A-Za-z0-9._-]+)(:\d{1,5})?$/.test(v)) { toast("error", "Not an address", "host, or host:port"); return; }
    const addr = v.includes(":") && !v.startsWith("[") ? v : `${v}:${draft.worker_port || 7701}`;
    if (draft.peers.some((p) => p.addr === addr)) { toast("warn", "Already listed", addr); return; }
    if (draft.peers.length >= 7) { toast("warn", "Seven workers at most"); return; }
    draft.peers.push({addr, enabled: true, label: "", http: null});
    $("pool-add-addr").value = "";
    touch();
  };
  $("pool-route-add").onsubmit = (e) => {
    e.preventDefault();
    const v = $("pool-route-addr").value.trim();
    if (!v) return;
    if (!/^(\[[0-9a-fA-F:.]+\]|[A-Za-z0-9._-]+)(:\d{1,5})?$/.test(v)) { toast("error", "Not an address", "host, or host:port"); return; }
    const addr = v.includes(":") && !v.startsWith("[") ? v : `${v}:8080`;
    if (draft.route_peers.some((p) => p.addr === addr)) { toast("warn", "Already listed", addr); return; }
    if (draft.route_peers.length >= 7) { toast("warn", "Seven PCs at most"); return; }
    draft.route_peers.push({addr, enabled: true, label: ""});
    $("pool-route-addr").value = "";
    touch();
  };
  $("pool-route-peers").onclick = (e) => {
    const b = e.target.closest("button[data-act]");
    if (!b) return;
    const i = Number(b.closest("tr").dataset.i);
    if (b.dataset.act === "toggle") draft.route_peers[i].enabled = !draft.route_peers[i].enabled;
    else if (b.dataset.act === "remove") draft.route_peers.splice(i, 1);
    touch();
  };
  $("pool-primary").onclick = () => { if (draft) { draft.primary = !draft.primary; touch(); } };
  $("pool-move").oninput = () => { draft.move_tokens = Math.max(0, Math.round(Number($("pool-move").value) || 0)); dirty = true; };
  $("pool-peers").onclick = (e) => {
    const b = e.target.closest("button[data-act]");
    if (!b) return;
    const i = Number(b.closest("tr").dataset.i);
    const ps = draft.peers;
    if (b.dataset.act === "toggle") ps[i].enabled = !ps[i].enabled;
    else if (b.dataset.act === "remove") ps.splice(i, 1);
    else if (b.dataset.act === "up" && i > 0) [ps[i - 1], ps[i]] = [ps[i], ps[i - 1]];
    else if (b.dataset.act === "down" && i + 1 < ps.length) [ps[i + 1], ps[i]] = [ps[i], ps[i + 1]];
    touch();
  };

  async function post(path, body) {
    const r = await fetch(path, {method: "POST", headers: headers(true), body: JSON.stringify(body || {})});
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error((j.error && j.error.message) || `HTTP ${r.status}`);
    return j;
  }

  $("pool-apply").onclick = async () => {
    if (!draft) return;
    const leavingChat = state.config.role !== "worker" && draft.role === "worker";
    if (leavingChat && !confirm("This PC becomes a worker: its model stops here and the chat on this page is off until " +
                                "you switch back. Continue?")) return;
    busy = true;
    render();
    try {
      const body = clone(draft);
      if (!body.secret) delete body.secret;           // hidden here: the server keeps its own
      await post("pool/config", body);
      dirty = false;
      toast("success", "Applied", draft.role === "off" ? "This PC runs the model alone." :
        draft.role === "router" ? "This PC shares requests with the PCs listed." :
        draft.role === "worker" ? "This PC now waits for a coordinator." : "The model restarts with the workers.");
    } catch (e) {
      toast("error", "Not applied", e.message, 6000);
    } finally {
      busy = false;
      load();
    }
  };
  $("pool-restart").onclick = async () => {
    busy = true;
    render();
    try {
      await post("pool/restart", {});
      toast("info", "Restarting the engine");
    } catch (e) {
      toast("error", "Could not restart", e.message, 6000);
    } finally {
      busy = false;
      load();
    }
  };
})();
