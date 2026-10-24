// The server owns preset selection and actual memory allocation. No browser persistence.
(function (root) {
  "use strict";
  const labels = {auto: "Automatic", full: "Full", daily: "Daily", busy: "Busy", off: "Off"};
  const object = value => value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const number = value => typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
  const format = value => value === null ? "—" : value.toLocaleString(undefined, {maximumFractionDigits: 1});
  function normalize(value) {
    const data = object(value);
    if (typeof data.enabled !== "boolean" || !["auto", "full", "daily", "busy"].includes(data.selection)) {
      throw new Error("Resource presets are unavailable on this server.");
    }
    return data;
  }
  function describe(data) {
    if (!data) return {effective: "Waiting for server", targets: "", memory: "", reason: ""};
    const memory = object(data.memory), current = object(memory.current);
    const actual = memory.current ? `Actual: ${format(number(current.resident_budget_gib))} GiB RAM budget · ${format(number(current.vram_reserve_mib))} MiB VRAM reserve` : "Actual allocation not reported";
    const notes = [];
    if (memory.error) notes.push(`Memory update error: ${typeof memory.error === "string" ? memory.error : object(memory.error).message || "update failed"}`);
    if (memory.pending) notes.push("Memory update pending");
    if (memory.limitation || memory.application === "limited") notes.push(`Memory update limited${typeof memory.limitation === "string" ? `: ${memory.limitation}` : ""}`);
    if (data.available === false) notes.push("Preset changes unavailable");
    const allocation = [...notes, actual].join(" · ");
    return {
      effective: data.enabled ? `Effective: ${labels[data.effective] || "—"}` : "Off · configured resource limits",
      targets: data.enabled ? `Target reserves: ${format(number(data.headroom_gib))} GiB RAM · ${format(number(data.vram_reserve_mib))} MiB VRAM` : "",
      memory: allocation,
      reason: typeof data.reason === "string" ? data.reason.replace(/_/g, " ") : "",
    };
  }
  function create({containers, request}) {
    let data = null, saving = false, connected = false, error = "", actionError = false, generation = 0, refreshId = 0, draft = "off";
    const views = containers.map(host => {
      const doc = host.ownerDocument;
      const element = (tag, className, value) => {
        const node = doc.createElement(tag); node.className = className;
        if (value !== undefined) node.textContent = value;
        return node;
      };
      const body = element("div", "resource-control");
      const label = element("label", "resource-control__label", "Resource preset");
      const select = element("select", "resource-control__select st-input");
      select.setAttribute("aria-label", "Resource preset");
      for (const id of ["auto", "full", "daily", "busy", "off"]) {
        const option = element("option", "", labels[id]); option.value = id; select.append(option);
      }
      label.append(select);
      const effective = element("div", "resource-control__effective");
      const targets = element("div", "resource-control__targets");
      const reason = element("div", "resource-control__reason");
      const status = element("div", "resource-control__status");
      status.setAttribute("role", "status"); status.setAttribute("aria-live", "polite");
      status.setAttribute("aria-atomic", "true");
      body.append(label, effective, targets, reason, status); host.replaceChildren(body);
      select.onchange = () => selectPreset(select.value);
      return {body, select, effective, targets, reason, status};
    });
    function text(node, value) { if (node.textContent !== value) node.textContent = value; }
    function render() {
      const view = describe(data);
      for (const nodes of views) {
        nodes.select.value = saving ? draft : data?.enabled ? data.selection : "off";
        nodes.select.disabled = saving || !connected || !data || data.available === false;
        const memory = object(data?.memory);
        nodes.body.dataset.state = saving ? "saving" : error ? "error" : !connected ? "offline"
          : memory.error ? "error" : memory.pending || memory.limitation ? "pending" : "ready";
        nodes.body.setAttribute("aria-busy", String(saving));
        text(nodes.effective, data && !connected ? `Last reported · ${view.effective}` : view.effective);
        text(nodes.targets, view.targets);
        nodes.targets.hidden = !view.targets;
        text(nodes.reason, view.reason); nodes.reason.hidden = !view.reason;
        const allocation = !connected ? view.memory.replace("Actual:", "Last reported:") : view.memory;
        text(nodes.status, saving ? "Selecting preset…" : [error, allocation].filter(Boolean).join(" · "));
      }
    }
    function update(value, version = generation) {
      if (saving || version !== generation) return;
      try { data = normalize(value); connected = true; if (!actionError) error = ""; }
      catch (failure) { connected = false; error = failure.message; actionError = false; }
      render();
    }
    function offline(version = generation) {
      if (saving || version !== generation) return;
      connected = false; error = "Server unavailable · selection cannot be changed"; actionError = false; render();
    }
    async function refresh({keepError = false} = {}) {
      if (saving) return;
      const version = generation, id = ++refreshId;
      try {
        const value = await request("v1/resources");
        if (saving || version !== generation || id !== refreshId) return;
        if (!keepError) { error = ""; actionError = false; }
        update(value, version);
      } catch (failure) {
        if (saving || version !== generation || id !== refreshId) return;
        connected = false;
        if (!keepError) { error = failure.message || "Resource presets could not be refreshed."; actionError = false; }
        render();
      }
    }
    async function selectPreset(selection) {
      if (saving || !connected || !data || data.available === false || !Object.hasOwn(labels, selection)) { render(); return false; }
      generation++; saving = true; draft = selection; error = ""; actionError = false; render();
      try {
        const value = await request("v1/resources", {enabled: selection !== "off", selection: selection === "off" ? "auto" : selection});
        data = normalize(value); connected = true;
        return true;
      } catch (failure) {
        error = `Selection not confirmed: ${failure.message || "request failed"}`; actionError = true;
        return false;
      } finally {
        generation++; // Also reject polls started during POST but delivered after its response.
        saving = false; render();
        // A failed response may still have reached the server; read back its authoritative selection.
        if (error) await refresh({keepError: true});
      }
    }
    render();
    return {update, refresh, offline, select: selectPreset, version: () => generation};
  }
  const api = {create, describe, normalize};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.StrataResourceControls = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
