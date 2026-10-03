// Named model selection; the page and its conversation stay open while the local server restarts.
"use strict";
(() => {
  const select = $("model-select"), status = $("model-switch-status"), wrapper = $("model-selector");
  let items = [], current = null, target = null, switchingSince = 0, lastOperation = null, polling = false;
  let requesting = false, expectedOperation = null;
  let serverBusy = false, canSwitch = false;
  const label = (id) => items.find((m) => m.id === id)?.name || id || "";
  function sync() {
    wrapper.hidden = window.StrataModelFamily !== "native";
    select.disabled = modelSwitching || Boolean(busy) || serverBusy || !canSwitch || wrapper.hidden;
  }
  function switching(on) {
    const changed = modelSwitching !== on;
    modelSwitching = on;
    if (on && !switchingSince) switchingSince = Date.now();
    if (!on) switchingSince = 0;
    if (changed) {
      if (on) { clearTimeout(contextTimer); contextRevision++; }
      setBusy(Boolean(busy));
    }
  }
  function render(data) {
    if (window.StrataProviderSwitching) return;
    if (requesting || (expectedOperation && data.switch?.id !== expectedOperation)) return;
    const signature = JSON.stringify(data.models);
    if (signature !== JSON.stringify(items)) {
      items = data.models;
      select.replaceChildren(...items.map((m) => {
        const option = document.createElement("option");
        option.value = m.id;
        option.textContent = m.name + (m.available ? "" : " (not installed)");
        option.disabled = !m.available;
        return option;
      }));
    }
    current = data.current;
    serverBusy = Boolean(data.busy); canSwitch = data.can_switch;
    const operation = data.switch || {};
    const pending = ["starting", "restoring"].includes(operation.status);
    const wasSwitching = modelSwitching;
    target = pending ? operation.target : null;
    switching(pending);
    select.value = target || current || "";
    sync();
    status.textContent = pending
      ? (operation.status === "restoring" ? "Restoring the previous model…" : "Loading…")
      : data.busy ? "Generating a response" : !data.can_switch ? "Start Strata with its launcher"
      : !data.loaded ? "Loads when you send a message" : "";
    if (!pending && wasSwitching) {
      loadHealth().then(loadMcp);
      if (operation.status === "ready") toast("success", "Switched to " + label(current));
    }
    if (operation.status === "failed" && operation.id !== lastOperation) {
      toast("error", "Could not switch the model", operation.message || "Check the server status using its launcher.", 9000);
    }
    if (!pending) { lastOperation = operation.id; expectedOperation = null; }
  }
  async function poll() {
    if (polling) return;
    polling = true;
    try {
      const response = await fetch("api/local-models", {headers: headers(), signal: AbortSignal.timeout(4000)});
      if (response.status === 404) { canSwitch = false; sync(); switching(false); return; }
      if (!response.ok) throw new Error("Could not retrieve the model list");
      render(await response.json());
    } catch (e) {
      if (modelSwitching) {
        select.disabled = true;
        status.textContent = "Loading… Waiting to reconnect";
        if (switchingSince && Date.now() - switchingSince > 15 * 60 * 1000) {
          switching(false);
          status.textContent = "Cannot connect. Check the server status using its launcher.";
        }
      } else if (!wrapper.hidden) {
        select.disabled = true;
        status.textContent = "Checking the connection…";
      }
    } finally { polling = false; }
  }
  select.addEventListener("change", async () => {
    const chosen = select.value;
    if (window.StrataModelFamily !== "native" || busy || modelSwitching || chosen === current) { select.value = current || ""; return; }
    target = chosen;
    requesting = true;
    switching(true);
    select.disabled = true;
    status.textContent = "Loading " + label(chosen) + "…";
    try {
      const response = await fetch("api/local-models/switch", {
        method: "POST", headers: headers(true), body: JSON.stringify({model: chosen}),
        signal: AbortSignal.timeout(10000),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error?.message || "Could not switch the model.");
      expectedOperation = result.id || null;
      if (result.status === "current") switching(false);
    } catch (e) {
      // An interrupted response may still have started the worker. Poll before accepting another selection.
      if (e.name !== "TimeoutError" && e.name !== "TypeError") {
        switching(false);
        select.value = current || "";
        toast("error", "Could not switch the model", e.message, 7000);
      }
    }
    requesting = false;
    poll();
  });
  window.StrataNativeModels = {refresh: poll, sync};
  poll();
  setInterval(poll, 2000);
})();
