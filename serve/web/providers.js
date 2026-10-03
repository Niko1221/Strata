// Local, OpenAI-compatible model connections. Conversation storage stays at this origin.
"use strict";
(() => {
  const select = $("provider-select"), wrapper = $("provider-selector"), result = $("provider-result");
  const family = $("model-family-select"), controls = $("model-controls"), status = $("provider-switch-status");
  let current, changing = false, polling = false, signature = "", profiles = [], revision = 0, connected = false;
  let lastAdditional = store.get("lastAdditionalProvider", null);
  window.StrataModelFamily = null;
  function sync() {
    const additional = !!current;
    window.StrataModelFamily = current === undefined ? null : additional ? "additional" : "native";
    controls.hidden = false;
    family.value = window.StrataModelFamily || "";
    family.disabled = !connected || changing || Boolean(busy) || modelSwitching;
    wrapper.hidden = !additional;
    select.value = current || "";
    select.disabled = family.disabled;
    $("provider-add").disabled = family.disabled;
    $("model-selector").hidden = window.StrataModelFamily !== "native";
    window.StrataNativeModels?.sync();
  }
  const presets = {
    ollama: {name: "Qwen / Ollama", url: "http://127.0.0.1:11434/v1", model: "qwen3.5:4b", context: 32768},
    bonsai: {name: "Bonsai 2 27B", url: "http://127.0.0.1:8082/v1", model: "bonsai2-27b", context: 16384},
    custom: {name: "Additional model", url: "http://127.0.0.1:8082/v1", model: "", context: 32768},
  };
  async function request(path, body, timeout = 15000) {
    const response = await fetch(path, {
      method: body === undefined ? "GET" : "POST", headers: headers(body !== undefined),
      body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(timeout),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || `HTTP ${response.status}`);
    return data;
  }
  function render(data) {
    if (changing) return;
    const changed = current !== data.current;
    current = data.current;
    profiles = data.providers;
    connected = true;
    if (current) { lastAdditional = current; store.set("lastAdditionalProvider", current); }
    if (changed) loadHealth().then(loadMcp);
    const next = JSON.stringify(data.providers.map(({id, name}) => ({id, name})));
    if (signature !== next) {
      signature = next;
      select.replaceChildren(...data.providers.map((provider) => {
        const option = document.createElement("option");
        option.value = provider.id;
        option.textContent = provider.name;
        return option;
      }));
    }
    status.textContent = "";
    sync();
    if (changed) window.StrataNativeModels?.refresh();
  }
  async function pollProviders() {
    if (polling || changing) return;
    polling = true;
    const ticket = revision;
    try { const data = await request("api/providers"); if (ticket === revision) render(data); }
    catch (error) {
      if (ticket === revision && !changing) { connected = false; status.textContent = "Check the connection to the server"; sync(); }
    }
    finally { polling = false; }
  }
  async function choose(id) {
    if (busy || modelSwitching || changing) throw new Error("Wait for the response or model loading to finish before switching.");
    id = id || null;
    if (id === current) { sync(); return; }
    revision++;
    changing = true; window.StrataProviderSwitching = true; modelSwitching = true;
    status.textContent = id ? `Switching to ${profiles.find((p) => p.id === id)?.name || "additional model"}…` : "Switching to Strata models…";
    setBusy(Boolean(busy)); sync();
    try {
      try { await request("api/providers/select", {id: id || null}, 45000); }
      catch (error) {
        // A timed out POST may have committed. Read the authoritative selection before reporting failure.
        const state = await request("api/providers", undefined, 45000);
        if (state.current !== (id || null)) throw error;
      }
      const confirmed = await request("api/providers", undefined, 45000);
      current = confirmed.current; profiles = confirmed.providers; connected = true;
      if (current) { lastAdditional = current; store.set("lastAdditionalProvider", current); }
      await loadHealth(); await loadMcp();
      if (current !== id) throw new Error("Another window changed the model. Check the current selection.");
      sync(); status.textContent = "";
      toast("success", id ? "Switched to an additional model" : "Switched to Strata models", health.model);
    } finally {
      changing = false; window.StrataProviderSwitching = false; modelSwitching = false;
      setBusy(Boolean(busy)); scheduleContext(); pollProviders();
      window.StrataNativeModels?.refresh();
    }
  }
  select.addEventListener("change", async () => {
    try { await choose(select.value); }
    catch (error) { sync(); toast("error", "Could not switch the connection", error.message, 8000); }
  });
  family.addEventListener("change", async () => {
    const wanted = family.value;
    const candidate = profiles.find((p) => p.id === lastAdditional) || profiles[0];
    if (wanted === "additional" && !candidate) {
      sync(); toast("info", "No additional models", "Add a model in About.");
      $("provider-add").click(); return;
    }
    try { await choose(wanted === "additional" ? candidate.id : null); }
    catch (error) { sync(); toast("error", "Could not switch the model", error.message, 8000); }
  });
  $("provider-add").onclick = () => { showTab("about"); $("provider-card").scrollIntoView({block: "start"}); $("provider-name").focus(); };
  $("provider-preset").onchange = () => {
    const preset = presets[$("provider-preset").value];
    $("provider-name").value = preset.name; $("provider-url").value = preset.url;
    $("provider-model").value = preset.model; $("provider-context").value = preset.context;
    $("provider-images").checked = false; $("provider-model-list").replaceChildren(); result.textContent = "";
  };
  $("provider-discover").onclick = async () => {
    result.textContent = "Checking available models…";
    try {
      const data = await request("api/provider-models?base_url=" + encodeURIComponent($("provider-url").value.trim()));
      $("provider-model-list").replaceChildren(...data.models.map((model) => {
        const option = document.createElement("option"); option.value = model.id; return option;
      }));
      result.textContent = `Found ${data.models.length} models. Choose a model ID from the list.`;
    } catch (error) { result.textContent = error.message; }
  };
  $("provider-form").onsubmit = async (event) => {
    event.preventDefault();
    const submit = event.submitter;
    if (busy || modelSwitching || changing) { result.textContent = "Wait for the response or loading to finish before adding a model."; return; }
    submit.disabled = true; result.textContent = "Checking the connection…";
    try {
      const data = await request("api/providers", {
        name: $("provider-name").value.trim(), base_url: $("provider-url").value.trim(),
        model: $("provider-model").value.trim(), context: Number($("provider-context").value),
        images: $("provider-images").checked,
        backend: {ollama: "ollama", bonsai: "llamacpp", custom: "generic"}[$("provider-preset").value],
        reasoning_map: $("provider-preset").value === "bonsai" ? {high: "xhigh"} : {},
      });
      await choose(data.provider.id);
      result.textContent = "Saved. This model is now available in Chat.";
    } catch (error) { result.textContent = error.message; }
    finally { submit.disabled = false; }
  };
  window.StrataProviderControls = {sync};
  pollProviders(); setInterval(pollProviders, 5000);
})();
