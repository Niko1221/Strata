"use strict";
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const number = n => n == null ? "Unknown" : Number(n).toLocaleString(undefined, {maximumFractionDigits:1});
let state = null, selectedPlan = null, pending = false, polling = false, tableMarkup = "";
function theme(value) {
  document.documentElement.dataset.theme = value;
  $("theme").querySelector("use").setAttribute("href", `web/sprite.svg#i-${value === "dark" ? "sun" : "moon"}`);
}
try { theme(localStorage.getItem("strata.theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")); } catch (_) { theme("light"); }
$("theme").onclick = () => {
  const value = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  theme(value); try { localStorage.setItem("strata.theme", value); } catch (_) {}
};
function error(message) { $("error").textContent = message; $("error").hidden = !message; }
async function request(action, body) {
  if (!state) throw new Error("Wait for the model catalog to load.");
  const response = await fetch(`/api/models/${action}`, {method:"POST", headers:{"Content-Type":"application/json", "X-Strata-Token":state.token}, body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The operation failed. Try again.");
  return data;
}
function render() {
  if (!state) return;
  const active = state.operation?.state === "running";
  const external = state.running.some(r => !r.managed);
  const disabled = pending || active;
  const current = state.running[0];
  $("running-title").textContent = current ? current.model : state.loading ? "Loading model…" : "No model running";
  $("running-detail").textContent = external ? "Close the existing Strata server window once, then use this table to start and switch models."
    : active ? "The Models page stays available while this operation runs."
    : current ? "Start another installed model to switch. Requests must finish before switching."
    : "Select an installed model to start it, or review a download below.";
  $("open-chat").hidden = !current || active || current.loaded === false;
  $("open-chat").href = current?.url || state.model_url;
  $("stop-model").hidden = !current || external;
  $("stop-model").disabled = disabled;
  const hw = state.hardware;
  const gpu = (hw.gpus || []).filter(g => g.usable).map(g => `${g.name} (${number(g.vram_gb)} GB VRAM)`).join(", ");
  $("hardware").textContent = `${number(hw.ram_gb)} GB RAM · ${gpu || "No supported GPU detected; setup checks compatibility"}`;
  $("data-folder").textContent = state.data_folder;
  $("lookup").textContent = state.lookup || "Setup places each model’s table with its files.";
  const markup = state.models.map(row => {
    const installation = row.installation;
    const fit = row.on_this_pc || "Setup checks this PC";
    const tight = fit.includes("tight") || fit.includes("does not fit");
    const running = installation && current?.model === installation.model_name;
    const installed = installation?.ready;
    return `<tr><td><span class="model-name ${running ? "active-model" : ""}">${esc(row.model)}${running ? " · Running" : ""}</span><span class="model-family">${esc(row.title)}</span><span class="model-status">${esc(installation ? installed ? "Installed · " + installation.context + " context" : "Installation needs repair" : row.about)}</span></td>
      <td data-label="RAM estimate">${number(row.ram_needed_gb)} GB<span class="model-status" data-fit="${tight ? "tight" : "fits"}">${esc(fit)}</span></td><td data-label="Full download">${number(row.download_gb)} GB</td>
      <td><div class="row-actions">${installed ? `<button class="st-btn st-btn--secondary" data-start="${esc(row.id)}" ${disabled || external || running ? "disabled" : ""}>${current ? "Switch" : "Start"}</button>` : ""}<button class="st-btn st-btn--secondary" data-plan="${esc(row.id)}" ${disabled ? "disabled" : ""}>${installation ? "Review setup" : "Review download"}</button></div></td></tr>`;
  }).join("");
  // Polling must not replace focused controls when the catalog is unchanged.
  if (markup !== tableMarkup) {
    $("models").innerHTML = markup;
    tableMarkup = markup;
  }
  for (const button of document.querySelectorAll("[data-plan]")) button.onclick = () => act(async () => {
    selectedPlan = await request("plan", {model:button.dataset.plan});
    renderPlan();
    $("plan").scrollIntoView({block:"nearest", behavior:matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth"});
  });
  for (const button of document.querySelectorAll("[data-start]")) button.onclick = () => act(async () => {
    await request("start", {model:button.dataset.start});
  });
  $("refresh").disabled = pending;
  $("confirm-install").disabled = disabled || external || !!selectedPlan?.plan.disk_short;
  const operation = state.operation;
  $("job").hidden = !operation;
  if (operation) {
    $("job").dataset.state = operation.state;
    $("job-title").textContent = operation.kind === "install" ? "Download and setup" : operation.kind === "stop" ? "Stopping model" : "Loading model";
    const job = state.install;
    $("job-message").textContent = active && operation.kind === "install" && job?.running ? job.step || "Starting setup…" : operation.message;
    const download = active && job?.running && operation.kind === "install" ? job.download : null;
    $("download-progress").hidden = !download;
    if (download) $("download-progress").value = download.percent;
    $("download-detail").textContent = download ? `${download.file}: ${number(download.done_gb)} / ${number(download.total_gb)} GB (${download.percent}%)` : "";
    $("job-log").textContent = (state.logs || []).join("\n");
    $("cancel-download").hidden = !(active && operation.kind === "install" && job?.running);
    $("cancel-download").disabled = pending;
  }
}
function renderPlan() {
  $("plan").hidden = !selectedPlan;
  if (!selectedPlan) return;
  const p = selectedPlan.plan;
  $("plan-title").textContent = p.title;
  $("plan-facts").innerHTML = [["Full model download", `${number(p.download_gb)} GB; existing files are reused`], ["Free disk space", `${number(p.disk_free_gb)} GB`], ["RAM", p.ram], ["Lookup table", p.lookup], ["Settings", p.settings], ...(p.license ? [["License", p.license]] : [])].map(([key, value]) => `<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join("");
  $("plan-note").textContent = p.disk_short || `${p.note} Downloading stops the current managed model first. You can start it again when setup finishes.`;
  $("plan-command").textContent = p.setup_command;
}
async function refresh() {
  if (polling) return;
  polling = true;
  try {
    const response = await fetch("/api/models/status");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The launcher is unavailable.");
    state = data;
    render();
  } catch (e) { error(e.message); } finally { polling = false; }
}
async function act(fn) {
  if (pending) return;
  pending = true; error(""); render();
  try { await fn(); } catch (e) { error(e.message); }
  finally { pending = false; await refresh(); render(); }
}
$("refresh").onclick = () => { error(""); refresh(); };
$("dismiss-plan").onclick = () => { selectedPlan = null; renderPlan(); };
$("confirm-install").onclick = () => act(async () => {
  if (!selectedPlan) return;
  await request("install", {plan_id:selectedPlan.plan_id});
  selectedPlan = null; renderPlan();
});
$("stop-model").onclick = () => act(() => request("stop", {}));
$("cancel-download").onclick = () => act(() => request("cancel", {}));
refresh();
setInterval(refresh, 3000);