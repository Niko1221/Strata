"use strict";
const $ = id => document.getElementById(id);
let apiKey = sessionStorage.getItem("strata.monitor.key") || "";
let running = false, tuning = false, lastStamp = "", reportStamp = "";
document.documentElement.dataset.theme = localStorage.getItem("strata.theme") || "dark";
const PROMPT_GROUPS = ["experts over PCIe", "dequant", "expert GEMM", "attention", "GDN", "other"];
const DECODE_GROUPS = ["GPU", "CPU experts", "draft", "adapt wait", "other"];
const num = (n, d = 0) => typeof n === "number" ? n.toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}) : "—";
const pct = n => typeof n === "number" ? `${Math.round(n * 100)}%` : "—";
function text(id, value) { const el = $(id); if (el.textContent !== String(value)) el.textContent = value; }
function notice(message) { $("error").hidden = !message; text("error", message || ""); }
async function api(path, body) {
  const response = await fetch(path, {cache:"no-store", headers:{...(apiKey ? {Authorization:`Bearer ${apiKey}`} : {}), ...(body !== undefined ? {"Content-Type":"application/json"} : {})}, ...(body !== undefined ? {method:"POST", body:JSON.stringify(body)} : {})});
  const value = await response.json();
  if (!response.ok) {
    if (response.status === 401) $("auth").hidden = false;
    throw new Error(value.error?.message || `HTTP ${response.status}`);
  }
  return value;
}
function bar(barId, legendId, names, values, unit) {
  // a stacked bar: `values[name]` in `unit` ("%" shares of 1, or "ms"), the legend with the numbers
  const total = names.reduce((s, n) => s + (values[n] || 0), 0) || 1;
  const bar = $(barId), legend = $(legendId);
  bar.replaceChildren(); legend.replaceChildren();
  names.forEach((name, i) => {
    const v = values[name] || 0, share = v / total;
    const seg = document.createElement("span"); seg.className = `c${i}`; seg.style.width = `${share * 100}%`;
    seg.title = `${name}: ${unit === "%" ? pct(v) : num(v, 1) + " ms"}`; if (share > 0.08) seg.textContent = unit === "%" ? pct(v) : num(v, 1);
    bar.append(seg);
    const item = document.createElement("span"); const sw = document.createElement("i"); sw.className = `c${i}`;
    item.append(sw, `${name} ${unit === "%" ? pct(v) : num(v, 1) + " ms"}`); legend.append(item);
  });
}
function renderLast(data) {
  const r = data.request;
  text("model", data.model || "—"); text("state", data.loaded ? "Loaded" : "Unloaded");
  if (!data.log) { text("last-note", "The engine has no log file (no \"log\" in the config): nothing to read."); }
  $("no-last").hidden = Boolean(r); $("last").hidden = !r;
  if (!r) return;
  const stamp = JSON.stringify(r);
  if (stamp === lastStamp) return;
  lastStamp = stamp;
  const p = r.prompt, d = r.decode, h = r.hit;
  text("decode-tps", num(d.tok_s, 1)); text("hit", pct(h?.hit_rate));
  text("pcie", h?.pcie_share ? `plus ${pct(h.pcie_share)} of all routed experts read by the card over PCIe` : "of the experts looked up while decoding");
  text("prompt-tps", p.tok_s ? num(p.tok_s) : "—"); text("prompt-note", p.read ? `${num(p.read)} tokens read, ${num(p.reused)} reused` : "tokens / second");
  text("last-note", `prompt ${num(p.tokens)} tokens · ${num(d.generated)} generated`);
  if (p.groups) bar("prompt-bar", "prompt-legend", PROMPT_GROUPS, p.groups.share, "%");
  else { $("prompt-bar").replaceChildren(); $("prompt-legend").textContent = p.read ? "no prefill timing line for this request" : "nothing read: the whole prompt was reused"; }
  if (d.groups) bar("decode-bar", "decode-legend", DECODE_GROUPS, d.groups.ms, "ms");
  else { $("decode-bar").replaceChildren(); $("decode-legend").textContent = "no decode timing line for this request"; }
  text("f-prompt", num(p.tokens)); text("f-reused", `${num(p.reused)} / ${num(p.read)}`); text("f-gen", num(d.generated));
  text("f-window", d.groups ? num(d.groups.ms_per_window, 1) : "—"); text("f-tpw", d.groups ? num(d.groups.tokens_per_window, 2) : "—");
  text("f-cpu", d.groups && typeof d.groups.cpu_experts === "number" ? `${num(d.groups.cpu_experts, 1)} (${num(d.groups.cpu_entries, 1)} entries)` : "—");
  text("kv", r.kv ? `KV streaming: ${pct(r.kv.kv_hit)} of this request's ${num(r.kv.block_reads)} block reads from VRAM, ${num(r.kv.ram_mib)} MiB from RAM` : "");
}
function table(id, header, rows) {
  const t = $(id); t.replaceChildren();
  const tr = document.createElement("tr"); header.forEach(h => { const th = document.createElement("th"); th.textContent = h; tr.append(th); }); t.append(tr);
  rows.forEach(row => { const tr = document.createElement("tr"); row.forEach(c => { const td = document.createElement("td"); td.textContent = c; tr.append(td); }); t.append(tr); });
}
function renderReport(rep) {
  const res = rep.results?.results;
  $("report").hidden = !res;
  if (!res) { text("report-path", rep.path ? `Report folder: ${rep.path} (its files could not be read)` : ""); return; }
  table("t-prompts", ["Read", "Tokens", "Prompt tok/s", "GPU ms", ...PROMPT_GROUPS],
        Object.entries(res.prompts || {}).map(([k, v]) => [k, num(v.prompt_tokens), num(v.summary?.prompt_tok_s ?? v.prompt_tps), v.groups ? num(v.groups.gpu_ms) : "—",
                                                            ...PROMPT_GROUPS.map(g => v.groups ? pct(v.groups.share[g]) : "—")]));
  table("t-decode", ["Request", "Context", "Decode tok/s", "Hit rate", "PCIe share", "ms/window", ...DECODE_GROUPS, "tokens/window", "CPU experts per layer-window"],
        Object.entries(res.decode || {}).map(([k, v]) => [k, num(v.context), num(v.decode_tps?.median, 1), pct(v.hit?.hit_rate), pct(v.hit?.pcie_share),
                                                           v.groups ? num(v.groups.ms_per_window, 1) : "—", ...DECODE_GROUPS.map(g => v.groups ? num(v.groups.ms[g], 1) : "—"),
                                                           v.groups ? num(v.groups.tokens_per_window, 2) : "—", v.groups ? num(v.groups.cpu_experts, 1) : "—"]));
  table("t-agent", ["Turn", "Prompt tokens", "Reused", "Read", "Prompt tok/s", "Decode tok/s", "Hit rate", "ms/window", "GPU", "CPU experts"],
        (res.agent?.turns || []).map(t => [t.turn, num(t.prompt_tokens), num(t.reused_tokens), num(t.read_tokens), num(t.prompt_tps), num(t.decode_tps),
                                           pct(t.hit?.hit_rate), t.groups ? num(t.groups.ms_per_window, 1) : "—", t.groups ? num(t.groups.ms.GPU, 1) : "—", t.groups ? num(t.groups.ms["CPU experts"], 1) : "—"]));
  const ul = $("findings"); ul.replaceChildren(); (res.findings || []).forEach(f => { const li = document.createElement("li"); li.textContent = f; ul.append(li); });
  text("report-path", `Report: ${rep.path} (README.md, results.json)`);
}
function renderTune(t, why) {
  tuning = Boolean(t) && t.state === "running";
  $("tune").disabled = !t || tuning || running;
  if (!t) { text("tune-state", why || "Not available"); $("tune-help").hidden = true; return; }
  const label = {idle: "Not run", running: `Running · ${num(t.elapsed_s)} s`, done: `Done in ${num(t.elapsed_s)} s`, failed: `Failed (exit ${t.returncode})`}[t.state] || t.state;
  text("tune-state", label);
  $("tune-help").hidden = t.state !== "idle";
  $("tune-log").hidden = !t.lines.length;
  if (t.lines.length) { const log = $("tune-log"); const v = t.lines.join("\n"); if (log.textContent !== v) { log.textContent = v; log.scrollTop = log.scrollHeight; } }
  text("tune-result", t.state === "done" ? `${t.report?.summary ? `Tuned: ${t.report.summary}. ` : ""}${t.note || ""}` : t.state === "failed" ? "The tuning failed; the settings stay as they were (the reason is in its output)" : "");
}
function renderStatus(st) {
  renderTune(st.tune, st.tune_unavailable);
  running = st.state === "running";
  $("run").disabled = running || tuning;
  const label = {idle: "No run yet", running: `Running · ${num(st.elapsed_s)} s`, done: `Done in ${num(st.elapsed_s)} s`, failed: `Failed (exit ${st.returncode})`}[st.state] || st.state;
  text("run-state", label);
  $("run-help").hidden = st.state !== "idle";
  $("run-log").hidden = !st.lines.length || (st.state === "done" && st.report?.results);
  if (st.lines.length) { const log = $("run-log"); const v = st.lines.join("\n"); if (log.textContent !== v) { log.textContent = v; log.scrollTop = log.scrollHeight; } }
  const stamp = st.report ? JSON.stringify([st.report.path, st.state]) : "";
  if (st.report && stamp !== reportStamp) { reportStamp = stamp; renderReport(st.report); }
  if (!st.report) $("report").hidden = true;
}
async function refresh() {
  try {
    const [last, st] = await Promise.all([api("/profile/last"), api("/profile/status")]);
    renderLast(last); renderStatus(st);
    text("updated", `Live · updated ${new Date().toLocaleTimeString()}`); notice("");
  } catch (error) { notice(error.message); text("updated", "Disconnected · retrying"); $("run").disabled = true; }
}
$("run").onclick = async () => {
  $("run").disabled = true;
  try { await api("/profile/run", {}); notice(""); }
  catch (error) { notice(error.message); }
  await refresh();
};
$("tune").onclick = async () => {
  if (!confirm("Tune this PC now? The model is unloaded for roughly 15-30 minutes (setup's estimate; longer on a slow card) and requests get 503 meanwhile.")) return;
  $("tune").disabled = true;
  try { await api("/profile/tune", {}); notice(""); }
  catch (error) { notice(error.message); }
  await refresh();
};
$("key").onclick = () => { $("auth").hidden = !$("auth").hidden; if (!$("auth").hidden) $("api-key").focus(); };
$("auth").onsubmit = async event => { event.preventDefault(); apiKey = $("api-key").value.trim(); sessionStorage.setItem("strata.monitor.key", apiKey); $("auth").hidden = true; await refresh(); };
$("theme").onclick = () => { const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = theme; localStorage.setItem("strata.theme", theme); };
async function poll() { await refresh(); setTimeout(poll, running || tuning ? 2000 : 3000); }
poll();
