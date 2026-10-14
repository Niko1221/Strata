/* One usage component for Strata's shared app bar. Values and editable rates come from the local server. */
"use strict";
window.StrataUsage = (() => {
  let root, getHeaders, dirty = false, screen, profit, latest, performance, navigate, selectedDate = null, historySignature = "";
  const ids = ["input_per_million", "output_per_million", "cached_input_per_million",
               "electricity_per_kwh", "electricity_currency", "usd_to_electricity_currency"];
  const status = window.StrataMetrics;
  const money = (value, currency = "USD") => value == null ? "Set rate" :
    new Intl.NumberFormat(undefined, {style: "currency", currency, minimumFractionDigits: 2,
      maximumFractionDigits: Math.abs(value) > 0 && Math.abs(value) < 0.01 ? 6 : 2}).format(value);
  const duration = (seconds) => {
    const s = Math.max(0, Math.floor(seconds || 0)), h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60;
    return h ? `${h}:${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}` :
      m ? `${m}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`;
  };
  function mount(element, headers, options = {}) {
    if (!element || root) return;
    root = element; getHeaders = headers;
    screen = options.screen; profit = options.profit; navigate = options.navigate;
    root.innerHTML = `<details class="usage" id="usage-details">
      <summary class="usage-summary" aria-label="Usage tracker and pricing settings">
        <span class="usage-stat"><span class="usage-label">Tokens</span><strong data-usage="tokens">—</strong></span>
        <span class="usage-stat"><span class="usage-label">Cached tokens</span><strong data-usage="cached-tokens">—</strong></span>
        <span class="usage-stat"><span class="usage-label">API value</span><strong data-usage="value">—</strong></span>
        <span class="usage-stat"><span class="usage-label">GPU active</span><strong data-usage="time">—</strong></span>
        <span class="usage-stat"><span class="usage-label">GPU energy</span><strong data-usage="energy">—</strong></span>
        <span class="usage-stat"><span class="usage-label">Power cost</span><strong data-usage="electricity">Set rate</strong></span>
      </summary>
      <section class="usage-panel" aria-label="Usage breakdown and editable prices">
        <div class="usage-heading"><div><span class="usage-eyebrow">ALL CONNECTED APPS</span><h2>Usage &amp; costs</h2></div>
          <button class="st-btn st-btn--icon" type="button" aria-label="Close usage tracker" id="usage-close">×</button></div>
        <p class="muted small" id="usage-model"></p>
        <dl class="usage-breakdown">
          <div><dt>Input tokens</dt><dd data-usage="input">—</dd></div>
          <div><dt>Cached input</dt><dd data-usage="cached">—</dd></div>
          <div><dt>Output + thinking</dt><dd data-usage="output">—</dd></div>
          <div><dt>With cache pricing</dt><dd data-usage="cache-value">—</dd></div>
          <div><dt>GPU draw while working</dt><dd data-usage="watts">—</dd></div>
        </dl>
        <p class="usage-note" id="usage-energy-note"></p>
        <form id="usage-prices">
          <fieldset class="usage-fieldset"><legend>Token rates <span>USD / 1M tokens</span></legend>
            <div class="usage-rate-grid">
              <label>Input<input class="st-input" id="usage-input_per_million" type="number" min="0" step="any" required></label>
              <label>Output<input class="st-input" id="usage-output_per_million" type="number" min="0" step="any" required></label>
              <label>Cached input<input class="st-input" id="usage-cached_input_per_million" type="number" min="0" step="any" required></label>
            </div>
          </fieldset>
          <fieldset class="usage-fieldset"><legend>Electricity <span>your price per kWh</span></legend>
            <div class="usage-electric-grid">
              <label>Price / kWh<input class="st-input" id="usage-electricity_per_kwh" type="number" min="0" step="any" placeholder="Enter your tariff"></label>
              <label>Currency<input class="st-input" id="usage-electricity_currency" type="text" minlength="3" maxlength="3" pattern="[A-Za-z]{3}" required placeholder="USD"></label>
            </div>
            <label class="usage-fx">Local currency per USD<input class="st-input" id="usage-usd_to_electricity_currency" type="number" min="0.000000001" step="any" placeholder="For example, 120 BDT = 1 USD"></label>
            <p class="usage-note usage-fx-note">Optional for USD. Required in another currency to calculate profit in USD.</p>
          </fieldset>
          <p class="usage-note">Regular API value uses all input and output tokens at these rates. This is a hosted-price comparison; local inference has no token bill. Price edits revalue the totals immediately.</p>
          <div class="usage-source"><a href="https://openrouter.ai/qwen/qwen3.8-flash" target="_blank" rel="noopener noreferrer">OpenRouter pricing ↗</a><span id="usage-price-date"></span></div>
          <div class="usage-footer"><span id="usage-save-message" role="status"></span><button class="st-btn st-btn--primary" id="usage-save" type="submit">Save prices</button></div>
        </form>
      </section>
    </details>`;
    const close = () => {
      root.querySelector("details").open = false;
      root.querySelector("summary").focus();
    };
    root.querySelector("#usage-close").onclick = close;
    const form = root.querySelector("form");
    form.oninput = () => {
      dirty = true;
      root.querySelector("#usage-save-message").textContent = "Unsaved changes";
    };
    form.onsubmit = async (event) => {
      event.preventDefault();
      if (!form.reportValidity()) return;
      const rates = {};
      for (const id of ids) {
        const value = root.querySelector(`#usage-${id}`).value.trim();
        rates[id] = id === "electricity_currency" ? value.toUpperCase() : value === "" ? null : Number(value);
      }
      const button = root.querySelector("#usage-save"), message = root.querySelector("#usage-save-message");
      button.disabled = true; message.textContent = "Saving…";
      try {
        const response = await fetch("usage/settings", {method: "POST", headers: getHeaders(), body: JSON.stringify({rates})});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error?.message || `HTTP ${response.status}`);
        dirty = false; update(data); message.textContent = "Saved · totals updated";
      } catch (error) {
        message.textContent = error.message;
      } finally { button.disabled = false; }
    };
    document.addEventListener("click", (event) => {
      if (!root.contains(event.target)) root.querySelector("details").open = false;
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && root.querySelector("details").open) close();
    });
    mountScreen();
    if (profit) profit.onclick = () => { selectedDate = "all"; navigate?.(); renderScreen(latest); };
  }
  const exactCount = (value) => new Intl.NumberFormat().format(value ?? 0);
  const rateText = (value) => Number.isFinite(value) && value >= 0 ? `${new Intl.NumberFormat(undefined,
    {minimumFractionDigits: 1, maximumFractionDigits: 1}).format(value)} t/s` : "Not recorded";
  const averagePowerText = (row) => {
    const watts = Number.isFinite(row.energy_wh) && row.energy_wh >= 0 &&
      Number.isFinite(row.measured_s) && row.measured_s > 0 ? row.energy_wh / row.measured_s * 3600 : null;
    return Number.isFinite(watts) ? `${new Intl.NumberFormat(undefined,
      {minimumFractionDigits: 1, maximumFractionDigits: 1}).format(watts)} W` : "Not recorded";
  };
  const cacheHitText = (row) => row.input_tokens > 0 ? new Intl.NumberFormat(undefined,
    {style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1}).format(
      Math.min(row.input_tokens, Math.max(0, row.cached_tokens || 0)) / row.input_tokens) : "\u2014";
  const cpuTime = (seconds) => seconds == null ? "Not recorded" : seconds < 60 ? `${seconds.toFixed(2)} CPU s` : duration(seconds);
  const energyText = (value) => value == null ? "Not recorded" : `${new Intl.NumberFormat(undefined, {maximumFractionDigits: 7}).format(value)} kWh`;
  function profitValue(row, rates) {
    if (row.profit_usd != null) return money(row.profit_usd);
    if (rates.electricity_per_kwh == null) return "Set tariff";
    if (rates.electricity_currency !== "USD" && !rates.usd_to_electricity_currency) return "Set exchange rate";
    return "Not measured";
  }
  function mountScreen() {
    if (!screen) return;
    const metric = (label, field, detail) => `<article class="st-card usage-metric"><span class="usage-metric-label">${label}</span><strong id="daily-${field}">—</strong><p id="daily-${field}-note">${detail}</p></article>`;
    screen.innerHTML = `<div class="usage-page-head"><div><span class="usage-eyebrow">ALL CONNECTED APPS</span><h1>Usage ledger</h1><p id="usage-screen-status">Waiting for usage totals…</p></div>
      <div class="usage-page-controls"><label>Day<input class="st-input" id="usage-day" type="date"></label>
        <button class="st-btn st-btn--secondary" id="usage-today" type="button">Today</button>
        <button class="st-btn st-btn--secondary" id="usage-all-time" type="button" aria-pressed="false">All time</button>
        <button class="st-btn st-btn--primary" id="usage-edit" type="button">Edit prices</button></div></div>
      <section class="st-card usage-profit-card" aria-label="Profit after energy"><div><span class="usage-eyebrow" id="usage-period">TODAY</span><h2>Profit after energy</h2>
        <strong id="daily-profit">—</strong><p id="daily-profit-note">API-equivalent value saved after GPU electricity.</p></div>
        <div class="usage-profit-equation"><div><span>API value</span><strong id="profit-api">—</strong></div><span aria-hidden="true">−</span>
          <div><span>Power cost in USD</span><strong id="profit-power">—</strong></div><p>Profit is an estimate of API spending avoided, after recorded GPU energy costs.</p></div></section>
      <div class="usage-metric-grid usage-stats-grid" aria-label="Usage statistics">
        ${metric("API value", "api", "Regular input and output pricing")}
        ${metric("Power cost", "cost", "Your electricity tariff per kWh")}
        ${metric("GPU energy", "energy", "Measured during active work only")}
        ${metric("Total tokens", "tokens", "Input + output, including thinking")}
        ${metric("Cached tokens", "cached", "Reused input, included in prompt tokens")}
        ${metric("Cached token hit %", "cache-hit", "Cached share of prompt tokens")}
        ${metric("Model active", "active", "Elapsed inference and GPU image work")}
        ${metric("Prompt tokens", "prompt", "Input, including cached reuse")}
        ${metric("Output tokens", "output", "Response + thinking tokens")}
        ${metric("CPU runtime", "cpu", "Process CPU time, summed across cores")}
        ${metric("Avg decode t/s", "decode", "Live / last completed request")}
        ${metric("Avg GPU power", "gpu-power", "Measured active work; idle excluded")}</div>
      <div class="usage-detail-strip"><span id="daily-token-detail">—</span><span id="daily-cache-detail">—</span><span id="daily-request-detail">—</span></div>
      <p class="usage-coverage" id="daily-coverage"></p>
      <section class="st-card usage-history-card"><div class="req-head"><h2>Daily history</h2><span class="muted small" id="usage-history-zone"></span></div>
        <div class="table-wrap"><table class="st-table usage-history"><thead><tr><th>Day</th><th>Tokens</th><th>Prompt</th><th>Output</th><th>Avg decode</th><th>Cache hit %</th><th>API value</th><th>CPU runtime</th><th>Model active</th><th>Energy</th><th>Power cost</th><th>Profit · USD</th></tr></thead><tbody id="usage-day-rows"></tbody></table></div>
        <p class="usage-note" id="usage-history-note"></p></section>`;
    screen.querySelector("#usage-day").onchange = (event) => {
      if (!event.target.reportValidity()) return;
      selectedDate = event.target.value || latest?.today; renderScreen(latest);
    };
    screen.querySelector("#usage-today").onclick = () => { selectedDate = null; renderScreen(latest); };
    screen.querySelector("#usage-all-time").onclick = () => { selectedDate = "all"; renderScreen(latest); };
    screen.querySelector("#usage-edit").onclick = (event) => {
      event.stopPropagation(); root.querySelector("details").open = true;
      root.querySelector("#usage-input_per_million").focus();
    };
  }
  function renderScreen(data) {
    if (!screen) return;
    if (!data) { screen.querySelector("#usage-screen-status").textContent = "Usage tracking is unavailable for this server."; return; }
    const all = selectedDate === "all", day = selectedDate || data.today;
    const unavailable = !all && day < data.daily_since_date;
    const row = all ? data : data.daily.find((entry) => entry.date === day) || {
      total_tokens: 0, input_tokens: 0, output_tokens: 0, cached_tokens: 0, api_value_usd: 0, cache_adjusted_usd: 0,
      cpu_s: 0, active_s: 0, energy_kwh: 0, electricity_cost: data.rates.electricity_per_kwh == null ? null : 0,
      electricity_cost_usd: data.rates.electricity_per_kwh != null &&
        (data.rates.electricity_currency === "USD" || data.rates.usd_to_electricity_currency) ? 0 : null,
      profit_usd: data.rates.electricity_per_kwh != null &&
        (data.rates.electricity_currency === "USD" || data.rates.usd_to_electricity_currency) ? 0 : null,
      requests: 0, measured_s: 0, cpu_measured_s: 0, profit_partial: false, decode_tok_s: null, decode_tokens: 0
    };
    const set = (id, value) => { screen.querySelector(`#${id}`).textContent = unavailable ? "Not recorded" : value; };
    const offset = data.timezone_offset_minutes / 60;
    const zone = `${data.timezone || "server local time"} · UTC${offset >= 0 ? "+" : ""}${offset}`;
    screen.querySelector("#usage-screen-status").textContent = `${data.model} · ${zone} · ${data.active ? "Working" : "Idle time excluded"}`;
    screen.querySelector("#usage-period").textContent = all ? "ALL TIME" : day === data.today ? "TODAY" : day;
    const picker = screen.querySelector("#usage-day");
    if (document.activeElement !== picker) picker.value = all ? data.today : day;
    picker.max = data.today;
    screen.querySelector("#usage-all-time").setAttribute("aria-pressed", String(all));
    screen.querySelector("#usage-today").setAttribute("aria-pressed", String(!all && day === data.today));
    set("daily-profit", profitValue(row, data.rates));
    screen.querySelector(".usage-profit-card").dataset.negative = String(row.profit_usd < 0);
    set("daily-profit-note", row.profit_partial ? "Estimate after recorded energy costs; energy coverage is incomplete." : "API-equivalent value saved after GPU electricity.");
    set("profit-api", money(row.api_value_usd));
    set("profit-power", row.electricity_cost_usd == null ? profitValue(row, data.rates) : money(row.electricity_cost_usd));
    set("daily-tokens", exactCount(row.total_tokens)); set("daily-api", money(row.api_value_usd));
    set("daily-prompt", exactCount(row.input_tokens)); set("daily-output", exactCount(row.output_tokens));
    set("daily-cached", exactCount(row.cached_tokens));
    screen.querySelector("#daily-cached").title = "Cached input tokens reused across completed calls. Included in prompt tokens; repeated uses count again.";
    set("daily-decode", row.output_tokens ? rateText(row.decode_tok_s) : "\u2014");
    set("daily-cache-hit", cacheHitText(row));
    set("daily-gpu-power", averagePowerText(row));
    screen.querySelector("#daily-gpu-power").title = unavailable ? "GPU energy was not recorded for this date." :
      `GPU board energy divided by ${duration(row.measured_s)} of measured active work. Unmeasured work and idle time are excluded; whole-PC power is not measured.`;
    const liveRate = performance?.live?.state === "generating" ? performance.live.tok_s : null;
    const lastRate = performance?.requests?.find((entry) => Number.isFinite(entry.decode_tok_s))?.decode_tok_s;
    screen.querySelector("#daily-decode-note").textContent = Number.isFinite(liveRate) ?
      `Live ${rateText(liveRate)}` : Number.isFinite(lastRate) ? `Last ${rateText(lastRate)}` : "No completed decode yet";
    screen.querySelector("#daily-decode").title = unavailable ? "Decode timing was not recorded for this date." :
      `Time-weighted average for ${exactCount(row.decode_tokens)} recorded engine-decoded tokens. Untimed work, prefill and idle time are excluded.`;
    set("daily-cpu", cpuTime(row.cpu_s)); set("daily-active", duration(row.active_s));
    set("daily-energy", energyText(row.energy_kwh));
    set("daily-cost", row.electricity_cost == null ? data.rates.electricity_per_kwh == null ? "Set tariff" : "Not measured" : money(row.electricity_cost, data.rates.electricity_currency));
    set("daily-api-note", `Cache-adjusted value ${money(row.cache_adjusted_usd)}`);
    set("daily-token-detail", `${exactCount(row.input_tokens)} input · ${exactCount(row.output_tokens)} output`);
    set("daily-cache-detail", `${exactCount(row.cached_tokens)} cached input`);
    set("daily-request-detail", `${exactCount(row.requests)} completed model calls`);
    screen.querySelector("#daily-coverage").textContent = unavailable ? "Daily history was not recorded on this date. Older totals remain in All time." :
      `GPU energy coverage ${duration(row.measured_s)} / ${duration(row.active_s)} active · CPU coverage ${duration(row.cpu_measured_s)} / ${duration(row.active_s)} active. CPU time can exceed elapsed time when several cores work together. GPU energy excludes whole-PC electricity.`;
    screen.querySelector("#usage-history-zone").textContent = zone;
    screen.querySelector("#usage-history-note").textContent = `Daily tracking began ${data.daily_since_date}. Tokens are assigned to the day a model call completes; work crossing midnight is split between days. All time includes ${exactCount(data.undated.total_tokens)} older tokens without daily history. Saved price changes revalue every day.`;
    const signature = JSON.stringify([data.daily, data.rates]);
    if (signature !== historySignature) {
      historySignature = signature;
      const body = screen.querySelector("#usage-day-rows"); body.replaceChildren();
      for (const entry of data.daily) {
        const tr = document.createElement("tr"), dateCell = document.createElement("td"), button = document.createElement("button");
        button.type = "button"; button.className = "usage-history-day"; button.textContent = entry.date;
        button.dataset.date = entry.date; button.onclick = () => { selectedDate = entry.date; renderScreen(latest); };
        dateCell.appendChild(button); tr.appendChild(dateCell);
        const cells = [exactCount(entry.total_tokens), exactCount(entry.input_tokens), exactCount(entry.output_tokens),
          entry.output_tokens ? rateText(entry.decode_tok_s) : "\u2014", cacheHitText(entry),
          money(entry.api_value_usd), cpuTime(entry.cpu_s), duration(entry.active_s), energyText(entry.energy_kwh),
          entry.electricity_cost == null ? "—" : money(entry.electricity_cost, data.rates.electricity_currency), profitValue(entry, data.rates)];
        for (const value of cells) { const td = document.createElement("td"); td.textContent = value; tr.appendChild(td); }
        body.appendChild(tr);
      }
    }
    for (const button of screen.querySelectorAll(".usage-history-day")) button.setAttribute("aria-pressed", String(!all && button.dataset.date === day));
  }
  function update(data, metrics) {
    if (!root) return;
    root.hidden = !data;
    const bar = root.closest(".usage-bar");
    if (bar) bar.hidden = !data;
    if (profit) profit.hidden = !data;
    if (!data) { renderScreen(data); return; }
    latest = data;
    if (metrics) performance = metrics;
    const set = (name, text) => { root.querySelector(`[data-usage="${name}"]`).textContent = text; };
    root.dataset.active = String(data.active);
    set("tokens", status.compact(data.total_tokens)); set("cached-tokens", status.compact(data.cached_tokens));
    set("value", status.money(data.api_value_usd));
    root.querySelector('[data-usage="cached-tokens"]').title = `${exactCount(data.cached_tokens)} cached input tokens reused across completed calls. Included in Tokens; repeated uses count again.`;
    set("time", status.duration(data.active_s));
    set("energy", data.energy_kwh == null ? "— kWh" : `${status.number(data.energy_kwh)} kWh`);
    set("electricity", data.rates.electricity_per_kwh == null ? "Set rate" :
      data.energy_kwh == null ? "—" : status.money(data.electricity_cost, data.rates.electricity_currency));
    for (const [field, name] of [["input_tokens", "input"], ["cached_tokens", "cached"], ["output_tokens", "output"]])
      set(name, new Intl.NumberFormat().format(data[field]));
    set("cache-value", money(data.cache_adjusted_usd));
    set("watts", data.power_w == null ? data.active ? "Not available" : "Idle · not counted" : `${data.power_w.toFixed(1)} W`);
    root.querySelector("#usage-model").textContent = `${data.model} · benchmark: ${data.pricing_model}`;
    const energySince = new Date(data.energy_since * 1000).toLocaleString();
    root.querySelector("#usage-energy-note").textContent = data.energy_kwh == null ?
      `Energy measurement starts ${energySince}. Historical power was not recorded. Idle, queue, model loading and tool time are excluded.` :
      `GPU energy since ${energySince}. ${data.energy_method === "energy_counter" ? "Measured with the GPU energy counter." : "Estimated from GPU power samples."} Idle, queue, model loading and tool time are excluded. Measured coverage: ${duration(data.measured_s)} of ${duration(data.active_s)} active work. Whole-PC electricity is not measured.`;
    root.querySelector("#usage-price-date").textContent = data.rates_updated_at ? "Manually updated" : `Checked ${data.pricing_checked}`;
    if (!dirty && !root.querySelector("form").contains(document.activeElement)) {
      for (const id of ids) root.querySelector(`#usage-${id}`).value = data.rates[id] ?? "";
    }
    const summary = root.querySelector("summary");
    summary.title = `${exactCount(data.total_tokens)} tokens · ${exactCount(data.cached_tokens)} cached input tokens · ${money(data.api_value_usd)} regular API value · ${duration(data.active_s)} active GPU work`;
    if (data.persistence_error) root.querySelector("#usage-save-message").textContent = "Totals could not be saved to disk.";
    if (profit) {
      profit.querySelector("strong").textContent = data.profit_usd == null ? profitValue(data, data.rates) : status.money(data.profit_usd);
      profit.dataset.negative = String(data.profit_usd < 0);
      profit.title = `API-equivalent value minus recorded GPU electricity. ${data.profit_partial ? "Historical energy is incomplete." : ""}`;
    }
    renderScreen(data);
  }
  return {mount, update};
})();
