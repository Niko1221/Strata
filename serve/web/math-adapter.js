// Full TeX adapter: same contract, optional isolated worker on the Strata host.
"use strict";
globalThis.StrataMathAdapter = {
  renderTimeoutMs: null,
  async initialize() {},
  async render(source, {display, signal}) {
    if (signal.aborted) return {ok: false, reason: "limit"};
    try {
      // Aborting fetch does not cancel the server's compiler. Keep the RPC pending
      // so the shared queue cannot start another conversion while that job holds
      // the server lock. Compilation has no elapsed-time cutoff.
      const response = await fetch("web/math/render", {method: "POST", headers: headers(true),
        body: JSON.stringify({source, display})});
      if (!response.ok) return {ok: false, reason: "unavailable"};
      const result = await response.json();
      if (signal.aborted) return {ok: false, reason: "limit"};
      if (!result.ok) return {ok: false, reason: result.reason};
      const element = document.createElement("img");
      element.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(result.svg);
      element.alt = source; element.style.width = result.width + "em"; element.style.height = result.height + "em";
      return {ok: true, element};
    } catch (_) { return {ok: false, reason: "unavailable"}; }
  },
};
