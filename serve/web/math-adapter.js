// Full TeX adapter: same contract, optional isolated worker on the Strata host.
"use strict";
globalThis.StrataMathAdapter = {
  async initialize() {},
  async render(source, {display, signal}) {
    try {
      const response = await fetch("web/math/render", {method: "POST", headers: headers(true),
        body: JSON.stringify({source, display}), signal});
      if (!response.ok) return {ok: false, reason: "unavailable"};
      const result = await response.json();
      if (!result.ok) return {ok: false, reason: result.reason};
      const element = document.createElement("img");
      element.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(result.svg);
      element.alt = source; element.style.width = result.width + "em"; element.style.height = result.height + "em";
      return {ok: true, element};
    } catch (_) { return {ok: false, reason: "unavailable"}; }
  },
};
