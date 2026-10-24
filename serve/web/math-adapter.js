// KaTeX adapter. No engine or font loads until Render math is enabled.
"use strict";
globalThis.StrataMathAdapter = {
  async initialize() {
    if (globalThis.katex && document.querySelector("link[data-katex-ready]")) return;
    const css = document.createElement("link"); css.rel = "stylesheet";
    css.href = "web/vendor/katex/dist/katex.min.css";
    const loaded = new Promise((resolve, reject) => { css.onload = resolve; css.onerror = reject; });
    document.head.append(css);
    await Promise.all([loaded, globalThis.katex ? Promise.resolve() : StrataMath.script("web/vendor/katex/dist/katex.min.js")]);
    css.dataset.katexReady = "1";
  },
  async render(source, {display, signal}) {
    if (signal.aborted) return {ok: false, reason: "unavailable"};
    const element = document.createElement("span");
    try {
      katex.render(source, element, {displayMode: display, output: "htmlAndMathml", trust: false,
        throwOnError: true, strict: "warn", maxExpand: 1000, maxSize: 20, macros: {}});
      return {ok: true, element: element.firstElementChild};
    } catch (_) { return {ok: false, reason: "invalid"}; }
  },
};
