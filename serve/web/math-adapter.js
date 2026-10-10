// MathJax adapter: local assets, explicit conversion, no document scanning.
"use strict";
globalThis.StrataMathAdapter = {
  async initialize() {
    if (globalThis.MathJax?.tex2chtmlPromise) return;
    globalThis.MathJax = {
      loader: {paths: {mathjax: "web/vendor/mathjax", "mathjax-newcm": "web/vendor/mathjax-newcm-font"},
        load: ["ui/safe", "a11y/assistive-mml"]},
      startup: {typeset: false},
      tex: {packages: {"[-]": ["autoload", "require", "noundefined", "configmacros", "textmacros"], "[+]": ["ams", "newcommand"]}, maxBuffer: 16384, maxMacros: 1000,
        formatError: (_jax, error) => { throw error; }},
      chtml: {fontURL: "web/vendor/mathjax-newcm-font/chtml/woff2",
        dynamicPrefix: "[mathjax-newcm]/chtml/dynamic"},
      options: {enableMenu: false, enableEnrichment: false, enableSpeech: false, enableBraille: false,
        enableExplorer: false, enableAssistiveMml: true,
        safeOptions: {allow: {URLs: "none", classes: "none", cssIDs: "none", styles: "none"}}},
    };
    await StrataMath.script("web/vendor/mathjax/tex-chtml.js");
    await MathJax.startup.promise;
    // The combined component's accessibility defaults are applied during startup.
    // This adapter supplies semantic MathML, without the speech/explorer worker.
    Object.assign(MathJax.startup.document.options, {enableEnrichment: false, enableSpeech: false,
      enableBraille: false, enableExplorer: false, enableAssistiveMml: true});
    Object.assign(MathJax.startup.document.options.a11y, {speech: false, braille: false});
  },
  async render(source, {display, signal}) {
    if (signal.aborted) return {ok: false, reason: "unavailable"};
    try {
      MathJax.texReset();
      // Newcommand's global definitions must never leak to another equation.
      const jax = MathJax.startup.document.inputJax[0];
      const maps = ["new-Command", "new-Delimiter", "new-Environment"].map(name => {
        const map = jax.parseOptions.handlers.retrieve(name).map;
        return [map, new Map(map)];
      });
      try {
        const element = await MathJax.tex2chtmlPromise(source, {display});
        if (element.querySelector("mjx-merror")) return {ok: false, reason: "invalid"};
        MathJax.startup.document.reset();
        MathJax.startup.document.updateDocument();
        return {ok: true, element};
      } finally {
        for (const [map, saved] of maps) { map.clear(); for (const [key, value] of saved) map.set(key, value); }
      }
    } catch (_) { return {ok: false, reason: "invalid"}; }
  },
};
