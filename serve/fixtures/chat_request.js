// serve/test_monitor.py ChatRequest: runs serve/web/app.js under node with a fake DOM, applies the sampling settings
// given as JSON (argv: app.js path, settings), sends "hi" through the chat's own send(), and prints the request body
// it POSTs to v1/chat/completions, the shared defaults it would save for other apps, and budgetProblem's verdicts.
const vm = require("vm"), fs = require("fs");
function any() {                                       // a stand-in for any DOM object: every property/call works
  const f = function () {};
  return new Proxy(f, {
    get(t, k) {
      if (k === Symbol.toPrimitive) return () => "";
      if (k === Symbol.iterator) return function* () {};
      if (k === "then") return undefined;
      if (k === "length") return 0;
      if (k in t) return t[k];
      return any();
    },
    set(t, k, v) { t[k] = v; return true; },
    apply() { return any(); }, construct() { return any(); },
  });
}
const els = {};                                        // elements keep what is set on them (the input's value)
const el = (id) => (els[id] = els[id] || any());
let posted = null;
const ctx = {
  document: Object.assign(any(), {getElementById: el, querySelectorAll: () => [], addEventListener() {}}),
  window: Object.assign(any(), {addEventListener() {}}), location: {hash: "", search: "", pathname: "/"},
  localStorage: {getItem: () => null, setItem() {}}, matchMedia: () => ({matches: false, addEventListener() {}}),
  fetch: (url, opts) => {
    if (String(url).includes("chat/completions")) posted = JSON.parse(opts.body);
    return new Promise(() => {});
  },
  innerHeight: 800, setTimeout() {}, setInterval() {}, clearTimeout() {}, requestAnimationFrame() {}, AbortController,
  navigator: any(), history: any(), sessionStorage: {getItem: () => null, setItem() {}}, Image: function () {},
  URLSearchParams, console, Number, Math, String, Date, JSON, Promise, Map, Set, Array, Object, RegExp,
};
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx);
ctx.__s = JSON.parse(process.argv[3]);
vm.runInContext("settings = {...DEFAULTS, ...__s}; busy = null;", ctx);
el("input").value = "hi";
vm.runInContext("send()", ctx);
const out = {
  body: posted,
  shared: vm.runInContext("sharedDefaults(settings)", ctx),
  problems: vm.runInContext(`[["", ""], ["", "2048"], ["4096", "2048"], ["4096", "4032"], ["4096", "4033"],
                              ["100", "50"], ["", "1.5"], ["", "-3"], ["", "0"]].map(([m, b]) => budgetProblem(m, b))`, ctx),
};
console.log(JSON.stringify(out));
