// serve/test_monitor.py MonitorTiles: runs serve/web/app.js under node with a recording fake DOM and prints the
// Monitor's GPU temp / Power / PCIe tiles for the readings given as JSON (argv: app.js path, readings or a list of
// readings, rendered one after another on the same page: one result per reading).
const vm = require("vm"), fs = require("fs");
const writes = {};
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
const els = {};
function el(id) {
  if (!els[id]) els[id] = new Proxy(any(), {set(t, k, v) { (writes[id] = writes[id] || {})[k] = v; return true; }});
  return els[id];
}
const ctx = {
  document: Object.assign(any(), {getElementById: el, querySelectorAll: () => [], addEventListener() {}}),
  window: Object.assign(any(), {addEventListener() {}}), location: {hash: "", search: "", pathname: "/"},
  localStorage: {getItem: () => null, setItem() {}}, matchMedia: () => ({matches: false, addEventListener() {}}),
  fetch: () => new Promise(() => {}), setTimeout() {}, setInterval() {}, clearTimeout() {}, requestAnimationFrame() {},
  navigator: any(), history: any(), sessionStorage: {getItem: () => null, setItem() {}}, Image: function () {}, URLSearchParams, console, Number, Math, String, Date, JSON, Promise, Map, Set, Array, Object,
};
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx);
const arg = JSON.parse(process.argv[3]);
const results = (Array.isArray(arg) ? arg : [arg]).map((hw) => {
  ctx.renderMonitor({state: "idle"}, hw, {}, {}, {}, null, [], {}, 0);
  const out = {};
  for (const k of ["temp", "power", "pcie"]) out[k] = {value: (writes[`mv-${k}`] || {}).innerHTML, sub: (writes[`ms-${k}`] || {}).textContent};
  return out;
});
console.log(JSON.stringify(Array.isArray(arg) ? results : results[0]));
