"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const controls = require("../serve/web/resource-controls.js");
const status = (extra = {}) => ({enabled: true, available: true, selection: "auto", effective: "daily",
  reason: "automatic_start", headroom_gib: 4, vram_reserve_mib: 700,
  memory: {current: {resident_budget_gib: 40, vram_reserve_mib: 700}, pending: false}, ...extra});
class Element {
  constructor(tag, doc) { this.tagName = tag; this.ownerDocument = doc; this.children = []; this.dataset = {}; this.attributes = {}; this._text = ""; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
}
const nodes = root => [root, ...root.children.flatMap(nodes)];
const find = (root, name) => nodes(root).find(node => node.className?.split(" ").includes(name));
function harness(request = async () => status()) {
  const doc = {createElement: tag => new Element(tag, doc)};
  const hosts = [new Element("div", doc), new Element("div", doc)], requests = [];
  const control = controls.create({containers: hosts, request: (...args) => { requests.push(args); return request(...args); }});
  return {control, hosts, requests, selects: hosts.map(host => find(host, "resource-control__select")),
    messages: () => hosts.map(host => find(host, "resource-control__status").textContent)};
}
function deferred() { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return {promise, resolve, reject}; }

test("both views use fixed native choices and stable accessible DOM on every poll", () => {
  const h = harness(), before = h.hosts.map(nodes);
  h.control.update(status()); h.control.update(status({effective: "busy", headroom_gib: 8, vram_reserve_mib: 1536}));
  for (const [i, host] of h.hosts.entries()) {
    assert.deepEqual(nodes(host), before[i]);
    assert.equal(h.selects[i].tagName, "select");
    assert.deepEqual(h.selects[i].children.map(option => option.textContent), ["Automatic", "Full", "Daily", "Busy", "Off"]);
    assert.equal(h.selects[i].attributes["aria-label"], "Resource preset");
    assert.equal(find(host, "resource-control__status").attributes["aria-live"], "polite");
    assert.equal(h.selects[i].value, "auto");
    assert.equal(find(host, "resource-control__effective").textContent, "Effective: Busy");
    assert.match(find(host, "resource-control__targets").textContent, /8 GiB RAM · 1,536 MiB VRAM/);
  }
});

test("selection POST uses the complete server block, mirrors draft and ignores poll while saving", async () => {
  const post = deferred(), h = harness(() => post.promise); h.control.update(status());
  const version = h.control.version(), selecting = h.control.select("full");
  const duringPost = h.control.version();
  assert.deepEqual(h.requests, [["v1/resources", {enabled: true, selection: "full"}]]);
  assert.ok(h.selects.every(select => select.value === "full" && select.disabled));
  assert.ok(h.messages().every(message => message === "Selecting preset…"));
  h.control.update(status({selection: "busy"}), version); h.control.offline(version);
  assert.ok(h.selects.every(select => select.value === "full"));
  post.resolve(status({selection: "full", effective: "full", memory: {pending: true, current: {resident_budget_gib: 40, vram_reserve_mib: 700}}}));
  assert.equal(await selecting, true);
  assert.ok(h.selects.every(select => select.value === "full" && !select.disabled));
  assert.ok(h.messages().every(message => /pending.*Actual: 40 GiB/.test(message)));
  assert.doesNotMatch(h.messages()[0], /saved|success|applied/i);
  h.control.update(status({selection: "auto"}), version); // The old response arrived after POST completed.
  h.control.update(status({selection: "daily"}), duringPost); // Started during POST, delivered afterward.
  assert.ok(h.selects.every(select => select.value === "full"));
});

test("GET started before selection cannot overwrite the completed POST", async () => {
  const get = deferred(), h = harness((path, body) => body ? status({selection: "daily"}) : get.promise);
  h.control.update(status()); const refreshing = h.control.refresh();
  await h.control.select("daily"); get.resolve(status({selection: "auto"})); await refreshing;
  assert.ok(h.selects.every(select => select.value === "daily"));
});

test("failed POST restores authoritative readback without claiming saved or applied", async () => {
  const h = harness((path, body) => body ? Promise.reject(new Error("Disk write failed")) : status({selection: "busy"}));
  h.control.update(status({selection: "daily"})); assert.equal(await h.control.select("full"), false);
  assert.ok(h.selects.every(select => select.value === "busy"));
  assert.equal(h.requests.length, 2); assert.equal(h.requests[1].length, 1);
  assert.match(h.messages()[0], /Selection not confirmed: Disk write failed/);
  h.control.update(status({selection: "busy"})); assert.match(h.messages()[0], /Selection not confirmed/);
});

test("ambiguous POST failure with unavailable readback keeps known selection and disables further changes", async () => {
  const h = harness(() => Promise.reject(new Error("Network disconnected"))); h.control.update(status({selection: "daily"}));
  assert.equal(await h.control.select("full"), false);
  assert.ok(h.selects.every(select => select.value === "daily" && select.disabled));
  assert.match(h.messages()[0], /Selection not confirmed/);
});

test("offline readings remain identified as stale and fresh metrics restore the control", () => {
  const h = harness(); h.control.update(status()); h.control.offline();
  assert.ok(h.selects.every(select => select.disabled));
  assert.match(h.messages()[0], /Server unavailable/);
  assert.match(h.messages()[0], /Last reported: 40 GiB/);
  assert.match(find(h.hosts[0], "resource-control__effective").textContent, /Last reported/);
  h.control.update(status({selection: "busy"}));
  assert.ok(h.selects.every(select => !select.disabled && select.value === "busy"));
  assert.doesNotMatch(h.messages()[0], /unavailable/);
});

test("legacy absence, invalid status, unsupported memory mode and auth errors degrade without POST", async () => {
  for (const value of [undefined, {}, status({available: false})]) {
    const h = harness(); h.control.update(value);
    assert.ok(h.selects.every(select => select.disabled));
    assert.equal(await h.control.select("daily"), false); assert.equal(h.requests.length, 0);
  }
  for (const message of ["Resource presets are unavailable on this server.", "API key needed"]) {
    const h = harness(() => Promise.reject(new Error(message))); await h.control.refresh();
    assert.ok(h.selects.every(select => select.disabled)); assert.match(h.messages()[0], new RegExp(message));
  }
});

test("disabled legacy mode has honest Off selection and can enable Automatic", async () => {
  const h = harness(); h.control.update(status({enabled: false}));
  assert.ok(h.selects.every(select => select.value === "off" && !select.disabled));
  assert.match(find(h.hosts[0], "resource-control__effective").textContent, /Off · configured/);
  await h.control.select("auto"); assert.deepEqual(h.requests[0][1], {enabled: true, selection: "auto"});
});

test("Off posts reversible opt-out and never invokes engine load, unload or context APIs", async () => {
  const h = harness(() => status({enabled: false})); h.control.update(status()); await h.control.select("off");
  assert.deepEqual(h.requests, [["v1/resources", {enabled: false, selection: "auto"}]]);
  assert.ok(h.selects.every(select => select.value === "off"));
});

test("native pending, error and limitation coexist with actual allocation rather than desired targets", () => {
  const view = controls.describe(status({headroom_gib: 2, vram_reserve_mib: 256,
    memory: {current: {resident_budget_gib: 38, vram_reserve_mib: 1536}, pending: true, error: "retry scheduled", limitation: "arena floor"}}));
  assert.match(view.targets, /2 GiB RAM · 256 MiB/);
  assert.match(view.memory, /error: retry scheduled.*pending.*limited: arena floor.*Actual: 38 GiB RAM budget · 1,536 MiB/);
  assert.match(controls.describe(status({memory: {current: null}})).memory, /not reported/);
  assert.doesNotMatch(controls.describe(status({headroom_gib: Infinity, vram_reserve_mib: "256"})).targets, /Infinity|256/);
});

test("module uses injected requests and plain text, no timers, browser storage or unsafe HTML", async () => {
  const forbidden = () => assert.fail("Unexpected I/O");
  const sandbox = {fetch: forbidden, setTimeout: forbidden, setInterval: forbidden,
    localStorage: {getItem: forbidden, setItem: forbidden}, sessionStorage: {getItem: forbidden}};
  vm.createContext(sandbox); vm.runInContext(fs.readFileSync(require.resolve("../serve/web/resource-controls.js"), "utf8"), sandbox);
  const h = harness(); h.control.update(status({reason: '<img src=x onerror="fixture()">', memory: {error: "<script>fixture()</script>"}}));
  assert.equal(find(h.hosts[0], "resource-control__reason").children.length, 0);
  assert.match(h.messages()[0], /<script>fixture/);
  assert.equal(typeof sandbox.StrataResourceControls.create, "function");
});

test("app integrates shared controls with authenticated requests and guards every metrics response", async () => {
  const source = fs.readFileSync(require.resolve("../serve/web/app.js"), "utf8");
  const start = source.indexOf("const resourceControls =");
  const refresh = "resourceControls.refresh();", refreshStart = source.indexOf(refresh, start);
  assert.ok(start >= 0 && refreshStart > start, "resource controls have a complete initialization block");
  const end = refreshStart + refresh.length;
  const ids = [], requests = [], sentinel = {}, sandbox = {
    StrataResourceControls: {create(options) { ids.push(...options.containers); sentinel.request = options.request; return {refresh() {}}; }},
    $: id => id, headers: json => ({Authorization: "Bearer fixture", ...(json ? {"Content-Type": "application/json"} : {})}),
    fetch: async (path, options) => { requests.push({path, options}); return {ok: true, status: 200, json: async () => status()}; },
  };
  vm.createContext(sandbox); vm.runInContext(source.slice(start, end), sandbox);
  const html = fs.readFileSync(require.resolve("../serve/web/index.html"), "utf8");
  assert.ok(ids.includes("monitor-resources"), "the full Monitor mounts resource controls");
  for (const id of ids) assert.ok(html.includes(`id="${id}"`), `declared resource host ${id} exists`);
  await sentinel.request("v1/resources", {enabled: true, selection: "busy"});
  assert.equal(requests[0].options.headers.Authorization, "Bearer fixture");
  assert.equal(requests[0].options.method, "POST"); assert.equal(requests[0].options.cache, "no-store");
  assert.deepEqual(JSON.parse(requests[0].options.body), {enabled: true, selection: "busy"});
  const pollStart = source.indexOf("async function poll()"), pollEnd = source.indexOf("function setPill", pollStart);
  const polling = source.slice(pollStart, pollEnd);
  assert.ok(polling.indexOf("resourceControls.version()") < polling.indexOf("await fetch"));
  assert.match(polling, /resourceControls\.update\(lastMetrics\.resources, resourceVersion\)/);
  assert.equal((polling.match(/resourceControls\.offline\(resourceVersion\)/g) || []).length, 2);
  assert.ok(html.indexOf('src="web/resource-controls.js"') < html.indexOf('src="web/app.js"'));
});
