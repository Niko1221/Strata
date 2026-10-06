"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const controls = require("./skill-controls.js");
const catalog = (extra = {}) => ({enabled: true, skills: [
  {name: "design", description: "Polish an interface"}, {name: "review", description: "Inspect supplied code"},
], ...extra});
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return {promise, resolve, reject}; };
function harness(fetchCatalog = async () => catalog()) {
  const doc = {listeners: {}, activeElement: null,
    defaultView: {Event: class {constructor(type, options) { this.type = type; Object.assign(this, options); }}},
    addEventListener(type, callback) { (this.listeners[type] ||= []).push(callback); }};
  class Element {
    constructor(tag) {
      this.ownerDocument = doc; this.tagName = tag; this.children = []; this.attributes = {}; this.listeners = {};
      this.hidden = false; this.disabled = false; this.value = ""; this.selectionStart = 0; this.textContent = "";
    }
    appendChild(node) { this.children.push(node); node.parentElement = this; return node; }
    replaceChildren() { this.children = []; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    getAttribute(name) { return this.attributes[name]; }
    addEventListener(type, callback) { (this.listeners[type] ||= []).push(callback); }
    dispatchEvent(event) { event.target ||= this; for (const callback of this.listeners[event.type] || []) callback(event); }
    contains(node) { return this === node || this.children.some(child => child.contains(node)); }
    focus() { doc.activeElement = this; for (const callback of doc.listeners.focusin || []) callback({target: this}); }
    setSelectionRange(a, b) { this.selectionStart = a; this.selectionEnd = b; }
    scrollIntoView() {}
  }
  doc.createElement = tag => new Element(tag);
  const input = new Element("textarea"), container = new Element("div"), signals = [];
  const component = controls.mount({input, container, fetchCatalog: signal => { signals.push(signal); return fetchCatalog(signal); }});
  const find = id => {
    const visit = node => node.id === id ? node : node.children.map(visit).find(Boolean);
    return visit(container);
  };
  const type = (value, caret = value.length) => { input.value = value; input.selectionStart = caret; input.dispatchEvent({type: "input"}); };
  const key = (name, extra = {}) => {
    const event = {type: "keydown", key: name, target: doc.activeElement || input, prevented: false,
      preventDefault() { this.prevented = true; }, ...extra};
    const handled = component.handleKey(event); return {handled, prevented: event.prevented};
  };
  const click = async node => { if (!node.disabled) for (const callback of node.listeners.click || []) await callback({target: node}); };
  const ready = async () => { component.setEnabled(true); await component.refresh(); input.focus(); };
  return {component, doc, Element, input, container, signals, find, type, key, click, ready,
    button: () => find("skills-btn"), rows: () => find("skills-list").children};
}

test("mount and MCP-off are inert; only explicit enabled refresh requests metadata", async () => {
  let calls = 0;
  const h = harness(async () => { calls++; return catalog(); });
  h.type("/design draft"); assert.deepEqual(h.component.request(), {}); assert.equal(calls, 0); assert.equal(h.container.hidden, true);
  await h.component.refresh(); assert.equal(calls, 0);
  await h.ready(); assert.equal(calls, 1); assert.deepEqual(h.component.request(), {strata_skill: "design"});
  h.component.setEnabled(false); assert.equal(h.container.hidden, true); assert.deepEqual(h.component.request(), {});
  assert.equal(h.input.value, "/design draft");
});

test("catalog is metadata only and descriptions are literal text", async () => {
  const h = harness(async () => catalog({skills: [{name: "design", description: "<script>bad</script>"},
    {name: "../escape", description: "bad"}, {name: "Upper", description: "bad"}, {name: "design", description: "duplicate"}]}));
  await h.ready(); await h.click(h.button());
  assert.equal(h.rows().length, 1); assert.equal(h.rows()[0].children[1].textContent, "<script>bad</script>");
  assert.equal(h.rows()[0].children[1].children.length, 0);
  assert.deepEqual(h.component.request("/design question"), {strata_skill: "design"});
  assert.deepEqual(controls.normalize(catalog()).skills, catalog().skills);
});

test("selection matches only a recognized current leading slash token", async () => {
  const h = harness(); await h.ready();
  for (const text of ["/unknown question", "/design-more question", "Mention /design", " /design question", "/Design question"]) {
    assert.deepEqual(h.component.request(text), {});
  }
  assert.deepEqual(h.component.request("/design\n\nExact body"), {strata_skill: "design"});
  assert.deepEqual(h.component.request("/review"), {strata_skill: "review"});
  h.component.setBusy(true); assert.deepEqual(h.component.request("/design question"), {});
});

test("button selection preserves the entire draft and unknown slash body", async () => {
  const h = harness(); await h.ready();
  for (const draft of ["  Original\n\nbody", "/unknown literal\nbody"]) {
    h.type(draft); await h.click(h.button()); await h.click(h.rows()[0]);
    assert.equal(h.input.value, `/design ${draft}`); assert.equal(h.doc.activeElement, h.input);
    assert.equal(h.input.selectionStart, 8);
  }
});

test("replacing a slash selection preserves its exact multiline suffix and stays closed", async () => {
  const h = harness(); await h.ready();
  h.type("/rev\n\n  Original body", 4);
  assert.deepEqual(h.key("Enter"), {handled: true, prevented: true});
  assert.equal(h.input.value, "/review\n\n  Original body");
  assert.equal(h.button().getAttribute("aria-expanded"), "false");
  assert.deepEqual(h.key("Enter"), {handled: false, prevented: false}, "The next Enter can send the chosen draft");
  assert.equal(controls.insertSkill("/design\t exact\nbody", "review", catalog().skills), "/review\t exact\nbody");
});

test("actual option keys move focus, select once, and Escape restores input without editing", async () => {
  const h = harness(); await h.ready(); h.type("/");
  assert.deepEqual(h.key("ArrowUp"), {handled: true, prevented: true});
  assert.equal(h.doc.activeElement, h.rows()[1]); assert.equal(h.rows()[1].getAttribute("aria-selected"), "true");
  h.doc.activeElement.dispatchEvent({type: "keydown", key: "Enter", preventDefault() {}});
  assert.equal(h.input.value, "/review "); assert.equal(h.doc.activeElement, h.input);
  h.type("/"); h.key("ArrowDown"); const before = h.input.value;
  assert.deepEqual(h.key("Escape"), {handled: true, prevented: true});
  assert.equal(h.input.value, before); assert.equal(h.doc.activeElement, h.input);
});

test("unknown slash, Tab, Shift+Enter and IME preserve normal editing", async () => {
  const h = harness(); await h.ready(); h.type("/unknown");
  assert.deepEqual(h.key("Enter"), {handled: false, prevented: false}); assert.equal(h.input.value, "/unknown");
  h.type("/des");
  assert.deepEqual(h.key("Enter", {isComposing: true}), {handled: false, prevented: false});
  assert.deepEqual(h.key("Enter", {shiftKey: true}), {handled: false, prevented: false});
  assert.deepEqual(h.key("Tab"), {handled: false, prevented: false}); assert.equal(h.input.value, "/des");
  assert.equal(h.button().getAttribute("aria-expanded"), "false");
});

test("busy and disabled states dismiss options and never alter the draft", async () => {
  const h = harness(); await h.ready(); h.type("/des draft", 4); h.key("ArrowDown");
  h.component.setBusy(true); assert.equal(h.button().getAttribute("aria-expanded"), "false"); assert.ok(h.button().disabled);
  assert.equal(h.input.value, "/des draft"); assert.equal(h.doc.activeElement, h.input);
  h.component.setBusy(false); assert.equal(h.button().disabled, false);
  h.component.setEnabled(false); assert.equal(h.container.hidden, true); assert.deepEqual(h.component.request(), {});
});

test("newer catalog success or failure wins even when an aborted request ignores its signal", async () => {
  for (const failure of [false, true]) {
    const first = deferred(); let calls = 0;
    const h = harness(() => ++calls === 1 ? first.promise : failure ? Promise.reject(new Error("offline")) : Promise.resolve(catalog({enabled: false})));
    h.component.setEnabled(true); const old = h.component.refresh(); await h.component.refresh();
    assert.equal(h.signals[0].aborted, true); first.resolve(catalog()); await old;
    assert.equal(h.container.hidden, true); assert.deepEqual(h.component.request("/design draft"), {});
  }
});

test("Escape and MCP disabling abort refresh; a late result cannot reactivate selection", async () => {
  for (const action of ["escape", "disable"]) {
    const pending = deferred(), h = harness(() => pending.promise); h.type("/design exact\n draft"); h.input.focus();
    h.component.setEnabled(true); const load = h.component.refresh();
    if (action === "escape") assert.deepEqual(h.key("Escape"), {handled: true, prevented: true});
    else h.component.setEnabled(false);
    assert.equal(h.signals[0].aborted, true); pending.resolve(catalog()); await load;
    assert.equal(h.input.value, "/design exact\n draft"); assert.deepEqual(h.component.request(), {}); assert.equal(h.container.hidden, true);
  }
});

test("an enabled catalog failure gives a bounded retry state without silently selecting a skill", async () => {
  let offline = false;
  const h = harness(async () => { if (offline) throw new Error("Private backend detail"); return catalog(); });
  await h.ready(); h.type("/design draft"); offline = true; await h.component.refresh();
  assert.deepEqual(h.component.request(), {}); assert.equal(h.container.hidden, false);
  const status = h.container.children[1]; assert.match(status.textContent, /unavailable/); assert.doesNotMatch(status.textContent, /Private/);
  offline = false; await h.click(h.button()); assert.equal(h.rows().length, 2); assert.equal(h.input.value, "/design draft");
});

function appHarness(h, mcp = true, tools = 2) {
  const app = fs.readFileSync(path.join(__dirname, "app.js"), "utf8"), requests = [];
  const take = (a, b) => { const start = app.indexOf(a), end = app.indexOf(b, start); assert.ok(start >= 0 && end > start); return app.slice(start, end); };
  const elements = {input: h.input, chat: {lastElementChild: {}}, "stop-btn": {}, "send-btn": {}, "composer-hint": {}};
  const context = vm.createContext({skillControls: h.component, $: id => elements[id], settings: {mcp, thinking: "high", temperature: 0},
    mcpInfo: {tools}, health: {model: "fixture"}, messages: [], attachments: [], busy: null, AbortController, TextDecoder,
    renderAttachments() {}, autosize() {}, renderChat() {}, updateAssistant() {}, saveChat() {}, scrollDown() {},
    projectionLoaded: () => false, toast: (...args) => { throw new Error(`Unexpected toast: ${args.join(" ")}`); }, headers: () => ({}),
    fetch: async (url, options) => { requests.push({url, body: JSON.parse(options.body), draft: h.input.value});
      return {ok: true, body: {getReader: () => ({read: async () => ({done: true})})}}; }});
  vm.runInContext(take("function apiMessages()", "function setBusy(on)") + take("function setBusy(on)", '$("composer").onsubmit') +
    take('$("input").addEventListener("keydown",', "function autosize(") + take("function fileBlock(f)", "function renderAttachments("), context);
  return {context, requests, send: () => vm.runInContext("send()", context)};
}

test("actual app key handler selects before sending; send captures the skill before clearing input", async () => {
  const h = harness(); await h.ready(); h.type("/des\n\nExact body", 4);
  const app = appHarness(h);
  h.input.dispatchEvent({type: "keydown", key: "Enter", preventDefault() {}});
  assert.equal(app.requests.length, 0); assert.equal(h.input.value, "/design\n\nExact body");
  await app.send();
  assert.equal(app.requests.length, 1); assert.equal(app.requests[0].draft, "");
  assert.equal(app.requests[0].body.strata_skill, "design"); assert.equal(app.requests[0].body.strata_mcp, true);
  assert.equal(app.requests[0].body.messages[0].content, "/design\n\nExact body");
});

test("actual app leaves ordinary slash messages and MCP-off requests unmodified", async () => {
  for (const [draft, mcp] of [["/unknown question", true], ["/design question", false], [" /design question", true]]) {
    const h = harness(); await h.ready(); h.type(draft);
    const app = appHarness(h, mcp); await app.send();
    assert.equal(app.requests[0].body.messages[0].content, draft.trim()); assert.equal(app.requests[0].body.strata_skill, undefined);
    if (!mcp) assert.equal(app.requests[0].body.strata_mcp, undefined);
  }
});

test("an explicit skill enables the instruction-only adapter when MCP has no tools", async () => {
  for (const draft of ["/design question", "plain question", "/unknown question"]) {
    const h = harness(); await h.ready(); h.type(draft);
    const app = appHarness(h, true, 0); await app.send();
    assert.equal(app.requests[0].body.messages[0].content, draft);
    assert.equal(app.requests[0].body.strata_mcp, draft.startsWith("/design ") ? true : undefined);
    assert.equal(app.requests[0].body.strata_skill, draft.startsWith("/design ") ? "design" : undefined);
  }
});