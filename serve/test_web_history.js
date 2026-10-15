// CPU-only behavior tests: execute the real Chat code and attachment serialization in a Node VM.
// The DOM is deliberately small. Layout, focus, keyboard and real DOM wiring need browser tests too.
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class Element {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {}; this.style = {};
    this.attrs = {}; this.listeners = {}; this.value = ""; this.hidden = false; this.disabled = false;
    this.scrollHeight = 100; this.clientHeight = 100; this.scrollTop = 0; this._html = "";
    this.textContent = ""; this.selectors = new Map();
    this.classList = {
      add: (...names) => { this.className = [...new Set([...(this.className || "").split(/\s+/), ...names])].join(" "); },
      remove: (...names) => { this.className = (this.className || "").split(/\s+/).filter(n => !names.includes(n)).join(" "); },
      contains: name => (this.className || "").split(/\s+/).includes(name),
      toggle: (name, on) => { if (on === undefined) on = !this.classList.contains(name); this.classList[on ? "add" : "remove"](name); return on; },
    };
  }
  set innerHTML(html) { this._html = String(html); this.children = []; this.selectors.clear(); }
  get innerHTML() { return this._html; }
  appendChild(el) { this.children.push(el); el.parentElement = this; return el; }
  append(...els) { for (const el of els) this.appendChild(typeof el === "string" ? Object.assign(new Element(), {textContent: el}) : el); }
  prepend(el) { this.children.unshift(el); el.parentElement = this; }
  remove() { this.removed = true; if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(e => e !== this); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  addEventListener(k, fn) { (this.listeners[k] ||= []).push(fn); }
  querySelector(selector) {
    const index = selector.match(/^\[data-i="(\d+)"\]$/);
    if (index) return this.children.find(e => String(e.dataset.i) === index[1]) || null;
    if (selector.startsWith(".")) {
      const found = this.children.find(e => (e.className || "").split(/\s+/).includes(selector.slice(1)));
      if (found) return found;
    }
    if (!this.selectors.has(selector)) {
      const el = new Element(selector.startsWith("details") ? "details" : "div");
      el.parentElement = this; this.selectors.set(selector, el);
    }
    return this.selectors.get(selector);
  }
  querySelectorAll(selector) {
    if (selector === ".st-msg") return this.children.filter(e => (e.className || "").split(/\s+/).includes("st-msg"));
    return [];
  }
  insertAdjacentHTML(position, html) { this._html += html; }
  get firstChild() { return this.children[0] || this.querySelector("first-child"); }
  get lastElementChild() { return this.children.at(-1); }
  focus() {} select() {} scrollIntoView() {} click() { if (this.onclick) this.onclick(); }
}

function flatten(el) {
  return `${el.innerHTML} ${el.textContent} ${el.className || ""} ${Object.keys(el.attrs).join(" ")} ` + el.children.map(flatten).join(" ");
}

const source = fs.readFileSync(path.join(__dirname, "web", "app.js"), "utf8");
const start = source.indexOf("// ------------------------------------------------------------------ Chat\n");
const end = source.indexOf("// ------------------------------------------------------------------ the sampling drawer\n", start);
assert(start >= 0 && end > start, "the actual Chat section must be loaded");
const chatSource = source.slice(start, end);

function setup({enabled = true, history = []} = {}) {
  const elements = new Map(), saved = new Map([["chat", JSON.parse(JSON.stringify(history))]]);
  if (enabled) saved.set("editHistory", true);
  const requests = [], toasts = [], blobs = [], pending = [];
  let hold = false, nextFrame = 1, failure = null;
  const get = id => {
    if (id === "history-undo") return elements.has(id) && !elements.get(id).removed ? elements.get(id) : null;
    if (!elements.has(id)) {
      const el = new Element();
      el.before = sibling => elements.set(sibling.id, sibling);
      elements.set(id, el);
    }
    return elements.get(id);
  };
  const context = vm.createContext({
    console, AbortController, TextDecoder, TextEncoder, Blob, Date, JSON, performance,
    $: get, document: {createElement: tag => new Element(tag), querySelectorAll: () => []},
    store: {get: (k, fallback) => saved.has(k) ? saved.get(k) : fallback, set: (k, v) => saved.set(k, JSON.parse(JSON.stringify(v)))},
    icon: name => `<svg data-icon="${name}"></svg>`,
    esc: s => String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c])),
    fmt: n => String(n), markdown: s => String(s),
    health: {model: "mock-model", images: true}, mcpInfo: {tools: 0},
    headers: () => ({"Content-Type": "application/json"}), projectionLoaded: () => false,
    toast: (...args) => toasts.push(args), copyText: () => {}, innerHeight: 800,
    requestAnimationFrame: () => nextFrame++, cancelAnimationFrame: () => {},
    setTimeout: () => 1, clearTimeout: () => {},
    URL: {createObjectURL: b => { blobs.push(b); return "blob:mock"; }, revokeObjectURL: () => {}},
    fetch: async (url, options) => {
      assert.equal(url, "v1/chat/completions", "the test must never contact a real endpoint");
      requests.push(JSON.parse(options.body));
      if (failure) { const error = failure; failure = null; throw error; }
      let emitted = false;
      return {ok: true, body: {getReader: () => ({read: () => {
        if (hold) return new Promise((resolve, reject) => {
          const abort = () => reject(Object.assign(new Error("stopped"), {name: "AbortError"}));
          if (options.signal.aborted) abort(); else options.signal.addEventListener("abort", abort, {once: true});
          pending.push({resolve, reject});
        });
        if (emitted) return Promise.resolve({done: true});
        emitted = true;
        return Promise.resolve({done: false, value: new TextEncoder().encode('data: {"choices":[{"delta":{"content":"fresh answer"}}]}\n\ndata: [DONE]\n\n')});
      }})}};
    },
  });
  vm.runInContext(chatSource, context, {filename: "web/app.js (Chat section)"});
  vm.runInContext("renderChat()", context);
  const run = code => vm.runInContext(code, context);
  const value = code => JSON.parse(JSON.stringify(run(code)));
  const settle = async () => {
    for (let i = 0; i < 50; i++) { await new Promise(resolve => setImmediate(resolve)); if (!run("busy")) return; }
    throw new Error("mock generation did not settle");
  };
  return {run, value, settle, requests, toasts, blobs, saved, get, setHold: value => { hold = value; },
          failNext: () => { failure = new Error("mock network failure"); }, pending};
}

const user = (text, extra = {}) => ({role: "user", text, time: 1, ...extra});
const assistant = (text, extra = {}) => ({role: "assistant", text, time: 2, ...extra});
const tool = {id: "old-call", name: "old_tool", tool: "old_tool", arguments: {old: true}, round: 0,
              result: "old secret result", state: "done", at: 0, rat: 0, ok: true};
const attached = user("original question", {
  images: [{kind: "image", name: "pic.png", url: "data:image/png;base64,AAA"}],
  files: [{kind: "file", name: "notes.md", text: "attached evidence"}],
});
const history = () => [attached, assistant("old answer", {reasoning: "old thoughts", tools: [tool]}), user("later question"), assistant("later answer")];

const cases = {
  async default_off() {
    const a = setup({enabled: false, history: [user("first"), assistant("answer")]});
    a.run("renderChat()");
    assert.equal(a.run("historyEditing"), false);
    assert.doesNotMatch(flatten(a.get("chat")), /history-actions|data-msg-(edit|remove|delete|retry|regenerate)/);
    const before = a.value("messages");
    a.run('editMessage(0); removeHistoryFrom(0); retryMessage(1);');
    assert.deepEqual(a.value("messages"), before);
    assert.equal(a.requests.length, 0);
    a.get("input").value = "next"; await a.run("send()");
    assert.deepEqual(a.requests[0], {model: "mock-model", messages: [
      {role: "user", content: "first"}, {role: "assistant", content: "answer"}, {role: "user", content: "next"}],
      stream: true, reasoning_effort: "high", temperature: 0.6, top_p: 0.95, top_k: 20});
  },
  async user_edit() {
    const a = setup({history: history()});
    a.run('editMessage(0); saveMessageEdit(0, "changed question")'); await a.settle();
    assert.equal(a.requests.length, 1);
    const request = a.requests[0].messages;
    assert.equal(request.length, 1);
    assert.deepEqual(request[0], {role: "user", content: [
      {type: "text", text: "changed question\n\nFile: notes.md\n```\nattached evidence\n```"},
      {type: "image_url", image_url: {url: "data:image/png;base64,AAA"}}]});
    const messages = a.value("messages");
    assert.equal(messages.length, 2); assert.equal(messages[1].text, "fresh answer");
    assert.deepEqual(messages[0].images, attached.images); assert.deepEqual(messages[0].files, attached.files);
    assert.doesNotMatch(JSON.stringify(request), /old answer|old-call|old secret|later question/);
  },
  async assistant_edit() {
    const old = assistant("wrong", {reasoning: "private", tools: [tool], error: "failure", meta: "old tokens", stopped: true, thinkSecs: 12, limit: 3});
    const a = setup({history: [user("question"), old, user("descendant"), assistant("descendant answer")]});
    a.run('editMessage(1); saveMessageEdit(1, "corrected answer")'); await a.settle();
    assert.equal(a.requests.length, 0);
    const messages = a.value("messages"); assert.equal(messages.length, 2);
    assert.equal(messages[1].text, "corrected answer");
    for (const key of ["reasoning", "tools", "error", "stopped", "thinkSecs", "limit"]) assert(!messages[1][key], `old ${key} survived`);
    assert.notEqual(messages[1].meta, "old tokens");
    assert.deepEqual(a.value("apiMessages()"), [{role: "user", content: "question"}, {role: "assistant", content: "corrected answer"}]);
  },
  async truncate() {
    const a = setup({history: history()}); a.run("removeHistoryFrom(1)");
    assert.deepEqual(a.value("messages"), [attached]);
    assert.equal(a.requests.length, 0);
    a.run("removeHistoryFrom(0)"); assert.deepEqual(a.value("messages"), []);
  },
  async retry() {
    const a = setup({history: history()}); a.run("retryMessage(1)"); await a.settle();
    assert.equal(a.requests.length, 1);
    const request = a.requests[0].messages; assert.equal(request.length, 1); assert.equal(request[0].role, "user");
    assert.match(request[0].content[0].text, /original question/);
    assert.doesNotMatch(JSON.stringify(request), /old answer|old-call|old secret|later question/);
    assert.equal(a.value("messages").length, 2);
  },
  async stop_then_edit() {
    const a = setup({history: [user("before"), assistant("before answer")]});
    a.setHold(true); a.get("input").value = "running";
    const sending = a.run("send()"); await new Promise(resolve => setImmediate(resolve));
    const before = a.value("messages");
    a.run('editMessage(0); saveMessageEdit(0, "blocked"); removeHistoryFrom(0); retryMessage(1); undoHistory();');
    assert.deepEqual(a.value("messages"), before); assert.equal(a.requests.length, 1);
    a.get("stop-btn").onclick();
    a.run("removeHistoryFrom(0)"); assert.deepEqual(a.value("messages"), before, "abort must finalize before editing");
    await sending; assert.equal(a.run("busy"), null);
    assert.equal(a.value("messages").at(-1).stopped, true);
    a.setHold(false); a.run('editMessage(0); saveMessageEdit(0, "after stop")'); await a.settle();
    assert.equal(a.requests.length, 2); assert.equal(a.requests[1].messages[0].content, "after stop");
    assert.equal(a.value("messages").at(-1).text, "fresh answer");
  },
  async undo() {
    const a = setup({history: history()}); const before = a.value("messages");
    a.run("removeHistoryFrom(1); undoHistory()");
    assert.deepEqual(a.value("messages"), before);
    assert.match(JSON.stringify(a.value("apiMessages()")), /old-call/);
    assert.equal(a.run("historyUndo"), null);
  },
  async feature_generation_undo() {
    const a = setup({history: history()}); const before = a.value("messages");
    a.run('editMessage(0); saveMessageEdit(0, "replacement")'); await a.settle();
    assert(a.run("historyUndo")); a.run("undoHistory()");
    assert.deepEqual(a.value("messages"), before, "feature-generated answer must not alter the Undo snapshot");
  },
  async new_send_invalidates_undo() {
    const a = setup({history: history()}); a.run("removeHistoryFrom(1)");
    const undoActions = a.toasts.map(t => t[4]).filter(x => x && x.label === "Undo");
    a.get("input").value = "new branch"; await a.run("send()");
    const after = a.value("messages"); assert.equal(a.run("historyUndo"), null);
    a.run("undoHistory()"); for (const action of undoActions) action.run();
    assert.deepEqual(a.value("messages"), after);
  },
  async new_chat_invalidates_undo() {
    const a = setup({history: history()}); a.run("removeHistoryFrom(1)");
    const oldActions = a.toasts.map(t => t[4]).filter(x => x && x.label === "Undo");
    a.get("new-btn").onclick(); assert.equal(a.run("historyUndo"), null);
    a.run("undoHistory()"); for (const action of oldActions) action.run();
    assert.deepEqual(a.value("messages"), []);
  },
  async composer_during_edit() {
    const a = setup({history: history()}); a.run("editMessage(0)");
    a.get("input").value = "must not send"; await a.run("send()");
    a.get("composer").onsubmit({preventDefault() {}}); await a.settle();
    assert.equal(a.requests.length, 0); assert.equal(a.value("messages").length, 4);
  },
  async reasoning_only() {
    const a = setup({enabled: false, history: [user("first"), assistant("", {reasoning: "only thoughts", stopped: true})]});
    a.get("input").value = "next"; await a.run("send()");
    assert.deepEqual(a.requests[0].messages[1], {role: "assistant", content: "", reasoning_content: "only thoughts"});
  },
  async export() {
    const a = setup({history: [user("question"), assistant("old", {reasoning: "old reasoning", tools: [tool]})]});
    a.run('editMessage(1); saveMessageEdit(1, "corrected")'); a.get("export-btn").onclick();
    assert.equal(a.blobs.length, 1); const md = await a.blobs[0].text();
    assert.match(md, /question/); assert.match(md, /corrected/); assert.doesNotMatch(md, /old reasoning|old secret result|old_tool/);
  },
  async legacy_undo_after_enabling() {
    const a = setup({enabled: false, history: [user("old chat"), assistant("old answer")]});
    a.get("new-btn").onclick();
    const oldUndo = a.toasts.at(-1)[4]; assert.equal(oldUndo.label, "Undo");
    a.run("historyEditing = true; renderChat()");
    a.get("input").value = "new chat"; await a.run("send()");
    const after = a.value("messages"); oldUndo.run();
    assert.deepEqual(a.value("messages"), after, "a toast from feature-off must not overwrite a new request after enabling");
  },
  async reloaded_attachments() {
    const original = setup({history: history()}); original.run("saveChat()");
    const a = setup({history: original.saved.get("chat")});
    a.run("retryMessage(1)"); await a.settle();
    assert.equal(a.requests.length, 0, "regenerate must not silently omit attachment contents lost on reload");
    assert.equal(a.value("messages").length, 4);
  },
  async network_failure() {
    const a = setup({history: history()}); const before = a.value("messages");
    a.failNext(); await a.run('editMessage(0); saveMessageEdit(0, "replacement")');
    assert.equal(a.run("busy"), null); assert(a.run("historyUndo"));
    assert.equal(a.value("messages").at(-1).error, "mock network failure");
    a.run("undoHistory()"); assert.deepEqual(a.value("messages"), before);
  },
};

(async () => {
  const name = process.argv[2];
  if (name) { assert(cases[name], `unknown test ${name}`); await cases[name](); console.log(`ok ${name}`); }
  else for (const [name, test] of Object.entries(cases)) { await test(); console.log(`ok ${name}`); }
})().catch(err => { console.error(err.stack || err); process.exitCode = 1; });
