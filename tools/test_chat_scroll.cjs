"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync(require.resolve("../serve/web/app.js"), "utf8");
function section(start, end) {
  const from = source.indexOf(start), to = source.indexOf(end, from);
  assert.ok(from >= 0 && to > from, `Missing app section: ${start}`);
  return source.slice(from, to);
}

function fixture() {
  const details = {hidden: true, open: false, dataset: {}};
  let reasoning = "", answer = "", outerTop = 400, innerTop = 0;
  const main = {clientHeight: 600, get scrollHeight() { return 1000 + body.clientHeight + answer.length; },
    get scrollTop() { return outerTop; }, set scrollTop(value) { outerTop = Math.max(0, Math.min(value, this.scrollHeight - this.clientHeight)); }};
  // Match the real capped .thinking viewport. Its scroll geometry is unavailable while closed.
  const body = {dataset: {}, get clientHeight() { return details.open && !details.hidden ? Math.min(420, reasoning.length) : 0; },
    get scrollHeight() { return details.open && !details.hidden ? reasoning.length : 0; },
    get scrollTop() { return innerTop; }, set scrollTop(value) { innerTop = Math.max(0, Math.min(value, this.scrollHeight - this.clientHeight)); },
    get textContent() { return reasoning; }, set textContent(value) { reasoning = value; this.scrollTop = innerTop; }};
  const bubble = {classList: {add() {}, remove() {}},
    get innerHTML() { return answer; }, set innerHTML(value) { answer = value; }};
  const fields = {"details.think": details, ".thinking": body, ".st-bubble": bubble,
    ".think-title": {}, ".meta-text": {}, "[data-msg-copy]": {}};
  const el = {querySelector: key => fields[key]};
  const elements = {"chat-scroll": main, input: {value: "Synthetic prompt"}, chat: {lastElementChild: el}};
  const sandbox = {$: id => elements[id], settings: {show: true}, el, frame: 1,
    fmt: n => String(n), answerHtml: m => m.text, m: {text: "", reasoning: ""}};
  vm.createContext(sandbox);
  vm.runInContext(section("function updateAssistant(", "function renderChat()") +
    section("function nearBottom(", '\n$("chat").addEventListener'), sandbox);
  const paintSource = section("  const paint = () =>", "\n  try {").trim();
  vm.runInContext(`${paintSource}\nglobalThis.paint = paint;`, sandbox);
  return {sandbox, main, body, details, el,
    paint(message) { sandbox.m = message; sandbox.paint(); },
    prepareReasoning(text) { details.hidden = false; details.open = true; details.dataset.touched = "1";
      body.textContent = text; body.scrollTop = body.scrollHeight; main.scrollTop = main.scrollHeight; }};
}

const atBottom = scroller => scroller.scrollHeight - scroller.clientHeight;

test("the first large Thinking delta follows both the newly opened inner pane and outer chat", () => {
  const f = fixture();
  f.paint({text: "", reasoning: "Synthetic thought. ".repeat(200)});
  assert.equal(f.details.open, true);
  assert.ok(f.body.scrollHeight > f.body.clientHeight);
  assert.equal(f.body.scrollTop, atBottom(f.body));
  assert.equal(f.main.scrollTop, atBottom(f.main));
});

test("expanded Thinking follows later deltas after its visible height has reached the cap", () => {
  const f = fixture(); f.prepareReasoning("Thought. ".repeat(100));
  f.paint({text: "", reasoning: "Thought. ".repeat(400)});
  assert.equal(f.body.clientHeight, 420);
  assert.equal(f.body.scrollTop, atBottom(f.body));
  assert.equal(f.main.scrollTop, atBottom(f.main));
});

test("manual scroll-up inside Thinking retains that reading position while new reasoning arrives", () => {
  const f = fixture(); f.prepareReasoning("Thought. ".repeat(200)); f.body.scrollTop = 100;
  f.paint({text: "", reasoning: "Thought. ".repeat(400)});
  assert.equal(f.body.scrollTop, 100);
  assert.equal(f.main.scrollTop, atBottom(f.main));
});

test("a large answer delta keeps a near-bottom chat following despite more than 120px growth", () => {
  const f = fixture();
  f.main.scrollTop = atBottom(f.main) - 80;
  f.paint({text: "Synthetic answer. ".repeat(200), reasoning: ""});
  assert.equal(f.main.scrollTop, atBottom(f.main));
});

test("manual outer scroll-up before a scheduled paint prevents snapping to the answer", () => {
  const f = fixture();
  f.main.scrollTop = 40;
  f.paint({text: "Synthetic answer. ".repeat(200), reasoning: ""});
  assert.equal(f.main.scrollTop, 40);
});

test("returning to near-bottom resumes following on the next delta", () => {
  const f = fixture(); f.main.scrollTop = 40;
  f.paint({text: "Synthetic answer. ".repeat(100), reasoning: ""});
  assert.equal(f.main.scrollTop, 40);
  f.main.scrollTop = atBottom(f.main);
  f.paint({text: "Synthetic answer. ".repeat(200), reasoning: ""});
  assert.equal(f.main.scrollTop, atBottom(f.main));
});

test("manual Thinking open/closed choices and Show thinking remain unchanged", () => {
  for (const mode of ["manually-closed", "show-disabled"]) {
    const f = fixture();
    if (mode === "manually-closed") f.details.dataset.touched = "1";
    else f.sandbox.settings.show = false;
    f.paint({text: "", reasoning: "Thought. ".repeat(200)});
    assert.equal(f.details.open, false, mode);
    assert.equal(f.body.scrollTop, 0, mode);
  }
  const f = fixture(); f.prepareReasoning("Thought. ".repeat(100));
  f.paint({text: "The answer.", reasoning: "Thought. ".repeat(100)});
  assert.equal(f.details.open, true);
});

async function finalChunk(away) {
  const f = fixture(), noop = () => {};
  let delivered = false, pendingPaint = false, canceledPaint = false;
  Object.assign(f.sandbox, {AbortController, TextDecoder, performance: {now: () => 1},
    requestAnimationFrame: () => { pendingPaint = true; return 1; },
    cancelAnimationFrame: () => { canceledPaint = true; }, busy: null, historyBusy: false,
    pendingAttachmentReads: 0, legacyRouteResolved: true, messages: [], attachments: [],
    health: {model: "fixture"}, selectedModel: "", settings: {show: true, thinking: "high", temperature: 0, max: "", mcp: false},
    mcpInfo: {tools: 0}, renderAttachments: noop, resetAttachmentReads: noop, autosize: noop,
    renderChat: noop, setBusy: noop, projectionLoaded: () => false, saveChat: noop,
    loadContextProfiles: noop, headers: () => ({}), prepareChatContext: async () => ({}),
    apiMessages: () => [{role: "user", content: "Synthetic prompt"}], toast: () => assert.fail("Unexpected send error"),
    fetch: async () => {
      if (away) f.main.scrollTop = 40; // The user scrolls up while waiting for the response.
      return {ok: true, body: {getReader: () => ({read: async () => {
        if (delivered) return {done: true}; delivered = true;
        const data = {choices: [{delta: {content: "Synthetic answer. ".repeat(200)}}]};
        return {done: false, value: new TextEncoder().encode(`data: ${JSON.stringify(data)}\n\ndata: [DONE]\n\n`)};
      }})}};
    }});
  vm.runInContext(section("async function send() {", '$("composer").onsubmit'), f.sandbox);
  await f.sandbox.send();
  assert.equal(pendingPaint, true); assert.equal(canceledPaint, true);
  return f;
}

test("a stream ending before its animation frame still follows the final unpainted answer", async () => {
  const f = await finalChunk(false);
  assert.equal(f.main.scrollTop, atBottom(f.main));
});

test("the final unpainted answer preserves a user who scrolled up during the request", async () => {
  const f = await finalChunk(true);
  assert.equal(f.main.scrollTop, 40);
});
