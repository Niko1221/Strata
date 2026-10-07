"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const activity = require("../serve/web/tool-activity.js");
const source = fs.readFileSync(path.join(__dirname, "../serve/web/app.js"), "utf8");

function section(start, end) {
  const a = source.indexOf(start), b = source.indexOf(end, a + start.length);
  assert.ok(a >= 0 && b > a, `Production source section exists: ${start}`);
  return source.slice(a, b);
}

// This DOM boundary records focus and rebuilds only the rendered details nodes.
// Grouping, disclosure decisions and click dispatch all run from app.js itself.
function harness(message) {
  const listeners = {}, updates = [], focus = [], scroll = [];
  const document = {activeElement: null};
  let rows = [];
  const classList = names => ({contains: name => names.has(name), add: name => names.add(name), remove: name => names.delete(name)});
  const findSummary = selector => {
    const match = selector.match(/\.(tool-group|tool-call)\[data-tool(?:-group)?="(\d+)"\] > summary/);
    return match ? rows.find(row => row.kind === match[1] && row.index === match[2])?.summary || null : null;
  };
  const bubble = {
    classList: classList(new Set()),
    contains: node => rows.some(row => row.summary === node),
    querySelector: findSummary,
    querySelectorAll: selector => selector === ".tool-group[open]" ? rows.filter(row => row.kind === "tool-group" && row.open) : [],
    get innerHTML() { return this.html || ""; },
    set innerHTML(html) {
      this.html = html;
      rows = [...html.matchAll(/<details class="st-collapse (tool-group|tool-call)" data-tool(?:-group)?="(\d+)"[^>]*>/g)].map(match => {
        const kind = match[1], index = match[2];
        const body = {scrollTop: 0};
        const row = {kind, index, open: /\sopen(?=\s|>)/.test(match[0]),
          dataset: kind === "tool-group" ? {toolGroup: index} : {tool: index},
          classList: classList(new Set([kind])), querySelector: () => body};
        row.summary = {parentElement: row,
          matches: selector => selector.split(", ").includes(`.${kind} > summary`),
          closest: selector => selector === ".st-msg" ? el : selector === `.${kind} > summary` ? row.summary : null,
          focus: options => { document.activeElement = row.summary; focus.push({kind, index, options: {...options}}); }};
        return row;
      });
    }
  };
  const think = {hidden: true, open: false, dataset: {}};
  const thinking = {textContent: "", dataset: {}};
  const title = {}, meta = {}, copy = {};
  const el = {dataset: {i: "0"}, querySelector: selector => ({"details.think": think, ".thinking": thinking,
    ".think-title": title, ".st-bubble": bubble, ".meta-text": meta, "[data-msg-copy]": copy}[selector] || findSummary(selector))};
  const archive = [];
  const context = vm.createContext({StrataToolActivity: activity, URL, document, messages: [message], busy: null,
    settings: {show: true}, icon: () => "", fmt: value => String(value),
    store: {set: (key, value) => archive.push({key, value})},
    $: id => { assert.equal(id, "chat"); return {addEventListener: (event, callback) => { listeners[event] = callback; }}; },
    nearBottom: () => true, scrollDown: force => scroll.push(force), copyText: () => { throw new Error("Unexpected copy action"); }});
  const escapeLine = source.match(/^const esc = .*;$/m);
  assert.ok(escapeLine, "Use the production HTML escaping function");
  vm.runInContext(escapeLine[0] + "\n" + section("function inline(s)", "// ------------------------------------------------------------------ Chat") +
    section("function saveChat()", "function timeStr(") +
    section("const toolGroupViews =", "function renderChat()") +
    section('$("chat").addEventListener("click",', '$("chat").addEventListener("toggle",') +
    "\nconst repaint = updateAssistant; updateAssistant = (...args) => { observedUpdates.push(args); return repaint(...args); };",
    context, {filename: "app.js (actual tool activity integration)"});
  context.observedUpdates = updates;
  context.element = el;
  const paint = streaming => vm.runInContext(`updateAssistant(element, messages[0], ${!!streaming}, false)`, context);
  const click = (kind, index) => {
    const target = findSummary(`.${kind}[data-${kind === "tool-group" ? "tool-group" : "tool"}="${index}"] > summary`);
    assert.ok(target, `Rendered ${kind} ${index} summary exists`);
    document.activeElement = target;
    const event = {target, prevented: false, preventDefault() { this.prevented = true; }};
    listeners.click(event);
    assert.equal(event.prevented, true, "Custom disclosure prevents the native double toggle");
    return target;
  };
  paint(false);
  updates.length = 0;
  return {context, el, bubble, document, updates, focus, scroll, archive, paint, click,
    row: (kind, index) => rows.find(row => row.kind === kind && row.index === String(index)),
    html: () => vm.runInContext("answerHtml(messages[0])", context),
    save: () => vm.runInContext("saveChat()", context),
    tool: event => { context.event = event; vm.runInContext("onTool(messages[0], event)", context); }};
}

const call = (id, at = 0, extra = {}) => ({id, name: `sample__${id}`, server: "sample", tool: id, at,
  state: "done", ok: true, arguments: {path: `${id}.txt`, nested: {value: `<${id}>`}},
  result: `${id} result <unchanged>`, chars: 20, truncated: false, ms: 10, ...extra});
const answer = tools => ({role: "assistant", text: "", reasoning: "", tools});
const clone = value => JSON.parse(JSON.stringify(value));

function ordered(html, strings) {
  let previous = -1;
  for (const string of strings) {
    const position = html.indexOf(string, previous + 1);
    assert.ok(position > previous, `${string} follows the preceding answer fragment`);
    previous = position;
  }
}

test("answerHtml preserves text/tool order and groups only adjacent calls at the same answer offset", () => {
  const m = answer([call("first", 0, {open: true}), call("second", 8, {open: true}),
    call("third", 8, {open: true}), call("fourth", 17, {open: true}), call("fifth", 17, {open: true})]);
  m.text = "Before\n\nBetween\n\nAfter";
  const before = clone(m), h = harness(m), html = h.html();
  ordered(html, ['data-tool="0"', "<p>Before</p>", 'data-tool-group="1"', 'data-tool="1"', 'data-tool="2"',
    "<p>Between</p>", 'data-tool-group="3"', 'data-tool="3"', 'data-tool="4"', "<p>After</p>"]);
  assert.equal((html.match(/class="st-collapse tool-group"/g) || []).length, 2);
  assert.deepEqual(m, before, "Rendering never rewrites tool offsets, arguments or results");
});

test("group disclosure stays out of serialized history and preserves original tool indexes", () => {
  const m = answer([call("first"), call("second"), call("third")]), before = clone(m), h = harness(m);
  h.save();
  h.context.busy = {msg: m};
  assert.equal(h.row("tool-group", 0).open, false);
  const originalSummary = h.click("tool-group", 0);
  assert.equal(h.row("tool-group", 0).open, true);
  assert.deepEqual(m, before, "Opening a group adds no disclosure property to archived messages");
  h.save();
  assert.deepEqual(clone(h.archive[1].value), clone(h.archive[0].value), "Group disclosure leaves the actual saved history unchanged");
  ordered(h.bubble.innerHTML, ['data-tool="0"', 'data-tool="1"', 'data-tool="2"']);
  assert.equal(h.updates[0][1], m);
  assert.equal(h.updates[0][2], true);
  assert.equal(h.updates[0][3], false, "Action repaint explicitly disables chat following");
  assert.notEqual(h.document.activeElement, originalSummary, "Focus points to the rebuilt summary");
  assert.equal(h.document.activeElement, h.row("tool-group", 0).summary);
  assert.deepEqual(h.focus.at(-1), {kind: "tool-group", index: "0", options: {preventScroll: true}});
  assert.deepEqual(h.scroll, [], "Disclosure does not scroll a user reading the chat");
});

test("opening a nested call addresses its original record and exposes unchanged escaped arguments/results", () => {
  const m = answer([call("first"), call("second"), call("third")]), before = clone(m.tools), h = harness(m);
  h.click("tool-group", 0);
  const originalSummary = h.click("tool-call", 1);
  assert.equal(m.tools[1].open, true);
  assert.equal(m.tools[0].open, undefined);
  assert.equal(m.tools[2].open, undefined);
  assert.match(h.bubble.innerHTML, /second result &lt;unchanged&gt;/);
  assert.match(h.bubble.innerHTML, /&quot;value&quot;: &quot;&lt;second&gt;&quot;/);
  assert.doesNotMatch(h.bubble.innerHTML, /first result|third result/);
  m.tools.forEach((record, index) => { const {open, ...rest} = record; assert.deepEqual(rest, before[index]); });
  assert.equal(h.updates.at(-1)[3], false);
  assert.notEqual(h.document.activeElement, originalSummary);
  assert.equal(h.document.activeElement, h.row("tool-call", 1).summary);
  assert.deepEqual(h.focus.at(-1), {kind: "tool-call", index: "1", options: {preventScroll: true}});
});

test("a streamed call arriving after group opening retains disclosure, focus and original indexes", () => {
  const m = answer([call("first"), call("second")]), before = clone(m.tools), h = harness(m);
  h.click("tool-group", 0);
  h.row("tool-group", 0).querySelector(".tool-group__body").scrollTop = 137;
  h.tool({event: "call", id: "third", name: "sample__third", server: "sample", tool: "third", arguments: {path: "third.txt"}, round: 2});
  h.paint(true);
  assert.equal(h.row("tool-group", 0).open, true);
  ordered(h.bubble.innerHTML, ['data-tool="0"', 'data-tool="1"', 'data-tool="2"']);
  assert.deepEqual(m.tools.slice(0, 2), before);
  assert.equal(m.tools[2].state, "running");
  assert.equal(h.row("tool-group", 0).querySelector(".tool-group__body").scrollTop, 137);
  assert.equal(h.document.activeElement, h.row("tool-group", 0).summary);
});

test("explicitly closing a group remains closed when its child is open or a new call arrives", () => {
  const m = answer([call("first"), call("second")]), h = harness(m);
  h.click("tool-group", 0);
  h.click("tool-call", 1);
  h.click("tool-group", 0);
  assert.equal(h.row("tool-group", 0).open, false);
  assert.equal(m.tools[1].open, true, "Closing a group preserves its child's prior disclosure");
  h.tool({event: "call", id: "third", name: "sample__third", arguments: {path: "third.txt"}, round: 2});
  h.paint(true);
  assert.equal(h.row("tool-group", 0).open, false, "An explicit false overrides child-open fallback");
  assert.equal((h.bubble.innerHTML.match(/<details/g) || []).length, 1, "Closed groups defer all individual rows");
  h.click("tool-group", 0);
  assert.equal(h.row("tool-call", 1).open, true);
  assert.equal(h.row("tool-call", 2).index, "2");
});

test("independent answer positions keep independent group disclosure through repaint", () => {
  const m = answer([call("first", 0), call("second", 0), call("third", 8), call("fourth", 8)]);
  m.text = "Between\nAfter";
  const h = harness(m);
  h.click("tool-group", 2);
  h.paint(true);
  assert.equal(h.row("tool-group", 0).open, false);
  assert.equal(h.row("tool-group", 2).open, true);
  ordered(h.bubble.innerHTML, ['data-tool-group="0"', "<p>Between</p>", 'data-tool-group="2"', 'data-tool="2"', 'data-tool="3"', "<p>After</p>"]);
  h.click("tool-group", 2);
  h.click("tool-group", 0);
  assert.equal(h.row("tool-group", 0).open, true);
  assert.equal(h.row("tool-group", 2).open, false);
});
