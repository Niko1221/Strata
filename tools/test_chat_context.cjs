"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const context = require("../serve/web/chat-context.js");
const wire = m => m.role === "user" ? [{role: "user", content: m.text}] : m.error || !m.text ? [] : [{role: "assistant", content: m.text}];
const archive = () => [
  {role: "user", text: "A".repeat(5000)}, {role: "assistant", text: "B".repeat(4000)},
  {role: "user", text: "recent question"}, {role: "assistant", text: "recent answer"},
  {role: "user", text: "current question"}, {role: "assistant", text: ""}
];
function harness() {
  const counted = [], summaries = [];
  return {counted, summaries,
    count: async (messages, mcp, cap) => {
      const result = {input_tokens: Math.ceil(JSON.stringify(messages).length / 5), max_context: 1024, effective_max_tokens: cap || 128, count_exact: true};
      counted.push({messages, mcp, cap, ...result});
      return result;
    },
    summarize: async (messages, cap) => { summaries.push({messages, cap}); return "Preserved goals and decisions."; }
  };
}
test("small chat does not summarize", async () => {
  const h = harness();
  const r = await context.prepare({archive: [{role: "user", text: "hello"}], wire, ...h});
  assert.equal(r.changed, false); assert.equal(h.summaries.length, 0);
});
test("80% threshold compacts even while reserved output fits", async () => {
  const h = harness(); let first = true;
  const count = async (...args) => { const c = await h.count(...args); if (first) { c.input_tokens = 820; first = false; } return c; };
  const r = await context.prepare({archive: archive(), wire, ...h, count});
  assert.equal(r.changed, true);
});
test("already overflowing history chunks safely, retaining recent rounds and archive", async () => {
  const h = harness(), original = archive(), before = JSON.stringify(original);
  const r = await context.prepare({archive: original, wire, ...h});
  assert.equal(r.changed, true); assert.ok(r.chunks > 1); assert.equal(r.state.through, 2);
  assert.equal(JSON.stringify(original), before);
  const active = context.active(original, r.state, wire);
  assert.equal(active[1].content, "recent question"); assert.equal(active[2].content, "recent answer");
  assert.equal(active[3].content, "current question");
  for (const c of h.counted.filter(c => c.cap)) { assert.equal(c.mcp, false); }
  for (const s of h.summaries) {
    const requestTokens = Math.ceil(JSON.stringify(s.messages).length / 5);
    assert.ok(requestTokens + s.cap + 8 <= 1024 * 0.75);
    assert.ok(!s.messages.some(m => m.tool_calls));
  }
  assert.ok(r.count.input_tokens + r.reserve + 8 <= r.count.max_context);
});
test("explicit output cap above quarter is honored and triggers below 80%", async () => {
  const h = harness(); let first = true;
  const count = async (...args) => { const c = await h.count(...args); c.max_context = 4096; if (!args[2]) c.effective_max_tokens = 1500; if (first) { c.input_tokens = 2800; first = false; } return c; };
  const r = await context.prepare({archive: archive(), wire, ...h, count});
  assert.equal(r.changed, true); assert.equal(r.reserve, 1500); assert.equal(r.replyLimit, 1500);
  assert.equal(context.budget({input_tokens: 1, max_context: 65536, effective_max_tokens: 0}).reserve, 8192);
});
test("unlimited completion uses headroom only; impossible explicit cap is rejected clearly", async () => {
  const count = async () => ({input_tokens: 100, max_context: 65536, effective_max_tokens: 0});
  const result = await context.prepare({archive: [{role: "user", text: "hello"}], wire, count,
    summarize: async () => { throw new Error("must not summarize"); }});
  assert.equal(result.replyLimit, null); assert.equal(result.reserve, 8192);
  assert.equal(context.budget({input_tokens: 1, max_context: 1024, effective_max_tokens: 0}).reserve, 256);
  await assert.rejects(context.prepare({archive: [{role: "user", text: "hello"}], wire,
    count: async () => ({input_tokens: 1, max_context: 1024, effective_max_tokens: 1016}),
    summarize: async () => { throw new Error("must not summarize"); }}), /configured reply limit/);
});
test("disabled auto fails closed on overflow and manual still compacts", async () => {
  const h = harness();
  await assert.rejects(context.prepare({archive: archive(), wire, ...h, auto: false}), /Compact now/);
  assert.equal(h.summaries.length, 0);
  assert.equal((await context.prepare({archive: archive(), wire, ...h, auto: false, force: true})).changed, true);
});
test("summary error or cancellation preserves context without retry", async () => {
  const original = archive(), before = JSON.stringify(original); let calls = 0;
  const state = {summary: "existing summary", through: 0};
  for (const error of [new Error("failed"), new DOMException("Stopped", "AbortError")]) {
    calls = 0;
    await assert.rejects(context.prepare({archive: original, state, wire, ...harness(), summarize: async () => { calls++; throw error; }}));
    assert.equal(calls, 1); assert.deepEqual(state, {summary: "existing summary", through: 0});
    assert.equal(JSON.stringify(original), before);
  }
});
test("empty summary, changed model limit and oversized recent turn fail closed", async () => {
  await assert.rejects(context.prepare({archive: archive(), wire, ...harness(), summarize: async () => ""}), /empty summary/);
  const h = harness(); let calls = 0;
  await assert.rejects(context.prepare({archive: archive(), wire, ...h, count: async (...args) => {
    const c = await h.count(...args); if (++calls > 1) c.max_context = 2048; return c;
  }}), /context changed/);
  await assert.rejects(context.prepare({archive: [{role: "user", text: "X".repeat(10000)}], wire, ...harness()}), /recent messages alone/);
});
test("successive compaction summarizes only new old rounds plus prior summary", async () => {
  const original = archive(); const first = await context.prepare({archive: original, wire, ...harness()});
  original.push({role: "assistant", text: "old second answer"}, {role: "user", text: "next"}, {role: "assistant", text: "answer"}, {role: "user", text: "last"});
  const h = harness();
  const second = await context.prepare({archive: original, state: first.state, wire, ...h, force: true});
  assert.ok(second.state.through > first.state.through);
  assert.ok(h.summaries[0].messages[1].content.includes(first.state.summary));
  assert.ok(!h.summaries[0].messages[1].content.includes("A".repeat(100)));
});
test("tool round boundary stays intact in active context", () => {
  const original = [{role: "user", text: "old"}, {role: "assistant", text: "old answer"},
    {role: "user", text: "tools question"}, {role: "assistant", text: "tools answer", tools: true}, {role: "user", text: "next"}];
  const toolWire = m => m.tools ? [{role: "assistant", content: "", tool_calls: [{id: "call1"}]}, {role: "tool", tool_call_id: "call1", content: "result"}, {role: "assistant", content: m.text}] : wire(m);
  const active = context.active(original, {summary: "old summary", through: 2}, toolWire);
  assert.deepEqual(active.slice(1).map(m => m.role), ["user", "assistant", "tool", "assistant", "user"]);
});
test("persist transaction retains full attachments; quota failure leaves caller state unchanged; restore uses archive", () => {
  const original = [{role: "user", text: "file", files: [{name: "notes", text: "original body"}], images: [{name: "pic", url: "data:image/png;base64,abc"}]}];
  let state = null;
  assert.throws(() => { state = context.commit(original, {summary: "new", through: 1}, () => false); }, /storage/);
  assert.equal(state, null);
  let persisted;
  state = context.commit(original, {summary: "new", through: 1}, value => { persisted = JSON.parse(JSON.stringify(value)); return true; });
  assert.deepEqual(persisted.messages, original);
  assert.equal(context.active(original, state, wire).length, 1);
  assert.equal(context.active(original, null, wire)[0].content, "file");
});
test("storage refusal prevents every summary request but does not block a short ordinary chat", async () => {
  const h = harness();
  const beforeCompact = () => { throw new Error("storage refused"); };
  await assert.rejects(context.prepare({archive: archive(), wire, ...h, beforeCompact}), /storage refused/);
  assert.equal(h.summaries.length, 0);
  const short = await context.prepare({archive: [{role: "user", text: "hello"}], wire, ...h, beforeCompact});
  assert.equal(short.changed, false);
});
test("native slack exact edge fits, one extra token fails; missing field defaults to eight", async () => {
  const options = {archive: [{role: "user", text: "hello"}], wire, auto: false,
    summarize: async () => { throw new Error("must not generate"); }};
  const count = async () => ({input_tokens: 888, max_context: 1024, effective_max_tokens: 128});
  assert.equal((await context.prepare({...options, count})).changed, false);
  await assert.rejects(context.prepare({...options, count: async () => ({...(await count()), input_tokens: 889})}), /context budget/);
  assert.equal(context.budget(await count()).slack, 8);
  assert.equal((await context.prepare({...options, count: async () => ({...(await count()), context_slack: 8})})).changed, false);
  await assert.rejects(context.prepare({...options, count: async () => ({...(await count()), context_slack: 9})}), /context budget/);
});
test("actual send keeps uncapped main request uncapped and preserves explicit high cap", async () => {
  const source = fs.readFileSync(require.resolve("../serve/web/app.js"), "utf8");
  const sendSource = source.slice(source.indexOf("async function send() {"), source.indexOf('$("composer").onsubmit'));
  assert.ok(sendSource.startsWith("async function send() {"));
  for (const cap of ["", "40000"]) {
    let outgoing;
    const elements = {input: {value: "synthetic prompt"}, chat: {lastElementChild: {}}};
    const noop = () => {};
    const sandbox = {AbortController, performance: {now: () => 1}, requestAnimationFrame: () => 0, cancelAnimationFrame: noop,
      $: id => elements[id], busy: null, historyBusy: false, legacyRouteResolved: true,
      messages: [], attachments: [], health: {model: "fixture"},
      settings: {thinking: "high", temperature: 0, max: cap, mcp: false}, mcpInfo: {tools: 0},
      renderAttachments: noop, autosize: noop, renderChat: noop, setBusy: noop, projectionLoaded: () => false,
      updateAssistant: noop, scrollDown: noop, saveChat: noop, headers: () => ({}),
      prepareChatContext: async () => ({reserve: 8192, replyLimit: cap ? +cap : null}),
      apiMessages: () => [{role: "user", content: "synthetic prompt"}],
      fetch: async (_path, options) => { outgoing = JSON.parse(options.body); return {ok: true, body: {getReader: () => ({read: async () => ({done: true})})}}; },
      TextDecoder, toast: (_kind, _title, error) => { throw new Error(error); }};
    vm.createContext(sandbox);
    vm.runInContext(sendSource, sandbox);
    await sandbox.send();
    if (cap) assert.equal(outgoing.max_tokens, 40000);
    else assert.equal(Object.hasOwn(outgoing, "max_tokens"), false);
    assert.equal(outgoing.reasoning_effort, "high");
  }
});
