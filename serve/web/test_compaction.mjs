// node serve/web/test_compaction.mjs - the fold rules of buildApiMessages, which decide what reaches
// the model: the cut boundary, chained folds (including the legacy gap shape), undo-after-a-merge,
// the placeholder and error rules, and the corrupt-marker clamp. App regressions below also exercise
// cut selection, complete message retention and the context gauge with mocked I/O.
import {buildApiMessages} from "./compaction.js";

const cv = (m) => m.role === "user" ? [{role: "user", content: m.text}]
                                 : [{role: "assistant", content: m.text || ""}];   // #1392: a no-text turn still goes back
let fails = 0;
function check(name, got, want) {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g === w) console.log(`ok - ${name}`);
  else { console.error(`FAIL - ${name}\n  got  ${g}\n  want ${w}`); fails++; }
}
const SUM = (t, n) => `[A summary of the conversation so far, standing in for the ${n} earlier messages that are no longer included]\n\n${t}`;

// marker created after 4 messages (q1,a1,q2,a2): folded=4, payload = [SUM] + after the cut
const msgs = [
  {role: "user", text: "q1"}, {role: "assistant", text: "a1"},
  {role: "user", text: "q2"}, {role: "assistant", text: "a2"},
  {role: "assistant", compact: true, text: "SUM1", folded: 4},
  {role: "user", text: "q3"}, {role: "assistant", text: "a3"},
];
check("the cut is exclusive and the marker is never sent",
      buildApiMessages(msgs, null, cv).map((m) => m.content), [SUM("SUM1", 4), "q3", "a3"]);

// a chained summarization input (upto = through q3): the earlier summary stands in for [0..4),
// and q2's era turns BETWEEN an older fold and its marker are kept - here there are none to keep
check("a chained summary is built from the earlier summary plus everything after its cut",
      buildApiMessages(msgs, 6, cv).map((m) => m.content), [SUM("SUM1", 4), "q3"]);

// the regression this file exists for: turns between an older marker's cut and the marker itself
// (kept when that fold was made, never summarized) must survive into the next summary's input
const nested = [
  {role: "user", text: "q1"}, {role: "assistant", text: "a1"},
  {role: "assistant", compact: true, text: "SUM0", folded: 2},   // folded q1,a1; q2,a2 kept below it
  {role: "user", text: "q2"}, {role: "assistant", text: "a2"},
  {role: "user", text: "q3"},
];
check("turns between an older fold's cut and its marker are not lost",
      buildApiMessages(nested, null, cv).map((m) => m.content), [SUM("SUM0", 2), "q2", "a2", "q3"]);

// the KEEP branch: a chained input over a legacy gap-shape marker - the turns between its cut
// and the marker (kept when that fold was made, never summarized) must actually be KEPT
check("a chained input keeps the turns between an older marker's cut and the marker",
      buildApiMessages(nested, 5, cv).map((m) => m.content), [SUM("SUM0", 2), "q2", "a2"]);

// an orphaned marker (above the last fold's boundary, a drifted or legacy state): its summary is
// emitted rather than its content vanishing - redundant context beats silent loss
const orphaned = [
  {role: "user", text: "q0"}, {role: "assistant", text: "a0"}, {role: "user", text: "q1"},
  {role: "assistant", compact: true, text: "ORPHAN", folded: 3},
  {role: "user", text: "q2"}, {role: "assistant", text: "a2"},
  {role: "user", text: "q3"},
  {role: "assistant", compact: true, text: "LAST", folded: 2},     // folds [0..2): the ORPHAN at 3 is above it
  {role: "user", text: "q4"},
];
check("an orphaned marker above the last fold keeps its summary in the payload",
      buildApiMessages(orphaned, null, cv).map((m) => m.content),
      [SUM("LAST", 2), "q1", SUM("ORPHAN", 3), "q2", "a2", "q3", "q4"]);

// undo after a merge: [fold, m1, kept, m2, tail]; peeling m2 falls back to m1's fold
const merged = [
  {role: "user", text: "q1"}, {role: "assistant", text: "a1"},
  {role: "assistant", compact: true, text: "SUM1", folded: 2},
  {role: "user", text: "q2"}, {role: "assistant", text: "a2"},
  {role: "assistant", compact: true, text: "SUM2", folded: 5},
  {role: "user", text: "q3"},
];
check("peeling the newest marker restores the prior fold's payload",
      buildApiMessages(merged.slice(0, 5), null, cv).map((m) => m.content), [SUM("SUM1", 2), "q2", "a2"]);

// no marker: the plain conversation, and the placeholder is never sent
check("no fold: full conversation, placeholder skipped",
      buildApiMessages([{role: "user", text: "q"}, {role: "assistant", compacting: true}, {role: "user", text: "r"}], null, cv)
        .map((m) => m.content), ["q", "r"]);

// a corrupt marker (folded beyond its own index) must not silently skip real messages
const corrupt = [
  {role: "user", text: "q1"},
  {role: "assistant", compact: true, text: "SUM", folded: 9},   // corrupt: claims 9 folded, only 1 exists
  {role: "user", text: "q2"}, {role: "assistant", text: "a2"},  // after the marker: must still be sent
];
check("a corrupt folded-beyond-index marker cannot drop the messages after it",
      buildApiMessages(corrupt, null, cv).map((m) => m.content), [SUM("SUM", 9), "q2", "a2"]);

// the benign case: a single marker IS the last - its summary prefaces, nothing orphaned
const lone = [
  {role: "user", text: "q1"}, {role: "assistant", text: "a1"}, {role: "user", text: "q2"},
  {role: "assistant", compact: true, text: "ONLY", folded: 3},
];
check("a lone marker: pre carries it, no orphan emission",
      buildApiMessages(lone, null, cv).map((m) => m.content), [SUM("ONLY", 3)]);

// an erroring assistant turn reaches the converter and is sent per #1392 (empty content), not
// dropped by the fold rules - history keeps alternating
check("an error turn is the converter's call, not the fold's (#1392)",
      buildApiMessages([{role: "user", text: "q"}, {role: "assistant", text: "", error: "x"}], null, cv)
        .map((m) => m.content), ["q", ""]);


// Exercise the app's compaction orchestration as well as the pure fold rules. The real message
// converters run here; only the DOM, storage and model response are replaced.
const {readFileSync} = await import("node:fs");
const vm = await import("node:vm");
const app = readFileSync(new URL("./app.js", import.meta.url), "utf8");
const section = (start, end) => app.slice(app.indexOf(start), app.indexOf(end, app.indexOf(start)));
function appHarness(initial) {
  const requests = [];
  const fill = {style: {}}, label = {};
  const classes = new Set();
  const pill = {hidden: true, classList: {
    remove(...names) { names.forEach((name) => classes.delete(name)); },
    toggle(name, on) { if (on) classes.add(name); else classes.delete(name); },
  }, querySelector(selector) { return selector === ".st-ctx__fill" ? fill : label; }};
  const context = vm.createContext({
    messages: initial, busy: null, panel: null, pending: null, health: {model: "test", max_context: 32768},
    COMPACT_KEEP_TOKENS: 5000, COMPACT_PROMPT: "Summarize", AbortController,
    historyLocked: () => !!context.busy || !!context.panel || !!context.pending,
    buildApiMessages, headers: () => ({}), safeToolMedia: () => true,
    $: (id) => id === "ctx-pill" ? pill : {lastElementChild: null},
    setBusy() {}, setPill() {}, renderChat() {}, saveChat() {}, settlePending() {},
    toast(kind, title, message) { if (kind === "error") throw new Error(message); },
    fmt: String, kfmt: String, ctxfmt: String,
    fetch: async (url, options) => {
      requests.push(JSON.parse(options.body));
      return {ok: true, json: async () => ({choices: [{message: {content: "PREFIX SUMMARY"}, finish_reason: "stop"}],
        usage: {prompt_tokens: 100, completion_tokens: 10}})};
    },
  });
  vm.runInContext([
    section("function updateContextPill()", "let settings ="),
    section("async function compactChat()", "function timeStr(t)"),
    section("function apiMessages(", "// An answer that used MCP tools"),
    section("function assistantMessages(", "function busyHint()"),
    section("function fileBlock(", "function renderAttachments()"),
  ].join("\n"), context);
  return {requests, fill, label, pill, classes,
    run: (code) => vm.runInContext(code, context)};
}

const retained = [
  {role: "user", text: "old question"}, {role: "assistant", text: "old answer"},
  {role: "user", text: "inspect", files: [{name: "input.txt", text: "EXACT FILE CONTENT"}],
    images: [{url: "data:image/png;base64,TEST"}]},
  {role: "assistant", text: "before after", tools: [{round: 0, at: 6, id: "call1", name: "inspect",
    arguments: {path: "input.txt"}, result: "EXACT TOOL RESULT", state: "done"}]},
];
const keep = appHarness(retained);
const expectedTail = keep.run("apiMessages().slice(2)");
await keep.run("compactChat()");
check("compaction preserves complete tail API messages, including files, images and tools",
      keep.run("apiMessages().slice(1)"), expectedTail);

const chain = appHarness([
  {role: "user", text: "ORIGINAL ".repeat(15000)}, {role: "assistant", text: "old answer"},
  {role: "user", text: "q2"}, {role: "assistant", text: "a2"},
  // A short recent tail must not let the backward scan cross this marker into old originals.
  {role: "assistant", compact: true, text: "PREVIOUS SUMMARY", folded: 2},
  {role: "user", text: "q3"}, {role: "assistant", text: "a3"},
]);
await chain.run("compactChat()");
check("second compaction uses the prior summary instead of the folded original",
      [chain.requests.length, JSON.stringify(chain.requests).includes("PREVIOUS SUMMARY"),
       JSON.stringify(chain.requests).includes("ORIGINAL")], [1, true, false]);
const count = chain.requests.length;
await chain.run("compactChat()");
check("an already compacted prefix is not summarized again without new turns", chain.requests.length, count);

const large = appHarness([
  {role: "user", text: "old question"}, {role: "assistant", text: "old answer"},
  {role: "user", text: "inspect", files: [{name: "large.txt", text: "LARGE FILE ".repeat(5000)}]},
  {role: "assistant", text: "done"},
]);
await large.run("compactChat()");
check("attachment contents count toward the retained-tail budget",
      JSON.stringify(large.requests).includes("LARGE FILE"), true);

const gauge = appHarness([{ctx: 24000}]);
gauge.run("updateContextPill()");
gauge.run("messages = []; updateContextPill()");
check("new chat clears the previous gauge reading and warning",
      [gauge.fill.style.width, gauge.label.textContent, gauge.pill.title, [...gauge.classes]],
      ["0%", "\u2013/32768", "Context usage is not measured yet", []]);

process.exit(fails ? 1 : 0);
