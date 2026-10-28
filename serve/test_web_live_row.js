// The live request row, verbatim out of app.js, run in Node (serve/test_web_live_row.py runs this too).
// A synthetic row with streaming counters while the engine works; every unreported field keeps the dash.
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const src = fs.readFileSync(path.join(__dirname, "web", "app.js"), "utf8");
const oneLine = (needle) => {
  const hits = src.split("\n").filter((l) => l.includes(needle));
  assert.equal(hits.length, 1, `exactly one line in app.js must carry: ${needle}`);
  return hits[0];
};

const fnMatch = src.match(/function liveRequestRow\(live, fsm, serverTime\) \{[\s\S]*?\n\}/);
assert(fnMatch, "liveRequestRow must exist in app.js");
const fmtLine = oneLine("const fmt = (n, d = 0)");
const liveRequestRow = new Function([oneLine('const esc = (s) => String(s)'), fmtLine, fnMatch[0], "return liveRequestRow;"].join("\n"))();
const fmt = new Function(fmtLine + ";return fmt;")();   // expectations quote the page's own formatting, not locale literals invented here

const cells = (row) => (row.match(/<td/g) || []).length;
const nums = (row) => row.match(/<td class="num">([^<]*)<\/td>/g).map((c) => c.replace(/<[^>]*>/g, ""));

// ---------------------------------------------------------------- reading phase
const reading = liveRequestRow(
  {prompt_tokens: 48518, reused: 0, prompt_read: 1200, elapsed_s: 1.2, prefill_tok_s_mean: null, tok_s: null, phase: null},
  {isReading: true, isGenerating: false}, 1700000000);
assert.equal(cells(reading), 9, "the live row carries the same nine columns as a history row");
assert.ok(reading.includes('class="st-badge st-badge--reading"'), "reading uses the page's own reading badge");
assert.ok(reading.includes("Reading…"));
// nothing the request has reported yet: reused 0 is not shown, prefill and decode are dashes while reading
assert.deepEqual(nums(reading), ["48,518", "–", "–", "–", "–", "–", "1.2 s"], "a live read reports only its prompt size and duration");

// ---------------------------------------------------------------- generating phase, a real RESUME happened
const generating = liveRequestRow(
  {prompt_tokens: 48518, reused: 12000, generated: 340, elapsed_s: 5.0, prefill_tok_s_mean: 900, tok_s: 21.37, phase: "thinking"},
  {isReading: false, isGenerating: true}, 1700000000);
assert.ok(generating.includes('class="st-badge st-badge--generating"'), "generating uses the generating badge");
assert.ok(generating.includes("Thinking…"), "the engine phase maps to one short word");
// prefill holds from the first PP line; decode streams; the expert hit rate is not a live fact — always the dash
assert.deepEqual(nums(generating), ["48,518", "12,000", "340", "900", "21.4", "–", "5.0 s"], "live streaming counters, no fake zero");

// ---------------------------------------------------------------- phase words
const wordOf = (phase) => liveRequestRow(
  {prompt_tokens: 10, generated: 1, elapsed_s: 0.1, tok_s: 10, phase},
  {isReading: false, isGenerating: true}, 1700000000).match(/st-badge st-badge--generating">([^<]+)/)[1];
assert.equal(wordOf("answering"), "Answering…");
assert.equal(wordOf("writing a tool call tail"), "Writing…");
assert.equal(wordOf("tool call complete: fetch"), "Running…");
assert.equal(wordOf("something new"), "Generating…", "an unknown phase still names the honest fallback");

// ---------------------------------------------------------------- timing
// no server clock: the row still builds and keeps a dash for the missing duration
const noClock = liveRequestRow({prompt_tokens: 10, generated: 1, tok_s: 10}, {isReading: false, isGenerating: true});
assert.ok(noClock.includes('class="num">–</td>'), "elapsed_s missing: duration is a dash");
assert.ok(!/NaN|undefined/.test(noClock), "no NaN or undefined reaches the page");
// the same server sample renders the same row — the clock is the server's, not browser memory
const stamp = (row) => row.match(/<td>([^<]+)<\/td>/)[1];
assert.equal(stamp(liveRequestRow({prompt_tokens: 1, generated: 1, elapsed_s: 2, tok_s: 1}, {isReading: false, isGenerating: true}, 1700000000)),
           stamp(liveRequestRow({prompt_tokens: 1, generated: 1, elapsed_s: 2, tok_s: 1}, {isReading: false, isGenerating: true}, 1700000000)));

console.log("live request row: reading + generating cells, phase words, dash rules, deterministic clock — green");
