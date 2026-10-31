// The phase machinery, verbatim out of app.js, run in Node (serve/test_web_kv_phase.py runs this too).
// 1. the kvLamp line over all 24 phase combinations — the lamp is one deterministic expression;
// 2. ksplit/ksplitKs: the legend column and the centre agree by construction, the residual goes on the biggest row;
// 3. pctFloor: an honest tenth, never a rounded-up false 100.
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const src = fs.readFileSync(path.join(__dirname, "web", "app.js"), "utf8");
const oneLine = (needle) => {
  const hits = src.split("\n").filter((l) => l.includes(needle));
  assert.equal(hits.length, 1, `exactly one line in app.js must carry: ${needle}`);
  assert.equal(hits[0].endsWith(";"), true, `the piece must be one complete statement: ${needle}`);
  return hits[0];
};

// ---------------------------------------------------------------- 1. the lamp truth table
const lampLine = oneLine("const kvLamp = (fsm, hasReuse, readingStarted, resumeFlash) =>");
const kvLamp = new Function("return (" + lampLine.slice("const kvLamp = ".length, -1) + ")")();

// the ruling in plain words, written independently of the expression:
// "a RESUME flash wins, then Reading, then Generating, else Free"
const ruling = (state, hasReuse, readingStarted, resumeFlash) => {
  if (resumeFlash) return "reuse";
  if (state === "reading") return hasReuse && !readingStarted ? "reuse" : "prefill";
  if (state === "generating") return "decode";
  return "free";
};
const phases = {
  reading:    {isReading: true,  isGenerating: false},
  generating: {isReading: false, isGenerating: true},
  quiet:      {isReading: false, isGenerating: false},
};
let checked = 0;
for (const [state, fsm] of Object.entries(phases))
  for (const hasReuse of [false, true])
    for (const readingStarted of [false, true])
      for (const resumeFlash of [false, true]) {
        const got = kvLamp(fsm, hasReuse, readingStarted, resumeFlash);
        assert.equal(got, ruling(state, hasReuse, readingStarted, resumeFlash),
                     `${state} hasReuse=${hasReuse} readingStarted=${readingStarted} resumeFlash=${resumeFlash} -> ${got}`);
        checked++;
      }
assert.equal(checked, 24, "3 phases × hasReuse × readingStarted × resumeFlash");

// ---------------------------------------------------------------- 2. the token math
const math = new Function([
  oneLine("const fmt = (n, d = 0)"),
  oneLine("const kfmt = (n)"),
  oneLine("const ksplitKs = (parts)"),
  oneLine("const ksplit = (parts)"),
  oneLine("const pctFloor = (x)"),
  "return {kfmt, ksplitKs, ksplit, pctFloor};",
].join("\n"))();

// the documented example: 48 518 + 82 554 — rows rounded on their own would read "49k" + "83k" beside a "131k" centre
assert.deepEqual(math.ksplitKs([48518, 82554, 0, 0]), [49, 82, 0, 0], "the residual goes on the biggest row");
assert.deepEqual(math.ksplit([48518, 82554, 0, 0]), ["49k", "82k", "0", "0"], "the tok column");
assert.equal(math.kfmt((49 + 82) * 1000), "131k", "the centre quotes the exact sum of the rows above it");
assert.equal(math.ksplitKs([9500, 82554, 0, 0]), null, "a non-zero row under 10k: no integer split");
assert.deepEqual(math.ksplit([9500, 82554, 0, 0]), ["9.5k", "83k", "0", "0"], "below the threshold both sides fall back to plain kfmt");
assert.deepEqual(math.ksplit([0, 0, 0, 0]), ["0", "0", "0", "0"], "an empty ring prints zeros, never –");

// ---------------------------------------------------------------- 3. honest percentages
assert.equal(math.pctFloor(48518 / 99000), "49.0%", "a floored tenth, one decimal");
assert.equal(math.pctFloor(0.99977), "99.9%", "99.977 never becomes a false 100");
assert.equal(math.pctFloor(0.1), "10.0%", "a tenth keeps its decimal");

console.log("kv phase machinery: 24 lamp combinations + the token split + floored percents — green");
