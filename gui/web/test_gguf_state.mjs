/* gui/web/test_gguf_state.mjs - node:test for gui/web/gguf_state.js (no DOM, no deps).
 *
 * The exact behaviors the Custom GGUF field depends on: the text field is the source of
 * truth, a stale detection is invalidated the moment it changes, a late async response is
 * ignored, and Prepare refuses any detection whose directory is not exactly the field.
 *
 *     node gui/web/test_gguf_state.mjs
 * (gui/test_manager.py runs this when node is installed; the unittest suite skips without it)
 */
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { canon, pathsEqual, GgufDetection } = require("./gguf_state.js");

const ABLITERATED = "E:\\Model\\Q2_0-Abliterated";
const ORIGINAL = "E:\\Model\\Q2_0";

// ---------------------------------------------------------------- path comparison
test("pathsEqual: same folder in different spellings is the same folder", () => {
  assert.equal(pathsEqual(ABLITERATED, "E:/Model/Q2_0-Abliterated"), true);   // slash direction
  assert.equal(pathsEqual(ABLITERATED, "E:\\Model\\Q2_0-Abliterated\\"), true); // trailing slash
  assert.equal(pathsEqual(ABLITERATED, "  " + ABLITERATED), true);              // surrounding space
  if (process.platform === "win32")
    assert.equal(pathsEqual(ABLITERATED, "e:\\model\\q2_0-abliterated"), true); // case
});

test("pathsEqual: a different folder is never equal", () => {
  assert.equal(pathsEqual(ABLITERATED, ORIGINAL), false);
  assert.equal(pathsEqual(ABLITERATED, "E:\\Model\\Q2_0-Abliterated.old"), false);
  assert.equal(pathsEqual(ABLITERATED, ""), false);
});

// ---------------------------------------------------------------- detection lifecycle
test("a detection is committed only for the path the field shows", () => {
  const s = new GgufDetection();
  assert.equal(s.prepareAllowed(ABLITERATED), false);            // nothing detected yet

  const t1 = s.invalidate();                                     // detect A…
  const detA = { ok: true, dir: ABLITERATED, shards: ["a-00001-of-00002.gguf"], tag: "q2_0" };
  assert.equal(s.accept(ABLITERATED, detA, t1), true);
  assert.equal(s.prepareAllowed(ABLITERATED), true);             // exact match -> enabled
  assert.equal(s.prepareAllowed(ORIGINAL), false);               // any other path refused
});

test("changing the field invalidates the previous detection immediately", () => {
  const s = new GgufDetection();
  const t1 = s.invalidate();
  const detA = { ok: true, dir: ABLITERATED, shards: ["a-00001-of-00002.gguf"], tag: "q2_0" };
  s.accept(ABLITERATED, detA, t1);
  assert.equal(s.prepareAllowed(ABLITERATED), true);

  s.invalidate();                                                // the user types / pastes another path
  assert.equal(s.prepareAllowed(ABLITERATED), false);            // old detection is dead NOW
  assert.equal(s.detect, null);
});

test("a late async response (for the previous value) is ignored", () => {
  const s = new GgufDetection();
  const t1 = s.invalidate();                                     // detect A (in flight)
  const t2 = s.invalidate();                                     // field changed to B before A returned
  const detA = { ok: true, dir: ABLITERATED, shards: ["a"], tag: "q2_0" };
  const detB = { ok: true, dir: ORIGINAL, shards: ["o"], tag: "q2_0" };
  assert.equal(s.accept(ABLITERATED, detA, t1), false);          // A's response: stale token -> dropped
  assert.equal(s.accept(ORIGINAL, detB, t2), true);              // B's response lands
  assert.equal(s.detect, detB);
  assert.equal(s.prepareAllowed(ORIGINAL), true);
  assert.equal(s.prepareAllowed(ABLITERATED), false);
});

test("a response whose dir does not match the field is refused", () => {
  const s = new GgufDetection();
  const t = s.invalidate();                                      // field = ORIGINAL
  // (the old bug: detection B for another folder could be committed while the field shows A)
  const detA = { ok: true, dir: ABLITERATED, shards: ["a"], tag: "q2_0" };
  assert.equal(s.accept(ORIGINAL, detA, t), false);              // dir ≠ field -> not committed
  assert.equal(s.prepareAllowed(ORIGINAL), false);
});

test("an error result is shown for the current value but never enables Prepare", () => {
  const s = new GgufDetection();
  const t = s.invalidate();
  assert.equal(s.accept(ABLITERATED, { ok: false, error: "no multi-shard GGUF files" }, t), true);
  assert.equal(s.prepareAllowed(ABLITERATED), false);
});

test("prepare always refuses the stale A-visible/B-detected state (the reported bug)", () => {
  // The report: the UI showed E:\Model\Q2_0-Abliterated but the internal detection referred to
  // E:\Model\Q2_0, so Prepare returned "already installed as strata-q2_0.json".  With the field
  // as the source of truth this combination can never be prepared:
  const s = new GgufDetection();
  const t = s.invalidate();
  assert.equal(s.accept(ABLITERATED, { ok: true, dir: ORIGINAL, shards: ["o"], tag: "q2_0" }, t), false);
  assert.equal(s.prepareAllowed(ABLITERATED), false);            // refused, never "already" surfaced
});

// ---------------------------------------------------------------- browse == manual input
test("browse and manual input reach the same committed state", () => {
  // "Use this folder" writes the path into the field and runs the SAME detect-on-change path
  // as typing: both go through invalidate(detect) -> accept(value) -> prepareAllowed(value).
  const byBrowse = new GgufDetection();
  const byTyping = new GgufDetection();
  for (const s of [byBrowse, byTyping]) {
    const t = s.invalidate();
    s.accept(ABLITERATED, { ok: true, dir: ABLITERATED, shards: ["a"], tag: "q2_0" }, t);
  }
  assert.equal(byBrowse.prepareAllowed(ABLITERATED), byTyping.prepareAllowed(ABLITERATED));
  assert.equal(byBrowse.detect.tag, byTyping.detect.tag);        // same committed detection
});