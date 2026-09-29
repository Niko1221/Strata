// node serve/web/test_fold.mjs - the fold rules of buildApiMessages, which decide what reaches
// the model: the cut boundary, chained folds (including the legacy gap shape), undo-after-a-merge,
// the placeholder and error rules, and the corrupt-marker clamp. Pure, so it runs without the app.
import {buildApiMessages} from "./fold.js";

const cv = (m) => m.role === "user" ? [{role: "user", content: m.text}]
                                 : m.text ? [{role: "assistant", content: m.text}] : [];
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
const orphan = [
  {role: "user", text: "q0"},
  {role: "assistant", compact: true, text: "ORPHAN", folded: 1},   // sits above the last fold's cut
  {role: "user", text: "q2"}, {role: "assistant", text: "a2"},
  {role: "user", text: "q3"},
];
// (no later marker here: the orphan IS the last, so pre covers it - the dangerous shape needs two)
const orphaned = [
  {role: "user", text: "q0"}, {role: "assistant", text: "a0"}, {role: "user", text: "q1"},
  {role: "assistant", compact: true, text: "ORPHAN", folded: 3},   // ABOVE the last fold's cut of 3? no:
  {role: "user", text: "q2"}, {role: "assistant", text: "a2"},     // the cut must fall BELOW this marker
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

// an erroring assistant turn is not sent (pre-existing rule the fold must preserve)
check("an error turn is dropped",
      buildApiMessages([{role: "user", text: "q"}, {role: "assistant", text: "", error: "x"}], null, cv)
        .map((m) => m.content), ["q"]);

process.exit(fails ? 1 : 0);
