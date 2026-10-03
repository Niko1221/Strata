// compaction.js - the conversation-to-payload rules, pure so they can be tested without the app
// (node serve/web/test_compaction.mjs). `convert(m)` turns one message into its API messages (the app's
// version keeps images, files and tool rounds); this module only decides WHICH messages are sent.
// The rules: in the live payload (upto unset) the LAST compact marker's summary stands in for
// everything older than its recorded cut, and markers themselves are never sent; a summarization
// input (upto set) folds at every marker it passes - keeping the turns BETWEEN that marker's cut
// and the marker itself (kept-then, never summarized) - so a chained summary loses nothing.
function buildApiMessages(messages, upto, convert) {
  const summaryFor = (m) => ({role: "user", content:
    `[A summary of the conversation so far, standing in for the ${m.folded} earlier messages that are no longer included]\n\n${m.text}`});
  const end = upto == null ? messages.length : upto;
  const out = [];
  let start = 0, pre = null, last = -1;
  if (upto == null) {
    last = messages.map((m) => !!m.compact).lastIndexOf(true);
    // a fold cannot extend beyond its own marker (a corrupt stored `folded` would otherwise skip
    // real messages silently); min() makes that state harmless instead of lossy
    if (last >= 0) { pre = summaryFor(messages[last]); start = Math.max(0, Math.min(messages[last].folded, last)); }
  }
  for (let i = start; i < end; i++) {
    const m = messages[i];
    if (m.compact) {
      // a marker ABOVE the last fold's boundary is an orphan (legacy or drifted state): its range is
      // claimed by nobody, so emit its summary rather than drop its content silently
      if (upto == null && i !== last && i > start) out.push({src: i, msg: summaryFor(m)});
      if (upto != null) {
        // the summary covers [0..m.folded) only: drop those contributions, KEEP [folded..i),
        // and the summary stands in front of what it replaces
        for (let j = out.length - 1; j >= 0; j--) if (out[j].src < m.folded) out.splice(j, 1);
        out.unshift({src: i, msg: summaryFor(m)});
      }
      continue;
    }
    if (m.compacting) continue;
    if (!m.error) for (const msg of convert(m)) out.push({src: i, msg});
  }
  const msgs = out.map((e) => e.msg);
  return pre ? [pre, ...msgs] : msgs;
}
if (typeof module !== "undefined") module.exports = {buildApiMessages};
