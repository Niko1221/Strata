/* gui/web/gguf_state.js - the pure state machine behind the Custom GGUF field (no DOM).
 *
 * The path text in the field is the SOURCE OF TRUTH: a detection is only ever usable when
 * it was made for the path the field shows right now, and its response arrived before the
 * field changed again.  app.js wires this to the input/browse/prepare events; the node
 * tests (gui/web/test_gguf_state.mjs) drive it directly - the exact behaviors behind
 * "stale detection invalidated on text change", "late async response ignored" and
 * "Prepare refuses a stale detection".
 *
 * UMD: browser attaches window.GgufState (index.html loads this before app.js), node
 * requires it.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.GgufState = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const WIN = (typeof process !== "undefined" && process.platform === "win32") ||
              (typeof navigator !== "undefined" && /win/i.test(navigator.platform || ""));

  // One canonical spelling for path comparison: trimmed, one separator direction, no trailing
  // separator; on Windows the comparison ignores case (the filesystem does).
  function canon(p) {
    if (typeof p !== "string") return "";
    let s = p.trim();
    if (!s) return "";
    s = s.replace(/[\\/]+/g, "/").replace(/\/+$/, "");
    return WIN ? s.toLowerCase() : s;
  }

  // Exact-path comparison: two values name the same folder only when their canonical spellings
  // are identical.  A trailing slash, a different slash direction or (on Windows) different
  // letter case do NOT make them different folders; a different directory always does.
  function pathsEqual(a, b) {
    return !!a && !!b && canon(a) === canon(b);
  }

  // The committed detection state for the field, with a generation token.
  //   token      - bumped by every invalidate(): a response started earlier carries the old
  //                token and is ignored (the "late async response" race)
  //   detect     - the last accepted /api/gguf-detect result (ok or error), committed only
  //                when it was made for the CURRENT field value
  function GgufDetection() {
    this.token = 0;
    this.detect = null;
    this.input = "";
  }

  // The field changed (typed, pasted, autofilled, "Use this folder"): the previous detection
  // is dead the moment the text differs - return the token this generation runs under.
  GgufDetection.prototype.invalidate = function () {
    this.token += 1;
    this.detect = null;
    return this.token;
  };

  // A detection response for `inputPath` arrived.  Accept it ONLY when it is the response of
  // the current generation (token matches) and, on success, the detected directory is exactly
  // the current input path (pathsEqual).  An error object is accepted for the current input
  // so the UI can show it and keep Prepare disabled.  Returns whether the state was updated.
  GgufDetection.prototype.accept = function (inputPath, det, token) {
    if (token !== this.token) return false;                        // a response for an older value
    if (!det || det.ok === false) {
      this.detect = det || null;
      this.input = canon(inputPath);
      return true;
    }
    if (!pathsEqual(inputPath, det.dir)) return false;             // not the field's folder
    this.detect = det;
    this.input = canon(inputPath);
    return true;
  };

  // Prepare may run only with a DETECTED folder that is EXACTLY the path the field shows now.
  // Visible path A + internal detection B (the stale-state bug) is refused here, always.
  GgufDetection.prototype.prepareAllowed = function (inputPath) {
    return !!this.detect && this.detect.ok === true && pathsEqual(inputPath, this.detect.dir);
  };

  return { canon: canon, pathsEqual: pathsEqual, GgufDetection: GgufDetection };
});