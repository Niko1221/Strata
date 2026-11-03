/**
 * Strata Dev (docs/DEV_BUILD.md): one log buffer per source for the Dev panel's Logs tab and /dev-debug.
 *
 * Sources: `host` (this process: console output once capture is on), `harness` (the agent's stdout/stderr, pushed by
 * agent/harness-runtime.mjs), `ui` (errors the page reports), `hooks` (the backend hook system). The engine's lines are not
 * copied here: they already sit in supervisor.mjs and native-control's buffers and are read from there when asked.
 *
 * Off by default, and then `push` does nothing: Live never fills it. index.mjs turns it on when STRATA_DEV=1.
 */
const LIMIT = 2000;
const buffers = new Map();
let enabled = false;
let verbose = false;
let seq = 0;
let lastError = null;

export function enableDevLog(options = {}) { enabled = true; verbose = options.verbose === true; }
export function setVerbose(value) { verbose = value === true; }
export const devLogEnabled = () => enabled;
export const devLogVerbose = () => enabled && verbose;

/** One or more lines from `source`. A chunk with several lines becomes several entries; empty lines are dropped. */
export function push(source, text, level = 'info') {
  if (!enabled) return;
  const lines = String(text ?? '').replace(/\r\n?/g, '\n').split('\n').filter(line => line.trim());
  if (!lines.length) return;
  let list = buffers.get(source);
  if (!list) buffers.set(source, list = []);
  const at = new Date().toISOString();
  for (const [index, line] of lines.entries()) {
    const entry = { seq: ++seq, at, source, level, line: line.length > 4000 ? `${line.slice(0, 4000)}…` : line };
    list.push(entry);
    // The first line of a chunk says what went wrong; the rest is usually its stack.
    if (level === 'error' && index === 0) lastError = { ...entry, detail: lines.slice(1, 40).join('\n') };
  }
  if (list.length > LIMIT) list.splice(0, list.length - LIMIT);
}

/** Entries of the named sources (all when none), oldest first, after `since` (a seq number), at most `limit`. */
export function entries({ sources = null, since = 0, limit = 500 } = {}) {
  const wanted = sources?.length ? sources : [...buffers.keys()];
  const all = wanted.flatMap(name => buffers.get(name) ?? []).filter(e => e.seq > since).sort((a, b) => a.seq - b.seq);
  return all.slice(-limit);
}

export const lastErrorEntry = () => lastError;
export const sourceNames = () => [...buffers.keys()];

/** Copies everything this process writes through `console` into the `host` buffer; the console output itself is unchanged. */
let captured = false;
export function captureConsole(target = console) {
  if (captured) return;
  captured = true;
  for (const [method, level] of [['log', 'info'], ['info', 'info'], ['warn', 'warn'], ['error', 'error'], ['debug', 'debug']]) {
    const original = target[method].bind(target);
    target[method] = (...args) => {
      original(...args);
      try { push('host', args.map(a => (a instanceof Error ? a.stack ?? a.message : typeof a === 'string' ? a : safeJson(a))).join(' '), level); } catch { /* logging never throws */ }
    };
  }
}

function safeJson(value) {
  try { return JSON.stringify(value); } catch { return String(value); }
}

/** For tests: forget everything and turn the buffer off again. */
export function resetDevLog() { buffers.clear(); enabled = false; verbose = false; seq = 0; lastError = null; }
