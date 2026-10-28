/**
 * Bounded log files. A long run must not fill the disk with one growing log:
 * above the limit the file becomes `<file>.1` and only that one old copy is
 * kept. New writes continue in a fresh file.
 */
import { appendFileSync, existsSync, mkdirSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';

export const LOG_LIMIT_BYTES = 64 * 1024 * 1024;

function renameOver(from, to) {
  try {
    renameSync(from, to);
    return true;
  } catch {
    // Windows cannot replace a file another program holds; drop the old copy first.
    try { rmSync(to, { force: true }); } catch { /* best effort */ }
    try { renameSync(from, to); return true; } catch { return false; }
  }
}

/** Rotate one file when it is larger than `limit`. Returns true when it moved. */
export function rotateLogIfNeeded(file, { limit = LOG_LIMIT_BYTES } = {}) {
  try {
    if (!existsSync(file)) return false;
    if (statSync(file).size <= limit) return false;
    return renameOver(file, `${file}.1`);
  } catch {
    return false;
  }
}

/** Append one line (or block), rotating first. A single block larger than the limit keeps its tail. */
export function appendLog(file, text, { limit = LOG_LIMIT_BYTES, mode = 0o600 } = {}) {
  mkdirSync(dirname(file), { recursive: true });
  let bytes = Buffer.from(String(text ?? ''), 'utf8');
  if (bytes.length > limit) bytes = bytes.subarray(bytes.length - limit);
  let existing = 0;
  try { existing = existsSync(file) ? statSync(file).size : 0; } catch { existing = 0; }
  // Keep every kept file at or below the limit: rotate before the line that would cross it.
  if (existing > 0 && existing + bytes.length > limit) renameOver(file, `${file}.1`);
  appendFileSync(file, bytes, { mode });
}

/**
 * Write a whole log file (the harness rewrites its log when a run ends).
 * Old content is rotated away first; oversized new text keeps its tail.
 */
export function writeLogFile(file, text, { limit = LOG_LIMIT_BYTES, mode = 0o600 } = {}) {
  mkdirSync(dirname(file), { recursive: true });
  rotateLogIfNeeded(file, { limit });
  let bytes = Buffer.from(String(text ?? ''), 'utf8');
  if (bytes.length > limit) bytes = bytes.subarray(bytes.length - limit);
  writeFileSync(file, bytes, { mode });
}

/** Rotate the matching log files in a folder. Used at host start, before engines write. */
export function rotateLogsIn(dir, { limit = LOG_LIMIT_BYTES, suffixes = ['.log'], olderThanMs = 0 } = {}) {
  const rotated = [];
  let names;
  try { names = readdirSync(dir); } catch { return rotated; }
  for (const name of names) {
    if (!suffixes.some(suffix => name.endsWith(suffix))) continue;
    if (name.endsWith('.1')) continue;
    const file = join(dir, name);
    if (olderThanMs > 0) {
      try {
        if (Date.now() - statSync(file).mtimeMs < olderThanMs) continue;
      } catch {
        continue;
      }
    }
    if (rotateLogIfNeeded(file, { limit })) rotated.push(basename(file));
  }
  return rotated;
}
