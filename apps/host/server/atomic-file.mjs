/**
 * Durable writes for the small files the host reads again after a restart.
 *
 * A plain writeFileSync leaves a truncated file when the machine stops between
 * the truncate and the last byte. These helpers write a temporary file in the
 * same folder, flush it to disk, rename it over the target (an atomic replace
 * on Windows and POSIX), and keep the previous good copy as `<file>.bak`.
 *
 * Only small state, configuration and index files belong here. Model weights,
 * picture bytes and other user content keep their streaming/download paths.
 */
import {
  closeSync,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeSync,
} from 'node:fs';
import { dirname } from 'node:path';

const RETRYABLE = new Set(['EACCES', 'EBUSY', 'EPERM']);

function sleepMs(ms) {
  try {
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
  } catch {
    /* no SharedArrayBuffer: retry immediately */
  }
}

/** The temporary file used for one target. A crash leaves this name behind; the next save overwrites it. */
export function atomicTempPath(file) {
  return `${file}.tmp`;
}

/** The one backup copy kept beside a target. */
export function atomicBackupPath(file) {
  return `${file}.bak`;
}

function renameWithRetry(from, to) {
  for (let attempt = 0; ; attempt += 1) {
    try {
      renameSync(from, to);
      return;
    } catch (error) {
      // Windows antivirus and indexers hold a file for a moment; a rename that
      // can still work should not become a failed save.
      if (!RETRYABLE.has(error?.code) || attempt >= 5) throw error;
      sleepMs(15 * (attempt + 1));
    }
  }
}

function fsyncDirectory(dir) {
  // Windows has no directory fsync. POSIX needs it so the rename itself is durable.
  if (process.platform === 'win32') return;
  let fd;
  try {
    fd = openSync(dir, 'r');
    fsyncSync(fd);
  } catch {
    /* a filesystem that cannot fsync a directory still gets the atomic rename */
  } finally {
    if (fd !== undefined) {
      try { closeSync(fd); } catch { /* already closed */ }
    }
  }
}

function writeDurable(file, data, mode) {
  const temp = atomicTempPath(file);
  let fd;
  try {
    fd = openSync(temp, 'w', mode);
    const bytes = typeof data === 'string' ? Buffer.from(data, 'utf8') : data;
    let offset = 0;
    while (offset < bytes.length) {
      const written = writeSync(fd, bytes, offset, bytes.length - offset);
      if (!(written > 0)) throw new Error(`Could not write ${file}.`);
      offset += written;
    }
    fsyncSync(fd);
  } catch (error) {
    try { closeSync(fd); } catch { /* not open */ }
    fd = undefined;
    try { unlinkSync(temp); } catch { /* no temporary file to remove */ }
    throw error;
  } finally {
    if (fd !== undefined) {
      try { closeSync(fd); } catch { /* already closed */ }
    }
  }
  try {
    renameWithRetry(temp, file);
  } catch (error) {
    try { unlinkSync(temp); } catch { /* no temporary file to remove */ }
    throw error;
  }
  fsyncDirectory(dirname(file));
}

function jsonValid(buffer) {
  try {
    JSON.parse(String(buffer).replace(/^\uFEFF/, ''));
    return true;
  } catch {
    return false;
  }
}

/**
 * Write one small file durably.
 *
 * `validateCurrent` decides whether the file already on disk may become the
 * backup. JSON writers pass a parser, so a truncated file never replaces the
 * last good `.bak`. The backup is best effort: a read-only or full disk still
 * gets the new main file, and the caller sees the actual write error.
 */
export function atomicWriteFileSync(file, data, { mode = 0o600, backup = true, validateCurrent = null } = {}) {
  mkdirSync(dirname(file), { recursive: true });

  if (backup && existsSync(file)) {
    let current;
    try { current = readFileSync(file); } catch { current = null; }
    if (current && (!validateCurrent || validateCurrent(current))) {
      try { writeDurable(atomicBackupPath(file), current, mode); } catch { /* the main file still matters more */ }
    }
  }

  writeDurable(file, data, mode);

  // First save: an empty `.bak` would be useless; keep a copy of this good file.
  if (backup && !existsSync(atomicBackupPath(file))) {
    const bytes = typeof data === 'string' ? Buffer.from(data, 'utf8') : data;
    try { writeDurable(atomicBackupPath(file), bytes, mode); } catch { /* best effort */ }
  }
}

/** JSON with a validated backup: a broken current file is never copied over the good copy. */
export function atomicWriteJsonSync(file, value, options = {}) {
  const serialized = JSON.stringify(value, null, 2);
  if (serialized === undefined) throw new TypeError(`Cannot save ${file}: the value is not JSON.`);
  const text = `${serialized}\n`;
  return atomicWriteFileSync(file, text, { ...options, validateCurrent: jsonValid });
}

/** Read the file, or its last good backup when the file is missing or truncated. */
export function readJsonWithBackupSync(file, fallback = null) {
  for (const candidate of [file, atomicBackupPath(file)]) {
    try {
      return JSON.parse(readFileSync(candidate, 'utf8').replace(/^\uFEFF/, ''));
    } catch {
      /* try the backup, then the caller's fallback */
    }
  }
  return fallback;
}

/** Read the file, or its backup when the file is missing or cannot be read. Text has no parser, so content is returned as it is. */
export function readTextWithBackupSync(file, fallback = null) {
  for (const candidate of [file, atomicBackupPath(file)]) {
    try {
      return readFileSync(candidate, 'utf8').replace(/^\uFEFF/, '');
    } catch {
      /* try the backup, then the caller's fallback */
    }
  }
  return fallback;
}
