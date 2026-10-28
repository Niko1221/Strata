/**
 * Refuse a long download or install before it starts when the target disk is
 * nearly full. A failed write after 30 GB is worse than one plain sentence now.
 */
import { existsSync, statfsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

export const DEFAULT_MIN_FREE_BYTES = 2 * 1024 ** 3; // 2 GiB
const GIB = 1024 ** 3;

function nearestExisting(path) {
  let at = resolve(String(path || '.'));
  for (let depth = 0; depth < 64; depth += 1) {
    if (existsSync(at)) return at;
    const up = dirname(at);
    if (up === at) return at;
    at = up;
  }
  return at;
}

/** Free bytes at a path, or null when the filesystem cannot answer. Null never blocks a download. */
export function freeBytesAt(path, { statfs = statfsSync } = {}) {
  try {
    const s = statfs(nearestExisting(path));
    const free = Number(s.bavail) * Number(s.bsize);
    return Number.isFinite(free) && free >= 0 ? free : null;
  } catch {
    return null;
  }
}

/** The floor in bytes. STRATA_MIN_FREE_GIB is for machines with a deliberate smaller disk. */
export function minimumFreeBytes(env = process.env) {
  const configured = Number(env.STRATA_MIN_FREE_GIB);
  return Number.isFinite(configured) && configured >= 0 ? configured * GIB : DEFAULT_MIN_FREE_BYTES;
}

function gb(bytes) {
  const value = bytes / GIB;
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} GB`;
}

/**
 * Throws a plain sentence when there is not enough room. `need` is the known
 * download size; the stated floor applies even when the size is unknown.
 */
export function ensureFreeSpace(path, { need = 0, label = 'this download', env = process.env, statfs = statfsSync, freeBytes = undefined } = {}) {
  const free = freeBytes === undefined ? freeBytesAt(path, { statfs }) : freeBytes;
  if (free === null) return { free: null, required: null, ok: true };
  const required = Math.max(minimumFreeBytes(env), Math.max(0, Number(need) || 0));
  if (free < required) {
    throw new Error(`There is not enough free space for ${label}. Strata needs ${gb(required)} free in ${nearestExisting(path)}, but only ${gb(free)} is free. Free some space, then try again.`);
  }
  return { free, required, ok: true };
}
