/**
 * Thinking Machines Inkling-2-Mini runs on its own shape engine program (strata-inkling, component "strata-inkling"),
 * which uses its own flags (--native, --vision, --max-context, --parallel 1..64, --draft dflash2|off, --vram-experts-gb,
 * --ram-experts-gb) rather than Qwen Flash-Next flags.
 */
import { accessSync, constants, existsSync, mkdirSync, statfsSync } from 'node:fs';
import { dirname, parse } from 'node:path';
import { setFlag } from './native-user-args.mjs';

export const INKLING_ARCH = 'inkling2_loop';
export const INKLING_FAMILY = 'inkling-2-mini';
/** Matches InkExpertSpool::disk_reserve (5 GiB) in engine/src/inkling2/ink_expert_spool.hpp. */
export const INKLING_SCRATCH_RESERVE_BYTES = 5 * 1024 * 1024 * 1024;
const GIB = 1024 * 1024 * 1024;

export const isInkling = config => config?.arch === INKLING_ARCH || config?.exe === 'component:strata-inkling' || /(^|[\\/:])strata-inkling(\.exe)?$/i.test(String(config?.exe ?? ''));
export const isInklingFamily = family => family === INKLING_FAMILY;

/** Estimated Expert scratch payload bytes for an Inkling variant (before the 5 GiB engine reserve). */
export function inklingScratchPayloadBytes(model) {
  const id = String(model ?? '').toLowerCase();
  if (/sq-20\b/.test(id)) return Math.round(6.0 * GIB);
  if (/sq-15\b/.test(id)) return Math.round(2.75 * GIB);
  if (/sq-10\b/.test(id)) return Math.round(0.5 * GIB);
  return Math.round(3.0 * GIB);
}

/** Return the `--expert-cache-dir <dir>` value from an Inkling args list, or null if absent. */
export function inklingExpertCacheDir(args) {
  if (!Array.isArray(args)) return null;
  const at = args.indexOf('--expert-cache-dir');
  if (at < 0 || at + 1 >= args.length) return null;
  const val = String(args[at + 1] ?? '').trim();
  return val && !val.startsWith('--') ? val : null;
}

function existingAnchor(targetPath, exists) {
  let cur = targetPath;
  for (let i = 0; i < 16 && cur; i++) {
    if (exists(cur)) return cur;
    const parent = dirname(cur);
    if (!parent || parent === cur) break;
    cur = parent;
  }
  return null;
}

function volumeFreeBytes(targetPath, { exists = existsSync, statfs = statfsSync } = {}) {
  const anchor = existingAnchor(targetPath, exists);
  if (!anchor) return -1;
  try {
    const info = statfs(anchor);
    const bsize = Number(info?.bsize);
    const bavail = Number(info?.bavail);
    if (!Number.isFinite(bsize) || !Number.isFinite(bavail) || bsize <= 0 || bavail < 0) return -1;
    return bsize * bavail;
  } catch {
    return -1;
  }
}

function volumeRootKey(targetPath) {
  const root = parse(String(targetPath ?? '')).root;
  return root ? root.replace(/[/\\]+$/, '').toUpperCase() : '';
}

/**
 * Discover writable fallback scratch directories across fixed disks and removable USB drives.
 * Priority: explicit STRATA_INK_SCRATCH_DIRS -> model/exe volume -> other mounted fixed & USB volumes (largest free first).
 */
export function discoverInklingScratchCandidates(config, {
  env = process.env,
  platform = process.platform,
  exists = existsSync,
  statfs = statfsSync,
  driveLetters,
} = {}) {
  const out = [];
  const seenDirs = new Set();
  const add = dir => {
    if (!dir || typeof dir !== 'string') return;
    const trimmed = dir.trim();
    if (!trimmed) return;
    const key = trimmed.toLowerCase();
    if (seenDirs.has(key)) return;
    seenDirs.add(key);
    out.push(trimmed);
  };

  const envDirs = String(env?.STRATA_INK_SCRATCH_DIRS ?? '').split(platform === 'win32' ? ';' : ':');
  for (const dir of envDirs) add(dir);

  const nativeIdx = Array.isArray(config?.args) ? config.args.indexOf('--native') : -1;
  const nativePath = nativeIdx >= 0 ? config.args[nativeIdx + 1] : null;
  for (const hint of [nativePath, config?.exe, config?.cwd]) {
    const root = volumeRootKey(hint);
    if (root && /^[A-Z]:$/.test(root)) {
      add(`${root}\\Strata\\scratch\\strata-inkling-experts`);
    }
  }

  if (platform === 'win32') {
    const letters = Array.isArray(driveLetters)
      ? driveLetters
      : 'DEFGHIJKLMNOPQRSTUVWXYZC'.split('');
    const probed = [];
    for (const letter of letters) {
      const root = `${letter.toUpperCase()}:\\`;
      if (!exists(root)) continue;
      const free = volumeFreeBytes(root, { exists, statfs });
      if (free <= 0) continue;
      probed.push({ letter: letter.toUpperCase(), free });
    }
    probed.sort((a, b) => b.free - a.free);
    for (const vol of probed) {
      add(`${vol.letter}:\\Strata\\scratch\\strata-inkling-experts`);
    }
  }

  return out;
}

/**
 * Ensure an Inkling `--expert-cache-dir` points to a writable volume that satisfies the engine's
 * 5 GiB reserve + variant scratch requirement, automatically falling back to fixed or USB drives.
 */
export function resolveInklingExpertCacheDir(args, config, model, deps = {}) {
  const configured = inklingExpertCacheDir(args);
  if (!configured) return { args, redirectedFrom: null, redirectedTo: null, note: null };

  const exists = deps.exists ?? existsSync;
  const statfs = deps.statfs ?? statfsSync;
  const mkdir = deps.mkdir ?? (dir => mkdirSync(dir, { recursive: true }));
  const canWrite = deps.canWrite ?? (dir => { accessSync(dir, constants.W_OK); return true; });
  const requiredBytes = inklingScratchPayloadBytes(model) + INKLING_SCRATCH_RESERVE_BYTES;

  const tryDirectory = dir => {
    const free = volumeFreeBytes(dir, { exists, statfs });
    if (free < requiredBytes) return { ok: false, free };
    try {
      if (!exists(dir)) mkdir(dir);
      canWrite(dir);
      return { ok: true, free };
    } catch {
      return { ok: false, free };
    }
  };

  const primary = tryDirectory(configured);
  if (primary.ok) return { args, redirectedFrom: null, redirectedTo: null, note: null };

  const candidates = Array.isArray(deps.candidates)
    ? deps.candidates
    : discoverInklingScratchCandidates(config, {
        env: deps.env,
        platform: deps.platform,
        exists,
        statfs,
        driveLetters: deps.driveLetters,
      });

  const configuredKey = configured.toLowerCase();
  for (const candidate of candidates) {
    if (candidate.toLowerCase() === configuredKey) continue;
    const probe = tryDirectory(candidate);
    if (!probe.ok) continue;
    const freeGb = (probe.free / GIB).toFixed(1);
    const reqGb = (requiredBytes / GIB).toFixed(1);
    return {
      args: setFlag(args, '--expert-cache-dir', candidate),
      redirectedFrom: configured,
      redirectedTo: candidate,
      note: `Expert scratch redirected from ${configured} to ${candidate} (${freeGb} GB free, needs >= ${reqGb} GB including 5 GB reserve).`,
    };
  }

  const reqGb = (requiredBytes / GIB).toFixed(1);
  const availGb = primary.free >= 0 ? `${(primary.free / GIB).toFixed(1)} GB` : 'unknown free space';
  throw new Error(
    `Not enough free disk space for Inkling Expert scratch at ${configured} (${availGb} free; needs at least ${reqGb} GB including the 5 GB engine reserve). Free space or connect a USB/external drive with enough free space.`
  );
}

/** A local SC117 or custom Inkling load may override the prepared context for this run only. */
export function sc117RunContext(model, arch, context, { custom = false } = {}) {
  const isSc117 = /^inkling-2-mini-sc117-sq-(10|15|20)$/i.test(String(model ?? ''));
  if (context === undefined || arch !== INKLING_ARCH || (!isSc117 && !custom)) return undefined;
  if (!Number.isSafeInteger(context) || context < 1024 || context > 1048576) {
    throw new Error('Choose an Inkling context between 1,024 and 1,048,576 tokens.');
  }
  return context;
}

export function withInklingContext(args, context) {
  return setFlag(setFlag(args, '--ctx', null), '--max-context', context);
}
/** Return the `--vision <path>` shard from an Inkling-2-Mini run config's args, or null if absent. */
export function inklingVisionShard(config) {
  if (!isInkling(config) || !Array.isArray(config?.args)) return null;
  const at = config.args.indexOf('--vision');
  if (at < 0 || at + 1 >= config.args.length) return null;
  const val = config.args[at + 1];
  return typeof val === 'string' && val.length > 0 && !val.startsWith('--') ? val : null;
}
/** Strip `--vision` and its following non-flag `.strata` shard path from an Inkling-2-Mini args list. */
export function withoutInklingVision(args) {
  const list = (Array.isArray(args) ? args : []).map(String), out = [];
  for (let i = 0; i < list.length; i++) {
    if (list[i] === '--vision') {
      if (i + 1 < list.length && !list[i + 1].startsWith('--')) i++;
      continue;
    }
    out.push(list[i]);
  }
  return out;
}
