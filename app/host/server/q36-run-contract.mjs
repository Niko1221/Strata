/** Qwen3.6 launch contract.

 * The native Qwen3.6 engine (strata-q36) accepts a short flag list and, in --serve,
 * skips anything else with a warning. A skipped flag still looks like a successful
 * setting in the UI. This module keeps the launch argv inside the flags the engine
 * actually reads and reports unsupported settings. Private-cache capability is
 * negotiated with the running engine before it accepts requests.
 *
 * Capability list is the `a == "--..."` options in src/qwen35/q36_engine.cu::run.
 * Q36_SOURCE_BASELINE is the reviewed public commit. The live flag list is
 * Q36_CAPABILITY_REVISION. --q36-expert-store mapped-pack is not put on the
 * command line unless that engine binary's digest is allowlisted, so an older
 * binary cannot ignore the flag and still allocate the full host arena.
 */
import { createHash } from 'node:crypto';
import { closeSync, fstatSync, openSync, readFileSync, realpathSync } from 'node:fs';
import { basename } from 'node:path';
import { flagOf, groups } from './native-user-args.mjs';
import { verifiedQ36MemoryCapability } from './q36-kv-capabilities.mjs';

export const Q36_SOURCE_BASELINE = '170e0d330edd9c03b5dcddfc64b77e498387ff86';
export const Q36_CAPABILITY_REVISION = 'q36-typed-kv-1';
// Private-cache support is negotiated with the running engine's INFO response
// by serve/server.py, before requests are accepted. It is not a binary allowlist.
/** Release strata.exe files that contain --q36-expert-store. A digest
 * written in model settings is not consulted. The first entry is the
 * 2026-11-03 sm_120 link of hakua040-nsis-native-20261102. The second is
 * H:\Strata\releases\hakua040-native-sync-20261103\Release\strata.exe.
 * An older binary is not listed. Legacy placement stays the default. */
export const Q36_MAPPED_PACK_DIGESTS = new Set([
  // Local 0.5.5: source b995af13; resource/numerical gates and protected bounded/legacy/restart checks passed.
  '659607ec2d7c19c65633e4b0e3bad5e0bdf30448f2092ca33cb2d4e1413b6256',
  // Local 0.5.4: source 7bb24974, fresh CPU/GPU/sanitizer and seven protected bounded-runtime requests.
  '57d4dc6e06ef50e40d52539dd69ebf87c52f017f87fa2ea0e2621dadbc260c34',
  // Local 0.5.3: source-bound CPU/GPU/sanitizer and seven short bounded-runtime checks; full-window is separate.
  '0932216088cb3a5813175d595a5f1f7f16633f4a7883c438439ded7f5c3bc410',
  // Personal 0.5.2 r3: source-bound mapped/bounded short lifecycle checks; full-window qualification is separate.
  '13fc3b96fe71afcb98990d828d2c743f85e8b59b16174a1f7da3eb4ef9b4f8e3',
  // 0.5.1 sm_120: source-bound mapped-pack verification with protected unchanged-file receipts.
  'e75716a9d5deb6289e87741d42e7003ba82b7696f6121d5203e615d9fe596a96',
  // 0.5.0 sm_120: source-bound upstream integration; mapped-pack ABI and CRC checks retained.
  '28c4dc3991fa05b900fa956c351720121088d9041202d6acefd9b7af2749370e',
  // 0.4.1 sm_120: source-bound slicing-by-eight CRC; unchanged verified pack bytes.
  'e79b9791bdfdacd1e5f20324b8e80978af287cc1dc464d22231da8819de0f7fc',
  // 0.4.1 sm_120: unchanged verified layout; optional bounded file staging.
  '5e37033b11e6478920450f39a3b0057c2ca6d4116b1e196a1b507bf344d4fbd0',
  // 0.4.1 sm_120 source-bound rebuild: KVA guards; unchanged mapped expert layout.
  'c86b5d36e5371d8276bc73d553d2be0149a674ccc74541fa51d7eb88a2ab55a0',
  // Same mapped layout, with final-prefill STOP acknowledgement fixed; 0.4.1 sm_120.
  '2dd5dbcd2874767096b9c1eb4ac4132a96f7979bcd019ec6b779f23c9dbacc0f',
  // 0.4.1 sm_120: verified mapped pack, 131072-cell READY and real requests on RTX 5060 Ti 16 GB.
  '3797f2d368360b184b055c2130ed300645a1ece20d0bdef05dd6a3ceecf7e608',
  '512d7f8bba66b57fbe0f8b90807825d3b3172a86dd039452bc29e685ca27d046',
  'dd53df07789e0fefcb87e33acab54f5bfe1faf312ef6bad515924736b715c5b5',
  // 2026-11-03 mapped-pack host pin-pool wire-up (sm_120), unsigned; kept allowed after .6
  '6fe110650940d0c19f09909d769393769650045e6bc6eab29ccc49734dbcab29',
  // 2026-11-04 help-honesty rebuild (--q36-expert-store in --help); .6 unsigned Q36 component / live-proven
  '64226fde7905821148c78b7de85ede4d291c4df8184d80866dccba014d7c02b7',
  // 2026-11-05 opt-in STRATA_Q36_HOTSPOT_PROFILE (T07 capture path); unsigned q36 side-by-side
  '907341c74d5a9f26c784e4aaad3460e376d4cabd84635c5d032879c59b08d150',
]);
export const Q36_ENGINE_DEFAULT_CONTEXT = 32768;


/** Options that take one following value. */
export const Q36_VALUE_FLAGS = new Set([
  '--model', '--native', '--max-new', '--pool-workers', '--gpu', '--max-context', '--ctx',
  '--chunk', '--prefill', '--tokens', '--n', '--force', '--emb', '--dump-full-logits',
  '--bench-pp', '--bench-tg', '--mtp', '--draft-vocab', '--batch', '--slots',
  '--expert-vram', '--cpu-threads', '--expert-slots', '--hybrid-rows', '--q36-expert-store', '--q36-memory',
  '--prompt-cache', '--conversation-cache-mib', '--cache-type-k', '--cache-type-v',
]);
/** Options that stand alone. */
export const Q36_SWITCH_FLAGS = new Set([
  '--no-token-graph', '--greedy', '--serve', '--no-graph', '--no-warmup', '--vision', '--batch-mtp', '--private-conversation-cache',
]);
export const Q36_SUPPORTED_FLAGS = new Set([...Q36_VALUE_FLAGS, ...Q36_SWITCH_FLAGS]);

const PRIVATE_CACHE_FLAGS = new Set(['--prompt-cache', '--conversation-cache-mib']);

/** SHA-256 of a file on disk. Null when it cannot be read. A digest written in model settings is not this. */
export function q36FileSha256(path) {
  try {
    return createHash('sha256').update(readFileSync(path)).digest('hex');
  } catch {
    return null;
  }
}

export function q36MappedPackAllowed(input = {}) {
  const sha = String(input.binarySha256 ?? '').toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(sha)) return false;
  const listed = new Set([...Q36_MAPPED_PACK_DIGESTS].map(item => String(item).toLowerCase()));
  for (const item of input.mappedPackDigests ?? []) listed.add(String(item).toLowerCase());
  return listed.has(sha);
}

const MAPPED_PACK_REFUSAL = 'This engine binary is not on the mapped expert-store allowlist. The launch was not switched, and the full host arena was not requested.';
const isFlag = text => typeof text === 'string' && text.startsWith('--');

export function isQwen36Config(config = {}) {
  if (!config || typeof config !== 'object') return false;
  if (config.arch === 'qwen35moe') return true;
  if (/strata-q36/i.test(String(config.exe ?? ''))) return true;
  const name = `${config.model_name ?? ''} ${config.family ?? ''}`;
  return /qwen3\.6|qwen36|35b-a3b/i.test(name);
}

function valueSpan(list, at) {
  return at + 1 < list.length && !isFlag(list[at + 1]) ? 1 : 0;
}

function fail(code, message) {
  return { ok: false, error: { code, message } };
}

/** Validate the mode and its actual argv prerequisites before any start side effect. */
export function q36MemoryRequest(args, requested = null, layers = []) {
  const read = (flag, list = args) => {
    const found = list.filter(a => a === flag || a.startsWith(flag + '='));
    const at = list.indexOf(flag);
    if (found.length > 1 || (found.length && (at < 0 || !valueSpan(list, at) || list[at + 1] === '')))
      throw new Error(`${flag} is malformed or repeated.`);
    return at < 0 ? null : list[at + 1];
  };
  try {
    // Merging valid overrides is allowed; masking an explicit malformed token is not.
    for (const layer of layers) read('--q36-memory', layer ?? []);
    const value = read('--q36-memory');
    const modes = [value, requested].filter(v => v != null && v !== '');
    if (new Set(modes).size > 1 || modes.some(v => !['legacy', 'bounded-experts-v1'].includes(v)))
      throw new Error('q36_memory is conflicting or unsupported.');
    const mode = modes[0] ?? 'legacy';
    if (mode === 'bounded-experts-v1') {
      if (read('--q36-expert-store') !== 'mapped-pack') throw new Error('q36_memory requires explicit --q36-expert-store mapped-pack.');
      const slots = read('--expert-slots');
      // Match q36_explicit_slots' checked native int parser, including leading zeroes.
      if (!/^\d+$/.test(slots ?? '') || Number(slots) < 1 || Number(slots) > 2147483647)
        throw new Error('q36_memory requires positive explicit --expert-slots.');
    }
    return { ok: true, mode };
  } catch (error) { return fail('q36_memory_invalid', error.message); }
}

/** Theoretical persistent bytes, excluding recurrent state and scratch. Geometry is explicit. */
export function q36PersistentKvBytes(cells, sequences = 1, { k = 'f32', v = 'f32', heads = 2, headDim = 256, attentionStates = 11 } = {}) {
  if (![cells, sequences, heads, headDim, attentionStates].every(n => Number.isSafeInteger(n) && n > 0) ||
      !['f32', 'f16'].includes(k) || !['f32', 'f16'].includes(v)) return fail('kv_invalid', 'Qwen3.6 KV geometry or format is invalid.');
  const one = BigInt(cells) * BigInt(heads) * BigInt(headDim);
  const keyBytes = one * (k === 'f32' ? 4n : 2n), valueBytes = one * (v === 'f32' ? 4n : 2n);
  const bytes = (keyBytes + valueBytes) * BigInt(attentionStates) * BigInt(sequences);
  if (bytes > (1n << 64n) - 1n) return fail('kv_overflow', 'Qwen3.6 KV allocation exceeds size_t.');
  return { ok: true, bytes, bytesDecimal: bytes.toString(), keyAllocationBytes: keyBytes.toString(), valueAllocationBytes: valueBytes.toString(),
    layoutVersion: 1, scaleSchema: 'none', basis: 'theoretical', attentionStates, heads, headDim };
}

/** Decimal GiB (up to 2 places) to bytes. 18.75 and 11.89 stay exact at hundredths of a GiB. */
export function gibToBytes(gib) {
  const text = String(gib);
  if (!/^\d+(\.\d{1,2})?$/.test(text)) {
    return { ok: false, error: { code: 'bytes_invalid', message: 'A memory quantity must be a non-negative GiB value.' } };
  }
  const [whole, frac = ''] = text.split('.');
  const hundredths = BigInt(whole) * 100n + BigInt((frac + '00').slice(0, 2));
  const bytes = hundredths * (1n << 30n) / 100n;
  return { ok: true, bytes, bytesDecimal: bytes.toString() };
}

/** Legacy expert arena versus commit headroom. Missing inputs stay unknown. Shared GPU memory is not an input. */
export function admitLegacyExperts({ expertBytes = null, commitHeadroomBytes = null } = {}) {
  if (expertBytes == null || commitHeadroomBytes == null) return { ok: true, fits: null, reason: 'unknown' };
  let expert, head;
  try {
    expert = BigInt(expertBytes);
    head = BigInt(commitHeadroomBytes);
  } catch {
    return { ok: false, fits: null, reason: 'invalid' };
  }
  if (expert < 0n || head < 0n) return { ok: false, fits: null, reason: 'invalid' };
  const fits = head >= expert;
  return { ok: true, fits, reason: fits ? 'ok' : 'commit_headroom' };
}

export function driveLetter(filePath) {
  const match = String(filePath ?? '').match(/^([A-Za-z]):[\\/]/);
  return match ? match[1].toUpperCase() : null;
}

/** A requested C: model must not be launched from an H: path. Same-file proof does not authorise the swap. */
export function bindLaunchPath({ requestedPath, resolvedPath }) {
  const requested = driveLetter(requestedPath);
  const resolved = driveLetter(resolvedPath);
  if (requested && resolved && requested !== resolved) {
    return fail(
      'path_drive_substituted',
      `The selected model is on ${requested}: and the resolved file is on ${resolved}:. Qwen3.6 will not start from the other drive.`,
    );
  }
  return { ok: true, launchPath: requestedPath };
}

export function fileIdentitySync(filePath) {
  const fd = openSync(filePath, 'r');
  try {
    const st = fstatSync(fd, { bigint: true });
    let resolvedPath = filePath;
    try { resolvedPath = realpathSync.native(filePath); } catch { resolvedPath = filePath; }
    return {
      requestedPath: filePath,
      resolvedPath,
      fileId: `${st.dev}:${st.ino}`,
      volumeId: String(st.dev),
      sizeBytes: String(st.size),
      basename: basename(filePath),
    };
  } finally {
    closeSync(fd);
  }
}

/** Same file index, or the same content hash. The file name is not an identity. */
export function sameModelIdentity(a, b) {
  if (!a || !b) return false;
  if (a.fileId && b.fileId) return a.fileId === b.fileId;
  if (a.contentSha256 && b.contentSha256) return a.contentSha256 === b.contentSha256;
  return false;
}

function canonical(value) {
  return JSON.stringify(value, (_, item) => {
    if (item && typeof item === 'object' && !Array.isArray(item)) {
      return Object.fromEntries(Object.entries(item).sort(([p], [q]) => (p < q ? -1 : p > q ? 1 : 0)));
    }
    return item;
  });
}

function sha256Text(text) {
  return createHash('sha256').update(text).digest('hex');
}

function readNumberFlag(argv, flags) {
  for (const flag of flags) {
    const at = argv.indexOf(flag);
    if (at >= 0 && at + 1 < argv.length) return argv[at + 1];
  }
  return null;
}

/**
 * Build the argv and contract for one Qwen3.6 launch.
 * `input` is copied immediately. Later edits to the caller's object do not enter the argv.
 * `userArgs` are the person's flags. Any other unsupported flag is removed from the launch
 * and reported. It is not deleted from `userArgs`.
 */
export function prepareQ36Launch(input = {}) {
  const args = [...(input.args ?? [])].map(String);
  const userArgs = [...(input.userArgs ?? [])].map(String);
  const requested = { ...(input.requested ?? {}) };
  const userFlags = new Set(groups(userArgs).map(flagOf).filter(Boolean));
  const warnings = [];

  const memory = q36MemoryRequest(args, requested.memoryMode, [userArgs]);
  if (!memory.ok) return memory;
  const memoryCapability = verifiedQ36MemoryCapability(input.memoryCapability, input.binarySha256 ?? null);
  const bounded = memory.mode === 'bounded-experts-v1';
  if (bounded && !memoryCapability?.modes.includes(memory.mode))
    return fail('q36_memory_unavailable', 'q36_memory bounded-experts-v1 is not verified for this engine binary. Clear the setting or choose legacy.');
  const memoryAt = args.indexOf('--q36-memory');
  // Legacy is the binary default, including engines that do not know this flag.
  if (!bounded && memoryAt >= 0) args.splice(memoryAt, 2);
  if (bounded && memoryAt < 0) args.push('--q36-memory', memory.mode);

  const privateCache = input.privacyRequired === true || [...args, ...userArgs].some(a => PRIVATE_CACHE_FLAGS.has(a.split('=')[0]) || a === '--private-conversation-cache');
  if (privateCache) {
    for (const flag of PRIVATE_CACHE_FLAGS) {
      for (const list of [args, userArgs]) {
        const occurrences = list.filter(a => a.split('=')[0] === flag);
        const at = list.indexOf(flag);
        if (occurrences.length > 1 || (occurrences.length && (at < 0 || list[at + 1] !== '0'))) {
          return fail('unsupported_setting', `${flag} must occur once with value 0 for private Qwen3.6 requests.`);
        }
      }
      if (!args.includes(flag)) {
        if (!input.privacyRequired) return fail('unsupported_setting', 'Private Qwen3.6 requests require both --prompt-cache 0 and --conversation-cache-mib 0.');
        args.push(flag, '0');
      }
    }
    if (input.privacyRequired && !args.includes('--private-conversation-cache')) args.push('--private-conversation-cache');
  }
  const cap = input.kvCapability ? structuredClone(input.kvCapability) : null;
  const verified = cap?.schema === 1 && cap.family === 'qwen35moe' && cap.layoutVersion === 1 && cap.scaleSchema === 'none' &&
    /^[0-9a-f]{64}$/.test(input.binarySha256 ?? '') && cap.binarySha256 === input.binarySha256;
  const kv = { k: 'f32', v: 'f32' };
  for (const [side, key, flag] of [['k', 'cacheTypeK', '--cache-type-k'], ['v', 'cacheTypeV', '--cache-type-v']]) {
    const occurrences = args.filter(a => a.split('=')[0] === flag);
    const at = args.indexOf(flag);
    if (occurrences.length > 1 || (occurrences.length && (at < 0 || !valueSpan(args, at)))) return fail('unsupported_setting', `${flag} is malformed or repeated; not applied.`);
    const argValue = at >= 0 ? args[at + 1] : null;
    const persistent = requested.persistentKv ? ({ fp32: 'f32', fp16: 'f16' }[requested.persistentKv] ?? 'unsupported') : null;
    const values = [argValue, requested[key], persistent].filter(v => v != null);
    if (new Set(values).size > 1) return fail('unsupported_setting', `${flag} conflicts with another requested KV format; not applied.`);
    const type = values[0] ?? 'f32';
    const supported = cap?.[side === 'k' ? 'cacheTypesK' : 'cacheTypesV'];
    if (!['f32', 'f16'].includes(type) || (values.length && type !== 'f32' && (!verified || !supported?.includes(type))))
      return fail('unsupported_setting', `${flag} ${type} was not applied. This binary has not verified that format; the legacy FP32 KV path remains available.`);
    // Even explicit f32 does not send an unknown flag to a legacy binary.
    if (values.length && !verified && at >= 0) args.splice(at, 2);
    if (values.length && verified && (!Array.isArray(supported) || !supported.includes(type))) return fail('unsupported_setting', `${flag} ${type} is not applied by this binary.`);
    if (values.length && verified && at < 0) args.push(flag, type);
    kv[side] = type;
  }
  if (requested.expertStoreMode && !['legacy', 'mapped-pack'].includes(requested.expertStoreMode)) {
    return fail('expert_store_not_available', 'The requested expert store is not part of this engine. The launch was not switched.');
  }
  if (requested.warmupMode && !['legacy', 'off'].includes(requested.warmupMode)) {
    return fail('unsupported_setting', `Qwen3.6 warmup mode "${requested.warmupMode}" is not in capability ${Q36_CAPABILITY_REVISION}.`);
  }
  for (const [key, flag] of [['spin8Mode', '--spin8-mode']]) {
    if (requested[key]) {
      return fail('unsupported_setting', `${flag} ${requested[key]} is not a Qwen3.6 option. It was not applied and the model was not started.`);
    }
  }

  const argv = [];
  const dropped = [];
  for (let i = 0; i < args.length; i++) {
    const item = args[i];
    if (!isFlag(item)) { argv.push(item); continue; }
    const takes = Q36_VALUE_FLAGS.has(item);
    const alone = Q36_SWITCH_FLAGS.has(item);
    const n = valueSpan(args, i);
    const value = n ? args[i + 1] : null;
    if (item === '--mtp' && value != null && !/^\d+$/.test(value)) {
      if (userFlags.has(item)) {
        return fail('unsupported_setting', `--mtp ${value} is not a draft count. Qwen3.6 was not started.`);
      }
      dropped.push({ flag: item, value });
      i += n;
      continue;
    }
    if (takes) {
      if (!n) return fail('unsupported_setting', `${item} needs a value. The model was not started.`);
      if ((item === '--max-context' || item === '--ctx' || item === '--mtp' || item === '--batch' || item === '--slots') && !/^\d+$/.test(value)) {
        return fail('unsupported_setting', `${item} needs an integer. The model was not started.`);
      }
      if ((item === '--prefill' || item === '--chunk') && value !== 'auto' && !/^\d+$/.test(value)) {
        return fail('unsupported_setting', `${item} needs auto or an integer. The model was not started.`);
      }
      if (item === '--q36-expert-store' && value !== 'legacy' && value !== 'mapped-pack') {
        return fail('unsupported_setting', `${item} ${value} is not a Qwen3.6 expert store. It was not applied and the model was not started.`);
      }
      if (item === '--q36-expert-store' && value === 'mapped-pack' && !bounded && !q36MappedPackAllowed(input)) {
        return fail('expert_store_not_available', MAPPED_PACK_REFUSAL);
      }
      argv.push(item, value);
      i += 1;
      continue;
    }
    if (alone) { argv.push(item); continue; }
    if (userFlags.has(item) || PRIVATE_CACHE_FLAGS.has(item)) {
      const shown = [item, value].filter(v => v != null).join(' ');
      return fail('unsupported_setting', `${shown} is not a Qwen3.6 option. It was not applied and the model was not started.`);
    }
    dropped.push({ flag: item, value });
    i += n;
  }

  const nativeAt = argv.indexOf('--native');
  const modelAt = argv.indexOf('--model');
  const pathAt = nativeAt >= 0 ? nativeAt : modelAt;
  if (pathAt >= 0) {
    const requestedPath = argv[pathAt + 1];
    const resolvedPath = input.modelIdentity?.resolvedPath ?? requestedPath;
    const bound = bindLaunchPath({ requestedPath, resolvedPath });
    if (!bound.ok) return bound;
    if (input.modelIdentity?.fileId && input.expectedFileId && input.modelIdentity.fileId !== input.expectedFileId) {
      return fail('model_identity_mismatch', 'The opened model file is not the file that was checked. The launch was stopped.');
    }
  }

  const contextText = readNumberFlag(argv, ['--max-context', '--ctx']);
  const context = contextText != null ? Number(contextText) : Q36_ENGINE_DEFAULT_CONTEXT;
  if (contextText == null) warnings.push({ code: 'engine_default_context', message: 'No --max-context was set. The engine default is 32768.' });
  const prefillText = readNumberFlag(argv, ['--prefill', '--chunk']);
  let prefill = null;
  if (prefillText == null) warnings.push({ code: 'effective_prefill_unobserved', message: 'Prefill was not set on the command line. The engine chooses it after VRAM is known.' });
  else if (prefillText === 'auto') warnings.push({ code: 'effective_prefill_unobserved', message: 'Prefill is auto. The effective chunk is known only after the engine measures VRAM.' });
  else prefill = Number(prefillText);
  const mtpText = readNumberFlag(argv, ['--mtp']);
  const mtp = mtpText != null ? Number(mtpText) : 0;
  const batchText = readNumberFlag(argv, ['--batch', '--slots']);
  const parallel = Number.isInteger(requested.parallel) ? requested.parallel : (batchText != null ? Number(batchText) : 1);
  if (privateCache && (parallel > 1 || Number(batchText) > 1 || argv.includes('--batch-mtp'))) {
    return fail('q36_private_cache_parallel', 'Private Qwen3.6 requests currently run one at a time. Set parallel requests to 1.');
  }
  const warmupMode = argv.includes('--no-warmup') || requested.warmupMode === 'off' ? 'off' : 'legacy';
  let expertStoreMode = 'legacy';
  const storeAt = argv.indexOf('--q36-expert-store');
  if (storeAt >= 0) expertStoreMode = argv[storeAt + 1];
  if (requested.expertStoreMode === 'mapped-pack' && expertStoreMode === 'legacy') {
    if (!q36MappedPackAllowed(input)) return fail('expert_store_not_available', MAPPED_PACK_REFUSAL);
    argv.push('--q36-expert-store', 'mapped-pack');
    expertStoreMode = 'mapped-pack';
  } else if (requested.expertStoreMode === 'legacy' && expertStoreMode === 'mapped-pack') {
    return fail('unsupported_setting', 'The launch asked for mapped-pack and also for the legacy store. It was not started.');
  }

  if (!input.binarySha256) {
    warnings.push({ code: 'binary_digest_unbound', message: 'The engine binary digest is not bound. Only the legacy expert placement is allowed.' });
  }

  const identity = input.modelIdentity ?? {};
  const contract = {
    schemaVersion: 1,
    runId: input.runId ?? null,
    generation: input.generation ?? null,
    backend: { family: 'qwen36', binarySha256: input.binarySha256 ?? null, capabilityRevision: Q36_CAPABILITY_REVISION, kvCapability: verified ? cap : null, memoryCapability: bounded ? memoryCapability : null },
    model: {
      requestedPath: identity.requestedPath ?? (pathAt >= 0 ? argv[pathAt + 1] : null),
      resolvedPath: identity.resolvedPath ?? null,
      fileId: identity.fileId ?? null,
      volumeId: identity.volumeId ?? null,
      sizeBytes: identity.sizeBytes ?? null,
      contentSha256: identity.contentSha256 ?? null,
      geometryDigest: identity.geometryDigest ?? null,
      tokenizerDigest: identity.tokenizerDigest ?? null,
    },
    requested: {
      context: requested.context ?? context,
      prefill: requested.prefill ?? (prefillText === 'auto' ? 'auto' : prefill),
      mtp: requested.mtp ?? mtp,
      parallel,
      kv_k: kv.k, kv_v: kv.v,
      q36_memory: memory.mode,
    },
    effective: {
      context,
      prefill,
      mtp,
      parallel,
      persistentKv: kv.k === kv.v ? (kv.k === 'f32' ? 'fp32' : 'fp16') : 'mixed',
      kv_k: kv.k, kv_v: kv.v,
      kvLayoutVersion: verified ? 1 : null,
      kvScaleSchema: verified ? 'none' : null,
      kvQuality: 'experimental-unqualified',
      kvEstimate: (() => { const estimate = q36PersistentKvBytes(context, parallel, kv); if (!estimate.ok) return null; const { bytes, ...serial } = estimate; return serial; })(),
      warmupMode,
      expertStoreMode,
      q36_memory: memory.mode,
      privateCache,
      scopedConversationCache: argv.includes('--private-conversation-cache'),
    },
    observed: { status: 'unobserved', generation: null, processId: null, context: null, prefill: null, parallel: null, kv_k: null, kv_v: null },
    securityPolicyDigest: input.securityPolicyDigest ?? null,
    argv,
    warnings,
  };
  // The digest continues to identify the immutable launch plan. Runtime observations may resolve auto values later.
  contract.planned = { ...contract.effective };
  contract.configDigest = sha256Text(canonical({
    capabilityRevision: Q36_CAPABILITY_REVISION,
    argv,
    effective: contract.effective,
    fileId: contract.model.fileId,
    contentSha256: contract.model.contentSha256,
  }));

  const note = dropped.length
    ? `Left out because this Qwen3.6 engine does not take them (not applied): ${dropped.map(d => [d.flag, d.value].filter(v => v != null).join(' ')).join(', ')}. Saved settings were kept.`
    : '';
  return { ok: true, argv, note, warnings, dropped, preservedUserArgs: userArgs, contract };
}

/** Only a receipt from this launch's real child may resolve requested/auto settings. */
export function observeQ36Runtime(contract, state) {
  const value = state?.effective;
  if (!contract?.generation || contract.generation !== state?.launchGeneration || !state.child?.processId ||
      value?.generation !== state.child.generation || value.processId !== state.child.processId) return contract;
  const mismatches = [];
  for (const key of ['context', 'prefill', 'mtp', 'parallel']) {
    const requested = contract.requested[key];
    if (value[key] != null && typeof requested === 'number' && value[key] !== requested) mismatches.push({ field: key, requested, observed: value[key] });
  }
  // Capability negotiation stays separate: observing f16 never grants permission to request it.
  for (const key of ['kv_k', 'kv_v']) {
    const wanted = contract.requested[key] ?? 'f32';
    if (value[key] != null && value[key] !== wanted) mismatches.push({ field: key, requested: wanted, observed: value[key] });
  }
  if (value.kv_layout_version === 1 && value.kv_scale_schema === 'none') {
    const actual = q36PersistentKvBytes(value.context, value.parallel, { k: value.kv_k, v: value.kv_v, heads: value.kv_heads, headDim: value.kv_head_dim, attentionStates: value.kv_attention_states });
    if (!actual.ok || actual.bytesDecimal !== value.kv_bytes) mismatches.push({ field: 'kv_bytes', requested: actual.bytesDecimal ?? null, observed: value.kv_bytes });
  }
  const complete = ['context', 'prefill', 'parallel', 'kv_k', 'kv_v', 'mtp', ...(contract.backend.kvCapability ? ['kv_layout_version', 'kv_scale_schema', 'kv_heads', 'kv_head_dim', 'kv_attention_states', 'kv_bytes'] : [])].every(key => value[key] != null);
  if (contract.backend.kvCapability && value.kv_layout_version != null && (value.kv_layout_version !== 1 || value.kv_scale_schema !== 'none')) mismatches.push({ field: 'kv_layout', requested: '1:none', observed: `${value.kv_layout_version}:${value.kv_scale_schema}` });
  const observed = { ...contract.observed, ...value, mismatches, status: mismatches.length ? 'mismatch' : complete ? 'observed' : 'partial' };
  const effective = { ...contract.effective };
  for (const key of ['context', 'prefill', 'mtp', 'parallel']) if (value[key] != null) effective[key] = value[key];
  return { ...contract, effective, observed };
}

/** Host arena measured for UD-Q4_K_XL on 2026-11-02: "no host memory for the experts (19195 MiB)". Not a size for every Qwen3.6 file. */
export const Q36_UD_Q4_K_XL_HOST_EXPERT_BYTES = 19195n << 20n;

function gibPhrase(bytes) {
  const gib = Number(bytes) / 1024 / 1024 / 1024;
  return (Math.round(gib * 10) / 10).toFixed(1);
}

/**
 * Legacy placement copies every expert into a private arena. For the measured UD-Q4_K_XL file,
 * refuse to start when free physical RAM cannot hold that arena plus a reserve (the larger of 4 GiB and 20% of RAM).
 * Other Qwen3.6 files stay unknown: their expert bytes are not invented.
 */
export function admitQ36LegacyPhysical({ modelName = '', availablePhysicalBytes = null, ramBytes = null } = {}) {
  if (!/q4_k_xl/i.test(String(modelName))) return { decision: 'unknown', fits: null, reason: 'expert_bytes_unknown' };
  if (availablePhysicalBytes == null || ramBytes == null) return { decision: 'unknown', fits: null, reason: 'memory_unknown' };
  let free, ram;
  try {
    free = BigInt(availablePhysicalBytes);
    ram = BigInt(ramBytes);
  } catch {
    return { decision: 'unknown', fits: null, reason: 'memory_unknown' };
  }
  if (free < 0n || ram < 0n) return { decision: 'unknown', fits: null, reason: 'memory_unknown' };
  const floor = 4n << 30n;
  const fifth = ram / 5n;
  const reserve = fifth > floor ? fifth : floor;
  const need = Q36_UD_Q4_K_XL_HOST_EXPERT_BYTES + reserve;
  if (free < need) {
    return {
      decision: 'rejected',
      fits: false,
      reason: 'physical_reserve',
      needBytes: need.toString(),
      reserveBytes: reserve.toString(),
      availableBytes: free.toString(),
      message: `Qwen3.6 UD-Q4_K_XL places about 18.7 GiB of experts in RAM with the current engine. ${gibPhrase(free)} GiB is free and ${gibPhrase(reserve)} GiB stays aside for the rest of the PC. The model was not started.`,
    };
  }
  return { decision: 'allowed', fits: true, reason: 'ok', needBytes: need.toString(), reserveBytes: reserve.toString(), availableBytes: free.toString() };
}
