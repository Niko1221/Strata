/** The bundled Qwen3.8 direction, using setup.py's existing control-vector contract. */
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, realpathSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { capture, groups, merge } from './native-user-args.mjs';
import { q36FileSha256 } from './q36-run-contract.mjs';

export const PROJECTION_KEY = 'refusal_direction_removal';
export const VECTOR_FILE = 'Qwen3.8-Flash-Next-experimental-speed-projection.gguf';
export const VECTOR_SHA256 = 'ef0724c5b79297e481017be85769832bc26eb220c984ecba2b0e1b0c8426312b';
export const VECTOR_BYTES = 483520;
const FLAGS = ['--control-vector-scaled', '--control-vector-layer-range', '--cvec-mode', '--cvec-dir'];
const isVectorFlag = token => /^--(?:control-vector(?:-scaled|-layer-range)?|cvec-(?:mode|dir))(?:=|$)/.test(token);
const cache = new Map();
const fail = message => { throw Object.assign(new Error(message), { code: 'refusal_projection_unavailable' }); };

/** A filename or version never grants support. Query the selected binary without a model or saved env. */
export function probeRefusalProjection(exe, { run = spawnSync, hash = q36FileSha256 } = {}) {
  const before = exe ? hash(exe) : null;
  if (!before) return null;
  const cached = cache.get(exe);
  if (run === spawnSync && cached?.binarySha256 === before) return cached;
  const env = Object.fromEntries(Object.entries(process.env).filter(([key]) =>
    ['PATH', 'SYSTEMROOT', 'WINDIR', 'LD_LIBRARY_PATH', 'CUDA_PATH'].includes(key.toUpperCase())));
  let help;
  try {
    const result = run(exe, ['--help'], { cwd: dirname(exe), env, encoding: 'utf8', windowsHide: true, timeout: 3000, maxBuffer: 65536 });
    if (result.error || result.status !== 0) return null;
    help = `${result.stdout ?? ''}\n${result.stderr ?? ''}`;
  }
  catch { return null; }
  if (hash(exe) !== before || Buffer.byteLength(help) > 65536 ||
      !FLAGS.every(flag => new RegExp('^\\s*' + flag + '\\s', 'm').test(help)) ||
      !/--cvec-mode\s+add\|project/.test(help) || !/cvec=0\|1/.test(help)) return null;
  const result = { binarySha256: before, project: true };
  if (run === spawnSync) cache.set(exe, result);
  return result;
}

export function projectionModelReason(config, { policy, family, backend } = {}) {
  if (policy?.id !== 'q38' || /prism|turboquant/i.test(String(backend ?? ''))) return 'This serving adapter does not support refusal-direction removal (Qwen3.6, PrismML, TurboQuant and DeepSeek are unsupported).';
  // setup.py enables the shipped vector only for these original residual-stream families.
  if (!['qwen', 'coder'].includes(config.family ?? family))
    return 'The bundled vector is for the original Qwen3.8-Flash-Next only; this model has no verified compatible vector.';
  return null;
}

export const projectionVectorPath = root => join(root, 'data', 'experimental-speed-projection', VECTOR_FILE);
export function verifyProjectionVector(root) {
  const path = projectionVectorPath(root);
  try {
    // Do not follow a replaced sidecar outside the packaged Control data directory.
    if (realpathSync(path) !== resolve(path) || !statSync(path).isFile() || statSync(path).size !== VECTOR_BYTES) throw new Error();
    const pin = JSON.parse(readFileSync(join(dirname(path), 'PROVENANCE.json'), 'utf8')).vector;
    if (pin.file !== VECTOR_FILE || pin.bytes !== VECTOR_BYTES || pin.sha256 !== VECTOR_SHA256 || pin.kind !== 'controlvector' || pin.default !== 'off' ||
        createHash('sha256').update(readFileSync(path)).digest('hex') !== VECTOR_SHA256) throw new Error();
  } catch { fail('The bundled refusal-direction vector is missing or does not match its trusted size and SHA-256.'); }
  return path;
}

export const projectionArgs = path => ['--control-vector-scaled', `${path}:1.0`, '--control-vector-layer-range', '4', '44', '--cvec-mode', 'project', '--cvec-dir', 'per-layer'];
const effectiveArgs = config => merge(config.args ?? [], capture(config));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
/** Recognise only the exact bundled setup representation. Never own arbitrary user control vectors. */
function ownedBundle(args, root) {
  const found = groups(args).filter(g => isVectorFlag(g[0]));
  const expected = groups(projectionArgs(projectionVectorPath(root)));
  return found.length === expected.length && expected.every(g => found.some(f => same(f, g)));
}
function withoutOwnedBundle(args, root) {
  return ownedBundle(args, root) ? groups(args).filter(g => !isVectorFlag(g[0])).flat() : args;
}
export function projectionRequested(config, root) {
  return config[PROJECTION_KEY] ?? ownedBundle(effectiveArgs(config), root);
}
/** A scalar owns no user arguments. Only setup's exact pinned bundle can be cleared with it. */
export function clearOwnedProjection(config, root) {
  const next = { ...config };
  for (const key of ['args', 'args_written', 'user_args']) if (Array.isArray(next[key])) next[key] = withoutOwnedBundle(next[key], root);
  return next;
}
export function projectionView(config, context) {
  const requested = projectionRequested(config, context.root);
  let reason = projectionModelReason(config, context);
  if (!reason && !context.capability?.project) reason = 'The selected engine binary has not verified the control-vector projection flags.';
  if (!reason) { try { verifyProjectionVector(context.root); } catch (error) { reason = error.message; } }
  const cleaned = clearOwnedProjection(config, context.root);
  if (!reason && effectiveArgs(cleaned).some(isVectorFlag)) reason = 'Custom control-vector arguments are present. Keep them, or remove them yourself before enabling the bundled vector.';
  return { requested, supported: !reason, reason, vectorSha256: VECTOR_SHA256, layers: '4–44' };
}
/** Admission is also called synchronously before creating a job or unloading the current model. */
export function admitProjection(config, context) {
  if (!projectionRequested(config, context.root)) return;
  const view = projectionView(config, context);
  if (!view.supported) fail(view.reason);
}
/** Default launch is identical. ON adds the same args as setup; OFF clears only our exact bundle. */
export function applyProjection(config, context) {
  const requested = projectionRequested(config, context.root);
  if (!requested && config[PROJECTION_KEY] !== false) return config;
  const next = clearOwnedProjection(config, context.root);
  if (!requested) return next;
  admitProjection(config, context);
  next.args = [...(next.args ?? []), ...projectionArgs(verifyProjectionVector(context.root))];
  return next;
}
