/** Opt-in metadata reader for an explicitly trusted installed App host. No server is started. */
import { lstat, open, opendir } from 'node:fs/promises';
import { dirname, isAbsolute, join, parse, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';

const CONTEXTS = [8192, 32768, 65536, 131072, 204800, 262144, 393216, 524288];
const MAX_FILE = 32 * 1024 * 1024, MAX_TOTAL = 128 * 1024 * 1024;
const MAX_OUTPUT = 2_000_000, TIMEOUT = 45_000;
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const failure = code => Object.assign(new Error(code), { code });
const codes = new Set(['official_contract_unavailable', 'official_source_linked', 'official_source_invalid',
  'official_source_too_large', 'official_source_changed', 'official_metadata_timeout',
  'official_metadata_unavailable', 'official_metadata_too_large', 'official_metadata_failed',
  'official_catalog_changed', 'official_idle_contract_changed', 'official_host_required',
  'official_app_version_needs_validation', 'official_host_contract_changed',
  'official_control_unavailable', 'official_catalog_unavailable']);
const guarded = async action => {
  try { return await action(); }
  catch (error) {
    if (codes.has(error?.code)) throw failure(error.code);
    throw failure('official_contract_unavailable');
  }
};

// Reject links in ancestors too, including Windows junctions. The caller trusts the
// host's executable modules; this is a consistency check, not a sandbox for hostile code.
async function unlinked(path) {
  const absolute = resolve(path), root = parse(absolute).root;
  let current = root, info = await lstat(root);
  for (const part of absolute.slice(root.length).split(/[\\/]/).filter(Boolean)) {
    current = join(current, part);
    info = await lstat(current);
    if (info.isSymbolicLink()) throw failure('official_source_linked');
  }
  return info;
}

async function bytes(path, budget = { total: 0 }) {
  const before = await unlinked(path);
  if (!before.isFile()) throw failure('official_source_invalid');
  if (before.size > MAX_FILE || budget.total + before.size > MAX_TOTAL)
    throw failure('official_source_too_large');
  const file = await open(path, 'r');
  try {
    const opened = await file.stat();
    if (opened.ino !== before.ino || opened.dev !== before.dev || opened.size !== before.size)
      throw failure('official_source_changed');
    const data = Buffer.alloc(before.size + 1);
    let used = 0;
    while (used < data.length) {
      const { bytesRead } = await file.read(data, used, data.length - used, used);
      if (!bytesRead) break;
      used += bytesRead;
    }
    const after = await unlinked(path), end = await file.stat();
    if (used !== before.size || after.ino !== before.ino || after.dev !== before.dev ||
        after.mtimeMs !== before.mtimeMs || end.mtimeMs !== before.mtimeMs || end.size !== before.size)
      throw failure('official_source_changed');
    budget.total += used;
    return data.subarray(0, used);
  } finally { await file.close(); }
}
const jsonFile = async path => JSON.parse((await bytes(path)).toString('utf8').replace(/^\uFEFF/, ''));
const identity = info => `${info.dev}:${info.ino}:${info.size}:${info.mtimeMs}:${info.ctimeMs}`;

async function fingerprint(hostDir, control = null) {
  const hash = createHash('sha256'), budget = { total: 0 };
  let count = 0;
  async function add(path, label) {
    const data = await bytes(path, budget);
    hash.update(`${label}\0${data.length}\0`).update(data);
  }
  async function tree(path, label, depth = 0) {
    if (depth > 24) throw failure('official_source_too_large');
    const info = await unlinked(path);
    if (!info.isDirectory()) throw failure('official_source_invalid');
    const entries = [];
    for await (const entry of await opendir(path)) {
      if (++count > 4096) throw failure('official_source_too_large');
      entries.push(entry);
    }
    for (const entry of entries.sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0)) {
      if (entry.isSymbolicLink()) throw failure('official_source_linked');
      const child = join(path, entry.name), name = `${label}/${entry.name}`;
      if (entry.isDirectory()) await tree(child, name, depth + 1);
      else if (entry.isFile()) await add(child, name);
      else throw failure('official_source_invalid');
    }
  }
  await add(join(hostDir, 'package.json'), 'package.json');
  await tree(join(hostDir, 'dist'), 'dist');
  await tree(join(hostDir, 'server'), 'server');
  if (control) {
    for (const name of ['BUNDLE.json', 'setup.py', 'tools/strata_mcp.py'])
      await add(join(control.dir, name), `control/${name}`);
  }
  return hash.digest('hex');
}

function run(executable, args, options) {
  return new Promise((accept, reject) => {
    const child = spawn(executable, args, { cwd: options.cwd, env: options.env,
      windowsHide: true, shell: false, stdio: ['pipe', 'pipe', 'pipe'] });
    let chunks = [], size = 0, errors = 0, done = false;
    const finish = (error, value) => {
      if (done) return;
      done = true; clearTimeout(alarm);
      if (error) { child.kill(); reject(error); } else accept(value);
    };
    const alarm = setTimeout(() => finish(failure('official_metadata_timeout')), options.timeoutMs);
    child.on('error', () => finish(failure('official_metadata_unavailable')));
    child.stdout.on('data', chunk => {
      size += chunk.length;
      if (size > MAX_OUTPUT) return finish(failure('official_metadata_too_large'));
      chunks.push(chunk);
    });
    child.stderr.on('data', chunk => {
      errors += chunk.length;
      if (errors > 100_000) finish(failure('official_metadata_too_large'));
    });
    child.on('close', code => finish(code === 0 ? null : failure('official_metadata_failed'),
      Buffer.concat(chunks).toString('utf8')));
    child.stdin.on('error', () => {});
    child.stdin.end(options.input);
  });
}

// Each nested record has its own allowlist. Never forward raw config, commands,
// paths, device serials, credentials, or future fields from Control.
const gpu = { index: 'number', name: 'string', vram_gb: 'number', usable: 'boolean', vendor: 'string' };
const option = { key: 'string', cards: ['number'], label: 'string', verdict: 'string', summary: 'string',
  words: 'string', vram_gb: 'number', spill_gb: 'number', ram_use_gb: 'number', spare_ram_gb: 'number',
  notes: ['string'], recommended: 'boolean' };
const size = { model: 'string', about: 'string', download_gb: 'number', experimental: 'boolean',
  images: 'boolean', images_reader: 'string', ram_needed_gb: 'number', experts_gb: 'number',
  mem_total: 'boolean', engine_min: 'string', memory_gb: 'number', on_this_pc: 'string',
  installed: 'boolean', id: 'string', gpu_options: { options: [option], recommended: 'string',
    ignored: [{ ...gpu, why_ignored: 'string' }], cards: 'number', vendor: 'string', other_vendor_note: 'string' } };
const catalogShape = {
  summary: 'string', contexts: ['number'], context_note: 'string',
  families: [{ family: 'string', title: 'string', about: 'string', experimental: 'boolean', images: 'boolean', sizes: [size] }],
  recommendation: { family: 'string', model: 'string', title: 'string', context: 'number', backend: 'string',
    gpu, download_gb: 'number', why: 'string', notes: ['string'] },
  this_pc: { ram_gb: 'number', gpus: [gpu], engine_vendor: 'string', gpu_note: 'string' },
};
function project(value, shape) {
  if (value === null) return null;
  if (Array.isArray(shape)) {
    if (!Array.isArray(value) || value.length > 1024) throw failure('official_catalog_changed');
    return value.map(item => project(item, shape[0]));
  }
  if (typeof shape === 'string') {
    if (typeof value !== shape || (shape === 'number' && !Number.isFinite(value)) ||
        (shape === 'string' && value.length > 16_384)) throw failure('official_catalog_changed');
    return value;
  }
  if (!object(value)) throw failure('official_catalog_changed');
  return Object.fromEntries(Object.entries(shape).filter(([key]) => Object.hasOwn(value, key))
    .map(([key, child]) => [key, project(value[key], child)]));
}

export function validateCatalog(raw, fields) {
  if (!object(raw) || !Array.isArray(raw.families) || !raw.families.length ||
      JSON.stringify(raw.contexts) !== JSON.stringify(CONTEXTS) || !object(raw.recommendation) ||
      !raw.families.every(f => object(f) && typeof f.family === 'string' && typeof f.title === 'string' &&
        Array.isArray(f.sizes) && f.sizes.length && f.sizes.every(s => object(s) && typeof s.model === 'string')))
    throw failure('official_catalog_changed');
  const rec = raw.recommendation;
  if (!(rec.family === null && rec.model === null) &&
      (!raw.families.some(f => f.family === rec.family && f.sizes.some(s => s.model === rec.model)) ||
       !raw.contexts.includes(rec.context))) throw failure('official_catalog_changed');
  const idle = Array.isArray(fields) ? fields.filter(f => f?.id === 'idle_unload_s') : [];
  if (idle.length !== 1 || idle[0].type !== 'number' || !Number.isSafeInteger(idle[0].min) ||
      !Number.isSafeInteger(idle[0].max) || idle[0].min < 0 || idle[0].min > 300 || idle[0].max < 1800)
    throw failure('official_idle_contract_changed');
  return { contexts: [...raw.contexts], idle: { minimum: idle[0].min, maximum: idle[0].max,
    shortcuts: [300, 600, 1800] }, catalog: project(raw, catalogShape) };
}

/** hostDir must be an explicit absolute path to official, trusted App code. */
export async function inspectApp(hostDir) {
  return guarded(async () => {
    if (typeof hostDir !== 'string' || !isAbsolute(hostDir)) throw failure('official_host_required');
    hostDir = resolve(hostDir);
    const { version: appVersion } = await jsonFile(join(hostDir, 'package.json'));
    if (appVersion !== '0.2.1') throw failure('official_app_version_needs_validation');
    const hostFingerprint = await fingerprint(hostDir);
    const source = (await bytes(join(hostDir, 'server/index.mjs'))).toString('utf8');
    for (const route of ['/api/native', '/api/events', '/api/updates/check', '/api/model-providers'])
      if (!source.includes(route)) throw failure('official_host_contract_changed');
    // Query strings also avoid stale module instances on a later inspection.
    const owner = await import(`${pathToFileURL(join(hostDir, 'server/control-runtime.mjs')).href}?contract=${hostFingerprint}`);
    const { NATIVE_FIELDS: fields } = await import(`${pathToFileURL(join(hostDir, 'server/native-settings.mjs')).href}?contract=${hostFingerprint}`);
    if (typeof owner.findControl !== 'function' || typeof owner.installerEnvironment !== 'function' ||
        typeof owner.componentsDirectory !== 'function') throw failure('official_host_contract_changed');
    const control = owner.findControl({ hostDir });
    if (!control || !isAbsolute(control.dir) || !isAbsolute(control.python)) throw failure('official_control_unavailable');
    const pythonInfo = await unlinked(control.python);
    if (!pythonInfo.isFile()) throw failure('official_control_unavailable');
    const bundle = await jsonFile(join(control.dir, 'BUNDLE.json'));
    const controlVersion = bundle.version;
    if (typeof controlVersion !== 'string' || !/^\d+\.\d+\.\d+(?:\.\d+)?(?:[-+][\w.-]+)?$/.test(controlVersion))
      throw failure('official_control_unavailable');
    const inspected = { hostDir, appVersion, controlVersion, control, owner, fields,
      pythonIdentity: identity(pythonInfo), fingerprint: await fingerprint(hostDir, control) };
    if (await fingerprint(hostDir) !== hostFingerprint ||
        (await jsonFile(join(hostDir, 'package.json'))).version !== appVersion ||
        (await jsonFile(join(control.dir, 'BUNDLE.json'))).version !== controlVersion)
      throw failure('official_source_changed');
    await unchanged(inspected);
    return inspected;
  });
}

async function unchanged(inspected) {
  const selected = inspected.owner.findControl({ hostDir: inspected.hostDir });
  if (selected?.dir !== inspected.control.dir || selected?.python !== inspected.control.python ||
      await fingerprint(inspected.hostDir, inspected.control) !== inspected.fingerprint)
    throw failure('official_source_changed');
  const pythonInfo = await unlinked(selected.python);
  if (!pythonInfo.isFile() || identity(pythonInfo) !== inspected.pythonIdentity)
    throw failure('official_source_changed');
}

/** run is a test seam only: (executable, args, {cwd, env, input, timeoutMs}) => stdout. */
export async function readCapabilities(inspected, { run: runner = run } = {}) {
  return guarded(async () => {
    await unchanged(inspected);
    const { owner, control } = inspected;
    const installDir = join(dirname(owner.componentsDirectory()), 'install');
    const input = JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'tools/call',
      params: { name: 'strata_models', arguments: {} } }) + '\n';
    const text = await runner(control.python, ['-B', join(control.dir, 'tools/strata_mcp.py'),
      '--root', installDir, '--src', control.dir], { cwd: control.dir,
      env: { ...owner.installerEnvironment(control, { installDir }), STRATA_NO_ACQUIRE: '1',
        PYTHONDONTWRITEBYTECODE: '1', PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' }, input, timeoutMs: TIMEOUT });
    await unchanged(inspected);
    if (typeof text !== 'string' || Buffer.byteLength(text) > MAX_OUTPUT) throw failure('official_metadata_too_large');
    let rows;
    try { rows = text.split(/\r?\n/).filter(line => line.trim()).map(line => JSON.parse(line)); }
    catch { throw failure('official_catalog_unavailable'); }
    const replies = rows.filter(row => row?.id === 1);
    const result = replies.length === 1 && !replies[0].error ? replies[0].result : null;
    if (!object(result) || result.isError) throw failure('official_catalog_unavailable');
    let raw = result.structuredContent;
    if (raw === undefined) {
      try { raw = JSON.parse(result.content.find(item => item.type === 'text').text); }
      catch { throw failure('official_catalog_unavailable'); }
    }
    const controls = validateCatalog(raw, inspected.fields);
    return { schema: 1, state: 'compatible', app: { version: inspected.appVersion,
      controlVersion: inspected.controlVersion, fingerprint: inspected.fingerprint }, ...controls,
      reasoning: { owner: 'official-composer', quickControl: false },
      update: { owner: 'official-app', quickInstall: false } };
  });
}
