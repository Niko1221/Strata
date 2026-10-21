import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm, readdir, symlink, open } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { inspectApp, readCapabilities, validateCatalog } from './contract.mjs';
import { reviewedSurface } from './surfaces.mjs';
import { exportCandidate, verifyCandidate, inventory } from './assets.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, '../..');
const contexts = [8192, 32768, 65536, 131072, 204800, 262144, 393216, 524288];
const fields = [{ id: 'idle_unload_s', type: 'number', min: 0, max: 86400 }];
// Representative official MCP shape, including private/future fields at every
// nested level. Values are fixture data, not a claim about the machine's hardware.
function catalog() {
  const privateFields = { config: { api_key: 'secret-config' }, setup_command: 'secret-command',
    path: 'secret-path', serial: 'secret-serial', future: { token: 'secret-future' } };
  return {
    ...privateFields, contexts, source: 'secret-source', summary: 'Official Fixture',
    families: [{ ...privateFields, family: 'qwen', title: 'Qwen3.8-Flash-Next',
      sizes: [{ ...privateFields, model: 'Q2_0', id: 'q2_0', download_gb: 66.4,
        ram_needed_gb: 48, installed: false, images: true, gpu_options: {
          ...privateFields, recommended: 'single', ignored: [{ ...privateFields, index: 1, why_ignored: 'Fixture' }],
          options: [{ ...privateFields, key: 'single', cards: [0], recommended: true }],
        } }] }],
    recommendation: { ...privateFields, family: 'qwen', model: 'Q2_0', context: 65536,
      backend: 'cuda', notes: ['Official Recommendation'], gpu: { ...privateFields, index: 0, name: 'Fixture GPU', vram_gb: 16 } },
    this_pc: { ...privateFields, ram_gb: 64, gpus: [{ ...privateFields, index: 0, vendor: 'nvidia', usable: true }] },
  };
}
const reply = raw => JSON.stringify({ jsonrpc: '2.0', id: 1, result: { structuredContent: raw } }) + '\n';
const rejects = (action, code) => assert.rejects(action, error => error.code === code && error.message === code);

async function fixture(t) {
  // Disposable files stay within the authorized checkout, never a live App or user profile.
  const root = await mkdtemp(join(here, '.contract-fixture-'));
  t.after(() => {
    const target = resolve(root);
    assert.equal(dirname(target), here);
    assert.ok(basename(target).startsWith('.contract-fixture-'));
    return rm(target, { recursive: true, force: true });
  });
  const host = join(root, 'host'), control = join(root, 'control');
  for (const dir of ['host/server', 'host/dist', 'control/python/bin', 'control/tools', 'control/strata_control'])
    await mkdir(join(root, dir), { recursive: true });
  await writeFile(join(host, 'package.json'), JSON.stringify({ type: 'module', version: '0.2.1' }));
  await writeFile(join(host, 'dist/index.html'), '<title>Official Fixture</title>');
  // Read source as text only. Importing index.mjs would start the real Host.
  await writeFile(join(host, 'server/index.mjs'), await readFile(join(repo, 'apps/host/server/index.mjs')));
  // Run the actual upstream resolver/environment logic against isolated component
  // directories. No production module import and no live installed App access.
  await writeFile(join(host, 'server/official-owner.mjs'), await readFile(join(repo, 'apps/host/server/control-runtime.mjs')));
  await writeFile(join(host, 'server/control-runtime.mjs'), `
    import * as official from './official-owner.mjs';
    import {fileURLToPath} from 'node:url';
    const components=fileURLToPath(new URL('../../components', import.meta.url));
    export const componentsDirectory=()=>components;
    export const findControl=options=>official.findControl({...options,env:{},components});
    export const installerEnvironment=(control,options)=>official.installerEnvironment(control,{...options,env:{},components});
  `);
  await writeFile(join(host, 'server/native-settings.mjs'), await readFile(join(repo, 'apps/host/server/native-settings.mjs')));
  await writeFile(join(control, 'BUNDLE.json'), JSON.stringify({ version: '0.1.46.3' }));
  await writeFile(join(control, 'setup.py'), '# Fixture; never executed\n');
  await writeFile(join(control, 'tools/strata_mcp.py'), '# Fixture; never executed\n');
  await writeFile(join(control, 'strata.py'), '# Fixture; never executed\n');
  await writeFile(join(control, 'python', process.platform === 'win32' ? 'python.exe' : 'bin/python3'), 'Not An Executable');
  await writeFile(join(root, 'app-settings.json'), '{"sentinel":"preserve"}');
  return { root, host, control };
}

async function digest(root) {
  const hash = createHash('sha256');
  async function walk(dir) {
    for (const entry of (await readdir(dir, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
      const path = join(dir, entry.name);
      hash.update(path);
      if (entry.isDirectory()) await walk(path);
      else hash.update(await readFile(path));
    }
  }
  await walk(root);
  return hash.digest('hex');
}

test('reads official values, strips nested private fields, and issues only the read-only MCP call', async t => {
  const { root, host, control } = await fixture(t), before = await digest(root);
  const inspected = await inspectApp(host);
  let calls = 0;
  const raw = catalog();
  const result = await readCapabilities(inspected, { run: async (exe, args, options) => {
    calls++;
    assert.equal(exe, join(control, 'python', process.platform === 'win32' ? 'python.exe' : 'bin/python3'));
    assert.deepEqual(args, ['-B', join(control, 'tools/strata_mcp.py'), '--root', join(root, 'install'), '--src', control]);
    assert.deepEqual(JSON.parse(options.input), { jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'strata_models', arguments: {} } });
    assert.equal(options.cwd, control);
    assert.equal(options.timeoutMs, 45000);
    for (const name of ['STRATA_NO_ACQUIRE', 'PYTHONUTF8', 'PYTHONDONTWRITEBYTECODE', 'STRATA_CONTROL_INSTALL']) assert.equal(options.env[name], '1');
    assert.equal(options.env.PYTHONIOENCODING, 'utf-8');
    return reply(raw);
  } });
  assert.equal(calls, 1);
  assert.deepEqual(result.app, { version: '0.2.1', controlVersion: '0.1.46.3', fingerprint: inspected.fingerprint });
  assert.match(inspected.fingerprint, /^[a-f0-9]{64}$/);
  assert.equal(result.schema, 1);
  assert.equal(result.state, 'compatible');
  assert.deepEqual(result.contexts, contexts);
  assert.deepEqual(result.idle, { minimum: 0, maximum: 86400, shortcuts: [300, 600, 1800] });
  assert.deepEqual(result.reasoning, { owner: 'official-composer', quickControl: false });
  assert.deepEqual(result.update, { owner: 'official-app', quickInstall: false });
  assert.deepEqual(result.surface, reviewedSurface(await readFile(join(host, 'server/index.mjs'), 'utf8')));
  assert.equal(result.surface.schema, 1);
  assert.equal(result.surface.validation, 'source-anchors-only');
  assert.deepEqual(result.surface.areas.find(a => a.area === 'context-idle'), {
    area: 'context-idle', owner: 'official-app-control', routes: [
      { path: '/api/native/models/:id/settings', method: 'GET' },
      { path: '/api/native/models/:id/settings', method: 'PATCH' }],
  });
  assert.deepEqual(result.surface.areas.map(a => a.area), ['models', 'monitor', 'providers', 'lifecycle',
    'context-idle', 'chat', 'tools-search', 'research', 'skills', 'tasks', 'workspaces', 'files', 'image-generation', 'updates']);
  assert.equal(result.catalog.recommendation.context, raw.recommendation.context);
  assert.equal(result.catalog.families[0].sizes[0].download_gb, 66.4);
  assert.equal(result.catalog.recommendation.gpu.name, 'Fixture GPU');
  assert.equal(result.catalog.families[0].sizes[0].gpu_options.options[0].recommended, true);
  assert.doesNotMatch(JSON.stringify(result), /secret-|setup_command|serial|api_key|config|future|hostDir/);
  assert.equal(await digest(root), before, 'metadata reads preserve every fixture file');
});

test('unsupported App version fails before importing host modules', async t => {
  const { host } = await fixture(t);
  await writeFile(join(host, 'package.json'), '{"version":"0.2.2"}');
  await writeFile(join(host, 'server/control-runtime.mjs'), 'throw Error("must not import");');
  await rejects(() => inspectApp(host), 'official_app_version_needs_validation');
  await rejects(() => inspectApp('relative/host'), 'official_host_required');
});

test('missing routes fail closed', async t => {
  const { host } = await fixture(t);
  await writeFile(join(host, 'server/index.mjs'), '// changed host routes');
  await rejects(() => inspectApp(host), 'official_host_contract_changed');
});

test('changed surface route, method and owner anchors fail before imports with safe CLI errors', async t => {
  for (const [before, after] of [
    ["path === '/api/model-providers' && method === 'GET'", "path === '/api/model-providers' && method === 'HEAD'"],
    ['/api/monitor', '/api/changed-monitor'],
    ['native.setDefault(nativeInstance[1])', 'changedOwner()'],
    ['native.save(nativeSettings[1],await readBody(req))', 'changedSettings()'],
    ['agent.researchCommand(', 'changedResearch('],
    ['agentTools.update(await readBody(req))', 'changedTools()'],
    ['/api/workspace/file', '/api/changed-file'],
    ['checkComponentUpdates(', 'changedUpdates('],
  ]) {
    const { host } = await fixture(t), path = join(host, 'server/index.mjs');
    const source = await readFile(path, 'utf8');
    assert.equal(reviewedSurface(source).schema, 1, 'the unmodified source must pass first');
    assert.ok(source.includes(before));
    await writeFile(path, source.replaceAll(before, after));
    await writeFile(join(host, 'server/control-runtime.mjs'), 'throw Error("secret-import-must-not-run");');
    await rejects(() => inspectApp(host), 'official_host_contract_changed');
    const cli = spawnSync(process.execPath, [join(here, 'cli.mjs'), 'snapshot', '--host', host],
      { encoding: 'utf8', windowsHide: true });
    assert.equal(cli.status, 1);
    assert.equal(cli.stderr, '');
    assert.deepEqual(JSON.parse(cli.stdout), { schema: 1, state: 'needs_validation', reason: 'official_host_contract_changed' });
  }
});

test('surface descriptors cannot inherit private metadata or retain caller mutations', async t => {
  const { host } = await fixture(t);
  const source = await readFile(join(host, 'server/index.mjs'), 'utf8');
  const first = reviewedSurface(source + '\n// secret-path secret-token https://private.invalid\n');
  for (const area of first.areas) {
    assert.deepEqual(Object.keys(area).sort(), ['area', 'owner', 'routes']);
    for (const route of area.routes) assert.deepEqual(Object.keys(route).sort(), ['method', 'path']);
  }
  assert.doesNotMatch(JSON.stringify(first), /secret-|https:|"anchors":|"commands":/);
  first.areas[0].routes[0].path = 'caller-value';
  assert.equal(reviewedSurface(source).areas[0].routes[0].path, '/api/native');
});

test('export binds the reviewed surface to source identity and preserves source and settings bytes', async t => {
  const { root, host, control } = await fixture(t), inspected = await inspectApp(host);
  const before = await digest(host), controlBefore = await digest(control);
  const settings = await readFile(join(root, 'app-settings.json'));
  const snapshot = await readCapabilities(inspected, { run: async () => reply(catalog()) });
  const output = join(root, 'candidate');
  const receipt = await exportCandidate(inspected, output, snapshot);
  const manifest = JSON.parse(await readFile(join(output, 'manifest.json')));
  assert.deepEqual(manifest.capabilities.surface, snapshot.surface);
  assert.equal(manifest.capabilities.app.fingerprint, inspected.fingerprint);
  assert.deepEqual(await inventory(join(output, 'dist')), await inventory(join(host, 'dist')));
  assert.doesNotMatch(JSON.stringify(manifest.capabilities), /secret-|hostDir|"commands":|"anchors":/);
  await verifyCandidate(output, { manifestSha256: receipt.manifestSha256, current: snapshot });
  assert.equal(await digest(host), before);
  assert.equal(await digest(control), controlBefore);
  assert.deepEqual(await readFile(join(root, 'app-settings.json')), settings);
});

test('host and selected Control drift before or during catalog reads fail closed', async t => {
  for (const name of ['host/dist/index.html', 'host/server/index.mjs', 'control/BUNDLE.json', 'control/setup.py', 'control/tools/strata_mcp.py']) {
    const { root, host } = await fixture(t);
    const inspected = await inspectApp(host);
    await rejects(() => readCapabilities(inspected, { run: async () => {
      await writeFile(join(root, name), 'Changed While Catalog Was Read');
      return reply(catalog());
    } }), 'official_source_changed');
  }
  const { host } = await fixture(t), inspected = await inspectApp(host);
  await writeFile(join(host, 'dist/index.html'), 'Already Changed');
  await rejects(() => readCapabilities(inspected, { run: () => assert.fail('must not run after drift') }), 'official_source_changed');
});

test('new official selected Control invalidates an earlier inspection', async t => {
  const { root, host } = await fixture(t), inspected = await inspectApp(host);
  const newer = join(root, 'components/control/newer');
  await mkdir(join(newer, process.platform === 'win32' ? 'python' : 'python/bin'), { recursive: true });
  await mkdir(join(newer, 'strata_control'));
  await writeFile(join(newer, 'python', process.platform === 'win32' ? 'python.exe' : 'bin/python3'), 'Not An Executable');
  await writeFile(join(newer, 'strata.py'), '# Fixture');
  await writeFile(join(newer, 'BUNDLE.json'), '{"version":"0.2.2"}');
  await writeFile(join(root, 'components/control/active.json'), '{"folder":"newer","version":"0.2.2"}');
  await rejects(() => readCapabilities(inspected, { run: () => assert.fail('must not run changed Control') }), 'official_source_changed');
});

test('replacing the selected bundled Python invalidates inspection without executing it', async t => {
  const { host } = await fixture(t), inspected = await inspectApp(host);
  await writeFile(inspected.control.python, 'A Different Non Executable Fixture');
  await rejects(() => readCapabilities(inspected, { run: () => assert.fail('must not execute replaced Python') }), 'official_source_changed');
});

test('source junctions/symlinks and oversized files are rejected', async t => {
  const { root, host } = await fixture(t);
  await mkdir(join(root, 'linked'));
  await symlink(join(root, 'linked'), join(host, 'dist/link'), process.platform === 'win32' ? 'junction' : 'dir');
  await rejects(() => inspectApp(host), 'official_source_linked');
  await rm(join(host, 'dist/link'));
  const file = await open(join(host, 'dist/large'), 'w');
  try { await file.truncate(32 * 1024 * 1024 + 1); } finally { await file.close(); }
  await rejects(() => inspectApp(host), 'official_source_too_large');
});

test('catalog and idle compatibility gates preserve official bounds and no-fit recommendations', () => {
  assert.deepEqual(validateCatalog(catalog(), [{ ...fields[0], min: 120, max: 7200 }]).idle,
    { minimum: 120, maximum: 7200, shortcuts: [300, 600, 1800] });
  const noFit = catalog();
  noFit.recommendation = { family: null, model: null, why: 'No Model Fits' };
  assert.deepEqual(validateCatalog(noFit, fields).catalog.recommendation, noFit.recommendation);
  for (const mutate of [c => c.contexts = [4096], c => c.families = [],
    c => c.recommendation.context = 4096, c => c.recommendation.model = 'unknown',
    c => c.families[0].sizes[0].download_gb = { credential: 'secret' }]) {
    const changed = catalog(); mutate(changed);
    assert.throws(() => validateCatalog(changed, fields), { code: 'official_catalog_changed' });
  }
  for (const changed of [undefined, [], [...fields, ...fields], [{ id: 'idle_unload_s', type: 'number' }],
    [{ ...fields[0], min: 301 }], [{ ...fields[0], max: 1799 }]])
    assert.throws(() => validateCatalog(catalog(), changed), { code: 'official_idle_contract_changed' });
});

test('malformed, failed, duplicate, and oversized MCP replies have safe errors; text fallback works', async t => {
  const { host } = await fixture(t), inspected = await inspectApp(host);
  for (const value of ['private path: broken', '{}', '{"id":1,"error":{"message":"secret"}}',
    '{"id":1,"result":{"isError":true}}', reply(catalog()) + reply(catalog())])
    await rejects(() => readCapabilities(inspected, { run: async () => value }), 'official_catalog_unavailable');
  await rejects(() => readCapabilities(inspected, { run: async () => 'x'.repeat(2_000_001) }), 'official_metadata_too_large');
  await rejects(() => readCapabilities(inspected, { run: async () => { throw Error('secret path'); } }), 'official_contract_unavailable');
  await rejects(() => readCapabilities(inspected, { run: async () => {
    throw Object.assign(Error('secret path'), { code: 'official_catalog_unavailable' });
  } }), 'official_catalog_unavailable');
  await rejects(() => readCapabilities(inspected, { run: async () => {
    throw Object.assign(Error('secret path'), { code: 'official_secret_path' });
  } }), 'official_contract_unavailable');
  const result = await readCapabilities(inspected, { run: async () => JSON.stringify({ id: 1,
    result: { content: [{ type: 'text', text: JSON.stringify(catalog()) }] } }) });
  assert.equal(result.catalog.recommendation.model, 'Q2_0');
});
