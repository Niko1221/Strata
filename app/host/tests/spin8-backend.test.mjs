import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { buildArgs, findBinary, start, restart, status, probeCapabilities } from '../server/supervisor.mjs';
import { buildSpin8Args, probeBackendCapabilities, spin8Request, SPIN8_MODES, BONSAI_SOURCES, BONSAI_SM120_PATCH } from '../server/spin8-backend.mjs';

const config = { beginner: { contextLength: '32k' }, expert: [] };
const options = { modelPath: '/models/model.gguf', config, hardware: { cpu: { cores: 8 } }, port: 8198 };
const sha = path => createHash('sha256').update(readFileSync(path)).digest('hex');

test('approved standard-sm120 source patch has the exact capability digest', () => {
  assert.equal(sha(new URL('../../../tools/spin8_backend_sm120.patch', import.meta.url)), BONSAI_SM120_PATCH.sha256);
});

test('omitted and explicit off produce the exact existing default command', () => {
  const expected = ['-m', '/models/model.gguf', '--host', '127.0.0.1', '--port', '8198', '-c', '32768',
    '--metrics', '--slots', '--jinja', '--no-webui', '-ngl', '999', '-t', '6', '-ctk', 'q8_0', '-ctv', 'q8_0',
    '-b', '2048', '-ub', '512', '-fa', 'on', '-np', '1', '--temp', '0.6', '--top-p', '0.95', '--top-k', '40', '--repeat-penalty', '1.05'];
  assert.deepEqual(buildArgs(options), expected);
  assert.deepEqual(buildArgs({ ...options, config: { ...config, spin8: { mode: 'off' } } }), expected);
  assert.deepEqual(buildSpin8Args(config), []);
});

test('all active modes and profiles fail typed unavailable before lifecycle state changes', async () => {
  const before = status();
  for (const mode of SPIN8_MODES.filter(mode => mode !== 'off')) {
    const changed = { ...options, modelId: 'test', config: { ...config, spin8: { mode } } };
    assert.throws(() => buildArgs(changed), { code: 'SPIN8_BACKEND_UNAVAILABLE' });
    await assert.rejects(start(changed), { code: 'SPIN8_BACKEND_UNAVAILABLE' });
    await assert.rejects(restart(changed), { code: 'SPIN8_BACKEND_UNAVAILABLE' });
  }
  assert.throws(() => buildSpin8Args({ spin8: { mode: 'off', profilePath: '/tmp/profile.json' } }), { code: 'SPIN8_BACKEND_UNAVAILABLE' });
  assert.deepEqual(status(), before);
});

test('bad modes and profile paths fail validation without spawning', () => {
  for (const spin8 of [null, [], 'on']) assert.throws(() => spin8Request({ spin8 }), { code: 'SPIN8_INVALID_CONFIG' });
  assert.throws(() => spin8Request({ spin8: { mode: 'magic' } }), { code: 'SPIN8_INVALID_MODE' });
  for (const profilePath of ['', '   ', 5, 'a\0b']) {
    assert.throws(() => spin8Request({ spin8: { profilePath } }), { code: 'SPIN8_INVALID_PROFILE' });
  }
});

test('Automatic resolver refuses root, version, CMake, activated generic and environment binaries', () => {
  const base = mkdtempSync(join(tmpdir(), 'strata-bonsai-resolve-'));
  try {
    const engine = join(base, 'engine');
    const modelsDir = join(base, 'models');
    const componentsDir = join(base, 'components');
    const dir = join(engine, 'prism', 'bin', 'Release');
    mkdirSync(dir, { recursive: true });
    const nested = join(dir, 'llama-server.exe');
    writeFileSync(nested, 'fixture');
    assert.equal(findBinary({ platform: 'win32', env: {}, modelsDir, componentsDir }), null);
    const root = join(engine, 'llama-server.exe');
    writeFileSync(root, 'fixture');
    assert.equal(findBinary({ platform: 'win32', env: {}, modelsDir, componentsDir }), null);
    const generic=join(componentsDir,'generic-gguf'),activated=join(generic,'v1');mkdirSync(activated,{recursive:true});
    const packaged=join(activated,'llama-server.exe');writeFileSync(packaged,'fixture');
    writeFileSync(join(generic,'active.json'),JSON.stringify({folder:'v1'}));
    assert.deepEqual(findBinary({ platform: 'win32', env: {}, modelsDir, componentsDir }),null);
    const outside=join(componentsDir,'llama-server.exe');writeFileSync(outside,'outside fixture');
    for(const folder of ['..','.']){
      writeFileSync(join(generic,'active.json'),JSON.stringify({folder}));
      assert.equal(findBinary({ platform: 'win32', env: {}, modelsDir, componentsDir }), null,`unsafe active pointer ${folder} is ignored`);
    }
    assert.equal(findBinary({ platform: 'win32', env: { STRATA_LLAMA_SERVER: nested }, modelsDir, componentsDir }), null);
  } finally { rmSync(base, { recursive: true, force: true }); }
});

test('missing binary and ordinary executable never claim Spin8 or verified Bonsai', async () => {
  const missing = await probeBackendCapabilities(join(tmpdir(), 'strata-no-such-server.exe'));
  assert.equal(missing.available, false);
  assert.equal(missing.code, 'BACKEND_UNAVAILABLE');
  const ordinary = await probeBackendCapabilities(process.execPath);
  assert.equal(ordinary.available, true);
  assert.equal(ordinary.bonsai.available, false);
  assert.deepEqual(ordinary.spin8.modes, ['off']);
  assert.equal(ordinary.spin8.available, false);
});

test('probing Automatic ignores a generic environment executable and preserves lifecycle state', async () => {
  const prior = process.env.STRATA_LLAMA_SERVER;
  const before = status();
  try {
    process.env.STRATA_LLAMA_SERVER = process.execPath;
    const result = await probeCapabilities({ manifestPath: null });
    assert.equal(result.available, false);
    assert.deepEqual(status(), before);
  } finally {
    if (prior === undefined) delete process.env.STRATA_LLAMA_SERVER;
    else process.env.STRATA_LLAMA_SERVER = prior;
  }
});

test('invalid provenance and traversal manifests do not grant Bonsai capabilities', async () => {
  const base = mkdtempSync(join(tmpdir(), 'strata-bonsai-manifest-'));
  try {
    const manifestPath = join(base, 'bonsai-runtime.json');
    const manifest = { schemaVersion: 1, backend: 'prism', source: { ...BONSAI_SOURCES.prism, dirty: false },
      build: { status: 'success' }, binary: { sha256: sha(process.execPath) },
      files: [{ path: '../outside.dll', sha256: '0'.repeat(64) }],
      spin8: { available: true, modes: SPIN8_MODES } };
    writeFileSync(manifestPath, JSON.stringify(manifest));
    let result = await probeBackendCapabilities(process.execPath, { manifestPath });
    assert.equal(result.bonsai.code, 'BONSAI_MANIFEST_INVALID');
    assert.equal(result.spin8.available, false, 'a manifest cannot enable a graph hook');
    manifest.source.dirty = true;
    writeFileSync(manifestPath, JSON.stringify(manifest));
    result = await probeBackendCapabilities(process.execPath, { manifestPath });
    assert.equal(result.bonsai.available, false);
  } finally { rmSync(base, { recursive: true, force: true }); }
});
