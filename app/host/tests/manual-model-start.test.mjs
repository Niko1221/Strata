import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'node:net';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { createLocalModels } from '../server/local-models.mjs';
import { ggufFixture } from './helpers/local-gguf.mjs';
import { ROOMY_HOST } from './helpers/hardware.mjs';

async function freePort() {
  const server = createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  await new Promise(resolve => server.close(resolve));
  return port;
}

for (const saved of [false, true]) test(`opening the real host leaves a ${saved ? 'previously loaded' : 'newly discovered'} local model unloaded`, async t => {
  const home = mkdtempSync(join(tmpdir(), 'strata-manual-start-'));
  const data = join(home, 'data'); mkdirSync(data);
  const model = join(home, 'fixture.gguf'); ggufFixture(model);
  const local = createLocalModels(data);
  const inspected = local.inspect(model); local.add(inspected);
  const id = inspected.variant.id;
  const registry = readFileSync(join(data, 'local-models.json'), 'utf8');
  const state = { current: saved ? id : null, loaded: saved ? id : null,
    beginner: { priority: 'balanced', contextLength: '2k', kvCache: 'FP16', readImages: false, gpuIdle: 'never' }, expert: { 'server.port': await freePort() } };
  writeFileSync(join(data, 'state.json'), JSON.stringify(state));
  const port = await freePort();
  let output = '';
  // A generic environment executable must never start, automatically or by
  // loading an import without an explicitly selected verified adapter.
  const host = spawn(process.execPath, ['server/index.mjs'], {
    cwd: fileURLToPath(new URL('..', import.meta.url)), stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'),
      STRATA_PORT: String(port), STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()),
      STRATA_WORKSPACE: join(home, 'workspace'), STRATA_REAL_ENGINE: '1', STRATA_LLAMA_SERVER: process.execPath, STRATA_CONTROL_DIR: join(home, 'absent-control'),
      STRATA_NO_ACQUIRE: '1', STRATA_PACKAGED: '1', STRATA_TEST_HARDWARE: ROOMY_HOST },
  });
  host.stdout.on('data', d => { output += d; }); host.stderr.on('data', d => { output += d; });
  t.after(async () => {
    if (host.exitCode === null) { const stopped = new Promise(resolve => host.once('exit', resolve)); host.kill(); await stopped; }
    rmSync(home, { recursive: true, force: true });
  });
  const get = async path => (await fetch(`http://127.0.0.1:${port}${path}`)).json();
  let up = false;
  for (let i = 0; i < 150 && !up; i++) { try { await get('/api/health'); up = true; } catch { await delay(100); } }
  assert.ok(up, output);
  await delay(1600); // includes the former 700 ms automatic-load timer
  const library = await get('/api/models');
  assert.equal(library.current, saved ? id : null, 'preserve an explicit prior selection without inventing a new one');
  assert.equal(library.loaded, null);
  assert.equal(library.engine.status, 'stopped', output);
  assert.equal(library.variants.find(v => v.id === id).state, 'installed');
  assert.doesNotMatch(output, /autostart.*loading|bad option: -m/);
  assert.equal(readFileSync(join(data, 'local-models.json'), 'utf8'), registry);
  const response = await fetch(`http://127.0.0.1:${port}/api/models/${id}/load`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}', signal: AbortSignal.timeout(15000),
  });
  const result = await response.json();
  assert.equal(response.status, 409);
  assert.match(result.error, /own engine/, 'an unselected generic backend remains refused on explicit load');
  assert.doesNotMatch(output, /autostart.*loading|bad option: -m/);
});
