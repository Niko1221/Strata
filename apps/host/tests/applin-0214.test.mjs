import './helpers/isolate-profile.mjs';   // a test run never writes the owner's real Strata profile
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createControl, ControlError } from '../server/control-runtime.mjs';
import { plainUpdateError, plainEngineError, UPDATE_WAITING_FOR_LOCK } from '../server/update-messages.mjs';
import { createNativeControl } from '../server/native-control.mjs';
import { checkComponentUpdates, engineFailedBeforeText, engineFirstRun } from '../server/updates.mjs';

// Lane APP-LIN (0.2.14): the update and first-run fixes of the 0.2.13 GUI tests.

const LOCK = 'another update is already running for this component folder';

function fixture({ update, cfgExtra = {}, reload } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-applin-'));
  const install = join(root, 'install'), data = join(root, 'data');
  mkdirSync(install, { recursive: true });
  process.env.STRATA_INSTALL_DIR = install;
  const config = join(install, 'strata-iq2_xs.json');
  mkdirSync(join(root, 'control', 'data'), { recursive: true });
  writeFileSync(join(root, 'control', 'data', 'expert-profile.bin'), 'x');
  writeFileSync(config, JSON.stringify({ exe: 'component:strata-engine', args: ['--max-context', '32768'], model_name: 'Qwen', port: 8080, ...cfgExtra }));
  const calls = [];
  const control = {
    control: { dir: join(root, 'control') }, version: '0.1.60',
    async installedModels() { return { install_dir: install, models: [{ id: 'iq2_xs', name: 'Qwen', source_config: config, component_status: 'present', context_tokens: 32768, backend: 'cuda' }] }; },
    async engine() { return { usable: true, backend: 'cuda', components_dir: '/c', engine: { version: '0.1.60', platform: 'linux-x86_64', backend: 'cuda', exe: 'e', root: 'r' } }; },
    async saveProfile() { return {}; }, async start() { return { ok: true, state: 'ready' }; }, async stop() { return { ok: true }; },
    ...(reload ? { reload() { calls.push(['reload']); return reload(this); } } : {}),
    async update(options) {
      calls.push(['update', options.engine, options.control, options.force === true]);
      await update?.(options, calls.length);
      return { ok: true, results: [{ component: 'engine', status: 'updated', installed_after: { version: '0.1.61' } }] };
    },
  };
  const native = createNativeControl({ control, directory: data, lockWaitMs: 300, lockPollMs: 20, spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({}), requests: () => [] });
  return { native, calls, root, config, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
const finish = async (native, id) => { for (let i = 0; i < 400; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };

// ------------------------------------------------------------------ item 4: exit 7 is two things

test('exit 7 from the update lock is "an update is already running", not "a model is running"', () => {
  for (const error of [Object.assign(new Error(LOCK), { exitCode: 7 }), Object.assign(new Error(LOCK), { exitCode: 7, code: 'engine_serving' }), new ControlError(LOCK, { exitCode: 7, code: 'update_busy' })]) {
    const plain = plainUpdateError(error, 'engine', 'Continue');
    assert.match(plain.message, /An update is already running/);
    assert.doesNotMatch(plain.message, /model is running/);
  }
  // a real serving refusal keeps its words
  const serving = plainUpdateError(Object.assign(new Error('app-1 is serving.'), { exitCode: 7, code: 'engine_serving' }));
  assert.match(serving.message, /A model is running/);
  assert.match(plainEngineError(Object.assign(new Error(LOCK), { exitCode: 7 })).message, /An update is running/);
  assert.match(plainEngineError(Object.assign(new Error('x is serving'), { exitCode: 7 })).message, /A model is running/);
});

test('createControl tells the lock from a serving engine by the text, for both failure shapes', async () => {
  const lockedStep = createControl({ control: { dir: '/c', python: 'p', cli: 'c' }, run: async () => ({ code: 7, json: { ok: false, results: [{ component: 'engine', status: 'failed', error: LOCK, exit_code: 7 }] }, stderr: '' }) });
  await assert.rejects(lockedStep.update({}), e => e.code === 'update_busy' && e.exitCode === 7);
  const lockedWhole = createControl({ control: { dir: '/c', python: 'p', cli: 'c' }, run: async () => ({ code: 7, json: { ok: false, error: { message: LOCK } }, stderr: '' }) });
  await assert.rejects(lockedWhole.update({}), e => e.code === 'update_busy');
  await assert.rejects(lockedWhole.updateCheck({}), e => e.code === 'update_busy');
  const serving = createControl({ control: { dir: '/c', python: 'p', cli: 'c' }, run: async () => ({ code: 7, json: { ok: false, results: [{ component: 'engine', status: 'failed', error: 'app-1 is serving.', exit_code: 7 }] }, stderr: '' }) });
  await assert.rejects(serving.update({}), e => e.code === 'engine_serving');
});

test('the first-run Download waits for the lock the start-up check holds, then installs', async () => {
  const f = fixture({ update: async (_o, n) => { if (n < 3) throw new ControlError(LOCK, { exitCode: 7, code: 'update_busy' }); } });
  try {
    const started = f.native.operation('update-engine', { firstRun: true });
    let sawWaiting = false;
    for (let i = 0; i < 50 && !sawWaiting; i++) { sawWaiting = f.native.state().jobs.find(j => j.id === started.id)?.summary === UPDATE_WAITING_FOR_LOCK; await new Promise(r => setTimeout(r, 5)); }
    const job = await finish(f.native, started.id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.calls.length, 3, 'tried again while the lock was held');
    assert.ok(sawWaiting, 'the job said it was waiting');
  } finally { await f.native.close(); f.cleanup(); }
});

test('a lock that stays held ends with the lock words, after the wait', async () => {
  const f = fixture({ update: async () => { throw new ControlError(LOCK, { exitCode: 7, code: 'update_busy' }); } });
  try {
    const job = await finish(f.native, f.native.operation('update-engine', { firstRun: true }).id);
    assert.equal(job.state, 'error');
    assert.match(job.error, /An update is already running/);
    assert.doesNotMatch(job.error, /model is running/);
    assert.ok(f.calls.length >= 2);
  } finally { await f.native.close(); f.cleanup(); }
});

test('a serving refusal is not retried', async () => {
  const f = fixture({ update: async () => { throw new ControlError('app-1 is serving.', { exitCode: 7, code: 'engine_serving' }); } });
  try {
    const job = await finish(f.native, f.native.operation('update-engine', {}).id);
    assert.equal(job.state, 'error');
    assert.equal(f.calls.length, 1);
    assert.match(job.error, /A model is running/);
  } finally { await f.native.close(); f.cleanup(); }
});

