/**
 * F03: a cancelled load must not unload a newer ready instance that reused the
 * same instance id / app-N profile after cancel marked the old job finished.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, readFileSync } from 'node:fs';
import { join, basename } from 'node:path';
import { tmpdir } from 'node:os';
import { createNativeControl } from '../server/native-control.mjs';

function cancelRaceFixture({ persistedJobs } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-load-cancel-'));
  const install = join(root, 'install'), data = join(root, 'data');
  if (persistedJobs) {
    mkdirSync(data, { recursive: true });
    writeFileSync(join(data, 'native-jobs.json'), JSON.stringify(persistedJobs));
  }
  mkdirSync(install, { recursive: true });
  const earlier = process.env.STRATA_INSTALL_DIR;
  process.env.STRATA_INSTALL_DIR = install;
  for (const id of ['iq2_xs', 'iq3_s']) {
    writeFileSync(join(install, `strata-${id}.json`), JSON.stringify({
      exe: 'component:strata-engine', tokenizer: join(install, 'tok'), args: ['--max-context', '32768'], model_name: `Qwen ${id}`, port: 8080,
    }));
  }
  const calls = [];
  const startGates = [];
  let gateStarts = true;
  const control = {
    control: { dir: join(root, 'control'), python: join(root, 'control', 'python', 'python.exe') }, version: '0.1.45',
    async installedModels(options) {
      calls.push(['installedModels', options?.installDir ?? null]);
      return {
        install_dir: install,
        models: ['iq2_xs', 'iq3_s'].map(id => ({ id, name: `Qwen ${id}`, source_config: join(install, `strata-${id}.json`), component_status: 'present', context_tokens: 32768, backend: 'cuda' })),
      };
    },
    async engine() {
      return { usable: true, backend: 'cuda', components_dir: 'E:\\c', engine: { version: '0.1.45', platform: 'windows-x86_64', backend: 'cuda', exe: 'E:\\engine\\strata.exe', root: 'E:\\engine' } };
    },
    async saveProfile(name, path) { calls.push(['saveProfile', name, basename(path)]); return {}; },
    async start(name, options) {
      calls.push(['start', name, options?.waitSeconds]);
      if (gateStarts) {
        await new Promise((resolve) => { startGates.push(resolve); });
      }
      return { ok: true, state: 'ready' };
    },
    async stop(name) { calls.push(['stop', name]); return { ok: true, state: 'stopped' }; },
  };
  const catalog = {
    families: [{ family: 'qwen', title: 'Qwen', about: 'x', images: true, sizes: [
      { id: 'iq2_xs', model: 'IQ2_XS', download_gb: 1, ram_needed_gb: 1, about: 'x', on_this_pc: 'fits', installed: true, images: true },
      { id: 'iq3_s', model: 'IQ3_S', download_gb: 1, ram_needed_gb: 1, about: 'x', on_this_pc: 'fits', installed: true, images: true },
    ] }],
    contexts: [32768], recommendation: { family: 'qwen', model: 'IQ2_XS', context: 32768 },
  };
  const hardware = { ram_gb: 64, cpu: { name: 'cpu', cores: 12 }, gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 5060 Ti', vram_gb: 16, usable: true }] };
  const spawnProcess = (command, args) => {
    if (!String(args[0]).endsWith('strata_mcp.py')) throw new Error('unexpected process ' + command);
    const child = new EventEmitter();
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    child.exitCode = null;
    child.kill = () => { child.exitCode = 0; queueMicrotask(() => child.emit('exit', 0)); return true; };
    child.stdin = {
      write(line) {
        const message = JSON.parse(line);
        if (!message.id) return;
        let result;
        if (message.method === 'initialize') result = { protocolVersion: '2024-11-05' };
        else if (message.method === 'tools/list') result = { tools: ['strata_models', 'strata_status', 'strata_install'].map(name => ({ name })) };
        else if (message.params.name === 'strata_models') result = { structuredContent: catalog };
        else if (message.params.name === 'strata_status') result = { structuredContent: { hardware, recommendation: catalog.recommendation, install: null, running: [], summary: 'x' } };
        else result = { structuredContent: { summary: 'ok' } };
        queueMicrotask(() => child.stdout.emit('data', Buffer.from(JSON.stringify({ jsonrpc: '2.0', id: message.id, result }) + '\n')));
      },
    };
    return child;
  };
  const fetcher = async () => Response.json({ status: 'ok' });
  const native = createNativeControl({ control, directory: data, spawnProcess, fetcher, requests: () => [] });
  return {
    native,
    calls,
    waitForStartGate: async () => {
      for (let i = 0; i < 400; i++) {
        if (startGates.length) return;
        await new Promise(r => setTimeout(r, 10));
      }
      throw new Error('start was never gated');
    },
    releaseOldestStart: () => { const release = startGates.shift(); release?.(); },
    allowImmediateStarts: () => { gateStarts = false; },
    cleanup: async () => {
      await native.close();
      if (earlier === undefined) delete process.env.STRATA_INSTALL_DIR;
      else process.env.STRATA_INSTALL_DIR = earlier;
      rmSync(root, { recursive: true, force: true });
    },
  };
}

async function finish(native, id) {
  for (let i = 0; i < 400; i++) {
    const job = native.state().jobs.find(j => j.id === id);
    if (job.state !== 'running') return job;
    await new Promise(r => setTimeout(r, 10));
  }
  throw new Error('job did not finish');
}

test('a cancellation interrupted by a host restart cannot block the next model load', async () => {
  const old = { id: 'old-cancel', kind: 'load', state: 'cancelling', startedAt: '2026-10-27T14:39:51.239Z', summary: 'Stopping…', model: 'iq2_xs' };
  const completed = { id: 'completed', kind: 'load', state: 'done', startedAt: '2026-10-27T14:35:24.126Z', summary: 'Loaded', model: 'iq3_s' };
  const f = cancelRaceFixture({ persistedJobs: [completed, old] });
  try {
    assert.equal(f.native.state().jobs.find(job => job.id === old.id).state, 'cancelled');
    assert.deepEqual(f.native.state().jobs.find(job => job.id === completed.id), completed);
    assert.equal(f.calls.filter(call => call[0] === 'stop').length, 0, 'recovery must not stop any existing runtime');
    await f.native.refresh(true);
    f.allowImmediateStarts();
    const next = f.native.operation('load', { model: 'iq2_xs', gpus: [0] });
    assert.equal((await finish(f.native, next.id)).state, 'done');
  } finally { await f.cleanup(); }
});

test('F03: cancelled load whose start resolves late must not unload the newer ready instance', async () => {
  const f = cancelRaceFixture();
  try {
    await f.native.refresh(true);

    const first = f.native.operation('load', { model: 'iq2_xs', gpus: [0] });
    await f.waitForStartGate();
    await f.native.cancel(first.id);
    assert.equal(f.native.state().jobs.find(j => j.id === first.id).state, 'cancelled');

    // Next load is allowed immediately after cancel; its start must succeed.
    f.allowImmediateStarts();
    const second = f.native.operation('load', { model: 'iq2_xs', gpus: [0] });
    const done = await finish(f.native, second.id);
    assert.equal(done.state, 'done', done.error);
    const ready = f.native.state().instances.find(i => i.status === 'ready');
    assert.ok(ready, 'second load reached ready');
    const readyId = ready.id;
    const stopsBefore = f.calls.filter(c => c[0] === 'stop').length;

    // Old start finally succeeds; the cancelled job must not tear down the new runtime.
    f.releaseOldestStart();
    await new Promise(r => setTimeout(r, 80));

    const after = f.native.state().instances.find(i => i.id === readyId);
    assert.ok(after, 'ready instance from the second load is still present');
    assert.equal(after.status, 'ready', 'ready instance was not stopped by the cancelled load');
    assert.equal(
      f.calls.filter(c => c[0] === 'stop').length,
      stopsBefore,
      'cancelled load must not stop the newer profile after its own start resolves late',
    );
  } finally {
    await f.cleanup();
  }
});

test('F03b: late-resolving cancelled start must stop its own profile so reload is not blocked', async () => {
  const f = cancelRaceFixture();
  try {
    await f.native.refresh(true);

    const first = f.native.operation('load', { model: 'iq2_xs', gpus: [0] });
    await f.waitForStartGate();
    await f.native.cancel(first.id);
    assert.equal(f.native.state().jobs.find(j => j.id === first.id).state, 'cancelled');

    const stopsBeforeRelease = f.calls.filter(c => c[0] === 'stop').length;
    f.releaseOldestStart();
    await finish(f.native, first.id);
    await new Promise(r => setTimeout(r, 50));

    const stopsAfter = f.calls.filter(c => c[0] === 'stop').length;
    assert.ok(
      stopsAfter > stopsBeforeRelease,
      'cancelled load must stop its profile after a late Control start resolves',
    );

    f.allowImmediateStarts();
    const second = f.native.operation('load', { model: 'iq2_xs', gpus: [0] });
    const done = await finish(f.native, second.id);
    assert.equal(done.state, 'done', done.error ?? done.summary);
    assert.ok(f.native.state().instances.some(i => i.status === 'ready'), 'reload reaches ready');
  } finally {
    await f.cleanup();
  }
});
