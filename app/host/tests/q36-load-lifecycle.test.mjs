/** T03 lifecycle: refuse unsupported Qwen3.6 settings before spawn, and do not mutate a failed start. */
import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, readFileSync } from 'node:fs';
import { join, basename } from 'node:path';
import { tmpdir } from 'node:os';
import { createNativeControl } from '../server/native-control.mjs';
import { finishJob, nativeFixture } from './helpers/native-fixture.mjs';
import { admitQ36LegacyPhysical, prepareQ36Launch, Q36_MAPPED_PACK_DIGESTS } from '../server/q36-run-contract.mjs';

const isWin = process.platform === 'win32';

function packagedQ36Fixture({ parallel = 4 } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-q36-life-'));
  const install = join(root, 'install');
  const data = join(root, 'data');
  mkdirSync(install, { recursive: true });
  mkdirSync(join(root, 'control', 'strata_control'), { recursive: true });
  writeFileSync(join(root, 'control', 'strata_control', 'runtime.py'), '');
  writeFileSync(join(root, 'control', 'strata_control', 'management.py'), '        wait = _seconds(wait, 0, 600, "start wait")\n');
  writeFileSync(join(root, 'control', 'strata_control', 'journal.py'), '        ranges = {"wait_s": (0, 600) if kind == "runtime.start" else\n');
  const earlierInstallEnv = process.env.STRATA_INSTALL_DIR;
  process.env.STRATA_INSTALL_DIR = install;
  const modelId = 'q36_ud';
  const path = join(install, `strata-${modelId}.json`);
  /* Name keeps isQwen36Config via arch, but omits UD-Q4_K_XL so the measured 18.75 GiB
     physical gate stays unknown and does not refuse before Control start. */
  writeFileSync(path, JSON.stringify({
    exe: 'component:strata-engine',
    arch: 'qwen35moe',
    tokenizer: join(install, 'tok'),
    args: ['--max-context', '65536', '--prefill', '2048', '--mtp', '2'],
    model_name: 'Qwen3.6-35B-A3B',
    parallel,
    port: 8080,
  }));
  const calls = [];
  let startFailure = null;
  let startFailureLeft = Infinity;
  const control = {
    control: { dir: join(root, 'control'), python: join(root, 'control', 'python', isWin ? 'python.exe' : 'python3') },
    version: '0.1.45',
    async installedModels() {
      return {
        install_dir: install,
        models: [{
          id: modelId,
          name: 'Qwen3.6-35B-A3B',
          source_config: path,
          component_status: 'present',
          context_tokens: 65536,
          backend: 'cuda',
          model_name: 'Qwen3.6-35B-A3B',
        }],
      };
    },
    async engine() {
      return {
        usable: true,
        backend: 'cuda',
        components_dir: 'E:\\c',
        engine: { version: '0.4.0', platform: 'windows-x86_64', backend: 'cuda', exe: 'E:\\engine\\strata.exe', root: 'E:\\engine' },
      };
    },
    async saveProfile(name, configPath) {
      calls.push(['saveProfile', name, basename(configPath), JSON.parse(readFileSync(configPath, 'utf8'))]);
      return {};
    },
    async start(name, options) {
      calls.push(['start', name, options?.waitSeconds]);
      if (startFailure && startFailureLeft-- > 0) {
        const error = new Error(startFailure.message);
        error.output = startFailure.output;
        throw error;
      }
      return { ok: true, state: 'ready' };
    },
    async stop(name) {
      calls.push(['stop', name]);
      return { ok: true, state: 'stopped' };
    },
  };
  /* 48 GB VRAM keeps the packaged default above one request slot so a non-Qwen family would dropParallel. */
  const hardware = { ram_gb: 32, cpu: { name: 'cpu', cores: 12 }, gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 5060 Ti', vram_gb: 48, usable: true }] };
  const catalog = {
    families: [{
      family: 'qwen',
      title: 'Qwen',
      images: false,
      sizes: [{ id: modelId, model: 'UD-Q4_K_XL', download_gb: 20, ram_needed_gb: 32, about: 'q36', on_this_pc: 'fits', installed: true, images: false }],
    }],
    contexts: [65536],
    recommendation: { family: 'qwen', model: 'UD-Q4_K_XL', context: 65536 },
  };
  const spawnProcess = (command, args) => {
    if (!String(args[0]).endsWith('strata_mcp.py')) throw new Error('unexpected process');
    const child = new EventEmitter();
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    child.exitCode = null;
    child.kill = () => {
      child.exitCode = 0;
      queueMicrotask(() => child.emit('exit', 0));
      return true;
    };
    child.stdin = {
      write(line) {
        const message = JSON.parse(line);
        if (!message.id) return;
        let result;
        if (message.method === 'initialize') result = { protocolVersion: '2024-11-05' };
        else if (message.method === 'tools/list') result = { tools: [{ name: 'strata_models' }, { name: 'strata_status' }] };
        else if (message.params?.name === 'strata_models') result = { structuredContent: catalog };
        else result = { structuredContent: { hardware, recommendation: catalog.recommendation, install: null, running: [], summary: 'x' } };
        queueMicrotask(() => child.stdout.emit('data', Buffer.from(`${JSON.stringify({ jsonrpc: '2.0', id: message.id, result })}\n`)));
      },
    };
    return child;
  };
  const requests = [];
  const fetcher = async () => Response.json({ status: 'ok' });
  const native = createNativeControl({ control, directory: data, spawnProcess, fetcher, requests: () => requests, engineLogPollMs: 20 });
  return {
    native,
    calls,
    path,
    modelId,
    failStart: (value, times = Infinity) => {
      startFailure = value;
      startFailureLeft = times;
    },
    cleanup: () => {
      if (earlierInstallEnv === undefined) delete process.env.STRATA_INSTALL_DIR;
      else process.env.STRATA_INSTALL_DIR = earlierInstallEnv;
      rmSync(root, { recursive: true, force: true });
    },
  };
}

const finish = async (native, id) => {
  for (let i = 0; i < 400; i++) {
    const job = native.state().jobs.find(j => j.id === id);
    if (job.state !== 'running') return job;
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  throw new Error('job did not finish');
};

test('private-cache flags are forwarded to the native server instead of blocking its start', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const path = join(f.repo, 'strata-iq3_s.json');
    const config = JSON.parse(readFileSync(path, 'utf8'));
    config.arch = 'qwen35moe';
    config.model_name = 'Qwen3.6-35B-A3B';
    config.args = ['--max-context', '65536', '--prompt-cache', '0', '--conversation-cache-mib', '0'];
    writeFileSync(path, JSON.stringify(config));
    const loaded = await finishJob(f.control, f.control.operation('load', { model: 'iq3_s' }).id);
    assert.equal(loaded.state, 'done', loaded.error);
    const starts = f.calls.filter(call => call.args?.some(arg => String(arg).endsWith('server.py')));
    assert.equal(starts.length, 1);
    const launched = JSON.parse(readFileSync(starts[0].args[starts[0].args.indexOf('--config') + 1], 'utf8'));
    for (const flag of ['--prompt-cache', '--conversation-cache-mib']) assert.equal(launched.args[launched.args.indexOf(flag) + 1], '0');
  } finally {
    await f.cleanup();
  }
});

test('mapped-pack skips the measured 18.75 GiB legacy physical gate; legacy still refuses tight free RAM', () => {
  const tight = admitQ36LegacyPhysical({
    modelName: 'Qwen3.6-35B-A3B-UD-Q4_K_XL',
    availablePhysicalBytes: 12n << 30n,
    ramBytes: 32n << 30n,
  });
  assert.equal(tight.decision, 'rejected');
  const digest = [...Q36_MAPPED_PACK_DIGESTS][0];
  const mapped = prepareQ36Launch({
    args: ['--max-context', '65536', '--prefill', '2048', '--mtp', '2'],
    binarySha256: digest,
    requested: { expertStoreMode: 'mapped-pack' },
  });
  assert.equal(mapped.ok, true);
  assert.equal(mapped.contract.effective.expertStoreMode, 'mapped-pack');
  assert.equal(mapped.contract.effective.persistentKv, 'fp32');
  assert.equal(mapped.contract.effective.context, 65536);
  assert.equal(mapped.contract.effective.mtp, 2);
});

test('packaged Hakua memory starts Qwen3.6 with scoped caching for runtime negotiation', async () => {
  const earlier = process.env.STRATA_HAKUA_CHATBOT;
  delete process.env.STRATA_HAKUA_CHATBOT;
  const f = packagedQ36Fixture({ parallel: 1 });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: f.modelId }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.calls.filter(call => call[0] === 'start').length, 1);
    const launched = f.calls.find(call => call[0] === 'saveProfile')[3];
    assert.equal(launched.hakua_harness.enabled, true);
    assert.ok(launched.args.includes('--private-conversation-cache'));
    for (const flag of ['--prompt-cache', '--conversation-cache-mib']) assert.equal(launched.args[launched.args.indexOf(flag) + 1], '0');
  } finally {
    await f.native.close();
    f.cleanup();
    if (earlier === undefined) delete process.env.STRATA_HAKUA_CHATBOT;
    else process.env.STRATA_HAKUA_CHATBOT = earlier;
  }
});

test('packaged Qwen3.6 does not get a parallel-drop second try after the first start fails', async () => {
  const earlier = process.env.STRATA_HAKUA_CHATBOT;
  process.env.STRATA_HAKUA_CHATBOT = '0';
  const f = packagedQ36Fixture({ parallel: 4 });
  try {
    await f.native.refresh(true);
    f.failStart({ message: 'server_exited', output: { log_tail: 'something else went wrong' } });
    const job = await finish(f.native, f.native.operation('load', { model: f.modelId }).id);
    assert.equal(job.state, 'error', job.error);
    assert.equal(f.calls.filter(call => call[0] === 'start').length, 1, 'Qwen3.6 keeps the first stop; parallel is not mutated for a second try');
    assert.doesNotMatch(String(job.details?.join(' ') ?? ''), /Second try/);
    const saved = JSON.parse(readFileSync(f.path, 'utf8'));
    assert.equal(saved.parallel, 4, 'the model file keeps the asked parallel; a failed start does not rewrite it');
  } finally {
    await f.native.close();
    f.cleanup();
    if (earlier === undefined) delete process.env.STRATA_HAKUA_CHATBOT;
    else process.env.STRATA_HAKUA_CHATBOT = earlier;
  }
});
