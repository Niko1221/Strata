/** Real supervisor/profile flow; only the child and its local HTTP API are fixtures. */
import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import childProcess from 'node:child_process';
import { syncBuiltinESMExports } from 'node:module';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { createServer } from 'node:http';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { supervisorLocalInstance, mergeAgentLocalInstances, supervisorIsReady, agentSkipReason } from '../server/agent-local.mjs';
import { buildProfile } from '../server/agent/profile.mjs';

async function fixture(t) {
  const home = mkdtempSync(join(tmpdir(), 'supervisor-context-'));
  const modelPath = join(home, 'fixture.gguf');
  writeFileSync(modelPath, 'synthetic model; no weights');
  const before = { home: process.env.STRATA_HOME, binary: process.env.STRATA_LLAMA_SERVER };
  process.env.STRATA_HOME = home;
  process.env.STRATA_LLAMA_SERVER = process.execPath;
  const children = [], requests = [];
  let props = { model_path: modelPath, default_generation_settings: { n_ctx: 32768 }, total_slots: 1 };
  let pending = null;
  let health = null, pendingHealth = null;
  const server = createServer((req, res) => {
    requests.push({ path: req.url, authorization: req.headers.authorization });
    const reply = value => { res.setHeader('content-type', 'application/json'); res.end(JSON.stringify(value)); };
    if (req.url === '/health') {
      if (health) { const respond = health; pendingHealth = () => respond(res, reply); return; }
      return reply({ status: 'ok' });
    }
    if (req.url === '/props') {
      if (typeof props === 'function') { const respond = props; pending = () => respond(res, reply); return; }
      return reply(props);
    }
    if (req.url === '/metrics') return res.end('llamacpp:prompt_tokens_total 0\n');
    res.statusCode = 404; res.end();
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const port = server.address().port;
  t.mock.method(childProcess, 'spawn', () => {
    const proc = Object.assign(new EventEmitter(), { pid: 424240 + children.length, exitCode: null, signalCode: null, stdout: new PassThrough(), stderr: new PassThrough() });
    proc.kill = signal => { proc.signalCode = signal; proc.emit('exit', null, signal); return true; };
    children.push(proc); return proc;
  });
  syncBuiltinESMExports();
  const supervisor = await import('../server/supervisor.mjs');
  t.after(async () => {
    await supervisor.stop();
    server.closeAllConnections(); await new Promise(r => server.close(r));
    for (const proc of children) { proc.stdout.destroy(); proc.stderr.destroy(); }
    t.mock.restoreAll(); syncBuiltinESMExports();
    if (before.home === undefined) delete process.env.STRATA_HOME; else process.env.STRATA_HOME = before.home;
    if (before.binary === undefined) delete process.env.STRATA_LLAMA_SERVER; else process.env.STRATA_LLAMA_SERVER = before.binary;
    assert.equal(resolve(home).startsWith(resolve(tmpdir()) + (process.platform === 'win32' ? '\\' : '/')), true);
    rmSync(home, { recursive: true, force: true });
  });
  const start = (contextLength = '32k', parallel = 1) => supervisor.start({ modelId: 'gguf-advertised-262144', modelPath, port,
    config: { beginner: { contextLength }, expert: [{ id: 'server.parallel', value: parallel }], serving: { apiKey: 'fixture-context-key' } },
    hardware: { cpu: { cores: 4 } }, runObservation: false });
  return { supervisor, start, requests, children, modelPath, setProps(value) { props = value; },
    setHealth(value) { health = value; },
    async waitForHealth() { for (let i = 0; i < 100 && !pendingHealth; i++) await new Promise(r => setTimeout(r, 5)); assert.ok(pendingHealth); },
    replyHealth() { const reply = pendingHealth; pendingHealth = null; reply(); },
    async waitForProps() { for (let i = 0; i < 100 && !pending; i++) await new Promise(r => setTimeout(r, 5)); assert.ok(pending); },
    replyPending() { const reply = pending; pending = null; reply(); } };
}

function profileFor(engine) {
  // Deliberately provide the old metadata option: it must have no influence.
  const row = supervisorLocalInstance(engine, { context: 262144, authenticationKey: () => 'fixture-context-key' });
  assert.ok(row);
  const profile = buildProfile({ local: [row], proxyBase: 'http://127.0.0.1:9999/proxy' });
  return { row, profile, patch: JSON.parse(profile.patch.replace(/^#.*\n/, '')) };
}

test('effective 32k reaches all profile budgets despite advertised 262144; 131072 and per-slot parallel remain exact', async t => {
  const f = await fixture(t);
  for (const [contextLength, parallel, capacity, output, headroom] of [['32k', 1, 32768, 9830, 2867], ['128k', 1, 131072, 32768, 12288], ['recommended', 1, 131072, 32768, 12288], ['128k', 4, 32768, 9830, 2867], ['128k', 1, 16384, 4915, 1433]]) {
    f.setProps({ model_path: f.modelPath, default_generation_settings: { n_ctx: capacity }, total_slots: parallel });
    const engine = await f.start(contextLength, parallel);
    assert.equal(engine.contextCapacity, capacity);
    assert.ok(!JSON.stringify(engine).includes('fixture-context-key'), 'public status never exposes the launch key');
    const { row, profile, patch } = profileFor(engine);
    assert.equal(row.context, capacity);
    assert.equal(row.key, 'fixture-context-key');
    assert.ok(profile.routes.every(r => r.contextWindow === capacity));
    const serialized = JSON.stringify(patch);
    assert.ok(!serialized.includes('fixture-context-key'));
    const models = [];
    const budgets = [];
    function walk(value) {
      if (!value || typeof value !== 'object') return;
      if (Object.hasOwn(value, 'contextWindow')) models.push(value);
      if (Object.hasOwn(value, 'headroomTokens')) budgets.push(value);
      for (const item of Object.values(value)) walk(item);
    }
    walk(patch);
    assert.ok(models.length > 1, 'ordinary and response-shaped providers are covered');
    for (const model of models) { assert.equal(model.contextWindow, capacity); assert.equal(model.maxTokens, output); }
    assert.ok(budgets.length > 1);
    assert.ok(budgets.every(b => b.headroomTokens === headroom));
    const count = f.requests.filter(r => r.path === '/props').length;
    assert.equal((await f.start(contextLength, parallel)).contextCapacity, capacity);
    for (let i = 0; i < 10; i++) profileFor(f.supervisor.status());
    await f.supervisor.pollMetrics();
    assert.equal(f.requests.filter(r => r.path === '/props').length, count, 'no chat/status/metrics re-probe');
    await f.supervisor.stop();
    assert.equal(f.supervisor.status().contextCapacity, null);
  }
  assert.equal(f.requests.filter(r => r.path === '/props').length, 5);
  assert.ok(f.requests.filter(r => r.path === '/props').every(r => r.authorization === 'Bearer fixture-context-key'));
});

test('missing, malformed, stale-model properties fail closed for the agent but keep the engine ready for plain chat', async t => {
  const f = await fixture(t);
  for (const props of [{}, { model_path: f.modelPath }, { default_generation_settings: { n_ctx: 32768 } }, ...[null, 0, -1, 1.25, '32768', Number.MAX_SAFE_INTEGER + 1].map(n_ctx => ({ model_path: f.modelPath, default_generation_settings: { n_ctx } })),
    { model_path: join(tmpdir(), 'previous-model.gguf'), default_generation_settings: { n_ctx: 262144 } }]) {
    f.setProps(props);
    const engine = await f.start();
    assert.equal(supervisorIsReady(engine), true);
    assert.equal(engine.contextCapacity, null);
    assert.equal(supervisorLocalInstance(engine, { context: 262144 }), null);
    assert.match(agentSkipReason({ chatId: 'fixture', supervisorContextUnavailable: true }), /plain chat/);
    assert.deepEqual(mergeAgentLocalInstances([{ id: 'native', context: 131072 }], null), [{ id: 'native', context: 131072 }]);
    await f.supervisor.stop();
  }
});

test('a stopped child cannot publish its late props response for a replacement with the same model id', async t => {
  const f = await fixture(t);
  f.setProps((res, reply) => reply({ model_path: f.modelPath, default_generation_settings: { n_ctx: 262144 } }));
  const oldStart = f.start();
  const oldRejected = assert.rejects(oldStart, /stopped while checking/);
  await f.waitForProps();
  await f.supervisor.stop();
  f.setProps({ model_path: f.modelPath, default_generation_settings: { n_ctx: 32768 }, total_slots: 1 });
  const replacement = await f.start();
  // Complete the old HTTP response only after the replacement is ready.
  f.replyPending();
  await oldRejected;
  assert.equal(replacement.contextCapacity, 32768);
  assert.equal(f.supervisor.status().pid, f.children[1].pid);
  f.children[0].emit('exit', 0, null);
  assert.equal(f.supervisor.status().contextCapacity, 32768, 'late old exit cannot clear current capacity');
});

test('HTTP errors, invalid JSON and a timed out props probe leave plain chat available', async t => {
  const f = await fixture(t);
  for (const respond of [res => { res.statusCode = 401; res.end(); }, res => res.end('{'), () => {}]) {
    f.setProps(respond);
    const start = f.start();
    await f.waitForProps();
    f.replyPending();
    const engine = await start;
    assert.equal(engine.status, 'running');
    assert.equal(engine.contextCapacity, null);
    assert.equal(supervisorLocalInstance(engine), null);
    assert.ok(f.supervisor.logs().some(entry => /context size unavailable/.test(entry.line)));
    await f.supervisor.stop();
  }
});

test('stale health success and transport rejection cannot stop or overwrite a replacement', async t => {
  const f = await fixture(t);
  for (const respond of [(_res, reply) => reply({ status: 'ok' }), res => res.destroy()]) {
    f.setHealth(respond);
    const oldStart = f.start();
    const rejected = assert.rejects(oldStart, /Engine exited early with signal SIGTERM/);
    await f.waitForHealth();
    await f.supervisor.stop();
    f.setHealth(null);
    const replacement = await f.start();
    f.replyHealth();
    await rejected;
    const current = f.supervisor.status();
    assert.equal(current.status, 'running');
    assert.equal(current.pid, replacement.pid);
    assert.equal(current.contextCapacity, 32768);
    assert.equal(current.error, null);
    assert.equal(f.children.at(-1).signalCode, null, 'old health cleanup never kills the replacement');
    await f.supervisor.stop();
  }
});

test('small measured capacities cannot consume all prompt room with unchanged profile reserves', async t => {
  const f = await fixture(t);
  for (const capacity of [1, 2048, 4096, 4608]) {
    f.setProps({ model_path: f.modelPath, default_generation_settings: { n_ctx: capacity }, total_slots: 1 });
    const engine = await f.start();
    assert.equal(engine.status, 'running');
    assert.equal(engine.contextCapacity, capacity, 'measured status remains truthful');
    assert.equal(supervisorLocalInstance(engine), null);
    const unsafeProfile = buildProfile({ local: [{ id: 'arithmetic-only', modelId: 'local', baseUrl: 'http://fixture', context: capacity }] });
    const entries = JSON.parse(unsafeProfile.patch.replace(/^#.*\n/, ''));
    const model = entries.find(entry => entry.id === 'llm-pi-ai').config.providers['strata-local'].models[0];
    assert.equal(model.contextWindow, capacity);
    assert.equal(model.maxTokens, 4096, 'existing output floor is preserved');
    assert.match(unsafeProfile.patch, /"headroomTokens": 512/);
    assert.ok(capacity - model.maxTokens - 512 <= 0);
    assert.match(agentSkipReason({ chatId: 'fixture', supervisorContextUnavailable: true }), /too small.*plain chat/);
    await f.supervisor.stop();
  }
});
