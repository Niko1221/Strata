// Strata Dev (docs/DEV_BUILD.md): the llama.cpp runner (server/dev-llama.mjs), run against a mock llama-server (fixtures/mock-llama-server.mjs).
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import net from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { FORBIDDEN_PORTS, aliasFor, createLlamaRunner, devLlamaApi, llamaArgs, llamaSettings, settingsProblem } from '../server/dev-llama.mjs';

const MOCK = fileURLToPath(new URL('./fixtures/mock-llama-server.mjs', import.meta.url));
const freePort = () => new Promise(resolve => { const s = net.createServer(); s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => resolve(port)); }); });

/** A runner whose "llama-server" is the mock (node + the fixture), with a fake model-providers store. */
async function setup({ mode = 'ok', healthTimeoutMs = 10_000, port } = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'strata-dev-llama-'));
  const gguf = join(dir, 'Tiny-Model-Q4_K_M.gguf');
  writeFileSync(gguf, 'GGUF');
  const settings = llamaSettings({ server: MOCK, port: port ?? await freePort(), healthTimeoutMs });
  const saved = [];
  let connections = [];
  const providers = {
    list: () => ({ catalog: [], connections }),
    save: body => { saved.push(body); const c = { ...body, id: body.id ?? 'conn-1' }; connections = [...connections.filter(x => x.id !== c.id), c]; return { catalog: [], connections }; },
  };
  const lines = [];
  const children = [];
  const runner = createLlamaRunner({
    getSettings: () => settings, providers, log: (line, level = 'info') => lines.push(`${level}: ${line}`),
    spawnImpl: (cmd, args, opts) => { const c = spawn(process.execPath, [cmd, ...args], { ...opts, env: { ...process.env, MOCK_LLAMA: mode } }); children.push(c); return c; },
  });
  return { dir, gguf, settings, saved, lines, runner, children, cleanup: async () => { await runner.stop(); for (const c of children) try { c.kill(); } catch { /* gone */ } rmSync(dir, { recursive: true, force: true }); } };
}

test('settings: defaults, the path check and the ports that belong to Strata', () => {
  assert.deepEqual(llamaSettings(undefined), { server: null, port: 8095, args: [], context: 8192, healthTimeoutMs: 180000 });
  assert.equal(llamaSettings({ port: 80, args: ['-ngl', 99, {}], context: 100 }).port, 8095);
  assert.deepEqual(llamaSettings({ args: ['-ngl', 99, {}] }).args, ['-ngl', '99']);
  assert.match(settingsProblem(llamaSettings({})), /No llama-server is set/);
  assert.match(settingsProblem(llamaSettings({ server: 'C:/nowhere/llama-server.exe' })), /is not at/);
  assert.match(settingsProblem(llamaSettings({ server: tmpdir() })), /is a folder/);
  for (const port of [8080, 8081, 8090, 8791, 8792, 47831, 47832]) {
    assert.ok(FORBIDDEN_PORTS.includes(port));
    assert.match(settingsProblem(llamaSettings({ server: MOCK, port })), /belongs to Strata/);
  }
  assert.equal(settingsProblem(llamaSettings({ server: MOCK })), null);
  assert.deepEqual(llamaArgs({ gguf: 'm.gguf', alias: 'm', port: 8095, context: 4096, extra: ['-ngl', '99'] }), ['-m', 'm.gguf', '--host', '127.0.0.1', '--port', '8095', '-c', '4096', '--alias', 'm', '-ngl', '99']);
  assert.equal(aliasFor('C:\\m\\Some Model-Q4_K_M-00001-of-00002.gguf'), 'some-model-q4_k_m');
});

test('start: waits for /health, saves one API connection marked "läuft über llama.cpp", stop ends the process', async () => {
  const t = await setup();
  try {
    const before = t.runner.state();
    assert.equal(before.status, 'stopped');
    assert.equal(before.available, true);
    const started = await t.runner.start({ gguf: t.gguf, label: 'Tiny', context: 4096 });
    assert.equal(started.status, 'running', started.error ?? '');
    assert.equal(started.model, 'tiny-model-q4_k_m');
    assert.equal(t.saved.length, 1);
    assert.equal(t.saved[0].baseUrl, `http://127.0.0.1:${t.settings.port}/v1`);
    assert.equal(t.saved[0].kind, 'custom');
    assert.deepEqual(t.saved[0].models, [{ id: 'tiny-model-q4_k_m', name: 'Tiny · läuft über llama.cpp', contextMax: 4096 }]);
    const models = await (await fetch(`http://127.0.0.1:${t.settings.port}/v1/models`)).json();
    assert.equal(models.data[0].id, 'tiny-model-q4_k_m');
    // A second start replaces the model and keeps the same connection.
    const again = await t.runner.start({ gguf: t.gguf, label: 'Tiny 2' });
    assert.equal(again.status, 'running');
    assert.equal(t.saved[1].id, 'conn-1');
    const stopped = await t.runner.stop();
    assert.equal(stopped.status, 'stopped');
    assert.ok(t.children.every(c => c.exitCode !== null || c.signalCode !== null), 'every llama-server ended');
    await assert.rejects(fetch(`http://127.0.0.1:${t.settings.port}/health`));
  } finally { await t.cleanup(); }
});

test('a crash while loading is an error with the server\'s last line, never a throw', async () => {
  const t = await setup({ mode: 'crash' });
  try {
    const r = await t.runner.start({ gguf: t.gguf });
    assert.equal(r.status, 'error');
    assert.match(r.error, /llama-server ended \(code 1\): llama_model_load: error loading model/);
    assert.ok(t.lines.some(l => l.startsWith('error:')));
    assert.equal(t.saved.length, 0, 'no connection for a server that never ran');
  } finally { await t.cleanup(); }
});

test('no answer in time: the process is ended and the state says so', async () => {
  const t = await setup({ mode: 'silent', healthTimeoutMs: 1500 });
  try {
    const r = await t.runner.start({ gguf: t.gguf });
    assert.equal(r.status, 'error');
    assert.match(r.error, /did not answer on port \d+ within 2 s|within 1 s/);
    await new Promise(res => setTimeout(res, 300));
    assert.ok(t.children[0].exitCode !== null || t.children[0].signalCode !== null);
  } finally { await t.cleanup(); }
});

test('a taken port, a missing .gguf and a missing llama-server are plain errors', async () => {
  const port = await freePort();
  const blocker = net.createServer().listen(port, '127.0.0.1');
  await new Promise(r => blocker.once('listening', r));
  const t = await setup({ port });
  try {
    assert.match((await t.runner.start({ gguf: t.gguf })).error, new RegExp(`Port ${port} is in use`));
    assert.match((await t.runner.start({ gguf: join(t.dir, 'missing.gguf') })).error, /No \.gguf at/);
    const none = createLlamaRunner({ getSettings: () => llamaSettings({}), providers: { list: () => ({ connections: [] }), save: () => ({}) }, log: () => {} });
    const r = await none.start({ gguf: t.gguf });
    assert.equal(r.status, 'error');
    assert.match(r.error, /No llama-server is set/);
    assert.equal(none.state().available, false);
  } finally { blocker.close(); await t.cleanup(); }
});

test('routes: state, start and stop under /api/dev/llama', async () => {
  const t = await setup();
  const out = [];
  const json = (res, status, body) => out.push([status, body]);
  const error = (res, status, message) => out.push([status, { error: message }]);
  try {
    const call = (method, path, body = {}) => devLlamaApi({}, {}, path, method, new URL(`http://x${path}`), { json, error, readBody: async () => body, runner: t.runner });
    assert.equal(await call('GET', '/api/other'), undefined);
    await call('GET', '/api/dev/llama');
    assert.equal(out.at(-1)[1].status, 'stopped');
    await call('POST', '/api/dev/llama/start', { gguf: t.gguf, label: 'Tiny' });
    assert.equal(out.at(-1)[1].status, 'running');
    await call('POST', '/api/dev/llama/stop');
    assert.equal(out.at(-1)[1].status, 'stopped');
    await call('DELETE', '/api/dev/llama');
    assert.equal(out.at(-1)[0], 404);
  } finally { await t.cleanup(); }
});
