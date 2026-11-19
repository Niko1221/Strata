import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer, get } from 'node:http';
import { spawn } from 'node:child_process';
import { EventEmitter, once } from 'node:events';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { EventHub } from '../server/util.mjs';

const root = resolve(import.meta.dirname, '..');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

async function listen(server) {
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  return `http://127.0.0.1:${server.address().port}`;
}

function stream(url) {
  return new Promise((resolve, reject) => {
    const request = get(url, { agent: false }, response => {
      let text = '', ended = false;
      const errors = [];
      response.on('error', error => errors.push(error.code));
      response.on('end', () => { ended = true; });
      const done = new Promise(resolve => response.on('close', () => resolve({
        text, ended, complete: response.complete, aborted: response.aborted, errors,
      })));
      response.on('data', chunk => { text += chunk; });
      response.once('data', () => resolve({ response, done }));
    });
    request.on('error', reject);
  });
}

test('normal shell shutdown completes the real host SSE HTTP message without aborting it', { timeout: 30000 }, async t => {
  const reservation = createServer();
  const base = await listen(reservation);
  await new Promise(resolve => reservation.close(resolve));
  const home = await mkdtemp(join(tmpdir(), 'strata-sse-shutdown-'));
  const secret = 'shutdown-regression-only-'.repeat(3);
  const env = { ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'),
    STRATA_SETTINGS_FILE: join(home, 'settings.json'), STRATA_COMPONENTS_DIR: join(home, 'components'),
    STRATA_WORKSPACE: join(home, 'work'), STRATA_PORT: new URL(base).port,
    STRATA_REAL_ENGINE: '1', STRATA_SHUTDOWN_TOKEN: secret, STRATA_NO_ACQUIRE: '1' };
  delete env.STRATA_CONTROL_DIR;
  delete env.STRATA_PACKAGED;
  const host = spawn(process.execPath, ['server/index.mjs'], { cwd: root, env, windowsHide: true,
    stdio: ['ignore', 'ignore', 'pipe', 'ipc'] });
  const exited = once(host, 'exit');
  let stderr = '';
  host.stderr.on('data', chunk => { stderr += chunk; });
  t.after(async () => {
    if (host.exitCode === null && host.signalCode === null) { host.kill(); await exited; }
    await rm(home, { recursive: true, force: true, maxRetries: 3 });
  });
  let ready = false;
  host.on('message', message => { if (message.type === 'strata-ready') ready = true; });
  for (let i = 0; i < 150 && !ready && host.exitCode === null; i++) await delay(100);
  assert.ok(ready, `host ready: ${stderr.slice(-500)}`);
  const events = await stream(`${base}/api/events`);
  t.after(() => events.response.destroy());
  const shutdown = await fetch(`${base}/api/control/shutdown`, { method: 'POST',
    headers: { 'x-strata-shutdown': secret }, signal: AbortSignal.timeout(5000) });
  assert.equal(shutdown.status, 202);
  assert.deepEqual(await shutdown.json(), { stopping: true });
  const result = await events.done;
  assert.match(result.text, /: connected\n\n/);
  assert.deepEqual({ ended: result.ended, complete: result.complete, aborted: result.aborted, errors: result.errors },
    { ended: true, complete: true, aborted: false, errors: [] });
  assert.deepEqual(await exited, [0, null]);
});

test('hub close drains loopback streams, tolerates a disconnected client and refuses reconnects', { timeout: 5000 }, async t => {
  const hub = new EventHub();
  const responses = [];
  const server = createServer((_req, res) => {
    const original = new Set(['error', 'close', 'finish'].flatMap(name => res.listeners(name)));
    responses.push({ res, original });
    hub.add(res);
  });
  const base = await listen(server);
  t.after(() => { server.close(); server.closeAllConnections(); });
  const disconnected = await stream(base);
  disconnected.response.destroy();
  await disconnected.done;
  // A round trip lets the server observe the earlier disconnect before close.
  const active = await stream(base);
  hub.send('test', { ok: true });
  const closing = hub.close();
  assert.equal(hub.close(), closing, 'repeated close shares the same bounded drain');
  await closing;
  hub.send('late', { ignored: true });
  const result = await active.done;
  assert.equal(result.complete, true);
  assert.equal(result.ended, true);
  assert.equal(result.aborted, false);
  assert.deepEqual(result.errors, []);
  assert.match(result.text, /event: test\ndata: {"ok":true}\n\n/);
  assert.doesNotMatch(result.text, /event: late/);
  assert.equal(hub.size, 0);
  for (const { res, original } of responses) {
    for (const name of ['error', 'close', 'finish']) {
      assert.ok(res.listeners(name).every(listener => original.has(listener)), `no hub ${name} listener remains`);
    }
  }
  const reconnect = await fetch(base);
  assert.equal(reconnect.status, 204);
  assert.equal(await reconnect.text(), '');
  await hub.close();
});

test('hub close stays bounded when a response never finishes and removes its heartbeat and listeners', async t => {
  // A stalled transport deliberately never emits finish/close; real wire completion is tested above.
  const response = new EventEmitter();
  response.writeHead = () => {};
  response.write = () => {};
  let ends = 0;
  response.end = () => { ends++; };
  const interval = {};
  t.mock.method(globalThis, 'setInterval', () => interval);
  const clear = t.mock.method(globalThis, 'clearInterval', () => {});
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const hub = new EventHub();
  hub.add(response);
  const closing = hub.close();
  assert.equal(hub.close(), closing);
  assert.equal(ends, 1);
  assert.equal(clear.mock.calls.length, 1);
  assert.equal(clear.mock.calls[0].arguments[0], interval);
  t.mock.timers.tick(1000);
  await closing;
  assert.equal(hub.size, 0);
  for (const name of ['finish', 'close', 'error']) assert.equal(response.listenerCount(name), 0);
  await hub.close();
  assert.equal(ends, 1);
});
