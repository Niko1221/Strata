// Stop during the Harness start ends the turn's wait for it, not the start (the turn returns at once, cancelled). A close() in that
// window found no runtime yet; the start then finished and its Harness outlived the bridge, with an event stream that kept the host
// process alive. Seen in the full host suite under load: tests/agent-tasks.test.mjs ("Stop on the task ends its agent run") stopped
// while its Harness was still starting, and the test file never exited. These tests drive the real bridge with a held fake start.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { HarnessProcess } from '../server/agent/harness-runtime.mjs';

const sleep = ms => new Promise(r => setTimeout(r, ms));
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
async function until(fn, ms = 5000) { const end = Date.now() + ms; while (Date.now() < end) { if (fn()) return true; await sleep(10); } return false; }

/** A bridge whose Harness start waits until the test releases it; every close of what that start made is recorded. */
function rig(t) {
  const dir = mkdtempSync(join(tmpdir(), 'strata-close-start-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const release = deferred();
  const closed = [];
  let starts = 0;
  const startRuntimeImpl = async profile => {
    starts++;
    await release.promise;
    return {
      hp: { running: true, pid: 4242, stop: async () => { closed.push('harness'); } },
      gw: { close() { closed.push('gateway'); }, async rpc() { return {}; }, openStream() { return { close() {} }; } },
      profile, clientId: 'c', events: { close() { closed.push('events'); } },
    };
  };
  const bridge = createAgentBridge({
    dataDir: dir, localInstances: () => [{ id: 'i1', label: 'S', baseUrl: 'http://127.0.0.1:9', modelId: 'm-i1', context: 32768, slots: 2 }],
    connections: () => [], env: process.env, startRuntimeImpl,
  });
  t.after(async () => { release.resolve(); await bridge.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  return { bridge, ws, release, closed, starts: () => starts };
}

test('close() after a Stop during the Harness start waits for that start and stops the Harness it made', async t => {
  const r = rig(t);
  const frames = [];
  const turn = r.bridge.runTurn({ chatId: 'chat-a', message: 'hello', workspacePath: r.ws, mode: 'bypass', onFrame: f => frames.push(f), requestId: 'req-a' });
  assert.equal(await until(() => r.starts() === 1), true, 'the Harness start began');

  assert.deepEqual(await r.bridge.stop('chat-a'), { stopped: true, ended: 'cancelled-before-start', ms: 0 });
  assert.deepEqual(await turn, { cancelled: true }, 'the turn ended while the start goes on');

  let done = false;
  const closing = r.bridge.close().then(() => { done = true; });
  await sleep(100);
  assert.equal(done, false, 'close() waits for the start that is still under way');
  assert.deepEqual(r.closed, []);

  r.release.resolve();
  await closing;
  assert.deepEqual(r.closed, ['events', 'gateway', 'harness'], 'the Harness that start made is stopped, not left running');
});

test('close() with no start under way returns at once', async t => {
  const r = rig(t);
  const t0 = Date.now();
  await r.bridge.close();
  assert.ok(Date.now() - t0 < 1000);
  assert.deepEqual(r.closed, []);
  assert.equal(r.starts(), 0);
});

test('aborting startup stops the child before it announces an address', { timeout: 30_000 }, async t => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-abort-harness-'));
  const bin = join(dir, 'fake-harness.cjs');
  const entered = join(dir, 'entered');
  writeFileSync(bin, `require('node:fs').writeFileSync(${JSON.stringify(entered)}, ''); setInterval(() => {}, 1000);`);
  const child = new HarnessProcess({ harness: { bin }, home: dir, cwd: dir, patchFile: join(dir, 'unused.yml'), nodePath: process.execPath });
  const lifetime = new AbortController();
  const reason = new Error('The bridge closed.');
  const outcome = child.start({ signal: lifetime.signal }).then(() => null, error => error);
  t.after(async () => {
    if (child.running && child.pid) process.kill(child.pid, 'SIGKILL');
    await outcome;
    rmSync(dir, { recursive: true, force: true, maxRetries: 3 });
  });
  assert.equal(await until(() => existsSync(entered), 15_000), true, 'the child started and has yet to announce an address');
  lifetime.abort(reason);
  assert.equal(await until(() => !child.running, 10_000), true, 'the child exited during startup');
  assert.equal(await outcome, reason);
});
