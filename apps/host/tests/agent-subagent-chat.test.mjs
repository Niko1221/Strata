/**
 * The HTTP surface of a helper's conversation: a chat id is required, a helper that does not belong
 * to that chat is not found, and a prompt needs real text. These checks answer before the agent or
 * the Harness is started, so they run without a built Harness.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { ROOMY_HOST } from './helpers/hardware.mjs';

const root = resolve(import.meta.dirname, '..');
const HOST = join(root, 'server', 'index.mjs');
const sleep = ms => new Promise(r => setTimeout(r, ms));
const freePort = () => new Promise(r => { const s = createServer(); s.listen(0, '127.0.0.1', () => { const port = s.address().port; s.close(() => r(port)); }); });

async function startHost(t) {
  const home = mkdtempSync(join(tmpdir(), 'strata-child-chat-'));
  const ws = join(home, 'workspace');
  mkdirSync(ws, { recursive: true });
  const port = await freePort();
  const env = {
    ...process.env,
    STRATA_HOME: home,
    STRATA_PORT: String(port),
    STRATA_WORKSPACE: ws,
    STRATA_NO_ACQUIRE: '1',
    STRATA_REAL_ENGINE: '0',
    STRATA_TEST_HARDWARE: ROOMY_HOST,
    STRATA_COMPONENTS_DIR: join(home, 'components'),
  };
  delete env.STRATA_HARNESS_DIR;
  const child = spawn(process.execPath, [HOST], { env, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
  let log = '';
  child.stdout.on('data', d => { log += d; });
  child.stderr.on('data', d => { log += d; });
  const stop = async () => {
    if (child.exitCode !== null) return;
    if (process.platform === 'win32') spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true });
    else child.kill('SIGTERM');
    await new Promise(r => child.once('exit', r));
    try { rmSync(home, { recursive: true, force: true, maxRetries: 3 }); } catch { /* the engine may still hold a file */ }
  };
  t.after(stop);
  const base = `http://127.0.0.1:${port}`;
  let up = false;
  for (let i = 0; i < 150 && !up; i += 1) {
    try { up = (await fetch(`${base}/api/health`)).ok; } catch { await sleep(100); }
  }
  assert.ok(up, `the host started\n${log}`);
  const get = async path => { const res = await fetch(`${base}${path}`); return { status: res.status, body: await res.json().catch(() => null) }; };
  const post = async (path, body) => {
    const res = await fetch(`${base}${path}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
    return { status: res.status, body: await res.json().catch(() => null) };
  };
  return { get, post, log: () => log };
}

test('reading a helper needs the chat, and a helper of another chat is not found', async t => {
  const host = await startHost(t);
  const noChat = await host.get('/api/chat/subagents/child-1');
  assert.equal(noChat.status, 400);
  assert.match(noChat.body.error, /sessionId/);
  const unknown = await host.get('/api/chat/subagents/child-1?sessionId=chat-1');
  assert.equal(unknown.status, 404);
  assert.equal(unknown.body.code, 'child-gone');
});

test('sending to a helper needs a chat, a helper and real text; an unknown helper is refused before the agent starts', async t => {
  const host = await startHost(t);
  const cases = [
    [{}, /sessionId/],
    [{ sessionId: 'chat-1' }, /childId/],
    [{ sessionId: 'chat-1', childId: 'child-1' }, /text/],
    [{ sessionId: 'chat-1', childId: 'child-1', text: '   ' }, /text/],
    [{ sessionId: 'chat-1', childId: 'child-1', text: 'x'.repeat(20_001) }, /too long/],
  ];
  for (const [body, message] of cases) {
    const answer = await host.post('/api/chat/subagent/prompt', body);
    assert.equal(answer.status, 400, JSON.stringify(body));
    assert.match(answer.body.error, message);
  }
  const unknown = await host.post('/api/chat/subagent/prompt', { sessionId: 'chat-1', childId: 'not-here', text: 'go on' });
  assert.equal(unknown.status, 404);
  assert.equal(unknown.body.code, 'child-gone');
});
