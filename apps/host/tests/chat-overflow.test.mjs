/** Chat-overflow guard (docs/DEV_BUILD.md): an old chat leaves the capped saved chat list only once it is backed up whole; off, nothing changes. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));

async function bundle(entry) {
  const result = await build({
    entryPoints: [here(entry)], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent',
    alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' },
  });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}

const { createSessionWriter, MAX_TOTAL_CHARS } = await bundle('../src/lib/session-persist.ts');
const { createOverflowGuard, RETRY_MS } = await bundle('../src/lib/chat-overflow.ts');
const { saveOverflow, listOverflow, overflowFileName, OVERFLOW_FILE } = await import('../server/chat-overflow.mjs');
const { devSettings } = await import('../server/dev-mode.mjs');
const { rescueFiles } = await import('../server/dev-rescue.mjs');

/** Five chats of ~1,000 characters, s0 the oldest; a cap of 3,500 leaves room for three. */
const chats = (n = 5) => Array.from({ length: n }, (_, i) => ({ id: `s${i}`, updatedAt: `2026-11-0${i + 1}T10:00:00.000Z`, title: `t${i}`, messages: [{ role: 'user', content: 'x'.repeat(950) }] }));
const ids = (json) => JSON.parse(json).map(s => s.id);
const tick = () => new Promise(r => setImmediate(r));

function harness({ fail = () => false, now } = {}) {
  const posted = [], events = [];
  let rewrite = () => {};
  const guard = createOverflowGuard({
    maxTotalChars: 3_500,
    now,
    post: async (session) => { posted.push(session); if (fail(session)) throw new Error('disk full'); return { dir: 'C:\\Dev\\home\\chat-overflow', file: `overflow-${session.id}.json` }; },
    handlers: { saved: (count, dir) => events.push(['saved', count, dir]), failed: (m) => events.push(['failed', m]), rewrite: () => { events.push(['rewrite']); rewrite(); } },
  });
  return { guard, posted, events, onRewrite: (fn) => { rewrite = fn; } };
}

test('without the guard the writer leaves out the oldest chats exactly as before', () => {
  const plain = [], guarded = [];
  const a = createSessionWriter({ write: j => plain.push(j), maxTotalChars: 3_500 }); a.schedule(chats()); a.flush();
  const b = createSessionWriter({ write: j => guarded.push(j), maxTotalChars: 3_500, overflow: () => null }); b.schedule(chats()); b.flush();
  assert.equal(guarded[0], plain[0], 'byte-identical with the guard off');
  assert.deepEqual(ids(plain[0]), ['s2', 's3', 's4']);
  assert.equal(MAX_TOTAL_CHARS, 3_500_000, 'the cap itself is unchanged');
});

test('a chat that is not backed up stays in the saved list and is sent for backup, oldest first', async () => {
  const h = harness();
  const writes = [];
  const writer = createSessionWriter({ write: j => writes.push(j), overflow: () => h.guard });
  writer.schedule(chats()); writer.flush();
  assert.deepEqual(ids(writes[0]), ['s0', 's1', 's2', 's3', 's4'], 'nothing is dropped before its backup');
  await tick(); await tick();
  assert.deepEqual(h.posted.map(s => s.id), ['s0', 's1']);
  assert.equal(h.posted[0].messages[0].content.length, 950, 'the chat is sent whole');
  assert.deepEqual(h.events, [['saved', 2, 'C:\\Dev\\home\\chat-overflow'], ['rewrite']]);
});

test('once backed up, the next write leaves the chats out, and a backed-up chat is not sent again', async () => {
  const h = harness();
  const writes = [];
  const writer = createSessionWriter({ write: j => writes.push(j), overflow: () => h.guard });
  const list = chats();
  h.onRewrite(() => { writer.schedule(list); writer.flush(); });
  writer.schedule(list); writer.flush();
  await tick(); await tick();
  assert.deepEqual(ids(writes.at(-1)), ['s2', 's3', 's4']);
  writer.schedule(list); writer.flush();
  await tick();
  assert.equal(h.posted.length, 2, 'no second backup of the same chat version');
});

test('a chat that changed after its backup is backed up again before it may leave', async () => {
  const h = harness();
  const list = chats();
  h.guard.backUp([list[0]]);
  await tick(); await tick();
  assert.equal(h.guard.isBackedUp(list[0]), true);
  assert.equal(h.guard.isBackedUp({ ...list[0], updatedAt: '2026-11-09T00:00:00.000Z' }), false);
});

test('a failed backup is a warning, the chat stays, and it is tried again only after the retry time', async () => {
  let clock = 1_000_000;
  const h = harness({ fail: s => s.id === 's0', now: () => clock });
  const writes = [];
  const writer = createSessionWriter({ write: j => writes.push(j), overflow: () => h.guard });
  const list = chats();
  h.onRewrite(() => { writer.schedule(list); writer.flush(); });
  writer.schedule(list); writer.flush();
  await tick(); await tick(); await tick();
  assert.ok(h.events.some(e => e[0] === 'failed' && e[1] === 'disk full'));
  assert.ok(ids(writes.at(-1)).includes('s0'), 's0 was not backed up, so it is still saved');
  assert.ok(!ids(writes.at(-1)).includes('s1'), 's1 was backed up and left the list');
  const before = h.posted.length;
  writer.schedule(list); writer.flush(); await tick();
  assert.equal(h.posted.length, before, 'no retry storm');
  clock += RETRY_MS;
  writer.schedule(list); writer.flush(); await tick(); await tick();
  assert.equal(h.posted.length, before + 1, 'tried again after the retry time');
});

test('pinned chats leave last and the newest chat always stays', async () => {
  const h = harness();
  const list = chats();
  list[0] = { ...list[0], pinned: true };
  const writes = [];
  const writer = createSessionWriter({ write: j => writes.push(j), overflow: () => h.guard });
  writer.schedule(list); writer.flush();
  await tick(); await tick();
  assert.deepEqual(h.posted.map(s => s.id), ['s1', 's2'], 'the pinned s0 is not the first to go');
  const tiny = createOverflowGuard({ maxTotalChars: 10, post: async () => ({ dir: 'd', file: 'f' }), handlers: { saved() {}, failed() {}, rewrite() {} } });
  const one = [];
  const w2 = createSessionWriter({ write: j => one.push(j), overflow: () => tiny });
  w2.schedule(chats(2)); w2.flush(); await tick(); await tick();
  w2.schedule(chats(2)); w2.flush();
  assert.deepEqual(ids(one.at(-1)), ['s1']);
});

test('the host writes one file per chat version, without the title in the name, and lists it for the Rescue tab', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-overflow-'));
  try {
    const session = { id: 'abc/../x', updatedAt: '2026-11-03T02:30:05.000Z', title: 'Geheimes Projekt', messages: [] };
    const first = saveOverflow(dir, session);
    assert.match(first.file, OVERFLOW_FILE);
    assert.ok(!first.file.includes('Geheim') && !first.file.includes('..'), first.file);
    assert.equal(first.existed, false);
    assert.equal(saveOverflow(dir, session).existed, true, 'the same version is written once');
    assert.deepEqual(JSON.parse(readFileSync(join(dir, first.file), 'utf8')), [session]);
    assert.ok(!readdirSync(dir).some(n => n.endsWith('.tmp')), 'no temp file left');
    assert.equal(listOverflow(dir).length, 1);
    assert.throws(() => saveOverflow(dir, { title: 'no id' }), /id/);
    const rescue = rescueFiles({}, [dir]).find(s => s.dir === dir);
    assert.equal(rescue.files[0].name, first.file);
    assert.equal(rescue.files[0].chats, 1);
    assert.equal(overflowFileName({ id: 'x' }).startsWith('overflow-'), true, 'a chat without a date still gets a name');
  } finally { rmSync(dir, { recursive: true, force: true }); }
  assert.equal(existsSync(dir), false);
});

test('dev.config.json "chatOverflowGuard": on unless switched off; maxTotalChars only within its bounds', () => {
  assert.deepEqual(devSettings({}).chatOverflow, { enabled: true });
  assert.deepEqual(devSettings({ chatOverflowGuard: false }).chatOverflow, { enabled: false });
  assert.deepEqual(devSettings({ chatOverflowGuard: { maxTotalChars: 200_000 } }).chatOverflow, { enabled: true, maxTotalChars: 200_000 });
  assert.deepEqual(devSettings({ chatOverflowGuard: { enabled: false, maxTotalChars: 5 } }).chatOverflow, { enabled: false });
});
