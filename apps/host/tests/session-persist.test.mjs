/**
 * #3604 / #3603: the chat list is saved at most once a second and only the chats that changed are written.
 * #5192: nothing is ever cut out of a saved chat; a full storage is reported, never trimmed around.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));

async function bundle(entry, plugins = []) {
  const result = await build({
    entryPoints: [here(entry)], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent',
    alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' }, plugins,
  });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}

const persist = await bundle('../src/lib/session-persist.ts');
const { createSessionWriter, serializeSession, loadStoredSessions, exportSessions, LEGACY_SESSIONS_KEY, SESSION_INDEX_KEY, SESSION_KEY_PREFIX } = persist;

/** A Storage that refuses a write over `limit` characters in total (keys and values), like a full browser quota. */
function fakeStorage(limit = Infinity) {
  const map = new Map();
  const used = () => [...map].reduce((n, [k, v]) => n + k.length + v.length, 0);
  return {
    map,
    get length() { return map.size; },
    key: (i) => [...map.keys()][i] ?? null,
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    removeItem: (k) => { map.delete(k); },
    setItem(k, v) {
      const before = map.has(k) ? k.length + map.get(k).length : 0;
      if (used() - before + k.length + String(v).length > limit) { const e = new Error('The quota has been exceeded.'); e.name = 'QuotaExceededError'; throw e; }
      map.set(k, String(v));
    },
  };
}
const chat = (id, n, len, extra = {}) => ({
  id, title: `Chat ${id}`, createdAt: `2026-11-0${(id.charCodeAt(0) % 8) + 1}`, updatedAt: id, workspace: 'C:/w', tokensIn: 0, tokensOut: 0,
  messages: Array.from({ length: n }, (_, i) => ({ id: `${id}-m${i}`, role: i % 2 ? 'assistant' : 'user', content: `${id}:${i}:` + String.fromCharCode(97 + (i % 26)).repeat(len) })),
  ...extra,
});
const save = (storage, sessions, opts = {}) => { const writer = createSessionWriter({ storage, ...opts }); writer.schedule(sessions); return { writer, result: writer.flush() }; };
const reload = (storage) => loadStoredSessions(storage).sessions;

test('2,000 schedules inside a second produce one write, and the newest state wins', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const storage = fakeStorage();
  let sets = 0;
  const raw = storage.setItem;
  storage.setItem = (k, v) => { if (k.startsWith(SESSION_KEY_PREFIX)) sets++; raw(k, v); };
  const writer = createSessionWriter({ storage, delayMs: 1_000 });
  for (let i = 0; i < 2_000; i++) writer.schedule([{ id: 's1', seq: i, messages: [] }]);
  assert.equal(sets, 0, 'nothing is written synchronously');
  t.mock.timers.tick(1_000);
  assert.equal(sets, 1);
  assert.equal(JSON.parse(storage.getItem(`${SESSION_KEY_PREFIX}s1`)).seq, 1_999);
  writer.schedule([{ id: 's1', seq: 2_000, messages: [] }]);
  writer.flush();
  assert.equal(sets, 2);
  assert.equal(JSON.parse(storage.getItem(`${SESSION_KEY_PREFIX}s1`)).seq, 2_000);
  writer.flush();
  assert.equal(sets, 2, 'a flush with nothing pending writes nothing');
});

test('only the chat that changed is serialised and written again', () => {
  const seen = [];
  const storage = fakeStorage();
  const wrote = [];
  const raw = storage.setItem;
  storage.setItem = (k, v) => { wrote.push(k); raw(k, v); };
  const writer = createSessionWriter({ storage, onSerialize: (s) => seen.push(s.id) });
  const a = { id: 'a', updatedAt: '1', messages: [{ id: 'm', content: 'x'.repeat(1000) }] };
  const b = { id: 'b', updatedAt: '2', messages: [] };
  writer.schedule([a, b]); writer.flush();
  assert.deepEqual(seen.sort(), ['a', 'b']);
  seen.length = 0; wrote.length = 0;
  for (let i = 0; i < 50; i++) { writer.schedule([a, { ...b, title: `t${i}` }]); writer.flush(); }
  assert.equal(seen.filter((id) => id === 'a').length, 0, 'the untouched chat is never stringified again');
  assert.equal(seen.filter((id) => id === 'b').length, 50);
  assert.equal(wrote.filter((k) => k === `${SESSION_KEY_PREFIX}a`).length, 0, 'the untouched chat is never written again');
  assert.equal(reload(storage).length, 2);
});

/* #5192: the register's own numbers. Twelve messages of 140,000 characters went through the real serializer and six came back. */
test('#5192: 12 messages of 140,000 characters are saved and restored whole, twice over (write, reload, write, reload)', () => {
  const storage = fakeStorage();
  const original = chat('a', 12, 140_000);
  const before = JSON.stringify(original);
  save(storage, [original]);
  const once = reload(storage);
  assert.equal(once.length, 1);
  assert.equal(once[0].messages.length, 12, 'all twelve messages come back');
  assert.equal(JSON.stringify(once[0]), before, 'byte for byte after one restart');
  save(storage, once);
  const twice = reload(storage);
  assert.equal(JSON.stringify(twice[0]), before, 'byte for byte after two restarts');
  assert.equal(JSON.stringify(original), before, 'the chat in memory was not changed');
});

test('#5192: more text than the old 3.5M-character list budget keeps every chat and every message', () => {
  const storage = fakeStorage();
  const chats = ['a', 'b', 'c', 'd', 'e'].map((id, i) => chat(id, 8, 100_000, { updatedAt: String(i) }));
  const total = chats.reduce((n, c) => n + JSON.stringify(c).length, 0);
  assert.ok(total > 4_000_000, `fixture is ${total} characters`);
  save(storage, chats);
  const back = reload(storage);
  assert.deepEqual(back.map((s) => s.id).sort(), ['a', 'b', 'c', 'd', 'e']);
  for (const c of chats) assert.equal(JSON.stringify(back.find((s) => s.id === c.id)), JSON.stringify(c));
});

test('#5192: a 250,000-character user message and a 1.6M-character tool result are not clipped', () => {
  const storage = fakeStorage();
  const c = { id: 'u', messages: [{ id: '1', role: 'user', content: 'p'.repeat(250_000) }, { id: '2', role: 'assistant', content: 'ok', tools: [{ name: 'read', result: 'y'.repeat(1_600_000) }] }] };
  save(storage, [c]);
  const back = reload(storage)[0];
  assert.equal(back.messages[0].content.length, 250_000);
  assert.equal(back.messages[1].tools[0].result.length, 1_600_000);
  assert.ok(!JSON.stringify(back).includes('shortened'));
  assert.equal(serializeSession(c), JSON.stringify(c));
});

test('#5192: a full storage fails visibly: the chat keeps its last good copy, the others still save, and the unsaved chat is offered whole', () => {
  const small = chat('s', 2, 20_000);
  const big = chat('b', 4, 40_000);
  const storage = fakeStorage(300_000);
  const results = [];
  const writer = createSessionWriter({ storage, onResult: (r) => results.push(r) });
  writer.schedule([small, big]); writer.flush();
  assert.equal(results.at(-1).ok, true);
  const goodCopy = storage.getItem(`${SESSION_KEY_PREFIX}b`);
  // The chat grows past what the storage can hold; a new small chat is added at the same time.
  const grown = { ...big, messages: [...big.messages, { id: 'b-big', role: 'user', content: 'g'.repeat(120_000) }] };
  const fresh = chat('n', 2, 20_000);
  writer.schedule([fresh, small, grown]);
  const r = writer.flush();
  assert.equal(r.ok, false);
  assert.deepEqual(r.failed.map((f) => [f.id, f.reason]), [['b', 'quota']]);
  assert.equal(storage.getItem(`${SESSION_KEY_PREFIX}b`), goodCopy, 'the last good copy of the chat is untouched');
  assert.ok(storage.getItem(`${SESSION_KEY_PREFIX}n`), 'the other chats are still saved');
  assert.deepEqual(writer.unsaved().map((s) => s.id), ['b']);
  assert.equal(JSON.parse(exportSessions(writer.unsaved()))[0].messages.length, 5, 'the unsaved chat is complete');
  assert.equal(results.at(-1).ok, false, 'the failure is reported');
  // Still too big: reported again at the next save, not forgotten.
  writer.schedule([fresh, small, grown]);
  assert.equal(writer.flush().ok, false);
  // Room is made (the other chats are deleted) and the next save writes the chat again without being asked.
  writer.schedule([grown]);
  const done = writer.flush();
  assert.equal(done.ok, true);
  assert.equal(writer.unsaved().length, 0);
  assert.equal(reload(storage)[0].messages.length, 5);
});

test('removed chats leave storage, an entry that cannot be read stays and does not stop the others loading', () => {
  const storage = fakeStorage();
  const a = chat('a', 2, 10), b = chat('b', 2, 10);
  const { writer } = save(storage, [a, b]);
  storage.setItem(`${SESSION_KEY_PREFIX}broken`, '{"id": "broken", "messa');
  const loaded = loadStoredSessions(storage);
  assert.deepEqual(loaded.sessions.map((s) => s.id).sort(), ['a', 'b']);
  assert.deepEqual(loaded.damaged, [`${SESSION_KEY_PREFIX}broken`]);
  writer.schedule([a]); writer.flush();
  assert.equal(storage.getItem(`${SESSION_KEY_PREFIX}b`), null, 'a deleted chat is removed');
  assert.ok(storage.getItem(`${SESSION_KEY_PREFIX}broken`), 'an unreadable entry is never deleted');
  assert.deepEqual(JSON.parse(storage.getItem(SESSION_INDEX_KEY)).ids, ['a']);
});

test('an empty list scheduled before anything was loaded does not delete saved chats', () => {
  const storage = fakeStorage();
  save(storage, [chat('a', 2, 10)]);
  const w = createSessionWriter({ storage });
  w.schedule([]); w.flush();
  assert.equal(reload(storage).length, 1);
});

test('migration: the old single list is moved chat by chat, whole, and then removed', () => {
  const storage = fakeStorage();
  const chats = [chat('x', 12, 140_000), chat('y', 3, 1_000)];
  storage.setItem(LEGACY_SESSIONS_KEY, JSON.stringify(chats));
  const loaded = loadStoredSessions(storage);
  assert.equal(loaded.unmigrated, 0);
  assert.deepEqual(loaded.sessions.map((s) => JSON.stringify(s)).sort(), chats.map((s) => JSON.stringify(s)).sort());
  assert.equal(storage.getItem(LEGACY_SESSIONS_KEY), null, 'the old entry is gone once every chat has its own');
  assert.equal(reload(storage).length, 2);
});

test('migration with a full storage loses nothing: unmoved chats stay in the old entry and are still loaded', () => {
  const chats = [chat('x', 3, 40_000), chat('y', 3, 40_000)];
  const legacy = JSON.stringify(chats);
  const storage = fakeStorage(legacy.length + 130_000); // room for one more chat on top of the old list, not two
  storage.setItem(LEGACY_SESSIONS_KEY, legacy);
  const loaded = loadStoredSessions(storage);
  assert.equal(loaded.sessions.length, 2, 'both chats are shown');
  const inOld = JSON.parse(storage.getItem(LEGACY_SESSIONS_KEY) ?? '[]').map((s) => s.id);
  const inNew = [...storage.map.keys()].filter((k) => k.startsWith(SESSION_KEY_PREFIX)).map((k) => k.slice(SESSION_KEY_PREFIX.length));
  assert.deepEqual([...new Set([...inOld, ...inNew])].sort(), ['x', 'y'], 'each chat is in the old or the new place');
  for (const s of loaded.sessions) assert.equal(JSON.stringify(s), JSON.stringify(chats.find((c) => c.id === s.id)));
});

test('migration leaves an old entry that is not JSON exactly as it is', () => {
  const storage = fakeStorage();
  storage.setItem(LEGACY_SESSIONS_KEY, '[{"id":"a","messages":[');
  const loaded = loadStoredSessions(storage);
  assert.equal(loaded.sessions.length, 0);
  assert.equal(storage.getItem(LEGACY_SESSIONS_KEY), '[{"id":"a","messages":[');
  const writer = createSessionWriter({ storage });
  writer.schedule([chat('n', 1, 5)]); writer.flush();
  assert.equal(storage.getItem(LEGACY_SESSIONS_KEY), '[{"id":"a","messages":[', 'a later save does not remove it either');
});

test('the order of the chats survives a restart, and a lost index loses no chat', () => {
  const storage = fakeStorage();
  save(storage, [chat('c', 1, 5), chat('a', 1, 5), chat('b', 1, 5)]);
  assert.deepEqual(reload(storage).map((s) => s.id), ['c', 'a', 'b']);
  storage.removeItem(SESSION_INDEX_KEY);
  assert.deepEqual(reload(storage).map((s) => s.id).sort(), ['a', 'b', 'c']);
});

/* End to end through the real store: a 1,500-token streamed answer. */
test('a streamed answer of 1,500 tokens is at most a few storage writes, and the final text is saved', async () => {
  const writes = [];
  globalThis.window ??= globalThis;
  globalThis.document ??= { addEventListener() {}, visibilityState: 'visible', documentElement: { dataset: {}, style: { setProperty() {} }, classList: { toggle() {}, add() {}, remove() {} } } };
  const memory = new Map();
  globalThis.localStorage = {
    get length() { return memory.size; },
    key: (i) => [...memory.keys()][i] ?? null,
    getItem: (k) => memory.get(k) ?? null,
    setItem: (k, v) => { if (k.startsWith(SESSION_KEY_PREFIX)) writes.push(v); memory.set(k, v); },
    removeItem: (k) => memory.delete(k),
  };
  globalThis.navigator ??= { language: 'en' };
  globalThis.matchMedia ??= () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} });
  const TOKENS = 1_500;
  const mockClient = {
    name: 'mock-client',
    setup(b) {
      b.onResolve({ filter: /\/client$/ }, () => ({ path: 'mock-client', namespace: 'mock' }));
      b.onLoad({ filter: /.*/, namespace: 'mock' }, () => ({
        loader: 'js',
        contents: `
          export class ApiError extends Error {}
          const reject = async () => { throw new Error('offline'); };
          export const api = { get: reject, post: async () => ({}), put: reject, patch: reject, del: reject, delete: reject };
          export function onEvent() { return () => {}; }
          export async function streamPost(path, body, onFrame) {
            onFrame({ type: 'start', model: 'm', contextMax: 8192, requestId: 'r1' });
            for (let i = 0; i < ${TOKENS}; i++) onFrame({ type: 'delta', text: 'word ' });
            onFrame({ type: 'done', tokensIn: 10, tokensOut: ${TOKENS}, durationMs: 100, usageSource: 'reported' });
          }`,
      }));
    },
  };
  const store = await bundle('../src/lib/store.ts', [mockClient]);
  const before = writes.length;
  await store.sendMessage('Write a long answer');
  const during = writes.length - before;
  assert.ok(during <= 6, `a 1,500-token answer wrote the chat ${during} times`);
  const saved = JSON.parse(writes.at(-1));
  const answer = saved.messages.find((m) => m.role === 'assistant');
  assert.ok(answer.content.length >= TOKENS * 5 - 10, 'the finished answer is what was saved');
  assert.equal(answer.streaming, false);
});
