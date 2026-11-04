/** #3604 / #3603: the chat list is saved at most once a second, changed chats only, and shortened by field; a streamed answer is a handful of writes. */
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
const { createSessionWriter, serializeSession } = persist;

test('2,000 schedules inside a second produce one write, and the newest state wins', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const writes = [];
  const writer = createSessionWriter({ write: (json) => writes.push(JSON.parse(json)), delayMs: 1_000 });
  for (let i = 0; i < 2_000; i++) writer.schedule([{ id: 's1', seq: i, messages: [] }]);
  assert.equal(writes.length, 0, 'nothing is written synchronously');
  t.mock.timers.tick(1_000);
  assert.equal(writes.length, 1);
  assert.equal(writes[0][0].seq, 1_999);
  writer.schedule([{ id: 's1', seq: 2_000, messages: [] }]);
  writer.flush();
  assert.equal(writes.length, 2);
  assert.equal(writes[1][0].seq, 2_000);
  writer.flush();
  assert.equal(writes.length, 2, 'a flush with nothing pending writes nothing');
});

test('only the chat that changed is serialised again', () => {
  const seen = [];
  const writes = [];
  const writer = createSessionWriter({ write: (j) => writes.push(j), onSerialize: (s) => seen.push(s.id) });
  const a = { id: 'a', updatedAt: '1', messages: [{ id: 'm', content: 'x'.repeat(1000) }] };
  const b = { id: 'b', updatedAt: '2', messages: [] };
  writer.schedule([a, b]); writer.flush();
  assert.deepEqual(seen.sort(), ['a', 'b']);
  seen.length = 0;
  for (let i = 0; i < 50; i++) { writer.schedule([a, { ...b, title: `t${i}` }]); writer.flush(); }
  assert.equal(seen.filter((id) => id === 'a').length, 0, 'the untouched chat is never stringified again');
  assert.equal(seen.filter((id) => id === 'b').length, 50);
  assert.equal(JSON.parse(writes.at(-1)).length, 2);
});

test('the saved copy is shortened by field: no message and no chat is ever dropped', () => {
  const big = 'y'.repeat(1_600_000);
  const one = serializeSession({ id: 'a', messages: [{ id: '1', tools: [{ result: big }] }] });
  assert.ok(one.length < 250_000, 'one 1.6M field is clipped');
  assert.equal(JSON.parse(one).messages.length, 1, 'the message survives the clip');
  const many = { id: 'b', messages: Array.from({ length: 40 }, (_, i) => ({ id: String(i), content: 'z'.repeat(250_000) })) };
  const text = serializeSession(many);
  assert.ok(text.length < JSON.stringify(many).length, 'a long chat is shortened by its oversized fields');
  const kept = JSON.parse(text).messages;
  assert.equal(kept.length, 40, 'every message of a long chat is kept');
  assert.equal(kept.at(-1).id, '39', 'the latest message is kept');
  assert.ok(kept.every((m) => String(m.content).includes('[shortened in the saved copy]')), 'the big fields, not the messages, are what shrink');

  const writes = [];
  const writer = createSessionWriter({ write: (j) => writes.push(JSON.parse(j)), maxTotalChars: 320_000 });
  const chats = ['old', 'mid', 'new'].map((id, i) => ({ id, updatedAt: String(i), messages: [{ id: 'm', content: 'q'.repeat(150_000) }] }));
  writer.schedule(chats); writer.flush();
  assert.deepEqual(writes[0].map((s) => s.id), ['old', 'mid', 'new'], 'every chat stays in the saved copy');
  const shrunk = writes[0].filter((s) => String(s.messages[0].content).includes('[shortened in the saved copy]'));
  assert.ok(shrunk.length >= 2, 'the oldest chats shrink their big texts instead of the list losing chats');
  assert.ok(writes[0].every((s) => String(s.messages[0].content).length <= 150_000), 'no chat is kept whole beyond its original size');
  assert.ok(JSON.stringify(writes[0]).length <= 320_000, 'the saved list fits its budget');
});

test('four agent turns with big tool results survive save, compaction and a reload: every question is still there', () => {
  const turns = [
    { id: 'u1', role: 'user', content: 'how much do we know about chemical composition of Saturn\'s moons?' },
    { id: 'a1', role: 'assistant', content: 'Short version: Titan and Enceladus.', tools: [{ id: 't1', result: 'x'.repeat(800_000) }, { id: 't2', result: 'w'.repeat(800_000) }] },
    { id: 'u2', role: 'user', content: 'what about the rings?' },
    { id: 'a2', role: 'assistant', content: 'The rings are mostly water ice.', tools: [{ id: 't3', result: 'v'.repeat(800_000) }, { id: 't4', result: 'r'.repeat(800_000) }] },
    { id: 'u3', role: 'user', content: 'how much do we know about Mars\' moons?' },
    { id: 'a3', role: 'assistant', content: 'Phobos and Deimos.', tools: [{ id: 't5', result: 'u'.repeat(800_000) }, { id: 't6', result: 'p'.repeat(800_000) }] },
    { id: 'u4', role: 'user', content: 'and their shapes?' },
    { id: 'a4', role: 'assistant', content: 'Both are lumpy.', tools: [{ id: 't7', result: 'q'.repeat(800_000) }, { id: 't8', result: 's'.repeat(800_000) }] },
  ];
  const session = { id: 's1', title: 'moons', updatedAt: '2026-10-22', messages: turns };
  const writes = [];
  const writer = createSessionWriter({ write: (j) => writes.push(j) });
  // Four turns: a schedule and a flush each, as the store does when a turn ends.
  for (let upto = 2; upto <= 8; upto += 2) { writer.schedule([{ ...session, messages: turns.slice(0, upto) }]); writer.flush(); }
  // Reload: the app reads the last saved copy back (that is the only stored copy) and saves once more with it.
  const reloaded = JSON.parse(writes.at(-1));
  writer.schedule(reloaded); writer.flush();
  const after = JSON.parse(writes.at(-1));
  const questions = after[0].messages.filter((m) => m.role === 'user').map((m) => m.content);
  assert.deepEqual(questions, ['how much do we know about chemical composition of Saturn\'s moons?', 'what about the rings?', 'how much do we know about Mars\' moons?', 'and their shapes?'], 'no question is lost');
  assert.equal(after[0].messages.filter((m) => m.role === 'assistant').length, 4, 'no answer is lost either');
});

test('the saved list keeps the format older builds read (a JSON array of chats)', () => {
  let out = '';
  const w = createSessionWriter({ write: (j) => { out = j; } });
  w.schedule([{ id: 'a', title: 'A', messages: [] }]); w.flush();
  assert.deepEqual(JSON.parse(out), [{ id: 'a', title: 'A', messages: [] }]);
});

/* End to end through the real store: a 1,500-token streamed answer. */
test('a streamed answer of 1,500 tokens is at most a few storage writes, and the final text is saved', async () => {
  const writes = [];
  globalThis.window ??= globalThis;
  globalThis.document ??= { addEventListener() {}, visibilityState: 'visible', documentElement: { dataset: {}, style: { setProperty() {} }, classList: { toggle() {}, add() {}, remove() {} } } };
  const memory = new Map();
  globalThis.localStorage = {
    getItem: (k) => memory.get(k) ?? null,
    setItem: (k, v) => { if (k === 'strata.sessions.v1') writes.push(v); memory.set(k, v); },
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
  assert.ok(during <= 6, `a 1,500-token answer wrote the chat list ${during} times`);
  const saved = JSON.parse(writes.at(-1));
  const answer = saved[0].messages.find((m) => m.role === 'assistant');
  assert.ok(answer.content.length >= TOKENS * 5 - 10, 'the finished answer is what was saved');
  assert.equal(answer.streaming, false);
});
