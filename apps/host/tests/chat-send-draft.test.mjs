/**
 * A message that never lands must not eat the draft or the quotes.
 *
 * The chat now clears the box only after the store says the text has a home (sent, queued, or handled as a
 * command), and restores once if a send is refused or throws before that point. These tests use the real
 * store with a stub transport, so no browser, host or engine is needed.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const here = (path) => fileURLToPath(new URL(path, import.meta.url));
const reactStub = here('./helpers/react-stub.mjs');
const storeEntry = here('../src/lib/store.ts');

function memoryStorage() {
  const values = new Map();
  return {
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => { values.set(key, String(value)); },
    removeItem: key => { values.delete(key); },
    clear: () => values.clear(),
  };
}

function installBrowser() {
  globalThis.window = globalThis;
  globalThis.document = {
    addEventListener() {},
    removeEventListener() {},
    visibilityState: 'visible',
    documentElement: { dataset: {}, style: { setProperty() {} }, classList: { toggle() {}, add() {}, remove() {} } },
  };
  Object.defineProperty(globalThis, 'location', { configurable: true, writable: true, value: { hash: '' } });
  globalThis.localStorage = memoryStorage();
  globalThis.sessionStorage = memoryStorage();
  globalThis.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} });
}

const stubClient = {
  name: 'stub-client',
  setup(build) {
    build.onResolve({ filter: /\/client$/ }, () => ({ path: 'stub-client', namespace: 'stub' }));
    build.onLoad({ filter: /.*/, namespace: 'stub' }, () => ({
      loader: 'js',
      contents: `
        export class ApiError extends Error {
          constructor(message, status) { super(message); this.status = status; }
        }
        export const api = {
          get: async () => ({}),
          post: async (path, body) => {
            if (path === '/api/chat/queue' && globalThis.__strataQueue === 'busy') throw new ApiError('A turn is running.', 409);
            return globalThis.__strataPost ? globalThis.__strataPost(path, body) : {};
          },
          put: async () => ({}),
          patch: async () => ({}),
          del: async () => ({}),
          delete: async () => ({}),
        };
        export function onEvent() { return () => {}; }
        export async function streamPost(path, body, onFrame) {
          if (globalThis.__strataStream === 'throw') throw new Error('engine offline');
          onFrame({ type: 'start', model: 'stub', contextMax: 8192, requestId: 'r1' });
          onFrame({ type: 'delta', text: 'An answer.' });
          onFrame({ type: 'done', tokensIn: 3, tokensOut: 3, durationMs: 1, usageSource: 'reported' });
        }`,
    }));
  },
};

let instances = 0;
async function loadStore() {
  globalThis.localStorage.clear();
  globalThis.sessionStorage.clear();
  delete globalThis.__strataPost;
  delete globalThis.__strataQueue;
  delete globalThis.__strataStream;
  const result = await build({
    entryPoints: [storeEntry],
    bundle: true,
    format: 'esm',
    platform: 'node',
    write: false,
    logLevel: 'silent',
    alias: { react: reactStub },
    loader: { '.json': 'json' },
    plugins: [stubClient],
  });
  // A unique suffix keeps Node's data-URL module cache from handing a later test the first store.
  const code = `${result.outputFiles[0].text}\n// store instance ${++instances}\n`;
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
}

/** A stand-in for the screen's draft and quote bar: the same clear and restore the chat performs. */
function composer() {
  const state = { draft: 'Hold this thought', quotes: [{ id: 'q1', text: 'A quoted line' }] };
  const saved = structuredClone(state);
  const calls = { clear: 0, restore: 0 };
  return {
    state,
    saved,
    calls,
    handlers: {
      clear() { calls.clear++; state.draft = ''; state.quotes = []; },
      restore() { calls.restore++; state.draft = saved.draft; state.quotes = saved.quotes; },
    },
  };
}

const messagesOf = (store, id) => store.getState().sessions.find(session => session.id === id)?.messages ?? [];

test.before(() => installBrowser());

test('a send refused for a model that is not ready leaves the draft and the quotes untouched', async (t) => {
  const store = await loadStore();
  const session = store.newSession();
  store.selectSession(session.id);
  const live = store.getState();
  live.connection = 'live';
  live.library = { variants: [], installed: [], loaded: null, current: null };
  live.remoteModel = null;
  t.after(() => store.flushSessions());

  const box = composer();
  const send = store.sendMessage('Hold this thought');
  const outcome = await store.settleComposerSend(send, box.handlers);
  await send;

  assert.equal(outcome.accepted, false);
  assert.equal(outcome.status, 'rejected');
  assert.equal(box.calls.clear, 0, 'the box is never cleared for a refusal');
  assert.equal(box.calls.restore, 1, 'a refusal puts the saved text back exactly once');
  assert.deepEqual(box.state, box.saved);
  assert.equal(messagesOf(store, session.id).length, 0, 'no message joins the chat');
});

test('/loop is handled, not refused: the box clears once and is not put back twice', async (t) => {
  const store = await loadStore();
  const session = store.newSession();
  store.selectSession(session.id);
  t.after(() => store.flushSessions());

  const box = composer();
  const send = store.sendMessage('/loop status');
  const outcome = await store.settleComposerSend(send, box.handlers);
  await send;

  assert.equal(outcome.accepted, true);
  assert.equal(outcome.status, 'handled');
  assert.equal(box.calls.clear, 1);
  assert.equal(box.calls.restore, 0, 'a command is not a failure, so nothing is restored');
  assert.deepEqual(box.state, { draft: '', quotes: [] });
  const messages = messagesOf(store, session.id);
  assert.equal(messages.filter(message => message.role === 'user').length, 0, 'the command is not a user message');
  assert.equal(messages.filter(message => message.local).length, 1, 'the command leaves its one note');
});

test('an accepted send clears the draft and the quotes exactly once', async (t) => {
  const store = await loadStore();
  const session = store.newSession();
  store.selectSession(session.id);
  t.after(() => store.flushSessions());

  const box = composer();
  const send = store.sendMessage('Hold this thought');
  const outcome = await store.settleComposerSend(send, box.handlers);
  await send;

  assert.equal(outcome.accepted, true);
  assert.equal(outcome.status, 'sent');
  assert.equal(box.calls.clear, 1);
  assert.equal(box.calls.restore, 0);
  assert.deepEqual(box.state, { draft: '', quotes: [] });
  const messages = messagesOf(store, session.id);
  assert.equal(messages.filter(message => message.role === 'user' && message.content === 'Hold this thought').length, 1);
});

test('a turn that fails after acceptance keeps the sent message and never puts the text back', async (t) => {
  const store = await loadStore();
  const session = store.newSession();
  store.selectSession(session.id);
  globalThis.__strataStream = 'throw';
  t.after(() => {
    delete globalThis.__strataStream;
    store.flushSessions();
  });

  const box = composer();
  const send = store.sendMessage('Hold this thought');
  const outcome = await store.settleComposerSend(send, box.handlers);
  await send;

  assert.equal(outcome.accepted, true);
  assert.equal(outcome.status, 'sent');
  assert.equal(box.calls.clear, 1);
  assert.equal(box.calls.restore, 0, 'the message is already in the chat, so the box is not put back');
  assert.deepEqual(box.state, { draft: '', quotes: [] });
  const messages = messagesOf(store, session.id);
  assert.equal(messages.filter(message => message.role === 'user' && message.content === 'Hold this thought').length, 1, 'the user message appears once');
  assert.ok(messages.some(message => message.role === 'assistant' && message.error), 'the failure is shown on the answer');
});

test('a message waiting in the queue counts as accepted and clears the box', async (t) => {
  const store = await loadStore();
  const session = store.newSession();
  store.selectSession(session.id);
  const live = store.getState();
  live.streaming = true;
  globalThis.__strataPost = path => (path === '/api/chat/queue' ? { queue: [] } : {});
  t.after(() => {
    live.streaming = false;
    delete globalThis.__strataPost;
    store.flushSessions();
  });

  const box = composer();
  const send = store.sendMessage('Wait in line');
  const outcome = await store.settleComposerSend(send, box.handlers);
  await send;

  assert.equal(outcome.accepted, true);
  assert.equal(outcome.status, 'queued');
  assert.equal(box.calls.clear, 1);
  assert.equal(box.calls.restore, 0);
  assert.deepEqual(box.state, { draft: '', quotes: [] });
  assert.equal(messagesOf(store, session.id).length, 0, 'it waits outside the transcript');
});

test('a message the queue refuses leaves the draft and the quotes alone', async (t) => {
  const store = await loadStore();
  const session = store.newSession();
  store.selectSession(session.id);
  const live = store.getState();
  live.streaming = true;
  globalThis.__strataQueue = 'busy';
  t.after(() => {
    live.streaming = false;
    delete globalThis.__strataQueue;
    store.flushSessions();
  });

  const box = composer();
  const send = store.sendMessage('Hold this thought');
  const outcome = await store.settleComposerSend(send, box.handlers);
  await send;

  assert.equal(outcome.accepted, false);
  assert.equal(outcome.status, 'rejected');
  assert.equal(box.calls.clear, 0);
  assert.equal(box.calls.restore, 1);
  assert.deepEqual(box.state, box.saved);
  assert.equal(messagesOf(store, session.id).filter(message => message.role === 'user').length, 0);
});

test('a send that throws before it is accepted restores once and adds no user message', async (t) => {
  const store = await loadStore();
  const session = store.newSession();
  store.selectSession(session.id);
  const live = store.getState();
  live.connection = 'live';
  live.remoteModel = null;
  // Fault injection: reading the model list throws before the user message can be added.
  Object.defineProperty(live, 'library', { configurable: true, get() { throw new Error('model list could not be read'); } });
  t.after(() => store.flushSessions());

  const box = composer();
  const send = store.sendMessage('Hold this thought');
  const outcome = await store.settleComposerSend(send, box.handlers);
  await assert.rejects(send, /model list could not be read/);

  assert.equal(outcome.accepted, false);
  assert.equal(outcome.status, 'rejected');
  assert.equal(box.calls.clear, 0);
  assert.equal(box.calls.restore, 1, 'a failed send restores once');
  assert.deepEqual(box.state, box.saved);
  assert.equal(messagesOf(store, session.id).filter(message => message.role === 'user').length, 0, 'the user message never appears');
});

test('the chat clears only after sendMessage settles, and writes only the chat the text came from', async () => {
  const chat = (await readFile(here('../src/screens/Chat.tsx'), 'utf8')).replace(/\r\n/g, '\n');
  const start = chat.indexOf('function submit()');
  const submit = chat.slice(start, chat.indexOf('function onComposerKey', start));

  assert.doesNotMatch(submit, /updateDraft\(''\);\s*clearQuotes\(\);\s*void sendMessage/, 'the old clear-before-send is gone');
  assert.ok(submit.indexOf('const send = sendMessage(text') > 0);
  assert.ok(submit.indexOf('void settleComposerSend(send') > submit.indexOf('const send = sendMessage(text'));
  assert.match(submit, /clear: \(\) => clearSentComposer\(fromSession, fromDraftKey, savedDraft, savedQuotes\)/);
  assert.match(submit, /restore: \(\) => restoreComposer\(fromSession, fromDraftKey, savedDraft, savedQuotes\)/);
  assert.doesNotMatch(submit, /setTimeout/, 'the verdict is awaited, never guessed with a timer');
  assert.match(submit, /if \(ok\)\s*clearSentComposer\(fromSession, fromDraftKey, savedDraft, savedQuotes\)/, 'a queued message clears only after the queue took it');

  assert.match(chat, /function clearSentComposer\(sessionId: string \| null, key: string/);
  assert.match(chat, /const left = readQuotes\(sessionId\)/);
  assert.match(chat, /writeQuotes\(sessionId, left\)/);
  assert.match(chat, /if \(sessionIdRef\.current !== sessionId\)\s*return;/, 'another open chat is not touched');
  assert.match(chat, /function restoreComposer\(sessionId: string \| null, key: string/);
  assert.match(chat, /writeQuotes\(sessionId, restored\)/);
  assert.match(chat, /if \(stored !== '' \|\| current\.length\)\s*return;/, 'a refusal never overwrites newer words or quotes');
  assert.match(chat, /if \(sessionIdRef\.current === sessionId\)\s*setDraft\(current => current === '' \? sentDraft : current\)/);

  const store = (await readFile(here('../src/lib/store.ts'), 'utf8')).replace(/\r\n/g, '\n');
  const loopStart = store.indexOf('if (loopCommand)');
  const loopBlock = store.slice(loopStart, store.indexOf('const demoPrompt', loopStart));
  assert.ok(loopBlock.indexOf('const handled = settle(') < loopBlock.indexOf('await runLoopCommand('), 'a loop command is accepted before its first answer runs');
  assert.match(loopBlock, /return handled;/);
});
