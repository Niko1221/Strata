/**
 * F01 / F02: a turn keeps the session, remote model and workspace that owned it,
 * even if the open chat changes while an await is in flight or a queued-release
 * fires after the turn ends.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));

function installDom() {
  globalThis.window ??= globalThis;
  globalThis.document ??= {
    addEventListener() {},
    visibilityState: 'visible',
    documentElement: { dataset: {}, style: { setProperty() {} }, classList: { toggle() {}, add() {}, remove() {} } },
  };
  globalThis.navigator ??= { language: 'en' };
  globalThis.matchMedia ??= () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} });
  globalThis.sessionStorage ??= {
    getItem() { return null; },
    setItem() {},
    removeItem() {},
  };
}

function seedStorage(sessions) {
  const memory = new Map();
  memory.set('strata.sessions.v1', JSON.stringify(sessions));
  globalThis.localStorage = {
    getItem: (k) => memory.get(k) ?? null,
    setItem: (k, v) => { memory.set(k, v); },
    removeItem: (k) => { memory.delete(k); },
  };
}

async function loadStore({ photoGate, capture, streamScript }) {
  installDom();
  const mockClient = {
    name: 'mock-client',
    setup(b) {
      b.onResolve({ filter: /\/client$/ }, () => ({ path: 'mock-client', namespace: 'mock' }));
      b.onLoad({ filter: /.*/, namespace: 'mock' }, () => ({
        loader: 'js',
        contents: streamScript ?? `
          export class ApiError extends Error {}
          const reject = async () => { throw new Error('offline'); };
          export const api = {
            get: async () => ({ path: '', entries: [] }),
            post: async () => ({}),
            put: reject, patch: reject, del: reject, delete: reject,
          };
          export function onEvent() { return () => {}; }
          export async function streamPost(path, body, onFrame) {
            globalThis.__streamBodies.push({ path, body: structuredClone(body) });
            onFrame({ type: 'start', model: 'm', contextMax: 8192, requestId: 'r1' });
            onFrame({ type: 'done', tokensIn: 1, tokensOut: 1, durationMs: 1, usageSource: 'reported' });
          }`,
      }));
    },
  };
  const mockPhotos = {
    name: 'mock-photos',
    setup(b) {
      b.onResolve({ filter: /\/chat-photos$/ }, () => ({ path: 'mock-photos', namespace: 'photos' }));
      b.onLoad({ filter: /.*/, namespace: 'photos' }, () => ({
        loader: 'js',
        contents: `
          export function takeChatPhotos() { return []; }
          export function pendingChatPhotos() { return []; }
          export async function uploadChatPhotos() { return []; }
          export function photoHistorySlots(messages) {
            const slots = new Set();
            for (let m = 0; m < messages.length; m++) {
              const photos = messages[m].photos ?? [];
              for (let p = 0; p < photos.length; p++) slots.add(m + ':' + p);
            }
            return slots;
          }
          export async function photoData(photo) {
            await new Promise((resolve) => {
              globalThis.__photoWaiters = globalThis.__photoWaiters || [];
              globalThis.__photoWaiters.push(resolve);
            });
            return { name: photo.name, url: 'data:image/png;base64,aa' };
          }`,
      }));
    },
  };
  globalThis.__streamBodies = capture;
  globalThis.__photoWaiters = [];
  const result = await build({
    entryPoints: [here('../src/lib/store.ts')],
    bundle: true,
    format: 'esm',
    platform: 'node',
    write: false,
    logLevel: 'silent',
    alias: { react: here('./helpers/react-stub.mjs') },
    loader: { '.json': 'json' },
    plugins: photoGate ? [mockClient, mockPhotos] : [mockClient],
  });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}

async function waitFor(predicate, label, ms = 2000) {
  const start = Date.now();
  while (Date.now() - start < ms) {
    if (predicate()) return;
    await new Promise((r) => setTimeout(r, 5));
  }
  throw new Error(`timed out waiting for ${label}`);
}

const remoteA = { connectionId: 'conn-a', modelId: 'model-a' };
const remoteB = { connectionId: 'conn-b', modelId: 'model-b' };
const now = () => new Date().toISOString();

test('F01: mid-await chat switch keeps session A remoteModel and workspace on the request', async () => {
  seedStorage([
    {
      id: 'sess-a',
      title: 'Chat A',
      defaultTitle: false,
      createdAt: now(),
      updatedAt: now(),
      messages: [{
        id: 'm-hist',
        role: 'user',
        content: 'see this picture',
        at: now(),
        photos: [{ id: 'photo-1', name: 'a.png' }],
      }],
      workspace: 'C:/ws-a',
      tokensIn: 0,
      tokensOut: 0,
      preferences: { remoteModel: remoteA },
    },
    {
      id: 'sess-b',
      title: 'Chat B',
      defaultTitle: false,
      createdAt: now(),
      updatedAt: now(),
      messages: [],
      workspace: 'C:/ws-b',
      tokensIn: 0,
      tokensOut: 0,
      preferences: { remoteModel: remoteB },
    },
  ]);

  const capture = [];
  const store = await loadStore({ photoGate: true, capture });
  store.selectSession('sess-a');
  store.setActiveWorkspace('ws-a');
  // Mirror the open chat's model into global state the way the UI does.
  assert.deepEqual(store.getState().remoteModel, remoteA);

  const send = store.sendMessage('follow-up from A');
  await waitFor(() => (globalThis.__photoWaiters?.length ?? 0) > 0, 'photoData gate');

  store.selectSession('sess-b');
  store.setActiveWorkspace('ws-b');
  assert.equal(store.getState().activeSession, 'sess-b');
  assert.deepEqual(store.getState().remoteModel, remoteB);
  assert.equal(store.getState().activeWorkspace, 'ws-b');

  for (const release of globalThis.__photoWaiters.splice(0)) release();
  await send;

  assert.equal(capture.length, 1, 'one stream request');
  const body = capture[0].body;
  assert.equal(body.sessionId, 'sess-a', 'session stays A');
  assert.deepEqual(body.remoteModel, remoteA, 'remote model stays A after B was selected mid-await');
  assert.equal(body.workspaceId, 'ws-a', 'workspace stays A after B was selected mid-await');
});

test('F02: queued-release resends to the owning session, not the open chat', async () => {
  seedStorage([
    {
      id: 'sess-a',
      title: 'Chat A',
      defaultTitle: false,
      createdAt: now(),
      updatedAt: now(),
      messages: [],
      workspace: 'C:/ws-a',
      tokensIn: 0,
      tokensOut: 0,
      preferences: { remoteModel: remoteA },
    },
    {
      id: 'sess-b',
      title: 'Chat B',
      defaultTitle: false,
      createdAt: now(),
      updatedAt: now(),
      messages: [],
      workspace: 'C:/ws-b',
      tokensIn: 0,
      tokensOut: 0,
      preferences: { remoteModel: remoteB },
    },
  ]);

  const capture = [];
  const streamScript = `
    export class ApiError extends Error {}
    const reject = async () => { throw new Error('offline'); };
    export const api = {
      get: async () => ({ path: '', entries: [] }),
      post: async () => ({}),
      put: reject, patch: reject, del: reject, delete: reject,
    };
    export function onEvent() { return () => {}; }
    export async function streamPost(path, body, onFrame) {
      globalThis.__streamBodies.push({ path, body: structuredClone(body) });
      onFrame({ type: 'start', model: 'm', contextMax: 8192, requestId: 'r1' });
      if (!globalThis.__releasedOnce) {
        globalThis.__releasedOnce = true;
        onFrame({ type: 'queued-release', messages: [{ text: 'queued for A' }] });
      }
      onFrame({ type: 'done', tokensIn: 1, tokensOut: 1, durationMs: 1, usageSource: 'reported' });
    }`;
  globalThis.__releasedOnce = false;
  const store = await loadStore({ photoGate: false, capture, streamScript });
  store.selectSession('sess-a');
  store.setActiveWorkspace('ws-a');

  await store.sendMessage('start A');
  assert.equal(capture.length, 1);
  assert.equal(capture[0].body.sessionId, 'sess-a');

  // Switch to B before the queued-release timer fires its follow-up send.
  store.selectSession('sess-b');
  store.setActiveWorkspace('ws-b');
  assert.deepEqual(store.getState().remoteModel, remoteB);

  await new Promise((r) => setTimeout(r, 30));
  await waitFor(() => capture.length >= 2, 'queued-release follow-up send');

  const follow = capture[1].body;
  assert.equal(follow.sessionId, 'sess-a', 'queued text returns to A, not open chat B');
  assert.deepEqual(follow.remoteModel, remoteA, 'follow-up keeps A remote model');
  assert.match(String(follow.message ?? ''), /queued for A/);
});
