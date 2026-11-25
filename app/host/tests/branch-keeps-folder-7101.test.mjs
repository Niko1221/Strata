/**
 * #7101: "Ask again", Edit and "Branch from here" made a new chat id, and the host keeps a chat's working folder by chat id, so the copy started in the default folder.
 * The copy now gets the folder of the chat it came from (and "Ask again" waits for that before it sends).
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const here = p => fileURLToPath(new URL(p, import.meta.url));
let loads = 0;

/** A fresh copy of the store (as chat-defaults.test.mjs does) with a host that keeps per-chat folders and can be told to answer late. */
async function load({ folders = {}, failSet = false, lateMs = 0 } = {}) {
  const memory = new Map();
  globalThis.window ??= globalThis;
  globalThis.window.location ??= { hash: '' };
  globalThis.document = {
    addEventListener() {}, visibilityState: 'visible',
    querySelector: () => null, querySelectorAll: () => [],
    documentElement: { dataset: {}, style: { setProperty() {}, removeProperty() {} }, classList: { toggle() {}, add() {}, remove() {} } },
  };
  globalThis.localStorage = { getItem: k => memory.get(k) ?? null, setItem: (k, v) => void memory.set(k, v), removeItem: k => void memory.delete(k) };
  globalThis.sessionStorage = { getItem: () => null, setItem() {}, removeItem() {} };
  globalThis.navigator ??= { language: 'en' };
  globalThis.matchMedia ??= () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} });
  const host = { folders: { ...folders }, posts: [], gets: [] };
  globalThis.__host7101 = host;
  const mockClient = {
    name: 'mock-client',
    setup(b) {
      b.onResolve({ filter: /\/client$/ }, () => ({ path: 'mock-client', namespace: 'mock' }));
      b.onLoad({ filter: /.*/, namespace: 'mock' }, () => ({
        loader: 'js',
        contents: `const h = () => globalThis.__host7101;
          const info = p => (p ? { path: p, name: p.split('/').pop(), exists: true } : null);
          export class ApiError extends Error {}
          const reject = async () => { throw new Error('offline'); };
          export const api = {
            get: async path => { h().gets.push(path); const m = /sessionId=([^&]+)/.exec(path); if (path.startsWith('/api/chat/folder') && m) return { folder: info(h().folders[decodeURIComponent(m[1])] ?? null) }; throw new Error('offline'); },
            post: async (path, body) => {
              if (path === '/api/chat/folder') {
                h().posts.push(body);
                if (${failSet}) throw new Error('This folder is not allowed.');
                if (${lateMs}) await new Promise(r => setTimeout(r, ${lateMs}));
                h().folders[body.sessionId] = body.path;
                return { folder: info(body.path) };
              }
              return {};
            },
            put: reject, patch: reject, del: reject, delete: reject,
          };
          export function onEvent() { return () => {}; }
          export async function streamPost() {}`,
      }));
    },
  };
  const result = await build({ entryPoints: [here('../src/lib/store.ts')], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent', alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' }, plugins: [mockClient] });
  const salted = `${result.outputFiles[0].text}\n/* strata-store-copy-7101 ${++loads} */\n`;
  const store = await import(`data:text/javascript;base64,${Buffer.from(salted).toString('base64')}`);
  const chatFolder = await import(`data:text/javascript;base64,${Buffer.from((await build({ entryPoints: [here('../src/lib/chat-folder.ts')], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent', alias: { react: here('./helpers/react-stub.mjs') }, plugins: [mockClient] })).outputFiles[0].text + `\n/* c ${loads} */`).toString('base64')}`);
  return { store, host, chatFolder };
}

/** A chat with a question and an answer, working in `folder` on the host. */
function chatWithTurn(store, host, folder) {
  const chat = store.newSession();
  chat.messages.push({ id: 'm-user', role: 'user', content: 'Fix the bug in main.c', createdAt: new Date().toISOString() },
    { id: 'm-bot', role: 'assistant', content: 'Done.', createdAt: new Date().toISOString() });
  if (folder) host.folders[chat.id] = folder;
  return chat;
}
const settle = () => new Promise(r => setTimeout(r, 30));

test('Branch from here: the branch works in the folder of the chat it came from', async () => {
  const { store, host } = await load();
  const chat = chatWithTurn(store, host, 'C:/projects/app');
  const branchId = store.branchFrom('m-bot');
  assert.ok(branchId && branchId !== chat.id, 'a new chat id');
  await settle();
  assert.deepEqual(host.posts, [{ sessionId: branchId, path: 'C:/projects/app' }]);
  assert.equal(host.folders[branchId], 'C:/projects/app');
  assert.equal(host.folders[chat.id], 'C:/projects/app', 'the original keeps its folder');
});

test('Edit and Ask again (rewindTo) branch the same way, so they keep the folder too', async () => {
  const { store, host } = await load();
  chatWithTurn(store, host, 'C:/projects/app');
  const prompt = store.rewindTo('m-user');
  assert.equal(prompt, 'Fix the bug in main.c');
  await settle();
  const branchId = store.getState().activeSession;
  assert.equal(host.folders[branchId], 'C:/projects/app');
});

test('a chat in the default folder gives a branch in the default folder: nothing is set', async () => {
  const { store, host } = await load();
  chatWithTurn(store, host, null);
  store.branchFrom('m-bot');
  await settle();
  assert.deepEqual(host.posts, []);
});

test('a folder the host refuses for the copy leaves the branch in the default folder and breaks nothing', async () => {
  const { store, host } = await load({ failSet: true });
  chatWithTurn(store, host, 'C:/projects/app');
  const branchId = store.branchFrom('m-bot');
  assert.ok(branchId);
  await settle();
  assert.equal(host.posts.length, 1, 'it was tried once');
  assert.equal(host.folders[branchId], undefined);
});

test('Ask again waits for the folder of the copy before it sends the question', async () => {
  const { chatFolder, host } = await load({ lateMs: 80 });
  host.folders.src = 'C:/projects/app';
  const copy = chatFolder.copyChatFolder('src', 'dst');
  const waited = chatFolder.folderCopied('dst');
  assert.equal(host.folders.dst, undefined, 'the host has not answered yet');
  await waited;
  assert.equal(host.folders.dst, 'C:/projects/app', 'by the time Ask again sends, the host knows the folder');
  await copy;
  await chatFolder.folderCopied('nothing-pending');
});
