/** Exercise the real desktop store, its streaming callback, and the durable host catalog together. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createConversations } from '../server/conversations.mjs';

const { outputFiles } = await build({ entryPoints: [fileURLToPath(new URL('../src/lib/store.ts', import.meta.url))], bundle: true, write: false, platform: 'node', format: 'cjs', plugins: [{ name: 'controlled-client', setup(builder) {
  builder.onResolve({ filter: /(?:^|\/)client$/ }, args => args.importer.includes('/src/') ? { path: 'controlled-client', namespace: 'controlled-client' } : undefined);
  builder.onLoad({ filter: /.*/, namespace: 'controlled-client' }, () => ({ contents: `export const api=harness.api; export const onEvent=(...args)=>harness.onEvent(...args); export const streamPost=(...args)=>harness.streamPost(...args); export class ApiError extends Error {}`, loader: 'js' }));
} }] });
const tick = () => new Promise(r => setImmediate(r));
async function until(predicate) { for (let n = 0; n < 100; n++) { if (predicate()) return; await new Promise(r => setTimeout(r, 10)); } throw new Error('State did not settle'); }
const at = '2026-10-20T20:00:00.000Z';

async function fixture(t) {
  const directory = mkdtempSync(join(tmpdir(), 'strata-deferred-migration-'));
  let enabled = false, listener, stream, blockedSync = null;
  const calls = [];
  const catalog = createConversations({ directory, enabled: () => enabled });
  const session = { id: 'mac-active-before-pairing', title: 'Existing Mac conversation', workspace: '/fixture/project', createdAt: at, updatedAt: at, tokensIn: 0, tokensOut: 0,
    preferences: { mode: 'manual', agent: false, remoteModel: { connectionId: 'fixture-api', modelId: 'fixture-model' } },
    messages: [{ id: 'before-user', role: 'user', content: 'Earlier question', at }, { id: 'before-answer', role: 'assistant', content: 'Earlier answer', at }] };
  const values = new Map([['strata.sessions.v1', JSON.stringify([session])]]);
  const api = async (method, path, body) => {
    calls.push({ method, path, body: structuredClone(body) });
    if (path === '/api/companion/conversations') return catalog.list();
    if (path === '/api/companion/conversations/sync') { if (blockedSync) await blockedSync; return catalog.sync(body.sessions); }
    if (path.startsWith('/api/companion/conversations/')) {
      if (path.endsWith('/stop')) return catalog.stop(path.split('/').at(-2));
      return catalog.detail(path.split('/').at(-1));
    }
    const fixtures = { '/api/setup': { complete: true }, '/api/models': { variants: [] }, '/api/skills': [], '/api/workspaces': { workspaces: [{ id: 'w', path: '/fixture/project', name: 'Project' }] }, '/api/runtime': { instances: [] }, '/api/serving': { stats: { running: 0, waiting: 0 }, revision: 1 }, '/api/model-providers': { connections: [{ id: 'fixture-api', protocol: 'openai-chat', models: [{ id: 'fixture-model' }] }] } };
    return fixtures[path] ?? {};
  };
  const harness = { api: Object.fromEntries(['get', 'post', 'patch', 'put', 'del'].map(method => [method, (path, body) => api(method, path, body)])),
    onEvent: handler => { listener = handler; return () => {}; }, streamPost: (path, body, onFrame) => new Promise((resolve, reject) => { stream = { onFrame, resolve, reject }; onFrame({ type: 'delta', text: 'Partial ' }); }),
  };
  const module = { exports: {} };
  new Function('require', 'module', 'exports', 'harness', 'localStorage', 'sessionStorage', 'window', 'document', 'matchMedia', 'navigator', outputFiles[0].text)(createRequire(import.meta.url), module, module.exports, harness,
    { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) }, { getItem: () => null, setItem() {} }, { location: { hash: '#/chat/mac-active-before-pairing' } },
    { documentElement: { dataset: {}, style: { setProperty() {}, removeProperty() {} } }, querySelector: () => null }, () => ({ matches: false, addEventListener() {} }), { language: 'en' });
  const store = module.exports;
  await store.bootstrap(); store.selectSession(session.id);
  assert.equal(store.getState().connection, 'live');
  t.after(async () => { await new Promise(r => setTimeout(r, 350)); for (const toast of store.getState().toasts) store.holdToast(toast.id); catalog.close(); rmSync(directory, { recursive: true, force: true }); });
  return { store, catalog, calls, id: session.id, get stream() { return stream; }, enable: () => { enabled = true; listener({ event: 'companion', data: { available: true } }); }, emitConversation: () => listener({ event: 'conversation', data: catalog.detail(session.id) }), blockImport() { let resolve; blockedSync = new Promise(r => { resolve = r; }); return () => { blockedSync = null; resolve(); }; } };
}

test('first pairing during a Mac reply defers its history until the complete result is available', async t => {
  const f = await fixture(t);
  const answering = f.store.sendMessage('Continue before pairing'); await until(() => f.stream);
  assert.equal(f.store.getState().sessions[0].messages.at(-1).content, 'Partial ');
  f.enable(); await until(() => f.calls.some(c => c.path.endsWith('/sync'))); await tick();
  assert.equal(f.catalog.list().conversations.length, 0, 'the unfinished, untracked chat must not appear idle or continuable on the phone');
  assert.equal(f.calls.find(c => c.path.endsWith('/sync')).body.sessions.length, 0);
  f.stream.onFrame({ type: 'delta', text: 'finished answer' }); f.stream.onFrame({ type: 'done', tokensIn: 2, tokensOut: 3 }); f.stream.resolve(); await answering;
  const saved = f.catalog.detail(f.id).conversation;
  assert.equal(saved.messages.length, 4);
  assert.equal(saved.messages.at(-1).content, 'Partial finished answer');
  assert.equal(saved.messages.at(-1).streaming, false);
  assert.equal(f.store.getState().sessions[0].messages.at(-1).content, 'Partial finished answer');
});

test('a reply ending while the first import is awaiting its response is saved after initialization', async t => {
  const f = await fixture(t);
  const answering = f.store.sendMessage('Continue before pairing'); await until(() => f.stream);
  const releaseImport = f.blockImport(); f.enable(); await until(() => f.calls.some(c => c.path.endsWith('/sync')));
  f.stream.onFrame({ type: 'delta', text: 'completed during import' }); f.stream.onFrame({ type: 'done' }); f.stream.resolve(); await answering;
  releaseImport(); await until(() => f.catalog.list().conversations.length === 1);
  assert.equal(f.catalog.detail(f.id).conversation.messages.at(-1).content, 'Partial completed during import');
});

test('a pre-sharing reply cancelled after enabling keeps partial text with streaming cleared', async t => {
  const f = await fixture(t);
  const answering = f.store.sendMessage('Continue before pairing'); await until(() => f.stream);
  f.enable(); await until(() => f.calls.some(c => c.path.endsWith('/sync'))); await tick();
  f.stream.reject(Object.assign(new Error('Cancelled'), { name: 'AbortError' })); await answering;
  const saved = f.catalog.detail(f.id).conversation.messages.at(-1);
  assert.equal(saved.content, 'Partial '); assert.equal(saved.streaming, false);
});

test('the task board follows and stops a phone-started chat and retains its dismissal after sharing', async t => {
  const f = await fixture(t);
  f.enable(); await until(() => f.catalog.list().conversations.length === 1);
  f.catalog.begin(f.id, 'Phone continuation');
  f.emitConversation();
  assert.deepEqual(f.store.liveChatIds(), [f.id]);
  assert.equal(f.store.getState().sessions.find(s => s.id === f.id).active, true);
  f.store.stopChat(f.id);
  await until(() => !f.store.liveChatIds().includes(f.id));
  assert.equal(f.calls.filter(c => c.path.endsWith('/stop')).length, 1);
  assert.equal(f.calls.some(c => c.path === '/api/chat/stop'), false);
  f.store.stopChat(f.id); await tick();
  assert.equal(f.calls.filter(c => c.path.endsWith('/stop')).length, 1, 'An ended shared chat cannot be stopped twice.');
  f.store.dismissChat(f.id);
  const ackedAt = f.store.getState().sessions.find(s => s.id === f.id).ackedAt;
  await until(() => f.catalog.detail(f.id).conversation.ackedAt === ackedAt);
  f.emitConversation();
  assert.equal(f.store.getState().sessions.find(s => s.id === f.id).ackedAt, ackedAt);
});
