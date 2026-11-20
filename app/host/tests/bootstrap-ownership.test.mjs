/** Real store, controlled HTTP/event boundary, fictional in-memory chats only. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {build} from 'esbuild';
import {fileURLToPath} from 'node:url';

const turn = () => new Promise(resolve => setImmediate(resolve));
const gate = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return {promise, resolve, reject}; };
const here = path => fileURLToPath(new URL(path, import.meta.url));
const answers = {
  '/api/setup': {complete: true}, '/api/models': {variants: [], loaded: null},
  '/api/engine': {beginner: {contextLength: '4k'}, expert: []}, '/api/serving': {revision: 1},
  '/api/workspaces': {workspaces: []}, '/api/hardware': {gpus: [], revision: 1},
  '/api/monitor': {phase: 'idle', revision: 1}, '/api/skills': [],
  '/api/model-providers': {connections: []}, '/api/image-models': {variants: []}, '/api/runtime': {instances: []},
  '/api/chat/active': {turns: []},
};
let sequence = 0;
async function fixture(t) {
  const events = [], reads = [], writes = [], pending = new Map();
  const memory = new Map([['strata.sessions.v1', JSON.stringify([{id: 'fictional', title: 'Garden notebook', messages: [{id:'note',role:'user',content:'Plan a fictional garden'}], goal:{objective:'Plan a fictional garden',phase:'active'}, updatedAt: '2026-11-07T00:00:00Z'}])]]);
  const drafts = new Map([['strata.chat-draft.fictional','A fictional unsent note']]);
  globalThis.window = globalThis;
  globalThis.location = {hash: '#/chat'};
  globalThis.document = {addEventListener() {}, querySelector: () => null, hasFocus: () => true, visibilityState: 'visible', documentElement: {dataset: {}, style: {setProperty() {}, removeProperty() {}}, classList: {toggle() {}, add() {}, remove() {}}}};
  globalThis.matchMedia = () => ({matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {}});
  globalThis.localStorage = {getItem: key => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, value), removeItem: key => memory.delete(key), key: i => [...memory.keys()][i], get length() { return memory.size; }};
  globalThis.sessionStorage = {getItem: key => drafts.get(key) ?? null, setItem: (key,value) => drafts.set(key,value), removeItem: key => drafts.delete(key)};
  globalThis.fetch = async () => { throw Error('fixture chat transport offline'); };
  const key = `__bootstrapFixture${++sequence}`;
  globalThis[key] = {
    get(path) { reads.push(path); return pending.get(path)?.shift()?.promise ?? Promise.resolve(structuredClone(answers[path] ?? {})); },
    post(path) { writes.push(path); return Promise.resolve({}); },
    subscribe(handler, status) { const event = {handler, status, closed: false}; events.push(event); return () => { event.closed = true; }; },
  };
  const result = await build({entryPoints: [here('../src/lib/store.ts')], bundle: true, write: false, format: 'esm', platform: 'node', alias: {react: here('./helpers/react-stub.mjs')}, plugins: [{name: 'controlled-client', setup(b) {
    b.onResolve({filter: /\/client$/}, () => ({path: 'client', namespace: 'fixture'}));
    b.onLoad({filter: /.*/, namespace: 'fixture'}, () => ({contents: `
      const c=globalThis[${JSON.stringify(key)}];
      export class ApiError extends Error {}
      export const api={get:c.get,post:c.post,put:async()=>({}),patch:async()=>({}),del:async()=>({})};
      export const onEvent=c.subscribe,streamPost=async()=>{};` }));
  }}]});
  const store = await import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text+'\n//# sourceURL=bootstrap-fixture-'+sequence+'.mjs').toString('base64')}`);
  Object.assign(store.bootstrapTiming, {timeoutMs: 2000, tries: 1});
  t.after(() => { for (const toast of store.getState().toasts) store.dismissToast(toast.id); delete globalThis[key]; });
  return {store, events, reads, writes, drafts, memory, defer(path) { const value = gate(); pending.set(path, [...(pending.get(path) ?? []), value]); return value; }};
}

test('HTTP bootstrap success is not evidence that EventSource opened', async t => {
  const {store, events} = await fixture(t);
  await store.bootstrap();
  assert.equal(store.getState().ready, true);
  assert.equal(store.getState().connection, 'connecting');
  events[0].status('live');
  assert.equal(store.getState().connection, 'live');
});

test('late bootstrap failure after newer success cannot change connection or add an error toast', async t => {
  const f = await fixture(t), olderEngine = f.defer('/api/engine');
  const old = f.store.bootstrap(); await turn();
  await f.store.bootstrap(); f.events.at(-1).status('live');
  f.store.setConfigDraft('fictional unsaved setting');
  const sessions = f.store.getState().sessions;
  assert.equal(sessions.length,1);assert.equal(sessions[0].goal.phase,'active');
  olderEngine.reject(Error('older request failed')); await old;
  assert.equal(f.store.getState().connection, 'live');
  assert.deepEqual(f.store.getState().toasts, []);
  assert.equal(f.store.getState().configDraft, 'fictional unsaved setting');
  assert.strictEqual(f.store.getState().sessions, sessions);
  assert.equal(f.drafts.get('strata.chat-draft.fictional'),'A fictional unsent note');
  assert.deepEqual(f.writes,['/api/shell/prefs'],'bootstrap must not start a model or cancel an ongoing goal');
});

test('late bootstrap success cannot replace the current stream or model snapshot', async t => {
  const f = await fixture(t), olderModels = f.defer('/api/models');
  const old = f.store.bootstrap(); await turn();
  await f.store.bootstrap(); const current = f.events.at(-1); current.status('live');
  olderModels.resolve({variants: [], loaded: 'obsolete'}); await old;
  assert.equal(f.store.getState().library.loaded, null);
  assert.equal(current.closed, false);
  assert.equal(f.events.length, 1);
});

test('optional reads from a superseded bootstrap cannot overwrite newer data', async t => {
  const f = await fixture(t);
  const oldReads = ['/api/hardware', '/api/monitor', '/api/skills', '/api/model-providers', '/api/image-models', '/api/runtime'].map(path => [path, f.defer(path)]);
  await f.store.bootstrap(); await f.store.bootstrap(); await turn();
  for (const [path, deferred] of oldReads) deferred.resolve({obsolete: path});
  await turn();
  for (const [field, path] of Object.entries({hardware: '/api/hardware', monitor: '/api/monitor', skills: '/api/skills', providers: '/api/model-providers', imageModels: '/api/image-models', runtime: '/api/runtime'})) assert.deepEqual(f.store.getState()[field], answers[path]);
});

test('fresh events and verified hardware refresh win over optional bootstrap reads', async t => {
  const f = await fixture(t), monitor = f.defer('/api/monitor'), hardware = f.defer('/api/hardware');
  await f.store.bootstrap();
  f.events[0].handler({event: 'monitor', data: {phase: 'generating', revision: 2}});
  await f.store.refreshHardware(); const verified = f.store.getState().hardware;
  monitor.resolve(answers['/api/monitor']); hardware.resolve({gpus: [], obsolete: true}); await turn();
  assert.equal(f.store.getState().monitor.revision, 2);
  assert.strictEqual(f.store.getState().hardware, verified);
});

test('revoked store subscription cannot write frames or launch another bootstrap', async t => {
  const f = await fixture(t);
  await f.store.bootstrap(); const old = f.events[0]; old.status('live');
  await f.store.bootstrap(); f.events.at(-1).status('live');
  const reads = f.reads.length;
  old.handler({event: 'monitor', data: {phase: 'error'}}); old.status('offline'); old.status('live'); await turn();
  assert.equal(f.store.getState().connection, 'live');
  assert.deepEqual(f.store.getState().monitor, answers['/api/monitor']);
  assert.equal(f.reads.length, reads);
});

test('real reconnect plus manual retry keeps the latest bootstrap as the only owner', async t => {
  const f = await fixture(t);
  await f.store.bootstrap();const old=f.events[0];old.status('live');old.status('offline');
  assert.equal(f.store.getState().connection,'offline');
  const deferred=f.defer('/api/engine');old.status('live');await turn();
  await f.store.bootstrap();f.events.at(-1).status('live');
  deferred.reject(Error('superseded automatic refresh'));await turn();
  assert.equal(f.store.getState().connection,'live');assert.deepEqual(f.store.getState().toasts,[]);
  assert.equal(f.events.filter(event=>!event.closed).length,1);
});

test('late initial chat hydration from a superseded bootstrap cannot replace newer conversations', async t => {
  const f=await fixture(t), oldList=gate();let lists=0,gets=0;
  f.memory.set('strata.chats.host.v1','1');
  globalThis.fetch=async(url,{method})=>{
    assert.equal(method,'GET','hydration must not write the fixture');
    if(url==='/api/chats')return {ok:true,json:async()=>++lists===1?oldList.promise:{chats:[{id:'fictional',rev:2}]}};
    if(url==='/api/chats/fictional')return {ok:true,json:async()=>({rev:++gets===1?2:1,session:{id:'fictional',title:gets===1?'Current garden':'Older garden',messages:[],updatedAt:gets===1?'2026-11-07T02:00:00Z':'2026-11-07T01:00:00Z'}})};
    throw Error('unexpected fixture URL');
  };
  const old=f.store.bootstrap();await turn();
  await f.store.bootstrap();f.events.at(-1).status('live');
  oldList.resolve({chats:[{id:'fictional',rev:1}]});await old;
  assert.equal(f.store.getState().sessions[0].title,'Current garden');
  assert.equal(f.store.getState().connection,'live');
});

test('late reconnect chat list cannot delete conversations kept by the newer bootstrap', async t => {
  const f=await fixture(t);f.memory.set('strata.chats.host.v1','1');
  const oldList=gate();let lists=0;
  globalThis.fetch=async(url,{method})=>{
    assert.equal(method,'GET');
    if(url==='/api/chats')return {ok:true,json:async()=>++lists===2?oldList.promise:{chats:[{id:'fictional',rev:2}]}};
    if(url==='/api/chats/fictional')return {ok:true,json:async()=>({rev:2,session:{id:'fictional',title:'Current garden',messages:[],updatedAt:'2026-11-07T02:00:00Z'}})};
    throw Error('unexpected fixture URL');
  };
  await f.store.bootstrap();
  const old=f.store.bootstrap();await turn();await f.store.bootstrap();
  oldList.resolve({chats:[]});await old;
  assert.equal(f.store.getState().sessions.length,1);
  assert.equal(f.store.getState().sessions[0].title,'Current garden');
});

test('a chat read already awaiting its body loses ownership with its bootstrap', async t => {
  const f=await fixture(t);f.memory.set('strata.chats.host.v1','1');
  const oldBody=gate();let revision=0,gets=0;
  const body=rev=>({rev,session:{id:'fictional',title:`Garden revision ${rev}`,messages:[],updatedAt:`2026-11-07T0${rev}:00:00Z`}});
  globalThis.fetch=async(url,{method})=>{
    assert.equal(method,'GET');
    if(url==='/api/chats')return {ok:true,json:async()=>({chats:[{id:'fictional',rev:++revision}]})};
    if(url==='/api/chats/fictional')return {ok:true,json:async()=>++gets===2?oldBody.promise:body(revision)};
    throw Error('unexpected fixture URL');
  };
  await f.store.bootstrap();const old=f.store.bootstrap();await turn();await f.store.bootstrap();
  oldBody.resolve(body(2));await old;
  assert.equal(f.store.getState().sessions[0].title,'Garden revision 3');
});
