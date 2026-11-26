import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createApi, actionBody } from './ui/api.mjs';
import { modelView, contextLabel, idleLabel, compatible, createPanelController } from './ui/state.mjs';
import { renderPanel, startPanel } from './ui/panel.mjs';
import { createDialogController, createSelectionCoordinator, detectTheme, frameMessage, mountExtension } from './ui/entry.mjs';
import { connectedSelection, pickerOption, selectInOfficialChat, waitForDom } from './ui/official.mjs';
import { observeConnectedModels } from './ui/connected.mjs';

const ORIGIN = 'http://127.0.0.1:47831';
// Test data only. Production options always come from the bridge's native catalog.
const catalogChoices = [8192, 32768, 65536, 131072, 262144, 393216, 524288];
const status = { enabled: true, compatible: true, bridgeApi: 1, extension: { id: 'strata-local-models', version: '1.0.0' } };
const model = () => ({ available: true,
  model: { id: 'physical-model-id', name: 'Shared Qwen', variant: 'IQ3_XXS', loaded: false, loading: false, apiBaseUrl: 'http://127.0.0.1:8080/v1' },
  context: { choices: catalogChoices, selected: 131072, effective: 131072, phase: 'ready', activeRequests: 0, inFlight: 0 },
  autoUnload: { selected: 1800, effective: 1800 },
});
const connection = () => ({ selection: { connectionId: 'carrier-connection', modelId: 'physical-model-id' },
  label: 'Shared Qwen', connectionName: 'Local Carrier', contextMax: 131072, changed: false });
const tick = () => new Promise(resolve => setImmediate(resolve));
function panelFixture(t, { api: override, openChat } = {}) {
  const calls = [], states = [], timers = new Map(); let id = 0, current = model();
  const api = override ?? { async get(action, signal) { calls.push({ action, signal, method: 'GET' }); return action === 'status' ? status : structuredClone(current); },
    async post(action, value, signal) { calls.push({ action, value, signal, method: 'POST' }); return action === 'connect' ? connection() : { status: 'requested' }; } };
  const controller = createPanelController({ api, onState: value => states.push(value), onOpenChat: openChat,
    setTimer: callback => { timers.set(++id, callback); return id; }, clearTimer: key => timers.delete(key) });
  t.after(() => controller.dispose());
  return { controller, calls, states, timers, setModel: value => { current = value; }, latest: () => states.at(-1) };
}

test('the API only reads its fixed same-origin routes and applies the mutation transport protections', async () => {
  const calls = [], controller = new AbortController();
  const api = createApi({ origin: ORIGIN, fetchImpl: async (url, options) => { calls.push({ url, ...options }); return { ok: true, json: async () => url.endsWith('/model') ? model() : {} }; } });
  await api.get('status', controller.signal); await api.get('model', controller.signal);
  for (const action of ['load', 'unload', 'connect']) await api.post(action, undefined, controller.signal);
  for (const seconds of [300, 600, 1800]) await api.post('idle', seconds, controller.signal);
  for (const tokens of catalogChoices) await api.post('context', tokens, controller.signal);
  for (const call of calls) {
    assert.equal(new URL(call.url).origin, ORIGIN);
    assert.ok(new URL(call.url).pathname.startsWith('/__strata_extension/api/'));
    assert.equal(call.mode, 'same-origin'); assert.equal(call.redirect, 'error'); assert.equal(call.cache, 'no-store');
    assert.deepEqual(call.headers, { 'content-type': 'application/json', 'x-strata-desktop': 'local-model' });
    assert.ok(call.signal instanceof AbortSignal);
    if (call.method === 'POST') {
      const action = new URL(call.url).pathname.split('/').at(-1);
      assert.deepEqual(Object.keys(JSON.parse(call.body)), action === 'context' ? ['tokens','expectedModel'] : action === 'idle' ? ['seconds','expectedModel'] : ['expectedModel'],
        'Only The Announced Physical Model, With No Effort, Fast Or Permission Overrides');
    }
  }
  assert.equal(calls[0].method, 'GET'); assert.equal(calls[0].body, undefined);
  assert.deepEqual(JSON.parse(calls[2].body), {expectedModel:'physical-model-id'}); assert.equal(JSON.parse(calls[5].body).seconds,300); assert.equal(JSON.parse(calls[8].body).tokens,8192);
});

test('context POST admission follows the latest API catalog with no client-defined tiers', async () => {
  let choices = [8192, 131072], calls = 0;
  const api = createApi({ origin: ORIGIN, fetchImpl: async () => { calls++; return { ok: true, json: async () => ({available:true,model:{id:'physical-model-id'},context:{choices}}) }; } });
  assert.throws(() => api.post('context', 131072), /Catalog/); assert.equal(calls, 0);
  await api.get('model'); await api.post('context', 8192);
  choices = [32768, 131072]; await api.get('model');
  const before = calls; assert.throws(() => api.post('context', 8192), /Catalog/); assert.equal(calls, before);
  await api.post('context', 32768);
});

test('mutations use the most recently announced physical model and stop when it is unavailable', async () => {
  let current = model(); const posts = [];
  const api = createApi({ origin: ORIGIN, fetchImpl: async (_url, options) => {
    if (options.method === 'POST') posts.push(JSON.parse(options.body));
    return {ok:true,json:async()=>current};
  } });
  assert.throws(() => api.post('load'), /Identity/);
  await api.get('model'); await api.post('load');
  current = {...model(),model:{...model().model,id:'another-physical-id'}};
  await api.get('model'); await api.post('unload');
  current.available = false; await api.get('model');
  assert.throws(() => api.post('idle', 600), /Identity/);
  assert.deepEqual(posts,[{expectedModel:'physical-model-id'},{expectedModel:'another-physical-id'}]);
});

test('an owner error acknowledges only its matching request and never retries a mutation', async t => {
  let data = model(), posts = 0;
  const f = panelFixture(t, {api:{
    async get(action) {return action === 'status' ? status : structuredClone(data);},
    async post() {posts++;return {requestId:'example-command'};},
  }});
  f.controller.setVisible(true); await tick();
  await f.controller.perform('context',65536); await tick();
  data.request = {id:'older-command',state:'error'}; await f.controller.refresh();
  assert.ok(f.latest().requested);
  data.request.id = 'example-command'; data.error = 'The Owner Rejected This Request.';
  await f.controller.refresh();
  assert.equal(f.latest().requested,null); assert.equal(f.latest().error,data.error);
  await f.controller.refresh(); assert.equal(posts,1);
});

test('a changed physical model cannot confirm an earlier model request or strand its controls', async t => {
  const f = panelFixture(t); f.controller.setVisible(true); await tick();
  await f.controller.perform('context',65536); await tick();
  const data = model(); data.model.id = 'another-physical-id'; data.context.selected = data.context.effective = 65536;
  f.setModel(data); await f.controller.refresh();
  assert.equal(f.latest().requested,null); assert.match(f.latest().error,/Model Changed/);
  assert.equal(f.latest().view.canControl,true);
  assert.equal(f.calls.filter(c=>c.method==='POST').length,1);
});

test('invalid routes, origins and choices fail before fetch', () => {
  let calls = 0;
  const api = createApi({ origin: ORIGIN, fetchImpl: () => { calls++; } });
  for (const route of ['https://foreign.example', '../config', 'model?secret=x', '__proto__']) assert.throws(() => api.get(route));
  for (const [action, value] of [['idle', 60], ['idle', '600'], ['context', 0], ['context', '131072'], ['context', -1], ['load', {}], ['connect', 'model'], ['https://foreign.example', undefined]]) assert.throws(() => api.post(action, value));
  for (const origin of ['null', 'file:///app', ORIGIN + '/path', ORIGIN + '?key=x']) assert.throws(() => createApi({ origin }));
  assert.equal(calls, 0);
  assert.deepEqual(actionBody('idle', 1800), { seconds: 1800 });
});

test('API failures expose text only and caller cancellation reaches fetch', async () => {
  const api = createApi({ origin: ORIGIN, fetchImpl: async () => ({ ok: false, json: async () => ({ error: '<img src=x onerror=alert(1)>' }) }) });
  await assert.rejects(api.get('model'), /<img src=x/);
  const outer = new AbortController(); let signal;
  const pending = createApi({ origin: ORIGIN, fetchImpl: (_url, options) => new Promise((_resolve, reject) => {
    signal = options.signal; signal.addEventListener('abort', () => reject(signal.reason), { once: true });
  }) }).get('model', outer.signal);
  outer.abort(new Error('Closed'));
  await assert.rejects(pending, /Closed/); assert.equal(signal.aborted, true);
});

test('saved 128K and 30 Minutes come from the DTO without an optimistic 256K default', () => {
  const view = modelView({ model: model(), status });
  assert.equal(view.selected, 131072); assert.equal(view.effective, 131072);
  assert.equal(view.idleSelected, 1800); assert.equal(view.idleEffective, 1800);
  assert.equal(contextLabel(view.effective), '128K'); assert.equal(idleLabel(view.idleEffective), '30 Minutes');
  assert.equal(view.can.load, true); assert.equal(view.can.unload, false);
});

test('unknown or malformed capacity, idle and model metadata remain unconfirmed', () => {
  for (const invalid of [null, undefined, '131072', true, 0, -1, 131072.5, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
    const data = model(); data.context.selected = data.context.effective = invalid;
    const view = modelView({ model: data, status });
    assert.equal(view.selected, null); assert.equal(view.effective, null); assert.equal(view.canControl, false);
    assert.equal(contextLabel(invalid), 'Unconfirmed');
  }
  for (const invalid of [null, undefined, '1800', false, 60]) {
    const data = model(); data.autoUnload.selected = data.autoUnload.effective = invalid;
    const view = modelView({ model: data, status });
    assert.equal(view.idleSelected, null); assert.equal(view.idleEffective, null); assert.equal(idleLabel(invalid), 'Unconfirmed');
  }
  const data = model(); data.model.loaded = null;
  assert.equal(modelView({ model: data, status }).canControl, false);
});

test('pending context and idle never masquerade as effective and disable actions', () => {
  const data = model(); data.context.selected = 524288; data.autoUnload.selected = 600;
  const view = modelView({ model: data, status });
  assert.equal(view.effective, 131072); assert.equal(view.idleEffective, 1800);
  assert.equal(view.contextPending, true); assert.equal(view.idlePending, true);
  assert.ok(Object.values(view.can).every(value => !value));
  assert.equal(contextLabel(view.selected), '512K'); assert.equal(contextLabel(view.effective), '128K');
});

test('catalog options are generic; missing choices never fabricate a default list or tested capacity', () => {
  const data = model();
  assert.deepEqual(modelView({ model: data, status }).choices, catalogChoices);
  data.context.choices = [8192, '65536', null, 8192, 0, 32768];
  assert.deepEqual(modelView({ model: data, status }).choices, [8192, 32768]);
  delete data.context.choices;
  const view = modelView({ model: data, status });
  assert.deepEqual(view.choices, []); assert.equal(view.can.context, false); assert.equal(view.effective, 131072);
  assert.equal(contextLabel(200000), '200,000 Tokens', 'Do Not Invent A Rounded K Label');
});

test('incompatible, disabled and offline states suppress stale effective settings', () => {
  for (const bridge of [null, { ...status, enabled: false }, { ...status, compatible: false }, { ...status, bridgeApi: 2 }, { ...status, extension: { id: 'other' } }]) {
    assert.equal(compatible(bridge), false);
    const view = modelView({ model: model(), status: bridge });
    assert.equal(view.effective, null); assert.equal(view.idleEffective, null); assert.equal(view.canControl, false);
  }
  const view = modelView({ model: { ...model(), available: false }, status });
  assert.equal(view.effective, null); assert.equal(view.loaded, null); assert.equal(view.canControl, false);
});

test('busy, transitioning and unknown request counts fail closed for controls', () => {
  for (const patch of [{ phase: 'draining' }, { phase: 'loading' }, { phase: 'unknown' }, { activeRequests: 1 }, { inFlight: 1 }, { activeRequests: null }, { inFlight: null }, { inFlight: '0' }]) {
    const data = model(); Object.assign(data.context, patch);
    assert.equal(modelView({ model: data, status }).canControl, false);
  }
  const data = model(); data.model.loading = true;
  assert.equal(modelView({ model: data, status }).canControl, false);
  assert.equal(modelView({ model: model(), status, action: 'connect' }).canControl, false);
});

test('nothing polls while closed; visible polling is GET-only and closing clears the timer', async t => {
  const f = panelFixture(t); await tick(); assert.equal(f.calls.length, 0);
  f.controller.setVisible(true); await tick();
  assert.deepEqual(f.calls.map(call => call.action), ['status', 'model']); assert.equal(f.timers.size, 1);
  const poll = [...f.timers.values()][0]; f.timers.clear(); poll(); await tick();
  assert.equal(f.calls.length, 4); assert.ok(f.calls.every(call => call.method === 'GET'));
  f.controller.setVisible(false); assert.equal(f.timers.size, 0);
  await f.controller.refresh(); assert.equal(f.calls.length, 4);
});

test('a late read is aborted and ignored when the panel closes', async t => {
  const requests = [];
  const f = panelFixture(t, { api: { get: (action, signal) => new Promise(resolve => requests.push({ action, signal, resolve })) } });
  f.controller.setVisible(true); f.controller.setVisible(false);
  assert.ok(requests.every(request => request.signal.aborted));
  for (const request of requests) request.resolve(request.action === 'status' ? status : model());
  await tick(); assert.equal(f.latest().model, null); assert.equal(f.timers.size, 0);
});

test('a failed poll cancels its other read rather than leaving a hidden request running', async t => {
  let modelSignal;
  const f = panelFixture(t, { api: { get(action, signal) {
    if (action === 'status') return Promise.reject(new Error('Bridge Unavailable.'));
    modelSignal = signal; return new Promise(() => {});
  } } });
  f.controller.setVisible(true); await tick();
  assert.equal(modelSignal.aborted, true); assert.equal(f.latest().view.canControl, false);
});

test('a recovered read clears an old connection error without sending a mutation', async t => {
  let offline = true, posts = 0;
  const f = panelFixture(t, { api: {
    async get(action) { if (offline) throw new Error('Connection Unavailable.'); return action === 'status' ? status : model(); },
    async post() { posts++; },
  } });
  f.controller.setVisible(true); await tick(); assert.equal(f.latest().view.canControl, false);
  assert.equal(f.latest().error, 'Connection Unavailable.'); offline = false;
  await f.controller.refresh(); assert.equal(f.latest().error, ''); assert.equal(f.latest().view.canControl, true); assert.equal(posts, 0);
});

test('a context mutation remains pending until the selected AND effective values are confirmed', async t => {
  const f = panelFixture(t); f.controller.setVisible(true); await tick();
  assert.equal(await f.controller.perform('context', 65536), true); await tick();
  assert.equal(f.latest().view.selected, 131072); assert.equal(f.latest().view.effective, 131072);
  assert.equal(f.latest().view.canControl, false); assert.equal(f.latest().requested.value, 65536);
  assert.equal(await f.controller.perform('load'), false);
  const data = model(); data.context.selected = 65536; f.setModel(data);
  await f.controller.refresh(); assert.equal(f.latest().requested.value, 65536);
  data.context.effective = 65536; f.setModel(data); await f.controller.refresh();
  assert.equal(f.latest().requested, null); assert.equal(f.latest().view.effective, 65536);
  assert.equal(f.calls.filter(call => call.method === 'POST').length, 1);
});

test('idle requests preserve the effective value until confirmation and no selected choice is posted automatically', async t => {
  const f = panelFixture(t); f.controller.setVisible(true); await tick();
  assert.equal(await f.controller.perform('idle', 1800), false);
  assert.equal(await f.controller.perform('idle', 600), true); await tick();
  assert.equal(f.latest().view.idleEffective, 1800);
  const data = model(); data.autoUnload.selected = data.autoUnload.effective = 600; f.setModel(data);
  await f.controller.refresh(); assert.equal(f.latest().requested, null);
  assert.deepEqual(f.calls.filter(call => call.method === 'POST').map(({ action, value }) => ({ action, value })), [{ action: 'idle', value: 600 }]);
});

test('Use In Chat passes the verified physical selection to the official adapter after connecting', async t => {
  const opened = [];
  const f = panelFixture(t, { openChat: value => { opened.push(value); } });
  f.controller.setVisible(true); await tick();
  assert.equal(await f.controller.perform('connect'), true); assert.deepEqual(opened, [connectedSelection(connection())]);
  assert.deepEqual(f.calls.filter(call => call.method === 'POST').map(call => call.action), ['connect']);
});

test('a failed connect never navigates and a preload waits for observed loaded state', async t => {
  let opened = 0, fail = true;
  const f = panelFixture(t, { openChat: () => { opened++; }, api: {
    async get(action) { return action === 'status' ? status : model(); },
    async post() { if (fail) throw new Error('Connection Metadata Was Not Confirmed.'); return { status: 'loaded' }; },
  } });
  f.controller.setVisible(true); await tick();
  assert.equal(await f.controller.perform('connect'), false); assert.equal(opened, 0);
  assert.match(f.latest().error, /Not Confirmed/);
  fail = false; assert.equal(await f.controller.perform('load'), true); await tick();
  assert.equal(f.latest().view.loaded, false); assert.equal(f.latest().view.canControl, false);
  assert.equal(f.latest().requested.action, 'load');
});

test('closing cancels an in-flight mutation and never navigates or retries its late result', async t => {
  let resolvePost, signal, posts = 0, opened = 0;
  const f = panelFixture(t, { openChat: () => { opened++; }, api: {
    async get(action) { return action === 'status' ? status : model(); },
    post(_action, _value, incoming) { posts++; signal = incoming; return new Promise(resolve => { resolvePost = resolve; }); },
  } });
  f.controller.setVisible(true); await tick(); const pending = f.controller.perform('connect');
  f.controller.setVisible(false); assert.equal(signal.aborted, true); resolvePost({}); await pending;
  assert.equal(opened, 0); assert.equal(posts, 1); assert.equal(f.timers.size, 0);
});

test('rendering selects only API values and writes even HTML-shaped errors as literal text', () => {
  const elements = new Map();
  const element = () => ({ textContent: '', dataset: {}, disabled: false, hidden: false, attributes: {},
    replaceChildren(...children) { this.children = children; },
    setAttribute(name, value) { this.attributes[name] = value; }, set innerHTML(_value) { throw new Error('HTML Is Forbidden'); } });
  const idles = [300, 600, 1800].map(value => ({ ...element(), dataset: { idle: String(value) } }));
  const doc = { getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
    createElement: () => element(), querySelectorAll() { return idles; } };
  const state = { model: model(), status, error: '<svg onload=alert(1)>', notice: '', action: null };
  renderPanel(doc, { ...state, view: modelView(state) });
  assert.equal(elements.get('error').textContent, state.error);
  const contextSelect = elements.get('context-choice');
  assert.deepEqual(contextSelect.children.map(option => option.value), catalogChoices.map(String));
  assert.deepEqual(contextSelect.children.map(option => option.textContent), catalogChoices.map(contextLabel));
  assert.equal(contextSelect.value, '131072');
  assert.deepEqual(idles.map(button => button.attributes['aria-pressed']), ['false', 'false', 'true']);
  assert.equal(elements.get('context-effective').textContent, '128K');
  assert.equal(elements.get('unload').disabled, true); assert.equal(elements.get('load').disabled, false);
});

function dialogFixture(t) {
  class Target extends EventTarget {}
  const doc = new Target(), win = new Target(), dialog = new Target(), frame = new Target(), messages = [];
  let focused = 0, shown = 0;
  const trigger = { isConnected: true, attributes: {}, setAttribute(key, value) { this.attributes[key] = value; }, focus() { focused++; } };
  Object.assign(doc, { visibilityState: 'visible', activeElement: { shadowRoot: { activeElement: trigger } } });
  Object.assign(win, { document: doc, location: { origin: ORIGIN, hash: '#/models' } });
  Object.assign(dialog, { open: false, showModal() { this.open = true; shown++; }, close() { this.open = false; this.dispatchEvent(new Event('close')); } });
  Object.assign(frame, { getAttribute() { return this.src; }, contentWindow: { postMessage(message, target) { messages.push({ ...message, target }); } } });
  const selected = [];
  const controller = createDialogController({ win, dialog, frame, trigger, theme: () => 'dark',
    onUseInChat: value => { selected.push(value); win.location.hash = '#/chat'; } });
  t.after(() => controller.dispose());
  return { controller, win, doc, dialog, frame, trigger, messages, selected, focused: () => focused, shown: () => shown };
}

test('dialog single-open, native Escape cancellation and shadow focus return work without touching the app', t => {
  const f = dialogFixture(t); f.controller.open(); f.controller.open();
  assert.equal(f.shown(), 1); assert.equal(f.frame.src, '/__strata_extension/ui/index.html');
  assert.equal(f.trigger.attributes['aria-expanded'], 'true');
  const cancel = new Event('cancel', { cancelable: true }); f.dialog.dispatchEvent(cancel);
  assert.equal(cancel.defaultPrevented, true); assert.equal(f.dialog.open, false); assert.equal(f.focused(), 1);
  assert.equal(f.messages.at(-1).visible, false); assert.equal(f.win.location.hash, '#/models');
});

test('only same-origin messages from the exact iframe can close or navigate', t => {
  const f = dialogFixture(t); f.controller.open();
  for (const data of [
    { origin: 'https://foreign.example', source: f.frame.contentWindow, data: { type: 'strata-extension:open-chat' } },
    { origin: ORIGIN, source: {}, data: { type: 'strata-extension:open-chat' } },
    { origin: ORIGIN, source: f.frame.contentWindow, data: { type: 'strata-extension:delete' } },
  ]) {
    assert.equal(frameMessage(data, f.frame, ORIGIN), null);
    const event = new Event('message'); Object.assign(event, data); f.win.dispatchEvent(event);
    assert.equal(f.dialog.open, true); assert.equal(f.win.location.hash, '#/models');
  }
  const invalid = new Event('message'); Object.assign(invalid, { origin: ORIGIN, source: f.frame.contentWindow, data: { type: 'strata-extension:open-chat' } });
  f.win.dispatchEvent(invalid); assert.equal(f.dialog.open, true); assert.equal(f.selected.length, 0);
  assert.equal(f.messages.at(-1).type, 'strata-extension:selection-error');
  const event = new Event('message'); Object.assign(event, { origin: ORIGIN, source: f.frame.contentWindow, data: { type: 'strata-extension:open-chat', connection: connection() } });
  f.win.dispatchEvent(event); assert.equal(f.dialog.open, false); assert.equal(f.win.location.hash, '#/chat');
  assert.deepEqual(f.selected, [connectedSelection(connection())]);
  assert.equal(frameMessage({ origin: ORIGIN, source: null, data: { type: 'strata-extension:open-chat' } }, { contentWindow: null }, ORIGIN), null);
});

test('the actual frame boot waits for verified visibility; iframe Escape sends only a same-origin close message', async t => {
  class Element extends EventTarget {
    constructor(dataset = {}) { super(); this.dataset = dataset; this.attributes = {}; }
    setAttribute(name, value) { this.attributes[name] = value; }
    replaceChildren(...children) { this.children = children; }
    set innerHTML(_value) { throw new Error('No HTML Injection'); }
  }
  const html = await readFile(new URL('./ui/index.html', import.meta.url), 'utf8');
  const elements = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(match => [match[1], new Element()]));
  const idles = [300, 600, 1800].map(value => new Element({ idle: String(value) }));
  const doc = new EventTarget(), win = new EventTarget(), sent = [], calls = [];
  const parent = { postMessage(data, target) { sent.push({ data, target }); } };
  Object.assign(doc, { visibilityState: 'visible', documentElement: { dataset: {} },
    getElementById: id => { assert.ok(elements.has(id), `Real HTML Is Missing ${id}`); return elements.get(id); },
    createElement: () => new Element(), querySelectorAll: () => idles,
  });
  Object.assign(win, { parent, document: doc, location: { origin: ORIGIN }, async fetch(url, options) {
    calls.push({ url, ...options }); return { ok: true, json: async () => url.endsWith('/status') ? status : url.endsWith('/connect') ? connection() : model() };
  } });
  const dispose = startPanel(win); t.after(dispose);
  await tick(); assert.equal(calls.length, 0);
  const message = (origin, source, visible) => {
    const event = new Event('message'); Object.assign(event, { origin, source, data: { type: 'strata-extension:visibility', visible, theme: 'light' } }); win.dispatchEvent(event);
  };
  message('https://foreign.example', parent, true); message(ORIGIN, {}, true); await tick(); assert.equal(calls.length, 0);
  message(ORIGIN, parent, true); await tick(); assert.equal(calls.length, 2);
  assert.equal(elements.get('context-effective').textContent, '128K'); assert.equal(doc.documentElement.dataset.theme, 'light');
  elements.get('connect').dispatchEvent(new Event('click')); await tick();
  assert.deepEqual(sent.at(-1), { data: { type: 'strata-extension:open-chat', connection: connectedSelection(connection()) }, target: ORIGIN });
  const incoming = (source, message) => { const event = new Event('message'); Object.assign(event, { origin: ORIGIN, source,
    data: { type: 'strata-extension:selection-error', message } }); win.dispatchEvent(event); };
  incoming({}, 'Untrusted'); assert.notEqual(elements.get('error').textContent, 'Untrusted');
  incoming(parent, '<b>Picker Unavailable</b>'); assert.equal(elements.get('error').textContent, '<b>Picker Unavailable</b>');
  const escape = new Event('keydown', { cancelable: true }); Object.assign(escape, { key: 'Escape' }); doc.dispatchEvent(escape);
  assert.equal(escape.defaultPrevented, true); assert.deepEqual(sent.at(-1), { data: { type: 'strata-extension:close' }, target: ORIGIN });
  const beforeClose = calls.length;
  message(ORIGIN, parent, false); elements.get('refresh').dispatchEvent(new Event('click')); await tick();
  assert.equal(calls.length, beforeClose); assert.equal(calls.filter(call => call.method === 'POST').length, 1);
  assert.deepEqual(JSON.parse(calls.find(call => call.method === 'POST').body), {expectedModel:'physical-model-id'});
});

test('parent window visibility stops iframe polling and theme stays within the extension', t => {
  const f = dialogFixture(t); f.controller.open();
  f.doc.visibilityState = 'hidden'; f.doc.dispatchEvent(new Event('visibilitychange'));
  assert.equal(f.messages.at(-1).visible, false); assert.equal(f.messages.at(-1).target, ORIGIN);
  f.doc.visibilityState = 'visible'; f.doc.dispatchEvent(new Event('visibilitychange'));
  assert.equal(f.messages.at(-1).visible, true); assert.equal(f.messages.at(-1).theme, 'dark');
  const win = { document: { documentElement: { dataset: { theme: 'light' } } }, getComputedStyle: () => ({ colorScheme: 'dark' }), matchMedia: () => ({ matches: true }) };
  assert.equal(detectTheme(win), 'light'); delete win.document.documentElement.dataset.theme;
  assert.equal(detectTheme(win), 'dark'); win.getComputedStyle = () => ({ colorScheme: 'normal' }); win.matchMedia = () => ({ matches: false });
  assert.equal(detectTheme(win), 'light');
});

test('repeated injection mounts one isolated custom element', () => {
  let mounted = null, added = 0;
  const win = { customElements: { get: () => true }, HTMLDialogElement: function () {}, document: {
    querySelector: () => mounted, createElement: tag => ({ tag }), body: { append(value) { mounted = value; added++; } },
  } };
  assert.equal(mountExtension(win).tag, 'strata-local-models-extension'); assert.equal(mountExtension(win), mounted); assert.equal(added, 1);
});

test('the unbundled frame ships only local stylesheet and module references, with no selected defaults', async () => {
  const html = await readFile(new URL('./ui/index.html', import.meta.url), 'utf8');
  assert.match(html, /type="module" src="\.\/panel.mjs"/); assert.match(html, /href="\.\/panel.css"/);
  assert.doesNotMatch(html, /https?:\/\/|aria-pressed="true"|<script[^>]+src="\/\//);
  assert.doesNotMatch(html, /data-context|64K|128K|256K|Everyday|Extended|Long Context/);
  assert.doesNotMatch(html, /<select[^>]*(?:effort|reasoning)|reasoning_effort|permissions|Fast Mode/i);
  assert.match(html, /<select id="context-choice"/); assert.match(html, /Conversation Context/);
  for (const label of ['Never', '5 Minutes', '10 Minutes', '30 Minutes', 'Current Model', 'API And Owner State']) assert.ok(html.includes(label));
});

test('connect identity validation rejects incomplete, stale and malformed targets before any picker handoff', async t => {
  for (const patch of [{ connectionName: '' }, { label: null }, { contextMax: '131072' },
    { selection: { connectionId: 'carrier-connection', modelId: 'alias-id' } }]) {
    let handoffs = 0;
    const f = panelFixture(t, { openChat: () => { handoffs++; }, api: {
      async get(action) { return action === 'status' ? status : model(); },
      async post() { return { ...connection(), ...patch }; },
    } });
    f.controller.setVisible(true); await tick();
    assert.equal(await f.controller.perform('connect'), false); assert.equal(handoffs, 0); assert.ok(f.latest().error);
  }
  for (const invalid of [null, {}, { ...connection(), connectionName: '\nLocal Carrier' }, { ...connection(), label: ' Qwen' }])
    assert.throws(() => connectedSelection(invalid));
  assert.deepEqual(connectedSelection({ ...connection(), untrusted: '<img>', effort: 'max', files: '/secret' }), connectedSelection(connection()));
});

test('an explicit Chat compatibility failure is actionable and a later explicit action can recover', async t => {
  let fail = true, attempts = 0;
  const f = panelFixture(t, { openChat: async () => { attempts++; if (fail) throw new Error('Give The Connections Unique Names.'); } });
  f.controller.setVisible(true); await tick();
  assert.equal(await f.controller.perform('connect'), false); await tick();
  assert.equal(f.latest().error, 'Give The Connections Unique Names.');
  fail = false; assert.equal(await f.controller.perform('connect'), true);
  assert.equal(attempts, 2); assert.equal(f.calls.filter(call => call.method === 'POST').length, 2);
});

// A small deterministic light/shadow DOM fixture. It has no network or browser runtime.
// Selectors follow the observed official markup; shadow trees are not searched from the document.
function domFixture() {
  const observers = new Set(), doc = new EventTarget(), win = new EventTarget();
  function notify(target, type, attributeName) {
    for (const observer of observers) {
      let inside = false;
      for (let node = target; node; node = node.parentElement) if (node === observer.target) { inside = true; break; }
      if (!inside || (type === 'attributes' && (!observer.options.attributes ||
          (observer.options.attributeFilter && !observer.options.attributeFilter.includes(attributeName)))) ||
          (type === 'childList' && !observer.options.childList)) continue;
      if (!observer.queued) {
        observer.queued = true;
        queueMicrotask(() => { observer.queued = false; if (observers.has(observer)) observer.callback(); });
      }
    }
  }
  const simple = (node, selector) => {
    const tag = selector.match(/^[\w-]+/);
    if (tag && node.tagName !== tag[0].toLowerCase()) return false;
    for (const [, name] of selector.matchAll(/\.([\w-]+)/g)) if (!node.className.split(/\s+/).includes(name)) return false;
    for (const [, name] of selector.matchAll(/#([\w-]+)/g)) if (node.id !== name) return false;
    for (const [, key, , value] of selector.matchAll(/\[([\w-]+)(?:=(["']?)([^\]"']+)\2)?\]/g))
      if (value === undefined ? !node.hasAttribute(key) : node.getAttribute(key) !== value) return false;
    return true;
  };
  function matches(node, selector) {
    return selector.split(',').some(part => {
      const segments = part.trim().split(/\s+/);
      if (!simple(node, segments.pop())) return false;
      let ancestor = node.parentElement;
      while (segments.length) {
        const segment = segments.pop(); while (ancestor && !simple(ancestor, segment)) ancestor = ancestor.parentElement;
        if (!ancestor) return false; ancestor = ancestor.parentElement;
      }
      return true;
    });
  }
  class Element extends EventTarget {
    constructor(tag = 'div') {
      super(); this.tagName = tag; this.children = []; this.attributes = new Map(); this.parentElement = null; this.literal = ''; this.clicks = 0;
      this.dataset = new Proxy({}, { set: (values, key, value) => { values[key] = value; this.setAttribute(`data-${key.replace(/[A-Z]/g, c => `-${c.toLowerCase()}`)}`, value); return true; } });
    }
    get isConnected() { return this === doc.documentElement || !!(this.parentElement?.isConnected || this.host?.isConnected); }
    get textContent() { return this.literal + this.children.map(child => child.textContent).join(''); }
    set textContent(value) { this.literal = String(value); for (const child of [...this.children]) child.remove(); notify(this, 'childList'); }
    set innerHTML(_value) { throw new Error('Untrusted HTML Is Forbidden'); }
    get className() { return this.getAttribute('class') ?? ''; }
    set className(value) { this.setAttribute('class', value); }
    get id() { return this.getAttribute('id') ?? ''; }
    set id(value) { this.setAttribute('id', value); }
    get hidden() { return this.hasAttribute('hidden'); }
    set hidden(value) { if (value) this.setAttribute('hidden', ''); else this.removeAttribute('hidden'); }
    get disabled() { return this.hasAttribute('disabled'); }
    set disabled(value) { if (value) this.setAttribute('disabled', ''); else this.removeAttribute('disabled'); }
    get nextSibling() { return this.parentElement?.children[this.parentElement.children.indexOf(this) + 1] ?? null; }
    setAttribute(key, value) { const next = String(value); if (this.attributes.get(key) !== next) { this.attributes.set(key, next); notify(this, 'attributes', key); } }
    getAttribute(key) { return this.attributes.get(key) ?? null; }
    hasAttribute(key) { return this.attributes.has(key); }
    removeAttribute(key) { if (this.attributes.delete(key)) notify(this, 'attributes', key); }
    append(...children) { for (const child of children) this.insertBefore(child, null); }
    insertBefore(child, before) {
      if (child.parentElement) child.remove();
      const index = before ? this.children.indexOf(before) : this.children.length;
      if (index < 0) throw new Error('Detached Insertion Anchor');
      this.literal = ''; this.children.splice(index, 0, child); child.parentElement = this;
      notify(this, 'childList'); if (child.isConnected) child.connectedCallback?.(); return child;
    }
    remove() {
      const parent = this.parentElement; if (!parent) return;
      parent.children.splice(parent.children.indexOf(this), 1); this.parentElement = null; this.disconnectedCallback?.(); notify(parent, 'childList');
    }
    matches(selector) { return matches(this, selector); }
    closest(selector) { for (let node = this; node; node = node.parentElement) if (node.matches(selector)) return node; return null; }
    querySelectorAll(selector) { return this.children.flatMap(child => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null; }
    getClientRects() { return this.isConnected && !this.closest('[hidden], [aria-hidden="true"], [inert]') ? [{}] : []; }
    attachShadow() { this.shadowRoot = new Element('shadow-root'); this.shadowRoot.host = this; return this.shadowRoot; }
    focus() { doc.activeElement = this; }
    click() { if (!this.disabled) { this.clicks++; this.dispatchEvent(new Event('click')); } }
    showModal() { this.open = true; }
    close() { this.open = false; this.dispatchEvent(new Event('close')); }
  }
  const registry = new Map();
  Object.assign(doc, { documentElement: new Element('html'), visibilityState: 'visible',
    createElement(tag) {
      const Type = registry.get(tag), node = Type ? new Type() : new Element(tag); node.tagName = tag;
      if (tag === 'iframe') node.contentWindow = { postMessage() {} }; return node;
    },
    querySelectorAll: selector => doc.documentElement.querySelectorAll(selector),
    querySelector: selector => doc.documentElement.querySelector(selector),
  });
  doc.body = doc.createElement('body'); doc.documentElement.append(doc.body);
  Object.assign(win, { document: doc, location: { origin: ORIGIN, hash: '#/models' }, HTMLElement: Element, HTMLDialogElement: Element,
    customElements: { get: tag => registry.get(tag), define: (tag, Type) => registry.set(tag, Type) },
    getComputedStyle: node => ({ display: node.hidden ? 'none' : 'block', visibility: 'visible', colorScheme: 'dark' }),
    matchMedia: () => Object.assign(new EventTarget(), { matches: true }),
    MutationObserver: class {
      constructor(callback) { this.callback = callback; }
      observe(target, options) { this.target = target; this.options = options; observers.add(this); }
      disconnect() { observers.delete(this); }
    },
    fetch() { throw new Error('Unstubbed Network Is Forbidden'); },
  });
  const element = (tag, attrs = {}, content) => { const node = doc.createElement(tag); for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value); if (content) node.textContent = content; return node; };
  return { win, doc, element, observers };
}

function legacyRows(f, menu, { name = 'Local Carrier', id = 'physical-model-id', label = 'Shared Qwen' } = {}) {
  const group = f.element('div'), heading = f.element('p', { class: 'model-picker-heading' }, name);
  const option = f.element('button', { class: 'menu__item model-picker-option model-picker-api', role: 'menuitemradio' });
  const span = f.element('span'); span.append(f.element('strong', {}, label), f.element('small', {}, id));
  option.append(span); group.append(heading, option); menu.append(group); return option;
}

function chatFixture({ native = false, duplicate = false, wrongProof = false, noChoice = false, delayedMenu = false } = {}) {
  const f = domFixture(), { doc, element } = f;
  const trigger = element('button', { class: 'chat-model-badge chat-model-picker', 'aria-expanded': 'false', title: 'Other Connection · other-model' });
  trigger.append(element('span', {}, 'Other Model')); doc.body.append(trigger);
  let menu, selected = 0;
  function populate() {
    menu = element('div', { class: 'menu', role: 'menu' }); doc.body.append(menu);
    if (noChoice) return;
    const add = () => native ? (() => { const button = element('button', { class: 'menu__item', role: 'menuitem' }, 'Shared Qwen · Local Carrier'); menu.append(button); return button; })() : legacyRows(f, menu);
    const option = add(); if (duplicate) add();
    option.addEventListener('click', () => {
      selected++; menu.remove(); trigger.setAttribute('aria-expanded', 'false');
      // Native's API choice switches to LegacyChatModelPicker; both expose this exact proof.
      const current = native ? element('button', { class: 'chat-model-picker', 'aria-expanded': 'false' }) : trigger;
      current.setAttribute('title', `Local Carrier · ${wrongProof ? 'alias-id' : 'physical-model-id'}`);
      current.textContent = ''; current.append(element('span', {}, 'Shared Qwen'));
      if (native) { trigger.remove(); doc.body.append(current); }
    });
  }
  trigger.addEventListener('click', () => {
    if (trigger.getAttribute('aria-expanded') === 'true') { trigger.setAttribute('aria-expanded', 'false'); menu?.remove(); }
    else { trigger.setAttribute('aria-expanded', 'true'); if (!delayedMenu) populate(); }
  });
  return { ...f, trigger, populate, selected: () => selected, menu: () => menu };
}

test('legacy picker requires the exact connection, label and physical ID, and rejects duplicate groups', () => {
  const f = domFixture(), menu = f.element('div', { role: 'menu' }); f.doc.body.append(menu);
  legacyRows(f, menu, { name: 'Other Connection' }); legacyRows(f, menu, { name: 'Another Connection', id: 'alias-id' });
  assert.equal(pickerOption(menu, connection()), null);
  const target = legacyRows(f, menu); assert.equal(pickerOption(menu, connection()), target);
  legacyRows(f, menu, { id: 'different-id' }); assert.throws(() => pickerOption(menu, connection()), /Unique Names/);
  assert.equal(target.clicks, 0);
});

test('native and legacy explicit selection click one picker and one exact row, then confirm the physical ID', async () => {
  for (const native of [false, true]) {
    const f = chatFixture({ native });
    const result = await selectInOfficialChat(f.win, connection());
    assert.equal(result.selection.modelId, 'physical-model-id'); assert.equal(f.win.location.hash, '#/chat');
    assert.equal(f.trigger.clicks, 1); assert.equal(f.selected(), 1); assert.equal(f.observers.size, 0);
    assert.equal(f.doc.querySelector('.chat-model-picker').getAttribute('title'), 'Local Carrier · physical-model-id');
  }
});

test('duplicate native labels fail closed without selecting a row; a fresh explicit action recovers', async () => {
  const f = chatFixture({ native: true, duplicate: true });
  await assert.rejects(selectInOfficialChat(f.win, connection()), /More Than One Chat Model/);
  assert.equal(f.selected(), 0); assert.equal(f.trigger.clicks, 2, 'Open Once And Close Our Menu Once');
  assert.equal(f.observers.size, 0);
  const fresh = chatFixture({ native: true }); await selectInOfficialChat(fresh.win, connection()); assert.equal(fresh.selected(), 1);
});

test('selecting an already confirmed physical target makes no clicks and does not change settings', async () => {
  const f = chatFixture(); f.trigger.setAttribute('title', 'Local Carrier · physical-model-id');
  f.trigger.querySelector('span').textContent = 'Shared Qwen';
  await selectInOfficialChat(f.win, connection()); assert.equal(f.trigger.clicks, 0); assert.equal(f.selected(), 0);
});

test('a row label alone is not success; wrong physical proof times out with no repeated selection', async () => {
  const f = chatFixture({ native: true, wrongProof: true });
  await assert.rejects(selectInOfficialChat(f.win, connection(), { timeout: 20 }), /Could Not Be Confirmed/);
  assert.equal(f.trigger.clicks, 1); assert.equal(f.selected(), 1); assert.equal(f.observers.size, 0);
});

test('menu DOM may mount later; a bounded observer waits without repeated clicks', async () => {
  const f = chatFixture({ delayedMenu: true });
  const pending = selectInOfficialChat(f.win, connection()); await tick();
  assert.equal(f.trigger.clicks, 1); f.populate(); await pending;
  assert.equal(f.trigger.clicks, 1); assert.equal(f.selected(), 1); assert.equal(f.observers.size, 0);
});

test('missing or disabled targets never cause a guessed row selection', async () => {
  const missing = chatFixture({ noChoice: true });
  await assert.rejects(selectInOfficialChat(missing.win, connection(), { timeout: 20 }), /Could Not Be Confirmed/);
  assert.equal(missing.selected(), 0); assert.equal(missing.trigger.clicks, 2); assert.equal(missing.observers.size, 0);
  const disabled = chatFixture({ delayedMenu: true });
  const pending = selectInOfficialChat(disabled.win, connection()); await tick(); disabled.populate();
  disabled.menu().querySelector('button').disabled = true;
  await assert.rejects(pending, /Picker Is Busy/); assert.equal(disabled.selected(), 0);
  const hidden = chatFixture({ delayedMenu: true }); const hiddenPending = selectInOfficialChat(hidden.win, connection()); await tick(); hidden.populate();
  hidden.menu().querySelector('button').hidden = true;
  await assert.rejects(hiddenPending, /Choice Changed/); assert.equal(hidden.selected(), 0);
});

test('aborting or detaching the picker stops the observer and never retries a native click', async () => {
  const f = chatFixture({ delayedMenu: true }), controller = new AbortController();
  const pending = selectInOfficialChat(f.win, connection(), { signal: controller.signal }); await tick();
  assert.equal(f.trigger.clicks, 1); controller.abort(); await assert.rejects(pending, { name: 'AbortError' });
  f.populate(); await tick(); assert.equal(f.selected(), 0); assert.equal(f.trigger.clicks, 1); assert.equal(f.observers.size, 0);
  const detached = chatFixture({ delayedMenu: true }); const p = selectInOfficialChat(detached.win, connection()); await tick();
  detached.trigger.remove(); await assert.rejects(p, /Picker Changed/); assert.equal(detached.selected(), 0);
});

test('navigation interruption fails closed; missing Chat UI has one deadline and no native action', async () => {
  const f = chatFixture({ delayedMenu: true }); const p = selectInOfficialChat(f.win, connection()); await tick();
  f.win.location.hash = '#/settings'; f.populate(); await assert.rejects(p, /Navigation Changed/); assert.equal(f.selected(), 0);
  const absent = domFixture(); await assert.rejects(selectInOfficialChat(absent.win, connection(), { timeout: 20 }), /Could Not Be Confirmed/);
  assert.equal(absent.observers.size, 0);
});

test('a cancelled DOM wait disconnects and a detached source is rejected before navigation', async () => {
  const f = domFixture(), controller = new AbortController();
  const waiting = waitForDom(f.win, () => null, { signal: controller.signal }); controller.abort();
  await assert.rejects(waiting, { name: 'AbortError' }); assert.equal(f.observers.size, 0);
  await assert.rejects(selectInOfficialChat(f.win, connection(), { source: f.element('div') }), /View Was Closed/);
  assert.equal(f.win.location.hash, '#/models');
});

test('the selection coordinator cancels on extension disposal and allows only one explicit handoff at a time', async () => {
  const f = domFixture(), states = []; let signal, release, calls = 0;
  const coordinator = createSelectionCoordinator({ win: f.win, onState: state => states.push(state), select: async (_win, value, options) => {
    calls++; assert.deepEqual(value, connectedSelection(connection())); signal = options.signal; await new Promise(resolve => { release = resolve; });
  } });
  const first = coordinator.useInChat(connection());
  await assert.rejects(coordinator.useInChat(connection()), /Already In Progress/); assert.equal(calls, 1);
  coordinator.dispose(); assert.equal(signal.aborted, true); release(); await first;
  assert.equal(states.length, 1); assert.equal(states[0].busy, true);
  await assert.rejects(coordinator.useInChat(connection()), /Was Closed/);
});

function nativeSurfaces(f) {
  const dialog = f.element('section', { class: 'dialog-surface', role: 'dialog' });
  const header = f.element('div', { class: 'dialog-head__actions' }), close = f.element('button', { title: '关闭' });
  close.addEventListener('click', () => dialog.remove()); header.append(close);
  const manage = f.element('div', { class: 'native-manage' });
  const empty = f.element('p', { class: 'native-manage-empty' }, 'No downloaded models');
  const foot = f.element('div', { class: 'native-manage-foot' }, 'Use A Local GGUF…'); manage.append(empty, foot); dialog.append(header, manage); f.doc.body.append(dialog);
  const storageCard = f.element('div', { class: 'st-card--flush' }), storage = f.element('section', { class: 'storage-directory' });
  const path = f.element('code', { class: 'storage-directory__path' }, 'C:\\Models\\Strata');
  const optimized = f.element('code', { class: 'storage-directory__path' }, 'C:\\Models\\Strata');
  const input = f.element('input', { id: 'model-directory-path' }); input.value = 'C:\\Models\\Strata';
  const versions = f.element('p', { role: 'status' }, '0 Versions Found');
  storage.append(path, optimized, input); storageCard.append(storage, versions); f.doc.body.append(storageCard);
  return { dialog, manage, foot, empty, close, storageCard, storage, path, optimized, input, versions };
}

function tileApi({ get: overrideGet, post: overridePost } = {}) {
  const calls = []; let data = model();
  const api = {
    async get(action, signal) { calls.push({ method: 'GET', action, signal }); return overrideGet ? overrideGet(action, signal) : action === 'status' ? status : structuredClone(data); },
    async post(action, value, signal) { calls.push({ method: 'POST', action, value, signal }); return overridePost ? overridePost(action, value, signal) : action === 'connect' ? connection() : {}; },
  };
  return { api, calls, setModel: value => { data = value; } };
}

test('Storage with two native directory sections has one connected tile per panel and preserves both sections', async t => {
  const f = domFixture(), native = nativeSurfaces(f), api = tileApi(); native.dialog.remove();
  native.storageCard.setAttribute('role', 'tabpanel'); native.storageCard.id = 'settings-panel-Storage';
  const directory = f.element('section', { class: 'storage-directory' });
  const heading = f.element('h3', {}, 'Model Directory');
  const path = f.element('code', { class: 'storage-directory__path' }, 'D:\\Local Models');
  const input = f.element('input'); input.value = 'D:\\Local Models';
  directory.append(heading, path, input); native.storageCard.append(directory);
  const observer = observeConnectedModels({ win: f.win, apiFactory: () => api.api, onUseInChat() {} }); t.after(() => observer.dispose());
  await tick(); const hosts = native.storageCard.querySelectorAll('strata-connected-local-models');
  assert.equal(hosts.length, 1);
  observer.reconcile(); await tick(); assert.equal(native.storageCard.querySelectorAll('strata-connected-local-models').length, 1);
  assert.deepEqual(native.storageCard.querySelectorAll('.storage-directory'), [native.storage, directory]);
  assert.equal(native.path.textContent, 'C:\\Models\\Strata'); assert.equal(native.optimized.textContent, 'C:\\Models\\Strata');
  assert.equal(native.input.value, 'C:\\Models\\Strata'); assert.equal(native.versions.textContent, '0 Versions Found');
  assert.equal(heading.textContent, 'Model Directory'); assert.equal(path.textContent, 'D:\\Local Models'); assert.equal(input.value, 'D:\\Local Models');
  assert.ok(api.calls.every(call => call.method === 'GET'));
  native.storage.remove(); await tick();
  assert.equal(native.storageCard.querySelector('strata-connected-local-models'), hosts[0], 'The Panel Anchor Survives A Directory Section Replacement');
  native.storageCard.remove(); await tick(); assert.equal(hosts[0].isConnected, false);
  f.doc.body.append(native.storageCard); await tick();
  assert.equal(native.storageCard.querySelectorAll('strata-connected-local-models').length, 1);
  assert.notEqual(native.storageCard.querySelector('strata-connected-local-models'), hosts[0]);
});

test('Manage Models and Storage show the one real connected model in isolated tiles without changing native counts or paths', async t => {
  const f = domFixture(), native = nativeSurfaces(f), api = tileApi();
  const data = model(); data.model.name = 'Example Local Model'; api.setModel(data);
  const observer = observeConnectedModels({ win: f.win, apiFactory: () => api.api, onUseInChat() {}, theme: 'light' }); t.after(() => observer.dispose());
  await tick(); const hosts = f.doc.querySelectorAll('strata-connected-local-models'); assert.equal(hosts.length, 2);
  assert.equal(native.manage.children.at(-2), hosts[0]); assert.equal(native.manage.children.at(-1), native.foot);
  assert.equal(native.storage.nextSibling, native.versions); assert.equal(native.versions.nextSibling, hosts[1]);
  for (const host of hosts) {
    const root = host.shadowRoot;
    assert.equal(host.dataset.theme, 'light'); assert.equal(root.querySelectorAll('strong').length, 1);
    assert.equal(root.querySelector('strong').textContent, 'Example Local Model'); assert.equal(root.querySelector('code').textContent, 'physical-model-id');
    assert.match(root.textContent, /Files Managed By Local Server/); assert.match(root.textContent, /Connected Local Model/);
    assert.match(root.querySelector('[role="status"]').textContent, /^Unloaded · Available On Demand/);
    assert.equal(root.querySelectorAll('button').find(button => button.textContent === 'Unload Model').disabled, true);
  }
  assert.equal(native.empty.textContent, 'No downloaded models'); assert.equal(native.versions.textContent, '0 Versions Found');
  assert.equal(native.path.textContent, 'C:\\Models\\Strata'); assert.equal(native.optimized.textContent, 'C:\\Models\\Strata'); assert.equal(native.input.value, 'C:\\Models\\Strata');
  assert.ok(api.calls.every(call => call.method === 'GET')); assert.equal(native.close.clicks, 0); assert.equal(f.win.location.hash, '#/models');
  observer.setTheme('dark'); assert.ok(hosts.every(host => host.dataset.theme === 'dark'));
});

test('connected tiles render literal metadata, optional files directory and truthful unconfirmed/offline state', async t => {
  const f = domFixture(), native = nativeSurfaces(f), api = tileApi(); native.dialog.remove();
  const data = model(); data.model.name = '<svg onload=alert(1)>'; data.model.filesDirectory = 'C:\\Models\\Strata\\weights'; data.model.loaded = true; api.setModel(data);
  const observer = observeConnectedModels({ win: f.win, apiFactory: () => api.api, onUseInChat() {} }); t.after(() => observer.dispose());
  await tick(); const root = native.storageCard.querySelector('strata-connected-local-models').shadowRoot;
  assert.equal(root.querySelector('strong').textContent, data.model.name); assert.match(root.textContent, /Model Files: C:\\Models\\Strata\\weights/);
  assert.match(root.querySelector('[role="status"]').textContent, /^Loaded · Ready To Chat/);
  data.available = false; api.setModel(data); root.querySelectorAll('button').find(button => button.textContent === 'Refresh Status').click(); await tick();
  assert.match(root.querySelector('[role="status"]').textContent, /Loaded State Unconfirmed/);
  assert.equal(root.querySelector('strong').textContent, 'Connected Model Unconfirmed');
  assert.ok(root.querySelectorAll('button').filter(button => button.textContent !== 'Refresh Status').every(button => button.disabled));
  assert.equal(native.input.value, 'C:\\Models\\Strata');
});

test('hidden or detached native surfaces stop reads, abort outstanding requests and accept replacement surfaces', async t => {
  const f = domFixture(), native = nativeSurfaces(f), pending = [];
  const api = tileApi({ get(action, signal) { return new Promise(resolve => pending.push({ action, signal, resolve })); } });
  native.dialog.hidden = true; native.storageCard.hidden = true;
  const observer = observeConnectedModels({ win: f.win, apiFactory: () => api.api, onUseInChat() {} }); t.after(() => observer.dispose());
  await tick(); assert.equal(api.calls.length, 0);
  native.dialog.hidden = false; await tick(); assert.equal(pending.length, 2);
  native.dialog.remove(); await tick(); assert.ok(pending.every(request => request.signal.aborted));
  for (const request of pending) request.resolve(request.action === 'status' ? status : model());
  await tick(); assert.equal(f.doc.querySelectorAll('strata-connected-local-models').length, 0);
  native.storageCard.hidden = false; await tick(); assert.equal(pending.length, 4);
  f.doc.visibilityState = 'hidden'; f.doc.dispatchEvent(new Event('visibilitychange')); assert.ok(pending.every(request => request.signal.aborted));
  native.storageCard.querySelector('strata-connected-local-models').shadowRoot.querySelectorAll('button').find(button => button.textContent === 'Refresh Status').click();
  assert.equal(api.calls.length, 4); assert.ok(api.calls.every(call => call.method === 'GET'));
});

test('unsupported Manage Models structure reports once, recovers when its footer returns, and never changes native contents', async t => {
  const f = domFixture(), native = nativeSurfaces(f), api = tileApi(), errors = [];
  native.storageCard.remove(); native.foot.remove();
  const observer = observeConnectedModels({ win: f.win, apiFactory: () => api.api, onUseInChat() {}, onError: error => errors.push(error.message) }); t.after(() => observer.dispose());
  observer.reconcile(); assert.equal(errors.length, 1); assert.match(errors[0], /Use Current Model/); assert.equal(api.calls.length, 0);
  native.manage.append(native.foot); await tick();
  assert.equal(native.manage.querySelectorAll('strata-connected-local-models').length, 1); assert.equal(api.calls.length, 2); assert.equal(native.empty.textContent, 'No downloaded models');
  const host = native.manage.querySelector('strata-connected-local-models'); host.remove(); await tick();
  assert.equal(native.manage.querySelectorAll('strata-connected-local-models').length, 1, 'A Replaced Target Recovers Without Duplicating A Tile');
  assert.notEqual(native.manage.querySelector('strata-connected-local-models'), host);
});

test('native tile Use In Chat is explicit, closes only its source dialog and selects the verified physical target', async t => {
  const f = chatFixture({ native: true }), native = nativeSurfaces(f), api = tileApi(); native.storageCard.remove();
  const observer = observeConnectedModels({ win: f.win, apiFactory: () => api.api,
    onUseInChat: (value, source) => selectInOfficialChat(f.win, value, { source }) }); t.after(() => observer.dispose());
  await tick(); assert.equal(f.trigger.clicks, 0); assert.equal(f.selected(), 0); assert.ok(api.calls.every(call => call.method === 'GET'));
  native.manage.querySelector('strata-connected-local-models').shadowRoot.querySelector('button').click(); await tick();
  assert.deepEqual(api.calls.filter(call => call.method === 'POST').map(({ action, value }) => ({ action, value })), [{ action: 'connect', value: undefined }]);
  assert.equal(native.close.clicks, 1); assert.equal(f.trigger.clicks, 1); assert.equal(f.selected(), 1);
  assert.equal(f.doc.querySelectorAll('strata-connected-local-models').length, 0);
});

test('closing Manage Models during connect cancels the response; no hidden picker selection or mutation retry occurs', async t => {
  const f = chatFixture(), native = nativeSurfaces(f); native.storageCard.remove(); let release, postSignal, handoffs = 0;
  const api = tileApi({ post(_action, _value, signal) { postSignal = signal; return new Promise(resolve => { release = resolve; }); } });
  const observer = observeConnectedModels({ win: f.win, apiFactory: () => api.api, onUseInChat() { handoffs++; } }); t.after(() => observer.dispose());
  await tick(); native.manage.querySelector('strata-connected-local-models').shadowRoot.querySelector('button').click();
  native.dialog.remove(); await tick(); assert.equal(postSignal.aborted, true); release(connection()); await tick();
  assert.equal(handoffs, 0); assert.equal(f.trigger.clicks, 0); assert.equal(api.calls.filter(call => call.method === 'POST').length, 1);
});

test('the mounted entry wires native tiles, verified iframe handoff, feedback and disposal together', async t => {
  const f = chatFixture({ native: true }), native = nativeSurfaces(f), calls = [];
  f.win.fetch = async (url, options) => {
    calls.push({ url, ...options }); assert.equal(new URL(url).origin, ORIGIN);
    return { ok: true, json: async () => url.endsWith('/status') ? status : url.endsWith('/connect') ? connection() : model() };
  };
  const host = mountExtension(f.win); t.after(() => host.remove()); await tick();
  assert.equal(mountExtension(f.win), host); assert.equal(f.doc.querySelectorAll('strata-connected-local-models').length, 2);
  assert.ok(calls.every(call => call.method === 'GET')); assert.equal(f.selected(), 0);
  // An explicit Storage action uses the same coordinator and leaves directory inputs intact.
  native.dialog.remove(); await tick();
  native.storageCard.querySelector('strata-connected-local-models').shadowRoot.querySelector('button').click(); await tick();
  assert.equal(f.selected(), 1); assert.match(host.shadowRoot.querySelector('aside').textContent, /Selected In Chat/);
  assert.equal(native.input.value, 'C:\\Models\\Strata'); assert.equal(native.versions.textContent, '0 Versions Found');
  assert.deepEqual(calls.filter(call => call.method === 'POST').map(call => ({ path: new URL(call.url).pathname, body: call.body })),
    [{ path: '/__strata_extension/api/connect', body: '{"expectedModel":"physical-model-id"}' }]);
  host.controller.open(); const frame = host.shadowRoot.querySelector('iframe');
  const message = new Event('message'); Object.assign(message, { origin: ORIGIN, source: frame.contentWindow,
    data: { type: 'strata-extension:open-chat', connection: connection() } }); f.win.dispatchEvent(message); await tick();
  assert.equal(host.shadowRoot.querySelector('dialog').open, false); assert.equal(f.selected(), 1, 'Already Confirmed Target Needs No Second Click');
  host.remove(); assert.equal(f.doc.querySelectorAll('strata-connected-local-models').length, 0); assert.equal(f.observers.size, 0);
});

test('a source dialog must really close before the adapter can touch the Chat picker', async () => {
  const f = chatFixture(), native = nativeSurfaces(f);
  // Keep a live modal after its standard Close click, as an incompatible app revision might.
  const sticky = f.element('button'); native.close.remove(); native.dialog.querySelector('.dialog-head__actions').append(sticky);
  await assert.rejects(selectInOfficialChat(f.win, connection(), { source: native.manage, timeout: 20 }), /Could Not Be Confirmed/);
  assert.equal(sticky.clicks, 1); assert.equal(f.trigger.clicks, 0); assert.equal(f.selected(), 0);
});
