import test from 'node:test';
import assert from 'node:assert/strict';
import https from 'node:https';
import http from 'node:http';
import { mkdtempSync, rmSync, statSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash, X509Certificate } from 'node:crypto';
import { Script } from 'node:vm';
import { createBridge, PairingAuthority, ensureIdentity, HostClient, validateUpstream, providerModels, normalizeModels } from '../bridge.mjs';

const temporary = () => mkdtempSync(join(tmpdir(), 'strata-companion-test-'));
const providers = { connections: [{ id: 'connection1', name: 'Saved Mac connection', baseUrl: 'https://private.example', apiKey: 'NEVER-ON-PHONE', apiKeySet: true, models: [{ id: 'test/model-one', name: 'Model One' }, { id: 'test/model-two', name: 'Model Two' }] }] };
function mockHost({ delay = 0, slow = 0, trailingNewline = true, complete = true } = {}) {
  const calls = [];
  return {
    calls,
    async json(path, options) {
      calls.push({ path, ...options });
      if (delay) await new Promise(resolve => setTimeout(resolve, delay));
      if (path === '/api/models') return { variants: [], loaded: null };
      if (path === '/api/native') return { instances: [], catalog: { families: [] } };
      if (path === '/api/monitor') return { phase: 'idle', outputTps: 12, hardware: { ramUsedGb: 8, ramTotalGb: 48 } };
      if (path === '/api/serving') return { requests: [{ state: 'generating' }] };
      if (path === '/api/model-providers') return providers;
      throw new Error(`Unexpected mock request ${path}`);
    },
    async request(path, options) {
      calls.push({ path, ...options });
      assert.equal(path, '/api/chat/stream');
      const frames = [{ type: 'start', requestId: 'req-1', path: '/private/internal', apiKey: 'NEVER-ON-PHONE' }, { type: 'delta', text: 'Hello from Strata' }, ...(complete ? [{ type: 'done', tokensOut: 4 }] : [])];
      if (slow) await new Promise((resolve, reject) => {
        const timer = setTimeout(resolve, slow);
        options.signal.addEventListener('abort', () => { clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')); }, { once: true });
      });
      return new Response(frames.map(frame => JSON.stringify(frame)).join('\n') + (trailingNewline ? '\n' : ''), { headers: { 'content-type': 'application/x-ndjson' } });
    },
  };
}

function request(bridge, path, { method = 'GET', token, body, headers = {}, admin = false, trusted = false } = {}) {
  return new Promise((resolve, reject) => {
    const client = admin ? http : https;
    const req = client.request({ hostname: '127.0.0.1', port: admin ? bridge.adminPort : bridge.port, path, method, agent: false, rejectUnauthorized: trusted, headers: { ...(token ? { authorization: `Bearer ${token}` } : {}), ...(body !== undefined ? { 'content-type': 'application/json' } : {}), ...headers } }, response => {
      let text = '';
      response.setEncoding('utf8'); response.on('data', chunk => { text += chunk; });
      response.on('end', () => {
        let json;
        try { json = JSON.parse(text); } catch {}
        resolve({ status: response.statusCode, text, json, headers: response.headers });
      });
    });
    req.on('error', reject);
    req.end(body === undefined ? undefined : JSON.stringify(body));
  });
}

async function fixture(t, options = {}) {
  const directory = temporary();
  const host = options.host ?? mockHost();
  const bridge = await createBridge({ directory, host, listen: '127.0.0.1', port: 0, adminPort: 0, advertiseHost: '127.0.0.1', qr: false, ...options });
  t.after(async () => { await bridge.close(); rmSync(directory, { recursive: true, force: true }); });
  return { bridge, host, directory };
}

async function pair(bridge, deviceName = 'Test iPhone') {
  const result = await request(bridge, '/v1/pair', { method: 'POST', body: { code: bridge.authority.code, deviceName } });
  assert.equal(result.status, 200); assert.match(result.json.token, /^[A-Za-z0-9_-]{43}$/);
  return result.json.token;
}

test('pairing is one-use, five-guesses globally, expires, and persists hashes privately', async t => {
  const directory = temporary(); t.after(() => rmSync(directory, { recursive: true, force: true }));
  let now = 1000;
  const authority = new PairingAuthority(directory, { now: () => now, ttl: 100, maxAttempts: 5 });
  const correct = authority.code;
  const wrong = correct === '000000' ? '000001' : '000000';
  for (let attempt = 0; attempt < 5; attempt++) assert.throws(() => authority.pair(wrong, 'Phone'), { status: 401 });
  assert.throws(() => authority.pair(correct, 'Phone'), { status: 410 });
  authority.rotate(); now += 101;
  assert.throws(() => authority.pair(authority.code, 'Phone'), { status: 410 });
  authority.rotate();
  const { token, session } = authority.pair(authority.code, 'Personal iPhone');
  assert.equal(authority.authorize(`Bearer ${token}`).id, session.id);
  assert.throws(() => authority.pair(authority.code, 'Replay'), { status: 410 });
  const file = join(directory, 'sessions.json');
  assert.equal(statSync(file).mode & 0o777, 0o600);
  assert.ok(!readFileSync(file, 'utf8').includes(token));
  const restored = new PairingAuthority(directory);
  assert.equal(restored.authorize(`Bearer ${token}`).id, session.id);
  restored.revoke(session.id);
  assert.equal(restored.authorize(`Bearer ${token}`), null);
});

test('TLS identity remains stable and invitation fingerprint pins the exact DER certificate', async t => {
  const { bridge, directory } = await fixture(t);
  const certHash = createHash('sha256').update(new X509Certificate(bridge.identity.cert).raw).digest('hex');
  assert.equal(bridge.identity.fingerprint, certHash);
  const invite = new URL(bridge.invitation());
  assert.equal(invite.searchParams.get('fingerprint'), certHash);
  assert.equal(invite.searchParams.get('url'), `https://127.0.0.1:${bridge.port}`);
  const stable = await ensureIdentity(directory);
  assert.equal(stable.fingerprint, certHash);
  assert.equal(statSync(directory).mode & 0o777, 0o700);
  assert.equal(statSync(join(directory, 'server-key.pem')).mode & 0o777, 0o600);
  await assert.rejects(request(bridge, '/v1/status', { trusted: true }), error => error.code === 'DEPTH_ZERO_SELF_SIGNED_CERT');
});

test('network routes require bearer auth, reject browser Origin and arbitrary proxy controls', async t => {
  const { bridge, host } = await fixture(t);
  assert.equal((await request(bridge, '/v1/status')).status, 401);
  assert.equal(host.calls.length, 0);
  const token = await pair(bridge);
  for (const path of ['/api/models', '/v1/proxy?url=https://attacker.example', '/v1/settings', '/v1/files', '/v1/shell']) assert.equal((await request(bridge, path, { token })).status, 404);
  assert.equal((await request(bridge, '/v1/status', { token, headers: { origin: 'https://attacker.example' } })).status, 403);
  assert.equal((await request(bridge, '/v1/chat', { method: 'POST', token, body: { message: 'x'.repeat(20_001) } })).status, 400);
  assert.equal(host.calls.length, 0);
  assert.equal((await request(bridge, '/v1/session', { method: 'DELETE', token })).status, 200);
  assert.equal((await request(bridge, '/v1/status', { token })).status, 401);
});

test('local admin resists DNS rebinding and cross-origin revocation and escapes device names', async t => {
  const { bridge } = await fixture(t);
  await pair(bridge, '<script>alert(1)</script>');
  const path = new URL(bridge.adminURL).pathname;
  assert.equal((await request(bridge, path, { admin: true, headers: { host: `attacker.example:${bridge.adminPort}` } })).status, 403);
  assert.equal((await request(bridge, path, { admin: true, headers: { origin: 'https://attacker.example' } })).status, 403);
  const page = await request(bridge, path, { admin: true });
  assert.equal(page.status, 200);
  assert.ok(page.text.includes('&lt;script&gt;alert(1)&lt;/script&gt;'));
  assert.ok(!page.text.includes('<script>alert(1)</script>'));
  assert.equal((await request(bridge, path + '/revoke', { admin: true, method: 'POST', body: { id: bridge.authority.sessions[0].id } })).status, 403);
  assert.equal(page.headers['x-frame-options'], 'DENY');
});

test('local pairing state refreshes after pair, removal and expiry without exposing credentials', async t => {
  let now = 1_000;
  const { bridge } = await fixture(t, { now: () => now, pairingTTL: 100 });
  const path = new URL(bridge.adminURL).pathname;
  const before = await request(bridge, path + '/state', { admin: true });
  assert.deepEqual(Object.keys(before.json).sort(), ['available', 'expiresAt', 'revision']);
  assert.equal(before.json.available, true);
  assert.equal(before.headers['cache-control'], 'no-store');
  assert.equal((await request(bridge, path + '/state', { admin: true, headers: { host: 'attacker.example' } })).status, 403);
  assert.equal((await request(bridge, path + '/state', { admin: true, headers: { origin: 'https://attacker.example' } })).status, 403);
  assert.equal((await request(bridge, '/admin/wrong/state', { admin: true })).status, 404);
  assert.equal((await request(bridge, path + '/state')).status, 404);
  const page = await request(bridge, path, { admin: true });
  for (const field of ['mac-address', 'mac-fingerprint', 'pairing-code']) assert.ok(page.text.includes(`id="${field}" readonly`));
  assert.ok(page.text.includes('setInterval(')); assert.ok(page.text.includes('3000'));
  assert.ok(page.text.includes('sessionStorage.setItem'));
  const script = page.text.match(/<script>([\s\S]*?)<\/script>/)[1];
  assert.doesNotThrow(() => new Script(script));
  const token = await pair(bridge);
  const paired = await request(bridge, path + '/state', { admin: true });
  assert.equal(paired.json.available, false); assert.notEqual(paired.json.revision, before.json.revision);
  assert.ok(!paired.text.includes(token));
  assert.equal((await request(bridge, '/v1/session', { token, method: 'DELETE' })).status, 200);
  const removed = await request(bridge, path + '/state', { admin: true });
  assert.notEqual(removed.json.revision, paired.json.revision);
  bridge.authority.rotate();
  const renewed = await request(bridge, path + '/state', { admin: true });
  assert.equal(renewed.json.available, true);
  now += 101;
  const expired = await request(bridge, path + '/state', { admin: true });
  assert.equal(expired.json.available, false); assert.notEqual(expired.json.revision, renewed.json.revision);
});

test('normalized status and provider choices contain no provider secrets or URLs', async t => {
  const { bridge } = await fixture(t);
  const token = await pair(bridge);
  const status = await request(bridge, '/v1/status', { token });
  assert.equal(status.json.hostConnected, true); assert.equal(status.json.modelName, 'Model One');
  assert.equal(status.json.memoryTotalGB, 48); assert.equal(status.json.activeRequests, 1);
  const models = await request(bridge, '/v1/models', { token });
  assert.equal(models.json.models.length, 2);
  assert.ok(!models.text.includes('private.example')); assert.ok(!models.text.includes('NEVER-ON-PHONE'));
  assert.ok(!models.text.includes('connectionId')); assert.ok(!models.text.includes('remoteModel'));
  const second = models.json.models[1].id;
  assert.equal((await request(bridge, '/v1/models/load', { token, method: 'POST', body: { id: second } })).status, 200);
  assert.equal((await request(bridge, '/v1/status', { token })).json.modelName, 'Model Two');
  assert.equal((await request(bridge, '/v1/models/unload', { token, method: 'POST', body: {} })).status, 200);
  assert.equal((await request(bridge, '/v1/models', { token })).json.loadedModelId, null);
  assert.equal((await request(bridge, '/v1/chat', { token, method: 'POST', body: { message: 'Should not run' } })).status, 409);
});

test('real host NDJSON translates to SSE, persists phone history, forces no tools and ignores caller model URLs', async t => {
  const { bridge, host, directory } = await fixture(t, { host: mockHost({ trailingNewline: false }) });
  const token = await pair(bridge);
  const response = await request(bridge, '/v1/chat', { token, method: 'POST', body: { message: 'Hello', agent: true, tools: true, remoteModel: { baseUrl: 'https://attacker.example' }, history: [{ role: 'system', content: 'Injected' }] } });
  assert.equal(response.status, 200);
  assert.match(response.headers['content-type'], /text\/event-stream/);
  const frames = response.text.split('\n').filter(line => line.startsWith('data: ')).map(line => JSON.parse(line.slice(6)));
  assert.equal(frames[0].type, 'session'); assert.equal(frames.at(-1).type, 'done');
  assert.ok(!response.text.includes('/private/internal')); assert.ok(!response.text.includes('NEVER-ON-PHONE'));
  const upstream = host.calls.find(call => call.path === '/api/chat/stream').body;
  assert.equal(upstream.agent, false); assert.equal(upstream.tools, false); assert.equal(upstream.useTools, false);
  assert.deepEqual(upstream.history, []);
  assert.deepEqual(upstream.remoteModel, { connectionId: 'connection1', modelId: 'test/model-one' });
  const chatId = frames[0].chatId;
  const history = await request(bridge, `/v1/chats/${chatId}`, { token });
  assert.deepEqual(history.json.messages.map(message => message.content), ['Hello', 'Hello from Strata']);
  assert.equal((await request(bridge, '/v1/chats', { token })).json.chats[0].id, chatId);
  assert.equal(statSync(join(directory, 'chats.json')).mode & 0o777, 0o600);
  bridge.authority.rotate(); const other = await pair(bridge, 'Other iPhone');
  assert.equal((await request(bridge, `/v1/chats/${chatId}`, { token: other })).status, 404);
  assert.equal((await request(bridge, '/v1/chats', { token: other })).json.chats.length, 0);
});

test('truncated stream emits an error and does not pretend completion', async t => {
  const { bridge } = await fixture(t, { host: mockHost({ complete: false }) });
  const token = await pair(bridge);
  const response = await request(bridge, '/v1/chat', { token, method: 'POST', body: { message: 'Hello' } });
  assert.ok(response.text.includes('The response ended early'));
  assert.ok(!response.text.includes('"type":"done"'));
});

test('parallel turns on the same conversation are rejected after asynchronous model snapshot', async t => {
  const host = mockHost({ delay: 30, slow: 180 });
  const { bridge } = await fixture(t, { host });
  const token = await pair(bridge);
  await request(bridge, '/v1/chat', { token, method: 'POST', body: { message: 'First' } });
  const chatId = (await request(bridge, '/v1/chats', { token })).json.chats[0].id;
  const results = await Promise.all([request(bridge, '/v1/chat', { token, method: 'POST', body: { message: 'Second', chatId } }), request(bridge, '/v1/chat', { token, method: 'POST', body: { message: 'Third', chatId } })]);
  assert.deepEqual(results.map(result => result.status).sort(), [200, 409]);
  assert.equal(host.calls.filter(call => call.path === '/api/chat/stream').length, 2);
});

test('removing a selected provider never silently sends a message to another model', async t => {
  const host = mockHost();
  const baseJSON = host.json.bind(host);
  let removed = false;
  host.json = (path, options) => path === '/api/model-providers' && removed ? Promise.resolve({ connections: [] }) : baseJSON(path, options);
  const { bridge } = await fixture(t, { host });
  const token = await pair(bridge);
  const second = (await request(bridge, '/v1/models', { token })).json.models[1].id;
  assert.equal((await request(bridge, '/v1/models/load', { token, method: 'POST', body: { id: second } })).status, 200);
  removed = true;
  assert.equal((await request(bridge, '/v1/models', { token })).json.loadedModelId, null);
  assert.equal((await request(bridge, '/v1/chat', { token, method: 'POST', body: { message: 'No fallback' } })).status, 409);
  assert.equal(host.calls.filter(call => call.path === '/api/chat/stream').length, 0);
});

test('native model default, explicit selection, chat and unload target the same ready instance', async t => {
  const native = {
    catalog: { families: [{ title: 'Fixture', sizes: [{ id: 'model-a', model: 'A', installed: true }, { id: 'model-b', model: 'B', installed: true }] }] },
    instances: [{ id: 'instance-a', model: 'model-a', status: 'ready', phase: 'idle-a' }, { id: 'instance-b', model: 'model-b', status: 'ready', phase: 'idle-b' }], defaultId: 'instance-b',
  };
  assert.equal(normalizeModels({ variants: [] }, native).loadedModelId, 'native-model-b');
  const host = mockHost();
  const baseJSON = host.json.bind(host);
  host.json = async (path, options) => {
    if (path === '/api/native') return structuredClone(native);
    if (path === '/api/model-providers') return { connections: [] };
    const unload = /^\/api\/native\/instances\/([^/]+)\/unload$/.exec(path);
    if (unload) { host.calls.push({ path, ...options }); native.instances = native.instances.filter(instance => instance.id !== unload[1]); return {}; }
    return baseJSON(path, options);
  };
  const { bridge } = await fixture(t, { host });
  const token = await pair(bridge);
  const initial = await request(bridge, '/v1/models', { token });
  assert.equal(initial.json.loadedModelId, 'native-model-b');
  assert.equal(initial.json.models.find(model => model.selected).id, 'native-model-b');
  assert.equal(initial.json.models.filter(model => model.selected).length, 1);
  assert.equal(initial.json.models.filter(model => model.loaded && model.state === 'loaded').length, 2);
  assert.equal((await request(bridge, '/v1/status', { token })).json.engineState, 'idle-b');
  await request(bridge, '/v1/chat', { token, method: 'POST', body: { message: 'Use Mac default' } });
  assert.equal(host.calls.filter(call => call.path === '/api/chat/stream').at(-1).body.nativeModelId, 'instance-b');
  assert.equal((await request(bridge, '/v1/models/load', { token, method: 'POST', body: { id: 'native-model-a' } })).status, 200);
  assert.equal(host.calls.filter(call => call.path === '/api/native/operations').length, 0);
  assert.equal(native.defaultId, 'instance-b');
  const selected = await request(bridge, '/v1/models', { token });
  assert.equal(selected.json.models.find(model => model.selected).id, 'native-model-a');
  assert.equal(selected.json.models.find(model => model.id === 'native-model-b').state, 'loaded');
  assert.equal((await request(bridge, '/v1/status', { token })).json.engineState, 'idle-a');
  assert.equal((await request(bridge, '/v1/status', { token })).json.modelName, 'Fixture A');
  await request(bridge, '/v1/chat', { token, method: 'POST', body: { message: 'Use my selected model', nativeModelId: 'instance-b' } });
  assert.equal(host.calls.filter(call => call.path === '/api/chat/stream').at(-1).body.nativeModelId, 'instance-a');
  assert.equal((await request(bridge, '/v1/models/unload', { token, method: 'POST', body: {} })).status, 200);
  assert.equal(host.calls.filter(call => call.path.endsWith('/unload')).at(-1).path, '/api/native/instances/instance-a/unload');
  assert.deepEqual(native.instances.map(instance => instance.id), ['instance-b']);
  assert.equal((await request(bridge, '/v1/models', { token })).json.loadedModelId, null);
  assert.equal((await request(bridge, '/v1/chat', { token, method: 'POST', body: { message: 'No fallback to B' } })).status, 409);
  await request(bridge, '/v1/models/unload', { token, method: 'POST', body: {} });
  assert.equal(host.calls.filter(call => call.path.endsWith('/unload')).length, 1);
});

test('native same-model instances prefer a matching ready desktop default, then another ready instance', () => {
  const native = {
    catalog: { families: [{ title: 'Fixture', sizes: [{ id: 'model-a', model: 'A', installed: true }, { id: 'model-b', model: 'B', installed: true }] }] },
    instances: [{ id: 'a-loading', model: 'model-a', status: 'loading' }, { id: 'a-ready', model: 'model-a', status: 'ready' }, { id: 'b-ready', model: 'model-b', status: 'ready' }], defaultId: 'b-ready',
  };
  const normalized = normalizeModels({ variants: [] }, native);
  assert.equal(normalized.loadedModelId, 'native-model-b');
  assert.equal(normalized.models.find(model => model.id === 'native-model-a').state, 'loaded');
  native.defaultId = 'a-ready';
  assert.equal(normalizeModels({ variants: [] }, native).loadedModelId, 'native-model-a');
  native.instances[1].status = 'loading';
  assert.equal(normalizeModels({ variants: [] }, native).models.find(model => model.id === 'native-model-a').loaded, false);
  assert.equal(normalizeModels({ variants: [] }, native).loadedModelId, 'native-model-b');
});

test('revocation during model snapshot prevents API selection, native load and native unload', async t => {
  for (const action of ['api-select', 'native-load', 'native-unload']) {
    let release; const gate = new Promise(resolve => { release = resolve; });
    let entered; const snapshotStarted = new Promise(resolve => { entered = resolve; });
    const mutations = [];
    const native = { catalog: { families: [{ title: 'Fixture', sizes: [{ id: 'model-a', model: 'A', installed: true }] }] }, instances: action === 'native-unload' ? [{ id: 'instance-a', model: 'model-a', status: 'ready' }] : [], defaultId: 'instance-a' };
    const host = {
      async json(path, options) {
        if (options?.method === 'POST') { mutations.push(path); return {}; }
        entered(); await gate;
        if (path === '/api/models') return { variants: [] };
        if (path === '/api/native') return native;
        if (path === '/api/model-providers') return providers;
        throw new Error('Unexpected snapshot route');
      },
    };
    const { bridge, directory } = await fixture(t, { host });
    const token = await pair(bridge);
    const session = bridge.authority.sessions[0];
    if (action === 'native-unload') session.selectedModel = 'native-model-a';
    const path = action === 'native-unload' ? '/v1/models/unload' : '/v1/models/load';
    const body = action === 'native-unload' ? {} : { id: action === 'api-select' ? providerModels(providers)[0].id : 'native-model-a' };
    const pending = request(bridge, path, { token, method: 'POST', body });
    await snapshotStarted;
    bridge.authority.revoke(session.id);
    release();
    const response = await pending;
    assert.equal(response.status, 401, action);
    assert.deepEqual(mutations, [], action);
    assert.deepEqual(bridge.authority.sessions, [], action);
    assert.deepEqual(JSON.parse(readFileSync(join(directory, 'sessions.json'), 'utf8')), [], action);
    assert.equal(bridge.authority.authorize(`Bearer ${token}`), null, action);
  }
});

test('upstream validation and host route/method guard reject remote targets and dangerous operations', async () => {
  for (const target of ['https://127.0.0.1:22', 'http://attacker.example:1234', 'http://127.0.0.1:47831/api/models', 'http://user:pass@127.0.0.1:47831/', 'http://127.0.0.1:47831/?url=evil']) assert.throws(() => validateUpstream(target));
  assert.equal(validateUpstream('http://127.0.0.1:47831'), 'http://127.0.0.1:47831');
  let calls = 0;
  const host = new HostClient({ upstream: 'http://127.0.0.1:47831', fetcher: async () => { calls++; return new Response('{}'); } });
  await assert.rejects(host.request('/api/chat/open-file', { method: 'POST', body: { path: '/tmp' } }), { status: 403 });
  await assert.rejects(host.request('/api/models', { method: 'DELETE' }), { status: 403 });
  await assert.rejects(host.request('/api/native/operations', { method: 'POST', body: { kind: 'remove', args: { model: 'q2_0' } } }), { status: 403 });
  await assert.rejects(host.request('/api/native/operations', { method: 'POST', body: { kind: 'load', args: { model: 'q2_0', values: { shell: 'unsafe' } } } }), { status: 403 });
  assert.equal(calls, 0);
  assert.equal(providerModels(providers)[0].id, providerModels(providers)[0].id);
});
