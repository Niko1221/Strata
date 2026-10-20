import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { readFileSync } from 'node:fs';
import { createRemoteServing, createRemoteServingRoutes } from '../server/remote-serving.mjs';
import { managementPath, remoteTargetId } from '../server/model-management.mjs';
import { createLocalShare } from '../server/local-share.mjs';
import { HostClient } from '../../companion-bridge/bridge.mjs';

const secret = 'fixture-private-api-key';
const connection = () => ({ id: 'saved-server', name: 'Headless server', kind: 'custom', protocol: 'openai-chat', baseUrl: 'http://fixture.invalid/v1', apiKey: secret, models: [], modelManagement: { transport: 'ssh', host: 'unrelated-ssh-alias' } });
function fixture(options = {}) {
  const saved = connection(), calls = [];
  let removed = false, hook;
  const status = { service: 'strata', model: 'Configured model', loaded: false, started: 1000, activity: { in_flight: 0 }, api_key: secret, args: ['PRIVATE ARGUMENT'] };
  const telemetry = { model: { id: status.model, loaded: false }, instance: { id: 'instance-fixture', state: 'ready' }, work: { admission_reserved: 0, active: 0, queued: 0, waiting: 0 } };
  const fetcher = async (url, init) => {
    calls.push({ url, ...init });
    if (hook) { const value = await hook(url, init); if (value) return value; }
    if (init.method === 'GET') return Response.json(url.endsWith('/status') ? status : telemetry);
    const loaded = url.endsWith('/load'); status.loaded = loaded; telemetry.model.loaded = loaded;
    return Response.json({ status: loaded ? 'loaded' : 'unloaded', ...status });
  };
  const service = createRemoteServing({ providers: { withSecrets: () => removed ? [] : [structuredClone(saved)] }, fetcher, ...options });
  return { saved, calls, status, telemetry, service, target: remoteTargetId(saved), hook: value => { hook = value; }, remove: () => { removed = true; } };
}
const posts = f => f.calls.filter(call => call.method === 'POST');
async function listen(t, handler) {
  const server = http.createServer(handler);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(async () => { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); });
  return { server, port: server.address().port, origin: `http://127.0.0.1:${server.address().port}` };
}
async function raw(port, path, { method = 'GET', headers = {}, body } = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: '127.0.0.1', port, path, method, headers: { host: `127.0.0.1:${port}`, ...headers } }, res => {
      let text = ''; res.on('data', chunk => { text += chunk; }); res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, text, body: text ? JSON.parse(text) : null }));
    });
    req.on('error', reject); req.end(body === undefined ? undefined : JSON.stringify(body));
  });
}

test('strict snapshots export display fields only and use the saved API independently of SSH settings', async () => {
  const f = fixture(), view = await f.service.snapshot(f.target);
  assert.deepEqual(view, { supported: true, targetName: 'Headless server', modelId: 'Configured model', loaded: false, busy: false, capabilities: { load: true, unload: false } });
  assert.equal(JSON.stringify(view).includes(secret), false); assert.equal(JSON.stringify(view).includes('PRIVATE'), false);
  assert.deepEqual(f.calls.map(call => call.url), ['http://fixture.invalid/v1/status', 'http://fixture.invalid/v1/telemetry']);
  assert.ok(f.calls.every(call => call.headers.authorization === `Bearer ${secret}` && call.redirect === 'error'));
  assert.equal(posts(f).length, 0);
});

test('missing keys, ordinary providers and loaded-unknown legacy servers remain read-only without probing model controls', async () => {
  for (const mode of ['key', 'ordinary', 'model', 'loaded']) {
    const f = fixture();
    if (mode === 'key') f.saved.apiKey = '';
    if (mode === 'ordinary') f.status.service = 'other';
    if (mode === 'model') delete f.status.model;
    if (mode === 'loaded') delete f.status.loaded;
    const state = await f.service.snapshot(f.target);
    assert.equal(state.supported, false); assert.equal(state.loaded, null); assert.deepEqual(state.capabilities, { load: false, unload: false }); assert.ok(state.reason);
    await assert.rejects(f.service.plan(f.target, { action: 'load' }), { status: 409 });
    assert.equal(f.calls.some(call => call.url.endsWith('/telemetry')), false); assert.equal(posts(f).length, 0);
    if (mode === 'key') assert.equal(f.calls.length, 0);
  }
});

test('busy admission, active, queued, waiting, in-flight and draining state all prevent previews; missing admission stays unknown', async () => {
  for (const field of ['admission_reserved', 'active', 'queued', 'waiting', 'in_flight', 'draining', 'missing']) {
    const f = fixture();
    if (field === 'missing') delete f.telemetry.work.admission_reserved;
    else if (field === 'in_flight') f.status.activity.in_flight = 1;
    else if (field === 'draining') f.telemetry.instance.state = 'draining';
    else f.telemetry.work[field] = 1;
    const view = await f.service.snapshot(f.target);
    assert.equal(view.busy, field === 'missing' ? null : true); assert.deepEqual(view.capabilities, { load: false, unload: false }); assert.ok(view.reason);
    await assert.rejects(f.service.plan(f.target, { action: 'load' }), { status: 409 }); assert.equal(posts(f).length, 0);
  }
  const f = fixture(); f.hook(url => url.endsWith('/telemetry') ? Response.json({}, { status: 404 }) : null);
  const view = await f.service.snapshot(f.target); assert.equal(view.busy, null); assert.equal(view.capabilities.load, false);
});

test('load and unload require reviewed one-use previews and post only the derived current model with zero unload wait', async () => {
  const f = fixture();
  const plan = await f.service.plan(f.target, { action: 'load' });
  assert.equal(plan.action, 'load'); assert.equal(plan.modelId, f.status.model); assert.ok(plan.warnings.some(text => /hooks/.test(text)));
  assert.equal(posts(f).length, 0);
  const loaded = await f.service.execute(f.target, { planId: plan.planId });
  assert.equal(loaded.loaded, true); assert.equal(posts(f)[0].url, 'http://fixture.invalid/v1/load'); assert.deepEqual(JSON.parse(posts(f)[0].body), { model: f.status.model });
  await assert.rejects(f.service.execute(f.target, { planId: plan.planId }), { status: 409 });
  const unload = await f.service.plan(f.target, { action: 'unload' }); assert.ok(unload.warnings.some(text => /automatically load/.test(text)));
  assert.equal((await f.service.execute(f.target, { planId: unload.planId })).loaded, false);
  assert.equal(posts(f)[1].url, 'http://fixture.invalid/v1/unload?wait=0'); assert.deepEqual(JSON.parse(posts(f)[1].body), { model: f.status.model });
  assert.equal(JSON.stringify([plan, loaded, unload]).includes(secret), false);
});

test('unknown fields, arbitrary model IDs/actions, paths and stale previews never reach a model control', async () => {
  let clock = 1000; const f = fixture({ now: () => clock });
  for (const body of [{ action: 'restart' }, { action: 'load', model: 'other-model' }, { action: 'load', url: 'https://other.invalid' }, [], null]) await assert.rejects(f.service.plan(f.target, body), { status: 400 });
  const plan = await f.service.plan(f.target, { action: 'load' }); clock += 60_001;
  await assert.rejects(f.service.execute(f.target, { planId: plan.planId }), { status: 409 });
  await assert.rejects(f.service.execute(f.target, { planId: 'invented', action: 'load' }), { status: 400 });
  await assert.rejects(f.service.snapshot('local'), { status: 404 });
  assert.equal(posts(f).length, 0);
});

test('fresh execution refuses model swaps, loaded-state changes, replaced instances and newly admitted work', async () => {
  for (const field of ['model', 'loaded', 'instance', 'started', 'busy']) {
    const f = fixture(), plan = await f.service.plan(f.target, { action: 'load' });
    if (field === 'model') f.status.model = f.telemetry.model.id = 'Other model';
    if (field === 'loaded') f.status.loaded = f.telemetry.model.loaded = true;
    if (field === 'instance') f.telemetry.instance.id = 'replacement-instance';
    if (field === 'started') f.status.started++;
    if (field === 'busy') f.telemetry.work.admission_reserved = 1;
    await assert.rejects(f.service.execute(f.target, { planId: plan.planId }), { status: 409 }); assert.equal(posts(f).length, 0);
  }
});

test('API base, key, protocol, removal and in-flight connection edits invalidate previews even with SSH configured', async () => {
  for (const change of ['baseUrl', 'apiKey', 'protocol', 'remove', 'awaited']) {
    const f = fixture(), plan = await f.service.plan(f.target, { action: 'load' });
    if (change === 'remove') f.remove();
    else if (change === 'awaited') f.hook(() => { f.saved.apiKey = 'edited-private-key'; });
    else f.saved[change] = change === 'protocol' ? 'anthropic' : change === 'baseUrl' ? 'http://another.invalid/v1' : 'edited-private-key';
    await assert.rejects(f.service.execute(f.target, { planId: plan.planId }), cause => [404, 409].includes(cause.status) && !cause.message.includes(secret));
    assert.equal(posts(f).length, 0);
  }
});

test('previews are consumed before asynchronous execution and disconnect prevents a later POST', async () => {
  const f = fixture(), plan = await f.service.plan(f.target, { action: 'load' });
  const result = await Promise.allSettled([f.service.execute(f.target, { planId: plan.planId }), f.service.execute(f.target, { planId: plan.planId })]);
  assert.equal(result.filter(item => item.status === 'fulfilled').length, 1); assert.equal(posts(f).length, 1);
  const second = fixture(), preview = await second.service.plan(second.target, { action: 'load' }), controller = new AbortController();
  second.hook(url => { if (url.endsWith('/telemetry')) controller.abort(); });
  await assert.rejects(second.service.execute(second.target, { planId: preview.planId }, { signal: controller.signal }), { status: 499 });
  assert.equal(posts(second).length, 0);
});

test('malformed, oversized and secret-bearing error responses are bounded and sanitized', async () => {
  for (const mode of ['malformed', 'oversized', 'unauthorized', 'busy']) {
    const f = fixture({ maxResponseBytes: 300 });
    f.hook(() => mode === 'malformed' ? new Response('PRIVATE RAW NOT JSON', { headers: { 'content-type': 'application/json' } })
      : mode === 'oversized' ? Response.json({ secret: secret.repeat(1000) }) : Response.json({ error: `${secret} /private/model/path` }, { status: mode === 'unauthorized' ? 401 : 409 }));
    if (['malformed', 'oversized'].includes(mode)) { const state = await f.service.snapshot(f.target); assert.equal(state.supported, false); assert.ok(!JSON.stringify(state).includes(secret)); }
    else await assert.rejects(f.service.snapshot(f.target), cause => cause.status === (mode === 'unauthorized' ? 401 : 409) && !cause.message.includes(secret) && !cause.message.includes('/private'));
  }
});

test('a status/telemetry model change and nonzero legacy busy flags cannot produce an actionable preview', async () => {
  for (const field of ['model', 'loaded', 'busy', 'queued']) {
    const f = fixture();
    if (field === 'model') f.telemetry.model.id = 'Replacement';
    if (field === 'loaded') f.telemetry.model.loaded = true;
    if (field === 'busy') f.status.busy = true;
    if (field === 'queued') f.status.queued = 1;
    await assert.rejects(f.service.plan(f.target, { action: 'load' }), { status: 409 }); assert.equal(posts(f).length, 0);
  }
});

test('real HTTP response deadlines also bound a body that never finishes', async t => {
  const remote = await listen(t, (req, res) => { res.writeHead(200, { 'content-type': 'application/json' }); res.write('{"service":'); });
  const saved = connection(); saved.baseUrl = remote.origin;
  const service = createRemoteServing({ providers: { withSecrets: () => [saved] }, readTimeoutMs: 30 });
  await assert.rejects(service.snapshot(remoteTargetId(saved)), { status: 504 });
});

test('real HTTP controls preserve auth and fixed paths for both Anthropic root/v1 and OpenAI bases, without redirects or inference prompts', async t => {
  const requests = []; let loaded = false;
  const remote = await listen(t, async (req, res) => {
    let raw = ''; for await (const chunk of req) raw += chunk;
    requests.push({ path: req.url, method: req.method, key: req.headers.authorization, body: raw ? JSON.parse(raw) : null });
    if (req.url === '/redirect/v1/status') { res.writeHead(302, { location: '/must-never-receive-key' }); return res.end(); }
    res.setHeader('content-type', 'application/json');
    if (req.url === '/v1/load' || req.url === '/v1/unload?wait=0') loaded = req.url === '/v1/load';
    res.end(JSON.stringify(req.url === '/v1/telemetry' ? { model: { id: 'Configured model', loaded }, instance: { id: 'http-instance', state: 'ready' }, work: { admission_reserved: 0, active: 0, queued: 0 } }
      : { service: 'strata', model: 'Configured model', loaded, started: 100, activity: { in_flight: 0 } }));
  });
  for (const [protocol, path] of [['anthropic', ''], ['anthropic', '/v1'], ['openai-chat', '/v1'], ['openai-responses', '']]) {
    loaded = false; const f = fixture(); f.saved.protocol = protocol; f.saved.baseUrl = remote.origin + path;
    const service = createRemoteServing({ providers: { withSecrets: () => [f.saved] } });
    const plan = await service.plan(f.target, { action: 'load' }); assert.equal((await service.execute(f.target, { planId: plan.planId })).loaded, true);
    const unload = await service.plan(f.target, { action: 'unload' }); assert.equal((await service.execute(f.target, { planId: unload.planId })).loaded, false);
  }
  assert.ok(requests.every(request => request.key === `Bearer ${secret}`));
  assert.ok(requests.filter(request => request.method === 'POST').every(request => ['/v1/load', '/v1/unload?wait=0'].includes(request.path) && Object.keys(request.body).join() === 'model'));
  const saved = connection(); saved.baseUrl = remote.origin + '/redirect';
  await assert.rejects(createRemoteServing({ providers: { withSecrets: () => [saved] } }).snapshot(remoteTargetId(saved)), { status: 503 });
  assert.equal(requests.some(request => request.path.includes('must-never')), false);
});

test('local routes reject cross-origin, rebinding, phone headers, query strings and unsupported methods before the service', async t => {
  const calls = [], service = Object.fromEntries(['snapshot', 'plan', 'execute'].map(action => [action, async (...args) => { calls.push({ action, args }); return { ok: true }; }]));
  let routes;
  const local = await listen(t, (req, res) => { res.setHeader('access-control-allow-origin', '*'); routes(req, res, new URL(req.url, 'http://localhost')); });
  routes = createRemoteServingRoutes({ service, port: local.port });
  const path = `/api/remote-serving/targets/${remoteTargetId(connection())}`;
  for (const options of [{ headers: { origin: 'https://other.invalid' } }, { headers: { origin: 'http://127.0.0.1:1' } }, { headers: { host: 'evil.invalid' } }, { headers: { 'x-strata-companion-device': 'a'.repeat(32) } }, { headers: { 'x-strata-companion-other': '' } }]) {
    const result = await raw(local.port, path, options); assert.equal(result.status, 403); assert.equal(result.headers['access-control-allow-origin'], undefined);
  }
  for (const suffix of ['?anything=1', '?', '/start', '/plan?x=1']) assert.equal((await raw(local.port, path + suffix)).status, 404);
  assert.equal((await raw(local.port, path, { method: 'OPTIONS' })).status, 404); assert.equal(calls.length, 0);
  assert.equal((await raw(local.port, path)).status, 200);
  assert.equal((await raw(local.port, path + '/plan', { method: 'POST', headers: { 'content-type': 'text/plain' }, body: { action: 'load' } })).status, 415);
  const planned = await raw(local.port, path + '/plan', { method: 'POST', headers: { 'content-type': 'application/json', origin: local.origin }, body: { action: 'load' } });
  assert.equal(planned.status, 200); assert.deepEqual(calls.at(-1).args[1], { action: 'load' }); assert.equal(planned.headers['cache-control'], 'no-store');
});

test('a real local client disconnect during refreshed status aborts preflight before any model control', async t => {
  const f = fixture(), preview = await f.service.plan(f.target, { action: 'load' });
  let entered, cancelled;
  const waiting = new Promise(resolve => { entered = resolve; }), aborted = new Promise(resolve => { cancelled = resolve; });
  f.hook((url, init) => new Promise((resolve, reject) => {
    entered(); init.signal.addEventListener('abort', () => { cancelled(); reject(init.signal.reason); }, { once: true });
  }));
  let routes;
  const local = await listen(t, (req, res) => routes(req, res, new URL(req.url, 'http://localhost')));
  routes = createRemoteServingRoutes({ service: f.service, port: local.port });
  const path = `/api/remote-serving/targets/${f.target}`;
  const request = http.request({ host: '127.0.0.1', port: local.port, path, method: 'POST', headers: { 'content-type': 'application/json' } });
  request.on('error', () => {}); request.end(JSON.stringify({ planId: preview.planId }));
  await waiting; request.destroy(); await aborted;
  assert.equal(posts(f).length, 0);
});

test('phone bridge, sharing listener and remote administration never expose the new local serving namespace', async t => {
  let forwarded = 0;
  const path = `/api/remote-serving/targets/${remoteTargetId(connection())}`;
  const client = new HostClient({ upstream: 'http://127.0.0.1:1', fetcher: async () => { forwarded++; throw Error('Unexpected bridge request'); } });
  await assert.rejects(client.request(path, { deviceId: 'a'.repeat(32) }), { status: 403 });
  assert.equal(managementPath('/v1/strata/models/remote-serving'), false);
  const share = createLocalShare({ host: '127.0.0.1', port: 0, hasKey: () => true, allowModelManagement: () => true, handle: () => { forwarded++; } });
  t.after(() => share.stop()); const state = await share.start(); const port = Number(new URL(state.url).port);
  const response = await new Promise((resolve, reject) => { http.get({ hostname: '127.0.0.1', port, path }, res => { res.resume(); resolve(res.statusCode); }).on('error', reject); });
  assert.equal(response, 404); assert.equal(forwarded, 0);
  const index = readFileSync(new URL('../server/index.mjs', import.meta.url), 'utf8');
  assert.ok(index.indexOf("return remoteServingRoutes(req, res, url)") < index.indexOf("if (method === 'OPTIONS')"));
});
