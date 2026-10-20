import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createModelManagement, createModelManagementProxy, createModelManagementRoutes, remoteTargetId } from '../server/model-management.mjs';

const fail = (status, message) => Object.assign(new Error(message), { status });
const phoneA = `phone:${'a'.repeat(32)}`, phoneB = `phone:${'b'.repeat(32)}`;
const modelId = 'native-q2_0';
const input = { modelId, readPictures: false };

function fakeNative() {
  const state = { available: true, catalog: { families: [{ family: 'qwen', title: 'Fixture', sizes: [{ id: 'q2_0', model: 'Q2_0', download_gb: 40, images: true, about: 'Test catalog model' }] }] }, installed: { models: [] }, instances: [], jobs: [] };
  const calls = { plan: [], operation: [], cancel: [], refresh: [] };
  const plan = { title: 'Fixture Q2_0', download_gb: 40, disk_free_gb: 100, disk_needed_gb: 48, disk_short: null, partly_downloaded: false };
  let refreshHook, planHook;
  const native = {
    state: () => state,
    refresh: async force => { calls.refresh.push(force); if (refreshHook) await refreshHook(force); return state; },
    plan: async args => { calls.plan.push(structuredClone(args)); if (planHook) return planHook(args); return { plan: structuredClone(plan) }; },
    operation: (kind, args) => {
      calls.operation.push({ kind, args: structuredClone(args) });
      const job = { id: `job-${calls.operation.length}`, kind, model: args.model, family: args.family, state: 'running', summary: 'Downloading model files…', result: { install: { download: { percent: 25, done_gb: 10, total_gb: 40 } } } };
      state.jobs.push(job); return job;
    },
    cancel: async id => { calls.cancel.push(id); const job = state.jobs.find(job => job.id === id); if (job) job.state = 'cancelled'; },
  };
  return { native, state, calls, plan, onRefresh: hook => { refreshHook = hook; }, onPlan: hook => { planHook = hook; } };
}

function fixture(options = {}) {
  const f = fakeNative();
  const service = createModelManagement({ native: f.native, name: 'Fixture Mac', modelsDirectory: () => '/private/models', browse: () => ({ entries: [] }), inspect: () => { throw new Error('No file selected'); }, adopt: () => {}, ...options });
  return { ...f, service };
}

async function listen(t, handler) {
  const server = http.createServer(handler);
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  t.after(async () => { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); });
  return { server, port: server.address().port, url: `http://127.0.0.1:${server.address().port}` };
}

async function json(url, { method = 'GET', body, headers = {} } = {}) {
  const response = await fetch(url, { method, redirect: 'error', headers: { ...(body !== undefined ? { 'content-type': 'application/json' } : {}), ...headers }, ...(body !== undefined ? { body: JSON.stringify(body) } : {}) });
  const text = await response.text();
  let value; try { value = JSON.parse(text); } catch {}
  return { status: response.status, value, text, headers: response.headers };
}

test('catalog preview maps only a server catalog ID and typed picture choice, without performing an operation', async () => {
  const { service, calls } = fixture();
  const catalog = await service.snapshot(phoneA);
  assert.equal(catalog.models[0].id, modelId); assert.equal(catalog.models[0].state, 'available'); assert.equal(catalog.models[0].readPictures, true);
  const plan = await service.downloadPlan(phoneA, { modelId, readPictures: true });
  assert.equal(plan.downloadGb, 40); assert.equal(plan.diskFreeGb, 100); assert.equal(plan.diskNeededGb, 48); assert.equal(plan.readPictures, true);
  assert.ok(plan.warnings.some(warning => /unloads/.test(warning))); assert.ok(plan.warnings.some(warning => /engine/.test(warning))); assert.ok(plan.warnings.some(warning => /not automatically loaded/.test(warning)));
  assert.deepEqual(calls.plan[0], { family: 'qwen', model: 'Q2_0', vision: 'yes', loadAfter: false, managedDownload: true });
  assert.deepEqual(calls.operation, []);
  for (const body of [{ modelId: 'https://attacker.example', readPictures: false }, { modelId, readPictures: 'false' }, { ...input, data_dir: '/private' }, { ...input, gguf_dir: '/private' }, { ...input, loadAfter: true }]) await assert.rejects(service.downloadPlan(phoneA, body), error => [400, 404].includes(error.status));
  assert.deepEqual(calls.operation, []);
});

test('download plans are owner-bound, expire, and are consumed before concurrent starts', async () => {
  let clock = 1000;
  const { service, calls } = fixture({ now: () => clock });
  const plan = await service.downloadPlan(phoneA, input);
  await assert.rejects(service.start(phoneB, { planId: plan.planId }), { status: 409 });
  const results = await Promise.allSettled([service.start(phoneA, { planId: plan.planId }), service.start(phoneA, { planId: plan.planId })]);
  assert.equal(results.filter(result => result.status === 'fulfilled').length, 1); assert.equal(calls.operation.length, 1);
  assert.equal(calls.operation[0].kind, 'install'); assert.equal(calls.operation[0].args.loadAfter, false);
  const expired = await service.downloadPlan(phoneA, input); clock += 300001;
  await assert.rejects(service.start(phoneA, { planId: expired.planId }), { status: 409 });
  assert.equal(calls.operation.length, 1);
});

test('disk shortage, busy work and catalog changes prevent installation and require a new preview', async () => {
  let busy = false;
  const { service, plan, state, calls } = fixture({ assertIdle: async () => { if (busy) throw fail(409, 'Busy'); } });
  plan.disk_short = 'Not enough disk space';
  const diskPlan = await service.downloadPlan(phoneA, input);
  assert.ok(diskPlan.diskShort); await assert.rejects(service.start(phoneA, { planId: diskPlan.planId }), { status: 409 });
  plan.disk_short = null;
  const busyPlan = await service.downloadPlan(phoneA, input); busy = true;
  await assert.rejects(service.start(phoneA, { planId: busyPlan.planId }), { status: 409 });
  busy = false;
  await assert.rejects(service.start(phoneA, { planId: busyPlan.planId }), { status: 409 });
  const changed = await service.downloadPlan(phoneA, input); state.catalog.families[0].sizes = [];
  await assert.rejects(service.start(phoneA, { planId: changed.planId }), { status: 409 });
  assert.deepEqual(calls.operation, []);
});

test('a materially changed download size or needed disk space must be reviewed again', async () => {
  for (const [field, value] of [['download_gb', 60], ['disk_needed_gb', 68]]) {
    const { service, plan, calls } = fixture();
    const preview = await service.downloadPlan(phoneA, input); plan[field] = value;
    await assert.rejects(service.start(phoneA, { planId: preview.planId }), { status: 409 }, field);
    assert.deepEqual(calls.operation, [], field);
  }
});

test('revocation or management-key disable during refresh and replan prevents the operation', async () => {
  for (const phase of ['refresh', 'replan', 'idle']) {
    let valid = true, idleChecks = 0;
    const f = fixture({ validateCaller: () => { if (!valid) throw fail(401, 'No longer authorized'); }, assertIdle: async () => { idleChecks++; if (phase === 'idle' && idleChecks === 2) valid = false; } });
    const preview = await f.service.downloadPlan(phoneA, input);
    if (phase === 'refresh') f.onRefresh(async force => { if (force) valid = false; });
    if (phase === 'replan') f.onPlan(async () => { valid = false; return { plan: f.plan }; });
    await assert.rejects(f.service.start(phoneA, { planId: preview.planId }), { status: 401 }, phase);
    assert.deepEqual(f.calls.operation, [], phase);
  }
  let enabled = true;
  const f = fixture({ validateCaller: owner => { if (owner.startsWith('remote:') && !enabled) throw fail(403, 'Disabled'); } });
  const preview = await f.service.downloadPlan('remote:management-key', input); enabled = false;
  await assert.rejects(f.service.start('remote:management-key', { planId: preview.planId }), { status: 403 });
  assert.deepEqual(f.calls.operation, []);
});

test('phones may cancel only their own model install, while desktop management may stop an install', async () => {
  const { service, state, calls } = fixture();
  const preview = await service.downloadPlan(phoneA, input);
  const started = await service.start(phoneA, { planId: preview.planId });
  assert.equal(started.job.canCancel, true);
  assert.equal((await service.snapshot(phoneB)).jobs[0].canCancel, false);
  await assert.rejects(service.cancel(phoneB, started.job.id, {}), { status: 403 });
  state.jobs.push({ id: 'tune-job', kind: 'tune', state: 'running', model: 'Q2_0' });
  await assert.rejects(service.cancel('desktop', 'tune-job', {}), { status: 409 });
  assert.deepEqual(calls.cancel, []);
  await service.cancel(phoneA, started.job.id, {});
  assert.deepEqual(calls.cancel, [started.job.id]);
  state.jobs.push({ id: 'desktop-install', kind: 'install', state: 'running', model: 'Q2_0', family: 'qwen' });
  await service.cancel('desktop', 'desktop-install', {});
  assert.deepEqual(calls.cancel, [started.job.id, 'desktop-install']);
});

test('import plans bind the inspected file signature and cannot be reused for changed files or another action', async t => {
  const directory = mkdtempSync(join(tmpdir(), 'strata-managed-import-'));
  t.after(() => rmSync(directory, { force: true, recursive: true }));
  const path = join(directory, 'fixture.gguf'); writeFileSync(path, 'GGUF-fixture');
  const adopted = [];
  const inspect = file => { const stat = statSync(file); return { variant: { id: 'local-fixture', name: 'Local fixture', diskGb: stat.size / 1e9, local: { path: file, bytes: stat.size, content: readFileSync(file, 'utf8') } }, files: [file], warnings: [] }; };
  const { service, calls } = fixture({ inspect, adopt: preview => adopted.push(preview) });
  const changed = await service.importPlan('desktop', { path });
  writeFileSync(path, 'GGUF-changed-fixture');
  await assert.rejects(service.importModel('desktop', { planId: changed.planId }), { status: 409 });
  assert.deepEqual(adopted, []);
  const valid = await service.importPlan('desktop', { path });
  await assert.rejects(service.start('desktop', { planId: valid.planId }), { status: 409 });
  const imported = await service.importModel('desktop', { planId: valid.planId });
  assert.deepEqual(imported, { model: { id: 'local-fixture', name: 'Local fixture' } });
  assert.equal(adopted.length, 1); assert.deepEqual(calls.operation, []);
  await assert.rejects(service.importModel('desktop', { planId: valid.planId }), { status: 409 });
});

test('proxy capability detection handles old or ordinary API servers, missing keys and authentication safely', async () => {
  const connection = { id: 'saved', name: 'Studio', kind: 'custom', protocol: 'openai', baseUrl: 'http://127.0.0.1:41000/v1', apiKey: 'PRIVATE-KEY' };
  const providers = { withSecrets: () => [structuredClone(connection)] };
  let calls = 0, status = 404, responseBody = {};
  const proxy = createModelManagementProxy({ providers, fetcher: async () => { calls++; return Response.json(responseBody, { status }); } });
  const target = remoteTargetId(connection);
  assert.equal((await proxy.call(target, '', 'GET')).supported, false);
  status = 200; responseBody = { object: 'list', data: [] };
  assert.equal((await proxy.call(target, '', 'GET')).supported, false);
  status = 401;
  await assert.rejects(proxy.call(target, '', 'GET'), { status: 401 });
  connection.apiKey = ''; const previous = calls;
  assert.equal((await proxy.call(target, '', 'GET')).supported, false); assert.equal(calls, previous);
  assert.ok(!JSON.stringify(proxy.targets()).includes('PRIVATE-KEY'));
});

test('proxy rejects redirect following, changed credentials and arbitrary management paths', async () => {
  const connection = { id: 'saved', name: 'Studio', kind: 'custom', protocol: 'openai', baseUrl: 'http://127.0.0.1:41000/v1', apiKey: 'PRIVATE-KEY' };
  const providers = { withSecrets: () => [structuredClone(connection)] };
  const calls = [];
  const proxy = createModelManagementProxy({ providers, fetcher: async (url, options) => {
    calls.push({ url, ...options }); connection.apiKey = 'CHANGED-KEY'; return Response.json({ supported: true, capabilities: { downloads: true }, models: [], jobs: [] });
  } });
  const target = remoteTargetId(connection);
  await assert.rejects(proxy.call(target, '', 'GET'), { status: 409 });
  assert.equal(calls[0].redirect, 'error'); assert.equal(calls[0].headers.authorization, 'Bearer PRIVATE-KEY');
  assert.equal(connection.apiKey, 'CHANGED-KEY');
  for (const path of ['/../native/operations', '/downloads/job-1/../cancel', '?url=https://attacker.example']) await assert.rejects(proxy.call(target, path, 'POST', {}), { status: 403 });
  assert.equal(calls.length, 1);
});

test('real HTTP redirects never receive the saved management credential', async t => {
  let sinkRequests = 0;
  const sink = await listen(t, (req, res) => { sinkRequests++; res.end('{}'); });
  const redirect = await listen(t, (req, res) => { res.writeHead(302, { location: `${sink.url}/credential-sink` }); res.end(); });
  const connection = { id: 'redirecting', name: 'Redirecting server', kind: 'custom', protocol: 'openai', baseUrl: `${redirect.url}/v1`, apiKey: 'PRIVATE-KEY' };
  const proxy = createModelManagementProxy({ providers: { withSecrets: () => [structuredClone(connection)] } });
  await assert.rejects(proxy.call(remoteTargetId(connection), '', 'GET'), { status: 503 });
  assert.equal(sinkRequests, 0);
});

test('proxy receipt plans and cancellation are bound to the phone, destination and operation', async () => {
  const connection = { id: 'saved', name: 'Studio', kind: 'custom', protocol: 'openai', baseUrl: 'http://127.0.0.1:41000/v1', apiKey: 'PRIVATE-KEY' };
  const calls = [];
  const proxy = createModelManagementProxy({ providers: { withSecrets: () => [connection] }, fetcher: async (url, options) => {
    calls.push({ url, ...options });
    if (url.endsWith('/plan')) return Response.json({ planId: 'destination-plan', expiresAt: Date.now() + 300000, warnings: [] });
    if (url.endsWith('/downloads')) return Response.json({ job: { id: 'destination-job', state: 'running' } });
    if (url.endsWith('/cancel')) return Response.json({ job: { id: 'destination-job', state: 'cancelled' } });
    return Response.json({ supported: true, capabilities: { downloads: true }, models: [], jobs: [{ id: 'destination-job', state: 'running' }] });
  } });
  const target = remoteTargetId(connection);
  const plan = await proxy.call(target, '/plan', 'POST', input, phoneA);
  assert.notEqual(plan.planId, 'destination-plan');
  await assert.rejects(proxy.call(target, '/downloads', 'POST', { planId: plan.planId }, phoneB), { status: 409 });
  const started = await proxy.call(target, '/downloads', 'POST', { planId: plan.planId }, phoneA);
  assert.equal(started.job.canCancel, true);
  assert.deepEqual(JSON.parse(calls.at(-1).body), { planId: 'destination-plan' });
  await assert.rejects(proxy.call(target, '/downloads', 'POST', { planId: plan.planId }, phoneA), { status: 409 });
  assert.equal((await proxy.call(target, '', 'GET', undefined, phoneB)).jobs[0].canCancel, false);
  await assert.rejects(proxy.call(target, '/downloads/destination-job/cancel', 'POST', {}, phoneB), { status: 403 });
  connection.apiKey = 'REPLACED-KEY';
  await assert.rejects(proxy.call(target, '/downloads/destination-job/cancel', 'POST', {}, phoneA), { status: 403 });
});

test('two temporary HTTP hosts perform authenticated preview, start, progress and owner-scoped cancel without model files', async t => {
  const destination = fixture();
  let enabled = true;
  const key = 'temporary-management-key';
  const destinationRoutes = createModelManagementRoutes({ service: destination.service, proxy: { targets: () => ({ targets: [] }) }, port: 0, allowRemote: () => enabled, companionAllowed: () => false });
  const remote = await listen(t, async (req, res) => {
    if (req.headers.authorization !== `Bearer ${key}`) { res.writeHead(401, { 'content-type': 'application/json' }); res.end(JSON.stringify({ error: 'Invalid key' })); return; }
    return destinationRoutes(req, res, new URL(req.url, 'http://localhost').pathname, req.method, true);
  });
  const connection = { id: 'saved-studio', name: 'Studio Mac', kind: 'custom', protocol: 'openai', baseUrl: `${remote.url}/v1`, apiKey: key };
  const proxy = createModelManagementProxy({ providers: { withSecrets: () => [connection] } });
  const paired = new Set(['a'.repeat(32), 'b'.repeat(32)]);
  let controllerRoutes;
  const controller = await listen(t, (req, res) => controllerRoutes(req, res, new URL(req.url, 'http://localhost').pathname, req.method));
  controllerRoutes = createModelManagementRoutes({ service: fixture().service, proxy, port: controller.port, allowRemote: () => false, companionAllowed: device => paired.has(device) });
  const target = remoteTargetId(connection), base = `${controller.url}/api/model-management/targets/${target}`;
  const headers = { 'x-strata-companion-device': 'a'.repeat(32) };
  const targets = await json(`${controller.url}/api/model-management/targets`, { headers });
  assert.equal(targets.status, 200); assert.equal(targets.value.targets[1].id, target);
  const catalog = await json(base, { headers });
  assert.equal(catalog.status, 200); assert.equal(catalog.value.supported, true); assert.equal(catalog.value.targetName, 'Studio Mac');
  const plan = await json(`${base}/plan`, { method: 'POST', headers, body: input });
  assert.equal(plan.status, 200); assert.equal(destination.calls.operation.length, 0);
  const start = await json(`${base}/downloads`, { method: 'POST', headers, body: { planId: plan.value.planId } });
  assert.equal(start.status, 200); assert.equal(destination.calls.operation.length, 1); assert.equal(start.value.job.canCancel, true);
  const progress = await json(base, { headers }); assert.equal(progress.value.jobs[0].percent, 25);
  const otherPhone = { 'x-strata-companion-device': 'b'.repeat(32) };
  assert.equal((await json(`${base}/downloads/${start.value.job.id}/cancel`, { method: 'POST', headers: otherPhone, body: {} })).status, 403);
  assert.equal((await json(`${base}/imports/browse`, { method: 'POST', headers, body: {} })).status, 403);
  const stopped = await json(`${base}/downloads/${start.value.job.id}/cancel`, { method: 'POST', headers, body: {} });
  assert.equal(stopped.status, 200); assert.equal(stopped.value.job.state, 'cancelled');
  assert.deepEqual(destination.calls.cancel, [start.value.job.id]);
  enabled = false;
  assert.equal((await json(base, { headers })).value.supported, false);
  for (const response of [catalog, plan, start, progress, stopped]) assert.ok(!response.text.includes(key));
});

test('local management routes reject cross-origin, rebinding and revoked callers before any service operation', async t => {
  const f = fixture(); let routes, paired = true;
  const host = await listen(t, (req, res) => routes(req, res, new URL(req.url, 'http://localhost').pathname, req.method));
  routes = createModelManagementRoutes({ service: f.service, proxy: { targets: () => ({ targets: [] }) }, port: host.port, allowRemote: () => false, companionAllowed: () => paired });
  const url = `${host.url}/api/model-management/targets/local/plan`;
  const header = { 'x-strata-companion-device': 'a'.repeat(32) };
  assert.equal((await json(url, { method: 'POST', body: input, headers: { ...header, origin: 'https://attacker.example' } })).status, 403);
  paired = false;
  assert.equal((await json(url, { method: 'POST', body: input, headers: header })).status, 401);
  const rebound = await new Promise((resolve, reject) => {
    const req = http.request({ hostname: '127.0.0.1', port: host.port, path: '/api/model-management/targets', headers: { host: `attacker.example:${host.port}` } }, res => { res.resume(); res.on('end', () => resolve(res.statusCode)); });
    req.on('error', reject); req.end();
  });
  assert.equal(rebound, 403); assert.deepEqual(f.calls.plan, []); assert.deepEqual(f.calls.operation, []);
  paired = true;
  const body = JSON.stringify(input);
  let bodyReceived;
  const received = new Promise(resolve => { bodyReceived = resolve; });
  host.server.once('request', bodyReceived);
  let pendingRequest;
  const result = new Promise((resolve, reject) => {
    pendingRequest = http.request({ hostname: '127.0.0.1', port: host.port, path: '/api/model-management/targets/local/plan', method: 'POST', headers: { ...header, 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) } }, res => { res.resume(); res.on('end', () => resolve(res.statusCode)); });
    pendingRequest.on('error', reject); pendingRequest.write(body.slice(0, 4));
  });
  await received;
  paired = false;
  pendingRequest.end(body.slice(4));
  assert.equal(await result, 401);
  assert.deepEqual(f.calls.plan, []); assert.deepEqual(f.calls.operation, []);
});

test('remote management requires explicit opt-in and rejects browser-origin requests despite valid inference authentication', async t => {
  const f = fixture(); let enabled = false;
  const routes = createModelManagementRoutes({ service: f.service, proxy: {}, port: 0, allowRemote: () => enabled, companionAllowed: () => false });
  const host = await listen(t, (req, res) => routes(req, res, new URL(req.url, 'http://localhost').pathname, req.method, true));
  const url = `${host.url}/v1/strata/models`;
  const headers = { authorization: 'Bearer temporary-valid-api-key' };
  assert.equal((await json(url, { headers })).status, 403);
  enabled = true;
  assert.equal((await json(url, { headers: { ...headers, origin: 'https://attacker.example' } })).status, 403);
  assert.equal((await json(url, { headers })).status, 200);
  assert.deepEqual(f.calls.operation, []);
});

test('Anthropic Strata connections discover management at root and v1 bases and invalidate changed protocol plans', async () => {
  for (const baseUrl of ['http://127.0.0.1:41000','http://127.0.0.1:41000/v1']) {
    const connection = { id:'anthropic-server', name:'Other Strata', kind:'custom', protocol:'anthropic', baseUrl, apiKey:'PRIVATE-KEY' };
    const calls=[];
    const proxy=createModelManagementProxy({providers:{withSecrets:()=>[structuredClone(connection)]},fetcher:async(url,options)=>{
      calls.push({url,options});
      return url.endsWith('/plan') ? Response.json({planId:'remote-plan',expiresAt:Date.now()+300_000}) : Response.json({supported:true,capabilities:{downloads:true},models:[],jobs:[]});
    }});
    const target=remoteTargetId(connection);
    assert.equal(proxy.targets().targets.some(value=>value.id===target),true);
    assert.equal((await proxy.call(target,'','GET')).supported,true);
    assert.equal(calls[0].url,'http://127.0.0.1:41000/v1/strata/models');
    assert.equal(calls[0].options.headers.authorization,'Bearer PRIVATE-KEY');
    const plan=await proxy.call(target,'/plan','POST',input,phoneA);
    if (!baseUrl.endsWith('/v1')) {
      connection.protocol='openai-chat';
      await assert.rejects(proxy.call(target,'/downloads','POST',{planId:plan.planId},phoneA),{status:409});
    }
  }
});

test('disconnect during destination preflight prevents a download from starting after the caller times out', async t => {
  const f=fixture();
  const routes=createModelManagementRoutes({service:f.service,proxy:{},port:0,allowRemote:()=>true,companionAllowed:()=>false});
  let onClose;
  const closed=new Promise(resolve=>{onClose=resolve;});
  const host=await listen(t,(req,res)=>{
    if(req.url.endsWith('/downloads'))res.once('close',onClose);
    return routes(req,res,new URL(req.url,'http://localhost').pathname,req.method,true);
  });
  const headers={authorization:'Bearer test-key'};
  const plan=(await json(host.url+'/v1/strata/models/plan',{method:'POST',headers,body:input})).value;
  let release,entered;
  const waiting=new Promise(resolve=>{entered=resolve;});
  f.onRefresh(async force=>{if(force){entered();await new Promise(resolve=>{release=resolve;});}});
  const controller=new AbortController();
  const request=fetch(host.url+'/v1/strata/models/downloads',{method:'POST',headers:{...headers,'content-type':'application/json'},body:JSON.stringify({planId:plan.planId}),signal:controller.signal});
  await waiting; controller.abort(); await assert.rejects(request); await closed; release();
  await new Promise(resolve=>setTimeout(resolve,20));
  assert.deepEqual(f.calls.operation,[]);
});

test('polling recovers an accepted download receipt after its start response is lost without granting another phone control', async () => {
  const destination=fixture();
  const connection={id:'saved',name:'Other Strata',kind:'custom',protocol:'openai-chat',baseUrl:'http://destination.invalid/v1',apiKey:'test-key'};
  const owner='remote:test-key';
  const proxy=createModelManagementProxy({providers:{withSecrets:()=>[structuredClone(connection)]},fetcher:async(url,options)=>{
    const body=options.body?JSON.parse(options.body):undefined;
    if(url.endsWith('/plan'))return Response.json(await destination.service.downloadPlan(owner,body));
    if(url.endsWith('/downloads')){await destination.service.start(owner,body);throw new Error('Synthetic response loss after acceptance');}
    if(url.endsWith('/cancel'))return Response.json(await destination.service.cancel(owner,destination.state.jobs[0].id,body));
    return Response.json(await destination.service.snapshot(owner));
  }});
  const target=remoteTargetId(connection);
  const plan=await proxy.call(target,'/plan','POST',input,phoneA);
  await assert.rejects(proxy.call(target,'/downloads','POST',{planId:plan.planId},phoneA),{status:503});
  assert.equal(destination.calls.operation.length,1);
  const recovered=await proxy.call(target,'','GET',undefined,phoneA);
  assert.equal(recovered.jobs[0].canCancel,true);
  assert.equal((await proxy.call(target,'','GET',undefined,phoneB)).jobs[0].canCancel,false);
  const jobId=recovered.jobs[0].id;
  await assert.rejects(proxy.call(target,`/downloads/${jobId}/cancel`,'POST',{},phoneB),{status:403});
  await proxy.call(target,`/downloads/${jobId}/cancel`,'POST',{},phoneA);
  assert.equal(destination.state.jobs[0].state,'cancelled');
});
