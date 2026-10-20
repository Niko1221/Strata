import test from 'node:test';
import assert from 'node:assert/strict';
import https from 'node:https';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createBridge, HostClient } from '../bridge.mjs';

const remoteId = `remote-${'a'.repeat(24)}`;
const root = '/v1/model-management/targets';
const local = `${root}/local`;
const job = { id: 'download-1', modelId: 'q2_0', name: 'Catalog model', kind: 'download', state: 'running', summary: 'Downloading model files…', canCancel: true, percent: 25, doneGb: 10, totalGb: 40 };
const fail = (status, message) => Object.assign(new Error(message), { status });

function request(bridge, path, { method = 'GET', token, body, headers = {} } = {}) {
  return new Promise((resolve, reject) => {
    const req = https.request({ hostname: '127.0.0.1', port: bridge.port, path, method, agent: false, rejectUnauthorized: false,
      headers: { ...(token ? { authorization: `Bearer ${token}` } : {}), ...(body !== undefined ? { 'content-type': 'application/json' } : {}), ...headers } }, res => {
      let text = ''; res.setEncoding('utf8'); res.on('data', chunk => { text += chunk; });
      res.on('end', () => { let json; try { json = JSON.parse(text); } catch {} resolve({ status: res.statusCode, json, text }); });
    });
    req.on('error', reject); req.end(body === undefined ? undefined : JSON.stringify(body));
  });
}

async function fixture(t, handler = async () => { throw fail(404, 'Not supported'); }) {
  const directory = mkdtempSync(join(tmpdir(), 'strata-management-bridge-'));
  const calls = [];
  const host = { json: async (path, options) => { calls.push({ path, ...options }); return handler(path, options); } };
  const bridge = await createBridge({ directory, host, listen: '127.0.0.1', port: 0, adminPort: 0, qr: false, advertiseHost: '127.0.0.1' });
  t.after(async () => { await bridge.close(); rmSync(directory, { force: true, recursive: true }); });
  const paired = await request(bridge, '/v1/pair', { method: 'POST', body: { code: bridge.authority.code, deviceName: 'Test iPhone' } });
  assert.equal(paired.status, 200);
  return { bridge, token: paired.json.token, calls };
}

test('management capabilities, plan preview, explicit start and exact cancellation use the paired device', async t => {
  let started = 0;
  const { bridge, token, calls } = await fixture(t, async (path, options) => {
    if (path === '/api/model-management/targets') return { targets: [{ id: 'local', name: 'My Mac', kind: 'local', connectionId: 'PRIVATE-CONNECTION' }, { id: remoteId, name: 'Studio', kind: 'remote', baseUrl: 'https://PRIVATE-HOST' }] };
    if (path === '/api/model-management/targets/local') return { supported: true, targetName: 'My Mac', capabilities: { downloads: true, imports: true }, models: [{ id: 'q2_0', name: 'Catalog model', sizeGb: 40, state: 'available', detail: 'Fits this Mac', readPictures: true }], jobs: [job], repo: '/private/models', apiKey: 'PRIVATE-KEY' };
    if (path.endsWith('/plan')) return { planId: 'plan-1', targetName: 'My Mac', modelName: 'Catalog model', downloadGb: 40, diskFreeGb: 100, diskNeededGb: 48, readPictures: false, warnings: ['Loaded native models will be unloaded.', 'Strata may download or update the engine.', 'The model will not load automatically.'], expiresAt: 1234567890, setup_command: './setup.sh --yes', model_folder: '/private/models' };
    if (path.endsWith('/downloads')) { started++; assert.deepEqual(options.body, { planId: 'plan-1' }); return { job }; }
    if (path.endsWith('/download-1/cancel')) { assert.deepEqual(options.body, {}); return { job: { ...job, state: 'cancelled', summary: 'Stopped. Completed downloads are kept.' } }; }
    throw new Error(`Unexpected path ${path}`);
  });
  const targets = await request(bridge, root, { token });
  assert.equal(targets.status, 200);
  assert.deepEqual(targets.json, { targets: [{ id: 'local', name: 'My Mac', kind: 'local' }, { id: remoteId, name: 'Studio', kind: 'remote' }] });
  const catalog = await request(bridge, local, { token });
  assert.equal(catalog.status, 200); assert.equal(catalog.json.capabilities.downloads, true); assert.equal(catalog.json.capabilities.imports, false);
  assert.equal(catalog.json.models[0].readPictures, true); assert.equal(catalog.json.jobs[0].percent, 25);
  assert.equal(started, 0);
  const plan = await request(bridge, `${local}/plan`, { token, method: 'POST', body: { modelId: 'q2_0', readPictures: false } });
  assert.equal(plan.status, 200); assert.equal(plan.json.planId, 'plan-1'); assert.equal(plan.json.diskNeededGb, 48); assert.equal(plan.json.warnings.length, 3);
  assert.equal(started, 0, 'Planning must not start an install');
  const startedJob = await request(bridge, `${local}/downloads`, { token, method: 'POST', body: { planId: 'plan-1' } });
  assert.equal(startedJob.status, 202); assert.deepEqual(startedJob.json, { job }); assert.equal(started, 1);
  const cancelled = await request(bridge, `${local}/downloads/download-1/cancel`, { token, method: 'POST', body: {} });
  assert.equal(cancelled.status, 200); assert.equal(cancelled.json.job.state, 'cancelled');
  for (const call of calls) {
    assert.equal(call.deviceId, bridge.authority.sessions[0].id);
    assert.ok(call.path.startsWith('/api/model-management/targets'));
  }
  for (const secret of ['PRIVATE-CONNECTION', 'PRIVATE-HOST', 'PRIVATE-KEY', '/private/models', 'setup_command', './setup.sh']) assert.ok(![targets, catalog, plan, startedJob, cancelled].some(response => response.text.includes(secret)), secret);
});

test('management rejects unpaired access, browser origins and arbitrary controls before calling the host', async t => {
  const { bridge, token, calls } = await fixture(t);
  for (const [path, method, body] of [[root, 'GET'], [local, 'GET'], [`${local}/plan`, 'POST', { modelId: 'q2_0', readPictures: false }], [`${local}/downloads`, 'POST', { planId: 'plan-1' }], [`${local}/downloads/download-1/cancel`, 'POST', {}]]) {
    assert.equal((await request(bridge, path, { method, body })).status, 401);
  }
  assert.equal((await request(bridge, root, { token, headers: { origin: 'https://attacker.example' } })).status, 403);
  const invalid = [
    [root, 'POST', {}], [local, 'POST', {}], [`${local}/plan`, 'GET'],
    [`${local}/plan`, 'POST', { modelId: 'q2_0', readPictures: 'false' }],
    [`${local}/plan`, 'POST', { modelId: 'q2_0' }],
    [`${local}/plan`, 'POST', { modelId: '../../private', readPictures: false }],
    [`${local}/plan`, 'POST', { modelId: 'q2_0', readPictures: false, data_dir: '/private' }],
    [`${local}/plan`, 'POST', { modelId: 'q2_0', readPictures: false, deviceId: 'a'.repeat(32) }],
    [`${local}/downloads`, 'POST', { planId: 'plan-1', confirm: true }],
    [`${local}/downloads`, 'POST', { modelId: 'q2_0' }],
    [`${local}/downloads`, 'POST', { planId: 'https://attacker.example' }],
    [`${local}/downloads/download-1/cancel`, 'POST', { id: 'another-job' }],
    [`${local}/imports/plan`, 'POST', { repository: 'org/model' }],
    [`${local}/operations`, 'POST', { kind: 'install', args: {} }],
    [`${root}/http%3A%2F%2Fattacker.example`, 'GET'],
    [`${local}/downloads/download-1/cancel`, 'DELETE'],
  ];
  for (const [path, method, body] of invalid) {
    const response = await request(bridge, path, { token, method, body });
    assert.ok([400, 403].includes(response.status), `${method} ${path}: ${response.status}`);
  }
  assert.equal((await request(bridge, `${root}?url=https://attacker.example`, { token })).status, 404);
  assert.deepEqual(calls, []);
});

test('old hosts are unsupported without affecting installed model control and errors never leak host diagnostics', async t => {
  const { bridge, token } = await fixture(t, async path => {
    if (path === '/api/model-management/targets') throw fail(404, 'No route /private/root');
    throw fail(409, 'busy /Users/person/secret?token=PRIVATE-TOKEN https://PRIVATE-HOST');
  });
  const old = await request(bridge, root, { token });
  assert.equal(old.status, 200); assert.deepEqual(old.json.targets, []); assert.match(old.json.reason, /Update Strata/);
  const failed = await request(bridge, `${local}/downloads`, { token, method: 'POST', body: { planId: 'plan-1' } });
  assert.equal(failed.status, 409); assert.match(failed.json.error, /Finish active work/);
  for (const secret of ['/Users', 'PRIVATE-TOKEN', 'PRIVATE-HOST']) assert.ok(!failed.text.includes(secret));
});

test('catalogs, plans and download failures export display fields without paths, URLs or credentials', async t => {
  const privateText = 'Failed /private/secret/model https://PRIVATE-HOST token=PRIVATE-TOKEN';
  const { bridge, token } = await fixture(t, async path => path.endsWith('/plan') ? {
    planId: 'plan-1', expiresAt: 123, targetName: 'Mac', modelName: 'Model', readPictures: false,
    downloadGb: -1, diskFreeGb: 20, diskNeededGb: 40, diskShort: privateText, warnings: [privateText], raw: privateText,
  } : {
    supported: true, targetName: 'Mac', capabilities: { downloads: true }, reason: privateText,
    models: [{ id: 'model-1', name: 'Model', sizeGb: 40, state: 'available', detail: privateText, readPictures: false, path: privateText }],
    jobs: [{ ...job, state: 'error', summary: privateText, error: privateText, canCancel: false, percent: 200, doneGb: -10, totalGb: 40, result: { raw: privateText } }],
  });
  const catalog = await request(bridge, local, { token });
  assert.equal(catalog.status, 200); assert.equal(catalog.json.jobs[0].percent, 100); assert.equal(catalog.json.jobs[0].doneGb, undefined);
  assert.equal(catalog.json.jobs[0].canCancel, false);
  assert.match(catalog.json.jobs[0].error, /could not finish/);
  const plan = await request(bridge, `${local}/plan`, { token, method: 'POST', body: { modelId: 'model-1', readPictures: false } });
  assert.equal(plan.status, 200); assert.equal(plan.json.downloadGb, undefined); assert.match(plan.json.diskShort, /not enough free disk space/);
  for (const response of [catalog, plan]) for (const secret of ['PRIVATE-HOST', 'PRIVATE-TOKEN', '/private/secret', '"raw"', '"result"', '"path"']) assert.ok(!response.text.includes(secret), secret);
});

test('revocation while reading management data or previewing a plan suppresses the result and later start', async t => {
  for (const action of ['catalog', 'plan', 'start', 'cancel']) {
    let entered, release;
    const gate = new Promise(resolve => { release = resolve; });
    const started = new Promise(resolve => { entered = resolve; });
    const { bridge, token, calls } = await fixture(t, async () => {
      entered(); await gate;
      if (action === 'plan') return { planId: 'plan-1', expiresAt: 123, readPictures: false, warnings: [] };
      if (action === 'start' || action === 'cancel') return { job };
      return { supported: true, capabilities: { downloads: true }, models: [], jobs: [] };
    });
    const path = action === 'catalog' ? local : action === 'plan' ? `${local}/plan` : action === 'start' ? `${local}/downloads` : `${local}/downloads/download-1/cancel`;
    const options = action === 'catalog' ? {} : { method: 'POST', body: action === 'plan' ? { modelId: 'model-1', readPictures: false } : action === 'start' ? { planId: 'plan-1' } : {} };
    const pending = request(bridge, path, { token, ...options });
    await started;
    bridge.authority.revoke(bridge.authority.sessions[0].id);
    release();
    assert.equal((await pending).status, 401, action);
    const before = calls.length;
    assert.equal((await request(bridge, `${local}/downloads`, { token, method: 'POST', body: { planId: 'plan-1' } })).status, 401, action);
    assert.equal(calls.length, before, action);
  }
});

test('revocation while the start body is arriving prevents any model-management host call', async t => {
  const { bridge, token, calls } = await fixture(t);
  let seen;
  const received = new Promise(resolve => { seen = resolve; });
  bridge.secureServer.once('request', seen);
  const body = JSON.stringify({ planId: 'plan-1' });
  let req;
  const response = new Promise((resolve, reject) => {
    req = https.request({ hostname: '127.0.0.1', port: bridge.port, path: `${local}/downloads`, method: 'POST', rejectUnauthorized: false,
      headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) } }, res => {
      res.resume(); res.on('end', () => resolve(res.statusCode));
    });
    req.on('error', reject);
    req.write(body.slice(0, 5));
  });
  await received;
  bridge.authority.revoke(bridge.authority.sessions[0].id);
  req.end(body.slice(5));
  assert.equal(await response, 401);
  assert.deepEqual(calls, []);
});

test('the host allowlist injects a device header only for exact model-management methods and payloads', async () => {
  const calls = [];
  const deviceId = 'b'.repeat(32);
  const host = new HostClient({ upstream: 'http://127.0.0.1:47831', fetcher: async (url, options) => { calls.push({ url, ...options }); return Response.json({}); } });
  const planPath = '/api/model-management/targets/local/plan';
  await host.json(planPath, { method: 'POST', body: { modelId: 'q2_0', readPictures: false }, deviceId });
  assert.equal(calls.length, 1); assert.equal(calls[0].url, `http://127.0.0.1:47831${planPath}`);
  assert.equal(calls[0].headers['x-strata-companion-device'], deviceId); assert.equal(calls[0].redirect, 'error');
  for (const [path, options] of [
    [planPath, { method: 'POST', body: { modelId: 'q2_0', readPictures: false } }],
    [planPath, { method: 'POST', body: { modelId: 'q2_0', readPictures: false }, deviceId: 'bad' }],
    [planPath, { method: 'GET', deviceId }],
    [planPath, { method: 'POST', body: { modelId: 'q2_0', readPictures: false, gguf_dir: '/private' }, deviceId }],
    ['/api/model-management/targets/local/imports/plan', { method: 'POST', body: {}, deviceId }],
    ['/api/model-management/targets/local/downloads/job-1/cancel', { method: 'POST', body: { id: 'other' }, deviceId }],
    ['/api/model-management/targets/local/plan/job-1/cancel', { method: 'POST', body: {}, deviceId }],
    ['/api/native/operations', { method: 'POST', body: { kind: 'install', args: { model: 'q2_0' } }, deviceId }],
  ]) await assert.rejects(host.json(path, options), error => [400, 403].includes(error.status));
  assert.equal(calls.length, 1);
  await host.json('/api/models', { deviceId });
  assert.equal(calls[1].headers['x-strata-companion-device'], undefined);
});
