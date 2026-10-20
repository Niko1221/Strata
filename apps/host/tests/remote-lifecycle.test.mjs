import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import http from 'node:http';
import { gzipSync } from 'node:zlib';
import { stageRemoteLifecycle } from '../../desktop/scripts/stage-remote-lifecycle.mjs';
import { createRemoteLifecycle, createRemoteLifecycleRoutes } from '../server/remote-lifecycle.mjs';
import { createRemoteLifecycleSSH, SSH_LIFECYCLE_COMMAND } from '../server/remote-lifecycle-ssh.mjs';
import { remoteTargetId, managementPath } from '../server/model-management.mjs';
import { HostClient } from '../../companion-bridge/bridge.mjs';

const originalKey = 'private-original-api-key', managedKey = 'private-managed-api-key';
const selected = () => ({ id: 'original-connection', kind: 'custom', name: 'Headless Strata', protocol: 'openai-chat', baseUrl: 'http://api-fixture.invalid:8080/v1', apiKey: originalKey, models: [], modelManagement: { transport: 'ssh', host: 'fixture-ssh', sourcePath: '/source with spaces' } });
const temporary = t => { const directory = mkdtempSync(join(tmpdir(), 'strata-lifecycle-node-')); t.after(() => rmSync(directory, { recursive: true, force: true })); return directory; };
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
function fixture(t, options = {}) {
  const directory = temporary(t), connections = [selected()], calls = [], saved = [], removed = []; let clock = 1000, hook, currentModel = 'Prepared model', currentReceipt = 'receipt-1';
  const providers = { withSecrets: () => structuredClone(connections), save: body => {
    saved.push(structuredClone(body)); const value = { ...structuredClone(body), id: body.id ?? `managed-${saved.length}`, baseUrl: body.baseUrl.replace(/\/$/, '') };
    const index = connections.findIndex(row => row.id === value.id); if (index < 0) connections.push(value); else connections[index] = value;
  }, remove: id => { removed.push(id); connections.splice(connections.findIndex(value => value.id === id), 1); } };
  const state = { supported: true, capabilities: { prepare: true, serve: true }, models: [{ id: 'downloaded-model', name: 'Downloaded model', state: 'downloaded', preparable: true, path: '/private/model.gguf' }], engines: [{ id: 'engine-fixture', name: 'Existing engine', configPath: '/private/config' }], profiles: [{ id: 'prepared-profile', name: 'Prepared model', modelId: 'downloaded-model', context: 32768, state: 'prepared' }], jobs: [], runtime: { state: 'stopped' }, apiKey: managedKey };
  const sshCall = async (connection, action, body, context) => {
    calls.push({ connection: structuredClone(connection), action, body: structuredClone(body), context });
    if (hook) { const response = await hook(connection, action, body, context); if (response) return response; }
    if (action === 'snapshot') return { status: 200, body: structuredClone(state) };
    if (action.endsWith('-plan')) return { status: 200, body: { planId: `remote-${action}-${calls.length}`, expiresAt: clock + 300000, modelName: 'Prepared model', engineName: 'Existing engine', context: 32768, diskNeededGb: 20, diskFreeGb: 100, diskShort: false, warnings: ['Existing APIs stay running.'], profileId: 'prepared-profile', host: '0.0.0.0', port: 18082, action: 'start' } };
    if (action === 'connect') return { status: body.receiptId === currentReceipt ? 200 : 409, body: body.receiptId === currentReceipt ? { endpoint: { host: '0.0.0.0', port: 18082, basePath: '/v1', modelId: currentModel, apiKey: managedKey, receiptId: currentReceipt } } : { error: 'This serving receipt is stale.' } };
    return { status: 200, body: { job: { id: 'job-fixture', kind: action === 'serve' ? 'serve' : 'prepare', state: 'queued', summary: 'Private job queued.', requestId: body?.planId ?? 'request-fixture', canCancel: true, result: { apiKey: managedKey, path: '/private/model' } } } };
  };
  const fetchCalls = [], fetcher = async (url, init) => { fetchCalls.push({ url, ...init }); return Response.json({ service: 'strata', model: currentModel, loaded: false, private: managedKey }); };
  const create = () => createRemoteLifecycle({ providers, directory, now: () => clock, sshCall, fetcher, ...options });
  return { directory, connections, providers, calls, saved, removed, state, fetchCalls, service: create(), create, target: remoteTargetId(connections[0]), hook: value => { hook = value; }, tick: () => { clock += 300001; }, switchModel: () => { currentModel = 'Second prepared model'; currentReceipt = 'receipt-2'; } };
}

test('lifecycle is opt-in per SSH connection and projects only bounded public display fields', async t => {
  const f = fixture(t), result = await f.service.snapshot(f.target);
  assert.equal(result.supported, true); assert.equal(result.models[0].preparable, true);
  assert.ok(!JSON.stringify(result).includes(managedKey)); assert.ok(!JSON.stringify(result).includes('/private')); assert.equal(f.calls[0].context.namespace.length, 64); assert.equal(f.calls[0].context.destination.length, 64);
  f.connections[0].modelManagement = undefined;
  assert.equal((await f.service.snapshot(f.target)).supported, false); assert.equal(f.calls.length, 1);
});

test('prepare/serve previews are typed, one-use, expiring and bound to kind and saved authority', async t => {
  const f = fixture(t), body = { modelId: 'downloaded-model', engineId: 'engine-fixture', context: 32768 };
  for (const invalid of [{ ...body, sourcePath: '/tmp' }, { ...body, context: '32768' }, { ...body, engineId: 'https://bad' }]) await assert.rejects(f.service.plan(f.target, 'prepare', invalid), { status: 400 });
  const preview = await f.service.plan(f.target, 'prepare', body); assert.ok(!preview.planId.startsWith('remote-')); assert.equal(f.calls.filter(value => value.action === 'prepare').length, 0);
  await assert.rejects(f.service.start(f.target, 'serve', { planId: preview.planId }), { status: 409 });
  const results = await Promise.allSettled([f.service.start(f.target, 'prepare', { planId: preview.planId }), f.service.start(f.target, 'prepare', { planId: preview.planId })]);
  assert.equal(results.filter(result => result.status === 'fulfilled').length, 1); assert.equal(f.calls.filter(value => value.action === 'prepare').length, 1);
  assert.ok(!JSON.stringify(results).includes(managedKey));
  const expired = await f.service.plan(f.target, 'serve', { profileId: 'prepared-profile', host: '0.0.0.0', port: 18082 }); f.tick();
  await assert.rejects(f.service.start(f.target, 'serve', { planId: expired.planId }), { status: 409 });
});

test('both SSH/source and inference API changes invalidate planned lifecycle actions', async t => {
  for (const field of ['ssh', 'source', 'api', 'key', 'protocol', 'removed', 'awaited']) {
    const f = fixture(t), plan = await f.service.plan(f.target, 'prepare', { modelId: 'downloaded-model' });
    if (field === 'ssh') f.connections[0].modelManagement.host = 'different';
    if (field === 'source') f.connections[0].modelManagement.sourcePath = '/different';
    if (field === 'api') f.connections[0].baseUrl = 'http://different.invalid/v1';
    if (field === 'key') f.connections[0].apiKey = 'different';
    if (field === 'protocol') f.connections[0].protocol = 'anthropic';
    if (field === 'removed') f.connections.length = 0;
    if (field === 'awaited') f.hook(() => { f.connections[0].apiKey = 'different'; });
    await assert.rejects(f.service.start(f.target, 'prepare', { planId: plan.planId }), cause => [404, 409].includes(cause.status));
    if (field !== 'awaited') assert.equal(f.calls.filter(value => value.action === 'prepare').length, 0);
  }
});

test('disconnect during preview suppresses its receipt and job cancellation is exact and helper-owned', async t => {
  const f = fixture(t), controller = new AbortController();
  f.hook(() => { controller.abort(); });
  await assert.rejects(f.service.plan(f.target, 'prepare', { modelId: 'downloaded-model' }, { signal: controller.signal }));
  f.hook(null);
  await assert.rejects(f.service.cancel(f.target, '../other', {}), { status: 400 });
  await assert.rejects(f.service.cancel(f.target, 'job-fixture', { arbitrary: true }), { status: 400 });
  const cancelled = await f.service.cancel(f.target, 'job-fixture', {}); assert.equal(cancelled.job.kind, 'prepare'); assert.equal(f.calls.at(-1).action, 'jobs/job-fixture/cancel');
});

test('Connect privately verifies its derived endpoint, preserves original APIs, and reuses one managed alias across switches and restarts', async t => {
  const f = fixture(t), original = structuredClone(f.connections[0]);
  const first = await f.service.connect(f.target, { receiptId: 'receipt-1' });
  assert.equal(f.connections.length, 2); assert.deepEqual(f.connections[0], original); assert.equal(f.fetchCalls[0].url, 'http://api-fixture.invalid:18082/v1/status'); assert.equal(f.fetchCalls[0].headers.authorization, `Bearer ${managedKey}`);
  assert.ok(!JSON.stringify(first).includes(managedKey)); assert.ok(!JSON.stringify(first).includes(originalKey)); assert.equal(statSync(join(f.directory, 'remote-lifecycle-links.json')).mode & 0o777, 0o600);
  await f.service.snapshot(f.target); const namespace = f.calls.at(-1).context.namespace;
  const restarted = f.create(); await restarted.snapshot(first.targetId); assert.equal(f.calls.at(-1).context.namespace, namespace);
  f.switchModel(); await assert.rejects(restarted.connect(f.target, { receiptId: 'receipt-1' }), { status: 409 });
  const second = await restarted.connect(f.target, { receiptId: 'receipt-2' });
  assert.equal(second.connectionId, first.connectionId); assert.equal(f.connections.length, 2); assert.equal(f.connections[1].models[0].id, 'Second prepared model'); assert.deepEqual(f.connections[0], original);
  f.connections[0].apiKey = 'edited-origin-key'; const count = f.calls.length;
  const unavailable = await restarted.snapshot(first.targetId); assert.equal(unavailable.supported, false); assert.equal(f.calls.length, count);
});

test('edited managed aliases and failed endpoint verification never overwrite connections or disclose receipts', async t => {
  const f = fixture(t), connected = await f.service.connect(f.target, { receiptId: 'receipt-1' });
  f.connections.find(value => value.id === connected.connectionId).name = 'User edited alias'; f.switchModel();
  await assert.rejects(f.service.connect(f.target, { receiptId: 'receipt-2' }), { status: 409 }); assert.equal(f.connections[1].name, 'User edited alias');
  const bad = fixture(t, { fetcher: async () => Response.json({ model: 'wrong', loaded: false, apiKey: managedKey }) });
  await assert.rejects(bad.service.connect(bad.target, { receiptId: 'receipt-1' }), cause => cause.status === 409 && !cause.message.includes(managedKey)); assert.equal(bad.connections.length, 1);
  const hostile = fixture(t); hostile.hook((connection, action) => action === 'connect' ? { status: 200, body: { endpoint: { host: 'other.invalid', port: 18082, basePath: '/v1', modelId: 'bad', apiKey: managedKey, receiptId: 'receipt-1' } } } : null);
  await assert.rejects(hostile.service.connect(hostile.target, { receiptId: 'receipt-1' }), { status: 502 }); assert.equal(hostile.fetchCalls.length, 0);
});

test('Connect rechecks concurrent alias edits and bounds actual status response streams/deadlines without saving keys', async t => {
  const f = fixture(t), first = await f.service.connect(f.target, { receiptId: 'receipt-1' });
  f.switchModel();
  f.hook((saved, action) => { if (action === 'connect') f.connections.find(value => value.id === first.connectionId).models = [{ id: 'user-edited-model', name: 'User model' }]; });
  await assert.rejects(f.service.connect(f.target, { receiptId: 'receipt-2' }), { status: 409 }); assert.equal(f.connections[1].models[0].id, 'user-edited-model');
  let mode = 'large';
  const server = http.createServer((req, res) => { res.writeHead(200, { 'content-type': 'application/json' }); res.write(mode === 'large' ? JSON.stringify({ private: managedKey.repeat(1000) }) : '{"service":'); if (mode === 'large') res.end(); });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve)); t.after(async () => { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); });
  for (mode of ['large', 'unfinished']) {
    const bounded = fixture(t, { fetcher: fetch, connectTimeoutMs: 30, maxConnectBytes: 200 }); bounded.connections[0].baseUrl = `http://127.0.0.1:${server.address().port}/v1`;
    bounded.hook((saved, action) => action === 'connect' ? { status: 200, body: { endpoint: { host: '0.0.0.0', port: server.address().port, basePath: '/v1', modelId: 'Prepared model', apiKey: managedKey, receiptId: 'receipt-1' } } } : null);
    await assert.rejects(bounded.service.connect(bounded.target, { receiptId: 'receipt-1' }), cause => cause.status === 409 && !cause.message.includes(managedKey)); assert.equal(bounded.connections.length, 1);
  }
});

test('private credentials in helper display/error fields are redacted and private endpoint/result objects are never projected', async t => {
  const f = fixture(t); f.state.models[0].name = originalKey; f.state.jobs = [{ id: 'job', kind: 'serve', state: 'done', summary: originalKey, apiKey: managedKey, result: { endpoint: { apiKey: managedKey } } }];
  const state = await f.service.snapshot(f.target); assert.ok(!JSON.stringify(state).includes(originalKey)); assert.ok(!JSON.stringify(state).includes(managedKey)); assert.equal(state.jobs[0].canCancel, false);
  f.hook(() => ({ status: 409, body: { error: originalKey } }));
  await assert.rejects(f.service.snapshot(f.target), cause => cause.status === 409 && !cause.message.includes(originalKey));
});

function bundle(t, extra = []) {
  const directory = temporary(t), path = join(directory, 'source.json.gz');
  const scripts = [['serve/server.py', '# fixture\n'], ['strata_control/runtime.py', '# fixture\n'], ['remote_model_manager.py', '# fixture\n'], ['remote_model_lifecycle.py', 'import json,sys\nr=json.load(sys.stdin)\nprint(json.dumps({"status":200,"body":{"owner":r["owner"],"sourcePath":r["sourcePath"],"catalogSourcePath":r.get("catalogSourcePath")}}))\n'], ...extra];
  writeFileSync(path, gzipSync(JSON.stringify({ schema: 1, files: scripts.map(([path, source]) => ({ path, sha256: hash(source), data: Buffer.from(source).toString('base64') })) })));
  return path;
}
function mockProcess(onRequest = () => ({ status: 200, body: { ok: true } })) {
  const calls = [], signals = [];
  return { calls, signals, spawnProcess(command, args, options) {
    const child = new EventEmitter(); Object.assign(child, { stdin: new PassThrough(), stdout: new PassThrough(), stderr: new PassThrough(), kill: signal => { signals.push(signal); queueMicrotask(() => child.emit('close', null)); } });
    let raw = ''; child.stdin.on('data', chunk => { raw += chunk; }); child.stdin.on('finish', () => {
      const payload = JSON.parse(raw); calls.push({ command, args, options, payload }); const result = onRequest(payload, child);
      if (result) { child.stdout.end(JSON.stringify(result)); child.emit('close', 0); }
    }); return child;
  } };
}
const scope = { namespace: 'a'.repeat(64), destination: 'b'.repeat(64) };
test('SSH lifecycle transfers only fixed command and bounded hashed source on stdin, never inference credentials', async t => {
  const process = mockProcess(), saved = selected(), call = createRemoteLifecycleSSH({ ...process, bundlePath: bundle(t) });
  await call(saved, 'prepare-plan', { modelId: 'downloaded-model' }, scope);
  const captured = process.calls[0]; assert.equal(captured.command, '/usr/bin/ssh'); assert.equal(captured.options.shell, false); assert.equal(captured.args.at(-1), SSH_LIFECYCLE_COMMAND);
  assert.ok(captured.args.includes('StrictHostKeyChecking=yes')); assert.equal(captured.payload.request.owner, 'desktop'); assert.equal(captured.payload.request.catalogSourcePath, saved.modelManagement.sourcePath);
  assert.equal(captured.payload.sha256, hash(readFileSync(bundle(t)))); assert.equal(captured.payload.bundle, undefined); assert.ok(!JSON.stringify(captured.payload).includes(originalKey)); assert.ok(!JSON.stringify(captured.args).includes(saved.modelManagement.sourcePath));
  await assert.rejects(call(saved, 'shell', {}, scope), { status: 400 });
  await assert.rejects(call(saved, 'snapshot', undefined, { ...scope, namespace: '../bad' }), { status: 400 });
});

test('SSH lifecycle output/diagnostics/deadlines and disconnect are bounded to the spawned client', async t => {
  const noisy = mockProcess((payload, child) => { child.stdout.write('x'.repeat(200)); });
  await assert.rejects(createRemoteLifecycleSSH({ ...noisy, bundlePath: bundle(t), maxOutputBytes: 100 })(selected(), 'snapshot', undefined, scope), { code: 'ssh_output_limit' }); assert.deepEqual(noisy.signals, ['SIGTERM']);
  const denied = mockProcess((payload, child) => { child.stderr.end('Permission denied PRIVATE HOST /private/path'); child.emit('close', 255); });
  await assert.rejects(createRemoteLifecycleSSH({ ...denied, bundlePath: bundle(t) })(selected(), 'snapshot', undefined, scope), cause => cause.status === 401 && !cause.message.includes('PRIVATE'));
  const slow = mockProcess(() => undefined);
  await assert.rejects(createRemoteLifecycleSSH({ ...slow, bundlePath: bundle(t), timeoutMs: 10 })(selected(), 'snapshot', undefined, scope), { code: 'ssh_timeout' }); assert.deepEqual(slow.signals, ['SIGTERM']);
});

test('source cache lookup works across sessions and only a typed pre-execution miss permits one bounded upload retry', async t => {
  const process = mockProcess((payload) => payload.bundle ? { status: 200, body: { ok: true } } : { status: 428, body: { code: 'lifecycle_bundle_missing' } });
  await createRemoteLifecycleSSH({ ...process, bundlePath: bundle(t) })(selected(), 'prepare', { planId: 'plan' }, scope);
  assert.equal(process.calls.length, 2); assert.equal(process.calls[0].payload.bundle, undefined); assert.ok(process.calls[1].payload.bundle);
  const repeated = mockProcess(() => ({ status: 428, body: { code: 'lifecycle_bundle_missing' } }));
  assert.equal((await createRemoteLifecycleSSH({ ...repeated, bundlePath: bundle(t) })(selected(), 'snapshot', undefined, scope)).status, 428); assert.equal(repeated.calls.length, 2);
  const tampered = mockProcess(() => ({ status: 503, body: { error: 'Cache integrity failure' } }));
  assert.equal((await createRemoteLifecycleSSH({ ...tampered, bundlePath: bundle(t) })(selected(), 'snapshot', undefined, scope)).status, 503); assert.equal(tampered.calls.length, 1);
});

const python = '/Applications/Strata.app/Contents/Resources/control/python/bin/python3';
test('actual owned source bundle and uploaded lifecycle helper perform an isolated read-only snapshot without external endpoints', { skip: !existsSync(python) }, async t => {
  const directory = temporary(t), bin = join(directory, 'bin'), home = join(directory, 'home'), bundlePath = join(directory, 'source.json.gz'); mkdirSync(bin); mkdirSync(home);
  writeFileSync(join(bin, 'python3'), `#!/bin/sh\nexec '${python}' "$@"\n`, { mode: 0o700 });
  const staged = stageRemoteLifecycle({ output: bundlePath }); assert.ok(staged.bytes < 8 * 1024 * 1024); assert.ok(staged.files > 100);
  const sshCall = createRemoteLifecycleSSH({ bundlePath, spawnProcess: (command, args, options) => {
    assert.equal(command, '/usr/bin/ssh');
    return spawn('/bin/sh', ['-c', args.at(-1)], { ...options, cwd: directory, env: { ...process.env, HOME: home, XDG_DATA_HOME: join(home, 'data'), XDG_CONFIG_HOME: join(home, 'config'), XDG_STATE_HOME: join(home, 'state'), PATH: `${bin}:/usr/bin:/bin`, PYTHONDONTWRITEBYTECODE: '1' } });
  } });
  const connection = selected(); delete connection.modelManagement.sourcePath;
  const service = createRemoteLifecycle({ directory: join(directory, 'host'), providers: { withSecrets: () => [connection] }, sshCall, fetcher: () => { throw Error('No API endpoint may be contacted by snapshot'); } });
  const state = await service.snapshot(remoteTargetId(connection));
  assert.equal(state.supported, true); assert.equal(state.runtime.state, 'stopped'); assert.deepEqual(state.jobs, []); assert.deepEqual(state.profiles, []); assert.deepEqual(state.engines, []);
  assert.equal(JSON.stringify(state).includes(originalKey), false);
});

test('real local bootstrap validates fixed paths/hashes, installs a private versioned source and preserves literal catalog paths', { skip: !existsSync(python) }, async t => {
  const directory = temporary(t), bin = join(directory, 'bin'), home = join(directory, 'home'); mkdirSync(bin); mkdirSync(home);
  writeFileSync(join(bin, 'python3'), `#!/bin/sh\nexec '${python}' "$@"\n`, { mode: 0o700 });
  const spawnProcess = (command, args, options) => { assert.equal(command, '/usr/bin/ssh'); return spawn('/bin/sh', ['-c', args.at(-1)], { ...options, env: { ...process.env, HOME: home, PATH: `${bin}:/usr/bin:/bin`, PYTHONDONTWRITEBYTECODE: '1' } }); };
  const saved = selected(); saved.modelManagement.sourcePath = '/literal; $(never execute)';
  const call = createRemoteLifecycleSSH({ spawnProcess, bundlePath: bundle(t) });
  const response = await call(saved, 'snapshot', undefined, scope); assert.equal(response.body.owner, 'desktop'); assert.equal(response.body.catalogSourcePath, saved.modelManagement.sourcePath);
  const cache = join(home, '.local/share/strata/remote-lifecycle-bundles'), installed = readdirSync(cache).filter(name => name.startsWith('source-')); assert.equal(installed.length, 1); assert.match(installed[0], /^source-[a-f0-9]{64}$/); assert.equal(statSync(response.body.sourcePath).mode & 0o777, 0o700);
  const restarted = createRemoteLifecycleSSH({ spawnProcess, bundlePath: bundle(t) });
  assert.equal((await restarted(saved, 'snapshot', undefined, scope)).status, 200);
  writeFileSync(join(response.body.sourcePath, 'serve/server.py'), '# tampered\n');
  assert.equal((await call(saved, 'snapshot', undefined, scope)).status, 503);
  const escaped = createRemoteLifecycleSSH({ spawnProcess, bundlePath: bundle(t, [['../must-not-exist', 'bad']]) });
  assert.equal((await escaped(saved, 'snapshot', undefined, scope)).status, 503); assert.equal(existsSync(join(cache, 'must-not-exist')), false);
});

test('desktop route guards reject phone/origin/rebinding/query/method before helper calls, and bridge/admin remain closed', async t => {
  const f = fixture(t); let route;
  const listener = http.createServer((req, res) => { res.setHeader('access-control-allow-origin', '*'); route(req, res, new URL(req.url, 'http://localhost')); });
  await new Promise(resolve => listener.listen(0, '127.0.0.1', resolve)); t.after(async () => { listener.closeAllConnections(); await new Promise(resolve => listener.close(resolve)); });
  const port = listener.address().port; route = createRemoteLifecycleRoutes({ service: f.service, port });
  const path = `/api/remote-lifecycle/targets/${f.target}`;
  const request = (headers = {}, suffix = '', method = 'GET') => new Promise((resolve, reject) => { const req = http.request({ hostname: '127.0.0.1', port, path: path + suffix, method, headers }, res => { let raw = ''; res.on('data', chunk => { raw += chunk; }); res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, raw })); }); req.on('error', reject); req.end(); });
  for (const headers of [{ origin: 'https://other.invalid' }, { origin: 'http://127.0.0.1:1' }, { host: 'evil.invalid' }, { 'x-strata-companion-device': 'a'.repeat(32) }]) { const result = await request(headers); assert.equal(result.status, 403); assert.equal(result.headers['access-control-allow-origin'], undefined); }
  assert.equal((await request({}, '?extra=1')).status, 404); assert.equal((await request({}, '', 'OPTIONS')).status, 404); assert.equal(f.calls.length, 0);
  assert.equal((await request()).status, 200);
  const client = new HostClient({ upstream: 'http://127.0.0.1:1', fetcher: () => { throw Error('Must not forward'); } });
  await assert.rejects(client.request(path, { deviceId: 'a'.repeat(32) }), { status: 403 }); assert.equal(managementPath('/v1/strata/models/lifecycle'), false);
  const index = readFileSync(new URL('../server/index.mjs', import.meta.url), 'utf8'); assert.ok(index.indexOf('return remoteLifecycleRoutes(req, res, url)') < index.indexOf("if (method === 'OPTIONS')"));
});
