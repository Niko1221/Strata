import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createRemoteModelSSH, sshManagementDestination, SSH_MODEL_COMMAND } from '../server/remote-model-ssh.mjs';
import { createModelManagementProxy, remoteTargetId } from '../server/model-management.mjs';

const owner = `phone:${'a'.repeat(32)}`;
const sshConnection = () => ({ id: 'saved-connection', kind: 'custom', name: 'Terminal computer', protocol: 'openai-chat', baseUrl: 'http://inference.invalid/v1', apiKey: 'INFERENCE-SECRET-NOT-FOR-SSH', modelManagement: { transport: 'ssh', host: 'my-ssh-alias' } });
const snapshot = { supported: true, capabilities: { downloads: true, imports: true }, models: [], jobs: [] };
const temporary = t => { const directory = mkdtempSync(join(tmpdir(), 'strata-ssh-test-')); t.after(() => rmSync(directory, { force: true, recursive: true })); return directory; };
const helper = t => { const path = join(temporary(t), 'helper.py'); writeFileSync(path, 'print("fixture")\n'); return path; };

function mockProcess(onRequest = () => ({ status: 200, body: snapshot })) {
  const calls = [], signals = [];
  const spawnProcess = (command, args, options) => {
    const child = new EventEmitter();
    Object.assign(child, { stdin: new PassThrough(), stdout: new PassThrough(), stderr: new PassThrough(), kill: signal => { signals.push(signal); queueMicrotask(() => child.emit('close', null)); return true; } });
    let input = '';
    child.stdin.on('data', chunk => { input += chunk; });
    child.stdin.on('finish', () => {
      const payload = JSON.parse(input); calls.push({ command, args, options, payload });
      Promise.resolve(onRequest(payload, child)).then(result => {
        if (result !== undefined) { child.stdout.end(JSON.stringify(result)); child.emit('close', 0); }
      });
    });
    return child;
  };
  return { spawnProcess, calls, signals };
}

test('SSH settings stay in argv/stdin and use verified noninteractive system authentication without credentials', async t => {
  const process = mockProcess();
  const connection = sshConnection();
  connection.modelManagement = { transport: 'ssh', host: '2001:db8::1', user: 'remote-user', port: 2222, sourcePath: '/models/Strata with spaces; $(never-run)' };
  const call = createRemoteModelSSH({ ...process, helperPath: helper(t) });
  assert.deepEqual(await call(connection, '/plan', 'POST', { modelId: 'native-model', readPictures: false }, owner), { status: 200, body: snapshot });
  const captured = process.calls[0];
  assert.equal(captured.command, '/usr/bin/ssh'); assert.equal(captured.options.shell, false);
  for (const option of ['BatchMode=yes', 'StrictHostKeyChecking=yes', 'ClearAllForwardings=yes', 'ForwardAgent=no', 'PermitLocalCommand=no']) assert.ok(captured.args.includes(option));
  assert.deepEqual(captured.args.slice(-7), ['-p', '2222', '-l', 'remote-user', '--', '2001:db8::1', SSH_MODEL_COMMAND]);
  assert.equal(captured.payload.request.sourcePath, connection.modelManagement.sourcePath);
  assert.equal(captured.payload.request.owner, owner); assert.equal(captured.payload.request.body.readPictures, false);
  assert.equal(createHash('sha256').update(Buffer.from(captured.payload.helper, 'base64')).digest('hex'), captured.payload.sha256);
  assert.ok(!JSON.stringify(captured.args).includes(connection.modelManagement.sourcePath));
  assert.ok(!JSON.stringify(captured).includes(connection.apiKey));
});

test('optional username and port honor SSH configuration; invalid options fail before spawning', async t => {
  const process = mockProcess(), call = createRemoteModelSSH({ ...process, helperPath: helper(t) });
  const connection = sshConnection();
  await call(connection, '', 'GET', undefined, owner);
  assert.ok(!process.calls[0].args.includes('-p')); assert.ok(!process.calls[0].args.includes('-l'));
  for (const config of [{ host: '-oProxyCommand=anything' }, { host: 'host name' }, { host: 'user@host' }, { host: 'host', user: '-root' }, { host: 'host', port: 0 }, { host: 'host', port: '22' }, { host: 'host', sourcePath: '/source\ncommand' }]) {
    await assert.rejects(call({ ...connection, modelManagement: { transport: 'ssh', ...config } }, '', 'GET', undefined, owner), { status: 400 });
  }
  await assert.rejects(call(connection, '/shell', 'POST', {}, owner), { status: 400 });
  await assert.rejects(call(connection, '/downloads', 'GET', {}, owner), { status: 400 });
  assert.equal(process.calls.length, 1);
});

test('SSH diagnostic errors are actionable and never disclose captured command, host, key or path text', async t => {
  for (const [diagnostic, code] of [['Host key verification failed.', 'ssh_host_key'], ['WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!', 'ssh_host_key'], ['Permission denied (publickey).', 'ssh_auth'], ['sh: python3: command not found', 'ssh_python'], ['Could not resolve hostname secret-host', 'ssh_host'], ['Connection refused /private/secret token=PRIVATE', 'ssh_connection']]) {
    const process = mockProcess((payload, child) => { child.stderr.end(diagnostic); child.emit('close', 255); });
    await assert.rejects(createRemoteModelSSH({ ...process, helperPath: helper(t) })(sshConnection(), '', 'GET', undefined, owner), error => error.code === code && !error.message.includes('secret-host') && !error.message.includes('/private/secret') && !error.message.includes('token=PRIVATE'));
  }
});

test('SSH output is bounded and malformed or extra protocol output is rejected', async t => {
  for (const [stream, limit, code] of [['stdout', 'maxOutputBytes', 'ssh_output_limit'], ['stderr', 'maxErrorBytes', 'ssh_error_limit']]) {
    const process = mockProcess((payload, child) => { child[stream].write('x'.repeat(100)); });
    await assert.rejects(createRemoteModelSSH({ ...process, [limit]: 20, helperPath: helper(t) })(sshConnection(), '', 'GET', undefined, owner), error => error.code === code);
    assert.deepEqual(process.signals, ['SIGTERM']);
  }
  for (const value of ['not-json', '{"status":200,"body":[]}','{"status":100,"body":{}}','{"status":900,"body":{}}', '{"status":200,"body":{}}\nextra']) {
    const process = mockProcess((payload, child) => { child.stdout.end(value); child.emit('close', 0); });
    await assert.rejects(createRemoteModelSSH({ ...process, helperPath: helper(t) })(sshConnection(), '', 'GET', undefined, owner), error => error.code === 'ssh_protocol');
  }
});

test('timeout and cancellation end only the spawned SSH process; an already-aborted call never spawns', async t => {
  const process = mockProcess(() => undefined), call = createRemoteModelSSH({ ...process, helperPath: helper(t), timeoutMs: 10 });
  await assert.rejects(call(sshConnection(), '', 'GET', undefined, owner), error => error.code === 'ssh_timeout');
  assert.deepEqual(process.signals, ['SIGTERM']);
  const second = mockProcess(() => undefined), controller = new AbortController();
  const pending = createRemoteModelSSH({ ...second, helperPath: helper(t) })(sshConnection(), '', 'GET', undefined, owner, { signal: controller.signal });
  controller.abort();
  await assert.rejects(pending, error => error.code === 'ssh_cancelled');
  assert.deepEqual(second.signals, ['SIGTERM']);
  const before = second.calls.length;
  await assert.rejects(createRemoteModelSSH({ ...second, helperPath: helper(t) })(sshConnection(), '', 'GET', undefined, owner, { signal: controller.signal }));
  assert.equal(second.calls.length, before);
});

test('SSH destination identity uses connection-specific transport settings, with no inference-key coupling', () => {
  const connection = sshConnection(), initial = sshManagementDestination(connection);
  assert.equal(sshManagementDestination({ ...connection, apiKey: 'new-inference-key', baseUrl: 'http://another-inference.invalid/v1' }), initial);
  for (const change of [{ host: 'other-host' }, { port: 2222 }, { user: 'other-user' }, { sourcePath: '~/other-Strata' }]) assert.notEqual(sshManagementDestination({ ...connection, modelManagement: { ...connection.modelManagement, ...change } }), initial);
  assert.notEqual(sshManagementDestination({ ...connection, id: 'other-saved-connection' }), initial);
});

test('management proxy selects configured SSH without an API key and preserves owner-bound plan/job receipts', async () => {
  const connection = { ...sshConnection(), apiKey: '' }, calls = [];
  let requestId;
  const proxy = createModelManagementProxy({ providers: { withSecrets: () => [structuredClone(connection)] }, fetcher: async () => { throw new Error('Inference API must remain unused'); }, sshCall: async (saved, action, method, body, actor, options) => {
    calls.push({ action, method, body, actor, options });
    if (action === '/plan') return { status: 200, body: { planId: 'remote-plan', expiresAt: Date.now() + 300000, warnings: [] } };
    if (action === '/downloads') { requestId = options.requestId; return { status: 200, body: { job: { id: 'remote-job', state: 'running', requestId } } }; }
    return { status: 200, body: { ...snapshot, jobs: [{ id: 'remote-job', state: 'running', requestId }] } };
  } });
  const target = remoteTargetId(connection), plan = await proxy.call(target, '/plan', 'POST', { modelId: 'native-fixture', readPictures: false }, owner);
  await assert.rejects(proxy.call(target, '/downloads', 'POST', { planId: plan.planId }, 'phone:other'), { status: 409 });
  const started = await proxy.call(target, '/downloads', 'POST', { planId: plan.planId }, owner);
  assert.equal(started.job.canCancel, true); assert.equal(requestId, 'remote-plan');
  assert.equal((await proxy.call(target, '', 'GET', undefined, 'phone:other')).jobs[0].canCancel, false);
  await assert.rejects(proxy.call(target, '/downloads/remote-job/cancel', 'POST', {}, 'phone:other'), { status: 403 });
  connection.modelManagement.host = 'different-computer';
  await assert.rejects(proxy.call(target, '/downloads/remote-job/cancel', 'POST', {}, owner), { status: 403 });
  assert.ok(calls.every(call => call.actor === owner || call.actor === 'phone:other'));
});

test('cold SSH proxies recover persisted phone ownership without trusting another phone or replacing it during desktop polling', async () => {
  const connection = sshConnection(), target = remoteTargetId(connection), records = new Map(), plans = new Map();
  let nextPlan = 0, cancellations = 0;
  const row = (job, actor) => ({ id: job.id, state: job.state, requestId: job.requestId, canCancel: actor === 'desktop' || actor === job.owner });
  const sshCall = async (saved, action, method, body, actor, options) => {
    if (action === '/plan') {
      const planId = `persisted-plan-${++nextPlan}`; plans.set(planId, actor);
      return { status: 200, body: { planId, expiresAt: Date.now() + 300000, warnings: [] } };
    }
    if (action === '/downloads') {
      assert.equal(plans.get(body.planId), actor); plans.delete(body.planId);
      const job = { id: `persisted-job-${nextPlan}`, state: 'running', requestId: options.requestId, owner: actor }; records.set(job.id, job);
      return { status: 200, body: { job: row(job, actor) } };
    }
    const cancel = /^\/downloads\/([^/]+)\/cancel$/.exec(action);
    if (cancel) {
      const job = records.get(cancel[1]); assert.equal(job.owner, actor); job.state = 'cancelled'; cancellations++;
      return { status: 200, body: { job: row(job, actor) } };
    }
    assert.equal(action, ''); assert.equal(method, 'GET');
    return { status: 200, body: { ...snapshot, jobs: [...records.values()].map(job => row(job, actor)) } };
  };
  const createProxy = () => createModelManagementProxy({ providers: { withSecrets: () => [structuredClone(connection)] }, sshCall });
  let proxy = createProxy();
  const start = async () => {
    const plan = await proxy.call(target, '/plan', 'POST', { modelId: 'native-fixture', readPictures: false }, owner);
    return (await proxy.call(target, '/downloads', 'POST', { planId: plan.planId }, owner)).job.id;
  };
  const warmJob = await start();
  assert.equal((await proxy.call(target, '', 'GET')).jobs[0].canCancel, true);
  await proxy.call(target, `/downloads/${warmJob}/cancel`, 'POST', {}, owner);
  const coldJob = await start();
  proxy = createProxy(); // The remote helper survives the local host restart; receipt maps do not.
  const otherPhone = await proxy.call(target, '', 'GET', undefined, 'phone:other');
  assert.equal(otherPhone.jobs.find(job => job.id === coldJob).canCancel, false);
  await assert.rejects(proxy.call(target, `/downloads/${coldJob}/cancel`, 'POST', {}, 'phone:other'), { status: 403 });
  const recovered = await proxy.call(target, '', 'GET', undefined, owner);
  assert.equal(recovered.jobs.find(job => job.id === coldJob).canCancel, true);
  await proxy.call(target, '', 'GET');
  await proxy.call(target, `/downloads/${coldJob}/cancel`, 'POST', {}, owner);
  assert.equal(cancellations, 2);
});

test('cold HTTP proxy receipts stay fail-closed even when the destination advertises cancellation', async () => {
  const connection = sshConnection(); delete connection.modelManagement;
  let calls = 0;
  const proxy = createModelManagementProxy({ providers: { withSecrets: () => [structuredClone(connection)] }, fetcher: async () => {
    calls++; return Response.json({ ...snapshot, jobs: [{ id: 'persisted-http-job', requestId: 'old-plan', state: 'running', canCancel: true }] });
  } });
  const target = remoteTargetId(connection);
  assert.equal((await proxy.call(target, '', 'GET', undefined, owner)).jobs[0].canCancel, false);
  await assert.rejects(proxy.call(target, '/downloads/persisted-http-job/cancel', 'POST', {}, owner), { status: 403 });
  assert.equal(calls, 1);
});

test('SSH results from an edited connection are refused before being retained', async () => {
  const connection = sshConnection();
  let calls = 0;
  const proxy = createModelManagementProxy({ providers: { withSecrets: () => [structuredClone(connection)] }, sshCall: async () => {
    calls++; connection.modelManagement.sourcePath = '~/changed-source';
    return { status: 200, body: { ...snapshot, jobs: [{ id: 'stale-destination-job', state: 'running', canCancel: true }] } };
  } });
  const target = remoteTargetId(connection);
  await assert.rejects(proxy.call(target, '', 'GET', undefined, owner), { status: 409 });
  await assert.rejects(proxy.call(target, '/downloads/stale-destination-job/cancel', 'POST', {}, owner), { status: 403 });
  assert.equal(calls, 1);
});

test('unsupported inference API connections explain the Terminal SSH option without silently enabling it', async () => {
  const connection = sshConnection(); delete connection.modelManagement;
  let status = 404, sshCalls = 0;
  const proxy = createModelManagementProxy({ providers: { withSecrets: () => [structuredClone(connection)] }, fetcher: async () => Response.json({}, { status }), sshCall: async () => { sshCalls++; throw new Error('SSH must remain opt-in'); } });
  for (status of [403, 404, 405, 501, 200]) {
    const result = await proxy.call(remoteTargetId(connection), '', 'GET', undefined, owner);
    assert.equal(result.supported, false); assert.match(result.reason, /Terminal server over SSH/);
  }
  assert.equal(sshCalls, 0);
});

const python = [process.env.STRATA_TEST_PYTHON, '/Applications/Strata.app/Contents/Resources/control/python/bin/python3', '/usr/bin/python3'].find(path => path && existsSync(path) && spawnSync(path, ['-c', 'import sys;sys.exit(0 if sys.version_info >= (3,10) else 1)']).status === 0);
test('static bootstrap installs one private content-hashed helper and transfers literal requests without shell interpretation', { skip: !python && 'Set STRATA_TEST_PYTHON to Python 3.10 or newer.' }, async t => {
  const directory = temporary(t), bin = join(directory, 'bin'), home = join(directory, 'home'); mkdirSync(bin); mkdirSync(home);
  writeFileSync(join(bin, 'python3'), `#!/bin/sh\nexec '${python.replace(/'/g, "'\\''")}' "$@"\n`, { mode: 0o700 });
  const helperPath = join(directory, 'fixture_helper.py');
  writeFileSync(helperPath, 'import json,sys\nrequest=json.load(sys.stdin)\nprint(json.dumps({"status":200,"body":{"sourcePath":request.get("sourcePath"),"owner":request["owner"]}}))\n');
  const attack = join(directory, 'must-never-exist');
  const connection = sshConnection(); connection.modelManagement.sourcePath = `/Strata folder; $(touch ${attack})`;
  const spawnProcess = (command, args, options) => {
    assert.equal(command, '/usr/bin/ssh'); assert.equal(args.at(-1), SSH_MODEL_COMMAND);
    return spawn('/bin/sh', ['-c', SSH_MODEL_COMMAND], { ...options, env: { ...process.env, HOME: home, PATH: `${bin}:/usr/bin:/bin`, PYTHONDONTWRITEBYTECODE: '1' } });
  };
  const call = createRemoteModelSSH({ spawnProcess, helperPath });
  for (let i = 0; i < 2; i++) assert.deepEqual((await call(connection, '', 'GET', undefined, owner)).body, { sourcePath: connection.modelManagement.sourcePath, owner });
  assert.equal(existsSync(attack), false);
  const installed = join(home, '.local/share/strata/remote-model-helpers'), files = readdirSync(installed);
  assert.equal(files.length, 1); assert.match(files[0], /^manager-[a-f0-9]{64}\.py$/);
  assert.equal(statSync(installed).mode & 0o777, 0o700); assert.equal(statSync(join(installed, files[0])).mode & 0o777, 0o600);
  assert.equal(readFileSync(join(installed, files[0]), 'utf8'), readFileSync(helperPath, 'utf8'));
});

test('Node proxy and the real uploaded Python helper preview pinned catalog files and reference GGUFs without touching serving files', { skip: !python && 'Set STRATA_TEST_PYTHON to Python 3.10 or newer.' }, async t => {
  const directory = temporary(t), bin = join(directory, 'bin'), home = join(directory, 'home'), source = join(directory, 'Strata source with spaces');
  for (const path of [bin, home, source]) mkdirSync(path);
  writeFileSync(join(bin, 'python3'), `#!/bin/sh\nexec '${python.replace(/'/g, "'\\''")}' "$@"\n`, { mode: 0o700 });
  const pinned = Buffer.from('offline fixture, never downloaded');
  writeFileSync(join(source, 'setup.py'), `MODELS = {'Q2_0': {'about': 'Pinned fixture', 'families': ('qwen',), 'download_gb': 0.001, 'vision': False}}\nFAMILIES = {'qwen': {'title': 'Fixture', 'hf': 'https://huggingface.co/demo/repo/resolve/${'a'.repeat(40)}/', 'file': 'Fixture-{q}-{i}.gguf', 'shards': 1, 'tag': ''}}\nOWN_PATH_FAMILIES = {}\nREPO_PINS = {'demo/repo': {'Fixture-Q2_0-1.gguf': (${pinned.length}, '${createHash('sha256').update(pinned).digest('hex')}')}}\nHF_REVISIONS = {'demo/repo': '${'a'.repeat(40)}'}\n`);
  writeFileSync(join(source, 'strata.py'), '# Synthetic Terminal source marker.\n'); mkdirSync(join(source, 'strata_control'));
  writeFileSync(join(source, 'strata_control/acquisition.py'), '# Synthetic Terminal source marker.\n');
  const serving = join(source, 'strata-serving.json');
  writeFileSync(serving, JSON.stringify({ api_key: 'ACTIVE-INFERENCE-SECRET', port: 8080, exe: '/serving/engine', args: ['--native', '/serving/model.gguf'] }));
  const u32 = value => { const b = Buffer.alloc(4); b.writeUInt32LE(value); return b; };
  const u64 = value => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(value)); return b; };
  const string = value => Buffer.concat([u64(Buffer.byteLength(value)), Buffer.from(value)]);
  const header = Buffer.concat([Buffer.from('GGUF'), u32(3), u64(1), u64(1), string('general.architecture'), u32(8), string('llama'), string('token_embd.weight'), u32(1), u64(32), u32(0), u64(0)]);
  const model = join(directory, 'existing model.gguf'); writeFileSync(model, Buffer.concat([header, Buffer.alloc((32 - header.length % 32) % 32), Buffer.alloc(128)]));
  const original = [serving, model, join(source, 'setup.py')].map(path => ({ path, bytes: readFileSync(path), mtime: statSync(path).mtimeMs }));
  const sshCall = createRemoteModelSSH({ spawnProcess: (command, args, options) => {
    assert.equal(command, '/usr/bin/ssh'); assert.equal(args.at(-1), SSH_MODEL_COMMAND);
    return spawn('/bin/sh', ['-c', SSH_MODEL_COMMAND], { ...options, cwd: directory, env: { ...process.env, HOME: home, XDG_DATA_HOME: join(home, '.local/share'), PATH: `${bin}:/usr/bin:/bin`, PYTHONDONTWRITEBYTECODE: '1' } });
  } });
  const connection = sshConnection(); connection.modelManagement.sourcePath = source;
  const proxy = createModelManagementProxy({ providers: { withSecrets: () => [structuredClone(connection)] }, sshCall, fetcher: async () => { throw new Error('No inference API request is permitted'); } });
  const target = remoteTargetId(connection);
  const catalog = await proxy.call(target, '', 'GET');
  assert.equal(catalog.supported, true); assert.equal(catalog.storageMode, 'files'); assert.equal(catalog.models[0].id, 'native-q2_0');
  const plan = await proxy.call(target, '/plan', 'POST', { modelId: 'native-q2_0', readPictures: false });
  assert.ok(plan.planId); assert.equal((await proxy.call(target, '', 'GET')).jobs.length, 0);
  const preview = await proxy.call(target, '/imports/inspect', 'POST', { path: model });
  const reference = await proxy.call(target, '/imports', 'POST', { planId: preview.planId });
  assert.ok(reference.model.id.startsWith('local-'));
  const final = await proxy.call(target, '', 'GET');
  assert.ok(final.models.some(row => row.id === reference.model.id && row.state === 'imported'));
  assert.equal(final.jobs.length, 0);
  for (const before of original) { assert.deepEqual(readFileSync(before.path), before.bytes); assert.equal(statSync(before.path).mtimeMs, before.mtime); }
  assert.ok(!JSON.stringify([catalog, plan, preview, reference, final]).includes('ACTIVE-INFERENCE-SECRET'));
});
