import test from 'node:test';
import assert from 'node:assert/strict';
import https from 'node:https';
import http from 'node:http';
import net from 'node:net';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createCompanionManager, companionBridgeURL } from '../server/companion.mjs';

const bridgeURL = companionBridgeURL();
const temporary = () => mkdtempSync(join(tmpdir(), 'strata-integrated-companion-'));

async function fixture(t, options = {}) {
  const directory = temporary();
  let bridge;
  const manager = createCompanionManager({
    hostPort: 43210, directory, port: 0, adminPort: 0,
    listen: '127.0.0.1', bridgeOptions: { advertiseHost: '127.0.0.1', qr: false },
    loadBridge: async () => {
      const module = await import(bridgeURL);
      return { ...module, createBridge: async options => (bridge = await module.createBridge(options)) };
    },
    ...options,
  });
  t.after(async () => { await manager.stop(); rmSync(directory, { recursive: true, force: true }); });
  return { manager, directory, bridge: () => bridge };
}

function phoneRequest(port, path, { method = 'GET', token, body } = {}) {
  return new Promise((resolve, reject) => {
    const req = https.request({
      hostname: '127.0.0.1', port, path, method, agent: false, rejectUnauthorized: false,
      headers: { ...(token ? { authorization: `Bearer ${token}` } : {}), ...(body ? { 'content-type': 'application/json' } : {}) },
    }, res => {
      let text = ''; res.setEncoding('utf8');
      res.on('data', chunk => { text += chunk; });
      res.on('end', () => resolve({ status: res.statusCode, body: JSON.parse(text) }));
    });
    req.on('error', reject); req.end(body ? JSON.stringify(body) : undefined);
  });
}

async function paired(manager, name = 'Personal iPhone') {
  const state = manager.state();
  const port = Number(new URL(state.invitation.address).port);
  const response = await phoneRequest(port, '/v1/pair', { method: 'POST', body: { code: state.invitation.code, deviceName: name } });
  assert.equal(response.status, 200);
  return { token: response.body.token, port };
}

test('desktop startup keeps pairing off, imports nothing and creates no identity folder', async t => {
  const base = temporary(); t.after(() => rmSync(base, { recursive: true, force: true }));
  const directory = join(base, 'private-identity');
  let imports = 0;
  const manager = createCompanionManager({ hostPort: 8791, directory, loadBridge: () => { imports++; throw new Error('must stay lazy'); } });
  assert.deepEqual(manager.state(), { enabled: false, invitation: null, devices: [], error: null });
  assert.equal(manager.qrImage(), null);
  assert.equal(manager.hasOptedIn(), false);
  assert.equal((await manager.restore()).enabled, false);
  await manager.close();
  await assert.rejects(manager.rotateInvitation(), { status: 409 });
  await assert.rejects(manager.revokeSession('a'.repeat(32)), { status: 409 });
  await manager.stop();
  assert.equal(imports, 0); assert.equal(existsSync(directory), false);
  for (const hostPort of [0, 65536, '8791', NaN]) assert.throws(() => createCompanionManager({ hostPort }), /local Strata host port/);
});

test('concurrent enable is single and a queued disable awaits startup and closes it', async t => {
  const directory = temporary(); t.after(() => rmSync(directory, { recursive: true, force: true }));
  let imports = 0, starts = 0, closes = 0, options;
  const manager = createCompanionManager({
    hostPort: 43211, authorization: 'Bearer local-host-only', directory,
    loadBridge: async () => {
      imports++;
      return {
        HostClient: class { constructor(value) { this.options = value; } },
        createBridge: async value => {
          starts++; options = value;
          return { state: () => ({ invitation: { available: true }, devices: [] }), close: async () => { closes++; }, qrImage: () => null };
        },
      };
    },
  });
  const first = manager.start(), second = manager.start(), stop = manager.stop();
  assert.equal((await first).enabled, true); assert.equal((await second).enabled, true);
  assert.equal((await stop).enabled, false);
  assert.equal(imports, 1); assert.equal(starts, 1); assert.equal(closes, 1);
  assert.deepEqual(options.host.options, { upstream: 'http://127.0.0.1:43211', authorization: 'Bearer local-host-only' });
  assert.equal(options.port, 8787); assert.equal(options.adminPort, 8788);
  assert.equal(options.directory, directory);
});

test('integrated controls pair, renew and revoke without exposing private credentials', async t => {
  const { manager, bridge } = await fixture(t);
  const enabled = await manager.start();
  assert.equal(enabled.enabled, true); assert.equal(enabled.invitation.available, true);
  assert.match(enabled.invitation.url, /^strata:\/\/pair\?/);
  assert.match(enabled.invitation.code, /^\d{6}$/);
  assert.match(enabled.invitation.fingerprint, /^[a-f0-9]{64}$/);
  assert.equal(manager.qrImage(), null);
  const { token, port } = await paired(manager);
  const current = manager.state();
  assert.equal(current.invitation.available, false); assert.equal(current.invitation.qrAvailable, false);
  assert.equal(current.devices.length, 1);
  assert.deepEqual(Object.keys(current.devices[0]).sort(), ['createdAt', 'deviceName', 'id']);
  for (const secret of [token, bridge().authority.sessions[0].hash, bridge().adminURL, 'server-key.pem']) assert.ok(!JSON.stringify(current).includes(secret));
  const next = await manager.rotateInvitation();
  assert.equal(next.invitation.available, true);
  assert.equal(next.invitation.fingerprint, current.invitation.fingerprint);
  await assert.rejects(manager.revokeSession('../sessions.json'), { status: 400 });
  const removed = await manager.revokeSession(next.devices[0].id);
  assert.equal(removed.devices.length, 0);
  assert.equal((await phoneRequest(port, '/v1/status', { token })).status, 401);
});

test('disabling closes both listeners while re-enable preserves the certificate and paired token', async t => {
  const { manager, bridge, directory } = await fixture(t);
  await manager.start();
  const { token, port } = await paired(manager);
  const adminPort = bridge().adminPort;
  const cert = readFileSync(join(directory, 'server-cert.pem'), 'utf8');
  const disabled = await manager.stop();
  assert.equal(disabled.enabled, false); assert.equal(disabled.invitation, null); assert.equal(disabled.devices.length, 1);
  await assert.rejects(phoneRequest(port, '/v1/status', { token }), error => error.code === 'ECONNREFUSED');
  await new Promise((resolve, reject) => {
    const req = http.get({ hostname: '127.0.0.1', port: adminPort }, () => reject(new Error('Pairing admin listener remained open')));
    req.on('error', error => { try { assert.equal(error.code, 'ECONNREFUSED'); resolve(); } catch (failure) { reject(failure); } });
  });
  await manager.start();
  assert.equal(readFileSync(join(directory, 'server-cert.pem'), 'utf8'), cert);
  assert.equal(manager.state().devices.length, 1);
  assert.ok(bridge().authority.authorize(`Bearer ${token}`));
  assert.equal((await phoneRequest(bridge().port, '/v1/not-a-route', { token })).status, 404);
});

test('disable closes held phone and local connections without waiting for request timeout', async t => {
  const { manager, bridge } = await fixture(t);
  await manager.start();
  const request = https.request({ hostname: '127.0.0.1', port: bridge().port, method: 'POST', path: '/v1/pair', rejectUnauthorized: false, headers: { 'content-type': 'application/json', 'content-length': '500' } });
  request.on('error', () => {}); t.after(() => request.destroy());
  await new Promise(resolve => request.on('socket', socket => socket.once('secureConnect', resolve)));
  request.write('{');
  const local = net.connect(bridge().adminPort, '127.0.0.1');
  local.on('error', () => {}); t.after(() => local.destroy());
  await new Promise(resolve => local.once('connect', resolve));
  local.write('GET / HTTP/1.1\r\n');
  let deadline;
  try {
    await Promise.race([manager.stop(), new Promise((_, reject) => { deadline = setTimeout(() => reject(new Error('Held request delayed disable')), 1000); })]);
  } finally { clearTimeout(deadline); }
  assert.equal(manager.state().enabled, false);
});

test('disable waits for an aborted chat to save its partial response before returning', async t => {
  let upstreamClosed = false;
  const module = await import(bridgeURL);
  const { manager, directory } = await fixture(t, {
    loadBridge: async () => ({
      ...module,
      HostClient: class {
        async json(path) {
          if (path === '/api/models') return { variants: [] };
          if (path === '/api/native') return { catalog: { families: [] }, instances: [] };
          if (path === '/api/model-providers') return { connections: [{ id: 'saved', name: 'Saved API', models: [{ id: 'test-model', name: 'Test model' }] }] };
          throw new Error(`Unexpected mock path ${path}`);
        }
        async request(path, { signal }) {
          assert.equal(path, '/api/chat/stream');
          return new Response(new ReadableStream({ start(controller) {
            controller.enqueue(Buffer.from(JSON.stringify({ type: 'delta', text: 'Partial response' }) + '\n'));
            const end = () => setTimeout(() => { upstreamClosed = true; controller.close(); }, 30);
            if (signal.aborted) end();
            else signal.addEventListener('abort', end, { once: true });
          } }));
        }
      },
    }),
  });
  await manager.start();
  const { token, port } = await paired(manager);
  let request;
  await new Promise((resolve, reject) => {
    request = https.request({ hostname: '127.0.0.1', port, path: '/v1/chat', method: 'POST', rejectUnauthorized: false, headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' } }, response => {
      let body = '';
      response.on('data', chunk => { body += chunk; if (body.includes('Partial response')) resolve(); });
      response.on('error', () => {});
    });
    request.on('error', reject);
    request.end(JSON.stringify({ message: 'Keep my partial answer' }));
  });
  t.after(() => request.destroy());
  await manager.stop();
  assert.equal(upstreamClosed, true);
  const chats = JSON.parse(readFileSync(join(directory, 'chats.json'), 'utf8'));
  assert.deepEqual(chats[0].messages.map(message => message.content), ['Keep my partial answer', 'Partial response']);
});

test('bridge import failures are lazy, redacted and recoverable', async t => {
  const directory = temporary(); t.after(() => rmSync(directory, { recursive: true, force: true }));
  let calls = 0;
  const manager = createCompanionManager({ hostPort: 8791, directory, loadBridge: async () => {
    calls++;
    if (calls === 1) throw new Error('/private/person/identity/session-token-secret');
    throw Object.assign(new Error('/private/path'), { code: 'EADDRINUSE' });
  } });
  assert.equal(calls, 0);
  await assert.rejects(manager.start(), { status: 503, message: 'iPhone pairing could not start. Try again, or reinstall this Strata build.' });
  assert.equal(manager.state().enabled, false);
  assert.ok(!manager.state().error.includes('private'));
  await assert.rejects(manager.start(), { status: 503, message: 'Pairing is already open in another app. Quit Strata Companion, then try again.' });
  assert.equal(calls, 2);
  assert.equal((await manager.stop()).error, null);
});

test('desktop relaunch restores opted-in pairing and trusted phones; explicit disable prevents restoration', async t => {
  const directory = temporary();
  const managers = [];
  const newManager = () => {
    const manager = createCompanionManager({ hostPort: 43214, directory, port: 0, adminPort: 0, listen: '127.0.0.1', bridgeOptions: { qr: false, advertiseHost: '127.0.0.1' } });
    managers.push(manager); return manager;
  };
  t.after(async () => { for (const manager of managers) await manager.close(); rmSync(directory, { recursive: true, force: true }); });
  const first = newManager();
  await first.start();
  const fingerprint = first.state().invitation.fingerprint;
  const { token } = await paired(first);
  const deviceId = first.state().devices[0].id;
  await first.close();
  const preferenceFile = join(directory, 'desktop-pairing.json');
  assert.deepEqual(JSON.parse(readFileSync(preferenceFile, 'utf8')), { enabled: true, hasOptedIn: true });
  assert.equal(statSync(preferenceFile).mode & 0o777, 0o600);
  const second = newManager();
  assert.equal(second.state().enabled, false);
  const restored = await second.restore();
  assert.equal(restored.enabled, true);
  assert.equal(restored.invitation.fingerprint, fingerprint);
  assert.equal(restored.devices[0].id, deviceId);
  assert.equal((await phoneRequest(Number(new URL(restored.invitation.address).port), '/v1/not-a-route', { token })).status, 404);
  await second.stop();
  assert.deepEqual(JSON.parse(readFileSync(preferenceFile, 'utf8')), { enabled: false, hasOptedIn: true });
  const third = newManager();
  assert.equal((await third.restore()).enabled, false);
  assert.ok(readFileSync(join(directory, 'sessions.json'), 'utf8').includes(deviceId));
});

test('unreadable pairing preferences fail closed with a safe visible error', async t => {
  const directory = temporary(); t.after(() => rmSync(directory, { recursive: true, force: true }));
  writeFileSync(join(directory, 'desktop-pairing.json'), '{private-broken-data');
  let imports = 0;
  const manager = createCompanionManager({ hostPort: 8791, directory, loadBridge: () => { imports++; throw new Error('Unexpected import'); } });
  assert.equal((await manager.restore()).enabled, false);
  assert.match(manager.state().error, /could not be read/);
  assert.ok(!manager.state().error.includes('private-broken-data'));
  assert.equal(imports, 0);
  await manager.close();
  assert.equal(readFileSync(join(directory, 'desktop-pairing.json'), 'utf8'), '{private-broken-data');
});

test('packaged Resources layout resolves the bundled bridge and starts with the host port supplied', async t => {
  const base = temporary(); t.after(() => rmSync(base, { recursive: true, force: true }));
  const resources = join(base, 'Strata.app', 'Contents', 'Resources');
  const server = join(resources, 'host', 'server');
  const companion = join(resources, 'companion-bridge');
  mkdirSync(server, { recursive: true }); mkdirSync(companion, { recursive: true });
  const managerPath = join(server, 'companion.mjs');
  copyFileSync(fileURLToPath(new URL('../server/companion.mjs', import.meta.url)), managerPath);
  copyFileSync(fileURLToPath(bridgeURL), join(companion, 'bridge.mjs'));
  const packaged = await import(pathToFileURL(managerPath));
  assert.equal(realpathSync(fileURLToPath(packaged.companionBridgeURL())), realpathSync(join(companion, 'bridge.mjs')));
  const manager = packaged.createCompanionManager({ hostPort: 43212, directory: join(base, 'identity'), port: 0, adminPort: 0, listen: '127.0.0.1', bridgeOptions: { qr: false, advertiseHost: '127.0.0.1' } });
  t.after(() => manager.stop());
  const state = await manager.start();
  assert.equal(state.enabled, true); assert.equal(state.invitation.available, true);
  assert.equal(state.devices.length, 0);
});

test('the shared bridge keeps the exact loopback host instead of discovering another Strata instance', async t => {
  const { HostClient, createBridge } = await import(bridgeURL);
  const calls = [];
  const directory = temporary();
  const host = new HostClient({ upstream: 'http://127.0.0.1:43213', fetcher: async url => { calls.push(url); return Response.json({ variants: [] }); } });
  const bridge = await createBridge({ directory, host, port: 0, adminPort: 0, listen: '127.0.0.1', qr: false, advertiseHost: '127.0.0.1' });
  t.after(async () => { await bridge.close(); rmSync(directory, { recursive: true, force: true }); });
  await host.json('/api/models');
  assert.deepEqual(calls, ['http://127.0.0.1:43213/api/models']);
});


test('sharing consent survives disable and notifies the desktop on its first enable', async t => {
  let notifications = 0;
  const { manager, directory } = await fixture(t, { changed: () => { notifications += 1; } });
  assert.equal(manager.hasOptedIn(), false);
  await manager.start();
  assert.equal(manager.hasOptedIn(), true);
  assert.equal(notifications, 1);
  await manager.stop();
  assert.equal(manager.hasOptedIn(), true);
  const restored = createCompanionManager({ hostPort: 43210, directory });
  assert.equal(restored.state().enabled, false);
  assert.equal(restored.hasOptedIn(), true);
  await restored.close();
});
