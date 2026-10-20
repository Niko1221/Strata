import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { createCompanionRoutes, isLocalCompanionRequest } from '../server/companion-routes.mjs';

test('pairing controls require exact desktop origin, loopback Host and loopback peer', () => {
  const req = (host, origin, remoteAddress = '127.0.0.1') => ({ headers: { host, ...(origin === undefined ? {} : { origin }) }, socket: { remoteAddress } });
  assert.equal(isLocalCompanionRequest(req('127.0.0.1:47831', 'http://127.0.0.1:47831'), 47831), true);
  assert.equal(isLocalCompanionRequest(req('localhost:47831', undefined, '::1'), 47831), true);
  assert.equal(isLocalCompanionRequest(req('[::1]:47831', 'http://[::1]:47831', '::ffff:127.0.0.1'), 47831), true);
  for (const [host, origin, peer] of [
    ['127.0.0.1:47831', 'http://127.0.0.1:9999', '127.0.0.1'],
    ['127.0.0.1:47831', 'http://evil.invalid', '127.0.0.1'],
    ['127.0.0.1:47831', 'null', '127.0.0.1'],
    ['rebind.invalid:47831', undefined, '127.0.0.1'],
    ['127.0.0.1:12345', undefined, '127.0.0.1'],
    ['127.0.0.1:47831', undefined, '192.168.1.5'],
    ['user@127.0.0.1:47831', undefined, '127.0.0.1'],
  ]) assert.equal(isLocalCompanionRequest(req(host, origin, peer), 47831), false, `${host}/${origin}/${peer}`);
});

test('local pairing routes expose bounded actions and never inherit wildcard CORS', async t => {
  const calls = [];
  const state = { enabled: false, invitation: null, devices: [], error: null };
  const manager = {
    state: () => state,
    start: async () => { calls.push('start'); state.enabled = true; return state; },
    stop: async () => { calls.push('stop'); state.enabled = false; return state; },
    rotateInvitation: async () => { calls.push('rotate'); return state; },
    revokeSession: async id => { calls.push(`revoke:${id}`); return state; },
    qrImage: () => state.enabled ? Buffer.from('PNG-fixture') : null,
  };
  let server;
  const route = createCompanionRoutes({ manager, port: () => server.address().port });
  server = http.createServer((req, res) => { res.setHeader('access-control-allow-origin', '*'); void route(req, res, new URL(req.url, 'http://localhost')); });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => { server.closeAllConnections(); server.close(); });
  const base = `http://127.0.0.1:${server.address().port}`;
  const request = (path, method = 'GET', headers = {}) => fetch(base + path, { method, headers });
  const disabled = await request('/api/companion');
  assert.deepEqual(await disabled.json(), state);
  assert.equal(disabled.headers.get('access-control-allow-origin'), null);
  assert.equal((await request('/api/companion/qr')).status, 404);
  assert.equal((await request('/api/companion/start', 'POST', { origin: 'http://localhost:5199' })).status, 403);
  const reboundStatus = await new Promise((resolve, reject) => {
    const raw = http.request(base + '/api/companion/start', { method: 'POST', headers: { host: 'evil.invalid' } }, response => {
      response.resume(); resolve(response.statusCode);
    });
    raw.on('error', reject); raw.end();
  });
  assert.equal(reboundStatus, 403);
  assert.deepEqual(calls, []);
  assert.equal((await request('/api/companion/start', 'POST', { origin: base })).status, 200);
  assert.equal(await (await request('/api/companion/qr')).text(), 'PNG-fixture');
  assert.equal((await request('/api/companion/invitation', 'POST')).status, 200);
  assert.equal((await request('/api/companion/devices/device_1', 'DELETE')).status, 200);
  assert.equal((await request('/api/companion/devices/../../secret', 'DELETE')).status, 404);
  assert.equal((await request('/api/companion/start')).status, 404);
  assert.equal((await request('/api/companion/stop', 'POST')).status, 200);
  assert.deepEqual(calls, ['start', 'rotate', 'revoke:device_1', 'stop']);
});
