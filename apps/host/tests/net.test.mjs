/**
 * The proxy decision, offline.
 *
 * Nothing here reaches the internet: the proxy and the origin in the last block listen on 127.0.0.1 only, and the target host name
 * (`example.test`) is never looked up - the proxy answers for it. What is checked: the order of the sources (environment, Windows
 * registry, saved choice), the 30 s registry cache, the loopback rule, and that a proxy that does not answer is an error instead of a
 * quiet direct connection.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  DETECT_TTL_MS, configureNetwork, detectProxy, isLoopbackHost, matchesBypassList, netFetch, normalizeNetworkPolicy,
  parseProxy, parseRegQuery, proxyFromEnv, readWindowsProxy, registryProxyServer, saveNetworkPolicy, toolProxyEnvironment,
} from '../server/net.mjs';

const directory = mkdtempSync(join(tmpdir(), 'strata-net-'));
const settingsFile = join(directory, 'net.json');
configureNetwork({ file: settingsFile });
test.after(() => rmSync(directory, { recursive: true, force: true }));

const env = values => ({ ...values });
const listen = server => new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server.address().port)));
const close = server => new Promise(resolve => server.close(resolve));
/** Each test starts from an empty 30 s registry cache, the way a fresh process does. */
const resetCache = () => configureNetwork({ file: settingsFile });

test('this computer’s own addresses are never proxied', () => {
  for (const host of ['127.0.0.1', '127.8.9.1', 'localhost', 'LOCALHOST', 'app.localhost', '::1', '[::1]']) {
    assert.equal(isLoopbackHost(host), true, host);
  }
  for (const host of ['example.com', '127.0.0.1.example.com', '10.0.0.1', '192.168.1.1', '']) {
    assert.equal(isLoopbackHost(host), false, host);
  }
});

test('a proxy address is read the way a browser reads it', () => {
  assert.deepEqual(parseProxy('127.0.0.1:7890'), { kind: 'http', host: '127.0.0.1', port: 7890, auth: '', url: 'http://127.0.0.1:7890' });
  assert.equal(parseProxy('https://proxy.example:8443').kind, 'https');
  const withLogin = parseProxy('http://user:secret@127.0.0.1:7890');
  assert.equal(withLogin.auth, `Basic ${Buffer.from('user:secret').toString('base64')}`);
  assert.equal(withLogin.url, 'http://127.0.0.1:7890');       // the sign-in is kept out of what the screen shows
  assert.equal(parseProxy('socks5://127.0.0.1:7891').kind, 'socks');
  assert.equal(parseProxy(''), null);
  assert.throws(() => parseProxy('http://'), /not a proxy address|has no proxy address/);
});

test('the environment is read in the order the person’s tools expect', () => {
  const https = new URL('https://api.example.com/');
  const plain = new URL('http://api.example.com/');
  assert.deepEqual(proxyFromEnv(https, env({ HTTPS_PROXY: 'http://a:1', HTTP_PROXY: 'http://b:2' })), { value: 'http://a:1', name: 'HTTPS_PROXY' });
  assert.deepEqual(proxyFromEnv(https, env({ http_proxy: 'http://b:2' })), { value: 'http://b:2', name: 'http_proxy' });
  assert.deepEqual(proxyFromEnv(https, env({ ALL_PROXY: 'http://c:3' })), { value: 'http://c:3', name: 'ALL_PROXY' });
  assert.deepEqual(proxyFromEnv(plain, env({ HTTP_PROXY: 'http://b:2', HTTPS_PROXY: 'http://a:1' })), { value: 'http://b:2', name: 'HTTP_PROXY' });
  assert.equal(proxyFromEnv(https, env({})), null);
});

test('NO_PROXY and ProxyOverride understand the forms people actually write', () => {
  assert.equal(matchesBypassList('*', 'example.com'), true);
  assert.equal(matchesBypassList('localhost, .internal.example.com', 'api.internal.example.com'), true);
  assert.equal(matchesBypassList('example.com', 'api.example.com'), true);
  assert.equal(matchesBypassList('example.com', 'notexample.com'), false);
  assert.equal(matchesBypassList('127.*', '127.1.2.3'), true);
  assert.equal(matchesBypassList('example.com:8443', 'example.com', '443'), false);
  assert.equal(matchesBypassList('<local>', 'printer'), true);
  assert.equal(matchesBypassList('<local>', 'printer.example.com'), false);
  assert.equal(matchesBypassList('', 'example.com'), false);
});

test('the registry’s ProxyServer is split per kind of address', () => {
  assert.equal(registryProxyServer('127.0.0.1:7890', 'https:'), '127.0.0.1:7890');
  assert.equal(registryProxyServer('http=127.0.0.1:7890;https=127.0.0.1:7891;socks=127.0.0.1:7892', 'https:'), '127.0.0.1:7891');
  assert.equal(registryProxyServer('http=127.0.0.1:7890;https=127.0.0.1:7891', 'http:'), '127.0.0.1:7890');
  assert.equal(registryProxyServer('socks=127.0.0.1:7892', 'https:'), '127.0.0.1:7892');
  assert.equal(registryProxyServer('', 'https:'), '');
});

test('the registry reading is parsed from what reg.exe prints', () => {
  const output = [
    'HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings',
    '    ProxyEnable    REG_DWORD    0x1',
    '    ProxyServer    REG_SZ    127.0.0.1:7890',
    '    ProxyOverride    REG_SZ    *.local;<local>',
    '',
  ].join('\r\n');
  assert.deepEqual(parseRegQuery(output), { ProxyEnable: '0x1', ProxyServer: '127.0.0.1:7890', ProxyOverride: '*.local;<local>' });
  const asked = [];
  const read = readWindowsProxy({ platform: 'win32', run: (command, args) => { asked.push(args[1]); return parseRegQuery(output); } });
  assert.equal(read.enabled, true);
  assert.equal(read.server, '127.0.0.1:7890');
  assert.equal(read.override, '*.local;<local>');
  assert.match(asked[0], /^HKCU\\/);
  assert.equal(readWindowsProxy({ platform: 'linux' }), null);
});

test('the saved choice is checked before it is stored', () => {
  assert.deepEqual(normalizeNetworkPolicy({ mode: 'system', url: 'http://ignored:1' }), { mode: 'system', url: '' });
  assert.deepEqual(normalizeNetworkPolicy({ mode: 'unknown' }), { mode: 'system', url: '' });
  assert.deepEqual(normalizeNetworkPolicy({ mode: 'custom', url: '127.0.0.1:7890' }), { mode: 'custom', url: '127.0.0.1:7890' });
  assert.throws(() => normalizeNetworkPolicy({ mode: 'custom', url: '' }), /Enter the proxy address/);
  assert.throws(() => normalizeNetworkPolicy({ mode: 'custom', url: 'socks5://127.0.0.1:7891' }), /HTTP proxy address/);
  saveNetworkPolicy({ mode: 'custom', url: 'http://127.0.0.1:7890' });
  assert.deepEqual(configureNetwork({ file: settingsFile }), { mode: 'custom', url: 'http://127.0.0.1:7890' });
});

test('the order of the sources: saved choice, environment, registry, direct', () => {
  resetCache();
  const target = new URL('https://api.example.com/');
  saveNetworkPolicy({ mode: 'direct' });
  assert.equal(detectProxy(target, { env: { HTTPS_PROXY: 'http://env:1' }, platform: 'win32', registry: () => null }).source, 'settings');
  saveNetworkPolicy({ mode: 'custom', url: 'http://saved:8080' });
  assert.deepEqual(detectProxy(target, { env: { HTTPS_PROXY: 'http://env:1' } }).proxy.url, 'http://saved:8080');

  saveNetworkPolicy({ mode: 'system' });
  let clock = 5_000_000;
  const at = () => { clock += DETECT_TTL_MS + 1; return clock; };      // each answer after the 30 s cache has run out
  const withEnv = detectProxy(target, { env: { HTTPS_PROXY: 'http://env:1' }, platform: 'win32', now: at(), registry: () => ({ enabled: true, server: '127.0.0.1:7890', pac: '', override: '' }) });
  assert.equal(withEnv.source, 'environment');
  assert.equal(withEnv.proxy.url, 'http://env:1');

  const bypassed = detectProxy(target, { env: { HTTPS_PROXY: 'http://env:1', NO_PROXY: 'api.example.com' }, platform: 'win32', now: at(), registry: () => null });
  assert.equal(bypassed.mode, 'direct');
  assert.equal(bypassed.source, 'no-proxy');

  const windows = detectProxy(target, { env: {}, platform: 'win32', now: at(), registry: () => ({ enabled: true, server: 'http=127.0.0.1:7890;https=127.0.0.1:7891', pac: '', override: '' }) });
  assert.equal(windows.source, 'windows');
  assert.equal(windows.proxy.url, 'http://127.0.0.1:7891');

  const pac = detectProxy(target, { env: {}, platform: 'win32', now: at(), registry: () => ({ enabled: false, server: '', pac: 'http://wpad/proxy.pac', override: '' }) });
  assert.equal(pac.source, 'pac');
  assert.equal(pac.mode, 'direct');                            // said out loud, not hidden: the note explains the PAC script

  const nothing = detectProxy(target, { env: {}, platform: 'win32', now: at(), registry: () => null });
  assert.equal(nothing.source, 'none');

  saveNetworkPolicy({ mode: 'system' });
});

test('the registry is read once per 30 seconds, not once per request', () => {
  resetCache();
  let reads = 0;
  const reader = () => { reads += 1; return { enabled: true, server: '127.0.0.1:7890', pac: '', override: '' }; };
  const target = new URL('https://api.example.com/');
  let now = 1_000_000;
  const first = detectProxy(target, { env: {}, platform: 'win32', now, registry: reader });
  const second = detectProxy(target, { env: {}, platform: 'win32', now: now + 1_000, registry: reader });
  assert.equal(first.proxy.url, second.proxy.url);
  assert.equal(reads, 1);
  now += DETECT_TTL_MS;
  detectProxy(target, { env: {}, platform: 'win32', now, registry: reader });
  assert.equal(reads, 2);
});

test('a request to this computer never touches the proxy, and a proxied request goes through it', async t => {
  const origin = http.createServer((request, response) => { response.writeHead(200, { 'content-type': 'text/plain' }); response.end('origin'); });
  const originPort = await listen(origin);
  t.after(() => close(origin));

  const seen = [];
  const proxy = http.createServer((request, response) => {
    seen.push({ url: request.url, auth: request.headers['proxy-authorization'] ?? null });
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ from: 'proxy', url: request.url }));
  });
  const proxyPort = await listen(proxy);
  t.after(() => close(proxy));

  saveNetworkPolicy({ mode: 'custom', url: `http://user:pw@127.0.0.1:${proxyPort}` });

  const local = await netFetch(`http://127.0.0.1:${originPort}/local`);
  assert.equal(await local.text(), 'origin');
  assert.deepEqual(seen, []);                                   // the proxy was not asked about a 127.0.0.1 address

  const through = await netFetch('http://example.test/hello?q=1', { headers: { accept: 'application/json' } });
  assert.deepEqual(await through.json(), { from: 'proxy', url: 'http://example.test/hello?q=1' });
  assert.equal(seen[0].auth, `Basic ${Buffer.from('user:pw').toString('base64')}`);
});

test('a proxy that does not answer is an error, not a quiet direct connection', async t => {
  const proxy = http.createServer((request, response) => { response.writeHead(200); response.end('proxy'); });
  const proxyPort = await listen(proxy);
  await close(proxy);                                           // nothing listens on that port now
  saveNetworkPolicy({ mode: 'custom', url: `http://127.0.0.1:${proxyPort}` });
  await assert.rejects(netFetch('http://example.test/'), error => error.code === 'PROXY_UNREACHABLE' && /proxy at http:\/\/127\.0\.0\.1/.test(error.message));
  saveNetworkPolicy({ mode: 'system' });
});

test('a proxy that wants a sign-in says 407 instead of trying directly', async t => {
  const seen = [];
  const proxy = http.createServer();
  proxy.on('connect', (request, socket) => {
    seen.push(request.url);
    socket.end('HTTP/1.1 407 Proxy Authentication Required\r\nContent-Length: 0\r\n\r\n');
  });
  const proxyPort = await listen(proxy);
  t.after(() => close(proxy));
  saveNetworkPolicy({ mode: 'custom', url: `http://127.0.0.1:${proxyPort}` });
  await assert.rejects(netFetch('https://example.test/'), error => error.code === 'PROXY_AUTH' && /407/.test(error.message));
  assert.deepEqual(seen, ['example.test:443']);
  saveNetworkPolicy({ mode: 'system' });
});

test('a shell command the assistant runs follows the same door (proxy variables)', () => {
  resetCache();
  saveNetworkPolicy({ mode: 'custom', url: '127.0.0.1:7890' });
  const custom = toolProxyEnvironment({}, { platform: 'linux' });
  assert.equal(custom.HTTPS_PROXY, 'http://127.0.0.1:7890');
  assert.equal(custom.HTTP_PROXY, 'http://127.0.0.1:7890');
  assert.match(custom.NO_PROXY, /127\.0\.0\.1/);

  // Following the system: a proxy only the registry knows is exported too, so curl does not go straight out.
  saveNetworkPolicy({ mode: 'system' });
  const system = toolProxyEnvironment({}, { platform: 'win32', registry: () => ({ enabled: true, server: 'http=127.0.0.1:7890;https=127.0.0.1:7891', pac: '', override: '<local>' }) });
  assert.equal(system.HTTPS_PROXY, 'http://127.0.0.1:7891');
  assert.equal(system.HTTP_PROXY, 'http://127.0.0.1:7890');

  // A Direct choice adds nothing: Strata does not take a variable away from the person's own shell.
  saveNetworkPolicy({ mode: 'direct' });
  assert.deepEqual(toolProxyEnvironment({ HTTPS_PROXY: 'http://10.0.0.1:3128' }, { platform: 'win32' }), {});
});