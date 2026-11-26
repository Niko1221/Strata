// #7130: which web pages may use the host's /api (server/request-guard.mjs). A page of another site, of another localhost port or of a rebound name is
// refused; the host's own page (with the token it was served) works; scripts without Origin work. Run on a real host started on a scratch profile.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, rm, writeFile, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import net from 'node:net';
import { fileURLToPath } from 'node:url';
import { transform } from 'esbuild';
import { createRequestGuard } from '../server/request-guard.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const delay = ms => new Promise(r => setTimeout(r, ms));
async function freePort() { const s = net.createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; }

async function startHost() {
  const home = await mkdtemp(join(tmpdir(), 'strata-host-guard-'));
  await mkdir(join(home, 'dist'));
  await writeFile(join(home, 'dist', 'index.html'), '<!doctype html><html><head><title>Strata</title></head><body><div id="root"></div></body></html>');
  const port = await freePort();
  const env = {
    ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'), STRATA_PORT: String(port), STRATA_DIST: join(home, 'dist'), STRATA_AGENT: 'off',
    STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()), STRATA_REAL_ENGINE: '1',
    STRATA_SETTINGS_FILE: join(home, 'app-settings.json'), STRATA_COMPONENTS_DIR: join(home, 'components'),
    LOCALAPPDATA: join(home, 'Local'), APPDATA: join(home, 'Roaming'), XDG_DATA_HOME: join(home, 'xdg-data'),
    XDG_STATE_HOME: join(home, 'xdg-state'), XDG_CONFIG_HOME: join(home, 'xdg-config'), HOME: home, USERPROFILE: home,
  };
  delete env.STRATA_PACKAGED; delete env.STRATA_ALLOWED_HOSTS; delete env.STRATA_DEV_ORIGINS;
  const host = spawn(process.execPath, ['server/index.mjs'], { cwd: root, stdio: 'ignore', env });
  const exit = new Promise(r => host.once('exit', (code, signal) => r({ code, signal })));
  let up = false;
  for (let i = 0; i < 150 && !up; i++) { try { up = (await fetch(`http://127.0.0.1:${port}/api/health`)).ok; } catch { await delay(100); } }
  assert.ok(up, 'host started');
  return { port, done: async () => { host.kill('SIGKILL'); await exit; await rm(home, { recursive: true, force: true }).catch(() => {}); } };
}

/** One HTTP/1.1 request on its own socket, headers exactly as given (Host included). Resolves { status, headers, body }. */
function request(port, method, path, headers = {}, body = '') {
  return new Promise(resolve => {
    const socket = net.connect(port, '127.0.0.1');
    let data = '';
    const all = { Host: `127.0.0.1:${port}`, Connection: 'close', ...headers };
    const lines = [`${method} ${path} HTTP/1.1`, ...Object.entries(all).map(([k, v]) => `${k}: ${v}`), ...(body ? ['Content-Type: application/json', `Content-Length: ${Buffer.byteLength(body)}`] : [])];
    socket.on('connect', () => socket.write(`${lines.join('\r\n')}\r\n\r\n${body}`));
    socket.on('data', c => { data += c; });
    const finish = () => {
      const [head, ...rest] = data.split('\r\n\r\n');
      const headerLines = head.split('\r\n');
      resolve({ status: Number(/^HTTP\/1\.[01] (\d{3})/.exec(headerLines[0])?.[1] ?? 0), headers: Object.fromEntries(headerLines.slice(1).map(l => [l.slice(0, l.indexOf(':')).toLowerCase(), l.slice(l.indexOf(':') + 1).trim()])), body: rest.join('\r\n\r\n') });
    };
    socket.on('end', finish);
    socket.on('error', finish);
    setTimeout(() => { socket.destroy(); }, 8000);
  });
}

const MODE = JSON.stringify({ sessionId: 'probe', mode: 'manual' });   // a harmless mode; the guard does not look at the body

test('the host refuses other sites, other localhost ports and rebound names; its own page works', { timeout: 120000 }, async () => {
  const h = await startHost();
  try {
    const own = `http://127.0.0.1:${h.port}`;
    const post = headers => request(h.port, 'POST', '/api/chat/mode', headers, MODE);

    // the page: carries the token for the host's own page and nobody else
    const page = await request(h.port, 'GET', '/');
    assert.equal(page.status, 200);
    const token = /<meta name="strata-token" content="([0-9a-f]{64})">/.exec(page.body)?.[1];
    assert.ok(token, 'the served page carries the launch token');
    assert.equal(page.headers['x-frame-options'], 'SAMEORIGIN');
    assert.match(page.headers['content-security-policy'], /frame-ancestors 'self'/);
    assert.equal(page.headers['access-control-allow-origin'], undefined, 'the page answers no other origin');
    assert.equal((await request(h.port, 'GET', '/', { Origin: 'https://evil.example' })).body.includes('strata-token'), false, 'a page of another site is not given the token');
    assert.equal((await request(h.port, 'GET', '/', { 'Sec-Fetch-Site': 'cross-site' })).body.includes('strata-token'), false);
    assert.equal((await request(h.port, 'GET', '/', { Origin: 'http://localhost:9999' })).body.includes('strata-token'), false, 'another localhost port is not given the token');
    const again = await request(h.port, 'GET', '/');
    assert.equal(/content="([0-9a-f]{64})"/.exec(again.body)?.[1], token, 'one token for the whole launch');

    // another site, another localhost port, "null" and unreadable origins: 403 on a change and on a read
    for (const origin of ['https://evil.example', 'http://evil.example:' + h.port, 'http://localhost:9999', 'http://127.0.0.1:3000', 'http://[::1]:9999', 'null', '', 'not a url', 'file://']) {
      const r = await post({ Origin: origin, 'x-strata-token': token });
      assert.equal(r.status, 403, `POST Origin ${JSON.stringify(origin)} even with the token`);
      assert.equal(r.headers['access-control-allow-origin'], undefined);
      assert.equal((await request(h.port, 'GET', '/api/chats', { Origin: origin, 'x-strata-client': 'x' })).status, 403, `GET Origin ${JSON.stringify(origin)}`);
    }
    // the same pages as a "simple" request (text/plain body) are refused the same way
    assert.equal((await request(h.port, 'POST', '/api/chat/mode', { Origin: 'http://localhost:9999', 'Content-Type': 'text/plain' }, MODE)).status, 403);

    // the host's own origin, no token: a change is refused, a read works
    const noToken = await post({ Origin: own, 'Sec-Fetch-Site': 'same-origin' });
    assert.equal(noToken.status, 403);
    assert.equal(JSON.parse(noToken.body).code, 'bad-token');
    assert.equal((await post({ Origin: own, 'x-strata-token': 'f'.repeat(64) })).status, 403, 'a wrong token');
    assert.equal((await request(h.port, 'GET', '/api/health', { Origin: own })).status, 200);
    // with the token: the app's own UI works, under both loopback names and the Tauri window's origin
    for (const origin of [own, `http://localhost:${h.port}`, `http://[::1]:${h.port}`]) {
      const r = await post({ Origin: origin, 'Sec-Fetch-Site': 'same-origin', 'x-strata-token': token });
      assert.notEqual(r.status, 403, `own origin ${origin} with the token`);
    }
    // fetch metadata: same-site (another port of this PC) and cross-site are refused whatever else is sent
    for (const site of ['same-site', 'cross-site']) {
      assert.equal((await post({ Origin: own, 'Sec-Fetch-Site': site, 'x-strata-token': token })).status, 403, site);
      assert.equal((await request(h.port, 'GET', '/api/health', { 'Sec-Fetch-Site': site })).status, 403, `GET ${site}`);
    }
    // a script (curl, a CLI, the shell): no Origin and no Sec-Fetch-Site, so no token is asked for
    assert.notEqual((await post({})).status, 403);
    assert.equal((await request(h.port, 'GET', '/api/health')).status, 200);
    assert.equal((await request(h.port, 'POST', '/api/control/shutdown', {})).status, 403, 'the shell routes keep their own secret');

    // DNS rebinding: a name that is not this PC's, on every path
    for (const hostHeader of [`evil.example:${h.port}`, `rebind.test`, `127.0.0.1.evil.example:${h.port}`, `localhost.evil.example:${h.port}`]) {
      for (const path of ['/api/health', '/api/chats', '/', '/v1/models']) {
        const r = await request(h.port, 'GET', path, { Host: hostHeader, 'x-strata-client': 'x' });
        assert.equal(r.status, 403, `Host ${hostHeader} ${path}`);
        assert.equal(r.body.includes('strata-token'), false);
      }
      assert.equal((await request(h.port, 'POST', '/api/chat/mode', { Host: hostHeader, Origin: `http://${hostHeader}`, 'x-strata-token': token }, MODE)).status, 403);
    }
    for (const hostHeader of [`localhost:${h.port}`, `127.0.0.1:${h.port}`, `[::1]:${h.port}`]) assert.equal((await request(h.port, 'GET', '/api/health', { Host: hostHeader })).status, 200, hostHeader);

    // CORS: /api and the page answer no other origin; the preflight of a page that may not send the request is refused; /v1 stays open to local web UIs
    for (const origin of ['https://evil.example', 'http://localhost:9999']) {
      const pre = await request(h.port, 'OPTIONS', '/api/chat/mode', { Origin: origin, 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'content-type,x-strata-token' });
      assert.equal(pre.status, 403, `preflight ${origin}`);
      assert.equal(pre.headers['access-control-allow-origin'], undefined);
    }
    const ownPre = await request(h.port, 'OPTIONS', '/api/chat/mode', { Origin: own, 'Access-Control-Request-Method': 'POST' });
    assert.equal(ownPre.status, 204);
    assert.equal(ownPre.headers['access-control-allow-origin'], own);
    const v1Pre = await request(h.port, 'OPTIONS', '/v1/chat/completions', { Origin: 'http://localhost:3000', 'Access-Control-Request-Method': 'POST' });
    assert.equal(v1Pre.status, 204);
    assert.equal(v1Pre.headers['access-control-allow-origin'], '*');
  } finally { await h.done(); }
});

test('guard rules in isolation: packaged accepts only its own origin, a development run also accepts Vite, a host open to the network skips the Host rule', () => {
  const req = (headers, method = 'POST') => ({ method, headers });
  const packaged = createRequestGuard({ port: 47831, bind: '127.0.0.1', env: { STRATA_PACKAGED: '1' }, token: 't'.repeat(64) });
  const tok = { 'x-strata-token': 't'.repeat(64) };
  assert.equal(packaged.refusal(req({ host: '127.0.0.1:47831', origin: 'http://127.0.0.1:47831', ...tok }), '/api/chat/mode', 'POST'), null);
  assert.equal(packaged.refusal(req({ host: '127.0.0.1:47831', origin: 'http://localhost:5199', ...tok }), '/api/chat/mode', 'POST')?.code, 'bad-origin', 'packaged: no Vite origin');
  assert.equal(packaged.refusal(req({ host: '127.0.0.1:47831', origin: 'http://localhost:47832', ...tok }), '/api/chat/mode', 'POST')?.code, 'bad-origin');
  assert.equal(packaged.refusal(req({ host: '127.0.0.1:47831', 'sec-fetch-site': 'same-origin' }), '/api/chat/mode', 'POST')?.code, 'bad-token');
  assert.equal(packaged.refusal(req({ host: '127.0.0.1:47831', 'sec-fetch-site': 'none' }, 'GET'), '/', 'GET'), null);

  const dev = createRequestGuard({ port: 8791, bind: '127.0.0.1', env: {}, token: 't'.repeat(64) });
  assert.equal(dev.refusal(req({ host: '127.0.0.1:8791', origin: 'http://localhost:5199' }), '/api/chat/mode', 'POST'), null, 'Vite through its proxy (no token in the dev page)');
  assert.equal(dev.refusal(req({ host: '127.0.0.1:8791', origin: 'http://localhost:9999' }), '/api/chat/mode', 'POST')?.code, 'bad-origin');
  const devExtra = createRequestGuard({ port: 8791, bind: '127.0.0.1', env: { STRATA_DEV_ORIGINS: 'http://localhost:6000' } });
  assert.equal(devExtra.refusal(req({ host: '127.0.0.1:8791', origin: 'http://localhost:6000' }), '/api/x', 'POST'), null);

  const lan = createRequestGuard({ port: 8791, bind: '0.0.0.0', env: { STRATA_PACKAGED: '1' }, token: 't'.repeat(64) });
  assert.equal(lan.refusal(req({ host: '192.168.1.5:8791', origin: 'http://192.168.1.5:8791', ...tok }), '/api/chat/mode', 'POST'), null, 'same origin as the Host, on a host that is open to the network');
  assert.equal(lan.refusal(req({ host: '192.168.1.5:8791', origin: 'http://192.168.1.5:3000', ...tok }), '/api/chat/mode', 'POST')?.code, 'bad-origin');
  const named = createRequestGuard({ port: 8791, bind: '127.0.0.1', env: { STRATA_ALLOWED_HOSTS: 'strata.lan' } });
  assert.equal(named.refusal(req({ host: 'strata.lan:8791' }, 'GET'), '/api/health', 'GET'), null);
});

test('the page adds the token to its own /api changes and to nothing else', async () => {
  const { code } = await transform(await readFile(new URL('../src/lib/host-token.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
  const calls = [];
  globalThis.document = { querySelector: selector => selector === 'meta[name="strata-token"]' ? { getAttribute: () => 'tok123' } : null };
  globalThis.window = { location: { href: 'http://127.0.0.1:47831/#/chat' }, fetch: (input, init) => { calls.push({ input: String(input?.url ?? input), headers: init?.headers }); return Promise.resolve(new Response('{}')); } };
  const mod = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
  mod.installHostToken();
  mod.installHostToken();   // twice is once
  const sent = i => calls[i].headers?.get?.('x-strata-token') ?? null;
  await window.fetch('/api/chat/mode', { method: 'POST', body: '{}', headers: { 'content-type': 'application/json' } });
  assert.equal(sent(0), 'tok123');
  assert.equal(calls[0].headers.get('content-type'), 'application/json', 'its own headers stay');
  await window.fetch('/api/chats/x', { method: 'PUT', body: '{}' });
  assert.equal(sent(1), 'tok123');
  await window.fetch('/api/health');
  assert.equal(calls[2].headers, undefined, 'a read is left alone');
  await window.fetch('https://example.com/api/chat/mode', { method: 'POST' });
  assert.equal(calls[3].headers, undefined, 'another address never gets the secret');
  await window.fetch('http://127.0.0.1:9999/api/chat/mode', { method: 'POST' });
  assert.equal(calls[4].headers, undefined, 'another localhost port never gets it');
  await window.fetch(new Request('http://127.0.0.1:47831/api/chat/stop', { method: 'POST', headers: { 'x-a': 'b' } }));
  assert.equal(sent(5), 'tok123');
  assert.equal(calls.length, 6);
  delete globalThis.window; delete globalThis.document;
});

test('the served page markup keeps the token slot and the server routes ask for it', async () => {
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  assert.match(html, /<\/head>/i, 'the token is placed before </head>');
  const server = await readFile(new URL('../server/index.mjs', import.meta.url), 'utf8');
  assert.match(server, /guard\.refusal\(req, path, method\)/);
  assert.doesNotMatch(server, /access-control-allow-origin', '\*'/, 'no blanket CORS in the host');
});
