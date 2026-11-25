import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, writeFile, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import net from 'node:net';
import { fileURLToPath } from 'node:url';
import { fileResponseHeaders } from '../server/chat-files.mjs';

/**
 * The follow-up of #7130: /api/files/raw serves a chat folder's pictures from the app's own address, and an SVG opened as a page there would run
 * its script as the app. Such files get `Content-Security-Policy: sandbox` (no script, opaque origin) and `nosniff`; pictures and PDFs are unchanged.
 */

const root = fileURLToPath(new URL('..', import.meta.url));
const delay = ms => new Promise(r => setTimeout(r, ms));
async function freePort() { const s = net.createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; }

test('only the types a browser can run script from get the sandbox policy', () => {
  for (const type of ['image/svg+xml', 'text/html', 'application/xhtml+xml', 'text/xml', 'application/xml', 'IMAGE/SVG+XML']) {
    const h = fileResponseHeaders(type, 5);
    assert.match(h['content-security-policy'], /^sandbox\b/, type);
    assert.match(h['content-security-policy'], /default-src 'none'/, type);
    assert.equal(h['x-content-type-options'], 'nosniff', type);
  }
  for (const type of ['image/png', 'image/jpeg', 'image/gif', 'image/webp', 'image/bmp', 'application/pdf', 'text/plain', 'text/csv', 'application/octet-stream']) {
    const h = fileResponseHeaders(type, 5);
    assert.equal(h['content-security-policy'], undefined, `${type}: unchanged (a PDF viewer does not open under sandbox)`);
    assert.equal(h['x-content-type-options'], 'nosniff', type);
    assert.equal(h['content-type'], type);
    assert.equal(h['content-length'], 5);
  }
});

test('both routes that hand a person\'s file to the browser use it', async () => {
  const index = await readFile(new URL('../server/index.mjs', import.meta.url), 'utf8');
  const raw = index.slice(index.indexOf("path === '/api/files/raw'"), index.indexOf("path === '/api/files/raw'") + 500);
  assert.match(raw, /res\.writeHead\(200, fileResponseHeaders\(f\.mime, f\.bytes\.length\)\)/);
  const drop = index.slice(index.indexOf('const dropGet ='), index.indexOf('const dropGet =') + 700);
  assert.match(drop, /res\.writeHead\(200, fileResponseHeaders\(file\.type, file\.bytes\.length\)\)/);
});

test('a real host: an SVG from the chat folder comes with the sandbox policy, a PNG and a PDF do not', async () => {
  const home = await mkdtemp(join(tmpdir(), 'strata-host-svg-'));
  await mkdir(join(home, 'dist')); await mkdir(join(home, 'work'));
  await writeFile(join(home, 'dist', 'index.html'), '<!doctype html><html><head><title>Strata</title></head><body><div id="root"></div></body></html>');
  // A harmless SVG: the point is the header, not what the file does.
  await writeFile(join(home, 'work', 'plain.svg'), '<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"><rect width="4" height="4"/></svg>');
  await writeFile(join(home, 'work', 'dot.png'), Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==', 'base64'));
  await writeFile(join(home, 'work', 'doc.pdf'), '%PDF-1.4\n%%EOF\n');
  const port = await freePort();
  const env = {
    ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'), STRATA_PORT: String(port), STRATA_DIST: join(home, 'dist'), STRATA_AGENT: 'off', STRATA_WORKSPACE: join(home, 'work'),
    STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()), STRATA_REAL_ENGINE: '1',
    STRATA_SETTINGS_FILE: join(home, 'app-settings.json'), STRATA_COMPONENTS_DIR: join(home, 'components'),
    LOCALAPPDATA: join(home, 'Local'), APPDATA: join(home, 'Roaming'), XDG_DATA_HOME: join(home, 'xdg-data'),
    XDG_STATE_HOME: join(home, 'xdg-state'), XDG_CONFIG_HOME: join(home, 'xdg-config'), HOME: home, USERPROFILE: home,
  };
  delete env.STRATA_PACKAGED; delete env.STRATA_ALLOWED_HOSTS; delete env.STRATA_DEV_ORIGINS;
  const host = spawn(process.execPath, ['server/index.mjs'], { cwd: root, stdio: 'ignore', env });
  const exit = new Promise(r => host.once('exit', r));
  try {
    let up = false;
    for (let i = 0; i < 150 && !up; i++) { try { up = (await fetch(`http://127.0.0.1:${port}/api/health`)).ok; } catch { await delay(100); } }
    assert.ok(up, 'host started');
    const get = name => fetch(`http://127.0.0.1:${port}/api/files/raw?sessionId=probe&path=${encodeURIComponent(name)}`);
    const svg = await get('plain.svg');
    assert.equal(svg.status, 200);
    assert.equal(svg.headers.get('content-type'), 'image/svg+xml');
    assert.match(svg.headers.get('content-security-policy') ?? '', /^sandbox; default-src 'none'/);
    assert.equal(svg.headers.get('x-content-type-options'), 'nosniff');
    assert.match(await svg.text(), /<svg /, 'the picture itself is unchanged');
    const png = await get('dot.png');
    assert.equal(png.status, 200);
    assert.equal(png.headers.get('content-type'), 'image/png');
    assert.equal(png.headers.get('content-security-policy'), null);
    const pdf = await get('doc.pdf');
    assert.equal(pdf.status, 200);
    assert.equal(pdf.headers.get('content-security-policy'), null, 'a PDF viewer does not open under sandbox');
  } finally { host.kill('SIGKILL'); await exit; await rm(home, { recursive: true, force: true }).catch(() => {}); }
});
