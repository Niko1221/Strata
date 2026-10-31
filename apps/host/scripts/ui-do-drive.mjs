#!/usr/bin/env node
/**
 * Drives the built app in a headless Chrome against a real host with a clean temporary profile (no engine, no model), for the 0.2.9.1
 * interface changes (branch f-0291-ui). Each case seeds the chat history it needs into the page's local storage, so the real screens
 * draw real data without a model.
 *
 *   npx vite build && node scripts/ui-do-drive.mjs <shots-dir> [case ...]
 *
 * Never point it at a real install: the host it starts uses a fresh STRATA_HOME in the temp folder and a random port.
 * Light and dark, at a normal and a narrow window. Prints "ok" / "FAIL" lines and exits 1 on a failure.
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { startScriptedWorker } from '../tests/helpers/scripted-worker.mjs';

const HOST_DIR = resolve(import.meta.dirname, '..');
const SHOTS = resolve(process.argv[2] ?? 'ui-shots');
const ONLY = process.argv.slice(3);
mkdirSync(SHOTS, { recursive: true });
const { chromium } = await import(pathToFileURL(join(HOST_DIR, 'node_modules/playwright-core/index.mjs')).href);
const CHROME = process.env.CHROME ?? ['C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome'].find(existsSync);
const freePort = () => new Promise(r => { const s = createServer(); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => r(p)); }); });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let failed = false;
export const check = (ok, label, extra = '') => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}${extra ? '  ' + extra : ''}`); if (!ok) failed = true; };

const home = mkdtempSync(join(tmpdir(), 'strata-ui-do-'));
const ws = join(home, 'workspace'); mkdirSync(ws, { recursive: true });
const port = await freePort();
const env = { ...process.env, STRATA_HOME: home, STRATA_PORT: String(port), STRATA_WORKSPACE: ws, STRATA_REAL_ENGINE: '0', STRATA_COMPONENTS_DIR: join(home, 'components'), STRATA_SETTINGS_FILE: join(home, 'app-settings.json'), LOCALAPPDATA: join(home, 'Local'), APPDATA: join(home, 'Roaming'), XDG_DATA_HOME: join(home, 'xdg-data'), XDG_CONFIG_HOME: join(home, 'xdg-config'), XDG_STATE_HOME: join(home, 'xdg-state') };
delete env.STRATA_HARNESS_DIR;
const child = spawn(process.execPath, [join(HOST_DIR, 'server', 'index.mjs')], { cwd: HOST_DIR, env, stdio: 'ignore', windowsHide: true });
export const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/health`)).ok) break; } catch { /* starting */ } await sleep(200); }
// A saved API connection to a scripted model gives the app a model to talk to, so it stays on the chat screen (no model loaded would send it to Models).
const worker = await startScriptedWorker({ model: 'cloud-model' });
const saved = await (await fetch(`${base}/api/model-providers`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ kind: 'custom', name: 'Scripted API', baseUrl: worker.url, protocol: 'openai-chat', apiKey: 'sk-ui-do-test-0123456789', models: [{ id: 'cloud-model', name: 'Cloud model', contextMax: 64000 }] }) })).json();
const remoteModel = { connectionId: saved.connections?.[0]?.id, modelId: 'cloud-model' };
check(!!remoteModel.connectionId, 'a scripted API model is connected');
// "Go straight to the app": the first-run screens are not what these cases look at.
await fetch(`${base}/api/setup/step`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ step: 1, dismissed: true, flowVersion: 2 }) });
const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'] });

const iso = (n = 0) => new Date(Date.UTC(2026, 9, 31, 9, 0, n)).toISOString();
export const user = (id, content, n = 0) => ({ id, role: 'user', content, at: iso(n) });
export const reply = (id, content, extra = {}) => ({ id, role: 'assistant', content, at: iso(5), tokens: 120, tps: 40, durationMs: 3000, ...extra });
export const chat = (id, title, messages) => ({ id, title, defaultTitle: false, updatedAt: iso(9), createdAt: iso(0), messages, workspace: ws, tokensIn: 0, tokensOut: 0, preferences: { useTools: true, mode: 'manual', remoteModel } });

/** A page with the given chats already in local storage. `prefs` are merged into the saved preferences. */
export async function open(scheme, width, height, sessions, { hash = '#/chat', prefs = {}, keep = false } = {}) {
  const context = await browser.newContext({ viewport: { width, height }, colorScheme: scheme, permissions: ['clipboard-read', 'clipboard-write'] });
  const page = await context.newPage();
  pages.push(page);
  const problems = [];
  page.on('pageerror', e => problems.push(e.message));
  page.problems = problems;
  await page.addInitScript(({ sessions, prefs }) => {
    try {
      if (sessionStorage.getItem('ui-do-seeded')) return;
      sessionStorage.setItem('ui-do-seeded', '1');
      if (sessions) localStorage.setItem('strata.sessions.v1', JSON.stringify(sessions));
      if (Object.keys(prefs).length) localStorage.setItem('strata.prefs.v1', JSON.stringify(prefs));
    } catch { /* storage refused */ }
  }, { sessions, prefs });
  await page.goto(`${base}/${hash}`);
  await page.waitForTimeout(1800);
  const skip = page.getByText('Go straight to the app');
  if (await skip.count()) { await skip.click(); await page.waitForTimeout(600); }
  return page;
}
export const shot = (page, name) => page.screenshot({ path: join(SHOTS, `${name}.png`) });
export const sizes = { wide: [1280, 860], narrow: [390, 800] };

export const cases = {};
const pages = [];

/* ------------------------------------------------------------------------------------------------------------------- #4725 */
const longReply = Array.from({ length: 70 }, (_, i) => `Paragraph ${i + 1}. This is part of a long answer from the assistant that goes on for many screens before it asks to change a file in the project.`).join('\n\n');
const askTool = (id, state = 'proposed', file = 'notes.txt') => ({ id, name: 'write', args: { file_path: file }, preview: `write ${file}\n\na\nb\nc`, effect: 'Creates a file.', scope: 'write', verb: 'Create a file', target: file, state, ask: { id: 'create-file', params: [file], why: 'replaces-if-exists' }, live: false });
const plainTool = (id, state = 'proposed') => ({ id, name: 'bash', args: {}, preview: 'npm install left-pad', effect: 'Runs a command on your computer.', scope: 'shell', verb: 'Run a command', target: 'npm install left-pad', state });
cases.approval = async () => {
  for (const scheme of ['light', 'dark']) for (const [sizeName, [w, h]] of Object.entries(sizes)) {
    const tag = `${scheme}-${sizeName}`;
    const sessions = [chat('s1', 'Long reply', [user('u1', 'Please write notes.txt'), reply('a1', longReply, { tools: [askTool('t1')] })])];
    const page = await open(scheme, w, h, sessions);
    const dock = page.locator('.approvaldock');
    await dock.waitFor({ timeout: 15000 });
    const view = page.viewportSize();
    const inView = async sel => { const b = await page.locator(sel).first().boundingBox(); return !!b && b.y >= 0 && b.y + b.height <= view.height + 1 && b.x >= 0 && b.x + b.width <= view.width + 1; };
    check(await page.locator('.askcard').count() === 1, `${tag}: one Approve/Deny card on the page`);
    check(await page.getByRole('button', { name: 'Approve' }).count() === 1 && await page.getByRole('button', { name: 'Deny' }).count() === 1, `${tag}: one Approve and one Deny`);
    check(await inView('.approvaldock'), `${tag}: the card is inside the window at the end of the reply`);
    const scroller = page.locator('.main__body.chat');
    const atEnd = () => scroller.evaluate(el => el.scrollHeight - el.scrollTop - el.clientHeight);
    check(await atEnd() < 4, `${tag}: the reply is still scrolled to its end`, String(await atEnd()));
    check(await page.locator('.approval-wait').count() === 1, `${tag}: the reply says it waits for an answer`);
    await shot(page, `4725-${tag}-end`);
    await scroller.evaluate(el => { el.scrollTop = 0; });
    await page.waitForTimeout(300);
    check(await inView('.approvaldock'), `${tag}: the card stays in the window when the reader is at the top`);
    const composerTop = await page.locator('.composer .st-composer').boundingBox();
    const dockBox = await dock.boundingBox();
    check(dockBox.y + dockBox.height <= composerTop.y, `${tag}: the card sits above the message box`);
    check((await page.evaluate(() => document.documentElement.scrollWidth)) <= w, `${tag}: no sideways scroll`);
    await shot(page, `4725-${tag}-top`);
    await page.getByRole('button', { name: 'Approve' }).click();
    await page.waitForTimeout(500);
    check(await page.locator('.approvaldock').count() === 0, `${tag}: the card goes away after Approve`);
    check(await page.locator('.approval-wait').count() === 0, `${tag}: and so does the waiting line`);
    check(page.problems.length === 0, `${tag}: no page error`, page.problems.join('; '));
    await page.context().close();
  }
  // Two waiting at once, one in each card style; Deny works from the dock.
  const sessions = [chat('s1', 'Two waiting', [user('u1', 'Do two things'), reply('a1', longReply.slice(0, 1500), { tools: [askTool('t1'), plainTool('t2')] })])];
  const page = await open('light', 1280, 860, sessions);
  await page.locator('.approvaldock').waitFor({ timeout: 15000 });
  check(await page.locator('.approvaldock > section').count() === 2, 'two waiting: both cards are in the dock');
  await shot(page, '4725-two-waiting');
  await page.getByRole('button', { name: 'Deny' }).first().click();
  await page.waitForTimeout(400);
  check(await page.locator('.approvaldock > section').count() === 1, 'two waiting: Deny on the first leaves the second');
  await page.context().close();
};

/* ------------------------------------------------------------------------------------------------------------------ run */
let ran = 0;
for (const [name, fn] of Object.entries(cases)) {
  if (ONLY.length && !ONLY.includes(name)) continue;
  console.log(`--- ${name}`);
  ran += 1;
  try { await fn(); } catch (e) {
    check(false, `${name} threw`, String(e?.stack ?? e).split('\n').slice(0, 4).join(' | '));
    const last = pages.at(-1);
    if (last && !last.isClosed()) {
      await last.screenshot({ path: join(SHOTS, `debug-${name}.png`) }).catch(() => {});
      console.log('PAGE TEXT:', ((await last.evaluate(() => document.body.innerText).catch(() => '')) || '').slice(0, 600).replace(/\n+/g, ' | '));
      console.log('PAGE ERRORS:', last.problems?.join('; '));
    }
  }
}
check(ran > 0, 'at least one case ran');
await browser.close();
await worker.close();
if (process.platform === 'win32') spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true }); else child.kill('SIGTERM');
try { rmSync(home, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ }
process.exit(failed ? 1 : 0);
