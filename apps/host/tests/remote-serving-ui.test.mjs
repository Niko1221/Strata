/** Real Serving UI over isolated host fixtures. No SSH, external API, model load or download is contacted. */
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { build } from 'esbuild';
import { chromium } from 'playwright-core';

const root = fileURLToPath(new URL('..', import.meta.url));
const executablePath = [process.env.STRATA_UI_BROWSER, '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].find(value => value && existsSync(value));
const skip = executablePath ? false : 'Set STRATA_UI_BROWSER to installed Chromium; this test never downloads a browser.';

test('Serving keeps local defaults and reviews current-model actions on the selected remote only', { skip, timeout: 45_000 }, async t => {
    const source = join(root, 'src');
    const entry = `import React from 'react'; import {createRoot} from 'react-dom/client'; import {ServingScreen} from ${JSON.stringify(join(source, 'screens/Serving.tsx'))};
        window.fixtureAborts=0; const originalFetch=window.fetch; window.fetch=(url,options)=>{options?.signal?.addEventListener('abort',()=>window.fixtureAborts++);return originalFetch(url,options)};
        const root=createRoot(document.getElementById('root')); window.unmountFixture=()=>root.unmount(); root.render(<ServingScreen/>);`;
    const { outputFiles } = await build({ stdin: { contents: entry, resolveDir: source, loader: 'tsx' }, bundle: true, write: false, outfile: 'harness.js', format: 'iife', platform: 'browser', jsx: 'automatic', define: { 'process.env.NODE_ENV': '"production"' }, plugins: [{ name: 'isolated-serving-state', setup(builder) {
        builder.onResolve({ filter: /(?:^|\/)store$/ }, args => args.importer.includes('/src/') ? { path: 'isolated-store', namespace: 'isolated-store' } : undefined);
        builder.onLoad({ filter: /.*/, namespace: 'isolated-store' }, () => ({ contents: `const state={serving:{baseUrl:'http://this-mac.invalid/v1',running:true,requests:[],stats:{running:0,waiting:0,completed:0},access:{apiKeySet:false,otherDevices:false,remoteModelManagement:false},sharing:{active:false},endpoints:{models:'/models',openai:'/chat/completions',anthropic:'/messages'}},runtime:null,hardware:null}; export const useStore=selector=>selector(state); const unexpected=async()=>{throw Error('Unexpected local serving mutation')}; export const cancelRequest=unexpected,setLocalSharing=unexpected,setServingAccess=unexpected,setCloudflare=unexpected; export const pushToast=()=>{},dismissActionError=()=>{},dismissToast=()=>{},holdToast=()=>{},releaseToast=()=>{};`, loader: 'js' }));
        builder.onResolve({ filter: /(?:^|\/)native$/ }, args => args.importer.includes('/src/') ? { path: 'isolated-native', namespace: 'isolated-native' } : undefined);
        builder.onLoad({ filter: /.*/, namespace: 'isolated-native' }, () => ({ contents: 'export const useNative=()=>null; export const nativeRouting=()=>null;', loader: 'js' }));
    } }] });
    const script = outputFiles.find(file => file.path.endsWith('.js')).text;
    const css = ['styles/kit/tokens.css', 'styles/kit/components.css', 'styles/app.css', 'styles/annotation-fixes.css'].map(path => readFileSync(join(source, path), 'utf8')).join('\n') + '\n' + (outputFiles.find(file => file.path.endsWith('.css'))?.text ?? '');
    const calls = [], exceptions = [], blocked = [], held = [];
    let loaded = false, busy = false, modelId = 'garage-current', mode = 'valid', planMode = 'valid', sequence = 0;
    const target = id => '/api/remote-serving/targets/' + id;
    const snapshot = (id = 'garage') => id === 'unsupported'
        ? { supported: false, targetName: 'Legacy server', loaded: null, busy: null, capabilities: { load: false, unload: false }, reason: 'This API does not advertise Strata current-model serving controls.' }
        : { supported: true, targetName: id === 'basement' ? 'Basement server' : id === 'slow' ? 'Slow server' : 'Garage server', modelId: id === 'garage' ? modelId : id + '-current', loaded: id === 'garage' ? loaded : false, busy: id === 'garage' ? busy : false, capabilities: { load: id === 'garage' ? busy === false && !loaded : true, unload: id === 'garage' ? busy === false && loaded : false }, ...(busy === true && id === 'garage' ? { reason: 'Wait for current requests to finish.' } : busy === null && id === 'garage' ? { reason: 'Admission counts are unavailable; controls stay read-only.' } : {}) };
    const response = (res, value, status = 200, type = 'application/json') => { if (res.destroyed) return; res.writeHead(status, { 'content-type': type, 'cache-control': 'no-store' }); res.end(type === 'application/json' ? JSON.stringify(value) : value); };
    const server = http.createServer(async (req, res) => {
        const path = new URL(req.url, 'http://fixture').pathname;
        if (path === '/') return response(res, '<!doctype html><meta charset="utf-8"><link rel="stylesheet" href="/harness.css"><div id="root" style="max-width:1100px;margin:25px auto"></div><script src="/harness.js"></script>', 200, 'text/html');
        if (path === '/harness.js') return response(res, script, 200, 'text/javascript');
        if (path === '/harness.css') return response(res, css, 200, 'text/css');
        if (path === '/icons/sprite.svg') return response(res, readFileSync(join(root, 'public/icons/sprite.svg')), 200, 'image/svg+xml');
        if (/^\/fonts\/GTWalsheimPro-(Regular|Medium|Bold|Black)\.woff2$/.test(path)) return response(res, readFileSync(join(root, 'public', path)), 200, 'font/woff2');
        if (path === '/favicon.ico') { res.writeHead(204); return res.end(); }
        let input = ''; for await (const chunk of req) input += chunk;
        const body = input ? JSON.parse(input) : undefined;
        calls.push({ path, method: req.method, body });
        if (path === '/api/model-management/targets') return response(res, { targets: [{ id: 'local', name: 'This Mac', kind: 'local' }, ...[['garage', 'Garage server'], ['basement', 'Basement server'], ['unsupported', 'Legacy server'], ['slow', 'Slow server']].map(([id, name]) => ({ id, name, kind: 'remote', connectionId: id }))] });
        const match = /^\/api\/remote-serving\/targets\/(garage|basement|unsupported|slow)(\/plan)?$/.exec(path);
        if (!match) return response(res, { error: 'Unexpected fixture endpoint' }, 404);
        const id = match[1];
        if (req.method === 'GET') {
            if (id === 'slow') { held.push(() => response(res, snapshot('slow'))); return; }
            if (id === 'garage' && mode === 'error') return response(res, { error: 'Fixture server is temporarily unreachable.' }, 503);
            return response(res, snapshot(id));
        }
        if (match[2]) {
            const plan = { planId: 'plan-' + ++sequence, targetName: snapshot(id).targetName, modelId: snapshot(id).modelId, action: body.action, warnings: body.action === 'load' ? ['Existing before-load hooks may run.'] : ['The next inference request may automatically load this model again.'], expiresAt: Date.now() + (planMode === 'expiring' ? 450 : 60_000) };
            if (planMode === 'held') { held.push(() => response(res, plan)); return; }
            return response(res, plan);
        }
        loaded = !loaded;
        return response(res, snapshot(id));
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    t.after(() => { server.closeAllConnections(); server.close(); });
    const origin = `http://127.0.0.1:${server.address().port}`;
    const browser = await chromium.launch({ executablePath, headless: true, args: ['--disable-background-networking', '--disable-component-update', '--disable-sync', '--no-first-run'] });
    t.after(() => browser.close());
    const page = await browser.newPage({ viewport: { width: 1250, height: 950 }, colorScheme: 'dark' });
    page.on('pageerror', error => exceptions.push(error.message));
    await page.route('**/*', route => {
        if (new URL(route.request().url()).origin !== origin) { blocked.push(route.request().url()); return route.abort(); }
        return route.continue();
    });
    const evidence = process.env.STRATA_SERVING_UI_EVIDENCE_DIR;
    async function screenshot(name) { if (evidence) { mkdirSync(evidence, { recursive: true }); await page.screenshot({ path: join(evidence, name + '.png') }); } }
    async function select(name) {
        await page.getByRole('combobox', { name: 'Serving destination', exact: true }).click();
        await page.getByRole('option', { name, exact: true }).click();
    }
    const reads = id => calls.filter(call => call.path === target(id) && call.method === 'GET').length;
    const commits = () => calls.filter(call => call.method === 'POST' && !call.path.endsWith('/plan'));
    await page.goto(origin);
    await page.getByRole('combobox', { name: 'Serving destination', exact: true }).waitFor();
    await page.getByText('http://this-mac.invalid/v1', { exact: true }).waitFor();
    await page.waitForFunction(() => document.querySelector('[role="combobox"]')?.textContent.includes('This Mac'));
    assert.equal(await page.getByRole('button', { name: 'Cloudflare tunnel', exact: true }).count(), 1);
    assert.equal(await page.getByRole('button', { name: 'Share', exact: true }).count(), 1);
    assert.equal(calls.filter(call => call.path.startsWith('/api/remote-serving/')).length, 0, 'Default local view must not inspect or change remote servers.');
    await select('Garage server');
    await page.getByText('garage-current', { exact: true }).waitFor();
    assert.equal(await page.getByText('http://this-mac.invalid/v1', { exact: true }).count(), 0);
    assert.equal(await page.getByRole('button', { name: 'Cloudflare tunnel', exact: true }).count(), 0);
    assert.equal(await page.getByRole('button', { name: 'Share', exact: true }).count(), 0);
    assert.equal(await page.getByRole('checkbox', { name: 'Allow remote model management', exact: true }).count(), 0);
    assert.equal(await page.getByRole('button', { name: 'Review load', exact: true }).isEnabled(), true);
    assert.equal(await page.getByRole('button', { name: 'Review unload', exact: true }).isDisabled(), true);
    await page.getByText('Unloading releases the current model from memory. The API listener remains available, and the next chat request may load the model again.', { exact: true }).waitFor();
    await screenshot('remote-serving-status');
    await page.getByRole('button', { name: 'Review load', exact: true }).click();
    const loadReview = page.getByRole('dialog', { name: 'Review load on Garage server', exact: true });
    await loadReview.waitFor();
    await loadReview.getByText('garage-current', { exact: true }).waitFor();
    assert.equal(commits().length, 0, 'Review must not load a model.');
    assert.deepEqual(calls.find(call => call.path === target('garage') + '/plan').body, { action: 'load' });
    assert.equal(await loadReview.getByRole('button', { name: 'Load on Garage server', exact: true }).isEnabled(), true, 'Epoch-millisecond plan expiry must stay actionable.');
    await screenshot('remote-serving-load-review');
    await loadReview.getByRole('button', { name: 'Load on Garage server', exact: true }).click();
    await page.getByText('Model loaded', { exact: true }).waitFor();
    assert.deepEqual(commits().map(call => ({ path: call.path, body: call.body })), [{ path: target('garage'), body: { planId: 'plan-1' } }]);
    planMode = 'expiring';
    await page.getByRole('button', { name: 'Review unload', exact: true }).click();
    const unloadReview = page.getByRole('dialog', { name: 'Review unload on Garage server', exact: true });
    await unloadReview.waitFor();
    await unloadReview.getByText('This review has expired. Go back and review the action again.', { exact: true }).waitFor();
    assert.equal(await unloadReview.getByRole('button', { name: 'Unload on Garage server', exact: true }).isDisabled(), true, 'A review must expire while it stays open.');
    await unloadReview.getByRole('button', { name: 'Back', exact: true }).click();
    assert.equal(commits().length, 1);
    planMode = 'valid'; busy = true;
    await page.getByRole('button', { name: 'Refresh', exact: true }).click();
    await page.getByText('Busy', { exact: true }).waitFor();
    assert.equal(await page.getByRole('button', { name: 'Review load', exact: true }).isDisabled(), true);
    assert.equal(await page.getByRole('button', { name: 'Review unload', exact: true }).isDisabled(), true);
    await select('Legacy server');
    await page.getByText('Serving controls are unavailable for this connection.', { exact: true }).waitFor();
    assert.equal(await page.getByRole('button', { name: 'Review load', exact: true }).isDisabled(), true);
    assert.equal(await page.getByRole('button', { name: 'Review unload', exact: true }).isDisabled(), true);
    busy = null; await select('Garage server');
    await page.getByText('Admission counts are unavailable; controls stay read-only.', { exact: true }).waitFor();
    assert.equal(await page.getByRole('button', { name: 'Review unload', exact: true }).isDisabled(), true);
    mode = 'error'; await page.getByRole('button', { name: 'Refresh', exact: true }).click();
    await page.getByText('Fixture server is temporarily unreachable.', { exact: true }).waitFor();
    assert.equal(await page.getByRole('button', { name: 'Review unload', exact: true }).isDisabled(), true, 'Failed status refresh must invalidate old actions.');
    mode = 'valid'; busy = false;
    await page.getByRole('button', { name: 'Refresh', exact: true }).click();
    await page.getByText('Idle', { exact: true }).waitFor();
    await page.getByRole('button', { name: 'Review unload', exact: true }).click();
    await unloadReview.waitFor(); modelId = 'garage-reconfigured';
    await unloadReview.getByText('The server state changed. Go back and review the action again.', { exact: true }).waitFor();
    assert.equal(await unloadReview.getByRole('button', { name: 'Unload on Garage server', exact: true }).isDisabled(), true, 'Polling must invalidate a review for a different model.');
    await unloadReview.getByRole('button', { name: 'Back', exact: true }).click();
    await page.getByRole('button', { name: 'Review unload', exact: true }).click();
    await unloadReview.waitFor();
    assert.equal(commits().length, 1, 'Reviewing unload must leave the model loaded.');
    await unloadReview.getByRole('button', { name: 'Unload on Garage server', exact: true }).click();
    await page.getByText('Model unloaded', { exact: true }).waitFor();
    assert.deepEqual(commits().at(-1).body, { planId: 'plan-4' });
    const aborts = await page.evaluate(() => window.fixtureAborts);
    await select('Slow server');
    await page.waitForFunction(() => document.body.textContent.includes('Checking serving status'));
    await select('Garage server');
    await page.getByText('garage-reconfigured', { exact: true }).waitFor();
    for (const finish of held.splice(0)) finish();
    await page.waitForTimeout(100);
    assert.equal(await page.getByText('slow-current', { exact: true }).count(), 0, 'A previous destination response cannot replace the selected server.');
    assert.ok(await page.evaluate(() => window.fixtureAborts) > aborts, 'Selection must abort its previous status request.');
    planMode = 'held';
    await page.getByRole('button', { name: 'Review load', exact: true }).click();
    await page.waitForTimeout(100);
    assert.equal(held.length, 1);
    await select('Basement server');
    await page.getByText('basement-current', { exact: true }).waitFor();
    for (const finish of held.splice(0)) finish();
    await page.waitForTimeout(100);
    assert.equal(await page.getByRole('dialog').count(), 0, 'A stale plan cannot open a confirmation for a different selection.');
    assert.equal(commits().length, 2);
    const beforePoll = reads('basement');
    await page.waitForTimeout(2700);
    assert.ok(reads('basement') > beforePoll, 'Selected remote status must poll.');
    await select('This Mac');
    await page.getByText('http://this-mac.invalid/v1', { exact: true }).waitFor();
    const afterLocal = reads('basement');
    await page.waitForTimeout(2700);
    assert.equal(reads('basement'), afterLocal, 'Returning to This Mac must stop remote polling.');
    await select('Slow server'); await page.waitForTimeout(100);
    const beforeUnmount = calls.length, abortsBeforeUnmount = await page.evaluate(() => window.fixtureAborts);
    await page.evaluate(() => window.unmountFixture());
    for (const finish of held.splice(0)) finish();
    await page.waitForTimeout(2700);
    assert.equal(calls.length, beforeUnmount, 'Unmounting must stop all polling.');
    assert.ok(await page.evaluate(() => window.fixtureAborts) > abortsBeforeUnmount, 'Unmounting must abort the in-flight status request.');
    assert.deepEqual(exceptions, []);
    assert.deepEqual(blocked, [], 'Serving UI must contact only the host fixture, never a saved remote address.');
});
