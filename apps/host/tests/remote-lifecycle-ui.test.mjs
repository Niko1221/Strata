/** Real production lifecycle panel with disposable local HTTP fixtures; no SSH or model process. */
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
const skip = executablePath ? false : 'Set STRATA_UI_BROWSER to an installed Chromium; no browser is downloaded.';

test('Mac lifecycle reviews preparation, owned switching and separate connection without leaking an old target action', { skip, timeout: 45_000 }, async t => {
    const source = join(root, 'src');
    const entry = `import React from 'react'; import {createRoot} from 'react-dom/client'; import {RemoteModelLifecycle} from ${JSON.stringify(join(source, 'components/RemoteModelLifecycle.tsx'))};
      const root=createRoot(document.getElementById('root')); window.fixtureAborts=0; const original=window.fetch; window.fetch=(url,options)=>{options?.signal?.addEventListener('abort',()=>window.fixtureAborts++);return original(url,options)};
      window.unmountFixture=()=>root.render(null); window.mountFixture=(id='garage')=>root.render(<RemoteModelLifecycle targetId={id} name={id==='garage'?'Garage server':'Other server'} onClose={()=>root.render(null)}/>);window.mountFixture();`;
    const { outputFiles } = await build({ stdin: { contents: entry, resolveDir: source, loader: 'tsx' }, bundle: true, write: false, outfile: 'harness.js', format: 'iife', platform: 'browser', jsx: 'automatic', define: { 'process.env.NODE_ENV': '"production"' }, plugins: [{ name: 'isolated-store', setup(builder) {
        builder.onResolve({ filter: /(?:^|\/)store$/ }, args => args.importer.includes('/src/') ? { path: 'fixture-store', namespace: 'fixture' } : undefined);
        builder.onLoad({ filter: /.*/, namespace: 'fixture' }, () => ({ contents: 'export const useStore=s=>s({});export const pushToast=()=>{},dismissActionError=()=>{},dismissToast=()=>{},holdToast=()=>{},releaseToast=()=>{};', loader: 'js' }));
    } }] });
    const script = outputFiles.find(file => file.path.endsWith('.js')).text;
    const css = ['styles/kit/tokens.css', 'styles/kit/components.css', 'styles/app.css', 'styles/annotation-fixes.css'].map(path => readFileSync(join(source, path), 'utf8')).join('\n') + '\n' + readFileSync(join(source, 'components/RemoteModelManager.css'), 'utf8') + '\n' + (outputFiles.find(file => file.path.endsWith('.css'))?.text ?? '');
    const calls = [], errors = [], blocked = [], held = [];
    let profiles = [], jobs = [], runtime = { state: 'stopped' }, planMode = 'valid', sequence = 0;
    const snapshot = id => ({ supported: true, targetName: id === 'garage' ? 'Garage server' : 'Other server', capabilities: { prepare: true, serve: true },
        models: [{ id: 'downloaded', name: 'Downloaded Qwen', state: 'downloaded', preparable: true }, { id: 'unsupported', name: 'Unsupported GGUF', state: 'imported', preparable: false, reason: 'This architecture is not supported by Strata preparation.' }],
        engines: [{ id: 'existing-engine', name: 'Existing Strata engine' }], profiles, jobs, runtime });
    const respond = (res, value, status = 200, type = 'application/json') => { if (res.destroyed) return; res.writeHead(status, { 'content-type': type, 'cache-control': 'no-store' });res.end(type === 'application/json' ? JSON.stringify(value) : value); };
    const server = http.createServer(async (req, res) => {
        const path = new URL(req.url, 'http://fixture').pathname;
        if (path === '/') return respond(res, '<!doctype html><html data-theme="dark"><meta charset="utf-8"><link rel="stylesheet" href="/harness.css"><div id="root"></div><script src="/harness.js"></script>', 200, 'text/html');
        if (path === '/harness.js') return respond(res, script, 200, 'text/javascript');
        if (path === '/harness.css') return respond(res, css, 200, 'text/css');
        if (path === '/icons/sprite.svg') return respond(res, readFileSync(join(root, 'public/icons/sprite.svg')), 200, 'image/svg+xml');
        if (/^\/fonts\/GTWalsheimPro-(Regular|Medium|Bold|Black)\.woff2$/.test(path)) return respond(res, readFileSync(join(root, 'public', path)), 200, 'font/woff2');
        if (path === '/favicon.ico') { res.writeHead(204);return res.end(); }
        let input = '';for await (const chunk of req) input += chunk;
        const body = input ? JSON.parse(input) : undefined;calls.push({ path, method: req.method, body });
        const match = /^\/api\/remote-lifecycle\/targets\/(garage|other)(.*)$/.exec(path);
        if (!match) return respond(res, { error: 'Unexpected fixture endpoint' }, 404);
        const [, id, suffix] = match;
        if (!suffix && req.method === 'GET') return respond(res, snapshot(id));
        if (suffix.endsWith('-plan')) {
            const prepare = suffix === '/prepare-plan';
            const plan = { planId: 'plan-' + ++sequence, targetName: id === 'garage' ? 'Garage server' : 'Other server', modelName: 'Downloaded Qwen', expiresAt: Date.now() + (planMode === 'expired' ? 450 : 60_000), warnings: prepare ? ['Copies and helper files stay in a separate library. Existing services are not stopped.'] : ['Only the managed Strata API is switched. The new model starts unloaded.'], ...(prepare ? { engineName: 'Existing Strata engine', context: body.context, diskNeededGb: 10, diskFreeGb: 80, diskShort: planMode === 'disk-short' } : { profileId: body.profileId, host: '0.0.0.0', port: 18090, action: runtime.state === 'ready' ? 'switch' : 'start' }) };
            if (planMode === 'held') { held.push(() => respond(res, plan)); return; }
            return respond(res, plan);
        }
        if (suffix === '/prepare') { jobs = [{ id: 'prepare-job', kind: 'prepare', state: 'running', requestId: body.planId, summary: 'Preparing private model files…', canCancel: true, percent: 20 }];return respond(res, { job: jobs[0] }); }
        if (suffix === '/jobs/prepare-job/cancel') { jobs[0] = { ...jobs[0], state: 'cancelled', summary: 'Preparation stopped. Existing services were preserved.' };return respond(res, { job: jobs[0] }); }
        if (suffix === '/serve') {
            const profileId = runtime.state === 'ready' ? 'profile-two' : 'profile-one';
            runtime = { state: 'ready', profileId, modelId: 'downloaded', loaded: false, busy: false, host: '0.0.0.0', port: 18090 };
            const job = { id: 'serve-' + sequence, kind: 'serve', state: 'done', requestId: body.planId, profileId, receiptId: 'connection-receipt', summary: 'Managed API is listening. The model is unloaded.', canCancel: false };
            jobs.push(job);return respond(res, { job });
        }
        if (suffix === '/connect') return respond(res, { connectionId: 'new-separate-connection', name: 'Garage server · managed Strata' });
        return respond(res, { error: 'Unexpected lifecycle mutation' }, 400);
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));t.after(() => { server.closeAllConnections();server.close(); });
    const origin = `http://127.0.0.1:${server.address().port}`;
    const browser = await chromium.launch({ executablePath, headless: true, args: ['--disable-background-networking', '--disable-component-update', '--disable-sync', '--no-first-run'] });t.after(() => browser.close());
    const page = await browser.newPage({ viewport: { width: 1200, height: 1000 }, colorScheme: 'dark' });page.on('pageerror', e => errors.push(e.message));
    await page.route('**/*', route => new URL(route.request().url()).origin === origin ? route.continue() : (blocked.push(route.request().url()), route.abort()));
    const evidence = process.env.STRATA_LIFECYCLE_UI_EVIDENCE_DIR;
    async function screenshot(name) { if (evidence) { mkdirSync(evidence, { recursive: true });await page.screenshot({ path: join(evidence, name + '.png') }); } }
    await page.goto(origin);
    const panel = page.getByRole('dialog');await panel.getByRole('button', { name: 'Review preparation', exact: true }).waitFor();
    assert.equal(calls.filter(call => call.method === 'POST').length, 0, 'Opening the manager only reads status.');
    await panel.getByRole('combobox', { name: 'Model files to prepare', exact: true }).click();await page.getByRole('option', { name: 'Unsupported GGUF', exact: true }).click();
    assert.equal(await panel.getByRole('button', { name: 'Review preparation', exact: true }).isDisabled(), true);
    await panel.getByText('This architecture is not supported by Strata preparation.', { exact: true }).waitFor();
    await panel.getByRole('combobox', { name: 'Model files to prepare', exact: true }).click();await page.getByRole('option', { name: 'Downloaded Qwen', exact: true }).click();
    await screenshot('prepare-form');
    for (const mode of ['expired', 'disk-short']) {
        planMode = mode;await panel.getByRole('button', { name: 'Review preparation', exact: true }).click();
        const confirm = panel.getByRole('button', { name: 'Prepare on Garage server', exact: true });await confirm.waitFor();
        if (mode === 'expired') await page.waitForTimeout(550);
        assert.equal(await confirm.isDisabled(), true);await panel.getByRole('button', { name: 'Back', exact: true }).click();
    }
    planMode = 'valid';await panel.getByRole('button', { name: 'Review preparation', exact: true }).click();
    assert.deepEqual(calls.filter(call => call.path.endsWith('/prepare-plan')).at(-1).body, { modelId: 'downloaded', engineId: 'existing-engine', context: 32768 });
    await panel.getByRole('button', { name: 'Prepare on Garage server', exact: true }).waitFor();await screenshot('prepare-review');
    assert.equal(calls.filter(call => call.path.endsWith('/prepare')).length, 0);await panel.getByRole('button', { name: 'Prepare on Garage server', exact: true }).click();
    await panel.getByText('Preparing private model files…', { exact: true }).waitFor();await panel.getByRole('button', { name: 'Stop preparation', exact: true }).click();
    await panel.getByText('Preparation stopped. Existing services were preserved.', { exact: true }).waitFor();
    await panel.getByRole('button', { name: 'Review preparation', exact: true }).click();await panel.getByRole('button', { name: 'Prepare on Garage server', exact: true }).click();
    await panel.getByText('Preparing private model files…', { exact: true }).waitFor();
    profiles = [{ id: 'profile-one', name: 'Qwen 32K', modelId: 'downloaded', context: 32768, state: 'prepared', executionCompatibility: 'unverified' }, { id: 'profile-two', name: 'Qwen 64K', modelId: 'downloaded', context: 65536, state: 'prepared', executionCompatibility: 'unverified' }];
    jobs[0] = { ...jobs[0], state: 'done', profileId: 'profile-one', summary: 'Preparation finished.' };
    await panel.getByRole('button', { name: 'Refresh', exact: true }).click();await panel.getByText('Preparation finished.', { exact: true }).waitFor();
    await panel.getByRole('button', { name: 'Review start API', exact: true }).first().click();await panel.getByRole('button', { name: 'Start API on Garage server', exact: true }).waitFor();await screenshot('serve-review');
    assert.equal(calls.filter(call => call.path.endsWith('/serve')).length, 0);await panel.getByRole('button', { name: 'Start API on Garage server', exact: true }).click();
    await panel.getByRole('button', { name: 'Save managed connection', exact: true }).waitFor();await panel.getByRole('button', { name: 'Save managed connection', exact: true }).click();
    await panel.getByText('Garage server · managed Strata is saved as a separate connection. Select it in the chat model picker or Serving → Destination.', { exact: true }).waitFor();
    assert.deepEqual(calls.find(call => call.path.endsWith('/connect')).body, { receiptId: 'connection-receipt' });
    await panel.getByRole('button', { name: 'Refresh', exact: true }).click();await panel.getByRole('button', { name: 'Current model', exact: true }).waitFor();
    runtime.busy = true;await panel.getByRole('button', { name: 'Refresh', exact: true }).click();await panel.getByText('The managed API is busy. Wait for its requests to finish before switching.', { exact: true }).waitFor();
    assert.equal(await panel.getByRole('button', { name: 'Review switch', exact: true }).isDisabled(), true);
    runtime.busy = null;await panel.getByRole('button', { name: 'Refresh', exact: true }).click();
    await panel.getByText('The managed API has not reported reliable idle status. Switching stays unavailable until it can be verified.', { exact: true }).waitFor();
    assert.equal(await panel.getByRole('button', { name: 'Review switch', exact: true }).isDisabled(), true);
    runtime.busy = false;await panel.getByRole('button', { name: 'Refresh', exact: true }).click();
    await panel.getByRole('button', { name: 'Review switch', exact: true }).click();await panel.getByRole('button', { name: 'Switch on Garage server', exact: true }).waitFor();await screenshot('switch-review');
    await panel.getByRole('button', { name: 'Switch on Garage server', exact: true }).click();
    await panel.getByRole('button', { name: 'Refresh', exact: true }).click();await panel.getByRole('button', { name: 'Current model', exact: true }).waitFor();
    assert.equal(calls.filter(call => call.path.endsWith('/serve')).length, 2);
    const postsBeforeStale = calls.filter(call => call.method === 'POST').length;
    planMode = 'held';await panel.getByRole('button', { name: 'Review preparation', exact: true }).click();await page.waitForTimeout(100);assert.equal(held.length, 1);
    const beforeAborts = await page.evaluate(() => window.fixtureAborts);await page.evaluate(() => window.mountFixture('other'));
    await page.getByRole('dialog', { name: 'Prepare and serve on Other server', exact: true }).waitFor();for (const finish of held.splice(0)) finish();await page.waitForTimeout(100);
    assert.equal(await page.getByRole('button', { name: 'Prepare on Garage server', exact: true }).count(), 0);assert.ok(await page.evaluate(() => window.fixtureAborts) > beforeAborts);
    assert.equal(calls.filter(call => call.method === 'POST').length, postsBeforeStale + 1, 'No stale preview can commit after a destination change.');
    const reads = calls.length;await page.evaluate(() => window.unmountFixture());await page.waitForTimeout(2800);assert.equal(calls.length, reads, 'Unmount stops lifecycle polling.');
    assert.deepEqual(errors, []);assert.deepEqual(blocked, []);
});
