/** Real connection-card and management UI over test-owned endpoints; no model or external server is contacted. */
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
const skip = executablePath ? false : 'Set STRATA_UI_BROWSER to an installed Chromium executable; this test never downloads a browser.';

test('saved connection manages remote downloads and existing GGUF files through the reviewed contract', { skip, timeout: 45_000 }, async t => {
    const source = join(root, 'src');
    const entry = `import React from 'react'; import {createRoot} from 'react-dom/client'; import {SettingsModels} from ${JSON.stringify(join(source, 'components/SettingsModels.tsx'))}; createRoot(document.getElementById('root')).render(<SettingsModels/>);`;
    const { outputFiles } = await build({ stdin: { contents: entry, resolveDir: source, loader: 'tsx' }, bundle: true, write: false, outfile: 'harness.js', format: 'iife', platform: 'browser', jsx: 'automatic', define: { 'process.env.NODE_ENV': '"production"' }, plugins: [{ name: 'isolated-settings-state', setup(builder) {
        builder.onResolve({ filter: /(?:^|\/)store$/ }, args => args.importer.includes('/src/') ? { path: 'isolated-store', namespace: 'isolated-store' } : undefined);
        builder.onLoad({ filter: /.*/, namespace: 'isolated-store' }, () => ({ contents: `const state={streaming:false,providers:{catalog:[{id:'fixture-provider',name:'Fixture provider',baseUrl:'https://provider.invalid/v1',protocol:'openai-chat'}],connections:[{id:'saved-connection',kind:'custom',name:'Garage server',baseUrl:'http://fixture.invalid/v1',protocol:'openai-chat',apiKeySet:true,models:[{id:'existing-model',name:'Existing model'}]}]}}; export const useStore=selector=>selector(state); export const removeModelConnection=async()=>{throw Error('Unexpected connection removal')}; export const saveModelConnection=async draft=>{const response=await fetch('/fixture-save',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(draft)});if(!response.ok)throw Error('Fixture save failed'); const connection={...draft,id:draft.id||'new-connection',apiKeySet:false}; state.providers.connections=state.providers.connections.some(c=>c.id===connection.id)?state.providers.connections.map(c=>c.id===connection.id?connection:c):[...state.providers.connections,connection]}; export const dismissActionError=()=>{}; export const dismissToast=()=>{}; export const holdToast=()=>{}; export const releaseToast=()=>{};`, loader: 'js' }));
    } }] });
    const script = outputFiles.find(file => file.path.endsWith('.js')).text;
    const css = ['styles/kit/tokens.css', 'styles/kit/components.css', 'styles/app.css', 'styles/annotation-fixes.css'].map(path => readFileSync(join(source, path), 'utf8')).join('\n') + '\n' + (outputFiles.find(file => file.path.endsWith('.css'))?.text ?? '');
    const calls = [], exceptions = [], blocked = [];
    let job = null, imported = false, planMode = 'valid', sshMode = false;
    const response = (res, value, status = 200, type = 'application/json') => { res.writeHead(status, { 'content-type': type, 'cache-control': 'no-store' }); res.end(type === 'application/json' ? JSON.stringify(value) : value); };
    const server = http.createServer(async (req, res) => {
        const path = new URL(req.url, 'http://fixture').pathname;
        if (path === '/') return response(res, '<!doctype html><meta charset="utf-8"><link rel="stylesheet" href="/harness.css"><div id="root" style="max-width:1000px;margin:30px auto;padding:20px"></div><script src="/harness.js"></script>', 200, 'text/html');
        if (path === '/harness.js') return response(res, script, 200, 'text/javascript');
        if (path === '/harness.css') return response(res, css, 200, 'text/css');
        if (path === '/icons/sprite.svg') return response(res, readFileSync(join(root, 'public/icons/sprite.svg')), 200, 'image/svg+xml');
        if (/^\/fonts\/GTWalsheimPro-(Regular|Medium|Bold|Black)\.woff2$/.test(path)) return response(res, readFileSync(join(root, 'public', path)), 200, 'font/woff2');
        if (path === '/favicon.ico') { res.writeHead(204); return res.end(); }
        let input = ''; for await (const chunk of req) input += chunk;
        let body; try { body = input ? JSON.parse(input) : undefined; } catch { return response(res, { error: 'Invalid fixture JSON' }, 400); }
        calls.push({ path, method: req.method, body });
        if (path === '/fixture-save' && req.method === 'POST') return response(res, { saved: true });
        const target = '/api/model-management/targets/target-fixture';
        if (path === '/api/model-management/targets' && req.method === 'GET') return response(res, { targets: [{ id: 'target-fixture', name: 'Garage server', kind: 'connection', connectionId: 'saved-connection' }] });
        if (path === target && req.method === 'GET') return response(res, { supported: true, targetName: 'Garage server', ...(sshMode ? { storageMode: 'files' } : {}), capabilities: { downloads: true, imports: true }, models: [
            { id: 'native-compact', name: 'Qwen Compact', sizeGb: 4, state: sshMode ? 'downloaded' : 'available', readPictures: true },
            { id: 'local-existing', name: 'Existing GGUF', sizeGb: 2, state: 'imported', readPictures: false },
            ...(imported ? [{ id: 'local-imported', name: 'Text model', sizeGb: 2, state: 'imported', readPictures: false }] : []),
        ], jobs: job ? [job] : [] });
        if (path === target + '/plan' && req.method === 'POST') return response(res, { planId: 'download-plan', targetName: 'Garage server', modelName: 'Qwen Compact', downloadGb: 4, diskFreeGb: 80, diskNeededGb: 12, diskShort: planMode === 'disk-short' ? 'Not enough free space on Garage server.' : null, readPictures: true, warnings: ['Loaded native models may be unloaded before preparation.'], expiresAt: Date.now() + (planMode === 'expired' ? -1000 : 60_000) });
        if (path === target + '/downloads' && req.method === 'POST') {
            job = { id: 'download-job', modelId: 'native-compact', name: 'Qwen Compact', kind: 'download', state: 'running', summary: 'Downloading model files', percent: 25, doneGb: 1, totalGb: 4 };
            return response(res, { job });
        }
        if (path === target + '/downloads/download-job/cancel' && req.method === 'POST') { job = { ...job, state: 'cancelled', summary: 'Stopped. Completed files are kept.' }; return response(res, { job }); }
        if (path === target + '/imports/browse' && req.method === 'POST') {
            const directory = body?.directory || '/models';
            return response(res, { directory, parent: directory === '/models' ? null : '/models', entries: directory === '/models' ? [{ name: 'More models', path: '/models/more', kind: 'folder' }] : [{ name: 'Text model.gguf', path: '/models/more/text-model.gguf', kind: 'model', bytes: 2e9 }] });
        }
        if (path === target + '/imports/inspect' && req.method === 'POST') return response(res, { planId: 'import-plan', targetName: 'Garage server', modelName: 'Text model', sizeGb: 2, files: ['text-model.gguf'], warnings: ['Existing files stay on Garage server.'], expiresAt: Date.now() + 60_000 });
        if (path === target + '/imports' && req.method === 'POST') { imported = true; return response(res, { model: { id: 'local-imported', name: 'Text model' } }); }
        return response(res, { error: 'Unexpected fixture endpoint' }, 404);
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
    const evidence = process.env.STRATA_MANAGEMENT_UI_EVIDENCE_DIR;
    async function screenshot(name) { if (evidence) { mkdirSync(evidence, { recursive: true }); await page.screenshot({ path: join(evidence, name + '.png') }); } }
    await page.goto(origin);
    await page.getByRole('button', { name: 'Manage models', exact: true }).waitFor();
    assert.equal(calls.length, 0, 'Opening Settings must not probe or mutate a remote computer.');
    await page.getByRole('button', { name: 'Manage models', exact: true }).click();
    const dialog = page.getByRole('dialog', { name: 'Models on Garage server', exact: true });
    await dialog.getByText('Qwen Compact', { exact: true }).waitFor();
    assert.equal(await dialog.getByRole('button', { name: 'Imported', exact: true }).isDisabled(), true);
    await screenshot('remote-catalog');
    await dialog.getByRole('button', { name: 'Review download', exact: true }).click();
    const download = dialog.getByRole('button', { name: 'Download on Garage server', exact: true });
    await download.waitFor(); assert.equal(await download.isEnabled(), true, 'Epoch-millisecond expiry must leave a current plan actionable.');
    assert.deepEqual(calls.find(call => call.path.endsWith('/plan')).body, { modelId: 'native-compact', readPictures: true });
    await screenshot('download-review'); await download.click();
    await dialog.getByText('Downloading model files', { exact: true }).waitFor();
    assert.deepEqual(calls.find(call => call.path.endsWith('/downloads')).body, { planId: 'download-plan' });
    assert.equal(await dialog.getByRole('progressbar').getAttribute('value'), '25');
    await screenshot('download-progress');
    await dialog.getByRole('button', { name: 'Stop download', exact: true }).click();
    await dialog.getByText('Stopped. Completed files are kept.', { exact: true }).waitFor();
    assert.deepEqual(calls.find(call => call.path.endsWith('/cancel')).body, {});
    await dialog.getByRole('button', { name: 'Import GGUF', exact: true }).click();
    await dialog.getByRole('button', { name: 'Open server model folder', exact: true }).click();
    await dialog.getByRole('button', { name: 'More models', exact: true }).click();
    await dialog.getByRole('button', { name: /Text model\.gguf/ }).click();
    const confirmImport = dialog.getByRole('button', { name: 'Import on Garage server', exact: true });
    await confirmImport.waitFor(); assert.equal(await confirmImport.isEnabled(), true);
    assert.deepEqual(calls.find(call => call.path.endsWith('/imports/inspect')).body, { path: '/models/more/text-model.gguf' });
    await screenshot('import-review'); await confirmImport.click();
    await dialog.getByText('Text model was imported on Garage server. Load it there to make it available through the API.', { exact: true }).waitFor();
    assert.deepEqual(calls.find(call => call.path.endsWith('/imports')).body, { planId: 'import-plan' });
    await dialog.getByRole('button', { name: 'Download models', exact: true }).click();
    assert.equal(await dialog.getByRole('button', { name: 'Imported', exact: true }).count(), 2);
    for (const mode of ['expired', 'disk-short']) {
        planMode = mode;
        await dialog.getByRole('button', { name: 'Review download', exact: true }).click();
        await download.waitFor(); assert.equal(await download.isDisabled(), true, `${mode} plans cannot be confirmed.`);
        await dialog.getByRole('button', { name: 'Back', exact: true }).click();
    }
    assert.equal(calls.filter(call => call.path.endsWith('/downloads')).length, 1);
    await dialog.getByRole('button', { name: 'Close', exact: true }).click();
    await page.waitForTimeout(100);
    const readsAfterClose = calls.length;
    await page.waitForTimeout(2800);
    assert.equal(calls.length, readsAfterClose, 'Closing the manager must stop polling.');
    await page.getByRole('button', { name: 'Add model provider', exact: true }).click();
    assert.equal(await page.getByRole('button', { name: 'Apply', exact: true }).isDisabled(), true, 'Provider connections still require a chat model.');
    await page.getByRole('button', { name: 'Custom model API', exact: true }).click();
    await page.getByText('Save this connection to manage models on another Strata computer, or fetch/add models for chat.', { exact: true }).waitFor();
    await page.getByLabel('Display name', { exact: true }).fill('Empty Strata server');
    await page.getByLabel(/^Base URL/).fill('http://empty-fixture.invalid/v1');
    assert.equal(await page.getByRole('button', { name: 'Apply', exact: true }).isEnabled(), true, 'An empty Strata server must be saveable before any model is loaded.');
    await page.getByRole('button', { name: 'Apply', exact: true }).click();
    await page.getByText('Connection saved. Use Manage models to download or import models on that computer.', { exact: true }).waitFor();
    const saved = calls.find(call => call.path === '/fixture-save').body;
    assert.equal(saved.kind, 'custom'); assert.equal(saved.name, 'Empty Strata server'); assert.deepEqual(saved.models, []);
    // Configure SSH on the existing empty connection, with no static login defaults.
    await page.getByTitle('Edit Empty Strata server', { exact: true }).click();
    const management = page.getByRole('combobox', { name: 'Model management', exact: true });
    await management.click();
    await page.getByRole('option', { name: 'Terminal server over SSH', exact: true }).click();
    assert.equal(await page.getByLabel('SSH username (optional)', { exact: true }).inputValue(), '');
    assert.equal(await page.getByLabel('SSH port (optional)', { exact: true }).inputValue(), '');
    assert.equal(await page.getByRole('button', { name: 'Apply', exact: true }).isDisabled(), true);
    await page.getByLabel('SSH host or alias', { exact: true }).fill('garage-ssh');
    await page.getByLabel('SSH username (optional)', { exact: true }).fill('alice');
    await page.getByLabel('SSH port (optional)', { exact: true }).fill('70000');
    assert.equal(await page.getByRole('button', { name: 'Apply', exact: true }).isDisabled(), true);
    await page.getByLabel('SSH port (optional)', { exact: true }).fill('2202');
    await page.getByLabel(/^Strata source folder/).fill('/opt/Strata');
    await screenshot('ssh-connection-settings');
    await page.getByRole('button', { name: 'Apply', exact: true }).click();
    await page.getByTitle('Edit Empty Strata server', { exact: true }).waitFor();
    const sshSaved = calls.filter(call => call.path === '/fixture-save').at(-1).body;
    assert.deepEqual(sshSaved.modelManagement, { transport: 'ssh', host: 'garage-ssh', user: 'alice', port: 2202, sourcePath: '/opt/Strata' });
    assert.equal(sshSaved.apiKey, '');
    await page.getByTitle('Edit Empty Strata server', { exact: true }).click();
    assert.equal(await page.getByLabel('SSH host or alias', { exact: true }).inputValue(), 'garage-ssh');
    assert.equal(await page.getByLabel('SSH username (optional)', { exact: true }).inputValue(), 'alice');
    assert.equal(await page.getByLabel('SSH port (optional)', { exact: true }).inputValue(), '2202');
    await management.click();
    await page.getByRole('option', { name: 'Strata desktop API', exact: true }).click();
    assert.equal(await page.getByLabel('SSH host or alias', { exact: true }).count(), 0);
    await page.getByRole('button', { name: 'Apply', exact: true }).click();
    await page.getByTitle('Edit Empty Strata server', { exact: true }).waitFor();
    assert.equal(calls.filter(call => call.path === '/fixture-save').at(-1).body.modelManagement, null);
    sshMode = true;
    await page.getByTitle('Manage models on Garage server', { exact: true }).click();
    await dialog.getByRole('button', { name: 'Downloaded', exact: true }).waitFor();
    assert.equal(await dialog.getByRole('button', { name: 'Downloaded', exact: true }).isDisabled(), true);
    await dialog.getByText('SSH manages files in a separate server library. Preparing and loading a model remain separate server actions. Existing serving models continue running.', { exact: true }).waitFor();
    await screenshot('ssh-files-library');
    await dialog.getByRole('button', { name: 'Close', exact: true }).click();
    assert.deepEqual(exceptions, []);
    assert.deepEqual(blocked, [], 'The UI must never contact the configured fixture API address directly.');
});
