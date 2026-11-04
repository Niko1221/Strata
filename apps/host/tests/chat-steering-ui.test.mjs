/** Real shared frontend in Chromium; all model/host responses stay inside this disposable fixture. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createServer } from 'node:http';
import { build } from 'esbuild';
import { chromium } from 'playwright-core';

const chrome = process.env.CHROME_PATH ?? ['/usr/bin/google-chrome', '/usr/bin/chromium'].find(existsSync);

test('desktop settings and composer expose all four modes, retain drafts on failure, and keep Stop', { skip: !chrome && 'Set CHROME_PATH to run the browser regression' }, async t => {
  const { outputFiles } = await build({ stdin: { resolveDir: process.cwd(), loader: 'tsx', contents: `
    import React from 'react';
    import { createRoot } from 'react-dom/client';
    import { ChatScreen } from './src/screens/Chat';
    import { SettingsScreen } from './src/screens/Settings';
    import * as app from './src/lib/store';
    import { applyNativeEvent } from './src/lib/native';
    const native = { available: true, jobs: [], instances: [{ id: 'fixture', model: 'fixture', label: 'Fixture', apiModelId: 'fixture', status: 'ready', gpus: [], metrics: null }], defaultId: 'fixture', fields: [], catalog: null, installed: null, hardware: null };
    const requests = [], steers = []; let releaseQueue;
    window.fixture = { app, requests, steers, failSteer: false, stops: 0,
      deliver: n => { const message = steers[n]; const request = requests.at(-1); request.frame({type:'steering-cut'}); request.frame({type:'steering',message:{id:message.id,text:message.message,state:'delivered'}}); request.frame({type:'delta',text:'Continued after correction'}); },
      finish: n => { requests[n].frame({type:'done',tokensOut:1}); requests[n].end(); },
      fail: n => requests[n].fail(),
      releaseQueue: () => releaseQueue(),
    };
    window.fetch = async (url, init) => {
      const path = new URL(url, location.origin).pathname;
      if (path === '/api/native') return Response.json(native);
      if (path === '/api/fonts') return Response.json({ fonts: [] });
      if (path === '/api/chat/stop') { window.fixture.stops++; return Response.json({ stopped: false }); }
      if (path === '/api/chat/steer') {
        const body = JSON.parse(init.body); steers.push(body);
        if (window.fixture.failSteer) return Response.json({error:'Same-turn steering is unavailable'}, {status:501});
        return Response.json({message:{id:body.id,text:body.message,state:'pending'}});
      }
      if (path === '/api/chat/queue' && init?.method === 'POST') {
        if (app.getState().whileBusy !== 'wait') return Response.json({error:'No goal is running'}, {status:409});
        return new Promise(resolve => { releaseQueue = () => resolve(Response.json({error:'No goal is running'}, {status:409})); });
      }
      if (path === '/api/chat/stream') {
        let controller;
        const request = { body: JSON.parse(init.body), frame: value => controller.enqueue(new TextEncoder().encode(JSON.stringify(value)+'\\n')), end: () => controller.close(), fail: () => controller.error(new Error('Fixture connection lost')) };
        const body = new ReadableStream({start(c){controller=c;}});
        init.signal.addEventListener('abort', () => controller.error(new DOMException('Stopped', 'AbortError')), {once:true});
        requests.push(request);
        request.frame({type:'start',model:'fixture',contextMax:32000,agent:true});
        request.frame({type:'delta',text:'Partial fixture answer '+requests.length});
        return new Response(body);
      }
      if (path === '/api/chat/queue') return Response.json({queue:[]});
      return Response.json({});
    };
    applyNativeEvent(native);
    if (!app.getState().sessions.length) app.newSession();
    createRoot(document.getElementById('root')).render(<><ChatScreen/><SettingsScreen/></>);
  ` }, bundle: true, write: false, format: 'iife', platform: 'browser', jsx: 'automatic', loader: { '.css': 'empty' }, logLevel: 'silent', plugins: [{ name: 'fixture-url', setup(b) {
    b.onResolve({ filter: /\?url$/ }, args => ({ path: args.path, namespace: 'fixture-url' }));
    b.onLoad({ filter: /.*/, namespace: 'fixture-url' }, () => ({ contents: 'export default "/unused-fixture-worker.js";' }));
  } }] });
  const server = createServer((req, res) => {
    res.setHeader('content-type', req.url === '/fixture.js' ? 'text/javascript' : 'text/html');
    res.end(req.url === '/fixture.js' ? outputFiles[0].text : '<!doctype html><html><body><div id="root"></div><script src="/fixture.js"></script></body></html>');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => server.close(resolve)));
  const browser = await chromium.launch({ executablePath: chrome, headless: true });
  t.after(() => browser.close());
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  page.setDefaultTimeout(10000);
  const errors = []; page.on('pageerror', error => { errors.push(error.message); console.error(error.stack); });
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const input = page.getByRole('textbox', { name: 'Message Strata', exact: true });
  const busyChoice = page.getByRole('combobox', { name: 'When you send while it is answering', exact: true });
  await busyChoice.waitFor();
  assert.match(await busyChoice.innerText(), /Wait/);
  await input.fill('First question');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await page.getByRole('button', { name: 'Stop', exact: true }).waitFor();
  assert.equal(await page.getByRole('button', { name: 'Send', exact: true }).count(), 0);
  await input.fill('Keep this draft'); await input.press('Enter');
  await page.waitForFunction(() => !!window.fixture.releaseQueue && window.fixture.app.getState().streaming);
  await input.fill('Newer notes');
  await page.evaluate(() => window.fixture.releaseQueue());
  await page.waitForFunction(() => document.querySelector('textarea').value === 'Keep this draft\n\nNewer notes');
  const choose = async label => { await busyChoice.click(); await page.getByRole('option', { name: label, exact: true }).click(); };
  await choose('Send after');
  await input.fill('Next question');
  await page.getByRole('button', { name: 'Send after', exact: true }).click();
  await page.getByText('Sends when this answer is done', { exact: true }).waitFor();
  assert.equal(await input.inputValue(), '');
  assert.equal(await page.evaluate(() => window.fixture.requests.length), 1);
  await page.evaluate(() => window.fixture.finish(0));
  await page.waitForFunction(() => window.fixture.requests.length === 2);
  assert.equal(await page.evaluate(() => window.fixture.requests[1].body.message), 'Next question');
  await choose('Interrupt');
  await input.fill('Correction');
  await page.getByRole('button', { name: 'Interrupt', exact: true }).click();
  await page.waitForFunction(() => window.fixture.requests.length === 3);
  assert.equal(await page.evaluate(() => window.fixture.requests[2].body.message), 'Correction');
  assert.ok(await page.evaluate(() => window.fixture.requests[2].body.history.some(m => m.content === 'Partial fixture answer 2')));
  await page.getByRole('button', { name: 'Stop', exact: true }).waitFor();
  await choose('Steer current run');
  await input.fill('Correct the current run');
  await page.getByRole('button', { name: 'Steer current run', exact: true }).click();
  await page.getByText('Waiting for a safe steering point', { exact: true }).waitFor();
  assert.equal(await page.evaluate(() => window.fixture.requests.length), 3);
  assert.equal(await page.evaluate(() => window.fixture.stops), 1);
  await page.evaluate(() => window.fixture.deliver(0));
  await page.getByText('Steering delivered', { exact: true }).waitFor();
  assert.equal(await page.evaluate(() => window.fixture.requests.length), 3);
  await choose('Send after');
  await input.fill('Recover after failure'); await input.press('Enter');
  await page.getByText('Sends when this answer is done', { exact: true }).waitFor();
  await page.evaluate(() => window.fixture.fail(2));
  await page.getByText('Not sent yet', { exact: true }).waitFor();
  await page.reload();
  await page.getByText('Not sent yet', { exact: true }).waitFor();
  assert.match(await busyChoice.innerText(), /Send after/);
  await page.getByRole('button', { name: 'Take back', exact: true }).click();
  assert.equal(await input.inputValue(), 'Recover after failure');
  await input.fill('New question'); await page.getByRole('button', { name: 'Send', exact: true }).click();
  await choose('Steer current run');
  await page.evaluate(() => { window.fixture.failSteer = true; });
  await input.fill('Keep failed steering'); await page.getByRole('button', { name: 'Steer current run', exact: true }).click();
  await page.getByText('Steering delivery not confirmed', { exact: true }).waitFor();
  await page.evaluate(() => { window.fixture.failSteer = false; });
  await page.getByRole('button', { name: 'Retry', exact: true }).click();
  await page.getByText('Waiting for a safe steering point', { exact: true }).waitFor();
  assert.equal(await page.evaluate(() => window.fixture.steers[0].id === window.fixture.steers[1].id), true);
  assert.equal(await page.evaluate(() => window.fixture.requests.length), 1);
  assert.deepEqual(errors, []);
});
