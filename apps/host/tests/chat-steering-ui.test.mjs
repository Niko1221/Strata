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
    const requests = [], steers = [], hostChats = new Map(JSON.parse(sessionStorage.getItem('fixture-host-chats') ?? '[]')); let releaseQueue;
    window.fixture = { app, requests, steers, hostChats, failSteer: false, stops: 0,
      deliver: n => { const message = steers[n]; const request = requests.at(-1); request.frame({type:'steering-cut'}); request.frame({type:'steering',message:{id:message.id,text:message.message,state:'delivered'}}); request.frame({type:'delta',text:'Continued after correction'}); },
      finish: n => { requests[n].frame({type:'done',tokensOut:1}); requests[n].end(); },
      fail: n => requests[n].fail(),
      releaseQueue: () => releaseQueue(),
    };
    window.fetch = async (url, init) => {
      const path = new URL(url, location.origin).pathname;
      if (path === '/api/chat/active') return Response.json({turns:[]});
      if (path === '/api/chats' && (!init?.method || init.method === 'GET')) return Response.json({chats:[...hostChats].map(([id,value])=>({id,rev:value.rev,updatedAt:value.session.updatedAt}))});
      if (path === '/api/chats/import' && init?.method === 'POST') {
        const imported = JSON.parse(init.body).sessions ?? [];
        for (const session of imported) if (!hostChats.has(session.id)) hostChats.set(session.id,{session,rev:1});
        sessionStorage.setItem('fixture-host-chats', JSON.stringify([...hostChats]));
        return Response.json({imported:imported.map(s=>s.id),skipped:[],failed:[]});
      }
      const chatId = path.startsWith('/api/chats/') ? decodeURIComponent(path.slice('/api/chats/'.length)) : '';
      if (chatId && !chatId.includes('/')) {
        const id = chatId, current = hostChats.get(id);
        if (init?.method === 'GET') return current ? Response.json(current) : Response.json({error:'No such chat.',code:'not-found'},{status:404});
        if (init?.method === 'PUT') { const session=JSON.parse(init.body), value={session,rev:(current?.rev??0)+1}; hostChats.set(id,value); sessionStorage.setItem('fixture-host-chats', JSON.stringify([...hostChats])); return Response.json({rev:value.rev,conflict:false}); }
        if (init?.method === 'DELETE') { hostChats.delete(id); sessionStorage.setItem('fixture-host-chats', JSON.stringify([...hostChats])); return Response.json({}); }
      }
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
        request.frame({type:'steering-availability',available:true});
        request.frame({type:'delta',text:'Partial fixture answer '+requests.length});
        return new Response(body);
      }
      if (path === '/api/chat/queue') return Response.json({queue:[]});
      return Response.json({});
    };
    applyNativeEvent(native);
    void app.loadChatsFromHost().then(() => {
      if (!app.getState().sessions.length) app.newSession();
      else app.selectSession(app.getState().sessions[0].id);
      createRoot(document.getElementById('root')).render(<><ChatScreen/><SettingsScreen/></>);
    });
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
  const busyChoice = page.getByRole('combobox', { name: 'While an answer is being written', exact: true });
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
  const choose = async label => { await busyChoice.click(); await page.getByRole('option', { name: label === 'Send after' ? 'Send after the answer' : label === 'Interrupt' ? 'Interrupt the answer' : label, exact: true }).click(); };
  await choose('Send after');
  await input.fill('Next question');
  await page.getByRole('button', { name: 'Send after', exact: true }).click();
  await page.getByText('Sends when this answer is done', { exact: true }).waitFor();
  assert.equal(await input.inputValue(), '');
  assert.equal(await page.evaluate(() => window.fixture.requests.length), 1);
  await input.fill('Another queued message');
  await page.getByRole('button', { name: 'Send after', exact: true }).click();
  await page.getByText('Waiting to send', { exact: true }).waitFor();
  await input.fill('Newer draft');
  await page.locator('.parked').getByRole('button', { name: 'Take back', exact: true }).click();
  assert.equal(await input.inputValue(), 'Another queued message\n\nNewer draft');
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
  await page.evaluate(() => window.fixture.requests.at(-1).frame({type:'steering-availability',available:false}));
  await input.fill('Keep while helpers work');
  assert.equal(await page.getByRole('button', { name: 'Steering unavailable', exact: true }).isDisabled(), true);
  await input.press('Enter');
  assert.equal(await input.inputValue(), 'Keep while helpers work');
  assert.equal(await page.evaluate(() => window.fixture.steers.length), 0);
  await page.evaluate(() => window.fixture.requests.at(-1).frame({type:'steering-availability',available:true}));
  await page.evaluate(() => window.fixture.app.addQuote(window.fixture.app.getState().activeSession, 'Keep the upstream quote', 'fixture-source', 'assistant'));
  await input.fill('Correct the current run');
  await page.getByRole('button', { name: 'Steer current run', exact: true }).click();
  await page.getByText('Waiting for a safe steering point', { exact: true }).waitFor();
  assert.equal(await page.evaluate(() => window.fixture.requests.length), 3);
  assert.equal(await page.evaluate(() => window.fixture.stops), 1);
  assert.match(await page.evaluate(() => window.fixture.steers[0].message), /Keep the upstream quote/);
  assert.equal(await page.evaluate(() => Object.keys(window.fixture.app.getState().quotes).length), 0);
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
