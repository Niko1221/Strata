// CPU-only browser contract test. A local fixture serves the real Chat page;
// no model, profile database, or user key is opened.
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {createServer} = require('node:http');
const {readFileSync, mkdtempSync, existsSync, rmSync} = require('node:fs');
const {tmpdir} = require('node:os');
const {join, resolve, extname} = require('node:path');
const {spawn} = require('node:child_process');
const {setTimeout: pause} = require('node:timers/promises');
const {createMemoryTranscripts, persistApiKey, prepareMemoryRequest} = require('./memory-chat.js');

test('memory request and transcript stay scoped to one key and one consent', () => {
  const original = [{role: 'user', text: 'ordinary private history'}];
  const transcripts = createMemoryTranscripts(original);
  assert.equal(transcripts.active(), original);
  assert.deepEqual(transcripts.select(true), []);
  const a = transcripts.setKey('a');
  a.push({role: 'user', text: 'alice turn'});
  assert.equal(transcripts.active().length, 1);
  const version = transcripts.version();
  assert.deepEqual(transcripts.setKey('b'), []);
  assert.ok(transcripts.version() > version);
  assert.equal(transcripts.select(false), original);
  const writes = [];
  const storage = {set: (...args) => {writes.push(args); return true;}};
  persistApiKey(storage, 'profile-secret', true);
  assert.deepEqual(writes, []);
  persistApiKey(storage, 'ordinary-key', false);
  assert.deepEqual(writes, [['apikey', 'ordinary-key']]);
  let n = 0;
  const crypto = {getRandomValues: bytes => { bytes.fill(++n); return bytes; }};
  const options = {key: 'a', entered: true, text: 'synthetic', attachments: [], read: true, storeSummary: false, crypto};
  const request = prepareMemoryRequest({messages: [{role: 'user', content: 'synthetic'}], strata_mcp: true}, options);
  assert.equal(request.strata_mcp, undefined);
  assert.deepEqual(request.strata_memory, {read: true, store_summary: false, turn_id: '01'.repeat(16)});
  assert.equal(prepareMemoryRequest({}, options).strata_memory.turn_id, '02'.repeat(16));
  assert.equal(request.strata_memory.turn_id, '01'.repeat(16), 'retry reuses the same prepared payload');
  assert.throws(() => prepareMemoryRequest({}, {...options, entered: false}), /profile key/);
  assert.throws(() => prepareMemoryRequest({}, {...options, attachments: [{}]}), /text only/);
  assert.throws(() => prepareMemoryRequest({}, {...options, read: false}), /permission/);
});

test('Chat page keeps a streaming turn on its original key and resets consent', async t => {
  const chrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
  if (!existsSync(chrome)) return t.skip('Chrome is unavailable');
  const web = resolve(__dirname, '..');
  const requests = [];
  let releaseFirst;
  const server = createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname === '/health') { res.setHeader('content-type', 'application/json'); res.end(JSON.stringify({status:'ok', model:'mock', max_context:131072})); return; }
    if (url.pathname === '/v1/chat/completions') {
      let raw = '';
      req.on('data', chunk => { raw += chunk; });
      req.on('end', () => {
        requests.push({body: JSON.parse(raw), key: req.headers.authorization});
        res.writeHead(200, {'content-type': 'text/event-stream'});
        res.write('data: '+JSON.stringify({choices:[{delta:{content:'synthetic answer'},finish_reason:null}]})+'\n\n');
        const finish = () => { res.write('data: '+JSON.stringify({choices:[{delta:{},finish_reason:'stop'}]})+'\n\n'); res.end('data: [DONE]\n\n'); };
        if (requests.length === 1) releaseFirst = finish;
        else finish();
      });
      return;
    }
    const file = resolve(web, '.' + (url.pathname === '/' ? '/web/index.html' : url.pathname));
    if (!file.startsWith(web + require('node:path').sep) || !existsSync(file)) { res.writeHead(404); res.end(); return; }
    res.setHeader('content-type', {'.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml','.html':'text/html'}[extname(file)] || 'text/plain');
    res.end(readFileSync(file));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => server.close());
  const origin = `http://127.0.0.1:${server.address().port}`;
  const profile = mkdtempSync(join(tmpdir(), 'strata-memory-browser-'));
  const child = spawn(chrome, ['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
    '--remote-debugging-port=0',`--user-data-dir=${profile}`,origin], {stdio:'ignore', windowsHide:true});
  t.after(async () => {
    if (child.exitCode === null) {
      child.kill();
      await Promise.race([new Promise(done => child.once('exit', done)), pause(2000)]);
    }
    if (resolve(profile).startsWith(resolve(tmpdir()) + require('node:path').sep)) {
      try { rmSync(profile,{recursive:true,force:true,maxRetries:10,retryDelay:200}); }
      catch (error) { if (error.code !== 'EPERM') throw error; }
    }
  });
  async function until(fn) { for (let i=0;i<120;i++) { const value = await fn(); if (value) return value; await pause(50); } throw new Error('browser wait timed out'); }
  const portFile = join(profile, 'DevToolsActivePort');
  const debugPort = await until(() => existsSync(portFile) && Number(readFileSync(portFile,'utf8').split('\n')[0]));
  const target = await until(async () => (await (await fetch(`http://127.0.0.1:${debugPort}/json/list`)).json()).find(x => x.type === 'page' && x.url.startsWith(origin)));
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve,reject) => {ws.addEventListener('open',resolve,{once:true}); ws.addEventListener('error',reject,{once:true});});
  t.after(() => ws.close());
  let serial = 0;
  const pending = new Map();
  ws.addEventListener('message', event => { const data = JSON.parse(event.data); if (data.id && pending.has(data.id)) { pending.get(data.id)(data); pending.delete(data.id); } });
  async function evaluate(expression) {
    const id = ++serial;
    const response = new Promise(resolve => pending.set(id, resolve));
    ws.send(JSON.stringify({id,method:'Runtime.evaluate',params:{expression,awaitPromise:true,returnByValue:true}}));
    const result = await response;
    if (result.error || result.result?.exceptionDetails) throw new Error(JSON.stringify(result.error || result.result.exceptionDetails));
    return result.result.result.value;
  }
  await until(() => evaluate('typeof send === "function"'));
  await evaluate('localStorage.setItem("strata.apikey", JSON.stringify("ordinary-key")); localStorage.setItem("strata.chat", JSON.stringify([{role:"user",text:"ordinary private history",time:1}])); location.reload()');
  await until(() => evaluate('document.querySelector("#chat .st-msg")?.textContent.includes("ordinary private history")'));
  assert.equal(await evaluate('document.querySelector("#memory-read").checked || document.querySelector("#memory-store").checked'), false);
  await evaluate('window.pendingFileReads=[]; window.FileReader=class {readAsText(){window.pendingFileReads.push(this)} readAsDataURL(){window.pendingFileReads.push(this)}}; addFiles([{name:"synthetic.txt",type:"text/plain",size:4}])');
  assert.equal(await evaluate('window.pendingFileReads.length'), 1);
  await evaluate('document.querySelector("#input").value="ordinary unsent draft"');
  await evaluate('document.querySelector("#memory-read").click()');
  await evaluate('window.pendingFileReads[0].result="stale ordinary attachment"; window.pendingFileReads[0].onload()');
  assert.equal(await evaluate('document.querySelector("#attachments").children.length'), 0, 'late FileReader callback cannot cross into memory chat');
  assert.equal(await evaluate('document.querySelector("#attach-btn").disabled'), true);
  assert.equal(await evaluate('document.querySelectorAll("#chat .st-msg").length'), 0);
  assert.equal(await evaluate('document.querySelector("#input").value'), '');
  assert.equal(await evaluate('document.querySelector("#api-key").value'), '');
  await evaluate('document.querySelector("#api-key").value="profile-a"; document.querySelector("#api-key").dispatchEvent(new Event("input"))');
  assert.equal(await evaluate('document.querySelector("#memory-read").checked'), false, 'key change removes old consent');
  await evaluate('document.querySelector("#memory-read").click()');
  await evaluate('document.querySelector("#input").value="synthetic first"; document.querySelector("#composer").requestSubmit()');
  await until(() => requests.length === 1);
  assert.equal(requests[0].key, 'Bearer profile-a');
  assert.deepEqual(requests[0].body.messages.map(m => m.content), ['synthetic first']);
  assert.deepEqual([requests[0].body.strata_memory.read, requests[0].body.strata_memory.store_summary], [true,false]);
  assert.equal(requests[0].body.strata_mcp, undefined);
  assert.equal(await evaluate('document.querySelector("#api-key").disabled'), true);
  await evaluate('document.querySelector("#api-key").value="profile-b"; document.querySelector("#api-key").dispatchEvent(new Event("input"))');
  assert.equal(await evaluate('document.querySelector("#api-key").value'), 'profile-a');
  releaseFirst();
  await until(() => evaluate('!document.querySelector("#api-key").disabled'));
  assert.equal(await evaluate('document.querySelector("#memory-read").checked || document.querySelector("#memory-store").checked'), false);
  assert.equal(await evaluate('JSON.parse(localStorage.getItem("strata.apikey"))'), 'ordinary-key');
  await evaluate('document.querySelector("#input").value="without fresh consent"; document.querySelector("#composer").requestSubmit()');
  await pause(100);
  assert.equal(requests.length, 1);
  await evaluate('document.querySelector("#memory-store").click(); document.querySelector("#input").value="synthetic second"; document.querySelector("#composer").requestSubmit()');
  await until(() => requests.length === 2);
  assert.deepEqual(requests[1].body.messages.filter(m => m.role==='user').map(m => m.content), ['synthetic first','synthetic second']);
  assert.deepEqual([requests[1].body.strata_memory.read, requests[1].body.strata_memory.store_summary], [false,true]);
  assert.notEqual(requests[0].body.strata_memory.turn_id, requests[1].body.strata_memory.turn_id);
  await until(() => evaluate('!document.querySelector("#send-btn").disabled'));
  await evaluate('document.querySelector("#new-btn").click(); document.querySelector("#input").value="principal A unsent draft"; document.querySelector("#api-key").value="profile-b"; document.querySelector("#api-key").dispatchEvent(new Event("input")); document.querySelector(".st-toast:last-child button")?.click()');
  assert.equal(await evaluate('document.querySelectorAll("#chat .st-msg").length'), 0);
  assert.equal(await evaluate('document.querySelector("#input").value'), '', 'principal change discards prior draft');
  await evaluate('document.querySelector("#memory-read").click(); document.querySelector("#input").value="synthetic other principal"; document.querySelector("#composer").requestSubmit()');
  await until(() => requests.length === 3);
  assert.equal(requests[2].key, 'Bearer profile-b');
  assert.deepEqual(requests[2].body.messages.map(m => m.content), ['synthetic other principal']);
  await until(() => evaluate('!document.querySelector("#send-btn").disabled'));
  await evaluate('document.querySelector("#input").value="principal B unsent draft"; document.querySelector("#memory-exit").click()');
  assert.equal(await evaluate('document.querySelector("#input").value'), '', 'leaving memory chat discards its draft');
  assert.equal(await evaluate('document.querySelector("#chat .st-msg")?.textContent.includes("ordinary private history")'), true);
  await evaluate('document.querySelector("#memory-read").click()');
  assert.equal(await evaluate('document.querySelectorAll("#chat .st-msg").length'), 0, 're-entry cannot expose the previous principal');
  await evaluate('document.querySelector("#memory-exit").click()');
  await evaluate('document.querySelector("#input").value="ordinary follow-up"; document.querySelector("#composer").requestSubmit()');
  await until(() => requests.length === 4);
  assert.equal(requests[3].key, 'Bearer ordinary-key');
  assert.equal(requests[3].body.strata_memory, undefined);
  await until(() => evaluate('!document.querySelector("#send-btn").disabled'));
  await evaluate('document.querySelector("#new-btn").click(); document.querySelector("#memory-read").click(); document.querySelector(".st-toast:last-child button")?.click()');
  assert.equal(await evaluate('document.querySelectorAll("#chat .st-msg").length'), 0);
  await evaluate('document.querySelector("#memory-exit").click()');
  assert.equal(await evaluate('document.querySelectorAll("#chat .st-msg").length'), 0);
});
