// Browser -> actual Python HTTP handler -> local MockEngine -> installed Hakua.
// All credentials and profile contents are synthetic; no GPU is started.
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {existsSync, mkdtempSync, readFileSync, rmSync} = require('node:fs');
const {join, resolve, sep} = require('node:path');
const {tmpdir} = require('node:os');
const {spawn} = require('node:child_process');
const {setTimeout: pause} = require('node:timers/promises');

test('real Python chat page reaches real Hakua writer with two synthetic principals', {timeout:30000}, async t => {
  const chrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
  if (!existsSync(chrome)) return t.skip('Chrome unavailable');
  const root = resolve(__dirname, '../..');
  const py = spawn('py', ['-3','-u','-m','serve.test_memory_browser_fixture'],
    {cwd:root,stdio:['pipe','pipe','pipe'],windowsHide:true});
  let port = 0, counts = null, stderr = '', stdout = '';
  py.stdout.setEncoding('utf8');
  py.stdout.on('data', chunk => {
    stdout += chunk;
    for (const line of stdout.split(/\r?\n/)) {
      if (line.startsWith('PORT ')) port = Number(line.slice(5));
      if (line.startsWith('COUNTS ')) counts = JSON.parse(line.slice(7));
    }
  });
  py.stderr.setEncoding('utf8'); py.stderr.on('data', chunk => {stderr += chunk;});
  t.after(async () => {
    if (py.exitCode === null) {py.stdin.end('\n'); await Promise.race([new Promise(done => py.once('exit',done)),pause(3000)]);}
    if (py.exitCode === null) py.kill();
  });
  async function until(fn) {
    for(let i=0;i<160;i++) {const value=await fn(); if(value)return value; if(py.exitCode!==null)throw new Error(`Python fixture exited: ${stderr.slice(-500)}`); await pause(50);}
    throw new Error(`timed out: ${stderr.slice(-500)}`);
  }
  await until(()=>port);
  const origin=`http://127.0.0.1:${port}`;
  const health=await (await fetch(origin+'/health')).json();
  assert.equal(health.service,'strata');
  const profile=mkdtempSync(join(tmpdir(),'strata-memory-python-browser-'));
  const child=spawn(chrome,['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
    '--remote-debugging-port=0',`--user-data-dir=${profile}`,origin],{stdio:'ignore',windowsHide:true});
  t.after(async()=>{
    if(child.exitCode===null){child.kill();await Promise.race([new Promise(done=>child.once('exit',done)),pause(2000)]);}
    if(resolve(profile).startsWith(resolve(tmpdir())+sep)){
      try{rmSync(profile,{recursive:true,force:true,maxRetries:10,retryDelay:200});}catch(error){if(error.code!=='EPERM')throw error;}
    }
  });
  const portFile=join(profile,'DevToolsActivePort');
  const debugPort=await until(()=>existsSync(portFile)&&Number(readFileSync(portFile,'utf8').split('\n')[0]));
  const target=await until(async()=>{
    const tabs=await(await fetch(`http://127.0.0.1:${debugPort}/json/list`)).json();
    return tabs.find(x=>x.type==='page'&&x.url.startsWith(origin));
  });
  const ws=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((done,fail)=>{ws.addEventListener('open',done,{once:true});ws.addEventListener('error',fail,{once:true});});
  t.after(()=>ws.close());
  let serial=0;const pending=new Map();
  ws.addEventListener('message',event=>{const data=JSON.parse(event.data);if(data.id&&pending.has(data.id)){pending.get(data.id)(data);pending.delete(data.id);}});
  async function evaluate(expression){
    const id=++serial,result=new Promise(done=>pending.set(id,done));
    ws.send(JSON.stringify({id,method:'Runtime.evaluate',params:{expression,awaitPromise:true,returnByValue:true}}));
    const value=await result;
    if(value.error||value.result?.exceptionDetails)throw new Error(JSON.stringify(value.error||value.result.exceptionDetails));
    return value.result.result.value;
  }
  await until(()=>evaluate('typeof send === "function"'));
  await evaluate('settings.thinking="none"'); // MockEngine emits plain text only in this mode.
  assert.equal(await evaluate('document.querySelector("#memory-read").checked'),false);
  await evaluate('document.querySelector("#memory-read").click();document.querySelector("#api-key").value="synthetic-browser-a";document.querySelector("#api-key").dispatchEvent(new Event("input"));document.querySelector("#memory-read").click();document.querySelector("#memory-store").click();document.querySelector("#input").value="synthetic A request";document.querySelector("#composer").requestSubmit()');
  try {await until(()=>evaluate('document.querySelector("#chat .st-msg--assistant .st-bubble")?.textContent.includes("synthetic visible answer") && !document.querySelector("#send-btn").disabled'));}
  catch(error){throw new Error(`${error.message}; page=${await evaluate('document.querySelector("#chat")?.textContent.slice(-500)')}; log=${stdout.slice(-600)}; err=${stderr.slice(-300)}`);}
  assert.equal(await evaluate('document.querySelector("#memory-read").checked||document.querySelector("#memory-store").checked'),false);
  await evaluate('document.querySelector("#api-key").value="synthetic-browser-b";document.querySelector("#api-key").dispatchEvent(new Event("input"));document.querySelector("#memory-read").click();document.querySelector("#input").value="synthetic B request";document.querySelector("#composer").requestSubmit()');
  await until(()=>evaluate('document.querySelector("#chat .st-msg--assistant .st-bubble")?.textContent.includes("synthetic visible answer") && !document.querySelector("#send-btn").disabled'));
  py.stdin.end('\n');
  await until(()=>counts);
  assert.deepEqual(counts,{a:1,b:0});
});
