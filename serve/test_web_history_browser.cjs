// Optional real-browser history regressions; serves only a local mock SSE server, with external requests blocked.
// Install puppeteer-core as a development tool, set CHROME_BIN to Chromium, then run from the repository root:
//   WEBCHAT_BASELINE_REF=<tested-main-commit> node serve/test_web_history_browser.cjs
// WEBCHAT_ARTIFACT_DIR optionally selects a directory for screenshots and the JSON result (otherwise temporary).
// No Python server, model, GPU, or runtime dependency is needed. The normal unittest suite does not require this.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const {execFileSync} = require('node:child_process');
const puppeteer = require('puppeteer-core');
const repo = process.env.WEBCHAT_REPO || path.resolve(__dirname, '..');
const baselineRepo = process.env.WEBCHAT_BASELINE_REPO || repo;
const baselineRef = process.env.WEBCHAT_BASELINE_REF || 'origin/main';
const artifacts = process.env.WEBCHAT_ARTIFACT_DIR || fs.mkdtempSync(path.join(require('node:os').tmpdir(), 'strata-history-'));
fs.mkdirSync(artifacts, {recursive:true});
if (!process.env.CHROME_BIN) throw new Error('Set CHROME_BIN to a Chromium executable.');
const requests = [], errors = [];
let hold = false, aborted = 0;
const rich = [
  {role:'user', text:'Explain this example', time:1000, images:[{name:'tiny.png',url:'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aC1kAAAAASUVORK5CYII='}], files:[{name:'example.txt',text:'original attachment'}]},
  {role:'assistant',text:'Repeated wrong answer',reasoning:'old reasoning',time:2000,error:'old error',stopped:true,thinkSecs:3,limit:4,tools:[{id:'tool1',round:0,name:'mock.tool',at:0,result:'old tool result',state:'done',arguments:{x:1}}]},
  {role:'user',text:'Later question',time:3000},
  {role:'assistant',text:'Later answer',reasoning:'later thoughts',time:4000},
];
const mime = {'.html':'text/html','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml','.woff2':'font/woff2'};
const server = http.createServer(async (req,res) => {
  let route = new URL(req.url,'http://localhost').pathname;
  const baseline = route.startsWith('/baseline/');
  route = route.replace(/^\/baseline(?=\/)/,'');
  const json = value => {res.setHeader('Content-Type','application/json');res.end(JSON.stringify(value));};
  if(route === '/health') return json({loaded:true,model:'mock-strata',images:true});
  if(route === '/metrics') return json({live:{state:'idle'},engine:{},requests:[]});
  if(route === '/mcp') return json({tools:0,servers:[]});
  if(route === '/settings') return json({shared:false});
  if(route === '/config') return json({keys:[]});
  if(route === '/v1/chat/completions') {
    let body=''; for await (const part of req) body += part;
    requests.push(JSON.parse(body));
    res.writeHead(200,{'Content-Type':'text/event-stream','Cache-Control':'no-cache'});
    const frame = value => res.write('data: '+JSON.stringify(value)+'\n\n');
    frame({choices:[{delta:{content:'Mock answer'}}]});
    if(hold) { res.on('close',()=>aborted++); return; }
    frame({choices:[{delta:{},finish_reason:'stop'}],usage:{completion_tokens:2}});
    res.end('data: [DONE]\n\n'); return;
  }
  const relative = route === '/' ? 'serve/web/index.html' : 'serve/'+route.slice(1);
  if(!relative.startsWith('serve/web/')) {res.writeHead(404);res.end();return;}
  try {
    const data = baseline ? execFileSync('git',['show',baselineRef+':'+relative],{cwd:baselineRepo}) : fs.readFileSync(path.join(repo,relative));
    res.setHeader('Content-Type',mime[path.extname(relative)] || 'application/octet-stream');res.end(data);
  } catch {res.writeHead(404);res.end();}
});
(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const url='http://127.0.0.1:'+server.address().port;
  const browser=await puppeteer.launch({executablePath:process.env.CHROME_BIN,headless:true,args:['--no-sandbox']});
  try {
    const newPage=async(baseline=false,enabled=false)=>{
      const context=await browser.createBrowserContext(),page=await context.newPage();
      await page.setViewport({width:1100,height:850});
      page.on('pageerror',e=>{errors.push(e.message);console.error('PAGE ERROR',e.message);});
      await page.setRequestInterception(true);
      page.on('request',r=>r.url().startsWith(url) || r.url().startsWith('data:') ? r.continue() : r.abort());
      await page.evaluateOnNewDocument((seed,on)=>{
        localStorage.setItem('strata.chat',JSON.stringify(seed));
        if(on!==null) localStorage.setItem('strata.editHistory',JSON.stringify(on));
      },rich,enabled);
      await page.goto(url+(baseline?'/baseline/':'/'),{waitUntil:'networkidle0'});
      return page;
    };
    const idle=page=>page.waitForFunction(()=>busy===null);
    const clickAction=async(page,i,label)=>page.evaluate((n,text)=>{
      const buttons=[...document.querySelector(`[data-i="${n}"] .history-actions`).querySelectorAll('button')];
      buttons.find(b=>b.textContent===text).click();
    },i,label);
    const reset=async page=>page.evaluate(seed=>{messages=seed;historyUndo=null;saveChat();renderChat();renderHistoryUndo();},rich);
    const probe=await newPage(false,null);
    const expectedDefault=process.env.WEBCHAT_HISTORY_DEFAULT==='true';
    assert.equal(await probe.evaluate(()=>historyEditing),expectedDefault);
    assert.equal(await probe.$$eval('.history-actions',els=>els.length),expectedDefault?4:0);
    assert.equal(await probe.$eval('#sampling-btn',e=>e.getAttribute('aria-label')),'Chat settings');
    for(const width of [320,390]) {
      await probe.setViewport({width,height:844});
      assert(await probe.$eval('#sampling-btn',e=>{const r=e.getBoundingClientRect();return r.left>=0&&r.right<=innerWidth;}));
      assert(await probe.$eval('#send-btn',e=>{const r=e.getBoundingClientRect();return r.left>=0&&r.right<=innerWidth;}));
    }
    await probe.locator('#sampling-btn').click();
    assert.equal(await probe.$eval('#drawer .card-title',e=>e.textContent),'Chat settings');
    if(expectedDefault) await probe.locator('#s-edit-history').click();
    await probe.locator('#s-apply').click();
    await probe.reload({waitUntil:'networkidle0'});
    assert.equal(await probe.evaluate(()=>historyEditing),false,'explicit off preference must survive reload');
    await probe.locator('#sampling-btn').click();await probe.locator('#s-reset').click();await probe.locator('#s-apply').click();
    assert.equal(await probe.evaluate(()=>historyEditing),expectedDefault,'Reset must use this deployment default');
    console.log('PASS visible Chat settings at 320/390px; deployment default, explicit off and Reset');
    const original=await newPage(true),off=await newPage();
    assert.equal(await off.$eval('#chat',e=>e.innerHTML),await original.$eval('#chat',e=>e.innerHTML));
    assert.equal(await off.$('.history-actions'),null);
    for(const page of [original,off]) {await page.type('#input','Same default request');await page.click('#send-btn');await idle(page);}
    assert.deepEqual(requests[0],requests[1]);
    console.log('PASS default-off message DOM and API payload match upstream');
    const page=await newPage();
    await page.locator('#sampling-btn').click();await page.locator('#s-edit-history').click();await page.locator('#s-apply').click();
    await page.waitForSelector('.st-toast',{hidden:true});
    assert.equal(await page.$$eval('.history-actions',els=>els.length),4);
    assert.equal(await page.evaluate(()=>localStorage.getItem('strata.editHistory')),'true');
    await page.evaluate(()=>{document.querySelector('#input').value='Unsent draft';attachments=[{kind:'file',name:'draft.txt',text:'draft attachment'}];renderAttachments();});
    await clickAction(page,0,'Edit');
    await page.$eval('.history-editor textarea',e=>e.value='改写：<img src=x onerror="window.bad=1">');
    await page.click('[data-edit-save]');await idle(page);
    const edited=requests.at(-1);
    assert.equal(edited.messages.length,1);
    assert.match(edited.messages[0].content[0].text,/original attachment/);
    assert.match(edited.messages[0].content[0].text,/改写/);
    assert.equal(edited.messages[0].content[1].image_url.url,rich[0].images[0].url);
    assert.equal(await page.evaluate(()=>messages.length),2);
    assert.equal(await page.$eval('#input',e=>e.value),'Unsent draft');
    assert.equal(await page.evaluate(()=>attachments[0].text),'draft attachment');
    assert.equal(await page.evaluate(()=>window.bad),undefined);
    await page.click('#history-undo button');
    assert.deepEqual(await page.evaluate(()=>messages),rich);
    console.log('PASS user edit truncates history, keeps attachments/draft, regenerates, and Undo restores');
    await clickAction(page,1,'Edit');
    const before=requests.length;
    await page.$eval('.history-editor textarea',e=>e.value='Corrected assistant answer');
    await page.click('[data-edit-save]');
    assert.equal(requests.length,before);
    assert.deepEqual(await page.evaluate(()=>messages[1]),{role:'assistant',text:'Corrected assistant answer',time:2000,meta:'Edited'});
    assert.equal(await page.evaluate(()=>messages.length),2);
    await page.focus('#input');await page.keyboard.press('Enter');
    await page.waitForFunction(()=>messages.length===4 && busy===null);
    assert.equal(requests.at(-1).messages.length,3);
    assert.equal(requests.at(-1).messages[1].content,'Corrected assistant answer');
    assert(!JSON.stringify(requests.at(-1)).includes('old tool result'));
    assert.equal(await page.$('#history-undo'),null);
    console.log('PASS assistant edit removes stale tools/reasoning; next request uses corrected text');
    await reset(page);
    await clickAction(page,1,'Regenerate');await idle(page);
    assert.equal(requests.at(-1).messages.length,1);
    assert.equal(await page.evaluate(()=>messages.length),2);
    await page.click('#history-undo button');
    assert.equal(await page.evaluate(()=>messages.length),4);
    await clickAction(page,2,'Remove from here');
    assert.equal(await page.evaluate(()=>messages.length),2);
    await page.click('#history-undo button');
    await clickAction(page,1,'Remove from here');
    assert.equal(await page.evaluate(()=>messages.length),1);
    await clickAction(page,0,'Regenerate');await idle(page);
    assert.equal(requests.at(-1).messages.length,1);
    console.log('PASS regenerate and suffix removal preserve complete turns, without duplicating users');
    await reset(page);hold=true;
    await page.$eval('#input',e=>e.value='Streaming stop test');await page.locator('#send-btn').click();
    await page.waitForFunction(()=>busy && busy.msg.text==='Mock answer',{timeout:4000});
    await page.setViewport({width:320,height:844});
    assert(await page.$$eval('#sampling-btn,#stop-btn,#send-btn',els=>els.every(e=>{const r=e.getBoundingClientRect();return r.left>=0&&r.right<=innerWidth;})));
    assert(await page.$$eval('.history-actions button',els=>els.every(e=>e.disabled)));
    const stopped=await page.evaluate(()=>{const n=messages.length;document.querySelector('#stop-btn').click();return {n,removed:removeHistoryFrom(0),after:messages.length};});
    assert.equal(stopped.removed,false);assert.equal(stopped.n,stopped.after);
    await idle(page);hold=false;
    await clickAction(page,stopped.n-1,'Edit');
    await page.$eval('.history-editor textarea',e=>e.value='Recovered after stopping');
    await page.click('[data-edit-save]');
    await new Promise(resolve=>setTimeout(resolve,80));
    assert.equal(await page.evaluate(()=>messages.at(-1).text),'Recovered after stopping');
    assert(aborted>=1);
    console.log('PASS Stop finalization prevents history races; stopped answer can then be edited');
    await page.setViewport({width:390,height:844});
    await clickAction(page,0,'Edit');
    assert.equal(await page.$eval('#send-btn',e=>e.disabled),true);
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await page.screenshot({path:path.join(artifacts,'mobile-edit.png'),fullPage:true});
    await page.click('[data-edit-cancel]');
    await page.setViewport({width:1100,height:850});
    await page.screenshot({path:path.join(artifacts,'desktop-history.png'),fullPage:true});
    await page.click('#new-btn');
    assert.equal(await page.$('#history-undo'),null);
    assert.equal(await page.evaluate(()=>undoHistory()),false);
    assert.equal(errors.length,0,errors.join('\n'));
    console.log('PASS mobile editor, keyboard-safe composer, New chat invalidation; no page errors');
    fs.writeFileSync(path.join(artifacts,'browser-results.json'),JSON.stringify({baseline:baselineRef,browser:await browser.version(),passed:7,history_default:expectedDefault,settings_label:'Chat settings',mobile_widths:[320,390],requests:requests.length,aborted,real_model_requests:0,errors},null,2)+'\n');
  } finally {await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});
