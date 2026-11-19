/** Real browser checks for startup recovery and per-turn memory consent. No model inference is mocked as a model pass. */
import assert from 'node:assert/strict';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {resolve,join} from 'node:path';
import {createServer} from 'vite';
import {chromium} from 'playwright-core';
import {findBrowser} from '../server/agent/browser-probe.mjs';

const root=resolve(import.meta.dirname,'..');
const output=process.argv[2];
if(!output)throw new Error('Usage: node scripts/test-hakua-ui.mjs <evidence-directory>');
mkdirSync(output,{recursive:true});
const version=JSON.parse(readFileSync(join(root,'package.json'),'utf8')).version;
const found=findBrowser();
if(!found.found)throw new Error('No supported installed browser found');
process.env.STRATA_PUBLIC_PACKAGE='1';
const fixtureId='\0hakua-ui-fixture.tsx';
const vite=await createServer({root,server:{host:'127.0.0.1',port:0,strictPort:false},plugins:[{
  name:'hakua-ui-test-fixture',
  resolveId(id){if(id==='/__hakua-ui-entry.tsx')return fixtureId;},
  load(id){if(id===fixtureId)return `
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {HakuaLlamaChat} from '/src/components/HakuaLlamaChat.tsx';
    import {setTranslationLanguage,validLanguage} from '/src/lib/i18n.ts';
    import '/src/styles/kit/tokens.css'; import '/src/styles/kit/components.css'; import '/src/styles/app.css';
    const language=new URL(location.href).searchParams.get('lang');
     if(validLanguage(language))await setTranslationLanguage(language);
    const view=createRoot(document.getElementById('root'));
    view.render(React.createElement(HakuaLlamaChat,{
      access:{profileKey:'synthetic-ui-principal',runtimeId:'synthetic-runtime'},
      onClose:()=>{view.unmount();document.getElementById('root').textContent='Closed';}
    }));`;},
}]});
let browser;
const results=[];
const rendererErrors=[];
try {
  await vite.listen();
  const base='http://127.0.0.1:'+vite.httpServer.address().port;
  const env={...process.env};
  for(const key of Object.keys(env))if(/(?:API_KEY|TOKEN|SECRET|PASSWORD|ATTEST_KEY)$/.test(key))delete env[key];
  browser=await chromium.launch({headless:true,executablePath:found.path,chromiumSandbox:true,env});
  async function pageFor() {
    const context=await browser.newContext({viewport:{width:1440,height:920}});
    await context.route('**/*',route=>new URL(route.request().url()).origin===base?route.continue():route.abort());
    const page=await context.newPage();
    page.on('pageerror',error=>{rendererErrors.push(error.message);console.error('Renderer error: '+error.message);});
    return {context,page};
  }
  for(const [name,identity] of [
    ['matched',{host:version,shell:version}],
    ['stale-host',{host:'0.0.1',shell:version}],
    ['stale-shell',{host:version,shell:'0.0.1'}],
    ['offline',null],
  ]) {
    const {context,page}=await pageFor();
    await page.route('**/api/**',route=>route.fulfill({status:404,json:{error:'Synthetic UI fixture: route unavailable'}}));
    await page.route('**/api/app/build',route=>identity?route.fulfill({json:identity}):route.abort());
    await page.goto(base,{waitUntil:'domcontentloaded'});
    if(name==='matched')await page.locator('.app').waitFor();
    else {
      await page.getByRole('heading',{name:'Strata could not start',exact:true}).waitFor();
      assert.equal(await page.locator('.app').count(),0);
      assert.equal(await page.getByRole('button',{name:'Reload',exact:true}).isEnabled(),true);
    }
    await page.screenshot({path:join(output,name+'.png')});
    results.push({name,passed:true});
    await context.close();
  }
  {
    const {context,page}=await pageFor();
    await page.route('**/src/main.tsx',route=>route.abort());
    await page.goto(base,{waitUntil:'domcontentloaded'});
    await page.getByRole('heading',{name:'Strata could not start',exact:true}).waitFor({timeout:25000});
    assert.equal(await page.getByRole('button',{name:'Reload'}).isVisible(),true);
    await page.screenshot({path:join(output,'missing-module.png')});
    results.push({name:'missing-module',passed:true});
    await context.close();
  }
  {
    const {context,page}=await pageFor(),requests=[];
    let hold=false,release=null,outcome='stored';
    const html=await vite.transformIndexHtml('/__hakua-ui-test.html',`<!doctype html><html><body><div id="root"></div>
      <script type="module" src="/__hakua-ui-entry.tsx"></script></body></html>`);
    await page.route('**/__hakua-ui-test.html',route=>route.fulfill({contentType:'text/html',body:html}));
    await page.route('**/api/hakua/chat',route=>{
      requests.push({body:route.request().postDataJSON(),authorization:route.request().headers().authorization});
      if(hold)return new Promise(resolve=>{release=()=>{void route.abort().catch(()=>{}).finally(resolve);};});
      return route.fulfill({contentType:'application/x-ndjson',body:
        JSON.stringify({event:{choices:[{delta:{content:'Synthetic answer for UI verification.'}}]}})+'\n'+
        JSON.stringify({done:true,memory:outcome})+'\n'});
    });
    await page.goto(base+'/__hakua-ui-test.html',{waitUntil:'domcontentloaded'});
    const checkboxes=page.getByRole('checkbox');
    await checkboxes.first().waitFor();
    assert.equal(await checkboxes.nth(0).isChecked(),false);
    assert.equal(await checkboxes.nth(1).isChecked(),false);
    await checkboxes.nth(0).check();
    await checkboxes.nth(1).check();
    await page.getByRole('textbox',{name:'Message',exact:true}).fill('Synthetic consent verification.');
    await page.getByRole('button',{name:'Send',exact:true}).click();
    await page.getByText('Summary saved to Hakua-memory.',{exact:true}).waitFor();
    assert.equal(requests.length,1);
    assert.equal(requests[0].authorization,'Bearer synthetic-ui-principal');
    assert.equal(requests[0].body.strata_memory.read,true);
    assert.equal(requests[0].body.strata_memory.store_summary,true);
    assert.equal(await checkboxes.nth(0).isChecked(),false);
    assert.equal(await checkboxes.nth(1).isChecked(),false);
    for(const width of [320,768,1024,1440]) {
      await page.setViewportSize({width,height:920});
      assert.equal(await page.getByRole('textbox',{name:'Message',exact:true}).isVisible(),true);
      assert.equal(await page.getByRole('button',{name:'Send',exact:true}).isVisible(),true);
      await page.screenshot({path:join(output,'memory-'+width+'.png')});
    }
    const storage=await page.evaluate(()=>[...Object.values(localStorage),...Object.values(sessionStorage)].join('\n'));
    assert.equal(storage.includes('Synthetic consent verification.'),false);
    assert.equal(storage.includes('synthetic-ui-principal'),false);
    outcome='no-summary';
    await checkboxes.nth(1).check();
    await page.getByRole('textbox',{name:'Message',exact:true}).fill('Synthetic empty summary verification.');
    await page.getByRole('button',{name:'Send',exact:true}).click();
    await page.getByText('The model did not return a useful summary. No memory was saved.',{exact:true}).waitFor();
    assert.equal(requests[1].body.strata_memory.read,false);
    assert.equal(requests[1].body.strata_memory.store_summary,true);
    hold=true;
    await page.getByRole('textbox',{name:'Message',exact:true}).fill('Synthetic cancellation verification.');
    const pendingRequest=page.waitForRequest(request=>request.url().endsWith('/api/hakua/chat'));
    await page.getByRole('button',{name:'Send',exact:true}).click();
    await pendingRequest;
    await page.getByRole('button',{name:'Stop',exact:true}).click();
    await page.getByText('Stopped. A completed summary may already have been saved.',{exact:true}).waitFor();
    release?.();
    assert.equal(requests[2].body.strata_memory.read,false);
    assert.equal(requests[2].body.strata_memory.store_summary,false);
    await page.getByRole('button',{name:'Close',exact:true}).click();
    await page.getByText('Closed',{exact:true}).waitFor();
    results.push({name:'memory-consent-reset-cancel-and-private-dialog',passed:true,widths:[320,768,1024,1440]});
    await context.close();
    const japanese=await pageFor();
    await japanese.page.route('**/__hakua-ui-test.html*',route=>route.fulfill({contentType:'text/html',body:html}));
    await japanese.page.goto(base+'/__hakua-ui-test.html?lang=ja',{waitUntil:'domcontentloaded'});
    const readConsent=japanese.page.getByRole('checkbox',{name:'この返答のために保存済みの記憶を参照する',exact:true});
    const saveConsent=japanese.page.getByRole('checkbox',{name:'この返答が完了したら要約を保存する',exact:true});
    await readConsent.waitFor();
    assert.equal(await readConsent.isChecked(),false);
    assert.equal(await saveConsent.isChecked(),false);
    await japanese.page.screenshot({path:join(output,'memory-ja.png')});
    results.push({name:'japanese-memory-consent-labels',passed:true});
    await japanese.context.close();
  }
  assert.deepEqual(rendererErrors,[],'No JavaScript renderer errors');
  writeFileSync(join(output,'results.json'),JSON.stringify({browser:browser.version(),results,rendererErrors},null,2));
  console.log(JSON.stringify(results));
} catch(error) {
  writeFileSync(join(output,'results.json'),JSON.stringify({results,error:error.stack},null,2));
  throw error;
} finally {await browser?.close();await vite.close();}
