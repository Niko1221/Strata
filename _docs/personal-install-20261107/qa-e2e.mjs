/** Prepared real packaged WebView2/Q36 E2E. No mocked API, engine or renderer. */
import assert from 'node:assert/strict';
import {spawn,spawnSync,execFileSync} from 'node:child_process';
import {readFileSync,writeFileSync,statSync,existsSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {createRequire} from 'node:module';
import {createServer} from 'node:net';
import {setTimeout as delay} from 'node:timers/promises';
import {sha256} from '../../app/host/scripts/release-integrity.mjs';
const root=resolve(import.meta.dirname,'../..'),out='H:/Strata/releases/personal-052-local2-20261107/app';
const args=process.argv.slice(2),python='C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe';
if(!args.includes('--execute')){console.log(JSON.stringify({wouldExecute:false,actualWebViewRequired:true,actualR3Required:true,context:4096,prefill:512,slots:5,mtp:2,privateOwner:true,releaseQualified:false}));process.exit(0);}
const commit=args[args.indexOf('--reviewed-commit')+1];assert.match(commit??'',/^[a-f0-9]{40}$/);
assert.equal(execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),commit);
assert.equal(execFileSync('git',['diff','HEAD','--name-only'],{cwd:root,encoding:'utf8'}).trim(),'');
assert.equal(execFileSync('git',['ls-files','--others','--exclude-standard'],{cwd:root,encoding:'utf8'}).trim(),'');
const fixture=JSON.parse(readFileSync(join(out,'e2e-fixture.json')));assert.equal(fixture.sourceCommit,commit);
const {home,app,modelId}=fixture,exe=join(app,'strata-desktop.exe');assert.equal(sha256(exe),fixture.shellSHA256);
assert.equal(existsSync(join(out,'functional-e2e.json')),false);
const modelPath=join(home,'install',`strata-${modelId}.json`);
const {chromium}=createRequire(join(root,'apps/host/package.json'))('playwright-core');
const result={ok:false,sourceCommit:commit,actualWebView2:false,actualNative:false,mocksUsed:false,releaseQualified:false,fullWindowQualified:false,checks:[],pageErrors:0};
let child,browser,page,base;
async function freePort(){const s=createServer();await new Promise(r=>s.listen(0,'127.0.0.1',r));const p=s.address().port;await new Promise(r=>s.close(r));return p;}
async function request(path,method='GET',body){const r=await fetch(base+path,{method,headers:{'content-type':'application/json'},...(body===undefined?{}:{body:JSON.stringify(body)}),signal:AbortSignal.timeout(60000)});assert.ok(r.ok,`QA endpoint failed (${r.status})`);return r.json();}
async function until(check,seconds=180){const end=Date.now()+seconds*1000;while(Date.now()<end){const found=await check();if(found)return found;await delay(500);}throw Error('Bounded E2E condition timed out');}
function ownership(pid){const run=spawnSync(python,['-B',join(import.meta.dirname,'probe-owned.py'),'--shell',String(child.pid),'--child',String(pid)],{encoding:'utf8',windowsHide:true,timeout:30000});assert.equal(run.status,0);const r=JSON.parse(run.stdout);assert.equal(r.owned,true);return r;}
async function start(){
 const probe=spawnSync(python,['-B',join(import.meta.dirname,'probe-owned.py')],{encoding:'utf8',windowsHide:true,timeout:15000});assert.equal(probe.status,0);const ready=JSON.parse(probe.stdout);
 assert.deepEqual(ready.processCounts,{shell:0,native:0,llama:0});assert.deepEqual(ready.resourceBlockers,[]);
 const port=await freePort();
 const env=Object.fromEntries(Object.entries(process.env).filter(([k])=>!/^(STRATA_|DSH_|WEBVIEW2_)/i.test(k)&&!/(?:API_KEY|TOKEN|SECRET|PASSWORD)$/i.test(k)));
 Object.assign(env,{STRATA_HOME:home,STRATA_INSTALL_DIR:join(home,'install'),STRATA_COMPONENTS_DIR:join(home,'components'),STRATA_PROFILE_DIR:join(home,'profiles'),STRATA_RUNTIME_DIR:join(home,'runtime'),STRATA_SETTINGS_FILE:join(home,'settings.json'),STRATA_MODELS_DIR:join(home,'models'),STRATA_WORKSPACE:join(home,'workspace'),STRATA_DESKTOP_DATA_DIR:join(home,'desktop'),STRATA_AGENT:'off',STRATA_REAL_ENGINE:'1',STRATA_NO_ACQUIRE:'1',STRATA_NO_BUNDLED_ENGINE:'1',STRATA_NO_REPAIR_ACCESS:'1',STRATA_NO_DIALOG:'1',DSH_HOME:join(home,'harness'),APPDATA:join(home,'Roaming'),LOCALAPPDATA:join(home,'Local'),TEMP:join(home,'temp'),TMP:join(home,'temp'),WEBVIEW2_USER_DATA_FOLDER:join(home,'webview'),WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS:'--remote-debugging-address=127.0.0.1 --remote-debugging-port='+port,PYTHONDONTWRITEBYTECODE:'1'});
 child=spawn(exe,[],{cwd:app,env,stdio:'ignore',windowsHide:true});
 await until(async()=>{assert.equal(child.exitCode,null);try{const r=await fetch(`http://127.0.0.1:${port}/json/version`,{signal:AbortSignal.timeout(1000)});return r.ok;}catch{return false;}},60);
 browser=await chromium.connectOverCDP(`http://127.0.0.1:${port}`);
 page=await until(()=>browser.contexts().flatMap(c=>c.pages()).find(p=>/^http:\/\/127\.0\.0\.1:\d+/.test(p.url())),60);
 page.setDefaultTimeout(20000);page.on('pageerror',()=>result.pageErrors++);base=new URL(page.url()).origin;
 await page.waitForLoadState('domcontentloaded');result.actualWebView2=true;
 assert.equal((await request('/api/app/build')).host,'0.5.2-local.2');
}
async function stop(){
 if(base){const state=await request('/api/native').catch(()=>null);for(const i of state?.instances??[])if(i.status==='ready')await request(`/api/native/instances/${i.id}/unload`,'POST',{});}
 await browser?.close().catch(()=>{});browser=null;
 if(child&&child.exitCode===null)child.kill();
 if(child)await until(()=>child.exitCode!==null||child.signalCode!==null,15);
 child=null;base=null;
}
async function language(value){await page.evaluate(language=>localStorage.setItem('strata.prefs.v2',JSON.stringify({...JSON.parse(localStorage.getItem('strata.prefs.v2')??'{}'),language})),value);await page.reload({waitUntil:'domcontentloaded'});}
async function modelScreen(){await page.goto(base+'/#/models');await page.locator('details.native-settings').waitFor();}
async function settingsJson(){const panel=page.locator('details.native-settings');await panel.evaluate(e=>e.open=true);await panel.getByRole('tab',{name:'JSON',exact:true}).click();return JSON.parse(await panel.getByRole('textbox',{name:'Native configuration JSON',exact:true}).inputValue());}
async function settingsMode(){const s=await request(`/api/native/models/${modelId}/settings`);assert.equal(s.values.q36_memory,'bounded-experts-v1');assert.ok(s.fields.some(f=>f.id==='q36_memory'&&f.supported!==false&&f.choices.includes('bounded-experts-v1')));return s;}
async function readyNative(){return until(async()=>{const state=await request('/api/native');const item=state.instances.find(i=>i.model===modelId&&i.status==='ready'&&i.q36Run?.observed?.status==='observed');return item??false;});}
function observed(item){
 const q=item.q36Run,info=item.metrics?.engine;assert.equal(q.backend.binarySha256,fixture.engineSHA256);
 for(const [key,value] of Object.entries({context:4096,prefill:512,mtp:2,parallel:1,kv_k:'f32',kv_v:'f32'}))assert.equal(q.observed[key],value);
 assert.equal(info.q36_memory_mode,'bounded-experts-v1');assert.equal(info.q36_memory_scope,'expert-working-storage');assert.equal(info.q36_memory_total_process_bound,0);
 assert.equal(info.q36_private_cache,2);assert.equal(info.prompt_cache,0);assert.equal(info.conversation_cache_mib,0);
 for(const key of ['q36_memory_cpu_stage_bytes','q36_memory_h2d_stage_bytes','q36_memory_io_scratch_bytes'])assert.ok(info[key]>0);
 assert.equal(q.effective.scopedConversationCache,true);ownership(q.observed.processId);result.actualNative=true;
 return {processId:q.observed.processId,generation:q.observed.generation,context:4096,prefill:512,mtp:2,kvK:'f32',kvV:'f32',privateOwnerProtocol:2};
}
const synthetic=(id,messages=[])=>({id,title:'Isolated personal candidate QA',createdAt:new Date().toISOString(),updatedAt:new Date().toISOString(),workspace:join(home,'workspace'),tokensIn:0,tokensOut:0,messages,preferences:{agent:false,useTools:false,thinking:'Off',thinkingBudget:0}});
try{
 await start();await language('en');await modelScreen();
 const initial=await request(`/api/native/models/${modelId}/settings`);assert.ok(initial.fields.some(f=>f.id==='q36_memory'&&f.supported!==false&&f.choices.includes('bounded-experts-v1')));
 const panel=page.locator('details.native-settings');await panel.evaluate(e=>e.open=true);
 await panel.getByRole('combobox',{name:'Q36 expert working storage',exact:true}).click();await page.getByRole('option',{name:/Bounded.*experimental|bounded-experts-v1/i}).click();
 await panel.getByRole('button',{name:'Save settings',exact:true}).click();await until(async()=>((await request(`/api/native/models/${modelId}/settings`)).values.q36_memory==='bounded-experts-v1'));
 await settingsMode();result.checks.push('real-r3-capability-and-native-settings-save');
 await page.getByRole('button',{name:/^Saved settings/}).click();await page.getByRole('menuitem',{name:'Save current settings…',exact:true}).click();await page.getByRole('dialog').getByRole('textbox',{name:'Name',exact:true}).fill('Personal bounded 4K QA');await page.getByRole('dialog').getByRole('button',{name:'Save',exact:true}).click();await page.getByRole('dialog').waitFor({state:'hidden'});
 const presets=await page.evaluate(()=>JSON.parse(localStorage.getItem('strata.native-presets.v1')));assert.equal(presets.at(-1).values.context,4096);
 await page.getByRole('combobox',{name:'Model version',exact:true}).click();await page.getByRole('option',{name:/UD-Q6_K_XL/}).click();
 await page.getByRole('combobox',{name:'Model version',exact:true}).click();await page.getByRole('option',{name:/UD-Q4_K_XL/}).click();
 await settingsMode();assert.equal((await settingsJson()).q36_memory,'bounded-experts-v1');result.checks.push('saved-native-settings-after-reselection');
 await page.getByRole('combobox',{name:'Conversation context',exact:true}).click();await page.getByRole('option',{name:'8K',exact:true}).click();
 await page.getByRole('button',{name:/^Saved settings/}).click();await page.getByRole('menuitem',{name:/Personal bounded 4K QA/}).click();assert.match(await page.getByRole('combobox',{name:'Conversation context',exact:true}).innerText(),/4K/);
 await stop();await start();await language('en');await modelScreen();assert.deepEqual(await page.evaluate(()=>JSON.parse(localStorage.getItem('strata.native-presets.v1'))),presets);await settingsMode();assert.equal((await settingsJson()).q36_memory,'bounded-experts-v1');result.checks.push('saved-profile-after-process-restart');
 await page.getByRole('button',{name:'Load model',exact:true}).click();let live=await readyNative();const first=observed(live);result.checks.push('gui-load-actual-info-private-owner');
 const configHash=sha256(modelPath),before=await request('/api/native');
 const invalid=await fetch(base+'/api/native/operations',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({kind:'load',args:{model:modelId,values:{q36_memory:'invalid-qa-mode'}}}),signal:AbortSignal.timeout(30000)});
 assert.ok([400,422].includes(invalid.status));const after=await request('/api/native');assert.equal(after.jobs.length,before.jobs.length);assert.equal(sha256(modelPath),configHash);live=after.instances.find(i=>i.id===live.id);assert.equal(live.status,'ready');assert.equal(live.q36Run.observed.processId,first.processId);ownership(first.processId);result.checks.push('invalid-choice-rejected-before-unload-or-job');
 const chatId='personal-qa-chat';await request('/api/chats/import','POST',{sessions:[synthetic(chatId)]});await page.goto(base+'/#/chat/'+chatId);await page.reload();
 await page.getByRole('textbox',{name:'Message Strata',exact:true}).fill('Reply with just the word Paris.');await page.getByRole('textbox',{name:'Message Strata',exact:true}).press('Enter');
 const reply=await until(async()=>{const {session}=await request('/api/chats/'+chatId);return session.messages.find(m=>m.role==='assistant'&&m.content?.trim()&&!m.streaming);});assert.ok(reply.content.trim().length>0);assert.notEqual(reply.demo,true);assert.ok(Number(reply.tokens??0)<=32);result.checks.push('real-gui-nonempty-chat');
 await modelScreen();await page.locator('.native-model-actions').getByRole('button',{name:'Unload',exact:true}).click();await until(async()=>!(await request('/api/native')).instances.some(i=>i.status==='ready'));
 await page.getByRole('button',{name:'Load model',exact:true}).click();live=await readyNative();const restarted=observed(live);assert.notEqual(restarted.processId,first.processId);result.checks.push('gui-stop-restart-new-owned-child');
 await modelScreen();await page.locator('.native-model-actions').getByRole('button',{name:'Unload',exact:true}).click();await until(async()=>!(await request('/api/native')).instances.some(i=>i.status==='ready'));
 const notices=['The reply was cut off. The partial answer is kept.','Response interrupted. Ask again to retry; the partial answer is kept.'];
 for(let index=0;index<notices.length;index++){
  const id='personal-qa-recovery-'+index;await request('/api/chats/import','POST',{sessions:[synthetic(id,[{id:'reply-'+index,role:'assistant',content:'Synthetic partial answer for rendering QA.',at:new Date().toISOString(),tps:12.4,tokens:8,error:notices[index]}])]});
  await page.goto(base+'/#/chat/'+id);await language('ja');await page.locator('.telemetry__chip').first().waitFor();
  const chip=await page.locator('.telemetry__chip').first().innerText();assert.doesNotMatch(chip,/\{\d+\}|\u0001|last reply/);assert.match(chip,/12/);
  const message=await page.locator('.st-msg--assistant').first().innerText();assert.ok(!message.includes(notices[index]));assert.match(message,/回答|応答/);
  await page.screenshot({path:join(home,'screenshots',`japanese-recovery-${index}.png`),fullPage:true});
 }
 result.checks.push('japanese-numeric-speed-and-both-recovery-notices');
 for(const [path,metadata] of Object.entries(fixture.modelFilesBefore)){const now=statSync(path,{bigint:true});assert.equal(Number(now.size),metadata.bytes);assert.equal(Number(now.mtimeNs),metadata.mtimeNs);}
 assert.equal(result.pageErrors,0);result.modelFilesPreserved=true;result.ok=true;
}catch(error){result.errorType=error.name;result.errorCode=error.code??null;process.exitCode=1;}
finally{
 try{await stop();result.ownedCleanup=true;}catch{result.ownedCleanup=false;result.ok=false;process.exitCode=1;if(child&&child.exitCode===null)child.kill();}
 writeFileSync(join(out,'functional-e2e.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(result));
}
