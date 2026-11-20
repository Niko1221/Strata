/** Prepared actual packaged GUI test for the reported GGUF; never operates on the normal profile. */
import assert from 'node:assert/strict';
import {spawn,spawnSync,execFileSync} from 'node:child_process';
import {readFileSync,writeFileSync,mkdirSync,cpSync,existsSync,statSync,createReadStream} from 'node:fs';
import {createHash} from 'node:crypto';
import {join,resolve} from 'node:path';
import {createRequire} from 'node:module';
import {createServer} from 'node:net';
import {setTimeout as delay} from 'node:timers/promises';
import {sha256,verifyEngine} from '../../app/host/scripts/release-integrity.mjs';
import {findLocalBackend,verifyLocalBackend} from '../../app/host/server/local-backends.mjs';
const root=resolve(import.meta.dirname,'../..'),out='H:/Strata/releases/personal-052-local2-20261107/app';
const args=process.argv.slice(2),py='C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe';
if(!args.includes('--execute')){console.log(JSON.stringify({wouldExecute:false,actualGuiRequired:true,prismVersion:'0.4.1',context:98304,q8Kv:true,flashAttention:true,userProfileMutation:false,fullWindowQualified:false}));process.exit(0);}
const commit=args[args.indexOf('--reviewed-commit')+1];assert.match(commit??'',/^[0-9a-f]{40}$/);
assert.equal(execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),commit);
assert.equal(execFileSync('git',['diff','HEAD','--name-only'],{cwd:root,encoding:'utf8'}).trim(),'');
assert.equal(execFileSync('git',['ls-files','--others','--exclude-standard'],{cwd:root,encoding:'utf8'}).trim(),'');
const qa=JSON.parse(readFileSync(join(out,'app-qualification.json'))),native=JSON.parse(readFileSync(join(out,'staged-native.json')));
assert.equal(qa.ok,true);assert.equal(qa.sourceCommit,commit);assert.equal(native.prismVersion,'0.4.1');assert.equal(native.signatureVerified,true);
const selection=JSON.parse(readFileSync(join(out,'prism041-r4/selection-plan.json')));assert.equal(selection.applied,false);
assert.equal(sha256(selection.privatePlan),selection.privatePlanSHA256);
const plan=JSON.parse(readFileSync(selection.privatePlan)),model=plan.target.path;
const hashFile=async path=>{const h=createHash('sha256');for await(const b of createReadStream(path))h.update(b);return h.digest('hex');};
assert.equal(await hashFile(model),'8cf793a36e62fb078ac05c2e1bdb2121ff94736dccbede7cf81b76346e868949');
const before=statSync(model,{bigint:true}),home=join(out,'prism-e2e-profile');assert.equal(existsSync(home),false);assert.equal(existsSync(join(out,'prism-e2e.json')),false);
const pre=spawnSync(py,['-B',join(import.meta.dirname,'probe-owned.py')],{encoding:'utf8',windowsHide:true,timeout:15000});assert.equal(pre.status,0);const ready=JSON.parse(pre.stdout);assert.deepEqual(ready.processCounts,{shell:0,native:0,llama:0});assert.deepEqual(ready.resourceBlockers,[]);
mkdirSync(home);for(const name of ['data','models','local','roaming','desktop','runtime','profiles','webview','temp','screenshots'])mkdirSync(join(home,name));
const componentName='0.5.2-windows-x86_64-cuda-cuda13',source=join(out,'signed-component-r4',componentName),component=join(home,'components/engine',componentName);
assert.equal(sha256(join(source,'engine-component.json')),native.signedManifestSHA256);cpSync(source,component,{recursive:true});verifyEngine(component,'0.5.2');
const isolatedRuntime=findLocalBackend('PrismML',{componentsDir:join(home,'components'),installRoot:qa.app});assert.equal((await verifyLocalBackend(isolatedRuntime,'PrismML')).version,'0.4.1');
writeFileSync(join(home,'data/state.json'),JSON.stringify({version:1,setup:{complete:true,flowVersion:2,step:4,history:[]},installed:{},current:null,loaded:null,beginner:{priority:'balanced',contextLength:'96k',readImages:false,gpuIdle:'never'},expert:{'sample.maxTokens':32,'sample.temp':0,'sample.speedProjection':false}}));
const {chromium}=createRequire(join(root,'apps/host/package.json'))('playwright-core');
let child,browser,page,base,id,firstPid;
const result={ok:false,sourceCommit:commit,actualPrism041:false,isolatedProfile:true,userProfileMutation:false,mocksUsed:false,fullWindowQualified:false,generalQualityQualified:false,checks:[],pageErrors:0};
const until=async(check,seconds=150)=>{const end=Date.now()+seconds*1000;while(Date.now()<end){const value=await check();if(value)return value;await delay(500);}throw Error('Prism E2E deadline');};
const call=async(path,method='GET',body)=>{const r=await fetch(base+path,{method,headers:{'content-type':'application/json'},...(body===undefined?{}:{body:JSON.stringify(body)}),signal:AbortSignal.timeout(60000)});assert.ok(r.ok);return r.json();};
async function start(){
 const s=createServer();await new Promise(r=>s.listen(0,'127.0.0.1',r));const port=s.address().port;await new Promise(r=>s.close(r));
 const env=Object.fromEntries(Object.entries(process.env).filter(([k])=>!/^(STRATA_|DSH_|WEBVIEW2_)/i.test(k)&&!/(?:API_KEY|TOKEN|SECRET|PASSWORD)$/i.test(k)));
 Object.assign(env,{STRATA_HOME:home,STRATA_WORKSPACE:join(home,'workspace'),STRATA_MODELS_DIR:join(home,'models'),STRATA_COMPONENTS_DIR:join(home,'components'),STRATA_SETTINGS_FILE:join(home,'settings.json'),STRATA_DESKTOP_DATA_DIR:join(home,'desktop'),STRATA_RUNTIME_DIR:join(home,'runtime'),STRATA_PROFILE_DIR:join(home,'profiles'),STRATA_AGENT:'off',STRATA_REAL_ENGINE:'1',STRATA_NO_ACQUIRE:'1',STRATA_NO_BUNDLED_ENGINE:'1',STRATA_NO_REPAIR_ACCESS:'1',STRATA_NO_DIALOG:'1',APPDATA:join(home,'roaming'),LOCALAPPDATA:join(home,'local'),TEMP:join(home,'temp'),TMP:join(home,'temp'),DSH_HOME:join(home,'harness'),WEBVIEW2_USER_DATA_FOLDER:join(home,'webview'),WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS:'--remote-debugging-address=127.0.0.1 --remote-debugging-port='+port});
 const automatic=plan.automaticEnvironmentBefore.Process??plan.automaticEnvironmentBefore.User??plan.automaticEnvironmentBefore.Machine;if(automatic)env.STRATA_LLAMA_SERVER=automatic;
 child=spawn(join(qa.app,'strata-desktop.exe'),[],{cwd:qa.app,env,windowsHide:true,stdio:'ignore'});
 await until(async()=>{assert.equal(child.exitCode,null);try{return (await fetch(`http://127.0.0.1:${port}/json/version`,{signal:AbortSignal.timeout(1000)})).ok;}catch{return false;}},60);
 browser=await chromium.connectOverCDP(`http://127.0.0.1:${port}`);page=await until(()=>browser.contexts().flatMap(c=>c.pages()).find(p=>/^http:\/\/127\.0\.0\.1:\d+/.test(p.url())),60);base=new URL(page.url()).origin;
 page.setDefaultTimeout(20000);page.on('pageerror',()=>result.pageErrors++);
 await page.evaluate(()=>localStorage.setItem('strata.prefs.v2',JSON.stringify({language:'en'})));await page.reload({waitUntil:'domcontentloaded'});
 assert.equal((await call('/api/app/build')).host,'0.5.2-local.2');
}
async function stop(){
 if(base){const status=await call('/api/engine/status').catch(()=>null);if(status?.running)await call('/api/models/unload','POST',{});}
 await browser?.close().catch(()=>{});browser=null;if(child?.exitCode===null)child.kill();if(child)await until(()=>child.exitCode!==null||child.signalCode!==null,15);child=null;base=null;
}
async function screen(){await page.goto(base+'/#/models/local?variant='+encodeURIComponent(id));await page.getByRole('button',{name:'Manage models',exact:true}).waitFor();}
async function observed(){
 const status=await until(async()=>{const value=await call('/api/engine/status');return value.running?value:false;});
 assert.equal(status.requestedBackend,'PrismML');assert.equal(status.binary.source,'PrismML');assert.equal(status.binary.version,'0.4.1');assert.equal(sha256(status.binary.path),'5ca5d8bfdfcb6fe46a96a06b71eaf45aedfbd1c6673bda2a10eb6412613be777');
 const probe=spawnSync(py,['-B',join(import.meta.dirname,'probe-prism-child.py'),'--shell',String(child.pid),'--child',String(status.pid),'--port',String(status.port),'--model',model],{encoding:'utf8',windowsHide:true,timeout:30000});assert.equal(probe.status,0);assert.equal(JSON.parse(probe.stdout).owned,true);
 const props=await (await fetch(`http://127.0.0.1:${status.port}/props`,{signal:AbortSignal.timeout(15000)})).json();assert.equal(props.default_generation_settings.n_ctx,98304);
 const allocation=(await call('/api/engine/logs?limit=2000')).logs.map(entry=>entry.line).join('\n');
 assert.match(allocation,/flash_attn\s*=\s*enabled/);assert.match(allocation,/K \(q8_0\).*V \(q8_0\)/);
 result.observedQ8Kv=true;result.observedFlashAttention=true;
 result.actualPrism041=true;return status;
}
try{
 await start();const imported=await call('/api/local-models','POST',{path:model});id=imported.variantId;assert.equal(id,plan.target.variantId);await screen();
 await page.getByRole('button',{name:'Expert mode',exact:true}).click();await page.getByRole('tab',{name:'JSON',exact:true}).click();
 const editor=page.getByRole('textbox',{name:'Run configuration JSON',exact:true});const config=JSON.parse(await editor.inputValue());
 config.beginner.contextLength='96k';config.beginner.readImages=false;Object.assign(config.expert,{'runtime.backend':'PrismML','ctx.length':'96K','ctx.kvMode':'Q8_0','ctx.kType':'Q8_0','ctx.vType':'Q8_0','perf.flashAttn':true,'server.parallel':1,'sample.maxTokens':32,'sample.temp':0});
 await editor.fill(JSON.stringify(config,null,2));await page.getByRole('button',{name:/^Saved settings/}).click();await page.getByRole('menuitem',{name:'Save current settings…',exact:true}).click();await page.getByRole('textbox',{name:'Settings name',exact:true}).fill('Prism041 reported GGUF QA');await page.getByRole('dialog').getByRole('button',{name:'Save settings',exact:true}).click();await page.getByRole('dialog').waitFor({state:'hidden'});
 const profiles=await page.evaluate(()=>JSON.parse(localStorage.getItem('strata.run-profiles.v1')));assert.equal(profiles.profiles.at(-1).variantId,id);assert.equal(profiles.profiles.at(-1).config.expert['runtime.backend'],'PrismML');
 for(const [cycle,prompt,answer] of [[0,'What is 2 + 3? Reply with only the number.','5'],[1,'What is 7 times 8? Reply with only the number.','56']]){
  if(cycle){await start();await screen();assert.deepEqual(await page.evaluate(()=>JSON.parse(localStorage.getItem('strata.run-profiles.v1'))),profiles);await page.getByRole('button',{name:/^Saved settings/}).click();await page.getByRole('menuitemradio',{name:/Prism041 reported GGUF QA/}).click();}
  await page.getByRole('button',{name:'Load model',exact:true}).click();const status=await observed();if(cycle)assert.notEqual(status.pid,firstPid);else firstPid=status.pid;
  const chatId='prism041-qa-'+cycle;await call('/api/chats/import','POST',{sessions:[{id:chatId,title:'Isolated Prism041 QA',createdAt:new Date().toISOString(),updatedAt:new Date().toISOString(),messages:[],tokensIn:0,tokensOut:0,workspace:join(home,'workspace'),preferences:{agent:false,useTools:false,thinking:'Off',thinkingBudget:0}}]});
  await page.goto(base+'/#/chat/'+chatId);await page.reload();await page.getByRole('textbox',{name:'Message Strata',exact:true}).fill(prompt);await page.getByRole('textbox',{name:'Message Strata',exact:true}).press('Enter');
  const reply=await until(async()=>{const value=await call('/api/chats/'+chatId);return value.session.messages.find(m=>m.role==='assistant'&&m.content?.trim()&&!m.streaming);});assert.equal(reply.content.trim(),answer);assert.notEqual(reply.demo,true);
  await page.screenshot({path:join(home,'screenshots',`prism041-cycle-${cycle}.png`),fullPage:true});result.checks.push(cycle?'actual-gui-restart-second-answer':'actual-gui-first-answer');
  await screen();await page.getByRole('button',{name:'Manage models',exact:true}).click();await page.getByRole('button',{name:'Unload model',exact:true}).click();await until(async()=>!(await call('/api/engine/status')).running);await stop();
 }
 const after=statSync(model,{bigint:true});assert.equal(after.size,before.size);assert.equal(after.mtimeNs,before.mtimeNs);assert.equal(result.pageErrors,0);
 result.modelBytesPreserved=true;result.savedTargetProfileRetained=true;result.automaticOverrideWasPreserved=true;result.ok=true;
}catch(error){result.errorType=error.name;result.errorCode=error.code??null;process.exitCode=1;}
finally{try{await stop();result.ownedCleanup=true;}catch{if(child?.exitCode===null)child.kill();result.ownedCleanup=false;result.ok=false;process.exitCode=1;}writeFileSync(join(out,'prism-e2e.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(result));}
