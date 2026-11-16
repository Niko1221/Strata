/** Actual installed shell/WebView2 and real RustXNG; the language-model API is an explicit local test fixture. */
import assert from 'node:assert/strict';
import http from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { mkdirSync, readFileSync, writeFileSync, existsSync, lstatSync } from 'node:fs';
import { join, resolve, dirname, basename } from 'node:path';
import { createHash } from 'node:crypto';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

/** Build only the isolated shell environment; importing this helper starts nothing. */
export function isolatedShellEnvironment(home,debugPort){
  const env=Object.fromEntries(Object.entries(process.env).filter(([key])=>!/^(STRATA_|DSH_|WEBVIEW2_|HERMES_HOME$)/i.test(key)&&!/(API_KEY|TOKEN|SECRET|PASSWORD)$/i.test(key)));
  Object.assign(env,{HERMES_HOME:join(home,'.hermes'),STRATA_HOME:home,STRATA_INSTALL_DIR:join(home,'install'),STRATA_COMPONENTS_DIR:join(home,'components'),STRATA_PROFILE_DIR:join(home,'profiles'),STRATA_RUNTIME_DIR:join(home,'runtime'),STRATA_SETTINGS_FILE:join(home,'settings.json'),STRATA_MODELS_DIR:join(home,'models'),STRATA_WORKSPACE:join(home,'workspace'),STRATA_DESKTOP_DATA_DIR:join(home,'desktop'),STRATA_NO_ACQUIRE:'1',STRATA_NO_BUNDLED_ENGINE:'1',STRATA_NO_REPAIR_ACCESS:'1',STRATA_NO_DIALOG:'1',DSH_HOME:join(home,'harness'),APPDATA:join(home,'Roaming'),LOCALAPPDATA:join(home,'Local'),TEMP:join(home,'temp'),TMP:join(home,'temp'),WEBVIEW2_USER_DATA_FOLDER:join(home,'webview'),WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS:`--remote-debugging-address=127.0.0.1 --remote-debugging-port=${debugPort}`});
  env.DSH_AGENTS_HOME=join(home,'.agents');
  return env;
}

const root='C:/Users/downl/Documents/ChatGPT/Strata';
const out='H:/Strata/releases/stable-060-20261116-codex';
async function main(){
if(process.argv[2]!=='--execute'){
  console.log(JSON.stringify({execute:false,entry:'delivery.py candidate-qa or installed-qa',mockModelOnly:true}));
  process.exit(0);
}
assert.equal(process.platform,'win32');
const digest=path=>createHash('sha256').update(readFileSync(path)).digest('hex');
const input=resolve(process.argv[3]??'');
assert.match(process.argv[4]??'',/^[a-f0-9]{64}$/);
assert.equal(digest(input),process.argv[4]);
const contract=JSON.parse(readFileSync(input,'utf8'));
const work=resolve(contract.work);
assert.equal(dirname(work),resolve(out));
assert.match(basename(work),/^(candidate|installed)-qa-r[1-9][0-9]*$/);
assert.equal(input,join(work,'qa-input.json'));
assert.equal(contract.ownerPID,process.ppid);
assert.equal(JSON.parse(readFileSync(join(work,'reservation.json'),'utf8')).token,process.env.HAKUA060_WORKER_TOKEN);
assert.ok(process.env.HAKUA060_WORKER_TOKEN);
const noLinks=path=>{for(let p=resolve(path);;p=dirname(p)){if(existsSync(p))assert.equal(lstatSync(p).isSymbolicLink(),false);if(dirname(p)===p)break;}};
noLinks(work);
const target=contract.target;assert.ok(['candidate','installed'].includes(target));
const app=resolve(contract.app);noLinks(app);
if(target==='installed')assert.equal(app,resolve('C:/Users/downl/AppData/Local/Programs/Strata'));
else {assert.equal(dirname(dirname(app)),resolve(out));assert.match(basename(dirname(app)),/^extract-r[1-9][0-9]*$/);assert.equal(basename(app),'extracted');}
assert.equal(digest(join(app,'strata-desktop.exe')),contract.nsisShellSHA256);
assert.equal(JSON.parse(readFileSync(join(app,'harness/rustxng/SOURCE.json'),'utf8')).profile,'full');
const home=join(work,'qa-home');noLinks(home);mkdirSync(home);
for(const name of ['.hermes','.agents','data','workspace','models','components','profiles','runtime','desktop','Local','Roaming','temp','screenshots'])mkdirSync(join(home,name));
for(const name of ['first','second','third'])writeFileSync(join(home,'workspace',name+'.txt'),'PRIVATE_FIXTURE_CONTENT_'+name,{flag:'wx'});
writeFileSync(join(home,'data/state.json'),JSON.stringify({version:1,setup:{complete:true,flowVersion:2,step:4,history:[]},installed:{},current:null,loaded:null,expert:{'sample.maxTokens':64}}),{flag:'wx'});
const requests=[];let mainResponse=null,mainClosed=false,mainReleased=false;
const model=http.createServer(async(req,res)=>{
  const parts=[];for await(const part of req)parts.push(part);
  const body=parts.length?JSON.parse(Buffer.concat(parts)):null;
  if(req.url.includes('/models')){res.setHeader('content-type','application/json');res.end(JSON.stringify({data:[{id:'qa-model',name:'QA model',context_length:8192,input_modalities:['text']}]}));return;}
  if(req.url==='/props'){res.setHeader('content-type','application/json');res.end(JSON.stringify({modalities:{vision:false}}));return;}
  requests.push(body);
  res.writeHead(200,{'content-type':'text/event-stream'});
  const completionId='qa-completion-060-'+requests.length;
  const send=value=>res.write('data: '+JSON.stringify({id:completionId,object:'chat.completion.chunk',model:'qa-model',created:Math.floor(Date.now()/1000),...value})+'\n\n');
  send({choices:[{index:0,delta:{role:'assistant'}}]});
  const texts=(body?.messages??[]).filter(row=>row.role==='user').map(row=>typeof row.content==='string'?row.content:(row.content??[]).map(part=>part.text??'').join('\n'));
  const text=[...texts].reverse().find(value=>/Observed workflow:\n|Please apply desktop validation|WORKFLOW_060|Side question:|MAIN_PENDING_060/.test(value))??texts.at(-1)??'';
  const full=JSON.stringify(body?.messages??[]);
  if(String(text).includes('Observed workflow:\n')){
    const packet=JSON.parse(String(text).split('Observed workflow:\n').at(-1));
    assert.equal(full.includes('PRIVATE_FIXTURE_CONTENT'),false);
    assert.equal((body.tools??[]).length,0);
    const answer=JSON.stringify({learn:true,name:'desktop-validation',title:'Desktop validation',description:'Use when repeating desktop validation.',triggers:['desktop validation','workflow review'],steps:packet.tools.slice(0,3).map((tool,index)=>({toolId:tool.id,note:['Inspect the first current input.','Compare the second current input.','Confirm the third current input before reporting.'][index]})),pitfalls:['Check the current inputs before reuse.']});
    send({choices:[{index:0,delta:{content:answer}}]});send({choices:[{index:0,delta:{},finish_reason:'stop'}],usage:{prompt_tokens:50,completion_tokens:80}});res.end('data: [DONE]\n\n');return;
  }
  if(String(text).includes('Please apply desktop validation')){
    assert.match(String(text),/Inspect the first current input/);
    send({choices:[{index:0,delta:{content:'AUTO_APPLIED_060'}}]});send({choices:[{index:0,delta:{},finish_reason:'stop'}],usage:{prompt_tokens:80,completion_tokens:12}});res.end('data: [DONE]\n\n');return;
  }
  if(String(text).includes('WORKFLOW_060')){
    if(!(body.messages??[]).some(row=>row.role==='tool')){
      send({choices:[{index:0,delta:{tool_calls:['first','second','third'].map((name,index)=>({index,id:'read-'+index,type:'function',function:{name:'read',arguments:JSON.stringify({file_path:join(home,'workspace',name+'.txt')})}}))}}]});
      send({choices:[{index:0,delta:{},finish_reason:'tool_calls'}],usage:{prompt_tokens:80,completion_tokens:50}});
    }else{send({choices:[{index:0,delta:{content:'WORKFLOW_DONE_060'}}]});send({choices:[{index:0,delta:{},finish_reason:'stop'}],usage:{prompt_tokens:100,completion_tokens:12}});}
    res.end('data: [DONE]\n\n');return;
  }
  if(String(text).includes('MAIN_PENDING_060')&&!String(text).includes('Side question:')){
    mainResponse=res;
    res.on('close',()=>{if(!mainReleased)mainClosed=true;});
    send({choices:[{index:0,delta:{content:'Main task running.\n'}}]});
    const heartbeat=setInterval(()=>res.write(': keepalive\n\n'),1000);res.on('close',()=>clearInterval(heartbeat));return;
  }
  const answer=String(text).includes('Side question:')?'SIDE_ANSWER_060':'FOLLOW_UP_ANSWER_060';
  send({choices:[{index:0,delta:{content:answer}}]});
  send({choices:[{index:0,delta:{},finish_reason:'stop'}],usage:{prompt_tokens:20,completion_tokens:8}});
  res.end('data: [DONE]\n\n');
});
await new Promise(resolve=>model.listen(0,'127.0.0.1',resolve));
const portProbe=http.createServer();await new Promise(resolve=>portProbe.listen(0,'127.0.0.1',resolve));const debugPort=portProbe.address().port;await new Promise(resolve=>portProbe.close(resolve));
const hostProbe=http.createServer();await new Promise(resolve=>hostProbe.listen(0,'127.0.0.1',resolve));const hostPort=hostProbe.address().port;await new Promise(resolve=>hostProbe.close(resolve));
assert.notEqual(hostPort,debugPort);
writeFileSync(join(home,'desktop/shell-prefs.json'),JSON.stringify({port:hostPort,tray:false,language:'en'}),{flag:'wx'});
const env=isolatedShellEnvironment(home,debugPort);
const child=spawn(join(app,'strata-desktop.exe'),[],{env,windowsHide:true,stdio:'ignore'});
const {chromium}=createRequire(join(root,'apps/host/package.json'))('playwright-core');
let browser,page,base;const errors=[],sideBodies=[];
const until=async(fn,ms=30_000)=>{const end=Date.now()+ms;while(Date.now()<end){const value=await fn();if(value)return value;await delay(150);}throw new Error('QA condition timed out');};
const call=async(path,method='GET',body)=>{const response=await fetch(base+path,{method,headers:{origin:base,'content-type':'application/json','x-strata-client':'isolated-feature-qa'},...(body===undefined?{}:{body:JSON.stringify(body)}),signal:AbortSignal.timeout(60_000)});assert.equal(response.status,200,`${path}: ${response.status}`);return response.json();};
const report={ok:false,target,sourceCommit:contract.sourceCommit,installerSHA256:contract.installerSHA256,actualInstalledExe:target==='installed',actualWebView2:true,isolatedHome:true,mockModelOnly:true,realRustxng:true,checks:[]};
try{
  await until(async()=>{try{return (await fetch(`http://127.0.0.1:${debugPort}/json/version`,{signal:AbortSignal.timeout(700)})).ok;}catch{return false;}},60_000);
  browser=await chromium.connectOverCDP(`http://127.0.0.1:${debugPort}`);
  page=await until(()=>browser.contexts().flatMap(context=>context.pages()).find(page=>/^http:\/\/127\.0\.0\.1:\d+/.test(page.url())));
  base=new URL(page.url()).origin;
  assert.equal(base,`http://127.0.0.1:${hostPort}`);
  page.on('pageerror',error=>errors.push(error.message));
  page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
  page.on('request',request=>{if(new URL(request.url()).pathname==='/api/chat/btw')sideBodies.push(JSON.parse(request.postData()));});
  const identity=await call('/api/app/build');assert.deepEqual(identity,{host:'0.6.0',shell:'0.6.0'});report.checks.push('installed-host-shell-renderer-identity');
  const provider=await call('/api/model-providers','POST',{kind:'custom',name:'Isolated QA model fixture',baseUrl:`http://127.0.0.1:${model.address().port}/v1`,protocol:'openai-chat',models:[{id:'qa-model',name:'QA model',vision:false,reasoning:false,contextMax:8192}]});
  const connectionId=provider.connections.find(row=>row.name==='Isolated QA model fixture').id;
  const id='qa-side-060';
  await call('/api/chats/import','POST',{sessions:[{id,title:'Isolated side-question QA',workspace:join(home,'workspace'),messages:[],tokensIn:0,tokensOut:0,createdAt:new Date().toISOString(),updatedAt:new Date().toISOString(),preferences:{agent:false,useTools:false,remoteModel:{connectionId,modelId:'qa-model'}}}]});
  await page.goto(base+'/#/chat/'+id);await page.reload();
  const input=page.getByRole('textbox',{name:/Message Strata|Strataにメッセージを送信/});await input.waitFor({timeout:60_000});
  await input.fill('MAIN_PENDING_060');await input.press('Enter');
  await until(()=>mainResponse);
  await page.getByText('Main task running.',{exact:true}).waitFor();
  await input.fill('/btw Explain the current status.');await input.press('Enter');
  await page.getByText('SIDE_ANSWER_060',{exact:true}).waitFor({timeout:30_000});
  assert.equal(mainClosed,false);assert.equal(mainReleased,false);assert.equal(sideBodies.length,1);
  const stored=await until(async()=>{const row=await call('/api/chats/'+id);return row.session.asides?.[0]?.answer==='SIDE_ANSWER_060'?row.session:null;});
  assert.equal(stored.messages.length,2);assert.equal(stored.messages.some(row=>row.content.includes('SIDE_ANSWER_060')),false);
  const sideRequest=requests.find(body=>JSON.stringify(body.messages).includes('Side question:'));
  assert.ok(sideRequest);assert.equal((sideRequest.tools??[]).length,0);assert.ok(stored.asides[0].agentId);
  await page.getByText(/^(Subagent answer · no tools|サブエージェントの回答 · ツールなし)$/).waitFor();
  report.checks.push('btw-typed-during-main-stream','main-not-cancelled-or-queued','btw-real-child-tool-free-api','separate-durable-aside');
  await page.screenshot({path:join(home,'screenshots/btw-desktop.png'),fullPage:true});
  mainReleased=true;
  mainResponse.write('data: '+JSON.stringify({choices:[{index:0,delta:{content:'Main task finished.'}}]})+'\n\n');
  mainResponse.write('data: '+JSON.stringify({choices:[{index:0,delta:{},finish_reason:'stop'}],usage:{prompt_tokens:20,completion_tokens:12}})+'\n\n');mainResponse.end('data: [DONE]\n\n');
  await until(async()=>{const row=await call('/api/chats/'+id);return row.session.messages.at(-1)?.streaming===false;});
  await input.fill('/deep');await page.getByRole('option').filter({hasText:'/deepresearch'}).waitFor();
  await input.press('Escape');await input.fill('');report.checks.push('deepresearch-slash-menu');
  await call('/api/agent/tools','PUT',{groups:{search:true},search:{provider:'rustxng'}});
  const search=await call('/api/chat/web-search','POST',{sessionId:id,query:'Tauri NSIS Windows installer documentation',confirmed:true});
  assert.ok(search.sources.length>0);report.rustxngSources=search.sources.map(({title,url})=>({title,url}));report.checks.push('installed-rustxng-full-real-search');
  await page.reload();await page.getByText('SIDE_ANSWER_060',{exact:true}).waitFor();report.checks.push('aside-survives-page-reload');
  await page.goto(base+'/#/skills');
  const learning=page.getByRole('switch',{name:/^(Learn from completed work|完了した作業から学ぶ)$/});
  const automatic=page.getByRole('switch',{name:/^(Automatically apply skills to agent workflows|実行ワークフローにスキルを自動適用する)$/});
  await learning.waitFor();assert.equal(await learning.getAttribute('aria-checked'),'false');assert.equal(await automatic.getAttribute('aria-checked'),'false');
  await automatic.click();await until(async()=>await automatic.getAttribute('aria-checked')==='true');
  await learning.click();await until(async()=>await learning.getAttribute('aria-checked')==='true');
  report.checks.push('learning-and-application-default-off-independent-ui-toggles');
  const workflowId='qa-workflow-060';
  await call('/api/chats/import','POST',{sessions:[{id:workflowId,title:'Workflow learning QA',workspace:join(home,'workspace'),messages:[],tokensIn:0,tokensOut:0,createdAt:new Date().toISOString(),updatedAt:new Date().toISOString(),preferences:{agent:true,useTools:true,mode:'bypass',remoteModel:{connectionId,modelId:'qa-model'}}}]});
  await page.goto(base+'/#/chat/'+workflowId);await page.reload();await input.waitFor();
  await input.fill('WORKFLOW_060 Perform desktop validation by reading the three fixture inputs.');await input.press('Enter');
  await page.getByText('WORKFLOW_DONE_060',{exact:true}).waitFor({timeout:60_000});
  await until(async()=>{const status=await call('/api/skills/learning');return status.last?.phase==='saved'&&!status.running;},60_000);
  const learned=(await call('/api/skills')).find(row=>row.id==='my-learned-desktop-validation');
  assert.ok(learned?.enabled);assert.ok(learned.learning);assert.match(learned.body,/Inspect the first current input/);
  assert.equal(learned.body.includes('PRIVATE_FIXTURE_CONTENT'),false);
  await input.fill('Please apply desktop validation.');await input.press('Enter');
  await page.getByText('AUTO_APPLIED_060',{exact:true}).waitFor({timeout:60_000});
  report.checks.push('real-harness-three-completed-reads','same-model-bounded-tool-free-review','private-skill-created-without-file-outputs','next-workflow-receives-full-matching-skill');
  await page.goto(base+'/#/skills');await page.getByText('Desktop validation',{exact:true}).first().waitFor();
  await page.screenshot({path:join(home,'screenshots/skills-learning-desktop.png'),fullPage:true});
  await page.getByRole('button',{name:/^(Undo last learned change|直前の学習を取り消す)$/}).click();
  await until(async()=>{const all=await call('/api/skills');return all.some(row=>row.id===learned.id&&!row.enabled&&row.body===learned.body);});
  await page.reload();await learning.waitFor();assert.equal(await learning.getAttribute('aria-checked'),'true');assert.equal(await automatic.getAttribute('aria-checked'),'true');
  report.checks.push('ui-undo-disables-retains-document','settings-and-undone-skill-survive-reload');
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:join(home,'screenshots/btw-mobile.png'),fullPage:true});
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+1);assert.equal(overflow,false);report.checks.push('mobile-no-horizontal-overflow');
  assert.deepEqual(errors,[]);report.consoleErrors=0;report.ok=true;
}finally{
  if(!report.ok&&page){await page.screenshot({path:join(home,'screenshots/failure.png'),fullPage:true}).catch(()=>{});}
  writeFileSync(join(home,'fixture-model-requests.json'),JSON.stringify(requests,null,2)+'\n',{flag:'wx'});
    if(child.exitCode===null){
    // Close only the shell this fixture spawned through its normal window path.
    const command=`$p=Get-Process -Id ${child.pid} -ErrorAction Stop; if(-not $p.CloseMainWindow()){throw 'Normal window close refused'}`;
    const close=spawnSync(join(process.env.SYSTEMROOT,'System32/WindowsPowerShell/v1.0/powershell.exe'),['-NoProfile','-NonInteractive','-Command',command],{windowsHide:true,stdio:'ignore',timeout:15000});
    if(close.status===0){try{await until(()=>child.exitCode!==null,30000);report.normalQuit=true;}catch{report.normalQuit=false;}}
    else report.normalQuit=false;
  }else report.normalQuit=false;
  if(!report.normalQuit)report.ok=false;
  await browser?.close().catch(()=>{});
  model.closeAllConnections();await new Promise(resolve=>model.close(resolve));
  writeFileSync(join(home,'result.json'),JSON.stringify({...report,errors},null,2)+'\n',{flag:'wx'});

}
console.log(JSON.stringify(report,null,2));
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))await main();
