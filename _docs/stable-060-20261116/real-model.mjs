/** Genuine installed-app producer. Importing is inert; execution is owner-bound. */
import assert from 'node:assert/strict';
import http from 'node:http';
import net from 'node:net';
import {spawn, spawnSync} from 'node:child_process';
import {createHash, randomUUID} from 'node:crypto';
import {createRequire} from 'node:module';
import {closeSync, createReadStream, existsSync, lstatSync, mkdirSync, openSync, readFileSync, readSync, readdirSync, renameSync, statSync, writeFileSync} from 'node:fs';
import {basename, dirname, join, resolve} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';
import {zstdDecompressSync} from 'node:zlib';
import {isolatedShellEnvironment} from './qa-installed.mjs';
import {noLinks} from './real-model-mcp.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '../..');
const OUT = resolve('H:/Strata/releases/stable-060-20261116-codex');
const APP = resolve('C:/Users/downl/AppData/Local/Programs/Strata');
const PRISM = '1965755ceee6738b9a04ceaf5d30e6305d6ccd00';
export const GROUPS = ['shell','files','find','jobs','fetch','search','helpers','goals','questions','todo','skills','present'];
export const CSV = Object.freeze({inventory: 'id,version\nalpha,1\nbeta,2\n', expected: 'id,expected_version\nalpha,2\nbeta,2\ngamma,3\n', requirements: 'id,required_check,status\nalpha,checksum,passed\nbeta,checksum,failed\ngamma,checksum,passed\n'});
export const QUALITY = Object.freeze({factualAccuracy:'NOT_REVIEWED',memoryAnswerRecall:'NOT_REVIEWED',generalModelQuality:'UNQUALIFIED',fullWindow:'UNQUALIFIED',gpuConcurrency:'UNQUALIFIED',releaseQualified:false});
const hash = path => {const fd=openSync(path,'r'),h=createHash('sha256'),buffer=Buffer.alloc(1024*1024);try{let n;while((n=readSync(fd,buffer,0,buffer.length,null)))h.update(buffer.subarray(0,n));return h.digest('hex');}finally{closeSync(fd);}};
const artifact = path => {noLinks(path);assert.ok(lstatSync(path).isFile());return {path,sha256:hash(path)};};
const json = path => JSON.parse(readFileSync(path,'utf8'));
const sha = text => createHash('sha256').update(text).digest('hex');
const until = async(fn, ms=30000) => {const end=Date.now()+ms;while(Date.now()<end){const value=await fn();if(value)return value;await delay(200);}throw new Error('Bounded wait expired');};
const closedPort = port => new Promise(resolveResult => {const socket=net.connect({host:'127.0.0.1',port});socket.setTimeout(500);socket.once('connect',()=>{socket.destroy();resolveResult(false);});socket.once('error',()=>resolveResult(true));socket.once('timeout',()=>{socket.destroy();resolveResult(false);});});
const alive = pid => {try{process.kill(pid,0);return true;}catch{return false;}};
const port = async() => {const s=http.createServer();await new Promise(r=>s.listen(0,'127.0.0.1',r));const value=s.address().port;await new Promise(r=>s.close(r));return value;};
function inventory(root) {noLinks(root);const out={};function visit(dir,prefix=''){for(const e of readdirSync(dir,{withFileTypes:true})){const path=join(dir,e.name);noLinks(path);if(e.isDirectory())visit(path,prefix+e.name+'/');else {assert.ok(e.isFile());out[prefix+e.name]=hash(path);}}}visit(root);return out;}

/** No response access key or authentication material is ever serialized. */
export function safeFrame(value) {
  if(Array.isArray(value))return value.map(safeFrame);
  if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).filter(([key])=>!/^(authorization|profileKey|apiKey|key|token|headers|env|environment)$/i.test(key)).map(([key,v])=>[key,safeFrame(v)]));
  return value;
}
export function checkAudit(audit) {
  assert.equal(typeof audit.criteria,'string');
  assert.deepEqual(audit.version_mismatches,[{id:'alpha',inventory_version:'1',expected_version:'2'}]);
  assert.deepEqual(audit.duplicates,[]);
  assert.deepEqual(audit.missing_ids,[{id:'gamma',missingFrom:['inventory']}]);
  assert.deepEqual(audit.failed_checks,[{id:'beta',required_check:'checksum',status:'failed'}]);
  for(const [key,count] of Object.entries({total_mismatches:1,total_duplicates:0,total_missing:1,total_failed:1}))assert.equal(audit[key],count);
}
export function checkSettings(settings, customId, research=false) {
  assert.deepEqual(settings.groups.map(g=>g.id).sort(),[...GROUPS].sort());
  assert.ok(settings.groups.every(g=>g.enabled === (research?['search','fetch'].includes(g.id):g.id==='skills')));
  assert.equal(settings.browser.enabled,false);assert.equal(settings.webWithoutAsking,false);
  assert.equal(settings.search.provider,'rustxng');
  assert.equal(settings.custom.length,1);assert.equal(settings.custom[0].id,customId);
  assert.equal(settings.custom[0].enabled,!research);assert.equal(settings.custom[0].kind,'server');
}
export function checkHeaders(rows, allowed, route) {
  assert.ok(rows.length>0);
  for(const row of rows){assert.equal(row.provider,route);assert.ok(row.model);assert.ok(row.tools.every(t=>allowed.includes(t.name)));}
}
function ancestry(pid, owner) {
  assert.ok(Number.isSafeInteger(pid)&&pid>0&&Number.isSafeInteger(owner)&&owner>0);
  // Output is captured privately, parsed in memory, then only the verified
  // executable and two non-secret argument checks are retained.
  const script=`[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); $n=${pid}; $rows=@(); for($i=0;$i -lt 20 -and $n -gt 0;$i++){ $p=Get-CimInstance Win32_Process -Filter ('ProcessId='+$n); if(!$p){break}; $rows+=@{pid=[int]$p.ProcessId;parent=[int]$p.ParentProcessId;exe=$p.ExecutablePath;command=$p.CommandLine}; if($n -eq ${owner}){break}; $n=[int]$p.ParentProcessId }; ConvertTo-Json -InputObject @($rows) -Compress`;
  const result=spawnSync('C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',['-NoProfile','-NonInteractive','-Command',script],{encoding:'utf8',windowsHide:true,timeout:15000});
  assert.equal(result.status,0);const rows=JSON.parse(result.stdout);assert.ok(rows.length&&rows.at(-1).pid===owner);return {...rows[0],chain:rows.map(({pid,parent,exe})=>({pid,parent,exe}))};
}

/** Canonical session event logs, scoped to the isolated agent root. */
export function requestHeaders(agentRoot) {
  if(!existsSync(agentRoot))return [];
  for(let attempt=0;attempt<3;attempt++) {
    const found=[];let vanishedAtomic=false;
    noLinks(agentRoot);
    function visit(dir){for(const e of readdirSync(dir,{withFileTypes:true})){const p=join(dir,e.name);
      try {noLinks(p);} catch(error) {
        // Harness atomically renames regular .UUID.tmp files. Restart the entire
        // snapshot, retaining no partial rows; all canonical-log errors stay fatal.
        if(error.code==='ENOENT'&&e.isFile()&&/^\.[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}\.tmp$/i.test(e.name)
          &&typeof error.path==='string'&&resolve(error.path)===resolve(p)) {
          noLinks(dir);vanishedAtomic=true;
        }
        throw error;
      }
      if(e.isDirectory())visit(p);else if(/\.jsonl(?:\.zstd)?$/.test(e.name)){
        assert.ok(statSync(p).size<=16*1024*1024);const bytes=readFileSync(p);const text=(p.endsWith('.zstd')?zstdDecompressSync(bytes,{maxOutputLength:32*1024*1024}):bytes).toString('utf8');
        for(const line of text.split('\n').filter(Boolean)){const row=JSON.parse(line),event=row.type==='event'?row.event:row;
          if(event?.type==='request/header'){const header=event.data.header;found.push({file:p,fileSHA256:sha(bytes),seq:event.seq,sessionId:event.sessionId??basename(dirname(p)),provider:header.config.provider,model:header.config.model,maxTokens:header.config.maxTokens??null,tools:header.tools??[]});}
        }
      }
    }}
    try {visit(agentRoot);return found;} catch(error) {
      if(!vanishedAtomic||attempt===2)throw error;
    }
  }
}

async function execute(inputPath, expectedSHA) {
  assert.equal(process.platform,'win32');assert.match(expectedSHA,/^[a-f0-9]{64}$/);noLinks(inputPath);assert.equal(hash(inputPath),expectedSHA);
  const contract=json(inputPath),work=resolve(contract.work),home=join(work,'real-home'),ws=join(home,'workspace');
  assert.equal(dirname(work),OUT);assert.match(basename(work),/^real-model-r[1-9][0-9]*$/);assert.equal(resolve(inputPath),join(work,'input.json'));
  assert.equal(contract.ownerPID,process.ppid);assert.ok(process.env.HAKUA060_WORKER_TOKEN);assert.equal(json(join(work,'reservation.json')).token,process.env.HAKUA060_WORKER_TOKEN);
  let gpuObservationIndex=0;
  const gpu=(minimumFreeMiB=8192)=>{
    assert.equal(resolve(contract.gpuObserver.path),join(HERE,'gpu-admission.py'));
    assert.equal(hash(contract.gpuObserver.path),contract.gpuObserver.sha256);
    assert.equal(hash(contract.pythonExecutable.path),contract.pythonExecutable.sha256);
    const result=spawnSync(contract.pythonExecutable.path,['-B',contract.gpuObserver.path],{
      input:JSON.stringify({...contract.gpuPolicy,minimumFreeMiB}),encoding:'utf8',
      windowsHide:true,timeout:90000,maxBuffer:8*1024*1024});
    writeFileSync(join(work,`gpu-observation-${++gpuObservationIndex}.json`),JSON.stringify({status:result.status,stdout:result.stdout,stderr:result.stderr}),{flag:'wx'});
    assert.equal(result.status,0,'GPU identity admission failed; private observation required');
    return JSON.parse(result.stdout);
  };
  assert.equal(resolve(contract.app),APP);noLinks(APP);assert.equal(hash(join(APP,'strata-desktop.exe')),contract.shellSHA256);
  assert.equal(hash(contract.component.archive),contract.component.entry.sha256);assert.equal(hash(join(work,'capability.json')),contract.capabilitySHA256);
  assert.equal(statSync(contract.model.path).size,contract.model.bytes);
  assert.equal(Object.keys(process.env).some(k=>/^(HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|NO_PROXY)$/i.test(k)),false);
  noLinks(work);mkdirSync(home);
  for(const name of ['.hermes','.agents','data','workspace','models','components','profiles','runtime','desktop','Local','Roaming','temp','webview'])mkdirSync(join(home,name));
  for(const cwd of [home,ws]){const initialized=spawnSync('git',['-c','init.defaultBranch=main','init','--quiet',cwd],{windowsHide:true,stdio:'ignore',timeout:15000});assert.equal(initialized.status,0);}
  for(const [name,text] of Object.entries(CSV))writeFileSync(join(ws,name+'.csv'),text,{flag:'wx'});
  writeFileSync(join(home,'data/state.json'),JSON.stringify({version:1,setup:{complete:true,flowVersion:2,step:4,history:[]},installed:{},current:null,loaded:null}),{flag:'wx'});
  const hostPort=await port(),debugPort=await port();assert.notEqual(hostPort,debugPort);
  writeFileSync(join(home,'desktop/shell-prefs.json'),JSON.stringify({port:hostPort,tray:false,language:'en'}),{flag:'wx'});
  const env=isolatedShellEnvironment(home,debugPort);env.STRATA_HAKUA_CHATBOT='1';
  for(const key of Object.keys(env))assert.ok(!/^(HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|NO_PROXY)$/i.test(key));
  const base=`http://127.0.0.1:${hostPort}`,ownedChats=new Set();
  const report={schema:'strata.real-model.observations.v1',sourceCommit:contract.sourceCommit,inputSHA256:expectedSHA,outcome:'INCOMPLETE',quality:QUALITY,checks:Object.fromEntries(['installed','component','runtime','chat','audit','btw','learning','autoApply','research','memoryStore','memoryRecall','cleanup'].map(k=>[k,'NOT_RUN'])),turns:{},runtime:[],headers:[],proxyAbsent:true,deniedUnexpectedTools:0};
  const checkpoint=()=>{writeFileSync(join(work,'observations.next.json'),JSON.stringify(report,null,2));renameSync(join(work,'observations.next.json'),join(work,'observations.json'));};
  const call=async(path,method='GET',body,ms=60000)=>{const r=await fetch(base+path,{method,headers:{origin:base,'content-type':'application/json','x-strata-client':'bounded-real-model-060'},...(body===undefined?{}:{body:JSON.stringify(body)}),signal:AbortSignal.timeout(ms),redirect:'error'});assert.equal(r.status,200);return r.json();};
  const child=spawn(join(APP,'strata-desktop.exe'),[],{env,windowsHide:true,stdio:'ignore'});
  assert.ok(child.pid);report.shellPID=child.pid;checkpoint();
  let browser,variantId,customId,stage='installed',currentRuntime;
  const auditPrefix=()=>`mcp__custom-${customId.replace(/_/g,'-').slice(0,25)}__`;
  async function makeChat(name) {const id='real060-'+name+'-'+randomUUID();ownedChats.add(id);await call('/api/chats/import','POST',{sessions:[{id,title:'Private bounded '+name,workspace:ws,messages:[],tokensIn:0,tokensOut:0,createdAt:new Date().toISOString(),updatedAt:new Date().toISOString(),preferences:{agent:true,useTools:true,mode:'manual',thinking:'off'}}]});return id;}
  async function attest() {
    const s=await call('/api/engine/status');assert.equal(s.status,'running');assert.equal(s.modelPath,contract.model.path);assert.equal(s.requestedBackend,'PrismML');assert.equal(s.contextCapacity,32768);
    const runtimeDir=join(home,'components/engine',contract.component.folder,'local-backends/prism');
    const manifest=json(join(runtimeDir,'runtime.json'));assert.equal(manifest.version,'0.4.1');assert.equal(manifest.source.commit,PRISM);
    const componentManifest=join(home,'components/engine',contract.component.folder,'engine-component.json');assert.equal(hash(componentManifest),contract.component.entry.manifest_sha256);
    assert.equal(json(componentManifest).files.find(f=>f.path==='local-backends/prism/runtime.json')?.sha256,hash(join(runtimeDir,'runtime.json')));
    const files={};for(const f of manifest.files){const p=resolve(runtimeDir,f.path);assert.ok(p.startsWith(runtimeDir+'\\'));noLinks(p);assert.equal(hash(p),f.sha256);files[f.path]=f.sha256;}
    const observation=ancestry(s.pid,child.pid);assert.equal(resolve(observation.exe),resolve(s.binary.path));assert.ok(resolve(s.binary.path).startsWith(runtimeDir+'\\'));
    assert.ok(observation.command.includes(contract.model.path));assert.match(observation.command,/(?:^|\s)(?:-c|--ctx-size)(?:=|\s+)32768(?:\s|$)/);
    const g=gpu(768);assert.ok(g.freeMiB>=768&&g.ownedPIDs.includes(s.pid)&&g.computePIDs.includes(s.pid));
    const row={backend:s.requestedBackend,runtimeVersion:manifest.version,runtimeCommit:manifest.source.commit,manifest:artifact(join(runtimeDir,'runtime.json')),componentManifest:artifact(componentManifest),runtimeFiles:files,executable:observation.exe,executableSHA256:hash(observation.exe),pid:s.pid,port:s.port,modelSHA256:contract.model.sha256,contextCapacity:s.contextCapacity,ownedAncestry:true,ancestry:observation.chain,modelArgumentMatched:true,contextArgumentMatched:true,gpuPIDs:g.observedPIDs,gpuFreeMiB:g.freeMiB,gpuObservation:g};
    report.runtime.push(row);currentRuntime=row;checkpoint();return row;
  }
  function observeHeaders(phase,allowed,sessionId=null) {
    const all=requestHeaders(join(home,'data/agent'));
    const old=new Set(report.headers.map(h=>h.file+'#'+h.seq));
    const fresh=all.filter(h=>!old.has(h.file+'#'+h.seq));
    if(sessionId)assert.ok(fresh.some(h=>h.sessionId===sessionId));
    checkHeaders(fresh,allowed,report.route);
    report.headers.push(...fresh.map(h=>({...h,phase})));checkpoint();return fresh;
  }
  async function approve(chat,frame,phase) {
    if(frame.type!=='agent-tool'||frame.tool?.state!=='proposed')return;
    const t=frame.tool,name=t.name,args=t.args??{};
    let allowed=phase==='research'&&['web_search','web_fetch'].includes(name);
    if(phase==='audit')allowed=name==='audit_inventory'||(name===auditPrefix()+'read_audit_csv'&&Object.keys(args).length===1&&Object.hasOwn(CSV,args.table))||(name===auditPrefix()+'save_audit'&&Object.keys(args).length===1&&typeof args.audit_json==='string'&&args.audit_json.length<=65536);
    if(!allowed)report.deniedUnexpectedTools++;
    await call('/api/chat/approval','POST',{sessionId:chat,toolId:t.id,decision:allowed?'approve':'deny'});
    assert.ok(allowed,'Unexpected tool approval denied');
  }
  async function stream(name,path,body,phase='audit',ms=240000,extra={}) {
    const frames=[];report.turns[name]={request:safeFrame(body),frames,startedAt:new Date().toISOString()};checkpoint();
    const r=await fetch(base+path,{method:'POST',headers:{origin:base,'content-type':'application/json','x-strata-client':'bounded-real-model-060',...extra},body:JSON.stringify(body),signal:AbortSignal.timeout(ms),redirect:'error'});assert.equal(r.status,200);
    const decoder=new TextDecoder();let pending='',size=0;
    for await(const bytes of r.body){size+=bytes.length;assert.ok(size<=2*1024*1024);pending+=decoder.decode(bytes,{stream:true});let at;while((at=pending.indexOf('\n'))>=0){const line=pending.slice(0,at);pending=pending.slice(at+1);if(!line.trim())continue;const f=JSON.parse(line);frames.push(safeFrame(f));checkpoint();await approve(body.sessionId,f,phase);assert.ok(!f.error&&f.type!=='error');}}
    assert.equal(pending.trim(),'');report.turns[name].endedAt=new Date().toISOString();checkpoint();return frames;
  }
  async function turn(name,message,phase='audit',ms=240000) {
    const id=await makeChat(name);const frames=await stream(name,'/api/chat/stream',{sessionId:id,message,history:[],agent:true,tools:true,mode:'manual',thinking:'off'},phase,ms);
    assert.ok(frames.some(f=>f.type==='start'&&f.agent===true&&f.remoteModel===false&&f.demo===false));
    assert.ok(frames.some(f=>f.type==='done'&&!f.cancelled&&['stop','completed'].includes(f.finishReason)));
    const schemas=observeHeaders(phase,phase==='research'?['web_search','web_fetch']:report.auditTools);
    assert.ok(schemas.length);
    report.turns[name].serving=(await call('/api/serving')).requests.filter(r=>r.endpoint==='/agent'&&r.instanceId===report.instanceId);
    assert.ok(report.turns[name].serving.some(r=>r.state==='done'&&r.tokensOut>0));
    checkpoint();return frames;
  }
  async function unload() {if(!currentRuntime)return;const prior=currentRuntime;const s=await call('/api/engine/status');assert.equal(s.pid,prior.pid);const startedAt=new Date().toISOString();await call('/api/models/unload','POST',{},30000);await until(()=>!alive(prior.pid),30000);await until(()=>closedPort(prior.port),15000);const logs=await call('/api/engine/logs?limit=200');const escalation=logs.logs.filter(r=>r.at>=startedAt&&r.stream==='supervisor'&&r.line==='engine ignored SIGTERM — killing').map(r=>({at:r.at,signal:'SIGKILL'}));currentRuntime=null;report.unloads??=[];report.unloads.push({pid:prior.pid,port:prior.port,pidExited:true,portClosed:true,startedAt,escalation,method:'production supervisor stop; Windows SIGTERM uses TerminateProcess; escalation field records retained explicit fallback log'});checkpoint();}
  try {
    await until(async()=>{try{return(await fetch(`http://127.0.0.1:${debugPort}/json/version`,{signal:AbortSignal.timeout(700)})).ok;}catch{return false;}},60000);
    const {chromium}=createRequire(join(ROOT,'apps/host/package.json'))('playwright-core');browser=await chromium.connectOverCDP(`http://127.0.0.1:${debugPort}`);
    const page=await until(()=>browser.contexts().flatMap(c=>c.pages()).find(p=>p.url().startsWith(base+'/')));
    assert.equal(new URL(page.url()).origin,base);
    const host=ancestry((await call('/api/app/build'))&&child.pid,process.pid);assert.equal(resolve(host.exe),join(APP,'strata-desktop.exe'));
    report.installed={build:await call('/api/app/build'),shellSHA256:hash(join(APP,'strata-desktop.exe')),rendererOrigin:base,actualWebView:true};assert.deepEqual(report.installed.build,{host:'0.6.0',shell:'0.6.0'});report.checks.installed='PASS';checkpoint();
    stage='component';
    const imported=await fetch(base+'/api/engine/from-file',{method:'POST',headers:{origin:base,'x-strata-client':'bounded-real-model-060','x-file-name':basename(contract.component.archive),'content-length':String(statSync(contract.component.archive).size)},body:createReadStream(contract.component.archive),duplex:'half',signal:AbortSignal.timeout(180000),redirect:'error'});assert.equal(imported.status,202);const operation=await imported.json();
    const job=await until(async()=>{const j=(await call('/api/engine/job')).job;if(j?.id===operation.id&&['done','error','cancelled'].includes(j.state))return j;return null;},300000);assert.equal(job.state,'done');assert.equal(job.kind,'install-engine-file');
    await call('/api/engines/use','POST',{family:'qwen',folder:contract.component.folder});
    report.component={archiveSHA256:contract.component.entry.sha256,job:safeFrame(job),engines:await call('/api/engines')};report.checks.component='PASS';checkpoint();
    stage='runtime';
    const preview=await call('/api/local-models/inspect','POST',{path:contract.model.path});report.modelPreview=safeFrame(preview);
    assert.equal(preview.variant.local.architecture,'qwen35');
    const {readGGUF}=await import(pathToFileURL(join(APP,'host/server/gguf.mjs')).href);
    const gguf=readGGUF(contract.model.path);assert.equal(gguf.tensorCount,427);assert.equal(gguf.metadata['general.architecture'],'qwen35');report.modelMetadata={tensorCount:gguf.tensorCount,architecture:gguf.metadata['general.architecture']};
    const registered=await call('/api/local-models','POST',{path:contract.model.path,name:'Private real-model 060 reference'});variantId=registered.variantId;assert.equal(typeof variantId,'string');report.variantId=variantId;report.instanceId='supervisor:'+variantId;report.route='strata-local';
    const configuration={variantId,beginner:{contextLength:'32k'},expert:Object.entries({'runtime.backend':'PrismML','server.parallel':1,'mem.mmap':true,'perf.batch':128}).map(([id,value])=>({id,value}))};
    const validation=await call('/api/engine/validate','POST',configuration);assert.equal(validation.ok,true);report.engineValidation=validation;await call('/api/engine','PATCH',configuration);report.engineConfiguration=configuration;
    const beforeLoad=gpu();report.gpuBeforeLoad=beforeLoad;checkpoint();
    await call('/api/models/'+encodeURIComponent(variantId)+'/load','POST',{},180000);await attest();report.checks.runtime='PASS';checkpoint();
    const settings=await call('/api/agent/tools','PUT',{groups:Object.fromEntries(GROUPS.map(k=>[k,k==='skills'])),browser:{enabled:false},webWithoutAsking:false,search:{provider:'rustxng'}});
    assert.ok(settings.custom.length===0);const created=await call('/api/agent/tools/custom','POST',{name:'Owned Audit',about:'Read exactly three owned synthetic CSVs and persist one fixed audit.',kind:'server',address:join(APP,'node/node.exe'),args:[join(HERE,'real-model-mcp.mjs'),ws]});
    assert.equal(created.custom.length,1);customId=created.custom[0].id;checkSettings(created,customId);report.toolSettings=created;
    report.auditTools=['audit_inventory',auditPrefix()+'read_audit_csv',auditPrefix()+'save_audit','list_mcp_resources','list_mcp_resource_templates','read_mcp_resource','skill'];checkpoint();
    stage='chat';await turn('chat','Reply with one short sentence acknowledging this synthetic local validation. Do not use any tools.');report.checks.chat='PASS';report.savedChatEvidence='NOT_RUN: endpoint turn; no durable UI answer claim';checkpoint();
    await call('/api/skills/learning','PUT',{enabled:true,autoApply:true});
    stage='audit';await turn('audit','Perform the repeatable inventory reconciliation procedure. Read inventory, expected and requirements using the owned read_audit_csv tool. Pass all three complete CSV strings to audit_inventory. Use the owned save_audit tool once to persist the returned JSON unchanged. Summarize the counts and the IDs. Do not use resource tools.', 'audit',300000);
    noLinks(join(ws,'audit.json'));report.audit=json(join(ws,'audit.json'));checkAudit(report.audit);report.auditFile=artifact(join(ws,'audit.json'));report.auditInputs=Object.fromEntries(Object.entries(CSV).map(([name,text])=>[name,{...artifact(join(ws,name+'.csv')),content:text}]));report.checks.audit='PASS';checkpoint();
    stage='learning';
    report.learning=await until(async()=>{const l=await call('/api/skills/learning');return !l.running&&l.last&&['saved','skipped','interrupted'].includes(l.last.phase)?l:null;},180000);
    report.learningRequests=(await call('/api/serving')).requests.filter(r=>r.endpoint==='/skill-learning');
    assert.ok(report.learningRequests.some(r=>r.state==='done'&&r.instanceId===report.instanceId&&r.tokensOut>0));
    if(report.learning.last.phase==='skipped'&&report.learning.last.reason==='not-reusable'){report.checks.learning='NOT_REUSABLE';report.checks.autoApply='NOT_REUSABLE';}
    else {assert.equal(report.learning.last.phase,'saved');const skills=await call('/api/skills');const entries=Array.isArray(skills)?skills:skills.skills;report.learnedSkill=entries.find(s=>s.id===report.learning.last.skill.id);assert.ok(report.learnedSkill);const slug=report.learnedSkill.id.replace(/^my-/,'');assert.match(slug,/^[a-z0-9-]+$/);report.learnedSkillFile=artifact(join(home,'data/skills',slug,'SKILL.md'));report.checks.learning='PASS';stage='autoApply';await turn('autoApply','Apply the saved '+report.learning.last.skill.name+' procedure to explain how to repeat the inventory reconciliation. Describe the steps only; do not save a second audit.');report.checks.autoApply='PASS';}
    checkpoint();stage='btw';
    await until(async()=>!(await call('/api/skills/learning')).running&&(await call('/api/chat/active')).turns.length===0,180000);
    await call('/api/agent/tools','PUT',{groups:{helpers:true}});
    const btwId=await makeChat('btw');const btw=await stream('btw','/api/chat/btw',{sessionId:btwId,message:'What does an inventory version mismatch mean? One sentence.',history:[],nativeModelId:report.instanceId},'btw',180000);
    const childFrame=btw.find(f=>f.type==='btw-agent');assert.ok(childFrame&&childFrame.childSessionId!==childFrame.parentSessionId);const bh=observeHeaders('btw',[],childFrame.childSessionId);assert.ok(bh.some(h=>h.maxTokens===1536));report.checks.btw='PASS';checkpoint();
    stage='research';
    const researchSettings=await call('/api/agent/tools','PUT',{groups:{search:true,fetch:true,helpers:false,skills:false},custom:{[customId]:false}});checkSettings(researchSettings,customId,true);report.researchToolSettings=researchSettings;
    const rust=json(join(APP,'harness/rustxng/SOURCE.json'));assert.equal(rust.profile,'full');assert.ok(JSON.stringify(rust).includes('0.31.1'));report.rustxng=rust;
    const workspaceBefore=inventory(ws);const researchFrames=await turn('research','/deepresearch Compare the definitions of UTC and International Atomic Time using ordinary public standards or observatory sources. Use at least three distinct searches and read four distinct relevant public pages. Separate documented statements from uncertainty, cite source URLs, and do not write files.','research',420000);
    const tools=researchFrames.filter(f=>f.type==='agent-tool'&&f.tool.state==='done').map(f=>f.tool);const end=researchFrames.findLast(f=>f.type==='research');assert.ok(end);
    report.research={queries:tools.filter(t=>t.name==='web_search').map(t=>t.args.query),urls:tools.filter(t=>t.name==='web_fetch').map(t=>t.args.url),sources:end.sources,terminal:end,workspaceBefore,workspaceAfter:inventory(ws)};assert.ok(new Set(report.research.queries).size>=3&&new Set(report.research.urls).size>=4);assert.deepEqual(report.research.workspaceAfter,workspaceBefore);report.checks.research='PASS';checkpoint();
    stage='memoryStore';
    await until(async()=>!(await call('/api/skills/learning')).running,180000);
    async function memoryAccess(){const value=await call('/api/hakua/access','POST',{});assert.equal(resolve(value.profileRoot),join(home,'data/hakua-memory/local'));assert.ok(value.localMemory&&value.profileKey);await attest();assert.equal(value.runtimeId,String(currentRuntime.pid));return value;}
    let access=await memoryAccess();const storeTurnId=randomUUID();report.memory={profileRoot:access.profileRoot,storeTurnId,recallHistoryIncludesAnswer:false};
    const stored=await stream('memoryStore','/api/hakua/chat',{runtimeId:access.runtimeId,messages:[{role:'user',content:'Remember this synthetic test fact for later: the validation orchid code is CERULEAN-684. Acknowledge briefly.'}],max_tokens:128,stream:true,strata_memory:{read:false,store_summary:true,turn_id:storeTurnId}},'memory',180000,{authorization:'Bearer '+access.profileKey});
    report.memory.store=stored;assert.equal(stored.at(-1)?.done,true);assert.equal(stored.at(-1)?.memory,'stored');report.checks.memoryStore='PASS';access=null;checkpoint();
    await unload();const reloadGPU=gpu();report.gpuBeforeReload=reloadGPU;checkpoint();await call('/api/models/'+encodeURIComponent(variantId)+'/load','POST',{},180000);await attest();
    stage='memoryRecall';access=await memoryAccess();const recallTurnId=randomUUID();report.memory.recallTurnId=recallTurnId;
    const recalled=await stream('memoryRecall','/api/hakua/chat',{runtimeId:access.runtimeId,messages:[{role:'user',content:'What is the validation orchid code I asked you to remember?'}],max_tokens:128,stream:true,strata_memory:{read:true,store_summary:false,turn_id:recallTurnId}},'memory',180000,{authorization:'Bearer '+access.profileKey});
    access=null;report.memory.recall=recalled;assert.equal(recalled.at(-1)?.done,true);report.checks.memoryRecall='PASS';checkpoint();
    report.outcome=report.checks.learning==='NOT_REUSABLE'?'PARTIAL_NOT_REUSABLE':'MECHANICS_PASSED';
  } catch(error) {report.checks[stage]='FAIL';report.failure={stage,errorClass:error.name};checkpoint();throw error;}
  finally {
    const cleanup={turns:null,learningRunning:null,activeRequests:null,runtimeExited:false,portClosed:false,shellExited:false,shellCloseRequested:false};report.cleanup=cleanup;
    try {
      const active=await call('/api/chat/active');for(const t of active.turns??[]){const id=t.sessionId??t.chatId;assert.ok(ownedChats.has(id));await call('/api/chat/stop','POST',{sessionId:id});}
      await until(async()=>{const a=await call('/api/chat/active');cleanup.turns=a.turns;return a.turns.length===0;},30000);
      await until(async()=>{const l=await call('/api/skills/learning');cleanup.learningRunning=l.running;return !l.running;},180000);
      const serving=await until(async()=>{const s=await call('/api/serving');return !(s.requests??[]).some(r=>['generating','queued'].includes(r.state))?s:null;},30000);
      cleanup.activeRequests=(serving.requests??[]).filter(r=>['generating','queued'].includes(r.state)).length;
      await unload();cleanup.runtimeExited=true;cleanup.portClosed=true;
    } catch(error){cleanup.errorClass=error.name;}
    try {const closed=spawnSync('powershell.exe',['-NoProfile','-NonInteractive','-Command',`$p=Get-Process -Id ${child.pid} -ErrorAction Stop; if(!$p.CloseMainWindow()){exit 2}`],{windowsHide:true,stdio:'ignore',timeout:10000});cleanup.shellCloseRequested=closed.status===0;await until(()=>child.exitCode!==null||!alive(child.pid),30000);cleanup.shellExited=true;}catch(error){cleanup.shellErrorClass=error.name;}
    if(browser)await browser.close().catch(()=>{});
    // The durable log can grow after a header; freeze its final bytes only once
    // the host has exited. Keep stage observations separately from final pins.
    if(cleanup.shellExited)for(const h of report.headers)h.fileSHA256=hash(h.file);
    report.checks.cleanup=cleanup.turns?.length===0&&cleanup.learningRunning===false&&cleanup.activeRequests===0&&cleanup.runtimeExited&&cleanup.portClosed&&cleanup.shellExited&&cleanup.shellCloseRequested?'PASS':'FAIL';checkpoint();
  }
  assert.equal(report.checks.cleanup,'PASS');assert.equal(hash(inputPath),expectedSHA);
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  if(process.argv[2]!=='--execute')console.log(JSON.stringify({execute:false,entry:'real-model.py',actualModel:'NOT_RUN'}));
  else execute(resolve(process.argv[3]??''),process.argv[4]??'').catch(error=>{console.error(JSON.stringify({failed:true,errorClass:error.name,rawDiagnosticsSuppressed:true}));process.exitCode=1;});
}
