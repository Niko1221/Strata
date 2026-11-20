/** Observe the original Prism startup contract with bounded independent health probes. */
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {createReadStream,existsSync,mkdirSync,readFileSync,statSync,writeFileSync} from 'node:fs';
import http from 'node:http';
import {createServer} from 'node:net';
import {join} from 'node:path';

const base='H:/Strata/releases/repair-055-20261110-opencode',attempt=process.argv[2];
assert.match(attempt??'',/^[a-z0-9-]+$/);
const out=join(base,'prism-trace-'+attempt);assert.equal(existsSync(out),false);mkdirSync(out);
const previous='H:/Strata/releases/upstream-0213-20261109-opencode/app-native054/qa/installed-prism-r1/prism-e2e-profile';
const saved=JSON.parse(readFileSync(join(previous,'data/state.json'),'utf8'));
const model='H:/Strata/models/新しいフォルダー/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf';
const modelBefore=statSync(model),modelHash=createHash('sha256');for await(const block of createReadStream(model))modelHash.update(block);
assert.equal(modelHash.digest('hex'),'8cf793a36e62fb078ac05c2e1bdb2121ff94736dccbede7cf81b76346e868949');
Object.assign(process.env,{STRATA_HOME:out,STRATA_MODELS_DIR:join(out,'models'),STRATA_COMPONENTS_DIR:join(previous,'components'),STRATA_INSTALL_DIR:join(out,'absent-install'),STRATA_NO_ACQUIRE:'1',LLAMA_ARG_LOG_VERBOSITY:'4'});
const supervisor=await import('../../app/host/server/supervisor.mjs');
const socket=createServer();await new Promise(resolve=>socket.listen(0,'127.0.0.1',resolve));const port=socket.address().port;await new Promise(resolve=>socket.close(resolve));
const gpu=()=>execFileSync('C:/Windows/System32/nvidia-smi.exe',['--query-gpu=name,driver_version,memory.free,memory.total','--format=csv,noheader,nounits'],{encoding:'utf8',timeout:10000,windowsHide:true}).trim();
const config={beginner:saved.beginner,expert:Object.entries(saved.expert).map(([id,value])=>({id,value})),serving:{apiKey:null},configExtra:saved.configExtra};
const options={modelId:saved.current,modelPath:model,port,config,hardware:{cpu:{cores:6}},timeoutMs:120000};
const args=supervisor.buildArgs(options);assert.ok(args.includes('98304')&&args.filter(a=>a==='q8_0').length===2);
const result={ok:false,diagnosticOnly:true,releaseQualified:false,originalHealthBudgetMs:120000,originalPrismManifest:'9c729c73eb050dc466203fe149a27acd5be48763a33ea2c5d2b89f78e4ddcbaa',gpu:gpu(),args,cycles:[]};
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
function probe(){return new Promise(resolve=>{let done=false;const finish=value=>{if(!done){done=true;resolve(value);}};const req=http.get({host:'127.0.0.1',port,path:'/health',agent:false,timeout:1000},res=>{let text='';res.on('data',chunk=>{if(text.length<2048)text+=chunk;});res.on('end',()=>{let body;try{body=JSON.parse(text);}catch{}finish({http:res.statusCode,status:body?.status??null,loading:body?.error?.message==='Loading model'});});res.on('error',()=>finish({error:'response-error'}));});req.on('timeout',()=>req.destroy(new Error('probe timeout')));req.on('error',error=>finish({error:error.code??'probe-timeout'}));});}
function alive(pid){try{process.kill(pid,0);return true;}catch{return false;}}
try{
 for(let cycle=0;cycle<3;cycle++){
  assert.ok(Number(gpu().split(',')[2].trim())>=12000,'Release other GPU work before this serial diagnostic');
  const row={cycle,started:new Date().toISOString(),health:[],passed:false};result.cycles.push(row);const started=Date.now();let observing=true;
  const observation=(async()=>{while(observing){const value=await probe();const current=supervisor.status();row.health.push({ms:Date.now()-started,pid:current.pid,state:current.status,...value});await sleep(750);}})();
  try{
   const ready=await supervisor.start(options);row.readyAfterMs=Date.now()-started;row.pid=ready.pid;row.status=ready;
   const response=await fetch(`http://127.0.0.1:${port}/v1/chat/completions`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({model:'local',messages:[{role:'user',content:'What is 7 times 8? Reply with only the number.'}],temperature:0,max_tokens:32,stream:false,reasoning_effort:'none',chat_template_kwargs:{enable_thinking:false}}),signal:AbortSignal.timeout(45000)});
   assert.equal(response.status,200);const answer=await response.json();row.answer=answer.choices?.[0]?.message?.content??'';row.finishReason=answer.choices?.[0]?.finish_reason;assert.match(row.answer,/56/);row.passed=true;
  }catch(error){row.error=error.message;row.status=supervisor.status();}
  finally{observing=false;await observation;row.logs=supervisor.logs(400);writeFileSync(join(out,`cycle-${cycle}.json`),JSON.stringify(row,null,2));const pid=supervisor.status().pid;await supervisor.stop();row.stopped=supervisor.status();row.childStillAlive=pid?alive(pid):false;assert.equal(row.childStillAlive,false,'Production stop returned before its actual child exit');assert.equal((await probe()).http,undefined,'The old engine listener is still alive');writeFileSync(join(out,`cycle-${cycle}.json`),JSON.stringify(row,null,2));}
  if(!row.passed)break;
 }
 assert.equal(result.cycles.length,3);assert.ok(result.cycles.every(row=>row.passed));assert.equal(statSync(model).size,modelBefore.size);assert.equal(statSync(model).mtimeMs,modelBefore.mtimeMs);result.ok=true;
}catch(error){result.error=error.message;}
finally{await supervisor.stop();writeFileSync(join(out,'result.json'),JSON.stringify(result,null,2));console.log(JSON.stringify({ok:result.ok,diagnosticOnly:true,cycles:result.cycles.map(({cycle,readyAfterMs,passed,error})=>({cycle,readyAfterMs,passed,error})),error:result.error}));}
if(!result.ok)process.exitCode=1;
