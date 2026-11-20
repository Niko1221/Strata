/** Authenticated, per-turn worker for the owned PrismML / TurboQuant runtime. */
import {spawn} from 'node:child_process';
import {createHash,timingSafeEqual} from 'node:crypto';
import {existsSync} from 'node:fs';
import {join} from 'node:path';
import {createInterface} from 'node:readline';
import {desktopMemoryIdentity,DESKTOP_MEMORY_KEY_ENV} from './hakua-desktop.mjs';
import {controlEnvironment} from './control-runtime.mjs';
import {json,readBody,error} from './util.mjs';

const digest=value=>createHash('sha256').update(value).digest();
export function sameMemoryOrigin(req) {
  try {
    const host=new URL('http://'+req.headers.host);
    if(!['127.0.0.1','localhost','[::1]'].includes(host.hostname))return false;
    return !req.headers.origin || new URL(req.headers.origin).origin===host.origin;
  } catch{return false;}
}
export function createLlamaMemory({directory,control,enable,target,onBeforeStop,spawnWorker=spawn,env=process.env,timeoutMs=600000}) {
  let identity=null,opening=null;
  const active=new Set();
  const cancelAll=()=>{for(const worker of active)worker.kill();};
  const detach=onBeforeStop?.(cancelAll);
  const ready=()=>{
    if(!control?.python||!existsSync(join(control.dir,'serve','llama_memory.py')))
      throw new Error('This installation needs the updated Hakua Control component.');
  };
  async function access() {
    if(opening)return opening;
    opening=(async()=>{
      ready();
      const localIdentity=desktopMemoryIdentity({directory,env});
      const runtime=await enable();
      identity=localIdentity;
      return {...identity,localMemory:true,runtimeId:String(runtime.pid)};
    })().finally(()=>{opening=null;});
    return opening;
  }
  async function chat(req,res) {
    res.removeHeader('access-control-allow-origin');
    if(!sameMemoryOrigin(req))return error(res,403,'Open memory chat in the local Strata app.');
    const given=/^Bearer ([^\s]+)$/.exec(req.headers.authorization??'')?.[1]??'';
    if(!identity||!timingSafeEqual(digest(given),digest(identity.profileKey)))
      return error(res,401,'A valid Hakua memory key is required.');
    ready();
    let body;
    try { body=await readBody(req,512*1024); }
    catch { return error(res,400,'Memory chat needs a valid JSON request within 512 KiB.'); }
    if(!body||typeof body!=='object'||Array.isArray(body))return error(res,400,'Memory chat needs a JSON object.');
    let runtime;
    try { runtime=target(); }
    catch { return error(res,409,'The loaded model stopped. Load it and reopen memory chat.'); }
    if(String(runtime.pid)!==body.runtimeId)return error(res,409,'The loaded model changed. Close and reopen memory chat.');
    if(active.size)return error(res,409,'Wait for this memory turn to finish or stop it.');
    const {runtimeId,...request}=body;
    const childEnv=controlEnvironment(control,{env});
    for(const name of Object.keys(childEnv))if(/(?:API_KEY|TOKEN|SECRET|PASSWORD|ATTEST_KEY)$/.test(name))delete childEnv[name];
    childEnv[DESKTOP_MEMORY_KEY_ENV]=identity.profileKey;
    childEnv.PYTHONDONTWRITEBYTECODE='1';
    const worker=spawnWorker(control.python,['-B','-m','serve.llama_memory'],{
      cwd:control.dir,env:childEnv,windowsHide:true,stdio:['pipe','pipe','pipe'],
    });
    active.add(worker);
    const config={...runtime,hakua_harness:{enabled:true,users:[{
      principal:'local',key_env:DESKTOP_MEMORY_KEY_ENV,root:identity.profileRoot,
    }]}};
    const timer=setTimeout(()=>fail('The memory turn timed out. Stop and retry with a shorter message.'),timeoutMs);
    let ended=false;
    function finish(){if(ended)return;ended=true;clearTimeout(timer);active.delete(worker);if(!res.writableEnded)res.end();}
    function fail(message,status=502){
      if(ended)return;
      if(!res.headersSent)json(res,status,{error:message});
      else res.write(JSON.stringify({error:message})+'\n');
      finish();worker.kill();
    }
    const closed=()=>{if(!res.writableEnded){finish();worker.kill();}};
    res.on('close',closed);
    req.on('aborted',closed);
    worker.on('error',()=>fail('The local memory worker could not start.'));
    worker.stdin.on('error',()=>fail('The local memory worker stopped before accepting the request.'));
    worker.stderr.resume(); // Never expose private worker diagnostics or conversations.
    const lines=createInterface({input:worker.stdout,crlfDelay:Infinity});
    lines.on('line',line=>{
      if(ended)return;
      if(line.length>2*1024*1024)return fail('The local memory response was too large.');
      let value;try{value=JSON.parse(line);}catch{return fail('The local memory worker returned an invalid response.');}
      if(value.error)return fail(value.error,value.status===400?400:502);
      if(!res.headersSent)res.writeHead(200,{'content-type':'application/x-ndjson; charset=utf-8','cache-control':'no-store','x-content-type-options':'nosniff'});
      if(!res.write(JSON.stringify(value)+'\n')){worker.stdout.pause();res.once('drain',()=>worker.stdout.resume());}
      if(value.done===true){finish();worker.kill();}
    });
    worker.once('close',()=>{if(!ended)fail('The local memory turn ended before completion.');});
    worker.stdin.write(JSON.stringify({config,request,key:identity.profileKey})+'\n');
  }
  return {access,chat,cancelAll,close(){cancelAll();detach?.();}};
}
