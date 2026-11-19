import test from 'node:test';
import assert from 'node:assert/strict';
import {EventEmitter} from 'node:events';
import {PassThrough} from 'node:stream';
import {createServer} from 'node:http';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {randomUUID} from 'node:crypto';
import {existsSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {createLlamaMemory,sameMemoryOrigin} from '../server/llama-memory.mjs';
import {privateLlamaArgs,PRIVATE_LLAMA_FLAGS} from '../server/llama-memory-policy.mjs';

test('private llama policy removes shared caches and CLI credentials; ordinary argv is untouched',()=>{
  const ordinary=['-m','synthetic.gguf','--slots','--api-key','synthetic-secret','--cache-ram=10','--cache-prompt'];
  const before=[...ordinary],args=privateLlamaArgs(ordinary);
  assert.deepEqual(ordinary,before);
  assert.deepEqual(args,['-m','synthetic.gguf',...PRIVATE_LLAMA_FLAGS]);
  assert.equal(args.includes('synthetic-secret'),false);
});
test('memory controls require the exact local origin and port',()=>{
  const check=(host,origin)=>sameMemoryOrigin({headers:{host,origin}});
  assert.equal(check('127.0.0.1:8000','http://127.0.0.1:8000'),true);
  assert.equal(check('localhost:8000',undefined),true);
  for(const origin of ['http://127.0.0.1:9000','https://example.com','http://localhost:8000','null'])
    assert.equal(check('127.0.0.1:8000',origin),false);
  assert.equal(check('example.com:8000',undefined),false);
});

async function fixture(t,{frames=[{event:{choices:[{delta:{content:'synthetic'}}]}},{done:true,memory:'stored'}],hold=false}={}){
  let starts=0,stopListener,worker,captured,command;
  const directory=resolve(tmpdir(),'strata-synthetic-memory-'+randomUUID());
  const target={pid:12345,port:23456,model:'synthetic',args:[...PRIVATE_LLAMA_FLAGS],server_key:'synthetic-operator-key'};
  const gateway=createLlamaMemory({
    directory,control:{dir:resolve(fileURLToPath(new URL('../../..',import.meta.url))),python:'synthetic-python'},
    enable:async()=>{starts++;return target;},target:()=>target,
    onBeforeStop:listener=>{stopListener=listener;return()=>{stopListener=null;};},
    env:{STRATA_HAKUA_LOCAL_KEY:'synthetic-profile-key',SYNTHETIC_API_KEY:'not-forwarded'},
    spawnWorker:(exe,args,options)=>{
      command={exe,args,options};
      worker=new EventEmitter();worker.stdin=new PassThrough();worker.stdout=new PassThrough();worker.stderr=new PassThrough();
      worker.kills=0;worker.kill=()=>{worker.kills++;worker.stdout.end();worker.emit('close',1);return true;};
      let input='';worker.stdin.on('data',c=>{input+=c;if(!input.includes('\n')||captured)return;
        captured=JSON.parse(input);
        if(!hold)setImmediate(()=>{for(const frame of frames)worker.stdout.write(JSON.stringify(frame)+'\n');worker.stdout.end();worker.emit('close',0);});
      });
      return worker;
    },
  });
  const server=createServer(async(req,res)=>{
    try{
      if(req.url==='/access'){res.setHeader('content-type','application/json');res.end(JSON.stringify(await gateway.access()));}
      else await gateway.chat(req,res);
    }catch{res.statusCode=409;res.end(JSON.stringify({error:'runtime unavailable'}));}
  });
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const base='http://127.0.0.1:'+server.address().port;
  t.after(async()=>{gateway.close();server.closeAllConnections();await new Promise(r=>server.close(r));});
  return {gateway,base,directory,target,get starts(){return starts;},get worker(){return worker;},
    get captured(){return captured;},get command(){return command;},stop:()=>stopListener?.()};
}
const request={runtimeId:'12345',messages:[{role:'user',content:'synthetic question'}],
  strata_memory:{read:true,store_summary:true,turn_id:'synthetic-turn'},stream:true};
async function chat(f,overrides={},key='synthetic-profile-key'){
  return fetch(f.base+'/chat',{method:'POST',headers:{'content-type':'application/json',authorization:'Bearer '+key},
    body:JSON.stringify({...request,...overrides})});
}
test('access only enables the owned model and does not create a store',async t=>{
  const f=await fixture(t);
  const [a,b]=await Promise.all([f.gateway.access(),f.gateway.access()]);
  assert.equal(f.starts,1);assert.deepEqual(a,b);
  assert.equal(a.runtimeId,'12345');assert.equal(a.localMemory,true);
  assert.equal(existsSync(f.directory),false);
});
test('worker receives operator config over stdin and streams only after principal auth',async t=>{
  const f=await fixture(t);await f.gateway.access();
  const response=await chat(f),text=await response.text();
  assert.equal(response.status,200);
  assert.match(text,/"memory":"stored"/);
  assert.equal(f.captured.config.server_key,'synthetic-operator-key');
  assert.equal(f.captured.key,'synthetic-profile-key');
  assert.equal(f.captured.request.runtimeId,undefined);
  assert.deepEqual(f.command.args,['-B','-m','serve.llama_memory']);
  assert.equal(JSON.stringify(f.command.args).includes('synthetic-profile-key'),false);
  assert.equal(f.command.options.env.SYNTHETIC_API_KEY,undefined);
  assert.equal(f.command.options.env.STRATA_HAKUA_LOCAL_KEY,'synthetic-profile-key');
});
test('wrong key, cross-origin request and stale model do not start workers',async t=>{
  const f=await fixture(t);await f.gateway.access();
  assert.equal((await chat(f,{},'wrong-key')).status,401);
  assert.equal((await chat(f,{runtimeId:'99999'})).status,409);
  const cross=await fetch(f.base+'/chat',{method:'POST',headers:{origin:'http://localhost:1'},body:'{}'});
  assert.equal(cross.status,403);assert.equal(f.worker,undefined);
});
test('model stop cancels an owned memory worker',async t=>{
  const f=await fixture(t,{hold:true});await f.gateway.access();
  const pending=chat(f);
  for(let i=0;i<100&&!f.worker;i++)await new Promise(r=>setTimeout(r,5));
  assert.ok(f.worker);f.stop();
  const response=await pending;await response.text();
  assert.ok(f.worker.kills>=1);
});
test('invalid worker request is a bounded visible error',async t=>{
  const f=await fixture(t,{frames:[{error:'Invalid memory request.',status:400}]});await f.gateway.access();
  const response=await chat(f);
  assert.equal(response.status,400);assert.match(await response.text(),/Invalid memory request/);
});

test('invalid JSON is rejected without reflecting private input or starting a worker',async t=>{
  const f=await fixture(t),access=await f.gateway.access();
  for(const body of ['synthetic-private-input {', 'null', '[]']) {
    const response=await fetch(f.base+'/chat',{method:'POST',headers:{
      'content-type':'application/json',authorization:'Bearer '+access.profileKey,
    },body});
    assert.equal(response.status,400);
    assert.equal((await response.text()).includes('synthetic-private-input'),false);
    assert.equal(f.worker,undefined);
  }
});
