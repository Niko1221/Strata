import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import {mkdtemp,writeFile,readFile,rm,readdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {existsSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {createOwnerApi,localApiUrl} from './owner_api.mjs';

const id='example-physical-model',choices=[8192,131072,204800];
async function listen(server){await new Promise(r=>server.listen(0,'127.0.0.1',r));return 'http://127.0.0.1:'+server.address().port;}
const send=(res,value)=>{res.writeHead(200,{'content-type':'application/json'});res.end(JSON.stringify(value));};
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'strata-owner-api-')),requests=[],metadata=[];
 let live={id,context:131072,loaded:false};
 const model=http.createServer((req,res)=>{
  requests.push({path:req.url,method:req.method,headers:req.headers});
  if(req.url==='/health')send(res,{status:'ok',model:live.id,max_context:live.context,loaded:live.loaded});
  else if(req.url==='/v1/status')send(res,{service:'strata',model:live.id,context:{native:live.context},cache_max_tokens:live.context,loaded:live.loaded,activity:{in_flight:0}});
  else send(res,{data:[{id:live.id,status:{value:live.loaded?'loaded':'unloaded'}}]});
 });
 const base=await listen(model)+'/v1';
 let owner={schema:1,captured:new Date().toISOString(),apiBaseUrl:base,modelId:id,loaded:false,api:'available',phase:'ready',model:'unloaded',active:0,busy:false,
  selected:{profile:'Example Model',context:131072,idle:1800},effective:{profile:'Example Model',context:131072,idle:1800}};
 const statePath=join(root,'extension-status.json');
 const save=async patch=>{owner={...owner,...patch};await writeFile(statePath,JSON.stringify(owner));};await save({});
 let connections=[{id:'saved',kind:'custom',name:'Example Local API',baseUrl:base,protocol:'openai-chat',apiKeySet:true,
  models:[{id,name:'Old Name',contextMax:8192},{id:'unrelated',name:'Keep Me',contextMax:32768}]}];
 let preservedKey='fixture-only-key';
 const api=createOwnerApi({ownerDirectory:root,baseUrl:base,modelLabels:{[id]:{name:'Example Qwen',variant:'Example GGUF · IQ3_XXS'}},apiKeyEnv:'STRATA_EXTENSION_TEST_KEY'});
 const app=http.createServer(async(req,res)=>{
  if(req.url==='/api/native'){send(res,{catalog:{contexts:choices}});return;}
  if(req.url==='/api/model-providers'){
   if(req.method==='POST'){
    let raw='';for await(const part of req)raw+=part;
    const body=JSON.parse(raw);metadata.push({body,headers:req.headers});
    if(body.apiKey)preservedKey=body.apiKey;
    connections=[{...body,apiKeySet:true}];delete connections[0].apiKey;
   }
   send(res,{connections});return;
  }
  if(!await api.handle(req,res,new URL(req.url,'http://'+req.headers.host))){res.writeHead(404);res.end();}
 });
 const origin=await listen(app);
 t.after(async()=>{
  for(const server of [app,model]){server.closeAllConnections();await new Promise(r=>server.close(r));}
  await rm(root,{recursive:true,force:true});
 });
 const post=(action,body={},headers={})=>fetch(origin+'/api/local-model/'+action,{method:'POST',headers:{origin,'content-type':'application/json','x-strata-desktop':'local-model',...headers},body:JSON.stringify({expectedModel:id,...body})});
 return {root,base,origin,requests,metadata,save,post,get:()=>fetch(origin+'/api/local-model').then(r=>r.json()),
  setLive:patch=>{live={...live,...patch};},connections:()=>connections,key:()=>preservedKey};
}

test('arbitrary physical IDs and loopback ports use the live catalog and never start a model on read',async t=>{
 const f=await fixture(t),data=await f.get();
 assert.equal(data.available,true);assert.equal(data.model.id,id);assert.equal(data.model.name,'Example Qwen');
 assert.equal(data.model.apiBaseUrl,f.base);assert.deepEqual(data.context.choices,choices);
 assert.equal(data.context.effective,131072);assert.equal(data.model.loaded,false);
 assert.deepEqual(await readdir(f.root),['extension-status.json']);assert.ok(f.requests.every(r=>r.method==='GET'));
});
test('context, idle, preload and unload publish one identity-bound command without bypassing the owner',async t=>{
 for(const [action,body,value] of [['context',{tokens:204800},204800],['idle',{seconds:0},0],['load',{},undefined],['unload',{},undefined]]){
  const f=await fixture(t),response=await f.post(action,body);assert.equal(response.status,202);
  const result=await response.json(),command=JSON.parse(await readFile(join(f.root,'extension-request.json'),'utf8'));
  assert.equal(command.schema,1);assert.equal(command.expectedModel,id);assert.equal(command.action,action);assert.equal(command.value,value);
  assert.equal(command.id,result.requestId);assert.match(command.id,/^[a-f0-9-]{36}$/);
  const bytes=await readFile(join(f.root,'extension-request.json'));
  assert.equal((await f.post(action,body)).status,409);
  assert.deepEqual(await readFile(join(f.root,'extension-request.json')),bytes);
  assert.ok(f.requests.every(r=>r.method==='GET'),'No direct HTTP load/unload bypass');
 }
});
test('a claimed running request blocks another command and a changed model cannot receive a stale click',async t=>{
 const f=await fixture(t);await writeFile(join(f.root,'extension-running.json'),'{}');
 assert.equal((await f.post('context',{tokens:204800})).status,409);
 await rm(join(f.root,'extension-running.json'));
 assert.equal((await f.post('load',{expectedModel:'old-model'})).status,409);
 assert.equal((await f.post('idle',{seconds:60})).status,400);
 assert.equal((await f.post('context',{tokens:200000})).status,400);
 assert.equal((await f.post('load',{permissions:'changed'})).status,400);
 assert.deepEqual(await readdir(f.root),['extension-status.json']);
});
test('stale, disabled, mismatched or busy ownership is never treated as permission to control a server',async t=>{
 for(const patch of [{captured:new Date(Date.now()-20000).toISOString()},{schema:2},{modelId:'foreign'},
  {apiBaseUrl:'http://127.0.0.1:1/v1'},{phase:'draining'},{active:1},{busy:true}]){
  const f=await fixture(t);await f.save(patch);assert.notEqual((await f.post('load')).status,202);
  assert.deepEqual(await readdir(f.root),['extension-status.json']);
 }
});
test('metadata inconsistency and unknown activity disable controls instead of fabricating effective state',async t=>{
 const f=await fixture(t);f.setLive({context:65536});let data=await f.get();assert.equal(data.context.effective,null);
 assert.equal((await f.post('load')).status,503);
 f.setLive({context:131072});await f.save({loaded:null});data=await f.get();assert.equal(data.model.loaded,null);
 assert.equal((await f.post('load')).status,503);
});
test('cross-origin requests, missing mutation markers and invalid transport cannot enqueue a command',async t=>{
 const f=await fixture(t);
 assert.equal((await f.post('load',{}, {origin:'https://example.com'})).status,403);
 assert.equal((await f.post('load',{}, {'x-strata-desktop':''})).status,403);
 assert.equal((await f.post('load',{}, {'content-type':'text/plain'})).status,415);
 assert.deepEqual(await readdir(f.root),['extension-status.json']);
});
test('Use In Chat retains authentication and unrelated models while confirming exact display metadata',async t=>{
 const f=await fixture(t);process.env.STRATA_EXTENSION_TEST_KEY='model-server-fixture-key';
 t.after(()=>delete process.env.STRATA_EXTENSION_TEST_KEY);
 const response=await f.post('connection'),result=await response.json();assert.equal(response.status,200);
 assert.deepEqual(result.selection,{connectionId:'saved',modelId:id});assert.equal(result.connectionName,'Example Local API');
 assert.equal(result.contextMax,131072);assert.equal(result.label,'Example Qwen');assert.equal(f.key(),'fixture-only-key');
 assert.deepEqual(f.connections()[0].models[1],{id:'unrelated',name:'Keep Me',contextMax:32768});
 assert.equal(f.metadata[0].body.apiKey,'');assert.equal(f.metadata[0].headers.authorization,undefined);
 assert.ok(f.requests.every(r=>r.headers.authorization==='Bearer model-server-fixture-key'));
});
test('remote, credential-bearing and malformed URLs are rejected before any request',()=>{
 for(const value of ['https://127.0.0.1/v1','http://example.com/v1','http://key:secret@localhost:8080/v1','http://localhost:8080/v1?key=x','http://localhost:8080/v2'])
  assert.throws(()=>localApiUrl(value));
 assert.equal(localApiUrl('http://localhost:32123/v1/'),'http://localhost:32123/v1');
});

const traySource=process.env.STRATA_TRAY_SOURCE??join(import.meta.dirname,'../windows-tray');
test('Node commands are consumed and acknowledged once by the real Python Tray IPC',
 {skip:!existsSync(join(traySource,'managed/extension.py'))?'Requires The Optional Tray Candidate (#3365)':false},async t=>{
 const script=`import sys,json,queue
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
sys.path.insert(0,sys.argv[1])
from extension import ExtensionIPC
root=Path(sys.argv[2]);snapshot=json.loads((root/'extension-status.json').read_text())
config={'model_name':snapshot['modelId']}
owner=SimpleNamespace(current_config=config,selected=snapshot['selected'],runtime=None,settings=SimpleNamespace(generated=lambda *args:config))
ipc=ExtensionIPC(root,urlparse(snapshot['apiBaseUrl']).port);ipc.publish(snapshot,config)
commands=queue.Queue();ipc.claim(commands);request=commands.get_nowait();command=request.prepare(owner)
assert request.id==sys.argv[3]
if command['action']=='select':
 snapshot['selected'].update(command['selection']);snapshot['effective'].update(command['selection'])
else:snapshot['model']='loaded' if command['action']=='load' else 'unloaded'
ipc.finish(request);ipc.publish(snapshot,config);ipc.claim(commands)
assert commands.empty() and not (root/'extension-running.json').exists()
print(json.dumps(command))`;
 for(const [action,body] of [['context',{tokens:204800}],['idle',{seconds:0}],['load',{}],['unload',{}]]){
  const f=await fixture(t),response=await f.post(action,body);assert.equal(response.status,202);
  const result=await response.json();
  const command=JSON.parse(execFileSync(process.env.STRATA_TEST_PYTHON??'python.exe',
   ['-B','-c',script,join(traySource,'managed'),f.root,result.requestId],{encoding:'utf8',windowsHide:true,timeout:10000}));
  if(action==='context')f.setLive({context:body.tokens});
  if(['load','unload'].includes(action))f.setLive({loaded:action==='load'});
  const data=await f.get();assert.deepEqual(data.request,{id:result.requestId,state:'done'});
  if(action==='context'){assert.equal(command.selection.context,204800);assert.equal(data.context.effective,204800);}
  else if(action==='idle'){assert.equal(command.selection.idle,0);assert.equal(data.autoUnload.effective,0);}
  else {assert.equal(command.action,action);assert.equal(data.model.loaded,action==='load');}
  assert.deepEqual(await readdir(f.root),['extension-status.json']);
  assert.ok(f.requests.every(r=>r.method==='GET'),'The Companion Never Bypasses The Owner Lifecycle');
 }
});
