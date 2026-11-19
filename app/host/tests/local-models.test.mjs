import {verifiedLocalRuntime} from './helpers/verified-local-runtime.mjs';
import { ROOMY_HOST } from './helpers/hardware.mjs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createServer } from 'node:net';
import { readGGUF } from '../server/gguf.mjs';
import { browseLocalModels, inspectLocalModel, createLocalModels } from '../server/local-models.mjs';
import { buildArgs } from '../server/supervisor.mjs';
import { ggufFixture } from './helpers/local-gguf.mjs';
const cwd=fileURLToPath(new URL('..',import.meta.url)),pause=ms=>new Promise(r=>setTimeout(r,ms));
function fixture(t){const root=mkdtempSync(join(tmpdir(),'strata-local-model-'));t.after(()=>rmSync(root,{recursive:true,force:true}));return root;}

test('generic GGUF metadata uses the filename while saved custom names survive reinspection and restart',t=>{
 const root=fixture(t),file=join(root,'Hikari-PQ2_0.gguf'),data=join(root,'data');
 ggufFixture(file,{name:'Hf'});
 const preview=inspectLocalModel(file);assert.equal(preview.variant.name,'Hikari-PQ2_0');
 const service=createLocalModels(data);service.add(preview);
 const custom=service.inspect(file);custom.variant.name='My Hikari';custom.variant.label='My Hikari · Q4_K_M';custom.variant.displayNameExplicit=true;service.add(custom);
 assert.equal(service.inspect(file).variant.name,'My Hikari');
 assert.equal(createLocalModels(data).inspect(file).variant.name,'My Hikari');
 custom.variant.name='Hf';service.add(custom);
 assert.equal(createLocalModels(data).inspect(file).variant.name,'Hf','an explicit user label is honored');
 custom.variant.displayNameExplicit=false;service.add(custom);
 assert.equal(createLocalModels(data).list()[0].name,'Hikari-PQ2_0','legacy placeholders recover on restart');
 ggufFixture(file,{name:'Publisher model name'});
 assert.equal(inspectLocalModel(file).variant.name,'Publisher model name');
});

test('Prism PQ2_0 uses its group-128 layout and refuses truncated weights',t=>{
 const root=fixture(t),file=join(root,'Prism-PQ2_0.gguf');
 ggufFixture(file,{fileType:141,tensorType:142,tensorElements:128,tensorBytes:34});
 assert.equal(inspectLocalModel(file).variant.quant,'PQ2_0');
 assert.equal(readGGUF(file).parameters,128);
 ggufFixture(file,{fileType:142,tensorType:142,tensorElements:128,tensorBytes:34,truncate:1});
 assert.throws(()=>readGGUF(file),/incomplete/);
 ggufFixture(file,{fileType:143,tensorType:143,tensorElements:128,tensorBytes:28});
 assert.equal(inspectLocalModel(file).variant.quant,'PTQ1_0');
});

test('bounded reader gets real metadata and rejects fake extensions, partial tensors, unsupported versions and duplicate metadata',t=>{
 const root=fixture(t),file=join(root,'huihui.gguf');ggufFixture(file,{version:2});const info=readGGUF(file);assert.equal(info.metadata['general.architecture'],'qwen3next');assert.equal(info.metadata['tokenizer.ggml.tokens'].count,1);assert.equal(info.parameters,16);
 const preview=inspectLocalModel(file);assert.equal(preview.variant.quant,'Q4_K_M');assert.equal(preview.variant.contextMax,32768);assert.equal(preview.variant.layerCount,48);assert.equal(preview.variant.experts,128);assert.equal(preview.variant.kvKbPerToken,12);assert.equal(preview.variant.family,'Local models');
 assert.deepEqual(preview.variant.kvModes,['INT8','FP16']);
 assert.throws(()=>inspectLocalModel(root),/\.gguf|folder/);writeFileSync(file,'not a model');assert.throws(()=>inspectLocalModel(file),/truncated|not a GGUF/);
 ggufFixture(file,{truncate:10});assert.throws(()=>readGGUF(file),/incomplete/);
 ggufFixture(file,{version:1});assert.throws(()=>readGGUF(file),/version/);
 ggufFixture(file,{extra:[['general.name',8,'duplicate']]});assert.throws(()=>readGGUF(file),/duplicate/);
 ggufFixture(file,{arch:'clip'});assert.throws(()=>inspectLocalModel(file),/architecture.*not a text-generation model/);
 ggufFixture(file,{omit:['general.architecture']});assert.throws(()=>inspectLocalModel(file),/missing or invalid model architecture/);
 for(const arch of ['deepseek2','gpt-oss','future_text_model']){
  ggufFixture(file,{arch});const custom=inspectLocalModel(file);
  assert.equal(custom.variant.local.architecture,arch);
  assert.equal(custom.variant.local.path,file);
 }
 ggufFixture(file,{arch:'future_text_model',omit:['future_text_model.block_count','future_text_model.embedding_length','future_text_model.attention.head_count','future_text_model.attention.head_count_kv']});
 const partial=inspectLocalModel(file);assert.equal(partial.variant.kvKbPerToken,null);assert.ok(partial.warnings.some(w=>w.includes('KV memory metadata is incomplete')));
 ggufFixture(file,{arch:'mamba',omit:['mamba.attention.head_count'],extra:[['mamba.attention.head_count',4,0]]});
 assert.equal(readGGUF(file).metadata['mamba.attention.head_count'],0);
 assert.equal(inspectLocalModel(file).variant.kvKbPerToken,null,'Mamba head count zero is a valid unknown KV estimate');
 ggufFixture(file,{arch:'future_text_model',omit:['future_text_model.attention.head_count'],extra:[['future_text_model.attention.head_count',9,['0','0']]]});
 assert.deepEqual(readGGUF(file).metadata['future_text_model.attention.head_count'],{count:2});
 assert.equal(inspectLocalModel(file).variant.kvKbPerToken,null,'per-layer head counts remain an unknown KV estimate');
 ggufFixture(file,{arch:'future_text_model',omit:['future_text_model.context_length']});
 const noContext=inspectLocalModel(file);assert.equal(noContext.variant.contextMax,4096);assert.ok(noContext.warnings.some(w=>w.includes('Context length is not declared')));
 ggufFixture(file,{tensorCount:0});assert.throws(()=>inspectLocalModel(file),/missing tensors/);
 ggufFixture(file,{context:512});assert.throws(()=>inspectLocalModel(file),/context_length/);
 ggufFixture(file,{context:2048,extra:[['qwen3next.attention.key_length',4,8],['qwen3next.attention.value_length',4,8]]});
 const narrow=inspectLocalModel(file);assert.equal(narrow.variant.contextMax,2048);assert.deepEqual(narrow.variant.kvModes,['FP16']);assert.ok(narrow.warnings.some(w=>w.includes('Choose FP16')));
 const oversized=Buffer.alloc(24);oversized.write('GGUF');oversized.writeUInt32LE(3,4);oversized.writeBigUInt64LE(2n**63n,8);writeFileSync(file,oversized);assert.throws(()=>readGGUF(file),/unsupported integer/);
});

test('Hikari-style qwen35 hybrid counts 16 attention / 48 recurrent for KV kb/token',t=>{
 const root=fixture(t),file=join(root,'Hikari-LoRA-fixed-scale-PQ2_0.gguf');
 ggufFixture(file,{
  arch:'qwen35',name:'Hikari fixture',context:262144,fileType:141,tensorType:142,tensorElements:128,tensorBytes:34,
  omit:['qwen35.block_count','qwen35.attention.head_count','qwen35.attention.head_count_kv','qwen35.embedding_length'],
  extra:[
   ['qwen35.block_count',4,64],['qwen35.embedding_length',4,2048],
   ['qwen35.attention.head_count',4,16],['qwen35.attention.head_count_kv',4,4],
   ['qwen35.attention.key_length',4,256],['qwen35.attention.value_length',4,256],
   ['qwen35.full_attention_interval',4,4],
  ],
 });
 const preview=inspectLocalModel(file);
 assert.equal(preview.variant.local.nFullAttention,16);
 assert.equal(preview.variant.local.nRecurrent,48);
 // Legacy FP16-element kb/token on attention layers only: 16*4*(256+256)/1024 = 32
 assert.equal(preview.variant.kvKbPerToken,32);
 assert.equal(preview.variant.memoryPlan.nFullAttention,16);
 assert.equal(preview.variant.memoryPlan.attentionKvGiB,4.25);
 assert.equal(preview.variant.memoryPlan.fits,null);
 assert.ok(preview.warnings.some(w=>/Hybrid Attention/i.test(w)));
});

test('choosing any shard finds the first, validates every shard and catches missing or mismatched sets',t=>{
 const root=fixture(t),first=join(root,'Orca-00001-of-00002.gguf'),second=join(root,'Orca-00002-of-00002.gguf');
 ggufFixture(first,{splitNo:0,splitCount:2,tensorName:'a.weight'});ggufFixture(second,{splitNo:1,splitCount:2,tensorName:'b.weight'});
 const model=inspectLocalModel(second);assert.equal(model.variant.local.shards,2);assert.ok(model.variant.local.path.endsWith('00001-of-00002.gguf'));assert.equal(model.variant.source.bytes,statSync(first).size+statSync(second).size);
 rmSync(second);assert.throws(()=>inspectLocalModel(first),/Keep every shard/);
 ggufFixture(second,{splitNo:0,splitCount:2});assert.throws(()=>inspectLocalModel(first),/matching set/);
 ggufFixture(second,{splitNo:1,splitCount:2,arch:'llama'});assert.throws(()=>inspectLocalModel(first),/different architectures/);
 rmSync(first);assert.throws(()=>inspectLocalModel(second),/first shard/);
});

test('a 2K custom GGUF keeps its original path and context in the llama-server command',t=>{
 const root=fixture(t),file=join(root,'small.gguf');ggufFixture(file,{arch:'future_text_model',context:2048});
 const model=inspectLocalModel(file).variant;
 const args=buildArgs({modelPath:model.local.path,config:{beginner:{contextLength:'2k'},expert:[]},hardware:{cpu:{cores:4}}});
 assert.equal(args[args.indexOf('-m')+1],file);
 assert.equal(args[args.indexOf('-c')+1],'2048');
});

test('folder browsing shows only folders/GGUFs; reference persistence, deduplication and removal preserve original weights',t=>{
 const root=fixture(t),file=join(root,'Orca.gguf');mkdirSync(join(root,'subfolder'));writeFileSync(join(root,'private.txt'),'keep');writeFileSync(join(root,'.hidden.gguf'),'ignore');const bytes=ggufFixture(file);
 const listing=browseLocalModels(root);assert.deepEqual(listing.entries.map(e=>e.name),['subfolder','Orca.gguf']);
 const service=createLocalModels(join(root,'data')),preview=inspectLocalModel(file);service.add(preview);service.add(preview);assert.equal(service.list().length,1);assert.equal(service.ready(service.list()[0]),true);
 const restarted=createLocalModels(join(root,'data'));assert.equal(restarted.list()[0].id,preview.variant.id);
 renameSync(file,join(root,'moved.gguf'));assert.equal(restarted.ready(restarted.list()[0]),false);renameSync(join(root,'moved.gguf'),file);
 restarted.remove(preview.variant.id);assert.deepEqual(readFileSync(file),bytes);assert.equal(readFileSync(join(root,'private.txt'),'utf8'),'keep');
});

test('native library uses the exact imported path and keeps its reference across directory changes',t=>{
 const root=fixture(t),file=join(root,'real-path.gguf');ggufFixture(file);const next=join(root,'next');mkdirSync(next);
 const result=spawnSync(process.execPath,['--input-type=module','-e',`
 import assert from 'node:assert/strict';import fs from 'node:fs';
 const catalog=await import('./server/catalog.mjs'),local=await import('./server/local-models.mjs'),models=await import('./server/models.mjs'),store=await import('./server/store.mjs');
 const preview=local.inspectLocalModel(${JSON.stringify(file)});catalog.localModels.add(preview);models.localModelAdded();
 const v=catalog.variant(preview.variant.id);assert.equal(models.pathFor(v),v.local.path);assert.ok(models.isOnDisk(v.id));assert.equal(models.library().variants.find(row=>row.id===v.id).state,'installed');
 store.changeModelsDirectory(${JSON.stringify(next)});models.init({send(){}});assert.ok(models.isOnDisk(v.id));
 models.remove(v.id);assert.equal(catalog.variant(v.id),null);assert.ok(fs.existsSync(${JSON.stringify(file)}));
 `],{cwd,env:{...process.env,STRATA_HOME:join(root,'host'),STRATA_MODELS_DIR:join(root,'host','models'),STRATA_REAL_ENGINE:'1'},encoding:'utf8'});assert.equal(result.status,0,result.stderr);
});

test('a failed local backend leaves unchanged GGUF files ready for a corrected load',t=>{
 const root=fixture(t),file=join(root,'retry.gguf');ggufFixture(file);
 const expert=verifiedLocalRuntime(t,root,{realChild:true});
 const result=spawnSync(process.execPath,['--input-type=module','-e',`
 import assert from 'node:assert/strict';import {createServer} from 'node:net';
 const catalog=await import('./server/catalog.mjs'),local=await import('./server/local-models.mjs'),models=await import('./server/models.mjs'),store=await import('./server/store.mjs');
 const listen=createServer();await new Promise(r=>listen.listen(0,'127.0.0.1',r));const port=listen.address().port;await new Promise(r=>listen.close(r));
 store.update({expert:{'server.port':port}});models.init({send(){}});
 const preview=local.inspectLocalModel(${JSON.stringify(file)});catalog.localModels.add(preview);models.localModelAdded();
 const config={beginner:{contextLength:'2k'},expert:${JSON.stringify(expert)}};
 for(let attempt=0;attempt<2;attempt++){
  await assert.rejects(models.load(preview.variant.id,{config,hardware:{cpu:{cores:4}}}),/Engine exited early.*bad option: -m/);
  const row=models.library().variants.find(v=>v.id===preview.variant.id);
  assert.equal(row.state,'installed');assert.ok(row.error);assert.ok(models.isOnDisk(row.id));
 }
 `],{cwd,env:{...process.env,STRATA_HOME:join(root,'host'),STRATA_MODELS_DIR:join(root,'host','models'),STRATA_REAL_ENGINE:'1'},encoding:'utf8',timeout:15_000});
 assert.equal(result.status,0,result.stderr);
});

async function freePort(){const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));const port=server.address().port;await new Promise(r=>server.close(r));return port;}
test('HTTP import leaves the running model intact, loads through normal controls, protects active references and preserves files on removal',{timeout:30000},async t=>{
 const root=fixture(t),home=join(root,'host'),file=join(root,'huihui-Orca.gguf');mkdirSync(join(home,'data'),{recursive:true});ggufFixture(file);
 writeFileSync(join(home,'data','state.json'),JSON.stringify({setup:{complete:true,step:4,flowVersion:2},installed:{'qwen38-q4_k_m':{simulated:true}},loaded:'qwen38-q4_k_m'}));
 const port=await freePort(),base=`http://127.0.0.1:${port}`,host=spawn(process.execPath,['server/index.mjs'],{cwd,env:{...process.env,STRATA_HOME:home,STRATA_MODELS_DIR:join(home,'models'),STRATA_REAL_ENGINE:'0',STRATA_TEST_HARDWARE:ROOMY_HOST,STRATA_PORT:String(port),STRATA_BIND:'127.0.0.1'},stdio:'ignore'});
 async function call(path,body,method){const response=await fetch(base+path,{method:method??(body===undefined?'GET':'POST'),headers:{'content-type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});return {status:response.status,body:await response.json()};}
 try {
 let ready=false;for(let i=0;i<100;i++){try{ready=(await call('/api/runtime')).status===200;if(ready)break;}catch{}await pause(100);}assert.ok(ready);
 const original=(await call('/api/runtime')).body.defaultId;
 assert.equal((await call('/api/local-models/inspect',{path:join(root,'missing.gguf')})).status,400);
 const added=await call('/api/local-models',{path:file,name:'My huihui version'});assert.equal(added.status,200,JSON.stringify(added.body));const id=added.body.variantId;
 assert.equal((await call('/api/runtime')).body.defaultId,original);assert.equal((await call('/api/local-models',{path:file})).body.variantId,id);
 const engine=(await call('/api/engine')).body,hardware=(await call('/api/hardware')).body;
 const config={beginner:{...engine.beginner,contextLength:'32k'},expert:engine.expert.map(p=>({id:p.id,value:p.id==='ctx.length'?'32K':p.value})),extra:{headroom:'shared'}};
 const loaded=await call('/api/runtime/load',{variantId:id,gpuIds:hardware.gpus.map(g=>g.id??`${g.vendor}:${g.index}`),config});assert.equal(loaded.status,200,JSON.stringify(loaded.body));assert.equal(loaded.body.state.instances.length,1);
 assert.equal((await call(`/api/models/${id}`,undefined,'DELETE')).status,409);
 ggufFixture(file,{name:'Changed metadata'});assert.equal((await call('/api/local-models',{path:file})).status,409);
 await call(`/api/runtime/${loaded.body.instance.id}/unload`,{});
 const reconnected=await call('/api/local-models',{path:file});assert.equal(reconnected.status,200);assert.equal(reconnected.body.variantId,id);
 assert.equal((await call(`/api/models/${id}`,undefined,'DELETE')).status,200);assert.ok(readFileSync(file).length>0);assert.equal((await call('/api/models')).body.variants.some(v=>v.id===id),false);
 assert.equal((await call('/api/runtime')).body.instances.length,0);
 } finally {host.kill('SIGTERM');await new Promise(r=>host.once('exit',r));}
});
test('the Flash-Next family\'s own architecture (qwen4exp) is accepted by the importer, not sent to llama-server (#4726)',t=>{
 const root=fixture(t),file=join(root,'flash-next.gguf');
 ggufFixture(file,{arch:'qwen4exp'});
 const info=inspectLocalModel(file);
 assert.equal(info.variant.local.architecture,'qwen4exp');
});
