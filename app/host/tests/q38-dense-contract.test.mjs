import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {denseRunConfig} from '../server/q38-dense-contract.mjs';
import {nativeServingPolicy,servingSettingsView} from '../server/native-serving-policy.mjs';
import {NATIVE_FIELDS} from '../server/native-settings.mjs';
import {configureDesktopMemory} from '../server/hakua-desktop.mjs';
import {nativeFixture,finishJob} from './helpers/native-fixture.mjs';
const base=()=>({arch:'qwen38_dense',native_runner:'qwen38_dense',exe:'component:strata-engine',args:['--model',resolve('original.gguf')],output_language:'ja',dense_kv:'auto',max_context:4096,parallel:1,drafting:'off',mtp:false});
test('dense defaults and explicit long context propagate without borrowed flags',()=>{
 const cfg=denseRunConfig(base(),{context:131072,kv:'spin8_k8v4',values:{output_language:'model-default'}});
 assert.deepEqual(cfg.args,['--model',resolve('original.gguf'),'--ctx-size','131072','--kv','spin8_k8v4','--output-language','model-default']);
 assert.equal(cfg.max_context,131072);assert.equal(cfg.dense_kv,'spin8_k8v4');
 const policy=nativeServingPolicy(cfg);
 assert.equal(policy.id,'q38dense');assert.ok(policy.ordinaryKv.includes('auto'));
 const fields=servingSettingsView(cfg,policy,NATIVE_FIELDS).fields;
 assert.equal(fields.find(f=>f.id==='output_language').supported,true);
 assert.equal(fields.find(f=>f.id==='mtp').supported,false);
 const old=nativeServingPolicy({arch:'qwen4exp'});
 assert.deepEqual(old.ordinaryKv,['fp16','int8','q4_0','k8v4']);
 assert.equal(servingSettingsView({},old,NATIVE_FIELDS).fields.find(f=>f.id==='output_language').supported,false);
});
test('invalid settings refuse instead of changing requested model/context',()=>{
 for(const patch of [{max_context:0},{max_context:262145},{parallel:2},{drafting:'mtp'},{output_language:'zh'},{dense_kv:'turbo5'},{args:['--model',resolve('a.gguf'),'--gpu-layers','64']},{args:['--model',resolve('a.gguf'),'--model',resolve('b.gguf')]}])assert.throws(()=>denseRunConfig({...base(),...patch}));
});

for(const source of ['user','raw','edited','generated'])test(`dense ${source} argument precedence keeps matching scalar reports`,()=>{
 const cfg=base(),model=[...cfg.args],choices=['--ctx-size','131072','--kv','f16','--output-language','model-default'];
 const defaults=['--ctx-size','4096','--kv','auto','--output-language','ja'];
 if(source==='user')Object.assign(cfg,{args:[...model,...defaults],args_written:[...model,...defaults],user_args:choices});
 else if(source==='raw')cfg.args=[...model,...choices];
 else if(source==='edited')Object.assign(cfg,{args:[...model,...choices],args_written:[...model,...defaults],user_args:defaults});
 else Object.assign(cfg,{args:[...model,...defaults],args_written:[...model,...defaults],max_context:131072,dense_kv:'f16',output_language:'model-default'});
 const run=denseRunConfig(cfg);
 assert.deepEqual(run.args,[...model,...choices]);
 assert.deepEqual([run.max_context,run.dense_kv,run.output_language],[131072,'f16','model-default']);
 assert.deepEqual(denseRunConfig(run).args,run.args);
});

test('explicit raw options cannot bypass supported flags or value validation',()=>{
 for(const user_args of [['--ctx-size','0'],['--ctx-size','1e5'],['--kv','invalid'],['--output-language','zh'],['--gpu-layers','64']])assert.throws(()=>denseRunConfig({...base(),user_args}));
});
test('semantic memory remains enabled without invented native private KV flags',()=>{
 const cfg=base(),identity=configureDesktopMemory(cfg,{enabled:true,directory:resolve('fixture-private'),env:{}});
 assert.ok(identity.profileRoot);assert.equal(cfg.hakua_harness.enabled,true);
 assert.deepEqual(cfg.args,base().args);
});
test('ordinary native Host load carries the actual dense config and skips acquisitions',async()=>{
 const f=nativeFixture();
 try{
  const path=join(f.repo,'strata-iq2_xs.json'),exe=join(f.repo,'strata-q38.exe');writeFileSync(exe,'fixture');
  const cfg={...base(),exe,tokenizer:resolve('original.gguf'),model_name:'renamed Flash original',aliases:['kept-name']};writeFileSync(path,JSON.stringify(cfg));
  await f.control.refresh(true);
  const done=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs',kv:'auto',values:{output_language:'ja'}}).id);
  assert.equal(done.state,'done',done.error);
  const start=f.calls.find(c=>/serve[/\\]server\.py$/.test(c.args[0]??''));assert.ok(start);
  const run=JSON.parse(readFileSync(start.args[start.args.indexOf('--config')+1]));
  assert.equal(run.arch,'qwen38_dense');assert.equal(run.output_language,'ja');assert.equal(run.max_context,4096);
  assert.equal(run.args.includes('--prompt-cache'),false);assert.equal(run.args.includes('--mtp'),false);
  assert.equal(run.args.includes('--gpu-layers'),false);assert.equal(run.args.includes('--batch'),false);
  assert.ok(run.aliases.includes('kept-name'));
  assert.equal(f.rpcCalls.some(c=>c.name==='strata_install'),false);
 }finally{await f.cleanup();}
});

test('Host load saves effective explicit choices and reloads without scalar-default substitution',async()=>{
 const f=nativeFixture();
 try{
  const path=join(f.repo,'strata-iq2_xs.json'),exe=join(f.repo,'strata-q38.exe');writeFileSync(exe,'fixture');
  const choices=['--ctx-size','131072','--kv','f16','--output-language','model-default'];
  writeFileSync(path,JSON.stringify({...base(),exe,tokenizer:resolve('original.gguf'),user_args:choices}));
  await f.control.refresh(true);
  for(let load=0;load<2;load++){
   const done=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs',kv:'auto',values:{output_language:'ja'}}).id);
   assert.equal(done.state,'done',done.error);
   const start=f.calls.filter(c=>/serve[/\\]server\.py$/.test(c.args[0]??'')).at(-1);
   const run=JSON.parse(readFileSync(start.args[start.args.indexOf('--config')+1]));
   const saved=JSON.parse(readFileSync(path));
   for(const cfg of [run,saved]){
    assert.deepEqual([cfg.max_context,cfg.dense_kv,cfg.output_language],[131072,'f16','model-default']);
    assert.deepEqual(cfg.args.slice(-6),choices);
    assert.deepEqual(cfg.user_args.slice(-6),choices);
   }
   await f.control.unload(f.control.state().instances[0].id);
  }
 }finally{await f.cleanup();}
});

test('dense persistence preserves a newer explicit file edit made while loading',async()=>{
 const f=nativeFixture();
 try{
  const path=join(f.repo,'strata-iq2_xs.json'),exe=join(f.repo,'strata-q38.exe');writeFileSync(exe,'fixture');
  const initial={...base(),exe,tokenizer:resolve('original.gguf'),user_args:['--ctx-size','131072','--kv','f16','--output-language','model-default']};
  writeFileSync(path,JSON.stringify(initial));await f.control.refresh(true);f.setHealthWait(true);
  const job=f.control.operation('load',{model:'iq2_xs'});
  for(let n=0;n<400&&!f.hasPendingHealth();n++)await new Promise(r=>setTimeout(r,10));
  assert.ok(f.hasPendingHealth());
  const newer=['--ctx-size','98304','--kv','spin8_k8v4','--output-language','ja'];
  writeFileSync(path,JSON.stringify({...initial,user_args:newer,aliases:['newer-name']}));
  f.releaseHealth();const done=await finishJob(f.control,job.id);assert.equal(done.state,'done',done.error);
  const saved=JSON.parse(readFileSync(path));
  assert.deepEqual([saved.max_context,saved.dense_kv,saved.output_language],[98304,'spin8_k8v4','ja']);
  assert.deepEqual(saved.args.slice(-6),newer);assert.deepEqual(saved.aliases,['newer-name']);
 }finally{f.releaseHealth();await f.cleanup();}
});
