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
