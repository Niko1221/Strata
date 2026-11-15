/** Production Host admission and launch builder, with process/network boundaries controlled. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { nativeFixture, finishJob } from './helpers/native-fixture.mjs';
import { nativeServingPolicy, servingSettingsView, assertServingSettings } from '../server/native-serving-policy.mjs';
import { NATIVE_FIELDS } from '../server/native-settings.mjs';

for (const kv of ['int8', 'q4_0', 'k8v4', 'fp16']) test(`Q38 explicit ${kv} reaches the actual launch config over saved/user KV`, async () => {
  const f = nativeFixture();
  try {
    const path = join(f.repo, 'strata-iq2_xs.json');
    const gguf = join(f.repo, 'own model.gguf'); writeFileSync(gguf, 'fixture header only');
    const original = { ...JSON.parse(readFileSync(path)), args: ['--native', gguf, '--kv', 'int8', '--kv', 'fp16'], user_args: ['--kv', 'k8v4'] };
    writeFileSync(path, JSON.stringify(original));
    await f.control.refresh(true);
    const done = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs', kv }).id);
    assert.equal(done.state, 'done', done.error);
    const start = f.calls.find(c => /serve[/\\]server\.py$/.test(c.args[0] ?? ''));
    const cfg = JSON.parse(readFileSync(start.args[start.args.indexOf('--config') + 1]));
    assert.equal(cfg.args[cfg.args.lastIndexOf('--kv') + 1], kv);
    assert.equal(cfg.args.filter(v => v === '--kv').length, 1);
    assert.equal(cfg.args[cfg.args.indexOf('--native') + 1], gguf);
    assert.equal(f.rpcCalls.some(c => c.name === 'strata_install'), false, 'KV-only changes must not prepare/download');
    const saved = JSON.parse(readFileSync(path));
    assert.equal(saved.args[saved.args.lastIndexOf('--kv') + 1], kv);
    assert.equal(saved.args[saved.args.indexOf('--native') + 1], gguf);
  } finally { await f.cleanup(); }
});

for (const values of [
  { cache_type_k: 'q8_0' }, { cache_type_v: 'q4_0' }, { cache_type_k: 'f32' }, { spin8_mode: 'off' },
  { spin8_mode: 'vector', spin8_native_config: 'capsule.json', spin8_native_sha256: 'a'.repeat(64) },
  { args: ['--cache-type-k', 'q8_0'] }, { args: ['-ctv', 'q4_0'] },
]) test(`Q38 rejects ${JSON.stringify(values)} before save/job/replacement`, async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const path = join(f.repo, 'strata-iq2_xs.json'), before = readFileSync(path, 'utf8');
    const calls = f.calls.length, jobs = f.control.state().jobs.length;
    assert.throws(() => f.control.save('iq2_xs', { values }), /serving|ordinary|capsule/i);
    assert.equal(readFileSync(path, 'utf8'), before);
    assert.throws(() => f.control.operation('load', { model: 'iq2_xs', values }), /serving|ordinary|capsule/i);
    assert.equal(f.control.state().jobs.length, jobs);
    assert.equal(f.calls.length, calls);
  } finally { await f.cleanup(); }
});

test('saved incompatible Q38 values are visible and block load until explicitly cleared', async () => {
  const f = nativeFixture();
  try {
    const path = join(f.repo, 'strata-iq2_xs.json');
    const config = { ...JSON.parse(readFileSync(path)), cache_type_k: 'q8_0', args: ['--cache-type-k', 'q8_0'], user_args: ['--cache-type-k', 'q8_0'] };
    writeFileSync(path, JSON.stringify(config)); await f.control.refresh(true);
    assert.equal(f.control.settings('iq2_xs').values.cache_type_k, 'q8_0');
    const view = await f.control.settingsKv('iq2_xs');
    assert.ok(view.settingsIssues?.some(i => i.field === 'cache_type_k'));
    assert.equal(view.fields.find(f => f.id === 'cache_type_k').supported, false);
    assert.throws(() => f.control.operation('load', { model: 'iq2_xs' }), /serving|ordinary|capsule/i);
    f.control.save('iq2_xs', { values: { cache_type_k: null } });
    assert.equal(f.control.settings('iq2_xs').values.cache_type_k, null);
    assert.equal((await f.control.settingsKv('iq2_xs')).settingsIssues.length, 0);
  } finally { await f.cleanup(); }
});

const cap={schema:1,family:'qwen35moe',layoutVersion:1,scaleSchema:'none',cacheTypesK:['f32','f16'],cacheTypesV:['f32'],flags:['--cache-type-k','--cache-type-v'],quality:'experimental-unqualified',binarySha256:'ab'.repeat(32)};
test('family policy uses saved adapter identity, no filename/GPU inference or capability union',()=>{
  assert.equal(nativeServingPolicy({arch:'deepseek41',exe:'strata-q36.exe',model_name:'Qwen3.8'},'qwen').id,'deepseek41');
  assert.equal(nativeServingPolicy({arch:'future',exe:'strata-q36.exe',model_name:'Qwen3.6'},'qwen').id,'unknown');
  assert.equal(nativeServingPolicy({exe:'strata-q36.exe',model_name:'Qwen3.6'},null).id,'unknown');
  assert.equal(nativeServingPolicy({arch:'future'},'qwen').runtimeArch,null);
  assert.equal(nativeServingPolicy({exe:'strata-q36.exe',model_name:'Qwen3.6'}).runtimeArch,null);
  const unknown=nativeServingPolicy({arch:'qwen35moe'}),verified=nativeServingPolicy({arch:'qwen35moe'},null,cap);
  assert.doesNotThrow(()=>assertServingSettings({cache_type_k:'f32'},unknown));
  assert.throws(()=>assertServingSettings({cache_type_k:'f16'},unknown),/not verified/);
  assert.doesNotThrow(()=>assertServingSettings({cache_type_k:'f16',cache_type_v:'f32'},verified));
  assert.throws(()=>assertServingSettings({cache_type_v:'f16'},verified),/not verified/);
  assert.deepEqual(servingSettingsView({},verified,NATIVE_FIELDS).fields.find(f=>f.id==='cache_type_v').choices,['f32']);
  for(const config of [{spin8_mode:'off'},{args:['--kv','int8']},{args:['--cache-type-k','f16','--cache-type-k','f16']},{args:['--cache-type-k','f32'],cache_type_k:'f16'}])
    assert.throws(()=>assertServingSettings(config,verified),/serving|malformed|verified/);
  assert.equal(nativeServingPolicy({arch:'qwen35moe'},null,{...cap,cacheTypesV:['turbo3']}).kvCapability,null);
});

test('DeepSeek packed controls retain routing 5/6 and reject generic Qwen KV/serving settings',async()=>{
  const f=nativeFixture();try{
    const path=join(f.repo,'strata-iq2_xs.json');writeFileSync(path,JSON.stringify({arch:'deepseek41',exe:'engine-dsv41/0.1.57-test/strata-dsv41.exe',args:[]}));
    await f.control.refresh(true);const view=await f.control.settingsKv('iq2_xs');
    assert.equal(view.servingPolicy.id,'deepseek41');assert.deepEqual(view.servingPolicy.ordinaryKv,[]);
    for(const field of ['cache_type_k','cache_type_v','spin8_mode'])assert.equal(view.fields.find(f=>f.id===field).supported,false);
    for(const choice of ['5','6'])assert.equal((await f.control.saveChecked('iq2_xs',{values:{ds_experts:choice}})).values.ds_experts,choice);
    for(const values of [{cache_type_k:'f16'},{drafting:'mtp'},{spin8_mode:'off'},{args:['--kv','int8']}])
      await assert.rejects(f.control.saveChecked('iq2_xs',{values}),/serving|packed/);
  }finally{await f.cleanup();}
});

test('HTTP save service negotiates Q36 without borrowing per-side formats; unknown retains FP32 only',async()=>{
  let capability=cap;const f=nativeFixture({q36CapabilityProbe:async()=>capability});try{
    const path=join(f.repo,'strata-iq2_xs.json');writeFileSync(path,JSON.stringify({arch:'qwen35moe',args:[]}));await f.control.refresh(true);
    assert.equal((await f.control.saveChecked('iq2_xs',{values:{cache_type_k:'f16'}})).values.cache_type_k,'f16');
    await assert.rejects(f.control.saveChecked('iq2_xs',{values:{cache_type_v:'f16'}}),/not verified/);
    capability=null;
    await assert.rejects(f.control.saveChecked('iq2_xs',{values:{parallel:1}}),/not verified/);
    assert.equal(f.control.settings('iq2_xs').values.cache_type_k,'f16','do not silently clear incompatible saved values');
    await f.control.saveChecked('iq2_xs',{values:{cache_type_k:null,cache_type_v:'f32'}});
    const done=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs'}).id);assert.equal(done.state,'done',done.error);
    const start=f.calls.find(c=>/serve[/\\]server\.py$/.test(c.args[0]??'')),cfg=JSON.parse(readFileSync(start.args[start.args.indexOf('--config')+1]));
    assert.equal(cfg.args.includes('--cache-type-v'),false,'legacy binary never receives typed cache flags');
  }finally{await f.cleanup();}
});

test('known invalid replacement has zero stop/start/profile/write/job effects on a ready packaged model',async()=>{
  const f=nativeFixture({packaged:true});try{
    await f.control.refresh(true);
    const done=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs',kv:'int8'}).id);assert.equal(done.state,'done',done.error);
    const path=join(f.repo,'strata-iq3_s.json'),before=readFileSync(path,'utf8'),files=readdirSync(f.repo),jobs=f.control.state().jobs.length,effects=[...f.effects],calls=f.calls.length;
    for(const args of [{values:{cache_type_k:'q8_0'}},{values:{spin8_mode:'vector',spin8_native_config:'x',spin8_native_sha256:'a'.repeat(64)}},{kv:'unknown'}])
      assert.throws(()=>f.control.operation('load',{model:'iq3_s',...args}),/serving/);
    assert.deepEqual(f.effects,effects);assert.equal(f.calls.length,calls);assert.equal(f.control.state().jobs.length,jobs);
    assert.equal(f.control.state().instances[0].status,'ready');assert.equal(readFileSync(path,'utf8'),before);assert.deepEqual(readdirSync(f.repo),files);
  }finally{await f.cleanup();}
});

test('changed ordinary KV survives packaged profile builder and metadata refresh; later edits still own future config',async()=>{
  const f=nativeFixture({packaged:true});try{
    await f.control.refresh(true);
    const path=join(f.repo,'strata-iq2_xs.json');
    const done=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs',kv:'q4_0'}).id);assert.equal(done.state,'done',done.error);
    const cfg=JSON.parse(readFileSync(f.launchConfigs.at(-1)));assert.equal(cfg.args[cfg.args.lastIndexOf('--kv')+1],'q4_0');
    assert.equal((await f.control.refresh(true)).installed.models[0].kv,'q4_0');
    f.setHealthWait(true);const job=f.control.operation('load',{model:'iq2_xs',kv:'k8v4'});
    for(let n=0;n<400&&!f.hasPendingHealth();n++)await new Promise(r=>setTimeout(r,10));assert.equal(f.hasPendingHealth(),true);
    const current=JSON.parse(readFileSync(path));writeFileSync(path,JSON.stringify({...current,args:['--kv','fp16'],user_args:['--kv','fp16'],sampling:{temperature:0.7}}));
    f.releaseHealth();assert.equal((await finishJob(f.control,job.id)).state,'done');
    const saved=JSON.parse(readFileSync(path));assert.equal(saved.args[saved.args.lastIndexOf('--kv')+1],'fp16');assert.equal(saved.sampling.temperature,0.7);
  }finally{await f.cleanup();}
});

test('preparation plan rejects incompatible saved/requested settings before installer callbacks or persistence',async()=>{
  const f=nativeFixture();try{
    await f.control.refresh(true);const effects=[...f.effects],calls=f.rpcCalls.length,path=join(f.repo,'strata-iq2_xs.json');
    const cfg={...JSON.parse(readFileSync(path)),args:['-ctv','q4_0']};writeFileSync(path,JSON.stringify(cfg));
    assert.equal(f.control.settings('iq2_xs').values.cache_type_v,'q4_0');
    await assert.rejects(f.control.plan({family:'qwen',model:'IQ2_XS',kv:'int8'}),/serving/);
    assert.throws(()=>f.control.operation('install',{family:'qwen',model:'IQ2_XS'}),/serving/);
    assert.equal(f.rpcCalls.length,calls);assert.deepEqual(f.effects,effects);assert.equal(f.control.state().jobs.length,0);
    f.control.save('iq2_xs',{values:{cache_type_v:null,args:cfg.args}});
    assert.equal(f.control.settings('iq2_xs').values.cache_type_v,null);
    assert.equal(JSON.parse(readFileSync(path)).args.includes('-ctv'),false);
  }finally{await f.cleanup();}
});

for (const identity of ['catalog', 'arch-alias']) test(`Q36 ${identity} identity negotiates saved FP16 through the actual load builder`, async () => {
  const probes=[]; const f=nativeFixture({packaged:true,q36CapabilityProbe:async exe=>{probes.push(exe);return cap;}});
  try {
    const path=join(f.repo,'strata-iq2_xs.json');
    f.catalog.families[0].family='qwen36';
    writeFileSync(path,JSON.stringify({model_name:'Neutral model',exe:'neutral-runtime.exe',parallel:1,args:['--max-context','4096'],...(identity==='arch-alias'?{arch:'qwen36'}:{})}));
    await f.control.refresh(true);
    const view=await f.control.settingsKv('iq2_xs');
    assert.equal(view.servingPolicy.id,'q36');assert.deepEqual(view.fields.find(f=>f.id==='cache_type_k').choices,['f32','f16']);
    await f.control.saveChecked('iq2_xs',{values:{cache_type_k:'f16'}});
    const before=probes.length;
    const done=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs'}).id);
    assert.equal(done.state,'done',done.error);
    assert.equal(probes.length,before+1,'load must independently negotiate the selected Q36 adapter');
    const cfg=JSON.parse(readFileSync(f.launchConfigs.at(-1)));
    assert.equal(cfg.arch,'qwen35moe','downstream server receives the canonical adapter, without name heuristics');
    assert.equal(cfg.args[cfg.args.indexOf('--cache-type-k')+1],'f16');
    assert.equal(cfg.q36_run.backend.family,'qwen36');assert.equal(cfg.q36_run.effective.kv_k,'f16');
    assert.equal(JSON.parse(readFileSync(path)).arch,identity==='arch-alias'?'qwen36':undefined,'runtime binding does not rewrite saved model identity');
  } finally { await f.cleanup(); }
});

test('a Q38 saved architecture cannot be overridden by Q36-looking model or executable names',async()=>{
  let probes=0; const f=nativeFixture({packaged:true,q36CapabilityProbe:async()=>{probes++;return cap;}});
  try {
    const path=join(f.repo,'strata-iq2_xs.json');
    writeFileSync(path,JSON.stringify({arch:'qwen4exp',model_name:'Qwen3.6-35B-A3B',exe:'strata-q36.exe',args:['--kv','int8']}));
    await f.control.refresh(true);assert.equal((await f.control.settingsKv('iq2_xs')).servingPolicy.id,'q38');
    const done=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs'}).id);assert.equal(done.state,'done',done.error);
    assert.equal(probes,0,'ordinary Q38 must never probe or execute the Q36 contract');
    const cfg=JSON.parse(readFileSync(f.launchConfigs.at(-1)));assert.equal(cfg.q36_run,undefined);assert.equal(cfg.args[cfg.args.indexOf('--kv')+1],'int8');
  } finally { await f.cleanup(); }
});

for (const identity of ['catalog','arch-alias']) test(`Q36 ${identity} retains explicit legacy FP32 without a capability marker`,async()=>{
  const f=nativeFixture({packaged:true,q36CapabilityProbe:async()=>null});
  try {
    f.catalog.families[0].family='qwen36';
    const path=join(f.repo,'strata-iq2_xs.json');
    writeFileSync(path,JSON.stringify({model_name:'Neutral model',exe:'neutral-runtime.exe',parallel:1,args:['--max-context','4096'],...(identity==='arch-alias'?{arch:'qwen36'}:{})}));
    await f.control.refresh(true);
    await f.control.saveChecked('iq2_xs',{values:{cache_type_k:'f32',cache_type_v:'f32'}});
    const done=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs'}).id);assert.equal(done.state,'done',done.error);
    const cfg=JSON.parse(readFileSync(f.launchConfigs.at(-1)));
    assert.equal(cfg.arch,'qwen35moe');assert.equal(cfg.q36_run.effective.kv_k,'f32');assert.equal(cfg.q36_run.effective.kv_v,'f32');
    assert.equal(cfg.args.includes('--cache-type-k'),false);assert.equal(cfg.args.includes('--cache-type-v'),false);
    await assert.rejects(f.control.saveChecked('iq2_xs',{values:{cache_type_k:'f16'}}),/not verified/);
  } finally { await f.cleanup(); }
});

test('a non-Q36 architecture wins over the Q36 catalog and cannot borrow its capability',async()=>{
  let probes=0;const f=nativeFixture({q36CapabilityProbe:async()=>{probes++;return cap;}});
  try {
    f.catalog.families[0].family='qwen36';
    const path=join(f.repo,'strata-iq2_xs.json');
    writeFileSync(path,JSON.stringify({arch:'deepseek41',model_name:'Qwen3.6',exe:'strata-q36.exe',cache_type_k:'f16',args:[]}));
    await f.control.refresh(true);
    const view=await f.control.settingsKv('iq2_xs');assert.equal(view.servingPolicy.id,'deepseek41');assert.equal(view.fields.find(f=>f.id==='cache_type_k').supported,false);
    await assert.rejects(f.control.saveChecked('iq2_xs',{values:{cache_type_k:'f16'}}),/serving adapter/);
    assert.throws(()=>f.control.operation('load',{model:'iq2_xs'}),/serving adapter/);
    assert.equal(probes,0);assert.equal(f.control.state().jobs.length,0);assert.deepEqual(f.effects,[]);
  } finally { await f.cleanup(); }
});

for(const family of ['qwen','swift','coder','unsloth','gyro','deepseek41'])test(`catalog ${family} binds its actual runtime adapter without rewriting saved identity`,async()=>{
  const f=nativeFixture({packaged:true,q36CapabilityProbe:async()=>{throw Error('Non-Q36 must not probe Q36');}});
  try{
    f.catalog.families[0].family=family;
    const path=join(f.repo,'strata-iq2_xs.json');
    const original={exe:family==='deepseek41'?join(f.repo,'strata-dsv41.exe'):'strata-q36.exe',model_name:'Qwen3.6-looking label',parallel:1,args:['--max-context','256']};
    if(family==='deepseek41')writeFileSync(original.exe,'inert dedicated-engine fixture; Control start is mocked');
    writeFileSync(path,JSON.stringify(original));await f.control.refresh(true);
    const done=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs',resourceProfile:'work',contextChosen:true}).id);
    assert.equal(done.state,'done',done.error);
    const run=JSON.parse(readFileSync(f.launchConfigs.at(-1)));
    assert.equal(run.arch,family==='deepseek41'?'deepseek41':'qwen4exp');
    assert.equal(run.q36_run,undefined);
    const saved=JSON.parse(readFileSync(path));
    for(const key of ['arch','exe','model_name'])assert.equal(saved[key],original[key]);
    assert.equal(saved.ui_context_chosen,true,'identity remains unchanged even when load-owned settings persist');
  }finally{await f.cleanup();}
});
