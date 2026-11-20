import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseQ36KvCapabilities, probeQ36KvCapabilities, q36KvFields } from '../server/q36-kv-capabilities.mjs';
import { prepareQ36Launch, q36PersistentKvBytes, observeQ36Runtime } from '../server/q36-run-contract.mjs';
import { nativeFixture, finishJob } from './helpers/native-fixture.mjs';

const cap = { schema: 1, family: 'qwen35moe', layoutVersion: 1, scaleSchema: 'none', cacheTypesK: ['f32','f16'], cacheTypesV: ['f32','f16'],
  flags: ['--cache-type-k','--cache-type-v'], quality: 'experimental-unqualified' };
const line = `Q36_KV_CAPABILITIES ${JSON.stringify(cap)}`;
const sha = 'ab'.repeat(32);
const verified = { ...cap, binarySha256: sha };

test('strict actual help marker: legacy, duplicate, malformed, proposal and oversized data stay unknown', () => {
  assert.deepEqual(parseQ36KvCapabilities(`ordinary help\n${line}\n`), cap);
  for (const text of ['version 99.0 f16', `${line}\n${line}`, 'Q36_KV_CAPABILITIES junk', `${'x'.repeat(65536)}\n${line}`,
    line.replace('"f16"','"q8_0"'), line.replace('"schema":1','"schema":true'), line.replace('"scaleSchema":"none"','"scaleSchema":"nvfp4"')]) {
    assert.equal(parseQ36KvCapabilities(text), null);
  }
});

test('probe bounds help, omits user environment, and rejects timeout, overflow or changed executable', async () => {
  const calls = [];
  const run = (exe,args,options,done) => { calls.push({exe,args,options}); done(null,line,''); };
  assert.deepEqual(await probeQ36KvCapabilities('C:/engine/strata.exe',{run,hash:()=>sha}),verified);
  assert.deepEqual(calls[0].args,['--help']);
  assert.equal(calls[0].options.timeout,3000);
  assert.equal(calls[0].options.maxBuffer,65536);
  assert.ok(Object.keys(calls[0].options.env).every(key => ['PATH','SYSTEMROOT','WINDIR','LD_LIBRARY_PATH','CUDA_PATH'].includes(key.toUpperCase())));
  for (const code of ['ETIMEDOUT','ERR_CHILD_PROCESS_STDIO_MAXBUFFER']) {
    assert.equal(await probeQ36KvCapabilities('C:/engine/strata.exe',{hash:()=>sha,run:(_e,_a,_o,done)=>done({code},line,'')}),null);
  }
  let reads=0;
  assert.equal(await probeQ36KvCapabilities('C:/engine/strata.exe',{run,hash:()=>reads++ ? 'cd'.repeat(32) : sha}),null);
});

test('mixed formats require this binary capability and retain exact scalar allocations', () => {
  for (const k of ['f32','f16']) for (const v of ['f32','f16']) {
    const result=prepareQ36Launch({binarySha256:sha,kvCapability:verified,args:['--max-context','131072'],requested:{cacheTypeK:k,cacheTypeV:v}});
    assert.equal(result.ok,true);
    assert.equal(result.contract.requested.kv_k,k);
    assert.equal(result.contract.effective.kv_v,v);
    const bytes=131072n*2n*256n*11n*BigInt((k==='f32'?4:2)+(v==='f32'?4:2));
    assert.equal(result.contract.effective.kvEstimate.bytesDecimal,String(bytes));
    assert.equal(q36PersistentKvBytes(98304,2,{k,v}).bytes,bytes*3n/2n);
  }
  for (const kvCapability of [null,{...verified,binarySha256:'cd'.repeat(32)}]) {
    assert.equal(prepareQ36Launch({binarySha256:sha,kvCapability,args:[],requested:{cacheTypeV:'f16'}}).ok,false);
  }
  const legacy=prepareQ36Launch({args:['--cache-type-k','f32'],requested:{cacheTypeV:'f32'}});
  assert.equal(legacy.ok,true); assert.deepEqual(legacy.argv,[]);
  for (const requested of [{cacheTypeK:'q8_0'},{cacheTypeV:'nvfp4'},{persistentKv:'fp16',cacheTypeK:'f32'}])
    assert.equal(prepareQ36Launch({args:[],binarySha256:sha,kvCapability:verified,requested}).ok,false);
  for (const cells of [0,-1,Number.MAX_SAFE_INTEGER+1]) assert.equal(q36PersistentKvBytes(cells).ok,false);
  assert.equal(q36PersistentKvBytes(Number.MAX_SAFE_INTEGER,Number.MAX_SAFE_INTEGER).ok,false);
});

test('observed bytes are checked using the engine actual geometry, including absent MTP', () => {
  const contract=prepareQ36Launch({generation:'launch',binarySha256:sha,kvCapability:verified,args:['--max-context','256'],requested:{cacheTypeK:'f16'}}).contract;
  const effective={generation:'child',processId:1,context:256,parallel:1,prefill:128,mtp:0,kv_k:'f16',kv_v:'f32',
    kv_layout_version:1,kv_scale_schema:'none',kv_heads:2,kv_head_dim:256,kv_attention_states:10,kv_bytes:String(256*2*256*10*6)};
  const state={launchGeneration:'launch',child:{generation:'child',processId:1},effective};
  assert.equal(observeQ36Runtime(contract,state).observed.status,'observed');
  assert.equal(observeQ36Runtime(contract,{...state,effective:{...effective,kv_bytes:'1'}}).observed.status,'mismatch');
});

test('selected-model fields expose only negotiated scalar formats and preserve other families', () => {
  const fields=[{id:'cache_type_k',choices:['q8_0','turbo4']},{id:'cache_type_v',choices:['q8_0']},{id:'spin8_mode',choices:['off']},{id:'parallel',min:1}];
  assert.deepEqual(q36KvFields(fields)[0].choices,['f32']);
  assert.deepEqual(q36KvFields(fields,verified)[1].choices,['f32','f16']);
  assert.equal(q36KvFields(fields,verified)[2].supported,false);
  assert.deepEqual(fields[0].choices,['q8_0','turbo4']);
});

test('actual Host launch config carries negotiated mixed flags and selected-model fields', async () => {
  const f=nativeFixture({q36CapabilityProbe:async()=>verified});
  try {
    await f.control.refresh(true);
    const path=join(f.repo,'strata-iq3_s.json');
    const config=JSON.parse(readFileSync(path,'utf8'));
    Object.assign(config,{arch:'qwen35moe',model_name:'Qwen3.6-35B-A3B',cache_type_k:'f16',cache_type_v:'f32',args:['--max-context','4096']});
    writeFileSync(path,JSON.stringify(config));
    const settings=await f.control.settingsKv('iq3_s');
    assert.deepEqual(settings.fields.find(f=>f.id==='cache_type_k').choices,['f32','f16']);
    const loaded=await finishJob(f.control,f.control.operation('load',{model:'iq3_s'}).id);
    assert.equal(loaded.state,'done',loaded.error);
    const call=f.calls.find(c=>c.args?.some(a=>String(a).endsWith('server.py')));
    const run=JSON.parse(readFileSync(call.args[call.args.indexOf('--config')+1],'utf8'));
    assert.equal(run.args[run.args.indexOf('--cache-type-k')+1],'f16');
    assert.equal(run.args[run.args.indexOf('--cache-type-v')+1],'f32');
    assert.equal(run.q36_run.backend.binarySha256,sha);
    assert.equal(run.q36_run.effective.persistentKv,'mixed');
    assert.equal(run.q36_run.observed.status,'unobserved');
  } finally { await f.cleanup(); }
});
