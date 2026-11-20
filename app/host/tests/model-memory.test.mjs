import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,rmSync,readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {ggufFixture} from './helpers/local-gguf.mjs';
import {readGGUF} from '../server/gguf.mjs';
import {inspectLocalModel,createLocalModels} from '../server/local-models.mjs';
import {modelMemoryFromMetadata,estimateModelMemory} from '../server/model-memory.mjs';
const GiB=1024**3;
const hikari={
 'general.architecture':'qwen35','qwen35.block_count':64,'qwen35.context_length':262144,
 'qwen35.embedding_length':5120,'qwen35.attention.head_count':24,'qwen35.attention.head_count_kv':4,
 'qwen35.attention.key_length':256,'qwen35.attention.value_length':256,'qwen35.full_attention_interval':4,
 'qwen35.ssm.conv_kernel':4,'qwen35.ssm.state_size':128,'qwen35.ssm.group_count':16,
 'qwen35.ssm.time_step_rank':48,'qwen35.ssm.inner_size':6144,
};
const memory=(extra={})=>modelMemoryFromMetadata({...hikari,...extra},{diskBytes:1000});
const estimate=(extra={})=>estimateModelMemory({modelMemory:memory(),resolvedKv:{k:'q8_0',v:'q8_0'},contextTokens:131072,sequences:1,...extra});
function fixture(t){const root=mkdtempSync(join(tmpdir(),'strata-memory-'));t.after(()=>rmSync(root,{recursive:true,force:true}));return join(root,'tiny.gguf');}
function hikariFixture(file,extra=[]){const keys=Object.keys(hikari).filter(k=>!k.startsWith('general.'));return ggufFixture(file,{arch:'qwen35',omit:keys,extra:[...keys.map(k=>[k,4,hikari[k]]),...extra]});}

test('Hikari splits attention and recurrence and exactly accounts for each K/V codec',()=>{
 const m=memory();assert.equal(m.trunkLayerCount,64);assert.equal(m.fullAttentionLayers,16);assert.equal(m.recurrentLayers,48);assert.equal(m.ssm.innerSize,6144);
 for(const [k,v,gib] of [['q8_0','q8_0',4.25],['q4_0','q4_0',2.25],['f16','f16',8],['q8_0','f16',6.125]]){
  const e=estimate({resolvedKv:{k,v}});assert.equal(e.components.attentionKv.bytes,gib*GiB);
  assert.equal(e.components.attentionK.bytes+e.components.attentionV.bytes,e.components.attentionKv.bytes);
  assert.equal(e.totalBytes,null);assert.equal(e.fits,null);assert.equal(e.components.recurrentState.bytes,null);
  assert.equal(e.components.backendWeights.bytes,null);assert.equal(e.components.workspace.bytes,null);
  assert.equal(e.knownSubtotalBytes,e.components.attentionKv.bytes);assert.equal(e.components.disk.bytes,1000);
 }
});

test('explicit recurrent map wins; malformed maps cannot fall back; MTP is outside trunk',()=>{
 const key='qwen35.attention.recurrent_layers';
 assert.equal(memory({[key]:Array(64).fill(false)}).fullAttentionLayers,64);
 assert.equal(memory({[key]:true}).fullAttentionLayers,0);
 for(const invalid of [[],Array(64),Array(63).fill(1),Array(64).fill(2),{count:64},'false'])assert.equal(memory({[key]:invalid}).kind,'unknown');
 assert.equal(memory({'qwen35.full_attention_interval':undefined}).fullAttentionLayers,16);
 for(const interval of [0,-1,65,1.5,null,Number.MAX_SAFE_INTEGER])assert.equal(memory({'qwen35.full_attention_interval':interval}).kind,'unknown');
 const mtp=memory({'qwen35.block_count':65,'qwen35.nextn_predict_layers':1});
 assert.equal(mtp.trunkLayerCount,64);assert.equal(mtp.layerCountAll,65);assert.equal(mtp.fullAttentionLayers,16);assert.equal(mtp.recurrentLayers,48);assert.equal(mtp.mtpLayers,1);
 assert.equal(memory({'qwen35.nextn_predict_layers':64}).kind,'unknown');
 assert.equal(memory({'qwen35.nextn_predict_layers':null}).kind,'unknown');
 const explicitMtp=memory({'qwen35.block_count':65,'qwen35.nextn_predict_layers':1,[key]:Array(65).fill(true)});
 assert.equal(explicitMtp.recurrentLayers,64);assert.equal(explicitMtp.fullAttentionLayers,0);
});

test('contexts and sequences are validated, padded explicitly, and multiplied exactly once',()=>{
 assert.equal(estimate({sequences:2}).components.attentionKv.bytes,8.5*GiB);
 assert.equal(estimate({contextTokens:1}).allocationTokensPerSequence,256);
 assert.equal(estimate({contextTokens:1,sequences:2}).allocationTokenCapacity,512);
 assert.equal(estimate().contextSemantics,'per-sequence');
 for(const contextTokens of [0,-1,1.5,262145,Infinity,Number.MAX_SAFE_INTEGER])assert.equal(estimate({contextTokens}).components.attentionKv.bytes,null);
 for(const sequences of [0,-1,1.5,Infinity,Number.MAX_SAFE_INTEGER])assert.equal(estimate({sequences}).components.attentionKv.bytes,null);
 assert.equal(estimate({resolvedKv:{k:'not-a-codec',v:'f16'}}).components.attentionKv.bytes,null);
 for(const k of ['constructor','__proto__','toString'])assert.equal(estimate({resolvedKv:{k,v:'f16'}}).components.attentionKv.bytes,null);
 assert.equal(estimate({resolvedKv:{k:'turbo2',v:'f16'}}).components.attentionKv.bytes,null);
 assert.equal(estimate({expert:[{id:'ctx.kvMode',value:'FP16'}],resolvedKv:undefined}).components.attentionKv.bytes,8*GiB);
});

test('invalid geometry and unknown architectures never become a dense fit; dense llama remains supported',()=>{
 for(const [key,value] of [['qwen35.block_count',0],['qwen35.attention.key_length',0],['qwen35.attention.key_length',1e12],['qwen35.attention.head_count_kv',null]])assert.equal(estimate({modelMemory:memory({[key]:value})}).components.attentionKv.bytes,null);
 assert.equal(estimate({modelMemory:memory({'qwen35.attention.key_length':33})}).components.attentionKv.bytes,null);
 assert.equal(estimate({modelMemory:memory({'qwen35.attention.key_length':128}),resolvedKv:{k:'q8_0',v:'f16'}}).components.attentionKv.bytes,5.0625*GiB);
 assert.equal(estimate({resolvedKv:{k:'bf16',v:'bf16'}}).components.attentionKv.bytes,8*GiB);
 const unknown=modelMemoryFromMetadata({...hikari,'general.architecture':'future_model'});
 assert.equal(unknown.kind,'unknown');assert.equal(estimate({modelMemory:unknown}).fits,null);
 const dense=modelMemoryFromMetadata({'general.architecture':'llama','llama.block_count':32,'llama.context_length':8192,'llama.embedding_length':4096,'llama.attention.head_count':32,'llama.attention.head_count_kv':8});
 const e=estimate({modelMemory:dense,contextTokens:8192,resolvedKv:{k:'f16',v:'f16'}});
 assert.equal(e.components.attentionKv.bytes,GiB);assert.equal(e.components.recurrentState.bytes,0);
 assert.equal(estimate({modelMemory:{...dense,kvHeads:Number.MAX_SAFE_INTEGER}}).components.attentionKv.bytes,null);
 const overflow=memory({'qwen35.block_count':4096,'qwen35.attention.head_count':8192,'qwen35.attention.head_count_kv':8192,'qwen35.attention.key_length':1_000_000,'qwen35.attention.value_length':1_000_000,'qwen35.context_length':100_000_000});
 assert.equal(estimate({modelMemory:overflow,contextTokens:100_000_000,resolvedKv:{k:'f16',v:'f16'}}).components.attentionKv.bytes,null);
 assert.match(estimate({modelMemory:overflow,contextTokens:100_000_000,resolvedKv:{k:'f16',v:'f16'}}).reason,/safe integer/);
});

test('tiny GGUF imports retain only bounded recurrent maps, SSM metadata and additive identity',t=>{
 const file=fixture(t);hikariFixture(file,[['qwen35.attention.recurrent_layers',9,{element:7,values:Array.from({length:64},(_,i)=>(i+1)%4!==0)}]]);
 const info=readGGUF(file),preview=inspectLocalModel(file),v=preview.variant;
 assert.equal(info.metadata['qwen35.attention.recurrent_layers'].length,64);
 assert.deepEqual(info.metadata['tokenizer.ggml.tokens'],{count:1});assert.equal(v.modelMemory.fullAttentionLayers,16);
 assert.equal(v.kvKbPerToken,32);assert.deepEqual(v.local.modelMemory,v.modelMemory);assert.equal(v.local.keyLength,256);
 assert.equal(v.local.nFullAttention,16);assert.equal(v.local.nRecurrent,48);
 assert.equal(v.memoryPlan.attentionKvGiB,4.25);assert.equal(v.memoryPlan.fits,null);
 const sig=v.local.metadataSignature;hikariFixture(file,[['qwen35.ssm.inner_size',4,6000]]); // duplicate is rejected
 assert.throws(()=>readGGUF(file),/duplicate/);
 hikariFixture(file,[['qwen35.attention.recurrent_layers',4,0]]);assert.notEqual(inspectLocalModel(file).variant.local.metadataSignature,sig);
 const service=createLocalModels(join(file,'..','data'));service.add(inspectLocalModel(file));assert.equal(service.list()[0].modelMemory.fullAttentionLayers,64);
 const saved=join(file,'..','data','local-models.json'),old=JSON.parse(readFileSync(saved,'utf8'));delete old.models[0].modelMemory;delete old.models[0].local.modelMemory;writeFileSync(saved,JSON.stringify(old));
 const bytes=readFileSync(saved);assert.equal(createLocalModels(join(file,'..','data')).list().length,1);assert.deepEqual(readFileSync(saved),bytes);
});

test('GGUF recurrent map wrong type/size stays unknown; oversized/truncated arrays fail safely',t=>{
 const file=fixture(t),key='qwen35.attention.recurrent_layers';
 for(const value of [{element:4,values:Array(63).fill(0)},['bad'],{element:4,values:Array(64).fill(2)}]){
  hikariFixture(file,[[key,9,value]]);const v=inspectLocalModel(file).variant;
  assert.equal(v.modelMemory.kind,'unknown');assert.equal(v.local.nFullAttention,null);
  assert.equal(v.memoryPlan.attentionKvGiB,null);assert.equal(v.kvKbPerToken,null);
 }
 hikariFixture(file,[[key,8,'false']]);assert.equal(inspectLocalModel(file).variant.modelMemory.kind,'unknown');
 hikariFixture(file,[[key,7,true]]);assert.equal(inspectLocalModel(file).variant.modelMemory.recurrentLayers,64);
 hikariFixture(file,[[key,9,{element:7,values:Array(4097).fill(false)}]]);assert.equal(inspectLocalModel(file).variant.modelMemory.kind,'unknown');
 const bytes=readFileSync(file),needle=Buffer.from(key),at=bytes.indexOf(needle)+needle.length+4+4;
 bytes.writeBigUInt64LE(2000001n,at);writeFileSync(file,bytes);assert.throws(()=>readGGUF(file),/array is too large/);
 hikariFixture(file,[[key,9,{element:7,values:Array(64).fill(false)}]]);const complete=readFileSync(file);writeFileSync(file,complete.subarray(0,complete.indexOf(needle)+needle.length+16));assert.throws(()=>readGGUF(file),/truncated/);
});
