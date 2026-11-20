import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,rmSync,statSync,writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import * as gguf from '../server/gguf.mjs';
import {ggufFixture} from './helpers/local-gguf.mjs';

// Tensor directory only, independently transcribed from the Qwen3.5-9B layout.
// No model weights are created or inferred from the model's marketing name.
function trunk() {
 const metadata={'general.architecture':'qwen35','general.type':'model','general.name':'DeepSeek renamed fixture',
  'qwen35.block_count':33,'qwen35.nextn_predict_layers':1,'qwen35.embedding_length':4096,
  'qwen35.feed_forward_length':12288,'qwen35.attention.head_count':16,'qwen35.attention.head_count_kv':4,
  'qwen35.attention.key_length':256,'qwen35.attention.value_length':256,'qwen35.full_attention_interval':4,
  'qwen35.ssm.conv_kernel':4,'qwen35.ssm.inner_size':4096,'qwen35.ssm.state_size':128,
  'qwen35.ssm.time_step_rank':32,'qwen35.ssm.group_count':16,'tokenizer.ggml.tokens':{count:248320},
  'qwen35.attention.recurrent_layers':Array.from({length:33},(_,i)=>i<32&&(i+1)%4!==0?1:0)};
 const tensors=[];
 const add=(name,shape)=>tensors.push({name,shape});
 add('token_embd.weight',[4096,248320]);add('output_norm.weight',[4096]);add('output.weight',[4096,248320]);
 for(let i=0;i<32;i++){
  const prefix=`blk.${i}.`;
  for(const name of ['attn_norm.weight','post_attention_norm.weight'])add(prefix+name,[4096]);
  for(const name of ['ffn_gate.weight','ffn_up.weight'])add(prefix+name,[4096,12288]);
  add(prefix+'ffn_down.weight',[12288,4096]);
  const shapes=(i+1)%4===0?{
   'attn_q.weight':[4096,8192],'attn_k.weight':[4096,1024],'attn_v.weight':[4096,1024],
   'attn_output.weight':[4096,4096],'attn_q_norm.weight':[256],'attn_k_norm.weight':[256],
  }:{'attn_qkv.weight':[4096,8192],'attn_gate.weight':[4096,4096],'ssm_conv1d.weight':[4,8192],
   'ssm_dt.bias':[32],'ssm_a':[32],'ssm_beta.weight':[4096,32],'ssm_alpha.weight':[4096,32],
   'ssm_norm.weight':[128],'ssm_out.weight':[4096,4096]};
  for(const [name,shape] of Object.entries(shapes))add(prefix+name,shape);
 }
 return {metadata,tensors,tensorCount:tensors.length};
}
const diagnose=info=>{assert.equal(typeof gguf.qwen35MissingMtp,'function');return gguf.qwen35MissingMtp(info);};

test('bounded reader optionally exposes tensor shapes and FFN metadata without writing weights',t=>{
 const dir=mkdtempSync(join(tmpdir(),'gguf-directory-'));t.after(()=>rmSync(dir,{recursive:true,force:true}));
 const path=join(dir,'fixture.gguf');ggufFixture(path,{extra:[['qwen3next.feed_forward_length',4,4096]]});
 const before=statSync(path),ordinary=gguf.readGGUF(path),directory=gguf.readGGUF(path,{includeTensors:true});
 assert.equal(ordinary.tensors,undefined);
 assert.equal(directory.metadata['qwen3next.feed_forward_length'],4096);
 assert.deepEqual(directory.tensors,[{name:'test.weight',shape:[16],type:0,offset:0,bytes:64}]);
 assert.equal(statSync(path).mtimeMs,before.mtimeMs);assert.equal(statSync(path).size,before.size);
});
test('recognizes all 427 qwen35 trunk tensors and diagnoses absent predictor without scalar override',()=>{
 const info=trunk(),before=structuredClone(info);assert.equal(info.tensorCount,427);
 const result=diagnose(info);assert.equal(result.code,'QWEN35_MISSING_MTP');
 assert.equal(result.trunkLayers,32);assert.equal(result.predictorLayers,1);
 assert.equal(result.scalarOverrideSafe,false);
 assert.match(result.message,/32.*trunk/i);assert.match(result.message,/runtime/i);assert.match(result.message,/33.*recurrent/i);
 assert.deepEqual(info,before);
});
test('rejects every missing trunk tensor, including the first/last block and all projections',()=>{
 const info=trunk();
 for(let i=0;i<info.tensors.length;i++){
  // A tied output head is valid; all other tensors are required by this graph.
  if(info.tensors[i].name==='output.weight')continue;
  const tensors=info.tensors.filter((_,j)=>i!==j);
  assert.equal(diagnose({...info,tensors,tensorCount:tensors.length}),null,info.tensors[i].name);
 }
});
test('rejects wrong shapes even when all tensor names remain present',()=>{
 const info=trunk();
 for(let i=0;i<info.tensors.length;i++){
  const changed=structuredClone(info);changed.tensors[i].shape[0]++;
  assert.equal(diagnose(changed),null,info.tensors[i].name);
 }
});
test('present/partial predictor, unknown tensors, duplicates and unsupported architectures are not normalized',()=>{
 for(const name of ['blk.32.attn_norm.weight','blk.32.nextn.eh_proj.weight','blk.33.attn_norm.weight','unknown.weight','blk.0.attn_norm.weight']){
  const info=trunk();info.tensors.push({name,shape:[4096]});info.tensorCount++;
  assert.equal(diagnose(info),null,name);
 }
 for(const arch of ['qwen35moe','deepseek2','qwen3next','llama']){
  const info=trunk();info.metadata['general.architecture']=arch;assert.equal(diagnose(info),null);
 }
});
test('wrong counts, geometry, maps and split/auxiliary models cannot be mistaken for this export',()=>{
 for(const [key,value] of [['qwen35.block_count',32],['qwen35.block_count',34],['qwen35.nextn_predict_layers',0],
  ['qwen35.nextn_predict_layers',2],['qwen35.embedding_length',2048],['qwen35.feed_forward_length',8192],
  ['qwen35.full_attention_interval',8],['split.count',2],['split.no',1],['general.type','adapter'],
  ['qwen35.attention.recurrent_layers',Array(33).fill(1)],['qwen35.attention.recurrent_layers',Array(32).fill(0)],
  ['qwen35.attention.recurrent_layers',{count:33}]]){
  const info=trunk();info.metadata[key]=value;assert.equal(diagnose(info),null,key);
 }
 const info=trunk();delete info.tensors;assert.equal(diagnose(info),null);
});
test('a valid tied output head is accepted; a filename or name alone is never evidence',()=>{
 const info=trunk();info.tensors=info.tensors.filter(t=>t.name!=='output.weight');info.tensorCount--;
 assert.equal(diagnose(info).trunkLayers,32);
 assert.equal(diagnose({metadata:{'general.name':'Qwen3.5-9B'}}),null);
});
test('missing map is not evidence of runtime override capability; malformed directory is inconclusive',()=>{
 const info=trunk();delete info.metadata['qwen35.attention.recurrent_layers'];
 assert.equal(diagnose(info).scalarOverrideSafe,false);
 const malformed=trunk();malformed.tensors[0]=null;assert.equal(diagnose(malformed),null);
});
test('compatibility directory rejects overlap, invalid quantized rows, incomplete payload and excessive retention',t=>{
 const dir=mkdtempSync(join(tmpdir(),'gguf-directory-invalid-'));t.after(()=>rmSync(dir,{recursive:true,force:true}));
 const path=join(dir,'fixture.gguf');
 const u32=n=>{const b=Buffer.alloc(4);b.writeUInt32LE(n);return b;};
 const u64=n=>{const b=Buffer.alloc(8);b.writeBigUInt64LE(BigInt(n));return b;};
 const string=s=>Buffer.concat([u64(Buffer.byteLength(s)),Buffer.from(s)]);
 const write=(tensors,payloadBytes)=>{
  const header=Buffer.concat([Buffer.from('GGUF'),u32(3),u64(tensors.length),u64(0),...tensors.map(({name,shape,type=0,offset=0})=>
   Buffer.concat([string(name),u32(shape.length),...shape.map(u64),u32(type),u64(offset)]))]);
  writeFileSync(path,Buffer.concat([header,Buffer.alloc((32-header.length%32)%32+payloadBytes)]));
 };
 write([{name:'a',shape:[16]},{name:'b',shape:[16]}],64);
 assert.throws(()=>gguf.readGGUF(path,{includeTensors:true}),/overlap/);
 write([{name:'a',shape:[128,2],type:12}],144);
 assert.throws(()=>gguf.readGGUF(path,{includeTensors:true}),/layout/);
 write([{name:'a',shape:[16]}],63);
 assert.throws(()=>gguf.readGGUF(path,{includeTensors:true}),/incomplete/);
 writeFileSync(path,Buffer.concat([Buffer.from('GGUF'),u32(3),u64(100001),u64(0)]));
 assert.throws(()=>gguf.readGGUF(path,{includeTensors:true}),/too large.*compatibility/);
});
