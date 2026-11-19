/** Diagnose the known Qwen3.5-9B trunk-only export. Never rewrite metadata or weights.
 * Geometry comes from Qwen/Qwen3.5-9B config.json and llama.cpp's qwen35 loader.
 * Requiring the complete directory avoids labelling a genuinely lost trunk as MTP.
 */
export function qwen35MissingMtp(info) {
 const m=info?.metadata,tensors=info?.tensors;
 if(m?.['general.architecture']!=='qwen35'||m['general.type']!=='model'||
    (m['split.count']??1)!==1||(m['split.no']??0)!==0||
    !Array.isArray(tensors)||info.tensorCount!==tensors.length)return null;
 // Deliberately narrow: other sizes/layouts need their own source-backed proof.
 const geometry={block_count:33,nextn_predict_layers:1,embedding_length:4096,feed_forward_length:12288,
  'attention.head_count':16,'attention.head_count_kv':4,'attention.key_length':256,'attention.value_length':256,
  full_attention_interval:4,'ssm.conv_kernel':4,'ssm.inner_size':4096,'ssm.state_size':128,
  'ssm.time_step_rank':32,'ssm.group_count':16};
 if(Object.entries(geometry).some(([key,value])=>m[`qwen35.${key}`]!==value)||
    m['tokenizer.ggml.tokens']?.count!==248320||(m['qwen35.expert_count']??0)!==0)return null;
 const recurrent=m['qwen35.attention.recurrent_layers'];
 if(recurrent!==undefined&&(!Array.isArray(recurrent)||recurrent.length!==33||
    recurrent.some((value,i)=>value!==(i<32&&(i+1)%4!==0?1:0))))return null;
 const expected=new Map([
  ['token_embd.weight',[4096,248320]],['output_norm.weight',[4096]],
 ]);
 if(tensors.some(t=>t?.name==='output.weight'))expected.set('output.weight',[4096,248320]);
 for(let i=0;i<32;i++){
  const add=(name,shape)=>expected.set(`blk.${i}.${name}`,shape);
  add('attn_norm.weight',[4096]);add('post_attention_norm.weight',[4096]);
  add('ffn_gate.weight',[4096,12288]);add('ffn_up.weight',[4096,12288]);add('ffn_down.weight',[12288,4096]);
  if((i+1)%4===0){
   add('attn_q.weight',[4096,8192]);add('attn_k.weight',[4096,1024]);add('attn_v.weight',[4096,1024]);
   add('attn_output.weight',[4096,4096]);add('attn_q_norm.weight',[256]);add('attn_k_norm.weight',[256]);
  }else{
   add('attn_qkv.weight',[4096,8192]);add('attn_gate.weight',[4096,4096]);
   add('ssm_conv1d.weight',[4,8192]);add('ssm_dt.bias',[32]);add('ssm_a',[32]);
   add('ssm_beta.weight',[4096,32]);add('ssm_alpha.weight',[4096,32]);
   add('ssm_norm.weight',[128]);add('ssm_out.weight',[4096,4096]);
  }
 }
 if(tensors.length!==expected.size)return null;
 for(const tensor of tensors){
  const shape=expected.get(tensor?.name);
  if(!shape||!Array.isArray(tensor.shape)||shape.length!==tensor.shape.length||
     shape.some((value,i)=>tensor.shape[i]!==value))return null;
  expected.delete(tensor.name);
 }
 if(expected.size)return null;
 return {code:'QWEN35_MISSING_MTP',architecture:'qwen35',trunkLayers:32,predictorLayers:1,
   // Header proof is not runtime capability or successful-load evidence.
   scalarOverrideSafe:false,
  message:'This qwen35 GGUF has all 32 trunk layers but declares one MTP predictor with no predictor tensors. '+
   'A runtime that supports trunk-only Qwen3.5 exports is required if loading fails on blk.32. '+
   (recurrent===undefined?'':
    'Its explicit 33-entry recurrent-layer map also prevents a scalar block-count override. ')+
   'Keep the original GGUF; changing context or cache precision does not supply the missing predictor.'};
}
