/** Pure metadata arithmetic. This is not a measurement of backend/GPU allocation. */
import {resolveKvSettings} from './kv-settings.mjs';

const REFERENCE='Prism 73bfe6cc485554822f80c473a3c0c4f25e80f41b';
// These families use ordinary GQA KV geometry. MLA, sliding-window variants and
// unqualified hybrid families must not silently receive this dense formula.
const DENSE=new Set(['llama','qwen2','qwen2moe','qwen3','qwen3moe','phi3','gemma']);
const HYBRID=new Set(['qwen35','qwen35moe']);
const positive=(n,max=Number.MAX_SAFE_INTEGER)=>Number.isSafeInteger(n)&&n>0&&n<=max;
const product=(...values)=>{let n=1;for(const v of values){if(!Number.isSafeInteger(v)||v<0||!Number.isSafeInteger(n*v))return null;n*=v;}return n;};
const quantity=(bytes,source,reason=null)=>({bytes,source,reason});

/** Retain validated, bounded geometry independently of full model-file identity.
 * Missing/invalid explicit maps are never replaced by an interval guess.
 * Counts describe the trunk; next-token prediction layers are recorded separately.
 */
export function modelMemoryFromMetadata(metadata={}, {diskBytes=null}={}) {
 const architecture=metadata['general.architecture'];
 const read=key=>metadata[`${architecture}.${key}`];
 const integer=(key,max)=>positive(read(key),max)?read(key):null;
 const layerCountAll=integer('block_count',4096),contextLimit=integer('context_length',100_000_000);
 const heads=integer('attention.head_count',8192),embedding=integer('embedding_length',1_000_000);
 const kvHeads=read('attention.head_count_kv')===undefined?heads:integer('attention.head_count_kv',8192);
 const keyLength=read('attention.key_length')===undefined?(embedding&&heads&&embedding%heads===0?embedding/heads:null):integer('attention.key_length',1_000_000);
 const valueLength=read('attention.value_length')===undefined?keyLength:integer('attention.value_length',1_000_000);
 const mtp=read('nextn_predict_layers')===undefined?0:read('nextn_predict_layers');
 const base={schema:1,architecture,kind:'unknown',source:'GGUF metadata',reference:REFERENCE,
  diskBytes:Number.isSafeInteger(diskBytes)&&diskBytes>=0?diskBytes:null,layerCountAll,contextLimit,
  heads,kvHeads,keyLength,valueLength,trunkLayerCount:null,mtpLayers:null,fullAttentionLayers:null,recurrentLayers:null,
  ssm:Object.fromEntries([['convKernel','conv_kernel'],['innerSize','inner_size'],['stateSize','state_size'],['timeStepRank','time_step_rank'],['groupCount','group_count']].map(([name,key])=>[name,integer(`ssm.${key}`,1_000_000)]))};
 const unknown=reason=>({...base,reason});
 if(!DENSE.has(architecture)&&!HYBRID.has(architecture))return unknown('Architecture has no qualified attention-memory formula.');
 if(!layerCountAll||!contextLimit||!heads||!kvHeads||kvHeads>heads||!keyLength||!valueLength)return unknown('Attention geometry or context limit is missing or invalid.');
 if(!Number.isSafeInteger(mtp)||mtp<0||mtp>=layerCountAll)return unknown('Next-token prediction layer count is invalid.');
 const trunkLayerCount=layerCountAll-mtp;
 let recurrentLayers=0,layerMapSource='dense architecture';
 if(HYBRID.has(architecture)){
  const explicit=read('attention.recurrent_layers');
  let map;
  if(explicit!==undefined){
   const bit=v=>typeof v==='boolean'||v===0||v===1;
   if(Array.isArray(explicit)){
    if(explicit.length!==layerCountAll||!Array.from(explicit).every(bit))return unknown('Explicit recurrent layer map is invalid or does not match block_count.');
    map=explicit;
   }else if(bit(explicit))map=Array(layerCountAll).fill(explicit);
   else return unknown('Explicit recurrent layer map has an unsupported type.');
   layerMapSource='explicit recurrent layer map';
  }else{
   const interval=read('full_attention_interval')===undefined?4:read('full_attention_interval');
   if(!positive(interval,trunkLayerCount))return unknown('Full attention interval is invalid.');
   map=Array.from({length:layerCountAll},(_,i)=>i<trunkLayerCount&&(i+1)%interval!==0);
   layerMapSource=read('full_attention_interval')===undefined?'reference default interval 4':'declared full attention interval';
  }
  // MTP blocks are recorded separately and never counted as recurrent trunk state.
  recurrentLayers=map.slice(0,trunkLayerCount).filter(Boolean).length;
 }
 return {...base,kind:HYBRID.has(architecture)?'hybrid':'dense',reason:null,trunkLayerCount,mtpLayers:mtp,
  fullAttentionLayers:trunkLayerCount-recurrentLayers,recurrentLayers,layerMapSource};
}

// ggml-common.h block structs include the scale: do not add quantization
// overhead again. ggml_row_size requires whole blocks, not ceil(width/block).
const ROW_LAYOUT={f32:[1,4],f16:[1,2],bf16:[1,2],q8_0:[32,34],q4_0:[32,18]};
/** Whole GGML rows only. No padding of an incompatible head width. */
export function ggmlRowBytes(type,elements){
 const key=String(type??'').toLowerCase();
 if(!Object.hasOwn(ROW_LAYOUT,key))throw Object.assign(new Error(`Unsupported cache type: ${type}`),{code:'KV_UNSUPPORTED_TYPE'});
 if(!positive(elements))throw new Error('row element count must be a positive safe integer');
 const [block,bytes]=ROW_LAYOUT[key];
 if(elements%block!==0)throw new Error('row element count must contain whole cache blocks');
 const size=product(elements/block,bytes);
 if(size===null)throw new Error('row size exceeds safe integer arithmetic');
 return size;
}
function rowBytes(codec,width,heads){
 const layout=typeof codec==='string'&&Object.hasOwn(ROW_LAYOUT,codec)?ROW_LAYOUT[codec]:null;
 if(!layout)return null;
 if(!positive(width,1_000_000)||!positive(heads,8192)||width%layout[0]!==0)return null;
 return product(ggmlRowBytes(codec,width),heads);
}

/** contextTokens is PER SEQUENCE. This API pads each sequence to 256 tokens
 * (reference llama-context.cpp), then multiplies sequences exactly once.
 * C must map runtime total capacity to that contract before calling this API.
 * backendProfile is reserved for an explicitly qualified allocation profile;
 * it cannot turn file bytes into device weights or assert TurboQuant support.
 */
export function estimateModelMemory({modelMemory,resolvedKv,expert=[],contextTokens,sequences=1,backendProfile=null}={}){
 const m=modelMemory??{},source=`theoretical GGML rows; ${REFERENCE}`;
 let reason=null,kv=resolvedKv;
 try{kv??=resolveKvSettings(expert);}catch(error){reason=error.message;}
 if(!['dense','hybrid'].includes(m.kind))reason??=m.reason??'Model memory geometry is unavailable. Re-import metadata.';
 if(!positive(contextTokens)||!positive(m.contextLimit)||contextTokens>m.contextLimit)reason??='Requested per-sequence context is invalid or exceeds the declared limit.';
 if(!positive(sequences))reason??='Sequence count must be a positive safe integer.';
 const allocationTokensPerSequence=positive(contextTokens)?product(Math.ceil(contextTokens/256),256):null;
 const allocationTokenCapacity=allocationTokensPerSequence!==null?product(allocationTokensPerSequence,sequences):null;
 if(allocationTokenCapacity===null)reason??='Context allocation exceeds safe integer arithmetic.';
 if(!Number.isSafeInteger(m.fullAttentionLayers)||m.fullAttentionLayers<0||m.fullAttentionLayers>4096||
  !positive(m.kvHeads,8192)||!positive(m.keyLength,1_000_000)||!positive(m.valueLength,1_000_000))reason??='Validated attention geometry is unavailable.';
 const kRow=rowBytes(kv?.k,m.keyLength,m.kvHeads),vRow=rowBytes(kv?.v,m.valueLength,m.kvHeads);
 if(kRow===null||vRow===null)reason??=[kv?.k,kv?.v].some(c=>String(c).startsWith('turbo'))?
  'TurboQuant row layout and backend capability are not qualified by this estimator.':'Unsupported cache codec or non-block-aligned attention head width.';
 let k=null,v=null,total=null;
 if(!reason){
  k=product(kRow,allocationTokenCapacity,m.fullAttentionLayers);v=product(vRow,allocationTokenCapacity,m.fullAttentionLayers);
  total=k!==null&&v!==null&&Number.isSafeInteger(k+v)?k+v:null;
  if(total===null){k=null;v=null;reason='Attention cache size exceeds safe integer arithmetic.';}
 }
 const recurrent=m.kind==='dense'?quantity(0,'dense architecture'):quantity(null,'unmeasured backend allocation','Recurrent state requires backend state/rollback allocation; it is not context times layers.');
 const components={disk:quantity(Number.isSafeInteger(m.diskBytes)&&m.diskBytes>=0?m.diskBytes:null,'GGUF file size','File bytes are not backend weights allocation.'),
  attentionK:quantity(k,source,reason),attentionV:quantity(v,source,reason),attentionKv:quantity(total,source,reason),
  backendWeights:quantity(null,'unmeasured backend allocation','Device/host weights placement is not established by disk bytes.'),recurrentState:recurrent,
  workspace:quantity(null,'unmeasured backend allocation','Compute workspace, allocator overhead and any MTP allocations are unmeasured.')};
 // Disk and the K/V detail rows are deliberately excluded from the subtotal.
 const knownSubtotalBytes=(total??0)+(recurrent.bytes??0);
 return {schema:1,confidence:reason?'unknown':'theoretical-attention-only',reason,resolvedKv:kv??null,
  contextSemantics:'per-sequence',contextTokens,sequences,allocationTokensPerSequence,allocationTokenCapacity,
  contextPaddingTokens:256,backendProfileId:typeof backendProfile?.id==='string'?backendProfile.id:null,
  components,knownSubtotalBytes,totalBytes:null,fits:null};
}
