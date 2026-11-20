/** Public memory APIs. Both importer plans and explicit per-sequence estimates
 * use the same validated GGUF geometry and whole-block arithmetic. */
import {createHash} from 'node:crypto';
import {modelMemoryFromMetadata,estimateModelMemory as estimateGeometry} from './model-memory-geometry.mjs';
export {modelMemoryFromMetadata,ggmlRowBytes} from './model-memory-geometry.mjs';
export const GiB=1024**3;

/** Compatibility helper for callers supplying an INDEX LIST, not raw GGUF.
 * GGUF recurrent_layers is a bitmap/scalar; extractHybridGeometry below reads
 * it through modelMemoryFromMetadata and never interprets it as indices. */
export function classifyHybridLayers({nLayerAll,fullAttentionInterval=null,recurrentLayers=null}={}){
 const unknown=(source,reason)=>({nFullAttention:null,nRecurrent:null,confidence:'low',source,reason});
 if(!Number.isSafeInteger(nLayerAll)||nLayerAll<1||nLayerAll>4096)return unknown('invalid','block_count missing or invalid');
 const n=nLayerAll,interval=fullAttentionInterval;
 const recurrent=Array.isArray(recurrentLayers)?recurrentLayers:recurrentLayers?.values;
 if(recurrentLayers!==null){
  if(!Array.isArray(recurrent)||!Array.from(recurrent).every(i=>Number.isSafeInteger(i)&&i>=0&&i<n)||new Set(recurrent).size!==recurrent.length)
   return unknown('attention.recurrent_layers','recurrent_layers has invalid or duplicate indices');
  if(interval!==null&&(!Number.isSafeInteger(interval)||interval<1||interval>n||recurrent.length!==n-Math.floor(n/interval)))
   return unknown('attention.recurrent_layers','recurrent_layers disagrees with full_attention_interval expectation');
  return {nFullAttention:n-recurrent.length,nRecurrent:recurrent.length,confidence:'high',source:'attention.recurrent_layers',reason:null,recurrentIndices:recurrent.toSorted((a,b)=>a-b)};
 }
 if(!Number.isSafeInteger(interval)||interval<1||interval>n)return unknown('full_attention_interval','full_attention_interval missing or invalid');
 return {nFullAttention:Math.floor(n/interval),nRecurrent:n-Math.floor(n/interval),confidence:'high',source:'full_attention_interval',reason:null};
}

export function extractHybridGeometry(metadata={}){
 const m=modelMemoryFromMetadata(metadata),interval=metadata[`${m.architecture}.full_attention_interval`];
 return {architecture:m.architecture,nLayerAll:m.layerCountAll,nFullAttention:m.fullAttentionLayers,nRecurrent:m.recurrentLayers,
  trunkLayerCount:m.trunkLayerCount,mtpLayers:m.mtpLayers,
  fullAttentionInterval:Number.isSafeInteger(interval)&&interval>0?interval:null,
  nKvHeads:m.kvHeads,keyLength:m.keyLength,valueLength:m.valueLength,contextLimit:m.contextLimit,ssm:m.ssm,
  confidence:m.kind==='unknown'?'low':'high',layerSource:m.layerMapSource??'unknown',layerReason:m.reason,
  metadataSignature:createHash('sha256').update(JSON.stringify(metadata)).digest('hex')};
}

/** Per-sequence context, padded to 256 cells; unsupported geometry returns null. */
export function estimateAttentionKvBytes({context,sequences=1,nFullAttention,nKvHeads,keyLength,valueLength,kType,vType}={}){
 return estimateGeometry({modelMemory:{kind:'dense',fullAttentionLayers:nFullAttention,kvHeads:nKvHeads,keyLength,valueLength,contextLimit:100_000_000},
  contextTokens:context,sequences,resolvedKv:{k:kType,v:vType}}).components.attentionKv.bytes;
}

/** Modern input uses modelMemory/contextTokens; legacy importer input uses
 * metadata/context. The latter retains its public plan keys and confidence
 * vocabulary. Unknown architectures now have null attention bytes in both. */
export function estimateModelMemory(options={}){
 const legacy=['metadata','context','kType','vType','diskBytes'].some(key=>Object.hasOwn(options,key));
 if(!legacy)return estimateGeometry(options);
 const {metadata={},diskBytes=null,context,sequences=1,kType='q8_0',vType='q8_0'}=options;
 const m=modelMemoryFromMetadata(metadata,{diskBytes}),geo=extractHybridGeometry(metadata);
 const estimate=estimateGeometry({modelMemory:m,contextTokens:context,sequences,resolvedKv:{k:kType,v:vType}});
 const attentionKvBytes=estimate.components.attentionKv.bytes;
 return {architecture:geo.architecture,metadataSignature:geo.metadataSignature,nLayerAll:geo.nLayerAll,
  nFullAttention:geo.nFullAttention,nRecurrent:geo.nRecurrent,nKvHeads:geo.nKvHeads,keyLength:geo.keyLength,valueLength:geo.valueLength,
  trunkLayerCount:m.trunkLayerCount,mtpLayers:m.mtpLayers,contextLimit:geo.contextLimit,
  contextRequested:Number.isSafeInteger(context)&&context>0?context:null,kType,vType,diskBytes:m.diskBytes,
  weightsAllocatedBytes:null,attentionKvBytes,attentionKvGiB:attentionKvBytes===null?null:Number((attentionKvBytes/GiB).toFixed(3)),
  recurrentBytes:estimate.components.recurrentState.bytes,workspaceBytes:null,reserveBytes:null,
  confidence:estimate.confidence==='unknown'?'low':'high',layerSource:geo.layerSource,fits:null,
  contextSemantics:estimate.contextSemantics,allocationTokensPerSequence:estimate.allocationTokensPerSequence,
  allocationTokenCapacity:estimate.allocationTokenCapacity,contextPaddingTokens:estimate.contextPaddingTokens,
  notes:[estimate.reason??`Attention KV theoretical ${Number((attentionKvBytes/GiB).toFixed(3))} GiB (${geo.nFullAttention} full-attention layers; recurrent excluded)`,
   'GGUF disk bytes are not GPU weight allocation','Recurrent/SSM state and MTP workspace require backend-specific measurements']};
}

/** Legacy KiB/token baseline: one byte per K/V element, before scales/padding. */
export function hybridKvKbPerToken(geometry){
 if(!geometry||!Number.isSafeInteger(geometry.nFullAttention)||geometry.nFullAttention<0||
  ![geometry.nKvHeads,geometry.keyLength,geometry.valueLength].every(n=>Number.isSafeInteger(n)&&n>0))return null;
 const elements=geometry.nFullAttention*geometry.nKvHeads*(geometry.keyLength+geometry.valueLength);
 return Number.isSafeInteger(elements)?elements/1024:null;
}
