/** Read the named GGUF through the production inspector without registering or loading it. */
import {inspectLocalModel} from '../../app/host/server/local-models.mjs';
import {readGGUF} from '../../app/host/server/gguf.mjs';
const input=process.argv[2];if(!input)throw Error('Model path required');
const {variant}=inspectLocalModel(input);
const gguf=readGGUF(variant.local.path,{includeTensors:true});
const blocks=new Set(gguf.tensors.map(t=>/^blk\.(\d+)\./.exec(t.name)?.[1]).filter(v=>v!==undefined).map(Number));
const recurrent=gguf.metadata['qwen35.attention.recurrent_layers'];
const result={variantId:variant.id,path:variant.local.path,bytes:variant.local.files.reduce((n,f)=>n+f.bytes,0),architecture:variant.local.architecture,layerCount:variant.layerCount,shards:variant.local.shards,
 tensorBlockCount:blocks.size,minTensorBlock:Math.min(...blocks),maxTensorBlock:Math.max(...blocks),recurrentMapLength:Array.isArray(recurrent)?recurrent.length:recurrent?.count??null};
console.log(JSON.stringify(process.argv.includes('--counts-only')?{qwen35:result.architecture==='qwen35',layerCount:result.layerCount,bytes:result.bytes,shards:result.shards,tensorBlockCount:result.tensorBlockCount,minTensorBlock:result.minTensorBlock,maxTensorBlock:result.maxTensorBlock,recurrentMapLength:result.recurrentMapLength}:result));
