/** Local GGUF references. Adding/removing a reference never copies or deletes the user's weights. */
import { accessSync, constants, mkdirSync, readdirSync, readFileSync, realpathSync, renameSync, statSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { basename, dirname, extname, isAbsolute, join } from 'node:path';
import { homedir } from 'node:os';
import { readGGUF } from './gguf.mjs';
import { extractHybridGeometry, estimateModelMemory, modelMemoryFromMetadata } from './model-memory.mjs';
// Reject known auxiliary/embedding architectures. The installed llama-server checks text-model support at load time.
const NON_GENERATIVE=new Set(`clip bert modern-bert nomic-bert nomic-bert-moe neo-bert jina-bert-v2 jina-bert-v3 eurobert
 t5encoder gemma-embedding llama-embed wavtokenizer-dec pangu-embedded eagle3 dflash paddleocr qwen3tts pockettts talkie`.trim().split(/\s+/));
const QUANTS={0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',7:'Q8_0',8:'Q5_0',9:'Q5_1',10:'Q2_K',11:'Q3_K_S',12:'Q3_K_M',13:'Q3_K_L',14:'Q4_K_S',15:'Q4_K_M',16:'Q5_K_S',17:'Q5_K_M',18:'Q6_K',19:'IQ2_XXS',20:'IQ2_XS',21:'Q2_K_S',22:'IQ3_XS',23:'IQ3_XXS',24:'IQ1_S',25:'IQ4_NL',26:'IQ3_S',27:'IQ3_M',28:'IQ2_S',29:'IQ2_M',30:'IQ4_XS',31:'IQ1_M',32:'BF16',141:'PQ2_0',142:'PQ2_0',143:'PTQ1_0'};
const clean=value=>String(value??'').replace(/[\x00-\x1f\x7f]/g,' ').trim().slice(0,160);
const genericName=value=>!clean(value)||/^(hf|hugging[ -]?face|model|unknown)$/i.test(clean(value));
const fileName=file=>basename(file).replace(/-00001-of-\d{5}/,'').replace(/\.gguf$/i,'');
function namedVariant(v,name){return {...v,name,label:`${name} · ${v.quant}`};}
function fullPath(input){if(typeof input!=='string'||!input.trim()||input.includes('\0'))throw new Error('Choose a model file or enter its full path.');const path=input.trim().replace(/^~(?=$|[\\/])/,homedir());if(!isAbsolute(path))throw new Error('Enter a full path.');try{return realpathSync(path);}catch{throw new Error('This file or folder is unavailable. Check its location and permissions.');}}
export function browseLocalModels(input){
 const directory=fullPath(input);if(!statSync(directory).isDirectory())throw new Error('Choose a folder to browse.');
 const entries=readdirSync(directory,{withFileTypes:true}).filter(e=>!e.name.startsWith('.')).map(e=>{try{const path=join(directory,e.name),s=statSync(path);if(s.isDirectory())return {name:e.name,path,kind:'folder'};if(s.isFile()&&extname(e.name).toLowerCase()==='.gguf')return {name:e.name,path,kind:'model',bytes:s.size};}catch{}return null;}).filter(Boolean).sort((a,b)=>(a.kind==='folder'?0:1)-(b.kind==='folder'?0:1)||a.name.localeCompare(b.name));
 return {directory,parent:dirname(directory)===directory?null:dirname(directory),entries:entries.slice(0,1000),truncated:entries.length>1000};
}
export function inspectLocalModel(input){
 let file=fullPath(input);if(extname(file).toLowerCase()!=='.gguf')throw new Error('Select a .gguf model file. Folders and adapters cannot be loaded as a chat model.');
 if(/^mmproj/i.test(basename(file)))throw new Error('This is a picture reader (mmproj), not a model. Choose the model’s own .gguf file; a model that reads pictures needs its reader in the same folder.');
 if(!statSync(file).isFile())throw new Error('Select the model’s .gguf file, rather than its folder.');
 accessSync(file,constants.R_OK);
 let selected=readGGUF(file);const split=basename(file).match(/^(.*)-(\d{5})-of-(\d{5})\.gguf$/i);
 const declared=selected.metadata['split.count']??1;
 if(declared>1||split){
  if(!split||Number(split[3])!==declared)throw new Error('Split GGUF filenames must include 00001-of-000NN and match their metadata.');
  file=join(dirname(file),`${split[1]}-00001-of-${split[3]}.gguf`);
  try{selected=readGGUF(file);}catch{throw new Error('The first shard is missing or invalid. Keep all model shards in the same folder.');}
 }
 const m=selected.metadata,arch=m['general.architecture'];
 if(typeof arch!=='string'||!arch||!(/^[a-z][a-z0-9_-]{0,63}$/).test(arch))throw new Error('This GGUF has a missing or invalid model architecture. Choose a complete text-generation model.');
 if(NON_GENERATIVE.has(arch))throw new Error(`The declared architecture ${clean(arch)} is not a text-generation model. Choose a chat model GGUF instead.`);
 if(m['general.type']&&m['general.type']!=='model')throw new Error('This is an adapter or auxiliary file. Choose the full model GGUF.');
 if(!m['tokenizer.ggml.model']||!m['tokenizer.ggml.tokens']?.count)throw new Error('This GGUF does not include its tokenizer. Choose the full model, not an adapter. A picture reader (mmproj) goes next to its model, not in its place.');
 const integer=(key,min,max)=>{const value=m[key];if(!Number.isInteger(value)||value<min||value>max)throw new Error(`Missing or invalid ${key} metadata. This model cannot be configured safely.`);return value;};
 const optionalInteger=(key,min,max)=>m[key]===undefined?null:integer(key,min,max);
 const estimateInteger=(key,max)=>Number.isInteger(m[key])&&m[key]>=1&&m[key]<=max?m[key]:null;
 const declaredContext=optionalInteger(`${arch}.context_length`,1024,100_000_000),context=declaredContext??4096;
 const layers=optionalInteger(`${arch}.block_count`,1,4096),embedding=optionalInteger(`${arch}.embedding_length`,1,1_000_000),heads=estimateInteger(`${arch}.attention.head_count`,8192);
 const kvHeads=m[`${arch}.attention.head_count_kv`]===undefined?heads:estimateInteger(`${arch}.attention.head_count_kv`,8192);
 const keyLength=m[`${arch}.attention.key_length`]===undefined?(embedding&&heads&&embedding%heads===0?embedding/heads:null):estimateInteger(`${arch}.attention.key_length`,1_000_000);
 const valueLength=m[`${arch}.attention.value_length`]===undefined?keyLength:estimateInteger(`${arch}.attention.value_length`,1_000_000);
 const experts=optionalInteger(`${arch}.expert_count`,0,100_000)??0;
 const memoryGeometry=modelMemoryFromMetadata(m);
 // Legacy display baseline is KiB/token at one byte per K/V element, before
 // codec scale bytes. Hybrid imports count full-attention trunk layers only.
 const attentionLayers=memoryGeometry.kind==='hybrid'?memoryGeometry.fullAttentionLayers:['qwen35','qwen35moe'].includes(arch)?null:layers;
 const kvKbPerToken=attentionLayers!==null&&heads&&kvHeads&&keyLength&&valueLength?attentionLayers*kvHeads*(keyLength+valueLength)/1024:null;
 const quantizedKvSupported=keyLength!==null&&valueLength!==null&&keyLength%32===0&&valueLength%32===0;
 const count=m['split.count']??1;if(!Number.isInteger(count)||count<1||count>256)throw new Error('This split model has an unsupported number of shards.');
 const files=[],names=[];let totalTensors=0,parameters=0;
 for(let i=0;i<count;i++){
  const path=count===1?file:join(dirname(file),`${split[1]}-${String(i+1).padStart(5,'0')}-of-${split[3]}.gguf`);
  let info;try{info=i===0?selected:readGGUF(path);}catch(e){throw new Error(`${basename(path)}: ${e.message} Keep every shard in the same folder.`);}
  if(count>1&&((info.metadata['split.no']!==i)||(info.metadata['split.count']!==count)||(info.metadata['split.tensors.count']!==m['split.tensors.count'])))throw new Error('The model shards do not form a matching set. Re-download the missing or mismatched shard.');
  if(info.metadata['general.architecture']&&info.metadata['general.architecture']!==arch)throw new Error('These shards declare different architectures.');
  files.push({path:realpathSync(path),bytes:info.size,mtimeMs:info.mtimeMs,ctimeMs:info.ctimeMs});names.push(basename(path));totalTensors+=info.tensorCount;parameters+=info.parameters;
 }
 if(totalTensors<1||count>1&&totalTensors!==m['split.tensors.count'])throw new Error('This model is missing tensors or model shards.');
 const bytes=files.reduce((sum,f)=>sum+f.bytes,0),name=genericName(m['general.name'])?clean(fileName(file)):clean(m['general.name']),quant=QUANTS[m['general.file_type']]??'GGUF';
 const id=`local-${createHash('sha256').update(file).digest('hex').slice(0,24)}`,gb=bytes/1024**3;
  const modelMemory={...memoryGeometry,diskBytes:bytes};
  const hybrid=extractHybridGeometry(m);
  const memoryPlan=estimateModelMemory({metadata:m,diskBytes:bytes,context:Math.min(context,131072),sequences:1,kType:'q8_0',vType:'q8_0'});
  const local={path:file,files,architecture:arch,hasChatTemplate:!!m['tokenizer.chat_template'],contextLength:context,keyLength,valueLength,shards:count,modelMemory,metadataSignature:hybrid.metadataSignature,
    nFullAttention:hybrid.nFullAttention,nRecurrent:hybrid.nRecurrent,fullAttentionInterval:hybrid.fullAttentionInterval,memoryConfidence:hybrid.confidence};
  const variant={id,name,family:'Local models',quant,label:`${name} · ${quant}`,params:parameters>=1e9?`${(parameters/1e9).toFixed(1)}B`:`${Math.round(parameters/1e6)}M`,blurb:'Local model. Engine compatibility is checked when loading.',role:'option',tier:'local',ramClass:{minGb:gb+2,maxGb:gb+6,label:'From GGUF metadata'},diskGb:gb,downloadGb:0,contextMax:Math.min(context,1048576),layerCount:layers,experts,kvKbPerToken,modelMemory,hasDraft:false,kvModes:quantizedKvSupported?['INT8','FP16']:['FP16'],tags:['local'],source:{repo:'local',file:basename(file),bytes},local,memoryPlan:{attentionKvGiB:memoryPlan.attentionKvGiB,nFullAttention:memoryPlan.nFullAttention,nRecurrent:memoryPlan.nRecurrent,confidence:memoryPlan.confidence,fits:null}};
  const hybridNote=hybrid.nFullAttention!==null&&hybrid.nRecurrent>0
    ?[`Hybrid Attention: ${hybrid.nFullAttention} full-attention / ${hybrid.nRecurrent} recurrent layers (KV grows only on attention layers).`]
    :[];
  return {variant,files:names,warnings:[...(local.hasChatTemplate?[]:['No chat template was declared. Response formatting depends on the engine.']),...(declaredContext===null?['Context length is not declared. Start at 4K tokens; the engine checks the actual limit when loading.']:[]),...(kvKbPerToken===null?['KV memory metadata is incomplete. Actual allocation is unknown; the engine checks it when loading.']:[]),...(!quantizedKvSupported?['This GGUF cannot use a quantized K/V cache with its declared attention width. Choose FP16 for KV mode and leave K/V cache precision at Inherit or FP16.']:[]),...hybridNote,'Memory is estimated from metadata; engine support and actual allocation are checked when loading. Attention KV theory is not a full GPU fit claim.']};
}
export function createLocalModels(dataDir){
 mkdirSync(dataDir,{recursive:true});const file=join(dataDir,'local-models.json');let records=[];
 try{const saved=JSON.parse(readFileSync(file,'utf8'));if(Array.isArray(saved.models))records=saved.models.filter(v=>v?.id?.startsWith('local-')&&v.local?.files?.length&&isAbsolute(v.local.path));}catch{}
 // Upgrade old placeholder labels in memory without modifying model weights.
 records=records.map(v=>genericName(v.name)&&!v.displayNameExplicit?namedVariant(v,clean(fileName(v.local.path))):v);
 const inspect=input=>{const preview=inspectLocalModel(input),previous=records.find(v=>v.id===preview.variant.id);if(previous)preview.variant={...namedVariant(preview.variant,previous.name),displayNameExplicit:previous.displayNameExplicit};return preview;};
 const persist=next=>{const temp=file+'.tmp';writeFileSync(temp,JSON.stringify({schema:1,models:next},null,2));renameSync(temp,file);records=next;};
 return {inspect,list:()=>structuredClone(records),add:preview=>{const v=preview.variant;const previous=records.find(row=>row.id===v.id);persist([...records.filter(row=>row.id!==v.id),{...v,importedAt:previous?.importedAt??new Date().toISOString()}]);return v;},remove:id=>persist(records.filter(v=>v.id!==id)),ready:v=>v.local.files.every(f=>{try{const s=statSync(f.path);return s.isFile()&&s.size===f.bytes&&s.mtimeMs===f.mtimeMs&&s.ctimeMs===f.ctimeMs;}catch{return false;}})};
}
