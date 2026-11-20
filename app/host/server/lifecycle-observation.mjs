/** Opt-in diagnostics. Only allowlisted settings/numbers leave this module; never log payloads. */
import {createHash, randomUUID} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {open, stat} from 'node:fs/promises';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';

export const OBSERVATION_SCHEMA_VERSION=1;
export const observationEnabled=(env=process.env)=>env.STRATA_RUN_OBSERVATION==='1';
const executeFile=promisify(execFile);
const fields=['contextCapacity','kCodec','vCodec','ngl','offloadLayers','batch','ubatch','parallel','flashAttention','nativeExperts','nativePrefill','memoryPolicyRevision','cachePolicyRevision'];
const codecs=new Set(['f32','f16','bf16','q8_0','q4_0','q4_1','q5_0','q5_1','iq4_nl','turbo2','turbo3','turbo4','int8','int4']);
const number=v=>v!==null&&v!==undefined&&String(v).trim()!==''&&Number.isFinite(Number(v))&&Number(v)>=0?Number(v):null;
const unknown=(reason='not_observed')=>({value:null,source:null,at:null,status:'unavailable',reason});
const measured=(value,source,at=new Date().toISOString())=>({value,source,at,status:'observed',reason:null});
const safeVersion=v=>typeof v==='string'&&/^(?:v?\d+[\w.+-]{0,80}|[a-f0-9]{7,64})$/i.test(v)?v:null;
const setting=(key,v)=>['kCodec','vCodec'].includes(key)?(codecs.has(v)?v:null):['flashAttention','nativeExperts'].includes(key)?(typeof v==='boolean'?v:null):['memoryPolicyRevision','cachePolicyRevision'].includes(key)?safeVersion(v):key==='nativePrefill'?(typeof v==='string'&&/^auto(?::[1-9]\d{0,6})?$/.test(v)?v:number(v)):number(v);

export function requestedFromArgs(args=[]){
 const get=(...names)=>{for(let i=args.length-2;i>=0;i--)if(names.includes(args[i]))return args[i+1];return null;};
 const fa=get('-fa','--flash-attn');
 return {contextCapacity:number(get('-c','--ctx-size','--max-context')),kCodec:get('-ctk','--cache-type-k'),vCodec:get('-ctv','--cache-type-v'),ngl:number(get('-ngl','--n-gpu-layers')),batch:number(get('-b','--batch-size')),ubatch:number(get('-ub','--ubatch-size')),parallel:number(get('-np','--parallel')),flashAttention:fa==='on'?true:fa==='off'?false:null,nativeExperts:args.includes('--native')?true:null,nativePrefill:get('--prefill')};
}

export function requestedFromNativeConfig(config={}){
 const r=requestedFromArgs(Array.isArray(config.args)?config.args:[]);
 const value=(flag)=>{const args=config.args??[];const i=args.lastIndexOf(flag);return i>=0?args[i+1]:null;};
 r.contextCapacity=number(config.max_context)??r.contextCapacity;
 r.parallel=number(config.parallel)??r.parallel;
 r.kCodec=config.cache_type_k??r.kCodec;r.vCodec=config.cache_type_v??r.vCodec;
 // Native --kv describes a different cache implementation; preserve its codec explicitly.
 if(value('--kv')!==null){r.kCodec??=value('--kv');r.vCodec??=value('--kv');}
 if(config.native_experts!==undefined)r.nativeExperts=config.native_experts;
 return Object.fromEntries(fields.map(k=>[k,setting(k,r[k])]));
}

export async function fingerprintFile(path,{headerOnly=false}={}){
 try{
  const info=await stat(path);if(!info.isFile())throw new Error('not_file');
  const hash=createHash('sha256');
  if(headerOnly){const file=await open(path,'r');try{const buffer=Buffer.alloc(Math.min(info.size,1024*1024));const {bytesRead}=await file.read(buffer,0,buffer.length,0);hash.update(buffer.subarray(0,bytesRead));}finally{await file.close();}}
  else for await(const chunk of createReadStream(path))hash.update(chunk);
  return {sha256:measured(hash.digest('hex'),headerOnly?'file_first_1MiB_sha256':'file_sha256'),sizeBytes:measured(info.size,'file_stat')};
 }catch{return {sha256:unknown('file_unavailable'),sizeBytes:unknown('file_unavailable')};}
}

export function createRunObservation({backend='unknown',requested={},requestedSource='launch_arguments',clock=()=>new Date().toISOString(),gpuProbe=probeGpuMemory}={}){
 const startedAt=clock(),effective=Object.fromEntries(fields.map(k=>[k,unknown()]));
 const data={schemaVersion:1,runId:randomUUID(),startedAt,backend:['llama.cpp','PrismML','TurboQuant','native','unknown'].includes(backend)?backend:'unknown',requestedSource:['launch_arguments','native_config','benchmark_declaration'].includes(requestedSource)?requestedSource:'launch_arguments',requested:Object.fromEntries(fields.map(k=>[k,setting(k,requested[k])])),effective,identity:{binding:unknown(),executable:{sha256:unknown(),version:unknown()},model:{metadataSignature:unknown(),sizeBytes:unknown()}},allocations:[],gpu: {dedicated:unknown(),shared_gpu_memory:unknown(),processes:unknown()},outcome:'starting',exitCode:null};
 let active=true,perSequence=false,oom=false,sampleSequence=0;const buffers={};
 const observe=(key,value,source)=>{if(value!==null)effective[key]=measured(value,source,clock());};
 function line(text){
  if(!active||text.length>4096)return;
  let m;
  if(data.backend!=='native'){
   if((m=/^llama_context:\s+n_ctx_per_seq\s*=\s*(\d+)\s*$/.exec(text))){perSequence=true;observe('contextCapacity',number(m[1]),'startup.n_ctx_per_seq');}
   else if(!perSequence&&data.requested.parallel===1&&(m=/^llama_context:\s+n_ctx\s*=\s*(\d+)\s*$/.exec(text)))observe('contextCapacity',number(m[1]),'startup.n_ctx_single_sequence');
   if((m=/^(?:load_tensors|llama_model_load|llama_model_loader|llama_model):\s+offloaded (\d+)\/\d+ layers to GPU\s*$/.exec(text)))observe('offloadLayers',number(m[1]),'startup.offloaded_layers');
   if((m=/^(?:build:|main: build =)\s*(\d+) \(([a-f0-9]{7,40})\)(?: with .*)?$/.exec(text)))data.identity.executable.version=measured(`${m[1]}+${m[2]}`,'startup.build',clock());
   if((m=/^llama_context:\s+(n_batch|n_ubatch|n_seq_max)\s*=\s*(\d+)\s*$/.exec(text))){observe({n_batch:'batch',n_ubatch:'ubatch',n_seq_max:'parallel'}[m[1]],number(m[2]),`startup.${m[1]}`);if(m[1]==='n_seq_max'&&Number(m[2])>1&&!perSequence)effective.contextCapacity=unknown('per_sequence_context_not_reported');}
   if((m=/^llama_context:\s+flash_attn\s*=\s*(0|1|on|off)\s*$/.exec(text)))observe('flashAttention',['1','on'].includes(m[1]),'startup.flash_attn');
   if((m=/^llama_kv_cache:\s+type_([kv])\s*=\s*'(\w+)'\s*$/.exec(text))&&codecs.has(m[2]))observe(m[1]==='k'?'kCodec':'vCodec',m[2],'startup.kv_type');
   if((m=/^(?:load_tensors|llama_model_load|llama_model_loader|llama_model|llama_kv_cache|llama_kv_cache_recurrent|llama_context):\s+(CPU|CPU_Mapped|CUDA\d+|CUDA_Host|Vulkan\d+|SYCL\d+|Metal)\s+(model|KV|recurrent|compute) buffer size\s*=\s*(\d+(?:\.\d+)?)\s+(MiB|GiB)\s*$/.exec(text))){
    const value=number(m[3]);if(value!==null){const kind={model:'weights',KV:'kv',recurrent:'recurrent',compute:'compute'}[m[2]],allocation={kind,device:m[1],value,unit:m[4],bytes:value*(m[4]==='GiB'?1024**3:1024**2),source:'startup.allocation',at:clock(),status:'observed'};const i=data.allocations.findIndex(a=>a.kind===kind&&a.device===m[1]);if(i>=0)data.allocations[i]=allocation;else if(data.allocations.length<128)data.allocations.push(allocation);}
   }
  }else{
   if((m=/^strata generate: (\d+) MiB of weights loaded from /.exec(text)))data.allocations=[{kind:'weights',device:'unknown',value:number(m[1]),unit:'MiB',bytes:number(m[1])*1024**2,source:'startup.native_weights',at:clock(),status:'observed'}];
  }
  if(/^(?:ggml_backend_cuda_buffer_type_alloc_buffer:|strata generate: cudaMalloc\()/.test(text)&&/(?:out of memory|failed)/i.test(text))oom=true;
 }
 return {
  consume(stream,chunk){if(!active)return;const text=(buffers[stream]??'')+String(chunk);const parts=text.split(/\r?\n/);buffers[stream]=parts.pop();if(buffers[stream].length>4096)buffers[stream]='\0';for(const part of parts)line(part);},
  ready(){active=false;for(const k of Object.keys(buffers))delete buffers[k];if(data.outcome==='starting')data.outcome='ready';},
  finish(outcome,exitCode=null){active=false;for(const k of Object.keys(buffers))delete buffers[k];if(data.outcome==='cancelled')return;if(outcome!=='cancelled'&&!['starting','ready'].includes(data.outcome)){if(['startup_exit','error','oom'].includes(data.outcome)&&data.exitCode===null&&Number.isInteger(exitCode))data.exitCode=exitCode;return;}data.outcome=outcome==='cancelled'?'cancelled':oom&&['error','startup_exit'].includes(outcome)?'oom':['startup_exit','error','stopped','completed'].includes(outcome)?outcome:'error';data.exitCode=Number.isInteger(exitCode)?exitCode:null;},
  failure(error,outcome='error'){if(/(?:out of memory|cudaMalloc.*failed|CUDA_ERROR_OUT_OF_MEMORY)/i.test(String(error?.message??'')))oom=true;this.finish(outcome,Number.isInteger(error?.output?.engine_exit_code)?error.output.engine_exit_code:null);},
  observeContext(value,source='native.metrics.max_context'){if(typeof value==='number'&&Number.isSafeInteger(value)&&value>0)observe('contextCapacity',value,source);},
  async identify({executable,version,modelPath,binding=true}={}){const [exe,model]=await Promise.all([fingerprintFile(executable),fingerprintFile(modelPath,{headerOnly:true})]);data.identity={binding:binding?measured(true,'launch_config',clock()):unknown('running_process_identity_unverified'),executable:{sha256:exe.sha256,version:data.identity.executable.version.value!==null?data.identity.executable.version:safeVersion(version)?measured(version,'verified_runtime_manifest',clock()):unknown('version_not_reported')},model:{metadataSignature:model.sha256,sizeBytes:model.sizeBytes}};},
  async sampleGpu(options){const sample=++sampleSequence;try{const gpu=await gpuProbe(options);if(sample===sampleSequence)data.gpu=gpu;}catch{/* diagnostic failure cannot fail a launch */}},
  snapshot(){const actual=effective.contextCapacity.value,asked=data.requested.contextCapacity;return structuredClone({...data,comparison:{scope:'context_capacity_only',status:asked!==null&&actual!==null?(actual<asked?'failed':'comparable'):'unconfirmed',reason:asked!==null&&actual!==null&&actual<asked?'effective_context_below_requested':actual===null?'effective_context_unobserved':asked===null?'requested_context_unavailable':null}});},
 };
}

/** Bounded queries; stderr/error text and full paths never become observation data. */
export async function probeGpuMemory({execute=executeFile,platform=process.platform,ownedPids=[]}={}){
 const opts={timeout:5000,maxBuffer:262144,windowsHide:true,encoding:'utf8'};
 const result={dedicated:unknown('nvidia_smi_unavailable'),shared_gpu_memory:unknown(platform==='win32'?'windows_counter_unavailable':'not_windows'),processes:unknown('nvidia_smi_unavailable')};
 const query=async args=>(await execute('nvidia-smi',args,opts)).stdout;
 await Promise.allSettled([
  (async()=>{const text=await query(['--query-gpu=index,uuid,memory.total,memory.free,memory.used','--format=csv,noheader,nounits']);const values=String(text).trim().split(/\r?\n/).slice(0,64).map(row=>{const [index,uuid,total,free,used]=row.split(',').map(s=>s.trim());return {index:number(index),uuid:/^GPU-[a-f0-9-]+$/i.test(uuid)?uuid:null,totalBytes:number(total)===null?null:number(total)*1024**2,freeBytes:number(free)===null?null:number(free)*1024**2,usedBytes:number(used)===null?null:number(used)*1024**2,status:[total,free,used].some(v=>number(v)===null)?'partial':'observed',reason:[total,free,used].some(v=>number(v)===null)?'memory_field_not_reported':null};}).filter(v=>Number.isInteger(v.index));if(values.length)result.dedicated=measured(values,'nvidia-smi.memory');})(),
  (async()=>{const text=await query(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits']);const values=String(text).trim().split(/\r?\n/).slice(0,512).flatMap(row=>{const [uuid,pid,...path]=row.split(',').map(s=>s.trim()),id=number(pid);if(!Number.isInteger(id)||id<=0)return [];const name=path.join(',').split(/[\\/]/).at(-1);return [{gpuUuid:/^GPU-[a-f0-9-]+$/i.test(uuid)?uuid:null,pid:id,executable:/^[a-z0-9_.+-]{1,100}$/i.test(name)?name:null,ownership:ownedPids.includes(id)?'current-Strata':/^[a-z0-9_.+-]{1,100}$/i.test(name)?'external':'unknown'}];});result.processes=measured(values,'nvidia-smi.compute-apps');})(),
  (async()=>{if(platform!=='win32')return;const script='Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory | Select-Object Name,SharedUsage | ConvertTo-Json -Compress';const {stdout}=await execute('powershell.exe',['-NoProfile','-NonInteractive','-Command',script],opts);const parsed=JSON.parse(stdout);const values=(Array.isArray(parsed)?parsed:[parsed]).slice(0,64).flatMap(row=>/^luid_0x[\da-f]+_0x[\da-f]+_phys_\d+$/i.test(row?.Name)&&number(row.SharedUsage)!==null?[{adapter:row.Name,usedBytes:number(row.SharedUsage)}]:[]);if(values.length)result.shared_gpu_memory=measured(values,'windows.GPUAdapterMemory.SharedUsage');})(),
 ]);
 return result;
}
