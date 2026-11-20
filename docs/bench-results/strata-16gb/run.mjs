#!/usr/bin/env node
/** Attach-only synthetic benchmark. Never launches, stops, moves or converts a model. */
import {writeFile,stat} from 'node:fs/promises';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRunObservation} from '../../apps/host/server/lifecycle-observation.mjs';

const finite=v=>typeof v==='number'&&Number.isFinite(v)&&v>=0?v:null;
const measurement=(value,source)=>({value:finite(value),source:finite(value)===null?null:source,at:new Date().toISOString(),status:finite(value)===null?'unavailable':'observed',reason:finite(value)===null?'not_reported':null});
function loopback(value){const url=new URL(value);if(url.protocol!=='http:'||!['127.0.0.1','localhost','[::1]'].includes(url.hostname)||url.username||url.password||url.search||url.hash||url.pathname!=='/')throw new Error('Use a loopback HTTP origin without credentials.');return url.origin;}
export function parseOptions(argv){
 if(!argv.includes('--enable-observation'))throw new Error('Pass --enable-observation to opt in.');
 const allowed=new Set(['--url','--model','--backend','--backend-kind','--context','--k','--v','--input-tokens','--output-tokens','--out','--api-key-env']);const options={};
 for(let i=0;i<argv.length;i++){const flag=argv[i];if(flag==='--enable-observation')continue;if(!allowed.has(flag)||!argv[i+1]||argv[i+1].startsWith('--')||flag in options)throw new Error('Unknown, repeated or missing option.');options[flag]=argv[++i];}
 for(const flag of ['--model','--backend','--context','--k','--v','--input-tokens','--out'])if(!options[flag])throw new Error('Supply model, backend, context, codecs, input-tokens and out.');
 const positive=(flag,fallback)=>{const n=Number(options[flag]??fallback);if(!Number.isSafeInteger(n)||n<=0)throw new Error('Token counts must be positive integers.');return n;};
 const context=positive('--context'),inputTokens=positive('--input-tokens'),outputTokens=positive('--output-tokens',32);
 if(inputTokens+outputTokens>context||inputTokens>131072||context>2097152||outputTokens>4096)throw new Error('Workload must fit the explicitly requested capacity.');
 if(!['q8_0','q4_0','q4_1','q5_0','q5_1','f16','bf16','turbo2','turbo3','turbo4','int8','int4'].includes(options['--k'])||!['q8_0','q4_0','q4_1','q5_0','q5_1','f16','bf16','turbo2','turbo3','turbo4','int8','int4'].includes(options['--v']))throw new Error('Unsupported explicit codec.');
 const backend=options['--backend-kind']??'llama.cpp';if(!['llama.cpp','PrismML','TurboQuant','native'].includes(backend))throw new Error('Choose a supported backend kind.');
 if(options['--api-key-env']&&!/^[a-z_][a-z0-9_]*$/i.test(options['--api-key-env']))throw new Error('Choose an environment-variable name for the API key.');
 return {url:loopback(options['--url']??'http://127.0.0.1:8080'),modelPath:resolve(options['--model']),backendPath:resolve(options['--backend']),backend,requested:{contextCapacity:context,kCodec:options['--k'],vCodec:options['--v']},inputTokens,outputTokens,out:resolve(options['--out']),apiKey:options['--api-key-env']?process.env[options['--api-key-env']]??'':''};
}

export async function collectRun({url,backendPath,modelPath,backend='llama.cpp',requested,inputTokens,outputTokens=32,apiKey='',fetcher=fetch,gpuProbe,signal}={}){
 url=loopback(url);
 const capture=createRunObservation({backend,requested,gpuProbe,requestedSource:'benchmark_declaration'});
 const report={reportVersion:1,observation:null,identityBinding:'supplied_local_files_not_verified_against_running_process',workload:{kind:'synthetic_token_array',targetInputTokens:inputTokens,requestedOutputTokens:outputTokens,cachePromptRequested:false,inputTokens:measurement(null),outputTokens:measurement(null),backendTokensCached:measurement(null),promptTokensPerSecond:measurement(null),outputTokensPerSecond:measurement(null),elapsedSeconds:measurement(null)},failure:null};
 const headers={'content-type':'application/json',...(apiKey?{authorization:`Bearer ${apiKey}`}:{})};
 const get=async path=>{try{const r=await fetcher(url+path,{headers,signal:AbortSignal.any([AbortSignal.timeout(5000),...(signal?[signal]:[])])});return r.ok?await r.json():null;}catch{return null;}};
 await Promise.all([capture.identify({executable:backendPath,modelPath,binding:false}),capture.sampleGpu()]);
 report.gpuBefore=capture.snapshot().gpu;
 const [props,status]=await Promise.all([get('/props'),get('/v1/status')]);
 // Only these named numeric capacity fields are accepted. Do not infer from available VRAM.
 const context=finite(props?.default_generation_settings?.n_ctx)??finite(status?.cache_max_tokens);
 if(context!==null)capture.observeContext(context,props?.default_generation_settings?.n_ctx!==undefined?'llama.props.n_ctx':'native.status.cache_max_tokens');
 if(capture.snapshot().comparison.status==='failed'){capture.finish('error');report.failure='effective_context_below_requested';report.observation=capture.snapshot();return report;}
 try{
  const post=async(path,body)=>{const r=await fetcher(url+path,{method:'POST',headers,body:JSON.stringify(body),signal:AbortSignal.any([AbortSignal.timeout(1800000),...(signal?[signal]:[])])});if(!r.ok)throw new Error('request_failed');return r.json();};
  // This generated corpus never reads a prompt, persona or memory file. The tokenizer selects the exact input count.
  const corpus='Synthetic benchmark sequence. '+Array.from({length:inputTokens+128},(_,i)=>` item ${i%997}`).join('');
  const tokenized=await post('/tokenize',{content:corpus,add_special:false});
  if(!Array.isArray(tokenized.tokens)||tokenized.tokens.length<inputTokens||tokenized.tokens.some(t=>!Number.isSafeInteger(t)||t<0))throw new Error('tokenizer_unavailable');
  const prompt=tokenized.tokens.slice(0,inputTokens),start=performance.now();
  const result=await post('/completion',{prompt,n_predict:outputTokens,temperature:0,seed:1,cache_prompt:false,stream:false});
  const actual=finite(result.timings?.prompt_n)??finite(result.tokens_evaluated);
  report.workload.inputTokens=measurement(actual,'completion.prompt_n');
  report.workload.outputTokens=measurement(finite(result.timings?.predicted_n)??finite(result.tokens_predicted),'completion.predicted_n');
  report.workload.backendTokensCached=measurement(result.tokens_cached,'completion.tokens_cached');
  report.workload.promptTokensPerSecond=measurement(result.timings?.prompt_per_second,'completion.prompt_per_second');
  report.workload.outputTokensPerSecond=measurement(result.timings?.predicted_per_second,'completion.predicted_per_second');
  report.workload.elapsedSeconds=measurement((performance.now()-start)/1000,'monotonic_clock');
  if(actual!==inputTokens){report.failure='actual_input_tokens_differ_from_target';capture.finish('error');}else capture.finish('completed');
 }catch{report.failure=signal?.aborted?'cancelled':'workload_or_tokenizer_unavailable';capture.finish(signal?.aborted?'cancelled':'error');}
 await capture.sampleGpu();report.observation=capture.snapshot();return report;
}

async function main(){
 const options=parseOptions(process.argv.slice(2));
 for(const path of [options.modelPath,options.backendPath])if(!(await stat(path)).isFile())throw new Error('Use existing local model and backend files.');
 const controller=new AbortController();process.once('SIGINT',()=>controller.abort());
 const report=await collectRun({...options,signal:controller.signal});
 await writeFile(options.out,JSON.stringify(report,null,2)+'\n',{flag:'wx',mode:0o600});
 console.log(JSON.stringify({outcome:report.observation.outcome,comparison:report.observation.comparison.status,failure:report.failure}));
 if(report.failure)process.exitCode=1;
}
if(process.argv[1]&&pathToFileURL(resolve(process.argv[1])).href===import.meta.url)main().catch(()=>{console.error('Benchmark could not run. Check explicit options, local files and a new output path.');process.exitCode=1;});
