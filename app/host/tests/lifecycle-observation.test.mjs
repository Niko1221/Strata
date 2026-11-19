import test from 'node:test';
import assert from 'node:assert/strict';
import {createRunObservation, requestedFromArgs, requestedFromNativeConfig, probeGpuMemory, fingerprintFile} from '../server/lifecycle-observation.mjs';
import {mkdtempSync, writeFileSync,readFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

const argv=['-c','131072','-ctk','q8_0','-ctv','q8_0','-ngl','999','-b','2048','-ub','512','-np','1','-fa','on'];
test('native actual capacity accepts numeric positive integers only',()=>{
 const c=createRunObservation({backend:'native',requested:{contextCapacity:131072}});
 for(const value of [null,false,true,'131072',0,-1,0.5,Infinity])c.observeContext(value);
 assert.equal(c.snapshot().effective.contextCapacity.value,null);
 c.observeContext(32768);assert.equal(c.snapshot().comparison.status,'failed');
});
test('requested settings are snapshotted and unknown actual values are never estimates',()=>{
 const requested=requestedFromArgs(argv),capture=createRunObservation({backend:'llama.cpp',requested});
 requested.contextCapacity=4096;
 const s=capture.snapshot();
 assert.equal(s.schemaVersion,1);assert.equal(s.requested.contextCapacity,131072);
 assert.deepEqual(s.effective.contextCapacity,{value:null,source:null,at:null,status:'unavailable',reason:'not_observed'});
 assert.equal(s.effective.kCodec.value,null);assert.equal(s.comparison.status,'unconfirmed');
});
test('known complete startup lines produce numeric allocations, split chunks work, raw/private text never survives',()=>{
 const capture=createRunObservation({backend:'llama.cpp',requested:requestedFromArgs(argv)});
 capture.consume('err','llama_context: n_ctx_per_');capture.consume('err','seq = 131072\n');
 capture.consume('err','llama_model_load: offloaded 49/49 layers to GPU\nllama_kv_cache: CUDA0 KV buffer size = 128.50 MiB\nllama_model_load: CUDA0 model buffer size = 123.00 MiB\nllama_context: CUDA0 compute buffer size = 12.00 MiB\n');
 capture.consume('out','prompt private-persona API_KEY=secret n_ctx = 1\n');
 capture.consume('err','llama_context: n_batch = 2048\nllama_context: n_ubatch = 512\n');
 const s=capture.snapshot();assert.equal(s.effective.contextCapacity.value,131072);assert.equal(s.effective.offloadLayers.value,49);
 assert.equal(s.allocations.find(a=>a.kind==='kv').bytes,128.5*1024**2);assert.equal(s.allocations.length,3);
 assert.equal(s.effective.batch.value,2048);assert.equal(s.effective.ubatch.value,512);
 assert.ok(!/private|secret|API_KEY|prompt/.test(JSON.stringify(s)));
});
test('smaller observed per-sequence context fails comparability and later global context cannot hide it',()=>{
 const c=createRunObservation({backend:'llama.cpp',requested:requestedFromArgs(argv)});
 c.consume('err','llama_context: n_ctx = 131072\nllama_context: n_ctx_per_seq = 32768\nllama_context: n_ctx = 131072\n');
 assert.equal(c.snapshot().effective.contextCapacity.value,32768);assert.equal(c.snapshot().comparison.status,'failed');
});
test('startup observation closes at ready and ignores subsequent prompt-shaped log lines',()=>{
 const c=createRunObservation({backend:'llama.cpp',requested:requestedFromArgs(argv)});
 c.ready();c.consume('err','llama_context: n_ctx_per_seq = 1\n');assert.equal(c.snapshot().effective.contextCapacity.value,null);
 c.finish('cancelled');assert.equal(c.snapshot().outcome,'cancelled');
});
test('OOM/startup exit record only bounded categories and numeric exit code',()=>{
 const c=createRunObservation({backend:'llama.cpp',requested:{}});
 c.consume('err','ggml_backend_cuda_buffer_type_alloc_buffer: allocating 123.00 MiB on device 0: cudaMalloc failed: out of memory\n');
 c.finish('startup_exit',7);const s=c.snapshot();assert.equal(s.outcome,'oom');assert.equal(s.exitCode,7);
});
test('probe separates dedicated and shared memory; multiple cards and process ownership preserve privacy',async()=>{
 const calls=[];const execute=async(command,args,options)=>{calls.push({command,args,options});
  if(command==='powershell.exe')return {stdout:JSON.stringify([{Name:'luid_0x00000000_0x00001234_phys_0',SharedUsage:4194304}])};
  if(args[0].startsWith('--query-gpu'))return {stdout:'0, GPU-aaaa, 16384, 12000, 4384\n1, GPU-bbbb, N/A, N/A, N/A\n'};
  return {stdout:'GPU-aaaa, 123, C:\\Users\\private\\llama-server.exe\nGPU-aaaa, 456, C:\\Users\\other\\python.exe\nGPU-bbbb, N/A, [Not Found]\n'};};
 const s=await probeGpuMemory({execute,platform:'win32',ownedPids:[123]});
 assert.equal(s.dedicated.value[0].totalBytes,16384*1024**2);assert.equal(s.dedicated.value[1].usedBytes,null);
 assert.equal(s.shared_gpu_memory.value[0].usedBytes,4194304);assert.equal(s.processes.value[0].ownership,'current-Strata');
 assert.equal(s.processes.value[1].ownership,'external');assert.equal(s.processes.value[0].executable,'llama-server.exe');
 assert.ok(!JSON.stringify(s).includes('Users'));assert.ok(calls.every(c=>c.options.timeout<=5000&&c.options.maxBuffer<=262144&&c.options.windowsHide));
});
test('unavailable GPU probe returns null and safe fixed reasons without stderr or exception text',async()=>{
 const s=await probeGpuMemory({execute:async()=>{throw new Error('C:\\Users\\secret\\password');},platform:'win32'});
 assert.equal(s.dedicated.value,null);assert.equal(s.processes.value,null);assert.equal(s.shared_gpu_memory.value,null);
 assert.ok(!JSON.stringify(s).includes('secret'));
});
test('identity uses actual content hashes without retaining local paths',async()=>{
 const dir=mkdtempSync(join(tmpdir(),'observation-')),path=join(dir,'private.gguf');writeFileSync(path,'fixture');
 const hash=await fingerprintFile(path);assert.match(hash.sha256.value,/^[a-f0-9]{64}$/);assert.equal(hash.sizeBytes.value,7);
 assert.ok(!JSON.stringify(hash).includes(dir));
});
test('native requested values are explicit config/argv values; private config is discarded',()=>{
 const r=requestedFromNativeConfig({args:['--max-context','131072','--prefill','512','--kv','int8','--native','/private/model'],parallel:2,cache_type_k:'q8_0',cache_type_v:'q8_0',system_prompt:'PRIVATE',api_key:'SECRET'});
 assert.equal(r.contextCapacity,131072);assert.equal(r.nativePrefill,512);assert.equal(r.nativeExperts,true);assert.equal(r.parallel,2);assert.equal(r.kCodec,'q8_0');assert.ok(!/PRIVATE|SECRET/.test(JSON.stringify(r)));
});
test('total context alone cannot confirm per-slot capacity with parallel requests',()=>{
 const c=createRunObservation({backend:'llama.cpp',requested:{contextCapacity:131072,parallel:2}});
 c.consume('err','llama_context: n_ctx = 131072\nllama_context: n_seq_max = 2\n');
 assert.equal(c.snapshot().effective.contextCapacity.value,null);assert.equal(c.snapshot().comparison.status,'unconfirmed');
});
test('current llama load_tensors messages and recurrent allocations are recognized',()=>{
 const c=createRunObservation({backend:'PrismML',requested:{}});
 c.consume('err','load_tensors: offloaded 41/41 layers to GPU\nload_tensors: CUDA0 model buffer size = 987.00 MiB\nllama_kv_cache_recurrent: CUDA0 recurrent buffer size = 4.00 MiB\n');
 assert.equal(c.snapshot().effective.offloadLayers.value,41);assert.equal(c.snapshot().allocations.length,2);
});
test('published schema defines all snapshot keys and measurement null semantics',()=>{
 const schema=JSON.parse(readFileSync(new URL('../../../bench/strata-16gb/run-observation.schema.json',import.meta.url),'utf8'));
 const s=createRunObservation().snapshot();assert.deepEqual(Object.keys(s).sort(),schema.required.toSorted());
 assert.deepEqual(Object.keys(s.requested).sort(),schema.$defs.requested.required.toSorted());
 assert.deepEqual(Object.keys(s.effective).sort(),schema.$defs.effective.required.toSorted());
 for(const value of Object.values(s.effective))assert.deepEqual(Object.keys(value).sort(),schema.$defs.measurement.required.toSorted());
});
test('numeric startup build wins over a runtime package version and discards compiler paths',async()=>{
 const c=createRunObservation({backend:'unknown'});
 c.consume('err','build: 10826 (c9c34bc9c) with C:\\Users\\private\\compiler\n');
 await c.identify({version:'0.4.0'});
 assert.equal(c.snapshot().identity.executable.version.value,'10826+c9c34bc9c');
 assert.ok(!JSON.stringify(c.snapshot()).includes('private'));
});
test('a native safer retry cannot relabel the original requested capacity',()=>{
 const config={args:['--max-context','131072','--kv','int8']};
 const c=createRunObservation({backend:'native',requested:requestedFromNativeConfig(config),requestedSource:'native_config'});
 config.args=['--max-context','32768','--kv','int8'];c.ready();c.observeContext(32768);
 const s=c.snapshot();assert.equal(s.requested.contextCapacity,131072);assert.equal(s.effective.contextCapacity.value,32768);assert.equal(s.comparison.status,'failed');
});
test('a propagated generic error preserves the first startup exit and OOM numeric code',()=>{
 for(const oom of [false,true]){
  const c=createRunObservation();
  if(oom)c.consume('err','ggml_backend_cuda_buffer_type_alloc_buffer: allocating 123.00 MiB on device 0: cudaMalloc failed: out of memory\n');
  c.finish('startup_exit',17);c.failure(new Error('generic source wrapper failure'));
  assert.equal(c.snapshot().outcome,oom?'oom':'startup_exit');assert.equal(c.snapshot().exitCode,17);
 }
});
