import test from 'node:test';
import assert from 'node:assert/strict';
import {collectRun,parseOptions} from '../../../bench/strata-16gb/run.mjs';

test('runner requires explicit opt-in, loopback target, local identities and fixed capacity/codecs',()=>{
 assert.throws(()=>parseOptions([]),/enable-observation/);
 assert.throws(()=>parseOptions(['--enable-observation','--url','https://external.test','--model','m','--backend','b','--context','131072','--k','q8_0','--v','q8_0','--input-tokens','4096','--out','r.json']),/loopback/);
 const o=parseOptions(['--enable-observation','--model','m','--backend','b','--context','131072','--k','q8_0','--v','q8_0','--input-tokens','4096','--out','r.json']);assert.equal(o.requested.contextCapacity,131072);assert.equal(o.requested.kCodec,'q8_0');
});
const gpuProbe=async()=>({dedicated:{value:null,source:null,at:null,status:'unavailable',reason:'fixture'},shared_gpu_memory:{value:null,source:null,at:null,status:'unavailable',reason:'fixture'},processes:{value:null,source:null,at:null,status:'unavailable',reason:'fixture'}});
test('existing runtime without observation route runs exact synthetic tokens but keeps missing actuals null',async()=>{
 const calls=[];const fetcher=async(url,options)=>{calls.push([url,options]);if(url.endsWith('/tokenize'))return Response.json({tokens:Array.from({length:5000},(_,i)=>i)});if(url.endsWith('/completion'))return Response.json({content:'DO NOT RETAIN OUTPUT',timings:{prompt_n:4096,predicted_n:32,prompt_per_second:321,predicted_per_second:12}});return new Response('',{status:404});};
 const r=await collectRun({url:'http://127.0.0.1:8080',requested:{contextCapacity:131072,kCodec:'q8_0',vCodec:'q8_0'},inputTokens:4096,fetcher,gpuProbe});
 assert.equal(r.observation.effective.contextCapacity.value,null);assert.equal(r.observation.comparison.status,'unconfirmed');assert.equal(r.workload.inputTokens.value,4096);
 const body=JSON.parse(calls.find(c=>c[0].endsWith('/completion'))[1].body);assert.equal(body.prompt.length,4096);assert.equal(body.cache_prompt,false);
 assert.ok(!JSON.stringify(r).includes('DO NOT RETAIN OUTPUT'));
});
test('a smaller actual context fails without silently shortening the request or sending workload',async()=>{
 const calls=[];const fetcher=async(url)=>{calls.push(url);return url.endsWith('/props')?Response.json({default_generation_settings:{n_ctx:65536}}):new Response('',{status:404});};
 const r=await collectRun({url:'http://localhost:8080',requested:{contextCapacity:131072},inputTokens:4096,fetcher,gpuProbe});
 assert.equal(r.observation.comparison.status,'failed');assert.equal(r.failure,'effective_context_below_requested');assert.ok(!calls.some(u=>u.endsWith('/completion')));
});
