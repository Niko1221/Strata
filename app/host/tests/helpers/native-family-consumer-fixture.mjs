/** Feed the actual Host builder's run configs to the Python consumer regression. No real engine. */
import {readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import assert from 'node:assert/strict';
import {nativeFixture,finishJob} from './native-fixture.mjs';
import {q36FileSha256} from '../../server/q36-run-contract.mjs';

process.env.STRATA_HAKUA_CHATBOT='1';
const cap={schema:1,family:'qwen35moe',layoutVersion:1,scaleSchema:'none',cacheTypesK:['f32','f16'],cacheTypesV:['f32'],
 flags:['--cache-type-k','--cache-type-v'],quality:'experimental-unqualified',binarySha256:'ab'.repeat(32)};
const result={};
for(const [key,family,arch] of [['qwen','qwen','qwen4exp'],['qwen-catalog','qwen',undefined],['qwen36','qwen36',undefined]]){
 const f=nativeFixture({packaged:true,q36CapabilityProbe:async()=>cap});
 try{
  f.catalog.families[0].family=family;
  const path=join(f.repo,'strata-iq2_xs.json');
  const original={parallel:1,args:['--max-context','256'],...(arch?{arch}:{}),
   ...(family==='qwen'?{exe:'C:/fixture/strata-q36-lookalike/strata.exe',model_name:'Qwen3.6-looking label'}
     :{exe:'C:/fixture/neutral-runtime.exe',model_name:'Neutral model'})};
  writeFileSync(path,JSON.stringify(original));
  await f.control.refresh(true);
  if(family==='qwen36')await f.control.saveChecked('iq2_xs',{values:{cache_type_k:'f16',cache_type_v:'f32'}});
  const job=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs'}).id);
  if(job.state!=='done')throw new Error(job.error);
  const run=JSON.parse(readFileSync(f.launchConfigs.at(-1),'utf8'));
  const saved=JSON.parse(readFileSync(path,'utf8'));
  for(const field of ['arch','exe','model_name'])assert.equal(saved[field],original[field],'saved identity must not be rewritten');
  // Export only the real family's launch inputs. Keys and task-local memory paths are not evidence.
  result[key]=Object.fromEntries(['arch','exe','model_name','args','parallel','cache_type_k','cache_type_v','q36_run']
   .filter(key=>key in run).map(key=>[key,run[key]]));
 }finally{await f.cleanup();}
}
if(process.argv.includes('--memory')){
 let f;f=nativeFixture({packaged:true,q36CapabilityProbe:async()=>({kvCapability:null,memoryCapability:{
  schema:1,family:'qwen35moe',modes:['bounded-experts-v1'],flag:'--q36-memory',requiresExpertStore:'mapped-pack',
  requiresExplicitSlots:true,scope:'expert-working-storage',totalProcessBound:false,qualification:'experimental-unqualified',binarySha256:q36FileSha256(join(f.repo,'strata.exe'))}})});
 try{
  f.catalog.families[0].family='qwen36';writeFileSync(join(f.repo,'strata.exe'),'inert cross-language memory fixture');
  writeFileSync(join(f.repo,'strata-iq2_xs.json'),JSON.stringify({parallel:1,args:['--max-context','256','--q36-expert-store','mapped-pack','--expert-slots','5']}));
  await f.control.refresh(true);await f.control.saveChecked('iq2_xs',{values:{q36_memory:'bounded-experts-v1'}});
  const done=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs'}).id);assert.equal(done.state,'done',done.error);
  const run=JSON.parse(readFileSync(f.launchConfigs.at(-1)));
  result.bounded=Object.fromEntries(['arch','args','q36_memory','q36_run'].map(key=>[key,run[key]]));
 }finally{await f.cleanup();}
}
console.log(JSON.stringify(result));
