import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,writeFileSync,rmSync,readFileSync,readdirSync,existsSync} from 'node:fs';
import {homedir,tmpdir} from 'node:os';
import {join} from 'node:path';
import {deepseekRuntime} from '../server/deepseek-runtime.mjs';
import {nativeFixture,finishJob} from './helpers/native-fixture.mjs';
import {inspectLocalModel} from '../server/local-models.mjs';
import {ggufFixture} from './helpers/local-gguf.mjs';

test('dedicated DeepSeek availability requires its own regular executable and honors the component report',()=>{
 const root=mkdtempSync(join(tmpdir(),'strata-ds-runtime-'));
 try{
  const configPath=join(root,'strata-ds.json'),exe=join(root,'strata-dsv41.exe');
  const cfg={arch:'deepseek41',exe:'strata-dsv41.exe'};
  assert.equal(deepseekRuntime(cfg,configPath).available,false);
  mkdirSync(exe);assert.equal(deepseekRuntime(cfg,configPath).available,false);rmSync(exe,{recursive:true});
  writeFileSync(exe,'fixture only');
  assert.equal(deepseekRuntime(cfg,configPath).available,true);
  assert.equal(deepseekRuntime(cfg,configPath).executionCompatibility,'unverified');
  assert.equal(deepseekRuntime(cfg,configPath,{missing:[{part:'engine',status:'invalid'}]}).available,false);
  assert.equal(deepseekRuntime({...cfg,exe:'component:strata-engine'},configPath).available,false);
  assert.equal(deepseekRuntime({...cfg,exe:process.execPath},configPath).available,false);
  assert.equal(deepseekRuntime({...cfg,cwd:root},join(root,'elsewhere','x.json')).exe,exe);
 }finally{rmSync(root,{recursive:true,force:true});}
});

test('dense Qwen metadata takes priority over a misleading DeepSeek filename and catalog label',()=>{
 const cfg={arch:'qwen35',exe:'strata.exe',model_name:'DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf'};
 assert.equal(deepseekRuntime(cfg,join(tmpdir(),'strata-qwen.json'),{family:'deepseek41'}),null);
});

test('importing a DeepSeek-named dense 9B GGUF retains its actual Qwen architecture',()=>{
 const root=mkdtempSync(join(tmpdir(),'strata-dense-identity-'));
 try{
  const file=join(root,'DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf');
  ggufFixture(file,{arch:'qwen35',omit:['qwen35.expert_count']});
  const preview=inspectLocalModel(file);
  assert.equal(preview.variant.local.architecture,'qwen35');
  assert.equal(deepseekRuntime({arch:preview.variant.local.architecture,model_name:file},join(root,'strata-test.json'),{family:'deepseek41'}),null);
 }finally{rmSync(root,{recursive:true,force:true});}
});

for(const packaged of [false,true])test(`DeepSeek without a regular dedicated executable is listed but cannot write or start (${packaged?'packaged':'source'})`,async()=>{
 const f=nativeFixture({packaged});
 try{
  const path=join(f.repo,'strata-iq2_xs.json');
  const cfg={...JSON.parse(readFileSync(path,'utf8')),arch:'deepseek41',exe:join(f.repo,'strata-dsv41.exe'),vision:null};
  writeFileSync(path,JSON.stringify(cfg));
  f.catalog.families[0].family='deepseek41';
  const before=readFileSync(path,'utf8');
  await f.control.refresh(true);
  const model=f.control.state().installed.models.find(m=>m.model==='iq2_xs');
  assert.equal(model.runtime.available,false);
  assert.equal(f.control.state().catalog.families[0].family,'deepseek41');
  for(const directory of [false,true]){
   if(directory)mkdirSync(cfg.exe);
   const calls=f.calls.length;
   assert.throws(()=>f.control.operation('load',{model:'iq2_xs'}),/dedicated DeepSeek engine/);
   assert.equal(f.control.state().jobs.length,0,'reject before the refresh or an operation is scheduled');
   assert.equal(readFileSync(path,'utf8'),before);
   assert.deepEqual(f.effects,[]);
   assert.equal(f.calls.slice(calls).some(c=>c.args.some(a=>String(a).endsWith('server.py'))),false);
   const work=join(f.directory,'native-instances');assert.deepEqual(existsSync(work)?readdirSync(work):[],[]);
  }
 }finally{await f.cleanup();}
});

test('a checked dedicated DeepSeek file does not borrow the generic Qwen engine availability',async()=>{
 const f=nativeFixture({packaged:true,genericEngineUsable:false});
 try{
  const path=join(f.repo,'strata-iq2_xs.json'),exe=join(f.repo,'strata-dsv41.exe');
  writeFileSync(exe,'inert dedicated-engine fixture; Control start is mocked');
  writeFileSync(path,JSON.stringify({arch:'deepseek41',exe,args:['--max-context','256']}));
  await f.control.refresh(true);
  assert.equal(f.control.state().installed.models[0].runtime.available,true);
  const job=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs'}).id);
  assert.equal(job.state,'done',job.error);
  assert.equal(JSON.parse(readFileSync(f.launchConfigs[0],'utf8')).exe,exe);
  assert.equal(f.packagedStarts.length,1);
 }finally{await f.cleanup();}
});

for(const spelling of ['relative without cwd','home-relative'])test(`DeepSeek load launches the checked canonical executable for ${spelling} and preserves saved settings`,async()=>{
 const f=nativeFixture({packaged:true,genericEngineUsable:false});
 const homeKey=process.platform==='win32'?'USERPROFILE':'HOME',originalHome=process.env[homeKey];
 try{
  const path=join(f.repo,'strata-iq2_xs.json'),exe=join(f.repo,'engine','strata-dsv41.exe');
  if(spelling==='home-relative'){process.env[homeKey]=f.repo;assert.equal(homedir(),f.repo);}
  writeFileSync(exe,'inert dedicated-engine fixture; Control start is mocked');
  const saved=JSON.stringify({arch:'deepseek41',exe:spelling==='home-relative'?'~/engine/strata-dsv41.exe':'engine/strata-dsv41.exe',args:['--max-context','256']});
  writeFileSync(path,saved);
  await f.control.refresh(true);
  const runtime=f.control.state().installed.models[0].runtime;
  assert.equal(runtime.available,true);assert.equal(runtime.exe,exe);
  const job=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs'}).id);
  assert.equal(job.state,'done',job.error);
  assert.equal(f.packagedStarts.length,1);
  assert.equal(JSON.parse(readFileSync(f.launchConfigs[0],'utf8')).exe,runtime.exe);
  assert.equal(readFileSync(path,'utf8'),saved,'the saved executable spelling and settings stay byte-identical');
 }finally{
  if(originalHome===undefined)delete process.env[homeKey];else process.env[homeKey]=originalHome;
  await f.cleanup();
 }
});

test('a dedicated executable removed after operation capture is rejected again before load effects',async()=>{
 const f=nativeFixture({packaged:true});
 try{
  const path=join(f.repo,'strata-iq2_xs.json'),exe=join(f.repo,'strata-dsv41.exe');
  writeFileSync(exe,'inert dedicated-engine fixture');
  const saved=JSON.stringify({arch:'deepseek41',exe,args:['--max-context','256']});writeFileSync(path,saved);
  await f.control.refresh(true);
  const job=f.control.operation('load',{model:'iq2_xs'});rmSync(exe);
  const done=await finishJob(f.control,job.id);
  assert.equal(done.state,'error');assert.match(done.error,/dedicated DeepSeek engine/);
  assert.deepEqual(f.effects,[]);assert.equal(readFileSync(path,'utf8'),saved);
 }finally{await f.cleanup();}
});
