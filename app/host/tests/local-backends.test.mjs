import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,writeFileSync,readFileSync,rmSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {LOCAL_BACKENDS,localBackendChoice,findLocalBackend,verifyLocalBackend,verifyLocalBackendSource,requireRuntimeKv} from '../server/local-backends.mjs';
import {findBinary,restart,status,buildArgs} from '../server/supervisor.mjs';
import {resolveKvSettings,requireModelKvCompatible} from '../server/kv-settings.mjs';
const setting=(id,value)=>({id,value});
// Reproduce the approved artifact's CRLF/no-final-newline serialization regardless
// of Git checkout settings. Production hashes raw bytes without normalization.
const prism041Text=readFileSync(new URL('./fixtures/prism-0.4.1-runtime.json',import.meta.url),'utf8').replaceAll('\r\n','\n').trimEnd().replaceAll('\n','\r\n');
const prism041=()=>JSON.parse(prism041Text);

test('Prism 0.4.1 is one exact private source/package approval; 0.4.0 remains explicit',()=>{
 assert.equal(createHash('sha256').update(prism041Text).digest('hex'),'9c729c73eb050dc466203fe149a27acd5be48763a33ea2c5d2b89f78e4ddcbaa');
 assert.equal(LOCAL_BACKENDS.PrismML.packageVersion,'0.4.1');
 assert.deepEqual(Object.keys(LOCAL_BACKENDS.PrismML.versions),['0.4.0','0.4.1']);
 const {approved,patchFiles}=verifyLocalBackendSource(prism041(),'PrismML');
 assert.equal(approved.commit,'1965755ceee6738b9a04ceaf5d30e6305d6ccd00');
 assert.equal(patchFiles.length,3);
 for(const version of ['0.3.3','0.4.2','0.4.1-local','1.0.0','99.0.0','__proto__','constructor']){
  const changed=prism041();changed.version=version;assert.throws(()=>verifyLocalBackendSource(changed,'PrismML'),/version is not approved/);
 }
});

for(const [name,alter] of [
 ['base',m=>m.source.baseCommit='0'.repeat(40)],
 ['commit',m=>m.source.commit='0'.repeat(40)],
 ['tree',m=>m.source.tree='0'.repeat(40)],
 ['repository',m=>m.source.repository='https://example.invalid/unreviewed'],
 ['dirty tree',m=>m.source.dirty=true],
 ['dirty string',m=>m.source.dirty='false'],
 ['unverified bytes',m=>m.source.byteVerification.verified=false],
 ['truthy proof',m=>m.source.byteVerification.verified=1],
 ['proof commit',m=>m.source.byteVerification.sourceCommit='0'.repeat(40)],
 ['proof tree',m=>m.source.byteVerification.sourceTree='0'.repeat(40)],
 ['tracked file count',m=>m.source.byteVerification.trackedFiles--],
 ['raw override count',m=>m.source.byteVerification.exactRawOverrides--],
 ['reported status',m=>m.source.gitStatusReportedDirty=false],
 ['first patch id',m=>m.source.patches[0].id+='-other'],
 ['first patch hash',m=>m.source.patches[0].sha256='0'.repeat(64)],
 ['first patch path',m=>m.source.patches[0].path='../patch'],
 ['second patch hash',m=>m.source.patches[1].sha256='0'.repeat(64)],
 ['missing patch',m=>m.source.patches.pop()],
 ['extra patch',m=>m.source.patches.push({...m.source.patches[0]})],
 ['non-array patches',m=>m.source.patches={}],
 ['combined diff',m=>m.source.sourcePatchSHA256='0'.repeat(64)],
 ['public qualification',m=>m.releaseQualified=true],
 ['private marker',m=>m.privateCandidate=false],
])test(`Prism 0.4.1 refuses changed ${name}`,()=>{
 const changed=prism041();alter(changed);
 assert.throws(()=>verifyLocalBackendSource(changed,'PrismML'),/source|qualification/);
});

test('Prism 0.4.1 rejects repackaged manifests and changed actual bytes',async t=>{
 const root=mkdtempSync(join(tmpdir(),'prism041-rejections-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
 const binary={path:join(root,'llama-server.exe'),manifestPath:join(root,'runtime.json')};
 for(const alter of [m=>m.files[0].sha256='0'.repeat(64),m=>m.files.pop(),m=>m.cacheTypesK.push('turbo3'),m=>m.build.jobs=2]){
  const changed=prism041();alter(changed);writeFileSync(binary.manifestPath,JSON.stringify(changed,null,2)+'\n');
  await assert.rejects(verifyLocalBackend(binary,'PrismML'),/exact approved package/);
 }
 writeFileSync(binary.manifestPath,prism041Text.replaceAll('\r\n','\n'));
 await assert.rejects(verifyLocalBackend(binary,'PrismML'),/exact approved package/);
 writeFileSync(binary.manifestPath,prism041Text);writeFileSync(join(root,'cublas64_13.dll'),'changed bytes');
 await assert.rejects(verifyLocalBackend(binary,'PrismML'),/payload changed/);
});

test('a pinned legacy component remains discoverable for explicit 0.4.0 rollback',async t=>{
 const root=mkdtempSync(join(tmpdir(),'prism-rollback-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
 const platform=process.platform==='win32'?'windows-x86_64':process.platform==='darwin'?'macos-arm64':'linux-x86_64';
 const names=['0.5.1-'+platform+'-cuda-cuda13','0.5.2-'+platform+'-cuda-cuda13'];
 const exe=process.platform==='win32'?'llama-server.exe':'llama-server';
 for(let i=0;i<names.length;i++){
  const component=join(root,'engine',names[i]),runtime=join(component,'local-backends/prism');mkdirSync(runtime,{recursive:true});
  writeFileSync(join(runtime,exe),'fixture');const text=JSON.stringify(i?prism041():{schema:1,backend:'prism',version:'0.4.0',source:{repository:LOCAL_BACKENDS.PrismML.repository,commit:'73bfe6cc485554822f80c473a3c0c4f25e80f41b',dirty:false,patches:[]},cacheTypes:['f16'],files:[{path:exe,sha256:createHash('sha256').update('fixture').digest('hex')}]});
  writeFileSync(join(runtime,'runtime.json'),text);
  writeFileSync(join(component,'engine-component.json'),JSON.stringify({id:'strata-engine',version:i?'0.5.2':'0.5.1',platform,backend:'cuda',variant:'cuda13',files:[{path:'local-backends/prism/runtime.json',sha256:createHash('sha256').update(text).digest('hex')}]}));
 }
 const options={componentsDir:root,installRoot:join(root,'absent')};
 assert.equal(findLocalBackend('PrismML',options).version,'0.4.1');
 mkdirSync(join(root,'.updates'));writeFileSync(join(root,'.updates/state.json'),JSON.stringify({engine:{pinned:names[0]}}));
 const rollback=findLocalBackend('PrismML',options);assert.equal(rollback.version,'0.4.0');
 assert.equal((await verifyLocalBackend(rollback,'PrismML')).version,'0.4.0');
});

test('explicit backends use their own payload and never pick an old generic engine',async t=>{
 const root=mkdtempSync(join(tmpdir(),'hakua-runtimes-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
 const options={componentsDir:root,installRoot:join(root,'empty'),platform:process.platform};
 const exe=process.platform==='win32'?'llama-server.exe':'llama-server';
  const generic=join(root,'generic-gguf');mkdirSync(generic);writeFileSync(join(generic,exe),'old generic');
  const automaticEnv={STRATA_LLAMA_SERVER:join(generic,exe)};
  assert.equal(findBinary({...options,env:automaticEnv,backend:'Automatic'}),null);
  assert.equal(localBackendChoice(),'Automatic');
  for(const name of ['other','__proto__','constructor'])assert.throws(()=>localBackendChoice([setting('runtime.backend',name)]));
 assert.equal(findBinary({...options,backend:'TurboQuant'}),null);
 for(const choice of Object.keys(LOCAL_BACKENDS)) {
  const definition=LOCAL_BACKENDS[choice],folder=join(root,'local-backends',definition.id);mkdirSync(folder,{recursive:true});
  const path=join(folder,exe),payload=Buffer.from(choice);writeFileSync(path,payload);
   const manifest={schema:1,version:'0.4.0',backend:definition.id,source:{repository:definition.repository,commit:definition.versions['0.4.0'].commit,dirty:false,patches:[]},cacheTypes:choice==='TurboQuant'?['f16','q8_0','turbo3']:['f16','q8_0'],files:[{path:exe,sha256:createHash('sha256').update(payload).digest('hex')}]};
  const save=()=>writeFileSync(join(folder,'runtime.json'),JSON.stringify(manifest));save();
  const selected=findLocalBackend(choice,options);assert.equal(selected.path,path);
   assert.equal(findBinary({...options,env:automaticEnv,backend:choice}).path,path,'an explicit backend does not rewrite or follow the Automatic environment');
  assert.equal((await verifyLocalBackend(selected,choice)).backend,definition.id);
  writeFileSync(join(folder,'unexpected.dll'),'stale payload');
  await assert.rejects(verifyLocalBackend(selected,choice),/Unexpected local runtime package membership/);
  rmSync(join(folder,'unexpected.dll'));
  const bundled=join(options.installRoot,'local-backends',definition.id);mkdirSync(bundled,{recursive:true});
  writeFileSync(join(bundled,exe),payload);
  writeFileSync(join(bundled,'runtime.json'),JSON.stringify(manifest));
  assert.equal(findLocalBackend(choice,options).path,join(bundled,exe),'the current bundle takes precedence over a same-version installed component');
   if(choice==='PrismML'){
    writeFileSync(join(bundled,'runtime.json'),JSON.stringify({...manifest,version:'0.4.1'}));
    assert.equal(findLocalBackend(choice,options).path,join(bundled,exe),'the approved newer version is considered before the legacy version; payload verification follows');
   }
   writeFileSync(join(bundled,'runtime.json'),JSON.stringify({...manifest,version:'99.0.0'}));
   assert.equal(findLocalBackend(choice,options).path,path,'an unapproved higher version cannot displace a known runtime');
  if(choice==='PrismML')assert.throws(()=>requireRuntimeKv(manifest,{k:'turbo3',v:'q8_0'}),/cannot use turbo3/);
  else assert.doesNotThrow(()=>requireRuntimeKv(manifest,{k:'turbo3',v:'f16'}));
  writeFileSync(path,'substituted');await assert.rejects(verifyLocalBackend(selected,choice),/payload changed/);
  writeFileSync(path,payload);manifest.source.dirty=true;save();await assert.rejects(verifyLocalBackend(selected,choice),/unreviewed/);
 }
});

test('each runtime admits only its exact reviewed standard-sm120 patch',async t=>{
 const root=mkdtempSync(join(tmpdir(),'hakua-patched-runtimes-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
 for(const [choice,definition] of Object.entries(LOCAL_BACKENDS)) {
  const folder=join(root,'local-backends',definition.id);mkdirSync(folder,{recursive:true});
  const exe=process.platform==='win32'?'llama-server.exe':'llama-server',payload=Buffer.from('fixture executable');
  const patch=readFileSync(new URL('../../../tools/'+(choice==='PrismML'?'spin8_backend_sm120.patch':'turboquant_backend_sm120.patch'),import.meta.url));
  const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
  writeFileSync(join(folder,exe),payload);writeFileSync(join(folder,'source-patch.diff'),patch);
   const manifest={schema:1,version:'0.4.0',backend:definition.id,source:{repository:definition.repository,commit:definition.versions['0.4.0'].commit,dirty:true,patches:[{id:definition.id+'-standard-sm120-v1',sha256:hash(patch)}]},cacheTypes:['f16'],files:[{path:exe,sha256:hash(payload)},{path:'source-patch.diff',sha256:hash(patch)}]};
  const save=()=>writeFileSync(join(folder,'runtime.json'),JSON.stringify(manifest));save();
  const binary=findLocalBackend(choice,{componentsDir:root,installRoot:join(root,'absent')});
  await verifyLocalBackend(binary,choice);
  manifest.source.patches[0].id+='-unknown';save();await assert.rejects(verifyLocalBackend(binary,choice),/unreviewed/);
  manifest.source.patches[0].id=definition.id+'-standard-sm120-v1';save();
  writeFileSync(join(folder,'source-patch.diff'),'unreviewed change');await assert.rejects(verifyLocalBackend(binary,choice),/payload changed/);
 }
});

test('TurboQuant settings produce real independent K/V flags and reject incompatible geometry',()=>{
 const expert=[setting('runtime.backend','TurboQuant'),setting('ctx.kType','TURBO3'),setting('ctx.vType','TURBO4')];
 assert.deepEqual(resolveKvSettings(expert),{k:'turbo3',v:'turbo4'});
 const args=buildArgs({modelPath:'fixture.gguf',config:{expert}});
 assert.equal(args[args.indexOf('-ctk')+1],'turbo3');assert.equal(args[args.indexOf('-ctv')+1],'turbo4');
 assert.throws(()=>requireModelKvCompatible({local:{keyLength:null,valueLength:128},kvModes:['INT8']},expert),/geometry/);
 for(const width of [64,128,256])assert.doesNotThrow(()=>requireModelKvCompatible({local:{keyLength:width,valueLength:width},kvModes:['INT8']},expert));
 assert.throws(()=>resolveKvSettings([...expert,setting('perf.flashAttn',false)]),/flash attention/);
});

test('the bundled TurboQuant policy rejects failed K modes while allowing qualified V modes',()=>{
 const manifest={backend:'turboquant',cacheTypes:['f16','turbo2','turbo3','turbo4'],cacheTypesK:['f16'],cacheTypesV:['f16','turbo2','turbo3','turbo4']};
 for(const v of ['turbo2','turbo3','turbo4']) {
  assert.doesNotThrow(()=>requireRuntimeKv(manifest,{k:'f16',v}));
  assert.throws(()=>requireRuntimeKv(manifest,{k:v,v:'f16'}),new RegExp(v+' K cache'));
 }
});

test('a missing selected runtime rejects before restart changes process state',async()=>{
 const previous=process.env.STRATA_COMPONENTS_DIR;
 process.env.STRATA_COMPONENTS_DIR=join(tmpdir(),'nonexistent-hakua-component-test');
 const before=status();
 try {await assert.rejects(restart({modelPath:'missing-model.gguf',config:{expert:[setting('runtime.backend','TurboQuant')]}}),/TurboQuant runtime/);assert.deepEqual(status(),before);}
 finally {if(previous===undefined)delete process.env.STRATA_COMPONENTS_DIR;else process.env.STRATA_COMPONENTS_DIR=previous;}
});
