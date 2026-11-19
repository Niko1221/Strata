import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,writeFileSync,readFileSync,rmSync,cpSync,symlinkSync,renameSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {findLocalBackend,verifyLocalBackend,LOCAL_BACKENDS} from '../server/local-backends.mjs';

test('the Windows app retains official private runtimes and excludes GPU packages',()=>{
 const config=JSON.parse(readFileSync(new URL('../../desktop/src-tauri/tauri.conf.json',import.meta.url),'utf8'));
 const resources=config.bundle.resources;
 for(const name of ['node','control','harness','notices'])assert.equal(resources['resources/'+name],name);
 for(const name of ['engine-bundle','local-backends'])assert.ok(!resources['resources/'+name],name+' is distributed as a component');
 assert.equal(config.bundle.windows.nsis.compression,'lzma');
});

test('installed engine components supply explicit backends and honor a component rollback',async t=>{
 const root=mkdtempSync(join(tmpdir(),'hakua-component-backends-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
 const options={componentsDir:root,installRoot:join(root,'absent'),platform:'win32'};
 const exe='llama-server.exe';
 const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
 const folders=[];
 for(const version of ['0.4.0','0.5.0']){
  const folder=version+'-windows-x86_64-cuda-cuda13',component=join(root,'engine',folder);folders.push(folder);
  const files=[];
  for(const [choice,definition] of Object.entries(LOCAL_BACKENDS)){
   const prefix='local-backends/'+definition.id,dir=join(component,prefix);mkdirSync(dir,{recursive:true});
   const payload=choice+' '+version;writeFileSync(join(dir,exe),payload);
    const manifest={schema:1,version:'0.4.0',backend:definition.id,source:{repository:definition.repository,commit:definition.versions['0.4.0'].commit,dirty:false,patches:[]},cacheTypes:['f16'],files:[{path:exe,sha256:hash(payload)}]};
   const bytes=JSON.stringify(manifest);writeFileSync(join(dir,'runtime.json'),bytes);
   files.push({path:prefix+'/runtime.json',sha256:hash(bytes)},{path:prefix+'/'+exe,sha256:hash(payload)});
  }
  writeFileSync(join(component,'engine-component.json'),JSON.stringify({id:'strata-engine',version,platform:'windows-x86_64',backend:'cuda',variant:'cuda13',files}));
 }
 const staged=join(root,'engine','0.6.0-windows-x86_64-cuda-cuda13.incoming');
 cpSync(join(root,'engine',folders[1]),staged,{recursive:true});
 const stagedManifest=JSON.parse(readFileSync(join(staged,'engine-component.json'),'utf8'));stagedManifest.version='0.6.0';
 writeFileSync(join(staged,'engine-component.json'),JSON.stringify(stagedManifest));
 const outside=join(root,'not-installed');cpSync(staged,outside,{recursive:true});
 symlinkSync(outside,join(root,'engine','0.6.0-windows-x86_64-cuda-cuda13'),process.platform==='win32'?'junction':'dir');
 for(const choice of Object.keys(LOCAL_BACKENDS)){
  const selected=findLocalBackend(choice,options);assert.ok(selected,'The official engine component must supply '+choice);
  assert.equal(selected.path,join(root,'engine',folders[1],'local-backends',LOCAL_BACKENDS[choice].id,exe),'linked and staging payloads are excluded');
   assert.equal(selected.version,'0.4.0');await verifyLocalBackend(selected,choice);
 }
 // Control uses a manifest-digest suffix when a same-version package is replaced.
 const previous=join(root,'engine',folders[1]);
 const digest=hash(readFileSync(join(previous,'engine-component.json'))).slice(0,8);
 folders[1]+='-'+digest;renameSync(previous,join(root,'engine',folders[1]));
 cpSync(join(root,'engine',folders[1]),join(root,'engine',folders[1].slice(0,-8)+(digest==='00000000'?'11111111':'00000000')),{recursive:true});
 for(const choice of Object.keys(LOCAL_BACKENDS))assert.equal(findLocalBackend(choice,options).path,join(root,'engine',folders[1],'local-backends',LOCAL_BACKENDS[choice].id,exe));
 // This is the real Control engine_component.UPDATE_STATE contract.
 mkdirSync(join(root,'.updates'),{recursive:true});
 writeFileSync(join(root,'.updates','state.json'),JSON.stringify({engine:{pinned:folders[0]}}));
 const selected=findLocalBackend('PrismML',options);assert.equal(selected.version,'0.4.0');
 const path=selected.manifestPath;const old=readFileSync(path,'utf8');writeFileSync(path,old.replace('0.4.0','0.4.1'));
 await assert.rejects(verifyLocalBackend({...selected,version:'0.4.1'},'PrismML'),/component.*manifest changed/i);
});
