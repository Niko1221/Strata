import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,writeFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {execFileSync} from 'node:child_process';
import {verifyTree,verifyHost,verifyEngine,verifyNativeBuild,hostSources,nativeSources,sha256,verifyPublicAssets,PUBLIC_ASSET_COMMIT} from '../scripts/release-integrity.mjs';

test('release freshness rejects stale, missing and leftover Python sources', t=>{
 const root=mkdtempSync(join(tmpdir(),'hakua-release-'));
 t.after(()=>rmSync(root,{recursive:true,force:true}));
 const source=join(root,'source'),staged=join(root,'staged');
 mkdirSync(source);mkdirSync(staged);
 writeFileSync(join(source,'runtime.py'),'current');
 writeFileSync(join(staged,'runtime.py'),'old');
 assert.throws(()=>verifyTree(source,staged,{pythonOnly:true}),/Stale staged source/);
 writeFileSync(join(staged,'runtime.py'),'current');
 assert.doesNotThrow(()=>verifyTree(source,staged,{pythonOnly:true}));
 writeFileSync(join(source,'memory.py'),'needed');
 assert.throws(()=>verifyTree(source,staged,{pythonOnly:true}),/membership/);
 writeFileSync(join(staged,'memory.py'),'needed');
 writeFileSync(join(staged,'removed.py'),'old');
 assert.throws(()=>verifyTree(source,staged,{pythonOnly:true}),/membership/);
});

test('GUI/host source and generated asset changes stop packaging',t=>{
 const root=mkdtempSync(join(tmpdir(),'hakua-gui-'));
 t.after(()=>rmSync(root,{recursive:true,force:true}));
 for(const name of ['src','public','server','dist'])mkdirSync(join(root,name));
 const source=join(root,'src','app.tsx'),asset=join(root,'dist','index.html');
 writeFileSync(source,'current');writeFileSync(asset,'built');
 writeFileSync(join(root,'dist','release-build.json'),JSON.stringify({sources:hostSources(root),assets:{'index.html':sha256(asset)}}));
 assert.doesNotThrow(()=>verifyHost(root));
 writeFileSync(source,'changed');assert.throws(()=>verifyHost(root),/Stale GUI/);
 writeFileSync(source,'current');writeFileSync(asset,'tampered');assert.throws(()=>verifyHost(root),/Changed GUI/);
});

test('public asset provenance rejects replaced artwork, missing notices and supplied fonts',t=>{
 const root=mkdtempSync(join(tmpdir(),'hakua-public-assets-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
 mkdirSync(join(root,'icons'));mkdirSync(join(root,'licenses'));
 const icon=join(root,'icons','sprite.svg'),notice=join(root,'licenses','Lucide-LICENSE.txt');
 writeFileSync(icon,'approved icons');writeFileSync(notice,'copyright and license');
 const record={lucide_commit:PUBLIC_ASSET_COMMIT,files:{'icons/sprite.svg':sha256(icon),'licenses/Lucide-LICENSE.txt':sha256(notice)}};
 assert.doesNotThrow(()=>verifyPublicAssets(root,record));
 writeFileSync(icon,'different icons');assert.throws(()=>verifyPublicAssets(root,record),/payload changed/);writeFileSync(icon,'approved icons');
 rmSync(notice);assert.throws(()=>verifyPublicAssets(root,record),/membership/);writeFileSync(notice,'copyright and license');
 const font=join(root,'GTWalsheimPro.woff2');writeFileSync(font,'unqualified font');record.files['GTWalsheimPro.woff2']=sha256(font);
 assert.throws(()=>verifyPublicAssets(root,record),/Unapproved/);
});

test('engine payload hash, version and entrypoint are required',t=>{
 const root=mkdtempSync(join(tmpdir(),'hakua-engine-'));
 t.after(()=>rmSync(root,{recursive:true,force:true}));
 const exe=join(root,'strata.exe');writeFileSync(exe,'same-length');
 const manifest={id:'strata-engine',version:'0.1.54',entrypoint:'strata.exe',files:[{path:'strata.exe',bytes:11,sha256:sha256(exe)}]};
 const save=()=>writeFileSync(join(root,'engine-component.json'),JSON.stringify(manifest));save();
 assert.doesNotThrow(()=>verifyEngine(root,'0.1.54'));
 writeFileSync(exe,'wrong-bytes');assert.throws(()=>verifyEngine(root,'0.1.54'),/Changed engine payload/);
 writeFileSync(exe,'same-length');assert.throws(()=>verifyEngine(root,'0.1.55'),/Stale/);
 writeFileSync(join(root,'leftover.dll'),'old');assert.throws(()=>verifyEngine(root,'0.1.54'),/membership/);rmSync(join(root,'leftover.dll'));
 manifest.entrypoint='missing.exe';save();assert.throws(()=>verifyEngine(root,'0.1.54'),/entrypoint/);
 manifest.entrypoint='strata.exe';manifest.files[0].path='../outside';save();assert.throws(()=>verifyEngine(root,'0.1.54'),/entrypoint|Unsafe/);
});

test('independent native source is explicit, frozen and still checks source, baseline and binary',t=>{
 const root=mkdtempSync(join(tmpdir(),'hakua-native-binding-'));
 t.after(()=>rmSync(root,{recursive:true,force:true}));
 const app=join(root,'app'),source=join(root,'native'),folder=join(root,'engine');
 for(const path of [app,source,folder])mkdirSync(path);
 for(const path of [app,source]){mkdirSync(join(path,'src'));writeFileSync(join(path,'CMakeLists.txt'),'project(strata VERSION 0.5.1)');}
 writeFileSync(join(app,'src/engine.cpp'),'new integrated work');
 writeFileSync(join(source,'src/engine.cpp'),'qualified source');
 writeFileSync(join(source,'baseline.json'),'qualified resources');
 const git=(...args)=>execFileSync('git',['-C',source,...args],{encoding:'utf8',windowsHide:true});
 git('init','--quiet');git('add','.');
 git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','--quiet','-m','native fixture');
 const commit=git('rev-parse','HEAD').trim();
 writeFileSync(join(folder,'strata.exe'),'qualified binary');
 const record={version:'0.5.1',entrypoint:'strata.exe',sha256:sha256(join(folder,'strata.exe')),sources:nativeSources(source),recipe:{cudaSpillBaseline:{path:'baseline.json',sha256:sha256(join(source,'baseline.json'))}}};
 writeFileSync(join(folder,'native-build.json'),JSON.stringify(record));
 const manifest={id:'strata-engine',version:'0.5.1',entrypoint:'strata.exe',files:[{path:'strata.exe',bytes:16,sha256:record.sha256},{path:'native-build.json',bytes:Buffer.byteLength(JSON.stringify(record)),sha256:sha256(join(folder,'native-build.json'))}]};
 writeFileSync(join(folder,'engine-component.json'),JSON.stringify(manifest));
 const options={sourceRoot:source,sourceCommit:commit,manifestSHA256:sha256(join(folder,'engine-component.json'))};
 assert.throws(()=>verifyNativeBuild(folder,app,manifest),/Native engine sources changed/);
 assert.doesNotThrow(()=>verifyEngine(folder,'0.5.1'));
 assert.equal(verifyNativeBuild(folder,app,manifest,options).sourceCommit,commit);
 for(const key of Object.keys(options)){
  const partial={...options};delete partial[key];
  assert.throws(()=>verifyNativeBuild(folder,app,manifest,partial),/requires an absolute root/);
 }
 assert.throws(()=>verifyNativeBuild(folder,app,manifest,{...options,sourceCommit:'0'.repeat(40)}),/commit changed/);
 assert.throws(()=>verifyNativeBuild(folder,app,manifest,{...options,manifestSHA256:'0'.repeat(64)}),/manifest changed/);
 writeFileSync(join(source,'src/engine.cpp'),'changed source');
 assert.throws(()=>verifyNativeBuild(folder,app,manifest,options),/Native engine sources changed/);
 writeFileSync(join(source,'src/engine.cpp'),'qualified source');
 writeFileSync(join(source,'baseline.json'),'changed resources');
 assert.throws(()=>verifyNativeBuild(folder,app,manifest,options),/CUDA resource baseline changed/);
 writeFileSync(join(source,'baseline.json'),'qualified resources');
 writeFileSync(join(folder,'strata.exe'),'changed binary');
 assert.throws(()=>verifyNativeBuild(folder,app,manifest,options),/Stale native build identity/);
 assert.throws(()=>verifyEngine(folder,'0.5.1'),/Changed engine payload/);
});
