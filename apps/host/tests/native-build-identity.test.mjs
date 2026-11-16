import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,writeFileSync,rmSync,statSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {nativeSources,q38Sources,sha256,verifyNativeBuild,verifyQ38Build,verifyEngine,ggmlIdentity} from '../scripts/release-integrity.mjs';

test('packaging rejects an engine from previous native source or a substituted binary',t=>{
 const root=mkdtempSync(join(tmpdir(),'hakua-native-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
 const stage=join(root,'stage');mkdirSync(stage);mkdirSync(join(root,'src'));
 const source=join(root,'src','engine.cpp'),exe=join(stage,'strata.exe');
 writeFileSync(source,'current code');writeFileSync(exe,'fresh binary');
 const manifest={version:'0.3.3',entrypoint:'strata.exe'};
 writeFileSync(join(stage,'engine-component.json'),JSON.stringify(manifest));
 writeFileSync(join(stage,'native-build.json'),JSON.stringify({...manifest,sha256:sha256(exe),sources:nativeSources(root)}));
 assert.doesNotThrow(()=>verifyNativeBuild(stage,root,manifest));
 writeFileSync(source,'newer code');assert.throws(()=>verifyNativeBuild(stage,root,manifest),/sources changed/);
 writeFileSync(source,'current code');writeFileSync(exe,'older binary');assert.throws(()=>verifyNativeBuild(stage,root,manifest),/identity/);
});

test('packaging binds the reviewed CUDA budget and rejects changed or escaping budgets',t=>{
 const root=mkdtempSync(join(tmpdir(),'hakua-native-budget-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
 const stage=join(root,'stage');mkdirSync(stage);mkdirSync(join(root,'src'));
 const exe=join(stage,'strata.exe'),budget=join(root,'budget.json');
 writeFileSync(exe,'fresh binary');writeFileSync(budget,'reviewed measurements');
 const manifest={version:'0.3.3',entrypoint:'strata.exe'};
 writeFileSync(join(stage,'engine-component.json'),JSON.stringify(manifest));
 const record={...manifest,sha256:sha256(exe),sources:nativeSources(root),recipe:{cudaSpillBaseline:{path:'budget.json',sha256:sha256(budget)}}};
 const save=()=>writeFileSync(join(stage,'native-build.json'),JSON.stringify(record));
 save();assert.doesNotThrow(()=>verifyNativeBuild(stage,root,manifest));
 writeFileSync(budget,'different allowances');assert.throws(()=>verifyNativeBuild(stage,root,manifest),/baseline changed/);
 record.recipe.cudaSpillBaseline.path='../budget.json';save();assert.throws(()=>verifyNativeBuild(stage,root,manifest),/Unsafe/);
 record.recipe.cudaSpillBaseline.path='..\\budget.json';save();assert.throws(()=>verifyNativeBuild(stage,root,manifest),/Unsafe/);
});

test('vendored ggml binds actual dependency bytes without borrowing the parent Git identity',t=>{
 const root=mkdtempSync(join(tmpdir(),'hakua-vendored-ggml-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
 const vendor=join(root,'third_party','strata-ggml'),pins=join(root,'third_party','ggml');
 mkdirSync(join(vendor,'ggml'),{recursive:true});mkdirSync(pins,{recursive:true});
 writeFileSync(join(pins,'VERSION.txt'),'a'.repeat(40)+' upstream pin\n');
 writeFileSync(join(vendor,'UPSTREAM.md'),'fixture patch history\n');
 const kernel=join(vendor,'ggml','kernel.cpp');writeFileSync(kernel,'original kernel\n');
 const before=ggmlIdentity(root,'');
 assert.equal(before.mode,'vendored');assert.equal(before.commit,'a'.repeat(40));
 assert.match(before.sha256,/^[a-f0-9]{64}$/);
 assert.deepEqual(ggmlIdentity(root,vendor),before);
 const sourceBefore=nativeSources(root);
 assert.equal(sourceBefore['third_party/strata-ggml/ggml/kernel.cpp'],sha256(kernel));
 assert.equal(sourceBefore['third_party/ggml/VERSION.txt'],sha256(join(pins,'VERSION.txt')));
 const stage=join(root,'stage');mkdirSync(stage);writeFileSync(join(stage,'strata.exe'),'fixture binary');
 const manifest={version:'0.6.0',entrypoint:'strata.exe'};
 writeFileSync(join(stage,'engine-component.json'),JSON.stringify(manifest));
 const record={...manifest,sha256:sha256(join(stage,'strata.exe')),sources:sourceBefore,recipe:{ggml:before,ggmlCommit:before.commit}};
 const save=()=>writeFileSync(join(stage,'native-build.json'),JSON.stringify(record));
 save();assert.doesNotThrow(()=>verifyNativeBuild(stage,root,manifest));
 record.recipe.ggml={...before,sha256:'0'.repeat(64)};save();
 assert.throws(()=>verifyNativeBuild(stage,root,manifest),/ggml identity changed/);
 record.recipe.ggml=before;save();
 writeFileSync(kernel,'changed kernel\n');
 assert.throws(()=>verifyNativeBuild(stage,root,manifest),/sources changed/);
 assert.notEqual(ggmlIdentity(root).sha256,before.sha256);
 assert.notDeepEqual(nativeSources(root),sourceBefore);
 writeFileSync(join(pins,'VERSION.txt'),'not an identified upstream revision');
 assert.throws(()=>ggmlIdentity(root),/pin is missing or invalid/);
});

test('Q38 requires covered helper, portable provenance, embedded profile, notices and DLLs',t=>{
 const root=mkdtempSync(join(tmpdir(),'hakua-q38-build-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
 for(const name of ['include','src/qwen38','spin8/src','spin8/include','spin8/third_party','tqcodec/src','tqcodec/include','data/spin8','src/kernels/cuda'])mkdirSync(join(root,name),{recursive:true});
 const source=join(root,'src/qwen38/engine.cu'),profile=join(root,'data/spin8/reference.profile');
 writeFileSync(source,'owned CUDA');writeFileSync(profile,'fixed profile');writeFileSync(join(root,'src/kernels/cuda/fused_gdn.cu'),'native kernel');
 writeFileSync(join(root,'src/qwen38/LICENSE-formats.txt'),'format licence');writeFileSync(join(root,'tqcodec/DIALECT_LICENSE.txt'),'dialect licence');
 const stage=join(root,'stage');mkdirSync(stage);
 for(const [name,text] of Object.entries({'strata.exe':'legacy','strata-q38.exe':'q38','required.dll':'runtime','LICENSE-q38-formats.txt':'format licence','LICENSE-turbo-dialects.txt':'dialect licence'}))writeFileSync(join(stage,name),text);
 const record={schema:1,version:'0.6.0',entrypoint:'strata-q38.exe',sourceCommit:'6ee425434c98277555332c7934e1ac51436eb198',sha256:sha256(join(stage,'strata-q38.exe')),sources:q38Sources(root),releaseQualified:false,
 recipe:{cudaArchs:'120',cudaRuntime:'static',runtimeImplementation:'Strata-owned',embeddedProfileSHA256:sha256(profile),compilerManifestSHA256:'40851d465307d73a8c3844222112be5bfab87ca23c070923539c3bab53be4fcb'},
 imports:['required.dll'],requiredLibraries:['required.dll'],runtimeLibraries:{'required.dll':sha256(join(stage,'required.dll'))},notices:Object.fromEntries(['LICENSE-q38-formats.txt','LICENSE-turbo-dialects.txt'].map(name=>[name,sha256(join(stage,name))]))};
 const names=['strata.exe','strata-q38.exe','required.dll','LICENSE-q38-formats.txt','LICENSE-turbo-dialects.txt','q38-build.json'];
 const manifest={id:'strata-engine',version:'0.6.0',entrypoint:'strata.exe',files:[]};
 const save=()=>{writeFileSync(join(stage,'q38-build.json'),JSON.stringify(record));manifest.files=names.map(path=>({path,bytes:statSync(join(stage,path)).size,sha256:sha256(join(stage,path))}));writeFileSync(join(stage,'engine-component.json'),JSON.stringify(manifest));};
 save();assert.doesNotThrow(()=>verifyQ38Build(stage,root,verifyEngine(stage,'0.6.0')));
 for(const member of ['strata-q38.exe','q38-build.json','required.dll','LICENSE-q38-formats.txt']){
  manifest.files=manifest.files.filter(entry=>entry.path!==member);assert.throws(()=>verifyQ38Build(stage,root,manifest),/Uncovered|Missing/);save();
 }
 record.version='0.5.9';save();assert.throws(()=>verifyQ38Build(stage,root,manifest),/identity/);record.version='0.6.0';save();
 record.sourceCommit='b'.repeat(40);save();assert.throws(()=>verifyQ38Build(stage,root,manifest),/identity/);record.sourceCommit='6ee425434c98277555332c7934e1ac51436eb198';save();
 writeFileSync(source,'changed source');assert.throws(()=>verifyQ38Build(stage,root,manifest),/source bytes/);writeFileSync(source,'owned CUDA');
 writeFileSync(profile,'different profile');assert.throws(()=>verifyQ38Build(stage,root,manifest),/source bytes/);writeFileSync(profile,'fixed profile');
 writeFileSync(join(stage,'required.dll'),'substituted runtime');assert.throws(()=>verifyEngine(stage,'0.6.0'),/Changed engine payload/);writeFileSync(join(stage,'required.dll'),'runtime');
 writeFileSync(join(stage,'strata-q38.exe'),'substituted helper');assert.throws(()=>verifyQ38Build(stage,root,manifest),/identity/);writeFileSync(join(stage,'strata-q38.exe'),'q38');
 rmSync(join(stage,'q38-build.json'));assert.throws(()=>verifyQ38Build(stage,root,manifest));
});
