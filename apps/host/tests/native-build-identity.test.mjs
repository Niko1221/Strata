import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,writeFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {nativeSources,sha256,verifyNativeBuild,ggmlIdentity} from '../scripts/release-integrity.mjs';

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
