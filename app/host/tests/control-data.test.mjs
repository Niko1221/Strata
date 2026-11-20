import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,cpSync,writeFileSync,rmSync,readFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {verifyControlData} from '../scripts/release-integrity.mjs';

const repo=resolve(import.meta.dirname,'../../..'),data='data/experimental-speed-projection';
function fixture(t) {
 const root=mkdtempSync(join(tmpdir(),'control-data-'));
 t.after(()=>rmSync(root,{recursive:true,force:true}));
 const source=join(root,'source'),staged=join(root,'staged');
 for(const out of [source,staged])cpSync(join(repo,data),join(out,data),{recursive:true});
 return {source,staged};
}
test('base runtime includes only the unchanged, licensed default-off control vector',t=>{
 const {source,staged}=fixture(t);
 const result=verifyControlData(source,staged);
 assert.equal(result.vector.default,'off');assert.equal(result.vector.kind,'controlvector');
});
for(const mode of ['missing','changed'])test(mode+' control-vector licence stops packaging',t=>{
 const {source,staged}=fixture(t),license=join(staged,data,'LICENSE');
 if(mode==='missing')rmSync(license);else writeFileSync(license,'not the upstream licence');
 assert.throws(()=>verifyControlData(source,staged),/ENOENT|Missing or changed/);
});
test('a second GGUF cannot be hidden beside the approved vector',t=>{
 const {source,staged}=fixture(t);
 writeFileSync(join(staged,data,'weights.gguf'),'GGUF');
 assert.throws(()=>verifyControlData(source,staged),/Unreviewed model data/);
});
for(const file of ['memory.db','.env'])test('live '+file+' cannot enter the runtime',t=>{
 const {source,staged}=fixture(t);
 writeFileSync(join(staged,file),'fixture-only');
 assert.throws(()=>verifyControlData(source,staged),/Live profile data/);
});
test('an altered source provenance cannot declare a different default',t=>{
 const {source,staged}=fixture(t),path=join(source,data,'PROVENANCE.json');
 const record=JSON.parse(readFileSync(path,'utf8'));record.vector.default='on';
 writeFileSync(path,JSON.stringify(record));
 assert.throws(()=>verifyControlData(source,staged),/Unreviewed control-vector provenance/);
});
