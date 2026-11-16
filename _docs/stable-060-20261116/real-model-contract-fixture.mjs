/** Pure producer contracts. No app, GPU, external network or inference. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {zstdCompressSync} from 'node:zlib';
import fs from 'node:fs';
import {syncBuiltinESMExports} from 'node:module';
import {GROUPS,checkAudit,checkSettings,checkHeaders,safeFrame,requestHeaders} from './real-model.mjs';

test('exact independent audit classifications reject plausible wrong results',()=>{
  const good={criteria:'explicit criteria',version_mismatches:[{id:'alpha',inventory_version:'1',expected_version:'2'}],duplicates:[],missing_ids:[{id:'gamma',missingFrom:['inventory']}],failed_checks:[{id:'beta',required_check:'checksum',status:'failed'}],total_mismatches:1,total_duplicates:0,total_missing:1,total_failed:1};
  checkAudit(good);
  assert.throws(()=>checkAudit({...good,missing_ids:[]}));
  assert.throws(()=>checkAudit({...good,total_failed:0}));
  assert.throws(()=>checkAudit({...good,version_mismatches:[{id:'beta',inventory_version:'1',expected_version:'2'}]}));
});
test('capability schema and configured off groups reject broad tools',()=>{
  const settings={groups:GROUPS.map(id=>({id,enabled:id==='skills'})),custom:[{id:'owned',enabled:true,kind:'server'}],search:{provider:'rustxng'},browser:{enabled:false},webWithoutAsking:false};
  checkSettings(settings,'owned');
  assert.throws(()=>checkSettings({...settings,groups:settings.groups.map(g=>g.id==='files'?{...g,enabled:true}:g)},'owned'));
  assert.throws(()=>checkHeaders([{provider:'strata-local',model:'local',tools:[{name:'run_code'}]}],['audit_inventory'],'strata-local'));
  assert.throws(()=>checkHeaders([{provider:'remote',model:'local',tools:[]}],[],'strata-local'));
});
test('redaction excludes access keys without printing fixture secrets',()=>{
  const result=safeFrame({profileKey:'synthetic',nested:{authorization:'synthetic',env:{TOKEN:'synthetic'},type:'done'}});
  assert.ok(!JSON.stringify(result).includes('synthetic'));
  assert.deepEqual(result,{nested:{type:'done'}});
});
test('durable compressed request headers retain exact schema and local route',()=>{
  const root=mkdtempSync(join(tmpdir(),'real-model-contract-')),session=join(root,'session-synthetic');mkdirSync(session);
  const event={type:'request/header',seq:2,data:{header:{config:{provider:'strata-local',model:'local',maxTokens:1536},tools:[]}}};
  writeFileSync(join(session,'session.v4.jsonl.zstd'),zstdCompressSync(Buffer.from(JSON.stringify(event)+'\n')));
  const headers=requestHeaders(root);assert.equal(headers.length,1);assert.equal(headers[0].sessionId,'session-synthetic');assert.equal(headers[0].maxTokens,1536);checkHeaders(headers,[],'strata-local');
  const atomic='.12345678-1234-1234-1234-123456789abc.tmp';
  function duringListing(folder,change,read) {
    const original=fs.readdirSync;let calls=0;
    fs.readdirSync=function(path,options){
      if(String(path)!==folder)return original(path,options);
      calls++;if(change.before)change.before(calls);
      const rows=original(path,options).sort((a,b)=>Number(a.name.endsWith('.tmp'))-Number(b.name.endsWith('.tmp')));
      change.after?.(calls,rows);return rows;
    };
    syncBuiltinESMExports();
    try {return read(()=>calls);} finally {fs.readdirSync=original;syncBuiltinESMExports();}
  }
  const race=mkdtempSync(join(tmpdir(),'real-model-atomic-'));
  writeFileSync(join(race,'first.jsonl'),JSON.stringify(event)+'\n');
  writeFileSync(join(race,atomic),JSON.stringify({...event,seq:3})+'\n');
  duringListing(race,{after(calls){if(calls===1)fs.renameSync(join(race,atomic),join(race,'second.jsonl'));}},calls=>{
    const snapshot=requestHeaders(race);
    assert.equal(calls(),2);assert.deepEqual(snapshot.map(h=>h.seq).sort(),[2,3]);
    checkHeaders(snapshot,[],'strata-local');
  });
  for(const failure of ['canonical','other-temp','not-regular','exhausted','denied']) {
    const target=mkdtempSync(join(tmpdir(),'real-model-atomic-negative-'));
    const name=failure==='canonical'?'session.jsonl':failure==='other-temp'?'.other.tmp':atomic;
    duringListing(target,{
      before(){if(failure==='denied')throw Object.assign(new Error('Synthetic denied enumeration'),{code:'EACCES'});writeFileSync(join(target,name),JSON.stringify(event)+'\n');},
      after(_calls,rows){fs.unlinkSync(join(target,name));if(failure==='not-regular')rows[0]={name,isFile:()=>false,isDirectory:()=>false};},
    },calls=>{
      assert.throws(()=>requestHeaders(target),error=>error.code===(failure==='denied'?'EACCES':'ENOENT'));
      assert.equal(calls(),failure==='exhausted'?3:1);
    });
  }
  const malformed=mkdtempSync(join(tmpdir(),'real-model-malformed-'));
  writeFileSync(join(malformed,'session.jsonl'),'not JSON\n');
  assert.throws(()=>requestHeaders(malformed),SyntaxError);
  assert.throws(()=>checkHeaders([],[],'strata-local'));
  const linked=mkdtempSync(join(tmpdir(),'real-model-reparse-')),owned=join(linked,'owned');mkdirSync(owned);
  const junction=join(linked,'junction');fs.symlinkSync(owned,junction,'junction');
  try {assert.throws(()=>requestHeaders(linked));} finally {fs.unlinkSync(junction);}
});
