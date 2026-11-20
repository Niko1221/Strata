import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,readFileSync,renameSync,rmSync} from 'node:fs';
import {join} from 'node:path';
import {tmpdir} from 'node:os';

test('durable store update rejects a real filesystem failure without publishing memory, and restart retains the prior state',async t=>{
 const home=mkdtempSync(join(tmpdir(),'strata-state-durable-'));
 const before={STRATA_HOME:process.env.STRATA_HOME,STRATA_MODELS_DIR:process.env.STRATA_MODELS_DIR};
 Object.assign(process.env,{STRATA_HOME:home,STRATA_MODELS_DIR:join(home,'models')});
 t.after(()=>{for(const [key,value] of Object.entries(before))if(value===undefined)delete process.env[key];else process.env[key]=value;rmSync(home,{recursive:true,force:true});});
 const store=await import(`../server/store.mjs?durability=${Date.now()}`);
 store.update({agentTools:{search:{provider:'builtin',key:''}}},{durable:true});
 const file=join(home,'data','state.json'),saved=join(home,'data','state.previous');
 const original=readFileSync(file,'utf8'),memory=structuredClone(store.get());
 // Windows and POSIX both refuse a file write/replace onto this directory.
 renameSync(file,saved);mkdirSync(file);
 try{
  assert.throws(()=>store.update({agentTools:{search:{provider:'exa',key:'fixture-secret'}}},{durable:true}));
  assert.deepEqual(store.get(),memory);
  assert.equal(readFileSync(saved,'utf8'),original);
 }finally{rmSync(file,{recursive:true,force:true});renameSync(saved,file);}
 const restarted=await import(`../server/store.mjs?restart=${Date.now()}`);
 assert.deepEqual(restarted.get().agentTools,memory.agentTools);
 store.update({agentTools:{search:{provider:'exa',key:'fixture-secret'}}},{durable:true});
 const committed=await import(`../server/store.mjs?committed=${Date.now()}`);
 assert.equal(committed.get().agentTools.search.key,'fixture-secret');
});

test('legacy store updates remain best-effort when the filesystem refuses a save',async t=>{
 const home=mkdtempSync(join(tmpdir(),'strata-state-legacy-'));
 const previous=process.env.STRATA_HOME;process.env.STRATA_HOME=home;
 t.after(()=>{if(previous===undefined)delete process.env.STRATA_HOME;else process.env.STRATA_HOME=previous;rmSync(home,{recursive:true,force:true});});
 const store=await import(`../server/store.mjs?legacy=${Date.now()}`);
 mkdirSync(join(home,'data','state.json'));
 assert.doesNotThrow(()=>store.update({legacyFixture:true}));assert.equal(store.get().legacyFixture,true);
});
