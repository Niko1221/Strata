import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,readFile,rm,readdir} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {host} from './helpers/reference-host.mjs';
const options={preloads:['./tests/fixtures/state-persistence-fault.mjs']};
const markdown=name=>`---\nname: ${name}\ndescription: Test instructions.\n---\n\nDURABILITY_SENTINEL\n`;
async function put(h,body){const r=await fetch(h.base+'/api/agent/tools',{method:'PUT',headers:{origin:h.base,'content-type':'application/json'},body:JSON.stringify(body)});return {status:r.status,data:await r.json()};}

for(const enabled of [false,true])test(`failed ${enabled?'workspace-scoped':'disabled'} import cannot be rediscovered with broader permissions after restart`,{timeout:90000},async t=>{
 const home=await mkdtemp(join(tmpdir(),'strata-import-durable-'));let h=await host(home,options);
 t.after(async()=>{await h.stop();await rm(home,{recursive:true,force:true});});
 const scope=(await h.call('/api/workspaces')).data.workspaces[0].id;
 assert.equal((await put(h,{search:{provider:'builtin'}})).status,200);
 const file=join(home,'data','state.json'),original=await readFile(file,'utf8');
 await writeFile(join(home,'fs-fault.json'),JSON.stringify({stateWrite:true,cleanup:true}));
 const result=await h.call('/api/skills',{markdown:markdown('failed-import'),importConfirmed:true,enabled,onlyIn:enabled?scope:null,triggers:['explicit trigger']});
 const live=(await h.call('/api/skills')).data;
 await h.stop();assert.equal(await readFile(file,'utf8'),original);
 await rm(join(home,'fs-fault.json'));h=await host(home,options);
 const after=(await h.call('/api/skills')).data;
 assert.equal(result.status,400,'EACCES must not be acknowledged as import success');
 assert.ok((await readdir(join(home,'data'))).some(name=>name.startsWith('.skill-')),'the cleanup failure really left an uncommitted staging directory');
 assert.equal(live.some(s=>s.id==='my-failed-import'),false);
 assert.equal(after.some(s=>s.id==='my-failed-import'),false,'even failed cleanup leaves no discoverable instructions');
 const retry=await h.call('/api/skills',{markdown:markdown('failed-import'),importConfirmed:true,enabled,onlyIn:enabled?scope:null,triggers:['explicit trigger']});
 assert.equal(retry.status,201);await h.stop();h=await host(home,options);
 const imported=(await h.call('/api/skills')).data.find(s=>s.id===retry.data.id);
 assert.equal(imported.enabled,enabled);assert.equal(imported.onlyIn,enabled?scope:null);assert.deepEqual(imported.triggers,['explicit trigger']);
});

test('Tools PUT rejects state EACCES without a memory-only key, then retry survives restart',{timeout:90000},async t=>{
 const home=await mkdtemp(join(tmpdir(),'strata-tools-durable-'));let h=await host(home,options);
 t.after(async()=>{await h.stop();await rm(home,{recursive:true,force:true});});
 assert.equal((await put(h,{search:{provider:'exa'}})).status,200);
 const file=join(home,'data','state.json'),original=await readFile(file,'utf8');
 await writeFile(join(home,'fs-fault.json'),JSON.stringify({stateWrite:true}));
 const failed=await put(h,{search:{provider:'exa',key:'fixture-secret'}}),live=(await h.call('/api/agent/tools')).data;
 await h.stop();assert.equal(await readFile(file,'utf8'),original);
 await rm(join(home,'fs-fault.json'));h=await host(home,options);
 assert.equal(failed.status,400);assert.equal(live.search.keySet,false);assert.equal((await h.call('/api/agent/tools')).data.search.keySet,false);
 assert.equal((await put(h,{search:{provider:'exa',key:'fixture-secret'}})).status,200);
 await h.stop();h=await host(home,options);
 const after=(await h.call('/api/agent/tools')).data;assert.equal(after.search.keySet,true);assert.ok(!JSON.stringify(after).includes('fixture-secret'));
});
