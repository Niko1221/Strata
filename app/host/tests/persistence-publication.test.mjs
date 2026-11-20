import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,readFile,rm,readdir} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {host} from './helpers/reference-host.mjs';
const options={preloads:['./tests/fixtures/state-persistence-fault.mjs']};
async function put(h,body){const r=await fetch(h.base+'/api/agent/tools',{method:'PUT',headers:{origin:h.base,'content-type':'application/json'},body:JSON.stringify(body)});return {status:r.status,data:await r.json()};}
for(const mode of ['statePublish','partialWrite'])test(`Tools ${mode} failure retains the complete committed file and key state across restart`,async t=>{
 const home=await mkdtemp(join(tmpdir(),'strata-publish-state-'));let h=await host(home,options);
 t.after(async()=>{await h.stop();await rm(home,{recursive:true,force:true});});
 assert.equal((await put(h,{search:{provider:'exa'}})).status,200);
 const file=join(home,'data','state.json'),original=await readFile(file,'utf8');
 await writeFile(join(home,'fs-fault.json'),JSON.stringify({[mode]:true}));
 const failed=await put(h,{search:{key:'uncommitted-key'}});
 assert.equal(failed.status,400);assert.equal((await h.call('/api/agent/tools')).data.search.keySet,false);
 assert.equal(await readFile(file,'utf8'),original);
 assert.ok(!(await readdir(join(home,'data'))).some(name=>/^state\.json\..*\.tmp$/.test(name)));
 await h.stop();await rm(join(home,'fs-fault.json'));h=await host(home,options);
 assert.equal((await h.call('/api/agent/tools')).data.search.keySet,false);
});

for(const rollbackFails of [false,true])test(`failed skill publication ${rollbackFails?'with failed rollback':'rolls preferences back'} never publishes instructions`,async t=>{
 const home=await mkdtemp(join(tmpdir(),'strata-publish-skill-'));let h=await host(home,options);
 t.after(async()=>{await h.stop();await rm(home,{recursive:true,force:true});});
 const scope=(await h.call('/api/workspaces')).data.workspaces[0].id;
 assert.equal((await put(h,{search:{provider:'builtin'}})).status,200);
 const file=join(home,'data','state.json'),original=await readFile(file,'utf8');
 await writeFile(join(home,'fs-fault.json'),JSON.stringify({skillPublish:true,cleanup:true,...(rollbackFails?{statePublishAfter:1}:{})}));
 const failed=await h.call('/api/skills',{markdown:'---\nname: pending-publication\ndescription: Staged instructions.\n---\nNEVER_PUBLISHED_SENTINEL',importConfirmed:true,enabled:false,onlyIn:scope,triggers:['explicit']});
 assert.equal(failed.status,400);
 const saved=JSON.parse(await readFile(file,'utf8'));
 if(rollbackFails){assert.deepEqual(saved.skillPreferences['my-pending-publication'],{enabled:false,onlyIn:scope});assert.match(failed.data.error,/preferences could not be restored/);}
 else assert.equal(await readFile(file,'utf8'),original);
 assert.equal((await h.call('/api/skills')).data.some(s=>s.id==='my-pending-publication'),false);
 assert.ok((await readdir(join(home,'data'))).some(name=>name.startsWith('.skill-')));
 await h.stop();await rm(join(home,'fs-fault.json'));h=await host(home,options);
 assert.equal((await h.call('/api/skills')).data.some(s=>s.id==='my-pending-publication'),false);
 const agentDirs=await readdir(join(home,'data','agent','skills-live'));
 assert.ok(!agentDirs.includes('pending-publication'));
});
