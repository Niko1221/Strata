import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,readdir,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {host} from './helpers/reference-host.mjs';
const markdown='---\nname: imported-check\ndescription: Test instructions.\n---\n\nExact line one.\nExact line two.\n';
test('local routes reject cross-origin/fake references; preview writes nothing; import persists exact body, enable choice and scope', {timeout:90000},async()=>{
  const home=await mkdtemp(join(tmpdir(),'strata-reference-routes-'));let h=await host(home);
  try{
    const scopes=(await h.call('/api/workspaces')).data.workspaces,scope=scopes[0].id;
    const directory=join(home,'data','skills');await mkdir(directory,{recursive:true});const before=await readdir(directory);
    assert.equal((await h.call('/api/skills/preview',{markdown},{origin:'http://localhost:1'})).status,403);
    const preview=await h.call('/api/skills/preview',{markdown});assert.equal(preview.status,200);assert.equal(preview.data.enabled,false);assert.deepEqual(await readdir(directory),before);
    assert.equal((await h.call('/api/skills',{markdown,onlyIn:scope})).status,400);assert.deepEqual(await readdir(directory),before);
    assert.equal((await h.call('/api/skills',{markdown,importConfirmed:true,onlyIn:'missing'})).status,400);assert.deepEqual(await readdir(directory),before);
    const imported=await h.call('/api/skills',{markdown,importConfirmed:true,onlyIn:scope,enabled:false,triggers:['explicit trigger']});assert.equal(imported.status,201);
    const skill=imported.data.skills.find(s=>s.id===imported.data.id);assert.equal(skill.body,preview.data.body);assert.equal(skill.enabled,false);assert.equal(skill.onlyIn,scope);assert.deepEqual(skill.triggers,['explicit trigger']);
    assert.match(await readFile(join(directory,'imported-check','SKILL.md'),'utf8'),/Exact line one\.\nExact line two\./);
    assert.equal((await h.call('/api/skills',{markdown,importConfirmed:true})).status,400);
    assert.equal((await h.call('/api/skills',{name:'Oversize',body:'x'.repeat(40001)})).status,400);
    await h.stop();h=await host(home);
    const listed=(await h.call('/api/skills')).data.find(s=>s.id===skill.id);assert.equal(listed.onlyIn,scope);assert.equal(listed.enabled,false);
    assert.equal((await h.call('/api/chat/stream',{sessionId:'a',message:'hello',webResultId:'invented'})).status,400);
    assert.equal((await h.call('/api/chat/web-search',{sessionId:'a',query:'explicit user query',confirmed:true},{origin:'http://localhost:1'})).status,403);
  }finally{await h.stop();await rm(home,{recursive:true,force:true});}
});
