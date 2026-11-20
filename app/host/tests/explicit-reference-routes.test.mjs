import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,readdir,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {host} from './helpers/reference-host.mjs';
import {createMockModelApi} from './mock-model-api.mjs';
test('attachment reaches only plain completion; untrusted snippet cannot activate enabled skills or autonomous tools', {timeout:90000},async()=>{
  const home=await mkdtemp(join(tmpdir(),'strata-reference-prompt-')),mock=await createMockModelApi(),h=await host(home);
  try{
    await h.call('/api/skills',{name:'Untrusted trigger skill',body:'PRIVILEGED_SENTINEL',enabled:true,triggers:['untrusted-trigger']});
    await h.call('/api/skills',{name:'Question trigger skill',body:'TRUSTED_SENTINEL',enabled:true,triggers:['original-question']});
    const connection=(await h.call('/api/model-providers',{kind:'custom',name:'Isolated mock',baseUrl:mock.baseUrl,protocol:'openai-chat',models:[{id:'mock-chat',reasoning:false,vision:false}]})).data.connections[0];
    const result=await h.call('/api/chat/web-search',{sessionId:'a',query:'explicit user query',confirmed:true,history:'PRIVATE_HISTORY'});assert.equal(result.status,200);assert.equal(result.data.sources.length,1);
    assert.equal((await h.call('/api/chat/stream',{sessionId:'b',message:'original-question',webResultId:result.data.resultId})).status,400);
    const res=await fetch(h.base+'/api/chat/stream',{method:'POST',headers:{origin:h.base,'content-type':'application/json'},body:JSON.stringify({sessionId:'a',message:'original-question',agent:true,useTools:true,webResultId:result.data.resultId,remoteModel:{connectionId:connection.id,modelId:'mock-chat'}})});
    assert.equal(res.status,200);const frames=(await res.text()).trim().split('\n').map(JSON.parse);
    assert.equal(frames.some(f=>f.agent===true||f.type==='tool'),false);
    const request=mock.requests.find(r=>r.path==='/v1/chat/completions');assert.ok(request);assert.equal(request.body.model,'mock-chat');assert.equal(request.body.tools,undefined);
    const system=request.body.messages.find(m=>m.role==='system').content;assert.match(system,/TRUSTED_SENTINEL/);assert.doesNotMatch(system,/PRIVILEGED_SENTINEL|untrusted-trigger/);
    const references=request.body.messages.find(m=>typeof m.content==='string'&&m.content.includes('untrusted-trigger'));assert.equal(references.role,'user');assert.match(references.content,/snippets only/);assert.doesNotMatch(references.content,/PRIVATE_HISTORY/);
    assert.equal(request.body.messages.at(-1).content,'original-question');
  }finally{await h.stop();await mock.close();await rm(home,{recursive:true,force:true});}
});
