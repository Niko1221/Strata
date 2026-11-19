import test from 'node:test';
import assert from 'node:assert/strict';
import { createChatWebSearch, providerFetch, publicSourceUrl, referenceHistory } from '../server/chat-web-search.mjs';

const enabled = () => ({ groups: [{id:'search',enabled:true}], search:{provider:'builtin'} });
test('manual search has no implicit calls; validates query, settings, session and confirmation', async () => {
  let calls=0, settings=enabled();
  const search=createChatWebSearch({settings:()=>settings, provider:{search:async r=>{calls++;assert.deepEqual(r,{query:'weather',maxResults:5});return {sources:[]};}}});
  assert.equal(calls,0);
  for(const input of [{query:'weather'}, {sessionId:'a',query:'weather'}, {sessionId:'a',query:'x'.repeat(401),confirmed:true}]) await assert.rejects(()=>search.search(input));
  settings={...enabled(),groups:[{id:'search',enabled:false}]};
  await assert.rejects(()=>search.search({sessionId:'a',query:'weather',confirmed:true}), /disabled/);
  settings={...enabled(),search:{provider:'exa',keySet:false}};
  await assert.rejects(()=>search.search({sessionId:'a',query:'weather',confirmed:true}), /Settings/);
  assert.equal(calls,0);settings=enabled();
  assert.deepEqual((await search.search({sessionId:'a',query:'weather',confirmed:true,history:'secret'})).sources,[]);
  assert.equal(calls,1);
});
test('opaque bounded records expire, reject fake/wrong chat IDs, and retain only public URLs', async()=>{
  let now=0;
  const sources=[{url:'javascript:alert(1)',title:'bad'}, {url:'https://127.0.0.1/',title:'bad'}, ...Array.from({length:8},(_,i)=>({url:`https://example.com/${i}`,title:'title',snippet:'trigger-secret'}))];
  const search=createChatWebSearch({settings:enabled,now:()=>now,ttlMs:20,maxRecords:1,provider:{search:async()=>({sources})}});
  const a=await search.search({sessionId:'a',query:'one',confirmed:true});
  assert.equal(a.sources.length,5);assert.match(a.resultId,/^[a-f0-9]{48}$/);
  assert.throws(()=>search.resolve(a.resultId,'b'));assert.throws(()=>search.resolve('fake','a'));
  const history=referenceHistory([{role:'assistant',content:'prior'}],search.resolve(a.resultId,'a'));
  assert.equal(history[1].role,'user');assert.match(history[1].content,/untrusted/);assert.match(history[1].content,/trigger-secret/);
  assert.equal(history.some(m=>m.role==='system'),false);
  const b=await search.search({sessionId:'a',query:'two',confirmed:true});assert.throws(()=>search.resolve(a.resultId,'a'));
  now=21;assert.throws(()=>search.resolve(b.resultId,'a'));
  for(const url of ['http://localhost/','https://10.0.0.1/','https://[::1]/','https://[::ffff:127.0.0.1]/','https://example.local/','file:///tmp/a','https://user:pass@example.com/']) assert.equal(publicSourceUrl(url),null,url);
});
test('provider redirects stay HTTPS on fixed search hosts and paths',async()=>{
  const visited=[];
  const fetch=providerFetch(async(url,init)=>{visited.push(String(url));assert.equal(init.redirect,'manual');return visited.length===1?new Response(null,{status:302,headers:{location:'https://lite.duckduckgo.com/lite/'}}):new Response('ok');});
  assert.equal(await (await fetch('https://html.duckduckgo.com/html/',{method:'POST',body:'q=weather'})).text(),'ok');
  assert.equal(visited.length,2);
  for(const location of ['http://html.duckduckgo.com/html/','https://127.0.0.1/','https://example.com/','https://www.bing.com/private']){
    let calls=0;const guarded=providerFetch(async()=>{calls++;return new Response(null,{status:302,headers:{location}});});
    await assert.rejects(()=>guarded('https://html.duckduckgo.com/html/'));assert.equal(calls,1);
  }
});
test('cancelled or late provider replies never create attachable records; failures stay plain',async()=>{
  const abort=new AbortController();let finish;
  const search=createChatWebSearch({settings:enabled,provider:{search:()=>new Promise(r=>finish=r)}});
  const pending=search.search({sessionId:'a',query:'weather',confirmed:true},abort.signal);
  abort.abort();finish({sources:[{url:'https://example.com',title:'late'}]});await assert.rejects(()=>pending);
  const failed=createChatWebSearch({settings:enabled,provider:{search:async()=>{throw new Error('private query or token');}}});
  await assert.rejects(()=>failed.search({sessionId:'a',query:'weather',confirmed:true}),e=>!e.message.includes('private'));
});
