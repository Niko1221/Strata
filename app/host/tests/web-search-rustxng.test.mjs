import test from 'node:test';
import assert from 'node:assert/strict';
import { createRustxngSearch, rustxngEndpoint } from '../server/web-search-rustxng.mjs';
import { createChatWebSearch } from '../server/chat-web-search.mjs';

test('RustXNG uses a fixed loopback endpoint, bounded fields, public sources and no raw page fetching', async () => {
  let called=0;
  const provider=createRustxngSearch({endpoint:'http://127.0.0.1:29483/',fetch:async(url,init)=>{
    called++; assert.equal(url.hostname,'127.0.0.1'); assert.equal(url.pathname,'/search'); assert.equal(url.searchParams.get('max_results'),'5'); assert.equal(url.searchParams.get('include_raw_content'),'0');assert.equal(init.redirect,'error');
    return Response.json({results:[{url:'http://127.0.0.1/private',title:'bad'},{url:'https://example.com/a',title:'Good',content:'snippet'},{url:'https://example.com/a',title:'duplicate'}]});
  }});
  assert.deepEqual((await provider.search({query:'public query'})).sources,[{url:'https://example.com/a',title:'Good',snippet:'snippet'}]); assert.equal(called,1);
  for(const url of ['https://example.com/','http://localhost:29483/','http://127.0.0.1:29483/other','http://user:pass@127.0.0.1:29483/','http://127.0.0.1:29483/?q=secret'])assert.throws(()=>rustxngEndpoint(url));
});
test('RustXNG failures, missing results, malformed/oversized JSON and cancellation are observable', async()=>{
  for (const response of [Response.json({results:[]}),Response.json({bad:[]}),new Response('invalid'),new Response('x'.repeat(1024*1024+1)),new Response(null,{status:429})]) {
    const search=createRustxngSearch({endpoint:'http://127.0.0.1:29483/',fetch:async()=>response});
    await assert.rejects(()=>search.search({query:'query'}),error=>/^WEB_/.test(error.code));
  }
  let calls=0; const controller=new AbortController(); controller.abort();
  await assert.rejects(()=>createRustxngSearch({endpoint:'http://127.0.0.1:29483/',fetch:async()=>{calls++;}}).search({query:'x'},controller.signal)); assert.equal(calls,0);
});
test('manual RustXNG respects explicit confirmation and the selected provider',async()=>{
  let calls=0;const search=createChatWebSearch({settings:()=>({groups:[{id:'search',enabled:true}],search:{provider:'rustxng'}}),rustxngProvider:{search:async()=>{calls++;return {sources:[{url:'https://example.com',title:'ok',snippet:'text'}]};}}});
  assert.equal(search.status().available,true);assert.match(search.status().destination,/Full/);
  await assert.rejects(()=>search.search({sessionId:'chat',query:'query'}));assert.equal(calls,0);
  assert.equal((await search.search({sessionId:'chat',query:'query',confirmed:true})).sources.length,1);assert.equal(calls,1);
});
