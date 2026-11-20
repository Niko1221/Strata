import test from 'node:test';
import assert from 'node:assert/strict';
import { btwRequest } from '../server/chat-btw.mjs';
import { researchPrompt, researchEvidence, researchWritePrompt } from '../server/agent/plan.mjs';

test('side question accepts only bounded user/assistant text and ignores state/tool instructions', () => {
  const history = [{role:'system',content:'must not pass'}, {role:'user',content:'context'}, {role:'assistant',content:'answer'}, {role:'tool',content:'not context'}];
  const body = {sessionId:'chat-1',message:'What does that mean?',history,agent:true,useTools:true,goalAction:'clear'};
  const side = btwRequest(body);
  assert.deepEqual(side.history,history.slice(1,3));
  assert.match(side.message,/no tools/);
  assert.match(side.message,/What does that mean/);
  assert.deepEqual(body.history,history);
  for (const change of [{sessionId:'../escape'}, {message:''}, {message:'x'.repeat(4001)}]) assert.throws(()=>btwRequest({...body,...change}));
  const bounded=btwRequest({...body,history:Array.from({length:100},()=>({role:'user',content:'x'.repeat(20_000)}))});
  assert.ok(bounded.history.reduce((sum,row)=>sum+row.content.length,0)<=64_000);
});

test('deep research counts distinct successful searches and page URLs, not retries or failures', () => {
  const cards = [
    ...['one','two','three','ONE'].map(query=>({name:'web_search',state:'done',args:{query}})),
    ...['https://example.com/a','https://example.com/b','https://example.org/c','https://example.org/d','https://example.org/d#fragment'].map(url=>({name:'web_fetch',state:'done',args:{url}})),
    {name:'web_fetch',state:'error',args:{url:'https://example.com/failure'}},
    {name:'web_fetch',state:'done',args:{url:'file:///private'}},
  ];
  const evidence=researchEvidence(cards.map(card=>({card})));
  assert.equal(evidence.complete,true); assert.equal(evidence.queries.length,3); assert.equal(evidence.urls.length,4);
  assert.equal(researchEvidence(cards.slice(0,2)).complete,false);
  assert.match(researchPrompt('topic',true),/three different/);
  assert.match(researchPrompt('topic',true),/Do not change files/);
  assert.match(researchWritePrompt('topic',researchEvidence([])),/DID NOT pass/);
  assert.match(researchWritePrompt('topic',evidence),/do not prove/);
  assert.doesNotMatch(researchPrompt('topic'),/Deep research/);
});
