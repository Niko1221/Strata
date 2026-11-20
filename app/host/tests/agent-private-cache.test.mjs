import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync, mkdirSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {buildProfile} from '../server/agent/profile.mjs';
import {createAgentBridge} from '../server/agent/agent-bridge.mjs';
import {resolveHarness} from '../server/agent/harness-runtime.mjs';
import {startScriptedWorker} from './helpers/scripted-worker.mjs';
import {privateConversationSnapshot} from '../server/agent/plugins/strata-context.mjs';

test('only a local engine acknowledging private conversation reuse gets Harness affinity', () => {
  const profile = enabled => buildProfile({local:[{id:'a',modelId:'m',baseUrl:'http://127.0.0.1:9001/v1',privateConversationCache:enabled}],proxyBase:'http://127.0.0.1:9002/proxy'});
  for(const enabled of [false,true]){
    const p=profile(enabled),providers=JSON.parse(p.patch.slice(p.patch.indexOf('['))).find(e=>e.id==='llm-pi-ai').config.providers;
    assert.equal(providers['strata-local'].headers?.['x-strata-conversation-cache'], enabled ? '1' : undefined);
    assert.equal(providers['strata-local~s0'].headers?.['x-strata-conversation-cache'], enabled ? '1' : undefined);
  }
});
test('private cache affinity stays in an immutable per-request snapshot', () => {
  const profile={headers:{'x-strata-conversation-cache':'1',existing:'kept'}};
  const original={profiles:new Map([['strata-local',profile]]),models:{}};
  const a=privateConversationSnapshot({provider:'strata-local',sessionId:'a'},original);
  const b=privateConversationSnapshot({provider:'strata-local',sessionId:'b'},original);
  assert.notEqual(a.profiles.get('strata-local').headers['x-session-affinity'],b.profiles.get('strata-local').headers['x-session-affinity']);
  assert.deepEqual(profile.headers,{'x-strata-conversation-cache':'1',existing:'kept'});
  assert.equal(a.models,original.models);
  assert.equal(a.profiles.get('strata-local').headers.existing,'kept');
  assert.equal(a.profiles.get('strata-local').headers['x-strata-conversation-cache'],undefined);
  assert.equal(privateConversationSnapshot({provider:'strata-local'},original).profiles.get('strata-local').headers['x-session-affinity'],undefined);
  assert.equal(privateConversationSnapshot({provider:'api-cloud',sessionId:'a'},original),original);
});
const skip=!process.env.STRATA_HARNESS_DIR || !resolveHarness({env:process.env}).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
test('real Harness keeps one chat affinity and changes it for another chat', {skip}, async t => {
  const worker=await startScriptedWorker();
  const root=mkdtempSync(join(tmpdir(),'strata-agent-cache-')),ws=join(root,'ws');mkdirSync(ws);
  const bridge=createAgentBridge({dataDir:root,localInstances:()=>[{id:'i',label:'S',baseUrl:worker.url,modelId:'scripted-model',context:65536,slots:1,privateConversationCache:true}],connections:()=>[],env:process.env,startTimeoutMs:90000});
  t.after(async()=>{await bridge.close();await worker.close();rmSync(root,{recursive:true,force:true,maxRetries:3});});
  const sessions=[];
  for(const chatId of ['a','a','b']){
    const before=worker.requests.length,frames=[];
    await bridge.runTurn({chatId,message:'SCRIPT:echo hello',workspacePath:ws,mode:'bypass',onFrame:f=>frames.push(f)});
    assert.equal(frames.at(-1).type,'done',JSON.stringify(frames.at(-1)));
    const sent=worker.requests.slice(before).filter(r=>r.directive==='echo' && !r.summary);
    assert.ok(sent.length>0);
    assert.ok(sent.every(r=>typeof r.cacheSession==='string' && r.cacheSession.length>0));
    sessions.push(sent.at(-1).cacheSession);
  }
  assert.equal(sessions[0],sessions[1]);
  assert.notEqual(sessions[0],sessions[2]);
});
