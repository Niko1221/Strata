import test from 'node:test';
import assert from 'node:assert/strict';
import {join,resolve} from 'node:path';
import {configureDesktopMemory,DESKTOP_MEMORY_KEY_ENV} from '../server/hakua-desktop.mjs';
import {applyNativeApiKeys} from '../server/native-control.mjs';

test('Hakua creates a separate local principal and private-cache settings only for the fork',()=>{
 const env={},cfg={args:['--prompt-cache','16','--conversation-cache-mib=64','--context','4096'],slot_save_path:'old'};
 const result=configureDesktopMemory(cfg,{enabled:true,directory:resolve('synthetic-profile'),env});
 assert.match(result.profileKey,/^[a-f0-9]{64}$/);
 assert.equal(result.profileRoot,join(resolve('synthetic-profile'),'hakua-memory','local'));
 assert.equal(cfg.slot_save_path,undefined);
 assert.deepEqual(cfg.args,['--context','4096','--prompt-cache','0','--conversation-cache-mib','0']);
 assert.equal(cfg.hakua_harness.users[0].key_env,DESKTOP_MEMORY_KEY_ENV);
 assert.equal(cfg.hakua_harness.users[0].principal,'local');
 const hostKey=applyNativeApiKeys(cfg,env);
 assert.notEqual(hostKey,result.profileKey);
 assert.ok(cfg.api_key.includes(result.profileKey));
 assert.ok(cfg.api_key.includes(hostKey));
 assert.equal('persona_sha256' in cfg.hakua_harness.users[0],false);
 assert.equal('strata_memory' in cfg,false);
});

test('upstream and explicit operator settings remain unchanged',()=>{
 for(const cfg of [{args:['a']},{hakua_harness:{enabled:false}},{hakua_harness:{enabled:true,users:[]}}]) {
  const before=structuredClone(cfg),env={};
  assert.equal(configureDesktopMemory(cfg,{enabled:cfg.args===undefined,directory:resolve('synthetic'),env}),null);
  assert.deepEqual(cfg,before);assert.deepEqual(env,{});
 }
 assert.throws(()=>configureDesktopMemory({}, {enabled:true,directory:resolve('synthetic'),env:{STRATA_BATCH_TRACE:'1'}}),/TRACE/);
 for(const value of ['0','false','off',' OFF ']) {
   const env={STRATA_BATCH_TRACE:value};
   configureDesktopMemory({}, {enabled:true,directory:resolve('synthetic'),env});
   assert.equal(env.STRATA_BATCH_TRACE,undefined,'an off value must not enable the Python protocol logger');
 }
});
