import {verifiedLocalRuntime} from './helpers/verified-local-runtime.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'node:net';

test('a backend that exits before health reports its error promptly', async t => {
  const home=mkdtempSync(join(tmpdir(),'strata-early-exit-'));
  const expert=verifiedLocalRuntime(t, home, {realChild:true});
  const previous={ home:process.env.STRATA_HOME, binary:process.env.STRATA_LLAMA_SERVER };
  process.env.STRATA_HOME=home;

  const supervisor=await import('../server/supervisor.mjs');
  const model=join(home,'dummy.gguf');writeFileSync(model,'fixture');
  const listener=createServer();await new Promise(resolve=>listener.listen(0,'127.0.0.1',resolve));
  const port=listener.address().port;await new Promise(resolve=>listener.close(resolve));
  t.after(async()=>{
    await supervisor.stop();
    if(previous.home===undefined)delete process.env.STRATA_HOME;else process.env.STRATA_HOME=previous.home;
    if(previous.binary===undefined)delete process.env.STRATA_LLAMA_SERVER;else process.env.STRATA_LLAMA_SERVER=previous.binary;
    rmSync(home,{recursive:true,force:true});
  });
  const started=Date.now(), secret='private-startup-key-fixture';
  const options={modelId:'fixture',modelPath:model,port,config:{beginner:{contextLength:'2k'},expert,serving:{apiKey:secret}},hardware:{cpu:{cores:4}},timeoutMs:10_000};
  assert.equal(supervisor.runObservation(),null);
  const ordinary=supervisor.buildArgs(options);
  await assert.rejects(supervisor.start({...options,runObservation:true,observationProbe:async()=>({dedicated:null,shared_gpu_memory:null,processes:null})}),/Engine exited early with code \d+.*bad option: -m/);
  const observation=supervisor.runObservation();
  assert.equal(observation.outcome,'startup_exit');assert.equal(observation.requested.contextCapacity,2048);
  assert.equal(observation.effective.contextCapacity.value,null);
  assert.ok(!JSON.stringify(observation).includes(secret));
  assert.deepEqual(supervisor.buildArgs(options),ordinary,'diagnostics do not alter launch arguments');
  assert.ok(!('observation' in supervisor.status()),'existing snapshot contract stays unchanged');
  assert.ok(Date.now()-started<5_000,'did not wait for the health timeout');
  assert.equal(supervisor.status().status,'error','cleanup must preserve the startup error');
  assert.match(supervisor.status().error,/Engine exited early/);
  let stoppedOther=false;
  await assert.rejects(supervisor.restart({modelId:'missing',modelPath:join(home,'absent.gguf'),config:{expert},hardware:{cpu:{cores:4}},beforeStart:()=>{stoppedOther=true;}}),/Model file missing/);
  assert.equal(stoppedOther,false,'a failed preflight must not unload another engine');
  assert.ok(!JSON.stringify(supervisor.status()).includes(secret),'public status must not expose the API key after startup failure');
  assert.ok(!JSON.stringify(supervisor.logs()).includes(secret),'startup logs must not expose the API key');
});
