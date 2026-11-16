import {verifiedLocalRuntime} from './helpers/verified-local-runtime.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import childProcess from 'node:child_process';
import {syncBuiltinESMExports} from 'node:module';
import {EventEmitter} from 'node:events';
import {PassThrough} from 'node:stream';
import {mkdtempSync,rmSync,writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

async function fixture(t, health) {
  const home=mkdtempSync(join(tmpdir(),'strata-health-'));
  const expert=verifiedLocalRuntime(t, home);
  const before={home:process.env.STRATA_HOME,binary:process.env.STRATA_LLAMA_SERVER};
  process.env.STRATA_HOME=home;
  const proc=Object.assign(new EventEmitter(),{pid:424243,exitCode:null,signalCode:null,stdout:new PassThrough(),stderr:new PassThrough()});
  proc.kill=signal=>{if(proc.exitCode!==null)return false;proc.signalCode=signal;proc.emit('exit',null,signal);return true;};
  t.mock.method(childProcess,'spawn',()=>proc);syncBuiltinESMExports();
  t.mock.method(globalThis,'fetch',async url=>String(url).endsWith('/health')?health(proc):new Response(''));
  const supervisor=await import('../server/supervisor.mjs');
  t.after(async()=>{proc.exitCode=0;proc.emit('exit',0,null);await supervisor.stop();t.mock.restoreAll();syncBuiltinESMExports();
    if(before.home===undefined)delete process.env.STRATA_HOME;else process.env.STRATA_HOME=before.home;
    if(before.binary===undefined)delete process.env.STRATA_LLAMA_SERVER;else process.env.STRATA_LLAMA_SERVER=before.binary;
    proc.stdout.destroy();proc.stderr.destroy();rmSync(home,{recursive:true,force:true});});
  const model=join(home,'fixture.gguf');writeFileSync(model,'fixture');
  return {proc,supervisor,options:{modelId:'fixture',modelPath:model,config:{beginner:{contextLength:'2k'},expert},hardware:{cpu:{cores:4}},timeoutMs:50}};
}

test('a ready HTTP response from a child that already exited cannot publish running',async t=>{
  const f=await fixture(t,proc=>{proc.exitCode=1;proc.emit('exit',1,null);return Response.json({status:'ok'});});
  await assert.rejects(f.supervisor.start(f.options),/Engine exited early with code 1/);
  assert.equal(f.supervisor.status().running,false);assert.equal(f.supervisor.status().pid,null);
});

test('startup timeout distinguishes live loading503 from transport silence',async t=>{
  const f=await fixture(t,()=>Response.json({error:{code:503,message:'Loading model',type:'unavailable_error'}},{status:503}));
  await assert.rejects(f.supervisor.start(f.options),/Engine did not become healthy in time.*HTTP 503.*loading/i);
  assert.equal(f.supervisor.status().running,false);
});

test('HTTP200 without the exact ready status remains unhealthy',async t=>{
  const f=await fixture(t,()=>Response.json({status:'loading'}));
  await assert.rejects(f.supervisor.start(f.options),/Engine did not become healthy in time/);
  assert.equal(f.supervisor.status().running,false);
});
