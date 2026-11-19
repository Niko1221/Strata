import test from 'node:test';
import assert from 'node:assert/strict';
import {nativeFixture, finishJob} from './helpers/native-fixture.mjs';
const observationProbe=async()=>Object.fromEntries(['dedicated','shared_gpu_memory','processes'].map(k=>[k,{value:null,source:null,at:null,status:'unavailable',reason:'fixture'}]));
for(const path of ['normal','retry','adopted'])test(`packaged ${path} terminal failure preserves engine exit code 23 in the actual observation`,async t=>{
 const failure=code=>Object.assign(new Error('Fixture startup failed'),{output:{engine_exit_code:code,log_tail:'private fixture path'}});
 const f=nativeFixture({observationProbe,packagedStartError:attempt=>path==='adopted'?new Error('startup_wait_expired'):failure(path==='retry'&&attempt===1?7:23)});
 t.after(()=>f.cleanup());await f.control.configure({repository:f.repo});
 let checks=0;if(path==='adopted')f.setRuntimeStatus(()=>++checks===1?{state:'starting'}:{state:'failed',engine_exit_code:23,error:'server_exited',log_tail:'private fixture path'});
 const job=f.control.operation('load',{model:'iq2_xs',runObservation:true,...(path==='retry'?{}:{retrySafer:false})});
 assert.equal((await finishJob(f.control,job.id)).state,'error');
 assert.equal(f.packagedStarts.length,path==='retry'?2:1,'exercise the intended launch path');
 const observation=f.control.runObservations()[0];assert.equal(observation.outcome,'error');assert.equal(observation.exitCode,23);
 assert.ok(!JSON.stringify(observation).includes('private'));
});

test('ordinary native load performs no observation or GPU probe',async t=>{
 let probes=0;const f=nativeFixture({observationProbe:async()=>{probes++;return observationProbe();}});t.after(()=>f.cleanup());
 await f.control.refresh();const job=f.control.operation('load',{model:'iq2_xs'});
 assert.equal((await finishJob(f.control,job.id)).state,'done');assert.equal(probes,0);assert.deepEqual(f.control.runObservations(),[]);
});

test('native diagnostics are off by default and explicitly reachable without changing snapshot',async t=>{
 const f=nativeFixture({observationProbe});t.after(()=>f.cleanup());await f.control.refresh();
 assert.deepEqual(f.control.runObservations(),[]);
 const job=f.control.operation('load',{model:'iq2_xs',runObservation:true});assert.equal((await finishJob(f.control,job.id)).state,'done');
 const rows=f.control.runObservations();assert.equal(rows.length,1);assert.equal(rows[0].backend,'native');
 assert.equal(rows[0].requested.contextCapacity,32768);assert.equal(rows[0].effective.contextCapacity.value,32768);
 assert.equal(rows[0].effective.nativeExperts.value,null);assert.equal(rows[0].outcome,'ready');
 assert.ok(!/STRATA_TOKEN|private|api_key|configPath|system_prompt/.test(JSON.stringify(rows)));
 assert.ok(!('observations' in f.control.state()));
});
test('packaged lost-runtime reconciliation finalizes capture and failed unload is not called stopped',async t=>{
 const f=nativeFixture({packaged:true,observationProbe});t.after(()=>f.cleanup());await f.control.configure({repository:f.repo});
 const job=f.control.operation('load',{model:'iq2_xs',runObservation:true});assert.equal((await finishJob(f.control,job.id)).state,'done');
 f.setMetricsStatus(503);f.setRuntimeStatus({state:'failed',engine_exit_code:19});
 for(let i=0;i<4;i++)await f.control.poll();
 assert.equal(f.control.runObservations()[0].outcome,'error');assert.equal(f.control.runObservations()[0].exitCode,19);
 f.setMetricsStatus(200);const next=f.control.operation('load',{model:'iq3_s',runObservation:true});assert.equal((await finishJob(f.control,next.id)).state,'done');
 f.setStopError(new Error('private stop failure'));
 await assert.rejects(f.control.unload(f.control.state().instances[0].id),/stop failure/);
 const capture=f.control.runObservations().at(-1);assert.equal(capture.outcome,'error');assert.ok(!JSON.stringify(capture).includes('private'));
 f.setStopError(null);
});
test('cancelled native load keeps its own terminal observation after late health response',async t=>{
 const f=nativeFixture({observationProbe});t.after(()=>f.cleanup());await f.control.refresh();f.setHealthWait(true);
 const job=f.control.operation('load',{model:'iq2_xs',runObservation:true});
 for(let i=0;i<100&&!f.hasPendingHealth();i++)await new Promise(r=>setTimeout(r,10));
 assert.ok(f.hasPendingHealth());await f.control.cancel(job.id);f.releaseHealth();
 assert.equal((await finishJob(f.control,job.id)).state,'cancelled');
 assert.equal(f.control.runObservations()[0].outcome,'cancelled');
});
test('source child exit retains startup_exit and numeric code after the load job rejects',async t=>{
 const f=nativeFixture({observationProbe});t.after(()=>f.cleanup());await f.control.refresh();f.setHealthStatus(503);
 const job=f.control.operation('load',{model:'iq2_xs',runObservation:true});
 for(let i=0;i<100&&!f.control.state().instances.length;i++)await new Promise(r=>setTimeout(r,10));
 const child=f.children.findLast(c=>c.exitCode===null&&!c.stdin);assert.ok(child);
 child.exitCode=17;child.emit('exit',17);
 assert.equal((await finishJob(f.control,job.id)).state,'error');
 const observation=f.control.runObservations()[0];assert.equal(observation.outcome,'startup_exit');assert.equal(observation.exitCode,17);
});
test('environment-enabled install loadAfter finalizes a packaged startup failure without retaining error text',async t=>{
 const before=process.env.STRATA_RUN_OBSERVATION;process.env.STRATA_RUN_OBSERVATION='1';
 const f=nativeFixture({observationProbe,packagedStartError:new Error('Fixture startup failed C:\\Users\\private\\api-key')});
 t.after(async()=>{await f.cleanup();if(before===undefined)delete process.env.STRATA_RUN_OBSERVATION;else process.env.STRATA_RUN_OBSERVATION=before;});
 await f.control.configure({repository:f.repo});assert.equal(f.control.state().available,true);
 const job=f.control.operation('install',{family:'qwen',model:'iq2_xs',loadAfter:true});
 const settled=await finishJob(f.control,job.id);assert.equal(settled.state,'error');assert.match(settled.error,/Fixture startup failed/);
 const observation=f.control.runObservations()[0];assert.ok(observation);assert.equal(observation.outcome,'error');assert.equal(observation.exitCode,null);
 assert.ok(!/private|api-key|Users/.test(JSON.stringify(observation)));
});
