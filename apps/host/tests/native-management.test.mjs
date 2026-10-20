import test from 'node:test';
import assert from 'node:assert/strict';
import { nativeFixture, finishJob } from './helpers/native-fixture.mjs';

test('managed preparation rechecks work after dispatch and preserves the loaded engine on refusal', async t => {
  let checks = 0;
  const f = nativeFixture({ beforePrepare: async () => { checks++; throw new Error('A goal started while reviewing.'); } });
  t.after(() => f.cleanup());
  await f.control.refresh(true);
  const loaded = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs' }).id);
  assert.equal(loaded.state, 'done');
  const prepared = await finishJob(f.control, f.control.operation('install', { family: 'qwen', model: 'IQ3_S', managedDownload: true, loadAfter: false }).id);
  assert.equal(checks, 1);
  assert.equal(prepared.state, 'error');
  assert.match(prepared.error, /goal started/);
  assert.equal(f.control.state().instances[0].id, loaded.result.instanceId);
  assert.equal(f.rpcCalls.some(call => call.name === 'strata_install' && call.arguments.confirm === true), false);
});

test('cancelling while managed preparation checks work does not unload a running model', async t => {
  let release, entered;
  const checking = new Promise(resolve => { entered = resolve; });
  const f = nativeFixture({ beforePrepare: () => { entered(); return new Promise(resolve => { release = resolve; }); } });
  t.after(() => f.cleanup());
  await f.control.refresh(true);
  const loaded = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs' }).id);
  const job = f.control.operation('install', { family: 'qwen', model: 'IQ3_S', managedDownload: true, loadAfter: false });
  await checking;
  await f.control.cancel(job.id); release();
  await new Promise(resolve => setTimeout(resolve, 20));
  assert.equal(f.control.state().jobs.find(value => value.id === job.id).state, 'cancelled');
  assert.equal(f.control.state().instances[0].id, loaded.result.instanceId);
  assert.equal(f.rpcCalls.some(call => call.name === 'strata_install' && call.arguments.confirm === true), false);
});
