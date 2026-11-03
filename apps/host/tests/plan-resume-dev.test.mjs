// Strata Dev: a plan resumed after its step's goal was already closed goes on with the next step, and a resume that fails leaves the plan
// paused (not 'running' with nothing behind it). Real Harness with a scripted model; skipped without STRATA_HARNESS_DIR.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { devSettings, setActiveDevSettings } from '../server/dev-mode.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
const wait = ms => new Promise(r => setTimeout(r, ms));

test('resume after the step goal was closed: that step is done and the plan goes on to the next one', { skip }, async (t) => {
  setActiveDevSettings(devSettings({ planGuidance: { enabled: true } }));
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-planresume-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const local = [{ id: 'inst-1', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { setActiveDevSettings(null); await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const base = (chatId, onFrame = () => {}) => ({ chatId, workspacePath: ws, mode: 'bypass', onFrame });
  await bridge.planCommand({ ...base('r1'), line: 'SCRIPT:planreply' });
  const plan = bridge.planState('r1');
  bridge.planControl({ chatId: 'r1', planId: plan.id, action: 'edit', steps: ['Quick one SCRIPT:goalstep', 'Slow one SCRIPT:slow 400', 'Last one SCRIPT:goalstep'] });
  const out = [];
  const run = bridge.planGoal({ ...base('r1', f => out.push(f)), planId: plan.id });
  for (let i = 0; i < 100 && !out.some(f => f.type === 'plan-step' && f.index === 2 && f.state === 'running'); i++) await wait(100);
  await wait(1500);
  await bridge.stop('r1'); await run;
  assert.equal(bridge.planState('r1').state, 'paused');
  // The model had closed the goal of step 2 just before the stop.
  await bridge.goalCommand({ ...base('r1'), line: 'complete' });
  assert.equal((await bridge.goalState('r1')).phase, 'complete');
  const again = [];
  await bridge.goalCommand({ ...base('r1', f => again.push(f)), line: 'resume' });
  const after = bridge.planState('r1');
  assert.equal(after.steps[1].state, 'done', 'the step whose goal was closed is done');
  assert.ok(['running', 'done'].includes(after.steps[2].state), 'the next step has run or is running');
  assert.ok(after.state !== 'running' || after.steps[2].state === 'running', 'never running with nothing behind it');
});

test('Dev planGuidance: a step before the last shows one done line instead of its closing words; the last step keeps its words; off: as before', { skip }, async (t) => {
  for (const dev of [true, false]) {
    setActiveDevSettings(dev ? devSettings({ planGuidance: { enabled: true } }) : null);
    const worker = await startScriptedWorker({});
    const dir = mkdtempSync(join(tmpdir(), 'strata-stepwords-'));
    const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
    const local = [{ id: 'inst-1', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }];
    const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], env: process.env, startTimeoutMs: 90_000 });
    try {
      const base = (onFrame = () => {}) => ({ chatId: 'w', workspacePath: ws, mode: 'bypass', onFrame });
      await bridge.planCommand({ ...base(), line: 'SCRIPT:planreply' });
      const out = [];
      await bridge.planGoal({ ...base(f => out.push(f)), planId: bridge.planState('w').id });
      const said = out.filter(f => f.type === 'delta').map(f => f.text).join('');
      const closings = said.match(/Step finished\. The result is checked\./g)?.length ?? 0;
      assert.equal(closings, dev ? 1 : 3, dev ? 'only the last step keeps its closing words' : 'Live: every step');
      assert.equal(out.filter(f => f.type === 'plan-step' && f.state === 'done').length, 3);
      assert.equal(bridge.planState('w').state, 'done');
    } finally { await bridge.close(); await worker.close(); setActiveDevSettings(null); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } }
  }
});
