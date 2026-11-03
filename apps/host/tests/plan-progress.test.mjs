// Strata Dev plan progress bar: only finished steps count, and only a plan that is going ahead has a bar.
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';

const built = await build({ stdin: { contents: "export * from './src/dev/plan-progress';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node' });
const { planProgress } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
const plan = (state, ...steps) => ({ id: 'p', title: '', source: 'plan', state, steps: steps.map((s, i) => ({ id: `s${i}`, text: `step ${i}`, state: s })) });

test('a running plan: finished steps over all steps, and the step being worked on', () => {
  assert.deepEqual(planProgress(plan('running', 'done', 'done', 'running', 'pending', 'pending')), { done: 2, total: 5, percent: 40, current: 'step 2' });
  assert.equal(planProgress(plan('running', 'running', 'pending')).percent, 0, 'a step in progress is not progress yet');
  assert.equal(planProgress(plan('running', 'done', 'skipped', 'pending')).done, 2, 'a skipped step has nothing left to do');
});

test('no bar for a proposal, a finished or stopped plan, or no plan; a paused plan keeps its bar', () => {
  for (const state of ['proposed', 'rejected', 'done', 'stopped']) assert.equal(planProgress(plan(state, 'done', 'pending')), null, state);
  assert.equal(planProgress(null), null);
  assert.equal(planProgress({ ...plan('running'), steps: [] }), null);
  assert.equal(planProgress(plan('paused', 'done', 'pending')).percent, 50);
  assert.equal(planProgress(plan('approved', 'pending', 'pending')).percent, 0);
});
