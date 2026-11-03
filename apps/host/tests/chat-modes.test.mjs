// Strata Dev's four chat modes: Manual, Plan + Manual, Plan + Auto, Auto (src/dev/chat-modes.ts).
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';

const built = await build({ stdin: { contents: "export * from './src/dev/chat-modes';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node' });
const { DEV_MODES, devModeOf, hostModeOf, plansFirst, planFirstText, wantsPlanFirst } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);

test('four modes, in the order Manual, Plan + Manual, Plan + Auto, Auto', () => {
  assert.deepEqual(DEV_MODES.map(m => m.label), ['Manual', 'Plan + Manual', 'Plan + Auto', 'Auto']);
  assert.deepEqual(DEV_MODES.map(m => m.value), ['manual', 'plan-manual', 'plan-auto', 'bypass']);
});

test('what runs: Manual asks, Plan + Manual plans then asks, Plan + Auto plans then allows everything, Auto allows everything', () => {
  const row = mode => [plansFirst(mode), hostModeOf(mode)];
  assert.deepEqual(row('manual'), [false, 'manual']);
  assert.deepEqual(row('plan-manual'), [true, 'manual']);
  assert.deepEqual(row('plan-auto'), [true, 'bypass']);
  assert.deepEqual(row('bypass'), [false, 'bypass']);
});

test('older or unknown stored modes: the host\'s read-only plan is Plan + Manual, anything unknown is Manual (the safe one)', () => {
  assert.equal(devModeOf('plan'), 'plan-manual');
  for (const odd of [undefined, null, '', 'auto', 'accept-edits', 'nonsense']) assert.equal(devModeOf(odd), 'manual');
  assert.equal(hostModeOf(undefined), 'manual');
});

test('a plain message becomes /plan; a command or an empty message stays as it is', () => {
  assert.equal(planFirstText('  build a small page '), '/plan build a small page');
  assert.equal(planFirstText('/compact'), '/compact');
  assert.equal(planFirstText('/goal ship it'), '/goal ship it');
  assert.equal(planFirstText('   '), '');
});

test('the message that carries out an approved plan, and a follow-up while a plan is under way, are never turned into a new /plan', () => {
  const ask = (extra = {}) => wantsPlanFirst({ mode: 'plan-auto', body: 'build a page', ...extra });
  assert.equal(ask(), true);
  assert.equal(ask({ planState: 'proposed' }), true, 'a new request next to a plan that waits for a yes is a new plan');
  assert.equal(ask({ planState: 'done' }), true);
  assert.equal(ask({ carryOut: true, body: 'Carry out this plan, step by step' }), false, 'approving a plan runs it (it was planned again: the 2026-11-03 stop)');
  for (const planState of ['approved', 'running', 'paused']) assert.equal(ask({ planState, body: 'continue' }), false, planState);
  assert.equal(ask({ body: '/compact' }), false);
  assert.equal(wantsPlanFirst({ mode: 'manual', body: 'x' }), false);
  assert.equal(wantsPlanFirst({ mode: 'bypass', body: 'x' }), false);
});
