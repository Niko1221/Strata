// Strata Dev's four chat modes: Manual, Plan + Manual, Plan + Auto, Auto (src/dev/chat-modes.ts).
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';

const built = await build({ stdin: { contents: "export * from './src/dev/chat-modes';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node' });
const { DEV_MODES, devModeOf, hostModeOf, plansFirst, planFirstText } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);

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
