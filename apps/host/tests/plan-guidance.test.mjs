// Strata Dev "planGuidance": the planning prompt carries rules for a good plan, and the planning turn thinks harder. Off: exactly as Live.
import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_PLAN_RULES, MAX_STEPS, parsePlan, planPrompt } from '../server/agent/plan.mjs';
import { devSettings, planGuidanceView, savePlanGuidance, setActiveDevSettings } from '../server/dev-mode.mjs';

test('without rules the planning prompt is exactly what it was', () => {
  const live = planPrompt('build a page');
  assert.equal(live, planPrompt('build a page', null));
  assert.equal(live, planPrompt('build a page', []));
  assert.doesNotMatch(live, /How to build a good plan/);
  assert.match(live, /Use 3 to 8 steps\./);
  assert.ok(live.endsWith('Request: build a page'));
});

test('with rules they sit between the format lines and the request', () => {
  const text = planPrompt('build a page', DEFAULT_PLAN_RULES);
  const at = text.indexOf('How to build a good plan:');
  assert.ok(at > text.indexOf('Write nothing after the list.') && at < text.indexOf('Request: build a page'));
  for (const rule of DEFAULT_PLAN_RULES) assert.ok(text.includes(`- ${rule}`));
  assert.ok(text.endsWith('Request: build a page'));
});

test('dev.config.json "planGuidance": off unless enabled; thinking and rules are checked', () => {
  for (const raw of [undefined, null, false, {}, { enabled: false }, { enabled: 'yes' }, 'on']) assert.equal(devSettings({ planGuidance: raw }).planGuidance, null);
  assert.deepEqual(devSettings({ planGuidance: { enabled: true } }).planGuidance, { enabled: true, thinking: null, maxSteps: 20, rules: null });
  assert.equal(devSettings({ planGuidance: { enabled: true, thinking: 'High' } }).planGuidance.thinking, 'High');
  assert.equal(devSettings({ planGuidance: { enabled: true, thinking: 'extreme' } }).planGuidance.thinking, null, 'a wrong level leaves the chat\'s own');
  const own = devSettings({ planGuidance: { enabled: true, rules: ['  Test early. ', '', 7, 'x'.repeat(500)] } }).planGuidance.rules;
  assert.deepEqual(own.slice(0, 1), ['Test early.']);
  assert.equal(own.length, 2); assert.equal(own[1].length, 300);
  assert.equal(devSettings({ planGuidance: { enabled: true, rules: Array.from({ length: 30 }, (_, i) => `rule ${i}`) } }).planGuidance.rules.length, 12);
});

test('the steps a plan keeps: 12 as in Live, the configured number in Dev, and the prompt names it', () => {
  const reply = ['## Big job', ...Array.from({ length: 25 }, (_, i) => `${i + 1}. Step ${i + 1}`)].join('\n');
  try {
    setActiveDevSettings(null);
    assert.equal(parsePlan(reply, 'x').steps.length, MAX_STEPS);
    setActiveDevSettings(devSettings({ planGuidance: { enabled: true } }));
    assert.equal(parsePlan(reply, 'x').steps.length, 20, 'Dev default');
    setActiveDevSettings(devSettings({ planGuidance: { enabled: true, maxSteps: 30 } }));
    assert.equal(parsePlan(reply, 'x').steps.length, 25, 'all of them under a limit of 30');
    setActiveDevSettings(devSettings({ planGuidance: { enabled: false, maxSteps: 30 } }));
    assert.equal(parsePlan(reply, 'x').steps.length, 12, 'off: as Live');
  } finally { setActiveDevSettings(null); }
  assert.match(planPrompt('x', DEFAULT_PLAN_RULES, 20), /Use 3 to 20 steps\./);
  assert.match(planPrompt('x'), /Use 3 to 8 steps\./, 'Live: unchanged');
  for (const bad of [7, 41, 12.5, '20', null]) assert.equal(devSettings({ planGuidance: { enabled: true, maxSteps: bad } }).planGuidance.maxSteps, 20, String(bad));
});

test('Settings page: saving writes planGuidance, checks every value first and keeps the rules and the rest of the file', () => {
  const files = { f: JSON.stringify({ version: 1, planGuidance: { enabled: false, thinking: 'Low', rules: ['Test early.'] }, badge: { label: 'X' } }) };
  const io = { read: f => { if (!(f in files)) throw Object.assign(new Error('no'), { code: 'ENOENT' }); return files[f]; }, write: (f, t) => { files[f] = t; }, rename: (a, b) => { files[b] = files[a]; delete files[a]; } };
  const saved = savePlanGuidance('f', { enabled: true, thinking: 'High', maxSteps: 16 }, io);
  assert.deepEqual(saved, { enabled: true, thinking: 'High', maxSteps: 16 });
  const after = JSON.parse(files.f);
  assert.deepEqual(after.planGuidance, { enabled: true, thinking: 'High', maxSteps: 16, rules: ['Test early.'] });
  assert.equal(after.badge.label, 'X');
  assert.equal('thinking' in JSON.parse((savePlanGuidance('f', { enabled: true, thinking: null, maxSteps: 16 }, io), files.f)).planGuidance, false, 'the chat own level: no thinking key');
  const before = files.f;
  for (const bad of [{ enabled: 'x', thinking: null, maxSteps: 16 }, { enabled: true, thinking: 'Huge', maxSteps: 16 }, { enabled: true, thinking: null, maxSteps: 5 }, { enabled: true, thinking: null, maxSteps: 16.5 }, null]) assert.throws(() => savePlanGuidance('f', bad, io));
  assert.equal(files.f, before, 'a refused value changes nothing');
  assert.deepEqual(planGuidanceView(undefined), { enabled: false, thinking: null, maxSteps: 20 });
});

test('plan hints: a cut plan is named, and so is a point of the request that no step mentions', async () => {
  const { planHints, requestPoints, parsePlan } = await import('../server/agent/plan.mjs');
  const request = ['Build a tool.', '1. The tool must keep a `retry_after` value per key.', '2. Rejected requests change no state at all.', '3. Reset works for one key or for all keys.'].join('\n');
  assert.equal(requestPoints(request).length, 3);
  assert.equal(requestPoints('Just one line, nothing numbered here at all.').length, 0, 'too few points: no check');
  const steps = [{ text: 'Write the retry_after helper', detail: '' }, { text: 'Test that rejected requests leave state alone', detail: '' }];
  const hints = planHints(request, { steps, cut: 0 });
  assert.equal(hints.length, 1);
  assert.match(hints[0], /No step seems to name/); assert.match(hints[0], /Reset works/); assert.doesNotMatch(hints[0], /retry_after/);
  assert.deepEqual(planHints(request, { steps: [...steps, { text: 'Reset one key or all keys', detail: '' }], cut: 0 }), []);
  const cutHint = planHints('x', { steps: Array.from({ length: 20 }, (_, i) => ({ text: `s${i}`, detail: '' })), cut: 7 }, 20);
  assert.match(cutHint[0], /wrote 27 steps; the plan keeps the first 20/);
  // parsePlan reports the cut only in Dev
  const reply = ['## Big', ...Array.from({ length: 25 }, (_, i) => `${i + 1}. Step ${i + 1}`)].join('\n');
  try {
    setActiveDevSettings(null);
    assert.equal('cut' in parsePlan(reply, 'x'), false, 'Live: no extra field');
    setActiveDevSettings(devSettings({ planGuidance: { enabled: true, maxSteps: 16 } }));
    assert.equal(parsePlan(reply, 'x').cut, 9);
  } finally { setActiveDevSettings(null); }
});

test('a plan step asks for a short closing report in Dev, and nothing more than before outside it', async () => {
  const { stepObjective } = await import('../server/agent/plan.mjs');
  const plan = { title: 'T', steps: [{ text: 'Write the file', detail: 'one file' }, { text: 'Test it', detail: '' }] };
  try {
    setActiveDevSettings(null);
    const live = stepObjective(plan, 0);
    assert.doesNotMatch(live, /at most two short sentences/);
    setActiveDevSettings(devSettings({ planGuidance: { enabled: true } }));
    const dev = stepObjective(plan, 0);
    assert.match(dev, /close with at most two short sentences/);
    assert.equal(dev.replace(/ When you are done, close with[^.]*\.[^.]*\./, ''), live, 'only that sentence is added');
    setActiveDevSettings(devSettings({ planGuidance: { enabled: false } }));
    assert.equal(stepObjective(plan, 0), live);
  } finally { setActiveDevSettings(null); }
});
