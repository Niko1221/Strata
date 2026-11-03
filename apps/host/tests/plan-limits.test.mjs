// Strata Dev "plan" limits: a long plan step is kept whole in Dev; outside Dev it is cut at 140 characters as it always was.
import test from 'node:test';
import assert from 'node:assert/strict';
import { parsePlan, editedSteps } from '../server/agent/plan.mjs';
import { devSettings, setActiveDevSettings } from '../server/dev-mode.mjs';

const long = 'Write a recursive-descent parser producing AST nodes for CREATE TABLE, INSERT, and the full SELECT grammar (joins, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT and OFFSET) with parse-time errors and a clean node layout';
const reply = ['Here is the plan:', '1. Write the tokenizer', `2. ${long}`, '3. Build the expression parser'].join('\n');

test('outside Dev a long step is cut at 140 characters with an ellipsis, as before', () => {
  setActiveDevSettings(null);
  const step = parsePlan(reply, 'x').steps[1];
  assert.ok(step.text.length <= 140 && step.text.endsWith('…'), step.text);
  assert.equal(editedSteps([long])[0].text.length <= 140, true);
});

test('in Dev the whole step is kept (4000 characters), and dev.config.json "plan" sets the limits within bounds', () => {
  try {
    setActiveDevSettings(devSettings({}));
    assert.equal(parsePlan(reply, 'x').steps[1].text, long);
    assert.equal(editedSteps([long])[0].text, long);
    assert.deepEqual(devSettings({}).plan, { stepTextMax: 4000, stepDetailMax: 8000 });
    assert.deepEqual(devSettings({ plan: { stepTextMax: 1200, stepDetailMax: 2000 } }).plan, { stepTextMax: 1200, stepDetailMax: 2000 });
    assert.deepEqual(devSettings({ plan: { stepTextMax: 5, stepDetailMax: 'x' } }).plan, { stepTextMax: 4000, stepDetailMax: 8000 }, 'wrong values leave the defaults');
    setActiveDevSettings(devSettings({ plan: { stepTextMax: 150 } }));
    assert.ok(parsePlan(reply, 'x').steps[1].text.length <= 150);
  } finally { setActiveDevSettings(null); }
});

test('the model gets the whole step: the goal for a step carries the full text and detail in Dev', async () => {
  const { stepObjective } = await import('../server/agent/plan.mjs');
  const text = `${long}, then double-check every error message against the spec before moving on`;
  try {
    setActiveDevSettings(devSettings({}));
    const plan = { title: 'SQL', steps: parsePlan(['1. Tokenizer', `2. ${text}`, '3. Executor'].join('\n'), 'x').steps };
    assert.ok(stepObjective(plan, 1).includes(text), 'the full step text reaches the model');
  } finally { setActiveDevSettings(null); }
});
