/**
 * The agent's own question (`ask_user_question`) answerer path — the parts that need no Harness.
 *
 * Pure tests only: the wire adapters, the gateway answer envelope, the preset opt-in and the Settings toggle.
 * The end-to-end round-trip (a real `user-questions/request` from the pinned Harness, held by the bridge and
 * answered back) needs STRATA_HARNESS_DIR and lives with the other Harness tests; it is NOT covered here.
 *
 * The inbound-question and outbound-answer field names are UNCONFIRMED static analysis (see
 * server/agent/user-questions.mjs); these tests pin the app's own behaviour, not the Harness's contract.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseQuestions, answerWire } from '../server/agent/user-questions.mjs';
import { GatewayClient } from '../server/agent/gateway-client.mjs';
import { STANDARD_PLUGIN_IDS, REMOVED_PLUGIN_IDS, standardPreset } from '../server/agent/preset.mjs';
import { createAgentTools } from '../server/agent/tools-settings.mjs';

test('parseQuestions reads a questions array, a single inline question, and object options', () => {
  assert.deepEqual(
    parseQuestions({ callId: 'c1', questions: [{ question: 'Which env?', options: ['dev', 'prod'] }] }),
    { id: 'c1', questions: [{ question: 'Which env?', options: ['dev', 'prod'] }] },
  );
  assert.deepEqual(
    parseQuestions({ callId: 'c2', question: 'Overwrite it?', choices: [{ label: 'Yes' }, { value: 'No' }] }),
    { id: 'c2', questions: [{ question: 'Overwrite it?', options: ['Yes', 'No'] }] },
  );
  // A payload with no recognisable question yields no questions (the bridge then has nothing to show).
  assert.deepEqual(parseQuestions({ callId: 'c3' }), { id: 'c3', questions: [] });
});

test('answerWire carries one answer per question as structured data', () => {
  assert.deepEqual(answerWire([{ question: 'Which env?', answer: 'prod' }]), { answers: [{ question: 'Which env?', answer: 'prod' }] });
  assert.deepEqual(answerWire([]), { answers: [] });
});

test('answerEvent keeps the approval wire unchanged and carries a structured question answer', async () => {
  const sent = [];
  const gw = new GatewayClient({ baseUrl: 'http://127.0.0.1:1', token: 't' });
  gw.rpc = async (method, args) => { sent.push({ method, args }); return { ok: true }; };

  // An approval (no `value`): exactly the wire it has always been.
  await gw.answerEvent({ clientId: 'cl', eventId: 'e1', outcome: 'allowed-once' });
  assert.deepEqual(sent[0], { method: '$events/result', args: { clientId: 'cl', eventId: 'e1', outcome: { kind: 'result', value: 'allowed-once' } } });

  // A skip stays `next`.
  await gw.answerEvent({ clientId: 'cl', eventId: 'e2', outcome: 'next' });
  assert.deepEqual(sent[1].args.outcome, { kind: 'next' });

  // A question answer carries the structured value.
  await gw.answerEvent({ clientId: 'cl', eventId: 'e3', outcome: 'answered', value: answerWire([{ question: 'q', answer: 'a' }]) });
  assert.deepEqual(sent[2].args.outcome, { kind: 'result', value: { answers: [{ question: 'q', answer: 'a' }] } });
});

test('the question tool stays out of the preset by default and is added only when asked for', () => {
  const off = standardPreset({ platform: 'win32' }).plugins.map(p => p.id);
  // Default path is unchanged: exactly the standard preset minus the removed plugins (and the posix shell).
  assert.deepEqual(off, STANDARD_PLUGIN_IDS.filter(id => !REMOVED_PLUGIN_IDS.includes(id) && id !== 'tool-bash'));
  assert.ok(!off.includes('tool-ask-user'));

  const on = standardPreset({ platform: 'win32', askUser: true }).plugins.map(p => p.id);
  assert.ok(on.includes('tool-ask-user'), 'the opt-in adds the question tool');
  // It sits in the standard order, immediately before the to-do tool.
  assert.equal(on[on.indexOf('tool-ask-user') + 1], 'tool-todo');
  assert.deepEqual(off, on.filter(id => id !== 'tool-ask-user'), 'nothing else about the preset changes');
});

test('Settings -> Tools persists the askUserQuestions toggle and exposes it to the profile, off by default', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-ask-'));
  try {
    let saved;
    const tools = createAgentTools({ read: () => saved, write: v => { saved = v; }, dataDir: dir });
    assert.equal(tools.view().askUserQuestions, false, 'off unless the person turned it on');
    assert.equal(tools.profileView().askUserQuestions, false);

    tools.update({ askUserQuestions: true });
    assert.equal(tools.view().askUserQuestions, true);
    assert.equal(tools.profileView().askUserQuestions, true);

    tools.update({ askUserQuestions: false });
    assert.equal(tools.profileView().askUserQuestions, false);
  } finally {
    try { rmSync(dir, { recursive: true, force: true }); } catch { /* best effort */ }
  }
});
