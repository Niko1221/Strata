/** Synthetic preparation evidence only. Never starts an app, model or network request. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {apply, manualDecision, MANUAL_PRESET} from '../../apps/host/server/agent/plugins/strata-gate.mjs';
import {MODE_PRESETS} from '../../apps/host/server/agent/profile.mjs';

test('Manual uses full access and cannot restrict reads to the requested workspace', () => {
  assert.equal(MODE_PRESETS.manual.sandbox, 'danger-full-access');
  // These are deliberately fictitious path strings. No file is opened.
  for (const name of ['read', 'glob', 'grep']) {
    assert.equal(manualDecision({name, arguments: {
      file_path: 'H:/synthetic-outside-workspace/sentinel.txt',
      path: 'H:/synthetic-outside-workspace', pattern: 'sentinel',
    }}), null);
  }
});

test('one Manual web approval bypasses later request-specific admission', async () => {
  let beforeTool;
  let approvals = 0;
  let executions = 0;
  const ctx = {
    on(name, handler) { assert.equal(name, 'tools/pre-execute'); beforeTool = handler; },
    inject(names, callback) { callback({commands: {register() {}}}); },
    permissionPresets: {current: () => MANUAL_PRESET},
    approval: {request: async () => { approvals++; return 'allowed-once'; }},
    sessions: {get() { throw new Error('No subagent should be consulted'); }},
  };
  apply(ctx, {web: {provider: 'synthetic provider', skipAsk: false}});
  const agent = {session: {id: 'synthetic-only', header: {}}};
  const next = async () => { executions++; return {kind: 'allow'}; };
  // Reserved .invalid names; this fixture invokes only the permission handler.
  await beforeTool({agent, name: 'web_search', arguments: {query: 'synthetic topic'}, callId: 'one'}, next);
  await beforeTool({agent, name: 'web_fetch', arguments: {url: 'https://unapproved.invalid/'}, callId: 'two'}, next);
  assert.equal(approvals, 1);
  assert.equal(executions, 2);
});

test('write operations still require an explicit approval', () => {
  const decision = manualDecision({name: 'write', arguments: {file_path: 'H:/synthetic-only/audit.json'}});
  assert.equal(decision.kind, 'ask');
});
