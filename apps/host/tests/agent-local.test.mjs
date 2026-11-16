/** Hikari / local GGUF must be eligible for the agent (web_search path). */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  supervisorIsReady,
  supervisorLocalInstance,
  mergeAgentLocalInstances,
  agentHasLocalModel,
  agentSkipReason,
} from '../server/agent-local.mjs';

test('supervisor ready gate rejects starting/stopped engines', () => {
  assert.equal(supervisorIsReady({ status: 'running', modelId: 'local-hikari', port: 8080 }), true);
  assert.equal(supervisorIsReady({ status: 'starting', modelId: 'local-hikari', port: 8080 }), false);
  assert.equal(supervisorIsReady({ status: 'running', modelId: null, port: 8080 }), false);
  assert.equal(supervisorIsReady({ status: 'running', modelId: 'x', port: 0 }), false);
});

test('supervisorLocalInstance builds an agent route for Hikari-style GGUF', () => {
  const row = supervisorLocalInstance(
    { status: 'running', modelId: 'local-e7d83e78598be18b75bf0498', port: 8080, contextCapacity: 131072 },
    { authenticationKey: () => 'sk-test', label: 'Hikari' },
  );
  assert.equal(row.id, 'supervisor:local-e7d83e78598be18b75bf0498');
  assert.equal(row.baseUrl, 'http://127.0.0.1:8080/v1');
  assert.equal(row.modelId, 'local');
  assert.equal(row.key, 'sk-test');
  assert.equal(row.label, 'Hikari');
  assert.equal(row.source, 'supervisor');
  assert.equal(supervisorLocalInstance({ status: 'stopped', modelId: 'x', port: 8080 }), null);
});

test('a supervisor-backed agent can open its serving request without native GPU fields', () => {
  const row = supervisorLocalInstance(
    { status: 'running', modelId: 'local-prism', port: 8080, contextCapacity: 32768 },
    { authenticationKey: () => 'fixture-owner-key', label: 'Prism' },
  );
  // runAgentTurn uses the same serving metadata contract for either local route.
  const request = { model: row.apiModelId, instanceId: row.id, gpuIds: row.gpus.map(String) };
  assert.deepEqual(request, { model: 'local', instanceId: 'supervisor:local-prism', gpuIds: [] });
  assert.ok(!JSON.stringify(request).includes('fixture-owner-key'));
});

test('merge prefers native ids and still adds supervisor Hikari', () => {
  const native = [{ id: 'n1', label: 'Native' }];
  const hikari = supervisorLocalInstance(
    { status: 'running', modelId: 'local-hikari', port: 8080, contextCapacity: 32768 },
    { label: 'Hikari' },
  );
  const merged = mergeAgentLocalInstances(native, hikari);
  assert.deepEqual(merged.map(r => r.id), ['n1', 'supervisor:local-hikari']);
  assert.equal(mergeAgentLocalInstances(native, null).length, 1);
});

test('unknown runtime capacity cannot fall back to metadata or the profile default', () => {
  for (const contextCapacity of [undefined, null, 0, -1, 1.5, '32768', Infinity, NaN, Number.MAX_SAFE_INTEGER + 1]) {
    assert.equal(supervisorLocalInstance(
      { status: 'running', modelId: 'gguf', port: 8080, contextCapacity },
      { context: 262144 },
    ), null);
  }
  assert.match(agentSkipReason({ chatId: 'c1', supervisorContextUnavailable: true }), /context size.*plain chat/);
  assert.equal(agentSkipReason({ chatId: 'c1', nativeLive: true, supervisorContextUnavailable: true }), null);
  assert.equal(agentSkipReason({ chatId: 'c1', remoteModel: true, supervisorContextUnavailable: true }), null);
});

test('agent skip allows supervisor-only Hikari (web search can reach it)', () => {
  assert.equal(agentHasLocalModel({ supervisorLive: true }), true);
  assert.equal(agentSkipReason({
    chatId: 'c1', remoteModel: false, nativeLive: false, supervisorLive: true,
  }), null);
  assert.match(agentSkipReason({
    chatId: 'c1', remoteModel: false, nativeLive: false, supervisorLive: false,
  }), /local GGUF|API model|Strata model/);
  assert.match(agentSkipReason({ plainChat: true, chatId: 'c1', supervisorLive: true }), /without the agent/);
});
