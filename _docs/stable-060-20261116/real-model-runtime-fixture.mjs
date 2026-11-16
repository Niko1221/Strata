/** Actual pinned Harness; synthetic loopback model, no GPU or external network. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {createAgentBridge} from '../../apps/host/server/agent/agent-bridge.mjs';
import {GROUP_IDS} from '../../apps/host/server/agent/tools-catalog.mjs';
import {startScriptedWorker} from '../../apps/host/tests/helpers/scripted-worker.mjs';
import {harnessFixtureEnvironment} from '../../apps/host/tests/helpers/harness-environment.mjs';
import {createFixture, TOOLS} from './real-model-mcp.mjs';

const CSV = {inventory: 'id,version\nalpha,1\nbeta,2\n', expected: 'id,expected_version\nalpha,2\nbeta,2\n', requirements: 'id,required_check,status\nalpha,checksum,passed\nbeta,checksum,failed\n'};
const AUDIT = {criteria: 'Synthetic fixture', version_mismatches: [{id: 'alpha'}], duplicates: [], missing_ids: [], failed_checks: [{id: 'beta'}], total_mismatches: 1, total_duplicates: 0, total_missing: 0, total_failed: 1};
function directory() {
  const root = mkdtempSync(join(tmpdir(), 'strata-real-model-synthetic-'));
  const ws = join(root, 'workspace'); mkdirSync(ws);
  for (const [name, text] of Object.entries(CSV)) writeFileSync(join(ws, name + '.csv'), text, {flag: 'wx'});
  return {root, ws};
}

test('enum MCP rejects path injection, mutation and repeat save with no raw diagnostics', () => {
  const {ws} = directory(), call = createFixture(ws);
  assert.equal(TOOLS.length, 2);
  assert.equal(call('read_audit_csv', {table: 'inventory'}).content[0].text, CSV.inventory);
  for (const args of [{table: '../private'}, {table: 'inventory', path: 'elsewhere'}, {table: 'C:/outside'}, {table: '__proto__'}]) assert.throws(() => call('read_audit_csv', args));
  assert.throws(() => call('run_code', {code: 'throw 1'}));
  call('save_audit', {audit_json: JSON.stringify(AUDIT)});
  assert.deepEqual(JSON.parse(readFileSync(join(ws, 'audit.json'), 'utf8')), AUDIT);
  assert.throws(() => call('save_audit', {audit_json: JSON.stringify(AUDIT)}));
  writeFileSync(join(ws, 'inventory.csv'), 'changed');
  assert.throws(() => call('read_audit_csv', {table: 'inventory'}));
});

test('actual Harness confines disabled tools and honours explicit agents-home while preserving Files on', {timeout: 150000}, async t => {
  assert.ok(process.env.STRATA_HARNESS_DIR, 'Pinned Harness required; no skip');
  const {root, ws} = directory();
  const marker = join(root, 'sibling-marker.txt'); writeFileSync(marker, 'SYNTHETIC_SIBLING_MARKER', {flag: 'wx'});
  const worker = await startScriptedWorker({chunkDelayMs: 1});
  const globalAgents = join(root, '.agents'), isolatedAgents = join(root, 'isolated-agents');
  mkdirSync(globalAgents); mkdirSync(isolatedAgents);
  writeFileSync(join(globalAgents, 'AGENTS.md'), 'SYNTHETIC_DEFAULT_AGENTS_060', {flag: 'wx'});
  writeFileSync(join(isolatedAgents, 'AGENTS.md'), 'SYNTHETIC_OVERRIDE_AGENTS_060', {flag: 'wx'});
  const env = harnessFixtureEnvironment(root);
  env.DSH_AGENTS_HOME = isolatedAgents;
  const off = new Set(GROUP_IDS);
  const bridge = createAgentBridge({dataDir: root,
    localInstances: () => [{id: 'bounded', baseUrl: worker.url, modelId: 'scripted-model', context: 32768}],
    connections: () => [], env,
    tools: () => ({off, search: {provider: 'builtin', key: ''}, browserEnabled: false,
      servers: {ownedAudit: {command: process.execPath, args: [resolve('_docs/stable-060-20261116/real-model-mcp.mjs'), ws]}}}),
  });
  t.after(async () => { await bridge.close(); await worker.close(); });
  // Retain tiny synthetic attempts under the guarded TEMP for independent review.
  const allowed = new Set(['audit_inventory', 'mcp__ownedAudit__read_audit_csv', 'mcp__ownedAudit__save_audit',
    'list_mcp_resources', 'list_mcp_resource_templates', 'read_mcp_resource']);
  for (const [name, args] of [
    ['read', {file_path: marker}], ['run_code', {description: 'disabled', code: 'return "SYNTHETIC_SIBLING_MARKER";'}],
    ['subagent', {prompt: 'read the sibling marker'}], ['write', {file_path: marker, content: 'changed'}],
  ]) {
    const frames = [];
    await bridge.runTurn({chatId: 'denied-' + name, message: `SCRIPT:call ${name} ${JSON.stringify(args)}`, workspacePath: ws, mode: 'bypass', thinking: 'Off', onFrame: f => frames.push(f)});
    assert.equal(readFileSync(marker, 'utf8'), 'SYNTHETIC_SIBLING_MARKER');
    assert.equal(frames.some(f => f.type === 'agent-tool' && f.tool.name === name && f.tool.state === 'done'), false);
  }
  const frames = [];
  await bridge.runTurn({chatId: 'read-owned', message: 'SCRIPT:call mcp__ownedAudit__read_audit_csv {"table":"inventory"}', workspacePath: ws, mode: 'bypass', thinking: 'Off', onFrame: f => frames.push(f)});
  assert.ok(frames.some(f => f.type === 'agent-tool' && f.tool.name === 'mcp__ownedAudit__read_audit_csv' && f.tool.state === 'done'));
  assert.ok(worker.requests.length >= 5);
  assert.ok(worker.requests.every(row => row.toolNames.every(name => allowed.has(name))),
    'Unexpected tool names: ' + [...new Set(worker.requests.flatMap(row => row.toolNames).filter(name => !allowed.has(name)))].join(','));
  assert.ok(worker.requests.some(row => row.toolNames.includes('audit_inventory') && row.toolNames.includes('mcp__ownedAudit__save_audit')));
  assert.equal(existsSync(join(ws, 'audit.json')), false);
  const outside = join(root, 'sibling-instructions'); mkdirSync(outside);
  writeFileSync(join(outside, 'AGENTS.md'), 'SYNTHETIC_OUTSIDE_INSTRUCTION_060', {flag: 'wx'});
  const changed = [];
  await bridge.runTurn({chatId: 'cwd-discovery', message: `SCRIPT:call working_directory ${JSON.stringify({cd: outside})}`,
    workspacePath: ws, mode: 'manual', thinking: 'Off', onFrame: f => changed.push(f)});
  assert.equal(changed.some(f => f.type === 'agent-tool' && f.tool.name === 'working_directory' && f.tool.state === 'done'), false);
  assert.equal(worker.requests.some(row => row.raw.includes('SYNTHETIC_OUTSIDE_INSTRUCTION_060')), false);
  assert.equal(worker.requests.some(row => row.raw.includes('SYNTHETIC_DEFAULT_AGENTS_060')), false);
  assert.ok(worker.requests.some(row => row.raw.includes('SYNTHETIC_OVERRIDE_AGENTS_060')));
  const disabledNames = [...new Set(worker.requests.flatMap(row => row.toolNames))].sort();
  assert.ok(disabledNames.every(name => allowed.has(name)));
  worker.requests.length = 0;
  delete env.DSH_AGENTS_HOME; // A fresh runtime now uses the ordinary synthetic global home.
  off.delete('files');
  const normal = [];
  await bridge.runTurn({chatId: 'cwd-default', message: `SCRIPT:call working_directory ${JSON.stringify({cd: outside})}`,
    workspacePath: ws, mode: 'manual', thinking: 'Off', onFrame: f => normal.push(f)});
  assert.ok(normal.some(f => f.type === 'agent-tool' && f.tool.name === 'working_directory' && f.tool.state === 'done'));
  assert.ok(worker.requests.some(row => row.toolNames.includes('working_directory')));
  assert.ok(worker.requests.some(row => row.raw.includes('SYNTHETIC_OUTSIDE_INSTRUCTION_060')));
  assert.ok(worker.requests.some(row => row.raw.includes('SYNTHETIC_DEFAULT_AGENTS_060')));
  assert.equal(worker.requests.some(row => row.raw.includes('SYNTHETIC_OVERRIDE_AGENTS_060')), false);
  writeFileSync(join(root, 'capability-admission.json'), JSON.stringify({schema: 'strata.real-model.capability-admission.v1',
    syntheticOnly: true, arbitraryGroupedCallsSucceeded: false, groupedCalls: ['read', 'write', 'run_code', 'subagent'],
    toolNames: disabledNames, workingDirectoryChangedWhileDisabled: false, syntheticOutsideInstructionsReachedModelWhileDisabled: false,
    explicitAgentsOverrideObserved: true, defaultAgentsExcludedWhileOverridden: true,
    filesOnWorkingDirectoryObserved: true, absentOverrideDefaultAgentsObserved: true}), {flag: 'wx'});
});
