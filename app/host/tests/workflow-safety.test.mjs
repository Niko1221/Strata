import test from 'node:test';
import assert from 'node:assert/strict';
import { auditInventory } from '../server/agent/inventory-audit.mjs';
import { apply } from '../server/agent/plugins/strata-workflows.mjs';
import { createLocalMemoryGuard, admitLocalMemory, GiB } from '../server/local-memory-guard.mjs';

const fixture = {
  inventory_csv: 'id,item,version\r\ndelta,Engine,0.5.8\r\nepsilon,Control,0.5.6\r\nzeta,Terminal,0.5.8\r\n',
  expected_csv: 'id,expected_version\ndelta,0.5.8\nepsilon,0.5.8\nzeta,0.5.8\n',
  requirements_csv: 'id,required_check,status\ndelta,numerical_check,passed\nepsilon,signature_check,passed\nzeta,terminal_contract,passed\n',
};
test('actual failed audit fixture: passed checks never require invented evidence', () => {
  const result = auditInventory(fixture);
  assert.deepEqual(result.version_mismatches, [{ id: 'epsilon', inventory_version: '0.5.6', expected_version: '0.5.8' }]);
  assert.deepEqual([result.total_mismatches, result.total_duplicates, result.total_missing, result.total_failed], [1, 0, 0, 0]);
});
test('CSV parser handles BOM, quoted commas, escaped quotes, embedded newlines and exact version strings', () => {
  const result = auditInventory({ ...fixture, inventory_csv: '\uFEFFid,item,version\r\ndelta,"An ""engine"",\npart",0.5.8\r\nepsilon,Control,0.5.6\r\nzeta,Terminal,0.5.8' });
  assert.equal(result.total_mismatches, 1);
  assert.equal(auditInventory({ ...fixture, inventory_csv: fixture.inventory_csv.replace('0.5.6', '0.5.8') }).total_mismatches, 0);
});
test('duplicates are explicit; distinct required checks per ID are valid; missing and failed checks are separate', () => {
  const result = auditInventory({ ...fixture, inventory_csv: fixture.inventory_csv + 'epsilon,Other,0.5.8\n', requirements_csv: fixture.requirements_csv + 'delta,other_check,failed\nnew,extra,passed\n' });
  assert.equal(result.total_duplicates, 1);
  assert.equal(result.total_mismatches, 0, 'ambiguous duplicate IDs are not arbitrarily matched');
  assert.deepEqual(result.missing_ids, [{ id: 'new', missingFrom: ['inventory', 'expected'] }]);
  assert.equal(result.total_failed, 1);
});
test('malformed, truncated, missing-header and oversized CSVs fail without invented data', () => {
  for (const inventory_csv of ['id,version\nx,"unterminated', 'id,id,version\nx,x,1', 'id,version\nx,1,extra', 'id,version\n,1', 'x'.repeat(1048577), 'id,version\nx,"1"bad'])
    assert.throws(() => auditInventory({ ...fixture, inventory_csv }));
});

test('actual model read display retains CSV semantics and requires consecutive complete line labels', () => {
  const display = value => value.trimEnd().split(/\r?\n/).map((line, i) => (i + 1) + ': ' + line).join('\n') + '\n';
  const rendered = Object.fromEntries(Object.entries(fixture).map(([key, value]) => [key, display(value)]));
  assert.deepEqual(auditInventory(rendered), auditInventory(fixture));
  for (const inventory_csv of [rendered.inventory_csv.replace('2: ', '3: '), rendered.inventory_csv.replace('2: ', ''), rendered.inventory_csv.split('\n').slice(1).join('\n')])
    assert.throws(() => auditInventory({ ...fixture, inventory_csv }));
  const quoted = { ...fixture, inventory_csv: 'id,item,version\ndelta,"line one\n2: literal data",0.5.8\nepsilon,Control,0.5.6\nzeta,Terminal,0.5.8\n' };
  assert.deepEqual(auditInventory({ ...quoted, inventory_csv: display(quoted.inventory_csv) }), auditInventory(quoted));
});

function workflowHarness() {
  const hooks = new Map(), commands = new Map(), policies = new Map(), agents = new Map(); let guard;
  const ctx = { llm: { streamWithRegistration: value => value }, sessions: agents, on: (name, handler) => hooks.set(name, handler), tools: { guard: fn => { guard = fn; }, register: () => {} }, commands: { register: row => commands.set(row.name, row) } };
  apply(ctx);
  const agent = id => { const a = { session: { id, header: {} }, ctx: { inject: async (_names, fn) => fn({ tools: { restrict: ({ allow }) => { const record = { allow }; policies.set(id, record); return () => { if (policies.get(id) === record) policies.delete(id); }; } } }) } }; agents.set(id, a.session); return a; };
  return { agent, hooks, policies, guard: exec => guard(exec), phase: (a, rawInput) => commands.get('strata-research').handler({ agent: a, rawInput }) };
}
test('research capabilities are scoped, deny mutations before approvals even in Bypass, and restore after completion', async () => {
  const h = workflowHarness(), a = h.agent('research'), b = h.agent('ordinary');
  assert.equal((await h.phase(a, 'gather')).kind, 'success');
  assert.deepEqual(h.policies.get('research').allow, ['web_search', 'web_fetch']);
  let asked = false;
  for (const name of ['write', 'edit', 'pwsh', 'bash', 'run_code', 'subagent', 'ask_user_question', 'mcp__server__write', 'audit_inventory']) {
    const exec = { name, agent: a };
    assert.equal((await h.hooks.get('tools/pre-execute')(exec, () => { asked = true; return { kind: 'allow' }; })).kind, 'deny');
    assert.match(h.guard(exec), /Research only/);
    assert.equal(h.guard({ name, agent: b }), undefined);
  }
  assert.equal(asked, false);
  assert.equal(h.guard({ name: 'web_search', agent: a }), undefined);
  await h.phase(a, 'write');
  assert.deepEqual(h.policies.get('research').allow, []);
  assert.match(h.guard({ name: 'web_fetch', agent: a }), /Research only/);
  await h.phase(a, 'off');
  assert.equal(h.policies.size, 0);
  assert.equal(h.guard({ name: 'write', agent: a }), undefined);
});
test('research child cannot evade its parent policy and invalid phase preserves restrictions', async () => {
  const h = workflowHarness(), parent = h.agent('parent'), child = h.agent('child');
  child.session.header.parentSession = 'parent';
  await h.phase(parent, 'gather');
  assert.match(h.guard({ name: 'write', agent: child }), /Research only/);
  assert.equal((await h.phase(parent, 'invalid')).kind, 'error');
  assert.match(h.guard({ name: 'write', agent: parent }), /Research only/);
});
test('RAM guard stops ahead of the hard floor during loading or generation, once, and cancels its timer', async () => {
  let free = 10 * GiB, time = 0, stops = 0, cleared = false;
  const guard = createLocalMemoryGuard({ sample: () => free, now: () => time, onPressure: async () => { stops++; }, setTimer: () => 1, clearTimer: () => { cleared = true; } });
  guard.start();
  time = 1000; free = 8 * GiB;
  await guard.check(); // A fast decline predicts pressure even while above 5 GiB.
  assert.equal(stops, 1); assert.equal(cleared, true);
  assert.ok(guard.snapshot().pressure.availableBytes > guard.snapshot().pressure.floorBytes);
  await guard.check(); assert.equal(stops, 1);
});
test('RAM admission and steady low-memory pressure fail closed; disposal never stops a replacement', async () => {
  assert.throws(() => admitLocalMemory(4 * GiB), /RAM protection/);
  assert.throws(() => admitLocalMemory(NaN), /RAM protection/);
  admitLocalMemory(6 * GiB);
  assert.throws(() => admitLocalMemory(13 * GiB, { fullyOffloaded: true, modelBytes: 5.3 * GiB, mapped: true }), /staging allowance/);
  admitLocalMemory(13 * GiB, { fullyOffloaded: true, modelBytes: 5.3 * GiB, mapped: false });
  admitLocalMemory(13 * GiB, { fullyOffloaded: false, modelBytes: 100 * GiB, mapped: true });
  let stops = 0;
  const guard = createLocalMemoryGuard({ sample: () => 4.9 * GiB, onPressure: async () => { stops++; throw new Error('stop failed'); } });
  await guard.check(); assert.equal(stops, 1); assert.equal(guard.snapshot().stopError, 'stop failed');
  const disposed = createLocalMemoryGuard({ sample: () => 0, onPressure: () => { stops++; } });
  disposed.dispose(); await disposed.check(); assert.equal(stops, 1);
});
