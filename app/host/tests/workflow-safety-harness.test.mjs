/** Actual pinned Harness; scripted model tests capabilities, not model quality. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { harnessFixtureEnvironment } from './helpers/harness-environment.mjs';
const skip = !process.env.STRATA_HARNESS_DIR ? 'Needs the actual pinned Harness' : false;

test('actual Harness hides research writes, rejects direct attempts without approval, then restores ordinary tools', { skip, timeout: 150000 }, async t => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-research-safety-')), ws = join(dir, 'ws'); mkdirSync(ws);
  const worker = await startScriptedWorker({ chunkDelayMs: 1 });
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', baseUrl: worker.url, modelId: 'scripted-model', context: 65536 }], connections: () => [], env: harnessFixtureEnvironment(dir), searchMock: 'offline', startTimeoutMs: 90000 });
  t.after(async () => { await bridge.close(); await worker.close(); rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); });
  const frames = [], file = join(ws, 'must-not-exist.txt');
  await bridge.researchCommand({ chatId: 'research', line: `SCRIPT:call write ${JSON.stringify({ file_path: file, content: 'unwanted' })}`, deep: true, workspacePath: ws, mode: 'manual', thinking: 'Off', onFrame: f => frames.push(f) });
  assert.equal(existsSync(file), false);
  assert.equal(frames.some(f => f.type === 'agent-tool' && f.tool.state === 'proposed'), false);
  assert.equal(frames.some(f => f.type === 'error'), false, JSON.stringify(frames.filter(f => f.type === 'error')));
  const researchRequests = [...worker.requests];
  assert.ok(researchRequests.some(r => r.toolNames.includes('web_search')));
  for (const r of researchRequests) assert.ok(r.toolNames.every(n => ['web_search', 'web_fetch'].includes(n)), JSON.stringify(r.toolNames));
  assert.ok(researchRequests.some(r => !r.toolNames.length), 'final writing has no tools');
  worker.requests.length = 0;
  await bridge.runTurn({ chatId: 'research', message: 'SCRIPT:echo ordinary again', workspacePath: ws, mode: 'manual', onFrame: () => {} });
  assert.ok(worker.requests.some(r => r.toolNames.includes('write')), 'same chat regains ordinary tools after completion');
  const auditFrames = [];
  await bridge.runTurn({ chatId: 'audit', message: `SCRIPT:call audit_inventory ${JSON.stringify({ inventory_csv: 'id,version\na,1\n', expected_csv: 'id,expected_version\na,2\n', requirements_csv: 'id,required_check,status\na,test,passed\n' })}`, workspacePath: ws, mode: 'manual', onFrame: f => auditFrames.push(f) });
  const audit = auditFrames.find(f => f.type === 'agent-tool' && f.tool.name === 'audit_inventory' && f.tool.state === 'done');
  assert.ok(audit, JSON.stringify(auditFrames.filter(f => f.type === 'error')));
  assert.match(audit.tool.result, /"total_mismatches":1/);
  assert.match(audit.tool.result, /"total_failed":0/);
  const stoppedFrames = [];
  const running = bridge.researchCommand({ chatId: 'research-stop', line: 'SCRIPT:slow 4000', deep: true, workspacePath: ws, mode: 'bypass', thinking: 'Off', onFrame: f => stoppedFrames.push(f) });
  const deadline = Date.now() + 20000;
  while (!worker.requests.some(r => r.directive === 'slow')) { assert.ok(Date.now() < deadline, 'research request must start before cancellation'); await new Promise(resolve => setTimeout(resolve, 25)); }
  const oldSession = bridge.sessionOf('research-stop');
  assert.ok(oldSession);
  await bridge.stop('research-stop'); await running;
  assert.equal(bridge.sessionOf('research-stop'), null, 'cancelled research session remains restricted and is retired');
  await bridge.runTurn({ chatId: 'research-stop', message: 'SCRIPT:echo fresh after stop', workspacePath: ws, mode: 'manual', onFrame: () => {} });
  assert.notEqual(bridge.sessionOf('research-stop'), oldSession, 'a late call from the old session cannot inherit restored tools');
});
