import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';

const skip = !process.env.STRATA_HARNESS_DIR || !resolveHarness({ env: process.env }).ok;
async function until(check, ms = 20_000) { const end = Date.now() + ms; while (!check()) { if (Date.now() > end) throw new Error('condition timed out'); await new Promise(r => setTimeout(r, 50)); } }

test('real Harness: /btw starts a tool-free child, leaves a running main chat untouched, cancels only itself and reuses its isolated parent', { skip, timeout: 120_000 }, async t => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-btw-'));
  const worker = await startScriptedWorker({ chunkDelayMs: 30 });
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'local', modelId: 'scripted-model', baseUrl: worker.url, context: 32768, slots: 4 }], connections: () => [], env: process.env });
  let passed = false;
  t.after(async () => { await bridge.close(); await worker.close();
    if (!passed && existsSync(join(dir, 'agent/harness.log'))) console.error(readFileSync(join(dir, 'agent/harness.log'), 'utf8').split('\n').filter(line => /strata-btw-tools|Error|restrict|scoped|exception/i.test(line)).slice(-20).join('\n'));
    rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); });
  const mainFrames = [];
  const main = bridge.runTurn({ chatId: 'main', message: 'SCRIPT:slow 900', mode: 'plan', onFrame: f => mainFrames.push(f) });
  await until(() => worker.requests.some(r => r.directive === 'slow'));
  const mainId = bridge.sessionOf('main');
  const side = [];
  await bridge.btw({ message: 'SCRIPT:echo SIDE_ANSWER', history: [{ role: 'user', content: 'CONTEXT_MARKER' }], onFrame: f => side.push(f) });
  const address = side.find(f => f.type === 'btw-agent');
  assert.ok(address?.childSessionId && address.parentSessionId);
  assert.notEqual(address.childSessionId, address.parentSessionId);
  assert.notEqual(address.parentSessionId, mainId);
  assert.notEqual(address.childSessionId, mainId);
  assert.match(side.filter(f => f.type === 'delta').map(f => f.text).join(''), /SIDE_ANSWER/);
  assert.equal(side.at(-1).type, 'done');
  const childRequest = worker.requests.find(r => /CONTEXT_MARKER/.test(r.raw));
  assert.ok(childRequest, 'the child receives the frozen conversation');
  assert.deepEqual(childRequest.toolNames, [], 'the child has no tool schema, rather than only a no-tools prompt');
  assert.equal(childRequest.model, 'scripted-model');
  assert.equal(childRequest.params.chat_template_kwargs?.enable_thinking, false);
  assert.equal(bridge.isRunning('main'), true);
  assert.equal(bridge.sessionOf('main'), mainId);
  assert.equal(mainFrames.some(f => f.type === 'done'), false);
  const controller = new AbortController(), cancelled = [];
  const sideRun = bridge.btw({ message: 'SCRIPT:slow 900', signal: controller.signal, onFrame: f => cancelled.push(f) });
  await until(() => cancelled.some(f => f.type === 'delta'));
  controller.abort();
  await assert.rejects(sideRun);
  assert.equal(cancelled.find(f => f.type === 'btw-agent').parentSessionId, address.parentSessionId, 'a cleaned-up dedicated parent is reused');
  assert.equal(bridge.isRunning('main'), true, 'aborting the side child does not stop the main turn');
  const after = [];
  await bridge.btw({ message: 'SCRIPT:echo AFTER_STOP', onFrame: f => after.push(f) });
  assert.match(after.filter(f => f.type === 'delta').map(f => f.text).join(''), /AFTER_STOP/);
  assert.equal(worker.requests.filter(r => r.toolNames.length).length, 1, 'only the main chat has ordinary tools; no hidden parent inference');
  await bridge.stop('main'); await main;
  assert.equal(worker.requests.filter(r => r.toolNames.length).some(r => /SIDE_ANSWER|CONTEXT_MARKER|AFTER_STOP/.test(r.raw)), false);
  assert.equal(bridge.status().activeTurns, 0);
  passed = true;
});

test('/btw respects the disabled helpers setting before starting a runtime', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-btw-off-'));
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [], connections: () => [], tools: () => ({ off: new Set(['helpers']) }) });
  try { await assert.rejects(bridge.btw({ message: 'hi' }), /turned off/); assert.equal(bridge._debug().rt, null); }
  finally { await bridge.close(); rmSync(dir, { recursive: true, force: true }); }
});

test('real Harness: an API-only side parent delegates to the selected API instead of its creation default', { skip, timeout: 90_000 }, async t => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-btw-api-'));
  const worker = await startScriptedWorker({ model: 'cloud-model', chunkDelayMs: 5 });
  const connection = { id: 'btw-api-fixture', name: 'Chosen API', baseUrl: worker.url, protocol: 'openai-chat', apiKey: 'test-fixture-key', models: [{ id: 'cloud-model', name: 'Cloud fixture', contextMax: 32768, reasoning: false }] };
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [], connections: () => [connection], env: process.env });
  t.after(async () => { await bridge.close(); await worker.close(); rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); });
  const frames = [];
  await bridge.btw({ choice: { connectionId: connection.id, modelId: 'cloud-model' }, message: 'SCRIPT:echo CHOSEN_API_ANSWER', onFrame: f => frames.push(f) });
  assert.match(frames.filter(f => f.type === 'delta').map(f => f.text).join(''), /CHOSEN_API_ANSWER/);
  assert.equal(worker.requests.length, 1, 'no parent inference or fallback model call');
  assert.equal(worker.requests[0].model, 'cloud-model');
  assert.deepEqual(worker.requests[0].toolNames, []);
  assert.equal(frames.at(-1).type, 'done');
});
