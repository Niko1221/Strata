/** Opt-in real-model check against an already loaded native instance. Never changes its configuration. */
import { mkdtempSync, mkdirSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import assert from 'node:assert/strict';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';

if (!process.env.STRATA_STEERING_MODEL_CONFIG || !process.env.STRATA_HARNESS_DIR) throw new Error('Set STRATA_STEERING_MODEL_CONFIG to an already loaded instance configuration and STRATA_HARNESS_DIR to the source-built steering component.');
const config = JSON.parse(readFileSync(process.env.STRATA_STEERING_MODEL_CONFIG, 'utf8'));
if (!Number.isInteger(config.port) || typeof config.api_key !== 'string') throw new Error('The selected instance has no running endpoint credentials.');
const modelId = process.env.STRATA_STEERING_MODEL_ID ?? config.aliases?.at(-1) ?? 'default';
const root = mkdtempSync(join(tmpdir(), 'strata-real-steering-')), workspace = join(root, 'workspace'); mkdirSync(workspace);
const local = [{ id: 'real-steering-test', label: 'Real local model', baseUrl: `http://127.0.0.1:${config.port}/v1`, modelId, context: config.max_context ?? 32768, key: config.api_key }];
const bridge = createAgentBridge({ dataDir: root, localInstances: () => local, connections: () => [], env: process.env });
let instruction, sent = false;
const frames = [];
const timer = setTimeout(() => { void bridge.stop('real-steering-test'); }, 180000);
try {
  await bridge.runTurn({ chatId: 'real-steering-test', workspacePath: workspace, mode: 'plan', message: 'Explain an implementation plan for a small Python text-search utility. Discuss cases and tradeoffs in detail. Do not use tools or change files.', onFrame(frame) {
    frames.push(frame);
    if (!sent && (frame.type === 'reasoning' || frame.type === 'delta') && frame.text) {
      sent = true;
      instruction = bridge.steer({ chatId: 'real-steering-test', id: 'real-model-correction', text: 'Change the task now: answer with exactly STEERING_OK and nothing else. Do not use tools.' });
      instruction.catch(() => {});
    }
  } });
  assert.ok(sent, 'the real model began generating before steering');
  await instruction;
  assert.equal(bridge.steeringState('real-steering-test')[0]?.state, 'delivered');
  assert.ok(frames.some(f => f.type === 'delta' && f.text.includes('STEERING_OK')) || frames.filter(f => f.type === 'delta').map(f => f.text).join('').includes('STEERING_OK'), 'the continued real response reflects the correction');
  const sessionId = bridge.sessionOf('real-steering-test'), runtime = bridge._debug().rt;
  const events = await new Promise((resolve, reject) => {
    const stream = runtime.gw.openStream('session/follow', { request: { address: { kind: 'session', sessionId } } }, { onItem(item) { if (item.type === 'snapshot') { stream.close(); resolve((item.records ?? []).filter(r => r.type === 'event').map(r => r.event)); } }, onError: reject });
  });
  assert.equal(events.filter(e => e.type === 'turn/start').length, 1);
  assert.equal(events.filter(e => e.type === 'turn/end').length, 1);
  assert.equal(events.filter(e => e.type === 'turn/end')[0].data.reason.kind, 'completed');
  assert.equal(events.filter(e => e.type === 'user/message' && e.data.source?.rpcId === 'real-model-correction').length, 1);
  console.log(JSON.stringify({ result: 'PASS', verification: 'real model through Harness Gateway and agent loop', model: modelId, turns: 1, delivered: 1, corrected: true }));
} finally {
  clearTimeout(timer); await bridge.close(); rmSync(root, { recursive: true, force: true });
}
