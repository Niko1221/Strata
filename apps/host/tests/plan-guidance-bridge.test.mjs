// Strata Dev "planGuidance" in the real Harness with a scripted model server: the planning turn carries the rules and thinks at the
// configured level; with the guidance off the same turn is as Live. Needs a built Harness (STRATA_HARNESS_DIR); without it these skip.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { devSettings, setActiveDevSettings } from '../server/dev-mode.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

/** Runs `/plan` once (the chat's own thinking level is Low) and returns the one request the model server saw. */
async function planRequest(t, guidance) {
  setActiveDevSettings(guidance ? devSettings({ planGuidance: guidance }) : null);
  const worker = await startScriptedWorker({ window: 32768 });
  const dir = mkdtempSync(join(tmpdir(), 'strata-planguide-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }], connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { setActiveDevSettings(null); await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  await bridge.planCommand({ chatId: 'c', line: 'SCRIPT:echo build a page', message: '', workspacePath: ws, mode: 'bypass', thinking: 'Low', choice: null, onFrame: () => {} });
  return worker.requests.find(r => !r.summary);
}

test('planGuidance on: the planning turn carries the rules and thinks at the configured level', { skip }, async (t) => {
  const on = await planRequest(t, { enabled: true, thinking: 'Max' });
  assert.match(on.raw, /How to build a good plan/);
  assert.match(on.raw, /Write the code and its tests together/);
  assert.notEqual(JSON.stringify(on.params), JSON.stringify((await planRequest(t, null)).params), 'the thinking level differs from the chat\'s own Low');
});

test('planGuidance off: no rules in the planning prompt, as in Live', { skip }, async (t) => {
  const off = await planRequest(t, null);
  assert.doesNotMatch(off.raw, /How to build a good plan/);
  assert.match(off.raw, /Use 3 to 8 steps\./);
});
