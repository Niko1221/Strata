/** Real shipped Harness with scripted inference: shared identity, tool effects, and detached observation. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { createConversations } from '../server/conversations.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
const skip = !process.env.STRATA_HARNESS_DIR || !resolveHarness().ok ? 'Set STRATA_HARNESS_DIR to a built Harness for real agent continuity checks.' : false;

test('Mac → phone preserves the real Harness session and tool effect; observation survives renderer detach', { skip, timeout: 120_000 }, async t => {
  const directory = mkdtempSync(join(tmpdir(), 'strata-conversation-agent-'));
  const workspace = join(directory, 'workspace'); mkdirSync(workspace);
  const worker = await startScriptedWorker({});
  let conversations;
  const observed = [];
  const bridge = createAgentBridge({ dataDir: directory, localInstances: () => [{ id: 'inst-1', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768 }], connections: () => [],
    onFrame: (chatId, frame) => { observed.push({ chatId, frame }); conversations?.record(chatId, frame); }, startTimeoutMs: 90_000 });
  conversations = createConversations({ directory, defaultWorkspace: () => workspace,
    execute: ({ session, message, mode, signal }) => bridge.runTurn({ chatId: session.id, message, workspacePath: session.workspace, mode, signal, onFrame: () => {} }),
  });
  t.after(async () => { await bridge.close(); await worker.close(); conversations.close(); rmSync(directory, { recursive: true, force: true }); });
  const { conversation } = conversations.create();
  conversations.begin(conversation.id, 'SCRIPT:tool marker.txt once');
  await bridge.runTurn({ chatId: conversation.id, message: 'SCRIPT:tool marker.txt once', workspacePath: workspace, mode: 'manual', onFrame: frame => {
    if (frame.type === 'agent-tool' && frame.tool.state === 'proposed') void bridge.approve({ chatId: conversation.id, callId: frame.tool.id, decision: 'approve' });
  } });
  const harnessSession = bridge.sessionOf(conversation.id);
  assert.ok(harnessSession);
  assert.equal(readFileSync(join(workspace, 'marker.txt'), 'utf8').trim(), 'once');
  assert.ok(conversations.detail(conversation.id).conversation.messages.at(-1).tools.some(t => t.state === 'done'));
  conversations.send(conversation.id, 'SCRIPT:echo Continued from phone');
  for (let n = 0; n < 600 && conversations.isRunning(conversation.id); n++) await new Promise(r => setTimeout(r, 25));
  assert.equal(conversations.isRunning(conversation.id), false);
  assert.equal(bridge.sessionOf(conversation.id), harnessSession);
  assert.equal(conversations.detail(conversation.id).conversation.messages.at(-1).content, 'Continued from phone');
  assert.equal(readFileSync(join(workspace, 'marker.txt'), 'utf8').trim(), 'once', 'a handoff must not replay the previous tool');
  const controller = new AbortController();
  conversations.begin(conversation.id, 'SCRIPT:slow 12');
  const result = await bridge.runTurn({ chatId: conversation.id, message: 'SCRIPT:slow 12', workspacePath: workspace, mode: 'manual', signal: controller.signal,
    onFrame: frame => { if (frame.type === 'delta') controller.abort(); },
  });
  if (result?.detached) await result.ended;
  assert.equal(conversations.isRunning(conversation.id), false);
  const reply = conversations.detail(conversation.id).conversation.messages.at(-1);
  assert.equal((reply.content.match(/w\d+ /g) ?? []).length, 12);
  assert.equal(observed.filter(item => item.chatId === conversation.id && item.frame.type === 'done').length, 3);
});
