// Strata Dev "contextGuard" in the real Harness with a scripted model server: a short message on a nearly full chat is summarised
// first, and a turn that ends fairly full is summarised on its own. The same chats with the guard off are left alone (as Live).
// Needs a built Harness (STRATA_HARNESS_DIR); without it these tests skip, loudly.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { FIT_TEXT } from '../server/agent/context-fit.mjs';
import { devSettings, setActiveDevSettings } from '../server/dev-mode.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
const dense = 'q=0x1f ok;';   // 10 characters, about 3.3 tokens by the scripted model server's count
const WINDOW = 32768;
const wait = (ms) => new Promise(r => setTimeout(r, ms));

/** A bridge on a scripted model server, with the guard set as `guard` (null: off, as Live). */
async function rig(t, guard) {
  setActiveDevSettings(guard ? devSettings({ contextGuard: { enabled: true, ...guard } }) : null);
  const worker = await startScriptedWorker({ window: WINDOW, charsPerToken: 3 });
  const dir = mkdtempSync(join(tmpdir(), 'strata-guard-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: WINDOW, slots: 2 }], connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { setActiveDevSettings(null); await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const send = async (message) => { const frames = []; await bridge.runTurn({ chatId: 'c', message, workspacePath: ws, mode: 'bypass', onFrame: f => frames.push(f) }); return frames; };
  return { worker, send };
}
const summaries = worker => worker.requests.filter(r => r.summary).length;
// The host's own line ("was X% and is now Y%"). The Harness reports its own summaries in other words, so this tells the two apart.
const noteOf = frames => frames.find(f => f.type === 'delta' && f.noteReason === 'compacted' && f.key === FIT_TEXT.summarized);
// 7,000 characters: about 2,300 tokens, echoed back once. Two of them take the chat to about 57 percent of the window; the Harness's own
// start point here is about 61 percent and it looks only before its next step.
const big = i => `SCRIPT:echo part${i} ${dense.repeat(700)}`;

// midTaskAt and the other two are set high so that only the part under test acts.
const BEFORE_SEND = { midTaskAt: 0.9, compactAfterTurnAt: 0.9, compactBeforeSendAt: 0.55 };
const AFTER_TURN = { midTaskAt: 0.9, compactAfterTurnAt: 0.55, compactBeforeSendAt: 0.9 };

test('before sending: a short message on a chat at 55 percent or more is summarised first, and the chat says so', { skip }, async (t) => {
  const { worker, send } = await rig(t, BEFORE_SEND);
  for (let i = 0; i < 2; i++) assert.equal((await send(big(i))).at(-1).type, 'done');
  assert.equal(summaries(worker), 0, 'nothing was summarised while the chat was filling');
  const before = worker.requests.length;
  const frames = await send('SCRIPT:echo hi');
  assert.equal(frames.at(-1).type, 'done');
  assert.ok(noteOf(frames), 'a line says the older messages were summarized');
  const later = worker.requests.slice(before);
  const firstAnswer = later.findIndex(r => !r.summary);
  assert.ok(later.slice(0, firstAnswer).some(r => r.summary), 'the summary was asked for before the short message was sent');
});

test('the same chat with the guard off: the host does not summarise before a short message, as in Live', { skip }, async (t) => {
  const { worker, send } = await rig(t, null);
  for (let i = 0; i < 2; i++) assert.equal((await send(big(i))).at(-1).type, 'done');
  const frames = await send('SCRIPT:echo hi');
  assert.equal(frames.at(-1).type, 'done');
  assert.equal(noteOf(frames), undefined, 'the host did not summarise before sending (the Harness may still do it by itself, as in Live)');
});

test('after a turn: a turn that ends at 55 percent or more is summarised at once, one that ends lower is not', { skip }, async (t) => {
  const { worker, send } = await rig(t, AFTER_TURN);
  assert.equal((await send(big(0))).at(-1).type, 'done');
  await wait(1500);
  assert.equal(summaries(worker), 0, 'a chat at about 50 percent is left alone');
  assert.equal((await send(big(1))).at(-1).type, 'done');
  const until = Date.now() + 30_000;
  while (summaries(worker) === 0 && Date.now() < until) await wait(250);
  assert.ok(summaries(worker) >= 1, 'the host summarised the chat after the turn, with no message sent');
});

test('the same two turns with the guard off: no summary afterwards', { skip }, async (t) => {
  const { worker, send } = await rig(t, null);
  for (let i = 0; i < 2; i++) assert.equal((await send(big(i))).at(-1).type, 'done');
  await wait(3000);
  assert.equal(summaries(worker), 0, 'Live behaviour: nothing summarised after the turn');
});

test('after a turn the chat is told about the summary at the start of its next message: was X percent, is now Y percent', { skip }, async (t) => {
  const { worker, send } = await rig(t, AFTER_TURN);
  for (let i = 0; i < 2; i++) assert.equal((await send(big(i))).at(-1).type, 'done');
  const until = Date.now() + 30_000;
  while (summaries(worker) === 0 && Date.now() < until) await wait(250);
  assert.ok(summaries(worker) >= 1, 'the host summarised the chat after the turn');
  await wait(1500);   // the summary is finished and measured before the chat is idle again
  const frames = await send('SCRIPT:echo hi');
  const note = noteOf(frames);
  assert.ok(note, 'the next message starts with the line about the summary');
  assert.equal(note.values.length, 2);
  assert.ok(note.values[0] > note.values[1], `the chat was fuller (${note.values[0]}%) than it is now (${note.values[1]}%)`);
});
