/**
 * Helper agents (the Harness's subagents) in the chat: rows, concurrency, one approval card, Stop, and /goal.
 *
 * Pure tests always run; the rest need a built Harness (STRATA_HARNESS_DIR) and skip loudly without it.
 * The model is scripted; the Harness, its subagent runtime, its approval service, the gate and the bridge are real.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge, plainBlocker, goalEnding, plainGoalError } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { ChildProjector, ChildTranscript, activityOf } from '../server/agent/children.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
const cleanup = dir => { try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } };
const fwd = p => p.replaceAll('\\', '/');

// ---------------------------------------------------------------------------------------------------------- pure

const ev = (type, data, extra = {}) => ({ type: 'event', event: { seq: 1, time: extra.time ?? 1000, type, data } });

test('a helper row follows its helper: queued, working, completed with its closing words, tokens counted', () => {
  let t = 0;
  const cp = new ChildProjector({ id: 'c1', name: 'Check the tests', task: 'Run the unit tests and report', now: () => (t += 100) });
  assert.equal(cp.agent().status, 'queued');
  assert.equal(cp.agent().description, 'Waiting for its turn. Run the unit tests and report');
  cp.consume(ev('step/start', { turn: 1, step: 1 }));
  assert.equal(cp.agent().status, 'queued', 'a request that was sent is still waiting until the engine says its first word');
  cp.consume({ type: 'assistant-stream', frame: { type: 'chunk', chunk: { type: 'text-delta', text: 'Running' } } });
  assert.equal(cp.agent().status, 'working');
  assert.equal(cp.agent().description, 'Run the unit tests and report');
  cp.consume({ type: 'assistant-stream', frame: { type: 'chunk', chunk: { type: 'usage', usage: { inputTokens: 100, outputTokens: 20 } } } });
  assert.equal(cp.agent().tokens, 120);
  cp.consume(ev('assistant/message', { message: { content: [{ type: 'text', text: 'All 12 tests pass.' }] } }));
  cp.consume(ev('turn/end', { turn: 1, reason: { kind: 'completed' } }));
  const row = cp.agent();
  assert.equal(row.status, 'completed'); assert.equal(row.description, 'All 12 tests pass.');
  assert.ok(row.durationMs > 0);
  assert.ok(['id', 'name', 'description', 'status', 'durationMs', 'tokens', 'updatedAt'].every(k => k in row), 'exactly the fields the existing rows read');
});

test('a direct message replayed after a restart keeps one line', () => {
  const transcript = new ChildTranscript();
  transcript.beginUser('排队测试：只回复 QUEUED', 1000, 'req-1');
  transcript.user('排队测试：只回复 QUEUED', 1000, 'req-1'); // the Harness echo
  transcript.message('QUEUED', 1100);
  transcript.user('排队测试：只回复 QUEUED', 1000, 'req-1'); // the session replay after a restart
  const lines = transcript.entries.filter(entry => entry.role === 'user');
  assert.equal(lines.length, 1, 'one direct message is one line');
  assert.equal(lines[0].sourceId, 'req-1');
});

test('a finished helper row summarizes its answer as one plain line, not raw Markdown', () => {
  const heading = new ChildProjector({ id: 'm1', name: 'A' });
  heading.consume(ev('step/start', {}));
  heading.consume(ev('assistant/message', { message: { content: [{ type: 'text', text: '## 秋天的清晨\n\n秋天的清晨是从一层薄雾开始的。' }] } }));
  heading.consume(ev('turn/end', { reason: { kind: 'completed' } }));
  assert.equal(heading.agent().description, '秋天的清晨');

  const marked = new ChildProjector({ id: 'm2', name: 'B' });
  marked.consume(ev('step/start', {}));
  marked.consume(ev('assistant/message', { message: { content: [{ type: 'text', text: '**写入完成**，文件 `C:\\a.txt` 已校验。\n\n第二段不该出现在行里。' }] } }));
  marked.consume(ev('turn/end', { reason: { kind: 'completed' } }));
  assert.equal(marked.agent().description, '写入完成，文件 C:\\a.txt 已校验。');
});

test('a helper that failed or was stopped says so in plain words', () => {
  const failed = new ChildProjector({ id: 'f', name: 'A' });
  failed.consume(ev('step/start', {}));
  failed.consume(ev('turn/end', { reason: { kind: 'error', error: { message: 'fetch failed: ECONNREFUSED' } } }));
  assert.equal(failed.agent().status, 'failed'); assert.match(failed.agent().description, /not reachable/);
  const stopped = new ChildProjector({ id: 's', name: 'B' });
  stopped.consume(ev('turn/end', { reason: { kind: 'aborted' } }));
  assert.deepEqual([stopped.agent().status, stopped.agent().description], ['cancelled', 'Stopped.']);
});

test('only a helper request that needs the person becomes a card, and its id names the helper', () => {
  const cp = new ChildProjector({ id: 'child-1', name: 'Writer' });
  assert.deepEqual(cp.consume(ev('tool/call', { callId: 'k1', name: 'read', arguments: '{"file_path":"a"}' })).frames, [], 'a read is not a card');
  cp.consume(ev('tool/result', { message: { toolCallId: 'k1', content: [{ type: 'text', text: 'x' }] } }));
  cp.consume(ev('tool/call', { callId: 'k2', name: 'write', arguments: '{"file_path":"C:/a.txt","content":"x"}' }));
  const asked = cp.consume(ev('approval/asked', { id: 'a1', callId: 'k2', reason: 'Create or replace the file C:/a.txt' })).frames[0].tool;
  assert.equal(asked.id, 'child-1:k2'); assert.equal(asked.state, 'proposed'); assert.match(asked.effect, /^Writer asks: Create or replace the file/);
  assert.equal(cp.consume(ev('approval/decided', { id: 'a1', outcome: 'rejected' })).frames[0].tool.result, 'Denied. Nothing ran.');
  const done = cp.consume(ev('tool/result', { message: { toolCallId: 'k2', isError: true, content: [{ type: 'text', text: 'Error: the user rejected tool "write"' } ] } })).frames[0].tool;
  assert.equal(done.state, 'denied'); assert.equal(done.result, 'Denied. Nothing ran.');
  assert.equal(activityOf('pwsh'), 'Running a command');
});

test('goal wording: plain, never a code', () => {
  assert.match(plainBlocker({ code: 'round-limit' }), /most steps/);
  assert.match(goalEnding({ phase: 'complete' }), /complete/);
  assert.match(plainGoalError(new Error('goal "x" is not finished; an unfinished goal must be edited')), /already has a goal/);
  assert.doesNotMatch(plainGoalError(new Error('boom')), /boom/);
});

// ------------------------------------------------------------------------------------------------- real Harness

async function rig(t, { slots = 4, serial = false, onGoal } = {}) {
  const worker = await startScriptedWorker({ serial });
  const dir = mkdtempSync(join(tmpdir(), 'strata-kids-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const local = [{ id: 'inst-1', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], env: process.env, startTimeoutMs: 90_000, ...(onGoal ? { onGoal } : {}) });
  t.after(async () => { await bridge.close(); await worker.close(); cleanup(dir); });
  const run = async (chatId, message, { mode = 'bypass', answer, onFrame, goal } = {}) => {
    const frames = [];
    const on = f => {
      frames.push({ ...f, at: Date.now() }); onFrame?.(f);
      if (answer && f.type === 'agent-tool' && f.tool.state === 'proposed') void bridge.approve({ chatId, callId: f.tool.id, decision: answer }).catch(e => frames.push({ type: 'test-error', message: e.message }));
    };
    const result = goal ? await bridge.runTurn({ chatId, workspacePath: ws, mode, goal, onFrame: on }) : await bridge.runTurn({ chatId, message, workspacePath: ws, mode, onFrame: on });
    return { frames, result };
  };
  return { worker, bridge, dir, ws, run };
}
const rows = frames => { const m = new Map(); for (const f of frames.filter(x => x.type === 'agent')) m.set(f.agent.id, [...(m.get(f.agent.id) ?? []), f.agent]); return m; };
const text = frames => frames.filter(f => f.type === 'delta').map(f => f.text).join('');

test('two helpers run at the same time as two rows, the run ends when both have reported, and the delegation tool shows no raw card', { skip }, async (t) => {
  const r = await rig(t);
  const { frames } = await r.run('kids', 'SCRIPT:spawn 2 slow 40');
  const byId = rows(frames);
  assert.equal(byId.size, 2);
  const names = [...byId.values()].map(history => history.at(-1).name).sort();
  assert.deepEqual(names, ['Helper 1', 'Helper 2']);
  for (const history of byId.values()) {
    assert.ok(history.some(a => a.status === 'working'), 'it worked');
    assert.equal(history.at(-1).status, 'completed');
    assert.ok(history.at(-1).tokens > 0 && history.at(-1).durationMs > 0);
    assert.match(history[1]?.description ?? history[0].description, /SCRIPT:slow 40/, 'the row names its task');
  }
  // concurrent: both were working before either finished (by the order the frames arrived in, not by the clock: a loaded computer delays a frame, it does not reorder it)
  const agentFrames = frames.filter(f => f.type === 'agent');
  const firstDone = agentFrames.findIndex(f => f.agent.status === 'completed');
  for (const id of byId.keys()) {
    const working = agentFrames.findIndex(f => f.agent.id === id && f.agent.status === 'working');
    assert.ok(working >= 0 && working < firstDone, `helper ${id} was working before the first helper finished`);
  }
  assert.equal(frames.filter(f => f.type === 'agent-tool').length, 0, 'no card for the delegation tool itself');
  assert.match(text(frames), /Noted/, 'the parent answered after the helpers reported back');
  assert.equal(frames.at(-1).type, 'done');
  assert.ok(frames.at(-1).at >= Math.max(...frames.filter(f => f.type === 'agent' && f.agent.status === 'completed').map(f => f.at)), 'done comes after the last helper');
  assert.equal(r.bridge.isRunning('kids'), false);
  assert.equal(r.worker.requests.filter(q => q.directive === 'slow').length, 2);
});

test('a helper conversation is kept for its chat after the run, and cannot be read from another chat', { skip }, async (t) => {
  const r = await rig(t);
  const { frames } = await r.run('keep-talk', 'SCRIPT:spawn 1 echo hi there');
  const id = [...rows(frames).keys()][0];
  const view = r.bridge.childView('keep-talk', id);
  assert.ok(view, 'the helper is still readable after its run');
  assert.equal(view.agent.status, 'completed');
  assert.equal(view.available, true);
  assert.ok(view.transcript.some(entry => entry.role === 'user' && /hi there/.test(entry.text)), 'the task is in the conversation');
  assert.ok(view.transcript.some(entry => entry.role === 'assistant' && /hi there/.test(entry.text)), 'the helper answer is in the conversation');
  assert.equal(r.bridge.childView('another-chat', id), null, 'another chat cannot read it');
  // The person can send this helper another message without starting a new chat turn.
  const sent = await r.bridge.promptChild({ chatId: 'keep-talk', childId: id, text: 'SCRIPT:echo second' });
  assert.ok(sent.messageId, 'the Harness accepted the message');
  for (let i = 0; i < 100; i += 1) {
    const now = r.bridge.childView('keep-talk', id);
    if (now.transcript.some(entry => entry.role === 'assistant' && /second/.test(entry.text))) break;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  const after = r.bridge.childView('keep-talk', id);
  assert.ok(after.transcript.some(entry => entry.role === 'assistant' && /second/.test(entry.text)), 'the helper answered the new message');
  assert.equal(after.agent.status, 'completed');
  assert.equal(after.transcript.filter(entry => entry.role === 'user' && /SCRIPT:echo second/.test(entry.text)).length, 1, 'the sent message is not duplicated by its Harness echo');
});

test('a timed-out client retry with the same client id delivers one helper message, not two', { skip }, async (t) => {
  const r = await rig(t);
  const { frames } = await r.run('dup-send', 'SCRIPT:spawn 1 echo first');
  const id = [...rows(frames).keys()][0];
  const first = await r.bridge.promptChild({ chatId: 'dup-send', childId: id, text: 'SCRIPT:echo once', clientId: 'same-client' });
  const retry = await r.bridge.promptChild({ chatId: 'dup-send', childId: id, text: 'SCRIPT:echo once', clientId: 'same-client' });
  assert.equal(retry.messageId, first.messageId, 'the retry joins the accepted request');
  for (let i = 0; i < 100; i += 1) {
    const now = r.bridge.childView('dup-send', id);
    if (now.transcript.some(entry => entry.role === 'assistant' && /once/.test(entry.text))) break;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  const view = r.bridge.childView('dup-send', id);
  assert.equal(view.transcript.filter(entry => entry.role === 'user' && /once/.test(entry.text)).length, 1, 'one user line');
  assert.equal(view.transcript.filter(entry => entry.role === 'assistant' && /once/.test(entry.text)).length, 1, 'one answer');
});

test('helpers are limited to the Harness cap (4, or the slot count if larger); the fifth is refused with a message the model reads', { skip }, async (t) => {
  const r = await rig(t, { slots: 2 });
  const { frames } = await r.run('few', 'SCRIPT:spawn 5 slow 30');
  assert.equal(rows(frames).size, 4);
});

test('Manual: a helper that wants to change a file reaches the person as the same single Approve card, and the answer is honoured', { skip }, async (t) => {
  const r = await rig(t);
  const approved = join(r.ws, 'from-helper.txt'), denied = join(r.ws, 'denied.txt');
  const ok = await r.run('kid-ok', `SCRIPT:spawn 1 call write ${JSON.stringify({ file_path: fwd(approved), content: 'by helper' })}`, { mode: 'manual', answer: 'approve' });
  const okCards = ok.frames.filter(f => f.type === 'agent-tool').map(f => f.tool);
  assert.deepEqual([...new Set(okCards.map(c => c.state))], ['proposed', 'running', 'done']);
  assert.equal(new Set(okCards.map(c => c.id)).size, 1); assert.match(okCards[0].id, /^[0-9a-f-]{36}:/);
  assert.match(okCards[0].effect, /^Helper 1 asks: Create or replace the file/);
  assert.equal(readFileSync(approved, 'utf8').trim(), 'by helper');
  const no = await r.run('kid-no', `SCRIPT:spawn 1 call write ${JSON.stringify({ file_path: fwd(denied), content: 'x' })}`, { mode: 'manual', answer: 'deny' });
  assert.equal(existsSync(denied), false);
  assert.equal(no.frames.filter(f => f.type === 'agent-tool').at(-1).tool.state, 'denied');
  assert.equal(no.frames.filter(f => f.type === 'agent-tool').at(-1).tool.result, 'Denied. Nothing ran.');
  assert.equal(rows(no.frames).size, 1);
});

test("a chat cannot answer another chat's helper question", { skip }, async (t) => {
  const r = await rig(t);
  let asked = null;
  const file = join(r.ws, 'x.txt');
  const run = r.run('owner', `SCRIPT:spawn 1 call write ${JSON.stringify({ file_path: fwd(file), content: 'x' })}`, { mode: 'manual', onFrame: f => { if (f.type === 'agent-tool' && f.tool.state === 'proposed') asked = f.tool; } });
  for (let i = 0; i < 200 && !asked; i++) await new Promise(res => setTimeout(res, 50));
  assert.ok(asked, 'the question reached the person');
  await assert.rejects(r.bridge.approve({ chatId: 'someone-else', callId: asked.id, decision: 'approve' }), /no longer waiting/);
  assert.equal(r.bridge.pendingFor('owner').length, 1);
  await r.bridge.approve({ chatId: 'owner', callId: asked.id, decision: 'deny' });
  await run;
  assert.equal(existsSync(file), false);
});

test('Stop on the chat stops its helpers, ends the run quickly and starts no further turn', { skip }, async (t) => {
  const r = await rig(t);
  const started = Date.now();
  let stopped = null;
  const { frames } = await r.run('stopper', 'SCRIPT:spawn 2 slow 2000', { onFrame: f => {
    if (f.type === 'agent' && f.agent.status === 'working' && !stopped) stopped = new Promise(res => setTimeout(res, 300)).then(() => r.bridge.stop('stopper'));
  } });
  const result = await stopped;
  assert.equal(result.stopped, true);
  const final = [...rows(frames).values()].map(h => h.at(-1).status);
  assert.deepEqual(final, ['cancelled', 'cancelled']);
  assert.ok(Date.now() - started < 40_000, 'seconds, not the 80 s the helpers would have taken (a generous bound: a loaded computer is slow, never that slow)');
  const done = frames.at(-1);
  assert.equal(done.type, 'done'); assert.equal(done.cancelled, true);
  assert.equal(r.bridge.isRunning('stopper'), false);
  assert.ok(!/Noted/.test(text(frames)), 'the parent was not woken to comment on stopped helpers');
});

test('a helper stopped on request reads "interrupted" and can be asked to go on', { skip }, async (t) => {
  const r = await rig(t);
  let stopping = null;
  const { frames } = await r.run('resume-helper', 'SCRIPT:spawn 1 slow 3000', { onFrame: f => {
    if (stopping || f.type !== 'agent' || f.agent.status !== 'working') return;
    stopping = r.bridge.stopChildren({ chatId: 'resume-helper', childId: f.agent.id }).then(answer => ({ id: f.agent.id, answer }));
  } });
  assert.ok(stopping, 'the helper reached working');
  const { id, answer } = await stopping;
  assert.equal(answer.stopped, 1);
  const stopped = r.bridge.childView('resume-helper', id);
  assert.equal(stopped.agent.status, 'interrupted');
  await r.bridge.promptChild({ chatId: 'resume-helper', childId: id, text: 'SCRIPT:echo again' });
  // Wait for the helper's terminal status, not merely its first answer words. The old loop broke on the
  // assistant/message event while the turn was still finishing, so a loaded machine made it read "working".
  let resumed = r.bridge.childView('resume-helper', id);
  for (let i = 0; i < 200; i += 1) {
    resumed = r.bridge.childView('resume-helper', id);
    const answered = resumed.transcript.some(entry => entry.role === 'assistant' && /again/.test(entry.text));
    if (answered && resumed.agent.status === 'completed') break;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  assert.ok(resumed.transcript.some(entry => entry.role === 'assistant' && /again/.test(entry.text)), 'the stopped helper went on');
  assert.equal(resumed.agent.status, 'completed');
  assert.equal(frames.at(-1).type, 'done');
});

test('a run that ends while a helper is still working seals its row instead of leaving it running', { skip }, async (t) => {
  const r = await rig(t);
  let finished = false;
  const { frames } = await r.run('seal-early', 'SCRIPT:spawn 1 slow 5000', { onFrame: f => {
    if (finished || f.type !== 'agent' || f.agent.status !== 'working') return;
    finished = true;
    const turn = r.bridge._debug().chats.get('seal-early')?.turn;
    assert.ok(turn, 'the run is still open while the helper works');
    turn.finish(null);
  } });
  assert.equal(finished, true, 'the helper reached working');
  const history = [...rows(frames).values()].at(-1);
  assert.equal(history.at(-1).status, 'interrupted');
  assert.ok(history.at(-1).description, 'the row keeps a description of where it stopped');
  assert.equal(r.bridge.isRunning('seal-early'), false);
  const snapshot = r.bridge.agentSnapshot('seal-early');
  assert.equal(snapshot.running, false);
  assert.equal(snapshot.agents.at(-1).status, 'interrupted');
  assert.equal(snapshot.agents.at(-1).live, false);
});

test('/goal: sets a goal, runs its rounds in the chat, says how it ended, refuses a second unfinished goal, and Stop pauses it', { skip }, async (t) => {
  const r = await rig(t);
  const frames = [];
  const say = line => { const out = []; return r.bridge.goalCommand({ chatId: 'g', line, workspacePath: r.ws, mode: 'bypass', onFrame: f => { out.push(f); frames.push(f); } }).then(() => out); };
  assert.match(text(await say('')), /No goal is set/);
  const ran = await r.run('g', null, { goal: { action: 'create', objective: 'say hello a few times', maxRounds: 3 } });
  // The round is a live status now (it used to be an italic line of text that stayed on screen after the model moved on).
  assert.ok(!/Working on the goal/.test(text(ran.frames)), 'no goal-step text in the message');
  assert.ok(ran.frames.some(f => f.type === 'status' && f.phase === 'goal' && f.code === 'goal-step' && /^Working on the goal…$/.test(f.text) && f.args.round >= 1), 'a goal-step status was sent');
  const goalFrames = ran.frames.filter(f => f.type === 'goal');
  assert.ok(goalFrames.length >= 2 && goalFrames[0].goal.objective === 'say hello a few times' && goalFrames[0].goal.maxRounds === 3, 'the goal frame carries the objective and the round limit');
  assert.equal(goalFrames.at(-1).goal.phase, 'blocked'); assert.match(goalFrames.at(-1).goal.blocked, /most steps/);
  assert.equal(ran.frames.filter(f => f.type === 'status').at(-1).phase, 'idle', 'the status line is cleared before the end');
  assert.ok(ran.frames.findIndex(f => f.type === 'status' && f.phase === 'idle') < ran.frames.findIndex(f => f.type === 'done'));
  assert.match(text(ran.frames), /The goal stopped: it worked for the most steps/);
  assert.equal(ran.frames.at(-1).type, 'done');
  assert.match(text(await say('')), /stopped/);
  assert.match((await say('a different goal')).find(f => f.type === 'error').message, /already has a goal/);
  assert.match(text(await say('clear')), /cleared/);
  // a long goal is paused by Stop, and can be resumed
  const slowGoal = r.run('g2', null, { goal: { action: 'create', objective: 'keep going', maxRounds: 50 } });
  r.worker.requests.length = 0;
  await new Promise(res => setTimeout(res, 3500));
  const stop = await r.bridge.stop('g2');
  assert.equal(stop.stopped, true);
  await slowGoal;
  const before = r.worker.requests.length;
  await new Promise(res => setTimeout(res, 1500));
  assert.equal(r.worker.requests.length, before, 'no further round starts after Stop');
  assert.match(text(await r.bridge.goalCommand({ chatId: 'g2', line: '', workspacePath: r.ws, mode: 'bypass', onFrame: f => frames.push(f) }).then(() => frames.filter(f => f.type === 'delta').slice(-1))), /paused/);
});

test('one request slot: the second helper shows as waiting for its turn, then works, and the run still ends cleanly', { skip }, async (t) => {
  const r = await rig(t, { slots: 1, serial: true });
  const { frames } = await r.run('queue', 'SCRIPT:spawn 2 slow 25');
  const byId = rows(frames);
  assert.equal(byId.size, 2, 'both helpers exist (they are not refused)');
  const histories = [...byId.values()];
  // walk the frames in order: at some moment one helper was working while the other was waiting for its turn
  const now = new Map(); let overlap = null;
  for (const f of frames.filter(x => x.type === 'agent')) {
    now.set(f.agent.id, f.agent);
    const all = [...now.values()];
    const waiting = all.find(a => a.status === 'queued' && a.description.startsWith('Waiting for its turn.'));
    if (waiting && all.some(a => a.status === 'working')) overlap ??= waiting;
  }
  assert.ok(overlap, 'one helper visibly waited for its turn while the other worked');
  for (const h of histories) assert.equal(h.at(-1).status, 'completed');
  assert.equal(frames.at(-1).type, 'done');
});

test('goal bar: the state, pause, edit and remove work from outside a turn, and every change is announced', { skip }, async (t) => {
  const announced = [];
  const r = await rig(t, { onGoal: (chatId, goal) => announced.push({ chatId, goal }) });
  assert.equal(await r.bridge.goalState('bar'), null, 'no goal in a chat that never had one');
  await assert.rejects(r.bridge.goalControl({ chatId: 'bar', action: 'pause' }), /no goal/i);
  await r.run('bar', null, { goal: { action: 'create', objective: 'say hello a few times', maxRounds: 1 } });
  const shown = await r.bridge.goalState('bar');
  assert.equal(shown.objective, 'say hello a few times'); assert.equal(shown.maxRounds, 1); assert.ok(['blocked', 'complete', 'paused', 'active'].includes(shown.phase));
  await assert.rejects(r.bridge.goalControl({ chatId: 'bar', action: 'edit', text: '  ' }), /few words/);
  const edited = await r.bridge.goalControl({ chatId: 'bar', action: 'edit', text: 'say goodbye instead' });
  assert.equal(edited.goal.objective, 'say goodbye instead');
  assert.equal(announced.at(-1).goal.objective, 'say goodbye instead', 'the edit was announced for the bar');
  const cleared = await r.bridge.goalControl({ chatId: 'bar', action: 'clear' });
  assert.deepEqual(cleared, { goal: null }); assert.equal(announced.at(-1).goal, null);
  assert.equal(await r.bridge.goalState('bar'), null);
  await assert.rejects(r.bridge.goalControl({ chatId: 'bar', action: 'explode' }), /no goal|not known/i);
});

test('mode change mid-run: a helper keeps the Manual mode it started with; switching the parent does not answer its question', { skip }, async (t) => {
  const r = await rig(t);
  const first = join(r.ws, 'before.txt');
  let asked = null;
  const run = r.run('mid', `SCRIPT:spawn 1 call write ${JSON.stringify({ file_path: fwd(first), content: 'one' })}`, { mode: 'manual', onFrame: f => { if (f.type === 'agent-tool' && f.tool.state === 'proposed') asked = f.tool; } });
  for (let i = 0; i < 200 && !asked; i++) await new Promise(res => setTimeout(res, 50));
  assert.ok(asked, 'Manual asked first');
  const switched = await r.bridge.setMode('mid', 'bypass');
  assert.deepEqual({ applied: switched.applied, mode: switched.mode }, { applied: true, mode: 'bypass' });
  assert.equal(switched.settled, 0, 'an existing helper keeps its own question');
  assert.equal(r.bridge.pendingFor('mid').length, 1, 'the helper is still waiting for the person');
  await r.bridge.approve({ chatId: 'mid', callId: asked.id, decision: 'approve' });
  await run;
  assert.equal(readFileSync(first, 'utf8').trim(), 'one');
  assert.equal(r.bridge.pendingFor('mid').length, 0);
  // A later parent switch still does not rewrite the finished helper's recorded mode.
  assert.equal((await r.bridge.setMode('mid', 'plan')).mode, 'plan');
  assert.deepEqual(await r.bridge.setMode('nobody-yet', 'manual'), { applied: false, mode: 'manual', reason: 'no session yet' });
  const view = r.bridge.childView('mid', asked.id.split(':')[0]);
  assert.equal(view.agent.mode, 'manual');
});

test('a Bypass helper keeps Bypass for a direct message after the parent switches to Plan', { skip }, async (t) => {
  const r = await rig(t);
  const file = join(r.ws, 'bypass-kept.txt');
  let childId = null;
  const run = r.run('keep-bypass', 'SCRIPT:spawn 1 slow 20', { mode: 'bypass', onFrame: f => {
    if (f.type === 'agent' && f.agent.status === 'working') childId = f.agent.id;
  } });
  for (let i = 0; i < 200 && !childId; i += 1) await new Promise(resolve => setTimeout(resolve, 50));
  assert.ok(childId, 'the helper started');
  assert.equal((await r.bridge.setMode('keep-bypass', 'plan')).mode, 'plan');
  await r.bridge.promptChild({ chatId: 'keep-bypass', childId, text: `SCRIPT:call write ${JSON.stringify({ file_path: fwd(file), content: 'still bypass' })}` });
  for (let i = 0; i < 300 && !existsSync(file); i += 1) await new Promise(resolve => setTimeout(resolve, 100));
  await run;
  assert.equal(existsSync(file), true, 'the pinned Bypass helper wrote without asking');
  assert.equal(readFileSync(file, 'utf8').trim(), 'still bypass');
  assert.equal(r.bridge.childView('keep-bypass', childId).agent.mode, 'bypass');
});

test('a Plan helper stays read-only for a direct message after the parent switches to Bypass', { skip }, async (t) => {
  const r = await rig(t);
  const first = join(r.ws, 'plan-denied.txt');
  const second = join(r.ws, 'plan-still-denied.txt');
  const firstRun = await r.run('keep-plan', `SCRIPT:spawn 1 call write ${JSON.stringify({ file_path: fwd(first), content: 'no' })}`, { mode: 'plan' });
  const childId = [...rows(firstRun.frames).keys()][0];
  assert.ok(childId, 'the Plan helper started');
  assert.equal(existsSync(first), false, 'the first Plan write did not run');
  assert.equal(r.bridge.childView('keep-plan', childId).agent.mode, 'plan');
  assert.equal((await r.bridge.setMode('keep-plan', 'bypass')).mode, 'bypass');
  await r.bridge.promptChild({ chatId: 'keep-plan', childId, text: `SCRIPT:call write ${JSON.stringify({ file_path: fwd(second), content: 'no' })}` });
  for (let i = 0; i < 100; i += 1) {
    const view = r.bridge.childView('keep-plan', childId);
    if (view.agent.status === 'completed' && view.transcript.some(entry => entry.role === 'assistant' && /Tool finished/.test(entry.text))) break;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  assert.equal(existsSync(second), false, 'the parent switch did not widen the Plan helper');
  assert.equal(r.bridge.childView('keep-plan', childId).agent.mode, 'plan');
});

test('a direct message sent while the helper works is queued and runs after that turn', { skip }, async (t) => {
  const r = await rig(t);
  let childId = null;
  const run = r.run('queued-talk', 'SCRIPT:spawn 1 slow 100', { mode: 'bypass', onFrame: f => {
    if (!childId && f.type === 'agent' && f.agent.status === 'working') childId = f.agent.id;
  } });
  for (let i = 0; i < 200 && !childId; i += 1) await new Promise(resolve => setTimeout(resolve, 50));
  assert.ok(childId, 'the helper reached working');
  const sent = await r.bridge.promptChild({ chatId: 'queued-talk', childId, text: 'SCRIPT:echo later' });
  assert.ok(sent.messageId, 'the Harness accepted the queued message');
  let seen = false;
  for (let i = 0; i < 300; i += 1) {
    const view = r.bridge.childView('queued-talk', childId);
    if (view.transcript.some(entry => entry.role === 'assistant' && /later/.test(entry.text))) { seen = true; break; }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  await run;
  assert.equal(seen, true, 'the queued message ran after the first turn');
});

test('a Manual helper stopped before any tool does not make the next Plan helper ask', { skip }, async (t) => {
  const r = await rig(t);
  const file = join(r.ws, 'plan-after-stop.txt');
  let stopping = null;
  const first = await r.run('stale-mode', 'SCRIPT:spawn 1 slow 4000', { mode: 'manual', onFrame: f => {
    if (stopping || f.type !== 'agent' || f.agent.status !== 'working') return;
    stopping = r.bridge.stopChildren({ chatId: 'stale-mode', childId: f.agent.id });
  } });
  assert.ok(stopping, 'the Manual helper reached working');
  await stopping;
  await first;
  assert.equal((await r.bridge.setMode('stale-mode', 'plan')).mode, 'plan');
  const cards = [];
  await r.run('stale-mode', `SCRIPT:spawn 1 call write ${JSON.stringify({ file_path: fwd(file), content: 'no' })}`, {
    mode: 'plan',
    onFrame: f => {
      if (f.type !== 'agent-tool' || f.tool.state !== 'proposed') return;
      cards.push(f);
      // Answer the card so a regression ends instead of waiting forever; the assertion still fails.
      void r.bridge.approve({ chatId: 'stale-mode', callId: f.tool.id, decision: 'deny' }).catch(() => {});
    },
  });
  assert.equal(existsSync(file), false, 'the Plan helper did not write');
  assert.equal(cards.length, 0, 'the Plan helper was refused instead of asking for approval');
});

test('the gate pins a helper to the mode its own log records, not the parent\'s current mode', async () => {
  const { apply, MANUAL_PRESET, PLAN_PRESET } = await import('../server/agent/plugins/strata-gate.mjs');
  let handler; const appended = [];
  const parent = { header: { origin: 'user' } };
  let childPreset = MANUAL_PRESET;
  const child = {
    header: { origin: 'subagent', parentSession: 'p' },
    ownEvents: () => childPreset ? [{ type: 'permission/preset', data: { preset: childPreset } }] : [],
    append: (type, data) => appended.push([type, data.policy]),
  };
  let parentPreset = MANUAL_PRESET; let override = 'never';
  const sessions = new Map([['p', parent], ['c', child]]);
  apply({ on: (event, h) => { if (event === 'tools/pre-execute') handler = h; }, inject: () => {}, sessions, permissionPresets: { current: s => s === parent ? parentPreset : childPreset }, approval: { overrideOf: () => override } });
  const write = { name: 'write', agent: { session: child }, input: { file_path: 'x' } };
  const allow = async () => ({ kind: 'allow' });

  assert.equal((await handler(write, allow)).kind, 'ask', 'the Manual child asks');
  assert.deepEqual(appended, [['approval/policy', 'ask']], 'the child is given the approval policy it needs');
  override = 'ask';
  parentPreset = 'strata-bypass';
  assert.equal((await handler(write, allow)).kind, 'ask', 'the parent switch does not stop the child asking');
  assert.equal(appended.length, 1, 'the answer is recorded once');

  childPreset = 'strata-bypass';
  child.ownEvents = () => [{ type: 'permission/preset', data: { preset: childPreset } }];
  child.header.origin = 'subagent';
  // A fresh child session with its own Bypass pin must not ask even while the parent is Manual.
  const bypassChild = {
    header: { origin: 'subagent', parentSession: 'p' },
    ownEvents: () => [{ type: 'permission/preset', data: { preset: 'strata-bypass' } }],
    append: (type, data) => appended.push([type, data.policy]),
  };
  parentPreset = MANUAL_PRESET; override = 'ask';
  const bypassExec = { name: 'write', agent: { session: bypassChild }, input: { file_path: 'x' } };
  assert.equal((await handler(bypassExec, allow)).kind, 'allow', 'Bypass child runs without a question');

  const planChild = {
    header: { origin: 'subagent', parentSession: 'p' },
    ownEvents: () => [{ type: 'permission/preset', data: { preset: PLAN_PRESET } }],
    append: (type, data) => appended.push([type, data.policy]),
  };
  const planExec = { name: 'mcp__other__write', agent: { session: planChild }, input: {} };
  assert.equal((await handler(planExec, allow)).kind, 'deny', 'Plan child cannot use a connected changing tool');
  // The child's own sandbox is seeded from the Harness default, so Plan must refuse built-in changes itself.
  assert.equal((await handler({ name: 'write', agent: { session: planChild }, arguments: { file_path: 'x', content: 'y' } }, allow)).kind, 'deny', 'Plan child cannot write a file');
  assert.equal((await handler({ name: 'edit', agent: { session: planChild }, arguments: { file_path: 'x' } }, allow)).kind, 'deny', 'Plan child cannot edit a file');
  assert.equal((await handler({ name: 'pwsh', agent: { session: planChild }, arguments: { command: 'Set-Content x y' } }, allow)).kind, 'deny', 'Plan child cannot run a changing command');
  assert.equal((await handler({ name: 'pwsh', agent: { session: planChild }, arguments: { command: 'Get-Content x' } }, allow)).kind, 'allow', 'Plan child can still read with a read-only command');
  assert.equal((await handler({ name: 'run_code', agent: { session: planChild }, arguments: {} }, allow)).kind, 'deny', 'Plan child cannot run code');
});

test('a delegation stopped before its first tool cannot leave its mode for the next helper', async () => {
  const { apply, MANUAL_PRESET, PLAN_PRESET } = await import('../server/agent/plugins/strata-gate.mjs');
  const handlers = new Map();
  const parent = { id: 'p', header: { origin: 'user' }, append() {} };
  let parentPreset = MANUAL_PRESET;
  apply({
    on: (event, fn) => { const list = handlers.get(event) ?? []; list.push(fn); handlers.set(event, list); },
    inject: () => {},
    sessions: new Map([['p', parent]]),
    permissionPresets: { current: session => (session === parent ? parentPreset : MANUAL_PRESET) },
    approval: { overrideOf: () => 'never' },
  });
  const pre = exec => handlers.get('tools/pre-execute')[0](exec, async () => ({ kind: 'allow' }));
  const created = child => handlers.get('session/created')[0](child);
  const childOf = id => ({ id, header: { origin: 'subagent', parentSession: 'p', createdAt: Date.now() }, ownEvents: () => [], append() {} });

  // A Manual turn starts a helper; it is stopped before it calls any tool.
  await pre({ name: 'subagent', agent: { session: parent }, arguments: {} });
  created(childOf('c1'));
  // The person switches to Plan and asks again. The new helper must be read-only from its first call,
  // even though its own fresh session still reports the profile's Manual default.
  parentPreset = PLAN_PRESET;
  await pre({ name: 'subagent', agent: { session: parent }, arguments: {} });
  const second = childOf('c2');
  created(second);
  const write = { name: 'write', agent: { session: second }, arguments: { file_path: 'x', content: 'y' } };
  assert.equal((await pre(write)).kind, 'deny', 'the Plan helper refuses the write instead of asking');
});
