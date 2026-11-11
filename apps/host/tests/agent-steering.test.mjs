/** Real Harness Gateway/agent/tool execution with a scripted HTTP model, not a real-model evaluation. */
import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtempSync, mkdirSync, existsSync, readFileSync, rmSync, writeFileSync, watch } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { SOFT_CUT_PATCH, SteeringStore, ParentSteering, isSteeringDelivery } from '../server/agent/steering.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';

const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
const harness = process.env.STRATA_HARNESS_DIR;
const supported = (() => { try { return JSON.parse(readFileSync(join(harness, 'BUNDLE.json'), 'utf8')).patches?.some(p => p.id === SOFT_CUT_PATCH); } catch { return false; } })();
const skip = !supported && 'STRATA_HARNESS_DIR must point to a source-built soft-cut Harness';
const waitFor = async predicate => {
  const until = Date.now() + 15000;
  while (!predicate()) { if (Date.now() > until) throw new Error('Expected lifecycle transition did not arrive'); await new Promise(r => setTimeout(r, 10)); }
};

test('parent lifecycle recovers late attachment and rejects stale steps and non-lifecycle activity', () => {
  const parent = new ParentSteering();
  const event = (seq, type, turn) => parent.observe({ seq, type, data: { turn } });
  event(10, 'step/start', 3); assert.equal(parent.turn, 3);
  event(11, 'turn/end', 3); assert.equal(parent.turn, null);
  event(9, 'turn/start', 3); event(12, 'step/start', 3);
  event(13, 'subagent/catalog', 4); event(14, 'user/message', 4);
  assert.equal(parent.turn, null);
  event(15, 'turn/start', 4); assert.equal(parent.turn, 4);
  parent.reject(4); event(16, 'step/start', 4); assert.equal(parent.turn, null);
  event(17, 'turn/start', 5); assert.equal(parent.turn, 5);
});

test('a closed parent with a working helper rejects steering, then publishes parent resumption', { skip, timeout: 60000 }, async t => {
  const root = mkdtempSync(join(tmpdir(), 'strata-steering-helper-'));
  const worker = await startScriptedWorker({});
  const bridge = createAgentBridge({ dataDir: root, localInstances: () => [{ id: 'local', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768 }], connections: () => [], env: process.env });
  t.after(async () => { await bridge.close(); await worker.close(); rmSync(root, { recursive: true, force: true }); });
  const frames = [];
  const done = bridge.runTurn({ chatId: 'helpers', workspacePath: root, message: 'SCRIPT:spawn 1 slow 70', mode: 'bypass', onFrame: f => frames.push(f) });
  await waitFor(() => frames.some(f => f.type === 'agent' && f.agent.status === 'working') && frames.filter(f => f.type === 'steering-availability').length >= 3 && frames.filter(f => f.type === 'steering-availability').at(-1).available === false);
  assert.equal(bridge.isRunning('helpers'), true);
  await assert.rejects(bridge.steer({ chatId: 'helpers', id: 'closed-parent', text: 'Keep this correction' }), /parent agent is not accepting/);
  assert.deepEqual(bridge.steeringState('helpers'), []);
  const requestId = frames.find(frame => frame.type === 'start')?.requestId;
  const reattached = [];
  const attachment = bridge.attach({ chatId: 'helpers', requestId, onFrame: frame => reattached.push(frame), signal: new AbortController().signal });
  assert.equal(attachment?.live, true);
  assert.equal(reattached.filter(frame => frame.type === 'steering-availability').at(-1)?.available, false);
  await done;
  assert.deepEqual(frames.filter(f => f.type === 'steering-availability').map(f => f.available), [false, true, false]);
  assert.deepEqual(reattached.filter(f => f.type === 'steering-availability').map(f => f.available), [false, true, false]);
});

test('reattaching an open parent turn republishes steering eligibility to the new page', { skip, timeout: 60000 }, async t => {
  const r = await rig(t, { holdFuture: true }); await r.prefix;
  await waitFor(() => r.frames.some(frame => frame.type === 'steering-availability' && frame.available === true));
  const requestId = r.frames.find(frame => frame.type === 'start')?.requestId;
  const reattached = [];
  const attachment = r.bridge.attach({ chatId: 'chat-a', requestId, onFrame: frame => reattached.push(frame), signal: new AbortController().signal });
  assert.equal(attachment?.live, true);
  assert.equal(reattached[0]?.resumed, true);
  assert.equal(reattached.filter(frame => frame.type === 'steering-availability').at(-1)?.available, true);
  r.releaseModel();
  await r.done;
});

test('an old or missing Harness rejects true steering without starting a replacement run', async t => {
  const root = mkdtempSync(join(tmpdir(), 'strata-steering-unsupported-'));
  const bridge = createAgentBridge({ dataDir: root, localInstances: () => [], connections: () => [], env: { STRATA_HARNESS_DIR: join(root, 'absent') } });
  t.after(async () => { await bridge.close(); rmSync(root, { recursive: true, force: true }); });
  await assert.rejects(bridge.steer({ chatId: 'chat', id: 'request', text: 'Correction' }), /does not support same-turn steering/);
  assert.equal(bridge._debug().rt, null);
  assert.deepEqual(bridge.steeringState('chat'), []);
});

test('steering receipts survive restart, reject ID reuse, and cannot downgrade delivered instructions', t => {
  const root = mkdtempSync(join(tmpdir(), 'strata-steering-receipts-')); t.after(() => rmSync(root, { recursive: true, force: true }));
  const file = join(root, 'receipts.json'), store = new SteeringStore(file);
  const value = { chatId: 'chat', sessionId: 'session', turnId: 'turn', id: 'instruction', text: 'Keep this' };
  store.add(value); store.add(value);
  assert.equal(store.list('chat').length, 1);
  assert.throws(() => store.add({ ...value, text: 'Different' }), /different text/);
  store.update('chat', value.id, { state: 'delivered', accepted: true });
  store.update('chat', value.id, { state: 'failed' });
  assert.equal(new SteeringStore(file).get('chat', value.id).state, 'delivered');
  assert.deepEqual(store.list('other-chat'), []);
  const missing = new SteeringStore(join(root, 'missing', 'file.json'));
  assert.throws(() => missing.add(value));
  assert.deepEqual(missing.list('chat'), [], 'a failed write publishes no accepted receipt');
});

async function rig(t, { tool = false, manual = false, holdFuture = false } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-steer-agent-')), workspace = join(root, 'work'); mkdirSync(workspace);
  const frames = [], requests = [], held = [], prefix = deferred(), proposed = deferred(), delivered = deferred(); let released = false;
  let requestArrived = deferred();
  const marker = join(workspace, 'operations'), began = join(workspace, 'began'), release = join(workspace, 'release');
  const server = createServer(async (req, res) => {
    if (req.url === '/v1/models') { res.setHeader('content-type', 'application/json'); res.end(JSON.stringify({ data: [{ id: 'scripted-model' }] })); return; }
    if (!req.url.endsWith('/chat/completions')) { res.writeHead(404); res.end(); return; }
    let raw = ''; for await (const piece of req) raw += piece;
    const body = JSON.parse(raw); requests.push(body);
    requestArrived.resolve(body); requestArrived = deferred();
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    const delta = chunk => res.write('data: ' + JSON.stringify({ choices: [{ index: 0, delta: chunk, finish_reason: null }] }) + '\n\n');
    const end = reason => { res.write('data: ' + JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: reason }] }) + '\n\ndata: [DONE]\n\n'); res.end(); };
    if (requests.length === 1 && tool) {
      const name = body.tools.find(x => ['bash', 'pwsh'].includes(x.function.name)).function.name;
      const psPath = path => `'${path.replaceAll("'", "''")}'`;
      const command = name === 'pwsh'
        ? `[System.IO.File]::WriteAllText(${psPath(began)}, 'started'); ${manual ? '' : `while (-not (Test-Path -LiteralPath ${psPath(release)})) { Start-Sleep -Milliseconds 50 }; `}[System.IO.File]::AppendAllText(${psPath(marker)}, "done\`n"); Write-Output 'actual-tool-result'`
        : `printf started > ${JSON.stringify(began)}; ${manual ? '' : `while [ ! -f ${JSON.stringify(release)} ]; do sleep 0.05; done; `}printf 'done\\n' >> ${JSON.stringify(marker)}; printf actual-tool-result`;
      delta({ tool_calls: [{ index: 0, id: 'side-effect', type: 'function', function: { name, arguments: JSON.stringify({ description: 'Perform the controlled operation once', command }) } }] }); end('tool_calls'); return;
    }
    if (requests.length === 1) {
      delta({ reasoning_content: 'Unfinished reasoning must not become history' });
      delta({ content: 'Preserved visible prefix' });
      if (holdFuture) held.push(() => end('stop'));
      else delta({ tool_calls: [{ index: 0, id: 'incomplete-call', type: 'function', function: { name: 'bash', arguments: '{"command":' } }] });
      return;
    }
    const finish = () => { if (!res.destroyed && !res.writableEnded) { delta({ content: 'Corrected answer' }); end('stop'); } };
    held.push(finish); if (released) finish();
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const local = [{ id: 'local', label: 'Scripted', baseUrl: `http://127.0.0.1:${server.address().port}`, modelId: 'scripted-model', context: 32768 }];
  const bridge = createAgentBridge({ dataDir: root, localInstances: () => local, connections: () => [], env: process.env });
  t.after(async () => { await bridge.close(); server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); rmSync(root, { recursive: true, force: true }); });
  const done = bridge.runTurn({ chatId: 'chat-a', workspacePath: workspace, message: tool ? 'Run the requested operation once.' : 'Start the task.', mode: manual ? 'manual' : 'bypass', onFrame(frame) {
    frames.push(frame);
    if (frame.type === 'delta' && frame.text.includes('Preserved visible prefix')) prefix.resolve();
    if (frame.type === 'agent-tool' && frame.tool.state === 'proposed') proposed.resolve(frame.tool);
    if (frame.type === 'steering' && frame.message.state === 'delivered') delivered.resolve();
  } });
  const waitFile = path => new Promise((resolve, reject) => {
    if (existsSync(path)) return resolve();
    const timer = setTimeout(() => { watcher.close(); reject(new Error(`Tool did not reach its controlled boundary: ${JSON.stringify(frames.filter(f => ['agent-tool', 'error'].includes(f.type)))}`)); }, 15000);
    const watcher = watch(workspace, () => { if (existsSync(path)) { clearTimeout(timer); watcher.close(); resolve(); } });
  });
  return { bridge, frames, requests, done, prefix: prefix.promise, proposed: proposed.promise, delivered: delivered.promise, marker, began, release, waitFile,
    nextModel() { return requestArrived.promise; },
    releaseModel() { released = !holdFuture; for (const finish of held.splice(0)) finish(); },
    async events() {
      const runtime = bridge._debug().rt, sessionId = bridge.sessionOf('chat-a');
      return new Promise((resolve, reject) => {
        const follow = runtime.gw.openStream('session/follow', { request: { address: { kind: 'session', sessionId } } }, { onItem(item) { if (item.type === 'snapshot') { follow.close(); resolve((item.records ?? []).filter(r => r.type === 'event').map(r => r.event)); } }, onError: reject });
      });
    },
  };
}

test('generation is soft-cut in the same turn; ordered duplicate-safe input survives a Gateway reconnect', { skip, timeout: 60000 }, async t => {
  const NativeWebSocket = globalThis.WebSocket, sockets = []; let nextSocket = deferred();
  globalThis.WebSocket = class extends NativeWebSocket { constructor(...args) { super(...args); sockets.push(this); this.addEventListener('open', () => nextSocket.resolve(this), { once: true }); } };
  t.after(() => { globalThis.WebSocket = NativeWebSocket; });
  const r = await rig(t); await r.prefix;
  const sessionId = r.bridge.sessionOf('chat-a');
  nextSocket = deferred(); sockets.at(-1).close(); await nextSocket.promise;
  await waitFor(() => r.frames.filter(f => f.type === 'steering-availability').length >= 4 && r.frames.filter(f => f.type === 'steering-availability').at(-1).available);
  const first = { chatId: 'chat-a', id: 'first', text: 'STEER: use another folder' }, second = { chatId: 'chat-a', id: 'second', text: 'STEER: preserve file names' };
  await Promise.all([r.bridge.steer(first), r.bridge.steer(first), r.bridge.steer(second)]);
  await r.delivered;
  assert.ok(r.frames.some(f => f.type === 'agent-tool-draft' && f.draft.id === 'incomplete-call' && f.draft.done), 'upstream draft tracking closes the discarded call before continued generation');
  await assert.rejects(r.bridge.steer({ ...first, chatId: 'chat-b' }), /no active agent run/);
  r.releaseModel(); await r.done;
  assert.equal(r.bridge.sessionOf('chat-a'), sessionId);
  const events = await r.events();
  assert.equal(events.filter(e => e.type === 'turn/start').length, 1);
  assert.equal(events.filter(e => e.type === 'turn/end').length, 1);
  assert.equal(events.filter(e => e.type === 'tool/call').length, 0, 'an incomplete tool call never executes');
  const users = events.filter(e => e.type === 'user/message' && ['first', 'second'].includes(e.data.source?.rpcId));
  assert.deepEqual(users.map(e => e.data.source.rpcId), ['first', 'second']);
  const blocks = events.filter(e => e.type === 'assistant/message').flatMap(e => e.data.message.content);
  assert.ok(blocks.some(b => b.type === 'text' && b.text.includes('Preserved visible prefix')));
  assert.ok(!blocks.some(b => b.type === 'reasoning' || b.type === 'tool-call'));
  assert.deepEqual(r.bridge.steeringState('chat-a').map(m => m.state), ['delivered', 'delivered']);
});

test('steering waits for an executing side effect and preserves its actual result without replay', { skip, timeout: 60000 }, async t => {
  const r = await rig(t, { tool: true }); await r.waitFile(r.began);
  await r.bridge.steer({ chatId: 'chat-a', id: 'during-tool', text: 'STEER: continue with the result' });
  assert.equal(r.requests.length, 1, 'no model step starts before the running operation settles');
  assert.equal(existsSync(r.marker), false);
  writeFileSync(r.release, 'finish'); r.releaseModel(); await r.done;
  assert.equal(readFileSync(r.marker, 'utf8'), 'done\n');
  assert.ok(JSON.stringify(r.requests.at(-1).messages).includes('actual-tool-result'));
  assert.ok(JSON.stringify(r.requests.at(-1).messages).includes('STEER: continue with the result'));
  assert.equal((await r.events()).filter(e => e.type === 'turn/start').length, 1);
});

test('steering keeps an outstanding approval; full Stop remains full cancellation', { skip, timeout: 60000 }, async t => {
  const r = await rig(t, { tool: true, manual: true }); const tool = await r.proposed;
  await r.bridge.steer({ chatId: 'chat-a', id: 'approval', text: 'STEER: use the approved result' });
  assert.equal(r.bridge.pendingFor('chat-a').length, 1);
  assert.equal(existsSync(r.marker), false);
  await r.bridge.approve({ chatId: 'chat-a', callId: tool.id, decision: 'approve' });
  await r.delivered;
  await r.bridge.stop('chat-a'); await r.done;
  assert.equal(existsSync(r.marker), false, 'a call awaiting approval has not started and may be skipped at the steering boundary');
  const events = await r.events();
  assert.equal(events.filter(e => e.type === 'turn/start').length, 1);
  assert.equal(events.filter(e => e.type === 'turn/end').at(-1).data.reason.kind, 'aborted');
});

test('a concurrent full Stop and steering never wake a replacement turn', { skip, timeout: 60000 }, async t => {
  const r = await rig(t); await r.prefix;
  await Promise.allSettled([r.bridge.steer({ chatId: 'chat-a', id: 'stop-race', text: 'STEER: correction at cancellation' }), r.bridge.stop('chat-a')]);
  await r.done;
  const events = await r.events();
  assert.equal(events.filter(e => e.type === 'turn/start').length, 1);
  assert.equal(events.filter(e => e.type === 'turn/end').at(-1).data.reason.kind, 'aborted');
  assert.ok(events.filter(e => e.type === 'user/message' && e.data.source?.rpcId === 'stop-race').length <= 1);
});

test('parent closure between admission and RPC keeps one failed recoverable receipt and starts no turn', { skip, timeout: 60000 }, async t => {
  const r = await rig(t); await r.prefix;
  const gw = r.bridge._debug().rt.gw, rpc = gw.rpc.bind(gw);
  gw.rpc = async (method, args, options) => {
    if (method === 'session/prompt' && args.request.mode === 'soft-cut') await rpc('session/cancel', { request: { sessionId: args.request.sessionId } });
    return rpc(method, args, options);
  };
  const input = { chatId: 'chat-a', id: 'closed-race', text: 'Preserve this correction' };
  await assert.rejects(r.bridge.steer(input), /parent agent is not accepting/);
  await r.done;
  const receipts = r.bridge.steeringState('chat-a');
  assert.equal(receipts.length, 1); assert.equal(receipts[0].id, input.id);
  assert.equal(receipts[0].text, input.text); assert.equal(receipts[0].state, 'failed');
  await assert.rejects(r.bridge.steer(input));
  const events = await r.events();
  assert.equal(events.filter(e => e.type === 'turn/start').length, 1);
  assert.equal(events.filter(e => e.type === 'user/message' && e.data.source?.rpcId === input.id).length, 0);
});

test('prepared steering for A cannot enter B after A closes before Gateway admission', { skip, timeout: 60000 }, async t => {
  const r = await rig(t, { holdFuture: true }); await r.prefix;
  const gw = r.bridge._debug().rt.gw, rpc = gw.rpc.bind(gw);
  const prepared = deferred(), admit = deferred(); let preparedRequest;
  gw.rpc = async (method, args, options) => {
    if (method === 'session/prompt' && args.request.mode === 'soft-cut') {
      preparedRequest = args.request; prepared.resolve(); await admit.promise;
    }
    return rpc(method, args, options);
  };
  const input = { chatId: 'chat-a', id: 'turn-a-only', text: 'A correction must never reach B' };
  const rejected = assert.rejects(r.bridge.steer(input), /parent agent is not accepting/);
  await prepared.promise;
  assert.equal(preparedRequest.expectedTurn, 1);
  r.releaseModel(); await r.done;
  const bStarted = r.nextModel();
  await rpc('session/prompt', { request: { requestId: 'parent-b', sessionId: preparedRequest.sessionId, mode: 'queue', content: [{ type: 'text', text: 'Parent B' }] } });
  await bStarted;
  admit.resolve(); await rejected;
  const failed = r.bridge.steeringState('chat-a')[0];
  assert.equal(failed.state, 'failed'); assert.equal(failed.accepted, false);
  assert.equal(failed.parentTurn, 1); assert.equal(failed.text, input.text);
  await assert.rejects(rpc('session/prompt', { request: preparedRequest }), /prompt rejected/);
  const bCorrection = { requestId: 'b-correction', sessionId: preparedRequest.sessionId, mode: 'soft-cut', expectedTurn: 2, content: [{ type: 'text', text: 'B correction' }] };
  const bContinued = r.nextModel();
  assert.deepEqual(await rpc('session/prompt', { request: bCorrection }), { accepted: true, turn: 2 });
  await bContinued;
  assert.deepEqual(await rpc('session/prompt', { request: bCorrection }), { accepted: true, turn: 2 });
  await assert.rejects(rpc('session/prompt', { request: { ...bCorrection, expectedTurn: 1 } }), /different parent turn/);
  const events = await r.events();
  assert.deepEqual(events.filter(e => e.type === 'turn/start').map(e => e.data.turn), [1, 2]);
  assert.equal(events.filter(e => e.type === 'user/message' && e.data.source?.rpcId === input.id).length, 0);
  const delivered = events.filter(e => e.type === 'user/message' && e.data.source?.rpcId === 'b-correction');
  assert.equal(delivered.length, 1); assert.equal(delivered[0].data.source.steeringTurn, 2);
  assert.ok(!JSON.stringify(r.requests.slice(1)).includes(input.text));
  r.releaseModel();
});

test('delivery requires authoritative parent-turn provenance and acknowledgement', { skip, timeout: 60000 }, async t => {
  const item = { sessionId: 'parent', turnId: 'host-run', parentTurn: 1, id: 'correction' };
  const event = { type: 'user/message', data: { source: { rpcId: item.id, steeringTurn: 2 } } };
  assert.equal(isSteeringDelivery(item, event, 'parent', 'host-run'), false);
  delete event.data.source.steeringTurn;
  assert.equal(isSteeringDelivery(item, event, 'parent', 'host-run'), false);
  event.data.source.steeringTurn = 1;
  assert.equal(isSteeringDelivery(item, event, 'child', 'host-run'), false);
  assert.equal(isSteeringDelivery(item, event, 'parent', 'other-run'), false);
  assert.equal(isSteeringDelivery(item, event, 'parent', 'host-run'), true);
  const r = await rig(t, { holdFuture: true }); await r.prefix;
  r.bridge._debug().rt.gw.rpc = async () => ({ accepted: true, turn: 2 });
  await assert.rejects(r.bridge.steer({ chatId: 'chat-a', id: 'wrong-ack', text: 'Keep this correction' }), /did not confirm the expected parent turn/);
  assert.equal(r.bridge.steeringState('chat-a')[0].accepted, false);
  assert.equal(r.bridge.steeringState('chat-a')[0].state, 'failed');
  r.releaseModel(); await r.done;
});
