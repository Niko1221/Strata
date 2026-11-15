/** Desktop steering uses the real store and NDJSON reader, with event-controlled responses. No engine or downloads. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createRequire } from 'node:module';

const { outputFiles } = await build({ stdin: { contents: `
  export * from './src/lib/store';
  export { tr, translationsReady } from './src/lib/i18n';
  export { applyGoalFrame } from './src/lib/goal';
  export { queueOf } from './src/lib/plan';
  export { pendingChatPhotos, restoreChatPhotos } from './src/lib/chat-photos';
`, resolveDir: process.cwd() }, bundle: true, write: false, platform: 'node', format: 'cjs' });
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };

function open(saved = {}) {
  const data = new Map(Object.entries(saved));
  const storage = { getItem: key => data.get(key) ?? null, setItem: (key, value) => data.set(key, value), removeItem: key => data.delete(key) };
  const requests = [], stops = [], queues = [], steers = [], waiting = [];
  let steerReply = body => Response.json({ message: { id: body.id, text: body.message, state: 'pending' } });
  let stopReply = () => Response.json({ stopped: false });
  let queueReply = () => Response.json({ error: 'No goal is running' }, { status: 409 });
  const fetcher = async (path, init) => {
    const body = JSON.parse(init?.body ?? '{}');
    if (path === '/api/chat/steer') { steers.push(body); return steerReply(body); }
    if (path === '/api/chat/stop') { stops.push(body); return stopReply(); }
    if (path === '/api/chat/queue') { queues.push(body); return queueReply(); }
    if (path !== '/api/chat/stream') return Response.json({});
    let controller;
    const request = { body, signal: init.signal, frame: value => controller.enqueue(new TextEncoder().encode(JSON.stringify(value) + '\n')), end: () => controller.close(), fail: () => controller.error(new Error('Connection lost')) };
    const stream = new ReadableStream({ start(value) { controller = value; } });
    init.signal.addEventListener('abort', () => controller.error(new DOMException('Stopped', 'AbortError')), { once: true });
    requests.push(request);
    waiting.shift()?.resolve(request);
    return new Response(stream);
  };
  const mod = { exports: {} };
  const root = { dataset: {}, style: { setProperty() {}, removeProperty() {} } };
  const timers = new Map(); let timerId = 0;
  new Function('require', 'module', 'exports', 'localStorage', 'sessionStorage', 'document', 'window', 'matchMedia', 'fetch', 'setTimeout', 'clearTimeout', outputFiles[0].text)(
    createRequire(import.meta.url), mod, mod.exports, storage, storage,
    { documentElement: root, querySelector: () => null }, { location: { hash: '' }, matchMedia: () => ({ matches: false }) },
    () => ({ matches: false, addEventListener() {} }), fetcher,
    (fn, ms) => { const id = ++timerId; timers.set(id, { fn, ms }); return id; }, id => timers.delete(id),
  );
  const app = mod.exports;
  return { ...app, data, requests, stops, queues, steers, timers,
    flushText() { for (const [id, timer] of timers) if (timer.ms === 90) { timers.delete(id); timer.fn(); } },
    steerWith: fn => { steerReply = fn; },
    stopWith: fn => { stopReply = fn; }, queueWith: fn => { queueReply = fn; },
    nextRequest() { const next = deferred(); waiting.push(next); return next.promise; },
    session: id => app.getState().sessions.find(s => s.id === id),
    pending: id => app.getState().pendingMessages[id],
  };
}

async function start(h, text = 'First question') {
  const id = h.getState().activeSession ?? h.newSession().id;
  const next = h.nextRequest(), done = h.sendMessage(text);
  const request = await next;
  request.frame({ type: 'start', model: 'fixture', contextMax: 32000 });
  request.frame({ type: 'steering-availability', available: true });
  return { id, request, done };
}
const finish = request => { request.frame({ type: 'done', tokensOut: 1 }); request.end(); };
const tick = () => new Promise(r => setImmediate(r));

test('Wait is the default for new, existing and invalid preferences; both opt-in modes persist', () => {
  for (const prefs of [{}, { theme: 'dark' }, { whileBusy: 'unknown' }]) {
    const h = open({ 'strata.prefs.v2': JSON.stringify(prefs) });
    assert.equal(h.getState().whileBusy, 'wait');
    for (const mode of ['queue', 'interject', 'steer', 'wait']) {
      h.setWhileBusy(mode);
      assert.equal(open(Object.fromEntries(h.data)).getState().whileBusy, mode);
    }
  }
});

test('same-turn steering delivers ordered corrections without aborting or sending another turn', async () => {
  const h = open(); h.setWhileBusy('steer'); const first = await start(h);
  first.request.frame({ type: 'start', agent: true });
  first.request.frame({ type: 'delta', text: 'Keep completed visible text' });
  first.request.frame({ type: 'reasoning', text: 'Incomplete private reasoning' }); await tick();
  first.request.frame({ type: 'agent-tool-draft', draft: { id: 'partial-write', name: 'write_file', bytes: 30, text: 'incomplete file' } }); await tick();
  assert.equal(await h.submitWhileBusy('First correction'), true);
  assert.equal(await h.submitWhileBusy('Second correction'), true); await tick();
  assert.deepEqual(h.steers.map(m => m.message), ['First correction', 'Second correction']);
  first.request.frame({ type: 'steering-cut' });
  for (const message of h.steers) first.request.frame({ type: 'steering', message: { id: message.id, text: message.message, state: 'delivered' } });
  first.request.frame({ type: 'steering', message: { id: h.steers[0].id, text: 'First correction', state: 'delivered' } });
  first.request.frame({ type: 'delta', text: 'Continued in the same run' }); await tick();
  h.flushText();
  assert.equal(h.requests.length, 1);
  assert.equal(h.stops.length, 0);
  assert.equal(first.request.signal.aborted, false);
  assert.equal(h.getState().streaming, true);
  const messages = h.session(first.id).messages;
  assert.deepEqual(messages.filter(m => m.role === 'user').map(m => m.content), ['First question', 'First correction', 'Second correction']);
  assert.ok(messages.some(m => m.content === 'Keep completed visible text'));
  assert.ok(!messages.some(m => m.reasoning?.includes('Incomplete private reasoning')));
  assert.ok(!messages.some(m => m.drafts?.length), 'the upstream tool-draft UI clears at the steering boundary');
  assert.equal(messages.at(-1).content, 'Continued in the same run');
  finish(first.request); await first.done;
});

test('uncertain steering stays recoverable and retries the same ID; delivery cannot be downgraded by a late acknowledgement', async () => {
  const h = open(); h.setWhileBusy('steer'); const first = await start(h);
  first.request.frame({ type: 'start', agent: true }); await tick();
  h.steerWith(() => Promise.reject(new Error('Lost acknowledgement')));
  await h.submitWhileBusy('Do not lose this instruction'); await tick();
  const message = h.getState().steeringMessages[first.id][0];
  assert.equal(message.state, 'failed');
  assert.equal(open(Object.fromEntries(h.data)).getState().steeringMessages[first.id][0].text, message.text);
  const reply = deferred(); h.steerWith(() => reply.promise);
  const retry = h.retrySteeringMessage(first.id, message.id); await tick();
  assert.equal(h.steers[0].id, h.steers[1].id);
  first.request.frame({ type: 'steering', message: { ...message, state: 'delivered' } }); await tick();
  reply.resolve(Response.json({ message: { ...message, state: 'pending' } })); await retry;
  assert.equal(h.getState().steeringMessages[first.id][0].state, 'delivered');
  assert.equal(h.requests.length, 1);
  assert.equal(h.stops.length, 0);
  finish(first.request); await first.done;
});

test('helper activity cannot enable steering; parent availability stays with its chat and clears on completion', async () => {
  const h = open(); h.setWhileBusy('steer'); const first = await start(h);
  first.request.frame({ type: 'start', agent: true });
  first.request.frame({ type: 'steering-availability', available: false });
  first.request.frame({ type: 'agent', agent: { id: 'helper', status: 'working' } }); await tick();
  assert.equal(h.getState().streaming, true);
  assert.equal(await h.submitWhileBusy('Keep my correction'), false);
  assert.equal(h.steers.length, 0); assert.equal(h.pending(first.id), undefined);
  first.request.frame({ type: 'steering-availability', available: true }); await tick();
  const other = h.newSession();
  assert.equal(h.getState().steeringAvailable[other.id], undefined);
  assert.equal(h.getState().steeringAvailable[first.id], true);
  assert.equal(await h.submitWhileBusy('Keep my correction', first.id), true); await tick();
  assert.equal(h.steers.length, 1); assert.equal(h.requests.length, 1); assert.equal(h.stops.length, 0);
  finish(first.request); await first.done;
  assert.equal(h.getState().steeringAvailable[first.id], false);
});

test('a stale reconnect snapshot cannot hide a newer failed steering receipt', async () => {
  const h = open(); h.setWhileBusy('steer'); const first = await start(h);
  first.request.frame({ type: 'start', agent: true }); await tick();
  await h.submitWhileBusy('Keep receipt ordering'); await tick();
  const message = h.steers[0];
  first.request.frame({ type: 'steering', message: { id: message.id, text: message.message, state: 'failed', revision: 2 } });
  first.request.frame({ type: 'steering', message: { id: message.id, text: message.message, state: 'pending', revision: 1 } }); await tick();
  assert.equal(h.getState().steeringMessages[first.id][0].state, 'failed');
  finish(first.request); await first.done;
});

test('Wait retains the ordinary-response refusal and the existing goal queue', async () => {
  const h = open(), first = await start(h);
  assert.equal(await h.submitWhileBusy('Keep this draft'), false);
  assert.equal(h.pending(first.id), undefined);
  assert.equal(h.requests.length, 1);
  assert.equal(h.stops.length, 0);
  h.queueWith(() => Response.json({ queue: [{ id: 'goal-q', text: 'After the step', deliverAt: 'step' }] }));
  assert.equal(await h.submitWhileBusy('After the step'), true);
  assert.equal(h.queueOf(first.id)[0].text, 'After the step');
  finish(first.request); await first.done;
  assert.equal(h.requests.length, 1);
});

test('upstream busy-message backlog drains in order after the first parked correction succeeds', async () => {
  const h = open(); h.setWhileBusy('queue'); const first = await start(h);
  await h.submitWhileBusy('First queued');
  await h.submitWhileBusy('Second queued');
  await h.submitWhileBusy('Third queued');
  assert.deepEqual(h.getState().parked[first.id], ['Second queued', 'Third queued']);
  const second = h.nextRequest(); finish(first.request); await first.done;
  const r2 = await second; assert.equal(r2.body.message, 'First queued');
  const third = h.nextRequest(); finish(r2);
  const r3 = await third; assert.equal(r3.body.message, 'Second queued');
  const fourth = h.nextRequest(); finish(r3);
  const r4 = await fourth; assert.equal(r4.body.message, 'Third queued');
  finish(r4); await tick();
  assert.equal(h.requests.length, 4);
  assert.equal(h.stops.length, 0);
  assert.equal(h.pending(first.id), undefined);
});

test('an unsuccessful interrupted stop keeps both the first correction and upstream backlog', async () => {
  const h = open(); h.setWhileBusy('interject'); const first = await start(h);
  first.request.frame({ type: 'start', agent: true }); await tick();
  h.stopWith(() => Response.json({ stopped: true, pending: true }));
  await h.submitWhileBusy('First correction');
  await h.submitWhileBusy('Later correction');
  finish(first.request); await first.done;
  assert.equal(h.requests.length, 1);
  assert.equal(h.pending(first.id).text, 'First correction');
  assert.deepEqual(h.getState().parked[first.id], ['Later correction']);
});

test('Send after parks one ordinary message and sends only after the stream closes', async () => {
  const h = open(); h.setWhileBusy('queue');
  const first = await start(h);
  assert.equal(await h.submitWhileBusy('Next question'), true);
  assert.equal(h.pending(first.id).text, 'Next question');
  assert.equal(h.requests.length, 1);
  const next = h.nextRequest();
  first.request.frame({ type: 'done' }); await tick();
  assert.equal(h.requests.length, 1, 'a done frame alone is not a settled request');
  first.request.end(); await first.done;
  const second = await next;
  assert.equal(second.body.message, 'Next question');
  assert.equal(second.body.sessionId, first.id);
  assert.equal(h.pending(first.id), undefined);
  finish(second); await tick();
});

test('Interrupt waits for the stop acknowledgement AND closed stream, keeping the partial reply in history and storage', async () => {
  const h = open(); await h.translationsReady(); h.setWhileBusy('interject');
  const ack = deferred(); h.stopWith(() => ack.promise);
  const first = await start(h);
  first.request.frame({ type: 'start', agent: true, model: 'fixture' });
  first.request.frame({ type: 'delta', text: 'Partial answer' }); await tick();
  assert.equal(await h.submitWhileBusy('Use the other folder'), true);
  assert.deepEqual(h.stops, [{ sessionId: first.id }]);
  first.request.end(); await tick();
  assert.equal(h.requests.length, 1, 'closing the stream before the stop acknowledgement cannot start a turn');
  const next = h.nextRequest(); ack.resolve(Response.json({ stopped: true }));
  await first.done; const second = await next;
  assert.equal(second.body.message, 'Use the other folder');
  assert.ok(second.body.history.some(m => m.role === 'assistant' && m.content === 'Partial answer'));
  const partial = h.session(first.id).messages[1];
  assert.equal(partial.content, 'Partial answer');
  assert.equal(partial.streaming, false);
  assert.equal(partial.error, h.tr('Response interrupted. Ask again to retry; the partial answer is kept.'));
  assert.match(JSON.stringify(h.session(first.id)), /Partial answer/);
  finish(second); await tick();
});

test('a non-agent interruption aborts the request, and an early acknowledgement still waits for an agent stream', async () => {
  const h = open(); h.setWhileBusy('interject');
  const first = await start(h), next = h.nextRequest();
  await h.submitWhileBusy('Correction'); await first.done;
  assert.equal(first.request.signal.aborted, true);
  finish(await next); await tick();
  const agent = await start(h);
  agent.request.frame({ type: 'start', agent: true }); await tick();
  h.stopWith(() => Response.json({ stopped: true }));
  await h.submitWhileBusy('Another correction'); await tick();
  assert.equal(h.requests.length, 3);
  const last = h.nextRequest(); agent.request.end(); await agent.done;
  finish(await last); await tick();
});

test('failed or timed-out agent stops leave the new draft parked instead of racing another turn', async () => {
  for (const reply of [() => Response.json({ stopped: false, pending: true }), () => Promise.reject(new Error('Stop failed'))]) {
    const h = open(); h.setWhileBusy('interject'); h.stopWith(reply);
    const first = await start(h);
    first.request.frame({ type: 'start', agent: true }); await tick();
    await h.submitWhileBusy('Recover me'); await first.done;
    assert.equal(h.requests.length, 1);
    assert.equal(h.takePendingMessage(first.id).text, 'Recover me');
  }
});

test('a failed answer or manual Stop retains the queued draft, and a failed follow-up remains recoverable', async () => {
  for (const stop of [false, true]) {
    const h = open(); h.setWhileBusy('queue');
    const first = await start(h); await h.submitWhileBusy('Still needed');
    if (stop) h.stopStreaming(); else first.request.fail();
    await first.done;
    assert.equal(h.requests.length, 1);
    assert.equal(h.pending(first.id).text, 'Still needed');
    const reopened = open(Object.fromEntries(h.data));
    assert.equal(reopened.pending(first.id).text, 'Still needed', 'a reload preserves the recoverable draft');
    const next = h.nextRequest(), sending = h.sendPendingMessage(first.id);
    (await next).fail(); await sending;
    assert.equal(h.pending(first.id).text, 'Still needed');
  }
});

test('an interrupted goal sends the correction before draining released goal messages in order', async () => {
  const h = open(); h.setWhileBusy('interject'); const first = await start(h);
  const correction = h.nextRequest();
  await h.submitWhileBusy('Correction first');
  first.request.frame({ type: 'queued-release', messages: [{ text: 'Queued earlier' }, { text: 'Queued last' }] });
  await first.done;
  const second = await correction;
  assert.equal(second.body.message, 'Correction first');
  const released = h.nextRequest(); finish(second);
  const third = await released;
  assert.equal(third.body.message, 'Queued earlier');
  assert.equal(third.body.sessionId, first.id);
  const last = h.nextRequest(); finish(third);
  const fourth = await last;
  assert.equal(fourth.body.message, 'Queued last');
  finish(fourth); await tick();
  assert.equal(h.requests.length, 4);
  assert.equal(h.stops.length, 1, 'released messages do not interrupt the correction');
});

test('successive interruptions do not resend the earlier parked correction', async () => {
  const h = open(); h.setWhileBusy('interject'); const first = await start(h);
  const next = h.nextRequest(); await h.submitWhileBusy('First correction'); await first.done;
  await next;
  const last = h.nextRequest(); await h.submitWhileBusy('Second correction');
  const third = await last; finish(third); await tick();
  assert.equal(h.requests.length, 3);
  assert.equal(h.pending(first.id), undefined);
});

test('Send after keeps the goal queue and recovers its refusal without losing the message', async () => {
  const h = open(); h.setWhileBusy('queue'); const first = await start(h);
  h.applyGoalFrame(first.id, { objective: 'Build', phase: 'active' });
  h.queueWith(() => Response.json({ queue: [{ id: 'goal-q', text: 'Between steps', deliverAt: 'step' }] }));
  assert.equal(await h.submitWhileBusy('Between steps'), true);
  assert.equal(h.pending(first.id), undefined);
  assert.equal(h.queueOf(first.id)[0].text, 'Between steps');
  h.queueWith(() => Promise.reject(new Error('Queue offline')));
  assert.equal(await h.submitWhileBusy('Keep in composer'), false);
  h.queueWith(() => Response.json({}, { status: 409 }));
  assert.equal(await h.submitWhileBusy('Fallback to local parking'), true);
  first.request.fail(); await first.done;
  assert.equal(h.pending(first.id).text, 'Fallback to local parking');
});

test('a late queue refusal sends to the settled chat and leaves newer attachments in the composer', async () => {
  const h = open(); h.setWhileBusy('queue'); const first = await start(h);
  const reply = deferred(); h.queueWith(() => reply.promise);
  const parked = h.submitWhileBusy('After completion');
  const newer = { id: 'newer', name: 'newer.png' }; h.restoreChatPhotos([newer], first.id);
  finish(first.request); await first.done;
  const next = h.nextRequest(); reply.resolve(Response.json({}, { status: 409 }));
  assert.equal(await parked, true);
  const second = await next;
  assert.equal(second.body.message, 'After completion');
  assert.equal(second.body.imageIds, undefined);
  assert.deepEqual(h.pendingChatPhotos(first.id), [newer]);
  finish(second); await tick();
});

test('an error frame followed by done keeps the parked draft for recovery', async () => {
  const h = open(); h.setWhileBusy('queue'); const first = await start(h);
  await h.submitWhileBusy('After a successful answer');
  first.request.frame({ type: 'error', message: 'Fixture failure' }); finish(first.request); await first.done;
  assert.equal(h.requests.length, 1);
  assert.equal(h.pending(first.id).text, 'After a successful answer');
});

test('parking stays with its original chat and model when another chat is opened', async () => {
  const h = open(); h.setWhileBusy('queue'); const first = await start(h);
  h.setChatPreferences(first.id, { remoteModel: { connectionId: 'a', modelId: 'model-a' } });
  await h.submitWhileBusy('For the first chat');
  const other = h.newSession().id;
  h.setChatPreferences(other, { remoteModel: { connectionId: 'b', modelId: 'model-b' } });
  const next = h.nextRequest(); finish(first.request); await first.done;
  const second = await next;
  assert.equal(second.body.sessionId, first.id);
  assert.deepEqual(second.body.remoteModel, { connectionId: 'a', modelId: 'model-a' });
  assert.equal(h.session(other).messages.length, 0);
  assert.equal(h.getState().activeSession, other);
  finish(second); await tick();
});

test('take back preserves the parked pictures alongside newer attachments; idle Send is unchanged', async () => {
  const h = open(); h.setWhileBusy('queue'); const first = await start(h);
  const picture = { id: 'photo-a', name: 'a.png' }, newer = { id: 'photo-b', name: 'b.png' };
  h.restoreChatPhotos([picture], first.id);
  await h.submitWhileBusy('With a picture');
  assert.deepEqual(h.pendingChatPhotos(first.id), []);
  h.restoreChatPhotos([newer], first.id);
  const pending = h.takePendingMessage(first.id); h.restoreChatPhotos(pending.photos, first.id);
  assert.deepEqual(h.pendingChatPhotos(first.id), [picture, newer]);
  assert.equal(h.pending(first.id), undefined);
  finish(first.request); await first.done;
  // A separate ordinary chat has no pending pictures or steering side effects.
  h.newSession(); const idle = await start(h, 'An ordinary message');
  assert.equal(idle.request.body.message, 'An ordinary message');
  assert.deepEqual(idle.request.body.history, []);
  assert.equal(h.stops.length, 0);
  finish(idle.request); assert.equal(await idle.done, true);
});
