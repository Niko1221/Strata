import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createConversations } from '../server/conversations.mjs';

const at = '2026-10-20T10:00:00.000Z';
const session = (id = 'desktop-session') => ({ id, title: 'Existing Mac work', workspace: '/Users/example/Project', createdAt: at, updatedAt: at, tokensIn: 32, tokensOut: 18,
  preferences: { remoteModel: { connectionId: 'saved-api', modelId: 'saved-model' }, mode: 'plan', agent: true, thinking: 'High', thinkingBudget: 4096 },
  messages: [{ id: 'existing-question', role: 'user', content: 'Keep this context', at }, { id: 'existing-reply', role: 'assistant', content: 'Existing answer', at }] });
function fixture(t, options = {}) {
  const directory = mkdtempSync(join(tmpdir(), 'strata-conversations-'));
  const service = createConversations({ directory, defaultWorkspace: () => '/Users/example/Default', ...options });
  t.after(() => { service.close(); rmSync(directory, { recursive: true, force: true }); });
  return { service, directory };
}
const nextTick = () => new Promise(resolve => setImmediate(resolve));

test('imports complete Mac history and preferences once, then ignores stale startup cache', t => {
  const { service, directory } = fixture(t);
  const original = session();
  original.messages = Array.from({ length: 150 }, (_, n) => ({ id: `m-${n}`, role: n % 2 ? 'assistant' : 'user', content: `History ${n}`, at }));
  service.sync([{ session: original }]);
  assert.deepEqual(service.detail(original.id).conversation.messages, original.messages);
  assert.deepEqual(service.detail(original.id).conversation.preferences, original.preferences);
  service.sync([{ session: { ...original, title: 'Stale title', messages: [] } }]);
  assert.equal(service.detail(original.id).conversation.title, original.title);
  assert.equal(service.detail(original.id).conversation.messages.length, 150);
  assert.equal(statSync(join(directory, 'conversations.json')).mode & 0o777, 0o600);
});

test('compare and swap refuses renderer snapshots captured before a phone turn', async t => {
  let complete;
  const { service } = fixture(t, { execute: async ({ onFrame }) => { onFrame({ type: 'delta', text: 'Phone output' }); await new Promise(r => { complete = r; }); } });
  const initial = service.sync([{ session: session() }]).conversations[0];
  service.send(initial.id, 'Continue the same work');
  await nextTick();
  service.sync([{ session: { ...initial, title: 'Stale', messages: [] }, baseRevision: initial.revision }]);
  assert.equal(service.detail(initial.id).conversation.messages.at(-1).content, 'Phone output');
  complete(); await nextTick();
  service.sync([{ session: initial, baseRevision: initial.revision }]);
  const final = service.detail(initial.id).conversation;
  assert.equal(final.messages.length, 4);
  assert.equal(final.messages.at(-1).content, 'Phone output');
  assert.equal(final.active, false);
});

test('explicit deletion persists a tombstone and stale imports cannot resurrect it', t => {
  const { service } = fixture(t);
  service.sync([{ session: session() }]); service.remove('desktop-session');
  const result = service.sync([{ session: session() }]);
  assert.equal(result.results[0].deleted, true);
  assert.equal(service.list().conversations.length, 0);
});

test('continuation keeps exact chat ID, folder, model, permission mode and all prior messages', async t => {
  let executed;
  const { service } = fixture(t, { execute: async args => { executed = args; args.onFrame({ type: 'delta', text: 'Continued' }); } });
  const original = session(); service.sync([{ session: original }]);
  const admitted = service.send(original.id, 'Continue from my phone');
  assert.equal(admitted.conversation.active, true);
  await nextTick();
  assert.deepEqual(executed.session, original);
  assert.equal(executed.mode, 'plan');
  assert.equal(executed.message, 'Continue from my phone');
  assert.equal(service.detail(original.id).conversation.messages.at(-1).content, 'Continued');
});

test('new chats use the default Mac folder and manual mode despite supplied permission fields', t => {
  const { service } = fixture(t);
  const { conversation } = service.create({ remoteModel: { connectionId: 'a', modelId: 'b' }, mode: 'bypass', workspace: '/' });
  assert.equal(conversation.workspace, '/Users/example/Default');
  assert.equal(conversation.preferences.mode, 'manual');
});

test('running work is independent of a phone connection and refuses simultaneous sends', async t => {
  let finish;
  const { service } = fixture(t, { execute: async ({ onFrame }) => { await new Promise(r => { finish = r; }); onFrame({ type: 'delta', text: 'Complete after leaving phone' }); } });
  const { conversation } = service.create(); service.send(conversation.id, 'Work'); await nextTick();
  assert.throws(() => service.send(conversation.id, 'Another'), e => e.status === 409);
  finish(); await nextTick();
  assert.equal(service.detail(conversation.id).conversation.messages.at(-1).content, 'Complete after leaving phone');
  assert.equal(service.detail(conversation.id).conversation.active, false);
});

test('browser cards, ordered text and grouped activity survive the durable projection', t => {
  const { service } = fixture(t);
  service.sync([{ session: session() }]); service.begin('desktop-session', 'Browse', { userMessageId: 'u-new', assistantMessageId: 'a-new' });
  service.record('desktop-session', { type: 'delta', text: 'I will check. ' });
  service.record('desktop-session', { type: 'agent-tool', tool: { id: 'browser-1', name: 'mcp__browser__navigate', verb: 'Opening a page', target: 'https://example.com', args: { url: 'https://example.com' }, state: 'running', scope: 'read', preview: 'Browse', effect: 'Read a page' } });
  service.record('desktop-session', { type: 'activity-group', id: 'g1', line: 'Opened a page', closed: false });
  service.record('desktop-session', { type: 'delta', text: 'The page says hello.' });
  service.record('desktop-session', { type: 'activity-group', id: 'g1', line: 'Read the page', closed: true });
  service.record('desktop-session', { type: 'agent-tool', tool: { id: 'browser-1', state: 'done', result: 'Page title', durationMs: 23 } });
  service.record('desktop-session', { type: 'done' });
  const reply = service.detail('desktop-session').conversation.messages.at(-1);
  assert.equal(reply.content, 'I will check. The page says hello.');
  assert.deepEqual(reply.blocks.map(b => b.kind), ['text', 'group', 'text']);
  assert.equal(reply.blocks[0].text, 'I will check. ');
  assert.equal(reply.blocks[2].text, 'The page says hello.');
  assert.equal(reply.blocks[1].closed, true);
  assert.equal(reply.tools[0].state, 'done');
  assert.equal(reply.tools[0].target, 'https://example.com');
});

test('tool-first answers keep the final text block and desktop IDs do not duplicate', t => {
  const { service } = fixture(t);
  const original = session(); original.messages.push({ id: 'u2', role: 'user', content: 'Go', at }, { id: 'a2', role: 'assistant', content: '', streaming: true, at });
  service.sync([{ session: original }]); service.begin(original.id, 'Go', { userMessageId: 'u2', assistantMessageId: 'a2' });
  service.record(original.id, { type: 'activity-group', id: 'group-first', line: 'Using the browser' });
  service.record(original.id, { type: 'delta', text: 'Final answer' }); service.finish(original.id);
  const final = service.detail(original.id).conversation;
  assert.equal(final.messages.length, 4);
  assert.equal(final.messages.at(-1).blocks[1].text, 'Final answer');
});

test('approval identity uses real pending call and keeps browser card details', async t => {
  let approved;
  const { service } = fixture(t, { pending: () => [{ callId: 'browser-ask', tool: 'mcp__browser__click', reason: 'Submit a form', at: 1 }], approve: async options => { approved = options; } });
  service.sync([{ session: session() }]); service.begin('desktop-session', 'Go');
  service.record('desktop-session', { type: 'agent-tool', tool: { id: 'browser-ask', name: 'mcp__browser__click', preview: 'Submit', effect: 'Writes to the website', scope: 'write', state: 'proposed', args: { button: 'Send' } } });
  const detail = service.detail('desktop-session');
  assert.equal(detail.pending[0].name, 'mcp__browser__click');
  assert.equal(detail.pending[0].preview, 'Submit');
  assert.equal(detail.pending[0]['0'], undefined);
  await service.approval('desktop-session', 'browser-ask', 'deny');
  assert.deepEqual(approved, { chatId: 'desktop-session', callId: 'browser-ask', decision: 'deny' });
});

test('a tool still stopping stays active and blocks a second turn', async t => {
  const { service } = fixture(t, { stop: async () => ({ stopped: false, pending: true }) });
  const { conversation } = service.create(); service.begin(conversation.id, 'Running');
  assert.equal((await service.stop(conversation.id)).conversation.active, true);
  assert.throws(() => service.send(conversation.id, 'Race'), e => e.status === 409);
});

test('restart keeps partial output but reports the interrupted turn honestly', t => {
  const { service, directory } = fixture(t);
  service.sync([{ session: session() }]); service.begin('desktop-session', 'Go'); service.record('desktop-session', { type: 'delta', text: 'Partial' }); service.close();
  const restored = createConversations({ directory });
  const detail = restored.detail('desktop-session').conversation;
  assert.equal(detail.active, false); assert.equal(detail.messages.at(-1).content, 'Partial');
  assert.match(detail.messages.at(-1).error, /restarted/);
  restored.close();
});

test('unavailable saved model refuses admission before creating messages', t => {
  const { service } = fixture(t, { validateSelection: () => { throw Object.assign(new Error('Model gone'), { status: 409 }); } });
  service.sync([{ session: session() }]);
  assert.throws(() => service.send('desktop-session', 'Continue'), e => e.status === 409);
  assert.equal(service.detail('desktop-session').conversation.messages.length, 2);
});

test('stopping an admitted turn before execution does not start detached work later', async t => {
  let executions = 0;
  const { service } = fixture(t, { execute: async () => { executions += 1; } });
  const { conversation } = service.create();
  service.send(conversation.id, 'Work');
  await service.stop(conversation.id); await nextTick();
  assert.equal(executions, 0);
  assert.equal(service.detail(conversation.id).conversation.active, false);
});


test('never enabled sharing creates no catalog, changes no messages, and publishes no events', t => {
  const directory = mkdtempSync(join(tmpdir(), 'strata-no-shared-optin-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  let enabled = false;
  const notifications = [];
  const service = createConversations({ directory, enabled: () => enabled, notify: (...args) => notifications.push(args) });
  const original = session();
  assert.equal(service.available(), false);
  assert.deepEqual(service.list(), { available: false, conversations: [], activeChatId: null });
  assert.deepEqual(service.sync([{ session: original }]).results, []);
  service.record(original.id, { type: 'delta', text: 'Unshared' });
  assert.throws(() => service.create(), e => e.status === 409);
  service.close();
  assert.equal(existsSync(join(directory, 'conversations.json')), false);
  assert.deepEqual(original, session());
  assert.deepEqual(notifications, []);
  enabled = true;
  service.sync([{ session: original }]);
  assert.equal(service.list().available, true);
  assert.deepEqual(service.detail(original.id).conversation.messages, original.messages);
  assert.equal(existsSync(join(directory, 'conversations.json')), true);
  service.close();
  const restored = createConversations({ directory, enabled: () => false });
  assert.equal(restored.available(), true, 'an existing shared catalog retains local history after pairing is disabled');
  assert.equal(restored.list().conversations.length, 1);
  restored.close();
});

test('a stopped queued task cannot finish the replacement turn before it executes', async t => {
  let complete;
  const calls = [];
  const { service } = fixture(t, { execute: async ({ message, onFrame }) => { calls.push(message); onFrame({ type: 'delta', text: 'Second reply' }); await new Promise(r => { complete = r; }); } });
  const id = service.create().conversation.id;
  service.send(id, 'First'); await service.stop(id); service.send(id, 'Second'); await nextTick();
  assert.deepEqual(calls, ['Second']);
  assert.equal(service.detail(id).conversation.active, true);
  assert.equal(service.detail(id).conversation.messages.at(-1).content, 'Second reply');
  complete(); await nextTick(); assert.equal(service.detail(id).conversation.active, false);
});

for (const settles of ['resolve', 'reject']) test(`late ${settles}, delta, and error from a stopped executor cannot change its successor`, async t => {
  const executions = [];
  const { service } = fixture(t, { stop: async () => ({ stopped: false }), execute: args => new Promise((resolve, reject) => executions.push({ ...args, resolve, reject })) });
  const id = service.create().conversation.id;
  service.send(id, 'First'); await nextTick(); await service.stop(id);
  service.send(id, 'Second'); await nextTick();
  assert.equal(executions.length, 2);
  executions[1].onFrame({ type: 'delta', text: 'Current output' });
  executions[0].onFrame({ type: 'delta', text: 'STALE OUTPUT' });
  executions[0].onFrame({ type: 'error', message: 'STALE ERROR' });
  executions[0][settles](new Error('STALE EXECUTOR FAILURE')); await nextTick();
  const active = service.detail(id).conversation;
  assert.equal(active.active, true);
  assert.equal(active.messages.at(-1).content, 'Current output');
  assert.equal(active.messages.at(-1).error, undefined);
  executions[1].resolve(); await nextTick(); assert.equal(service.detail(id).conversation.active, false);
});

test('a late stop response cannot cancel a successor that started after the old turn ended', async t => {
  let replyToStop;
  const executions = [];
  const { service } = fixture(t, { stop: () => new Promise(r => { replyToStop = r; }), execute: args => new Promise(resolve => executions.push({ ...args, resolve })) });
  const id = service.create().conversation.id;
  service.send(id, 'First'); await nextTick();
  const stopping = service.stop(id);
  executions[0].onFrame({ type: 'done' }); service.send(id, 'Second'); await nextTick();
  replyToStop({ stopped: false }); await stopping;
  executions[0].resolve(); await nextTick();
  assert.equal(service.detail(id).conversation.active, true);
  assert.equal(executions[1].signal.aborted, false);
  executions[1].onFrame({ type: 'delta', text: 'Unaffected' }); executions[1].resolve(); await nextTick();
  assert.equal(service.detail(id).conversation.messages.at(-1).content, 'Unaffected');
});

test('captured Mac stream callbacks and finalizers cannot write into a newer turn', t => {
  const { service } = fixture(t);
  const id = service.create().conversation.id;
  service.begin(id, 'First'); const first = service.capture(id);
  first.record({ type: 'delta', text: 'First output' }); first.finish();
  service.begin(id, 'Second'); const second = service.capture(id);
  first.record({ type: 'delta', text: 'STALE' }); first.finish('STALE ERROR');
  assert.equal(first.current(), false); assert.equal(second.current(), true);
  assert.equal(service.detail(id).conversation.active, true);
  assert.equal(service.detail(id).conversation.messages.at(-1).content, '');
  second.record({ type: 'delta', text: 'Second output' }); second.finish();
  assert.equal(service.detail(id).conversation.messages.at(-1).content, 'Second output');
});

test('reasoning starts on its first text and freezes at the first answer without resetting', t => {
  let milliseconds = 100;
  const { service } = fixture(t, { clock: () => milliseconds });
  const id = service.create().conversation.id; service.begin(id, 'Think, then answer');
  const reply = () => service.detail(id).conversation.messages.at(-1);
  milliseconds = 200; service.record(id, { type: 'reasoning', text: '' });
  assert.equal(reply().reasoningStartedAt, undefined);
  milliseconds = 1000; service.record(id, { type: 'reasoning', text: 'First thought. ' });
  milliseconds = 1700; service.record(id, { type: 'reasoning', text: 'Second thought.' });
  assert.equal(reply().reasoningStartedAt, 1000);
  assert.equal(reply().reasoningMs, undefined);
  milliseconds = 2000; service.record(id, { type: 'delta', text: '' });
  assert.equal(reply().reasoningMs, undefined, 'an empty frame is not an answer');
  milliseconds = 2500; service.record(id, { type: 'delta', text: 'The answer' });
  assert.equal(reply().reasoningMs, 1500);
  milliseconds = 4000; service.record(id, { type: 'reasoning', text: ' Additional thought.' });
  milliseconds = 8000; service.record(id, { type: 'delta', text: ' is complete.' }); service.record(id, { type: 'done' });
  assert.equal(reply().reasoningStartedAt, 1000);
  assert.equal(reply().reasoningMs, 1500, 'later reasoning, text, and completion cannot extend frozen thinking time');
  assert.equal(reply().reasoning, 'First thought. Second thought. Additional thought.');
});

for (const ending of ['done', 'error']) test(`reasoning-only ${ending} records elapsed thinking even with a zero start timestamp`, t => {
  let milliseconds = 0;
  const { service } = fixture(t, { clock: () => milliseconds });
  const id = service.create().conversation.id; service.begin(id, 'Reason without an answer');
  service.record(id, { type: 'reasoning', text: 'Considering the request' });
  milliseconds = 725; service.record(id, { type: 'reasoning', text: ' carefully.' });
  milliseconds = 1750; service.record(id, { type: ending, ...(ending === 'error' ? { message: 'The model stopped' } : {}) });
  const reply = service.detail(id).conversation.messages.at(-1);
  assert.equal(reply.reasoningStartedAt, 0);
  assert.equal(reply.reasoningMs, 1750);
  assert.equal(reply.streaming, false);
  assert.equal(reply.content, '');
  if (ending === 'error') assert.equal(reply.error, 'The model stopped');
});

test('imported thinking timestamps and frozen durations survive shared continuation and persistence', t => {
  let milliseconds = 350;
  const { service, directory } = fixture(t, { clock: () => milliseconds });
  const original = session();
  original.messages[1] = { ...original.messages[1], reasoning: 'Historical thought', reasoningStartedAt: 10, reasoningMs: 20 };
  original.messages.push({ id: 'ongoing-user', role: 'user', content: 'Continue', at }, { id: 'ongoing-assistant', role: 'assistant', content: '', reasoning: 'Already thinking', reasoningStartedAt: 100, streaming: true, at });
  service.sync([{ session: original }]);
  service.begin(original.id, 'Continue', { userMessageId: 'ongoing-user', assistantMessageId: 'ongoing-assistant' });
  service.record(original.id, { type: 'reasoning', text: ' more.' });
  assert.equal(service.detail(original.id).conversation.messages.at(-1).reasoningStartedAt, 100);
  milliseconds = 600; service.record(original.id, { type: 'delta', text: 'Answer' }); service.finish(original.id);
  assert.equal(service.detail(original.id).conversation.messages.at(-1).reasoningMs, 500);
  const historical = service.detail(original.id).conversation.messages[1];
  assert.equal(historical.reasoningStartedAt, 10); assert.equal(historical.reasoningMs, 20);
  service.close();
  const restored = createConversations({ directory });
  assert.equal(restored.detail(original.id).conversation.messages.at(-1).reasoningMs, 500);
  assert.equal(restored.detail(original.id).conversation.messages[1].reasoningMs, 20);
  restored.close();
});

test('answers without reasoning do not invent a thinking timer', t => {
  const { service } = fixture(t, { clock: () => 9999 });
  const id = service.create().conversation.id; service.begin(id, 'Answer directly');
  service.record(id, { type: 'delta', text: 'Direct answer' }); service.finish(id);
  const reply = service.detail(id).conversation.messages.at(-1);
  assert.equal(reply.reasoningStartedAt, undefined); assert.equal(reply.reasoningMs, undefined);
});
