/**
 * A helper's conversation, kept by the host for the row's dialog: bounded lines, bounded text,
 * one line per tool call, and a replayed follow snapshot that is not written twice.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  ChildProjector, ChildTranscript, createChildTranscriptStore, advanceEvent,
  MAX_TRANSCRIPT_CHARS, MAX_TRANSCRIPT_ENTRIES, SENT_NOTICE, TASK_MISSING,
} from '../server/agent/children.mjs';

const ev = (type, data, seq = 1, time = 1000) => ({ type: 'event', event: { seq, time, type, data } });
const chars = entries => entries.reduce((total, entry) => total + entry.text.length, 0);

test('a transcript keeps the latest configured number of lines and drops the oldest', () => {
  const transcript = new ChildTranscript({ now: () => 1 });
  for (let i = 0; i < MAX_TRANSCRIPT_ENTRIES + 50; i += 1) transcript.user(`line ${i}`, i);
  const view = transcript.view();
  assert.equal(view.length, MAX_TRANSCRIPT_ENTRIES);
  assert.equal(view[0].text, 'line 50');
  assert.equal(view.at(-1).text, `line ${MAX_TRANSCRIPT_ENTRIES + 49}`);
});

test('a transcript never holds more than its character bound, even when every line is long', () => {
  const transcript = new ChildTranscript({ maxChars: 10_000, now: () => 1 });
  for (let i = 0; i < 50; i += 1) transcript.user(`${i}:${'x'.repeat(3990)}`, i);
  const view = transcript.view();
  assert.ok(view.length < 50, 'the oldest lines were dropped');
  assert.ok(chars(view) <= 10_000, `${chars(view)} <= 10000`);
  assert.match(view.at(-1).text, /^49:/, 'the newest line is kept');
});

test('one huge line is clipped instead of breaking the bound', () => {
  const transcript = new ChildTranscript({ now: () => 1 });
  transcript.user('y'.repeat(80_000), 1);
  const view = transcript.view();
  assert.equal(view.length, 1);
  assert.ok(view[0].text.length < 70_000);
  assert.ok(view[0].text.endsWith('…'));
});

test('streamed words grow one assistant line and the final message replaces it', () => {
  const transcript = new ChildTranscript({ now: () => 1 });
  transcript.stream('Hel', 10);
  transcript.stream('lo', 11);
  assert.deepEqual(transcript.view().map(entry => [entry.role, entry.text]), [['assistant', 'Hello']]);
  transcript.message('Hello there.', 12);
  assert.deepEqual(transcript.view().map(entry => [entry.role, entry.text]), [['assistant', 'Hello there.']]);
  transcript.message('A second answer.', 13);
  assert.equal(transcript.view().length, 2);
});

test('a message sent while the helper writes does not split its answer around the person line', () => {
  const transcript = new ChildTranscript({ now: () => 1 });
  transcript.user('Read the file.', 1);
  transcript.stream('One Two', 100);
  transcript.user('Queued message.', 200);
  transcript.message('One Two Three', 300);
  assert.deepEqual(transcript.view().map(entry => [entry.role, entry.text]), [
    ['user', 'Read the file.'],
    ['assistant', 'One Two Three'],
    ['user', 'Queued message.'],
  ]);
});

test('the same answer committed twice is one message, live and after a reload', () => {
  const transcript = new ChildTranscript({ now: () => 1 });
  transcript.stream('All done.', 100);
  transcript.message('All done.', 200);
  transcript.message('All done.', 300);
  assert.deepEqual(transcript.view().map(entry => [entry.role, entry.text]), [['assistant', 'All done.']], 'a second reader commits the same words');
  transcript.user('Please repeat it.', 20_000);
  transcript.message('All done.', 20_100);
  assert.deepEqual(transcript.view().map(entry => [entry.role, entry.text]), [['assistant', 'All done.'], ['user', 'Please repeat it.'], ['assistant', 'All done.']], 'a real repeat after the person speaks is kept');

  const loaded = new ChildTranscript({ now: () => 1 });
  loaded.load([
    { id: 'e1', role: 'assistant', kind: 'message', author: 'assistant', text: 'Same words.', at: 1000 },
    { id: 'e2', role: 'assistant', kind: 'message', author: 'assistant', text: 'Same words.', at: 4000 },
  ], 2);
  assert.deepEqual(loaded.view().map(entry => entry.text), ['Same words.'], 'a saved duplicate is merged on opening');
});

test('a sent message and its Harness echo stay one line; a tool call is one line', () => {
  const transcript = new ChildTranscript({ now: () => 1 });
  transcript.beginUser('Check the tests.', 1);
  transcript.user('Check the tests.', 2);
  transcript.tool('Running a command: npm test', 3);
  transcript.stream('All pass.', 4);
  transcript.close(5);
  const view = transcript.view();
  assert.deepEqual(view.map(entry => entry.role), ['user', 'tool', 'assistant']);
  assert.equal(view.filter(entry => entry.role === 'user').length, 1);
  for (const entry of view) assert.deepEqual(Object.keys(entry).sort(), ['at', 'role', 'text']);
});

test('a failed send takes its unconfirmed line back out', () => {
  const transcript = new ChildTranscript({ now: () => 1 });
  const echo = transcript.beginUser('Never delivered.', 1);
  transcript.cancelUser(echo);
  assert.deepEqual(transcript.view(), []);
});

test('the store keeps one transcript per chat and helper, and drops the oldest beyond its cap', () => {
  const store = createChildTranscriptStore({ maxTranscripts: 3, now: () => 1 });
  const first = store.acquire('chat-a', 'child-1');
  first.user('one', 1);
  assert.equal(store.acquire('chat-a', 'child-1'), first, 'the same helper keeps its conversation');
  store.acquire('chat-a', 'child-2');
  store.acquire('chat-b', 'child-1');
  store.acquire('chat-b', 'child-2');
  assert.equal(store.size, 3);
  assert.equal(store.has('chat-a', 'child-1'), false, 'the least recently used transcript is gone');
  assert.equal(store.has('chat-b', 'child-2'), true);
  assert.equal(store.view('chat-b', 'child-2').length, 0);
  store.forget('chat-b');
  assert.equal(store.size, 1);
});

test('a replayed snapshot is not written into the transcript twice', () => {
  const transcript = new ChildTranscript({ now: () => 1 });
  assert.equal(advanceEvent(transcript, { seq: 4 }), true);
  assert.equal(advanceEvent(transcript, { seq: 4 }), false);
  assert.equal(advanceEvent(transcript, { seq: 5 }), true);
  assert.equal(advanceEvent(transcript, { seq: 3 }), false);
  assert.equal(advanceEvent(transcript, { seq: 'x' }), true, 'an event without a sequence is not guessed about');
});

test('the projector keeps the helper conversation, not the Harness context snapshots', () => {
  const cp = new ChildProjector({ id: 'c1', name: 'Writer', task: 'Write the notes.', now: () => 1 });
  cp.consume(ev('user/message', { role: 'user', source: { kind: 'time-context', form: 'snapshot' }, content: [{ type: 'text', text: 'Time sampled while preparing turn 1' }] }));
  cp.consume(ev('user/message', { role: 'user', source: { kind: 'skill-catalog', form: 'catalog' }, content: [{ type: 'text', text: '<available_skills>' }] }));
  cp.consume(ev('user/message', { role: 'user', source: { kind: 'user' }, content: [{ type: 'text', text: 'Write the notes.' }] }));
  assert.deepEqual(cp.transcript.view().map(entry => entry.text), ['Write the notes.']);
});

test('the projector records the helper task, its answer and its tool calls in order', () => {
  const cp = new ChildProjector({ id: 'c1', name: 'Writer', task: 'Write the notes.', now: () => 1 });
  cp.consume(ev('user/message', { role: 'user', content: [{ type: 'text', text: 'Write the notes.' }] }));
  cp.consume({ type: 'assistant-stream', frame: { type: 'chunk', chunk: { type: 'text-delta', text: 'Writing' } } });
  cp.consume(ev('assistant/message', { message: { content: [{ type: 'text', text: 'Writing the notes.' }] } }));
  cp.consume(ev('tool/call', { callId: 'k1', name: 'pwsh', arguments: '{"command":"npm test"}' }));
  cp.consume(ev('tool/result', { message: { toolCallId: 'k1', content: [{ type: 'text', text: 'ok' }] } }));
  cp.consume(ev('turn/end', { reason: { kind: 'completed' } }));
  assert.deepEqual(cp.transcript.view().map(entry => [entry.role, entry.text]), [
    ['user', 'Write the notes.'],
    ['assistant', 'Writing the notes.'],
    ['tool', 'Ran a command: npm test'],
  ]);
});

test('the first spawn message becomes a task card from the main assistant, never the internal guidance', () => {
  const cp = new ChildProjector({ id: 'c1', name: 'Writer', task: 'Research the three prices.', now: () => 1 });
  cp.consume(ev('user/message', {
    role: 'user',
    source: { kind: 'user' },
    content: [{ type: 'text', text: 'Research the three prices. Your parent agent id is "p1". Before you finish, send your result with send_message(...).' }],
  }));
  const first = cp.transcript.viewRich()[0];
  assert.equal(first.kind, 'task');
  assert.equal(first.author, 'parent');
  assert.equal(first.text, 'Research the three prices.');
  assert.doesNotMatch(first.text, /parent agent id|send_message/);
  assert.equal(cp.transcript.view()[0].role, 'user', 'the legacy view still calls the task a user-role line');
});

test('a direct message after the task is labelled as the person, not as the main assistant', () => {
  const cp = new ChildProjector({ id: 'c1', name: 'Writer', task: 'Write the notes.', now: () => 1 });
  cp.consume(ev('user/message', { source: { kind: 'user' }, content: [{ type: 'text', text: 'Write the notes.' }] }));
  cp.consume(ev('user/message', { source: { kind: 'user', rpcId: 'r1' }, content: [{ type: 'text', text: 'Now check the dates.' }] }));
  const entries = cp.transcript.viewRich();
  assert.deepEqual(entries.map(entry => [entry.kind, entry.author, entry.text]), [
    ['task', 'parent', 'Write the notes.'],
    ['message', 'user', 'Now check the dates.'],
  ]);
});

test('the task card is replaced when the parent tool result names the full task', () => {
  const cp = new ChildProjector({ id: 'c1', name: 'Writer', task: 'Helper 1', now: () => 1 });
  cp.consume(ev('user/message', { source: { kind: 'user' }, content: [{ type: 'text', text: 'raw spawn guidance' }] }));
  cp.setTask('Write a three-page brief with sources.');
  assert.equal(cp.transcript.viewRich()[0].text, 'Write a three-page brief with sources.');
});

test('a send_message call is a quiet notice, not a raw tool card', () => {
  const cp = new ChildProjector({ id: 'c1', name: 'Writer', task: 'Write the notes.', now: () => 1 });
  cp.consume(ev('tool/call', { callId: 's1', name: 'send_message', arguments: '{"agent_id":"p1","message":"done"}' }));
  const entries = cp.transcript.viewRich();
  assert.equal(entries.length, 1);
  assert.equal(entries[0].kind, 'notice');
  assert.equal(entries[0].text, SENT_NOTICE);
  assert.equal(entries.some(entry => entry.tool), false);
});

test('a tool row is readable by default and its raw arguments stay in the expanded tool data', () => {
  const cp = new ChildProjector({ id: 'c1', name: 'Writer', task: 'Read the file.', now: () => 1 });
  cp.consume(ev('tool/call', { callId: 'k1', name: 'read', arguments: '{"file_path":"src/app.ts"}' }));
  cp.consume(ev('tool/result', { message: { toolCallId: 'k1', content: [{ type: 'text', text: 'const app = 1;' }] } }));
  const row = cp.transcript.viewRich().find(entry => entry.kind === 'tool');
  assert.match(row.text, /Read a file/);
  assert.match(row.text, /src\/app\.ts/);
  assert.doesNotMatch(row.text, /\{|\}/, 'the default line is not raw JSON');
  assert.equal(row.tool.state, 'done');
  assert.equal(row.tool.args.file_path, 'src/app.ts');
  assert.match(row.tool.result, /const app/);
});

test('a transcript survives a store roundtrip, and an open answer is marked partial after a restart', t => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-child-transcript-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  let store = createChildTranscriptStore({ dir });
  const transcript = store.acquire('chat-a', 'child-1');
  transcript.task('Persist this task.', 1);
  transcript.user('And this reply.', 2);
  transcript.stream('The answer is', 3);
  transcript.message('The answer is ready.', 4);
  transcript.tool({ verb: 'Read a file', target: 'a.txt', state: 'done', args: { file_path: 'a.txt' }, result: 'ok' }, 5);
  assert.equal(store.flush('chat-a', 'child-1'), true);
  const before = transcript.viewRich();
  store.clear();

  store = createChildTranscriptStore({ dir });
  assert.equal(store.has('chat-a', 'child-1'), true);
  assert.deepEqual(store.view('chat-a', 'child-1'), before);
  assert.ok(store.revision('chat-a', 'child-1') > 0);

  const toolOnly = store.acquire('chat-a', 'child-2');
  toolOnly.task('A task with a running tool.', 10);
  toolOnly.tool({ verb: 'Running a command', target: 'npm test', state: 'running' }, 11);
  const partial = store.acquire('chat-a', 'child-3');
  partial.task('A task that was cut off.', 20);
  partial.stream('Half an answer', 21);
  assert.equal(store.flush('chat-a', 'child-2'), true);
  assert.equal(store.flush('chat-a', 'child-3'), true);
  store.clear();

  const restarted = createChildTranscriptStore({ dir });
  const toolEntries = restarted.view('chat-a', 'child-2');
  assert.equal(toolEntries[1].tool.state, 'interrupted');
  assert.match(toolEntries[1].tool.result, /Stopped before it finished/);
  const entries = restarted.view('chat-a', 'child-3');
  assert.equal(entries[1].partial, true);
  assert.equal(entries[1].open, undefined);
});

test('an empty parent task still renders a task card with a safe placeholder', () => {
  const cp = new ChildProjector({ id: 'c1', name: 'Writer', now: () => 1 });
  cp.consume(ev('user/message', { source: { kind: 'user' }, content: [{ type: 'text', text: 'Your parent agent id is "p1".' }] }));
  const first = cp.transcript.viewRich()[0];
  assert.equal(first.kind, 'task');
  assert.equal(first.text, TASK_MISSING);
  assert.doesNotMatch(first.text, /parent agent id/);
});
