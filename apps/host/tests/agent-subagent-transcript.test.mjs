/**
 * A helper's conversation, kept by the host for the row's dialog: bounded lines, bounded text,
 * one line per tool call, and a replayed follow snapshot that is not written twice.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ChildProjector, ChildTranscript, createChildTranscriptStore, advanceEvent,
  MAX_TRANSCRIPT_CHARS, MAX_TRANSCRIPT_ENTRIES,
} from '../server/agent/children.mjs';

const ev = (type, data, seq = 1, time = 1000) => ({ type: 'event', event: { seq, time, type, data } });
const chars = entries => entries.reduce((total, entry) => total + entry.text.length, 0);

test('a transcript keeps the latest 200 lines and drops the oldest', () => {
  const transcript = new ChildTranscript({ now: () => 1 });
  for (let i = 0; i < 250; i += 1) transcript.user(`line ${i}`, i);
  const view = transcript.view();
  assert.equal(view.length, MAX_TRANSCRIPT_ENTRIES);
  assert.equal(view[0].text, 'line 50');
  assert.equal(view.at(-1).text, 'line 249');
});

test('a transcript never holds more than 64 KB of text, even when every line is long', () => {
  const transcript = new ChildTranscript({ now: () => 1 });
  for (let i = 0; i < 50; i += 1) transcript.user(`${i}:${'x'.repeat(3990)}`, i);
  const view = transcript.view();
  assert.ok(view.length < 50, 'the oldest lines were dropped');
  assert.ok(chars(view) <= MAX_TRANSCRIPT_CHARS, `${chars(view)} <= ${MAX_TRANSCRIPT_CHARS}`);
  assert.match(view.at(-1).text, /^49:/, 'the newest line is kept');
});

test('one huge line is clipped instead of breaking the bound', () => {
  const transcript = new ChildTranscript({ now: () => 1 });
  transcript.user('y'.repeat(40_000), 1);
  const view = transcript.view();
  assert.equal(view.length, 1);
  assert.ok(view[0].text.length < 10_000);
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
  const cp = new ChildProjector({ id: 'c1', name: 'Writer', now: () => 1 });
  cp.consume(ev('user/message', { role: 'user', source: { kind: 'time-context', form: 'snapshot' }, content: [{ type: 'text', text: 'Time sampled while preparing turn 1' }] }));
  cp.consume(ev('user/message', { role: 'user', source: { kind: 'skill-catalog', form: 'catalog' }, content: [{ type: 'text', text: '<available_skills>' }] }));
  cp.consume(ev('user/message', { role: 'user', source: { kind: 'user' }, content: [{ type: 'text', text: 'Write the notes.' }] }));
  assert.deepEqual(cp.transcript.view().map(entry => entry.text), ['Write the notes.']);
});

test('the projector records the helper task, its answer and its tool calls in order', () => {
  const cp = new ChildProjector({ id: 'c1', name: 'Writer', now: () => 1 });
  cp.consume(ev('user/message', { role: 'user', content: [{ type: 'text', text: 'Write the notes.' }] }));
  cp.consume({ type: 'assistant-stream', frame: { type: 'chunk', chunk: { type: 'text-delta', text: 'Writing' } } });
  cp.consume(ev('assistant/message', { message: { content: [{ type: 'text', text: 'Writing the notes.' }] } }));
  cp.consume(ev('tool/call', { callId: 'k1', name: 'pwsh', arguments: '{"command":"npm test"}' }));
  cp.consume(ev('tool/result', { message: { toolCallId: 'k1', content: [{ type: 'text', text: 'ok' }] } }));
  cp.consume(ev('turn/end', { reason: { kind: 'completed' } }));
  assert.deepEqual(cp.transcript.view().map(entry => [entry.role, entry.text]), [
    ['user', 'Write the notes.'],
    ['assistant', 'Writing the notes.'],
    ['tool', 'Running a command: npm test'],
  ]);
});
