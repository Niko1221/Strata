/** F08: huge tool-call blocks must shrink in the summariser copy; originals stay intact. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { clipMessage, estimateMessage, splitMessages } from '../server/agent/context-budget.mjs';

test('clipMessage bounds a giant tool-call for summary without mutating the original', () => {
  const args = JSON.stringify({ blob: 'x'.repeat(80_000) });
  const original = {
    role: 'assistant',
    content: [{ type: 'tool-call', id: 'c1', name: 'read_file', arguments: args }],
  };
  const before = structuredClone(original);
  const clipped = clipMessage(original, 1024, 1);
  assert.ok(estimateMessage(clipped) <= 1024 + 64, `clipped estimate ${estimateMessage(clipped)}`);
  assert.equal(clipped.content[0].type, 'text');
  assert.match(clipped.content[0].text, /tool-call read_file/);
  assert.deepEqual(original, before, 'original history is untouched');
  assert.equal(original.content[0].arguments.length, args.length);
});

test('splitMessages keeps a huge tool-call run under the given budget', () => {
  const args = 'y'.repeat(100_000);
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'please read' }] },
    { role: 'assistant', content: [{ type: 'tool-call', id: 'c', name: 'read', arguments: args }] },
    { role: 'tool', content: [{ type: 'tool-result', tool_use_id: 'c', output: 'z'.repeat(50_000) }] },
  ];
  const runs = splitMessages(messages, 1024, 1);
  for (const run of runs) {
    const cost = run.reduce((n, m) => n + estimateMessage(m), 0);
    assert.ok(cost <= 1024 + 128, `run cost ${cost}`);
  }
  assert.equal(messages[1].content[0].arguments.length, 100_000, 'source tool args unchanged');
});
