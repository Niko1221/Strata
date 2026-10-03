const test = require('node:test');
const assert = require('node:assert/strict');
const policy = require('./context.js');

test('compression leaves room for generation and starts before the limit', () => {
  assert.equal(policy.limit(65536, ''), 57336);
  assert.equal(policy.limit(65536, 16000), 49528);
  assert.equal(policy.limit(65536, 70000), 0);
  assert.ok(policy.limit(4096, '') < 4096);
});
test('compaction keeps recent turns together and can compact the last unsummarized turns', () => {
  const history = Array.from({length: 8}, (_, i) => ({role: i % 2 ? 'assistant' : 'user'}));
  assert.equal(policy.cut(history, 0), 4);
  assert.equal(policy.cut(history, 4), 8);
  assert.equal(policy.cut(history.slice(0, 2), 0), 2);
  assert.equal(policy.cut(history.slice(0, 7), 0), 2);
});
test('a stale or incomplete summary cannot hide stored history on reload', () => {
  const history = [{role: 'user'}];
  assert.deepEqual(policy.validMemory({through: 100, summary: 'stale'}, history), policy.emptyMemory());
  assert.deepEqual(policy.validMemory({through: 1, summary: ''}, history), policy.emptyMemory());
  const memory = {through: 1, summary: 'memory', count: 2};
  assert.equal(policy.validMemory(memory, history), memory);
});
test('image conversations keep their existing generation path', () => {
  assert.equal(policy.hasImages([{role: 'user', content: 'text'}]), false);
  assert.equal(policy.hasImages([{role: 'user', content: [{type: 'text', text: 'hi'}, {type: 'image_url'}]}]), true);
});
