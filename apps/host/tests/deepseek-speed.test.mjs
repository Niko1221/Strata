/** The Speed label names the default the way the engine decides it: integrated / unified-memory GPU, or cards totalling 16 GiB or less. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const out = await build({ entryPoints: [fileURLToPath(new URL('../src/lib/deepseek-speed.ts', import.meta.url))], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent' });
const { fasterByDefault, speedOptions } = await import(`data:text/javascript;base64,${Buffer.from(out.outputFiles[0].text).toString('base64')}`);
const card = (vram_gb, extra = {}) => ({ index: 0, name: 'GPU', vram_gb, ...extra });

test('Strix Halo (an integrated / unified-memory GPU) defaults to Faster, whatever its memory figure says', () => {
  assert.equal(fasterByDefault([card(96, { uma: true })]), true, 'a 96 GB share is not "16 GiB or less", the integrated rule applies');
  assert.equal(fasterByDefault([card(0, { unified_gb: 128 })]), true);
  assert.equal(fasterByDefault([card(64, { integrated: true })]), true);
  assert.equal(fasterByDefault([card(24), card(96, { uma: true })]), true, 'an integrated card among the cards in use');
});

test('discrete cards: Faster at 16 GiB or less in total, Full quality above', () => {
  assert.equal(fasterByDefault([card(11.9)]), true, 'RTX 5070 / 3060 12 GB');
  assert.equal(fasterByDefault([card(15.99)]), true, 'a 16 GB card reports 16376 MiB');
  assert.equal(fasterByDefault([card(16)]), true);
  assert.equal(fasterByDefault([card(8), card(8)]), true, 'two 8 GB cards total 16');
  assert.equal(fasterByDefault([card(16.05)]), false);
  assert.equal(fasterByDefault([card(24)]), false);
  assert.equal(fasterByDefault([card(8), card(12)]), false);
  assert.equal(fasterByDefault([card(24, { usable: false }), card(12)]), true, 'a card the engine cannot use does not count');
  assert.equal(fasterByDefault([]), false);
  assert.equal(fasterByDefault(null), false);
});

test('one RDNA4 Radeon (1 x R9700, 32 GB) defaults to Faster: the CPU lane does most of the experts', () => {
  const r9700 = (extra = {}) => card(32, { vendor: 'amd', arch: 'gfx1201', ...extra });
  assert.equal(fasterByDefault([r9700()]), true);
  assert.equal(fasterByDefault([card(16, { vendor: 'amd', arch: 'gfx1200' })]), true);
  assert.equal(fasterByDefault([r9700({ arch: 'gfx1100' })]), false, 'an RX 7000 card with 32 GB is not RDNA4');
  assert.equal(fasterByDefault([r9700({ vendor: 'nvidia' })]), false);
  assert.equal(fasterByDefault([card(32)]), false, 'no arch in the report: the old rule');
  assert.equal(fasterByDefault([r9700(), r9700({ index: 1 })]), false, 'two cards are the other engine rule (they must hold half of the experts)');
  assert.equal(fasterByDefault([r9700(), r9700({ index: 1 })], false), false, 'the catalog says the two cards hold under half');
  const four = [0, 1, 2, 3].map(index => r9700({ index }));
  assert.equal(fasterByDefault(four, true), true, 'four R9700 hold most of the experts: the engine runs 5, so the label says Faster');
  assert.equal(fasterByDefault(four, null), false, 'no answer from the catalog: the old rule');
  assert.equal(fasterByDefault([r9700({ vendor: 'nvidia' }), r9700({ index: 1 })], true), false, 'only for RDNA4 cards');
  assert.deepEqual(speedOptions(four, true).map(o => o.label), ['Default (Faster on this PC)', 'Faster (default on this PC)', 'Full quality']);
  assert.equal(fasterByDefault([r9700(), r9700({ index: 1, usable: false })]), true, 'a card the engine cannot use does not count');
  assert.deepEqual(speedOptions([r9700()]).map(o => o.label), ['Default (Faster on this PC)', 'Faster (default on this PC)', 'Full quality']);
});

test('the labels say "(default on this PC)" beside the choice that is the default, once', () => {
  const faster = speedOptions([card(12)]).map(o => o.label);
  assert.deepEqual(faster, ['Default (Faster on this PC)', 'Faster (default on this PC)', 'Full quality']);
  const full = speedOptions([card(24)]).map(o => o.label);
  assert.deepEqual(full, ['Default (Full quality on this PC)', 'Faster', 'Full quality (default on this PC)']);
  const strix = speedOptions([card(96, { uma: true })]).map(o => o.label);
  assert.deepEqual(strix, ['Default (Faster on this PC)', 'Faster (default on this PC)', 'Full quality']);
});
