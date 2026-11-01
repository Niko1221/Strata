/** Owner's GUI feedback round 3 (2026-11-01), items 1 to 14 that are UI: each pinned where the source can show it. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const read = path => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');
const models = read('src/components/NativeModels.tsx');

test('1: the Models page carries no DFlash2 paragraph, no per-card line and no picture-reader sentence; the Drafting dropdown stays', () => {
  assert.ok(!/MTP is the default\. DFlash2/.test(models), 'the DFlash2 paragraph is gone');
  assert.ok(!/dflash2Support/.test(models), 'the per-card line is gone');
  assert.ok(!/engine installed here does not include the picture reader/.test(models), 'the picture-reader sentence is gone');
  assert.match(models, /<Select ariaLabel=\{tr\("Drafting"\)\}/, 'the dropdown stays');
});
