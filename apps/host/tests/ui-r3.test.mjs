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

test('2: pictures are on by default; a missing reader says nothing wordy and saves nothing', () => {
  const host = read('server/native-control.mjs');
  // The row is shown for every model that can read pictures; without the reader it is off, disabled and carries a tooltip, no paragraph.
  assert.match(models, /const noReader = !!selectedSize\?\.imagesCapable && !selectedSize\.images/);
  assert.match(models, /selectedSize\?\.images \|\| noReader \? <><div className="native-image-reading"><label title=\{noReader \? tr\("Needs an engine update"\) : undefined\}>/);
  assert.match(models, /checked=\{!noReader && vision !== 'no'\} disabled=\{noReader \|\| !!active\}/);
  assert.match(models, /else \{\s*setVision\('yes'\);|setVision\('yes'\);/, 'a model that is not downloaded starts with pictures on');
  // A load without a reader used to save "off" (ui_vision_enabled false), and the model stayed off after the engine got the reader.
  assert.match(models, /\.\.\.\(noReader \? \{ visionUnavailable: true \} : \{\}\)/);
  assert.match(host, /const noReader=args\.visionUnavailable===true;/);
  assert.match(host, /const visionPreferences=args\.vision!==undefined&&!noReader\?/, 'nothing is saved when the reader is missing');
  assert.match(host, /runtimeVision\(config,noReader\?'no':args\.vision\)/);
  // An "off" that sits on a model without the picture files is not a choice of the person: it starts on.
  assert.match(models, /installed\.visionEnabled === false && installed\.visionPrepared !== false \? 'no'/);
});
