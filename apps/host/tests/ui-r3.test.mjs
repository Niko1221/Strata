/** Owner's GUI feedback round 3 (2026-11-01), items 1 to 14 that are UI: each pinned where the source can show it. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const read = path => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');
const models = read('src/components/NativeModels.tsx');
/** Load a src/lib module (with its imports) the way the other tests do. */
async function lib(file) {
  const built = await build({ entryPoints: [fileURLToPath(new URL(`../${file}`, import.meta.url))], bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' }, logLevel: 'silent' });
  return import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
}

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

test('6: the conversation context opens at 128K for every configuration, or the model largest size when that is smaller; a warning, never a refusal', async () => {
  const { defaultContext, openingContext, contextIsTight, DEFAULT_CONTEXT } = await lib('src/lib/default-context.ts');
  assert.equal(DEFAULT_CONTEXT, 131072);
  assert.equal(defaultContext([8192, 32768, 65536, 131072, 262144]), 131072);
  assert.equal(defaultContext([8192, 32768, 65536]), 65536, 'a model whose largest size is under 128K opens at its largest');
  assert.equal(defaultContext([8192, 32768, 65536, 131072, 262144, 524288, 1048576]), 131072);
  assert.equal(defaultContext(undefined), 131072);
  assert.equal(openingContext(32768, [8192, 32768, 131072]), 131072, 'a saved 32K is the old default, not a choice');
  assert.equal(openingContext(65536, [8192, 32768, 65536, 131072]), 65536, 'a size that was chosen is kept');
  assert.equal(openingContext(undefined, [8192, 131072]), 131072);
  // What it costs: the conversation memory of 128K is 1.9 GB (int8) against 0.5 GB at 32K, so it is tight on cards of about 8 GB and below.
  assert.equal(contextIsTight(6, 131072), true);
  assert.equal(contextIsTight(8, 131072), true);
  assert.equal(contextIsTight(12, 131072), false, 'the 12 GB card of the owner is fine');
  assert.equal(contextIsTight(24, 131072), false);
  assert.equal(contextIsTight(8, 32768), false, '32K is the baseline: never a warning');
  assert.equal(contextIsTight(0, 131072), false, 'no card figure: no claim');
  assert.equal(contextIsTight(null, 131072), false);
  // The page opens with the default, ignores the engine recommended context, warns inline and never lowers by itself.
  assert.match(models, /useKept\('models\.context', DEFAULT_CONTEXT\)/);
  assert.ok(!/rec\.context/.test(models), 'the engine recommendation is not what the page opens with');
  assert.match(models, /setContext\(first\?\.multi_file \? 8192 : defaultContext\(native\.catalog\.contexts\)\)/);
  assert.match(models, /contextIsTight\(fitHardware\.vram_gb, context, kv\) \? <p className="small muted native-spill-note"><Icon name="warning"/);
  assert.ok(!/setContext\([^)]*contextIsTight/.test(models), 'the warning never changes the context');
  assert.match(models, /cache=\{\{ context, kv \}\}/, 'the version list follows the chosen context, as the card does');
  const guided = read('src/components/GuidedSetup.tsx');
  assert.match(guided, /pickedContext \?\? \(contextIsTight\(topCardGb, DEFAULT_CONTEXT\) \? OLD_DEFAULT_CONTEXT : defaultContext\(native\?\.catalog\?\.contexts\)\)/);
});
