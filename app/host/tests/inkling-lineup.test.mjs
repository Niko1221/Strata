/**
 * The Inkling-2-Mini lineup of 0.2.23 (owner decision 2026-11-26) in the app:
 * Strata packs first (SQ-20, SQ-15, SQ-10), then GSQ-RCO IQ3_S, then Unsloth UD-Q4_K_XL; a quant that is no longer offered is listed only
 * while it is installed, last, marked "no longer offered"; DFlash 2 is the one drafting word.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { checkRunConfig } from '../server/run-config-check.mjs';

const bundle = async file => {
  const built = await build({ entryPoints: [fileURLToPath(new URL(file, import.meta.url))], bundle: true, write: false, platform: 'node', format: 'esm', loader: { '.json': 'json' } });
  return import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
};
const catalog = await bundle('../src/lib/native-catalog.ts');
const read = name => readFileSync(fileURLToPath(new URL(name, import.meta.url)), 'utf8');

const size = (model, extra = {}) => ({
  id: `inkling-2-mini-${model.toLowerCase()}`, model, about: '', download_gb: 12, ram_needed_gb: 12, experts_gb: 10, on_this_pc: 'fits', installed: false, images: true, ...extra,
});
const family = sizes => ({ family: 'inkling-2-mini', title: 'Inkling-2-Mini', about: '', images: true, sizes });
const OFFERED = ['SQ-20', 'SQ-15', 'SQ-10', 'GSQ-RCO-IQ3_S', 'UD-Q4_K_XL'];
const inkling = families => catalog.nativeCatalogModels(families).find(g => g.family === 'inkling-2-mini');

test('the five offered versions are listed in the owner order, whatever order Control sent them in', () => {
  const asSent = ['UD-Q4_K_XL', 'GSQ-RCO-IQ3_S', 'SQ-10', 'SQ-15', 'SQ-20'];
  assert.deepEqual(inkling([family(asSent.map(m => size(m)))]).sizes.map(s => s.model), OFFERED);
  assert.deepEqual(inkling([family(OFFERED.map(m => size(m)))]).sizes.map(s => s.model), OFFERED);
});

test('the Strata packs are ordered SQ-20, SQ-15, SQ-10, and the group comes before GSQ-RCO and Unsloth', () => {
  const sent = ['SQ-15', 'UD-Q4_K_XL', 'SQ-10', 'GSQ-RCO-IQ3_S', 'SQ-20'];
  assert.deepEqual(catalog.orderInklingSizes(sent.map(m => ({ model: m }))).map(s => s.model), OFFERED);
  assert.deepEqual(OFFERED.map(m => catalog.inklingRank(m)), [0, 1, 2, 3, 4]);
  assert.ok(catalog.inklingRank('SQ-99') > catalog.inklingRank('UD-Q4_K_XL') - 1, 'an unknown Strata pack goes after the known ones');
  assert.ok(catalog.inklingRank('UD-Q8_K_XL', false) > catalog.inklingRank('UD-Q4_K_XL'), 'a legacy quant is last');
});

test('a quant that is no longer offered is hidden unless it is installed, then it comes last and keeps its flag', () => {
  const sizes = [...OFFERED.map(m => size(m)), size('UD-IQ3_XXS', { offered: false, installed: false }), size('UD-Q8_K_XL', { offered: false, installed: true })];
  const shown = inkling([family(sizes)]).sizes;
  assert.deepEqual(shown.map(s => s.model), [...OFFERED, 'UD-Q8_K_XL']);
  assert.equal(shown.at(-1).offered, false);
  assert.equal(shown.at(-1).installed, true);
  assert.ok(shown.slice(0, 5).every(s => s.offered !== false));
});

test('labels: GSQ-RCO is its own edition, Unsloth keeps its, the Strata packs have none', () => {
  const group = inkling([family(OFFERED.map(m => size(m)))]);
  const by = Object.fromEntries(group.sizes.map(s => [s.model, s]));
  assert.equal(by['GSQ-RCO-IQ3_S'].edition, 'GSQ-RCO');
  assert.equal(by['UD-Q4_K_XL'].edition, 'Unsloth');
  assert.equal(by['SQ-20'].edition, '');
  assert.equal(catalog.catalogVersionLabel(by['GSQ-RCO-IQ3_S']), 'GSQ-RCO · IQ3_S');
  assert.equal(catalog.catalogVersionLabel(by['UD-Q4_K_XL']), 'Unsloth · UD-Q4_K_XL');
  assert.equal(catalog.nativeVersionLabel('SQ-20'), 'Lossless');
});

test('the version list builds its Inkling rows in that order, shows the credit and the no-longer-offered tag, and names GSQ-RCO', () => {
  const src = read('../src/components/VersionPicker.tsx');
  assert.match(src, /orderInklingSizes/);
  assert.match(src, /GSQ-RCO \$\{v\.model\.replace/);
  assert.match(src, /v\.credit \? tr\(v\.credit\)/);
  assert.match(src, /tr\("No longer offered"\)/);
  assert.match(src, /have && v\.update_available/);
  assert.match(src, /tr\("Update available"\)/);
  assert.match(src, /'GSQ-RCO GGUF'/);
  const native = read('../src/lib/native.ts');
  assert.match(native, /offered\?:boolean/);
  assert.match(native, /credit\?:string/);
});

test('DFlash 2 is the one drafting choice of Inkling-2-Mini; the old word is still accepted in a saved config', () => {
  const settings = read('../src/components/NativeSettings.tsx');
  assert.match(settings, /isInkling \? \['dflash2', 'off'\]/);
  assert.doesNotMatch(settings, /\['dspark', 'off'\]/);
  const config = { arch: 'inkling2_loop', exe: 'component:strata-inkling', args: ['--native', 'Inkling-2-Mini-SQ-20.strata'] };
  for (const drafting of ['dflash2', 'dspark', 'off']) {
    const checked = checkRunConfig(JSON.stringify({ ...config, drafting }));
    assert.equal(checked.ok, true, `drafting ${drafting}: ${JSON.stringify(checked.issues)}`);
  }
  assert.doesNotMatch(read('../server/native-control.mjs'), /DSpark Block-8 is Inkling/);
});

test('every locale has the new tag, so no language shows it in English', () => {
  for (const lang of ['en', 'de', 'es', 'fr', 'he', 'ja', 'ko', 'pt', 'ru', 'zh']) {
    const data = JSON.parse(read(`../src/locales/${lang}.json`));
    assert.ok(data.s9160 && data.s9161, `${lang} has s9160 and s9161`);
    if (lang !== 'en') assert.notEqual(data.s9160, 'No longer offered', lang);
  }
});
