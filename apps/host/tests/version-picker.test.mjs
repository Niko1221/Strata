/** The version list: for one base model, the recommendation and the downloaded versions on top, every other version once below, a fit word per row. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const { code } = await transform(await readFile(new URL('../src/lib/native-fit.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const { versionGroups, versionFit, recommendedVersions, PICK_NOTE } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const source = await readFile(new URL('../src/components/VersionPicker.tsx', import.meta.url), 'utf8');

const v = (sourceFamily, model, download_gb, extra = {}) => ({ id: sourceFamily === 'qwen' ? model.toLowerCase() : `${sourceFamily}-${model.toLowerCase()}`, sourceFamily, model, download_gb, memory_gb: download_gb * 0.62, ...extra });
const qwen38 = [v('qwen', 'Q2_0', 66.4), v('qwen', 'IQ2_XS', 68), v('qwen', 'IQ3_XXS', 75.8), v('qwen', 'IQ3_S', 83.6), v('swift', 'IQ2_XS', 68), v('swift', 'IQ3_S', 83.6), v('coder', 'IQ1_M', 58.4), v('unsloth', 'UD-Q4_K_XL', 111.3), v('gyro', 'Gyro-S', 58.5), v('gyro', 'Gyro-M', 92)];
const ids = list => list.map(x => x.id);

test('top: the recommendation and the downloaded versions; below: all the others; nothing doubled, nothing lost', () => {
  const have = new Set(['gyro-gyro-s', 'swift-iq2_xs']);
  const { top, more } = versionGroups(qwen38, new Set(), have, 'iq3_s');
  assert.deepEqual(ids(top), ['iq3_s', 'gyro-gyro-s', 'swift-iq2_xs'], 'the pick first, then the downloaded ones by size');
  assert.equal(top.length + more.length, qwen38.length);
  assert.deepEqual([...ids(top), ...ids(more)].sort(), ids(qwen38).sort());
  assert.ok(!more.some(x => top.includes(x)), 'a version at the top never appears again below');
  assert.deepEqual(more.map(x => x.download_gb), [...more.map(x => x.download_gb)].sort((a, b) => a - b), 'the others by size');
});

test('IQ3_S downloaded and recommended: it is at the top once, as downloaded, and nowhere else', () => {
  const pick = { family: 'qwen', model: 'IQ3_S', why: '64 GB tier' };
  const pickId = recommendedVersions(qwen38, pick).find(r => r.r.note === PICK_NOTE)?.v.id;
  assert.equal(pickId, 'iq3_s');
  const { top, more } = versionGroups(qwen38, new Set(), new Set(['iq3_s']), pickId);
  assert.equal(top[0].id, 'iq3_s');
  assert.equal([...top, ...more].filter(x => x.id === 'iq3_s').length, 1);
  assert.match(source, /recommended = v\.id === pickId && !have/, 'the Recommended tag is not shown once it is downloaded');
});

test('no downloads and no pick: everything is in the one list below, and the first-run page leaves out what it shows elsewhere', () => {
  const { top, more } = versionGroups(qwen38, new Set(['iq3_s', 'gyro-gyro-s']), undefined, undefined);
  assert.equal(top.length, 0);
  assert.equal(more.length, qwen38.length - 2);
  assert.ok(!more.some(x => x.id === 'iq3_s' || x.id === 'gyro-gyro-s'));
});

test('a row says whether the version fits this PC, from the same figures as the card', () => {
  const big = { id: 'x', download_gb: 111, memory_gb: 78 }, small = { id: 'y', download_gb: 66, memory_gb: 35 };
  assert.equal(versionFit(small, { ram_gb: 128, vram_gb: 24 }), 'fits');
  assert.equal(versionFit(big, { ram_gb: 64, vram_gb: 12 }), 'ssd', 'part of the experts read from the SSD');
  assert.equal(versionFit({ ...small, on_this_pc: 'does not fit' }, { ram_gb: 128, vram_gb: 24 }), 'no', "the engine's own verdict wins");
  assert.equal(versionFit(small, null), 'unknown', 'no hardware figures: no claim');
  assert.match(source, /'Fits with SSD streaming'/);
  assert.ok(!/nativeVersionLabel|plainLabel/.test(source), 'no Fastest, Compact, High quality words');
});

test("the list is for one base model: the page passes only the chosen model's versions", async () => {
  const screen = await readFile(new URL('../src/components/NativeModels.tsx', import.meta.url), 'utf8');
  assert.match(screen, /<VersionPicker [^>]*versions=\{selectedFamily\?\.sizes \?\? \[\]\}/);
  assert.ok(!/models=\{families\}/.test(screen));
});

test('the picker has no maker groups, and the technical lines sit behind Show details', () => {
  assert.ok(!/VERSION_GROUPS|groupOpen|version-group/.test(source), 'no Swift / GSQ-RCO / Unsloth group headings');
  assert.ok(!/\(GSQ-RCO\)/.test(source), 'the maker is not part of the main name');
  assert.match(source, /details \? <span className="version-row__tech">/, 'maker, RAM + VRAM and the SSD figure are shown only with details on');
});
