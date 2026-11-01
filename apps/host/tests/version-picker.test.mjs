/** The version list: every version once, the pick first, sizes in order, the maker and the memory behind Details. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const { code } = await transform(await readFile(new URL('../src/lib/native-fit.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const { versionSections, recommendedVersions, PICK_NOTE } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const source = await readFile(new URL('../src/components/VersionPicker.tsx', import.meta.url), 'utf8');

const v = (sourceFamily, model, download_gb, extra = {}) => ({ id: sourceFamily === 'qwen' ? model.toLowerCase() : `${sourceFamily}-${model.toLowerCase()}`, sourceFamily, model, download_gb, ...extra });
const qwen38 = { family: 'qwen3.8', title: 'Qwen3.8-Flash-Next', sizes: [v('qwen', 'Q2_0', 66.4), v('qwen', 'IQ2_XS', 68), v('qwen', 'IQ3_XXS', 75.8), v('qwen', 'IQ3_S', 83.6), v('swift', 'IQ2_XS', 68), v('swift', 'IQ3_S', 83.6), v('coder', 'IQ1_M', 58.4), v('unsloth', 'UD-Q4_K_XL', 111.3), v('gyro', 'Gyro-S', 58.5), v('gyro', 'Gyro-M', 92)] };
const qwen36 = { family: 'qwen3.6', title: 'Qwen3.6-35B-A3B', sizes: [v('qwen36', 'UD-Q4_K_XL', 22.9), v('qwen36', 'Q8_0', 37.8)] };
const deepseek = { family: 'deepseek41', title: 'DeepSeek-V4.1-Flash', sizes: [v('deepseek41', 'IST-DAS-2.00bpw', 256.8, { multi_file: true, disk_gb: 262.8 }), v('deepseek41', 'IST-DAS-1.75bpw', 239.3, { multi_file: true, disk_gb: 245.3 })] };
const models = [qwen38, qwen36, deepseek];
const all = models.flatMap(m => m.sizes);

test('every version is in the list exactly once (nothing lost, nothing doubled), DeepSeek and Qwen3.6 included', () => {
  const sections = versionSections(all, models, new Set(), 'iq3_s');
  const ids = sections.flatMap(s => s.versions.map(x => x.id));
  assert.equal(ids.length, all.length);
  assert.deepEqual([...ids].sort(), all.map(x => x.id).sort());
  assert.deepEqual(sections.map(s => s.title), ['Qwen3.8-Flash-Next', 'Qwen3.6-35B-A3B', 'DeepSeek-V4.1-Flash'], 'a model name, small, only where there is more than one model');
});

test('IQ3_S downloaded and recommended shows once, first, not again under a maker group', () => {
  const pick = { family: 'qwen', model: 'IQ3_S', why: '64 GB tier' };
  const pickId = recommendedVersions(all.filter(x => x.sourceFamily !== 'qwen36' && x.sourceFamily !== 'deepseek41'), pick).find(r => r.r.note === PICK_NOTE)?.v.id;
  assert.equal(pickId, 'iq3_s');
  const first = versionSections(all, models, new Set(), pickId)[0].versions;
  assert.equal(first[0].id, 'iq3_s');
  assert.equal(first.filter(x => x.model === 'IQ3_S' && x.sourceFamily === 'qwen').length, 1);
  assert.deepEqual(first.slice(1).map(x => x.download_gb), [...first.slice(1).map(x => x.download_gb)].sort((a, b) => a - b), 'the rest in order of download size');
});

test('one model: no section title; the first-run list leaves out what it shows elsewhere', () => {
  const sections = versionSections(qwen38.sizes, undefined, new Set(['iq3_s', 'gyro-gyro-s']));
  assert.equal(sections.length, 1);
  assert.equal(sections[0].title, '');
  assert.ok(!sections[0].versions.some(x => x.id === 'iq3_s' || x.id === 'gyro-gyro-s'));
  assert.equal(sections[0].versions.length, qwen38.sizes.length - 2);
});

test('the picker has no maker groups or collapsible headings, and the technical lines sit behind Show details', () => {
  assert.ok(!/VERSION_GROUPS|groupOpen|version-group/.test(source), 'no Swift / GSQ-RCO / Unsloth group headings');
  assert.ok(!/\(GSQ-RCO\)/.test(source), 'the maker is not part of the main name');
  assert.match(source, /details \? <span className="version-row__tech">/, 'maker, RAM + VRAM and the SSD figure are shown only with details on');
  assert.match(source, /Recommended/);
  assert.match(source, /Downloaded/);
});
