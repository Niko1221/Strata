// #7134: Ornith-1.5-35B-A3B runs on the Qwen3.6 engine path (arch qwen35moe) but is its own model. Its sizes are a display group of their own
// in every list (never under Qwen3.6, which the "35b-a3b" in its title would match), the group has no KV-format choice like Qwen3.6, its short
// name is "Ornith Q4_K_M", and a client that sends one of its model names gets the same "not loaded" answer as for the other Strata ids.
// Patch and tests of the Control side by eSPiYa (PR #7140).
import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { looksLikeStrataModelId, sameModelName } from '../server/model-ids.mjs';

const compile = async rel => {
  const built = await build({ entryPoints: [fileURLToPath(new URL(rel, import.meta.url))], bundle: true, write: false, platform: 'node', format: 'esm', loader: { '.json': 'json' } });
  return import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
};
const catalog = await compile('../src/lib/native-catalog.ts');
const names = await compile('../src/lib/native-names.ts');

const size = (id, model) => ({ id, model });
const FAMILIES = [
  { family: 'qwen', title: 'Qwen3.8-Flash-Next', images: true, sizes: [size('iq3_s', 'IQ3_S')] },
  { family: 'qwen36', title: 'Qwen3.6-35B-A3B', images: true, sizes: [size('qwen36-ud-q4_k_xl', 'UD-Q4_K_XL'), size('qwen36-q8_0', 'Q8_0')] },
  { family: 'ornith', title: 'Ornith-1.5-35B-A3B', images: true,
    sizes: ['Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0'].map(q => size(`ornith-1.5-35b-a3b-${q.toLowerCase()}`, q)) },
];

test('Ornith is a group of its own, after Qwen3.6, and Qwen3.6 keeps only its own sizes', () => {
  const groups = catalog.nativeCatalogModels(FAMILIES);
  assert.deepEqual(groups.map(g => g.title), ['Qwen3.8-Flash-Next', 'Qwen3.6-35B-A3B', 'Ornith-1.5-35B-A3B']);
  assert.deepEqual(groups[1].sizes.map(s => s.id), ['qwen36-ud-q4_k_xl', 'qwen36-q8_0']);
  assert.deepEqual(groups[2].sizes.map(s => s.model), ['Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0']);
  assert.ok(groups[2].sizes.every(s => s.sourceFamily === 'ornith' && s.images === true));
  assert.equal(groups[2].images, true);
});

test('an older Control without Ornith: no Ornith group, Qwen3.6 as before', () => {
  const groups = catalog.nativeCatalogModels(FAMILIES.slice(0, 2));
  assert.deepEqual(groups.map(g => g.title), ['Qwen3.8-Flash-Next', 'Qwen3.6-35B-A3B']);
});

test('the Ornith group has no KV-format choice, as Qwen3.6 has none', () => {
  const groups = catalog.nativeCatalogModels(FAMILIES);
  assert.equal(catalog.hasKvChoice(groups[2].family), false);
  assert.equal(catalog.hasKvChoice(groups[0].family), true);
});

test('the short name of an installed Ornith size', () => {
  assert.equal(names.nativeShortName('ornith-1.5-35b-a3b-q4_k_m'), 'Ornith Q4_K_M');
  assert.equal(names.nativeShortName('qwen36-q8_0'), 'Qwen3.6 Q8_0');
});

test('the gateway knows Ornith model names as Strata ids', () => {
  assert.equal(looksLikeStrataModelId('ornith-1.5-35b-a3b-q4_k_m'), true);
  assert.equal(looksLikeStrataModelId('Ornith-1.5-35B-A3B-Q8_0'), true);
  assert.equal(looksLikeStrataModelId('gpt-4o'), false);
  assert.equal(sameModelName('ornith-1.5-35b-a3b-q4_k_m', 'q4_k_m'), true);
  assert.equal(sameModelName('ornith-1.5-35b-a3b-q4_k_m', 'q8_0'), false);
});
