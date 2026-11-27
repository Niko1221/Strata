// 0.2.23 custom models: the person's own compatible .strata and .gguf files, used in place (CUSTOM-MODELS-0223, GUI part).
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { build } from 'esbuild';
import { browseLocalModels, inspectLocalModel, createLocalModels, customLabel, cleanLabel } from '../server/local-models.mjs';
import { ggufFixture } from './helpers/local-gguf.mjs';

const fixture = t => { const root = mkdtempSync(join(tmpdir(), 'strata-custom-model-')); t.after(() => rmSync(root, { recursive: true, force: true })); return root; };
const MAGIC = Buffer.from([0x53, 0x54, 0x52, 0x41, 0x54, 0x41, 0x00, 0x01]);
/** A synthetic .strata file: the 8-byte magic then zeros up to `size` (4 KiB or 16 KiB headers are the two real header sizes). */
const strataFixture = (file, size = 4096) => { const bytes = Buffer.alloc(size); MAGIC.copy(bytes); writeFileSync(file, bytes); return bytes; };
const bundle = async entry => { const out = await build({ stdin: { contents: `export * from '${entry}';`, resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } }); return import(`data:text/javascript;base64,${Buffer.from(out.outputFiles[0].text).toString('base64')}`); };
const src = async path => readFile(new URL(`../${path}`, import.meta.url), 'utf8');

test('a local .strata file (4 KiB and 16 KiB headers) is read by its magic and size only', t => {
  const root = fixture(t);
  for (const size of [4096, 16384]) {
    const file = join(root, `My Finetune ${size}.strata`);
    strataFixture(file, size);
    const preview = inspectLocalModel(file);
    assert.equal(preview.variant.quant, 'STRATA');
    assert.equal(preview.variant.local.architecture, 'strata');
    assert.equal(preview.variant.local.shards, 1);
    assert.equal(preview.variant.source.bytes, size);
    assert.equal(preview.variant.name, `My Finetune ${size}`);
    assert.ok(preview.variant.local.path.endsWith('.strata'));
  }
});

test('a .strata file without the magic, a cut-short file, a helper shard and a folder are refused', t => {
  const root = fixture(t);
  const bad = join(root, 'bad.strata'); writeFileSync(bad, Buffer.alloc(4096, 7));
  assert.throws(() => inspectLocalModel(bad), /STRATA header/);
  const tiny = join(root, 'tiny.strata'); writeFileSync(tiny, MAGIC);
  assert.throws(() => inspectLocalModel(tiny), /empty or cut short/);
  const vision = join(root, 'Inkling-2-Mini-Vision.strata'); strataFixture(vision);
  assert.throws(() => inspectLocalModel(vision), /picture reader or helper/);
  mkdirSync(join(root, 'dir.strata'));
  assert.throws(() => inspectLocalModel(join(root, 'dir.strata')), /rather than its folder/);
  assert.throws(() => inspectLocalModel(join(root, 'missing.strata')), /unavailable/);
});

test('the name of a .strata model is one printable line: no path pieces, at most 160 characters', t => {
  const root = fixture(t);
  const file = join(root, `a  b  ${'x'.repeat(200)}.strata`);
  strataFixture(file);
  const name = inspectLocalModel(file).variant.name;
  assert.equal(name.length, 160); assert.doesNotMatch(name, /[\u0000-\u001f\\/]/); assert.match(name, /^a b x+$/);
  assert.equal(cleanLabel('a\u0007b'), 'a b');
  assert.equal(cleanLabel('../../etc/passwd'), '.. .. etc passwd');
});

test('folder browsing lists .strata and .gguf files and still hides other files', t => {
  const root = fixture(t);
  strataFixture(join(root, 'model.strata')); ggufFixture(join(root, 'other.gguf')); writeFileSync(join(root, 'notes.txt'), 'keep'); mkdirSync(join(root, 'sub'));
  const listing = browseLocalModels(root);
  assert.deepEqual(listing.entries.map(e => e.name), ['sub', 'model.strata', 'other.gguf']);
  assert.equal(listing.entries.find(e => e.name === 'model.strata').kind, 'model');
});

test('GGUF files of Strata engine architectures (inkling2, inkling2_loop, hy4) are accepted; unrelated ones keep the Custom Model API guidance', t => {
  const root = fixture(t);
  for (const arch of ['inkling2', 'inkling2_loop', 'hy4', 'hy4flash', 'qwen35moe', 'qwen4exp']) {
    const file = join(root, `${arch}.gguf`); ggufFixture(file, { arch });
    assert.equal(inspectLocalModel(file).variant.local.architecture, arch);
  }
  const clip = join(root, 'clip.gguf'); ggufFixture(clip, { arch: 'clip' });
  assert.throws(() => inspectLocalModel(clip), /supported by the Strata engine; add it as a Custom model API \(llama-server\)/);
});

test('removing a local reference keeps the person\'s .strata and .gguf files byte for byte', t => {
  const root = fixture(t);
  const a = join(root, 'a.strata'), b = join(root, 'b.gguf');
  const bytesA = strataFixture(a, 16384), bytesB = ggufFixture(b);
  const service = createLocalModels(join(root, 'data'));
  const first = inspectLocalModel(a), second = inspectLocalModel(b);
  service.add(first); service.add(second);
  assert.equal(service.list().length, 2);
  service.remove(first.variant.id); service.remove(second.variant.id);
  assert.equal(service.list().length, 0);
  assert.deepEqual(readFileSync(a), bytesA); assert.deepEqual(readFileSync(b), bytesB);
  assert.equal(statSync(a).size, 16384);
});

test('the label sent to setup is a short printable name, never a path', () => {
  assert.equal(customLabel('  My merge v2  '), 'My merge v2');
  assert.equal(customLabel('C:\\Users\\me\\models\\x.gguf'), 'C Users me models x.gguf');
  assert.equal(customLabel('../../x'), 'x');
  assert.equal(customLabel('a/b\\c'), 'a b c');
  assert.equal(customLabel('line\none\u0000two'), 'line one two');
  assert.equal(customLabel('x'.repeat(200)).length, 80);
  for (const empty of ['', '   ', '...', undefined, null, 5, {}]) assert.equal(customLabel(empty), undefined);
});

const size = (id, model, gb, extra = {}) => ({ id, model, about: '', download_gb: gb, ram_needed_gb: gb + 2, experts_gb: gb - 1, on_this_pc: null, installed: true, ...extra });
const families = () => [
  { family: 'qwen', title: 'Qwen3.8-Flash-Next', about: '', experimental: false, images: true, sizes: [size('q2_0', 'Q2_0', 10)] },
  { family: 'qwen36', title: 'Qwen3.6-35B-A3B', about: '', experimental: false, images: false, sizes: [size('qwen36-ud-iq4_xs', 'UD-IQ4_XS', 20)] },
  { family: 'hy4-flash', title: 'HY-4 Flash', about: '', experimental: false, images: true, sizes: [size('hy4-flash-sq-64', 'SQ-64', 60, { hy4: { placement: { vram_gb: 1, ram_gb: 2, ssd_gb: 3 }, mode: 'ram_vram_zero_ssd', recommended: true, unified: false, engram_gb: 1 } })] },
  { family: 'inkling-2-mini', title: 'Inkling-2-Mini', about: '', experimental: false, images: true, sizes: [size('inkling-2-mini-sq-20', 'SQ-20', 20, { inkling: { placement: { vram_gb: 1, ram_gb: 2, ssd_gb: 3 }, mode: 'full_vram', recommended: true, unified: false, context: 32768, kv_gb: 1 } }), size('inkling-2-mini-sq-15', 'SQ-15', 15)] },
];
const row = (model, base_family, base_model, extra = {}) => ({ model, config: `strata-${model}.json`, model_name: `${model} alias`, ready: true, custom: true, display_name: `Mine ${model}`, base_family, base_model, custom_bytes: 12 * 1024 ** 3, ...extra });
const official = (model) => ({ model, config: `strata-${model}.json`, model_name: model, ready: true });

test('installed custom builds join their parent model in the version list, beside the official builds', async () => {
  const { nativeCatalogModels, catalogVersionLabel } = await bundle('./src/lib/native-catalog');
  const installed = [official('q2_0'), official('qwen36-ud-iq4_xs'), official('hy4-flash-sq-64'), official('inkling-2-mini-sq-20'),
    row('custom-q38', 'qwen', 'Q2_0'), row('qwen36-custom-a', 'qwen36', 'UD-IQ4_XS'), row('hy4-flash-custom-a', 'hy4-flash', 'SQ-64'),
    row('inkling-2-mini-custom-a', 'inkling-2-mini', 'SQ-20'), row('inkling-2-mini-custom-b', 'inkling-2-mini', 'SQ-15', { display_name: 'Merge B' })];
  const groups = nativeCatalogModels(families(), installed);
  const byFamily = Object.fromEntries(groups.map(g => [g.family, g]));
  assert.deepEqual(groups.map(g => g.family), ['qwen3.8', 'qwen3.6', 'hy4-flash', 'inkling-2-mini']);
  assert.deepEqual(byFamily['qwen3.8'].sizes.map(s => s.id), ['q2_0', 'custom-q38']);
  assert.deepEqual(byFamily['qwen3.6'].sizes.map(s => s.id), ['qwen36-ud-iq4_xs', 'qwen36-custom-a']);
  assert.deepEqual(byFamily['hy4-flash'].sizes.map(s => s.id), ['hy4-flash-sq-64', 'hy4-flash-custom-a']);
  assert.deepEqual(byFamily['inkling-2-mini'].sizes.map(s => s.id), ['inkling-2-mini-sq-20', 'inkling-2-mini-sq-15', 'inkling-2-mini-custom-a', 'inkling-2-mini-custom-b']);
  for (const g of groups) {
    const custom = g.sizes.filter(s => s.custom), official = g.sizes.filter(s => !s.custom);
    assert.ok(custom.length && official.length, `${g.family} lists both kinds`);
    for (const c of custom) {
      assert.equal(c.installed, true); assert.equal(c.offered, true); assert.equal(c.edition, 'Custom');
      assert.equal(c.customBytes, 12 * 1024 ** 3); assert.equal(c.download_gb, 12);
      assert.equal(catalogVersionLabel(c), c.displayName);
      assert.ok(!official.some(o => o.id === c.id), 'distinct ids');
    }
  }
  const c = byFamily['inkling-2-mini'].sizes.find(s => s.id === 'inkling-2-mini-custom-a');
  assert.equal(c.model, 'SQ-20'); assert.equal(c.sourceFamily, 'inkling-2-mini'); assert.equal(c.inkling.recommended, false);
  assert.equal(byFamily['hy4-flash'].sizes[1].hy4.recommended, false);
  assert.equal(byFamily['hy4-flash'].sizes[0].hy4.recommended, true, 'the official pick is unchanged');
});

test('a custom build\'s memory figures follow its real file size, not the official size of the build it is closest to', async () => {
  const { nativeCatalogModels } = await bundle('./src/lib/native-catalog');
  const groups = nativeCatalogModels(families(), [row('custom-q38', 'qwen', 'Q2_0', { custom_bytes: 14 * 1024 ** 3 })]);
  const [base, custom] = groups[0].sizes;
  assert.equal(custom.download_gb, 14);
  assert.equal(+(custom.ram_needed_gb - base.ram_needed_gb).toFixed(3), 4);    // 14 GB file against the 10 GB official build
  assert.equal(+(custom.experts_gb - base.experts_gb).toFixed(3), 4);
});

test('the official list is unchanged when nothing custom is installed, and an unknown family is never listed', async () => {
  const { nativeCatalogModels } = await bundle('./src/lib/native-catalog');
  const plain = nativeCatalogModels(families());
  assert.deepEqual(nativeCatalogModels(families(), [official('q2_0')]), plain);
  assert.deepEqual(nativeCatalogModels(families(), [row('custom-x', 'llama', 'Q4_0')]), plain);
  assert.deepEqual(nativeCatalogModels([], [row('custom-q38', 'qwen', 'Q2_0')]), [], 'no catalog yet: nothing is made up');
});

test('the display name starts from the check, else the file name, and is cleaned', async () => {
  const { suggestedDisplayName, cleanDisplayName, customTarget, customGroupOf, CUSTOM_FILE_TYPES } = await bundle('./src/lib/custom-models');
  assert.equal(suggestedDisplayName('Orca Merge', '/x/y.gguf'), 'Orca Merge');
  assert.equal(suggestedDisplayName('', '/models/My-Merge-00001-of-00003.gguf'), 'My-Merge');
  assert.equal(suggestedDisplayName(undefined, 'C:\\models\\Inkling Tune.strata'), 'Inkling Tune');
  assert.equal(suggestedDisplayName('  ', ''), '');
  assert.equal(cleanDisplayName('a<b>:c|d'), 'a b c d');
  assert.equal(cleanDisplayName('z'.repeat(200)).length, 80);
  assert.equal(customTarget('/m/x.strata', '/m', true), '/m/x.strata');
  assert.equal(customTarget('/m', '/m', true), '/m');
  assert.equal(customTarget('/m/x.gguf', '/m', false), '/m');
  assert.deepEqual([customGroupOf('qwen'), customGroupOf('unsloth'), customGroupOf('qwen36'), customGroupOf('hy4-flash'), customGroupOf('inkling-2-mini'), customGroupOf('ornith'), customGroupOf('x')], ['qwen3.8', 'qwen3.8', 'qwen3.6', 'hy4-flash', 'inkling-2-mini', 'ornith', undefined]);
  assert.deepEqual(CUSTOM_FILE_TYPES['hy4-flash'], ['strata']);
  assert.deepEqual(CUSTOM_FILE_TYPES['inkling-2-mini'], ['strata', 'gguf']);
});

test('the chat picker names a custom build by its display name', async () => {
  const { pickerNativeModels, pickerNativeName } = await bundle('./src/lib/model-picker');
  const rows = [{ model: 'custom-a', model_name: 'alias', custom: true, display_name: 'My Merge', ready: true }, { model: 'q2_0', model_name: 'Qwen3.8-Flash-Next-q2_0', ready: true }];
  assert.equal(pickerNativeName(rows[0]), 'My Merge'); assert.equal(pickerNativeName(rows[1]), 'Qwen3.8-Flash-Next-q2_0');
  assert.deepEqual(pickerNativeModels(rows, [{ label: 'My Merge' }]).map(r => r.model), ['q2_0']);
});

test('the import screens say .strata and .gguf, list all four engine families and keep llama-server guidance for other architectures', async () => {
  const screen = await src('src/components/NativeLocalModel.tsx'), dialog = await src('src/components/LocalModelImport.tsx'), picker = await src('src/components/VersionPicker.tsx'), models = await src('src/components/NativeModels.tsx');
  for (const text of [screen, models, picker]) assert.match(text, /Use local model files \(\.strata or \.gguf\)…/);
  for (const family of ['Qwen3.8-Flash-Next', 'Qwen3.6-35B-A3B', 'HY-4 Flash', 'Inkling-2-Mini']) assert.ok(screen.includes(family), family);
  assert.doesNotMatch(screen, /Use a local GGUF/);
  assert.match(dialog, /\\\.\(gguf\|strata\)\$/);
  assert.match(screen, /Custom model API instead: start it with llama-server/, 'unrelated architectures keep the Custom Model API guidance');
  assert.match(screen, /candidate_selection\?\.suggested_label/);
  assert.match(screen, /custom_label: customName/);
  assert.match(screen, /aria-label=\{tr\("Display name"\)\}/);
  for (const text of [screen, dialog, picker, models]) assert.doesNotMatch(text, /jailbreak|uncensor|abliterat|nsfw/i, 'neutral wording');
});

test('the host passes only a cleaned label to setup, never reuses the official install for a custom build, and removal adds no file deletion', async () => {
  const control = await src('server/native-control.mjs');
  assert.match(control, /import \{ customLabel \} from '\.\/local-models\.mjs'/);
  assert.match(control, /if\(custom\)allowed\.push\('custom_label'\)/);
  assert.match(control, /const existing=custom\?undefined:/);
  const start = control.indexOf('async function removeModel'), end = control.indexOf('/* The configuration a load starts the engine with', start);
  const remove = control.slice(start, end);
  assert.ok(start > 0 && end > start);
  assert.match(remove, /strata_remove/);
  assert.doesNotMatch(remove, /rmSync|unlinkSync|rmdirSync|\brm\(|rename|copyFileSync|writeFileSync/, 'the host deletes nothing itself');
  assert.match(remove, /Your model files were left in place/);
});

test('every instance of a custom build is listed under its display name with a Custom badge, and its id is the /v1/models name', async () => {
  const control = await src('server/native-control.mjs');
  assert.match(control, /\.\.\.\(entry\.custom===true\?\{custom:true\}:\{\}\),label:\(\(entry\.custom===true&&\(entry\.display_name\|\|entry\.model_name\)\)\|\|friendlyModelName/);
  assert.match(control, /diskBytes:entry\.diskBytes\?\?entry\.disk_bytes\?\?\(entry\.custom===true/);
  const chat = await src('src/components/ChatModelPicker.tsx');
  assert.match(chat, /version-tag--custom/);
});
