// DeepSeek-V4.1-Flash's context sizes in the app, 8K up to the pack's own limit (Project Maya #442 by @needmorevram did the same in its dashboard).
// The pack's limit is `max_position_embeddings` in its ds4.txt; a pack without it behaves as before (up to 256K). Past 256K the row says it is untested.
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { mkdtempSync, writeFileSync, rmSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { packContextLimit, configContextLimit } from '../server/dsv41-context.mjs';
import { nativeFixture } from './helpers/native-fixture.mjs';

const read = rel => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8').replace(/\r\n/g, '\n');
const out = await build({ entryPoints: [fileURLToPath(new URL('../src/lib/deepseek-context.ts', import.meta.url))], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent' });
const { CONTEXT_CHOICES, TESTED_CONTEXT, NATIVE_CONTEXT, UNTESTED_NOTE, deepseekContexts, deepseekContextProblem, deepseekContextNote } = await import(`data:text/javascript;base64,${Buffer.from(out.outputFiles[0].text).toString('base64')}`);

function pack(line) {
  const dir = mkdtempSync(join(tmpdir(), 'strata-ds4-pack-'));
  writeFileSync(join(dir, 'ds4.txt'), `vocab_size 129280\n${line ? line + '\n' : ''}num_hidden_layers 43\n`);
  return dir;
}

test('the choices go up to the pack\'s limit: 1M for the official pack, less for a pack that says less, 256K while nothing says', () => {
  assert.deepEqual(CONTEXT_CHOICES, [8192, 32768, 65536, 131072, 262144, 524288, 1048576]);
  assert.deepEqual(deepseekContexts(1048576), CONTEXT_CHOICES);
  assert.deepEqual(deepseekContexts(524288), [8192, 32768, 65536, 131072, 262144, 524288]);
  assert.deepEqual(deepseekContexts(262144), [8192, 32768, 65536, 131072, 262144]);
  assert.deepEqual(deepseekContexts(null), [8192, 32768, 65536, 131072, 262144], 'a pack without ds4.txt behaves as before');
  assert.deepEqual(deepseekContexts(undefined), deepseekContexts(null));
});

test('one million is accepted and two million refused; the pack\'s limit is the last word', () => {
  assert.equal(deepseekContextProblem(1048576), '');
  assert.equal(deepseekContextProblem(524288), '');
  assert.match(deepseekContextProblem(2097152), /more than this model reads/);
  assert.notEqual(deepseekContextProblem(1048577), '');
  assert.notEqual(deepseekContextProblem(524288, 262144), '');
  assert.equal(deepseekContextProblem(262144, 262144), '');
  assert.notEqual(deepseekContextProblem(100), '');
});

test('past 256K the row says what it costs and that it is untested; 256K and below say nothing', () => {
  assert.equal(TESTED_CONTEXT, 262144);
  assert.equal(NATIVE_CONTEXT, 1048576);
  assert.equal(deepseekContextNote(262144), '');
  assert.equal(deepseekContextNote(8192), '');
  assert.equal(deepseekContextNote(524288), UNTESTED_NOTE);
  assert.equal(deepseekContextNote(1048576), 'Not tested yet. A context this long takes graphics memory from the model and reads long prompts slowly.');
});

test('the pack\'s limit is read from ds4.txt, and a config names its pack with --pack', () => {
  const full = pack('max_position_embeddings 1048576'), small = pack('max_position_embeddings 262144'), none = pack(null);
  try {
    assert.equal(packContextLimit(full), 1048576);
    assert.equal(packContextLimit(small), 262144);
    assert.equal(packContextLimit(none), null);
    assert.equal(packContextLimit(join(full, 'nowhere')), null);
    assert.equal(packContextLimit(null), null);
    assert.equal(configContextLimit({ args: ['--pack', small, '--max-context', '8192'] }), 262144);
    assert.equal(configContextLimit({ args: ['--native', 'm.gguf'] }), null, 'a Qwen model has no pack');
    assert.equal(configContextLimit({}), null);
  } finally { for (const d of [full, small, none]) rmSync(d, { recursive: true, force: true }); }
});

test('through the host: an installed DeepSeek model reports its pack\'s limit, a Qwen model none', async () => {
  const f = nativeFixture();
  try {
    const dir = pack('max_position_embeddings 1048576');
    const p = join(f.repo, 'strata-iq3_s.json');
    writeFileSync(p, JSON.stringify({ ...JSON.parse(readFileSync(p, 'utf8')), arch: 'deepseek41', args: ['--pack', dir, '--max-context', '8192'] }));
    await f.control.refresh();
    const models = f.control.state().installed?.models ?? [];
    const byId = Object.fromEntries(models.map(m => [m.model, m]));
    assert.equal(byId.iq3_s?.contextMax, 1048576);
    assert.equal(byId.iq2_xs?.contextMax, undefined);
    rmSync(dir, { recursive: true, force: true });
  } finally { await f.cleanup(); }
});

test('the Models screen honours the choice (it no longer forces 8K), offers the DeepSeek list, and shows the note on the existing note markup', () => {
  const screen = read('../src/components/NativeModels.tsx');
  assert.doesNotMatch(screen, /multi_file \? 8192 : context/, 'the chosen context is what is installed and loaded');
  assert.match(screen, /contextChoices\(selectedSize\?\.multi_file \? deepseekContexts\(installed\?\.contextMax\) : native\?\.catalog\?\.contexts \?\? \[8192, 32768, 65536, 131072, 262144\], context\)/);
  assert.match(screen, /selectedSize\?\.multi_file && deepseekContextNote\(context\) \? <p className="small muted native-spill-note"><Icon name="info" size=\{16\}\/>\{tr\(deepseekContextNote\(context\)\)\}<\/p> : null/);
  assert.match(screen, /if \(selectedSize\.multi_file\)\s+setContext\(8192\)/, 'the default for a DeepSeek version that is not installed stays 8K');
  assert.match(screen, /if \(installed\.context\)\s+setContext\(installed\.context\)/, 'an installed DeepSeek model shows and loads the context it was prepared with, whatever was kept from another model');
  assert.match(screen, /installed\?\.ready && context === installed\.context/, 'a load at the prepared context starts from the saved file; only a changed choice runs setup again');
});

test('the note is in the catalog in all nine languages, translated', () => {
  const find = lang => Object.values(JSON.parse(read(`../src/locales/fix-${lang}.json`)));
  assert.ok(find('en').includes(UNTESTED_NOTE));
  const en = JSON.parse(read('../src/locales/fix-en.json'));
  const id = Object.keys(en).find(k => en[k] === UNTESTED_NOTE);
  for (const lang of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr']) {
    const text = JSON.parse(read(`../src/locales/fix-${lang}.json`))[id];
    assert.ok(text && text !== UNTESTED_NOTE, `${lang} is translated`);
  }
});
