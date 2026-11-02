// The DeepSeek model card in the Models screen names the free disk the set needs (from the set itself: 2.00 and 2.25 differ), in the existing card markup,
// and the version line says "{N} GB RAM + VRAM, plus about {disk} GB of disk". Other models keep their card.
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const read = rel => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8').replace(/\r\n/g, '\n');
const stdin = { contents: "import { createElement } from 'react'; import { renderToStaticMarkup } from 'react-dom/server'; import { PcFitCard } from './src/components/PcFitCard'; export { setTranslationLanguage } from './src/lib/i18n'; export const render = props => renderToStaticMarkup(createElement(PcFitCard, props));", resolveDir: fileURLToPath(new URL('..', import.meta.url)), loader: 'ts' };
const built = await build({ stdin, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json', '.tsx': 'tsx' }, jsx: 'automatic', logLevel: 'silent', banner: { js: "import { createRequire } from 'node:module'; const require = createRequire(import.meta.url);" } });
const dir = mkdtempSync(join(tmpdir(), 'strata-card-test-'));
writeFileSync(join(dir, 'card.mjs'), built.outputFiles[0].text);       // a file, not a data: URL: the bundled react-dom/server needs require()
const { render, setTranslationLanguage } = await import(pathToFileURL(join(dir, 'card.mjs')).href);
rmSync(dir, { recursive: true, force: true });
const fit = (spillGb, known = true) => ({ memoryGb: 250, vramUseGb: 10, ramUseGb: 240, spillGb, known });
const card = props => render({ gpuName: 'RTX 5070', ramGb: 64, vramGb: 12, ...props });

test('a DeepSeek card with no spill says what it needs on the disk, not just "Fits fully in memory"', () => {
  setTranslationLanguage('en');
  const html = card({ fit: fit(0), diskGb: 262.8 });
  assert.match(html, /Needs about 263 GB of free disk space\. What does not fit in memory is read from your SSD\./);
  assert.doesNotMatch(html, /Fits fully in memory/);
  assert.match(html, /class="pc-note"/, 'the existing note markup');
});

test('the disk comes from the set: 2.00 and 2.25 say different figures', () => {
  setTranslationLanguage('en');
  assert.match(card({ fit: fit(0), diskGb: 262.8 }), /about 263 GB/);
  assert.match(card({ fit: fit(0), diskGb: 279.8 }), /about 280 GB/);
});

test('a DeepSeek card that does spill keeps its spill line beside the disk line', () => {
  setTranslationLanguage('en');
  const html = card({ fit: fit(120), diskGb: 262.8 });
  assert.match(html, /Needs about 263 GB of free disk space/);
  assert.match(html, /About 120 GB will be read from the SSD, so answers can be a little slower\./);
});

test('every other model keeps its card as it was', () => {
  setTranslationLanguage('en');
  assert.match(card({ fit: fit(0) }), /Fits fully in memory\./);
  assert.match(card({ fit: fit(0), diskGb: null }), /Fits fully in memory\./);
  assert.doesNotMatch(card({ fit: fit(0) }), /free disk space/);
  assert.match(card({ fit: fit(30) }), /About 30 GB will be read from the SSD/);
});

test('the words are in all eight languages with the same slots', () => {
  const en = JSON.parse(read('../src/locales/fix-en.json'));
  const ids = Object.keys(en).filter(k => /^Needs about \{0\} GB of free disk space|^\{0\} RAM \+ VRAM, plus about \{1\} GB of disk$/.test(en[k]));
  assert.equal(ids.length, 2);
  for (const lang of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko']) {
    const cat = JSON.parse(read(`../src/locales/fix-${lang}.json`));
    for (const id of ids) {
      assert.ok(cat[id] && cat[id] !== en[id], `${lang} ${id} is translated`);
      assert.deepEqual((cat[id].match(/\{\d+\}/g) ?? []).sort(), (en[id].match(/\{\d+\}/g) ?? []).sort(), `${lang} ${id} slots`);
    }
  }
  setTranslationLanguage('de');
  assert.match(card({ fit: fit(0), diskGb: 262.8 }), /Braucht etwa 263 GB freien Speicherplatz/);
  setTranslationLanguage('en');
});

test('the Models screen passes the set\'s disk to the card, and the version line states it for DeepSeek only', () => {
  const screen = read('../src/components/NativeModels.tsx');
  assert.match(screen, /<PcFitCard [^>]*diskGb=\{selectedSize\?\.multi_file \? selectedSize\.disk_gb \?\? selectedSize\.download_gb : null\}/);
  const picker = read('../src/components/VersionPicker.tsx');
  assert.match(picker, /tr\("\{0\} RAM \+ VRAM, plus about \{1\} GB of disk", \[gb\(fit\.memoryGb\), Math\.round\(selected\.disk_gb\)\]\)/);
  assert.match(picker, /: fit \? `\$\{gb\(fit\.memoryGb\)\} RAM \+ VRAM` : ''/, 'every other model keeps its line');
});
