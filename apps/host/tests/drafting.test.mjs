// The Drafting choice in the app (MTP / DFlash2 / Off / Auto): the Models screen row, the setting the host saves, what the host asks Control before
// a load, the download plan line, and the words in all nine languages. The table of which card gets which drafter lives in Python
// (strata_control/drafting.py, tests/terminal/test_drafting.py); the app never reads it.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { NATIVE_FIELDS, APP_OWNED_FIELDS, validateNativeSettings, withoutDraftLayer } from '../server/native-settings.mjs';
import { createControl } from '../server/control-runtime.mjs';

const read = rel => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8').replace(/\r\n/g, '\n');

test('Drafting is a setting of the model with four choices, saved by the app itself', () => {
  const field = NATIVE_FIELDS.find(f => f.id === 'drafting');
  assert.ok(field);
  assert.deepEqual(field.choices, ['auto', 'mtp', 'dflash2', 'off']);
  assert.equal(field.type, 'enum');
  assert.match(field.hint, /Auto \(recommended\)/);
  assert.match(field.hint, /Qwen3\.8-Flash-Next only/);
  assert.ok(APP_OWNED_FIELDS.includes('drafting'), 'the engine\'s own source never lists it: the app applies it');
  for (const ok of ['auto', 'mtp', 'dflash2', 'off']) assert.deepEqual(validateNativeSettings({ drafting: ok }), { drafting: ok });
  assert.deepEqual(validateNativeSettings({ drafting: null }), { drafting: null }, 'cleared = Auto');
  assert.throws(() => validateNativeSettings({ drafting: 'fastest' }), /Drafting: choose a supported option/);
});

test('Off leaves the draft layer out of every start, the same as "Draft layer (MTP)" off', () => {
  assert.deepEqual(withoutDraftLayer(['--pack', 'p', '--mtp', '/m', '--spec', '4', '--spec-min-p', '0.5', '--batch-mtp']), ['--pack', 'p', '--spec', '4']);
  const source = read('../server/native-control.mjs');
  assert.match(source, /\(cfg\.mtp===false\|\|cfg\.drafting==='off'\)&&Array\.isArray\(cfg\.args\)\)cfg\.args=withoutDraftLayer\(cfg\.args\)/);
});

test('the Models screen offers the choice next to context, DFlash2 only for Qwen3.8-Flash-Next, and carries it to the plan and the load', () => {
  const screen = read('../src/components/NativeModels.tsx');
  assert.match(screen, /draftingOptions\(selectedSize\.sourceFamily \?\? selectedFamily\?\.family, native\?\.capabilities\?\.dflash2\)/);
  assert.match(screen, /draftingWords: Record<string, string> = \{ auto: 'Auto \(recommended\)', mtp: 'MTP', dflash2: 'DFlash2', off: 'Off' \}/);
  const row = screen.slice(screen.indexOf('<div className="native-choice__row">'), screen.indexOf('{roleCards.length > 1 ? <GpuRoles'));
  assert.ok(row.indexOf('Conversation context') < row.indexOf('tr("Drafting")') && row.indexOf('tr("Drafting")') < row.indexOf('Computer use'), 'in the same row, between context and computer use');
  assert.match(row, /<Select ariaLabel=\{tr\("Drafting"\)\}/, 'the existing Select, no new component');
  assert.match(screen, /setDrafting\(installed\.drafting \?\? 'auto'\)/, 'a model shows what it was saved with');
  assert.match(screen, /const downloadValues = \{ \.\.\.values, \.\.\.\(draftingChoices \? \{ drafting \} : \{\}\)/);
  assert.match(screen, /\.\.\.\(draftingChoices \? \{ values: \{ drafting \} \} : \{\}\)/);
  assert.match(screen, /Choosing DFlash2 also downloads its drafter: about \{0\} GB, once\. If it does not fit in graphics memory, the model uses the MTP instead\./);
  assert.match(screen, /loadedSelected\?\.metrics\?\.engine\?\.drafting_note/, 'what the engine said is shown');
});

test('the host asks Control before a load, and Control decides (the app never reads the table)', async () => {
  const calls = [];
  const control = createControl({ control: { dir: 'C:/c', python: 'py', cli: 'strata.py' }, run: async (args, options) => {
    calls.push({ args, options });
    return args[1] === 'plan'
      ? { code: 0, json: { ok: true, needed: true, gb: 1.2, line: 'Drafting: DFlash2' }, text: '', stderr: '' }
      : { code: 0, json: { ok: true, event: 'result', pack: 'C:/d/dflash2/x.pack' }, text: '', stderr: '' };
  } });
  const plan = await control.draftingPlan({ mode: 'dflash2', family: 'gyro', dataDir: 'C:/d' });
  assert.deepEqual(calls[0].args, ['drafting', 'plan', '--mode', 'dflash2', '--family', 'gyro', '--data-dir', 'C:/d', '--output', 'json']);
  assert.equal(plan.needed, true);
  const events = [];
  await control.draftingFetch({ configPath: 'C:/w/cfg.json', mode: 'dflash2', onEvent: e => events.push(e) });
  assert.deepEqual(calls[1].args, ['drafting', 'fetch', '--config', 'C:/w/cfg.json', '--mode', 'dflash2', '--if-needed', '--output', 'jsonl']);
  assert.equal(calls[1].options.acquire, true, 'only this explicit step may reach the network');
  assert.equal(calls[0].options.acquire, undefined, 'planning never does');
  const failing = createControl({ control: { dir: 'C:/c', python: 'py', cli: 'strata.py' }, run: async () => ({ code: 5, json: { ok: false, error: { code: 'drafting_failed', message: 'The DFlash2 download did not finish.' } }, text: '', stderr: '' }) });
  await assert.rejects(failing.draftingFetch({ configPath: 'x' }), /DFlash2 download did not finish/);
});

test('a load fetches the drafter first, never stops for it, and says what ran', () => {
  const source = read('../server/native-control.mjs');
  const prepare = source.slice(source.indexOf('async function prepareDrafter'), source.indexOf('function rememberSetting'));
  assert.match(prepare, /rememberSetting\(configFile,'drafting',asked===null\|\|asked==='auto'\?null:asked\)/, 'the choice is kept for the next load');
  assert.match(prepare, /cfg\.drafting==='auto'\|\|cfg\.drafting==='mtp'\|\|cfg\.drafting==='off'\)return/, 'Auto, MTP and Off never download');
  assert.match(prepare, /control\.draftingFetch\(/);
  assert.match(prepare, /catch\(error\)\{if\(job\.state==='running'\)\{job\.drafterNote=error\.message;/, 'a failed download leaves the MTP and the job says so');
  assert.match(source, /await prepareDrafter\(job,args,cfg,configFile\)/);
  assert.match(source, /async function noteDrafting\(job,instance,key\)/);
  assert.match(source, /instance\.status='ready';instance\.phase='idle';await noteDrafting\(job,instance,key\)/);
  assert.match(source, /choice!=='auto'&&typeof control\.draftingPlan==='function'/, 'the plan lists the drafter only when it is chosen and Control says it is needed');
  assert.match(source, /drafting:cfg\.drafting\?\?'auto'/, 'the installed model reports its saved choice');
});

test('every sentence of the Drafting choice is in the catalog in all nine languages, with the same slots', () => {
  const langs = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr'];
  const catalogs = Object.fromEntries(langs.map(l => [l, { ...JSON.parse(read(`../src/locales/${l}.json`)), ...JSON.parse(read(`../src/locales/df2-${l}.json`)) }]));
  const idOf = text => Object.entries(catalogs.en).find(([, v]) => v === text)?.[0];
  const sentences = ['Drafting', 'Auto (recommended)', 'MTP', 'DFlash2',
    "Drafting: MTP (DFlash2 didn't fit)", "Drafting: MTP (DFlash2 isn't available in this mode)", 'Drafting: MTP (DFlash2 is for Qwen3.8-Flash-Next only)',
    "Drafting: MTP (the DFlash2 drafter isn't downloaded yet)",
    'Choosing DFlash2 also downloads its drafter: about {0} GB, once. If it does not fit in graphics memory, the model uses the MTP instead.',
    'Downloading the DFlash2 drafter…', 'Packing the DFlash2 drafter for the engine…', 'The DFlash2 drafter is ready.',
    'The DFlash2 drafter could not be downloaded, so the model starts with the MTP. Loading the model…', 'Loading the model…',
    // the DFlash2 option (df2-<lang>.json): the paragraph under the dropdown, the lines about the card, the two new notes
    "Drafting: MTP (DFlash2 isn't available on this graphics card)", "There is no internet connection, so the DFlash2 drafter can't be downloaded now.",
    'It downloads about 1.2 GB once, works with one graphics card and one request at a time, and uses MTP otherwise.',
    'On graphics cards like yours, DFlash2 was faster than MTP on some kinds of text and slower on others in our tests.',
    'On graphics cards like yours, DFlash2 was slower than MTP in our tests.', 'DFlash2 has not been tested on your graphics card yet.', 'SSD: not needed'];
  const slots = t => (t.match(/\{\d+\}/g) ?? []).sort().join();
  for (const text of sentences) {
    const id = idOf(text);
    assert.ok(id, `"${text.slice(0, 60)}" is in the English catalog`);
    for (const l of langs) {
      const t = catalogs[l][id];
      assert.ok(typeof t === 'string' && t.trim(), `${id} in ${l}`);
      assert.equal(slots(t), slots(text), `${id} in ${l} has the same slots`);
      if (l !== 'en' && text.length > 25) assert.notEqual(t, text, `${id} is translated in ${l}`);
    }
  }
});

test('the Drafting dropdown exists only where there is something to choose: Qwen3.8-Flash-Next (qwen, gyro) on a computer whose engine can run DFlash2', async () => {
  const { transform } = await import('esbuild');
  const code = (await transform(read('../src/lib/drafting.ts'), { loader: 'ts', format: 'esm' })).code;
  const { draftingChoices } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
  const all = ['auto', 'mtp', 'dflash2', 'off'];
  // the engine here cannot run it (Metal, Intel, no card), or Control has not answered yet: no dropdown for anyone
  for (const family of ['qwen', 'gyro', 'coder', 'swift', 'deepseek41', undefined]) {
    assert.equal(draftingChoices(family, false), null, `${family}, not served`);
    assert.equal(draftingChoices(family, undefined), null, `${family}, Control has not answered`);
    assert.equal(draftingChoices(family, null), null);
  }
  // it can: only the models it is for. DeepSeek has no MTP (its DSpark drafter is chosen by the engine); the other families have the MTP and nothing to pick against it.
  assert.deepEqual(draftingChoices('qwen', true), all);
  assert.deepEqual(draftingChoices('gyro', true), all);
  for (const family of ['coder', 'swift', 'unsloth', 'nvfp4', 'qwen36', 'deepseek41', undefined]) assert.equal(draftingChoices(family, true), null, `${family}, served`);
  // the app learns it from Control, once, and false until Control answers
  const host = read('../server/native-control.mjs');
  assert.match(host, /dflash2:dflash2Served===true,dflash2Support\}/);
  assert.match(host, /dflash2Served=p\.serves===true/);
  assert.match(host, /function probeDrafting\(\)\{if\(!packaged\|\|dflash2Served!==null/);
  const screen = read('../src/components/NativeModels.tsx');
  assert.match(screen, /\{tr\(draftingChoices \? <div className="native-choice__col">/, 'no choices: no column');
});

test('Settings > Native engine settings hides the MTP and Drafting rows for DeepSeek and the Drafting row where it has no choice', () => {
  const byId = Object.fromEntries(NATIVE_FIELDS.map(f => [f.id, f]));
  for (const id of ['mtp', 'drafting', 'batchMtp']) assert.equal(byId[id].notForModel, 'deepseek', id);
  assert.equal(byId.ds_experts.forModel, 'deepseek');
  const settings = read('../src/components/NativeSettings.tsx');
  assert.match(settings, /\(!f\.notForModel \|\| !model\.toLowerCase\(\)\.includes\(f\.notForModel\)\) && \(f\.id !== 'drafting' \|\| draftingOffered\)/);
});

test('the DFlash2 drafter is fetched only by the Models screen own Load: a restart, a tuning run or any other load never downloads', () => {
  const source = read('../server/native-control.mjs');
  const prepare = source.slice(source.indexOf('async function prepareDrafter'), source.indexOf('function rememberSetting'));
  assert.match(prepare, /asked===undefined\|\|!cfg\.drafting/, 'no explicit choice in this job: return before any fetch');
  const loadScreen = read('../src/components/NativeModels.tsx');
  assert.match(loadScreen, /\.\.\.\(draftingChoices \? \{ values: \{ drafting \} \} : \{\}\)/, 'the screen passes the choice only when it showed the dropdown');
  const tune = source.slice(source.indexOf("const loaded=await loadEngine(job,{model:args.model,multiple:false"), source.indexOf("const loaded=await loadEngine(job,{model:args.model,multiple:false") + 140);
  assert.doesNotMatch(tune, /values:/, 'the tuning run loads without a choice');
});

test('the paragraph under the dropdown quotes the measurements without a verdict', () => {
  const screen = read('../src/components/NativeModels.tsx');
  for (const part of ['7.8% faster than MTP on 19 short prompts', '1.1% faster on long thinking', 'faster on 21 of 23 prompts', 'the answers were identical']) assert.ok(screen.includes(part), part);
  const en = JSON.parse(read('../src/locales/df2-en.json'));
  for (const text of Object.values(en)) for (const word of ['recommend', 'best ', 'better', 'should']) assert.ok(!text.toLowerCase().includes(word), `${word} in "${text.slice(0, 50)}"`);
});
