/**
 * The options the app hands to the DeepSeek engine (strata-dsv41), which exits with "unknown option X" on a name it does not take:
 *  - #4358 #4359 #4505 #4507: a model with pictures off had its `--vision off` pair cut in half (`--eos 1 off --dspark auto`);
 *  - #4504: the Settings field "GPU memory for other apps" wrote --vram-reserve-mib, the resource preference wrote --pool-workers.
 * Nothing here starts an engine: Control is a fake that records the run config the host saved for the start (a scratch profile,
 * never the real one).
 */
import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { createNativeControl } from '../server/native-control.mjs';
import { DS_VALUE_FLAGS, DS_SWITCH_FLAGS, DS_RENAMED, dsEngineArgs, dsUserArgs, dsArgsNote, withoutVisionFlag } from '../server/native-ds-args.mjs';

const DS_ARGS = ['--pack', '/p/ds', '--max-context', '32768', '--eos', '1', '--vision', 'off', '--dspark', 'auto', '--dspark-conf', '0.65', '--prefill-chunk', '1024'];

function fixture(config = {}, id = 'deepseek-v4.1-flash-ist-das-2.25bpw') {
  const root = mkdtempSync(join(tmpdir(), 'strata-ds-args-'));
  const install = join(root, 'install'), data = join(root, 'data');
  mkdirSync(install, { recursive: true });
  process.env.STRATA_INSTALL_DIR = install;
  const path = join(install, `strata-${id}.json`);
  writeFileSync(path, JSON.stringify({
    exe: 'C:\\Users\\x\\AppData\\Local\\Strata\\engine-dsv41\\0.1.52-windows-x86_64-cuda-cuda13\\strata-dsv41.exe', arch: 'deepseek41',
    tokenizer: join(install, 'tok'), args: DS_ARGS, model_name: 'DeepSeek V4.1', port: 8080, ...config }));
  const saved = [];
  const control = {
    control: { dir: join(root, 'control') }, version: '0.2.7',
    async installedModels() { return { install_dir: install, models: [{ id, name: 'DeepSeek V4.1', source_config: path, component_status: 'present', context_tokens: 32768 }] }; },
    async engine() { return { usable: true, backend: 'cuda', components_dir: 'E:\\c', engine: { version: '0.1.52', platform: 'windows-x86_64', backend: 'cuda', exe: 'E:\\e\\strata.exe', root: 'E:\\e' } }; },
    async saveProfile(name, file) { saved.push(JSON.parse(readFileSync(file, 'utf8'))); return {}; },
    async start() { return { ok: true, state: 'ready' }; },
    async stop() { return { ok: true }; },
  };
  const native = createNativeControl({ control, hardwareProbe: () => [{ index: 0, vram_gb: 24 }], directory: data, spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({ status: 'ok' }), requests: () => [] });
  return { native, saved, id, path, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
const finish = async (native, id) => { for (let i = 0; i < 500; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };
/** Every word of the argument list that is not an option name or the value right after one: a stray word means a flag lost its name. */
function strayWords(args, valued) {
  const stray = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith('-')) { if (valued.has(args[i]) && i + 1 < args.length && !args[i + 1].startsWith('--')) i++; continue; }
    stray.push(args[i]);
  }
  return stray;
}

test('the DeepSeek option list is a subset of what ds4_main.cpp parses (it cannot drift ahead of the engine)', () => {
  const source = fileURLToPath(new URL('../../../src/dsv41/ds4_main.cpp', import.meta.url));
  if (!existsSync(source)) return;                                   // a package without the engine source
  const parsed = new Set([...readFileSync(source, 'utf8').matchAll(/a == "(-{1,2}[a-z][a-z0-9-]*)"/g)].map(m => m[1]));
  assert.ok(parsed.has('--eos') && parsed.has('--reserve-mib'), 'the pattern found the engine option parser');
  for (const flag of [...DS_VALUE_FLAGS, ...DS_SWITCH_FLAGS, ...Object.values(DS_RENAMED)]) assert.ok(parsed.has(flag), `${flag} is not an option of strata-dsv41`);
  for (const flag of Object.keys(DS_RENAMED)) assert.ok(!parsed.has(flag), `${flag} is the Qwen engine's name, strata-dsv41 rejects it`);
  assert.ok(!parsed.has('--pool-workers'), '--pool-workers is rejected by strata-dsv41');
});

test('dsEngineArgs renames --vram-reserve-mib, leaves out --pool-workers with its value and keeps the --vision pair', () => {
  const r = dsEngineArgs([...DS_ARGS, '--vram-reserve-mib', '512', '--pool-workers', '20', '--resident-budget-gib', '40']);
  assert.deepEqual(r.args, [...DS_ARGS, '--reserve-mib', '512']);
  assert.deepEqual(r.renamed, [{ from: '--vram-reserve-mib', to: '--reserve-mib' }]);
  assert.deepEqual(r.dropped.map(d => [d.flag, d.value]), [['--pool-workers', '20'], ['--resident-budget-gib', '40']]);
  assert.match(dsArgsNote(r), /--vram-reserve-mib is passed as --reserve-mib/);
  assert.match(dsArgsNote(r), /--pool-workers 20, --resident-budget-gib 40/);
  assert.equal(dsArgsNote(dsEngineArgs(DS_ARGS)), '', 'nothing changed, nothing said');
});

test('dsEngineArgs: a reserve under both names keeps the engine\'s own, a switch at the end and values that look like numbers survive', () => {
  assert.deepEqual(dsEngineArgs(['--reserve-mib', '300', '--vram-reserve-mib', '512']).args, ['--reserve-mib', '300']);
  assert.deepEqual(dsEngineArgs(['--pack', '/p', '--no-expert-cache']).args, ['--pack', '/p', '--no-expert-cache']);
  assert.deepEqual(dsEngineArgs(['--pool-workers']).args, [], 'a flag with no value');
  assert.deepEqual(dsEngineArgs(['--spec', '4', '--tp', '2', '--kv', 'int8', '--experts', 'auto']).args, ['--tp', '2', '--experts', 'auto']);
  assert.deepEqual(dsEngineArgs(undefined).args, []);
});

test('dsUserArgs drops what the engine rejects and keeps the options it takes and the app\'s reserve', () => {
  assert.deepEqual(dsUserArgs(['--pool-workers', '20', '--vram-reserve-mib', '512', '--cpu-threads', '8']), ['--vram-reserve-mib', '512', '--cpu-threads', '8']);
  assert.deepEqual(dsUserArgs(['--pool-workers', '20']), []);
});

test('withoutVisionFlag takes the Qwen engine\'s stand-alone --vision out, and a picture word an old file left after it', () => {
  assert.deepEqual(withoutVisionFlag(['--max-context', '4096', '--vision', '--kv', 'int8']), ['--max-context', '4096', '--kv', 'int8']);
  assert.deepEqual(withoutVisionFlag(['--a', '1', '--vision']), ['--a', '1']);
  assert.deepEqual(withoutVisionFlag(['--vision', 'off', '--eos', '1']), ['--eos', '1']);
  assert.deepEqual(withoutVisionFlag(['--vision', 'gpu', '--x']), ['--x']);
});

test('#4358 #4359 #4505 #4507: a DeepSeek load with pictures off leaves the --vision pair whole and no stray word', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    // what the Models screen sends for a model it does not list pictures for: vision "no"
    const job = await finish(f.native, f.native.operation('load', { model: f.id, vision: 'no' }).id);
    assert.equal(job.state, 'done', job.error);
    const args = f.saved[0].args;
    assert.equal(args[args.indexOf('--vision') + 1], 'off', args.join(' '));
    assert.equal(args[args.indexOf('--eos') + 1], '1', 'the word after --eos is its value, not a left-over of --vision');
    assert.equal(args[args.indexOf('--eos') + 2], '--vision');
    assert.deepEqual(strayWords(args, new Set(['--pack', '--max-context', '--eos', '--vision', '--dspark', '--dspark-conf', '--prefill-chunk'])), []);
    assert.deepEqual(args, DS_ARGS, 'nothing was added or removed');
  } finally { await f.native.close(); f.cleanup(); }
});

test('a DeepSeek model with pictures on keeps `--vision on` too', async () => {
  const f = fixture({ args: DS_ARGS.map(a => (a === 'off' ? 'on' : a)) });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: f.id, vision: 'no' }).id);
    assert.equal(job.state, 'done', job.error);
    const args = f.saved[0].args;
    assert.equal(args[args.indexOf('--vision') + 1], 'on');
    assert.equal(args[args.indexOf('--eos') + 1], '1');
  } finally { await f.native.close(); f.cleanup(); }
});

test('#4504: the reserve field and the resource preference reach strata-dsv41 as --reserve-mib, with no --pool-workers', async () => {
  const f = fixture({ vram_reserve_mib: 512, ui_resource_profile: 'work', args: [...DS_ARGS, '--pool-workers', '20', '--vram-reserve-mib', '512'], args_written: [...DS_ARGS, '--pool-workers', '20', '--vram-reserve-mib', '512'], user_args: ['--pool-workers', '20'] });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: f.id, vision: 'no', resourceProfile: 'work' }).id);
    assert.equal(job.state, 'done', job.error);
    const saved = f.saved[0], args = saved.args;
    assert.ok(!args.includes('--pool-workers') && !args.includes('20'), args.join(' '));
    assert.ok(!args.includes('--vram-reserve-mib'), args.join(' '));
    assert.equal(args[args.indexOf('--reserve-mib') + 1], '512', args.join(' '));
    assert.equal(args.filter(a => a === '--reserve-mib').length, 1);
    assert.ok(!(saved.user_args ?? []).includes('--pool-workers'), 'the person\'s saved options lose it too');
    for (const flag of args.filter(a => a.startsWith('--'))) assert.ok(DS_VALUE_FLAGS.has(flag) || DS_SWITCH_FLAGS.has(flag), `${flag} is an option of strata-dsv41`);
  } finally { await f.native.close(); f.cleanup(); }
});

test('the resource preference no longer writes Qwen-engine options into a DeepSeek model\'s file when it is tuned', async () => {
  const f = fixture({ args: DS_ARGS });
  try {
    await f.native.refresh(true);
    const before = JSON.parse(readFileSync(f.path, 'utf8'));
    const job = await finish(f.native, f.native.operation('tune', { model: f.id, profile: 'work', calibrate: false }).id);
    void job;                                                          // the load and benchmark after it need a real engine; the file is what matters
    const after = JSON.parse(readFileSync(f.path, 'utf8'));
    assert.ok(!(after.args ?? []).includes('--pool-workers'), (after.args ?? []).join(' '));
    assert.ok(!(after.args ?? []).includes('--vram-reserve-mib'), (after.args ?? []).join(' '));
    assert.deepEqual(after.args.slice(0, before.args.length), before.args);
  } finally { await f.native.close(); f.cleanup(); }
});

test('a Qwen model with pictures off still loses its stand-alone --vision (nothing else changed)', async () => {
  const root = mkdtempSync(join(tmpdir(), 'strata-ds-args-q-'));
  const install = join(root, 'install');
  mkdirSync(install, { recursive: true });
  process.env.STRATA_INSTALL_DIR = install;
  const path = join(install, 'strata-q2_0.json');
  writeFileSync(path, JSON.stringify({ exe: 'component:strata-engine', tokenizer: join(install, 'tok'), args: ['--pack', '/p', '--max-context', '4096', '--vision', '--kv', 'int8'], model_name: 'Qwen', port: 8080 }));
  const saved = [];
  const control = {
    control: { dir: join(root, 'control') }, version: '0.2.7',
    async installedModels() { return { install_dir: install, models: [{ id: 'q2_0', name: 'Qwen', source_config: path, component_status: 'present', context_tokens: 4096 }] }; },
    async engine() { return { usable: true, backend: 'cuda', components_dir: 'E:\\c', engine: { version: '0.2.0', platform: 'windows-x86_64', backend: 'cuda', exe: 'E:\\e\\strata.exe', root: 'E:\\e' } }; },
    async saveProfile(name, file) { saved.push(JSON.parse(readFileSync(file, 'utf8'))); return {}; },
    async start() { return { ok: true, state: 'ready' }; },
    async stop() { return { ok: true }; },
  };
  const native = createNativeControl({ control, hardwareProbe: () => [{ index: 0, vram_gb: 12 }], directory: join(root, 'data'), spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({ status: 'ok' }), requests: () => [] });
  try {
    await native.refresh(true);
    const job = await finish(native, native.operation('load', { model: 'q2_0', vision: 'no' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.deepEqual(saved[0].args, ['--pack', '/p', '--max-context', '4096', '--kv', 'int8']);
  } finally { await native.close(); rmSync(root, { recursive: true, force: true }); }
});
