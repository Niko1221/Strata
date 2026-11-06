/**
 * DeepSeek and the app's "Concurrent requests" setting (Expert mode, the one the other models have): off by default, and a number set there reaches the
 * run config as "parallel": N, which the server turns into the engine's `--parallel N`. Nothing is added for a model that never had the setting, and the
 * Qwen engine's slots (--batch) are never written for a DeepSeek model. Nothing here starts an engine: Control is a fake that records the run config.
 */
import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createNativeControl } from '../server/native-control.mjs';
import { NATIVE_FIELDS } from '../server/native-settings.mjs';

const DS_ARGS = ['--pack', '/p/ds', '--max-context', '32768', '--eos', '1', '--vision', 'off', '--dspark', 'auto', '--prefill-chunk', '1024'];

function fixture(config = {}, id = 'deepseek-v4.1-flash-ist-das-2.25bpw') {
  const root = mkdtempSync(join(tmpdir(), 'strata-ds-par-'));
  const install = join(root, 'install'), data = join(root, 'data');
  mkdirSync(install, { recursive: true });
  mkdirSync(join(root, 'control', 'serve'), { recursive: true });
  writeFileSync(join(root, 'control', 'serve', 'server.py'), '# the installed server: "parallel" is a setting it knows');   // the app lists a setting only when the installed server knows it
  process.env.STRATA_INSTALL_DIR = install;
  const path = join(install, `strata-${id}.json`);
  writeFileSync(path, JSON.stringify({
    exe: 'C:\\Users\\x\\AppData\\Local\\Strata\\engine-dsv41\\0.1.55-windows-x86_64-cuda-cuda13\\strata-dsv41.exe', arch: 'deepseek41',
    tokenizer: join(install, 'tok'), args: DS_ARGS, model_name: 'DeepSeek V4.1', port: 8080, ...config }));
  const saved = [];
  const control = {
    control: { dir: join(root, 'control') }, version: '0.2.9',
    async installedModels() { return { install_dir: install, models: [{ id, name: 'DeepSeek V4.1', source_config: path, component_status: 'present', context_tokens: 32768 }] }; },
    async engine() { return { usable: true, backend: 'cuda', components_dir: 'E:\\c', engine: { version: '0.1.55', platform: 'windows-x86_64', backend: 'cuda', exe: 'E:\\e\\strata.exe', root: 'E:\\e' } }; },
    async saveProfile(name, file) { saved.push(JSON.parse(readFileSync(file, 'utf8'))); return {}; },
    async start() { return { ok: true, state: 'ready' }; },
    async stop() { return { ok: true }; },
  };
  const native = createNativeControl({ control, hardwareProbe: () => [{ index: 0, vram_gb: 24 }], directory: data, spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({ status: 'ok' }), requests: () => [] });
  return { native, saved, id, path, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
const finish = async (native, id) => { for (let i = 0; i < 500; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };

test('the setting is a field for every model: DeepSeek is not left out of it', () => {
  const field = NATIVE_FIELDS.find(f => f.id === 'parallel');
  assert.ok(field, 'Concurrent requests exists');
  assert.equal(field.forModel, undefined);
  assert.deepEqual([field.min, field.max], [1, 8]);
});

test('a DeepSeek model nobody set Concurrent requests for loads one request at a time: no parallel, no slots', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: f.id, vision: 'no' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.saved[0].parallel, undefined);
    assert.ok(!f.saved[0].args.includes('--batch') && !f.saved[0].args.includes('--slots') && !f.saved[0].args.includes('--parallel'), f.saved[0].args.join(' '));
    assert.deepEqual(f.saved[0].args, DS_ARGS);
  } finally { await f.native.close(); f.cleanup(); }
});

test('Concurrent requests saved for a DeepSeek model reaches the run config as parallel and stays saved', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    f.native.save(f.id, { values: { parallel: 4 } });
    assert.equal(JSON.parse(readFileSync(f.path, 'utf8')).parallel, 4);
    const job = await finish(f.native, f.native.operation('load', { model: f.id, vision: 'no' }).id);
    assert.equal(job.state, 'done', job.error);
    const cfg = f.saved[0];
    assert.equal(cfg.parallel, 4);
    assert.ok(!cfg.args.includes('--batch') && !cfg.args.includes('--slots'), 'the Qwen slots (--batch) are not written for DeepSeek');
    assert.ok(!cfg.args.includes('--batch-mtp'));
    assert.deepEqual(cfg.args, DS_ARGS, 'the engine arguments are what they were');
    assert.equal(JSON.parse(readFileSync(f.path, 'utf8')).parallel, 4, 'the saved number is still in the model file after the load');
    f.native.save(f.id, { values: { parallel: null } });                  // cleared: back to one at a time
    assert.equal(JSON.parse(readFileSync(f.path, 'utf8')).parallel, undefined);
  } finally { await f.native.close(); f.cleanup(); }
});
