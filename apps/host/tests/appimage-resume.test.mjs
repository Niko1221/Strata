/**
 * The AppImage's `--resume` (STRATA_RESUME=1): the model you were using comes back at start, with the settings it was loaded with.
 * The shell hands the host STRATA_RESUME=1 (apps/desktop/src-tauri/src/supervisor.rs); the host replays the load it recorded.
 * Nothing here starts an engine: Control is a fake that records what the host saved and asked it to start.
 */
import './helpers/isolate-profile.mjs';   // a test run never writes the owner's real Strata profile
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createNativeControl, resumeRequested } from '../server/native-control.mjs';

const ARGS = ['--pack', '/p', '--expert-cache', 'auto', '--max-context', '40000', '--kv', 'int8'];

function fixture({ data, install, missing = false, removed = false } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-resume-test-'));
  const folder = install ?? join(root, 'install');
  mkdirSync(folder, { recursive: true });
  process.env.STRATA_INSTALL_DIR = folder;
  const path = join(folder, 'strata-q2_0.json');
  writeFileSync(path, JSON.stringify({ exe: 'component:strata-engine', tokenizer: join(folder, 'tok'), args: ARGS, model_name: 'Qwen q2_0', port: 8080 }));
  const started = [];
  const control = {
    control: { dir: join(root, 'control') }, version: '0.2.0',
    async installedModels() {
      if (removed) return { install_dir: folder, models: [] };
      return { install_dir: folder, models: [{ id: 'q2_0', name: 'Qwen q2_0', source_config: path, component_status: missing ? 'absent' : 'present', context_tokens: 40000 }] };
    },
    async engine() { return { usable: true, backend: 'cuda', components_dir: join(root, 'components'), engine: { version: '0.2.0', platform: 'linux-x86_64', backend: 'cuda', exe: join(root, 'engine', 'strata'), root: join(root, 'engine') } }; },
    async saveProfile(name, file) { started.push(JSON.parse(readFileSync(file, 'utf8'))); return {}; },
    async start() { return { ok: true, state: 'ready' }; },
    async stop() { return { ok: true }; },
  };
  const native = createNativeControl({ control, hardwareProbe: () => [{ index: 0, vram_gb: 24 }], directory: data ?? join(root, 'data'), spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({ status: 'ok' }), requests: () => [] });
  return { native, started, path, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
const finish = async (native, id) => { for (let i = 0; i < 500; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };

test('nothing loads at start unless the switch is on', () => {
  assert.equal(resumeRequested({}), false);
  assert.equal(resumeRequested({ STRATA_RESUME: '' }), false);
  assert.equal(resumeRequested({ STRATA_RESUME: '0' }), false);
  assert.equal(resumeRequested({ STRATA_RESUME: 'browser' }), false);
  for (const value of ['1', 'on', 'TRUE', 'yes', 'resume', ' Resume ']) {
    assert.equal(resumeRequested({ STRATA_RESUME: value }), true, value);
  }
});

test('a load remembers the model and the settings it was loaded with', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: 'q2_0', vision: 'no' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.native.lastLoad().args.model, 'q2_0');
    assert.equal(f.native.lastLoad().args.vision, 'no');
  } finally { await f.native.close(); f.cleanup(); }
});

test('resume starts the model of the last session again, with the same settings', async () => {
  const root = mkdtempSync(join(tmpdir(), 'strata-resume-session-'));
  const data = join(root, 'data'), install = join(root, 'install');
  try {
    const first = fixture({ data, install });
    await first.native.refresh(true);
    await finish(first.native, first.native.operation('load', { model: 'q2_0', vision: 'no' }).id);
    await first.native.close();

    // a fresh host over the same data folder: nothing is loaded yet, so resume has to load it
    const second = fixture({ data, install });
    const answer = await second.native.resumeLastLoad();
    assert.equal(answer.loaded, true, answer.reason);
    assert.equal(answer.model, 'q2_0');
    const job = await finish(second.native, answer.jobId);
    assert.equal(job.state, 'done', job.error);
    assert.equal(second.started.length, 1, 'the resume asked Control to start the engine');
    assert.deepEqual(second.started[0].args, first.started[0].args, 'the engine starts with the arguments of the load it replaces');
    await second.native.close();
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('a second resume while the model is loaded says so, and starts nothing', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    await finish(f.native, f.native.operation('load', { model: 'q2_0', vision: 'no' }).id);
    const again = await f.native.resumeLastLoad();
    assert.deepEqual({ loaded: again.loaded, already: again.already, model: again.model }, { loaded: true, already: true, model: 'q2_0' });
    assert.equal(f.started.length, 1, 'one start only');
  } finally { await f.native.close(); f.cleanup(); }
});

test('a start with nothing recorded says what happened, and downloads nothing', async () => {
  const f = fixture();
  try {
    const answer = await f.native.resumeLastLoad();
    assert.equal(answer.loaded, false);
    assert.match(answer.reason, /no record of a model you loaded/);
    assert.equal(f.started.length, 0);
  } finally { await f.native.close(); f.cleanup(); }
});

test('a model whose files are gone is reported: the flag is never a request to download', async () => {
  const root = mkdtempSync(join(tmpdir(), 'strata-resume-gone-'));
  const data = join(root, 'data'), install = join(root, 'install');
  try {
    const first = fixture({ data, install });
    await first.native.refresh(true);
    await finish(first.native, first.native.operation('load', { model: 'q2_0', vision: 'no' }).id);
    await first.native.close();

    const absent = fixture({ data, install, missing: true });
    const files = await absent.native.resumeLastLoad();
    assert.equal(files.loaded, false);
    assert.equal(files.model, 'q2_0');
    assert.match(files.reason, /not on this computer/);
    await absent.native.close();

    const removed = fixture({ data, install, removed: true });
    const gone = await removed.native.resumeLastLoad();
    assert.equal(gone.loaded, false);
    assert.match(gone.reason, /not set up in this Strata folder/);
    await removed.native.close();
  } finally { rmSync(root, { recursive: true, force: true }); }
});
