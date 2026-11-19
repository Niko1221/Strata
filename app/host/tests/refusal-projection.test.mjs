import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { cpSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { build } from 'esbuild';
import http from 'node:http';
import { createRequire } from 'node:module';
import { nativeFixture, finishJob } from './helpers/native-fixture.mjs';
import { PROJECTION_KEY, VECTOR_FILE, VECTOR_SHA256, probeRefusalProjection, projectionArgs, projectionVectorPath,
  verifyProjectionVector, applyProjection, clearOwnedProjection } from '../server/refusal-projection.mjs';
import { proxyCompletion } from '../server/inference.mjs';
import { update } from '../server/store.mjs';
import { createLlmProxy, samplingFromExpert } from '../server/agent/llm-shape.mjs';

const root = resolve(import.meta.dirname, '../../..');
const cap = { project: true, binarySha256: 'ab'.repeat(32) };
function fixture(options = {}) {
  let capability = cap;
  const probes = [];
  const f = nativeFixture({ packaged: true, refusalCapabilityProbe: exe => { probes.push(exe); return typeof capability === 'function' ? capability(exe) : capability; }, ...options });
  const data = join(f.repo, 'data', 'experimental-speed-projection');
  mkdirSync(join(f.repo, 'data'), { recursive: true });
  cpSync(join(root, 'data', 'experimental-speed-projection'), data, { recursive: true });
  return { ...f, data, probes, setCapability: value => { capability = value; } };
}
const read = (f, model = 'iq2_xs') => JSON.parse(readFileSync(join(f.repo, `strata-${model}.json`), 'utf8'));
const write = (f, config, model = 'iq2_xs') => writeFileSync(join(f.repo, `strata-${model}.json`), JSON.stringify(config));
const launch = async (f, model = 'iq2_xs') => {
  const job = await finishJob(f.control, f.control.operation('load', { model, vision: 'no' }).id);
  assert.equal(job.state, 'done', job.error);
  return JSON.parse(readFileSync(f.launchConfigs.at(-1), 'utf8'));
};

test('default and explicit OFF preserve launch args byte-for-byte, without requiring a vector', () => {
  const config = { args: ['--native', '/own/model.gguf', '--spec', '2'], sampling: { temperature: .4 } };
  const context = { root: '/missing', policy: { id: 'q38' }, family: 'qwen' };
  assert.equal(applyProjection(config, context), config);
  const off = { ...config, [PROJECTION_KEY]: false };
  assert.equal(JSON.stringify(applyProjection(off, context)), JSON.stringify(off));
});

test('capability comes from bounded selected-binary help (stdout or stderr), never filename/version', () => {
  const help = ' --control-vector-scaled FILE:SCALE\n cvec=0|1\n --control-vector-layer-range A B\n --cvec-mode add|project\n --cvec-dir per-layer|single:L\n';
  const hash = () => 'ab'.repeat(32);
  for (const stream of ['stdout', 'stderr']) {
    const got = probeRefusalProjection('renamed-q36.exe', { hash, run: (exe, args, options) => {
      assert.equal(exe, 'renamed-q36.exe'); assert.deepEqual(args, ['--help']);
      assert.equal(options.timeout, 3000); assert.equal(options.maxBuffer, 65536);
      assert.equal(Object.keys(options.env).some(k => /TOKEN|SECRET|API_KEY/.test(k)), false);
      return { status: 0, [stream]: help };
    } });
    assert.deepEqual(got, cap);
  }
  for (const result of [{ status: 0, stdout: 'Qwen3.8 engine 9.9' }, { status: 1, stdout: help }, { status: 0, stdout: help.replace('add|project', 'add') }, { status: 0, stdout: help + 'x'.repeat(65536) }])
    assert.equal(probeRefusalProjection('Qwen3.8-supported.exe', { hash, run: () => result }), null);
  let hashes = 0;
  assert.equal(probeRefusalProjection('engine.exe', { hash: () => (++hashes === 1 ? 'ab' : 'cd').repeat(32), run: () => ({ status: 0, stdout: help }) }), null);
});

test('ON persists in the existing native settings API and reaches the real profile builder; OFF unloads only the feature', async () => {
  const f = fixture();
  try {
    await f.control.refresh(true);
    const original = read(f), before = await launch(f);
    assert.equal(f.probes.length, 0, 'the default launch does not run extra capability probes');
    assert.equal(before.args.some(a => a.startsWith('--control-vector')), false);
    const view = await f.control.saveChecked('iq2_xs', { values: { [PROJECTION_KEY]: true } });
    assert.equal(view.values[PROJECTION_KEY], true); assert.equal(view.refusalProjection.supported, true);
    assert.equal(read(f)[PROJECTION_KEY], true); assert.deepEqual(read(f).args, original.args);
    const on = await launch(f);
    const at = on.args.indexOf('--control-vector-scaled');
    assert.deepEqual(on.args.slice(at, at + 9), projectionArgs(projectionVectorPath(f.repo)));
    assert.equal(on.args.filter(a => a === '--cvec-mode').length, 1);
    assert.equal(f.control.state().instances[0].refusalProjection.vectorVerified, true);
    assert.equal(f.control.state().instances[0].metrics?.engine?.cvec, undefined, 'API ready does not manufacture an INFO receipt');
    await f.control.saveChecked('iq2_xs', { values: { [PROJECTION_KEY]: false } });
    const probesBeforeOff = f.probes.length;
    const off = await launch(f);
    assert.equal(f.probes.length, probesBeforeOff, 'an OFF launch does not probe the engine or require the vector');
    assert.deepEqual(off.args, before.args); assert.equal(read(f)[PROJECTION_KEY], false);
    assert.equal(f.rpcCalls.some(c => c.name === 'strata_install'), false);
    assert.deepEqual(read(f).unknown_native_key, original.unknown_native_key);
  } finally { await f.cleanup(); }
});

for (const mismatch of ['q36', 'deepseek', 'prism', 'turboquant', 'swift', 'wrong-shape', 'unknown-binary', 'explicit-unsupported', 'backend-prism', 'backend-turboquant', 'missing', 'bytes', 'pin'])
  test(`${mismatch}: refuses before job, profile write, unload or saved-config change`, async () => {
    const f = fixture();
    try {
      await f.control.refresh(true); await launch(f);
      const config = read(f, 'iq3_s');
      if (mismatch === 'q36') config.arch = 'qwen35moe';
      if (mismatch === 'deepseek') config.arch = 'deepseek41';
      if (['prism', 'turboquant', 'swift'].includes(mismatch)) f.catalog.families[0].family = mismatch;
      if (mismatch === 'wrong-shape') config.family = 'swift';
      if (mismatch === 'unknown-binary') f.setCapability(null);
      if (mismatch === 'explicit-unsupported') { config.exe = join(f.repo, 'renamed-unsupported.exe'); f.setCapability(exe => exe === config.exe ? null : cap); }
      if (mismatch.startsWith('backend-')) config.backend = mismatch.slice('backend-'.length);
      if (mismatch === 'missing') rmSync(join(f.data, VECTOR_FILE));
      if (mismatch === 'bytes') { const bytes = readFileSync(join(f.data, VECTOR_FILE)); bytes[bytes.length - 1] ^= 1; writeFileSync(join(f.data, VECTOR_FILE), bytes); }
      if (mismatch === 'pin') { const path = join(f.data, 'PROVENANCE.json'), pin = JSON.parse(readFileSync(path)); pin.vector.sha256 = 'ff'.repeat(32); writeFileSync(path, JSON.stringify(pin)); }
      write(f, config, 'iq3_s'); await f.control.refresh(true);
      const before = readFileSync(join(f.repo, 'strata-iq3_s.json'), 'utf8'), jobs = f.control.state().jobs.length, effects = [...f.effects];
      const view = await f.control.settingsKv('iq3_s');
      assert.equal(view.refusalProjection.supported, false);
      assert.equal(view.fields.find(f => f.id === PROJECTION_KEY).supported, false);
      await assert.rejects(f.control.saveChecked('iq3_s', { values: { [PROJECTION_KEY]: true } }), /vector|projection|adapter/i);
      assert.throws(() => f.control.operation('load', { model: 'iq3_s', values: { [PROJECTION_KEY]: true } }), /vector|projection|adapter/i);
      assert.equal(f.control.state().jobs.length, jobs); assert.deepEqual(f.effects, effects);
      assert.equal(f.control.state().instances[0].status, 'ready');
      assert.equal(readFileSync(join(f.repo, 'strata-iq3_s.json'), 'utf8'), before);
    } finally { await f.cleanup(); }
  });

test('custom vectors/user_args survive OFF, default and rejected ON, even with the same basename elsewhere', async () => {
  const f = fixture();
  try {
    const custom = projectionArgs(join(f.repo, 'my vectors', VECTOR_FILE));
    write(f, { ...read(f), args: ['--max-context', '32768', ...custom], user_args: custom, args_written: ['--max-context', '32768'] });
    await f.control.refresh(true);
    const original = read(f);
    assert.deepEqual(clearOwnedProjection(original, f.repo), original);
    await assert.rejects(f.control.saveChecked('iq2_xs', { values: { [PROJECTION_KEY]: true } }), /Custom control-vector/);
    await f.control.saveChecked('iq2_xs', { values: { [PROJECTION_KEY]: false } });
    const run = await launch(f);
    assert.ok(run.args.join(' ').includes(custom.join(' '))); assert.deepEqual(read(f).user_args, custom);
    assert.deepEqual(read(f).args, original.args);
  } finally { await f.cleanup(); }
});

test('setup-owned exact bundle can be switched off in args, args_written and user_args; changed ranges remain user-owned', async () => {
  const f = fixture();
  try {
    const bundle = projectionArgs(projectionVectorPath(f.repo));
    const config = { ...read(f), args: ['--spec', '2', ...bundle], user_args: ['--mine', 'yes', ...bundle], args_written: ['--spec', '2', ...bundle] };
    write(f, config); await f.control.refresh(true);
    assert.equal(f.control.settings('iq2_xs').values[PROJECTION_KEY], true, 'unrelated GUI saves retain setup-owned ON');
    assert.equal((await f.control.settingsKv('iq2_xs')).refusalProjection.requested, true);
    await f.control.saveChecked('iq2_xs', { values: { [PROJECTION_KEY]: false } });
    assert.deepEqual(read(f).args, ['--spec', '2']); assert.deepEqual(read(f).user_args, ['--mine', 'yes']);
    assert.deepEqual(read(f).args_written, ['--spec', '2']);
    const changed = { args: bundle.map(v => v === '44' ? '43' : v) };
    assert.deepEqual(clearOwnedProjection(changed, f.repo), changed);
    assert.equal(verifyProjectionVector(f.repo), projectionVectorPath(f.repo));
    assert.equal((await f.control.settingsKv('iq2_xs')).refusalProjection.vectorSha256, VECTOR_SHA256);
  } finally { await f.cleanup(); }
});

for (const supported of [true, false]) test(`a transformed split binary is independently verified (${supported ? 'supported' : 'unsupported'})`, async () => {
  const f = fixture({
    draftingPlan: async () => ({ serves: true, choice: 'mtp', split: true, needed: false, mtp_needed: false }),
    splitEngine: async ({ configPath, outPath }) => {
      const candidate = JSON.parse(readFileSync(configPath));
      candidate.exe = join(f.repo, 'split-runtime.exe');
      delete candidate[PROJECTION_KEY]; candidate.args = ['--max-context', '32768'];
      writeFileSync(outPath, JSON.stringify(candidate));
      return { split: true, written: true, line: 'Split fixture selected' };
    },
  });
  try {
    f.setCapability(exe => supported || !String(exe).endsWith('split-runtime.exe') ? cap : null);
    await f.control.refresh(true);
    await f.control.saveChecked('iq2_xs', { values: { [PROJECTION_KEY]: true } });
    const run = await launch(f);
    assert.equal(run.exe === join(f.repo, 'split-runtime.exe'), supported);
    assert.ok(run.args.includes('--control-vector-scaled'), 'the chosen engine always retains the verified vector flags');
    assert.ok(f.probes.some(exe => String(exe).endsWith('split-runtime.exe')));
    assert.equal(f.control.state().instances[0].splitNote === 'Split fixture selected', supported);
  } finally { await f.cleanup(); }
});

const compiled = await build({ stdin: { contents: `import React from 'react'; import {renderToStaticMarkup} from 'react-dom/server'; import {RefusalProjectionStatus} from './src/components/RefusalProjectionStatus'; export * from './src/lib/refusal-projection'; export * from './src/lib/i18n'; export const markup = instance => renderToStaticMarkup(React.createElement(RefusalProjectionStatus, {instance}));`, resolveDir: process.cwd(), loader: 'tsx' }, bundle: true, write: false, platform: 'node', format: 'cjs' });
const ui = { exports: {} };
new Function('require', 'module', 'exports', compiled.outputFiles[0].text)(createRequire(import.meta.url), ui, ui.exports);
const { actualProjectionState, setTranslationLanguage, tr, markup } = ui.exports;
test('actual INFO and same-child request history drive status; requested settings/API acceptance do not', () => {
  const instance = { status: 'ready', phase: 'idle', refusalProjection: { vectorVerified: true }, metrics: { engine: { cvec: 'project:4-44', native_child: { alive: true, generation: 'child-1' } }, requests: [] } };
  assert.equal(actualProjectionState(instance), 'loaded');
  instance.metrics.requests = [{ projection: true, projection_child_generation: 'child-1', finish: 'stop' }];
  assert.equal(actualProjectionState(instance), 'applied');
  instance.metrics.requests[0].projection = false; assert.equal(actualProjectionState(instance), 'request-off');
  instance.metrics.engine.native_child.generation = 'child-2'; assert.equal(actualProjectionState(instance), 'loaded');
  instance.metrics.engine.native_child.alive = false; assert.equal(actualProjectionState(instance), 'unobserved');
  instance.metrics.engine.native_child.alive = true; instance.metrics.engine.cvec = 0; assert.equal(actualProjectionState(instance), 'off');
  instance.metrics.engine.cvec = 'add:4-44'; assert.equal(actualProjectionState(instance), 'other');
  instance.metrics.engine.cvec = 'project:4-44'; instance.refusalProjection.vectorVerified = false; assert.equal(actualProjectionState(instance), 'other');
  delete instance.metrics.engine.cvec; assert.equal(actualProjectionState(instance), 'unobserved');
  instance.status = 'error'; assert.equal(actualProjectionState(instance), 'unobserved');
});

test('Japanese UI copy names refusal-direction removal and every catalog supplies the new keys', async () => {
  await setTranslationLanguage('ja');
  assert.equal(tr('Refusal-direction removal'), '拒否方向ベクトルの除去');
  assert.match(tr('Engine reports: vector not loaded (off).'), /未読み込み/);
  for (const lang of ['en', 'ja', 'zh', 'de', 'es', 'ru', 'pt', 'ko', 'fr']) {
    const catalog = JSON.parse(readFileSync(`src/locales/fix-${lang}.json`, 'utf8'));
    for (let n = 1; n <= 18; n++) assert.ok(catalog[`rp${String(n).padStart(2, '0')}`]?.trim(), lang);
  }
  assert.match(markup(undefined), /data-refusal-projection="unobserved"/);
  assert.match(markup(undefined), /実際のベクトル状態/);
});

test('ordinary and agent requests propagate explicit ON and OFF; defaults retain previous request shape', async t => {
  const seen = [];
  const server = http.createServer(async (req, res) => {
    let body = ''; for await (const chunk of req) body += chunk;
    seen.push(JSON.parse(body)); res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.end('data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n');
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  t.after(() => new Promise(r => { server.closeAllConnections(); server.close(r); }));
  let expert = {};
  const proxy = createLlmProxy({ targetOf: () => `http://127.0.0.1:${server.address().port}/v1`, samplingNow: () => samplingFromExpert(expert) });
  const agentServer = http.createServer((req, res) => proxy(req, res, 'fixture', '/v1/chat/completions'));
  await new Promise(r => agentServer.listen(0, '127.0.0.1', r));
  t.after(() => new Promise(r => { agentServer.closeAllConnections(); agentServer.close(r); }));
  const agentRequest = async () => {
    const response = await fetch(`http://127.0.0.1:${agentServer.address().port}/v1/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ messages: [] }) });
    assert.equal(response.status, 200); await response.text();
  };
  for (const value of [false, true]) {
    update({ expert: { 'sample.speedProjection': value } });
    await proxyCompletion({ baseUrl: `http://127.0.0.1:${server.address().port}/v1`, message: 'hello', onFrame: () => {} });
    assert.equal(seen.at(-1).experimental_speed_projection, value);
    expert = { 'sample.speedProjection': value }; await agentRequest();
    assert.equal(seen.at(-1).experimental_speed_projection, value, 'the real agent proxy forwards the explicit switch');
  }
  update({ expert: {} });
  assert.equal(samplingFromExpert({}), null);
  await proxyCompletion({ baseUrl: `http://127.0.0.1:${server.address().port}/v1`, message: 'hello', onFrame: () => {} });
  assert.equal(seen.at(-1).experimental_speed_projection, true, 'existing ordinary default stays byte-identical');
  expert = {}; await agentRequest(); assert.deepEqual(seen.at(-1), { messages: [] }, 'unconfigured agent payload is unchanged');
});
