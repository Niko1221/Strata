import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { appendFileSync, mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { finishJob, nativeFixture } from './helpers/native-fixture.mjs';
import {
  admitLegacyExperts,
  bindLaunchPath,
  fileIdentitySync,
  gibToBytes,
  prepareQ36Launch,
  observeQ36Runtime,
  Q36_MAPPED_PACK_DIGESTS,
  q36FileSha256,
  q36PersistentKvBytes,
  sameModelIdentity,
} from '../server/q36-run-contract.mjs';

test('a C: selection is not launched from H:, and the file name is not the model identity', () => {
  const swapped = bindLaunchPath({ requestedPath: 'C:\\models\\model.gguf', resolvedPath: 'H:\\models\\model.gguf' });
  assert.equal(swapped.ok, false);
  assert.equal(swapped.error.code, 'path_drive_substituted');
  const root = mkdtempSync(join(tmpdir(), 'q36-id-'));
  try {
    const realDir = join(root, 'real');
    const otherDir = join(root, 'other');
    mkdirSync(realDir);
    mkdirSync(otherDir);
    const real = join(realDir, 'model.gguf');
    const twin = join(otherDir, 'model.gguf');
    writeFileSync(real, 'same-name-a');
    writeFileSync(twin, 'same-name-b');
    const a = fileIdentitySync(real);
    const b = fileIdentitySync(twin);
    assert.equal(a.basename, b.basename);
    assert.equal(sameModelIdentity(a, b), false);
    const spaced = join(root, 'my models');
    mkdirSync(spaced);
    const jp = join(spaced, 'モデル.gguf');
    writeFileSync(jp, 'unicode-path');
    const opened = fileIdentitySync(jp);
    assert.equal(opened.sizeBytes, String(Buffer.byteLength('unicode-path')));
    const link = join(root, 'link');
    symlinkSync(realDir, link, process.platform === 'win32' ? 'junction' : 'dir');
    const via = fileIdentitySync(join(link, 'model.gguf'));
    assert.equal(sameModelIdentity(a, via), true);
    assert.equal(sameModelIdentity({ fileId: null, contentSha256: 'abc', basename: 'model.gguf' }, { fileId: null, contentSha256: 'abc', basename: 'other.gguf' }), true);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('the launch contract is frozen from the snapshot that was checked', () => {
  const input = { args: ['--max-context', '65536', '--prefill', 'auto', '--mtp', '2'], userArgs: [], binarySha256: 'abc' };
  const out = prepareQ36Launch(input);
  input.args.push('--cache-type-v', 'turbo3');
  assert.equal(out.ok, true);
  assert.deepEqual(out.argv, ['--max-context', '65536', '--prefill', 'auto', '--mtp', '2']);
  assert.equal(out.contract.effective.prefill, null);
  assert.equal(out.contract.requested.prefill, 'auto');
  assert.equal(out.contract.effective.persistentKv, 'fp32');
  assert.equal(out.contract.effective.expertStoreMode, 'legacy');
  assert.equal(out.contract.effective.mtp, 2);
  assert.equal(out.preservedUserArgs.includes('--cache-type-v'), false);
});

test('an unobserved auto prefill stays unknown and requested/effective differences are explicit', () => {
  const contract = prepareQ36Launch({ generation: 'launch', args: ['--max-context', '65536', '--prefill', 'auto'] }).contract;
  assert.equal(observeQ36Runtime(contract, { launchGeneration: 'old' }), contract);
  assert.equal(contract.observed.status, 'unobserved');
  assert.equal(contract.effective.prefill, null);
  const state = { launchGeneration: 'launch', child: { generation: 'child', processId: 123 }, effective: {
    generation: 'child', processId: 123, context: 32768, prefill: 1024, parallel: 1, mtp: 0, kv_k: 'f16', kv_v: 'f32', kv_bytes: '9007199254740993',
  } };
  const result = observeQ36Runtime(contract, state);
  assert.equal(result.observed.status, 'mismatch');
  assert.deepEqual(result.observed.mismatches.map(item => item.field), ['context', 'kv_k']);
  assert.equal(result.effective.prefill, 1024);
  assert.equal(result.observed.kv_bytes, '9007199254740993');
  assert.equal(result.effective.persistentKv, 'fp32'); // Observation did not grant a new capability.
  assert.equal(observeQ36Runtime(contract, { ...state, child: { generation: 'next-child', processId: 124 } }), contract);
});

test('the real Host launch consumer owns the generation and resolves only its actual child log', async () => {
  const f = nativeFixture();
  const until = async predicate => {
    const deadline = Date.now() + 4000;
    while (Date.now() < deadline) { if (predicate()) return; await new Promise(resolve => setTimeout(resolve, 20)); }
    assert.fail('Host child telemetry was not observed');
  };
  try {
    await f.control.refresh(true);
    const path = join(f.repo, 'strata-iq3_s.json');
    const config = JSON.parse(readFileSync(path, 'utf8'));
    Object.assign(config, { arch: 'qwen35moe', model_name: 'Qwen3.6-35B-A3B', args: ['--max-context', '65536', '--prefill', 'auto'] });
    writeFileSync(path, JSON.stringify(config));
    f.setHealthWait(true);
    const job = f.control.operation('load', { model: 'iq3_s' });
    await until(() => f.hasPendingHealth());
    const call = f.calls.find(c => c.args?.some(arg => String(arg).endsWith('server.py')));
    const run = JSON.parse(readFileSync(call.args[call.args.indexOf('--config') + 1], 'utf8'));
    const instance = () => f.control.state().instances[0];
    const generation = run.q36_run.generation;
    assert.match(generation, /^[0-9a-f-]{36}$/);
    assert.equal(run.q36_run.runId, instance().id);
    assert.equal(instance().q36Run.observed.status, 'unobserved');
    const receipt = (launch, child, state, pid = 'null') => `Q36_CHILD schema=1 launch_generation=${launch} generation=${child} state=${state} process_id=${pid}\n`;
    appendFileSync(run.log, receipt('old-launch', 'old-child', 'begin') + receipt(generation, 'child-one', 'begin') + receipt(generation, 'child-one', 'spawned', '8765'));
    appendFileSync(run.log, 'Q36_EFFECTIVE schema=1 generation=old-child context=7 prefill=7\nQ36_EFFECTIVE schema=1 generation=child-one engine=strata-q36 context=65536 prefill=1024 parallel=1 mtp=0 kv_k=f32 kv_v=f32 kv_bytes=9007199254740993\n');
    await until(() => instance().q36Run.observed.status === 'observed');
    assert.equal(instance().q36Run.effective.prefill, 1024);
    assert.equal(instance().q36Run.observed.kv_bytes, '9007199254740993');
    assert.equal(instance().q36Run.observed.processId, 8765);
    assert.equal(instance().q36Run.generation, generation);
    // A server-owned child restart keeps its launch token but invalidates its predecessor's observation.
    appendFileSync(run.log, receipt(generation, 'child-two', 'begin') + receipt(generation, 'child-two', 'spawned', '8766'));
    await until(() => instance().q36Run.observed.status === 'unobserved');
    assert.equal(instance().q36Run.effective.prefill, null);
    assert.equal(instance().context, run.q36_run.planned.context);
    assert.equal(instance().parallel, run.q36_run.planned.parallel);
    appendFileSync(run.log, 'Q36_EFFECTIVE schema=1 generation=child-one prefill=4096\nQ36_EFFECTIVE schema=1 generation=child-two context=32768 prefill=512 parallel=2 mtp=0 kv_k=f32 kv_v=f32\n');
    await until(() => instance().q36Run.observed.status === 'mismatch');
    assert.equal(instance().q36Run.effective.prefill, 512);
    assert.equal(instance().q36Run.observed.processId, 8766);
    assert.equal(instance().context, 32768);
    assert.equal(instance().parallel, 2);
    // This child fails before emitting any effective values. Public metadata must not retain child-two's values.
    appendFileSync(run.log, receipt(generation, 'child-three', 'begin') + receipt(generation, 'child-three', 'spawned', '8767') +
      'Q36_STARTUP_DETAIL schema=1 generation=child-three stage=initial_slot_h2d state=error process_elapsed_ms=1000 stage_elapsed_ms=800 completed_bytes=384 total_bytes=1024 layer_index=3 error_code=cuda_error\n');
    await until(() => instance().q36Run.observed.status === 'unobserved' && instance().q36Detail?.state === 'error');
    assert.equal(instance().context, run.q36_run.planned.context);
    assert.equal(instance().parallel, run.q36_run.planned.parallel);
    assert.equal(instance().q36Run.effective.prefill, null);
    assert.equal(instance().q36Detail.completed_bytes, '384');
    f.releaseHealth();
    const loaded = await finishJob(f.control, job.id);
    assert.equal(loaded.state, 'done', loaded.error);
    assert.equal(instance().q36Run.observed.status, 'unobserved');
  } finally { f.releaseHealth(); await f.cleanup(); }
});

test('an explicit unsupported setting stops the launch and is not called applied', () => {
  for (const [flag, value] of [['--cache-type-k', 'q8_0'], ['--cache-type-v', 'turbo3'], ['--spin8-mode', 'consensus']]) {
    const denied = prepareQ36Launch({
      args: ['--max-context', '65536', flag, value],
      userArgs: [flag, value],
    });
    assert.equal(denied.ok, false, flag);
    assert.equal(denied.error.code, 'unsupported_setting');
    assert.match(denied.error.message, /not applied/);
    assert.equal(denied.argv, undefined);
  }
  const store = prepareQ36Launch({ args: ['--max-context', '65536'], requested: { expertStoreMode: 'mapped-pack' } });
  assert.equal(store.ok, false);
  assert.equal(store.error.code, 'expert_store_not_available');
  const bounded = prepareQ36Launch({ args: ['--max-context', '65536'], requested: { warmupMode: 'bounded' } });
  assert.equal(bounded.ok, false);
  assert.equal(bounded.error.code, 'unsupported_setting');
});

test('T08 refuse-before-serve: non-FP32 KV is not started (measured FP16 arm stays NOT_RUN)', () => {
  for (const storage of ['fp16', 'int8', 'q4_0', 'k8v4']) {
    const denied = prepareQ36Launch({
      args: ['--max-context', '65536'],
      requested: { persistentKv: storage },
    });
    assert.equal(denied.ok, false, storage);
    assert.equal(denied.error.code, 'unsupported_setting');
    assert.match(denied.error.message, /FP32 KV/);
    assert.equal(denied.argv, undefined);
  }
  for (const value of ['fp16', 'int8']) {
    const denied = prepareQ36Launch({
      args: ['--max-context', '65536', '--kv', value],
      userArgs: ['--kv', value],
    });
    assert.equal(denied.ok, false, `--kv ${value}`);
    assert.equal(denied.error.code, 'unsupported_setting');
    assert.match(denied.error.message, /not applied/);
    assert.equal(denied.argv, undefined);
  }
  const fp32 = prepareQ36Launch({
    args: ['--max-context', '65536'],
    requested: { persistentKv: 'fp32' },
  });
  assert.equal(fp32.ok, true);
  assert.equal(fp32.contract.effective.persistentKv, 'fp32');
});

test('the engine digest is read from the file, and a missing file is not an allowlisted digest', () => {
  const dir = mkdtempSync(join(tmpdir(), 'q36-sha-'));
  try {
    const file = join(dir, 'strata.exe');
    writeFileSync(file, 'q36-engine');
    assert.equal(q36FileSha256(file), createHash('sha256').update('q36-engine').digest('hex'));
    assert.equal(q36FileSha256(join(dir, 'missing.exe')), null);
    const forged = prepareQ36Launch({
      args: ['--model', 'm.gguf', '--max-context', '65536'],
      requested: { expertStoreMode: 'mapped-pack' },
      binarySha256: q36FileSha256(file),
    });
    assert.equal(forged.ok, false);
    assert.equal(forged.error.code, 'expert_store_not_available');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('the recorded sm_120 engine digests are the mapped-pack allowlist, and legacy stays the default', () => {
  const listed = [...Q36_MAPPED_PACK_DIGESTS].sort();
  assert.deepEqual(listed, [
    '659607ec2d7c19c65633e4b0e3bad5e0bdf30448f2092ca33cb2d4e1413b6256',
    '57d4dc6e06ef50e40d52539dd69ebf87c52f017f87fa2ea0e2621dadbc260c34',
    '0932216088cb3a5813175d595a5f1f7f16633f4a7883c438439ded7f5c3bc410',
    '13fc3b96fe71afcb98990d828d2c743f85e8b59b16174a1f7da3eb4ef9b4f8e3',
    '28c4dc3991fa05b900fa956c351720121088d9041202d6acefd9b7af2749370e',
    '2dd5dbcd2874767096b9c1eb4ac4132a96f7979bcd019ec6b779f23c9dbacc0f',
    '3797f2d368360b184b055c2130ed300645a1ece20d0bdef05dd6a3ceecf7e608',
    '512d7f8bba66b57fbe0f8b90807825d3b3172a86dd039452bc29e685ca27d046',
    '5e37033b11e6478920450f39a3b0057c2ca6d4116b1e196a1b507bf344d4fbd0',
    '64226fde7905821148c78b7de85ede4d291c4df8184d80866dccba014d7c02b7',
    '6fe110650940d0c19f09909d769393769650045e6bc6eab29ccc49734dbcab29',
    '907341c74d5a9f26c784e4aaad3460e376d4cabd84635c5d032879c59b08d150',
    'c86b5d36e5371d8276bc73d553d2be0149a674ccc74541fa51d7eb88a2ab55a0',
    'dd53df07789e0fefcb87e33acab54f5bfe1faf312ef6bad515924736b715c5b5',
    'e75716a9d5deb6289e87741d42e7003ba82b7696f6121d5203e615d9fe596a96',
    'e79b9791bdfdacd1e5f20324b8e80978af287cc1dc464d22231da8819de0f7fc',
  ].sort());
  for (const sha of listed) assert.match(sha, /^[0-9a-f]{64}$/);
  for (const sha of listed) {
    const bound = prepareQ36Launch({
      args: ['--model', 'm.gguf', '--max-context', '65536'],
      requested: { expertStoreMode: 'mapped-pack' },
      binarySha256: sha,
    });
    assert.equal(bound.ok, true, sha);
    assert.equal(bound.contract.effective.expertStoreMode, 'mapped-pack');
  }
  const settingsDigest = { binarySha256: listed[1], expertStoreMode: 'mapped-pack' };
  const untouched = prepareQ36Launch({
    args: ['--model', 'm.gguf', '--max-context', '65536'],
    binarySha256: listed[1],
    modelSettings: settingsDigest,
  });
  assert.equal(untouched.ok, true);
  assert.equal(untouched.contract.effective.expertStoreMode, 'legacy');
  assert.equal(untouched.argv.includes('--q36-expert-store'), false);
});

test('mapped-pack stays off the command line unless this engine digest is allowlisted', () => {
  const sha = 'ab'.repeat(32);
  const other = 'cd'.repeat(32);
  const allowed = prepareQ36Launch({
    args: ['--model', 'm.gguf', '--max-context', '65536'],
    requested: { expertStoreMode: 'mapped-pack' },
    binarySha256: sha,
    mappedPackDigests: [sha],
  });
  assert.equal(allowed.ok, true);
  assert.equal(allowed.argv[allowed.argv.indexOf('--q36-expert-store') + 1], 'mapped-pack');
  assert.equal(allowed.contract.effective.expertStoreMode, 'mapped-pack');
  const denied = prepareQ36Launch({
    args: ['--model', 'm.gguf', '--q36-expert-store', 'mapped-pack'],
    binarySha256: other,
    mappedPackDigests: [sha],
  });
  assert.equal(denied.ok, false);
  assert.equal(denied.error.code, 'expert_store_not_available');
  assert.equal(denied.argv, undefined);
  const junk = prepareQ36Launch({
    args: ['--model', 'm.gguf', '--q36-expert-store', 'spin'],
    userArgs: ['--q36-expert-store', 'spin'],
    binarySha256: sha,
    mappedPackDigests: [sha],
  });
  assert.equal(junk.ok, false);
  assert.equal(junk.error.code, 'unsupported_setting');
});

test('app-owned flags that this engine ignores are left out of argv and kept in the saved user list', () => {
  const userArgs = ['--pool-workers', '4'];
  const out = prepareQ36Launch({
    args: ['--max-context', '65536', '--vram-reserve-mib', '1536', '--resident-budget-gib', '12', '--pool-workers', '4'],
    userArgs,
  });
  assert.equal(out.ok, true);
  assert.equal(out.argv.includes('--vram-reserve-mib'), false);
  assert.equal(out.argv.includes('--resident-budget-gib'), false);
  assert.deepEqual(out.argv, ['--max-context', '65536', '--pool-workers', '4']);
  assert.match(out.note, /not applied/);
  assert.match(out.note, /Saved settings were kept/);
  assert.deepEqual(out.preservedUserArgs, userArgs);
});

test('private-cache flags reach runtime negotiation without a binary allowlist', () => {
  const required = prepareQ36Launch({ args: ['--max-context', '65536'], privacyRequired: true });
  assert.equal(required.ok, true);
  assert.ok(required.argv.includes('--private-conversation-cache'));
  const present = prepareQ36Launch({ args: ['--max-context', '65536', '--prompt-cache', '0', '--conversation-cache-mib', '0'], userArgs: [] });
  assert.equal(present.ok, true);
  assert.deepEqual(present.argv, ['--max-context', '65536', '--prompt-cache', '0', '--conversation-cache-mib', '0']);
});

test('FP32 KV bytes and the legacy 18.75 GiB arena use commit headroom, not a shared-VRAM sum', () => {
  assert.equal(q36PersistentKvBytes(65536, 1).bytesDecimal, '2952790016');
  assert.equal(q36PersistentKvBytes(65536, 2).bytesDecimal, '5905580032');
  assert.equal(q36PersistentKvBytes(131072, 1).bytesDecimal, '5905580032');
  assert.equal(q36PersistentKvBytes(16384, 1).bytesDecimal, String(45056n * 16384n));
  assert.equal(q36PersistentKvBytes(Number.NaN, 1).ok, false);
  const expert = gibToBytes('18.75');
  const head = gibToBytes('11.89');
  assert.equal(admitLegacyExperts({ expertBytes: expert.bytesDecimal, commitHeadroomBytes: head.bytesDecimal }).fits, false);
  assert.equal(admitLegacyExperts({ expertBytes: null, commitHeadroomBytes: head.bytesDecimal }).fits, null);
  assert.notEqual(q36PersistentKvBytes(131072, 1).bytes, 4563402752n);
});

test('loading Qwen3.6 drops ignored memory flags and refuses an explicit cache flag before spawn', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const path = join(f.repo, 'strata-iq3_s.json');
    const config = JSON.parse(readFileSync(path, 'utf8'));
    config.arch = 'qwen35moe';
    config.model_name = 'Qwen3.6-35B-A3B';
    config.args = ['--max-context', '65536', '--prefill', '2048', '--mtp', '2', '--resident-budget-gib', '8'];
    writeFileSync(path, JSON.stringify(config));
    const loaded = await finishJob(f.control, f.control.operation('load', { model: 'iq3_s', resourceProfile: 'performance' }).id);
    assert.equal(loaded.state, 'done', loaded.error);
    const spawn = f.calls.find(call => call.args?.some(arg => String(arg).endsWith('server.py')));
    const configPath = spawn.args[spawn.args.indexOf('--config') + 1];
    const launched = JSON.parse(readFileSync(configPath, 'utf8'));
    assert.equal(launched.args.includes('--vram-reserve-mib'), false);
    assert.equal(launched.args.includes('--resident-budget-gib'), false);
    assert.ok(launched.args.includes('--max-context'));
    assert.equal(launched.q36_run.effective.context, 65536);
    assert.equal(launched.q36_run.effective.prefill, 2048);
    assert.equal(launched.q36_run.effective.persistentKv, 'fp32');
    await f.control.unload(loaded.result.instanceId);
    const handwritten = JSON.parse(readFileSync(path, 'utf8'));
    handwritten.args = ['--max-context', '65536', '--vram-reserve-mib', '1536'];
    writeFileSync(path, JSON.stringify(handwritten));
    const reserve = await finishJob(f.control, f.control.operation('load', { model: 'iq3_s' }).id);
    assert.equal(reserve.state, 'error');
    assert.match(reserve.error, /--vram-reserve-mib 1536/);
    assert.match(reserve.error, /not applied/);
    config.user_args = ['--cache-type-k', 'q8_0'];
    config.args = ['--max-context', '65536', '--cache-type-k', 'q8_0'];
    writeFileSync(path, JSON.stringify(config));
    const jobs = f.control.state().jobs.length, calls = f.calls.length, effects = f.effects.length;
    const journal = readFileSync(join(f.directory, 'native-jobs.json'), 'utf8');
    assert.throws(() => f.control.operation('load', { model: 'iq3_s' }), error => {
      assert.equal(error.code, 'unsupported_serving_setting');
      assert.match(error.message, /cache_type_k.*not verified/);
      return true;
    });
    assert.equal(f.control.state().jobs.length, jobs);
    assert.equal(f.calls.length, calls);
    assert.equal(f.effects.length, effects);
    assert.equal(readFileSync(join(f.directory, 'native-jobs.json'), 'utf8'), journal);
    assert.deepEqual(JSON.parse(readFileSync(path, 'utf8')), config);
  } finally {
    await f.cleanup();
  }
});
