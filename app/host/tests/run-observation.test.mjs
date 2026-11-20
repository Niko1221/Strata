/** PR-A: RunObservation schema and VRAM separation. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
  RUN_OBSERVATION_SCHEMA_VERSION,
  createRunObservation,
  sealRunObservation,
  parseBackendAllocLog,
  compareContextCapacity,
  observationIdentity,
} from '../server/run-observation.mjs';
import {
  parseNvidiaMemoryCsv,
  parseWindowsGpuAdapterMemory,
  mergeVramObservation,
  nullVramObservation,
} from '../server/vram-observation.mjs';

test('RunObservation separates requested vs effective and never invents missing numbers', () => {
  const obs = createRunObservation({
    backend: { kind: 'prism', executablePath: null, version: 'fixture' },
    model: { id: 'local-hikari', metadataSignature: 'abc', fileRevision: 'c42fdf9' },
    requested: {
      context: 131072,
      kType: 'q8_0',
      vType: 'q8_0',
      gpuPlacement: 'Automatic',
      ngl: null,
      batch: 2048,
      ubatch: 512,
      parallel: 1,
      flashAttention: true,
      expertCache: null,
      memoryMode: 'ordinary',
    },
  });
  assert.equal(obs.schemaVersion, RUN_OBSERVATION_SCHEMA_VERSION);
  assert.equal(obs.requested.context, 131072);
  assert.equal(obs.effective, null);
  assert.equal(obs.vram.dedicated.totalMib, null);
  assert.equal(obs.vram.shared.usedMib, null);
  assert.equal(obs.cachePolicy, null);
  assert.ok(obs.diagnosticId);
  assert.doesNotMatch(JSON.stringify(obs), /sk-|api[_-]?key|Bearer /i);
});

test('sealRunObservation records effective context and fails comparison when shrunk', () => {
  const base = createRunObservation({
    backend: { kind: 'native', executablePath: 'C:\\\\fake\\\\strata.exe', version: '0.4.0' },
    model: { id: 'm', metadataSignature: 'sig' },
    requested: { context: 131072, kType: 'q8_0', vType: 'q8_0', gpuPlacement: 'All layers', ngl: 999,
      batch: 2048, ubatch: 512, parallel: 1, flashAttention: true, expertCache: 512, memoryMode: 'hakua-private' },
  });
  const sealed = sealRunObservation(base, {
    effective: {
      context: 32768,
      argv: ['--max-context', '32768', '-ngl', '999'],
      gpuLayers: null,
      offloadReported: 'unknown',
      binarySha256: 'deadbeef',
    },
    vram: nullVramObservation({ source: 'none', observedAt: '2026-11-02T00:00:00.000Z' }),
    processes: [{ name: 'LM Studio', pid: 4242, owner: null }],
  });
  const cmp = compareContextCapacity(sealed);
  assert.equal(cmp.ok, false);
  assert.equal(cmp.requested, 131072);
  assert.equal(cmp.effective, 32768);
  assert.match(cmp.reason, /smaller|shrunk|mismatch/i);
  assert.equal(sealed.effective.gpuLayers, null);
  assert.equal(sealed.effective.offloadReported, 'unknown');
});

test('parseBackendAllocLog leaves unknowns as unknown, never guesses GiB', () => {
  const parsed = parseBackendAllocLog([
    'loading model',
    'llama_kv:      KV buffer size =  4352.00 MiB',
    'load_tensors: offloaded 40/64 layers to GPU',
    'available context size = 131072 tokens',
  ].join('\n'));
  assert.equal(parsed.effectiveContext, 131072);
  assert.equal(parsed.gpuLayersOffloaded, 40);
  assert.equal(parsed.gpuLayersTotal, 64);
  assert.equal(parsed.kvAllocatedMib, 4352);
  assert.equal(parsed.weightsAllocatedMib, null);
  assert.equal(parsed.recurrentAllocatedMib, null);
  assert.equal(parsed.oom, false);

  const empty = parseBackendAllocLog('nothing useful');
  assert.equal(empty.effectiveContext, null);
  assert.equal(empty.gpuLayersOffloaded, null);
  assert.equal(empty.kvAllocatedMib, null);
  assert.equal(empty.oom, false);

  const oom = parseBackendAllocLog('CUDA error: out of memory');
  assert.equal(oom.oom, true);
});

test('nvidia-smi CSV maps to dedicated VRAM only; shared stays separate', () => {
  const csv = 'NVIDIA GeForce RTX 5060 Ti, 16311 MiB, 9481 MiB, 6570 MiB, 261 MiB\n';
  const nv = parseNvidiaMemoryCsv(csv);
  assert.equal(nv.length, 1);
  assert.equal(nv[0].dedicated.totalMib, 16311);
  assert.equal(nv[0].dedicated.usedMib, 9481);
  assert.equal(nv[0].dedicated.freeMib, 6570);
  assert.equal(nv[0].dedicated.reservedMib, 261);
  assert.equal(nv[0].shared.usedMib, null);
  assert.equal(nv[0].shared.totalMib, null);
});

test('Windows GPU adapter memory keeps DedicatedUsage and SharedUsage apart', () => {
  const rows = parseWindowsGpuAdapterMemory([
    { Name: 'luid_0x0_0x1_phys_0', DedicatedUsage: 0, SharedUsage: 8192, TotalCommitted: 598016 },
    { Name: 'luid_0x0_0x2_phys_0', DedicatedUsage: 9957437440, SharedUsage: 215961600, TotalCommitted: 10424496128 },
  ]);
  assert.equal(rows.length, 2);
  assert.equal(rows[1].dedicated.usedMib, Math.round(9957437440 / 1024 / 1024));
  assert.equal(rows[1].shared.usedMib, Math.round(215961600 / 1024 / 1024));
  assert.notEqual(rows[1].dedicated.usedMib, rows[1].shared.usedMib);
});

test('Windows missing, invalid and blank memory counters stay unavailable; measured zero survives', () => {
  for (const value of [null, undefined, '', ' ', false, true, [], {}, -1, 'NaN', 'Infinity']) {
    const [row] = parseWindowsGpuAdapterMemory({Name:'fixture',DedicatedUsage:value,SharedUsage:value,TotalCommitted:value});
    assert.equal(row.dedicated.usedMib,null,String(value));
    assert.equal(row.shared.usedMib,null,String(value));
    assert.equal(row.totalCommittedMib,null,String(value));
  }
  const [zero] = parseWindowsGpuAdapterMemory({Name:'fixture',DedicatedUsage:0,SharedUsage:'0'});
  assert.equal(zero.dedicated.usedMib,0);
  assert.equal(zero.shared.usedMib,0);
});

test('mergeVramObservation never coerces missing fields to 0', () => {
  const merged = mergeVramObservation({
    nvidia: [{
      name: 'RTX 5060 Ti',
      dedicated: { totalMib: 16311, usedMib: 1000, freeMib: 15000, reservedMib: null },
      shared: { totalMib: null, usedMib: null },
    }],
    windows: [{
      luid: 'luid_x',
      dedicated: { usedMib: 950 },
      shared: { usedMib: 200 },
    }],
    observedAt: '2026-11-02T01:00:00.000Z',
  });
  assert.equal(merged.dedicated.totalMib, 16311);
  assert.equal(merged.dedicated.reservedMib, null);
  assert.equal(merged.shared.usedMib, 200);
  assert.equal(merged.shared.totalMib, null);
  assert.ok(merged.sources.includes('nvidia-smi'));
  assert.ok(merged.sources.includes('windows-gpu-counters'));

  const missing = nullVramObservation({ source: 'unavailable', observedAt: '2026-11-02T01:00:00.000Z' });
  assert.equal(missing.dedicated.totalMib, null);
  assert.equal(missing.dedicated.usedMib, null);
  assert.equal(missing.shared.usedMib, null);
  assert.notEqual(missing.dedicated.totalMib, 0);
});

test('observationIdentity hashes non-secret metadata only', () => {
  const a = observationIdentity({
    backendKind: 'prism',
    binarySha256: 'aa',
    metadataSignature: 'bb',
    context: 131072,
    kType: 'q8_0',
    vType: 'q8_0',
    gpuPlacement: 'Automatic',
    memoryMode: 'ordinary',
  });
  const b = observationIdentity({
    backendKind: 'prism',
    binarySha256: 'aa',
    metadataSignature: 'bb',
    context: 131072,
    kType: 'q8_0',
    vType: 'q8_0',
    gpuPlacement: 'Automatic',
    memoryMode: 'ordinary',
  });
  const c = observationIdentity({
    backendKind: 'prism',
    binarySha256: 'aa',
    metadataSignature: 'bb',
    context: 65536,
    kType: 'q8_0',
    vType: 'q8_0',
    gpuPlacement: 'Automatic',
    memoryMode: 'ordinary',
  });
  assert.equal(a, b);
  assert.notEqual(a, c);
  assert.equal(a, createHash('sha256').update(JSON.stringify({
    backendKind: 'prism', binarySha256: 'aa', metadataSignature: 'bb',
    context: 131072, kType: 'q8_0', vType: 'q8_0', gpuPlacement: 'Automatic', memoryMode: 'ordinary',
  })).digest('hex').slice(0, 32));
});
