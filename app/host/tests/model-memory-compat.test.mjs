/** PR-B: architecture-aware hybrid memory estimator fixtures. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { ggufFixture } from './helpers/local-gguf.mjs';
import { readGGUF } from '../server/gguf.mjs';
import {
  GiB,
  ggmlRowBytes,
  classifyHybridLayers,
  extractHybridGeometry,
  estimateAttentionKvBytes,
  estimateModelMemory,
} from '../server/model-memory.mjs';

const HIKARI = {
  arch: 'qwen35',
  nLayers: 64,
  interval: 4,
  nKvHeads: 4,
  keyLength: 256,
  valueLength: 256,
  contextLimit: 262144,
};

function hikariFixture(file, extras = []) {
  ggufFixture(file, {
    arch: HIKARI.arch,
    name: 'Hikari fixture',
    context: HIKARI.contextLimit,
    omit: [
      `${HIKARI.arch}.block_count`,
      `${HIKARI.arch}.attention.head_count`,
      `${HIKARI.arch}.attention.head_count_kv`,
      `${HIKARI.arch}.embedding_length`,
    ],
    extra: [
      [`${HIKARI.arch}.block_count`, 4, HIKARI.nLayers],
      [`${HIKARI.arch}.embedding_length`, 4, 2048],
      [`${HIKARI.arch}.attention.head_count`, 4, 16],
      [`${HIKARI.arch}.attention.head_count_kv`, 4, HIKARI.nKvHeads],
      [`${HIKARI.arch}.attention.key_length`, 4, HIKARI.keyLength],
      [`${HIKARI.arch}.attention.value_length`, 4, HIKARI.valueLength],
      [`${HIKARI.arch}.full_attention_interval`, 4, HIKARI.interval],
      ...extras,
    ],
  });
}

test('ggml row bytes follow block sizes (not naive INT8=1)', () => {
  assert.equal(ggmlRowBytes('q8_0', 1024), 1024 * 34 / 32);
  assert.equal(ggmlRowBytes('q4_0', 1024), 1024 * 18 / 32);
  assert.equal(ggmlRowBytes('f16', 1024), 2048);
  assert.equal(ggmlRowBytes('bf16', 1024), 2048);
  assert.equal(ggmlRowBytes('f32', 1024), 4096);
  assert.throws(() => ggmlRowBytes('turbo99', 32), /Unsupported/);
  assert.throws(() => ggmlRowBytes('q8_0', 0), /positive/);
  assert.throws(() => ggmlRowBytes('q8_0', 33), /whole cache blocks/);
  assert.throws(() => ggmlRowBytes('constructor', 32), /Unsupported/);
});

test('legacy attention helper pads per sequence and rejects a nonaligned head even if combined heads align',()=>{
 const shape={nFullAttention:16,nKvHeads:4,keyLength:256,valueLength:256,kType:'q8_0',vType:'f16'};
 assert.equal(estimateAttentionKvBytes({...shape,context:1,sequences:2}),estimateAttentionKvBytes({...shape,context:256,sequences:2}));
 assert.equal(estimateAttentionKvBytes({...shape,context:256,keyLength:8}),null);
 assert.equal(estimateAttentionKvBytes({...shape,context:256,kType:'turbo2'}),null);
});

test('Hikari: 64 layers / interval 4 → attention 16, recurrent 48', () => {
  const layers = classifyHybridLayers({
    nLayerAll: 64,
    fullAttentionInterval: 4,
  });
  assert.equal(layers.nFullAttention, 16);
  assert.equal(layers.nRecurrent, 48);
  assert.equal(layers.confidence, 'high');
  assert.equal(layers.source, 'full_attention_interval');
});

test('explicit attention.recurrent_layers array takes priority when sizes agree', () => {
  const recurrent = Array.from({ length: 48 }, (_, i) => (i % 4 === 3 ? i + 1 : i)).filter((_, i) => i % 4 !== 3);
  // Build the true recurrent index list: layers not equal to interval-1 mod interval
  const recurrentIdx = [];
  for (let i = 0; i < 64; i++) if (i % 4 !== 3) recurrentIdx.push(i);
  const layers = classifyHybridLayers({
    nLayerAll: 64,
    fullAttentionInterval: 4,
    recurrentLayers: recurrentIdx,
  });
  assert.equal(layers.nFullAttention, 16);
  assert.equal(layers.nRecurrent, 48);
  assert.equal(layers.source, 'attention.recurrent_layers');
});

test('mismatched recurrent_layers array is unknown, never invents fits=true', () => {
  const layers = classifyHybridLayers({
    nLayerAll: 64,
    fullAttentionInterval: 4,
    recurrentLayers: [0, 1, 2], // too short
  });
  assert.equal(layers.confidence, 'low');
  assert.equal(layers.nFullAttention, null);
  assert.equal(layers.nRecurrent, null);
  assert.match(layers.reason, /recurrent_layers/);
});

test('Hikari Q8_0 Attention KV at 131072 tokens is exactly 4.25 GiB', () => {
  const bytes = estimateAttentionKvBytes({
    context: 131072,
    sequences: 1,
    nFullAttention: 16,
    nKvHeads: 4,
    keyLength: 256,
    valueLength: 256,
    kType: 'q8_0',
    vType: 'q8_0',
  });
  assert.equal(bytes, 4.25 * GiB);
  assert.equal(Number((bytes / GiB).toFixed(2)), 4.25);
});

test('Hikari Q4_0 Attention KV at 131072 tokens is exactly 2.25 GiB', () => {
  const bytes = estimateAttentionKvBytes({
    context: 131072,
    sequences: 1,
    nFullAttention: 16,
    nKvHeads: 4,
    keyLength: 256,
    valueLength: 256,
    kType: 'q4_0',
    vType: 'q4_0',
  });
  assert.equal(bytes, 2.25 * GiB);
  assert.equal(Number((bytes / GiB).toFixed(2)), 2.25);
});

test('Hikari FP16 Attention KV at 131072 tokens is exactly 8.00 GiB', () => {
  const bytes = estimateAttentionKvBytes({
    context: 131072,
    sequences: 1,
    nFullAttention: 16,
    nKvHeads: 4,
    keyLength: 256,
    valueLength: 256,
    kType: 'f16',
    vType: 'f16',
  });
  assert.equal(bytes, 8 * GiB);
  assert.equal(Number((bytes / GiB).toFixed(2)), 8);
});

test('mixed K=q8_0 V=f16 yields an intermediate Attention KV size', () => {
  const mixed = estimateAttentionKvBytes({
    context: 131072,
    sequences: 1,
    nFullAttention: 16,
    nKvHeads: 4,
    keyLength: 256,
    valueLength: 256,
    kType: 'q8_0',
    vType: 'f16',
  });
  const q8 = 4.25 * GiB;
  const f16 = 8 * GiB;
  assert.ok(mixed > q8 && mixed < f16);
  // K half of q8 pair + V half of f16 pair = 2.125 + 4.0 = 6.125 GiB
  assert.equal(Number((mixed / GiB).toFixed(3)), 6.125);
});

test('extractHybridGeometry reads Hikari GGUF metadata', t => {
  const root = mkdtempSync(join(tmpdir(), 'strata-hybrid-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const file = join(root, 'Hikari.gguf');
  hikariFixture(file);
  const geo = extractHybridGeometry(readGGUF(file).metadata);
  assert.equal(geo.architecture, 'qwen35');
  assert.equal(geo.nLayerAll, 64);
  assert.equal(geo.nFullAttention, 16);
  assert.equal(geo.nRecurrent, 48);
  assert.equal(geo.nKvHeads, 4);
  assert.equal(geo.keyLength, 256);
  assert.equal(geo.valueLength, 256);
  assert.equal(geo.contextLimit, 262144);
  assert.equal(geo.confidence, 'high');
  assert.ok(geo.metadataSignature);
});

test('estimateModelMemory separates disk bytes, Attention KV, and recurrent unknowns', t => {
  const root = mkdtempSync(join(tmpdir(), 'strata-hybrid-mem-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const file = join(root, 'Hikari.gguf');
  hikariFixture(file);
  const info = readGGUF(file);
  const plan = estimateModelMemory({
    metadata: info.metadata,
    diskBytes: info.size,
    context: 131072,
    sequences: 1,
    kType: 'q8_0',
    vType: 'q8_0',
  });
  assert.equal(plan.nFullAttention, 16);
  assert.equal(plan.nRecurrent, 48);
  assert.equal(plan.attentionKvBytes, 4.25 * GiB);
  assert.equal(plan.attentionKvGiB, 4.25);
  assert.equal(plan.diskBytes, info.size);
  assert.notEqual(plan.diskBytes, plan.weightsAllocatedBytes);
  assert.equal(plan.weightsAllocatedBytes, null, 'GPU weight allocation is not the GGUF disk size');
  assert.equal(plan.recurrentBytes, null, 'recurrent state needs backend-specific layout');
  assert.equal(plan.fits, null, 'never claim fits from optimistic arithmetic alone');
  assert.equal(plan.confidence, 'high');
  assert.match(plan.notes.join(' '), /Attention KV/);
});

test('unknown architecture stays unqualified through the legacy plan adapter', () => {
  const plan = estimateModelMemory({
    metadata: {
      'general.architecture': 'future_text_model',
      'future_text_model.block_count': 32,
      'future_text_model.attention.head_count_kv': 8,
      'future_text_model.attention.key_length': 128,
      'future_text_model.attention.value_length': 128,
      'future_text_model.context_length': 8192,
    },
    diskBytes: 1_000_000,
    context: 4096,
    sequences: 1,
    kType: 'f16',
    vType: 'f16',
  });
  assert.equal(plan.confidence, 'low');
  assert.equal(plan.nFullAttention, null, 'an unknown architecture has no validated attention map');
  assert.equal(plan.nRecurrent, null);
  assert.equal(plan.attentionKvBytes, null);
  assert.equal(plan.fits, null);
});

test('abnormal metadata never returns NaN or negative budgets', () => {
  for (const metadata of [
    { 'general.architecture': 'qwen35', 'qwen35.block_count': 0, 'qwen35.full_attention_interval': 4 },
    { 'general.architecture': 'qwen35', 'qwen35.block_count': 64, 'qwen35.full_attention_interval': 4,
      'qwen35.attention.head_count_kv': 0, 'qwen35.attention.key_length': 256, 'qwen35.attention.value_length': 256 },
    { 'general.architecture': 'qwen35', 'qwen35.block_count': 64, 'qwen35.full_attention_interval': 4,
      'qwen35.attention.head_count_kv': 4, 'qwen35.attention.key_length': 1e12, 'qwen35.attention.value_length': 256 },
  ]) {
    const plan = estimateModelMemory({
      metadata,
      diskBytes: 100,
      context: 131072,
      sequences: 1,
      kType: 'q8_0',
      vType: 'q8_0',
    });
    assert.equal(plan.fits, null);
    if (plan.attentionKvBytes !== null) {
      assert.ok(Number.isFinite(plan.attentionKvBytes));
      assert.ok(plan.attentionKvBytes >= 0);
    }
    assert.notEqual(plan.attentionKvBytes, Number.NaN);
  }
});
