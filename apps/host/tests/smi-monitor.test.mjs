// SMI2000: where the experts are (VRAM, RAM, NVMe) on the Monitor's "Memory and experts" card.
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { originalDashboard } from '../server/monitor-dashboard.mjs';

const built = await build({ stdin: { contents: "export { expertTiers, kvDetails, vramSegments } from './src/lib/smi-expert-tiers';", resolveDir: process.cwd(), loader: 'ts' }, bundle: true, write: false, format: 'esm', platform: 'node' });
const { expertTiers, kvDetails, vramSegments } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);

test('Qwen: VRAM from the GPU cache, RAM from arena_mib, the rest on disk (measured on an RTX 3060 12 GB, 48 GB RAM, Q6_K_XL)', () => {
  const t = expertTiers({ used: 609, total: 24576, memoryGb: 2572 / 1024, totalGb: 104187 / 1024, ramGb: 24571 / 1024, source: 'engine' });
  assert.equal(t.vram.count, 609);
  assert.equal(t.ram.approx, true);
  assert.ok(Math.abs(t.ram.count - 5796) <= 2, String(t.ram.count));
  assert.equal(t.vram.count + t.ram.count + t.disk.count, 24576);
  assert.ok(Math.abs(t.vram.pct + t.ram.pct + t.disk.pct - 100) < 1e-6);
  assert.ok(Math.abs(t.disk.gb - (104187 - 2572 - 24571) / 1024) < 1e-6);
});

test('a full RAM copy never counts an expert twice', () => {
  const t = expertTiers({ used: 100, total: 1000, memoryGb: 1, totalGb: 10, ramGb: 10, source: 'engine' });
  assert.equal(t.disk.count, 0);
  assert.equal(t.ram.count, 900);
});

test('DeepSeek: counts from ram_experts, no sizes', () => {
  const t = expertTiers({ used: 300, total: 1000, ramUsed: 500, source: 'engine' });
  assert.deepEqual([t.vram.count, t.ram.count, t.disk.count], [300, 500, 200]);
  assert.equal(t.ram.approx, false);
  assert.equal(t.disk.pct, 20);
});

test('no total or no RAM figure: nothing is guessed', () => {
  assert.equal(expertTiers({ used: 609, total: NaN, memoryGb: 2.5, source: 'engine' }), null);
  assert.equal(expertTiers({ used: 609, total: 24576, memoryGb: 2.5, source: 'engine' }), null);
  assert.equal(expertTiers({ used: 0, total: 0, source: 'none' }), null);
});

test('the dashboard mapping carries the new engine facts', () => {
  const fallback = { context: { maxTokens: 4096 }, hardware: { gpus: [] }, totals: {}, telemetry: {} };
  const qwen = originalDashboard({ live: { state: 'idle' }, engine: { expert_slots: 609, expert_cache_mib: 2572, experts: 24576, expert_total_mib: 104187, arena_mib: 24571 } }, null, fallback).expertCache;
  assert.equal(qwen.total, 24576); assert.equal(qwen.totalGb, 104187 / 1024); assert.equal(qwen.ramGb, 24571 / 1024);
  const ds4 = originalDashboard({ live: { state: 'idle' }, engine: { expert_slots: 300, cpu_experts: 700, ram_experts: 500 } }, null, fallback).expertCache;
  assert.equal(ds4.total, 1000); assert.equal(ds4.ramUsed, 500);
});

test('the KV format: k8v4 with the server estimate, streaming said, an unknown mode named as it is', () => {
  const kv = kvDetails({ kv: 'k8v4', kv_mib_est: 820, context: 65536, kv_resident: 0 });
  assert.equal(kv.format, '8-bit keys, 4-bit values.');
  assert.ok(Math.abs(kv.gb - 820 / 1024) < 1e-9);
  assert.ok(kv.kbPerToken > 11 && kv.kbPerToken < 11.3, String(kv.kbPerToken));
  assert.equal(kv.streaming, false);
  assert.equal(kvDetails({ kv: 'int8', kv_resident: 32768 }).streaming, true);
  assert.equal(kvDetails({ kv: 'fp4' }).gb, null);
  assert.equal(kvDetails({ kv: 'zz9' }).format, 'zz9');
  assert.equal(kvDetails({}), null);
});

test('the VRAM bar: the engine parts, the KV estimate, the rest and the free memory fill the card exactly', () => {
  const engine = { vram_parts: { weights: 4480, draft: 1152, buffers: 114 }, expert_cache_mib: 2572, kv_mib_est: 820, context: 65536, kv_resident: 0 };
  const s = vramSegments(engine, 11670 / 1024, 12);
  assert.deepEqual(s.map(x => x.id), ['weights', 'draft', 'experts', 'kv', 'buffers', 'other', 'free']);
  assert.ok(Math.abs(s.reduce((n, x) => n + x.pct, 0) - 100) < 1e-9);
  assert.ok(Math.abs(s.find(x => x.id === 'other').gb - (11670 - 9138) / 1024) < 1e-9);
  assert.equal(s.find(x => x.id === 'kv').approx, true);
  const over = vramSegments(engine, 4, 12);   // parts above what the sensor reads: scaled down, never over 100%
  assert.ok(Math.abs(over.reduce((n, x) => n + x.pct, 0) - 100) < 1e-9);
  assert.equal(vramSegments({ kv_mib_est: 800 }, 8, 12), null);
  assert.equal(vramSegments(engine, null, 12), null);
});

test('the dashboard mapping passes the engine facts and the sampling on', () => {
  const fallback = { context: { maxTokens: 4096 }, hardware: { gpus: [] }, totals: {}, telemetry: {} };
  const sampling = { config: { temperature: 0.2 }, thinking: { temperature: 0.2 }, instruct: { temperature: 0.2 } };
  const out = originalDashboard({ live: { state: 'idle' }, engine: { kv: 'k8v4' }, sampling }, null, fallback);
  assert.equal(out.engineInfo.kv, 'k8v4');
  assert.deepEqual(out.sampling, sampling);
});

test('the host clears a loaded model\'s requests through its server and keeps the fresh figures', async () => {
  const { nativeFixture, finishJob } = await import('./helpers/native-fixture.mjs');
  const f = nativeFixture();
  try {
    await f.control.refresh();
    const done = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(done.state, 'done', done.error);
    const state = await f.control.resetMetrics(done.result.instanceId);
    const instance = state.instances.find(i => i.id === done.result.instanceId);
    assert.equal(instance.metrics.engine.max_context, 32768);
    assert.equal('key' in instance, false);
    await assert.rejects(() => f.control.resetMetrics('no-such-model'), /loaded model/);
  } finally { await f.cleanup(); }
});
