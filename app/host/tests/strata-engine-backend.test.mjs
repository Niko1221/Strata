/** Hikari PQ2_0 must load via Strata’s PrismML engine, not native Q2_0 preparation. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { transform } from 'esbuild';
import { readFile } from 'node:fs/promises';

const source = await readFile(new URL('../src/lib/strata-engine-backend.ts', import.meta.url), 'utf8');
const { code } = await transform(source, { loader: 'ts', format: 'esm' });
const { resolveStrataEngineBackend, pinStrataEngineBackend } = await import(
  `data:text/javascript;base64,${Buffer.from(code).toString('base64')}`
);

test('PQ2_0 local GGUF resolves to PrismML without a hint', () => {
  assert.equal(resolveStrataEngineBackend({ local: {}, quant: 'PQ2_0' }), 'PrismML');
  assert.equal(resolveStrataEngineBackend({ local: {}, quant: 'PTQ1_0' }), 'TurboQuant');
  assert.equal(resolveStrataEngineBackend({ local: {}, quant: 'Q4_K_M' }), null);
  assert.equal(resolveStrataEngineBackend({ quant: 'PQ2_0' }), null);
});

test('explicit Load with Strata hint wins over quant', () => {
  assert.equal(resolveStrataEngineBackend({ local: {}, quant: 'Q4_K_M' }, 'PrismML'), 'PrismML');
  assert.equal(resolveStrataEngineBackend({ local: {}, quant: 'PQ2_0' }, 'TurboQuant'), 'TurboQuant');
});

test('pinStrataEngineBackend sets runtime.backend for Hikari', () => {
  const doc = { beginner: {}, expert: { 'ctx.length': '32K' } };
  const pinned = pinStrataEngineBackend(doc, { local: true, quant: 'PQ2_0' });
  assert.equal(pinned.expert['runtime.backend'], 'PrismML');
  assert.equal(pinned.expert['ctx.length'], '32K');
  assert.notEqual(pinned, doc);
  assert.equal(pinStrataEngineBackend(pinned, { local: true, quant: 'PQ2_0' }), pinned);
});
