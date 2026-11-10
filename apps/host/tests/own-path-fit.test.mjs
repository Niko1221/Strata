/**
 * Qwen3.6-35B-A3B's conversation memory per token (GitHub #5715): the design catalog's figure is the one strata_control/own_path_fit.py derives from the
 * engine's allocations (47,104 B: fp32 K / V of the 10 attention layers and the MTP block, and the fp16 copy at the full context); it said 20 KB.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const repo = resolve(root, '../..');
const py = readFileSync(resolve(repo, 'strata_control/own_path_fit.py'), 'utf8');
const BYTES = Number(/^BYTES_PER_TOKEN = 11 \* 2 \* 256 \* 2 \* 4 \+ 2 \* 256 \* 2 \* 2/m.test(py) ? 11 * 2 * 256 * 2 * 4 + 2 * 256 * 2 * 2 : NaN);

test('the design catalog has Qwen3.6\'s conversation memory per token', () => {
  const src = readFileSync(resolve(root, 'server/catalog.mjs'), 'utf8');
  const kb = Number(/id:'qwen36'[^\n]*kvKbPerToken:(\d+)/.exec(src)?.[1]);
  assert.equal(BYTES, 47104);
  assert.equal(kb, Math.round(BYTES / 1024));
});
