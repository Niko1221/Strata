import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { saferStart } from '../server/native-resources.mjs';
import { Q36_CAPABILITY_REVISION, Q36_SUPPORTED_FLAGS, isQwen36Config } from '../server/q36-run-contract.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');

test('the capability table matches the flags strata-q36 actually reads', () => {
  const source = readFileSync(join(root, 'src', 'qwen35', 'q36_engine.cu'), 'utf8');
  const start = source.indexOf('int strata::qwen35::run');
  const run = source.slice(start, source.indexOf('if (model.empty())', start));
  const found = [...run.matchAll(/a == "(--[^"]+)"/g)].map(match => match[1]);
  assert.ok(found.length > 10);
  // The native parser recognises this name only to explicitly refuse it.
  assert.match(run, /else if \(a == "--vram-reserve-mib"[\s\S]*?not an option of this engine[\s\S]*?return 2;/);
  for (const flag of found.filter(f => !['--help','--vram-reserve-mib'].includes(f))) assert.equal(Q36_SUPPORTED_FLAGS.has(flag), true, flag);
  assert.equal(Q36_SUPPORTED_FLAGS.has('--resident-budget-gib'), false);
  assert.equal(Q36_SUPPORTED_FLAGS.has('--vram-reserve-mib'), false);
  assert.equal(Q36_SUPPORTED_FLAGS.has('--cache-type-k'), true);
  assert.equal(Q36_SUPPORTED_FLAGS.has('--q36-expert-store'), true);
  assert.equal(Q36_SUPPORTED_FLAGS.has('--q36-memory'), true);
  assert.equal(Q36_SUPPORTED_FLAGS.has('--prompt-cache'), true);
  assert.equal(Q36_SUPPORTED_FLAGS.has('--conversation-cache-mib'), true);
  assert.equal(Q36_SUPPORTED_FLAGS.has('--private-conversation-cache'), true);
  assert.equal(Q36_CAPABILITY_REVISION, 'q36-typed-kv-1');
});

test('Qwen3.6 does not get a resident-budget or shorter-context retry; another family still does', () => {
  const hardware = { cpu: { cores: 12 }, ram_gb: 32, gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 5060 Ti', vram_gb: 16 }] };
  const q36 = { arch: 'qwen35moe', model_name: 'qwen3.6-35b-a3b-ud-q4_k_xl', args: ['--max-context', '65536'] };
  assert.equal(isQwen36Config(q36), true);
  assert.equal(saferStart(q36, hardware, { arenaGb: 18.75 }, 'server_exited: no host memory for the experts (19195 MiB)'), null);
  assert.equal(saferStart(q36, hardware, { arenaGb: 18.75 }, 'server_exited: out of memory'), null);
  const flash = saferStart(
    { gpu: 0, args: ['--native', 'x.gguf', '--max-context', '32768'] },
    { cpu: { cores: 12 }, ram_gb: 63, gpus: [{ index: 0, vram_gb: 12 }] },
    { arenaGb: 50.3 },
    'server_exited: memory guard: 1.9 GiB free',
  );
  assert.equal(flash.config.args.includes('--resident-budget-gib'), true);
  assert.equal(isQwen36Config({ model_name: 'Qwen iq3_s', args: ['--max-context', '32768'] }), false);
});
