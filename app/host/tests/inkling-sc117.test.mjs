import test from 'node:test';
import assert from 'node:assert/strict';
import {
  INKLING_SCRATCH_RESERVE_BYTES,
  discoverInklingScratchCandidates,
  inklingExpertCacheDir,
  inklingScratchPayloadBytes,
  resolveInklingExpertCacheDir,
  sc117RunContext,
  withInklingContext,
} from '../server/inkling.mjs';

test('sc117RunContext validates range and applies --max-context without leaving --ctx behind', () => {
  assert.equal(sc117RunContext('inkling-2-mini-sc117-sq-10', 'inkling2_loop', 131072), 131072);
  assert.equal(sc117RunContext('custom-inkling-sc117', 'inkling2_loop', 98304, { custom: true }), 98304);
  assert.equal(sc117RunContext('inkling-2-mini-sq-10', 'inkling2_loop', 131072), undefined);
  assert.throws(
    () => sc117RunContext('inkling-2-mini-sc117-sq-15', 'inkling2_loop', 512),
    /1,024 and 1,048,576/
  );
  assert.deepEqual(
    withInklingContext(['--ctx', '8192', '--kv-dtype', 'k8v8', '--max-context', '16384'], 131072),
    ['--kv-dtype', 'k8v8', '--max-context', '131072']
  );
});

test('resolveInklingExpertCacheDir redirects low-space volume to a writable drive with enough headroom', () => {
  const GIB = 1024 * 1024 * 1024;
  assert.equal(INKLING_SCRATCH_RESERVE_BYTES, 5 * GIB);
  assert.equal(inklingScratchPayloadBytes('inkling-2-mini-sc117-sq-10'), Math.round(0.5 * GIB));
  assert.equal(inklingScratchPayloadBytes('inkling-2-mini-sc117-sq-15'), Math.round(2.75 * GIB));
  assert.equal(inklingScratchPayloadBytes('inkling-2-mini-sc117-sq-20'), Math.round(6.0 * GIB));

  const created = [];
  const freeByRoot = { 'C:': 5.9 * GIB, 'D:': 10.1 * GIB, 'E:': 0, 'H:': 4.0 * GIB };
  const exists = p => {
    const u = String(p).toUpperCase();
    return u.startsWith('C:') || u.startsWith('D:') || u.startsWith('E:') || u.startsWith('H:');
  };
  const statfs = p => {
    const root = String(p).slice(0, 2).toUpperCase();
    return { bsize: 4096, bavail: Math.floor((freeByRoot[root] ?? 0) / 4096) };
  };
  const baseArgs = ['--native', 'C:\\models\\sq15.strata', '--expert-cache-dir', 'C:\\Temp\\scratch', '--max-context', '131072'];
  assert.equal(inklingExpertCacheDir(baseArgs), 'C:\\Temp\\scratch');

  const candidates = discoverInklingScratchCandidates(
    { args: baseArgs, exe: 'C:\\bin\\strata-inkling.exe' },
    { env: { STRATA_INK_SCRATCH_DIRS: 'G:\\FastNVMe\\scratch' }, platform: 'win32', driveLetters: ['C', 'D', 'E', 'H'], exists, statfs }
  );
  assert.equal(candidates[0], 'G:\\FastNVMe\\scratch');

  const res = resolveInklingExpertCacheDir(baseArgs, { args: baseArgs, exe: 'C:\\bin\\strata-inkling.exe' }, 'inkling-2-mini-sc117-sq-15', {
    platform: 'win32',
    driveLetters: ['C', 'D', 'E', 'H'],
    exists,
    statfs,
    mkdir: dir => created.push(dir),
    canWrite: () => true,
  });
  assert.equal(res.redirectedFrom, 'C:\\Temp\\scratch');
  assert.equal(res.redirectedTo, 'D:\\Strata\\scratch\\strata-inkling-experts');
  assert.match(res.note, /D:\\Strata\\scratch\\strata-inkling-experts/);
  assert.equal(res.args[res.args.indexOf('--expert-cache-dir') + 1], 'D:\\Strata\\scratch\\strata-inkling-experts');

  assert.throws(
    () => resolveInklingExpertCacheDir(baseArgs, { args: baseArgs, exe: 'C:\\bin\\strata-inkling.exe' }, 'inkling-2-mini-sc117-sq-20', {
      platform: 'win32',
      driveLetters: ['C', 'D', 'E', 'H'],
      exists,
      statfs,
      mkdir: () => {},
      canWrite: () => true,
    }),
    /Not enough free disk space for Inkling Expert scratch.*USB/i
  );
});
