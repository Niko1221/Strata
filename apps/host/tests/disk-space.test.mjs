import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DEFAULT_MIN_FREE_BYTES, ensureFreeSpace, freeBytesAt, minimumFreeBytes } from '../server/disk-space.mjs';

const GiB = 1024 ** 3;

test('the default floor is 2 GiB and a hard disk below it is refused before a download', () => {
  assert.equal(DEFAULT_MIN_FREE_BYTES, 2 * GiB);
  assert.equal(minimumFreeBytes({}), 2 * GiB);
  assert.equal(minimumFreeBytes({ STRATA_MIN_FREE_GIB: '1.5' }), 1.5 * GiB);

  assert.throws(
    () => ensureFreeSpace('C:\\Strata', { freeBytes: 1.2 * GiB, label: 'a model' }),
    /not enough free space/i,
  );
  assert.throws(
    () => ensureFreeSpace('C:\\Strata', { freeBytes: 1.2 * GiB, label: 'a model' }),
    /2\.0 GB/,
  );
  assert.doesNotThrow(() => ensureFreeSpace('C:\\Strata', { freeBytes: 3 * GiB }));
});

test('a known download size joins the floor', () => {
  assert.throws(
    () => ensureFreeSpace('C:\\Strata', { freeBytes: 3 * GiB, need: 4 * GiB, label: 'a 4 GB model' }),
    /4\.0 GB/,
  );
  assert.doesNotThrow(() => ensureFreeSpace('C:\\Strata', { freeBytes: 6 * GiB, need: 4 * GiB }));
});

test('an unknown filesystem never blocks a download and the caller can tune the floor', () => {
  assert.deepEqual(ensureFreeSpace('C:\\Strata', { freeBytes: null }), { free: null, required: null, ok: true });
  assert.doesNotThrow(() => ensureFreeSpace('C:\\Strata', { freeBytes: 1.5 * GiB, env: { STRATA_MIN_FREE_GIB: '1' } }));
});

test('a real folder answers with a non-negative number or null', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-disk-'));
  try {
    const free = freeBytesAt(dir);
    assert.equal(typeof free, 'number');
    assert.ok(free >= 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
