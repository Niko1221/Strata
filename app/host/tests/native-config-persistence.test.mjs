/**
 * F04 / F05: load READY must not rewrite the on-disk settings snapshot, and a
 * mid-load edit of future settings must survive. Explicit save failures must not
 * stop a ready model.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { chmodSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { nativeFixture, finishJob } from './helpers/native-fixture.mjs';

async function waitHealth(f) {
  for (let i = 0; i < 400; i++) {
    if (f.hasPendingHealth()) return;
    await new Promise(r => setTimeout(r, 10));
  }
  throw new Error('load never reached health');
}

test('F05: edits made while a load is waiting for health stay on disk after READY', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const path = join(f.repo, 'strata-iq2_xs.json');

    f.setHealthWait(true);
    const job = f.control.operation('load', { model: 'iq2_xs', gpus: [0] });
    await waitHealth(f);

    const mid = { ...JSON.parse(readFileSync(path, 'utf8')), temperature: 0.8, model_name: 'edited-during-load' };
    writeFileSync(path, JSON.stringify(mid, null, 2));

    f.releaseHealth();
    const done = await finishJob(f.control, job.id);
    assert.equal(done.state, 'done', done.error);
    assert.equal(f.control.state().instances[0]?.status, 'ready');

    const after = JSON.parse(readFileSync(path, 'utf8'));
    assert.equal(after.temperature, 0.8, 'future settings must keep the mid-load edit');
    assert.equal(after.model_name, 'edited-during-load');
  } finally {
    await f.cleanup();
  }
});

test('F04: READY does not depend on rewriting the model settings file', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const path = join(f.repo, 'strata-iq2_xs.json');
    const stamp = readFileSync(path, 'utf8');

    f.setHealthWait(true);
    const job = f.control.operation('load', { model: 'iq2_xs', gpus: [0] });
    await waitHealth(f);

    writeFileSync(path, stamp);
    try { chmodSync(path, 0o444); } catch { /* best effort on Windows */ }

    f.releaseHealth();
    const done = await finishJob(f.control, job.id);
    assert.equal(done.state, 'done', done.error);
    assert.equal(f.control.state().instances[0]?.status, 'ready', 'model stays ready without settings rewrite');
  } finally {
    await f.cleanup();
  }
});

test('explicit save failure leaves a ready model running', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const loaded = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(loaded.state, 'done', loaded.error);
    const path = join(f.repo, 'strata-iq2_xs.json');
    const original = f.control.settings('iq2_xs');
    // A read-only destination inode is replaceable on POSIX when its parent is writable.
    // Occupy the atomic-write target instead: this deterministically fails before replacement on every OS.
    mkdirSync(path + '.ui-tmp');
    assert.throws(
      () => f.control.save('iq2_xs', { values: { ...original.values, parallel: 2 } }),
      /EACCES|EPERM|EISDIR|denied|read-only|EROFS|EBUSY|EIO|Cannot|access|permission/i,
    );
    assert.equal(f.control.state().instances[0]?.status, 'ready');
  } finally {
    await f.cleanup();
  }
});
