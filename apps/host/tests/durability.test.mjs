import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { createNativeControl } from '../server/native-control.mjs';
import { nativeFixture, finishJob } from './helpers/native-fixture.mjs';

test('a truncated native-jobs.json is read from .bak after a restart', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh();
    const job = await finishJob(f.control, f.control.operation('update-strata').id);
    assert.equal(job.state, 'error');
    const file = join(f.directory, 'native-jobs.json');
    assert.ok(existsSync(`${file}.bak`), 'the previous operation list is kept');
    assert.ok(JSON.parse(readFileSync(file, 'utf8')).some(saved => saved.id === job.id));

    writeFileSync(file, '[{"id":');
    const restarted = createNativeControl({ directory: f.directory, defaultRepo: f.repo });
    try {
      const recovered = restarted.state().jobs;
      assert.ok(recovered.some(saved => saved.id === job.id), 'the operation list came from the backup');
      assert.match(recovered.find(saved => saved.id === job.id).error, /local edits/);
    } finally {
      await restarted.close();
    }
  } finally {
    await f.cleanup();
  }
});

test('a truncated settings file holds the start with a clear message and the good backup is kept', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh();
    const original = f.control.settings('iq2_xs');
    f.control.save('iq2_xs', { values: { ...original.values, parallel: 2 }, mcp: original.mcp });

    const file = join(f.repo, 'strata-iq2_xs.json');
    writeFileSync(file, '{"model": "iq2_xs", "args": [');
    let said = '';
    try {
      const loaded = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs' }).id);
      said = String(loaded.error ?? '');
      assert.equal(loaded.state, 'error', 'a damaged file does not start a model');
    } catch (error) { said = error instanceof Error ? error.message : String(error); }
    assert.match(said, /settings file for this model has a mistake|not valid JSON/i, said);
    assert.ok(readFileSync(`${file}.bak`, 'utf8').trim().endsWith('}'), 'the backup was not replaced by the truncated bytes');
  } finally {
    await f.cleanup();
  }
});

test('a truncated model configuration falls back to .bak and a later save keeps that good backup', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh();
    const original = f.control.settings('iq2_xs');
    f.control.save('iq2_xs', { values: { ...original.values, parallel: 2 }, mcp: original.mcp });

    const file = join(f.repo, 'strata-iq2_xs.json');
    assert.equal(JSON.parse(readFileSync(file, 'utf8')).parallel, 2);
    assert.ok(existsSync(`${file}.bak`));

    writeFileSync(file, '{"model":');
    const recovered = f.control.settings('iq2_xs');
    assert.equal(recovered.values.parallel, null, 'the backup is the configuration from before the failed save');

    f.control.save('iq2_xs', { values: { ...recovered.values, parallel: 3 }, mcp: recovered.mcp });
    assert.equal(JSON.parse(readFileSync(file, 'utf8')).parallel, 3);
    const backup = JSON.parse(readFileSync(`${file}.bak`, 'utf8'));
    assert.equal('parallel' in backup, false, 'the truncated bytes were not copied over the good backup');
    assert.ok(backup.unknown_native_key, 'the backup is still a complete configuration');
  } finally {
    await f.cleanup();
  }
});
