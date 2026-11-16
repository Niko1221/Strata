import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import fs from 'node:fs';
import { syncBuiltinESMExports } from 'node:module';
import { join } from 'node:path';
import { nativeFixture, finishJob } from './helpers/native-fixture.mjs';

const configOf = f => join(f.repo, 'strata-iq2_xs.json');
async function cleanup(f) {
  // RED must also drain an incorrectly admitted job before removing its isolated files.
  for (const job of f.control.state().jobs.filter(j => j.state === 'running')) await finishJob(f.control, job.id);
  await f.control.refresh(true);
  await f.cleanup();
}

for (const packaged of [false, true]) for (const validFirst of [true, false]) test(`${packaged ? 'packaged' : 'source'} load ${validFirst ? 'uses a valid' : 'refuses an invalid'} captured config even if the file changes after reading`, async t => {
  const f = nativeFixture({ packaged, draftingPlan: async () => null });
  try {
    await f.control.refresh(true);
    const path = configOf(f), good = JSON.parse(readFileSync(path, 'utf8'));
    good.parallel = 2;
    const valid = '\uFEFF' + JSON.stringify(good), invalid = JSON.stringify({ ...good, parallel: 99 });
    writeFileSync(path, validFirst ? valid : invalid);
    // Let status discovery finish normally. Replace the real file immediately after
    // the first read following the model-path check, before a second read can occur.
    const realpath = fs.realpathSync, read = fs.readFileSync;
    let capture = false;
    t.mock.method(fs, 'realpathSync', (...args) => {
      const result = realpath(...args);
      if (args[0] === path) capture = true;
      return result;
    });
    t.mock.method(fs, 'readFileSync', (...args) => {
      const result = read(...args);
      if (capture && args[0] === path) {
        capture = false;
        writeFileSync(path, validFirst ? invalid : valid);
      }
      return result;
    });
    syncBuiltinESMExports();
    if (validFirst) {
      const done = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs' }).id);
      assert.equal(done.state, 'done', done.error);
      const instance = f.control.resolveInstance(done.result.instanceId);
      assert.equal(JSON.parse(read(instance.configPath, 'utf8')).parallel, 2);
    } else {
      const jobs = f.control.state().jobs.length, journal = read(join(f.directory, 'native-jobs.json'), 'utf8');
      assert.throws(() => f.control.operation('load', { model: 'iq2_xs' }), /"parallel" must be a whole number/);
      assert.equal(f.control.state().jobs.length, jobs);
      assert.equal(read(join(f.directory, 'native-jobs.json'), 'utf8'), journal);
      assert.equal(f.control.state().instances.length, 0);
      assert.equal(f.calls.some(c => c.args.some(a => /serve[/\\]server\.py$/.test(String(a)))), false);
    }
  } finally {
    t.mock.restoreAll();
    syncBuiltinESMExports();
    await cleanup(f);
  }
});

test('the settings file is found, checked, and opened in the text editor', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh();
    const info = f.control.settingsFile('iq2_xs');
    assert.equal(info.path, configOf(f));
    assert.equal(info.name, 'strata-iq2_xs.json');
    assert.equal(info.ok, true);
    assert.match(info.note, /apply the next time the model starts/);
    assert.ok(info.issues.every(i => i.level === 'warning'), 'the fixture’s own extra key is a warning at most');
    const seen = [];
    const opened = await f.control.openSettingsFile('iq2_xs', { recorder: seen, platform: 'win32' });
    assert.deepEqual(seen, [['notepad.exe', info.path]]);
    assert.equal(opened.name, 'strata-iq2_xs.json');
    assert.throws(() => f.control.settingsFile('../etc/passwd'), /installed model/);
  } finally { await cleanup(f); }
});

test('a mistake in the file holds the start back with its line; a typo’d key only warns', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const path = configOf(f), good = readFileSync(path, 'utf8');
    // 1. a syntax slip (missing comma): not valid JSON, said with the line, and nothing is started
    writeFileSync(path, JSON.stringify(JSON.parse(good), null, 2).replace('"gpu": 0,', '"gpu": 0\n  "x": 1,'));
    assert.equal(f.control.settingsFile('iq2_xs').ok, false);
    assert.throws(() => f.control.operation('load', { model: 'iq2_xs' }), /^Error: The settings file for this model has a mistake, so the model was not started\. Line \d+: The settings file is not valid JSON/);
    assert.equal(f.control.state().jobs.length, 0);
    assert.equal(f.control.state().instances.length, 0, 'no engine was started');
    assert.equal(f.calls.filter(c => /strata\.py|serve/.test(JSON.stringify(c.args))).length, 0);
    // 2. a bad value
    const config = JSON.parse(good); config.parallel = 99; config.lazy_load = 'yes';
    writeFileSync(path, JSON.stringify(config, null, 2));
    assert.throws(() => f.control.operation('load', { model: 'iq2_xs' }), error => {
      assert.match(error.message, /"parallel" must be a whole number from 1 to 8, not 99/);
      assert.match(error.message, /"lazy_load" must be true or false, not "yes"/);
      return true;
    });
    assert.equal(f.control.state().jobs.length, 0);
    // 3. an unknown key only warns and the model loads
    const typo = JSON.parse(good); typo.paralel = 2;
    writeFileSync(path, JSON.stringify(typo, null, 2));
    const info = f.control.settingsFile('iq2_xs');
    assert.equal(info.ok, true);
    assert.ok(info.issues.some(i => i.code === 'unknown-key-near' && i.params.near === 'parallel'));
    const done = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(done.state, 'done', done.error);
  } finally { await cleanup(f); }
});

for (const [name, text, message] of [
  ['malformed JSON', '{"args": [}', /not valid JSON/],
  ['non-object JSON', '[]', /must hold one object/],
  ['invalid known value', '{"parallel":99}', /"parallel" must be a whole number/],
  ['incomplete Spin8 pair', '{"spin8_native_config":"capsule.json"}', /must be set together/],
  ['Spin8 off with capsule', JSON.stringify({spin8_mode:'off',spin8_native_config:'capsule.json',spin8_native_sha256:'a'.repeat(64)}), /cannot be used/],
]) test(`${name} is rejected synchronously without a job, write, stop or spawn`, async t => {
  const f = nativeFixture({ packaged: true });
  try {
    await f.control.refresh(true);
    const ready = await finishJob(f.control, f.control.operation('load', {model:'iq3_s'}).id);
    assert.equal(ready.state, 'done', ready.error);
    const path = configOf(f); writeFileSync(path, text);
    const before = {jobs:f.control.state().jobs.length,calls:f.calls.length,effects:f.effects.length};
    const journal = readFileSync(join(f.directory, 'native-jobs.json'), 'utf8'), writes = [];
    const write = fs.writeFileSync;
    const spy = t.mock.method(fs, 'writeFileSync', (...args) => { writes.push(String(args[0])); return write(...args); });
    syncBuiltinESMExports();
    let error;
    try { f.control.operation('load', {model:'iq2_xs'}); } catch (e) { error = e; }
    finally { spy.mock.restore(); syncBuiltinESMExports(); }
    assert.deepEqual({jobs:f.control.state().jobs.length,calls:f.calls.length,effects:f.effects.length,writes}, {...before,writes:[]});
    assert.ok(error instanceof Error, 'refusal must be synchronous'); assert.match(error.message, message);
    assert.equal(readFileSync(join(f.directory, 'native-jobs.json'), 'utf8'), journal);
    assert.equal(readFileSync(path, 'utf8'), text);
    assert.equal(f.control.state().instances[0].status, 'ready');
  } finally { t.mock.restoreAll(); syncBuiltinESMExports(); await cleanup(f); }
});
