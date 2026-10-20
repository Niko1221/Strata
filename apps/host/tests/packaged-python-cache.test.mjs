import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { tmpdir } from 'node:os';
import { bundledPython, controlEnvironment, installerEnvironment } from '../server/control-runtime.mjs';

test('packaged Control and installer forbid bytecode writes even when the caller sets zero', () => {
  for (const inherited of [undefined, '0', '1']) {
    const env = { STRATA_PACKAGED: '1', ...(inherited === undefined ? {} : { PYTHONDONTWRITEBYTECODE: inherited }) };
    assert.equal(controlEnvironment(null, { env }).PYTHONDONTWRITEBYTECODE, '1');
    assert.equal(installerEnvironment(null, { env }).PYTHONDONTWRITEBYTECODE, '1');
    assert.equal(controlEnvironment(null, { env, acquire: true }).PYTHONDONTWRITEBYTECODE, '1');
    assert.equal(env.PYTHONDONTWRITEBYTECODE, inherited, 'the caller environment stays unchanged');
  }
});

test('source Control keeps the existing bytecode preference and does not add one by default', () => {
  for (const packaged of [undefined, '0']) {
    for (const inherited of [undefined, '0', '1']) {
      const env = {
        ...(packaged === undefined ? {} : { STRATA_PACKAGED: packaged }),
        ...(inherited === undefined ? {} : { PYTHONDONTWRITEBYTECODE: inherited }),
      };
      const result = controlEnvironment(null, { env });
      assert.equal(result.PYTHONDONTWRITEBYTECODE, inherited);
      assert.equal('PYTHONDONTWRITEBYTECODE' in result, inherited !== undefined);
    }
  }
});

const controlDir = '/Applications/Strata.app/Contents/Resources/control';
const python = bundledPython(controlDir);

function snapshot(directory) {
  const rows = [];
  function visit(current) {
    for (const entry of readdirSync(current, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      const path = join(current, entry.name);
      rows.push([relative(directory, path), entry.isDirectory() ? 'directory' : createHash('sha256').update(readFileSync(path)).digest('hex')]);
      if (entry.isDirectory()) visit(path);
    }
  }
  visit(directory);
  return rows;
}

test('the real bundled Python imports fresh and stale-cache modules without creating or changing bytecode', {
  skip: existsSync(python) ? false : 'The installed Strata bundled Python is unavailable.',
  timeout: 10000,
}, t => {
  const directory = realpathSync(mkdtempSync(join(tmpdir(), 'strata-packaged-python-cache-')));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const fresh = join(directory, 'fresh_probe.py');
  const stale = join(directory, 'stale_probe.py');
  writeFileSync(fresh, 'value = 41\n');
  writeFileSync(stale, 'value = 42\n');
  const inherited = { ...process.env, STRATA_PACKAGED: '1', PYTHONPYCACHEPREFIX: join(directory, 'cache') };
  delete inherited.PYTHONDONTWRITEBYTECODE;
  const env = controlEnvironment({ dir: controlDir }, { env: inherited, components: join(directory, 'components') });
  const run = (args, options = {}) => {
    const result = spawnSync(python, args, { cwd: directory, env, encoding: 'utf8', timeout: 5000, ...options });
    assert.equal(result.error, undefined, result.error?.message);
    assert.equal(result.status, 0, result.stderr);
    return result.stdout.trim();
  };
  // Keep every possible cache write in this temporary fixture, even if the
  // packaged environment regresses. The setup query itself explicitly uses -B.
  const cacheFile = run(['-B', '-c', 'import importlib.util, sys; print(importlib.util.cache_from_source(sys.argv[1]))', stale]);
  assert.ok(relative(directory, cacheFile).startsWith('cache/'));
  mkdirSync(dirname(cacheFile), { recursive: true });
  const existingBytes = Buffer.from('A stale signed bytecode file must remain byte-identical.');
  writeFileSync(cacheFile, existingBytes);
  const before = snapshot(directory);
  // No -B: the environment passed to real packaged Control must prevent writes.
  assert.equal(run(['-c', 'import fresh_probe, stale_probe; print(fresh_probe.value + stale_probe.value)']), '83');
  assert.deepEqual(snapshot(directory), before, 'imports neither add cache files nor replace a stale cache');
  assert.deepEqual(readFileSync(cacheFile), existingBytes);
  assert.equal(existsSync(join(directory, '__pycache__')), false);
  const writable = { ...env };
  delete writable.PYTHONDONTWRITEBYTECODE;
  run(['-c', 'import fresh_probe, stale_probe'], { env: writable });
  assert.notDeepEqual(snapshot(directory), before, 'the same writable fixture caches imports without the packaged guard');
  assert.notDeepEqual(readFileSync(cacheFile), existingBytes, 'the control run replaces the invalid cache');
});
