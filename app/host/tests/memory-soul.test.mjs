import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { copyFileSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import test from 'node:test';

function fixture(t) {
  const root = mkdtempSync(join(tmpdir(), 'strata-soul-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const repo = join(root, 'repo'), home = join(root, 'user'), cwd = join(root, 'work');
  const module = join(repo, 'apps/host/server/memory.mjs');
  mkdirSync(join(repo, 'apps/host/server'), { recursive: true });
  mkdirSync(join(home, '.hermes'), { recursive: true });
  mkdirSync(cwd);
  copyFileSync(new URL('../server/memory.mjs', import.meta.url), module);
  const soul = (directory, text) => {
    mkdirSync(directory, { recursive: true });
    writeFileSync(join(directory, 'SOUL.md'), text);
  };
  soul(join(home, '.hermes'), '  SYNTHETIC_HOME  \n');
  soul(repo, '  SYNTHETIC_REPO  \n');
  const read = override => {
    const script = `
      const override = JSON.parse(process.argv[1]);
      process.env = { ...process.env };
      if (override === null) delete process.env.HERMES_HOME;
      else process.env.HERMES_HOME = override;
      const { getSoulDirective } = await import(${JSON.stringify(pathToFileURL(module).href)});
      process.stdout.write(JSON.stringify(getSoulDirective()));`;
    const result = spawnSync(process.execPath, ['--input-type=module', '-e', script, JSON.stringify(override)], {
      cwd, env: { ...process.env, USERPROFILE: home, HOME: home }, encoding: 'utf8', timeout: 10_000,
    });
    assert.equal(result.status, 0, result.stderr);
    return JSON.parse(result.stdout);
  };
  return { root, repo, home, cwd, soul, read };
}

test('absent and exactly empty home preserve trimmed home then repository fallback', t => {
  const f = fixture(t);
  for (const override of [null, '']) assert.equal(f.read(override), 'SYNTHETIC_HOME');
  f.soul(join(f.home, '.hermes'), ' \n ');
  for (const override of [null, '']) assert.equal(f.read(override), 'SYNTHETIC_REPO');
  rmSync(join(f.home, '.hermes/SOUL.md'));
  mkdirSync(join(f.home, '.hermes/SOUL.md'));
  for (const override of [null, '']) assert.equal(f.read(override), 'SYNTHETIC_REPO');
  f.soul(f.repo, '  ');
  assert.equal(f.read(''), '');
});

test('explicit home reads exclusively and missing, empty or unreadable files fail closed', t => {
  const f = fixture(t), explicit = join(f.root, 'explicit');
  assert.equal(f.read(explicit), '');
  f.soul(explicit, ' \n SYNTHETIC_EXPLICIT \n ');
  assert.equal(f.read(explicit), 'SYNTHETIC_EXPLICIT');
  f.soul(explicit, '\n \t');
  assert.equal(f.read(explicit), '');
  rmSync(join(explicit, 'SOUL.md'));
  mkdirSync(join(explicit, 'SOUL.md'));
  assert.equal(f.read(explicit), '');
});

test('relative and spaced paths keep their raw whitespace', t => {
  const f = fixture(t);
  f.soul(join(f.cwd, ' relative home'), 'RELATIVE_WITH_LEADING_SPACE');
  assert.equal(f.read(' relative home'), 'RELATIVE_WITH_LEADING_SPACE');
  f.soul(join(f.cwd, 'space inside'), 'INTERNAL_SPACE');
  assert.equal(f.read('./space inside'), 'INTERNAL_SPACE');
  f.soul(f.cwd, 'CWD_MUST_NOT_BE_SELECTED_BY_TRIMMING');
  assert.equal(f.read(' '), '');
  assert.equal(f.read(resolve(f.cwd, 'space inside')), 'INTERNAL_SPACE');
});

test('current-user tilde forms select the synthetic user root', t => {
  const f = fixture(t);
  f.soul(f.home, 'CURRENT_USER');
  f.soul(join(f.home, 'custom home'), 'CURRENT_USER_CUSTOM');
  assert.equal(f.read('~'), 'CURRENT_USER');
  for (const value of ['~/custom home', '~\\custom home']) assert.equal(f.read(value), 'CURRENT_USER_CUSTOM');
});

test('malformed overrides never use the synthetic fallback files', t => {
  const f = fixture(t);
  for (const value of ['bad\0path', 'bad\rpath', 'bad\npath', '~other-user', '~other-user/custom']) {
    assert.equal(f.read(value), '', JSON.stringify(value));
  }
  if (process.platform === 'win32') {
    for (const value of ['C:relative', 'C:', '/drive-less', '\\drive-less']) assert.equal(f.read(value), '', value);
    const explicit = join(f.root, 'explicit');
    f.soul(explicit, 'AMBIGUOUS_PATH_MUST_NOT_LOAD');
    // Both ambiguous spellings would otherwise reach this existing synthetic file.
    for (const value of [explicit.slice(2), explicit.slice(2).replaceAll('\\', '/'), `${explicit.slice(0, 2)}../explicit`]) {
      assert.equal(f.read(value), '', value);
    }
  }
});
