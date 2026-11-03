import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { defaultHostPort, devConfigFile, devSummary, isDevMode, loadDevConfig } from '../server/dev-mode.mjs';
import { enginePort } from '../server/ports.mjs';
import { shellPrefsFile } from '../server/shell-prefs.mjs';
import { stateRoot, tuningDir } from '../server/platform-paths.mjs';

test('Dev mode is on only for STRATA_DEV=1, and then the host and engine ports are not Live ones', () => {
  assert.equal(isDevMode({}), false);
  assert.equal(isDevMode({ STRATA_DEV: '0' }), false);
  assert.equal(isDevMode({ STRATA_DEV: '1' }), true);
  assert.equal(defaultHostPort({}), 8791);
  assert.equal(defaultHostPort({ STRATA_DEV: '1' }), 8792);
  assert.equal(enginePort({}), 8080);
  assert.equal(enginePort({}, 18500), 18500);
  assert.equal(enginePort({ STRATA_ENGINE_PORT: '8081' }), 8081);
  assert.equal(enginePort({ STRATA_ENGINE_PORT: '8081' }, 18500), 8081);
  for (const bad of ['', 'x', '80', '70000', '8081.5']) assert.equal(enginePort({ STRATA_ENGINE_PORT: bad }), 8080, bad);
});

test('dev.config.json: missing is the empty config, broken is a warning and never a throw', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-dev-config-'));
  try {
    const file = join(dir, 'dev.config.json');
    const missing = loadDevConfig({ file });
    assert.deepEqual(missing.warnings, []);
    assert.equal(missing.config.badge.label, 'DEV');

    writeFileSync(file, '{ "badge": { "label": "DEV" }, ');
    const broken = loadDevConfig({ file });
    assert.equal(broken.warnings.length, 1);
    assert.match(broken.warnings[0], /not valid JSON/);
    assert.deepEqual(broken.config.features, {});

    writeFileSync(file, '[1, 2]');
    assert.match(loadDevConfig({ file }).warnings[0], /one JSON object/);

    writeFileSync(file, '\uFEFF{ "badge": { "label": "  TEST  " }, "features": [], "custom": 3 }');
    const odd = loadDevConfig({ file });
    assert.equal(odd.config.badge.label, 'TEST');
    assert.deepEqual(odd.config.features, {});
    assert.equal(odd.config.custom, 3, 'keys the host does not know yet are kept for later phases');
    assert.match(odd.warnings[0], /"features" must be an object/);
    const summary = devSummary(odd);
    assert.deepEqual({ label: summary.label, configFile: summary.configFile, warnings: summary.warnings }, { label: 'TEST', configFile: file, warnings: odd.warnings });
    assert.equal(summary.verbose, false);
    assert.deepEqual(summary.tokens, {});

    const unreadable = loadDevConfig({ file, read: () => { throw Object.assign(new Error('denied'), { code: 'EACCES' }); } });
    assert.match(unreadable.warnings[0], /could not be read/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('the repository dev.config.json is valid and STRATA_DEV_CONFIG names another file', () => {
  const repo = loadDevConfig({ file: devConfigFile({}) });
  assert.deepEqual(repo.warnings, []);
  assert.equal(devConfigFile({ STRATA_DEV_CONFIG: join(tmpdir(), 'x.json') }), join(tmpdir(), 'x.json'));
});

test('Dev data folders: shell prefs follow STRATA_DESKTOP_DATA_DIR and the state root follows STRATA_STATE_DIR', () => {
  const live = { LOCALAPPDATA: 'C:\\Users\\me\\AppData\\Local' };
  assert.equal(stateRoot(live, 'win32'), 'C:\\Users\\me\\AppData\\Local\\Strata');
  const dev = { ...live, STRATA_STATE_DIR: 'C:\\Users\\me\\AppData\\Local\\StrataDev\\state' };
  assert.equal(stateRoot(dev, 'win32'), dev.STRATA_STATE_DIR);
  assert.equal(tuningDir(dev, 'win32'), 'C:\\Users\\me\\AppData\\Local\\StrataDev\\state\\tuning');
  const desktop = join(tmpdir(), 'StrataDev', 'desktop');
  assert.equal(shellPrefsFile({ env: { STRATA_DESKTOP_DATA_DIR: desktop } }), join(desktop, 'shell-prefs.json'));
  assert.ok(!shellPrefsFile({ env: { STRATA_COMPONENTS_DIR: join(tmpdir(), 'c', 'components') } }).includes('StrataDev'));
});
