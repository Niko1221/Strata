/** Run the current source regression group against the managed pinned Harness. */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const [rootArg, workArg, action] = process.argv.slice(2);
const root = resolve(rootArg), work = resolve(workArg);
assert.ok(['inspect', 'run'].includes(action));
assert.equal(process.version, 'v24.22.3');
const mod = name => import(pathToFileURL(join(root, name)).href);
const { sha256 } = await mod('apps/host/scripts/release-integrity.mjs');
const { HARNESS_BUDGET_PATCH: patch } = await mod('apps/desktop/scripts/harness-budget.mjs');
const { missingHarnessImports } = await mod('apps/desktop/scripts/harness-prune.mjs');
const harness = join(root, 'apps/desktop/src-tauri/resources/harness');
const bundle = JSON.parse(readFileSync(join(harness, 'BUNDLE.json'), 'utf8'));
assert.equal(bundle.commit, '571902028a75c64c1d767f4ae5fd099485f0ddf0');
const applied = bundle.patches?.find(row => row.id === patch.id);
assert.equal(applied?.beforeSha256, patch.beforeSha256);
assert.equal(applied?.afterSha256, sha256(join(harness, patch.file)));
assert.deepEqual(missingHarnessImports(harness), []);
if (action === 'inspect') {
  process.stdout.write(JSON.stringify({ runtime: process.version, commit: bundle.commit,
    bundleSHA256: sha256(join(harness, 'BUNDLE.json')), patch: applied, importClosure: 'complete' }));
} else {
  const { isolatedTestEnvironment } = await mod('apps/host/scripts/test-environment.mjs');
  const scratch = join(work, 'scratch');
  mkdirSync(scratch);
  for (const name of ['home', 'tmp', 'Local', 'Roaming', 'components', 'xdg-data', 'xdg-state', 'xdg-config'])
    mkdirSync(join(scratch, name));
  const env = isolatedTestEnvironment(process.env, {
    STRATA_HARNESS_DIR: harness, STRATA_NO_ACQUIRE: '1', STRATA_TEST_SCRATCH: scratch,
    TEMP: join(scratch, 'tmp'), TMP: join(scratch, 'tmp'), TMPDIR: join(scratch, 'tmp'),
    HOME: join(scratch, 'home'), USERPROFILE: join(scratch, 'home'), STRATA_HOME: join(scratch, 'home'),
    LOCALAPPDATA: join(scratch, 'Local'), APPDATA: join(scratch, 'Roaming'),
    STRATA_COMPONENTS_DIR: join(scratch, 'components'), XDG_DATA_HOME: join(scratch, 'xdg-data'),
    XDG_STATE_HOME: join(scratch, 'xdg-state'), XDG_CONFIG_HOME: join(scratch, 'xdg-config'),
  });
  const contract = JSON.parse(readFileSync(join(work, 'harness-contract.json'), 'utf8'));
  const command = [process.execPath, '--import',
    pathToFileURL(join(root, 'apps/host/scripts/test-fs-guard.mjs')).href,
    '--test', '--test-reporter=tap', '--test-concurrency=1', '--test-timeout=600000', ...contract.files];
  writeFileSync(join(work, 'harness-command.json'), JSON.stringify(command, null, 2), { flag: 'wx' });
  const child = spawnSync(command[0], command.slice(1), { cwd: join(root, 'apps/host'), env,
    windowsHide: true, encoding: 'utf8', timeout: 5400000, maxBuffer: 8 * 1024 ** 2 });
  writeFileSync(join(work, 'harness.tap'), child.stdout ?? '', { flag: 'wx' });
  writeFileSync(join(work, 'harness-stderr.log'), child.stderr ?? '', { flag: 'wx' });
  if (child.error || child.signal || child.status !== 0) throw new Error('Harness failed; retained TAP and stderr');
}
