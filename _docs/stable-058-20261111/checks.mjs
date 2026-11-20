import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import { mkdtempSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { harnessFixtureEnvironment } from '../../app/host/tests/helpers/harness-environment.mjs';
const root = resolve(import.meta.dirname, '../../apps/host');
const mode = process.argv[2];
const groups = {
  harness: ['--test', '--test-concurrency=2', 'tests/agent-context.test.mjs', 'tests/agent-tools.test.mjs', 'tests/agent-turn-guard.test.mjs', 'tests/harness-environment.test.mjs'],
  full: ['scripts/test.mjs'],
  typecheck: ['node_modules/typescript/bin/tsc', '--noEmit'],
  focused: ['--test', '--test-concurrency=2', ...process.argv.slice(3)],
};
if (!groups[mode]) throw new Error('Choose harness, full or typecheck');
const env = mode === 'focused' ? harnessFixtureEnvironment(mkdtempSync(resolve(tmpdir(), 'strata-058-focused-'))) : { ...process.env };
if (mode === 'focused') for (const name of ['HOME','USERPROFILE','APPDATA','LOCALAPPDATA','XDG_DATA_HOME','XDG_STATE_HOME','XDG_CONFIG_HOME']) mkdirSync(env[name], { recursive: true });
if (process.platform === 'win32') env.STRATA_TEST_WINDOWS_PROFILE = process.env.USERPROFILE;
if (mode === 'harness') env.STRATA_HARNESS_DIR = resolve(root, '../desktop/src-tauri/resources/harness');
else delete env.STRATA_HARNESS_DIR;
const child = spawn(process.execPath, groups[mode], { cwd: root, env, stdio: 'inherit', windowsHide: true });
child.once('error', error => { console.error(error); process.exitCode = 1; });
child.once('exit', code => { process.exitCode = code ?? 1; });
