import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { HarnessProcess, agentEnvironment } from '../server/agent/harness-runtime.mjs';
import { harnessFixtureEnvironment } from './helpers/harness-environment.mjs';

test('the spawned Harness uses the explicit fixture home and filtered environment, with only named credentials', async t => {
  const tempRoot = resolve(tmpdir());
  const parentProfile = process.env.USERPROFILE;
  const directory = mkdtempSync(join(tempRoot, 'strata-harness-env-'));
  const home = join(directory, 'home');
  const output = join(directory, 'environment.json');
  const bin = join(directory, 'probe.mjs');
  const baseEnv = harnessFixtureEnvironment(directory, { ...process.env, AMBIENT_TEST_TOKEN: 'synthetic-ambient' });
  baseEnv.DSH_AGENTS_HOME = join(directory, 'shared-agents');
  writeFileSync(bin, `import {writeFileSync} from 'node:fs'; import {homedir} from 'node:os';
writeFileSync(${JSON.stringify(output)},JSON.stringify({home:homedir(),profile:process.env.USERPROFILE,harness:process.env.DSH_HOME,agents:process.env.DSH_AGENTS_HOME,ambient:process.env.AMBIENT_TEST_TOKEN??null,explicit:process.env.EXPLICIT_TEST_TOKEN}));
console.log('http://127.0.0.1:12345/?token=synthetic_fixture_token'); setInterval(()=>{},1000);`);
  const runtime = new HarnessProcess({ harness: { bin }, home, patchFile: join(directory, 'unused.patch.yml'),
    cwd: join(directory, 'run'), baseEnv, env: { EXPLICIT_TEST_TOKEN: 'synthetic-explicit' } });
  t.after(async () => {
    await runtime.stop();
    assert.equal(dirname(resolve(directory)), tempRoot, 'cleanup stays inside this fixture temp root');
    rmSync(directory, { recursive: true, force: true, maxRetries: 3 });
  });
  await runtime.start({ timeoutMs: 10_000 });
  assert.deepEqual(JSON.parse(readFileSync(output, 'utf8')), {
    home: directory, profile: directory, harness: home, agents: join(directory, 'shared-agents'), ambient: null, explicit: 'synthetic-explicit',
  });
  assert.equal(process.env.USERPROFILE, parentProfile, 'the parent profile is unchanged');
});

test('only the explicit agents-home key crosses the base Harness filter', () => {
  const env = agentEnvironment({DSH_AGENTS_HOME: ' relative agents ', DSH_TOOLS_MODE: 'ptc',
    DSH_HOME: 'ambient-home', dsh_agents_home: 'wrong-case', AMBIENT_TEST_TOKEN: 'synthetic-only'});
  assert.equal(env.DSH_AGENTS_HOME, ' relative agents ');
  for (const key of ['DSH_TOOLS_MODE', 'DSH_HOME', 'dsh_agents_home', 'AMBIENT_TEST_TOKEN']) assert.equal(Object.hasOwn(env, key), false);
  for (const value of [undefined, '', ' \t ']) assert.equal(Object.hasOwn(agentEnvironment({DSH_AGENTS_HOME: value}), 'DSH_AGENTS_HOME'), false);
  assert.equal(agentEnvironment({DSH_AGENTS_HOME: 'base'}, {DSH_AGENTS_HOME: 'trusted-extra'}).DSH_AGENTS_HOME, 'trusted-extra');
});
