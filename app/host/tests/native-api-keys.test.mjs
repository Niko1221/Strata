import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync, rmSync} from 'node:fs';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {applyNativeApiKeys, nativeApiKeyList} from '../server/native-control.mjs';
import {KNOWN_KEYS, checkRunConfig} from '../server/run-config-check.mjs';

test('default load still replaces api_key with one fresh Host key', () => {
  const cfg = {api_key: 'operator-kept', model: 'iq2_xs'};
  const key = applyNativeApiKeys(cfg, {});
  assert.equal(typeof key, 'string');
  assert.equal(key.length, 64);
  assert.equal(cfg.api_key, key);
  assert.notEqual(key, 'operator-kept');
});

test('hakua_harness keeps principal keys and adds a distinct Host key', () => {
  const root = mkdtempSync(join(tmpdir(), 'strata-hakua-keys-'));
  try {
    const cfg = {
      api_key: ['operator-extra', 'alice-key'],
      hakua_harness: {
        enabled: true,
        users: [
          {principal: 'alice', key_env: 'STRATA_TEST_ALICE', root: join(root, 'alice')},
          {principal: 'bob', key_env: 'STRATA_TEST_BOB', root: join(root, 'bob')},
        ],
      },
    };
    const env = {STRATA_TEST_ALICE: 'alice-key', STRATA_TEST_BOB: 'bob-key'};
    const host = applyNativeApiKeys(cfg, env);
    assert.equal(host.length, 64);
    assert.ok(Array.isArray(cfg.api_key));
    assert.deepEqual(new Set(cfg.api_key), new Set(['alice-key', 'bob-key', 'operator-extra', host]));
    assert.ok(!['alice-key', 'bob-key', 'operator-extra'].includes(host));
    assert.equal(nativeApiKeyList(cfg.api_key).includes(host), true);
  } finally {
    rmSync(root, {recursive: true, force: true});
  }
});

test('hakua_harness refuses missing, non-ASCII or shared principal keys before load', () => {
  const root = mkdtempSync(join(tmpdir(), 'strata-hakua-keys-bad-'));
  try {
    const users = [
      {principal: 'alice', key_env: 'STRATA_TEST_ALICE', root: join(root, 'alice')},
      {principal: 'bob', key_env: 'STRATA_TEST_BOB', root: join(root, 'bob')},
    ];
    assert.throws(() => applyNativeApiKeys({hakua_harness: {enabled: true, users}}, {}),
      /distinct ASCII existing key/);
    assert.throws(() => applyNativeApiKeys({hakua_harness: {enabled: true, users}},
      {STRATA_TEST_ALICE: 'alice-key', STRATA_TEST_BOB: 'café'}), /distinct ASCII existing key/);
    assert.throws(() => applyNativeApiKeys({hakua_harness: {enabled: true, users}},
      {STRATA_TEST_ALICE: 'same', STRATA_TEST_BOB: 'same'}), /distinct ASCII existing key/);
    assert.throws(() => applyNativeApiKeys({hakua_harness: {enabled: true, users: 'nope'}}, {}),
      /enabled and users only/);
  } finally {
    rmSync(root, {recursive: true, force: true});
  }
});

test('run-config check knows hakua_harness so Host does not claim it is ignored', () => {
  assert.equal(KNOWN_KEYS.has('hakua_harness'), true);
  const checked = checkRunConfig(JSON.stringify({
    exe: 'component:strata-engine',
    args: ['--max-context', '32768'],
    hakua_harness: {enabled: false, users: []},
  }));
  assert.equal(checked.ok, true);
  assert.equal(checked.issues.some(i => i.key === 'hakua_harness'), false);
});
