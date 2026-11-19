import {test} from 'node:test';
import assert from 'node:assert/strict';
import {buildMismatch} from '../src/lib/build-identity.ts';

test('renderer accepts only its own Host and, when present, shell version', () => {
  const version = '0.4.0-preview.20261101.1';
  assert.equal(buildMismatch(version, {host: version, shell: version}), null);
  assert.equal(buildMismatch(version, {host: version, shell: null}), null);
  assert.match(buildMismatch(version, {host: '0.3.3', shell: version}), /app service/);
  assert.match(buildMismatch(version, {host: version, shell: '0.3.2'}), /desktop/);
  for (const identity of [null, {}, [], {host: version}, {host: null, shell: null}])
    assert.equal(typeof buildMismatch(version, identity), 'string');
});
