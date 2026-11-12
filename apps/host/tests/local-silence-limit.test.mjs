import test from 'node:test';
import assert from 'node:assert/strict';
import { localSilenceLimit } from '../server/agent/agent-bridge.mjs';

test('local silence allowance is opt-in and bounded', () => {
  for (const value of [undefined, null, '']) assert.equal(localSilenceLimit(value, 240_000), 240_000);
  assert.equal(localSilenceLimit('1800000', 240_000), 1_800_000);
  for (const value of ['0', '-1', 'Infinity', 'NaN', '1.5', '86400001', '10ms', ' 1000 ']) {
    assert.throws(() => localSilenceLimit(value, 240_000), /STRATA_LOCAL_STALL_MS/);
  }
});
