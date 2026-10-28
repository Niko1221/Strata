import test from 'node:test';
import assert from 'node:assert/strict';
import { addEnergy, reportedPower } from '../server/energy-meter.mjs';

test('an unmetered iGPU does not prevent energy accumulation from a metered GPU', () => {
  const readings = [
    { id: 'GPU-discrete', powerW: 72, powerLimitW: 575 },
    { id: 'GPU-integrated', powerW: null, powerLimitW: null },
  ];
  assert.equal(reportedPower(readings), 72);
  assert.equal(reportedPower(readings, 'powerLimitW'), 575);
  const start = addEnergy(reportedPower(readings), 1_000);
  assert.equal(addEnergy(reportedPower(readings), 6_000) - start, 0.1);
});

test('power remains unavailable when no GPU reports it, and multiple readings are summed', () => {
  assert.equal(reportedPower([]), null);
  assert.equal(reportedPower([{ powerW: null }, { powerW: NaN }]), null);
  assert.equal(reportedPower([{ powerW: 20 }, { powerW: 30 }]), 50);
});
