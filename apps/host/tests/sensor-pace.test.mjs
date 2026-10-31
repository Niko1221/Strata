import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { sensorPace, ACTIVE_MS, IDLE_MS, WATCH_MS } from '../server/sensor-pace.mjs';
import { readGpuTelemetry } from '../server/device-probes.mjs';

test('an idle app with the Monitor out of sight asks the card rarely (#4066)', () => {
  const idle = sensorPace({ live: false, watchedUntil: 0, now: 1_000 });
  assert.equal(idle.intervalMs, IDLE_MS);
  assert.equal(idle.transfer, false);
  assert.ok(IDLE_MS >= 10 * ACTIVE_MS);
});

test('a running model or an open Monitor keeps the 3 s pace', () => {
  assert.equal(sensorPace({ live: true, now: 1_000 }).intervalMs, ACTIVE_MS);
  assert.equal(sensorPace({ live: true, now: 1_000 }).transfer, true);
  const watched = sensorPace({ live: false, watchedUntil: 1_000 + WATCH_MS, now: 1_000 });
  assert.equal(watched.intervalMs, ACTIVE_MS);
  assert.equal(watched.transfer, false, 'the PCIe sample only means something while a model runs');
  assert.equal(sensorPace({ live: false, watchedUntil: 1_000, now: 1_000 + 1 }).intervalMs, IDLE_MS);
});

const gpus = [{ id: 'GPU-1', vendor: 'nvidia', index: 0, sensorIndex: 0 }];
const calls = [];
const run = async (cmd, args) => {
  calls.push(args[0]);
  return args[0] === 'dmon' ? '# gpu rxpci txpci\n# Idx MB/s MB/s\n 0 12 3' : 'GPU-1, 41, 5, 1024, 30.5, 200, 4, 4, 16';
};

test('without the PCIe sample one nvidia-smi process is started, with it two', async () => {
  calls.length = 0;
  const idle = await readGpuTelemetry(gpus, { transfer: false, run });
  assert.deepEqual(calls.map(c => c === 'dmon' ? 'dmon' : 'query'), ['query']);
  assert.equal(idle[0].temperatureC, 41);
  calls.length = 0;
  await readGpuTelemetry(gpus, { run });
  assert.deepEqual(calls.map(c => c === 'dmon' ? 'dmon' : 'query'), ['query', 'dmon']);
});

test('the monitor uses the pace, the Monitor screen asks while it is open, and /api/monitor marks it watched', async () => {
  const monitor = await readFile(new URL('../server/monitor.mjs', import.meta.url), 'utf8');
  assert.match(monitor, /sensorPace\(/);
  assert.doesNotMatch(monitor, /sensorsAt>=3000/);
  const index = await readFile(new URL('../server/index.mjs', import.meta.url), 'utf8');
  assert.match(index, /monitor\.watch\(\); return json\(res, 200, monitor\.snapshot\(\)\)/);
  assert.match(index, /monitor\.setLiveProbe\(/);
  const screen = await readFile(new URL('../src/screens/Monitor.tsx', import.meta.url), 'utf8');
  assert.match(screen, /api\.get\('\/api\/monitor'\)/);
});

test('the drive-temperature reader starts only while the Monitor is watched and ends after it is gone (#4661)', async () => {
  const monitor = await readFile(new URL('../server/monitor.mjs', import.meta.url), 'utf8');
  assert.match(monitor, /STRATA_REAL_ENGINE==='1'&&Date\.now\(\)<watchedUntil\?readTemperatures\(\)/);
  assert.match(monitor, /stopTemperatures\(\)/);
  const { readTemperatures, stopTemperatures } = await import('../server/temperature-telemetry.mjs');
  assert.equal(typeof stopTemperatures, 'function');
  // nothing is started by importing or by stopping: on a PC that never opens the Monitor no PowerShell and no disk query exists
  stopTemperatures();
  assert.deepEqual(await readTemperatures({ platform: 'darwin' }), { cpuTempC: null, ssdTempC: null });
});
