import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { luidName, parseDxAdapters, windowsOtherGpus, parseGpuCounterLine, createWindowsGpuReader } from '../server/windows-gpu.mjs';

const dxRaw = JSON.stringify([
  { Description: 'AMD Radeon RX 9070 XT', VendorId: '4098', Luid: '90370', Dedicated: '17094934528', Software: '0' },
  { Description: 'Microsoft Basic Render Driver', VendorId: '5140', Luid: '1', Dedicated: '0', Software: '1' },
]);

test('the LUID is spelled as the performance counters spell it', () => {
  assert.equal(luidName('90370'), 'luid_0x00000000_0x00016102');
  assert.equal(luidName(String(0x12n * 2n ** 32n + 5n)), 'luid_0x00000012_0x00000005');
  assert.equal(luidName('nope'), null);
});

test('an AMD card on Windows gets its real memory size and a LUID, virtual monitors are not cards (#3727, #3836)', () => {
  const dx = parseDxAdapters(dxRaw);
  const cards = windowsOtherGpus([
    { Name: 'AMD Radeon RX 9070 XT', PNPDeviceID: 'PCI-A' },
    { Name: 'Meta Virtual Monitor', PNPDeviceID: 'ROOT-1' },
    { Name: 'Virtual Desktop Monitor', PNPDeviceID: 'ROOT-2' },
  ], dx);
  assert.equal(cards.length, 1);
  assert.equal(cards[0].vendor, 'amd');
  assert.equal(cards[0].vramGb, 15.9);
  assert.equal(cards[0].luid, 'luid_0x00000000_0x00016102');
});

test('without the registry list an AMD card is still listed, with no size', () => {
  const cards = windowsOtherGpus([{ Name: 'AMD Radeon RX 7900 XT', PNPDeviceID: 'PCI-B' }], parseDxAdapters('garbage'));
  assert.deepEqual(cards.map(c => [c.vendor, c.vramGb, c.luid]), [['amd', 0, null]]);
});

test('counter lines are read, and the load and VRAM in use reach the card by its LUID', () => {
  assert.equal(parseGpuCounterLine('nonsense'), null);
  const line = parseGpuCounterLine('G luid_0x00000000_0x00016102 37 4294967296');
  assert.equal(line.utilizationPct, 37);
  assert.equal(line.memoryUsedGb, 4);
  const procs = [];
  const reader = createWindowsGpuReader({ spawnFn: () => { const p = new EventEmitter(); p.stdout = new EventEmitter(); p.kill = () => {}; procs.push(p); return p; }, now: () => 5 });
  assert.equal(reader.read('luid_0x00000000_0x00016102'), null, 'nothing before the first line');
  procs[0].stdout.emit('data', 'G luid_0x00000000_0x00016102 37 4294967296\nG luid_0x00000000_0x00000001 0 0\n');
  assert.equal(reader.read('LUID_0x00000000_0x00016102').utilizationPct, 37);
  assert.equal(reader.read('luid_0x00000000_0x00099999'), null);
  assert.equal(procs.length, 1, 'one process for every read');
});
