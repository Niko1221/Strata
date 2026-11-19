/** The actual detector; all OS probes, CPU inventory, disk and memory are controlled. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { resolve } from 'node:path';
import { createCpuDetector } from '../server/cpu-topology.mjs';

const root = resolve(import.meta.dirname, '..');
const bundle = await build({
  entryPoints: [resolve(root, 'server/hardware.mjs')], bundle: true, write: false, format: 'esm', platform: 'node',
  define: { 'process.platform': '"win32"', 'process.arch': '"x64"' },
  plugins: [{ name: 'controlled-os', setup(b) {
    b.onResolve({ filter: /^node:os$/ }, () => ({ path: 'os', namespace: 'boundary' }));
    b.onResolve({ filter: /\/store\.mjs$|^\.\/store\.mjs$/ }, () => ({ path: 'store', namespace: 'boundary' }));
    b.onResolve({ filter: /\/util\.mjs$|^\.\/util\.mjs$/ }, () => ({ path: 'util', namespace: 'boundary' }));
    b.onResolve({ filter: /^\.\/catalog\.mjs$/ }, () => ({ path: 'catalog', namespace: 'boundary' }));
    b.onLoad({ filter: /.*/, namespace: 'boundary' }, ({ path }) => ({ contents: path === 'os'
      ? `export default {cpus:()=>globalThis.__cpu.inventory,totalmem:()=>32*1024**3,freemem:()=>16*1024**3,hostname:()=>"test",release:()=>"test"}`
      : path === 'store' ? `export const MODELS_DIR="fixture",diskFreeGb=()=>100,diskTotalGb=()=>200;`
      : path === 'catalog' ? `export const CONTEXT_TOKENS={recommended:32768},VARIANTS=[{id:'fixture'}],footprint=()=>({totalGb:1});`
      : `export const round=(n,d)=>Number(n.toFixed(d));export const probe=async(command,args)=>{if(!args.join(' ').includes('Win32_Processor'))throw Error('Unexpected device probe');return globalThis.__cpu.probe()};` }));
  } }],
});
let seq = 0;
const row = (id = 'CPU0', cores = 6, threads = 12) => ({ DeviceID: id, SocketDesignation: id, Name: 'CPU', NumberOfCores: cores, NumberOfLogicalProcessors: threads, MaxClockSpeed: 3000 });
async function detector(t, probe) {
  const previous = process.env.STRATA_TEST_HARDWARE;
  process.env.STRATA_TEST_HARDWARE = JSON.stringify({ totalGb: 32, freeGb: 16, gpus: [] });
  globalThis.__cpu = { inventory: Array.from({ length: 12 }, () => ({ model: 'CPU', speed: 3000 })), probe };
  t.after(() => { if (previous === undefined) delete process.env.STRATA_TEST_HARDWARE; else process.env.STRATA_TEST_HARDWARE = previous; delete globalThis.__cpu; });
  return import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text + `\n// fixture ${++seq}`).toString('base64')}`);
}
for (const raw of ['', '{bad', JSON.stringify([row(), { DeviceID: 'CPU1' }]), JSON.stringify(row('CPU0', 3, 6))])
  test(`cold incomplete topology stays unverified: ${raw}`, async t => {
    const d = await detector(t, async () => raw), hw = await d.detect({ force: true });
    assert.equal(hw.cpu.topologyKnown, false);
    assert.deepEqual(hw.processors, []);
    assert.equal(hw.cpu.topologySource, 'unavailable');
    globalThis.__cpu.probe = async () => JSON.stringify(row());
    const verified = await d.detect({ force: true });
    assert.equal(verified.cpu.cores, 6);
    assert.equal(verified.cpu.topologySource, 'live');
  });
test('a transient failure retains only compatible verified CPU shape and labels it cached', async t => {
  const d = await detector(t, async () => JSON.stringify(row()));
  const good = await d.detect({ force: true });
  globalThis.__cpu.probe = async () => '';
  d.invalidate();
  const failed = await d.detect({ force: true });
  assert.deepEqual(failed.processors, good.processors);
  assert.equal(failed.cpu.topologyKnown, true);
  assert.equal(failed.cpu.topologySource, 'cached');
  globalThis.__cpu.inventory[0].model = 'Different CPU';
  const changed = await d.detect({ force: true });
  assert.equal(changed.cpu.topologyKnown, false);
  assert.deepEqual(changed.processors, []);
});

test('verified topology updates and complete socket order is stable; duplicates are not cached as a new machine', async t => {
  let raw=JSON.stringify([row('CPU1',3,6),row('CPU0',3,6)]);
  const d=await detector(t,async()=>raw),first=await d.detect({force:true});
  assert.deepEqual(first.processors.map(p=>p.id),['CPU0','CPU1']);
  raw=JSON.stringify([row('CPU0',3,6),row('CPU1',3,6)]);
  assert.deepEqual((await d.detect({force:true})).processors,first.processors);
  raw=JSON.stringify([row('CPU0',3,6),row('CPU0',3,6)]);
  const failed=await d.detect({force:true});assert.equal(failed.cpu.topologySource,'cached');assert.deepEqual(failed.processors,first.processors);
  raw=JSON.stringify(row('CPU0',12,12));const changed=await d.detect({force:true});
  assert.equal(changed.cpu.cores,12);assert.equal(changed.cpu.topologySource,'live');assert.notDeepEqual(changed.processors,first.processors);
});
test('late failed and successful probes cannot overwrite a newer successful reading',async t=>{
  const pending=[];const d=await detector(t,()=>new Promise(resolve=>pending.push(resolve)));
  const older=d.detect({force:true}),newer=d.detect({force:true});
  pending[1](JSON.stringify(row()));const good=await newer;pending[0]('');
  assert.deepEqual((await older).cpu,good.cpu);assert.deepEqual(d.cachedHardware().cpu,good.cpu);
  const slow=d.detect({force:true}),fast=d.detect({force:true});pending[3](JSON.stringify(row('CPU0',12,12)));await fast;pending[2](JSON.stringify(row()));
  assert.equal((await slow).cpu.cores,12);assert.equal(d.cachedHardware().cpu.cores,12);
});
test('changed Node logical count invalidates previous shape even if an older probe later succeeds',async t=>{
  const d=await detector(t,async()=>JSON.stringify(row()));await d.detect({force:true});
  let finish;globalThis.__cpu.probe=()=>new Promise(resolve=>finish=resolve);const stale=d.detect({force:true});
  globalThis.__cpu.inventory=globalThis.__cpu.inventory.slice(0,6);globalThis.__cpu.probe=async()=>{throw Error('controlled timeout');};
  const cold=await d.detect({force:true});assert.equal(cold.cpu.topologyKnown,false);
  finish(JSON.stringify(row()));assert.equal((await stale).cpu.topologyKnown,false);assert.equal(d.cachedHardware().cpu.topologyKnown,false);
});
test('platform and architecture changes invalidate the process-local verified topology',async()=>{
  let arch='x64',raw=JSON.stringify(row());
  const cpu=createCpuDetector({cpus:()=>Array.from({length:12},()=>({model:'CPU',speed:3000})),architecture:()=>arch,run:async()=>raw});
  assert.equal((await cpu.detect('win32')).topologySource,'live');
  arch='arm64';raw='';assert.equal((await cpu.detect('win32')).topologySource,'unavailable');
  arch='x64';raw=JSON.stringify(row());assert.equal((await cpu.detect('win32')).topologySource,'live');
  raw='';assert.equal((await cpu.detect('darwin')).topologySource,'unavailable');
});
