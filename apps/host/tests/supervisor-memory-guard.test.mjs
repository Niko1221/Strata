import {verifiedLocalRuntime} from './helpers/verified-local-runtime.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import cp from 'node:child_process';
import { syncBuiltinESMExports } from 'node:module';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
const GiB = 1024 ** 3;

test('Windows supervisor admission and pressure during both load and generation preserve ownership and errors', { skip: process.platform !== 'win32' }, async t => {
  const dir = mkdtempSync(join(os.tmpdir(), 'strata-physical-guard-'));
  const expert=verifiedLocalRuntime(t, dir);
  const before = { home: process.env.STRATA_HOME, binary: process.env.STRATA_LLAMA_SERVER };
  process.env.STRATA_HOME = dir;
  let free = 9 * GiB, pressureOnSpawn = false, proc, kills = 0;
  t.mock.method(os, 'freemem', () => free);
  t.mock.method(cp, 'spawn', () => {
    proc = Object.assign(new EventEmitter(), { pid: 424242, exitCode: null, signalCode: null, stdout: new PassThrough(), stderr: new PassThrough() });
    proc.kill = signal => { kills++; proc.signalCode = signal; proc.emit('exit', null, signal); return true; };
    if (pressureOnSpawn) free = 4.9 * GiB;
    return proc;
  });
  syncBuiltinESMExports();
  t.mock.method(globalThis, 'fetch', async () => Response.json({ status: 'ok' }));
  const supervisor = await import('../server/supervisor.mjs');
  t.after(async () => {
    await supervisor.stop(); t.mock.timers.reset(); t.mock.restoreAll(); syncBuiltinESMExports();
    for (const [key, name] of [['home', 'STRATA_HOME'], ['binary', 'STRATA_LLAMA_SERVER']]) { if (before[key] === undefined) delete process.env[name]; else process.env[name] = before[key]; }
    rmSync(dir, { recursive: true, force: true });
  });
  const modelPath = join(dir, 'fixture.gguf'); writeFileSync(modelPath, 'fixture');
  const args = { modelId: 'fixture', modelPath, config: { beginner: { contextLength: '2k' }, expert }, hardware: { cpu: { cores: 4 } } };
  let stoppedOther = false;
  free = 4 * GiB;
  await assert.rejects(supervisor.start({ ...args, beforeStart: () => { stoppedOther = true; } }), /RAM protection/);
  assert.equal(stoppedOther, false); assert.equal(kills, 0);
  free = 9 * GiB;
  await supervisor.start(args);
  t.mock.timers.enable({ apis: ['setInterval'] });
  // Restart after enabling the fake interval so the owned running timer is driven.
  await supervisor.stop(); free = 9 * GiB; kills = 0;
  await supervisor.start(args);
  free = 4.9 * GiB; t.mock.timers.tick(250); await new Promise(setImmediate);
  assert.equal(kills, 1); assert.equal(supervisor.status().pid, null);
  assert.equal(supervisor.status().status, 'error'); assert.match(supervisor.status().error, /RAM protection/);
  assert.ok(supervisor.status().memoryGuard.pressure.availableBytes > 3 * GiB);
  t.mock.timers.reset();
  free = 9 * GiB; pressureOnSpawn = true;
  await assert.rejects(supervisor.start(args), /RAM protection/);
  assert.equal(supervisor.status().running, false, 'health cannot promote a pressure-stopped child');
  assert.match(supervisor.status().error, /RAM protection/);
});
