import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { endTree, endTreeThenKill, groupAlive, ownGroup } from '../server/process-tree.mjs';
import { runCommand } from '../server/agent/custom-command-run.mjs';

const isWin = process.platform === 'win32';
const alive = pid => { try { process.kill(pid, 0); return true; } catch (error) { return error.code === 'EPERM'; } };
const gone = async (pid, ms = 6000) => { for (let waited = 0; waited < ms && alive(pid); waited += 100) await new Promise(r => setTimeout(r, 100)); return !alive(pid); };

// A program that starts a program of its own and reports its pid, the way Stop meets a job and its revision command (#4529 #4570).
const parentScript = `const { spawn } = require('node:child_process'); const k = spawn(process.execPath, ['-e', 'setTimeout(()=>{}, 120000)'], { stdio: 'ignore' }); console.log(k.pid); setTimeout(()=>{}, 120000);`;

async function tree() {
  const child = spawn(process.execPath, ['-e', parentScript], { detached: ownGroup(), stdio: ['ignore', 'pipe', 'ignore'], windowsHide: true });
  const kid = await new Promise(resolve => child.stdout.once('data', b => resolve(Number(String(b).trim()))));
  return { child, kid };
}

test('endTree ends the program and the program it started, and nothing else (#4529 #4570)', async () => {
  const outsider = spawn(process.execPath, ['-e', 'setTimeout(()=>{}, 120000)'], { stdio: 'ignore', windowsHide: true });   // same program, not started by the job
  const { child, kid } = await tree();
  try {
    assert.ok(alive(kid));
    endTree(child);
    assert.ok(await gone(child.pid), 'the job survived');
    assert.ok(await gone(kid), 'what the job started survived');
    assert.ok(alive(outsider.pid), 'a process of the same program that the job did not start was ended');
  } finally { outsider.kill(); try { process.kill(kid, 'SIGKILL'); } catch { /* gone */ } child.kill('SIGKILL'); }
});

test('endTreeThenKill follows a TERM that is ignored with a KILL of the group', { skip: isWin && 'POSIX signals' }, async () => {
  const stubborn = `const { spawn } = require('node:child_process'); process.on('SIGTERM', () => {}); const k = spawn(process.execPath, ['-e', "process.on('SIGTERM',()=>{});setTimeout(()=>{},120000)"], { stdio: 'ignore' }); console.log(k.pid); setTimeout(()=>{}, 120000);`;
  const child = spawn(process.execPath, ['-e', stubborn], { detached: true, stdio: ['ignore', 'pipe', 'ignore'] });
  const kid = await new Promise(resolve => child.stdout.once('data', b => resolve(Number(String(b).trim()))));
  try {
    await new Promise(r => setTimeout(r, 300));
    endTreeThenKill(child, 500);
    assert.ok(await gone(kid, 8000), 'the group survived SIGTERM and was not killed');
    assert.equal(groupAlive(child), false);
  } finally { try { process.kill(-child.pid, 'SIGKILL'); } catch { /* gone */ } }
});

test('a child that leads no group of its own is still ended, alone, as before', async () => {
  const child = spawn(process.execPath, ['-e', 'setTimeout(()=>{}, 120000)'], { stdio: 'ignore', windowsHide: true });
  endTree(child);
  assert.ok(await gone(child.pid));
});

test('a custom command that runs out of time takes the programs it started with it', { skip: isWin && 'POSIX shell' }, async () => {
  const marker = `/tmp/strata-cc-${process.pid}.pid`;
  const result = await runCommand({ command: `sleep 120 & echo $! > ${marker}; wait`, timeoutSec: 1 });
  assert.equal(result.ok, false);
  const { readFileSync, rmSync } = await import('node:fs');
  const pid = Number(readFileSync(marker, 'utf8').trim());
  rmSync(marker, { force: true });
  assert.ok(await gone(pid), 'the command\'s own child outlived the time limit');
});
