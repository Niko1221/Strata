import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { createKeepAwake, keepAwakeCommand } from '../server/keep-awake.mjs';

function fakeChild(record) {
  const child = new EventEmitter();
  child.stdin = new EventEmitter();
  child.stdin.end = () => { record.stdinEnded = true; };
  child.kill = () => { record.killed = true; child.emit('exit', 0); };
  return child;
}

function fakeSpawn(record) {
  return (command, args, options) => {
    record.calls.push({ command, args, options });
    const child = fakeChild(record);
    record.children.push(child);
    return child;
  };
}

test('the three platforms use the documented wake-lock commands', () => {
  const win = keepAwakeCommand({ platform: 'win32' });
  assert.equal(win.command, 'powershell.exe');
  assert.match(win.args.join(' '), /SetThreadExecutionState/);
  assert.match(win.args.join(' '), /2147483649/);
  assert.equal(win.stdin, true);

  const mac = keepAwakeCommand({ platform: 'darwin', parentPid: 4242 });
  assert.equal(mac.command, 'caffeinate');
  assert.deepEqual(mac.args, ['-i', '-w', '4242']);
  assert.equal(mac.stdin, false);

  const linux = keepAwakeCommand({ platform: 'linux' });
  assert.equal(linux.command, 'systemd-inhibit');
  assert.ok(linux.args.includes('--what=idle:sleep'));
  assert.equal(linux.args.at(-1), 'cat');
  assert.equal(linux.stdin, true);

  assert.equal(keepAwakeCommand({ platform: 'freebsd' }), null);
});

test('one child is started for any number of holds and stopped after the last release', () => {
  const record = { calls: [], children: [], killed: false, stdinEnded: false };
  const lines = [];
  const awake = createKeepAwake({ platform: 'linux', enabled: true, env: {}, spawnProcess: fakeSpawn(record), log: line => lines.push(line), check: () => null });

  const first = awake.hold('a model turn');
  const second = awake.hold('a download');
  assert.equal(record.calls.length, 1, 'both holds share one child');
  assert.equal(record.calls[0].command, 'systemd-inhibit');
  assert.equal(record.calls[0].options.stdio[0], 'pipe');
  assert.match(lines.join('\n'), /stay awake/);

  first();
  assert.equal(record.killed, false, 'the download still needs the wake lock');
  second();
  assert.equal(record.killed, true);
  assert.equal(record.stdinEnded, true, 'closing stdin lets `cat` end even on a killed host');
  assert.deepEqual(awake.state(), { enabled: true, active: false, reason: null, platform: 'linux', disabledByEnv: false, underTest: false });
  awake.close();
});

test('a check function can hold and release the wake lock without an explicit hold', () => {
  const record = { calls: [], children: [], killed: false, stdinEnded: false };
  let busy = 'a Time-skip job';
  const awake = createKeepAwake({ platform: 'darwin', enabled: true, env: {}, spawnProcess: fakeSpawn(record), check: () => busy });
  assert.equal(awake.syncNow(), true);
  assert.equal(record.calls.length, 1);
  busy = null;
  assert.equal(awake.syncNow(), false);
  assert.equal(record.killed, true);
  awake.close();
});

test('a refused or missing helper is one log line and never an exception', () => {
  const lines = [];
  const throwing = createKeepAwake({
    platform: 'win32', enabled: true, env: {},
    spawnProcess: () => { throw new Error('access denied'); },
    log: line => lines.push(line),
    check: () => 'a model turn',
  });
  assert.doesNotThrow(() => throwing.syncNow());
  assert.equal(throwing.state().active, false);
  assert.equal(throwing.state().enabled, true);
  assert.equal(lines.length, 1, 'one line, not one every five seconds');
  assert.match(lines[0], /could not start/);
  assert.match(lines[0], /access denied/);
  throwing.close();
});

test('a child that cannot start after spawn is dropped and logged once', () => {
  const record = { calls: [], children: [], killed: false, stdinEnded: false };
  const lines = [];
  const awake = createKeepAwake({
    platform: 'win32', enabled: true, env: {},
    spawnProcess: fakeSpawn(record),
    log: line => lines.push(line),
    check: () => 'a model turn',
  });
  assert.equal(awake.syncNow(), true);
  record.children[0].emit('error', new Error('refused by policy'));
  return new Promise(resolve => setImmediate(() => {
    assert.equal(awake.state().active, false);
    assert.match(lines.join('\n'), /could not start .*refused by policy/);
    assert.equal(lines.filter(line => /could not start/.test(line)).length, 1);
    awake.close();
    resolve();
  }));
});

test('the wake lock is off when the environment asks for it or when node:test starts the host', () => {
  const record = { calls: [], children: [], killed: false, stdinEnded: false };
  const off = createKeepAwake({ platform: 'linux', env: { STRATA_NO_KEEP_AWAKE: '1' }, spawnProcess: fakeSpawn(record), check: () => 'work' });
  assert.equal(off.state().enabled, false);
  assert.equal(off.syncNow(), false);
  assert.equal(record.calls.length, 0);
  assert.equal(off.hold('work') instanceof Function, true);
  off.close();

  const tests = createKeepAwake({ platform: 'linux', env: { NODE_TEST_CONTEXT: 'child-v8' }, spawnProcess: fakeSpawn(record), check: () => 'work' });
  assert.equal(tests.state().enabled, false);
  assert.equal(tests.state().underTest, true);
  tests.close();
});
