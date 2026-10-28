/**
 * Keep the computer awake while Strata is doing work that must not be suspended.
 *
 * A run of a model, a download, an install or a Time-skip job can take hours.
 * When the system sleeps, the engine stops receiving time and network and a
 * turn can look frozen. The host therefore asks the operating system for a
 * "do not idle-sleep" state while any such work is active, and releases it as
 * soon as all work ends or the host closes.
 *
 * Per platform:
 *   Windows  PowerShell owns one thread that calls
 *            SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) and
 *            blocks on stdin, so the state ends when the host closes its pipe.
 *   macOS    `caffeinate -i -w <host pid>`; it exits with this process.
 *   Linux    `systemd-inhibit ... cat`; `cat` ends on the closed stdin pipe.
 *
 * Every failure is a silent downgrade with exactly one log line: a machine
 * where the call is refused still runs Strata, it may only sleep.
 */
import { spawn } from 'node:child_process';

export const KEEP_AWAKE_DISABLED_ENV = 'STRATA_NO_KEEP_AWAKE';

const truthy = value => ['1', 'true', 'yes', 'on'].includes(String(value ?? '').trim().toLowerCase());

/** The command that holds the platform's wake lock. Exported so it can be tested without starting it. */
export function keepAwakeCommand({ platform = process.platform, parentPid = process.pid } = {}) {
  if (platform === 'win32') {
    const script = [
      "$ErrorActionPreference = 'SilentlyContinue'",
      "Add-Type -Namespace StrataPower -Name Api -MemberDefinition '[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint esFlags);'",
      'if ([StrataPower.Api]::SetThreadExecutionState([uint32]2147483649) -eq 0) { exit 2 }',
      'try { while ($null -ne [Console]::In.ReadLine()) { } } finally { [StrataPower.Api]::SetThreadExecutionState([uint32]2147483648) | Out-Null }',
    ].join('\n');
    return {
      command: 'powershell.exe',
      args: ['-NoProfile', '-NonInteractive', '-WindowStyle', 'Hidden', '-Command', script],
      stdin: true,
      label: 'SetThreadExecutionState',
    };
  }
  if (platform === 'darwin') {
    return { command: 'caffeinate', args: ['-i', '-w', String(parentPid)], stdin: false, label: 'caffeinate' };
  }
  if (platform === 'linux') {
    return {
      command: 'systemd-inhibit',
      args: ['--what=idle:sleep', '--who=Strata', '--why=Strata is working', '--mode=block', 'cat'],
      stdin: true,
      label: 'systemd-inhibit',
    };
  }
  return null;
}

export function createKeepAwake({
  platform = process.platform,
  env = process.env,
  spawnProcess = spawn,
  log = line => console.log(line),
  check = () => null,
  intervalMs = 5000,
  parentPid = process.pid,
  enabled = undefined,
} = {}) {
  const disabledByEnv = truthy(env[KEEP_AWAKE_DISABLED_ENV]);
  // The host is started by node:test for the UI tests; those children inherit
  // NODE_TEST_CONTEXT. They must not wake the machine that runs the tests.
  const underTest = !!env.NODE_TEST_CONTEXT;
  const on = enabled ?? (!disabledByEnv && !underTest);
  const holds = new Map();
  let child = null;
  let timer = null;
  let warned = false;
  let announced = false;
  let closed = false;
  let lastReason = null;

  const warn = message => {
    if (warned) return;
    warned = true;
    try { log(message); } catch { /* logging must never stop the host */ }
  };

  const reasonNow = () => {
    if (holds.size) return holds.values().next().value;
    try {
      const found = check();
      return typeof found === 'string' ? found : found ? 'Strata is working' : null;
    } catch {
      return null;
    }
  };

  function stopChild() {
    const running = child;
    child = null;
    if (!running) return;
    try { running.stdin?.end(); } catch { /* already closed */ }
    try { running.kill(); } catch { /* already gone */ }
  }

  function startChild(plan) {
    if (child) return;
    let started;
    try {
      started = spawnProcess(plan.command, plan.args, {
        windowsHide: true,
        stdio: [plan.stdin ? 'pipe' : 'ignore', 'ignore', 'ignore'],
      });
    } catch (error) {
      warn(`keep-awake · ${plan.label} could not start (${error?.message ?? error}); the computer may sleep while Strata works.`);
      return;
    }
    if (!started) {
      warn(`keep-awake · ${plan.label} did not start; the computer may sleep while Strata works.`);
      return;
    }
    child = started;
    // A wake lock must never be the reason the host cannot exit: the child
    // watches stdin (or the parent pid) and ends when this process is gone.
    try { started.unref?.(); started.stdin?.unref?.(); } catch { /* not every stream has unref */ }
    try { started.stdin?.on?.('error', () => { /* the child owns the pipe; its exit is handled below */ }); } catch { /* no stdin */ }
    started.once?.('error', error => {
      if (child === started) child = null;
      warn(`keep-awake · ${plan.label} could not start (${error?.message ?? error}); the computer may sleep while Strata works.`);
    });
    started.once?.('exit', code => {
      if (child === started) child = null;
      if (code !== 0 && code !== null) {
        warn(`keep-awake · ${plan.label} stopped (exit ${code}); the computer may sleep while Strata works.`);
      }
    });
    if (!announced) {
      announced = true;
      try { log('keep-awake · the computer will stay awake while Strata is working.'); } catch { /* best effort */ }
    }
  }

  function syncNow() {
    if (closed) return false;
    if (!on) {
      lastReason = null;
      stopChild();
      return false;
    }
    const reason = reasonNow();
    lastReason = reason;
    if (!reason) {
      stopChild();
      return false;
    }
    const plan = keepAwakeCommand({ platform, parentPid });
    if (!plan) {
      warn(`keep-awake · no wake lock is available on ${platform}; the computer may sleep while Strata works.`);
      return false;
    }
    startChild(plan);
    return true;
  }

  function hold(reason) {
    const token = Symbol(String(reason ?? 'work'));
    holds.set(token, String(reason ?? 'Strata is working'));
    syncNow();
    let released = false;
    return () => {
      if (released) return;
      released = true;
      holds.delete(token);
      syncNow();
    };
  }

  function start() {
    if (!on || timer || closed) {
      if (!on) syncNow();
      return;
    }
    timer = setInterval(syncNow, intervalMs);
    timer.unref?.();
    syncNow();
  }

  function close() {
    closed = true;
    if (timer) clearInterval(timer);
    timer = null;
    holds.clear();
    stopChild();
  }

  return {
    start,
    close,
    hold,
    syncNow,
    state: () => ({
      enabled: on,
      active: !!child,
      reason: child ? lastReason : null,
      platform,
      disabledByEnv,
      underTest,
    }),
  };
}
