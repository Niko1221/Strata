import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { findFreePort } from '../../server/ports.mjs';
import { ROOMY_HOST } from './hardware.mjs';

const root = new URL('../..', import.meta.url);

/** Start the real HTTP host and wait for its ready message. */
export async function startRouteHost(env, args = []) {
  const port = await findFreePort(0);
  const child = spawn(process.execPath, ['server/index.mjs', ...args], {
    cwd: root, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
    env: {
      ...env, STRATA_PORT: String(port), STRATA_BIND: '127.0.0.1',
      STRATA_TEST_HARDWARE: ROOMY_HOST,
    },
  });
  let log = '';
  const capture = chunk => { log = (log + chunk).slice(-4000); };
  child.stdout.on('data', capture);
  child.stderr.on('data', capture);
  const exited = new Promise(resolve => child.once('close', (code, signal) => resolve(signal ?? code)));
  const alive = () => child.exitCode === null && child.signalCode === null;
  const stop = async () => {
    if (alive() && child.pid) child.kill('SIGKILL');
    await exited;
  };
  const wait = new AbortController();
  const timeout = setTimeout(() => wait.abort(new Error('no ready message within 30 s')), 30_000);
  try {
    const [ready] = await Promise.race([
      once(child, 'message', { signal: wait.signal }),
      exited.then(code => { throw new Error(`host exited (${code})`); }),
    ]);
    if (ready?.type !== 'strata-ready' || ready.port !== port) throw new Error('unexpected ready message');
    return { base: `http://127.0.0.1:${port}`, stop, alive };
  } catch (error) {
    await stop();
    throw new Error(`Host startup failed: ${error.cause?.message ?? error.message}\n${log}`, { cause: error });
  } finally {
    clearTimeout(timeout);
    wait.abort();
  }
}
