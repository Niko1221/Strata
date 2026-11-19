/** A verified, lazy, host-owned RustXNG Full process. Never acquires or updates software. */
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import { findFreePort } from './ports.mjs';
import { boundedJson } from './web-search-rustxng.mjs';

export const RUSTXNG_VERSION = '0.31.1';
export const RUSTXNG_SHA256 = 'a509798f040722db4a3df9eebb3551437c69189ed0003bb2a0b62bf749cf6775';
const HERE = dirname(fileURLToPath(import.meta.url));
export function findRustxng() {
  return [resolve(HERE, '../../harness/rustxng/rustxng.exe'), resolve(HERE, '../../desktop/src-tauri/resources/harness/rustxng/rustxng.exe')].find(existsSync) ?? null;
}

export function createRustxngRuntime({ binary = findRustxng(), fetch: fetchImpl = globalThis.fetch, spawn: spawnImpl = spawn, freePort = findFreePort } = {}) {
  let child = null, endpoint = null, starting = null, closed = false;
  async function start() {
    if (closed) throw new Error('RustXNG has been closed.');
    if (endpoint && child?.exitCode === null) return endpoint;
    if (starting) return starting;
    starting = (async () => {
      if (!binary || createHash('sha256').update(readFileSync(binary)).digest('hex') !== RUSTXNG_SHA256) throw new Error('The bundled RustXNG executable is missing or has changed. Reinstall Strata.');
      const port = await freePort(29483);
      if (closed) throw new Error('RustXNG has been closed.');
      const env = Object.fromEntries(['SystemRoot', 'WINDIR', 'PATH', 'TEMP', 'TMP', 'USERPROFILE', 'LOCALAPPDATA', 'APPDATA'].filter(key => process.env[key]).map(key => [key, process.env[key]]));
      Object.assign(env, { RUST_LOG: 'error', RUSTXNG_IDLE_TIMEOUT: '600', SEARXNG_PORT: String(port) });
      const owned = spawnImpl(binary, ['--bind', '127.0.0.1', '--port', String(port), '--profile', 'full'], { cwd: dirname(binary), windowsHide: true, stdio: 'ignore', env });
      child = owned;
      let failed = false;
      owned.once('error', () => { failed = true; });
      owned.once('exit', () => { if (child === owned) { endpoint = null; child = null; } });
      const base = `http://127.0.0.1:${port}/`;
      try {
        const deadline = Date.now() + 25_000;
        while (Date.now() < deadline) {
          if (closed || failed || owned.exitCode !== null) throw new Error('RustXNG did not start.');
          try {
            const response = await fetchImpl(new URL('config', base), { redirect: 'error', signal: AbortSignal.timeout(1000) });
            if (response.ok) {
              const config = await boundedJson(response);
              if (config.version === `RustXNG ${RUSTXNG_VERSION}` && Array.isArray(config.engines) && config.brand?.GIT_URL === 'https://github.com/sahenjp/RustXNG') { endpoint = base; return base; }
              throw new Error('Unexpected RustXNG service identity.');
            }
            await response.body?.cancel();
          } catch (error) { if (error.message === 'Unexpected RustXNG service identity.') throw error; }
          await delay(200);
        }
        throw new Error('RustXNG did not become ready within 25 seconds.');
      } catch (error) { owned.kill(); if (child === owned) { child = null; endpoint = null; } throw error; }
    })();
    try { return await starting; } finally { starting = null; }
  }
  return {
    start,
    endpoint: () => endpoint,
    available: () => !!binary,
    async close() {
      closed = true;
      const owned = child;
      if (owned && owned.exitCode === null) {
        const exited = new Promise(resolve => owned.once('exit', resolve));
        owned.kill();
        await Promise.race([exited, delay(3000)]);
        if (owned.exitCode === null) owned.kill('SIGKILL');
      }
      if (starting) await starting.catch(() => {});
      endpoint = null;
    },
  };
}
