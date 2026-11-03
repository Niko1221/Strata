/** Ports the host listens on: a free one for the helpers, and a plain sentence when the main one cannot be taken. */
import net from 'node:net';
import { spawnSync } from 'node:child_process';

/** The ranges in `netsh int ipv4 show excludedportrange protocol=tcp` output: every line with a start and an end port. */
export function parseExcludedPortRanges(text) {
  const ranges = [];
  for (const line of String(text ?? '').split(/\r?\n/)) {
    const m = /^\s*(\d{1,5})\s+(\d{1,5})(\s+\*)?\s*$/.exec(line);
    if (m && Number(m[1]) <= Number(m[2])) ranges.push({ start: Number(m[1]), end: Number(m[2]), admin: !!m[3] });
  }
  return ranges;
}

/**
 * Strata Dev only (index.mjs asks when STRATA_DEV=1): the range Windows keeps that holds `port`, or null (no such range, not
 * Windows, or netsh did not answer). Hyper-V, WSL and Docker reserve these; a port inside one cannot be listened on.
 */
export function windowsReservedRange(port, { platform = process.platform, run = spawnSync } = {}) {
  if (platform !== 'win32') return null;
  try {
    const r = run('netsh', ['int', 'ipv4', 'show', 'excludedportrange', 'protocol=tcp'], { encoding: 'utf8', windowsHide: true, timeout: 5000 });
    if (r.status !== 0) return null;
    return parseExcludedPortRanges(r.stdout).find(range => port >= range.start && port <= range.end) ?? null;
  } catch {
    return null;
  }
}

/** The Dev message for a port inside a reserved range: which range, and what to do. */
export function reservedPortMessage(port, range) {
  return `Port ${port} ist von Windows reserviert (Bereich ${range.start}-${range.end}, gehalten von Hyper-V, WSL oder Docker). Starte Strata Dev mit einem anderen Port (STRATA_PORT) oder gib den Bereich frei ("netsh int ipv4 show excludedportrange protocol=tcp" zeigt alle Bereiche).`;
}

/** Why a listen failed, in words a person can act on (Windows keeps whole ranges for Hyper-V, WSL and Docker). */
export function listenProblem(error, port, host = '127.0.0.1') {
  const code = error?.code;
  if (code === 'EADDRINUSE') return `Port ${port} is already in use. Strata may already be running; close it or the other program, or start Strata with another port (STRATA_PORT).`;
  if (code === 'EACCES') return `Windows or a firewall does not let Strata use port ${port}. Hyper-V, WSL and Docker keep ranges of ports for themselves (see "netsh int ipv4 show excludedportrange protocol=tcp"). Start Strata with another port (STRATA_PORT), for example 8801.`;
  if (code === 'EADDRNOTAVAIL') return `${host} is not an address of this computer, so Strata cannot listen on it. Check STRATA_BIND.`;
  return `Strata cannot listen on ${host}:${port} (${error?.message ?? code}).`;
}

function tryListen(port, host) {
  return new Promise(resolve => {
    const probe = net.createServer();
    probe.once('error', () => resolve(null));
    probe.listen(port, host, () => { const got = probe.address().port; probe.close(() => resolve(got)); });
  });
}

/** The engine's usual port (`fallback`), or STRATA_ENGINE_PORT when it names a usable one: Strata Dev sets 8081 (docs/DEV_BUILD.md). */
export function enginePort(env = process.env, fallback = 8080) {
  const port = Number(env.STRATA_ENGINE_PORT);
  return Number.isInteger(port) && port >= 1024 && port <= 65535 ? port : fallback;
}

/** `preferred` when it can be taken, else the next free port above it (up to `tries` of them), else any free port. Never throws. */
export async function findFreePort(preferred, host = '127.0.0.1', { skip = [], tries = 20 } = {}) {
  for (let offset = 0; offset < tries; offset++) {
    const port = preferred + offset;
    if (port > 65535 || skip.includes(port)) continue;
    const got = await tryListen(port, host);
    if (got) return got;
  }
  return (await tryListen(0, host)) ?? preferred;
}
