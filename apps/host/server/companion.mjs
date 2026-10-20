import { homedir } from 'node:os';
import { join } from 'node:path';
import { chmodSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from 'node:fs';

/** Both checkouts and installed apps put companion-bridge beside the host folder. */
export function companionBridgeURL(hostModuleURL = import.meta.url) {
  return new URL('../../companion-bridge/bridge.mjs', hostModuleURL);
}

const failure = (status, message) => Object.assign(new Error(message), { status });

/** Pairing stays off until the person enables it in the desktop app. */
export function createCompanionManager({
  hostPort,
  authorization,
  directory = join(homedir(), 'Library', 'Application Support', 'Strata Companion'),
  port = 8787,
  adminPort = 8788,
  listen = '0.0.0.0',
  loadBridge = () => import(companionBridgeURL()),
  bridgeOptions = {},
  changed = () => {},
} = {}) {
  if (!Number.isInteger(hostPort) || hostPort < 1 || hostPort > 65535) throw new Error('A local Strata host port is required.');
  let bridge = null;
  let devices = [];
  let error = null;
  const preferenceFile = join(directory, 'desktop-pairing.json');
  let preferenceExists = false;
  let restoreEnabled = false;
  let optedIn = false;
  try {
    const preference = JSON.parse(readFileSync(preferenceFile, 'utf8'));
    preferenceExists = true;
    if (!preference || Array.isArray(preference) || typeof preference.enabled !== 'boolean') throw new Error('Invalid pairing preference.');
    restoreEnabled = preference.enabled;
    // Earlier builds wrote this file only after enabling pairing. A later disable keeps that consent.
    optedIn = preference.enabled || preference.hasOptedIn !== false;
  } catch (cause) {
    if (cause.code !== 'ENOENT') {
      preferenceExists = true;
      error = 'The iPhone pairing setting could not be read. Enable pairing again to save it.';
    }
  }
  let pending = Promise.resolve();
  const state = () => {
    const current = bridge?.state();
    return { enabled: !!bridge, invitation: current?.invitation ?? null, devices: current?.devices ?? devices, error };
  };
  // An enable immediately followed by disable must close the resulting listener.
  const serialized = action => {
    const result = pending.then(action);
    pending = result.catch(() => {});
    return result;
  };
  const requireBridge = () => {
    if (!bridge) throw failure(409, 'Enable iPhone pairing first.');
    return bridge;
  };
  const remember = enabled => {
    if (!enabled && !preferenceExists && !restoreEnabled) return;
    mkdirSync(directory, { recursive: true, mode: 0o700 });
    const temp = `${preferenceFile}.${process.pid}.tmp`;
    try {
      writeFileSync(temp, JSON.stringify({ enabled, hasOptedIn: optedIn || enabled }), { mode: 0o600 });
      chmodSync(temp, 0o600);
      renameSync(temp, preferenceFile);
      preferenceExists = true; restoreEnabled = enabled; optedIn ||= enabled;
    } finally { try { unlinkSync(temp); } catch {} }
  };
  const closeBridge = async () => {
    if (bridge) {
      devices = bridge.state().devices;
      await bridge.close();
      bridge = null;
    }
    return state();
  };
  const startBridge = async () => {
    if (bridge) return state();
    error = null;
    try {
      const { createBridge, HostClient } = await loadBridge();
      const started = await createBridge({
        ...bridgeOptions,
        directory, port, adminPort, listen,
        host: new HostClient({ upstream: `http://127.0.0.1:${hostPort}`, authorization }),
      });
      try { remember(true); }
      catch (cause) { await started.close(); throw cause; }
      bridge = started;
      changed();
      return state();
    } catch (cause) {
      error = cause.code === 'EADDRINUSE'
        ? 'Pairing is already open in another app. Quit Strata Companion, then try again.'
        : 'iPhone pairing could not start. Try again, or reinstall this Strata build.';
      throw failure(503, error);
    }
  };
  return {
    state,
    hasOptedIn: () => optedIn,
    start: () => serialized(startBridge),
    restore: () => serialized(() => restoreEnabled ? startBridge() : state()),
    close: () => serialized(closeBridge),
    stop: () => serialized(async () => {
      let saved = false;
      try { remember(false); saved = true; }
      catch { restoreEnabled = false; }
      finally { await closeBridge(); }
      if (!saved) {
        error = 'Pairing stopped, but the setting could not be saved. Check this Mac’s app permissions.';
        throw failure(503, error);
      }
      error = null;
      return state();
    }),
    rotateInvitation: () => serialized(async () => {
      await requireBridge().rotateInvitation();
      return state();
    }),
    revokeSession: id => serialized(async () => {
      requireBridge().revokeSession(id);
      return state();
    }),
    qrImage: () => bridge?.qrImage() ?? null,
  };
}
