/** A separate, key-protected listener for inference only. Never serves the UI or host APIs. */
import http from 'node:http';
import os from 'node:os';
import { apiPath } from './api-paths.mjs';
import { listenProblem } from './ports.mjs';
/** Sharing without a key is allowed (owner decision, 0.2.2): the person is told once, in plain words, and the app keeps offering to make a key. */
export const NO_KEY_WARNING = 'No API key is set, so any device on this network can use your model. Create an API key to protect it.';
/**
 * Should the listener come up with Strata, without anyone opening a page first? The desktop shell sets STRATA_SHARE_ON_START=1
 * when it was started with `--lan` or STRATA_LAN=1 (apps/desktop/src-tauri/src/supervisor.rs); a person can set it the same way.
 * The words accepted are the shell's: 1, on, true, yes, lan. Nothing else is on by default: without the switch the model stays
 * on this computer.
 */
export function shareOnStartRequested(env = process.env) {
  return ['1', 'on', 'true', 'yes', 'lan'].includes(String(env.STRATA_SHARE_ON_START ?? '').trim().toLowerCase());
}
/** The one line the host prints when it shares on the network at start, so a terminal (and the app's log) shows the address and the key warning. */
export function shareOnStartLine(state) {
  const parts = [`[strata] sharing the model on your network at ${state.url}`];
  if (state.warning) parts.push(`[strata] ${state.warning}`);
  return parts.join('\n');
}

export function createLocalShare({ handle, hasKey, changed = () => {}, host = '0.0.0.0', port = 8792 }) {
  let listener = null, active = false;
  const address = () => Object.values(os.networkInterfaces()).flat().find(n => n?.family === 'IPv4' && !n.internal)?.address ?? '127.0.0.1';
  const state = () => ({ active, url: active ? `http://${host === '127.0.0.1' ? host : address()}:${listener.address().port}/v1` : null, warning: active && !hasKey() ? NO_KEY_WARNING : null });
  async function start() {
    if (active) return state();
    if (listener) throw new Error('Sharing is already starting.');
    listener = http.createServer(async (req, res) => {
      const path = apiPath(new URL(req.url ?? '/', 'http://localhost').pathname);
      const method = req.method ?? 'GET';
      // #3208: with an API key set, a browser page of any site can call the shared API - the key decides, not the
      // origin. The browser's preflight (OPTIONS) never carries the key (the spec says so), so it is answered here,
      // before the path check and the key check; the real request that follows still goes through both. Without a
      // key no CORS headers go out, as before: the listener stays "a device on this network" (SECURITY.md), and a
      // web page of another site on the internet cannot read the model through it.
      if (hasKey()) {
        res.setHeader('access-control-allow-origin', '*');
        res.setHeader('access-control-allow-methods', 'GET, POST, OPTIONS');
        // the headers the preflight asks for (SDKs add their own; Anthropic browser clients send anthropic-version)
        res.setHeader('access-control-allow-headers', String(req.headers['access-control-request-headers'] ?? '') || 'authorization, content-type, x-api-key, x-strata-client, anthropic-version, anthropic-beta');
        res.setHeader('access-control-max-age', '600');
        if (method === 'OPTIONS') { res.writeHead(204); res.end(); return; }
      }
      if (!['/v1/models','/v1/chat/completions','/v1/messages'].includes(path)) { res.writeHead(404); res.end('Not found'); return; }
      try { await handle(req, res, path, method); }
      catch { if (!res.headersSent) res.writeHead(500); res.end('Request failed'); }
    });
    try { await new Promise((resolve,reject) => { listener.once('error',reject); listener.listen(port,host,resolve); }); }
    catch(e) { listener.close(); listener=null; throw new Error(listenProblem(e, port, host)); }
    active=true; changed(); return state();
  }
  async function stop() {
    if (!listener) return state();
    const current=listener; listener=null; active=false;
    current.closeAllConnections(); await new Promise(resolve=>current.close(resolve)); changed(); return state();
  }
  return { state, start, stop };
}
