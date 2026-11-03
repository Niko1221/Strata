/**
 * Strata Dev (docs/DEV_BUILD.md): the llama.cpp runner. Only index.mjs creates it, and only when STRATA_DEV=1.
 *
 * A .gguf the Strata engine is not tuned for (SUPPORTED_SELECTIONS) runs on llama.cpp's own server instead: the runner starts the
 * person's `llama-server` (path in dev.config.json, never downloaded), waits until its /health answers, and saves it as an API
 * connection (server/model-providers.mjs), so the chat's model picker, the agent, skills and hooks use it without a new path.
 * One model at a time. Stop ends the process; the app ending (host shutdown, or the host process exiting) ends it too.
 *
 * dev.config.json:
 *   "llamacpp": { "server": "C:\\llama.cpp\\llama-server.exe", "port": 8095, "args": ["-ngl", "99"], "context": 8192 }
 *
 * Every failure (no or a wrong path, a taken or forbidden port, a crash, no answer in time) is a state with a plain message, in the
 * Dev panel's Logs (source "llama") and in the model menu. Nothing here throws into the host.
 */
import { spawn } from 'node:child_process';
import { existsSync, statSync } from 'node:fs';
import net from 'node:net';
import { basename } from 'node:path';
import { push } from './dev-log.mjs';

/** Ports the runner never takes: Live's and Dev's own (window, host, engine), the share/tunnel helpers beside them, 8090. */
export const FORBIDDEN_PORTS = Object.freeze([8080, 8081, 8090, 8791, 8792, 8793, 8794, 47831, 47832]);
export const DEFAULT_PORT = 8095;
const CONNECTION_NAME = 'llama.cpp (Strata Dev)';

/** The llamacpp section of dev.config.json in plain form; wrong values left out (never a throw). */
export function llamaSettings(raw) {
  const s = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {};
  const port = Number.isInteger(s.port) && s.port >= 1024 && s.port <= 65535 ? s.port : DEFAULT_PORT;
  return {
    server: typeof s.server === 'string' && s.server.trim() ? s.server.trim() : null,
    port,
    args: Array.isArray(s.args) ? s.args.filter(a => typeof a === 'string' || typeof a === 'number').map(String) : [],
    context: Number.isInteger(s.context) && s.context >= 512 ? s.context : 8192,
    healthTimeoutMs: Number.isInteger(s.healthTimeoutMs) && s.healthTimeoutMs >= 1000 ? s.healthTimeoutMs : 180_000,
  };
}

/** Why the runner cannot start with these settings (null: it can try). */
export function settingsProblem(settings, { exists = existsSync, isFile = p => statSync(p).isFile() } = {}) {
  if (!settings.server) return 'No llama-server is set. Put its path in dev.config.json: "llamacpp": { "server": "C:\\\\llama.cpp\\\\llama-server.exe" }.';
  if (!exists(settings.server)) return `llama-server is not at ${settings.server}. Check "llamacpp.server" in dev.config.json.`;
  try { if (!isFile(settings.server)) return `${settings.server} is a folder; "llamacpp.server" names the llama-server program itself.`; } catch { /* checked above */ }
  if (FORBIDDEN_PORTS.includes(settings.port)) return `Port ${settings.port} belongs to Strata (Live or Dev). Choose another "llamacpp.port", for example ${DEFAULT_PORT}.`;
  return null;
}

/** The llama-server command line for one model. `--alias` is the model id the connection uses. */
export function llamaArgs({ gguf, alias, port, context, extra = [] }) {
  return ['-m', gguf, '--host', '127.0.0.1', '--port', String(port), '-c', String(context), '--alias', alias, ...extra];
}

export const aliasFor = gguf => basename(String(gguf)).replace(/\.gguf$/i, '').replace(/-0+1-of-0*\d+$/i, '').toLowerCase().replace(/[^a-z0-9._-]+/g, '-').slice(0, 80) || 'model';

function portFree(port) {
  return new Promise(resolve => {
    const probe = net.createServer();
    probe.once('error', () => resolve(false));
    probe.listen(port, '127.0.0.1', () => probe.close(() => resolve(true)));
  });
}

/**
 * The runner. `providers`: the host's model-providers (list/save/remove). `spawnImpl` and `fetchImpl` exist for the tests (a mock server).
 * `getSettings()` is read at every start, so a Reload of dev.config.json applies to the next start.
 */
export function createLlamaRunner({ getSettings, providers, spawnImpl = spawn, fetchImpl = fetch, log = (line, level) => push('llama', line, level), isPortFree = portFree, exists = existsSync }) {
  let child = null;
  let state = { status: 'stopped', model: null, gguf: null, port: null, pid: null, error: null, connectionId: null, startedAt: null };
  const tail = [];
  const keep = text => { for (const line of String(text).split(/\r?\n/).filter(l => l.trim())) { tail.push(line); if (tail.length > 40) tail.shift(); log(line); } };
  const set = patch => { state = { ...state, ...patch }; return view(); };
  const view = () => ({ ...state, available: !settingsProblem(getSettings(), { exists }), problem: settingsProblem(getSettings(), { exists }), port: state.port ?? getSettings().port, lastLines: tail.slice(-12) });
  const fail = (message) => { log(message, 'error'); return set({ status: 'error', error: message, pid: null }); };

  function saveConnection(port, alias, label, context) {
    // One connection for the runner, replaced at every start (model-providers.mjs: list() and save() return { catalog, connections }).
    const existing = (providers.list().connections ?? []).find(c => c.name === CONNECTION_NAME);
    const saved = providers.save({ ...(existing ? { id: existing.id } : {}), kind: 'custom', protocol: 'openai-chat', name: CONNECTION_NAME, baseUrl: `http://127.0.0.1:${port}/v1`, apiKey: '', models: [{ id: alias, name: `${label} · läuft über llama.cpp`, contextMax: context }] });
    return (saved?.connections ?? []).find(c => c.name === CONNECTION_NAME)?.id ?? existing?.id ?? null;
  }

  async function waitHealthy(port, timeoutMs, exited) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      if (exited()) return false;
      try { const r = await fetchImpl(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(2000) }); if (r.status === 200) return true; } catch { /* not up yet */ }
      await new Promise(r => setTimeout(r, 300));
    }
    return false;
  }

  async function start({ gguf, label = null, context = null } = {}) {
    try {
      if (child) await stop();
      const settings = getSettings();
      const problem = settingsProblem(settings, { exists });
      if (problem) return fail(problem);
      if (typeof gguf !== 'string' || !gguf.trim() || !exists(gguf)) return fail(`No .gguf at ${gguf}.`);
      if (!(await isPortFree(settings.port))) return fail(`Port ${settings.port} is in use by another program. Free it or set another "llamacpp.port" in dev.config.json.`);
      const alias = aliasFor(gguf), ctx = Number.isInteger(context) && context >= 512 ? context : settings.context;
      const args = llamaArgs({ gguf, alias, port: settings.port, context: ctx, extra: settings.args });
      tail.length = 0;
      log(`starting ${settings.server} ${args.join(' ')}`);
      let exited = false, spawnError = null;
      let proc;
      try { proc = spawnImpl(settings.server, args, { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true }); }
      catch (e) { return fail(`llama-server could not be started: ${e.message}`); }
      child = proc;
      set({ status: 'starting', model: alias, gguf, port: settings.port, pid: proc.pid ?? null, error: null, startedAt: new Date().toISOString() });
      proc.stdout?.on('data', keep);
      proc.stderr?.on('data', keep);
      proc.on('error', e => { spawnError = e; exited = true; });
      proc.on('exit', (code, signal) => {
        exited = true;
        if (child === proc) child = null;
        if (state.status === 'stopping' || state.status === 'stopped') { set({ status: 'stopped', pid: null }); return; }
        const why = `llama-server ended (${signal ?? `code ${code}`})${tail.length ? `: ${tail.at(-1)}` : ''}`;
        fail(state.status === 'running' ? `${why}. The model is no longer served.` : why);
      });
      const ok = await waitHealthy(settings.port, settings.healthTimeoutMs, () => exited);
      if (!ok) {
        if (spawnError) return fail(`llama-server could not be started: ${spawnError.message}`);
        if (exited) return view();
        try { proc.kill(); } catch { /* gone */ }
        return fail(`llama-server did not answer on port ${settings.port} within ${Math.round(settings.healthTimeoutMs / 1000)} s.`);
      }
      let connectionId = null;
      try { connectionId = saveConnection(settings.port, alias, label || alias, ctx); }
      catch (e) { log(`the API connection could not be saved: ${e.message}`, 'error'); }
      log(`llama-server serves ${alias} on http://127.0.0.1:${settings.port}/v1 (connection "${CONNECTION_NAME}")`);
      return set({ status: 'running', connectionId });
    } catch (e) {
      return fail(`llama.cpp runner: ${e?.message ?? e}`);
    }
  }

  async function stop() {
    const proc = child;
    if (!proc) return set({ status: state.status === 'error' ? 'error' : 'stopped', pid: null });
    set({ status: 'stopping' });
    await new Promise(resolve => {
      const timer = setTimeout(() => { try { proc.kill('SIGKILL'); } catch { /* gone */ } resolve(); }, 5000);
      proc.once('exit', () => { clearTimeout(timer); resolve(); });
      try { proc.kill(); } catch { clearTimeout(timer); resolve(); }
    });
    child = null;
    log('llama-server stopped');
    return set({ status: 'stopped', pid: null });
  }

  /** For the host's exit: no waiting, the process is ended now. */
  function killNow() { try { child?.kill(); } catch { /* gone */ } }

  return { start, stop, killNow, state: view };
}

/**
 *   GET  /api/dev/llama          the runner's state (and why it cannot start, when it cannot)
 *   POST /api/dev/llama/start    { gguf, label?, context? }
 *   POST /api/dev/llama/stop
 */
export async function devLlamaApi(req, res, path, method, url, { json, error, readBody, runner }) {
  if (!path.startsWith('/api/dev/llama')) return undefined;
  if (path === '/api/dev/llama' && method === 'GET') return json(res, 200, runner.state());
  if (path === '/api/dev/llama/start' && method === 'POST') { const b = await readBody(req); return json(res, 200, await runner.start({ gguf: b.gguf, label: typeof b.label === 'string' ? b.label : null, context: Number.isInteger(b.context) ? b.context : null })); }
  if (path === '/api/dev/llama/stop' && method === 'POST') return json(res, 200, await runner.stop());
  return error(res, 404, `No route for ${method} ${path}`);
}
