/**
 * Strata Dev (docs/DEV_BUILD.md): backend hooks from dev/extensions/backend. Only index.mjs creates this, and only when STRATA_DEV=1.
 *
 * A hook file is one of:
 *   name.mjs / name.js  an ES module. Any of these exports is used:
 *                         before_request(body, ctx)    the chat request (/api/chat/stream) before it runs; return an object to replace it
 *                         on_token(frame, ctx)         every streamed `delta`/`reasoning` frame, synchronously; return an object to replace it
 *                         after_response(result, ctx)  the finished chat answer (text, requestId, error, durationMs)
 *                         on_model_load(model, ctx)    a model became ready
 *                         on_tool_call(call, ctx)      runs in the agent (agent/plugins/strata-dev-hooks.mjs); return { deny: 'why' } to refuse
 *                         routes { 'GET /ping': handler }   served at /dev/<name>/ping; a handler gets { method, path, query, body } and
 *                                                           returns an object (JSON), a string, or { status, json | text }
 *   name.py             a Python script, run once per call with `{ "event", "payload" }` as JSON on stdin; it prints one JSON object
 *                       (`{ "payload": ... }` to replace, `{ "deny": "..." }` for a tool call, or nothing). The events it takes are named
 *                       in a comment near the top: `# strata-hooks: before_request, after_response`. Python cannot take on_token.
 *
 * The fault boundary: a hook that throws, prints something that is not JSON or takes longer than its time limit is switched off
 * and logged, and the request goes on without it. The host never stops because of a hook. A hook is switched on again by Reload
 * (the Dev panel) or by its switch in the panel. A synchronous endless loop cannot be interrupted in JavaScript; keep on_token small.
 */
import { spawn } from 'node:child_process';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { basename, extname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';
import { push } from './dev-log.mjs';

export const HOOK_EVENTS = Object.freeze(['before_request', 'after_response', 'on_token', 'on_model_load', 'on_tool_call']);
const PY_DEFAULT_EVENTS = ['before_request', 'after_response'];
const TRACE_LIMIT = 200;
const ROUTE_NAME = /^[A-Za-z0-9_-]{1,64}$/;

/** The events a Python hook names in its `# strata-hooks:` line (the first 30 lines are read). on_token is never one of them. */
export function pythonEvents(source) {
  const line = String(source).split(/\r?\n/).slice(0, 30).find(l => /^\s*#\s*strata-hooks\s*:/i.test(l));
  if (!line) return [...PY_DEFAULT_EVENTS];
  return line.split(':').slice(1).join(':').split(/[\s,]+/).filter(e => HOOK_EVENTS.includes(e) && e !== 'on_token');
}

/** Runs `work` with a time limit. Resolves `{ value }`, or `{ error }` for a throw, a rejection or the timeout. Never rejects. */
export async function withTimeout(work, ms) {
  let timer;
  try {
    const value = await Promise.race([
      Promise.resolve().then(work),
      new Promise((_, reject) => { timer = setTimeout(() => reject(Object.assign(new Error(`took longer than ${ms} ms`), { code: 'timeout' })), ms); }),
    ]);
    return { value };
  } catch (error) {
    return { error };
  } finally {
    clearTimeout(timer);
  }
}

/** One call of a Python hook. Resolves the parsed JSON it printed (or null for no output); rejects for a failure or the time limit. */
export function runPythonHook({ python, file, event, payload, timeoutMs }) {
  return new Promise((resolveRun, reject) => {
    let child;
    try {
      child = spawn(python.command, [...(python.args ?? []), file], { stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true, env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' } });
    } catch (error) { reject(error); return; }
    let out = '', err = '', done = false;
    const finish = (fn, value) => { if (done) return; done = true; clearTimeout(timer); fn(value); };
    const timer = setTimeout(() => { try { child.kill(); } catch { /* gone */ } finish(reject, Object.assign(new Error(`took longer than ${timeoutMs} ms`), { code: 'timeout' })); }, timeoutMs);
    child.stdout.on('data', c => { out += c; });
    child.stderr.on('data', c => { err += c; if (err.length > 20_000) err = err.slice(-20_000); });
    child.on('error', error => finish(reject, error));
    child.on('close', code => {
      if (err.trim()) push('hooks', `${basename(file)}: ${err.trim()}`, code ? 'error' : 'info');
      if (code) return finish(reject, new Error(`exited with code ${code}${err.trim() ? `: ${err.trim().split('\n').pop()}` : ''}`));
      const text = out.trim();
      if (!text) return finish(resolveRun, null);
      const last = text.split(/\r?\n/).pop();
      try { finish(resolveRun, JSON.parse(last)); } catch { finish(reject, new Error(`printed something that is not JSON: ${last.slice(0, 200)}`)); }
    });
    child.stdin.on('error', () => { /* a script that does not read stdin */ });
    child.stdin.end(JSON.stringify({ event, payload }));
  });
}

const isPlainObject = value => !!value && typeof value === 'object' && !Array.isArray(value);

/**
 * The hook system. `dir` is dev/extensions/backend; `settings` is devSettings().hooks; `python` is `{ command, args }`.
 * `importModule` and `now` exist for the tests.
 */
export function createDevHooks({ dir, settings = {}, python = { command: process.platform === 'win32' ? 'python' : 'python3', args: [] }, importModule = url => import(url), now = () => performance.now(), onLog = null } = {}) {
  let options = { timeoutMs: 2000, tokenBudgetMs: 50, disabled: [], ...settings };
  /** @type {Array<{ file: string, path: string, name: string, kind: 'js'|'py', events: string[], handlers: Record<string, Function>, routes: Record<string, Function>, enabled: boolean, error: string|null, stats: Record<string, { calls: number, totalMs: number, maxMs: number }> }>} */
  let hooks = [];
  const traces = [];
  let generation = 0;

  // In the agent's process (the on_tool_call plugin) there is no Dev log buffer: its lines go to the console, which the host collects.
  const log = (line, level = 'info') => (onLog ? onLog(line, level) : push('hooks', line, level));

  function disable(hook, event, error) {
    hook.enabled = false;
    hook.error = `${event}: ${error?.message ?? error}`;
    log(`hook ${hook.file} switched off (${hook.error}); the request went on without it`, 'error');
  }

  function note(hook, event, ms, trace, error) {
    const s = hook.stats[event] ??= { calls: 0, totalMs: 0, maxMs: 0 };
    s.calls++; s.totalMs += ms; s.maxMs = Math.max(s.maxMs, ms);
    if (trace) {
      const row = trace.hooks.find(h => h.file === hook.file && h.event === event);
      if (row) { row.calls++; row.ms += ms; if (error) row.error = String(error.message ?? error); }
      else trace.hooks.push({ file: hook.file, event, calls: 1, ms, ...(error ? { error: String(error.message ?? error) } : {}) });
    }
  }

  async function load() {
    const loading = ++generation;
    let names = [];
    try { names = readdirSync(dir).filter(n => /\.(mjs|js|py)$/i.test(n) && !n.startsWith('_') && !n.startsWith('.')).sort(); }
    catch (error) { if (error?.code !== 'ENOENT') log(`cannot read ${dir}: ${error.message}`, 'error'); }
    const next = [];
    for (const file of names) {
      const path = join(dir, file);
      const kind = extname(file).toLowerCase() === '.py' ? 'py' : 'js';
      const hook = { file, path, name: basename(file, extname(file)), kind, events: [], handlers: {}, routes: {}, enabled: !options.disabled.includes(file), error: null, stats: {} };
      try {
        if (kind === 'py') {
          hook.events = pythonEvents(readFileSync(path, 'utf8'));
        } else {
          // A new URL each load, so Reload picks up an edited file without a host restart.
          const mod = await importModule(`${pathToFileURL(path).href}?v=${statSync(path).mtimeMs}-${loading}`);
          const source = { ...(isPlainObject(mod.default) ? mod.default : {}), ...mod };
          if (typeof source.name === 'string' && ROUTE_NAME.test(source.name)) hook.name = source.name;
          for (const event of HOOK_EVENTS) if (typeof source[event] === 'function') { hook.events.push(event); hook.handlers[event] = source[event]; }
          if (isPlainObject(source.routes)) {
            for (const [key, handler] of Object.entries(source.routes)) {
              const m = /^(GET|POST|PUT|PATCH|DELETE)\s+(\/\S*)$/i.exec(key.trim());
              if (m && typeof handler === 'function') hook.routes[`${m[1].toUpperCase()} ${m[2]}`] = handler;
              else log(`${file}: route "${key}" is not "METHOD /path" with a function; it is ignored`, 'warn');
            }
          }
        }
      } catch (error) {
        hook.enabled = false;
        hook.error = `load: ${error?.message ?? error}`;
        log(`hook ${file} could not be loaded: ${error?.stack ?? error}`, 'error');
      }
      next.push(hook);
    }
    if (loading !== generation) return list();
    hooks = next;
    log(`hooks loaded from ${dir}: ${hooks.length ? hooks.map(h => `${h.file}${h.enabled ? '' : ' (off)'}`).join(', ') : 'none'}`);
    return list();
  }

  function list() {
    return hooks.map(({ file, name, kind, events, routes, enabled, error, stats }) => ({ file, name, kind, events, routes: Object.keys(routes).map(r => r.replace(' ', ` /dev/${name}`)), enabled, error, stats }));
  }

  function setEnabled(file, enabled) {
    const hook = hooks.find(h => h.file === file);
    if (!hook) return false;
    hook.enabled = enabled === true;
    if (hook.enabled) hook.error = null;
    log(`hook ${file} switched ${hook.enabled ? 'on' : 'off'} in the Dev panel`);
    return true;
  }

  function configure(next = {}) { options = { ...options, ...next }; }

  /** Calls every enabled hook for `event` in file order. A hook may replace the payload (an object it returns). */
  async function run(event, payload, { trace = null, ctx = {} } = {}) {
    let current = payload;
    for (const hook of hooks) {
      if (!hook.enabled || !hook.events.includes(event)) continue;
      const started = now();
      const context = { event, file: hook.file, log: (line, level) => log(`${hook.file}: ${line}`, level), ...ctx };
      const result = hook.kind === 'py'
        ? await runPythonHook({ python, file: hook.path, event, payload: current, timeoutMs: options.timeoutMs }).then(value => ({ value }), error => ({ error }))
        : await withTimeout(() => hook.handlers[event](current, context), options.timeoutMs);
      const ms = now() - started;
      note(hook, event, ms, trace, result.error);
      if (result.error) { disable(hook, event, result.error); continue; }
      const value = hook.kind === 'py' ? (isPlainObject(result.value) && 'payload' in result.value ? result.value.payload : isPlainObject(result.value) && 'deny' in result.value ? result.value : undefined) : result.value;
      if (event === 'on_tool_call') { if (isPlainObject(value) && typeof value.deny === 'string') return { deny: value.deny, by: hook.file }; continue; }
      if (isPlainObject(value) && event === 'before_request') current = value;
    }
    return current;
  }

  /** on_token: synchronous, per frame. A hook over its budget (dev.config.json hooks.tokenBudgetMs) or one that throws is switched off. */
  function runToken(frame, trace) {
    let current = frame;
    for (const hook of hooks) {
      if (!hook.enabled || hook.kind !== 'js' || !hook.handlers.on_token) continue;
      const started = now();
      let value, error = null;
      try { value = hook.handlers.on_token(current, { event: 'on_token', file: hook.file }); } catch (e) { error = e; }
      const ms = now() - started;
      if (!error && value && typeof value.then === 'function') { value.catch?.(() => {}); value = undefined; error = null; }
      if (!error && ms > options.tokenBudgetMs) error = new Error(`took ${ms.toFixed(1)} ms for one token (budget ${options.tokenBudgetMs} ms)`);
      note(hook, 'on_token', ms, trace, error);
      if (error) { disable(hook, 'on_token', error); continue; }
      if (isPlainObject(value) && typeof value.type === 'string') current = value;
    }
    return current;
  }

  function newTrace(kind, info = {}) {
    const trace = { id: `trace-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`, kind, at: new Date().toISOString(), requestId: null, hooks: [], ...info };
    traces.push(trace);
    if (traces.length > TRACE_LIMIT) traces.splice(0, traces.length - TRACE_LIMIT);
    return trace;
  }

  /**
   * The chat route (/api/chat/stream): before_request on the body, then the response is watched line by line (on_token for
   * text frames) and after_response runs once it ended. `body` is changed in place. Call before the route writes anything.
   */
  async function chatStream(req, res, body) {
    const started = Date.now();
    const trace = newTrace('chat', { sessionId: typeof body.sessionId === 'string' ? body.sessionId : null });
    const replaced = await run('before_request', body, { trace, ctx: { route: '/api/chat/stream' } });
    if (replaced !== body && isPlainObject(replaced)) { for (const key of Object.keys(body)) delete body[key]; Object.assign(body, replaced); }
    // Watched even with no on_token/after_response hook: the Requests tab links this trace to the request id the frames carry.
    let text = '', error = null, frames = 0;
    const write = res.write.bind(res), end = res.end.bind(res);
    const inspect = line => {
      let frame;
      try { frame = JSON.parse(line); } catch { return line; }
      frames++;
      if (typeof frame.requestId === 'string') trace.requestId ??= frame.requestId;
      if (frame.type === 'error') error = frame.message ?? 'error';
      if ((frame.type === 'delta' || frame.type === 'reasoning') && hooks.some(h => h.enabled && h.handlers.on_token)) {
        const next = runToken(frame, trace);
        if (next !== frame) { frame = next; line = JSON.stringify(frame); }
      }
      if (frame.type === 'delta' && typeof frame.text === 'string') text += frame.text;
      return line;
    };
    res.write = (chunk, ...rest) => {
      if ((typeof chunk === 'string' || Buffer.isBuffer(chunk)) && String(chunk).endsWith('\n')) {
        try { chunk = String(chunk).split('\n').map(l => (l ? inspect(l) : l)).join('\n'); } catch (e) { log(`chat stream watch: ${e.message}`, 'error'); }
      }
      return write(chunk, ...rest);
    };
    let ended = false;
    res.end = (...args) => {
      const out = end(...args);
      if (!ended) {
        ended = true;
        trace.durationMs = Date.now() - started;
        trace.error = error;
        void run('after_response', { sessionId: trace.sessionId, requestId: trace.requestId, text, error, frames, durationMs: trace.durationMs }, { trace }).catch(() => {});
      }
      return out;
    };
    return trace;
  }

  /** The native engine's state changed (index.mjs, `changed`): on_model_load for each instance that just became ready. */
  const ready = new Set();
  function observeModels(state) {
    const instances = Array.isArray(state?.instances) ? state.instances : [];
    const now = new Set(instances.filter(i => i?.status === 'ready').map(i => i.id));
    for (const instance of instances) {
      if (!now.has(instance.id) || ready.has(instance.id)) continue;
      const trace = newTrace('model', { modelId: instance.apiModelId ?? instance.id });
      void run('on_model_load', { id: instance.id, label: instance.label ?? null, apiModelId: instance.apiModelId ?? null, port: instance.port ?? null, gpus: instance.gpus ?? [] }, { trace }).catch(() => {});
    }
    ready.clear();
    for (const id of now) ready.add(id);
  }

  /** /dev/<name>/<path>: true when a hook's route answered (or the name belongs to a hook), false to fall through to 404. */
  async function route(req, res, path, method, url, readBody) {
    const m = /^\/dev\/([^/]+)(\/.*)?$/.exec(path);
    if (!m) return false;
    const hook = hooks.find(h => h.name === m[1] && h.kind === 'js');
    const handler = hook?.routes[`${method} ${m[2] ?? '/'}`];
    if (!hook || !handler) return false;
    const send = (status, value) => {
      const text = typeof value === 'string' ? value : JSON.stringify(value ?? null);
      res.writeHead(status, { 'content-type': typeof value === 'string' ? 'text/plain; charset=utf-8' : 'application/json; charset=utf-8', 'cache-control': 'no-store' });
      res.end(text);
    };
    if (!hook.enabled) { send(503, { error: `The hook ${hook.file} is switched off${hook.error ? ` (${hook.error})` : ''}.` }); return true; }
    const body = ['POST', 'PUT', 'PATCH'].includes(method) ? await readBody(req).catch(() => ({})) : null;
    const started = now();
    const result = await withTimeout(() => handler({ method, path: m[2] ?? '/', query: Object.fromEntries(url.searchParams), body, headers: req.headers }), options.timeoutMs);
    note(hook, `route ${method} ${m[2] ?? '/'}`, now() - started, null, result.error);
    if (result.error) {
      log(`route ${method} ${path} in ${hook.file} failed: ${result.error?.stack ?? result.error}`, 'error');
      send(result.error.code === 'timeout' ? 504 : 500, { error: String(result.error.message ?? result.error) });
      return true;
    }
    const value = result.value;
    if (isPlainObject(value) && Number.isInteger(value.status) && ('json' in value || 'text' in value)) send(value.status, 'text' in value ? String(value.text) : value.json);
    else send(200, value === undefined ? { ok: true } : value);
    return true;
  }

  return {
    load, list, setEnabled, configure, run, runToken, chatStream, observeModels, route,
    traces: () => [...traces],
    traceFor: requestId => traces.find(t => t.requestId === requestId) ?? null,
    get dir() { return dir; },
  };
}
