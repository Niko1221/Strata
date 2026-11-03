/**
 * Strata Dev (docs/DEV_BUILD.md): the host routes of the Dev panel and of /dev-debug, under /api/dev/. index.mjs answers them
 * only when STRATA_DEV=1; outside Dev they do not exist (404 like any unknown path).
 *
 *   GET  /api/dev/state          config summary, hooks, log sources
 *   GET  /api/dev/logs           ?sources=host,engine,harness,ui,hooks&q=text&limit=N  merged and oldest first
 *   GET  /api/dev/requests       the Serving request log (serving.mjs) with each request's hook run times
 *   GET  /api/dev/debug          one text report for /dev-debug: last error, failing requests, hooks, the last log lines
 *   POST /api/dev/reload         dev.config.json and the backend hooks again (no host restart)
 *   POST /api/dev/hooks          { file, enabled } a hook's switch
 *   POST /api/dev/client-error   { message, stack, componentStack, extension } an error the page saw (Logs tab, source "ui")
 */
import * as devLog from './dev-log.mjs';

const ENGINE = 'engine';
const SOURCES = ['host', ENGINE, 'harness', 'hooks', 'ui', 'llama'];

/** Engine lines from supervisor.mjs (one engine) and native-control (Control's engines), as log entries. */
export function engineEntries({ supervisorLogs = [], nativeLogs = [] }) {
  return [
    ...supervisorLogs.map(e => ({ seq: 0, at: e.at, source: ENGINE, level: e.stream === 'stderr' ? 'warn' : 'info', line: String(e.line ?? '') })),
    ...nativeLogs.map(e => ({ seq: 0, at: e.at, source: ENGINE, level: /error|fail|traceback/i.test(String(e.line)) ? 'error' : 'info', line: `${e.instanceId ? `[${e.instanceId}] ` : ''}${String(e.line ?? '')}` })),
  ];
}

/** Every source merged by time, filtered by source names and a case-insensitive text, newest `limit` kept. */
export function mergedLogs({ sources = SOURCES, q = '', limit = 1000, engine = [] } = {}) {
  const wanted = sources.filter(s => SOURCES.includes(s));
  const own = devLog.entries({ sources: wanted.filter(s => s !== ENGINE), limit: 5000 });
  const rows = [...own, ...(wanted.includes(ENGINE) ? engine : [])].sort((a, b) => (a.at < b.at ? -1 : a.at > b.at ? 1 : a.seq - b.seq));
  const needle = String(q).trim().toLowerCase();
  const filtered = needle ? rows.filter(r => r.line.toLowerCase().includes(needle) || r.level === needle) : rows;
  return filtered.slice(-limit);
}

/** The Serving log joined with the hook traces: duration, tokens, error and the time each hook took for that request. */
export function requestRows({ requests = [], traces = [] }) {
  const byRequest = new Map(traces.filter(t => t.requestId).map(t => [t.requestId, t]));
  const rows = requests.map(r => {
    const trace = byRequest.get(r.id);
    return {
      id: r.id, at: r.at, client: r.client, endpoint: r.endpoint, model: r.model, state: r.state,
      durationMs: r.durationMs ?? null, tokensIn: r.tokensIn ?? null, tokensOut: r.tokensOut ?? null, tps: r.tps ?? null,
      error: r.error ?? null, hooks: trace?.hooks ?? [], hookMs: round((trace?.hooks ?? []).reduce((s, h) => s + h.ms, 0)),
    };
  });
  // Turns with hook work but no Serving entry (an agent turn on an API model, a model load) still show.
  const seen = new Set(rows.map(r => r.id));
  for (const t of traces) {
    if (t.requestId && seen.has(t.requestId)) continue;
    if (!t.hooks.length) continue;
    rows.push({ id: t.requestId ?? t.id, at: t.at, client: t.kind === 'model' ? 'Model load' : 'Strata chat', endpoint: t.kind === 'model' ? 'on_model_load' : '/api/chat/stream', model: t.modelId ?? null, state: 'hooks', durationMs: t.durationMs ?? null, tokensIn: null, tokensOut: null, tps: null, error: t.error ?? t.hooks.find(h => h.error)?.error ?? null, hooks: t.hooks, hookMs: round(t.hooks.reduce((s, h) => s + h.ms, 0)) });
  }
  return rows.sort((a, b) => (a.at < b.at ? 1 : -1));
}

const round = n => Math.round(n * 10) / 10;

/** A plain-text report for /dev-debug (.claude/commands/dev-debug.md). */
export function debugReport({ summary, hooks, requests, logs }) {
  const last = devLog.lastErrorEntry();
  const failing = requests.filter(r => r.error).slice(0, 10);
  const lines = [
    `# Strata Dev debug report (${new Date().toISOString()})`,
    '',
    `config: ${summary.configFile}`,
    ...summary.warnings.map(w => `config warning: ${w}`),
    `verbose logging: ${summary.verbose ? 'on' : 'off'}`,
    '',
    '## Last error',
    last ? `${last.at} [${last.source}] ${last.line}${last.detail ? `\n${last.detail}` : ''}` : '(none)',
    '',
    '## Hooks',
    ...(hooks.length ? hooks.map(h => `- ${h.file} (${h.kind}) ${h.enabled ? 'on' : 'OFF'} events=${h.events.join(',') || '-'}${h.routes.length ? ` routes=${h.routes.join(',')}` : ''}${h.error ? ` error=${h.error}` : ''}`) : ['(none)']),
    '',
    '## Failed requests',
    ...(failing.length ? failing.map(r => `- ${r.at} ${r.endpoint} ${r.model ?? ''} ${r.durationMs ?? '?'} ms: ${r.error}`) : ['(none)']),
    '',
    '## Errors and warnings in the log',
    ...logs.filter(l => l.level === 'error' || l.level === 'warn').slice(-40).map(l => `${l.at} [${l.source}/${l.level}] ${l.line}`),
    '',
    '## Last log lines',
    ...logs.slice(-150).map(l => `${l.at} [${l.source}] ${l.line}`),
  ];
  return `${lines.join('\n')}\n`;
}

/**
 * The route handler. `deps`: { json, error, readBody, summary(), reload(), hooks, serving, supervisorLogs(), nativeLogs() }.
 * Returns undefined when the path is not one of its own (the caller goes on to its 404).
 */
export async function devApi(req, res, path, method, url, deps) {
  if (!path.startsWith('/api/dev/')) return undefined;
  const { json, error, readBody, hooks } = deps;
  const engine = () => engineEntries({ supervisorLogs: deps.supervisorLogs(), nativeLogs: deps.nativeLogs() });
  if (path === '/api/dev/state' && method === 'GET') return json(res, 200, { dev: deps.summary(), hooks: hooks.list(), sources: SOURCES, hooksDir: hooks.dir });
  if (path === '/api/dev/logs' && method === 'GET') {
    const sources = (url.searchParams.get('sources') ?? '').split(',').map(s => s.trim()).filter(Boolean);
    return json(res, 200, { entries: mergedLogs({ sources: sources.length ? sources : SOURCES, q: url.searchParams.get('q') ?? '', limit: Math.min(5000, Number(url.searchParams.get('limit')) || 1000), engine: engine() }) });
  }
  if (path === '/api/dev/requests' && method === 'GET') return json(res, 200, { requests: requestRows({ requests: deps.serving.state().requests, traces: hooks.traces() }) });
  if (path === '/api/dev/debug' && method === 'GET') {
    const text = debugReport({ summary: deps.summary(), hooks: hooks.list(), requests: requestRows({ requests: deps.serving.state().requests, traces: hooks.traces() }), logs: mergedLogs({ limit: 2000, engine: engine() }) });
    res.writeHead(200, { 'content-type': 'text/plain; charset=utf-8', 'cache-control': 'no-store' });
    res.end(text);
    return true;
  }
  if (path === '/api/dev/reload' && method === 'POST') { await deps.reload(); return json(res, 200, { dev: deps.summary(), hooks: hooks.list() }); }
  if (path === '/api/dev/hooks' && method === 'POST') {
    const b = await readBody(req);
    if (typeof b.file !== 'string' || typeof b.enabled !== 'boolean') return error(res, 400, 'file and enabled are required');
    if (!hooks.setEnabled(b.file, b.enabled)) return error(res, 404, `No hook named ${b.file}`);
    return json(res, 200, { hooks: hooks.list() });
  }
  if (path === '/api/dev/client-error' && method === 'POST') {
    const b = await readBody(req);
    const where = typeof b.extension === 'string' && b.extension ? `extension ${b.extension}: ` : '';
    devLog.push('ui', [`${where}${String(b.message ?? 'error').slice(0, 2000)}`, typeof b.stack === 'string' ? b.stack.slice(0, 6000) : '', typeof b.componentStack === 'string' ? `component stack:${b.componentStack.slice(0, 6000)}` : ''].filter(Boolean).join('\n'), b.level === 'warn' ? 'warn' : 'error');
    return json(res, 200, { ok: true });
  }
  return error(res, 404, `No route for ${method} ${path}`);
}
