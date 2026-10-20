/** Reviewed current-model controls for a saved headless Strata API connection. */
import { createHash, randomUUID } from 'node:crypto';
import { remoteTargetId } from './model-management.mjs';
import { isLocalCompanionRequest } from './companion-routes.mjs';
import { error, json, readBody } from './util.mjs';

const fail = (status, message) => Object.assign(new Error(message), { status });
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const identifier = value => typeof value === 'string' && value.length > 0 && value.length <= 256 && !/[\x00-\x1f\x7f]/.test(value);
const count = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
const unsupported = (targetName, reason, modelId, loaded = null) => ({ supported: false, targetName, ...(modelId ? { modelId } : {}), loaded, busy: null, capabilities: { load: false, unload: false }, reason });
function exact(body, keys) {
  if (!object(body) || Object.keys(body).some(key => !keys.includes(key))) throw fail(400, 'Choose a supported current-model action.');
}
function destination(connection) {
  let url;
  try { url = new URL(connection.baseUrl); } catch { throw fail(400, 'Save a valid API address for this Strata connection.'); }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw fail(400, 'Save a valid API address for this Strata connection.');
  let path = url.pathname.replace(/\/+$/, '');
  if (!path.endsWith('/v1')) path += '/v1';
  url.pathname = path;
  return { base: url.href.replace(/\/$/, ''), fingerprint: createHash('sha256').update(JSON.stringify([connection.id, connection.baseUrl, connection.apiKey, connection.protocol])).digest('hex') };
}

export function createRemoteServing({ providers, fetcher = fetch, now = Date.now, timeoutMs = 120_000, readTimeoutMs = 10_000, maxResponseBytes = 512 * 1024, planLifetimeMs = 60_000 }) {
  const plans = new Map(), running = new Set();
  function connection(targetId) {
    const saved = providers.withSecrets().find(saved => saved.kind === 'custom' && remoteTargetId(saved) === targetId);
    if (!saved) throw fail(404, 'This Strata connection is no longer available.');
    return structuredClone(saved);
  }
  function check(targetId, expected) {
    if (destination(connection(targetId)).fingerprint !== expected) throw fail(409, 'This connection changed. Refresh its serving status and review again.');
  }
  async function request(saved, targetId, expected, path, method, body, { signal } = {}, optional = false) {
    signal?.throwIfAborted(); check(targetId, expected);
    const deadline = AbortSignal.timeout(method === 'GET' ? readTimeoutMs : timeoutMs);
    const combined = signal ? AbortSignal.any([signal, deadline]) : deadline;
    let response;
    try {
      response = await fetcher(destination(saved).base + path, { method, redirect: 'error', signal: combined,
        headers: { accept: 'application/json', authorization: `Bearer ${saved.apiKey}`, ...(body === undefined ? {} : { 'content-type': 'application/json' }) },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
      check(targetId, expected); combined.throwIfAborted();
      if (optional && [404, 405, 501].includes(response.status)) { await response.body?.cancel(); return null; }
      if (!response.ok || response.redirected) {
        await response.body?.cancel();
        if ([401, 403].includes(response.status)) throw fail(401, 'This server rejected the saved API key or this serving action.');
        if (response.status === 409) throw fail(409, 'This server is busy. Wait for current requests to finish and review again.');
        if ([404, 405, 501].includes(response.status)) throw fail(404, 'This API does not provide current-model serving controls.');
        throw fail(502, 'This server could not finish the serving request. Refresh its status and try again.');
      }
      if (!response.headers?.get('content-type')?.toLowerCase().includes('application/json')) throw fail(502, 'This server did not return Strata serving status.');
      const reader = response.body?.getReader();
      if (!reader) throw fail(502, 'This server did not return Strata serving status.');
      let size = 0; const chunks = [];
      try {
        while (true) {
          const { value, done } = await reader.read();
          combined.throwIfAborted(); check(targetId, expected);
          if (done) break;
          size += value.byteLength;
          if (size > maxResponseBytes) throw fail(502, 'This server sent an oversized serving response.');
          chunks.push(Buffer.from(value));
        }
      } catch (cause) { await reader.cancel().catch(() => {}); throw cause; }
      finally { reader.releaseLock(); }
      let result; try { result = JSON.parse(Buffer.concat(chunks).toString('utf8')); } catch { throw fail(502, 'This server did not return Strata serving status.'); }
      if (!object(result)) throw fail(502, 'This server did not return Strata serving status.');
      check(targetId, expected); combined.throwIfAborted();
      return result;
    } catch (cause) {
      await response?.body?.cancel().catch(() => {});
      if (cause.status) throw cause;
      if (signal?.aborted) throw fail(499, 'This serving request was cancelled before completion. Refresh its status.');
      if (deadline.aborted) throw fail(504, 'This server did not answer in time. Refresh its serving status before trying again.');
      throw fail(503, 'Could not reach this Strata API. Check the saved connection and try again.');
    }
  }
  async function inspect(targetId, context) {
    const saved = connection(targetId), name = typeof saved.name === 'string' ? saved.name.slice(0, 160) : 'Strata server';
    if (!saved.apiKey) return { view: unsupported(name, 'Save this server’s API key in the connection before using serving controls.') };
    const fingerprint = destination(saved).fingerprint;
    let status;
    try { status = await request(saved, targetId, fingerprint, '/status', 'GET', undefined, context); }
    catch (cause) { if ([404, 502].includes(cause.status)) return { view: unsupported(name, 'This API does not advertise Strata current-model serving controls.') }; throw cause; }
    const modelId = identifier(status.model) ? status.model : undefined;
    if (status.service !== 'strata' || !modelId) return { view: unsupported(name, 'This API does not advertise Strata current-model serving controls.') };
    if (typeof status.loaded !== 'boolean') return { view: unsupported(name, 'This server does not report whether its current model is loaded. Serving controls require a Terminal version that reports this state.', modelId) };
    const telemetry = await request(saved, targetId, fingerprint, '/telemetry', 'GET', undefined, context, true);
    if (telemetry?.model?.id !== undefined && telemetry.model.id !== modelId || typeof telemetry?.model?.loaded === 'boolean' && telemetry.model.loaded !== status.loaded) throw fail(409, 'The server’s model changed while reading its status. Refresh and review again.');
    const work = object(telemetry?.work) ? telemetry.work : {};
    const counts = [work.admission_reserved, work.active, work.queued, work.waiting, status.activity?.in_flight, status.active, status.queued].map(count);
    const busy = status.busy === true || counts.some(value => value !== null && value > 0) || telemetry?.instance?.state === 'draining' ? true
      : count(work.admission_reserved) === 0 ? false : null;
    const actionable = busy === false;
    const view = { supported: true, targetName: name, modelId, loaded: status.loaded, busy,
      capabilities: { load: actionable && !status.loaded, unload: actionable && status.loaded },
      ...(busy === true ? { reason: 'Wait for this server’s current requests to finish before changing the loaded state.' }
        : busy === null ? { reason: 'This server does not report reliable request admission counts. Serving controls stay read-only.' } : {}) };
    return { view, saved, fingerprint, instanceId: identifier(telemetry?.instance?.id) ? telemetry.instance.id : null, started: count(status.started) };
  }
  function requireAction(current, action) {
    if (!current.view.supported || current.view.busy !== false || !current.view.capabilities[action]) throw fail(409, current.view.reason ?? 'This model’s loaded state changed. Refresh and review again.');
  }
  async function snapshot(targetId, context = {}) { return (await inspect(targetId, context)).view; }
  async function plan(targetId, body, context = {}) {
    exact(body, ['action']);
    if (!['load', 'unload'].includes(body.action)) throw fail(400, 'Choose Load or Unload for the current server model.');
    const current = await inspect(targetId, context); requireAction(current, body.action);
    for (const [key, value] of plans) if (value.expiresAt <= now()) plans.delete(key);
    if (plans.size >= 100) throw fail(429, 'Close an old serving preview and try again shortly.');
    const planId = randomUUID(), expiresAt = now() + planLifetimeMs;
    plans.set(planId, { targetId, fingerprint: current.fingerprint, modelId: current.view.modelId, loaded: current.view.loaded, instanceId: current.instanceId, started: current.started, action: body.action, expiresAt });
    return { planId, targetName: current.view.targetName, modelId: current.view.modelId, action: body.action, expiresAt,
      warnings: body.action === 'load' ? ['Load only the model already configured on this server. Existing before-load hooks may run.', 'The API key, listening address, port and model configuration stay as configured on the server.']
        : ['Unload this server’s current model from memory. Its API stays available.', 'The next inference request may automatically load this model again.'] };
  }
  async function execute(targetId, body, context = {}) {
    exact(body, ['planId']);
    const preview = typeof body.planId === 'string' ? plans.get(body.planId) : null;
    if (!preview || preview.targetId !== targetId || preview.expiresAt <= now()) throw fail(409, 'This serving preview expired or was already used. Review the action again.');
    plans.delete(body.planId);
    check(targetId, preview.fingerprint);
    if (running.has(targetId)) throw fail(409, 'A serving action is already running for this connection.');
    running.add(targetId);
    try {
      const current = await inspect(targetId, context);
      if (current.fingerprint !== preview.fingerprint || current.view.modelId !== preview.modelId || current.view.loaded !== preview.loaded || current.instanceId !== preview.instanceId || current.started !== preview.started) throw fail(409, 'This server changed after the preview. Refresh its status and review again.');
      requireAction(current, preview.action);
      context.signal?.throwIfAborted(); check(targetId, preview.fingerprint);
      await request(current.saved, targetId, preview.fingerprint, preview.action === 'load' ? '/load' : '/unload?wait=0', 'POST', { model: preview.modelId }, context);
      const final = await inspect(targetId, context);
      if (final.view.modelId !== preview.modelId || final.instanceId !== preview.instanceId || final.started !== preview.started) throw fail(409, 'The server changed during this action. Refresh its serving status before another action.');
      return final.view;
    } finally { running.delete(targetId); }
  }
  return { snapshot, plan, execute };
}

export function createRemoteServingRoutes({ service, port }) {
  return async function remoteServingRoutes(req, res, url) {
    for (const name of ['access-control-allow-origin', 'access-control-allow-headers', 'access-control-allow-methods']) res.removeHeader(name);
    res.setHeader('cache-control', 'no-store'); res.setHeader('x-content-type-options', 'nosniff');
    const controller = new AbortController();
    const disconnected = () => { if (!res.writableEnded) controller.abort(); };
    req.once('aborted', disconnected); res.once('close', disconnected);
    try {
      if (!isLocalCompanionRequest(req, typeof port === 'function' ? port() : port) || Object.keys(req.headers).some(name => name.startsWith('x-strata-companion'))) throw fail(403, 'Open Serving in the Strata app on this Mac to control a saved server.');
      const match = /^\/api\/remote-serving\/targets\/(remote-[a-f0-9]{24})(\/plan)?$/.exec(url.pathname);
      if (!match || String(req.url).includes('?') || url.hash || !['GET', 'POST'].includes(req.method) || req.method === 'GET' && match[2]) throw fail(404, 'Unknown remote serving action.');
      if (req.aborted || res.destroyed) controller.abort();
      const context = { signal: controller.signal }, targetId = match[1];
      let body;
      if (req.method === 'POST') {
        if (!req.headers['content-type']?.toLowerCase().startsWith('application/json')) throw fail(415, 'Send this serving action as JSON.');
        body = await readBody(req, 2 * 1024);
      }
      controller.signal.throwIfAborted();
      const result = req.method === 'GET' ? await service.snapshot(targetId, context)
        : match[2] ? await service.plan(targetId, body, context) : await service.execute(targetId, body, context);
      if (!res.destroyed) return json(res, 200, result);
    } catch (cause) {
      if (!res.destroyed) return error(res, cause.status ?? 400, cause.status ? cause.message : 'This serving action could not finish. Refresh the server status and try again.');
    } finally { req.removeListener('aborted', disconnected); res.removeListener('close', disconnected); }
  };
}
