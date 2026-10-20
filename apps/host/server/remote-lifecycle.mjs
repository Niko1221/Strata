/** Desktop-only prepare/serve workflow. No browser receives remote credentials or source paths. */
import { createHash, randomUUID } from 'node:crypto';
import { existsSync, lstatSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { isLocalCompanionRequest } from './companion-routes.mjs';
import { remoteTargetId } from './model-management.mjs';
import { sshManagementDestination } from './remote-model-ssh.mjs';
import { createRemoteLifecycleSSH } from './remote-lifecycle-ssh.mjs';
import { error, json, readBody } from './util.mjs';

const fail = (status, message) => Object.assign(new Error(message), { status });
const id = value => typeof value === 'string' && /^[a-zA-Z0-9_-]{1,160}$/.test(value);
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const namespaceOf = connection => hash(['strata-desktop-lifecycle', connection.id]);
const fingerprint = connection => hash([sshManagementDestination(connection), connection.baseUrl, connection.apiKey, connection.protocol]);
const managedFingerprint = connection => hash([connection.id, connection.name, connection.baseUrl, connection.apiKey, connection.protocol, connection.modelManagement, connection.models]);
const number = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined;
const text = (value, fallback = '') => typeof value === 'string' && !/[\x00-\x1f\x7f]|https?:\/\/|(?:^|\s)(?:\/[\w.-]+\/|~\/)/.test(value) ? value.slice(0, 1000) : fallback;
const pick = (value, keys) => Object.fromEntries(keys.filter(key => value[key] !== undefined).map(key => [key, value[key]]));
const warnings = value => Array.isArray(value) ? value.slice(0, 24).map(value => text(value)).filter(Boolean) : [];
function exact(body, keys, required = []) {
  if (!object(body) || Object.keys(body).some(key => !keys.includes(key)) || required.some(key => !Object.hasOwn(body, key))) throw fail(400, 'Choose a supported model lifecycle action.');
}
function publicJob(value) {
  if (!object(value) || !id(value.id)) throw fail(502, 'The SSH computer did not return a valid lifecycle job.');
  return { id: value.id, kind: ['prepare', 'serve'].includes(value.kind) ? value.kind : 'unknown', state: ['queued', 'running', 'done', 'error', 'cancelled'].includes(value.state) ? value.state : 'unknown',
    summary: text(value.summary, 'Refresh this operation to check its status.'), canCancel: value.kind === 'prepare' && value.canCancel === true,
    ...(id(value.requestId) ? { requestId: value.requestId } : {}), ...(id(value.profileId) ? { profileId: value.profileId } : {}), ...(id(value.receiptId) ? { receiptId: value.receiptId } : {}),
    ...(number(value.percent) !== undefined && value.percent <= 100 ? { percent: value.percent } : {}) };
}
function publicSnapshot(value, name) {
  if (!object(value) || !Array.isArray(value.models) || !Array.isArray(value.engines) || !Array.isArray(value.profiles) || !Array.isArray(value.jobs)) throw fail(502, 'The SSH computer did not return a valid model lifecycle snapshot.');
  const runtime = object(value.runtime) ? value.runtime : {};
  return { supported: value.supported === true, targetName: name, capabilities: { prepare: value.capabilities?.prepare === true, serve: value.capabilities?.serve === true },
    ...(value.reason ? { reason: text(value.reason, 'Review this computer’s lifecycle settings.') } : {}),
    models: value.models.filter(row => object(row) && id(row.id)).slice(0, 1000).map(row => ({ id: row.id, name: text(row.name, row.id), state: ['available', 'downloaded', 'imported', 'partial', 'downloading', 'prepared'].includes(row.state) ? row.state : 'unknown', preparable: row.preparable === true, ...(row.reason ? { reason: text(row.reason, 'This model needs attention before preparing.') } : {}), ...(id(row.profileId) ? { profileId: row.profileId } : {}) })),
    engines: value.engines.filter(row => object(row) && id(row.id)).slice(0, 128).map(row => ({ id: row.id, name: text(row.name, row.id), ...(row.backend ? { backend: text(row.backend) } : {}) })),
    profiles: value.profiles.filter(row => object(row) && id(row.id)).slice(0, 256).map(row => ({ id: row.id, name: text(row.name, row.id), ...(id(row.modelId) ? { modelId: row.modelId } : {}), ...(number(row.context) !== undefined ? { context: row.context } : {}), state: row.state === 'prepared' ? 'prepared' : 'unknown', executionCompatibility: 'unverified' })),
    jobs: value.jobs.slice(0, 256).map(publicJob), runtime: { state: ['stopped', 'ready', 'starting', 'failed', 'stale', 'unavailable'].includes(runtime.state) ? runtime.state : 'unknown',
      ...(id(runtime.profileId) ? { profileId: runtime.profileId } : {}), ...(runtime.modelId ? { modelId: text(runtime.modelId) } : {}), loaded: typeof runtime.loaded === 'boolean' ? runtime.loaded : null,
      busy: typeof runtime.busy === 'boolean' ? runtime.busy : null, ...(['0.0.0.0', '127.0.0.1'].includes(runtime.host) ? { host: runtime.host } : {}), ...(Number.isInteger(runtime.port) && runtime.port >= 1024 && runtime.port <= 65535 ? { port: runtime.port } : {}) } };
}

export function createRemoteLifecycle({ providers, directory, sshCall = createRemoteLifecycleSSH(), fetcher = fetch, now = Date.now, connectTimeoutMs = 10_000, maxConnectBytes = 512 * 1024 }) {
  const plans = new Map(), connecting = new Set();
  const linkFile = directory ? join(directory, 'remote-lifecycle-links.json') : null;
  let links = [];
  if (linkFile && existsSync(linkFile)) {
    const details = lstatSync(linkFile);
    if (!details.isFile() || details.isSymbolicLink() || details.size > 512 * 1024) throw fail(503, 'The private managed-connection records need repair.');
    const saved = JSON.parse(readFileSync(linkFile, 'utf8'));
    if (saved.schema !== 1 || !Array.isArray(saved.links) || saved.links.length > 1000 || saved.links.some(link => !id(link.id) || !id(link.originId) || !/^[a-f0-9]{64}$/.test(link.originFingerprint) || !/^[a-f0-9]{64}$/.test(link.managedFingerprint) || !/^[a-f0-9]{64}$/.test(link.namespace) || !id(link.receiptId))) throw fail(503, 'The private managed-connection records need repair.');
    links = saved.links;
  }
  function saveLinks(next) {
    if (!linkFile) throw fail(503, 'Private managed-connection storage is unavailable.');
    if (next.length > 1000) throw fail(429, 'Remove an old managed connection before adding another.');
    mkdirSync(directory, { recursive: true, mode: 0o700 });
    const temporary = `${linkFile}.${randomUUID()}.tmp`;
    writeFileSync(temporary, JSON.stringify({ schema: 1, links: next }), { flag: 'wx', mode: 0o600 }); renameSync(temporary, linkFile); links = next;
  }
  const all = () => providers.withSecrets().filter(value => value.kind === 'custom');
  function expose(value) {
    const secrets = all().map(saved => saved.apiKey).filter(value => typeof value === 'string' && value.length);
    const redact = value => typeof value === 'string' ? secrets.reduce((text, secret) => text.split(secret).join('[redacted]'), value)
      : Array.isArray(value) ? value.map(redact) : object(value) ? Object.fromEntries(Object.entries(value).map(([key, value]) => [key, redact(value)])) : value;
    return redact(value);
  }
  function resolve(targetId) {
    const saved = all().find(saved => remoteTargetId(saved) === targetId);
    if (!saved) throw fail(404, 'This Strata connection is no longer available.');
    if (saved.modelManagement?.transport !== 'ssh') throw fail(409, 'Configure Terminal server over SSH on this connection to prepare or serve model files.');
    const selected = structuredClone(saved), ownFingerprint = fingerprint(selected), link = links.find(link => link.id === selected.id);
    if (link) {
      const origin = all().find(saved => saved.id === link.originId);
      if (!origin || fingerprint(origin) !== link.originFingerprint || namespaceOf(origin) !== link.namespace || sshManagementDestination({ ...selected, id: origin.id }) !== sshManagementDestination(origin)) throw fail(409, 'The original SSH connection changed or was removed. Review it before controlling this managed API.');
      return { selected, fingerprint: ownFingerprint, namespace: link.namespace, originId: origin.id, originFingerprint: link.originFingerprint };
    }
    return { selected, fingerprint: ownFingerprint, namespace: namespaceOf(selected), originId: selected.id, originFingerprint: ownFingerprint };
  }
  function recheck(targetId, current) {
    const fresh = resolve(targetId);
    if (fresh.fingerprint !== current.fingerprint || fresh.namespace !== current.namespace || fresh.originFingerprint !== current.originFingerprint) throw fail(409, 'This connection changed. Refresh the model lifecycle and review again.');
  }
  async function call(targetId, current, action, body, { signal } = {}) {
    signal?.throwIfAborted(); recheck(targetId, current);
    const response = await sshCall(current.selected, action, body, { signal, namespace: current.namespace, destination: current.fingerprint });
    signal?.throwIfAborted(); recheck(targetId, current);
    if (!response || !Number.isInteger(response.status) || !object(response.body)) throw fail(502, 'The SSH computer did not return a valid lifecycle response.');
    if (response.status < 200 || response.status >= 300) throw fail(response.status, expose(text(response.body.error, 'The lifecycle action could not finish. Refresh the selected computer and review again.')));
    return response.body;
  }
  async function snapshot(targetId, context = {}) {
    let current;
    try { current = resolve(targetId); }
    catch (cause) { if (cause.status !== 409) throw cause; return { supported: false, targetName: all().find(saved => remoteTargetId(saved) === targetId)?.name ?? 'Strata server', reason: cause.message, capabilities: { prepare: false, serve: false }, models: [], engines: [], profiles: [], jobs: [], runtime: { state: 'unknown', loaded: null, busy: null } }; }
    return expose(publicSnapshot(await call(targetId, current, 'snapshot', undefined, context), text(current.selected.name, 'Strata server')));
  }
  async function plan(targetId, kind, body, context = {}) {
    if (kind === 'prepare') {
      exact(body, ['modelId', 'engineId', 'context'], ['modelId']);
      if (!id(body.modelId) || body.engineId !== undefined && !id(body.engineId) || body.context !== undefined && ![8192, 16384, 32768, 65536, 131072].includes(body.context)) throw fail(400, 'Choose a model, engine and supported context from this server’s library.');
    } else {
      exact(body, ['profileId', 'host', 'port'], ['profileId']);
      if (!id(body.profileId) || body.host !== undefined && !['0.0.0.0', '127.0.0.1'].includes(body.host) || body.port !== undefined && (!Number.isInteger(body.port) || body.port < 1024 || body.port > 65535)) throw fail(400, 'Choose a prepared profile and a separate serving address and port.');
    }
    const current = resolve(targetId), remote = await call(targetId, current, `${kind}-plan`, body, context);
    if (!id(remote.planId) || !Number.isFinite(remote.expiresAt) || remote.expiresAt <= now() || remote.expiresAt > now() + 301_000) throw fail(502, 'The SSH computer did not return a valid lifecycle preview.');
    for (const [key, value] of plans) if (value.expiresAt <= now()) plans.delete(key);
    if (plans.size >= 100) throw fail(429, 'Close an old lifecycle preview and try again shortly.');
    const planId = randomUUID(); plans.set(planId, { ...current, targetId, kind, remoteId: remote.planId, expiresAt: remote.expiresAt });
    const view = { planId, expiresAt: remote.expiresAt, targetName: text(current.selected.name, 'Strata server'), modelName: text(remote.modelName, 'Selected model'), warnings: warnings(remote.warnings) };
    if (kind === 'prepare') return expose({ ...view, engineName: text(remote.engineName, 'Selected Strata engine'), context: number(remote.context), diskNeededGb: number(remote.diskNeededGb), diskFreeGb: number(remote.diskFreeGb), diskShort: remote.diskShort === true });
    if (!id(remote.profileId) || !['start', 'switch'].includes(remote.action) || !['0.0.0.0', '127.0.0.1'].includes(remote.host) || !Number.isInteger(remote.port) || remote.port < 1024 || remote.port > 65535) { plans.delete(planId); throw fail(502, 'The SSH computer did not return a valid serving preview.'); }
    return expose({ ...view, ...pick(remote, ['profileId', 'host', 'port', 'action']) });
  }
  async function start(targetId, kind, body, context = {}) {
    exact(body, ['planId'], ['planId']);
    const saved = plans.get(body.planId);
    if (!saved || saved.targetId !== targetId || saved.kind !== kind || saved.expiresAt <= now()) throw fail(409, 'This lifecycle preview expired or was already used. Review it again.');
    plans.delete(body.planId); recheck(targetId, saved);
    return expose({ job: publicJob((await call(targetId, saved, kind, { planId: saved.remoteId }, context)).job) });
  }
  async function cancel(targetId, jobId, body, context = {}) {
    exact(body, []); if (!id(jobId)) throw fail(400, 'Choose a lifecycle job from this computer.');
    const current = resolve(targetId);
    return expose({ job: publicJob((await call(targetId, current, `jobs/${jobId}/cancel`, {}, context)).job) });
  }
  async function connect(targetId, body, context = {}) {
    exact(body, ['receiptId'], ['receiptId']); if (!id(body.receiptId)) throw fail(400, 'Choose a completed serving receipt.');
    const current = resolve(targetId), token = current.namespace;
    const existing = links.find(link => link.namespace === current.namespace && link.originFingerprint === current.originFingerprint && all().some(saved => saved.id === link.id));
    const linked = existing && all().find(saved => saved.id === existing.id);
    if (linked) {
      if (managedFingerprint(linked) !== existing.managedFingerprint) throw fail(409, 'The saved managed connection changed. Review its settings before connecting again.');
    }
    if (connecting.has(token)) throw fail(409, 'This managed connection is already being saved.');
    connecting.add(token);
    try {
      const endpoint = (await call(targetId, current, 'connect', body, context)).endpoint;
      if (!object(endpoint) || endpoint.receiptId !== body.receiptId || !['0.0.0.0', '127.0.0.1'].includes(endpoint.host) || !Number.isInteger(endpoint.port) || endpoint.port < 1024 || endpoint.port > 65535 || endpoint.basePath !== '/v1' || typeof endpoint.apiKey !== 'string' || !endpoint.apiKey || endpoint.apiKey.length > 4096 || /[\x00-\x20\x7f]/.test(endpoint.apiKey) || typeof endpoint.modelId !== 'string' || !endpoint.modelId || endpoint.modelId.length > 256 || /[\x00-\x1f\x7f]/.test(endpoint.modelId)) throw fail(502, 'The SSH computer did not return a valid managed serving receipt.');
      const base = new URL(current.selected.baseUrl);
      const loopback = ['localhost', '127.0.0.1', '[::1]'].includes(base.hostname);
      if (endpoint.host === '127.0.0.1' && !loopback) throw fail(409, 'This managed API is local-only on its server. Serve on the network to connect it from this Mac.');
      base.protocol = 'http:'; base.port = String(endpoint.port); base.pathname = '/v1'; base.search = ''; base.hash = '';
      const controller = AbortSignal.timeout(connectTimeoutMs), signal = context.signal ? AbortSignal.any([controller, context.signal]) : controller;
      let response;
      try {
        recheck(targetId, current); signal.throwIfAborted();
        response = await fetcher(base.href + '/status', { method: 'GET', redirect: 'error', signal, headers: { accept: 'application/json', authorization: `Bearer ${endpoint.apiKey}` } });
        if (!response.ok || response.redirected || !response.headers.get('content-type')?.includes('application/json')) throw Error();
        const reader = response.body?.getReader(); if (!reader) throw Error();
        let total = 0; const chunks = [];
        try { while (true) { const { value, done } = await reader.read(); signal.throwIfAborted(); if (done) break; total += value.byteLength; if (total > maxConnectBytes) throw Error(); chunks.push(Buffer.from(value)); } }
        catch (cause) { await reader.cancel().catch(() => {}); throw cause; } finally { reader.releaseLock(); }
        const status = JSON.parse(Buffer.concat(chunks).toString());
        if (status.service !== 'strata' || status.model !== endpoint.modelId || typeof status.loaded !== 'boolean') throw Error();
        recheck(targetId, current); signal.throwIfAborted();
      } catch { await response?.body?.cancel().catch(() => {}); throw fail(409, 'The managed API could not be verified from this Mac. Check its network address and port; the serving job remains on its server.'); }
      if (linked && managedFingerprint(all().find(saved => saved.id === linked.id) ?? {}) !== existing.managedFingerprint) throw fail(409, 'The saved managed connection changed. Review its settings before connecting again.');
      const before = new Set(all().map(saved => saved.id));
      const name = `${text(all().find(saved => saved.id === current.originId)?.name, 'Strata server')} · Managed ${text(endpoint.modelId, 'model')}`.slice(0, 160);
      providers.save({ ...(linked ? { id: linked.id } : {}), kind: 'custom', name, protocol: 'openai-chat', baseUrl: base.href, apiKey: endpoint.apiKey, modelManagement: current.selected.modelManagement, models: [{ id: endpoint.modelId, name: endpoint.modelId }] });
      const saved = all().find(saved => (linked ? saved.id === linked.id : !before.has(saved.id)) && saved.baseUrl.replace(/\/$/, '') === base.href.replace(/\/$/, '') && saved.apiKey === endpoint.apiKey);
      if (!saved) throw fail(503, 'The managed connection could not be saved.');
      try { saveLinks([...links.filter(link => link.id !== saved.id), { id: saved.id, originId: current.originId, originFingerprint: current.originFingerprint, managedFingerprint: managedFingerprint(saved), namespace: current.namespace, receiptId: body.receiptId }]); }
      catch (cause) { if (linked) providers.save({ ...linked, apiKey: linked.apiKey }); else providers.remove(saved.id); throw cause; }
      return { connectionId: saved.id, name: saved.name, targetId: remoteTargetId(saved) };
    } finally { connecting.delete(token); }
  }
  return { snapshot, plan, start, cancel, connect };
}

export function createRemoteLifecycleRoutes({ service, port }) {
  return async function routes(req, res, url) {
    for (const name of ['access-control-allow-origin', 'access-control-allow-headers', 'access-control-allow-methods']) res.removeHeader(name);
    res.setHeader('cache-control', 'no-store'); res.setHeader('x-content-type-options', 'nosniff');
    const controller = new AbortController(), disconnected = () => { if (!res.writableEnded) controller.abort(); };
    req.once('aborted', disconnected); res.once('close', disconnected);
    try {
      if (!isLocalCompanionRequest(req, typeof port === 'function' ? port() : port) || Object.keys(req.headers).some(name => name.startsWith('x-strata-companion'))) throw fail(403, 'Prepare and serve models from the Strata app on this Mac.');
      const match = /^\/api\/remote-lifecycle\/targets\/(remote-[a-f0-9]{24})(?:\/(prepare-plan|prepare|serve-plan|serve|connect|jobs\/([a-zA-Z0-9_-]{1,160})\/cancel))?$/.exec(url.pathname);
      if (!match || String(req.url).includes('?') || url.hash || req.method !== (match[2] ? 'POST' : 'GET')) throw fail(404, 'Unknown model lifecycle action.');
      let body;
      if (req.method === 'POST') { if (!/^application\/json(?:\s*;|$)/i.test(req.headers['content-type'] ?? '')) throw fail(415, 'Send this lifecycle action as JSON.'); body = await readBody(req, 4096); }
      if (req.aborted || res.destroyed) controller.abort(); controller.signal.throwIfAborted();
      const [, targetId, action, jobId] = match, context = { signal: controller.signal };
      const result = !action ? await service.snapshot(targetId, context)
        : action.endsWith('-plan') ? await service.plan(targetId, action.slice(0, -5), body, context)
        : ['prepare', 'serve'].includes(action) ? await service.start(targetId, action, body, context)
        : action === 'connect' ? await service.connect(targetId, body, context) : await service.cancel(targetId, jobId, body, context);
      if (!res.destroyed) return json(res, 200, result);
    } catch (cause) { if (!res.destroyed) return error(res, cause.status ?? 400, cause.status ? cause.message : 'This model lifecycle action could not finish. Refresh the selected computer and review again.'); }
    finally { req.removeListener('aborted', disconnected); res.removeListener('close', disconnected); }
  };
}
