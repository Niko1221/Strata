import http from 'node:http';
import https from 'node:https';
import { createHash, randomBytes, randomInt, timingSafeEqual, X509Certificate } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, chmodSync, renameSync, existsSync, unlinkSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { homedir, hostname, networkInterfaces } from 'node:os';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

const run = promisify(execFile);
const here = dirname(fileURLToPath(import.meta.url));
const digest = value => createHash('sha256').update(value).digest('hex');
const same = (a, b) => typeof a === 'string' && typeof b === 'string' && a.length === b.length && timingSafeEqual(Buffer.from(a), Buffer.from(b));
const id = () => randomBytes(16).toString('hex');
const fail = (status, message) => Object.assign(new Error(message), { status });
const finite = value => Number.isFinite(value) ? value : null;
const encode = value => encodeURIComponent(String(value));
const safeId = value => typeof value === 'string' && /^[a-zA-Z0-9_-]{1,120}$/.test(value);
const safeToolId = value => typeof value === 'string' && /^[a-zA-Z0-9_.:-]{1,300}$/.test(value);
const shortText = (value, max) => typeof value === 'string' ? value.slice(0, max) : '';
const managementTarget = '(?:local|remote-[a-f0-9]{16,64})';
const managementRoute = new RegExp(`^/api/model-management/targets(?:/(${managementTarget})(?:/(plan|downloads)(?:/([a-zA-Z0-9_-]{1,120})/cancel)?)?)?$`);

function managementRequest(path, method, body, deviceId) {
  const match = managementRoute.exec(path);
  if (!match || !/^[a-f0-9]{32}$/.test(deviceId ?? '')) throw fail(403, 'Unsupported model management request.');
  const [, targetId, action, jobId] = match;
  const expected = action ? 'POST' : 'GET';
  if (method !== expected || (jobId && action !== 'downloads')) throw fail(403, 'Unsupported model management method.');
  if (!action) {
    if (body !== undefined) throw fail(400, 'Read model information without changing settings.');
  } else if (!body || typeof body !== 'object' || Array.isArray(body)) {
    throw fail(400, 'Send a model management request.');
  } else if (action === 'plan') {
    if (Object.keys(body).some(key => !['modelId', 'readPictures'].includes(key)) || !safeId(body.modelId) || typeof body.readPictures !== 'boolean') throw fail(400, 'Choose a catalog model and its picture-reading option.');
  } else if (jobId) {
    if (Object.keys(body).length) throw fail(400, 'Stop only this download.');
  } else if (Object.keys(body).length !== 1 || !safeId(body.planId)) {
    throw fail(400, 'Review a download plan before starting.');
  }
  return { targetId, action, jobId };
}

function privateJSON(file, value) {
  const temp = `${file}.${process.pid}.tmp`;
  writeFileSync(temp, JSON.stringify(value, null, 2), { mode: 0o600 });
  chmodSync(temp, 0o600);
  renameSync(temp, file);
}

function readJSON(file, fallback) {
  if (!existsSync(file)) return fallback;
  // A damaged credentials file must never silently become a new trusted state.
  return JSON.parse(readFileSync(file, 'utf8'));
}

export async function ensureIdentity(directory) {
  mkdirSync(directory, { recursive: true, mode: 0o700 });
  chmodSync(directory, 0o700);
  const keyFile = join(directory, 'server-key.pem');
  const certFile = join(directory, 'server-cert.pem');
  if (existsSync(keyFile) !== existsSync(certFile)) throw new Error('The bridge TLS identity is incomplete. Restore both identity files or use a new private data folder.');
  if (!existsSync(keyFile)) {
    const tempKey = `${keyFile}.tmp`;
    const tempCert = `${certFile}.tmp`;
    try {
      await run('/usr/bin/openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', tempKey, '-out', tempCert, '-days', '3650', '-sha256', '-subj', '/CN=Strata iPhone Companion', '-addext', 'subjectAltName=DNS:localhost,IP:127.0.0.1'], { timeout: 30_000 });
      chmodSync(tempKey, 0o600); chmodSync(tempCert, 0o600);
      renameSync(tempKey, keyFile); renameSync(tempCert, certFile);
    } finally {
      for (const file of [tempKey, tempCert]) try { unlinkSync(file); } catch {}
    }
  }
  chmodSync(keyFile, 0o600); chmodSync(certFile, 0o600);
  const key = readFileSync(keyFile);
  const cert = readFileSync(certFile);
  const fingerprint = digest(new X509Certificate(cert).raw);
  return { key, cert, fingerprint };
}

export class PairingAuthority {
  constructor(directory, { now = Date.now, ttl = 300_000, maxAttempts = 5 } = {}) {
    this.now = now; this.ttl = ttl; this.maxAttempts = maxAttempts;
    this.file = join(directory, 'sessions.json');
    this.sessions = readJSON(this.file, []);
    if (!Array.isArray(this.sessions)) throw new Error('Invalid bridge sessions file.');
    if (existsSync(this.file)) chmodSync(this.file, 0o600);
    this.rotate();
  }
  rotate() {
    this.code = String(randomInt(0, 1_000_000)).padStart(6, '0');
    this.expiresAt = this.now() + this.ttl;
    this.attempts = 0; this.used = false;
    return { code: this.code, expiresAt: this.expiresAt };
  }
  pair(code, deviceName) {
    if (this.used || this.now() >= this.expiresAt || this.attempts >= this.maxAttempts) throw fail(410, 'Pairing invitation expired. Create a new invitation on your Mac.');
    this.attempts++;
    if (!/^\d{6}$/.test(String(code)) || !same(String(code), this.code)) throw fail(401, 'The pairing code does not match.');
    if (this.sessions.length >= 20) throw fail(409, 'Remove an old paired device on your Mac first.');
    const token = randomBytes(32).toString('base64url');
    const session = { id: id(), hash: digest(token), deviceName: shortText(deviceName, 80) || 'iPhone', createdAt: new Date(this.now()).toISOString() };
    this.sessions.push(session); this.used = true; this.save();
    return { token, session };
  }
  authorize(header) {
    if (typeof header !== 'string' || !/^Bearer [A-Za-z0-9_-]{43}$/.test(header)) return null;
    const hash = digest(header.slice(7));
    return this.sessions.find(session => same(session.hash, hash)) ?? null;
  }
  revoke(sessionId) {
    const before = this.sessions.length;
    this.sessions = this.sessions.filter(session => session.id !== sessionId);
    this.save(); return before !== this.sessions.length;
  }
  save() { privateJSON(this.file, this.sessions); }
  publicSessions() { return this.sessions.map(({ hash, ...rest }) => rest); }
}

export function validateUpstream(value) {
  const url = new URL(value);
  if (url.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) || url.username || url.password || url.pathname !== '/' || url.search || url.hash || !url.port) throw new Error('The Strata host address must be an HTTP loopback origin with an explicit port.');
  return url.origin;
}

/** Discover the host process, never an arbitrary remote server or the authenticated engine port. */
export async function discoverHost() {
  const { stdout } = await run('/usr/sbin/lsof', ['-nP', '-iTCP', '-sTCP:LISTEN', '-Fpcn'], { timeout: 4_000, maxBuffer: 1_000_000 });
  const candidates = [];
  let processId, command;
  for (const line of stdout.split('\n')) {
    if (line.startsWith('p')) processId = line.slice(1);
    if (line.startsWith('c')) command = line.slice(1);
    if (!line.startsWith('n') || !/node|strata/i.test(command ?? '')) continue;
    const match = /^n(?:127\.0\.0\.1|\[::1\]|localhost):(\d+)$/.exec(line);
    if (match) candidates.push({ processId, port: match[1] });
  }
  for (const candidate of candidates) {
    try {
      const [{ stdout: args }, { stdout: cwd }] = await Promise.all([
        run('/bin/ps', ['-p', candidate.processId, '-o', 'command='], { timeout: 2_000 }),
        run('/usr/sbin/lsof', ['-a', '-p', candidate.processId, '-d', 'cwd', '-Fn'], { timeout: 2_000 }),
      ]);
      // No arguments or credentials are ever printed or returned.
      if (!/host\/server\/index\.mjs/.test(args) && !/\/host\s*$/m.test(cwd)) continue;
      const origin = `http://127.0.0.1:${candidate.port}`;
      const response = await fetch(`${origin}/api/setup`, { signal: AbortSignal.timeout(2_000), redirect: 'error' });
      const body = await response.json();
      if (response.ok && typeof body.complete === 'boolean' && body.app?.version) return origin;
    } catch {}
  }
  throw fail(503, 'Open the Strata app on your Mac, then try again.');
}

export class HostClient {
  constructor({ upstream, authorization, fetcher = fetch } = {}) {
    this.fixed = upstream ? validateUpstream(upstream) : null;
    this.cached = null; this.discovery = null; this.authorization = authorization; this.fetcher = fetcher;
  }
  async origin() {
    if (this.fixed || this.cached) return this.fixed ?? this.cached;
    this.discovery ??= discoverHost();
    try { return this.cached = await this.discovery; }
    finally { this.discovery = null; }
  }
  async request(path, { method = 'GET', body, signal, deviceId } = {}) {
    const conversation = /^\/api\/companion\/conversations(?:\/[a-zA-Z0-9_-]{1,120}(?:\/(messages|stop|approval))?)?$/.exec(path);
    const management = path.startsWith('/api/model-management/') ? managementRequest(path, method, body, deviceId) : null;
    // Callers cannot supply full URLs, query strings, arbitrary files, settings or operations.
    if (!/^\/api\/(setup|hardware|monitor|models|native|runtime|serving|model-providers|chat\/stream)$/.test(path)
      && !/^\/api\/models\/[a-zA-Z0-9_-]+\/load$/.test(path)
      && path !== '/api/models/unload'
      && path !== '/api/native/operations'
      && !/^\/api\/native\/instances\/[a-zA-Z0-9_-]+\/unload$/.test(path)
      && !/^\/api\/serving\/requests\/[a-zA-Z0-9_-]+\/cancel$/.test(path)
      && !conversation && !management) throw fail(403, 'Unsupported host control.');
    const mutation = path.endsWith('/load') || path.endsWith('/unload') || path.endsWith('/cancel') || path === '/api/native/operations' || path === '/api/chat/stream';
    if (management) {
      // The paired device comes from the bridge's authenticated session, never a phone field.
    } else if (conversation) {
      const create = path === '/api/companion/conversations' && method === 'POST';
      if (method !== (conversation[1] || create ? 'POST' : 'GET')) throw fail(403, 'Unsupported host control method.');
      if (path === '/api/companion/conversations/sync') throw fail(403, 'Desktop synchronization is unavailable from iPhone.');
    } else if (method !== (mutation ? 'POST' : 'GET')) throw fail(403, 'Unsupported host control method.');
    if (path === '/api/native/operations' && (body?.kind !== 'load' || !safeId(body?.args?.model) || Object.keys(body).some(key => !['kind', 'args'].includes(key)) || Object.keys(body.args).some(key => !['model', 'values'].includes(key)) || Object.keys(body.args.values ?? {}).length)) throw fail(403, 'Only an installed model can be loaded from iPhone.');
    const origin = await this.origin();
    try {
      return await this.fetcher(`${origin}${path}`, { method, redirect: 'error', headers: { 'content-type': 'application/json', ...(this.authorization ? { authorization: this.authorization } : {}), ...(management ? { 'x-strata-companion-device': deviceId } : {}) }, ...(body !== undefined ? { body: JSON.stringify(body) } : {}), signal: signal ?? AbortSignal.timeout(management ? 170_000 : 15_000) });
    } catch (error) {
      if (!this.fixed) this.cached = null;
      if (error.name === 'AbortError') throw error;
      throw fail(503, 'Strata is unavailable. Open the Mac app and try again.');
    }
  }
  async json(path, options) {
    const response = await this.request(path, options);
    let body;
    try { body = await response.json(); } catch { throw fail(502, 'Strata returned an unreadable response.'); }
    if (!response.ok) throw fail(response.status, typeof body.error === 'string' ? body.error : body.error?.message ?? 'Strata could not complete this request.');
    return body;
  }
}

function nativeInstanceForModel(native, model) {
  if (!model) return null;
  const instances = (native?.instances ?? []).filter(instance => [model.id, model.model].includes(instance.model));
  return instances.find(instance => instance.id === native.defaultId && instance.status === 'ready')
    ?? instances.find(instance => instance.status === 'ready')
    ?? instances.find(instance => instance.id === native.defaultId)
    ?? instances[0] ?? null;
}

function nativeCatalogModel(native, modelId) {
  return (native?.catalog?.families ?? []).flatMap(family => family.sizes ?? []).find(model => `native-${model.id}` === modelId) ?? null;
}

export function normalizeModels(library, native) {
  const models = [];
  const installed = native?.installed?.models ?? [];
  for (const family of native?.catalog?.families ?? []) {
    for (const model of family.sizes ?? []) {
      const loaded = nativeInstanceForModel(native, model);
      const ready = model.installed || installed.some(item => item.ready !== false && [model.id, model.model].includes(item.model));
      models.push({ id: `native-${model.id}`, name: `${family.title} ${model.model}`, size: Number.isFinite(model.download_gb) ? `${model.download_gb} GB` : null, state: loaded ? loaded.status === 'ready' ? 'loaded' : loaded.status : ready ? 'installed' : 'available', detail: ready ? model.about ?? null : 'Download on your Mac first', loaded: loaded?.status === 'ready', nativeModel: model.id });
    }
  }
  for (const model of library?.variants ?? []) {
    // Packaged Mac installs offer the native catalog; keep the legacy model library when it has actual local models.
    if (models.length && !['installed', 'loaded'].includes(model.state)) continue;
    models.push({ id: model.id, name: model.label ?? model.name ?? model.id, size: Number.isFinite(model.diskGb) ? `${model.diskGb} GB` : null, state: model.state ?? 'available', detail: model.family ?? null, loaded: library.loaded === model.id || model.state === 'loaded' });
  }
  const defaultInstance = (native?.instances ?? []).find(instance => instance.id === native.defaultId && instance.status === 'ready');
  const defaultModel = models.find(model => model.loaded && model.nativeModel && [model.nativeModel, nativeCatalogModel(native, model.id)?.model].includes(defaultInstance?.model));
  return { models: models.map(({ nativeModel, ...model }) => model), loadedModelId: defaultModel?.id ?? models.find(model => model.loaded)?.id ?? null };
}

// IDs are stable but do not contain a provider URL, API key or raw connection record.
export function providerModels(providers) {
  return (providers?.connections ?? []).flatMap(connection => (connection.models ?? []).map(model => ({
    id: `api-${digest(`${connection.id}\0${model.id}`).slice(0, 32)}`,
    name: model.name ?? model.id, size: null, state: 'installed', detail: `API model via Mac · ${connection.name}`,
    remoteModel: { connectionId: connection.id, modelId: model.id },
  })));
}

function managementText(value, fallback, max = 1_000) {
  const text = shortText(value, max).replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g, '');
  // The host supplies display text. A future host must not accidentally export
  // an installer command, filesystem path, endpoint or credential in that text.
  return !text || /https?:\/\/|file:\/\/|~[\\/]|[a-z]:[\\/]|(?:^|[\s('"=])\/[\w.-]+\/|bearer\s+[a-z0-9_-]{16,}|(?:api[_-]?key|token)\s*[=:]\s*\S+/i.test(text) ? fallback : text;
}

function managementNumber(value) {
  return Number.isFinite(value) && value >= 0 ? value : undefined;
}

function publicDownloadJob(job) {
  if (!job || !safeId(job.id)) throw fail(502, 'Strata returned an unreadable download.');
  const state = ['queued', 'running', 'done', 'error', 'cancelled'].includes(job.state) ? job.state : 'error';
  const result = {
    id: job.id, name: managementText(job.name, 'Model download', 200), kind: 'download', state,
    summary: managementText(job.summary, state === 'done' ? 'Download complete.' : state === 'cancelled' ? 'Download stopped. Completed files are kept.' : state === 'error' ? 'The download could not finish. Check Strata on that computer.' : 'Downloading model files…'),
  };
  if (safeId(job.modelId)) result.modelId = job.modelId;
  if (typeof job.canCancel === 'boolean') result.canCancel = job.canCancel;
  for (const key of ['doneGb', 'totalGb']) if (managementNumber(job[key]) !== undefined) result[key] = job[key];
  if (managementNumber(job.percent) !== undefined) result.percent = Math.min(100, job.percent);
  if (typeof job.error === 'string' && job.error) result.error = managementText(job.error, 'The download could not finish. Check Strata on that computer.');
  return result;
}

function publicManagement(body, { targetId, action, jobId }) {
  if (!targetId) {
    return { targets: (Array.isArray(body?.targets) ? body.targets : []).slice(0, 100).filter(target =>
      target && new RegExp(`^${managementTarget}$`).test(target.id) && ['local', 'remote'].includes(target.kind)
      && (target.id === 'local') === (target.kind === 'local'))
      .map(target => ({ id: target.id, name: managementText(target.name, 'Strata computer', 200), kind: target.kind })),
      ...(typeof body?.reason === 'string' ? { reason: managementText(body.reason, 'Model downloads are unavailable on this Strata build.') } : {}),
    };
  }
  if (action === 'plan') {
    if (!body || !safeId(body.planId) || !Number.isFinite(body.expiresAt)) throw fail(502, 'Strata returned an unreadable download plan.');
    const result = {
      planId: body.planId, targetName: managementText(body.targetName, 'Strata computer', 200), modelName: managementText(body.modelName, 'Catalog model', 200),
      readPictures: body.readPictures === true, expiresAt: body.expiresAt,
      warnings: (Array.isArray(body.warnings) ? body.warnings : []).slice(0, 20).filter(warning => typeof warning === 'string')
        .map(warning => managementText(warning, 'Review this download in Strata on the target computer for more details.')),
    };
    for (const key of ['downloadGb', 'diskFreeGb', 'diskNeededGb']) if (managementNumber(body[key]) !== undefined) result[key] = body[key];
    if (typeof body.diskShort === 'string' && body.diskShort) result.diskShort = managementText(body.diskShort, 'There is not enough free disk space on this computer.');
    return result;
  }
  if (action === 'downloads' || jobId) return { job: publicDownloadJob(body?.job) };
  if (!body || typeof body.supported !== 'boolean') throw fail(502, 'Strata returned unreadable model management information.');
  return {
    supported: body.supported, targetName: managementText(body.targetName, 'Strata computer', 200),
    capabilities: { downloads: body.capabilities?.downloads === true, imports: false },
    models: (Array.isArray(body.models) ? body.models : []).slice(0, 500).filter(model => model && safeId(model.id)).map(model => ({
      id: model.id, name: managementText(model.name, 'Catalog model', 200),
      state: ['available', 'installed', 'loaded', 'imported', 'downloading', 'loading', 'unloading', 'error'].includes(model.state) ? model.state : 'unknown', readPictures: model.readPictures === true,
      ...(managementNumber(model.sizeGb) !== undefined ? { sizeGb: model.sizeGb } : {}),
      ...(typeof model.detail === 'string' ? { detail: managementText(model.detail, 'Available from the target computer’s catalog.') } : {}),
    })),
    jobs: (Array.isArray(body.jobs) ? body.jobs : []).slice(0, 100).map(publicDownloadJob),
    ...(typeof body.reason === 'string' ? { reason: managementText(body.reason, 'Model downloads are unavailable on this computer.') } : {}),
  };
}

function managementFailure(error) {
  const messages = {
    400: 'This model request is no longer valid. Refresh the catalog and try again.',
    401: 'Model management is not enabled for this connection. Check Strata on the Mac.',
    403: 'Model management is not enabled for this connection. Check Strata on the Mac.',
    404: 'This model management option is unavailable. Refresh the catalog and try again.',
    409: 'Finish active work on the target computer, then review a new download plan.',
    410: 'This download plan expired. Review a new plan before starting.',
    507: 'There is not enough free disk space on the target computer.',
  };
  const status = Number.isInteger(error?.status) && error.status >= 400 && error.status < 600 ? error.status : 502;
  return fail(status, messages[status] ?? 'Model management is unavailable. Open Strata on the target computer and try again.');
}

// Share the same human-readable conversation and action cards as the Mac. Raw
// runtime frames, tool arguments and provider configuration are never exported.
export function publicTool(tool) {
  if (!tool || !safeToolId(tool.id)) return null;
  const result = { id: tool.id, name: shortText(tool.name, 200),
    state: ['proposed', 'approved', 'running', 'done', 'denied', 'error'].includes(tool.state) ? tool.state : 'running' };
  for (const key of ['preview', 'effect', 'scope', 'verb', 'target', 'result']) {
    if (typeof tool[key] === 'string') result[key] = shortText(tool[key], key === 'result' ? 32_000 : 8_000);
  }
  for (const key of ['live', 'approved']) if (typeof tool[key] === 'boolean') result[key] = tool[key];
  if (Number.isFinite(tool.durationMs) && tool.durationMs >= 0) result.durationMs = tool.durationMs;
  if (tool.ask && typeof tool.ask === 'object') {
    result.ask = {};
    for (const key of ['why', 'detail']) if (typeof tool.ask[key] === 'string') result.ask[key] = shortText(tool.ask[key], 8_000);
  }
  return result;
}

function publicMessageBlocks(message, tools) {
  if (!Array.isArray(message.blocks)) return undefined;
  const blocks = [], groups = new Map();
  for (const block of message.blocks) {
    if (!block || typeof block !== 'object') continue;
    if (block.kind === 'text' && typeof block.text === 'string' && block.text) {
      blocks.push({ kind: 'text', text: block.text });
    } else if (block.kind === 'group' && safeId(block.id) && typeof block.line === 'string' && block.line.trim()) {
      const clean = { kind: 'group', id: block.id, line: shortText(block.line, 8_000), closed: block.closed === true,
        toolIds: Array.isArray(block.toolIds) ? block.toolIds : [] };
      for (const key of ['added', 'removed']) if (Number.isFinite(block[key]) && block[key] >= 0) clean[key] = block[key];
      // A growing group replaces its earlier snapshot at the same position.
      if (groups.has(clean.id)) blocks[groups.get(clean.id)] = clean;
      else { groups.set(clean.id, blocks.length); blocks.push(clean); }
    }
  }
  // Some older caches contain only part of the ordered answer. In that case
  // keep the complete content fallback rather than hiding or repeating text.
  if (!blocks.length || blocks.filter(block => block.kind === 'text').map(block => block.text).join('') !== message.content) return undefined;
  const allowed = new Set((tools ?? []).map(tool => tool.id)), assigned = new Set();
  for (const block of blocks) if (block.kind === 'group') {
    block.toolIds = block.toolIds.filter(id => {
      if (!allowed.has(id) || assigned.has(id)) return false;
      assigned.add(id); return true;
    });
  }
  return blocks;
}

export function publicConversation(conversation, { summary = false } = {}) {
  if (!conversation || !safeId(conversation.id)) throw fail(502, 'Strata returned an unreadable conversation.');
  const messages = Array.isArray(conversation.messages) ? conversation.messages : [];
  const result = { id: conversation.id, title: shortText(conversation.title, 500) || 'Conversation',
    updatedAt: shortText(conversation.updatedAt, 80), active: conversation.active === true };
  if (typeof conversation.workspace === 'string') result.workspace = shortText(conversation.workspace, 4_000);
  if (typeof conversation.status === 'string') result.status = shortText(conversation.status, 1_000);
  const mode = conversation.preferences?.mode ?? conversation.mode;
  if (typeof mode === 'string') result.mode = shortText(mode, 40);
  const model = conversation.modelName ?? [...messages].reverse().find(message => typeof message.model === 'string')?.model;
  if (typeof model === 'string') result.modelName = shortText(model, 300);
  if (summary) {
    const last = [...messages].reverse().find(message => typeof message.content === 'string' && message.content.trim());
    result.preview = shortText(last?.content, 300);
  } else {
    result.messages = messages.filter(message => message && safeId(message.id) && ['user', 'assistant', 'system'].includes(message.role)).map(message => {
      const clean = { id: message.id, role: message.role, content: typeof message.content === 'string' ? message.content : '' };
      for (const key of ['at', 'error']) if (typeof message[key] === 'string') clean[key] = shortText(message[key], key === 'error' ? 4_000 : 80);
      if (typeof message.streaming === 'boolean') clean.streaming = message.streaming;
      if (Array.isArray(message.tools)) clean.tools = message.tools.map(publicTool).filter(Boolean);
      if (message.role === 'assistant') {
        // This is the same model-supplied Thinking text already visible on the Mac.
        if (typeof message.reasoning === 'string' && message.reasoning.trim()) clean.reasoning = message.reasoning;
        if (Number.isFinite(message.reasoningMs) && message.reasoningMs >= 0) clean.reasoningMs = message.reasoningMs;
        const blocks = publicMessageBlocks({ ...message, content: clean.content }, clean.tools);
        if (blocks) clean.blocks = blocks;
      }
      return clean;
    });
  }
  return result;
}

function publicConversationDetail(body) {
  return { conversation: publicConversation(body.conversation), pending: Array.isArray(body.pending) ? body.pending.map(publicTool).filter(Boolean) : [] };
}

function sendJSON(res, status, value) {
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' });
  res.end(JSON.stringify(value));
}

async function readBody(req, limit = 64_000) {
  if (!String(req.headers['content-type'] ?? '').startsWith('application/json')) throw fail(415, 'Send a JSON request.');
  let size = 0; const buffers = [];
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) throw fail(413, 'The request is too large.');
    buffers.push(chunk);
  }
  try {
    const body = JSON.parse(Buffer.concat(buffers).toString('utf8'));
    if (!body || typeof body !== 'object' || Array.isArray(body)) throw new Error();
    return body;
  } catch { throw fail(400, 'Send a JSON object.'); }
}

function lanAddress() {
  const interfaces = networkInterfaces();
  for (const [name, addresses] of Object.entries(interfaces).sort(([a], [b]) => a === 'en0' ? -1 : b === 'en0' ? 1 : a.localeCompare(b))) {
    for (const address of addresses ?? []) if (address.family === 'IPv4' && !address.internal && !/^utun|^bridge|^vmenet/.test(name)) return address.address;
  }
  return '127.0.0.1';
}

export async function createBridge({ directory = join(homedir(), 'Library', 'Application Support', 'Strata Companion'), host = new HostClient(), listen = '0.0.0.0', port = 8787, adminPort = 8788, advertiseHost = lanAddress(), now = Date.now, pairingTTL = 300_000, qr = true } = {}) {
  if (!/^[a-zA-Z0-9.:-]+$/.test(advertiseHost)) throw new Error('Invalid advertised host.');
  const identity = await ensureIdentity(directory);
  const authority = new PairingAuthority(directory, { now, ttl: pairingTTL });
  const chatFile = join(directory, 'chats.json');
  const chats = readJSON(chatFile, []);
  if (!Array.isArray(chats)) throw new Error('Invalid bridge chats file.');
  if (existsSync(chatFile)) chmodSync(chatFile, 0o600);
  const active = new Map();
  const invalidAuth = new Map();
  const macName = hostname().replace(/\.local$/, '');
  const adminSecret = randomBytes(24).toString('base64url');
  const csrf = randomBytes(24).toString('base64url');
  const adminPath = `/admin/${adminSecret}`;
  const qrFile = join(directory, 'pairing.png');
  let networkPort, localPort, closing = false, closed;
  let qrReady = false;
  const saveChats = () => privateJSON(chatFile, chats);
  const invitation = () => {
    const urlHost = advertiseHost.includes(':') ? `[${advertiseHost}]` : advertiseHost;
    const params = new URLSearchParams({ url: `https://${urlHost}:${networkPort}`, code: authority.code, fingerprint: identity.fingerprint });
    return `strata://pair?${params}`;
  };
  const adminState = () => {
    const available = !authority.used && authority.expiresAt > now() && authority.attempts < authority.maxAttempts;
    const expiresAt = authority.expiresAt;
    const revision = digest(JSON.stringify({ available, expiresAt, devices: authority.publicSessions().map(session => [session.id, session.deviceName, session.createdAt]) }));
    return { available, revision, expiresAt };
  };
  async function updateQR() {
    qrReady = false;
    try { unlinkSync(qrFile); } catch {}
    if (!qr) return;
    try {
      const helper = join(here, 'qr-helper');
      await run(existsSync(helper) ? helper : '/usr/bin/swift', [...(existsSync(helper) ? [] : [join(here, 'qr.swift')]), invitation(), qrFile], { timeout: 30_000, maxBuffer: 20_000 });
      chmodSync(qrFile, 0o600); qrReady = true;
    } catch { /* Manual invitation remains available without the Swift command line tools. */ }
  }
  // Local in-process controls let the desktop app own pairing without opening the
  // standalone admin page. Never return the certificate key, session hashes or tokens.
  function localState() {
    const { available, expiresAt } = adminState();
    return {
      invitation: {
        url: invitation(), code: authority.code, fingerprint: identity.fingerprint,
        address: new URL(invitation()).searchParams.get('url'), expiresAt, available,
        qrAvailable: available && qrReady && !closing,
      },
      devices: authority.publicSessions().map(({ id, deviceName, createdAt }) => ({ id, deviceName, createdAt })),
    };
  }
  async function rotateInvitation() {
    if (closing) throw fail(409, 'Enable iPhone pairing first.');
    authority.rotate(); await updateQR();
    return localState();
  }
  function revokeSession(sessionId) {
    if (closing) throw fail(409, 'Enable iPhone pairing first.');
    if (typeof sessionId !== 'string' || !/^[a-f0-9]{32}$/.test(sessionId)) throw fail(400, 'Choose a paired device.');
    authority.revoke(sessionId);
    for (const request of active.values()) if (request.owner === sessionId) request.abort.abort();
    return localState();
  }
  async function snapshot() {
    const [library, native, monitor, serving, providers] = await Promise.all([
      host.json('/api/models'), host.json('/api/native'), host.json('/api/monitor'), host.json('/api/serving'), host.json('/api/model-providers'),
    ]);
    return { library, native, monitor, serving, providers };
  }
  function choices(session, library, native, providers) {
    const local = normalizeModels(library, native);
    const remote = providerModels(providers);
    // Reuse a loaded local model first. With API-only Strata, start with the first configured API model.
    const selectedId = session.selectedModel ?? local.loadedModelId ?? remote[0]?.id;
    const selectedRemote = remote.find(model => model.id === selectedId);
    const deselected = selectedId === 'none';
    const selectedLocal = local.models.find(model => model.id === selectedId && model.loaded);
    const nativeModel = nativeCatalogModel(native, selectedId);
    const nativeInstance = nativeModel ? nativeInstanceForModel(native, nativeModel) : null;
    const selectedNativeInstance = nativeInstance?.status === 'ready' ? nativeInstance : null;
    return {
      selectedId, selectedRemote, selectedNativeInstance,
      models: [...remote.map(({ remoteModel, ...model }) => ({ ...model, selected: model.id === selectedId, loaded: model.id === selectedId, state: model.id === selectedId ? 'loaded' : 'installed' })), ...local.models.map(model => ({ ...model, selected: model.id === selectedId }))],
      loadedModelId: deselected ? null : selectedRemote?.id ?? selectedLocal?.id ?? null,
    };
  }
  const secureServer = https.createServer({ key: identity.key, cert: identity.cert, minVersion: 'TLSv1.2', requestTimeout: 30_000, headersTimeout: 10_000 }, async (req, res) => {
    res.setHeader('cache-control', 'no-store');
    res.setHeader('x-content-type-options', 'nosniff');
    try {
      // Native clients do not send Origin. The bridge has no browser CORS surface.
      if (req.headers.origin) throw fail(403, 'Use the paired Strata iPhone app.');
      const url = new URL(req.url, 'https://bridge.invalid');
      const path = url.pathname;
      const method = req.method;
      if (url.search || !path.startsWith('/v1/')) throw fail(404, 'Unknown companion endpoint.');
      if (path === '/v1/pair' && method === 'POST') {
        const body = await readBody(req, 1_000);
        if (closing) throw fail(409, 'iPhone pairing is disabled.');
        const result = authority.pair(body.code, body.deviceName);
        return sendJSON(res, 200, { token: result.token, macName });
      }
      const address = req.socket.remoteAddress;
      const failed = invalidAuth.get(address);
      if (failed && failed.until > now() && failed.count >= 30) throw fail(429, 'Too many invalid requests. Try again in a minute.');
      const session = authority.authorize(req.headers.authorization);
      if (!session) {
        if (invalidAuth.size > 1_000) invalidAuth.clear();
        invalidAuth.set(address, { count: failed?.until > now() ? failed.count + 1 : 1, until: now() + 60_000 });
        throw fail(401, 'Pair this iPhone with your Mac first.');
      }
      const requireCurrentSession = () => {
        if (closing) throw fail(409, 'iPhone pairing is disabled.');
        if (!authority.sessions.some(saved => saved.id === session.id)) throw fail(401, 'This device was removed. Pair it again.');
      };
      if (path === '/v1/session' && method === 'DELETE') {
        authority.revoke(session.id);
        for (const request of active.values()) if (request.owner === session.id) request.abort.abort();
        return sendJSON(res, 200, { revoked: true });
      }
      if (path === '/v1/status' && method === 'GET') {
        try {
          const { library, native, monitor, serving, providers } = await snapshot();
          const models = choices(session, library, native, providers);
          const instance = models.selectedNativeInstance;
          return sendJSON(res, 200, { macName, hostConnected: true, modelName: models.models.find(model => model.selected)?.name ?? null, engineState: models.selectedRemote ? 'API via Mac' : instance?.phase ?? monitor.phase ?? 'idle', tokensPerSecond: finite(monitor.outputTps), memoryUsedGB: finite(monitor.hardware?.ramUsedGb), memoryTotalGB: finite(monitor.hardware?.ramTotalGb), activeRequests: (serving.requests ?? []).filter(request => ['queued', 'generating'].includes(request.state)).length });
        } catch {
          return sendJSON(res, 200, { macName, hostConnected: false, modelName: null, engineState: 'offline', tokensPerSecond: null, memoryUsedGB: null, memoryTotalGB: null, activeRequests: 0 });
        }
      }
      if (path === '/v1/models' && method === 'GET') {
        const [library, native, providers] = await Promise.all([host.json('/api/models'), host.json('/api/native'), host.json('/api/model-providers')]);
        const { models, loadedModelId } = choices(session, library, native, providers);
        return sendJSON(res, 200, { models, loadedModelId });
      }
      if (path === '/v1/model-management' || path.startsWith('/v1/model-management/')) {
        const upstreamPath = path.replace(/^\/v1\//, '/api/');
        const input = method === 'POST' ? await readBody(req, 2_000) : undefined;
        const route = managementRequest(upstreamPath, method, input, session.id);
        requireCurrentSession();
        let body;
        const controller = new AbortController();
        const disconnected = () => { if (!res.writableEnded) controller.abort(); };
        req.once('aborted', disconnected); res.once('close', disconnected);
        try {
          body = await host.json(upstreamPath, { method, ...(input !== undefined ? { body: input } : {}), deviceId: session.id, signal: AbortSignal.any([controller.signal, AbortSignal.timeout(170_000)]) });
        } catch (error) {
          requireCurrentSession();
          if (!route.targetId && [404, 501].includes(error.status)) return sendJSON(res, 200, { targets: [], reason: 'Update Strata on your Mac to manage model downloads from your iPhone.' });
          throw managementFailure(error);
        } finally {
          req.removeListener('aborted', disconnected); res.removeListener('close', disconnected);
        }
        requireCurrentSession();
        return sendJSON(res, route.action === 'downloads' && !route.jobId ? 202 : 200, publicManagement(body, route));
      }
      if (path === '/v1/conversations' && method === 'GET') {
        const body = await host.json('/api/companion/conversations');
        requireCurrentSession();
        const conversations = (Array.isArray(body.conversations) ? body.conversations : []).map(conversation => publicConversation(conversation, { summary: true }));
        return sendJSON(res, 200, { conversations, ...(safeId(body.activeChatId) ? { activeChatId: body.activeChatId } : {}) });
      }
      if (path === '/v1/conversations' && method === 'POST') {
        // Only the server's saved model selection can configure a new chat.
        // Continuing a chat always inherits its existing Mac model and permissions.
        const body = await readBody(req, 1_000);
        if (Object.keys(body).length) throw fail(400, 'Create a conversation without changing Mac settings.');
        const [library, native, providers] = await Promise.all([host.json('/api/models'), host.json('/api/native'), host.json('/api/model-providers')]);
        requireCurrentSession();
        const selection = choices(session, library, native, providers);
        if (!selection.loadedModelId) throw fail(409, 'Choose a model in the iPhone Models tab first.');
        const options = { ...(selection.selectedRemote ? { remoteModel: selection.selectedRemote.remoteModel } : {}), ...(selection.selectedNativeInstance ? { nativeModelId: selection.selectedNativeInstance.id } : {}) };
        const created = await host.json('/api/companion/conversations', { method: 'POST', body: options });
        requireCurrentSession();
        return sendJSON(res, 201, publicConversationDetail(created));
      }
      const conversationMatch = /^\/v1\/conversations\/([a-zA-Z0-9_-]{1,120})(?:\/(messages|stop|approval))?$/.exec(path);
      if (conversationMatch) {
        const [, conversationId, action] = conversationMatch;
        if (conversationId === 'sync') throw fail(404, 'Unknown companion endpoint.');
        if ((!action && method !== 'GET') || (action && method !== 'POST')) throw fail(405, 'This conversation action is unavailable.');
        let input;
        if (action) {
          const body = await readBody(req);
          if (action === 'messages') {
            const message = shortText(body.message, 20_001).trim();
            if (!message || message.length > 20_000 || Object.keys(body).some(key => key !== 'message')) throw fail(400, 'Write a message of up to 20,000 characters.');
            input = { message };
          } else if (action === 'approval') {
            if (!safeToolId(body.toolId) || !['approve', 'deny'].includes(body.decision) || Object.keys(body).some(key => !['toolId', 'decision'].includes(key))) throw fail(400, 'Choose Allow or Deny for this action.');
            input = { toolId: body.toolId, decision: body.decision };
          } else {
            if (Object.keys(body).length) throw fail(400, 'Stop this conversation without changing Mac settings.');
            input = {};
          }
        }
        requireCurrentSession();
        const body = await host.json(`/api/companion/conversations/${encode(conversationId)}${action ? `/${action}` : ''}`, { method: action ? 'POST' : 'GET', ...(action ? { body: input } : {}) });
        requireCurrentSession();
        return sendJSON(res, action === 'messages' ? 202 : 200, publicConversationDetail(body));
      }
      if (path === '/v1/chats' && method === 'GET') return sendJSON(res, 200, { chats: chats.filter(chat => chat.owner === session.id).map(({ id, title, updatedAt }) => ({ id, title, updatedAt })) });
      const chatMatch = /^\/v1\/chats\/([a-f0-9]{32})$/.exec(path);
      if (chatMatch && method === 'GET') {
        const chat = chats.find(chat => chat.id === chatMatch[1] && chat.owner === session.id);
        if (!chat) throw fail(404, 'This conversation is unavailable.');
        return sendJSON(res, 200, { id: chat.id, title: chat.title, messages: chat.messages });
      }
      const cancelMatch = /^\/v1\/chat\/([a-f0-9]{32})\/cancel$/.exec(path);
      if (cancelMatch && method === 'POST') {
        requireCurrentSession();
        const request = active.get(cancelMatch[1]);
        if (!request || request.owner !== session.id) throw fail(404, 'This conversation has no active request.');
        request.abort.abort(); return sendJSON(res, 200, { cancelled: true });
      }
      const requestCancel = /^\/v1\/requests\/([a-zA-Z0-9_-]+)\/cancel$/.exec(path);
      if (requestCancel && method === 'POST') {
        requireCurrentSession();
        const request = [...active.values()].find(request => request.requestId === requestCancel[1] && request.owner === session.id);
        if (!request) throw fail(404, 'This request is unavailable.');
        request.abort.abort(); return sendJSON(res, 200, { cancelled: true });
      }
      if (path === '/v1/chat' && method === 'POST') {
        const body = await readBody(req);
        const message = shortText(body.message, 20_001).trim();
        if (!message || message.length > 20_000) throw fail(400, 'Write a message of up to 20,000 characters.');
        const chatId = body.chatId ?? body.sessionId;
        if (chatId !== undefined && !safeId(chatId)) throw fail(400, 'Invalid conversation ID.');
        let chat = chatId ? chats.find(chat => chat.id === chatId && chat.owner === session.id) : null;
        if (chatId && !chat) throw fail(404, 'This conversation is unavailable.');
        const [library, native, providers] = await Promise.all([host.json('/api/models'), host.json('/api/native'), host.json('/api/model-providers')]);
        const selection = choices(session, library, native, providers);
        // Reserve synchronously after the await: parallel requests cannot claim the same conversation.
        requireCurrentSession();
        if (chat && active.has(chat.id)) throw fail(409, 'Wait for this conversation to finish first.');
        if ([...active.values()].filter(request => request.owner === session.id).length >= 3) throw fail(429, 'Finish an active conversation first.');
        if (!selection.loadedModelId) throw fail(409, 'Choose a model in the iPhone Models tab first.');
        if (!chat) {
          if (chats.filter(chat => chat.owner === session.id).length >= 200) throw fail(409, 'This device has reached its conversation limit.');
          chat = { id: id(), owner: session.id, title: message.slice(0, 60), messages: [], updatedAt: new Date(now()).toISOString() };
          chats.unshift(chat);
        }
        const history = chat.messages.filter(message => ['user', 'assistant'].includes(message.role)).slice(-100).map(({ role, content }) => ({ role, content }));
        chat.messages.push({ id: id(), role: 'user', content: message });
        const answer = { id: id(), role: 'assistant', content: '' };
        chat.messages.push(answer); chat.updatedAt = new Date(now()).toISOString(); saveChats();
        const abort = new AbortController();
        let finish;
        const finished = new Promise(resolve => { finish = resolve; });
        const request = { owner: session.id, abort, requestId: null, finished };
        active.set(chat.id, request);
        res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache, no-transform', connection: 'keep-alive', 'x-accel-buffering': 'no' });
        const emit = frame => { if (!res.destroyed) res.write(`event: frame\ndata: ${JSON.stringify({ ...frame, chatId: chat.id, sessionId: chat.id })}\n\n`); };
        emit({ type: 'session' });
        res.on('close', () => { if (!res.writableEnded) abort.abort(); });
        const timer = setTimeout(() => abort.abort(), 15 * 60_000);
        try {
          if (selection.selectedNativeInstance && !safeId(selection.selectedNativeInstance.id)) throw fail(502, 'The selected model has an invalid identifier.');
          const upstream = await host.request('/api/chat/stream', { method: 'POST', body: { message, history, sessionId: `iphone-${chat.id}`, agent: false, tools: false, useTools: false, ...(selection.selectedRemote ? { remoteModel: selection.selectedRemote.remoteModel } : {}), ...(selection.selectedNativeInstance ? { nativeModelId: selection.selectedNativeInstance.id } : {}) }, signal: abort.signal });
          if (!upstream.ok) throw fail(upstream.status, 'Strata could not start the response. Check the Mac app.');
          let buffer = ''; let done = false;
          const decoder = new TextDecoder();
          const consume = line => {
            if (!line.trim()) return;
            const frame = JSON.parse(line);
            if (frame.requestId) request.requestId = String(frame.requestId);
            if (frame.type === 'delta' && typeof frame.text === 'string') answer.content += frame.text;
            if (frame.type === 'done') done = true;
            const clean = { type: frame.type };
            for (const key of ['text', 'message', 'code', 'tps', 'tokensIn', 'tokensOut', 'durationMs', 'requestId', 'model', 'demo', 'phase', 'label', 'thinking']) if (['string', 'number', 'boolean'].includes(typeof frame[key])) clean[key] = frame[key];
            emit(clean);
            if (answer.content.length > 1_000_000) throw new Error('The response reached the companion limit.');
          };
          for await (const chunk of upstream.body) {
            buffer += decoder.decode(chunk, { stream: true });
            if (buffer.length > 1_000_000) throw new Error('Strata returned an oversized frame.');
            let boundary;
            while ((boundary = buffer.indexOf('\n')) >= 0) {
              const line = buffer.slice(0, boundary).trim(); buffer = buffer.slice(boundary + 1);
              // Only data useful to a text client crosses the LAN, never paths or tool payloads.
              consume(line);
            }
          }
          buffer += decoder.decode();
          consume(buffer);
          if (!done) emit({ type: 'error', message: 'The response ended early. Try again.' });
        } catch (error) {
          emit({ type: 'error', message: abort.signal.aborted ? 'Response stopped.' : error.status ? error.message : 'Strata could not finish this response. Check the Mac app.' });
        } finally {
          clearTimeout(timer); abort.abort();
          active.delete(chat.id);
          try { saveChats(); }
          finally { res.end(); finish(); }
        }
        return;
      }
      const loadMatch = /^\/v1\/models\/([a-zA-Z0-9_-]+)\/load$/.exec(path);
      if ((path === '/v1/models/load' || loadMatch) && method === 'POST') {
        const body = await readBody(req, 1_000);
        const modelId = loadMatch?.[1] ?? body.id;
        if (!safeId(modelId)) throw fail(400, 'Choose an installed model.');
        const [library, native, providers] = await Promise.all([host.json('/api/models'), host.json('/api/native'), host.json('/api/model-providers')]);
        requireCurrentSession();
        const model = choices(session, library, native, providers).models.find(model => model.id === modelId);
        if (!model || !['installed', 'loaded'].includes(model.state)) throw fail(409, 'Download this model in the Mac app first.');
        if (modelId.startsWith('api-')) {
          session.selectedModel = modelId; authority.save();
          return sendJSON(res, 200, { accepted: true });
        }
        if (modelId.startsWith('native-')) {
          const existing = nativeInstanceForModel(native, nativeCatalogModel(native, modelId));
          if (existing?.status === 'ready') {
            // Select an already running instance privately; leave the desktop default alone.
            session.selectedModel = modelId; authority.save();
            return sendJSON(res, 200, { accepted: true });
          }
          const job = await host.json('/api/native/operations', { method: 'POST', body: { kind: 'load', args: { model: modelId.slice(7), values: {} } } });
          requireCurrentSession();
          session.selectedModel = modelId; authority.save();
          return sendJSON(res, 202, { accepted: true, operationId: job.id ?? null });
        }
        await host.json(`/api/models/${encode(modelId)}/load`, { method: 'POST', body: {} });
        requireCurrentSession();
        session.selectedModel = modelId; authority.save();
        return sendJSON(res, 200, { accepted: true });
      }
      if (path === '/v1/models/unload' && method === 'POST') {
        await readBody(req, 1_000);
        const [library, native, providers] = await Promise.all([host.json('/api/models'), host.json('/api/native'), host.json('/api/model-providers')]);
        requireCurrentSession();
        const selection = choices(session, library, native, providers);
        if (selection.selectedRemote) {
          // Selecting a remote model never starts or stops the Mac's local engine.
          session.selectedModel = 'none'; authority.save();
          return sendJSON(res, 200, { accepted: true });
        }
        if (!selection.loadedModelId) return sendJSON(res, 200, { accepted: true });
        if (selection.selectedNativeInstance) {
          const instance = selection.selectedNativeInstance;
          if (!safeId(instance.id)) throw fail(502, 'The loaded model has an invalid identifier.');
          await host.json(`/api/native/instances/${encode(instance.id)}/unload`, { method: 'POST', body: {} });
        } else await host.json('/api/models/unload', { method: 'POST', body: {} });
        requireCurrentSession();
        session.selectedModel = 'none'; authority.save();
        return sendJSON(res, 200, { accepted: true });
      }
      throw fail(404, 'Unknown companion endpoint.');
    } catch (error) {
      if (!res.headersSent) sendJSON(res, error.status ?? 500, { error: error.status ? error.message : 'The companion could not complete this request.' });
      else res.end();
    }
  });
  const adminServer = http.createServer({ requestTimeout: 10_000 }, async (req, res) => {
    try {
      const expectedHost = `127.0.0.1:${localPort}`;
      if (req.socket.remoteAddress !== '127.0.0.1' || req.headers.host !== expectedHost || (req.headers.origin && req.headers.origin !== `http://${expectedHost}`)) throw fail(403, 'Open pairing on this Mac.');
      const url = new URL(req.url, `http://${expectedHost}`);
      res.setHeader('cache-control', 'no-store'); res.setHeader('x-frame-options', 'DENY'); res.setHeader('referrer-policy', 'no-referrer');
      res.setHeader('content-security-policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'");
      if (url.pathname === `${adminPath}/state` && req.method === 'GET') return sendJSON(res, 200, adminState());
      if (url.pathname === `${adminPath}/qr.png` && req.method === 'GET' && qrReady) { res.writeHead(200, { 'content-type': 'image/png' }); res.end(readFileSync(qrFile)); return; }
      if (req.method === 'POST' && [adminPath + '/rotate', adminPath + '/revoke'].includes(url.pathname)) {
        if (!same(req.headers['x-strata-csrf'], csrf)) throw fail(403, 'Reload the pairing page.');
        const body = await readBody(req, 1_000);
        if (url.pathname.endsWith('/rotate')) await rotateInvitation();
        else revokeSession(body.id);
        return sendJSON(res, 200, { ok: true });
      }
      if (url.pathname !== adminPath || req.method !== 'GET') throw fail(404, 'Pairing page unavailable.');
      const escape = value => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
      const { available, revision } = adminState();
      const deviceRows = authority.publicSessions().map(session => `<li><span>${escape(session.deviceName)}</span><button onclick="act('revoke',{id:'${escape(session.id)}'})">Remove</button></li>`).join('');
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
      res.end(`<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><title>Strata · Pair iPhone</title><style>:root{color-scheme:dark}body{font:16px system-ui;background:#0e1113;color:#eef1f3;max-width:670px;margin:50px auto;padding:24px}h1{font-size:35px;letter-spacing:-1px}.accent{color:#34d399}.card{background:#161a1d;border:1px solid #2a3036;border-radius:24px;padding:28px;margin:24px 0}img{display:block;width:260px;height:260px;image-rendering:pixelated;background:white;border:18px solid white;border-radius:14px;margin:20px auto}code{font-size:34px;letter-spacing:8px;display:block;text-align:center}button{background:#34d399;color:#052e20;border:0;border-radius:12px;padding:12px 20px;font-weight:600;cursor:pointer}li{display:flex;justify-content:space-between;align-items:center;padding:12px 0}ul{padding:0}p{line-height:1.55;color:#8e979f}label{display:block;font-size:13px;color:#8e979f;margin-top:16px}.manual-row{display:flex;gap:10px;align-items:center}.manual-row input{flex:1;min-width:0;font:13px ui-monospace,monospace;background:#0e1113;color:#eef1f3;border:1px solid #2a3036;border-radius:12px;padding:12px}.manual-row button{padding:11px 14px}summary{cursor:pointer;color:#34d399;margin:18px 0}textarea{width:100%;height:90px;background:#0e1113;color:#34d399;border:1px solid #2a3036;border-radius:12px;margin:14px 0;padding:10px;box-sizing:border-box}small{overflow-wrap:anywhere;color:#8e979f}</style><h1>Strata <span class="accent">for iPhone</span></h1><p>Keep Strata open. Connect your iPhone and Mac to the same trusted Wi-Fi network.</p><section class="card"><h2>Pair your iPhone</h2>${available ? `${qrReady ? `<img src="${adminPath}/qr.png" alt="Scan this invitation in Strata on iPhone">` : ''}<code>${authority.code}</code><p>Open Strata on your iPhone and scan this code, or paste the invitation below. It expires at ${escape(new Date(authority.expiresAt).toLocaleTimeString())} and works once.</p><textarea id="invitation" readonly aria-label="Pairing invitation">${escape(invitation())}</textarea><button onclick="copyField('invitation',this)">Copy invitation</button><details><summary>Enter connection details manually</summary><label for="mac-address">HTTPS address</label><div class="manual-row"><input id="mac-address" readonly value="${escape(new URL(invitation()).searchParams.get('url'))}"><button onclick="copyField('mac-address',this)">Copy</button></div><label for="mac-fingerprint">Certificate fingerprint (SHA-256)</label><div class="manual-row"><input id="mac-fingerprint" readonly value="${identity.fingerprint}"><button onclick="copyField('mac-fingerprint',this)">Copy</button></div><label for="pairing-code">Pairing code</label><div class="manual-row"><input id="pairing-code" readonly value="${authority.code}"><button onclick="copyField('pairing-code',this)">Copy</button></div></details>` : '<p>This invitation is used or expired. Create another to pair a new device.</p>'}<button onclick="act('rotate',{})">New invitation</button></section><section class="card"><h2>Paired devices</h2><ul>${deviceRows || '<p>No paired devices yet.</p>'}</ul></section><p>Your phone uses encrypted HTTPS with this Mac’s pinned certificate. Removing a device immediately revokes access. Paired devices can read and continue Mac conversations and answer pending tool approvals. Earlier iPhone-only chats remain separate.</p><small>Certificate SHA-256: ${identity.fingerprint}</small><script>const shownRevision='${revision}';function reloadKeepingScroll(){try{sessionStorage.setItem('strata.pairing.scroll',String(scrollY));}catch{}location.reload();}try{const saved=sessionStorage.getItem('strata.pairing.scroll');if(saved!==null){sessionStorage.removeItem('strata.pairing.scroll');requestAnimationFrame(()=>scrollTo(0,Number(saved)||0));}}catch{}async function copyField(id,button){const field=document.getElementById(id);try{await navigator.clipboard.writeText(field.value);}catch{field.focus();field.select();document.execCommand('copy');}button.textContent='Copied';setTimeout(()=>button.textContent=id==='invitation'?'Copy invitation':'Copy',1500);}let polling=false;setInterval(async()=>{if(polling)return;polling=true;try{const r=await fetch(location.pathname+'/state',{cache:'no-store'});if(r.ok&&(await r.json()).revision!==shownRevision)reloadKeepingScroll();}catch{}finally{polling=false;}},3000);async function act(action,body){const r=await fetch(location.pathname+'/'+action,{method:'POST',headers:{'content-type':'application/json','x-strata-csrf':'${csrf}'},body:JSON.stringify(body)});if(r.ok)reloadKeepingScroll();else alert('Could not update pairing. Reload and try again.');}</script>`);
    } catch (error) { sendJSON(res, error.status ?? 500, { error: error.status ? error.message : 'Pairing page unavailable.' }); }
  });
  await new Promise((resolve, reject) => { secureServer.once('error', reject); secureServer.listen(port, listen, resolve); });
  networkPort = secureServer.address().port;
  try { await new Promise((resolve, reject) => { adminServer.once('error', reject); adminServer.listen(adminPort, '127.0.0.1', resolve); }); }
  catch (error) { secureServer.close(); throw error; }
  localPort = adminServer.address().port;
  await updateQR();
  return {
    authority, identity, secureServer, adminServer,
    port: networkPort, adminPort: localPort,
    adminURL: `http://127.0.0.1:${localPort}${adminPath}`,
    invitation,
    state: localState,
    rotateInvitation,
    revokeSession,
    qrImage: () => qrReady && adminState().available && !closing ? readFileSync(qrFile) : null,
    async close() {
      if (closing) return closed;
      closing = true;
      const running = [...active.values()];
      for (const request of running) request.abort.abort();
      // Disabling access also ends held requests; a client cannot delay shutdown
      // by leaving a request body or TLS connection unfinished.
      const listeners = [secureServer, adminServer].map(server => new Promise(resolve => {
        server.close(resolve);
        server.closeAllConnections();
      }));
      closed = Promise.all([...listeners, ...running.map(request => request.finished)]);
      await closed;
    },
  };
}
