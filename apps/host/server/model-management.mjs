/** Explicit, bounded model administration. Inference credentials alone never enable it. */
import { createHash, randomUUID } from 'node:crypto';
import { json, error, readBody } from './util.mjs';
import { isLocalCompanionRequest } from './companion-routes.mjs';

const fail = (status, message) => Object.assign(new Error(message), { status });
const text = value => typeof value === 'string' ? value.replace(/[\x00-\x1f\x7f]/g, ' ').slice(0, 1000) : '';
const number = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined;
const id = value => typeof value === 'string' && /^[a-zA-Z0-9_-]{1,160}$/.test(value);
function exact(body, keys) {
  if (!body || Array.isArray(body) || typeof body !== 'object' || Object.keys(body).some(key => !keys.includes(key))) throw fail(400, 'Choose a supported model action.');
}
export const managementPath = path => /^\/v1\/strata\/models(?:\/(?:plan|downloads(?:\/[a-zA-Z0-9_-]+\/cancel)?|imports(?:\/(?:browse|inspect))?))?$/.test(path);
export const remoteTargetId = connection => `remote-${createHash('sha256').update(connection.id).digest('hex').slice(0,24)}`;

export function createModelManagement({ native, name, modelsDirectory, browse, inspect, imported = () => [], adopt, assertIdle = async () => {}, validateCaller = () => {}, now = Date.now }) {
  const plans = new Map(), jobOwners = new Map(), jobReceipts = new Map();
  const targetName = () => text(typeof name === 'function' ? name() : name) || 'This computer';
  function catalog() { return (native.state().catalog?.families ?? []).flatMap(family => (family.sizes ?? []).map(size => ({ family, size }))); }
  function entry(modelId) { return catalog().find(({ size }) => `native-${size.id}` === modelId); }
  function remember(owner, data) {
    for (const [key, value] of plans) if (value.expiresAt <= now()) plans.delete(key);
    if (plans.size >= 100) throw fail(429, 'Too many model previews. Close an old preview and try again shortly.');
    const planId = randomUUID(), expiresAt = now() + 5 * 60_000;
    plans.set(planId, { ...data, owner, expiresAt });
    return { planId, expiresAt };
  }
  function consume(owner, planId, kind) {
    if (!id(planId)) throw fail(400, 'Review this model again before continuing.');
    const plan = plans.get(planId);
    if (!plan || plan.owner !== owner || plan.kind !== kind || plan.expiresAt <= now()) throw fail(409, 'This preview expired or was already used. Review the model again.');
    plans.delete(planId);
    return plan;
  }
  function args(row, readPictures) { return { family: row.family.family, model: row.size.model, vision: readPictures ? 'yes' : 'no', loadAfter: false, managedDownload: true }; }
  function signature(row, plan) { return JSON.stringify({ family: row.family.family, size: row.size, title: plan.title, downloadGb: plan.download_gb, neededGb: plan.disk_needed_gb, context: plan.context, images: plan.images }); }
  function jobView(job, owner) {
    const progress = job.result?.install?.download;
    const row = catalog().find(({ family, size }) => size.model === job.model && (!job.family || family.family === job.family));
    return { id: job.id, kind: 'download', modelId: row ? `native-${row.size.id}` : undefined,
      name: row ? text(`${row.family.title} ${row.size.model}`) : 'Model download',
      state: ['running','done','error','cancelled'].includes(job.state) ? job.state : 'error',
      summary: job.state === 'error' ? 'Download could not finish. Completed files are kept; review and retry.' : job.state === 'cancelled' ? 'Stopped. Completed files are kept.' : job.state === 'done' ? 'Model files are ready. Load the model on the destination computer to use it.' : 'Preparing model and engine files…',
      canCancel: !owner?.startsWith('phone:') || jobOwners.get(job.id) === owner,
      requestId: jobReceipts.get(job.id),
      percent: number(progress?.percent) === undefined ? undefined : Math.min(100, progress.percent),
      doneGb: number(progress?.done_gb), totalGb: number(progress?.total_gb),
      ...(job.state === 'error' ? { error: 'Check the destination computer’s model diagnostics, then retry.' } : {}) };
  }
  async function snapshot(owner) {
    validateCaller(owner);
    const state = await native.refresh(); validateCaller(owner);
    const installed = state.installed?.models ?? [];
    return { supported: true, targetName: targetName(), capabilities: { downloads: state.available === true, imports: true },
      reason: state.available ? undefined : 'Model downloads need a working Strata installation on this computer.',
      models: [...catalog().map(({ family, size }) => ({ id: `native-${size.id}`, name: text(`${family.title} ${size.model}`),
        sizeGb: number(size.download_gb), state: (state.instances ?? []).some(instance => [size.id,size.model].includes(instance.model) && instance.status === 'ready') ? 'loaded' : size.installed || installed.some(model => model.ready !== false && [size.id,size.model].includes(model.model)) ? 'installed' : 'available',
        detail: text(size.about), readPictures: size.images === true })),
        ...imported().map(model => ({ id: model.id, name: text(model.name), sizeGb: number(model.diskGb), state: 'imported', detail: 'Existing GGUF files on this computer', readPictures: false }))],
      jobs: (state.jobs ?? []).filter(job => job.kind === 'install').map(job => jobView(job, owner)) };
  }
  async function downloadPlan(owner, body) {
    exact(body, ['modelId','readPictures']);
    if (!id(body.modelId) || typeof body.readPictures !== 'boolean') throw fail(400, 'Choose a model and its picture-reading option.');
    validateCaller(owner); await native.refresh(); validateCaller(owner);
    const row = entry(body.modelId);
    if (!row) throw fail(404, 'Choose a model from this computer’s catalog.');
    if (body.readPictures && !row.size.images) throw fail(400, 'This model does not support picture reading.');
    const result = await native.plan(args(row, body.readPictures)); validateCaller(owner);
    if (!result?.plan) throw fail(502, 'The destination could not prepare a download preview.');
    const plan = result.plan;
    return { ...remember(owner, { kind: 'download', modelId: body.modelId, readPictures: body.readPictures, signature: signature(row, plan) }), targetName: targetName(), modelName: text(plan.title) || text(`${row.family.title} ${row.size.model}`),
      downloadGb: number(plan.download_gb), diskFreeGb: number(plan.disk_free_gb), diskNeededGb: number(plan.disk_needed_gb),
      diskShort: plan.disk_short ? 'There is not enough free disk space on this computer. Free space there and review again.' : undefined,
      readPictures: body.readPictures, warnings: ['Files are stored on this computer, not on the device controlling it.',
        'Preparing files unloads this computer’s loaded native models. Active work must finish first.',
        'Strata may also download or update its matching engine. The downloaded model is not automatically loaded.',
        ...(plan.partly_downloaded ? ['Completed files are checked and reused.'] : [])] };
  }
  async function start(owner, body, { signal } = {}) {
    signal?.throwIfAborted();
    exact(body, ['planId']); validateCaller(owner);
    const saved = consume(owner, body.planId, 'download');
    await assertIdle(); signal?.throwIfAborted(); validateCaller(owner);
    await native.refresh(true); signal?.throwIfAborted(); validateCaller(owner);
    const row = entry(saved.modelId);
    if (!row || saved.readPictures && !row.size.images) throw fail(409, 'The model catalog changed. Review this model again.');
    const options = args(row, saved.readPictures);
    const current = await native.plan(options); signal?.throwIfAborted(); validateCaller(owner);
    if (!current?.plan || current.plan.disk_short) throw fail(409, 'The model no longer fits on this computer’s disk. Free space and review again.');
    if (signature(row, current.plan) !== saved.signature) throw fail(409, 'The model or download requirements changed. Review the updated preview before continuing.');
    await assertIdle(); signal?.throwIfAborted(); validateCaller(owner);
    const job = native.operation('install', options);
    jobOwners.set(job.id, owner);
    jobReceipts.set(job.id, body.planId);
    return { job: jobView(job, owner) };
  }
  async function cancel(owner, jobId, body) {
    exact(body, []); validateCaller(owner);
    if (!id(jobId)) throw fail(400, 'Choose a model download.');
    const job = native.state().jobs.find(job => job.id === jobId && job.kind === 'install');
    if (!job || job.state !== 'running') throw fail(409, 'This download is no longer running.');
    if (owner.startsWith('phone:') && jobOwners.get(jobId) !== owner) throw fail(403, 'Stop this download from the computer that started it.');
    await native.cancel(jobId); validateCaller(owner);
    return { job: jobView(native.state().jobs.find(job => job.id === jobId), owner) };
  }
  async function browseFiles(owner, body) {
    exact(body, ['directory']); validateCaller(owner);
    if (body.directory !== undefined && (typeof body.directory !== 'string' || body.directory.length > 4096)) throw fail(400, 'Choose a folder on this computer.');
    return browse(body.directory ?? modelsDirectory());
  }
  async function importPlan(owner, body) {
    exact(body, ['path']); validateCaller(owner);
    if (typeof body.path !== 'string' || body.path.length > 4096) throw fail(400, 'Choose a GGUF file on this computer.');
    const preview = inspect(body.path);
    return { ...remember(owner, { kind: 'import', path: body.path, signature: JSON.stringify(preview.variant.local) }), targetName: targetName(),
      modelName: text(preview.variant.name), sizeGb: number(preview.variant.diskGb), files: preview.files.map(text), warnings: [...preview.warnings,
        'These files are already on the destination computer. Importing adds a reference; it does not upload, copy or load the model.'] };
  }
  async function importModel(owner, body) {
    exact(body, ['planId']); validateCaller(owner);
    const saved = consume(owner, body.planId, 'import');
    const preview = inspect(saved.path);
    if (JSON.stringify(preview.variant.local) !== saved.signature) throw fail(409, 'These model files changed. Inspect them again before importing.');
    validateCaller(owner); await adopt(preview); validateCaller(owner);
    return { model: { id: preview.variant.id, name: text(preview.variant.name) } };
  }
  return { snapshot, downloadPlan, start, cancel, browseFiles, importPlan, importModel };
}

/** A fixed administration subtree on a saved custom connection; no arbitrary proxy URL. */
export function createModelManagementProxy({ providers, fetcher = fetch }) {
  const plans = new Map(), jobs = new Map(), starts = new Map();
  const connections = () => providers.withSecrets().filter(connection => connection.kind === 'custom');
  function targets() { return { targets: [{ id: 'local', name: 'This Mac', kind: 'local' }, ...connections().map(connection => ({ id: remoteTargetId(connection), name: connection.name, kind: 'remote', connectionId: connection.id }))] }; }
  async function call(targetId, suffix, method, body, owner = 'desktop', { signal } = {}) {
    const connection = connections().find(connection => remoteTargetId(connection) === targetId);
    if (!connection) throw fail(404, 'This Strata connection is no longer available.');
    const path = `/v1/strata/models${suffix}`;
    if (!managementPath(path)) throw fail(403, 'Unsupported remote model action.');
    if (!connection.apiKey) {
      if (method === 'GET') return unsupported('Set an API key on the destination Strata computer and save it in this connection.');
      throw fail(403, 'Set an API key on the destination Strata computer and save it in this connection.');
    }
    const base = connection.protocol === 'anthropic' ? connection.baseUrl.replace(/\/v1\/?$/, '') + '/v1' : connection.baseUrl.replace(/\/$/, '');
    const destination = createHash('sha256').update(`${base}\0${connection.apiKey}`).digest('hex');
    if (method === 'POST' && ['/downloads','/imports'].includes(suffix)) {
      exact(body, ['planId']);
      const plan = plans.get(body.planId);
      if (!plan || plan.owner !== owner || plan.targetId !== targetId || plan.destination !== destination || plan.expiresAt <= Date.now() || plan.kind !== suffix) throw fail(409, 'This preview expired. Review the model again.');
      plans.delete(body.planId); body = { planId: plan.remoteId };
      if (suffix === '/downloads') starts.set(`${targetId}:${plan.remoteId}`, { owner, destination });
    }
    const cancelId = /^\/downloads\/([a-zA-Z0-9_-]+)\/cancel$/.exec(suffix)?.[1];
    const owns = jobId => { const receipt = jobs.get(`${targetId}:${jobId}`); return receipt?.owner === owner && receipt.destination === destination; };
    if (cancelId && owner.startsWith('phone:') && !owns(cancelId)) throw fail(403, 'Stop this download from the computer that started it.');
    let response;
    try {
      response = await fetcher(`${base}/strata/models${suffix}`, { method, headers: { 'content-type': 'application/json', authorization: `Bearer ${connection.apiKey}` }, redirect: 'error', signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(150_000)]) : AbortSignal.timeout(150_000), ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
    } catch { throw fail(503, 'Could not reach this Strata computer. Check its connection and try again.'); }
    const saved = providers.withSecrets().find(saved => saved.id === connection.id);
    if (saved?.baseUrl !== connection.baseUrl || saved?.apiKey !== connection.apiKey || saved?.protocol !== connection.protocol) { await response.body?.cancel(); throw fail(409, 'This connection changed. Reopen model management.'); }
    if ([403,404,405,501].includes(response.status) && method === 'GET') { await response.body?.cancel(); return unsupported('Update Strata on this computer, then enable Serving → Allow remote model management.'); }
    if (!response.ok) { await response.body?.cancel(); throw fail(response.status === 401 ? 401 : response.status === 409 ? 409 : 400, response.status === 401 ? 'This Strata computer rejected the API key.' : response.status === 409 ? 'The destination is busy or this preview expired. Refresh and review again.' : 'The destination could not finish this model action. Check its model status and try again.'); }
    const raw = await response.text();
    if (raw.length > 2 * 1024 * 1024) throw fail(502, 'The destination sent an oversized model response.');
    let result; try { result = JSON.parse(raw); } catch { throw fail(502, 'This server did not return a Strata model response.'); }
    if (method === 'GET' && (result.supported !== true || !Array.isArray(result.models) || !result.capabilities)) return unsupported('This API server does not support Strata model management.');
    if (method === 'POST' && ['/plan','/imports/inspect'].includes(suffix)) {
      if (!id(result.planId) || !Number.isFinite(result.expiresAt)) throw fail(502, 'The destination did not return a valid model preview.');
      for (const [key, value] of plans) if (value.expiresAt <= Date.now()) plans.delete(key);
      if (plans.size >= 100) throw fail(429, 'Close an old model preview and try again shortly.');
      const planId = randomUUID();
      plans.set(planId, { owner, targetId, destination, remoteId: result.planId, expiresAt: result.expiresAt, kind: suffix === '/plan' ? '/downloads' : '/imports' });
      result.planId = planId;
    }
    if (suffix === '/downloads' && result.job?.id) jobs.set(`${targetId}:${result.job.id}`, { owner, destination });
    if (Array.isArray(result.jobs)) result.jobs = result.jobs.map(job => {
      const receipt = starts.get(`${targetId}:${job.requestId}`);
      if (receipt?.owner === owner && receipt.destination === destination) jobs.set(`${targetId}:${job.id}`, receipt);
      return { ...job, canCancel: !owner.startsWith('phone:') || owns(job.id) };
    });
    if (result.job && owner.startsWith('phone:')) result.job.canCancel = owns(result.job.id);
    return { ...result, targetName: connection.name };
  }
  return { targets, call };
}
function unsupported(reason) { return { supported: false, reason, capabilities: { downloads: false, imports: false }, models: [], jobs: [] }; }

export function createModelManagementRoutes({ service, proxy, port, allowRemote, companionAllowed }) {
  async function dispatch(owner, suffix, method, body, context) {
    if (suffix === '' && method === 'GET') return service.snapshot(owner);
    if (suffix === '/plan' && method === 'POST') return service.downloadPlan(owner, body);
    if (suffix === '/downloads' && method === 'POST') return service.start(owner, body, context);
    const cancel = /^\/downloads\/([a-zA-Z0-9_-]+)\/cancel$/.exec(suffix);
    if (cancel && method === 'POST') return service.cancel(owner, cancel[1], body);
    if (suffix === '/imports/browse' && method === 'POST') return service.browseFiles(owner, body);
    if (suffix === '/imports/inspect' && method === 'POST') return service.importPlan(owner, body);
    if (suffix === '/imports' && method === 'POST') return service.importModel(owner, body);
    throw fail(404, 'Unknown model management action.');
  }
  return async function routes(req, res, path, method, remote = false) {
    res.removeHeader('access-control-allow-origin'); res.setHeader('cache-control', 'no-store');
    const controller = new AbortController();
    const disconnected = () => { if (!res.writableEnded) controller.abort(new Error('The model management request ended.')); };
    req.once('aborted', disconnected); res.once('close', disconnected);
    try {
      let targetId = 'local', suffix, owner = 'desktop';
      if (remote) {
        if (!allowRemote()) throw fail(403, 'Enable remote model management in Strata’s Serving screen first.');
        if (req.headers.origin !== undefined) throw fail(403, 'Remote model administration is unavailable to browser origins.');
        if (!managementPath(path)) throw fail(404, 'Unknown model management action.');
        suffix = path.slice('/v1/strata/models'.length);
        owner = `remote:${createHash('sha256').update(String(req.headers.authorization?.replace(/^Bearer /, '') ?? req.headers['x-api-key'])).digest('hex')}`;
      } else {
        if (!isLocalCompanionRequest(req, typeof port === 'function' ? port() : port)) throw fail(403, 'Open Strata on this computer to manage models.');
        const device = req.headers['x-strata-companion-device'];
        if (device !== undefined) {
          if (typeof device !== 'string' || !/^[a-f0-9]{32}$/.test(device) || !companionAllowed(device)) throw fail(401, 'This phone is no longer paired.');
          owner = `phone:${device}`;
        }
        if (path === '/api/model-management/targets' && method === 'GET') return json(res, 200, proxy.targets());
        const match = /^\/api\/model-management\/targets\/(local|remote-[a-f0-9]{24})(.*)$/.exec(path);
        if (!match || !managementPath(`/v1/strata/models${match[2]}`)) throw fail(404, 'Unknown model management target.');
        [, targetId, suffix] = match;
        if (owner.startsWith('phone:') && suffix.startsWith('/imports')) throw fail(403, 'Import files from Strata on your computer.');
      }
      const body = method === 'POST' ? await readBody(req, 8 * 1024) : undefined;
      if (owner.startsWith('phone:') && !companionAllowed(owner.slice(6))) throw fail(401, 'This phone is no longer paired.');
      controller.signal.throwIfAborted();
      const context = { signal: controller.signal };
      const result = targetId === 'local' ? await dispatch(owner, suffix, method, body, context) : await proxy.call(targetId, suffix, method, body, owner, context);
      if (owner.startsWith('phone:') && !companionAllowed(owner.slice(6))) throw fail(401, 'This phone is no longer paired.');
      return json(res, 200, result);
    } catch (cause) { if (!res.destroyed) return error(res, cause.status ?? 400, cause.status ? cause.message : 'This model action could not finish. Check the destination computer and try again.'); }
    finally { req.removeListener('aborted', disconnected); res.removeListener('close', disconnected); }
  };
}
