/** Additive desktop bridge. Official frontend and APIs remain upstream-owned. */
import { readFileSync, lstatSync, realpathSync } from 'node:fs';
import { resolve, join, sep, extname } from 'node:path';
import { createHash } from 'node:crypto';
import { createOwnerApi } from './owner_api.mjs';

export const PREFIX = '/__strata_extension/';
export const BRIDGE_API = 1;
const types = { '.html':'text/html; charset=utf-8', '.mjs':'text/javascript; charset=utf-8',
  '.js':'text/javascript; charset=utf-8', '.css':'text/css; charset=utf-8', '.svg':'image/svg+xml',
  '.png':'image/png', '.json':'application/json', '.woff2':'font/woff2' };
const sha = data => createHash('sha256').update(data).digest('hex');
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const assetName = value => typeof value === 'string' && /^ui\/[A-Za-z0-9_./-]+$/.test(value)
  && !value.split('/').some(p => !p || p === '.' || p === '..') && !!types[extname(value)];
function bytes(path, maximum = 65536) {
  const stat = lstatSync(path);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size > maximum) throw new Error('Invalid File');
  return readFileSync(path);
}
const json = path => JSON.parse(bytes(path).toString('utf8').replace(/^\uFEFF/, ''));
function directory(path) {
  const stat = lstatSync(path);
  if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error('Invalid Directory');
  return realpathSync(path);
}
function publicState(plan) {
  return { enabled:plan.enabled, compatible:plan.compatible, reason:plan.reason, bridgeApi:BRIDGE_API,
    appVersion:plan.appVersion, extension:plan.extension ?? null, preferences:{uiCopyStyle:'title_case'} };
}

export function loadExtension({ extensionRoot, appVersion, nodeVersion = process.versions.node }) {
  const plan = { enabled:false, compatible:false, reason:'not_installed', appVersion:appVersion ?? null };
  try {
    const root = directory(resolve(extensionRoot)), state = json(join(root,'active.json'));
    if (state?.schema !== 1 || typeof state.enabled !== 'boolean') throw new Error('Invalid State');
    plan.enabled = state.enabled;
    if (!state.enabled) return {...plan, compatible:true, reason:'disabled'};
    const active = state.active;
    if (!object(active) || !/^\d+\.\d+\.\d+-[a-f0-9]{12}$/.test(active.release ?? '')
      || !/^[a-f0-9]{64}$/.test(active.sha256 ?? '') || !active.release.endsWith(active.sha256.slice(0,12))) throw new Error('Invalid Release');
    const release = directory(join(directory(join(root,'releases')), active.release));
    const raw = bytes(join(release,'manifest.json'));
    if (sha(raw) !== active.sha256) throw new Error('Changed Manifest');
    const manifest = JSON.parse(raw.toString('utf8'));
    if (manifest.schema !== 1 || manifest.id !== 'strata-local-models' || !/^\d+\.\d+\.\d+$/.test(manifest.version ?? '')
      || !active.release.startsWith(manifest.version+'-')) throw new Error('Invalid Manifest');
    plan.extension = {id:manifest.id, version:manifest.version, release:active.release};
    if (manifest.bridgeApi !== BRIDGE_API) return {...plan, reason:'unsupported_bridge_api'};
    if(manifest.features!==undefined)throw new Error('Unsupported Feature Contract');
    const version = /^(\d+)\.(\d+)\.\d+(?:-[a-zA-Z0-9.-]+)?$/.exec(appVersion ?? '');
    if (!version || !object(manifest.app) || Number(version[1]) !== manifest.app.major
      || Number(version[2]) !== manifest.app.minor || Number(nodeVersion.split('.')[0]) < 24)
      return {...plan, reason:'unsupported_app_version'};
    if (!Array.isArray(manifest.files) || !manifest.files.length || manifest.files.length > 32
      || !assetName(manifest.entry) || !assetName(manifest.panel)) throw new Error('Invalid Inventory');
    const assets = new Map();
    let total = 0;
    for (const item of manifest.files) {
      if (!object(item) || !assetName(item.path) || assets.has(item.path)
        || !Number.isSafeInteger(item.bytes) || item.bytes < 0 || item.bytes > 2097152
        || !/^[a-f0-9]{64}$/.test(item.sha256 ?? '')) throw new Error('Invalid Asset');
      const parts = item.path.split('/');
      for (let index=1; index<parts.length; index++) directory(join(release,...parts.slice(0,index)));
      const path = join(release,...parts);
      if (!realpathSync(path).startsWith(release+sep)) throw new Error('Asset Outside Release');
      const data = bytes(path,2097152);
      total += data.length;
      if (total > 8388608 || data.length !== item.bytes || sha(data) !== item.sha256) throw new Error('Changed Asset');
      assets.set(item.path,{data,type:types[extname(item.path)]});
    }
    if (!assets.has(manifest.entry) || !assets.has(manifest.panel)) throw new Error('Missing Entry');
    return {...plan,compatible:true,reason:'ready',manifest,assets,hash:active.sha256};
  } catch { return {...plan,compatible:false,reason:'invalid_package'}; }
}

function localRequest(req) {
  if(!/^(?:127\.|::ffff:127\.|::1$)/.test(req.socket?.remoteAddress??''))return false;
  const host = req.headers.host;
  if (typeof host !== 'string' || !/^(?:127\.0\.0\.1|localhost)(?::[1-9][0-9]{0,4})?$/i.test(host)) return false;
  return req.headers.origin === undefined || req.headers.origin === new URL('http://'+host).origin;
}
function send(res, code, data) {
  res.writeHead(code,{'content-type':'application/json; charset=utf-8','cache-control':'no-store'});
  res.end(JSON.stringify(data));
}

export function createExtensionBridge({extensionRoot,appVersion,officialWebRoot,ownerDirectory,modelLabels,apiKeyEnv,
  baseUrl='http://127.0.0.1:8080/v1',fetchImpl=fetch,port=null}={}) {
  const api = createOwnerApi({ownerDirectory,baseUrl,modelLabels,apiKeyEnv,fetchImpl});
  const getPlan = () => {
    const plan = loadExtension({extensionRoot,appVersion});
    if (plan.enabled && plan.compatible) {
      try {
        plan.html = bytes(join(officialWebRoot,'index.html'),2097152).toString('utf8');
        if (!/<\/head\s*>/i.test(plan.html) || /http-equiv\s*=\s*["']?content-security-policy/i.test(plan.html)) throw new Error();
      } catch { return {...plan,compatible:false,reason:'unsupported_host_layout'}; }
    }
    return plan;
  };
  const aliases = new Map([['model','/api/local-model'],['context','/api/local-model/context'],
    ['idle','/api/local-model/idle'],['load','/api/local-model/load'],['unload','/api/local-model/unload'],
    ['connect','/api/local-model/connection']]);
  async function handle(req,res) {
    if (port !== null && req.socket.localPort !== port) return false;
    let url;
    try { url = new URL(req.url ?? '/', 'http://'+(req.headers.host ?? '127.0.0.1')); } catch { return false; }
    const own = url.pathname.startsWith(PREFIX);
    const home = ['/','/index.html'].includes(url.pathname) && ['GET','HEAD'].includes(req.method) && !req.headers.range;
    if (!own && !home) return false;
    if (!localRequest(req)) {
      if (own) { send(res,403,{error:'A Same-Origin Local App Request Is Required.'}); return true; }
      return false;
    }
    const plan = getPlan();
    if (url.pathname === PREFIX+'api/status') {
      if (req.method !== 'GET') send(res,405,{error:'Use GET For Extension Status.'});
      else send(res,200,publicState(plan));
      return true;
    }
    if (!plan.enabled || !plan.compatible) {
      if (own) { send(res,503,{error:'The Local Model Extension Is Unavailable.',...publicState(plan)}); return true; }
      return false;
    }
    if (home) {
      const tag = `<script type="module" data-strata-local-models-extension src="${PREFIX}${plan.manifest.entry}?v=${plan.hash}"></script>`;
      const output = Buffer.from(plan.html.replace(/<\/head\s*>/i,tag+'\n</head>'));
      res.writeHead(200,{'content-type':'text/html; charset=utf-8','cache-control':'no-store','content-length':output.length});
      res.end(req.method === 'HEAD' ? undefined : output); return true;
    }
    if (url.pathname.startsWith(PREFIX+'api/')) {
      const target = aliases.get(url.pathname.slice((PREFIX+'api/').length));
      if (target) return api.handle(req,res,new URL(target,url.origin));
      send(res,404,{error:'Unknown Extension API.'}); return true;
    }
    const name = url.pathname.slice(PREFIX.length);
    if (!['GET','HEAD'].includes(req.method)) send(res,405,{error:'Use GET For Extension Assets.'});
    else if (!plan.assets.has(name)) send(res,404,{error:'Extension Asset Not Found.'});
    else {
      const asset = plan.assets.get(name);
      res.writeHead(200,{'content-type':asset.type,'cache-control':'no-store','content-length':asset.data.length,'x-content-type-options':'nosniff'});
      res.end(req.method === 'HEAD' ? undefined : asset.data);
    }
    return true;
  }
  return {handle,status:()=>publicState(getPlan())};
}
