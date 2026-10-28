/**
 * One way out to the internet for the host and for the agent's web-search plugin.
 *
 * Why this file exists: Node's own fetch ignores the proxy this computer is set up to use, so on a PC that reaches the internet
 * through a proxy (Clash, v2ray, a company proxy, an antivirus that scans HTTPS) every download, update check, model call and web
 * search went straight out and failed, or worse, quietly went out without the proxy the person expected. This module answers one
 * question for every outgoing request - "through which door do I leave?" - and then uses that door:
 *
 *   1. this computer's own services (127.0.0.1, localhost, ::1) never go through a proxy and never will;
 *   2. the person's choice in Settings, Tools, Network: follow the system (the default), direct, or one address they typed;
 *   3. following the system: HTTPS_PROXY / HTTP_PROXY / ALL_PROXY (with NO_PROXY), then the Windows registry
 *      (ProxyEnable / ProxyServer / AutoConfigURL / ProxyOverride), then a direct connection.
 *
 * The registry is read once per 30 seconds, not once per request. A proxy that is set but does not answer is an ERROR, never a
 * quiet fallback to a direct connection: the person is told which proxy failed and can choose "Direct" in Settings.
 *
 * HTTP client: a direct request uses Node's own fetch (unchanged behaviour). A proxied request uses node:http through the proxy -
 * absolute form for http:// targets, a CONNECT tunnel plus TLS for https:// targets - because Node ships no fetch proxy agent.
 * The Response it builds is an ordinary Response: status, headers, text(), json(), and a streaming body all work, so a streamed
 * model answer still streams. Only http/https/http(s)-proxy are supported; a SOCKS address is refused with a plain message
 * instead of silently going direct.
 */
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, renameSync, statSync, writeFileSync } from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import { isIP } from 'node:net';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';
import { Readable } from 'node:stream';
import tls from 'node:tls';
import { makePrivate } from './private-file.mjs';

/** How long a registry reading is reused. */
export const DETECT_TTL_MS = 30_000;
const HANDSHAKE_TIMEOUT_MS = 20_000;
const MAX_REDIRECTS = 20;
const REDIRECTS = new Set([301, 302, 303, 307, 308]);
const DEFAULT_POLICY = Object.freeze({ mode: 'system', url: '' });

/** The error the Settings screen turns into one plain sentence (src/lib/net-errors.ts). */
export class NetworkFailure extends Error {
  constructor(message, code, extra = {}) {
    super(message, extra.cause ? { cause: extra.cause } : undefined);
    this.name = 'NetworkError';
    this.code = code;
    Object.assign(this, extra);
  }
}

/* ------------------------------------------------------------------ hosts --- */

/** This computer's own services: never proxied (a proxy in front of them would break local models and the app itself). */
export function isLoopbackHost(hostname) {
  const host = String(hostname ?? '').trim().toLowerCase().replace(/^\[|\]$/g, '');
  if (!host) return false;
  return host === 'localhost' || host.endsWith('.localhost') || host === '::1' || /^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(host);
}

/** A private address on this computer's own network: never sent to a proxy, even one typed in Settings. */
export function isPrivateHost(hostname) {
  const host = String(hostname ?? '').trim().toLowerCase().replace(/^\[|\]$/g, '');
  if (!host) return false;
  if (host === 'host.docker.internal' || host.endsWith('.local') || host.endsWith('.lan') || host.endsWith('.internal')) return true;
  const v4 = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(host);
  if (v4) {
    const a = Number(v4[1]), b = Number(v4[2]);
    if (a === 10 || a === 127) return true;
    if (a === 169 && b === 254) return true;
    if (a === 172 && b >= 16 && b <= 31) return true;
    if (a === 192 && b === 168) return true;
    return false;
  }
  return /^f[cd][0-9a-f]{2}:/.test(host) || /^fe80:/.test(host);
}

/* ------------------------------------------------------------------ proxy --- */

/**
 * One proxy address as the rest of this file needs it: { kind, host, port, auth, url }. `kind` is http, https (the proxy itself speaks
 * TLS) or socks. A bare `127.0.0.1:7890` means http://, the way a browser treats it.
 */
export function parseProxy(value) {
  const raw = String(value ?? '').trim();
  if (!raw) return null;
  const withScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(raw) ? raw : `http://${raw}`;
  let url;
  try { url = new URL(withScheme); } catch { throw new NetworkFailure(`"${raw}" is not a proxy address. Use an address like http://127.0.0.1:7890.`, 'PROXY_INVALID'); }
  const scheme = url.protocol.replace(':', '').toLowerCase();
  if (!['http', 'https', 'socks', 'socks5', 'socks5h'].includes(scheme)) {
    throw new NetworkFailure(`Strata does not support "${scheme}" proxies. Use an HTTP proxy address like http://127.0.0.1:7890.`, 'PROXY_UNSUPPORTED');
  }
  if (!url.hostname) throw new NetworkFailure(`"${raw}" has no proxy address in it. Use an address like http://127.0.0.1:7890.`, 'PROXY_INVALID');
  const kind = scheme.startsWith('socks') ? 'socks' : scheme;
  const port = Number(url.port || (kind === 'https' ? 443 : kind === 'http' ? 80 : 1080));
  const login = url.username ? decodeURIComponent(url.username) : '';
  const password = url.password ? decodeURIComponent(url.password) : '';
  const auth = login ? `Basic ${Buffer.from(`${login}:${password}`).toString('base64')}` : '';
  return { kind, host: url.hostname.replace(/^\[|\]$/g, ''), port, auth, url: `${kind === 'socks' ? 'socks5' : kind}://${url.hostname}${url.port ? `:${url.port}` : ''}` };
}

const envValue = (env, keys) => {
  for (const key of keys) {
    const value = env?.[key];
    if (typeof value === 'string' && value.trim()) return { value: value.trim(), name: key };
  }
  return null;
};

/** The proxy named by the environment for this target, in the order the person's tools expect it. */
export function proxyFromEnv(url, env = process.env) {
  const keys = url.protocol === 'https:'
    ? ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy']
    : ['HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy', 'HTTPS_PROXY', 'https_proxy'];
  const found = envValue(env, keys);
  return found ? { value: found.value, name: found.name } : null;
}

/** NO_PROXY as tools write it: comma or space separated, `*`, `.example.com`, `example.com:8443`, `127.*`, `<local>`. */
export function matchesBypassList(list, hostname, port = '') {
  const host = String(hostname ?? '').toLowerCase().replace(/^\[|\]$/g, '');
  if (!host) return false;
  for (const raw of String(list ?? '').split(/[;,\s]+/)) {
    let entry = raw.trim().toLowerCase();
    if (!entry) continue;
    if (entry === '*') return true;
    if (entry === '<local>') { if (!host.includes('.')) return true; continue; }
    entry = entry.replace(/^[a-z][a-z0-9+.-]*:\/\//, '');
    const slash = entry.indexOf('/');
    if (slash >= 0) entry = entry.slice(0, slash);
    let entryPort = '';
    const colon = entry.lastIndexOf(':');
    if (colon > 0 && /^\d+$/.test(entry.slice(colon + 1))) { entryPort = entry.slice(colon + 1); entry = entry.slice(0, colon); }
    if (entryPort && String(port) && entryPort !== String(port)) continue;
    const bare = entry.replace(/^\[|\]$/g, '');
    if (!bare) continue;
    if (bare === host) return true;
    if (bare.startsWith('*.')) { if (host.endsWith(bare.slice(1))) return true; continue; }
    if (bare.startsWith('.')) { if (host === bare.slice(1) || host.endsWith(bare)) return true; continue; }
    if (bare.endsWith('*')) { if (host.startsWith(bare.slice(0, -1))) return true; continue; }
    if (host.endsWith(`.${bare}`)) return true;      // a bare domain also covers its subdomains, as in curl
  }
  return false;
}

/** "http=127.0.0.1:7890;https=127.0.0.1:7890;socks=127.0.0.1:7891" or a single "127.0.0.1:7890". */
export function registryProxyServer(value, protocol) {
  const parts = String(value ?? '').split(';').map(part => part.trim()).filter(Boolean);
  if (!parts.length) return '';
  if (parts.length === 1 && !parts[0].includes('=')) return parts[0];
  const table = new Map();
  for (const part of parts) {
    const at = part.indexOf('=');
    if (at > 0) table.set(part.slice(0, at).trim().toLowerCase(), part.slice(at + 1).trim());
  }
  const order = protocol === 'https:' ? ['https', 'http', 'socks', 'socks5'] : ['http', 'https', 'socks', 'socks5'];
  for (const key of order) if (table.get(key)) return table.get(key);
  return '';
}

/** `reg query` output -> { ProxyEnable: '0x1', ProxyServer: '...' }. */
export function parseRegQuery(output) {
  const values = {};
  for (const line of String(output ?? '').split(/\r?\n/)) {
    const match = /^\s{2,}(.+?)\s{2,}(REG_[A-Z_]+)\s{2,}(.*)$/.exec(line);
    if (match) values[match[1].trim()] = match[3].trim();
  }
  return values;
}

const REGISTRY_KEY = 'Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings';

/**
 * What Windows itself is set to. One `reg query` per key (HKCU first, then HKLM), no PowerShell window, 2 s at most.
 * Returns null when this is not Windows or the registry cannot be read.
 */
export function readWindowsProxy({ platform = process.platform, run = defaultRegQuery } = {}) {
  if (platform !== 'win32') return null;
  // One budget for both keys: a wedged `reg.exe` must not hold a request for four seconds.
  const until = Date.now() + 1200;
  for (const hive of ['HKCU', 'HKLM']) {
    const left = Math.max(200, until - Date.now());
    const values = run('reg.exe', ['query', `${hive}\\${REGISTRY_KEY}`], { timeout: left });
    if (!values) continue;
    const enabled = /^(0x1|1|true)$/i.test(values.ProxyEnable ?? '');
    const server = values.ProxyServer ?? '';
    const pac = values.AutoConfigURL ?? '';
    const override = values.ProxyOverride ?? '';
    if (enabled || pac) return { enabled, server, pac, override, source: hive === 'HKCU' ? 'windows-user' : 'windows-machine' };
  }
  return null;
}

function defaultRegQuery(command, args, { timeout = 2000 } = {}) {
  try {
    const result = spawnSync(command, args, { encoding: 'utf8', windowsHide: true, timeout });
    if (result.error || result.status !== 0 || !result.stdout) return null;
    return parseRegQuery(result.stdout);
  } catch { return null; }
}

/* --------------------------------------------------------------- settings --- */

/** Where the person's choice is kept. Same data folder as the rest of the host's state. */
export function defaultSettingsFile(env = process.env, home = homedir()) {
  if (typeof env?.STRATA_NET_FILE === 'string' && env.STRATA_NET_FILE.trim()) return env.STRATA_NET_FILE;
  return join(env?.STRATA_HOME || join(home, '.strata'), 'data', 'net.json');
}

/** The saved choice, checked: an unknown mode is "follow the system", and a custom address must be usable. */
export function normalizeNetworkPolicy(raw) {
  const value = raw && typeof raw === 'object' ? raw : {};
  const mode = ['system', 'direct', 'custom'].includes(value.mode) ? value.mode : 'system';
  const url = typeof value.url === 'string' ? value.url.trim().slice(0, 500) : '';
  if (mode !== 'custom') return { mode, url: '' };
  if (!url) throw new NetworkFailure('Enter the proxy address, for example http://127.0.0.1:7890.', 'PROXY_INVALID');
  const proxy = parseProxy(url);
  if (proxy.kind === 'socks') throw new NetworkFailure('Strata needs an HTTP proxy address; a SOCKS address cannot be used here. Use the HTTP address of your proxy app, for example http://127.0.0.1:7890.', 'PROXY_UNSUPPORTED');
  return { mode, url };      // kept as typed: a proxy's user name and password live in it
}

let settingsFile = defaultSettingsFile();
let policyCache = { file: '', mtime: -1, policy: DEFAULT_POLICY };

/** The saved choice, re-read when the file changes (the agent's plugin process sees the same file). */
export function networkPolicy({ file = settingsFile } = {}) {
  try {
    const mtime = statSync(file).mtimeMs;
    if (policyCache.file === file && policyCache.mtime === mtime) return policyCache.policy;
    const policy = normalizeNetworkPolicy(JSON.parse(readFileSync(file, 'utf8')));
    policyCache = { file, mtime, policy };
    return policy;
  } catch { return DEFAULT_POLICY; }
}

/** Save the choice and apply it to this process at once. The file can hold a proxy's user name and password, so it is private. */
export function saveNetworkPolicy(raw, { file = settingsFile } = {}) {
  const policy = normalizeNetworkPolicy(raw);
  mkdirSync(dirname(file), { recursive: true });
  const temp = `${file}.${process.pid}.tmp`;
  writeFileSync(temp, `${JSON.stringify({ version: 1, ...policy }, null, 2)}\n`, { encoding: 'utf8', mode: 0o600 });
  makePrivate(temp);
  renameSync(temp, file);
  makePrivate(file);
  policyCache = { file, mtime: statSync(file).mtimeMs, policy };
  return policy;
}

/** Point the module at one folder (the host calls this with its own data folder). */
export function configureNetwork({ file = settingsFile } = {}) {
  settingsFile = file;
  policyCache = { file: '', mtime: -1, policy: DEFAULT_POLICY };
  registryCache = { at: 0, value: null, read: false };
  return networkPolicy();
}

/* --------------------------------------------------------------- detection --- */

let registryCache = { at: 0, value: null, read: false };

function windowsProxy({ platform, now, registry }) {
  if (typeof registry === 'function') {                    // a test or the caller supplies the reader; it is cached the same way
    if (registryCache.read && now - registryCache.at < DETECT_TTL_MS) return registryCache.value;
    const value = registry({ platform });
    registryCache = { at: now, value, read: true };
    return value;
  }
  if (registry !== undefined) return registry;             // a plain answer
  if (platform !== 'win32') return null;
  if (registryCache.read && now - registryCache.at < DETECT_TTL_MS) return registryCache.value;
  const value = readWindowsProxy({ platform });
  registryCache = { at: now, value, read: true };
  return value;
}

/**
 * Which door this request leaves through. Pure apart from the saved choice and the cached registry reading, so the answer can be
 * shown in Settings as "what Strata is doing right now".
 *
 * @returns {{mode:'direct'|'proxy', source:string, proxy:object|null, pac:string|null, note:string}}
 */
export function detectProxy(target, { env = process.env, platform = process.platform, now = Date.now(), registry } = {}) {
  const url = target instanceof URL ? target : new URL(String(target));
  if (isLoopbackHost(url.hostname)) return { mode: 'direct', source: 'this-computer', proxy: null, pac: null, note: 'This address is on this computer, so no proxy is used for it.' };

  const policy = networkPolicy();
  if (policy.mode === 'direct') return { mode: 'direct', source: 'settings', proxy: null, pac: null, note: 'Settings is set to a direct connection.' };
  if (policy.mode === 'custom') {
    // A model or service on this network is this computer's own business: it never goes through the proxy, so a key or a
    // prompt meant for a machine in the next room is not read by the proxy (even one typed in Settings).
    if (isPrivateHost(url.hostname)) return { mode: 'direct', source: 'this-network', proxy: null, pac: null, note: 'This address is on your own network, so no proxy is used for it.' };
    return { mode: 'proxy', source: 'settings', proxy: parseProxy(policy.url), pac: null, note: 'Using the proxy saved in Settings.' };
  }

  const fromEnv = proxyFromEnv(url, env);
  if (fromEnv) {
    if (matchesBypassList(env.NO_PROXY || env.no_proxy, url.hostname, url.port)) return { mode: 'direct', source: 'no-proxy', proxy: null, pac: null, note: `NO_PROXY lists ${url.hostname}, so no proxy is used for it.`, noteCode: 'NO_PROXY lists {0}, so no proxy is used for it.', noteValues: [url.hostname] };
    return { mode: 'proxy', source: 'environment', proxy: parseProxy(fromEnv.value), pac: null, note: `Using the proxy in ${fromEnv.name}.`, noteCode: 'Using the proxy in {0}.', noteValues: [fromEnv.name] };
  }

  const windows = windowsProxy({ platform, now, registry });
  if (windows?.enabled && windows.server) {
    const address = registryProxyServer(windows.server, url.protocol);
    if (!address) return { mode: 'direct', source: 'none', proxy: null, pac: null, note: 'Windows has a proxy set for other kinds of addresses, but none for this one.' };
    if (matchesBypassList(windows.override, url.hostname, url.port) || matchesBypassList(env.NO_PROXY || env.no_proxy, url.hostname, url.port)) {
      return { mode: 'direct', source: 'no-proxy', proxy: null, pac: null, note: 'Windows lists this address to be reached without the proxy.' };
    }
    return { mode: 'proxy', source: 'windows', proxy: parseProxy(address), pac: null, note: 'Using the proxy Windows is set to use.' };
  }
  if (windows?.pac) {
    return { mode: 'direct', source: 'pac', proxy: null, pac: windows.pac, note: 'Windows is set to choose a proxy with a script. Strata cannot run that script, so it uses a direct connection. Choose the proxy address in Settings, Tools, Network to use it.' };
  }
  return { mode: 'direct', source: 'none', proxy: null, pac: null, note: 'No proxy is set on this computer.' };
}

/** What the Settings screen shows: the saved choice, what the system says, and what a request would do now. */
export function networkView({ env = process.env, platform = process.platform, now = Date.now(), registry } = {}) {
  const policy = networkPolicy();
  const probe = new URL('https://api.github.com/');
  const windows = windowsProxy({ platform, now, registry });
  const fromEnv = proxyFromEnv(probe, env);
  let effective;
  try { effective = detectProxy(probe, { env, platform, now, registry }); }
  catch (error) { effective = { mode: 'direct', source: 'error', proxy: null, pac: null, note: error.message }; }
  let systemProxy = null;
  try {
    if (fromEnv) systemProxy = parseProxy(fromEnv.value)?.url ?? null;
    else if (windows?.enabled && windows.server) systemProxy = parseProxy(registryProxyServer(windows.server, 'https:'))?.url ?? null;
  } catch { systemProxy = null; }
  return {
    policy,
    system: {
      source: fromEnv ? 'environment' : windows ? 'windows' : 'none',
      proxy: systemProxy,
      pac: windows?.pac || null,
      note: fromEnv ? `The ${fromEnv.name} setting names this proxy.` : windows?.pac ? 'Windows uses a proxy script (PAC).' : '',
    },
    effective: { mode: effective.mode, proxy: effective.proxy?.url ?? null, source: effective.source, note: effective.note, noteCode: effective.noteCode, noteValues: effective.noteValues },
  };
}

/**
 * The proxy a program the assistant runs should use, written the way curl, git and Python read it. The app's own requests go through
 * netFetch; a shell command is a separate process, and without these it would go straight out on a computer that expects the proxy.
 * Only a proxy this computer or the person named is exported. A Direct choice adds nothing: Strata does not take a variable away from
 * the person's own shell.
 */
export function toolProxyEnvironment(base = process.env, { platform = process.platform, now = Date.now(), registry } = {}) {
  const loopback = ['localhost', '127.0.0.1', '::1'];
  const withLocal = value => {
    const list = String(value ?? '').split(',').map(part => part.trim()).filter(Boolean);
    for (const host of loopback) if (!list.includes(host)) list.push(host);
    return list.join(',');
  };
  const scheme = address => /^[a-z][a-z0-9+.-]*:\/\//i.test(address) ? address : `http://${address}`;
  const policy = networkPolicy();
  if (policy.mode === 'direct') return {};
  if (policy.mode === 'custom') {
    const url = scheme(policy.url);
    return { HTTP_PROXY: url, HTTPS_PROXY: url, NO_PROXY: withLocal(base.NO_PROXY || base.no_proxy) };
  }
  if (proxyFromEnv(new URL('https://example.invalid/'), base)) return { NO_PROXY: withLocal(base.NO_PROXY || base.no_proxy) };
  const windows = windowsProxy({ platform, now, registry });
  if (!windows?.enabled || !windows.server) return {};
  const vars = {};
  const https = registryProxyServer(windows.server, 'https:');
  const plain = registryProxyServer(windows.server, 'http:');
  if (https) vars.HTTPS_PROXY = scheme(https);
  if (plain) vars.HTTP_PROXY = scheme(plain);
  if (!https && !plain) return {};
  vars.NO_PROXY = withLocal(base.NO_PROXY || base.no_proxy || windows.override || '');
  return vars;
}

/* ------------------------------------------------------------------ fetch --- */
const abortReason = signal => signal?.reason instanceof Error ? signal.reason : Object.assign(new Error('The request was cancelled.'), { name: 'AbortError', code: 'ABORT_ERR' });

function toURL(input) {
  if (input instanceof URL) return input;
  if (typeof input === 'string') return new URL(input);
  if (input && typeof input.url === 'string') return new URL(input.url);
  throw new TypeError('netFetch needs a web address.');
}

function bodyBytes(body) {
  if (body === undefined || body === null) return null;
  if (typeof body === 'string') return Buffer.from(body, 'utf8');
  if (Buffer.isBuffer(body)) return body;
  if (body instanceof Uint8Array) return Buffer.from(body);
  if (body instanceof URLSearchParams) return Buffer.from(body.toString(), 'utf8');
  throw new NetworkFailure('This kind of request body cannot be sent through a proxy.', 'BODY_UNSUPPORTED');
}

function headersToObject(headers) {
  const out = {};
  for (const [name, value] of headers) out[name] = value;
  return out;
}

function responseFrom(message, url) {
  const headers = new Headers();
  for (const [name, value] of Object.entries(message.headers ?? {})) {
    if (Array.isArray(value)) for (const item of value) headers.append(name, item);
    else if (value !== undefined) headers.set(name, String(value));
  }
  const empty = message.statusCode === 204 || message.statusCode === 205 || message.statusCode === 304;
  const response = new Response(empty ? null : Readable.toWeb(message), { status: message.statusCode ?? 502, statusText: message.statusMessage ?? '', headers });
  Object.defineProperty(response, 'url', { value: url.href, configurable: true });
  return response;
}

/**
 * A request straight to the origin through Node's own fetch. `redirect: 'manual'` because the caller's redirect rule is applied one
 * hop at a time in `netFetch`, where every hop is checked against the loopback rule and the proxy again.
 */
async function directFetch(url, { method, headers, body, signal }) {
  const init = { method, headers: headersToObject(headers), signal, redirect: 'manual' };
  if (body) init.body = body;
  return globalThis.fetch(url.href, init);
}

/** Open the CONNECT tunnel a proxied https request needs. */
function openTunnel(proxy, authority, signal) {
  return new Promise((resolve, reject) => {
    const transport = proxy.kind === 'https' ? https : http;
    const request = transport.request({
      host: proxy.host,
      port: proxy.port,
      method: 'CONNECT',
      path: authority,
      headers: { host: authority, ...(proxy.auth ? { 'proxy-authorization': proxy.auth } : {}) },
    });
    let settled = false;
    const done = (error, socket) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener?.('abort', onAbort);
      if (error) { request.destroy(); reject(error); } else resolve(socket);
    };
    const timer = setTimeout(() => done(new NetworkFailure(`The proxy at ${proxy.url} did not answer within ${Math.round(HANDSHAKE_TIMEOUT_MS / 1000)} seconds. Start your proxy app, or choose a direct connection in Settings, Tools, Network.`, 'PROXY_TIMEOUT', { proxy: proxy.url })), HANDSHAKE_TIMEOUT_MS);
    const onAbort = () => done(abortReason(signal));
    if (signal?.aborted) return done(abortReason(signal));
    signal?.addEventListener?.('abort', onAbort, { once: true });
    request.once('error', error => done(new NetworkFailure(`Strata could not reach the proxy at ${proxy.url} (${error.code ?? error.message}). Start your proxy app, or choose a direct connection in Settings, Tools, Network.`, 'PROXY_UNREACHABLE', { cause: error, proxy: proxy.url })));
    request.once('connect', (response, socket) => {
      if (response.statusCode === 200) return done(null, socket);
      socket.destroy();
      if (response.statusCode === 407) return done(new NetworkFailure(`The proxy at ${proxy.url} needs a sign-in and refused this request (407). Put your proxy user name and password in the proxy address, for example http://user:password@127.0.0.1:7890.`, 'PROXY_AUTH', { proxy: proxy.url }));
      done(new NetworkFailure(`The proxy at ${proxy.url} refused to reach ${authority} (HTTP ${response.statusCode}).`, 'PROXY_REFUSED', { proxy: proxy.url }));
    });
    request.end();
  });
}

/** TLS to the origin over an open tunnel. A certificate error keeps its own code so Settings can name its cause. */
function tlsOverTunnel(socket, hostname, signal) {
  return new Promise((resolve, reject) => {
    const options = { socket, ALPNProtocols: ['http/1.1'], rejectUnauthorized: true };
    if (!isIP(hostname)) options.servername = hostname;          // SNI must not carry an IP address
    const secure = tls.connect(options);
    let settled = false;
    const finish = error => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener?.('abort', onAbort);
      if (error) { secure.destroy(); reject(error); } else resolve(secure);
    };
    const timer = setTimeout(() => finish(new NetworkFailure(`The secure connection to ${hostname} did not finish in time.`, 'TLS_TIMEOUT', { cause: new Error('The TLS handshake timed out.') })), HANDSHAKE_TIMEOUT_MS);
    const onAbort = () => finish(abortReason(signal));
    if (signal?.aborted) return finish(abortReason(signal));
    signal?.addEventListener?.('abort', onAbort, { once: true });
    secure.once('error', finish);
    secure.once('secureConnect', () => finish());
  });
}

/** One hop through the proxy. http targets use the proxy's absolute form; https targets get a CONNECT tunnel. */
async function proxiedFetch(url, { method, headers, body, signal }, proxy) {
  if (proxy.kind === 'socks') {
    throw new NetworkFailure('Strata cannot use a SOCKS proxy. Use the HTTP address of your proxy app, for example http://127.0.0.1:7890, in Settings, Tools, Network.', 'PROXY_UNSUPPORTED', { proxy: proxy.url });
  }
  const secure = url.protocol === 'https:';
  const port = Number(url.port || (secure ? 443 : 80));
  const authority = `${url.hostname}:${port}`;
  const socket = secure ? await tlsOverTunnel(await openTunnel(proxy, authority, signal), url.hostname, signal) : null;

  return new Promise((resolve, reject) => {
    const options = secure
      ? { host: url.hostname, port, path: `${url.pathname}${url.search}`, method }
      : { host: proxy.host, port: proxy.port, path: url.href.replace(/#.*$/, '').replace(/^([a-z]+:\/\/)[^/@]*@/i, '$1'), method };
    options.headers = { ...headersToObject(headers), host: url.host };
    // Node's own fetch always sends these three. A server that sees no User-Agent answers with GitHub's "Request forbidden by
    // administrative rules" 403, and a compressed body would arrive here undecoded, so the encoding is asked for as identity.
    if (!options.headers.accept) options.headers.accept = '*/*';
    if (!options.headers['user-agent']) options.headers['user-agent'] = 'node';
    if (!options.headers['accept-encoding']) options.headers['accept-encoding'] = 'identity';
    if (!secure && proxy.auth) options.headers['proxy-authorization'] = proxy.auth;
    if (socket) {
      // An http request over a socket that is already TLS: this agent exists only to hand the request that one socket.
      const agent = new http.Agent({ keepAlive: false, maxSockets: 1 });
      agent.createConnection = () => socket;
      options.agent = agent;
    }
    const request = http.request(options);
    let answered = false;
    const fail = error => {
      clearTimeout(timer);
      signal?.removeEventListener?.('abort', onAbort);
      if (!answered) { answered = true; request.destroy(); reject(error); }
    };
    const timer = setTimeout(() => fail(new NetworkFailure(secure ? 'The server behind the proxy did not answer in time.' : `The proxy at ${proxy.url} did not answer in time.`, 'NETWORK_TIMEOUT', { proxy: proxy.url })), HANDSHAKE_TIMEOUT_MS);
    const onAbort = () => fail(abortReason(signal));
    if (signal?.aborted) return fail(abortReason(signal));
    signal?.addEventListener?.('abort', onAbort, { once: true });
    request.once('error', error => {
      // Unencrypted means the request never left the proxy; behind a tunnel the error belongs to the origin.
      fail(!secure
        ? new NetworkFailure(`Strata could not reach the proxy at ${proxy.url} (${error.code ?? error.message}). Start your proxy app, or choose a direct connection in Settings, Tools, Network.`, 'PROXY_UNREACHABLE', { cause: error, proxy: proxy.url })
        : Object.assign(error, { proxy: proxy.url }));
    });
    request.once('response', message => {
      answered = true;
      clearTimeout(timer);
      signal?.removeEventListener?.('abort', onAbort);
      resolve(responseFrom(message, url));
    });
    if (body) request.write(body);
    request.end();
  });
}

async function sendOnce(url, { method, headers, body, signal, env, platform, now, registry }) {
  const decision = detectProxy(url, { env, platform, now, registry });
  if (decision.mode === 'direct') return directFetch(url, { method, headers, body, signal });
  return proxiedFetch(url, { method, headers, body, signal }, decision.proxy);
}

/**
 * The one fetch the host uses for the internet.
 *
 * Drop-in for fetch for the calls this app makes (string or URL, method, headers, string body, signal, redirect), with one
 * difference: it goes through the proxy this computer is set to use, and it never sends this computer's own addresses to a proxy.
 * `init.network` can name { env, platform, now, registry } for tests; nothing in the app passes it.
 */
export async function netFetch(input, init = {}) {
  const target = toURL(input);
  const redirectMode = init.redirect ?? 'follow';
  const options = init.network ?? {};
  let url = target;
  let method = String(init.method ?? (init.body === undefined ? 'GET' : 'POST')).toUpperCase();
  let body = bodyBytes(init.body);
  const headers = new Headers(init.headers ?? {});
  if (body && !headers.has('content-length')) headers.set('content-length', String(body.length));
  if (body && !headers.has('content-type')) headers.set('content-type', 'text/plain;charset=UTF-8');

  for (let hop = 0; hop <= MAX_REDIRECTS; hop++) {
    const response = await sendOnce(url, { method, headers, body, signal: init.signal, ...options });
    const location = response.headers.get('location');
    if (!REDIRECTS.has(response.status) || !location) return response;
    if (redirectMode === 'error') {
      await response.body?.cancel().catch(() => {});
      throw new TypeError(`Request failed: ${response.status} redirect to ${location}`);
    }
    if (redirectMode === 'manual') return response;
    const next = new URL(location, url);
    await response.body?.cancel().catch(() => {});
    if (!['http:', 'https:'].includes(next.protocol)) throw new NetworkFailure(`The server redirected to ${next.protocol}, which cannot be followed.`, 'REDIRECT_BLOCKED');
    if (response.status === 303 || ((response.status === 301 || response.status === 302) && method !== 'GET' && method !== 'HEAD')) {
      method = 'GET';
      body = null;
      headers.delete('content-length');
      headers.delete('content-type');
    }
    url = next;
  }
  throw new NetworkFailure('The server redirected too many times.', 'TOO_MANY_REDIRECTS');
}

/* ------------------------------------------------------------------ state --- */

/**
 * One small request through the door the person just chose, for the "Check connection" button. It answers the same question the
 * real requests will: which route was used, did anything answer, and if not, what went wrong.
 */
export async function probeNetwork({ url = 'https://api.github.com/', timeoutMs = 10_000 } = {}) {
  const started = Date.now();
  let decision;
  try { decision = detectProxy(new URL(url)); }
  catch (error) { return { ok: false, code: error.code ?? '', message: error.message, via: 'direct', proxy: null, source: 'settings', ms: 0 }; }
  const via = { via: decision.mode, proxy: decision.proxy?.url ?? null, source: decision.source };
  try {
    const response = await netFetch(url, { method: 'HEAD', redirect: 'manual', signal: AbortSignal.timeout(timeoutMs) });
    await response.body?.cancel().catch(() => {});
    return { ok: response.status < 500, status: response.status, ms: Date.now() - started, ...via };
  } catch (error) {
    return { ok: false, code: error?.code ?? '', message: error?.message ?? 'The check failed.', ms: Date.now() - started, ...via };
  }
}

/** The saved choice for the whole host, and the small surface the Settings screen writes it through. */
export function createNetwork({ directory } = {}) {
  const file = directory ? join(directory, 'net.json') : settingsFile;
  configureNetwork({ file });
  return {
    file,
    view: options => networkView(options),
    save: body => { saveNetworkPolicy(body, { file }); return networkView(); },
    probe: options => probeNetwork(options),
    policy: () => networkPolicy({ file }),
  };
}
