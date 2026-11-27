// Who may use the host's HTTP API from a web browser (#7130).
//
// The host listens on 127.0.0.1 and the app's window is a page served by the host itself (http://127.0.0.1:<port>/). A browser lets any page, on any
// site and on any other localhost port, send requests to that address, so the host has to decide which pages are the app's own. The rules, all of
// them applied to the app's own listener only (LAN sharing and the tunnel are other listeners with their own API key):
//
//   1. Host. While the host listens on this PC only, the Host header must name this PC (localhost, 127.x.x.x, [::1], or a name in STRATA_ALLOWED_HOSTS).
//      A page of another site that points its own name at 127.0.0.1 (DNS rebinding) sends its own name and is refused.
//   2. Origin. A /api request that carries an Origin must come from the host's own page: a loopback name on the host's own port (or the same
//      origin as the Host header), never another localhost port. A development run (STRATA_PACKAGED is not 1) also accepts Vite on 5199.
//   3. Fetch metadata. A /api request a browser marks cross-site or same-site (another port of localhost is "same-site") is refused.
//   4. Token. A /api request that changes something (not GET/HEAD/OPTIONS) and comes from a browser (it carries an Origin or Sec-Fetch-Site) must
//      send this launch's secret in x-strata-token. The host puts the secret in the page it serves (<meta name="strata-token">) and the page's own
//      script adds it to its requests. Scripts, curl and the shell send neither Origin nor Sec-Fetch-Site and are not browsers.
//   5. CORS. Only /v1 (the OpenAI-compatible API, for local web UIs) answers any origin; /api and the page answer none.
import { randomBytes, timingSafeEqual } from 'node:crypto';

const LOOPBACK_NAME = /^(?:localhost|127(?:\.\d{1,3}){3}|\[::1\])$/i;
export const TOKEN_HEADER = 'x-strata-token';
const DEV_PORTS = [5199];

export const isLoopbackName = name => LOOPBACK_NAME.test(String(name ?? ''));
export const isLoopbackBind = bind => ['127.0.0.1', 'localhost', '::1', '[::1]'].includes(String(bind ?? '').toLowerCase()) || /^127(\.\d{1,3}){3}$/.test(String(bind ?? ''));

function sameToken(given, token) {
  const a = Buffer.from(typeof given === 'string' ? given : '');
  const b = Buffer.from(token);
  return a.length === b.length && timingSafeEqual(a, b);
}

function hostName(hostHeader) {
  try { return new URL(`http://${hostHeader}`).hostname.toLowerCase(); } catch { return null; }
}

/** `env`: process.env. `port`: the port the host listens on. `bind`: the address it listens on. */
export function createRequestGuard({ port, bind, env = process.env, token = randomBytes(32).toString('hex') } = {}) {
  const packaged = env.STRATA_PACKAGED === '1';
  const loopbackOnly = isLoopbackBind(bind);
  const allowedHosts = new Set(String(env.STRATA_ALLOWED_HOSTS ?? '').split(',').map(s => s.trim().toLowerCase()).filter(Boolean));
  const devOrigins = new Set();
  if (!packaged) {
    for (const p of DEV_PORTS) for (const n of ['localhost', '127.0.0.1', '[::1]']) devOrigins.add(`http://${n}:${p}`);
    for (const o of String(env.STRATA_DEV_ORIGINS ?? '').split(',').map(s => s.trim()).filter(Boolean)) devOrigins.add(o);
  }

  function hostRefusal(req) {
    const host = req.headers?.host;
    if (!loopbackOnly || host === undefined) return null;
    const name = hostName(host);
    if (name && (isLoopbackName(name) || allowedHosts.has(name))) return null;
    return { status: 403, message: 'Open the local Strata app to use host controls.', code: 'bad-host' };
  }

  /** The origin header's own origin, or the reason it is not one ("null", an unreadable value). */
  function parseOrigin(value) {
    try {
      const u = new URL(String(value).trim());
      return /^https?:$/.test(u.protocol) && u.host ? u : null;
    } catch { return null; }
  }

  /** 'self': this host's own page. 'dev': a development page (not packaged). null: anyone else. */
  function originKind(value, req) {
    const u = parseOrigin(value);
    if (!u) return null;
    const effectivePort = Number(u.port || (u.protocol === 'https:' ? 443 : 80));
    if (u.protocol === 'http:' && isLoopbackName(u.hostname) && effectivePort === port) return 'self';
    const host = req.headers?.host;
    if (typeof host === 'string' && u.host.toLowerCase() === host.toLowerCase() && (!loopbackOnly || isLoopbackName(u.hostname) || allowedHosts.has(u.hostname.toLowerCase()))) return 'self';
    if (devOrigins.has(`${u.protocol}//${u.host.toLowerCase()}`)) return 'dev';
    return null;
  }

  function apiRefusal(req, method) {
    const origin = req.headers?.origin;
    const site = String(req.headers?.['sec-fetch-site'] ?? '').toLowerCase();
    let kind = null;
    if (origin !== undefined) {
      kind = originKind(origin, req);
      if (!kind) return { status: 403, message: 'Open the local Strata app to use host controls.', code: 'bad-origin' };
    }
    if ((site === 'cross-site' || site === 'same-site') && kind !== 'dev') return { status: 403, message: 'Open the local Strata app to use host controls.', code: 'bad-site' };
    const changes = !['GET', 'HEAD', 'OPTIONS'].includes(method);
    const browser = origin !== undefined || site !== '';
    if (changes && browser && kind !== 'dev' && !sameToken(req.headers?.[TOKEN_HEADER], token)) return { status: 403, message: 'Reload the Strata page and try again.', code: 'bad-token' };
    return null;
  }

  return {
    token,
    /** null when the request may go on, else { status, message, code }. Run before routing, for every method including OPTIONS. */
    refusal(req, path, method) {
      const host = hostRefusal(req);
      if (host) return host;
      if (path.startsWith('/api/')) return apiRefusal(req, method);
      return null;
    },
    /** The CORS headers for this response: only /v1 answers any page; /api answers its own page's origin (needed by none today). */
    cors(req, path) {
      if (path.startsWith('/v1/')) return { 'access-control-allow-origin': '*', 'access-control-allow-headers': 'content-type, authorization, x-api-key, x-strata-client', 'access-control-allow-methods': 'GET, POST, PATCH, DELETE, OPTIONS' };
      const origin = req.headers?.origin;
      if (path.startsWith('/api/') && origin !== undefined && originKind(origin, req)) return { 'access-control-allow-origin': String(origin), vary: 'Origin', 'access-control-allow-headers': `content-type, x-strata-client, x-strata-window, x-strata-base-rev, ${TOKEN_HEADER}`, 'access-control-allow-methods': 'GET, POST, PUT, PATCH, DELETE, OPTIONS' };
      return {};
    },
    /** May this static request be given the token? Only the page's own origin, or a navigation with no Origin. */
    mayHaveToken(req) {
      const origin = req.headers?.origin;
      const site = String(req.headers?.['sec-fetch-site'] ?? '').toLowerCase();
      if (site === 'cross-site' || site === 'same-site') return false;
      return origin === undefined || originKind(origin, req) !== null;
    },
    /** The page with this launch's token in it (and the same-origin framing rule). */
    inject(html) {
      const tag = `<meta name="strata-token" content="${token}">`;
      return /<\/head>/i.test(html) ? html.replace(/<\/head>/i, `${tag}</head>`) : `${tag}${html}`;
    },
  };
}
