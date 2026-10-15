import http from 'node:http';
import { readFileSync } from 'node:fs';
import { createHash, timingSafeEqual } from 'node:crypto';
import { pipeline } from 'node:stream';

function readConfig() {
  const config = JSON.parse(readFileSync(process.argv[2], 'utf8').replace(/^\uFEFF/, ''));
  if (typeof config.apiKey !== 'string' || config.apiKey.length < 32) {
    throw new Error('A generated API key is required. Run Start-Remote.ps1 first.');
  }
  return config;
}
const config = readConfig();
const port = config.port ?? 8787;
const maxBody = 32 * 1024 * 1024;
const routes = new Map([
  ['/v1/models', 'GET'],
  ['/v1/chat/completions', 'POST'],
  ['/v1/responses', 'POST'],
  ['/v1/messages', 'POST'],
  ['/v1/messages/count_tokens', 'POST'],
]);
const blockedHeaders = new Set([
  'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
  'te', 'trailer', 'transfer-encoding', 'upgrade', 'host',
  'authorization', 'x-api-key', 'cookie', 'origin', 'referer',
  'forwarded', 'x-forwarded-for', 'x-forwarded-host', 'x-forwarded-proto',
]);

function json(res, status, message, type = 'invalid_request_error') {
  const body = JSON.stringify({ error: { message, type } });
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(body),
    'cache-control': 'no-store',
    'x-content-type-options': 'nosniff',
  });
  res.end(body);
}

function requestAccess(req, modelRoute) {
  const auth = req.headers.authorization ?? '';
  let supplied = /^Bearer /i.test(auth)
    ? auth.slice(7).trim() : (req.headers['x-api-key'] ?? '');
  try {
    // Read the protected source of truth, so an old process cannot retain a revoked key.
    const current = readConfig();
    const dashboardLogin = !modelRoute && current.exposeDashboard === true;
    if (current.exposeDashboard === true && /^Basic /i.test(auth)) {
      const credentials = Buffer.from(auth.slice(6).trim(), 'base64').toString('utf8');
      supplied = credentials.startsWith('strata:') ? credentials.slice(7) : '';
    }
    const expectedKey = createHash('sha256').update(current.apiKey).digest();
    return { config: current, dashboardLogin,
      authorized: timingSafeEqual(createHash('sha256').update(supplied).digest(), expectedKey) };
  } catch {
    return null;
  }
}

function cleanHeaders(headers) {
  const extra = new Set((headers.connection ?? '').split(',').map(x => x.trim().toLowerCase()));
  return Object.fromEntries(Object.entries(headers).filter(([name]) =>
    !blockedHeaders.has(name) && !extra.has(name)));
}

const active = { api: 0, dashboard: 0 };
const server = http.createServer(async (req, res) => {
  let target;
  try {
    target = new URL(req.url, 'http://127.0.0.1');
  } catch {
    json(res, 400, 'Invalid request URL.');
    return;
  }
  const pathname = target.pathname.replace(/\/$/, '');
  const method = routes.get(pathname);
  const access = requestAccess(req, Boolean(method));
  if (!access?.authorized) {
    res.setHeader('www-authenticate', access?.dashboardLogin ? 'Basic realm="Strata Dashboard", charset="UTF-8"' : 'Bearer realm="Strata"');
    res.setHeader('connection', 'close');
    json(res, 401, access?.dashboardLogin ? 'Missing or incorrect dashboard login.' : 'Missing or incorrect API key.', 'authentication_error');
    return;
  }
  // A browser remembers Basic credentials. Do not let a foreign page spend inference or use local-image inputs.
  if (req.method === 'POST' && /^Basic /i.test(req.headers.authorization ?? '')) {
    let sameOrigin = false;
    try {
      const origin = new URL(req.headers.origin);
      sameOrigin = origin.origin === req.headers.origin && origin.host === req.headers.host &&
        (origin.protocol === 'https:' || (origin.protocol === 'http:' &&
          ['127.0.0.1', 'localhost'].includes(origin.hostname)));
    } catch { /* Missing, null and malformed origins fail closed for browser mutations. */ }
    if (!sameOrigin || req.headers['sec-fetch-site'] === 'cross-site') {
      json(res, 403, 'Dashboard browser requests must come from the same origin. Use Bearer authentication in API clients.');
      return;
    }
  }
  const requestConfig = access.config;
  res.setHeader('x-strata-gateway-version', '2');
  const methods = method ? [method] : requestConfig.exposeDashboard === true ? ['GET', 'POST'] : [];
  if (!methods.length) {
    json(res, 404, 'This endpoint is not available through the public gateway.');
    return;
  }
  if (!methods.includes(req.method)) {
    res.setHeader('allow', methods.join(', '));
    json(res, 405, 'Method not allowed.');
    return;
  }
  const pool = method ? 'api' : 'dashboard';
  if (active[pool] >= 16) {
    json(res, 429, 'The gateway is busy. Retry shortly.');
    return;
  }
  active[pool]++;
  let upstream;
  let release = false;
  res.on('close', () => {
    if (!release) { active[pool]--; release = true; }
    if (!res.writableFinished) upstream?.destroy();
  });
  try {
    const chunks = [];
    let size = 0;
    // Bound buffered prompts, while streaming model output without buffering.
    if (Number(req.headers['content-length'] ?? 0) > maxBody) {
      res.setHeader('connection', 'close');
      json(res, 413, 'Request body exceeds 32 MiB.');
      return;
    }
    for await (const chunk of req) {
      size += chunk.length;
      if (size > maxBody) {
        res.setHeader('connection', 'close');
        json(res, 413, 'Request body exceeds 32 MiB.');
        return;
      }
      chunks.push(chunk);
    }
    if (res.destroyed) return;
    const body = Buffer.concat(chunks);
    const headers = cleanHeaders(req.headers);
    headers.host = '127.0.0.1:8080';
    headers['content-length'] = body.length;
    headers['accept-encoding'] = 'identity';
    // The upstream remains local; gateway credentials never enter its logs.
    if (requestConfig.upstreamApiKey) headers.authorization = `Bearer ${requestConfig.upstreamApiKey}`;
    upstream = http.request({
      hostname: '127.0.0.1', port: 8080,
      path: (pathname || '/') + target.search,
      method: req.method, headers,
    }, response => {
      const responseHeaders = cleanHeaders(response.headers);
      responseHeaders['cache-control'] = 'no-store';
      responseHeaders['x-content-type-options'] = 'nosniff';
      res.writeHead(response.statusCode, responseHeaders);
      res.flushHeaders();
      pipeline(response, res, error => { if (error) upstream.destroy(); });
    });
    upstream.setTimeout(600_000, () => upstream.destroy(new Error('Upstream timed out.')));
    upstream.on('error', () => {
      if (!res.headersSent && !res.destroyed) {
        json(res, 502, 'Strata is unavailable. Start the model on this PC.', 'server_error');
      } else {
        res.destroy();
      }
    });
    upstream.end(body);
  } catch {
    upstream?.destroy();
    if (!res.headersSent && !res.destroyed) json(res, 400, 'Unable to read request.');
  }
});
server.headersTimeout = 30_000;
server.requestTimeout = 120_000;
server.keepAliveTimeout = 5_000;
server.maxHeadersCount = 100;
server.on('error', error => { console.error(error.message); process.exitCode = 1; });
server.listen(port, '127.0.0.1', () => {
  console.log(`Authenticated Strata gateway listening at 127.0.0.1:${server.address().port}`);
});
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => { server.close(); server.closeAllConnections(); });
}
