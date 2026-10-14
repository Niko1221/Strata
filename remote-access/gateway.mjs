import http from 'node:http';
import { readFileSync } from 'node:fs';
import { createHash, timingSafeEqual } from 'node:crypto';
import { pipeline } from 'node:stream';

const config = JSON.parse(readFileSync(process.argv[2], 'utf8').replace(/^\uFEFF/, ''));
if (typeof config.apiKey !== 'string' || config.apiKey.length < 32) {
  throw new Error('A generated API key is required. Run Start-Remote.ps1 first.');
}
const expectedKey = createHash('sha256').update(config.apiKey).digest();
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

function authorized(req) {
  const auth = req.headers.authorization ?? '';
  const supplied = /^Bearer /i.test(auth)
    ? auth.slice(7).trim() : (req.headers['x-api-key'] ?? '');
  return timingSafeEqual(
    createHash('sha256').update(supplied).digest(), expectedKey,
  );
}

function cleanHeaders(headers) {
  const extra = new Set((headers.connection ?? '').split(',').map(x => x.trim().toLowerCase()));
  return Object.fromEntries(Object.entries(headers).filter(([name]) =>
    !blockedHeaders.has(name) && !extra.has(name)));
}

let active = 0;
const server = http.createServer(async (req, res) => {
  if (!authorized(req)) {
    res.setHeader('www-authenticate', 'Bearer realm="Strata"');
    res.setHeader('connection', 'close');
    json(res, 401, 'Missing or incorrect API key.', 'authentication_error');
    return;
  }
  let target;
  try {
    target = new URL(req.url, 'http://127.0.0.1');
  } catch {
    json(res, 400, 'Invalid request URL.');
    return;
  }
  const pathname = target.pathname.replace(/\/$/, '');
  const method = routes.get(pathname);
  if (!method) {
    json(res, 404, 'This endpoint is not available through the public gateway.');
    return;
  }
  if (req.method !== method) {
    res.setHeader('allow', method);
    json(res, 405, 'Method not allowed.');
    return;
  }
  if (active >= 16) {
    json(res, 429, 'The gateway is busy. Retry shortly.');
    return;
  }
  active++;
  let upstream;
  let release = false;
  res.on('close', () => {
    if (!release) { active--; release = true; }
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
    if (config.upstreamApiKey) headers.authorization = `Bearer ${config.upstreamApiKey}`;
    upstream = http.request({
      hostname: '127.0.0.1', port: 8080,
      path: pathname + target.search,
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
  console.log(`Authenticated Strata gateway listening at 127.0.0.1:${port}`);
});
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => { server.close(); server.closeAllConnections(); });
}
