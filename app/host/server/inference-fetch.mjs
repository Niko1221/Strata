/** Local inference can spend more than fetch's five-minute header wait reading a prompt. */
import http from 'node:http';
import { Readable } from 'node:stream';

export const INFERENCE_DEADLINE_MS = 60 * 60 * 1000;

/** A bounded, cancellable local request. Remote providers retain normal fetch behavior. */
export function inferenceFetch(input, init = {}, { timeoutMs = INFERENCE_DEADLINE_MS } = {}) {
  const url = new URL(input);
  if (url.protocol !== 'http:' || !['127.0.0.1', '[::1]'].includes(url.hostname)) return fetch(input, init);
  if (url.username || url.password) throw new TypeError('Inference URL must not contain credentials');
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0 || timeoutMs > INFERENCE_DEADLINE_MS) throw new RangeError('Invalid inference deadline');
  if (init.body != null && typeof init.body !== 'string' && !Buffer.isBuffer(init.body)) throw new TypeError('Local inference expects a JSON request body');
  return new Promise((resolve, reject) => {
    const headers = new Headers(init.headers);
    headers.set('accept-encoding', 'identity');
    if (init.body != null) headers.set('content-length', String(Buffer.byteLength(init.body)));
    let timer;
    const request = http.request(url, { method: init.method ?? 'GET', headers: Object.fromEntries(headers), signal: init.signal }, reply => {
      reply.once('close', () => clearTimeout(timer));
      const responseHeaders = new Headers();
      for (let i = 0; i < reply.rawHeaders.length; i += 2) responseHeaders.append(reply.rawHeaders[i], reply.rawHeaders[i + 1]);
      const empty = init.method === 'HEAD' || [204, 205, 304].includes(reply.statusCode);
      if (empty) reply.resume();
      resolve(new Response(empty ? null : Readable.toWeb(reply), { status: reply.statusCode, headers: responseHeaders }));
    });
    request.once('error', error => { clearTimeout(timer); reject(error); });
    timer = setTimeout(() => request.destroy(new DOMException('Local inference exceeded its request deadline', 'TimeoutError')), timeoutMs);
    timer.unref();
    request.end(init.body);
  });
}
