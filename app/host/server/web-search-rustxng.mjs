/** RustXNG's fixed loopback JSON API. No page bodies or credentials leave this adapter. */
import { SearchUnavailable } from './web-search-ddg.mjs';
import { publicSourceUrl } from './chat-web-search.mjs';

export const RUSTXNG_PROVIDER_ID = 'strata-rustxng';

export function rustxngEndpoint(value) {
  const url = new URL(value);
  if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || !url.port || url.username || url.password || url.pathname !== '/' || url.search || url.hash) throw new Error('RustXNG requires an app-owned loopback address.');
  return url;
}

export async function boundedJson(response, maxBytes = 1024 * 1024) {
  const reader = response.body?.getReader();
  if (!reader) throw new Error('Empty search response.');
  const chunks = []; let bytes = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > maxBytes) throw new Error('Search response is too large.');
      chunks.push(value);
    }
    return JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch (error) { await reader.cancel().catch(() => {}); throw error; }
}

export function createRustxngSearch({ endpoint, fetch: fetchImpl = globalThis.fetch } = {}) {
  return {
    id: RUSTXNG_PROVIDER_ID,
    available: () => !!endpoint,
    async search(request, signal) {
      if (typeof request?.query !== 'string' || !request.query.trim() || request.query.length > 2000) throw new SearchUnavailable('Enter a search query of 1–2000 characters.', 'WEB_PROVIDER_ERROR');
      try {
        signal?.throwIfAborted();
        const base = typeof endpoint === 'function' ? await endpoint(signal) : endpoint;
        const url = new URL('search', rustxngEndpoint(base));
        const limit = Math.max(1, Math.min(10, Number.isInteger(request.maxResults) ? request.maxResults : 5));
        url.search = new URLSearchParams({ q: request.query.trim(), format: 'json', max_results: String(limit), include_raw_content: '0' }).toString();
        const timeout = AbortSignal.timeout(45_000);
        const response = await fetchImpl(url, { redirect: 'error', headers: { accept: 'application/json' }, signal: signal ? AbortSignal.any([signal, timeout]) : timeout });
        if (!response.ok) {
          await response.body?.cancel();
          throw new SearchUnavailable(response.status === 429 ? 'RustXNG is rate limited. Try again later.' : 'RustXNG could not complete the search.', response.status === 429 ? 'WEB_RATE_LIMITED' : 'WEB_PROVIDER_ERROR');
        }
        const body = await boundedJson(response);
        if (!Array.isArray(body.results)) throw new Error('Invalid RustXNG result list.');
        const seen = new Set();
        const sources = body.results.flatMap(row => {
          const url = publicSourceUrl(row?.url);
          if (!url || seen.has(url)) return [];
          seen.add(url);
          return [{ url, title: String(row.title ?? '').slice(0, 500), snippet: String(row.content ?? '').slice(0, 2000) }];
        }).slice(0, limit);
        if (!sources.length) throw new SearchUnavailable('RustXNG returned no usable results. Upstream providers may be unavailable; do not retry repeatedly.', 'WEB_PROVIDER_ERROR');
        return { sources, truncated: body.results.length > sources.length };
      } catch (error) {
        if (signal?.aborted) throw new SearchUnavailable('The search was cancelled.', 'WEB_ABORTED');
        if (error instanceof SearchUnavailable) throw error;
        throw new SearchUnavailable('RustXNG is unavailable. Check its bundled files and the connection, then try again.', 'WEB_PROVIDER_ERROR', { cause: error });
      }
    },
  };
}
