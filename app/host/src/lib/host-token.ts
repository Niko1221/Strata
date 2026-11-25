/**
 * The host puts this launch's secret in the page it serves (<meta name="strata-token">), and refuses any change a browser sends to /api without it
 * (server/request-guard.mjs, #7130). Every request the page makes goes through fetch, so one wrapper adds the header to the page's own /api
 * requests that change something; reads need nothing. A request to any other address never gets the secret.
 */
export const TOKEN_HEADER = 'x-strata-token';

export function hostToken(doc: Pick<Document, 'querySelector'> | undefined = typeof document === 'undefined' ? undefined : document): string {
  return doc?.querySelector('meta[name="strata-token"]')?.getAttribute('content') ?? '';
}

/** True for a request this page sends to its own host's /api that is not a read. */
export function needsToken(input: RequestInfo | URL, init: RequestInit | undefined, base: string): boolean {
  const request = typeof Request !== 'undefined' && input instanceof Request ? input : null;
  const method = String(init?.method ?? request?.method ?? 'GET').toUpperCase();
  if (['GET', 'HEAD', 'OPTIONS'].includes(method)) return false;
  try {
    const url = new URL(request ? request.url : String(input instanceof URL ? input.href : input), base);
    return url.origin === new URL(base).origin && url.pathname.startsWith('/api/');
  } catch { return false; }
}

export function installHostToken(): void {
  if (typeof window === 'undefined' || (window.fetch as { strataToken?: boolean }).strataToken) return;
  const original = window.fetch.bind(window);
  const wrapped = ((input: RequestInfo | URL, init?: RequestInit) => {
    const token = hostToken();
    if (!token || !needsToken(input, init, window.location.href)) return original(input, init);
    const headers = new Headers(init?.headers ?? (typeof Request !== 'undefined' && input instanceof Request ? input.headers : undefined));
    headers.set(TOKEN_HEADER, token);
    return original(input, { ...init, headers });
  }) as typeof window.fetch & { strataToken?: boolean };
  wrapped.strataToken = true;
  window.fetch = wrapped;
}
