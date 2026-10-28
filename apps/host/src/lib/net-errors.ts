/**
 * The four network failures a person can actually act on, in words a person can act on.
 *
 * Every outbound request in Strata goes through one door (server/net.mjs), so every failure arrives here with a code the operating
 * system or the proxy gave it: a name that could not be looked up, a door that refused, a proxy that wants a sign-in, or a certificate
 * that could not be trusted. Instead of showing "fetch failed" or "ECONNREFUSED" the Settings screen shows one sentence about what
 * happened and one thing to do next. The raw text stays available under Details.
 *
 * Four classes only, because there are four different things a person can do about it. Anything else falls back to `other`, which
 * still says what happened and offers the proxy setting as the place to look.
 *
 * The sentences are templates with `{0}` slots and the values are returned beside them, so the screen can pass both through tr()
 * and every language can carry the sentence.
 */

import { tr } from './i18n';

export type NetworkErrorKind = 'dns' | 'refused' | 'proxy-auth' | 'certificate' | 'other';

export interface NetworkProblem {
  kind: NetworkErrorKind;
  /** One plain sentence about what happened, with `{0}`, `{1}` slots. */
  message: string;
  /** One action the person can take. */
  next: string;
  /** What fills the slots of `message` and `next`. */
  values: string[];
  /** The original code and text, for the Details line. */
  detail: string;
}

const DNS_CODES = new Set(['ENOTFOUND', 'EAI_AGAIN', 'EAI_FAIL', 'EAI_NODATA', 'EAI_NONAME', 'ENODATA', 'ERR_NAME_NOT_RESOLVED']);
const REFUSED_CODES = new Set([
  'ECONNREFUSED', 'ECONNRESET', 'ECONNABORTED', 'EHOSTUNREACH', 'ENETUNREACH', 'ENETDOWN', 'ETIMEDOUT', 'EPIPE', 'UND_ERR_CONNECT_TIMEOUT',
  'PROXY_UNREACHABLE', 'PROXY_TIMEOUT', 'PROXY_REFUSED', 'NETWORK_TIMEOUT', 'TLS_TIMEOUT', 'UND_ERR_SOCKET',
]);
const CERTIFICATE_CODES = new Set([
  'CERT_HAS_EXPIRED', 'CERT_NOT_YET_VALID', 'CERT_REVOKED', 'CERT_UNTRUSTED', 'CERT_SIGNATURE_FAILURE', 'DEPTH_ZERO_SELF_SIGNED_CERT',
  'SELF_SIGNED_CERT_IN_CHAIN', 'UNABLE_TO_VERIFY_LEAF_SIGNATURE', 'UNABLE_TO_GET_ISSUER_CERT', 'UNABLE_TO_GET_ISSUER_CERT_LOCALLY',
  'UNABLE_TO_DECRYPT_CERT_SIGNATURE', 'ERR_TLS_CERT_ALTNAME_INVALID', 'ERR_SSL_WRONG_VERSION_NUMBER', 'ERR_SSL_PACKET_LENGTH_TOO_LONG',
  'EPROTO', 'ERR_TLS_HANDSHAKE_TIMEOUT',
]);
const AUTH_HINTS = /407|proxy authentication|proxy-authorization|PROXY_AUTH/i;
const CERTIFICATE_HINTS = /certificate|self[- ]signed|CERT_|altnames|ERR_TLS_CERT/i;

const text = (value: unknown): string => {
  if (value instanceof Error) {
    const code = (value as { code?: unknown }).code;
    return `${code ? `${String(code)}: ` : ''}${value.message}`;
  }
  if (value && typeof value === 'object' && typeof (value as { message?: unknown }).message === 'string') {
    const record = value as { code?: unknown; message: string };
    return `${record.code ? `${String(record.code)}: ` : ''}${record.message}`;
  }
  if (typeof value === 'string') return value;
  try { return JSON.stringify(value); } catch { return String(value); }
};

function codeOf(error: unknown): string {
  if (!error || typeof error !== 'object') return '';
  const record = error as { code?: unknown; cause?: unknown; name?: unknown };
  if (typeof record.code === 'string') return record.code.toUpperCase();
  if (record.cause) return codeOf(record.cause);           // Node wraps fetch failures: the code lives on the cause
  if (record.name === 'TimeoutError') return 'ETIMEDOUT';
  return '';
}

/**
 * Turn whatever came out of a request into one sentence and one action.
 * `proxy` is the proxy that was in use, when the caller knows it: a refused connection to the proxy is a different problem from a
 * refused connection to a server, and the person fixes it in a different place.
 */
export function classifyNetworkError(error: unknown, { proxy = null }: { proxy?: string | null } = {}): NetworkProblem {
  const code = codeOf(error);
  const raw = text(error);
  const at = proxy ? String(proxy) : '';

  if (code === 'PROXY_AUTH' || AUTH_HINTS.test(raw)) {
    return {
      kind: 'proxy-auth',
      message: at ? 'The proxy at {0} wants a user name and password, and refused this request without them.' : 'The proxy wants a user name and password, and refused this request without them.',
      next: 'Put your proxy user name and password in the proxy address, like http://user:password@127.0.0.1:7890, in Settings, Tools, Network.',
      values: at ? [at] : [],
      detail: raw,
    };
  }
  if (CERTIFICATE_CODES.has(code) || (code === '' && CERTIFICATE_HINTS.test(raw))) {
    return {
      kind: 'certificate',
      message: 'The secure connection could not be trusted, so Strata stopped instead of sending anything to an unverified server.',
      next: 'If an antivirus or a company proxy inspects HTTPS connections, allow Strata in it or install its certificate; otherwise check that this computer’s date and time are right.',
      values: [],
      detail: raw,
    };
  }
  if (DNS_CODES.has(code)) {
    return {
      kind: 'dns',
      message: at
        ? 'The address of that server could not be looked up through the proxy at {0}, so the request never left this computer.'
        : 'The address of that server could not be looked up, so the request never left this computer.',
      next: at ? 'Check that the proxy is running and can look up names, then try again.' : 'Check this computer’s internet connection and try again.',
      values: at ? [at] : [],
      detail: raw,
    };
  }
  if (REFUSED_CODES.has(code)) {
    const refusedProxy = /^PROXY_/.test(code) || code === 'PROXY_UNREACHABLE';
    if (refusedProxy && at) {
      return {
        kind: 'refused',
        message: 'Nothing answered at the proxy at {0}.',
        next: 'Start your proxy app, or choose a direct connection in Settings, Tools, Network.',
        values: [at],
        detail: raw,
      };
    }
    return {
      kind: 'refused',
      message: at ? 'The proxy at {0} could not reach that server.' : 'The server refused the connection, or did not answer in time.',
      next: 'Try again in a moment. If it keeps failing, check the address and your internet connection.',
      values: at ? [at] : [],
      detail: raw,
    };
  }
  return {
    kind: 'other',
    message: 'The request could not be completed.',
    next: 'Try again. If it keeps failing, open Settings, Tools, Network and check how Strata reaches the internet.',
    values: [],
    detail: raw,
  };
}

/** One line for a toast or a log: the sentence with its slots filled, then the action. */
export function networkErrorSentence(error: unknown, options: { proxy?: string | null } = {}): string {
  const problem = classifyNetworkError(error, options);
  // Translate the explanation and the next step on their own, because no key holds the two joined into one sentence.
  return `${tr(problem.message, problem.values)} ${tr(problem.next)}`;
}
