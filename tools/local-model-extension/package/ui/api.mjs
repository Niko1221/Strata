const ROOT = '/__strata_extension/api/';
export const IDLES = Object.freeze([0, 300, 600, 1800]);
export const validContext = value => Number.isSafeInteger(value) && value > 0;
export const contextChoices = values => Array.isArray(values) ? [...new Set(values.filter(validContext))] : [];

export function actionBody(action, value) {
  if (action === 'context' && validContext(value)) return { tokens: value };
  if (action === 'idle' && IDLES.includes(value)) return { seconds: value };
  if (['load', 'unload', 'connect'].includes(action) && value === undefined) return {};
  throw new TypeError('Unsupported Current Model Action.');
}

/** The frame has one fixed same-origin API surface, with no arbitrary URL seam. */
export function createApi({ origin = globalThis.location?.origin, fetchImpl = globalThis.fetch } = {}) {
  const base = new URL(origin);
  let announcedContexts = [];
  let announcedModel = null;
  if (!['http:', 'https:'].includes(base.protocol) || base.origin !== origin) throw new TypeError('A Same-Origin HTTP App Is Required.');
  async function request(action, body, signal) {
    const controller = new AbortController();
    const abort = () => controller.abort(signal.reason);
    if (signal?.aborted) abort(); else signal?.addEventListener('abort', abort, { once: true });
    const deadline = setTimeout(() => controller.abort(new Error('The Local Model Request Timed Out.')),
      action === 'load' ? 3600000 : action === 'unload' ? 90000 : 15000);
    try {
    const response = await fetchImpl(new URL(ROOT + action, base).href, {
      method: body === undefined ? 'GET' : 'POST', mode: 'same-origin', credentials: 'same-origin',
      redirect: 'error', cache: 'no-store', signal: controller.signal,
      headers: { 'content-type': 'application/json', 'x-strata-desktop': 'local-model' },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    let value;
    try { value = await response.json(); }
    catch { throw new Error('The Extension Returned An Invalid Response.'); }
    if (!response.ok) {
      const message = typeof value?.error === 'string' ? value.error : value?.error?.message;
      throw new Error(typeof message === 'string' ? message.slice(0, 2000) : 'The Local Model Action Could Not Be Completed.');
    }
    if (controller.signal.aborted) throw controller.signal.reason;
    if (action === 'model') {
      announcedContexts = contextChoices(value?.context?.choices);
      announcedModel = value?.available === true && typeof value?.model?.id === 'string' ? value.model.id : null;
    }
    return value;
    } finally { clearTimeout(deadline); signal?.removeEventListener('abort', abort); }
  }
  return {
    get(action, signal) {
      if (!['model', 'status'].includes(action)) throw new TypeError('Unsupported Extension Read.');
      return request(action, undefined, signal);
    },
    post(action, value, signal) {
      const body = actionBody(action, value);
      if (action === 'context' && !announcedContexts.includes(value)) throw new TypeError('Choose A Context Reported By The Model Catalog.');
      if (!announcedModel) throw new TypeError('Refresh The Physical Model Identity Before Acting.');
      return request(action, { ...body, expectedModel: announcedModel }, signal);
    },
  };
}
