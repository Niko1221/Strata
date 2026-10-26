/**
 * The two ends of the agent's own question (`ask_user_question`), kept in one place because both are
 * assumptions about the pinned Harness's wire shape, not facts read from this repo.
 *
 * The Harness forwards a `user-questions/request` waterfall (agent-bridge.mjs `onHostEvent`) and expects an
 * answer back through `$events/result` (gateway-client.mjs `answerEvent`). This repo shows the event name and
 * the `$events/result` envelope, but NOT the field names inside either payload — those live in the external
 * Harness. Both adapters below are therefore UNCONFIRMED STATIC ANALYSIS: they parse defensively across the
 * plausible shapes and must be checked against the Harness before this path is enabled by default.
 * See the scope issue: "[GUI] Give the agent's own questions an answerer surface".
 */

/**
 * Read the incoming `user-questions/request` payload into a plain card the page can render.
 * @param {object} request the event's `request` object (already known to carry a `callId`).
 * @returns {{ id: string, questions: Array<{ question: string, options: string[] }> }}
 */
export function parseQuestions(request) {
  const id = String(request?.callId ?? '');
  // Plausible shapes, widest first: a `questions`/`items` array, or a single inline question.
  const raw = Array.isArray(request?.questions) ? request.questions
    : Array.isArray(request?.items) ? request.items
    : (request?.question || request?.prompt || request?.text) ? [request]
    : [];
  const questions = raw.map(q => ({
    question: String(q?.question ?? q?.prompt ?? q?.text ?? q ?? '').trim(),
    options: optionsOf(q),
  })).filter(q => q.question);
  return { id, questions };
}

function optionsOf(q) {
  const list = Array.isArray(q?.options) ? q.options
    : Array.isArray(q?.choices) ? q.choices
    : Array.isArray(q?.answers) ? q.answers
    : [];
  // An option may be a bare string or an object with a label/value.
  return list.map(o => String(typeof o === 'string' ? o : (o?.label ?? o?.value ?? o?.text ?? '')).trim()).filter(Boolean);
}

/**
 * Build the `$events/result` value that carries the person's answer back to the Harness.
 * @param {Array<{ question: string, answer: string }>} answers one answer per asked question.
 * @returns {object} the `value` for `{ kind: 'result', value }`.
 */
export function answerWire(answers) {
  const list = (answers ?? []).map(a => ({ question: a.question ?? '', answer: String(a.answer ?? '') }));
  // UNCONFIRMED: the Harness may want `answers`, a bare array, or a single string when one question was asked.
  return { answers: list };
}
