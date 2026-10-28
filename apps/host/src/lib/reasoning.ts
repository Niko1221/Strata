/** The newest line of the model's reasoning, trimmed for one compact row (while it is still reasoning, the row shows what it is thinking about). */
export function lastReasoningLine(text: string | undefined, max = 110): string {
  const lines = String(text ?? '').split(/\r?\n/).map(l => l.replace(/^[\s>*#\-•\d.]+/, '').replace(/[*_`]/g, '').trim()).filter(Boolean);
  const line = lines.at(-1) ?? '';
  return line.length > max ? `${line.slice(0, max - 1)}…` : line;
}

/** "38s", "1m 04s": how long the model reasoned. null when it is too short to be worth saying. */
export function thoughtDuration(ms: number | undefined): string | null {
  if (typeof ms !== 'number' || !Number.isFinite(ms) || ms < 1000) return null;
  const total = Math.round(ms / 1000);
  return total < 60 ? `${total}s` : `${Math.floor(total / 60)}m ${String(total % 60).padStart(2, '0')}s`;
}

/**
 * How long the model has thought on this message: the segments that ended plus the one running now (only while the
 * message streams). A segment ends when the answer starts or a tool call begins, so tool runs between two thoughts are
 * not counted as thinking.
 */
export function reasoningTime(message: { reasoningMs?: number; reasoningStartedAt?: number; reasoningOpen?: boolean; streaming?: boolean }, now = Date.now()): number {
  const base = typeof message.reasoningMs === 'number' && Number.isFinite(message.reasoningMs) ? message.reasoningMs : 0;
  if (message.streaming && message.reasoningOpen && typeof message.reasoningStartedAt === 'number')
    return base + Math.max(0, now - message.reasoningStartedAt);
  return base;
}

/** End the running reasoning segment: its time joins the total, and the start marker is cleared for the next one. */
export function closeReasoningSegment<T extends { reasoningMs?: number; reasoningStartedAt?: number; reasoningOpen?: boolean }>(message: T, now = Date.now()): T {
  if (!message.reasoningOpen)
    return message;
  const started = typeof message.reasoningStartedAt === 'number' ? message.reasoningStartedAt : now;
  const total = (message.reasoningMs ?? 0) + Math.max(0, now - started);
  return { ...message, reasoningMs: total, reasoningOpen: false, reasoningStartedAt: undefined };
}

/** About how many tokens a reasoning text is: a CJK character counts as one, other text as four characters per token. */
export function approximateTokens(text: string | undefined): number {
  const value = String(text ?? '');
  if (!value)
    return 0;
  let cjk = 0;
  for (const character of value)
    if (/[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]/.test(character))
      cjk += 1;
  return cjk + Math.ceil((value.length - cjk) / 4);
}

/** Is a scrolling box at its bottom (within a few pixels)? Reasoning sticks to the bottom only while the person has not scrolled up. */
export function atBottom(el: { scrollHeight: number; scrollTop: number; clientHeight: number }, slack = 24): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= slack;
}
