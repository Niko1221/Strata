import type { Session } from './types';

/**
 * Saving the chat list without hurting the page (#3604, #3603).
 *
 * Every streamed token used to stringify every chat and write the lot to localStorage. On a long history that is
 * megabytes of string work per token, and the WebView2 renderer grew to gigabytes before it died (the window goes
 * white). Three rules fix it, and the stored format is unchanged, so older builds and the flow scripts still read it:
 *
 *  1. Coalesce: the newest state wins and at most one write happens per `delayMs`. The caller flushes when a turn ends
 *     or the page goes away.
 *  2. Only the chats that changed are serialised again. A chat is replaced by a new object when it changes (the store
 *     never mutates), so the previous JSON of every other chat is reused by object identity.
 *  3. The stored copy is shortened by field, never by message or chat: a huge text (mostly tool results) is clipped,
 *     and if the whole list still does not fit, the oldest chats' fields are clipped harder. Dropping messages or
 *     whole chats would destroy them: this saved copy is the only stored copy of a chat (the store reads the same
 *     key back on load), so a failed or trimmed write is a permanent loss.
 */

/** Longest text kept for one field in the saved copy. */
export const MAX_FIELD_CHARS = 200_000;
/** Longest text kept for one field when the list is too big and the oldest chats are shortened to make room. */
export const SHRINK_FIELD_CHARS = 20_000;
/** Longest saved copy of one chat (JSON characters). Only advisory: an over-long chat is kept, its fields are shortened instead. */
export const MAX_SESSION_CHARS = 1_500_000;
/** Longest saved chat list (JSON characters). localStorage allows about 5M; keep a margin for everything else. */
export const MAX_TOTAL_CHARS = 3_500_000;

type Json = unknown;

function clipFields(value: Json, maxChars: number, depth = 0): Json {
  if (typeof value === 'string') return value.length > maxChars ? `${value.slice(0, maxChars)}\n[shortened in the saved copy]` : value;
  if (depth > 6 || value === null || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map(v => clipFields(v, maxChars, depth + 1));
  const out: Record<string, Json> = {};
  for (const [k, v] of Object.entries(value as Record<string, Json>)) out[k] = clipFields(v, maxChars, depth + 1);
  return out;
}

/**
 * The JSON text kept for one chat: whole when it fits, otherwise its oversized fields are clipped.
 * Every message stays: the questions and answers are small, the tool results are what grow, and removing a message
 * here would remove it for good (the load reads this same copy back).
 */
export function serializeSession(session: Session, maxFieldChars: number = MAX_FIELD_CHARS): string {
  const whole = JSON.stringify(session);
  if (whole.length <= MAX_SESSION_CHARS && maxFieldChars === MAX_FIELD_CHARS) return whole;
  return JSON.stringify(clipFields(session, maxFieldChars));
}

export interface SessionWriterOptions {
  /** Where the list is saved (the storage write). Receives the finished JSON text. */
  write: (json: string) => void;
  delayMs?: number;
  maxTotalChars?: number;
  /** Counts a serialisation of one chat; for tests. */
  onSerialize?: (session: Session) => void;
}

export function createSessionWriter(options: SessionWriterOptions) {
  const delayMs = options.delayMs ?? 1_000;
  const maxTotal = options.maxTotalChars ?? MAX_TOTAL_CHARS;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let pending: Session[] | null = null;
  // Chat object -> its JSON, at the normal field cap and at the shrink cap. WeakMap: a replaced chat is forgotten with its object.
  const cache = new WeakMap<Session, { full: string; shrunk: string | null }>();

  const entryOf = (s: Session) => {
    let entry = cache.get(s);
    if (entry === undefined) {
      options.onSerialize?.(s);
      entry = { full: serializeSession(s), shrunk: null };
      cache.set(s, entry);
    }
    return entry;
  };

  const flush = () => {
    if (timer !== null) { clearTimeout(timer); timer = null; }
    const sessions = pending;
    pending = null;
    if (!sessions) return;
    const parts = sessions.map(s => ({ s, entry: entryOf(s) }));
    let total = parts.reduce((n, p) => n + p.entry.full.length + 1, 2);
    if (total > maxTotal) {
      // Shorten the oversized fields of the oldest unpinned chats until the rest fits: the texts lose length, not
      // messages, and no chat is left out. Pinned and newest chats shrink last; the newest is shrunk only if needed.
      const order = parts
        .map((p, i) => ({ i, p }))
        .sort((a, b) => Number(!!a.p.s.pinned) - Number(!!b.p.s.pinned) || (a.p.s.updatedAt < b.p.s.updatedAt ? -1 : 1));
      for (const { i, p } of order) {
        if (total <= maxTotal) break;
        if (p.entry.shrunk === null) p.entry.shrunk = serializeSession(p.s, SHRINK_FIELD_CHARS);
        total -= p.entry.full.length - p.entry.shrunk.length;
      }
    }
    options.write(`[${parts.map(p => p.entry.shrunk ?? p.entry.full).join(',')}]`);
  };

  const schedule = (sessions: Session[]) => {
    pending = sessions;
    if (timer === null) timer = setTimeout(flush, delayMs);
  };

  return { schedule, flush, hasPending: () => pending !== null };
}
