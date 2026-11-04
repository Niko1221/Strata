import type { Session } from './types';

/**
 * Saving the chat list without hurting the page, and without losing any of it (#3604, #3603, #5192).
 *
 * Every streamed token used to stringify every chat and write the lot to localStorage. On a long history that is
 * megabytes of string work per token, and the WebView2 renderer grew to gigabytes before it died (the window goes
 * white). The write side keeps three rules from that fix:
 *
 *  1. Coalesce: the newest state wins and at most one write happens per `delayMs`. The caller flushes when a turn ends
 *     or the page goes away.
 *  2. Only the chats that changed are serialised again (a chat is replaced by a new object when it changes, the store
 *     never mutates), so the previous JSON of every other chat is reused by object identity.
 *  3. Only the chats that changed are written again: every chat has its own storage entry.
 *
 * The first version of rule 3 capped the saved copy and left text out of it (a long chat lost its oldest messages, an
 * old chat left the list, a long field was cut) on the theory that the saved list is a cache. It is not: the saved
 * list is the only place the chats are kept, and every restart trimmed them again (#5192). Now nothing is ever cut:
 *
 *  - a chat is saved exactly as it is (JSON.stringify of the chat, no clipping, no dropping of messages or chats);
 *  - when the browser refuses a write (storage full), the chat keeps its last good saved copy, stays in memory, stays
 *    marked as unsaved, is tried again on every later save, and the failure is reported through `onResult`, so the
 *    caller can say so and offer the unsaved chats (`unsaved()`, `exportSessions`);
 *  - the other chats are still saved: one chat that does not fit does not stop the rest.
 *
 * Layout: one entry per chat (`strata.session.v2.<id>`) and a small index (`strata.sessions.v2`) that only keeps the
 * order. The index is a hint: the chats are found by their own entries, so a lost index loses nothing.
 * The old single entry (`strata.sessions.v1`, the whole list in one string) is moved over chat by chat, and a chat is
 * removed from the old entry only after its new entry was written.
 */

/** The old whole-list entry (read for the migration, then emptied chat by chat). */
export const LEGACY_SESSIONS_KEY = 'strata.sessions.v1';
/** The order of the chats (a JSON object `{ v: 2, ids: string[] }`). */
export const SESSION_INDEX_KEY = 'strata.sessions.v2';
/** One chat per entry: this prefix plus the chat id. */
export const SESSION_KEY_PREFIX = 'strata.session.v2.';

/** The part of `Storage` that is used (a test passes a fake). */
export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
  key(index: number): string | null;
  readonly length: number;
}

/** The JSON text kept for one chat: the chat exactly as it is. */
export function serializeSession(session: Session): string {
  return JSON.stringify(session);
}

export interface SaveFailure {
  id: string;
  /** `quota` when the browser said the storage is full; `other` for any other refusal. */
  reason: 'quota' | 'other';
  message: string;
}

export interface SaveResult {
  ok: boolean;
  failed: SaveFailure[];
  /** How many chats were written by this save. */
  written: number;
}

function quotaError(error: unknown): boolean {
  const e = error as { name?: string; code?: number } | null;
  return !!e && (e.name === 'QuotaExceededError' || e.name === 'NS_ERROR_DOM_QUOTA_REACHED' || e.code === 22 || e.code === 1014);
}

const bodyKey = (id: string) => `${SESSION_KEY_PREFIX}${id}`;

function readIndex(storage: StorageLike): string[] {
  try {
    const raw = storage.getItem(SESSION_INDEX_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as { ids?: unknown };
    return Array.isArray(parsed.ids) ? parsed.ids.filter((id): id is string => typeof id === 'string') : [];
  } catch {
    return [];
  }
}

function legacyEntries(storage: StorageLike): { sessions: Session[]; ok: boolean } {
  let raw: string | null = null;
  try { raw = storage.getItem(LEGACY_SESSIONS_KEY); } catch { return { sessions: [], ok: false }; }
  if (!raw) return { sessions: [], ok: true };
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return { sessions: [], ok: false };
    const sessions: Session[] = [];
    parsed.forEach((entry, n) => {
      if (!entry || typeof entry !== 'object') return;
      const s = entry as Session;
      sessions.push(typeof s.id === 'string' && s.id ? s : { ...s, id: `legacy_${n}` });
    });
    return { sessions, ok: true };
  } catch {
    // Not readable as JSON: it is left exactly as it is, never rewritten or removed.
    return { sessions: [], ok: false };
  }
}

export interface LoadedSessions {
  sessions: Session[];
  /** Entries that could not be read. They are left in storage untouched. */
  damaged: string[];
  /** Chats still only in the old entry because the new entry could not be written (storage full). */
  unmigrated: number;
}

/**
 * Every saved chat, newest layout first and the old single entry second.
 * The old entry is moved over: each chat is written to its own entry and only then taken out of the old one.
 */
export function loadStoredSessions(storage: StorageLike): LoadedSessions {
  const damaged: string[] = [];
  const legacy = legacyEntries(storage);
  let remaining = legacy.sessions;
  if (legacy.ok && remaining.length) {
    const left: Session[] = [];
    for (let i = 0; i < remaining.length; i++) {
      const s = remaining[i];
      let have = false;
      try { have = storage.getItem(bodyKey(s.id)) !== null; } catch { /* treated as not there */ }
      if (!have) {
        try { storage.setItem(bodyKey(s.id), serializeSession(s)); } catch { left.push(s); continue; }
      }
      // Written (or already there): this chat can leave the old entry.
      try {
        const keep = [...left, ...remaining.slice(i + 1)];
        if (keep.length) storage.setItem(LEGACY_SESSIONS_KEY, JSON.stringify(keep)); else storage.removeItem(LEGACY_SESSIONS_KEY);
      } catch { /* the old entry stays as it was: the chat is in both places, which is safe */ }
    }
    remaining = left;
  }

  const found = new Map<string, Session>();
  try {
    const keys: string[] = [];
    for (let i = 0; i < storage.length; i++) {
      const key = storage.key(i);
      if (key && key.startsWith(SESSION_KEY_PREFIX)) keys.push(key);
    }
    for (const key of keys) {
      try {
        const parsed = JSON.parse(storage.getItem(key) ?? 'null') as Session | null;
        if (parsed && typeof parsed === 'object' && typeof parsed.id === 'string') found.set(parsed.id, parsed);
        else damaged.push(key);
      } catch {
        damaged.push(key);
      }
    }
  } catch { /* storage not readable: nothing to load */ }
  // Chats that are only in the old entry (storage was full during the move) are still shown.
  for (const s of remaining) if (!found.has(s.id)) found.set(s.id, s);

  const order = readIndex(storage);
  const rank = new Map(order.map((id, i) => [id, i]));
  const sessions = [...found.values()].sort((a, b) => {
    const ra = rank.get(a.id), rb = rank.get(b.id);
    if (ra !== undefined && rb !== undefined) return ra - rb;
    if (ra !== undefined) return -1;
    if (rb !== undefined) return 1;
    return String(b.createdAt ?? '') < String(a.createdAt ?? '') ? -1 : 1;
  });
  return { sessions, damaged, unmigrated: remaining.length };
}

/** The chats as one JSON text, for saving a copy by hand when the browser storage refuses them. */
export function exportSessions(sessions: Session[]): string {
  return JSON.stringify(sessions);
}

export interface SessionWriterOptions {
  storage: StorageLike;
  delayMs?: number;
  /** Counts a serialisation of one chat; for tests. */
  onSerialize?: (session: Session) => void;
  /** Called after every save, with what was written and which chats the browser refused. */
  onResult?: (result: SaveResult) => void;
  /** Ids of the chats that were loaded, so that one removed later is also removed from storage. */
  knownIds?: Iterable<string>;
}

export function createSessionWriter(options: SessionWriterOptions) {
  const storage = options.storage;
  const delayMs = options.delayMs ?? 1_000;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let pending: Session[] | null = null;
  // Chat object -> its JSON. WeakMap: a replaced chat is forgotten with its object.
  const cache = new WeakMap<Session, string>();
  // The JSON of each chat as last written (or confirmed already in storage).
  const stored = new Map<string, string>();
  // Ids this writer has seen in the list; only these are ever removed from storage.
  const known = new Set<string>(options.knownIds ?? []);
  let lastIndex = '';
  let failedIds = new Set<string>();
  let lastSessions: Session[] = [];
  let checkLegacy = true;

  const textOf = (s: Session): string => {
    let text = cache.get(s);
    if (text === undefined) {
      options.onSerialize?.(s);
      text = serializeSession(s);
      cache.set(s, text);
    }
    return text;
  };

  const flush = (): SaveResult => {
    if (timer !== null) { clearTimeout(timer); timer = null; }
    const sessions = pending;
    pending = null;
    if (!sessions) return { ok: failedIds.size === 0, failed: [], written: 0 };
    lastSessions = sessions;
    const ids = new Set(sessions.map(s => s.id));

    // Removed chats first: it frees room for the ones that follow.
    for (const id of [...known]) {
      if (ids.has(id)) continue;
      try { storage.removeItem(bodyKey(id)); known.delete(id); stored.delete(id); failedIds.delete(id); } catch { /* tried again at the next save */ }
    }

    const failed: SaveFailure[] = [];
    let written = 0;
    for (const s of sessions) {
      known.add(s.id);
      const text = textOf(s);
      let current = stored.get(s.id);
      if (current === undefined) {
        try { current = storage.getItem(bodyKey(s.id)) ?? undefined; } catch { current = undefined; }
        if (current !== undefined) stored.set(s.id, current);
      }
      if (current === text) { failedIds.delete(s.id); continue; }
      try {
        storage.setItem(bodyKey(s.id), text);
        stored.set(s.id, text);
        failedIds.delete(s.id);
        written++;
      } catch (error) {
        // The last good copy of this chat stays in storage; the chat stays in memory and is tried again next time.
        failedIds.add(s.id);
        failed.push({ id: s.id, reason: quotaError(error) ? 'quota' : 'other', message: error instanceof Error ? error.message : String(error) });
      }
    }

    const index = JSON.stringify({ v: 2, ids: sessions.map(s => s.id) });
    if (index !== lastIndex) {
      try { storage.setItem(SESSION_INDEX_KEY, index); lastIndex = index; } catch { /* only the order; the chats are found by their own entries */ }
    }

    // The old whole-list entry goes once every chat in it has its own entry.
    if (checkLegacy && failed.length === 0 && failedIds.size === 0) {
      try {
        const old = storage.getItem(LEGACY_SESSIONS_KEY);
        if (old === null) checkLegacy = false;
        else {
          const legacy = legacyEntries(storage);
          if (!legacy.ok) checkLegacy = false; // not readable: left exactly as it is
          else if (legacy.sessions.every(l => stored.has(l.id) || storage.getItem(bodyKey(l.id)) !== null)) {
            storage.removeItem(LEGACY_SESSIONS_KEY);
            checkLegacy = false;
          }
        }
      } catch { /* kept */ }
    }

    const result: SaveResult = { ok: failedIds.size === 0, failed, written };
    options.onResult?.(result);
    return result;
  };

  const schedule = (sessions: Session[]) => {
    pending = sessions;
    if (timer === null) timer = setTimeout(flush, delayMs);
  };

  return {
    schedule,
    flush,
    hasPending: () => pending !== null,
    /** The chats whose newest state is not in storage (the browser refused the write). */
    unsaved: (): Session[] => lastSessions.filter(s => failedIds.has(s.id)),
  };
}
