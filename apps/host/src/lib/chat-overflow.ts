import { api } from './client';
import type { OverflowGuard } from './session-persist';
import type { Session } from './types';

/**
 * Chat-overflow guard: a chat leaves the saved chat list (session-persist.ts, capped at MAX_TOTAL_CHARS) only after the host has written
 * it whole to a file (server/chat-overflow.mjs, POST /api/chat-overflow). Until then it stays in the saved copy. A failed backup is a
 * warning and the chat stays; it is tried again after RETRY_MS. Off until configureChatOverflow() turns it on (Strata Dev:
 * dev.config.json "chatOverflowGuard", on by default there); off, the writer behaves as it always did.
 */

export interface ChatOverflowSettings { enabled: boolean; maxTotalChars?: number }
export interface ChatOverflowHandlers {
  /** Chats were backed up: the notice ("n older chats were moved out, saved under …"). */
  saved: (count: number, dir: string) => void;
  /** A backup failed; the chats stay in the list. */
  failed: (message: string) => void;
  /** Write the chat list again, now that chats may leave it. */
  rewrite: () => void;
}
type Post = (session: Session) => Promise<{ dir: string; file: string; existed?: boolean }>;

export const RETRY_MS = 5 * 60_000;

/** The guard for one page. `post` sends one chat to the host; `now` is for tests. */
export function createOverflowGuard({ handlers, post, maxTotalChars, now = Date.now }: { handlers: ChatOverflowHandlers; post: Post; maxTotalChars?: number; now?: () => number }): OverflowGuard {
  // A chat version is "<id>@<updatedAt>": a chat that changed after its backup is backed up again before it may leave.
  const key = (s: Session) => `${s.id}@${s.updatedAt}`;
  const saved = new Set<string>();
  const inFlight = new Set<string>();
  const failedAt = new Map<string, number>();
  return {
    ...(maxTotalChars ? { maxTotalChars } : {}),
    isBackedUp: s => saved.has(key(s)),
    backUp: sessions => {
      const todo = sessions.filter(s => {
        const k = key(s);
        const failed = failedAt.get(k);
        return !saved.has(k) && !inFlight.has(k) && (failed === undefined || now() - failed >= RETRY_MS);
      });
      if (!todo.length) return;
      for (const s of todo) inFlight.add(key(s));
      void (async () => {
        let count = 0, dir = '', error = '';
        // One at a time: each chat is a request of its own, so one huge chat never holds the others back in one body.
        for (const s of todo) {
          const k = key(s);
          try {
            const r = await post(s);
            saved.add(k); failedAt.delete(k); count++; dir = r.dir;
          } catch (e) {
            failedAt.set(k, now());
            error ||= e instanceof Error ? e.message : String(e);
          } finally {
            inFlight.delete(k);
          }
        }
        if (count) { handlers.saved(count, dir); handlers.rewrite(); }
        if (error) handlers.failed(error);
      })();
    },
  };
}

let handlers: ChatOverflowHandlers | null = null;
let active: OverflowGuard | null = null;

/** store.ts: what a backup shows and how the list is written again. */
export function setChatOverflowHandlers(next: ChatOverflowHandlers) { handlers = next; }

/** Turns the guard on (settings.enabled) or off (null). dev.tsx calls it once the host said it runs as Strata Dev. */
export function configureChatOverflow(settings: ChatOverflowSettings | null | undefined) {
  active = settings?.enabled && handlers
    ? createOverflowGuard({ handlers, maxTotalChars: settings.maxTotalChars, post: session => api.post('/api/chat-overflow', { session }) })
    : null;
}

/** The guard the chat-list writer asks at every write; null when off. */
export const chatOverflowGuard = (): OverflowGuard | null => active;
