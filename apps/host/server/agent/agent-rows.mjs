/**
 * The helper rows of every chat, kept on disk.
 *
 * A helper (subagent) row is the page's only record of that helper: its name, what it did, whether it is
 * still working, how long it ran and how many tokens it used. The bridge used to keep those rows in memory
 * only, so a host restart erased them and the page had to guess which saved "working" row was a ghost
 * (#4218). This store makes the host the single owner: every row the bridge sees is written here, a restart
 * seals the rows that were still open, and the page reads the whole set instead of reconciling a guess.
 *
 * The file is replaced atomically. The previous good copy stays beside it as `<file>.bak`; a file that
 * cannot be parsed is moved aside (never deleted) and the backup is read instead. If neither can be read the
 * store reports `authoritative: false`, which tells the page to keep its own rows rather than drop them.
 */
import { copyFileSync, existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { dirname } from 'node:path';

const VERSION = 1;
const STATUSES = new Set(['working', 'completed', 'failed', 'queued', 'cancelled', 'interrupted']);
/** The chat mode pinned when the helper started; absent on rows written before modes were recorded. */
const MODES = new Set(['manual', 'plan', 'bypass']);
const FINAL = new Set(['completed', 'failed', 'cancelled', 'interrupted']);
const LIVE = new Set(['working', 'queued']);
export const MAX_ROWS_PER_CHAT = 500;
export const MAX_CHATS = 2000;

const clip = (value, max) => String(value ?? '').replace(/\s+/g, ' ').trim().slice(0, max);
const count = (value, fallback = 0) => (typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : fallback);
/** The Harness's own settled time for a helper, when its projection cache still has one. */
const settledOf = (repair, id) => {
  try {
    const ms = repair?.(id);
    return typeof ms === 'number' && Number.isFinite(ms) && ms >= 0 ? ms : null;
  } catch {
    return null;
  }
};

function normalize(raw, previous, now) {
  if (!raw || typeof raw !== 'object') return null;
  const id = clip(raw.id, 200);
  if (!id) return null;
  const status = STATUSES.has(raw.status) ? raw.status : previous?.status ?? 'queued';
  const reported = count(raw.updatedAt, previous?.updatedAt ?? now);
  // Timestamps come from this host. An imported page row may carry a clock that is ahead of it: clamp it so
  // one bad clock cannot make a row win every later merge.
  const updatedAt = Math.min(reported, now);
  const startedAt = Math.min(updatedAt, count(raw.startedAt, previous?.startedAt ?? updatedAt));
  const final = FINAL.has(status);
  let durationMs = count(raw.durationMs, previous?.durationMs ?? 0);
  const endedAt = final
    ? Math.min(now, Math.max(startedAt, count(raw.endedAt, previous?.endedAt ?? startedAt + durationMs)))
    : null;
  if (final && raw.durationMs == null) durationMs = Math.max(durationMs, Math.max(0, (endedAt ?? updatedAt) - startedAt));
  const description = typeof raw.description === 'string' ? clip(raw.description, 400) : previous?.description ?? '';
  const descriptionKey = typeof raw.descriptionKey === 'string' && raw.descriptionKey.trim()
    ? clip(raw.descriptionKey, 200)
    : (typeof raw.description === 'string' && raw.description !== previous?.description ? null : previous?.descriptionKey ?? null);
  const warning = LIVE.has(status)
    ? null
    : (typeof raw.warning === 'string' && raw.warning.trim() ? clip(raw.warning, 300) : previous?.warning ?? null);
  const mode = MODES.has(raw.mode) ? raw.mode : previous?.mode ?? null;
  return {
    id,
    name: clip(raw.name, 120) || previous?.name || 'Helper',
    description,
    ...(descriptionKey ? { descriptionKey } : {}),
    ...(mode ? { mode } : {}),
    status,
    durationMs,
    tokens: count(raw.tokens, previous?.tokens ?? 0),
    startedAt,
    endedAt,
    updatedAt,
    ...(warning ? { warning } : {}),
  };
}

function readFile(file, now) {
  const raw = JSON.parse(readFileSync(file, 'utf8'));
  if (!raw || typeof raw !== 'object' || !raw.chats || typeof raw.chats !== 'object' || Array.isArray(raw.chats)) {
    throw new Error('the agent row file has an unknown shape');
  }
  const chats = {};
  for (const [chatId, rows] of Object.entries(raw.chats)) {
    if (!chatId || !Array.isArray(rows)) continue;
    chats[chatId] = rows.map(row => normalize(row, null, now)).filter(Boolean).slice(-MAX_ROWS_PER_CHAT);
  }
  return chats;
}

/**
 * @param {{file:string, now?:()=>number, log?:(line:string)=>void, maxPerChat?:number, maxChats?:number}} options
 */
export function createAgentRowStore({ file, now = Date.now, log = () => {}, maxPerChat = MAX_ROWS_PER_CHAT, maxChats = MAX_CHATS }) {
  let chats = {};
  let authoritative = true;
  let dirty = false;
  let timer = null;

  if (existsSync(file)) {
    try {
      chats = readFile(file, now());
    } catch (error) {
      const aside = `${file}.corrupt-${Date.now()}`;
      try { renameSync(file, aside); } catch { /* keep going: the backup is the next try */ }
      log(`agent · the helper row file could not be read (${error.message}); kept it as ${aside}`);
      try {
        if (!existsSync(`${file}.bak`)) throw error;
        chats = readFile(`${file}.bak`, now());
        log('agent · helper rows were restored from the backup copy');
      } catch {
        chats = {};
        authoritative = false;
        log('agent · no readable helper row copy is left; saved rows are kept until the host can take them');
      }
    }
  }

  function writeNow() {
    if (!dirty) return;
    dirty = false;
    try {
      mkdirSync(dirname(file), { recursive: true });
      const temp = `${file}.${randomUUID()}.tmp`;
      writeFileSync(temp, JSON.stringify({ version: VERSION, chats }, null, 2));
      try { if (existsSync(file)) copyFileSync(file, `${file}.bak`); } catch { /* a backup is a courtesy, not a condition */ }
      renameSync(temp, file);
    } catch (error) {
      dirty = true;
      log(`agent · helper rows could not be saved: ${error.message}`);
    }
  }

  function schedule(immediate = false) {
    dirty = true;
    if (immediate) {
      if (timer) { clearTimeout(timer); timer = null; }
      writeNow();
      return;
    }
    if (timer) return;
    timer = setTimeout(() => { timer = null; writeNow(); }, 250);
    timer.unref?.();
  }

  function trimChat(chatId) {
    const rows = chats[chatId];
    if (!rows || rows.length <= maxPerChat) return;
    const live = rows.filter(row => LIVE.has(row.status));
    const final = rows.filter(row => !LIVE.has(row.status)).sort((a, b) => (a.updatedAt ?? 0) - (b.updatedAt ?? 0));
    const keep = new Set(final.slice(Math.max(0, final.length - Math.max(0, maxPerChat - live.length))).map(row => row.id));
    chats[chatId] = rows.filter(row => LIVE.has(row.status) || keep.has(row.id));
  }

  function trimChats() {
    const ids = Object.keys(chats);
    if (ids.length <= maxChats) return;
    const ranked = ids.map(id => ({ id, at: Math.max(0, ...(chats[id] ?? []).map(row => row.updatedAt ?? 0)) }))
      .sort((a, b) => b.at - a.at);
    const keep = new Set(ranked.slice(0, maxChats).map(item => item.id));
    for (const id of ids) {
      if (keep.has(id)) continue;
      if ((chats[id] ?? []).some(row => LIVE.has(row.status))) continue;
      delete chats[id];
    }
  }

  const rowsOf = chatId => chats[chatId] ?? [];

  /** Add or update one row. A final row never goes back to live on an older frame; a newer frame may restart it. */
  function upsert(chatId, raw) {
    if (!chatId || typeof chatId !== 'string') return null;
    const at = now();
    const previous = rowsOf(chatId).find(row => row?.id === raw?.id) ?? null;
    const incoming = normalize(raw, previous, at);
    if (!incoming) return previous;
    if (previous && incoming.updatedAt < (previous.updatedAt ?? 0)) return previous;
    if (previous && FINAL.has(previous.status) && LIVE.has(incoming.status) && incoming.updatedAt <= (previous.updatedAt ?? 0)) return previous;
    const rows = previous ? rowsOf(chatId).map(row => (row.id === incoming.id ? incoming : row)) : [...rowsOf(chatId), incoming];
    chats[chatId] = rows;
    trimChat(chatId);
    trimChats();
    schedule(FINAL.has(incoming.status));
    return incoming;
  }

  /** Attach a note (a failed interrupt, for example) without moving the row's clock. */
  function annotate(chatId, id, patch) {
    const row = rowsOf(chatId).find(item => item.id === id);
    if (!row) return null;
    const next = {
      ...row,
      ...(typeof patch.warning === 'string' && patch.warning.trim() ? { warning: clip(patch.warning, 300) } : {}),
      ...(typeof patch.descriptionKey === 'string' && patch.descriptionKey.trim() ? { descriptionKey: clip(patch.descriptionKey, 200) } : {}),
    };
    chats[chatId] = rowsOf(chatId).map(item => (item.id === id ? next : item));
    schedule(true);
    return next;
  }

  /** Rows oldest first, so the panel keeps the order the work happened in. */
  function all(chatId) {
    return rowsOf(chatId).map(row => ({ ...row })).sort((a, b) => (a.startedAt ?? 0) - (b.startedAt ?? 0) || (a.updatedAt ?? 0) - (b.updatedAt ?? 0));
  }

  function snapshot(chatId) {
    return { agents: all(chatId), authoritative };
  }

  /**
   * Take rows a page saved before this host owned them. The host wins on every row it already has; an
   * imported live row is sealed unless the host is really following it right now.
   */
  function importRows(chatId, list, { liveIds = new Set(), repair = null } = {}) {
    if (!chatId || !Array.isArray(list) || !list.length) return 0;
    const at = now();
    const known = new Set(rowsOf(chatId).map(row => row.id));
    let added = 0;
    for (const raw of list.slice(0, maxPerChat)) {
      const row = normalize(raw, null, at);
      if (!row || known.has(row.id)) continue;
      if (LIVE.has(row.status) && !liveIds.has(row.id)) {
        row.status = 'interrupted';
        row.endedAt = row.startedAt + row.durationMs;
        row.updatedAt = at;
      }
      if (FINAL.has(row.status)) {
        // The page wrote "Stopped." for a cancelled helper before this version. Say it in a sentence the
        // interface can show in the person's language, the same way a new row does.
        if (row.description === 'Stopped.') {
          row.description = row.status === 'cancelled' ? 'Cancelled before it finished.' : 'Interrupted before it finished.';
          row.descriptionKey = row.description;
        }
        // A page row that was left live kept counting while the app was closed (the #4218 ghost duration).
        // The Harness's own settled time is the truth; use it when its cache still has one.
        if (row.status !== 'completed') {
          const settled = settledOf(repair, row.id);
          if (settled != null) { row.durationMs = settled; row.endedAt = row.startedAt + settled; }
        }
      }
      chats[chatId] = [...rowsOf(chatId), row];
      known.add(row.id);
      added += 1;
    }
    if (!added) return 0;
    trimChat(chatId);
    trimChats();
    schedule(true);
    return added;
  }

  /**
   * Seal every row a dead host left open. `repair(id)` may return the Harness's own settled duration for a
   * helper; it replaces a smaller reading so a restart shows the work that really happened, not the gap.
   */
  function sealOpen(chatId = null, { repair = null } = {}) {
    const ids = chatId ? [chatId] : Object.keys(chats);
    let sealed = 0;
    for (const id of ids) {
      const rows = rowsOf(id);
      if (!rows.some(row => LIVE.has(row.status))) continue;
      const at = now();
      chats[id] = rows.map(row => {
        if (!LIVE.has(row.status)) return row;
        sealed += 1;
        const duration = Math.max(0, row.durationMs ?? 0);
        const reported = settledOf(repair, row.id);
        const settled = reported != null && reported > duration ? reported : duration;
        return {
          ...row,
          status: 'interrupted',
          durationMs: settled,
          endedAt: settled ? row.startedAt + settled : at,
          updatedAt: at,
        };
      });
    }
    if (sealed) schedule(true);
    return sealed;
  }

  /**
   * Fix the two things an older build could leave on a finished row: a duration the page kept counting while
   * the app was closed (#4218), and the bare "Stopped." wording that has no key in the locale files. The
   * Harness cache has the real run time. Completed rows already carry the final reading, so their time stays.
   */
  function repairFinished({ repair = null } = {}) {
    let fixed = 0;
    let changedAny = false;
    for (const id of Object.keys(chats)) {
      let changed = false;
      chats[id] = rowsOf(id).map(row => {
        let next = row;
        if (FINAL.has(next.status) && next.description === 'Stopped.') {
          const description = next.status === 'cancelled' ? 'Cancelled before it finished.' : 'Interrupted before it finished.';
          next = { ...next, description, descriptionKey: description };
          changed = true;
          fixed += 1;
        }
        if (next.status === 'completed' || LIVE.has(next.status)) return next;
        const settled = settledOf(repair, next.id);
        if (settled == null || settled === next.durationMs) return next;
        changed = true;
        fixed += 1;
        return { ...next, durationMs: settled, endedAt: next.startedAt + settled };
      });
      if (changed) changedAny = true;
    }
    if (changedAny) schedule(true);
    return fixed;
  }

  function remove(chatId) {
    if (!chatId || !(chatId in chats)) return false;
    delete chats[chatId];
    schedule(true);
    return true;
  }

  function flush() {
    if (timer) { clearTimeout(timer); timer = null; }
    writeNow();
  }

  function close() {
    flush();
  }

  return {
    upsert, annotate, all, snapshot, import: importRows, sealOpen, repairFinished, remove, flush, close,
    get authoritative() { return authoritative; },
    get size() { return Object.values(chats).reduce((total, rows) => total + rows.length, 0); },
    _debug: () => ({ chats, file }),
  };
}
