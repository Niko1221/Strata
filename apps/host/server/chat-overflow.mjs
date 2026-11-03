/**
 * Chat-overflow guard (docs/DEV_BUILD.md, "Chat-Liste: nichts mehr still verlieren"): the page keeps its chat list in localStorage and
 * leaves the oldest chats out of the saved copy when the list grows past its cap (src/lib/session-persist.ts). Before a chat is left out,
 * the page sends it here, whole, and it is written to a file; only then may the page drop it. On in Strata Dev unless dev.config.json
 * switches "chatOverflowGuard" off; index.mjs does not route here otherwise.
 *
 *   POST /api/chat-overflow   { session }   writes overflow-<the chat's last change>-<id>.json (the same chat version once), answers { dir, file }
 *   GET  /api/chat-overflow                the files: name, bytes
 *
 * File names carry no title (a title can say what the chat is about); the title is only inside the file. Each file holds `[session]`,
 * the format the Dev panel's Rescue tab loads.
 */
import { existsSync, mkdirSync, readdirSync, renameSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

export const OVERFLOW_FILE = /^overflow-\d{8}-\d{6}-[A-Za-z0-9_-]+\.json$/;
/** One chat per request; a long chat with attachments can be tens of MB. */
const BODY_LIMIT = 64 * 1024 * 1024;

const stampOf = value => {
  const d = new Date(typeof value === 'string' ? value : NaN);
  const t = Number.isNaN(d.getTime()) ? new Date() : d;
  const p = n => String(n).padStart(2, '0');
  return `${t.getFullYear()}${p(t.getMonth() + 1)}${p(t.getDate())}-${p(t.getHours())}${p(t.getMinutes())}${p(t.getSeconds())}`;
};

/** The file name for one chat version: its last change and its id (only letters, digits, - and _). */
export function overflowFileName(session) {
  const id = String(session?.id ?? '').replace(/[^A-Za-z0-9_-]/g, '_').slice(0, 80) || 'chat';
  return `overflow-${stampOf(session?.updatedAt)}-${id}.json`;
}

/** Writes one chat; an existing file for the same version is kept as it is. Atomic: a temp file, then a rename. */
export function saveOverflow(dir, session) {
  if (!session || typeof session !== 'object' || Array.isArray(session) || typeof session.id !== 'string' || !session.id) throw new Error('A chat with an id is needed.');
  mkdirSync(dir, { recursive: true });
  const file = overflowFileName(session);
  const path = join(dir, file);
  if (existsSync(path)) return { dir, file, existed: true };
  const temp = `${path}.${process.pid}.tmp`;
  writeFileSync(temp, JSON.stringify([session]));
  renameSync(temp, path);
  return { dir, file, existed: false };
}

export function listOverflow(dir) {
  if (!existsSync(dir)) return [];
  return readdirSync(dir).filter(n => OVERFLOW_FILE.test(n)).sort().reverse().map(name => ({ name, bytes: statSync(join(dir, name)).size }));
}

export async function chatOverflowApi(req, res, path, method, { json, error, readBody, dir }) {
  if (path !== '/api/chat-overflow') return error(res, 404, `No route for ${method} ${path}`);
  if (method === 'GET') return json(res, 200, { dir, files: listOverflow(dir) });
  if (method !== 'POST') return error(res, 405, 'GET or POST.');
  let body;
  try { body = await readBody(req, BODY_LIMIT); } catch (e) { return error(res, 413, e.message); }
  try {
    return json(res, 200, saveOverflow(dir, body?.session));
  } catch (e) {
    return error(res, e.message === 'A chat with an id is needed.' ? 400 : 500, e.message);
  }
}
