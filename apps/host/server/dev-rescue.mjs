/**
 * Strata Dev (docs/DEV_BUILD.md): the chats dev/tools/rescue-chats.mjs rebuilt, for the Dev panel's "Rescue" tab. Only with STRATA_DEV=1.
 * Read only: the files in the rescue folder. The page adds them to Strata Dev's own chat list; nothing here touches Live.
 *
 * The folder: STRATA_DEV_RESCUE, else %LOCALAPPDATA%\StrataDev\rescue, else "rescue" beside STRATA_HOME (Strata Dev's home is
 * %LOCALAPPDATA%\StrataDev\home). Every candidate is listed with what was found there, so an empty list says why.
 *
 *   GET /api/dev/rescue              the import files: name, size, chats, workspaces; and the folders looked at
 *   GET /api/dev/rescue/<file>       one import file (rescued-chats-import-*.json, or overflow-*.json from the chat-overflow guard)
 *
 * `extraDirs` (index.mjs): the chat-overflow folder (server/chat-overflow.mjs), listed after the rescue folders.
 */
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

const IMPORT = /^(rescued-chats-import-[0-9-]+|overflow-\d{8}-\d{6}-[A-Za-z0-9_-]+)\.json$/i;

export function rescueDirs(env = process.env, extraDirs = []) {
  const dirs = [];
  if (env.STRATA_DEV_RESCUE) dirs.push(resolve(env.STRATA_DEV_RESCUE));
  if (env.LOCALAPPDATA) dirs.push(resolve(env.LOCALAPPDATA, 'StrataDev', 'rescue'));
  if (env.STRATA_HOME) dirs.push(resolve(dirname(resolve(env.STRATA_HOME)), 'rescue'));
  // A tool run from the Claude desktop app (a Store app) has its AppData writes redirected into its package folder, which other
  // programs do not see; the rescue files are therefore also looked for in a folder under Documents.
  if (env.USERPROFILE) dirs.push(resolve(env.USERPROFILE, 'Documents', 'Strata-Rescue-Backup'));
  for (const d of extraDirs) dirs.push(resolve(d));
  return [...new Set(dirs.map(d => d.toLowerCase()))].map(low => dirs.find(d => d.toLowerCase() === low));
}
export const rescueDir = (env = process.env) => rescueDirs(env)[0] ?? resolve('rescue');

/** What is in one folder: the import files, or why there are none. */
export function scan(dir) {
  const report = { dir, exists: false, entries: 0, error: null, files: [] };
  try {
    report.exists = existsSync(dir);
    if (!report.exists) return report;
    const names = readdirSync(dir);
    report.entries = names.length;
    for (const name of names.filter(n => IMPORT.test(n)).sort().reverse()) {
      const path = join(dir, name);
      let chats = null;
      try { chats = JSON.parse(readFileSync(path, 'utf8')); } catch (e) { report.error = `${name}: ${e.message}`; }
      const workspaces = {};
      for (const c of Array.isArray(chats) ? chats : []) workspaces[c.workspace] = (workspaces[c.workspace] ?? 0) + 1;
      report.files.push({ name, dir, bytes: statSync(path).size, chats: Array.isArray(chats) ? chats.length : null, workspaces });
    }
  } catch (e) { report.error = e.message; }
  return report;
}

export function rescueFiles(env = process.env, extraDirs = []) {
  return rescueDirs(env, extraDirs).map(scan);
}

export async function devRescueApi(req, res, path, method, url, { json, error, extraDirs = [] }) {
  if (!path.startsWith('/api/dev/rescue')) return undefined;
  if (method !== 'GET') return error(res, 405, 'Read only.');
  const scans = rescueFiles(process.env, extraDirs);
  if (path === '/api/dev/rescue') return json(res, 200, { dir: rescueDir(), folders: scans.map(({ files, ...s }) => ({ ...s, files: files.length })), files: scans.flatMap(s => s.files) });
  const name = decodeURIComponent(path.slice('/api/dev/rescue/'.length));
  if (!IMPORT.test(name)) return error(res, 400, 'Only rescued-chats-import-*.json and overflow-*.json files.');
  const hit = scans.flatMap(s => s.files).find(f => f.name === name);
  if (!hit) return error(res, 404, `${name} was not found in ${scans.map(s => s.dir).join(' or ')}.`);
  res.writeHead(200, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' });
  res.end(readFileSync(join(hit.dir, name)));
  return true;
}
