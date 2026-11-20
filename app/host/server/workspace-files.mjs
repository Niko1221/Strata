/** Explicit workspace file operations. Deletion preserves the item in that workspace's trash. */
import { createHash, randomUUID } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { closeSync, existsSync, fsyncSync, lstatSync, mkdirSync, openSync, readFileSync, readdirSync, realpathSync, renameSync, unlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, isAbsolute, join, toNamespacedPath } from 'node:path';
import { isInside, openFile, refusedFolder } from './chat-files.mjs';
import { makePrivateDir } from './private-file.mjs';

export const WORKSPACE_TEXT_LIMIT = 200 * 1024;
const TRASH = '.strata-trash';
// Keep device names out of paths before Windows can interpret them as devices.
const WINDOWS_RESERVED = new Set(['CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$',
  ...Array.from('123456789¹²³', digit => `COM${digit}`), ...Array.from('123456789¹²³', digit => `LPT${digit}`)]);
const fail = (status, message) => { throw Object.assign(new Error(message), { status }); };
const revision = st => createHash('sha256').update(`${st.dev}:${st.ino}:${st.size}:${st.mtimeNs}:${st.ctimeNs}`).digest('hex');

function parts(value, allowRoot = false) {
  if (typeof value !== 'string' || value.length > 2048 || /[\x00-\x1f\x7f:]/.test(value) || isAbsolute(value)) fail(400, 'Choose a relative workspace path.');
  const names = value.replaceAll('\\', '/').split('/');
  if (allowRoot && value === '') return [];
  if (names.some(name => !name || name === '.' || name === '..' || /[. ]$/.test(name) || ['.git', TRASH].includes(name.toLowerCase())
      || WINDOWS_RESERVED.has(name.split('.')[0].replace(/ +$/, '').toUpperCase()))) fail(400, 'Choose a file inside this workspace.');
  return names;
}

function locate(root, value, { allowRoot = false, create = false } = {}) {
  const names = parts(value, allowRoot);
  const base = realpathSync(root);
  if (refusedFolder(base)) fail(403, 'Choose one of your own workspace folders.');
  if (!lstatSync(base).isDirectory()) fail(400, 'This workspace folder is unavailable.');
  let path = base;
  for (let i = 0; i < names.length; i++) {
    path = join(path, names[i]);
    if (create && i === names.length - 1) break;
    let st;
    try { st = lstatSync(path); } catch (error) { if (error.code === 'ENOENT') fail(404, 'That item is no longer there.'); throw error; }
    if (st.isSymbolicLink() || !isInside(base, realpathSync(path))) fail(403, 'Linked files and folders cannot be changed here.');
    if (i < names.length - 1 && !st.isDirectory()) fail(400, 'Choose an existing parent folder.');
  }
  if (!isInside(base, path)) fail(403, 'That item is outside this workspace.');
  return { base, path, name: names.join('/') };
}

function checked(root, path, expected) {
  const item = locate(root, path);
  const st = lstatSync(item.path, { bigint: true });
  if (!st.isFile() && !st.isDirectory()) fail(400, 'Choose a regular file or folder.');
  if (typeof expected !== 'string' || revision(st) !== expected) fail(409, 'This item changed on disk. Refresh it before trying again.');
  return { ...item, st };
}

function textBytes(text) {
  if (typeof text !== 'string') fail(400, 'File text is required.');
  const bytes = Buffer.from(text, 'utf8');
  if (bytes.length > WORKSPACE_TEXT_LIMIT) fail(413, 'This file is too large to edit here.');
  if (bytes.includes(0)) fail(400, 'Use a text file.');
  return bytes;
}

function existingEntry(path) {
  try { return lstatSync(path, { bigint: true }); }
  catch (error) { if (error.code === 'ENOENT') return null; throw error; }
}

export function listWorkspaceFiles(root, path = '', offset = 0) {
  if (!Number.isSafeInteger(offset) || offset < 0) fail(400, 'Choose a valid file-list page.');
  const item = locate(root, path, { allowRoot: true });
  if (!lstatSync(item.path).isDirectory()) fail(400, 'Choose a folder.');
  const names = readdirSync(item.path, { withFileTypes: true }).filter(entry => !['.git', TRASH].includes(entry.name.toLowerCase()));
  names.sort((a, b) => Number(b.isDirectory()) - Number(a.isDirectory()) || a.name.localeCompare(b.name));
  const entries = [];
  for (const entry of names.slice(offset, offset + 200)) {
    const full = join(item.path, entry.name);
    try {
      const st = lstatSync(full, { bigint: true });
      if (st.isSymbolicLink() || !st.isFile() && !st.isDirectory()) continue;
      entries.push({ name: entry.name, path: path ? `${path}/${entry.name}` : entry.name,
        kind: st.isDirectory() ? 'dir' : 'file', sizeKb: Number(st.size) / 1024,
        updatedAt: new Date(Number(st.mtimeMs)).toISOString(), revision: revision(st) });
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  return { path: item.name, entries, nextOffset: offset + 200 < names.length ? offset + 200 : null };
}

export function readWorkspaceFile(root, path) {
  const item = locate(root, path);
  const st = lstatSync(item.path, { bigint: true });
  if (!st.isFile()) fail(400, 'Choose a file, not a folder.');
  const result = { path: item.name, revision: revision(st), sizeKb: Number(st.size) / 1024,
    updatedAt: new Date(Number(st.mtimeMs)).toISOString(), text: '' };
  if (st.size > BigInt(WORKSPACE_TEXT_LIMIT)) return { ...result, tooLarge: true };
  const bytes = readFileSync(item.path);
  try {
    if (bytes.includes(0)) return { ...result, binary: true };
    return { ...result, text: new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes) };
  } catch { return { ...result, binary: true }; }
}

export function createWorkspaceFile(root, { path, text = '', kind = 'file' }) {
  if (!['file', 'dir'].includes(kind)) fail(400, 'Choose a file or folder.');
  const item = locate(root, path, { create: true });
  if (kind === 'dir') mkdirSync(item.path);
  else writeFileSync(item.path, textBytes(text), { flag: 'wx', mode: 0o600 });
  return { ok: true, path: item.name };
}

export function updateWorkspaceFile(root, { path, text, revision: expected }) {
  const item = checked(root, path, expected);
  const previous = readWorkspaceFile(root, path);
  if (previous.binary || previous.tooLarge) fail(400, 'This file cannot be edited as text.');
  let next = text;
  if (typeof next === 'string' && previous.text.includes('\r\n') && !/(?<!\r)\n/.test(previous.text)) next = next.replace(/\r?\n/g, '\r\n');
  if (typeof next === 'string' && previous.text.startsWith('\uFEFF') && !next.startsWith('\uFEFF')) next = '\uFEFF' + next;
  const bytes = textBytes(next);
  const trash = trashDirectory(item.base), id = randomUUID();
  const temporary = join(trash, `${id}.tmp`), backup = join(trash, id);
  const original = readFileSync(item.path);
  writeFileSync(join(trash, `${id}.json`), JSON.stringify({ path: item.name, revision: expected, kind: 'edit-backup', deletedAt: new Date().toISOString() }), { flag: 'wx', mode: 0o600 });
  writeFileSync(backup, original, { flag: 'wx', mode: 0o600 });
  let fd;
  let completed = false;
  try {
    fd = openSync(temporary, 'wx', Number(item.st.mode) & 0o777);
    writeFileSync(fd, bytes); fsyncSync(fd); closeSync(fd); fd = undefined;
    checked(root, path, expected);
    if (process.platform === 'win32') {
      // ReplaceFile preserves destination ACLs/streams; never ignore metadata merge failures.
      const script = "$ErrorActionPreference='Stop';$h=[Security.Cryptography.SHA256]::Create();try{$d=[BitConverter]::ToString($h.ComputeHash([IO.File]::ReadAllBytes($env:STRATA_WS_TARGET))).Replace('-','').ToLowerInvariant()}finally{$h.Dispose()};if($d -ne $env:STRATA_WS_DIGEST){exit 23};[IO.File]::Replace($env:STRATA_WS_TEMP,$env:STRATA_WS_TARGET,$env:STRATA_WS_BACKUP,$false)";
      const result = spawnSync(join(process.env.SystemRoot || 'C:\\Windows', 'System32/WindowsPowerShell/v1.0/powershell.exe'), ['-NoProfile', '-NonInteractive', '-Command', script], {
        env: { ...process.env, STRATA_WS_TEMP: toNamespacedPath(temporary), STRATA_WS_TARGET: toNamespacedPath(item.path), STRATA_WS_BACKUP: toNamespacedPath(backup),
          STRATA_WS_DIGEST: createHash('sha256').update(original).digest('hex'),
          PSModuleAnalysisCachePath: process.env.PSModuleAnalysisCachePath || join(tmpdir(), 'strata-workspace-powershell-cache') },
        stdio: 'ignore', timeout: 15000, windowsHide: true,
      });
      if (result.status === 23) fail(409, 'This item changed on disk. Refresh it before trying again.');
      if (result.error || result.status !== 0) fail(403, 'This file could not be saved. Its previous contents are retained in workspace trash.');
    } else renameSync(temporary, item.path);
    completed = true;
  } finally {
    if (fd !== undefined) closeSync(fd);
    // A failed Windows replacement can leave either name in place; retain both recovery copies.
    if (completed && existsSync(temporary)) unlinkSync(temporary);
  }
  return readWorkspaceFile(root, path);
}

export function renameWorkspaceFile(root, { path, newPath, revision: expected }) {
  const item = checked(root, path, expected), destination = locate(root, newPath, { create: true });
  if (item.path === destination.path) return { ok: true, path: destination.name };
  const existing = existingEntry(destination.path);
  // Case-insensitive filesystems (including default APFS) may resolve the old
  // entry through the new spelling. A separately named hardlink stays a conflict.
  if (existing && item.path.toLowerCase() === destination.path.toLowerCase()
      && !existing.isSymbolicLink() && item.st.ino && item.st.dev === existing.dev && item.st.ino === existing.ino
      && !readdirSync(dirname(destination.path)).includes(basename(destination.path))) {
    checked(root, path, expected);
    renameSync(item.path, destination.path);
    return { ok: true, path: destination.name };
  }
  if (existing) fail(409, 'An item with that name already exists.');
  if (item.st.isDirectory() && isInside(item.path, destination.path)) fail(400, 'A folder cannot be moved inside itself.');
  checked(root, path, expected);
  renameSync(item.path, destination.path);
  return { ok: true, path: destination.name };
}

function trashDirectory(base) {
  const path = join(base, TRASH);
  if (!existsSync(path)) mkdirSync(path, { mode: 0o700 });
  if (lstatSync(path).isSymbolicLink() || !lstatSync(path).isDirectory() || !isInside(base, realpathSync(path))) fail(403, 'Workspace trash is unavailable.');
  makePrivateDir(path);
  const ignore = join(path, '.gitignore');
  if (!existsSync(ignore)) writeFileSync(ignore, '*\n', { flag: 'wx', mode: 0o600 });
  return path;
}

export function trashWorkspaceFile(root, { path, revision: expected }) {
  const item = checked(root, path, expected), trash = trashDirectory(item.base), id = randomUUID();
  const metadata = join(trash, `${id}.json`), retained = join(trash, id);
  writeFileSync(metadata, JSON.stringify({ path: item.name, revision: expected, deletedAt: new Date().toISOString() }), { flag: 'wx', mode: 0o600 });
  try { checked(root, path, expected); renameSync(item.path, retained); }
  catch (error) { unlinkSync(metadata); throw error; }
  return { ok: true, trashId: id, path: item.name, retained: true };
}

/** Retained deletions and edit backups remain recoverable after the UI reloads. */
export function listWorkspaceTrash(root, offset = 0) {
  if (!Number.isSafeInteger(offset) || offset < 0) fail(400, 'Choose a valid file-list page.');
  const { base } = locate(root, '', { allowRoot: true }), trash = join(base, TRASH);
  if (!existsSync(trash)) return { entries: [], nextOffset: null };
  if (lstatSync(trash).isSymbolicLink() || !lstatSync(trash).isDirectory() || !isInside(base, realpathSync(trash))) fail(403, 'Workspace trash is unavailable.');
  const candidates = readdirSync(trash, { withFileTypes: true }).filter(item => item.isFile() && /^[0-9a-f-]{36}\.json$/.test(item.name));
  const entries = [];
  for (const item of candidates) {
    try {
      const metadata = join(trash, item.name), retained = join(trash, item.name.slice(0, -5));
      if (lstatSync(metadata).size > 8192 || !existsSync(retained)) continue;
      const info = JSON.parse(readFileSync(metadata, 'utf8')), st = lstatSync(retained);
      if (!info || typeof info !== 'object' || typeof info.path !== 'string') continue;
      if (st.isSymbolicLink() || !st.isFile() && !st.isDirectory()) continue;
      parts(info.path);
      entries.push({ trashId: item.name.slice(0, -5), path: info.path,
        kind: info.kind === 'edit-backup' ? 'edit-backup' : st.isDirectory() ? 'dir' : 'file',
        deletedAt: typeof info.deletedAt === 'string' ? info.deletedAt : '' });
    } catch (error) {
      if (error.code !== 'ENOENT' && !(error instanceof SyntaxError) && error.status !== 400) throw error;
    }
  }
  entries.sort((a, b) => b.deletedAt.localeCompare(a.deletedAt) || a.trashId.localeCompare(b.trashId));
  return { entries: entries.slice(offset, offset + 200), nextOffset: offset + 200 < entries.length ? offset + 200 : null };
}

export function restoreWorkspaceFile(root, id) {
  if (typeof id !== 'string' || !/^[0-9a-f-]{36}$/.test(id)) fail(400, 'Choose a valid trash item.');
  const base = realpathSync(root), trash = trashDirectory(base), metadata = join(trash, `${id}.json`), retained = join(trash, id);
  if (!existsSync(metadata) || !existsSync(retained)) fail(404, 'That trash item is no longer available.');
  if (lstatSync(metadata).isSymbolicLink() || lstatSync(retained).isSymbolicLink() || lstatSync(metadata).size > 8192) fail(403, 'That trash item cannot be restored here.');
  const info = JSON.parse(readFileSync(metadata, 'utf8'));
  const destination = locate(base, info.path, { create: true });
  if (existingEntry(destination.path)) fail(409, 'An item with that name already exists.');
  renameSync(retained, destination.path);
  return { ok: true, path: destination.name };
}

export function openWorkspaceFile(root, path) {
  const item = locate(root, path, { allowRoot: true });
  return openFile(item.base, item.path, 'app');
}

export function workspaceFileError(error) {
  if (error.status) return { status: error.status, message: error.message };
  const messages = { EEXIST: [409, 'An item with that name already exists.'], ENOENT: [404, 'That item is no longer there.'],
    EACCES: [403, 'This item is read-only or in use.'], EPERM: [403, 'This item is read-only or in use.'],
    ENOTDIR: [400, 'Choose an existing parent folder.'], ENOSPC: [507, 'There is not enough disk space.'] };
  const [status, message] = messages[error.code] ?? [400, 'The file operation could not be completed.'];
  return { status, message };
}
