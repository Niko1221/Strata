import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync, linkSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createWorkspaceFile, listWorkspaceFiles, readWorkspaceFile, renameWorkspaceFile, restoreWorkspaceFile, trashWorkspaceFile, updateWorkspaceFile, WORKSPACE_TEXT_LIMIT } from '../server/workspace-files.mjs';
import { host } from './helpers/reference-host.mjs';
import { makePrivate } from '../server/private-file.mjs';

function fixture(t) {
  const directory = mkdtempSync(join(tmpdir(), 'strata-file-crud-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const root = join(directory, 'workspace'); mkdirSync(root);
  return { directory, root };
}

test('old UTF-8 files can be read and edited; CRLF/BOM and conflicting external edits are preserved', t => {
  const { root } = fixture(t);
  writeFileSync(join(root, '昔のメモ.md'), '\uFEFFbefore\r\n', 'utf8');
  const before = readWorkspaceFile(root, '昔のメモ.md');
  assert.equal(before.text, '\uFEFFbefore\r\n');
  const saved = updateWorkspaceFile(root, { path: before.path, text: 'after\n', revision: before.revision });
  assert.equal(readFileSync(join(root, before.path), 'utf8'), '\uFEFFafter\r\n');
  writeFileSync(join(root, before.path), 'a different external edit');
  assert.throws(() => updateWorkspaceFile(root, { path: before.path, text: 'must not win', revision: saved.revision }), { status: 409 });
  assert.equal(readFileSync(join(root, before.path), 'utf8'), 'a different external edit');
  assert.ok(!readdirSync(root).some(name => name.startsWith('.strata-edit-')));
});

test('new files/folders, rename, trash and undo retain bytes and never replace a collision', t => {
  const { root } = fixture(t);
  createWorkspaceFile(root, { path: 'old-notes', kind: 'dir' });
  createWorkspaceFile(root, { path: 'old-notes/first.md', text: 'retained bytes' });
  assert.throws(() => createWorkspaceFile(root, { path: 'old-notes/first.md', text: 'overwrite' }), { code: 'EEXIST' });
  const old = readWorkspaceFile(root, 'old-notes/first.md');
  renameWorkspaceFile(root, { path: old.path, newPath: 'old-notes/renamed.md', revision: old.revision });
  assert.equal(existsSync(join(root, old.path)), false);
  const renamed = readWorkspaceFile(root, 'old-notes/renamed.md');
  const removed = trashWorkspaceFile(root, renamed);
  assert.equal(existsSync(join(root, renamed.path)), false);
  assert.equal(readFileSync(join(root, '.strata-trash', removed.trashId), 'utf8'), 'retained bytes');
  createWorkspaceFile(root, { path: renamed.path, text: 'new item' });
  assert.throws(() => restoreWorkspaceFile(root, removed.trashId), { status: 409 });
  const collision = readWorkspaceFile(root, renamed.path);
  trashWorkspaceFile(root, collision);
  restoreWorkspaceFile(root, removed.trashId);
  assert.equal(readFileSync(join(root, renamed.path), 'utf8'), 'retained bytes');
});

test('nonempty old folders can be trashed and restored without deleting their contents', t => {
  const { root } = fixture(t);
  mkdirSync(join(root, 'old')); writeFileSync(join(root, 'old', 'note.txt'), 'old');
  const entry = listWorkspaceFiles(root).entries.find(row => row.name === 'old');
  const removed = trashWorkspaceFile(root, entry);
  restoreWorkspaceFile(root, removed.trashId);
  assert.equal(readFileSync(join(root, 'old', 'note.txt'), 'utf8'), 'old');
});

test('case-only renames keep bytes while a different existing entry is never replaced', t => {
  const { root } = fixture(t);
  writeFileSync(join(root, 'OldName.md'), 'retained');
  let entry = readWorkspaceFile(root, 'OldName.md');
  renameWorkspaceFile(root, { path: entry.path, newPath: 'oldname.md', revision: entry.revision });
  assert.ok(readdirSync(root).includes('oldname.md'));
  assert.equal(readFileSync(join(root, 'oldname.md'), 'utf8'), 'retained');
  writeFileSync(join(root, 'another.md'), 'another original');
  entry = readWorkspaceFile(root, 'oldname.md');
  assert.throws(() => renameWorkspaceFile(root, { path: entry.path, newPath: 'another.md', revision: entry.revision }), { status: 409 });
  assert.equal(readFileSync(join(root, 'another.md'), 'utf8'), 'another original');
});

test('a separately named case-variant hardlink is never replaced', t => {
  const { root } = fixture(t);
  writeFileSync(join(root, 'CaseLink.md'), 'retained');
  if (existsSync(join(root, 'caselink.md'))) return t.skip('The filesystem resolves both spellings to one entry');
  linkSync(join(root, 'CaseLink.md'), join(root, 'caselink.md'));
  const entry = readWorkspaceFile(root, 'CaseLink.md');
  assert.throws(() => renameWorkspaceFile(root, { path: entry.path, newPath: 'caselink.md', revision: entry.revision }), { status: 409 });
  assert.ok(readdirSync(root).includes('CaseLink.md'));
  assert.ok(readdirSync(root).includes('caselink.md'));
});

test('Windows saves retain destination access rules, named streams and the previous contents', { skip: process.platform !== 'win32' }, t => {
  const { root } = fixture(t), target = join(root, "private ' old.md");
  writeFileSync(target, 'previous private contents');
  writeFileSync(target + ':Zone.Identifier', '[ZoneTransfer]\r\nZoneId=3\r\n');
  makePrivate(target);
  const accessDigest = () => createHash('sha256').update(execFileSync(join(process.env.SystemRoot || 'C:\\Windows', 'System32/WindowsPowerShell/v1.0/powershell.exe'),
    ['-NoProfile', '-NonInteractive', '-Command', '[IO.File]::GetAccessControl($env:STRATA_TEST_FILE).GetSecurityDescriptorSddlForm([Security.AccessControl.AccessControlSections]::Access)'],
    { env: { ...process.env, STRATA_TEST_FILE: target }, encoding: 'utf8', timeout: 15000, windowsHide: true }).trim()).digest('hex');
  const before = accessDigest(), file = readWorkspaceFile(root, "private ' old.md");
  updateWorkspaceFile(root, { path: file.path, text: 'saved private contents', revision: file.revision });
  assert.equal(accessDigest(), before, 'Destination DACL digest changed');
  assert.equal(readFileSync(target + ':Zone.Identifier', 'utf8'), '[ZoneTransfer]\r\nZoneId=3\r\n');
  assert.equal(readFileSync(target, 'utf8'), 'saved private contents');
  const trash = join(root, '.strata-trash');
  const metadata = readdirSync(trash).find(name => name.endsWith('.json'));
  assert.ok(metadata);
  assert.equal(readFileSync(join(trash, metadata.slice(0, -5)), 'utf8'), 'previous private contents');
});

test('deep existing workspace paths can still be saved on Windows', { skip: process.platform !== 'win32' }, t => {
  const { root } = fixture(t);
  const directory = Array.from({ length: 6 }, (_, n) => `old-${n}-` + 'nested'.repeat(8)).join('/');
  mkdirSync(join(root, directory), { recursive: true });
  const path = directory + '/old-note.md';
  writeFileSync(join(root, path), 'deep previous content');
  const before = readWorkspaceFile(root, path);
  updateWorkspaceFile(root, { path, text: 'deep saved content', revision: before.revision });
  assert.equal(readFileSync(join(root, path), 'utf8'), 'deep saved content');
});

test('workspace roots, traversal, drive/ADS paths and internal control directories cannot be mutated', t => {
  const { directory, root } = fixture(t);
  writeFileSync(join(directory, 'outside.txt'), 'outside sentinel');
  for (const path of ['', '../outside.txt', 'a/../../outside.txt', 'C:\\outside.txt', '\\\\server\\share', '/outside', 'note.txt:stream', '.git/config', '.strata-trash/item', 'a\\..\\outside.txt', 'NUL', 'con.txt', 'CONIN$', 'COM¹.md', 'LPT9.log']) {
    assert.throws(() => createWorkspaceFile(root, { path, text: 'escape' }), error => [400, 403].includes(error.status), path);
  }
  assert.equal(readFileSync(join(directory, 'outside.txt'), 'utf8'), 'outside sentinel');
  assert.throws(() => renameWorkspaceFile(root, { path: '', newPath: 'moved', revision: 'anything' }), { status: 400 });
});

test('linked directories cannot expose or modify files outside the selected workspace', t => {
  const { directory, root } = fixture(t);
  const outside = join(directory, 'outside'); mkdirSync(outside); writeFileSync(join(outside, 'secret.txt'), 'unchanged');
  try { symlinkSync(outside, join(root, 'link'), process.platform === 'win32' ? 'junction' : 'dir'); }
  catch (error) { if (['EPERM', 'ENOSYS', 'EACCES'].includes(error.code)) return t.skip('This filesystem cannot create test links.'); throw error; }
  assert.equal(listWorkspaceFiles(root).entries.some(row => row.name === 'link'), false);
  assert.throws(() => readWorkspaceFile(root, 'link/secret.txt'), { status: 403 });
  assert.throws(() => createWorkspaceFile(root, { path: 'link/new.txt', text: 'escape' }), { status: 403 });
  assert.equal(readFileSync(join(outside, 'secret.txt'), 'utf8'), 'unchanged');
  assert.equal(existsSync(join(outside, 'new.txt')), false);
});

test('large/non-UTF8 files stay read-only in the text editor and listing pages include old files', t => {
  const { root } = fixture(t);
  writeFileSync(join(root, 'large.txt'), Buffer.alloc(WORKSPACE_TEXT_LIMIT + 1, 65));
  writeFileSync(join(root, 'binary.dat'), Buffer.from([255, 254]));
  const large = readWorkspaceFile(root, 'large.txt'), binary = readWorkspaceFile(root, 'binary.dat');
  assert.equal(large.tooLarge, true); assert.equal(binary.binary, true);
  assert.throws(() => updateWorkspaceFile(root, { ...large, text: 'truncate' }), { status: 400 });
  assert.throws(() => createWorkspaceFile(root, { path: 'too-long.txt', text: '文'.repeat(WORKSPACE_TEXT_LIMIT) }), { status: 413 });
  for (let n = 0; n < 230; n++) writeFileSync(join(root, `old-${String(n).padStart(3, '0')}.md`), '');
  const first = listWorkspaceFiles(root), second = listWorkspaceFiles(root, '', first.nextOffset);
  assert.equal(first.entries.length, 200); assert.equal(second.entries.length, 32); assert.equal(second.nextOffset, null);
  assert.equal(new Set([...first.entries, ...second.entries].map(row => row.path)).size, 232);
});

test('actual Host routes enforce registered scope, browser origin, CRUD conflicts and reversible deletion', { timeout: 60000 }, async t => {
  const directory = mkdtempSync(join(tmpdir(), 'strata-file-routes-'));
  const profile = join(directory, 'profile'); mkdirSync(join(profile, 'work'), { recursive: true });
  const h = await host(profile);
  t.after(async () => { await h.stop(); rmSync(directory, { recursive: true, force: true }); });
  const request = async (method, path, body, origin = h.base) => {
    const response = await fetch(h.base + path, { method, headers: { origin, 'content-type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });
    return { status: response.status, data: await response.json() };
  };
  const root = (await h.call('/api/workspaces')).data.workspaces[0].id;
  assert.equal((await request('POST', '/api/workspace/file', { root: 'missing', path: 'old.txt', text: 'x' })).status, 404);
  assert.equal((await request('POST', '/api/workspace/file', { root, path: 'old.txt', text: 'x' }, 'https://example.invalid')).status, 403);
  assert.equal((await request('POST', '/api/workspace/file', { root, path: 'old.txt', text: 'old' })).status, 201);
  let file = (await h.call(`/api/workspace/file?root=${root}&path=old.txt`)).data;
  assert.equal((await request('PUT', '/api/workspace/file', { root, path: file.path, revision: file.revision, text: 'edited' })).status, 200);
  assert.equal((await request('PUT', '/api/workspace/file', { root, path: file.path, revision: file.revision, text: 'stale' })).status, 409);
  file = (await h.call(`/api/workspace/file?root=${root}&path=old.txt`)).data;
  const removed = await request('DELETE', `/api/workspace/file?root=${root}&path=old.txt&revision=${file.revision}`);
  assert.equal(removed.status, 200);
  assert.equal((await h.call(`/api/workspace/file?root=${root}&path=old.txt`)).status, 404);
  const history = await request('GET', `/api/workspace/trash?root=${root}`);
  assert.equal(history.status, 200, 'a newly opened trash view must find retained deletions');
  assert.ok(history.data.entries.some(item => item.trashId === removed.data.trashId && item.path === 'old.txt'));
  assert.equal((await request('GET', '/api/workspace/trash?root=unregistered')).status, 404);
  assert.equal((await request('POST', '/api/workspace/trash', { root, trashId: removed.data.trashId })).status, 200);
  assert.equal((await h.call(`/api/workspace/file?root=${root}&path=old.txt`)).data.text, 'edited');
  assert.ok(!(await request('GET', `/api/workspace/trash?root=${root}`)).data.entries.some(item => item.trashId === removed.data.trashId), 'restored rows disappear without removing other retained copies');
});
