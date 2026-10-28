import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  atomicBackupPath,
  atomicTempPath,
  atomicWriteFileSync,
  atomicWriteJsonSync,
  readJsonWithBackupSync,
} from '../server/atomic-file.mjs';

const scratch = t => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-atomic-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  return dir;
};

test('a JSON save leaves the file, one backup and no temporary file', t => {
  const dir = scratch(t);
  const file = join(dir, 'state.json');

  atomicWriteJsonSync(file, { version: 1, name: 'first' });
  assert.deepEqual(readJsonWithBackupSync(file), { version: 1, name: 'first' });
  assert.ok(existsSync(atomicBackupPath(file)), 'the first good copy is the backup');
  assert.equal(existsSync(atomicTempPath(file)), false, 'the temporary file is gone after the rename');

  atomicWriteJsonSync(file, { version: 2, name: 'second' });
  assert.deepEqual(readJsonWithBackupSync(file), { version: 2, name: 'second' });
  assert.deepEqual(JSON.parse(readFileSync(atomicBackupPath(file), 'utf8')), { version: 1, name: 'first' });
  assert.equal(existsSync(atomicTempPath(file)), false);
});

test('a half-written temporary file and a truncated main file both read as the last good copy', t => {
  const dir = scratch(t);
  const file = join(dir, 'state.json');
  atomicWriteJsonSync(file, { version: 1 });

  // The machine stopped after the temporary file was written, before the rename.
  writeFileSync(atomicTempPath(file), JSON.stringify({ version: 2 }));
  assert.deepEqual(readJsonWithBackupSync(file), { version: 1 }, 'the main file was never replaced');
  assert.deepEqual(JSON.parse(readFileSync(atomicBackupPath(file), 'utf8')), { version: 1 });

  // The machine stopped during a plain overwrite of the main file.
  writeFileSync(file, '{"version":');
  assert.deepEqual(readJsonWithBackupSync(file), { version: 1 }, 'the backup holds the last valid content');

  // Saving again must not copy the truncated bytes over the good backup.
  atomicWriteJsonSync(file, { version: 3 });
  assert.deepEqual(readJsonWithBackupSync(file), { version: 3 });
  assert.deepEqual(JSON.parse(readFileSync(atomicBackupPath(file), 'utf8')), { version: 1 }, 'the backup was preserved');
  assert.equal(existsSync(atomicTempPath(file)), false, 'the next save cleared the leftover temporary file');
});

test('a backup recovers a deleted main file and the caller fallback survives two broken files', t => {
  const dir = scratch(t);
  const file = join(dir, 'state.json');
  atomicWriteJsonSync(file, { ok: true });
  rmSync(file);
  assert.deepEqual(readJsonWithBackupSync(file), { ok: true }, 'a missing main file falls back to .bak');

  writeFileSync(file, '{');
  writeFileSync(atomicBackupPath(file), 'not json');
  assert.equal(readJsonWithBackupSync(file, 'fallback'), 'fallback');
});

test('a text save is durable and keeps the previous text', t => {
  const dir = scratch(t);
  const file = join(dir, 'SKILL.md');
  atomicWriteFileSync(file, 'first\n');
  atomicWriteFileSync(file, 'second\n');
  assert.equal(readFileSync(file, 'utf8'), 'second\n');
  assert.equal(readFileSync(atomicBackupPath(file), 'utf8'), 'first\n');
  assert.equal(existsSync(atomicTempPath(file)), false);
});

test('a save creates the parent folder instead of failing', t => {
  const dir = scratch(t);
  const file = join(dir, 'a', 'b', 'state.json');
  atomicWriteJsonSync(file, { nested: true });
  assert.deepEqual(readJsonWithBackupSync(file), { nested: true });
});
