import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { appendLog, rotateLogIfNeeded, rotateLogsIn, writeLogFile } from '../server/log-files.mjs';

const scratch = t => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-log-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  return dir;
};

test('a file under the limit is left alone; one over it becomes exactly one .1', t => {
  const dir = scratch(t);
  const file = join(dir, 'engine.server.log');
  writeFileSync(file, '0123456789');
  assert.equal(rotateLogIfNeeded(file, { limit: 10 }), false, 'exactly at the limit is not over it');
  writeFileSync(file, '0123456789X');
  assert.equal(rotateLogIfNeeded(file, { limit: 10 }), true);
  assert.equal(existsSync(file), false);
  assert.equal(readFileSync(`${file}.1`, 'utf8'), '0123456789X');

  writeFileSync(file, 'again');
  writeFileSync(file, '0123456789Y');
  rotateLogIfNeeded(file, { limit: 10 });
  assert.equal(readFileSync(`${file}.1`, 'utf8'), '0123456789Y', 'the old .1 is replaced, not multiplied');
});

test('append rotates before the line that crosses the limit', t => {
  const dir = scratch(t);
  const file = join(dir, 'host.log');
  writeFileSync(file, 'x'.repeat(21));
  appendLog(file, 'new line\n', { limit: 20 });
  assert.equal(readFileSync(`${file}.1`, 'utf8'), 'x'.repeat(21));
  assert.equal(readFileSync(file, 'utf8'), 'new line\n');
});

test('a whole log file is capped by keeping its newest bytes', t => {
  const dir = scratch(t);
  const file = join(dir, 'harness.log');
  writeFileSync(file, 'old content that is over the limit');
  writeLogFile(file, 'abcdefghij', { limit: 8 });
  assert.equal(readFileSync(file, 'utf8'), 'cdefghij');
  assert.ok(statSync(file).size <= 8);
  assert.ok(existsSync(`${file}.1`), 'the older run is kept once');
});

test('only matching engine and host logs are rotated in a folder', t => {
  const dir = scratch(t);
  writeFileSync(join(dir, 'app-1.server.log'), 'a'.repeat(11));
  writeFileSync(join(dir, 'app-1.host.log'), 'b'.repeat(11));
  writeFileSync(join(dir, 'notes.log'), 'c'.repeat(11));
  writeFileSync(join(dir, 'app-2.server.log.1'), 'd'.repeat(11));

  const rotated = rotateLogsIn(dir, { limit: 10, suffixes: ['.server.log', '.host.log'] });
  assert.deepEqual(rotated.sort(), ['app-1.host.log', 'app-1.server.log']);
  assert.ok(existsSync(join(dir, 'app-1.server.log.1')));
  assert.ok(!existsSync(join(dir, 'notes.log.1')), 'a log the host does not own is untouched');
  assert.ok(!existsSync(join(dir, 'app-2.server.log.1.1')));
});

test('a recent log is left alone when the caller asks for old ones only', t => {
  const dir = scratch(t);
  writeFileSync(join(dir, 'desktop-now.log'), 'a'.repeat(11));
  const rotated = rotateLogsIn(dir, { limit: 10, suffixes: ['.log'], olderThanMs: 60 * 60 * 1000 });
  assert.deepEqual(rotated, []);
  assert.ok(existsSync(join(dir, 'desktop-now.log')));
});
