import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, mkdirSync, rmSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const script = fileURLToPath(new URL('../../../tools/spin8_backend.ps1', import.meta.url));
const verifier = fileURLToPath(new URL('../../../tools/spin8_backend_verify.ps1', import.meta.url));

test('both build verification gates use the shared base-commit patch verifier', () => {
  const source = readFileSync(script, 'utf8');
  assert.equal((source.match(/Assert-BonsaiSourcePatch -Source \$buildSource/g) ?? []).length, 2);
  assert.match(readFileSync(verifier, 'utf8'), /git -C \$Source diff HEAD --binary --full-index/);
});

test('offline approved patch passes, but an additional staged tracked edit is rejected', {
  skip: process.platform === 'win32' ? false : 'PowerShell 7 build verifier is a Windows tool',
}, () => {
  const base = mkdtempSync(join(tmpdir(), 'strata-bonsai-source-'));
  const source = join(base, 'source');
  mkdirSync(source);
  const git = args => execFileSync('git', ['-C', source, ...args], { encoding: 'utf8', windowsHide: true });
  const patch = join(base, 'approved.patch');
  const output = join(base, 'actual.patch');
  const runner = join(base, 'verify.ps1');
  // Paths remain argv, so no fixture-controlled values enter PowerShell code.
  writeFileSync(runner, `param($Verifier,$Source,$Commit,$Patch,$Output)\n$ErrorActionPreference='Stop'\n. $Verifier\nAssert-BonsaiSourcePatch -Source $Source -BaseCommit $Commit -PatchPath $Patch -OutputPath $Output\n`);
  try {
    git(['init', '--quiet']);
    git(['config', 'core.autocrlf', 'false']);
    writeFileSync(join(source, 'approved.txt'), 'original\n');
    writeFileSync(join(source, 'unrelated.txt'), 'original\n');
    git(['add', '.']);
    git(['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '--quiet', '-m', 'fixture']);
    const commit = git(['rev-parse', 'HEAD']).trim();
    writeFileSync(join(source, 'approved.txt'), 'approved change\n');
    git(['diff', 'HEAD', '--binary', '--full-index', '--no-ext-diff', '--src-prefix=a/', '--dst-prefix=b/', `--output=${patch}`]);
    const verify = () => execFileSync('pwsh', ['-NoLogo', '-NoProfile', '-NonInteractive', '-File', runner,
      verifier, source, commit, patch, output], { encoding: 'utf8', windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
    assert.match(verify(), /[a-f0-9]{64}/);
    writeFileSync(join(source, 'unrelated.txt'), 'unexpected staged edit\n');
    git(['add', 'unrelated.txt']);
    assert.equal(git(['diff', '--binary', '--full-index', '--no-ext-diff', '--src-prefix=a/', '--dst-prefix=b/']), readFileSync(patch, 'utf8'), 'an unstaged diff would miss the staged change');
    assert.throws(verify, error => error.stderr.includes('differs from the exact recorded patch'));
  } finally { rmSync(base, { recursive: true, force: true }); }
});
