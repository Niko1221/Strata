import assert from 'node:assert/strict';
import { statSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { currentUserSid } from '../../server/private-file.mjs';

/**
 * The effective protection of a file: mode 0600 on POSIX; on NTFS a protected DACL whose only entry is full control
 * for this user. The SDDL spelling (the DACL flags, FA against its hex rights) differs between Windows images, so
 * the check reads the DACL's meaning, not one exact string.
 */
export function assertPrivate(path) {
  if (process.platform !== 'win32') { assert.equal(statSync(path).mode & 0o777, 0o600); return; }
  const sddl = execFileSync(join(process.env.SystemRoot, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe'), ['-NoProfile', '-NonInteractive', '-Command', `(Get-Acl -LiteralPath '${path.replace(/'/g, "''")}').Sddl`],
    { encoding: 'utf8', timeout: 30000, windowsHide: true }).trim();
  const at = sddl.indexOf('D:');
  assert.ok(at >= 0, `no DACL in the security descriptor: ${sddl}`);
  const dacl = sddl.slice(at);
  assert.ok(/^D:[A-Z]*P[A-Z]*(?:\(|$)/.test(dacl), `DACL is not protected (inherits entries): ${sddl}`);
  const one = dacl.replace(/^D:[A-Z]+/, 'D:').replace(/\b0x1f01ff\b/i, 'FA').toLowerCase();
  assert.equal(one, `d:(a;;fa;;;${currentUserSid().toLowerCase()})`, `ACL is not owner-only: ${sddl}`);
}
