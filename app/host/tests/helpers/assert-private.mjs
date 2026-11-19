import assert from 'node:assert/strict';
import { statSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { currentUserSid } from '../../server/private-file.mjs';

/** Effective protection: mode 0600 on POSIX; one protected full-access ACE for this SID on NTFS. */
export function assertPrivate(path) {
  if (process.platform !== 'win32') { assert.equal(statSync(path).mode & 0o777, 0o600); return; }
  const descriptor = execFileSync(join(process.env.SystemRoot, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe'), ['-NoProfile', '-NonInteractive', '-Command', `$a=[System.IO.File]::GetAccessControl('${path.replace(/'/g, "''")}');$r=[System.Security.AccessControl.RawSecurityDescriptor]::new($a.GetSecurityDescriptorBinaryForm(),0);$a.AreAccessRulesProtected;$r.DiscretionaryAcl.Count;foreach($e in $r.DiscretionaryAcl){$e.SecurityIdentifier.Value;$e.AccessMask;$e.AceFlags.ToString();$e.AceType.ToString()}`],
    { encoding: 'utf8', timeout: 30000, windowsHide: true }).trim();
  // SDDL may serialize the runner's Administrator SID as LA. Compare the
  // resolved identity and complete ACE instead of an account-name abbreviation.
  assert.deepEqual(descriptor.split(/\r?\n/), ['True', '1', currentUserSid(), String(0x1f01ff), 'None', 'AccessAllowed'], `ACL is not owner-only: ${descriptor}`);
}
