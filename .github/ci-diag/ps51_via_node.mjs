// DIAGNOSTIC ONLY (ci/checks-green; dropped before main). Windows PowerShell 5.1 children started by NODE (as the app host does), from a
// pwsh step: (raw) the environment as inherited, (kept) PowerShell 7's PSModulePath entries dropped, (kept+scratch) also APPDATA/LOCALAPPDATA
// in a scratch folder, as apps/host/scripts/test.mjs sets them. Exit code, output and first error line per probe.
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const probes = JSON.parse(readFileSync(new URL('./ps51_probes.json', import.meta.url), 'utf8'));
const exe = join(process.env.SystemRoot, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe');
const ps7 = p => { const l = p.toLowerCase().replace(/\//g, '\\'); return l.includes('\\powershell\\') && !l.includes('\\windowspowershell\\'); };
const kept = (process.env.PSModulePath || '').split(';').filter(p => p && !ps7(p)).join(';');
const scratch = mkdtempSync(join(tmpdir(), 'cig-ps51-'));
const variants = {
  raw: { ...process.env },
  kept: { ...process.env, PSModulePath: kept },
  'kept+scratch': { ...process.env, PSModulePath: kept, APPDATA: join(scratch, 'Roaming'), LOCALAPPDATA: join(scratch, 'Local') },
};
console.log('node parent; PSModulePath as inherited: ' + process.env.PSModulePath);
for (const [name, env] of Object.entries(variants)) {
  console.log(`=== node -> powershell.exe, ${name}`);
  for (const [what, script] of probes) {
    const enc = Buffer.from(script, 'utf16le').toString('base64');
    let rc = 0, out = '', err = '';
    try { out = execFileSync(exe, ['-NoProfile', '-NonInteractive', '-EncodedCommand', enc], { env, encoding: 'utf8', timeout: 60000, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] }); }
    catch (e) { rc = e.status ?? -1; out = String(e.stdout || ''); err = String(e.stderr || ''); }
    const first = err.split(/\r?\n/).find(l => l.trim()) || '';
    console.log(`${name.padEnd(12)} rc=${rc} out=${out.trim().replace(/\s+/g, ' ').slice(0, 300)} err=${first.slice(0, 200)}`);
    console.log(`             ${what}`);
  }
}
