/**
 * Windows graphics cards that nvidia-smi does not cover (AMD, Intel): their size and their load.
 *  - size: the DirectX adapter list in the registry (HKLM\SOFTWARE\Microsoft\DirectX) has a 64-bit DedicatedVideoMemory and the adapter
 *    LUID. Win32_VideoController.AdapterRAM is 32 bits and reads 0 or 4 GB for a 16 GB card (#3727, #3836).
 *  - load and memory in use: the Windows GPU performance counters (the ones Task Manager shows), read by one long-lived PowerShell and
 *    matched to the card by LUID. Temperature, power and the PCIe link need the vendor's own library, so they stay "not available".
 */
import { createPowerShellLoop, endWithHost } from './windows-poll.mjs';
import { utf8Script } from './powershell-utf8.mjs';

export const DX_ADAPTERS_SCRIPT = utf8Script(
  "Get-ChildItem HKLM:\\SOFTWARE\\Microsoft\\DirectX | ForEach-Object { $p = Get-ItemProperty $_.PSPath; " +
  "if ($p.Description -and $null -ne $p.AdapterLuid) { [pscustomobject]@{ Description = [string]$p.Description; VendorId = [string]$p.VendorId; " +
  "Luid = [string]$p.AdapterLuid; Dedicated = [string]$p.DedicatedVideoMemory; Software = [string]$p.SoftwareAdapter } } } | ConvertTo-Json -Compress");

/** luid_0x<high>_0x<low> in the spelling of the performance counters, from the registry's 64-bit number. */
export function luidName(value) {
  try {
    const n = BigInt.asUintN(64, BigInt(String(value)));
    const hex = v => '0x' + v.toString(16).padStart(8, '0');
    return `luid_${hex(n >> 32n)}_${hex(n & 0xffffffffn)}`;
  } catch { return null; }
}

/** The adapters the DirectX registry lists: [{ description, vendorId, luid, bytes, software }]. Never throws. */
export function parseDxAdapters(raw) {
  try {
    const parsed = JSON.parse(raw), rows = Array.isArray(parsed) ? parsed : [parsed];
    return rows.filter(r => r?.Description).map(r => ({
      description: String(r.Description), vendorId: Number(r.VendorId) || 0, luid: luidName(r.Luid),
      bytes: Number(r.Dedicated) || 0, software: String(r.Software) === '1' || String(r.Software).toLowerCase() === 'true',
    }));
  } catch { return []; }
}

const norm = text => String(text ?? '').toLowerCase().replace(/[^a-z0-9]/g, '');
const VIRTUAL = /virtual|remote|indirect|mirror|basic render|meta |parsec|displaylink|idd/i;

/**
 * Win32_VideoController rows + the DirectX adapter list -> the cards this app can show. A card keeps the size from the registry;
 * a display driver for a virtual monitor (no size, no known maker) is not a graphics card and is dropped.
 */
export function windowsOtherGpus(videoRows, dx) {
  const used = new Set();
  const out = [];
  for (const row of videoRows) {
    if (!row?.Name) continue;
    const match = dx.find((a, i) => !used.has(i) && !a.software && norm(a.description) === norm(row.Name));
    if (match) used.add(dx.indexOf(match));
    const vendor = /amd|radeon/i.test(row.Name) ? 'amd' : /intel/i.test(row.Name) ? 'intel' : 'unknown';
    const vramGb = match?.bytes ? Math.round(match.bytes / 1024 ** 3 * 10) / 10 : 0;
    if (vendor === 'unknown' && (!vramGb || VIRTUAL.test(row.Name))) continue;
    out.push({ id: String(row.PNPDeviceID), vendor, name: row.Name, vramGb, luid: match?.luid ?? null });
  }
  return out;
}

/** One counter line: "G <luid> <load percent> <dedicated bytes in use>". */
export function parseGpuCounterLine(line) {
  const m = /^G\s+(luid_0x[0-9a-f]+_0x[0-9a-f]+)\s+(\d+(?:\.\d+)?)\s+(\d+)\s*$/i.exec(String(line));
  return m ? { luid: m[1].toLowerCase(), utilizationPct: Math.max(0, Math.min(100, Number(m[2]))), memoryUsedGb: Number(m[3]) / 1024 ** 3 } : null;
}

export function createWindowsGpuReader({ spawnFn, intervalSec = 4, maxStarts = 3, now = Date.now } = {}) {
  const readings = new Map();
  const script = utf8Script(
    `while($true){` +
    `$e=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine);$m=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory);` +
    `$u=@{};foreach($r in $e){if($r.Name -match '(luid_0x[0-9A-Fa-f]+_0x[0-9A-Fa-f]+)_phys_\\d+_eng_\\d+_engtype_(\\w+)'){$k=$matches[1].ToLower()+'|'+$matches[2];$u[$k]=[double]$u[$k]+[double]$r.UtilizationPercentage}};` +
    `$b=@{};foreach($k in $u.Keys){$l=$k.Split('|')[0];if(-not $b.ContainsKey($l) -or $u[$k] -gt $b[$l]){$b[$l]=$u[$k]}};` +
    `foreach($r in $m){if($r.Name -match '(luid_0x[0-9A-Fa-f]+_0x[0-9A-Fa-f]+)'){$l=$matches[1].ToLower();[Console]::Out.WriteLine('G '+$l+' '+[math]::Round([double]$b[$l],0)+' '+$r.DedicatedUsage)}};` +
    `[Console]::Out.Flush();Start-Sleep -Seconds ${intervalSec}}`);
  const loop = createPowerShellLoop({ script, spawnFn, maxStarts, onLine: line => {
    const parsed = parseGpuCounterLine(line);
    if (parsed) readings.set(parsed.luid, { ...parsed, at: now() });
    return !!parsed;
  } });
  return {
    read(luid) { loop.start(); const r = readings.get(String(luid ?? '').toLowerCase()); return r && now() - r.at < 6 * intervalSec * 1000 ? r : null; },
    stop() { loop.stop(); },
  };
}

let shared = null;
/** The card's load and memory in use from the Windows counters; null when there is no reading. */
export function windowsGpuReading(luid) {
  if (process.platform !== 'win32' || !luid) return null;
  shared ??= endWithHost(createWindowsGpuReader());
  return shared.read(luid);
}
