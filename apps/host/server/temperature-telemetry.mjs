/**
 * CPU and SSD temperatures for the Monitor's Power card (#3628). A value that this PC cannot give stays null and its line is left out,
 * so the card never shows a made-up number.
 *
 * Linux: the kernel's hwmon sensors (coretemp, k10temp, zenpower for the processor; nvme and drivetemp for the drives).
 * Windows: the drive reports its own temperature to a query that needs no administrator (IOCTL_STORAGE_QUERY_PROPERTY with the device
 *   temperature property). Get-StorageReliabilityCounter is tried first because it is the documented route; on a normal account it is
 *   refused, then the query is used. One long-lived PowerShell asks every 30 s (windows-poll.mjs).
 *   There is no unprivileged, trustworthy processor temperature on Windows, so none is shown there.
 * macOS: none.
 */
import { readFile, readdir } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { createPowerShellLoop, endWithHost } from './windows-poll.mjs';

const CPU_SENSORS = new Set(['coretemp', 'k10temp', 'zenpower', 'cpu_thermal', 'cpu-thermal']);
const DISK_SENSORS = new Set(['nvme', 'drivetemp']);
const plausible = c => Number.isFinite(c) && c > 0 && c < 150;

/** hwmon entries [{name, temps: [milli-degrees…]}] -> the hottest processor and drive sensor. */
export function pickTemperatures(entries) {
  let cpu = null, ssd = null;
  for (const { name, temps } of entries) {
    const hottest = Math.max(...temps.map(t => t / 1000).filter(plausible), -Infinity);
    if (!Number.isFinite(hottest)) continue;
    if (CPU_SENSORS.has(name)) cpu = Math.max(cpu ?? -Infinity, hottest);
    else if (DISK_SENSORS.has(name)) ssd = Math.max(ssd ?? -Infinity, hottest);
  }
  return { cpuTempC: cpu === null ? null : Math.round(cpu), ssdTempC: ssd === null ? null : Math.round(ssd) };
}

export async function readLinuxTemperatures({ base = '/sys/class/hwmon', read = readFile, list = readdir } = {}) {
  const entries = [];
  let folders = [];
  try { folders = await list(base); } catch { return { cpuTempC: null, ssdTempC: null }; }
  for (const folder of folders) {
    try {
      const name = String(await read(`${base}/${folder}/name`, 'utf8')).trim();
      if (!CPU_SENSORS.has(name) && !DISK_SENSORS.has(name)) continue;
      const files = (await list(`${base}/${folder}`)).filter(f => /^temp\d+_input$/.test(f));
      const temps = [];
      for (const file of files) temps.push(Number(String(await read(`${base}/${folder}/${file}`, 'utf8')).trim()));
      entries.push({ name, temps });
    } catch { /* a sensor that disappears is skipped */ }
  }
  return pickTemperatures(entries);
}

/** One line of the Windows loop: "ssd 53" or "ssd -". */
export function parseSsdLine(line) {
  const m = /^\s*ssd\s+(-?\d+|-)\s*$/.exec(String(line));
  if (!m) return undefined;
  const value = m[1] === '-' ? NaN : Number(m[1]);
  return plausible(value) ? value : null;
}

const WINDOWS_SCRIPT = String.raw`
$ErrorActionPreference='SilentlyContinue'
$cs=@'
using System;
using System.Runtime.InteropServices;
public static class StrataDiskTemp {
  [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)] static extern IntPtr CreateFile(string n, uint a, uint s, IntPtr p, uint d, uint f, IntPtr t);
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool DeviceIoControl(IntPtr h, uint c, byte[] i, uint il, byte[] o, uint ol, out uint r, IntPtr ov);
  [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr h);
  public static int Read(int n) {
    IntPtr h = CreateFile("\\\\.\\PhysicalDrive" + n, 0, 3, IntPtr.Zero, 3, 0, IntPtr.Zero);
    if (h == (IntPtr)(-1)) return -1;
    try {
      byte[] q = new byte[12]; q[0] = 0x34; byte[] o = new byte[512]; uint r;
      if (!DeviceIoControl(h, 0x002D1400, q, 12, o, 512, out r, IntPtr.Zero)) return -1;
      if (BitConverter.ToUInt16(o, 12) < 1) return -1;
      return BitConverter.ToInt16(o, 26);
    } finally { CloseHandle(h); }
  }
}
'@
$compiled=$false
while($true){
  $max=-1
  $c=@(Get-PhysicalDisk | Get-StorageReliabilityCounter | Where-Object { $_.Temperature -gt 0 })
  if($c.Count -gt 0){ $max=($c | Measure-Object Temperature -Maximum).Maximum }
  else {
    if(-not $compiled){ Add-Type -TypeDefinition $cs; $compiled=$true }
    foreach($n in 0..15){ $v=[StrataDiskTemp]::Read($n); if($v -gt $max){ $max=$v } }
  }
  if($max -gt 0){ [Console]::Out.WriteLine('ssd '+$max) } else { [Console]::Out.WriteLine('ssd -') }
  [Console]::Out.Flush()
  Start-Sleep -Seconds 30
}`;

export function createWindowsSsdReader({ spawnFn = spawn, maxStarts = 2, now = Date.now, intervalSec = 30 } = {}) {
  let latest = null;
  const loop = createPowerShellLoop({
    script: WINDOWS_SCRIPT, spawnFn, maxStarts,
    onLine: line => {
      const parsed = parseSsdLine(line);
      if (parsed === undefined) return false;
      latest = { value: parsed, at: now() };
      return true;
    },
  });
  return {
    read() { loop.start(); return latest && now() - latest.at < 4 * intervalSec * 1000 ? latest.value : null; },
    stop() { loop.stop(); },
    get starts() { return loop.starts; },
  };
}

let windowsReader = null;
let linuxCache = { at: 0, value: { cpuTempC: null, ssdTempC: null } };

/** The temperatures this PC can give now: { cpuTempC, ssdTempC } (null where unreadable). Cheap to call every few seconds. */
export async function readTemperatures({ platform = process.platform, now = Date.now } = {}) {
  if (platform === 'linux') {
    if (now() - linuxCache.at > 10_000) linuxCache = { at: now(), value: await readLinuxTemperatures() };
    return linuxCache.value;
  }
  if (platform === 'win32') {
    windowsReader ??= endWithHost(createWindowsSsdReader());
    return { cpuTempC: null, ssdTempC: windowsReader.read() };
  }
  return { cpuTempC: null, ssdTempC: null };
}
