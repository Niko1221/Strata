/**
 * Dedicated VRAM vs Windows shared GPU memory observation (PR-A).
 * Missing values stay null — never coerced to 0.
 */
import { spawnSync } from 'node:child_process';

const numberOrNull = value => {
  if (value === null || value === undefined) return null;
  const text = String(value).trim().replace(/\s*MiB\s*$/i, '');
  if (text === '' || /^\[?n\/a\]?$/i.test(text)) return null;
  const n = Number(text);
  return Number.isFinite(n) ? n : null;
};

const bytesToMib = bytes => {
  if ((typeof bytes !== 'number' && typeof bytes !== 'string')
    || (typeof bytes === 'string' && bytes.trim() === '')) return null;
  const n = Number(bytes);
  return Number.isFinite(n) && n >= 0 ? Math.round(n / 1024 / 1024) : null;
};

export function nullVramObservation({ source = 'unavailable', observedAt = new Date().toISOString() } = {}) {
  return {
    dedicated: { totalMib: null, usedMib: null, freeMib: null, reservedMib: null },
    shared: { totalMib: null, usedMib: null },
    sources: [source],
    observedAt,
  };
}

/**
 * Parse `nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free[,memory.reserved] --format=csv`
 * (with or without units / header). Shared memory is not provided by nvidia-smi.
 */
export function parseNvidiaMemoryCsv(raw) {
  const lines = String(raw ?? '').split(/\r?\n/).map(l => l.trim()).filter(Boolean)
    .filter(l => !/^name\s*,/i.test(l));
  return lines.flatMap(line => {
    const parts = line.split(',').map(s => s.trim());
    if (parts.length < 4) return [];
    const [name, total, used, free, reserved] = parts;
    if (!name || /^name$/i.test(name)) return [];
    return [{
      name,
      dedicated: {
        totalMib: numberOrNull(total),
        usedMib: numberOrNull(used),
        freeMib: numberOrNull(free),
        reservedMib: reserved === undefined ? null : numberOrNull(reserved),
      },
      shared: { totalMib: null, usedMib: null },
    }];
  });
}

/** Windows GPUAdapterMemory counter rows → dedicated / shared MiB. */
export function parseWindowsGpuAdapterMemory(rows) {
  const list = Array.isArray(rows) ? rows : rows ? [rows] : [];
  return list.filter(r => r && (r.Name || r.name)).map(r => ({
    luid: String(r.Name ?? r.name ?? ''),
    dedicated: { usedMib: bytesToMib(r.DedicatedUsage ?? r.dedicatedUsage) },
    shared: { usedMib: bytesToMib(r.SharedUsage ?? r.sharedUsage) },
    totalCommittedMib: bytesToMib(r.TotalCommitted ?? r.totalCommitted),
  }));
}

/**
 * Prefer nvidia-smi for dedicated totals/free; overlay Windows SharedUsage.
 * When both report dedicated used, keep nvidia-smi as the dedicated source of truth.
 */
export function mergeVramObservation({ nvidia = [], windows = [], observedAt = new Date().toISOString() } = {}) {
  const sources = [];
  const primary = nvidia[0] ?? null;
  const win = windows
    .filter(w => (w.dedicated?.usedMib ?? 0) > 0 || (w.shared?.usedMib ?? 0) > 0)
    .sort((a, b) => (b.dedicated?.usedMib ?? 0) - (a.dedicated?.usedMib ?? 0))[0]
    ?? windows[0]
    ?? null;

  if (primary) sources.push('nvidia-smi');
  if (win) sources.push('windows-gpu-counters');
  if (!sources.length) return nullVramObservation({ source: 'unavailable', observedAt });

  return {
    dedicated: {
      totalMib: primary?.dedicated?.totalMib ?? null,
      usedMib: primary?.dedicated?.usedMib ?? win?.dedicated?.usedMib ?? null,
      freeMib: primary?.dedicated?.freeMib ?? null,
      reservedMib: primary?.dedicated?.reservedMib ?? null,
    },
    shared: {
      totalMib: null,
      usedMib: win?.shared?.usedMib ?? null,
    },
    sources,
    observedAt,
    devices: {
      nvidia: nvidia.map(d => ({ name: d.name, dedicated: d.dedicated })),
      windows: windows.map(d => ({ luid: d.luid, dedicatedUsedMib: d.dedicated?.usedMib ?? null, sharedUsedMib: d.shared?.usedMib ?? null })),
    },
  };
}

function runNvidiaSmi() {
  try {
    const r = spawnSync('nvidia-smi', [
      '--query-gpu=name,memory.total,memory.used,memory.free,memory.reserved',
      '--format=csv',
    ], { encoding: 'utf8', windowsHide: true, timeout: 4000 });
    if (r.status !== 0 || !r.stdout) return [];
    return parseNvidiaMemoryCsv(r.stdout);
  } catch {
    return [];
  }
}

function runWindowsGpuCounters() {
  if (process.platform !== 'win32') return [];
  try {
    const script = "Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory | Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted | ConvertTo-Json -Compress";
    const r = spawnSync('powershell.exe', ['-NoProfile', '-Command', script], {
      encoding: 'utf8', windowsHide: true, timeout: 8000,
    });
    if (r.status !== 0 || !r.stdout?.trim()) return [];
    const parsed = JSON.parse(r.stdout);
    return parseWindowsGpuAdapterMemory(parsed);
  } catch {
    return [];
  }
}

/** Live snapshot. Failures yield null fields, not zeros. */
export function captureVramObservation({ now = () => new Date().toISOString() } = {}) {
  const observedAt = now();
  return mergeVramObservation({
    nvidia: runNvidiaSmi(),
    windows: runWindowsGpuCounters(),
    observedAt,
  });
}
