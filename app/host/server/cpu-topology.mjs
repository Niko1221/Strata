/** Verified CPU shape is process-local. A display fallback is never provenance. */
import os from 'node:os';
import { readFile } from 'node:fs/promises';
import { parseCpuTopology, parseWindowsProcessors } from './device-probes.mjs';
import { probe, round } from './util.mjs';
import { utf8Script } from './powershell-utf8.mjs';

export function completeTopology(processors, threads) {
  return threads > 0 && processors.length > 0 &&
    new Set(processors.map(p => p.id)).size === processors.length &&
    new Set(processors.map(p => p.socket)).size === processors.length &&
    processors.every(p => p.id && p.socket && p.model && p.model !== 'Unknown processor' &&
      Number.isSafeInteger(p.cores) && p.cores > 0 && Number.isSafeInteger(p.threads) && p.threads >= p.cores) &&
    processors.reduce((n, p) => n + p.threads, 0) === threads;
}

export function createCpuDetector({ cpus = () => os.cpus(), run = probe, read = readFile, architecture = () => process.arch } = {}) {
  let binding = null, generation = 0, sequence = 0, applied = 0, verified = null, latest = null;
  const key = (platform, arch, logical) => JSON.stringify([platform, arch, logical.length, logical.map(p => p.model?.trim() ?? '').sort()]);
  const bindingKey = platform => key(platform, architecture(), cpus());
  async function detect(platform) {
    const logical = cpus(), arch = architecture(), nextBinding = key(platform, arch, logical);
    if (nextBinding !== binding) { binding = nextBinding; generation++; verified = null; latest = null; applied = 0; }
    const visit = generation, request = ++sequence, threads = logical.length;
    let model = logical[0]?.model?.trim() || 'Unknown processor', processors = [];
    const ghz = round((logical[0]?.speed ?? 0) / 1000, 2);
    try {
      if (platform === 'win32') {
        const raw = await run('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', utf8Script('Get-CimInstance Win32_Processor | Select-Object DeviceID,SocketDesignation,Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress')]);
        processors = parseWindowsProcessors(raw, arch);
      } else if (platform === 'darwin') {
        const [brand, physical, threadCount, packages] = await Promise.all([
          run('sysctl', ['-n', 'machdep.cpu.brand_string']), run('sysctl', ['-n', 'hw.physicalcpu']),
          run('sysctl', ['-n', 'hw.logicalcpu']), run('sysctl', ['-n', 'hw.packages']),
        ]);
        if (brand?.trim()) model = brand.trim();
        const cores = Number(physical), count = Number(packages), totalThreads = Number(threadCount);
        if (Number.isSafeInteger(count) && count > 0 && cores > 0 && cores % count === 0 && totalThreads > 0 && totalThreads % count === 0)
          processors = Array.from({ length: count }, (_, i) => ({ id: `socket-${i}`, socket: String(i), model, cores: cores / count, threads: totalThreads / count, arch, ghz, ...(count === 1 ? { logicalIds: logical.map((_, index) => index) } : {}) }));
      } else if (platform === 'linux') {
        const raw = await run('lscpu', ['-b', '-p=CPU,CORE,SOCKET']), names = new Map();
        try {
          for (const block of (await read('/proc/cpuinfo', 'utf8')).split(/\n\s*\n/)) {
            const fields = Object.fromEntries(block.split('\n').filter(l => l.includes(':')).map(l => { const i = l.indexOf(':'); return [l.slice(0, i).trim(), l.slice(i + 1).trim()]; }));
            if (fields['physical id'] && fields['model name']) names.set(fields['physical id'], fields['model name']);
          }
        } catch { /* lscpu can still provide a complete topology with Node's model. */ }
        processors = parseCpuTopology(raw, model, arch, ghz, names);
      }
    } catch { processors = []; }
    const fallback = { model, cores: threads || 1, threads: threads || 1, arch, ghz, processors: [], topologyKnown: false, topologySource: 'unavailable' };
    // Neither a late result for another binding nor an older completion may replace a newer reading.
    if (visit !== generation) return latest ? structuredClone(latest) : fallback;
    if (request < applied) return structuredClone(latest);
    applied = request;
    if (completeTopology(processors, threads)) {
      processors.sort((a, b) => a.socket.localeCompare(b.socket) || a.id.localeCompare(b.id));
      const names = [...new Set(processors.map(p => p.model))];
      verified = { model: names.join(' + '), cores: processors.reduce((n, p) => n + p.cores, 0), threads, arch, ghz, processors, topologyKnown: true };
      latest = { ...verified, topologySource: 'live' };
    } else latest = verified ? { ...verified, topologySource: 'cached' } : fallback;
    return structuredClone(latest);
  }
  return { detect, bindingKey };
}
