import { execFile } from 'node:child_process';
import { dirname } from 'node:path';
import { q36FileSha256 } from './q36-run-contract.mjs';

export const Q36_HELP_LIMIT = 65536;
/** Per-model UI fields: generic Q38/Turbo choices never describe a Q36 engine. */
export function q36KvFields(fields, capability = null) {
  return fields.map(field => {
    if (['cache_type_k', 'cache_type_v'].includes(field.id)) {
      const choices = capability?.[field.id.endsWith('_k') ? 'cacheTypesK' : 'cacheTypesV'] ?? ['f32'];
      return { ...field, supported: true, choices: [...choices], hint: capability
        ? 'Verified engine storage formats. FP16 is experimental and model quality is unqualified. Default is FP32.'
        : 'Engine capability is unknown. Default uses the legacy FP32 path.' };
    }
    return field.id.startsWith('spin8_') ? { ...field, supported: false, choices: [] } : field;
  });
}
/** Accept only the implemented scalar layout, never version strings or prose help. */
export function parseQ36KvCapabilities(text) {
  if (typeof text !== 'string' || Buffer.byteLength(text) > Q36_HELP_LIMIT) return null;
  const lines = text.split(/\r?\n/).filter(line => line.startsWith('Q36_KV_CAPABILITIES'));
  if (lines.length !== 1 || !lines[0].startsWith('Q36_KV_CAPABILITIES ')) return null;
  try {
    const c = JSON.parse(lines[0].slice('Q36_KV_CAPABILITIES '.length));
    if (Object.keys(c).sort().join(',') !== 'cacheTypesK,cacheTypesV,family,flags,layoutVersion,quality,scaleSchema,schema' ||
        c.schema !== 1 || c.family !== 'qwen35moe' || c.layoutVersion !== 1 || c.scaleSchema !== 'none' ||
        c.quality !== 'experimental-unqualified' || JSON.stringify(c.flags) !== '["--cache-type-k","--cache-type-v"]') return null;
    for (const types of [c.cacheTypesK, c.cacheTypesV]) {
      if (!Array.isArray(types) || !types.length || new Set(types).size !== types.length || !types.includes('f32') ||
          types.some(type => !['f32', 'f16'].includes(type))) return null;
    }
    return c;
  } catch { return null; }
}

/** Memory is an independent contract; never extend the strict KV marker schema. */
export function parseQ36MemoryCapabilities(text) {
  if (typeof text !== 'string' || Buffer.byteLength(text) > Q36_HELP_LIMIT) return null;
  const lines = text.split(/\r?\n/).filter(line => line.startsWith('Q36_MEMORY_CAPABILITIES'));
  if (lines.length !== 1 || !lines[0].startsWith('Q36_MEMORY_CAPABILITIES ')) return null;
  try {
    const c = JSON.parse(lines[0].slice('Q36_MEMORY_CAPABILITIES '.length));
    if (!c || Object.keys(c).sort().join(',') !== 'family,flag,modes,qualification,requiresExpertStore,requiresExplicitSlots,schema,scope,totalProcessBound' ||
        c.schema !== 1 || c.family !== 'qwen35moe' || c.flag !== '--q36-memory' || c.requiresExpertStore !== 'mapped-pack' ||
        c.requiresExplicitSlots !== true || c.scope !== 'expert-working-storage' || c.totalProcessBound !== false ||
        c.qualification !== 'experimental-unqualified' || !['[]','["bounded-experts-v1"]'].includes(JSON.stringify(c.modes))) return null;
    return c;
  } catch { return null; }
}

export function verifiedQ36MemoryCapability(capability, sha = capability?.binarySha256) {
  if (!capability || !/^[a-f0-9]{64}$/.test(sha ?? '') || capability.binarySha256 !== sha) return null;
  const { binarySha256, ...marker } = capability;
  return parseQ36MemoryCapabilities(`Q36_MEMORY_CAPABILITIES ${JSON.stringify(marker)}`) ? structuredClone(capability) : null;
}

/** No model/user args or saved environment reaches this bounded, no-model help query. */
export async function probeQ36Capabilities(exe, { run = execFile, hash = q36FileSha256 } = {}) {
  const unknown = { kvCapability: null, memoryCapability: null };
  const before = exe ? hash(exe) : null;
  if (!before) return unknown;
  const env = Object.fromEntries(Object.entries(process.env).filter(([key]) =>
    ['PATH', 'SYSTEMROOT', 'WINDIR', 'LD_LIBRARY_PATH', 'CUDA_PATH'].includes(key.toUpperCase())));
  const output = await new Promise(resolve => {
    run(exe, ['--help'], { cwd: dirname(exe), windowsHide: true, timeout: 3000, maxBuffer: Q36_HELP_LIMIT, env },
      (error, stdout, stderr) => resolve(error ? null : `${stdout ?? ''}\n${stderr ?? ''}`));
  });
  if (hash(exe) !== before) return unknown;
  const bind = cap => cap ? { ...cap, binarySha256: before } : null;
  return { kvCapability: bind(parseQ36KvCapabilities(output)), memoryCapability: bind(parseQ36MemoryCapabilities(output)) };
}

/** Public KV-only return shape is retained for existing callers. */
export async function probeQ36KvCapabilities(exe, options) {
  return (await probeQ36Capabilities(exe, options)).kvCapability;
}
