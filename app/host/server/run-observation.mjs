/**
 * RunObservation — requested vs effective launch diagnostics (PR-A).
 * Secrets (prompt text, API keys, absolute private paths) must not enter this record.
 */
import { createHash, randomUUID } from 'node:crypto';
import { nullVramObservation } from './vram-observation.mjs';

export const RUN_OBSERVATION_SCHEMA_VERSION = 1;

function digestArgv(argv) {
  if (!Array.isArray(argv)) return null;
  // Drop absolute path values after known file flags to keep identity non-secret.
  const PATH_FLAGS = new Set(['-m', '--model', '--pack', '--native', '--mtp', '--ple-gguf', '--expert-profile']);
  const redacted = [];
  for (let i = 0; i < argv.length; i++) {
    const tok = String(argv[i]);
    redacted.push(tok);
    if (PATH_FLAGS.has(tok) && i + 1 < argv.length) {
      redacted.push('<path>');
      i += 1;
    }
  }
  return createHash('sha256').update(JSON.stringify(redacted)).digest('hex');
}

export function observationIdentity(parts) {
  const payload = {
    backendKind: parts.backendKind ?? null,
    binarySha256: parts.binarySha256 ?? null,
    metadataSignature: parts.metadataSignature ?? null,
    context: parts.context ?? null,
    kType: parts.kType ?? null,
    vType: parts.vType ?? null,
    gpuPlacement: parts.gpuPlacement ?? null,
    memoryMode: parts.memoryMode ?? null,
  };
  return createHash('sha256').update(JSON.stringify(payload)).digest('hex').slice(0, 32);
}

export function createRunObservation({
  backend = {},
  model = {},
  requested = {},
  cachePolicy = null,
  policyRevision = null,
  diagnosticId = null,
  createdAt = new Date().toISOString(),
} = {}) {
  const req = {
    context: requested.context ?? null,
    kType: requested.kType ?? null,
    vType: requested.vType ?? null,
    gpuPlacement: requested.gpuPlacement ?? null,
    ngl: requested.ngl ?? null,
    batch: requested.batch ?? null,
    ubatch: requested.ubatch ?? null,
    parallel: requested.parallel ?? null,
    flashAttention: requested.flashAttention ?? null,
    expertCache: requested.expertCache ?? null,
    memoryMode: requested.memoryMode ?? null,
  };
  const id = diagnosticId ?? observationIdentity({
    backendKind: backend.kind,
    binarySha256: backend.binarySha256 ?? null,
    metadataSignature: model.metadataSignature,
    context: req.context,
    kType: req.kType,
    vType: req.vType,
    gpuPlacement: req.gpuPlacement,
    memoryMode: req.memoryMode,
  }) + '-' + randomUUID().slice(0, 8);

  return {
    schemaVersion: RUN_OBSERVATION_SCHEMA_VERSION,
    diagnosticId: id,
    createdAt,
    backend: {
      kind: backend.kind ?? null,
      version: backend.version ?? null,
      executablePath: backend.executablePath ?? null,
      binarySha256: backend.binarySha256 ?? null,
    },
    model: {
      id: model.id ?? null,
      metadataSignature: model.metadataSignature ?? null,
      fileRevision: model.fileRevision ?? null,
    },
    requested: req,
    estimated: null,
    effective: null,
    vram: nullVramObservation({ source: 'pending', observedAt: createdAt }),
    cpu: { loadPct: null, privateCommitMib: null },
    ram: { totalMib: null, availableMib: null },
    processes: [],
    cachePolicy,
    policyRevision,
    allocLog: null,
    comparabilityKey: observationIdentity({
      backendKind: backend.kind,
      binarySha256: backend.binarySha256 ?? null,
      metadataSignature: model.metadataSignature,
      context: req.context,
      kType: req.kType,
      vType: req.vType,
      gpuPlacement: req.gpuPlacement,
      memoryMode: req.memoryMode,
    }),
  };
}

export function sealRunObservation(observation, {
  effective = {},
  vram = null,
  cpu = null,
  ram = null,
  processes = null,
  allocLog = null,
  estimated = null,
  sealedAt = new Date().toISOString(),
} = {}) {
  const argv = effective.argv ?? null;
  return {
    ...observation,
    sealedAt,
    estimated: estimated ?? observation.estimated,
    effective: {
      context: effective.context ?? null,
      argvDigest: digestArgv(argv),
      argv: Array.isArray(argv) ? argv.map(String) : null,
      gpuLayers: effective.gpuLayers ?? null,
      offloadReported: effective.offloadReported ?? 'unknown',
      binarySha256: effective.binarySha256 ?? observation.backend?.binarySha256 ?? null,
      batch: effective.batch ?? observation.requested?.batch ?? null,
      ubatch: effective.ubatch ?? observation.requested?.ubatch ?? null,
      parallel: effective.parallel ?? observation.requested?.parallel ?? null,
      kType: effective.kType ?? observation.requested?.kType ?? null,
      vType: effective.vType ?? observation.requested?.vType ?? null,
    },
    vram: vram ?? observation.vram,
    cpu: cpu ?? observation.cpu,
    ram: ram ?? observation.ram,
    processes: processes ?? observation.processes,
    allocLog: allocLog ?? observation.allocLog,
  };
}

/** Extract allocation facts from backend logs. Unmatched fields stay null. */
export function parseBackendAllocLog(text) {
  const raw = String(text ?? '');
  const out = {
    effectiveContext: null,
    gpuLayersOffloaded: null,
    gpuLayersTotal: null,
    kvAllocatedMib: null,
    weightsAllocatedMib: null,
    recurrentAllocatedMib: null,
    computeAllocatedMib: null,
    oom: /out of memory|\boom\b|cuda(?:error)?.*memory/i.test(raw),
  };

  const ctx = raw.match(/available context size\s*=\s*(\d+)\s*tokens/i)
    ?? raw.match(/(?:max[_\s-]?context|context size)\s*[:=]\s*(\d+)/i);
  if (ctx) out.effectiveContext = Number(ctx[1]);

  const offload = raw.match(/offloaded\s+(\d+)\s*\/\s*(\d+)\s+layers\s+to\s+GPU/i);
  if (offload) {
    out.gpuLayersOffloaded = Number(offload[1]);
    out.gpuLayersTotal = Number(offload[2]);
  }

  const kv = raw.match(/KV buffer size\s*=\s*([\d.]+)\s*MiB/i);
  if (kv) out.kvAllocatedMib = Number(kv[1]);

  const weights = raw.match(/(?:model|weights?)\s+(?:buffer|memory)\s+size\s*=\s*([\d.]+)\s*MiB/i);
  if (weights) out.weightsAllocatedMib = Number(weights[1]);

  const recurrent = raw.match(/recurrent\s+(?:buffer|state|memory)\s+size\s*=\s*([\d.]+)\s*MiB/i);
  if (recurrent) out.recurrentAllocatedMib = Number(recurrent[1]);

  const compute = raw.match(/compute\s+buffer\s+size\s*=\s*([\d.]+)\s*MiB/i);
  if (compute) out.computeAllocatedMib = Number(compute[1]);

  return out;
}

export function compareContextCapacity(observation) {
  const requested = observation?.requested?.context ?? null;
  const effective = observation?.effective?.context ?? null;
  if (requested === null) {
    return { ok: false, requested, effective, reason: 'requested context missing' };
  }
  if (effective === null) {
    return { ok: false, requested, effective, reason: 'effective context unknown — do not treat as comparable' };
  }
  if (effective < requested) {
    return { ok: false, requested, effective, reason: `effective context ${effective} is smaller than requested ${requested} (shrunk; non-comparable)` };
  }
  return { ok: true, requested, effective, reason: null };
}
