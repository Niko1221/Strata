/** Idle baseline capture helper for run-windows.ps1 (PR-A). */
import { readFileSync, writeFileSync } from 'node:fs';
import { createRunObservation, sealRunObservation } from '../../apps/host/server/run-observation.mjs';
import { captureVramObservation } from '../../apps/host/server/vram-observation.mjs';

const pinsPath = process.argv[2];
const outPath = process.argv[3];
if (!pinsPath || !outPath) {
  console.error('usage: node capture-baseline.mjs <pins.json> <out.json>');
  process.exit(2);
}

const pins = JSON.parse(readFileSync(pinsPath, 'utf8').replace(/^\uFEFF/, ''));
const vram = captureVramObservation();
const obs = createRunObservation({
  backend: { kind: 'diagnostic', version: null, binarySha256: null },
  model: { id: 'baseline-idle', metadataSignature: null, fileRevision: pins.hikariRevision ?? null },
  requested: {
    context: 131072, kType: 'q8_0', vType: 'q8_0', gpuPlacement: 'none',
    ngl: null, batch: 2048, ubatch: 512, parallel: 1, flashAttention: true,
    expertCache: null, memoryMode: 'ordinary',
  },
  cachePolicy: 'diagnostic-idle',
  policyRevision: 'pr-a-2026-11-02',
});
const sealed = sealRunObservation(obs, {
  effective: { context: null, argv: null, gpuLayers: null, offloadReported: 'unknown', binarySha256: null },
  vram,
  processes: pins.processes ?? [],
  ram: pins.ram ?? null,
  cpu: pins.cpu ?? null,
});
const payload = {
  captureKind: 'baseline-idle',
  pins,
  observation: sealed,
  notes: [
    'Idle baseline: no model loaded by this script.',
    'Dedicated VRAM and shared GPU memory are separate fields.',
    'Do not treat this as a speed result.',
  ],
};
writeFileSync(outPath, JSON.stringify(payload, null, 2));
console.log(outPath);
