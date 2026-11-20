/** Separate dense Qwen3.5 P0 admission measurement; never starts a model. */
import {createHash} from 'node:crypto';
import {readFileSync, statSync, writeFileSync, mkdirSync} from 'node:fs';
import {freemem, totalmem, cpus} from 'node:os';
import {join} from 'node:path';
import {setTimeout} from 'node:timers/promises';
import {componentsDirectory} from '../../app/host/server/control-runtime.mjs';
import {findLocalBackend, verifyLocalBackendSource} from '../../app/host/server/local-backends.mjs';
import {admitLocalMemory, RAM_FLOOR_BYTES, RAM_STOP_BYTES} from '../../app/host/server/local-memory-guard.mjs';

const name = process.argv[2];
if (!/^[a-zA-Z0-9_-]{1,80}$/.test(name ?? '')) throw new Error('Fresh run name required');
const output = join('H:/Strata/evidence/ram32gb-p0-resume-20261115', name);
mkdirSync(output);
const save = (name, value) => writeFileSync(join(output, name), JSON.stringify(value, null, 2) + '\n', {flag: 'wx'});
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const profilePaths = [
  'C:/Users/downl/.strata/data/state.json',
  'C:/Users/downl/.strata/data/local-models.json',
  'C:/Users/downl/.strata/data/native-host.json',
  'C:/Users/downl/AppData/Roaming/Strata/settings.json',
];
const profileHashes = () => Object.fromEntries(profilePaths.map(path => [path, sha(readFileSync(path))]));
const before = profileHashes();
const model = JSON.parse(readFileSync(profilePaths[1], 'utf8')).models
  .find(item => item.id === 'local-cbca9a0eca0f30752f82f2ce');
if (model?.local?.architecture !== 'qwen35' || model.quant !== 'Q4_K_M' || model.experts !== 0)
  throw new Error('Explicit dense Qwen3.5 Q4_K_M identity required');
const modelBytes = statSync(model.local.path).size;
if (modelBytes !== 5629111584) throw new Error('Declared model size changed');
const runtime = findLocalBackend('PrismML', {componentsDir: componentsDirectory()});
if (!runtime) throw new Error('Existing PrismML runtime unavailable');
const manifestBytes = readFileSync(runtime.manifestPath);
const runtimeManifest = JSON.parse(manifestBytes);
const {approved} = verifyLocalBackendSource(runtimeManifest, 'PrismML');
if (approved.manifestSHA256 && sha(manifestBytes) !== approved.manifestSHA256)
  throw new Error('Approved runtime manifest digest mismatch');

// Reuse the production admission calculation, with the prior bounded 9B run's
// no-mmap policy. Do not reduce its staging allowance or try loading on refusal.
const loadPolicy = {fullyOffloaded: true, modelBytes, mapped: false};
const samples = [];
for (let i = 0; i < 21; i++) {
  samples.push({at: new Date().toISOString(), availableBytes: freemem()});
  if (i < 20) await setTimeout(250);
}
const available = samples.at(-1).availableBytes;
let admitted = true, admissionError = null;
try { admitLocalMemory(available, loadPolicy); }
catch (error) { admitted = false; admissionError = error.message; }
const after = profileHashes();
const unchanged = JSON.stringify(before) === JSON.stringify(after);
const result = {
  schema_version: 1, run_id: name, family: 'qwen35_dense', backend: 'PrismML',
  scope: 'separate_existing_9b_track_not_q36_baseline',
  status: admitted ? 'ADMITTED_NOT_STARTED' : 'ADMISSION_REJECTED',
  admission: {admitted, availableBytes: available, requiredExclusiveBytes: RAM_STOP_BYTES + modelBytes,
    deficitBytes: Math.max(0, RAM_STOP_BYTES + modelBytes + 1 - available),
    loadPolicy, error: admissionError, implementation: 'apps/host/server/local-memory-guard.mjs'},
  model: {id: model.id, architecture: model.local.architecture, quant: model.quant,
    path: model.local.path, bytes: modelBytes, declaredMetadataSignature: model.local.metadataSignature,
    sha256: null, hashStatus: 'NOT_RUN_ADMISSION_ONLY'},
  runtime: {path: runtime.path, version: runtimeManifest.version,
    declaredSourceCommit: runtimeManifest.source.commit, manifestSHA256: sha(manifestBytes),
    manifestApproved: true, payloadVerification: 'NOT_RUN_ADMISSION_ONLY'},
  proposedSettings: {contextCapacity: 65536, actualInputTokens: null, parallel: 1,
    mmap: false, batch: 256, ubatch: 128, gpuLayers: 999, kvK: 'f16', kvV: 'f16'},
  hardware: {cpu: cpus()[0]?.model, totalPhysicalBytes: totalmem()},
  protection: {physicalFloorBytes: RAM_FLOOR_BYTES, stopBaseBytes: RAM_STOP_BYTES,
    dynamicStopMaximumBytes: 8 * 1024 ** 3, commitFloorBytes: 4 * 1024 ** 3,
    gpuReserveMinimumBytes: 768 * 1024 ** 2, HFreeFloorBytes: 10 * 1024 ** 3, CFreeFloorBytes: 5 * 1024 ** 3},
  profileHashesBefore: before, profileHashesAfter: after, profileUnchanged: unchanged,
  modelStarted: false, requestsStatus: 'NOT_RUN', performance_eligible: false, releaseQualified: false,
};
save('run_manifest.json', result);
writeFileSync(join(output, 'phase_metrics.jsonl'), samples.map(row => JSON.stringify({phase: 'admission',
  scope: 'system_available_physical_ram', ...row})).join('\n') + '\n', {flag: 'wx'});
writeFileSync(join(output, 'report.md'), `# Separate 9B P0 admission\n\nStatus: ${result.status}.\n\n` +
  `Available RAM: ${available} bytes. Existing no-mmap GPU-load allowance requires more than ${RAM_STOP_BYTES + modelBytes} bytes.\n\n` +
  'No model process or inference request was started by this admission measurement. Load, prefill, decode, cancel and unload remain NOT_RUN. This is not a Q36 comparison or a P1 pass.\n\n' +
  `Original profile hashes unchanged: ${unchanged}. Performance eligibility and release qualification remain false.\n`, {flag: 'wx'});
console.log(JSON.stringify({status: result.status, availableBytes: available,
  requiredExclusiveBytes: result.admission.requiredExclusiveBytes, profileUnchanged: unchanged}));
if (!unchanged) process.exitCode = 1;
