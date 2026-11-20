/** Q36_STARTUP lines from strata-q36. A later progress line does not clear an error. */

const STAGES = new Set([
  'identity', 'dense_load', 'buffers', 'expert_model_verify', 'expert_pack_verify', 'expert_host_alloc', 'expert_repack', 'expert_upload',
  'warmup_prefill', 'warmup_decode', 'warmup_spec', 'engine_ready',
]);
const STATES = new Set(['begin', 'progress', 'end', 'error', 'cancelled']);
const DETAIL_STAGES = new Set(['model_hash', 'pack_payload_verify', 'pin_pool_alloc', 'initial_slot_h2d', 'hybrid_buffers', 'cpu_pool_init']);

function record(line, prefix) {
  const match = String(line ?? '').trimEnd().match(new RegExp(`(?:^|\\s)${prefix}\\s+(.+)$`));
  if (!match) return null;
  const fields = Object.fromEntries([...match[1].matchAll(/(\w+)=([^\s]+)/g)].map(m => [m[1], m[2]]));
  return fields.schema === '1' ? fields : null;
}
const number = value => value != null && /^\d+$/.test(value) && Number.isSafeInteger(Number(value)) ? Number(value) : null;
const decimal = value => value != null && /^\d+$/.test(value) ? value : null;

export function parseQ36DetailLine(line) {
  const f = record(line, 'Q36_STARTUP_DETAIL');
  if (!f) return null;
  return { accepted: DETAIL_STAGES.has(f.stage) && STATES.has(f.state), schema: 1, stage: f.stage, state: f.state,
    generation: f.generation ?? null, process_elapsed_ms: number(f.process_elapsed_ms), stage_elapsed_ms: number(f.stage_elapsed_ms),
    completed_bytes: decimal(f.completed_bytes), total_bytes: decimal(f.total_bytes),
    layer_index: f.layer_index === '-1' ? null : number(f.layer_index), error_code: f.error_code === 'null' ? null : f.error_code };
}

export function parseQ36EffectiveLine(line) {
  const f = record(line, 'Q36_EFFECTIVE');
  if (!f) return null;
  const out = { schema: 1, generation: f.generation ?? null };
  for (const key of ['context', 'prefill', 'parallel', 'cpu_workers', 'mtp', 'kv_layout_version', 'kv_heads', 'kv_head_dim', 'kv_attention_states']) out[key] = number(f[key]);
  for (const key of ['engine', 'kv_k', 'kv_v', 'kv_scale_schema', 'graph_mode', 'graph_reason', 'warmup', 'expert_store', 'pin_mode', 'pin_pipeline', 'file_staging', 'memory_stage']) {
    out[key] = f[key] && f[key] !== 'null' ? f[key] : null;
  }
  out.slots = f.slots && /^\d+(,\d+)*$/.test(f.slots) ? f.slots.split(',').map(number) : null;
  if (out.slots?.some(value => value == null)) out.slots = null;
  for (const key of ['kv_bytes', 'kv_k_allocation_bytes', 'kv_v_allocation_bytes', 'recurrent_bytes', 'expert_host_bytes', 'expert_mapped_bytes', 'expert_device_bytes', 'pin_pool_bytes', 'hybrid_host_bytes', 'hybrid_device_bytes', 'vram_free_bytes', 'vram_total_bytes', 'ram_free_bytes', 'ram_total_bytes']) out[key] = decimal(f[key]);
  return out;
}

const SENTENCE = {
  model_hash: 'Verifying the model file',
  pack_payload_verify: 'Verifying the expert cache payload',
  pin_pool_alloc: 'Allocating the transfer pool',
  initial_slot_h2d: 'Copying initial expert slots to the GPU',
  hybrid_buffers: 'Allocating hybrid buffers',
  cpu_pool_init: 'Preparing CPU workers',
  dense_load: 'Loading weights onto the GPU',
  buffers: 'Preparing the context',
  expert_model_verify: 'Verifying the model file',
  expert_pack_verify: 'Verifying the expert cache',
  expert_host_alloc: 'Allocating expert memory',
  expert_repack: 'Arranging experts',
  expert_upload: 'Copying experts to the GPU',
  warmup_prefill: 'Warming the prompt path',
  warmup_decode: 'Warming decode',
  warmup_spec: 'Warming speculative decode',
  engine_ready: 'Engine ready',
};

export function parseQ36StartupLine(line) {
  const fields = record(line, 'Q36_STARTUP');
  if (!fields) return null;
  const stage = fields.stage ?? null;
  const state = fields.state ?? null;
  const accepted = STAGES.has(stage) && STATES.has(state);
  const num = key => {
    if (fields[key] == null || fields[key] === 'null') return null;
    const n = Number(fields[key]);
    return Number.isFinite(n) ? n : null;
  };
  return {
    accepted,
    schema: 1,
    stage,
    state,
    elapsed_ms: num('elapsed_ms'),
    stage_elapsed_ms: num('stage_elapsed_ms'),
    error_code: fields.error_code && fields.error_code !== 'null' ? fields.error_code : null,
    generation: fields.generation && fields.generation !== 'null' ? fields.generation : null,
  };
}

/** Keep the error. Ignore a different generation. Unknown events do not move the previous one. */
export function reduceQ36Startup(prev, event) {
  if (!event?.accepted) return prev ?? null;
  if (prev?.accepted && prev.state === 'error' && event.state !== 'error') return prev;
  if (prev?.generation && event.generation && prev.generation !== event.generation) return prev;
  return event;
}

export function q36StartupSentence(event) {
  if (!event?.accepted) return null;
  const label = SENTENCE[event.stage] ?? event.stage;
  if (event.state === 'error') {
    const why = event.error_code ? `: ${event.error_code}` : '';
    return `${label} failed${why}.`;
  }
  if (event.state === 'end' && event.stage_elapsed_ms != null) return `${label} finished in ${(event.stage_elapsed_ms / 1000).toFixed(1)} s.`;
  if (event.state === 'begin' || event.state === 'progress') return `${label}…`;
  if (event.state === 'cancelled') return `${label} was cancelled.`;
  return `${label}.`;
}

const WARMING = new Set(['warmup_prefill', 'warmup_decode', 'warmup_spec']);

/** Coarse report of an accepted Q36_STARTUP stage. Names stay loading, warming, ready, and error. */
export function q36ReportPhase(event) {
  if (!event?.accepted) return null;
  if (event.state === 'error') return 'error';
  if (event.stage === 'engine_ready') return 'ready';
  if (WARMING.has(event.stage)) return 'warming';
  return 'loading';
}

/**
 * The engine writes Q36_STARTUP to its stderr. serve opens the run config's `log` as that stderr.
 * strata_control/runtime.py `_paths` returns `{profile}.host.log` (the owner host) and the server
 * process writes `{profile}.server.log` beside the state file. Neither of those is the engine log.
 */
export function q36EngineLogPath(cfg) {
  const log = cfg?.log;
  if (typeof log !== 'string') return null;
  const path = log.trim();
  if (!path || path.includes('\0')) return null;
  return path;
}

/** Feed one chunk of the engine log. Unknown stages are ignored. A later progress line does not clear an error. */
export function pushEngineLogChunk(prev, chunk) {
  const text = `${prev?.pending ?? ''}${chunk ?? ''}`;
  const parts = text.split('\n');
  const pending = parts.pop() ?? '';
  let startup = prev?.startup ?? null;
  let phase = prev?.phase ?? null;
  let sentence = prev?.sentence ?? null;
  const launchGeneration = prev?.launchGeneration ?? null;
  let child = prev?.child ?? null;
  let detail = prev?.detail ?? null;
  let lastCompletedDetail = prev?.lastCompletedDetail ?? null;
  let effective = prev?.effective ?? null;
  let moved = false;
  for (const line of parts) {
    const receipt = record(line, 'Q36_CHILD');
    if (receipt && launchGeneration && receipt.launch_generation === launchGeneration && /^[a-zA-Z0-9-]+$/.test(receipt.generation ?? '')) {
      if (receipt.state === 'begin') {
        child = { generation: receipt.generation, processId: null };
        startup = detail = effective = lastCompletedDetail = null;
        phase = 'loading';
        sentence = null;
        moved = true;
      } else if (receipt.state === 'spawned' && child?.generation === receipt.generation && number(receipt.process_id) > 0) {
        child = { ...child, processId: number(receipt.process_id) };
        if (effective?.generation === child.generation) effective = { ...effective, processId: child.processId };
        moved = true;
      }
      continue;
    }
    const observed = parseQ36EffectiveLine(line);
    if (observed && child?.generation === observed.generation && detail?.state !== 'error' && startup?.state !== 'error') {
      effective = { ...effective, ...Object.fromEntries(Object.entries(observed).filter(([, value]) => value != null)), processId: child.processId };
      moved = true;
      continue;
    }
    const substage = parseQ36DetailLine(line);
    if (substage?.accepted && child?.generation === substage.generation && detail?.state !== 'error' && startup?.state !== 'error') {
      detail = substage;
      if (substage.state === 'end') lastCompletedDetail = substage;
      phase = substage.state === 'error' ? 'error' : 'loading';
      sentence = q36StartupSentence(substage);
      moved = true;
      continue;
    }
    const event = parseQ36StartupLine(line);
    if (!event?.accepted) continue;
    // Legacy engines have no generation; retain their coarse progress only.
    if (launchGeneration && event.generation && event.generation !== child?.generation) continue;
    if (detail?.state === 'error') continue;
    const next = reduceQ36Startup(startup, event);
    if (next === startup) continue;
    startup = next;
    const reported = q36ReportPhase(next);
    if (reported) phase = reported;
    const said = q36StartupSentence(next);
    if (said && next === event) sentence = said;
    moved = true;
  }
  return { pending, startup, phase, sentence, moved, launchGeneration, child, detail, lastCompletedDetail, effective };
}
