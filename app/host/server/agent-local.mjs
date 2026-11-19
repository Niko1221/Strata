/**
 * Local models the agent can drive: native Strata instances and a running
 * Prism/TurboQuant llama-server (supervisor). Hikari-style local GGUFs live on
 * the supervisor path — without them, web_search never reaches the model.
 */
import { localOutputBudget, compactionHeadroom } from './agent/profile.mjs';

/** @param {{status:string, modelId?:string|null, port?:number|null}} engine */
export function supervisorIsReady(engine) {
  return !!engine && engine.status === 'running'
    && typeof engine.modelId === 'string' && engine.modelId
    && Number.isInteger(engine.port) && engine.port > 0;
}

/**
 * Shape one supervisor-backed instance for the agent profile.
 * @param {{status:string, modelId:string, port:number, contextCapacity?:number|null}} engine
 * @param {{authenticationKey?:()=>string|null, servingApiKey?:string|null, label?:string|null, vision?:boolean}} [opts]
 */
export function supervisorLocalInstance(engine, opts = {}) {
  if (!supervisorIsReady(engine)) return null;
  // Model metadata describes training capacity, not this process's usable slot.
  // An unknown runtime window must not reach the profile's default window.
  if (!Number.isSafeInteger(engine.contextCapacity) || engine.contextCapacity <= 0) return null;
  const output = localOutputBudget(engine.contextCapacity);
  // Preserve the profile's reserves and require room for a prompt. This is
  // only admission arithmetic; the actual system/tools/message still must fit.
  if (engine.contextCapacity <= output + compactionHeadroom(engine.contextCapacity, output)) return null;
  const key = opts.authenticationKey?.() || opts.servingApiKey || null;
  return {
    id: `supervisor:${engine.modelId}`,
    label: opts.label || engine.modelId,
    baseUrl: `http://127.0.0.1:${engine.port}/v1`,
    // llama-server accepts any model id; inference uses "local" for this path.
    modelId: 'local',
    // Serving/unload accounting consumes the native instance shape too. A
    // supervisor route has no measured per-GPU attribution; do not invent one.
    apiModelId: 'local',
    gpus: [],
    context: engine.contextCapacity,
    key,
    vision: opts.vision === true,
    slots: 1,
    mcpServers: undefined,
    source: 'supervisor',
    variantId: engine.modelId,
  };
}

/** Native ready rows already shaped for the agent, plus an optional supervisor row. */
export function mergeAgentLocalInstances(nativeRows = [], supervisorRow = null) {
  const out = [];
  const seen = new Set();
  for (const row of nativeRows) {
    if (!row?.id || seen.has(row.id)) continue;
    seen.add(row.id);
    out.push(row);
  }
  if (supervisorRow?.id && !seen.has(supervisorRow.id)) out.push(supervisorRow);
  return out;
}

/** True when the agent has a local model or an API model to talk to. */
export function agentHasLocalModel({ remoteModel = false, nativeLive = false, supervisorLive = false } = {}) {
  return !!(remoteModel || nativeLive || supervisorLive);
}

export function agentSkipReason({
  agentOff = false,
  plainChat = false,
  chatId = null,
  remoteModel = false,
  nativeLive = false,
  supervisorLive = false,
  supervisorContextUnavailable = false,
} = {}) {
  if (agentOff) return 'The agent is turned off (STRATA_AGENT=off).';
  if (plainChat) return 'This chat runs without the agent.';
  if (!chatId) return 'This request did not name a chat.';
  if (!agentHasLocalModel({ remoteModel, nativeLive, supervisorLive })) {
    if (supervisorContextUnavailable) return 'The local model is ready, but its context size is unavailable or too small for the agent\'s reserved budgets. This reply uses plain chat; check the context setting, then unload and load the model again.';
    return 'The agent needs a loaded Strata model, a local GGUF (Prism/TurboQuant), or an API model.';
  }
  return null;
}
