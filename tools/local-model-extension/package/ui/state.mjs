import { actionBody, contextChoices, validContext, IDLES } from './api.mjs';
import { connectedSelection } from './official.mjs';

const choice = (value, choices) => choices.includes(value) ? value : null;
const count = value => Number.isSafeInteger(value) && value >= 0 ? value : null;
export const contextLabel = value => validContext(value) ? value % 1024 === 0 ? `${value / 1024}K` : `${value.toLocaleString('en-US')} Tokens` : 'Unconfirmed';
export const idleLabel = value => value === 0 ? 'Never' : IDLES.includes(value) ? `${value / 60} Minutes` : 'Unconfirmed';
export const compatible = status => status?.enabled === true && status.compatible === true
  && status.bridgeApi === 1 && status.extension?.id === 'strata-local-models';

/** Never infer effective capacity, loaded state or idle settings from a selection. */
export function modelView({ model: data, status, action = null, requested = null } = {}) {
  const available = compatible(status) && data?.available === true;
  const choices = contextChoices(data?.context?.choices);
  const selected = validContext(data?.context?.selected) ? data.context.selected : null;
  const effective = available && validContext(data?.context?.effective) ? data.context.effective : null;
  const idleSelected = choice(data?.autoUnload?.selected, IDLES), idleEffective = available ? choice(data?.autoUnload?.effective, IDLES) : null;
  const loaded = available && typeof data?.model?.loaded === 'boolean' ? data.model.loaded : null;
  const active = count(data?.context?.activeRequests), inFlight = count(data?.context?.inFlight);
  const contextPending = selected !== null && selected !== effective;
  const idlePending = idleSelected !== null && idleSelected !== idleEffective;
  const transitioning = data?.context?.phase !== 'ready';
  const busy = !!action || data?.model?.loading === true || (active ?? 0) > 0 || (inFlight ?? 0) > 0;
  const confirmed = available && typeof data?.model?.id === 'string' && !!data.model.id && effective !== null
    && selected !== null && loaded !== null && data?.model?.loading === false;
  const canControl = confirmed && !busy && !transitioning && !contextPending && !idlePending
    && active === 0 && inFlight === 0 && !requested;
  return { choices, selected, effective, idleSelected, idleEffective, available, loaded, active, inFlight,
    contextPending, idlePending, transitioning, busy, canControl,
    can: { context: canControl && choices.length > 0, idle: canControl && !!data?.autoUnload,
      load: canControl && loaded === false, unload: canControl && loaded === true, connect: canControl },
    stateLabel: !available ? 'Connection Unavailable' : data?.model?.loading === true ? 'Loading Model'
      : transitioning || contextPending || idlePending || requested ? 'Applying Settings'
      : busy ? 'In Use' : loaded === true ? 'Ready To Chat' : loaded === false ? 'Available On Demand' : 'State Unconfirmed',
  };
}

function observed(requested, model) {
  if (requested.modelId !== model?.model?.id) return false;
  if (requested.action === 'context') return model?.context?.selected === requested.value && model.context.effective === requested.value && model.context.phase === 'ready';
  if (requested.action === 'idle') return model?.autoUnload?.selected === requested.value && model.autoUnload.effective === requested.value && model.context?.phase === 'ready';
  return model?.model?.loaded === (requested.action === 'load') && model.model.loading === false;
}

/** Visible-only reads and explicit actions. Closing cancels HTTP, never retries a mutation. */
export function createPanelController({ api, onState = () => {}, onOpenChat = () => {},
  setTimer = setTimeout, clearTimer = clearTimeout, interval = 4000 } = {}) {
  const state = { model: null, status: null, action: null, requested: null, error: '', notice: '' };
  let visible = false, disposed = false, generation = 0, timer = null, reading = null, writing = null, readError = false;
  const emit = () => onState({ ...state, view: modelView(state) });
  const unschedule = () => { if (timer !== null) clearTimer(timer); timer = null; };
  function schedule() { if (visible && !disposed && !writing) timer = setTimer(() => { timer = null; void refresh(); }, interval); }
  async function refresh() {
    if (!visible || disposed || reading || writing) return;
    unschedule();
    const controller = new AbortController(), current = generation;
    reading = controller;
    try {
      const [status, model] = await Promise.all([api.get('status', controller.signal), api.get('model', controller.signal)]);
      if (controller.signal.aborted || current !== generation) return;
      state.status = status; state.model = model;
      if (readError) { state.error = ''; readError = false; }
      if (state.requested && model?.available === true && state.requested.modelId !== model?.model?.id) {
        state.requested = null; state.notice = ''; state.error = 'The Selected Model Changed. Refresh Before Acting.';
      }
      if (state.requested?.requestId && model?.request?.id === state.requested.requestId && model.request.state === 'error') {
        state.requested = null; state.notice = ''; state.error = model.error || 'The Model Owner Could Not Apply The Request.';
      }
      if (state.requested && compatible(status) && model?.available === true && observed(state.requested, model)) {
        state.requested = null; state.notice = 'Saved Settings Confirmed.';
      }
    } catch (error) {
      if (controller.signal.aborted || current !== generation) return;
      controller.abort();
      state.model = null; state.status = null; state.error = error.message || 'Connection Unavailable.'; readError = true;
    } finally {
      if (reading === controller) { reading = null; if (visible) { emit(); schedule(); } }
    }
  }
  async function perform(action, value) {
    // Validate before any IO, even if a caller bypasses a disabled control.
    actionBody(action, value);
    if (!visible || disposed || writing || !modelView(state).can[action]) return false;
    if (action === 'context' && !modelView(state).choices.includes(value)) throw new TypeError('Choose A Context Reported By The Model Catalog.');
    if ((action === 'context' && value === modelView(state).selected) || (action === 'idle' && value === modelView(state).idleSelected)) return false;
    unschedule(); reading?.abort(); reading = null;
    const controller = new AbortController(), current = generation;
    writing = controller; state.action = action; state.error = ''; state.notice = ''; readError = false; emit();
    try {
      const result = await api.post(action, value, controller.signal);
      if (controller.signal.aborted || current !== generation) return false;
      if (action === 'connect') await onOpenChat(connectedSelection(result, state.model?.model?.id));
      else {
        state.requested = { action, value, modelId: state.model?.model?.id, requestId: result?.requestId };
        state.notice = action === 'context' ? `${contextLabel(value)} Context Requested. Waiting For Confirmation.`
          : action === 'idle' ? `${idleLabel(value)} Auto Unload Requested. Waiting For Confirmation.`
          : action === 'load' ? 'Preload Requested. Waiting For Model Status.' : 'Unload Requested. Waiting For Model Status.';
      }
      return true;
    } catch (error) {
      if (!controller.signal.aborted && current === generation) state.error = error.message || 'The Action Could Not Be Completed.';
      return false;
    } finally {
      if (writing === controller) { writing = null; state.action = null; if (visible) { emit(); void refresh(); } }
    }
  }
  return { refresh, perform,
    reportError(message) { state.error = typeof message === 'string' ? message.slice(0, 2000) : 'Chat Selection Could Not Be Confirmed.'; if (visible && !disposed) emit(); },
    setVisible(value) {
      if (disposed || visible === value) return;
      visible = value; generation++; unschedule(); reading?.abort(); writing?.abort(); reading = writing = null;
      state.action = null; state.model = state.status = null;
      if (visible) { emit(); void refresh(); }
    },
    dispose() { this.setVisible(false); disposed = true; },
  };
}
