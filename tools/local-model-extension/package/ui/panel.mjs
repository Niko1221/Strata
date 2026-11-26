import { createApi } from './api.mjs';
import { compatible, contextLabel, idleLabel, createPanelController } from './state.mjs';

function apiLabel(value) {
  try { const url=new URL(value);return `Local API · ${url.port || '80'}`; }
  catch { return 'Local API'; }
}

/** All API-derived copy is text, including errors and the physical model ID. */
export function renderPanel(doc, state) {
  const { model: data, status, view } = state;
  const text = (id, value) => { doc.getElementById(id).textContent = value; };
  const show = (id, value) => { const element = doc.getElementById(id); element.textContent = value; element.hidden = !value; };
  text('connection', compatible(status) ? view.available ? 'Connected Locally' : 'Carrier Unavailable' : status ? 'Extension Unavailable' : state.error ? 'Connection Unavailable' : 'Checking Connection');
  doc.getElementById('connection').dataset.ready = String(view.available);
  text('model-name', typeof data?.model?.name === 'string' ? data.model.name : 'Model Unconfirmed');
  text('model-variant', typeof data?.model?.variant === 'string' ? data.model.variant : 'Variant Unconfirmed');
  text('model-state', state.action ? { load: 'Loading Model…', unload: 'Unloading Model…', connect: 'Opening Chat…', context: 'Saving Context…', idle: 'Saving Auto Unload…' }[state.action] : view.stateLabel);
  text('context-fact', contextLabel(view.effective)); text('idle-fact', idleLabel(view.idleEffective));
  text('api-label', apiLabel(data?.model?.apiBaseUrl));
  for (const action of ['connect', 'load', 'unload']) doc.getElementById(action).disabled = !view.can[action];
  const select = doc.getElementById('context-choice');
  const selectedIsListed = view.choices.includes(view.selected);
  const optionsKey = JSON.stringify([view.choices, selectedIsListed]);
  if (select.dataset.options !== optionsKey) {
    const options = view.choices.map(value => {
      const option = doc.createElement('option'); option.value = String(value); option.textContent = contextLabel(value); return option;
    });
    if (!selectedIsListed) {
      const placeholder = doc.createElement('option'); placeholder.value = ''; placeholder.disabled = true;
      placeholder.textContent = view.choices.length ? 'Choose A Context' : 'Context Choices Unavailable'; options.unshift(placeholder);
    }
    select.replaceChildren(...options); select.dataset.options = optionsKey;
  }
  select.value = selectedIsListed ? String(view.selected) : '';
  select.disabled = !view.can.context;
  for (const button of doc.querySelectorAll('[data-idle]')) {
    const value = Number(button.dataset.idle);
    button.setAttribute('aria-pressed', String(view.idleSelected === value));
    button.disabled = !view.can.idle || view.idleSelected === value;
  }
  text('context-selected', contextLabel(view.selected)); text('context-effective', contextLabel(view.effective));
  text('idle-selected', idleLabel(view.idleSelected)); text('idle-effective', idleLabel(view.idleEffective));
  text('context-pending', view.contextPending ? 'Waiting For The Selected Context To Become Effective.' : view.effective ? 'Your Saved Choice Is Restored Automatically.' : 'Effective Context Has Not Been Confirmed.');
  text('idle-pending', view.idlePending ? 'Waiting For The Selected Auto Unload Time To Become Effective.' : view.idleEffective !== null ? 'Your Next Message Loads The Model On Demand.' : 'Effective Auto Unload Has Not Been Confirmed.');
  text('control-hint', state.requested ? 'Request Accepted. Waiting For The Carrier To Confirm.' : view.busy ? 'Controls Are Available After The Current Action Finishes.' : view.canControl ? 'Use In Chat Selects This Model. Reasoning Effort Is Managed In Chat.' : 'Waiting For Confirmed, Idle Model State.');
  show('notice', state.notice); show('error', state.error || data?.error || '');
  text('extension-state', status?.enabled === true ? 'Enabled' : status?.enabled === false ? 'Disabled' : 'Unconfirmed');
  text('compatibility', compatible(status) ? 'Compatible · Bridge API 1' : status ? 'Not Confirmed' : 'Unconfirmed');
  text('api-url', typeof data?.model?.apiBaseUrl === 'string' ? data.model.apiBaseUrl : 'Unconfirmed');
  text('model-id', typeof data?.model?.id === 'string' ? data.model.id : 'Unconfirmed');
  text('active-requests', view.active === null ? 'Unconfirmed' : String(view.active));
  text('in-flight', view.inFlight === null ? 'Unconfirmed' : String(view.inFlight));
}

export function startPanel(win = window) {
  const doc = win.document, parent = win.parent, origin = win.location.origin;
  const send = (type, connection) => { if (parent !== win) parent.postMessage({ type, ...(connection ? { connection } : {}) }, origin); };
  const controller = createPanelController({ api: createApi({ origin, fetchImpl: win.fetch.bind(win) }),
    onState: state => renderPanel(doc, state), onOpenChat: connection => {
      if (parent === win) throw new Error('Open Current Model From The Strata App To Select A Chat Model.');
      send('strata-extension:open-chat', connection);
    } });
  const events = new AbortController(), options = { signal: events.signal };
  let panelVisible = parent === win;
  const visibility = () => controller.setVisible(panelVisible && doc.visibilityState !== 'hidden');
  win.addEventListener('message', event => {
    if (event.origin !== origin || event.source !== parent) return;
    if (event.data?.type === 'strata-extension:selection-error') { controller.reportError(event.data.message); return; }
    if (event.data?.type !== 'strata-extension:visibility') return;
    if (typeof event.data.visible !== 'boolean') return;
    panelVisible = event.data.visible;
    if (['light', 'dark'].includes(event.data.theme)) doc.documentElement.dataset.theme = event.data.theme;
    visibility();
  }, options);
  doc.addEventListener('visibilitychange', visibility, options);
  doc.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !event.defaultPrevented) { event.preventDefault(); send('strata-extension:close'); }
  }, options);
  for (const action of ['connect', 'load', 'unload']) doc.getElementById(action).addEventListener('click', () => void controller.perform(action), options);
  doc.getElementById('context-choice').addEventListener('change', event => void controller.perform('context', Number(event.currentTarget.value)), options);
  for (const button of doc.querySelectorAll('[data-idle]')) button.addEventListener('click', () => void controller.perform('idle', Number(button.dataset.idle)), options);
  doc.getElementById('refresh').addEventListener('click', () => void controller.refresh(), options);
  doc.getElementById('close').addEventListener('click', () => send('strata-extension:close'), options);
  const dispose = () => { events.abort(); controller.dispose(); };
  win.addEventListener('pagehide', dispose, { once: true });
  visibility();
  return dispose;
}
if (typeof window !== 'undefined' && window.document) startPanel();
