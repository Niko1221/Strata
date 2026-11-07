import { createApi } from './api.mjs';
import { createPanelController, contextLabel } from './state.mjs';
import { visible } from './official.mjs';

/** One verified carrier model, distinct from the app's prepared/downloaded library. */
export function connectedTile({ win, target, kind, onUseInChat, theme, api = createApi({ origin: win.location.origin, fetchImpl: win.fetch.bind(win) }) }) {
  const doc = win.document, host = doc.createElement('strata-connected-local-models'), root = host.attachShadow({ mode: 'open' });
  host.dataset.theme = theme;
  const styles = doc.createElement('link'); styles.rel = 'stylesheet'; styles.href = '/__strata_extension/ui/connected.css';
  const section = doc.createElement('section'); section.setAttribute('aria-label', 'Connected Local Models');
  const heading = doc.createElement('h3'); heading.textContent = 'Connected Local Models';
  const note = doc.createElement('p'); note.className = 'note';
  note.textContent = 'Files Managed By Local Server. This Connection Is Separate From Native Prepared Models; No Download Or File Copy Is Needed.';
  const label = doc.createElement('p'); label.className = 'muted'; label.textContent = 'Connected Local Model';
  const name = doc.createElement('strong'), variant = doc.createElement('p'), id = doc.createElement('code'), state = doc.createElement('p'), folder = doc.createElement('p');
  name.textContent = 'Checking Connected Model…'; variant.className = 'muted'; state.setAttribute('role', 'status'); folder.className = 'muted'; folder.hidden = true;
  const actions = doc.createElement('div'); actions.className = 'actions';
  const use = doc.createElement('button'), load = doc.createElement('button'), unload = doc.createElement('button'), refresh = doc.createElement('button');
  use.textContent = 'Use In Chat'; use.className = 'primary'; load.textContent = 'Preload Model'; unload.textContent = 'Unload Model'; refresh.textContent = 'Refresh Status';
  for (const button of [use, load, unload, refresh]) { button.type = 'button'; if (button !== refresh) button.disabled = true; }
  const error = doc.createElement('p'); error.className = 'error'; error.setAttribute('role', 'alert'); error.hidden = true;
  const notice = doc.createElement('p'); notice.className = 'muted'; notice.setAttribute('role', 'status'); notice.hidden = true;
  actions.append(use, load, unload, refresh); section.append(heading, note, label, name, variant, id, state, folder, actions, error, notice); root.append(styles, section);
  const controller = createPanelController({ api, onOpenChat: connection => onUseInChat(connection, host), onState: snapshot => {
    const data = snapshot.model, view = snapshot.view;
    name.textContent = view.available && typeof data?.model?.name === 'string' ? data.model.name : 'Connected Model Unconfirmed';
    variant.textContent = view.available && typeof data?.model?.variant === 'string' ? data.model.variant : 'Variant Unconfirmed';
    id.textContent = view.available && typeof data?.model?.id === 'string' ? data.model.id : 'Physical Model ID Unconfirmed';
    state.textContent = `${view.loaded === true ? 'Loaded' : view.loaded === false ? 'Unloaded' : 'Loaded State Unconfirmed'} · ${view.stateLabel} · Effective Context: ${contextLabel(view.effective)}`;
    folder.textContent = typeof data?.model?.filesDirectory === 'string' ? `Model Files: ${data.model.filesDirectory}` : '';
    folder.hidden = !view.available || !folder.textContent;
    use.disabled = !view.can.connect; load.disabled = !view.can.load; unload.disabled = !view.can.unload;
    error.textContent = snapshot.error || data?.error || ''; error.hidden = !error.textContent;
    notice.textContent = snapshot.notice; notice.hidden = !snapshot.notice;
  } });
  const events = new AbortController(), options = { signal: events.signal };
  use.addEventListener('click', () => void controller.perform('connect'), options);
  load.addEventListener('click', () => void controller.perform('load'), options);
  unload.addEventListener('click', () => void controller.perform('unload'), options);
  refresh.addEventListener('click', () => void controller.refresh(), options);
  if (kind === 'manage') target.insertBefore(host, target.querySelector('.native-manage-foot'));
  else target.append(host);
  return { host, controller,
    sync() { controller.setVisible(win.document.visibilityState !== 'hidden' && visible(win, target) && host.isConnected); },
    dispose() { events.abort(); controller.dispose(); host.remove(); },
  };
}

/** DOM-only compatibility anchors. Never edit native rows, counts or directory fields. */
export function observeConnectedModels({ win, onUseInChat, onError = () => {}, theme = 'dark', apiFactory } = {}) {
  const doc = win.document, records = new Map(), rejected = new WeakSet(), events = new AbortController();
  let disposed = false, currentTheme = theme;
  function reconcile() {
    if (disposed) return;
    for (const [target, tile] of records) {
      if (!target.isConnected || !tile.host.isConnected) { tile.dispose(); records.delete(target); }
    }
    for (const anchor of doc.querySelectorAll('.native-manage, .storage-directory')) {
      const kind = anchor.matches('.native-manage') ? 'manage' : 'storage';
      const target = kind === 'manage' ? anchor : anchor.closest('[role="tabpanel"]') ?? anchor.parentElement;
      if (records.has(target) || !visible(win, anchor)) continue;
      const foot = target.querySelector('.native-manage-foot');
      if (kind === 'manage' && (!target.closest('[role="dialog"]') || foot?.parentElement !== target)) {
        if (!rejected.has(target)) { rejected.add(target); onError(new Error('Manage Models Layout Is Unsupported. Use Current Model To Access The Connected Model.')); }
        continue;
      }
      rejected.delete(target);
      try {
        records.set(target, connectedTile({ win, target, kind, onUseInChat, theme: currentTheme, ...(apiFactory ? { api: apiFactory() } : {}) }));
      } catch (error) { onError(error); }
    }
    for (const tile of records.values()) tile.sync();
  }
  const observer = new win.MutationObserver(reconcile);
  observer.observe(doc.documentElement, { childList: true, subtree: true, attributes: true,
    attributeFilter: ['hidden', 'aria-hidden', 'inert', 'class', 'style', 'open'] });
  doc.addEventListener('visibilitychange', reconcile, { signal: events.signal });
  reconcile();
  return { reconcile,
    setTheme(value) { currentTheme = value; for (const tile of records.values()) tile.host.dataset.theme = value; },
    dispose() { disposed = true; observer.disconnect(); events.abort(); for (const tile of records.values()) tile.dispose(); records.clear(); },
  };
}
