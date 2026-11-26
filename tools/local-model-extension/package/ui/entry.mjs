import { connectedSelection, selectInOfficialChat } from './official.mjs';
import { observeConnectedModels } from './connected.mjs';

const TAG = 'strata-local-models-extension';
const PANEL = '/__strata_extension/ui/index.html';

export function frameMessage(event, frame, origin) {
  return !!frame.contentWindow && event.origin === origin && event.source === frame.contentWindow
    && ['strata-extension:close', 'strata-extension:open-chat'].includes(event.data?.type) ? event.data.type : null;
}

export function detectTheme(win) {
  const root = win.document.documentElement;
  if (['light', 'dark'].includes(root.dataset.theme)) return root.dataset.theme;
  const scheme = win.getComputedStyle(root).colorScheme;
  if (scheme === 'light' || scheme === 'dark') return scheme;
  return win.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

/** Only a verified frame and explicit Use In Chat may hand off a connection. */
export function createDialogController({ win, dialog, frame, trigger, theme = () => detectTheme(win),
  onUseInChat = connection => selectInOfficialChat(win, connection), onError = () => {} }) {
  const events = new AbortController(), options = { signal: events.signal };
  let previous = null;
  function sync() {
    frame.contentWindow?.postMessage({ type: 'strata-extension:visibility',
      visible: dialog.open && win.document.visibilityState !== 'hidden', theme: theme() }, win.location.origin);
  }
  function close() { if (dialog.open) dialog.close(); sync(); }
  function restore() { trigger.setAttribute('aria-expanded', 'false'); (previous?.isConnected ? previous : trigger).focus(); sync(); }
  dialog.addEventListener('close', restore, options);
  dialog.addEventListener('cancel', event => { event.preventDefault(); close(); }, options);
  frame.addEventListener('load', sync, options);
  win.document.addEventListener('visibilitychange', sync, options);
  win.addEventListener('message', async event => {
    const type = frameMessage(event, frame, win.location.origin);
    if (!type || !dialog.open) return;
    if (type === 'strata-extension:close') { close(); return; }
    let connection;
    try { connection = connectedSelection(event.data.connection); }
    catch (error) {
      frame.contentWindow.postMessage({ type: 'strata-extension:selection-error', message: error.message }, win.location.origin);
      return;
    }
    close();
    try { await onUseInChat(connection); } catch (error) { onError(error); }
  }, options);
  return { sync, close,
    open() {
      if (dialog.open) return;
      previous = win.document.activeElement;
      while (previous?.shadowRoot?.activeElement) previous = previous.shadowRoot.activeElement;
      if (!frame.getAttribute('src')) frame.src = PANEL;
      dialog.showModal(); trigger.setAttribute('aria-expanded', 'true'); sync();
    },
    dispose() { close(); events.abort(); },
  };
}

/** Accepted handoffs outlive their source dialog, but never the extension itself. */
export function createSelectionCoordinator({ win, onState = () => {}, select = selectInOfficialChat }) {
  let active = null, disposed = false;
  return {
    async useInChat(value, source) {
      const connection = connectedSelection(value);
      if (disposed) throw new Error('The Local Model Extension Was Closed. Reopen It And Try Again.');
      if (active) throw new Error('A Chat Selection Is Already In Progress. Wait For It To Finish.');
      const controller = new AbortController(); active = controller;
      onState({ message: 'Selecting Connected Model In Chat…', busy: true, error: false });
      try {
        await select(win, connection, { source, signal: controller.signal });
        if (!controller.signal.aborted) onState({ message: 'Connected Model Selected In Chat.', busy: false, error: false });
      } catch (error) {
        if (!controller.signal.aborted) onState({ message: error.message || 'Chat Selection Could Not Be Confirmed. Open Current Model And Try Again.', busy: false, error: true });
        throw error;
      } finally { if (active === controller) active = null; }
    },
    dispose() { disposed = true; active?.abort(); active = null; },
  };
}

export function mountExtension(win = window) {
  const doc = win.document;
  const existing = doc.querySelector(TAG);
  if (existing) return existing;
  if (!win.customElements || !win.HTMLDialogElement || !doc.body) return null;
  if (!win.customElements.get(TAG)) {
    class LocalModelsExtension extends win.HTMLElement {
      connectedCallback() {
        if (this.shadowRoot) return;
        const shadow = this.attachShadow({ mode: 'open' });
        const styles = doc.createElement('link'); styles.rel = 'stylesheet'; styles.href = '/__strata_extension/ui/entry.css';
        const trigger = doc.createElement('button'); trigger.type = 'button'; trigger.className = 'launcher';
        trigger.textContent = 'Current Model'; trigger.setAttribute('aria-haspopup', 'dialog'); trigger.setAttribute('aria-expanded', 'false');
        trigger.setAttribute('title', 'Open Current Model Controls'); trigger.setAttribute('aria-controls', 'current-model-dialog');
        const dialog = doc.createElement('dialog'); dialog.id = 'current-model-dialog'; dialog.setAttribute('aria-label', 'Current Model');
        const header = doc.createElement('header'), title = doc.createElement('strong'); title.textContent = 'Local Model Extension';
        const close = doc.createElement('button'); close.type = 'button'; close.className = 'close'; close.textContent = 'Close';
        close.setAttribute('aria-label', 'Close Current Model'); close.autofocus = true;
        const frame = doc.createElement('iframe'); frame.title = 'Current Model Controls'; frame.referrerPolicy = 'no-referrer';
        const feedback = doc.createElement('aside'); feedback.className = 'feedback'; feedback.hidden = true;
        const message = doc.createElement('p'); message.setAttribute('role', 'status'); message.setAttribute('aria-live', 'polite');
        const retry = doc.createElement('button'); retry.type = 'button'; retry.textContent = 'Open Current Model';
        const dismiss = doc.createElement('button'); dismiss.type = 'button'; dismiss.textContent = 'Dismiss';
        feedback.append(message, retry, dismiss);
        const showState = state => {
          message.textContent = state.message; message.setAttribute('role', state.error ? 'alert' : 'status');
          feedback.hidden = false; retry.hidden = !state.error; dismiss.disabled = state.busy;
        };
        this.selection = createSelectionCoordinator({ win, onState: showState });
        const useInChat = (connection, source) => this.selection.useInChat(connection, source);
        const showError = error => showState({ message: error.message || 'The App Controls Could Not Be Confirmed. Open Current Model And Try Again.', error: true, busy: false });
        header.append(title, close); dialog.append(header, frame); shadow.append(styles, feedback, trigger, dialog);
        this.controller = createDialogController({ win, dialog, frame, trigger, theme: () => this.dataset.theme,
          onUseInChat: useInChat, onError: showError });
        trigger.addEventListener('click', () => this.controller.open()); close.addEventListener('click', () => this.controller.close());
        retry.addEventListener('click', () => this.controller.open()); dismiss.addEventListener('click', () => { feedback.hidden = true; });
        this.connectedModels = observeConnectedModels({ win, onUseInChat: useInChat, onError: showError, theme: detectTheme(win) });
        const updateTheme = () => { this.dataset.theme = detectTheme(win); this.connectedModels.setTheme(this.dataset.theme); this.controller.sync(); };
        this.observer = new win.MutationObserver(updateTheme);
        this.observer.observe(doc.documentElement, { attributes: true, attributeFilter: ['data-theme', 'style', 'class'] });
        this.media = win.matchMedia('(prefers-color-scheme: dark)'); this.themeChanged = updateTheme;
        this.media.addEventListener('change', updateTheme); updateTheme();
      }
      disconnectedCallback() {
        this.selection?.dispose(); this.connectedModels?.dispose(); this.controller?.dispose();
        this.observer?.disconnect(); this.media?.removeEventListener('change', this.themeChanged);
      }
    }
    win.customElements.define(TAG, LocalModelsExtension);
  }
  const host = doc.createElement(TAG); doc.body.append(host); return host;
}

if (typeof window !== 'undefined' && window.document) {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => mountExtension(), { once: true });
  else mountExtension();
}
