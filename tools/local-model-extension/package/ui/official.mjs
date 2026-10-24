const incomplete = 'Connection Details Are Incomplete. Refresh Current Model And Try Use In Chat Again.';
const text = node => (node?.textContent ?? '').replace(/\s+/g, ' ').trim();
const field = value => typeof value === 'string' && value.length > 0 && value.length <= 512
  && value === value.trim() && !/[\x00-\x1f\x7f]/.test(value);

/** Only the connect response supplies these identities. No inferred aliases. */
export function connectedSelection(value, expectedModel) {
  if (!field(value?.selection?.connectionId) || !field(value.selection.modelId)
      || !field(value.label) || !field(value.connectionName)
      || !Number.isSafeInteger(value.contextMax) || value.contextMax <= 0) throw new Error(incomplete);
  if (expectedModel !== undefined && value.selection.modelId !== expectedModel)
    throw new Error('The Connected Model Changed. Refresh Current Model Before Selecting It In Chat.');
  return { selection: { connectionId: value.selection.connectionId, modelId: value.selection.modelId },
    label: value.label, connectionName: value.connectionName, contextMax: value.contextMax };
}

export function visible(win, node) {
  if (!node?.isConnected || node.closest('[hidden], [aria-hidden="true"], [inert]')) return false;
  const style = win.getComputedStyle(node);
  return style.display !== 'none' && style.visibility !== 'hidden' && node.getClientRects().length > 0;
}

const selectable = node => !node.disabled && node.getAttribute('aria-disabled') !== 'true';
function one(nodes, message) {
  if (nodes.length > 1) throw new Error(message);
  return nodes[0] ?? null;
}

export function pickerOption(menu, connection) {
  if ([...menu.querySelectorAll('p.model-picker-heading')].filter(node => text(node) === connection.connectionName).length > 1)
    throw new Error('More Than One API Connection Has This Name. Give The Connections Unique Names, Then Try Again.');
  const legacy = [...menu.querySelectorAll('button.model-picker-api')].filter(button => {
    const heading = [...button.parentElement.children].find(child => child.matches('p.model-picker-heading'));
    return text(heading) === connection.connectionName && text(button.querySelector('strong')) === connection.label
      && text(button.querySelector('small')) === connection.selection.modelId;
  });
  const native = [...menu.querySelectorAll('button[role="menuitem"]')].filter(button =>
    text(button) === `${connection.label} · ${connection.connectionName}`);
  const match = one([...legacy, ...native], 'More Than One Chat Model Matches. Give The API Connection And Model Unique Names, Then Try Again.');
  if (match && !selectable(match)) throw new Error('The Chat Model Picker Is Busy. Finish The Current Response, Then Try Use In Chat Again.');
  return match;
}

/** One deadline, mutation-driven waits, and no repeated click attempts. */
export function waitForDom(win, read, { signal, deadline = Date.now() + 8000 } = {}) {
  return new Promise((resolve, reject) => {
    let observer, timer, done = false;
    function finish(error, result) {
      if (done) return;
      done = true; observer?.disconnect(); clearTimeout(timer); signal?.removeEventListener('abort', abort);
      if (error) reject(error); else resolve(result);
    }
    const abort = () => finish(signal.reason ?? new Error('Chat Selection Was Cancelled.'));
    const check = () => {
      if (signal?.aborted) { abort(); return; }
      try { const value = read(); if (value) finish(null, value); } catch (error) { finish(error); }
    };
    if (signal?.aborted) { abort(); return; }
    signal?.addEventListener('abort', abort, { once: true });
    observer = new win.MutationObserver(check);
    observer.observe(win.document.documentElement, { childList: true, subtree: true, characterData: true,
      attributes: true, attributeFilter: ['title', 'class', 'hidden', 'disabled', 'aria-expanded', 'aria-checked'] });
    timer = setTimeout(() => finish(new Error('Chat Selection Could Not Be Confirmed. Open Chat, Refresh Its Model List, Then Try Use In Chat Again.')),
      Math.max(0, deadline - Date.now()));
    check();
  });
}

/** Dismiss only the source's standard Close control, as part of an explicit action. */
export function dismissSource(win, source) {
  if (!source) return;
  if (!visible(win, source)) throw new Error('This Model View Was Closed. Reopen It Before Using The Model In Chat.');
  const dialog = source.closest('[role="dialog"]');
  if (!dialog) return;
  const close = [...dialog.querySelectorAll('.dialog-head__actions button')];
  if (close.length !== 1 || !selectable(close[0]))
    throw new Error('Close This App Dialog, Then Use Current Model To Select The Model In Chat.');
  close[0].click();
  return dialog;
}

export async function selectInOfficialChat(win, value, { signal, timeout = 8000, source } = {}) {
  const connection = connectedSelection(value), deadline = Date.now() + timeout;
  if (signal?.aborted) throw signal.reason;
  const sourceDialog = dismissSource(win, source);
  win.location.hash = '#/chat';
  const doc = win.document;
  const assertChat = () => {
    if (win.location.hash !== '#/chat') throw new Error('Chat Navigation Changed. Try Use In Chat Again When Ready.');
  };
  const trigger = await waitForDom(win, () => {
    assertChat();
    if (sourceDialog?.isConnected && visible(win, sourceDialog)) return null;
    return one([...doc.querySelectorAll('button.chat-model-picker')].filter(node => visible(win, node)),
      'More Than One Chat Model Picker Is Open. Close Extra Chat Views And Try Again.');
  }, { signal, deadline });
  const confirmed = () => {
    assertChat();
    const candidates = [...doc.querySelectorAll('button.chat-model-picker')].filter(node => visible(win, node));
    const current = one(candidates, 'Chat Selection Is Ambiguous. Select The API Connection Explicitly In Chat.');
    return current && current.getAttribute('title') === `${connection.connectionName} · ${connection.selection.modelId}`
      && text(current.querySelector('span')) === connection.label ? current : null;
  };
  if (confirmed()) return connection;
  if (!selectable(trigger)) throw new Error('The Chat Model Picker Is Unavailable. Finish The Current Response And Try Again.');
  const before = new Set(doc.querySelectorAll('[role="menu"]'));
  const opened = trigger.getAttribute('aria-expanded') !== 'true';
  try {
    if (opened) trigger.click();
    const menu = await waitForDom(win, () => {
      assertChat();
      if (!trigger.isConnected) throw new Error('The Chat Model Picker Changed. Try Use In Chat Again.');
      return one([...doc.querySelectorAll('[role="menu"]')].filter(node => visible(win, node) && (!opened || !before.has(node))),
        'More Than One App Menu Is Open. Close Extra Menus And Try Use In Chat Again.');
    }, { signal, deadline });
    const button = await waitForDom(win, () => {
      assertChat();
      if (!menu.isConnected || !trigger.isConnected) throw new Error('The Chat Model Menu Closed Before Selection. Try Again.');
      return pickerOption(menu, connection);
    }, { signal, deadline });
    if (signal?.aborted) throw signal.reason;
    assertChat();
    if (!visible(win, button) || !selectable(button)) throw new Error('The Chat Model Choice Changed. Refresh Its List And Try Again.');
    button.click();
    await waitForDom(win, confirmed, { signal, deadline });
    return connection;
  } finally {
    if (opened && !signal?.aborted && trigger.isConnected && trigger.getAttribute('aria-expanded') === 'true') trigger.click();
  }
}
