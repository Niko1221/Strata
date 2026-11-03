/**
 * Example UI extension for Strata Dev (docs/DEV_BUILD.md, "UI-Extensions"): an info button in the chat toolbar.
 * Loaded only when STRATA_DEV=1. With `npm run dev` (dev-build.ps1 -HostOnly) an edit here reloads the page at once.
 *
 * An extension exports one setup function. `strata` is the extension API (apps/host/src/dev/dev.tsx, ExtensionApi):
 *   strata.slot(slot, { id, label?, order?, render })   slots: sidebar, chat.toolbar, chat.message.actions, settings.tabs,
 *                                                       monitor.widgets, preview.panel
 *   strata.hide(id) / rename(id, label) / move(id, order) / replace(id, Component)   existing elements by id (Dev panel -> Inspector)
 *   strata.toast(title, body?), strata.api.get/post(path), strata.log(...), strata.ui.{Btn, Icon, IconBtn}, strata.React
 */
export default function setup(strata) {
  const { IconBtn } = strata.ui;
  strata.slot('chat.toolbar', {
    id: 'info',
    render: ({ sessionId }) => (
      <IconBtn
        name="info"
        title="Strata Dev: info about this chat"
        onClick={() => strata.toast('Strata Dev', `Chat ${sessionId ?? '(new)'} · config ${strata.dev.configFile}`)}
      />
    ),
  });
}
