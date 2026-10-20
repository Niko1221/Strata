# Pair an iPhone from Strata on Mac

Choose **Pair iPhone** in the sidebar, or **Settings → iPhone**, then **Enable iPhone pairing**. Scan the QR code from the Strata iPhone app and confirm the connection. Both devices must be on the same trusted network.

Pairing stays off until enabled. Once enabled, Strata restores the connection service on subsequent launches. **Disable pairing** closes its listeners without forgetting paired phones. **Remove device** revokes that phone's access. New invitation codes expire after five minutes and work once. The panel also provides copyable connection details for manual setup.

The Mac's existing inference host remains on loopback, and the phone receives only the companion's narrow authenticated HTTPS controls. Pairing data is private to the local user and survives app restarts. Paired devices can read shared conversation history, continue work with each chat's existing permissions, and answer pending approvals; remove a device to revoke that access.

On iPhone, open **Chat → History** to reopen a Mac conversation and continue its work. The same messages and browser/tool activity appear on both screens. Existing conversations keep their model, folder and permission mode. Pending tool requests can be allowed or denied on the phone. Closing the phone app leaves Mac work running; **Stop** ends the selected turn explicitly.

The updated Mac app imports existing desktop chat history into a private conversation store. If you enable pairing for the first time while a Mac reply is already running, that conversation is shared after the current reply finishes; keep its Mac window open until then. Earlier phone-only chats remain available separately. Existing paired phones stay paired through this update. The iPhone includes the Mac's Thinking text in an expandable section and keeps progress text and activity summaries in their original order. Browser action cards show addresses, progress and text results; they do not stream the browser's screen.

The iPhone's **Models → Download models** can prepare models on the paired Mac or its saved Strata connections. Review the named destination, storage and preparation warnings before starting. Downloads continue on that computer when the phone closes. See [models on another computer](../host/REMOTE-MODELS.md) for remote downloads and GGUF imports from the Mac app.

## Build the Mac app

Follow the [desktop packaging instructions](../../packaging/README.md) to prepare Node, Control, Harness and the engine resources. From `apps/desktop`, the normal build commands stage the pairing service and compile the local Core Image QR helper before building the app:

```sh
npm run build:app
```

`npm run dmg` also stages these resources. When invoking Cargo directly, run `node scripts/stage-companion.mjs` first. macOS command line tools are needed to compile the QR helper; Developer ID signing and notarization remain separate distribution steps.

The running app must be relaunched to load a newly installed host. Finish active Strata tasks before quitting; its existing shutdown path ends app-owned agent turns and models. The fixed desktop origin is preserved to retain desktop conversation storage.
