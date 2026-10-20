# Strata iPhone companion bridge

The bridge connects a paired iPhone app to Strata on your Mac. The desktop app runs it inside its existing host after you enable **Pair iPhone**. Enabling or disabling pairing does not restart Strata, change its address, download a model, or stop its engine.

## Connect your iPhone

1. Open Strata on the Mac. Put your Mac and iPhone on the same trusted Wi-Fi network.
2. In the Mac Strata app, choose **Pair iPhone** in the sidebar or **Settings → iPhone**, then enable pairing. The panel shows a QR code and a six-digit code.
3. Open Strata on iPhone and scan the QR code in **Pair with Mac**, or paste the full invitation. The invitation includes the Mac's certificate fingerprint, address, and one-use code. A six-digit code by itself does not establish certificate trust.
4. Open **Chat → History** to reopen a Mac conversation. Its history and current browser or tool activity appear on your phone. Send a message to continue in the same Mac session. Choose a model in **Models** when starting a new conversation.

An invitation lasts five minutes and allows five guesses in total. Choose **New invitation** on the Mac after it expires. The Mac pairing window also lists paired devices. **Remove** revokes a device immediately. The phone's disconnect action revokes its own session.

The Mac window refreshes automatically when an invitation is used, expires, or a paired device is removed. For manual setup, expand **Enter connection details manually** to copy the HTTPS address, certificate fingerprint, and pairing code individually.

The QR code is generated locally with Apple's Core Image framework. The native Mac package includes a compiled helper. Running the source directly can use the Swift command line tools; the manual invitation remains available if they are missing. No online QR service receives the invitation.

## Run from source

Requires Node 24.12.0 or newer and the macOS system OpenSSL command. No npm packages are installed.

```sh
cd apps/companion-bridge
npm start
```

The process emits one JSON `ready` line containing the local pairing-page URL, listener ports, and public certificate fingerprint. Open `adminURL` on this Mac. It includes a random page capability; do not share it. Pairing codes and session bearer tokens are never logged.

The authenticated HTTPS listener uses port **8787** and the Mac-only pairing window uses **127.0.0.1:8788**. Quit with Control-C. Command-line options:

```sh
node index.mjs --data-dir "/private/path/Strata Companion" \
  --port 8787 --admin-port 8788 --advertise-host 192.168.1.20
```

Normally the bridge discovers the running Strata host by its local process and listening port. If needed, supply an explicit loopback address with `--upstream http://127.0.0.1:47831`. Remote upstream addresses, redirects, paths, and embedded credentials are refused. The existing Strata host remains bound to loopback.

The default private data folder is `~/Library/Application Support/Strata Companion`. Its directory is mode 0700. Certificate, private key, session hashes, and phone conversations are mode 0600. Keep this folder to preserve your Mac's identity and pairing between launches. Changing the certificate requires pairing again. A token is 32 random bytes and is stored only as a hash on the Mac; the phone keeps its token in Keychain.

## What you can control

- View Mac connection, memory, model and active-request status.
- Choose among the API models already configured on the Mac. Provider credentials and endpoint addresses stay on the Mac. An API choice belongs to that paired phone and does not switch the desktop's selected chat model.
- Load an already downloaded local model or unload the phone's selected instance. Choosing an already running local model selects that instance for the phone without restarting it or changing the desktop's default. The existing host applies its own memory and active-request safeguards. Downloads, install operations, settings edits, files, shell commands, and arbitrary proxy routes are unavailable from the bridge.
- Read and continue the Mac's conversations. Messages, browser actions and other tool cards update while the Chat tab is open. Existing chats keep their model, working folder and permission mode. New chats use the phone's selected model and start in Manual mode.
- Answer a pending action with **Allow** or **Deny**, or stop the selected conversation. Closing the phone app leaves Mac agent work running. Stop is an explicit action.

The Mac imports its existing web-view history into a private host conversation store. If you enable pairing for the first time while a Mac reply is already running, that conversation is shared after the current reply finishes; keep its Mac window open until then. Both screens use the same conversation IDs. Versioned updates prevent a stale desktop cache from replacing newer phone messages. Removing a conversation on the Mac also removes it from the shared catalog.

Paired devices can access the shared Mac conversation catalog and approve its pending actions. Earlier phone-only conversations remain in **Earlier iPhone chats** and stay private to their original paired device; their existing text-only route is retained for compatibility. Those earlier chats do not run agent tools.

Browser activity is shown as action cards with the address, state and available text result. This is an activity view, not a live browser video feed. Raw tool arguments, runtime protocol and saved provider configuration are excluded from the phone API. Conversation text and human-readable tool results are shared with the paired phone.

The paired phone also receives the same model-supplied Thinking text that the Mac displays, with its recorded duration when available. Ordered text and activity summaries keep progress such as “Ran 3 commands” between the corresponding parts of the answer. Group payloads expose only display text, completion state, change counts and any explicit references to that message's sanitized tool cards; raw file metadata and protocol fields stay private. Older incomplete ordered snapshots use the complete answer text instead. Conversation-list summaries omit Thinking text and ordered blocks.

When the Mac has no loaded local model, the initial phone selection is the first API model configured in Strata. Selecting another API model is private to the phone. Unloading an API selection clears that phone's selection; it does not stop a remote service. Removing a provider on the Mac makes an old phone selection unavailable until another model is selected.

## Phone API

All routes use HTTPS. The iPhone must compare the leaf certificate's SHA-256 DER fingerprint to the invitation before sending the pairing code or credentials. All routes except pairing require `Authorization: Bearer <token>`. Browser origins are rejected; no CORS access is provided.

| Method | Path | Request / result |
| --- | --- | --- |
| POST | `/v1/pair` | `{code,deviceName}` → `{token,macName}` |
| DELETE | `/v1/session` | Revokes this phone's session |
| GET | `/v1/status` | `{macName,hostConnected,modelName,engineState,tokensPerSecond,memoryUsedGB,memoryTotalGB,activeRequests}` |
| GET | `/v1/models` | `{models:[{id,name,size,state,detail,loaded,selected}],loadedModelId}` |
| POST | `/v1/models/load` | `{id}`; alias `/v1/models/:id/load` with `{}` |
| POST | `/v1/models/unload` | `{}` |
| GET | `/v1/conversations` | Shared Mac conversation summaries and `activeChatId` |
| POST | `/v1/conversations` | `{}` → `{conversation,pending}`; creates a shared conversation |
| GET | `/v1/conversations/:id` | `{conversation,pending}` with messages, tool cards and active state |
| POST | `/v1/conversations/:id/messages` | `{message}` → 202 with current conversation; work continues on Mac |
| POST | `/v1/conversations/:id/stop` | `{}` → current conversation |
| POST | `/v1/conversations/:id/approval` | `{toolId,decision:"approve"|"deny"}` → current conversation |
| GET | `/v1/chats` | `{chats:[{id,title,updatedAt}]}` |
| GET | `/v1/chats/:id` | `{id,title,messages:[{id,role,content}]}` |
| POST | `/v1/chat` | `{message,chatId?}` (`sessionId` accepted as alias) → SSE |
| POST | `/v1/chat/:id/cancel` | Stops this device's active conversation |
| POST | `/v1/requests/:id/cancel` | Stops a request ID previously streamed to this device |

Shared conversations are polled while the phone is active; sending a message returns promptly and never ties agent execution to the phone connection. Shared tool cards retain the Mac's readable verbs, targets, states and results. The API does not accept model, folder, permissions or history overrides when continuing a conversation.

The earlier phone-only chat route uses SSE events named `frame`, with a JSON `data:` line. The first frame has `type: "session"`, `chatId`, and `sessionId`. Text deltas use `text`. An interrupted legacy stream without a host `done` frame emits an error. A successful model `load` can return 202 while the Mac completes loading.

`selected` identifies this phone's model choice. Several native models can have `state: "loaded"`; chat and unload resolve the selected model's ready instance explicitly. The initial native selection follows the Mac's default instance, rather than catalog order. Unloading clears the phone's selection, so another loaded instance is not selected silently.

## Verification

```sh
npm test
```

Tests use a temporary TLS identity and mock host. They cover certificate fingerprint/stability, default TLS rejection without a pin, pairing expiry/guess limits/replay, private token hashes, auth/revocation, narrow routes and methods, admin CSRF/DNS rebinding/escaping, sanitized provider selection, real NDJSON framing into SSE, per-device private history, stream truncation, and concurrent conversation rejection. They do not claim physical iPhone acceptance or real model inference; record those separately when performed.
