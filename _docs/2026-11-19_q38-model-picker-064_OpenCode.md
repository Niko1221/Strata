# 0.6.4 application candidate: native Q38 chat model picker

## Report and scope

The installed Qwen3.8-27B native model was not reliably selectable
from the chat picker. The old renderer had two different menus.
With a native instance already loaded it omitted installed/unloaded
models; otherwise clicking one navigated away to Models without
starting a load. Installed rows were deduplicated by display label
instead of stable model ID. The preview signing key also needed
explicit session trust in 0.6.3.

## Implementation

- `ChatModelPicker.tsx` keeps one menu through native load transitions.
  A registered `q38dense` entry starts the existing native load job.
  It changes the chat choice only after the returned instance is
  ready and confirmed as the native default. For failures, show the
  error without clearing the valid remote route. Do not change another
  chat's remote preference if the user switches chats while loading.
  Other installed model types keep the existing Models navigation.
- `model-picker.ts` deduplicates by stable model ID and does not
  mistake an errored instance for a loaded instance.
- `native.ts` waits on the exact native load job and selects the exact
  ready instance. No engine inference defaults, model geometry, or
  resource floors changed.
- Shipped Control adds public key `9c835e4839a4c6f3` alongside the
  previous trust keys. No private key data or updater trust changes.
- Product/app metadata advances to 0.6.4. Control and engine ABI
  stay 0.6.3. Windows continues to distribute the GPU engine
  separately. The opt-in app-only packaging path pins the already
  published signed preview companion manifest and verifies every
  payload hash with shipped trust and checks the native version.
  Default release verification remains in place.

## Verification and next action

`npm run typecheck` and `node --test tests/model-picker.test.mjs
tests/chat-model-picker.test.mjs tests/native-two-models.test.mjs`
passed (17/17). The first pass hit a stale source-substring count
after merging the two menus; the assertion now checks the single
menu. The two new mounted tests exercise successful deferred
selection and load failure without clearing the chat route. These
use fixture boundaries, not the real installed runtime.

Started the five-job, offline NSIS app build with pinned public
assets and adjacent Control trust data. Build result, installed
WebView2 picker-to-generation check, normal-profile overwrite
verification, and tag/publication are outstanding. Do not publish
on the strength of fixture tests. The regular desktop process was
present at inspection and has not been closed. Preserve prior
models, chats, preferences, and unrelated working-tree edits.
