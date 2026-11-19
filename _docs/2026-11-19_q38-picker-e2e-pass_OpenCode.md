# Installed 0.6.4 GUI: Qwen3.8 picker E2E passed (isolated profile)

Evidence: `H:/Strata/evidence/q38-063-signed-recovery-B6VcGa/result.json`

Ran `q38-063-recovery-smoke.mjs` with the signed, hash-pinned preview
component in a fresh isolated profile using the installed 0.6.4 app
(host/shell 0.6.4). The signed archive and imported manifest hashes
matched; the runner was `eeb2667440d5028ed2bcbc6f7470bd8385ad57ec75661162f94de4919c9b17a7`.

Confirmed:
- The chat model picker trigger shows
  `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP` (`pickerShowsQ38: true`).
  The picker menu lists the loaded Qwen3.8 model.
- Two Japanese turns rendered (19 and 9 characters), both with
  16 short-F16 prefill calls and zero fallbacks.
- Stop after rendered output: 14 tokens generated then `cancel=True`.
- Explicit unload returned 200, native PID 908 exited, shell quit
  normally. No renderer page errors.

This proves the ChatModelPicker change is correct and that the loaded
native Qwen3.8 model is reflected in the picker. It does **not** prove
the normal installed profile, where Control's model-list scan takes
78-81 s and the old 60 s host timeout returns `models=[]`. That is a
separate defect with a separate fix (180 s timeout, 23 tests + typecheck
pass, r2 installer build pending). Publish that fix only after
overwriting the installed app and re-running this check in the normal
profile. Performance parity with Prism remains unproven.
