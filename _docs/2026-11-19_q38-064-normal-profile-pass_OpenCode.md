# Normal-profile picker fix verified in installed 0.6.4

Evidence: `H:/Strata/evidence/q38-picker-diag.json` (rewritten 2026-11-19
after installing the r2 NSIS with the 180 s Control timeout).

Installed host `control-runtime.mjs` SHA-256
`beca8aee03ba4ef9ee0c4126d5b3c70b5c2b87180a087ada6450ae336182e52b`,
with `installedModels()` using `timeoutMs: 180_000`. Before this fix the
60 s timeout returned `models=[]` at 60 s; Control's real `models list`
takes 78-81 s on this host.

New normal-profile readback with the installed 0.6.4 app:
- `/api/native` completed in **64.9 s** with `available:true`,
  `checking:false`, `models:4`, `error:null`.
- The ready entry is `q38-dense-20acacc6c8778c27` /
  `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP`.
- The chat picker menu now shows
  `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP` and `モデルを読み込む`
  ("Load model") instead of "no local models downloaded".

This closes the normal-profile picker defect. Generation, Stop after
14 tokens, explicit unload and normal exit with the same signed native
payload were separately verified in an isolated profile
(`q38-063-signed-recovery-B6VcGa/result.json`). Together these cover
the installed 0.6.4 artifact's model-selection and lifecycle E2E.

Performance parity with Prism remains unproven. FFN sub-phase trace
(`q38-ffn-split-7557fd1d`) shows gate/up 7.45 ms/step (~64% of FFN),
down 4.03 ms, activation 0.22-0.26 ms; candidates remain rejected.
