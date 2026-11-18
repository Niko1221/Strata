# Q38 preview handoff: work not merged into main

## What was selected

The distribution target is the improved native engine, not Prism. The
tested opt-in short F16 prefill source is isolated in commit
`c463ef1cc0ddd1c146958a4b333cd01f75342020` under `v0.6.4-preview.1`.
Its four source changes were compared against the build-time patch.
The immutable archive retains component ABI 0.6.3 and requires a
supplemental public key and explicit opt-in. See
[installation, rollback and file pins](../docs/Q38_NATIVE_PREFILL_PREVIEW.md).
`main` gets the README and Pages source documentation, not the preview
native changes.

## What remains outside main

| Work | Status and next action |
|---|---|
| Short F16 prefill (engine, KV, and KV regression tests) | Selected for preview, not merged. Start a future integration from the four-file tagged source, not from the mixed working tree. |
| Short-F16 decode and IQ2_S cached-grid decode | No repeatable gain. The cached-grid tests matched all logits but lost three of four resident pairs. Do not enable or merge as a product optimization. |
| Shared-FP16 and INT8 IQ2_S projection prototypes | Numerically screened after correcting the 16-value weight-scale boundary; slower than cuBLAS. Diagnostic only. |
| Position and GDN graph tracing; phase probes and benchmark runners | Local diagnostic tools only. FFN and projections dominated the GDN bucket; tracing and benchmarks are not runtime defaults. |
| Host timeout/reliability test edits | Left untouched and uncommitted. Whole Host test suite was not qualified; do not include these edits incidentally. |
| Internal execution logs and QA scripts | Retained in the local workspace and evidence directories. Review and sanitize before wider sharing; the public instructions do not depend on them. |

Keep both the isolated release source and the separate local benchmark
worktree; the latter adds a repeat loop and is not the release binary.
Do not overwrite the mixed workspace edits or repurpose the release tag.
The private signing key is not a handoff artifact; the release contains
only the public trust file.

## Acceptance and remaining limits

The signed archive passed byte readback (152 payloads plus manifest).
Installed GUI 0.6.3 passed two Japanese turns, mid-response Stop after
nine tokens, explicit unload and clean native/shell exit with this exact
payload, supplemental trust, and F16 KV. Both completed turns used 16
fast calls with no fallback. The two process-first-request comparisons
do not establish warm parity; decode took 3.24–11.87% longer than Prism.
The separate repeat-enabled warm build likewise lost in both orders:
native prefill 311.604/387.518 ms versus Prism 210.827/207.996 ms;
native decode 27.33/27.10 versus Prism 30.52/30.91 steps/s.
Full-window inference, general model quality, and GPU concurrency
remain unqualified. A new installer with built-in trust or a change of
default would require separate qualification.

## Publication procedure

Stage only the tested archive, public key, pinned source patch, and
checksum list. Verify draft asset sizes and SHA-256 digests before
making the prerelease public. Keep stable Latest and the old tag intact.
Check the exact remote `main` and `gh-pages` refs and published Pages
content; do not claim publication from a local commit alone.

## Publication readback

Published `v0.6.4-preview.1` as a GitHub prerelease. Before publishing,
`gh release view` confirmed all four uploaded asset sizes and SHA-256
digests match their local counterparts. After publication, anonymous
HTTP HEAD returned 200 on all four asset URLs. This was not a second
full anonymous download. GitHub Latest remained `v0.6.3`.

The README translations and Pages source were pushed through `main`
commits `7d74a66c5` and `f1b0a046c`. The live Pages branch was pushed
at `050900239`; the deployed page returned HTTP 200 and contained the
new preview link. Only documentation was committed on `main` for this
delivery. The separate tag target and rejected experiments remain
as described above.
