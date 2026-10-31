## Problem and result

Constrained decoding (Phase 1) refused every speculative request and - the incident that triggered this rework - degraded long Chinese constrained answers into filler characters. This PR makes grammar requests ride the MTP drafter / `--batch-mtp` with per-row masks, and fixes two vocabulary-layer defects the acceptance suite was blind to:

- A constrained request no longer refuses `--mtp` / `--batch-mtp`: every row of a drafted verify window samples from the mask of the state its draft prefix leaves (a scratch clone walks the drafts before the window; a draft the grammar refuses truncates the chain, and the tail rows can never be emitted because the row before them was masked against the state the draft jumps from). Related: #3862, and PR #3922 (closed at the author's request; this reworked branch is that continuation).
- Long technical Chinese at temperature 1.0 produced `符号`→`符好`, `好好好` runs under any grammar. Root cause: the grammar vocab loader decoded only part of the GPT2 `bytes_to_unicode` map, leaving 88,990 of 248,320 pieces invalid UTF-8, so every grammar mask banned ~36% of the real vocabulary (most CJK second bytes). With the map complete, a second defect surfaced: the byte-0x00 token's piece is a real NUL, which the accept walk treats as the code_points terminator - it "matched" every state as an empty piece and its untrained logits won greedy picks, leaking a control byte into constrained JSON. Both are fixed and pinned by unit tests.

Result on the rig (RTX 4090, Swift IQ3_S, engine args `--spec 2 --mtp <rt>`): JSON schema suite 50/50 with 0 canary 502s, tool calls 30/30 with 0 rescued, constrained 137.2 vs plain 130.5 tok/s median - the drafter more than covers the mask cost. Plain requests (no grammar named) are byte-identical to before: the sampler mask is a null pointer.

Related discussion: #3862 (scope). PR #3922 was the Phase 1 cut, closed to rework speculative-decoding coexistence into this branch.

## Approach

- The mask ring holds three bands: the solo window's rows `[0, kVerifyMaxT)`, the batch slots' base rows (sized with `--batch`), and `--batch-mtp`'s chain rows indexed by window row (never by slot, so any `--batch` stays in bounds). `--batch-mtp` is not capped to `kVerifyMaxT` slots (the windows rotate), which the first cut got wrong - caught in code review before it could ship.
- The drafter itself draws unmasked: a proposal the grammar refuses dies at the verify (its row's mask is built past the draft's prefix), so masking the drafter would only hide proposals. `STRATA_GRAMMAR_DEBUG=1` traces the chain row by row.
- The grammar vocab loader now decodes the complete GPT2 byte map (the same table HF's tokenizers and llama.cpp's converter apply), and `grammar_build_mask` bans pieces with an embedded NUL exactly as llama.cpp's `apply_impl` bans `piece[0] == 0`.
- A layer split's pipelined windows decode a constrained request serially (the chain rides the serial windows, as penalties do). DFlash2 drafts only in the CLI run in this build, so serving never meets it.

## Validation

Passed (RTX 4090 48GiB, Windows, Swift-Qwen3.8-Flash-Next IQ3_S, engine 0.1.50, commit 1de0d048):
- `python -m pytest`-style unit level: `grammar_test`, `grammar_registry_test`, `grammar_decode_test` (new chain closed-loop: legal and refused-draft windows; new byte-map and NUL-piece pins) - all pass.
- ctest full suite: 160 passed, 1 failed (`ple_parity` - environmental, the rig has no Q2_0 fixture; fails identically on the pre-change base), 7 skipped.
- `python -m unittest discover -s serve`: 1058 tests, 0 failures (after fixing the winjob test's CJK-locale `tasklist` decode on this machine).
- Acceptance `tools/grammar_acceptance.py --stamp nul-fix` with `--mtp`: JSON 50/50, canary 0, tools 30/30, plain 130.54 vs constrained 137.17 tok/s median. Raw logs and item-level JSONL in `bench/results/nul-fix-grammar-4090/`.
- Raw-pipe check: `root ::= "yes" | "no"` yields `n`, `o`, `<|im_end|>` and stops; before the NUL fix it spammed the NUL byte.
- An earlier gate on the same rig without `--mtp` (pre-merge base): JSON 50/50, tools 30/30, constrained cost -2.6% (`bench/results/2026-10-24-1332-grammar-4090`).

Failed: none observed on this rig after the fixes. (Two intermediate gate runs during debugging were contaminated by my own process cleanup and an orphan engine; their numbers are superseded and noted in `bench/results/vocab-fix3-grammar-4090/notes.json`.)

Not run:
- HIP and SYCL builds of this branch (no ROCm / oneAPI on this rig). The SYCL tree's sampler already reads per-row masks, but its program side has no grammar wiring (the phase-1 recorded boundary); HIP is untried. These backends need builds and hardware runs before this leaves draft.
- Mac (Metal) and CPU engines: no grammar capability announced, old paths - untested here.
- Multi-GPU: a layer split serializes constrained requests (correctness kept, speed cost); not measured on a second GPU.

## Limits

- The named `spec_min_p` + grammar combination stays refused (the startup `--spec-min-p` is allowed); the floor's drafts vs the chain ordering is unmeasured.
- Constrained requests on a layer split decode serially - no pipelined speedup there.
- The GBNF surface is the vendored llama.cpp converter's, unchanged.
- Byte-fallback tokens (single high bytes, ~96 tokens) stay banned by the grammar, as in llama.cpp.
- The batch_mtp + grammar concurrency path has a 2-slot smoke and the unit tests; a wider concurrent soak is future work.
