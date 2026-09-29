# Shared snapshot core: validation record

Date: 2026-09-28. Engine source: `3657b8f`, based on upstream `b38c183`
(0.1.18). This is development-branch evidence for the RAM tier/shared core,
not acceptance of a combined RAM/NVMe implementation or a deployment report.

## Configuration and provenance

- Linux, AMD EPYC 7532, approximately 247 GiB physical RAM, NVIDIA RTX 4090.
- GCC 15.2, CUDA 13.4, SM 89, portable/AVX2 CPU kernels.
- Qwen3.8-Flash-Next-GSQ-RCO IQ3_S; INT8 KV, 131,072-token maximum context,
  32,768 resident KV cells, automatic prefill sizing, 8,817 expert slots.
- Paired engines run sequentially with fixed expert residency. Baseline has
  conversation caching disabled; candidate normally has an 8,192 MiB budget,
  four parked slots and a 2,560 MiB physical-memory floor. Pressure/oversize/
  exchange/admission gates explicitly change the relevant limit.
- Tested executable SHA-256:
  `f31d1fe2eecf7d27f440ea1ddfdb39133a16f471c70c1869491582614a16c2f9`.

Do not infer a speedup over the earlier 0.1.15 results: maximum context,
expert residency and prefill configuration differ. These tests compare cache
behavior within this revision, not engine versions.

## Component and offline checks

| Gate | Result |
|---|---|
| CPU RAM ownership/prefix/budget policy | 35 checks passed |
| Available-memory parsing/admission boundaries | 23 checks passed |
| Real-GPU KV and complete-session fixtures | 1,071 checks passed |
| Whole-image rejection before mutation, devices hidden | 780 checks passed |
| Wrapped transfer/synchronization failure control flow | 1,020 checks passed (includes the 780 validation checks) |
| CPU policy/memory and host validation/transfer tests under ASan/UBSan | Passed |
| Python server tests plus cache-harness tests | 45 tests passed |
| Cache-harness tests under Python `-O` | 22 tests passed |
| CLI parsing, help, disabled-feature warnings | 20 checks passed |

GPU fixtures cover FP16/INT8/Q4 KV, resident/streamed layouts, distinct indexer
spare keys, checkpoint rewind, and 256/512-expert geometry metadata. Host fixtures
also cover zero-QSA and PLE-present/absent layouts. Small geometry fixtures do not
constitute a full Coder/pruned-model run. Link-wrapped host failures exercise the
real restore control flow, not actual CUDA hardware-context recovery.

## Full-model private-engine gates

All nine gates below passed with this executable:

1. Single-token A/B/A and checkpoint output/main-model state parity.
2. Long speculative A/B/A and checkpoint recovery beyond resident KV.
3. Byte-pressure eviction and correct cold fallback.
4. Oversized-entry rejection without disrupting inference.
5. Incoming/outgoing exchange-budget enforcement.
6. Image-content/grid identity isolation using synthetic embeddings.
7. Add-mode control-vector identity isolation.
8. Project-mode control-vector identity isolation.
9. Physical-memory admission denied by a 1 TiB floor: three admission skips,
   no budget skips or parked entries, and output/state parity for A/B/C/A.

State comparisons include GDN, PLE, indexer tail, `idx_dead`, pooled rows
including the spare row, main KV, previous PLE tokens and consumed length.
Unused draft cells are not part of the main-model fingerprint; draft snapshot
copies are covered by the component fixtures. Earlier prototype fingerprints
did not include `idx_dead` and must not be treated as equivalent coverage.

In the long speculative gate (`spec=4`, `mtp_max=4`, lookup disabled):

| Candidate request | Prompt tokens | Reused | Prompt-path time |
|---|---:|---:|---:|
| Initial A | 51,133 | 0 | 23.3721 s |
| Initial B | 51,135 | 0 | 22.3445 s |
| A continuation after B | 51,155 | 51,133 | 1.2371 s |
| A checkpoint continuation | 51,155 | 51,148 | 0.5659 s |

Return-path times include parking/restoration and the remaining new prompt
tokens, but not decoding. Both A continuations matched uninterrupted baseline
outputs and the complete main-model fingerprints listed above. These synthetic
requests are not a controlled real-agent task benchmark.

## Known-answer soak

`tools/conversation_cache_soak.py --prompt-tokens 2048,40000,120000 --cycles 30`
passed against the same executable. Actual prompt lengths were 2,026, 39,985 and
119,987 tokens (the largest is 91.5% of maximum context). Single-token decode
was enforced with `spec=2`, `mtp_max=1` and lookup disabled.

Each conversation contains a distinct secret word inside its history. After an
unrelated worker request, all 30 returns produced the expected word and exactly
matched uninterrupted baseline output tokens and main-model state fingerprints.
Each size was restored ten times; the final three retained-payload samples per
size were identical:

| Active conversation length | Parked payload after its answer |
|---|---:|
| 2,026 tokens | 4,099,851,588 bytes |
| 39,985 tokens | 3,283,651,392 bytes |
| 119,987 tokens | 1,824,199,544 bytes |

Occupancy differs because the active conversation is not parked; these figures
are not the snapshot sizes of the active conversations. Maximum logged parked
payload was 4,249,016,776 bytes, below the 8 GiB budget. The policy also accounts
for an incoming image held during exchange; the logged parked count alone does
not measure that transient total. Request-end RSS was sampled as a diagnostic,
not as a peak-memory bound or a proof against every allocator leak. Thirty cycles
establish this finite regression gate, not indefinite stability.

## Private HTTP smoke

All 12 requests in `tools/conversation_cache_http_smoke.py` passed on a separate
localhost-only endpoint using the same executable and normal adaptive settings
(reported `spec=6`, `mtp_max=4`, lookup 3, 8,816 expert slots). This was not the
fixed-residency/single-token configuration used for deterministic state parity.

- A/B/A preserved the answer and reused 1,987 of 1,994 prompt tokens.
- Exceeding the four-slot limit evicted A; its next request correctly reused zero.
- A streaming disconnect cancelled after five output tokens; an unrelated
  request then succeeded, and returning to the cancelled conversation reused
  1,085 of 1,092 prompt tokens.
- Frontend and engine process identities remained unchanged throughout the
  smoke. The private test instance was then stopped; no production endpoint or
  configuration was replaced.

## Remaining acceptance work

- Real vision encoder/image-recognition smoke (current image gate uses synthetic
  embeddings); full-model 256-expert/Coder coverage.
- Windows runtime/memory-pressure review, including constrained-memory hardware;
  Linux container/job-limit-aware admission is not implemented.
- Controlled real Pi/Hermes task with fixed initial files, tool limits,
  independent success checks, wall-time breakdown and peak memory.
- Agreed ownership/interface for the unified PR, NVMe adapter, bounded staging,
  persistent compatibility identity, corruption/restart/durability tests.

No private conversations, raw request captures, local configuration files or
machine-specific paths are needed to reproduce these synthetic fixtures.
See [the design and test commands](shared-conversation-snapshots.md).
