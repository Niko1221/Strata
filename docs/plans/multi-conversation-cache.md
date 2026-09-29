# Multi-conversation caching

## Objective and evidence

Keep Pi controller and worker conversations warm when their requests interleave,
without requiring a Pi-specific session ID or changing the generated prompt.
The existing server caches one conversation. Observed controller requests reread
46,856, 46,974, and 47,644 tokens after worker requests, spending 143 seconds in
prompt processing. A subsequent packet capture verified 21 continuing turns with
unchanged history and reasoning: do not modify Pi's reasoning serialization.

This feature does not shorten tool results or fix the separate, unproven cause of
one historical within-conversation 12,038-token replay.

## Design

- Opt-in engine flags: `--conversation-cache-mib N` (default 0/off) and
  `--conversation-cache-slots N` (default 4 parked conversations).
- Keep existing single-conversation behavior and fast-path cost when disabled.
- Search the active state and parked states for the longest exact token prefix.
  Images and control-vector mode must also match. Prefer active state on ties.
- Before overwriting an eligible active state, snapshot its running state,
  checkpoints, main-layer K/V, completed indexer rows, and draft-layer K/V.
- Park in ordinary host RAM, not additional VRAM. Preserve graph-bound addresses
  by copying back into existing allocations. Streamed K/V snapshots use the
  authoritative host pools; restore invalidates main-layer residency maps and
  refills the draft ring before reuse.
- Save only used pages, not the full maximum context allocation. Full pages
  preserve the page/head/cell layout, including partially filled final pages.
- Evict least-recently-active parked entries to satisfy slot and byte limits.
  Count an incoming restore image against the byte budget during an exchange.
  An oversized snapshot or allocation failure skips parking, not inference.
- Cache misses retain normal prompt processing; image or steering changes cannot
  reuse incompatible state. No persistence to disk and no raw conversation logs.
- Keep checkpoint fallback available inside each parked conversation.

The budget covers snapshot payload allocations, including retained checkpoint
buffers. It is additional to the active engine's normal state/checkpoint memory;
allocator bookkeeping and small container overhead are not exact RSS accounting.

## Correctness gates

1. CPU policy tests: A/B/A reuse, checkpoint fallback, image identity, steering
   isolation, longest prefix and ties, byte/slot eviction, oversized entries,
   disabled cache, short/equal prompts, and in-flight restore accounting.
2. Snapshot tests: byte-exact save/restore across FP16, INT8 and Q4 K/V, resident
   and streamed layouts, draft rings, partial pages, and indexer boundaries.
3. Full engine builds; disabled option retains existing behavior.
4. Controlled A/B/A serving parity versus an uninterrupted A baseline, including
   returning through a checkpoint, with fixed expert residency and deterministic
   sampling. Verify restore tokens and compare running/KV state hashes.
5. Measure long-context switching time and memory bounds; test budget eviction
   and cancellation. Do not claim end-to-end speedups from component tests.

## Operational constraints

Full-model tests load private engines and require an available GPU. Run them in
a maintenance window or on separate hardware; do not load a second model beside
a production engine unless there is sufficient GPU and host memory.

## Status

This document's historical results describe the original prototype and its
0.1.15 integration. The separate 0.1.18 development branch now has a
[shared-core proposal and acceptance checklist](shared-conversation-snapshots.md).
Its audit found that the original snapshots/fingerprints omitted `idx_dead` and
the moving indexer spare row; the new core preserves these and the new harnesses
require the expanded fingerprint. Earlier parity results remain valid only for
the fields and fixtures actually measured, not those previously omitted fields.

Feature implemented; all five correctness gates above have passing evidence for
the original feature revision `9d154ed`, before the upstream integration below.
Verified on 2026-09-28:

- Full CUDA 13.4 / SM89 portable-AVX2 engine build and help smoke test pass.
- CPU policy: 35 checks pass, also under AddressSanitizer/UBSan.
- GPU snapshots: 891 checks pass across FP16/INT8/Q4, resident/streamed/ring
  storage, zero/full/partial pages, indexer boundaries, malformed payloads, and
  incompatible geometry, and invalid/overflowing extents. GPU fixtures are small
  and do not load a model.
- All 17 existing Python server tests pass using mock engines.
- 12 CLI parsing checks pass; new C++ modules pass `-Wall -Wextra -Werror`.
- Seventeen harness unit tests pass normally and under Python `-O`, including
  rejection of incomplete evidence, native-IQ window configuration, byte-pressure
  evidence validation, offline HTTP orchestration, and no-traffic dry-run checks.

Full-model IQ3_S gates passed in an isolated test window, using a 262144-token
context, INT8 K/V with 32768
resident cells, and the same 8000 expert slots in each paired run:

- Single-token decoding: A/B/A and parked-checkpoint recovery reproduce both
  output tokens and byte-exact main-model state fingerprints.
- Speculative decoding (`--spec 4`, suffix drafting off, fixed expert residency):
  the same checks pass with 49909-token A and 49911-token B. Main-model state
  fingerprints also match, although this gate only requires output parity.
- Initial A prompt read: 48.809 seconds. Returning to A after B: 49909 tokens
  reused, 22 new tokens read, 1.961 seconds of prompt processing including
  parking/restoring. This is a synthetic switching result, not a general decode
  speedup or a matched uncached A/B/A replay benchmark.
- Returning through A's checkpoint: 49924 tokens reused, 7 read, 1.101 seconds.
  Parking A used 1352930652 bytes; parking the continued A used 1471553864 bytes.
  Restoring a parked state took 182–201 ms in this long-context run.
- HTTP checks on the candidate with normal production speculative/adaptive
  settings: A/B/A returns the same text, reusing 1957 of 1964 prompt tokens.
  Five intervening distinct conversations evict A; requesting it again reads
  from zero. Logs show four parked slots and increasing eviction counts.
- HTTP streaming disconnect after five output tokens: the engine stays alive,
  serves an unrelated B, then restores 1054 of the cancelled A's 1061 tokens.
  The frontend and engine remained running without a restart throughout these checks.
- Byte pressure: a 400 MiB cache holds each ~254 MiB snapshot individually, but
  not two. A/B/C/A produced two byte-driven evictions with only one parked slot
  occupied. Output and main-model state match the cache-disabled sequence.
- Oversized snapshots: with a 1 MiB budget, all three attempted parks are skipped;
  ordinary inference continues with output and state matching the baseline.
- In-flight budget: A/B/A and checkpoint recovery also pass at 400 MiB. Two
  outgoing parks are skipped while the incoming snapshot is held, preserving the
  restore and keeping allocated parked/transfer payload within the budget.
- Image isolation: real `GENI` requests with synthetic embeddings reject reuse
  when embedding values or image-grid geometry change. Returning to the original
  image restores 773 of 780 tokens with exact output/main-model state parity.
- Steering isolation: nonzero synthetic control vectors in both add and project
  modes change actual model state. Switching on/off never reuses incompatible
  state; returning to each mode restores 766 of 773 tokens with exact output and
  main-model state parity. These are cache tests, not steering-quality tests.
- The packaged HTTP smoke test also passes against the restored service:
  A/B/A, four-slot eviction, disconnect cancellation and cached recovery. Engine
  and frontend remained running without a restart throughout that test.

Raw synthetic test evidence is retained locally in the ignored
`logs/conversation-validation-2026-09-28/`: `spec1-supported`, `spec4-long`,
`pressure`, `oversized`, `exchange`, `image`, `steering-add`, and `steering-project`
(each contains engine logs and `results.json`), plus `http-smoke.json` for the
packaged HTTP check.
The original `spec1` attempt failed at startup because native IQ packs require
engine `--spec >= 2`; the harness now uses the supported configuration below.

`tools/conversation_cache_parity.py` prepares the controlled runtime gate. It
defaults to a dry run and requires `--run` to launch private model processes.
Provide `--config`, `--engine`, and a new `--output` directory. Run first with
`--spec 1` for byte-exact main-model state, then `--spec 4` for speculative output
parity. The single-token gate configures engine `--spec 2 --mtp-max-t 1` because
native IQ packs require a speculative-capable engine even with one-token decode
windows. It compares an uninterrupted A/A+ baseline against A/B/A+, then exercises
a parked checkpoint by switching away and repeating A+. It never connects to an
existing HTTP server. Use only in a separately authorized GPU test window.

The image tests exercise the serving engine's embedding-input and M-RoPE paths,
not a vision encoder or image-recognition quality. The live Pi configuration
remains text-only with no control vector loaded.

### Reproducible checks

The parity script now accepts `--scenario pressure` and `--scenario oversized`.
Both run A/B/C/A with one output token per request, compare against the same
cache-disabled sequence, and require output and main-model state parity. For
this IQ3_S model at the default 128 paragraphs, use `--cache-mib 400` for byte
pressure (one snapshot fits, two do not) and `--cache-mib 1` for oversized skips.
The byte-pressure gate rejects slot-limit eviction as insufficient evidence;
the oversized gate requires all three attempted parks to be skipped. Both gates
passed in the full-model test window.

`--scenario exchange --cache-mib 400` instead runs the A/B/A continuation and
checkpoint sequence while requiring two outgoing parks to be skipped because an
incoming snapshot is still held. This specifically tests the temporary byte
budget during restoration, in addition to ordinary byte-driven eviction. This
gate also passed.

`tools/conversation_cache_isolation.py` takes the same `--config`, `--engine`,
`--output`, and explicit `--run` arguments. Run `--scenario image`, `--scenario add`,
and `--scenario project` in separate new output directories. All three gates
passed. They generate small deterministic embedding/control-vector fixtures and
load baseline/candidate engines sequentially. Do not run any model gate beside
the production engine unless sufficient GPU capacity is separately available.

For standalone component tests, configure with
`-DSTRATA_BUILD_CONVERSATION_TESTS=ON`; build `conversation_cache_test` and, with
CUDA enabled, `conversation_snapshot_test`. Run:

```sh
ctest --test-dir build-conversations -R '^conversation_(cache|snapshot)_test$' --output-on-failure
python -m unittest tools.test_conversation_cache_parity tools.test_conversation_cache_isolation tools.test_conversation_cache_http_smoke
```

`tools/conversation_cache_http_smoke.py --output NEW_RESULTS.json` defaults to
no traffic. Adding `--run` targets `http://127.0.0.1:11434` (override with `--url`)
and exercises A/B/A, slot eviction, streaming cancellation, and recovery. Use
only with an exclusive idle test endpoint: it intentionally displaces cached
conversations. An idle check cannot lock out new clients, so pause other clients
first. Both the one-off sequence and the packaged script have passed against the
real cache-enabled HTTP server.

### Upstream integration (2026-09-28)

Merged upstream `main` at `d551edf` (engine 0.1.15). The serving INFO-line
conflict was resolved by retaining both engine-version reporting and the
conversation-cache settings. Upstream prefill/MMQ and AVX2 changes are retained.

The merged tree passes a separate CUDA 13.4 / SM89 portable-AVX2 engine build,
35 CPU cache-policy checks, 891 GPU snapshot checks, 12 CLI parsing/help checks,
and 35 Python tests (17 cache-harness tests and 18 server tests).

The merge itself did not restart or reconfigure the running engine. In a
subsequently authorized maintenance window, all eight full-model gates passed
on merged revision `eb6a392`: single-token reuse/checkpoint state parity, long
speculative reuse/checkpoint parity, byte-pressure eviction, oversized snapshots,
incoming/outgoing exchange budgeting, image identity, and control-vector
isolation in add/project modes. The 12-request HTTP smoke test also passed on
an exclusive localhost endpoint with normal adaptive settings, including slot
eviction, streaming cancellation, and cached recovery without a process restart.

The merged long-context test used 51133-token A and 51135-token B, fixed 8700
expert slots, INT8 KV with 32768 resident cells, spec 4, and suffix drafting off.
Initial A read took 46.558 seconds. Returning after B reused 51133 tokens and
read 22 in 1.363 seconds including parking/restoring; checkpoint recovery reused
51148 tokens and read 7 in 0.580 seconds. Both restored outputs and main-model
state fingerprints match the uninterrupted baseline. This synthetic switch test
is not a matched performance comparison against the original feature binary.

Evidence is retained locally under the ignored
`logs/upstream-merge-validation-20260928-F1emt1/` directory, including paired
engine logs/results and `http-smoke.json`. The earlier measurements above remain
specific to the original feature revision.

## Deployment and rollback

The original Pi trial used an 8192 MiB host-RAM budget and four parked slots, with
the existing text-only model configuration otherwise unchanged. HTTP health
and generation checks pass; image/steering tests ran in separate private engines.

After the upstream-integration gates passed, the merged engine was deployed
with the same 8192 MiB/four-slot cache settings. The normal launch configuration
selects the merged binary; the prior binary and configuration are preserved for
rollback. Final HTTP generation and idle/empty-queue health checks passed.

To try this feature, preserve the original binary and configuration, then add
the two opt-in flags to the engine arguments and restart in a maintenance window.
To disable parking, remove the flags or set `--conversation-cache-mib 0` and
restart. To roll back the binary, gracefully stop the candidate, verify its engine
exited, then start the original binary with the original configuration. Never
start both model processes on the same GPU without sufficient free memory.
