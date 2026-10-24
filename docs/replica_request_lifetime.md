# Current revision: 0.2.0-beta re-test

The repair now preserves the official selected-replica callback and is based on main `f57c0adcb31c00f3f216550c004dac4e713a9f70`. Current CPU protocol controls and 102 selected serving tests are recorded in [v020-beta](../tests/data/replica_lifetime/v020-beta/README.md). No GPU run of the new release is claimed.

The replayer now defaults to release `3d1fb48`. Historical commands/results below describe their original revisions; use `--baseline 477ac13cbe8e582c20484871ce459d20d82df09d` to select that older control explicitly.

## Historical 2026-10-17 investigation and GPU qualification

# Replica request ownership: two independent failures

Measured 2026-10-17 on two RTX 3090s. This is a Python serving correctness fix,
not a CUDA/CPU kernel optimization or a new default configuration.

The proposed code is scheme A. The other schemes below were offline prototypes,
not additional production implementations. Results and synthetic raw excerpts:
[`tests/data/replica_lifetime`](../tests/data/replica_lifetime).

## Affected path

With two single-GPU replicas and one unbatched decoder per replica,
`ReplicaEngine.batch` reports two serving slots. `Service` therefore allows
concurrent callers without its single-engine FIFO. Each replica's `run_lock`
must cover the complete native request, including STOP/DONE cleanup.

Both KVGROW and resident-KV unbatched replicas use this path. An empty reply
observed with KVGROW does not by itself establish an elastic-KV arithmetic bug.
The layer-split single-engine production profile does not instantiate this
wrapper. Batched native engines already keep their native `last` per thread.

## Root cause 1: child lifetime extends beyond pipe ownership

`ReplicaEngine.generate` forwards the native iterator with `for ...: yield`.
The frontend closes the wrapper on EOS, a stop string, or disconnection. The
wrapper's old `finally` releases the per-replica lock without explicitly closing
the child generator. The child's native `finally` sends STOP and drains DONE,
but can run only after the wrapper frame releases its child reference.

A legal schedule is:

1. A stops consuming at EOS and closes the wrapper.
2. The wrapper releases the lock while A's native DONE remains unread.
3. B acquires the same replica and sends its GEN.
4. B reads A's DONE first and returns without a token.
5. A's child finalizer can consume B's tokens/DONE while cleaning up.

The native engine did generate B's tokens. The HTTP client did not manufacture
the empty reply. Depending on the schedule, this also loses cleanup errors as
an ignored generator-finalizer exception. A normal `for` loop does not provide
the close delegation of [PEP380](https://peps.python.org/pep-0380/#finalization).

## Root cause 2: DONE statistics remain shared after correct cleanup

Fixing the child close alone stops the pipe handoff problem, but does not make
`ReplicaEngine.last` request-owned. For an unbatched `StrataEngine`, `last` is
the shared native `_last` dictionary. After A unlocks, B can replace it before
`Service` reads A's result. A's token stream can be correct while its prompt,
decode, cache, or speculative statistics belong to B.

This is independently forced by yielding execution immediately after lock
release, after A's own DONE has already been consumed. Close-only passes 16/18;
the remaining failures are the caller-statistics test and the real-OS-pipe test
which also asserts statistics. This is not another instance of an unread DONE.

## Real failure and corrected request

The synthetic identifiers retain their historical `RVN` label; the actual model
was **GSQ-RCO IQ3_S**, not RVN. Request payload bytes are unchanged.

| Case | Input tokens | Old API output | Old borrowed native timing row | Fixed API / its own native DONE |
| --- | ---: | --- | --- | --- |
| SLOT-1-RVN-8020 |4165|0, finish=length|4167 input /13 generated|12, correct identifier, finish=stop;4165/12|
| SLOT-3-RVN-23858 |4167|13, correct identifier|4165 input /12 generated|13, correct identifier, finish=stop;4167/13|

The old native timing lines uniquely associate with the API timing fields within
their rounding tolerance, but the original GPU run did not record a per-token
thread trace. Source inspection plus event-controlled replay establishes the
mechanism; the GPU excerpts alone must not be described as such a trace.
[`gpu-historical-cases.json`](../tests/data/replica_lifetime/gpu-historical-cases.json)
contains the timings, exact payload SHA256s and fixed native lines; `cases/`
contains the two synthetic requests and original/fixed SSE streams.

## Three evaluated repairs and why A was selected

| Scheme | Ownership mechanism | Offline contracts | GPU/API qualification | Tradeoff |
| --- | --- | --- | --- | --- |
| **A: existing lock + caller snapshot** | Explicit child close; snapshot DONE under the existing lock; publish caller-local `last` before unlock |18/18|Completed below|Smallest serving change; no new worker, mailbox, or global serialization|
| B: request-scoped NativeTurn | One request object owns lock, iterator and result; `yield from` forwards close; idempotent turn close snapshots before release |18/18|NOT_RUN|Clearer structured ownership, extra object/iterator plumbing and a larger refactor|
| C: one native owner per replica | Dedicated worker alone reads/writes its pipe; bounded request-specific token/result channels |18/18|NOT_RUN|Stronger structural isolation, additional queue/scheduling cost; overload, restart/unload and real watchdog integration incomplete|

For C, the prototype used 32-token output channels and 8-entry mailboxes, with
backpressure. Its fixture's 5-second cleanup guard is not a production timeout
for native long prefill. None of the alternatives adds a global cross-replica
lock. Only A is proposed in the serving code.

The earlier control-only microbenchmark used the actual reader/pump and a CPU
producer over real OS pipes,13 symbolic token IDs per request, 1 CPU/768 MiB,
8 rotated rounds.65-ID requests had 2000 samples per adapter after 100 excluded
warmups; 4097-ID requests had 640 samples without a separate excluded warmup.

| Scheme |65-ID first fixture token p50/p99 ms|4097-ID first fixture token p50/p99 ms|
| --- | --- | --- |
| Original |0.143/0.227|1.351/6.617|
| A |0.132/0.191|1.285/6.502|
| B |0.139/0.204|1.356/6.716|
| C |0.428/0.671|1.719/6.379|

These are not model TTFT, prefill or GPU TPS. There are no confidence intervals,
and OS/GIL/quota tail latency was not decomposed. The small A/B differences are
not a model speedup claim. C's extra handoff cost and integration scope made A
the lower-risk first repair. Exact summaries are in
[`offline-alternatives-microbench.json`](../tests/data/replica_lifetime/offline-alternatives-microbench.json).

## Implementation of A

- Keep the native child in `gen`, initially None, so queued cancellation has no
  child to close and cannot borrow an earlier DONE.
- Explicitly call `gen.close()` before releasing the existing per-replica lock.
  The native implementation performs its existing STOP/DONE drain.
- Compare native `last` identity before/after this generation. Copy new DONE
  statistics into caller thread-local storage while pipe ownership is held.
- Keep the caller's pre-DONE identity stable. Without a new DONE, `Service` must
  not interpret an old engine result as this request's timing segment.
- Release the lock and balance inflight/router bookkeeping in `finally`, while
  allowing cleanup failures to propagate. Preserve existing nonfinite routing
  invalidation, independent replicas, embeddings and batched generation.

Native protocol, CUDA/CPU math, routing cost, context, sampling defaults,
speculation and CUDA Graph remain unchanged. The GPU-qualified original and
patched images have the same native binary SHA256:
`4ea6a2fa3361ec22d764944a570dd76ec03106295687e17b343798a18e106ff4`.
The fixed `serve/replicas.py` SHA256 is
`76b68ca40ebf0aa26f974cc66351322843f3bc4d77275c781f0f511639a03fe1`.

## Reproduce the public contracts

With Strata's Python dependencies installed, from the repository root:

```sh
python -m unittest -v serve.test_replica_lifetime
python tools/replay_replica_lifetime.py --output /tmp/replica-controls-new
```

The output directory must not exist. The replayer reads the original source
from commit 477ac13cbe8e582c20484871ce459d20d82df09d; fetch that commit first in
a shallow checkout, or use `--baseline-file` with its captured `replicas.py`.
It never edits the checkout. It requires the original 9/18, close-only 16/18 and
fixed 18/18 outcomes; original/close-only logs intentionally contain failures.

These tests import the repository's real `StrataEngine.generate`, `_pump`,
`_parse_done`, `last`, and `ReplicaEngine`. There is no AST method copy or global
`serve.server` stub. Only the native producer is simulated. Two tests use real
OS stdin/stdout subprocess pipes. Events force the handoff; sleeps are not the
repair. Coverage includes EOS, early close, stop-like close, slow consumers,
caller exceptions, delayed DONE, queued cancellation, native ERR, failed drain,
nonfinite cache invalidation, independent replicas, and 80 turns/four callers.
The failed-drain fixture checks error propagation and logical non-reuse, not
physical GPU process termination/restart.

## Completed validation and version boundaries

**Current main 477ac13 + A:** Windows CPython 3.12.14 and Linux CPython 3.12.3 both
give original 9/18, close-only 16/18, A 18/18 using the public real-reader tests.
Linux additionally passed 186 selected existing serving tests, each module in a
separate interpreter: 204 green tests including the new 18. This is not the whole
repository suite or a native GPU run of current main.

**GPU qualification 296b59d + the same A file:** engine 0.1.44.1, two RTX 3090
24 GiB SM86, dual physical EPYC 7663, VM 64vCPUs/two NUMA nodes, 384GiB assigned RAM,
Ubuntu 26.04 LTS VM host, driver 610.43.02. Container Python 3.12.3; CUDA 13.0.48,
libcudart 13/libcublas 13; GGML/llama.cpp pin
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Actual package/build/link metadata
is in [`dependencies.json`](../tests/data/replica_lifetime/dependencies.json).
The model was `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` plus the
corresponding PLE second shard. Exact model Hub revision was not captured.
Power limits and PCIe width were not isolated benchmark variables.

Replicas used one decoder and 31 CPU workers per GPU, prefill 8192, KV int8,
context 262144, spec 4/minp0.5, per-replica cache 16384MiB / 4slots / min 8192.
Resident used `--kv-resident 32768`; KVGROW removed it and used
`STRATA_KV_GROW=1`, INIT 16384, STEP 32768, HOLD 1. These profiles are not an
isolated full-VRAM KVGROW ON/OFF comparison. Relevant configurations and each
request's numbers are in `gpu-qualification.json`.

| Real GPU gate | Result |
| --- | --- |
| Resident and KVGROW initial generation/history |20/20 each|
| Four HTTP callers over two native decoders: EOS, stop strings, slow consumers and successors |60 completed replies/profile,120 total; zero empty completed replies|
| Intentional TCP disconnect after two content chunks, then exact-nonce successors |4/profile,8 total; all successors correct; cancelled requests excluded from successful throughput|
| Backend API, including tools/followup, Anthropic, Responses and input validation |9/9 per replica profile|
| KVGROW 262144-cell growth on both cards |Observed in each native log|
| Two distinct260088-input/2048-output requests, cold1 + warm 3 rounds |8/8 exact outputs, three warm output intervals overlap2 and telemetry observes both replicas busy|
| Generation/history after large KV growth |20/20|
| Long-output gross-degeneration audit |8/8 native C2 outputs plus11/11 B0 outputs pass the recorded thresholds|
| Original layer-split C4 shape using patched image |B0 11/11; history20/20;4 native decoding slots observed; backend9/9; authenticated API10/10; separate client-host nonce pass|

The earlier pinned-image selected serving suite passed 183/183 with modules
isolated. Its first combined run was 182/183: an existing hardware-sampler
thread-count assertion failed. The assertion was retained; the Responses
module passed 40/40 independently and the original combined control 183/183.
The unique scheduling trigger of that sampler assertion was not established.

B0 was 8K smoke, 4K excluded warmup, 131072/260088/independent260088 inputs three
times each, 2048 output for long requests. Context262144 leaves 8 tokens at the
native long shape; this is not a shape-matched Dominik comparison. Frozen B0
TTFT starts after HTTP headers, unlike the separate benchmark's pre-submit
E2E timer. All 11 have per-card 1 Hz VRAM; companion metrics started mid-B0 and do
not reconstruct an earlier time series.

For KVGROW 260K C2, cold prefill was 2420.5/2425.2 tok/s, max E2E TTFT 108.440s;
warm 3 median combined E2E throughput 114.54 tok/s, max TTFT median 3.183s,
cached 260081 per request. The original-shape layer C1 short control was
76.82 ->74.58 E2E tok/s (-2.92%), TTFT 75.1 -> 60.0ms, one epoch each with
five rounds/two excluded warmups. The wrapper is not exercised by that layer
shape; this point comparison does not establish a causal speed effect of A.

Production uses the patched image with its existing layer 24/24,C4,context 262144,
workers 47,groups 1,KV int8/resident 32768,prefill auto,cache 32768MiB/8 slots,
spec 4/minp0.5,temperature 0.7/top_p 0.9/top_k 30 and ADAPT_OVERLAP 0. Replica C2
profiles are qualification candidates, not a claim of four native decoders.

Not tested: GPU execution of current main 477ac13, HIP/SYCL/Metal hardware,
full-vocabulary KL/perplexity, long soak, real native crash/missing-ACK/restart,
unload/reboot recovery, fixed-full-KV/GROW0 or replicas C4 with this repair.
KVGROW startup reported only 463/464MiB CUDA free, below its 512MiB warning;
the recorded tests completed, but that is not a long-term headroom guarantee.

The complete private 1171-file raw archive was CRC/per-file-SHA256 verified;
its SHA256 is
`1ddeed7a9908d525a5310769b96978401d14aec151ed7e7702e0cc7ec476cb9d`.
It is not publicly hosted. This PR includes focused synthetic SSE/native
evidence, per-request results and control logs with a public manifest; private
host addresses, home paths and credentials are not published.
