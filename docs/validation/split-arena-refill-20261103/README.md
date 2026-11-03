# Experimental cross-window refills on a two-GPU expert arena

This is supporting evidence for a draft engine change, not a production qualification or a recommendation to
change defaults. The experiment removes part of the blocking adaptive-refill wait on a serial layer split.
The implementation remains behind both a compiler definition and an environment variable.

## Problem and approach

The existing split-arena path refuses `--adapt-async 1`. On this Q6 configuration, a refill round on the blocking
path spent about 18.6 ms waiting for CUDA events, or about 4.9 ms per decode window in dual-request batches.

The experiment reuses the asynchronous tier's state machine. At a global window gap, the main thread picks
the incoming experts, marks evicted experts nonresident, uploads that residency state to all cards, and submits
the copies to the owning cards' refill streams. The next verification window can compute while the copies
land. Misses use the CPU path during that interval. New residency is published after all used device events
complete. CUDA submission stays on the main thread to avoid introducing helper-thread doorbell interactions.

The change does not introduce an expert kernel or change the quantization. Different residency histories,
CPU/GPU rounding and `STRATA_ROUTE_RESIDENT=0.1` can nevertheless change the generated output and MTP acceptance.
No numerical-equivalence or semantic-quality result is claimed.

## Source and activation

Measured base: `99866a47a2b8a21b131aad162a584ee9da92f8f0`, app 0.2.9 / engine 0.1.55.
The submitted source starts from that exact revision. The only post-measurement source change extends each
experiment guard to exclude `STRATA_HIP_GFX906` and `STRATA_GPU_METAL`, in addition to `STRATA_USE_HIP`.
SYCL uses its separate migrated source. CUDA-enabled and compiler-opt-out preprocessing comparisons are
recorded in the submission validation. This is not a claim of HIP, Metal or SYCL build coverage.

The build has **no new CMake option**: enable the existing experimental compiler gate explicitly:

```sh
cmake -S . -B build-refill \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_PORTABLE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON \
  -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TAB=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86-real \
  -DSTRATA_GGML_DIR="$GGML_ROOT" \
  -DCMAKE_CXX_FLAGS=-DSTRATA_SPLIT_ARENA_ASYNC_EXPERIMENT=1
cmake --build build-refill --target strata strata_kq -j6
```

`GGML_ROOT` is a llama.cpp checkout at `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
The measured build used Debian 12, GCC 12 and a complete CUDA 13.1 toolkit; the archive contains the exact
build script, logs, compiler configuration and binary hashes. The experimental engine SHA-256 is
`4a30c8f7d6afa834fa8eafc85669272618523df1a210ed60346f154a631fbd1b`.
That fingerprint is in both campaigns' `source-origin.json`. The `e4406b47...` engine fingerprint in the
archived recovery reports identifies the original production engine restored after testing, not the A/B executable.

Both benchmark arms used that same executable, `--adapt-async 1`, and:

```sh
export STRATA_ADAPT_NOWAIT=0
export STRATA_ADAPT_LAG=1
export STRATA_SPLIT_ARENA_ASYNC=0  # A, original blocking arena path
# Set STRATA_SPLIT_ARENA_ASYNC=1 for B.
```

The path requires serve mode, multiple GPUs, the RAM arena, batch slots, one batch group, no multi-window
pipeline, no peer device, no remote helper, and no existing adaptive-overlap/NOWAIT/altered-LAG mechanism.
The existing asynchronous-tier checks still require adaptive rounds and swaps to be enabled. Confirm the
`strata experiment: split arena async: inline pick` startup marker; setting an environment variable alone
does not establish eligibility. `NOWAIT=0,LAG=1` is required for this state machine, not a new general default.

**Important control:** unmodified SM86 production uses `NOWAIT=1,LAG=2`. These A/B results compare matching
experimental scheduling overrides, not the patch against the untouched production scheduling defaults.

## Hardware and serving configuration

- Two RTX 3080 cards, 20 GiB each, 280 W limits; measured startup H2D probes about 11.1/11.2 GB/s.
- Xeon E5-2696 v4, 22 cores / 44 threads, AVX2. Sixteen expert-pool workers bound to CPUs 0-15,
  host core sibling, auxiliary CPUs 23-37; container affinity retained from production.
- Ubuntu 26.04.1 host; Debian 12 container. A read-only submission-time inventory reports Linux
  7.0.0-42-generic and NVIDIA driver 595.99.02; this later inventory is not a historical driver measurement.
- Qwen3.8-Flash-Next, Unsloth UD-Q6_K_XL, three configured slots, context 262144 per slot.
- Layer split 23: CUDA0 layers 0-22, CUDA1 layers 23-47. Total 5624 expert-cache slots, about 23.3 GiB;
  expert arena about 101.75 GiB. Exact startup placement is in each arm's engine log.
- KV int8, resident KV 32768; prefill auto:32768; reserves 1200/1500 MiB; PLE in RAM; trim stage weights.
- MTP configured with `--spec 4 --spec-min-p 0.5 --batch-mtp`, `STRATA_SPLIT_MTP_BATCH=1`.
- `STRATA_ROUTE_RESIDENT=0.1`, `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_HC_Q8=1`, `STRATA_MMQ_KQ=1`.
- Existing projection vector retained at scale 1.0, layers 4-44, project/per-layer. CPU vision, 16 threads;
  these were text benchmarks, so multimodal execution was not exercised.
- The configuration retained a server PCIe tuning field of 0.15; the engine's startup probe/status reported
  0.31. The redacted configuration and actual engine logs are both preserved; do not treat these as equivalent.

## Long-context results

Every row has one A sample and one B sample. Each decode request generated exactly 1536 tokens and finished
by length. Input accounting matched exactly. Temperature was zero and thinking was disabled. Synthetic tasks
were a long Chinese programming tutorial and SVG code, with maintenance text supplying the long context.

| Input per request | Concurrency | A decode tok/s | B decode tok/s | Change | A prefill tok/s | B prefill tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8000 | 1 | 50.60 | 55.40 | +9.49% | 805.74 | 806.45 |
| 8000 | 2 | 52.38 | 53.82 | +2.76% | 807.77 | 809.56 |
| 60000 | 1 | 52.70 | 56.20 | +6.64% | 2011.98 | 2013.39 |
| 60000 | 2 | 51.28 | 55.17 | +7.57% | 1982.23 | 1980.22 |
| 160000 | 1 | 51.80 | 53.00 | +2.32% | 2335.04 | 2335.89 |
| 160000 | 2 | 49.02 | 51.95 | +5.97% | 2310.72 | 2309.41 |

Single-request decode is the native engine rate. Dual-request decode is aggregate throughput in a trimmed
continuous interval where both slots decode and no prompt is read. It is not the sum of two request-wide
averages with different start/end times. Per-request native and SSE values remain in the archived summaries.

Cold prefill was measured with separate one-output-token probes, cooling between probes. For a dual case,
the two cold probes were serial; the table aggregates tokens divided by summed native prompt times.
After cooling to at most 50 C, the same contexts were replayed for decode, reusing at least input length minus
64 tokens. This is **not** an uninterrupted cold-prefill-plus-decode test or simultaneous cold-dual TTFT.
Prefill changed by at most 0.23% in magnitude; no prefill optimization is claimed.

| Dual input per request | A window ms | B window ms | A run ms | B run ms | A refill-wait ms | B refill-wait ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8000 | 71.60 | 68.52 | 63.74 | 65.62 | 4.99 | 0.25 |
| 60000 | 72.49 | 68.03 | 64.46 | 64.86 | 4.90 | 0.27 |
| 160000 | 75.98 | 71.97 | 67.92 | 68.79 | 4.93 | 0.27 |

These are host-side timings, not isolated CUDA operator times. Reduced refill wait is partly offset by
increased run time. Routing, output, MTP acceptance and adaptive-cache history are confounders, so the whole
end-to-end difference cannot be attributed solely to DMA overlap. A then B were separate cold boots with the
same six-case order; there are no repeated pairs or statistical-significance claims.

## Earlier short experiment and failures

The earlier short campaign used 1817/1836-token article/SVG prompts, cached contexts, 512 output tokens and
A1/B1/A2 order. Four A dual samples were 58.31, 56.42, 57.11, 59.71 tok/s; two B samples were 61.47, 59.33.
Means: 57.89 to 60.40 tok/s (+4.34%). Single-request native rates were A 51.9/51.8 and B 59.4 tok/s
(+14.56% against the A mean), with different MTP acceptance. This small sample was not treated as stable proof.

All failed/aborted attempts are retained and excluded from the accepted tables:

1. Short preflight 1 never enabled the experiment.
2. Short preflight 2 hit the conservative startup temperature guard.
3. Short preflight 3 discovered automatic SM86 NOWAIT/LAG defaults preventing eligibility.
4. Short preflight 4 rejected a startup CPU quota that Docker had not removed; no requests were sent.
5. The first uninterrupted long-context campaign reached the 65 C active guard during A 60K dual,
   with only 128/502 output tokens. IMC thermal-event counts were still zero. Its earlier completed cases
   are also kept separate from the staged campaign.

The accepted long campaign reached 63 C in requests, 64 C including reload/restore; accepted short requests
reached 61 C. Recorded IMC thermal events were zero and PMU running proportions exceeded 99.9%.
The 65 C long-campaign cancellation threshold was a conservative operational guard, not a measured thermal wall.
Startup used a two-CPU quota, removed and checked before requests. Cooling time is excluded from rate metrics.
Independent watchdogs restored the original service if the controller failed. The experimental image/config
was not deployed permanently; production hashes and restore checks are retained in the evidence.

## Validation and remaining work

The original measured CUDA engine and K-quant library compiled and linked. A separate diagnostic run reported
zero residency mismatches / stale CPU activation reports over 256 windows. Synchronizing residency diagnostics
were disabled for performance measurements. This checks bounded bookkeeping consistency, not model quality.

Submission-time checks include archived accounting reanalysis, macro-disabled preprocessing equality against
the original translation unit, CUDA-enabled preprocessing equality against the measured translation unit,
and a fresh CUDA-host translation-unit compilation. See the PR's Passed/Failed/Not run lists for outcomes.

**Unconfirmed static-analysis concern requiring follow-up:** the added all-device failure drain runs through
`adapt_quiesce`, while some direct error returns and EOF exits bypass that function. The source arena is
destroyed before later-device stages. Host unregister/staging may provide implicit completion, but the patch
does not establish an explicit all-owning-stream drain on every exit. No corruption was reproduced. An
experiment-only scope guard and EOF/fault-injection tests are proposed follow-up work before merge/deployment.

Other missing validation: HIP/gfx906/SYCL/Metal builds, Windows execution, model parity/KL or semantic-quality
testing, fault injection, sustained soak, more than two GPUs, pipelined groups, mixed GPUs, and repeated A/B
against unmodified SM86 production defaults. CUDA event waits can hang on device failure; an independent
external watchdog is required for any trial. Pageable host copies can block submission and reduce overlap.

## Inspect and reproduce the evidence

`evidence.tar.gz` contains the original short/long reports, methodology, fixtures, native records, full synthetic
SSE outputs, temperature/PMU observations, failed attempts, redacted configurations, engine logs, build logs,
source/artifact provenance, and original analysis/controller scripts. Model weights, binaries, source tarballs,
API keys and private production profiles are omitted. Private host paths/addresses are replaced with placeholders;
measurement numbers and generated content are preserved. Machine-specific controllers are records, not turnkey
instructions: read the methodology and adapt paths/service management before running anything.

Run the accounting checks with Python 3.12 or newer, without contacting any service:

```sh
python3 docs/validation/split-arena-refill-20261103/recheck.py
```

This verifies every archive member's SHA-256, then reruns both original analysis scripts in a temporary
directory. The archive contains its own `SHA256.json`. Source builds and model execution are separate checks.
The benchmark is a legacy community measurement, not an approved result under the draft standard protocol.
Preparation used AI assistance; the report distinguishes executed checks from static analysis and untested cases.
