# DISCOVER implementation: 128K context, 10 GiB VRAM

Implementation and available validation are recorded below. Final near-limit
request/memory validation and runtime-image packaging remain blocked by the
current sandbox (Docker socket access and localhost connections are denied).
`DONE` has intentionally not been created.
The user authorized implementation after DISCOVER.md was created. The original
working tree was dirty. `starting-state.json` and `starting.patch` capture that
state; it was copied to `build-discover-control/source` before implementation.
No deployment, image replacement, or merge was performed.

Hardware: RX 7700 XT, gfx1101, 12,272 MiB physical VRAM; separate gfx1036 iGPU
hidden with `HIP_VISIBLE_DEVICES=0`. CPU affinity exposes 12 physical/24 logical
cores. ROCm 7.2.1, hipBLASLt 100202, pinned GGML from the existing build cache.
The runtime image is unchanged and mounts separately built control/candidate
engines. Dedicated server: localhost:19931, container `strata-discover-bench`.

## Configuration and methodology

`control-config.json` and `candidate-config.json` contain complete engine
arguments, model/artifact paths, KV format, sampling configuration, and cache
settings. Both retain exactly 131,072 context tokens, IQ3_XXS, int8 KV, MTP,
speculation window 4, static profile, no adaptation or PCIe promotion, prefill
512, and explicit cache budget 512 largest-blob slots. Native per-pair sizing
turns this into 671 resident slots / about 1.11 GiB. The baseline engine uses
the same HIP allocation policy through LD_PRELOAD; the candidate links it.
The early control/candidate arm reserved 512 MiB for runtime resources and
256 MiB slack. The final guard raises the runtime reserve to 1,024 MiB, caps
tracked allocations at 8,960 MiB, and retains 256 MiB slack. Later 1K/2K chunk
arms use that final guard.

`tools/hip/bench_discover.py` counts the real tokenizer/template tokens, verifies
API context, uses greedy sampling, asserts zero reused tokens on fresh requests,
and records streaming TTFT, wall time, engine timing, output and usage. Five
repetitions per short workload follow an excluded warmup. `bench_prefill.py`
records alternating 4,210/8,830-token coding prompts and corresponding follow-ups,
with 128-token output caps. These capped answers measure throughput, not coding
correctness. `check_coding_task.py` separately requests a completed interval
merging function and tests empty/touching/nested/negative intervals, input
immutability, and 200 randomized cases against an independent coverage oracle.
That is a smoke test, not a broad quality evaluation.

VRAM is sampled every 50 ms with no desktop subtraction and zero tolerance.
Thus the measured raw-card bound is stricter than Strata's process budget, but
sampling cannot prove the absence of every shorter transient. The allocator
admits rounded explicit device allocations before calling HIP and clamps free
memory for cache sizing. Runtime/driver resources are covered by a reserve,
not falsely counted as tracked allocations. Managed/async pool APIs are rejected
under the guard; changing allocation APIs or ROCm requires reassessing coverage.

The first GPU diagnostic ran briefly during the first control coding prompt;
that initial pilot timing is not a clean isolated kernel measurement. Later
five-repeat streaming probes and kernel experiments ran serially on the GPU.

## Opportunity status

| Opportunity | Status | Evidence |
| --- | --- | --- |
| Tool-call parser | Retained | Incremental structural scan and fragment buffers preserve tool events/IDs and literal closing tags. Five repetitions at 64/128/256 KiB, both streaming modes: `parser.json`. |
| CPU workers | Retained | Automatic sizing selects 11 rather than 23 workers. Coding decode improves; short repetitive prefill regresses slightly. Raw samples in `*-prefill.json` and `*-stream.json`. |
| Prefill batching | Retained with pending near-limit validation | Configuration-specific `STRATA_PREFILL_MEMORY_REPORT` uses the existing exact buffer planner and cache-slot layout; context remains 128K. |
| Routing overlap | Rejected | Host grouping is below 0.1% of the measured timeline and profiling probes show no reliable gain. Restored original routing. `route-stream.json`, `rejected-routing-overlap.patch`. |
| hipBLASLt descriptor reuse | Rejected | Five isolated 200-call samples at T=4096 and T=37: differences within noise. Restored per-call descriptors; retained a 512-entry cap on algorithm and fallback metadata. `lt-tail-isolated.log`, `rejected-lt-descriptors.patch`. |
| AMD pooled-key scoring reuse | Rejected | Bitwise parity, NaN/tails/128K positions and three graph replays pass; isolated batches regress approximately 50–70% at long positions. Production kernel restored. `qsa-batch-isolated.log`, `rejected-qsa.patch`, `rejected-qsa-test.cpp`. |
| AMD matrix attention / GPU grouping | Deferred | No validated implementation or demonstrated need within the memory budget. Existing HIP numerical fallbacks remain. |
| Top-k dispatch | Deferred | 32,770 blocks at 128K fit the existing 33,792-block fast path; capacity does not justify a new dispatch. |

## Correctness and compatibility

Final production CTest run: all 45 available tests pass, including allocation
accounting, CPU topology, and HIP numerical tests (`ctest-final.log`). The full
production inventory has 48 tests; three require unavailable fixtures. The
initial experimental run had 44 passes and a memlock configuration failure. `platform_memory_test` initially failed because the test
container omitted unlimited memlock; it passes with `--ulimit memlock=-1:-1`
(`platform-memory.log`). Three tests require unavailable original fixtures:
`ple_parity` needs the Q2_0 PLE table and oracle captures; `expert_parity` and
`pool_test` need the original uniform Q2_0 `pack/full/experts.bin`. The native
IQ3_XXS pack cannot be substituted as though it were that format. These are
reported as unavailable, not passing. Actual native PLE and expert paths are
exercised by server requests; that does not replace the missing oracle tests.

The calibrated gfx1101 Lt table exercises BF16/F16, padded output strides,
output offsets, beta=0/1, exact/tail chunks, and repeated calls. The focused
candidate passes repeated Lt calls with zero fallbacks and preserves tolerances.
The table is a focused test artifact, not production tuning for all model shapes.

Python suite: 76 tests pass with two environmental skips. Six launcher contract
tests pass and reject budgets above 10 GiB and context values other than 128K. Dry-run
output verifies tuning variables and that an explicit CLI worker override comes
last. gfx1100 compilation passes; gfx1100 execution and CUDA compilation/execution
are unavailable on this host (no NVIDIA GPU or CUDA compiler).

## Measurements and memory ledger

Five fresh coding trials (three 4,210-token and two 8,830-token prompts) and
five corresponding cached follow-ups, excluding warmups:

| Configuration | Fresh prefill tok/s | Fresh decode tok/s | Fresh wall seconds | Follow-up decode tok/s | Follow-up wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control: 23 workers, chunk 512, native cache 671 | 126.4 | 18.0 | 42.18 | 15.7 | 11.46 |
| Automatic 11 workers, chunk 512, native cache 671 | 125.7 | 29.8 | 39.50 | 26.7 | 8.13 |
| Tuned: automatic 11 workers, chunk 2048, native cache 971 | 253.7 | 30.2 | 21.84 | 26.6 | 7.51 |

Values are medians across the mixed prompt lengths, not five trials per length.
The tuned chunk and cache change together; these measurements do not isolate
the contribution of each. Worker-only repetitive prefill regresses about 3–4%,
while coding decode improves. Existing CPU partitioning can change rounding and
some capped greedy replies differ; completed interval-merging smoke tests pass
205 cases in every retained configuration. This does not establish broad model
quality equivalence.

Five serial streaming samples per size: control/tuned median TTFT is
6.331/4.875 seconds at 1,024 tokens and 23.218/14.728 seconds at 4,096 tokens.
Parser median speedups at 64/128/256 KiB are 6.1/10.9/21.2x for complete tool
arguments and 3.9/7.1/13.5x for streamed arguments (`parser.json`).

The final launcher defaults are workers 0 (automatic), prefill 2048, cache
budget 800, later explicit allowance 768 MiB, runtime reserve 1024 MiB, slack
256 MiB, and context 131072. CLI tuning overrides are appended after defaults;
ring/thread/Lt tuning environment variables are forwarded. The Dockerfile and
entrypoint carry the same defaults. An old image cannot safely enforce these
allocations: `run.sh` checks its guard label and requires rebuilding.

The native pack's memory components, rounded in the guard, are approximately:

| Component | Bytes |
| --- | ---: |
| Canonical dense | 1,538,064,384 |
| Native matrices | 1,877,934,080 |
| Full 128K session (KV, indexer, recurrent state, scratch) | 2,024,931,328 |
| MTP dense/experts/session | 996,343,808 |
| Native Q5 head | 437,059,584 |
| Baseline cache (671 slots) | 1,191,772,160 |
| Tuned cache (971 slots, requested bytes) | 1,713,126,400 |
| Verify window | 73,007,104 |
| Loop logits | 5,963,776 |

Small metadata/PLE/graph allocations are also charged; these rows are not a
sum-of-independent-peaks bound. Workspace lifetimes overlap differently.
The 2,048-token scratch is 1,280,477,952 bytes borrowed from cache, rather than
an additional allocation. 4096/6144/8192 chunks fail the exact planner for this
cache, before their extra buffers are allocated. The tuned startup trace reaches
8,669,167,616 tracked bytes (8,267.56 MiB), below the 8,960 MiB admission cap.
The 768 MiB later allowance exceeds the observed approximately 80 MiB later
explicit allocations; runtime and slack are reserved separately.

Completed audits: control raw-card peak 9,802.70 MiB over 10,669 samples; the
1K-chunk raw-card peak 9,819.58 MiB and attributed Strata peak 8,134.18 MiB over
8,465 samples. Both pass the 10 GiB limit. The latter uses deduplicated AMD DRM
allocation totals, with no desktop subtraction. Explicit ledger peak on the
smaller-cache arm is 7,770.31 MiB, leaving measured non-ledger resources roughly
343–364 MiB; the final runtime reserve is 1,024 MiB. Sampling and reserved
headroom do not prove a bound for every future driver allocation or new API.
The final 2K-arm audit has not been finalized; do not claim its process peak or
near-limit requests passed.

## Remaining validation and reproducible continuation

The environment changed to a restricted sandbox after the five-trial tuned
benchmark. Docker API access now returns permission denied; localhost sockets
return Operation not permitted. See `runtime-image-blocked.log` and
`long-context-blocked.log`. No runtime image was replaced. The dedicated test
container/guards could not be stopped or finalized through the now-inaccessible
Docker API. Their current state must be checked when Docker access returns.

Before creating the requested `DONE` file:

1. Run fresh and cached continuation probes at 32768, 65536, and 130944 real
   tokenizer tokens, with output room inside the unchanged 131072 context:

   ```sh
   python3 tools/hip/bench_discover.py --url http://127.0.0.1:19931 \
     --model strata-discover \
     --tokenizer /mnt/storage/Development/strata-work/packs/iq3_xxs/tokenizer \
     --engine-log build-discover-candidate/chunk2048-engine.log \
     --output bench/results/2026-10-01-discover-128k-10g/long-context.json \
     --label final-long-context --sizes 32768,65536,130944 \
     --repetitions 1 --followups --timeout 7200
   ```

   These are capacity/correctness probes, not five-repeat performance claims.
   Follow-ups must report positive cache reuse. Repeat the completed coding
   smoke test afterward and exercise cancellation/recovery while both raw-card
   and per-process guards sample at 50 ms.
2. Finalize the 2K-arm raw/process audit JSON, gracefully quit the dedicated
   engine, and save its final tracked peak. Stop only the dedicated test resources.
3. Build a separate review image without replacing existing tags:

   ```sh
   docker build -f docker/Dockerfile.hip \
     --build-context engine=./build-discover-candidate \
     --build-context ggufpy=./build-hip/_deps/strata_llamacpp-src/gguf-py \
     --build-arg STRATA_HIP_ARCH=gfx1101 -t strata-hip:gfx1101-discover .
   ```

   Verify runtime dependencies, guard label, and default generated config.
   `entrypoint-test.log` already validates the mounted-script config with explicit
   tuning values; final packaged defaults still require validation.
4. Review this report, mark the remaining gates with actual results, and create
   repository-root `DONE` only after they complete.
