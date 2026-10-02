# Performance discovery and implementation plan

This document is the execution plan. The user subsequently authorized its
implementation. See [implementation results](bench/results/2026-10-01-discover-128k-10g/README.md)
for retained changes, rejected experiments, correctness checks, and limitations.

## Prompt for the implementing agent

Improve Strata's measured performance on the existing AMD HIP/Docker deployment
by following the phases below. Establish a reproducible baseline, implement
small changes independently, run correctness checks, and retain only changes
supported by measurements. Complete the feasible opportunities and document
rejected or deferred experiments with evidence.

Two constraints take precedence over every performance objective:

1. **Strata may use at most 10 GiB of VRAM: 10,240 MiB / 10,737,418,240 bytes.**
   This includes engine allocations, runtime/library allocations, transient
   buffers, staging, graphs, workspaces, and lazy initialization. An average,
   startup reading, or successful allocation is not proof of compliance. Never
   deliberately approach an OOM or trigger a GPU reset to discover the limit.
2. **Keep the configured context at exactly 128K tokens: 131,072.** Preserve that
   capacity in the launcher, generated configuration, engine, and API. Do not
   reduce context or silently truncate prompts to obtain a speedup. Count chat
   template tokens and output room when constructing near-limit tests.

Use the current model, quantization, tokenizer, template, MTP artifacts, and
expert profile for comparisons. Preserve existing user changes. Keep the pinned
GGML dependency. Build HIP for the actual GPU architecture; keep the iGPU hidden
and `HSA_OVERRIDE_GFX_VERSION` unset. Existing gfx1100 and CUDA paths must remain
build-compatible; distinguish compilation from execution on available hardware.

Work autonomously through successful gates. A failed gate means repair, revert
the candidate, or document the blocker before dependent experiments. Report
progress and evidence without turning every phase into a permission request.
Use a dedicated benchmark instance; do not stop unrelated services or replace
the user's running deployment. Finish with reviewable code, tests, results, and
a recommended configuration. Do not deploy, publish, or merge the result.

## Starting observations

These are findings from the reviewed working tree, not promised speedups. Verify
them again if the code changes before execution.

| Opportunity | Evidence | Initial priority |
| --- | --- | --- |
| CPU worker sizing | `docker/entrypoint-hip.sh` chooses `nproc - 1`; `src/kernels/cpu/pool.cpp` already supports automatic physical-core sizing with zero workers. The reviewed host exposed 24 logical CPUs / 12 physical cores: 23 launcher workers versus 11 automatic workers. | First, low implementation cost |
| Prefill batching | Docker defaults to 512 tokens. `src/prefill/prefill.cpp` enables the larger streaming ring from 2,048 tokens. Larger chunks amortize uploads but require more scratch and can displace cached experts. | High, after memory sizing |
| HIP QSA kernels | `qsa_block_scores_tc` and `qsa_prompt_attn_batch` return false for HIP; their CUDA matrix kernels are unavailable on AMD. | Potentially high, substantial kernel work |
| Host routing overlap | Prefill queues routing and shared-expert GEMMs, then synchronizes the compute stream before CPU grouping. | Medium/high, profile first |
| Top-k dispatch | The register kernel supports 33,792 blocks. At 128K, capacity is `131072 / 4 + 2 = 32770`, so the optimized path already fits. | Defer capacity-based dispatch changes |
| Tool-call parsing | `OutputParser.feed` calls `call_end` on the growing body for every delta. Synthetic eight-character deltas took approximately 0.069 / 0.260 / 1.019 seconds for 64 / 128 / 256 KiB bodies, consistent with quadratic work. | Bounded frontend improvement |
| hipBLASLt descriptors | Algorithm choices are cached, but `try_hipblaslt` constructs and destroys operation/layout descriptors on each successful launch. | Low-risk host-overhead experiment |

Incremental production detokenization, reusable expert staging, and overlapping
GPU cache hits with CPU misses already exist. Preserve those optimizations.
HIP MMQ is enabled by default when built and supported in this tree; do not
mistake an older document's runtime opt-in description for current behavior.

## Phase 0: baseline and VRAM enforcement

### Record the exact starting state

- Capture revision, dirty diff, new files relevant to the build, binary hash,
  compiler/runtime/library versions, GPU architecture, CPU topology/affinity,
  available RAM and cgroup limits, model/artifact identities, engine arguments,
  environment, and actual expert-cache slots. Do not reset the working tree.
- Build in an isolated directory/image with Release, HIP enabled, CUDA disabled,
  MMQ enabled, and tests enabled. Use the existing builder workflow. Preserve a
  runnable control binary/image and its complete effective configuration.
- Verify `--max-context 131072` and engine/API-reported capacity. Keep the same
  sampling, reasoning, speculation, KV format, cache adaptation, and output cap
  between arms. Default KV remains int8 unless a separately measured experiment
  explicitly changes storage format while preserving context and correctness.

### Make the memory constraint effective before GPU experiments

Inspect `docker/hipinfo.py`, `docker/entrypoint-hip.sh`,
`docker/vram-guard.py`, `src/program/generate.cpp`, and prefill/session/verifier
allocation sizing. Produce a peak-memory ledger covering:

- Native/dense weights and output heads; session and recurrent state; KV,
  indexer, RoPE and attention scratch at the full 128K configured capacity.
- Expert cache, MMQ buffers, prompt scratch, host-to-device staging rings,
  BLAS workspaces, graph resources, draft heads, and every supported speculative
  window, including windows first allocated on a later request.
- Temporary coexistence during buffer growth, cache borrowing/refill, graph
  creation, library initialization, failed-path cleanup, and repeated requests.
- Device allocations outside Strata's explicit allocator and uncertainties that
  must be covered by conservative headroom.

The existing `STRATA_VRAM_LATER_MIB` default is based on a 512-token prompt
configuration. **It is not a valid blanket allowance for a larger prefill sweep.**
Derive configuration-specific bounds before running each arm. Preserve at least
the existing 256 MiB internal slack; increase reserve or reduce expert-cache
slots when additional scratch is needed. Never increase the 10 GiB budget or
remove headroom to make an experiment fit. Reject an unsafe configuration before
its allocations begin. If complete peak bounds cannot be established, keep the
conservative configuration and record the blocked experiment.

Check whether current sizing actually enforces the bound. Implement any necessary
budget accounting/preflight before performance experiments, with explicit
handling for late allocations and safe fallback to a smaller expert cache.
An environment variable or reservation estimate alone is not a hard allocator
limit. Do not rely on allocation failure as the fallback trigger.

Run an independent memory audit from before startup through initialization,
first prompt, first use of supported verify windows, longest-context requests,
cache refill, cancellation, restart, and repeated requests. Use the guard with
`--budget-mib 10240 --tolerance 0`; its default 2% tolerance violates this task's
hard limit. Remember that the existing guard only reports: it cannot prevent a
breach. Supplement periodic samples with allocation/peak accounting; sampling
can miss short-lived allocations and the current MiB readings round down.

The budget is Strata's complete device footprint, while desktop usage is
separate. Record raw card occupancy as well. Prefer per-process attribution where
available. Subtracting a single desktop reading from later raw totals is only an
estimate if other usage changes; it cannot establish a strict pass. Keep the
desktop stable or use a conservative bound, and preserve physical free space
for it. If auditing cannot reliably establish compliance, the gate is incomplete.

**Gate:** configuration remains 128K, peak bounds fit within 10 GiB with headroom,
the control completes safely, and correctness checks establish a usable baseline.
If the control is already unsafe, correct memory sizing before measuring speed.

## Phase 1: fix worker sizing and measure configuration choices

Change the Docker worker default to engine auto-sizing (`--pool-workers 0`),
retaining explicit positive overrides. Confirm command-line and environment
overrides reach the engine through `run.sh` and the entrypoint; do not silently
ignore a supported override. Log the effective worker count.

Validate physical-core selection under SMT and restricted CPU affinity. Use
existing pool tests, including sleeping workers and tiny batches. Add a focused
launcher/configuration check only where it verifies the default or override
contract. Benchmark automatic sizing and a few nearby physical-core counts;
compare against the original logical-CPU-derived count. Track host CPU load,
pool wait/compute time, decode throughput, and request latency.

After Phase 0 admits each configuration, sweep prefill sizes in ascending order:
512, 1,024, 2,048, 4,096, and 8,192. Skip sizes whose peak bounds do not fit.
Keep 128K context in every arm. Record staging-ring depth, cache borrowing,
cache slots before/after prefill, hit rate, and post-prefill decode performance.
Do not independently maximize ring depth: it also consumes VRAM. Verify smaller
chunks and final partial chunks still work.

Use only hipBLASLt tuning tables calibrated for the actual architecture and
library version. The existing gfx1100 table is not a gfx1101 calibration. Measure
actual fallback shapes before deciding whether local recalibration is useful;
any tuning tool's allocations must also fit the budget and run without a
concurrent engine that would exhaust it.

**Gate:** choose a configuration using repeated request measurements, preserving
decode behavior, full context, and the memory ceiling. Document rejected sizes.

## Phase 2: remove bounded host overhead

### Tool-call parser

Make delimiter discovery incremental instead of rescanning the complete call
body on every feed. Retain scan state and only inspect new text plus the suffix
needed for split delimiters. Accumulate completed body fragments and join when
needed, rather than repeatedly copying a growing string. Streaming raw JSON
parameters must also avoid rescanning their entire growing value.

Preserve existing behavior for terminators embedded in parameter values, split
tags, whitespace, multiple parameters/calls, string versus structured values,
Unicode, streaming argument JSON, call IDs, and unfinished calls/cancellation.
Keep final arguments identical to streamed arguments.

Use `serve.test_server.ToolCallTerminators` and relevant existing parser/API
tests. Add regression cases around changed state transitions. Benchmark fixed
1/8/32-character deltas and realistic token fragments at 32/64/128/256 KiB;
include ordinary content as a control. Report total time, late-feed latency, and
peak host memory. Aim for approximately linear total work; do not use brittle
wall-clock thresholds in correctness tests.

### hipBLASLt descriptor reuse

Cache operation and matrix-layout descriptors alongside validated algorithm
choices, keyed by dtype, shape, strides, and any descriptor-affecting attributes.
Preserve beta-dependent algorithm validation. Scope ownership to the existing
GEMM state/device, bound cache growth for tail shapes, handle workspace rebinding,
and retain current failure/fallback semantics. Check actual threading ownership
before sharing descriptors. Ensure teardown and failed initialization release
resources.

Run the existing Lt GEMM parity checks and compare supported calls, unsupported
shapes, nonzero beta, strided output, repeated calls, and workspace changes.
Measure host submission overhead and request throughput separately.

**Gate:** frontend outputs and GEMM tolerances remain correct; memory stays
bounded; retain each change independently according to its measured benefit.

## Phase 3: overlap routing/grouping with shared-expert GEMMs

Profile `src/prefill/prefill.cpp` with `STRATA_PREFILL_TIMING=1`. The current
stream synchronization before host grouping also waits for shared-expert work.
First attempt the smaller scheduling change:

1. Publish/copy router IDs immediately after `route` and record a completion
   event after the publication.
2. Queue shared-expert GEMMs on the compute stream.
3. Wait for the router-publication event on the host and group IDs while those
   GEMMs execute. Preserve previous-layer ordering for reused slot/source/bounds
   tables and mapped host memory.
4. Queue grouping-dependent device work with explicit dependencies. Keep staging
   ownership and expert arrival/release rules intact.

Use reusable events and buffers. Do not replace the synchronization with an
unprotected host read, weaken memory visibility, or introduce device-wide waits
or per-layer allocations. Validate mapped and explicit-copy grouping paths,
resident and streamed experts, MMQ and fallback products, multi-chunk prompts,
tail chunks, cancellation, and repeated requests. Preserve deterministic
grouping/scatter order and cache refill correctness.

If grouping remains a meaningful bottleneck, consider GPU histogram/prefix/scatter
as a separate experiment. Retain the host metadata required for expert staging;
GPU grouping is not automatically beneficial if the host immediately needs a
blocking readback. Charge all new buffers against Phase 0's budget.

**Gate:** traces demonstrate overlap and a repeated end-to-end improvement;
numerical and sequence-state behavior remains within existing contracts.

## Phase 4: AMD QSA acceleration

Treat scoring and prompt attention as separate changes, each with an independent
fallback, benchmark, and numerical gate. Inspect actual gfx1101 matrix-instruction
support and compiler output; CUDA inline PTX cannot be enabled on HIP by removing
the guard. Preserve gfx1100 support. Prefer a supported HIP implementation or
architecture-specific dispatch with the existing fallback for other cases.

For scoring, batch queries so pooled keys can be reused. Preserve four indexer
heads, per-head ReLU before head summation, causal visibility, tail/dead-block
handling, the partial-block bonus, NaN behavior, and selection tie rules.

For attention, preserve supported KV formats, scales, page-table addressing,
causal selections, grouped query heads, stable softmax, output layout, and the
rotation/unrotation semantics of existing quantized paths. Start with the
production int8 configuration. Do not silently drop other supported modes;
dispatch them to the verified fallback until accelerated and tested.

Use existing QSA selection, prompt attention, native QSA/indexer, KV streaming,
and quantization parity tests as appropriate. Extend them with partial blocks,
near ties, extreme inputs, continuation across chunks, repeated graph replay,
and near-128K positions. Preserve existing tolerances; do not loosen them merely
to accept a faster kernel. Where matrix arithmetic changes reduction order,
report error distributions and selection disagreements explicitly, then assess
downstream logits and completed tasks. A numerical tolerance pass alone does
not establish unchanged answer quality.

Measure occupancy/register pressure, score/key reuse, attention time, memory
traffic, total prefill latency, and decode after prefill. Kernel scratch must fit
within the same 10 GiB ledger and headroom. Introduce dispatch thresholds only
when measurements show where the accelerated path wins.

**Gate:** existing fallbacks remain usable, supported builds pass, the tested
128K configuration remains safe, and the new kernels improve actual requests.

## Phase 5: verify top-k behavior; defer unnecessary redesign

Confirm production prefill, decode, and verification use the optimized register
top-k path at 128K, with `STRATA_TOPK_OLD` unset and supported geometry. Record
capacity, dispatch, and time in the baseline profile. Add boundary coverage if
other changes affect this dispatch.

The previously suggested active-length dispatch change mainly addresses larger
configured contexts, which are outside this task. Do not spend implementation
time on it unless profiling reveals a distinct bottleneck within the fixed 128K
configuration. Never increase configured context to manufacture evidence.
Any later top-k work must preserve ascending IDs, lowest-index tie handling,
graph capture, and capacity/stride correctness.

## Measurement and correctness protocol

- Use `tools/hip/bench_prefill.py` for the existing small fresh/follow-up workload,
  with a dedicated server and exactly one completed timing record per request.
  It does not cover 128K behavior or peak VRAM by itself.
- Add a reproducible benchmark driver covering approximately 1K, 4K, 8K, 32K,
  64K, and near-128K rendered prompts. The largest prompt plus output allowance
  must fit 131,072 tokens. Assert zero reused KV tokens for fresh-prompt arms;
  use unique cases and controlled restart/cache state. Measure cached follow-ups
  separately and report actual reused tokens.
- Include short and long decode, greedy reproducibility, representative sampled
  settings, production reasoning settings, large tool-call bodies, cancellation,
  and subsequent requests. Keep sampling/reasoning identical in each comparison.
  Test supported speculative window sizes and first-use allocation paths.
- Separate initialization/first-use, warmed filesystem state, and conversation
  cache reuse. Zero KV reuse does not imply a cold filesystem. Do not flush the
  host's global filesystem cache or change unrelated services.
- Run uninstrumented throughput comparisons with diagnostic timing/profiling
  disabled. Use `STRATA_PREFILL_TIMING=1` and `STRATA_VERIFY_PROFILE=1` only for
  separate attribution runs; keep the independent VRAM audit active.
- Warm up each arm, run at least five matched repetitions, and alternate control
  and candidate order. Record median and spread, raw samples, TTFT, request wall
  time, fresh prefill tokens/s, decode tokens/s, output/reuse counts, finish
  reason, speculation acceptance, cache hit rate, peak VRAM, and host RAM/CPU.
  Account for temperature, power limits, competing load, and measurement noise.
- Accept a performance change when the targeted gain is repeatable beyond the
  observed variability and representative workloads show no material regression.
  Treat a repeatable regression above 3% as requiring explanation and rework or
  rejection; do not claim gains within noise. Report workload-specific tradeoffs.
- Run the relevant existing CTest and Python suites for each change, then the
  supported HIP regression set for the final candidate. Enumerate tests first;
  report missing fixtures or environmental limitations individually instead of
  blanket exclusions or claimed passes. Test GPU cases serially when concurrent
  execution could exceed the combined VRAM budget.
- Use deterministic output/logit comparisons for scheduling/descriptor changes.
  For altered arithmetic, retain numerical tests and complete representative
  coding tasks with independent runtime tests. Capped throughput replies are
  not completed-task quality evidence.

## Delivery and completion criteria

Keep independent changes reviewable and record rejected experiments. Write results
to a new `bench/results/<date>-discover-128k-10g/` directory containing:

- A README with hardware/software, complete configuration, commands, methodology,
  workload definitions, conclusions, and limitations.
- Control/candidate identities including dirty-state patches where relevant;
  raw benchmark samples; phase profiles; memory ledgers/audits; correctness output.
- Per-opportunity status: retained, rejected, or deferred, with measured reasons.
- The recommended launcher settings and an explicit check that overrides reach
  the container/engine. Update user-facing documentation only with measured claims.

The final candidate must retain exactly 128K context, fit Strata's entire peak
device footprint within 10 GiB, preserve desktop headroom, pass applicable
correctness checks, restore cache state after prefill, and show repeatable gains
on real requests. Any opportunity that cannot meet these constraints remains
deferred; it is not a reason to relax them.
