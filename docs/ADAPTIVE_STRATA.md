# Adaptive Strata: research and experimental groundwork

Status: isolated research branch, 2026-10-08. Not a released feature or a measured speed improvement.
Production configuration is unchanged. The intended objective is correct agentic task completion in less
wall-clock time while leaving resources available to foreground applications.

## Scope

Memory capacity and compute availability are separate constraints. A busy CPU should not automatically
cause GPU cache eviction. A busy GPU should not automatically evict useful RAM. The eventual controller
should choose among supported CPU and GPU execution paths, RAM and VRAM residency, and reads from backing
model files according to measured completion cost. Returning resources to Strata after another workload
finishes is part of the objective.

This branch currently provides live expert-cache resizing and an experimental single-GPU routing heuristic.
It does not provide general tensor relocation, live CPU-worker resizing, multi-GPU balancing, SSD selection,
storage-aware routing or a calibrated cost model. Some allocations still have a fixed GPU minimum.

## Existing Strata work and attribution

The live allocator, MEMORY protocol, acknowledgement handling, RAM blocks, GPU VMM resizing and resource
presets are based on medking82's [PR #726](https://github.com/Niko1221/Strata/pull/726), ported onto
v0.1.40.3. Its author is retained in Git history. That PR was closed without merging during history cleanup;
closure should not be described as a technical rejection.

Other related work must be considered before an upstream submission:

- [PR #1117](https://github.com/Niko1221/Strata/pull/1117): automatic GPU cache resizing, excluding the
  resident CPU complement. Its code is not part of this branch's current implementation.
- [PR #1324](https://github.com/Niko1221/Strata/pull/1324): elastic RAM LRU and file fallback, with Windows
  commit pressure. Its code is not part of the current implementation; both changes need a common policy owner.
- [PR #1461](https://github.com/Niko1221/Strata/pull/1461): peer-tier cooperation and changing roles across
  two GPUs. Relevant to a future multi-GPU design, not validated on this laptop.

Changes made during this port include draining newer asynchronous file readers before changing resident
storage, considering available Windows commit as well as physical RAM, updating server test fixtures for
current slot state, and using application-neutral workload signals. These are port fixes, not techniques
copied from the external research below.

## External work worth using

### ATSInfer: measured placement under changing load

[Paper, v2](https://arxiv.org/html/2607.10183v2): Yangyijian Liu, Hongyi Ye, Mingyang Li and Wu-Jun Li,
*Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices*, arXiv:2607.10183v2,
14 July 2026. It profiles CPU execution, GPU execution and transfers,
then revises scheduling as runtime costs change. Its load experiment reports a 13% reduction in the
increase in time per token under 60% external CPU load, and 38% under 60% GPU plus PCIe load, compared with
its static policy. These are not local Strata gains. Runtime promotion of CPU-default tensors does not
make it a complete bidirectional memory and disk manager. An original author implementation repository
was not verified in this search.

An [independent implementation](https://github.com/lordnyx0/ik_llama-atsinfer) records negative results:
[findings](https://github.com/lordnyx0/ik_llama-atsinfer/blob/main/docs/atsinfer-findings.md). In its tests,
splitting fused operations made static placement much slower; a dynamic transfer could cost more than
the CPU work it avoided. This makes preserving Strata's useful fused expert operations and measuring
transfer break-even points requirements, not optional optimizations.

### StarPU: a general heterogeneous runtime

[Features](https://starpu.gitlabpages.inria.fr/features.html),
[repository](https://github.com/starpu-runtime/starpu),
[out-of-core design](https://github.com/starpu-runtime/starpu/blob/master/doc/doxygen/chapters/starpu_extensions/out_of_core.doxy).
StarPU schedules supported CPU/GPU task implementations using performance models, data location and
dependencies, and manages transfers and eviction. Its out-of-core support adds disk-backed storage.
This is the strongest broader systems precedent for the requested design. Its data-aware scheduler
compares expected completion and transfer costs. Adopt those principles in Strata's existing execution
path; importing the entire runtime would be a major integration project.

### Meta TMO / Senpai: react to lost progress, not just percentages

[Engineering description](https://engineering.fb.com/2022/06/20/data-infrastructure/transparent-memory-offloading-more-memory-at-a-fraction-of-the-cost-and-power/),
[Senpai code](https://github.com/facebookincubator/senpai).
The controller uses Linux memory-pressure stall measurements to probe for reclaimable memory and back
off when reclaim harms progress. Windows needs its own observations: inference wait times, hard faults,
commit headroom and storage latency. The Linux controller is not a Windows drop-in. No source was copied.

### DAMON / DAMOS: bounded movement and stability

[Kernel documentation](https://docs.kernel.org/admin-guide/mm/damon/usage.html).
Access monitoring, hot/cold classification, migration quotas and watermarks are useful patterns for
preventing oscillation. A Strata controller should cap bytes moved per interval and distinguish emergency
release from gradual recovery. Linux kernel mechanisms are references, not settings to enable on Windows.

### SGLang and vLLM: keep conversation state separate from weights

[SGLang HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design) and
[vLLM KV offloading](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/) provide GPU/host/storage
cache hierarchies with asynchronous transfers and lifecycle management. These are useful for prefix and
conversation state. They do not by themselves decide where Strata's expert computation should run.
Immutable expert weights and mutable KV/recurrent state require different eviction and correctness rules.

### XSched: foreground compute responsiveness

[Repository](https://github.com/XpuOS/xsched). Accelerator queue scheduling and preemption address compute
sharing independently of memory capacity. Its documented tested CUDA architectures do not establish
compatibility with this RTX 4070 Laptop. No interposition or driver changes were applied.

### Additional leads, with limitations

- [Adaptive KV streaming fork](https://github.com/troed/llama.cpp-adaptive-kv-streaming): useful bounded
  GPU cache and host staging ideas, but its documented path does not provide automatic VRAM-pressure eviction.
- [Swap-MoE](https://github.com/ek15072809/Swap-MoE): expert reads from SSD and prefetch hints. Its GPU
  support limitations prevent treating it as a ready replacement for Strata's hybrid path.
- [SparkInfer](https://github.com/ganminghao/SparkInfer): activation-driven CPU/GPU neuron balancing.
  Activation sparsity needs separate quality validation and is not equivalent to transparent exact placement.
- [NeuroPrefetcher](https://github.com/nobeldhar/NeuroPrefetcher): explicit NVMe scheduling on Jetson;
  different hardware and sparse execution. Results are not transferable to this configuration.
- [FlexLLMGen](https://github.com/FMInference/FlexLLMGen): useful three-tier weight movement precedent,
  but archived and focused on batch throughput rather than interactive desktop coexistence.
- [Reddit discussion](https://www.reddit.com/r/LocalLLaMA/comments/1ssczvu/how_do_you_actually_manage_vram_when_running/):
  users report the same application-sharing problem. Replies mostly describe caps, unloading or routing to
  another device. These anecdotes establish demand, not correctness or speed of an adaptive solution.

No external code from these projects has been incorporated. Sources are design references. Review licensing
and preserve attribution if code is used later. No complete drop-in implementation matching the proposed
Windows Strata system was verified; this is not a claim that none exists.

Any upstream design or implementation PR should cite the source at the point where its idea is used and
state whether the contribution is inspired by the design, adapts source code, or only compares results.
Do not attribute local port mistakes or their corrections to unrelated papers. Keep paper version, original
repository and exact revision for code reuse; retain required license notices and original commit authors.

## Proposed architecture

1. Discover separate memory and execution resources. Track each GPU, CPU topology, shared host memory,
   actual model file locations and physical storage devices. Count iGPU memory as shared system memory,
   not additional capacity. Multiple partitions are not automatically independent disks.
2. Measure phase-specific costs. Keep separate prefill, decode, MTP verification and conversation-restore
   observations. Record CPU work/wait, GPU work/wait, unhidden transfer time, file-read latency, cache hits
   and pressure. Total GPU utilization during inference cannot identify another application's GPU load.
3. Rank legal actions by predicted completion time. Include queue delay, kernel duration, required reads,
   transfers not hidden by overlap, migration amortization and uncertainty. Do not add overlapping intervals
   twice. Prefer an existing resident copy when its expected finish time is lower.
4. Use one capacity owner. Reserve fixed GPU workspaces and minimum staging/host budgets first, then assign
   elastic budgets. Respect physical RAM, commit and actual device allocator limits independently. Never
   assume nominal RAM plus nominal VRAM is one allocation pool.
5. Move at verified safe boundaries. Preserve dependent buffers, wait for active readers, apply bounded
   steps, acknowledge actual allocations and leave the old mapping usable on failure. Cancel a proposed
   move when its destination becomes pressured.
6. Release quickly, recover gradually. Use minimum residence times, sustained fresh readings, migration
   byte/time quotas and backoff after negative benefit. Re-evaluate when resource or inference phase changes.
7. Maintain correct state. Expert weights already present in the GGUF can be evicted without writing a
   second copy to SSD. KV/recurrent and conversation state must be saved and restored using compatible,
   complete snapshots. Do not lower quantization, drop experts or change context to disguise a fit failure.
8. If no supported configuration fits the non-evictable minimum, stop admitting work or pause safely.
   General dense/attention CPU fallback would require additional backend work; cache resizing alone cannot
   release every byte of GPU memory.

Expected behavior:

| Condition | Candidate response, subject to measured benefit and supported kernels |
| --- | --- |
| CPU busy, GPU has compute and memory room | Increase GPU execution; retain useful GPU experts; reduce active CPU work when supported |
| GPU busy, CPU and RAM available | Reduce GPU expert work and cache; retain or grow hot CPU-resident experts |
| RAM pressured, GPU has room | Promote useful weights; release redundant/cold host blocks; read misses from model files |
| RAM and VRAM pressured | Keep minimum execution state; reduce both elastic caches; bound file prefetch |
| SSD busy | Avoid eviction of useful resident data; reduce speculative reads; prefer existing copies |
| Other workload ends | Gradually refill the most useful tier, measure the benefit and stop at reserve limits |

An example decision: a GPU kernel taking 0.4 ms plus 1.2 ms of exposed transfer is worse than a 1.0 ms CPU
path. If CPU contention stretches that same path to 3.0 ms, promotion may become worthwhile. These are
illustrative numbers, not measurements. The controller must learn actual costs for each inference phase.

## Current heuristic and its limits

The new `coadaptive` block is disabled when absent. `{"enabled": true}` defaults to `"mode": "shadow"`:
it samples and reports suggestions, but does not alter resource targets or routing. Live heuristic testing
requires explicit `"mode": "live"`, enabled live memory, an explicit `--pcie-frac`, and resource presets off.
An explicit per-request PCIe fraction still wins.

The heuristic treats CPU and GPU pressure independently, requires fresh readings, returns to baseline
after sustained quiet, and immediately cancels promotion into a newly pressured destination. GPU occupancy
while Strata is active is not attributed to another application; an attributable provider or recent idle
measurement is needed. Host commit pressure is included on Windows.

Its fixed thresholds and fractional routing adjustments are not calibrated. It changes routing only at
request boundaries and cannot yet reduce CPU worker count or follow changing load in a long generation.
Shadow mode is the default specifically because passing direction tests does not prove lower completion time.

## Local validation of the port

Machine: Ryzen 9 7940HS, RTX 4070 Laptop 8 GiB, 64 GiB installed DDR5; Windows; IQ3_S; 65,536 configured
context. This short test did not fill 64K, test vision inputs, run Hermes tools or simulate external workloads.

- CUDA Release build succeeded. Five native CTest targets passed, covering live memory, conversation cache,
  conversation memory/snapshot and serving-window behavior.
- Full server suite: 715 tests, six skipped, passed before the final shadow-default change. The final
  affected suite passed 96 tests, including shadow non-actuation and destination-pressure cancellation.
- First native load refused an unreachable 768 MiB reserve: only 506 MiB could be freed at its prompt-cache
  floor. This is a retained limitation, not a successful fallback.
- A second load with a 384 MiB reserve succeeded. The same native process answered a short arithmetic request
  before resizing, after shrinking resident RAM from about32 to30 GiB, and after restoring it to32 GiB.
- GPU cache moved 736 ->640 ->672 MiB; native free-VRAM reports were410,456 and424 MiB at load/resize
  acknowledgements. A later verification-window log reported380 MiB. This is sampled evidence, not a proof
  that every instantaneous allocation stayed above250 MiB.
- The one-second NVML watcher reported a minimum1059 MiB free, different from the engine's CUDA allocation
  headroom under WDDM. Do not use the NVML number as a substitute for the native allocator's smaller value.
- Sampled system available RAM remained at least11.03 GiB. The shrink and regrow took about7.06 and9.16 s.
  Requests after each operation retained37 prompt tokens, showing reuse survived this limited exercise.
- Acknowledgements included `ram_capacity_or_rounding` with `applied`: actual RAM was a few MiB below the
  requested target. Claims must use actual sizes, not requested budgets.

These are functional smoke results. Prompt reuse, changing output length, rounding and changed placement
make the three response durations unsuitable for a speed comparison. There is no demonstrated agentic
completion-time gain or complete adaptive-resource implementation yet.

## Acceptance gates before an upstream feature PR

Record a fixed baseline and run matched cold/warm requests with identical prompts, sampling, model and
context configuration. Then test CPU-only contention, GPU-only contention, RAM pressure, combined pressure,
storage contention and recovery independently before combinations. Start with bounded perturbations, not
an uncontrolled stress test of the user's applications.

Measure successful tool/task completion, total elapsed time including compilation, first-token delay,
prefill, decode and MTP acceptance, cache reuse, migration cost, queue delay and foreground job completion.
Use native device headroom plus OS physical/commit data. Include allocation failure, stale telemetry,
disconnect and rapid pressure reversal. Test actual long context and vision separately.

Only promote policies that preserve correctness and meet resource limits with repeatable end-to-end benefit.
First upstream work should be a reviewable compatibility/safety port and measurements; calibrated scheduling,
worker controls, multi-GPU topology and storage-aware policy can follow as separate contributions.
