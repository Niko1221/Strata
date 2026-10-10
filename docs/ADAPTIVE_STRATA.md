# Adaptive Strata: research and experimental groundwork

Status: isolated research branch, 2026-10-08. Not a released feature or a proven general speed improvement.
Production configuration is unchanged. The intended objective is correct agentic task completion in less
wall-clock time while leaving resources available to foreground applications.

The source port includes upstream **v0.1.42**, commit
[`61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`](https://github.com/Niko1221/Strata/commit/61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406).
Its queue heartbeats and CPU-assignment restart metadata are composed with adaptive admission and lease
ownership. Replica and helper-GPU modes remain unsupported for the adaptive controller and are rejected
before native or vision allocation. Ordinary replica serving remains available without adaptive ownership.
Live memory also requires `--adapt-async 0`: asynchronous expert swaps can still own buffers when a resize
needs them. The server rejects this combination before startup, matching the existing native guard.
The upstream route-tail-skip default is retained in source; matched quality comparisons must explicitly use
`STRATA_ROUTE_TAIL_SKIP=0`. Earlier validation below retains its original source and binary identities.
Results are not transferred to this revision; port validation is pending.

The vision teardown also adapts [weituotian's PR #1862](https://github.com/Niko1221/Strata/pull/1862):
close buffered stdin after confirmed process death so a failed `QUIT` does not flush again during garbage
collection. An encoder that cannot be confirmed stopped still retains its process and blocks replacement.
See [implementation provenance](ADAPTIVE_PROVENANCE.md) for the exact source and attribution boundary.

## Scope

Memory capacity and compute availability are separate constraints. A busy CPU should not automatically
cause GPU cache eviction. A busy GPU should not automatically evict useful RAM. The eventual controller
should choose among supported CPU and GPU execution paths, RAM and VRAM residency, and reads from backing
model files according to measured completion cost. Returning resources to Strata after another workload
finishes is part of the objective.

This branch currently provides live expert-cache resizing, cooperative background waits and pacing,
opt-in active text-request parking, opt-in pressure-triggered idle unloading with guarded reload admission,
and experimental single-GPU routing from matched measurements.
Uncalibrated routing keeps the configured split. These are separate controls with separate validation
boundaries; enabling one does not establish that every resource can be relocated or released.
It does not provide general tensor relocation, live CPU-worker resizing, multi-GPU balancing, SSD selection,
storage-aware routing or an online tensor cost model. Some allocations still have a fixed GPU minimum.

## Current background and request-continuation boundaries

The background-control work extends the earlier acceptance campaign recorded below. That campaign's
timings and test counts describe its named revisions, not the newer cooperative-prefill or request-parking
changes. Report new measurements against the exact tested source and binary; do not combine historical
counts into a claim that the current revision passed them all.

The native engine can yield at completed decode windows and cooperative prompt-chunk boundaries. Active
file readers and borrowed cache views must be drained before a cache is resized. During a background wait,
expert workers use their sleeping path, control heartbeats continue, and cancellation remains available.
The control lease is bounded: loss of the Python controller cannot permanently retain an old pause command.
This is cooperative scheduling, not arbitrary CUDA-kernel preemption, and the response time includes reaching
the next safe boundary. A lower CPU percentage alone is not evidence of a faster foreground application.

When both supported caches have reached their reclaimable floors, a separate, explicitly enabled
`coadaptive.request_parking` policy can suspend one active **text-only** request:

1. Sustained fresh hard-pressure observations request a native STOP at a committed boundary. The existing
   request owner drains all accepted output through the matching DONE acknowledgement; an error or an
   ambiguous end never authorizes replay.
2. A private journal records the exact native token prefix plus accepted output IDs, the effective sampling
   seed and the remaining output budget. The server retains the HTTP stream, output parser, tool-call IDs
   and FIFO ownership. It does not resend already delivered text or execute a tool action.
3. Only after the journal is committed does the owner unload the native/vision processes. This releases
   their process-owned execution allocations as well as their caches; memory held by the driver or other
   applications is outside that ownership.
4. The same request waits with heartbeats and cancellation for fresh physical-RAM, commit and GPU capacity
   sufficient for a full reload, including margins and a stable recovery interval. It then reloads and
   reprocesses the exact saved prefix before emitting only new output. Recognized pre-READY allocation
   failures have a bounded backoff/retry path after verified cleanup. Generation errors are not retried.

The journal is not a KV/recurrent-state snapshot. Reprocessing the prefix is intentional and can be costly;
preserving token IDs and a seed does not prove bit-identical subsequent generation after reload. Current
parking support is one running server and one active response, not crash recovery, reconnect/resume, parallel
serving or image-request migration. Image requests retain their ordinary vision handling and do not enter
this parking path. A configured vision encoder alone does not make a text request an image request.

Parking is disabled by default and requires a compatible live coadaptive/native-control configuration and
an explicit absolute journal directory. Windows journals require persistent filesystem ACLs and a protected
current-user/SYSTEM DACL. Disk-write or identity-validation failure cannot authorize destructive unloading.
Cancellation waits for lifecycle ownership to settle before releasing the request queue; it does not leave
an unowned reload thread running. The initial and post-STOP measured footprints are combined conservatively,
but sampled capacity still cannot exclude a new allocation racing the reload.

**Between requests, a separate opt-in `coadaptive.idle_parking` policy now implements full engine unloading
under sustained memory pressure.** The final revision passed a bounded real-pressure Hermes/tool-wait
acceptance; its [separate evidence packet](../bench/results/2026-10-08-idle-pressure) records the exact
source, binary and limits. Live MEMORY control can first return reclaimable
RAM/VRAM caches, while idle parking can release the native and vision processes and their fixed execution
allocations. It acts only with no active generation, queued request or HTTP preparation owner. CPU activity
alone is not a trigger. This makes a compiler-tool interval eligible when the agent has finished one model
call and has not yet submitted its next one; it does not infer the tool's type or special-case Hermes.

The monitor owns FIFO before capturing a fresh, conservative full-reload footprint and model identity,
then unloads the native and configured vision processes. There is no active response to journal or replay.
The next request reserves preparation before any load or vision encoding, waits for capacity while retaining
FIFO ownership, then reloads and processes its normal supplied history. OpenAI, Anthropic and Responses
streaming requests receive waiting heartbeats; disconnect or shutdown cancels admission and joins owned
lifecycle cleanup before releasing the queue. Stale or missing required readings do not authorize unloading
or a reload. A verified process exit is required before claiming that its allocations were returned.

Reload admission requires the measured full startup footprint plus reserves in physical RAM, Windows
commit and GPU memory. RAM and commit each use the largest of `min_ram_headroom_gib`,
`pressure_ram_available_gib` and `pressure_commit_available_gib`; GPU memory uses the larger of
`min_vram_headroom_mib` and `pressure_vram_free_mib`, also respecting the existing server load floor.
Capacity must stay sufficient through the recovery interval. This prevents the model's own unloading from
being counted as recovery while the competing workload still occupies the space needed to restart.
Pre-READY capacity failures use bounded retry/backoff after verified cleanup; arbitrary generation failures
are not automatically replayed.

The physical-RAM estimate retains the largest of current process working sets plus the arena restoration
delta, valid Windows working-set peaks, and the configured startup arena plus current vision working set;
it adds the configured overhead once and checks private commit separately.
These are sampled bounds, not allocation guarantees: without working-set peaks, OS-trimmed fixed allocations
may remain unmeasured even though the startup arena floor is retained.

Strata's existing timed idle-unload/autoload mechanism remains separate. When this new policy is enabled,
manual or timed unloading also captures a reload footprint and uses guarded admission next time. The policy
does not introduce the existing timer. Model weights evicted from RAM are read again from their existing
model files; no additional SSD model copy is written. Full idle unloading discards process-local cache state,
so later loading and prompt reprocessing can be expensive.

### Idle-parking configuration and limits

The `idle_parking` block is disabled when absent. These are its defaults; explicitly set `enabled` to `true`
only in a compatible experimental configuration:

```json
"coadaptive": {
  "enabled": true,
  "mode": "live",
  "idle_parking": {
    "enabled": false,
    "pressure_seconds": 10,
    "pressure_ram_available_gib": 3,
    "pressure_commit_available_gib": 3,
    "pressure_vram_free_mib": 250,
    "recovery_seconds": 10,
    "retry_seconds": 30,
    "min_ram_headroom_gib": 3,
    "min_vram_headroom_mib": 320,
    "max_sample_age_seconds": 5
  }
}
```

Idle parking requires live coadaptive and memory policies, resource presets disabled, an explicit
`--pcie-frac`, a single compatible stoppable native engine, and its `BACKGROUND` control capability.
Configuration currently requires the engine and any configured vision encoder to be loaded. It is
incompatible with initial lazy loading and with a `before_load` hook: an arbitrary hook's allocations and
side effects are not covered by measured reload admission. A journal `directory` is rejected for idle
parking; the separately enabled active-request path still requires its protected journal directory.

An image request can wait for admission before vision preparation and then follow normal image handling.
This does not add migration of an already-running image request. Initial startup still uses the existing
fit/load checks; idle parking is not a substitute for a measured first load. Thresholds must be consistent
with a feasible reload footprint, or requests can remain waiting until the competing workload releases
enough resources. Polling and the default ten-second pressure interval mean the response is not immediate.

Status is available under `memory_policy.idle_parking`: `disabled`, `ready`, `unloading`, `suspended`,
`admitting`, `reloading` or `unavailable`, with the captured footprint, reason and admission result when
applicable. `memory_policy.preparing_requests` reports preparation ownership. In unloaded state the public
metrics omit stale process-allocation counters while retaining internal capabilities and model identity.
This reporting detail follows the stale-counter observation in
[miskahm's PR #1093](https://github.com/Niko1221/Strata/pull/1093); its monitor UI and slider code are not imported.

The final acceptance verified native and vision process replacement, capacity waiting and cancellation,
then normal completion in the same Hermes session after actual compiler feedback. The four-job C++ build
passed 2,096 cases, with 104 independent checks. Agent wall time was 190.442 seconds; minimum sampled
headroom was 5.623 GiB RAM and 294 MiB native/DXGI VRAM. A separate real 3 GiB holder and deliberately
extended tool interval supplied controlled pressure. The 2.296-second build finished before unloading,
so this is a continuity test, not a compilation speedup. Final HTTP vision/tool checks also passed.
The final Python suite ran 893 tests: 886 passed and seven skipped. Five targeted native tests passed.

Separate short static-profile probes showed that a 20 GiB resident cap used about 12.15 GiB less peak
process RAM than 32 GiB and still passed arithmetic, vision and tool continuation, with slower replies.
Those probes used an earlier server revision, retain their own hashes and sensor limitations, and do not
qualify near-64K reload admission. Automatic smaller-profile selection is not implemented; the current
idle policy waits for its full measured startup footprint. The evidence packet preserves this boundary.

There remains a finite working-set floor whenever inference is actually running. If no supported execution
shape fits, the safe options are waiting or parking, not treating nominal RAM, VRAM and SSD sizes as one
interchangeable pool. An application can allocate faster than a sampled controller responds; disk errors,
device loss, kernel faults, insufficient reload capacity and client/proxy timeouts remain possible. This
prototype makes no unconditional "never OOM," "never stop," immediate adaptation or speed guarantee.
Acceptance must establish correct task completion, bounded resource release and foreground progress for
the tested workload, including the time spent waiting, unloading, loading and reprocessing the prompt.

See [implementation provenance](ADAPTIVE_PROVENANCE.md) for actual inherited code, design inspiration and
API references. The broader research survey below is not a list of imported implementations.

## Existing Strata work and attribution

The live allocator, MEMORY protocol, acknowledgement handling, RAM blocks, GPU VMM resizing and resource
presets are based on medking82's [PR #726](https://github.com/Niko1221/Strata/pull/726), ported onto
v0.1.40.3, then carried forward with the upstream v0.1.40.4 update above. Its author is retained in Git history.
That PR was closed without merging during history cleanup;
closure should not be described as a technical rejection.

Other related work must be considered before an upstream submission:

- [PR #1117](https://github.com/Niko1221/Strata/pull/1117): automatic GPU cache resizing, excluding the
  resident CPU complement. Its code is not part of this branch's current implementation.
- [PR #1324](https://github.com/Niko1221/Strata/pull/1324): elastic RAM LRU and file fallback, with Windows
  commit pressure. Its code is not part of the current implementation; both changes need a common policy owner.
- [PR #1461](https://github.com/Niko1221/Strata/pull/1461): peer-tier cooperation and changing roles across
  two GPUs. Relevant to a future multi-GPU design, not validated on this laptop.
- [PR #1471](https://github.com/Niko1221/Strata/pull/1471): offload idle session state through CUDA VMM while
  keeping model weights loaded. It does not park an active HTTP request or release the entire engine.
- [PR #1480](https://github.com/Niko1221/Strata/pull/1480), stacked on
  [#1271](https://github.com/Niko1221/Strata/pull/1271) and
  [#1269](https://github.com/Niko1221/Strata/pull/1269): disk conversation caches and streaming session
  restore. These preserve reusable model state; the current request journal instead reprocesses exact token
  history after unloading. Integrating a proven streaming snapshot could reduce that cost later, but none
  of these patches is imported here.

These related feature PRs (#1117, #1324, #1461, #1471, #1480 and its underlying spill/restore work)
were still open at the earlier 2026-10-08 duplication check, when the release base was v0.1.40.3.
The branch has since incorporated v0.1.40.4. Searches for open
elasticity, pressure and background-control proposals did not identify an equivalent active-request
journal/unload/reload controller. This is a scoped duplication check, not a claim that the broader idea
is novel. Recheck heads and maintainer feedback before publication. PR #1324's author explicitly distinguishes
gradual pressure response from a guarantee, and the PR #1117 author offers to separate its arena from policy.

Changes made during this port include draining newer asynchronous file readers before changing resident
storage, considering available Windows commit as well as physical RAM, updating server test fixtures for
current slot state, and using application-neutral workload signals. These are port fixes, not techniques
copied from the external research below.

## External work evaluated

This is a research survey, not a list of imported implementations. The current routing gate uses
principle-level inspiration from ATSInfer and StarPU; the other projects below were evaluated as related
work. [Implementation provenance](ADAPTIVE_PROVENANCE.md) identifies actual code reuse, design influence
and API references separately, including the substantial Strata PR #726 port.

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

No external code from the projects in this research survey has been incorporated. Only the ATSInfer and
StarPU principles identified above currently inform the routing implementation. Review licensing
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

## Current controller and its limits

The new `coadaptive` block is disabled when absent. `{"enabled": true}` defaults to `"mode": "shadow"`:
it samples and reports suggestions, but does not alter resource targets or routing. Live testing
requires explicit `"mode": "live"`, enabled live memory, an explicit `--pcie-frac`, and resource presets off.
An explicit per-request PCIe fraction still wins.

The capacity controller treats CPU and GPU pressure independently, requires fresh readings, returns to baseline
after sustained quiet, and immediately cancels promotion into a newly pressured destination. GPU occupancy
while Strata is active is not attributed to another application; an attributable provider or recent idle
measurement is needed. Host commit pressure is included on Windows.

Capacity thresholds are conservative controls, not a speed model. Native `CAPACITY` messages report CUDA
allocator headroom, resident RAM and GPU cache size at existing host safe points, at most once per second.
Readings expire after five seconds and belong to one native process. Missing native capacity freezes
allocation decisions instead of falling back to a more optimistic NVML number. These are sampled readings;
they cannot guarantee an instantaneous reserve against another application's sudden allocation.

Startup fitting reserve and running headroom have separate meanings. Optional `min_free_vram_mib` in
`coadaptive` sets the latter (at least 256 MiB); the local experiment uses 320 MiB and a 448 MiB startup
reserve. This extra room covers observed lazy allocations. Native acknowledgements, not requested sizes,
determine actual allocations. An unchanged, unreachable prompt-cache floor does not repeatedly stall
inference with the same request. RAM relief remains possible independently. Recovery includes the final
partial two-GiB RAM step.

`serve/routing_costs.py` applies the measured-cost principle from ATSInfer sections 4.3-4.4 and StarPU,
without copying their source or tensor scheduling algorithms. Optional `routing_profile` points to a
schema-1 calibration file with validated matched samples. The fingerprint includes the engine binary,
launch arguments, model asset metadata, relevant environment, CPU/GPU identity and OS. This is an
invalidation key, not proof that the model's entire contents were hashed.

Routing requires fresh attributable external CPU/GPU readings, matching RAM/cache allocations and a
calibrated context range. At least three common pairs must each beat baseline by more than five percent
and ten milliseconds, plus any supplied switch penalty. The measured wall time already includes exposed
transfers and file waits; do not add those same intervals again. Profiles require a creation timestamp and
expire after one day. Sampling settings and prompt-read/output bounds must also match; a warm decode
sample cannot authorize a cold or longer request. The current server conservatively assumes the entire
input may need reading; it does not guess which recurrent/prefix snapshot will be reusable.
Wrong identity, noise, stale readings, changed capacity or an uncovered workload keeps the
configured split. No background prompts or autonomous calibration are submitted.

The current integration decides after the request queue, before generation. Explicit request settings
win, vision requests retain their configured route, and parallel serving is excluded by live memory.
It cannot yet adjust CPU workers or routing within a long generation. The small local decode calibration
does not predict reasoning length or complete agent tasks; a faster token rate does not establish shorter
successful completion. No local alternative has passed the conservative routing gate so far.

Example additions to a compatible single-GPU configuration:

```json
"memory_policy": {
  "enabled": true, "mode": "live", "min_ram_headroom_gib": 4,
  "pressure_seconds": 4, "recovery_seconds": 30
},
"coadaptive": {
  "enabled": true, "mode": "shadow", "min_free_vram_mib": 320
}
```

Change the latter mode to `live` only for an isolated validation. Leave `routing_profile` absent until
there is trustworthy calibration for that exact runtime. The RAM cap and startup reserve still belong
in the native arguments. Shadow mode is the default because direction tests do not prove faster tasks.

## Cooperative-control and parking validation before the v0.1.40.4 update (2026-10-08)

That Python suite ran **851 tests: 844 passed and seven skipped**. The native build and five focused
native control/memory tests passed. These counts concern the earlier cooperative-control campaign,
separately from the older campaign below and the new idle-parking implementation. The real parking run used native binary SHA-256
`b17c3ef7a55ff1b00e217fb5a5251a23583734b2ce6e2d57d1423012215d83aa` and server source SHA-256
`f714e58627696099ecda6969307c066fee597a6b28412d218805045a98ea0e45`.

On Windows with a Ryzen 9 7940HS, RTX 4070 Laptop 8 GiB and 64 GiB installed RAM, three real-engine
service-level streaming cases passed: uninterrupted control, park/resume, and cancellation while parked.
The model was IQ3_S with 65,536 configured context; this was a short text prompt, not a filled-64K parking
test. Sampling was greedy with seed 42, high reasoning, an 8,192-token reasoning allowance and a
12,288-token total output limit.

The parking trigger was an **injected sustained critical-pressure decision**. The harness did not consume
the machine's remaining memory or invent free-capacity readings. Process unloading, observed RAM/VRAM
release, capacity admission, reload and continuation were real. A deliberate roughly 20-second hold kept
the request suspended so heartbeat and cancellation behavior could be checked. This validates the lifecycle
under a controlled trigger; it does not establish response time to an actual foreground allocation burst.

| Case | Outcome | Request elapsed | Observed system RAM made available | Observed global GPU free-memory increase |
| --- | --- | ---: | ---: | ---: |
| Uninterrupted control | Natural completion; generated C++ passed 104 independent cases | 64.781 s | Not unloaded | Not unloaded |
| Park, unload and resume | Natural completion; generated C++ passed 104 independent cases | 116.098 s | 37.348 GiB | 6,802.715 MiB |
| Cancel while parked | Cancelled without reloading | 28.861 s including the hold | 37.442 GiB | 6,836.715 MiB |

Request times exclude the subsequent independent compile/test check (about one second for each completed
answer). The resumed request took longer, including deliberate suspension, model reload and prefix processing;
this experiment demonstrates resource release and correct continuation, not a speed improvement.
The control generated 1,094 tokens and the resumed case 740; their difference in elapsed time is therefore
not an isolated measurement of suspension overhead, and their future continuations were not bit-identical.
The release figures are differences in system/global readings around the verified process exit, not an exact
accounting of every driver allocation. Native allocator and DXGI budget headroom were checked separately.

The continuation check compared exact accepted token IDs, effective seed and remaining budget across the
park boundary, and verified a new native process after the old one exited. A requested Unicode marker appeared
once in each completed answer. Heartbeats remained available during suspension. The cancellation acknowledgement
arrived **0.037 seconds** after cancellation was requested, with no reload. This is one controlled measurement,
not a worst-case cancellation bound during every startup or I/O operation.

Sampled native/DXGI GPU headroom stayed at least **326 MiB** in control/resume and **344 MiB** in the
cancellation case. Sampled available RAM stayed at least **10.59 GiB** across these cases. Samples cannot
exclude a brief unobserved allocation peak. Image-input parking, exact future-output equivalence after a
reload, client reconnect and real uncontrolled memory exhaustion were not tested by this campaign.

### Background pacing with real CPU contention

A separate off/on/on/off comparison used the same 94-token prompt, high reasoning and a 4,096-token
output limit. The foreground load was a fixed eight-worker C++ workload totaling 16 billion iterations,
not a compiler invoked by Hermes. Model inference and the foreground job overlapped for 43.00-43.85 seconds
in every arm. Expert RAM stayed at 32,767 MiB and GPU expert cache at 672 MiB; this tests pacing with fixed
caches, not every part of the adaptive memory policy.

| Arm | Foreground elapsed | Model request elapsed | Generated tokens | Decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Off, first | 45.52 s | 114.27 s | 1,742 | 15.59 |
| On, first | 45.84 s | 64.59 s | 785 | 12.64 |
| On, second | 45.79 s | 66.60 s | 873 | 13.62 |
| Off, second | 44.99 s | 126.76 s | 1,927 | 15.58 |

All four generated C++ answers reached natural completion, compiled and passed 104 deterministic cases.
All eight foreground result checksums matched across all four arms. External CPU readings were complete
throughout the sampled overlap. The on arms requested positive delays, reaching 10 and 14 ms; off arms
requested none. These are observed controller commands, not measured counts or durations of native sleeps.

Median foreground time was **45.25 seconds off versus 45.81 seconds on**, so this trial shows no foreground
speed gain. The on arms produced roughly half as many tokens and had **lower** decode throughput. Their
shorter model request times therefore cannot be attributed to a throughput optimization. With two samples
per arm and varying generated answers, this is bounded correctness and control evidence, not a causal
completion-time improvement. Sampled available RAM remained at least **10.658 GiB**; fresh native CUDA and
DXGI budget headroom each remained at least **324 MiB**. These are separate measurements whose minima happened
to match, and still do not exclude brief unsampled peaks.

### Current HTTP vision and tool acceptance

The normal HTTP interface passed an image request and a function-call/result continuation with request
parking enabled. The image followed the ordinary vision path and explicitly reported parking unsupported
for that request; it was not unloaded or migrated. The image request took 12.972 seconds. The complete
two-request tool cycle took 13.566 seconds and returned the computed result correctly. Its continuation
reused 435 of 453 input tokens, reading 18 new tokens. These checks establish compatible ordinary handling,
not image parking or a speed comparison. Sampled available RAM stayed at least **9.465 GiB** and native
GPU headroom at least **340 MiB**.

### Actual Hermes compiler-feedback acceptance: first attempt incomplete

A visible, isolated Hermes CLI session was asked to read an existing C++ function, run its failing build,
repair the function, then compile and test again. The four-job compiler ran and its real failure diagnostics
reached Hermes. Hermes repaired the code but reached its six-turn ceiling before completing the required
final build and verification. The attempt took **311.219 seconds**, made **eight recorded API calls**, and
ended unsuccessfully. It did not receive a successful build result in its own conversation.

An independent check of the resulting function compiled and passed all **104 cases**; immutable test/build
fixtures remained unchanged. That proves the repaired candidate passed this test, but does not convert the
agent's incomplete workflow into a successful end-to-end run. Sampled available RAM stayed at least
**7.856 GiB** and native GPU headroom at least **278 MiB**, above the 250 MiB guard. The production Hermes
configuration hash was unchanged. This was one bounded attempt with no matched nonadaptive Hermes control,
so neither faster agent completion nor a general autonomy claim is supported.

The **same Hermes session subsequently completed** after its disposable profile's turn limit was raised
from six to twelve. The full prior transcript and repaired candidate were retained; the continuation did
not supply replacement code. Hermes ran the four-job compiler, received its successful result of **2,096
deterministic cases**, and completed normally. An independent check again passed all **104 cases**. The
continuation took **203.183 seconds** and made three recorded API calls. Including the unsuccessful first
attempt, the combined agent time was **514.401 seconds** and eleven recorded API calls; intervening setup
and idle time are not included in that sum.

The continuation's sampled minimum available RAM was **8.352 GiB**; native CUDA and DXGI budget headroom each
reached **276 MiB**. Original evidence, prior build results, immutable project fixtures and the production
Hermes configuration remained unchanged. This is bounded success after an explicit same-session resume,
not an uninterrupted autonomous success or a matched demonstration of faster completion. It also does not
establish a foreground compiler speed gain: these ordinary short compiler jobs did not impose controlled
memory pressure.

### Near-64K prompt with real RAM pressure

A separate current-build test processed a **60,270-token prompt** in the configured 65,536-token context.
Unlike the synthetic parking trigger above, this test allocated and held **2 GiB of real competing RAM**
during prompt processing. The resident expert arena shrank from 32,767 to 30,404 MiB in **7.156 seconds**,
while prompt processing was still at 768 of 60,270 tokens. The native process stayed alive throughout.
This exercises cooperative prefill relief, not full process parking.

| Request | Request elapsed | Prompt processing | Reused / newly read input tokens | Generated tokens | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Cold long prompt under RAM pressure | 839.035 s | 829.5252 s | 0 / 60,270 | 140 | All four requested keys correct |
| Repeated long prompt with pressure still held | 10.940 s | 0.5859 s | 60,265 / 5 | 140 | All four requested keys correct |
| Different conversation after recovery | 5.739 s | 2.4995 s | 0 / 49 | 78 | Correct answer |

The cold and repeated request times exclude the initial 40.039-second model load. The repeated request
demonstrates preserved prompt reuse across live memory changes; it is not a cold-prefill speed comparison.
After the competing allocation was released, the resident arena recovered to 32,707 MiB, within 64 MiB of
its initial size. The different-conversation check then completed correctly in the same process.

Across 982 telemetry samples, available system RAM reached a minimum of **8.805 GiB**. Of these samples,
943 contained native capacity information no more than five seconds old, representing 351 distinct native
updates. Fresh native CUDA and DXGI budget headroom each reached a minimum of **294 MiB**, above the requested
250 MiB guard. Sampling still cannot exclude shorter unobserved peaks. This test supports long-prompt
correctness, relief during prefill and cache reuse; it does not show unrestricted memory safety, a response
bound for arbitrary competing applications, or a speed gain against older tests with different memory
behavior.

## Initial local validation of the port

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

## Earlier same-branch measurements (2026-10-08)

Same laptop/model, 65,536 configured context, seven CPU workers, MTP four, IQ3_S, existing projection
and vision configuration. Isolated live-memory engine: 32 GiB resident cap, 448 MiB startup fitting reserve,
320 MiB runtime headroom target. These are not a before/after comparison against the daily 36 GiB runtime.

Automatic RAM control, using real physical/commit readings and changing the desired free-RAM target,
released 32 -> 30.488 GiB in 13.03 seconds and recovered to 31.999 GiB in 39.17 seconds, including the
stability dwell. The same process answered correctly before and after both changes. This avoids allocating
a dangerous competing RAM load, but does not substitute for a long foreground-application pressure test.
The first harness used a one-second poll and repeatedly saw duplicate sensor samples; the normal service
cadence of two seconds passed. Stale/replayed samples still cannot earn a stability window.

An exploratory sweep used 24 warm 128-token decode measurements at `pcie_frac` 0.15, 0.37, 0.65 and 0.85,
both idle and under a four-thread external CPU load. No alternative cleared the conservative matched
decode-cost gate. One 0.15 CPU-load quality check exhausted a 1,024-token reasoning budget without final
code, so those timing rows were excluded. Some initial idle samples overlapped a unit suite; this sweep
is exploratory and was not used to change defaults.

A separate comparison, with no concurrent build/test suite, alternated 0.37 and 0.65 for three repetitions
per condition. Each response implemented a C++20 `unique_sorted` function and had to compile and pass
104 executable cases (empty input, negatives, duplicates, integer extremes and deterministic random data).
High thinking remained enabled, with greedy sampling, seed 42 and a 2,048-token output limit.

| External load | PCIe fraction | Correct runs | Median complete task (range), seconds | Median decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Idle | 0.37 | 3/3 | 38.24 (31.51-48.55) | 17.84 |
| Idle | 0.65 | 3/3 | 46.44 (41.06-97.05) | 18.09 |
| Four CPU threads | 0.37 | 3/3 | 55.10 (39.23-69.82) | 17.02 |
| Four CPU threads | 0.65 | 3/3 | 42.71 (35.73-43.55) | 17.19 |

Times include generation, compilation and execution. Only the first idle 0.37 request was cold (1.594 s
prompt processing; the other idle prompts took about 0.23-0.26 s). Three samples of one small function
cannot establish a general code-agent win. Output length varied substantially despite identical sampling:
the loaded-CPU medians favored 0.65 by about 22.5%, while idle medians were about 21.4% worse. This is a
workload-dependent lead, not proof of a scheduling improvement or bit-exact generation.

Sampled native free VRAM stayed at least 324 MiB during the repeated task comparison and available RAM
at least 12.39 GiB. The exploratory sweep's minima were 326 MiB and 10.39 GiB. Native samples can miss a
short allocation peak; these are not instantaneous guarantees. No foreground game, filled 64K context or
long Hermes project was tested. The interactive experiment keeps 0.37 and omits the warm-decode profile.

The full server suite ran 736 tests with six skips before the final GPU-floor recovery correction. The
final affected suite passed 126 tests; five native memory/conversation tests also passed. The GPU test
found that a known `prefill_cache_floor` error incorrectly delayed a later lower target by ten minutes.
Known floor errors now suppress repeated impossible requests while allowing normal debounced recovery;
generic allocation errors retain their backoff. The real-model retest released GPU cache from 672 to
640 MiB in 9.05 seconds, then recovered to 704 MiB in 34.13 seconds including dwell. It reported a
deliberately unreachable target in 9.03 seconds and held the request count steady for another 16 seconds.
The same process continued answering correctly at each stage. This test changed the requested headroom;
it did not allocate a competing GPU workload. Native free VRAM was sampled at no less than 392 MiB.

An isolated normal HTTP server then passed a real image request (red square) and a tool-call/result
continuation. It advertised 65,536 context and vision enabled, with high reasoning defaults; sampled
native free VRAM stayed at least 342 MiB and available RAM at least 11.56 GiB. This is API acceptance,
not a completed Hermes project. All test processes were stopped afterward; production was not modified.

## Earlier completed acceptance campaign (2026-10-08)

Decision: retain this as an isolated prototype. Do not promote it to the daily launcher or submit the
whole branch as a finished adaptive scheduler. The tests demonstrate useful memory-control behavior,
but not repeatably faster successful agent tasks. The earlier measurements above are retained as history;
they are not the unchanged-upstream comparison below.

### Method and final controller corrections

The unchanged baseline is v0.1.40.3, commit `d5ea713`. Both native binaries were built with the same
MSVC/CUDA toolchain, Release settings and CUDA architecture 89. Both used portable ggml plus native
expert kernels. The latter were compiled with `/arch:AVX512`; the portable ggml AVX-512 cache switches
do not describe the separate native IQ3_S kernels. Seven pinned pool workers plus the calling thread
participate across eight physical cores. Generated C++ test programs used ordinary `-O2`, without an
AVX-512 requirement.

The common model settings were IQ3_S, 65,536 context, high reasoning, MTP four and the configured
projection/vision assets. The isolated experiment used a 32 GiB resident cap, 448 MiB startup fitting
reserve, 320 MiB running free-VRAM target and fixed PCIe fraction 0.37. Routing remained uncalibrated;
no alternate placement passed the routing gate. Tests ran sequentially on a separate service port.
These are not comparisons against the daily launcher with a different resident budget.

Final server corrections for the earlier campaign:

- Send an absolute running free-VRAM target to native `MEMORY`; do not compound a fitting reserve with
  each observed shortfall. A renewed shortfall can require action even when the target is unchanged.
- Keep RAM relief independent of an inflated GPU reserve, and retain bounded GPU recovery.
- Do not interpret low free VRAM by itself as another application's GPU compute load. Strata's own
  prompt workspace can consume headroom without a competing application.

Seven added regression tests cover these cases. The final full Python suite ran 745 tests: 739 passed
and six skipped. Nine selected native tests passed: conversation cache, conversation memory, live memory,
conversation snapshot, serving window, file expert source, VMM, expert profile save and platform memory.
This is not a claim that every native test in the repository ran.

### Completed tasks, including failures

The C++ test requested a `unique_sorted` function, compiled its answer and checked 104 executable cases.
Sampling was greedy with seed 42 and high reasoning. Wall time includes inference, compilation and tests.
The 2,048-token campaign used reverse load order (ABBA), two requests per load, four attempts per arm.
The 4,096-token qualification used only two attempts per arm and was not a balanced timing study.

| Total output limit | Build | Accepted attempts | Median complete task | Median decode tokens/s |
| --- | --- | ---: | ---: | ---: |
| 2,048 | Unchanged upstream | 4/4 | 79.93 s | 18.10 |
| 2,048 | Adaptive prototype | 2/4 | Not reported: two attempts failed | 19.24 |
| 4,096 | Unchanged upstream | 2/2 | 49.06 s | 19.09 |
| 4,096 | Adaptive prototype | 2/2 | 65.83 s | 20.04 |

Both failed adaptive attempts exhausted the 2,048-token limit while reasoning and supplied no final
function. All follow-up answers used fewer than 2,048 tokens, so the later successes do not demonstrate
that raising the limit fixed the failures. Output length and placement varied. The small follow-up's
adaptive median was 34.2% slower despite a higher decode rate; it does not isolate a causal scheduler
regression, but it rules out claiming a demonstrated completion-time gain. Do not discard failed attempts
or compare only their successful subset. The first harness accidentally returned zero despite row failures;
its saved per-attempt results are authoritative, and future invocations now return failure if any row fails.

An earlier diagnostic comparison used eight capped 192-token requests per arm. Median request times were
7.938 s upstream, 8.060 s with the prototype controller off, and 7.786 s with it enabled. Those capped outputs
were not completed tasks and preceded the final server corrections. A roughly 1.9% median wall difference
is not a qualified speed claim. Adaptive load took 36.36-38.96 s versus 12.26-12.87 s upstream: about three
times as long in these two loads. Changed cache placement also prevents a bit-exact comparison claim.

### Real memory pressure and long conversation reuse

A helper actually allocated and touched 2 GiB of host RAM. The same native process reduced resident
experts from 32 to 30.488 GiB in 19.61 s and recovered in 42.27 s, including the stability dwell. Arithmetic
requests remained correct before, during and after resizing. Available RAM remained at least 9.676 GiB.
An elevated free-RAM target let this exercise relief without exhausting the machine.

The 64K service then read an actual 60,270-token prompt containing four distributed lookup keys. All four
were correct. Cold processing took 663.17 s overall, including 655.41 s of prompt processing. Repeating
the prompt took 8.02 s overall and 0.581 s of prompt processing, reusing 60,265 tokens and reading five.
A new-conversation check also passed. This validates long-prefix reuse in this test, not a new improvement
over upstream caching. The run included the absolute-reserve correction but preceded the capacity-versus-
compute correction; a 0.7-second focused unit check overlapped the long read, so these are not strict A/B
performance measurements. Sampled native free VRAM stayed at least 294 MiB.

A separate combined test held both 2 GiB host RAM and a 48 MiB external CUDA allocation. Answers remained
correct; RAM shrank in 14.08 s and recovered in 41.17 s. Native free VRAM stayed at least 424 MiB, above
the configured target, so GPU reclamation was not exercised. A first harness incorrectly required eviction
without an observed deficit and timed out; that attempt is retained as inconclusive rather than an engine
failure or a successful GPU-pressure demonstration. A separate CUDA observer reported about 7,020 MiB
free while the native engine reported 424 MiB. It cannot substitute for native feedback.

CUDA documents free memory as an OS estimate and does not guarantee that all reported bytes can be
allocated. Current-context and concurrent-allocation limits matter; Windows WDDM also virtualizes GPU
memory. See [CUDA memory information](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html)
and [WDDM GPU virtual memory](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-virtual-memory-in-wddm-2-0).
The mismatch above is a local observation, not a calibrated conversion between those interfaces.

### Actual Hermes and HTTP acceptance

A visible Hermes CLI session used a disposable project, high reasoning, 64K and one synchronous child.
It repaired a C++ interval-merging implementation, then independently compiled and passed 506 external
acceptance cases; the visible test file's hash was unchanged. Agent wall time was 690.16 s, with 14 API
calls. One child review took 344.24 s and reached its six-turn ceiling, but returned a review; the parent
completed and retested. This is a bounded success, not evidence of unrestricted autonomy. There was no
matched upstream Hermes run, so no agentic speedup can be claimed. Consecutive parent turns reused
roughly 93-98% of their prompts; parent/child transitions preserved snapshots.

The test's YAML `agent.max_tokens: 4096` did not establish an effective total output cap in the installed
Hermes CLI. Requests used the service's high-reasoning budget of 8,192 and the remaining context budget.
The explicit native 4,096-token comparison above is separate. Installed Hermes was
`0.21.5+8915.gc538ec5.dirty`; its existing local source changes were retained, not upgraded during this test.
Sampled native free VRAM was at least 310 MiB and available RAM at least 8.207 GiB.

After the final corrections, the normal HTTP interface passed a real red-square image request and a
function-call/tool-result continuation. It advertised 65,536 context and vision. Sampled native free VRAM
was at least 342 MiB and available RAM at least 10.585 GiB. These samples cannot rule out brief unobserved
peaks or protect against arbitrary new allocations by other applications.

Early diagnostic attempts with insufficient startup headroom were stopped and excluded before accepted
comparisons. They must not be represented as meeting the reserve requirement. Accepted runs used native
admission checks with extra room for lazy buffers; the minimum observed across the accepted pressure and
agent/API validations was 294 MiB. All test processes were stopped afterwards. Production Strata source,
settings and launchers were not modified.

### Release gates recorded at that campaign's revision

The first item below describes the earlier prefill implementation. Cooperative prompt-chunk relief and
active-request parking were implemented afterward; their present boundaries are described near the top of
this document. Implementation alone does not close the real-model correctness and latency validation gate.
The historical campaign and its failed attempts remain evidence for their tested revision.

1. Live cache changes currently wait for prompt processing to finish, because borrowed cache views remain
   in use. The approximately eleven-minute cold read demonstrates how long relief may be deferred. Safe,
   bounded prompt-chunk boundaries need a lifetime/reader-drain design and correctness tests.
2. Actual external GPU pressure and recovery remain unqualified on this Windows setup. Concurrent small
   allocations are not enough; verify both a real observed deficit and native cache shrink without crossing
   the safety floor. An instantaneous reserve cannot be guaranteed against arbitrary third-party allocations.
3. Preserve or improve complete-task acceptance on a larger matched task set, and account for reasoning
   length, compilation and foreground workload completion. The current C++ comparison does not pass this gate.
4. Resolve overlap with PRs #1117 and #1324 and reduce the contribution to a reviewable scope. A possible first
   contribution is native capacity/acknowledgement and allocation-lifetime hardening, followed separately by
   policy changes. Recheck upstream before porting; fixes to this prototype are not automatically upstream bugs.
5. Profile the approximately threefold startup cost. Do not describe the prototype as free adaptation.

This remains expert-cache elasticity and guarded request-boundary routing, not unrestricted VRAM-to-RAM-
to-SSD relocation. Fixed execution buffers, conversation state, multi-GPU roles, CPU worker counts, storage
topology and calibrated foreground-aware scheduling still require distinct work. No public PR or comment
was submitted for this acceptance campaign; original-code attribution and design references are retained.

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
