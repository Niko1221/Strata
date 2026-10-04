# Live expert cache capacity

This experimental mode changes the space available to expert weights without unloading the model. It does
not change model weights, quantization, context length or the attention/KV allocation. The default allocation
and request-boundary reload policy remain available.

## Enabling it

Keep the existing `--resident-budget-gib` cap and `--vram-reserve-mib` floor in the model configuration's `args`.
Add this top-level setting and restart the server while idle:

```json
"memory_policy": {"enabled": true, "mode": "live"}
```

Use a locally built engine with live-memory support. The server adds `--live-memory` before startup and checks
`INFO live_memory=1 memory_protocol=1`. The engine validates the hardware and cache configuration; it does not
silently substitute another allocation strategy. This implementation supports the single-GPU CUDA, file-backed
native-expert serving path with an expert profile and graphed residency. HIP, multi-GPU, peer caches and the
other layouts are excluded.

The existing percentage ceilings, RAM headroom, reserve floor and debounce/cooldown settings still apply.
The policy can observe pressure while an answer is running. Resizing runs only when native readers and GPU
work have reached a safe boundary. A large change takes several steps and may briefly pause generation.
Idle unload is independent: set `idle_unload_s` according to how long the model should remain loaded.

## What stays stable

VRAM uses a reserved virtual address range with independently mapped physical blocks. Slot addresses and
offset metadata retain their addresses while capacity changes. Shrink publishes reduced residency before
releasing mappings; grow fills new slots before publishing them. Existing CUDA graphs keep their base pointers.

RAM experts occupy independently releasable pinned or pageable blocks. Pinning uses the existing host-memory
budget limit; blocks fall back to ordinary memory when that limit is reached or the driver refuses pinning.
The startup log reports each amount. Their current owner follows adaptive expert exchanges;
resizing uses live residency rather than the startup profile's original placement. Experts removed from RAM
remain available from the unchanged model files. All background readers and pending copies are drained before
storage is retired. With prompt borrowing enabled, profile-ranked GPU experts may also have RAM copies;
those copies share the existing resident budget. The loan log reports RAM coverage and file fallback.
Before borrowing overwrites a missing RAM copy, live mode can retain the current GPU bytes in an existing
same-layer RAM slot using exchange scratch. All readers are drained first. Current borrowers cannot be donors;
GPU-backed duplicates are preferred, followed by the coldest measured RAM occupant. Without a donor or scratch,
immutable-file fallback remains available. Capacity stays unchanged; evicting a RAM-only donor can increase
later decode reads, so fewer refill reads alone do not establish a speedup.
Prefill temporarily borrows the active cache tail and refills the current experts before decode. A control
received during borrowed prefill waits until the entire prompt/refill has completed, including on STOP.
Before shrink and after growth, all borrowed views are rebound to current mapped slots. Chunk sizes shrink
and recover with the available cache, within the initial host-buffer allocation.
After all prompt and verifier buffers are allocated, startup checks actual free VRAM and trims expert mappings
to preserve the configured reserve before reporting READY. If that reserve cannot be met, startup fails.

Borrowing keeps enough mapped slots for a 256-token prompt chunk plus 128 decode slots. If a requested reserve
would need to retire this floor, the terminal acknowledgement reports `error=prefill_cache_floor` with actual
committed sizes. Fresh prompts can still run using the smaller chunk. Explicit `--no-prefill-borrow` retains
owned prompt buffers and the earlier zero-cache behavior.

The engine process, generation session, KV cache, speculative state and parked conversations remain intact.
Physical allocation still has limits: fixed model buffers and KV storage cannot be reclaimed by this feature.
Larger supported models may benefit from cache resizing, but this mode adds no new model architecture support
and makes no throughput guarantee.

## Internal control protocol

The server serializes control commands with generation, stop and quit commands on the engine's private stdin:

```text
MEMORY <positive-request-id> <resident-mib> <vram-reserve-mib>
```

Only one change is in flight. Native validates integer fields, rejects another request while busy and clamps
the reserve to at least 256 MiB. The server additionally enforces its configured RAM cap and reserve floor.
The native loop checks for work at 100 ms intervals or the next generation safe point. A step adjusts at most
128 MiB of expert VRAM and one 32 MiB RAM block, subject to allocation granularity and available experts.

Acknowledgements include the request ID, `status=progress|applied|error`, `resident_mib`, `expert_cache_mib`,
`expert_slots`, `vram_free_mib`, `vram_reserve_mib` and an `error` reason. `applied` means the operation reached
its bounded result; rounding or exhausted capacity can leave less resident RAM than requested. The server
records actual sizes, not the target, and exposes a separate `limitation` for completed but capacity-limited
allocations. It rejects stale acknowledgements from a different engine process or
request. Control lines never enter the generated-token stream.

Completed steps remain committed when a later step fails. The status exposes the resulting actual sizes and
error, and the policy backs off. Allocation refusal does not trigger a model reload. A fatal GPU/driver error
can still end the process, as with ordinary inference; this protocol is not device-failure recovery.

A request that shrinks RAM or raises the VRAM reserve cannot grow the GPU cache midway through the operation.
RAM release takes precedence. For an admitted growth request, a refused GPU increment is rolled back and
further GPU growth is capped for that request; permitted RAM work can finish. If the reserve is met, status
reports the completed partial capacity with `limitation=gpu_pressure_cap`. An unmet reserve remains an error.

## Validation

`serve/test_live_memory.py` covers capability admission, serialized writes, acknowledgement parsing and
identity, idle/busy operation, partial errors/backoff and engine lifecycle. Run it alongside
`serve/test_memory_policy.py` and the affected server regression suite.

Build the `live_memory_test` target with native expert tests enabled. It uses synthetic expert files to check
physical release/regrowth, stable addresses, replay of a captured CUDA graph, failed-growth rollback,
zero-capacity recovery, pinned/pageable RAM, current adaptive ownership and file fallback. Its
`--protocol-only` option omits GPU allocation checks.

Before deployment, also test real-model generation with capacity changes during streaming and idle, confirm
an unchanged engine PID and subsequent prompt-prefix reuse, and measure throughput under comparable memory
pressure. Passing the synthetic tests alone does not establish real-model performance or deployment readiness.

### Initial Windows measurement (2026-10-04)

Real-model functional acceptance passed six capacity changes during one generation and the same six while
idle: automatic RAM shrink/regrowth and reserve pressure/recovery, plus direct physical VRAM shrink/regrowth.
The engine PID stayed unchanged and the next prompt reused 1,857 tokens. This functional run used a 24 GiB
resident cap to leave regrowth headroom, synthetic policy capacity/time inputs, and real native allocation
guards; it did not create system pressure. Generation continued after every active change, with a largest
observed token gap of 0.64 s. Releasing about 3 GiB of RAM took 35 s across bounded steps, rather than one pause.

On an RTX 5080 (16 GiB), Ryzen 7 9700X and 64 GiB system RAM, the initial Huihui
IQ3_S/IQ4_NL run was slower than the installed 0.1.38 release. Both used a 65,536-token context,
int8 KV, a 42 GiB resident cap, a 1,536 MiB reserve setting and the same model/profile/MTP files.
One warmup preceded two measured requests of 6,165–6,166 input tokens and 512 output tokens.
Changing the first system word prevented prompt-prefix reuse. No controller mutations ran during this benchmark.

| Measurement | Installed release | Experimental live mode |
| --- | ---: | ---: |
| Mean decode (tokens/s) | 35.13 | 22.79 |
| Mean prefill (tokens/s) | 1,021.55 | 318.31 |
| Mean first-token latency (s) | 6.09 | 19.45 |
| Initial expert VRAM (MiB, reported) | 4,754 | 3,103 |
| Initial expert RAM (MiB) | 36,564 | 35,520 |
| Free VRAM after startup (MiB) | 973 | 1,586 |

These are sequential workstation observations, not an isolated estimate of the cost of VMM or resizing.
The live path owns its prompt buffers and enforces the reserve after all buffers are allocated; the release
borrows expert storage for prefill. Available RAM also differed. Outputs and speculative acceptance can differ.
That regression kept the first implementation out of the daily installation.
This table records the first implementation with borrowing disabled. Guarded borrowing, refreshed loan layouts
and budgeted RAM coverage are implemented in the follow-up below.

### Guarded borrowing follow-up (2026-10-04)

The rebuilt native suite passed all 230 live-memory checks and the three affected CTest targets. Real-model
prefill acceptance queued a MEMORY command during a 16,562-token prompt: acknowledgements arrived after
REUSED, following completed refill. Fresh prompts ran after cache shrink/regrowth. STOP during borrowed
prefill cancelled cleanly with no generated tokens. An unreachable reserve reported `prefill_cache_floor`
with the resulting actual capacity; a fresh 2,126-token prompt then ran in at least eight chunks at that floor.
After regrowth, an 8,562-token prompt completed and a follow-up reused 8,625 KV tokens with the same process.
This functional test used a 24 GiB RAM cap and did not allocate artificial system pressure.

The follow-up also repeated all 12 decode/idle capacity changes with real native allocation and synthetic
policy capacity/time inputs. One generation continued through every active change, with a largest observed
token gap of 0.58 s; the following turn reused 1,887 KV tokens. The process and model/context/KV arguments
stayed unchanged. This separate functional run also used a 24 GiB RAM cap to leave regrowth headroom.

The original benchmark workload was repeated with the installed release and guarded borrowing, using the
same model/profile/MTP, context, KV precision, 42 GiB cap and 1,536 MiB reserve. It again used one warmup,
two measured fresh prompts and 512 output tokens, without controller mutations.

| Measurement | Installed release, repeated | Live mode with guarded borrowing |
| --- | ---: | ---: |
| Mean decode (tokens/s) | 44.52 | 45.87 |
| Mean prefill (tokens/s) | 1,310.17 | 1,327.36 |
| Mean first-token latency (s) | 4.76 | 4.68 |
| Initial expert VRAM (MiB, reported) | 5,274 | 4,927 |
| Initial expert RAM (MiB) | 40,181 | 39,377 |
| Free VRAM after startup (MiB) | 1,174 | 1,931 |

These sequential observations show performance recovery close to this run's release baseline; two measured
requests do not establish a general speedup. Available RAM differed between runs and from the first table.
The original release does not enforce the reserve after all startup buffers, which also changes cache sizes.
The enabled server policy may choose or trim a smaller resident budget to preserve RAM headroom; this
benchmark does not establish sustained throughput after those automatic changes. Live mode remains opt-in.

### Refreshing current loan coverage (2026-10-04)

Adaptive swaps left later borrowers outside RAM: two follow-up prompts in the preceding run needed 831 and
975 MiB of file fallback, corresponding to 374 and 439 additional refill blob reads. The bounded retention
path above was then measured against the accepted borrowing implementation in A/B/A/B order. Each process
ran one warmup and four measured requests, giving eight measured requests per version with identical prompt
token IDs, native arguments, model/profile/MTP and 512-token output limits. No controller mutations ran.
These measurements preceded the serving-prefix correction described below; the final binary is measured
separately rather than inheriting results from that earlier binary.

| Mean measurement | Guarded borrowing | With current-loan coverage |
| --- | ---: | ---: |
| Decode (tokens/s) | 47.39 | 47.18 |
| Prefill (tokens/s) | 1,452.96 | 1,639.61 |
| First-token latency (s) | 4.28 | 3.80 |
| Whole request (s) | 15.10 | 14.65 |

Observed prefill throughput increased 12.8% and first-token latency decreased 11.3%. Decode differed by -0.45%,
within broad observed ranges (41.82–52.23 versus 42.79–52.18 tokens/s); no decode speedup is established.
Every candidate loan in this workload had complete RAM coverage and zero additional refill file-blob reads.
Logical file counters do not measure physical NVMe I/O. These remain sequential workstation observations:
outputs differ, and actual resident RAM was 40,776–40,901 MiB before versus 40,870–41,056 MiB after; expert
VRAM was 4,894 MiB before versus 4,767–4,863 MiB after. The donor-eviction tradeoff above still applies;
these observations do not guarantee that every workload benefits.

### Serving output limits and reusable prefixes

Real-model acceptance exposed an existing serving bug at the output limit: a fully accepted four-token
speculative window committed four inputs even when only two outputs could be emitted. The invisible tail
prevented the next chat turn from reusing its exact prefix. An accepted tail after EOS had the same risk.
Serving now bounds the proposed window to the remaining output allowance before verification and limits
the committed prefix to the first accepted EOS. The existing verifier restores recurrent, indexer and PLE
state to that prefix; the last emitted token remains the unconsumed head. Non-serving generation is unchanged.

The CPU-only `serve_window_test` reproduces the old mismatch and covers output allowances, partial/full
draft matches, accepted/rejected EOS and successive windows. Real-model acceptance then reused all 8,625
consumed tokens after the previously failing output-limited request; a separate natural-EOS continuation
also reused its complete consumed prefix. Long borrowed prompts, queued controls, STOP, the reduced prefill
floor and regrowth passed with the same native process and a 65,536-token context.

The coverage/prefix binary passed all 251 live-memory checks and six relevant CTest targets, including the new
serving-prefix regression and the existing conversation-cache/draft checks. It also passed all 12 real-model
decode/idle resize cases again: output continued through every active change, the process/arguments stayed
unchanged, and the following turn reused 1,911 KV tokens. The largest observed active token gap was 0.53 s;
releasing 3 GiB of RAM completed over 29.91 s of bounded steps. This functional run used a 24 GiB RAM cap,
synthetic policy time/capacity inputs and real native allocation guards, without artificial system pressure.

That binary ran one warmup and four fresh measured requests with the same prompt IDs, arguments and
512-token output limits used above. Against the preceding eight guarded-borrowing samples, mean decode was
47.75 versus 47.39 tokens/s, prefill 1,665.89 versus 1,452.96 tokens/s, first-token latency 3.74 versus 4.28 s,
and whole-request time 14.45 versus 15.10 s. The observed prefill gain was 14.7% and first-token latency
decreased 12.7%. Decode differed by only 0.74%; a decode speedup is still not established. All five loans
had complete RAM coverage and zero additional refill file-blob reads. Final expert RAM/VRAM were
41,025/4,894 MiB, with 1,587 MiB free VRAM after startup. These are sequential workstation measurements with
eight baseline samples versus four final samples, not a controlled trial or a throughput guarantee.

### Completing rounded RAM shrink requests

Actual daily-policy observation exposed a live-control completion bug despite the isolated resize cases
passing: a whole RAM block could take the allocation below an arbitrary MiB target. Exact-equality completion
then left the request pending, and a later step attempted growth into the rounded gap. Under real pressure,
the growth headroom guard correctly refused it, but the original shrink reported `ram_resize`.

RAM shrink now finishes when actual capacity reaches or falls below the target. That completion stays
latched for the whole MEMORY request while remaining VRAM steps finish, then resets for the next request.
The acknowledgement still reports actual rounded capacity. Genuine growth continues to enforce headroom
and can fail; no capacity, reserve or model-quality setting is relaxed.

The native regression uses a non-aligned target with unavailable positive headroom, checks that subsequent
VRAM-only steps cannot refill the undershoot, and requires a fresh genuine-growth request to remain an error.
It runs with pageable, pinned and mixed RAM blocks. The live-memory target now passes 272 assertions and all
six relevant CTest targets pass. Real-model and actual daily-policy verification remain required for deployment.

The repaired binary repeated long prefill, queued controls, STOP, reduced floor/regrowth, complete 8,625-token
continuation reuse and natural-EOS prefix reuse. The temporary pressure fixture reserved 768 MiB above
observed initial free VRAM so later lazy decode-graph captures left room to exercise real cache regrowth;
the production reserve/headroom and strict regrowth assertion were unchanged. An earlier 256 MiB fixture
correctly hit the native pressure cap and was retained as a failed regrowth test.

All 12 decode/idle resize cases then passed with the same process/arguments and 1,836-token continuation
reuse. The largest token gap across the whole stream was 0.77 s; the largest gap measured within an active
resize stage was 0.53 s. Releasing 3 GiB of RAM completed across 32.27 s of bounded steps. As before, this
functional run used a 24 GiB RAM cap, synthetic policy time/capacity inputs and real native guards.

The repaired binary's daily-settings benchmark repeated the same four fresh measured prompts: decode
40.35 tokens/s, prefill 1,278.41 tokens/s and first-token latency 4.86 s. A fresh guarded-borrowing baseline
then measured 44.30/1,226.10 tokens/s and 5.06 s, but obtained 38,568 MiB expert RAM versus 37,168 MiB for
the repaired binary (expert VRAM 5,149 versus 5,183 MiB). These unequal allocations cannot isolate a code effect.

A short comparison therefore fixed both versions to a temporary 32 GiB RAM cap and 1,900 GPU expert slots,
with the same model/context/KV/spec/reserve/profile/MTP and prompt IDs. Both reported 32,767 MiB expert RAM
and 4,221 MiB expert VRAM. One warmup preceded two fresh measured 512-output-token requests per version.
Guarded-borrowing versus repaired means were decode 29.41 versus 29.46 tokens/s, prefill 856.44 versus
946.60 tokens/s, first token 7.25 versus 6.56 s, and whole request 24.64 versus 23.97 s. Prefill increased
10.5% while decode differed by only 0.17%; this supports a prompt-processing benefit, not a decode gain.
Final loans had complete RAM backing versus 1,351/1,593 MiB fallback in the baseline's measured requests.
Only two sequential samples per version were measured, so this is no general throughput guarantee. The
temporary fixture differs from daily auto-cache/42 GiB settings and does not change the daily configuration.
