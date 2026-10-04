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
storage is retired. In live mode, prefill does not borrow expert storage, which may reduce prompt throughput.
After all prompt and verifier buffers are allocated, startup checks actual free VRAM and trims expert mappings
to preserve the configured reserve before reporting READY. If that reserve cannot be met, startup fails.

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
The experimental mode therefore remains opt-in; the daily installation retains the release engine.
Safe prefill borrowing, refreshed loan layouts and budgeted RAM coverage for borrowed experts need further
work and comparable measurements before claiming a performance improvement.
