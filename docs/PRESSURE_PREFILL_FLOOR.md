# Pressure-only retained expert prefix (isolated experiment)

`--live-prefill-min-retained N` allows a live-memory controller to retain fewer
than 128 expert slots outside a borrowed prompt buffer when fresh GPU headroom
is below its requested reserve. It accepts integers from 0 to 128. The default
is 128; startup sizing and startup admission always retain the existing prefix.

This experiment starts from adaptive revision
`eef15a7d276dbaaf6c3041a5e7433c66b47a01fa`, based on Strata v0.1.42. It has not
been deployed. Model-level correctness and the tool-to-model return are separate
qualification gates; compilation and allocator tests alone do not prove them.

## What can be reclaimed

The runtime still maps the complete workspace counted by
`Prefill::bytes_needed(..., 256)`. A minimum of zero means that this smallest
prompt loan may leave no expert hits outside it. It does **not** mean zero
GPU-cache capacity or that scratch can be paged to the SSD. Before decode, the
borrowed slots are refilled from their authoritative RAM/file-backed source.

The normal startup floor and lower pressure floor are separate. The controller
uses the latter only for an actual fresh GPU deficit and only when lowering it
can release a whole VMM mapping block. Sized slot offsets determine the floor;
the driver-compatible mapping quantum determines physical commitment. A healthy
RAM-only request cannot use the new option to shrink the GPU cache gratuitously.
After a shrink, a held/no-growth request keeps that smaller allocation even if
free memory rebounds. Existing explicit, bounded recovery can restore it.

For each selected active cache size, arithmetic chooses the largest retained
prefix between the configured minimum and 128 that can fund the full minimum
prompt buffer. It then chooses a fitting chunk from the existing chunk list,
never above the host allocation made at initialization. Thus `N=0` does not force
every request to retain exactly zero slots.

## Ownership and limits

The existing cooperative safe point drains prompt readers, returns the old loan,
drains outstanding copies/source reads/verifier work, and only then changes
ownership. Prompt views are rebound before mappings retire. Reduced residency
is published before unmapping. The virtual base and complete offset table stay
stable. Existing cancellation checks prevent re-lending after a stop.

`Prefill::relayout` now rejects a loan below its conservative counted buffer size
before changing GEMM or raw carve pointers. This closes an existing failure path:
a failed outer borrowed allocation could leave a null nested allocator base,
which means owned allocation rather than a failed loan. The live controller
also restores its prior valid layout on a later relayout failure; failure of
that restoration terminates the engine rather than publishing a successful
control acknowledgement or continuing generation with uncertain views.

The option is limited to the existing supported single-GPU CUDA live-memory,
borrowed-prefill path. It requires fixed KV, including the effective
`STRATA_KV_GROW` environment override, and rejects `STRATA_KV_STAGE_OWN` even
when its value is `0` (that diagnostic switch is enabled by presence). Existing
exclusions for split/peer/helper caches, asynchronous adaptation, batch serving,
segmented manual VRAM control and missing graph residency remain unchanged.

Context, MTP, sampling, projection settings, base headroom, the default **256 MiB
extra return allowance**, and AUTO's full unload/reload fallback are unchanged.
The allowance remains an admission allowance, not a permanently reserved
allocation. Ordinary recovery still requires its existing stable-headroom dwell.
Lazy MMQ/sampler/graph allocation and external applications can consume memory
later, so the real first-return test must also check actual headroom throughout.
There is a finite non-reclaimable model/KV/workspace floor; this is not a promise
of never running out of memory or faster token generation.

## Diagnostics

Only when the option is explicitly supplied, stderr records:

```text
strata live prefill: shrink id=... minimum_keep=... selected_keep=... slots=... normal_floor=... pressure_floor=... mapped_before=... mapped_after=...
strata live prefill: memory id=... status=... minimum_keep=... selected_keep=... slots=... normal_floor=... pressure_floor=... mapped_bytes=... available_bytes=... reserve_bytes=... error=...
```

Byte values are bytes, not MiB. `available_bytes` is the minimum of current CUDA
free memory and the current local DXGI budget headroom when available. A shrink
record is not an admission certificate; use the corresponding terminal memory
acknowledgement and the server's fresh post-acknowledgement admission sample.
The original default protocol and logging are retained when the option is omitted.

## Validation boundary

- `live_prefill_policy_test` compares sized-offset plans with an independent
  brute-force oracle, checks default128 equivalence, physical block rounding,
  healthy/pressure/held transitions, strict parsing and rollback outcomes.
- `live_memory_test` adds real VMM/stream fixtures for retained minima 0 and 32:
  old-reader drain, loan return, reduced ownership, physical release, source-byte
  refill and replay of a captured stable-address read after recovery.
- `prefill_relayout_test` uses real Prefill initialization and relayout with
  source-absent and source-present allocator descriptors, without model weights
  or inference. It checks rejection before chunk-state mutation, no substantial
  extra allocation on rejection, restoration and later valid carves.

Exact build and executed-test results are recorded with the candidate. CUDA is
the intended qualified backend. HIP and SYCL have not been built for this change;
the candidate remains isolated and is not presented as ready for those backends.
No model-level speed or successful resident-return claim is made by this document.

The isolated Windows CUDA build on 2026-10-10 passed all 16 selected native tests,
including 9,789 pure policy checks, 708 live-memory checks and 32 real Prefill
allocator checks, plus the IQ3_S fused/MMQ reference comparison. Four native CLI
test methods covered 19 invocations without loading a model. The toolchain was
MSVC 19.51.36260 (toolset 14.51.36231), CUDA 13.4.59, SM89. The first run's newly
added VMM fixture failed because its write and read used unordered CUDA streams;
placing them on the same stream corrected that fixture, and the complete rerun
passed. This is test-harness repair, not an engine performance finding.

## Provenance

The implementation is an original extension of Strata's existing parameterized
`live_prefill_first`/`live_prefill_floor`, cooperative loan return, `ExpertCache`
VMM resizing and live MEMORY protocol. Their earlier ownership lineage, including
the original [PR #726](https://github.com/Niko1221/Strata/pull/726), remains recorded
in [ADAPTIVE_PROVENANCE.md](ADAPTIVE_PROVENANCE.md). No external implementation or
paper algorithm was copied for this change. The relayout rejection fixes a local
failure path and is not attributed to an unrelated performance paper.

The bounded duplication review inspected adjacent proposals
[#1581](https://github.com/Niko1221/Strata/pull/1581),
[#1117](https://github.com/Niko1221/Strata/pull/1117),
[#1131](https://github.com/Niko1221/Strata/pull/1131),
[#1231](https://github.com/Niko1221/Strata/pull/1231), and
[#563](https://github.com/Niko1221/Strata/pull/563). They address different lifetime,
elastic-KV, decode-resource or cache-release mechanisms. They are reviewed prior
art, not imported code or evidence that this experiment improves performance.
