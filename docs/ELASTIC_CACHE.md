# Optional elastic expert cache

Add `--elastic` to the engine's arguments to allow a single CUDA GPU's expert
cache to grow and return whole physical chunks during `--serve`. It is off by
default. The cache reserves a virtual address range and maps physical memory
into it; resizing keeps both the arena address and the slot-offset table stable.

The startup profile fills a fixed core and a rotating tail. The core is not
lent to prefill, replaced by adaptive swaps, or removed by shrink. The tail can
be lent, replaced, grown, or reduced. Resizing runs between requests or at a
decode safe point, after pending expert fills are synchronized and published.
Prefill replans its loan and lays out its buffers again after a resize.

| Argument | Default | Meaning |
|---|---:|---|
| `--elastic-reserve-mib` | 500 | Free-memory target for the resize policy |
| `--elastic-chunk-mib` | 64 | Physical mapping chunk; rounded to CUDA's granularity |
| `--elastic-period-ms` | 1000 | Minimum interval between decode policy checks |
| `--elastic-stable-ms` | 5000 | Sustained free room required before growth |
| `--elastic-tail-mib` | -1 | Automatic tail: half the initial cache, capped at 6144 MiB; 0 disables the fixed core; a positive value sets the initial tail limit |
| `--elastic-shrink-step-mib` | 1024 | Maximum reduction per policy check, clamped to at least 64 MiB; the core remains |

The reserve is a policy target, not a guarantee that Windows or another process
cannot allocate that memory. Growth also keeps a margin for observed variation,
limits each step to 512 MiB, and backs off after a growth is quickly reversed.
Only whole mapping chunks are physically released.

The current port supports a native expert pack with a startup expert profile
and sized slots on one local CUDA GPU in serve
mode. CLI generation, layer splits, peer or remote expert tiers, and the resident CPU complement keep the
fixed cache. An unavailable VMM arena also falls back to the fixed allocation.
The HIP implementation keeps the fixed cache; this contribution does not claim
AMD elastic-cache support.

Changing expert placement or the automatic prefill chunk can change floating
point rounding, as with the existing adaptive and prefill settings. The arena
parity test therefore holds chunk geometry and expert placement constant, and
tests normal automatic operation separately.

## Functional validation

Build with `STRATA_BUILD_TESTS=ON` and `STRATA_NATIVE_EXPERTS=ON`, then run:

```sh
ctest --test-dir build --output-on-failure -R 'expert_cache_(elastic|per_layer)_test|expert_profile_save_test'
```

`expert_cache_elastic_test` uses real CUDA VMM with small allocations. It checks
growth, shrink, regrowth, stable addresses and offsets, preserved expert bytes,
zeroed new slots, unchanged state after rejected requests, and fixed-cache reopening.
It does not allocate a separate load to simulate a saturated PC.

## Relation to `--vram-elastic` (#533)

`--vram-elastic` resizes the cache on request (`POST /v1/vram`, between
requests, segmented arena). `--elastic` resizes it automatically from the free
VRAM it observes, with a fixed core and a rotating tail. The two never open
together: with `--elastic`, `--vram-elastic` reports that the automatic cache
resizes the arena and stays off. The methods are named `elastic_grow` and
`elastic_shrink` so they do not overlap with #533's byte-sized `grow`/`shrink`.

The elastic cache originated in the community Strata-2080Ti Daily fork
(https://github.com/imanu86/Strata-2080Ti, written with Claude Opus 5.5), where it
has run as the default configuration of a modified RTX 2080 Ti 22 GB since
September 2026. This port separates it from the checkpoint and SM75 kernel
changes and adapts it to upstream 0.1.39.
