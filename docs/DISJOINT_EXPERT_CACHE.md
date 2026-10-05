# Avoid duplicate expert slots across helper and primary GPUs

## Why this helps

On a PC with two small GPUs, helper experts start out separate from the primary
cache. Later, the adaptive primary cache can promote an expert that the helper
already holds. Both cards then spend memory on the same expert.

We observed about 1,900-2,000 duplicate slots during sustained requests.
This opt-in change keeps helper-owned experts out of primary promotion candidates,
in both the server and command-line generation paths.

## Use

Set `STRATA_DISJOINT_ADAPT=1` before starting the engine. It is off by default.
It leaves the existing peer-tier check and expert arithmetic unchanged.
The reservation mask is copied after successful static helper loading. When the
upstream dynamic helper optimizer is enabled, the initial mask is not frozen:
the live ownership query decides whether the helper still holds an expert.
The server keeps upstream's live ownership check, and the opt-in CLI guard uses
that check instead of stale initial reservations.

## Validation environment

- Two modified RTX 3080 20 GB cards; CUDA SM86, CUDA 13.1, Linux.
- Intel Xeon E5-2686 v4 (18 cores / 36 threads), about 94 GiB usable RAM.
- PCIe 3.0, no GPU peer-to-peer access; 17 expert-pool workers.
- Qwen3.8-Flash-Next IQ3_S, 131,072-token context, INT8 KV,
  32,768 resident KV cells.
- Medium thinking: 2,048-token reasoning cap, 8,192-token output cap.
- Physical GPU power limits stayed at 250 W and 280 W. No power increase.
- Historical performance tests used an isolated engine based on official
  v0.1.38 plus reviewed architectds/Strata changes (best, 05c0f36).
  This PR contains only our change, ported to official main (99f3dbd).
  The historical timings are NOT a measured speedup of this pure-upstream port.

## Historical benefit

Same binary and helper capacity, six complete JSON tasks per arm:

| 32K cached request | Original | Ownership guard |
| --- | ---: | ---: |
| Generation | 81.5 tokens/s | 94.1 tokens/s |
| Complete request | 12.997 s | 11.417 s |

Generation improved by 15.5%; request time fell by 12.2%.
The tasks returned complete 100-record arrays and correct source facts.
All 18 checks passed across the original, guarded, and larger-cache arms.
The larger-helper arm is not included in the comparison above.
These repetitive tasks do not represent all agent workloads.

## Validation and remaining work

- Standalone ownership test passes, including disabled, empty, unowned,
  owned, and out-of-range entries, plus candidate filtering.
- Linux compile/link against pure official main (99f3dbd) passed. No engine
  was started and no production settings were changed.
- Pure-upstream GPU performance and session/vision/cancellation regression
  checks still need to be run. Keep this PR as a draft until then.
- Historical combined builds passed those safety checks, but that is not a
  substitute for validating this independent port.

## 0.1.39 follow-up

The branch now includes official 0.1.39 (`6f32ec0`) and preserves the dynamic
helper optimizer. A combined custom build passed a five-arm, 240-request HTTP
suite using SC117 abliterated IQ3_S and unchanged 250/280W power limits.
Its recommended serial configuration decoded a 110K cached prompt at 130.1
tokens/s versus 107.5 for official 0.1.39. This is a WHOLE-BUILD result including
other optimizations, not a new speed claim for this ownership guard alone.
No independent dynamic-optimizer GPU regression or isolated speed measurement
is claimed here. The PR remains opt-in and draft.

The exact revised branch compiled and linked independently against official
0.1.39 on the Linux host above. The standalone ownership/candidate-filter test
passed. No model server was launched for this branch-specific check.
