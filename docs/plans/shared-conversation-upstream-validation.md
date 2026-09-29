# Conversation cache validation

The core review branch is based on upstream 0.1.26 (`4c68013`). Recorded model
results below belong to the earlier 0.1.25 implementation (`cabd50c` through `1d9e4e7`),
not the new base. The Windows admission test is @midhatn's `32cf918`, retained as
`b319d43`. Historical records and the general Pi benchmark remain on local branch
`feat/conversation-cache-upstream` at `cb8d90c`; they are outside the core patch.

## Current offline checks

After separating the general benchmark tooling and merging 0.1.26, all 47 tool
tests and the 22 cache-harness tests under Python `-O` pass. Recoverable snapshot
rejection now clears its diagnostic error before the new batched draft-prefill
path runs. That C++ integration change still needs an engine build and model
validation; these Python passes do not establish it. No benchmark server or GPU
was used for these checks.

## Recorded Linux evidence (2026-09-29)

EPYC 7532, RTX 4090, GCC 15.2, CUDA 13.4, SM89, portable AVX2, IQ3_S. Paired
engines used 131,072 context, INT8 KV, 32,768 resident cells, 7,846 expert slots,
and PCIe share 0.55. Evidence remains in that worktree's `logs/upstream-20260929/`.

| Gate | Result and scope |
| --- | --- |
| Components | Seven CTest targets passed: cache policy, admission, GPU round trips, host validation, injected transfers, checkpoint retention, sampler parity |
| Python | 65 server tests (three skipped), 47 tool tests, 22 harness tests under `-O` |
| ASan/UBSan | 35 policy, 23 admission, 1,116 host validation/transfer checks passed |
| CLI | Ten malformed/range/layer-split checks passed |
| Full model | Nine paired gates: reuse/checkpoints, long speculation, pressure, oversize, exchange, admission denial, synthetic image/grid and add/project steering isolation |
| Soak | 30 known-answer returns at 2,026 / 39,985 / 119,987 tokens; exact outputs/main-state fingerprints and stable retained payload |
| HTTP | Twelve requests passed reuse, eviction, disconnect and recovery without an engine restart |

The nine INT8 gates and soak used executable SHA-256
`0d8cbca38ce27153cea516cf6454df4653196ae875e0224fd3dd3c51e1d27ea0`.
K8V4 then exposed incorrect hybrid-format flags. After correction, executable
`4e174e8a4b602f1f8efe846cf19570478fcc7d2b50f3c48373c799d39b3610c7`
passed K8V4/INT8 model parity, the three affected CTest targets, the 1,116 sanitizer
checks, and HTTP recovery. The full nine-gate suite was not repeated on that fix.
An initial unequal-expert-residency comparison was rejected, not counted as a pass.

Four bounded Pi coding runs passed ten independent checks each. Return prompt
processing took 4.33–5.78 s with parking off and 1.50–1.52 s with it on, with
similar but unequal prompts. Whole-task timing favored opposite modes in the two
pairs, so there is no demonstrated overall speedup/regression. The benchmark
harness, detailed results and failed preliminary tool-budget trial are separate
from the core patch. These are not version-to-version or Windows measurements.

## Reproduction

Configure the engine normally with `-DSTRATA_BUILD_CONVERSATION_TESTS=ON`, then:

```sh
cmake --build build --target conversation_cache_test conversation_memory_test conversation_snapshot_test conversation_validation_test conv_cache_test sampler_parity
ctest --test-dir build -R '^(conversation_.*|conv_cache_test|sampler_parity)$' --output-on-failure
python -m unittest tools.test_conversation_cache_parity tools.test_conversation_cache_isolation tools.test_conversation_cache_soak tools.test_conversation_cache_http_smoke
```

On GNU/Clang ELF CUDA builds also build `conversation_transfer_test`; it wraps
copies and synchronization for host fault injection. It does not simulate recovery
of a broken CUDA context. GPU fixtures cover partial pages, distinct indexer spare
keys, early checkpoints, zero-QSA and 256/512-expert metadata.

Model tools default to a dry run. With `--config CONFIG --engine ENGINE --output
NEW_DIRECTORY --run`, run parity at `--spec 1` and `--spec 4`, then scenarios
`pressure`, `oversized`, `exchange` and `admission`. Select byte budgets that
actually force the named condition; use a RAM floor above available memory for
denial. Isolation scenarios are `image`, `add`, and `project`. The soak requires
at least 30 cycles and three lengths crossing resident KV and approaching the
configured context limit. HTTP smoke requires an exclusive idle test endpoint.
Run model/GPU gates only in an exclusive test window.

## Outstanding evidence

- Repeat affected build/model gates on 0.1.26, including batched draft prompt KV
  and its fingerprint. Previous main-model hashes excluded draft scratch/state.
- NVMe restart, corruption, foreign identity, eviction/promotion and explicit
  staging bounds; no disk acceptance is claimed yet.
- Windows runtime: only contributor-reported 25 MSVC admission checks, not locally
  reproduced. HIP execution, multi-GPU parking, real vision encoder and full
  Coder-model runs are untested; synthetic fixtures do not substitute for them.
- Full optional upstream test configuration was blocked on 0.1.25 by missing
  `native_mmvq_multi.cpp` and `hit_cpu_order_parity.cu`; focused tests were used.
