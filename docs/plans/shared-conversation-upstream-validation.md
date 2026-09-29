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

On the dependent NVMe branch, five CPU-only CTest targets pass: RAM policy,
memory admission, file codec, disk store and checkpoint retention. The codec also passes 6,727 checks under
ASan/UBSan, compiled with `-Wall -Wextra -Werror`. Coverage includes every
single-byte mutation and truncation of a fixture, foreign identities, missing
assets, middle-of-file asset changes, staging/floor rejection, all encoded KV
formats and checkpoint buffers, and metadata-only matching against the RAM
policy. Codec format coverage does not establish model compatibility or disk
restart/promotion correctness. No GPU or benchmark endpoint was used.

The store tests cover reopening, foreign identities, corruption, LRU/byte/entry
retention, no I/O when disabled, eviction-only callbacks, failed spills and protected
promotion candidates under tight RAM/disk budgets. Linux syscall wrappers inject
write, file-sync, rename and directory-sync failures and observe quota usage during
writes. Child processes exit during writing and immediately after rename to verify
lock release, temporary-file cleanup and complete-file discovery after restart.
These are host lifecycle tests, not power-loss simulation or full-model restoration.
The latest ASan/UBSan results are 141 store, 47 RAM-policy and 23 admission checks.
Progress-callback tests cover large reads/writes, unchanged encoded bytes,
forwarding through the store, and no heartbeat for disabled or initially rejected
operations. Local commands and logs are in `logs/nvme-host-20260929/` on the NVMe worktree.

The serve-loop integration passes GCC C++20 syntax checking against CUDA 13.4
headers with disk support both enabled and disabled, including the shared geometry
key export. This is not a linked engine build. The frontend suite passes 67 tests
(three skipped), including tokenizer/template forwarding from the actual frontend
paths. Full-model NVMe promotion, restart, output parity and state hashes are still
unverified; no running benchmark server or GPU was used for these checks.

The draft read-back diagnostic passes 1,188 host transfer checks under ASan/UBSan,
including injected corruption and failed copies. The host backend wraps CUDA
copies/synchronization; unused ring/stream GPU functions abort if reached. The
real GPU fixture now checks the ring produced by restore directly, without an
extra refill that could mask a bug; that updated GPU fixture has not run yet.
State fingerprint reads now use fixed-size chunks and reject transfer failures.

`tools/conversation_cache_disk.py` prepares six sequential private engines for
baseline, eviction, restart, admission denial, changed tokenizer identity and
corrupted files. It requires known answers, token/main-state parity, matching
draft read-back fingerprints across restart, and the requested draft-prefill
path. Its five offline verifier tests pass, including 28 rejected evidence
mutations and optimized Python. This establishes the verifier's checks, not
full-model success. All model executions remain pending the exclusive test window.

```sh
cmake -S . -B build-conversation-host -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DSTRATA_ENABLE_CONVERSATION_DISK=ON
cmake --build build-conversation-host --target conversation_file_test conversation_store_test conversation_cache_test conversation_memory_test conv_cache_test -j 1
ctest --test-dir build-conversation-host -R '^(conversation_file_test|conversation_store_test|conversation_cache_test|conversation_memory_test|conv_cache_test)$' --output-on-failure
```

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

The disk gate is dry-run by default. Run it separately for batched draft prefill
and the draft ring; each run owns a new output directory and modifies only its
copied tokenizer and generated cache files:

```sh
python tools/conversation_cache_disk.py --config CONFIG --engine ENGINE --output NEW_DIRECTORY --draft-path batched
python tools/conversation_cache_disk.py --config CONFIG --engine ENGINE --output ANOTHER_NEW_DIRECTORY --draft-path ring
```

Add `--run` in the exclusive window. Run INT8 and K8V4 configurations where
supported. The gate records actual paths/modes and refuses to count an absent
batched pass or ring restore as coverage.

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

## Review updates (2026-09-29)

[QilinWan's identity review](https://github.com/Niko1221/Strata/issues/57#issuecomment-5898749306)
is mapped to named inputs and coordinate meanings in the design document. The
current digest refuses foreign identities but cannot identify the first differing
field in a log. Page-store deduplication suggested on #52 remains a follow-up.

[midhatn reports a 0.1.27 Windows test](https://github.com/Niko1221/Strata/issues/57#issuecomment-5900605241)
of a separate auxiliary snapshot patch: 3,216 reused tokens out of a 3,223-token
lookup, correct answers and shorter return latency. This is contributor evidence,
not validation of this branch; K8V4, HIP and batched draft prefill remain untested
there. No local Windows claim is made.

Upstream 0.1.27 (`a790805`) was inspected after the maintainer requested 0.1.26.
It adds HIP-only compilation fixes, Turing support, frontend image-marker fixes
and a changed draft vocabulary. The review branches remain based on 0.1.26;
merging that newer base requires rerunning affected frontend/model checks. Neither
release's upstream results establish this integration's model correctness.
