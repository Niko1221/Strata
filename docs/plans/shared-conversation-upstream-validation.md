# Conversation cache validation

The focused core branch includes upstream 0.1.27 (`a790805`). Final Linux checks
ran on 2026-09-30 after the user's Pi benchmark completed. This record describes
the review binaries, not the original server restored after the test window.
Windows admission coverage retains @midhatn's original authorship in `b319d43`.

## Hardware and artifacts

EPYC 7532, RTX 4090, GCC 15.2, CUDA 13.4, SM89, portable AVX2, IQ3_S.
All model engines used `numactl --interleave=all --physcpubind=0-31` and matched
expert residency (7,846 slots, PCIe share 0.55). INT8 streamed/ring tests used
131,072 context and 32,768 main resident cells; batched INT8/K8V4 tests used
16,384 context with full device KV. These are correctness gates, not a Pi speed
comparison or measurements at the production server's 262,144 context limit.

Core executable SHA-256:
`f3eda68ad0dce9a1604345743ba5432ed73f2efeaf6b4f84f435317d7f2ba9ff`.
Local commands, configs, raw logs and result JSON are under
`logs/review-0.1.27/`; full-model artifacts are in its `model-window/` directory.
Documentation-only commits after this build do not change the tested code.

## Passed gates

| Gate | Evidence |
| --- | --- |
| Components | Policy, admission, checkpoint retention, validation, injected transfers and sampler parity CTests |
| GPU snapshot fixture | 1,298 checks; FP16/INT8/Q4/K8V4 supported layouts, partial pages, indexer spare keys, early checkpoints, zero-QSA, 256/512-expert metadata and draft ring read-back |
| Python | 66 frontend tests (three skipped), 47 tool tests, 22 cache-harness tests under Python `-O` |
| Transfer diagnostics | 1,188 ASan/UBSan host checks, including corrupted bytes and failed copies |
| RAM model reuse | Streamed INT8; batched INT8 and K8V4; known answers, exact output tokens and main-state parity |
| Long speculation | Speculation width four beyond resident KV, with output/state parity |
| Capacity/fallback | 400 MiB pressure and exchange, 1 MiB oversized entry, impossible RAM-floor rejection in ring and batched paths |
| Isolation | Synthetic image/grid identity and add/project steering isolation |
| Soak | 30 returns at 2,026 / 39,985 / 119,987 tokens; exact output/state parity and bounded retention |
| HTTP | Twelve requests passed reuse, slot eviction, disconnect and recovery on a private endpoint |

The opt-in `STRATA_SNAPSHOT_VERIFY=1` diagnostic records the actual draft-prefill
path and verifies restored authoritative bytes and resident ring pages before
emitting a fingerprint. Main-model state hashes alone do not cover draft state.
The GPU fixture checks the ring produced by restore without refilling it first.
Host fault injection does not simulate recovery of a broken CUDA context.

## Reproduction

Configure the normal engine with `-DSTRATA_BUILD_CONVERSATION_TESTS=ON`, then:

```sh
cmake --build build --target conversation_cache_test conversation_memory_test conversation_snapshot_test conversation_validation_test conv_cache_test sampler_parity
ctest --test-dir build -R '^(conversation_.*|conv_cache_test|sampler_parity)$' --output-on-failure
python -m unittest tools.test_conversation_cache_parity tools.test_conversation_cache_isolation tools.test_conversation_cache_soak tools.test_conversation_cache_http_smoke
```

On GNU/Clang ELF CUDA builds also build `conversation_transfer_test` for wrapped
copy/synchronization fault injection. Focused targets were used; this record does
not claim the full optional upstream test configuration passes.

Model tools default to dry runs. Supply `--config CONFIG --engine ENGINE --output
NEW_DIRECTORY --run` to parity at `--spec 1` and `--spec 4`, then scenarios
`pressure`, `oversized`, `exchange` and `admission`. Budgets must force the named
condition. Isolation scenarios are `image`, `add` and `project`. The soak requires
at least 30 cycles and three lengths crossing residency and approaching context.
Use `STRATA_SNAPSHOT_VERIFY=1` and verify actual batched-prefill evidence when
claiming that path. HTTP smoke needs an exclusive idle test endpoint. Run model
and GPU checks only in an exclusive test window.

## Limits and separate work

- Windows implementation reviewed, not locally executed. Contributor reports
  concern separate Windows work and do not validate this branch.
- HIP execution, real vision encoding and the full Coder model are untested.
  Synthetic image inputs test cache identity, not the vision encoder.
- Whole-conversation multi-GPU parking and hybrid K8V4 streaming/rings are
  unsupported; ordinary upstream stage checkpoints remain available.
- Optional disk persistence is a dependent change with its own acceptance record.

Historical 0.1.25 evidence remains on local `feat/conversation-cache-upstream`
at `cb8d90c`. General Pi tooling is isolated on
`tools/conversation-cache-benchmark`. Four historical bounded Pi trials passed
correctness, but overall timing favored opposite modes in the two pairs; no
whole-task speedup or version-to-version performance claim follows from them.
