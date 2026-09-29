# Shared-core integration onto upstream 0.1.25

Integration base: upstream `8fc40dd`, fetched 2026-09-29. Implementation starts
at `cabd50c`; Windows test contribution is preserved as `b319d43` (original
`32cf918`, by @midhatn). The earlier 0.1.18 validation record describes a different
revision and remains historical evidence.

Issue #57 references: [unified PR ownership](https://github.com/Niko1221/Strata/issues/57#issuecomment-5875083621),
[current upstream/single-GPU scope](https://github.com/Niko1221/Strata/issues/57#issuecomment-5887959836),
and the [Windows test contribution](https://github.com/jeremiahritchey/Strata/pull/1).
Upstream had advanced from the requested 0.1.22 to 0.1.25 when fetched; this
record pins the exact revision rather than referring to a moving main branch.

## Integration decisions

- Preserve upstream's root-pinned/LRU checkpoint policy and retain last-use
  stamps when parking a conversation.
- Preserve the existing layer-split checkpoint representation. Whole-session
  parking with `--layer-split` is explicitly rejected before loading the model;
  multi-GPU parking is deferred as permitted in issue #57.
- Retain chunked indexer processing and restore the per-sequence spare key,
  block-position metadata and spare pooled row at checkpoint resume.
- Add K8V4 snapshot layout support: independently size/copy INT8 K, K scales and
  Q4 V; the draft ring remains plain INT8. Reject hybrid streaming/ring layouts.
- Use the selected CUDA/HIP runtime target in CMake; CUDA linker-wrap fault
  tests are not registered for HIP. No HIP runtime validation is claimed.

## Windows review

The contributed change modifies only `conversation_memory_test.cpp`. Under
Windows it checks that `GlobalMemoryStatusEx` succeeds and that the helper's
sample is known and no larger than total physical RAM. Comparing against total
RAM avoids a race between two available-memory samples. It neither allocates
pressure memory nor changes admission behavior. No review defect was found.

This machine cannot run Windows/MSVC tests. The contributor reported 25 passing
MSVC checks in their PR; those are their results, not locally reproduced results.
Local Linux compilation runs the existing 23 admission checks. Full Windows
restore, memory-pressure and agent acceptance remain outstanding.

## Local environment and evidence

Linux, EPYC 7532, RTX 4090; GCC 15.2, CUDA 13.4, Release build, SM89, portable
AVX2. IQ3_S model; INT8 KV, 131,072 context and 32,768 GPU-resident KV cells for
the primary full-model gates. Engines run sequentially on the exclusive GPU.

The private evidence directory is `logs/upstream-20260929/`. For paired model
tests, `--expert-cache 6000` yields 7,846 variable-sized expert slots; the PCIe
share is fixed to 0.55. Both engines must report matching residency and other
inference settings. The first attempt requested 8,700 uniform slots, was trimmed
to different residency by VRAM availability, and failed the comparison guard.
That attempt is retained under `spec1/` and excluded from accepted evidence.

The entire optional upstream test build cannot configure because
`bench/micro/native_mmvq_multi.cpp` and `bench/micro/hit_cpu_order_parity.cu` are
absent at this revision. Focused conversation, checkpoint-retention and sampler
targets are built instead. Do not report this as a full upstream test-suite pass.

The controlled local Pi task is documented in [agent-cache-benchmark.md](agent-cache-benchmark.md).
Results below must remain tied to the actual tested executable/configuration;
synthetic reuse timings are not version-to-version performance comparisons.

## Results

Completed checks:

- Seven focused CTest targets: RAM policy, admission, GPU snapshot round trips,
  whole-image validation, injected transfer failures, upstream root/LRU retention,
  and sampler parity. K8V4 and checkpoint-stamp fixtures are included.
- 65 server Python tests (three skipped) and 47 tool tests passed; the 22 cache
  harness tests also passed with Python assertions disabled (`-O`).
- ASan/UBSan: 35 RAM-policy checks, 23 admission checks, and 1,116 host validation/
  transfer-failure checks passed. The latter compiles snapshot/state code with
  sanitizers and uses wrapped copies/synchronization, not CUDA hardware failures.
- Ten CLI checks: malformed/negative/overflowing options rejected and enabled
  layer-split parking refused before loading a model.
- Nine full-model paired gates passed: single-token and long speculative A/B/A,
  checkpoint recovery, byte pressure, oversized entries, incoming/outgoing
  exchange budget, deliberate RAM-admission denial, synthetic image identity,
  and add/project steering isolation.

In the long speculative run, returning to the 55,009-token A reused all 55,009
and processed 22 new tokens in 1.2559 seconds including snapshot work. Checkpoint
return reused 55,024 tokens and processed seven in 0.5746 seconds. Both output
and main-model fingerprints matched the uninterrupted baseline. This is an
intra-revision cache test, not a comparison against 0.1.18.

The 30-cycle soak passed at approximately 2k, 40k and 120k tokens. Every output
and main-model fingerprint matched its uninterrupted baseline. Cache payload
plateaued for each repeated context rather than growing each cycle.

The nine INT8 gates and soak above used executable SHA-256
`0d8cbca38ce27153cea516cf6454df4653196ae875e0224fd3dd3c51e1d27ea0`.
A subsequent full-model K8V4 gate exposed a format-flag mismatch: upstream uses
`kv_hybrid=true` with `kv_int8=false`, while the initial fixture incorrectly set
both. Parking was skipped; the gate failed its reuse assertion. The layout and
fixture now reflect upstream's distinct flags. The failed attempt is retained
in `k8v4/`; final checks use `k8v4-fixed/` and executable SHA-256
`4e174e8a4b602f1f8efe846cf19570478fcc7d2b50f3c48373c799d39b3610c7`.

On the corrected executable, K8V4 and INT8 full-model A/B/A plus checkpoint
parity both passed (including byte-exact main-model fingerprints). The three
snapshot/validation/transfer CTest targets and 1,116 ASan/UBSan transfer checks
were rerun successfully. The twelve-request HTTP smoke passed reuse, slot
eviction, disconnect cancellation and recovery with the same engine process.

The initial Pi pair (`pi-off-on/`) used a 12-tool/16-turn cap. Cache-off hit the
tool cap while correcting its own sample-data test expectations and did not
reach the CLI phase; cache-on completed all ten independent checks in 108.7 s.
This incomplete pair does not support a speedup claim. Both modes were repeated
with the same revised allowance of 24 tools/32 turns per phase; the 240-second
phase timeout, fixed task inputs and independent validators remain unchanged.

Both revised pairs passed all ten independent checks in all four runs:

| Order | Parking | Task seconds | Requests | Output tokens | Prompt seconds | Decode seconds | Engine peak RSS GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off → on | off | 91.59 | 20 | 6,319 | 23.259 | 64.865 | 49.734 |
| off → on | on | 108.37 | 28 | 7,361 | 25.750 | 78.780 | 53.433 |
| on → off | on | 104.57 | 23 | 7,274 | 22.823 | 78.350 | 53.433 |
| on → off | off | 111.39 | 26 | 7,525 | 27.948 | 79.471 | 51.979 |

The controller's first return request took 4.334 s with parking off (15,431
prompt tokens, zero reused), versus 1.503 s with parking on (16,333 prompt
tokens, 15,810 reused). Total task time increased in this pair while the
agent followed a different tool path and generated more output. This is not
evidence of an end-to-end speedup. RSS is Linux VmHWM for the engine, including
startup and model mappings, not cache payload or GPU memory.

In reverse order, the return request took 5.777 s with parking off (17,113
prompt tokens, zero reused), versus 1.522 s with parking on (16,165 prompt
tokens, 15,620 reused). Whole-task timing favored cache-on in this repetition
(104.57 vs 111.39 s), reversing the first pair. With different generated output
and tool paths, two pairs do not establish an overall speedup or regression.
The consistent finding is reduced return-prefill work: approximately 2.9–3.8×
faster for these observed requests, with higher engine peak RSS. Prompt lengths
are similar but not identical, so this is a workload observation, not a matched
microbenchmark ratio.

Task wall time excludes server/model startup. All runs used 7,846 expert slots,
INT8 KV, 131,072 context, 32,768 resident cells and the same speculative settings.
The runner verified matching engine settings within each pair. It records
executable/task-script hashes and identical initial-file hashes for each mode;
per-run tool execution totaled 0.96–1.15 s. Private detail is retained in
`pi-bounded-off-on/`, `pi-bounded-on-off/`, and `pi-comparison.json`.

## Scope and remaining validation

These results cover this Linux/RTX 4090 host and the local IQ3_S model. Windows
runtime acceptance, HIP execution, multi-GPU parking, real vision-encoder
integration, and the larger Coder model remain untested here. Synthetic image
identity and model-geometry fixtures do not replace those runtime checks.
The NVMe backend remains a separate integration; this work validates the RAM
backend and shared snapshot representation. No upstream PR or issue comment
was posted as part of this local validation.

This benchmark compares conversation parking on/off within the integrated
0.1.25 executable. It cannot establish whether 0.1.25 is faster or slower than
the earlier 0.1.15/0.1.18 branches, nor reproduce the previous Windows workload.

## Local deployment

The tested executable is running from `Strata-conversation-cache-upstream`,
branch `feat/conversation-cache-upstream`, in tmux pane `%15` on port 11434.
The runtime configuration is `logs/runtime-20260929-191116/engine-config.json`.
It preserves the user's automatic expert allocation, 131,072 context, INT8 KV,
32,768 resident cells, 8 GiB conversation budget, four slots and 2,560 MiB RAM
floor. Automatic allocation selected 8,814 expert slots; this differs from the
fixed 7,846-slot benchmark allocation and is not included in its timing claims.
Health and a non-streaming generation request passed (`OK`), with the server
idle and zero queued requests afterward. Earlier worktrees remain intact.
