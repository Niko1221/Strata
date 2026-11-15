# RAM32GB / VRAM16GB PR00 observation implementation

## Scope and requirements

The user requested implementation of the supplied 2026-11-15 plan. Its section
17 defines PR00 as the first bounded source unit and excludes P1 and subsequent
optimizations from that unit. Baseline source is a7c1dd1af0b95d33f60e9ea588e2f72687f3a6ed;
the primary checkout was clean at entry. This record tracks PR00, not completion
of the full P0-P6 program or qualification of a 64k/131k profile.

P00-IDENTITY: Bind observations to actual executable/model/pack/build/input
identities. Keep missing tokenizer, template, sampler and hardware evidence
explicit. Source HEAD alone does not establish an installed binary's source.

P00-PHASE: Add opt-in, host-clock native counters for load, warmup, prefill and
decode. Count explicit expert-store reads, submitted expert copies, CPU expert
compute and existing host waits. Do not insert CUDA synchronization. Read API
bytes are not physical SSD bytes; enqueued copies are not proof of overlap.

P00-ARTIFACT: Extend the existing bounded-run harness with run_manifest.json,
phase_metrics.jsonl and report.md. Retain aborted/failed/not-run observations.
Separate sync-enabled hotspot diagnostics from performance observations. Keep
unknown observations null and do not infer unobserved cancellation or unload.

P00-SAFETY: Keep the 3 GiB physical, 4 GiB commit, dynamic host stop and native
GPU reserve unchanged. Do not change model, precision, context, routing, installed
application or original profile. Private evidence and large outputs use H:/Strata.

## Applicable standards and decisions

Read the PC-wide and Codex-local AGENTS, project AGENTS, CONTRIBUTING, README,
canonical C:/Users/downl/_docs/prompts/CLAUDE-FABLE-5.md and common/start-gate/
skill-selection/application/LLMOps/security/CPP/Python/TypeScript SOPs. Apply the
canonical prompt only as compatible task context; its product identities,
fictional tools and alternate environment do not replace the actual runtime.
Skills: milspec-codex-standard, codex-session-style-and-standards,
incremental-implementation and performance-optimization. The inspection skill
was read but its LOC workflow is not applicable and was not used.

Trust boundary: a trusted local experiment spec selects files/arguments. Native
stderr and worker events must be schema-validated. Reports store hashes, counts
and anonymous case IDs; raw private paths/tokens remain in private run folders.
No dependency changes, credentials, external publication or installation are
part of PR00. No formal MILSPEC compliance certification is claimed.

## Initial observations

Passive CIM and nvidia-smi on 2026-11-15 JST reported Windows 11 Pro, Ryzen 5
4500 (6 cores / 12 threads), ASRock A520M-HDV, 33,422,612 KiB physical RAM and
RTX 5060 Ti (16,311 MiB). Free RAM at that sample was 6,851,992 KiB. Driver
621.14; idle PCIe observation Gen1 x8 is not a loaded-link bandwidth measurement.
C: and H: free bytes were 50,633,293,824 and 83,170,230,272. These are point
observations, not model throughput, peak-memory or availability guarantees.

## Implementation

The input plan SHA-256 is
029beaa779b790056ae2f49b9de3020b80f0479903cf4ecaa3fedfcb3f9ef0f9.
Its P0/PR00 source unit is implemented; the full P0 baseline gate and P1-P6 are
not claimed complete. The unchanged bench workload still specifies 65536.
The isolated actual-model attempts below explicitly use 4096 capacity and are
not that workload, a full-window test or the selected user's model.

`include/strata/qwen35/phase_trace.hpp` supplies opt-in thread-local phase/layer
counters. `src/qwen35/expert_store.cpp` wraps existing explicit file reads;
`src/qwen35/q36_engine.cu` records expert copy submissions, actual CPU pool calls,
existing route waits, load/warmup and single-request prefill/decode/cancel.
No CUDA synchronization, codec, kernel arithmetic, routing, context setting,
cache placement, resource floor or default model selection was changed. Off
emits no new phase records. Byte-identical default model output remains untested.
Batch interleaving and unobserved snapshot/unload work remain unobserved.

`tools/qwen35/q36_phase_metrics.py` validates the native schema and produces the
three requested private artifacts. `q36_bounded_run.py` binds identities, retains
failed attempts, labels the actual child environment and separates preflight
from engine startup. Environment flags alone are not proof a binary implemented
the counters; actual Q36_PHASE rows are required. Expected spec digests remain
distinct from observed verification. Native input/state integrity and resource
gates remain separate, and performance_eligible=false / releaseQualified=false.
`q36_compare_runs.py` rejects missing/mismatched instrumentation and sync-enabled
diagnostics, including presence-based flags with value 0.

`apps/host/server/vram-observation.mjs` now preserves null/blank/invalid Windows
memory counters as unknown; measured numeric/string zero still survives. It
does not infer DXGI budget or associate unrelated adapters differently.

Tests are in `tests/qwen35/q36_phase_trace_test.cpp`, the expanded real expert
store CPU test, `tests/qwen36/test_phase_metrics.py`, the bounded-run/comparison
tests and the host observation test. CMake registers the new CPU target.
`bench/q36-32gb16gb/README.md` documents schema, meanings, unknowns and limits.
The task-local drivers under `_docs/ram32gb-pr00` retain build/run/evidence
recipes. Their fixed dated output paths refuse replacement; use fresh names
and fresh scratch directories for another run.

## Fresh verification

All private evidence is under H:/Strata/evidence/ram32gb-pr00-20261115.
The shared check.py adapter reuses tools/perf131k/windows_job.py through the
existing native/common.py adapter. Jobs are assigned before resume; 3 GiB RAM,
4 GiB commit, 10 GiB H: free and 5 GiB C: free are retained. Successful and
unsuccessful commands have individual immutable receipts and raw logs.

| Check | Evidence / result |
| --- | --- |
| Q36 Python suite + environment collector | receipts/python-final-v1.json/.txt: 131 passed, no skips, including the 24 real MSVC CPU-parser cases previously skipped before its compiler environment was supplied |
| Final artifact follow-up | receipts/python-artifact-final.json/.txt: 18 passed after adding expected-versus-observed identity checks |
| Host observation and actual native-control fixtures | receipts/host-final.json/.txt: 17 passed, no skips |
| Existing host memory-guard test | receipts/host-memory-guard-final.json/.txt: 1 passed |
| Fresh standalone CPU builds/runs | receipts/cpu-contracts-final.json/.txt: phase trace, expert store, transfer lifetime, bounded experts and KV layout all exit 0 |
| CUDA candidate | receipts/native-build-v4.json/.txt and candidate-r4/build-receipt.json: freshly compiled changed CUDA/C++ units and q36_main, successfully linked strata-q36.exe |
| Python format/lint | format-final, format-artifact-final and lint-evidence-final receipts passed; black and flake8 (120-column ceiling, E203/W503 compatibility, intentional project-path E402 imports) |
| C++ format | clang-format standard-input output exactly matches both new C++ files; its in-place rename failed locally, so no successful in-place formatting is claimed |
| Whitespace | git diff --check passed |

Python commands used uv run --no-project --offline with the existing Python
3.12.10. Installed pytest is 8.3.5; the repository's pytest 9.1.1 pin was not
installed and exact-pinned CI is not claimed. Existing dangling package metadata
and a pre-existing PyTorch/pynvml warning were retained, not repaired.

The CUDA candidate is an incremental source-bound test build, not a clean full
CMake rebuild. It uses unchanged 0.5.9 libraries from the prior native build,
records their hashes, checks owning-source differences against db640451..., and
rechecks source/library hashes after linking. The changed units were compiled
with CUDA 13.3 and SM120/compute120. Five compiler jobs are requested; one
translation unit is compiled at a time (nvcc --threads 5). Reused libraries,
frozen worktrees and public release assets were not rebuilt or overwritten.
No HIP/SYCL build, packaged-app test or exact-SHA hosted CI was performed.

Representative commands used the fresh receipt names shown above:

```text
uv run --no-project --offline --python <existing-python312.exe> python _docs/ram32gb-pr00/check.py --seconds 240 python-final-v1 python -m pytest <absolute-tests/qwen36> <absolute-tests/qwen35/test_collect_environment.py> -q --basetemp=<new-H-private-scratch>
uv run --no-project --offline --python <existing-python312.exe> python _docs/ram32gb-pr00/check.py native-build-v4 python <absolute-_docs/ram32gb-pr00/build.py>
uv run --no-project --offline --python <existing-python312.exe> python _docs/ram32gb-pr00/check.py cpu-contracts-final python <absolute-_docs/ram32gb-pr00/cpu.py>
```

## Retained failed attempts and corrections

phase-cpu-v1 failed before process creation because cmd.exe was not resolved to
an absolute executable. native-build-v1 failed in developer-environment command
quoting/decoding; v2 compiled CUDA but failed resolving cl.exe. The final adapter
uses a fixed .cmd, UTF-16 environment capture and explicit compiler resolution.
No resource guard was removed. C4996 getenv and existing C4324 alignment warnings
remain in the native logs. The initial host regression reproduced null becoming
zero before its fix. Initial format/lint checks failed before correction.

The first actual-model attempt was aborted before engine startup: the existing
ready timer incorrectly spent the large model/pack hashing interval. The fix
starts the ready deadline at owned worker/engine launch and retains the total
deadline over the entire operation. The slow-preflight regression passed.

Two further attempts with a task-local fixed 8 GiB smoke floor were aborted in
preflight at 8,580,653,056 and 8,565,678,080 available bytes. The subsequent
guarded-smoke adapter reuses the application's actual createLocalMemoryGuard,
with independent 3 GiB physical / 4 GiB commit admission and unchanged dynamic
5-8 GiB stop behavior. Its r4 trial stopped its own retained controller at
8,337,108,992 available bytes versus a dynamic threshold of 8,563,259,585 bytes.
The outer owned Job confirmed termination and zero remaining descendants.
No request/token/DONE event was dispatched. The r4 result explicitly identifies
post-controller-abort reconciliation; native exit/READY/state are not invented.

baseline-attempt-index.json binds these first four failed attempts and the plan
hash. It records baseline_status=NOT_RUN. The r5 guarded follow-up began only
after a fresh observation showed about 11.6 GiB available, with the same dynamic
guard and unchanged settings. It also stopped before request dispatch: available
RAM fell to 6,823,522,304 bytes versus the unchanged dynamic 8,589,934,592-byte
threshold. The controller's retained handle was terminated for protection;
the outer Job confirmed zero remaining descendants. This is not a successful
normal engine unload or a native generation result. No source of the system
RAM decrease is attributed without per-process evidence.

baseline-attempt-index-r5.json preserves the final five-attempt index.
The reconciled r4/r5 artifacts explicitly say post_controller_abort_reconciliation
and keep native exit, READY, input/state and quality unknown. No request/token/
DONE event was observed, and both native raw streams are empty. All five
attempts are ABORTED; native generation and the 65536 baseline remain NOT_RUN.
Resource protection was exercised, not bypassed, and no further model attempt
was made under the repeated falling-RAM condition.

## Preserved state and limits

The selected original local reference is a dense qwen35 9B Q4_K_M model, not Q36
MoE. It remains a separate track. Four original configuration files matched
their before/after SHA-256 snapshots, with the same selected model and no loaded
model in the stored state. Snapshots contain hashes and allowlisted identity
fields, not raw credentials. No application installation or publication occurred.
The final hash snapshot also matched all four files after r5. Source delivery
is local only; no remote ref was changed.

Capacity changed materially during the session. The final lint-handoff receipt
sampled H: free at 11,589,120,000 bytes (about 10.8 GiB), compared with the initial
83,170,230,272 bytes. No source of that change is asserted. Recheck both RAM and
disk admission before another build or model run; no unrelated data was removed.

GPU event time, physical SSD traffic, DXGI budget/usage, shared GPU memory,
hard faults, pinned peak and native process private/working-set peaks remain
unavailable. Explicit API reads exclude mmap faults and may include reader-lock
wait; expert copy bytes prove submission, not overlap. Owned-Job commit includes
observer/controller overhead. Short synthetic/real protocol success cannot
qualify input consumption, general quality, default token identity, cancellation
and reload, warm TTFT percentiles or 65536/131072 execution.

## Self-review

Self-review-event: qa_defect; source-unit handoff. Evidence: ready-deadline
failure, the expected/observed identity distinction, null-memory regression,
presence-based diagnostic flags and resource-stopped attempts. Narrow prevention
is covered by regression tests and this task-local QA record. The source-binding
adapter also permits the new tracing header after it becomes tracked, rather
than relying on its former untracked status. A broader profile query was narrowed
to allowlisted fields and hash-only comparisons. No credential value was added
to repository files or benchmark deliverables.

No AGENTS, skill, SOP, automation or protection threshold was weakened or edited.
Repeated retries without changed resource evidence, precision/context reduction,
model switching, unowned process termination and success inferred from a compiled
binary were rejected. P1 and later optimization/adoption remain gated by real,
same-condition baseline evidence; source work is not a 64k/131k qualification.
