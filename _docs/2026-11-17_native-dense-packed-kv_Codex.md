# Native dense packed KV, automatic admission and bounded upload

The scope is the owned CUDA Qwen35/Qwen3.8 runner on branch
`codex/native-dense-own-cuda13`, based on
`60e7dfc61fe9c573c4089e125d7ca55b0a91fc9a`. The root task authorised native
Spin8 and Turbo V cache formats, later adding automatic selection under VRAM
pressure and a bounded upload reader after an actual IQ4_XS upload stopped on
the unchanged RAM floor. No app/release files, main checkout source or original
checkpoint bytes were changed. The old executable remains untouched.

The canonical prompt, PC and project AGENTS, MILSPEC Codex skill, implementation
start gate, common engineering, skill selection, security, LLMOps, C++ and Python
SOPs were consulted. The SDD implementer contract applies. Python uses the
existing project resource-guard wrapper and its established Python 3.12 binary,
the custom-wrapper exception to the UV-first SOP. No dependencies were installed.
This work does not assert formal MILSPEC certification.

Requirements map to `src/qwen38/kv_config.cpp` for immutable profile admission,
deployed F32 audit, fingerprints, storage math and auto selection;
`kv.cu`/`kv.hpp` for packed history and bounded scratch; `ops.cu` for the narrow
post-normalization/post-RoPE handoff; `engine.cu` for CLI, INFO, full-budget
admission and actual forward integration; `session_state.hpp` for shared whole
reset; and `upload_reader.hpp` for bounded original-byte reads. The only codec
change extracts an existing Turbo scalar decoder for direct packed attention.
`kv_reference.hpp`, `kv_numerics.cu`, `kv_cpu_test.cpp`, CLI tests and the existing
CUDA dialect tests provide the corresponding evidence. CMake embeds the exact
reference profile bytes, including the source checkout's line endings.

Auto is the explicitly authorised new-runner default. It evaluates F16, F16 K
with Turbo4 V, then Spin8 Q8_0 K/Q4_0 V. Named modes remain exact. The request's
context never changes, and there is no CPU weight offload, weight conversion,
Turbo K, combined Spin8/Turbo, or out-of-memory retry. Full requested context,
weights, recurrent state, scratch, conversion peak, allocator allowances and
the unchanged 768 MiB CUDA reserve share one checked integer budget. INFO records
requested/effective mode, separate K/V, reason and byte estimates. RAM 3 GiB,
commit 4 GiB, H: 10 GiB and C: 5 GiB floors remain unchanged.

Spin8 uses the actual audited plan and one selected vector/plus/minus matrix.
Q/K rotate after normalization and partial RoPE; V rotates once. Packed attention
reads original stored coordinates and transposes the aggregate before gating.
Turbo V reuses pinned TheTom codec arithmetic and its inverse ordering, with
ordinary Q/F16 K. Two allocations hold all layers' K/V; per-layer views are
non-owning. No full-history floating cache exists. F32 score storage is separate
from KV and is 24 * context elements. Cache scratch is fixed across capacities.

The 16 MiB bounded reader copies exact admitted file ranges through pageable
staging and checks resource floors before every chunk. Failed or short reads
do not deliver partial chunks; construction RAII discards partial GPU state.
Metadata and per-token embedding reads retain the existing read-only mapping.
The actual IQ load replay remains root-owned; the observed RAM pressure is not
claimed to have a confirmed sole cause or a proven full-model fix yet.

Verification used the existing guarded runner at
`C:/Users/downl/Documents/ChatGPT/Strata/_docs/upstream-sync-20261116/check.py`.
All receipts and complete command outputs are in
`H:/Strata/evidence/upstream-sync-20261116/receipts/`. The separate build is
`H:/Strata/evidence/native-dense-own-cuda13-20261117/build-kv`.
Configure used CUDA 13.3.73, MSVC 14.44 and SM120. Builds request five jobs.
Final build receipt is `dense-kv-build-final.json`; final verbose CTest is
`dense-kv-ctest-final.json` with 9/9 groups passed. CLI receipt
`dense-kv-cli-r1.json` passed 50 CPU admission/flag cases. Consumer receipt
`dense-kv-done-final.json` passed the actual three DONE formatter-to-Python cases.

The production CUDA suite covers all six modes and all three Spin8 views:
eight mode/view combinations, 257 positions each, all packed bytes, GQA 24/4/256,
causal lengths 1/31/32/33/127/128/129/255/256/257, scale ties and signed extrema,
nonuniform gates, poisoned-cache rejection, finite-overflow errors and whole
KV/GDN/conv reset/reuse. The separate F16 route is bit-exact to legacy attention
for the tested normalization/RoPE sequence. Arithmetic tolerances were declared
before execution and never relaxed: attention 3e-5 absolute + 4e-5 relative;
pure F32 inverse 5e-7 + 5e-7 relative. Final attention maximum error is 2.0738e-6;
Spin8 inverse maximum is 1.78814e-7. These compare quantized storage arithmetic,
not quality against unquantized models. The reused dialect regression passed
11013 checks with zero failures, including the pinned binary golden fixture.

At capacity 4096 with 16 layers, measured payload/observed CUDA-used delta were
268435456/270532608 bytes for F16, 109051904/111149056 for plain K8/V4 and Spin8,
152043520/155189248 for Turbo2, 160432128/163577856 for Turbo3 and
168820736/171966464 for Turbo4. Measurements at capacity 513 are in the same
verbose receipt. Requested cache scratch stays 4, 24580 or 57348 bytes by mode;
Spin8 coefficients are 256 bytes. Device deltas include backing/pool effects and
are snapshots, not a universal exact allocator model. The planner explicitly
budgets conservative per-allocation 2 MiB slack rather than counting it as KV.

Retained failures: first build failed only in a fixture include that required
half helper declarations; the correction is covered by later passing builds.
The initial allocation test exposed 32 per-layer buffers requesting 33619968
bytes but consuming 67108864 bytes at capacity 513. Contiguous owners reduced
this to 39845888 bytes; the remaining backing/pool slack was incorporated in
the production planner and the allocation assertion. Original failed logs are
retained. The initial literal profile did not account for source CRLF bytes;
this was fixed before its first execution by compile-time byte embedding.
CUDA external-header C4819 warnings remain unsuppressed; a test-local C4244 was
fixed. No numerical tolerance was changed after a failure.

Self-review-event: qa_defect and ticket_completed. Review-only under
agent-self-review-evolution: allocation accounting was corrected at its narrow
production owner, source-backed Q8/Q4 rounding was inspected, and selected modes
cannot change live cache semantics. No governance or memory file was edited.
Source changes stay in the owned worktree. Actual model PQ2_0/Turbo V and
IQ4_XS/Spin8 answers, real auto-fit, loading under RAM pressure, 98k/131k full
window, throughput, app integration, installation and publication remain
NOT_RUN by this worker and belong to the root acceptance work. No speed claim
or release qualification follows from these fixtures.
