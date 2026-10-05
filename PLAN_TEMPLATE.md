# Plan — <short, outcome-focused title>

**Status:** Draft / Proposed / Approved / In progress / Implemented / Deferred / Superseded
**Scope:** <modules, primary files, and documentation affected>
**Targets:** <model architecture/family, quantization, CUDA/HIP/CPU path, OS, and GPU configuration>
**Related:** <request/issue, prerequisite plans, and plans superseded by this one, or none>

> Copy this template to `plans/<descriptive-kebab-case-name>.md`. Replace placeholders with
> concrete decisions and evidence; scale detail to the change. Keep the behavior, evidence,
> scope, tests, and verification sections even for a small fix. Mark other sections not applicable
> with a reason rather than inventing work. Remove these authoring instructions in the finished plan.
>
> State whether the user requested planning only or also authorized execution; honor authorization
> already given in the conversation. Read `AGENTS.md` first. Historical plans are design records,
> not proof of current behavior: recheck source and contracts before using them.

## Repository context

Use the sources relevant to the change; do not copy old measurements as current results:

- Engine: C++20 in `src/` and `include/strata/`, built with `CMakeLists.txt`; CUDA and HIP
  are separate build backends. See [docs/DETAILS.md](docs/DETAILS.md) and
  [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for engine behavior.
- Serving: `serve/server.py`, the OpenAI- and Anthropic-compatible APIs, and `serve/web/`.
- Installation and launch: `setup.py`, `START-HERE.bat`, `setup.sh`, and the Docker workflow in
  `build.sh`, `run.sh`, `run2.sh`, `run3.sh`, and `docker/`.
- Qwen35MoE/Ornith: [docs/ORNITH_QWEN35MOE.md](docs/ORNITH_QWEN35MOE.md).
  Keep its own `ModelKind` and backend; Qwen3.8 behavior and `run.sh`/`run2.sh` must not regress.
- GPU validation: [docs/AMD_HIP.md](docs/AMD_HIP.md) and [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- User setup: [docs/AI_SETUP.md](docs/AI_SETUP.md); Strata's MCP interface:
  [docs/MCP_SERVER.md](docs/MCP_SERVER.md).

These links are relative to this root template. Adjust them when copying a plan into `plans/`.

## 1. Goal and requested behavior

<Describe the problem, who it affects, and the observable outcome. For a bug, distinguish the
reported symptom from the independently confirmed defect.>

- <Required behavior, stated so a test can decide whether it holds.>
- <Behavior that must remain unchanged.>
- <Exact user-visible text, formatting, or data semantics when those are part of the request.>

For stateful behavior, fill in a table or short sequence:

| Starting state | Action / event | Expected outcome | Must not happen |
| --- | --- | --- | --- |
| <state> | <trigger> | <visible result and state change> | <forbidden side effect> |

## 2. Current behavior and evidence

Inspect the current worktree; do not rely on old line numbers or assume a prior plan was executed.

| Evidence | Location / command | Finding |
| --- | --- | --- |
| <source path and symbol, optionally current line range> | <inspection or reproducer> | <verified behavior> |
| <existing test or offline probe> | <exact command> | <actual result, or explicitly not run> |

- **Root cause / gap:** <Explain the causal path, not just the symptom. Label hypotheses.>
- **Existing coverage:** <What proves current behavior; why it misses this case.>
- **Baseline:** <Relevant pre-existing failures, modified files to preserve, or unknowns.>
- **Environment:** <Commit/worktree, OS, CPU/RAM, GPU/VRAM, driver/toolkit, build flags,
  model artifact/quantization, and fixtures available; omit irrelevant details for a docs-only change.>

Use minimal, deterministic reproducers and owned temporary state. Never record credentials or
read real user auth/session state for evidence.

## 3. Scope and non-goals

**In scope:** <The smallest coherent change and any adjacent change required for correctness.>

**Out of scope:**

- <Unrelated cleanup, redesign, compatibility behavior, or follow-up deliberately excluded.>
- <Existing behavior or limitation this work does not fix.>

**Affected boundaries:** <Owning modules and their consumers; dependencies added/moved, or none.>

## 4. Design and decisions

Describe the relevant path: launcher/configuration, GGUF or pack loading, architecture dispatch,
engine execution, token protocol, API response, and web rendering. Name the owner of validation
and state changes, and use existing helpers before introducing abstractions.

| Decision | Chosen design | Rationale / rejected alternative |
| --- | --- | --- |
| <design question> | <concrete choice> | <why it meets the contract with less risk> |

### Contracts and invariants

Explicitly record changes or “unchanged” for each relevant boundary:

- **Architecture / artifacts:** <`ModelKind`, GGUF metadata/tensor shapes, quantization,
  tokenizer/chat template, pack format, and draft/MTP compatibility. Keep each model's artifacts
  together and reject unsupported combinations explicitly.>
- **Numerical correctness:** <Reference implementation, precision, comparison metrics/tolerances,
  greedy-token agreement, and speculative verification/rollback as applicable.>
- **GPU / host memory:** <Device ownership, VRAM reserve and allocations, expert-cache admission,
  pinned/mapped RAM, mmap/storage behavior, streams/synchronization, and multi-GPU handoffs.>
- **Session / cache state:** <KV and recurrent state, reset/checkpoint/restore, prompt reuse,
  conversation isolation, cancellation, and cleanup on engine failure or client disconnect.>
- **Engine protocol / APIs:** <CLI/config defaults, token and `DONE` fields, producer/consumer
  updates, OpenAI/Anthropic streaming, usage accounting, finish reasons, tool calls, and errors.>
- **Setup / launch / packaging:** <Windows/Linux paths, device selection, pinned dependencies,
  prebuilt engine/runtime contents, Docker configuration, and resumable download behavior.>
- **Security / filesystem:** <Bind to `127.0.0.1` by default; exposure beyond loopback requires
  `--api-key`. Keep credentials out of evidence and use owned temporary files for tests.>
- **Web app / lifecycle:** <Request rendering, settings, engine startup/restart, and shutdown.>

For a substantial state-machine change, include states, transition ownership, and event ordering.
For a small fix, a precise branch description is sufficient.

## 5. Ordered implementation steps

List executable, dependency-ordered work. Each step should identify files/symbols, the change,
its validation, and a completion condition. For behavior changes, prefer regression coverage at
the lowest owning boundary. For reversible, low-impact edits, use an appropriate existing check
or inspection rather than adding tests that only mirror the implementation.

1. **Regression / contract — `<module or path>`**
   - <Reproduce the defect or pin the requested behavior with a deterministic test or probe.>
   - Done when: <the evidence demonstrates the gap without unrelated failures.>
2. **Implementation — `<module or path>`**
   - <Minimal change, relevant helper reuse, and failure handling.>
   - Done when: <the regression passes and preserved invariants remain covered.>
3. **Integration / documentation — `<consumers, tests, docs>`**
   - <Update all affected consumers and authoritative docs in the same change.>
   - Done when: <cross-module behavior and release assets agree.>

Reinspect `git status --short` before execution and preserve unrelated user work. Stay within the
authorized scope; planning alone does not authorize a model download, driver installation, or
deployment. Regenerate derived artifacts through their owning tools when required by the change.

## 6. Edge cases and regression matrix

Select cases relevant to the change; explain material exclusions. Include negative assertions
(no duplicate dispatch, mutation, event, leak, or misleading success) as well as positive results.

| Scenario | Expected behavior / invariant | Test boundary and test name |
| --- | --- | --- |
| <normal path> | <result> | <lowest owning module/test> |
| <empty, invalid, or boundary input> | <rejection/no-op and unchanged state> | <validation test> |
| <failure or cancellation> | <terminal outcome, cleanup, preserved input/state> | <failure test> |
| <repeated, late, or racing operation> | <ordering and idempotency> | <deterministic race test> |
| <restart/recovery or rollback, if applicable> | <same durable observable state> | <session/integration test> |
| <Qwen3.8 vs Qwen35MoE, relevant model families/quantizations> | <correct dispatch and preserved behavior> | <layout/numerical/launcher test> |
| <CUDA vs HIP, Windows vs Linux, single vs multiple GPUs> | <applicable supported behavior> | <backend/platform test or explicit unverified gate> |
| <affected API/web/setup/runtime path> | <end-to-end contract> | <mock-server/launcher/packaging test> |

Keep host regression checks independent of GPUs, model downloads, credentials, and live services
where possible. Use synthetic GGUFs, mock engines, and owned temporary resources. List GPU/model
gates separately with their fixtures and prerequisites; missing fixtures are not correctness passes.

## 7. Verification and acceptance

### Planned commands (not results)

Select only checks relevant to the change and replace placeholders with actual targets/test names.
Run from the repository root, narrowest first. Python examples use `python`; select the interpreter
from the repository environment when dependencies are needed.

Host checks (no GPU or model download required):

```bash
python tools/test_setup_choices.py
python tools/test_setup_amd.py
python -m unittest serve.test_server serve.test_parser_stream serve.test_lifecycle -v
python -m unittest discover -s docker -p 'test_*.py'
git diff --check
```

Use the corresponding `tools/test_setup_<name>.py` for other setup changes. For host C++ checks,
configure a separate build directory with CMake 3.24+ and a C++20 compiler. This example disables
native experts to avoid fetching ggml; it does not validate native expert execution:

```bash
cmake -S . -B build-plan-host -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=ON
cmake --build build-plan-host --target gguf_reader_test gguf_split_test \
  qwen35_layout_test qwen35_gdn_test qwen35_layers_test -j2
ctest --test-dir build-plan-host --output-on-failure --no-tests=error \
  -R '^(gguf_reader_test|gguf_split_test|qwen35_layout_test|qwen35_gdn_test|qwen35_layers_test)$'
```

For GPU engine changes, record the exact CUDA or HIP configure flags, GPU architecture, build
targets, and focused CTest selection, then run the relevant backend suite. Enable only one GPU
backend per build. Check `ctest --test-dir <build-dir> -N` before selecting tests; registrations
depend on build options and available sources. Use `--no-tests=error` so an empty selection fails.
For offline native-expert builds, set `STRATA_GGML_DIR` to a checkout matching
`third_party/ggml/VERSION.txt`.

- HIP: follow `docs/AMD_HIP.md`; `./build.sh --tests` builds/packages through Docker and runs
  HIP tests on the visible GPU. Record required Docker/device access and any fixture exclusions.
- Qwen35MoE: follow `docs/ORNITH_QWEN35MOE.md` for reference parity and real-artifact gates.
  Validate the affected `run3.sh` contract and Qwen3.8 `run.sh`/`run2.sh` regressions.
- Serving/web: use mock-engine tests for API contracts; check the affected browser flow when
  rendering or interaction changes. A GPU engine test alone does not verify API behavior.
- Setup/packaging: exercise affected model/backend/platform choices and runtime contents without
  downloading models in host tests. Follow `docs/AI_SETUP.md` for an authorized user installation.

Document unavailable hardware, missing fixtures, and excluded tests individually with reasons.
Do not turn a docs-only or unrelated change into a mandatory model/GPU benchmark.

### Measurement plan (when performance or capacity changes)

Define baseline and candidate runs with the same model, hardware, settings, and prompts. Store
commands and raw results under a descriptive `bench/results/<date>-<topic>/` directory.

| Item | Required evidence |
| --- | --- |
| Environment | Commit/build flags, OS, CPU/RAM, GPU/VRAM, driver/toolkit, storage |
| Model/settings | Artifact/quantization, tokenizer, context capacity, actual prompt/output lengths, KV type, prefill chunk, cache, pool workers, MTP/spec, sampling/seed |
| Workload | Cold/warm runs, repetition count, prompt/conversation fixture, cache reuse, concurrency |
| Results | Prefill and decode tok/s separately, latency, peak VRAM/RAM, quality/parity, raw logs |

Set acceptance thresholds before measurement. Configured context capacity is not proof of a filled
context run; VRAM occupancy is not throughput. Use plain words and attribute every measured number
to its environment. Label projections and unverified claims explicitly.

### Acceptance checklist

- [ ] Requested behavior and preserved behavior have appropriate tests or inspection evidence.
- [ ] Relevant numerical, failure, cancellation, cache, ownership, and cleanup paths are covered.
- [ ] Affected model/backend/platform boundaries are checked, including Qwen3.8 regressions when applicable.
- [ ] Protocol/API consumers, authoritative documentation, and release packaging are aligned.
- [ ] Relevant host checks and applicable GPU/model gates passed, or blockers are explicitly reported.
- [ ] Performance/capacity claims include reproducible measurements and correctness evidence, if applicable.
- [ ] Complete diff reviewed; whitespace checked; unrelated work preserved.
- [ ] Remaining limitations and unverified gates are disclosed without overstating confidence.

## 8. Risks, open decisions, and follow-ups

| Risk / question | Impact | Mitigation / decision | Blocking? |
| --- | --- | --- | --- |
| <uncertainty> | <failure mode> | <evidence needed or chosen resolution> | <yes/no> |

Resolve behavior-defining questions before implementation. Separate deferred improvements from
acceptance requirements. Describe rollback/reversion constraints when durable state or external
side effects make reverting nontrivial.

## 9. Execution record and handoff

Keep this separate from the proposed design. Update status and record actual deltas, not an
assumption that every planned step was executed.

- **Implemented:** <Behavior/files changed, or “plan only; no implementation performed.”>
- **Deviations:** <Changes from the approved design and rationale, or none.>
- **Documentation updated:** <Paths, or none with reason.>

| Exact command | Actual result | Notes / blocker |
| --- | --- | --- |
| <command actually run> | <passed / failed / blocked> | <failure detail or evidence location> |

**Not run:** <Gates and concrete reasons; do not label unrun checks as passed.>
**Measurement artifacts:** <Raw logs, comparison results, and benchmark directory, or not applicable.>
**Remaining work:** <Unresolved acceptance items, limitations, or deferred follow-ups.>
