# Plan — Drop Ornith, keep one `run.sh`, catch the RDNA3/Docker branch up with `main`

**Status:** Proposed
**Scope:** `src/qwen35/`, `src/core/qwen35*`, `include/strata/**/qwen35.hpp`, `include/strata/core/model_kind.hpp`, `CMakeLists.txt`, `docker/` (Dockerfile.hip, hfmodel.py, entrypoint-ornith.sh, test_hfmodel_ornith.py), `run.sh`, `run2.sh`, `run3.sh`, `serve/server.py`, `src/kernels/{cpu/pool.cpp,cuda/iq_kernels.cu,native_expert_parity.cpp}`, `AGENTS.md`, docs; the merge of `main` (313 commits) into `feature/rdna3-support`.
**Targets:** Qwen3.8-Flash-Next only (IQ3_XXS default, IQ3_S/IQ2_XS/Q2_0/IQ1_M switchable); HIP/RDNA3 (gfx1101, RX 7700 XT) Docker build/run; Linux; host checks backend-agnostic.
**Related:** supersedes the Ornith track of [../docs/ORNITH_QWEN35MOE.md](../docs/ORNITH_QWEN35MOE.md) (kept as a frozen design record); continues [../docs/DOCKER_GFX1101_PLAN.md](../docs/DOCKER_GFX1101_PLAN.md). No other plan is superseded.

Planning only was requested; no implementation is authorized by this document.

## 1. Goal and requested behavior

`feature/rdna3-support` carries two independent things the user no longer wants together: Ornith/Qwen35MoE model support (to be removed) and the AMD RDNA3 Docker workflow (to be kept and kept working). The branch is also 313 commits behind `main`. After this work the branch should be current with `main`, still build and run Qwen3.8 through Docker exactly as measured in `docs/DOCKER_GFX1101_PLAN.md`, and expose exactly one launcher: `./run.sh`.

- Required: `./run.sh` is the only launcher on the repository root; `./run*.sh` matches one file.
- Required: `./run.sh` reproduces **both** current launchers: `./run.sh` == old `./run.sh` (IQ3_XXS default, 800 expert slots) and `./run.sh --model IQ3_S` == old `./run2.sh` (680 slots on IQ3_S, the same 10 GiB ceiling, the same warnings). `./run.sh --check`, `--dry-run`, `--offline`, `--detach`, `--model`, `--expert-cache`, `--hf-cache`, `--work` keep their current semantics (f79cb28: pinned `-latest` tag, logged resolved image).
- Required: `build.sh`, `docker/Dockerfile.hip(-builder)`, `docker/build-engine.sh`, `entrypoint-hip.sh`, `hfmodel.py`, `bootstrap-model.sh`, `vram-guard.py`, `hipinfo.py` keep their build/run semantics for the Qwen3.8 family, merged with whatever `main` changed in their consumers.
- Required: an Ornith GGUF is refused as an unknown architecture, before any download or load — same observable outcome as today's explicit refusal, minus the run3-specific text.
- Must not change: Qwen3.8 engine behavior, the OpenAI/Anthropic server contract, and the `-latest` image tagging. Qwen3.8 must not regress (AGENTS.md rule; here the rule simplifies to "Qwen3.8 is the only model").
- Must not happen: the branch's `main`-only files are lost in the merge; `docker/test_runtime_contract.py` stops passing; tokenizer identifiers that merely contain the string `qwen35` in `main` are deleted (see §3).

| Starting state | Action / event | Expected outcome | Must not happen |
| --- | --- | --- | --- |
| merged branch, `./run.sh --dry-run` | launcher probe | same docker command as pre-change `./run.sh --dry-run`, image tag resolved and logged | IQ3_S slot table applied to IQ3_XXS; any `entrypoint-ornith.sh` / `strata-qwen35` reference |
| merged branch, `./run.sh --model IQ3_S --dry-run` | launcher probe | same docker command as pre-change `./run2.sh --dry-run` (680 slots, `STRATA_MODEL=IQ3_S`) | 800 slots, raised VRAM budget |
| container serving Qwen3.8 | `POST /v1/chat/completions` | normal streaming response | any change to API contract |
| `./run.sh --model ornith` (or any non-Qwen3.8 release) | launcher validation | refusal naming the shipped quants, before download | container start, partial download |
| engine pointed at an Ornith GGUF | `--check`/load | refusal: architecture not served by this build | a Qwen4Exp-shaped load attempt or wrong-answer inference |

## 2. Current behavior and evidence

Inspected at `05c7fed` on `feature/rdna3-support`; `main` tip diverges at merge-base `1678de3`.

| Evidence | Location / command | Finding |
| --- | --- | --- |
| branch position | `git log main..HEAD` / `git log HEAD..main` | 24 branch commits, `main` is 313 ahead; ~20 branch commits are qwen35/Ornith work, 4 are kept work (RDNA3 `a730750`, quant-quantize parallelism `7a96f2d`, `-latest` pinning `f79cb28`, hfmodel `STRATA_MTP` fix `bd2b09f`) |
| Ornith is branch-only | `git ls-tree -r main | grep -i 'qwen35|ornith'` | none in `main`; only `serve/test_detok.py` / `tools/strata_tokenizer.py` contain `QWEN35_PATTERN`, main's Qwen3.5-family **tokenizer pre-tokenizer** identifier — not model support, must survive |
| merge conflict set | `git merge-tree --write-tree main HEAD` | conflicts in exactly 4 files: `CMakeLists.txt`, `serve/server.py`, `src/kernels/cpu/pool.cpp`, `src/kernels/cuda/iq_kernels.cu`; `docker/`, `run*.sh`, `AGENTS.md` do not exist in `main` or merge cleanly |
| main's churn in conflicted files | `git diff --stat $(git merge-base main HEAD)..main -- <files>` | `server.py` +1694, `iq_kernels.cu` +1278, `CMakeLists.txt` +249, `pool.cpp` ±114 — main rewrote these; branch deltas must be re-applied by hand |
| Ornith code inventory | `git ls-files | grep -iE 'ornith|qwen35'` | 29 tracked files: `src/qwen35/` (7), `src/core/qwen35*.cpp`, `include/strata/{core,qwen35}/qwen35.hpp`, `docker/entrypoint-ornith.sh`, `docker/test_hfmodel_ornith.py`, `run3.sh`, `tools/ornith_*.py`, `docs/ORNITH_QWEN35MOE.md`, `docs/ornith/gguf-layout.txt`, `bench/results/2026-10-02-ornith-rdna3/` (12 files) |
| Ornith references in kept code | `git grep -lEi 'qwen35|ornith'` (kept files) | `AGENTS.md`, `CMakeLists.txt` (qwen35 targets, L151–211, 990–1023), `include/strata/core/model_kind.hpp` (`Qwen35Moe` kind), `docker/Dockerfile.hip` (copies `strata-qwen35`, `entrypoint-ornith.sh`), `docker/hfmodel.py` (`ornith` family, `single_name`/`mtp` machinery), `src/program/generate.cpp` (~L1010 refusal text), `src/kernels/native_expert_parity.cpp` (runtime geometry, generic), `src/kernels/cuda/iq_kernels.cu` (Q4_K-down admission comment) |
| launcher comparison | `diff run.sh run2.sh` (63 diff lines) | run2.sh differs only by: default `MODEL=IQ3_S`, per-quant expert-cache table (`IQ3_XXS`→800, else 680), over-budget warning, and 84/51 GB fallbacks vs 80/43 in the capacity-check awk defaults. No docker-flag differences |
| run3.sh uniqueness | `head run3.sh` | Ornith-only launcher (own container name, work dir, `--mtp`); exists solely to serve the removed model |
| launcher references | `grep -rlE 'run2\.sh|run3\.sh'` | only branch files: `run2.sh`, `run3.sh`, `AGENTS.md`, `docs/ORNITH_QWEN35MOE.md`, `docker/{Dockerfile.hip,entrypoint-ornith.sh}`, plus untracked `ornith-1.5.md`; `docker/README.md` and `test_runtime_contract.py` reference neither |
| untracked local state | `git status --short` | untracked `PLAN_TEMPLATE.md`, `tmp/`, `ornith-1.5.md`, two `bench/results/2026-10-*ornith*` dirs — not part of the change; preserve untouched |

- **Root cause / gap:** the branch grew Ornith and RDNA3 as one line of work; the user now wants only the Docker/RDNA3 result, one launcher, and `main`'s 313 commits.
- **Existing coverage:** `docker/test_runtime_contract.py`, `python -m unittest serve.*`, host CTest (`gguf_reader_test`, `gguf_split_test`, pool tests), and the dry-run fixtures in `bench/results/2026-10-01-discover-128k-10g/` (`launcher-dry-run.txt`, `launcher-defaults.txt`). Nothing proves launcher equivalence after consolidation — that gap is closed by step 1.
- **Baseline:** no GPU-free proof that `run.sh`/`run2.sh` dry-run outputs are byte-stable beyond the checked-in fixtures; `serve/server.py` and `iq_kernels.cu` conflict resolution needs main's rewrites read, not guessed. Untracked files above must be preserved.
- **Environment:** Linux, RX 7700 XT (gfx1101), ROCm via Docker; host checks need only CMake 3.24+/C++20 and Python. 20.1 GB Ornith artifact stays in the user's HF cache untouched; never deleted by this work.

## 3. Scope and non-goals

**In scope:** (1) a single removal commit dropping Ornith/Qwen35MoE support and consolidating `run2.sh` into `run.sh`; (2) merging `main` with the 4 conflicts resolved; (3) doc/AGENTS.md updates; (4) launcher dry-run equivalence checks.

**Out of scope:**
- Deleting Ornith **measurement records**: `bench/results/2026-10-02-ornith-rdna3/` and `docs/ornith/gguf-layout.txt` stay (immutable records; the design doc is archived, see §4).
- Renaming `QWEN35_PATTERN` / `pre = "qwen35"` in `tools/strata_tokenizer.py`, `serve/test_detok.py`, `docker/hfmodel.py` tokenizer defaults — these are main-owned Qwen3.5-family tokenizer identifiers, unrelated to model support.
- Touching the `run3` local variable in `src/core/layer.cpp` (L1214) — a stage-prefix predicate, unrelated to `run3.sh`.
- Re-tuning expert-cache budgets for IQ2_XS/Q2_0/IQ1_M beyond adopting run2.sh's existing table; no engine kernel work; no new GPU benchmarks beyond launcher dry-runs and the existing HIP suite.
- Deleting the local 20 GB Ornith artifacts, `tmp/`, or untracked `ornith-1.5.md`; pushing or opening a PR.

**Affected boundaries:** engine build graph (`CMakeLists.txt`), artifact identity guard (`model_kind.hpp`, `generate.cpp`), model resolver (`docker/hfmodel.py`), container image contents (`Dockerfile.hip`), user entry points (`run.sh`), and this repository's own AGENTS.md/docs. No serve/protocol change; no new dependencies.

## 4. Design and decisions

Order matters: remove first, then merge. Removing against the current, well-tested branch tree produces one self-contained diff with existing tests as evidence; merging `main` afterwards faces the same 4-file conflict set minus all qwen35 hunks, so conflict resolution touches only RDNA3-vs-main deltas.

| Decision | Chosen design | Rationale / rejected alternative |
| --- | --- | --- |
| Removal mechanism | One squashed removal commit (file deletions + reference reverts), not `git revert` of ~20 commits | The reverts would replay and then unwind interleaved shared-file edits (hfmodel.py, CMakeLists); one diff is reviewable against live tests. History stays intact on the branch; the tree is what matters |
| Removal before merge | Yes | Rejected merge-first: `CMakeLists.txt` (249 main lines) and `iq_kernels.cu` (1278 main lines) conflicts would additionally carry qwen35 hunks, and the removal's test evidence would be muddied by 313 unrelated commits |
| `run.sh` consolidation | Adopt `run2.sh`'s body as `run.sh` (it already keeps `run.sh`'s IQ3_XXS tuning), set default `MODEL=IQ3_XXS`; keep per-quant table `IQ3_XXS→800`, others `→680`, plus the over-budget warning | run2.sh is the superset; its comments state the measured rationale (arena 50.3 GB vs 42.9 GB, same 10 GiB ceiling). Rejected: keeping `MODEL=IQ3_S` default — `run.sh`'s documented, measured default is IQ3_XXS (docs, README, fixtures) and the dry-run fixtures pin it |
| Capacity-check fallbacks (80/43 vs 84/51) | Keep fallbacks keyed to the resolved model's `hfmodel.py` JSON; retain 80/43 only as the last-resort default | they are `awk` fallbacks used when the resolver provides no sizes; the dry-run equivalence check pins actual behavior per model |
| `run3.sh` | Delete outright | its entire contract is the removed model |
| `ModelKind` / `model_kind.hpp` | Delete the file, the `Qwen35Moe` kind, and the guards that branch on it (`src/core/qwen35.cpp` guard code goes with it) | Introduced solely by `3b49d89` to separate two architectures; with one architecture it is dead indirection, and `main` never adopted it. Rejected: a one-valued enum that every future merge must wonder about |
| Ornith GGUF refusal | Keep a generic "architecture not served by this build" refusal in `generate.cpp`/the layout guard's unknown-architecture path; reword `4dfedc1` text so it names nothing about run3 | an Ornith GGUF must still be rejected, not mis-loaded as Qwen4Exp; the pre-download launcher refusal (`hfmodel.py` removing the `ornith` family) already refuses `--model ornith` |
| Expert-kernel deltas | Keep `native_expert_parity.cpp`'s runtime expert geometry (generic harness improvement) and `iq_kernels.cu`'s Q4_K admission (Q4_K is used by Unsloth UD-Q4_K_XL on `main` too); delete only the `native_expert_parity_ornith_iq4_xs_q4_K` registration and reword Ornith-specific comments | surgically removing admitted quant paths risks the Qwen3.8 expert path `main` now relies on; the kept deltas are exercised by main's own parity tests |
| hfmodel.py | Delete the `ornith` family entries and `test_hfmodel_ornith.py`; leave the generic single-file/external-MTP resolver helpers in place, unreferenced by Qwen3.8 | removing the mechanism is extra churn in the file `main` consumers touch; `bd2b09f` (never clobber `STRATA_MTP` on eval) is a general fix and stays |
| Design record | `git mv docs/ORNITH_QWEN35MOE.md plans/ornith-qwen35moe-2026-10-frozen.md` with a header noting support was removed and pointing at the kept bench dir | the template's own rule: plans are design records, not claims of current behavior. Rejected: outright delete (loses the measured geometry/parity record the bench dir cites) |
| Dockerfile.hip | Drop `strata-qwen35`, `strata-qwen35-check`, `entrypoint-ornith.sh` COPY/chmod lines; builder stage, engine `strata` binary, entrypoint-hip path untouched | keeps the image build/run semantics identical minus the removed binaries; `test_runtime_contract.py` pins the contract |

### Contracts and invariants

- **Architecture / artifacts:** `ModelKind` mechanism removed; the layout guard keeps rejecting any non-Qwen4Exp architecture with a generic message. Qwen3.8 GGUF metadata/tensor/pack/PLE/MTP contracts unchanged and still the only servable set. `hfmodel.py` ships the Qwen3.8 family only.
- **Numerical correctness:** no kernel math changes (only the ornith parity test registration and comments); `main`'s `iq_kernels.cu` rewrite wins wherever it overlaps, with branch gfx1101 fixes re-applied and proven by main's parity tests + HIP suite.
- **GPU / host memory:** 10 GiB VRAM ceiling and 768 MiB later-slice default unchanged; expert-cache budget becomes a per-quant table inside `run.sh` (800 IQ3_XXS / 680 others) — same measured spend as the two removed launchers.
- **Session / cache state:** unchanged; no serve protocol edits expected in the removal (server.py changes during merge are pure conflict resolution against main).
- **Engine protocol / APIs:** unchanged; refusal-of-unknown-architecture stays before download/load.
- **Setup / launch / packaging:** `build.sh` unchanged; `Dockerfile.hip` image loses two binaries and one script; `run.sh` absorbs `run2.sh`; `run3.sh` deleted; `-latest` pinning and resolved-image logging (f79cb28) preserved; `setup.py`/`START-HERE` paths untouched by this branch and unchanged by the merge.
- **Security / filesystem:** loopback-only bind + `--api-key` rule unchanged; no credentials in evidence; dry-runs write nothing.
- **Web app / lifecycle:** unchanged.

## 5. Ordered implementation steps

Reinspect `git status --short` first; the untracked files listed in §2 stay untouched throughout.

1. **Regression / contract — launcher equivalence pin**
   - Capture `./run.sh --dry-run`, `./run.sh --model IQ3_S --dry-run`, `./run2.sh --dry-run`, `./run2.sh --model IQ3_XXS --dry-run` (pre-change), normalize volatile fields (timestamps, container id), store under `bench/results/<date>-launcher-consolidation/`. Assert `run.sh` vs `run2.sh` cross-equivalence as a small shell/python check run from `docker/test_runtime_contract.py`'s suite or a sibling `docker/test_launcher_contract.py`.
   - Done when: the four captures exist and the cross-equivalence assertion passes pre-change (proving the pin, not the fix).
2. **Removal — Ornith code and wiring**
   - Delete `src/qwen35/`, `src/core/qwen35.cpp`, `src/core/qwen35_layout_test.cpp`, `include/strata/core/qwen35.hpp`, `include/strata/qwen35/`, `tools/ornith_*.py`, `docker/entrypoint-ornith.sh`, `docker/test_hfmodel_ornith.py`; delete `model_kind.hpp` and its `Qwen35Moe`/guard references; strip qwen35 targets from `CMakeLists.txt` (L151–211, 990–1023) including `native_expert_parity_ornith_iq4_xs_q4_K`; drop the `ornith` family from `hfmodel.py`; reword `generate.cpp` refusal; strip `Dockerfile.hip` ornith lines; `git rm -r docs/ORNITH_QWEN35MOE.md` → archive per §4 and update AGENTS.md/docs indexes.
   - Done when: host build + CTest and `python -m unittest discover -s docker -p 'test_*.py'` pass, and `git grep -iE 'qwen35moe|ornith|run3'` matches only the frozen plan record, bench dirs, tokenizer `QWEN35_PATTERN` identifiers, and this plan.
3. **Consolidation — `run.sh` absorbs `run2.sh`; delete `run2.sh`/`run3.sh`**
   - Apply §4's design (run2 body, IQ3_XXS default, per-quant table, fallback keying); update every reference (`AGENTS.md` comes in step 5; `docker/README.md` if touched).
   - Done when: `ls run*.sh` lists only `run.sh`, and the step-1 pin passes post-change for both launch paths.
4. **Merge — `git merge main`**
   - Resolve the 4 conflicts: `CMakeLists.txt` (main's 249 lines + kept branch targets, no qwen35), `serve/server.py` (take main's rewrite; re-apply only branch deltas that survive review — check `frontend.py` interplay), `src/kernels/cpu/pool.cpp` (branch quant-parallelism `7a96f2d` vs main's 114-line rework — main wins unless the parallelization is re-verified by `pool_quant_test`/`pool_topology_test`), `src/kernels/cuda/iq_kernels.cu` (main's rewrite wins; re-apply gfx1101/Q4_K admission deltas, verified by parity tests).
   - Done when: `git log --merges` records it, host + python + docker suites pass on the merged tree, and step-1 pins still hold.
5. **Integration / documentation**
   - AGENTS.md: resolve to main's text plus the RDNA3/Docker workflow with the single `run.sh`; remove the Qwen35MoE paragraph; sweep `docs/` for `run2.sh`/`run3.sh` mentions; note the removal in `docs/DETAILS.md`/`docs/AMD_HIP.md` only where they cite removed artifacts.
   - Done when: `grep -rE 'run2\.sh|run3\.sh'` outside bench/frozen-plan is empty and docs agree with the tree.

## 6. Edge cases and regression matrix

| Scenario | Expected behavior / invariant | Test boundary and test name |
| --- | --- | --- |
| `./run.sh` and `./run.sh --model IQ3_S` after merge | match pre-change `run.sh` / `run2.sh` normalized dry-runs | `docker/test_launcher_contract.py` |
| `./run.sh --model ornith` / unknown quant | refusal naming shipped quants + `STRATA_HF_REPO` escape, no download | launcher dry-run case in the same test |
| `--expert-cache 900` with IQ3_S | warning about the tuned 680, value still passed through (engine caps it) | launcher dry-run case |
| Ornith GGUF to engine `--check` | generic unknown-architecture refusal, nonzero exit | existing guard path via host CLI check; no synthetic ornith fixture retained |
| Qwen3.8 pack/MTP resolution | unchanged two-file release paths, `--mtp` shell-eval fix intact | `docker/test_hfmodel_*.py` (non-ornith), hfmodel existing cases |
| Image contents | `strata-qwen35*` binaries absent; `strata`, `strata-device`, entrypoints present; image tag `-latest` resolved and logged | `docker/test_runtime_contract.py` (extend assertion for absence) |
| `main`'s new server features | survive the `server.py` merge untouched | `python -m unittest serve.test_server serve.test_parser_stream serve.test_lifecycle` |
| Kernel parity after merge | Qwen3.8 IQ-kernels incl. UD-Q4_K_XL paths unchanged vs main | `native_expert_parity` + main's iq tests, HIP build on gfx1101 |
| CUDA vs HIP | HIP fully validated on gfx1101; CUDA build compiles (no CUDA hardware gate claimed) | CMake configure CUDA-on/HIP-off + build |
| repeated merge churn | the merge introduces no orphan qwen35 references | step-2 grep gate re-run post-merge |

GPU/model gates (RX 7700 XT, Docker, HF cache populated) are listed in §7 and are not replaced by host checks; missing fixtures are not passes.

## 7. Verification and acceptance

### Planned commands (not results)

Host checks (no GPU, no downloads):

```bash
python -m unittest discover -s docker -p 'test_*.py'
python -m unittest serve.test_server serve.test_parser_stream serve.test_lifecycle -v
git diff --check
./run.sh --dry-run | tee bench/results/<date>-launcher-consolidation/run-iq3xxs.txt
./run.sh --model IQ3_S --dry-run | tee bench/results/<date>-launcher-consolidation/run-iq3s.txt
```

Host C++ (note: `qwen35_*` targets no longer exist post-removal — their disappearance is part of the check):

```bash
cmake -S . -B build-plan-host -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=ON
cmake --build build-plan-host -j2
ctest --test-dir build-plan-host --output-on-failure --no-tests=error
```

GPU gate (RX 7700 XT, gfx1101; after the merge, since kernels changed):

```bash
./build.sh --tests              # packages + runs HIP suite per docs/AMD_HIP.md
docker run ... strata --check   # against the cached Qwen3.8 artifact
./run.sh && ./run.sh --check    # end-to-end serve, then one streamed chat completion
```

Record blocker details for any gate that cannot run; do not mark unrun gates passed.

### Acceptance checklist

- [ ] Launcher pins: `run.sh` reproduces both removed launchers; `ls run*.sh` → `run.sh` only.
- [ ] Ornith absent from code/build/docs per the step-2 grep gate; design record and bench dirs retained and labeled.
- [ ] Merge with `main` complete; 4 conflicts resolved with main's rewrites preserved; Qwen3.8 regression suite green.
- [ ] Docker image builds; contract test green; refusal paths (launcher, engine) verified.
- [ ] HIP gate on gfx1101 run or its blocker recorded.
- [ ] Diff reviewed; untracked local files untouched; docs (AGENTS.md/DETAILS/AMD_HIP) agree.

## 8. Risks, open decisions, and follow-ups

| Risk / question | Impact | Mitigation / decision | Blocking? |
| --- | --- | --- | --- |
| `serve/server.py` conflict resolution drops a branch delta main lacks | lost Docker-path serve behavior | diff branch deltas (`git diff $(git merge-base main HEAD)..HEAD -- serve/server.py`) line-by-line during resolution; serve unittests after | no (gated by tests) |
| pool.cpp: is `7a96f2d`'s parallel quantization still wanted after main's 114-line rework? | perf regression on CPU quantize, or duplicate logic | default to main's code; re-apply only if `pool_quant_test`/bench shows a gap; record choice in merge commit message | no |
| main may have moved since this plan's `313`-commit count | conflict set shifts | re-run `git merge-tree` at execution time and update the file list before starting | no |
| 680-slot default for IQ2_XS/Q2_0/IQ1_M (they ran 800 under old `run.sh`, 680 under `run2.sh`'s catch-all) | fewer resident experts on those quants via the new default | accepted — matches run2.sh's measured catch-all; the warning text tells users to raise it; flag for a follow-up per-quant measurement | no |
| User may later want Ornith back | history is on this branch | removal is one revertable commit; design record and bench dir kept | no |

## 9. Execution record and handoff

- **Implemented:** plan only; no implementation performed.
- **Deviations:** none yet.
- **Documentation updated:** none yet.

| Exact command | Actual result | Notes / blocker |
| --- | --- | --- |
| (to be filled at execution) | | |

**Not run:** all §7 gates — execution not authorized.
**Measurement artifacts:** n/a; launcher pins will land in `bench/results/<date>-launcher-consolidation/`.
**Remaining work:** execution of steps 1–5; follow-up per-quant expert-cache measurement (§8).
