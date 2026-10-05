# Plan — Optimize the shipped RX 7700 XT deployment using benchmark evidence

**Status:** Proposed
**Scope:** `run.sh` defaults, `docker/Dockerfile.hip` env defaults, `docker/test_launcher_contract.py` pins, engine prefill/cache planning in `src/program/generate.cpp` (measurement-gated only), hipBLASLt tuning table tooling `tools/hip/`, docs (`docs/DETAILS.md`, `docs/AMD_HIP.md`), new results under `bench/results/`.
**Targets:** Qwen3.8-Flash-Next, IQ3_S (the shipped default quant; not changed), HIP backend, gfx1101 (RX 7700 XT, 12,272 MiB), Linux, Docker launch via `./run.sh`, 128K context, 10 GiB VRAM ceiling.
**Related:** `plans/merge-main-drop-ornith-single-run-launcher.md` (launcher consolidation, implemented); `DISCOVER.md` + `bench/results/2026-10-01-discover-128k-10g/README.md` (prior tuning on this host, IQ3_XXS); no superseded plans.

> The user requested planning only for this document; no benchmark, code change, model
> download, or deployment is authorized by it. Authorization in the conversation:
> optimization **may** include code changes/refactor and **may** change `./run.sh`, always
> within the constraints below.

Hard constraints (each is testable, none may be traded for speed):

1. Everything runs through `./run.sh` — no side-launched engines or bespoke containers.
2. Context stays exactly 131,072 tokens; `run.sh` already refuses any other value (line 123).
3. Quantization stays IQ3_S, the shipped default since commit `11a6026`.
4. Strata itself uses at most 10 GiB VRAM (10,240 MiB) including runtime, transients and
   graphs — the room above that (the card has 12,272 MiB) stays for the OS and the GUI.
   The measured contract is per-process attribution (`docker/vram-guard.py --pid`, AMD DRM
   totals), zero tolerance; raw-card sampling (desktop included) is recorded alongside as
   observation. `run.sh` caps `--budget` at 10,240 MiB (line 121) and the HIP allocation
   guard admits at most 8,960 MiB of tracked allocations (1,024 MiB runtime reserve +
   256 MiB slack).
5. Nothing new on the NVMe drive: the root filesystem (`/dev/nvme1n1p2`, 915 GB) has 78 GB
   free and cannot hold a ~76 GB quant. All model, pack, and benchmark data stay on
   `/mnt/storage` (4 TB, 3.4 TB free) via `~/Development` — which is already what `run.sh`
   defaults to (`DEFAULT_MODEL_DIR=$HOME/Development/models`, work dir beside it).
6. (Added during execution, user request) Swift 1.5 Flash-Next GSQ-RCO IQ3_XXS is measured
   as a **comparison arm** through the same `./run.sh` path (`-e STRATA_HF_REPO=... -e
   STRATA_PACK_DIR=/work/packs/swift-iq3_xxs`, no pack collision). It never replaces the
   IQ3_S shipping default; its numbers are reported as a different model+quant.

## Repository context

- Engine: `src/` + `include/strata/`; HIP build backend; cache/auto sizing lives in
  `src/program/generate.cpp` (`--expert-cache auto` around line 422/3212, `--prefill auto`
  line 432, free-room clamp line 3313). See `docs/DETAILS.md`, `docs/AMD_HIP.md`.
- Launch: `run.sh` is the single launcher (`run2.sh`/`run3.sh` were removed in `95404ab`);
  its per-quant expert-cache defaults and warnings are pinned by `docker/test_launcher_contract.py`;
  the image carries matching defaults in `docker/Dockerfile.hip` (ENV `STRATA_EXPERT_CACHE=800`,
  `STRATA_PREFILL=2048`, `STRATA_POOL_WORKERS=0`).
- Bench tooling retained from the DISCOVER work: `tools/hip/bench_discover.py` (real-token
  streaming/TTFT probes with follow-ups), `tools/hip/bench_prefill.py` (alternating
  4,210/8,830-token coding prompts), `tools/hip/check_coding_task.py` (205-case completion
  smoke test), `tools/hip/tune_hipblaslt.cpp`, and the VRAM sampler pattern from
  `bench/results/2026-10-01-discover-128k-10g/` (50 ms, zero tolerance, no desktop subtraction).

## 1. Goal and requested behavior

`./run.sh` on this machine (RX 7700 XT, gfx1101, 10 GiB ceiling, 128K, IQ3_S) should start
the configuration with the best measured prefill/decode throughput on this host, with every
default backed by a measurement taken on this hardware — not by arithmetic. The one prior
tuned measurement on this card (`bench/results/2026-10-01-discover-128k-10g`) was made on
IQ3_XXS and never completed its final gates; the shipped default has since switched to IQ3_S,
whose expert-cache budget (680 slots) was scaled from IQ3_XXS's 800 by the ~17% blob-size
difference and has no measurement behind it.

- Required: after this plan, `./run.sh` (no flags) starts the fastest configuration that
  passes all four hard constraints, and its defaults are stated with numbers measured on this
  host and stored under `bench/results/`.
- Required: the 10 GiB ceiling on Strata's own share holds under load with zero tolerance —
  per-process DRM attribution; raw-card sampling recorded alongside.
- Preserved: context 131,072; IQ3_S artifacts; launch through `./run.sh` only; all data on
  `/mnt/storage`; Qwen3.8 behavior on other quants/backends; the guard label check that
  refuses pre-guard images.

| Starting state | Action / event | Expected outcome | Must not happen |
| --- | --- | --- | --- |
| Server stopped, IQ3_S cached on `/mnt/storage` | `./run.sh --detach`, then `./run.sh --check` | model reports `loaded`, `ctx=131072` | any VRAM reading > 10,240 MiB; any byte written to `~/.cache/huggingface` or root disk beyond Docker's own state |
| Server running at tuned defaults | `tools/hip/bench_prefill.py` coding workload | prefill ≥ baseline arm, decode ≥ baseline arm (thresholds in §7) | silently reduced chunk, cache, or context to hit the number |
| 130,944-token fresh prompt | follow-up continuation request | positive prompt-cache reuse in engine log | truncation, context growth, OOM |
| Explicit `--expert-cache` above the tuned value | `./run.sh --expert-cache 900` | warning names the tuned value, value passes through, engine clamps under the guard | ceiling raise |

## 2. Current behavior and evidence

| Evidence | Location / command | Finding |
| --- | --- | --- |
| This machine | `docker/hipinfo.py`, `lscpu`, `free`, `df`, `lsblk` | gfx1101, 12,272 MiB VRAM; Ryzen 9 7900 (12 C / 24 T); 122 GiB RAM (~102 GiB available); host ROCm 7.2.4 |
| NVMe space | `df /` | `/dev/nvme1n1p2` 915 GB, **78 GB free (92 % used)** — cannot hold the IQ3_S quant or its pack |
| Big filesystem | `readlink -f ~/Development`, `df` | `~/Development` → `/mnt/storage/Development` (ext4, 4 TB, 3.4 TB free); repo, HF cache, and work dir all live there |
| Artifacts present | `ls ~/Development/strata-work/packs/`, `./run.sh --check` | IQ3_S pack (49 GB incl. `experts.bin`) built; "IQ3_S is cached (both shards)"; server currently down (port 9931 refused) |
| Image present | `docker images` | `strata-hip:gfx1101-latest` (sha `db59c9d2d225`, commit `0d8388a`) with the guard label (run.sh's check passes at build time) |
| Prior tuning, this host | `bench/results/2026-10-01-discover-128k-10g/README.md` | IQ3_XXS, 128K, 10 GiB: tuned arm (11 workers, chunk 2048, 971 slots) reached **253.7 fresh-prefill / 30.2 decode tok/s** vs control 126.4 / 18.0; TTFT 1K 6.33→4.88 s, 4K 23.2→14.7 s; raw-card peaks 9,802.70 / 9,819.58 MiB passed the 10 GiB limit |
| Untuned default | `run.sh` lines 124–137 | IQ3_S expert cache **680 = 800 ÷ 1.17 by arithmetic**; never benchmarked on this host |
| Unfinished gates | same README, "Remaining validation" | 32,768 / 65,536 / 130,944-token probes, the 2K-arm raw/process VRAM audit, and packaged-default validation were **blocked by sandbox and remain unrun**; no `DONE` |
| Rejected experiments | same README tables | routing overlap (<0.1 % of timeline), hipBLASLt descriptor reuse (within noise), AMD pooled-key scoring reuse (50–70 % regression at long positions) — restored, do not retry |
| Prefill planner limit | same README, memory ledger | 2,048-token scratch = 1,280,477,952 B **borrowed from cache**; 4096/6144/8192 chunks **fail the exact planner** for the 971-slot IQ3_XXS cache — smaller caches were not swept |
| Engine auto sizing exists | `src/program/generate.cpp` 422, 3212–3314, 3385–3395 | `--expert-cache auto` sizes from free room under the reserve, with a shrink-and-retry path (`auto_cache && failed < 8`); `--prefill auto` picks the largest chunk the cache can lend, up to 8192 |
| Tuning pass-through | `run.sh` lines 326–330 | `STRATA_PREFILL_RING`, `STRATA_STAGER_RING/THREADS`, `STRATA_IO_THREADS`, `STRATA_HIPBLASLT_TUNING`, `STRATA_PREFILL_TIMING`, `STRATA_PREFILL_MEMORY_REPORT`, `STRATA_PREFILL_LEND_PCT` are forwarded; `STRATA_EXPERT_CACHE` accepts `auto` |
| Lt table is not production | `bench/results/2026-10-01-discover-128k-10g/gfx1101-lt.tsv` (4 rows) | described in the README as "a focused test artifact, not production tuning for all model shapes" |
| Contract pins | `docker/test_launcher_contract.py` lines 78–96 | asserts `STRATA_EXPERT_CACHE=800` (IQ3_XXS) and `680` (IQ3_S), and the "tuned 680" warning — any changed default must move these pins in the same change |

- **Root cause / gap:** the shipped default's performance knobs (680 slots, chunk 2048,
  auto workers) are inherited or scaled, not measured, on the only quant (`IQ3_S`) this
  machine now ships; and the one measured configuration was never validated at 32K–131K
  context or finalized for VRAM on its 2K arm.
- **Existing coverage:** launcher contract tests prove the *shape* of the launch line
  (values, warnings, refusals) but no test measures throughput or VRAM; DISCOVER harnesses
  exist and work but have only run against IQ3_XXS.
- **Baseline:** worktree clean except untracked `PLAN_TEMPLATE.md`; commit `11a6026`.
  Setup tests (`python tools/test_setup_*.py`, `python -m unittest discover -s docker`)
  were last reported green by `bench/results/2026-10-04-launcher-consolidation/README.md`;
  re-run before execution, not assumed.
- **Environment:** as §2 rows; engine image = `strata-hip:gfx1101-latest`; KV int8, MTP,
  speculation window 4 (per `control-config.json`/`candidate-config.json`); greedy sampling
  in benchmarks.

## 3. Scope and non-goals

**In scope:** measurement of the shipped `./run.sh` configuration on this host; a bounded
sweep of the launch-exposed knobs (`--expert-cache N|auto`, `--prefill N|auto`,
`--pool-workers`, forwarded ring/lend variables); measurement-gated code changes to the
cache/prefill planner in `src/program/generate.cpp` and to hipBLASLt production tuning
(`tools/hip/tune_hipblaslt.cpp` + table loading); updating `run.sh` / `Dockerfile.hip`
defaults and their contract tests; closing the unfinished 128K/VRAM gates for the final
configuration; documenting results.

**Out of scope:**

- Quantization changes (constraint), context changes (constraint), VRAM ceiling changes
  (constraint), moving any data onto the NVMe drive (constraint).
- Re-runs of the three rejected DISCOVER experiments (routing overlap, Lt descriptor reuse,
  pooled-key scoring reuse) — they failed on this host with raw samples; reopening needs new
  evidence, not optimism.
- AMD matrix attention / GPU grouping and top-k dispatch changes — deferred there with
  reasons; decode here is already ~30 tok/s, and the kernel work is substantial.
- Docker image tag cleanup on NVMe (frees space but touches user state; listed as a risk).
- CUDA/gfx1100 execution: this host has no NVIDIA GPU and only gfx1101 — compile checks
  only, disclosed as unverified gates.

**Affected boundaries:** launcher (`run.sh`) and its contract test; image defaults
(`docker/Dockerfile.hip`, entrypoint); engine cache/prefill planning (only if a sweep shows
the planner — not the budget — is the binding constraint); docs tables in `docs/DETAILS.md`
/ `docs/AMD_HIP.md`.

## 4. Design and decisions

Path per run: `run.sh` (card check → guard label → docker run with env) → entrypoint
generates the engine config → engine sizes expert cache under the guard → prefill chunks
borrow scratch from the cache → decode with MTP. Owners: `run.sh` owns defaults, the engine
owns clamping and admission, the guard owns the ceiling.

| Decision | Chosen design | Rationale / rejected alternative |
| --- | --- | --- |
| What to measure first | The shipped default as-is (IQ3_S, 680, chunk 2048, workers 0) via `./run.sh --detach` + DISCOVER harnesses | First-ever measurement of the shipping configuration; everything else is a candidate delta against it |
| Cache budget choice | Sweep 680 vs `auto` vs explicit values above 680; keep the value the guard *actually admits* (engine log records the clamp), and prefer `auto` if it lands within noise of the best fixed arm and survives the shrink-retry path | 680 is arithmetic; the IQ3_XXS data (971 > 671 slots, faster) shows "tuned" defaults can underfill the ceiling — but only measurement says where 10 GiB actually binds for IQ3_S's wider slots |
| Prefill chunk | Sweep 2048 vs 4096 vs `auto` at the chosen cache; 4096 failed the exact planner at 971 IQ3_XXS slots, a smaller IQ3_S cache may admit it; `STRATA_PREFILL_MEMORY_REPORT` traces each arm | Chunk was the largest single win before (126→254 tok/s); the planner, not hardware, was the gate |
| Code change gate | Only change planner code if a memory report shows a *planner* rejection of a chunk that fits the 10 GiB guard after lending; the change is then in the exact-buffer/lend logic (`--prefill auto`, lend budget around lines 4744/4834), with `STRATA_PREFILL_LEND_PCT` as the measurement probe first | Refactoring the planner without a trace showing it binding is guessing; the trace is one env var away |
| hipBLASLt | Regenerate a production table for IQ3_S shapes on this host with `tools/hip/tune_hipblaslt.cpp`; ship it in the image, loaded via the existing `STRATA_HIPBLASLT_TUNING`; keep the per-call-descriptor design (rejected change) | The current gfx1101 table has 4 rows and self-declares as a test artifact; the loader and arch/version guard already exist |
| Workers | Keep automatic (11 physical); spot-check `--pool-workers 6` only if decode regresses against the 29.8–30.2 tok/s prior | 0→auto was already the measured win; no signal says otherwise |
| Default mechanism | If a tuned value wins: bake it into `run.sh` per-quant case, `Dockerfile.hip` ENV, and the contract test in one commit; keep the "above tuned" warning semantics | Three sources of the same default already drift-prone; the contract test exists to catch exactly that |
| Bench placement | `./run.sh` with `--work /mnt/storage/.../strata-work` (default) and a dedicated port (19931) via `-p`; harness output JSON to `bench/results/<date>-gfx1101-iq3s-tuning/` inside the repo (on `/mnt/storage`) | Repo is on the big disk; NVMe gets nothing but Docker's own image state |

### Contracts and invariants

- **Architecture / artifacts:** unchanged — `ModelKind`, IQ3_S pack (`experts.bin` 49 GB at
  `/mnt/storage/.../packs/iq3_s`), tokenizer, MTP draft layer, int8 KV. No re-packing;
  pack reuse is a prerequisite check, not a step.
- **Numerical correctness:** greedy sampling in all probes; completed-coding smoke test
  (205 cases) must pass in every retained arm; any planner change keeps the exact-buffer
  planner as the admission authority (never admit a chunk the guard cannot hold).
- **GPU / host memory:** 10,240 MiB ceiling on Strata's own share, proven by
  `vram-guard.py --pid` with zero tolerance; raw-card sampling (desktop included) recorded
  as observation; the guard caps tracked allocations at 8,960 MiB (1,024 MiB runtime
  reserve, 256 MiB slack) and `STRATA_VRAM_LATER_MIB` (default 768) is the launch knob for
  the "later explicit buffers" allowance inside that guard; scratch borrows from cache,
  never adds; iGPU stays hidden (`HIP_VISIBLE_DEVICES=0`), `HSA_OVERRIDE_GFX_VERSION` unset.
- **Session / cache state:** unchanged engine semantics; follow-up probes must show positive
  prompt-cache reuse at 128K positions.
- **Engine protocol / APIs:** unchanged CLI surface; `run.sh` keeps refusing
  `--max-context != 131072` and `--budget > 10240`; tuned-value warning keeps its wording
  with the new number.
- **Setup / launch / packaging:** `run.sh` may change only defaults and the tuned-value
  warning text; `Dockerfile.hip` ENV must agree; guard-label check preserved; new image
  tagged, old tags left alone (NVMe pressure noted as risk, not cleanup scope).
- **Security / filesystem:** bind stays `127.0.0.1`; bench token counts come from the pack
  tokenizer, not guessed; no credentials in logs; owned temp files only.
- **Web app / lifecycle:** untouched; server start/stop via `run.sh --detach` / graceful
  `docker stop` between arms, with a settled VRAM reading between runs so arms never share
  the card.

## 5. Ordered implementation steps

1. **Baseline — shipped default on this host**
   - `git status --short` clean check; `python -m unittest discover -s docker -v` and
     `python tools/test_setup_choices.py` green first.
   - `./run.sh --detach`, wait for `./run.sh --check` to report `loaded ctx=131072`;
     sample raw-card VRAM at 50 ms through startup and the workload.
   - Run `bench_prefill.py` (alternating 4,210/8,830 prompts, 128-token caps) and
     `bench_discover.py --sizes 1024,4096 --repetitions 5 --followups`; save
     `*-config.json`, engine logs, VRAM JSON to the new `bench/results/` dir.
   - Done when: fresh prefill/decode, follow-up decode, TTFT, and raw-card peak medians
     exist for the exact shipping line `./run.sh`.
2. **Cache-budget sweep — launch-exposed only**
   - Arms: `--expert-cache auto`, and fixed values bracketing 680 up to the value the
     engine admits (read the clamp from each engine log); one variable per run; server
     restarted between arms.
   - Done when: one table maps budget → admitted slots → decode/follow-up tok/s → raw-card
     peak, each arm under 10,240 MiB.
3. **Prefill-chunk sweep — gated on the memory report**
   - At the chosen budget: arms 2048, 4096 (if planner admits), and `auto`, each with
     `STRATA_PREFILL_MEMORY_REPORT=1`; record planner verdicts, not just throughput.
   - Done when: the binding constraint for 4096 (planner vs scratch vs guard) is named from
     the trace, and the chunk winner is recorded — or the planner rejection is captured as
     evidence for step 4.
4. **Code changes (only if step 3 shows them binding)**
   - (a) planner/lend change in `src/program/generate.cpp` to admit a chunk the guard can
     hold, extending the existing exact-planner check rather than bypassing it; add/extend a
     focused CTest in the HIP suite; rebuild via `./build.sh`, new tag, run.sh picks
     `-latest`.
   - (b) hipBLASLt production table: `tune_hipblaslt.cpp` over the IQ3_S shapes on this
     card; validate the table loads with zero fallbacks on repeated calls, A/B with and
     without via `STRATA_HIPBLASLT_TUNING`.
   - Done when: each change individually beats its gate arm without regressing decode or
     the smoke test, and the HIP CTest suite passes.
5. **Defaults + consumers**
   - Update the per-quant tuned value in `run.sh`, `Dockerfile.hip` ENV, and
     `docker/test_launcher_contract.py` pins (warning text too) in one commit; update
     `docs/DETAILS.md` / `docs/AMD_HIP.md` with measured numbers attributed to this host.
   - Done when: `--dry-run` output equals the measured winning line byte-for-byte and the
     contract suite is green.
6. **Final gates for the shipping configuration**
   - Long-context probes at 32,768 / 65,536 / 130,944 real tokens with follow-ups and
     cancellation/recovery, raw-card + process guards sampling; finalize the raw/process
     VRAM audit; `check_coding_task.py` 205/205; graceful engine quit with the final tracked
     peak saved.
   - Done when: all gates pass or each failure is recorded as a blocker with its log.

Reinspect `git status --short` before each step; preserve unrelated work. Planning alone
authorizes none of the above — GPU time on this machine and the (already-present) artifacts
must be confirmed as available when execution is authorized.

## 6. Edge cases and regression matrix

| Scenario | Expected behavior / invariant | Test boundary and test name |
| --- | --- | --- |
| `./run.sh` with no flags | Starts IQ3_S, ctx=131072, budget 10240, measured expert-cache default | `docker/test_launcher_contract.py` (updated pins) |
| `--max-context 65536`, `--budget 12288` | Refused before docker, exit 1 | existing contract tests (preserved) |
| `--expert-cache <tuned+100>` | Warning names the tuned value; value passes through; engine clamps under guard | contract test (reworded) + engine-log evidence |
| `auto` cache with shrink-retry | Retry path lands on an admitted size, no OOM, ceiling never breached | engine log + raw-card audit |
| Chunk 4096 under exact planner | Admitted with trace, or refused with the engine running unchanged at 2048 — never a mid-request OOM | `STRATA_PREFILL_MEMORY_REPORT` trace + smoke test |
| 130,944-token prompt + follow-up | Accepted inside 131,072 incl. template + output room; follow-up shows positive reuse; Strata share ≤ 10,240 MiB at 50 ms zero tolerance (raw card recorded) | `bench_discover.py` long-context arm |
| Cancellation mid-prompt at 64K | Terminal outcome, buffers returned to cache, next request correct | bench probe + engine log |
| Restart between arms | VRAM settles to idle baseline before the next arm; no arm shares the card | VRAM sampler gaps |
| Greedy token drift vs prior arms | Differences explained only by documented CPU partitioning; smoke test 205/205 every retained arm | `check_coding_task.py` |
| IQ3_XXS / Q2_0 / Coder via `--model` | Their tuned values and the `run.sh` path unchanged | contract tests for non-IQ3_S quants |
| gfx1100 compile, CUDA compile | Still compile; execution explicitly unverified (no hardware) | `./build.sh --arch gfx1100` compile check; disclosed gap |
| Pack or shards missing on `/mnt/storage` | `run.sh` space gate fires; nothing downloaded to NVMe | existing gate + `df` check before/after |

## 7. Verification and acceptance

### Planned commands (not results)

Host checks (no GPU needed):

```bash
python -m unittest discover -s docker -v
python tools/test_setup_choices.py
git diff --check
```

Baseline and sweeps (server on the dedicated port; results to
`bench/results/2026-10-<dd>-gfx1101-iq3s-tuning/`):

```bash
./run.sh --fresh --detach && ./run.sh --check
python3 tools/hip/bench_prefill.py ...            # saved flags per prior README
python3 tools/hip/bench_discover.py --url http://127.0.0.1:19931 \
  --model strata --tokenizer /mnt/storage/Development/strata-work/packs/iq3_s/tokenizer \
  --sizes 1024,4096 --repetitions 5 --followups ...
# per-arm: --expert-cache / --prefill / -e STRATA_PREFILL_MEMORY_REPORT=1 via ./run.sh
```

Final gates (exact long-context form preserved from the prior README's "Remaining
validation"): sizes 32768,65536,130944 with `--followups --repetitions 1 --timeout 7200`,
`check_coding_task.py`, 50 ms raw-card + per-process sampling. HIP suite per
`docs/AMD_HIP.md` (`./build.sh --tests`) if any engine code changes; CTest inventory first
via `ctest --test-dir <dir> -N`, `--no-tests=error`. `ple_parity`, `expert_parity`,
`pool_test` stay fixture-unavailable (Q2_0-era pack absent) — reported as unavailable,
never as passing.

### Measurement plan

Same model (IQ3_S), hardware, prompts, seeds, warmup-excluded five repeats; only one knob
varies per arm; medians reported; every arm raw-card-sampled.

| Item | Required evidence |
| --- | --- |
| Environment | Commit, image tag/sha, ROCm 7.2.4, gfx1101 12,272 MiB, Ryzen 9 7900, 122 GiB RAM, `/mnt/storage` storage |
| Model/settings | IQ3_S pack paths, int8 KV, MTP, window 4, greedy, chunk/budget/workers per arm from each `config.json` |
| Workload | 4,210/8,830-token coding prompts, 5 fresh + 5 follow-ups, excluded warmup; long-context single-shot probes |
| Results | Prefill and decode tok/s separately, TTFT, raw-card peak MiB + sample count, tracked-byte peak, smoke-test result, raw logs |

Acceptance thresholds set before measuring: the shipping candidate must beat the step-1
baseline by ≥ 5 % fresh-prefill median tok/s with decode regression ≤ 2 % (or be rejected
unchanged); Strata's attributed share < 10,240 MiB with zero violating 50 ms samples (raw
card recorded alongside); smoke 205/205;
follow-up reuse > 0 at every size. No claim is made from the 1K-chunk or IQ3_XXS numbers.

### Acceptance checklist

- [ ] Shipping `./run.sh` line equals the measured winning configuration (dry-run diff).
- [ ] All four hard constraints hold under load with zero-tolerance raw-card evidence.
- [ ] Engine code changes (if any) pass focused CTest and never bypass exact-planner admission.
- [ ] Contract tests, image defaults, and docs carry the same numbers with attribution.
- [ ] Long-context, cancellation, smoke, and VRAM gates passed or explicitly blocked.
- [ ] Unverified gates (gfx1100/CUDA execution, three fixture tests) disclosed.
- [ ] Complete diff reviewed; `bench/results/` holds raw logs; unrelated work preserved.

## 8. Risks, open decisions, and follow-ups

| Risk / question | Impact | Mitigation / decision | Blocking? |
| --- | --- | --- | --- |
| Docker image state on the 78 GB NVMe; a rebuild adds ~22 GB | Disk fills mid-build, breaking docker for everything | Check `df /` before any build; with the user, prune oldest commit tags first (their call — user state) | yes, before step 4a |
| Prior README was written under a sandbox that later blocked Docker | Some recorded states (2K audit, container cleanup) may be stale | Re-check live container state at step 1; treat prior final numbers as IQ3_XXS-only context, not baseline | no |
| `auto` may not land on the best fixed budget (reserve-adaptive sizing) | A tuned fixed number beats `auto` | Decide from step-2 data; either is acceptable if measured | no |
| IQ3_S greedy outputs may differ from prior IQ3_XXS arms | Cross-quant comparisons are invalid | All comparisons stay within-IQ3_S arms | no |
| Planner change could admit a chunk that transients push over 10 GiB | Ceiling breach — the unforgivable failure | Guard cap + raw-card gate on every arm with the change; ship only if audit passes | gates step 4a |
| 128K probe may expose long-position regressions the isolated QSA test hinted at | Decode collapses at long positions | The 32K/64K/128K ladder is the detector; no QSA kernel changes in scope | gates step 6 |
| Follow-up: image tag housekeeping; production Lt table for other quants | Space/tuning debt | Deferred; recorded here | no |

## 9. Execution record and handoff

- **Implemented:** plan only; no implementation performed.
- **Deviations:** none.
- **Documentation updated:** none yet (this file only).

| Exact command | Actual result | Notes / blocker |
| --- | --- | --- |
| `docker/hipinfo.py`, `df`, `lsblk`, `./run.sh --check` (evidence pass) | passed | facts recorded in §2 |

**Not run:** every benchmark and gate above — the user authorized planning, not execution.
**Measurement artifacts:** to be created under `bench/results/2026-10-<dd>-gfx1101-iq3s-tuning/`.
**Remaining work:** all of §5; execution authorization required.
