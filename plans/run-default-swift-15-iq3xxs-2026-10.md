# Plan — Make ./run.sh start Swift 1.5 (IQ3_XXS) by default on the gfx1101 container

**Status:** Implemented (2026-10-05)
**Scope:** `run.sh`, `docker/hfmodel.py`, `docker/entrypoint-hip.sh`, `docker/bootstrap-model.sh`,
`docker/Dockerfile.hip`, `docker/test_launcher_contract.py`, `docker/test_runtime_contract.py`,
`docker/test_hfmodel_ple.py` (extension), docs: `docs/DOCKER_GFX1101_PLAN.md`,
`docs/AMD_HIP.md` (container section), `bench/results/2026-10-05-run-default-swift/` (new)
**Targets:** HIP backend, gfx1101 (RX 7700 XT, 12 272 MiB), Linux container; Swift 1.5
(`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`) IQ3_XXS as the new launch default; Qwen3.8
IQ3_S/IQ3_XXS/IQ2_XS/Q2_0/IQ1_M must keep launching exactly as today
**Related:** follows `docs/DOCKER_GFX1101_PLAN.md` Part H items 8-9 and
`bench/results/2026-10-04-iq3s-tuning/` (the IQ3_S tuning and the Swift comparison that motivate
this); partially executes `plans/gfx1101-iq3s-10g-benchmark-optimization.md` (its tuning legs
landed 2026-10-04 as commit `3b128a0`; its hipBLASLt leg stays open and is not needed here);
prerequisites already in the tree: `3b128a0` (tuned IQ3_S defaults) and `1387fdf` (per-family PLE
file); supersedes nothing

> Planning only: this document authorizes no download, image rebuild, or default flip. Section 5
> is executable once approved. AGENTS.md read; links adjusted for `plans/`.

## Repository context

Verified against the current worktree (commit `1387fdf`), not against old plans:

- Launcher: `run.sh` (364 lines) + `docker/` (entrypoint, bootstrap, `hfmodel.py`, `hipinfo.py`,
  contracts). `run2.sh`/`run3.sh` no longer exist; the Qwen35MoE/Ornith notes in the template do
  not apply to this change.
- The 10 GiB VRAM contract and its guards: `docker/hipinfo.py`, `docker/vram-guard.py`,
  `docs/DOCKER_GFX1101_PLAN.md` "The 10 GiB VRAM contract" and Part H items 3 and 8.
- Measured Swift evidence: `bench/results/2026-10-04-iq3s-tuning/` (README has the full matrix;
  `swift-*.json` are the Swift arm).

## 1. Goal and requested behavior

Yesterday's comparison (same card, same tuned knobs) measured Swift IQ3_XXS ahead of the shipped
IQ3_S line on every speed axis while *lowering* Strata's VRAM share: fresh prefill 252.9 vs
235.1 tok/s, decode 33.5 vs 30.0, TTFT@4K 14.84 vs 16.27 s, share peak 8,696 vs 8,723 MiB, coding
smoke 205/205 both. The request: `./run.sh` with no arguments should start **that** model —
Swift 1.5 at IQ3_XXS — and Qwen3.8 must remain one flag away, unchanged.

- `./run.sh` (no args) starts `swift-1.5-iq3_xxs`: health/`/v1/models` report exactly that id,
  from packs under `packs/swift-iq3_xxs`, with `STRATA_EXPERT_CACHE=auto`,
  `STRATA_VRAM_LATER_MIB=700`, `STRATA_PREFILL_RING=48` (the measured combination), under the
  same 10 240 MiB ceiling, which is **not** raised.
- `./run.sh --release qwen --model IQ3_S` reproduces today's shipping line **byte for byte**
  (fixture diff against `bench/results/2026-10-04-iq3s-tuning/post-tune-iq3s.txt`).
- `./run.sh --release qwen --model IQ3_XXS` reproduces the pre-change pin byte for byte
  (`../2026-10-04-launcher-consolidation/pre-run-iq3xxs.txt`).
- A pack built for one release is never loaded for another (see §4 invariant); the launcher or
  bootstrap must reject the mismatch, not run it.
- The startup log names the release and its license (Swift Open License 1.0) before starting.

| Starting state | Action / event | Expected outcome | Must not happen |
| --- | --- | --- | --- |
| clean shell, packs as on this host today | `./run.sh --dry-run` | docker line names swift repo, `STRATA_MODEL=IQ3_XXS`, pack dir + model name `swift-1.5-iq3_xxs`, cache auto / later 700 / ring 48 | qwen paths appear in the line; budget > 10 240 |
| `packs/iq3_xxs` holds the Qwen pack | `./run.sh` (swift default) | uses `packs/swift-iq3_xxs`; if that pack is absent, builds it from the Swift shards | the Qwen pack is read as Swift experts (silent quality corruption) |
| swift server running | `./run.sh --check` | lists `swift-1.5-iq3_xxs  ready/loaded` | reports a stale qwen id |
| swift server running | `./run.sh --offline --check-only` | inspects model/pack without download, exit 0 | port collision with the running server |
| `--model IQ1_M` (Coder's quant) | `--release qwen --model IQ1_M` | refused: quant belongs to the coder release | launches qwen with a Coder quant |

## 2. Current behavior and evidence

| Evidence | Location / command | Finding |
| --- | --- | --- |
| default quant and per-quant tuning pins | `run.sh:33`, `run.sh:132-136` | `MODEL=${STRATA_MODEL:-IQ3_S}`; tuning keyed on quant alone: `IQ3_S -> auto/700/48` (measured), `IQ3_XXS -> 800/768/-`, else `680/768/-` |
| no release/family concept in the launcher | `run.sh:153-157`, `run.sh:217` | unknown quants pass only with a hand-set `$STRATA_HF_REPO`; the space/cached check calls `hfmodel.py --model <quant>` **without `--repo`**, so it always resolves the Qwen family |
| pack dir is quant-only | `docker/entrypoint-hip.sh:77` | `STRATA_PACK_DIR` default `packs/<quant lower>` — two releases of one quant collide |
| model id is qwen-hardcoded | `docker/entrypoint-hip.sh:100` | default `qwen3.8-flash-next-<quant>`; Swift would advertise a wrong API model id |
| bootstrap check is provenance-blind | `docker/bootstrap-model.sh:84-108` | pack check = `experts.bin` + `tokenizer/vocab.json` exist; nothing ties the pack to the release it was packed from |
| provenance is available | `packs/swift-iq3_xxs/experts.bin.src.json` | records shard names, sizes, `native_experts_sha256`; the older `packs/iq3_s` pack also carries it (`packs/iq3_xxs` does not - it predates the file) |
| per-family PLE already fixed | `docker/hfmodel.py` FAMILIES `"ple"`, `--print ple`, `STRATA_PLE_FILE`; `docker/test_hfmodel_ple.py` | Swift PLE resolves to shard 1, qwen to shard 2; 16/16 docker tests green (`python -m unittest discover -s docker`) |
| release names/tags exist in `setup.py` | `setup.py` FAMILIES (`swift-` tag, `"name": "swift-1.5"`, license URL field) | hfmodel.py's FAMILIES has no `name`/`tag`/`license` fields yet — the sync point this plan extends |
| Swift arm measured, gated partially | `bench/results/2026-10-04-iq3s-tuning/swift-*.{json,log}`, README | 252.9 / 33.5 / 14.84 / 8,696 PASS / smoke 205/205 / hit rate 12.4%; launched with explicit `-e` overrides; **128K ladder and cancel probe were NOT run for Swift** (only for the IQ3_S shipping line) |
| Swift starts with zero overrides today | same, `swift-relaunch.log` + `swift-nooverride-smoke.json` | after `1387fdf` + `./build.sh --runtime-only`: READY, PLE from shard 1, smoke 205/205 |
| launcher contract fixtures | `docker/test_launcher_contract.py` | default today must equal `post-tune-iq3s.txt` (IQ3_S); explicit IQ3_XXS equals the consolidation pin |

- **Root cause / gap:** the launcher equates "model" with "quant". Everything release-shaped —
  repo, pack dir, advertised model id, tuning pins, download numbers — is either qwen-hardcoded
  or hand-passed via `-e`. Switching the default without fixing the pack-dir collision and the
  id derivation would create a silent-wrong-pack path and a lying API id.
- **Existing coverage:** the two contract tests pin the docker `-e` line byte-for-byte from
  fixtures; they miss release-shape behavior because none exists yet.
- **Baseline:** `tools/test_setup_amd.py::WindowsDetection::test_prebuilt_hip_zip` fails at
  HEAD~3 identically (pre-existing, unrelated — do not fix here, do not let it block).
  Working tree is clean except untracked `PLAN_TEMPLATE.md` (user's).
- **Environment:** commit `1387fdf`; RX 7700 XT (gfx1101, 12 272 MiB), ROCm 7.2.1 container
  image `strata-hip:gfx1101-latest` (built `3b128a0`+tree); host cache
  `~/Development/models` holds both repos (Swift IQ3_XXS both shards, snapshot
  `b22d729e…`); work dir holds `packs/iq3_s`, `packs/iq3_xxs` (Qwen), `packs/swift-iq3_xxs`,
  `mtp/rt`.

## 3. Scope and non-goals

**In scope:** a release axis in the launcher (`--release`), release-aware defaults (repo, quant,
pack dir, model id, tuning pins), the pack-provenance guard, image ENV alignment, contract
re-pins, one documented default flip, and the runtime gates listed in §7 *before* the flip.

**Out of scope:**

- Changing the host-side installer (`setup.py`, `START-HERE.bat`, `setup.sh`) — it already ships
  Swift as a selectable release with correct names and its own pack naming.
- Per-release MTP drafts. The shared Qwen draft is used as measured (82/97 acceptance observed);
  a Swift-trained draft would change decode speed and is a separate experiment.
- Vision/mmproj (NVIDIA-only on this backend), Coder/Unsloth release work beyond the mismatch
  guard, the pre-existing Windows setup test failure, hipBLASLt gfx1101 tuning.
- Raising or re-opening the 10 240 MiB ceiling.

**Affected boundaries:** `run.sh` → `docker/hfmodel.py` (name/tag/license fields),
`docker/entrypoint-hip.sh` + `docker/bootstrap-model.sh` (pack dir default, provenance check),
`docker/Dockerfile.hip` (ENV alignment → runtime-only image rebuild), serve (unchanged; it reads
what the entrypoint hands it), docs.

## 4. Design and decisions

Path: `run.sh` resolves `--release` + `--model` -> passes `--repo` to `hfmodel.py` for the
host-side cached/space check and forwards `STRATA_HF_REPO`, `STRATA_PACK_DIR`,
`STRATA_MODEL_NAME` explicitly -> entrypoint/bootstrap behave as today plus the provenance check.

| Decision | Chosen design | Rationale / rejected alternative |
| --- | --- | --- |
| how to name the release axis | `--release qwen\|swift\|coder` (values = `hfmodel.py` FAMILIES keys), default `swift` after the flip; unknown repo ids still work via `$STRATA_HF_REPO` | FAMILIES is already the shared layout table (and already carries `"ple"`); rejected encoding release in `--model` (`SWIFT_IQ3_XXS`) — it forks the quant vocabulary and breaks `--model IQ3_XXS` semantics |
| default quant for swift | `IQ3_XXS` (`MODEL` default becomes `IQ3_XXS` together with `RELEASE` default `swift`) | the only Swift quant measured on this card, and the one whose pack and cache already exist; other quants work through the existing family glob but get no tuned pins |
| tuning pins | key the case table on `release+quant`: `swift+IQ3_XXS -> auto/700/48` (measured), `qwen+IQ3_S -> auto/700/48` (measured), `qwen+IQ3_XXS -> 800/768/-`, else `680/768/-` | keeps both measured lines byte-identical and refuses to guess pins for unmeasured combinations |
| pack dir | `<packs>/<release-tag><quant lower>` where tag = `""` for qwen, `swift-` for swift, `coder-` for coder; computed in `run.sh` from hfmodel's new `NAME_TAG` shell key and passed as `STRATA_PACK_DIR`; entrypoint default gains the same tag fallback | matches what is already on disk (`swift-iq3_xxs`) so this host needs no move; keeps qwen dirs unchanged so existing packs keep working; rejected reusing `packs/iq3_xxs` for two releases — silent expert corruption |
| API model id | derive `MODEL_NAME` from hfmodel (`NAME_TAG` + quant lower: `swift-1.5-iq3_xxs`... actually `name` + quant, mirroring `setup.py`'s `"name": "swift-1.5"`), keep `STRATA_MODEL_NAME` override | the id must name the model actually running; yesterday's arm proved the exact target string `swift-1.5-iq3_xxs` |
| pack provenance guard | bootstrap: if `packs/.../experts.bin.src.json` exists, its shard names must start with the selected release's file prefix (or contain the quant with the right repo prefix); mismatch = die naming both packs; missing file (older pack) = warn once, proceed only if the pack dir matches the release tag | closes the one silent-corruption door the switch opens; src.json already exists for packs built since the file landed |
| quant-vs-release conflicts | quant known in `MODELS` with a fixed family (IQ1_M → coder) launches only under that release unless `$STRATA_HF_REPO` is set explicitly | hfmodel already routes IQ1_M to coder; the launcher must not pretend qwen ships it |
| image ENV | align `Dockerfile.hip` ENV to the shipping default: `STRATA_MODEL=IQ3_XXS` + `STRATA_HF_REPO=ukisai/…` (+ keep tuned values); requires `./build.sh --runtime-only` | today ENV is only visible to manual `docker run` (run.sh passes everything with `-e`), and yesterday proved the family resolution works inside the image; alternative "leave qwen as image ENV" keeps a second, lying source of truth |
| MTP | keep `/work/mtp` shared, unchanged | measured working (smoke 205/205 through spec-4 verify); draft-model license stays as disclosed in docs |
| license disclosure | `run.sh` startup log names "Swift 1.5 — Swift Open License 1.0: <url from hfmodel/setup.py>"; docs repeat it | the default now runs someone else's fine-tune; AGENTS.md wants honest attribution |

### Contracts and invariants

- **Architecture / artifacts:** unchanged engine paths; IQ3_XXS pack format unchanged; PLE from
  shard 1 for swift (already fixed, `1387fdf`); Swift keeps the same 320,001,536-row PLE table.
  New invariant: **pack dir and provenance are release-qualified**; `(release, quant)` selects
  exactly one pack path, and a pack whose `experts.bin.src.json` names the other release's
  shards is rejected before the engine starts.
- **Numerical correctness:** smoke gate is the existing 205-case coding task (already green for
  Swift); speculative verify unchanged — draft acceptance only affects speed.
- **GPU / host memory:** 10 240 MiB ceiling, `--expert-cache auto`, later-allowance 700, ring 48
  for the new default; guards unchanged; expected share ~8.7 GiB (measured 8,696). Swift IQ3_XXS
  experts.bin is ~7 GB smaller than IQ3_S's — mapped host RAM goes **down**, RAM gate reuses the
  qwen IQ3_XXS numbers (same quant, near-identical size: 70.6 vs 70.7 GiB of shards measured in
  the cache, so 75.8 GB download / 42.9 GB pack apply to both).
- **Session / cache state:** untouched (serve unchanged).
- **Engine protocol / APIs:** only the advertised model id changes with the default:
  `qwen3.8-flash-next-iq3_s` -> `swift-1.5-iq3_xxs`. Clients pinning the old id must repoint —
  called out in the release note and startup log.
- **Setup / launch / packaging:** `run.sh` gains `--release` and a qwen/swift default flip in one
  revertible commit; Dockerfile ENV aligned; `setup.py` untouched; no new downloads required on
  this host (cache + packs already present).
- **Security / filesystem:** loopback default, offline mode, and space gates unchanged; the
  download instruction printed for a missing swift quant names the ukisai repo and glob.
- **Web app / lifecycle:** unchanged; the web UI reads `/v1/models`, so it shows the new id.

## 5. Ordered implementation steps

Reinspect `git status --short` first; preserve `PLAN_TEMPLATE.md` (untracked, user's).

1. **Contract — `docker/test_launcher_contract.py` (write first, expect red)**
   - New fixtures via `./run.sh --dry-run > plans`-adjacent scratch: `default-swift.txt`
     (target line), and assertions that `--release qwen --model IQ3_S` equals
     `post-tune-iq3s.txt` and `--release qwen --model IQ3_XXS` equals the consolidation pin
     byte-for-byte; default line asserts `STRATA_MODEL=IQ3_XXS`,
     `STRATA_HF_REPO=ukisai/…`, `STRATA_PACK_DIR=/work/packs/swift-iq3_xxs`,
     `STRATA_MODEL_NAME=swift-1.5-iq3_xxs`, cache auto / later 700 / ring 48.
   - Provenance test: bootstrap-style call with the Qwen pack pointed at a swift launch -> non-
     zero with a naming message (stub `experts.bin.src.json` in a tmpdir; no GPU).
   - Done when: new-default tests fail for the right reasons; qwen byte-identical tests already
     pass (they pin preserved behavior).
2. **Resolver — `docker/hfmodel.py`**
   - Add `name`/`tag`/`license` fields to FAMILIES (sync with `setup.py:185-198`); emit
     `STRATA_NAME_TAG` (pack-dir tag) and `STRATA_MODEL_NAME_DEFAULT` from `--print shell`;
     extend `docker/test_hfmodel_ple.py` or add cases to the launcher contract.
   - Done when: shell output carries the keys for all three releases; `--print ple` behavior
     unchanged (16/16 still green).
3. **Launcher — `run.sh`**
   - `--release` parsing; `MODEL` default `IQ3_XXS` when release is swift (explicit
     `--model`/`$STRATA_MODEL` still wins); tuning case keyed on `release+quant`; host-side
     cached/space check gains `--repo`; pass `STRATA_HF_REPO`, `STRATA_PACK_DIR`,
     `STRATA_MODEL_NAME`, and the license log line; conflict rule for fixed-family quants;
     header/usage text updated with measured numbers and the qwen escape hatch.
   - Done when: step 1's launcher fixtures pass; `bash -n` clean; `--offline` / `--check-only` /
     `--dry-run` paths re-checked.
4. **Guard — `docker/entrypoint-hip.sh` + `docker/bootstrap-model.sh`**
   - Release-tagged pack-dir default (fallback when run.sh passed none); provenance check on
     `experts.bin.src.json` naming both the pack's shards and the selected release, in check and
     prepare modes.
   - Done when: mismatch dies before download or engine start; old-format packs warn-and-pass
     only when the directory tag already matches.
5. **Image — `docker/Dockerfile.hip`**
   - ENV aligned to the new default (model, repo, tuned values); `./build.sh --runtime-only`;
     run.sh's image-label guard unchanged.
   - Done when: manual `docker run` of the image (no `-e`) reaches READY with the swift id.
6. **Runtime gates before the flip (this host, GPU)** — section 7 list, recorded under
   `bench/results/2026-10-05-run-default-swift/`.
7. **Flip + docs — one commit**
   - Default `RELEASE=swift`, `MODEL=IQ3_XXS`; docs: `docs/DOCKER_GFX1101_PLAN.md` Part H entry,
     `docs/AMD_HIP.md` container note one-liner (default release + license + how to go back),
     results README with the gate matrix.
   - Done when: acceptance checklist below holds; `git revert` of that one commit returns the
     qwen default with everything else intact.

## 6. Edge cases and regression matrix

| Scenario | Expected behavior / invariant | Test boundary and test name |
| --- | --- | --- |
| default launch (swift) | docker line == `default-swift.txt`; health id `swift-1.5-iq3_xxs` | launcher contract + live gate |
| `--release qwen --model IQ3_S` / `--model IQ3_XXS` | byte-identical `-e` lines to the two fixtures | launcher contract (renamed cases) |
| qwen `packs/iq3_xxs` present, swift default | swift uses `packs/swift-iq3_xxs`; qwen pack untouched, never opened | provenance unit test + live default start |
| pack built for qwen pointed via `-e STRATA_PACK_DIR=packs/iq3_xxs` under swift | bootstrap dies naming the shard/release mismatch | provenance unit test |
| pre-src.json pack (`packs/iq3_xxs`) under its own release | warns, proceeds | provenance unit test |
| `--release qwen --model IQ1_M` | refused, message names coder | launcher contract |
| swift quant not cached, `--offline` | dies with the exact `hf download ukisai/… --include '*IQ2_XS*.gguf'` style command; no partial state | launcher inspection of `--offline` path (dry-run + live with a fake quant) |
| fresh work dir (no swift pack) | bootstrap builds `packs/swift-iq3_xxs` from cached shards (gate on this host needs ~49 GB free — run once, keep the pack) | live gate, optional on this host since the pack exists |
| running server + `--check` / `--check-only` | works without port collision, reports the swift id | preserved from consolidation contract |
| mid-stream cancel, then next request | next request `stop`, guards clean | `cancel-probe.py` re-run on swift |
| 128K ladder (32 768 / 65 536 / 130 944 fresh + follow) | no truncation, no long-context cliff, prefill within run noise of ~250 tok/s | `gate-run.sh` adapted for swift — **new evidence, never run for Swift** |
| VRAM share under all of the above | <= 10 240 MiB at every sample; expect ~8,700 | both guards, 0.05 s |
| API consumers | only visible change is the model id; usage/finish/tool-call contracts untouched (serve unchanged) | inspection; serve tests unrun (no serve changes) |
| CUDA / Windows / other quants / host installer | untouched by this change | explicit non-gates, reasons in §3 |

## 7. Verification and acceptance

### Planned commands (not results)

Host checks (no GPU, no downloads):

```bash
bash -n run.sh docker/entrypoint-hip.sh docker/bootstrap-model.sh
python -m unittest discover -s docker -p 'test_*.py'    # includes test_hfmodel_ple.py
python tools/test_setup_choices.py                      # unchanged code, regression
python tools/test_setup_amd.py                          # known pre-existing failure: 1
git diff --check
./run.sh --dry-run                                       # == default-swift.txt fixture
./run.sh --release qwen --model IQ3_S --dry-run          # == post-tune-iq3s.txt
./run.sh --release qwen --model IQ3_XXS --dry-run        # == consolidation pin
```

GPU gates (this host only, gfx1101; fixtures: both repos cached, packs, `mtp/rt`; runners already
exist in `bench/results/2026-10-04-iq3s-tuning/`, copied to the new results dir):

```bash
./run.sh --fresh --detach                              # the new default, no flags
bash bench/results/2026-10-05-run-default-swift/arm-run.sh swift-default   # prefill/stream/smoke+guards
bash bench/results/2026-10-05-run-default-swift/gate-run.sh swift-default  # 128K ladder + cancel + guards
./run.sh --fresh --release qwen --model IQ3_S --detach  # regression line boots (no full arm)
```

No C++/CMake gates: the engine is untouched. No `serve` unit gates: no serving changes (documented
as unrun with reason).

### Measurement plan

Baseline = the already-recorded IQ3_S shipping numbers (`baked-*`, `baked-gate-*`); candidate =
the new default, same host/image, same fixtures (`arm-run.sh`, `gate-run.sh`,
`check_coding_task.py` 205 cases), both guards at 0.05 s. Thresholds set now, before measuring:

| Item | Accept if |
| --- | --- |
| fresh prefill (median) | >= 235 tok/s (the 2026-10-04 swift arm: 252.9) |
| decode (median) | >= 29 tok/s (arm: 33.5; IQ3_S line: 30.0) |
| TTFT @4,096 fresh | <= 17 s (arm: 14.84) |
| share guard, whole window | every sample <= 10 240 MiB (arm: 8,696; gate adds 128K headroom) |
| 128K ladder | 130,944 fresh completes with follow-up reuse, no truncation, prefill >= 200 tok/s |
| cancel probe | next request `finish_reason=stop` |
| coding smoke | 205/205 PASS |
| follow-up reuse at every size | reused == target-26 tokens (as measured on both releases) |

### Acceptance checklist

- [ ] Default starts swift-1.5-iq3_xxs; preserved qwen lines byte-identical (contract fixtures).
- [ ] Provenance guard kills cross-release pack use (unit test) and warns for legacy packs.
- [ ] 128K + cancel + smoke + both guards green on the new default, raw results committed.
- [ ] API id change disclosed in startup log, results README, and AMD_HIP one-liner.
- [ ] License named at startup and in docs; `run.sh --help` and header agree with behavior.
- [ ] `tools/test_setup_amd.py` shows only the known pre-existing failure; docker suite green.
- [ ] One revertible default-flip commit; packs, cache, and MTP untouched on disk.
- [ ] Claims in docs carry their measured environment; no unrun gate written as passed.

## 8. Risks, open decisions, and follow-ups

| Risk / question | Impact | Mitigation / decision | Blocking? |
| --- | --- | --- | --- |
| Wrong-release pack silently reused (quant-only pack dir today) | silent quality corruption | §4 invariant: tagged dir + src.json check; unit test | yes |
| API model id change breaks pinning clients | their requests 404 by id | disclosure in log/docs; qwen stays one flag away; id also queryable at `/v1/models` | no |
| Swift answers shorter/thinner on tasks outside its fine-tune's sweet spot (only 205 coding cases + this host measured) | user-visible quality delta | measured comparison recorded 2026-10-04; smoke + gates re-run before flip; rollback = one revert | no |
| License (Swift Open License 1.0) now on the default path | legal/packaging posture for this fork | user's launcher + explicit license line; setup.py already ships Swift with the same license field | no |
| Other Swift quants (IQ3_S etc.) have no tuned pins and no measurements | slow first run elsewhere | they get the conservative `680/768/-` table row and a warning line naming the unmeasured combination | no |
| Manual `docker run` changes model when ENV aligns | surprise for shell one-liners | documented; label guard + run.sh remain the supported path | no |
| Gate host is this card only | claims don't generalize to 7900 XTX / CUDA | every number attributed to gfx1101 in docs; nothing generalized | no |

Follow-ups (explicitly deferred): hipBLASLt gfx1101 solution table (prefill lever for both
releases), a Swift-specific draft check, retirement of the pre-src.json pack format, and the
pre-existing Windows prebuilt-HIP test failure.

## 9. Execution record and handoff

- **Implemented:** everything in §5, on the gfx1101 host, plus one defect found on the way (§9a).
  `docker/hfmodel.py`: release axis (`--release`, `--print release`, shell keys
  `FAMILY`/`PACK_TAG`/`MODEL_NAME_DEFAULT`/`LICENSE`), coder-only quant rule, and a name-blind
  glob fix. `docker/bootstrap-model.sh`: pack provenance guard against `experts.bin.src.json`.
  `docker/entrypoint-hip.sh`: release-tagged pack directory, release-derived model id, license log,
  early resolver eval with the pre-release-axis fallbacks preserved. `run.sh`: `--release`,
  per-release default quant, release+quant tuning table with an unmeasured-combination warning,
  release-qualified `STRATA_PACK_DIR`/`STRATA_MODEL_NAME` and real `STRATA_HF_REPO` forwarding,
  resolver refusals dying instead of being swallowed by `eval "$(...)"`, header/usage rewritten.
  `docker/Dockerfile.hip` ENV aligned; runtime image rebuilt twice. Default flipped to Swift 1.5
  IQ3_XXS. Tests: 22 -> 34 in the docker suite (release axis, provenance, launcher fixtures) plus
  `docker/_stub_runtime.py` shared stub.
- **Deviations:**
  1. The plan's step 1 (write today's default line red, then implement) was replaced by per-module
     commits with the suite green at each step: the default-line fixture test was written when the
     flag existed, and the final flip re-pinned it in the same commit as the default change. Same
     coverage, cleaner history.
  2. Quant/release conflicts are refused by the data rule that actually holds: the quant vocabulary
     is shared between qwen/swift, so only the expert-pruned Coder's quants are exclusive
     (`--release qwen --model IQ1_M` dies; `--release swift --model IQ3_XXS` is allowed with a
     stderr note). The plan's original wording ("quant belongs to coder release") would have wrongly
     blocked Swift's shared quant names.
  3. Per-release default quants were added (qwen IQ3_S, swift IQ3_XXS, coder IQ1_M) so
     `--release qwen` alone still means what the old default did.
  4. Docs: `docs/AMD_HIP.md` has no container/`run.sh` section to append the planned one-liner to
     (the container line is documented in `docs/DOCKER_GFX1101_PLAN.md` and the script headers on
     this branch), so the release/default change is recorded in Part H item 10 and in
     `bench/results/2026-10-05-run-default-swift/README.md` instead.
  5. Image ENV alignment and the final runtime rebuild happened once, in the flip change, instead of
     a separate ENV-only commit; the manual `docker run` gate was run after that rebuild.
  6. Swift's 128K ladder and cancel probe ("never run for Swift") were run on the release-flag line
     **before** the default flip, and the no-flag default was then verified to be byte-identical to
     that line and re-smoked live.
- **Documentation updated:** `docs/DOCKER_GFX1101_PLAN.md` (Part H item 10),
  `bench/results/2026-10-05-run-default-swift/README.md` (gate matrix),
  `bench/results/2026-10-05-run-default-swift/default-swift-explicit.txt` (pinned line);
  `run.sh` header/usage carries the behavior, the measured numbers and the breaking change.

| Exact command | Actual result | Notes / blocker |
| --- | --- | --- |
| `python -m unittest discover -s docker -p 'test_*.py'` | passed, 34/34 | no GPU, no downloads |
| `python tools/test_setup_choices.py` | passed, 35/35 | unrelated to this change |
| `python tools/test_setup_amd.py` | failed, 1/28 | pre-existing `WindowsDetection::test_prebuilt_hip_zip`; fails identically at HEAD~3, unrelated |
| `python -m unittest serve.test_server serve.test_parser_stream serve.test_lifecycle` | passed | no serve changes; run as a courtesy regression |
| `git diff --check` | passed | whitespace clean |
| `./run.sh --dry-run` vs `default-swift-explicit.txt` | passed | the flip is byte-identical to the gated line |
| `./run.sh --release qwen --model IQ3_S --dry-run` vs `post-tune-iq3s.txt` | passed | Qwen pin preserved |
| `./run.sh --release qwen --model IQ3_XXS --dry-run` vs `pre-run-iq3xxs.txt` | passed | Qwen pin preserved |
| live: `./run.sh --release swift --model IQ3_XXS --fresh --detach` + arm + gates | passed | prefill 255.0, decode 32.9, TTFT@4K 15.67 s; 130,944-token prompt 229.7 tok/s with follow-up reuse 130,937; cancel/recovery `stop`; smoke 205/205; share peak 8,697 MiB PASS |
| live: `./run.sh --release qwen --model IQ3_S --fresh --detach` + smoke | passed | tuned shapes (695 slots, 534 borrowed), 205/205 |
| live: `./run.sh --fresh --detach` (no flags) | passed | READY `swift-1.5-iq3_xxs`, smoke 205/205 |
| live: `docker run` with no `-e` | passed | READY `swift-1.5-iq3_xxs` from the image ENV |
| live: `./run.sh --check-only` | passed | model ok, pack provenance ok, pack ok, MTP ok |
| live: swift launch pointed at `pack/iq3_s` (`-e STRATA_PACK_DIR`) | passed | refused: "was built from another release's shards", listing the offending file |
| live: `./run.sh --offline --model IQ2_XS` | passed | dies with `hf download ukisai/Swift-1.5-... --include '*IQ2_XS*.gguf'` instead of starting a server |

**Not run:** Windows/CUDA paths (no such machine here; this change touches only the Linux HIP
container line), other Swift quants (not fetched; the unmeasured-combination warning covers them),
a fresh-work-dir pack build (the Swift pack already exists on this host - the guard and the
provenance path were exercised instead), `serve` browser flow (no renderer change).
**Measurement artifacts:** `bench/results/2026-10-05-run-default-swift/` (arm, gates, guards,
launcher fixtures, README matrix) against the `2026-10-04-iq3s-tuning/` baseline.
**Remaining work:** none for acceptance. Deferred: Swift-specific draft layer, other Swift quants'
measurements, hipBLASLt gfx1101 tuning table, retiring the pre-`src.json` pack format, the
pre-existing Windows setup test failure.

### 9a. Defect found and fixed during execution (not in the plan)

`docker/hfmodel.py::_find_in_snapshot` ended its known-release search with a name-blind catch-all
(`*0000{i}-of-00002.gguf`). Asking for a quantization the release does not ship therefore resolved
a **sibling** shard: `--model IQ2_XS` against Swift returned the Swift IQ3_XXS file, `STRATA_CACHED`
said 1, and the server would have advertised `swift-1.5-iq2_xs` while loading IQ3_XXS. The
catch-all is now only for releases the table has not been taught; a table-known release matches
only file names containing the requested quant, so an absent quant is honestly "not cached" and
`--offline` prints the right `hf download ... --include '*IQ2_XS*.gguf'`. Regression test
`test_hfmodel_release.py::test_a_quant_absent_from_a_release_is_never_a_sibling_file`, mutation-
checked (restoring the old glob fails it).
