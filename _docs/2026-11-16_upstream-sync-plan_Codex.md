# Upstream integration plan, 2026-11-16

## Binding request and pinned inputs

User goal: preserve all fork features, incorporate official latest features,
vulnerability updates and bug fixes, follow official APIs, then overwrite-install,
tag and distribute the official-style installer in tar.7z and rewrite all README
translations. User additionally requested Python-driven three-way merging. The attached
RAM32/VRAM16 implementation plan supplies context and resource constraints;
it does not replace this newer integration goal or qualify the unfinished P0.

Fork input: 6f13c7555a7d50289f0e3587ab780627c65a8000.
Merge base: f33c8c30588ff00e24b6b63bad0bf6a14da19081.
Official input: 09b603886925994d0d4461857b01321ef3aa431c, checked via Git fetch
and GitHub API on 2026-11-16 JST. Official-only history has 900 commits;
fork-only history has 447. Upstream diff has 2263 files. The pinned integration
target is official main, including changes newer than published 0.2.13.

## Global Constraints

- Preserve Hakua workflow learning/application, /btw, /deepresearch, RustXNG,
  file/sidebar CRUD, scoped conversation/memory behavior, PrismML/TurboQuant,
  native bounded experts, integrity verification, P0 observation and tests.
- Preserve independent fork version identity and signed fork update/feed/component
  trust. Upstream app 0.2.14 and engine 0.1.60 are separate source versions;
  do not relabel old installed/released binaries. Build new 0.6.0 fork artifacts
  after integration validation, retain old assets, then tag, publish/read back
  checksums and perform backed-up standard overwrite installation.
- Preserve existing DeepSeek compatibility as a fork extension where upstream
  removed it, without advertising an unavailable native Windows backend.
  Adapt boundary APIs; do not undo unrelated upstream improvements wholesale.
- Keep physical RAM floor 3 GiB, commit headroom 4 GiB, dynamic 5-8 GiB stop,
  GPU reserve minimum 768 MiB for protected experiments on this dedicated-NVIDIA
  PC (retain upstream UMA defaults and explicit saved settings), H free 10 GiB
  and C free 5 GiB. Five compiler
  jobs maximum. No downloads of removed models, no changes to normal profiles,
  no closing unowned applications and no qualification inferred from capacity.
- Work only in H:/Strata/worktrees/upstream-sync-20261116 on its codex branch.
  Main at C:/Users/downl/Documents/ChatGPT/Strata remains intact. Existing
  standards apply; upstream AGENTS edits are documentation, not new authority.
- Do not overwrite unrelated edits or delete retained evidence. Installation,
  tags, publication and main integration are authorized by the updated goal
  after concrete verification/backups. Failures/unavailable checks remain explicit.

## Task 1: Python, server, Terminal and Control API compatibility

Ownership: setup.py, conftest.py, serve/, strata_cli/, strata_control/,
tools/*.py and tools/conftest.py, tests/terminal/, tests/snapshot/ and their
goldens. No ownership of native source, app JS/TS, top-level CMake, workflows,
release versions, AGENTS or README files. You are not alone; preserve others'
edits. Do not dispatch subagents or commit the shared merge yourself.

Resolve relevant staged merge conflicts against all three versions. Preserve
fork security, ownership, integrity and offline tests while bringing forward
upstream network/retry, drafting, cancellation and API fixes. DeepSeek removals
need explicit compatibility adapters, not accidental deletion or wholesale
old-file retention. Inspect even cleanly merged API boundaries for removals
that break retained paths. Record decisions and remaining checks in the task
report. Do not manufacture updated golden outputs: regenerate with real fixture
drivers when needed, preserving actual setup-write observation safeguards.

Use the existing Python 3.12 environment through uv offline. No implicit model,
package or runtime downloads. Run focused CPU compatibility tests under the
existing Windows Job/resource wrapper, using new names and private H outputs;
coordinate command adapter with controller. Before handing back, no conflict
markers may remain in your owned files. Report implementation versus tested
scope separately; known host/native conflicts may prevent integrated tests.

## Task 2: Host and desktop API compatibility

Ownership: all apps/host/ including server/native-control.mjs and
server/native-settings.mjs (controller hands these two over to Task2), and apps/desktop/ except
version-only package/Cargo manifests handled by controller. Preserve all fork
features in Global Constraints, adapt new official endpoint/policy/fit/drafting
interfaces, and retain isolation and auth. Resolve conflicts using all three
versions, including tests and locale files. Do not replace fork agent/workspace
flows with upstream's smaller surface. No subagents or shared merge commits.
Run appropriate host tests and frontend type/build checks under guards, using
existing exact dependencies when lockfile content remains equivalent. Record
actual failures separately from unrun UI/model validation.

## Task 3: Native, build, packaging and documentation integration

Controller owns CMake, src/, include/, metal/, sycl/, native host control/settings,
workflows, version manifests, install scripts, README, AGENTS and release docs.
Resolve genuine native conflicts while retaining bounded experts, source/pack
integrity, KV declarations, fork version and P0 counters. Carry upstream native
correctness/features and runtime API changes. Retain fork signed distribution
and resource floors. Update source provenance docs while preserving historical
release records. Document upstream removals retained as fork extensions.
Compile/test available backends with five jobs inside existing resource guard;
report unavailable HIP/SYCL/Metal hardware/build checks without claiming pass.

## Task 4: Regression, security and feature preservation evidence

After the merge is conflict-free, run focused then broader CPU suites, host
type/build/integration tests, dependency advisory checks and native compile/unit
checks that fit current resources. Source findings from official commits,
maintainer advisories and registry/audit records; an empty project advisory
page is not proof of no dependency vulnerabilities. Preserve independent model
tracks. Audit fork-only modules and public API contracts, and run meaningful
regressions for retained features. Isolated fake-model UI tests and actual
inference must be labelled separately. Verify no user profile or installed
binary changed. Fix integration failures without weakening tests.

## Task 5: Independent review and final handoff

Generate scoped diff/report packages for independent spec and code review;
resolve material findings. Create durable _docs evidence with exact fork,
upstream and resulting commits, verification commands, failed/unrun checks,
security source URLs and residual risk. Commit the verified integration branch.
Complete backed-up standard overwrite installation, immutable 0.6.0 tag and
official-style installer plus tar.7z publication/checksum readback, and rewrite
all README translations for actual fork behavior and current API provenance.
Goal is complete only once the requested
integration and appropriate validation are actually done, not at a time/token
boundary. Leave the goal active if work remains.
