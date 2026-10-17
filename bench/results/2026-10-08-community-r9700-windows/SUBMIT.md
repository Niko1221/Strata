# How to submit this report

This folder is prepared and **not pushed**. Nothing in this repository was pushed, no fork was created, and no
pull request was opened: this machine has no GitHub credentials and no `gh` CLI, and the `origin` remote points
at the upstream repository, which this user cannot write to. The steps below are yours to run.

The contribution guide this follows is [`docs/COMMUNITY_BENCHMARKS.md`](../../../docs/COMMUNITY_BENCHMARKS.md).

## What is in the package

```
bench/results/2026-10-08-community-r9700-windows/
├── README.md      the report, in the guide's report template (English)
├── SUBMIT.md      this file
├── raw/           26 arm JSONs + 26 harness summary tables + 3 greedy outputs + FIELDS.md
└── scripts/       the harness, the per-arm configs, the 5 drivers, and a provenance README
```

Total about 340 KB. No model files, no pack, no engine logs, no credentials. Verified against the guide's
requirement to "keep large model files and generated packs out of the PR".

## Before you start

- [ ] Read `README.md` end to end. In particular the **"Which windows are trustworthy"** table — two of the
      five measurement windows are reported as invalid, and the report says so prominently. Do not remove that
      if you disagree with it; a reviewer who discovers it later will distrust everything else.
- [ ] Decide whether you want to keep the claim that the `final` window drifted +17.4% in 8.4 minutes. It is
      computed from `final-base.json` and `final-base-b.json` and is the single most important caveat in the
      report.
- [ ] Optional, and **not** done here because it edits a tracked file: add one line to the "Community reports"
      list in `docs/COMMUNITY_BENCHMARKS.md`, matching the existing entries. The guide says maintainers decide
      where to include a report, so leaving the docs alone is the conservative choice. If you add the line, put
      it in a **separate commit** from the results folder so the results stay reviewable on their own.
- [ ] Optional: fill in the gaps listed as "not recorded" in the README — GGUF SHA-256s, the exact
      `tools/iq_pack.py` invocation, the model repository revision, the PCIe link width, and the GPU power
      limit. These are the items that most limit reproducibility.

## Warning: this working tree has unrelated changes

`git status` on this machine is **not clean**, and none of these belong in the benchmark PR. Verify this
yourself before staging anything:

- `tests/hip/handoff.cpp` — a **tracked file with local modifications** (+112/-2 lines).
- Untracked scratch: `tools/opt/`, `docs/opt/`, `check-output.txt`, `rocm-install.err`,
  `src/kernels/cuda/iq_kernels.cu.bak-alignas`, `src/kernels/cuda/iq_kernels.cu.worktree-alignas`,
  `tools/_scratch_zstd_read.py`, and several `strata-iq3_xxs.json.bak-*` files.

Every command below stages **only** the results folder. Never use `git add -A`, `git add .`, or
`git commit -a` for this submission.

## Step 1: create a fork and point a remote at it

`origin` is the **upstream** repository:

```text
origin  https://github.com/Niko1221/Strata.git (fetch)
origin  https://github.com/Niko1221/Strata.git (push)
```

You have no write access to `Niko1221/Strata`, so a plain `git push` will fail with a 403. Fork the repository in
the GitHub UI (**Fork** button, top right), or:

```bash
gh repo fork Niko1221/Strata --clone=false --remote=false
```

Then add your fork as a separate remote and leave `origin` alone, so `git fetch upstream` keeps working:

```bash
git remote add fork https://github.com/<YOUR-USERNAME>/Strata.git
git remote -v          # confirm: fork -> your repo, origin -> Niko1221/Strata
```

## Step 2: branch

```bash
git checkout -b community-bench-r9700
```

## Step 3: stage only this report

```bash
git add bench/results/2026-10-08-community-r9700-windows
git status --short     # expect only lines starting with "A  bench/results/2026-10-08-community-r9700-windows/"
```

Check that `tests/hip/handoff.cpp` and `tools/opt/` are **not** in the staged list. If they are, you staged too
much — reset with `git reset` and redo this step.

## Step 4: commit

Suggested message:

```
bench: add community benchmark report, AMD Radeon AI PRO R9700 (gfx1201), Windows

Decode and prompt throughput for Qwen3.8-Flash-Next IQ3_XXS at 17,154 / 99,696 /
132,886 / 199,316 actual prompt tokens on one R9700, plus a single-variable A/B
showing STRATA_SH_STREAM=0 worth +25% to +42% decode throughput, with the
mechanism visible as a drop in the engine's per-window GPU-reach wait.

Results only: no engine, no docs and no source changes. All 26 arms, the
harness, the per-arm configs and the drivers are included; model files, the
generated pack and the engine logs are left out per
docs/COMMUNITY_BENCHMARKS.md.

Each comparison uses a base anchor from the same window, and windows whose two
base anchors differ by more than 5% are reported as invalid (two of five).
n=3 per arm, no confidence intervals, TTFT not measured, synthetic prompts
only, and the GPU was shared with other sessions part of the day. No correctness
claim is made: the engine is not deterministic run to run on this build, so the
acceptance of STRATA_SH_STREAM=0 rests on a structural argument (it only moves
streams and synchronisation points) and on unchanged distributions, not on
numerical equivalence.

Co-Authored-By: <YOUR NAME> <YOUR EMAIL>
```

Replace that trailer with whatever attribution you want, or delete it. Everything above it is load-bearing.

```bash
git commit -F <your-message-file>
```

## Step 5: push to your fork

```bash
git push -u fork community-bench-r9700
```

## Step 6: open the pull request

```bash
gh pr create --repo Niko1221/Strata `
  --base main `
  --head <YOUR-USERNAME>:community-bench-r9700 `
  --title "Community benchmark: AMD Radeon AI PRO R9700 (gfx1201), Windows, 2026-10-08" `
  --body-file pr-body.md
```

Without `gh`, open <https://github.com/Niko1221/Strata/compare/main>...`<YOUR-USERNAME>`:community-bench-r9700?expand=1`
and paste the body below.

Target branch: **`main`** on **`Niko1221/Strata`**.

## Pull request body

````markdown
Community benchmark report from one user, contributed per
docs/COMMUNITY_BENCHMARKS.md. Results only — no engine, source or docs changes.

**Hardware.** 1x AMD Radeon AI PRO R9700, 32 GB VRAM, gfx1201 (RDNA4, wave32),
AMD Ryzen 9 9950X, 48 GB RAM, NVMe SSD, Windows, driver 32.0.31041.5005.

**Software.** Strata `6f32ec070f23ced9f50e704d854d775da52591ab` (tag v0.1.39), engine
0.1.39, **release prebuilt** HIP binary, ROCm 10.2.0a20260930 (TheRock wheels),
HIP 7.17.26391, hipBLASLt 100500.

**Model.** Qwen3.8-Flash-Next-GSQ-RCO, IQ3_XXS, two GGUF shards, custom native
pack from `tools/iq_pack.py`, bundled expert profile, MTP runtime.

**Configurations tested.** One engine configuration, varied by single environment
variables and by one flag:

- `STRATA_SH_STREAM=0` vs the default, at 17,154 / 99,696 / 132,886 prompt tokens
- `STRATA_HC_SPLIT=0`, `STRATA_DOORBELL_STORE=1`, `STRATA_VERIFY_COHERENT=1`,
  `STRATA_VERIFY_DEVICE_PLAN=1` — measured, **no measurable effect**
- `--spec-min-p 0` on top of `STRATA_SH_STREAM=0` — mechanism reproduces,
  throughput effect **not established**

**Headline.** With `STRATA_SH_STREAM=0`, decode throughput rises **+25% to +42%**
in every anchor-valid window, and the engine's own per-window `GPU-reach wait`
drops from 27.4-28.8 ms to 16.5-17.9 ms with the two ranges not overlapping, while
draft acceptance, cache hit rate and tokens per window are unchanged. The
variable only disables the shared-expert per-layer stream fork/join
(`src/core/verify.cpp:903-907`); on this card the fork appears to be pure
synchronisation overhead.

**Limitations, stated up front in the report.** n=3 per arm with no confidence
intervals; only median and range. Comparisons use a base anchor from the same
window, and **two of the five windows failed the 5% anchor-drift test and are
reported as invalid** — one of them drifted +17.4% in 8.4 minutes. The GPU was
shared with other sessions for part of the day: one arm was rejected with
`exit=3` because another session held the card, and three arms in the same band
died with the "VRAM-release crash". TTFT was not measured. Prompt throughput is
n=1 per arm (one cold prefill). Synthetic prompts only, 128-token output cap, no
real conversation or agent load. **No correctness check was run** — in
particular `tools/needle_bench.py` was not executed — and the engine proved
non-deterministic run to run on this build, so token-for-token equality is not
available as a criterion.

**Contents.** `README.md` in the guide's template, 26 arm JSONs plus the harness's
own summaries and a `FIELDS.md` documenting every field and unit, the harness, the
per-arm configs, and the five PowerShell drivers. ~340 KB. No model files, no
pack, no engine logs, no credentials.

**Caveat for reviewers.** These are one user's measurements on one shared card.
Nothing in the report has been verified by a maintainer, and it deliberately does
not ask for an engine change on the strength of these numbers alone. If anyone
wants to reproduce the `STRATA_SH_STREAM=0` claim, the report's anchored-window
method, the invalidation rule and the drivers are all included for that purpose.
````

## If you would rather not include the `shstream0` conclusion

The report is written so it can be submitted as-is. If you want it to be purely descriptive, delete the
`### STRATA_SH_STREAM=0` and `### Mechanism` sections, and the corresponding rows stay in the results table as
plain measurements. Do **not** delete the "Which windows are trustworthy" table or the limitations list.

## Cleanup after submitting

The package is a copy; the originals stay in `tools/opt/`. If you want to remove the copies:

```bash
Remove-Item -Recurse -Force bench\results\2026-10-08-community-r9700-windows   # before committing
git rm -r bench/results/2026-10-08-community-r9700-windows                     # after committing
```

Do not delete `tools/opt/` — it is not tracked by git, so a `git clean` would take it and you would lose the
engine logs and the drivers.