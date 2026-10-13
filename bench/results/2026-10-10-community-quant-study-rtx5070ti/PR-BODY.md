### What this is

Updated community quality battery requested via Xavier: **10 local files, 3 quant/finetune labs, 20 local effort arms, 18 local tool arms**, on **RTX 5070 Ti 16 GB / Ryzen 9800X3D / 64 GB RAM / Windows + WSL2**. Eight stock files use **0.1.40.3 (`34CDE150B21148E6`)**; Gyro-S uses historical AgentionAI rc1 and Gyro-M rc2. Shared evaluators do not isolate model-only stock/fork differences. A separate GPT-6-luna MAX (OpenAI-hosted API reference) anchors the suite results; it is not a recommendation or a local RTX run.

**[Download the full PDF](https://github.com/Zauberio/Strata/raw/study/quant-battery-rtx5070ti/bench/results/2026-10-10-community-quant-study-rtx5070ti/strata-community.pdf)** (14 pages) · **[GitHub report](https://github.com/Zauberio/Strata/blob/study/quant-battery-rtx5070ti/bench/results/2026-10-10-community-quant-study-rtx5070ti/README.md)** · **[Canonical metrics](https://github.com/Zauberio/Strata/blob/study/quant-battery-rtx5070ti/bench/results/2026-10-10-community-quant-study-rtx5070ti/data/ALL-METRICS.json)** · **[Hosted API reference detail](https://github.com/Zauberio/Strata/blob/study/quant-battery-rtx5070ti/bench/results/2026-10-10-community-quant-study-rtx5070ti/data/hosted-api-reference.json)** · **[Rebuild/audit scripts](https://github.com/Zauberio/Strata/tree/study/quant-battery-rtx5070ti/bench/results/2026-10-10-community-quant-study-rtx5070ti/scripts)**.

### What changed in this revision

- Gyro-M is integrated into the common coding, tools, knowledge/debugging, economics, speed and final-board views — **no standalone model chapter**.
- GPT-6-luna MAX is integrated as a **separate OpenAI-hosted reference row/bar, score-ordered where comparable, with no medal decoration**; it is excluded from local file/arm counts, recommendations, ranks, composites and local performance claims.
- Whole tables stay on one page; section headings/explainers stay with their first result. Every table has an automated PDF-page proof.
- Final Scoreboard and Pick Your File share one generated registry: **Q-Q2_0 (ISTA-DASLab) xhigh is #1 stock all-rounder**. Lower-composite reliability/latency/debugging choices are explicitly named specialists, not alternate overall winners.
- Historical partial coding denominators (155/163, 157/163, 153/162) are qualified and excluded from complete-coverage medals/composites. Five-run category means replace last-trial category values; inflated prompt throughput is retired.

### Measured headlines, with scope

- GPT-6-luna MAX external reference: HumanEval+ 95.7% final (93.3% first); TEB 90.0 +/- 1.9, Pass@5 92.4%, Pass^5 75.0%, Hard Mode 87.8%; FiveBugs 25/25 visible / 20/25 hidden. Sampled suites and full protocols are in the linked detail/README; not a local run or recommendation.
- Matched hosted decode performance is unavailable; invalid/partial rates are withheld. Provider-reported HE+ 42.0 tok/s includes hidden reasoning and is not visible-text speed or a local decode measurement.
- Gyro-M HE+: 156/164 low (95.1%), 157/164 xhigh (95.7%). Two xhigh length-limited blank finals remain failures.
- Same pinned TEB cohort: mean 86.0 +/- 1.4 low / 87.8 +/- 1.1 xhigh; floor 78.3 / 71.7. Hard Mode improves while empirical reliability falls; no model-only causal conclusion.
- The highest stock floor remains S-Q2_0 (UkisAI) low (78.3); Gyro-M low ties the observed floor on an experimental fork, not a stock-composite rank.
- Minimum 205 MiB free VRAM, peak 78 C. This 16 GB Gyro-M result is experimental, not a supported daily-driver recommendation.

### Validity and engine findings

Campaign **`completed_with_invalid_slices`**: CTest 49 passed / 2 skipped / 1 failed (`ple_parity`, missing fixtures); separate reader selftest passed. Both HE audits exist, but a step-name merge defect made done.json mark them absent; blank-final validity handling was too permissive. Raw status and failures are preserved.

Gyro-S rc1 has no valid tool-eval baseline and malformed speed profiles remain n/a. Gyro-M rc2 live forced tools passed streaming/non-streaming; **Gyro-S on rc2 was not tested**. Separate Swift v2.8.0 / 97-scenario tool data is not blended in. No engine/server/default code is changed by this PR.

### What we did NOT test

Vision quality; multi-GPU; other GPUs/VRAM sizes; multi-day agents; additional seeds; full knowledge datasets; Gyro-M beyond 131K; Gyro-S on rc2. Results are same-rig historical observations, not same-day model-only stock/fork controls. For the hosted API reference, MAX is explicitly pinned; unsupported/unverified Chat Completions sampling fields (temperature, presence_penalty, seed, top-p/top-k) are not forwarded by the Codex Responses bridge. The matched decode profile is incomplete/unavailable and all hosted throughput rates are withheld.

### Reproduce and audit

`data/ALL-METRICS.json -> scripts/build_revision.py -> PDF + README + ten charts + claims.json + FINAL-SCOREBOARD.json`. Task JSONs, all 18 raw local five-trial tool arms, measured local perf profiles, the sanitized hosted reference detail, source manifest and validation scripts are included. Hosted detail contains audited scores/protocol/source hashes only — no prompts, model outputs, thoughts or credentials. Coding raw data is losslessly gzip-compressed; secrets/rig addresses are redacted. Scope stays one report folder plus the existing community-benchmarks row. See [TRIMMED.md](https://github.com/Zauberio/Strata/blob/study/quant-battery-rtx5070ti/bench/results/2026-10-10-community-quant-study-rtx5070ti/TRIMMED.md) for exclusions.

If committing a PDF is undesirable, I can move it to a release/attachment without changing the measured report.
