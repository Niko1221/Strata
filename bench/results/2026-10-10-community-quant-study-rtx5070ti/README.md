# Qwen3.8-Flash-Next: which quant should you run?

Updated **2026-10-14** · **10 local files / 3 quant or finetune labs / 20 local effort arms / 20 local tool arms**. The separate **GPT-6-luna MAX and GPT-6.1-SOL ultra hosted quality references** are outside those counts and are not recommendations.

[Download the designed PDF](strata-community.pdf) (17 pages) · [Canonical metrics](data/ALL-METRICS.json) · [Luna detail](data/hosted-api-reference.json) · [SOL detail](data/hosted-api-reference-sol.json) · [Claim registry](claims.json) · [Audit](qa/AUDIT.json)

Measured on **RTX 5070 Ti 16 GB, Ryzen 9800X3D, 64 GB RAM, Windows + WSL2**. Eight stock files use Strata 0.1.40.3; Gyro-S and Gyro-M use separate rc2 setups. Gyro-S valid rc2 math, algebra, IFEval, Five Bugs and Needle results are imported. Shared evaluators do not isolate model-only stock/fork differences. Both hosted references appear inline only in comparable quality suites, not as local runs or picks.

## Straight answer

**Recommended daily-driver default: Swift IQ3_S (UkisAI), low, on the stock study setup.** This is an evidence-based judgment across coding, tools, instructions, sampled knowledge, hidden debugging, retrieval, total wall time and deployment caveats—not an arbitrary master score. It is mid-pack on speed (75m03s code+debug+5-run tool); if total workload time dominates, Swift Q2_0 low is fastest (55m54s). It does not change the deployed model or certify multi-day stability. The final matrix and Pick Your File use the same recommendation registry.

## The cast

The ten local files all descend from the same Qwen3.8-Flash-Next base model; file sizes are decimal GB, not VRAM requirements. GPT-6-luna MAX and GPT-6.1-SOL ultra are separately identified OpenAI-hosted quality references, not local files or recommendations, and are excluded from the inventory and recommendation registry. Gyro-S and Gyro-M use separate rc2 setups; Gyro-S sampled math/instructions/debug/recall use current rc2 evidence; old rc1 data is archived, not mixed into the current profile. Every common result table includes both fork models inline with engine labels. There is no standalone fork chapter.

| File / lab | Quant | File GB | Engine | Status |
| --- | --- | --- | --- | --- |
| Q-IQ3_S (ISTA-DASLab) | IQ3_S | 83.6 | 0.1.40.3 | reference |
| Q-Q2_0 (ISTA-DASLab) | Q2_0 | 66.4 | 0.1.40.3 | reference |
| Q-IQ3_XXS (ISTA-DASLab) | IQ3_XXS | 75.8 | 0.1.40.3 | reference |
| Q-IQ2_XS (ISTA-DASLab) | IQ2_XS | 67.2 | 0.1.40.3 | reference |
| S-IQ3_XXS (UkisAI) | IQ3_XXS | 76.0 | 0.1.40.3 | reference |
| S-IQ2_XS (UkisAI) | IQ2_XS | 68.2 | 0.1.40.3 | reference |
| S-IQ3_S (UkisAI) | IQ3_S | 83.7 | 0.1.40.3 | reference |
| S-Q2_0 (UkisAI) | Q2_0 | 66.6 | 0.1.40.3 | reference |
| Gyro-S (AgentionAI) | TQ1_0 | 58.5 | rc2 | rc2 |
| Gyro-M (AgentionAI) | TQ2_0 | 92.0 | rc2 | experimental on this 16 GB rig |

Downloads: [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) · [UkisAI Swift](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) · [AgentionAI Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF).

## Reliability and agentic tools

[Tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench): pinned `2.7.1.dev16+gf7c34a130`, 92 scenarios x five runs per effort; 0/1/2 scoring. **Pass@5** = full marks at least once (capability ceiling); **Pass^5** = full marks in ALL five (reliability floor). Category values use all-five means. Warning counts describe the last primary trial, not a five-run union. The table is ordered by Pass^5 floor (descending; mean breaks ties). Evaluator rating is omitted because the hosted artifact provides no comparable rating. Safety-capped is not inferred merely from the warning count. Historical Gyro-S rc1 had no usable tool baseline; the separate rc2 low/xhigh runs are included as setup observations.

| Arm / lab | Engine / route | Mean +/- SD | Pass@5 | Pass^5 floor | Ceiling-floor gap | Hard Mode mean | Warnings (last trial) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-6.1-SOL ultra (OpenAI) · hosted Codex reference | OpenAI Codex CLI 0.162.1 · ultra | 90.2 +/- 0.4 | 88.0 | 83.7 | 4.3 | 85.4 | 0 |
| Gyro-S (AgentionAI) low | rc2 | 88.6 +/- 1.5 | 88.0 | 79.3 | 8.7 | 78.6 | 2 |
| Gyro-M (AgentionAI) low | rc2 | 86.0 +/- 1.4 | 84.8 | 78.3 | 6.5 | 73.8 | 6 |
| S-Q2_0 (UkisAI) low | 0.1.40.3 | 85.0 +/- 1.0 | 82.6 | 78.3 | 4.3 | 72.8 | 5 |
| S-IQ2_XS (UkisAI) xhigh | 0.1.40.3 | 88.2 +/- 1.5 | 88.0 | 77.2 | 10.8 | 79.8 | 2 |
| Gyro-S (AgentionAI) xhigh | rc2 | 89.0 +/- 1.2 | 88.0 | 76.1 | 11.9 | 88.6 | 1 |
| S-IQ3_S (UkisAI) low | 0.1.40.3 | 88.2 +/- 1.5 | 84.8 | 76.1 | 8.7 | 81.0 | 4 |
| Q-IQ3_XXS (ISTA-DASLab) low | 0.1.40.3 | 87.0 +/- 2.0 | 87.0 | 76.1 | 10.9 | 71.2 | 7 |
| GPT-6-luna MAX (OpenAI) · hosted API reference | OpenAI Codex Responses API · max | 90.0 +/- 1.9 | 92.4 | 75.0 | 17.4 | 87.8 | 0 |
| S-IQ2_XS (UkisAI) low | 0.1.40.3 | 87.2 +/- 1.3 | 87.0 | 75.0 | 12.0 | 77.0 | 4 |
| S-IQ3_XXS (UkisAI) low | 0.1.40.3 | 84.6 +/- 0.5 | 84.8 | 75.0 | 9.8 | 69.6 | 4 |
| Q-Q2_0 (ISTA-DASLab) xhigh | 0.1.40.3 | 89.4 +/- 0.5 | 89.1 | 73.9 | 15.2 | 86.8 | 2 |
| Q-IQ3_S (ISTA-DASLab) low | 0.1.40.3 | 86.8 +/- 0.8 | 85.9 | 73.9 | 12.0 | 75.2 | 3 |
| Q-IQ2_XS (ISTA-DASLab) low | 0.1.40.3 | 86.2 +/- 0.8 | 82.6 | 73.9 | 8.7 | 68.4 | 5 |
| Q-IQ3_XXS (ISTA-DASLab) xhigh | 0.1.40.3 | 88.6 +/- 1.1 | 89.1 | 72.8 | 16.3 | 83.6 | 3 |
| S-Q2_0 (UkisAI) xhigh | 0.1.40.3 | 86.8 +/- 2.2 | 87.0 | 72.8 | 14.2 | 82.2 | 2 |
| S-IQ3_XXS (UkisAI) xhigh | 0.1.40.3 | 86.6 +/- 1.5 | 89.1 | 72.8 | 16.3 | 74.4 | 4 |
| Gyro-M (AgentionAI) xhigh | rc2 | 87.8 +/- 1.1 | 89.1 | 71.7 | 17.4 | 86.0 | 2 |
| Q-IQ2_XS (ISTA-DASLab) xhigh | 0.1.40.3 | 88.6 +/- 1.8 | 88.0 | 70.7 | 17.3 | 82.6 | 1 |
| Q-Q2_0 (ISTA-DASLab) low | 0.1.40.3 | 84.8 +/- 0.4 | 84.8 | 70.7 | 14.1 | 74.4 | 7 |
| Q-IQ3_S (ISTA-DASLab) xhigh | 0.1.40.3 | 86.8 +/- 1.3 | 87.0 | 68.5 | 18.5 | 80.2 | 4 |
| S-IQ3_S (UkisAI) xhigh | 0.1.40.3 | 85.6 +/- 0.5 | 85.9 | 68.5 | 17.4 | 75.6 | 5 |

Confidence intervals and evaluator-rating categories are omitted from this merged table because they are not available on the same basis for both hosted references and local runs. The hosted rows are separate external observations; SOL used Codex CLI with a JSON-schema tool bridge and concurrency 4, versus study concurrency 1. One trial-2 TC-56 timeout was recovered through a validated original-prefix continuation; the first-upload source retains both records. Both stay outside the local recommendation cohort.

Luna sampling caveat: MAX is explicitly pinned; unsupported/unverified Chat Completions sampling fields (temperature, presence_penalty, seed, top-p/top-k) are not forwarded by the Codex Responses bridge. SOL sampling caveat: GSM8K used seed 42, but the original study sampling seed was not retained, so exact sample identity is unverified. MMLU is only the first 100 abstract_algebra questions (5-shot); IFEval is only the first 50 prompts. Tool trials used concurrency 4 versus the reference study concurrency 1. Requested temperature, seed and token budgets are not enforced by Codex CLI. The SOL values come only from the initial upload; subsequent feedback-assisted Five Bugs validations are excluded.

![All-model tool results](charts/F3-reliability.png)

![All-model tool results](charts/F9-decomposition.png)

![All-model tool results](charts/F4-hardmode.png)

Gyro-M Hard Mode gains do not imply reliability gains: its floor drops at xhigh and mean confidence intervals overlap. Engine/configuration differences prevent a model-only causal comparison.

## Coding

[EvalPlus](https://github.com/evalplus/evalplus) through BenchKit: 164 attempted function-writing tasks, greedy generation plus one verifier-guided replacement. First = one-shot score; final includes replacement. Blank/length-limited outputs remain scored failures. Excluded harness errors reduce coverage; those rows are qualified and excluded from complete-coverage medals/composites.

![Coding](charts/F1-coding.png)

![Time to quality](charts/F2-wall.png)

Rows are ordered by complete coverage first, then final score. Hosted timing and token-rate fields are omitted because no common performance basis exists across the local engine, Luna Responses API and SOL Codex CLI.

| Arm / lab | Engine / route | First % | Final % | Solved / scored | Repairs | Infra / harness excluded |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.1-SOL ultra (OpenAI) · hosted Codex reference | OpenAI Codex CLI 0.162.1 · ultra | 95.1 | 97.6 | 160/164 | 4/8 | 0 |
| Q-IQ3_S (ISTA-DASLab) xhigh | 0.1.40.3 | 93.3 | 96.3 | 158/164 | 5/11 | 0 |
| Q-IQ3_XXS (ISTA-DASLab) xhigh | 0.1.40.3 | 92.7 | 95.7 | 157/164 | 5/11 | 0 |
| Q-IQ2_XS (ISTA-DASLab) low | 0.1.40.3 | 90.9 | 95.7 | 157/164 | 8/14 | 0 |
| S-IQ3_S (UkisAI) low | 0.1.40.3 | 92.7 | 95.7 | 157/164 | 5/11 | 0 |
| S-Q2_0 (UkisAI) low | 0.1.40.3 | 90.2 | 95.7 | 157/164 | 9/16 | 0 |
| Gyro-S (AgentionAI) low | rc2 | 90.2 | 95.7 | 157/164 | 9/15 | 0 |
| Gyro-S (AgentionAI) xhigh | rc2 | 88.4 | 95.7 | 157/164 | 12/19 | 0 |
| Gyro-M (AgentionAI) xhigh | rc2 | 92.1 | 95.7 | 157/164 | 6/13 | 0 |
| GPT-6-luna MAX (OpenAI) · hosted API reference | OpenAI Codex Responses API · max | 93.3 | 95.7 | 157/164 | 4/11 | 0 |
| Q-Q2_0 (ISTA-DASLab) low | 0.1.40.3 | 92.7 | 95.1 | 156/164 | 4/10 | 0 |
| Q-Q2_0 (ISTA-DASLab) xhigh | 0.1.40.3 | 88.4 | 95.1 | 156/164 | 11/17 | 0 |
| S-IQ3_XXS (UkisAI) low | 0.1.40.3 | 92.7 | 95.1 | 156/164 | 4/12 | 0 |
| S-IQ3_XXS (UkisAI) xhigh | 0.1.40.3 | 92.1 | 95.1 | 156/164 | 5/10 | 0 |
| S-IQ2_XS (UkisAI) low | 0.1.40.3 | 89.6 | 95.1 | 156/164 | 9/16 | 0 |
| Gyro-M (AgentionAI) low | rc2 | 93.3 | 95.1 | 156/164 | 3/11 | 0 |
| Q-IQ3_S (ISTA-DASLab) low | 0.1.40.3 | 92.7 | 94.5 | 155/164 | 3/12 | 0 |
| Q-IQ3_XXS (ISTA-DASLab) low | 0.1.40.3 | 92.1 | 94.5 | 155/164 | 4/13 | 0 |
| Q-IQ2_XS (ISTA-DASLab) xhigh | 0.1.40.3 | 89.6 | 94.5 | 155/164 | 8/16 | 0 |
| S-IQ2_XS (UkisAI) xhigh | 0.1.40.3 | 91.5 | 94.5 | 155/164 | 5/11 | 0 |
| S-IQ3_S (UkisAI) xhigh | 0.1.40.3 | 92.7 | 94.5 | 155/164 | 3/11 | 0 |
| S-Q2_0 (UkisAI) xhigh | 0.1.40.3 | 89.6 | 93.3 | 153/164 | 6/14 | 0 |

The Luna provider-reported 42 tok/s includes hidden reasoning and is not visible-text throughput or local decode speed. SOL has no inference-performance measurements. Both hosted references are excluded from local speed claims.

GPT-6.1-SOL uses only the first-upload independent-run results: [commit a48292f](https://github.com/Jhacarreiro/Strata/commit/a48292f5766f66efcdb1e23396569b0907c44c04). The later feedback-assisted Five Bugs validation is excluded.

Coding coverage correction: Q-IQ3_XXS low 155/163 becomes 155/164 (94.5%; first 92.1%) after offline adjudication of a model SyntaxError failure, retaining its measured 865.8 s wall and 99.2% throughput coverage. Swift IQ3_S low 157/163 is superseded by its complete matched run: 157/164 (95.7%; first 92.7%), 831.7 s wall, 94.4% throughput coverage; no retry splicing. Historical Gyro-S rc1 xhigh 153/162 after two HTTP 502s is superseded and not mixed into its complete rc2 score. Gyro-S rc2 scored 157/164 at both efforts. Gyro-M xhigh blank HumanEval/39 and /99 ended on length and remain failures in 157/164. No raw score/status was silently upgraded.

## Sampled knowledge, debugging and recall

GSM8K: 100 shuffled questions, 8-shot; SOL used seed 42, but the original study seed was not retained, so exact sample identity is unverified. MMLU: first 100 entirely abstract algebra, 5-shot (NOT full MMLU). IFEval: first 50 prompts / 76 constraints. [Five Bugs](https://github.com/vyang472/five-bugs): five files x five sessions. SOL uses only the first-upload aggregate (25/25 visible, 20/25 hidden); no per-task outcome matrix was included, and later feedback-assisted validations are not independent model scores. Needle context bases and transport differ, as marked.

Rows are ordered by hidden Five Bugs score; both hosted references are positioned by that metric, but remain external observations. The top score is tied at 20/25 among S-IQ3_S (UkisAI), GPT-6-luna MAX (OpenAI-hosted API reference), GPT-6.1-SOL ultra (OpenAI-hosted Codex reference). Local rows precede hosted ties, so the first row is not a unique win. Gyro-S values use valid rc2 artifacts, including a scored algebra run.

| File / lab | Engine / route | GSM8K | Algebra | IFEval prompt / instruction % | Bugs visible | Bugs hidden | Needle % @ measured context |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S-IQ3_S (UkisAI) | 0.1.40.3 | 98 | 85 | 84.0 / 89.5 | 25/25 | 20/25 | 100 @ 260,096 effective |
| GPT-6-luna MAX (OpenAI) · hosted API reference | OpenAI Codex Responses API · max | 97 | 93 | 86.0 / 90.8 | 25/25 | 20/25 | 100 @ 260,096 effective |
| GPT-6.1-SOL ultra (OpenAI) · hosted Codex reference | OpenAI Codex CLI 0.162.1 · ultra | 98 | 96 | 90.0 / 92.1 | 25/25 | 20/25 | 100 @ 221,877 actual |
| Q-IQ3_S (ISTA-DASLab) | 0.1.40.3 | 97 | 82 | 86.0 / 90.8 | 24/25 | 19/25 | 100 @ 260,096 effective |
| Q-Q2_0 (ISTA-DASLab) | 0.1.40.3 | 97 | 78 | 84.0 / 89.5 | 24/25 | 19/25 | 100 @ 260,096 effective |
| Q-IQ2_XS (ISTA-DASLab) | 0.1.40.3 | 97 | 83 | 84.0 / 89.5 | 24/25 | 19/25 | 100 @ 260,096 effective |
| S-IQ3_XXS (UkisAI) | 0.1.40.3 | 96 | 80 | 80.0 / 86.8 | 24/25 | 19/25 | 100 @ 260,096 effective |
| S-IQ2_XS (UkisAI) | 0.1.40.3 | 97 | 83 | 82.0 / 88.2 | 23/25 | 18/25 | 100 @ 260,096 effective |
| S-Q2_0 (UkisAI) | 0.1.40.3 | 90 | 72 | 80.0 / 86.8 | 23/25 | 18/25 | 100 @ 260,096 effective |
| Gyro-M (AgentionAI) | rc2 | 96 | 79 | 78.0 / 82.9 | 23/25 | 18/25 | 100 @ 129,024 effective |
| Q-IQ3_XXS (ISTA-DASLab) | 0.1.40.3 | 99 | 79 | 80.0 / 85.5 | 21/25 | 16/25 | 100 @ 260,096 effective |
| Gyro-S (AgentionAI) | rc2 | 96 | 79 | 76.0 / 82.9 | 21/25 | 16/25 | 100 @ 129,024 effective |

![Debugging](charts/F5-debug.png)

These small samples are not a whole-model intelligence ranking or long-session guarantee. SOL Needle reached 25/25 at a 260,096-token target; the five largest requests were 221,595–221,877 actual tokens and used same-role history injection. Luna’s 25/25 is recorded at 260,096 effective tokens. These are not equal-length, identical-transport tests.

Source for the SOL reference: [first upload, commit a48292f](https://github.com/Jhacarreiro/Strata/commit/a48292f5766f66efcdb1e23396569b0907c44c04).

## Reasoning economics

Local completion tokens include thinking, repair and failures. Passing/failed task costs come from response times, not from subtracting job wall time; hosted references have no comparable token-cost rows and are omitted.

![Completion tokens](charts/F6-tokens.png)

| Arm / lab | Engine | Output tokens | Passing tokens | Passing seconds | Failed response-time share % | Wall |
| --- | --- | --- | --- | --- | --- | --- |
| Q-IQ3_S (ISTA-DASLab) xhigh | 0.1.40.3 | 304,962 | 200,835 | 2,461 | 38.1 | 67m19s |
| Q-IQ3_XXS (ISTA-DASLab) xhigh | 0.1.40.3 | 390,930 | 230,123 | 2,328 | 48.5 | 76m17s |
| Q-IQ2_XS (ISTA-DASLab) low | 0.1.40.3 | 140,583 | 97,239 | 883 | 32.1 | 22m44s |
| S-IQ3_S (UkisAI) low | 0.1.40.3 | 61,395 | 51,845 | 623 | 18.9 | 13m52s |
| S-Q2_0 (UkisAI) low | 0.1.40.3 | 87,634 | 64,502 | 542 | 23.4 | 12m56s |
| Gyro-S (AgentionAI) low | rc2 | 111,805 | 65,403 | 710 | 42.4 | 21m36s |
| Gyro-S (AgentionAI) xhigh | rc2 | 497,873 | 310,023 | 3,018 | 39.8 | 84m30s |
| Gyro-M (AgentionAI) xhigh | rc2 | 391,467 | 180,234 | 2,334 | 52.5 | 82m51s |
| Q-Q2_0 (ISTA-DASLab) low | 0.1.40.3 | 277,084 | 174,148 | 1,498 | 48.8 | 49m50s |
| Q-Q2_0 (ISTA-DASLab) xhigh | 0.1.40.3 | 332,296 | 238,341 | 2,020 | 45.1 | 62m18s |
| S-IQ3_XXS (UkisAI) low | 0.1.40.3 | 68,616 | 49,192 | 486 | 25.1 | 11m56s |
| S-IQ3_XXS (UkisAI) xhigh | 0.1.40.3 | 175,860 | 133,095 | 1,363 | 50.7 | 47m11s |
| S-IQ2_XS (UkisAI) low | 0.1.40.3 | 83,746 | 56,476 | 468 | 36.3 | 13m23s |
| Gyro-M (AgentionAI) low | rc2 | 80,523 | 60,016 | 752 | 23.2 | 17m20s |
| Q-IQ3_S (ISTA-DASLab) low | 0.1.40.3 | 66,966 | 49,239 | 594 | 23.9 | 14m05s |
| Q-IQ3_XXS (ISTA-DASLab) low | 0.1.40.3 | 82,843 | 59,863 | 602 | 24.7 | 14m26s |
| Q-IQ2_XS (ISTA-DASLab) xhigh | 0.1.40.3 | 347,704 | 257,698 | 2,214 | 50.5 | 75m40s |
| S-IQ2_XS (UkisAI) xhigh | 0.1.40.3 | 118,588 | 105,555 | 2,472 | 33.6 | 63m04s |
| S-IQ3_S (UkisAI) xhigh | 0.1.40.3 | 137,839 | 81,261 | 1,017 | 60.2 | 43m32s |
| S-Q2_0 (UkisAI) xhigh | 0.1.40.3 | 243,330 | 151,062 | 1,329 | 60.0 | 56m20s |

## Measured performance

llama-benchy 0.4.0: prompt 2,048, generation 256, concurrency 1, three repeats/depth/effort. Retired inflated raw prompt-throughput values. **Prompt-to-first-token rate** = total input tokens / end-to-end TTFT, including queue/network/reasoning; not kernel prefill. Deep contexts differ: stock 131,072, rc2 122,880. Legacy Gyro-S rc1 synthetic speed files with malformed million-token/s/null cells are excluded; the distinct rc2 full-battery engine-log profiles are measured at three depths. Gyro-M client speed remains withheld because endpoint routing is unverified. GPT-6-luna MAX matched decode comparison is unavailable for matched decode comparison: requested 3/3/3 repeats at 0 / 32,768 / 122,880 tokens, with only 1/2/3 valid visible-rate samples. GPT-6.1-SOL has no inference-performance measurements. All hosted rates are withheld; Luna’s provider-reported HE+ token rate is not a speed substitute.

![Low-effort profiles](charts/F7-decode-all.png)

![Both efforts](charts/F8-decode-effort.png)

### low

| File / lab | Engine | Contexts 0/mid/deep | Decode tok/s 0/mid/deep | First-token rate 0/mid/deep | Deep TTFT seconds |
| --- | --- | --- | --- | --- | --- |
| Q-IQ3_S (ISTA-DASLab) | 0.1.40.3 | 0 / 32,768 / 131,072 | 90.9 / 94.0 / 91.7 | 1,641 / 3,164 / 3,041 | 43.77 |
| Q-Q2_0 (ISTA-DASLab) | 0.1.40.3 | 0 / 32,768 / 131,072 | 136.6 / 131.8 / 128.4 | 2,309 / 3,522 / 3,667 | 36.30 |
| Q-IQ3_XXS (ISTA-DASLab) | 0.1.40.3 | 0 / 32,768 / 131,072 | 110.1 / 106.2 / 107.4 | 2,174 / 3,284 / 3,421 | 38.91 |
| Q-IQ2_XS (ISTA-DASLab) | 0.1.40.3 | 0 / 32,768 / 131,072 | 128.7 / 129.4 / 120.0 | 2,011 / 3,195 / 3,280 | 40.59 |
| S-IQ3_XXS (UkisAI) | 0.1.40.3 | 0 / 32,768 / 131,072 | 108.7 / 111.3 / 106.0 | 2,025 / 3,329 / 3,412 | 39.01 |
| S-IQ2_XS (UkisAI) | 0.1.40.3 | 0 / 32,768 / 131,072 | 129.1 / 129.1 / 119.9 | 1,970 / 3,283 / 3,334 | 39.93 |
| S-IQ3_S (UkisAI) | 0.1.40.3 | 0 / 32,768 / 131,072 | 92.6 / 88.5 / 96.5 | 1,594 / 3,206 / 3,374 | 39.45 |
| S-Q2_0 (UkisAI) | 0.1.40.3 | 0 / 32,768 / 131,072 | 128.8 / 116.9 / 121.9 | 1,892 / 3,229 / 3,329 | 39.99 |
| Gyro-S (AgentionAI) | rc2 | 0 / 32,768 / 122,880 | 99.0 / 88.6 / 96.0 | 1,090 / 2,106 / 2,364 | 48.36 |
| Gyro-M (AgentionAI) | rc2 | 0 / 32,768 / 122,880 | 93.4 / 87.5 / 86.5 | 1,239 / 2,144 / 2,821 | 44.28 |

### xhigh

| File / lab | Engine | Contexts 0/mid/deep | Decode tok/s 0/mid/deep | First-token rate 0/mid/deep | Deep TTFT seconds |
| --- | --- | --- | --- | --- | --- |
| Q-IQ3_S (ISTA-DASLab) | 0.1.40.3 | 0 / 32,768 / 131,072 | 92.2 / 91.2 / 86.6 | 1,945 / 3,216 / 3,382 | 39.36 |
| Q-Q2_0 (ISTA-DASLab) | 0.1.40.3 | 0 / 32,768 / 131,072 | 131.6 / 130.5 / 117.2 | 2,676 / 3,836 / 3,663 | 36.35 |
| Q-IQ3_XXS (ISTA-DASLab) | 0.1.40.3 | 0 / 32,768 / 131,072 | 106.5 / 101.7 / 95.3 | 2,406 / 3,361 / 3,409 | 39.05 |
| Q-IQ2_XS (ISTA-DASLab) | 0.1.40.3 | 0 / 32,768 / 131,072 | 125.7 / 118.9 / 117.5 | 2,135 / 3,341 / 3,339 | 39.87 |
| S-IQ3_XXS (UkisAI) | 0.1.40.3 | 0 / 32,768 / 131,072 | 106.8 / 101.1 / 99.4 | 2,420 / 3,339 / 3,449 | 38.59 |
| S-IQ2_XS (UkisAI) | 0.1.40.3 | 0 / 32,768 / 131,072 | 122.7 / 123.9 / 113.8 | 2,209 / 3,286 / 3,334 | 39.93 |
| S-IQ3_S (UkisAI) | 0.1.40.3 | 0 / 32,768 / 131,072 | 85.8 / 85.5 / 85.8 | 1,933 / 3,269 / 3,388 | 39.29 |
| S-Q2_0 (UkisAI) | 0.1.40.3 | 0 / 32,768 / 131,072 | 123.0 / 121.9 / 128.9 | 2,219 / 3,167 / 3,332 | 39.96 |
| Gyro-S (AgentionAI) | rc2 | 0 / 32,768 / 122,880 | 89.3 / 89.1 / 87.4 | 1,580 / 2,546 / 2,268 | 50.10 |
| Gyro-M (AgentionAI) | rc2 | 0 / 32,768 / 122,880 | 87.7 / 80.5 / 80.7 | 1,754 / 2,690 / 2,778 | 44.96 |

## The final scorecard — comprehensive daily-driver decision matrix

No master score, weights or numeric overall rank. Display order is recommended default, stock low-effort alternatives, higher-effort options, experimental setups. Shared per-file math/algebra/instruction/debug/recall samples are not separate effort measurements. Gyro-S current rc2 math, algebra, instructions, debugging and retrieval are included. These Gyro-S auxiliary suites ran once at xhigh; repeated row values are context, not separate low-effort measurements. TPS remains in the speed diagnostics. Heatmap orange/cream/teal bands compare the observed range WITHIN each test column; raw labels and per-column ranges stay visible. This magnifies relative differences, not absolute pass/fail or statistical significance; unavailable cells stay gray.

![All-test evidence matrix](charts/F10-scoreboard.png)

| Setup / effort | Code first/final % | Tool mean±SD / floor | Hard Mode | Math / algebra % | IFE prompt / instruction % | Bugs visible / hidden | Needle % @ tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S-IQ3_S (UkisAI) low | 92.7 / 95.7 | 88.2±1.5 / 76.1 | 81.0 | 98.0 / 85.0 | 84.0 / 89.5 | 25/25 / 20/25 | 100.0 @ 260,096 |
| Q-IQ3_XXS (ISTA-DASLab) low | 92.1 / 94.5 | 87.0±2.0 / 76.1 | 71.2 | 99.0 / 79.0 | 80.0 / 85.5 | 21/25 / 16/25 | 100.0 @ 260,096 |
| Q-IQ3_S (ISTA-DASLab) low | 92.7 / 94.5 | 86.8±0.8 / 73.9 | 75.2 | 97.0 / 82.0 | 86.0 / 90.8 | 24/25 / 19/25 | 100.0 @ 260,096 |
| Q-IQ2_XS (ISTA-DASLab) low | 90.9 / 95.7 | 86.2±0.8 / 73.9 | 68.4 | 97.0 / 83.0 | 84.0 / 89.5 | 24/25 / 19/25 | 100.0 @ 260,096 |
| Q-Q2_0 (ISTA-DASLab) low | 92.7 / 95.1 | 84.8±0.4 / 70.7 | 74.4 | 97.0 / 78.0 | 84.0 / 89.5 | 24/25 / 19/25 | 100.0 @ 260,096 |
| S-IQ2_XS (UkisAI) low | 89.6 / 95.1 | 87.2±1.3 / 75.0 | 77.0 | 97.0 / 83.0 | 82.0 / 88.2 | 23/25 / 18/25 | 100.0 @ 260,096 |
| S-IQ3_XXS (UkisAI) low | 92.7 / 95.1 | 84.6±0.5 / 75.0 | 69.6 | 96.0 / 80.0 | 80.0 / 86.8 | 24/25 / 19/25 | 100.0 @ 260,096 |
| S-Q2_0 (UkisAI) low | 90.2 / 95.7 | 85.0±1.0 / 78.3 | 72.8 | 90.0 / 72.0 | 80.0 / 86.8 | 23/25 / 18/25 | 100.0 @ 260,096 |
| Q-IQ3_XXS (ISTA-DASLab) xhigh | 92.7 / 95.7 | 88.6±1.1 / 72.8 | 83.6 | 99.0 / 79.0 | 80.0 / 85.5 | 21/25 / 16/25 | 100.0 @ 260,096 |
| Q-IQ3_S (ISTA-DASLab) xhigh | 93.3 / 96.3 | 86.8±1.3 / 68.5 | 80.2 | 97.0 / 82.0 | 86.0 / 90.8 | 24/25 / 19/25 | 100.0 @ 260,096 |
| Q-IQ2_XS (ISTA-DASLab) xhigh | 89.6 / 94.5 | 88.6±1.8 / 70.7 | 82.6 | 97.0 / 83.0 | 84.0 / 89.5 | 24/25 / 19/25 | 100.0 @ 260,096 |
| Q-Q2_0 (ISTA-DASLab) xhigh | 88.4 / 95.1 | 89.4±0.5 / 73.9 | 86.8 | 97.0 / 78.0 | 84.0 / 89.5 | 24/25 / 19/25 | 100.0 @ 260,096 |
| S-IQ2_XS (UkisAI) xhigh | 91.5 / 94.5 | 88.2±1.5 / 77.2 | 79.8 | 97.0 / 83.0 | 82.0 / 88.2 | 23/25 / 18/25 | 100.0 @ 260,096 |
| S-IQ3_S (UkisAI) xhigh | 92.7 / 94.5 | 85.6±0.5 / 68.5 | 75.6 | 98.0 / 85.0 | 84.0 / 89.5 | 25/25 / 20/25 | 100.0 @ 260,096 |
| S-IQ3_XXS (UkisAI) xhigh | 92.1 / 95.1 | 86.6±1.5 / 72.8 | 74.4 | 96.0 / 80.0 | 80.0 / 86.8 | 24/25 / 19/25 | 100.0 @ 260,096 |
| S-Q2_0 (UkisAI) xhigh | 89.6 / 93.3 | 86.8±2.2 / 72.8 | 82.2 | 90.0 / 72.0 | 80.0 / 86.8 | 23/25 / 18/25 | 100.0 @ 260,096 |
| Gyro-M (AgentionAI) low | 93.3 / 95.1 | 86.0±1.4 / 78.3 | 73.8 | 96.0 / 79.0 | 78.0 / 82.9 | 23/25 / 18/25 | 100.0 @ 129,024 |
| Gyro-M (AgentionAI) xhigh | 92.1 / 95.7 | 87.8±1.1 / 71.7 | 86.0 | 96.0 / 79.0 | 78.0 / 82.9 | 23/25 / 18/25 | 100.0 @ 129,024 |
| Gyro-S (AgentionAI) low | 90.2 / 95.7 | 88.6±1.5 / 79.3 | 78.6 | 96.0 / 79.0 | 76.0 / 82.9 | 21/25 / 16/25 | 100.0 @ 129,024 |
| Gyro-S (AgentionAI) xhigh | 88.4 / 95.7 | 89.0±1.2 / 76.1 | 88.6 | 96.0 / 79.0 | 76.0 / 82.9 | 21/25 / 16/25 | 100.0 @ 129,024 |

Sample sizes: GSM8K 100; MMLU first 100 abstract-algebra questions, not general knowledge coverage; IFEval 50 prompts / 76 constraints; Five Bugs 25 sessions; Needle 25 cells. Retrieval at 129,024 and 260,096 tokens is not equivalent capacity. Missing matched rc2 evidence is not imputed.

| Setup / effort | Code wall | Debug wall | 5-run tool wall | Code output tokens | Stored rating / warnings | Decision / caveat |
| --- | --- | --- | --- | --- | --- | --- |
| S-IQ3_S (UkisAI) low | 13m52s | 3m36s | 57m36s | 61,395 | 4/5 Good; 4 warnings | Recommended default; Stock study setup; no long-duration stability or comprehensive safety certification |
| Q-IQ3_XXS (ISTA-DASLab) low | 14m26s | 9m01s | 54m06s | 82,843 | 3/5 Adequate (safety-capped); 7 warnings | Stock alternative; Stock study setup; no long-duration stability or comprehensive safety certification |
| Q-IQ3_S (ISTA-DASLab) low | 14m05s | 7m57s | 58m18s | 66,966 | 4/5 Good; 3 warnings | Stock alternative; Stock study setup; no long-duration stability or comprehensive safety certification |
| Q-IQ2_XS (ISTA-DASLab) low | 22m44s | 4m28s | 44m42s | 140,583 | 3/5 Adequate (safety-capped); 5 warnings | Stock alternative; Stock study setup; no long-duration stability or comprehensive safety certification |
| Q-Q2_0 (ISTA-DASLab) low | 49m50s | 4m36s | 40m12s | 277,084 | 3/5 Adequate (safety-capped); 7 warnings | Stock alternative; Stock study setup; no long-duration stability or comprehensive safety certification |
| S-IQ2_XS (UkisAI) low | 13m23s | 2m24s | 42m24s | 83,746 | 3/5 Adequate (safety-capped); 4 warnings | Stock alternative; Stock study setup; no long-duration stability or comprehensive safety certification |
| S-IQ3_XXS (UkisAI) low | 11m56s | 2m10s | 48m24s | 68,616 | 4/5 Good; 4 warnings | Stock alternative; Stock study setup; no long-duration stability or comprehensive safety certification |
| S-Q2_0 (UkisAI) low | 12m56s | 2m34s | 40m24s | 87,634 | 4/5 Good; 5 warnings | Stock alternative; Stock study setup; no long-duration stability or comprehensive safety certification |
| Q-IQ3_XXS (ISTA-DASLab) xhigh | 76m17s | 9m01s | 69m42s | 390,930 | 3/5 Adequate (safety-capped); 3 warnings | Task-specific effort; Stock study setup; no long-duration stability or comprehensive safety certification |
| Q-IQ3_S (ISTA-DASLab) xhigh | 67m19s | 7m57s | 84m36s | 304,962 | 3/5 Adequate (safety-capped); 4 warnings | Task-specific effort; Stock study setup; no long-duration stability or comprehensive safety certification |
| Q-IQ2_XS (ISTA-DASLab) xhigh | 75m40s | 4m28s | 54m06s | 347,704 | 4/5 Good; 1 warnings | Task-specific effort; Stock study setup; no long-duration stability or comprehensive safety certification |
| Q-Q2_0 (ISTA-DASLab) xhigh | 62m18s | 4m36s | 52m12s | 332,296 | 5/5 Excellent; 2 warnings | Task-specific effort; Stock study setup; no long-duration stability or comprehensive safety certification |
| S-IQ2_XS (UkisAI) xhigh | 63m04s | 2m24s | 53m06s | 118,588 | 5/5 Excellent; 2 warnings | Task-specific effort; Stock study setup; no long-duration stability or comprehensive safety certification |
| S-IQ3_S (UkisAI) xhigh | 43m32s | 3m36s | 81m12s | 137,839 | 3/5 Adequate (safety-capped); 5 warnings | Task-specific effort; Stock study setup; no long-duration stability or comprehensive safety certification |
| S-IQ3_XXS (UkisAI) xhigh | 47m11s | 2m10s | 66m24s | 175,860 | 3/5 Adequate (safety-capped); 4 warnings | Task-specific effort; Stock study setup; no long-duration stability or comprehensive safety certification |
| S-Q2_0 (UkisAI) xhigh | 56m20s | 2m34s | 76m12s | 243,330 | 4/5 Good; 2 warnings | Task-specific effort; Stock study setup; no long-duration stability or comprehensive safety certification |
| Gyro-M (AgentionAI) low | 17m20s | 6m16s | 58m59s | 80,523 | 3/5 Adequate (safety-capped); 6 warnings | Experimental; not default; Only 205 MiB minimum free VRAM; CTest ple_parity failure; completed_with_invalid_slices; decode unverified |
| Gyro-M (AgentionAI) xhigh | 82m51s | 6m16s | 82m49s | 391,467 | 4/5 Good; 2 warnings | Experimental; not default; Only 205 MiB minimum free VRAM; CTest ple_parity failure; completed_with_invalid_slices; decode unverified |
| Gyro-S (AgentionAI) low | 21m36s | 7m21s | 50m34s | 111,805 | 4/5 Good; 2 warnings | Experimental; not default; rc2 math/instructions/debug/recall/algebra verified; experimental fork, not daily-default endorsement |
| Gyro-S (AgentionAI) xhigh | 84m30s | 7m21s | 61m55s | 497,873 | 5/5 Excellent; 1 warnings | Experimental; not default; rc2 math/instructions/debug/recall/algebra verified; experimental fork, not daily-default endorsement |

## Pick Your File — the same evidence-based decisions

Default rationale: Swift IQ3_S low combines complete coding 157/164 in 831.7 s with tool mean/floor 88.2/76.1, math 98%, algebra 85%, IFEval 84.0/89.5%, hidden debugging 20/25 and 100% retrieval at 260,096 effective tokens. These are limited study observations, not a universal winner claim. Warnings remain; neither rating nor benchmark pass proves production safety.

| Decision | Setup / effort | Rationale / limitation |
| --- | --- | --- |
| Recommended daily-driver default | S-IQ3_S (UkisAI) low | Broad stock-cohort evidence: strong coding, tools, sampled knowledge, instructions and hidden debugging. Total workload wall is mid-pack (75m03s), not the fastest; chosen for breadth, not speed. |
| Fastest total workload time | S-Q2_0 (UkisAI) low | Lowest total wall (code+debug+5-run tool) at 55m54s and the best stock all-five floor; sampled math/algebra and hidden debugging trail the recommended default. |
| Tool-reliability specialist | S-Q2_0 (UkisAI) low | Highest observed stock all-five floor, and it doubles as the fastest total wall (55m54s); weaker sampled math/algebra and hidden debugging than the recommended default. |
| Coding-latency specialist | S-IQ3_XXS (UkisAI) low | Shortest complete coding job; total workload wall 62m31s. Not the strongest broad sampled capability profile. |
| Coding-score specialist | Q-IQ3_S (ISTA-DASLab) xhigh | Highest observed stock final coding score; substantially longer total wall (160m) than low-effort Swift options. |
| Sampled knowledge / debugging choice | S-IQ3_S (UkisAI) low | Highest local sampled algebra accuracy and hidden debugging score; limited samples, not a whole-model knowledge exam. |
| Experimental comparison, not a default pick | Gyro-M (AgentionAI) low | Scored quality is useful, but narrow 16 GB memory margin, raw safety-gate failures, CTest/campaign caveats and the slowest total wall (172m at xhigh) prevent a daily-driver endorsement. |

## Maintainer findings, validity and corrections

- rc1 forced tools were ignored with Gyro-S; rc2 live forced calls passed with Gyro-M in streaming/non-streaming. Gyro-S also has scored rc2 low/xhigh TEB arms. These are serving-stack findings, not model-only quality changes.
- Gyro-S campaign overall status is **`aborted_or_invalid`**: required HE+/TEB slices are eligible, but auxiliary `plugin-mmlu` is invalid and the controller exited with RuntimeError('S battery exited 2'). Restoration closeout passed; valid slices are used individually. Four valid rc2 auxiliary suites are imported. A subsequent guarded algebra-only rerun on the pinned rc2 model/engine produced a valid scored algebra result (79%, first 100 abstract-algebra, 5-shot), imported with provenance; the original HTTP 429 failure log is retained.
- The Gyro-M full campaign finished `2026-10-12T17:27:48+02:00`, status **`completed_with_invalid_slices`**. Its 74 resource samples recorded minimum **205 MiB free VRAM**, peak **78 C** (correcting the earlier 64 C summary). Experimental on this 16 GB rig; no supported daily-driver claim.
- CTest: 49 passed / 2 skipped / 1 failed (`ple_parity`, missing Q2_0 PLE/capture fixtures). Separate `ple_reader_selftest` passed.
- Both coding audits ran; a dictionary merge overwrote audit step names in done.json. Xhigh validity logic missed length-limited blank finals. Original status/results and logs are preserved.
- Historical partial denominators are resolved with provenance-bearing scoring adjudication and a complete matched replacement run; category comparisons use five-run means; inflated prompt rate is retired; ceiling-minus-floor gap is correctly named.
- Separate Swift TEB v2.8.0 / 97-scenario runs are not mixed into this 92-scenario cohort. Production restoration passed; this document update made no production/model changes.

## What we did NOT test

Vision quality, multi-day agents, other GPUs/VRAM sizes, multi-GPU, additional seeds, full knowledge datasets, additional independent sampled-suite repeats, or Gyro-M beyond 131K. Same-evaluator historical observations are not a same-day model-only stock/fork control.

## Provenance

| Item | Value |
| --- | --- |
| Stock version / digest | 0.1.40.3 / 34CDE150B21148E6 |
| Fork engine commit | 9ff02c9fa9c86a88088370925b3c9131cc5d6c26 |
| Fork binary SHA256 | 08d70e30b9176dee5273ba74bd67c4d4c435bef92d449323168d7a8a3497585f |
| Gyro-M revision | 2cd4f24bcb0fa053149a4e4ec15506ecc64cafab |
| Gyro-M SHA256 | 7aa36d15131d255179ab6c87fa9c9f81a0b541fc9e81e959d4c3e29eb20feb5f |
| Fork configured context / KV | 131,072 / int8 |
| Tool evaluator | 2.7.1.dev16+gf7c34a130 |

## Rebuild and verify

Python, Chromium and Poppler are required. Install the pinned report dependencies in a private virtual environment (not system Python):

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-report.txt
python -I scripts/build_revision.py
mkdir -p qa
pdftoppm -png -r 50 strata-community.pdf qa/page
python -I scripts/audit_revision.py
python -I scripts/audit_package.py
```

The compiler reads only packaged data and rewrites PDF, charts, README, scoreboard and the shared recommendation registry. The numeric audit independently recomputes HumanEval+ task counts and all 92 x 5 tool arrays, and verifies every table starts/ends on one page using T01-style identifiers. `SOURCE-MANIFEST.json` hashes the committed files; rebuilds change PDF timestamps, so verify that manifest **before** rebuilding. Raw coding files are gzip-compressed JSON with all task records retained. See [TRIMMED.md](TRIMMED.md) for exclusions and redaction boundaries.
