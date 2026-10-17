# Community report: RTX 3090 + RTX 3060 + dual V100

Measured by @techrian on October 14–16, 2026. **Unmodified Strata 0.1.43.1**, commit
`a0116b2782a5b82622358c04a06496485d12b2d6`, source-built with CUDA 12.8 for sm70 and sm86.

[Complete setup, build instructions and recorded experiments](https://github.com/techrian/mixed-gpu-llm-lab/tree/7b2f013786a478a59417ebe1937b477eb42dcc08) ·
[Qwen profile](https://github.com/techrian/mixed-gpu-llm-lab/tree/7b2f013786a478a59417ebe1937b477eb42dcc08/models/qwen3.8-flash-next) ·
[Full agent task evidence and Excel](https://github.com/techrian/mixed-gpu-llm-lab/tree/7b2f013786a478a59417ebe1937b477eb42dcc08/benchmarks/agent-comparison/2026-10-16)

## Hardware and software

| Physical index | GPU | VRAM | Observed link | Engine index | Base layers, half-open |
| --- | --- | ---: | --- | ---: | --- |
| 2 | Tesla PG500-216 / V100 | 32 GiB | Gen3 x4 | CUDA0 | [0,16) |
| 0 | RTX 3090 | 24 GiB | Gen4 x16 | CUDA1 | [16,29) |
| 1 | RTX 3060 | 12 GiB | Gen4 x1 | CUDA2 | [29,33) |
| 3 | Tesla PG500-216 / V100 | 32 GiB | Gen3 x4 | CUDA3 | [33,48), output / MTP |

Ryzen 7 9700X (8C/16T), 64 GB class RAM (64,933,089,280 OS-visible bytes), Ubuntu 24.04.5 LTS,
kernel 7.0.0-38-generic, driver 580.178.04. Aggregate VRAM is 100 GiB.
Power limits and clocks were not controlled. No simultaneous benchmark inference; normal host conditions were not isolated as a causal factor.
[hardware.json](hardware.json) records the measured PCIe links. GPU indices are remapped by `gpu: [2,0,1,3]`.

## Model, build and configuration

- `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF`, Q4_K_M, three shards, 119,150,722,944 bytes total.
- 262144 context ceiling; FP16 GPU KV; prefill 6144; reserve 256 MiB/card.
- All 24576 expert-cache slots are allocated. Prefill lends some expert VRAM to scratch buffers with a pinned RAM complement; this is not constant full expert residency during prefill.
- PLE in RAM (`--ple-io ram`); expert resident budget 16 GiB; `--trim-stage-weights`.
- Native Qwen MTP, `--spec 4 --spec-min-p 0.5`, q2_0 expert draft pack with CJK draft vocabulary.
- F16 vision encoder on **physical GPU 2**, separate from engine CUDA2 (the RTX3060). Vision was loaded but image quality was not tested.
- Native expert pack built with `--compat-bf16`; CUDA 12.8, `STRATA_EXPERIMENTAL_SM60=ON`, architectures `70;86`.
- llama.cpp dependency `3cf03257f219afbe7334045ff7c6a06ac68c627d`; original MTP checkpoint pin and build metadata are in the linked profile.
- Process swap blocked by `MemorySwapMax=0`; global Linux swap remains enabled.

[configuration.template.json](configuration.template.json) preserves engine settings with explicit path placeholders.
[startup-excerpt.txt](startup-excerpt.txt) records allocation. Model files, packs and binaries are excluded.
Model source revision and full local shard hashes were not recorded; filenames alone do not prove byte identity.

## 1. Earlier prompt/decode throughput measurements

The 22 unique requests in [throughput-summary.csv](throughput-summary.csv) and [throughput/](throughput/) were measured on October 14–15.
Each uses a 61,679-token synthetic archive prompt, a 512-completion-token cap, and zero observed prompt reuse.
The positional labels do not change running server settings; the matching configuration was applied before each group.
The summary `source` column names the original comparison directory; the basename selects the JSON stored here.

The final F010–F012 group uses prefill 6144, reserve 256, server defaults temperature 1.0 / top_p .95 / top_k 20 / min_p 0.0,
swap blocked, and a restarted service. Three runs; median [minimum, maximum]:

| Metric | Final F010–F012 |
| --- | --- |
| Fresh prompt tok/s | 2215.51 [2204.25, 2216.39] |
| Decode tok/s | 70.60 [66.20, 72.30] |
| Whole request seconds | 35.13 [35.11, 35.60] |

Earlier arms use prefill 512/1024/2048/4096/8192, usually two measured requests per arm, and temperature 0.7.
The 1024-to-2048 change also changes reserve 700 to 256 MiB. Later 6144 groups change sampling and swap policy.
These are distinct conditions, not one controlled experiment; all unique requests are published.
All 512 generated tokens hit the limit during internal reasoning. These throughput requests are not completed-answer quality tests.
The script obtains decode rate from engine metrics, not completion tokens divided by full request time.
Prompt text and scripts are in the [pinned throughput archive](https://github.com/techrian/mixed-gpu-llm-lab/tree/7b2f013786a478a59417ebe1937b477eb42dcc08/benchmarks/throughput).

## 2. October 16 agent task outcomes

72 episodes: 24 custom assistant tasks, each repeated three times in a reset workspace, with a common 12-turn / 4096-token-per-turn / 180-second budget.
Business APIs and knowledge are synthetic; file, SQLite and Python/Node work execute in isolated Docker containers.
The loop uses the native OpenAI tool API with 3–8 tools/task. It is Hermes-style, not a run of Hermes itself.
No model judge is used. Every actual state/protected-state/verification check and a final response must pass.

| Qwen measurement | Value |
| --- | ---: |
| Strict complete success | 69/72 (95.8%) |
| Tasks passed all three times | 21/24 |
| Successful episode median / p95 | 20.25 / 42.38 s |
| Successful completions/hour, all failed time included | 161.6 |
| Tool argument/schema validity | 399/407 (98.0%) |

Wall time includes reasoning, tools, retries and the final answer, and excludes model loading, fixture setup and hidden grading.
Fresh session tags prevent cross-task prefix reuse; within-episode reuse is retained. One unscored native-tool warm-up precedes the run.
All 72 observations and checks are in [agent-results.json](agent-results.json); full requests, responses, artifacts, task definitions and grading audit are linked above.

Three strict failures: one fenced-JSON answer, one quote interpreted as 100 quotes instead of the frozen target of one quote for 100 units,
and one empty final API response after code/state checks passed. The quote wording is ambiguous; excluding that task gives 67/69 (97.1%).
Equivalent path/metadata/decimal representations were audited using one final grader for both systems, with original verdicts retained.

A separate paired comparison in the linked lab uses Project Maya 1.1.11 / GLM Maya-M on the same PC.
Both deployed systems scored 69/72; among 67 joint successes the Qwen/GLM geometric mean time ratio was 0.870 [0.806, 0.943] (task-cluster bootstrap 95%).
Native quantization, prefill, KV and MTP differ. This is a deployment comparison, not an engine parity test or a universal model ranking.

## Reproduction and limits

Clone `techrian/mixed-gpu-llm-lab` and check out `7b2f013786a478a59417ebe1937b477eb42dcc08` for the full configurations, frozen prompts, container recipe, public loop adapter and workbook generator.
The data here is copied from those recorded measurements; it was not remeasured on current upstream main for this PR.
No engine/runtime source, defaults or backend build files are changed by this report.

These custom task scores are not official BFCL/tau/Terminal-Bench/Hermes scores. Short workflows and three repetitions are a limited sample;
long-term unattended operation, full tool discovery, live accounts, browser/GUI, vision quality and maximum-context saturation were not tested.
GLM ran before Qwen; model order and thermal effects were not counterbalanced.
The exported clean-install helper scripts have not been exercised as a fresh installation.
