# RTX 5080 + RTX 5060 Ti: four draft-standard Flash-Next runs and 15 legacy-model results

I ran Strata 0.2.7 (engine 0.1.52) on one RTX 5080 16 GB and one RTX 5060 Ti 16 GB, with the 5060 Ti as the first/helper card and the 5080 last/main (`gpu: [1,0]`). This report has two separate data sets:

- **Standard draft protocol** (`strata-qwen38-thinking-v1-draft.1`) on the four original Qwen3.8-Flash-Next quantization files: Q2_0, IQ2_XS, IQ3_XXS and IQ3_S. These are the only rows I call standard results.
- **Legacy/quick suite** on 15 models: the four above, plus Coder IQ1_M, three Swift files, Unsloth UD-IQ4_XS and UD-Q4_K_XL, two NVFP4 packs and three Qwen3.6-35B-A3B files. These are not standard-protocol results; they use `tools/strata_bench.py --long` and the quality/needle scripts.

The earlier submission from this machine (PR #3621) was a Strata 0.1.45, one-5080, pre-protocol quick-suite report. This report replaces it with 0.2.7 dual-GPU measurements, does not reuse its numbers, and keeps the standard/legacy split explicit. The draft protocol is not an approved standard; the label `strata-qwen38-thinking-v1-draft.1` stays visible in every standard folder.

## What I found

- **4 of 4** standard runs finished with status `complete`; the sanity score was 12/12 in each of them.
- Q2_0 on the two cards: **127.8 (123.0-131.1) tok/s decode**, **1628.1 (1611.2-1652.2) tok/s** on the 4K prompt and **2506.4 (2503.7-2510.3) tok/s** on the 16K prompt (median (min-max), n=3).
- The same Q2_0 model on **one 5080** did **141.5 (137.5-153.5) tok/s decode**, **2823.0 (2792.0-2826.1)** on 4K and **3631.4 (3628.9-3631.7)** on 16K. The second card adds capacity, not speed for this 38 GB-expert model.
- 15 of the 15 legacy models have quick-suite results in this folder; the larger Q5/Q6 test is reported separately below. The Qwen3.6 64K result is a window-limit caveat, not a recall failure.

## Machine

| | |
| --- | --- |
| GPU 0 (main) | NVIDIA GeForce RTX 5080, 16 GB, 250 W limit |
| GPU 1 (helper) | NVIDIA GeForce RTX 5060 Ti, 16 GB, 150 W limit |
| CPU | AMD Ryzen 7 9800X3D, 8 cores / 16 threads |
| RAM | 96 GB DDR5 (95.7 GiB visible) |
| Storage | HP SSD FX900 Pro 4TB, NVMe |
| OS | Windows 11 26200 (10.0.26200) |
| Driver / CUDA | 621.01 / CUDA 13.0 runtime |
| Strata / engine | 0.2.7 / 0.1.52, official prebuilt Windows CUDA 13 component |
| PCIe | 5.0 x8 at measurement time (the link reported 8 of 16 lanes) |

The standard runner recorded 95.7 GiB of RAM and 2 GPUs. The engine ran from a git checkout, so the runner called it a `source` build; the engine binary itself is the official prebuilt 0.1.52 component from the release feed. That distinction is in `software.build_kind` and the build options.

## Method

### Standard protocol (four rows only)

Each standard folder is one fresh `strata bench run --profile <config> --suite standard-v1` against a loopback server. The runner starts the engine, hashes the model files, sends one warm-up, three repetitions of the short / 4K / 16K prompts and the twelve sanity checks, then stops the engine. No request was retried. Sampling is the model card's (temperature 1.0, top_p 0.95, top_k 20, thinking on, reasoning effort xhigh), and the output cap is part of the frozen suite. Prompt/decode rates are the engine's own timings; TTFT is the client's monotonic clock. Loading is not included in `required_duration_s`.

### Legacy quick suite (all other rows)

The quick suite is `python tools/strata_bench.py --long` against an already-running server: chat, fresh 4K/16K/32K prompts and an agent session, three runs per case, thinking off. I also ran GSM8K-100, MMLU-100, HumanEval-50 and needle recall at 32K/128K/200K (Qwen3.6 at 32K/60K). These checks are not part of the standard protocol and do not make the rows standard.

The four standard runs in this folder were made with `--track custom` because the launch settings are not a stock single-card profile: setup-generated dual-GPU configs, per-model `--calibrate`, then a 2 GiB / 1.5 GiB VRAM reserve and `--prefill auto:4096`. The standard runner has no stock track for that combination, so `custom` is the honest value; the exact settings are in each `config.json`.

## Standard results

| Model | Status | Sanity | short decode tok/s | 4K prompt tok/s | 16K prompt tok/s | After load s | 4K / 16K prompt tokens | Run |
| --- | --- | ---: | --- | --- | --- | ---: | --- | --- |
| `Q2_0` | complete | 12/12 | 127.8 (123.0-131.1) | 1628.1 (1611.2-1652.2) | 2506.4 (2503.7-2510.3) | 95.1 | 4093 / 16387 | [run](standard/qwen-q2_0/README.md) |
| `IQ2_XS` | complete | 12/12 | 116.1 (113.1-128.1) | 1438.0 (1428.9-1450.2) | 2295.3 (2292.1-2298.0) | 107.3 | 4093 / 16387 | [run](standard/qwen-iq2_xs/README.md) |
| `IQ3_XXS` | complete | 12/12 | 109.4 (107.1-121.0) | 1354.4 (1350.2-1362.6) | 2222.5 (2219.0-2226.2) | 115.3 | 4093 / 16387 | [run](standard/qwen-iq3_xxs/README.md) |
| `IQ3_S` | complete | 12/12 | 96.6 (91.1-102.6) | 1255.1 (1251.3-1259.7) | 2104.4 (2095.8-2105.1) | 123.9 | 4093 / 16387 | [run](standard/qwen-iq3_s/README.md) |

Cells are median (min-max), n=3. The short prompt is about 80 tokens, so its prompt rate is a request fixed cost and is not shown. `Capped` and per-request outcomes are in each run's `README.md` and `results.json`. The 4K/16K requests reached the 512-token cap while still reasoning, as the draft suite expects; their total time is the time to the cap, not to an answer.

## Same protocol, one card vs two cards (Q2_0)

| Configuration | short decode tok/s | 4K prompt tok/s | 16K prompt tok/s | After load s |
| --- | --- | --- | --- | ---: |
| RTX 5080 only (separately calibrated) | 141.5 (137.5-153.5) | 2823.0 (2792.0-2826.1) | 3631.4 (3628.9-3631.7) | 81.2 |
| RTX 5080 + RTX 5060 Ti (dual) | 127.8 (123.0-131.1) | 1628.1 (1611.2-1652.2) | 2506.4 (2503.7-2510.3) | 95.1 |

Dual/single median ratios: short decode 0.90×, 4K prompt 0.58×, 16K prompt 0.69×. The single-card run uses its own calibration and a 700 MiB reserve; the dual run uses 2 GiB / 1.5 GiB. This is a baseline, not a fifth standard submission; its folder is `standard-baseline-single/qwen-q2_0`.

## Calibration

`setup.py --calibrate` measured the PCIe share, draft floor, CPU workers, expert tier and prefill chunks on the two-card layer split. The fast values differed by model; Q2_0 kept the setup defaults and the others selected a PCIe share from 0.00 to 0.55, usually with draft floor 0.70.

| Model | pcie-frac | spec-min-p | adapt-every | adapt-swaps | adapt-decay | prefill-pipe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Q2_0` | — | 0.5 | — | — | — | — |
| `IQ2_XS` | 0.20 | 0.70 | 1 | 80 | 0.97 | — |
| `IQ3_XXS` | 0.55 | 0.70 | — | — | — | — |
| `IQ3_S` | 0.20 | 0.70 | — | — | — | — |
| `Coder IQ1_M` | — | 0.5 | — | — | — | 3072 |
| `Swift IQ2_XS` | — | 0.5 | — | — | — | — |
| `Swift IQ3_XXS` | 0.35 | 0.70 | 1 | 80 | 0.97 | — |
| `Swift IQ3_S` | 0.00 | 0.70 | — | — | — | — |
| `UD-IQ4_XS` | 0.55 | 0.70 | — | — | — | — |
| `UD-Q4_K_XL` | 1.00 | 0.70 | — | — | — | — |

Calibration is not a benchmark result: it changes launch flags to make later runs faster. The reserve and `--prefill auto:4096` were added after calibration, so the measured rates are not the same numbers the tuner printed during its own short probes.

## Legacy/quick results

These rows are **not** standard results. They answer a different question: how the same machine handles the other families and the larger quants with the quick suite. The quick suite turns thinking off, so its chat decode is not directly comparable to the standard short-decode column.

| Model | Chat decode tok/s | Chat TTFT s | 4K prompt tok/s | 16K prompt tok/s | 32K prompt tok/s | Agent decode tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| `Q2_0` | 116.1 | 0.332 | 1571.7 | 2398.9 | 2747.5 | 123.4 |
| `IQ2_XS` | 113.3 | 0.328 | 1411.3 | 2223.9 | 2635.8 | 125.7 |
| `IQ3_XXS` | 98.9 | 0.321 | 1329.8 | 2148.1 | 2466.8 | 117.2 |
| `IQ3_S` | 84.3 | 0.300 | 1231.2 | 2048.9 | 2354.3 | 110.3 |
| `Coder IQ1_M` | 95.3 | 0.355 | 1729.2 | 2519.8 | 2932.6 | 101.7 |
| `Swift IQ2_XS` | 109.8 | 0.293 | 1430.2 | 2266.1 | 2647.4 | 135.0 |
| `Swift IQ3_XXS` | 99.3 | 0.343 | 1331.5 | 2141.7 | 2477.8 | 120.6 |
| `Swift IQ3_S` | 87.3 | 0.315 | 1236.5 | 2034.9 | 2343.6 | 107.4 |
| `UD-IQ4_XS` | 52.0 | 0.704 | 1186.5 | 1890.1 | 1967.4 | 64.1 |
| `UD-Q4_K_XL` | 43.3 | 0.751 | 1000.1 | 1715.8 | 1793.5 | 51.4 |
| `NVFP4 (GPTQ)` | 48.3 | 0.735 | 1154.8 | 1856.4 | 1935.5 | 61.9 |
| `NVFP4-NVIDIA` | 46.6 | 0.698 | 1150.0 | 1855.6 | 1932.0 | 58.0 |
| `Qwen3.6 UD-Q4_K_XL` | 59.7 | 0.214 | 2847.9 | 2859.6 | 2698.7 | 60.4 |
| `Qwen3.6 UD-Q6_K_XL` | 44.0 | 0.296 | 1917.7 | 1918.3 | 1900.1 | 46.8 |
| `Qwen3.6 Q8_0` | 39.5 | 0.338 | 1647.7 | 1647.2 | 1635.8 | 42.5 |

The full quick-suite request records are in `legacy/<key>/results.json`; each folder also has the engine-log excerpt. The quick suite's 4K/16K/32K prompts are fresh and tagged; the agent session reuses its own conversation cache.

## Long context and recall

| Model | 32K | 60K | 64K | 128K | 200K | 128K prompt tok/s | 200K prompt tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Q2_0` | 3/3 | — | — | 3/3 | 3/3 | 3173.1 | 2791.9 |
| `IQ2_XS` | 3/3 | — | — | 3/3 | 3/3 | 2995.8 | 2553.5 |
| `IQ3_XXS` | 3/3 | — | — | 3/3 | 3/3 | 2879.6 | 2716.4 |
| `IQ3_S` | 3/3 | — | — | 3/3 | 3/3 | 2674.6 | 2608.6 |
| `Coder IQ1_M` | 3/3 | — | — | 3/3 | 3/3 | 3209.8 | 2694.4 |
| `Swift IQ2_XS` | 3/3 | — | — | 3/3 | 3/3 | 2988.3 | 2554.6 |
| `Swift IQ3_XXS` | 3/3 | — | — | 3/3 | 3/3 | 2880.5 | 2718.4 |
| `Swift IQ3_S` | 3/3 | — | — | 3/3 | 3/3 | 2672.8 | 2606.3 |
| `UD-IQ4_XS` | 3/3 | — | — | 3/3 | 3/3 | 1971.0 | 2289.6 |
| `UD-Q4_K_XL` | 3/3 | — | — | 3/3 | 3/3 | 1735.1 | 1811.5 |
| `NVFP4 (GPTQ)` | 3/3 | — | — | 3/3 | 3/3 | 1939.2 | 2270.9 |
| `NVFP4-NVIDIA` | 3/3 | — | — | 3/3 | 3/3 | 1950.4 | 2269.0 |
| `Qwen3.6 UD-Q4_K_XL` | 3/3 | 3/3 | - | — | — | — | — |
| `Qwen3.6 UD-Q6_K_XL` | 3/3 | 3/3 | - | — | — | — | — |
| `Qwen3.6 Q8_0` | 3/3 | 3/3 | - | — | — | — | — |

The Flash-Next files tested 32K, 128K and 200K at depths 10/50/90. The Qwen3.6-35B-A3B files are limited to a 65,536-token window: a 64K needle prompt lands at 65,584-65,586 tokens with this tokenizer, past the limit, so the 64K cells are blank by design and I used a 60K fallback instead. The fallback passed 3/3 at 32K and 3/3 at 60K; this is a window caveat, not a Q4/Q6/Q8 difference.

## Quality checks

| Model | GSM8K-100 | MMLU-100 | HumanEval-50 |
| --- | ---: | ---: | ---: |
| `Q2_0` | 99% | 87% | 100% |
| `IQ2_XS` | 99% | 84% | 98% |
| `IQ3_XXS` | 99% | 83% | 96% |
| `IQ3_S` | 99% | 85% | 98% |
| `Coder IQ1_M` | 100% | 71% | 100% |
| `Swift IQ2_XS` | 99% | 84% | 98% |
| `Swift IQ3_XXS` | 99% | 84% | 100% |
| `Swift IQ3_S` | 96% | 84% | 100% |
| `UD-IQ4_XS` | 99% | 86% | 98% |
| `UD-Q4_K_XL` | 99% | 86% | 96% |
| `NVFP4 (GPTQ)` | 99% | 87% | 100% |
| `NVFP4-NVIDIA` | 99% | 87% | 98% |
| `Qwen3.6 UD-Q4_K_XL` | 97% | 84% | 100% |
| `Qwen3.6 UD-Q6_K_XL` | 99% | 82% | 98% |
| `Qwen3.6 Q8_0` | 99% | 83% | 98% |

These are small checks, not an intelligence ranking and not proof of lossless quantization. Thinking is on for GSM8K/HumanEval and off for MMLU; the per-question records are in the quality JSONs. The scores are separate from the throughput tables.

## Larger models: Q5 and Q6

| Model | Download GB | RAM budget | Chat decode tok/s | 4K prompt tok/s | 16K prompt tok/s | Result |
| --- | ---: | --- | --- | --- | --- | --- |
| `UD-Q5_K_XL` | 158.3 | 72 | 34.6 | 535.8 | 814.3 | quick suite complete |
| `UD-Q6_K_XL` | 169.2 | 72 | 24.8 | 447.4 | 668.4 | quick suite complete |

Both are larger than the 96 GB RAM budget: setup reserved 72 GiB for experts and the rest is read from the SSD. These rows are capacity tests, not the standard protocol; the Q6 file in particular exceeds the 118 GB recommendation and can be much slower than Q5. Exact failures and ranges are in `legacy/<key>/results.json`.

## Limits

- One machine, one operator, one Windows install. The numbers are not a hardware ranking.
- The standard suite has three repetitions per case; min-max is shown, but three samples cannot describe run-to-run variability. The adaptive expert cache can move decode by several percent between runs.
- The four standard runs in this folder are the clean final batch: all four were made after the Q6 download was stopped, with no other workload on the GPUs or the CPU. Earlier in-pipeline copies are not used here; the per-run background field records the clean-batch condition.
- The standard runner labels this install a `source` build because it is a git checkout. The engine binary is the official prebuilt 0.1.52 Windows CUDA 13 component; no engine source was rebuilt.
- The 4K and 16K performance prompts reached the 512-token cap while reasoning, so the protocol's total time is the time to the cap and the first-answer time is missing for most of them.
- The Q2_0 pack was rebuilt for engine 0.1.52. The old `experts.bin` produced NaN logits after the engine upgrade; the rebuild was a prerequisite, not a tuning choice.
- The dual-GPU split can be slower than one card for a model that already fits. The 5060 Ti is the helper, so the 5080 waits on it in some phases; Q2_0 is the clear example. This report does not generalize that to every model.
- Quality checks are small samples (100/100/50) and are not a substitute for a full evaluation.

## Reproduce

The standard folders are self-contained: `results.json` has the hardware, software, model hashes, configuration and every request; `outputs.jsonl` has the raw stream chunks; `engine.log` is the run's engine log; `config.json` is the effective profile after calibration. To repeat one run:

```powershell
strata bench run --profile D:\ai\Strata\strata-q2_0.json --suite standard-v1 --track custom `
  --model-revision ed59f92082b1e93c0e96d60a8b11aab089b52f09 `
  --source-revision de4b8e4d43b917e7706784d8bb445c9af86a3540 `
  --tokenizer-revision ed59f92082b1e93c0e96d60a8b11aab089b52f09 `
  --background "no other workload" --out .\result
```

The legacy suite command was `python tools/strata_bench.py --long --out <folder>` against a running server; the quality commands were `llm-eval/run_evals.py --jobs gsm8k=100`, the 100-question MMLU script, the 50-task HumanEval script and `tools/needle_bench.py --lengths 32k,128k,200k --depths 10,50,90`. Qwen3.6 used `--lengths 32k,60k`.

## Files

- `standard/<model>/` — four draft-protocol runs: README, results, config, engine log, outputs, provenance.
- `standard-baseline-single/qwen-q2_0/` — one-card Q2_0 baseline (not a standard submission).
- `legacy/<model>/` — quick-suite results and engine-log excerpt for the 15 legacy rows.
- `quality/<model>/` — GSM8K, MMLU, HumanEval and needle JSON.
- `inspect/<model>/inspect.log` — the setup GGUF inspection where available.
- `data/standard.csv`, `data/legacy.csv`, `data/quality.csv`, `data/models.csv` — flat copies of the tables.
- `MANIFEST.sha256` — hashes of every published file.

## What is deliberately not here

- No claim that the legacy rows are standard, and no pooled average across protocols.
- No 64K Qwen3.6 recall score: the prompts do not fit the 65,536-token window.
- No Q8_0 Flash-Next run: the setup estimate is 136 GB RAM on a 96 GB machine. I did not download or pretend to measure it.
- No DeepSeek result: the smallest file is far beyond this machine's 32 GB VRAM + 96 GB RAM and would not be a comparable standard run.
