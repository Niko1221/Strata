# Community benchmark: RTX 5080 16 GB, Strata 0.1.45 (15 models)

Measured on 2026-10-21 on Windows 11 (build 26200). Fifteen models on one RTX 5080 16 GB: twelve Qwen3.8-Flash-Next files (Qwen, Coder, Swift 1.5, Unsloth, two NVFP4 packs) and three Qwen3.6-35B-A3B files. All fifteen ran the official `tools/strata_bench.py --long` suite with three runs per test and no failed or skipped official test; the twelve 256K models also ran needle checks at 32K/128K/200K.

Protocol note: this is the older `tools/strata_bench.py --long` community format, measured on Strata 0.1.45 before the newer `strata bench run` standard protocol. It is a pre-protocol report, not a standard benchmark result.

Main limitations: one machine and one model session per configuration; the official suite is repeated three times per test, but the 128K/200K prompt rates come from the needle requests, not from the three-run suite. Quality is a small check (100/100/50 questions), not a paper-grade evaluation. Memory, power, PCIe link, and image inputs were not measured. See [Correctness and limitations](#correctness-and-limitations).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5080, 16,303 MiB VRAM (one GPU, GPU 0); PCIe link width/speed not recorded.
- CPU: AMD Ryzen 7 9800X3D, 8 cores / 16 threads, AVX-512; 93.7 GiB RAM.
- Storage: local NVMe (D:), model files and packs; filesystem cache not flushed between runs.
- OS: Windows 11 build 26200 (`strata_bench` reports it as "Windows 10 (10.0.26200)").
- Driver 617.14; CUDA 13.4 runtime; the engine is the official Windows 0.1.45 build.
- Strata 0.1.45, commit `477ac13c`; engine 0.1.45; Python 3.11.9 for the harness.
- No other heavy workload was run during the suite; power limits and clocks were not recorded.

## Models and configurations

Every file was downloaded by `setup.py` from the pinned repository revision below; setup checks the published size/SHA-256. GGUF names and byte sizes are in [model-manifest.json](model-manifest.json). NVFP4 is a prepared pack (`pack/experts.bin`, `ple-fp8.gguf`, dense GGUF), not a single GGUF, so the usual `--inspect` numbers are unavailable for it.

| Model | Repository @ revision | Quant | Context | Base weights | Routed bits | Disk |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `Q2_0` | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09) @ `ed59f920` | Q2_0 | 256K | 125.7 B | 2.25 | 66.4 GB |
| `IQ2_XS` | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09) @ `ed59f920` | IQ2_XS | 256K | 125.7 B | 2.35 | 68.9 GB |
| `IQ3_XXS` | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09) @ `ed59f920` | IQ3_XXS | 256K | 125.7 B | 2.84 | 76.7 GB |
| `IQ3_S` | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09) @ `ed59f920` | IQ3_S | 256K | 125.7 B | 3.33 | 84.5 GB |
| `Coder IQ1_M` | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/tree/5348543e0147355ac9cbcb031184a3546350988e) @ `5348543e` | IQ1_M | 256K | 65.3 B | 3.33 | 58.4 GB |
| `Swift IQ2_XS` | [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/b22d729eae29b5796f76fb70f91aef549b9fc52c) @ `b22d729e` | IQ2_XS | 256K | 125.7 B | 2.35 | 69.1 GB |
| `Swift IQ3_XXS` | [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/b22d729eae29b5796f76fb70f91aef549b9fc52c) @ `b22d729e` | IQ3_XXS | 256K | 125.7 B | 2.84 | 76.9 GB |
| `Swift IQ3_S` | [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/b22d729eae29b5796f76fb70f91aef549b9fc52c) @ `b22d729e` | IQ3_S | 256K | 125.7 B | 3.33 | 84.7 GB |
| `UD-IQ4_XS` | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66) @ `38bb39ee` | UD-IQ4_XS | 256K | 125.7 B | 3.94 | 94.6 GB |
| `UD-Q4_K_XL` | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66) @ `38bb39ee` | UD-Q4_K_XL | 256K | 125.7 B | 5.10 | 112.2 GB |
| `NVFP4 (GPTQ)` | [Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata/tree/640a1db97446ddd5e89d2b70ecd0297987192c93) @ `640a1db9` | NVFP4 | 256K | not inspectable | - | 128.0 GB |
| `NVFP4-NVIDIA` | [Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata/tree/cca8f7fee5ce0af932097340a75e06df25c4e57d) @ `cca8f7fe` | NVFP4-NVIDIA | 256K | not inspectable | - | 128.9 GB |
| `Qwen3.6 UD-Q4_K_XL` | [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/tree/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d) @ `5bc3e238` | UD-Q4_K_XL | 64K | 35.5 B | 4.87 | 23.8 GB |
| `Qwen3.6 UD-Q6_K_XL` | [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/tree/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d) @ `5bc3e238` | UD-Q6_K_XL | 64K | 35.5 B | 7.24 | 33.5 GB |
| `Qwen3.6 Q8_0` | [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/tree/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d) @ `5bc3e238` | Q8_0 | 64K | 35.5 B | 8.50 | 38.7 GB |

Base weights exclude the 51.2B PLE table; `setup.py --inspect` on every shard, with the PLE counted once, gives 176.9B total for the full Flash-Next files and 116.5B for the Coder (half the experts). The NVFP4 packs have the same 125.7B base + 51.2B PLE, with 4-bit experts and an FP8 PLE. Disk is the model directory in decimal GB, including vision encoders and pack files. A dash in the model and result tables means not measured or not applicable; in the calibration table it means the engine default. `not inspectable` is used where the value exists but the pack cannot be read by `--inspect`.

Shared Flash-Next launch settings: `--max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --expert-cache auto --prefill auto --spec 4 --mtp rt`, plus the calibrated `--pcie-frac`, `--spec-min-p`, and `--adapt-*` values below. Qwen3.6 uses its own path: `--max-context 65536 --prefill auto --mtp 2 --vision` (fp32 KV). Full commands are in [model-manifest.json](model-manifest.json) and each `runs/<model>/results.json`.

All twelve Flash-Next files ran `setup.py --calibrate`; Qwen3.6 is not covered by that calibration. The resulting settings:

| Model | pcie-frac | spec-min-p | adapt-every | adapt-swaps | adapt-decay | pool-workers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Q2_0` | 0 (default) | 0.5 | 1 | 80 | 0.97 | - |
| `IQ2_XS` | 0.35 | 0.70 | 1 | 160 | 0.97 | - |
| `IQ3_XXS` | 0.35 | 0.50 | 1 | 160 | 0.97 | - |
| `IQ3_S` | 0.35 | 0.50 | 1 | 80 | 0.97 | - |
| `Coder IQ1_M` | 0.35 | 0.50 | 1 | 160 | 0.97 | - |
| `Swift IQ2_XS` | 0.35 | 0.50 | 1 | 80 | 0.97 | - |
| `Swift IQ3_XXS` | 0.35 | 0.70 | 1 | 80 | 0.97 | - |
| `Swift IQ3_S` | 0.35 | 0.70 | 1 | 160 | 0.97 | - |
| `UD-IQ4_XS` | 0.35 | 0.70 | 1 | 160 | 0.97 | - |
| `UD-Q4_K_XL` | 0.35 | 0.70 | 1 | 160 | 0.97 | 4 |
| `NVFP4 (GPTQ)` | 0.35 | 0.70 | 1 | 160 | 0.97 | 4 |
| `NVFP4-NVIDIA` | 0.35 | 0.50 | 1 | 80 | 0.97 | 4 |

`Q2_0` keeps the engine default `--pcie-frac 0`; every other Flash-Next file uses 0.35. `--pool-workers 4` appears only where setup chose it; a dash means the engine default. The Qwen3.6 files use the engine defaults.

## Method

- **Official suite:** `python tools/strata_bench.py --long` against an already-running server. A warm-up chat and a 1K prompt are discarded. Chat: three questions, up to 100 new tokens, 3 runs. 4K/16K/32K: fresh word-list prompts with a unique tag, 24 new tokens, 3 runs. Agent: three sessions of six turns, each turn adding about 1,500 words and continuing from the model's own answers, 64 new tokens per turn. Loading is not timed. Values are engine-reported prompt/decode rates and client-measured TTFT. Sampling is temperature 0; thinking is off for the suite (on for GSM8K/HumanEval, off for MMLU).
- **Drafts and profiles:** Flash-Next launches use the shared `expert-profile.bin` from the Strata install and the model's `rt` MTP draft; Qwen3.6 uses `--mtp 2`. The NVFP4 packs add `pack/dense.bin` and `pack/experts.bin`.
- **Long context:** `python tools/needle_bench.py --lengths 32k,128k,200k --depths 10,50,90` for the 12 Flash-Next files; `32k,64k` plus a 60K fallback for Qwen3.6 (all three 64K targets rendered to 65,584-65,586 tokens, over the 65,536 window; see below). Each length is three requests at 10%/50%/90% depth. The 128K rate is the median of the two fresh requests (the 90% request reused 16,384 prefix tokens and is excluded); the 200K rate is the median of all three, each reusing 16,384 tokens, so it covers 183,936 freshly read tokens.
- **Quality (not part of the official suite):** GSM8K 100 questions, thinking on, max 3,072 new tokens; MMLU 100 questions randomly sampled from `cais/mmlu` with seed 1234, thinking off, answer letter only; HumanEval first 50 tasks, greedy/temperature 0, thinking on, local unit tests. These are quick checks; the limits are 100/100/50, not the full sets.
- **Weights:** `setup.py --inspect` on every shard, summing groups and counting the PLE table once. NVFP4 is marked not inspectable instead of guessing its bits.

All successful and failed requests are in [measurements.csv](measurements.csv) (official suite and agent), [long-context.csv](long-context.csv) (needles) and [quality.csv](quality.csv). Each `runs/<model>/` has the tool's original `results.json`, engine-log excerpt, quality JSON and the combined inspect table.

## Results: official suite

Medians with [minimum-maximum] over three runs (chat: nine requests; agent: fifteen follow-up turns). Actual prompt counts are 4,019 / 16,058 / 32,049 tokens for 4K/16K/32K in every file; the chat prompts are 43-51 tokens.

| Model | Context | Chat decode tok/s | Chat TTFT s | 4K prompt tok/s | 16K prompt tok/s | 32K prompt tok/s | Agent decode tok/s |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| `Q2_0` | 256K | 122.6 [110.2-154.7] | 0.247 [0.213-0.282] | 3371.9 [1595.0-3375.6] | 4586.2 [3447.0-4605.9] | 4629.3 [4606.1-4649.6] | 135.7 [69.2-175.5] |
| `IQ2_XS` | 256K | 118.3 [102.4-153.5] | 0.250 [0.214-0.282] | 3048.2 [1514.4-3049.8] | 3840.2 [3005.0-3842.0] | 3904.0 [3891.3-3907.2] | 133.8 [109.2-164.1] |
| `IQ3_XXS` | 256K | 99.8 [92.4-136.6] | 0.312 [0.264-0.373] | 2660.2 [1415.2-2661.4] | 3886.8 [3034.9-3892.8] | 3964.2 [3952.7-3965.6] | 120.3 [75.5-144.5] |
| `IQ3_S` | 256K | 89.2 [72.2-116.8] | 0.357 [0.303-0.420] | 2253.4 [1283.9-2253.7] | 3777.6 [2988.7-3795.4] | 3812.9 [3806.7-3826.6] | 98.9 [79.3-136.3] |
| `Coder IQ1_M` | 256K | 85.8 [78.0-121.0] | 0.344 [0.319-0.399] | 3358.1 [1500.7-3376.7] | 4109.1 [3181.4-4122.3] | 4169.0 [4157.9-4171.4] | 106.4 [89.4-122.9] |
| `Swift IQ2_XS` | 256K | 120.0 [105.8-149.9] | 0.274 [0.229-0.287] | 3034.6 [1489.1-3051.2] | 3846.8 [3030.9-3847.6] | 3897.9 [3897.3-3911.8] | 135.1 [106.9-177.9] |
| `Swift IQ3_XXS` | 256K | 96.5 [83.6-126.8] | 0.334 [0.275-0.370] | 2660.2 [1413.4-2669.9] | 3908.7 [3060.1-3912.4] | 3950.9 [3946.0-3959.0] | 112.8 [92.0-145.0] |
| `Swift IQ3_S` | 256K | 91.6 [82.9-120.0] | 0.363 [0.323-0.396] | 2253.7 [1311.3-2254.4] | 3793.2 [2988.5-3796.1] | 3816.1 [3810.4-3844.8] | 100.6 [79.1-118.2] |
| `UD-IQ4_XS` | 256K | 64.4 [53.7-81.2] | 0.521 [0.483-0.621] | 1852.2 [1167.3-1856.7] | 3453.2 [2783.4-3466.5] | 3455.4 [3445.0-3457.2] | 72.0 [62.0-81.8] |
| `UD-Q4_K_XL` | 256K | 51.8 [43.0-61.1] | 0.665 [0.617-0.796] | 1419.3 [979.8-1425.5] | 2592.3 [2208.9-2599.1] | 2577.7 [2565.8-2579.2] | 54.4 [46.3-75.2] |
| `NVFP4 (GPTQ)` | 256K | 56.5 [52.1-71.6] | 0.572 [0.516-0.665] | 1585.7 [1053.5-1590.7] | 3013.1 [2491.2-3015.5] | 3000.9 [2993.2-3004.1] | 59.2 [41.6-68.6] |
| `NVFP4-NVIDIA` | 256K | 53.3 [46.8-66.2] | 0.588 [0.532-0.695] | 1581.8 [1050.4-1590.7] | 3012.3 [2481.0-3015.4] | 2996.3 [2990.3-3000.6] | 57.9 [48.5-75.4] |
| `Qwen3.6 UD-Q4_K_XL` | 64K | 50.1 [47.2-55.3] | 0.251 [0.229-0.302] | 2861.3 [2770.2-2863.4] | 2864.2 [2860.2-2864.8] | 2737.6 [2732.1-2743.4] | 52.1 [48.8-55.4] |
| `Qwen3.6 UD-Q6_K_XL` | 64K | 37.1 [35.0-39.4] | 0.318 [0.303-0.367] | 1920.8 [1885.9-1920.9] | 1919.8 [1918.2-1919.8] | 1906.5 [1902.1-1906.5] | 40.3 [35.1-43.9] |
| `Qwen3.6 Q8_0` | 64K | 32.8 [30.3-34.1] | 0.386 [0.351-0.413] | 1649.4 [1626.3-1649.4] | 1647.9 [1647.1-1648.3] | 1639.4 [1636.7-1639.6] | 36.0 [31.6-39.0] |

The 4K minima include the first, cold request of each session; the median is the second/third run. Decode is the engine's accepted-token rate, including MTP drafts, not a wall-clock division. TTFT is measured by the client from HTTP submission to the first non-empty delta. The output caps are 100 new tokens (chat), 24 (4K/16K/32K) and 64 (agent); every request's actual `generated_tokens` is in [measurements.csv](measurements.csv).

### Long-context recall and observed prompt rate

| Model | 32K | 60K/64K | 128K | 200K | 128K prompt read tok/s | 200K prompt read tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Q2_0` | 3/3 | - | 3/3 | 3/3 | 2852.8 | 2333.9 |
| `IQ2_XS` | 3/3 | - | 3/3 | 3/3 | 2611.9 | 2306.9 |
| `IQ3_XXS` | 3/3 | - | 3/3 | 3/3 | 2651.2 | 2311.8 |
| `IQ3_S` | 3/3 | - | 3/3 | 3/3 | 2632.9 | 2325.0 |
| `Coder IQ1_M` | 3/3 | - | 3/3 | 3/3 | 2783.0 | 2317.5 |
| `Swift IQ2_XS` | 3/3 | - | 3/3 | 3/3 | 2609.3 | 2319.9 |
| `Swift IQ3_XXS` | 3/3 | - | 3/3 | 3/3 | 2661.8 | 2320.0 |
| `Swift IQ3_S` | 3/3 | - | 3/3 | 3/3 | 2627.8 | 2321.5 |
| `UD-IQ4_XS` | 3/3 | - | 3/3 | 3/3 | 2504.3 | 2310.2 |
| `UD-Q4_K_XL` | 3/3 | - | 3/3 | 3/3 | 1857.2 | 1906.6 |
| `NVFP4 (GPTQ)` | 3/3 | - | 3/3 | 3/3 | 2337.1 | 2284.1 |
| `NVFP4-NVIDIA` | 3/3 | - | 3/3 | 3/3 | 2338.7 | 2282.4 |
| `Qwen3.6 UD-Q4_K_XL` | 3/3 | 64K 0/3 ; 60K 3/3 | - | - | - | - |
| `Qwen3.6 UD-Q6_K_XL` | 3/3 | 64K 0/3 ; 60K 3/3 | - | - | - | - |
| `Qwen3.6 Q8_0` | 3/3 | 64K 0/3 ; 60K 3/3 | - | - | - | - |

All 12 Flash-Next files returned the needle 3/3 at every length and depth. All three Qwen3.6 files returned 3/3 at 32K and at the 60K fallback; their 64K target rendered to 65,584-65,586 tokens plus 40 new tokens, over the 65,536-token window, and every 64K request was rejected (HTTP 400). The 128K rate is a median of two fresh depth requests; the 90%-depth 128K request reused 16,384 tokens and is excluded. Every 200K request reused 16,384 prefix tokens and the rate covers the 183,936 freshly read tokens. These are observations, not a head-to-head speed claim at long context.

## Results: quality checks

| Model | GSM8K-100 | MMLU-100 | HumanEval-50 |
| --- | ---: | ---: | ---: |
| `Q2_0` | 97% | 87% | 100% |
| `IQ2_XS` | 99% | 83% | 98% |
| `IQ3_XXS` | 99% | 83% | 96% |
| `IQ3_S` | 99% | 83% | 98% |
| `Coder IQ1_M` | 99% | 73% | 98% |
| `Swift IQ2_XS` | 99% | 84% | 94% |
| `Swift IQ3_XXS` | 99% | 83% | 98% |
| `Swift IQ3_S` | 96% | 84% | 98% |
| `UD-IQ4_XS` | 99% | 86% | 98% |
| `UD-Q4_K_XL` | 99% | 87% | 98% |
| `NVFP4 (GPTQ)` | 99% | 89% | 100% |
| `NVFP4-NVIDIA` | 99% | 86% | 96% |
| `Qwen3.6 UD-Q4_K_XL` | 99% | 84% | 100% |
| `Qwen3.6 UD-Q6_K_XL` | 98% | 83% | 96% |
| `Qwen3.6 Q8_0` | 99% | 82% | 98% |

GSM8K is flexible-extract exact match (strict and flexible were equal here; the raw lm-eval JSON reports `n = 200` because it counts both metrics, while `sample_len` is 100). MMLU is one random 100-question sample (seed 1234), so a few points are noise; Coder's lower MMLU is the clearest gap. HumanEval is pass@1 on the first 50 tasks only. These checks are not comparable to full GSM8K/MMLU/HumanEval numbers.

## Failures and skipped cases

- All three Qwen3.6 files at the 64K needle target: three HTTP 400 errors each. The rendered prompts were 65,586 / 65,584 / 65,584 tokens, plus a 40-token cap, against a 65,536-token context. The 60K fallback passed 3/3 for all three. The errors are in `runs/qwen36-*/quality/needles.json` and as error rows in [long-context.csv](long-context.csv).
- No official-suite failures and no skipped 4K/16K/32K tests. Every model loaded and served the full suite.
- No image, concurrency, memory, power, or PCIe-link measurements. Vision was enabled but not exercised.
- The 200K requests are single samples per depth; they are not three independent fresh runs.

## Correctness and limitations

- This is one RTX 5080 and one Windows installation. The numbers are not a ranking of quantizations in isolation: different files have different expert placement, cache behavior, and calibrated flags, which is exactly what the report measures.
- Small quality sets: MMLU 100 questions has a standard error of roughly +/-4 points; one or two questions can move a model's row. GSM8K/HumanEval limits are 100/50.
- One session per model. Expert caches warm within a session; the OS file cache was not flushed. The 4K/16K/32K prompts are fresh (0 reused) but the expert cache state at the start of each test is the state left by the previous test, as in the official tool.
- `strata_bench` reports the OS as Windows 10 on this build; the machine is Windows 11 26200.
- NVFP4 is a prepared pack; no full-GGUF inspect, so its weight/bits cells are marked unavailable. The 4-bit expert claim is from the pack documentation, not from a local bit count.

## Reproduce

```powershell
# 1) Setup and calibrate (example: IQ3_S). Use --family/--model for each row.
python setup.py --family qwen --model IQ3_S --context 262144 --vision yes --yes --no-start
python setup.py --calibrate --family qwen --model IQ3_S --yes --no-start

# 2) Start the server, then run the official suite and recall checks.
python serve/server.py --engine strata --config strata-iq3_s.json --port 8080
python tools/strata_bench.py --long --config strata-iq3_s.json --url http://127.0.0.1:8080 --out <report-dir>
python tools/needle_bench.py --url http://127.0.0.1:8080 --lengths 32k,128k,200k --depths 10,50,90 --out <report-dir>\quality\needles.json
```

The checked-in `measurements.csv`, `long-context.csv` and `runs/*/results.json` are the tool outputs; [scripts/summarize.py](scripts/summarize.py) rebuilds the main tables from them. The quality scripts and the orchestration wrapper were local and are not required to read the data.
