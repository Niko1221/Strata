# Community benchmark on RTX 5090: UD-Q4_K_XL and UD-Q5_K_XL, with and without images, engine 0.1.41

Measured on 2026-10-08 by CYoung83. On one machine with 89 GiB of RAM, this run measured:

- the speed of unsloth's UD-Q4_K_XL and UD-Q5_K_XL, with and without the image encoder, at a 48 GiB RAM budget, on 4K and 32K prompts;
- how far each quant's next-token distribution is from unsloth's Q8_0 on the same engine;
- a three-question image check on each quant.

Main limitations:

- One machine and three runs per configuration.
- UD-Q5_K_XL only opens with the loader fix in #1612 (issue #1611) or with a padded copy of its first shard.
- The image check is one generated picture.

The image check is the sample asked for in #967 (images on UD-Q4_K_XL), plus the same check on UD-Q5_K_XL.

## Hardware and software

- **GPU:** RTX 5090 32 GB, power limit 450 W, PCIe 5.0 x16.
- **CPU:** Ryzen 9 9950X3D (16 cores, AVX-512).
- **RAM:** 89 GiB as `free -g` reports it.
- **Storage:** not recorded.
- **System:** Ubuntu 26.04, kernel 7.0.0-38-generic, driver 610.57.04, CUDA 13.3 (V13.3.73). See [configs/nvcc.txt](configs/nvcc.txt).
- **Strata:** a source build of tag v0.1.41 (`fb58e0d`) with the two commits of #1612 on top ([configs/engine-commits.txt](configs/engine-commits.txt)). The local hashes differ from the PR's because the patches were applied with `git am`; the diff is the same.
  - CMake: Release, `CMAKE_CUDA_ARCHITECTURES=120`, `-DSTRATA_Q6K_EXPERTS=ON`, everything else default. See [configs/cmake-options.txt](configs/cmake-options.txt).
  - Q6K experts are on because the UD-Q5_K_XL pack's expert table lists Q6_K (type 14) gate/up tensors for some layers; layer 2 appears in [configs/q5-pack-native_experts-excerpt.txt](configs/q5-pack-native_experts-excerpt.txt).
- **`strata-vision`:** built from `tools/vision` with `-DSTRATA_VISION_CUDA=ON -DSTRATA_PORTABLE=OFF`, sm_120.
- **Background:** the machine's other services (two Python web services, a chat gateway, Xvfb) were stopped for the whole run. Before the first cell the GPU showed 2 MiB used and no compute apps ([speed/progress.log](speed/progress.log)). Docker Engine was installed with apt while this run was going on; the logs don't show where that fell relative to the speed cells. The IQ3_S baseline at the end of the speed run is within 0.6% of the one at the start.

## Model and configuration

- **Weights:** [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF): `UD-Q4_K_XL` (4 shards), `UD-Q5_K_XL` (6 shards) and `Q8_0` (6 shards, the quality reference). Sizes and SHA-256 are checked against the Hugging Face LFS hashes in [configs/weights.md](configs/weights.md).
- **Packs:** made with `tools/iq_pack.py --compat-bf16`, as setup does for the unsloth family. The pack tool is byte-identical between 0.1.40.3 and 0.1.41, so packs made with 0.1.40.3's copy were reused.
  - `--compat-bf16` rounds the Q8_0 hyper-connection projections to BF16 in every unsloth pack, Q8_0's included. The reference is Q8_0 as Strata runs it.
- **Image encoder:** `mmproj-Qwen3.8-Flash-Next-BF16.gguf` from ISTA-DASLab, 907,543,008 bytes, SHA-256 `b1a82259...49bd0` ([configs/mmproj.sha256](configs/mmproj.sha256)), the file setup pins.
- **MTP draft head and expert profile:** setup's MTP draft head (`--mtp .../mtp/rt`) and `data/expert-profile.bin` from the 0.1.41 tree, the same for every cell.

Settings shared by every speed cell (full argument lists in `speed/*/config.json`):

```text
strata --pack <pack> --native <shard 1> --expert-profile data/expert-profile.bin --expert-cache auto
  --prefill auto --spec 4 --mtp <mtp/rt> --max-context 250000 --kv int8 --kv-resident 32768
  --pcie-frac 0.00 --spec-min-p 0.70 --conversation-cache-mib 6144 --conversation-cache-slots 3
  --conversation-cache-min-free-mib 2048 --resident-budget-gib 48
  [+ --vision --vram-reserve-mib 700 and a "vision" section with "gpu": true, in the +vision cells]
env STRATA_PF_FUSED=1
```

The +vision cells used the same `vision` section as the configuration in #967 (exe, mmproj, model = shard 1, `gpu: true`, `max_tokens: 1024`).

The speed and quality cells for UD-Q5_K_XL ran the fixed engine on a copy of shard 1 padded with 6 zero bytes, made before the fix existed. #1612 shows the fixed engine gives byte-identical output on the original and padded files.

## Speed

### Method

[scripts/strata_ablate.py](scripts/strata_ablate.py), as run.

- **Engine and warm-up:** a fresh engine per configuration, one discarded 32K warm-up request, then three measured runs at each prompt size.
- **Requests:** greedy (`temperature 0`), `reasoning_effort: none`, a 256-token cap. Every run generated 256 tokens.
- **Prompts:** identical in every cell, built with Strata's tokenizer for 4,096 and 32,768 tokens. The engine counted 4,052 and 32,692; the table uses the engine's counts.
- **No reuse:** every measured run read its whole prompt fresh (0 reused tokens in every engine line).
- **Prompt and decode tok/s:** from the engine's `prompt N tokens = 0 reused + N read in X ms (... tok/s), 256 generated in Y ms (... tok/s)` line.
- **TTFT:** measured at the client over the streaming API, to the first non-empty content or reasoning delta. It includes the prompt read.
- **Expert placement:** experts in the VRAM expert cache run GPU kernels, the rest CPU kernels from the 48 GiB page-locked RAM copy.
  - UD-Q4_K_XL: 48.7 GiB of its experts don't fit the VRAM cache, so the RAM copy holds nearly all of them.
  - UD-Q5_K_XL: 68.6 GiB don't fit, so the remainder are read from the model files through the OS file cache. The engine logs say so (`FileExpertSource: RAM budget 48.00 GiB ...`).
- **Monitoring:** GPU telemetry was sampled every 2 s (`speed/*/telemetry.csv`).

### Results

Median of 3 runs, range in brackets. Per-run numbers are in `speed/*/result.json`.

| Configuration | Prompt tokens (engine) | Prompt tok/s | Decode tok/s | TTFT s | Draft acceptance | VRAM expert slots |
| --- | ---: | --- | --- | --- | ---: | ---: |
| UD-Q4_K_XL | 4,052 | 2,230 [2,067-2,358] | 109.7 [92.4-111.6] | 1.83 [1.73-1.97] | 81% | 7,896 |
| UD-Q4_K_XL | 32,692 | 4,302 [4,289-4,365] | 111.8 [111.1-116.2] | 7.62 [7.52-7.65] | 84% | |
| UD-Q4_K_XL + images | 4,052 | 2,335 [2,239-2,390] | 101.2 [93.3-107.0] | 1.75 [1.71-1.82] | 81% | 7,323 |
| UD-Q4_K_XL + images | 32,692 | 4,164 [4,161-4,192] | 104.1 [100.9-106.6] | 7.88 [7.83-7.88] | 83% | |
| UD-Q5_K_XL | 4,052 | 1,267 [1,101-1,395] | 60.6 [60.0-61.0] | 3.22 [2.98-3.70] | 77% | 6,181 |
| UD-Q5_K_XL | 32,692 | 2,528 [2,326-2,740] | 64.4 [56.9-65.2] | 12.98 [11.97-14.09] | 81% | |
| UD-Q5_K_XL + images | 4,052 | 817 [661-1,152] | 51.8 [50.1-56.1] | 4.99 [3.53-6.17] | 77% | 5,733 |
| UD-Q5_K_XL + images | 32,692 | 2,421 [2,275-2,534] | 60.9 [47.9-61.1] | 13.55 [12.94-14.41] | 81% | |

- **Drift check:** the machine's own IQ3_S configuration (ISTA-DASLab GSQ-RCO IQ3_S with images) ran first and last in the same session.
  - It ran engine 0.1.39, not 0.1.41, so it isn't a like-for-like comparison with the rows above.
  - First run, 4K / 32K: decode 188.4 / 177.7 tok/s, prompt 5,288 / 6,428 tok/s.
  - Last run: decode 187.2 / 177.0, prompt 5,267 / 6,394.
  - Differences between the two: -0.6% / -0.4% decode, -0.4% / -0.5% prompt.
- **Memory:** VRAM peaked at 32,066-32,077 MiB in every 0.1.41 cell; the expert cache takes the free VRAM. Each engine allocated a 48.00 GiB page-locked RAM copy (`arena_mib` ~49,150 in `engine_info`). Peak GPU power: 460 W for UD-Q4_K_XL, 346 W for UD-Q5_K_XL.
- **Not measured:** 128K prompts on 0.1.41, and RAM budgets other than 48 GiB.

## Quality against Q8_0

### Method

[scripts/strata_quality.py](scripts/strata_quality.py), as run. This is the logprob method from `docs/UNSLOTH_Q4.md`.

- **Engine:** a serve engine run with `STRATA_LOGPOS=<file> STRATA_LOGPOS_TOPK=256`, `--short-read 1024` and `--adapt-every 100000`, at a 56 GiB RAM budget. It writes the true next token's log-probability and the 256 most likely tokens for every prompt position it reads through the verify windows.
- **Requests:** every quant read the same 30 requests ([quality/requests.json.gz](quality/requests.json.gz), SHA-256 of the uncompressed file in `requests.json.sha256`), each scoring about 560 tokens:
  - **prose:** wikitext-2-raw;
  - **code:** llama.cpp source at Strata's pinned `3cf0325`;
  - **agent:** NousResearch hermes-function-calling-v1.
  - The URLs and hashes are in [quality/REPORT.md](quality/REPORT.md).
  - The "short" requests are the scored tokens alone. The 8192 and 32768 requests put that many tokens of the same stream first.
- **Measures:**
  - KL(Q8_0 || quant) in nats, over Q8_0's top 256 plus one bucket for the rest;
  - top-1 agreement, at all positions and at "decisive" ones (where Q8_0's top-2 gap is at least 0.5 nats);
  - top-10 overlap;
  - perplexity of the true text.
- **Intervals:** 95% bootstrap over requests.
- **What it measures:** KL is a distance between token distributions on these texts. It is not a task benchmark score.

### Results (engine 0.1.41, 30 requests, 17,339 positions)

| Quant | Mean KL [95% CI] | Median KL | p99 KL | Top-1 same | Top-1 same (decisive) | Top-10 overlap | PPL quant / Q8_0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UD-Q4_K_XL | 0.0558 [0.0412, 0.0714] | 0.0040 | 0.827 | 94.0% | 96.9% | 89.8% | 52.81 / 51.87 |
| UD-Q5_K_XL | 0.0346 [0.0249, 0.0450] | 0.0021 | 0.501 | 95.4% | 97.9% | 92.2% | 54.34 / 51.87 |

Paired difference, UD-Q4_K_XL minus UD-Q5_K_XL, on the same positions: mean KL +0.0212 [+0.0122, +0.0307]. UD-Q4_K_XL is further from Q8_0 on 83% of the requests, and agrees with Q8_0's top-1 1.03 points less often at decisive positions.

By source and context length (mean KL; the full tables are in [quality/REPORT.md](quality/REPORT.md)):

| Quant | prose | code | agent | short | 8192 | 32768 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UD-Q4_K_XL | 0.0449 | 0.0660 | 0.0564 | 0.0401 | 0.0523 | 0.0796 |
| UD-Q5_K_XL | 0.0251 | 0.0363 | 0.0424 | 0.0217 | 0.0352 | 0.0509 |
| Q4 minus Q5 [95% CI] | +0.020 [+0.006, +0.036] | +0.030 [+0.012, +0.049] | +0.014 [+0.002, +0.027] | +0.019 [+0.008, +0.032] | +0.017 [+0.003, +0.035] | +0.029 [+0.010, +0.050] |

**Perplexity doesn't order the quants the way KL does.** Overall, UD-Q5_K_XL's perplexity is higher than UD-Q4_K_XL's (54.34 against 52.81). By category it moves both ways. Against Q8_0, UD-Q4_K_XL's perplexity is lower on prose, agent, 8192 and 32768 and higher on code and short. UD-Q5_K_XL's is higher on every category except agent. Both are reported as measured. Each request scores about 560 tokens, and the per-category perplexities range from 8.3 to 810.

**Floor from engine numerics.** These requests were also run on engine 0.1.40.3 with the same packs ([quality/quality-versions.txt](quality/quality-versions.txt)). Q8_0 on 0.1.40.3 against Q8_0 on 0.1.41 (same weights, two engine versions) gives a mean KL of 0.0224 and 96.6% top-1 agreement. For scale: that is a change of engine with the weights held fixed, and the quants sit 0.012 (UD-Q5_K_XL) and 0.033 (UD-Q4_K_XL) above it. No same-version floor (Q8_0 against itself with a different expert cache) was run on 0.1.41. 0.1.41's start-up log puts the effect of its new prompt CPU share at "mean KL ~0.004". On these requests, which are read in chunks under 1,024 tokens, the measured version-to-version difference is larger.

Both quants against the Q8_0 of the same version:

| Quant | KL vs Q8_0, 0.1.40.3 | KL vs Q8_0, 0.1.41 |
| --- | ---: | ---: |
| UD-Q4_K_XL | 0.0518 | 0.0558 |
| UD-Q5_K_XL | 0.0317 | 0.0346 |

The 0.1.40.3 logprob files aren't included; they're about 1 GB.

## Image check

[scripts/vision_check.py](scripts/vision_check.py) generates one picture: a red circle, a blue square and the number 4271 ([vision/q4/image.png](vision/q4/image.png)). It asks three questions through `/v1/chat/completions` with the image attached. The server ran each quant's production configuration: the settings above, at 48 GiB, with `--vision`.

| Quant | "What number is written…" | "What color is the circle…" | "What shape is the blue object…" | Passed |
| --- | --- | --- | --- | ---: |
| UD-Q4_K_XL | 4271 (4.98 s) | Red (1.25 s) | square (1.30 s) | 3/3 |
| UD-Q5_K_XL | 4271 (8.11 s) | Red (2.54 s) | square (2.37 s) | 3/3 |

Prompts were 412-413 tokens with the image. The first question's time includes the first image encoding. Per-question data is in `vision/*/result.json`.

## Correctness and limitations

- One machine, three runs per speed configuration, and 30 quality requests.
- The reference is unsloth's Q8_0 as Strata runs it (with `--compat-bf16`), not BF16. No task benchmarks (coding, tool use, recall) were run.
- The image check is one generated picture with three short answers. It shows the image path works on both quants; it says nothing about harder image questions.
- Without #1612, UD-Q5_K_XL needs a padded copy of shard 1 (`truncate -s 10946624`). The Q6_K expert build option was set for the whole run; this report doesn't test a build without it.
- At a 48 GiB budget on this machine, part of UD-Q5_K_XL's experts were read through the OS file cache. Its speed depends on how much free RAM the cache gets.
- The speed baseline is IQ3_S on engine 0.1.39. No IQ3_S cell was run on 0.1.41.
