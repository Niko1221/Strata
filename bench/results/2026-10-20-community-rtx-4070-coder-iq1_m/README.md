# Community benchmark on NVIDIA GeForce RTX 4070 (sm_89, 12 GB)

Measured on 2026-10-20 by a community contributor (RTX 4070, Ryzen 7 5700X, 31 GiB RAM).
This is the **first measurement on an Ada (sm_89) card** and the first for the Coder IQ1_M on Ada; it
covers the decode and prompt paths of the shared-endpoint setup (a 12 GB card that keeps the experts in
RAM and streams the misses), plus a sweep of the engine's opt-in switches on this architecture.

**Read the noise floor first.** The stock configuration was measured twice, 40 minutes apart: decode
**44.1 vs 43.7 tok/s (0.9% apart)**, time to first token 0.77 vs 0.78 s, the 16K prompt rate 1532 vs
1634 tok/s (6.3%), and the **4K prompt rate 1825 vs 1418 tok/s (22.3% apart)**. So on this box the
decode numbers are reproducible to about 1%, the 16K prompt to about 6%, and **the 4K prompt number
is not a discriminator at all** - it moved 22% between two runs of the same configuration.

What that leaves, on the 16K prompt (the only prompt measure with usable resolution):

- `STRATA_DEQUANT_WIDE=1` - the wide dequantization, on by default only on cc 6.1 and 8.6 - is
  **+18% on the 16K prompt and -15% on its time to first token**, and it is bitwise equal to the
  default, so the answers do not change. It is the one switch here worth turning on.
- `--expert-cache-per-layer` **costs 15% on the 16K prompt and adds 18% to its time to first token**,
  and it changes which side (GPU or CPU) computes an expert, so a greedy answer can change in wording.
  It does not buy decode here (inside the 1% noise).
- Cutting the context to 96k recovers what the per-layer cache loses, but buys nothing over the stock
  configuration, so the 131,072-token window was kept.
- The sm_89 MMVQ-IL table and an expert profile rebuilt from the new model's routing traces changed
  nothing measurable on this model (see the note in *Correctness and limitations*).

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4070, 12282 MiB (46 SMs, sm_89); CPU: AMD Ryzen 7 5700X 8-Core Processor (8 cores / 16 threads, no AVX-512); installed RAM: 31.3 GiB, **no swap**; storage: NVMe (LUKS), 614 MB/s measured cold direct read; PCIe link: 26.2 GB/s host-to-device (the engine's own probe)
- OS: Linux 6.8.0-90-generic; driver 580.126.09; CUDA 13.0 toolkit, sm_89 build
- Strata commit: `90767252`; engine version 0.1.45.3; **source build** (the engine was compiled here, not a release package)
- Background workloads: none during the measurements; the box is a desktop, no power limit set.

## Model and configuration

- Model repository and revision: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` at revision
  `bba88856f8c3c0c9e40e45f82d05e228b814ac43` (2026-10-19, the release that replaced the first shard);
  quantization IQ1_M; GGUF files `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` (the model) and
  `...-00002-of-00002.gguf` (the 28.8 GB n-gram table, byte-identical to the previous release)
- Vision encoder: `mmproj-Qwen3.8-Flash-Next-BF16.gguf` on the **CPU** (8 threads, the card is under 24 GB);
  custom pack: `tools/iq_pack.py --compat-bf16 --experts-bin` (native experts, 25,146,163,200 bytes);
  expert profile: the shipped `expert-profile-coder.bin`
- Context 131072 tokens; KV `int8` (no KV streaming: the RAM is full); expert cache 1864 slots / 3672 MiB; prefill chunk 7168 tokens; low-RAM mode: **on** (22857 MiB of experts resident in RAM, the rest read from the file)
- MTP: on, draft head over 4 windows, `--spec 4 --spec-min-p 0.5`; reasoning: the model's own; sampling: temperature 0.6, top_p 0.95, top_k 20; no calibration, no experimental speed projection

```text
strata --pack coder-iq1_m-new --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --resident-experts --vision --vram-reserve-mib 700
```

## Method

`python tools/strata_bench.py --config <config> --runs 5` (the repository's own harness), one fresh
engine per configuration, thinking off, temperature 0. The suite sends: three short chats (up to 100
generated tokens), a fresh 4,096-token prompt and a fresh 16,384-token prompt (each carries a unique
tag, so no prompt is ever served from the cache), and an agent session that grows a conversation with
a project file. Prompt and decode rates are the engine's own `prompt_per_second` /
`predicted_per_second`; the time to first token is measured by the client.

Per-request data is in `results.json` (one entry per request, with the token counts and the engine's
timings). The engine's start-up lines are in `engine-log-excerpt.txt`. `matrix-summary.md` holds the
wider switch sweep (48 configurations) that the numbers above come from, with its caveat.

## Results

| Configuration | Prompt tokens (4K / 16K) | Reused tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| stock settings (the engine's own defaults) | 4019 / 16058 | 0 | 100 (chat) | 15 | 4K: 1336-1868 | 38.1-46.8 | 0.66-0.82 |
| stock settings + `--expert-cache-per-layer` | 4019 / 16058 | 0 | 100 (chat) | 15 | 4K: 1478-1879 | 38.9-46.3 | 0.65-0.83 |
| stock settings + `STRATA_DEQUANT_WIDE=1` | 4019 / 16058 | 0 | 100 (chat) | 15 | 4K: 1173-1952 | 37.4-48.5 | 0.65-0.83 |
| stock settings + `--expert-cache-per-layer` + `STRATA_DEQUANT_WIDE=1` | 4019 / 16058 | 0 | 100 (chat) | 15 | 4K: 1608-1976 | 39.1-47.2 | 0.64-0.82 |
| the same, with `--max-context 98304` | 4019 / 16058 | 0 | 100 (chat) | 15 | 4K: 1673-1958 | 40.7-47.8 | 0.62-0.79 |
| the same, plus the sm_89 MMVQ-IL table (`STRATA_MMVQ_IL_ROWS`) | 4019 / 16058 | 0 | 100 (chat) | 15 | 4K: 1570-1984 | 39.1-48.0 | 0.64-0.83 |
| the same, plus an expert profile rebuilt from the new model's routing traces | 4019 / 16058 | 0 | 100 (chat) | 15 | 4K: 1554-1945 | 39.4-52.9 | 0.65-0.81 |
| the stock settings again, later in the session (the drift check) | 4019 / 16058 | 0 | 100 (chat) | 15 | 4K: 1264-1827 | 40.2-47.0 | 0.65-0.83 |

16K prompt: prompt tok/s and time to first token, same runs:

| Configuration | Prompt tok/s median (range) | TTFT s median (range) | vs the stock settings |
| --- | --- | --- | --- |
| stock settings (the engine's own defaults) | 1482-1644 | 9.80-10.87 |  |
| stock settings + `--expert-cache-per-layer` | 1240-1327 | 12.13-12.98 |  (-18%) |
| stock settings + `STRATA_DEQUANT_WIDE=1` | 1577-1854 | 8.69-10.21 |  (+10%) |
| stock settings + `--expert-cache-per-layer` + `STRATA_DEQUANT_WIDE=1` | 1288-1485 | 10.85-12.50 |  (-11%) |
| the same, with `--max-context 98304` | 1456-1550 | 10.38-11.06 |  (-4%) |
| the same, plus the sm_89 MMVQ-IL table (`STRATA_MMVQ_IL_ROWS`) | 1316-1557 | 10.34-12.24 |  (-8%) |
| the same, plus an expert profile rebuilt from the new model's routing traces | 1343-1414 | 11.39-11.99 |  (-12%) |
| the stock settings again, later in the session (the drift check) | 1580-1840 | 8.76-10.19 |  (+9%) |

Draft acceptance (the MTP draft head's proposals that the target accepted), median over the chat runs:

| Configuration | Drafts accepted |
| --- | ---: |
| stock settings (the engine's own defaults) | 57% |
| stock settings + `--expert-cache-per-layer` | 52% |
| stock settings + `STRATA_DEQUANT_WIDE=1` | 54% |
| stock settings + `--expert-cache-per-layer` + `STRATA_DEQUANT_WIDE=1` | 59% |
| the same, with `--max-context 98304` | 56% |
| the same, plus the sm_89 MMVQ-IL table (`STRATA_MMVQ_IL_ROWS`) | 58% |
| the same, plus an expert profile rebuilt from the new model's routing traces | 59% |
| the stock settings again, later in the session (the drift check) | 56% |

## Correctness and limitations

- **The answers are not compared between configurations here.** The per-layer expert cache changes
  which side (GPU or CPU) computes an expert, and the two round differently, so a greedy answer can
  change in wording while staying equivalent in substance. Five code prompts were checked by hand for
  that. The wide dequantization is bitwise equal to the default and cannot change an answer.
- No recall (needle) test is included in this report.
- The **sm_89 MMVQ-IL table** (a separate PR) and an **expert profile rebuilt from this model's own
  routing traces** were both measured here and neither changed the numbers measurably: the IL table
  moves the head's dense projections, and the bottleneck on this card is the expert path, not the head;
  the profile is only a tie-breaker in the engine's cache order (the learned routing heat comes first,
  `src/core/expert_cache.cpp:160-164`), so a profile that no longer matches the shipped model is
  self-correcting at runtime.
- The VRAM budget is the binding constraint on this card: with everything loaded the engine reports
  **209 MiB of VRAM free** and warns that requests may stall. That is why the
  KV format and the context length move the numbers more than any kernel switch does.
- The CPU pool is the other half of the decode path: about a fifth of the routed experts per token are
  computed on the 8 CPU cores (the engine measures 123 us per missed expert here), so the PCIe share
  (`--pcie-frac`) trades decode against prompt reading and was measured as such in the sweep.
- Every number here is from one machine, one driver and one model, with 5 runs per test. The
  run-to-run spread of this box is about 1% for decode, 6% for the 16K prompt and 22% for the 4K prompt
  (the same configuration, measured twice).

