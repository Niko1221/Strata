# Community benchmark: RX 6900 XT 16 GB (gfx1030) x1 and x2, Ryzen 5 5600X, 128 GB RAM - Strata 0.1.43.1: GSQ-RCO IQ3_S and Unsloth UD-Q4_K_XL, one card and the layer split, stock and with the open gfx103x pull requests

Measured on 2026-10-13 by Vop3p, on the machine and with the method of the
[0.1.42 report](../2026-10-11-community-rx-6900xt-ud-q4_k_xl-iq3_s-0.1.42/README.md) (#1927), so the two can be read
cell by cell. Ten arms, two runs per prompt length, greedy, 256 tokens out. Main limitation: two runs per cell, so
most decode differences between the releases are inside the run-to-run range.

The short answer:

- **0.1.43.1 reads prompts exactly as 0.1.42 did** on this card family, in every arm (within 1%).
- **Stock decode is unchanged** except the IQ3_S layer split with `--pipeline-windows 2`, whose medians are 4% / 8%
  higher at 4K / 32K (only the 32K ranges do not overlap; 128K is equal). The UD-Q4_K_XL split without pipeline
  windows is identical to 0.1.42 (43.2 / 45.7 / 47.0 against 43.3 / 45.6 / 47.0).
- **With the open pull requests** (the 0.1.42 report's set plus #2226, the persistent hyper-connection read) the
  medians decode 2-7% faster than the 0.1.42 report's "prs" arms, the prompt is unchanged. The bundle changed (0.1.43.1
  and #2226), so this is not attributed; #2226's own one-switch A/B is -4.2% / -2.8% ms per verify window.
- **UD-Q4_K_XL on the split wants the IQ3_S decode configuration too**: `--spec 3 --spec-min-p 0.7
  --pipeline-windows 2` decodes 9-16% faster than `--spec 4 --spec-min-p 0.5` on stock (50.3 / 53.2 / 51.4 against
  43.2 / 45.7 / 47.0). With that configuration the pull requests add prompt speed only (decode 50.9 / 51.9 / 51.2).
- No arm stalled, no request failed.

## Hardware and software

- 2x AMD Radeon RX 6900 XT 16 GB (gfx1030, Navi 21), each on PCIe 4.0 x8. One-card arms use the second card (PCI 0f,
  no display); split arms use both, the vision encoder off, as in the 0.1.42 report.
- AMD Ryzen 5 5600X (6 cores / 12 threads, AVX2, no AVX-512), 128 GB DDR4-3200 (4 DIMMs, FCLK 1600); model files
  on an NVMe SSD.
- GPU board power cap 245 W on both cards (default 255 W).
- Ubuntu 26.04, kernel 7.0.0-38-generic, ROCm 10.0 (HIP clang from `/opt/rocm/core-10.0`), rocBLAS 5.6.0.
- Strata **0.1.43.1** (tag `v0.1.43.1`, `a0116b27`), two trees, both built with `-DSTRATA_ENABLE_HIP=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON
  -DSTRATA_BUILD_TESTS=OFF` (Release), the pinned upstream ggml:
  - **stock**: the tag as is. Binary `02b536aa...`.
  - **prs**: the tag plus the same seven commits as the 0.1.42 report's prs tree (#1149, #1151, #1167 and two small
    local commits: a per-device rocBLAS solution cache, its Windows path) plus **#2226** (`STRATA_HC_PERSIST=1`).
    Binary `278c7a14...`.
- Server: `serve/server.py --engine strata` from each tree, Python 3.14 venv, port 8089, loopback.

## Model and configuration

- **GSQ-RCO IQ3_S**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` (native pack `iq3_s`, PLE shard, MTP draft `rt`), 131,072
  context, `--kv int8 --kv-resident 32768 --ple-io ram`, `--expert-cache auto --prefill auto`.
- **Unsloth UD-Q4_K_XL**: `Qwen3.8-Flash-Next-UD-Q4_K_XL` (unsloth, revision `38bb39e`; native pack built with
  `tools/iq_pack.py --compat-bf16`; `--ple-io direct`), the same context and cache settings; 71.7 GiB of experts in
  RAM against 46.8 GiB for IQ3_S.
- One card: `--spec 4 --spec-min-p 0.5`, `gpu: 1`. Layer split: `gpu: [1, 0]`, `layer_split: auto` (layers 0-23 /
  24-47 for both formats), `gpu_order: as_given`. IQ3_S split: `--spec 3 --spec-min-p 0.7 --pipeline-windows 2`.
  UD-Q4_K_XL split: `--spec 4 --spec-min-p 0.5` without pipeline windows (the 0.1.42 report's configuration, kept for
  the comparison) and, in the two `udq4-split-pw-*` arms, `--spec 3 --spec-min-p 0.7 --pipeline-windows 2` (this
  machine's current configuration for that model).
- Expert cache slots (the same in stock and prs, and as in 0.1.42 within one slot): one card IQ3_S 4,860, UD-Q4_K_XL
  2,830; split IQ3_S 6,036 + 4,770, UD-Q4_K_XL 3,536 + 3,010.
- **stock arms**: no environment variables at all.
- **prs arms**: `STRATA_HIP_PROMPT_F16=1 STRATA_SH_STREAM=1 STRATA_HIP_ADAPT_KERNEL_COPY=1 STRATA_SELECT_SGEMM=1
  STRATA_IO_THREADS=128 STRATA_PREFILL_CPU_SHARE=auto STRATA_HC_PERSIST=1`, the rocBLAS table and cache of #1167, and
  on the split `STRATA_SPLIT_MTP_BATCH=1` - the 0.1.42 report's set plus `STRATA_HC_PERSIST=1`. Each prs engine log
  has the `STRATA_HC_PERSIST=1: ... one persistent kernel` line once per card; no stock log has it.

## Method

[benchmark.py](benchmark.py) is the script of the earlier reports (unchanged): deterministic synthetic Python filler
with a different nonce near the start of each request, counted with Strata's tokenizer. Every arm started its own
server and ran `benchmark.py --targets 4096,32768,128000 --runs 2` (serial, increasing length, after one excluded
warm-up). Every speed request read its whole prompt (0 reused tokens) and generated 256 tokens. Prompt throughput is
freshly read tokens / `prompt_ms`, decode throughput the engine's `decode_tok_s`; TTFT is streaming over loopback.
`reasoning_effort: none`, temperature 0.

The arms ran one after another between 00:18 and 01:41 local time on 2026-10-13. Each folder has `config.json` (paths
rewritten: `/work/stock`, `/work/prs`, `/m`, `/bench`), `summary.json`, `results.json` (every request and output, the
engine's per-request accounting) and `BUILD.txt` (tree, binary SHA-256, environment, GFXOFF reading at start).
[table.py](table.py) prints the medians.

## Results

Each cell is the median **[minimum-maximum]** of two runs, tok/s; TTFT in seconds (median); the last column is the
lowest MemAvailable during the arm (sampled once a second), GiB. The 0.1.42 columns are that report's medians.

### One card (PCI 0f)

| arm | prompt 4K | 32K | 128K | decode 4K | 32K | 128K | TTFT 4K / 32K / 128K | RAM free |
|---|---|---|---|---|---|---|---|---|
| IQ3_S, stock | 477 [472-481] | 506 [505-506] | 489 [489-489] | 45.2 [42.6-47.9] | 47.1 [47.0-47.2] | 46.7 [46.2-47.2] | 8.6 / 64.9 / 261.8 | 66 |
| IQ3_S, prs | **967** [942-993] | **1,155** [1,152-1,157] | **1,130** [1,129-1,131] | 50.1 [46.7-53.5] | 54.9 [53.9-56.0] | 50.8 [49.9-51.6] | 4.3 / 28.4 / 113.4 | 66 |
| UD-Q4_K_XL, stock | 475 [471-478] | 506 [505-506] | 490 [490-491] | 29.0 [28.3-29.7] | 30.6 [30.5-30.8] | 31.0 [30.4-31.5] | 8.7 / 64.9 / 261.2 | 41 |
| UD-Q4_K_XL, prs | **681** [679-682] | **1,123** [1,120-1,125] | **1,089** [1,087-1,090] | 33.7 [32.3-35.1] | 33.8 [33.7-33.9] | 32.8 [32.3-33.2] | 6.1 / 29.3 / 117.7 | 41 |

| against 0.1.42 (medians) | prompt 4K / 32K / 128K | decode 4K / 32K / 128K |
|---|---|---|
| IQ3_S, stock | 476 / 504 / 488 -> 477 / 506 / 489 | 45.1 / 47.1 / 46.2 -> 45.2 / 47.1 / 46.7 |
| IQ3_S, prs | 959 / 1,153 / 1,129 -> 967 / 1,155 / 1,130 | 47.7 / 52.5 / 47.4 -> 50.1 / 54.9 / 50.8 |
| UD-Q4_K_XL, stock | 471 / 505 / 489 -> 475 / 506 / 490 | 29.5 / 31.3 / 31.5 -> 29.0 / 30.6 / 31.0 |
| UD-Q4_K_XL, prs | 676 / 1,117 / 1,087 -> 681 / 1,123 / 1,089 | 31.6 / 32.6 / 32.9 -> 33.7 / 33.8 / 32.8 |

### Layer split across both cards

| arm | prompt 4K | 32K | 128K | decode 4K | 32K | 128K | TTFT 4K / 32K / 128K | RAM free |
|---|---|---|---|---|---|---|---|---|
| IQ3_S, stock | 462 [457-467] | 764 [762-767] | 870 [867-873] | 73.6 [69.6-77.6] | 78.1 [76.3-80.0] | 71.8 [66.7-76.9] | 8.9 / 42.9 / 147.2 | 65 |
| IQ3_S, prs | **975** [953-997] | **1,782** [1,778-1,787] | **2,033** [2,028-2,038] | 79.4 [75.5-83.4] | 81.6 [77.5-85.6] | 77.7 [75.1-80.4] | 4.2 / 18.4 / 63.0 | 65 |
| UD-Q4_K_XL, stock, spec 4 / 0.5 | 463 [459-466] | 762 [761-764] | 870 [868-872] | 43.2 [40.3-46.1] | 45.7 [45.2-46.1] | 47.0 [46.8-47.1] | 8.9 / 43.0 / 147.3 | 40 |
| UD-Q4_K_XL, prs, spec 4 / 0.5 | **751** [744-757] | **1,739** [1,738-1,740] | **1,975** [1,973-1,977] | 47.6 [45.4-49.9] | 51.9 [50.4-53.3] | 49.2 [48.0-50.5] | 5.5 / 18.9 / 64.9 | 39 |
| UD-Q4_K_XL, stock, spec 3 / 0.7 + pipeline windows 2 | 462 [457-468] | 762 [761-763] | 869 [866-872] | **50.3** [48.0-52.6] | **53.2** [52.2-54.2] | **51.4** [50.9-51.9] | 8.9 / 43.1 / 147.4 | 40 |
| UD-Q4_K_XL, prs, spec 3 / 0.7 + pipeline windows 2 | **744** [735-754] | **1,737** [1,733-1,741] | **1,968** [1,957-1,978] | 50.9 [48.0-53.8] | 51.9 [50.2-53.5] | 51.2 [50.2-52.1] | 5.5 / 18.9 / 65.2 | 39 |

| against 0.1.42 (medians) | prompt 4K / 32K / 128K | decode 4K / 32K / 128K |
|---|---|---|
| IQ3_S, stock | 464 / 766 / 873 -> 462 / 764 / 870 | 70.6 / 72.4 / 71.8 -> 73.6 / 78.1 / 71.8 |
| IQ3_S, prs | 985 / 1,792 / 2,046 -> 975 / 1,782 / 2,033 | 76.6 / 81.9 / 74.5 -> 79.4 / 81.6 / 77.7 |
| UD-Q4_K_XL, stock, spec 4 / 0.5 | 462 / 763 / 873 -> 463 / 762 / 870 | 43.3 / 45.6 / 47.0 -> 43.2 / 45.7 / 47.0 |
| UD-Q4_K_XL, prs, spec 4 / 0.5 | 756 / 1,748 / 1,981 -> 751 / 1,739 / 1,975 | 44.6 / 51.1 / 48.7 -> 47.6 / 51.9 / 49.2 |

Notes:

- The prompt columns are the 0.1.42 report's to within 1% in all eight comparable arms; the pull requests still
  lift the prompt 1.4-2.3x on one card and 1.6-2.3x on the split (the low ends are UD-Q4_K_XL at 4K), as on 0.1.42.
- Stock decode: the only difference is the IQ3_S split, the one stock arm with pipeline windows. Its 4K and 128K
  ranges overlap 0.1.42's (64.2-77.0, 71.5-72.1); its 32K range does not (76.3-80.0 against 69.8-74.9). Two runs
  per cell are not enough to call it.
- prs decode: the medians are 2-7% above 0.1.42's prs arms in eight of twelve cells and within 2% in the other four;
  the two runs' ranges overlap in most. The bundle now also has #2226, whose one-switch A/B on this machine (one binary, 6 alternating rounds, IQ3_S
  split) is -4.2% / -2.8% ms per verify window at 4K / 32K.
- UD-Q4_K_XL on the split: `--spec 3 --spec-min-p 0.7 --pipeline-windows 2` instead of `--spec 4 --spec-min-p 0.5`
  is +16% / +16% / +9% decode on stock, prompt unchanged. With it, the prs arm decodes the same as stock (50.9 /
  51.9 / 51.2 against 50.3 / 53.2 / 51.4): the pull requests add prompt speed only there. The 0.1.42 report's split
  decode comparison of the two formats (UD-Q4_K_XL 35% slower) was made with the slower configuration for UD-Q4_K_XL;
  with the same configuration for both, UD-Q4_K_XL stock decodes 32% / 32% / 28% slower than IQ3_S stock.

## Stalls, memory, power

- Zero stalls in all ten arms (`stall report` / `timed out at layer` / `no progress for`: 0 in every `engine.log`),
  including every 128K request of the stock arms.
- VRAM at the end of each arm (`mem_info_vram_used`): one card 15,892-16,041 MiB on the card in use; split
  15,926-16,225 MiB on the first card and 15,812-16,041 MiB on the second.
- RAM: the lowest MemAvailable was 39 GiB (UD-Q4_K_XL split arms), 65-66 GiB for IQ3_S.
- Power: both cards ran with a 245 W board power cap (`power1_cap`; the default is 255 W). Clocks and fan
  settings were not recorded.

## Limitations

- One machine, two quantizations, one context size (131,072), a small synthetic greedy workload with a 256-token
  cap; no recall, vision, concurrency, sampling or thermal run. Two runs per cell.
- The "prs" arms bundle three open pull requests, two local commits, #2226 and the gfx103x switches; only the bundle
  was measured here.
- `--pipeline-windows 2` and `STRATA_SPLIT_MTP_BATCH=1` make greedy output differ between server starts on this
  machine.
- The two formats are not compared for answer quality here.

Developed with an AI coding assistant; every number above was measured on 2x RX 6900 XT (gfx1030, PCIe 4.0 x8 each) / Ryzen 5 5600X, ROCm 10.0.
