# Strata on an Intel Arc: performance

Every speed measured for the Intel path, kept in one place. [INTEL.md](INTEL.md) explains the port, its setup and
why things are the way they are; this file holds the numbers. It has three parts:

- **Current numbers:** the latest build.
- **History:** what each change bought, with dates.
- **Other people's cards:** results posted on the upstream PR.

**Our machine.**

- Card: Arc Pro B70 (32 GB, Xe2, 608 GB/s), in a PCIe 3.0 x8 slot. The card trains at Gen3 x8 there; it can do
  Gen5 x16.
- Host: Ryzen (Zen 1), Ubuntu 24.04, 23 GB of RAM given to the engine.
- Toolchain: oneAPI DPC++ 2026.1 and an AOT build (`AOT=bmg-g31`).

**How the engine numbers are measured.**

- Unless a row says otherwise: a greedy run of the engine (INTEL.md, "How to run it by hand"), 256 new tokens,
  MTP draft layer on (`--spec 4 --spec-min-p 0.5 --mtp`), `--stream-experts`.
- Prompts:
  - The 20-token prompt (written "19 tokens" in older rows): "Write a Python function that returns the n-th
    Fibonacci number".
  - The 2,185-token prompt (written "2,184 tokens" in older rows: the engine reports 2,184 prompt tokens read): a
    fixed code-and-prose text.
  - Both are in `sycl/bench/v1`.
  - Long prompts: that same text repeated to length.
- Decode with speculative decoding depends on the text: code drafts well, prose less so.
- "Through the API" rows use the served model with its sampling defaults (temperature 0.6, top_p 0.95, top_k 20,
  repetition penalty 1.05).

## Current numbers (engine 0.1.38-sycl, 2026-10-03)

### Benchy v1 on our B70 (the served configs, 2026-10-03)

`sycl/benchy.sh`, unchanged ("Submitting numbers" below): each model with the serve config it is actually served
with, from a cold page cache. Its report, as written:

| Model / configuration | Input tokens | PP (tok/s) | TTFT (s) | TG (tok/s) | Drafts accepted | Actual output tokens | Completed requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| coder-iq1_m, 256K context | 20 | 13.44 | 1.7 | 76.40 | 74% | 256 | 1/1 |
| coder-iq1_m, 256K context | 2,185 | 478.60 | 4.9 | 73.56 | 73% | 256 | 1/1 |
| coder-iq1_m, 256K context | 8,000 | 818.88 | 10.1 | 71.18 | 73% | 256 | 1/1 |
| coder-iq1_m, 256K context | 40,000 | 1,006.09 | 40.1 | 65.30 | 70% | 256 | 1/1 |
| coder-iq1_m, 256K context | 128,000 | 910.09 | 140.9 | 62.61 | 74% | 256 | 1/1 |
| coder-iq1_m, 256K context | 256,000 | 779.43 | 328.8 | 54.29 | 68% | 256 | 1/1 |
| iq2_xs, 128K context | 20 | 11.12 | 2.0 | 53.39 | 75% | 256 | 1/1 |
| iq2_xs, 128K context | 2,185 | 297.12 | 7.7 | 63.51 | 78% | 256 | 1/1 |
| iq2_xs, 128K context | 8,000 | 566.28 | 14.4 | 64.37 | 79% | 256 | 1/1 |
| iq2_xs, 128K context | 40,000 | 706.51 | 56.9 | 59.38 | 70% | 256 | 1/1 |
| iq2_xs, 128K context | 128,000 | 670.67 | 191.2 | 56.88 | 77% | 256 | 1/1 |
| swift-iq2_xs, 128K context | 20 | 11.09 | 2.0 | 52.29 | 74% | 256 | 1/1 |
| swift-iq2_xs, 128K context | 2,185 | 301.63 | 7.5 | 62.01 | 69% | 256 | 1/1 |
| swift-iq2_xs, 128K context | 8,000 | 572.62 | 14.3 | 64.47 | 77% | 256 | 1/1 |
| swift-iq2_xs, 128K context | 40,000 | 706.44 | 56.9 | 52.50 | 66% | 256 | 1/1 |
| swift-iq2_xs, 128K context | 128,000 | 668.57 | 191.7 | 56.23 | 69% | 256 | 1/1 |

| Model / configuration | Input tokens | Experts in VRAM | Offloaded to the RAM mirror | Slots lent to the prompt | Peak VRAM (GB) | RAM (GB) | SSD read at load (GB) | SSD read, request (GB) | PLE rows from SSD (MB) | Load (s) | Avg power (W) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coder-iq1_m | 20 | 12,288 (23.4 GiB) | 0 (0.0 GiB) | 294 | 29.6 | 6.1 | 30.7 | 0.18 | 18 | 38 | 167 |
| coder-iq1_m | 2,185 | 12,288 (23.4 GiB) | 0 (0.0 GiB) | 808 | 29.6 | 6.1 | 30.8 | 0.86 | 125 | 39 | 181 |
| coder-iq1_m | 8,000 | 12,288 (23.4 GiB) | 0 (0.0 GiB) | 1,186 | 29.6 | 6.1 | 30.5 | 1.63 | 122 | 37 | 185 |
| coder-iq1_m | 40,000 | 12,288 (23.4 GiB) | 0 (0.0 GiB) | 1,186 | 29.6 | 6.2 | 30.5 | 1.65 | 123 | 37 | 212 |
| coder-iq1_m | 128,000 | 12,288 (23.4 GiB) | 0 (0.0 GiB) | 1,186 | 29.7 | 6.1 | 30.5 | 1.66 | 122 | 37 | 218 |
| coder-iq1_m | 256,000 | 12,288 (23.4 GiB) | 0 (0.0 GiB) | 1,186 | 29.7 | 6.2 | 30.6 | 1.59 | 122 | 39 | 226 |
| iq2_xs | 20 | 18,487 (24.8 GiB) | 6,089 (8.2 GiB) | 291 | 30.1 | 11.9 | 42.9 | 0.46 | 17 | 48 | 143 |
| iq2_xs | 2,185 | 18,487 (24.8 GiB) | 6,089 (8.2 GiB) | 880 | 30.1 | 12.2 | 42.8 | 1.41 | 126 | 47 | 154 |
| iq2_xs | 8,000 | 18,487 (24.8 GiB) | 6,089 (8.2 GiB) | 1,400 | 30.1 | 12.3 | 42.8 | 2.23 | 120 | 47 | 161 |
| iq2_xs | 40,000 | 18,302 (24.6 GiB) | 6,274 (8.4 GiB) | 1,400 | 29.9 | 12.4 | 42.8 | 2.42 | 122 | 46 | 180 |
| iq2_xs | 128,000 | 18,302 (24.6 GiB) | 6,274 (8.4 GiB) | 1,400 | 30.0 | 12.5 | 42.9 | 2.33 | 120 | 47 | 188 |
| swift-iq2_xs | 20 | 18,487 (24.8 GiB) | 6,089 (8.2 GiB) | 291 | 30.1 | 12.0 | 42.9 | 0.36 | 18 | 47 | 159 |
| swift-iq2_xs | 2,185 | 18,487 (24.8 GiB) | 6,089 (8.2 GiB) | 880 | 30.1 | 12.0 | 43.0 | 1.29 | 128 | 48 | 153 |
| swift-iq2_xs | 8,000 | 18,487 (24.8 GiB) | 6,089 (8.2 GiB) | 1,400 | 30.1 | 12.2 | 43.0 | 2.19 | 121 | 49 | 167 |
| swift-iq2_xs | 40,000 | 18,487 (24.8 GiB) | 6,089 (8.2 GiB) | 1,400 | 30.1 | 12.2 | 42.8 | 2.38 | 124 | 45 | 178 |
| swift-iq2_xs | 128,000 | 18,487 (24.8 GiB) | 6,089 (8.2 GiB) | 1,400 | 30.3 | 12.3 | 42.8 | 2.34 | 122 | 45 | 188 |

System and build:
- Intel(R) Arc(TM) Pro B70 Graphics, 32 GB, PCIe Gen3 x8 (card max Gen5 x16).
- AMD Ryzen 7 1700X Eight-Core Processor (16 threads), 23.4 GiB RAM; models on nvme0n1 (PCIe SSD).
- Ubuntu 24.04.5 LTS, kernel 7.0.0-31-generic.
- Engine 0.1.38-sycl, commit `f530f9f`, image `strata-sycl-dev 4c52c3a3d252`, `build-sycl-aot/strata`. Each configuration's engine args: `config-<name>.txt`.

Caveats:
- One run per row, greedy, 256 new tokens, a fresh engine each time; the page cache dropped before every run (cold start).
- TG includes speculative decoding (the MTP draft layer): it depends on the text, and on how often drafts are accepted.
- RAM is the drop in the host's available memory (pinned memory included); VRAM is every client's resident VRAM from fdinfo, less what was in use before.
- Skipped (context too small): iq2_xs 256,000 (needs --max-context 256256), swift-iq2_xs 256,000 (needs --max-context 256256).

The configs (`config-*.txt`):

- `coder-iq1_m`: `--pack /work/Strata-data/packs/coder-iq1_m --native /work/models/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf /work/models/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile /work/Strata_B70/data/expert-profile-coder.bin --expert-cache auto --prefill 4096 --spec 4 --spec-min-p 0.5 --mtp /work/Strata-data/mtp/rt --max-context 262144 --kv int8 --stream-experts --vram-reserve-mib 2048 --kv-resident 32768`
- `iq2_xs`: `--pack /work/Strata-data/packs/iq2_xs --native /work/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf /work/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile /work/Strata_B70/data/expert-profile.bin --expert-cache auto --prefill 4096 --spec 4 --spec-min-p 0.5 --mtp /work/Strata-data/mtp/rt --max-context 131072 --kv int8 --stream-experts --vram-reserve-mib 2048 --kv-resident 32768`
- `swift-iq2_xs`: `--pack /work/Strata-data/packs/swift-iq2_xs --native /work/models/SWIFT-IQ2_XS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf /work/models/SWIFT-IQ2_XS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --expert-profile /work/Strata_B70/data/expert-profile.bin --expert-cache auto --prefill 4096 --spec 4 --spec-min-p 0.5 --mtp /work/Strata-data/mtp/rt --max-context 131072 --kv int8 --stream-experts --vram-reserve-mib 2048 --kv-resident 32768`

- **Short prompts read slowly here, by design of the configs.** Above 32K context the prompt path borrows cache
  slots (about a second per prompt), and every run starts cold, so the PLE rows come from the SSD. The 32K test
  config reads the same 2,185 tokens at 784-799 tok/s (below).
- **The Swift config's `--ple-gguf`** names shard 1 of its GGUF. Swift's shards split differently from the other two
  models, and the engine finds its per-layer embedding table (320,001,536 rows) in shard 1.

### At a 32K context (the engine's test config, 2026-10-03)

The engine's own test runs ("How the engine numbers are measured" above): 32K context, `--vram-reserve-mib 1024`, no
borrowing, warm page cache.

| model | prompt | decode | prompt reading |
|---|---|---|---|
| Coder IQ1_M (32K, INT8 KV) | 20 tokens | **78.5 tok/s** | - |
| Coder IQ1_M (32K, INT8 KV) | 2,185 tokens | **76.3 tok/s** | **784-799 tok/s** (three runs) |
| original IQ2_XS (32K) | 20 tokens | 58.9 tok/s | - |
| original IQ2_XS (32K) | 2,185 tokens | 64.3 tok/s | 557 tok/s |
| Swift 1.5 IQ2_XS | short questions | 48 tok/s (raw decode, 2026-10-01) | - |

- **Where the experts live:**
  - Coder: all 12,288 experts in VRAM.
  - IQ2_XS: 18,329 of 24,576 in VRAM (24.6 GiB); the other 6,247 in the pinned host mirror (8.4 GiB).
- **Swift 1.5:** setup's `swift` family. Correct answers, 51-118 words of thinking on short questions.

### Long contexts, Coder IQ1_M

Setup's flags: `--kv-resident 32768 --vram-reserve-mib 2048 --prefill 4096`. The prompt path borrows cache slots,
which is the default above 32K, and stream-all is on. 256 greedy tokens follow the prompt.

| context | KV | prompt reading | decode after | peak VRAM | experts in the host mirror |
|---|---|---|---|---|---|
| 40K | int8 | **1,160 tok/s** | 69 tok/s | - | none |
| 128K | int8, KV streaming | 920 tok/s | **67 tok/s** | 29.9 GB | none |
| 128K | k8v4, KV in VRAM | **1,041 tok/s** | 66.8 tok/s | 30.7 GB | 103 (0.2 GB) |
| 128K | q4_0, KV streaming | 848 tok/s | 62.8 tok/s | 29.7 GB | none |
| 256K | int8, KV streaming | 780 tok/s | **55 tok/s** | 30.2 GB | none |
| 256K | k8v4, KV in VRAM | 823 tok/s | 52.2 tok/s | 30.9 GB | 865 (1.7 GB) |
| 256K | q4_0, KV streaming | 737 tok/s | 51.7 tok/s | 30.0 GB | none |

- **Measurement dates:**
  - 40K, int8 and k8v4 prompt reading: 2026-10-03, with stream-all.
  - Everything else: 2026-10-01, on 0.1.32-sycl. Later builds reproduce those runs exactly; stream-all only
    changes the prompt column.
- **k8v4:** keeps its KV in VRAM, because it has no KV streaming yet. That pushes experts into the host mirror.

### Stream-all against routed-only (Coder, 2026-10-03)

Outputs are identical either way.

| prompt | routed-only | stream-all (the default when the VRAM holds more than 90% of the experts) |
|---|---|---|
| 40K int8 | 1,106 tok/s | 1,160 |
| 128K int8, KV streaming | 897 | 920 |
| 128K k8v4 | 1,006 | 1,041 |
| 256K int8, KV streaming | 760 | 780 |

The IQ2_XS keeps a quarter of its experts in the RAM mirror. With stream-all forced on, its 2,184-token prompt fell
from ~560 to 254 tok/s, which is why stream-all is gated.

### Prompt-slot borrowing

Borrowing lends VRAM cache slots to the prompt path and refills them after the prompt.

| | prompt | decode after |
|---|---|---|
| 40K, borrowing (default above 32K) | 1,111 tok/s | **69.1 tok/s** |
| 40K, `--no-prefill-borrow` | 1,192 tok/s | 65.6 tok/s |
| 2,184 tokens, borrowing | 610 tok/s | - |
| 2,184 tokens, no borrowing (default up to 32K) | 792 tok/s | - |

- The rows above: 0.1.33-sycl, 2026-10-01.
- **Without borrowing, by context** (0.1.31-sycl): ~1,000-2,300 experts move to the host mirror. The prompt reads
  2-5% faster and decode after it is 10-23% slower:

  | context | int8 | q4_0 | k8v4 |
  |---|---|---|---|
  | 128K | 934 / 59.5 | 887 / 53.4 | 1,026 / 54.4 |
  | 256K | 784 / 48.7 | 763 / 43.2 | 838 / 42.3 |

  Each cell is prompt reading / decode after, in tok/s.

### Through the API (served model, sampling on)

| | |
|---|---|
| a ~190-token chat answer, prompt included | 48-72 tok/s (the first request after a start is the slower one) |
| 5,000-18,000-token answers (2026-09-30 build) | 55-59 tok/s |
| code (continuing the Fibonacci function, 77-85% of drafts accepted) | 45-51 tok/s (2026-09-30 build) |
| a prose chat answer (55-72% accepted) | 30-39 tok/s (2026-09-30 build) |
| logprobs on (`"logprobs": true, "top_logprobs": 3`, IQ2_XS) | no measurable time; "Paris" at -0.0019 (p 0.998) |

Sampling and the repetition penalty cost nothing measurable.

### Start time (cold page cache, 2026-10-02)

| | before (serial fill) | now (pipelined fill) |
|---|---|---|
| Coder: expert cache fill (23.4 GiB) | 76.2 s (0.33 GB/s) | **18.8 s** (1.34 GB/s) |
| Coder: launch to first token | 82 s | **26 s** |
| IQ2_XS: launch to first token | 120 s | **41 s** (its 8.2 GB host mirror is ~9 s of that) |

### VRAM and RAM

| | |
|---|---|
| Coder, 32K | all 12,288 experts resident with `--vram-reserve-mib 1024`; ~1.9 GB free with everything loaded |
| reserve too small (1,536 MiB, 2026-09-30) | the cache came up 128 experts short: decode fell to 5-10 tok/s (those experts are read from the SSD and computed on the CPU) |
| host RAM | no host copy of the experts (`--stream-experts`): the engine runs in 23 GiB; upstream needs 32 GB for this model |
| IQ2_XS | host RAM never below 12 GB free with an 8.4 GiB host mirror |

### Where the time goes

| prompt | expert dequant | expert GEMMs | attention | other |
|---|---|---|---|---|
| 2,184 tokens (2026-09-30) | 28% | 22% | 7% | host grouping 9.5%, embeddings + PLE rows 13% |
| 8,000 tokens (2026-09-30) | 27% | 20% | 14% | host grouping 9%, embeddings + PLE rows 5% |
| 40K tokens (2026-10-03) | 15.7% | 23.4% | 19.2% | QSA select 6.8%, host grouping 6.5%, DeltaNet recurrence 5.8%, gather 5.7%, hyper-connection reads 4.6%, combine 2.4% |

- **An 80,000-token prompt** (GPU time, 2026-09-30):
  - expert down GEMM 26.0 s, attention 15.4 s, dequant 12.7 s;
  - QSA block selection 9.3 s (0.2 s at 8K: it scans every block of the context per query);
  - gate/up GEMM 8.9 s, gather 6.1 s, per-layer grouping sync 5.2 s, GDN recurrence 4.8 s;
  - the PLE rows are free at this scale (97.6% row-cache hits).
- **A decode round** (2026-09-30): 80-85% kernel time, with ~5 us of launch gap per node over 2,400-2,600 nodes.
  - The expert dot kernels are ALU-bound (77% XVE active).
  - The dense projections are memory-latency bound (92-128 GB/s at 84-93% occupancy).

## llama.cpp on the same card (2026-09-29, Coder IQ1_M, 32K context, q8_0 KV)

This is the comparison point: llama.cpp's SYCL backend serving the same GGUF.

| | llama.cpp SYCL | Strata SYCL port (32K test config) |
|---|---|---|
| load | ~100 s | 26 s |
| decode | 23-25 tok/s; GPU 92% busy at 165 W, CPU idle | 76-78 tok/s |
| prompt reading | 149 tok/s on 2,701 tokens; 424 tok/s on 104,798 tokens (131K context, read in 247 s) | 784-799 tok/s on 2,184; 920 at 128K |
| VRAM | 28.4 of 32 GB | ~30 of 32 GB (every expert resident) |
| longest context measured | 131K | 256K |

- **llama.cpp's speed by prompt size:** prompt reading speeds up with size, because the work batches better. A full
  128K window costs about five minutes to read.
- **llama.cpp's VRAM by context:** see INTEL.md, "How much context fits".
- **Quality:** correct code on every test.

## History

The numbers below are dated. Each was the state of the port at the time, and later rows supersede earlier ones.

### First end-to-end run (2026-09-29)

Setup: 2,185-token prompt, 64 greedy tokens, AOT build, `--no-prefill-borrow`.

| | llama.cpp SYCL | Strata SYCL port |
|---|---|---|
| prompt reading | 138-319 tok/s | 560 tok/s (693 at 2,000 tokens, 180 at 300) |
| decode, suffix drafter only (`--spec 2`) | 24.2-26.0 tok/s | 20.2-20.8 tok/s |
| decode, MTP draft layer (`--spec 4 --mtp`) | - | 42.7 tok/s (81% accepted, 2.9 tokens/round); 45.3 on a short prompt |

- **Rounds:** a speculative round cost ~53 ms whatever its size.
- **Suffix drafter:** accepted 9% of the time. `--spec 4` gave 16.9 tok/s against 20.8 for `--spec 2`.
- **Kernel bandwidth:** the kernels ran at 130-160 GB/s of weights. The card streams 600 GB/s
  (`sycl/probe/bw.cpp`).
- **MTP draft layer:** drafted for the Coder at 87% acceptance, 3.4 tokens per 53 ms round, 45 tok/s.
- **JIT:** without AOT, compiling costs ~47 s on the first run.

### After the 0.1.25-0.1.27 merge (2026-09-30)

| prompt | prompt reading | decode | drafts accepted |
|---|---|---|---|
| 19 tokens (64 greedy tokens) | - | 46.5 tok/s | 85% |
| 2,184 tokens | 566 tok/s | 38.3 tok/s | 77% |

The draft layer's prompt pass took 38.6 ms for 2,184 tokens, down from 105 ms for 19.

### Speed work (2026-09-30)

| change | effect |
|---|---|
| window graphs captured at load | the first request no longer pays 250-290 ms of captures |
| commit graph overlapped with the drafter's round | decode 38.3 -> 43.0 tok/s at 2,184 tokens, 46.5 -> 49.8 short |
| upstream 0.1.29 merged | 2,184-token prompt 765 -> 792 tok/s, decode there 43.1 -> 45.2 |
| draft layer's batched prompt pass under 64 rows | its cost on 19 tokens 106 -> 16 ms |
| one device module per kernel | first prompt-path launch 245 -> 1 ms |
| vector stores in the expert dequant | dequant per expert 0.085 -> 0.030 ms; prompt 496 -> 575 tok/s at 2,184, 720 -> 841 at 8,000 |
| PLE row reader 16 -> 64 threads | the rows of a 2,184-token prompt 466 -> 330 ms (the drive tops out near 85k IOPS) |
| a short first prompt chunk (256 tokens) | 2,184 tokens 711 -> 765 tok/s, 8,000 tokens 857 -> 986; ~150 ms less to first token |

- **Draft policy sweep** (2,184-token prompt, 128 tokens):

  | setting | decode |
  |---|---|
  | `--spec 4` | 41.6-43.5 tok/s |
  | `--spec 2` | 33.6 tok/s |
  | `--spec 6` | 35-37 tok/s |
  | `--spec-min-p` 0.3-0.7 | within noise |

- **Two-speed runs.** Identical greedy runs decoded at either ~45 or ~39 tok/s.
  - The slow runs pay one 226 ms PLE read stall in the first decode round.
  - Prompt time plus decode time was the same either way (4.95-5.17 s).
  - So compare runs on time to first token plus decode.

### The Q6_K alignment fix (2026-09-30)

A Q6_K block is 210 bytes, so its runs start only 2-byte aligned. The fix loads them as two aligned 16-byte loads
plus a shift.

| shape (`mmvq_bench`) | before | after |
|---|---|---|
| 2560 -> 10240 at 1 / 2 / 4 / 6 columns | 160 / 145 / 141 / 101 GB/s | 493 / 432 / 331 / 273 GB/s |
| the 248K-row head, 1 column | 150 GB/s | 407 GB/s |
| Coder decode (2K prompt) | 44.9 tok/s | 54 tok/s |

- **Repacked layout:** the same weights repacked at a 224-byte stride ran 2.3-4.7x faster (`q6k_align_bench`).
- **Other block types, before their own fixes:**
  - IQ4_XS (136 B) ran at ~100 GB/s.
  - Q4_K/Q5_K ran at 230-280 GB/s.
  - The dense decode kernels streamed 100-280 GB/s of the 608 GB/s.
- **SIMD16 builds** (`mmvq_sg_bench`), measured alone:
  - up to 1.45x on IQ4_XS and 1.3x on the 640-row shared-expert projections;
  - 10-25% slower on the large K-quant kernels;
  - no gain in the engine.

### Decode round 2 (2026-09-30 to 2026-10-01)

Each step was measured on its own: 19-token prompt, 256 tokens, two runs each.

| | decode |
|---|---|
| after the Q6_K alignment fix | 53.9 tok/s |
| + `resident_plan` grouping in parallel | 57.1 |
| + the GR down kernel reads activations directly | 58.7 |
| + wide 16-byte-load kernels for Q4_K, Q5_K, IQ4_XS | 62.6 |
| + wide kernels for Q8_0 and IQ4_NL (shexp down); Q6_K activation loads aligned | 65.6 |
| + the GR down projection sliced by columns | 70.6 |
| + IQ4_NL expert dots (the down projection in 39 of 48 layers) | 76.1 |
| + GR norm per (token, stream); aligned loads in the IQ3_XXS/IQ3_S/IQ2_S gate/up dots | **77.9** |

Kernel level:

| kernel | before | after |
|---|---|---|
| Q8_0, 2560 x 10240, 2 columns | 189 us | 31 us |
| IQ4_NL, 640 -> 2560 | 28 us | 6.6 us |
| GR read, 6 tokens (sliced down) | 108.6 us | 76.5 us |
| IQ4_XS, 2 columns | 107 GB/s | 323 GB/s |
| IQ4_XS, 6 columns | 61 GB/s | 212 GB/s |
| Q4_K/Q5_K | - | 1.3-1.6x faster |

### The 80,000-token prompt (2026-09-30)

| run | result |
|---|---|
| default reserve (1,536 MiB) | 30.95 of 32.6 GB in use; the driver migrated buffers and the run never finished |
| `--vram-reserve-mib 3072 --prefill 4096`, no borrowing | correct; 101 s = 790 tok/s, peak 28.8 GB, decode after it 2 tok/s (~1,900 experts evicted) |
| `--vram-reserve-mib 2048 --prefill 4096`, borrowing | 75 s = **1,062 tok/s**, decode after it 35-40 tok/s |

### The host mirror and KV streaming (2026-09-30)

- **Pinned host mirror, forced test** (`--expert-cache 8000`, 1,879 experts / 3.6 GiB out of VRAM): output tokens
  identical to the full-cache run.

  | | before the mirror | with the mirror | full cache |
  |---|---|---|---|
  | decode | 2.6 tok/s | 40.9 tok/s | 43.3 tok/s |
  | prompt reading | 451 tok/s | 640 tok/s | 776 tok/s |

- **KV streaming** (`--kv-resident`): 256K decoded at 31 tok/s, up from 4-5. Without it, the KV pushed experts out
  and decode after a 128K-256K prompt was 4-9 tok/s.
- **k8v4 at 256K,** after the mirror: 42 tok/s, up from 3.8.

### Upstream merges

| merge | Coder decode, 19 / 2,184-token prompt | IQ2_XS decode | other |
|---|---|---|---|
| 0.1.31 (2026-10-01) | 78.2 / 75.7 tok/s | 58.6 / 64.2 (from 50.8 / 60.6) | upstream's expert kernels (`STRATA_EXPERT_SPLIT=1`): 65.6-69.5 / 67.2 / 49.1, -11 to -16%; `STRATA_GR_V3=1`: 73.2 / 72.4, -4 to -6% |
| 0.1.32 (2026-10-01) | 77.7-78.1 / 75.7 | 58.5 | 40K prompt 1,201 tok/s, then 65.1 decode |
| 0.1.33 (2026-10-01) | 78.55 / 76.31 | 58.88 / 64.31 | 128K int8 894.9 / 67.0 |
| 0.1.35 + 7 PRs (2026-10-02) | - | +1.7% (#363) | 128K int8 888 -> 895 / 66.6 -> 67.0 (#453) |
| 0.1.38 (2026-10-03) | 78.5 / 76.3 | 58.9 | 128K int8 892 / 66.8 |

- **0.1.31:** expert-kernel lanes per row (`STRATA_GU_LANES` / `STRATA_DOWN_LANES`, gate/up 8 or 16 x down 4 or 8)
  all decode in 37.7-38.6 ms per verify round.
- **#374:** credited at first with 784 -> 825 tok/s on the 2,184-token prompt. Three runs each on later builds read
  784-799, so the 825 was session variance.
- **#413, per-key-head DeltaNet:** bit-identical but 8% slower here (727 vs 788 tok/s): opt-in.
- **`STRATA_GR_DOWN_MAX4=1`:** neutral (793 vs 788 tok/s prompt, 76.2 vs 76.3 decode).

### XMX experiments (all kept opt-in or not built)

| experiment | result |
|---|---|
| oneMKL FP16 GEMM (the path in use) | 30-60 TFLOP/s in `xmx_gemm_bench` |
| `xmx_gemm_iq`: fused dequant + FP16 GEMM | 4-5x slower than dequant + oneMKL (0.15-0.19x at 32-256 rows, IQ4_NL down) |
| `qsa_prompt_attn_xmx` v1 | 3x slower than the FP32 fallback per chunk |
| `qsa_prompt_attn_xmx` v2 | 1.4-1.5x faster than v1, still ~2x slower (13.4 vs 5.5 ms per chunk, INT8, 32K) |
| v2 in the full prompt (`STRATA_PROMPT_ATTN_XMX=1`) | 23-33% slower: 128K int8 666 vs 960 tok/s, k8v4 596 vs 983; 256K k8v4 506 vs 752, int8 536 vs 718 |
| expert dots on int8 DPAS for decode (three versions) | 1.4x and 2-3x slower than dp4a; the third hung the GPU |
| int8 DPAS GEMM straight from IQ4_NL (`xmx_int8_bench`, 2026-10-03) | within 0.5% of exact; vs dequant + oneMKL: 0.94-0.98x at 16-32 rows, 0.49-0.73x at 64-128, 0.37-0.55x at 256, 0.25-0.38x at 512 |

- **Grouped prompt attention, never built:** measured on the last chunk of an 80K prompt, 8 consecutive positions
  select 3.3x one position's 2,051 cells (16 positions: 5.1x). Only 12% of a selection is shared by all 8.
  Grouping would cut the gather to ~40% but multiply the arithmetic 3-5x: at best 5-8 s of an 80K prompt.

## Submitting numbers

Results from other cards are welcome. Post them on the PR or in an issue, and include **both your config and every
column of the matrix below**. A table without the config cannot be compared with anything, and a config without
the matrix shows nothing.

**The easy way: benchy.** On an idle card (stop the served model first), run:

    sycl/benchy.sh

Benchy is the standard bench, versioned (now **v1**), so every card runs the same thing.

- **What runs:** every model you set up with `sycl/setup_intel.py`, each with its own serve config, unchanged.
- **Prompts:** the v1 prompts in `sycl/bench/v1`, at 20, 2,185, 8,000, 40,000, 128,000 and 256,000 tokens, each
  followed by 256 greedy tokens. Sizes past a config's context are skipped and listed.
- **Each run:** a fresh engine from a cold page cache. It re-runs itself with sudo for the cache drop and the VRAM
  reading; `--warm` runs without either.
- **Time:** 30-60 minutes on a B70.
- **Output, in `sycl/benchy-results/v1-<date>/`:**
  - `matrix.md`: a speed table in the layout of gresstant's B580 report below, a resources table, the system and
    build, and the caveats;
  - `config-<name>.txt`: each model's engine args;
  - the raw `matrix.jsonl` and one log per run.

  Post `matrix.md` and the `config-*.txt` files.

The prompts, sizes and columns stay fixed within a version; a change to any of them is a new version. Say which
version you ran.

**What to include if you measure another way.**

1. **The config:**
   - the engine's full command line, or the serve config JSON's `args`;
   - any `STRATA_*` / `ONEAPI_*` environment variables you set;
   - the commit you built, plus any local patches (what they change);
   - the AOT target.
2. **The machine:**
   - the card(s) and VRAM, and the PCIe link the card trained at (generation and width);
   - CPU, RAM, and the SSD that holds the models;
   - OS, GPU driver, and the oneAPI / Level Zero versions.
3. **The matrix**, one row per model and prompt size:

   | column | what |
   |---|---|
   | model | Coder IQ1_M, original IQ2_XS, Swift 1.5, ... |
   | prompt | prompt tokens |
   | ctx | `--max-context` |
   | PP tok/s | prompt reading speed |
   | TTFT s | time to first token |
   | decode tok/s | generation speed, and how many tokens were generated |
   | drafts acc. | share of speculative drafts accepted (it moves decode a lot) |
   | experts in VRAM | expert-cache slots and GiB |
   | offloaded to RAM mirror | experts (and GiB) that live in pinned host memory instead of VRAM |
   | lent slots | cache slots the prompt path borrowed |
   | peak VRAM GB | |
   | RAM GB | host memory the engine took (pinned memory included) |
   | SSD load GB / SSD request GB | bytes read from the SSD while loading, and during the request |
   | PLE SSD MB | the per-layer embedding rows read from the SSD for the request |
   | load s | engine start to ready |
   | peak W | board power |

   Say whether the page cache was cold, and whether each run completed (exit code, any device loss).

## Other people's cards (from the upstream PR)

These are copied from [Niko1221/Strata#423](https://github.com/Niko1221/Strata/pull/423) as their posters
reported them, on 2026-10-03. They were not re-measured here.

- **Builds:** each ran its own build, and two of the three are forks with local patches.
- **Missing columns:** they predate "Submitting numbers" above. None has the offloaded experts, RAM, VRAM or SSD
  reads, and mpisat's names no build or flags.
- **Tables:** verbatim, with their own column names. PP is prompt processing (reading) and TG is token generation
  (decode), both in tok/s.

### Arc Pro B50 16 GB, by mpisat

[Comment](https://github.com/Niko1221/Strata/pull/423#issuecomment-5966236499).

- Card: Arc Pro B50 16 GB, 224 GB/s memory bandwidth, 70 W TDP.
- Host: AMD machine with 96 GB DDR5-5200.
- Build: their own fork of this code for the B50, made with the help of Codex. No flags posted.

| Model | Input size | PP | TG | Actual output |
| --- | ---: | ---: | ---: | --- |
| Coder IQ1_M | ~8K | 464.2 | 22.8 | 2,048 |
| Original IQ2_XS | ~8K | 425.0 | 27.2 | 2,048 |
| Coder IQ1_M | 119,990 | 429.4 | 24.1 / 24.2 | 8,192 each; 2/3 completed |
| Original IQ2_XS | 127,998 | 394.3 | **24.6 median** | 2,048 each; 3/3 completed |

### Arc B580, by gresstant

[Comment](https://github.com/Niko1221/Strata/pull/423#issuecomment-5969166342).

- Card: Arc B580, Windows driver 32.0.101.9034.
- Host: Ryzen 7 9700X (8C/16T), ~56 GiB usable system RAM.
- OS: Ubuntu 24.04 under WSL2, 48 GB WSL memory limit, 8 GB swap.
- Toolchain: oneAPI DPC++ 2026.1.1, oneMKL 2026.1, Level Zero 25.18.33578.15.
- Build: commit `4ba35fdbb0941f08184274bef2f3cd05dde37cfe` with local B580 compatibility/tuning patches by GPT 6.1
  Sol, AOT target `bmg-g21`. The tuned settings were not posted.

| Model / configuration | Input tokens | PP (tok/s) | TG (tok/s) | Actual output tokens | Completed requests |
|---|---:|---:|---:|---:|---:|
| Original IQ2_XS, baseline settings (code) | 108 | 30.37 | 21.04 | 256 | 3/3 |
| Original IQ2_XS, baseline settings (long) | 2102 | 88.97 | 22.60 | 256 | 3/3 |
| Original IQ2_XS, baseline settings (chinese) | 112 | 31.58 | 17.75 | 146 | 1/1 |
| Coder IQ1_M, baseline settings (code) | 108 | 31.37 | 15.04 | 256 | 3/3 |
| Coder IQ1_M, baseline settings (long) | 2102 | 104.65 | 14.01 | 256 | 3/3 |
| Coder IQ1_M, tuned settings (code) | 108 | 30.14 | 15.92 | 256 | 3/3 |
| Coder IQ1_M, tuned settings (long) | 2102 | 159.92 | 17.27 | 256 | 3/3 |
| Coder IQ1_M, tuned settings (chinese) | 112 | 32.81 | 13.56 | 107 | 1/1 |

Their caveats, as posted:

- Successful suites are reported, but device loss has also been observed.
- The largest measured input is ~2K, not 8K or 128K.
- Success at the generation cap does not establish long-output reliability.
- These come from a local benchmark, not a matched-harness comparison with other published tables.

### 2x Arc Pro B70 32 GB, by tmking01

[Comment](https://github.com/Niko1221/Strata/pull/423#issuecomment-5972868284).

- Cards: two Arc Pro B70 32 GB.
- Build: this port with their one-line `stage_room()` fix (merged here 2026-10-03), and the image's
  `ONEAPI_DEVICE_SELECTOR` opened to both cards.

| model / configuration | prompt | prompt reading | decode | notes |
|---|---|---|---|---|
| Coder IQ1_M, one card | short arithmetic | - | 74.5 tok/s | - |
| Coder IQ1_M, forced `--layer-split 24` | short arithmetic | - | **70.7 tok/s** | layers 24-47 on CUDA1, expert cache 6,144 slots (12.35 GiB) there; correct answer (408) |
| Flash-Next IQ3_XXS, `--layer-split auto` (split at layer 29) | 1,564 tokens (summarization) | **394.4 tok/s** | **66.0 tok/s** | 100% of experts resident across both cards' VRAM; correct output on a short arithmetic prompt too |

- **Before the fix:** every layer-split configuration crashed in `libur_loader.so` at the second card's
  expert-cache setup. They tried several model sizes, `--expert-cache` auto/explicit/0, and `--mmap-experts`
  against `--stream-experts`.
- **With the image's single-card selector,** they saw an infinite repetition loop at ~30x slower decode.
