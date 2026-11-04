# Community benchmark on 4x NVIDIA RTX 4090 48 GB (sm_89)

Measured on 2026-11-05 by @Moeblack. Tests the stock 0.2.10 release (`strata-dsv41` 0.1.56 `sm_89` build) with the 2.25 bpw `IQ2_XS` DeepSeek-V4.1-Flash set across 256K and 1M context configurations (`--parallel 4`), including multi-turn sessions up to 301K tokens and `DS4_SPLIT=9,11,10,10` to keep all 15,744 experts in VRAM at 1M context.

## Hardware and software

- **GPU and VRAM**: 4 x NVIDIA GeForce RTX 4090 48 GB (`sm_89`, 49,140 MiB each, PCIe, no NVLink, 450 W power limit)
- **CPU and RAM**: 2 x Intel Xeon Gold 6330 @ 2.00 GHz (56 cores / 112 threads across 2 NUMA nodes), 251 GiB DDR4 RAM
- **Storage**: 14 TB enterprise disk (`/data`)
- **OS, driver, and CUDA**: Ubuntu 22.04.5 LTS (Linux 6.8.0-138-generic), NVIDIA driver 595.99.02, CUDA 13.2
- **Strata and engine version**: Strata 0.2.10 (`ffa80d73`), stock prebuilt engine `0.1.56-linux-x86_64-cuda-cuda13` (`06698c674`)
- **Background workloads**: none during single-stream measurements (`nvidia-persistenced` enabled, `vm.swappiness=10`, `LimitMEMLOCK=infinity`)

## Model and configuration

- **Model**: `deepseek-v4.1-flash-ist-das-2.25bpw` (`IQ2_XS` 4-shard abliterated set `AtomicChat/DeepSeek-V4.1-Flash-1017-Abliterated-GSQ-RCO-GGUF`, where shards 1-3 are byte-identical to `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`, plus `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`, Unsloth `DSpark-Q8_0`, and `mmproj-BF16.gguf`)
- **Pack preparation**: packed with `tools/dsv41/pack_gguf.py` and the shipped `data/dsv41/profile-2.25.bin` hot-expert profile
- **Context and concurrency**: tested at `262144` (256K) and `1048576` (1M) max context with `--parallel 4`
- **Drafter and vision**: `--dspark auto --dspark-conf 0.65 --vision on` (`--vision-resident` enabled in Arm 3)

```text
strata-dsv41 --serve \
  --pack <PACK_DIR>/deepseek-v4.1-flash-ist-das-2.25bpw \
  --max-context <262144|1048576> \
  --eos 1 \
  --vision on \
  --dspark auto \
  --dspark-conf 0.65 \
  --tp 2 \
  --reserve-mib 900 \
  --layer-split auto \
  --parallel 4
```

On the stock 0.1.56 engine, `--tp 2` logs `tensor parallel 2 -> 1: the experts' 256-value blocks split n_ff 2304 no other way (each GPU its own layers)` and runs contiguous layer-split across the 4 GPUs.

## Method

- All prompt and decode speeds are taken directly from the engine's timing lines (`maya ds4 serve: request N: ...`) and HTTP response `timings`.
- Three stock configurations were measured:
  - **Arm 1 (`256K default`)**: `--max-context 262144`, automatic layer split (`5, 12, 11, 12`).
  - **Arm 2 (`1M default`)**: `--max-context 1048576`, automatic layer split (`5, 12, 11, 12`), including multi-turn conversations reaching 286K-301K tokens.
  - **Arm 3 (`1M + DS4_SPLIT=9,11,10,10`)**: `--max-context 1048576 --vision-resident` with `DS4_SPLIT=9,11,10,10` so GPU0 holds 9 trunk layers (`[0..8]`) plus the MTP drafter (`[40..42]`), balancing VRAM across all 4 cards.
-Startup logs and timing lines are recorded in [`engine-log-excerpt.txt`](engine-log-excerpt.txt) and [`results.json`](results.json).

## Results

### 1. VRAM placement and expert residency

| Configuration | Context | Trunk layer split (GPU0..3) | Resident VRAM experts (GPU0 / GPU1 / GPU2 / GPU3) | Free VRAM (MiB) | CPU-lane spill |
| --- | ---: | --- | --- | --- | --- |
| **Arm 1 (default)** | 262,144 | `5, 12, 11, 12` | `2304/2304`, `4608/4608`, `4224/4224`, `4542/4608` | `20015 / 963 / 2239 / 893` | **66 experts (0.6 GB)** |
| **Arm 2 (default)** | 1,048,576 | `5, 12, 11, 12` | `2304/2304`, `4419/4608`, `4180/4224`, `4404/4608` | `19301 / 889 / 885 / 883` | **437 experts (4.2 GB)** |
| **Arm 3 (`DS4_SPLIT=9,11,10,10`)** | 1,048,576 | `9, 11, 10, 10` | `3840/3840`, `4224/4224`, `3840/3840`, `3840/3840` | `3969 / 2987 / 5143 / 6365` | **0 experts (0.0 GB)** |

### 2. Prompt and decode throughput

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s | Decode tok/s | TTFT (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Arm 1 (256K)** · short prompt decode | 25 | 0 | 600 | 1 | 181.3 | **131.76** | 0.14 |
| **Arm 1 (256K)** · 5.7K cold prompt | 5,752 | 0 | 63 | 1 | **1,150.5** | 84.70 | 5.00 |
| **Arm 1 (256K)** · 9K cold prompt | 9,010 | 0 | 12 | 1 | **1,809.7** | 89.64 | 4.98 |
| **Arm 1 (256K)** · 4-way concurrent slots | 9–15 | 0 | 130–400 | 4 | 113.3–166.8 | **21.90–52.03** (125.2 total) | 0.07–0.09 |
| **Arm 2 (1M default)** · short prompt decode | 25 | 0 | 600 | 1 | 179.8 | **121.91** | 0.14 |
| **Arm 2 (1M default)** · 286K–300K multi-turn | 286,329–300,114 | 284,628–299,546 | 62–398 | 4 | **484.1–843.1** (fresh 568–6,563 tok) | **125.14–145.29** | 1.17–7.92 |
| **Arm 3 (1M, `DS4_SPLIT=9,11,10,10`)** · Python LRU code | 40 | 0 | 768 | 1 | 192.1 | **122.80** | 0.213 |

## Correctness and limitations

- **Correctness**: Verified deterministic arithmetic (`123 * 456` -> `56088`) and 768-token Python LRU cache implementation (`temperature = 0`, thinking disabled), as well as coherent multi-turn responses at 286K–301K context depth.
- **Layer split behavior on 4x 48 GB cards**: Because GPU0 holds the MTP drafter, the default splitter assigns 5 trunk layers (`[0..4]`) to GPU0 and 35 trunk layers (`[5..39]`) to GPU1–3 before subtracting the 1M KV cache (`--parallel 4`). At 1M context this leaves 19.3 GiB free on GPU0 while spilling 437 experts (4.2 GB) from GPU1–3 to the CPU lane. Setting `DS4_SPLIT=9,11,10,10` moves 4 trunk layers to GPU0 and keeps 100% of all 15,744 experts in VRAM with 3–6 GiB free per GPU.
