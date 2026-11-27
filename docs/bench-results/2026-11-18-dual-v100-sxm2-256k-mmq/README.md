# Dual NVIDIA Tesla V100-SXM2-16GB (`NV6` NVLink) — `v0.2.15` `-DSTRATA_MMQ=ON` + `STRATA_MMQ_TIER_SORT` + NVLink P2P Evaluation (`64Ki` & `256Ki`)

Clean single-variable verification of **Strata `v0.2.15` (`c77572806`)** built with `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_MMQ=ON -DCMAKE_CUDA_ARCHITECTURES=70-real` (self-reporting `engine 0.1.61`, branch `gpuos-tiered-mmq-schedule`) against the prebuilt `v0.2.15` CUDA 12 release (`engine 0.1.60`, `-DSTRATA_MMQ=OFF` / `STRATA_PF_OWN_ONLY`) on a **2 × Tesla V100-SXM2-16GB (`NV6` NVLink)** server at **`64Ki` (`65,536` max context / `63,637` prompt tokens)** and **`256Ki` (`262,144` max context / `257,774` prompt tokens)**.

All runs use identical model files, `DFlash2` drafter, needle-in-a-haystack prompts (`temperature = 0`, `max_tokens = 1024`), engine-default `STRATA_ROUTE_TAIL_SKIP=7` (unset across every run), and continuous 250 ms DCGM NVLink sampling (`dcgmi dmon -e 1011,1012 -d 250`) alongside per-second `nvidia-smi` thermal/clock/ECC guards.

---

## 1. Headline Findings (Isolated Single-Variable Attribution)

1. **`-DSTRATA_MMQ=ON` on a partial-residency V100 rig (`2 × 16 GiB`, `UD-IQ2_XS`):**
   - Even with `STRATA_PF_WMMA=0` (`dp4a` MMQ, `STRATA_MMQ_TIER_SORT=0`), enabling `-DSTRATA_MMQ=ON` lifts `--layer-split auto` prefill from **`1,595.5 -> 2,362.7 tok/s` (`+48.1%`) at `64Ki`** and **`1,565.6 -> 2,182.5 tok/s` (`+39.4%`) at `256Ki`** over prebuilt `0.1.60` (`native_expert_i8`).
   - Adding **`STRATA_PF_WMMA=1` + `STRATA_PF_WMMA2=1`** (Volta FP16 Tensor Core `gemm_iq_f16_grouped` on `47/48` layers; layer `4` `Q5_0` down stays on `dp4a` MMQ) further increases `--layer-split auto` prefill to **`2,490.9 tok/s` at `64Ki`** and **`2,268.0 tok/s` at `256Ki`**.
   - **Scope vs. 100%-resident Volta setups:** Independent `3 × V100-PCIE-32GB` (`UD-IQ4_XS`, 100% Tier-0) measurements measured `-25.9%` prefill (`2,471.9 -> 1,832.2 tok/s`) from `dp4a` MMQ without `STRATA_PF_WMMA=1`. On a starved-VRAM partial-residency rig (`~50%` host-streamed), `-DSTRATA_MMQ=ON` is strongly positive; on 100%-resident `sm_70` i-quants without `STRATA_PF_WMMA=1`, `-DSTRATA_MMQ=OFF` remains faster.
2. **PR `#6023` (`STRATA_MMQ_TIER_SORT=1`) in `--peer-device 1 --peer-reserve-mib 1600`:**
   - Prebuilt `0.1.60` (`-DSTRATA_MMQ=OFF`) cannot run prompt peer prefill (`prefill peer: needs the MMQ prompt path`) and leaves `GPU 1` idle during prompt ingestion (`993.4 tok/s` at `64Ki`, `834.6 tok/s` at `256Ki`, `0.0 GiB` NVLink).
   - Flipping **`STRATA_MMQ_TIER_SORT` `0 -> 1`** enables `wmma_l` (`gemm_iq_f16_grouped`), `order_peer_res` row-count sorting, `mmq::gather_native_group`, and **`FP16 m.mixed_h`** P2P copies on `--peer-device 1`, cutting `GPU 0 -> GPU 1` NVLink input traffic in half (**`28.64 -> 14.38 GiB`** at `64Ki`, **`118.33 -> 58.03–58.23 GiB`** at `256Ki`) and lifting peer prefill **`1,775.3 -> 1,894.8 tok/s` (`+6.7%`, `+90.7%` vs `0.1.60`) at `64Ki`** and **`1,369.1 -> 1,423.1 tok/s` (`+3.9%`, `+70.5%` vs `0.1.60`) at `256Ki`**.
3. **Direct NVLink P2P Stage Handoff (`p2p_stage`, `--layer-split auto`) & Decode Scatter (`hit_out_dev`, `--peer-device 1`):**
   - `STRATA_MMQ_TIER_SORT=1` routes `--layer-split auto` chunk activations directly over `NV6` NVLink (`cudaMemcpyPeerAsync`, DCGM `2.26 GiB` RX / `3.06 GiB` TX at `64Ki`, `9.18 GiB` RX / `10.18 GiB` TX at `256Ki`, `3.34–3.36 GB/s` peak vs `0.0 GiB` host bounce), raising prefill to **`2,512.1 tok/s` at `64Ki` (`+57.4%` vs `0.1.60`)** and **`2,301.2 tok/s` at `256Ki` (`+47.0%` vs `0.1.60`)**.
   - Unconditional P2P decode scatter (`hit_out_dev` when `peer->p2p()` + `direct_mode()`) plus `hand[si]` VRAM state on `--peer-device 1` (`--spec 6 --draft-conf 0.35`) raises decode from **`100.1 -> 106.5–110.5 tok/s`** at `64Ki` and from **`92.7 -> 102.4–102.5 tok/s` (`+10.5–10.6%`) at `256Ki`**. Enforcing `--spec 8 --draft-conf 0.30` via `"user_args"` (`prof-peer1-true-s8-raw`) measured **`106.6 tok/s`** at `64Ki` (`639/1061`, `824` tokens, `stop`), slightly slower than paired `--spec 6 --draft-conf 0.35` (`108.1 tok/s`, `762/1148`), so `--spec 6 --draft-conf 0.35` remains faster on `--peer-device 1`.
4. **Honest measurement of `GPU 0` readiness-tier sorting alone on staged chunks (`T < 1024`):**
   - On isolated `15,725`-token runs forced through `17` staged chunks (`--prefill 960`), flipping `STRATA_MMQ_TIER_SORT` `0 -> 1` measured **`586.3 / 586.4` vs `585.2 / 585.6 tok/s` (`-0.17%`)** in pinned Tier-1 RAM, **`550.0 / 551.2` vs `549.2 / 551.4 tok/s` (`-0.05%`)** with `--mmap-experts` (warm page cache), and **`465.5` vs `467.3 tok/s` (`+0.39%`)** on a natural `2,036`-token prompt. On a `6.2 GB/s` PCIe host link, host-to-device expert DMA dominates `960`-token staged-chunk latency, so `GPU 0` tier-order reordering alone is neutral within run noise.

---

## 2. Verified Hardware, OS & Model Configuration

| Component | Exact Measured Specification |
| --- | --- |
| **GPUs** | 2 × **NVIDIA Tesla V100-SXM2-16GB** (`16,384 MiB` HBM2 each = `32 GiB` total, `sm_70`, ECC Enabled, Persistence Mode Enabled) |
| **Power / Clocks** | Locked `300.00 W` PL, `1,530 MHz` SM / `877 MHz` HBM2 (`nvidia-smi -pm 1 && nvidia-smi -pl 300 && nvidia-smi -ac 877,1530`) |
| **NVLink / PCIe Link** | `NV6` (6 × `25.781 GB/s` links active between `GPU0` and `GPU1`); host PCIe probe `6.2 GB/s` H2D |
| **Topology (`nvidia-smi topo -m`)** | `GPU0 <-> GPU1 = NV6`; both `GPU0` and `GPU1` report `CPU Affinity 0-17,36-53`, `NUMA Affinity 0` (`GPU NUMA ID N/A`) |
| **CPUs** | 2 × **Intel(R) Xeon(R) CPU E5-2695 v4 @ 2.10GHz** (`18` cores / `36` threads per socket = `36C/72T`, `2 × 45 MiB` L3, `performance` governor) |
| **System RAM** | `121.5 GiB` usable (`MemTotal = 130,462,416,896 B`, 2 NUMA nodes: node 0 `64,021 MiB`, node 1 `60,397 MiB`, `numactl --interleave=all`) |
| **OS / Driver / Toolchain** | **Ubuntu 26.04.1 LTS** (`Linux 7.0.0-43-generic` `x86_64`), Driver **`580.196.07`**, **CUDA `12.9`** (`nvcc V12.9.86`), **`g++-14`** (`Release`) |
| **Target Model** | **Qwen3.8-Flash-Next** (`qwen4exp`, GGUF `size_label = 512x56B`), `48` layers, `n_embd = 2560`, `512` experts (`k = 10`, `expert_ff = 640`), `full_attention_interval = 4` |
| **GGUF Shards & Param Count** | Unsloth v31 `UD-IQ2_XS` (`3` shards = `101,432,216,218 B` / `94.47 GiB` + `mmproj` BF16 `907,543,008 B`): **`125.74B`** weight elements (`125,743,653,760` across 1,223 tensors) + **`51.20B`** PLE table elements (`320,001,536 × 160` in shard 3) = `176.94B` GGUF elements |
| **48-Layer Expert Formats (`native_experts.txt`)** | `(IQ2_XXS, IQ4_NL)`: **3** layers · `(IQ2_XXS, Q2_0)`: **5** layers · `(IQ2_S, IQ4_NL)`: **21** layers · `(IQ2_S, Q2_0)`: **18** layers · **`(IQ2_S, Q5_0)`: 1 layer (layer `4`)** — layer `4` stays on `dp4a` MMQ (`47/48` layers take `STRATA_PF_WMMA=1`) |
| **Drafter** | `DFlash2` (`qwen3.8-flash-next-dflash2.q4_0.pack`, `339,694,592 B`, SHA-256 `4df2e5ab2b47c064afbe662e16e4f06569877125c2d0e2123a4ac755266c664c`), `--draft-fallback none --mtp-draft-vocab draft_vocab.bin` |
| **CMake Build Flags (PR `#6023` C++/CUDA tree)** | `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_MMQ=ON -DCMAKE_CUDA_ARCHITECTURES=70-real -DCMAKE_BUILD_TYPE=Release` (SHA-256 `02acd9e55b5f5b02d41afb9bc790e820fd036d1f2cf43ff102ee2151286d670e`) |

**Shared V100 tuning environment (`V100_BASE_ENV`)** held constant across every comparison run unless overridden by the single variable under test:
`STRATA_HC_Q8=1`, `STRATA_SM70_TABLE=1`, `STRATA_FUSE_DECODE=1`, `STRATA_ARGMAX_ROWS_SM70=1`, `STRATA_PIPED_CONCURRENT=1`, `STRATA_B6_MMVQ_ROWS=0`, `STRATA_GDN_REC_WARP=2`, `STRATA_SPLIT_OVERLAP=1`, `STRATA_SELECT_TWO_PASS=1`, `STRATA_GR_NORM_FAST=1`, `STRATA_ADAPT_OVERLAP=1`, `STRATA_ADAPT_ASYNC=1`, `STRATA_NVML=0`, and engine default **`STRATA_ROUTE_TAIL_SKIP=7`** (unset on every run).

---

## 3. `64Ki` Context Verification (`63,637` Prompt Tokens, `max-context 65536`)

Prompt SHA-256: `0ce74e1d42a58d8a90372a6a753205e52283eeb15d751c5851b4f9892bc31dc8` (`temperature = 0`, `max_tokens = 1024`, `--kv int8 --kv-resident 32768 --prefill auto:16384`). All 11 runs returned `3/3` needles (`ALPHA-7284-KITE`, `BRAVO-3916-REED`, `DELTA-8452-MOSS`) with zero throttling (`sw_slow=false`, `hw_slow=false`, `ecc=0`).

| Run | Mode / Flags | Single Variable Changed | Load (s) | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Draft Accept | DCGM NVLink Prefill (`0->1` / `1->0` peak) | DCGM NVLink Decode | Peak RAM (GiB) | Peak GPU0 / GPU1 Temp (°C) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `0.1.60` prebuilt (`-DSTRATA_MMQ=OFF`) | `--layer-split auto --spec 6 --draft-conf 0.35` | Baseline release (`native_expert_i8`) | 34.52 | 40.00 | **1,595.5** | **100.2** (`1024`) | 759/1130 | 0.0 GiB (0.0 / 0.0) | 0.0 GiB (0.0 GB/s) | 53.09 | 49 / 37 |
| `HEAD` (`dp4a` MMQ) | `--layer-split auto --spec 6 --draft-conf 0.35` | `-DSTRATA_MMQ=ON`, `PF_WMMA=0`, `TIER_SORT=0` | 34.51 | 27.04 | **2,362.7** (`+48.1%`) | **97.3** (`1024`) | 746/1150 | 0.0 GiB (0.0 / 0.0) | 0.0 GiB (0.0 GB/s) | 52.94 | 54 / 40 |
| `HEAD` (`WMMA`, `TIER_SORT=0`) | `--layer-split auto --spec 6 --draft-conf 0.35` | `+ STRATA_PF_WMMA=1` + `PF_WMMA2=1` | 34.51 | 25.66 | **2,490.9** (`+56.1%`) | **98.9** (`1024`) | 755/1142 | 0.0 GiB (0.0 / 0.0) | 0.0 GiB (0.0 GB/s) | 53.01 | 56 / 40 |
| `HEAD` (`WMMA`, `TIER_SORT=1`) | `--layer-split auto --spec 6 --draft-conf 0.35` | `+ STRATA_MMQ_TIER_SORT=1` (`p2p_stage`) | 34.51 | 25.45 | **2,512.1** (`+57.4%`) | **99.9** (`1024`) | 755/1142 | 3.06 TX / 2.26 RX GiB (`3.34` RX, `0->1`) | 0.0 GiB (0.0 GB/s) | 53.04 | 56 / 40 |
| `HEAD` minimal env (`TIER_SORT=1`) | `--layer-split auto --spec 6 --draft-conf 0.35` | Only `PF_WMMA=1`, `TIER_SORT=1`, `NVML=0` | 34.51 | 25.85 | **2,472.8** (`+55.0%`) | **98.4** (`1017`) | 753/1124 | 2.26 GiB (`3.35` GB/s pk) | 0.0 GiB (0.0 GB/s) | 52.97 | 56 / 39 |
| `0.1.60` prebuilt (`-DSTRATA_MMQ=OFF`) | `--peer-device 1 --peer-reserve-mib 1600 --spec 6` | Peer decode only (GPU1 idle in prefill) | 28.52 | 64.16 | **993.4** | **100.1** (`930`) | 698/1016 | 0.0 GiB (0.0 / 0.0) | 0.0 GiB (0.0 GB/s) | 51.84 | 60 / 30 |
| `HEAD` (`WMMA`, `TIER_SORT=0`) | `--peer-device 1 --peer-reserve-mib 1600 --spec 6` | Peer prefill (`dp4a` MMQ, FP32 `m.mixed`) + P2P decode | 28.51 | 35.94 | **1,775.3** (`+78.7%`) | **103.3** (`914`) | 661/1072 | 28.64 GiB (`1.62` / `7.83`) | 2.98 GiB (`0.42` GB/s) | 51.46 | 60 / 33 |
| `HEAD` (`WMMA`, `TIER_SORT=1`) | `--peer-device 1 --peer-reserve-mib 1600 --spec 6` | `+ STRATA_MMQ_TIER_SORT=1` (`wmma_l` + FP16 `m.mixed_h`) | 28.51 | 33.68 | **1,894.8** (`+90.7%`) | **106.5** (`932`) | 678/1047 | **14.38 GiB (`0.81` / `7.82`)** | 2.80 GiB (`0.42` GB/s) | 51.53 | 61 / 32 |
| `HEAD` (`WMMA`, `TIER_SORT=0`, repeat 2) | `--peer-device 1 --peer-reserve-mib 1600 --spec 6` | Repeat run (`user_args` unset; `TIER_SORT=0`) | 28.51 | 36.02 | **1,771.1** (`+78.3%`) | **110.1** (`1018`) | 751/1124 | 28.23 GiB (`1.63` / `7.39`) | 3.02 GiB (`0.43` GB/s) | 51.69 | 63 / 33 |
| `HEAD` (`WMMA`, `TIER_SORT=1`, repeat 2) | `--peer-device 1 --peer-reserve-mib 1600 --spec 6` | Repeat run (`user_args` unset; `TIER_SORT=1`) | 28.51 | 33.70 | **1,893.6** (`+90.6%`) | **110.5** (`1024`) | 762/1148 | **14.52 GiB (`0.81` / `7.86`)** | 3.11 GiB (`0.42` GB/s) | 51.45 | 62 / 32 |
| `HEAD` (`WMMA`, `TIER_SORT=1`, true `s8`) | `--peer-device 1 --peer-reserve-mib 1600 --spec 8 --draft-conf 0.30` | `user_args` `--spec 8 --draft-conf 0.30` (`DECODE_TIMING=1`) | 28.52 | 33.58 | **1,900.6** (`+91.3%`) | **106.6** (`824`) | 639/1061 | **14.94 GiB (`0.82` / `8.08`)** | 2.72 GiB (`0.45` GB/s) | 51.67 | 51 / 31 |

*(Parenthesized numbers in `Decode` show generated token count before stop/`max_tokens`. The `repeat 2` runs initially passed `--spec 8 --draft-conf 0.30` only inside `"args"` without `"user_args"`, so `strata_control/drafting.py` launched them with `--spec 6 --draft-conf 0.35`, verified from `engine started:` in every log; when `--spec 8 --draft-conf 0.30` was later enforced via `"user_args"`, it measured `106.6 tok/s` vs `108.1 tok/s` in the paired `--spec 6 --draft-conf 0.35` timing run.)*

---

## 4. `256Ki` Context Verification (`257,774` Prompt Tokens, `max-context 262144`)

Prompt SHA-256: `76f94806537bd695d5f599debd5da99a0e5ed871193b09b39e5c7ab2c787b7df` (`temperature = 0`, `max_tokens = 1024`, `--kv int8 --kv-resident 32768 --prefill auto:16384`). All 8 runs returned `3/3` needles (`ALPHA-7284-KITE`, `BRAVO-3916-REED`, `DELTA-8452-MOSS`) with zero throttling or ECC errors.

| Run | Mode / Flags | Single Variable Changed | Load (s) | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Draft Accept | DCGM NVLink Prefill (`0->1` / `1->0` peak) | DCGM NVLink Decode | Peak RAM (GiB) | Peak GPU0 / GPU1 Temp (°C) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `0.1.60` prebuilt (`-DSTRATA_MMQ=OFF`) | `--layer-split auto --spec 6 --draft-conf 0.35` | Baseline release (`native_expert_i8`) | 36.02 | 164.99 | **1,565.6** | **92.6** (`1024`) | 759/1151 | 0.0 GiB (0.0 / 0.0) | 0.0 GiB (0.0 GB/s) | 55.70 | 64 / 42 |
| `HEAD` (`dp4a` MMQ) | `--layer-split auto --spec 6 --draft-conf 0.35` | `-DSTRATA_MMQ=ON`, `PF_WMMA=0`, `TIER_SORT=0` | 36.01 | 118.43 | **2,182.5** (`+39.4%`) | **93.2** (`1024`) | 761/1154 | 0.0 GiB (0.0 / 0.0) | 0.0 GiB (0.0 GB/s) | 55.65 | 72 / 44 |
| `HEAD` (`WMMA`, `TIER_SORT=0`) | `--layer-split auto --spec 6 --draft-conf 0.35` | `+ STRATA_PF_WMMA=1` + `PF_WMMA2=1` | 36.01 | 113.99 | **2,268.0** (`+44.9%`) | **93.0** (`907`) | 674/986 | 0.0 GiB (0.0 / 0.0) | 0.0 GiB (0.0 GB/s) | 55.67 | 72 / 44 |
| `HEAD` (`WMMA`, `TIER_SORT=1`) | `--layer-split auto --spec 6 --draft-conf 0.35` | `+ STRATA_MMQ_TIER_SORT=1` (`p2p_stage`) | 36.01 | 112.35 | **2,301.2** (`+47.0%`) | **93.7** (`907`) | 674/986 | 10.18 TX / 9.18 RX GiB (`3.35` RX, `0->1`) | 0.0 GiB (0.0 GB/s) | 55.69 | 73 / 45 |
| `0.1.60` prebuilt (`-DSTRATA_MMQ=OFF`) | `--peer-device 1 --peer-reserve-mib 1600 --spec 6` | Peer decode only (GPU1 idle in prefill) | 31.51 | 309.17 | **834.6** | **92.7** (`1024`) | 765/1128 | 0.0 GiB (0.0 / 0.0) | 0.0 GiB (0.0 GB/s) | 54.58 | 67 / 31 |
| `HEAD` (`WMMA`, `TIER_SORT=0`) | `--peer-device 1 --peer-reserve-mib 1600 --spec 6` | Peer prefill (`dp4a` MMQ, FP32 `m.mixed`) + P2P decode | 31.51 | 188.59 | **1,369.1** (`+64.0%`) | **101.4** (`1024`) | 762/1151 | 118.33 GiB (`1.53` / `7.09`) | 3.07 GiB (`0.38` GB/s) | 54.42 | 75 / 34 |
| `HEAD` (`WMMA`, `TIER_SORT=1`) | `--peer-device 1 --peer-reserve-mib 1600 --spec 6` | `+ STRATA_MMQ_TIER_SORT=1` (`wmma_l` + FP16 `m.mixed_h`) | 31.51 | 181.45 | **1,423.1** (`+70.5%`) | **102.5** (`1024`) | 762/1125 | **58.03 GiB (`1.12` / `9.37`)** | 3.09 GiB (`0.38` GB/s) | 54.40 | 75 / 33 |
| `HEAD` (`WMMA`, `TIER_SORT=1`, repeat 2) | `--peer-device 1 --peer-reserve-mib 1600 --spec 6` | Repeat run (`user_args` unset; `strata_control` launched `--spec 6 --draft-conf 0.35`) | 31.51 | 181.36 | **1,423.7** (`+70.6%`) | **102.4** (`1024`) | 763/1133 | **60.18 GiB (`1.12` / `8.64`)** | 3.09 GiB (`0.38` GB/s) | 54.42 | 75 / 32 |

Note on `p2p_stage` (`--layer-split auto` at `256Ki`): `--prefill auto:16384` processes `257,774` tokens in `16` chunks (`15 × 16,384 + 12,014` tokens), transferring `m.R` (`T × 4 × 2,560 × 4 B = 40 KiB/tok`, `640 MiB` per full chunk = **`9.83 GiB`** total theoretical `GPU 0 -> GPU 1` payload; `--peer-device 1` separately picked `14,592`-token chunks); DCGM continuous 250 ms polling measured **`10.18 GiB` TX / `9.18 GiB` RX**.

---

## 5. Staged-Chunk Isolation (`T < stream_all_min()`) — Where `GPU 0` Tier Sorting Executes

On partial-residency chunks `>= 1,024` tokens (`--prefill auto:16384`), `GPU 0` uses `stream_all` (`P.pseq` streaming order) and skips `mmq_resident_sort`. To directly isolate PR `#6023`'s `GPU 0` readiness-tier (`0/1/2`) and row-count sorting on staged chunks (`!stream_all`), interleaved A/B pairs were run on `HEAD` with `STRATA_MMQ_TIER_SORT` flipped `0 <-> 1`:

| Test Condition (`--layer-split auto`, `max-context 65536`) | Prompt Tokens | `TIER_SORT=0` (tok/s) | `TIER_SORT=1` (tok/s) | Delta |
| --- | ---: | ---: | ---: | ---: |
| Tier-1 pinned host arena, `--prefill 960`, `STRATA_PREFILL_CPU_SHARE=0` (`r1 / r2`) | 15,725 (`17` chunks) | 586.3 / 586.4 | 585.2 / 585.6 | **-0.17%** (noise) |
| Tier-2 `--mmap-experts` (warm page cache), `--prefill 960` (`r1 / r2` after warm-up) | 15,725 (`17` chunks) | 550.0 / 551.2 | 549.2 / 551.4 | **-0.05%** (noise) |
| Default pinned arena, `--prefill auto:16384`, default CPU share ON | 2,036 (`1` chunk) | 465.5 | 467.3 | **+0.39%** (noise) |

---

## 6. Reference Comparison vs. `1Cat-vLLM v1.5.3` on `Qwen3.8-Flash-Next` (Published `4 × V100-32GB` Numbers vs. This `2 × V100-16GB` Rig)

`1Cat-vLLM v1.5.3` (`NVFP4 + native MTP4`) requires ~`29.9 / 30.6 GiB` VRAM per GPU across **`4 × V100-SXM2-32GB` (`128 GiB` total VRAM)** (`378,652`-token KV capacity, `262,144` limit) and does not fit `Qwen3.8-Flash-Next` in `2 × 16 GiB` (`32 GiB`) VRAM (their experimental `Flash-Next IQ3_S` profile also targets `4 × 16 GiB = 64 GiB`). On this dual-`16 GiB` machine, `Strata` runs the full `256Ki` context using `32 GiB` VRAM + host RAM:

| Engine & Hardware | Quant & Drafter | `64Ki` Prefill / TTFT | `256Ki` Prefill / TTFT | `64Ki` Single-Req Decode | `250K–256Ki` Single-Req Decode |
| --- | --- | --- | --- | --- | --- |
| **`Strata v0.2.15 + PR #6023`** (`2 × V100-SXM2-16GB`, `32 GiB` VRAM + RAM, `--layer-split auto`) | `UD-IQ2_XS + DFlash2` (`--spec 6`) | **2,512.1 tok/s** (`25.45 s`) | **2,301.2 tok/s** (`112.35 s`) | 99.9 tok/s (`1024`) | 93.7 tok/s (`907`) |
| **`Strata v0.2.15 + PR #6023`** (`2 × V100-SXM2-16GB`, `32 GiB` VRAM + RAM, `--peer-device 1`) | `UD-IQ2_XS + DFlash2` (`--spec 6`) | **1,894.8 tok/s** (`33.68 s`) | **1,423.1 tok/s** (`181.45 s`) | **106.5–110.5 tok/s** | **102.4–102.5 tok/s** |
| **`1Cat-vLLM v1.5.3`** (published `4 × V100-SXM2-32GB`, `128 GiB` VRAM, `TP4`, temp `0`) | `NVFP4 + native MTP4` (FP16 attention KV + host FP8 QSA) | `12.49 s` cold TTFT (~5.2K tok/s) | `61.40 s` cold TTFT at 250K (~4.1K tok/s) | **136 tok/s** measured synthetic (`17.76 ms`/step; 214 at 70% est.) | `19.71 ms`/step (193 at 70% est.; HumanEval `276 tok/s` at 91%) |

---

## 7. Unit Tests & Reproduction Commands

- Header-only scheduler unit tests (`mmq_resident_sort_test.cpp`): compiled on the dual-V100 host (`g++ -std=c++17`) and passed (`mmq_resident_sort_test: ok`).
- Settings register check (`python -m unittest tools.test_settings_register`): 8/8 passed.
- Docs check (`python tools/check --only docs`): passed (6/6).

```sh
# Build with MMQ enabled for Volta (sm_70)
cmake -S . -B build-mmq \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DSTRATA_MMQ=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70-real \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-mmq --target strata -j

# Launch (256Ki context, --layer-split auto for max prefill; add --peer-device 1 --peer-reserve-mib 1600 instead of --layer-split auto for max decode)
STRATA_PF_WMMA=1 STRATA_PF_WMMA2=1 STRATA_MMQ_TIER_SORT=1 \
numactl --interleave=all ./build-mmq/strata --serve \
  --pack /path/to/unsloth-v31-ud-iq2_xs-q8 \
  --native /path/to/Qwen3.8-Flash-Next-UD-IQ2_XS-00001-of-00003.gguf \
  --expert-profile /path/to/expert-profile.bin \
  --expert-cache auto --prefill auto:16384 --max-context 262144 \
  --kv int8 --ple-io mmap --kv-resident 32768 --pool-workers 32 \
  --aux-cpus auto --batch-groups auto --vram-reserve-later-mib 1011 \
  --draft dflash2 --draft-model /path/to/qwen3.8-flash-next-dflash2.q4_0.pack \
  --draft-fallback none --mtp-draft-vocab /path/to/draft_vocab.bin \
  --spec 6 --draft-conf 0.35 --vision --layer-split auto
```

---

## 8. `v0.2.16` (`STRATA_PF_OWN_ONLY`, MMQ-Free) Rebase Verification (`64Ki` & `256Ki`)

After upstream `v0.2.16` (`26ae4fa2` / `58ee322`) removed `-DSTRATA_MMQ`, PR `#6023` was rebased onto `STRATA_PF_OWN_ONLY` with `pf_wmma()` re-enabled for CUDA `sm_70+`, `Prefill::set_peer()` unlocked for both `native_expert_i8` (`48/48` layers, `q8_1` NVLink P2P) and `wmma_l` (`47/48` layers + layer `4` `Q5_0` on `native_expert_i8`) plus `35%` peer PCIe streaming, and async cross-device `cudaEvent_t` handoff for `p2p_stage`.

Built with CUDA `12.9` (`-DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=70-real -DCMAKE_BUILD_TYPE=Release`) on the same `2 × Tesla V100-SXM2-16GB` (`NV6` NVLink) host (`temperature = 0`, `max_tokens = 1024`, `--kv int8 --kv-resident 32768 --prefill auto:16384 --spec 6 --draft-conf 0.35`, `V100_BASE_ENV`). Every run returned `3/3` needles (`ALPHA-7284-KITE`, `BRAVO-3916-REED`, `DELTA-8452-MOSS`) with `0` ECC errors; `STRATA_ADAPT_CHECKRES=1` passed `256/256` windows with `0` mismatches in both `--layer-split auto` and `--peer-device 1`.

### `64Ki` Context (`61,694` Prompt Tokens, `max-context 65536`, `qwen64-bios-benchmark-prompt.json`)

| Build & Mode | Key Flags | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Tokens | Wall (s) |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| `v0.2.16` `main` (`58ee322`), `--layer-split auto` | `V100_BASE_ENV` | 26.617 | **2,327.7** | **100.9** | 1024 (`length`) | 36.729 |
| `v0.2.16` + PR `#6023` + `#6210`, `--layer-split auto` (`WMMA`) | `+ PF_WMMA=1, PF_WMMA2=1, TIER_SORT=1, SIDE_MIXER=2, GRAPH_PRIO=1, FUSE_HITS=1, --pcie-frac 0` | **24.804** | **2,498.2** (`+7.3%`) | **111.1** (`+10.1%`) | 1024 (`length`) | **33.982** |
| `v0.2.16` `main` (`58ee322`), `--peer-device 1` | `V100_BASE_ENV` (GPU 1 idle in prefill) | 39.407 | **1,569.1** | **96.3** | 1024 (`length`) | 50.022 |
| `v0.2.16` + PR `#6023` + `#6210`, `--peer-device 1` (`WMMA`) | `+ PF_WMMA=1, PF_WMMA2=1, TIER_SORT=1, SIDE_MIXER=2, --pcie-frac 0` | **29.490** | **2,098.5** (`+33.7%`) | **117.6** (`+22.1%`) | 1024 (`length`) | **38.180** |
| `v0.2.16` + PR `#6023` + `#6210`, `--peer-device 1` (`int8`) | `+ PF_WMMA=0, TIER_SORT=1, SIDE_MIXER=2` | 31.291 | **1,977.3** (`+26.0%`) | **113.9** (`+18.3%`) | 1024 (`length`) | 40.264 |

### `256Ki` Context (`257,774` Prompt Tokens, `max-context 262144`, `qwen256-updated-benchmark-prompt.json`)

| Build & Mode | Key Flags | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Tokens | Wall (s) |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| `v0.2.16` `main` (`58ee322`), `--layer-split auto` | `V100_BASE_ENV` | 119.587 | **2,161.4** | **97.7** | 914 (`stop`) | 128.901 |
| `v0.2.16` + PR `#6023` + `#6210`, `--layer-split auto` (`WMMA`) | `+ PF_WMMA=1, PF_WMMA2=1, TIER_SORT=1, SIDE_MIXER=2, GRAPH_PRIO=1, FUSE_HITS=1, --pcie-frac 0` | **112.536** | **2,297.1** (`+6.3%`) | **100.4** (`+2.8%`) | 949 (`stop`) | **121.949** |
| `v0.2.16` + PR `#6023` + `#6210`, `--layer-split auto` (`int8`) | `+ PF_WMMA=0, TIER_SORT=1, SIDE_MIXER=2, GRAPH_PRIO=1, FUSE_HITS=1, --pcie-frac 0` | 118.485 | **2,182.1** (`+1.0%`) | **107.1** (`+9.6%`) | 1004 (`stop`) | 127.822 |
| `v0.2.16` `main` (`58ee322`), `--peer-device 1` | `V100_BASE_ENV` (GPU 1 idle in prefill) | 214.966 | **1,200.8** | **86.5** | 914 (`stop`) | 225.510 |
| `v0.2.16` + PR `#6023` + `#6210`, `--peer-device 1` (`WMMA`) | `+ PF_WMMA=1, PF_WMMA2=1, TIER_SORT=1, SIDE_MIXER=2, --pcie-frac 0` | **176.555** | **1,462.4** (`+21.8%`) | **100.8** (`+16.5%`) | 1024 (`length`) | **186.690** |
| `v0.2.16` + PR `#6023` + `#6210`, `--peer-device 1` (`int8`) | `+ PF_WMMA=0, TIER_SORT=1, SIDE_MIXER=2, --pcie-frac 0` | 183.076 | **1,410.4** (`+17.5%`) | **108.7** (`+25.7%`) | 917 (`stop`) | 191.494 |

---

## `v0.2.17` Controlled Clock-Locked Re-verification (`STRATA_ROUTE_TAIL_SKIP=0`)

> Note: the `v0.2.15` / `v0.2.16` tables above are historical runs taken with unlocked boost clocks and `STRATA_ROUTE_TAIL_SKIP=7`. The runs below use `origin/main` (`3142b66b9`, `-DSTRATA_MMQ=OFF`, `sm_70`) + the `peer_stream_ring` fix under `--minimal-env`, **`STRATA_ROUTE_TAIL_SKIP=0`**, and fixed GPU limits (`nvidia-smi -pm 1 -pl 200 -lgc 1020,1020 -ac 877,1020`, `sm_min=sm_max=1020 MHz`, `mem_clk=877 MHz`, `ecc=0`).

### 1. Peer-Streaming Ring Fix Verification (`4,171` Prompt Tokens, `--peer-device 1`, `STRATA_PF_WMMA=0`, `STRATA_PREFILL_CPU_SHARE=0`)

Fixes the `>= 2K` `peer_i8` non-finite logits / silent corruption bug when `STRATA_PF_PEER_STREAM > 0`: `p_release_to` previously advanced `p_issue_until(pk + RP)` mid-group across skipped unrouted experts while earlier experts still held `pgg_slots`, and compact groups lacked a ring-span split (`< RP`). Both `peer_stream_ring_test` and `mmq_resident_sort_test` pass (`ok`).

| Build & Settings | Prefill (tok/s) | Decode (tok/s) | Output `sha256` prefix | Non-finite / Notes |
| --- | ---: | ---: | --- | --- |
| `v0.2.17` `base` (`STRATA_PF_PEER_STREAM=0`) | 1,004.5 | 70.8 | `632fe447bfa0` | 0 non-finite (bit-exact reference) |
| `v0.2.17` `base` (`STRATA_PF_PEER_STREAM=0.35`, `RING=16`) | — | — | `01ba4719c80b` | **FAILED** (`DONE nonfinite` at prompt pos `4164`, decode `4170`) |
| `v0.2.17` `base` (`STRATA_PF_PEER_STREAM=0.35`, `RING=48` default) | 1,189.2 | 63.2 | `1a2b533e73ea` | Silent ring wraparound divergence |
| `v0.2.17` + fix (`STRATA_PF_PEER_STREAM=0.35`, `RING=16`) | 1,006.2 | 71.7 | `632fe447bfa0` | 0 non-finite, **bit-exact** with `PS=0` |
| `v0.2.17` + fix (`STRATA_PF_PEER_STREAM=0.35`, `RING=16`, `STRATA_PEER_NO_P2P=1`) | 863.0 | 70.1 | `632fe447bfa0` | 0 non-finite, **bit-exact** over host bounce |
| `v0.2.17` + fix (`STRATA_PF_PEER_STREAM=0.35`, `RING=48` default) | **1,180.1** (`+17.5%`) | 71.6 | `632fe447bfa0` | 0 non-finite, **bit-exact** with `PS=0` |

### 2. `64Ki` Context (`61,694` Prompt Tokens, `256` Generated Tokens, All `3/3` Needles Passing)

| Mode & Settings | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Wall (s) | Output `sha256` prefix |
| --- | ---: | ---: | ---: | ---: | --- |
| `--peer-device 1`, `PS=0` (`PF_WMMA=0`, `P2P_DECODE_SCATTER=1`) | 43.066 | 1,435.7 | 67.9 | 46.812 | `affdf99307c6` |
| `--peer-device 1`, `PS=0.22` (`PF_WMMA=0`, `P2P_DECODE_SCATTER=1`) | 40.633 | 1,521.9 (`+6.0%`) | 67.9 | 44.384 | `affdf99307c6` *(momentary `0x4` power flag at `190.5W` peak avg; `sm=1020` held)* |
| `--peer-device 1`, `PS=0.28` (`PF_WMMA=0`, `P2P_DECODE_SCATTER=1`) | **40.355** | **1,532.6** (`+6.7%`) | 67.9 | **44.101** | `affdf99307c6` |
| `--peer-device 1`, `PS=0.35` (`PF_WMMA=0`, `P2P_DECODE_SCATTER=1`) | 40.876 | 1,513.0 (`+5.4%`) | 67.9 | 44.622 | `affdf99307c6` |
| `--layer-split auto`, `PF_WMMA=0, TIER_SORT=0, P2P_HANDOFF=0` | 34.111 | 1,814.7 | 65.7 | 37.963 | `4447ffa9a0d9` |
| `--layer-split auto`, `PF_WMMA=1, TIER_SORT=1, P2P_HANDOFF=1` | **31.069** | **1,993.2** (`+9.8%`) | **67.0** | **34.843** | `e517d6a18a04` *(FP16 WMMA activations)* |

### 3. Community Multi-Hardware Context
- **`@noahark` (`3 × V100-PCIE-32GB`, `100%` Tier-0 `UD-IQ4_XS`):** Measured `-25.9%` prefill regression with `-DSTRATA_MMQ=ON` vs `OFF` on `v0.2.15`, and found `STRATA_MMQ_TIER_SORT=1 + STRATA_PF_WMMA=1` neutral (`±3–4%`) on `v0.2.17` when all experts already fit in Tier-0 VRAM, confirming the `+39–57%` prefill gains track Tier-1/2 residency shortfall.
- **`@xrip` (`2 × CMP 50HX`, PCIe Gen2 x16 PHB, no NVLink):** Measured `-0.6%` on `64K` cold reads and `-0.9%` on `8K` decode with `STRATA_P2P_HANDOFF=1` over PCIe PHB with bit-identical outputs.
