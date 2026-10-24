# Community benchmark on NVIDIA GeForce RTX 4070 12 GB, CachyOS Linux, Core i5-12600K, 64 GB DDR5-5800

Measured on 2026-10-18 by [carteakey](https://github.com/carteakey). Strata **v0.1.45** (source build, native SM89 Ada), Flash-Next **IQ3_XXS** (ISTA-DASLab GSQ-RCO), single consumer GPU with 12 GB installed VRAM, 131,072-token context (`128k`).

Evaluates steady-state decode throughput, prompt prefill scaling across DDR5-5600 vs DDR5-5800 memory frequencies, Linux 2 MB Transparent Huge Page (THP) kernel acceleration, and uncompressed Q8_0 Per-Layer Embedding (PLE) SSD streaming. Main limitation: single desktop node, single-GPU 12 GB VRAM capacity requiring host-RAM offload for non-cached experts.

## Hardware and software

- **GPU:** NVIDIA GeForce RTX 4070, 12 GB GDDR6X (AD104, 12,282 MiB addressable VRAM, sm_89, power limit capped at **165 W** via `nvidia-smi` [down from stock 200 W default; min 100 W, max 216 W]). Driver 615.78.08, CUDA 12.8 / 13.0. PCIe link negotiated at Gen4 x16 (24.8 GB/s measured host-to-device throughput).
- **CPU / RAM:** Intel Core i5-12600K (10 cores / 16 threads: 6 Performance cores @ up to 4.9 GHz, 4 Efficient cores @ up to 3.6 GHz, AVX2, no AVX-512; 5 pool workers + 1 host thread). 64 GB installed RAM (4x16 GiB DDR5; currently operating at DDR5-5200 JEDEC/baseline, with scaling comparisons at DDR5-5600 and DDR5-5800).
- **Storage:** WD_BLACK SN770 2 TB NVMe SSD (PCIe 4.0 x4, sequential read ~5,150 MB/s; PLE n-gram table mmap-streamed directly without host RAM caching).
- **OS:** CachyOS Linux (x86_64, kernel `7.2.9-2-cachyos`). CPU scaling governor locked to `performance`, Energy Performance Preference (`epp`) set to `performance`.
- **Engine:** Strata **v0.1.45** source build (native SM89 Ada + planar decode kernels + dynamic expert VRAM caching + native MTP speculative decoding + `madvise(MADV_HUGEPAGE)` kernel THP acceleration).
- **Background:** Dedicated inference node; router service stopped during benchmark execution for 100% full-VRAM and CPU isolation; desktop display idle with baseline VRAM usage under 260 MiB.

## Model and configuration

- **Model:** `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (`IQ3_XXS`):
  - Shard 1 (backbone routed experts + attention weights): `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` (47,039,860,096 bytes, SHA-256 `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15`).
  - Shard 2 (PLE n-gram lookup table): `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` (28,800,138,432 bytes, SHA-256 `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`) for standard `IQ4_NL`, and `Q8_0` uncompressed PLE (54.4 GB, 54,400,264,192 bytes) streamed on-demand directly from NVMe SSD.
- **Pack & Profile:** Pack `Strata-data/packs/iq3_xxs` with expert profile `data/expert-profile.bin` (24,576 ranked pairs).
- **Speculative Decoding:** Native MTP draft head `Strata-data/mtp/rt` (Q2_0 draft head, `--spec 4 --spec-min-p 0.5`, 950 MiB VRAM allocation).
- **Context & KV:** 131,072 tokens (`--max-context 131072`), KV cache format `k8v4`.
- **Expert Cache:** `auto` (allocates 3,086 hot experts across all 48 transformer layers in GPU VRAM, occupying ~5.04-5.34 GiB VRAM; observed hit rate 75-78%).
- **Transparent Huge Pages (THP):** Linux kernel THP enabled (`[always] madvise never`) with `madvise(MADV_HUGEPAGE)` on the pinned 43+ GB host RAM expert arena, collapsing page table overhead by 512x into CPU L2/L3 TLB caches.
- **Prefill:** `auto` (7,680-token prompt chunks).
- **Sampling:** Tested both greedy decoding (`temperature 0.0`) and official Thinking Mode sampling (`temperature 1.0`, `top_p 0.95`, `top_k 20`, `min_p 0.0`).

Server configuration (`strata-iq3_xxs.json`):

```json
{
 "exe": "/home/kchauhan/repos/l3ms/vendor/strata-latest/build/strata",
 "args": [
  "--pack", "/home/kchauhan/repos/l3ms/vendor/Strata-data/packs/iq3_xxs",
  "--native", "/home/kchauhan/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
  "--ple-gguf", "/home/kchauhan/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
  "--expert-profile", "/home/kchauhan/repos/l3ms/vendor/strata-latest/data/expert-profile.bin",
  "--expert-cache", "auto",
  "--prefill", "auto",
  "--spec", "4",
  "--spec-min-p", "0.5",
  "--mtp", "/home/kchauhan/repos/l3ms/vendor/Strata-data/mtp/rt",
  "--max-context", "131072",
  "--kv", "k8v4"
 ],
 "cwd": "/home/kchauhan/repos/l3ms/vendor/strata-latest",
 "tokenizer": "/home/kchauhan/repos/l3ms/vendor/Strata-data/packs/iq3_xxs/tokenizer",
 "model_name": "qwen3.8-flash-next-iq3_xxs",
 "port": 8088,
 "sampling": {
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0.0
 }
}
```

## Method

All runs were performed using the standardized A/B harness (`bench-models/bench-qwen38-refresh-ab.sh`) under verified system telemetry:

1. **Warmup:** Three discarded warmup generations (256 tokens) per arm to populate the dynamic expert VRAM cache, establish memory access patterns, and avoid cold-load transients.
2. **Decode Probes:** Four independent code generation requests (256 output tokens each) with unique prompt prefixes to guarantee no cache reuse. Timings are taken directly from the engine's `predicted_per_second` metrics.
3. **Prefill Probes:** Two distinct fresh-prefix prompt sweeps at 2,048 and 8,192 tokens. Timings are taken from the engine's `prompt_per_second` metrics.
4. **Correctness Check:** A 4,096-token needle-in-a-haystack retrieval verification probe to confirm long-context integrity and draft acceptance stability.
5. **Memory Monitoring:** Process VRAM allocation monitored via NVIDIA NVML / `nvidia-smi` snapshots; host RAM measured via `/proc/meminfo` and `/proc/<pid>/io read_bytes`.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| **DDR5-5200 Live State (2026-10-21)** | 1,759 / 6,796 | 0 | 256 | 4 | 924.3 (911.2–937.3) @ 2k, 1,899.3 (1,896.2–1,902.4) @ 8k | **59.00** (54.6–65.1) | 1.89 (1.85–1.93) @ 2k, 4.38 (4.32–4.45) @ 8k |
| **DDR5-5800 + IQ3_XXS (Thinking mode)** | 1,759 / 6,796 | 0 | 256 | 4 | 947.0 (926.2–956.6) @ 2k, 1,885.2 (1,882.0–1,995.1) @ 8k | **65.80** (65.1–66.5, peak burst 70.0) | 1.88 (1.84–1.90) @ 2k, 4.35 (4.28–4.42) @ 8k |
| **DDR5-5600 + IQ3_XXS Baseline** | 1,759 / 6,796 | 0 | 256 | 4 | 938.3 (925.2–958.4) @ 2k, 1,988.8 (1,972.9–1,999.4) @ 8k | **62.20** (59.6–64.8, peak burst 90.2) | 1.87 (1.83–1.90) @ 2k, 4.12 (4.06–4.18) @ 8k |
| **DDR5-5800 + Full Q8_0 PLE (NVMe)** | 1,759 / 6,796 | 0 | 256 | 4 | 944.5 (937.3–952.1) @ 2k, 1,902.4 (1,896.2–1,910.8) @ 8k | **74.80** (73.8–75.4) | 1.86 (1.82–1.89) @ 2k, 4.32 (4.25–4.38) @ 8k |
| **v0.1.40.2 Historical (DDR5-5600)** | 1,759 / 6,796 | 0 | 256 | 4 | 948.6 (942.3–956.6) @ 2k, 1,998.3 (1,990.0–2,000.1) @ 8k | 56.08 (55.9–56.3) | 1.85 (1.82–1.88) @ 2k, 4.09 (4.05–4.12) @ 8k |

### Key Findings and System Context

- **Memory Frequency Scaling (+5.8% decode):** On a 12 GB GPU where 43-45 expert layers drain from host system memory per token, raising memory clock from DDR5-5600 to DDR5-5800 lifts steady-state decode from 62.20 tok/s to 65.80 tok/s, eliminating dips below 65 tok/s across all code probes.
- **Uncompressed Q8_0 PLE Streaming (+13.7% decode):** Streaming an uncompressed 54.4 GB Q8_0 PLE table from NVMe SSD costs **0 MiB additional VRAM** and consumes only ~0.15-0.65 MB/s of NVMe bandwidth (<0.02% of drive rating). It increases decode throughput to **74.80 tok/s** by eliminating CPU lookup-table dequantization overhead and boosting MTP draft acceptance from 72.1% to 78.2%.
- **VRAM and Host RAM Footprint:**
  - GPU VRAM: 11,692-11,700 MiB allocated out of 12,282 MiB (holding 3,086 hot experts, K8V4 KV cache, MTP draft layer, and verify buffers). 582 MiB free headroom.
  - Host RAM: 47.0 GiB resident for the pinned host expert arena (14 GiB free headroom in 64 GB installed RAM).
  - SSD traffic during steady decode: ~5-10 KB per generated token, matching PLE row retrieval design.

## Correctness and limitations

- **Needle Retrieval:** The 4,096-token needle-in-a-haystack verification check passed with 100% accuracy (1/1) in all tested configurations.
- **Reasoning Quality:** Thinking mode sampling (`temp 1.0 / top-p 0.95 / top-k 20`) generates coherent reasoning traces without repetition loops, incurring a minimal ~4.9% throughput cost compared to greedy argmax decoding.
- **Single Hardware Setup:** Measurements represent a single tuned desktop with dual-channel DDR5-5800 and PCIe 4.0 NVMe. Results on platforms with DDR4 memory or slower PCIe buses will reflect lower bandwidth when offloading non-cached experts to host RAM.
