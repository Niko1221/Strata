# V100 Stage 1 — Benchmarks

Phase 8. All numbers: `build-sm70/strata` (branch `feature/v100-moe`),
Swift-1.5 IQ3_XXS, production config (expert-profile 8,000 pairs,
`--expert-cache auto`, `--prefill 2048`, `--spec 4 --spec-min-p 0.5`,
`--mtp mtp/rt`, `--max-context 8192`, fp16 KV), memlock raised for the 40 GiB
arena pin. Machine: 2× Xeon E5-2680 v4 (28 physical / 56 logical per socket,
AVX2, no AVX-512), 125 GB RAM.

## Throughput

| metric | GPU0 (32 GB PCIE) | GPU4 (16 GB SXM2) |
|---|---|---|
| arena load | 39.97 GiB at 7.3–9.6 GiB/s | same (same SSD) |
| expert cache | 8,000 slots / 12.93 GiB | 6,321 slots / 10.23 GiB (auto-fit) |
| prefill, 2,047 tokens | **401.1 tok/s** (5,104 ms; TTFT 5.27 s) | **423.0 tok/s** (4,839 ms; TTFT 5.01 s) |
| prefill, 28 tokens (cold, warmup-heavy) | 20.2 tok/s (TTFT 1.53 s) | 21.7 tok/s (TTFT 1.45 s) |
| decode, 256 tokens sustained | **37.1 tok/s** (6,903 ms) | **32.8 tok/s** (7,796 ms) |
| decode, 32 tokens | 28.4 tok/s | 27.3 tok/s |
| peak VRAM (32-token run) | **18.3 GiB** of 32 GB (18,712 MiB) | **15.6 GiB** of 16 GB (15,949 MiB) |

Observations:

- **Prefill is ~400 tok/s on both cards.** The 16 GB card is marginally faster
  on the 2,047-token prefill: its smaller auto cache (6,321 vs 8,000 slots)
  borrows fewer cache slots for the prompt path, so less VRAM traffic around
  the streamed-expert DMA. 2,047 tokens of prefill at 401 tok/s is the
  practical long-context number (the 28-token "prefill" line includes cold
  graph-capture warmup and is not comparable).
- **Decode is CPU-pool-bound, not GPU-bound.** Sustained decode differs by
  only ~12 % between the 32 GB and 16 GB cards (37.1 vs 32.8 tok/s) while the
  MoE math is the same AVX2 CPU pool in both cases; the GPU difference shows
  up mostly in prefill and TTFT. The 32-token numbers (~28 tok/s) are lower
  than the 256-token sustained numbers because short runs carry the MTP
  drafting ramp and cold-cache overhead per token.
- **Memory footprint.** The engine pins a ~40 GiB expert arena in host RAM
  (cudaHostRegister PORTABLE, 4 KB pages — no hugetlb pool configured), loads
  ~3.2 GiB of weights + 0.8 GiB MTP into VRAM, plus the expert cache
  (12.93 GiB on GPU0 / 10.23 GiB on GPU4) and KV. The 16 GB card fits the
  whole production path with the cache auto-sized to 6,321 slots; the 32 GB
  card takes the full 8,000-slot profile.

## Reference (same box, same quant)

llama.cpp (vanilla build `427291b5b`, CUDA_ARCH 70), CPU-only, 28 threads,
IQ3_XXS: prefill 13.4–15.9 tok/s, decode 3.7–3.9 tok/s. Strata's V100 decode
is ~9–10× faster and its prefill ~25–30× faster on this workload.

## Phase 9 — optimizations (measured before/after, GPU0, 256-token decode)

| variant | decode | notes |
|---|---|---|
| baseline (55 pool workers, 4 KB pages) | 35.3–37.1 tok/s | run-to-run spread |
| `--pool-workers 28` (physical-core count) | **39.9 tok/s** (both runs) | ~9–13 % faster |
| 2 MB hugepages for the arena (`nr_hugepages=20600`) | 12.2 tok/s | kept off; see below |

1. **Pool worker count — adopt 28.** The default pool size is `physical_cores(true)`
   minus the host core; on this box that function returns the 56 *logical* CPUs
   (the Linux affinity branch), so the default runs 55 workers on 28 physical
   cores. With `--pool-workers 28` the AVX2 expert kernels stop contending
   across SMT siblings for execution units and TLB entries: 39.93 tok/s,
   repeat run 39.66 tok/s, vs 35.34–37.08 at the 55-worker default.
   Recommend 28 (physical core count) for this machine.
2. **2 MB hugepages — tried, measured, left off.** The arena code already
   prefers `mmap(MAP_HUGETLB|MAP_HUGE_2MB)` when the kernel pool is configured
   (`src/core/pinned.cu`); it falls back to 4 KB pages and says so. Configuring
   a 40 GiB pool (20,600 × 2 MB, split 10,300 per NUMA node) and rerunning gave
   **12.23 tok/s decode** (vs 35–37) and a slower arena load (1.80 vs 8.18
   GiB/s): with 40 GiB of locked arena + the 72 GB GGUF PLE table in page
   cache, the normal-page pool left only ~7 GB of free memory, so PLE random
   reads and the page cache went under reclaim pressure, and 2 MB faults are
   more expensive at load time. On this machine the 4 KB-page arena wins; the
   startup line reports which backing was used either way.

Net: the recommended production setting on this box is the standard production
config plus `--pool-workers 28`.

## Method

- decode: 256-token greedy generation from the 29-token prompt; the engine's
  own `decode N tokens in X ms` line (MTP on, spec window 4).
- prefill: 2,047-token deterministic prompt, `--max-new 16`; the engine's
  `prefill` line.
- peak VRAM: `nvidia-smi --query-gpu=memory.used` polled every 2 s while the
  32-token run is alive (per-card).
- runs are sequential per card; no other load on GPU0/GPU4 during the suite
  (GPU3 = agent LLM, GPU1 = 27B service, GPU2 = router — untouched).
