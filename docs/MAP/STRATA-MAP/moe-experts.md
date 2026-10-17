# Strata map: MoE experts end to end

Source: Strata commit `296b59d2`. Paths are relative to the Strata repo root.

## 1. Overview

- Strata keeps the dense weights on the GPU and stores the routed experts in tiers: (a) a **VRAM expert cache** (`ExpertCache`, slot arena + `(layer,expert)->slot` residency table).
- (b) A **host tier**, either `ArenaExpertSource` (experts.bin or GGUF loaded into pinned/large-page RAM, `PinnedArena`) or `FileExpertSource` (mmap of experts.bin / GGUF shards, page cache, madvise/pread prefetch).
- (c) Optional helper/peer GPUs (`RemoteExperts`, `PeerExperts`) and the opt-in `ForesightSwap` slots.
- At startup the VRAM tier is filled from a frequency-ranked **profile** (`--expert-profile`, e.g. data/expert-profile.bin), or by **compulsory-miss** (first routed, first admitted, never evicted) if there is no profile.
- During **decode** (verify windows) each layer's routed ids are split into VRAM hits (GPU grouped kernel), a `pcie_frac` share of the misses (GPU reads pinned host blobs over PCIe/DMA), and the rest (CPU `ExpertPool`).
- The same dispatch also bumps a decayed `usage` counter.
- Every `--adapt-every` windows, the **adaptive tier** uses that counter to swap hot missing experts into the slots of cold resident ones in the same layer.
- **Prefill** routes per layer per chunk.
- It uses the routed ids only to decide which non-resident experts to stream into a staging ring for this chunk; for chunks >= 1024 tokens it streams every non-resident expert regardless of routing.
- Prefill does **not** feed the placement counters, and the adaptive tier stands still while a prompt is read.

## 2. Files

| path | lines | purpose |
|---|---|---|
| include/strata/core/expert_cache.hpp | 245 | VRAM tier: slot arena, residency table, admit/replace, profile read/write, elastic shrink/grow |
| src/core/expert_cache.cpp | 778 | its implementation (alloc checks, fill/verify slots, profile I/O at src/core/expert_cache.cpp:72) |
| include/strata/core/expert_source.hpp | 1027 | ExpertSource interface, RouterLookahead, GpuPlanSink, ExpertDispatch, File/ArenaExpertSource |
| src/core/expert_source.cpp | 5178 | mmap/arena sources, resident RAM mode, the decode dispatch (pool + GPU plan + usage counting) |
| include/strata/core/pinned.hpp / src/core/pinned.cu | 195 / 958 | PinnedArena (LargePages/NormalPages/PinnedByCuda), load stats |
| include/strata/core/pinned_ring.hpp | 147 | PinnedRing staging ring (used by prefill) |
| include/strata/core/foresight_swap.hpp | 107 | opt-in per-layer miss slots (STRATA_FS_*) fed by misses + router lookahead |
| include/strata/core/peer_experts.hpp / src/core/peer_experts.cpp | 106 / 332 | second GPU as an adaptive expert tier over P2P |
| include/strata/core/remote_experts.hpp / src/core/remote_experts.cpp | 172 / 456 | CUDA1..3 helper caches (--expert-cache-deviceN) |
| include/strata/core/remote_expert_opt.hpp / src/core/remote_expert_opt.cu | 53 / 197 | optional remote-tier adapt() |
| src/program/generate.cpp | 14182 | CLI, pcie probe + PcieAuto, cache auto sizing, profile fill, adaptive tier (blocking + async), serve loop |
| src/prefill/prefill.cpp | 4362 | prompt path: per-layer routing readback, staging/streaming of non-resident experts, cache-slot loan |
| src/core/verify.cpp | 3639 | verify-window GPU side: PCIe-share grouped launch at src/core/verify.cpp:1655 (comment src/core/verify.cpp:1653), AdaptGate |
| include/strata/kernels/cpu/pool.hpp / src/kernels/cpu/pool.cpp | 303 / 907 | CPU ExpertPool (ExpertJob/ExpertJobMulti, affinity) |
| include/strata/kernels/cpu/expert.hpp; include/strata/kernels/cpu/expert_layout.hpp; include/strata/kernels/cpu/native_expert.hpp; src/kernels/cpu/expert.cpp; src/kernels/cpu/expert_layout.cpp; src/kernels/cpu/native_expert.cpp | 207/101/59/633/495/281 | CPU expert math and blob layout (blob_bytes, max_blob, native formats) |
| src/kernels/cuda/s2_expert_grouped.cu; src/kernels/cuda/native_moe.cu; src/kernels/cuda/router_top10.cu; src/kernels/cuda/native_router.cu; src/kernels/cuda/route_prior.cu | 1238/229/497/329/79 | GPU grouped expert + router kernels (gpu-cpu-kernels.md area) |
| src/prefill/moe_fused.cu; src/prefill/moe_fused_iq.cu; src/prefill/moe_mmq.cu | 728/1222/476 | prompt-path expert GEMMs (prefill.md, gpu-cpu-kernels.md) |
| docs/DISJOINT_EXPERT_CACHE.md | 59 | design note for STRATA_DISJOINT_ADAPT |
| data/expert-profile.bin | n/a (binary) | shipped residency profile (STRP format) |
| data/expert-profile-coder.bin | n/a (binary) | shipped residency profile, Coder (STRP format) |
| data/expert-profile-sycl.bin | n/a (binary) | shipped residency profile, SYCL (STRP format) |

## 3. Behavior by stage

- **Prefill routing and placement:**
  - No placement from prefill.
  - In `Prefill::run_impl` (src/prefill/prefill.cpp:2087), each layer of each chunk runs the router on the GPU. For staged (small) chunks, the ids are downloaded synchronously, once per layer (`m.up_ring.d2h_sync(m.ids_host...)`, src/prefill/prefill.cpp:3133), grouped by expert on the host, and the routed non-resident experts are copied through an 8-slot staging ring (`STAGE`, src/prefill/prefill.cpp:122).
  - For chunks of `stream_all_min()` = 1024 tokens or more (src/prefill/prefill.cpp:164), `stream_all` (src/prefill/prefill.cpp:2341) builds a fixed sequence of **every non-resident expert of every layer** (skipping `host_res>=0` and peer-held experts, src/prefill/prefill.cpp:2395) and streams it through a big ring ahead of the routing.
  - Routing is not consulted for that list.
  - No `usage` update exists in src/prefill/prefill.cpp (the only increment is src/core/expert_source.cpp:3717-3719, in the decode dispatch).
  - The serve loop sets `in_prompt_read` (src/program/generate.cpp:9851, src/program/generate.cpp:11640) so the adaptive tier does not run between a prompt's chunks (src/program/generate.cpp:9993, src/program/generate.cpp:10031).
  - Prefill borrows the top cache slots for its buffers and refills them afterwards (`no_prefill_borrow` doc at src/program/generate.cpp:548).
  - During `--kv-grow` it also uses heat to decide which slots to give up (src/program/generate.cpp:6798-6833).
- **What goes at startup**:
  - src/program/generate.cpp reads the profile (`read_expert_profile`, src/core/expert_cache.cpp:72; called at src/program/generate.cpp:3455, src/program/generate.cpp:3659). "Fill the tier from the profile" (src/program/generate.cpp:5455) copies the top-ranked pairs with `fill_slot_blocking` (src/program/generate.cpp:5493).
  - With no profile, it uses compulsory-miss via `ExpertCache::admit` (include/strata/core/expert_cache.hpp:145, which never evicts).
  - `--expert-cache-per-layer` gives each layer `slots/n_layers` slots (include/strata/core/expert_cache.hpp:154-170) because the global counter gave every slot to about 26 layers of position 0 (measured 2.97% hits).
- **Adaptive placement (decode)**:
  - `ExpertDispatch::usage` (include/strata/core/expert_source.hpp:470) is allocated only when `adapt_every>0 && adapt_swaps>0 && !all_experts_resident` (src/program/generate.cpp:8666).
  - It is incremented +1 per routed (token,k) in `expert_pool_dispatch_multi` (src/core/expert_source.cpp:3717).
  - The blocking `adapt` lambda (src/program/generate.cpp:8833) works per layer.
  - A candidate is a non-resident expert with usage >= 2.0 that no peer/remote/helper holds; a victim is a resident expert.
  - Candidates are sorted descending and victims ascending, and they are paired while `cand >= vict + adapt_min_gain`.
  - The pairs are then sorted by gain and capped at `adapt_swaps` (96).
  - Copies are async; `apply_pending` (src/program/generate.cpp:8806) publishes `host_res[i]=slot`, releases the RAM/page copy, and re-uploads the residency table.
  - After each round, usage *= `adapt_decay` (0.7, src/program/generate.cpp:8987).
  - The async variant is `--adapt-async 1` (src/program/generate.cpp:8990-9300, JobThread src/program/generate.cpp:1410).
  - Swaps are same-layer only (`ExpertCache::replace`, include/strata/core/expert_cache.hpp:148).
- **Decode split (hit / PCIe / CPU)**:
  - `expert_pool_dispatch_multi` (src/core/expert_source.cpp:3680) runs per layer.
  - It takes distinct experts in routing order: resident ones go to VRAM (kind 0), the last `pcie_num/256` of the missed ones go to PCIe (kind 1), and the rest go to the CPU (-1).
  - It publishes `GpuPlanSink` *first* so the GPU starts while the CPU works (comment at src/core/expert_source.cpp:3720).
  - `pcie_mode` 0=DMA into staging via copy engine, 1=kernel reads mapped arena, 2=copy kernel in graph (include/strata/core/expert_source.hpp:333).
  - Results land in `parts` at the router index.
- **Transfer/compute overlap**:
  - Decode: the GPU plan is published before the CPU pool runs, and the hit kernel gets a "poke" (`hit_poke`, on by default; `--no-hit-poke`), because otherwise lazy submission delayed GPU start until the pool returned (17 vs 9,190 of 10,562 layers).
  - Adaptive copies run on a refill stream.
  - `STRATA_ADAPT_OVERLAP=1` (src/program/generate.cpp:1397) lets them land during the next window, gated per layer (AdaptGate, include/strata/core/verify.hpp:46).
  - `STRATA_ADAPT_LAG` (src/program/generate.cpp:1385) and `STRATA_ADAPT_NOWAIT` (src/program/generate.cpp:1375) are related knobs.
  - Prefill: the stream-all ring issues copies `ring` entries ahead, using per-slot `used` events (src/prefill/prefill.cpp:2337-2338), so the copy engine keeps working through attention.
  - The shared expert is deferred to overlap the routed-only uploads (src/prefill/prefill.cpp:3019-3021).
  - Stager threads do host copies for transient (GGUF in-place) blobs (`struct Stager` src/prefill/prefill.cpp:424).
- **CPU expert fallback.** The misses go to `strata::kernels::cpu::ExpertPool` (include/strata/kernels/cpu/pool.hpp:134), one job per distinct missed expert across the window's tokens. In prefill, opt-in/auto `STRATA_PREFILL_CPU_SHARE` (src/prefill/prefill.cpp:169-206) lets the idle decode pool compute rarely-routed non-resident experts of small chunks. This is default `auto` on CUDA single-GPU, off on HIP (src/prefill/prefill.cpp:178-182).
- **mmap / RAM tiers**:
  - `FileExpertSource` mmaps experts.bin (src/core/expert_source.cpp:974) or every file that native_experts.txt names (src/core/expert_source.cpp:1173).
  - It warms pages with madvise WILLNEED / pread I/O threads (src/core/expert_source.cpp:2173-2185) and releases VRAM-resident pages with MADV_DONTNEED (src/core/expert_source.cpp:2847, src/core/expert_source.cpp:3143).
  - `RouterLookahead` (include/strata/core/expert_source.hpp:247) runs layer l+1's BF16 router on layer l's input to warm predicted file-tier pages (`STRATA_IO_PREFETCH_DEPTH` 1-4, src/program/generate.cpp:5828).
  - The resident RAM mode (`--resident-experts`, include/strata/core/expert_source.hpp:552, src/program/generate.cpp:2209) keeps a compact RAM copy that adaptive swaps exchange (`exchange_cache_complement`, include/strata/core/expert_source.hpp:137).
- **`--expert-cache auto` sizing.** src/program/generate.cpp:5085-5200: it takes `device_free_bytes()` (UMA: MemAvailable minus 6 GiB, `STRATA_UMA_HEADROOM_GIB`), subtracts `vram_reserve_mib` (700 by default), the prompt buffers when not borrowing (`160 + chunk*680/1024` MiB unless `STRATA_OWNED_PRICE=exact`), and the pipeline-window MiB. A Windows-HIP floor of 2560 MiB applies on cards with >= 15 GiB. A smaller reserve applies on small cards. `ExpertCache::open` re-checks against the real free VRAM.
- **pcie_frac probe**:
  - `probe_pcie_h2d_gbps` (src/program/generate.cpp:1662) copies 4 × 256 MiB pinned H2D bursts and takes the fastest.
  - `pcie_frac_for_gbps` (src/program/generate.cpp:1725) = base × min(1, gbps/20).
  - Base (src/program/generate.cpp:2861) is 0.25 for NVFP4 packs (`gu_type==40`), 0.55 for native packs, and 0.2 for the canonical Q2_0 pack.
  - `PcieAuto` (src/program/generate.cpp:1736) refines it on CUDA single-GPU serve: f = link/(link + 1.85·pool_rate), rounded to 0.05, clamped to 0.10-0.60, with at most 3 updates (after 48, then 96+96 windows).
- **Hand-offs.** gpu-cpu-kernels.md: grouped hit/PCIe kernels (src/kernels/cuda/s2_expert_grouped.cu, src/kernels/cuda/native_moe.cu, src/kernels/cuda/iq_kernels.cu STRATA_EXPERT_V2) and routers. prefill.md: src/prefill/prefill.cpp streaming/loan/MMQ/fused. engine-core.md: src/core/verify.cpp windows, session_loop, VMM/kv-grow. server.md: the serve loop in src/program/generate.cpp (adapt timing, profile-save, VRAM command). model-spec-ngram-platform.md: expert_layout/native packs/profiles. intel-sycl-backend.md, apple-metal-backend.md: separate backends (`sycl/`, `metal/`). Metal prefill goes through its decode window (src/prefill/prefill.cpp:103-111). build-system.md: STRATA_NATIVE_EXPERTS, STRATA_USE_HIP, STRATA_PREFILL_MMQ/FUSED ifdefs. tests-bench-docs.md: src/kernels/s2_expert_grouped_parity.cpp:1, src/kernels/native_grouped_parity.cpp:1, src/kernels/cpu/expert_parity.cpp:1, src/kernels/route_resident_parity.cu:1, pool tests (src/kernels/cpu/pool_test.cpp:1).

## 4. Types, flags, env

**Types:**

| Name | Where / notes |
|---|---|
| `ExpertCache` | include/strata/core/expert_cache.hpp:81 |
| `kNotResident` | include/strata/core/expert_cache.hpp:38 |
| `ExpertSource` | include/strata/core/expert_source.hpp:160 |
| `FileExpertSource` | include/strata/core/expert_source.hpp:527 |
| `ArenaExpertSource` | include/strata/core/expert_source.hpp:922 |
| `RouterLookahead` | include/strata/core/expert_source.hpp:247 |
| `GpuPlanSink` | include/strata/core/expert_source.hpp:312 |
| `ExpertDispatch` | include/strata/core/expert_source.hpp:339 |
| `ForesightSwap` | include/strata/core/foresight_swap.hpp:36 |
| `PinnedArena` | include/strata/core/pinned.hpp:32 |
| `PinnedRing` | include/strata/core/pinned_ring.hpp:27 |
| `Stager` | src/prefill/prefill.cpp:424 |
| `PcieAuto` | src/program/generate.cpp:1736 |

**CLI (parsed at src/program/generate.cpp:2025-2209; defaults in the Options struct (src/program/generate.cpp:337), e.g. src/program/generate.cpp:525-628):**

| Name | Where / notes |
|---|---|
| `--expert-cache N\|auto` | src/program/generate.cpp:2025, auto=-1 |
| `--expert-cache-device1..3 N\|auto` | src/program/generate.cpp:2029 |
| `--expert-cache-remote-placement stripe\|layer` | src/program/generate.cpp:2036 |
| `--expert-cache-per-layer` | src/program/generate.cpp:2194 |
| `--expert-cache-cpu-order` | src/program/generate.cpp:2193 |
| `--expert-profile` | src/program/generate.cpp:2201 |
| `--expert-profile-save` | src/program/generate.cpp:2202 |
| `--peer-device` | src/program/generate.cpp:2195 |
| `--pcie-frac` | src/program/generate.cpp:2091 |
| `--pcie-mode auto\|dma\|kernel\|direct` | src/program/generate.cpp:2102 |
| `--adapt-every` | default 4, src/program/generate.cpp:598 |
| `--adapt-swaps` | 96, src/program/generate.cpp:628 |
| `--adapt-decay` | default 0.7, src/program/generate.cpp:599 |
| `--adapt-min-gain` | default 1.5, src/program/generate.cpp:602 |
| `--adapt-async` | src/program/generate.cpp:2095 |
| `--vram-reserve-mib` | default 700, src/program/generate.cpp:525 |
| `--vram-elastic` | src/program/generate.cpp:2070 |
| `--resident-experts` | src/program/generate.cpp:2209 |
| `--no-prefill-borrow` | src/program/generate.cpp:2069 |
| `--prefill N\|auto[:max]` | src/program/generate.cpp:2041 |
| `--no-hit-poke` | src/program/generate.cpp:2200 |

**Env:**

| Name | Where / notes |
|---|---|
| `STRATA_PCIE_FRAC_DEFAULT` | src/program/generate.cpp:2863; `=old` |
| `STRATA_PCIE_AUTO_CANON` | src/program/generate.cpp:2867 |
| `STRATA_PCIE_AUTO_SPLIT` | src/program/generate.cpp:1624 |
| `STRATA_ADAPT_OVERLAP / _LAG / _NOWAIT / _SIGNAL` | src/program/generate.cpp:1397 (OVERLAP), src/program/generate.cpp:1385 (LAG), src/program/generate.cpp:1375 (NOWAIT), src/program/generate.cpp:8763 (SIGNAL) |
| `STRATA_HIP_ADAPT_KERNEL_COPY` | src/program/generate.cpp:193; HIP adaptive swap copies with a kernel instead of SDMA |
| `STRATA_DISJOINT_ADAPT` | src/program/generate.cpp:5786 |
| `STRATA_FS_SLOTS/_BUDGET/_ADMIT/_DEPTH/_AHEAD/_VERIFY` | src/program/generate.cpp:5808 |
| `STRATA_IO_PREFETCH(_DEPTH)` | src/program/generate.cpp:4870; _DEPTH src/program/generate.cpp:5828 |
| `STRATA_LOOKAHEAD_STATS` | src/program/generate.cpp:5811 |
| `STRATA_ROUTE_TAIL_SKIP` | src/program/generate.cpp:8624 |
| `STRATA_UMA_HEADROOM_GIB` | src/core/expert_cache.cpp:55 |
| `STRATA_OWNED_PRICE` | src/program/generate.cpp:5100 |
| `STRATA_PREFILL_STREAM_MIN` / `_STREAM_AHEAD` / `_RING` / `_CPU_SHARE(_MAX)` | src/prefill/prefill.cpp:165; src/prefill/prefill.cpp:127; src/prefill/prefill.cpp:293; src/prefill/prefill.cpp:186; src/prefill/prefill.cpp:203 |
| `STRATA_RING_BYTES` | src/prefill/prefill.cpp:283 |
| `STRATA_STAGER_RING/_BATCH/_SLEEP` | src/prefill/prefill.cpp:453; src/prefill/prefill.cpp:1157; src/prefill/prefill.cpp:147 |
| `STRATA_AUX_STAGER` | src/prefill/prefill.cpp:473 |
| `STRATA_TEST_PAGEABLE` | src/prefill/prefill.cpp:138 |
| `STRATA_POOL_TRACE` | src/core/expert_source.cpp:3709 |
| `STRATA_EXPERT_V2(K)` | src/core/arch_defaults.cpp:50 |

## 5. Gotchas

- The cache **never evicts** on admit (by design). Only the adaptive tier moves experts, and only within the same layer. Compulsory-miss with the global counter is known bad (2.97% hits).
- Prefill routing is not counted into `usage`, and the adaptive tier pauses during prompt reads. A long prompt does not reshape VRAM; only decode windows do. The only usage increment is at src/core/expert_source.cpp:3719.
- Chunks of >= 1024 tokens stream **all** non-resident experts of every layer, which is a lot of PCIe traffic per chunk. That is why bigger chunks (`--prefill auto`, up to 8192 by default; 16384 on Linux HIP gfx1200/gfx1201 unless STRATA_GFX12_CHUNK=0 or STRATA_GFX12_DEFAULTS=0) pay off. Below about 1000 tokens, the comment says Q2_0 output changed with chunk size.
- Candidate threshold `u >= 2.0f` is hardcoded (src/program/generate.cpp:8845). Other hardcoded constants: STAGE=8, RING_MAX=1024, K=10, NE=512, N=2560 (src/prefill/prefill.cpp:118, src/prefill/prefill.cpp:122, src/prefill/prefill.cpp:157; Qwen-geometry constants), PcieAuto kPoolFactor 1.85, probe 4×256 MiB, 20 GB/s knee.
- GPU hits round differently from the CPU, so replies can differ slightly with the cache on (log text at src/program/generate.cpp:5441). `--adapt-async` is not bit-exact run to run. ROUTE_TAIL_SKIP changes answers.
- NVFP4 packs get pcie base 0.25 (2.76 MB experts).
- HIP-specific: the adaptive copies through ROCclr staging can block the host (src/program/generate.cpp:169). 2× RX 6900 XT hung with MMQ + adaptive swaps (src/program/generate.cpp:186). `--vram-elastic` is refused on HIP. PcieAuto is CUDA-only by default (the HIP split share is opt-in via STRATA_PCIE_AUTO_SPLIT). The prefill CPU share is off on HIP.
- The profile-fill block is src/program/generate.cpp:5455; the PCIe-share launch is src/core/verify.cpp:1655; the disjoint-adapt design note is docs/DISJOINT_EXPERT_CACHE.md:1.
- `STRATA_DISJOINT_ADAPT` filters candidates in three places, in both server and CLI paths (docs/DISJOINT_EXPERT_CACHE.md:10-11): the serve blocking `adapt` lambda (src/program/generate.cpp:8833, filter src/program/generate.cpp:8845, which also excludes peer- and remote-held experts), the serve `--adapt-async` tier (src/program/generate.cpp:9071-9076, filter src/program/generate.cpp:9105), and the CLI `adapt` lambda (src/program/generate.cpp:13615, filter src/program/generate.cpp:13638). The SYCL copy has its own gate (sycl/src/program/generate.cpp:5190-5197) and filter (sycl/src/program/generate.cpp:7984).
- docs/DISJOINT_EXPERT_CACHE.md:52-53 describes a "standalone ownership test"; no test file in the repo covers `helper_holds`, `STRATA_DISJOINT_ADAPT` or `RemoteExperts::holds`.
