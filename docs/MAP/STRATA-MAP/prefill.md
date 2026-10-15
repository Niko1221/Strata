# Strata map: prefill

Source: Strata commit `296b59d2`. Paths are relative to the Strata repo root. Only the prefill-related parts of `src/program/generate.cpp` are covered here.

## 1. Overview

- `src/prefill/` is the **batched prompt path**: the `strata::prefill::Prefill` class (include/strata/prefill/prefill.hpp:47, chunk loop `run_impl` at src/prefill/prefill.cpp:2087) reads a prompt in chunks of T tokens.
- For each chunk it runs every layer over all T rows at once: embeddings, GDN (DeltaNet recurrent) or QSA (attention) halves, then the MoE half.
- Routed experts that are not resident are **streamed host→device through a ring of whole expert blobs**, which is the main throughput lever and the main bottleneck.
- The GEMM, MMQ (llama.cpp's), fused int8 and WMMA kernels for that path live next to it.
- `src/plan/` is NOT an execution-plan or graph system.
- It is only the `strata-plan` CLI that prints the **VRAM memory plan** (KV cache vs expert cache vs state).
- `src/program/generate.cpp` is the driver (`strata generate` / `--serve`).
- It parses `--prefill`, picks the chunk size by a VRAM fit scan, lends expert-cache slots to the prompt path (`borrow`), calls `sp.init` and `sp.run`, and hands the last rows to decode.
- No CUDA/HIP graph capture or replay exists in the prefill path (no `cudaGraph*` or `StreamBeginCapture` in src/prefill/).
- The only `cudaGraph*` names in src/program/generate.cpp are comments at src/program/generate.cpp:1276 and src/program/generate.cpp:9546 about the decode side's verify and batch windows.

## 2. File table

| path | lines | purpose |
|---|---|---|
| src/prefill/prefill.cpp | 4362 | Prefill class: buffer carving, expert stream ring, chunk loop, layer loop, KV append, layer-split stages, CPU share |
| src/prefill/kernels.cu | 2225 | prompt-path kernels (GDN conv/recurrence e.g. gdn_rec_kh_kernel / gdn_rec_cols_pipe_kernel, mrope, router) |
| src/prefill/gemm.cu | 955 | `Gemm` (see include/strata/prefill/gemm.hpp:15): dense f16/bf16 projections via cuBLAS/hipBLAS(Lt), MMQ hooks |
| src/prefill/wmma_gemm.cu / src/prefill/wmma_gemm.h | 737 / 47 | gfx11 (gfx1100 to gfx1151, src/prefill/wmma_gemm.cu:41-42) WMMA FP16/BF16 GEMM Y = beta*Y + W·X |
| src/prefill/moe_fused.cu | 728 | Fused int8 tensor-core experts for the Q2_0 pack, GPU-side grouping |
| src/prefill/moe_fused_iq.cu | 1222 | the same fused int8 kernels for "native" (IQ*) packs |
| src/prefill/moe_mmq.cu | 476 | llama.cpp MMQ routed-expert products |
| src/prefill/mmq_nvfp4_w4a8.cu / src/prefill/mmq_nvfp4_w4a4.cu | 24 / 53 | NVFP4 MMQ: W4A8 (int8 activations, default) / W4A4 (sm_120a FP4, opt-in) |
| src/prefill/mmq_kq_lib.cu / src/prefill/mmq_kq_loader.cpp | 60 / 124 | K-quant MMQ kernels split into a separately loaded shared lib `strata_kq` |
| src/prefill/ggml_cuda_host.cu | 147 | host symbols of ggml-cuda that MMQ needs |
| src/prefill/hipblaslt_tuning.hpp | 135 | hipBLASLt tuning table loader |
| src/prefill/mmq_resident_sort.hpp | 62 | host-only opt-in grouping permutation |
| src/prefill/gdn_rec_parity.cu | 1383 | test: GDN recurrence kernels bit-exact and timed |
| src/prefill/gemm_bf16_parity.cu; src/prefill/gemm_f32_parity.cu | 155 / 129 | tests: Gemm vs cuBLAS |
| src/prefill/mmq_nvfp4_parity.cpp; src/prefill/mmq_expert_bench.cpp | 350 / 128 | test/bench: NVFP4 MMQ vs exact reference; expert MMQ timing |
| src/plan/plan_main.cpp | 73 | `strata-plan` CLI: prints memory plan from VRAM + context (include/strata/plan/plan.hpp, 168 lines) |
| src/program/generate.cpp | 14182 | the driver: options, VRAM planning, chunk choice, prefill→decode, serve loop |
| src/program/batch_groups_test.cpp | 112 | test: pipelined slot groups for --batch |
| src/program/conv_cache_test.cpp | 92 | test: conversation-cache checkpoint retention |
| src/program/draft_kv_plan_test.cpp | 62 | test: which path computes the MTP draft layer's prompt K/V |
| src/program/message_boundary_test.cpp; src/program/split_ring_test.cpp | 33 / 84 | small header tests |
| include/strata/prefill/prefill.hpp | 217 | Prefill public API |

## 3. Where it does X

- **Prefill entry point**: `Prefill::run` src/prefill/prefill.cpp:4338. On Metal it forwards to `metal_engine::prompt_run` (src/prefill/prefill.cpp:4342, apple-metal-backend.md hand-off). Otherwise it calls `run_impl` (src/prefill/prefill.cpp:2087, called at src/prefill/prefill.cpp:4349), then `drain_pipeline` (src/prefill/prefill.cpp:4317-4336, called at src/prefill/prefill.cpp:4352), which waits for the next layer-split stage's future (`next_run_`) and does the same for that stage recursively.
- **Serve call site**: src/program/generate.cpp:11572-11596 `read_part`. It calls `sp.run(ids+a0, b0-a0, a0, e)`. With `--batch` and no pipelined groups it reads the prompt in pieces (otherwise in one run): one chunk each when there is no layer split or a slot is decoding, and on a layer split with no slot decoding pieces of `STRATA_BATCH_YIELD_CHUNKS` chunks (default 8, 0 = one run, src/program/generate.cpp:11573-11581, include/strata/program/batch_read.hpp:13-27), so a BYIELD/BSTOP can preempt between pieces (server.md hand-off). Short fresh prompts (≤ `--short-read`, default 64, src/program/generate.cpp:818-819) go through the decode windows instead of prefill.
- **Chunk loop**: src/prefill/prefill.cpp:2178-2181 `for (c0 = 0; c0 < n; c0 += m.T)`, with `T = min(m.T, n-c0)` and `p0 = pos0+c0`. Each chunk checks `should_stop`, then uploads either embeddings (batched gather, "C-4") or the previous stage's hand-off rows (`hand_in_`, src/prefill/prefill.cpp:2179-2195). It fills per-token QSA step records (src/prefill/prefill.cpp:2300) and runs the layer loop at src/prefill/prefill.cpp:2538, which goes over `[LB, LE)` = the stage's layers.
- **Chunk sizing**:
  - CLI `--prefill CHUNK|auto|auto:N` parsed at src/program/generate.cpp:2041-2062. auto defaults to max 8192 (`prefill_auto_max`, src/program/generate.cpp:541). auto:16384/32768 or `STRATA_PREFILL_AUTO_MAX` allow more. A bare `auto` on gfx12 Linux raises the ceiling to 16384 unless `STRATA_GFX12_CHUNK=0` or `STRATA_GFX12_DEFAULTS=0` (src/program/generate.cpp:2256-2270). It goes back to 8192 on a layer split unless `STRATA_SPLIT_CHUNK16K=1` (src/program/generate.cpp:6616-6627), and when available RAM does not cover the extra slot loan plus 6 GiB (with --resident-experts also every expert the cache does not hold plus the 8192 loan) unless `STRATA_GFX12_CHUNK=1` (src/program/generate.cpp:6628-6656).
  - A bare `auto` on Windows HIP is capped (src/program/generate.cpp:2050-2060).
  - If the own buffers don't fit, the code steps down `kStepChunks = {6144,4096,3072,2048,1536,1024,512}` (src/program/generate.cpp:7615).
  - The step-down uses `Prefill::bytes_needed` (src/prefill/prefill.cpp:1868) plus 512 MiB headroom, because "a chunk that fits to the last MiB left hipBLAS nothing.. gfx1201.. the prompt hung" (src/program/generate.cpp:7618-7620).
  - The 0.1.39 price rule is `160 + chunk*680/1024` MiB (src/program/generate.cpp:3741, src/program/generate.cpp:5094-5101).
- **Buffer layout ("plan construction")**:
  - `Prefill::init` (include/strata/prefill/prefill.hpp:70; called from src/program/generate.cpp:7602, src/program/generate.cpp:7610) and `carve` (src/prefill/prefill.cpp:1242).
  - Carve lays out one region shared by the attention half's and the MoE half's scratch: max(gdn_set_bytes, qsa_set_bytes, moe_set_bytes), at src/prefill/prefill.cpp:979, src/prefill/prefill.cpp:1060, src/prefill/prefill.cpp:1268.
  - `relayout` (src/prefill/prefill.cpp:1343) re-lays the buffers inside borrowed expert-cache slots ("the prompt path is laid out in the tail of it", src/program/generate.cpp:3726).
  - Nothing is captured or replayed.
  - Layouts are reused while chunk and slots are unchanged (src/prefill/prefill.cpp:2144-2146), but the MMQ row table (`mmq::iota`) is rewritten every prompt because the post-prompt refill overwrites the borrowed slots.
- **Expert streaming (throughput core)**:
  - src/prefill/prefill.cpp:115-320.
  - Chunks ≥ `stream_all_min()` (1024, src/prefill/prefill.cpp:164-167) stream every non-resident expert of every layer in a fixed order through `ring_slots(T)` (src/prefill/prefill.cpp:292).
  - The copy engine keeps working during attention.
  - Smaller chunks stage only the routed experts through an 8-slot `STAGE` ring (src/prefill/prefill.cpp:122).
  - By default the streamed ring is 0.1.39's slot count: 1024 (fused) or 384 slots with pinned share ≥ 0.9, else 96 (src/prefill/prefill.cpp:302-313). Only chunks above the size 0.1.39's rule would have picked take the auto chunk scan's byte budget (`set_ring_budget`, src/prefill/prefill.cpp:1861-1864): `ring_bytes()` (1024 or 384 slots × the Q2_0 blob 1,382,400 B, src/prefill/prefill.cpp:262-266) divided by the pack's max blob (`ring_budget_slots`, src/prefill/prefill.cpp:286-291), used only with pinned share ≥ 0.9 while the unpinned arm stays 96 (src/prefill/prefill.cpp:1953-1955); either way capped at 512/1024 (`ring_cap`, src/prefill/prefill.cpp:259).
  - On HIP with `STRATA_HIP_WMMA=1` the ring is 96 slots for chunks of at least `stream_all_min()` tokens (smaller chunks keep STAGE) (src/prefill/prefill.cpp:295-301; the comment quotes measured gains on RDNA4 cards).
  - Stager threads use `STRATA_STAGER_THREADS/SSD/BATCH/RING` (src/prefill/prefill.cpp:1133-1169, src/prefill/prefill.cpp:453).
- **CPU share**: src/prefill/prefill.cpp:169-209, `arm_cpu_share` src/prefill/prefill.cpp:1960. For staged chunks below `cpu_share_max()`, the idle decode CPU pool computes a measured fraction of non-resident experts instead of moving them over PCIe. It is on by default only on CUDA single-GPU without batch slots. **HIP stays off** (comment at src/prefill/prefill.cpp:182).
- **GDN / recurrent state during prefill**:
  - src/prefill/prefill.cpp:2682-2705.
  - Per GDN layer: projections (attn_qkv, attn_gate, ssm_alpha/beta) → `gdn_gates` → `gdn_conv(conv,…)` → `gdn_recurrence(state,…)` over all T tokens → ssm_out.
  - State is updated in place in `ss.gdn_state + (gdn_index-ss.gdn_ord0)*gdn_floats`, with the conv history right after it (`gdn_floats` src/prefill/prefill.cpp:2119).
  - There is no separate prefill copy, so the state carries over to decode and the next chunk.
  - Kernel variants are kh vs cols_pipe (`STRATA_GDN_KEYHEAD`, see src/prefill/gdn_rec_parity.cu:3-5).
- **KV handling**:
  - the QSA half (src/prefill/prefill.cpp:2706-2937) projects K/V for the chunk and appends to the paged cache at position p0 with `kv_append` / `kv_append_q4` (src/prefill/prefill.cpp:2754-2778), using page_table and the int8/q4/K8V4-hybrid formats from `QsaState`.
  - KV streaming (`kv_mode == 1`, host-resident K/V) stages a layer's whole K/V per layer and chunk into a staging pool carved from borrowed expert slots (src/prefill/prefill.cpp:870-911, src/prefill/prefill.cpp:1199; `STRATA_KV_STAGE_OWN` gives it its own allocation).
  - With `STRATA_KV_PREFETCH=1` (default off, src/prefill/prefill.cpp:2101-2102) the next streamed layer's [0,p0) prefix is prefetched on its own stream (src/prefill/prefill.cpp:2308-2324).
  - The MTP drafter's prompt K/V is `draft_kv` (include/strata/prefill/prefill.hpp:55, impl src/prefill/prefill.cpp:1381-1555).
  - It declines when `STRATA_MTP_BATCH=0`, with no scratch region, for host-streamed KV (`kv_mode == 1`), for a ring only when `STRATA_MTP_BATCH_RING=0`, for hybrid KV, and when the drafter is on another device (src/prefill/prefill.cpp:1398-1403).
- **Layer split (multi-GPU pipeline)**: `set_stage` (include/strata/prefill/prefill.hpp:156-163) chains stages. Stage i reads chunk c+1 while stage i+1 reads chunk c, and the last stage calls `on_chunk`. For single-chunk prompts the next stage's idle GPU can, opt-in, compute a share of experts (`STRATA_PREFILL_HELP=1`, default off, src/prefill/prefill.cpp:1557-1558; `set_stage_helper`/`bind_stage_helper` src/prefill/prefill.cpp:1582, src/prefill/prefill.cpp:1624, `STRATA_PREFILL_HELP[_FRAC]`). There is a peer-device path (`set_peer` src/prefill/prefill.cpp:1687, `STRATA_PF_PEER_*`).
- **Prefill → decode hand-off**:
  - `on_chunk(R_rows, T, pos0, err)` (include/strata/prefill/prefill.hpp:138-141, called at src/prefill/prefill.cpp:4223) gives each chunk's final residual rows to src/program/generate.cpp, which builds the MTP draft layer's prompt K/V from them (`draft_kv` or `mtp.prefill`) and prints the PP progress line (src/program/generate.cpp:8508-8540).
  - KV and GDN state are already in `SessionState`.
  - Hand-offs elsewhere: engine-core.md (SessionState, QsaState, ModelGeometry, expert cache, peer), gpu-cpu-kernels.md (strata/kernels: kv_append, qsa, router_top10, mrope, dequant), moe-experts.md (ring/stager/cache borrow, CPU pool), apple-metal-backend.md (prompt_run), intel-sycl-backend.md (include/strata/prefill/xmx_moe.hpp:1), model-spec-ngram-platform.md (WeightTable names such as "ssm_conv1d.weight"), server.md (read_part/yield), build-system.md (STRATA_PREFILL_MMQ / STRATA_USE_HIP macros, strata_kq lib), tests-bench-docs.md (the parity and test files listed in section 2).
- **Memory plan**: src/plan/plan_main.cpp:13-72 → `strata::plan::make_plan(ctx, Geometry{}, Costs, pool)` (include/strata/plan/plan.hpp:129). pool = free VRAM − dense − embd. It throws `DoesNotClose` if the plan doesn't fit. The default free VRAM is 12e9 ("RTX 5070 nominal", src/plan/plan_main.cpp:15).

## 4. Key types and knobs

**Types:**

| Name | Where / notes |
|---|---|
| `Prefill` | include/strata/prefill/prefill.hpp:47 |
| `Prefill::Impl` | src/prefill/prefill.cpp:757, src/prefill/prefill.cpp:873-874 (kv_copy stream, kv_released/kv_ready events) |
| `MmqPlan` | src/prefill/prefill.cpp:1005, `MMQ_GROUP=16` experts per launch src/prefill/prefill.cpp:1000 |
| `MoeBufs` | src/prefill/prefill.cpp:1051 |
| `QsaState` / `SessionState` | include/strata/core/layer.hpp:209, include/strata/core/session.hpp:31 (see engine-core.md) |
| `plan::Geometry/Costs/Plan/DoesNotClose` | include/strata/plan/plan.hpp:23, include/strata/plan/plan.hpp:62, include/strata/plan/plan.hpp:103, include/strata/plan/plan.hpp:118 |
| `Options::prefill_chunk/prefill_auto_max` | src/program/generate.cpp:535, src/program/generate.cpp:541 |

**CLI:**

| Name | Where / notes |
|---|---|
| `--prefill CHUNK\|auto[:N]` | src/program/generate.cpp:2041 (help text src/program/generate.cpp:843) |
| `--short-read N` | `src/program/generate.cpp:818` |
| `--peer-prefill-rows N` | `src/program/generate.cpp:894` |
| `--batch` | src/program/generate.cpp:2075 |
| `--batch-groups` | src/program/generate.cpp:2078 |
| `--prompt-cache(-every)` | src/program/generate.cpp:2106; src/program/generate.cpp:2127 |
| `strata-plan` | src/plan/plan_main.cpp:45-46; flags `--max-context --vram --dense --embd --workspace --state --pool` |

**Env vars** (read in `src/prefill/prefill.cpp` unless noted):

| Name | Where / notes |
|---|---|
| `STRATA_PREFILL_STREAM_AHEAD` | `src/prefill/prefill.cpp:127` |
| `STRATA_TEST_PAGEABLE` | `src/prefill/prefill.cpp:138` |
| `STRATA_STAGER_SLEEP` | `src/prefill/prefill.cpp:147` |
| `STRATA_PREFILL_STREAM_MIN` | `src/prefill/prefill.cpp:165` |
| `STRATA_PREFILL_CPU_SHARE` | `src/prefill/prefill.cpp:186` |
| `STRATA_PREFILL_CPU_SHARE_MAX` | `src/prefill/prefill.cpp:203`; `src/prefill/prefill.cpp:206` |
| `STRATA_EMB_REUSE_ACCOUNT` | `src/prefill/prefill.cpp:279` |
| `STRATA_RING_BYTES` | `src/prefill/prefill.cpp:283` |
| `STRATA_PREFILL_RING` | `src/prefill/prefill.cpp:293` |
| `STRATA_HIP_WMMA` | `src/prefill/prefill.cpp:298` |
| `STRATA_PREFILL_BF16X2` | `src/prefill/prefill.cpp:325` |
| `STRATA_PF_SWITCH_MIN_T` | `src/prefill/prefill.cpp:335` |
| `STRATA_HC_UPMIX` | `src/prefill/prefill.cpp:339` |
| `STRATA_PF_HCDOWN` | `src/prefill/prefill.cpp:349` |
| `STRATA_HCD_EXACT` | `src/prefill/prefill.cpp:357` |
| `STRATA_CVEC_FUSE` | `src/prefill/prefill.cpp:362` |
| `STRATA_PF_PAD/PF_GEMM` | `src/prefill/prefill.cpp:370` |
| `STRATA_GR_UNFUSED` | `src/prefill/prefill.cpp:379` |
| `STRATA_STAGER_RING` | `src/prefill/prefill.cpp:453`; `src/prefill/prefill.cpp:1153` |
| `STRATA_AUX_STAGER` | `src/prefill/prefill.cpp:473` |
| `STRATA_KV_STAGE_OWN` | `src/prefill/prefill.cpp:894` |
| `STRATA_PREFILL_MMQ` | `src/prefill/prefill.cpp:1016` |
| `STRATA_STAGER_THREADS/SSD/BATCH` | `src/prefill/prefill.cpp:1133`; `src/prefill/prefill.cpp:1145`; `src/prefill/prefill.cpp:1157` |
| `STRATA_GROUP_COPY` | `src/prefill/prefill.cpp:1169` |
| `STRATA_MTP_BATCH(_RING/_F16)` | `src/prefill/prefill.cpp:1391`; `src/prefill/prefill.cpp:1396`; `src/prefill/prefill.cpp:1432` |
| `STRATA_PREFILL_HELP(_FRAC)` | `src/prefill/prefill.cpp:1561`; `src/prefill/prefill.cpp:1573` |
| `STRATA_PF_PEER_*` | `src/prefill/prefill.cpp:1692-1788` |
| `STRATA_PREFILL_TIMING` | `src/prefill/prefill.cpp:2012`; `src/prefill/prefill.cpp:2048` |
| `STRATA_KV_PREFETCH` | `src/prefill/prefill.cpp:2101` |
| `STRATA_PLE_BATCH` | `src/prefill/prefill.cpp:2154` |
| `STRATA_TRACE` | `src/prefill/prefill.cpp:2180` |
| `STRATA_PREFILL_ISSUER` | `src/prefill/prefill.cpp:2460` |
| `STRATA_PF_STEP_SYNC` | `src/prefill/prefill.cpp:2527` |
| `STRATA_KV_HOST_DMA` | `src/prefill/prefill.cpp:2748` |
| `STRATA_INDEXER_PER_TOKEN` | `src/prefill/prefill.cpp:2798` |
| `STRATA_SELECT_OLD` | `src/prefill/prefill.cpp:2820` |
| `STRATA_SEL_OVERLAP` | `src/prefill/prefill.cpp:2831` |
| `STRATA_PROMPT_ATTN_OLD` | `src/prefill/prefill.cpp:2921` |
| `STRATA_PREFILL_GROUP_GATHER` | `src/prefill/prefill.cpp:3831` |
| `STRATA_DBG_FEAT` / `STRATA_DBG_ROUTE_OVERRIDE` | `src/prefill/prefill.cpp:2948-2953`; `src/prefill/prefill.cpp:2968` (opt-in research hooks) |
| `STRATA_QSA_DUMP` | src/prefill/prefill.cpp:2899 |
| `STRATA_PREFILL_DUMP_R(_ALL)` | `src/prefill/prefill.cpp:4180`; `src/prefill/prefill.cpp:4193` |
| `STRATA_STATE_HASH_GDN` | `src/prefill/prefill.cpp:4300` |
| `STRATA_DRAFT_TIMING` | `src/prefill/prefill.cpp:1463` |
| `STRATA_HC_UPMIX_CHECK` | `src/prefill/prefill.cpp:2643` |
| `STRATA_IDX_FP16_CHECK` | `src/prefill/prefill.cpp:2853` |
| `STRATA_QSA_DUMP_LAST` | `src/prefill/prefill.cpp:2900-2901` |
| `STRATA_DBG_FEAT_LAYERS` | `src/prefill/prefill.cpp:2958` |
| `STRATA_DBG_FORCE_SLOW_LAYER` | `src/prefill/prefill.cpp:3047` |
| `STRATA_DBG_CPU_GATE` | `src/prefill/prefill.cpp:3231` |
| `STRATA_DUMP_MOE_INPUT` / `STRATA_DUMP_MOE_LAYER` | `src/prefill/prefill.cpp:3363`; `src/prefill/prefill.cpp:3365` |
| `STRATA_DBG_NAN` | `src/prefill/prefill.cpp:4058`; `src/prefill/prefill.cpp:4232` |
| `STRATA_CVEC_FUSE_CHECK` | `src/prefill/prefill.cpp:4103` |

**Also in other files:**

| Name | Where / notes |
|---|---|
| `STRATA_PREFILL_NVFP4` | src/prefill/moe_mmq.cu:190-193; `=w4a8` (default), `=w4a4` or `=fp16`; also read by the test/bench tools at src/prefill/mmq_nvfp4_parity.cpp:337, src/prefill/mmq_expert_bench.cpp:112 |
| `STRATA_GDN_KEYHEAD` | src/prefill/kernels.cu:697 |
| `STRATA_DENSE_F32` | src/prefill/gemm.cu:570 |
| `STRATA_BF16_TC` | src/prefill/gemm.cu:522 |
| `STRATA_MMQ_KQ` | src/prefill/moe_mmq.cu:31; src/prefill/mmq_kq_loader.cpp:110 |
| `STRATA_GDN_CHUNKED` | src/prefill/kernels.cu:1038; src/prefill/kernels.cu:1993; src/prefill/gdn_rec_parity.cu:567 (test copy) |
| `STRATA_GDN_CONV_SERIAL` / `STRATA_GDN_CONVL2` | src/prefill/kernels.cu:1919; src/prefill/kernels.cu:1923 |
| `STRATA_GDN_PP` / `STRATA_GDN_NOY` | src/prefill/kernels.cu:1964; src/prefill/kernels.cu:1971 |
| `STRATA_GDN_REC_HEADS` | src/prefill/kernels.cu:1985; src/prefill/kernels.cu:2013 |
| `STRATA_GDN_PIPELINE` | src/prefill/kernels.cu:1989 |
| `STRATA_GDN_HEAD` | src/prefill/kernels.cu:2012 |
| `STRATA_HIPBLASLT_VERBOSE` | src/prefill/gemm.cu:142; src/prefill/gemm.cu:233; src/prefill/gemm.cu:313 |
| `STRATA_HIPBLASLT_TUNING` | src/prefill/gemm.cu:187 |
| `STRATA_HIP_PROMPT_F16` | src/prefill/gemm.cu:359 |
| `STRATA_WMMA_GEMM` / `STRATA_WMMA_BF16` | src/prefill/gemm.cu:632; src/prefill/gemm.cu:633; src/prefill/gemm.cu:734 |
| `STRATA_PF_GEMM` / `STRATA_PF_SWITCH_MIN_T` / `STRATA_PF_PAD` (also read here) | src/prefill/gemm.cu:735-736; src/prefill/gemm.cu:907-909 |
| `STRATA_DENSE_MMQ` | src/prefill/gemm.cu:818 |
| `STRATA_DBG_NAN` (also read here) | src/prefill/gemm.cu:885 |
| `STRATA_PF_OCC` | src/prefill/moe_fused.cu:589; src/prefill/moe_fused_iq.cu:996 |
| `STRATA_PF_OCC_W` | src/prefill/moe_fused_iq.cu:1009 |
| `STRATA_PF_FUSED` | src/prefill/moe_fused.cu:658; src/prefill/moe_fused.cu:669; src/prefill/moe_fused_iq.cu:1128 |
| `STRATA_PF_FUSED_KQ` / `STRATA_PF_FUSED_TILE` / `STRATA_PF_FUSED_NATIVE` | src/prefill/moe_fused_iq.cu:1071; src/prefill/moe_fused_iq.cu:1107; src/prefill/moe_fused_iq.cu:1120 |
| `STRATA_PF_BK64` | src/prefill/wmma_gemm.cu:693 |
| `STRATA_MMQ_RESIDENT_SORT_NE` | src/prefill/mmq_resident_sort.hpp:17 |

**In src/program/generate.cpp:**

| Name | Where / notes |
|---|---|
| `STRATA_PREFILL_AUTO_MAX` | `src/program/generate.cpp:2045` |
| `STRATA_GFX12_CHUNK` | `src/program/generate.cpp:2262`; `src/program/generate.cpp:6636` |
| `STRATA_GFX12_DEFAULTS` | src/program/generate.cpp:2263 |
| `STRATA_BATCH_YIELD_CHUNKS` | `src/program/generate.cpp:11576` |
| `STRATA_OWNED_PRICE` | src/program/generate.cpp:5100 |
| `STRATA_SPLIT_CHUNK16K` | src/program/generate.cpp:6621 |
| `STRATA_PREFILL_LEND_PCT` | src/program/generate.cpp:3904; src/program/generate.cpp:5634; src/program/generate.cpp:6608 |
| `STRATA_PREFILL_EQUAL` | src/program/generate.cpp:6587 |
| `STRATA_PREFILL_CPU_SHARE` (also read here) | src/program/generate.cpp:7221 |
| `STRATA_PREFILL_HELPER_SOURCE` | src/program/generate.cpp:7254 |

## 5. Gotchas / constants / surprises

- **Model shape is hardcoded** in src/prefill/prefill.cpp:118-119: `N=2560, HC=4, D=N*HC, LR=320, K=10 (top-k), NE=512 experts, C=10240, ZV=6144, HV=48`. The prompt path assumes this one geometry. Any other geometry needs these constants changed.
- "plan/" is a memory planner, not an execution plan. Prefill has no graph capture or replay. Every chunk is a fresh host-driven sequence of launches, with per-layer host syncs on routing (the expert upload follows routing for staged chunks).
- Expert transfer is the bottleneck. Comments quote 87→402 tok/s from ring sizing alone (src/prefill/prefill.cpp:223), and chunks below 1024 tokens stream only the routed experts. The 1024 floor exists because "Below ~1,000 tokens the output changed on Q2_0" (src/prefill/prefill.cpp:160). **Output bits depend on chunk size, ring size and resident/streamed mix**.
- HIP-specific: the CPU share is never on by default on HIP (an explicit STRATA_PREFILL_CPU_SHARE still arms it, src/prefill/prefill.cpp:1960-1971), the ring is 96 slots with STRATA_HIP_WMMA=1 on chunks of at least stream_all_min() tokens (src/prefill/prefill.cpp:295-301), gfx12 on Linux lets a bare `--prefill auto` reach chunks of 16384 (src/program/generate.cpp:2256-2260), and gfx1201 hung when hipBLAS had no headroom (the 512 MiB rule, src/program/generate.cpp:7618-7620). On gfx12 "the native kernels only" run and `fused::enabled()` is false (src/prefill/prefill.cpp:244-246). src/prefill/wmma_gemm.cu targets the gfx11 WMMA parts (gfx1100, gfx1101, gfx1102, gfx1103, gfx1150 and gfx1151; src/prefill/wmma_gemm.cu:41-42), not RDNA4. It does not run on gfx120x (runtime gate src/prefill/wmma_gemm.cu:270-272).
- The NVFP4 prompt experts default to W4A8 (int8 path) on every GPU, Blackwell included; W4A4 is opt-in (src/prefill/moe_mmq.cu:190-193, src/prefill/mmq_nvfp4_w4a8.cu:1-5, src/prefill/mmq_nvfp4_w4a4.cu:1). W4A4 drifts KL 0.0004-0.03 (src/prefill/mmq_nvfp4_w4a8.cu:3-4).
- GDN state is mutated in place across chunks. A failed or cancelled chunk leaves partial state (KvDrain only drains DMA, src/prefill/prefill.cpp:2113-2117). Not restored.
- Hardcoded ring constants: `kSplitHelpRing=48`, `DQ=2` dequant ring, `RING_MAX=1024`, `STAGE=8` (src/prefill/prefill.cpp:123, src/prefill/prefill.cpp:315, src/prefill/prefill.cpp:157, src/prefill/prefill.cpp:122). `strata-plan --state` used to default to 0 and silently overcommitted 117.7 MB (src/plan/plan_main.cpp:47-49).
