# Strata map: engine core (`src/core/`, `include/strata/`)

Source: Strata commit `296b59d2`. Paths are relative to the Strata repo root.

## 1. Overview

- src/core is the runtime core of Strata: device facts and a bump-only VRAM arena (src/core/device.cu), a pinned host arena that holds the MoE expert weights and loads them in parallel (src/core/pinned.cu), the dense-weight loader into VRAM in "engine form" (src/core/weights.cpp, reading a converted *pack*, not GGUF directly), layout/shape checks (src/core/layout.cpp, include/strata/core/family_layout.hpp), the per-layer GDN/QSA block composition (src/core/layer.cpp), CUDA/HIP graph capture-replay per layer (src/core/graph.cpp), the one-token host loop across 48 layers with a doorbell protocol to the CPU expert pool (src/core/session.cpp), the speculative VERIFY window (src/core/verify.cpp, 3639 lines) plus the MTP drafter (src/core/mtp.cpp), expert sources/caches/tiers (src/core/expert_source.cpp 5178 lines, src/core/expert_cache.cpp, peer/remote experts), elastic KV via VMM (src/core/vmm.cpp) and conversation save/restore (conversation_*).
- GGUF parsing and model-family detection live header-only in include/strata/artifact/. The top-level generate loop is NOT in core: it is src/program/generate.cpp (see generate-program.md and server.md).
- The engine is hard-shaped for Qwen3.8-Flash-Next (`qwen4exp`): 48 layers = 36 GDN + 12 QSA, 512 experts top-10.

## 2. File table (src/core + include/strata/core + include/strata/artifact)

| path | lines | purpose |
|---|---|---|
| src/core/arch_defaults.cpp | 84 | per-GPU-arch default values of exact switches (gfx1151 etc.) |
| src/core/concurrent_main.cpp | 197 | experiment binary: do CPU expert path and GPU stream path run concurrently |
| src/core/conversation_file.cpp | 1256 | on-disk session file format v1 (host only) |
| src/core/conversation_memory.cpp | 300 | host memory probing / limits for parked conversations |
| src/core/conversation_snapshot.cpp | 300 | per-layer K/V image (ConversationKv): conversation_kv_save/validate/restore/verify/source/part_sizes; no running state |
| src/core/conversation_state.cpp | 407 | running-state checkpoint (GDN recurrence+conv history, PLE hist, QSA indexer tail/dead/block_pos; no K/V; restore patches dead into idx_pooled) + whole SavedConversation snapshot_save/validate/restore (live checkpoint + checkpoints + K/V images; K/V restored before running state). |
| src/core/conversation_checked.hpp / src/core/conversation_copy.hpp | 24 / 90 | private helpers for checked copies |
| src/core/device.cu | 373 | device info, arch checks, mem_get_info, DeviceArena |
| src/core/device_main.cpp | 122 | `strata-device` tool: GPU, plan, arena exercise |
| src/core/expert_cache.cpp | 778 | VRAM-resident expert slot cache + residency profile |
| src/core/expert_source.cpp | 5178 | expert pool adapter: File/Arena sources, prefetch, lookahead, dispatch |
| src/core/graph.cpp | 146 | CapturedGraph + GraphRegistry |
| src/core/layer.cpp | 1509 | one GDN layer composed from kernels (+KV ring env switches) |
| src/core/layout.cpp | 173 | pack tensor shape checks vs ModelGeometry |
| src/core/load_main.cpp | 122 | `strata-load` tool: expert arena load timing |
| src/core/mtp.cpp | 1837 | MTP draft layer on GPU (MtpDrafter) |
| src/core/native_dense.cpp | 541 | NativeDense (dense weight path; gfx12 defaults env) |
| src/core/native_head.cpp | 213 | NativeHead (LM head) + NativeEmbed (token embedding gather) |
| src/core/overlap_main.cpp | 169 | `strata-overlap` tool: expert streaming vs compute overlap |
| src/core/peer_experts.cpp | 332 | second GPU expert tier |
| src/core/pinned.cu | 958 | PinnedArena (hugepages/THP/mlock/shared), parallel expert loaders |
| src/core/remote_expert_opt.cu / src/core/remote_experts.cpp | 197 / 456 | remote (other host/GPU) expert path |
| src/core/session.cpp | 991 | one token through 48 layers; capture/replay; session_loop |
| src/core/verify.cpp | 3639 | Verifier: T-token speculative verify window, batch slots, commit |
| src/core/vmm.cpp | 170 | VmmRange for elastic KV (--kv-grow); stubs when no VMM |
| src/core/weights.cpp | 516 | WeightTable: dense pack loader into VRAM arena |
| src/core/*_test.cpp | various | CPU-only tests (conversation, spec_prob, coupled_draft, arch_defaults, logit_bias, pinned_shared, save_reclaim) |
| include/strata/core/*.hpp | 39 files | headers for the above; key long ones: expert_source 1027, layer 646, verify 533, session 436, conversation_cache 356, mtp 296, family_layout 285 |
| include/strata/artifact/gguf_reader.hpp | 807 | header-only GGUF v3 reader (mmap, no ggml) — generated from src/artifact/gguf_reader.cpp (its header names scripts/split_artifact.py, which is not in the tree) |
| include/strata/artifact/gguf_split.hpp | 52 | resolve shards of split GGUF |
| include/strata/artifact/model_family.hpp | 65 | ModelFamily from general.architecture |
| include/strata/artifact/dequant.hpp | 308 | scalar reference dequantizers transcribed from ggml |

Other include/strata subdirs (kernels/, prefill/, program/, spec/, platform/, plan/, ngram/, qwen35/, hip_compat/) belong to other areas (gpu-cpu-kernels.md, prefill.md, server.md, build-system.md, model-spec-ngram-platform.md, qwen35-engine.md) and are not mapped here.

## 3. Where it does X

- **Model family detection** — include/strata/artifact/model_family.hpp:21 `model_family_from_arch`: "qwen4exp"→Qwen4Exp, "qwen35moe"→Qwen35Moe, else Unknown. include/strata/artifact/model_family.hpp:54 `model_family_runs` returns true ONLY for Qwen4Exp.
- **GGUF parse**:
  - include/strata/artifact/gguf_reader.hpp:351 `class GgufFile` (mmap at include/strata/artifact/gguf_reader.hpp:415, MAP_PRIVATE), include/strata/artifact/gguf_reader.hpp:519 `GgufModel` (multi-shard; `meta()` = shard 0 at include/strata/artifact/gguf_reader.hpp:540, `find` include/strata/artifact/gguf_reader.hpp:542, `in_bounds` include/strata/artifact/gguf_reader.hpp:549).
  - Bounds-checked `Cursor` include/strata/artifact/gguf_reader.hpp:248.
  - Tensor data = base + data_start + offset (include/strata/artifact/gguf_reader.hpp:384).
- **Geometry from file** — include/strata/artifact/gguf_reader.hpp:631 `read_geometry(GgufFile, ModelGeometry&)`; also include/strata/core/family_layout.hpp:31 `geometry_from_gguf` (fills from "<arch>.*" keys, special branch for Qwen35Moe at include/strata/core/family_layout.hpp:63). Layout table check: include/strata/core/family_layout.hpp:240 `check_gguf_layout`; note include/strata/core/family_layout.hpp:237 `layout_table_complete` is true only for Qwen35Moe.
- **Dense weights load**:
  - src/core/weights.cpp:174 `WeightTable::load(pack_dir, arena_base, arena_bytes,..)`; lookups `find` src/core/weights.cpp:511; bf16 host read src/core/weights.cpp:146.
  - Reads a converted PACK directory (manifest + files), converting by `WeightKind` (include/strata/core/weights.hpp:41: Verbatim / Bf16InF32 take high 16 bits / F32 / F16InF32 real convert).
  - Hands quantized planes to gpu-cpu-kernels.md with code_bits/group_elems attributes (include/strata/core/weights.hpp:48 WeightRef).
- **Shape checks** — src/core/layout.cpp:149 `check_all` → src/core/layout.cpp:141 `check_layer` → src/core/layout.cpp:45 `check_one` vs ModelGeometry.
- **VRAM arena** — src/core/device.cu:320 `DeviceArena` ctor (one cudaMalloc per planner region, optional poison), src/core/device.cu:354 `alloc` bump with alignment, no frees (include/strata/core/device.hpp:72). Device facts: src/core/device.cu:249 `device_info`, arch gate src/core/device.cu:217 `gpu_arch_problem`, src/core/device.cu:184 `mem_get_info` wrapper (has a Windows DXGI budget path at src/core/device.cu:136).
- **Expert host arena + load** — src/core/pinned.cu:400 `PinnedArena` ctor (sliced), src/core/pinned.cu:502 ByNeed variant; backing chosen in src/core/pinned.cu:91 `reserve` (hugepages/THP/shared file). Loaders src/core/pinned.cu:655 `load_experts`, src/core/pinned.cu:662 `load_experts_direct`, src/core/pinned.cu:806, src/core/pinned.cu:812 `load_experts_ranges`; integrity via FNV-1a src/core/pinned.cu:332. LoadStats (include/strata/core/pinned.hpp:88) has `ok`/`error`: a short read must refuse the pack.
- **Expert source contract (handoff to moe-experts.md)**:
  - include/strata/core/expert_source.hpp:160 `class ExpertSource` (`blob(layer,e)`, `blob_stable`, `pinned`, `device_alias` for PCIe direct read, `begin_layer`, `prefetch`, `release`).
  - Implementations: include/strata/core/expert_source.hpp:527 `FileExpertSource` (reads model file, prefetch threads), include/strata/core/expert_source.hpp:922 `ArenaExpertSource` (pinned arena).
  - Dispatch struct include/strata/core/expert_source.hpp:339 `ExpertDispatch`, lookahead include/strata/core/expert_source.hpp:247 `RouterLookahead`.
- **VRAM expert cache** — src/core/expert_cache.cpp:469 `ExpertCache::open`, src/core/expert_cache.cpp:561 `open_sized`, src/core/expert_cache.cpp:295 `open_segmented` (CUDA cuMem VMM; not on HIP, see src/core/expert_cache.cpp:4-5), src/core/expert_cache.cpp:670, src/core/expert_cache.cpp:692, src/core/expert_cache.cpp:723 `fill_slot*`, src/core/expert_cache.cpp:382, src/core/expert_cache.cpp:416 `shrink`/`grow` (gives VRAM to elastic KV). Seam with host loop: include/strata/core/hit_hook.hpp:39 `HitFn`.
- **Per-layer graphs** — src/core/graph.cpp:104 `GraphRegistry::record(LayerType, n_tokens, body)`, src/core/graph.cpp:128 `find`, src/core/graph.cpp:133 `launch(…, timeout_ms)`; CapturedGraph begin/end/launch/wait_ms src/core/graph.cpp:47-84.
- **One token decode (main core entry)**:
  - src/core/session.cpp:802 `session_token(tables, g, pos, pos_base, …)`; capture src/core/session.cpp:204 `session_capture`, src/core/session.cpp:850 `session_capture_token`; run src/core/session.cpp:924 `session_run_token`.
  - The heart is src/core/session.cpp:589 `session_loop(g,pos,pos_base,s,gr,PoolFn pool,HitFn hits,…)`: replays one captured graph per layer, the host polls a doorbell (`s.db`) and calls the CPU expert pool (`PoolFn`, include/strata/core/session.hpp:314; implemented in gpu-cpu-kernels.md include/strata/kernels/cpu/pool.hpp) with normed x_f + k miss ids + router weights, then writes `parts` back (k*n_embd*4 bytes).
  - Layers are graphs, the loop is not, so CPU experts can overlap GPU work (include/strata/core/session.hpp:11-14).
  - Init/reset: include/strata/core/session.hpp:108 `session_zero`, include/strata/core/session.hpp:105 `session_release`.
- **Speculative verify window** — include/strata/core/verify.hpp:78 `class Verifier`: include/strata/core/verify.hpp:99 `init`, include/strata/core/verify.hpp:104 `run(T,tokens,pos0,PoolMultiFn,…)`, include/strata/core/verify.hpp:180 `commit(n_keep)`, multi-slot batch include/strata/core/verify.hpp:194, include/strata/core/verify.hpp:197, include/strata/core/verify.hpp:204, pipelined include/strata/core/verify.hpp:266 `pl_launch`/include/strata/core/verify.hpp:277 `pl_finish`/include/strata/core/verify.hpp:280 `pl_commit_async`, logits include/strata/core/verify.hpp:106 `copy_logits`. Used by src/program/generate.cpp (server.md/program area).
- **MTP drafter** — src/core/mtp.cpp:249 `MtpDrafter::load(rt_dir,…)`, src/core/mtp.cpp:659 `bind`, graph captures src/core/mtp.cpp:1247, src/core/mtp.cpp:1265, src/core/mtp.cpp:1315, src/core/mtp.cpp:1343 `kv_restore`. Acceptance math: include/strata/core/spec_prob.hpp (STRATA_SPEC_PROB), include/strata/core/coupled_draft.hpp (STRATA_SPEC_COUPLED).
- **LM head / embedding** — src/core/native_head.cpp:27 `NativeHead::load(shards,n_in,n_out)`, src/core/native_head.cpp:83 `run(mixed→logits)`; src/core/native_head.cpp:120 `NativeEmbed::load`, src/core/native_head.cpp:205 `gather_dev`.
- **Elastic KV** — src/core/vmm.cpp:90 `VmmRange::reserve`, src/core/vmm.cpp:117 `map_one`, src/core/vmm.cpp:143 `unmap` (2 MiB granularity, include/strata/core/vmm.hpp); stubs returning false at src/core/vmm.cpp:157-167 when built without it.
- **Conversation persistence** — include/strata/core/conversation_file.hpp (format v1, fail-closed read), include/strata/core/conversation_cache.hpp (serve parking/reuse: token equality + image identity + steering mode required).
- **Multi-GPU / remote** — include/strata/core/peer_experts.hpp:35 `PeerExperts`; include/strata/core/remote_experts.hpp:20 `RemoteExperts`, include/strata/core/remote_experts.hpp:113 `RemotePrefillSource` (hands to prefill.md).

## 4. Key types and env vars

**Types:**

| Name | Where / notes |
|---|---|
| `ModelGeometry` | include/strata/core/geometry.hpp:23 |
| `WeightTable` | include/strata/core/weights.hpp:110 |
| `WeightRef` | include/strata/core/weights.hpp:48 |
| `DeviceArena` | include/strata/core/device.hpp:72 |
| `DeviceInfo` | include/strata/core/device.hpp:21 |
| `PinnedArena` | include/strata/core/pinned.hpp:32 |
| `LoadStats` | include/strata/core/pinned.hpp:88 |
| `ExpertSource` | include/strata/core/expert_source.hpp:160 |
| `ExpertCache` | include/strata/core/expert_cache.hpp:81 |
| `SessionState` | include/strata/core/session.hpp:31 |
| `SessionGraphs` | include/strata/core/session.hpp:137 |
| `SessionLoopScratch` | include/strata/core/session.hpp:369 |
| `TokenGraph` | include/strata/core/session.hpp:399 |
| `Verifier` | include/strata/core/verify.hpp:78 |
| `GraphRegistry` | include/strata/core/graph.hpp:92 |
| `VmmRange` | include/strata/core/vmm.hpp:23 |
| `GgufFile/GgufModel` | include/strata/artifact/gguf_reader.hpp:351, include/strata/artifact/gguf_reader.hpp:519 |
| `TensorInfo` | include/strata/artifact/gguf_reader.hpp:233 |
| `ModelFamily` | include/strata/artifact/model_family.hpp:15 |

**Env:**

- `STRATA_ARENA_PIN_GIB` include/strata/core/pinned.hpp:27
- `STRATA_ARENA_THP` src/core/pinned.cu:297
- `STRATA_ARENA_LOCK` src/core/pinned.cu:460
- `STRATA_ARENA_MMAP` include/strata/core/expert_source.hpp:981
- `STRATA_FETCH_THREADS` include/strata/core/expert_source.hpp:650
- `STRATA_IO_PREFETCH` include/strata/core/expert_source.hpp:726
- `STRATA_IO_PREFETCH_DEPTH` include/strata/core/expert_source.hpp:263
- `STRATA_IO_STATS` include/strata/core/expert_source.hpp:739
- `STRATA_FS_SLOTS` include/strata/core/foresight_swap.hpp:2 (opt-in swap tier)
- `STRATA_HIP_ARCHS` include/strata/core/device.hpp:33 (compile definition set by cmake/hip_backend.cmake:89, not an env var)
- `STRATA_GFX12_DEFAULTS` src/core/native_dense.cpp:49
- `STRATA_GFX1151_DEFAULTS` include/strata/core/arch_defaults.hpp:10
- `STRATA_HIP_WMMA` include/strata/core/arch_defaults.hpp:7
- `STRATA_KV_RING_OFF` / STRATA_KV_MAIN_OFF src/core/layer.cpp:639
- `STRATA_DBG_NAN` src/core/verify.cpp:2478
- `STRATA_LOGPOS` include/strata/core/verify.hpp:286
- `STRATA_MEMORY_LIMIT_MIB` include/strata/core/conversation_memory.hpp:17
- `STRATA_SPEC_PROB` / STRATA_SPEC_COUPLED (include/strata/core/spec_prob.hpp, include/strata/core/coupled_draft.hpp)
- `STRATA_EMULATE_CC` include/strata/core/emulate.hpp (tests only)

**CLI flags mentioned in headers:**

| Name | Where / notes |
|---|---|
| `--kv-grow` | include/strata/core/vmm.hpp |
| `--serve` | include/strata/core/conversation_cache.hpp |
| `--batch` | include/strata/core/batch_rows.hpp |

Flag parsing itself is outside core (see src/program).

## 5. Gotchas / surprises

- **No safetensors anywhere in this area.** Weights come from GGUF (artifact/) and a converted "pack" dir (src/core/weights.cpp).
- NVFP4 exists only as a GGUF tensor type: include/strata/artifact/gguf_reader.hpp:113 name "NVFP4", include/strata/artifact/gguf_reader.hpp:216 `case 40`: 64 values = 4 UE4M3 sub-block scales + 32 bytes E2M1 codes. No "nvfp4"/"fp4" string in src/core or include/strata/core. CPU NVFP4 kernels exist under gpu-cpu-kernels.md (include/strata/kernels/cpu/nvfp4_avx2.hpp, include/strata/kernels/cpu/nvfp4_avx512.hpp).
- Hardcoded geometry defaults (include/strata/core/geometry.hpp:24-60): n_embd 2560, 48 layers, QSA every 4th (3,7,…,47), vocab 248320, 512 experts top-10, n_ff 640, head_dim 256, n_head 24/kv 2, idx_top_k 2048, hc 4. Kernels are compiled for this shape.
- `model_family_runs` (include/strata/artifact/model_family.hpp:54) only allows qwen4exp; qwen35moe is recognised but refused.
- VMM segmented expert cache is CUDA-only; on HIP, src/core/expert_cache.cpp falls back (src/core/expert_cache.cpp:4-6). Elastic KV is unavailable on HIP: src/core/vmm.cpp:3 excludes STRATA_USE_HIP, and vmm_available() returns false (src/core/vmm.cpp:157).
- DeviceArena never frees (bump only). LoadStats short read = wrong load, must refuse.
- include/strata/artifact/gguf_reader.hpp is GENERATED (per its header comment; scripts/split_artifact.py is not in the tree) and includes <windows.h> on Windows.
- src/core/session.cpp:143 forward-declares `window_reset()` from metal/src/engine/window.cpp (apple-metal-backend.md area coupling inside core).
- Very large env surface (104 distinct `STRATA_*` getenv names in src/core, include/strata/core and include/strata/artifact); some defaults differ per arch via src/core/arch_defaults.cpp.
