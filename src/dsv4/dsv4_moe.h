// src/dsv4/dsv4_moe.h - the S2 MoE half-layer of the DeepSeek V4-Flash engine path.
//
// One MoE half-layer, exactly the dsv4ref.cpp host contract (the golden oracle's semantics, MANIFEST's 22
// traps), on the GPU:
//   hc_pre (ffn side: 24 mixes over the flattened 4*n_embd, sigmoid gates, comb -> sinkhorn(20) in the
//   reference's exact iteration order) -> rmsnorm(ffn_norm) -> sqrt(softplus(x @ gate_w.T)) routing
//   (hash layers 0-2: tid2eid selects, weights STILL gathered from the gate scores - trap 14; routed
//   layers: top-6 on score + bias, bias enters the selection only - trap 13; weights = score / sum *
//   route_scale with the double sum of dsv4ref.cpp:1170-1176) -> NVFP4 expert FFN with the V4 swiglu
//   clamp variant (E2: gate = silu(min(g, 10)), up = clamp(u, +-10), FP32 router weight applied to the
//   hidden BEFORE the down projection - trap 15) -> shared expert unweighted (trap 22) -> hc_post.
//
// Everything is new code in this translation unit: the shared libraries carry Qwen's swiglu/clamp
// conventions (the opposite clamp placement), so NO shared kernel file is touched - the NVFP4 grouped
// GEMV (E3) and the V4 clamp variant (E2) are implemented here from the bit-level contract of
// dsv4ref.cpp:94-135 (dequant) and :1098-1115 (expert_apply).
//
// The expert bytes come from the pack's experts.bin through FileExpertSource (host mmap); each needed
// blob is staged to the device once per layer.  STRATA_DSV4_MOE_CPU=1 runs the expert FFN stage on the
// host instead (the CPU-pool path of this slice: same dequant + FP32 arithmetic as dsv4ref, used as the
// GPU kernels' cross-check and as the fallback when a session pins experts to the CPU).

#pragma once

#include <cstdint>
#include <string>

#include <cuda_runtime.h>

namespace strata::core {
class FileExpertSource;
class ExpertCache;
}

namespace strata::dsv4 {

struct Dsv4Geometry;

/// The per-layer weights the MoE half consumes, all device pointers gathered by the engine (which owns
/// the Model).  Null where the layer type says the tensor does not exist.
struct Dsv4MoeLayer {
    const float* ffn_norm = nullptr;    ///< blk.L.ffn_norm.weight, [n_embd]
    const uint16_t* gate_w = nullptr;   ///< blk.L.ffn_gate_inp.weight, [n_expert][n_embd] RAW BF16
                                        ///< (pack kind 4 = raw16 copy; convert in the kernel, never
                                        ///< dereference as f32 - the arena stores 2 B/elem here)
    const float* gate_b = nullptr;      ///< blk.L.ffn_exp_probs_b.bias, [n_expert]; null on hash layers 0-2
    const float* hc_fn = nullptr;       ///< blk.L.hc_ffn_fn.weight, [24][4*n_embd]
    const float* hc_base = nullptr;     ///< blk.L.hc_ffn_base.weight, [24]
    const float* hc_scale = nullptr;    ///< blk.L.hc_ffn_scale.weight, [3]
    const int32_t* tid2eid = nullptr;   ///< blk.L.ffn_gate_tid2eid.weight I32 [vocab][topk]; null on routed layers
    // shared expert, Q8_0 GGUF blocks on the device
    const void* sh_gate = nullptr;      ///< blk.L.ffn_gate_shexp.weight [n_ff][n_embd]
    const void* sh_up = nullptr;        ///< blk.L.ffn_up_shexp.weight   [n_ff][n_embd]
    const void* sh_down = nullptr;      ///< blk.L.ffn_down_shexp.weight [n_embd][n_ff]
    int sh_type = 8;                    ///< ggml type of the shared-expert tensors (kGgufQ8_0)
};

/// Opaque device workspace, one per forward (sized by the token count).
struct Dsv4MoeWs;

Dsv4MoeWs* dsv4_moe_ws_new(const Dsv4Geometry& g, int64_t s, std::string& err);
void dsv4_moe_ws_free(Dsv4MoeWs* ws);
/// [diag] print the accumulated STRATA_DSV4_TIME MoE split (tag names the pass).
void dsv4_moe_ws_report(Dsv4MoeWs* ws, const char* tag);
/// [S7a-2] the layer count for the predictive-warm window's wrap (0: warm off).
void dsv4_moe_ws_set_layers(Dsv4MoeWs* ws, int n_layers);
/// [S7a-3] the VRAM expert tier: c0 on the compute device, c1 on the peer.  Either may be null.
/// (layer, expert) is homed to cache[expert & 1]; a miss is admitted first-touch and filled on the
/// MoE stream, costing exactly the one H2D the staging path paid, and a hit costs no host traffic at
/// all.  With c1 set, device-1 experts run in a peer pass on device 1 (its slots are never read over
/// P2P by device-0 kernels: that path measured ~0.9 GiB/s inside the GEMV on 2xL20).
void dsv4_moe_ws_set_cache(Dsv4MoeWs* ws, strata::core::ExpertCache* c0, strata::core::ExpertCache* c1);
void dsv4_moe_ws_cache_stats(const Dsv4MoeWs* ws, int64_t& hits, int64_t& misses);
/// [S7b] seed the online heat table from the profile ranking (n = n_layers * n_expert).  With no
/// seed the table starts empty and eviction stays off (first-touch only, the S7a policy).
void dsv4_moe_ws_set_heat(Dsv4MoeWs* ws, const float* heat, int64_t n);
/// [S7b] heat-driven eviction count (a cold resident swapped for a clearly hotter miss).
int64_t dsv4_moe_ws_evictions(const Dsv4MoeWs* ws);
/// [S7b] zero the STRATA_DSV4_TIME accumulators (decode-only splits: reset after the prefill pass).
void dsv4_moe_ws_reset_timing(Dsv4MoeWs* ws);
/// [S7a-2b] the predictive warm window, issued at the LAYER HEAD of `layer` (the engine calls this
/// before the layer's attention half): the previous pass's routing for the pf_depth layers just ahead
/// goes to the file tier's I/O workers, so their preads overlap the attention GPU wait instead of only
/// the MoE staging after the route sync.  Pure I/O scheduling - no numeric effect either way.
void dsv4_moe_prefetch_ahead(Dsv4MoeWs* ws, strata::core::FileExpertSource& experts, int64_t layer);

/// The MoE half of `layer`: h is the [s][4][n_embd] HC state, updated in place (hc_post); d_tokens the
/// [s] prompt ids (the hash layers' tid2eid lookup).  false + err on a hard failure.
bool dsv4_moe_half(const Dsv4Geometry& g, int64_t layer, const Dsv4MoeLayer& w,
                   strata::core::FileExpertSource& experts, float* d_h, const int32_t* d_tokens,
                   int64_t s, Dsv4MoeWs* ws, cudaStream_t stream, std::string& err);

}  // namespace strata::dsv4
