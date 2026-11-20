#pragma once
// Opt-in Qwen3.6 hotspot / transfer counters for T07 profile capture (and T06 inputs).
// Emit only when STRATA_Q36_HOTSPOT_PROFILE is set. Default path stays silent.
// Wall times are steady-clock milliseconds on the diagnostic arm (may sync CUDA).
// hardware_counters stays "null" until an Nsight Compute capture is recorded.
#include <cstdio>
#include <cstdlib>

namespace strata::qwen35 {

inline bool q36_hotspot_enabled() {
    static const bool on = std::getenv("STRATA_Q36_HOTSPOT_PROFILE") != nullptr;
    return on;
}

/// One stderr summary line. phase is warmup|forward|shutdown|manual.
inline void q36_hotspot_summary(const char* phase, long long layers_gdn, long long layers_attn, double gdn_host_ms,
                                double attn_host_ms, double moe_gpu_host_ms, double moe_cpu_ms,
                                double stream_copy_wait_ms, double stream_mmq_ms, long long stream_h2d_bytes,
                                long long stream_d2d_bytes, long long hy_cpu, long long hy_gpu, long long hy_swaps,
                                long long rebalance_h2d_bytes, const char* note) {
    std::fprintf(stderr,
                 "Q36_HOTSPOT schema=1 phase=%s layers_gdn=%lld layers_attn=%lld gdn_host_ms=%.3f attn_host_ms=%.3f "
                 "moe_gpu_host_ms=%.3f moe_cpu_ms=%.3f stream_copy_wait_ms=%.3f stream_mmq_ms=%.3f "
                 "stream_h2d_bytes=%lld stream_d2d_bytes=%lld hy_cpu=%lld hy_gpu=%lld hy_swaps=%lld "
                 "rebalance_h2d_bytes=%lld hardware_counters=null note=%s\n",
                 phase && phase[0] ? phase : "unknown", layers_gdn, layers_attn, gdn_host_ms, attn_host_ms,
                 moe_gpu_host_ms, moe_cpu_ms, stream_copy_wait_ms, stream_mmq_ms, stream_h2d_bytes, stream_d2d_bytes,
                 hy_cpu, hy_gpu, hy_swaps, rebalance_h2d_bytes, note && note[0] ? note : "null");
}

}  // namespace strata::qwen35
