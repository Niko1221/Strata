// Selected only by STRATA_CUDA_SM86_PREFETCH_ONE. Reuse the unchanged shared
// implementation and give its two entry points private-to-this-TU usage names.
#include "strata/prefill/moe_fused_iq.hpp"
#define native_supported sm86_base_native_supported
#define experts_native sm86_base_experts_native
#include "../moe_fused_iq.cu"
#undef experts_native
#undef native_supported
#include "native_fused_iq_sm86.cuh"

namespace strata::prefill::fused {
namespace {
bool prefetch_one() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_PF_PREFETCH_ONE");
        return v && std::strcmp(v, "1") == 0;
    }();
    return enabled;
}
const DevInfo& sm86_info() {
    const DevInfo& base = dev_info();
    int dev = 0;
    ck(cudaGetDevice(&dev), "SM86 device");
    static std::mutex mutex;
    static DevInfo info[32];
    std::lock_guard<std::mutex> lock(mutex);
    DevInfo& d = info[dev & 31];
    if (d.done) return d;
    d.done = true;
    int major = 0, minor = 0;
    ck(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev), "SM86 major");
    ck(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev), "SM86 minor");
    if (!base.ok || !sm86_target(major, minor)) {
        std::fprintf(stderr, "STRATA_PF_PREFETCH_ONE inactive: requires SM86\n");
        return d;
    }
    d.sms = base.sms;
    d.occ = base.occ;
    sm86_setup_pair<T_IQ2_XXS, true, 4, 1>(d.occ);
    sm86_setup_pair<T_IQ2_XS, true, 4, 1>(d.occ);
    sm86_setup_pair<T_IQ2_S, true, 4, 1>(d.occ);
    sm86_setup_pair<T_IQ3_XXS, true, 4, 1>(d.occ);
    sm86_setup_pair<T_IQ3_S, true, 4, 1>(d.occ);
    sm86_setup_pair<T_IQ4_XS, true, 4, 1>(d.occ);
    d.ok = true;
    std::fprintf(stderr, "prefill SM86: STRATA_PF_PREFETCH_ONE=1\n");
    return d;
}
} // namespace

bool native_supported(int gu_type, int d_type) {
    const bool ok = sm86_base_native_supported(gu_type, d_type);
    if (ok && prefetch_one()) (void) sm86_info();
    return ok;
}

void experts_native(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch,
                    const void* xa, const int32_t* src, void* ha, float* dm, void* stream) {
    if (!prefetch_one() || !sm86_info().ok || !gu_covered(g.gu_type) || !d_covered(g.d_type))
        return sm86_base_experts_native(b, g, n_expert, n, scratch, xa, src, ha, dm, stream);
    if (b.e1 <= b.e0 || n <= 0) return;
    const DevInfo& d = sm86_info();
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables(const_cast<void*>(scratch), n_expert);
    const int ww = pick_ww(n, n_expert);
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
    const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * (1280 / weight_rows(ww)), (int64_t) d.sms * d.occ);
    const unsigned g_d = (unsigned) std::min<int64_t>(tiles * (2560 / weight_rows(ww)), (int64_t) d.sms * d.occ);
    switch (g.gu_type) {
        case T_IQ2_XXS: sm86_launch<T_IQ2_XXS, true, 4, 1>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ2_XS: sm86_launch<T_IQ2_XS, true, 4, 1>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ2_S: sm86_launch<T_IQ2_S, true, 4, 1>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ3_XXS: sm86_launch<T_IQ3_XXS, true, 4, 1>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ3_S: sm86_launch<T_IQ3_S, true, 4, 1>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        default: sm86_launch<T_IQ4_XS, true, 4, 1>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
    }
    if (g.d_type == T_Q2_0) launch<T_Q2_0, false>(ww, g_d, b, g, tb, ha, src, nullptr, dm, s);
    else launch<T_IQ4_NL, false>(ww, g_d, b, g, tb, ha, src, nullptr, dm, s);
    ck(cudaGetLastError(), "SM86 prefetch experts");
}
} // namespace strata::prefill::fused
