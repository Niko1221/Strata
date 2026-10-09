// Selected only by STRATA_CUDA_SM86_IQ3_STAGE2. The original shared source and
// its HIP implementation are included unchanged as the fallback.
#include "strata/prefill/moe_fused_iq.hpp"
#define native_supported sm86_base_native_supported
#define experts_native sm86_base_experts_native
#include "../moe_fused_iq.cu"
#undef experts_native
#undef native_supported
#include "native_fused_iq_sm86.cuh"

namespace strata::prefill::fused {
namespace {
bool iq3_stage2() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_PF_IQ3_STAGE2");
        return v && std::strcmp(v, "1") == 0;
    }();
    return enabled;
}
bool stage2_pair(int gu, int down) { return (gu == T_IQ3_S || gu == T_IQ3_XXS) && down == T_IQ4_NL; }
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
        std::fprintf(stderr, "STRATA_PF_IQ3_STAGE2 inactive: requires SM86\n");
        return d;
    }
    d.sms = base.sms;
    d.occ = base.occ;
    sm86_setup_pair<T_IQ3_S, true, 2, 2>(d.occ);
    sm86_setup_pair<T_IQ3_XXS, true, 2, 2>(d.occ);
    sm86_setup_pair<T_IQ4_NL, false, 2, 2>(d.occ);
    d.ok = true;
    std::fprintf(stderr, "prefill SM86: STRATA_PF_IQ3_STAGE2=1\n");
    return d;
}
} // namespace

bool native_supported(int gu_type, int d_type) {
    const bool ok = sm86_base_native_supported(gu_type, d_type);
    if (ok && iq3_stage2() && stage2_pair(gu_type, d_type)) (void) sm86_info();
    return ok;
}

void experts_native(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch,
                    const void* xa, const int32_t* src, void* ha, float* dm, void* stream) {
    if (!iq3_stage2() || !stage2_pair(g.gu_type, g.d_type) || !sm86_info().ok)
        return sm86_base_experts_native(b, g, n_expert, n, scratch, xa, src, ha, dm, stream);
    if (b.e1 <= b.e0 || n <= 0) return;
    const DevInfo& d = sm86_info();
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables(const_cast<void*>(scratch), n_expert);
    const int ww = pick_ww(n, n_expert);
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
    const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * (1280 / weight_rows(ww)), (int64_t) d.sms * d.occ);
    const unsigned g_d = (unsigned) std::min<int64_t>(tiles * (2560 / weight_rows(ww)), (int64_t) d.sms * d.occ);
    if (g.gu_type == T_IQ3_S) sm86_launch<T_IQ3_S, true, 2, 2>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s);
    else sm86_launch<T_IQ3_XXS, true, 2, 2>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s);
    sm86_launch<T_IQ4_NL, false, 2, 2>(ww, g_d, b, g, tb, ha, src, nullptr, dm, s);
    ck(cudaGetLastError(), "SM86 IQ3 stage2 experts");
}
} // namespace strata::prefill::fused
