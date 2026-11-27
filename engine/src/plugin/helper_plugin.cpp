// helper_plugin.cpp — strata-mi50, 2026-10-23
// strata_helper_gfx906.dll: the helper-GPU expert tier (RemoteExperts) built in the gfx906 / HIP 5.x tree and loaded by an engine
// that runs on another HIP runtime (the RX 9070 XT's).  Same class, same kernels as the in-process helper tier; only the device
// handling differs: here the runtime sees the helper GPU alone, so it is "standalone" (no primary device to keep clear of).
// ABI: include/strata/core/helper_plugin_abi.h.  Plan: PIANO_DUAL_GPU_2026-10-23.md.
#include "strata/core/helper_plugin_abi.h"
#include "strata/core/remote_experts.hpp"
#include "strata/core/remote_expert_opt.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "helper_prefill.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {
struct State {
    std::string pack;
    int64_t layers = 0, experts = 0;
    int device = 0;
    bool layout = false;
};
State& state() { static State s; return s; }

void put(char* err, int len, const std::string& msg) {
    if (err != nullptr && len > 0) std::snprintf(err, (size_t) len, "%s", msg.c_str());
}

struct Tier {
    strata::core::RemoteExperts remote;
    // 2026-10-26: --remote-expert-opt (ABI 2).  Declared after `remote`, so it is destroyed first (its destructor uses remote).
    std::unique_ptr<strata::core::RemoteExpertOpt> opt;
    int64_t tokens = 0, k = 0;   ///< the window begin_opt reduced
    // 2026-10-27: STRATA_PREFILL_HELPER (ABI 5): the work space of the prompt path's expert share
    strata::plugin::PrefillWs* pf = nullptr;
    ~Tier() { strata::plugin::prefill_destroy(pf); }
};
}  // namespace

extern "C" {

STRATA_HELPER_EXPORT int strata_helper_abi_version(void) { return STRATA_HELPER_ABI_VERSION; }

STRATA_HELPER_EXPORT int strata_helper_init(const char* pack_dir, int64_t n_layers, int64_t n_expert, int device,
                                            double* free_gib, char* err, int err_len) {
    State& st = state();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || device < 0 || device >= count) {
        cudaGetLastError();
        put(err, err_len, "the plugin's HIP runtime sees " + std::to_string(count) + " device(s); device " + std::to_string(device) +
                              " is not one of them");
        return 1;
    }
    if (cudaSetDevice(device) != cudaSuccess) { put(err, err_len, "cudaSetDevice failed"); return 1; }
    st.device = device;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { put(err, err_len, "cudaMemGetInfo failed"); return 1; }
    if (free_gib != nullptr) *free_gib = (double) free_b / 1073741824.0;
    if (n_layers > 0 && (!st.layout || st.pack != pack_dir || st.layers != n_layers || st.experts != n_expert)) {
        std::string e;
        if (!strata::kernels::cpu::expert_layout_load(pack_dir, n_layers, n_expert, e)) {
            put(err, err_len, "expert layout: " + e);
            return 1;
        }
        st.pack = pack_dir;
        st.layers = n_layers;
        st.experts = n_expert;
        st.layout = true;
    }
    return 0;
}

STRATA_HELPER_EXPORT void* strata_helper_open2(int slots, const int32_t* ranked, int64_t n_ranked, strata_helper_primary_fn primary,
                                               strata_helper_blob_fn blob, void* ctx, uint8_t* claimed, int auto_size, int reduce,
                                               char* err, int err_len) {
    State& st = state();
    if (!st.layout) { put(err, err_len, "strata_helper_init with the model's layers first"); return nullptr; }
    std::vector<std::pair<int32_t, int32_t>> pairs;
    pairs.reserve((size_t) n_ranked);
    for (int64_t i = 0; i < n_ranked; ++i) pairs.emplace_back(ranked[2 * i], ranked[2 * i + 1]);
    const size_t n = (size_t) st.layers * (size_t) st.experts;
    std::vector<uint8_t> taken(claimed, claimed + n);
    auto tier = std::make_unique<Tier>();
    std::string e;
    if (reduce != 0) {   // attach before open: open sizes the metadata upload for the reduction
        tier->opt = std::make_unique<strata::core::RemoteExpertOpt>();
        tier->opt->attach(tier->remote);
    }
    const bool ok = tier->remote.open_with(
        st.device, slots, st.layers, st.experts, pairs,
        [&](int32_t l, int32_t x) { return primary != nullptr && primary(ctx, l, x) != 0; },
        [&](int64_t l, int64_t x) { return blob(ctx, l, x); }, taken, e, auto_size != 0, true);
    if (!ok) { put(err, err_len, e); return nullptr; }
    if (tier->opt && !tier->opt->init(e)) { put(err, err_len, e); return nullptr; }
    std::memcpy(claimed, taken.data(), n);
    return tier.release();
}

STRATA_HELPER_EXPORT void* strata_helper_open(int slots, const int32_t* ranked, int64_t n_ranked, strata_helper_primary_fn primary,
                                              strata_helper_blob_fn blob, void* ctx, uint8_t* claimed, int auto_size,
                                              char* err, int err_len) {
    return strata_helper_open2(slots, ranked, n_ranked, primary, blob, ctx, claimed, auto_size, 0, err, err_len);
}

STRATA_HELPER_EXPORT void strata_helper_close(void* h) { delete (Tier*) h; }

STRATA_HELPER_EXPORT int strata_helper_begin(void* h, int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                                             const int32_t* kind, const int32_t* primary_res, char* err, int err_len) {
    std::string e;
    if (!((Tier*) h)->remote.begin(layer, x, ids, n_tok, k, kind, primary_res, e)) { put(err, err_len, e); return 1; }
    return 0;
}

STRATA_HELPER_EXPORT int strata_helper_finish(void* h, float* out, char* err, int err_len) {
    std::string e;
    if (!((Tier*) h)->remote.finish(out, e)) { put(err, err_len, e); return 1; }
    return 0;
}

STRATA_HELPER_EXPORT int strata_helper_begin_opt(void* h, int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                                                 const int32_t* kind, const int32_t* primary_res, const float* weights,
                                                 char* err, int err_len) {
    Tier& t = *(Tier*) h;
    if (!t.opt || weights == nullptr) { put(err, err_len, "helper plugin: begin_opt on a tier opened without reduce"); return 1; }
    t.opt->begin(weights, 0, (int) n_tok);   // the plugin's own sum covers this window alone
    t.tokens = n_tok;
    t.k = k;
    std::string e;
    if (!t.remote.begin(layer, x, ids, n_tok, k, kind, primary_res, e)) { t.opt->end(); put(err, err_len, e); return 1; }
    return 0;
}

STRATA_HELPER_EXPORT int strata_helper_finish_opt(void* h, float* sum, uint8_t* owned, char* err, int err_len) {
    Tier& t = *(Tier*) h;
    if (!t.opt) { put(err, err_len, "helper plugin: finish_opt on a tier opened without reduce"); return 1; }
    std::string e;
    // with the reduction active finish() adds the reduced rows to the plugin's sum (zeroed by begin) and writes nothing to `out`
    const bool ok = t.remote.finish(nullptr, e);
    if (ok) {
        std::memcpy(sum, t.opt->host_sum(), (size_t) t.tokens * (size_t) strata::kernels::cpu::H * sizeof(float));
        for (int64_t i = 0; i < t.tokens * t.k; ++i) owned[i] = t.remote.owns(i) ? 1 : 0;
    }
    t.opt->end();
    if (!ok) { put(err, err_len, e); return 1; }
    return 0;
}

STRATA_HELPER_EXPORT int strata_helper_adapt(void* h, const float* usage, const int32_t* primary, uint8_t* resident, int64_t n,
                                             int max_swaps, strata_helper_blob_fn blob, void* ctx, int* swapped, char* err, int err_len) {
    int got = 0;
    std::string e;
    const bool ok = ((Tier*) h)->remote.adapt_swaps(usage, primary, resident, n, max_swaps,
                                                    [&](int64_t l, int64_t x) { return blob(ctx, l, x); }, got, e);
    if (swapped != nullptr) *swapped = got;
    if (!ok) { put(err, err_len, e); return 1; }
    return 0;
}

STRATA_HELPER_EXPORT int strata_helper_copy_blobs(void* h, const int32_t* layers, const int32_t* experts, uint8_t* const* dst,
                                                  int64_t n, uint8_t* hit, char* err, int err_len) {
    std::string e;
    if (!((Tier*) h)->remote.copy_cached_blobs(layers, experts, dst, (size_t) n, hit, e)) { put(err, err_len, e); return 1; }
    return 0;
}

STRATA_HELPER_EXPORT int strata_helper_prefill_reserve(void* h, int64_t t_max, uint8_t* layer_ok, int64_t n_layers, uint16_t** x16,
                                                       uint16_t** sum16, char* err, int err_len) {
    Tier& t = *(Tier*) h;
    std::vector<uint8_t> ok((size_t) n_layers, 0);
    std::string e;
    if (!strata::plugin::prefill_reserve(t.pf, state().device, t_max, ok, x16, sum16, e)) { put(err, err_len, e); return 1; }
    std::memcpy(layer_ok, ok.data(), ok.size());
    return 0;
}

STRATA_HELPER_EXPORT int strata_helper_prefill_begin(void* h, int64_t layer, int64_t t, int64_t k, const uint16_t* x16, const int32_t* ids,
                                                     const float* w, const uint8_t* take, char* err, int err_len) {
    Tier& tr = *(Tier*) h;
    std::string e;
    if (!strata::plugin::prefill_begin(tr.pf, tr.remote, layer, t, k, x16, ids, w, take, e)) { put(err, err_len, e); return 1; }
    return 0;
}

STRATA_HELPER_EXPORT int strata_helper_prefill_finish(void* h, uint16_t* sum16, double* ms_wait, char* err, int err_len) {
    std::string e;
    if (!strata::plugin::prefill_finish(((Tier*) h)->pf, sum16, ms_wait, e)) { put(err, err_len, e); return 1; }
    return 0;
}

STRATA_HELPER_EXPORT int strata_helper_owns(void* h, int64_t index) { return ((Tier*) h)->remote.owns(index) ? 1 : 0; }

STRATA_HELPER_EXPORT int strata_helper_holds(void* h, int64_t layer, int32_t expert) {
    return ((Tier*) h)->remote.holds(layer, expert) ? 1 : 0;
}

STRATA_HELPER_EXPORT void strata_helper_stats(void* h, int64_t* c, double* d) {
    const auto& r = ((Tier*) h)->remote;
    c[0] = r.resident(); c[1] = r.computed(); c[2] = r.launched_layers();
    c[3] = (int64_t) r.returned_bytes(); c[4] = (int64_t) r.full_row_bytes();
    d[0] = r.gib(); d[1] = r.ms_begin(); d[2] = r.ms_wait();
}

}  // extern "C"
