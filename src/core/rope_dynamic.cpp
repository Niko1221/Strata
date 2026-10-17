// src/core/rope_dynamic.cpp - see include/strata/core/rope_dynamic.hpp.
#include "strata/core/rope_dynamic.hpp"

#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/rope_dynamic.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

namespace strata::core {

double RopeDynamic::factor_for(int64_t positions) const {
    if ((double) positions <= orig) return 1.0;
    const double f = std::ceil((double) positions / orig / step - 1e-9) * step;
    return on_grid(std::max(1.0, std::min(max_factor, f)));
}

strata::kernels::RopeScaling RopeDynamic::config(double f) const {
    if (f <= 1.0) {
        strata::kernels::RopeScaling none;
        none.freq_base = base.freq_base;
        return none;
    }
    strata::kernels::RopeScaling s = base;
    s.type = strata::kernels::RopeScalingType::YaRN;
    s.factor = f;
    return s;
}

bool rope_rerotate_state(const QsaState& st, int64_t cells, bool index, const strata::kernels::RopeScaling& from,
                         const strata::kernels::RopeScaling& to, void* stream, std::string& error) {
    using namespace strata::kernels;
    if (cells <= 0) return true;
    if (st.kv_mode != 0 && !st.host.present()) { error = "dynamic rope: a streamed K/V without its host copy"; return false; }
    if (st.kv_q4 || st.kv_hybrid) { error = "dynamic rope: --kv q4_0 / k8v4 is not supported"; return false; }
    if (cells > st.max_cells) { error = "dynamic rope: more cells than the state holds"; return false; }
    const QsaShapes s = qsa_real_shapes();
    const RopeRates a = rope_rates((int) s.n_rot, from), b = rope_rates((int) s.n_rot, to);
    RopeKPool k;
    if (st.kv_mode != 0) {   // streamed or a ring: the host copy is the K/V (identity layout); the slots are refetched
        if (st.kv_int8) { k.k_q = st.host.k_q; k.k_scale = st.host.k_scale; }
        else k.k_pool = st.host.k_pool;
    } else {
        if (st.kv_int8) { k.k_q = st.k_q; k.k_scale = st.k_scale; }
        else k.k_pool = st.k_pool;
        k.page_table = st.page_table;
    }
    k.n_kv_heads = (int) s.n_head_kv;
    k.page_size = (int) s.page_size;
    k.head_dim = (int) s.head_dim;
    k.hadamard = st.kv_rot;
    rope_rerotate_k(k, cells, a, b, stream);
    if (st.kv_mode == 1) kv_stream_reset(st.map, stream);   // every slot fetched again from the rotated host copy
    if (index && st.idx_pooled != nullptr)
        rope_rerotate_pooled(st.idx_pooled, st.idx_dead, cells / s.idx_block, (int) s.idx_block, (int) s.idx_dim, a, b,
                             stream);
    return true;
}

bool rope_table_rebuild(const QsaState& owner, const strata::kernels::RopeScaling& to, std::string& error) {
    using namespace strata::kernels;
    if (owner.cos_tab == nullptr || owner.sin_tab == nullptr || owner.max_cells <= 0) {
        error = "dynamic rope: no rope table";
        return false;
    }
    const QsaShapes s = qsa_real_shapes();
    const size_t n = (size_t) owner.max_cells * (size_t) (s.n_rot / 2);
    std::vector<float> hc(n), hs(n);
    const int threads = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    build_rope_table_mt((int) s.n_rot, to, (int) owner.max_cells, hc.data(), hs.data(), threads);
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(owner.cos_tab, hc.data(), n * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(owner.sin_tab, hs.data(), n * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
        error = "dynamic rope: the table upload failed";
        return false;
    }
    rope_table_set(owner.cos_tab, owner.sin_tab, (int) owner.max_cells, to);
    rope_scaling_set(to);
    return true;
}

}  // namespace strata::core
