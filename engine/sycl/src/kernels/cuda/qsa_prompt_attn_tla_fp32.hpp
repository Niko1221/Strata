// FP32 kernels for 24 query heads, two KV heads and a selection capacity of 2051, on Battlemage.
#pragma once

// Included by qsa_prompt_attn_tla.dp.cpp with its FP32 geometry and key conversion helpers.
namespace strata::kernels::tla_prompt_fp32 {
// The projection stores each 256-value query immediately before its gate. Normalize
// with the original 32-lane reduction and rotate the first 64 dimensions in registers.
class PrepareQueries;
void prepare_queries(sycl::queue& queue, const float* q_full, const float* norm, float* q, int64_t nq, int64_t pos0,
                     float eps, const RopeScaling& scaling) {
    const float theta_scale = std::pow(float(scaling.freq_base), -2.f / 64.f);
    const auto args = scaling.kernel_args(64);
    const auto table = rope_table_for(scaling);
    const auto mtab = mrope_table();
    queue.parallel_for<PrepareQueries>(
        sycl::nd_range<2>({size_t(nq * 24), 256}, {1, 256}),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
            const int row = it.get_group(0), d = it.get_local_id(1), lane = d % 32;
            auto sg = it.get_sub_group();
            auto& partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[8]>(it.get_group());
            float x = q_full[size_t(row) * 512 + d], v = x * x;
#pragma unroll
            for (int off = 16; off; off /= 2)
                v += sycl::permute_group_by_xor(sg, v, off);
            if (lane == 0)
                partial[d / 32] = v;
            it.barrier(sycl::access::fence_space::local_space);
            float total = d < 8 ? partial[d] : 0.f;
            if (d < 32) {
#pragma unroll
                for (int off = 16; off; off /= 2)
                    total += sycl::permute_group_by_xor(sg, total, off);
                if (d == 0)
                    partial[0] = sycl::rsqrt(total / 256.f + eps);
            }
            it.barrier(sycl::access::fence_space::local_space);
            x = partial[0] * x * norm[d];
            auto& rotary = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[64]>(it.get_group());
            if (d < 64)
                rotary[d] = x;
            it.barrier(sycl::access::fence_space::local_space);
            if (d < 32) {
                float sn, cs;
                const int pos = mrope_pos(mtab, int(pos0 + row / 24), lane);
                if (!rope_tab_cs(table, pos, lane, cs, sn)) {
                    const float theta = float(pos) * dpct::pow(theta_scale, float(lane));
                    rope_scaled_angle(theta, args.freq_scale, args.corr_low, args.corr_high, args.ext_factor,
                                      args.attn_factor, lane, cs, sn);
                }
                const float a = rotary[lane], b = rotary[lane + 32];
                q[size_t(row) * 256 + lane] = a * cs - b * sn;
                q[size_t(row) * 256 + lane + 32] = a * sn + b * cs;
            } else if (d >= 64)
                q[size_t(row) * 256 + d] = x;
        });
}
constexpr unsigned reverse_four_bits(unsigned value) {
    return ((value & 1) << 3) | ((value & 2) << 1) | ((value & 4) >> 1) | ((value & 8) >> 3);
}
class PackSparseKeys;
// Four rows are one KV head's physical page. Interleaving their 16-byte dot
// operands makes four neighboring cells one cache line at each step of the reduction.
void pack_sparse_keys(sycl::queue& queue, const int8_t* keys, uint64_t* packed, size_t pages, int64_t cells) {
    queue.parallel_for<PackSparseKeys>(sycl::range<1>(pages * 2 * 16 * 4), [=](sycl::id<1> id) {
        const size_t entry = id[0], head_page = entry / 64, cell = entry % 4, leaf = (entry / 4) % 16;
        if ((head_page / 2) * 4 + cell >= size_t(cells))
            return;
        const size_t row = head_page * 4 + cell;
        const auto* codes = reinterpret_cast<const uint64_t*>(keys + row * HD);
        const auto index = reverse_four_bits(leaf);
        using namespace cute;
        alignas(16) uint64_t pair[2] = {codes[index], codes[index + 16]};
        auto from = make_tensor(make_rmem_ptr(pair), make_shape(_2{}));
        auto to = make_tensor(make_gmem_ptr(packed + entry * 2), make_shape(_2{}));
        copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, from, to);
    });
}
class ExpandPrefixValues;
// Decode V once for short prefixes that fit in the existing projection buffer.
void expand_prefix_values(sycl::queue& queue, QsaAttnPools pools, float* output, size_t pages, size_t cells) {
    queue.parallel_for<ExpandPrefixValues>(sycl::range<1>(pages * 2 * 4 * HD), [=](sycl::id<1> id) {
        const size_t i = id[0], row = i / HD;
        if ((row / 8) * 4 + row % 4 < cells)
            output[i] = float(pools.v_q[i]) * h2f(pools.v_scale[i / 64]);
    });
}
// One work-item owns a selected cell. Reuse its decoded key across all 12 heads.
// The recursion folds lane 0 with 8, then 4, 2 and 1 as in the original SIMD reduction.
struct HeadScores {
    float head[12];
};
template <int Stride, bool Packed>
inline HeadScores tree_score(const uint64_t* key, const uint16_t* scale, const float* query, int leaf,
                             const uint64_t* packed, size_t row) {
    if constexpr (Stride == 16) {
        using namespace cute;
        alignas(16) uint64_t words[2];
        if constexpr (Packed) {
            const auto slot = (row / 4 * 16 + reverse_four_bits(leaf)) * 4 + row % 4;
            auto from = make_tensor(make_gmem_ptr(packed + slot * 2), make_shape(_2{}));
            auto to = make_tensor(make_rmem_ptr(words), make_shape(_2{}));
            copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, from, to);
        } else {
#pragma unroll
            for (int side = 0; side < 2; ++side) {
                auto src = make_tensor(make_gmem_ptr(key + leaf + 16 * side), make_shape(_1{}));
                auto dst = make_tensor(make_rmem_ptr(&words[side]), make_shape(_1{}));
                copy(AutoVectorizingCopyWithAssumedAlignment<64>{}, src, dst);
            }
        }
        auto lo = sycl::bit_cast<sycl::vec<int8_t, 8>>(words[0]).convert<float, sycl::rounding_mode::automatic>();
        auto hi = sycl::bit_cast<sycl::vec<int8_t, 8>>(words[1]).convert<float, sycl::rounding_mode::automatic>();
        lo *= h2f(scale[leaf / 8]);
        hi *= h2f(scale[leaf / 8 + 2]);
        HeadScores out;
#pragma unroll
        for (int h = 0; h < 12; ++h) {
            const float* q = query + h * 256 + leaf * 8;
            float l = q[1] * lo[1], u = q[129] * hi[1];
            l = sycl::fma(q[0], lo[0], l);
            u = sycl::fma(q[128], hi[0], u);
#pragma unroll
            for (int j = 2; j < 8; ++j) {
                l = sycl::fma(q[j], lo[j], l);
                u = sycl::fma(q[128 + j], hi[j], u);
            }
            out.head[h] = l + u;
        }
        return out;
    } else {
        auto a = tree_score<Stride * 2, Packed>(key, scale, query, leaf, packed, row);
        auto b = tree_score<Stride * 2, Packed>(key, scale, query, leaf + Stride, packed, row);
#pragma unroll
        for (int h = 0; h < 12; ++h)
            a.head[h] += b.head[h];
        return a;
    }
}
template <bool Expanded>
inline sycl::float4 sparse_value4(QsaAttnPools pools, const float* decoded, size_t row, int dim) {
    const size_t index = row * HD + dim;
    if constexpr (Expanded)
        return *reinterpret_cast<const sycl::float4*>(decoded + index);
    const auto codes = *reinterpret_cast<const sycl::vec<int8_t, 4>*>(pools.v_q + index);
    return codes.convert<float, sycl::rounding_mode::automatic>() * h2f(pools.v_scale[row * 4 + dim / 64]);
}
template <bool Packed> class SparseCore;
class PrefixCore;
template <bool Packed> class SparseMerge;
class PrefixMerge;
class PrefixSubgroupMerge;
template <bool Packed, bool Prefix = false>
void launch_sparse(sycl::queue& queue, const float* q, QsaAttnPools p, const int32_t* ids, const int32_t* steps,
                   int cap, QsaShapes sh, float* scratch, float* out, int nq, const float* q_full, uint16_t* gated,
                   int64_t ld, const uint64_t* packed, int active, const float* decoded_v) {
    using namespace cute;
    constexpr int chunks = 33, nkv = 2, heads = 24, page = 4;
    const int core_chunks = Prefix ? active : 32;
    const int blocks = Prefix ? active : chunks;
    const long long stride = Prefix ? blocks * heads * (HD + 2) : qsa_decode_attn_scratch_floats(cap, sh);
    float *pm = scratch + size_t(blocks) * heads * HD, *pl = pm + size_t(blocks) * heads;

    queue.parallel_for<std::conditional_t<Prefix, PrefixCore, SparseCore<Packed>>>(
        sycl::nd_range<3>({1, 1, size_t(nq) * nkv * core_chunks * Threads}, {1, 1, Threads}),
        [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
            const int group = it.get_group(2), pair = group / core_chunks;
            const int qi = pair / 2, kv = pair % 2, chunk = group % core_chunks;
            const int t = it.get_local_id(2), lane = t & 15, sgid = t >> 4;
            const auto sg = it.get_sub_group();
            auto& sp = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][CH]>(it.get_group());
            auto& srow = *sycl::ext::oneapi::group_local_memory_for_overwrite<long long[CH]>(it.get_group());
            const int width = steps[size_t(qi) * kStepCount + kStepWidth], c0 = chunk * CH,
                      n_here = sycl::min(CH, width - c0), slot = kv * blocks + chunk;
            float *pa = scratch + size_t(qi) * stride, *ms = pm + size_t(qi) * stride, *ls = pl + size_t(qi) * stride;
            const float* query = q + (size_t(qi) * heads + kv * G) * HD;
            if (n_here <= 0) {
                if (t < G) {
                    ms[slot * G + t] = -FLT_MAX;
                    ls[slot * G + t] = 0.f;
                }
                return;
            }
            long long row_check = -1;
            if (t < n_here) {
                const int cell = ids[size_t(qi) * 2051 + c0 + t];
                const int pg = p.page_table[cell / page];
                if (pg >= 0)
                    row_check = (static_cast<long long>(pg) * nkv + kv) * page + cell % page;
            }
            srow[t] = row_check;
            const float down = 1.f / 16.f;
            auto& flags = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[4]>(it.get_group());
            const bool resident = sycl::all_of_group(sg, row_check >= 0);
            if (lane == 0)
                flags[sgid] = int(resident);
            it.barrier(sycl::access::fence_space::local_space);
            const bool full_resident = n_here == CH && flags[0] && flags[1] && flags[2] && flags[3];
            const auto row = srow[t];
            if (row >= 0) {
                auto scores = tree_score<1, Packed>(reinterpret_cast<const uint64_t*>(p.k_q + row * HD),
                                                    p.k_scale + row * 4, query, 0, packed, size_t(row));
#pragma unroll
                for (int h = 0; h < G; ++h)
                    sp[h][t] = scores.head[h] * down;
            } else {
                for (int h = 0; h < G; ++h)
                    sp[h][t] = -FLT_MAX;
            }
            it.barrier(sycl::access::fence_space::local_space);
            // Fold original warp32 offset16 first, then reproduce offsets8/4/2/1.
            for (int h = sgid; h < G; h += Threads / 16) {
                float m = sycl::fmax(sycl::fmax(sp[h][lane], sp[h][lane + 32]),
                                     sycl::fmax(sp[h][lane + 16], sp[h][lane + 48]));
#pragma unroll
                for (int off = 8; off > 0; off >>= 1)
                    m = sycl::fmax(m, sycl::permute_group_by_xor(sg, m, off));
                const float ea = (lane < n_here && srow[lane] >= 0) ? sycl::native::exp(sp[h][lane] - m) : 0.f;
                const float eb =
                    (lane + 32 < n_here && srow[lane + 32] >= 0) ? sycl::native::exp(sp[h][lane + 32] - m) : 0.f;
                const float ec =
                    (lane + 16 < n_here && srow[lane + 16] >= 0) ? sycl::native::exp(sp[h][lane + 16] - m) : 0.f;
                const float ed =
                    (lane + 48 < n_here && srow[lane + 48] >= 0) ? sycl::native::exp(sp[h][lane + 48] - m) : 0.f;
                sp[h][lane] = ea;
                sp[h][lane + 32] = eb;
                sp[h][lane + 16] = ec;
                sp[h][lane + 48] = ed;
                float l = (ea + eb) + (ec + ed);
#pragma unroll
                for (int off = 8; off > 0; off >>= 1)
                    l += sycl::permute_group_by_xor(sg, l, off);
                if (lane == 0) {
                    ms[slot * G + h] = m;
                    ls[slot * G + h] = l;
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            auto do_pv = [&](auto tag) {
                constexpr bool All = decltype(tag)::value;
                const int d = 4 * t;
                sycl::float4 acc[G];
#pragma unroll
                for (int h = 0; h < G; ++h)
                    acc[h] = sycl::float4(0.f);
                for (int c = 0; c < (All ? CH : n_here); c += 8) {
                    long long rows[8];
                    bool full = All || c + 8 <= n_here;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        if constexpr (All)
                            rows[j] = srow[c + j];
                        else {
                            rows[j] = c + j < n_here ? srow[c + j] : -1;
                            full = full && rows[j] >= 0;
                        }
                    }
                    if (full) {
                        sycl::float4 values[8];
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            values[j] = sparse_value4<Prefix>(p, decoded_v, rows[j], d);
                        }
#pragma unroll
                        for (int h = 0; h < G; ++h) {
                            const auto p0 = *reinterpret_cast<const sycl::float4*>(&sp[h][c]);
                            const auto p1 = *reinterpret_cast<const sycl::float4*>(&sp[h][c + 4]);
                            acc[h] = sycl::fma(sycl::float4(p0[0]), values[0], acc[h]);
                            acc[h] = sycl::fma(sycl::float4(p0[1]), values[1], acc[h]);
                            acc[h] = sycl::fma(sycl::float4(p0[2]), values[2], acc[h]);
                            acc[h] = sycl::fma(sycl::float4(p0[3]), values[3], acc[h]);
                            acc[h] = sycl::fma(sycl::float4(p1[0]), values[4], acc[h]);
                            acc[h] = sycl::fma(sycl::float4(p1[1]), values[5], acc[h]);
                            acc[h] = sycl::fma(sycl::float4(p1[2]), values[6], acc[h]);
                            acc[h] = sycl::fma(sycl::float4(p1[3]), values[7], acc[h]);
                        }
                    } else {
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            if (rows[j] < 0)
                                continue;
                            const auto value = sparse_value4<Prefix>(p, decoded_v, rows[j], d);
#pragma unroll
                            for (int h = 0; h < G; ++h)
                                acc[h] = sycl::fma(sycl::float4(sp[h][c + j]), value, acc[h]);
                        }
                    }
                }
#pragma unroll
                for (int h = 0; h < G; ++h) {
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        pa[(size_t(slot) * G + h) * HD + d + j] = acc[h][j];
                }
            };
            if (full_resident)
                do_pv(std::true_type{});
            else
                do_pv(std::false_type{});
        });

    // For a causal tile shorter than 32 chunks there is no tail. One subgroup can
    // keep all merge weights in registers and write one complete output head.
    if constexpr (Prefix) {
        if (active < 32 && reinterpret_cast<uintptr_t>(scratch) % alignof(sycl::float8) == 0) {
            queue.parallel_for<PrefixSubgroupMerge>(
                sycl::nd_range<1>({size_t(nq) * 24 * 32}, {32}),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                    const auto sg = it.get_sub_group();
                    const int lane = int(sg.get_local_linear_id());
                    const size_t head_index = it.get_global_linear_id() / 32;
                    const int qi = int(head_index / 24), head = int(head_index % 24);
                    const int slot = (head / G) * blocks * G + (head % G);
                    const float* partial = scratch + size_t(qi) * stride + slot * HD;
                    const float *maxima = pm + size_t(qi) * stride + slot, *sums = pl + size_t(qi) * stride + slot;
                    const float m = lane < active ? maxima[lane * G] : -FLT_MAX;
                    const float maximum = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
                    const float weight = m != -FLT_MAX ? sycl::native::exp(m - maximum) : 0.f;
                    const float length = lane < active ? sums[lane * G] : 0.f;
                    const unsigned valid =
                        sycl::reduce_over_group(sg, m != -FLT_MAX ? (1u << lane) : 0u, sycl::bit_or<unsigned>());
                    float sum = 0.f;
                    constexpr int values_per_lane = HD / 32;
                    sycl::vec<float, values_per_lane> values(0.f);
                    const int dim = lane * values_per_lane;
                    for (int c = 0; c < active; ++c) {
                        const float w = sycl::select_from_group(sg, weight, sycl::id<1>(c));
                        const float l = sycl::select_from_group(sg, length, sycl::id<1>(c));
                        if (!(valid & (1u << c)))
                            continue;
                        sum = sycl::fma(l, w, sum);
                        auto v = *reinterpret_cast<const sycl::vec<float, values_per_lane>*>(partial +
                                                                                             size_t(c) * G * HD + dim);
                        values = sycl::fma(v, sycl::vec<float, values_per_lane>(w), values);
                    }
                    const size_t offset = (size_t(qi) * heads + head) * HD + dim;
#pragma unroll
                    for (int j = 0; j < values_per_lane; ++j) {
                        const float value = sum > 0.f ? values[j] / sum : 0.f;
                        if (gated) {
                            const float g = q_full[(size_t(qi) * heads + head) * 2 * HD + HD + dim + j];
                            const float x = value * (1.f / (1.f + sycl::native::exp(-g)));
                            gated[size_t(qi) * ld + head * HD + dim + j] = sycl::bit_cast<uint16_t>(
                                sycl::vec<float, 1>(x).convert<sycl::half, sycl::rounding_mode::rte>()[0]);
                        } else
                            out[offset + j] = value;
                    }
                });
            return;
        }
    }
    {
        // One workgroup merges one output head. Its first three subgroups compute the
        // final zero to three selected cells, which never pass through global scratch.
        constexpr int chunk_count = 33;
        queue.parallel_for<std::conditional_t<Prefix, PrefixMerge, SparseMerge<Packed>>>(
            sycl::nd_range<3>({size_t(nq), size_t(heads), HD}, {1, 1, HD}),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(32)]] {
                const int qi = it.get_group(0), head = it.get_group(1), dim = it.get_local_id(2);
                const int head_slot = (head / G) * blocks * G + head % G;
                const size_t query_offset = size_t(qi) * stride;
                const float* maxima = pm + query_offset + head_slot;
                const float* sums = pl + query_offset + head_slot;
                const float* partial = scratch + query_offset + head_slot * HD + dim;
                auto group = it.get_group();
                auto& weights = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[chunk_count]>(group);
                auto& lengths = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[chunk_count]>(group);
                auto& valid = *sycl::ext::oneapi::group_local_memory_for_overwrite<unsigned[2]>(group);
                auto& tail_rows = *sycl::ext::oneapi::group_local_memory_for_overwrite<long long[3]>(group);
                auto& tail_score = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[3]>(group);
                const int count = sycl::clamp(int(steps[qi * kStepCount + kStepWidth]) - 2048, 0, 3);
                const int lane32 = dim % 32, sgnum = dim / 32;
                if (sgnum < 3) {
                    long long row = -1;
                    if (sgnum < count) {
                        int cell = ids[qi * 2051 + 2048 + sgnum], pg = p.page_table[cell / 4];
                        if (pg >= 0)
                            row = (static_cast<long long>(pg) * 2 + head / 12) * 4 + cell % 4;
                    }
                    float v = -FLT_MAX;
                    if (row >= 0) {
                        const float* query = q + (size_t(qi) * 24 + (head / 12) * 12) * 256;
                        auto bytes = exact_signed_bytes(reinterpret_cast<const uint64_t*>(p.k_q + row * HD)[lane32]);
                        float k[8], values[8];
                        const float sc = h2f(p.k_scale[row * 4 + lane32 / 8]);
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            k[j] = bytes[j] * sc;
                            values[j] = query[(head % 12) * 256 + lane32 * 8 + j];
                        }
                        float x = values[1] * k[1];
                        x = sycl::fma(values[0], k[0], x);
#pragma unroll
                        for (int j = 2; j < 8; ++j)
                            x = sycl::fma(values[j], k[j], x);
                        v = sycl::reduce_over_group(it.get_sub_group(), x, sycl::plus<float>()) * (1.f / 16.f);
                    }
                    if (!lane32) {
                        tail_rows[sgnum] = row;
                        tail_score[sgnum] = v;
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                const float tail_max = sycl::fmax(tail_score[0], sycl::fmax(tail_score[1], tail_score[2]));
                float tail_p[3];
#pragma unroll
                for (int j = 0; j < 3; ++j)
                    tail_p[j] = tail_rows[j] >= 0 ? sycl::native::exp(tail_score[j] - tail_max) : 0.f;
                float tail_value = 0.f;
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    long long r = tail_rows[j];
                    if (r >= 0)
                        tail_value = sycl::fma(tail_p[j], float(p.v_q[r * HD + dim]) * h2f(p.v_scale[r * 4 + dim / 64]),
                                               tail_value);
                }
                if (dim < 32) {
                    const auto sg = it.get_sub_group();
                    const float m = (dim < core_chunks ? maxima[dim * G] : -FLT_MAX), tail = tail_max;
                    const float maximum =
                        sycl::reduce_over_group(sg, sycl::fmax(-FLT_MAX, sycl::fmax(m, tail)), sycl::maximum<float>());
                    weights[dim] = m != -FLT_MAX ? sycl::native::exp(m - maximum) : 0.f;
                    lengths[dim] = dim < core_chunks ? sums[dim * G] : 0.f;
                    const unsigned bits =
                        sycl::reduce_over_group(sg, m != -FLT_MAX ? (1u << dim) : 0u, sycl::bit_or<unsigned>());
                    if (dim == 0) {
                        valid[0] = bits;
                        valid[1] = unsigned(tail != -FLT_MAX);
                        weights[32] = tail != -FLT_MAX ? sycl::native::exp(tail - maximum) : 0.f;
                        lengths[32] = (tail_p[0] + tail_p[2]) + tail_p[1];
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                float sum = 0.f, value = 0.f;
                auto accumulate = [&](auto all_valid) {
#pragma unroll 4
                    for (int c = 0; c < core_chunks; ++c) {
                        if constexpr (!decltype(all_valid)::value) {
                            if (!(valid[c / 32] & (1u << (c % 32))))
                                continue;
                        }
                        sum = sycl::fma(lengths[c], weights[c], sum);
                        value = sycl::fma(partial[c * G * HD], weights[c], value);
                    }
                };
                if (valid[0] == 0xffffffffu && valid[1])
                    accumulate(std::true_type{});
                else
                    accumulate(std::false_type{});
                if (valid[1]) {
                    sum = sycl::fma(lengths[32], weights[32], sum);
                    value = sycl::fma(tail_value, weights[32], value);
                }
                const float result = sum > 0.f ? value / sum : 0.f;
                if (gated) {
                    const float gate = q_full[(size_t(qi) * heads + head) * 2 * HD + HD + dim];
                    const float x = result * (1.f / (1.f + sycl::native::exp(-gate)));
                    gated[size_t(qi) * ld + head * HD + dim] = sycl::bit_cast<uint16_t>(
                        sycl::vec<float, 1>(x).convert<sycl::half, sycl::rounding_mode::rte>()[0]);
                } else
                    out[(size_t(qi) * heads + head) * HD + dim] = result;
            });
    }
}

} // namespace strata::kernels::tla_prompt_fp32
