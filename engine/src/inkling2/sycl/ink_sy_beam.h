// SYCL implementation of the Inkling CUDA DSpark beam search. The target
// always chooses the emitted token, including when the draft misses.
#pragma once

#include "ink_sy_dspark.h"
#include <algorithm>
#include <cmath>
#include <climits>

namespace ink_sycl {
constexpr int beam_width = 4;
constexpr int screen_size = 512;
constexpr int tree_capacity = 12;
constexpr int pool_capacity = 1 + 4 + 6 * 8;

struct BeamNode {
    int token, parent, depth;
    float logp;
};
struct BeamPool {
    BeamNode nodes[pool_capacity];
    int size, active[beam_width], from[beam_width];
};
struct BeamPick {
    int token, beam;
    float logp;
};

struct VerifyTree {
    int count = 0;
    int tokens[tree_capacity]{}, parent[tree_capacity]{}, depth[tree_capacity]{};
    int ancestry[tree_capacity][8]{};
};

// CUDA's sparse-MoE sizing model (eight routed of 128 experts).
inline double beam_verify_cost(int rows) { return 1.0 + 4.2 * (1.0 - std::pow(15.0 / 16.0, rows)); }

inline VerifyTree prune_beams(const BeamPool& pool, bool tight = false) {
    int order[pool_capacity]{}, remap[pool_capacity]{};
    bool keep[pool_capacity]{true};
    for (int i = 1; i < pool.size; ++i) order[i - 1] = i;
    std::sort(order, order + pool.size - 1, [&](int a, int b) {
        const auto &x = pool.nodes[a], &y = pool.nodes[b];
        if (x.logp != y.logp) return x.logp > y.logp;
        if (x.depth != y.depth) return x.depth < y.depth;
        return a < b;
    });
    double prefix[pool_capacity]{1.0}, best = 0.0;
    for (int i = 1; i < pool.size; ++i) {
        const auto& node = pool.nodes[order[i - 1]];
        const double p = node.logp;
        const double confidence = tight ? std::exp(2 * p) * std::pow(0.9, node.depth - 1) : std::exp(p);
        prefix[i] = prefix[i - 1] + (p > -20 ? confidence : 0.0);
    }
    int want = 1;
    for (int m : {2, 3, 4, 5, 6, 8, 10, 12}) {
        if (m > pool.size || (tight && m > 6)) break;
        if (m == 2 && !tight) continue;
        // Fit the SG16 SQ-20 verifier on single-card B70, with a margin on extra rows.
        constexpr double verifier_ms[] = {0., 0., 24.413, 27.892, 31.371, 34.478, 38.156};
        const double cost = tight ? verifier_ms[m] : beam_verify_cost(m);
        double rate = prefix[m - 1] / cost;
        if (rate > best) {
            want = m;
            best = rate;
        }
    }
    for (int i = 0, n = 1; i < pool.size - 1 && n < want; ++i) {
        const int id = order[i];
        if (keep[pool.nodes[id].parent]) {
            keep[id] = true;
            ++n;
        }
    }
    VerifyTree tree;
    for (int i = 0; i < pool.size; ++i) {
        if (!keep[i]) continue;
        const auto& node = pool.nodes[i];
        const int row = tree.count++;
        remap[i] = row;
        tree.tokens[row] = node.token;
        tree.depth[row] = node.depth;
        const int parent = tree.parent[row] = i ? remap[node.parent] : -1;
        for (int j = 0; j < node.depth; ++j) tree.ancestry[row][j] = tree.ancestry[parent][j];
        tree.ancestry[row][node.depth] = row;
    }
    return tree;
}

// Draft positions use the same BF16 vocabulary head. Accumulate them
// together so each subgroup loads its weight row once. Keep the SG16 dot order.
template<int Depths = 7>
inline void k_dspark_screen_logits(const uint16_t* weights, const float* hidden, float* logits, int vocab) {
    Q().parallel_for(NR((vocab + 15) / 16, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int tid = it.get_local_id(0), lane = tid % 16;
        const int token = static_cast<int>(it.get_group(0)) * 16 + tid / 16;
        if (token >= vocab) return;
        const uint16_t* row = weights + (size_t)token * INK_D;
        float sum[Depths] = {};
        for (int k = lane * 4; k < INK_D; k += 64) {
            const float w0 = bf16_to_f32(row[k]), w1 = bf16_to_f32(row[k + 1]);
            const float w2 = bf16_to_f32(row[k + 2]), w3 = bf16_to_f32(row[k + 3]);
            for (int i = 0; i < Depths; ++i) {
                const float* x = hidden + i * INK_D + k;
                sum[i] += w0 * x[0] + w1 * x[1] + w2 * x[2] + w3 * x[3];
            }
        }
        for (int i = 0; i < Depths; ++i) {
            const float value = sg_sum(it, sum[i]);
            if (!lane) logits[(size_t)i * vocab + token] = value;
        }
    });
}

// As in the CUDA screen, keep two candidates from each striped 32-lane
// partition, then take 512 from the resulting 2,048. IDs break score ties.
inline void screen_beam_vocab(const float* logits, int vocab, int* short_ids, int* ids, int depths = 7) {
    Q().parallel_for(NR(depths * 1024, 32), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int group = it.get_group(0), row = group / 1024, part = group % 1024;
        int first = vocab, second = vocab;
        const float* scores = logits + (size_t)row * vocab;
        for (int i = part * 32 + it.get_local_id(0); i < vocab; i += 32768) {
            if (first == vocab || scores[i] > scores[first]) {
                second = first;
                first = i;
            } else if (second == vocab || scores[i] > scores[second])
                second = i;
        }
        for (int rank = 0; rank < 2; ++rank) {
            const float score = first < vocab ? scores[first] : -INFINITY;
            const float best = sycl::reduce_over_group(it.get_group(), score, sycl::maximum<float>());
            const int id =
                sycl::reduce_over_group(it.get_group(), score == best ? first : vocab, sycl::minimum<int>());
            if (it.get_local_id(0) == 0) short_ids[group * 2 + rank] = id;
            if (id == first) first = second;
        }
    });
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> keys(2048, h);
        h.parallel_for(NR(depths, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int row = it.get_group(0), lane = it.get_local_id(0);
            const float* scores = logits + (size_t)row * vocab;
            for (int i = lane; i < 2048; i += 256) keys[i] = short_ids[row * 2048 + i];
            sycl::group_barrier(it.get_group());
            for (int span = 2; span <= 2048; span *= 2) {
                for (int gap = span / 2; gap; gap /= 2) {
                    for (int j = lane; j < 1024; j += 256) {
                        const int i = 2 * j - (j & (gap - 1)), k = i + gap;
                        const int a = keys[i], b = keys[k];
                        const float av = a < vocab ? scores[a] : -INFINITY;
                        const float bv = b < vocab ? scores[b] : -INFINITY;
                        if ((av < bv || (av == bv && a > b)) != ((i & span) != 0)) {
                            keys[i] = b;
                            keys[k] = a;
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }
            }
            for (int i = lane; i < screen_size; i += 256) ids[row * screen_size + i] = keys[i];
        });
    });
}

inline void score_beam_candidates(const uint16_t* weights, const float* corr, const float* bias, const int* ids,
                                  const float* base, int beams, float cap, float* scores) {
    Q().parallel_for(NR(beams * screen_size, 32), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int group = it.get_group(0), beam = group / screen_size, id = ids[group % screen_size];
        float sum = 0;
        for (int d = it.get_local_id(0); d < INK_D; d += 32)
            sum += (corr[beam * INK_D + d] + bias[d]) * bf16_to_f32(weights[(size_t)id * INK_D + d]);
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        if (it.get_local_id(0) == 0) scores[group] = cap * sycl::tanh((base[id] + sum) / cap);
    });
}

inline void expand_beams(BeamPool* pool, const int* ids, const float* scores, BeamPick* picks, int beams,
                         int depth, const float* state, float* next_state) {
    Q().parallel_for(NR(beams, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int b = it.get_group(0), lane = it.get_local_id(0);
        float x = scores[b * screen_size + lane], y = scores[b * screen_size + 256 + lane];
        const auto g = it.get_group();
        const float maxval = sycl::reduce_over_group(g, sycl::fmax(x, y), sycl::maximum<float>());
        const float z =
            sycl::reduce_over_group(g, sycl::exp(x - maxval) + sycl::exp(y - maxval), sycl::plus<float>());
        const float logz = maxval + sycl::log(z);
        for (int r = 0; r < beam_width; ++r) {
            const float best = sycl::reduce_over_group(g, sycl::fmax(x, y), sycl::maximum<float>());
            const int id = sycl::reduce_over_group(
                g, sycl::min(x == best ? ids[lane] : INT_MAX, y == best ? ids[256 + lane] : INT_MAX),
                sycl::minimum<int>());
            if (lane == 0) picks[b * beam_width + r] = {id, b, pool->nodes[pool->active[b]].logp + (best - logz)};
            if (ids[lane] == id) x = -INFINITY;
            if (ids[256 + lane] == id) y = -INFINITY;
        }
    });
    Q().single_task([=] {
        int parents[beam_width];
        for (int b = 0; b < beams; ++b) parents[b] = pool->active[b];
        for (int i = 1; i < beams * beam_width; ++i) {
            const BeamPick v = picks[i];
            int j = i;
            while (j && picks[j - 1].logp < v.logp) {
                picks[j] = picks[j - 1];
                --j;
            }
            picks[j] = v;
        }
        const int n = sycl::min(beams * beam_width, depth == 1 ? 4 : 8);
        for (int i = 0; i < n; ++i) {
            const auto p = picks[i];
            const int node = pool->size++;
            pool->nodes[node] = {p.token, parents[p.beam], depth, p.logp};
            if (i < beam_width) {
                pool->active[i] = node;
                pool->from[i] = p.beam;
            }
        }
    });
    Q().parallel_for(sycl::range<1>(beam_width * DSP_RANK), [=](sycl::id<1> idx) {
        const int i = idx[0];
        next_state[i] = state[pool->from[i / DSP_RANK] * DSP_RANK + i % DSP_RANK];
    });
}

struct TreeConvHistory {
    float *live, *snapshot;
    const float* staged;
    int width;
};

// Each entry is one ShortConv in one visit. Copying the snapshot in the
// commit kernel also permits rollback to the pre-verification prefix.
inline void restore_tree_history(const TreeConvHistory* histories, int count, int row, bool snapshot) {
    Q().parallel_for(sycl::range<2>(count, 3 * INK_D), [=](sycl::id<2> idx) {
        const auto hist = histories[idx[0]];
        const int d = idx[1], size = 3 * hist.width;
        if (d >= size) return;
        if (snapshot) hist.snapshot[d] = hist.live[d];
        hist.live[d] = row >= 0 ? hist.staged[row * size + d] : hist.snapshot[d];
    });
}

struct TreeKvCache {
    half* k;
    half* v;
    int stride;
};
struct AcceptedPath {
    int rows[8];
    int length;
};

// Slot zero only. Each work-item owns one channel across all accepted rows.
// Parent rows precede children, so ascending compaction cannot clobber a source.
inline void commit_tree_kv(TreeKvCache* caches, int layers, int root, AcceptedPath path) {
    Q().parallel_for(sycl::range<2>(layers, 1024), [=](sycl::id<2> idx) {
        const auto cache = caches[idx[0]];
        for (int j = 0; j < path.length; ++j) {
            if (path.rows[j] == j) continue;
            const size_t src = (size_t)((root + path.rows[j]) % cache.stride) * 1024 + idx[1];
            const size_t dst = (size_t)((root + j) % cache.stride) * 1024 + idx[1];
            cache.k[dst] = cache.k[src];
            cache.v[dst] = cache.v[src];
        }
    });
}
} // namespace ink_sycl
