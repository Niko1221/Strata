#pragma once
#include "ink_sy_ops.h"
#include <array>

namespace ink_sycl {

// Store eight consecutive steps of one logical SG16 lane together. This keeps
// the FP32 dot and reduction order of the standard 64 x 4608 projection.
constexpr int ctrl_rank = 64;
constexpr int ctrl_k = 2 * INK_D;
inline int ctrl_packed_index(int k) {
    return k / 128 * 128 + k % 16 * 8 + k % 128 / 16;
}

struct InkCtrlWeights {
    const uint16_t* down;
    const uint16_t* gate_pair;
    const float* norm;
};
struct InkCtrlSource {
    const float* down;
    const float* gain;
    const float* forget;
    const float* norm;
};

// Each FP32 payload must survive conversion by a bit shift alone. If any
// payload has low bits set, the runner keeps the original controller path.
inline void k_ctrl_pack_weights(InkCtrlSource src, uint16_t* packed, uint32_t* inexact) {
    constexpr int matrix = ctrl_rank * ctrl_k;
    Q().parallel_for(NR(matrix / 128, 128), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int i = it.get_global_id(0);
        const uint32_t w = sycl::bit_cast<uint32_t>(src.down[i]);
        packed[i / ctrl_k * ctrl_k + ctrl_packed_index(i % ctrl_k)] = w >> 16;
        uint32_t bits = w;
        if (i < INK_D * ctrl_rank) {
            const int d = i / ctrl_rank, r = i % ctrl_rank;
            const int offset = ((d / 16 * (ctrl_rank / 8) + r / 8) * 16 + d % 16) * 16 + r % 8;
            const uint32_t g = sycl::bit_cast<uint32_t>(src.gain[i]);
            const uint32_t f = sycl::bit_cast<uint32_t>(src.forget[i]);
            packed[matrix + offset] = g >> 16;
            packed[matrix + offset + 8] = f >> 16;
            bits |= g | f;
        }
        if (sycl::any_of_group(it.get_group(), (bits & 0xffffu) != 0) && it.get_local_id(0) == 0) {
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(*inexact).fetch_or(1u);
        }
    });
}

template<int Count> struct InkCtrlSet { InkCtrlWeights weights[Count]; };

template<int Count, bool Tree = false>
inline void k_ctrl_decode_batch(InkCtrlSet<Count> weights, float* cur, const float* msg,
                                bool shared_cur, float* scratch, float* gates, int nodes = 1) {
    const int batch = Tree ? nodes : 1;
    // A workgroup normalizes both inputs, then eight SG16s each compute one gate.
    constexpr int rows = 8;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> input(ctrl_k, h);
        sycl::local_accessor<sycl::float2, 1> partial(16, h);
        h.parallel_for(NR(batch * Count * ctrl_rank / rows, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int g = it.get_group(0), vector = g / (ctrl_rank / rows), tid = it.get_local_id(0);
            const int lane = tid % 16, sg = tid / 16;
            const int bank = vector / batch;
            const float* c = cur + (shared_cur ? vector % batch : vector) * INK_D;
            const float* m = msg + vector * INK_D;
            float cv[INK_D / 256], mv[INK_D / 256], cs = 0.0f, ms = 0.0f;
#pragma unroll
            for (int k = 0; k < INK_D / 256; ++k) {
                cv[k] = c[tid + 256 * k]; mv[k] = m[tid + 256 * k];
                cs += cv[k] * cv[k]; ms += mv[k] * mv[k];
            }
            cs = sg_sum(it, cs); ms = sg_sum(it, ms);
            if (lane == 0) partial[sg] = sycl::float2(cs, ms);
            sycl::group_barrier(it.get_group());
            const sycl::float2 sum = partial[lane];
            const float ic = sycl::rsqrt(sg_sum(it, sum.x()) / float(INK_D) + INK_RMS_EPS);
            const float im = sycl::rsqrt(sg_sum(it, sum.y()) / float(INK_D) + INK_RMS_EPS);
#pragma unroll
            for (int k = 0; k < INK_D / 256; ++k) {
                const int offset = ctrl_packed_index(tid + 256 * k);
                input[offset] = cv[k] * ic;
                input[INK_D + offset] = mv[k] * im;
            }
            sycl::group_barrier(it.get_group());
            if (sg >= rows) return;
            const int row = g % (ctrl_rank / rows) * rows + sg;
            const uint16_t* w = weights.weights[bank].down + row * ctrl_k;
            float acc = 0.0f;
#pragma unroll 3
            for (int k = 0; k < ctrl_k; k += 128) {
                const int offset = k + lane * 8;
                const auto v = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(w + offset);
#pragma unroll
                for (int j = 0; j < 8; ++j) acc += bf16_to_f32(v[j]) * input[offset + j];
            }
            const float dot = sg_sum(it, acc);
            if (lane == 0) gates[vector * ctrl_rank + row] = dot;
        });
    });

    // All nine values in each original RMS accumulator (d, d+256, ...) now
    // belong to one group. The following launch finishes the same SG16 tree.
    constexpr int stripes = INK_D / 256;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> values(stripes * 16, h);
        h.parallel_for(NR(batch * Count * 16, stripes * 16), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int group = it.get_group(0), vector = group / 16, bank = vector / batch, tid = it.get_local_id(0);
            const int sg = tid / 16, lane = tid % 16;
            const int d = sg * 256 + group % 16 * 16 + lane, i = vector * INK_D + d;
            const uint16_t* pair = weights.weights[bank].gate_pair;
            float gain = 0.0f, forget = 0.0f;
            for (int r = 0; r < ctrl_rank; r += 8) {
                const int off = ((d / 16 * (ctrl_rank / 8) + r / 8) * 16 + d % 16) * 16;
                const auto g = *reinterpret_cast<const sycl::vec<float, 8>*>(gates + vector * ctrl_rank + r);
                const auto a = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(pair + off);
                const auto b = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(pair + off + 8);
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    gain += bf16_to_f32(a[t]) * g[t];
                    forget += bf16_to_f32(b[t]) * g[t];
                }
            }
            const float v = (1.0f - sycl::tanh(sycl::clamp(forget, -15.0f, 15.0f))) * cur[shared_cur ? (vector % batch) * INK_D + d : i]
                          + sycl::tanh(sycl::clamp(gain, -15.0f, 15.0f)) * msg[i];
            scratch[i] = v; values[tid] = v;
            sycl::group_barrier(it.get_group());
            if (sg == 0) {
                float ss = 0.0f;
#pragma unroll
                for (int t = 0; t < stripes; ++t) {
                    const float x = values[t * 16 + lane];
                    ss += x * x;
                }
                const float sum = sg_sum(it, ss);
                if (lane == 0) scratch[batch * Count * INK_D + group] = sum;
            }
        });
    });
    Q().parallel_for(NR(batch * Count * INK_D / 64, 64), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int i = it.get_global_id(0), vector = i / INK_D, bank = vector / batch, lane = it.get_local_id(0) % 16;
        const float ss = sg_sum(it, scratch[batch * Count * INK_D + vector * 16 + lane]);
        const float inv = sycl::rsqrt(ss / float(INK_D) + INK_RMS_EPS);
        cur[i] = scratch[i] * inv * weights.weights[bank].norm[i % INK_D];
    });
}

struct InkControllerCache {
    struct Layer { InkCtrlSet<3> attention; InkCtrlSet<1> mlp; };
    std::vector<Layer> layers;
    uint16_t* storage = nullptr;

    void release() {
        dfree(storage);
        storage = nullptr;
        layers.clear();
    }
    void prepare(const InkSyModel& model) {
        const auto& m = model.man;
        const auto& c = m.c;
        if (!ink_bmg_option("INK_SYCL_BMG_DENSE") || c.hidden != INK_D || c.gate_rank != ctrl_rank ||
            model.ep2_active || model.n_host_experts != 0 || c.num_passes <= 1 || c.core_start < 1 ||
            c.core_end <= c.core_start || c.core_end > c.n_phys) return;
        std::vector<std::array<InkCtrlSource, 4>> sources(c.core_end - c.core_start);
        auto matrix = [&](const std::string& name, int rows, int cols) -> const float* {
            auto it = m.T.find(name);
            if (it == m.T.end() || it->second.fmt != FMT_F32 || it->second.rows() != rows ||
                it->second.cols() != cols || it->second.bytes < size_t(rows) * cols * sizeof(float)) return nullptr;
            return reinterpret_cast<const float*>(it->second.dev_ptr);
        };
        const char* names[] = {"attn", "k", "v", "mlp"};
        const char* norms[] = {"q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight", "mlp_norm.weight"};
        for (int ip = c.core_start; ip < c.core_end; ++ip) {
            auto& layer = sources[ip - c.core_start];
            const std::string p = "model.llm.layers." + std::to_string(ip) + ".";
            for (int j = 0; j < 4; ++j) {
                const std::string ctrl = p + "controllers." + names[j] + ".";
                auto& w = layer[j];
                w = {matrix(ctrl + "down_weight", ctrl_rank, ctrl_k),
                     matrix(ctrl + "gain_weight", INK_D, ctrl_rank), matrix(ctrl + "forget_weight", INK_D, ctrl_rank),
                     matrix(p + (j < 3 && c.is_swa[ip] ? "attn_norm.weight" : norms[j]), 1, INK_D)};
                if (!w.down || !w.gain || !w.forget || !w.norm) return;
            }
        }
        // Each bank stores its down matrix followed by interleaved gain/forget.
        constexpr size_t bank_size = 2 * ctrl_rank * ctrl_k;
        const size_t elements = sources.size() * 4 * bank_size;
        const size_t bytes = elements * sizeof(uint16_t) + sizeof(uint32_t);
        try { storage = static_cast<uint16_t*>(sycl::malloc_device(bytes, Q())); }
        catch (const sycl::exception& e) {
            if (e.code() != sycl::errc::memory_allocation) throw;
        }
        if (!storage) return;
        auto* invalid = reinterpret_cast<uint32_t*>(storage + elements);
        Q().memset(invalid, 0, sizeof(*invalid));
        std::vector<Layer> selected(sources.size());
        uint16_t* next = storage;
        for (size_t l = 0; l < sources.size(); ++l) {
            for (int j = 0; j < 4; ++j) {
                auto& w = j == 3 ? selected[l].mlp.weights[0] : selected[l].attention.weights[j];
                const auto src = sources[l][j];
                k_ctrl_pack_weights(src, next, invalid);
                w = {next, next + ctrl_rank * ctrl_k, src.norm};
                next += bank_size;
            }
        }
        uint32_t inexact = 0;
        Q().memcpy(&inexact, invalid, sizeof(inexact)).wait_and_throw();
        if (inexact) { release(); return; }
        dev_bytes_allocated() += bytes;
        layers = std::move(selected);
    }
};
} // namespace ink_sycl
