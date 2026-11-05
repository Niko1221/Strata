// cann/tests/moe_parity.cpp - the whole FFN of blk.0 on the NPU, from the model's own weights.
//
// This is the block qwen35moe runs in every layer:
//
//   logits = W_router x                      (256 experts)
//   probs  = softmax(logits)
//   top8   = argsort_top_k(probs, 8), weights normalized to sum 1
//   moe    = sum_i w_i * W_down_i ( silu(W_gate_i x) * (W_up_i x) )
//   shared = W_down_s ( silu(W_gate_s x) * (W_up_s x) ) * sigmoid(W_gate_inp_s x)
//   y      = moe + shared
//
// The expert slabs are Q4_K (gate, up) and Q5_K (down), read per expert out of the 256-expert
// tensors; the router and the shared-expert gate are F32; the shared expert itself is Q8_0.
#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/cann/device.hpp"
#include "strata/cann/gguf.hpp"
#include "strata/cann/ops.hpp"
#include "strata/cann/quants.hpp"
#include "strata/cann/tensor_load.hpp"

using namespace strata::cann;

namespace {

const char * kDefaultPath = "/data/strata-qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf";
constexpr int kTopK = 8;

std::vector<float> round_f16(const std::vector<float> & v) {
    std::vector<uint16_t> h(v.size());
    f32_to_f16(v.data(), h.data(), v.size());
    std::vector<float> out(v.size());
    f16_to_f32(h.data(), out.data(), out.size());
    return out;
}

// One expert's slab out of a [cols, rows, n_expert] tensor.
std::vector<float> expert_slab(const Gguf & g, const GgufTensor & t, int expert, int64_t rows, int64_t cols) {
    const uint64_t n = (uint64_t) rows * cols;
    const int block_bytes = quant_type_size((int) t.type);
    const int block_elems = quant_block_elements((int) t.type);
    if (block_bytes == 0 || n % (uint64_t) block_elems != 0) {
        throw std::runtime_error("moe: unsupported expert tensor type");
    }
    const uint64_t bytes = n / (uint64_t) block_elems * (uint64_t) block_bytes;
    const std::vector<uint8_t> raw = g.tensor_bytes(t, (uint64_t) expert * bytes, bytes);
    std::vector<float> out((size_t) n);
    if (!dequantize((int) t.type, raw.data(), out.data(), (int64_t) n)) {
        throw std::runtime_error("moe: dequantize failed");
    }
    return out;
}

// W[m][k] * x[k] on the device, result left in `out` (m values).
void gemv(Stream & stream, const std::vector<float> & w, int64_t m, int64_t k, const Tensor & x, Buffer & out) {
    Buffer bw(w.size() * 2);
    std::vector<uint16_t> hw(w.size());
    f32_to_f16(w.data(), hw.data(), hw.size());
    bw.copy_from_host(hw.data(), hw.size() * 2);
    Tensor tw{ bw.data(), DType::F16, k, m };
    Tensor to{ out.data(), DType::F16, 1, m };
    matmul(stream, tw, x, to);
}

std::vector<float> to_host(Stream & stream, const Buffer & b, int64_t n) {
    std::vector<uint16_t> h((size_t) n);
    b.copy_to_host(h.data(), h.size() * 2);
    stream.synchronize();
    std::vector<float> out((size_t) n);
    f16_to_f32(h.data(), out.data(), out.size());
    return out;
}

void upload(const std::vector<float> & v, Buffer & b) {
    std::vector<uint16_t> h(v.size());
    f32_to_f16(v.data(), h.data(), h.size());
    b.copy_from_host(h.data(), h.size() * 2);
}

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string path = argc > 1 ? argv[1] : kDefaultPath;
    try {
        const Gguf g(path);
        const GgufTensor * router = g.find("blk.0.ffn_gate_inp.weight");
        const GgufTensor * gate_exps = g.find("blk.0.ffn_gate_exps.weight");
        const GgufTensor * up_exps = g.find("blk.0.ffn_up_exps.weight");
        const GgufTensor * down_exps = g.find("blk.0.ffn_down_exps.weight");
        const GgufTensor * gate_shexp = g.find("blk.0.ffn_gate_shexp.weight");
        const GgufTensor * up_shexp = g.find("blk.0.ffn_up_shexp.weight");
        const GgufTensor * down_shexp = g.find("blk.0.ffn_down_shexp.weight");
        const GgufTensor * gate_inp_shexp = g.find("blk.0.ffn_gate_inp_shexp.weight");
        if (!router || !gate_exps || !up_exps || !down_exps || !gate_shexp || !up_shexp || !down_shexp ||
            !gate_inp_shexp) {
            printf("MoE tensors not found\n");
            return 2;
        }

        const int64_t hidden = (int64_t) router->dims[0];   // 2048
        const int64_t n_expert = (int64_t) router->dims[1]; // 256
        const int64_t inter = (int64_t) gate_exps->dims[1]; // 512

        const std::vector<float> Wr = round_f16(load_tensor(g, *router));
        const std::vector<float> Wgs = round_f16(load_tensor(g, *gate_shexp));
        const std::vector<float> Wus = round_f16(load_tensor(g, *up_shexp));
        const std::vector<float> Wds = round_f16(load_tensor(g, *down_shexp));
        const std::vector<float> Wgi = round_f16(load_tensor(g, *gate_inp_shexp));

        std::mt19937 rng(23);
        std::normal_distribution<float> dist(0.0f, 0.5f);
        std::vector<float> x((size_t) hidden);
        for (auto & v : x) {
            v = dist(rng);
        }
        x = round_f16(x);

        // ---- CPU reference ----
        std::vector<float> logits((size_t) n_expert, 0.0f);
        for (int64_t e = 0; e < n_expert; ++e) {
            double acc = 0.0;
            for (int64_t j = 0; j < hidden; ++j) {
                acc += (double) Wr[(size_t) e * hidden + j] * x[(size_t) j];
            }
            logits[(size_t) e] = (float) acc;
        }
        double mx = logits[0];
        for (float v : logits) {
            mx = std::max(mx, (double) v);
        }
        std::vector<float> probs((size_t) n_expert);
        double sum = 0.0;
        for (int64_t e = 0; e < n_expert; ++e) {
            probs[(size_t) e] = (float) std::exp((double) logits[(size_t) e] - mx);
            sum += probs[(size_t) e];
        }
        for (auto & v : probs) {
            v = (float) (v / sum);
        }
        std::vector<int> order((size_t) n_expert);
        for (int i = 0; i < n_expert; ++i) {
            order[(size_t) i] = i;
        }
        std::partial_sort(order.begin(), order.begin() + kTopK, order.end(),
                          [&](int a, int b) { return probs[(size_t) a] > probs[(size_t) b]; });
        std::vector<int> sel(order.begin(), order.begin() + kTopK);
        std::vector<float> w((size_t) kTopK);
        double wsum = 0.0;
        for (int i = 0; i < kTopK; ++i) {
            w[(size_t) i] = probs[(size_t) sel[(size_t) i]];
            wsum += w[(size_t) i];
        }
        wsum = std::max(wsum, 6.103515625e-5);
        for (auto & v : w) {
            v = (float) (v / wsum);
        }

        std::vector<float> y_ref((size_t) hidden, 0.0f);
        for (int i = 0; i < kTopK; ++i) {
            const int e = sel[(size_t) i];
            const std::vector<float> Wg = round_f16(expert_slab(g, *gate_exps, e, inter, hidden));
            const std::vector<float> Wu = round_f16(expert_slab(g, *up_exps, e, inter, hidden));
            const std::vector<float> Wd = round_f16(expert_slab(g, *down_exps, e, hidden, inter));
            std::vector<float> act((size_t) inter);
            for (int64_t r = 0; r < inter; ++r) {
                double a1 = 0.0, a2 = 0.0;
                for (int64_t j = 0; j < hidden; ++j) {
                    a1 += (double) Wg[(size_t) r * hidden + j] * x[(size_t) j];
                    a2 += (double) Wu[(size_t) r * hidden + j] * x[(size_t) j];
                }
                const float gv = (float) a1;
                act[(size_t) r] = (gv / (1.0f + std::exp(-gv))) * (float) a2;
            }
            for (int64_t r = 0; r < hidden; ++r) {
                double acc = 0.0;
                for (int64_t j = 0; j < inter; ++j) {
                    acc += (double) Wd[(size_t) r * inter + j] * act[(size_t) j];
                }
                y_ref[(size_t) r] += w[(size_t) i] * (float) acc;
            }
        }
        // shared expert
        {
            std::vector<float> act((size_t) inter);
            for (int64_t r = 0; r < inter; ++r) {
                double a1 = 0.0, a2 = 0.0;
                for (int64_t j = 0; j < hidden; ++j) {
                    a1 += (double) Wgs[(size_t) r * hidden + j] * x[(size_t) j];
                    a2 += (double) Wus[(size_t) r * hidden + j] * x[(size_t) j];
                }
                const float gv = (float) a1;
                act[(size_t) r] = (gv / (1.0f + std::exp(-gv))) * (float) a2;
            }
            double gate_dot = 0.0;
            for (int64_t j = 0; j < hidden; ++j) {
                gate_dot += (double) Wgi[(size_t) j] * x[(size_t) j];
            }
            const float sg = 1.0f / (1.0f + std::exp((float) -gate_dot));
            for (int64_t r = 0; r < hidden; ++r) {
                double acc = 0.0;
                for (int64_t j = 0; j < inter; ++j) {
                    acc += (double) Wds[(size_t) r * inter + j] * act[(size_t) j];
                }
                y_ref[(size_t) r] += sg * (float) acc;
            }
        }

        // ---- NPU ----
        Device device(0);
        Stream stream(device);
        Buffer bx((size_t) hidden * 2);
        upload(x, bx);
        Tensor tx{ bx.data(), DType::F16, 1, hidden };

        Buffer b_logits((size_t) n_expert * 2);
        gemv(stream, Wr, n_expert, hidden, tx, b_logits);
        std::vector<float> logits_dev = to_host(stream, b_logits, n_expert);

        Buffer acc_a((size_t) hidden * 2), acc_b((size_t) hidden * 2), tmp((size_t) hidden * 2);
        Buffer b_act((size_t) inter * 2), b_act2((size_t) inter * 2), b_silu((size_t) inter * 2);
        Buffer b_gate((size_t) inter * 2), b_up((size_t) inter * 2), b_y((size_t) hidden * 2);
        Tensor t_tmp{ tmp.data(), DType::F16, 1, hidden };
        Tensor t_acc_a{ acc_a.data(), DType::F16, 1, hidden };
        Tensor t_acc_b{ acc_b.data(), DType::F16, 1, hidden };
        std::vector<float> zeros((size_t) hidden, 0.0f);
        upload(zeros, tmp);
        muls(stream, t_tmp, 1.0f, t_acc_a); // acc = 0
        Buffer * acc = &acc_a;
        Buffer * other = &acc_b;

        Tensor t_act{ b_act.data(), DType::F16, 1, inter };
        Tensor t_act2{ b_act2.data(), DType::F16, 1, inter };
        Tensor t_silu{ b_silu.data(), DType::F16, 1, inter };

        for (int i = 0; i < kTopK; ++i) {
            const int e = sel[(size_t) i];
            const std::vector<float> Wg = round_f16(expert_slab(g, *gate_exps, e, inter, hidden));
            const std::vector<float> Wu = round_f16(expert_slab(g, *up_exps, e, inter, hidden));
            const std::vector<float> Wd = round_f16(expert_slab(g, *down_exps, e, hidden, inter));

            gemv(stream, Wg, inter, hidden, tx, b_gate);
            gemv(stream, Wu, inter, hidden, tx, b_up);
            Tensor g_t{ b_gate.data(), DType::F16, 1, inter };
            Tensor u_t{ b_up.data(), DType::F16, 1, inter };
            silu(stream, g_t, t_silu);
            mul(stream, t_silu, u_t, t_act);

            gemv(stream, Wd, hidden, inter, t_act, b_y);
            Tensor y_t{ b_y.data(), DType::F16, 1, hidden };
            muls(stream, y_t, w[(size_t) i], t_tmp);
            Tensor acc_t{ acc->data(), DType::F16, 1, hidden };
            Tensor oth_t{ other->data(), DType::F16, 1, hidden };
            add(stream, acc_t, t_tmp, oth_t);
            std::swap(acc, other);
        }

        // shared expert
        {
            gemv(stream, Wgs, inter, hidden, tx, b_gate);
            gemv(stream, Wus, inter, hidden, tx, b_up);
            Tensor g_t{ b_gate.data(), DType::F16, 1, inter };
            Tensor u_t{ b_up.data(), DType::F16, 1, inter };
            silu(stream, g_t, t_silu);
            mul(stream, t_silu, u_t, t_act);
            gemv(stream, Wds, hidden, inter, t_act, b_y);

            Buffer b_sg(2);
            gemv(stream, Wgi, 1, hidden, tx, b_sg);
            const float gate_dot = to_host(stream, b_sg, 1)[0];
            const float sg = 1.0f / (1.0f + std::exp(-gate_dot));

            Tensor y_t{ b_y.data(), DType::F16, 1, hidden };
            muls(stream, y_t, sg, t_tmp);
            Tensor acc_t{ acc->data(), DType::F16, 1, hidden };
            Tensor oth_t{ other->data(), DType::F16, 1, hidden };
            add(stream, acc_t, t_tmp, oth_t);
            std::swap(acc, other);
        }

        const std::vector<float> y_dev = to_host(stream, *acc, hidden);

        auto max_abs = [](const std::vector<float> & a, const std::vector<float> & b) {
            double m = 0.0;
            for (size_t i = 0; i < a.size(); ++i) {
                m = std::max(m, std::fabs((double) a[i] - b[i]));
            }
            return m;
        };
        auto rel = [](const std::vector<float> & a, const std::vector<float> & b) {
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < a.size(); ++i) {
                num += (double) (a[i] - b[i]) * (a[i] - b[i]);
                den += (double) b[i] * b[i];
            }
            return den > 0.0 ? std::sqrt(num / den) : 0.0;
        };

        printf("block   : blk.0 FFN = 8 of %lld experts + shared expert\n", (long long) n_expert);
        printf("shapes  : x[%lld] -> %lld experts [%lld] -> %lld\n\n", (long long) hidden, (long long) kTopK,
               (long long) inter, (long long) hidden);
        printf("routing :");
        for (int i = 0; i < kTopK; ++i) {
            printf(" %d(%.4f)", sel[(size_t) i], w[(size_t) i]);
        }
        printf("\n\n");

        const double d_logits = max_abs(logits_dev, logits);
        const double d_y = max_abs(y_dev, y_ref);
        const double r_y = rel(y_dev, y_ref);

        printf("%-22s %12s %12s  %s\n", "stage", "max|diff|", "rel L2", "result");
        printf("%-22s %12.6f %12.6f  %s\n", "router logits", d_logits, rel(logits_dev, logits),
               d_logits <= 0.05 ? "PASS" : "FAIL");
        printf("%-22s %12.6f %12.6f  %s\n", "ffn out", d_y, r_y, (d_y <= 1.0 && r_y <= 0.02) ? "PASS" : "FAIL");

        const bool ok = d_logits <= 0.05 && d_y <= 1.0 && r_y <= 0.02;
        shutdown();
        if (!ok) {
            printf("\nMoE block differs from the CPU reference\n");
            return 1;
        }
        printf("\nthe full FFN of blk.0 ran on the NPU\n");
        return 0;
    } catch (const std::exception & e) {
        printf("FAILED: %s\n", e.what());
        return 1;
    }
}
