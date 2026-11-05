// cann/tests/layer_parity.cpp - one real block on the NPU, built from the model's own weights.
//
// The shared expert of blk.0 is a plain SwiGLU MLP, so it is the smallest piece of a real layer
// the port can run end to end:
//
//   g = W_gate x,  u = W_up x,  a = silu(g) * u,  y = W_down a
//
// W_gate and W_up are Q8_0, W_down is Q8_0; x is a random activation. Everything is compared
// against a float CPU reference that sees the same fp16-rounded operands.
#include <acl/acl.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/cann/device.hpp"
#include "strata/cann/gguf.hpp"
#include "strata/cann/ops.hpp"
#include "strata/cann/tensor_load.hpp"

using namespace strata::cann;

namespace {

const char * kDefaultPath = "/data/strata-qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf";

std::vector<float> round_f16(const std::vector<float> & v) {
    std::vector<uint16_t> h(v.size());
    f32_to_f16(v.data(), h.data(), v.size());
    std::vector<float> out(v.size());
    f16_to_f32(h.data(), out.data(), out.size());
    return out;
}

// One GEMV: out[m] = W[m][k] * x[k]
void gemv(Stream & stream, const std::vector<float> & w, int64_t m, int64_t k, const std::vector<float> & x,
          std::vector<float> & out) {
    Buffer bw(w.size() * 2), bx(x.size() * 2), bo((size_t) m * 2);
    std::vector<uint16_t> hw(w.size()), hx(x.size());
    f32_to_f16(w.data(), hw.data(), hw.size());
    f32_to_f16(x.data(), hx.data(), hx.size());
    bw.copy_from_host(hw.data(), hw.size() * 2);
    bx.copy_from_host(hx.data(), hx.size() * 2);
    Tensor tw{ bw.data(), DType::F16, k, m };
    Tensor tx{ bx.data(), DType::F16, 1, k };
    Tensor to{ bo.data(), DType::F16, 1, m };
    matmul(stream, tw, tx, to);
    std::vector<uint16_t> ho((size_t) m);
    bo.copy_to_host(ho.data(), ho.size() * 2);
    stream.synchronize();
    out.assign((size_t) m, 0.0f);
    f16_to_f32(ho.data(), out.data(), out.size());
}

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string path = argc > 1 ? argv[1] : kDefaultPath;
    try {
        const Gguf g(path);
        const GgufTensor * wg = g.find("blk.0.ffn_gate_shexp.weight");
        const GgufTensor * wu = g.find("blk.0.ffn_up_shexp.weight");
        const GgufTensor * wd = g.find("blk.0.ffn_down_shexp.weight");
        if (!wg || !wu || !wd) {
            printf("shared-expert tensors not found\n");
            return 2;
        }
        const int64_t hidden = (int64_t) wg->dims[0]; // 2048
        const int64_t inter = (int64_t) wg->dims[1];  // 512

        std::vector<float> Wg = round_f16(load_tensor(g, *wg));
        std::vector<float> Wu = round_f16(load_tensor(g, *wu));
        std::vector<float> Wd = round_f16(load_tensor(g, *wd));

        std::mt19937 rng(17);
        std::normal_distribution<float> dist(0.0f, 0.5f);
        std::vector<float> x((size_t) hidden);
        for (auto & v : x) {
            v = dist(rng);
        }
        x = round_f16(x);

        // CPU reference
        std::vector<float> g_ref((size_t) inter), u_ref((size_t) inter), a_ref((size_t) inter);
        for (int64_t i = 0; i < inter; ++i) {
            double acc = 0.0;
            for (int64_t j = 0; j < hidden; ++j) {
                acc += (double) Wg[(size_t) i * hidden + j] * x[(size_t) j];
            }
            g_ref[(size_t) i] = (float) acc;
            acc = 0.0;
            for (int64_t j = 0; j < hidden; ++j) {
                acc += (double) Wu[(size_t) i * hidden + j] * x[(size_t) j];
            }
            u_ref[(size_t) i] = (float) acc;
            a_ref[(size_t) i] = (g_ref[(size_t) i] / (1.0f + std::exp(-g_ref[(size_t) i]))) * u_ref[(size_t) i];
        }
        std::vector<float> y_ref((size_t) hidden, 0.0f);
        for (int64_t i = 0; i < hidden; ++i) {
            double acc = 0.0;
            for (int64_t j = 0; j < inter; ++j) {
                acc += (double) Wd[(size_t) i * inter + j] * a_ref[(size_t) j];
            }
            y_ref[(size_t) i] = (float) acc;
        }

        // NPU: the same four steps, as operations
        Device device(0);
        Stream stream(device);
        std::vector<float> g_dev, u_dev;
        gemv(stream, Wg, inter, hidden, x, g_dev);
        gemv(stream, Wu, inter, hidden, x, u_dev);

        Buffer bg(g_dev.size() * 2), bu(u_dev.size() * 2), bs(g_dev.size() * 2), ba(g_dev.size() * 2);
        std::vector<uint16_t> hg(g_dev.size()), hu(u_dev.size());
        f32_to_f16(g_dev.data(), hg.data(), hg.size());
        f32_to_f16(u_dev.data(), hu.data(), hu.size());
        bg.copy_from_host(hg.data(), hg.size() * 2);
        bu.copy_from_host(hu.data(), hu.size() * 2);
        Tensor tg{ bg.data(), DType::F16, 1, inter };
        Tensor tu{ bu.data(), DType::F16, 1, inter };
        Tensor ts{ bs.data(), DType::F16, 1, inter };
        Tensor ta{ ba.data(), DType::F16, 1, inter };
        silu(stream, tg, ts);
        mul(stream, ts, tu, ta);

        std::vector<uint16_t> ha((size_t) inter);
        ba.copy_to_host(ha.data(), ha.size() * 2);
        stream.synchronize();
        std::vector<float> a_dev((size_t) inter);
        f16_to_f32(ha.data(), a_dev.data(), a_dev.size());

        std::vector<float> y_dev;
        gemv(stream, Wd, hidden, inter, a_dev, y_dev);

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

        printf("block      : blk.0 shared expert (SwiGLU)\n");
        printf("weights    : gate %s  up %s  down %s\n", wg->type == 8 ? "Q8_0" : "?", wu->type == 8 ? "Q8_0" : "?",
               wd->type == 8 ? "Q8_0" : "?");
        printf("shapes     : x[%lld] -> gate/up[%lld] -> down[%lld]\n\n", (long long) hidden, (long long) inter,
               (long long) hidden);

        const double d_g = max_abs(g_dev, g_ref);
        const double d_u = max_abs(u_dev, u_ref);
        const double d_a = max_abs(a_dev, a_ref);
        const double d_y = max_abs(y_dev, y_ref);
        const double r_y = rel(y_dev, y_ref);

        printf("%-22s %12s %12s  %s\n", "stage", "max|diff|", "rel L2", "result");
        printf("%-22s %12.6f %12.6f  %s\n", "gate  = W_gate x", d_g, rel(g_dev, g_ref), d_g <= 0.2 ? "PASS" : "FAIL");
        printf("%-22s %12.6f %12.6f  %s\n", "up    = W_up x", d_u, rel(u_dev, u_ref), d_u <= 0.2 ? "PASS" : "FAIL");
        printf("%-22s %12.6f %12.6f  %s\n", "act   = silu(g) * u", d_a, rel(a_dev, a_ref), d_a <= 0.2 ? "PASS" : "FAIL");
        printf("%-22s %12.6f %12.6f  %s\n", "y     = W_down act", d_y, r_y, (d_y <= 0.5 && r_y <= 0.02) ? "PASS" : "FAIL");

        const bool ok = d_g <= 0.2 && d_u <= 0.2 && d_a <= 0.2 && d_y <= 0.5 && r_y <= 0.02;
        shutdown();
        if (!ok) {
            printf("\nblock differs from the CPU reference\n");
            return 1;
        }
        printf("\nreal block ran on the NPU from the model's own weights\n");
        return 0;
    } catch (const std::exception & e) {
        printf("FAILED: %s\n", e.what());
        return 1;
    }
}
