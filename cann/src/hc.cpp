// cann/src/hc.cpp - hyper-connections, the piece that replaces a layer's residual and norm.
#include "strata/cann/hc.hpp"

#include <cmath>
#include <stdexcept>

#include "strata/cann/device.hpp"
#include "strata/cann/quants.hpp"

namespace strata::cann {

namespace {

int64_t hidden_of(const std::vector<float> & w_norm) {
    if (w_norm.size() % kHc != 0) {
        throw std::runtime_error("hc: norm weight is not a multiple of the stream count");
    }
    return (int64_t) w_norm.size() / kHc;
}

// the buffers hold f16, so the host side has to be converted, not reinterpreted
void upload_floats(const std::vector<float> & v, Buffer & b) {
    std::vector<uint16_t> h(v.size());
    f32_to_f16(v.data(), h.data(), v.size());
    b.copy_from_host(h.data(), h.size() * 2);
}

// the buffer hands back whatever the kernel wrote, which for a matmul output is f16
std::vector<float> read_back_f16(Buffer & b, int64_t n) {
    std::vector<uint16_t> h((size_t) n);
    b.copy_to_host(h.data(), h.size() * 2);
    std::vector<float> out((size_t) n);
    f16_to_f32(h.data(), out.data(), (size_t) n);
    return out;
}

} // namespace

std::vector<float> hc_group_norm(const std::vector<float> & x, const std::vector<float> & w_norm) {
    const int64_t hidden = hidden_of(w_norm);
    if ((int64_t) x.size() != hidden * kHc) {
        throw std::runtime_error("hc: activation has the wrong size");
    }
    std::vector<float> xn(x.size());
    for (int64_t s = 0; s < kHc; ++s) {
        double ss = 0.0;
        for (int64_t i = 0; i < hidden; ++i) {
            const double v = x[(size_t) (s * hidden + i)];
            ss += v * v;
        }
        const float scale = 1.0f / std::sqrt((float) (ss / (double) hidden) + 1e-6f);
        for (int64_t i = 0; i < hidden; ++i) {
            const size_t k = (size_t) (s * hidden + i);
            xn[k] = x[k] * scale * w_norm[k];
        }
    }
    return xn;
}

HcMixed hc_mix(Stream & stream, const std::vector<float> & x, const std::vector<float> & w_norm,
               const std::vector<float> & w_down, const std::vector<float> & w_up,
               const std::vector<float> & w_inject) {
    const int64_t hidden = hidden_of(w_norm);
    const int64_t hc_dim = hidden * kHc;
    if (w_down.size() != (size_t) (hc_dim * kHcLowRank)) {
        throw std::runtime_error("hc: down weight has the wrong size");
    }
    if (w_up.size() != (size_t) (hc_dim * kHcLowRank)) {
        throw std::runtime_error("hc: up weight has the wrong size");
    }
    if (w_inject.size() != (size_t) (hc_dim * kHc)) {
        throw std::runtime_error("hc: inject weight has the wrong size");
    }

    const std::vector<float> xn = hc_group_norm(x, w_norm);

    Buffer b_xn((size_t) hc_dim * 2), b_lo((size_t) kHcLowRank * 2), b_gate((size_t) hc_dim * 2);
    Buffer b_down((size_t) w_down.size() * 2), b_up((size_t) w_up.size() * 2);
    Buffer b_inj((size_t) w_inject.size() * 2), b_inject_out((size_t) kHc * 2);
    Buffer b_mixed((size_t) hidden * 2);

    upload_floats(xn, b_xn);
    upload_floats(w_down, b_down);
    upload_floats(w_up, b_up);
    upload_floats(w_inject, b_inj);

    Tensor t_xn{ b_xn.data(), DType::F16, 1, hc_dim };
    Tensor t_lo{ b_lo.data(), DType::F16, 1, kHcLowRank };
    Tensor t_gate{ b_gate.data(), DType::F16, 1, hc_dim };
    Tensor t_down{ b_down.data(), DType::F16, hc_dim, kHcLowRank };
    Tensor t_up{ b_up.data(), DType::F16, kHcLowRank, hc_dim };
    Tensor t_inj{ b_inj.data(), DType::F16, hc_dim, kHc };
    Tensor t_inject_out{ b_inject_out.data(), DType::F16, 1, kHc };

    // the low-rank gate is shared by both norms, scaled by 1/hc and rectified
    matmul(stream, t_down, t_xn, t_lo);
    std::vector<float> lo = read_back_f16(b_lo, kHcLowRank);
    for (float & v : lo) {
        const float u = v / (float) kHc;
        v = u / (1.0f + std::exp(-u));
    }
    upload_floats(lo, b_lo);

    matmul(stream, t_up, t_lo, t_gate);
    const std::vector<float> gate = read_back_f16(b_gate, hc_dim);

    // the block reads the mean of the gated streams
    std::vector<float> mixed((size_t) hidden, 0.0f);
    for (int64_t s = 0; s < kHc; ++s) {
        for (int64_t i = 0; i < hidden; ++i) {
            const size_t k = (size_t) (s * hidden + i);
            mixed[(size_t) i] += xn[k] * (1.0f / (1.0f + std::exp(-gate[k])));
        }
    }
    for (float & v : mixed) {
        v /= (float) kHc;
    }

    // the scatter weights come from the same normalised streams
    matmul(stream, t_inj, t_xn, t_inject_out);
    std::vector<float> inject = read_back_f16(b_inject_out, kHc);
    for (float & v : inject) {
        v /= (float) kHc;
    }

    HcMixed out;
    out.input = std::move(mixed);
    out.inject = std::move(inject);
    return out;
}

void hc_combine(std::vector<float> & residual, const std::vector<float> & block_out,
                const std::vector<float> & inject) {
    if (inject.size() != (size_t) kHc) {
        throw std::runtime_error("hc: inject has the wrong size");
    }
    const int64_t hidden = (int64_t) block_out.size();
    if ((int64_t) residual.size() != hidden * kHc) {
        throw std::runtime_error("hc: residual has the wrong size");
    }
    for (int64_t s = 0; s < kHc; ++s) {
        // centred on 1, so a zero injection leaves a plain residual add
        const float w = 2.0f / (1.0f + std::exp(-inject[(size_t) s]));
        for (int64_t i = 0; i < hidden; ++i) {
            residual[(size_t) (s * hidden + i)] += block_out[(size_t) i] * w;
        }
    }
}

} // namespace strata::cann
