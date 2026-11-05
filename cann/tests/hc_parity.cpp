// cann/tests/hc_parity.cpp - hyper-connections against a double-precision reference.
//
// The reference repeats the documented maths in double on the host. The device path stores and
// multiplies in f16, so some difference is expected; what matters is that it is f16-sized and not
// a layout error, which would show up as a large fraction rather than a small one.
#include <cmath>
#include <cstdio>
#include <vector>

#include "strata/cann/device.hpp"
#include "strata/cann/hc.hpp"

using namespace strata::cann;

namespace {

constexpr int64_t kHidden = 2048;

struct Rng {
    uint64_t s = 0x9e3779b97f4a7c15ull;
    float next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return (float) ((double) (s >> 11) / (double) (1ull << 53)) * 2.0f - 1.0f;
    }
};

std::vector<float> random_vec(Rng & r, size_t n, float scale) {
    std::vector<float> v(n);
    for (auto & x : v) {
        x = r.next() * scale;
    }
    return v;
}

void reference(const std::vector<float> & x, const std::vector<float> & w_norm,
               const std::vector<float> & w_down, const std::vector<float> & w_up,
               const std::vector<float> & w_inject, const std::vector<float> & block_out,
               std::vector<double> & mixed_ref, std::vector<double> & final_ref) {
    const int64_t hc_dim = kHidden * kHc;
    std::vector<double> xn((size_t) hc_dim);
    for (int64_t s = 0; s < kHc; ++s) {
        double ss = 0.0;
        for (int64_t i = 0; i < kHidden; ++i) {
            const double v = x[(size_t) (s * kHidden + i)];
            ss += v * v;
        }
        const double scale = 1.0 / std::sqrt(ss / (double) kHidden + 1e-6);
        for (int64_t i = 0; i < kHidden; ++i) {
            const size_t k = (size_t) (s * kHidden + i);
            xn[k] = (double) x[k] * scale * (double) w_norm[k];
        }
    }
    // lo = silu((w_down^T xn) / hc)
    std::vector<double> lo((size_t) kHcLowRank, 0.0);
    for (int64_t j = 0; j < kHcLowRank; ++j) {
        double acc = 0.0;
        for (int64_t k = 0; k < hc_dim; ++k) {
            acc += (double) w_down[(size_t) (k * kHcLowRank + j)] * xn[(size_t) k];
        }
        const double u = acc / (double) kHc;
        lo[(size_t) j] = u / (1.0 + std::exp(-u));
    }
    // gate = w_up^T lo, then the mean over streams
    mixed_ref.assign((size_t) kHidden, 0.0);
    for (int64_t s = 0; s < kHc; ++s) {
        for (int64_t i = 0; i < kHidden; ++i) {
            const int64_t row = s * kHidden + i;
            double acc = 0.0;
            for (int64_t j = 0; j < kHcLowRank; ++j) {
                acc += (double) w_up[(size_t) (j * hc_dim + row)] * lo[(size_t) j];
            }
            const double g = 1.0 / (1.0 + std::exp(-acc));
            mixed_ref[(size_t) i] += xn[(size_t) row] * g / (double) kHc;
        }
    }
    // inject = (w_inject^T xn) / hc
    std::vector<double> inject((size_t) kHc, 0.0);
    for (int64_t s = 0; s < kHc; ++s) {
        double acc = 0.0;
        for (int64_t k = 0; k < hc_dim; ++k) {
            acc += (double) w_inject[(size_t) (k * kHc + s)] * xn[(size_t) k];
        }
        inject[(size_t) s] = acc / (double) kHc;
    }
    final_ref.assign((size_t) hc_dim, 0.0);
    for (int64_t s = 0; s < kHc; ++s) {
        const double w = 2.0 / (1.0 + std::exp(-inject[(size_t) s]));
        for (int64_t i = 0; i < kHidden; ++i) {
            final_ref[(size_t) (s * kHidden + i)] = (double) x[(size_t) (s * kHidden + i)] +
                                                    (double) block_out[(size_t) i] * w;
        }
    }
}

double rel_err(const std::vector<float> & got, const std::vector<double> & want) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double) got[i] - want[i];
        num += d * d;
        den += want[i] * want[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

} // namespace

int main() {
    Rng r;
    const int64_t hc_dim = kHidden * kHc;
    const std::vector<float> x = random_vec(r, (size_t) hc_dim, 0.1f);
    const std::vector<float> w_norm = random_vec(r, (size_t) hc_dim, 0.1f);
    const std::vector<float> w_down = random_vec(r, (size_t) (hc_dim * kHcLowRank), 0.02f);
    const std::vector<float> w_up = random_vec(r, (size_t) (hc_dim * kHcLowRank), 0.02f);
    const std::vector<float> w_inject = random_vec(r, (size_t) (hc_dim * kHc), 0.05f);
    const std::vector<float> block_out = random_vec(r, (size_t) kHidden, 0.1f);

    std::vector<double> mixed_ref, final_ref;
    reference(x, w_norm, w_down, w_up, w_inject, block_out, mixed_ref, final_ref);

    Device device;
    Stream stream(device);
    const HcMixed m = hc_mix(stream, x, w_norm, w_down, w_up, w_inject);
    std::vector<float> residual = x;
    hc_combine(residual, block_out, m.inject);

    const double e_mix = rel_err(m.input, mixed_ref);
    const double e_fin = rel_err(residual, final_ref);
    printf("hidden %lld, streams %lld, low rank %lld\n", (long long) kHidden, (long long) kHc,
           (long long) kHcLowRank);
    printf("inject     %s\n", [&] {
        char * buf = new char[128];
        snprintf(buf, 128, "%.6f %.6f %.6f %.6f", m.inject[0], m.inject[1], m.inject[2], m.inject[3]);
        std::string s(buf);
        delete[] buf;
        return s;
    }().c_str());
    printf("mixed   rel err vs double = %.3e\n", e_mix);
    printf("combine rel err vs double = %.3e\n", e_fin);
    const bool ok = e_mix < 2e-2 && e_fin < 2e-2;
    printf("%s\n", ok ? "hc mix and combine match the reference" : "MISMATCH");
    return ok ? 0 : 1;
}
