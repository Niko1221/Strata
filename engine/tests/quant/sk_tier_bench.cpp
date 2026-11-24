// sk_tier_bench: one thread, the S_K / SF4 / MX8 codecs per ISA tier (base = plain C++, avx2, avx512), Inkling-2-Mini expert shapes.
//   gate/up: 2304 rows x 2304 columns (9 super-blocks per row), down: 2304 rows x 1152 columns (paired rows), m tokens per call (1 = decode, 8 / 16 = prompt tile).
// The weights are spread over a buffer far bigger than the caches (default 512 MiB), so cold-expert memory traffic is included.
//   sk_tier_bench [MiB] [reps]
#include "sk_tiers.hpp"
#include "strata/quant/isa.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace strata::quant::sk;

namespace {

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template <class F>
double best_of(int reps, F&& f) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now();
        f();
        const double dt = now() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    const size_t mib = argc > 1 ? (size_t)std::atoi(argv[1]) : 512;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
    const auto& c = strata::quant::caps();
    const bool has2 = c.avx2, has5 = c.avx512 && c.avx512_vnni;
    std::printf("sk_tier_bench: avx2 %s, avx512+vnni %s, buffer %zu MiB, best of %d, dispatcher '%s'\n", has2 ? "yes" : "no", has5 ? "yes" : "no", mib, reps, active_sk_isa());
    std::vector<uint8_t> buf(mib << 20);
    uint32_t s = 2463534242u;
    for (auto& b : buf) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; b = (uint8_t)(s >> 9); }
    // sane headers are not needed for timing (any bytes decode to some number)
    const SkCode codes[] = {SkCode::S1_5, SkCode::S2_K, SkCode::S3_K, SkCode::S4_K, SkCode::S5_K, SkCode::S6_K, SkCode::S8_K};
    std::printf("%-5s %-9s %3s %11s %11s %11s   %9s %9s %9s  (GB/s of weights)\n", "fmt", "shape", "m", "base ms", "avx2 ms", "avx512 ms", "base", "avx2", "avx512");
    for (SkCode code : codes) {
        for (int shape = 0; shape < 2; ++shape) {   // 0 gate/up (2304 x 2304), 1 down (2304 x 1152 paired)
            const size_t nc = shape == 0 ? 2304 : 1152, nr = 2304;
            const size_t bytes = nr * nc / QK_S * superblock_bytes(code);
            const size_t n_exp = buf.size() / bytes - 1;
            for (size_t m : {1, 8, 16}) {
                std::vector<float> x(nc * m);
                for (size_t i = 0; i < x.size(); ++i) x[i] = 0.01f * (float)((int)((i * 2654435761u) >> 20) % 201 - 100);
                std::vector<BlockQ8_1_G64> q(nc / SUB_G * m);
                base::quantize_q8_1_g64(x.data(), q.data(), nc * m);
                std::vector<float> out(nr * m);
                auto run = [&](auto&& fn) {
                    return best_of(reps, [&] {
                        for (size_t e = 0; e < n_exp && e < 24; ++e) fn(buf.data() + e * bytes, code, q.data(), out.data(), nr, nc, m);
                    }) / (double)std::min<size_t>(n_exp, 24);
                };
                auto call = [&](auto gemm, auto gemm2) {
                    return [=](const uint8_t* w, SkCode cd, const BlockQ8_1_G64* xs, float* o, size_t r, size_t cc, size_t mm) {
                        if (shape == 0) gemm(w, cd, xs, o, r, cc, mm); else gemm2(w, cd, xs, o, r, cc, mm);
                    };
                };
                const double tb = run(call(&base::gemm_sk_q8_1, &base::gemm_paired2_sk_q8_1));
                const double t2 = has2 ? run(call(&avx2::gemm_sk_q8_1, &avx2::gemm_paired2_sk_q8_1)) : 0.0;
                const double t5 = has5 ? run(call(&avx512::gemm_sk_q8_1, &avx512::gemm_paired2_sk_q8_1)) : 0.0;
                std::printf("%-5d %-9s %3zu %11.3f %11.3f %11.3f   %9.2f %9.2f %9.2f\n", (int)code, shape == 0 ? "gate/up" : "down", m, tb * 1e3, t2 * 1e3, t5 * 1e3,
                            bytes / tb / 1e9, t2 > 0 ? bytes / t2 / 1e9 : 0.0, t5 > 0 ? bytes / t5 / 1e9 : 0.0);
            }
        }
    }
    return 0;
}
