// ink_host_simd_bench: one thread, the host dot kernels of the Inkling fast lane per tier (AVX2 vs AVX-512), per format.
// A "row" is one 2304-weight gate/up row (9 super-blocks = 36 sub-groups of 64) for the S_K formats, a 2304-wide row for SF4 / MX8.
// The weights are spread over a buffer far bigger than the caches (default 512 MiB) so the numbers include the memory traffic a cold expert has.
//   ink_host_simd_bench [reps] [MiB]
// Prints ns per row and GB/s of weight bytes for each tier that the CPU can run, and the AVX-512 / AVX2 ratio.
#include "ink_host_avx2.h"
#include "ink_host_avx512.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct Luts {
    float e4[256];
    float u8[256];
    int8_t s15[256][5];
    Luts() {
        for (int c = 0; c < 256; ++c) {
            const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
            float v = (e == 15 && m == 7) ? 0.0f : (e == 0 ? (float)m / 512.0f : (1.0f + (float)m * 0.125f) * std::ldexp(1.0f, e - 7));
            e4[c] = s ? -v : v;
            uint32_t u = c == 255 ? 0 : (c == 0 ? 0x00400000u : (uint32_t)c << 23);
            std::memcpy(&u8[c], &u, 4);
        }
        for (int b = 0; b < 256; ++b) { int v = b; for (int k = 0; k < 5; ++k) { s15[b][k] = (int8_t)((v % 3) - 1); v /= 3; } }
    }
};

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace

int main(int argc, char** argv) {
    using namespace strata::ink;
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    const size_t mib = argc > 2 ? (size_t)std::atoi(argv[2]) : 512;
    const bool has2 = host_avx2::available(), has5 = host_avx512::available();
    std::printf("ink_host_simd_bench: avx2 %s, avx512 %s, buffer %zu MiB, %d reps (best of)\n", has2 ? "yes" : "no", has5 ? "yes" : "no", mib, reps);
    static const Luts L;
    const int cols = 2304;
    struct Fmt { const char* name; int code; size_t row_bytes; };   // code 100 = SF4, 101 = MX8
    const Fmt fmts[] = {{"S1_5", 1, 9 * 58}, {"S2_K", 2, 9 * 70}, {"S3_K", 3, 9 * 102}, {"S4_K", 4, 9 * 134}, {"S5_K", 5, 9 * 166},
                        {"S6_K", 6, 9 * 198}, {"S8_K", 8, 9 * 262}, {"SF4", 100, (size_t)cols / 2 + cols / 16}, {"MX8", 101, (size_t)cols + cols / 32}};
    std::vector<uint8_t> buf(mib << 20);
    uint32_t s = 88172645u;
    for (auto& b : buf) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; b = (uint8_t)(s >> 7); }
    for (size_t i = 0; i < buf.size(); i += 1024) buf[i] = 120;   // keep SF4 / MX8 scale bytes (anywhere) from being all NaN-ish: harmless for timing
    alignas(64) int8_t qs[64];
    for (int i = 0; i < 64; ++i) qs[i] = (int8_t)(i * 7 - 100);
    std::vector<float> x(cols);
    for (int i = 0; i < cols; ++i) x[i] = 0.01f * (float)((i * 37) % 101 - 50);

    std::printf("%-6s %12s %12s %10s %10s %8s\n", "fmt", "avx2 ns/row", "avx512 ns/row", "avx2 GB/s", "avx512 GB/s", "ratio");
    volatile float sink = 0.0f;
    for (const Fmt& f : fmts) {
        const size_t nrows = buf.size() / f.row_bytes - 1;
        auto run = [&](int tier) {
            double best = 1e30;
            for (int r = 0; r < reps; ++r) {
                float acc = 0.0f;
                const double t0 = now();
                for (size_t row = 0; row < nrows; ++row) {
                    const uint8_t* p = buf.data() + row * f.row_bytes;
                    if (f.code == 100) {
                        const uint8_t* sc = p + cols / 2;
                        acc += tier == 5 ? host_avx512::dot_sf4_row(p, sc, 0.5f, x.data(), cols, L.e4) : host_avx2::dot_sf4_row(p, sc, 0.5f, x.data(), cols, L.e4);
                    } else if (f.code == 101) {
                        const uint8_t* sc = p + cols;
                        acc += tier == 5 ? host_avx512::dot_mx8_row(p, sc, x.data(), cols, L.e4, L.u8) : host_avx2::dot_mx8_row(p, sc, x.data(), cols, L.e4, L.u8);
                    } else {
                        const size_t sb = f.row_bytes / 9;
                        for (int sbi = 0; sbi < 9; ++sbi) {
                            const uint8_t* pl = p + sbi * sb + 6;
                            for (int g = 0; g < 4; ++g)
                                acc += tier == 5 ? host_avx512::dot_sk_sub64(pl, f.code, g, qs, 0.01f, 1.0f, 0.5f, L.s15)
                                                 : host_avx2::dot_sk_sub64(pl, f.code, g, qs, 0.01f, 1.0f, 0.5f, L.s15);
                        }
                    }
                }
                const double dt = now() - t0;
                sink = acc;
                if (dt < best) best = dt;
            }
            return best / (double)nrows;   // seconds per row
        };
        const double t2 = has2 ? run(2) : 0.0, t5 = has5 ? run(5) : 0.0;
        std::printf("%-6s %12.0f %12.0f %10.2f %10.2f %8.2f\n", f.name, t2 * 1e9, t5 * 1e9, t2 > 0 ? f.row_bytes / t2 / 1e9 : 0.0, t5 > 0 ? f.row_bytes / t5 / 1e9 : 0.0,
                    (t2 > 0 && t5 > 0) ? t2 / t5 : 0.0);
    }
    (void)sink;
    return 0;
}
