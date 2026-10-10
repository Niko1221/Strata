// The Q2_0 rows of src/kernels/cpu/q2_avx1.cpp (CPUs without AVX2) bitwise against the scalar reference, CPU only:
// every one of the 65,536 half scales, then random rows with 1..12 tokens, partial row ranges, padded and unaligned
// rows, activations at -127 / 127; and the time of one expert's down rows.  Exit 77 on a CPU without AVX.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace c = strata::kernels::cpu;
namespace {

std::mt19937 rng(20261008);

void fill_act(c::ActQ& a, int nblocks, int mode) {
    std::uniform_int_distribution<int> q(-127, 127), edge(0, 2);
    std::uniform_real_distribution<float> sc(1e-4f, 0.5f);
    a.nchunks = 2 * nblocks;
    for (int k = 0; k < a.nchunks; ++k) {
        int32_t sum = 0;
        for (int j = 0; j < c::QKA; ++j) {
            const int v = mode == 1 ? (edge(rng) == 0 ? -127 : 127) : q(rng);
            a.q[k * c::QKA + j] = (int8_t) v;
            sum += v;
        }
        a.scale[k] = mode == 2 ? 0.0f : sc(rng);
        a.sum[k] = sum;
        a.hx[k] = a.scale[k] * (float) sum;
    }
}

bool same(float x, float y) {
    if (std::isnan(x) || std::isnan(y)) return std::isnan(x) && std::isnan(y);
    uint32_t a, b;
    std::memcpy(&a, &x, 4);
    std::memcpy(&b, &y, 4);
    return a == b;
}

// every half bit pattern as the scale of a one-block row
bool all_halves() {
    const int rows = 65536;
    std::vector<uint8_t> w((size_t) rows * 18);
    std::uniform_int_distribution<int> byte(0, 255);
    for (int r = 0; r < rows; ++r) {
        const uint16_t h = (uint16_t) r;
        std::memcpy(&w[(size_t) r * 18], &h, 2);
        for (int i = 2; i < 18; ++i) w[(size_t) r * 18 + i] = (uint8_t) byte(rng);
    }
    auto act = std::make_unique<c::ActQ>();
    fill_act(*act, 1, 0);
    const c::ActQ* ap[1] = {act.get()};
    std::vector<float> ref(rows), got(rows);
    float* rp[1] = {ref.data()};
    float* gp[1] = {got.data()};
    c::q2_0_gguf_rows_multi_scalar(w.data(), 18, 1, ap, 1, rp, 0, rows);
    c::q2_0_gguf_rows_multi_avx1(w.data(), 18, 1, ap, 1, gp, 0, rows);
    for (int r = 0; r < rows; ++r)
        if (!same(ref[r], got[r])) {
            std::fprintf(stderr, "FAIL half 0x%04x: scalar %.9g avx1 %.9g\n", r, ref[r], got[r]);
            return false;
        }
    std::puts("ok   Q2_0 rows: all 65536 half scales bitwise");
    return true;
}

bool rows_case(int nblocks, int nt, int r0, int r1, int pad, int offset, int mode) {
    const int rows = r1 + 3;
    const size_t row_bytes = (size_t) nblocks * 18 + pad;
    std::vector<uint8_t> buf(row_bytes * rows + offset);
    uint8_t* w = buf.data() + offset;
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> nd(0.f, 0.02f);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nblocks; ++b) {
            uint8_t* blk = w + (size_t) r * row_bytes + (size_t) b * 18;
            const float d = nd(rng);
            uint32_t f;
            std::memcpy(&f, &d, 4);   // f32 -> f16 by truncation: any half pattern is fine for a parity test
            const uint32_t e = (f >> 23) & 0xff;
            const uint16_t h = (uint16_t) (((f >> 16) & 0x8000) | (e > 112 && e < 143 ? ((e - 112) << 10) | ((f >> 13) & 0x3ff) : 0));
            std::memcpy(blk, &h, 2);
            for (int i = 2; i < 18; ++i) blk[i] = (uint8_t) byte(rng);
        }
    std::vector<std::unique_ptr<c::ActQ>> acts;
    std::vector<const c::ActQ*> ap;
    for (int t = 0; t < nt; ++t) {
        acts.push_back(std::make_unique<c::ActQ>());
        fill_act(*acts.back(), nblocks, mode);
        ap.push_back(acts.back().get());
    }
    std::vector<std::vector<float>> ref(nt, std::vector<float>(rows, -1.f)), got(nt, std::vector<float>(rows, -2.f));
    std::vector<float*> rp, gp;
    for (int t = 0; t < nt; ++t) { rp.push_back(ref[t].data()); gp.push_back(got[t].data()); }
    c::q2_0_gguf_rows_multi_scalar(w, row_bytes, nblocks, ap.data(), nt, rp.data(), r0, r1);
    c::q2_0_gguf_rows_multi_avx1(w, row_bytes, nblocks, ap.data(), nt, gp.data(), r0, r1);
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r)
            if (!same(ref[t][r], got[t][r])) {
                std::fprintf(stderr, "FAIL nblocks=%d nt=%d rows=[%d,%d) pad=%d offset=%d mode=%d: token %d row %d: "
                             "scalar %.9g avx1 %.9g\n", nblocks, nt, r0, r1, pad, offset, mode, t, r, ref[t][r], got[t][r]);
                return false;
            }
    for (int t = 0; t < nt; ++t)   // rows outside [r0, r1) untouched
        if (got[t][r0 > 0 ? r0 - 1 : r1] != -2.f) {
            std::fprintf(stderr, "FAIL: a row outside the range was written\n");
            return false;
        }
    return true;
}

}  // namespace

int main() {
    if (!c::cpu_avx1_ok()) {
        std::puts("SKIP: this CPU has no AVX");
        return 77;
    }
    if (!all_halves()) return 1;
    int n = 0;
    for (int nt = 1; nt <= 12; ++nt)
        for (int nblocks : {1, 8, 40})
            for (int mode = 0; mode < 3; ++mode) {
                if (!rows_case(nblocks, nt, 0, 64, 0, 0, mode)) return 1;
                if (!rows_case(nblocks, nt, 5, 37, 7, 3, mode)) return 1;
                n += 2;
            }
    std::printf("ok   Q2_0 rows: %d cases (1..12 tokens, 1/8/40 blocks, partial ranges, padded/unaligned rows, "
                "edge and zero-scale activations) bitwise\n", n);
    {   // time: one expert's Q2_0 down rows (H rows of FF / 64 blocks), 1 and 4 tokens, scalar vs AVX
        const int nblocks = c::FF / 64, rows = c::H;
        std::vector<uint8_t> w((size_t) rows * nblocks * 18);
        std::uniform_int_distribution<int> byte(0, 255);
        for (size_t i = 0; i < w.size(); ++i) w[i] = (uint8_t) (i % 18 == 1 ? byte(rng) & 0x3b : byte(rng));
        std::vector<std::unique_ptr<c::ActQ>> acts;
        std::vector<const c::ActQ*> ap;
        std::vector<std::vector<float>> o(4, std::vector<float>(rows));
        std::vector<float*> op;
        for (int t = 0; t < 4; ++t) {
            acts.push_back(std::make_unique<c::ActQ>());
            fill_act(*acts.back(), nblocks, 0);
            ap.push_back(acts.back().get());
            op.push_back(o[t].data());
        }
        for (int nt : {1, 4}) {
            double ms[2];
            for (int k = 0; k < 2; ++k) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int rep = 0; rep < 50; ++rep)
                    (k == 0 ? c::q2_0_gguf_rows_multi_scalar : c::q2_0_gguf_rows_multi_avx1)(
                        w.data(), (size_t) nblocks * 18, nblocks, ap.data(), nt, op.data(), 0, rows);
                ms[k] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 50;
            }
            std::printf("     one expert's down rows, %d token(s), one core: scalar %.3f ms, avx1 %.3f ms (%.1fx)\n", nt,
                        ms[0], ms[1], ms[0] / ms[1]);
        }
    }
    std::puts("PASS: q2_avx1 rows");
    return 0;
}
