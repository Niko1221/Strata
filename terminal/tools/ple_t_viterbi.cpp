// terminal/tools/ple_t_viterbi.cpp - the T58 / T58e PLE encoder's search (tools/ple_lowbit_emulate.py --format t58 | t58e):
// for each row of 160 normalized values, the trellis states whose values lut[state] are closest in squared error.
//
// Build (a Visual Studio x64 prompt, from the repository root; build/ is not tracked):
//     md build
//     cl /nologo /O2 /arch:AVX2 /LD /EHsc /Fo:build\ /Fe:build\ple_t_viterbi.dll terminal\tools\ple_t_viterbi.cpp
//
// The trellis (strata/kernels/ple_lowbit.hpp, ple_t58_dequant_row) generalizes C5's (ple_c5_viterbi.cpp) to a fractional
// rate: a row's codes are one circular bit stream, value j shifts b[j] new bits (1..4) into an L-bit register,
// state_j = ((state_{j-1} << b[j]) | chunk_j) mod 2^L, so state_j is the L stream bits ending at b[0] + ... + b[j], most
// significant first, and value j is lut[state_j]. T58 is b = 2 3 3 3 3 repeated (448 bits); b = 3 everywhere is C5's
// trellis and gives C5's states. Tail-biting as in C5: pass 1 on the row rotated by 80 values (b must repeat with
// period 80) with a free start and end gives W, the low L - b[0] bits of state 159; pass 2 starts at W and must end in
// a state whose low L - b[0] bits are W.
//
// One step t: m holds the best cost of a path for each value of state_{t-1}'s low L - b[t] bits; for each state_t =
// p * 2^(L - b[t+1]) + h, the cost is m[state_t >> b[t]] + (y - lut[state_t])^2 and m'[h] keeps the least over the top
// b[t+1] bits p, with that p kept for the trace back. The float operations are those of test_ple_lowbit_emulate.py's
// numpy reference (no FMA: e = y - lut, e * e, then the add); ties go to the lowest p and the lowest final state.
#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {
constexpr int kT = 160;

struct Viterbi {
    int L, S;
    const uint8_t* b;
    std::vector<float> m, mn;   // before step j: S >> b[j] costs, indexed by state_{j-1}'s low L - b[j] bits (+16 pad)
    std::vector<uint8_t> back;  // step j < 159: S >> b[j+1] argmins (state_j's top b[j+1] bits)
    std::vector<size_t> boff;
    std::vector<uint16_t> sr;
    Viterbi(int l, const uint8_t* bb) : L(l), S(1 << l), b(bb), m((S >> 1) + 16), mn((S >> 1) + 16), boff(kT), sr(kT) {
        size_t o = 0;
        for (int j = 0; j + 1 < kT; ++j) { boff[j] = o; o += (size_t) (S >> b[j + 1]); }
        back.resize(o);
    }

    // start: -1 any register, else state_{-1}'s low L - b[0] bits. end: -1 any final state, else its low L - b[0] bits.
    void run(const float* y, const float* lut, int start, int end, uint16_t* st) {
        const int H0 = S >> b[0];
        for (int h = 0; h < H0; ++h) m[h] = start < 0 || h == start ? 0.0f : INFINITY;
        for (int t = 0; t + 1 < kT; ++t) {
            const int bj = b[t], nb = b[t + 1], Hn = S >> nb, P = 1 << nb;
            const __m256i perm = bj == 1 ? _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3)
                               : bj == 2 ? _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1) : _mm256_setzero_si256();
            const __m256 vy = _mm256_set1_ps(y[t]);
            uint8_t* bk = back.data() + boff[t];
            const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            for (int h = 0; h < Hn; h += 32) {
                __m256i b4[4];
                for (int q = 0; q < 4; ++q) {
                    const int hq = h + 8 * q;
                    __m256 e = _mm256_sub_ps(vy, _mm256_loadu_ps(lut + hq));
                    __m256 vm = _mm256_add_ps(_mm256_permutevar8x32_ps(_mm256_loadu_ps(m.data() + (hq >> bj)), perm), _mm256_mul_ps(e, e));
                    __m256 vb = _mm256_setzero_ps();
                    for (int p = 1; p < P; ++p) {
                        const int s0 = p * Hn + hq;
                        e = _mm256_sub_ps(vy, _mm256_loadu_ps(lut + s0));
                        const __m256 vj = _mm256_add_ps(_mm256_permutevar8x32_ps(_mm256_loadu_ps(m.data() + (s0 >> bj)), perm), _mm256_mul_ps(e, e));
                        const __m256 lt = _mm256_cmp_ps(vj, vm, _CMP_LT_OQ);
                        vm = _mm256_blendv_ps(vm, vj, lt);
                        vb = _mm256_blendv_ps(vb, _mm256_castsi256_ps(_mm256_set1_epi32(p)), lt);
                    }
                    _mm256_storeu_ps(mn.data() + hq, vm);
                    b4[q] = _mm256_castps_si256(vb);
                }
                // 32 argmins as bytes, in h order
                const __m256i w = _mm256_packus_epi16(_mm256_packus_epi32(b4[0], b4[1]), _mm256_packus_epi32(b4[2], b4[3]));
                _mm256_storeu_si256((__m256i*) (bk + h), _mm256_permutevar8x32_epi32(w, order));
            }
            m.swap(mn);
        }
        const int bl = b[kT - 1];
        const float yl = y[kT - 1];
        int s = 0;
        float best = INFINITY;
        for (int c = 0; c < S; ++c) {
            if (end >= 0 && (c & (H0 - 1)) != end) continue;
            const float e = yl - lut[c];
            const float j = m[c >> bl] + e * e;
            if (j < best) { best = j; s = c; }
        }
        st[kT - 1] = (uint16_t) s;
        for (int t = kT - 1; t > 0; --t) {
            const int h = s >> b[t];
            s = ((int) back[boff[t - 1] + h] << (L - b[t])) | h;
            st[t - 1] = (uint16_t) s;
        }
    }
};
}  // namespace

/// The interface version tools/ple_lowbit_emulate.py checks.
extern "C" __declspec(dllexport) int ple_t_abi() { return 1; }

/// y: n rows of 160 floats; lut: 2^L floats; b: 160 shifts (1..4, b[j + 80] == b[j]); states: n x 160 uint16 out (the
/// tail-biting path). 0 on success, -1 for an L outside 9..16, -2 for a shift pattern it cannot search.
extern "C" __declspec(dllexport) int ple_t_viterbi(const float* y, int64_t n, const float* lut, int L, const uint8_t* b,
                                                   uint16_t* states) {
    if (L < 9 || L > 16) return -1;
    for (int j = 0; j < kT; ++j)
        if (b[j] < 1 || b[j] > 4 || b[j] != b[(j + kT / 2) % kT]) return -2;
    Viterbi v(L, b);
    float yr[kT];
    for (int64_t r = 0; r < n; ++r) {
        const float* x = y + r * kT;
        uint16_t* st = states + r * kT;
        for (int t = 0; t < kT; ++t) yr[t] = x[(t + kT / 2) % kT];
        v.run(yr, lut, -1, -1, v.sr.data());
        const int w = v.sr[kT / 2 - 1] & ((v.S >> b[0]) - 1);
        v.run(x, lut, w, w, st);
    }
    return 0;
}
