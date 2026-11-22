// terminal/tools/ple_c5_viterbi.cpp - the C5 PLE encoder's search (tools/ple_lowbit_emulate.py --format c5): for each row
// of 160 normalized values, the 160 3-bit codes whose trellis values lut[state] are closest in squared error.
//
// Build (a Visual Studio x64 prompt, from the repository root; build/ is not tracked):
//     md build
//     cl /nologo /O2 /arch:AVX2 /LD /EHsc /Fo:build\ /Fe:build\ple_c5_viterbi.dll terminal\tools\ple_c5_viterbi.cpp
//
// The trellis (strata/kernels/ple_lowbit.hpp, ple_c5_dequant_row): a shift register of L bits takes each 3-bit code in
// at the bottom, state_j = ((state_{j-1} << 3) | code_j) mod 2^L, and value j is lut[state_j]. It is tail-biting: the
// register starts with the row's own last L-3 code bits, so every value has a full L-bit state and no row needs bits
// beyond its 480. Viterbi finds the best such path in two passes: the row rotated by 80 values with a free start and
// end gives those last L-3 bits W from the middle of its path, then the row itself starts at W and must end in a state
// whose low L-3 bits are W.
//
// One step, all states at once: m[h] is the best cost of a path whose state after the step has h as its high L-3
// bits (state >> 3); the step's new m'[h] = min over p of m[(p*2^(L-3) + h) >> 3] + (y - lut[p*2^(L-3) + h])^2, with the
// minimizing p kept for the trace back. The float operations are the same as test_ple_lowbit_emulate.py's numpy
// reference (no FMA: e = y - lut, e * e, then the add) and ties go to the lowest p and the lowest final state, so both
// give the same codes.
#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {
constexpr int kT = 160;

struct Viterbi {
    int L, S, H, G;
    std::vector<float> m, mn;
    std::vector<uint8_t> back;
    explicit Viterbi(int l) : L(l), S(1 << l), H(S >> 3), G(H >> 3), m(H), mn(H), back((size_t) kT * H) {}

    // start: -1 any register, else the register's low L-3 bits before the first code. end: -1 any final state, else
    // the final state's low L-3 bits.
    void run(const float* y, const float* lut, int start, int end, uint8_t* codes) {
        for (int h = 0; h < H; ++h) m[h] = start < 0 || h == start ? 0.0f : INFINITY;
        const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        for (int t = 0; t + 1 < kT; ++t) {
            const __m256 vy = _mm256_set1_ps(y[t]);
            uint8_t* bk = back.data() + (size_t) t * H;
            for (int h = 0; h < H; h += 32) {
                __m256i b4[4];
                for (int q = 0; q < 4; ++q) {
                    const int hq = h + 8 * q;
                    __m256 e = _mm256_sub_ps(vy, _mm256_loadu_ps(lut + hq));
                    __m256 vm = _mm256_add_ps(_mm256_set1_ps(m[hq >> 3]), _mm256_mul_ps(e, e));
                    __m256 vb = _mm256_setzero_ps();
                    for (int p = 1; p < 8; ++p) {
                        e = _mm256_sub_ps(vy, _mm256_loadu_ps(lut + (size_t) p * H + hq));
                        const __m256 vj = _mm256_add_ps(_mm256_set1_ps(m[p * G + (hq >> 3)]), _mm256_mul_ps(e, e));
                        const __m256 lt = _mm256_cmp_ps(vj, vm, _CMP_LT_OQ);
                        vm = _mm256_blendv_ps(vm, vj, lt);
                        vb = _mm256_blendv_ps(vb, _mm256_castsi256_ps(_mm256_set1_epi32(p)), lt);
                    }
                    _mm256_storeu_ps(mn.data() + hq, vm);
                    b4[q] = _mm256_castps_si256(vb);
                }
                // 32 argmins (0..7) as bytes, in h order
                const __m256i w = _mm256_packus_epi16(_mm256_packus_epi32(b4[0], b4[1]), _mm256_packus_epi32(b4[2], b4[3]));
                _mm256_storeu_si256((__m256i*) (bk + h), _mm256_permutevar8x32_epi32(w, order));
            }
            m.swap(mn);
        }
        const float yl = y[kT - 1];
        int st = 0;
        float best = INFINITY;
        for (int s = 0; s < S; ++s) {
            if (end >= 0 && (s & (H - 1)) != end) continue;
            const float e = yl - lut[s];
            const float j = m[s >> 3] + e * e;
            if (j < best) { best = j; st = s; }
        }
        for (int t = kT - 1; t >= 0; --t) {
            codes[t] = (uint8_t) (st & 7);
            const int h = st >> 3;
            if (t > 0) st = ((int) back[(size_t) (t - 1) * H + h] << (L - 3)) | h;
        }
    }
};

// the last L-3 bits of a row's code stream, code 159 in the low 3 bits
int tail_bits(const uint8_t* c, int L) {
    int w = 0;
    for (int k = 0; 3 * k < L - 3; ++k) w |= (int) c[kT - 1 - k] << (3 * k);
    return w & ((1 << (L - 3)) - 1);
}
}  // namespace

/// The interface version tools/ple_lowbit_emulate.py checks.
extern "C" __declspec(dllexport) int ple_c5_abi() { return 1; }

/// y: n rows of 160 floats, lut: 2^L floats, codes: n x 160 bytes (0..7) out. 0 on success, -1 for an L outside 8..16.
extern "C" __declspec(dllexport) int ple_c5_viterbi(const float* y, int64_t n, const float* lut, int L, uint8_t* codes) {
    if (L < 8 || L > 16) return -1;
    Viterbi v(L);
    float yr[kT];
    uint8_t cr[kT];
    for (int64_t r = 0; r < n; ++r) {
        const float* x = y + r * kT;
        uint8_t* c = codes + r * kT;
        for (int t = 0; t < kT; ++t) yr[t] = x[(t + kT / 2) % kT];
        v.run(yr, lut, -1, -1, cr);
        for (int t = 0; t < kT; ++t) c[(t + kT / 2) % kT] = cr[t];
        const int w = tail_bits(c, L);
        v.run(x, lut, w, w, c);
    }
    return 0;
}
