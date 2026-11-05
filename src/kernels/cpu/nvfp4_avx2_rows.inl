// src/kernels/cpu/nvfp4_avx2_rows.inl - the NVFP4 row kernels of nvfp4_avx2.cpp, included there twice like
// q2_avx2_rows.inl: the AVX2 form (STRATA_ROWS_VNNI 0, namespace plain) and the AVX-VNNI one (1, namespace vnni),
// every function under STRATA_ROWS_FN (GCC/Clang's target("avxvnni") in the VNNI copy).  Not a header.

// The doubled E2M1 magnitudes (the unsigned side, |w| <= 12 of kvalues_fp4) against the sign-carrying Q8_0
// activation, four products per int32 lane.  vpdpbusd is one uop where maddubs + madd(ones) take two on the multiply
// ports.  maddubs cannot saturate on this data: a pair is at most 2 * 12 * 127 = 3048 against 32767, and the lane it
// feeds at most 6096 against 2^31, so neither form loses a bit and the two leave the same sums.  Widen the weights
// past 4 bits, or fold a scale into the activation, and that bound is gone - nvfp4_cpu_parity is what catches it.
STRATA_ROWS_FN inline __m256i dot4(__m256i mags, __m256i act) {
#if STRATA_ROWS_VNNI
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), mags, act);
#else
    return _mm256_madd_epi16(_mm256_maddubs_epi16(mags, act), _mm256_set1_epi16(1));
#endif
}

// ggml_vec_dot_nvfp4_q8_0's AVX2 path, step for step: the same integer sums, the same FMA order per 32 values, the
// same hsum_float_8.  The accumulator is one float vector per token on purpose (expert.hpp: reducing it per block
// would change the bits), so what row_dot costs is its FMA chain, not dot4.
template <int NT> STRATA_ROWS_FN
inline void row_dot(const uint8_t* row, const Acts& a, float* res) {
    const __m128i kv = _mm_loadu_si128((const __m128i*) kvalues_fp4);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    const block_nvfp4* x = (const block_nvfp4*) row;
    const int prefetch_ahead = prefetch_distance();
    for (int ib = 0; ib < a.nb; ++ib) {
        if (prefetch_ahead > 0) _mm_prefetch((const char*) (x + ib) + prefetch_ahead, _MM_HINT_T0);
        const __m256i v01 = values32(x[ib].qs, kv), v23 = values32(x[ib].qs + 16, kv);
        const __m256i a01 = _mm256_sign_epi8(v01, v01), a23 = _mm256_sign_epi8(v23, v23);
        // lanes 0-3 ue4m3(d[0]), 4-7 d[1] (and d[2], d[3]): times a broadcast d_q8, each lane is ggml's scalar
        // ue4m3(d[s]) * d_q8, rounded the same
        const __m256 u01 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x[ib].d[1]]), _mm_set1_ps(kUe4m3.v[x[ib].d[0]]));
        const __m256 u23 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x[ib].d[3]]), _mm_set1_ps(kUe4m3.v[x[ib].d[2]]));
        for (int t = 0; t < NT; ++t) {
            const block_q8_0* yb = a.y[t] + 2 * ib;
            const __m256i q01 = _mm256_loadu_si256((const __m256i*) yb[0].qs);
            const __m256i q23 = _mm256_loadu_si256((const __m256i*) yb[1].qs);
            const __m256i p1 = dot4(a01, _mm256_sign_epi8(q01, v01));
            const __m256i p2 = dot4(a23, _mm256_sign_epi8(q23, v23));
            const float* dy = a.dy + (size_t) t * 2 * a.nb + 2 * ib;
            const __m256 s01 = _mm256_mul_ps(u01, _mm256_broadcast_ss(dy));
            const __m256 s23 = _mm256_mul_ps(u23, _mm256_broadcast_ss(dy + 1));
            acc[t] = _mm256_fmadd_ps(s01, _mm256_cvtepi32_ps(p1), acc[t]);
            acc[t] = _mm256_fmadd_ps(s23, _mm256_cvtepi32_ps(p2), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(acc[t]);
}

template <int NT> STRATA_ROWS_FN
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, const Acts& a, float* const* ff, int r0, int r1,
             float sg, float su) {
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(blob + (size_t) r * gu_row, a, g);
        row_dot<NT>(blob + up_off + (size_t) r * gu_row, a, u);
        for (int t = 0; t < NT; ++t) {   // native_gu_rows' ggml path, operation for operation
            const float gs = g[t] * sg, us = u[t] * su;
            ff[t][r] = (gs / (1.f + std::exp(-gs))) * us;
        }
    }
}

template <int NT> STRATA_ROWS_FN
void dot_rows(const uint8_t* w, size_t row_bytes, const Acts& a, float* const* out, int r0, int r1, float scale) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(w + (size_t) r * row_bytes, a, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t] * scale;
    }
}

// One switch per ISA form, not one per form of every caller: `nt` is at most kMaxW, the slice's width.
STRATA_ROWS_FN void gu_rows_nt(const uint8_t* blob, size_t gu_row, size_t up_off, const Acts& a, int nt,
                               float* const* ff, int r0, int r1, float sg, float su) {
    switch (nt) {
        case 1: gu_rows<1>(blob, gu_row, up_off, a, ff, r0, r1, sg, su); break;
        case 2: gu_rows<2>(blob, gu_row, up_off, a, ff, r0, r1, sg, su); break;
        case 3: gu_rows<3>(blob, gu_row, up_off, a, ff, r0, r1, sg, su); break;
        case 4: gu_rows<4>(blob, gu_row, up_off, a, ff, r0, r1, sg, su); break;
        case 5: gu_rows<5>(blob, gu_row, up_off, a, ff, r0, r1, sg, su); break;
        case 6: gu_rows<6>(blob, gu_row, up_off, a, ff, r0, r1, sg, su); break;
        case 7: gu_rows<7>(blob, gu_row, up_off, a, ff, r0, r1, sg, su); break;
        default: gu_rows<8>(blob, gu_row, up_off, a, ff, r0, r1, sg, su); break;
    }
}

STRATA_ROWS_FN void dot_rows_nt(const uint8_t* w, size_t row_bytes, const Acts& a, int nt, float* const* out,
                                int r0, int r1, float scale) {
    switch (nt) {
        case 1: dot_rows<1>(w, row_bytes, a, out, r0, r1, scale); break;
        case 2: dot_rows<2>(w, row_bytes, a, out, r0, r1, scale); break;
        case 3: dot_rows<3>(w, row_bytes, a, out, r0, r1, scale); break;
        case 4: dot_rows<4>(w, row_bytes, a, out, r0, r1, scale); break;
        case 5: dot_rows<5>(w, row_bytes, a, out, r0, r1, scale); break;
        case 6: dot_rows<6>(w, row_bytes, a, out, r0, r1, scale); break;
        case 7: dot_rows<7>(w, row_bytes, a, out, r0, r1, scale); break;
        default: dot_rows<8>(w, row_bytes, a, out, r0, r1, scale); break;
    }
}

#undef STRATA_ROWS_VNNI
#undef STRATA_ROWS_FN
