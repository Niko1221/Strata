#include "strata/hy4flash_gguf/cpu/native_q81.hpp"
#include "ggml.h"
#include <immintrin.h>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace strata::hy4_flash::cpu {
namespace {
float half_at(const uint8_t* p){ggml_fp16_t h;std::memcpy(&h,p,2);return ggml_fp16_to_fp32(h);}
int sum4(__m128i v){v=_mm_add_epi32(v,_mm_shuffle_epi32(v,_MM_SHUFFLE(1,0,3,2)));v=_mm_add_epi32(v,_mm_shuffle_epi32(v,_MM_SHUFFLE(2,3,0,1)));return _mm_cvtsi128_si32(v);}
void scale_min(const uint8_t*s,int j,int&sc,int&mn){if(j<4){sc=s[j]&63;mn=s[j+4]&63;}else{sc=(s[j+4]&15)|((s[j-4]>>6)<<4);mn=(s[j+4]>>4)|((s[j]>>6)<<4);}}
}
// Preserve the scalar 16,8,4,2,1 reduction tree; do not use horizontal
// additions with a different pairing or reciprocal-multiply division.
void native_quant_q81_avx2(const float* x, void* dst, int n) {
    auto* y = static_cast<uint8_t*>(dst);
    const __m256 sign = _mm256_set1_ps(-0.f);
    const __m256 finite_max = _mm256_set1_ps(std::numeric_limits<float>::max());
    for (int b = 0; b < n / 32; ++b) {
        __m256 v[4], a[4];
        bool finite = true;
        for (int j = 0; j < 4; ++j) {
            v[j] = _mm256_loadu_ps(x + b * 32 + j * 8);
            a[j] = _mm256_andnot_ps(sign, v[j]);
            finite &= _mm256_movemask_ps(_mm256_cmp_ps(a[j], finite_max, _CMP_LE_OQ)) == 255;
        }
        if (!finite) { native_quant_q81_scalar(x + b * 32, y + b * 36, 32); continue; }
        __m256 mx = _mm256_max_ps(_mm256_max_ps(a[0], a[1]), _mm256_max_ps(a[2], a[3]));
        __m128 m = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ss(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1,1,1,1)));
        const float amax = _mm_cvtss_f32(m);
        float d = amax / 127.f;
        // Scalar retains its original handling of zero, underflow and exceptional inputs.
        if (d == 0.f || !std::isfinite(d)) {
            native_quant_q81_scalar(x + b * 32, y + b * 36, 32); continue;
        }
        if (std::fabs(d) > 65504.f) d = std::copysign(65504.f, d);
        __m256 s8 = _mm256_add_ps(_mm256_add_ps(v[0], v[2]), _mm256_add_ps(v[1], v[3]));
        __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
        __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        float sum = _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, _MM_SHUFFLE(1,1,1,1))));
        if (!std::isfinite(sum)) {
            native_quant_q81_scalar(x + b * 32, y + b * 36, 32); continue;
        }
        const float raw = std::fabs(sum) > 65504.f ? std::copysign(65504.f, sum) : sum;
        const ggml_fp16_t ds[2] = {ggml_fp32_to_fp16(d), ggml_fp32_to_fp16(raw)};
        std::memcpy(y + b * 36, ds, 4);
        __m256i codes[4];
        for (int j = 0; j < 4; ++j) {
            __m256 q = _mm256_div_ps(v[j], _mm256_set1_ps(d));
            // Clamp before integer conversion: large finite inputs can exceed int32.
            const __m256 aq = _mm256_min_ps(_mm256_andnot_ps(sign, q), _mm256_set1_ps(127.f));
            const __m256 whole = _mm256_round_ps(aq, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            const __m256 frac = _mm256_sub_ps(aq, whole);
            const __m256 bump = _mm256_and_ps(_mm256_cmp_ps(frac, _mm256_set1_ps(.5f), _CMP_GE_OQ), _mm256_set1_ps(1.f));
            const __m256 rounded = _mm256_or_ps(_mm256_add_ps(whole, bump), _mm256_and_ps(q, sign));
            codes[j] = _mm256_cvttps_epi32(rounded);
        }
        __m256i q01 = _mm256_packs_epi32(codes[0], codes[1]);
        __m256i q23 = _mm256_packs_epi32(codes[2], codes[3]);
        q01 = _mm256_permute4x64_epi64(q01, _MM_SHUFFLE(3,1,2,0));
        q23 = _mm256_permute4x64_epi64(q23, _MM_SHUFFLE(3,1,2,0));
        __m256i packed = _mm256_packs_epi16(q01, q23);
        packed = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3,1,2,0));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(y + b * 36 + 4), packed);
    }
}
float native_dot_q81_avx2(int type,const uint8_t* row,const void* act,int n) {
    if(type!=12 && type!=14)throw std::invalid_argument("Q8_1 CPU dot needs Q4_K/Q6_K");
    const auto* a=static_cast<const uint8_t*>(act);
    const __m256i mask15=_mm256_set1_epi8(15),ones8=_mm256_set1_epi8(1),ones16=_mm256_set1_epi16(1);
    double total=0;
    for(int b=0;b<n/256;++b) {
        const uint8_t*w=row+b*(type==12?144:210);
        const float dw=half_at(w+(type==12?0:208)),dm=type==12?half_at(w+2):0.f;
        for(int g=0;g<8;++g) {
            const uint8_t*q8=a+(b*8+g)*36;const __m256i aq=_mm256_loadu_si256((const __m256i*)(q8+4));
            const __m256i sa16=_mm256_maddubs_epi16(ones8,aq);
            __m256i q;
            if(type==12) {
                q=_mm256_loadu_si256((const __m256i*)(w+16+(g/2)*32));
                if(g&1)q=_mm256_srli_epi16(q,4);
                q=_mm256_and_si256(q,mask15);
            } else {
                const int h=g/4,p=g%4;
                q=_mm256_loadu_si256((const __m256i*)(w+h*64+(p&1)*32));
                if(p>=2)q=_mm256_srli_epi16(q,4);
                q=_mm256_and_si256(q,mask15);
                __m256i hi=_mm256_loadu_si256((const __m256i*)(w+128+h*32));
                hi=_mm256_srl_epi16(hi,_mm_cvtsi32_si128(2*p));
                hi=_mm256_slli_epi16(_mm256_and_si256(hi,_mm256_set1_epi8(3)),4);
                q=_mm256_or_si256(q,hi);
            }
            __m256i prod=_mm256_maddubs_epi16(q,aq); // max pair 2*63*127, no saturation.
            if(type==14)prod=_mm256_sub_epi16(prod,_mm256_slli_epi16(sa16,5));
            prod=_mm256_madd_epi16(prod,ones16);
            const int lo=sum4(_mm256_castsi256_si128(prod)),hi=sum4(_mm256_extracti128_si256(prod,1));
            const float da=half_at(q8);
            if(type==12) {
                int sc,mn;scale_min(w+4,g,sc,mn);
                const __m256i sums=_mm256_madd_epi16(sa16,ones16);
                const int asum=sum4(_mm256_castsi256_si128(sums))+sum4(_mm256_extracti128_si256(sums,1));
                total+=(double)dw*da*((lo+hi)*sc)-(double)dm*da*(asum*mn);
            } else total+=(double)dw*da*(lo*(int8_t)w[192+g*2]+hi*(int8_t)w[193+g*2]);
        }
    }
    return (float)total;
}
void native_dot_q81_pair_q4_avx2(const uint8_t* gate, const uint8_t* up,
                                const void* act, int n, float& gout, float& uout) {
    const auto* a = static_cast<const uint8_t*>(act);
    const __m256i mask15 = _mm256_set1_epi8(15), ones8 = _mm256_set1_epi8(1),
                  ones16 = _mm256_set1_epi16(1);
    double gt = 0, ut = 0;
    for (int b = 0; b < n / 256; ++b) {
        const uint8_t* gw = gate + b * 144;
        const uint8_t* uw = up + b * 144;
        const float gdw = half_at(gw), gdm = half_at(gw + 2);
        const float udw = half_at(uw), udm = half_at(uw + 2);
        for (int j = 0; j < 8; ++j) {
            // Only input work is shared. Each projection retains the original b,j order,
            // integer reduction, double expression, and final float conversion.
            const uint8_t* q8 = a + (b * 8 + j) * 36;
            const __m256i aq = _mm256_loadu_si256((const __m256i*)(q8 + 4));
            const __m256i sums = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, aq), ones16);
            const int asum = sum4(_mm256_castsi256_si128(sums)) +
                             sum4(_mm256_extracti128_si256(sums, 1));
            const float da = half_at(q8);
            __m256i gq = _mm256_loadu_si256((const __m256i*)(gw + 16 + (j / 2) * 32));
            __m256i uq = _mm256_loadu_si256((const __m256i*)(uw + 16 + (j / 2) * 32));
            if (j & 1) { gq = _mm256_srli_epi16(gq, 4); uq = _mm256_srli_epi16(uq, 4); }
            gq = _mm256_and_si256(gq, mask15);
            uq = _mm256_and_si256(uq, mask15);
            const __m256i gp = _mm256_madd_epi16(_mm256_maddubs_epi16(gq, aq), ones16);
            const __m256i upv = _mm256_madd_epi16(_mm256_maddubs_epi16(uq, aq), ones16);
            const int gl = sum4(_mm256_castsi256_si128(gp)), gh = sum4(_mm256_extracti128_si256(gp, 1));
            const int ul = sum4(_mm256_castsi256_si128(upv)), uh = sum4(_mm256_extracti128_si256(upv, 1));
            int gs, gm, us, um;
            scale_min(gw + 4, j, gs, gm);
            scale_min(uw + 4, j, us, um);
            gt += (double)gdw * da * ((gl + gh) * gs) - (double)gdm * da * (asum * gm);
            ut += (double)udw * da * ((ul + uh) * us) - (double)udm * da * (asum * um);
        }
    }
    gout = (float)gt;
    uout = (float)ut;
}
} // namespace strata::hy4_flash::cpu
