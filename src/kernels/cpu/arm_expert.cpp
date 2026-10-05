// ARM64 scalar CPU fallback for NVIDIA GB10 hosts.
// The model's primary path is CUDA; this implementation keeps the small CPU
// expert path correct when Strata is built on an ARM64 host without x86 SIMD.
#include "strata/kernels/cpu/expert.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {
namespace {
std::atomic<bool> g_oracle{false};

float h2f(const uint8_t *p) {
    uint16_t h; std::memcpy(&h, p, sizeof h);
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    uint32_t bits;
    if (!e) {
        if (!m) bits = s << 31;
        else { uint32_t mm = m; int ee = 127 - 15 + 1; while (!(mm & 0x400)) { mm <<= 1; --ee; }
               bits = (s << 31) | ((uint32_t)ee << 23) | ((mm & 1023) << 13); }
    } else if (e == 31) bits = (s << 31) | 0x7f800000u | (m << 13);
    else bits = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    float f; std::memcpy(&f, &bits, sizeof f); return f;
}

void quant(const float *x, int n, ActQ &a) {
    a.nchunks = n / QKA;
    for (int k = 0; k < a.nchunks; ++k) {
        const float *p = x + k * QKA; float mx = 0.f;
        for (int j = 0; j < QKA; ++j) mx = std::max(mx, std::fabs(p[j]));
        const float s = mx > 0.f ? mx / 127.f : 0.f, inv = s > 0.f ? 1.f / s : 0.f;
        int32_t sum = 0;
        for (int j = 0; j < QKA; ++j) { int v = (int)(p[j] * inv + (p[j] * inv >= 0.f ? .5f : -.5f));
            v = std::max(-127, std::min(127, v)); a.q[k * QKA + j] = (int8_t)v; sum += v; }
        a.scale[k] = s; a.sum[k] = sum; a.hx[k] = s * (float)sum;
    }
}

float row_dot(const uint8_t *codes, const uint8_t *scales, const ActQ &a, int blocks) {
    float out = 0.f;
    for (int b = 0; b < blocks; ++b) {
        const float d = h2f(scales + 2 * b);
        for (int j = 0; j < QK; ++j) {
            const int q = ((codes[b * 16 + (j >> 2)] >> (2 * (j & 3))) & 3) - 1;
            out += q * d * a.scale[2 * b + j / QKA] * (float)a.q[2 * b * QKA + j];
        }
    }
    return out;
}

void one(const uint8_t *blob, const ActQ &a1, float *out, ExpertScratch &ws) {
    for (int r = 0; r < FF; ++r) {
        const float g = row_dot(blob + O_GU_CODES + (size_t)(2*r)*ROW_GU, blob + O_GU_SCALES + (size_t)(2*r)*SC_GU*2, a1, SC_GU);
        const float u = row_dot(blob + O_GU_CODES + (size_t)(2*r+1)*ROW_GU, blob + O_GU_SCALES + (size_t)(2*r+1)*SC_GU*2, a1, SC_GU);
        ws.ff[r] = (g / (1.f + std::exp(-g))) * u;
    }
    quant(ws.ff, FF, ws.a2);
    for (int r = 0; r < H; ++r) out[r] = row_dot(blob + O_D_CODES + (size_t)r*ROW_D, blob + O_D_SCALES + (size_t)r*SC_D*2, ws.a2, SC_D);
}
}

const char *CpuFeatures::reason() const { return "ARM64 scalar fallback"; }
CpuFeatures cpu_features() { return {}; }
void cpu_require_expert_support() {}
void expert_set_oracle_q8_0(bool enabled) { g_oracle.store(enabled); }
bool expert_oracle_q8_0_enabled() { return g_oracle.load(); }
void act_quant_q8_1(const float *x, int n, ActQ &a) { quant(x, n, a); }
void s2_expert_vnni(const uint8_t *blob, const float *x, float *out, ExpertScratch &ws) { quant(x, H, ws.a1); one(blob, ws.a1, out, ws); }
void s2_expert_vnni_q(const uint8_t *blob, const ActQ &a1, float *out, ExpertScratch &ws) { one(blob, a1, out, ws); }
void s2_expert_scalar(const uint8_t *blob, const float *x, float *out, bool) { ExpertScratch ws; s2_expert_vnni(blob, x, out, ws); }

void s2_expert_gu_rows(const uint8_t *b, const ActQ &a, float *ff, int r0, int r1) { for (int r=r0;r<r1;++r) {
    float g=row_dot(b+O_GU_CODES+(size_t)(2*r)*ROW_GU,b+O_GU_SCALES+(size_t)(2*r)*SC_GU*2,a,SC_GU);
    float u=row_dot(b+O_GU_CODES+(size_t)(2*r+1)*ROW_GU,b+O_GU_SCALES+(size_t)(2*r+1)*SC_GU*2,a,SC_GU); ff[r]=(g/(1+std::exp(-g)))*u; } }
void s2_expert_down_rows(const uint8_t *b, const ActQ &a, float *o, int r0, int r1) { for (int r=r0;r<r1;++r) o[r]=row_dot(b+O_D_CODES+(size_t)r*ROW_D,b+O_D_SCALES+(size_t)r*SC_D*2,a,SC_D); }

void s2_expert_gu_rows_multi(const uint8_t *b, const ActQ *const *a, int n, float *const *ff, int r0, int r1) { for (int t=0;t<n;++t) s2_expert_gu_rows(b,*a[t],ff[t],r0,r1); }
void s2_expert_down_rows_multi(const uint8_t *b, const ActQ *const *a, int n, float *const *o, int r0, int r1) { for (int t=0;t<n;++t) s2_expert_down_rows(b,*a[t],o[t],r0,r1); }
void s2_expert_vnni_multi(const uint8_t *b, const ActQ *const *a, int n, float *const *o, ExpertScratchMulti &ws) { for (int t=0;t<n;++t) one(b,*a[t],o[t],ws.single); }

void q2_0_gguf_rows_multi(const uint8_t *w, size_t row_bytes, int blocks, const ActQ *const *a, int nt, float *const *out, int r0, int r1) {
    for (int r=r0;r<r1;++r) for (int t=0;t<nt;++t) {
        const uint8_t *row = w + (size_t)r * row_bytes; float v = 0.f;
        for (int b=0;b<blocks;++b) { const uint8_t *blk = row + (size_t)b * 18; const float d = h2f(blk);
            for (int j=0;j<QK;++j) { int q=((blk[2+(j>>2)]>>(2*(j&3)))&3)-1;
                v += q*d*a[t]->scale[2*b+j/QKA]*(float)a[t]->q[2*b*QKA+j]; } }
        out[t][r] = v;
    }
}
void q2_0_gguf_rows_multi_avx2(const uint8_t *w, size_t rb, int nb, const ActQ *const *a, int nt, float *const *o, int r0, int r1) { q2_0_gguf_rows_multi(w,rb,nb,a,nt,o,r0,r1); }
void act_quant_q8_1_avx2(const float *x, int n, ActQ &a) { quant(x,n,a); }
}
