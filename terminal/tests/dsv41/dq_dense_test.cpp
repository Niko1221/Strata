// Strata. The DeepSeek-V4.1 engine's dense weights on the device against a CPU reference, with no model file.
//
// What it is for: a dense dequantizer or GEMV that is right for the types and shapes the test sets happen to use and wrong
// for another type or another row count (2026-10-31: dq_rows_kernel staged a warp's blocks through a 704-byte shared buffer,
// Q6_K's 840 bytes overflowed it, gfx11 chunked prompts of UD-IQ1_S / UD-IQ1_M gave NaN rows - ds4_iq_test never saw it: its
// matrices are 512 wide, and the staging only runs when a row holds a multiple of 32 sub-blocks, k % 1024 == 0).
//
// What it runs, for every type in the list (random legal blocks: any bit pattern is one once the F16 scales are sane):
//   rows   iq::dequant_rows  (the chunked prompt's dense dequantizer) through BOTH kernels (DS4_DQ_OLD=0 the coalesced
//          dq_rows_kernel2, =1 the row-per-lane dq_rows_kernel, and the device's own default), to F16 and BF16, over shapes
//          that include k % 1024 == 0 (the staged path), odd row counts, and a view that starts mid-matrix.  Every output
//          element must be written (the buffer starts as NaN), must match the CPU decode bit for bit (1 ulp is reported, not
//          failed), the old and the new kernel must agree, and nothing outside the buffer may change.
//   mv     iq::mv  (the GEMV of the token path, ncol 1, and of the verify window, ncol up to 16) against a double reference on
//          the same int8-rounded activations; a column alone must equal the same column in a wider call bit for bit.
//   f16    iq::f16_rows (the busy experts' path) for the types it takes; a refusal is printed, never silent.
//   gguf   (-DDQ_TEST_GGUF) ds4_gguf.cu's own gg::rows and gg::mv on BF16, Q8_0 and ggml-type matrices - the entry points the
//          engine really calls - including the views the chunked prompt cuts (gemm() takes 128 rows at a time).
//   gemm   (-DDQ_TEST_GEMM) the chunk GEMM after the dequantizer: the library GEMM (cuBLAS / hipBLAS) and, on HIP, the
//          engine's own WMMA kernel strata_pf_gemm_f16_ld, against a double reference on the dequantized weights.
//
//   dq_dense_test [--reps N] [--shape KxN,KxN] (repeat path) [--device N] [--type 14,13,...|all] [--kernel new|old|default|all] [--path rows,mv,f16,gguf,gemm] [--quick]
// Prints one RESULT line per (type, path, kernel) and "SUMMARY ... fails=N"; the exit code is 1 when anything failed.
// A kernel that dies with a sticky device error ends the process: run one --type and --kernel a process (run_dq_test.sh does).
#include "../../src/dsv41/ds4_iq.cuh"
#include "strata/dsv41/hip_or_cuda.h"   // HIP names on a HIP build, the CUDA runtime otherwise
#if defined(DQ_TEST_GGUF) || defined(DQ_TEST_GEMM)
#include "../../src/dsv41/ds4_gguf.cuh"
#endif
#if defined(DQ_TEST_GEMM)
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hipblas/hipblas.h>
#include "../../src/prefill/wmma_gemm.h"
#define DQ_HIP 1
#else
#include <cublas_v2.h>
#endif
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace iq = strata::ds4::gguf::iq;
#if defined(DQ_TEST_GGUF) || defined(DQ_TEST_GEMM)
namespace gg = strata::ds4::gguf;
namespace x3 = strata::dsv41k::exl3;
#endif

namespace {
[[noreturn]] void bail(int code);   // prints the RESULT lines so far, then exits
}
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("DEVICE ERROR %s at line %d: %s\n", hipGetErrorString(e_), __LINE__, #x); std::fflush(stdout); bail(3); } } while (0)

namespace {

constexpr int kQ8_0 = 8, kBF16 = 30;   // ggml type ids (the dense matrices that are not in ds4_iq.cuh's list)

// ------------------------------------------------------------------------------------------------ conversions
float h2f(uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    float f = e == 0 ? std::ldexp((float) m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(1.0f + m / 1024.0f, e - 15);
    return s ? -f : f;
}
uint16_t f2h(float f) {   // round to nearest even, subnormals and overflow included
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u;
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) (s | 0x7e00u);
    const float a = std::fabs(f);
    if (a >= 65520.0f) return (uint16_t) (s | 0x7c00u);
    if (a < 6.103515625e-05f) return (uint16_t) (s | (uint32_t) std::nearbyint(std::ldexp((double) a, 24)));
    const int e = (int) ((u >> 23) & 0xff) - 127 + 15;
    uint32_t v = ((uint32_t) e << 10) | ((u & 0x7fffffu) >> 13);
    const uint32_t rem = u & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) ++v;
    return (uint16_t) (s | v);
}
float bf2f(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40u);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
bool is_nan16(uint16_t h, bool bf16) { return bf16 ? ((h & 0x7fffu) > 0x7f80u) : ((h & 0x7c00u) == 0x7c00u && (h & 0x3ffu) != 0); }
bool is_inf16(uint16_t h, bool bf16) { return bf16 ? ((h & 0x7fffu) == 0x7f80u) : ((h & 0x7fffu) == 0x7c00u); }

void set_env(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}

// ------------------------------------------------------------------------------------------------ the types
struct TypeInfo {
    int id;
    const char* name;
    int bv, bb;       // values and bytes of a block
    bool ref;         // a CPU decode below
    bool dense;       // a type the DeepSeek loader accepts for a dense matrix (ds4_model.cu mat_bf16: Q4_K, Q5_K, Q6_K, Q8_0, BF16)
    bool rows_api;    // iq::dequant_rows takes it
};
const TypeInfo kTypes[] = {
    {iq::kQ6_K, "Q6_K", 256, 210, true, true, true},
    {iq::kQ5_K, "Q5_K", 256, 176, true, true, true},
    {iq::kQ4_K, "Q4_K", 256, 144, true, true, true},
    {iq::kQ3_K, "Q3_K", 256, 110, true, false, true},
    {iq::kQ2_K, "Q2_K", 256, 84, true, false, true},
    {iq::kIQ2_XXS, "IQ2_XXS", 256, 66, false, false, true},
    {iq::kIQ2_XS, "IQ2_XS", 256, 74, false, false, true},
    {iq::kIQ2_S, "IQ2_S", 256, 82, false, false, true},
    {iq::kIQ3_XXS, "IQ3_XXS", 256, 98, false, false, true},
    {iq::kIQ3_S, "IQ3_S", 256, 110, false, false, true},
    {iq::kIQ4_XS, "IQ4_XS", 256, 136, false, false, true},
    {iq::kIQ1_S, "IQ1_S", 256, 50, false, false, true},
    {iq::kIQ1_M, "IQ1_M", 256, 56, false, false, true},
    {iq::kIQ4_NL, "IQ4_NL", 32, 18, false, false, false},   // (dequant_rows has no case for it; the engram table reads it elsewhere)
    {kQ8_0, "Q8_0", 32, 34, true, true, false},              // (ds4_gguf.cu's q8 kernels, the gguf path)
    {kBF16, "BF16", 1, 2, true, true, false},                // (ds4_gguf.cu's bf16 kernels, the gguf path)
};
const TypeInfo* find_type(int id) {
    for (const auto& t : kTypes) if (t.id == id) return &t;
    return nullptr;
}
const TypeInfo* find_type_name(const std::string& n) {
    for (const auto& t : kTypes) if (n == t.name) return &t;
    return nullptr;
}

// F16 scales at the right offsets, sane (0.02 .. 0.14): a random scale is NaN / Inf one time in 32
void put_h(uint8_t* p, uint16_t h) { std::memcpy(p, &h, 2); }
void sane_scales(int type, uint8_t* p, std::mt19937_64& rng) {
    auto d = [&] { return f2h(0.02f + 0.0002f * (float) (rng() % 600)); };
    switch (type) {
        case iq::kQ2_K: put_h(p + 80, d()); put_h(p + 82, d()); break;
        case iq::kQ3_K: put_h(p + 108, d()); break;
        case iq::kQ4_K: case iq::kQ5_K: put_h(p, d()); put_h(p + 2, d()); break;
        case iq::kQ6_K: put_h(p + 208, d()); break;
        case iq::kIQ1_M: {   // no d field: the top nibbles of the four u16 scales
            const uint16_t h = d();
            p[49] = (uint8_t) ((p[49] & 0x0f) | ((h & 0x000f) << 4));
            p[51] = (uint8_t) ((p[51] & 0x0f) | (h & 0x00f0));
            p[53] = (uint8_t) ((p[53] & 0x0f) | ((h & 0x0f00) >> 4));
            p[55] = (uint8_t) ((p[55] & 0x0f) | ((h & 0xf000) >> 8));
            break;
        }
        default: put_h(p, d()); break;   // IQ*, Q8_0: d first
    }
}
std::vector<uint8_t> make_bytes(int type, size_t nblk, std::mt19937_64& rng) {
    const TypeInfo& t = *find_type(type);
    std::vector<uint8_t> v(nblk * (size_t) t.bb);
    for (auto& b : v) b = (uint8_t) rng();
    if (type == kBF16) {   // random normal BF16 values
        std::normal_distribution<float> nd(0.f, 0.05f);
        for (size_t i = 0; i < nblk; ++i) { const uint16_t b = f2bf(nd(rng)); std::memcpy(&v[i * 2], &b, 2); }
        return v;
    }
    for (size_t b = 0; b < nblk; ++b) sane_scales(type, &v[b * (size_t) t.bb], rng);
    return v;
}

// ------------------------------------------------------------------------------------------------ CPU decode (ggml's dequantize_row_*, same float operations in the same order)
void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
float rd_h(const uint8_t* p) { uint16_t h; std::memcpy(&h, p, 2); return h2f(h); }
// one block of the type -> bv values
void decode_block(int type, const uint8_t* b, float* y) {
    if (type == kQ8_0) {
        const float d = rd_h(b);
        for (int j = 0; j < 32; ++j) y[j] = d * (float) (int8_t) b[2 + j];
    } else if (type == iq::kQ4_K) {
        const float d = rd_h(b), mn = rd_h(b + 2);
        const uint8_t* q = b + 16;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            scale_min_k4(is + 0, b + 4, sc, m);
            const float d1 = d * sc, m1 = mn * m;
            scale_min_k4(is + 1, b + 4, sc, m);
            const float d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    } else if (type == iq::kQ5_K) {
        const float d = rd_h(b), mn = rd_h(b + 2);
        const uint8_t* qh = b + 16;
        const uint8_t* ql = b + 48;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            scale_min_k4(is + 0, b + 4, sc, m);
            const float d1 = d * sc, m1 = mn * m;
            scale_min_k4(is + 1, b + 4, sc, m);
            const float d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    } else if (type == iq::kQ6_K) {
        const float d = rd_h(b + 208);
        const uint8_t* ql = b;
        const uint8_t* qh = b + 128;
        const int8_t* sc = (const int8_t*) (b + 192);
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = (int8_t) ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int8_t) ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    } else if (type == iq::kQ2_K) {
        const float d = rd_h(b + 80), mn = rd_h(b + 82);
        const uint8_t* q = b + 16;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = b[is++];
                float dl = d * (sc & 0xF), ml = mn * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t) ((q[l] >> shift) & 3)) - ml;
                sc = b[is++];
                dl = d * (sc & 0xF);
                ml = mn * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t) ((q[l + 16] >> shift) & 3)) - ml;
                shift += 2;
            }
            q += 32;
        }
    } else if (type == iq::kQ3_K) {
        const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
        const float d_all = rd_h(b + 108);
        const uint8_t* q = b + 32;
        const uint8_t* hm = b;
        uint8_t m = 1;
        uint32_t aux[4];
        std::memcpy(aux, b + 96, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = (const int8_t*) aux;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t) ((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t) ((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}
// a row of k values (BF16: k values of 2 bytes)
void decode_row(int type, const uint8_t* row, int k, float* y) {
    if (type == kBF16) {
        for (int i = 0; i < k; ++i) { uint16_t b; std::memcpy(&b, row + 2 * i, 2); y[i] = bf2f(b); }
        return;
    }
    const TypeInfo& t = *find_type(type);
    for (int b = 0; b < k / t.bv; ++b) decode_block(type, row + (size_t) b * t.bb, y + (size_t) b * t.bv);
}
size_t row_bytes(int type, int k) { const TypeInfo& t = *find_type(type); return (size_t) (k / t.bv) * (size_t) t.bb; }

// the engine's q8_quant: sc = max|x| / 127 per 32, q = round-half-even(x / sc)
struct Q8 { std::vector<int8_t> q; std::vector<float> d, xhat; };
Q8 quant(const float* x, int k) {
    Q8 r;
    r.q.resize(k); r.d.resize(k / 32); r.xhat.resize(k);
    for (int b = 0; b < k / 32; ++b) {
        float am = 0;
        for (int j = 0; j < 32; ++j) am = std::max(am, std::fabs(x[b * 32 + j]));
        const float sc = am / 127.0f;
        r.d[b] = sc;
        for (int j = 0; j < 32; ++j) {
            const int q = am > 0 ? (int) std::nearbyint(x[b * 32 + j] / sc) : 0;
            r.q[b * 32 + j] = (int8_t) q;
            r.xhat[b * 32 + j] = (float) q * sc;
        }
    }
    return r;
}

// ------------------------------------------------------------------------------------------------ device helpers
template <typename T> T* up(const std::vector<T>& v) {
    T* d;
    CK(hipMalloc(&d, std::max<size_t>(16, v.size() * sizeof(T))));
    if (!v.empty()) CK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> v(n);
    if (n) CK(hipMemcpy(v.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
}
void fill16(uint16_t* d, size_t n, uint16_t val) {
    std::vector<uint16_t> v(n, val);
    CK(hipMemcpy(d, v.data(), n * 2, hipMemcpyHostToDevice));
}

// ------------------------------------------------------------------------------------------------ results
struct Tally {
    int cases = 0, fail = 0, nan = 0, mism = 0, ulp = 0, unwritten = 0, refused = 0, diff = 0, guard = 0, coldiff = 0;
    std::string first;
};
std::map<std::string, Tally> g_res;
int g_fails = 0;
bool g_verbose = false;

void note_fail(Tally& t, const std::string& what) {
    ++t.fail;
    ++g_fails;
    if (t.first.empty()) t.first = what;
    std::printf("  FAIL %s\n", what.c_str());
}
std::string key(const char* path, const char* type, const char* kernel) { return std::string(type) + " " + path + " " + kernel; }

// ------------------------------------------------------------------------------------------------ rows: iq::dequant_rows
struct Kern { const char* name; const char* env; };
const Kern kKerns[] = {{"new", "0"}, {"old", "1"}, {"default", nullptr}};

// compare an n x k output with the reference, element by element
void compare_rows(Tally& t, const std::string& tag, const std::vector<uint16_t>& out, const std::vector<float>& ref, bool have_ref, bool bf16) {
    size_t nan = 0, inf = 0, unw = 0, bad = 0, ulp = 0;
    size_t first_bad = (size_t) -1;
    for (size_t i = 0; i < out.size(); ++i) {
        const uint16_t o = out[i];
        if (o == 0xFFFFu) { ++unw; if (first_bad == (size_t) -1) first_bad = i; continue; }   // never written: the buffer started as NaN
        if (is_nan16(o, bf16)) { ++nan; if (first_bad == (size_t) -1) first_bad = i; continue; }
        if (is_inf16(o, bf16)) { ++inf; if (first_bad == (size_t) -1) first_bad = i; continue; }
        if (!have_ref) continue;
        const uint16_t want = bf16 ? f2bf(ref[i]) : f2h(ref[i]);
        if (o == want) continue;
        const double dv = bf16 ? bf2f(o) : h2f(o), wv = bf16 ? bf2f(want) : h2f(want);
        // one step of the format at the reference
        const double step = bf16 ? std::ldexp(1.0, std::ilogb(std::max(1e-30, std::fabs(wv))) - 7) : std::max(std::ldexp(1.0, std::ilogb(std::max(1e-30, std::fabs(wv))) - 10), std::ldexp(1.0, -24));
        if (std::fabs(dv - wv) <= step * 1.0001) { ++ulp; continue; }
        ++bad;
        if (first_bad == (size_t) -1) first_bad = i;
    }
    t.nan += (int) (nan + inf);
    t.unwritten += (int) unw;
    t.mism += (int) bad;
    t.ulp += (int) ulp;
    if (nan + inf + unw + bad > 0) {
        char b[400];
        std::snprintf(b, sizeof b, "%s: %zu NaN, %zu Inf, %zu unwritten, %zu mismatches of %zu (first at %zu: got 0x%04x%s)", tag.c_str(), nan, inf, unw, bad,
                      out.size(), first_bad, first_bad < out.size() ? out[first_bad] : 0u,
                      have_ref && first_bad < ref.size() ? (std::string(" want ") + std::to_string(ref[first_bad])).c_str() : "");
        note_fail(t, b);
    }
}

void run_rows(const TypeInfo& ty, const Kern& kn, bool quick) {
    if (!ty.rows_api) return;
    const int ks[] = {256, 512, 768, 1024, 1280, 2048, 2304, 4096, 5120, 8192};
    const int ns[] = {1, 2, 7, 31, 32, 33, 128, 129, 257};
    const int offs[] = {0, 1};   // rows before the view (the chunk path hands over raw + c0 ld)
    for (bool bf16 : {false, true}) {
        Tally& t = g_res[key(bf16 ? "rows-bf16" : "rows-f16", ty.name, kn.name)];
        for (int k : ks) {
            if (quick && !(k == 512 || k == 1024 || k == 4096 || k == 5120)) continue;
            for (int n : ns) {
                if ((size_t) n * k > (size_t) 4 << 20) continue;
                if (quick && !(n == 1 || n == 33 || n == 129)) continue;
                for (int off : offs) {
                    if (off && !(k % 1024 == 0 && (n == 7 || n == 33))) continue;
                    std::mt19937_64 rng((uint64_t) ty.id * 1000003 + (uint64_t) k * 131 + (uint64_t) n * 7 + (uint64_t) off);
                    const size_t rb = row_bytes(ty.id, k);
                    const size_t nblk = (size_t) (n + off) * (size_t) (k / ty.bv);
                    std::vector<uint8_t> raw = make_bytes(ty.id, nblk, rng);
                    uint8_t* d_raw = up(raw);
                    const size_t cnt = (size_t) n * k, guard = 2048;
                    uint16_t* d_buf;
                    CK(hipMalloc(&d_buf, (cnt + 2 * guard) * 2));
                    fill16(d_buf, cnt + 2 * guard, 0xFFFFu);
                    set_env("DS4_DQ_OLD", kn.env);
                    const bool ok = iq::dequant_rows(ty.id, d_raw + (size_t) off * rb, rb, k, n, d_buf + guard, bf16, 0);
                    hipError_t e = hipDeviceSynchronize();
                    ++t.cases;
                    char tag[160];
                    std::snprintf(tag, sizeof tag, "%s %s kernel=%s k=%d n=%d off=%d", ty.name, bf16 ? "bf16" : "f16", kn.name, k, n, off);
                    if (e != hipSuccess) {
                        std::printf("STICKY DEVICE ERROR in %s: %s\n", tag, hipGetErrorString(e));
                        note_fail(t, std::string(tag) + ": device error " + hipGetErrorString(e));
                        std::printf("RESULT %s cases=%d fail=%d STICKY\n", key("rows", ty.name, kn.name).c_str(), t.cases, t.fail);
                        bail(3);
                    }
                    if (!ok) {
                        ++t.refused;
                        if (ty.dense) note_fail(t, std::string(tag) + ": dequant_rows refused a dense type");
                        else std::printf("  note: %s refused by dequant_rows\n", tag);
                    } else {
                        const auto all = down(d_buf, cnt + 2 * guard);
                        for (size_t i = 0; i < guard; ++i)
                            if (all[i] != 0xFFFFu || all[guard + cnt + i] != 0xFFFFu) { ++t.guard; note_fail(t, std::string(tag) + ": wrote outside the buffer"); break; }
                        std::vector<uint16_t> out(all.begin() + guard, all.begin() + guard + cnt);
                        std::vector<float> ref;
                        if (ty.ref) {
                            ref.resize(cnt);
                            for (int r = 0; r < n; ++r) decode_row(ty.id, &raw[(size_t) (r + off) * rb], k, &ref[(size_t) r * k]);
                        }
                        compare_rows(t, tag, out, ref, ty.ref, bf16);
                        // old against new on the same bytes: the values are meant to be the same bit for bit
                        if (!std::strcmp(kn.name, "old") || !std::strcmp(kn.name, "default")) {
                            set_env("DS4_DQ_OLD", "0");
                            uint16_t* d_b2;
                            CK(hipMalloc(&d_b2, cnt * 2 + 64));
                            fill16(d_b2, cnt, 0xFFFFu);
                            iq::dequant_rows(ty.id, d_raw + (size_t) off * rb, rb, k, n, d_b2, bf16, 0);
                            CK(hipDeviceSynchronize());
                            const auto o2 = down(d_b2, cnt);
                            size_t nd = 0;
                            for (size_t i = 0; i < cnt; ++i) nd += o2[i] != out[i];
                            if (nd) { ++t.diff; note_fail(t, std::string(tag) + ": differs from kernel2 in " + std::to_string(nd) + " elements"); }
                            CK(hipFree(d_b2));
                        }
                    }
                    CK(hipFree(d_buf));
                    CK(hipFree(d_raw));
                    set_env("DS4_DQ_OLD", nullptr);
                }
            }
        }
    }
}


// ------------------------------------------------------------------------------------------------ repeat: the same call again and again
// A kernel that stages through shared memory with a missing barrier or a fence can pass one run and not the next.  Each shape is run
// `reps` times into a buffer that starts as NaN; every run must equal the CPU decode (1 ulp reported) and every run must equal the
// first one bit for bit.  Reports the first wrong element and the run it showed in.
std::vector<std::pair<int, int>> g_rep_shapes;   // --shape KxN,...: the repeat path's shapes instead of the default list
int g_reps = 0;                                  // --reps N
void run_repeat(const TypeInfo& ty, const Kern& kn, bool quick) {
    if (!ty.rows_api || !ty.ref) return;
    Tally& t = g_res[key("repeat", ty.name, kn.name)];
    const int reps = g_reps > 0 ? g_reps : quick ? 5 : 25;
    std::vector<std::pair<int, int>> shapes = {{1024, 300}, {4096, 2048}, {5120, 1000}, {8192, 700}, {1024, 4097}};
    if (!g_rep_shapes.empty()) shapes = g_rep_shapes;
    for (const auto& sh : shapes) {
        const int k = sh.first, n = sh.second;
        for (bool bf16 : {false, true}) {
            std::mt19937_64 rng((uint64_t) ty.id * 4243 + (uint64_t) k + (uint64_t) n * 3);
            const size_t rb = row_bytes(ty.id, k);
            auto raw = make_bytes(ty.id, (size_t) n * (size_t) (k / ty.bv), rng);
            uint8_t* d_raw = up(raw);
            const size_t cnt = (size_t) n * k;
            uint16_t* d_out;
            CK(hipMalloc(&d_out, cnt * 2 + 64));
            std::vector<float> ref(cnt);
            for (int r = 0; r < n; ++r) decode_row(ty.id, &raw[(size_t) r * rb], k, &ref[(size_t) r * k]);
            std::vector<uint16_t> first;
            set_env("DS4_DQ_OLD", kn.env);
            for (int rep = 0; rep < reps; ++rep) {
                fill16(d_out, cnt, 0xFFFFu);
                const bool ok = iq::dequant_rows(ty.id, d_raw, rb, k, n, d_out, bf16, 0);
                hipError_t e = hipDeviceSynchronize();
                ++t.cases;
                char tag[200];
                std::snprintf(tag, sizeof tag, "%s %s kernel=%s k=%d n=%d run %d of %d", ty.name, bf16 ? "bf16" : "f16", kn.name, k, n, rep + 1, reps);
                if (e != hipSuccess) { std::printf("STICKY DEVICE ERROR in %s: %s\n", tag, hipGetErrorString(e)); bail(3); }
                if (!ok) { note_fail(t, std::string(tag) + ": refused"); break; }
                const auto out = down(d_out, cnt);
                if (rep == 0) { first = out; compare_rows(t, tag, out, ref, true, bf16); }
                else {
                    size_t nd = 0, at = (size_t) -1;
                    for (size_t i = 0; i < cnt; ++i) if (out[i] != first[i]) { if (at == (size_t) -1) at = i; ++nd; }
                    if (nd) { ++t.diff; note_fail(t, std::string(tag) + ": " + std::to_string(nd) + " elements differ from run 1, first at " + std::to_string(at) + " (row " + std::to_string(at / k) + ", col " + std::to_string(at % k) + ")"); break; }
                }
            }
            set_env("DS4_DQ_OLD", nullptr);
            CK(hipFree(d_out));
            CK(hipFree(d_raw));
        }
    }
}

// ------------------------------------------------------------------------------------------------ mv: iq::mv
void run_mv(const TypeInfo& ty, bool quick) {
    if (!iq::supported(ty.id) || ty.id == iq::kIQ4_NL) return;
    const int ks[] = {256, 1024, 4096, 5120, 8192};
    const int ns[] = {1, 5, 8, 9, 64, 129};
    const int ncs[] = {1, 2, 4, 5, 8, 16};
    Tally& t = g_res[key("mv", ty.name, "-")];
    std::mt19937_64 xr(77);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int k : ks) {
        if (quick && !(k == 1024 || k == 5120)) continue;
        for (int n : ns) {
            if (quick && !(n == 1 || n == 129)) continue;
            std::mt19937_64 rng((uint64_t) ty.id * 7919 + (uint64_t) k * 17 + (uint64_t) n);
            const size_t rb = row_bytes(ty.id, k);
            std::vector<uint8_t> raw = make_bytes(ty.id, (size_t) n * (size_t) (k / ty.bv), rng);
            uint8_t* d_raw = up(raw);
            std::vector<float> wf;
            if (ty.ref) {
                wf.resize((size_t) n * k);
                for (int r = 0; r < n; ++r) decode_row(ty.id, &raw[(size_t) r * rb], k, &wf[(size_t) r * k]);
            }
            std::vector<float> bias(n);
            for (auto& b : bias) b = nd(xr);
            float* d_bias = up(bias);
            for (int nc : ncs) {
                if (quick && !(nc == 1 || nc == 5 || nc == 16)) continue;
                std::vector<float> x((size_t) nc * k);
                for (auto& v : x) v = nd(xr) * (1.0f + 3.0f * (xr() % 11 == 0));
                std::vector<int8_t> xq((size_t) nc * k);
                std::vector<float> xd((size_t) nc * k / 32);
                std::vector<Q8> qx(nc);
                for (int c = 0; c < nc; ++c) {
                    qx[c] = quant(&x[(size_t) c * k], k);
                    std::memcpy(&xq[(size_t) c * k], qx[c].q.data(), k);
                    std::memcpy(&xd[(size_t) c * k / 32], qx[c].d.data(), (size_t) k / 32 * 4);
                }
                int8_t* d_xq = up(xq);
                float* d_xd = up(xd);
                float* d_y;
                const size_t ycs = (size_t) n + 3;
                CK(hipMalloc(&d_y, (size_t) nc * ycs * 4));
                std::vector<float> nanv((size_t) nc * ycs, NAN);
                CK(hipMemcpy(d_y, nanv.data(), nanv.size() * 4, hipMemcpyHostToDevice));
                iq::MvJob j;
                j.w = d_raw; j.ld = rb; j.type = ty.id; j.k = k; j.n = n; j.ncol = nc; j.xq = d_xq; j.xd = d_xd; j.y = d_y; j.ycs = ycs;
                j.bias = d_bias; j.alpha = 0.5f;
                const bool ok = iq::mv(&j, 1, 0);
                hipError_t e = hipDeviceSynchronize();
                ++t.cases;
                char tag[160];
                std::snprintf(tag, sizeof tag, "%s mv k=%d n=%d ncol=%d", ty.name, k, n, nc);
                if (e != hipSuccess) {
                    std::printf("STICKY DEVICE ERROR in %s: %s\n", tag, hipGetErrorString(e));
                    bail(3);
                }
                if (!ok) {
                    ++t.refused;
                    note_fail(t, std::string(tag) + ": iq::mv refused");
                } else {
                    const auto y = down(d_y, (size_t) nc * ycs);
                    size_t nanc = 0, bad = 0;
                    double worst = 0;
                    for (int c = 0; c < nc; ++c)
                        for (int r = 0; r < n; ++r) {
                            const float v = y[(size_t) c * ycs + r];
                            if (!std::isfinite(v)) { ++nanc; continue; }
                            if (!ty.ref) continue;
                            double s = 0, sa = 0;
                            for (int i = 0; i < k; ++i) { const double tt = (double) wf[(size_t) r * k + i] * qx[c].xhat[i]; s += tt; sa += std::fabs(tt); }
                            const double refv = 0.5 * s + bias[r];
                            const double err = std::fabs((double) v - refv) / (0.5 * sa + std::fabs((double) bias[r]) + 1e-30);
                            worst = std::max(worst, err);
                            if (err > 2e-5) ++bad;
                        }
                    t.nan += (int) nanc;
                    t.mism += (int) bad;
                    if (nanc || bad) {
                        char b[300];
                        std::snprintf(b, sizeof b, "%s: %zu non-finite, %zu beyond 2e-5 (worst %.2e)", tag, nanc, bad, worst);
                        note_fail(t, b);
                    }
                    // the same column alone: a call of ncol 1 on column c's activations must give column c's bits
                    for (int c : {0, nc - 1}) {
                        iq::MvJob j1 = j;
                        j1.ncol = 1;
                        j1.xq = d_xq + (size_t) c * k;
                        j1.xd = d_xd + (size_t) c * k / 32;
                        float* d_y1;
                        CK(hipMalloc(&d_y1, (size_t) n * 4 + 64));
                        j1.y = d_y1; j1.ycs = (size_t) n;
                        iq::mv(&j1, 1, 0);
                        CK(hipDeviceSynchronize());
                        const auto y1 = down(d_y1, (size_t) n);
                        size_t nd2 = 0;
                        for (int r = 0; r < n; ++r) nd2 += std::memcmp(&y1[r], &y[(size_t) c * ycs + r], 4) != 0;
                        if (nd2) { ++t.coldiff; note_fail(t, std::string(tag) + ": column " + std::to_string(c) + " alone differs from the same column in the wide call (" + std::to_string(nd2) + " rows)"); }
                        CK(hipFree(d_y1));
                    }
                }
                CK(hipFree(d_y)); CK(hipFree(d_xq)); CK(hipFree(d_xd));
            }
            CK(hipFree(d_raw)); CK(hipFree(d_bias));
        }
    }
    // two jobs of one type in one call (the shared expert's gate and up, wo_a's groups)
    {
        const int k = 1024, n1 = 33, n2 = 70, nc = 3;
        std::mt19937_64 rng(5 + (uint64_t) ty.id);
        const size_t rb = row_bytes(ty.id, k);
        auto r1 = make_bytes(ty.id, (size_t) n1 * (k / ty.bv), rng), r2 = make_bytes(ty.id, (size_t) n2 * (k / ty.bv), rng);
        uint8_t *d1 = up(r1), *d2 = up(r2);
        std::vector<float> x((size_t) nc * k);
        for (auto& v : x) v = nd(xr);
        std::vector<int8_t> xq((size_t) nc * k);
        std::vector<float> xd((size_t) nc * k / 32);
        std::vector<Q8> qx(nc);
        for (int c = 0; c < nc; ++c) {
            qx[c] = quant(&x[(size_t) c * k], k);
            std::memcpy(&xq[(size_t) c * k], qx[c].q.data(), k);
            std::memcpy(&xd[(size_t) c * k / 32], qx[c].d.data(), (size_t) k / 32 * 4);
        }
        int8_t* dq = up(xq);
        float* dd = up(xd);
        float *y1, *y2;
        CK(hipMalloc(&y1, (size_t) nc * n1 * 4)); CK(hipMalloc(&y2, (size_t) nc * n2 * 4));
        iq::MvJob js[2];
        js[0].w = d1; js[0].ld = rb; js[0].type = ty.id; js[0].k = k; js[0].n = n1; js[0].ncol = nc; js[0].xq = dq; js[0].xd = dd; js[0].y = y1; js[0].ycs = n1;
        js[1] = js[0]; js[1].w = d2; js[1].n = n2; js[1].y = y2; js[1].ycs = n2;
        ++t.cases;
        if (!iq::mv(js, 2, 0)) note_fail(t, std::string(ty.name) + " mv two jobs: refused");
        else {
            CK(hipDeviceSynchronize());
            const auto a = down(y1, (size_t) nc * n1), b = down(y2, (size_t) nc * n2);
            size_t nanc = 0;
            for (float v : a) nanc += !std::isfinite(v);
            for (float v : b) nanc += !std::isfinite(v);
            t.nan += (int) nanc;
            if (nanc) note_fail(t, std::string(ty.name) + " mv two jobs: non-finite output");
        }
        CK(hipFree(d1)); CK(hipFree(d2)); CK(hipFree(dq)); CK(hipFree(dd)); CK(hipFree(y1)); CK(hipFree(y2));
    }
}

// ------------------------------------------------------------------------------------------------ f16_rows (the busy experts' path)
void run_f16(const TypeInfo& ty) {
    if (!ty.rows_api) return;
    Tally& t = g_res[key("f16rows", ty.name, "-")];
    const bool takes = iq::f16_rows_ok(ty.id);
    std::mt19937_64 rng(31 + (uint64_t) ty.id);
    for (int k : {256, 1024, 4096}) for (int n : {1, 9, 64}) {
        const size_t rb = row_bytes(ty.id, k);
        auto raw = make_bytes(ty.id, (size_t) n * (k / ty.bv), rng);
        uint8_t* d_raw = up(raw);
        uint16_t* d_o;
        CK(hipMalloc(&d_o, (size_t) n * k * 2 + 64));
        fill16(d_o, (size_t) n * k, 0xFFFFu);
        ++t.cases;
        const bool ok = iq::f16_rows(ty.id, d_raw, k, n, d_o, k, 0);
        CK(hipDeviceSynchronize());
        char tag[120];
        std::snprintf(tag, sizeof tag, "%s f16_rows k=%d n=%d", ty.name, k, n);
        if (!ok) {
            ++t.refused;
            if (takes) note_fail(t, std::string(tag) + ": refused although f16_rows_ok");
            else if (t.refused == 1) std::printf("  note: %s: f16_rows_ok is false and f16_rows refuses it (the caller must not send it there)\n", ty.name);
        } else {
            if (!takes) note_fail(t, std::string(tag) + ": accepted although f16_rows_ok is false");
            const auto out = down(d_o, (size_t) n * k);
            std::vector<float> ref;
            if (ty.ref) {
                ref.resize((size_t) n * k);
                for (int r = 0; r < n; ++r) decode_row(ty.id, &raw[(size_t) r * rb], k, &ref[(size_t) r * k]);
            }
            compare_rows(t, tag, out, ref, ty.ref, false);
        }
        CK(hipFree(d_o)); CK(hipFree(d_raw));
    }
}

#if defined(DQ_TEST_GGUF) || defined(DQ_TEST_GEMM)
// ------------------------------------------------------------------------------------------------ gguf: ds4_gguf.cu's entry points on Mat
x3::Mat make_mat(const TypeInfo& ty, const uint8_t* dev, int k, int n) {
    x3::Mat m{};
    if (ty.id == kBF16) { m.fmt = gg::kFmtBF16; m.ld = k; }
    else if (ty.id == kQ8_0) { m.fmt = gg::kFmtQ8_0; m.ld = (int) row_bytes(ty.id, k); }
    else { m.fmt = gg::kFmtGG; m.K = ty.id; m.ld = (int) row_bytes(ty.id, k); }
    m.raw = dev;
    m.k = k;
    m.n = n;
    return m;
}
#endif

#if defined(DQ_TEST_GGUF)
void run_gguf(const TypeInfo& ty, const Kern& kn, bool quick) {
    if (!ty.dense && ty.id != iq::kQ3_K && ty.id != iq::kQ2_K) return;   // (the dense types, and the two K-quants that share their kernels)
    Tally& t = g_res[key("gguf-rows", ty.name, kn.name)];
    const int ks[] = {256, 1024, 2048, 4096, 5120, 8192};
    const int ns[] = {1, 33, 130, 257, 640};
    // rows(): the chunk path's views: gemm() cuts the matrix in runs of `cols` rows, 128-aligned
    for (int k : ks) {
        if (ty.id == kQ8_0 || ty.id == kBF16) { /* any k % 32 */ } else if (k % 256) continue;
        if (quick && !(k == 1024 || k == 5120)) continue;
        for (int n : ns) {
            if ((size_t) n * k > (size_t) 4 << 20) continue;
            if (quick && !(n == 33 || n == 257)) continue;
            for (bool bf16 : {false, true}) {
                if (ty.id == kBF16 && false) continue;
                std::mt19937_64 rng((uint64_t) ty.id * 31337 + (uint64_t) k + (uint64_t) n * 11);
                const size_t rb = row_bytes(ty.id, k);
                auto raw = make_bytes(ty.id, (size_t) n * (k / ty.bv), rng);
                uint8_t* d_raw = up(raw);
                x3::Mat m = make_mat(ty, d_raw, k, n);
                uint16_t* d_o;
                CK(hipMalloc(&d_o, (size_t) n * k * 2 + 64));
                fill16(d_o, (size_t) n * k, 0xFFFFu);
                set_env("DS4_DQ_OLD", kn.env);
                const int e0 = gg::launch_errors();
                // whole, then in views of 128-row runs the way gemm() does
                for (int c0 = 0; c0 < n; c0 += 128) {
                    const int nc = std::min(128, n - c0);
                    gg::rows(nc == n ? m : gg::view(m, c0, nc), d_o + (size_t) c0 * k, bf16, 0);
                }
                CK(hipDeviceSynchronize());
                ++t.cases;
                char tag[160];
                std::snprintf(tag, sizeof tag, "%s gg::rows %s kernel=%s k=%d n=%d", ty.name, bf16 ? "bf16" : "f16", kn.name, k, n);
                if (gg::launch_errors() != e0) note_fail(t, std::string(tag) + ": gg::rows reported an error");
                const auto out = down(d_o, (size_t) n * k);
                std::vector<float> ref((size_t) n * k);
                for (int r = 0; r < n; ++r) decode_row(ty.id, &raw[(size_t) r * rb], k, &ref[(size_t) r * k]);
                compare_rows(t, tag, out, ref, true, bf16);
                CK(hipFree(d_o)); CK(hipFree(d_raw));
                set_env("DS4_DQ_OLD", nullptr);
            }
        }
    }
    // mv(): the GEMV through gg::mv (BF16, Q8_0 and ggml types), ncol 1..16
    Tally& tm = g_res[key("gguf-mv", ty.name, "-")];
    std::mt19937_64 xr(91);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int mks[] = {256, 288, 1024, 4096, 5120};
    for (int k : mks) {
        if (ty.id != kQ8_0 && ty.id != kBF16 && k % 256) continue;
        if (ty.id == kBF16 && (k % 8)) continue;
        if (quick && !(k == 288 || k == 5120)) continue;
        for (int n : {1, 8, 33, 129}) {
            if (quick && n != 129 && n != 1) continue;
            for (int nc : {1, 2, 3, 4, 8, 16}) {
                if (quick && !(nc == 1 || nc == 4 || nc == 16)) continue;
                std::mt19937_64 rng((uint64_t) ty.id * 101 + (uint64_t) k * 3 + (uint64_t) n);
                const size_t rb = row_bytes(ty.id, k);
                auto raw = make_bytes(ty.id, (size_t) n * (k / ty.bv), rng);
                uint8_t* d_raw = up(raw);
                x3::Mat m = make_mat(ty, d_raw, k, n);
                std::vector<float> x((size_t) nc * k);
                for (auto& v : x) v = nd(xr);
                float* d_x = up(x);
                const size_t ycs = (size_t) n + 5;
                float* d_y;
                CK(hipMalloc(&d_y, (size_t) nc * ycs * 4));
                std::vector<float> nanv((size_t) nc * ycs, NAN);
                CK(hipMemcpy(d_y, nanv.data(), nanv.size() * 4, hipMemcpyHostToDevice));
                x3::MvJob j{};
                j.m = &m; j.xf = d_x; j.xf_cs = (size_t) k; j.y = d_y; j.y_cs = ycs; j.alpha = 1.0f; j.ncol = nc;
                const int e0 = gg::launch_errors();
                const bool ok = gg::mv(&j, 1, 0);
                hipError_t e = hipDeviceSynchronize();
                ++tm.cases;
                char tag[160];
                std::snprintf(tag, sizeof tag, "%s gg::mv k=%d n=%d ncol=%d", ty.name, k, n, nc);
                if (e != hipSuccess) { std::printf("STICKY DEVICE ERROR in %s: %s\n", tag, hipGetErrorString(e)); bail(3); }
                if (!ok || gg::launch_errors() != e0) { ++tm.refused; note_fail(tm, std::string(tag) + ": refused"); }
                else {
                    const auto y = down(d_y, (size_t) nc * ycs);
                    size_t nanc = 0, bad = 0;
                    double worst = 0;
                    std::vector<float> wf((size_t) n * k);
                    for (int r = 0; r < n; ++r) decode_row(ty.id, &raw[(size_t) r * rb], k, &wf[(size_t) r * k]);
                    for (int c = 0; c < nc; ++c) {
                        Q8 q = quant(&x[(size_t) c * k], k);
                        const bool quantised = ty.id != kBF16;   // BF16 takes the F32 activations as they are
                        for (int r = 0; r < n; ++r) {
                            const float v = y[(size_t) c * ycs + r];
                            if (!std::isfinite(v)) { ++nanc; continue; }
                            double s = 0, sa = 0;
                            for (int i = 0; i < k; ++i) {
                                const double tt = (double) wf[(size_t) r * k + i] * (quantised ? q.xhat[i] : x[(size_t) c * k + i]);
                                s += tt; sa += std::fabs(tt);
                            }
                            const double err = std::fabs((double) v - s) / (sa + 1e-30);
                            worst = std::max(worst, err);
                            if (err > 2e-5) ++bad;
                        }
                    }
                    tm.nan += (int) nanc;
                    tm.mism += (int) bad;
                    if (nanc || bad) {
                        char b[300];
                        std::snprintf(b, sizeof b, "%s: %zu non-finite, %zu beyond 2e-5 (worst %.2e)", tag, nanc, bad, worst);
                        note_fail(tm, b);
                    }
                }
                CK(hipFree(d_y)); CK(hipFree(d_x)); CK(hipFree(d_raw));
            }
        }
    }
}
#endif

#if defined(DQ_TEST_GEMM)
// ------------------------------------------------------------------------------------------------ gemm: the chunk path after the dequantizer
void run_gemm(const TypeInfo& ty, const Kern& kn, bool quick) {
    if (!ty.dense) return;
    Tally& t = g_res[key("gemm", ty.name, kn.name)];
#if defined(DQ_HIP)
    hipblasHandle_t blas;
    if (hipblasCreate(&blas) != HIPBLAS_STATUS_SUCCESS) { std::printf("hipblasCreate failed\n"); bail(3); }
#else
    cublasHandle_t blas;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) { std::printf("cublasCreate failed\n"); bail(3); }
#endif
    std::normal_distribution<float> nd(0.f, 1.f);
    struct Shape { int rows, n, k; };
    std::vector<Shape> shapes = {{64, 512, 1024}, {100, 640, 4096}, {130, 1024, 5120}, {257, 2048, 2048}};
    if (quick) shapes = {{100, 640, 4096}};
    long long pf_taken = 0, pf_tried = 0;
    for (const Shape& s : shapes) {
        std::mt19937_64 rng((uint64_t) ty.id * 977 + (uint64_t) s.k + (uint64_t) s.n);
        const size_t rb = row_bytes(ty.id, s.k);
        auto raw = make_bytes(ty.id, (size_t) s.n * (s.k / ty.bv), rng);
        uint8_t* d_raw = up(raw);
        x3::Mat m = make_mat(ty, d_raw, s.k, s.n);
        std::vector<uint16_t> x16((size_t) s.rows * s.k);
        for (auto& v : x16) v = f2h(nd(rng));
        uint16_t* d_x = up(x16);
        const int cols = 256;   // the chunk path's run of rows (gemm() takes w16_elems / k, multiples of 128)
        uint16_t* d_w;
        CK(hipMalloc(&d_w, (size_t) cols * s.k * 2));
        std::vector<float> ref_y((size_t) s.rows * s.n), got_y((size_t) s.rows * s.n);
        // CPU: the dequantized weights rounded to F16 (what the GEMM multiplies), times the F16 activations
        std::vector<float> wf((size_t) s.n * s.k);
        for (int r = 0; r < s.n; ++r) decode_row(ty.id, &raw[(size_t) r * rb], s.k, &wf[(size_t) r * s.k]);
        for (auto& v : wf) v = h2f(f2h(v));
        std::vector<double> sabs((size_t) s.rows * s.n);
        for (int r = 0; r < s.rows; ++r)
            for (int c = 0; c < s.n; ++c) {
                double a = 0, aa = 0;
                for (int i = 0; i < s.k; ++i) { const double tt = (double) wf[(size_t) c * s.k + i] * h2f(x16[(size_t) r * s.k + i]); a += tt; aa += std::fabs(tt); }
                ref_y[(size_t) r * s.n + c] = (float) a;
                sabs[(size_t) r * s.n + c] = aa;
            }
        for (int path = 0; path < 2; ++path) {   // 0 the library GEMM, 1 the engine's WMMA kernel (HIP gfx11)
            if (path == 1) {
#if !defined(DQ_HIP)
                continue;
#endif
            }
            float* d_y;
            CK(hipMalloc(&d_y, (size_t) s.rows * s.n * 4));
            std::vector<float> nanv((size_t) s.rows * s.n, NAN);
            CK(hipMemcpy(d_y, nanv.data(), nanv.size() * 4, hipMemcpyHostToDevice));
            set_env("DS4_DQ_OLD", kn.env);
            bool taken = true;
            for (int c0 = 0; c0 < s.n; c0 += cols) {
                const int nc = std::min(cols, s.n - c0);
                gg::rows(nc == s.n ? m : gg::view(m, c0, nc), d_w, false, 0);
#if defined(DQ_HIP)
                if (path == 1) {
                    ++pf_tried;
                    if (s.rows >= 64 && ((uintptr_t) d_x & 15u) == 0 && strata_pf_gemm_f16_ld(d_x, s.k, d_w, s.k, d_y + c0, s.rows, nc, s.k, s.n, 0.0f, nullptr)) { ++pf_taken; continue; }
                    taken = false;
                    continue;
                }
                const float one = 1.0f, zero = 0.0f;
                if (hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, nc, s.rows, s.k, &one, d_w, HIPBLAS_R_16F, s.k, d_x, HIPBLAS_R_16F, s.k, &zero,
                                  d_y + c0, HIPBLAS_R_32F, s.n, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT) != HIPBLAS_STATUS_SUCCESS) {
                    note_fail(t, "hipblasGemmEx failed");
                    taken = false;
                }
#else
                const float one = 1.0f, zero = 0.0f;
                if (cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, nc, s.rows, s.k, &one, d_w, CUDA_R_16F, s.k, d_x, CUDA_R_16F, s.k, &zero,
                                 d_y + c0, CUDA_R_32F, s.n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) {
                    note_fail(t, "cublasGemmEx failed");
                    taken = false;
                }
#endif
            }
            CK(hipDeviceSynchronize());
            set_env("DS4_DQ_OLD", nullptr);
            char tag[160];
            std::snprintf(tag, sizeof tag, "%s chunk gemm (%s) kernel=%s rows=%d n=%d k=%d", ty.name, path == 0 ? "library" : "wmma", kn.name, s.rows, s.n, s.k);
            if (!taken) { std::printf("  note: %s: the engine's WMMA kernel declined (device or shape), the library GEMM runs instead\n", tag); continue; }
            ++t.cases;
            got_y = down(d_y, (size_t) s.rows * s.n);
            size_t nanc = 0, bad = 0;
            double worst = 0;
            for (size_t i = 0; i < got_y.size(); ++i) {
                if (!std::isfinite(got_y[i])) { ++nanc; continue; }
                const double err = std::fabs((double) got_y[i] - ref_y[i]) / (sabs[i] + 1e-30);
                worst = std::max(worst, err);
                if (err > 2e-4) ++bad;
            }
            t.nan += (int) nanc;
            t.mism += (int) bad;
            if (nanc || bad) {
                char b[300];
                std::snprintf(b, sizeof b, "%s: %zu non-finite, %zu beyond 2e-4 (worst %.2e)", tag, nanc, bad, worst);
                note_fail(t, b);
            }
            CK(hipFree(d_y));
        }
        CK(hipFree(d_w)); CK(hipFree(d_x)); CK(hipFree(d_raw));
    }
#if defined(DQ_HIP)
    std::printf("  note: %s WMMA GEMM taken %lld of %lld runs\n", ty.name, pf_taken, pf_tried);
    hipblasDestroy(blas);
#else
    cublasDestroy(blas);
#endif
}
#endif

[[noreturn]] void bail(int code) {
    for (const auto& kv : g_res) {
        const Tally& t = kv.second;
        std::printf("RESULT %-34s cases=%d fail=%d nan=%d mismatch=%d ulp1=%d unwritten=%d refused=%d old_vs_new=%d guard=%d coldiff=%d %s\n", kv.first.c_str(), t.cases,
                    t.fail, t.nan, t.mism, t.ulp, t.unwritten, t.refused, t.diff, t.guard, t.coldiff, t.fail ? "FAIL" : "ok");
        if (!t.first.empty()) std::printf("  first: %s\n", t.first.c_str());
    }
    std::printf("SUMMARY fails=%d%s\n", g_fails, code == 3 ? " (stopped on a device error)" : "");
    std::fflush(stdout);
    std::exit(code);
}

std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        if (const TypeInfo* ti = find_type_name(p)) { v.push_back(ti->id); }
        else v.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p) ++p;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    int dev = 0;
    bool quick = false;
    std::vector<int> types;
    std::string kernel = "all", paths = "rows,mv,f16,gguf,gemm"   /* repeat is on request: --path repeat */;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) dev = std::atoi(argv[++i]);
        else if (a == "--type" && i + 1 < argc) { const std::string v = argv[++i]; if (v != "all") types = parse_list(v.c_str()); }
        else if (a == "--kernel" && i + 1 < argc) kernel = argv[++i];
        else if (a == "--path" && i + 1 < argc) paths = argv[++i];
        else if (a == "--quick") quick = true;
        else if (a == "--reps" && i + 1 < argc) g_reps = std::atoi(argv[++i]);
        else if (a == "--shape" && i + 1 < argc) {   // 5120x1280,4096x1280
            for (const char* q = argv[++i]; *q;) { int k = 0, n = 0; if (std::sscanf(q, "%dx%d", &k, &n) == 2) g_rep_shapes.push_back({k, n}); while (*q && *q != ',') ++q; if (*q) ++q; }
        }
        else if (a == "-v") g_verbose = true;
        else { std::printf("unknown argument %s\n", a.c_str()); return 2; }
    }
    CK(hipSetDevice(dev));
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    hipDeviceProp_t pr;
    hipGetDeviceProperties(&pr, dev);
    std::printf("DEVICE %d: %s (%s)\n", dev, pr.name, pr.gcnArchName);
#else
    cudaDeviceProp pr;
    cudaGetDeviceProperties(&pr, dev);
    std::printf("DEVICE %d: %s (sm_%d%d)\n", dev, pr.name, pr.major, pr.minor);
#endif
    if (types.empty()) for (const auto& t : kTypes) types.push_back(t.id);
    auto want = [&](const char* p) { return paths.find(p) != std::string::npos; };
    for (int id : types) {
        const TypeInfo* ty = find_type(id);
        if (!ty) { std::printf("not a type this test knows: %d\n", id); return 2; }
        for (const Kern& kn : kKerns) {
            if (kernel != "all" && kernel != kn.name) continue;
            if (want("rows")) run_rows(*ty, kn, quick);
            if (want("repeat")) run_repeat(*ty, kn, quick);
#if defined(DQ_TEST_GGUF)
            if (want("gguf")) run_gguf(*ty, kn, quick);
#endif
#if defined(DQ_TEST_GEMM)
            if (want("gemm")) run_gemm(*ty, kn, quick);
#endif
        }
        if (want("mv")) run_mv(*ty, quick);
        if (want("f16")) run_f16(*ty);
    }
    bail(g_fails ? 1 : 0);
}

