// engine/src/inkling2/hip/ink_hip_moe_pf_test.hip.cpp
// Standalone bit-identity + timing test of the grouped-MoE prefill kernels v2 k_moe_pf_sf4_i8<BM, W13>
// (ink_hip_moe_pf.hip.h) against k_moe_grouped_{w13_situ,w2}_wmma_sf4_i8_128 (ink_hip_codecs.hip.h) on the Inkling 2 Mini
// shapes (D 2304, inter 1152, 128 experts, top_k 8), RDNA4 only.
// PASS = every element of bucket_act (f16) and bucket_y (f32) bit-identical (sentinel-initialised outputs) for every
// B in 64..2048, 4 routings (uniform, Zipf, 8 hot experts, edge counts) and BM 32 / 64; a negative control (one input bit
// flipped) must differ. Then us per call (w13 + w2) for the _128 kernels and both BM.
// Build (from engine/src/inkling2/hip):
//   clang++ -x hip --offload-arch=gfx1201 -O3 -std=c++20 -I../../.. -I../../../include -I.. -I. -DINKLING_ENABLE_HIP=1 \
//           ink_hip_moe_pf_test.hip.cpp -o ink_hip_moe_pf_test -lamdhip64
// Run: ./ink_hip_moe_pf_test [ITERS]   (exit code 0 = PASS)
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#include "ink_hip_moe_pf.hip.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace strata::inkling2::hip;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

static constexpr int D = 2304, INTER = 1152, E = 128, TOPK = 8, N13 = 2 * INTER;
static std::mt19937_64 rng(20261124);
static int g_fail = 0, g_checks = 0;

static size_t body_bytes(size_t* dn_off) {
    const size_t w13 = 16 + size_t(N13) * (D / 2) + size_t(N13) * (D / 16);
    const size_t dn = (w13 + 255) & ~size_t(255);
    *dn_off = dn;
    return dn + size_t(D) * (INTER / 2) + size_t(D) * (INTER / 16);
}
static uint8_t rand_e4m3() {
    const uint64_t r = rng() % 1000;
    if (r < 10) return 0;                                   // zero scale
    const uint8_t exp = uint8_t(rng() % 5);                 // 2^-7 (subnormal range at 0) .. 2^-3: keeps w13 sums O(1..10)
    const uint8_t man = uint8_t(rng() % 8);
    return uint8_t((exp << 3) | man);                       // never 0x7F / 0xFF (NaN)
}
static void fill_body(std::vector<uint8_t>& b, size_t dn_off) {
    float h[4];
    std::uniform_real_distribution<float> u(0.5f, 2.0f);
    for (float& x : h) x = u(rng);
    std::memcpy(b.data(), h, 16);
    uint8_t* p = b.data() + 16;
    for (size_t i = 0; i < size_t(N13) * (D / 2); i += 8) { const uint64_t r = rng(); std::memcpy(p + i, &r, 8); }
    uint8_t* s = p + size_t(N13) * (D / 2);
    for (size_t i = 0; i < size_t(N13) * (D / 16); ++i) s[i] = rand_e4m3();
    uint8_t* p2 = b.data() + dn_off;
    for (size_t i = 0; i < size_t(D) * (INTER / 2); i += 8) { const uint64_t r = rng(); std::memcpy(p2 + i, &r, 8); }
    uint8_t* s2 = p2 + size_t(D) * (INTER / 2);
    for (size_t i = 0; i < size_t(D) * (INTER / 16); ++i) s2[i] = rand_e4m3();
}

struct Routing { std::vector<int32_t> off, btk; };   // exp_off [E + 1], bucket_tok [B * TOPK] (token per bucket row)
// mode 0 uniform, 1 Zipf (s = 1.1), 2 eight hot experts only, 3 exact edge counts
static Routing make_routing(int B, int mode) {
    std::vector<std::vector<int>> per(E);
    std::vector<double> w(E);
    for (int e = 0; e < E; ++e) w[e] = (mode == 1) ? 1.0 / std::pow(double(e + 1), 1.1) : 1.0;
    std::discrete_distribution<int> pick(w.begin(), w.end());
    if (mode == 3) {
        // fill experts 0.. with the edge counts first, the rest uniform
        const int edge[] = {15, 16, 17, 31, 32, 33, 64, 65};
        std::vector<int> need(E, 0);
        for (int i = 0; i < 8; ++i) need[i] = std::min(edge[i], B);
        for (int b = 0; b < B; ++b) {
            std::vector<int> chosen;
            for (int e = 0; e < 8 && int(chosen.size()) < TOPK; ++e) if (need[e] > 0) { --need[e]; chosen.push_back(e); }
            while (int(chosen.size()) < TOPK) {
                const int e = 8 + int(rng() % (E - 8));
                if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) chosen.push_back(e);
            }
            for (int e : chosen) per[e].push_back(b);
        }
    } else {
        for (int b = 0; b < B; ++b) {
            std::vector<int> chosen;
            while (int(chosen.size()) < TOPK) {
                const int e = (mode == 2) ? int(rng() % 8) * 16 : pick(rng);
                if (mode == 2) { if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) chosen.push_back(e);
                                 if (int(chosen.size()) == 8) break; continue; }
                if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) chosen.push_back(e);
            }
            for (int e : chosen) per[e].push_back(b);
        }
    }
    Routing r;
    r.off.assign(E + 1, 0);
    for (int e = 0; e < E; ++e) r.off[e + 1] = r.off[e] + int(per[e].size());
    for (int e = 0; e < E; ++e) for (int b : per[e]) r.btk.push_back(b);
    return r;
}

struct Dev {
    GpuExpertDesc* exps = nullptr; int32_t* off = nullptr; int32_t* btk = nullptr;
    int8_t* xq = nullptr; float* xd = nullptr; half_t* xf16 = nullptr;
    int8_t* aq = nullptr; float* ad = nullptr; half_t* af16 = nullptr;
    half_t* act = nullptr; float* by = nullptr;
};
static const int kMaxB = 2048;

static void upload_inputs(Dev& d, int B, const Routing& r, bool perturb) {
    CK(hipMemcpy(d.off, r.off.data(), (E + 1) * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d.btk, r.btk.data(), r.btk.size() * 4, hipMemcpyHostToDevice));
    std::vector<int8_t> xq(size_t(B) * D), aq(size_t(B) * TOPK * INTER);
    std::vector<float> xd(size_t(B) * (D / 16)), ad(size_t(B) * TOPK * (INTER / 16));
    std::uniform_int_distribution<int> q(-127, 127);
    std::uniform_real_distribution<float> s(2e-3f, 2e-2f);   // with the e4m3 scales above: w13 sums O(1..10), SITU not saturated
    for (auto& v : xq) v = int8_t(rng() % 17 == 0 ? 0 : q(rng));
    for (auto& v : aq) v = int8_t(rng() % 17 == 0 ? 0 : q(rng));
    for (auto& v : xd) v = (rng() % 50 == 0) ? 0.0f : s(rng);
    for (auto& v : ad) v = (rng() % 50 == 0) ? 0.0f : s(rng);
    if (perturb) { xq[size_t(r.btk[r.off[0]]) * D + 5] ^= 1; aq[size_t(r.off[0]) * INTER + 7] ^= 1; }
    CK(hipMemcpy(d.xq, xq.data(), xq.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(d.xd, xd.data(), xd.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d.aq, aq.data(), aq.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(d.ad, ad.data(), ad.size() * 4, hipMemcpyHostToDevice));
}
static int mtiles128(int B) { return (B >= 1024) ? 8 : ((B >= 256) ? 4 : 2); }

static void run_orig(Dev& d, int B, hipStream_t st) {
    k_moe_grouped_w13_situ_wmma_sf4_i8_128<<<dim3((N13 + 127) / 128, E, mtiles128(B)), 256, 0, st>>>(
        d.exps, d.off, d.btk, d.xq, d.xd, d.act, D, INTER, 4.0f, 25.0f);
    k_moe_grouped_w2_wmma_sf4_i8_128<<<dim3((D + 127) / 128, E, mtiles128(B)), 256, 0, st>>>(d.exps, d.off, d.aq, d.ad, d.by, D, INTER);
}
static void run_pf(Dev& d, int B, int bm, hipStream_t st) {
    launch_moe_pf_sf4_i8(bm, B, E, D, INTER, 4.0f, 25.0f, d.exps, d.off, d.btk, d.xq, d.xd, d.act, true, nullptr, nullptr, nullptr, st);
    launch_moe_pf_sf4_i8(bm, B, E, D, INTER, 0.0f, 0.0f, d.exps, d.off, nullptr, nullptr, nullptr, nullptr, false, d.aq, d.ad, d.by, st);
}

struct Out { std::vector<uint16_t> act; std::vector<uint32_t> by; };
static Out run_get(Dev& d, int B, int arm) {   // arm 0 = the _128 kernels, else v2 with BM = arm
    const size_t na = size_t(B) * TOPK * INTER, nb = size_t(B) * TOPK * D;
    CK(hipMemset(d.act, 0x55, na * 2));
    CK(hipMemset(d.by, 0x7f, nb * 4));
    if (arm == 0) run_orig(d, B, nullptr); else run_pf(d, B, arm, nullptr);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    Out o; o.act.resize(na); o.by.resize(nb);
    CK(hipMemcpy(o.act.data(), d.act, na * 2, hipMemcpyDeviceToHost));
    CK(hipMemcpy(o.by.data(), d.by, nb * 4, hipMemcpyDeviceToHost));
    return o;
}
static size_t ndiff(const Out& a, const Out& b, size_t* nact) {
    size_t n1 = 0, n2 = 0;
    for (size_t i = 0; i < a.act.size(); ++i) n1 += a.act[i] != b.act[i];
    for (size_t i = 0; i < a.by.size(); ++i) n2 += a.by[i] != b.by[i];
    *nact = n1;
    return n1 + n2;
}

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::max(1, std::atoi(argv[1])) : 20;
    size_t dn_off = 0;
    const size_t bb = body_bytes(&dn_off);
    std::printf("ink_hip_moe_pf_test: D %d inter %d experts %d top_k %d, body %.2f MB x %d, iters %d\n", D, INTER, E, TOPK, bb / 1e6, E, iters);
    uint8_t* bodies = nullptr;
    CK(hipMalloc(&bodies, bb * E));
    {
        std::vector<uint8_t> h(bb);
        for (int e = 0; e < E; ++e) { fill_body(h, dn_off); CK(hipMemcpy(bodies + bb * e, h.data(), bb, hipMemcpyHostToDevice)); }
    }
    std::vector<GpuExpertDesc> hd(E);
    for (int e = 0; e < E; ++e) { hd[e].body = bodies + bb * e; hd[e].dn_offset = uint32_t(dn_off); hd[e].gu_code = CODE_SF4; hd[e].dn_code = CODE_SF4; hd[e].pad = 0; }
    Dev d;
    CK(hipMalloc(&d.exps, E * sizeof(GpuExpertDesc)));
    CK(hipMemcpy(d.exps, hd.data(), E * sizeof(GpuExpertDesc), hipMemcpyHostToDevice));
    CK(hipMalloc(&d.off, (E + 1) * 4));
    CK(hipMalloc(&d.btk, size_t(kMaxB) * TOPK * 4));
    CK(hipMalloc(&d.xq, size_t(kMaxB) * D));
    CK(hipMalloc(&d.xd, size_t(kMaxB) * (D / 16) * 4));
    CK(hipMalloc(&d.xf16, size_t(kMaxB) * D * 2));
    CK(hipMalloc(&d.aq, size_t(kMaxB) * TOPK * INTER));
    CK(hipMalloc(&d.ad, size_t(kMaxB) * TOPK * (INTER / 16) * 4));
    CK(hipMalloc(&d.af16, size_t(kMaxB) * TOPK * INTER * 2));
    CK(hipMalloc(&d.act, size_t(kMaxB) * TOPK * INTER * 2));
    CK(hipMalloc(&d.by, size_t(kMaxB) * TOPK * D * 4));
    CK(hipMemset(d.xf16, 0, size_t(kMaxB) * D * 2));
    CK(hipMemset(d.af16, 0, size_t(kMaxB) * TOPK * INTER * 2));

    // ---- identity ----
    const int Bs[] = {64, 65, 113, 256, 300, 512, 513, 640, 768, 1000, 1024, 1536, 2048};
    const char* mname[] = {"uniform", "zipf", "hot8", "edge"};
    const int arms[] = {32, 64};
    const char* aname[] = {"v2bm32", "v2bm64"};
    for (int B : Bs) {
        for (int mode = 0; mode < 4; ++mode) {
            Routing r = make_routing(B, mode);
            upload_inputs(d, B, r, false);
            Out a = run_get(d, B, 0);
            int mmax = 0; for (int e = 0; e < E; ++e) mmax = std::max(mmax, r.off[e + 1] - r.off[e]);
            std::printf("B %4d %-7s max m_e %4d:", B, mname[mode], mmax);
            for (int k = 0; k < 2; ++k) {
                Out b = run_get(d, B, arms[k]);
                size_t na = 0; const size_t n = ndiff(a, b, &na);
                ++g_checks;
                if (n) ++g_fail;
                std::printf(" %s %s", aname[k], n ? "FAIL" : "ok");
                if (n) std::printf("(act %zu y %zu)", na, n - na);
            }
            std::printf("\n");
        }
    }
    // ---- negative control ----
    {
        const int B = 113;
        Routing r = make_routing(B, 0);
        std::mt19937_64 save = rng;
        upload_inputs(d, B, r, false);
        Out base = run_get(d, B, 32);
        rng = save;
        upload_inputs(d, B, r, true);
        Out pert = run_get(d, B, 0);
        size_t na = 0; const size_t n = ndiff(base, pert, &na);
        {
            std::vector<uint16_t> v(base.act);
            std::sort(v.begin(), v.end());
            const size_t distinct = size_t(std::unique(v.begin(), v.end()) - v.begin());
            std::printf("value spread (B 113 uniform): %zu distinct f16 act values of %zu (a saturated SITU would give few)\n", distinct, base.act.size());
        }
        std::printf("negative control (1 activation bit flipped, B 113): %s (diffs act %zu, y %zu)\n", n ? "PASS (differs)" : "FAIL (no difference seen)", na, n - na);
        if (!n) ++g_fail;
    }
    std::printf("%s: %d of %d identity checks failed\n", g_fail ? "BITWISE FAIL" : "BITWISE PASS", g_fail, g_checks);

    // ---- timing ----
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    const int Ts[] = {113, 256, 512, 1024, 2048};
    std::printf("timing: us per call w13+w2 (x44 SF4 visits = ms per chunk) | _128 kernels | v2 BM32 | v2 BM64\n");
    for (int tmode = 0; tmode < 2; ++tmode)
    for (int B : Ts) {
        Routing r = make_routing(B, tmode);
        upload_inputs(d, B, r, false);
        float us[3];
        const int arm_of[3] = {0, 32, 64};
        for (int k = 0; k < 3; ++k) {
            auto go = [&]() { const int a = arm_of[k]; if (a == 0) run_orig(d, B, nullptr); else run_pf(d, B, a, nullptr); };
            for (int w = 0; w < 3; ++w) go();
            CK(hipEventRecord(e0, nullptr));
            for (int i = 0; i < iters; ++i) go();
            CK(hipEventRecord(e1, nullptr));
            CK(hipEventSynchronize(e1));
            float ms = 0; CK(hipEventElapsedTime(&ms, e0, e1));
            us[k] = ms * 1000.0f / iters;
        }
        const double gmac = double(B) * TOPK * (double(N13) * D + double(D) * INTER) * 1e-9;
        std::printf("%s B %4d: _128 %8.1f (%6.1f ms) | bm32 %8.1f (%6.1f) | bm64 %8.1f (%6.1f) | best v2 %.1f TOPS\n",
                    tmode ? "zipf   " : "uniform", B, us[0], us[0] * 44e-3, us[1], us[1] * 44e-3, us[2], us[2] * 44e-3,
                    2.0 * gmac / (std::min(us[1], us[2]) * 1e-6) * 1e-3);
    }
    CK(hipGetLastError());
    return g_fail ? 1 : 0;
}
