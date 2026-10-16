// gdn_bench: the decode window's GDN recurrence (gdn_step_norm_multi, verify: state read, not written) at the
// Coder's shape (h_k 16, h_v 48, S 128, conv channels 10240) for 1-6 tokens; 48 layers' own states (3 MB each) so
// they stream from VRAM as a window's layers do; host-timed; checksum of the outputs. STRATA_GDN_SPLIT etc apply.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include "strata/kernels/verify_kernels.hpp"
#include <cstdlib>
int main() {
    sycl::queue* s = new sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    constexpr int HK = 16, HV = 48, S = 128, C = 2 * HK * S + HV * S, ZV = HV * S, TM = 8, NL = 48;
    const size_t st_floats = (size_t) HV * S * S;
    uint32_t r = 7;
    auto rnd = [&] { r = r * 1664525u + 1013904223u; return ((r >> 8) & 0xffff) / 65536.0f - 0.5f; };
    std::vector<float> hs(st_floats), hh((size_t) TM * C), hg((size_t) TM * HV), hb((size_t) TM * HV), hz((size_t) TM * ZV), hgam(S);
    for (auto& v : hs) v = 0.02f * rnd();
    for (auto& v : hh) v = rnd();
    for (auto& v : hg) v = -0.1f + 0.05f * rnd();
    for (auto& v : hb) v = 0.5f + 0.2f * rnd();
    for (auto& v : hz) v = rnd();
    for (auto& v : hgam) v = 1.0f + 0.1f * rnd();
    auto dev = [&](const void* p, size_t b) { void* d = sycl::malloc_device(b, *s); s->memcpy(d, p, b).wait(); return (float*) d; };
    std::vector<float*> st(NL);
    for (int l = 0; l < NL; ++l) st[l] = dev(hs.data(), st_floats * 4);
    float *h = dev(hh.data(), hh.size() * 4), *g = dev(hg.data(), hg.size() * 4), *b = dev(hb.data(), hb.size() * 4),
          *z = dev(hz.data(), hz.size() * 4), *gam = dev(hgam.data(), hgam.size() * 4);
    float* y = sycl::malloc_device<float>((size_t) TM * ZV, *s);
    // GDN_CBATCH=1: the commit of 36 GDN layers (conv history + state replay) per layer vs gdn_commit_replay_layers: every
    // byte of the 36 layers' state + history compared, and the time of each, for n_keep = 1..6 (window rows TM)
    if (const char* e = std::getenv("GDN_CBATCH"); e && e[0] == '1') {
        constexpr int L = 36;
        const size_t conv_f = (size_t) C * 3, lay_f = st_floats + conv_f;
        std::vector<float> hl(lay_f * L), hh2((size_t) L * TM * C), hgb((size_t) L * TM * HV), hbb((size_t) L * TM * HV);
        for (auto& v : hl) v = 0.02f * rnd();
        for (auto& v : hh2) v = rnd();
        for (auto& v : hgb) v = -0.1f + 0.05f * rnd();
        for (auto& v : hbb) v = 0.5f + 0.2f * rnd();
        float *A = sycl::malloc_device<float>(hl.size(), *s), *B = sycl::malloc_device<float>(hl.size(), *s);
        float *dh = dev(hh2.data(), hh2.size() * 4), *dg = dev(hgb.data(), hgb.size() * 4), *db = dev(hbb.data(), hbb.size() * 4);
        int32_t* nk = sycl::malloc_device<int32_t>(1, *s);
        auto per_layer = [&](float* base) {
            for (int i = 0; i < L; ++i) {
                float* stl = base + (size_t) i * lay_f;
                strata::kernels::gdn_conv_commit(stl + st_floats, dh + (size_t) i * TM * C, C, nk, s);
                strata::kernels::gdn_step_norm_multi(stl, dh + (size_t) i * TM * C, C, dg + (size_t) i * TM * HV, db + (size_t) i * TM * HV,
                                                     z, gam, 1e-6f, y, HK, HV, TM, nk, s, TM);
            }
        };
        auto batched = [&](float* base) {
            return strata::kernels::gdn_commit_replay_layers(base, (int64_t) lay_f, (int64_t) st_floats, dh, dh, (int64_t) TM * C,
                                                            dg, db, (int64_t) TM * HV, C, HK, HV, TM, L, nk, s);
        };
        for (int n = 1; n <= 6; ++n) {
            s->memcpy(nk, &n, 4).wait();
            s->memcpy(A, hl.data(), hl.size() * 4).wait();
            s->memcpy(B, hl.data(), hl.size() * 4).wait();
            per_layer(A);
            const bool ok = batched(B);
            s->wait();
            std::vector<float> ha(hl.size()), hb2(hl.size());
            s->memcpy(ha.data(), A, ha.size() * 4); s->memcpy(hb2.data(), B, hb2.size() * 4); s->wait();
            const size_t diff = [&] { size_t d = 0; for (size_t i = 0; i < ha.size(); ++i) d += std::memcmp(&ha[i], &hb2[i], 4) != 0; return d; }();
            auto timeit = [&](auto&& f) {
                for (int i = 0; i < 20; ++i) f(A);
                s->wait();
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 200; ++i) f(A);
                s->wait();
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 200;
            };
            const double tp = timeit(per_layer), tb = timeit([&](float* b) { batched(b); });
            std::printf("GDN commit, 36 layers, n_keep %d: per layer %7.1f us, batched %7.1f us (%s)  %zu of %zu floats differ\n",
                        n, tp, tb, ok ? "batched" : "DECLINED", diff, ha.size());
        }
        return 0;
    }
    for (int T = 1; T <= 6; ++T) {
        for (int i = 0; i < 96; ++i)
            strata::kernels::gdn_step_norm_multi(st[i % NL], h, C, g, b, z, gam, 1e-6f, y, HK, HV, T, nullptr, s);
        s->wait();
        const int it = 480;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < it; ++i)
            strata::kernels::gdn_step_norm_multi(st[i % NL], h, C, g, b, z, gam, 1e-6f, y, HK, HV, T, nullptr, s);
        s->wait();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
        std::vector<float> hy((size_t) T * ZV);
        s->memcpy(hy.data(), y, hy.size() * 4).wait();
        uint64_t cs = 1469598103934665603ull;
        for (float v : hy) { uint32_t u; std::memcpy(&u, &v, 4); cs = (cs ^ u) * 1099511628211ull; }
        std::printf("GDN recurrence, %d tokens: %7.1f us (%.0f GB/s of state)  checksum %016llx\n", T, us,
                    st_floats * 4.0 / us / 1e3, (unsigned long long) cs);
    }
    return 0;
}
