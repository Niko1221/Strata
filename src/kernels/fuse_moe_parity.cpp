// src/kernels/fuse_moe_parity.cpp - the verify window's MoE-path launch fusions (STRATA_FUSE_*) against the launch
// sequences they replace, BITWISE: every output byte of the fused launch is compared with the old sequence's, on
// random inputs (GPU, synthetic, no model).  Exit 0 when all match.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
template <typename T> T* dev(size_t n) {
    T* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T) + 64), "cudaMalloc");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) {
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
// mapped host memory and its device alias
template <typename T> T* mapped(size_t n, T** d) {
    T* h = nullptr;
    check(cudaHostAlloc((void**) &h, n * sizeof(T) + 64, cudaHostAllocMapped), "hostalloc");
    check(cudaHostGetDevicePointer((void**) d, h, 0), "devptr");
    std::memset(h, 0, n * sizeof(T) + 64);
    return h;
}
template <typename T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

// STRATA_FUSE_FETCH: wait_fetch_rebase vs wait_flag_ge(_or) + fetch_blobs + rebase_ptrs.  Blobs in device memory and in
// mapped host memory (the PCIe share's source), 0..cap of them, the skip word equal to the ring or not.
// STRATA_FUSE_PLAN: wait_copy_i32_from_mapped_unless vs wait_flag_ge_or + copy_i32_from_mapped_unless - the destination
// words (copied, or untouched when the device planned the group), and the wait held until the host raises the flag
int test_plan(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    int bad = 0, cases = 0;
    uint32_t* flag_d = nullptr;
    uint32_t* flag = mapped<uint32_t>(4, &flag_d);
    uint32_t* skip = dev<uint32_t>(4);
    for (const int n : {0, 1, 37, 128, 129, 1000}) {
        int32_t* src_d = nullptr;
        int32_t* src = mapped<int32_t>((size_t) n + 1, &src_d);
        for (int i = 0; i < n; ++i) src[i] = (int32_t) rng();
        for (const int sk : {0, 1}) {
            const uint32_t ring = 7 + (uint32_t) (rng() % 100);
            *(volatile uint32_t*) flag = ring + (uint32_t) (rng() % 2);
            const uint32_t skv = sk ? ring : ring - 1;
            check(cudaMemcpy(skip, &skv, 4, cudaMemcpyHostToDevice), "skip");
            std::vector<int32_t> junk((size_t) n + 1);
            for (auto& v : junk) v = (int32_t) rng();
            int32_t* a = dev<int32_t>((size_t) n + 1);
            int32_t* b = dev<int32_t>((size_t) n + 1);
            up(a, junk);
            up(b, junk);
            wait_flag_ge_or(flag_d, ring, skip, s);
            copy_i32_from_mapped_unless(a, src_d, n, skip, ring, s);
            wait_copy_i32_from_mapped_unless(flag_d, ring, skip, b, src_d, n, s);
            check(cudaStreamSynchronize(s), "sync");
            ++cases;
            if (!same(down(a, (size_t) n + 1), down(b, (size_t) n + 1))) {
                std::printf("    *** wait_copy_i32_unless n %d skip %d differs ***\n", n, sk);
                ++bad;
            }
            cudaFree(a); cudaFree(b);
        }
        cudaFreeHost(src);
    }
    {   // the wait: held while the flag is below the ring and the skip word differs
        const uint32_t ring = 1000, skv = 1;
        *(volatile uint32_t*) flag = ring - 1;
        check(cudaMemcpy(skip, &skv, 4, cudaMemcpyHostToDevice), "skip");
        int32_t* src_d = nullptr;
        int32_t* src = mapped<int32_t>(64, &src_d);
        for (int i = 0; i < 64; ++i) src[i] = i * 3 + 1;
        int32_t* b = dev<int32_t>(64);
        check(cudaMemset(b, 0, 256), "memset");
        wait_copy_i32_from_mapped_unless(flag_d, ring, skip, b, src_d, 64, s);
        { struct timespec ts{0, 50000000}; nanosleep(&ts, nullptr); }
        const bool held = cudaStreamQuery(s) == cudaErrorNotReady;
        for (int i = 0; i < 64; ++i) src[i] = i * 5 + 2;   // written before the flag rises: these are what it copies
        *(volatile uint32_t*) flag = ring;
        check(cudaStreamSynchronize(s), "sync");
        const auto got = down(b, 64);
        bool ok = held;
        for (int i = 0; i < 64; ++i) ok = ok && got[i] == i * 5 + 2;
        ++cases;
        if (!ok) { std::printf("    *** wait_copy_i32_unless: wait not held (%d) or wrong words ***\n", (int) held); ++bad; }
        cudaFree(b);
        cudaFreeHost(src);
    }
    cudaFreeHost(flag);
    cudaFree(skip);
    std::printf("  %-60s %s (%d cases)\n", "STRATA_FUSE_PLAN wait_copy_i32_unless == wait_or + copy_unless",
                bad ? "*** NO ***" : "bitwise", cases);
    return bad;
}

int test_fetch(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    int bad = 0, cases = 0;
    const int nblob_src = 24;
    for (const int64_t blob : {(int64_t) 16, (int64_t) 4096 + 16, (int64_t) 302080}) {
        const size_t srcb = (size_t) nblob_src * (size_t) blob;
        std::vector<uint8_t> hs(srcb);
        for (auto& v : hs) v = (uint8_t) rng();
        uint8_t* dsrc = dev<uint8_t>(srcb);
        up(dsrc, hs);
        uint8_t* hsrc_d = nullptr;
        uint8_t* hsrc = mapped<uint8_t>(srcb, &hsrc_d);
        std::memcpy(hsrc, hs.data(), srcb);
        uint32_t* flag_d = nullptr;
        uint32_t* flag = mapped<uint32_t>(4, &flag_d);
        uint32_t* skip = dev<uint32_t>(4);
        for (const int cap : {16, 8}) {
            for (const int n : {0, 1, 3, cap}) {
                for (const int src_kind : {0, 1}) {   // 0: device blobs, 1: mapped host blobs
                    for (const int sk : {0, 1}) {     // skip word: 1 = equal to the ring (device plan)
                        const uint32_t ring = 7 + (uint32_t) (rng() % 100);
                        *(volatile uint32_t*) flag = ring + (uint32_t) (rng() % 2);   // the flag has risen
                        const uint32_t skv = sk ? ring : ring - 1;
                        check(cudaMemcpy(skip, &skv, 4, cudaMemcpyHostToDevice), "skip");
                        // the plan: counts[2] = n, ptr2[0, cap) blob pointers (some repeated), the rest garbage
                        std::vector<int32_t> cnt = {5, 9, n, 0};
                        std::vector<unsigned long long> ptr(cap);
                        uint8_t* base = src_kind ? hsrc_d : dsrc;
                        for (int k = 0; k < cap; ++k) ptr[k] = (unsigned long long) (base + (size_t) (rng() % nblob_src) * blob);
                        int32_t* dcnt = dev<int32_t>(4);
                        up(dcnt, cnt);
                        unsigned long long* p_old = dev<unsigned long long>(cap);
                        unsigned long long* p_src = dev<unsigned long long>(cap);
                        unsigned long long* p_out = dev<unsigned long long>(cap);
                        up(p_old, ptr);
                        up(p_src, ptr);
                        std::vector<unsigned long long> junk(cap, 0xdeadbeefcafef00dull);
                        up(p_out, junk);
                        const size_t stb = (size_t) cap * (size_t) blob;
                        std::vector<uint8_t> fill(stb);
                        for (auto& v : fill) v = (uint8_t) rng();
                        uint8_t* st_old = dev<uint8_t>(stb);
                        uint8_t* st_new = dev<uint8_t>(stb);
                        up(st_old, fill);
                        up(st_new, fill);
                        // the old launches
                        if (sk) wait_flag_ge_or(flag_d, ring, skip, s);
                        else wait_flag_ge(flag_d, ring, s);
                        fetch_blobs(p_old, dcnt + 2, st_old, blob, cap, s);
                        rebase_ptrs(p_old, dcnt + 2, st_old, blob, s);
                        // the fused one
                        wait_fetch_rebase(flag_d, ring, sk ? skip : nullptr, p_src, dcnt + 2, st_new, blob, cap, p_out, s);
                        check(cudaStreamSynchronize(s), "sync");
                        const auto a = down(st_old, stb), b = down(st_new, stb);
                        const auto pa = down(p_old, cap), pb = down(p_out, cap), ps = down(p_src, cap);
                        bool ok = same(a, b) && same(ps, ptr);   // the staging bytes; src untouched
                        for (int k = 0; k < cap && ok; ++k) {
                            // the first n pointers: the rebased ones (relative to each staging buffer); the others untouched
                            if (k < n) ok = pa[k] - (unsigned long long) st_old == pb[k] - (unsigned long long) st_new;
                            else ok = pb[k] == junk[k] && pa[k] == ptr[k];
                        }
                        ++cases;
                        if (!ok) {
                            std::printf("    *** wait_fetch_rebase blob %lld cap %d n %d src %d skip %d differs ***\n",
                                        (long long) blob, cap, n, src_kind, sk);
                            ++bad;
                        }
                        cudaFree(dcnt); cudaFree(p_old); cudaFree(p_src); cudaFree(p_out); cudaFree(st_old); cudaFree(st_new);
                    }
                }
            }
        }
        // the wait itself: a flag below the ring holds the fused launch until the host raises it
        {
            const uint32_t ring = 1000;
            *(volatile uint32_t*) flag = ring - 1;
            const uint32_t skv = 0;
            check(cudaMemcpy(skip, &skv, 4, cudaMemcpyHostToDevice), "skip");
            int32_t* dcnt = dev<int32_t>(4);
            up(dcnt, std::vector<int32_t>{0, 0, 2, 0});
            std::vector<unsigned long long> ptr = {(unsigned long long) dsrc, (unsigned long long) (dsrc + blob)};
            unsigned long long* p_src = dev<unsigned long long>(2);
            unsigned long long* p_out = dev<unsigned long long>(2);
            up(p_src, ptr);
            uint8_t* st = dev<uint8_t>((size_t) 2 * blob);
            wait_fetch_rebase(flag_d, ring, skip, p_src, dcnt + 2, st, blob, 2, p_out, s);
            int polls = 0;
            while (cudaStreamQuery(s) == cudaErrorNotReady && polls < 50) { ++polls; struct timespec ts{0, 2000000}; nanosleep(&ts, nullptr); }
            const bool held = polls == 50;
            *(volatile uint32_t*) flag = ring;
            check(cudaStreamSynchronize(s), "sync");
            const auto b = down(st, (size_t) 2 * blob);
            const bool ok = held && std::memcmp(b.data(), hs.data(), (size_t) 2 * blob) == 0;
            ++cases;
            if (!ok) { std::printf("    *** wait_fetch_rebase: the wait (held %d) or the copy after it is wrong ***\n", (int) held); ++bad; }
            cudaFree(dcnt); cudaFree(p_src); cudaFree(p_out); cudaFree(st);
        }
        cudaFree(dsrc); cudaFreeHost(hsrc); cudaFreeHost(flag); cudaFree(skip);
    }
    std::printf("  %-60s %s (%d cases)\n", "STRATA_FUSE_FETCH wait_fetch_rebase == wait + fetch + rebase",
                bad ? "*** NO ***" : "bitwise", cases);
    return bad;
}

// STRATA_FUSE_SHPAIR (the binary run with STRATA_FUSE_SHPAIR=1): native_mmvq_pair for Q4_K / Q5_K / Q6_K / IQ4_XS
// vs native_mmvq twice; where the pair declines, nothing may be written.  Random blocks, small finite fp16 scales.
int test_pair(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    const char* env = std::getenv("STRATA_FUSE_SHPAIR");
    const bool any = env == nullptr || std::atoi(env) == 0;   // switch off: native_mmvq_pair_any (the same pairs)
    struct Fmt { int type, bytes; std::vector<int> halves; const char* name; };
    const Fmt fmts[] = {{12, 144, {0, 2}, "Q4_K"}, {13, 176, {0, 2}, "Q5_K"}, {14, 210, {208}, "Q6_K"}, {23, 136, {0}, "IQ4_XS"}};
    int bad = 0, cases = 0, ran = 0;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (const auto& f : fmts) {
        for (const int n_in : {2560, 1024}) {
            for (const int n_out : {512, 640, 1536, 6144, 10240}) {
                const size_t row = (size_t) (n_in / 256) * f.bytes, wb = row * n_out;
                std::vector<uint8_t> w1(wb), w2(wb);
                for (auto* w : {&w1, &w2}) {
                    for (auto& b : *w) b = (uint8_t) rng();
                    for (size_t o = 0; o < wb; o += f.bytes)
                        for (const int h : f.halves) {   // 2^-12 .. 2^-6, either sign
                            const uint16_t v = (uint16_t) ((rng() & 1 ? 0x8000 : 0) | ((3 + rng() % 7) << 10) | (rng() & 1023));
                            std::memcpy(&(*w)[o + h], &v, 2);
                        }
                }
                uint8_t *d1 = dev<uint8_t>(wb), *d2 = dev<uint8_t>(wb);
                up(d1, w1);
                up(d2, w2);
                for (int nc = 2; nc <= 8; ++nc) {
                    std::vector<float> x((size_t) nc * n_in);
                    for (auto& v : x) v = nd(rng);
                    float* dx = dev<float>(x.size());
                    up(dx, x);
                    uint8_t* xq = dev<uint8_t>((size_t) nc * (n_in / 32) * 36 + 256);
                    native_quantize_q8_1(dx, xq, n_in, nc, s);
                    const size_t no = (size_t) nc * n_out;
                    float *a1 = dev<float>(no), *a2 = dev<float>(no), *b1 = dev<float>(no), *b2 = dev<float>(no);
                    for (float* o : {a1, a2, b1, b2}) check(cudaMemset(o, 0x7f, no * 4), "memset");
                    native_mmvq(f.type, d1, xq, a1, n_in, n_out, nc, s);
                    native_mmvq(f.type, d2, xq, a2, n_in, n_out, nc, s);
                    const bool r = any ? native_mmvq_pair_any(f.type, d1, d2, xq, b1, b2, n_in, n_out, nc, s)
                                       : native_mmvq_pair(f.type, d1, d2, xq, b1, b2, n_in, n_out, nc, s);
                    check(cudaStreamSynchronize(s), "sync");
                    ++cases;
                    ran += r;
                    const auto ra = down(a1, no), rb = down(a2, no), ga = down(b1, no), gb = down(b2, no);
                    std::vector<float> untouched(no);
                    std::memset(untouched.data(), 0x7f, no * 4);
                    if (r ? !(same(ra, ga) && same(rb, gb)) : !(same(ga, untouched) && same(gb, untouched))) {
                        std::printf("    *** pair %s %d x %d, %d columns, %s: differs ***\n", f.name, n_in, n_out, nc,
                                    r ? "paired" : "declined");
                        ++bad;
                    }
                    cudaFree(dx); cudaFree(xq); cudaFree(a1); cudaFree(a2); cudaFree(b1); cudaFree(b2);
                }
                cudaFree(d1); cudaFree(d2);
            }
        }
    }
    // two formats (native_mmvq_pair_mixed, the switch on only): every ordered pair, the same checks
    int mbad = 0, mcases = 0, mran = 0;
    for (const auto& f1 : fmts) {
        for (const auto& f2 : fmts) {
            if (any || f1.type == f2.type) continue;
            for (const int n_in : {2560, 1024}) {
                for (const int n_out : {512, 640, 1537, 6144}) {
                    auto make = [&](const Fmt& f) {
                        const size_t wb = (size_t) (n_in / 256) * f.bytes * n_out;
                        std::vector<uint8_t> w(wb);
                        for (auto& b : w) b = (uint8_t) rng();
                        for (size_t o = 0; o < wb; o += f.bytes)
                            for (const int h : f.halves) {
                                const uint16_t v = (uint16_t) ((rng() & 1 ? 0x8000 : 0) | ((3 + rng() % 7) << 10) | (rng() & 1023));
                                std::memcpy(&w[o + h], &v, 2);
                            }
                        uint8_t* d = dev<uint8_t>(wb);
                        up(d, w);
                        return d;
                    };
                    uint8_t *d1 = make(f1), *d2 = make(f2);
                    for (int nc = 2; nc <= 8; ++nc) {
                        std::vector<float> x((size_t) nc * n_in);
                        for (auto& v : x) v = nd(rng);
                        float* dx = dev<float>(x.size());
                        up(dx, x);
                        uint8_t* xq = dev<uint8_t>((size_t) nc * (n_in / 32) * 36 + 256);
                        native_quantize_q8_1(dx, xq, n_in, nc, s);
                        const size_t no = (size_t) nc * n_out;
                        float *a1 = dev<float>(no), *a2 = dev<float>(no), *b1 = dev<float>(no), *b2 = dev<float>(no);
                        for (float* o : {a1, a2, b1, b2}) check(cudaMemset(o, 0x7f, no * 4), "memset");
                        native_mmvq(f1.type, d1, xq, a1, n_in, n_out, nc, s);
                        native_mmvq(f2.type, d2, xq, a2, n_in, n_out, nc, s);
                        const bool r = native_mmvq_pair_mixed(f1.type, f2.type, d1, d2, xq, b1, b2, n_in, n_out, nc, s);
                        check(cudaStreamSynchronize(s), "sync");
                        ++mcases;
                        mran += r;
                        const auto ra = down(a1, no), rb = down(a2, no), ga = down(b1, no), gb = down(b2, no);
                        std::vector<float> untouched(no);
                        std::memset(untouched.data(), 0x7f, no * 4);
                        if (r ? !(same(ra, ga) && same(rb, gb)) : !(same(ga, untouched) && same(gb, untouched))) {
                            std::printf("    *** pair_mixed %s/%s %d x %d, %d columns, %s: differs ***\n", f1.name, f2.name, n_in,
                                        n_out, nc, r ? "paired" : "declined");
                            ++mbad;
                        }
                        cudaFree(dx); cudaFree(xq); cudaFree(a1); cudaFree(a2); cudaFree(b1); cudaFree(b2);
                    }
                    cudaFree(d1); cudaFree(d2);
                }
            }
        }
    }
    if (!any)
        std::printf("  %-60s %s (%d cases, %d paired)\n", "STRATA_FUSE_SHPAIR native_mmvq_pair_mixed == native_mmvq x 2",
                    mbad || mran == 0 ? "*** NO ***" : "bitwise", mcases, mran);
    bad += mbad + (!any && mran == 0);
    std::printf("  %-60s %s (%d cases, %d paired)\n", any ? "native_mmvq_pair_any == native_mmvq x 2"
                                                           : "STRATA_FUSE_SHPAIR native_mmvq_pair == native_mmvq x 2",
                bad || ran == 0 ? "*** NO ***" : "bitwise", cases, ran);
    return bad + (ran == 0);
}

// STRATA_FUSE_SHBEL: native_mmvq_pair_router_bell (the router + bell in block 0, the gate/up pair's blocks at the
// block prefix) vs native_mmvq_pair[_any] then native_router_top10_multi_bell, for every format pair and 2-8 columns.
// All five outputs (y1, y2, ids, weights, the published rows/ids and the ring) are compared byte for byte; the
// published-row buffers start from a sentinel because with every expert resident neither side writes them.
int test_pair_routebell(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    struct Fmt { int type, bytes; std::vector<int> halves; const char* name; };
    const Fmt fmts[] = {{12, 144, {0, 2}, "Q4_K"}, {13, 176, {0, 2}, "Q5_K"}, {14, 210, {208}, "Q6_K"}, {23, 136, {0}, "IQ4_XS"}};
    const int N = 2560, NO = 640, NE = 512, K = 10;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto make_dev = [&](const Fmt& f) {
        const size_t wb = (size_t) (N / 256) * f.bytes * NO;
        std::vector<uint8_t> w(wb);
        for (auto& b : w) b = (uint8_t) rng();
        for (size_t o = 0; o < wb; o += f.bytes)
            for (const int h : f.halves) {   // 2^-12 .. 2^-6, either sign
                const uint16_t v = (uint16_t) ((rng() & 1 ? 0x8000 : 0) | ((3 + rng() % 7) << 10) | (rng() & 1023));
                std::memcpy(&w[o + h], &v, 2);
            }
        uint8_t* d = dev<uint8_t>(wb);
        up(d, w);
        return d;
    };
    int bad = 0, cases = 0, declined = 0;
    for (int a = 0; a < 4; ++a) {
        for (int b = a; b < 4; ++b) {
            uint8_t* d1 = make_dev(fmts[a]);
            uint8_t* d2 = make_dev(fmts[b]);
            for (int nc = 2; nc <= 8; ++nc) {
                for (const int miss : {0, 1}) {
                    std::vector<float> xv((size_t) nc * N), lg((size_t) nc * NE);
                    for (auto& v : xv) v = nd(rng);
                    for (auto& v : lg) v = nd(rng) * 2.0f;
                    std::vector<int32_t> res(NE, 0);
                    if (miss) for (int e = 0; e < NE; e += 5) res[(size_t) e] = -1;
                    float* dx = dev<float>(xv.size());
                    float* dl = dev<float>(lg.size());
                    int32_t* dres = dev<int32_t>(NE);
                    up(dx, xv);
                    up(dl, lg);
                    up(dres, res);
                    uint8_t* xq = dev<uint8_t>((size_t) nc * (N / 32) * 36 + 256);
                    native_quantize_q8_1(dx, xq, N, nc, s);
                    const size_t nk = (size_t) nc * K, nx = (size_t) nc * N, no = (size_t) nc * NO;
                    float *y1r = dev<float>(no), *y2r = dev<float>(no), *y1f = dev<float>(no), *y2f = dev<float>(no);
                    int32_t *ir = dev<int32_t>(nk), *ifu = dev<int32_t>(nk), *ior = dev<int32_t>(nk), *iof = dev<int32_t>(nk);
                    float *wr = dev<float>(nk), *wf = dev<float>(nk);
                    float *xor_ = dev<float>(nx), *xof = dev<float>(nx);
                    uint32_t *seqr = dev<uint32_t>(1), *seqf = dev<uint32_t>(1);
                    for (float* o : {y1r, y2r, y1f, y2f, xof}) check(cudaMemset(o, 0x3c, (o == xof ? nx : no) * 4), "memset");
                    check(cudaMemset(xor_, 0x3c, nx * 4), "memset");
                    for (int32_t* o : {ir, ifu, ior, iof}) check(cudaMemset(o, 0x5a, nk * 4), "memset");
                    for (float* o : {wr, wf}) check(cudaMemset(o, 0x5a, nk * 4), "memset");
                    const uint32_t zero = 0;
                    check(cudaMemcpy(seqr, &zero, 4, cudaMemcpyHostToDevice), "seq");
                    check(cudaMemcpy(seqf, &zero, 4, cudaMemcpyHostToDevice), "seq");
                    bool r = false;
                    if (fmts[a].type == fmts[b].type)
                        r = native_mmvq_pair_any(fmts[a].type, d1, d2, xq, y1r, y2r, N, NO, nc, s);
                    else
                        r = native_mmvq_pair_mixed(fmts[a].type, fmts[b].type, d1, d2, xq, y1r, y2r, N, NO, nc, s);
                    native_router_top10_multi_bell(dl, ir, wr, nc, dx, dres, NE, (int64_t) nx, (int64_t) nk, xor_, ior, seqr, s);
                    const bool f = native_mmvq_pair_router_bell(
                        fmts[a].type, fmts[b].type, d1, d2, xq, y1f, y2f, N, NO, nc, dl, ifu, wf, nc, dx, dres, NE,
                        (int64_t) nx, (int64_t) nk, xof, iof, seqf, s);
                    check(cudaStreamSynchronize(s), "sync");
                    ++cases;
                    if (!r || !f) {
                        std::printf("    *** pairbell %s/%s %d columns miss %d: declined (ref %d fused %d) ***\n", fmts[a].name,
                                    fmts[b].name, nc, miss, (int) r, (int) f);
                        ++declined;
                    } else {
                        const auto a1 = down(y1r, no), a2 = down(y2r, no), b1 = down(y1f, no), b2 = down(y2f, no);
                        const auto vi = down(ir, nk);
                        const auto vif = down(ifu, nk);
                        const auto wo = down(wr, nk);
                        const auto wof = down(wf, nk);
                        const auto xo = down(xor_, nx), xof_h = down(xof, nx);
                        const auto io = down(ior, nk), iof_h = down(iof, nk);
                        const auto sr = down(seqr, 1), sf = down(seqf, 1);
                        if (!same(a1, b1) || !same(a2, b2) || !same(vi, vif) || !same(wo, wof) || !same(xo, xof_h) ||
                            !same(io, iof_h) || sr[0] != sf[0]) {
                            std::printf("    *** pairbell %s/%s %d columns miss %d differs ***\n", fmts[a].name, fmts[b].name,
                                        nc, miss);
                            ++bad;
                        }
                    }
                    cudaFree(dx); cudaFree(dl); cudaFree(dres); cudaFree(xq);
                    cudaFree(y1r); cudaFree(y2r); cudaFree(y1f); cudaFree(y2f);
                    cudaFree(ir); cudaFree(ifu); cudaFree(ior); cudaFree(iof);
                    cudaFree(wr); cudaFree(wf); cudaFree(xor_); cudaFree(xof); cudaFree(seqr); cudaFree(seqf);
                }
            }
            cudaFree(d1); cudaFree(d2);
        }
    }
    std::printf("  %-60s %s (%d cases, %d declined)\n", "STRATA_FUSE_SHBEL router+bell+pair == pair + router_bell",
                bad || declined || cases == 0 ? "*** NO ***" : "bitwise", cases, declined);
    return bad + declined + (cases == 0);
}

// STRATA_FUSE_QGEMV: bf16_gemv_fp32_mmvf_multi_quant vs the router's GEMV (with and without the shared-gate aux row)
// then native_quantize_q8_1: every GEMV output and every q8_1 byte.
int test_qgemv(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    const int N = 2560, NO = 512;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto make_w = [&](int rows) {
        std::vector<uint16_t> w((size_t) rows * N);
        for (auto& v : w) v = (uint16_t) (rng() & 0x3fff) | ((rng() & 1) << 15);
        uint16_t* d = dev<uint16_t>(w.size());
        up(d, w);
        return d;
    };
    uint16_t* w = make_w(NO);
    uint16_t* wa = make_w(1);
    int bad = 0, cases = 0, declined = 0;
    for (int nc = 2; nc <= 8; ++nc) {
        for (const int aux : {0, 1}) {
            std::vector<float> x((size_t) nc * N);
            for (auto& v : x) v = nd(rng);
            float* dx = dev<float>(x.size());
            up(dx, x);
            const size_t no = (size_t) nc * NO, nq = (size_t) nc * (N / 32) * 36;
            float *yr = dev<float>(no), *yf = dev<float>(no), *ar = dev<float>(nc), *af = dev<float>(nc);
            uint8_t* qr = dev<uint8_t>(nq + 64);
            uint8_t* qf = dev<uint8_t>(nq + 64);
            for (float* o : {yr, yf}) check(cudaMemset(o, 0x3c, no * 4), "memset");
            for (float* o : {ar, af}) check(cudaMemset(o, 0x3c, nc * 4), "memset");
            check(cudaMemset(qr, 0x5a, nq + 64), "memset");
            check(cudaMemset(qf, 0x5a, nq + 64), "memset");
            if (aux) bf16_gemv_fp32_mmvf_multi_aux(dx, N, w, yr, NO, N, NO, nc, wa, ar, 1, s);
            else bf16_gemv_fp32_mmvf_multi(dx, N, w, yr, NO, N, NO, nc, s);
            native_quantize_q8_1(dx, qr, N, nc, s);
            const bool f = bf16_gemv_fp32_mmvf_multi_quant(dx, N, w, yf, NO, N, NO, nc, aux ? wa : nullptr,
                                                            aux ? af : nullptr, aux ? 1 : 0, dx, qf, (int64_t) nc * N, s);
            check(cudaStreamSynchronize(s), "sync");
            ++cases;
            if (!f) {
                std::printf("    *** qgemv %d columns aux %d: declined ***\n", nc, aux);
                ++declined;
            } else {
                const auto y1 = down(yr, no), y2 = down(yf, no);
                const auto a1 = down(ar, (size_t) nc), a2 = down(af, (size_t) nc);
                const auto b1 = down(qr, nq + 64), b2 = down(qf, nq + 64);
                if (!same(y1, y2) || (aux && !same(a1, a2)) || !same(b1, b2)) {
                    std::printf("    *** qgemv %d columns aux %d differs ***\n", nc, aux);
                    ++bad;
                }
            }
            cudaFree(dx); cudaFree(yr); cudaFree(yf); cudaFree(ar); cudaFree(af); cudaFree(qr); cudaFree(qf);
        }
    }
    cudaFree(w); cudaFree(wa);
    std::printf("  %-60s %s (%d cases, %d declined)\n", "STRATA_FUSE_QGEMV router GEMV + q8_1 == GEMV + quantize",
                bad || declined || cases == 0 ? "*** NO ***" : "bitwise", cases, declined);
    return bad + declined + (cases == 0);
}

// strict: every byte; nan_cases: also NaN inputs, where two NaN operands meet in one add and the hardware keeps one
// of their sign / payload bits by operand order, which the compiler may commute: a NaN result must stay a NaN there
bool g_nan = false;
// mismatching floats: -1 if any differs other than NaN against NaN, else the count of NaN-vs-NaN bit differences
long diff_floats(const std::vector<float>& a, const std::vector<float>& b, size_t* first) {
    long nanx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], 4) == 0) continue;
        if (std::isnan(a[i]) && std::isnan(b[i])) { ++nanx; continue; }
        *first = i;
        return -1;
    }
    return nanx;
}
// random floats with the awkward values in: -0.0, +0.0, denormals, large, inf (and NaN when g_nan)
float odd_float(std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    switch (rng() % 64) {
        case 0: return -0.0f;
        case 1: return 0.0f;
        case 2: return 1e-40f * (rng() % 2 ? 1.0f : -1.0f);
        case 3: return 3e30f * nd(rng);
        case 4: return g_nan && rng() % 2 ? (rng() % 2 ? std::nanf("") : -std::nanf("")) : (rng() % 2 ? 1.0f : -1.0f) * std::numeric_limits<float>::infinity();
        default: return nd(rng);
    }
}

// STRATA_FUSE_COMBINE: native_moe_combine_multi_hits_mapped vs
//   copy_rows_from_mapped_after + native_moe_combine_multi_hits_added                  (skip == nullptr)
//   wait_flag_ge_or + copy_or_zero_from_mapped + native_moe_combine_multi_hits_added   (skip: the device plan)
// gated (STRATA_FUSE_SHGATE): the reference scales `shared` by sigmoid(sg) first (shared_expert_multi's last launch)
int test_combine(std::mt19937& rng, cudaStream_t s, bool gated) {
    using namespace strata::kernels;
    native_moe_combine_set_enabled(true);
    int bad = 0, cases = 0;
    long nan_bits = 0;
    const int K = 10;
    for (const int N : {2560, 2048, 64}) {
        for (int n = 1; n <= 8; ++n) {
            for (const int dp : {0, 1}) {
                for (const int sk : {0, 1}) {
                    if (!dp && sk) continue;
                    for (const int hit_kind : {0, 1, 2}) {   // no hits, some, all
                        const size_t rows = (size_t) n * K, nf = rows * N;
                        std::vector<float> mp(nf), hits(nf), w(rows), sh((size_t) n * N);
                        for (auto& v : mp) v = odd_float(rng);
                        for (auto& v : hits) v = odd_float(rng);
                        for (auto& v : w) v = (rng() % 16 == 0) ? -0.0f : (float) (rng() % 1000) / 997.0f;
                        for (auto& v : sh) v = odd_float(rng);
                        std::vector<int32_t> ent(rows);
                        for (size_t i = 0; i < rows; ++i) ent[i] = (int32_t) i;
                        std::shuffle(ent.begin(), ent.end(), rng);
                        const int nh = hit_kind == 0 ? 0 : hit_kind == 2 ? (int) rows : (int) (rng() % rows);
                        std::vector<int32_t> dsth(rows, -1);
                        for (int i = 0; i < nh; ++i) dsth[i] = ent[i];
                        std::vector<int32_t> cnt = {3, nh, 0, 0};
                        float* m_d = nullptr;
                        float* m_h = mapped<float>(nf, &m_d);
                        std::memcpy(m_h, mp.data(), nf * 4);
                        uint32_t* flag_d = nullptr;
                        uint32_t* flag = mapped<uint32_t>(4, &flag_d);
                        const uint32_t ring = 5 + (uint32_t) (rng() % 50);
                        *(volatile uint32_t*) flag = ring;
                        uint32_t* skip = dev<uint32_t>(4);
                        const uint32_t skv = sk ? ring : ring + 1;
                        check(cudaMemcpy(skip, &skv, 4, cudaMemcpyHostToDevice), "skip");
                        float *parts = dev<float>(nf), *hd = dev<float>(nf), *wd = dev<float>(rows), *shd = dev<float>((size_t) n * N);
                        float *o1 = dev<float>((size_t) n * N), *o2 = dev<float>((size_t) n * N);
                        int32_t *dd = dev<int32_t>(rows), *cd = dev<int32_t>(4);
                        std::vector<float> junk(nf);
                        for (auto& v : junk) v = odd_float(rng);
                        up(parts, junk);
                        up(hd, hits); up(wd, w); up(shd, sh); up(dd, dsth); up(cd, cnt);
                        check(cudaMemset(o1, 0x7f, (size_t) n * N * 4), "memset");
                        check(cudaMemset(o2, 0x3c, (size_t) n * N * 4), "memset");
                        if (dp) {
                            wait_flag_ge_or(flag_d, ring, skip, s);
                            copy_or_zero_from_mapped(parts, m_d, (long long) nf, skip, ring, s);
                        } else {
                            copy_rows_from_mapped_after(flag_d, ring, parts, m_d, (int64_t) rows, N, dd, cd + 1, s);
                        }
                        float* sgd = dev<float>(8);
                        float* shs = dev<float>((size_t) n * N);   // the scaled copy (reference)
                        if (gated) {
                            std::vector<float> sg(8);
                            for (auto& v : sg) v = rng() % 8 == 0 ? odd_float(rng) : 4.0f * std::normal_distribution<float>(0, 1)(rng);
                            up(sgd, sg);
                            check(cudaMemcpy(shs, shd, (size_t) n * N * 4, cudaMemcpyDeviceToDevice), "copy");
                            shared_expert_sigmoid_scale_rows(shs, sgd, N, n, s);
                        }
                        native_moe_combine_multi_hits_added(parts, hd, dd, cd + 1, wd, gated ? shs : shd, o1, N, K, n, s);
                        native_moe_combine_multi_hits_mapped(m_d, hd, dd, cd + 1, wd, shd, o2, N, K, n, flag_d, ring,
                                                             dp ? skip : nullptr, gated ? sgd : nullptr, s);
                        check(cudaStreamSynchronize(s), "sync");
                        const auto a = down(o1, (size_t) n * N), b = down(o2, (size_t) n * N);
                        ++cases;
                        size_t i = 0;
                        const long d = diff_floats(a, b, &i);
                        if (d < 0 || (d > 0 && !g_nan)) {
                            std::printf("    *** combine mapped N %d n %d dp %d skip %d hits %d differs at %zu: %a vs %a ***\n",
                                        N, n, dp, sk, nh, i, a[i], b[i]);
                            ++bad;
                        }
                        nan_bits += d > 0 ? d : 0;
                        cudaFreeHost(m_h); cudaFreeHost(flag); cudaFree(skip); cudaFree(parts); cudaFree(hd); cudaFree(wd);
                        cudaFree(shd); cudaFree(sgd); cudaFree(shs); cudaFree(o1); cudaFree(o2); cudaFree(dd); cudaFree(cd);
                    }
                }
            }
        }
    }
    std::printf("  %-60s %s (%d cases%s)\n", gated ? "STRATA_FUSE_SHGATE combine_hits_mapped == scale + copy + combine"
                                                     : "STRATA_FUSE_COMBINE combine_hits_mapped == copy + combine_hits",
                bad ? "*** NO ***" : g_nan ? "bitwise but NaN-vs-NaN" : "bitwise", cases,
                g_nan ? (", NaN inputs: " + std::to_string(nan_bits) + " NaN outputs differ in sign/payload only").c_str() : "");
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    int dc = 0;
    if (cudaGetDeviceCount(&dc) != cudaSuccess || dc == 0) { std::printf("no GPU: skipped\n"); return 77; }
    const std::string only = argc > 1 ? argv[1] : "";
    std::mt19937 rng(1234);
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    int bad = 0;
    if (only.empty() || only == "fetch") bad += test_fetch(rng, s);
    if (only.empty() || only == "pair") bad += test_pair(rng, s);
    if (only.empty() || only == "pairbell") bad += test_pair_routebell(rng, s);
    if (only.empty() || only == "qgemv") bad += test_qgemv(rng, s);
    if (only.empty() || only == "plan") bad += test_plan(rng, s);
    if (only.empty() || only == "combine") {
        for (const bool gated : {false, true}) {
            bad += test_combine(rng, s, gated);
            g_nan = true;   // the same with NaN inputs: every non-NaN output bitwise, NaN outputs NaN
            bad += test_combine(rng, s, gated);
            g_nan = false;
        }
    }
    std::printf("fuse_moe_parity: %s\n", bad ? "FAILED" : "all bitwise");
    return bad ? 1 : 0;
}
