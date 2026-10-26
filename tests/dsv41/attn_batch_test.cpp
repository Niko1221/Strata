// Strata. The attention kernel with T queries over two disjoint regions against the same queries one at a time: the
// property M2 stands on (docs/DSV41_BATCHING.md, "M2, re-scoped").  GPU, small shapes, no model.
//
//   ds4_attn_batch_test
//
// Query 0 reads window rows [0, 8) and compressed rows [0, 4); query 1 window rows [8, 16) and compressed rows [16, 20).
// The two together in one call must give exactly what each gives alone.
#include "ds4_kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace strata::ds4;

static int fails = 0;
#define CHECK(x)                                                                                    \
    do {                                                                                            \
        if (!(x)) {                                                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);                                \
            ++fails;                                                                                \
        }                                                                                           \
    } while (0)

static uint16_t bf(float v) {   // truncating float -> BF16, enough for a test's data and stable
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return (uint16_t) (b >> 16);
}

int main() {
    const int heads = 8, d = 512, T = 2, nwin = 8, nsel = 4, rows = 32;
    const float scale = 1.0f / 16.0f;
    std::vector<float> q((size_t) T * heads * d), sink((size_t) heads), out2((size_t) T * heads * d), o1a((size_t) heads * d),
        o1b((size_t) heads * d);
    std::vector<uint16_t> win((size_t) 2 * nwin * d), ckv((size_t) rows * d);
    std::vector<int> sel((size_t) T * nsel);
    unsigned s = 12345;
    auto rnd = [&] { s = s * 1664525u + 1013904223u; return ((float) (s >> 8) / (float) (1u << 24)) - 0.5f; };
    for (auto& v : q) v = rnd() * 2.0f;
    for (auto& v : win) v = bf(rnd());
    for (auto& v : ckv) v = bf(rnd());
    for (auto& v : sink) v = rnd();
    for (int i = 0; i < nsel; ++i) {   // the sequences' own rows
        sel[(size_t) i] = i;                     // query 0: compressed rows 0..3
        sel[(size_t) nsel + i] = 16 + i;         // query 1: compressed rows 16..19
    }
    float *dq = nullptr, *dsink = nullptr, *dout2 = nullptr, *do1a = nullptr, *do1b = nullptr;
    uint16_t *dwin = nullptr, *dckv = nullptr;
    int* dsel = nullptr;
    CHECK(cudaMalloc(&dq, q.size() * 4) == cudaSuccess);
    CHECK(cudaMalloc(&dwin, win.size() * 2) == cudaSuccess);
    CHECK(cudaMalloc(&dckv, ckv.size() * 2) == cudaSuccess);
    CHECK(cudaMalloc(&dsel, sel.size() * 4) == cudaSuccess);
    CHECK(cudaMalloc(&dsink, sink.size() * 4) == cudaSuccess);
    CHECK(cudaMalloc(&dout2, out2.size() * 4) == cudaSuccess);
    CHECK(cudaMalloc(&do1a, o1a.size() * 4) == cudaSuccess);
    CHECK(cudaMalloc(&do1b, o1b.size() * 4) == cudaSuccess);
    CHECK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(dwin, win.data(), win.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(dckv, ckv.data(), ckv.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(dsel, sel.data(), sel.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(dsink, sink.data(), sink.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);

    // one call, two queries, each on its own rows
    AttnRows a2;
    a2.gen = 0;
    a2.nwin[0] = nwin; a2.nwin[1] = nwin;
    a2.nsel[0] = nsel; a2.nsel[1] = nsel;
    a2.win0[0] = 0;    a2.win0[1] = nwin;
    sparse_attn(dq, dwin, dckv, dsel, nsel, a2, dsink, heads, d, T, scale, dout2, nullptr, kKvBf16);

    // the same queries, one at a time
    AttnRows a1;
    a1.gen = 0;
    a1.nwin[0] = nwin;
    a1.nsel[0] = nsel;
    a1.win0[0] = 0;
    sparse_attn(dq, dwin, dckv, dsel, nsel, a1, dsink, heads, d, 1, scale, do1a, nullptr, kKvBf16);
    a1.win0[0] = nwin;
    sparse_attn(dq + (size_t) heads * d, dwin, dckv, dsel + nsel, nsel, a1, dsink, heads, d, 1, scale, do1b, nullptr,
                kKvBf16);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(cudaMemcpy(out2.data(), dout2, out2.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(o1a.data(), do1a, o1a.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(o1b.data(), do1b, o1b.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess);

    const float* q0 = out2.data();
    const float* q1 = out2.data() + (size_t) heads * d;
    int bad0 = 0, bad1 = 0;
    for (int i = 0; i < heads * d; ++i) {
        if (q0[i] != o1a[i]) ++bad0;
        if (q1[i] != o1b[i]) ++bad1;
    }
    CHECK(bad0 == 0);
    CHECK(bad1 == 0);
    if (bad0 || bad1) std::printf("  %d / %d values differ (query 0 / query 1)\n", bad0, bad1);

    cudaFree(dq); cudaFree(dwin); cudaFree(dckv); cudaFree(dsel);
    cudaFree(dsink); cudaFree(dout2); cudaFree(do1a); cudaFree(do1b);
    if (fails) {
        std::printf("%d check(s) failed\n", fails);
        return 1;
    }
    std::printf("attn_batch_test: ok\n");
    return 0;
}
