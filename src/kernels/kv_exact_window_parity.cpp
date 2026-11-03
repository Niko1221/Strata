// src/kernels/kv_exact_window_parity.cpp - the exact KV window (GPU, synthetic, no model): the sinks stay, the ring
// evicts in position order, a gather reports hits and misses correctly, and the values come back exactly (the
// pattern is integer-valued, so fp16 is lossless here and any difference is a bug).
#include "strata/kernels/kv_exact_window.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace K = strata::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
float pat(int pos, int h, int d) { return (float) (pos * 10 + h * 2 + (d & 1)); }
}  // namespace

int main() {
    const K::QsaShapes s = K::qsa_real_shapes();
    const int layers = 2, sinks = 4, tail = 8, stride = 4;
    const int heads = (int) s.n_head_kv, hd = (int) s.head_dim;
    K::KvExactWindow w;
    w.layers = layers; w.sinks = sinks; w.tail = tail; w.n_head_kv = heads; w.head_dim = hd; w.pos_stride = stride;
    const size_t rows = (size_t) layers * (sinks + tail);
    const size_t row_vals = (size_t) heads * hd;
    ck(cudaMalloc(&w.k, rows * row_vals * 2), "malloc k");
    ck(cudaMalloc(&w.v, rows * row_vals * 2), "malloc v");
    ck(cudaMalloc(&w.tag, rows * sizeof(int32_t)), "malloc tag");
    K::kv_exact_window_init(w, s, layers, sinks, tail, stride, nullptr);

    // append positions 0..19 one cell at a time, layer 0 only (layer 1 stays empty on purpose)
    const int n_pos = 20;
    int32_t* d_step = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;
    ck(cudaMalloc(&d_step, stride * sizeof(int32_t)), "malloc step");
    ck(cudaMalloc(&d_k, row_vals * sizeof(float)), "malloc kcur");
    ck(cudaMalloc(&d_v, row_vals * sizeof(float)), "malloc vcur");
    std::vector<float> hkv(row_vals);
    for (int pos = 0; pos < n_pos; ++pos) {
        std::vector<int32_t> hstep(stride, -7);
        hstep[K::kStepPos] = pos;
        for (int h = 0; h < heads; ++h)
            for (int d = 0; d < hd; ++d) hkv[(size_t) h * hd + d] = pat(pos, h, d);
        ck(cudaMemcpy(d_step, hstep.data(), stride * sizeof(int32_t), cudaMemcpyHostToDevice), "step");
        ck(cudaMemcpy(d_k, hkv.data(), row_vals * sizeof(float), cudaMemcpyHostToDevice), "k");
        ck(cudaMemcpy(d_v, hkv.data(), row_vals * sizeof(float), cudaMemcpyHostToDevice), "v");
        K::kv_exact_window_append(w, 0, d_step, d_k, d_v, 1, nullptr);
    }
    ck(cudaDeviceSynchronize(), "after appends");

    // gather: sinks (0..3), the newest eight (12..19) and two that must miss (11, 99)
    const int want[] = {0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19, 11, 99};
    const int n = (int) (sizeof(want) / sizeof(want[0]));
    int32_t* d_pos = nullptr;
    int32_t* d_hit = nullptr;
    uint16_t* d_ko = nullptr;
    uint16_t* d_vo = nullptr;
    ck(cudaMalloc(&d_pos, n * sizeof(int32_t)), "malloc pos");
    ck(cudaMalloc(&d_hit, n * sizeof(int32_t)), "malloc hit");
    ck(cudaMalloc(&d_ko, (size_t) n * row_vals * 2), "malloc ko");
    ck(cudaMalloc(&d_vo, (size_t) n * row_vals * 2), "malloc vo");
    ck(cudaMemcpy(d_pos, want, n * sizeof(int32_t), cudaMemcpyHostToDevice), "pos");
    K::kv_exact_window_gather(w, 0, d_pos, n, d_ko, d_vo, d_hit, nullptr);
    ck(cudaDeviceSynchronize(), "after gather");

    std::vector<int32_t> hit(n);
    std::vector<uint16_t> ko((size_t) n * row_vals), vo((size_t) n * row_vals);
    ck(cudaMemcpy(hit.data(), d_hit, n * sizeof(int32_t), cudaMemcpyDeviceToHost), "hit");
    ck(cudaMemcpy(ko.data(), d_ko, ko.size() * 2, cudaMemcpyDeviceToHost), "ko");
    ck(cudaMemcpy(vo.data(), d_vo, vo.size() * 2, cudaMemcpyDeviceToHost), "vo");

    int bad = 0;
    std::printf("pos  expect  hit  values\n");
    for (int i = 0; i < n; ++i) {
        const bool expect = want[i] < sinks || (want[i] >= n_pos - tail && want[i] < n_pos);
        int vbad = 0;
        if (hit[i]) {
            for (int h = 0; h < heads; ++h)
                for (int d = 0; d < hd; ++d) {
                    const uint16_t want_bits = K::f16_from_f32(pat(want[i], h, d));
                    if (ko[(size_t) i * row_vals + (size_t) h * hd + d] != want_bits) ++vbad;
                    if (vo[(size_t) i * row_vals + (size_t) h * hd + d] != want_bits) ++vbad;
                }
        }
        const bool ok = (hit[i] != 0) == expect && vbad == 0;
        if (!ok) ++bad;
        std::printf("%3d  %-6s  %-3s  %s\n", want[i], expect ? "hit" : "miss", hit[i] ? "hit" : "miss",
                    vbad ? "WRONG VALUES" : (ok ? "ok" : "unexpected"));
    }

    // ---- the prompt path: one contiguous block, positions 0..19 in one call, `end` = 20.  Only the sinks and the
    // last `tail` cells may land; everything in between can never be in the finished prompt's window.
    K::KvExactWindow w2;
    w2.layers = 1; w2.sinks = sinks; w2.tail = tail; w2.n_head_kv = heads; w2.head_dim = hd; w2.pos_stride = stride;
    ck(cudaMalloc(&w2.k, rows * row_vals * 2), "malloc k2");
    ck(cudaMalloc(&w2.v, rows * row_vals * 2), "malloc v2");
    ck(cudaMalloc(&w2.tag, rows * sizeof(int32_t)), "malloc tag2");
    K::kv_exact_window_init(w2, s, 1, sinks, tail, stride, nullptr);
    {
        std::vector<float> blk((size_t) n_pos * row_vals);
        for (int pos = 0; pos < n_pos; ++pos)
            for (int h = 0; h < heads; ++h)
                for (int d = 0; d < hd; ++d) blk[(size_t) pos * row_vals + (size_t) h * hd + d] = pat(pos, h, d);
        float* d_kb = nullptr;
        float* d_vb = nullptr;
        ck(cudaMalloc(&d_kb, blk.size() * 4), "malloc range k");
        ck(cudaMalloc(&d_vb, blk.size() * 4), "malloc range v");
        ck(cudaMemcpy(d_kb, blk.data(), blk.size() * 4, cudaMemcpyHostToDevice), "range k");
        ck(cudaMemcpy(d_vb, blk.data(), blk.size() * 4, cudaMemcpyHostToDevice), "range v");
        K::kv_exact_window_append_range(w2, 0, 0, n_pos, d_kb, d_vb, n_pos, nullptr);
        ck(cudaDeviceSynchronize(), "after range append");
        std::vector<int32_t> all(n_pos);
        for (int i = 0; i < n_pos; ++i) all[(size_t) i] = i;
        int32_t* d_all = nullptr;
        uint16_t* d_ko2 = nullptr;
        uint16_t* d_vo2 = nullptr;
        ck(cudaMalloc(&d_all, n_pos * 4), "malloc all");
        ck(cudaMalloc(&d_ko2, (size_t) n_pos * row_vals * 2), "malloc ko2");
        ck(cudaMalloc(&d_vo2, (size_t) n_pos * row_vals * 2), "malloc vo2");
        ck(cudaMemcpy(d_all, all.data(), n_pos * 4, cudaMemcpyHostToDevice), "all");
        // hit = null: the "gather after the quantized pool" form the engine uses
        K::kv_exact_window_gather(w2, 0, d_all, n_pos, d_ko2, d_vo2, nullptr, nullptr);
        ck(cudaDeviceSynchronize(), "after range gather");
        std::vector<uint16_t> ko2((size_t) n_pos * row_vals);
        ck(cudaMemcpy(ko2.data(), d_ko2, ko2.size() * 2, cudaMemcpyDeviceToHost), "ko2");
        int rbad = 0;
        for (int pos = 0; pos < n_pos; ++pos) {
            const bool expect = pos < sinks || pos >= n_pos - tail;
            int vbad = 0;
            if (expect)
                for (int h = 0; h < heads; ++h)
                    for (int d = 0; d < hd; ++d)
                        if (ko2[(size_t) pos * row_vals + (size_t) h * hd + d] != K::f16_from_f32(pat(pos, h, d))) ++vbad;
            if (vbad) ++rbad;
        }
        std::printf("\nprompt range 0..%d with end=%d: sinks and the last %d cells written, the rest skipped: %s\n",
                    n_pos - 1, n_pos, tail, rbad ? "WRONG" : "ok");
        if (rbad) ++bad;
    }

    const double per_layer = (double) K::kv_exact_window_bytes_per_layer(s, 4, 64);
    std::printf("\nwindow cost at the real geometry (sinks 4 + tail 64, one layer): %.2f MiB, x12 layers %.1f MiB\n",
                per_layer / (1024 * 1024), 12 * per_layer / (1024 * 1024));
    std::printf("%s\n", bad ? "FAIL" : "ok: sinks kept, ring evicts in order, values exact");
    return bad ? 1 : 0;
}
