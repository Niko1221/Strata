// src/qwen35/qwen35_gpu.cu - the GPU (HIP/CUDA) dense matvec for Qwen35MoE.
//
// The token's cost is dominated by GGUF matvecs.  This uploads every DENSE projection (attention, GDN, shared
// expert, output head; ~2.7 GB, which fits beside the Qwen4Exp engine's 10 GiB budget) to VRAM once, and runs
// each matvec through the engine's existing `native_mmvq` kernels.  Routed experts are NOT uploaded (the full
// set is ~28 GB); they stay on ggml-cpu.  The activation is quantized with `native_quantize_q8_1` on device.
//
// This is a first GPU tier: each matvec copies its activation in and its result out, which is fine at these
// widths (a few KB) and keeps the whole layer structure in one place.  A future tier keeps the activations
// resident and moves the GDN/attention/MoE kernels themselves to the GPU.
#include "strata/qwen35/qwen35.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace strata::qwen35 {
namespace {

cudaStream_t g_stream = nullptr;
float* d_x = nullptr;
float* d_y = nullptr;
float* d_out = nullptr;
void* d_q8 = nullptr;
void* d_weights = nullptr;
int64_t cap_x = 0, cap_y = 0, cap_out = 0;
size_t cap_q8 = 0, cap_w = 0;
uint64_t g_dense = 0;
const float* g_last_x = nullptr;
int64_t g_last_n = 0;

bool ensure_in(int64_t n) {
    if (n <= cap_x) return true;
    if (d_x) cudaFree(d_x);
    if (cudaMalloc((void**) &d_x, (size_t) n * sizeof(float)) != cudaSuccess) return false;
    cap_x = n;
    g_last_x = nullptr;
    return true;
}
bool ensure_q8(size_t sb) {
    if (sb <= cap_q8) return true;
    if (d_q8) cudaFree(d_q8);
    if (cudaMalloc(&d_q8, sb) != cudaSuccess) return false;
    cap_q8 = sb;
    g_last_x = nullptr;
    return true;
}

bool gpu_matvec(const Mat& m, const float* x, float* y) {
    if (!m.dev || !g_stream) return false;
    if (m.n_in > cap_x) {
        if (d_x) cudaFree(d_x);
        if (cudaMalloc((void**) &d_x, (size_t) m.n_in * sizeof(float)) != cudaSuccess) return false;
        cap_x = m.n_in;
    }
    if (m.n_out > cap_y) {
        if (d_y) cudaFree(d_y);
        if (cudaMalloc((void**) &d_y, (size_t) m.n_out * sizeof(float)) != cudaSuccess) return false;
        cap_y = m.n_out;
    }
    if (!ensure_in(m.n_in) || !ensure_q8(kernels::native_q8_1_bytes((int) m.n_in, 1))) return false;
    cudaMemcpyAsync(d_x, x, (size_t) m.n_in * sizeof(float), cudaMemcpyHostToDevice, g_stream);
    kernels::native_quantize_q8_1(d_x, d_q8, (int) m.n_in, 1, g_stream);
    kernels::native_mmvq(m.type, m.dev, d_q8, d_y, (int) m.n_in, (int) m.n_out, 1, g_stream);
    cudaMemcpyAsync(y, d_y, (size_t) m.n_out * sizeof(float), cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);
    return true;
}

/// Several projections of the same width, each with its own input, in ONE upload/sync/download.  The input is
/// re-uploaded only when it changes (the GDN's four projections share one activation), and the outputs are all
/// downloaded after the last kernel, so a layer needs one round trip instead of one per matvec.
int gpu_batch(const Mat* const* mats, const float* const* xs, float* const* ys, int count) {
    if (!g_stream || count <= 0) return 0;
    const int64_t n_in = mats[0]->n_in;
    size_t maxw = 0;
    for (int i = 0; i < count; ++i) {
        if (mats[i]->n_in != n_in || !mats[i]->dev) return 0;   // device-resident only; streaming measured slower than ggml-cpu
        if (!kernels::native_mmvq_supported(mats[i]->type)) return 0;
        if (!mats[i]->dev) maxw = std::max(maxw, mats[i]->row_bytes * (size_t) mats[i]->n_out);
    }
    if (!ensure_in(n_in) || !ensure_q8(kernels::native_q8_1_bytes((int) n_in, 1))) return 0;
    // Weights the GPU tier did not upload (the routed experts) stream through one reusable device buffer: on a
    // single stream the next upload cannot overwrite the buffer before this matvec has read it.
    if (maxw > 0) {
        if (d_weights == nullptr || maxw > cap_w) {
            if (d_weights) cudaFree(d_weights);
            if (cudaMalloc(&d_weights, maxw) != cudaSuccess) return 0;
            cap_w = maxw;
        }
    }
    int64_t total = 0;
    for (int i = 0; i < count; ++i) total += mats[i]->n_out;
    if (total > cap_out) {
        if (d_out) cudaFree(d_out);
        if (cudaMalloc((void**) &d_out, (size_t) total * sizeof(float)) != cudaSuccess) return 0;
        cap_out = total;
    }
    int64_t off = 0;
    for (int i = 0; i < count; ++i) {
        const void* wdev = mats[i]->dev;
        if (!wdev) {
            cudaMemcpyAsync(d_weights, mats[i]->data, mats[i]->row_bytes * (size_t) mats[i]->n_out,
                            cudaMemcpyHostToDevice, g_stream);
            wdev = d_weights;
        }
        if (xs[i] != g_last_x || g_last_n != n_in) {
            cudaMemcpyAsync(d_x, xs[i], (size_t) n_in * sizeof(float), cudaMemcpyHostToDevice, g_stream);
            kernels::native_quantize_q8_1(d_x, d_q8, (int) n_in, 1, g_stream);
            g_last_x = xs[i];
            g_last_n = n_in;
        }
        kernels::native_mmvq(mats[i]->type, wdev, d_q8, d_out + off, (int) n_in, (int) mats[i]->n_out, 1, g_stream);
        off += mats[i]->n_out;
    }
    cudaStreamSynchronize(g_stream);
    off = 0;
    for (int i = 0; i < count; ++i) {
        cudaMemcpyAsync(ys[i], d_out + off, (size_t) mats[i]->n_out * sizeof(float), cudaMemcpyDeviceToHost, g_stream);
        off += mats[i]->n_out;
    }
    cudaStreamSynchronize(g_stream);
    return count;
}

}  // namespace

bool qwen35_gpu_init(std::string& err) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) { err = "no GPU visible"; return false; }
    if (g_stream) return true;
    if (cudaStreamCreate(&g_stream) != cudaSuccess) { err = "cudaStreamCreate failed"; return false; }
    return true;
}

bool qwen35_gpu_upload(TrunkWeights& w, std::string& err) {
    if (!g_stream) { err = "qwen35_gpu_init was not called"; return false; }
    auto up = [&](Mat& m) -> bool {
        if (!m.data || m.dev) return true;
        if (!kernels::native_mmvq_supported(m.type)) return true;   // leave it on the CPU path
        const size_t bytes = m.row_bytes * (size_t) m.n_out;
        void* d = nullptr;
        if (cudaMalloc(&d, bytes) != cudaSuccess) { err = "cudaMalloc failed (" + std::to_string(bytes) + " B)"; return false; }
        if (cudaMemcpy(d, m.data, bytes, cudaMemcpyHostToDevice) != cudaSuccess) { err = "cudaMemcpy failed"; return false; }
        m.dev = d;
        g_dense += bytes;
        return true;
    };
    for (GdnLayerWeights& d : w.gdn) {
        if (!up(d.wqkv) || !up(d.wgate) || !up(d.ssm_beta) || !up(d.ssm_alpha) || !up(d.ssm_out)) return false;
    }
    for (AttnLayerWeights& d : w.attn) {
        if (!up(d.wq) || !up(d.wk) || !up(d.wv) || !up(d.wo)) return false;
    }
    for (MoeLayerWeights& d : w.moe) {
        if (!up(d.gate_shexp) || !up(d.up_shexp) || !up(d.down_shexp)) return false;
    }
    if (!up(w.output)) return false;
    g_gpu_batch = gpu_batch;
    g_gpu_matvec = gpu_matvec;
    return true;
}

uint64_t qwen35_gpu_dense_bytes() { return g_dense; }

}  // namespace strata::qwen35
