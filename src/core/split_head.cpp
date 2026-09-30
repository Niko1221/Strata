// src/core/split_head.cpp - see include/strata/core/split_head.hpp.
#include "strata/core/split_head.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <immintrin.h>

#include <chrono>
#include <exception>

namespace strata::core {
namespace {

struct DeviceScope {   // this GPU current for one call, the main one again afterwards
    int main;
    DeviceScope(int dev, int main_dev) : main(main_dev) { cudaSetDevice(dev); }
    ~DeviceScope() { cudaSetDevice(main); }
};

}  // namespace

SplitHead::~SplitHead() {
    if (dev_ < 0) return;
    DeviceScope scope(dev_, main_);
    if (s_) cudaStreamSynchronize(s_);
    for (auto& e : exec_)
        for (auto& x : e) if (x) cudaGraphExecDestroy(x);
    void* dev[] = {w_, mixed_, out_, xq_, xil_, scratch_};
    for (void* p : dev) if (p) cudaFree(p);
    void* host[] = {h_in_, h_hand_, h_done_, h_idx_, h_val_, h_logits_};
    for (void* p : host) if (p) cudaFreeHost(p);
    if (s_) cudaStreamDestroy(s_);
}

bool SplitHead::init(int device, int main_device, const std::vector<std::string>& shards, int64_t n_in,
                     int64_t n_vocab, int64_t split, int max_t, std::string& err) {
    if (split <= 0 || split >= n_vocab || max_t < 1 || max_t > strata::kernels::kVerifyMaxT) {
        err = "split head: bad arguments";
        return false;
    }
    dev_ = device;
    main_ = main_device;
    max_t_ = max_t;
    n_in_ = n_in;
    split_ = split;
    rows_ = n_vocab - split;
    DeviceScope scope(dev_, main_);
    try {
        const strata::GgufModel model(shards);
        size_t at = 0;
        const strata::TensorInfo* t = model.find("output.weight", &at);
        if (!t || !strata::kernels::native_mmvq_supported((int) t->type)) {
            err = "split head: output.weight is missing or not a native format";
            return false;
        }
        type_ = (int) t->type;
        const uint64_t row = strata::kernels::native_mmvq_weight_bytes(type_, (int) n_in, 1);
        bytes_ = row * (uint64_t) rows_;
        const unsigned pm = cudaHostAllocPortable | cudaHostAllocMapped;
        const size_t T = (size_t) max_t;
        cudaError_t e = cudaSuccess;
        const char* step = nullptr;
        auto run = [&](const char* what, cudaError_t r) {
            if (e == cudaSuccess && r != cudaSuccess) { e = r; step = what; }
        };
        run("context", cudaFree(nullptr));
        run("stream", cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking));
        run("its rows", cudaMalloc(&w_, bytes_));
        run("its rows", cudaMemcpy(w_, model.shard(at).tensor_data(*t) + (uint64_t) split * row, bytes_,
                                   cudaMemcpyHostToDevice));
        run("buffers", cudaMalloc((void**) &mixed_, T * (size_t) n_in * 4));
        run("buffers", cudaMalloc((void**) &out_, T * (size_t) rows_ * 4));
        run("buffers", cudaMalloc((void**) &xq_, strata::kernels::native_q8_1_bytes((int) n_in, max_t)));
        run("buffers", cudaMalloc((void**) &xil_, strata::kernels::native_q8_1_il_bytes((int) n_in, max_t)));
        run("buffers", cudaMalloc((void**) &scratch_, strata::kernels::argmax_rows_scratch_bytes(max_t)));
        run("buffers", cudaMemset(scratch_, 0, strata::kernels::argmax_rows_scratch_bytes(max_t)));
        run("mapped", cudaHostAlloc((void**) &h_in_, T * (size_t) n_in * 4, pm));
        run("mapped", cudaHostAlloc((void**) &h_hand_, 64, pm));
        run("mapped", cudaHostAlloc((void**) &h_done_, 64, pm));
        run("mapped", cudaHostAlloc((void**) &h_idx_, T * 4 + 64, pm));
        run("mapped", cudaHostAlloc((void**) &h_val_, T * 4 + 64, pm));
        run("mapped", cudaHostAlloc((void**) &h_logits_, T * (size_t) rows_ * 4, pm));
        if (e != cudaSuccess) {
            err = std::string("split head: ") + step + ": " + cudaGetErrorString(e);
            return false;
        }
        *h_hand_ = 0;
        *h_done_ = 0;
    } catch (const std::exception& x) {
        err = std::string("split head: ") + x.what();
        return false;
    }
    return true;
}

// One window's part: the input from mapped memory, quantized as the main GPU's head does, the rows' projection, the
// per-token argmax (and logit) to mapped memory, the count raised.
bool SplitHead::capture(int T, bool logits, std::string& err) {
    using namespace strata::kernels;
    if (exec_[T][logits]) return true;
    if (cudaStreamBeginCapture(s_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "split head: begin capture";
        return false;
    }
    try {
        copy_from_mapped(mixed_, h_in_, (int64_t) T * n_in_, s_);
        if (T >= 2) {
            native_quantize_q8_1_il(mixed_, xq_, xil_, (int) n_in_, T, s_);
            native_mmvq_il(type_, w_, xq_, xil_, out_, (int) n_in_, (int) rows_, T, s_);
        } else {
            native_quantize_q8_1(mixed_, xq_, (int) n_in_, T, s_);
            native_mmvq(type_, w_, xq_, out_, (int) n_in_, (int) rows_, T, s_);
        }
        argmax_rows(out_, T, (int) rows_, scratch_, h_idx_, s_, h_val_);
        if (logits) copy_from_mapped(h_logits_, out_, (int64_t) T * rows_, s_);
        mapped_bump(h_done_, s_);
    } catch (const std::exception& x) {
        cudaGraph_t g = nullptr;
        cudaStreamEndCapture(s_, &g);
        if (g) cudaGraphDestroy(g);
        err = std::string("split head: ") + x.what();
        return false;
    }
    cudaGraph_t g = nullptr;
    if (cudaStreamEndCapture(s_, &g) != cudaSuccess || cudaGraphInstantiate(&exec_[T][logits], g, 0) != cudaSuccess) {
        if (g) cudaGraphDestroy(g);
        err = "split head: capture failed";
        return false;
    }
    cudaGraphDestroy(g);
    cudaGraphUpload(exec_[T][logits], s_);
    cudaStreamSynchronize(s_);
    return true;
}

bool SplitHead::submit(int T, bool logits, std::string& err) {
    if (T < 1 || T > max_t_) { err = "split head: window size out of range"; return false; }
    DeviceScope scope(dev_, main_);
    if (!capture(T, logits, err)) return false;
    if (cudaGraphLaunch(exec_[T][logits], s_) != cudaSuccess) {
        err = std::string("split head: launch: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    (void) cudaStreamQuery(s_);   // submit now
    return true;
}

bool SplitHead::wait(uint32_t n, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    while (*(volatile uint32_t*) h_done_ < n) {
        _mm_pause();
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(5)) {
            DeviceScope scope(dev_, main_);
            err = std::string("split head: its part never finished (") + cudaGetErrorString(cudaStreamQuery(s_)) + ")";
            return false;
        }
    }
    return true;
}

}  // namespace strata::core
