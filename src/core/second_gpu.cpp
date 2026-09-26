// src/core/second_gpu.cpp - see include/strata/core/second_gpu.hpp.
#include "strata/core/second_gpu.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <immintrin.h>

#include <chrono>
#include <cstring>
#include <exception>

namespace strata::core {
namespace {

// the plan block: [n_groups, pad x3] [starts: cap+1] [dst: cap] [tok: cap] (8-byte aligned) [slot addresses: cap]
int64_t ptr_off(int64_t cap) { return (4 + (cap + 1) + 2 * cap + 1) / 2 * 2; }   // in int32 units
size_t plan_bytes(int64_t cap) { return (size_t) ptr_off(cap) * 4 + (size_t) cap * 8; }

// the grouped kernels' grids are sized for the bucket a layer's group count falls in
int bucket(int groups) {
    for (int b : {1, 2, 3, 4, 6, 8, 12, 16, 24, 32}) if (groups <= b) return b;
    return groups <= 48 ? 48 : groups;
}

struct DeviceScope {   // the second GPU current for one call, the main one again afterwards
    int main;
    DeviceScope(int dev, int main_dev) : main(main_dev) { cudaSetDevice(dev); }
    ~DeviceScope() { cudaSetDevice(main); }
};

}  // namespace

SecondGpu::~SecondGpu() {
    if (dev_ < 0) return;
    DeviceScope scope(dev_, main_);
    if (s_) cudaStreamSynchronize(s_);
    if (pre_s_) cudaStreamSynchronize(pre_s_);
    for (auto& gr : graphs_) cudaGraphExecDestroy(gr.exec);
    cache_.close();
    void* dev[] = {d_xq_, d_plan_, d_scratch_, d_rows_, d_pre_};
    for (void* p : dev) if (p) cudaFree(p);
    void* host[] = {h_x_, h_out_, h_plan_};
    for (void* p : host) if (p) cudaFreeHost(p);
    if (ev_) cudaEventDestroy(ev_);
    if (pre_ev_) cudaEventDestroy(pre_ev_);
    if (s_) cudaStreamDestroy(s_);
    if (pre_s_) cudaStreamDestroy(pre_s_);
}

bool SecondGpu::init(int device, int main_device, int64_t n_embd, int64_t n_ff, int max_t, int64_t k, std::string& err) {
    dev_ = device;
    main_ = main_device;
    max_t_ = max_t;
    n_embd_ = n_embd;
    n_ff_ = n_ff;
    cap_ = (int64_t) max_t * k;
    DeviceScope scope(dev_, main_);
    const size_t xb = (size_t) max_t * (size_t) n_embd * sizeof(float), ob = (size_t) cap_ * (size_t) n_embd * sizeof(float);
    const unsigned pm = cudaHostAllocPortable | cudaHostAllocMapped;
    const char* step = nullptr;
    cudaError_t e = cudaSuccess;
    auto run = [&](const char* what, cudaError_t r) { if (e == cudaSuccess && r != cudaSuccess) { e = r; step = what; } };
    run("context", cudaFree(nullptr));
    run("stream", cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking));
    run("event", cudaEventCreateWithFlags(&ev_, cudaEventDisableTiming));
    run("pinned staging", cudaHostAlloc((void**) &h_x_, xb, pm));
    run("pinned staging", cudaHostAlloc((void**) &h_out_, ob, pm));
    run("pinned staging", cudaHostAlloc((void**) &h_plan_, plan_bytes(cap_), pm));
    run("mapped staging", cudaHostGetDevicePointer((void**) &m_x_, h_x_, 0));
    run("mapped staging", cudaHostGetDevicePointer((void**) &m_out_, h_out_, 0));
    run("mapped staging", cudaHostGetDevicePointer((void**) &m_plan_, h_plan_, 0));
    run("device buffers", cudaMalloc((void**) &d_xq_, (size_t) max_t * (size_t) (n_embd / 32) * 36));
    run("device buffers", cudaMalloc((void**) &d_plan_, plan_bytes(cap_)));
    run("device buffers", cudaMalloc((void**) &d_scratch_, strata::kernels::native_expert_scratch_bytes(cap_, n_ff)));
    run("device buffers", cudaMalloc((void**) &d_rows_, ob));
    if (e != cudaSuccess) {
        err = std::string("second GPU: ") + step + ": " + cudaGetErrorString(e);
        return false;
    }
    std::memset(h_x_, 0, xb);
    std::memset(h_plan_, 0, plan_bytes(cap_));
    pending_.reserve((size_t) cap_);
    return true;
}

bool SecondGpu::init_prefetch(int slots, uint64_t blob_bytes, std::string& err) {
    slots = (std::min)(slots, kPrefetchMax);
    if (slots <= 0) return true;
    DeviceScope scope(dev_, main_);
    pre_cap_ = (blob_bytes + 255) / 256 * 256;
    if (cudaStreamCreateWithFlags(&pre_s_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&pre_ev_, cudaEventDisableTiming) != cudaSuccess ||
        cudaMalloc((void**) &d_pre_, (size_t) slots * pre_cap_) != cudaSuccess) {
        err = std::string("second GPU: prefetch slots: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    pre_max_ = slots;
    return true;
}

bool SecondGpu::prefetch(int64_t layer, const int32_t* ids, const uint8_t* const* src, int n, uint64_t bytes,
                         std::string& err) {
    if (n <= 0) return true;
    if (n > pre_max_ || bytes > pre_cap_) { err = "second GPU: a prefetch does not fit its slots"; return false; }
    DeviceScope scope(dev_, main_);
    bool ok = cudaStreamWaitEvent(pre_s_, ev_, 0) == cudaSuccess;   // the last submitted layer may read the slots
    for (int i = 0; ok && i < n; ++i)
        ok = cudaMemcpyAsync(d_pre_ + (size_t) i * pre_cap_, src[i], (size_t) bytes, cudaMemcpyHostToDevice, pre_s_) ==
             cudaSuccess;
    if (!ok || cudaEventRecord(pre_ev_, pre_s_) != cudaSuccess) {
        err = std::string("second GPU: prefetch: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    (void) cudaStreamQuery(pre_s_);   // submit now, not at the next driver call
    pre_layer_ = layer;
    pre_n_ = n;
    for (int i = 0; i < n; ++i) pre_ids_[i] = ids[i];
    prefetch_copied += n;
    return true;
}

bool SecondGpu::graph_for(int gu_type, int d_type, int groups, cudaGraphExec_t& exec, std::string& err) {
    for (const auto& gr : graphs_)
        if (gr.gu == gu_type && gr.d == d_type && gr.groups == groups) { exec = gr.exec; return true; }
    const strata::kernels::NativeExpertLayout L = strata::kernels::native_expert_layout(gu_type, d_type, n_embd_, n_ff_);
    const auto* pi = (const int32_t*) d_plan_;
    cudaGraph_t graph = nullptr;
    if (cudaStreamBeginCapture(s_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "second GPU: cannot capture";
        return false;
    }
    try {
        // a kernel copies the plan (a copy node costs ~15 us on this link); the grouped kernels read the group
        // count from it and skip the groups past it
        strata::kernels::copy_i32_from_mapped((int32_t*) d_plan_, (const int32_t*) m_plan_, (int64_t) plan_bytes(cap_) / 4, s_);
        strata::kernels::quantize_q8_1_rows(m_x_, max_t_, n_embd_, d_xq_, s_);
        strata::kernels::native_expert_grouped(L, (const unsigned long long*) (pi + ptr_off(cap_)), pi + 4, pi,
                                               pi + 4 + cap_ + 1, pi + 4 + 2 * cap_ + 1, groups, cap_, d_xq_, d_scratch_,
                                               d_rows_, s_);
        strata::kernels::native_expert_rows_out(d_rows_, pi, pi + 4, n_embd_, cap_, m_out_, s_);
    } catch (const std::exception& ex) {
        cudaStreamEndCapture(s_, &graph);
        if (graph) cudaGraphDestroy(graph);
        err = std::string("second GPU: ") + ex.what();
        return false;
    }
    const cudaError_t ce = cudaStreamEndCapture(s_, &graph);
    if (ce != cudaSuccess || cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("second GPU: graph capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    cudaGraphUpload(exec, s_);
    cudaStreamSynchronize(s_);
    graphs_.push_back({gu_type, d_type, groups, exec});
    return true;
}

bool SecondGpu::submit(int64_t layer, const float* x, int n_tok, int64_t k, const int32_t* slots, const int32_t* starts,
                       const int32_t* entries, int n_groups, std::string& err) {
    const int n = starts[n_groups];
    if (n_groups <= 0 || n > cap_ || n_tok > max_t_) { err = "second GPU: a layer's share is out of range"; return false; }
    // the plan: rows 0..n-1 in entry order
    auto* pi = (int32_t*) h_plan_;
    int32_t* start = pi + 4;
    int32_t* dst = start + cap_ + 1;
    int32_t* tok = dst + cap_;
    auto* ptr = (unsigned long long*) (pi + ptr_off(cap_));
    pi[0] = n_groups;
    pending_.assign(entries, entries + n);
    for (int g = 0; g <= n_groups; ++g) start[g] = starts[g];
    for (int j = 0; j < n; ++j) {
        dst[j] = j;
        tok[j] = (int32_t) (entries[j] / k);
    }
    bool pre = false;
    for (int g = 0; g < n_groups; ++g) {
        if (slots[g] <= -2) {   // a prefetch slot
            ptr[g] = (unsigned long long) (d_pre_ + (size_t) (-2 - slots[g]) * pre_cap_);
            pre = true;
            ++prefetch_used;
        } else {
            ptr[g] = (unsigned long long) cache_.device_slot(slots[g]);
        }
    }
    std::memcpy(h_x_, x, (size_t) n_tok * (size_t) n_embd_ * sizeof(float));
    const auto& f = strata::kernels::cpu::expert_layout().fmt[(size_t) layer];
    DeviceScope scope(dev_, main_);
    cudaGraphExec_t exec = nullptr;
    if (!graph_for(f.gu_type, f.d_type, bucket(n_groups), exec, err)) return false;
    if ((pre && cudaStreamWaitEvent(s_, pre_ev_, 0) != cudaSuccess) || cudaGraphLaunch(exec, s_) != cudaSuccess ||
        cudaEventRecord(ev_, s_) != cudaSuccess) {
        err = std::string("second GPU: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    ++layers;
    experts += n_groups;
    entries_done += n;
    return true;
}

bool SecondGpu::finish(float* out, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    cudaError_t q;
    while ((q = cudaEventQuery(ev_)) == cudaErrorNotReady) _mm_pause();
    ms_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (q != cudaSuccess) {
        err = std::string("second GPU: ") + cudaGetErrorString(q);
        return false;
    }
    for (size_t j = 0; j < pending_.size(); ++j)
        std::memcpy(out + (size_t) pending_[j] * (size_t) n_embd_, h_out_ + j * (size_t) n_embd_,
                    (size_t) n_embd_ * sizeof(float));
    pending_.clear();
    return true;
}

}  // namespace strata::core
