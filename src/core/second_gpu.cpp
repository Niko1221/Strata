// src/core/second_gpu.cpp - see include/strata/core/second_gpu.hpp.
#include "strata/core/second_gpu.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <immintrin.h>

#include <cstring>
#include <exception>

namespace strata::core {
namespace {

// the plan block: [groups of part 0, of part 1, ring, pad] [starts, part 0: cap+1] [part 1: cap+1] [dst: cap]
// [tok: cap] (8-byte aligned) [slot addresses, part 0: cap] [part 1: cap] [out] [flag after part 0] [after part 1]
struct PlanLayout {
    int64_t starts[2], dst, tok, ptr;   // int32 offsets; `ptr` even
    size_t bytes;
    explicit PlanLayout(int64_t cap)
        : starts{4, 4 + cap + 1}, dst(4 + 2 * (cap + 1)), tok(dst + cap), ptr((tok + cap + 1) & ~1ll),
          bytes((size_t) ptr * 4 + (size_t) (2 * cap + 3) * 8) {}
};

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
    void* dev[] = {d_xq_, d_plan_, d_scratch_, d_rows_, d_count_, d_pre_};
    for (void* p : dev) if (p) cudaFree(p);
    void* host[] = {h_x_, h_plan_};
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
    const size_t xb = (size_t) max_t * (size_t) (n_embd / 32) * 36;   // q8_1 rows
    const size_t ob = (size_t) cap_ * (size_t) n_embd * sizeof(float);
    const size_t pb = PlanLayout(cap_).bytes;
    const unsigned pm = cudaHostAllocPortable | cudaHostAllocMapped;
    const char* step = nullptr;
    cudaError_t e = cudaSuccess;
    int unified = 0;
    auto run = [&](const char* what, cudaError_t r) { if (e == cudaSuccess && r != cudaSuccess) { e = r; step = what; } };
    run("context", cudaFree(nullptr));
    run("unified addressing", cudaDeviceGetAttribute(&unified, cudaDevAttrUnifiedAddressing, dev_));
    run("stream", cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking));
    run("event", cudaEventCreateWithFlags(&ev_, cudaEventDisableTiming));
    run("pinned staging", cudaHostAlloc((void**) &h_x_, xb, pm));
    run("pinned staging", cudaHostAlloc((void**) &h_plan_, pb, pm));
    run("mapped staging", cudaHostGetDevicePointer((void**) &m_x_, h_x_, 0));
    run("mapped staging", cudaHostGetDevicePointer((void**) &m_plan_, h_plan_, 0));
    run("device buffers", cudaMalloc((void**) &d_xq_, xb));
    run("device buffers", cudaMalloc((void**) &d_plan_, pb));
    run("device buffers", cudaMalloc((void**) &d_scratch_, strata::kernels::native_expert_scratch_bytes(cap_, n_ff)));
    run("device buffers", cudaMalloc((void**) &d_rows_, ob));
    run("device buffers", cudaMalloc((void**) &d_count_, sizeof(unsigned)));
    run("device buffers", cudaMemset(d_count_, 0, sizeof(unsigned)));
    if (e != cudaSuccess) {
        err = std::string("second GPU: ") + step + ": " + cudaGetErrorString(e);
        return false;
    }
    if (!unified) {   // it writes its rows through the host pointers of the first GPU's mapped memory
        err = "second GPU: no unified addressing";
        return false;
    }
    std::memset(h_x_, 0, xb);
    std::memset(h_plan_, 0, pb);
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
    void* dst[kPrefetchMax];
    size_t size[kPrefetchMax];
    for (int i = 0; i < n; ++i) {
        dst[i] = d_pre_ + (size_t) i * pre_cap_;
        size[i] = (size_t) bytes;
    }
    if (cudaStreamWaitEvent(pre_s_, ev_, 0) != cudaSuccess ||   // the last submitted layer may read the slots
        !copy_blobs(dst, (const void* const*) src, size, (size_t) n, pre_s_) ||
        cudaEventRecord(pre_ev_, pre_s_) != cudaSuccess) {
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

bool SecondGpu::graph_for(int gu_type, int d_type, int groups, int part, bool prep, cudaGraphExec_t& exec,
                          std::string& err) {
    for (const auto& gr : graphs_)
        if (gr.gu == gu_type && gr.d == d_type && gr.groups == groups && gr.part == part && gr.prep == prep) {
            exec = gr.exec;
            return true;
        }
    const strata::kernels::NativeExpertLayout L = strata::kernels::native_expert_layout(gu_type, d_type, n_embd_, n_ff_);
    const PlanLayout P(cap_);
    const auto* pi = (const int32_t*) d_plan_;
    const auto* p64 = (const unsigned long long*) (pi + P.ptr);
    cudaGraph_t graph = nullptr;
    if (cudaStreamBeginCapture(s_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "second GPU: cannot capture";
        return false;
    }
    try {
        if (prep) {
            // kernels copy the plan and the activations (a copy node costs ~15 us on this link); the grouped kernels
            // read the group count from the plan and skip the groups past it
            strata::kernels::copy_i32_from_mapped((int32_t*) d_plan_, (const int32_t*) m_plan_, (int64_t) P.bytes / 4,
                                                  s_);
            strata::kernels::copy_from_mapped((float*) d_xq_, (const float*) m_x_, max_t_ * (n_embd_ / 32) * 9, s_);
        }
        strata::kernels::native_expert_grouped(L, p64 + part * cap_, pi + P.starts[part], pi + part, pi + P.dst,
                                               pi + P.tok, groups, cap_, d_xq_, d_scratch_, d_rows_, s_);
        strata::kernels::native_expert_rows_out(d_rows_, pi + part, pi + P.starts[part], pi + P.dst, n_embd_, cap_,
                                                (float* const*) (p64 + 2 * cap_),
                                                (uint32_t* const*) (p64 + 2 * cap_ + 1 + part), pi + 2, d_count_, s_);
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
    graphs_.push_back({gu_type, d_type, groups, part, prep, exec});
    return true;
}

bool SecondGpu::submit(int64_t layer, const uint8_t* x, int n_tok, int64_t k, const int32_t* slots,
                       const int32_t* starts, const int32_t* entries, int n_groups, float* out, uint32_t* flag,
                       uint32_t ring, std::string& err) {
    const int n = starts[n_groups];
    if (n_groups <= 0 || n > cap_ || n_tok > max_t_) { err = "second GPU: a layer's share is out of range"; return false; }
    cudaError_t q;
    while ((q = cudaEventQuery(ev_)) == cudaErrorNotReady) _mm_pause();
    if (q != cudaSuccess) {
        err = std::string("second GPU: ") + cudaGetErrorString(q);
        return false;
    }
    // the plan: the experts in its cache first (part 0), then the prefetched ones (part 1), their entries in that order
    const PlanLayout P(cap_);
    auto* pi = (int32_t*) h_plan_;
    auto* p64 = (unsigned long long*) (pi + P.ptr);
    int ng[2] = {0, 0}, e = 0;
    for (int part = 0; part < 2; ++part) {
        int32_t* st = pi + P.starts[part];
        for (int g = 0; g < n_groups; ++g) {
            const bool pre = slots[g] <= -2;
            if (pre != (part == 1)) continue;
            p64[(size_t) part * (size_t) cap_ + (size_t) ng[part]] =
                pre ? (unsigned long long) (d_pre_ + (size_t) (-2 - slots[g]) * pre_cap_)
                    : (unsigned long long) cache_.device_slot(slots[g]);
            st[ng[part]++] = e;
            for (int j = starts[g]; j < starts[g + 1]; ++j, ++e) {
                pi[P.dst + e] = entries[j];
                pi[P.tok + e] = (int32_t) (entries[j] / k);
            }
        }
        st[ng[part]] = e;
        pi[part] = ng[part];
    }
    pi[2] = (int32_t) ring;
    p64[2 * cap_] = (unsigned long long) out;
    p64[2 * cap_ + 1] = ng[1] == 0 ? (unsigned long long) flag : 0;   // the last launch raises it
    p64[2 * cap_ + 2] = ng[1] > 0 ? (unsigned long long) flag : 0;
    std::memcpy(h_x_, x, (size_t) n_tok * (size_t) (n_embd_ / 32) * 36);
    const auto& f = strata::kernels::cpu::expert_layout().fmt[(size_t) layer];
    DeviceScope scope(dev_, main_);
    cudaGraphExec_t own = nullptr, pre = nullptr;
    if ((ng[0] > 0 && !graph_for(f.gu_type, f.d_type, bucket(ng[0]), 0, true, own, err)) ||
        (ng[1] > 0 && !graph_for(f.gu_type, f.d_type, bucket(ng[1]), 1, ng[0] == 0, pre, err)))
        return false;
    if ((own != nullptr && cudaGraphLaunch(own, s_) != cudaSuccess) ||
        (pre != nullptr &&
         (cudaStreamWaitEvent(s_, pre_ev_, 0) != cudaSuccess || cudaGraphLaunch(pre, s_) != cudaSuccess)) ||
        cudaEventRecord(ev_, s_) != cudaSuccess) {
        err = std::string("second GPU: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    ++layers;
    experts += n_groups;
    entries_done += n;
    prefetch_used += ng[1];
    return true;
}

bool SecondGpu::healthy(std::string& err) const {
    const cudaError_t q = cudaEventQuery(ev_);
    if (q == cudaSuccess || q == cudaErrorNotReady) return true;
    err = std::string("second GPU: ") + cudaGetErrorString(q);
    return false;
}

}  // namespace strata::core
