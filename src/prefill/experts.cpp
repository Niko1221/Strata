// src/prefill/experts.cpp - see include/strata/prefill/experts.hpp.
#include "strata/prefill/experts.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>

namespace strata::prefill {
namespace {

constexpr int64_t N = 2560, FF2 = 1280, FF = 640, K = 10;
// Batches of consecutive ids, at most G experts and ROWS rows (an expert with more rows is a batch of its own, in
// pieces of ROWS).  Per-expert launches made the host, not the GPU, the limit (~10 calls for each of ~500 experts a
// layer).
constexpr int G = 16;
constexpr int64_t ROWS = 4096;
constexpr int SLOTS = 2;                 // batches in staging: the copies of one overlap the work on the other
constexpr size_t GEMM_WS = 32u << 20;    // cuBLAS workspace
constexpr int NPTR = 7;                  // per-expert device pointers: blob, xs, gate/up, gu, h, down, d
constexpr int PRE_GROUP = 16;            // prefetched blobs per copy event
constexpr size_t MMQ_TAIL = 4096;        // zeroed after a batch's gathered experts: MMQ reads past the last one

struct OnDevice {   // a remote runner's GPU current for one call, the main one again afterwards
    int main;
    bool set;
    OnDevice(int dev, int main_dev, bool remote) : main(main_dev), set(remote) { if (set) cudaSetDevice(dev); }
    ~OnDevice() { if (set) cudaSetDevice(main); }
};

// Bump allocation from a region; without one it only counts (bytes_needed).
struct Alloc {
    uint8_t* base = nullptr;
    uint64_t cap = 0, used = 0;
    template <typename T> T* take(size_t n, bool& ok) {
        const uint64_t bytes = ((uint64_t) n * sizeof(T) + 255) & ~255ull;
        T* p = base != nullptr ? (T*) (base + used) : nullptr;
        used += bytes;
        if (base != nullptr && used > cap) ok = false;
        return p;
    }
};

size_t blob_cap() { return ((size_t) strata::kernels::cpu::expert_layout().max_blob + 255) & ~(size_t) 255; }

// the pinned upload block: NPTR x n_expert pointers, then the adds (at most 3 per row, 2 per expert) with each
// batch's row bounds (2 per expert), rows, weights
size_t adds_cap(int64_t chunk, int64_t n_expert) { return (size_t) (chunk * K * 3 + n_expert * 4); }

// a native layer whose expert types llama.cpp's MMQ covers takes it (STRATA_PREFILL_MMQ=0: FP16 for every layer)
bool mmq_layer(const strata::kernels::cpu::NativeFmt& f) {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_PREFILL_MMQ");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return on && strata::kernels::cpu::expert_layout().native && mmq::supported(f.gu_type) && mmq::supported(f.d_type);
}
bool mmq_any() {
    const auto& lay = strata::kernels::cpu::expert_layout();
    for (const auto& f : lay.fmt) if (mmq_layer(f)) return true;
    return false;
}
size_t up_bytes(int64_t chunk, int64_t n_expert) {
    return (size_t) NPTR * (size_t) n_expert * sizeof(void*) + (adds_cap(chunk, n_expert) + (size_t) chunk * K * 2) * 4;
}

}  // namespace

struct ExpertRunner::Impl {
    int dev = -1, main = 0;
    bool remote = false, streams = false;
    core::ExpertSource* src = nullptr;
    const core::ExpertCache* cache = nullptr;
    const int32_t* res = nullptr;
    int64_t n_expert = 0, max_chunk = 0, T = 0;   // T: the chunk the buffers are bound for
    cudaStream_t s = nullptr, copy = nullptr;
    cudaEvent_t copied[SLOTS] = {}, used[SLOTS] = {};
    std::vector<cudaEvent_t> piece_ev;   // remote: a piece of the sums is in host_sum
    cudaEvent_t t0 = nullptr, t1 = nullptr;   // the previous layer's GPU time (remote), read at the next one
    bool timed = false, live[SLOTS] = {};
    Gemm gemm;
    std::unique_ptr<mmq::Context> mmq_ctx;   // the MMQ layers' launches (null: none)
    void* owned = nullptr;   // the buffers' own allocation (no region)
    uint16_t *input = nullptr, *xs = nullptr, *hh = nullptr, *dq_gu = nullptr, *dq_d = nullptr;
    float *sum = nullptr, *w = nullptr, *gu = nullptr, *d = nullptr;
    int32_t* rows_src = nullptr;
    int32_t* adds = nullptr;   // per batch: its tokens, their row lists' starts, the row lists (moe_gather_add), bounds
    uint8_t* xq = nullptr;     // MMQ: a batch's rows as q8_1 (gate/up's input, then down's)
    int32_t* ident = nullptr;  // MMQ: the identity row map
    std::vector<int32_t> mark, first, count;
    uint8_t* stage[SLOTS] = {};
    void** ptrs = nullptr;                    // NPTR arrays of n_expert device pointers
    // a layer's uploads, staged in pinned memory (a pageable copy blocks the host behind queued expert copies):
    // [pointers | adds | rows | weights], for chunks of up to max_chunk tokens
    uint8_t* h_up = nullptr;
    cudaEvent_t up_done = nullptr;   // the previous layer's uploads have been read
    bool up_live = false;
    uint16_t *h_input = nullptr, *h_sum = nullptr;
    // a layer's blobs copied ahead: at pre_off[e] in `pre` (-1: not there), covered by event pre_ev[pre_grp[e]]
    uint8_t* pre = nullptr;
    uint64_t pre_cap = 0;
    int64_t pre_layer = -1;
    std::vector<int64_t> pre_off;
    std::vector<int32_t> pre_grp;
    std::vector<cudaEvent_t> pre_ev;
    cudaEvent_t pre_used = nullptr;   // the last read of the area
    bool pre_live = false;

    // the same sequence counted (bytes_needed) or carved
    static void carve(Alloc& a, int64_t T, int64_t n_expert, bool remote, bool streams, Impl* m, void** ws, bool& ok) {
        const size_t t = (size_t) T;
        uint16_t* input = remote ? a.take<uint16_t>(t * N, ok) : nullptr;
        float* sum = remote ? a.take<float>(t * N, ok) : nullptr;
        int32_t* rows_src = a.take<int32_t>(t * K, ok);
        float* w = a.take<float>(t * K, ok);
        int32_t* adds = a.take<int32_t>(adds_cap(T, n_expert), ok);
        uint16_t* xs = a.take<uint16_t>((size_t) ROWS * N, ok);
        float* gu = a.take<float>((size_t) ROWS * FF2, ok);
        uint16_t* hh = a.take<uint16_t>((size_t) ROWS * FF, ok);
        float* d = a.take<float>((size_t) ROWS * N, ok);
        uint16_t* dq_gu = a.take<uint16_t>((size_t) G * FF2 * N, ok);
        uint16_t* dq_d = a.take<uint16_t>((size_t) G * N * FF, ok);
        uint8_t* stage[SLOTS] = {};
        if (streams)
            for (auto& p : stage) p = a.take<uint8_t>((size_t) G * blob_cap(), ok);
        void** ptrs = a.take<void*>((size_t) NPTR * (size_t) n_expert, ok);
        *ws = a.take<uint8_t>(GEMM_WS, ok);
        const bool q = mmq_any();
        uint8_t* xq = q ? a.take<uint8_t>(mmq::q8_bytes(ROWS, N), ok) : nullptr;
        int32_t* ident = q ? a.take<int32_t>((size_t) ROWS, ok) : nullptr;
        if (m == nullptr) return;
        m->xq = xq; m->ident = ident;
        m->input = input; m->sum = sum; m->rows_src = rows_src; m->w = w; m->adds = adds;
        m->xs = xs; m->gu = gu; m->hh = hh; m->d = d; m->dq_gu = dq_gu; m->dq_d = dq_d;
        for (int i = 0; i < SLOTS; ++i) m->stage[i] = stage[i];
        m->ptrs = ptrs;
    }
};

ExpertRunner::ExpertRunner() : impl_(new Impl) {}

ExpertRunner::~ExpertRunner() {
    Impl& m = *impl_;
    if (m.dev < 0) return;
    OnDevice on(m.dev, m.main, m.remote);
    if (m.s) cudaStreamSynchronize(m.s);
    if (m.copy) cudaStreamSynchronize(m.copy);
    for (int i = 0; i < SLOTS; ++i) {
        if (m.copied[i]) cudaEventDestroy(m.copied[i]);
        if (m.used[i]) cudaEventDestroy(m.used[i]);
    }
    for (cudaEvent_t e : {m.t0, m.t1, m.pre_used, m.up_done}) if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : m.pre_ev) if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : m.piece_ev) if (e) cudaEventDestroy(e);
    if (m.h_input) cudaFreeHost(m.h_input);
    if (m.h_sum) cudaFreeHost(m.h_sum);
    if (m.h_up) cudaFreeHost(m.h_up);
    if (m.owned) cudaFree(m.owned);
    if (m.copy) cudaStreamDestroy(m.copy);
    if (m.remote && m.s) cudaStreamDestroy(m.s);
}

uint64_t ExpertRunner::bytes_needed(int64_t chunk, int64_t n_expert, bool remote, bool streams) {
    Alloc a;
    bool ok = true;
    void* ws = nullptr;
    Impl::carve(a, chunk, n_expert, remote, streams, nullptr, &ws, ok);
    return a.used;
}

bool ExpertRunner::init(int device, int main_device, void* stream, core::ExpertSource* src, const core::ExpertCache* cache,
                        const int32_t* res, int64_t n_expert, int64_t max_chunk, bool streams, std::string& err) {
    Impl& m = *impl_;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (stream == nullptr && !lay.native) {
        err = "prompt experts on a second GPU: need a native pack";
        return false;
    }
    m.dev = device;
    m.main = main_device;
    m.remote = stream == nullptr;
    m.streams = streams;
    m.src = src;
    m.cache = cache;
    m.res = res;
    m.n_expert = n_expert;
    m.max_chunk = max_chunk;
    m.mark.assign((size_t) max_chunk, -1);
    m.first.resize((size_t) max_chunk);
    m.count.resize((size_t) max_chunk);
    OnDevice on(device, main_device, m.remote);
    bool ok = cudaEventCreateWithFlags(&m.up_done, cudaEventDisableTiming) == cudaSuccess &&
              cudaEventCreate(&m.t0) == cudaSuccess && cudaEventCreate(&m.t1) == cudaSuccess &&
              cudaHostAlloc((void**) &m.h_up, up_bytes(max_chunk, n_expert), cudaHostAllocDefault) == cudaSuccess;
    if (m.remote) ok = ok && cudaStreamCreateWithFlags(&m.s, cudaStreamNonBlocking) == cudaSuccess;
    else m.s = (cudaStream_t) stream;
    if (streams) {
        ok = ok && cudaStreamCreateWithFlags(&m.copy, cudaStreamNonBlocking) == cudaSuccess;
        for (int i = 0; ok && i < SLOTS; ++i)
            ok = cudaEventCreateWithFlags(&m.copied[i], cudaEventDisableTiming) == cudaSuccess &&
                 cudaEventCreateWithFlags(&m.used[i], cudaEventDisableTiming) == cudaSuccess;
        m.pre_ev.assign((size_t) ((n_expert + PRE_GROUP - 1) / PRE_GROUP), nullptr);
        for (auto& e : m.pre_ev) ok = ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
        ok = ok && cudaEventCreateWithFlags(&m.pre_used, cudaEventDisableTiming) == cudaSuccess;
        m.pre_off.assign((size_t) n_expert, -1);
        m.pre_grp.assign((size_t) n_expert, 0);
    }
    if (!ok || !m.gemm.init(m.s, err, true)) {
        if (err.empty()) err = "prompt experts: streams or events";
        return false;
    }
    if (mmq_any()) m.mmq_ctx = std::make_unique<mmq::Context>();
    // the input and the sums: pinned and portable, both GPUs copy them
    if (m.remote && (cudaHostAlloc((void**) &m.h_input, (size_t) max_chunk * N * 2, cudaHostAllocPortable) != cudaSuccess ||
                     cudaHostAlloc((void**) &m.h_sum, (size_t) max_chunk * N * 2, cudaHostAllocPortable) != cudaSuccess)) {
        err = "prompt experts: pinned input and sum buffers";
        return false;
    }
    return true;
}

bool ExpertRunner::bind(void* region, uint64_t bytes, int64_t chunk, std::string& err, void* area, uint64_t area_bytes) {
    Impl& m = *impl_;
    OnDevice on(m.dev, m.main, m.remote);
    m.pre_layer = -1;
    if (region == nullptr) {
        if (m.owned != nullptr) return true;
        chunk = m.max_chunk;
        bytes = bytes_needed(chunk, m.n_expert, m.remote, m.streams);
        if (cudaMalloc(&m.owned, bytes) != cudaSuccess) {
            m.owned = nullptr;
            err = "prompt experts: " + std::to_string(bytes >> 20) + " MiB of device buffers";
            return false;
        }
        region = m.owned;
    }
    if (chunk > m.max_chunk) {
        err = "prompt experts: a chunk longer than the one they were set up for";
        return false;
    }
    Alloc a;
    a.base = (uint8_t*) region;
    a.cap = bytes;
    bool ok = true;
    void* ws = nullptr;
    Impl::carve(a, chunk, m.n_expert, m.remote, m.streams, &m, &ws, ok);
    if (!ok) {
        err = "prompt experts: their buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit";
        return false;
    }
    m.gemm.set_buffers(nullptr, 0, ws, GEMM_WS);
    m.T = chunk;
    if (area == nullptr && region != m.owned && bytes > a.used) {
        area = (uint8_t*) region + a.used;
        area_bytes = bytes - a.used;
    }
    m.pre = m.streams ? (uint8_t*) area : nullptr;
    m.pre_cap = m.pre != nullptr ? area_bytes : 0;
    return true;
}

uint64_t ExpertRunner::bytes_for(int64_t chunk, const int32_t* skip) const {
    const Impl& m = *impl_;
    return bytes_needed(chunk, m.n_expert, m.remote, m.streams) + (chunk >= kPrefetchMin ? prefetch_need(skip) : 0);
}

uint64_t ExpertRunner::prefetch_need(const int32_t* skip) const {
    const Impl& m = *impl_;
    if (!m.streams) return 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    uint64_t most = 0;
    for (int64_t l = 0; l < lay.n_layers; ++l) {
        const size_t stride = ((size_t) lay.blob_bytes(l) + 255) & ~(size_t) 255;
        uint64_t sum = 0;
        for (int64_t e = 0; e < m.n_expert; ++e) {
            const size_t i = (size_t) (l * m.n_expert + e);
            if ((m.res == nullptr || m.res[i] < 0) && (skip == nullptr || skip[i] < 0)) sum += stride;
        }
        most = std::max(most, sum);
    }
    return most + most / 10;   // the slots lent for the prompt make a few more experts stream
}

bool ExpertRunner::prefetch(int64_t layer, const std::vector<int32_t>& experts, std::string& err) {
    Impl& m = *impl_;
    if (m.pre_cap == 0) return true;
    OnDevice on(m.dev, m.main, m.remote);
    const auto& lay = strata::kernels::cpu::expert_layout();
    const size_t bb = (size_t) lay.blob_bytes(layer);
    const size_t stride = (bb + 255) & ~(size_t) 255;
    std::fill(m.pre_off.begin(), m.pre_off.end(), -1);
    m.pre_layer = layer;
    if (m.pre_live) cudaStreamWaitEvent(m.copy, m.pre_used, 0);   // the previous layer has read the area
    uint64_t at = 0;
    int32_t grp = 0, in_grp = 0;
    const size_t n = experts.size();
    for (size_t j = 0; j < n;) {
        const int32_t e = experts[j];
        const bool held = m.res && m.cache && m.res[(size_t) layer * (size_t) m.n_expert + (size_t) e] >= 0;
        if (held) { ++j; continue; }
        if (at + stride > m.pre_cap) break;
        // a run: the next experts' blobs follow this one in the arena, and fit
        const uint8_t* b0 = m.src->blob(layer, e);
        if (b0 == nullptr) { err = "prompt experts: the expert source has no blob"; return false; }
        size_t k = j + 1;
        if (m.src->pinned(layer, e) && stride == bb)
            while (k < n && experts[(size_t) k] == e + (int32_t) (k - j) && at + (k - j + 1) * stride <= m.pre_cap &&
                   !(m.res && m.cache && m.res[(size_t) layer * (size_t) m.n_expert + (size_t) experts[k]] >= 0) &&
                   m.src->pinned(layer, experts[k]) && m.src->blob(layer, experts[k]) == b0 + (k - j) * bb &&
                   in_grp + (int32_t) (k - j) < PRE_GROUP) ++k;
        cudaMemcpyAsync(m.pre + at, b0, (k - j) * bb, cudaMemcpyHostToDevice, m.copy);   // pageable blobs: staged
        for (size_t i = j; i < k; ++i) {
            m.pre_off[(size_t) experts[i]] = (int64_t) (at + (i - j) * stride);
            m.pre_grp[(size_t) experts[i]] = grp;
        }
        at += (k - j) * stride;
        experts_prefetched += (int64_t) (k - j);
        experts_streamed += (int64_t) (k - j);
        in_grp += (int32_t) (k - j);
        j = k;
        if (in_grp >= PRE_GROUP) {
            cudaEventRecord(m.pre_ev[(size_t) grp++], m.copy);
            in_grp = 0;
        }
    }
    if (in_grp > 0) cudaEventRecord(m.pre_ev[(size_t) grp], m.copy);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { err = std::string("prompt experts: prefetch: ") + cudaGetErrorString(e); return false; }
    return true;
}

uint16_t* ExpertRunner::host_input() const { return impl_->h_input; }
uint16_t* ExpertRunner::host_sum() const { return impl_->h_sum; }
cudaEvent_t ExpertRunner::piece_done(int64_t k) const { return impl_->piece_ev[(size_t) k]; }

void ExpertRunner::stage_input(int64_t t0, int64_t rows, cudaEvent_t ready) {
    Impl& m = *impl_;
    OnDevice on(m.dev, m.main, m.remote);
    cudaStreamWaitEvent(m.s, ready, 0);   // behind the previous layer's sums, which pass through `input`
    cudaMemcpyAsync(m.input + t0 * N, m.h_input + t0 * N, (size_t) rows * N * 2, cudaMemcpyHostToDevice, m.s);
}

bool ExpertRunner::run_layer(int64_t layer, int64_t T, const std::vector<int32_t>& experts, const std::vector<int32_t>& off,
                             const std::vector<int32_t>& src, const std::vector<float>& w, int64_t piece,
                             const uint16_t* input, float* sum, std::string& err) {
    Impl& m = *impl_;
    const size_t n_exp = experts.size(), n_rows = src.size();
    if (T > m.T || off.size() != n_exp + 1 || w.size() != n_rows || (size_t) off.back() != n_rows ||
        n_rows > (size_t) T * K || n_exp > (size_t) m.n_expert || (!m.remote && (input == nullptr || sum == nullptr))) {
        err = "prompt experts: a layer's share is out of range";
        return false;
    }
    OnDevice on(m.dev, m.main, m.remote);
    const auto h0 = std::chrono::steady_clock::now();
    if (m.timed && cudaEventQuery(m.t1) == cudaSuccess) {   // the previous layer (the main GPU waited for it)
        float ms = 0;
        cudaEventElapsedTime(&ms, m.t0, m.t1);
        ms_gpu += ms;
    }
    m.timed = false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto& f = lay.fmt.empty() ? strata::kernels::cpu::NativeFmt{} : lay.fmt[(size_t) layer];
    const size_t bb = (size_t) lay.blob_bytes(layer);
    const size_t stride = (bb + 255) & ~(size_t) 255;   // staging: a run of arena blobs lands as one copy when equal
    const bool use_mmq = m.mmq_ctx != nullptr && mmq_layer(f);
    const size_t gub = use_mmq ? mmq::matrix_bytes(f.gu_type, FF2, N) : 0;
    const size_t db = use_mmq ? mmq::matrix_bytes(f.d_type, N, FF) : 0;
    auto held = [&](int32_t e) -> int32_t {
        return m.res && m.cache ? m.res[(size_t) layer * (size_t) m.n_expert + (size_t) e] : -1;
    };
    auto pre = [&](int32_t e) -> int64_t { return m.pre_layer == layer ? m.pre_off[(size_t) e] : -1; };
    // ---- the batches, and every expert's device pointers (one upload)
    std::vector<std::pair<size_t, size_t>> batches;
    for (size_t j0 = 0; j0 < n_exp;) {
        size_t j1 = j0 + 1;
        int64_t rows = off[j0 + 1] - off[j0];
        if (rows <= ROWS)
            while (j1 < n_exp && j1 - j0 < (size_t) G && rows + (off[j1 + 1] - off[j1]) <= ROWS) rows += off[j1 + 1] - off[j1++];
        batches.emplace_back(j0, j1);
        j0 = j1;
    }
    if (m.up_live) cudaEventSynchronize(m.up_done);   // the host block is free again
    const size_t E = (size_t) m.n_expert;   // array stride
    void** P = (void**) m.h_up;
    int32_t* h_adds = (int32_t*) (P + NPTR * E);
    int32_t* h_rows = h_adds + adds_cap(m.max_chunk, m.n_expert);
    float* h_w = (float*) (h_rows + (size_t) m.max_chunk * K);
    std::vector<int> rows_of(n_exp);
    for (size_t b = 0; b < batches.size(); ++b) {
        const auto [j0, j1] = batches[b];
        size_t streamed = 0;
        for (size_t j = j0; j < j1; ++j) {
            const int64_t r = off[j] - off[j0];
            const int32_t slot = held(experts[j]);
            if (slot < 0 && !m.streams) {
                err = "prompt experts: an expert its cache does not hold, and no staging";
                return false;
            }
            P[0 * E + j] = slot >= 0 ? (void*) m.cache->device_slot(slot)
                         : pre(experts[j]) >= 0 ? (void*) (m.pre + pre(experts[j]))
                         : (void*) (m.stage[b % SLOTS] + streamed++ * stride);
            P[1 * E + j] = m.xs + r * N;
            P[2 * E + j] = m.dq_gu + (j - j0) * (size_t) (FF2 * N);
            P[3 * E + j] = m.gu + r * FF2;
            P[4 * E + j] = m.hh + r * FF;
            P[5 * E + j] = m.dq_d + (j - j0) * (size_t) (N * FF);
            P[6 * E + j] = m.d + r * N;
            rows_of[j] = off[j + 1] - off[j];
        }
    }
    // per batch, its tokens and each one's rows in order (a token may be routed to several of the batch's experts):
    // [tokens | starts (n + 1) | rows] at adds_at[b]
    std::vector<size_t> adds_at(batches.size()), bounds_at(batches.size());
    std::vector<int32_t> ntok(batches.size());
    size_t n_adds = 0;
    for (size_t b = 0; b < batches.size(); ++b) {
        const int32_t r0 = off[batches[b].first], r1 = off[batches[b].second];
        std::vector<int32_t> toks;
        for (int32_t r = r0; r < r1; ++r) {
            const int32_t t = src[(size_t) r];
            if (m.mark[(size_t) t] != (int32_t) b) {
                m.mark[(size_t) t] = (int32_t) b;
                m.first[(size_t) t] = (int32_t) toks.size();
                m.count[(size_t) toks.size()] = 0;
                toks.push_back(t);
            }
            ++m.count[(size_t) m.first[(size_t) t]];
        }
        const size_t nt = toks.size(), at = n_adds;
        adds_at[b] = at;
        ntok[b] = (int32_t) nt;
        n_adds = at + nt + (nt + 1) + (size_t) (r1 - r0);
        int32_t* h_tok = h_adds + at;
        int32_t* h_start = h_tok + nt;
        int32_t* h_list = h_start + nt + 1;
        h_start[0] = 0;
        for (size_t i = 0; i < nt; ++i) {
            h_tok[i] = toks[i];
            h_start[i + 1] = h_start[i] + m.count[i];
            m.count[i] = h_start[i];   // the fill cursor
        }
        for (int32_t r = r0; r < r1; ++r) h_list[m.count[(size_t) m.first[(size_t) src[(size_t) r]]]++] = r;
        for (int32_t t : toks) m.mark[(size_t) t] = -1;
        bounds_at[b] = n_adds;   // MMQ: the batch's experts' rows, from the batch's first
        for (size_t j = batches[b].first; j <= batches[b].second; ++j) h_adds[n_adds++] = off[j] - r0;
    }
    if (m.remote) cudaEventRecord(m.t0, m.s);
    const uint16_t* in = m.remote ? m.input : input;
    float* out = m.remote ? m.sum : sum;
    std::copy(src.begin(), src.end(), h_rows);
    std::copy(w.begin(), w.end(), h_w);
    if (n_adds > 0) cudaMemcpyAsync(m.adds, h_adds, n_adds * 4, cudaMemcpyHostToDevice, m.s);
    cudaMemcpyAsync(m.ptrs, P, NPTR * E * sizeof(void*), cudaMemcpyHostToDevice, m.s);
    if (n_rows > 0) {
        cudaMemcpyAsync(m.rows_src, h_rows, n_rows * 4, cudaMemcpyHostToDevice, m.s);
        cudaMemcpyAsync(m.w, h_w, n_rows * 4, cudaMemcpyHostToDevice, m.s);
    }
    if (use_mmq) mmq::iota(m.ident, ROWS, m.s);   // each layer: the local runner's buffers share the dense steps'
    cudaEventRecord(m.up_done, m.s);
    m.up_live = true;
    if (m.remote) cudaMemsetAsync(m.sum, 0, (size_t) T * N * 4, m.s);
    // ---- staging: the streamed experts of batch b into slot b % SLOTS, a copy per run of adjacent arena blobs
    auto stage_batch = [&](size_t b) {
        const auto [j0, j1] = batches[b];
        const int sl = (int) (b % SLOTS);
        bool any = false;
        for (size_t j = j0; j < j1;) {
            const int32_t e = experts[j];
            if (held(e) >= 0 || pre(e) >= 0) { ++j; continue; }
            if (!any && m.live[sl]) cudaStreamWaitEvent(m.copy, m.used[sl], 0);   // its previous batch is dequantized
            any = true;
            const uint8_t* b0 = m.src->blob(layer, e);
            uint8_t* dst = (uint8_t*) P[0 * E + j];
            size_t k = j + 1;   // the run: the next experts' blobs follow this one in the arena and in staging
            if (m.src->pinned(layer, e) && stride == bb)
                while (k < j1 && held(experts[k]) < 0 && pre(experts[k]) < 0 && m.src->pinned(layer, experts[k]) &&
                       m.src->blob(layer, experts[k]) == b0 + (k - j) * bb &&
                       (uint8_t*) P[0 * E + k] == dst + (k - j) * bb) ++k;
            cudaMemcpyAsync(dst, b0, (k - j) * bb, cudaMemcpyHostToDevice, m.copy);   // pageable blobs: staged
            experts_streamed += (int64_t) (k - j);
            j = k;
        }
        if (any) {
            cudaEventRecord(m.copied[sl], m.copy);
            m.live[sl] = true;
        }
        return any;
    };
    std::vector<char> has_stream(batches.size(), 0);
    if (!batches.empty()) has_stream[0] = stage_batch(0);
    auto** dp = (const uint16_t**) m.ptrs;   // device arrays by index
    for (size_t b = 0; b < batches.size(); ++b) {
        if (b + 1 < batches.size()) has_stream[b + 1] = stage_batch(b + 1);   // the next batch copies meanwhile
        const auto [j0, j1] = batches[b];
        const int n = (int) (j1 - j0);
        const int sl = (int) (b % SLOTS);
        if (has_stream[b]) cudaStreamWaitEvent(m.s, m.copied[sl], 0);
        int32_t last_grp = -1;   // the batch's prefetched blobs have landed with their last group
        for (size_t j = j0; j < j1; ++j)
            if (held(experts[j]) < 0 && pre(experts[j]) >= 0) last_grp = std::max(last_grp, m.pre_grp[(size_t) experts[j]]);
        if (last_grp >= 0) cudaStreamWaitEvent(m.s, m.pre_ev[(size_t) last_grp], 0);
        const int64_t r0 = off[j0], rows = off[j1] - r0;
        const bool q = use_mmq && rows <= ROWS;   // an expert with more rows is multiplied in pieces, in FP16
        if (q) {
            // the batch's GGUF blocks side by side in the dequantization buffers' place (MMQ reads the experts at
            // one stride), a zeroed tail after them
            for (size_t j = j0; j < j1; ++j) {
                const uint8_t* bl = (const uint8_t*) P[0 * E + j];
                mmq::gather_native(bl, bl + f.up_off, gub / 2, bl + f.down_off, db, (uint8_t*) m.dq_gu + (j - j0) * gub,
                                   (uint8_t*) m.dq_d + (j - j0) * db, m.s);
            }
            cudaMemsetAsync((uint8_t*) m.dq_gu + (size_t) n * gub, 0, MMQ_TAIL, m.s);
            cudaMemsetAsync((uint8_t*) m.dq_d + (size_t) n * db, 0, MMQ_TAIL, m.s);
        } else if (lay.native)
            strata::kernels::iq_dequant_experts_f16(f.gu_type, f.d_type, (const uint8_t* const*) (m.ptrs + 0 * E + j0), n,
                                                    f.up_off, f.down_off, f.n_ff, f.n_embd, m.dq_gu, m.dq_d, m.s);
        else
            for (size_t j = j0; j < j1; ++j)
                blob_dequant_f16((const uint8_t*) P[0 * E + j], (uint16_t*) P[2 * E + j], (uint16_t*) P[5 * E + j], m.s);
        if (has_stream[b]) cudaEventRecord(m.used[sl], m.s);
        if (last_grp >= 0) {
            cudaEventRecord(m.pre_used, m.s);
            m.pre_live = true;
        }
        for (size_t j = j0; j < j1; ++j) if (held(experts[j]) >= 0) ++experts_resident;
        if (rows > ROWS) {   // one expert: its rows in pieces
            for (int64_t p = 0; p < rows; p += ROWS) {
                const int64_t nr = std::min<int64_t>(ROWS, rows - p);
                gather_rows16(in, m.rows_src + r0 + p, m.xs, nr, N, m.s);
                m.gemm.f16(m.xs, m.dq_gu, m.gu, nr, FF2, N);
                swiglu_interleaved(m.gu, m.hh, nr, m.s);
                m.gemm.f16(m.hh, m.dq_d, m.d, nr, N, FF);
                moe_scatter_add(out, m.d, m.w + r0 + p, m.rows_src + r0 + p, nr, m.s);
            }
            continue;
        }
        if (q) {
            // the rows in FP32 in down's output place, as q8_1 for the gate/up product; h in the FP16 rows' place
            gather_rows16_f32(in, m.rows_src + r0, m.d, rows, N, m.s);
            mmq::quantize(m.d, nullptr, m.xq, f.gu_type, N, N, rows, m.s);
            int64_t maxr = 0;
            for (size_t j = j0; j < j1; ++j) maxr = std::max<int64_t>(maxr, off[j + 1] - off[j]);
            mmq::Product p;
            p.w = m.dq_gu; p.type = f.gu_type; p.w_rows = FF2; p.w_cols = N; p.expert_bytes = gub; p.n = n;
            p.xq = m.xq; p.bounds = m.adds + bounds_at[b]; p.ids = m.ident; p.total_rows = rows; p.max_rows = maxr;
            p.dst = m.gu; p.ld_dst = FF2;
            m.mmq_ctx->run(p, m.s);
            float* h = (float*) m.xs;
            mmq::swiglu(m.gu, h, rows, FF, false, m.s);
            mmq::quantize(h, nullptr, m.xq, f.d_type, FF, FF, rows, m.s);
            p.w = m.dq_d; p.type = f.d_type; p.w_rows = N; p.w_cols = FF; p.expert_bytes = db;
            p.dst = m.d; p.ld_dst = N;
            m.mmq_ctx->run(p, m.s);
            const int32_t* a = m.adds + adds_at[b];
            moe_gather_add(out, m.d, r0, m.w, a, a + ntok[b], a + 2 * ntok[b] + 1, ntok[b], m.s);
            continue;
        }
        gather_rows16(in, m.rows_src + r0, m.xs, rows, N, m.s);
        m.gemm.f16_grouped(dp + 1 * E + j0, dp + 2 * E + j0, (float* const*) (m.ptrs + 3 * E + j0),
                           (const uint16_t* const*) (P + 1 * E + j0), (const uint16_t* const*) (P + 2 * E + j0),
                           (float* const*) (P + 3 * E + j0), rows_of.data() + j0, n, FF2, N);
        swiglu_interleaved(m.gu, m.hh, rows, m.s);
        m.gemm.f16_grouped(dp + 4 * E + j0, dp + 5 * E + j0, (float* const*) (m.ptrs + 6 * E + j0),
                           (const uint16_t* const*) (P + 4 * E + j0), (const uint16_t* const*) (P + 5 * E + j0),
                           (float* const*) (P + 6 * E + j0), rows_of.data() + j0, n, N, FF);
        const int32_t* a = m.adds + adds_at[b];
        moe_gather_add(out, m.d, r0, m.w, a, a + ntok[b], a + 2 * ntok[b] + 1, ntok[b], m.s);
    }
    if (m.remote) {   // the sums as FP16 in the input's place (half the bytes over the first GPU's x4 link), in pieces
        const int64_t pc = piece > 0 ? piece : T;
        for (int64_t t0 = 0, k = 0; t0 < T; t0 += pc, ++k) {
            const int64_t n = std::min(pc, T - t0);
            if ((size_t) k == m.piece_ev.size()) {
                m.piece_ev.push_back(nullptr);
                cudaEventCreateWithFlags(&m.piece_ev.back(), cudaEventDisableTiming);
            }
            sums_to_f16(m.sum + t0 * N, m.input + t0 * N, n * N, m.s);
            cudaMemcpyAsync(m.h_sum + t0 * N, m.input + t0 * N, (size_t) n * N * 2, cudaMemcpyDeviceToHost, m.s);
            cudaEventRecord(m.piece_ev[(size_t) k], m.s);
        }
        cudaEventRecord(m.t1, m.s);
        m.timed = true;
    }
    ms_host += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { err = std::string("prompt experts: ") + cudaGetErrorString(e); return false; }
    return true;
}

}  // namespace strata::prefill
