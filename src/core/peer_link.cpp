// src/core/peer_link.cpp - see include/strata/core/peer_link.hpp.
#include "strata/core/peer_link.hpp"

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define STRATA_LINK_PAUSE() _mm_pause()
#else
#define STRATA_LINK_PAUSE() do {} while (0)
#endif

namespace strata::core {

namespace {

constexpr uint64_t kMagic = 0x4b4e494c52455050ull;   // "PEERLINK"
constexpr int32_t kVersion = 1;
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t MAXT = strata::kernels::cpu::MAXT;
constexpr int64_t CAP = MAXT * 10;                    // entries per layer: MAXT tokens x top-10 (as the peer tier)
constexpr int kStage = 16;                            // blobs of no-longer-resident experts per pass

struct alignas(64) Chan {                             // requests FROM role r TO role 1 - r
    alignas(64) std::atomic<uint64_t> seq;            // the requester bumps it once the request is written
    alignas(64) std::atomic<uint64_t> done;           // the owner sets it to `seq` once the rows are written
    int32_t status;                                   // 0 ok, else the owner failed (its log says why)
    int64_t t_post_ns;                                // the requester's steady clock when it posted (timing)
    int32_t layer, n_tok, groups, rows;
    int32_t expert[CAP];
    int32_t start[CAP + 1];
    int32_t tok[CAP];
    float x[MAXT * H];
    float out[CAP * H];
};

struct Meta {                                         // one block, uploaded with one copy per pass
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count[4];
};

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("peer link: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

}  // namespace

struct PeerLinkShm {
    uint64_t magic;
    int32_t version, n_layers, n_expert, h, cap, pad;
    alignas(64) std::atomic<int32_t> serving[2];
    Chan chan[2];
    // then: int32_t res[2][n_layers * n_expert]
};

static size_t shm_bytes(int64_t n_layers, int64_t n_expert) {
    const size_t b = sizeof(PeerLinkShm) + 2 * (size_t) (n_layers * n_expert) * sizeof(int32_t);
    return (b + 4095) / 4096 * 4096;
}

PeerLink::~PeerLink() { close(); }

bool PeerLink::open(const std::string& path, int role, int64_t n_layers, int64_t n_expert, double wait_s,
                    std::string& err) {
#if !defined(__linux__)
    (void) path; (void) role; (void) n_layers; (void) n_expert; (void) wait_s;
    err = "peer link: Linux only";
    return false;
#else
    close();
    if (role != 0 && role != 1) { err = "peer link: --peer-link-role must be 0 or 1"; return false; }
    const size_t bytes = shm_bytes(n_layers, n_expert);
    // whichever engine starts first creates the file (the server removes a stale one before it starts either)
    int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
    const bool creator = fd >= 0;
    if (creator) {
        if (::ftruncate(fd, (off_t) bytes) != 0) {
            err = "peer link: cannot size " + path + ": " + std::strerror(errno);
            ::close(fd);
            return false;
        }
    } else {
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            fd = ::open(path.c_str(), O_RDWR);
            struct stat st {};
            if (fd >= 0 && ::fstat(fd, &st) == 0 && (size_t) st.st_size == bytes) break;
            if (fd >= 0) { ::close(fd); fd = -1; }
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > wait_s) {
                err = "peer link: " + path + " exists but never got its size (another model geometry?)";
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) { err = std::string("peer link: mmap: ") + std::strerror(errno); return false; }
    auto* s = (PeerLinkShm*) p;
    const size_t nres = (size_t) (n_layers * n_expert);
    int32_t* res = (int32_t*) ((uint8_t*) p + sizeof(PeerLinkShm));
    if (creator) {
        s->version = kVersion;
        s->n_layers = (int32_t) n_layers;
        s->n_expert = (int32_t) n_expert;
        s->h = (int32_t) H;
        s->cap = (int32_t) CAP;
        s->serving[0].store(0);
        s->serving[1].store(0);
        for (int r = 0; r < 2; ++r) {
            s->chan[r].seq.store(0);
            s->chan[r].done.store(0);
        }
        for (size_t i = 0; i < 2 * nres; ++i) res[i] = -1;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        __atomic_store_n(&s->magic, kMagic, __ATOMIC_RELEASE);
    } else {
        const auto t0 = std::chrono::steady_clock::now();
        while (__atomic_load_n(&s->magic, __ATOMIC_ACQUIRE) != kMagic) {
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > wait_s) {
                ::munmap(p, bytes);
                err = "peer link: " + path + " never became ready";
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    if (s->version != kVersion || s->n_layers != n_layers || s->n_expert != n_expert || s->h != H || s->cap != CAP) {
        ::munmap(p, bytes);
        err = "peer link: the other engine runs another model geometry or engine version";
        return false;
    }
    shm_ = s;
    map_bytes_ = bytes;
    role_ = role;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    res_mine_ = res + (size_t) role * nres;
    res_partner_ = res + (size_t) (1 - role) * nres;
    seq_ = shm_->chan[role].seq.load();
    last_seq_ = shm_->chan[1 - role].seq.load();
    shm_->chan[1 - role].done.store(last_seq_);
    return true;
#endif
}

void PeerLink::close() {
    stop();
#if defined(__linux__)
    if (shm_ != nullptr) {
        if (registered_) cudaHostUnregister(shm_);
        registered_ = false;
        ::munmap(shm_, map_bytes_);
    }
#endif
    shm_ = nullptr;
    res_mine_ = nullptr;
    res_partner_ = nullptr;
}

bool PeerLink::partner_has(int64_t layer, int64_t expert) const {
    return shm_->serving[1 - role_].load(std::memory_order_relaxed) != 0 &&
           res_partner_[(size_t) (layer * n_expert_ + expert)] >= 0;
}

bool PeerLink::request(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                       const int32_t* kind, std::string& err) {
    rows_posted_ = 0;
    row_of_.clear();
    const int64_t n = n_tok * k;
    if (n > CAP || n_tok > MAXT) { err = "peer link: window too large"; return false; }
    Chan& c = shm_->chan[role_];
    int groups = 0, rows = 0;
    for (int64_t i = 0; i < n; ++i) {   // distinct experts in routing order; each one's entries become compact rows
        if (kind[i] != 2) continue;
        bool seen = false;
        for (int64_t j = 0; j < i; ++j)
            if (kind[j] == 2 && ids[j] == ids[i]) { seen = true; break; }
        if (seen) continue;
        c.expert[groups] = ids[i];
        c.start[groups] = rows;
        for (int64_t j = i; j < n; ++j)
            if (kind[j] == 2 && ids[j] == ids[i]) {
                c.tok[rows] = (int32_t) (j / k);
                row_of_.push_back((int32_t) j);
                ++rows;
            }
        ++groups;
        ++experts_;
    }
    if (groups == 0) return true;
    c.start[groups] = rows;
    c.layer = (int32_t) layer;
    c.n_tok = (int32_t) n_tok;
    c.groups = groups;
    c.rows = rows;
    std::memcpy(c.x, x, (size_t) (n_tok * H) * sizeof(float));
    entries_ += rows;
    rows_posted_ = rows;
    c.t_post_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch()).count();
    c.seq.store(++seq_, std::memory_order_release);
    return true;
}

bool PeerLink::wait(float* out, std::string& err) {
    if (rows_posted_ == 0) return true;
    Chan& c = shm_->chan[role_];
    const auto t0 = std::chrono::steady_clock::now();
    uint32_t spins = 0;
    while (c.done.load(std::memory_order_acquire) != seq_) {
        STRATA_LINK_PAUSE();
        if ((++spins & 0xffffu) == 0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > 10.0) {
            err = "peer link: the other engine did not answer within 10 s";
            return false;
        }
    }
    ms_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (c.status != 0) { err = "peer link: the other engine failed to compute the rows (see its log)"; return false; }
    for (size_t r = 0; r < row_of_.size(); ++r)
        std::memcpy(out + (size_t) row_of_[r] * H, c.out + r * H, (size_t) H * sizeof(float));
    rows_posted_ = 0;
    return true;
}

bool PeerLink::serve(ExpertCache& cache, ExpertSource& src, const std::vector<int32_t>& host_res, std::string& err) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native) { err = "peer link: only native packs are supported"; return false; }
    cache_ = &cache;
    src_ = &src;
    cudaGetDevice(&device_);
    int64_t ff = strata::kernels::cpu::FF;
    for (const auto& f : lay.fmt) ff = std::max<int64_t>(ff, f.n_ff);
    const size_t scratch = std::max<size_t>((size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, ff),
                                            strata::kernels::native_expert_scratch_bytes(CAP, ff));
    stage_blob_ = (lay.max_blob + 255) / 256 * 256;
    int lo = 0, hi = 0;
    cudaDeviceGetStreamPriorityRange(&lo, &hi);   // the other engine waits on these rows: the highest priority
    cudaStream_t s = nullptr;
    const bool ok =
        ck(cudaHostRegister(shm_, map_bytes_, cudaHostRegisterPortable | cudaHostRegisterMapped), "register the link file",
           err) &&
        (registered_ = true) &&
        ck(cudaHostGetDevicePointer(&d_shm_, shm_, 0), "the link file's device address", err) &&
        ck(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, hi), "stream", err) &&
        ck(cudaHostAlloc(&h_meta_, sizeof(Meta), cudaHostAllocPortable | cudaHostAllocMapped), "plan staging", err) &&
        ck(cudaHostGetDevicePointer(&d_meta_, h_meta_, 0), "the plan's device address", err) &&
        ck(cudaMalloc((void**) &d_q8_, (size_t) MAXT * (H / 32) * 36), "activations", err) &&
        ck(cudaMalloc(&d_scratch_, scratch), "scratch", err) &&
        ck(cudaMalloc((void**) &d_stage_, (size_t) kStage * stage_blob_), "fallback blobs", err);
    stream_ = s;
    if (!ok) return false;
    svc_usage_.assign((size_t) (n_layers_ * n_expert_), 0.0f);
    publish(host_res);
    stop_.store(false);
    thread_ = std::thread([this] { loop(); });
    shm_->serving[role_].store(1, std::memory_order_release);
    return true;
}

void PeerLink::set_serving(bool on) {
    if (shm_ != nullptr && role_ >= 0) shm_->serving[role_].store(on ? 1 : 0, std::memory_order_release);
}

void PeerLink::hold() {
    mu_.lock();
    if (hold_depth_++ == 0 && stream_ != nullptr) {
        int prev = 0;
        cudaGetDevice(&prev);
        if (prev != device_) cudaSetDevice(device_);
        cudaStreamSynchronize((cudaStream_t) stream_);   // what the service launched has finished
        if (prev != device_) cudaSetDevice(prev);
    }
}

void PeerLink::unhold() {
    --hold_depth_;
    mu_.unlock();
}

void PeerLink::publish(const std::vector<int32_t>& host_res) {
    if (shm_ == nullptr || host_res.size() != (size_t) (n_layers_ * n_expert_)) return;
    std::lock_guard<std::recursive_mutex> g(mu_);
    svc_res_ = host_res;
    std::memcpy(res_mine_, host_res.data(), host_res.size() * sizeof(int32_t));
}

void PeerLink::decay_served(float f) {
    for (float& v : svc_usage_) v *= f;   // races the service's += harmlessly: these are routing counts
}

void PeerLink::stop() {
    if (shm_ != nullptr && role_ >= 0) shm_->serving[role_].store(0, std::memory_order_release);
    if (thread_.joinable()) {
        stop_.store(true);
        thread_.join();
    }
    if (stream_ != nullptr) {
        int prev = 0;
        cudaGetDevice(&prev);
        if (prev != device_) cudaSetDevice(device_);
        cudaStreamSynchronize((cudaStream_t) stream_);
        cudaStreamDestroy((cudaStream_t) stream_);
        if (d_q8_) cudaFree(d_q8_);
        if (d_scratch_) cudaFree(d_scratch_);
        if (d_stage_) cudaFree(d_stage_);
        if (h_meta_) cudaFreeHost(h_meta_);
        if (prev != device_) cudaSetDevice(prev);
    }
    stream_ = nullptr;
    d_x_ = d_out_ = nullptr;
    d_meta_ = h_meta_ = d_scratch_ = nullptr;
    d_q8_ = d_stage_ = nullptr;
}

#if defined(__linux__)
// The service thread is created after the engine pinned its host thread to one core, so it inherits that one core and
// would time-slice with the spinning host loop (measured: 0.37 ms to pick a request up).  It moves to that core's SMT
// sibling (STRATA_LINK_CORE=N: core N instead).
static void link_pin_thread() {
    int target = -1;
    if (const char* v = std::getenv("STRATA_LINK_CORE")) target = std::atoi(v);
    cpu_set_t cur;
    CPU_ZERO(&cur);
    if (target < 0 && pthread_getaffinity_np(pthread_self(), sizeof(cur), &cur) == 0 && CPU_COUNT(&cur) == 1) {
        int c = -1;
        for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &cur)) { c = i; break; }
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
        std::string list;
        if (f && std::getline(f, list)) {   // "0,16" or "0-1"
            for (char& ch : list) if (ch == '-') ch = ',';
            size_t p = 0;
            while (p < list.size()) {
                const int x = std::atoi(list.c_str() + p);
                if (x != c) { target = x; break; }
                const size_t q = list.find(',', p);
                if (q == std::string::npos) break;
                p = q + 1;
            }
        }
    }
    if (target < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(target, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0)
        std::fprintf(stderr, "strata: peer link service on logical processor %d\n", target);
}
#endif

void PeerLink::loop() {
#if defined(__linux__)
    link_pin_thread();
#endif
    cudaSetDevice(device_);
    Chan& c = shm_->chan[1 - role_];
    auto idle_since = std::chrono::steady_clock::now();
    while (!stop_.load(std::memory_order_relaxed)) {
        const uint64_t s = c.seq.load(std::memory_order_acquire);
        if (s == last_seq_) {
            // spin while requests are coming (one per layer and window); after 20 ms without one, nap briefly
            if (std::chrono::steady_clock::now() - idle_since < std::chrono::milliseconds(20)) {
                for (int i = 0; i < 64; ++i) STRATA_LINK_PAUSE();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
            continue;
        }
        std::string err;
        const auto th0 = std::chrono::steady_clock::now();
        t_recv_ms_ += (double) (std::chrono::duration_cast<std::chrono::nanoseconds>(th0.time_since_epoch()).count() -
                                c.t_post_ns) / 1e6;
        c.status = handle(err) ? 0 : 1;
        t_handle_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - th0).count();
        ++n_handled_;
        if (c.status != 0) std::fprintf(stderr, "strata: %s\n", err.c_str());
        last_seq_ = s;
        c.done.store(s, std::memory_order_release);
        idle_since = std::chrono::steady_clock::now();
    }
}

bool PeerLink::handle(std::string& err) {
    Chan& c = shm_->chan[1 - role_];
    const auto& lay = strata::kernels::cpu::expert_layout();
    const int64_t layer = c.layer, groups = c.groups, rows = c.rows, n_tok = c.n_tok;
    if (layer < 0 || layer >= n_layers_ || groups <= 0 || groups > CAP || rows <= 0 || rows > CAP || n_tok <= 0 ||
        n_tok > MAXT) {
        err = "peer link: a malformed request";
        return false;
    }
    const cudaStream_t s = (cudaStream_t) stream_;
    // zero-copy: the kernels read the activations and the plan straight from (mapped) host memory and write the rows
    // straight into the link file - no copy-engine traffic, which would queue behind this card's own expert copies
    const auto dev = [this](const void* host) {
        return (uint8_t*) d_shm_ + ((const uint8_t*) host - (const uint8_t*) shm_);
    };
    const float* dx = (const float*) dev(c.x);
    float* dout = (float*) dev(c.out);
    const auto& f = lay.fmt[(size_t) layer];
    const auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
    const uint64_t bb = lay.blob_bytes(layer);
    Meta& m = *(Meta*) h_meta_;
    Meta* dm = (Meta*) d_meta_;
    const auto tq0 = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::recursive_mutex> g(mu_);   // this engine is not writing its slots meanwhile
        strata::kernels::quantize_q8_1_rows(dx, n_tok, H, d_q8_, s);
        // passes: every resident expert in the first, plus up to kStage experts copied in from RAM per pass
        std::vector<int32_t> todo((size_t) groups);
        for (int64_t gi = 0; gi < groups; ++gi) todo[(size_t) gi] = (int32_t) gi;
        bool first = true;
        while (!todo.empty()) {
            if (!first && !ck(cudaStreamSynchronize(s), "pass", err)) return false;   // the plan + staging are reused
            int ng = 0, nr = 0, staged = 0;
            std::vector<int32_t> later;
            for (int32_t gi : todo) {
                const int32_t e = c.expert[gi];
                if (e < 0 || e >= n_expert_) { err = "peer link: an expert id is out of range"; return false; }
                const size_t ri = (size_t) (layer * n_expert_ + e);
                const int32_t slot = svc_res_[ri];
                unsigned long long ptr = 0;
                if (slot >= 0) {
                    ptr = (unsigned long long) cache_->device_slot(slot);
                } else {
                    if (staged == kStage) { later.push_back(gi); continue; }
                    const uint8_t* b = src_->blob_stable(layer, e);
                    if (b == nullptr) { err = "peer link: an expert is in neither tier"; return false; }
                    uint8_t* dst = d_stage_ + (size_t) staged * stage_blob_;
                    if (!ck(cudaMemcpyAsync(dst, b, (size_t) bb, cudaMemcpyHostToDevice, s), "fallback copy", err))
                        return false;
                    ptr = (unsigned long long) dst;
                    ++staged;
                    served_fallback_.fetch_add(1, std::memory_order_relaxed);
                }
                m.ptr[ng] = ptr;
                m.start[ng] = nr;
                for (int32_t r = c.start[gi]; r < c.start[gi + 1]; ++r) {
                    m.dst[nr] = r;        // the compact row the requester expects
                    m.tok[nr] = c.tok[r];
                    ++nr;
                }
                svc_usage_[ri] += (float) (c.start[gi + 1] - c.start[gi]);
                ++ng;
            }
            m.start[ng] = nr;
            m.count[0] = ng;
            m.count[1] = nr;
            std::atomic_thread_fence(std::memory_order_seq_cst);   // the plan is in host memory the kernel reads
            strata::kernels::native_expert_grouped(L, dm->ptr, dm->start, dm->count, dm->dst, dm->tok, ng, nr, d_q8_,
                                                   d_scratch_, dout, s);
            todo.swap(later);
            first = false;
        }
    }
    served_entries_.fetch_add(rows, std::memory_order_relaxed);
    const auto tq = std::chrono::steady_clock::now();
    t_enqueue_ms_ += std::chrono::duration<double, std::milli>(tq - tq0).count();
    cudaError_t e;   // spin: a blocking sync would sleep this thread and wake it late
    while ((e = cudaStreamQuery(s)) == cudaErrorNotReady) STRATA_LINK_PAUSE();
    return ck(e, "compute", err);
}

}  // namespace strata::core
