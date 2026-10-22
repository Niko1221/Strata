// src/qwen35/q36_peer_prefill.cu - see src/qwen35/q36_peer_prefill.cuh.
//
// THREE PARTS, and the split is what keeps the hot call free of host work:
//   * open()    binds to one helper's VRAM cache, finds that helper's CUDA device from the blob addresses the cache
//               itself hands out, establishes the route the window crosses by (the peering link when the two GPUs have
//               one, pinned host memory when they do not), and allocates every buffer this executor owns - the
//               compute ones on the helper, the staging ones in host memory - all freed in close().
//   * prepare() stages the window's activations and routing on the helper and turns them into a ROW PLAN there:
//               which entry is rank r of this helper's run, which expert each activation row belongs to.  This is
//               the only place that reads a number back to the host (8 bytes), because MMQ's launches are sized on
//               the host.
//   * launch()  is the primary's MMQ branch, restricted to the entries this helper owns: q8_1 activations, gate+up
//               as one [2 FF] product read IN PLACE from the cache, SwiGLU, down, and the run back to the primary's
//               `eparts` - one peer copy, or one copy into host staging where there is no link.  No host
//               synchronisation.
//
// WHY THE DEVICE, NOT THE HOST, DECIDES THE ROW ORDER.  The row contract (see the header) is a prefix count over
// entry index, so it is a scan over `n_tok * k` entries - 4096 of them for a 512-token chunk.  Doing it on the
// helper keeps the routing on the device where it already is; only the two counts MMQ's host-side launch geometry
// needs come back.
#include "q36_peer_prefill.cuh"

#include "strata/core/device.hpp"   // primary_device() (STRATA_PRIMARY_DEVICE): the device whose memory we copy from
#include "strata/core/remote_experts.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace strata::qwen35 {
namespace {

namespace MMQ = strata::prefill::mmq;

constexpr int kThreads = 256;

inline unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }

/// Switches the calling thread to `dev` and back.  The engine's own thread lives on the primary device, so every
/// CUDA call that must run on the helper is inside one of these.
struct On {
    int prev = 0;
    explicit On(int dev) {
        cudaGetDevice(&prev);
        if (prev != dev) cudaSetDevice(dev);
    }
    ~On() {
        int cur = 0;
        cudaGetDevice(&cur);
        if (cur != prev) cudaSetDevice(prev);
    }
};

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("q36 peer prefill: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

// ---- the row plan, on the helper ----
//
// `owned` is one byte per expert id: whether the helper's cache holds it (so an entry's routing id answers the whole
// "does this entry belong to me" question with one indexed load).  `pos` is the expert's position in the plan's
// expert order (-1 when the cache does not hold it), which is the order the products consume the activation rows in.

/// rank[e] = the number of owned entries BELOW e (the row contract's prefix count), for every owned e.  One block:
/// each thread counts a contiguous chunk, the block scans the chunk counts in shared memory, then each thread walks
/// its chunk again handing out ranks.  A chunk per thread and not a grid-stride loop, because the answer depends on
/// the order of e and only contiguous chunks compose into an ascending rank.
__global__ void k_peer_rank(const int32_t* __restrict__ ids, const uint8_t* __restrict__ owned, int n_ent,
                            int32_t* __restrict__ rank) {
    __shared__ int cnt[kThreads];
    __shared__ int off[kThreads];
    const int tid = threadIdx.x;
    const int chunk = (n_ent + kThreads - 1) / kThreads;
    const int lo = tid * chunk;
    const int hi = min(n_ent, lo + chunk);
    int c = 0;
    for (int e = lo; e < hi; ++e)
        if (owned[ids[e]]) ++c;
    cnt[tid] = c;
    __syncthreads();
    if (tid == 0) {
        int p = 0;
        for (int t = 0; t < kThreads; ++t) { off[t] = p; p += cnt[t]; }
    }
    __syncthreads();
    int r = off[tid];
    for (int e = lo; e < hi; ++e)
        if (owned[ids[e]]) rank[e] = r++;
}

__global__ void k_peer_count(const int32_t* __restrict__ ids, const uint8_t* __restrict__ owned,
                             const int32_t* __restrict__ pos, int n_ent, int32_t* __restrict__ cnt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    if (!owned[ids[e]]) return;
    const int p = pos[ids[e]];
    if (p >= 0) atomicAdd(&cnt[p], 1);
}

/// bounds[pos] / cur[pos] from the counts, plus the widest expert (the products' launch grid) and the owned total.
__global__ void k_peer_scan(int32_t* __restrict__ cnt, int n_pos, int32_t* __restrict__ bounds,
                            int32_t* __restrict__ cur, int32_t* __restrict__ max_rows) {
    if (threadIdx.x != 0) return;
    int p = 0, mx = 0;
    for (int s = 0; s < n_pos; ++s) {
        bounds[s] = p;
        cur[s] = p;
        p += cnt[s];
        mx = max(mx, cnt[s]);
    }
    bounds[n_pos] = p;
    *max_rows = mx;
}

/// Each owned entry to a row: `src` = its token (the quantizer's gather), `dst` = its RANK (the product's output row
/// inside the helper's run).  The expert order decides which activation row it lands on; the rank decides where its
/// result goes - MMQ writes dst row `ids[activation row]`, so the run's order and the plan's order stay independent.
__global__ void k_peer_scatter(const int32_t* __restrict__ ids, const uint8_t* __restrict__ owned,
                               const int32_t* __restrict__ pos, const int32_t* __restrict__ rank, int n_ent, int k,
                               int32_t* __restrict__ cur, int32_t* __restrict__ src, int32_t* __restrict__ dst) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    if (!owned[ids[e]]) return;
    const int p = pos[ids[e]];
    if (p < 0) return;
    const int r = atomicAdd(&cur[p], 1);
    src[r] = e / k;
    dst[r] = rank[e];
}

}  // namespace

struct PeerPrefill36::Impl {
    int device = -1;
    int primary = -1;   // the ordinal that OWNS the primary's memory: every peer copy names it, never a literal 0
    int64_t n_layers = 0, n_expert = 0, k = 0, n_embd = 0, n_ff = 0, cap = 0;
    std::vector<strata::kernels::NativeExpertLayout> layouts;   // the SAME table RemoteExperts::open() got
    const strata::core::RemoteExperts* helper = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t ev = nullptr;
    std::unique_ptr<MMQ::Context> mctx;   // per executor, created on the helper

    // WHICH ROUTE THE WINDOW CROSSES BY - decided once in open() and fixed for the life of the executor.  true: the
    // helper reads the primary's activations and writes the primary's eparts over the peering link.  false: the two
    // GPUs have no link (measured on 192.168.1.98, see open()), so the same bytes go through the pinned buffers below.
    bool p2p = true;
    // The host route's staging, pinned, allocated in open() ONLY when that route is taken - a box with a working link
    // allocates none of it, and its allocations are therefore what they were.
    float* h_x = nullptr;        // [cap][n_embd] the primary's activations, on their way down
    int32_t* h_ids = nullptr;    // [cap]          the primary's routing
    float* h_out = nullptr;      // [cap][n_embd] this helper's run, on its way back
    // What a host-route launch() left in h_out, and where it belongs.  It is remembered rather than uploaded at once
    // because wait() is the only method the caller hands the PRIMARY's stream to, and the upload has to be ordered
    // against the primary's own work there.
    float* pending_eparts = nullptr;
    int64_t pending_base = 0, pending_n = 0;

    // everything below is allocated on `device`
    float* d_x = nullptr;      // [cap][n_embd] the staged activations
    float* d_gu = nullptr;     // [cap][2 n_ff] the gate+up product
    float* d_h = nullptr;      // [cap][n_ff]   SwiGLU out
    float* d_out = nullptr;    // [cap][n_embd] this helper's run, before the one peer copy
    float* d_ys = nullptr;     // [cap]         NVFP4 w4a4 per-row activation scales (unused otherwise)
    void* d_xq = nullptr;      // q8_1 of the activations
    void* d_hq = nullptr;      // q8_1 of the SwiGLU out
    int32_t* d_ids = nullptr;      // [cap] the staged routing
    int32_t* d_ident = nullptr;    // [cap] identity row map: the products' "no reordering" table
    int32_t* d_src = nullptr;      // [cap] activation row -> token
    int32_t* d_map = nullptr;      // [cap] activation row -> output row of the run
    int32_t* d_rank = nullptr;     // [cap] entry -> its rank among owned entries
    int32_t* d_pos = nullptr;      // [n_expert] expert -> position in the plan's expert order
    int32_t* d_cnt = nullptr;      // [n_expert + 1] rows per position
    int32_t* d_cur = nullptr;      // [n_expert + 1] the scatter's cursors
    int32_t* d_bounds = nullptr;   // [n_expert + 1] the products' expert bounds
    int32_t* d_maxr = nullptr;     // [1]           the widest expert
    uint8_t* d_owned = nullptr;    // [n_expert]    the residency predicate

    // host scratch, reused per layer
    std::vector<uint8_t> owned_host;
    std::vector<int32_t> pos_host;
    std::vector<std::pair<uintptr_t, int32_t>> ptrs;   // (blob address, expert), resident only, sorted

    /// One MMQ::Product's worth of experts: `w + z * delta` is expert z's blob.  A layer's resident experts usually
    /// form ONE such run (the cache hands slots out in admission order and a layer's blobs all have that layer's
    /// size); a run splits where the gap between two blobs differs.  Splitting is not a correctness compromise: each
    /// run is still a contiguous slice of the plan's expert order, so the shared `bounds` array indexes straight
    /// into it.
    struct Group {
        const uint8_t* w = nullptr;
        size_t delta = 0;
        int first = 0;   // its first expert's position in the plan's expert order
        int n = 0;
    };
    std::vector<Group> groups;
    int64_t n_pos = 0;
    /// Whether the last prepared layer's resident experts are EQUIDISTANT, i.e. one product can address all of
    /// them as `w + z * expert_bytes`.  A layer that is not equidistant is not served by the peer path at all
    /// (the engine's peer_owned_layer() asks layer_uniform() first).
    bool uniform = true;

    int64_t prep_layer = -1;
    bool prepared = false;
    int64_t n_owned = 0, max_rows = 0;
    int64_t rows = 0, launches = 0;
};

PeerPrefill36::PeerPrefill36() : impl_(new Impl()) {}
PeerPrefill36::~PeerPrefill36() { close(); }

bool PeerPrefill36::open(const strata::core::RemoteExperts& helper, int64_t n_layers, int64_t n_expert, int64_t k,
                         int64_t n_embd, int64_t n_ff, int64_t cap_entries,
                         const strata::kernels::NativeExpertLayout* layouts, std::string& err) {
    close();
    Impl& m = *impl_;
    if (layouts == nullptr) { err = "q36 peer prefill: no expert layout table"; return false; }
    if (n_layers <= 0 || n_expert <= 0 || k <= 0) { err = "q36 peer prefill: bad layer/expert/k geometry"; return false; }
    if (n_embd <= 0 || n_ff <= 0 || cap_entries <= 0) { err = "q36 peer prefill: bad expert width or window size"; return false; }
    if (cap_entries > 0x7fffffffLL) { err = "q36 peer prefill: window too wide for a 32-bit entry index"; return false; }
    if (!MMQ::built()) {
        err = "q36 peer prefill: this build has no MMQ (the helpers' prompt path is the primary's MMQ branch)";
        return false;
    }

    // THE HELPER'S DEVICE, FROM THE TIER.  RemoteExperts::device() is the accessor PeerExperts has had all along
    // (origin/main include/strata/core/peer_experts.hpp: `int device() const { return device_; }`); this class used to
    // have no such accessor, so the device was read back out of the first resident blob's pointer with
    // cudaPointerGetAttributes over an O(n_layers * n_expert) scan of slot_ptr().  Both are gone: the tier answers.
    // device() < 0 means open() never ran or failed, and an executor cannot bind to that.
    const int dev = helper.device();
    if (dev < 0) {
        err = "q36 peer prefill: the helper tier is not open (device() < 0), so there is no device to bind to";
        return false;
    }
    // THE PRIMARY'S DEVICE, ASKED RATHER THAN ASSUMED.  Every peer copy below names the device that owns the memory on
    // the other side, and the engine's own session may run on any visible ordinal (STRATA_PRIMARY_DEVICE; the engine's
    // peer_open asks this same function).  A literal 0 names a card that does not hold the activations or `eparts` on
    // such a box, so the peer-access request below fails for a GPU the model is not even running on and the whole
    // prompt path silently stays local.
    const int prim = strata::core::primary_device();
    if (prim < 0) {
        err = "q36 peer prefill: the primary device is not set (primary_device() < 0), so there is no device to copy from";
        return false;
    }
    if (prim == dev) {   // a helper cannot be the primary: every copy below would name one and the same card
        err = "q36 peer prefill: the helper is on CUDA" + std::to_string(dev) +
              ", which is the primary device; a helper cannot be the device the model runs on";
        return false;
    }
    // ONE consistency check, on one resident blob: the tier's own ordinal must agree with where its cache's memory
    // actually lives.  A GPU whose device() disagreed with its pointers would put every buffer this executor allocates
    // on the wrong card, and the failure would look like a wrong answer rather than a wrong device - so it is checked
    // once here and refused if it does not hold.  A tier holding nothing is also refused: there is nothing to compute
    // and the cache is filled at open time, so an empty one is a programming error, not a wait.
    const uint8_t* probe = nullptr;
    for (int64_t l = 0; l < n_layers && probe == nullptr; ++l)
        for (int64_t e = 0; e < n_expert && probe == nullptr; ++e) probe = helper.slot_ptr(l, (int32_t) e);
    if (probe == nullptr) {
        err = "q36 peer prefill: the helper's VRAM cache holds no expert, so there is nothing for it to compute";
        return false;
    }
    {
        cudaPointerAttributes attr;
        std::memset(&attr, 0, sizeof(attr));
        if (!ck(cudaPointerGetAttributes(&attr, probe), "cudaPointerGetAttributes", err)) return false;
        if (attr.type != cudaMemoryTypeDevice || attr.device != dev) {
            err = "q36 peer prefill: the helper tier reports CUDA" + std::to_string(dev) + " but its resident blobs are " +
                  (attr.type == cudaMemoryTypeDevice ? "on CUDA" + std::to_string(attr.device)
                                                     : std::string("not on a device at all"));
            return false;
        }
    }

    m.device = dev;
    m.primary = prim;
    m.helper = &helper;
    m.n_layers = n_layers;
    m.n_expert = n_expert;
    m.k = k;
    m.n_embd = n_embd;
    m.n_ff = n_ff;
    m.cap = cap_entries;
    m.layouts.assign(layouts, layouts + (size_t) n_layers);
    for (const auto& L : m.layouts) {
        if (L.gu_type < 0 || L.d_type < 0 || L.bytes == 0 || L.down_off >= L.bytes) {
            err = "q36 peer prefill: the expert layout table has an empty or inconsistent layer";
            close();
            return false;
        }
        if (!MMQ::supported(L.gu_type) || !MMQ::supported(L.d_type)) {
            err = "q36 peer prefill: MMQ has no kernel for this layer's expert types";
            close();
            return false;
        }
        if (!MMQ::fits(L.gu_type, 2 * n_ff) || !MMQ::fits(L.d_type, n_embd)) {
            err = "q36 peer prefill: MMQ has no tile that fits this layer's expert matrices on a visible GPU";
            close();
            return false;
        }
    }
    m.owned_host.assign((size_t) n_expert, 0);
    m.pos_host.assign((size_t) n_expert, -1);

    On on(m.device);
    // THE ROUTE.  The peering link is tried FIRST, and on a box that has one it is the only route ever taken.  The
    // point of asking for it is that the helper then reads the primary's activations and writes the primary's eparts
    // directly instead of the driver staging every one of those copies through host memory.
    //
    // But on the one box this feature runs on (192.168.1.98: a Tesla P100 as the primary, three P104-100 helpers)
    // there is no link at all: CanAccessPeer is 0 for all twelve ordered pairs, cudaDeviceEnablePeerAccess(0->1)
    // answers "peer access is not supported between these two devices", and cudaMemcpyPeer(0->1) returns success
    // having moved nothing (the driver stages it).  Refusing the helper there - what this did before - left the whole
    // prompt path dead on that box, so the route is now RECORDED instead: with no link the three copies go through
    // pinned host memory (prepare()'s down leg, launch()/wait()'s up leg), which is where the driver would have put
    // them anyway, only without a second staging copy.  Flash-Next's prompt path splits the same way (src/prefill/
    // prefill.cpp: Prefill::set_peer's P.p2p and the `!P.p2p` device->host then host->device copy pairs, with pinned
    // host_x / host_rows) and carries the same escape hatch: STRATA_Q36_PREFILL_PEERS_HOST=0 restores this class's old
    // answer - a helper without a link does not open, and the engine's clean local fallback (peer_open() simply does
    // not take the helper) applies.
    cudaError_t pca = cudaDeviceEnablePeerAccess(m.primary, 0);
    if (pca == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        pca = cudaSuccess;
    }
    // The capability query as a second witness, in the direction that matters: the helper reaching the primary's
    // memory.  A link the enable call accepts but this denies is the shape #2666 found (a copy that reports success
    // and carries nothing), and the host route is the safe answer there as well.
    int can = 0;
    const cudaError_t pcq = cudaDeviceCanAccessPeer(&can, m.primary, m.device);
    if (pcq != cudaSuccess) cudaGetLastError();
    std::string no_p2p;   // why the link is not usable: empty means it is
    if (pca != cudaSuccess) {
        no_p2p = cudaGetErrorString(pca);
    } else if (pcq != cudaSuccess) {
        no_p2p = "CUDA refused to ask whether the two GPUs have a peer path";
    } else if (can == 0) {
        no_p2p = "CUDA reports no peer path from this helper to the primary's memory";
    }
    if (!no_p2p.empty()) cudaGetLastError();   // a recorded route, not this run's error: drop the sticky code
    // OPT-IN, and the exact string "1" - the same parse q36_engine.cu's peer_host_route() makes, mirrored rather
    // than reached into.  The route is off for every other value, because it does not survive a decode step yet:
    // measured on 192.168.1.98, `--bench-pp 160 --bench-tg 0` is clean but any `--bench-tg > 0` after the prompt dies
    // with an illegal memory access, while the same run with the route off is clean (see peer_host_route()'s comment
    // in q36_engine.cu for the full matrix).  A default-on route would turn a working run into a crash.
    const bool host_route_ok = [] {
        const char* v = std::getenv("STRATA_Q36_PREFILL_PEERS_HOST");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    if (!no_p2p.empty() && !host_route_ok) {
        err = std::string("q36 peer prefill: no peer link between CUDA") + std::to_string(m.device) + " and CUDA" +
              std::to_string(m.primary) + ": " + no_p2p +
              " (the prompt path writes the primary's rows over P2P; set STRATA_Q36_PREFILL_PEERS_HOST=1 to stage "
              "them through host memory instead - experimental, it does not survive a decode step yet)";
        close();
        return false;
    }
    m.p2p = no_p2p.empty();
    if (!ck(cudaStreamCreateWithFlags(&m.stream, cudaStreamNonBlocking), "helper stream", err) ||
        !ck(cudaEventCreateWithFlags(&m.ev, cudaEventDisableTiming), "helper event", err)) {
        close();
        return false;
    }
    // MMQ's context holds a per-device pool, so it is created (and destroyed) on the helper, one per executor.
    m.mctx.reset(new MMQ::Context());

    const size_t nf = (size_t) cap_entries;
    const bool ok =
        ck(cudaMalloc((void**) &m.d_x, nf * (size_t) n_embd * sizeof(float)), "activation staging", err) &&
        ck(cudaMalloc((void**) &m.d_gu, nf * (size_t) (2 * n_ff) * sizeof(float)), "gate+up staging", err) &&
        ck(cudaMalloc((void**) &m.d_h, nf * (size_t) n_ff * sizeof(float)), "SwiGLU staging", err) &&
        ck(cudaMalloc((void**) &m.d_out, nf * (size_t) n_embd * sizeof(float)), "result run", err) &&
        ck(cudaMalloc((void**) &m.d_ys, nf * sizeof(float)), "activation scales", err) &&
        ck(cudaMalloc(&m.d_xq, MMQ::q8_bytes(cap_entries, n_embd)), "q8 activations", err) &&
        ck(cudaMalloc(&m.d_hq, MMQ::q8_bytes(cap_entries, n_ff)), "q8 SwiGLU", err) &&
        ck(cudaMalloc((void**) &m.d_ids, nf * sizeof(int32_t)), "routing staging", err) &&
        ck(cudaMalloc((void**) &m.d_ident, nf * sizeof(int32_t)), "identity rows", err) &&
        ck(cudaMalloc((void**) &m.d_src, nf * sizeof(int32_t)), "gather rows", err) &&
        ck(cudaMalloc((void**) &m.d_map, nf * sizeof(int32_t)), "output rows", err) &&
        ck(cudaMalloc((void**) &m.d_rank, nf * sizeof(int32_t)), "entry ranks", err) &&
        ck(cudaMalloc((void**) &m.d_pos, (size_t) (n_expert + 1) * sizeof(int32_t)), "expert order", err) &&
        ck(cudaMalloc((void**) &m.d_cnt, (size_t) (n_expert + 1) * sizeof(int32_t)), "expert counts", err) &&
        ck(cudaMalloc((void**) &m.d_cur, (size_t) (n_expert + 1) * sizeof(int32_t)), "expert cursors", err) &&
        ck(cudaMalloc((void**) &m.d_bounds, (size_t) (n_expert + 1) * sizeof(int32_t)), "expert bounds", err) &&
        ck(cudaMalloc((void**) &m.d_maxr, sizeof(int32_t)), "widest expert", err) &&
        ck(cudaMalloc((void**) &m.d_owned, (size_t) n_expert), "residency predicate", err);
    if (!ok) {
        close();
        return false;
    }
    // THE HOST ROUTE'S STAGING, and only on that route: a box with a working link allocates nothing here, so its
    // allocation list is exactly what it was.  Pinned, because every copy that touches these buffers is a
    // cudaMemcpyAsync on a stream - a pageable buffer would make each one a synchronising copy.  Sized from the same
    // cap_entries the helper's GPU buffers are: h_x is [cap][n_embd] floats, which bounds n_tok * n_embd because a
    // window is capped at n_tok * k entries with k >= 1.  Allocated with the primary current: pinned memory is
    // host-wide, but the copies that read and write it name the primary.
    if (!m.p2p) {
        On onp(m.primary);
        const bool hok =
            ck(cudaHostAlloc((void**) &m.h_x, nf * (size_t) n_embd * sizeof(float), cudaHostAllocDefault),
               "pinned activations", err) &&
            ck(cudaHostAlloc((void**) &m.h_ids, nf * sizeof(int32_t), cudaHostAllocDefault), "pinned routing", err) &&
            ck(cudaHostAlloc((void**) &m.h_out, nf * (size_t) n_embd * sizeof(float), cudaHostAllocDefault),
               "pinned result rows", err);
        if (!hok) {   // a null staging buffer would turn every copy below into a silent no-op: refuse the helper
            close();
            return false;
        }
    }
    // The products' identity row map, built once: the gate+up product writes row r for activation row r, so `dst`
    // row == bounds row and the SwiGLU/down chain keeps one index space.  An all-zero buffer (what cudaMalloc gives)
    // would send every row to row 0, so this is filled before anything can read it, on the same stream.
    if (!ck(cudaMemsetAsync(m.d_ident, 0, nf * sizeof(int32_t), m.stream), "clear identity rows", err)) {
        close();
        return false;
    }
    MMQ::iota(m.d_ident, (int64_t) cap_entries, m.stream);
    if (!ck(cudaGetLastError(), "identity rows", err)) return false;
    // One line per executor, and only once it is open: the host route is a real change in how a layer's bytes travel,
    // so it is said out loud - naming the helper, the card on the other side and CUDA's own reason.  A box with a
    // working link prints nothing here, so its stderr is exactly what it was.
    if (!m.p2p)
        std::fprintf(stderr, "q36 peer prefill: CUDA%d's prompt rows cross through host staging (no peer link with "
                             "CUDA%d: %s)\n", m.device, m.primary, no_p2p.c_str());
    return true;
}

void PeerPrefill36::close() {
    Impl& m = *impl_;
    if (m.device < 0 && m.stream == nullptr) {
        m.helper = nullptr;
        return;
    }
    if (m.device >= 0) {
        On on(m.device);
        if (m.stream != nullptr) cudaStreamSynchronize(m.stream);
        m.mctx.reset();   // its pool belongs to the helper: destroy it there
        void* frees[] = {m.d_x,  m.d_gu, m.d_h,       m.d_out, m.d_ys,   m.d_xq,     m.d_hq,
                         m.d_ids, m.d_ident, m.d_src, m.d_map, m.d_rank, m.d_pos, m.d_cnt,   m.d_cur,
                         m.d_bounds, m.d_maxr, m.d_owned};
        for (void* p : frees)
            if (p != nullptr) cudaFree(p);
        if (m.ev != nullptr) cudaEventDestroy(m.ev);
        if (m.stream != nullptr) cudaStreamDestroy(m.stream);
        cudaGetLastError();
    }
    // The host route's staging, freed after the helper's stream is gone (its last copy into it was on that stream) and
    // after the PRIMARY has settled (wait() issued its H2D on a stream this class only ever sees as an argument, so
    // the one handle it can drain is the device's).  Both routes clear the pointers, so a close() of a link box leaves
    // nothing to confuse the next open().
    if (m.primary >= 0 && (m.h_x != nullptr || m.h_ids != nullptr || m.h_out != nullptr)) {
        On onp(m.primary);
        cudaDeviceSynchronize();
        if (m.h_x != nullptr) cudaFreeHost(m.h_x);
        if (m.h_ids != nullptr) cudaFreeHost(m.h_ids);
        if (m.h_out != nullptr) cudaFreeHost(m.h_out);
        cudaGetLastError();
    }
    m.device = -1;
    m.primary = -1;
    m.p2p = true;
    m.h_x = nullptr;
    m.h_ids = nullptr;
    m.h_out = nullptr;
    m.pending_eparts = nullptr;
    m.pending_base = 0;
    m.pending_n = 0;
    m.stream = nullptr;
    m.ev = nullptr;
    m.d_x = m.d_gu = m.d_h = m.d_out = m.d_ys = nullptr;
    m.d_xq = m.d_hq = nullptr;
    m.d_ids = m.d_ident = m.d_src = m.d_map = m.d_rank = nullptr;
    m.d_pos = m.d_cnt = m.d_cur = m.d_bounds = m.d_maxr = nullptr;
    m.d_owned = nullptr;
    m.helper = nullptr;
    m.groups.clear();
    m.ptrs.clear();
    m.n_pos = 0;
    m.prepared = false;
    m.prep_layer = -1;
    m.n_owned = m.max_rows = 0;
    m.rows = m.launches = 0;
}

bool PeerPrefill36::valid() const { return impl_ && impl_->device >= 0 && impl_->stream != nullptr; }

bool PeerPrefill36::owns(int64_t layer, int32_t expert) const {
    const Impl& m = *impl_;
    if (m.device < 0 || m.helper == nullptr) return false;
    if (layer < 0 || layer >= m.n_layers) return false;
    if (expert < 0 || expert >= m.n_expert) return false;
    return m.helper->slot_ptr(layer, expert) != nullptr;
}

int64_t PeerPrefill36::cap_entries() const { return impl_->cap; }

bool PeerPrefill36::prepare(int64_t layer, const float* h_primary, const int32_t* ids_primary, int64_t n_tok,
                            std::string& err) {
    Impl& m = *impl_;
    if (m.device < 0 || m.stream == nullptr) { err = "q36 peer prefill: prepare() on a closed executor"; return false; }
    if (h_primary == nullptr || ids_primary == nullptr) { err = "q36 peer prefill: null activations or routing"; return false; }
    if (layer < 0 || layer >= m.n_layers) { err = "q36 peer prefill: layer out of range"; return false; }
    if (n_tok <= 0) { err = "q36 peer prefill: empty window"; return false; }
    if (n_tok * m.k > m.cap) {
        err = "q36 peer prefill: a window of " + std::to_string(n_tok * m.k) + " entries exceeds the " +
              std::to_string(m.cap) + " open() allocated";
        return false;
    }
    m.prepared = false;
    const int n_ent = (int) (n_tok * m.k);
    const int64_t ne = m.n_expert;

    // ---- the layer's residency, and the expert order the products will consume, from the cache itself (host side,
    // no CUDA call): which experts this helper holds, where each blob lies, and the arithmetic runs that the MMQ
    // product's `w + z * expert_bytes` addressing can cover. ----
    m.ptrs.clear();
    std::fill(m.owned_host.begin(), m.owned_host.end(), (uint8_t) 0);
    for (int32_t e = 0; e < ne; ++e) {
        const uint8_t* p = m.helper->slot_ptr(layer, e);
        if (p == nullptr) continue;
        m.owned_host[(size_t) e] = 1;
        m.ptrs.emplace_back((uintptr_t) p, e);
    }
    std::sort(m.ptrs.begin(), m.ptrs.end());
    std::fill(m.pos_host.begin(), m.pos_host.end(), -1);
    m.groups.clear();
    // THE GROUPS, and the rule that changed: a layer's resident experts are ONE run whenever their blobs are
    // EQUIDISTANT, even when the first two happen to be the closest pair.  The old rule compared every gap with
    // the PREVIOUS group's delta, and the first group was created with delta = 0 - so the first expert was always
    // alone in a group of its own, and the second started a new group even when every gap was the same.  That
    // split is what broke the products: MMQ's `expert_bounds` (the `bounds` a Product carries) is indexed by the
    // product's OWN expert numbers, and the products read `xq` through `expert_bounds[z] .. expert_bounds[z + 1]`
    // (third_party/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:1014).  A group that starts at the plan's expert `first`
    // must therefore carry the bounds of THAT slice, and its `xq` rows must be the slice's rows - which is only
    // true when the slice is the whole plan.  One group over the whole layer is that case, and it is also the
    // case the plan's own order wants (the plan's expert order IS the address order).
    //
    // When the blobs are NOT equidistant the layer cannot be served by one product, and the plan must not claim
    // it: `uniform` says so, and the engine falls back to the local route for this layer (see
    // PeerPrefill36::layer_uniform).
    bool uniform = true;
    size_t delta = 0;
    for (size_t i = 0; i < m.ptrs.size(); ++i) {
        const uintptr_t p = m.ptrs[i].first;
        m.pos_host[(size_t) m.ptrs[i].second] = (int32_t) i;
        if (i == 0) continue;
        const size_t d = p - m.ptrs[i - 1].first;
        if (i == 1) delta = d;
        else if (d != delta) { uniform = false; break; }
    }
    m.uniform = uniform;
    if (uniform && !m.ptrs.empty())
        m.groups.push_back(Impl::Group{(const uint8_t*) m.ptrs[0].first, delta, 0, (int) m.ptrs.size()});
    else
        for (size_t i = 0; i < m.ptrs.size(); ++i) {
            const uintptr_t p = m.ptrs[i].first;
            if (i == 0) {
                m.groups.push_back(Impl::Group{(const uint8_t*) p, 0, 0, 1});
                continue;
            }
            const size_t d = p - m.ptrs[i - 1].first;
            if (d != m.groups.back().delta) m.groups.push_back(Impl::Group{(const uint8_t*) p, d, (int) i, 1});
            else m.groups.back().n += 1;
        }
    m.n_pos = (int64_t) m.ptrs.size();
    // DIAG: does `w + z * delta` really land on expert z's blob?  A mismatch here is the difference between the
    // first expert's rows (which come out right) and every other expert's (which come out NaN).  Remove after the fix.
    if (std::getenv("STRATA_Q36_PREFILL_DIAG") != nullptr && uniform && m.ptrs.size() > 1) {
        int64_t bad = 0;
        uintptr_t first_bad = 0;
        int64_t first_bad_z = -1;
        for (size_t z = 0; z < m.ptrs.size(); ++z) {
            const uintptr_t want = m.ptrs[0].first + z * delta;
            if (want != m.ptrs[z].first) {
                if (bad == 0) { first_bad = want; first_bad_z = (int64_t) z; }
                ++bad;
            }
        }
        std::fprintf(stderr, "DIAG helper dev=%d layer %lld addr check: %lld of %zu experts mismatch "
                             "(first: z=%lld want=%p have=%p delta=%llu)\n",
                     m.device, (long long) layer, (long long) bad, m.ptrs.size(), (long long) first_bad_z,
                     (void*) first_bad, (void*) (first_bad_z >= 0 ? m.ptrs[(size_t) first_bad_z].first : 0),
                     (unsigned long long) delta);
        // The first few of this layer's experts and the address step between them: the cache lays its slots out in
        // the order it was offered the experts, so this says whether the experts of ONE layer are side by side in
        // the arena (which is what MMQ's `w + z * step` needs) or spread across it.
        {
            std::fprintf(stderr, "DIAG helper dev=%d layer %lld addrs:", m.device, (long long) layer);
            for (size_t z = 0; z < m.ptrs.size() && z < 5; ++z)
                std::fprintf(stderr, " e%d@%p(+%llu)", m.ptrs[z].second, (void*) m.ptrs[z].first,
                             (unsigned long long) (m.ptrs[z].first - m.ptrs[0].first));
            std::fprintf(stderr, "\n");
        }
    }
    // DIAG: the FIRST bytes of the helper's cached blob for this layer's first expert, and the bytes of the SAME
    // expert in the host blob the primary reads.  Equal => the cache holds the model's own bytes; different => the
    // cache holds something else and the MMQ would read a type it does not expect.  Remove after the fix.
    if (std::getenv("STRATA_Q36_PREFILL_DIAG") != nullptr && !m.ptrs.empty()) {
        const int32_t e0 = m.ptrs[0].second;
        unsigned char hb[16] = {};
        cudaMemcpy(hb, (const void*) m.ptrs[0].first, 16, cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "DIAG helper dev=%d layer %lld expert %d bytes:", m.device, (long long) layer, e0);
        for (int i = 0; i < 16; ++i) std::fprintf(stderr, " %d", hb[i]);
        std::fprintf(stderr, "\n");
    }
    m.prep_layer = layer;
    m.n_owned = 0;
    m.max_rows = 0;
    m.rows = 0;

    // ---- THE HOST ROUTE'S DOWN LEG, and it runs BEFORE the helper's device scope below.  WHICH COPIES RUN HERE IS
    // THE ROUTE'S BUSINESS, so the whole window copy is in one place.
    //
    // Link: nothing happens here at all - the two cudaMemcpyPeerAsync below go straight off the primary's memory onto
    // the helper's stream, and nothing waits on the host.
    //
    // No link: two BLOCKING device-to-host copies into the pinned staging.  A blocking copy is correct here and is
    // NOT a new synchronisation point: the caller (q36's MMQ branch) has already synchronized the primary's stream
    // immediately before this call - it needs the routing on the host to build its own row plan - so the activations
    // are quiescent and there is nothing left for the copy to wait for.  It costs n_tok * (n_embd * 4 + k * 4) bytes
    // once per layer, and it buys the whole feature on a box whose two GPUs have no path between them.
    if (!m.p2p) {
        On onp(m.primary);   // the two source pointers are the primary's: the copies run on its device
        if (!ck(cudaMemcpy(m.h_x, h_primary, (size_t) n_tok * (size_t) m.n_embd * sizeof(float),
                           cudaMemcpyDeviceToHost), "stage activations through host", err) ||
            !ck(cudaMemcpy(m.h_ids, ids_primary, (size_t) n_ent * sizeof(int32_t), cudaMemcpyDeviceToHost),
                "stage routing through host", err))
            return false;
    }

    On on(m.device);
    // ---- the window onto the helper: the primary's activations and routing.  On the link route the two peer copies
    // are the whole story; on the host route the pinned staging above was just filled, and only the upload is left.
    if (m.p2p) {
        if (!ck(cudaMemcpyPeerAsync(m.d_x, m.device, h_primary, m.primary,
                                    (size_t) n_tok * (size_t) m.n_embd * sizeof(float), m.stream), "stage activations",
                    err) ||
            !ck(cudaMemcpyPeerAsync(m.d_ids, m.device, ids_primary, m.primary, (size_t) n_ent * sizeof(int32_t),
                                    m.stream), "stage routing", err))
            return false;
    } else if (m.n_pos != 0) {
        // Skipped for a layer this helper holds nothing of: n_owned can only be 0 there, so launch() returns before any
        // kernel reads d_x - and an upload nobody waits for would still be queued on the helper's stream when the NEXT
        // layer's blocking D2H refills h_x, which arrives before that stream is ever synchronised.  Leaving it out is
        // cheaper than ordering around it.
        if (!ck(cudaMemcpyAsync(m.d_x, m.h_x, (size_t) n_tok * (size_t) m.n_embd * sizeof(float),
                                cudaMemcpyHostToDevice, m.stream), "upload activations", err) ||
            !ck(cudaMemcpyAsync(m.d_ids, m.h_ids, (size_t) n_ent * sizeof(int32_t), cudaMemcpyHostToDevice, m.stream),
                "upload routing", err))
            return false;
    }
    if (m.n_pos == 0) {   // nothing resident for this layer: no entry can be owned, and no kernel has work to do
        m.prepared = true;
        return true;
    }
    if (!ck(cudaMemcpyAsync(m.d_pos, m.pos_host.data(), (size_t) ne * sizeof(int32_t), cudaMemcpyHostToDevice, m.stream),
            "upload expert order", err) ||
        !ck(cudaMemcpyAsync(m.d_owned, m.owned_host.data(), (size_t) ne, cudaMemcpyHostToDevice, m.stream),
            "upload residency predicate", err) ||
        !ck(cudaMemsetAsync(m.d_cnt, 0, (size_t) (ne + 1) * sizeof(int32_t), m.stream), "clear expert counts", err))
        return false;
    k_peer_rank<<<1, kThreads, 0, m.stream>>>(m.d_ids, m.d_owned, n_ent, m.d_rank);
    k_peer_count<<<grid1(n_ent), kThreads, 0, m.stream>>>(m.d_ids, m.d_owned, m.d_pos, n_ent, m.d_cnt);
    k_peer_scan<<<1, 1, 0, m.stream>>>(m.d_cnt, (int) m.n_pos, m.d_bounds, m.d_cur, m.d_maxr);
    k_peer_scatter<<<grid1(n_ent), kThreads, 0, m.stream>>>(m.d_ids, m.d_owned, m.d_pos, m.d_rank, n_ent, (int) m.k,
                                                            m.d_cur, m.d_src, m.d_map);
    if (!ck(cudaGetLastError(), "row plan kernels", err)) return false;

    // THE ONE NUMBER THE HOST NEEDS.  MMQ's launches are sized on the host (rows, the launch grid), so the row count
    // and the widest expert come back - 8 bytes, after work that is one window copy (peer, or staged through host)
    // and four small kernels.  Doing it here rather than in launch() is what keeps the hot call free of host
    // synchronisation.
    int32_t h_owned = 0, h_max = 0;
    if (!ck(cudaMemcpyAsync(&h_owned, m.d_bounds + m.n_pos, sizeof(int32_t), cudaMemcpyDeviceToHost, m.stream),
            "read owned rows", err) ||
        !ck(cudaMemcpyAsync(&h_max, m.d_maxr, sizeof(int32_t), cudaMemcpyDeviceToHost, m.stream), "read widest expert",
            err) ||
        !ck(cudaStreamSynchronize(m.stream), "row plan", err))
        return false;
    if (h_owned < 0 || h_owned > n_ent || h_max < 0) {
        err = "q36 peer prefill: the row plan produced an impossible entry count";
        return false;
    }
    m.n_owned = h_owned;
    m.max_rows = h_max;
    m.prepared = true;
    return true;
}

bool PeerPrefill36::launch(int64_t layer, float* eparts_primary, int64_t eparts_row_base, std::string& err) {
    Impl& m = *impl_;
    if (m.device < 0 || m.stream == nullptr) { err = "q36 peer prefill: launch() on a closed executor"; return false; }
    if (!m.prepared || layer != m.prep_layer) {
        err = "q36 peer prefill: launch() without a prepare() of this layer";
        return false;
    }
    if (eparts_primary == nullptr && m.n_owned > 0) { err = "q36 peer prefill: null eparts"; return false; }
    ++m.launches;
    m.rows = m.n_owned;
    if (m.n_owned == 0) return true;   // no entry of this window is this helper's: nothing to compute, nothing to send

    On on(m.device);
    const auto& L = m.layouts[(size_t) layer];
    const int64_t n = m.n_owned;

    // 1. the activations to q8_1, one row per owned entry, gathered in the plan's expert order (row r reads its
    //    entry's token).  One quantization per entry: two entries of the same token are quantized twice, which is
    //    what the primary's MMQ branch does too.  `yscale` is only read by NVFP4 in w4a4 mode (MMQ::fp4_activations),
    //    where the product needs the same buffer as Product::y_scale; everywhere else MMQ ignores it and null is what
    //    the primary passes.
    float* const ys_gu = MMQ::fp4_activations(L.gu_type) ? m.d_ys : nullptr;
    float* const ys_d = MMQ::fp4_activations(L.d_type) ? m.d_ys : nullptr;
    // DIAG: the helper's own inputs and outputs, so a NaN can be told apart from a bad upload.  Remove after the fix.
    if (std::getenv("STRATA_Q36_PREFILL_DIAG") != nullptr) {
        cudaStreamSynchronize(m.stream);
        float x4[4] = {}, s4[4] = {};
        int32_t src4[4] = {};
        cudaMemcpy(x4, m.d_x, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(s4, m.d_x + (size_t) m.n_embd, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(src4, m.d_src, 4 * sizeof(int32_t), cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "DIAG helper dev=%d layer %lld n=%lld d_x[0..3]=%g %g %g %g d_x[1][0..3]=%g %g %g %g "
                             "d_src[0..3]=%d %d %d %d\n",
                     m.device, (long long) layer, (long long) n, x4[0], x4[1], x4[2], x4[3], s4[0], s4[1], s4[2],
                     s4[3], src4[0], src4[1], src4[2], src4[3]);
    }
    MMQ::quantize(m.d_x, m.d_src, m.d_xq, L.gu_type, m.n_embd, m.n_embd, n, m.stream, ys_gu);
    // DIAG: the quantized activation and the weight rows the products read.  A NaN in d_xq means the gather or the
    // quantizer is wrong; a clean d_xq with a NaN d_out means the weight rows or the product are.  Remove after the fix.
    if (std::getenv("STRATA_Q36_PREFILL_DIAG") != nullptr) {
        cudaStreamSynchronize(m.stream);
        float xq4[4] = {};
        unsigned char q4[4] = {};
        cudaMemcpy(xq4, m.d_xq, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(q4, m.d_xq, 4, cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "DIAG helper dev=%d layer %lld d_xq as float[0..3]=%g %g %g %g as bytes=%d %d %d %d "
                             "gu_type=%d d_type=%d groups=%zu\n",
                     m.device, (long long) layer, xq4[0], xq4[1], xq4[2], xq4[3], q4[0], q4[1], q4[2], q4[3], L.gu_type,
                     L.d_type, m.groups.size());
        // the weight rows the products read: the first group's address, its step between experts, and the bytes at
        // that address.  A wrong step (the address difference instead of the expert size) is the classic cause.
        for (size_t gi = 0; gi < m.groups.size() && gi < 4; ++gi) {
            const Impl::Group& g0 = m.groups[gi];
            unsigned char wb[8] = {};
            cudaMemcpy(wb, g0.w, 8, cudaMemcpyDeviceToHost);
            std::fprintf(stderr, "DIAG helper dev=%d layer %lld group%zu w=%p delta=%llu n=%d bytes=%d %d %d %d\n",
                         m.device, (long long) layer, gi, (const void*) g0.w, (unsigned long long) g0.delta, g0.n,
                         wb[0], wb[1], wb[2], wb[3]);
        }
        std::fprintf(stderr, "DIAG helper dev=%d layer %lld n_pos=%lld ptrs=%zu n_owned=%lld expert_bytes=%llu\n",
                     m.device, (long long) layer, (long long) m.n_pos, m.ptrs.size(), (long long) m.n_owned,
                     (unsigned long long) m.layouts[(size_t) layer].bytes);
        // the row plan the products read: the bounds (each expert's row interval), the output row map, and how
        // many of the run's first rows are NaN.  Remove after the fix.
        {
            int32_t bd[4] = {}, mp[8] = {}, id4[4] = {};
            cudaMemcpy(bd, m.d_bounds, 4 * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(bd + 2, m.d_bounds + (m.n_pos - 1), 2 * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(mp, m.d_map, 8 * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(id4, m.d_ident, 4 * sizeof(int32_t), cudaMemcpyDeviceToHost);
            std::fprintf(stderr, "DIAG helper dev=%d layer %lld bounds[0..1]=%d %d bounds[n_pos-1..n_pos]=%d %d "
                                 "map[0..7]=%d %d %d %d %d %d %d %d ident[0..3]=%d %d %d %d\n",
                         m.device, (long long) layer, bd[0], bd[1], bd[2], bd[3], mp[0], mp[1], mp[2], mp[3], mp[4],
                         mp[5], mp[6], mp[7], id4[0], id4[1], id4[2], id4[3]);
        }
    }

    // 2. gate+up as one [2 n_ff] product per arithmetic run of experts, the weights read in place from the cache.
    //    Output rows stay in the PLAN's expert order (the identity table), because that is the index space `bounds`
    //    speaks: SwiGLU and the down product's activation rows are read through those bounds.  The run's own order
    //    is imposed one stage later, by the down product's `ids`.
    for (const Impl::Group& g : m.groups) {
        MMQ::Product gu;
        gu.w = g.w;
        gu.type = L.gu_type;
        gu.w_rows = 2 * m.n_ff;
        gu.w_cols = m.n_embd;
        gu.expert_bytes = g.delta;
        gu.n = g.n;
        gu.xq = m.d_xq;
        gu.bounds = m.d_bounds + g.first;   // the run is a contiguous slice of the plan's expert order
        gu.ids = m.d_ident;  // identity: gu row r IS activation row r (null would make MMQ ignore `bounds`)
        gu.total_rows = n;
        gu.max_rows = m.max_rows;
        gu.dst = m.d_gu;
        gu.ld_dst = 2 * m.n_ff;
        gu.y_scale = ys_gu;
        m.mctx->run(gu, m.stream);
    }
    // DIAG: how much of d_gu the gate/up product actually wrote.  If only the first few floats are non-zero, the
    // product's output layout does not match what swiglu() reads back.  Remove after the fix.
    if (std::getenv("STRATA_Q36_PREFILL_DIAG") != nullptr) {
        cudaStreamSynchronize(m.stream);
        std::vector<float> g2(1024);
        cudaMemcpy(g2.data(), m.d_gu, g2.size() * sizeof(float), cudaMemcpyDeviceToHost);
        int64_t nz = 0, nan_cnt = 0;
        int first_nz = -1, first_nan = -1;
        for (size_t i = 0; i < g2.size(); ++i) {
            if (g2[i] != g2[i]) { ++nan_cnt; if (first_nan < 0) first_nan = (int) i; }
            else if (g2[i] != 0.f) { ++nz; if (first_nz < 0) first_nz = (int) i; }
        }
        std::fprintf(stderr, "DIAG helper dev=%d layer %lld d_gu[0..1023]: nonzero=%lld first_nz=%d nan=%lld first_nan=%d\n",
                     m.device, (long long) layer, (long long) nz, first_nz, (long long) nan_cnt, first_nan);
    }
    MMQ::swiglu(m.d_gu, m.d_h, n, m.n_ff, false, m.stream);
    MMQ::quantize(m.d_h, nullptr, m.d_hq, L.d_type, m.n_ff, m.n_ff, n, m.stream, ys_d);
    // DIAG: the gate/up output, the SwiGLU output and its quantized image: which of the three stages first holds a
    // NaN.  A NaN in d_gu means the gate/up product (weights or bounds); in d_h means the SwiGLU; in d_hq means the
    // down product's input.  Remove after the fix.
    if (std::getenv("STRATA_Q36_PREFILL_DIAG") != nullptr) {
        cudaStreamSynchronize(m.stream);
        const int64_t probe = n < 64 ? n : 64;
        std::vector<float> gu((size_t) probe * m.n_ff), h((size_t) probe * m.n_ff);
        cudaMemcpy(gu.data(), m.d_gu, gu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h.data(), m.d_h, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
        int64_t nan_gu = 0, nan_h = 0;
        for (size_t i = 0; i < gu.size(); ++i) if (!(gu[i] == gu[i])) ++nan_gu;
        for (size_t i = 0; i < h.size(); ++i) if (!(h[i] == h[i])) ++nan_h;
        std::fprintf(stderr, "DIAG helper dev=%d layer %lld stage gu nan=%lld/%zu h nan=%lld/%zu gu[0..3]=%g %g %g %g "
                             "h[0..3]=%g %g %g %g\n",
                     m.device, (long long) layer, (long long) nan_gu, gu.size(), (long long) nan_h, h.size(), gu[0],
                     gu[1], gu[2], gu[3], h[0], h[1], h[2], h[3]);
    }

    // 3. down, from the same run's blobs at the layout's own down offset.  Here `ids` is NOT the identity: its row
    //    is the entry's RANK in this helper's run, so the product writes the final rows in the order the header's
    //    row contract demands, whatever order the experts were multiplied in.
    for (const Impl::Group& g : m.groups) {
        MMQ::Product dn;
        dn.w = g.w + L.down_off;
        dn.type = L.d_type;
        dn.w_rows = m.n_embd;
        dn.w_cols = m.n_ff;
        dn.expert_bytes = g.delta;
        dn.n = g.n;
        dn.xq = m.d_hq;
        dn.bounds = m.d_bounds + g.first;
        dn.ids = m.d_map;
        dn.total_rows = n;
        dn.max_rows = m.max_rows;
        dn.dst = m.d_out;
        dn.ld_dst = m.n_embd;
        dn.y_scale = ys_d;
        m.mctx->run(dn, m.stream);
    }

    // 4. the run, back to the primary.  Rank r's row lands in eparts_row_base + r, which is the run the header's row
    //    contract gives this helper.
    //
    //    Link: ONE peer copy on the helper's stream; the caller orders the primary against it with wait().
    //    No link: one device-to-host copy of the same rows into the pinned h_out, on the same helper stream, and the
    //    PRIMARY-side upload is left for wait() - the only method that is handed the primary's stream.  What it must
    //    write is remembered rather than passed along, because launch() has no primary stream to issue it on.
    if (m.p2p) {
        if (!ck(cudaMemcpyPeerAsync(eparts_primary + eparts_row_base * m.n_embd, m.primary, m.d_out, m.device,
                                    (size_t) n * (size_t) m.n_embd * sizeof(float), m.stream), "copy the run", err))
            return false;
    } else {
        // DIAG: the helper's result rows, before they go back to the primary.  Remove after the fix.
        if (std::getenv("STRATA_Q36_PREFILL_DIAG") != nullptr) {
            cudaStreamSynchronize(m.stream);
            float o4[4] = {};
            cudaMemcpy(o4, m.d_out, 4 * sizeof(float), cudaMemcpyDeviceToHost);
            std::fprintf(stderr, "DIAG helper dev=%d layer %lld d_out[0..3]=%g %g %g %g\n", m.device, (long long) layer,
                         o4[0], o4[1], o4[2], o4[3]);
            // how many of the run's rows are NaN, and the first good one: a partial NaN names a row range, an
            // all-NaN run names the arithmetic.
            const int64_t probe = n < 64 ? n : 64;
            std::vector<float> rows((size_t) probe * m.n_embd);
            cudaMemcpy(rows.data(), m.d_out, rows.size() * sizeof(float), cudaMemcpyDeviceToHost);
            int64_t nnan = 0, first_good = -1;
            for (int64_t r = 0; r < probe; ++r) {
                bool bad = false;
                for (int64_t c = 0; c < m.n_embd; ++c)
                    if (!(rows[(size_t) r * m.n_embd + c] == rows[(size_t) r * m.n_embd + c])) { bad = true; break; }
                if (bad) ++nnan;
                else if (first_good < 0) first_good = r;
            }
            std::fprintf(stderr, "DIAG helper dev=%d layer %lld rows[0..%lld): nan=%lld first_good=%lld\n", m.device,
                         (long long) layer, (long long) probe, (long long) nnan, (long long) first_good);
        }
        if (!ck(cudaMemcpyAsync(m.h_out, m.d_out, (size_t) n * (size_t) m.n_embd * sizeof(float),
                                cudaMemcpyDeviceToHost, m.stream), "copy the run to host staging", err))
            return false;
        m.pending_eparts = eparts_primary;
        m.pending_base = eparts_row_base;
        m.pending_n = n;
    }
    return ck(cudaGetLastError(), "helper launches", err);
}

bool PeerPrefill36::wait(cudaStream_t st_primary, std::string& err) {
    Impl& m = *impl_;
    if (m.device < 0 || m.stream == nullptr) { err = "q36 peer prefill: wait() on a closed executor"; return false; }
    if (st_primary == nullptr) { err = "q36 peer prefill: wait() without a primary stream"; return false; }
    On on(m.device);
    if (!ck(cudaEventRecord(m.ev, m.stream), "record helper completion", err) ||
        !ck(cudaStreamWaitEvent(st_primary, m.ev, 0), "make the primary wait", err))
        return false;
    // THE HOST ROUTE'S UP LEG.  launch() already left this helper's rows in the pinned h_out; here they go to where
    // the header's row contract says they belong, on the PRIMARY's stream.  What orders it, without a new
    // synchronisation: the primary's H2D above waits for `ev`, and `ev` is recorded on the helper's stream AFTER that
    // stream's device-to-host into h_out, so h_out is complete before the H2D reads it.  What orders the NEXT layer's
    // write into h_out (its launch(), on the same helper stream) after this read is the caller: it synchronises the
    // primary's stream before every plan (the same sync prepare()'s blocking D2H already stands on), which retires
    // this H2D on the device before the next launch() can be issued.  pending_n is cleared so a stray second wait()
    // posts no copy.
    if (!m.p2p && m.pending_n > 0) {
        const int64_t n = m.pending_n;
        m.pending_n = 0;
        On onp(m.primary);   // `eparts` and the stream are the primary's: the copy runs on its device
        if (!ck(cudaMemcpyAsync(m.pending_eparts + m.pending_base * m.n_embd, m.h_out,
                                (size_t) n * (size_t) m.n_embd * sizeof(float), cudaMemcpyHostToDevice, st_primary),
                "copy the run into eparts", err))
            return false;
    }
    return ck(cudaGetLastError(), "helper completion", err);
}

int64_t PeerPrefill36::rows() const { return impl_->rows; }
int64_t PeerPrefill36::launches() const { return impl_->launches; }

}  // namespace strata::qwen35
