// sycl/src/kernels/cuda/allreduce.dp.cpp - the cross-GPU sum tensor parallelism needs, on SYCL/Level Zero.
//
// The CUDA original (src/kernels/cuda/allreduce.cu) stages each rank's half through mapped pinned host
// memory, meets on a device-side arrival token and adds the peer's half in place.  This port keeps the
// protocol and extends it to up to four ranks:
//
//   * TWO RANKS: each rank stages its half, signals, waits for the peer and adds the peer's half in place.
//   * THREE OR FOUR RANKS: every rank stages its partial and signals; rank 0 waits for every arrival, sums
//     the staged partials into its own staging region, and publishes a done token; every rank then reads
//     rank 0's sum.  The result is bit-identical on every rank, which is the requirement: the router must
//     pick the same expert ids on every rank or the ranks run different models.
//
// WHAT CHANGED FOR SYCL, and why:
//
//   * Host memory is `strata::host_malloc_polled` (zeMemAllocHost with the uncached bias on Level Zero).
//     A plain host USM allocation is served from the GPU's cache on Arc (xe), so a peer's staging bytes and
//     the arrival token would not be visible until the kernel ends - the same finding the doorbell records
//     (include/strata/sycl_doorbell.hpp).
//   * The payload stores and reads go through the doorbell's uncached annotated_ptr hints (`sys_store_mapped`
//     / `sys_load_mapped`), for the same reason.
//   * The arrival/done/timeout/progress words use `sys_store` / `sys_load` (uncached, system scope).
//   * There is no `cudaHostGetDevicePointer`: USM host memory has one address for the host and every device.
//   * `flush` submits a trailing barrier on the rank's queue.
#include "strata/kernels/allreduce.hpp"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include "strata/sycl_queue.hpp"
#include "strata/sycl_doorbell.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

namespace strata::kernels {
namespace {

/// One arrival slot per block, 64 B apart: two cards polling one cache line is the false-sharing stall the
/// upstream comment calls out.  The stride is a compile-time constant because the KERNEL and the host both
/// index these slots.
constexpr int kArrivalInts = 16;                       // 64 B / sizeof(uint32_t)
constexpr size_t kSlotBytes = kArrivalInts * sizeof(uint32_t);

/// The give-up bound.  A waiter spins with a bounded loop; a genuinely stuck window is caught by the engine's
/// own watchdog.  4M turns is ~200 ms at the pessimistic microsecond per turn.
constexpr uint64_t kGiveUpSpins = 4000000ull;
constexpr uint32_t kTimedOut = 0xDEADBEEFu;

/// bf16 as this tree stores it: the high 16 bits of an f32 (include/strata/kernels/gr.hpp:10).  Round to
/// nearest even, the same rule the CUDA kernel uses; every rank must agree bit for bit.
inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
inline float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

/// Bitwise equality of two wire values (a float compare would never match a NaN).
inline bool ar_same_bits(float a, float b) { return sycl::bit_cast<uint32_t>(a) == sycl::bit_cast<uint32_t>(b); }
inline bool ar_same_bits(uint16_t a, uint16_t b) { return a == b; }

/// EIGHT BYTES PER PCIe TRANSACTION.  The staging used to be stored and loaded one wire value at a time (2 B for
/// bf16), and the time of a reduce was the number of transactions: 4 ranks, 15,360 floats, 2.0 ms a call on
/// 4x Arc Pro B60.  A word holds 4 bf16 or 2 f32 in the same byte order the single stores wrote, so the staging
/// bytes - and every rank's result - are unchanged.
inline uint64_t ar_bits(uint16_t v) { return v; }
inline uint64_t ar_bits(float v) { return sycl::bit_cast<uint32_t>(v); }
inline void ar_unbits(uint64_t b, uint16_t& v) { v = (uint16_t) b; }
inline void ar_unbits(uint64_t b, float& v) { v = sycl::bit_cast<float>((uint32_t) b); }
template <typename T> inline uint64_t ar_pack(const T* v) {
    uint64_t w = 0;
    for (int k = 0; k < (int) (8 / sizeof(T)); ++k) w |= ar_bits(v[k]) << (k * 8 * (int) sizeof(T));
    return w;
}
template <typename T> inline void ar_unpack(uint64_t w, T* v) {
    for (int k = 0; k < (int) (8 / sizeof(T)); ++k) ar_unbits(w >> (k * 8 * (int) sizeof(T)), v[k]);
}

template <typename Wire> struct WireOps;
template <> struct WireOps<float> {
    using type = float;
    static constexpr int elems = 4;                                       // 16 B per store
    static type from(float v) { return v; }
    static float to(type v) { return v; }
};
template <> struct WireOps<uint16_t> {
    using type = uint16_t;
    static constexpr int elems = 8;                                       // 16 B per store
    static type from(float v) { return f32_to_bf16(v); }
    static float to(type v) { return bf16_to_f32(v); }
};

inline size_t ar_region_bytes(int64_t max_floats, ArWire w) {
    const size_t raw = (size_t) max_floats * (w == ArWire::F32 ? sizeof(float) : sizeof(uint16_t));
    return (raw + 15u) & ~(size_t) 15u;     // the staging stores are 16 B
}

/// Where each piece sits inside one rank's mapped-host allocation.  One place computes it, so the host and
/// device pointer arithmetic cannot drift apart.
struct ArLayout {
    size_t region = 0;
    size_t arrival = 0;
    size_t done = 0;
    size_t timeout = 0;
    size_t progress = 0;
    size_t total = 0;
};
inline ArLayout ar_layout(size_t region_bytes, int blocks, int ranks) {
    ArLayout L;
    L.arrival = 2 * region_bytes;
    L.done = L.arrival + (size_t) ranks * 2 * (size_t) blocks * kSlotBytes;
    L.timeout = L.done + 2 * (size_t) blocks * kSlotBytes;
    L.progress = L.timeout + kSlotBytes;
    L.total = L.progress + kSlotBytes;
    return L;
}

/// One reduce.  Every block stages its own stripe of `x` into this rank's half of the staging region,
/// signals its arrival token, then meets the peers:
///
///   rank 0: waits for every rank's arrival, sums the staged partials into its own staging region, publishes
///           the done token, and writes the sum into `x`.
///   rank > 0: waits for rank 0's done token and reads rank 0's staging region into `x`.
///
/// Every operand rounds through the wire type before the add, so every rank's result is bit-identical.
template <typename Wire>
void ar_reduce_kernel(sycl::queue& q, float* x, int rank, int ranks, typename WireOps<Wire>::type* mine0,
                      typename WireOps<Wire>::type* mine1, const typename WireOps<Wire>::type* const* peers0,
                      const typename WireOps<Wire>::type* const* peers1, const uint32_t* const* peer_timeout,
                      int count, uint32_t* my_arrival, const uint32_t* const* peer_arrival, uint32_t* done,
                      const uint32_t* peer_done, uint32_t* token, uint32_t* timed_out, uint32_t* progress,
                      int blocks, int threads, uint32_t spin_bound) {
    using Ops = WireOps<Wire>;
    using T = typename Ops::type;
    constexpr int ELEMS = (int) (8 / sizeof(T));   // one 8-byte word per staging transaction
    const int nvec = count / ELEMS, tail = nvec * ELEMS;

    // THE KERNEL TAKES THESE TABLES BY VALUE.  The parameters point at the caller's stack arrays; a lambda
    // that captured the POINTERS would dereference host stack memory on the device.  On Arc (xe) that does
    // not fault: the driver mirrors the CPU address, migrates the page to VRAM, and the kernel crashed when
    // the host touched the page again (general protection fault in drm_pagemap_migrate_to_ram, 4x Arc Pro
    // B60, Linux 7.0.0-42).  In a captured graph the stack is gone by the replay as well.
    std::array<const T*, kArRanks> peers0_v{};
    std::array<const uint32_t*, kArRanks> peer_timeout_v{};
    std::array<const uint32_t*, kArRanks> peer_arrival_v{};
    for (int j = 0; j < ranks && j < kArRanks; ++j) {
        peers0_v[j] = peers0[j];
        peer_timeout_v[j] = peer_timeout[j];
        peer_arrival_v[j] = peer_arrival[j];
    }
    (void) peers1;

    q.parallel_for(sycl::nd_range<1>((size_t) blocks * (size_t) threads, (size_t) threads), [=](sycl::nd_item<1> it) {
        const int bid = (int) it.get_group(0);
        const int tid = (int) it.get_local_id(0);
        const int gtid = bid * threads + tid, gnt = blocks * threads;
        if (bid == 0 && tid == 0) strata::sys_store(progress + kArProgressEntered, 1u);

        // The staging half is chosen by the HOST (`calls_[r] & 1`, one value per rank); every rank uses the
        // same parity for the same logical call.  `mine1` is unused by this call (the host passes the other
        // parity so the two calls never share bytes).
        T* const mine_h = mine0;
        const T* const peer0 = peers0_v[rank];

        // THE LAST STORE OF EVERY THREAD IS READ BACK BEFORE THE BARRIER.  The arrival token is thread 0's
        // store; the payload is every thread's.  Without the read-back the token reached the peer before
        // some threads' payload did, and the peer summed stale bytes (measured on 2 and 4 Arc Pro B60:
        // the standalone test failed 4-6 runs of 6, mostly on the first call, with either spin bound).
        const T* last_p = nullptr;
        T last_v{};
        const uint64_t* last_wp = nullptr;
        uint64_t last_wv = 0;
        for (int i = gtid; i < nvec; i += gnt) {
            const int off = i * ELEMS;
            T v[ELEMS];
            for (int k = 0; k < ELEMS; ++k) v[k] = Ops::from(x[off + k]);
            last_wv = ar_pack<T>(v);
            last_wp = (const uint64_t*) (mine_h + off);
            strata::sys_store_mapped((uint64_t*) (mine_h + off), last_wv);
        }
        if (bid == 0 && tid < count - tail) {
            last_v = Ops::from(x[tail + tid]);
            last_p = mine_h + tail + tid;
            strata::sys_store_mapped(mine_h + tail + tid, last_v);
        }
        sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        if (last_wp != nullptr)
            for (uint32_t s = 0; s < spin_bound && strata::sys_load_mapped(last_wp) != last_wv; ++s) {}
        if (last_p != nullptr)
            for (uint32_t s = 0; s < spin_bound && !ar_same_bits(strata::sys_load_mapped(last_p), last_v); ++s) {}

        sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group());

        if (bid == 0 && tid == 0) strata::sys_store(progress + kArProgressStaged, 1u);

        if (tid == 0) {
            // ONE BUMP PER EXECUTION, PER BLOCK.  Each block owns a counter; a device atomic, not a host value:
            // a captured graph would otherwise replay the same token forever.
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> tok_ref(token[bid]);
            const uint32_t tok = tok_ref.fetch_add(1u) + 1u;
            strata::sys_store(my_arrival + (size_t) bid * kArrivalInts, tok);
            if (bid == 0) strata::sys_store(progress + kArProgressSignalled, tok);
            sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);

            if (ranks == 2) {
                // TWO RANKS: THE ORIGINAL PEER-ADD.  Each rank stages its half, waits for the other's arrival
                // and adds the peer's half in place.  No done token, no rank-0 sum.
                const int other = 1 - rank;
                uint32_t spins = 0;
                while (strata::sys_load(peer_arrival_v[other] + (size_t) bid * kArrivalInts) < tok) {
                    if (peer_timeout_v[other] != nullptr && strata::sys_load(peer_timeout_v[other]) == kTimedOut) {
                        strata::sys_store(timed_out, kTimedOut);
                        break;
                    }
                    if (++spins >= spin_bound) { strata::sys_store(timed_out, kTimedOut); break; }
                }
                if (bid == 0) strata::sys_store(progress + kArProgressSawPeer,
                                                strata::sys_load(peer_arrival_v[other]));
            } else if (rank == 0) {
                // WAIT FOR EVERY RANK'S ARRIVAL, then sum the staged partials into this rank's staging.
                uint32_t spins = 0;
                for (int r = 0; r < ranks; ++r) {
                    while (strata::sys_load(peer_arrival_v[r] + (size_t) bid * kArrivalInts) < tok) {
                        bool give_up = false;
                        for (int q2 = 0; q2 < ranks; ++q2)
                            if (peer_timeout_v[q2] != nullptr && strata::sys_load(peer_timeout_v[q2]) == kTimedOut)
                                give_up = true;
                        if (give_up) { strata::sys_store(timed_out, kTimedOut); break; }
                        if (++spins >= spin_bound) { strata::sys_store(timed_out, kTimedOut); break; }
                    }
                    if (strata::sys_load(timed_out) == kTimedOut) break;
                }
                if (bid == 0) strata::sys_store(progress + kArProgressSawPeer,
                                                strata::sys_load(peer_arrival_v[ranks - 1]));
            } else {
                // WAIT FOR RANK 0'S SUM.
                uint32_t spins = 0;
                while (strata::sys_load(peer_done + (size_t) bid * kArrivalInts) < tok) {
                    bool give_up = false;
                    for (int q2 = 0; q2 < ranks; ++q2)
                        if (peer_timeout_v[q2] != nullptr && strata::sys_load(peer_timeout_v[q2]) == kTimedOut)
                            give_up = true;
                    if (give_up) { strata::sys_store(timed_out, kTimedOut); break; }
                    if (++spins >= spin_bound) { strata::sys_store(timed_out, kTimedOut); break; }
                }
                if (bid == 0) strata::sys_store(progress + kArProgressSawPeer,
                                                strata::sys_load(peer_done));
            }
        }
        sycl::group_barrier(it.get_group());
        sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);   // acquire the sum

        if (ranks > 2 && rank == 0) {
            // RANK 0 SUMS, WITH EVERY THREAD.  The wait above is thread 0's alone, but the sum is striped
            // over the whole block like the staging: a sum done by thread 0 only covers one stripe in
            // `threads` and leaves the rest of rank 0's region holding rank 0's partial (measured: TP=4
            // generated different tokens from the single card from the first token on).
            const bool ok = strata::sys_load(timed_out) != kTimedOut;
            if (ok) {
                // this rank's stripe plus every peer's, accumulated in float, stored as wire
                const T* sum_p = nullptr;
                T sum_v{};
                const uint64_t* sum_wp = nullptr;
                uint64_t sum_wv = 0;
                for (int i = gtid; i < nvec; i += gnt) {
                    const int off = i * ELEMS;
                    T v[ELEMS];
                    float acc[ELEMS];
                    ar_unpack<T>(strata::sys_load_mapped((const uint64_t*) (mine_h + off)), v);
                    for (int k = 0; k < ELEMS; ++k) acc[k] = Ops::to(v[k]);
                    for (int r = 1; r < ranks; ++r) {   // the same order of adds as one value at a time
                        ar_unpack<T>(strata::sys_load_mapped((const uint64_t*) (peers0_v[r] + off)), v);
                        for (int k = 0; k < ELEMS; ++k) acc[k] += Ops::to(v[k]);
                    }
                    for (int k = 0; k < ELEMS; ++k) v[k] = Ops::from(acc[k]);
                    sum_wv = ar_pack<T>(v);
                    sum_wp = (const uint64_t*) (mine_h + off);
                    strata::sys_store_mapped((uint64_t*) (mine_h + off), sum_wv);
                }
                if (bid == 0 && tid < count - tail) {
                    float acc = Ops::to(strata::sys_load_mapped(mine_h + tail + tid));
                    for (int r = 1; r < ranks; ++r)
                        acc += Ops::to(strata::sys_load_mapped(peers0_v[r] + tail + tid));
                    sum_v = Ops::from(acc);
                    sum_p = mine_h + tail + tid;
                    strata::sys_store_mapped(mine_h + tail + tid, sum_v);
                }
                // the same read-back as the staging: the done token must not overtake the sum
                sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
                if (sum_wp != nullptr)
                    for (uint32_t s = 0; s < spin_bound && strata::sys_load_mapped(sum_wp) != sum_wv; ++s) {}
                if (sum_p != nullptr)
                    for (uint32_t s = 0; s < spin_bound && !ar_same_bits(strata::sys_load_mapped(sum_p), sum_v); ++s) {}
            }
            sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
            sycl::group_barrier(it.get_group());
            // the done token is this block's arrival token: published once the whole block has summed
            if (ok && tid == 0)
                strata::sys_store(done + (size_t) bid * kArrivalInts,
                                  strata::sys_load(my_arrival + (size_t) bid * kArrivalInts));
            sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        }

        if (ranks == 2) {
            // add the peer's staged half to x, in place
            const int other = 1 - rank;
            for (int i = gtid; i < nvec; i += gnt) {
                const int off = i * ELEMS;
                T v[ELEMS];
                ar_unpack<T>(strata::sys_load_mapped((const uint64_t*) (peers0_v[other] + off)), v);
                for (int k = 0; k < ELEMS; ++k) x[off + k] = Ops::to(Ops::from(x[off + k])) + Ops::to(v[k]);
            }
            if (bid == 0 && tid < count - tail)
                x[tail + tid] = Ops::to(Ops::from(x[tail + tid])) +
                                Ops::to(strata::sys_load_mapped(peers0_v[other] + tail + tid));
        } else {
            // Rank 0's staging holds the sum; every rank reads it.  Rank 0 reads its own region (the sum it
            // just wrote); ranks > 0 read RANK 0's region, which is `peers0_v[0]` - not their own staging.
            const T* const sum_h = (rank == 0) ? mine_h : peers0_v[0];
            for (int i = gtid; i < nvec; i += gnt) {
                const int off = i * ELEMS;
                T v[ELEMS];
                ar_unpack<T>(strata::sys_load_mapped((const uint64_t*) (sum_h + off)), v);
                for (int k = 0; k < ELEMS; ++k) x[off + k] = Ops::to(v[k]);
            }
            if (bid == 0 && tid < count - tail) x[tail + tid] = Ops::to(strata::sys_load_mapped(sum_h + tail + tid));
        }
    });
}

}  // namespace

ArPair::~ArPair() { shutdown(); }

bool ArPair::init(const int* ordinals, int ranks, int64_t max_floats, ArWire wire, int blocks, std::string& err) {
    if (ready_) { err = "allreduce: the pair is already set up"; return false; }
    if (ordinals == nullptr) { err = "allreduce: no devices given"; return false; }
    if (ranks < 2 || ranks > kArRanks) {
        err = "allreduce: " + std::to_string(ranks) + " ranks is outside 2.." + std::to_string(kArRanks);
        return false;
    }
    if (max_floats < 1) {
        err = "allreduce: max_floats must be positive (got " + std::to_string(max_floats) + ")";
        return false;
    }
    if (blocks < 1 || blocks > 64) {
        err = "allreduce: blocks must be 1..64 (got " + std::to_string(blocks) + ")";
        return false;
    }
    for (int r = 0; r < ranks; ++r)
        for (int q = r + 1; q < ranks; ++q)
            if (ordinals[r] == ordinals[q]) {
                err = "allreduce: ranks " + std::to_string(r) + " and " + std::to_string(q) +
                      " are on device " + std::to_string(ordinals[r]) + " - this is not a tensor-parallel group";
                return false;
            }
    ranks_ = ranks;
    for (int r = 0; r < ranks; ++r) dev_[r] = ordinals[r];
    wire_ = wire;
    blocks_ = blocks;
    max_floats_ = max_floats;

    const size_t region_bytes = ar_region_bytes(max_floats, wire);
    const ArLayout L = ar_layout(region_bytes, blocks, ranks);

    for (int r = 0; r < ranks; ++r) {
        if (DPCT_CHECK_ERROR(dpct::select_device(dev_[r])) != 0) {
            err = "allreduce: select_device(" + std::to_string(dev_[r]) + ") failed";
            shutdown();
            return false;
        }
        sycl::queue& q = dpct::get_in_order_queue();
        // Mapped + portable in CUDA terms: one host allocation every card can see.  PLAIN host USM: the
        // uncached zeMemAllocHost allocation is tied to one context and a peer's kernel faults on it
        // (measured: UR_RESULT_ERROR_DEVICE_LOST).  The uncached access hints below make the bytes visible
        // mid-kernel on Arc (xe), and plain host USM is visible across contexts (measured).
        void* h = sycl::malloc_host(L.total, q);
        if (h == nullptr) {
            err = "allreduce: rank " + std::to_string(r) + ": " + std::to_string(L.total >> 20) +
                  " MiB of mapped host memory";
            shutdown();
            return false;
        }
        host_[r] = h;
        std::memset(h, 0, L.total);

        uint8_t* hb = (uint8_t*) h;
        view_[r] = ArRankView{};
        view_[r].dev = dev_[r];
        for (int p = 0; p < 2; ++p) {
            view_[r].mine[p] = hb + L.region + (size_t) p * region_bytes;
            view_[r].arrival[p] = (uint32_t*) (hb + L.arrival + (size_t) p * (size_t) blocks * kSlotBytes);
            h_arrival_[r][p] = (uint32_t*) (hb + L.arrival + (size_t) p * (size_t) blocks * kSlotBytes);
        }
        view_[r].timeout = (uint32_t*) (hb + L.timeout);
        h_timeout_[r] = (uint32_t*) (hb + L.timeout);
        view_[r].progress = (uint32_t*) (hb + L.progress);
        h_progress_[r] = (uint32_t*) (hb + L.progress);
        // the token counters are the one piece that is real device memory: the kernel bumps them.
        view_[r].token = sycl::malloc_device<uint32_t>((size_t) blocks, q);
        if (view_[r].token == nullptr) {
            err = "allreduce: rank " + std::to_string(r) + ": the token counters could not be allocated";
            shutdown();
            return false;
        }
        q.memset(view_[r].token, 0, (size_t) blocks * sizeof(uint32_t)).wait();
        pinned_bytes_ += L.total;
    }

    // Cross-wire the views.  Every rank can read every rank's staging and arrival slots; rank 0's done slots
    // are published to all ranks.  USM host memory has one address for every device.
    for (int r = 0; r < ranks; ++r) {
        for (int j = 0; j < ranks; ++j) {
            uint8_t* pb = (uint8_t*) host_[j];
            for (int p = 0; p < 2; ++p) {
                view_[r].peers[j][p] = pb + L.region + (size_t) p * region_bytes;
                view_[r].peer_arrival[j][p] =
                    (const uint32_t*) (pb + L.arrival + (size_t) p * (size_t) blocks * kSlotBytes);
            }
            view_[r].peer_timeout[j] = (const uint32_t*) (pb + L.timeout);
        }
        uint8_t* r0 = (uint8_t*) host_[0];
        for (int p = 0; p < 2; ++p) {
            view_[r].done[p] = (uint32_t*) (r0 + L.done + (size_t) p * (size_t) blocks * kSlotBytes);
            view_[r].peer_done[p] = (const uint32_t*) (r0 + L.done + (size_t) p * (size_t) blocks * kSlotBytes);
        }
    }
    for (int p = 0; p < 2; ++p) h_done_[p] = view_[0].done[p];

    // Arc has no peer access: the staged path is what runs.  Report it honestly.
    peer_mode_ = 0;

    for (int r = 0; r < ranks; ++r) calls_[r] = 0;
    ready_ = true;
    return true;
}

void ArPair::shutdown() {
    for (int r = 0; r < ranks_; ++r) {
        if (view_[r].token != nullptr) {
            (void) DPCT_CHECK_ERROR(dpct::select_device(dev_[r] < 0 ? 0 : dev_[r]));
            sycl::free(view_[r].token, dpct::get_in_order_queue());
        }
        if (host_[r] != nullptr) {
            (void) DPCT_CHECK_ERROR(dpct::select_device(dev_[r] < 0 ? 0 : dev_[r]));
            sycl::free(host_[r], dpct::get_in_order_queue());
        }
        host_[r] = nullptr;
        view_[r] = ArRankView{};
        h_arrival_[r][0] = h_arrival_[r][1] = nullptr;
        h_timeout_[r] = nullptr;
        h_progress_[r] = nullptr;
    }
    h_done_[0] = h_done_[1] = nullptr;
    pinned_bytes_ = 0;
    peer_mode_ = -1;
    ready_ = false;
    ranks_ = 2;
}

namespace {
const ArPair* g_ar_debug = nullptr;
/// The multi-device contexts the staging allocations live in.  A `sycl::context` must outlive the native
/// handle used by `zeMemAllocHost`, so the contexts are kept here for the process's life.
std::vector<std::unique_ptr<sycl::context>>& ar_contexts() {
    static std::vector<std::unique_ptr<sycl::context>> v;
    return v;
}
}

const void* ArPair::host_staging(int rank, int half) const {
    if (rank < 0 || rank >= ranks_ || host_[rank] == nullptr) return nullptr;
    const size_t rb = ar_region_bytes(max_floats_, wire_);
    return (const uint8_t*) host_[rank] + (size_t) (half ? 1 : 0) * rb;
}

void ar_debug_set_pair(ArPair* p) { g_ar_debug = p; }

const void* ar_debug_staging(int rank, int half) {
    return g_ar_debug == nullptr ? nullptr : g_ar_debug->host_staging(rank, half);
}

void ArPair::launch(int rank, float* x, int count, void* stream) {
    const int r = rank <= 0 ? 0 : (rank >= ranks_ ? ranks_ - 1 : rank);
    ++calls_[r];
    const ArRankView& v = view_[r];
    // The host call count selects the staging parity: every rank must use the same parity for the same
    // logical call, and alternating parities keep call k+1 from overwriting the staging call k is reading.
    const int parity = (int) (calls_[r] & 1ull);
    sycl::queue* q = strata::q_of(stream);
    if (wire_ == ArWire::F32) {
        const float* peers0[kArRanks] = {};
        const float* peers1[kArRanks] = {};
        const uint32_t* pt[kArRanks] = {};
        const uint32_t* pa[kArRanks] = {};
        for (int j = 0; j < ranks_; ++j) {
            peers0[j] = (const float*) v.peers[j][parity];
            peers1[j] = (const float*) v.peers[j][1 - parity];
            pt[j] = v.peer_timeout[j];
            pa[j] = v.peer_arrival[j][parity];
        }
        ar_reduce_kernel<float>(*q, x, r, ranks_, (float*) v.mine[parity], (float*) v.mine[1 - parity], peers0,
                                peers1, pt, count, v.arrival[parity], pa, v.done[parity], v.peer_done[parity],
                                v.token, v.timeout, v.progress, blocks_, kArThreads, strata::spin_max(*q));
    } else {
        const uint16_t* peers0[kArRanks] = {};
        const uint16_t* peers1[kArRanks] = {};
        const uint32_t* pt[kArRanks] = {};
        const uint32_t* pa[kArRanks] = {};
        for (int j = 0; j < ranks_; ++j) {
            peers0[j] = (const uint16_t*) v.peers[j][parity];
            peers1[j] = (const uint16_t*) v.peers[j][1 - parity];
            pt[j] = v.peer_timeout[j];
            pa[j] = v.peer_arrival[j][parity];
        }
        ar_reduce_kernel<uint16_t>(*q, x, r, ranks_, (uint16_t*) v.mine[parity], (uint16_t*) v.mine[1 - parity],
                                   peers0, peers1, pt, count, v.arrival[parity], pa, v.done[parity],
                                   v.peer_done[parity], v.token, v.timeout, v.progress, blocks_, kArThreads,
                                   strata::spin_max(*q));
    }
}

bool ArPair::reduce_host(int rank, float* x, int count, void* stream, std::string& err) {
    if (!ready_) { err = "allreduce: reduce before a successful init"; return false; }
    if (rank < 0 || rank >= ranks_) {
        err = "allreduce: rank " + std::to_string(rank) + " is outside 0.." + std::to_string(ranks_ - 1);
        return false;
    }
    if (x == nullptr) { err = "allreduce: null buffer"; return false; }
    if (count < 1 || count > max_floats_) {
        err = "allreduce: count " + std::to_string(count) + " is outside 1.." + std::to_string(max_floats_) +
              "; the pair is sized for a verify window";
        return false;
    }
    if (DPCT_CHECK_ERROR(dpct::select_device(dev_[rank])) != 0) {
        err = "allreduce: select_device(" + std::to_string(dev_[rank]) + ") failed";
        return false;
    }
    launch(rank, x, count, stream);
    return true;
}

void ArPair::flush(int rank, void* stream) {
    // NO SUBMISSION HERE.  The CUDA original needed a trailing event under WDDM; on Level Zero an in-order
    // queue submits immediately, and inside a captured window graph a barrier would be recorded as an extra
    // node (or refused).  The graph's own end_recording is the submission.
    (void) rank;
    (void) stream;
}

void ArPair::release() {
    if (!ready_) return;
    // Past any token, in both parities of every rank: `peer_arrival < token` is then false for every waiter,
    // so a rank whose peer is gone drains instead of spinning.  The values are NOT the sum afterwards, so the
    // caller must refuse the window.
    for (int r = 0; r < ranks_; ++r)
        for (int p = 0; p < 2; ++p)
            for (int b = 0; b < blocks_; ++b)
                strata::sys_store(h_arrival_[r][p] + (size_t) b * kArrivalInts, 0xFFFFFFFFu);
    for (int p = 0; p < 2; ++p)
        for (int b = 0; b < blocks_; ++b)
            strata::sys_store(h_done_[p] + (size_t) b * kArrivalInts, 0xFFFFFFFFu);
}

void ArPair::clear_timeout() {
    if (!ready_) return;
    for (int r = 0; r < ranks_; ++r)
        if (h_timeout_[r] != nullptr) strata::sys_store(h_timeout_[r], 0u);
}

int ArPair::timed_out_rank() const {
    if (!ready_) return -1;
    for (int r = 0; r < ranks_; ++r)
        if (strata::sys_load(h_timeout_[r]) == kTimedOut) return r;
    return -1;
}

bool ArPair::token_counts(uint32_t* out) const {
    if (!ready_ || out == nullptr) return false;
    for (int r = 0; r < ranks_; ++r) {
        if (DPCT_CHECK_ERROR(dpct::select_device(dev_[r])) != 0) return false;
        sycl::queue& q = dpct::get_in_order_queue();
        q.memcpy(&out[r], view_[r].token, sizeof(uint32_t)).wait();
    }
    return true;
}

uint32_t ArPair::peek_progress(int rank, int which) const {
    if (!ready_ || which < 0 || which > kArProgressSawPeer) return 0;
    return strata::sys_load(h_progress_[rank <= 0 ? 0 : (rank >= ranks_ ? ranks_ - 1 : rank)] + which);
}

uint32_t ArPair::peek_arrival(int rank, bool peer, int parity, int block) const {
    if (!ready_) return 0;
    const int r = !peer ? (rank <= 0 ? 0 : (rank >= ranks_ ? ranks_ - 1 : rank)) : 0;
    const int p = parity & 1;
    const int b = block < 0 ? 0 : (block >= blocks_ ? blocks_ - 1 : block);
    return strata::sys_load(h_arrival_[r][p] + (size_t) b * kArrivalInts);
}

}  // namespace strata::kernels
