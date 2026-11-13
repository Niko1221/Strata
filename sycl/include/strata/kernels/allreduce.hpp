// include/strata/kernels/allreduce.hpp - the cross-GPU sum tensor parallelism needs.
//
// The protocol is llama.cpp's, extended from two ranks to up to four: every rank stages its partial sum
// through mapped pinned host memory, meets on a device-side arrival token, and the ranks add the partials.
//
// FOR TWO RANKS the original protocol is kept: each rank stages its half, signals, waits for the peer and adds
// the peer's half in place.  FOR THREE OR FOUR RANKS the same staging is used, but the meeting is a
// gather/broadcast through rank 0: every rank signals its arrival, rank 0 waits for all of them, sums the
// staged halves into its own staging region, and publishes a done token; every rank then reads rank 0's sum.
// That is one extra hop per reduce compared with a ring, and it is the simplest protocol that keeps the result
// BIT-IDENTICAL on every rank - which is the requirement, because the router must pick the same expert ids on
// every rank or the ranks run different models.
//
// THREE DELIBERATE PROPERTIES, each a bug in the captured-graph world:
//
//   1. THE TOKEN COMES FROM THE DEVICE.  A verify window is ONE CAPTURED GRAPH replayed hundreds of times; a
//      token baked into a graph is the same on every replay.  Each block owns a device counter and bumps it
//      with one atomic per execution.
//   2. PING-PONG STAGING.  With a single staging region the peer's call k+1 overwrites the bytes this rank is
//      still reading for call k.  Two regions selected by call parity fix it.
//   3. MONOTONIC ARRIVAL (`>= token`, not `== token`).  Equality cannot be released from the host, which does
//      not know the device-side token.  With a monotonic compare, writing UINT32_MAX into the arrival slots lets
//      a stalled waiter fall through, which is what the watchdog and the give-up paths need.
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace strata::kernels {

/// What the ranks round through before adding.  BF16 is the default: the model's own activations are bf16 in
/// the contract the kernels follow, and a verify window is kVerifyMaxT * n_embd floats.  The wire rounding is
/// what makes the ranks' results bit-identical: every rank truncates to the same type before adding.
enum class ArWire : int {
    F32 = 0,      ///< wire == the accumulator; the honest baseline for a parity test
    Bf16 = 1,
};

/// The largest rank count this collective supports.  TP=2 is the original pair; TP=3/4 use the rank-0
/// gather/broadcast path.  The engine's shard table decides whether a geometry can use a given count.
inline constexpr int kArRanks = 4;
/// Blocks per reduce.  Each block stripes a disjoint slice of the payload and has its own arrival slot, so
/// more blocks pump more PCIe stores in parallel.
inline constexpr int kArBlocks = 8;
inline constexpr int kArThreads = 256;

/// One rank's device-side view of the group's memory.  These are the pointers a captured graph records, so
/// they must outlive every graph that captured a reduce.
struct ArRankView {
    int dev = -1;
    void* mine[2] = {};                              ///< this rank WRITES here, indexed by call parity
    const void* peers[kArRanks][2] = {};             ///< every rank's staging half, indexed [rank][parity]
    uint32_t* arrival[2] = {};                       ///< its own arrival tokens: [parity][block]
    const uint32_t* peer_arrival[kArRanks][2] = {};  ///< every rank's arrival slots
    uint32_t* done[2] = {};                          ///< rank 0's done tokens: [parity][block] (rank 0 writes)
    const uint32_t* peer_done[2] = {};               ///< rank 0's done tokens, read by every rank
    uint32_t* token = nullptr;                       ///< [block] device counters, one bump per execution
    uint32_t* timeout = nullptr;
    const uint32_t* peer_timeout[kArRanks] = {};     ///< every rank's give-up word
    /// [4] mapped progress words, host-readable with no sync: entered / staged / signalled / saw-peer.
    uint32_t* progress = nullptr;
};

/// The indices into `ArRankView::progress`.
enum : int {
    kArProgressEntered = 0,     ///< block 0 began executing
    kArProgressStaged = 1,      ///< its stripe is in the staging region and fenced
    kArProgressSignalled = 2,   ///< it wrote its arrival token
    kArProgressSawPeer = 3,     ///< its wait finished
};

/// The ranks a tensor-parallel group spans, and the memory their collectives stage through.
///
/// ONE INSTANCE PER PROCESS, shared by every rank.  The engine runs its ranks as host threads in one process
/// (a rank cannot share a stream, a doorbell or a CPU expert pool with its peer), and each rank's kernels have
/// to address every rank's staging areas anyway, which is exactly what one mapped allocation per rank gives.
class ArPair {
public:
    ArPair() = default;
    ~ArPair();
    ArPair(const ArPair&) = delete;
    ArPair& operator=(const ArPair&) = delete;

    /// `ordinals[0..ranks)` are the ranks in order (the engine's stage order: the card that drives the display
    /// and carries the head is rank 0).  `max_floats` is the largest count one reduce will carry - a verify
    /// window, so `kVerifyMaxT * n_embd`.  A prompt chunk is deliberately NOT served by this kernel: at
    /// thousands of rows the staged bytes dominate and llama.cpp switches to a copy-engine route.
    bool init(const int* ordinals, int ranks, int64_t max_floats, ArWire wire, int blocks, std::string& err);
    void shutdown();
    bool ready() const { return ready_; }

    ArWire wire() const { return wire_; }
    int blocks() const { return blocks_; }
    int ranks() const { return ranks_; }
    int64_t max_floats() const { return max_floats_; }
    /// 1 when every direction reported peer access, 0 when the staged path is what runs, -1 before `init`.
    int peer_mode() const { return peer_mode_; }

    /// HOST view of a rank's staging half.  The region is mapped host memory, so the host can read exactly the
    /// bytes a rank's kernel staged - the one link in the reduce that has never been observed directly.
    const void* host_staging(int rank, int half) const;
    /// Mapped host bytes the pair holds, for the startup line and the memory guard.
    uint64_t pinned_bytes() const { return pinned_bytes_; }
    const ArRankView& rank(int r) const { return view_[r <= 0 ? 0 : (r >= ranks_ ? ranks_ - 1 : r)]; }

    /// Sum `x` (count floats, ON RANK rank's DEVICE) with every peer's half, in place, on `stream`, which must
    /// be a stream of that rank's device.  Callable inside a graph capture: no host work and no synchronisation.
    /// Every rank MUST record its call with the same `count`, in the same order - the kernels wait for each
    /// other, and a rank that skips one leaves its peers spinning.
    void launch(int rank, float* x, int count, void* stream);
    /// `launch` with the argument checks and the launch error.  For host-driven callers.
    bool reduce_host(int rank, float* x, int count, void* stream, std::string& err);

    /// FORCE THIS RANK'S STREAM TO BE SUBMITTED.  On Level Zero an in-order queue submits immediately, so this
    /// is a trailing barrier rather than a flush; it keeps the CUDA contract and costs host microseconds.
    void flush(int rank, void* stream);

    /// Raise every rank's arrival slots past any token, so a waiter whose peer is gone drains instead of
    /// spinning forever.  Only for the give-up paths: afterwards the reduced values are NOT the sum, so the
    /// caller must refuse the window.
    void release();
    /// Which rank's kernel gave up waiting, or -1.  Reads mapped memory, so it needs no sync.
    int timed_out_rank() const;
    /// CLEAR the give-up flag on every rank.  The kernel only ever SETS it and the host only ever READS it.
    void clear_timeout();
    /// The token rank r's block-0 counter currently holds.  Copies from the device, so it synchronises.
    bool token_counts(uint32_t* out) const;
    /// How many reduces `launch` has recorded for rank r.
    uint64_t calls(int rank) const { return calls_[rank <= 0 ? 0 : (rank >= ranks_ ? ranks_ - 1 : rank)]; }
    /// Host read of one mapped arrival slot, with NO synchronisation.
    uint32_t peek_arrival(int rank, bool peer, int parity, int block) const;
    /// Host read of one progress word.  Also no synchronisation.
    uint32_t peek_progress(int rank, int which) const;

private:
    bool ready_ = false;
    int ranks_ = 2;
    int dev_[kArRanks] = {-1, -1, -1, -1};
    ArWire wire_ = ArWire::Bf16;
    int blocks_ = kArBlocks;
    int64_t max_floats_ = 0;
    int peer_mode_ = -1;
    uint64_t pinned_bytes_ = 0;
    uint64_t calls_[kArRanks] = {0, 0, 0, 0};
    void* host_[kArRanks] = {nullptr, nullptr, nullptr, nullptr};
    ArRankView view_[kArRanks];
    uint32_t* h_arrival_[kArRanks][2] = {};
    uint32_t* h_done_[2] = {nullptr, nullptr};
    uint32_t* h_timeout_[kArRanks] = {};
    uint32_t* h_progress_[kArRanks] = {};
};

/// Diagnostics: publish a pair, then read a rank's staging half from the HOST.
void ar_debug_set_pair(ArPair* p);
const void* ar_debug_staging(int rank, int half);

}  // namespace strata::kernels
