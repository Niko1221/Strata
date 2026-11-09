// metal/src/engine/stream.h - the SSD-streaming expert arena for Macs whose memory cannot hold the model (MAC-PLAN
// Phase 3, the small-Mac path; measured on the M4 Mac mini 32 GB, phaseA-tests/notes-mac-ssd.md).
//
//   * The dense weights (everything but the routed experts) are read once into one resident shared buffer; the
//     GGUF is never wrapped as a whole (a 37.6 GB NoCopy buffer cannot be made resident in 32 GB).
//   * The routed experts live in a resident arena of slots (one shared MTLBuffer), filled straight from the GGUF
//     with F_NOCACHE preads: one SSD read per expert slice, no staging copy, no page cache.  smap (48 x 512 uint32:
//     the slot's 16 KiB page in the arena, or ~0) is what the expert kernels read (moe.metal, function constant 20).
//   * The decode window is one command buffer in 48 segments; after each layer's router the GPU signals a shared
//     event and waits for the host (mtl_rt run_graph_seg).  The host's turn (at_layer): the window's routed experts
//     of that layer are looked up, the missing ones read into LRU victims (several reads in flight), smap updated;
//     then the experts the NEXT layer's router predicts from this layer's FFN input are read ahead on the I/O
//     threads while the GPU computes.
//   * The arena starts with the expert profile's top-ranked pairs (data/expert-profile.bin).
// The numbers are exact: a slot holds the expert's GGUF bytes unchanged, so the answers are those of the resident
// engine (the kernels read the same bytes at another address).
#pragma once

#include "gguf_lite.h"
#include "mtl_rt.h"

#include <functional>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::metal_engine {

class ExpertStream {
public:
    ~ExpertStream();
    // On when STRATA_METAL_STREAM=1, or (unset / auto) when the GGUF's experts do not fit in the GPU working set
    // with room to spare.  =0: never (the whole GGUF wrapped, as on the M5 Max).
    static bool wanted(const gguf::File& f, const mtl::Context& cx);
    bool open(const gguf::File& f, std::string& err);
    // pread `len` bytes at file offset `off` (F_NOCACHE)
    bool read(void* dst, uint64_t len, uint64_t off) const;
    // The dense tensors into one buffer; every tensor's offset in `f` is rewritten to its offset in `out`.
    // Tensors named in `skip` are left out (their offset is set to 0: they must not be read any more).
    bool load_dense(gguf::File& f, mtl::Context& cx, const std::vector<std::string>& skip, mtl::Buffer& out,
                    std::string& err);
    // load_dense's reads again, into `base` (the buffer macOS emptied while it was volatile)
    bool reload_dense(uint8_t* base);
    // The arena: `budget` bytes of slots, filled from the profile's ranking (or layer-interleaved if none).
    // min_slots: each size class at least this many slots (576 with prompt chunks, else 64)
    // recheck (optional): just before the arena is committed, the bytes it may have now (arena_shrink); when that is
    // less than the arena laid out, nothing is allocated, shrink_to() says how much, and the caller tries again
    bool make_arena(mtl::Context& cx, uint64_t budget, const std::string& profile, int min_slots, std::string& err,
                    const std::function<uint64_t()>& recheck = {},
                    const std::function<bool(uint64_t)>& grow = {});
    uint64_t shrink_to() const { return shrink_to_; }
    uint64_t arena_used_bytes() const { return arena_live_; }   // the arena's bytes in use (its retired slots left out)
    uint64_t wired_bytes() const { return wired_ ? arena_live_ : 0; }   // the arena's mlocked bytes
    bool ready() const { return !chunks_.empty(); }
    // the arena's buffers (up to 64, each within the device's maxBufferLength): chunk(i) for i >= count is chunk 0
    const mtl::Buffer& chunk(int i) const { return chunks_[(size_t) i < chunks_.size() ? (size_t) i : 0]; }
    // each chunk's GPU address: the kernels' arena_slot reads it (bound as buffer 7)
    const mtl::Buffer& table() const { return table_; }
    const mtl::Buffer& smap() const { return smap_; }
    uint64_t up_in_slot(int l) const { return lay_[(size_t) l].gs; }
    uint64_t down_in_slot(int l) const { return lay_[(size_t) l].gs + lay_[(size_t) l].us; }
    // The host's turn at layer l: ids = the window's T x topk routed ids; pred = T x 512 router logits of layer l + 1
    // computed from this layer's FFN input (nullptr: no read-ahead).
    // pred2: layer l + 2's, from the same input (STRATA_METAL_PREFETCH2 = its top-k, 0 = off)
    void at_layer(int l, int T, int topk, const int32_t* ids, const float* pred, const float* pred2 = nullptr);
    int ahead2() const { return pre_k2_; }
    // the arena has a slot for every expert (STRATA_METAL_ARENA_GB above the experts' size): a prompt's
    // buffers get their own memory, and complete() (every expert in a slot, nothing shed) lets the window skip the host's
    // per-layer turns, so it runs as fast as the whole GGUF wrapped (resident)
    bool full() const { return full_; }
    // Layers held whole. At a token boundary (no GPU work in flight) hold() picks, per size class, as
    // many whole layers as its live slots allow beside a pool for the rest (STRATA_METAL_HELD_POOL of each other
    // layer's experts, default 0.25), queues reads for their missing experts at the lowest priority, and marks the
    // held layers whose every expert is in a slot: skip(l) - the window needs no host turn there (victim() never
    // takes a held layer's slot). STRATA_METAL_HELD=0 turns it off.
    void hold();
    bool skip(int l) const { return (size_t) l < skip_.size() && skip_[(size_t) l] != 0; }
    bool holding() const { return held_on_; }
    bool complete() const { return full_ && missing_ == 0 && shed_.empty(); }
    // a prompt's scratch buffers inside the arena: region() places them (once), hold_region() empties and holds its
    // slots while a prompt runs and gives them back after
    uint8_t* region(uint64_t bytes);
    void hold_region(bool hold);
    void report(FILE* f) const;
    void reset_stats();
    // Between requests (STRATA_METAL_IDLE_RELEASE_MS, an arena that is not elastic): release() empties every slot,
    // unwires the arena and gives its pages back to macOS (MADV_FREE_REUSABLE); refill() reads the profile's ranked pairs
    // back in, as the first fill did (not wired again: an idle arena is released, not compressed). Both only while no
    // request runs.
    bool release();
    bool refill();
    // STRATA_METAL_ARENA_ELASTIC=1: the arena resized between requests. shrink() gives back up to
    // `bytes` from each size class's LRU back (never below the slots a prompt chunk needs, never the prompt region's
    // slots): those slots leave the LRU, so the next requests run on a smaller arena. grow() takes up to `bytes` of them
    // back, the most recently used first: their experts are read back in, wired again and rejoin the LRU's back.
    uint64_t shrink(uint64_t bytes);
    uint64_t grow(uint64_t bytes);
    // the idle release sheds the prompt band too (the floor's slots above the decode core), and the
    // next request takes it back, whatever memory is spare, before anything runs
    uint64_t shed_prompt_band();
    uint64_t prompt_band_back();
    // the same grow during a request. grow_start() picks what grow(bytes) would take back and reads it
    // on a thread of its own into those shed slots (out of every LRU, in no smap entry: nothing the decode or the GPU
    // touches) and wires them; grow_commit(), at a token boundary (Window::run: no GPU work in flight), publishes them
    // as grow() does. grow() commits a finished one instead of growing; shrink() and release() drop an unfinished one.
    bool grow_start(uint64_t bytes);
    uint64_t grow_commit();   // 0 while no started grow has finished its reads
    bool growing() const { return bg_.joinable(); }
    uint64_t shed_bytes() const { return shed_bytes_; }
    uint64_t shed_resident() const;   // the shed slots' pages macOS has not taken yet (still this process's, reclaimable)

private:
    std::vector<std::pair<int, int>> fill_;   // the ranked pairs the first fill read, best first
    bool wired_ = false, released_ = false;
    struct Gone { int slot, layer, e; };   // a shed slot and what it held (layer -1: was empty)
    std::vector<Gone> shed_;   // shrink()'s slots, coldest first (grow() takes them from the back)
    struct DenseRead { uint64_t dst, len, src; };
    std::vector<DenseRead> dense_reads_;   // load_dense's reads, for reload_dense
    uint64_t shed_bytes_ = 0;
    int min_slots_ = 64;
    bool full_ = false;
    int missing_ = 0;   // experts in no slot (full arena only; kept by fill_missing)
    void fill_missing();
    // An elastic arena is laid out in tiers: each class's first min_slots (the floor) in buffers [0, extra0_), the rest
    // above, in chunks of ~1 GB (STRATA_METAL_ARENA_CHUNK_GB). The GPU wires every page of a buffer it is given, shed
    // slots too, so the buffers above are shed whole, the top one first: the buffer is released and its pages replaced,
    // and chunk() returns buffer 0 for it until it is back
    int extra0_ = -1;
    int core0_ = -1;   // buffers [0, core0_) the decode core, [core0_, extra0_) the prompt band
    int live_bufs_ = 0;   // buffers [live_bufs_, chunks_.size()) are parked
    std::vector<std::pair<void*, size_t>> parked_mem_;   // each buffer's memory, for wrapping it again
    mtl::Context* cx_ = nullptr;
    int buf_of(int s) const { return (int) (code_[(size_t) s] >> 26); }
    uint64_t shed_top(int floor);
    // what a grow of `bytes` takes back (grow's choice, no state changed) and the publishing step
    struct GrowPlan { std::vector<Gone> back, keep; int live = 0; uint64_t got = 0; };
    GrowPlan plan_grow(uint64_t bytes) const;
    uint64_t apply_grow(GrowPlan& p, double t0, uint64_t b0, double t_read, double t_wired, bool bg);
    void drop_bg();
    std::thread bg_;
    std::atomic<bool> bg_done_{false}, bg_stop_{false};
    GrowPlan bg_plan_;
    double bg_t0_ = 0, bg_tr_ = 0, bg_tw_ = 0;
    uint64_t bg_bytes_ = 0;   // written by the thread, read after its join
    std::vector<uint8_t> bg_mark_;   // the slots it fills: not shed-resident (reclaimable) memory while it runs
    uint64_t shrink_to_ = 0;
    struct Lay { uint64_t g, gs, u, us, d, ds; int cls; };
    struct Slot { int16_t layer = -1; int16_t e = -1; int prev = -1, next = -1; bool pre = false; int left = 0; uint64_t pin = 0; std::atomic<int> state{0}; };  // 0 idle, 1 queued, 2 reading
    struct Cls { uint64_t pages = 0; int n = 0; int base = 0; int head = -1, tail = -1; };  // slot range [base, base+n)
    struct Job { int slot; int layer, e, part; };   // part: slice * split_ + piece
    std::deque<Job> fill_q_;   // a held layer's missing experts, read when nothing else waits
    std::vector<uint8_t> hold_, skip_;   // per layer: held (victim() skips it); held and whole
    bool held_on_ = true;
    double held_pool_ = 0.25;
    int held_n_ = -1, skip_n_ = -1;   // last logged
    void drop_queued();   // the queued reads not started (read-ahead and fill), their slots freed; mu_ held
    void lru_unlink(int s);
    void lru_front(int s);
    void lru_back(int s);
    int cls_of(int s) const;
    int victim(int cls);
    void read_part(int slot, int l, int e, int part);
    void cancel_slot(int s);
    void promote_slot(int s);
    void io_main();
    void ahead(int ln, int T, const float* pred, int K);
    void submit(int slot, int l, int e, bool urgent, bool fill = false);
    void wait_slot(int s);
    uint8_t* standalone_region(uint64_t bytes);   // the prompt's buffers as their own allocation, when no region of the arena's slots takes them

    int fd_ = -1;
    // <gguf>.experts (metal/tools/stream_experts.py): each expert's gate, up and down rows together, read with one
    // request (cfd_ >= 0); else the GGUF's three slices.  STRATA_METAL_CONTIG=0: the GGUF always.
    int cfd_ = -1;
    std::vector<uint64_t> coff_;
    std::vector<Lay> lay_;
    std::vector<Cls> cls_;
    std::vector<Slot> slots_;
    std::vector<uint8_t> retired_;   // slots left unused when the stepped fill stopped at macOS's floor (never in an LRU)
    std::vector<int32_t> where_;      // (l * 512 + e) -> slot, -1
    std::vector<mtl::Buffer> chunks_;
    mtl::Buffer table_;   // chunk -> GPU address (a parked chunk: chunk 0's)
    void set_chunk(int k, bool live);   // wraps (live) or releases chunk k: table, used set
    std::vector<int> region_slots_;
    uint8_t* region_ = nullptr;
    uint64_t region_bytes_ = 0;
    uint64_t region_own_ = 0;   // the prompt buffers' own allocation
    std::vector<uint32_t> code_;      // slot -> its smap value (chunk << 26 | page)
    std::vector<uint8_t*> host_;      // slot -> its CPU address
    mtl::Buffer smap_;
    uint64_t arena_bytes_ = 0;
    uint64_t arena_live_ = 0;
    // the I/O threads
    std::vector<std::thread> io_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::deque<Job> urgent_q_, pre_q_;
    bool stop_ = false;
    int outstanding_ = 0;
    int nio_ = 6, pre_k_ = 10, pre_k2_ = 0, split_ = 2;
    bool prefetch_ = true, fake_ = false, pre_mru_ = false;
    int ne_ = 512;   // routed experts per layer of this model (512; 256 for the expert-pruned Coder), read from the GGUF at open(); stream.cpp's NE is the largest (array sizes, the 512-wide key / smap stride)
    // stats
    uint64_t stamp_ = 1;   // the handshake count: slots pinned with the current one are in use on the GPU
    uint64_t p_lookups_ = 0, p_hits_ = 0, p_pre_ = 0, p_misses_ = 0, p_streamed_ = 0, p_chunks_ = 0;
    double p_wait_ms_ = 0, p_host_ms_ = 0;
    uint64_t lookups_ = 0, hits_ = 0, pre_hits_ = 0, late_ = 0, misses_ = 0, pre_issued_ = 0, pre_cancel_ = 0, windows_ = 0;
    std::atomic<uint64_t> bytes_read_{0}, reads_{0};
    double wait_ms_ = 0, host_ms_ = 0, fill_s_ = 0;
    uint64_t fill_bytes_ = 0;
};

}  // namespace strata::metal_engine
