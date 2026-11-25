// metal/src/engine/stream.h - the SSD-streaming expert arena for Macs whose memory cannot hold the model.
//
//   * The dense weights (everything but the routed experts) are read once into one resident shared buffer; the
//     GGUF is never wrapped as a whole (one NoCopy buffer over a model larger than the machine's memory cannot be
//     made resident).
//   * The routed experts live in a resident arena of slots (one shared MTLBuffer), filled straight from the GGUF
//     with F_NOCACHE preads: one SSD read per expert slice, no staging copy, no page cache.  smap (NL x 512 uint32:
//     the slot's 16 KiB page in the arena, or ~0) is what the expert kernels read (moe.metal, function constant 20).
//   * The decode window is one command buffer in one segment per expert layer; after each layer's router the GPU
//     signals a shared event and waits for the host (mtl_rt run_graph_seg).  The host's turn (at_layer): the
//     window's routed experts of that layer are looked up, the missing ones read into LRU victims (several reads in
//     flight), smap updated; then the experts the NEXT layer's router predicts from this layer's FFN input are read
//     ahead on the I/O threads while the GPU computes.
//   * The arena starts with the expert profile's top-ranked pairs (data/expert-profile.bin).
// The numbers are exact: a slot holds the expert's GGUF bytes unchanged, so the answers are those of the resident
// engine (the kernels read the same bytes at another address).
//
// THE LAYER INDEX IS AN EXPERT ORDINAL, NOT A BLOCK ID.  Qwen3.8-Flash-Next has routed experts in every one of its
// 48 blocks, so there ordinal == block id and the two readings coincide.  hy4-flash does not: its block 0 is a dense
// FFN with no experts at all, blocks 1..41 are ordinals 0..40 and the MTP block takes the last ordinal
// (include/strata/hy4flash_gguf/geometry.hpp).  So the number of layers is a run-time argument and `open` takes the GGUF
// name prefix of each ordinal; everything keyed by layer - lay_, smap, where_, the profile's ranking, the read-ahead
// and the report - counts ordinals.
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
    // with room to spare.  =0: never (the whole GGUF wrapped).
    static bool wanted(const gguf::File& f, const mtl::Context& cx);
    // The GGUF's descriptor alone (F_NOCACHE, no read-ahead), without the per-layer expert table.  `load_dense`
    // needs only the descriptor, and a model whose first block has no routed experts (hy4-flash's leading dense
    // block) cannot pass open(); open() calls this first, so the Qwen path is unchanged.
    bool open_file(const std::string& path, std::string& err);
    // Qwen's geometry: 48 ordinals whose GGUF prefixes are "blk.0." .. "blk.47." (ordinal == block id).
    bool open(const gguf::File& f, std::string& err);
    // `block_prefix[o]` is the GGUF name prefix of expert ordinal o ("blk.7."), so a model whose experts do not
    // start at block 0, or do not cover every block, passes its own mapping (see the header comment).  The
    // routed-expert count comes from `block_prefix[0]`'s gate tensor.
    bool open(const gguf::File& f, const std::vector<std::string>& block_prefix, std::string& err);
    int n_layers() const { return nl_; }         // expert ordinals, not blocks
    int n_experts() const { return ne_; }
    // pread `len` bytes at file offset `off` (F_NOCACHE)
    bool read(void* dst, uint64_t len, uint64_t off) const;
    // The dense tensors into one buffer; every tensor's offset in `f` is rewritten to its offset in `out`.
    // Tensors named in `skip` are left out (their offset is set to 0: they must not be read any more).
    bool load_dense(gguf::File& f, mtl::Context& cx, const std::vector<std::string>& skip, mtl::Buffer& out,
                    std::string& err);
    // The arena: `budget` bytes of slots, filled from the profile's ranking (or layer-interleaved if none).
    // min_slots: each size class at least this many slots (576 with prompt chunks, else 64)
    // recheck (optional): just before the arena is committed, the bytes it may have now (arena_shrink); when that is
    // less than the arena laid out, nothing is allocated, shrink_to() says how much, and the caller tries again
    bool make_arena(mtl::Context& cx, uint64_t budget, const std::string& profile, int min_slots, std::string& err,
                    const std::function<uint64_t()>& recheck = {},
                    const std::function<bool(uint64_t)>& grow = {});
    uint64_t shrink_to() const { return shrink_to_; }
    uint64_t arena_used_bytes() const { return arena_live_; }   // the arena's bytes in use (its retired slots left out)
    bool ready() const { return !chunks_.empty(); }
    // the arena's buffers (at most 4, each within the device's maxBufferLength): chunk(i) for i >= count is chunk 0
    const mtl::Buffer& chunk(int i) const { return chunks_[(size_t) i < chunks_.size() ? (size_t) i : 0]; }
    // How many there really are.  A caller that dispatches one kernel per buffer needs it: dispatching for four
    // when the arena has one means three dispatches whose group slots are all empty, and an empty group still
    // launches its thread groups (engine/metal/src/hy4flash_gguf/forward.cpp's MoE loop).
    int n_chunks() const { return (int) chunks_.size(); }
    const mtl::Buffer& smap() const { return smap_; }
    uint64_t up_in_slot(int l) const { return lay_[(size_t) l].gs; }
    uint64_t down_in_slot(int l) const { return lay_[(size_t) l].gs + lay_[(size_t) l].us; }
    // The host's turn at layer l: ids = the window's T x topk routed ids; pred = T x 512 router logits of layer l + 1
    // computed from this layer's FFN input (nullptr: no read-ahead).
    // pred2: layer l + 2's, from the same input (STRATA_METAL_PREFETCH2 = its top-k, 0 = off)
    void at_layer(int l, int T, int topk, const int32_t* ids, const float* pred, const float* pred2 = nullptr);
    int ahead2() const { return pre_k2_; }
    // at_layer in two halves (the MoE split): begin_layer looks the window's experts up, pins them, submits
    // the misses and returns WITHOUT waiting; pending(l, e) then says whether expert e is one this layer still has
    // to wait for (its slot is assigned, its bytes are not there yet - slot_code_of() gives where they will be);
    // finish_layer() waits for exactly those and then issues the read-ahead.  at_layer(...) is begin_layer(...) +
    // finish_layer(), so a caller that does not split is unchanged.
    void begin_layer(int l, int T, int topk, const int32_t* ids, const float* pred, const float* pred2 = nullptr);
    void finish_layer();
    bool pending(int l, int e) const;
    // the smap value expert e of ordinal l has, or will have once its read completes (~0 when it has no slot)
    uint32_t slot_code_of(int l, int e) const {
        const int s = where_[(size_t) (l * 512 + e)];
        return s < 0 ? 0xFFFFFFFFu : code_[(size_t) s];
    }
    int ahead1() const { return prefetch_ ? pre_k_ : 0; }
    // A model's own default read-ahead depth, for when STRATA_METAL_PREFETCH is not set (that variable always
    // wins).  Call after open(), before make_arena().  A model that never calls it keeps open()'s expert-size rule.
    void default_read_ahead(int k);
    // Issue the read-ahead for l + 1 (and l + 2) BEFORE waiting for layer l's demand misses rather than after.
    // The read-aheads queue behind the demand reads (urgent first), so they start as soon as an I/O thread
    // is free instead of after the whole wait.  Off by default: Qwen keeps its order; hy4-flash turns it on.
    void set_ahead_before_wait(bool on) { ahead_early_ = on; }
    // How many handshakes a slot stays safe from eviction after the GPU was handed it.  0 - the default - means the
    // cut handshake IS the fence: `run_graph_seg` signals its shared event only after everything encoded before the
    // cut has run, so by the time at_layer(l + 1) is entered the GPU has finished layer l's expert matmuls and those
    // slots are free again.  An engine that encodes more than one layer ahead of the host's turn sets 1 or more, and
    // then no slot a still-running layer reads can become a victim.
    void set_fence_lag(int n) { fence_lag_ = n > 0 ? n : 0; }
    // TWO BYTE HIT RATES, and they are not interchangeable.
    //   byte_hit_rate()       1 - the DEMAND path's missed bytes / demanded bytes.  An expert the read-ahead
    //                         already fetched, or still has in flight, counts as a HIT here, so this is the rate
    //                         of "the arena AND the read-ahead together".
    //   arena_byte_hit_rate() 1 - the bytes the SSD delivered FOR DECODE / the bytes decode demanded.  The
    //                         read-ahead moves bytes from the demand path to the I/O threads; it does not remove
    //                         them, so THIS is the arena's own rate, the one comparable with an offline LRU
    //                         simulation, and `decode_read_bytes / windows` is what a tok/s budget must divide by
    //                         the SSD's bandwidth.
    // BOTH SIDES OF THE SECOND ONE COUNT DECODE ONLY.  Divided by anything that also holds the arena's fill or a
    // prompt chunk it is meaningless - it comes out negative, or orders of magnitude too high.  `decode_read_bytes`
    // is attributed at completion from the mode the slot was submitted in, so neither the fill nor a prompt can
    // leak into it.
    double byte_hit_rate() const { return d_bytes_ ? 1.0 - (double) m_bytes_ / (double) d_bytes_ : 0.0; }
    double arena_byte_hit_rate() const {
        return d_bytes_ ? 1.0 - (double) dr_bytes_ / (double) d_bytes_ : 0.0;
    }
    uint64_t decode_read_bytes() const { return dr_bytes_; }   // what the SSD delivered for decode windows
    uint64_t demand_bytes() const { return d_bytes_; }
    uint64_t miss_bytes() const { return m_bytes_; }
    uint64_t hits() const { return hits_; }        // resident when the demand path asked
    uint64_t misses() const { return misses_; }    // absent, so the demand path started the read itself
    uint64_t reads() const { return reads_.load(); }   // experts whose last slice completed
    uint64_t prefetch_issued() const { return pre_issued_; }
    uint64_t prefetch_hits() const { return pre_hits_; }
    uint64_t prefetch_dropped() const { return pre_cancel_; }
    uint64_t prefetch_late() const { return late_; }
    uint64_t windows() const { return windows_; }
    uint64_t bytes_read() const { return bytes_read_.load(); }
    double wait_ms() const { return wait_ms_; }
    // decode's wait_ms() split by cause: cuts that waited for a read the demand path started itself, and cuts
    // that waited only for a read-ahead still in flight.  host_ms() - wait_ms() is the host's own work at the cuts.
    double miss_wait_ms() const { return miss_wait_ms_; }
    double late_wait_ms() const { return late_wait_ms_; }
    uint64_t miss_wait_cuts() const { return miss_cuts_; }
    uint64_t late_wait_cuts() const { return late_cuts_; }
    double host_ms() const { return host_ms_; }
    // The prompt-chunk counters, kept apart from decode's throughout (at_layer's `prompt` branch).
    // `prompt_wait_ms` is the I/O half of a prefill chunk: the time at_layer spent blocked on the SSD for a
    // chunk's experts.
    uint64_t prompt_chunks() const { return p_chunks_; }
    uint64_t prompt_lookups() const { return p_lookups_; }
    uint64_t prompt_misses() const { return p_misses_; }
    uint64_t prompt_streamed() const { return p_streamed_; }
    double prompt_wait_ms() const { return p_wait_ms_; }
    double prompt_host_ms() const { return p_host_ms_; }
    // ---- what a (ordinal, expert) pair occupies, for the tests that compare the arena against the file
    uint64_t slice_offset(int l, int e, int k) const;   // k: 0 gate, 1 up, 2 down - its byte offset in the GGUF
    uint64_t slice_bytes(int l, int k) const;
    uint64_t expert_bytes(int l) const;                 // gate + up + down of one expert at ordinal l
    int size_class(int l) const { return lay_[(size_t) l].cls; }
    int n_classes() const { return (int) cls_.size(); }
    int class_slots(int c) const { return cls_[(size_t) c].n; }
    uint64_t class_bytes(int c) const { return cls_[(size_t) c].pages * 16384ull; }
    int slot_of(int l, int e) const { return where_[(size_t) (l * 512 + e)]; }
    const uint8_t* slot_host(int s) const { return host_[(size_t) s]; }
    uint32_t smap_of(int l, int e) const { return smap_.as<uint32_t>()[l * 512 + e]; }
    uint32_t slot_code(int s) const { return code_[(size_t) s]; }
    int n_slots() const { return (int) slots_.size(); }
    // "" when every arena invariant holds, else the first one that does not.  Quiescent only (no read in flight):
    // the tests call it between windows.
    std::string check_invariants() const;
    // a prompt's scratch buffers inside the arena: region() places them (once), hold_region() empties and holds its
    // slots while a prompt runs and gives them back after
    uint8_t* region(uint64_t bytes);
    void hold_region(bool hold);
    void report(FILE* f) const;
    void reset_stats();

private:
    uint64_t shrink_to_ = 0;
    struct Lay { uint64_t g, gs, u, us, d, ds; int cls; };
    // `mode` is what the arena was doing when this slot's read was submitted: 0 the initial fill, 1 a decode
    // window, 2 a prompt chunk.  It is what lets the decode-only byte counter exist (arena_byte_hit_rate).
    struct Slot { int16_t layer = -1; int16_t e = -1; int prev = -1, next = -1; bool pre = false; uint8_t mode = 0; int left = 0; uint64_t pin = 0; std::atomic<int> state{0}; };  // 0 idle, 1 queued, 2 reading
    struct Cls { uint64_t pages = 0; int n = 0; int base = 0; int head = -1, tail = -1; };  // slot range [base, base+n)
    struct Job { int slot; int layer, e, part; };   // part: slice * split_ + piece
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
    void submit(int slot, int l, int e, bool urgent);
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
    std::vector<int> region_slots_;
    uint8_t* region_ = nullptr;
    uint64_t region_bytes_ = 0;
    std::vector<uint32_t> code_;      // slot -> its smap value (chunk << 28 | page)
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
    int nl_ = 0;         // expert ordinals of this model (48 for Qwen, 41 for hy4-flash's trunk), from open()
    int fence_lag_ = 0;  // set_fence_lag
    bool prefetch_ = true, fake_ = false, pre_mru_ = false, ahead_early_ = false;
    // begin_layer -> finish_layer
    int pend_l_ = -1, pend_T_ = 0;
    bool pend_prompt_ = false, pend_fake_ = false, pend_any_miss_ = false, pend_ahead_now_ = false;
    const float* pend_pred_ = nullptr;
    const float* pend_pred2_ = nullptr;
    std::vector<int> pend_wait_;
    int ne_ = 512;   // routed experts per layer of this model (512; 256 for the expert-pruned Coder), read from the GGUF at open(); stream.cpp's NE is the largest (array sizes, the 512-wide key / smap stride)
    // stats
    uint64_t stamp_ = 1;   // the handshake count: slots pinned with the current one are in use on the GPU
    uint64_t p_lookups_ = 0, p_hits_ = 0, p_pre_ = 0, p_misses_ = 0, p_streamed_ = 0, p_chunks_ = 0;
    double p_wait_ms_ = 0, p_host_ms_ = 0;
    uint64_t lookups_ = 0, hits_ = 0, pre_hits_ = 0, late_ = 0, misses_ = 0, pre_issued_ = 0, pre_cancel_ = 0, windows_ = 0;
    uint64_t d_bytes_ = 0, m_bytes_ = 0;   // decode: the bytes looked up and the bytes the demand path missed
    uint64_t dr_bytes_ = 0;                // decode: the bytes the SSD actually delivered (demand + read-ahead)
    int mode_ = 0;                         // Slot::mode for the reads submitted now
    std::atomic<uint64_t> bytes_read_{0}, reads_{0};
    double wait_ms_ = 0, host_ms_ = 0, fill_s_ = 0;
    double miss_wait_ms_ = 0, late_wait_ms_ = 0;
    uint64_t miss_cuts_ = 0, late_cuts_ = 0;
    uint64_t fill_bytes_ = 0;
};

}  // namespace strata::metal_engine
