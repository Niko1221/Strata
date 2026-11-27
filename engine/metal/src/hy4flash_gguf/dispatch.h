// engine/metal/src/hy4flash_gguf/dispatch.h - the hy4-flash engine's own dispatch rules.
//
// WHY THE QWEN ENGINE'S HELPERS ARE NOT REUSED: `metal/src/engine/window.cpp` chooses its GEMV tiles from the
// token count and from environment variables, has a half-activation / half-accumulation path
// (STRATA_METAL_F16) and a timing path that produces WRONG results (STRATA_METAL_SKIP).  hy4-flash's numerics are
// frozen on f32 activations and on T invariance - the same position must give bit-identical output for any
// DECODE window size T = 1..8 - so the tile strategy here is fixed and those switches are REFUSED rather than
// honoured.  (T invariance covers DECODE only: a chunk of more than 8 positions runs the fp32 MMA below, a
// trailing chunk of 8 or fewer runs the GEMV, and the two order their sums differently, so prefill is judged by
// tolerance against decode rather than bitwise.  Every DECODE path is bitwise.)  Refusing beats a silent
// downgrade: a run that quietly used half activations would pass nothing and look like a numerics bug in the
// kernels.
//
// The tile choice itself is in msl/hy4_flash_kernels.h, shared with the kernel parity tests so the two cannot drift.
// `DenseGemvF32` and the MoE helpers below are shared the same way - the kernel tests' own gemv / moe headers
// forward to this one, so there is a single copy.
#pragma once

#include "hy4_flash_kernels.h"
#include "mtl_rt.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace strata::hy4_flash {

/// "" when the environment is one this engine may run in, else the variable that forbids it and why.
///
/// The list itself lives in `mtl::deterministic_env_violation` (mtl_rt.h), because `Context::lock_safe_math`
/// has to end the process on the same set at compile time: STRATA_METAL_MATH=fast is in it because `mtl_rt.mm`
/// lets that variable override the `fast_math=false` a caller passes to `Context::compile`, so passing the
/// argument is not enough to get safe math.  This is the host-side face of the same check, run before the engine
/// has a device.
std::string rejected_env();

// ---- the dense GEMV, f32 activations -------------------------------------------------------------------------------
// mirrors GemvArgs in metal/kernels/gemv.metal
struct GemvArgs {
    uint32_t n_in, n_out, n_tok, row_bytes, x_stride, y_stride;
};

/// Clears the memory barrier of every dispatch in `ds` after index `at`, i.e. says "these may overlap".
///
/// `mtl::Context::run_graph` puts a device memory barrier before every dispatch whose `barrier` is set, and a
/// barrier on this hardware costs about as much as a small dispatch, so serialising a large group of small
/// dispatches can be most of a prefill chunk's GPU time.  Use this ONLY for a group of dispatches that write
/// disjoint memory and read nothing the group writes - the splits of one `DenseGemvF32::append` (disjoint token
/// rows), the arena chunks of one MoE sub-batch (disjoint entry ranges).  It is a SCHEDULING change: bitwise
/// identical, and `--prefill-consistency` is the evidence.
/// `HY4_FLASH_OVERLAP` is a bit mask over the groups below, so one of them can be turned off on its own when a
/// bisection needs it.  Default: all of them.
/// (kOverlapHeads, bit 2, now selects nothing: the per-head GEMVs it covered are one dispatch per 8 tokens, so
/// their splits fall under kOverlapGemvSplit.  It stays reserved so an existing mask keeps its meaning.)
enum { kOverlapGemvSplit = 1, kOverlapHeads = 2, kOverlapMoeGu = 4, kOverlapMoeDown = 8, kOverlapAll = 15 };
inline int overlap_mask() {
    static const int m = [] {
        if (const char* e = std::getenv("HY4_FLASH_OVERLAP")) return std::atoi(e);
        return (int) kOverlapAll;
    }();
    return m;
}
inline void relax_after(std::vector<mtl::Dispatch>& ds, size_t at, int group) {
    if ((overlap_mask() & group) == 0) return;
    for (size_t i = at + 1; i < ds.size(); ++i) ds[i].barrier = false;
}

/// mirrors Hy4FlashDenseMmArgs in msl/hy4_flash_mm.metal (the name carries "Dense" because hy4_flash_moe_mm.metal - the prefill
/// MoE grouped GEMM, compiled into the SAME library - has its own Hy4FlashMmArgs with different fields)
struct MmArgs {
    uint32_t n_in, n_out, n_tok, row_bytes, x_stride, y_stride;
};

/// Prefill's dense GEMMs on the matrix units (msl/hy4_flash_mm.metal): a window of more than kHy4FlashGemvMaxTok rows runs as
/// ONE fp32 MMA dispatch instead of ceil(T / 8) GEMVs.  OPT-IN PER SHAPE (`allow_mm`), because "more than 8 rows"
/// alone does NOT mean prefill: Mtp::catch_up builds the draft block's cache 1024 rows at a time (mtp.cpp
/// encode_rows), and that feeds drafting, which is decode.  Forward's dense shapes opt in (its only windows above
/// kMaxWindow are prefill chunks); Mtp's do not, so every MTP value stays bitwise what it was.
///
/// The result is NOT bitwise the GEMV's (the sums are ordered differently; no operand is rounded, see the .metal),
/// which prefill tolerates rather than requires to match.  HY4_FLASH_PREFILL_DENSE_MM=0 keeps the GEMV path (an A/B, and
/// the bitwise behaviour).
/// (tests flip it with set_prefill_mm to compare against the GEMV split; the engine only reads the environment)
inline bool& prefill_mm_flag() {
    static bool on = [] {
        const char* e = std::getenv("HY4_FLASH_PREFILL_DENSE_MM");
        if (e == nullptr || *e == 0) return true;
        return std::string(e) == "half" || std::atoi(e) != 0;   // "0" / "off" / "no" all mean the GEMV
    }();
    return on;
}
inline bool prefill_mm_enabled() { return prefill_mm_flag(); }
/// HY4_FLASH_PREFILL_DENSE_MM=half: the MMA rounds both operands to fp16 when it stages them (hy4_flash_mmh), still
/// accumulating in fp32.  Allowed for prefill only, and only because the operands on this path stay far inside
/// fp16's range - HY4_FLASH_MM_RANGE below is how that is checked for a new shape.
inline bool prefill_mm_half() {
    static const bool h = [] {
        const char* e = std::getenv("HY4_FLASH_PREFILL_DENSE_MM");
        return e != nullptr && std::string(e) == "half";
    }();
    return h;
}
inline void set_prefill_mm(bool on) { prefill_mm_flag() = on; }

/// HY4_FLASH_MM_RANGE=1: every prefill MMA shape records max |w| and max |x| over all its dispatches (msl/hy4_flash_mm.metal's
/// range probe), printed to stderr at exit - whether each operand would fit fp16 (max 65504).  The slots live in
/// host memory this owns, so the report does not depend on the Metal context still existing.  A DIAGNOSTIC: it
/// takes a slot per shape while shapes are built (single-threaded, like every other pipeline here), never frees its
/// 16 KiB, and is off unless the variable is set.
class MmRange {
public:
    static MmRange& get() {
        static MmRange r;
        return r;
    }
    bool on() const { return on_; }
    /// A slot for one shape; -1 when the probe is off or full.
    int slot(mtl::Context& cx, const std::string& name) {
        if (!on_) return -1;
        if (!buf_.ptr) {
            void* p = nullptr;
            if (posix_memalign(&p, 16384, kBytes) != 0) return -1;
            std::memset(p, 0, kBytes);
            host_ = (uint32_t*) p;
            buf_ = cx.wrap(p, kBytes);
            std::atexit([] { MmRange::get().report(); });
        }
        if (names_.size() >= kSlots) return -1;
        names_.push_back(name);
        return (int) names_.size() - 1;
    }
    const mtl::Buffer& buffer() const { return buf_; }
    void report() const {
        if (!host_) return;
        for (size_t i = 0; i < names_.size(); ++i) {
            float w, x;
            std::memcpy(&w, &host_[2 * i], 4);
            std::memcpy(&x, &host_[2 * i + 1], 4);
            if (w == 0.0f && x == 0.0f) continue;   // never ran (decode-only shape)
            std::fprintf(stderr, "hy4_flash mm range  %-34s max|w| %.6g  max|x| %.6g%s\n", names_[i].c_str(), w, x,
                         (w > 65504.0f || x > 65504.0f) ? "  EXCEEDS FP16" : "");
        }
    }

private:
    MmRange() {
        const char* e = std::getenv("HY4_FLASH_MM_RANGE");
        on_ = e != nullptr && e[0] == '1';
    }
    static constexpr size_t kSlots = 2048, kBytes = 16384;   // 2 uints a slot
    bool on_ = false;
    mtl::Buffer buf_;
    uint32_t* host_ = nullptr;
    std::vector<std::string> names_;
};

/// The strides between the heads of a per-head weight read in ONE dispatch (gemv.metal function constants 10-12):
/// head h's matrix starts `w` bytes after head h - 1's, its x slice `x` floats after, its y slice `y` floats after.
struct GemvHeads {
    uint32_t w = 0, x = 0, y = 0;
    /// Is this a per-head shape at all?  One predicate, because it decides both which tile is chosen and whether
    /// the head function constants are bound, and two copies of it could drift apart.
    bool any() const { return w != 0 || x != 0 || y != 0; }
};

/// One weight shape's GEMV.  gemv.metal holds acc[NR][8], so a dispatch covers at most 8 tokens: a window of T rows
/// runs as ceil(T / 8) dispatches of the SAME frozen tile, which is arithmetically what decode does for each of
/// those positions.  Q8_0 / F32 / BF16 read units.metal's RAW x plane, so there is no xprep pass and no activation
/// quantization anywhere on this path.
///
/// With `heads`, the shape is ONE head of a per-head weight (wk_b / wv_b) and `append_heads` covers H heads
/// in one dispatch, grid = {row groups, H}.  The tile is gemv_head_tile(type, n_in, n_out) of one head, and thread
/// group (x, h) runs exactly what thread group x of a per-head dispatch bound at head h's offsets ran (gemv.metal's
/// note on function constants 10-12): the same unit sequence per lane, the same lane_sum tree and K-split merge.
/// So it is bitwise the per-head dispatches; the change is that 32 or 64 dispatches of 16-32 thread groups each -
/// too few to fill the GPU - become one.
///
/// What the bitwise argument rests on is the per-ROW correspondence - each row's unit sequence, its reduction tree
/// and its lack of a K merge - NOT a per-thread-group one: a shape whose head tile differs from gemv_tile's gets 32
/// rows a thread group instead of 16, so the rows are grouped differently.
class DenseGemvF32 {
public:
    DenseGemvF32() = default;
    DenseGemvF32(mtl::Context& cx, int lib, int ggml_type, uint32_t n_in, uint32_t n_out, uint32_t row_bytes,
                 GemvHeads heads = {}, bool allow_mm = false)
        : type_(ggml_type),
          // a per-head weight (append_heads, grid = {row groups, H}) takes gemv_head_tile, which gives the narrow
          // band NR 2 for the one geometry its guard admits.  Same tile for every T.  NOTE for anyone reading a
          // kernel timeline: this decides the PIPELINE NAME of the affected shape, so attn_k_b appears as
          // gemv_30_8_2_4_1 rather than gemv_30_8_1_4_1.
          tile_(heads.any() ? strata_hy4_flash_metal::gemv_head_tile(ggml_type, n_in, n_out)
                            : strata_hy4_flash_metal::gemv_tile(ggml_type, n_in, n_out)),
          n_in_(n_in), n_out_(n_out), row_bytes_(row_bytes), heads_(heads) {
        const bool h = heads.any();
        if (h && !strata_hy4_flash_metal::gemv_f32_raw_x(ggml_type)) {
            // gemv.metal offsets the s16 plane like x, which is only right for formats that never read it
            std::fprintf(stderr, "hy4_flash dense gemv: per-head dispatch needs a raw-x format, got ggml type %d\n",
                         ggml_type);
            std::abort();
        }
        for (int t = 0; t < strata_hy4_flash_metal::kHy4FlashGemvMaxTok; ++t) {
            mtl::Constants c;
            c.i(0, t + 1);
            if (h) c.u(10, heads.w).u(11, heads.x).u(12, heads.y);
            p_[t] = cx.pipeline(lib, tile_.engine_name(ggml_type), c);
        }
        // prefill's MMA (hy4_flash_mm.metal): the formats it decodes, K in steps of 32, and x rows / head slices that
        // start on a float4 (every hy4-flash dense matrix qualifies; anything else keeps the GEMV)
        mm_ok_ = allow_mm && (ggml_type == 0 || ggml_type == 8 || ggml_type == 30) && n_in % 32 == 0 &&
                 heads.x % 4 == 0;
        if (mm_ok_ && prefill_mm_enabled()) {   // =0 does not even compile them
            mtl::Constants c;
            if (h) c.u(10, heads.w).u(11, heads.x).u(12, heads.y);
            MmRange& rg = MmRange::get();
            mm_slot_ = rg.slot(cx, "type " + std::to_string(ggml_type) + " " + std::to_string(n_in) + "x" +
                                       std::to_string(n_out) + (h ? " (per head)" : ""));
            if (mm_slot_ >= 0) c.b(13, true).u(14, (uint32_t) mm_slot_);
            pm_ = cx.pipeline(lib, (prefill_mm_half() ? "hy4_flash_mmh_" : "hy4_flash_mm_") + std::to_string(ggml_type), c);
        } else {
            mm_ok_ = false;
        }
    }

    const strata_hy4_flash_metal::GemvTile& tile() const { return tile_; }
    std::string tile_name() const { return tile_.engine_name(type_); }
    uint32_t n_in() const { return n_in_; }
    uint32_t n_out() const { return n_out_; }

    /// Appends the dispatches that write y[t0 .. t0 + n_tok) from x[t0 .. t0 + n_tok).  `x_base` / `y_base` are
    /// extra element offsets into the row (a slice of a packed row).  For a shape built with `heads`, this is head
    /// 0 alone (grid.y = 1); use append_heads for all of them.
    void append(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& W, size_t w_off, const mtl::Buffer& x,
                uint32_t x_stride, const mtl::Buffer& y, uint32_t y_stride, uint32_t t0, uint32_t n_tok,
                size_t x_base = 0, size_t y_base = 0) const {
        append_heads(ds, W, w_off, x, x_stride, y, y_stride, t0, n_tok, 1, x_base, y_base);
    }

    /// `append` for `n_heads` heads of a per-head weight in one dispatch per 8 tokens: head h is the matrix at
    /// w_off + h * heads.w bytes applied to x's elements [x_base + h * heads.x, + n_in) of each token row, into
    /// y's [y_base + h * heads.y, + n_out).
    void append_heads(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& W, size_t w_off, const mtl::Buffer& x,
                      uint32_t x_stride, const mtl::Buffer& y, uint32_t y_stride, uint32_t t0, uint32_t n_tok,
                      uint32_t n_heads, size_t x_base = 0, size_t y_base = 0) const {
        if (n_heads > 1 && heads_.w == 0 && heads_.x == 0 && heads_.y == 0) {
            std::fprintf(stderr, "hy4_flash dense gemv: %u heads asked of a shape built without head strides\n", n_heads);
            std::abort();
        }
        if (n_tok > (uint32_t) strata_hy4_flash_metal::kHy4FlashGemvMaxTok && mm_ok_ && prefill_mm_enabled()) {
            // a prefill chunk: ONE fp32 MMA dispatch for all n_tok rows (decode and MTP verify never get here -
            // their windows are at most kHy4FlashGemvMaxTok rows; MTP's own 1024-row cache rows do not either, because
            // Mtp's shapes are built without allow_mm)
            if ((x_base % 4) != 0 || (x_stride % 4) != 0) {
                std::fprintf(stderr, "hy4_flash dense mm: x is not float4-aligned (base %zu, stride %u)\n", x_base,
                             x_stride);
                std::abort();
            }
            mtl::Dispatch d;
            d.p = &pm_;
            MmArgs a{n_in_, n_out_, n_tok, row_bytes_, x_stride, y_stride};
            const size_t xo = 4 * ((size_t) t0 * x_stride + x_base), yo = 4 * ((size_t) t0 * y_stride + y_base);
            d.args = {mtl::Arg::B(W, w_off), mtl::Arg::B(x, xo), mtl::Arg::B(y, yo), mtl::Arg::V(a)};
            if (mm_slot_ >= 0) d.args.push_back(mtl::Arg::B(MmRange::get().buffer()));
            d.grid[0] = (n_out_ + 63) / 64;
            d.grid[1] = (n_tok + 31) / 32;
            d.grid[2] = n_heads;
            d.tg[0] = 128;
            mtl::require_fit(d, "hy4_flash dense mm");
            ds.push_back(d);
            return;
        }
        const size_t at = ds.size();
        for (uint32_t done = 0; done < n_tok; done += strata_hy4_flash_metal::kHy4FlashGemvMaxTok) {
            const uint32_t nt = std::min<uint32_t>(strata_hy4_flash_metal::kHy4FlashGemvMaxTok, n_tok - done);
            const uint32_t t = t0 + done;
            mtl::Dispatch d;
            d.p = &p_[nt - 1];
            GemvArgs a{n_in_, n_out_, nt, row_bytes_, x_stride, y_stride};
            const size_t xo = 4 * ((size_t) t * x_stride + x_base), yo = 4 * ((size_t) t * y_stride + y_base);
            // buffer 2 is the S16 plane: the RAW formats never read it (units.metal X<F>::load leaves xsum at 0),
            // so x is bound again rather than allocating a plane that would only be zeroed.
            d.args = {mtl::Arg::B(W, w_off), mtl::Arg::B(x, xo), mtl::Arg::B(x, xo), mtl::Arg::B(y, yo),
                      mtl::Arg::V(a)};
            // (the engine's thread group shape, GemvTile::engine_tile; tile_ is the logical tile the bits follow)
            const strata_hy4_flash_metal::GemvTile run = tile_.engine_tile(type_);
            d.grid[0] = (n_out_ + run.rows_per_tg() - 1) / run.rows_per_tg();
            d.grid[1] = n_heads;
            d.tg[0] = run.sg * 32;
            mtl::require_fit(d, "hy4_flash dense gemv");
            ds.push_back(d);
        }
        // the ceil(T / 8) splits of ONE append write disjoint token rows of y and read disjoint rows of x, so
        // they may overlap (relax_after's contract); the first keeps its barrier against whatever wrote x
        relax_after(ds, at, kOverlapGemvSplit);
    }

private:
    int type_ = 0;
    strata_hy4_flash_metal::GemvTile tile_{1, 1, 2, 1};
    uint32_t n_in_ = 0, n_out_ = 0, row_bytes_ = 0;
    GemvHeads heads_;
    mtl::Pipeline p_[strata_hy4_flash_metal::kHy4FlashGemvMaxTok];
    bool mm_ok_ = false;
    int mm_slot_ = -1;
    mtl::Pipeline pm_;
};

// ---- the routed experts -------------------------------------------------------------------------------------------
// mirrors Hy4FlashMoeArgs in engine/metal/src/hy4flash_gguf/msl/hy4_flash_moe.metal
struct MoeArgs {
    uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
    uint32_t gu_row, d_row, n_latent, n_ff, n_tok, topk, n_group;
    uint32_t off_gexp, off_gstart, off_etok, off_ew, off_tokent;
    float limit;
};

/// The int buffer the three MoE kernels read, laid out back to back:
///   gexp [n_group] / gstart [n_group + 1] / etok [n_entry] / ew [n_entry] / tokent [n_tok * topk]
///
/// `ids` / `w` are the router's outputs for the window's tokens, [n_tok][topk].  This is the simple layout, built
/// once per window from a known routing - what the kernel parity tests use.  The engine needs a layout whose
/// section offsets are known BEFORE the routing is (its dispatches are encoded before the GPU has run the router),
/// which is `MoeSlots` below.
struct MoePlan {
    std::vector<int32_t> buf;
    uint32_t n_group = 0, n_entry = 0;
    uint32_t off_gexp = 0, off_gstart = 0, off_etok = 0, off_ew = 0, off_tokent = 0;

    void build(const int32_t* ids, const float* w, uint32_t n_tok, uint32_t topk) {
        std::map<int32_t, std::vector<uint32_t>> by_expert;   // expert id -> selection indices, both ascending
        for (uint32_t i = 0; i < n_tok * topk; ++i) by_expert[ids[i]].push_back(i);
        n_group = (uint32_t) by_expert.size();
        n_entry = n_tok * topk;
        off_gexp = 0;
        off_gstart = n_group;
        off_etok = off_gstart + n_group + 1;
        off_ew = off_etok + n_entry;
        off_tokent = off_ew + n_entry;
        buf.assign(off_tokent + n_tok * topk, 0);
        uint32_t g = 0, c = 0;
        for (const auto& kv : by_expert) {
            buf[off_gexp + g] = kv.first;
            buf[off_gstart + g] = (int32_t) c;
            for (uint32_t sel : kv.second) {
                buf[off_etok + c] = (int32_t) (sel / topk);
                std::memcpy(&buf[off_ew + c], &w[sel], 4);
                buf[off_tokent + sel] = (int32_t) c;
                ++c;
            }
            ++g;
        }
        buf[off_gstart + n_group] = (int32_t) c;
    }
};

/// The engine's plan: the SAME sections, but every offset and every group count is a function of the window size
/// alone, so the dispatches can be encoded before the router has run (the host's turn happens inside an
/// already-committed command buffer).
///
/// Each arena chunk gets a fixed section of `g_max` group slots - the most groups one dispatch can have - and the
/// ones a window does not use are left EMPTY (gstart[g] == gstart[g + 1]), which the kernels' entry loop skips.
/// The streaming arena is up to four MTLBuffers (stream.h) and a kernel takes one `W`, so a chunk's section is
/// what one dispatch covers.
///
/// TWO THINGS A PREFILL CHUNK NEEDS, both of them here rather than in a second plan type:
///
///   * `n_experts` CAPS g_max.  A window's groups cannot outnumber either its selections (n_tok * topk) or the
///     layer's experts, and for 1024 positions the second bound is 32 times smaller - 512 against 16384.  Without
///     the cap every MoE dispatch would launch 16384 group threadgroups per arena chunk to do 512 groups' work.
///     For a decode window of 8 the cap never binds (128 < 512), so the layout is unchanged.
///   * SUB-BATCHES.  `sub` positions of the chunk are one plan, and a chunk of 1024 holds ceil(1024 / sub) of them
///     back to back in the same buffer.  The entry-indexed scratch (h, its prepared plane and ye) is then
///     `sub * topk` rows instead of `n_tok * topk`: 1536-wide rows at 1024 x 16 entries are 635 MB, at 256 x 16
///     they are 152 MB.  The price is that the experts are re-read from the ARENA once per sub-batch - never from
///     the SSD, which `ExpertStream::at_layer` does once per chunk - so this trades GPU bandwidth for footprint
///     and changes no number: an entry's arithmetic does not know which sub-batch it is in (hy4_flash_moe.metal's
///     T-invariance note), and the combine still adds a token's top-k in slot order.
///
/// Sub-batch s owns tokens [s * sub, min(T, (s + 1) * sub)) and numbers its tokens and entries from 0, so the
/// caller offsets x / y by `s * sub` rows and the kernels need no further change.
struct MoeSlots {
    uint32_t n_chunk = 1, g_max = 0, n_entry = 0;      ///< n_entry is per sub-batch
    uint32_t sub = 1, cap_tok = 0, topk = 0;           ///< positions per sub-batch, and what layout() was given
    uint32_t plan_stride = 0;                          ///< ints between sub-batch plans
    uint32_t off_gexp = 0, off_gstart = 0, off_etok = 0, off_ew = 0, off_tokent = 0;
    std::vector<int32_t> buf;

    /// `n_experts` = the layer's routed expert count (0: no cap).  `positions_per_sub` = 0 or >= n_tok: one plan.
    void layout(uint32_t n_tok, uint32_t topk_, uint32_t chunks, uint32_t n_experts = 0,
                uint32_t positions_per_sub = 0) {
        n_chunk = chunks > 0 ? chunks : 1;
        cap_tok = n_tok;
        topk = topk_;
        sub = positions_per_sub > 0 && positions_per_sub < n_tok ? positions_per_sub : n_tok;
        g_max = sub * topk_;
        if (n_experts > 0 && n_experts < g_max) g_max = n_experts;
        n_entry = sub * topk_;
        off_gexp = 0;
        off_gstart = off_gexp + n_chunk * g_max;
        off_etok = off_gstart + n_chunk * (g_max + 1);
        off_ew = off_etok + n_entry;
        off_tokent = off_ew + n_entry;
        plan_stride = off_tokent + n_entry;
        buf.assign((size_t) plan_stride * subs_for(n_tok), 0);
    }

    /// The sub-batches a run of T positions takes, and the positions of sub-batch s.  Pure functions of T, so the
    /// encoder (which runs before the router) and `fill` (which runs after it) cannot disagree.
    uint32_t subs_for(uint32_t T) const { return sub == 0 ? 1 : (T + sub - 1) / sub; }
    uint32_t sub_first(uint32_t s) const { return s * sub; }
    uint32_t sub_tokens(uint32_t T, uint32_t s) const { return std::min(sub, T - s * sub); }

    /// `where(e)` gives the value the kernel multiplies by gu_stride / d_stride (an expert's index in a contiguous
    /// blob, or its slot's 16 KiB page in the arena) and the chunk it lives in.  Returns "" or the problem.
    struct Place {
        int32_t ex = 0;
        int chunk = 0;
    };
    /// `late(e)` (optional): expert e's bytes are not in the arena yet.  Each chunk's groups are
    /// then ordered resident-first, and `n_res[s * n_chunk + ch]` says how many of sub-batch s / chunk ch's groups are
    /// resident - split_plans() cuts the plan there.  Without it the order is (chunk, expert) as it always was.
    /// The order of the groups decides only which h / ye rows an entry uses, never a value: every group is one
    /// expert's entries, computed alone, and the combine adds them back in SLOT order through tokent.
    std::vector<uint32_t> n_res;
    std::string fill(const int32_t* ids, const float* w, uint32_t n_tok, uint32_t topk_,
                     const std::function<Place(int32_t)>& where,
                     const std::function<bool(int32_t)>& late = {}) {
        if (n_tok > cap_tok || topk_ != topk)
            return "the plan was laid out for " + std::to_string(cap_tok) + " x " + std::to_string(topk) +
                   " but was filled with " + std::to_string(n_tok) + " x " + std::to_string(topk_);
        std::fill(buf.begin(), buf.end(), 0);
        n_res.assign((size_t) subs_for(n_tok) * n_chunk, 0);
        std::map<int32_t, Place> place;
        std::map<int32_t, bool> is_late;
        for (uint32_t s = 0; s < subs_for(n_tok); ++s) {
            const uint32_t t0 = sub_first(s), nt = sub_tokens(n_tok, s);
            int32_t* b = buf.data() + (size_t) s * plan_stride;
            // (chunk, late, expert) -> selections, ascending; `late` is always 0 without the predicate
            std::map<std::tuple<int, int, int32_t>, std::vector<uint32_t>> by;
            for (uint32_t i = 0; i < nt * topk_; ++i) {
                const int32_t e = ids[(size_t) t0 * topk_ + i];
                auto it = place.find(e);
                if (it == place.end()) it = place.emplace(e, where(e)).first;
                if (it->second.chunk < 0 || (uint32_t) it->second.chunk >= n_chunk)
                    return "expert " + std::to_string(e) + " is in arena chunk " +
                           std::to_string(it->second.chunk) + ", outside the " + std::to_string(n_chunk) +
                           " the plan was laid out for";
                int lt = 0;
                if (late) {
                    auto jt = is_late.find(e);
                    if (jt == is_late.end()) jt = is_late.emplace(e, late(e)).first;
                    lt = jt->second ? 1 : 0;
                }
                by[{it->second.chunk, lt, e}].push_back(i);
            }
            uint32_t c = 0;   // entries, numbered within the sub-batch so h / ye rows do not collide
            std::vector<uint32_t> g(n_chunk, 0), cend(n_chunk, 0);
            for (const auto& kv : by) {
                const uint32_t ch = (uint32_t) std::get<0>(kv.first);
                if (g[ch] >= g_max) return "more groups in one chunk than the sub-batch can have";
                b[off_gexp + ch * g_max + g[ch]] = place[std::get<2>(kv.first)].ex;
                if (std::get<1>(kv.first) == 0) n_res[(size_t) s * n_chunk + ch] = g[ch] + 1;
                b[off_gstart + ch * (g_max + 1) + g[ch]] = (int32_t) c;
                for (uint32_t sel : kv.second) {
                    b[off_etok + c] = (int32_t) (sel / topk_);
                    std::memcpy(&b[off_ew + c], &w[(size_t) t0 * topk_ + sel], 4);
                    b[off_tokent + sel] = (int32_t) c;
                    ++c;
                }
                ++g[ch];
                cend[ch] = c;   // where THIS chunk's entries end
            }
            // The terminator of each chunk's group list is that chunk's OWN entry end, and so are the slots it did
            // not use (an empty range, so the kernels' `for (k < ne)` runs zero times).
            //
            // IT MUST NOT BE THE WINDOW'S TOTAL `c` FOR EVERY CHUNK: chunk 0's last group would then claim every
            // later chunk's entries as well and recompute them with the wrong expert, coming out right only
            // because the dispatches run in chunk order and the owner writes last - correct by accident, at the
            // cost of the duplicated work, and it forbids the dispatches from overlapping.  `by` is ordered by
            // chunk, so a chunk's entries are contiguous and `cend` is all this needs.
            for (uint32_t ch = 0; ch < n_chunk; ++ch)
                for (uint32_t k = g[ch]; k <= g_max; ++k)
                    b[off_gstart + ch * (g_max + 1) + k] = (int32_t) cend[ch];
        }
        return {};
    }

    /// The plan cut in two at n_res (after a fill with `late`): `a` runs every chunk's resident groups, `b` its late
    /// ones, each with the other's group slots empty (an empty range: the kernels run zero entries there).  etok,
    /// ew and tokent are the full plan's in both, so the entries keep their rows and the combine reads either.
    void split_plans(std::vector<int32_t>& a, std::vector<int32_t>& b) const {
        a = buf;
        b = buf;
        for (uint32_t s = 0; s < (uint32_t) (n_res.size() / n_chunk); ++s) {
            const size_t at = (size_t) s * plan_stride;
            for (uint32_t ch = 0; ch < n_chunk; ++ch) {
                const uint32_t nr = n_res[(size_t) s * n_chunk + ch];
                const int32_t* gx = buf.data() + at + off_gexp + ch * g_max;
                const int32_t* gs = buf.data() + at + off_gstart + ch * (g_max + 1);
                int32_t* ax = a.data() + at + off_gexp + ch * g_max;
                int32_t* as = a.data() + at + off_gstart + ch * (g_max + 1);
                int32_t* bx = b.data() + at + off_gexp + ch * g_max;
                int32_t* bs = b.data() + at + off_gstart + ch * (g_max + 1);
                for (uint32_t k = nr; k <= g_max; ++k) as[k] = gs[nr];          // a: groups [0, nr)
                for (uint32_t k = nr; k < g_max; ++k) ax[k] = 0;
                for (uint32_t k = 0; k + nr <= g_max; ++k) bs[k] = gs[k + nr];  // b: groups [nr, g_max), shifted
                for (uint32_t k = 0; k < g_max; ++k) bx[k] = k + nr < g_max ? gx[k + nr] : 0;
                for (uint32_t k = g_max - nr + 1; k <= g_max && nr > 0; ++k) bs[k] = gs[g_max];
            }
        }
    }

    /// `a` addressed at sub-batch `s`, arena chunk `ch`.  n_group is always g_max: the empty slots cost one
    /// entry-count read.  The caller has already put the sub-batch's token count in `a.n_tok`.
    MoeArgs for_chunk(const MoeArgs& a, uint32_t ch, uint32_t s = 0) const {
        const uint32_t at = s * plan_stride;
        MoeArgs r = a;
        r.n_group = g_max;
        r.off_gexp = at + off_gexp + ch * g_max;
        r.off_gstart = at + off_gstart + ch * (g_max + 1);
        r.off_etok = at + off_etok;
        r.off_ew = at + off_ew;
        r.off_tokent = at + off_tokent;
        return r;
    }
};

/// One ordinal's expert GEMVs for one window.  The dispatch order:
///
///   xprep(x)                -> the prepared activation plane the gate/up format needs (skipped for a raw format)
///   hy4_flash_moe_gu              -> h[entry][n_ff], the clamped SwiGLU
///   xprep(h)                -> the plane the down format needs
///   hy4_flash_moe_down            -> ye[entry][n_latent], already multiplied by the routing weight
///   hy4_flash_moe_combine_latent  -> y[token][n_latent], the top-k added in SLOT order
class MoeF32 {
public:
    MoeF32() = default;
    /// `down_tile` (tests only) runs the down with another instantiated tile in place of moe_down_tile's - the
    /// bitwise check of a tile change compares the two.
    MoeF32(mtl::Context& cx, int lib, int gu_type, int d_type, uint32_t n_latent, uint32_t n_ff,
           const mtl::Constants& probe = {}, const strata_hy4_flash_metal::GemvTile* down_tile = nullptr)
        : gu_type_(gu_type), d_type_(d_type), n_latent_(n_latent), n_ff_(n_ff),
          tile_(strata_hy4_flash_metal::moe_tile(gu_type, n_latent)),
          dn_tile_(down_tile ? *down_tile : strata_hy4_flash_metal::moe_down_tile(d_type, n_ff)) {
        p_xprep_ = cx.pipeline(lib, "xprep");
        p_gu_ = cx.pipeline(lib, strata_hy4_flash_metal::moe_gu_name(gu_type, n_latent), probe);
        p_dn_ = cx.pipeline(lib, down_name(), probe);
        p_comb_ = cx.pipeline(lib, "hy4_flash_moe_combine_latent");
    }

    std::string gu_name() const { return strata_hy4_flash_metal::moe_gu_name(gu_type_, n_latent_); }
    std::string down_name() const { return strata_hy4_flash_metal::moe_name("down", d_type_, dn_tile_); }
    int rows_per_tg() const { return tile_.rows_per_tg(); }        ///< the gate/up's
    int down_rows_per_tg() const { return dn_tile_.rows_per_tg(); }
    bool gu_prep() const { return strata_hy4_flash_metal::xprep_needed(gu_type_); }
    bool d_prep() const { return strata_hy4_flash_metal::xprep_needed(d_type_); }

    // Scratch sizes in floats, for a window of n_tok tokens with topk slots each.
    size_t xp_floats(uint32_t n_tok) const { return strata_hy4_flash_metal::xprep_floats(n_latent_, n_tok); }
    size_t hp_floats(uint32_t n_entry) const { return strata_hy4_flash_metal::xprep_floats(n_ff_, n_entry); }

    void append_xprep_x(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& x, size_t x_off, const mtl::Buffer& xp,
                        uint32_t n_tok) const {
        if (!gu_prep()) return;
        append_xprep(ds, x, x_off, xp, n_latent_, n_tok, "xprep (gate/up)");
    }
    void append_xprep_h(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& h, const mtl::Buffer& hp,
                        uint32_t n_entry) const {
        if (!d_prep()) return;
        append_xprep(ds, h, 0, hp, n_ff_, n_entry, "xprep (down)");
    }

    void append_gu(std::vector<mtl::Dispatch>& ds, const MoeArgs& a, const mtl::Buffer& W, const mtl::Buffer& plan,
                   const mtl::Buffer& x, size_t x_off, const mtl::Buffer& xp, const mtl::Buffer& h) const {
        if (a.n_group == 0) return;
        const size_t plane = (size_t) a.n_tok * n_latent_;
        const mtl::Buffer& gb = gu_prep() ? xp : x;
        const size_t gx = gu_prep() ? 4 * (size_t) strata_hy4_flash_metal::xprep_kind(gu_type_) * plane : x_off;
        const size_t gs = gu_prep() ? 4 * 4 * plane : x_off;   // S16 follows the four planes
        mtl::Dispatch d;
        d.p = &p_gu_;
        d.args = {mtl::Arg::B(W), mtl::Arg::B(gb, gx), mtl::Arg::B(h), mtl::Arg::B(plan), mtl::Arg::V(a),
                  mtl::Arg::B(gb, gs)};
        d.grid[0] = (n_ff_ + tile_.rows_per_tg() - 1) / tile_.rows_per_tg();
        d.grid[1] = launch_groups(a);
        d.tg[0] = tile_.sg * 32;
        mtl::require_fit(d, "hy4_flash_moe_gu");
        ds.push_back(d);
    }

    void append_down(std::vector<mtl::Dispatch>& ds, const MoeArgs& a, const mtl::Buffer& W, const mtl::Buffer& plan,
                     const mtl::Buffer& h, const mtl::Buffer& hp, const mtl::Buffer& ye) const {
        if (a.n_group == 0) return;
        const uint32_t ne = a.n_tok * a.topk;
        const size_t plane = (size_t) ne * n_ff_;
        const mtl::Buffer& db = d_prep() ? hp : h;
        const size_t dx = d_prep() ? 4 * (size_t) strata_hy4_flash_metal::xprep_kind(d_type_) * plane : 0;
        const size_t dsum = d_prep() ? 4 * 4 * plane : 0;
        mtl::Dispatch d;
        d.p = &p_dn_;
        d.args = {mtl::Arg::B(W), mtl::Arg::B(db, dx), mtl::Arg::B(ye), mtl::Arg::B(plan), mtl::Arg::V(a),
                  mtl::Arg::B(db, dsum)};
        d.grid[0] = (n_latent_ + dn_tile_.rows_per_tg() - 1) / dn_tile_.rows_per_tg();   // the DOWN's tile
        d.grid[1] = launch_groups(a);
        d.tg[0] = dn_tile_.sg * 32;
        mtl::require_fit(d, "hy4_flash_moe_down");
        ds.push_back(d);
    }

    void append_combine(std::vector<mtl::Dispatch>& ds, const MoeArgs& a, const mtl::Buffer& ye,
                        const mtl::Buffer& plan, const mtl::Buffer& y, size_t y_off) const {
        mtl::Dispatch d;
        d.p = &p_comb_;
        d.args = {mtl::Arg::B(ye), mtl::Arg::B(y, y_off), mtl::Arg::B(plan), mtl::Arg::V(a)};
        d.tg[0] = 256;
        d.grid[0] = (n_latent_ + 255) / 256;
        d.grid[1] = a.n_tok;
        mtl::require_fit(d, "hy4_flash_moe_combine_latent");
        ds.push_back(d);
    }

    /// The whole chain for one sub-window on a single-buffer expert blob (what the kernel parity tests use).
    void append(std::vector<mtl::Dispatch>& ds, const MoeArgs& a, const mtl::Buffer& W, const mtl::Buffer& plan,
                const mtl::Buffer& x, size_t x_off, const mtl::Buffer& xp, const mtl::Buffer& h,
                const mtl::Buffer& hp, const mtl::Buffer& ye, const mtl::Buffer& y, size_t y_off) const {
        append_xprep_x(ds, x, x_off, xp, a.n_tok);
        append_gu(ds, a, W, plan, x, x_off, xp, h);
        append_xprep_h(ds, h, hp, a.n_tok * a.topk);
        append_down(ds, a, W, plan, h, hp, ye);
        append_combine(ds, a, ye, plan, y, y_off);
    }

    static uint32_t n_entry_of(const MoeArgs& a) { return a.n_tok * a.topk; }

    /// THE LAUNCH BOUND.  A plan's section holds `n_group` slots, but its groups are PACKED from slot 0
    /// (MoeSlots::fill numbers a chunk's groups 0, 1, ...; MoePlan::build has exactly as many as it uses) and there
    /// cannot be more of them than the window's selections, n_tok x topk.  The decode plan is laid out for the
    /// widest window (8 x 16 = 128 slots a chunk) whatever T is, so without this bound a T = 1 window would launch
    /// 128 group rows a chunk to fill at most 16.  The slots past n_tok x topk are empty in every plan, so not
    /// launching them moves no value; the kernels also leave an empty group before their table load
    /// (hy4_flash_moe.metal).
    static uint32_t launch_groups(const MoeArgs& a) {
        const uint32_t ne = a.n_tok * a.topk;
        return HY4_FLASH_MOE_FULL_GRID() || ne == 0 ? a.n_group : std::min(a.n_group, ne);
    }
    /// HY4_FLASH_MOE_FULL_GRID=1: the unbounded grid (every slot), for an A/B.
    static bool HY4_FLASH_MOE_FULL_GRID() {
        static const bool on = [] { const char* e = std::getenv("HY4_FLASH_MOE_FULL_GRID"); return e != nullptr && std::atoi(e) != 0; }();
        return on;
    }

private:
    void append_xprep(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& src, size_t off, const mtl::Buffer& dst,
                      uint32_t n, uint32_t rows, const char* where) const {
        mtl::Dispatch d;
        d.p = &p_xprep_;
        d.args = {mtl::Arg::B(src, off), mtl::Arg::B(dst), mtl::Arg::V(n), mtl::Arg::V(rows)};
        d.tg[0] = 256;
        d.grid[0] = (uint32_t) (((size_t) rows * (n / 16) + 255) / 256);
        mtl::require_fit(d, where);
        ds.push_back(d);
    }

    int gu_type_ = 0, d_type_ = 0;
    uint32_t n_latent_ = 0, n_ff_ = 0;
    strata_hy4_flash_metal::GemvTile tile_{8, 1, 4, 1};      ///< gate/up
    strata_hy4_flash_metal::GemvTile dn_tile_{8, 1, 4, 1};   ///< down (moe_down_tile)
    mtl::Pipeline p_xprep_, p_gu_, p_dn_, p_comb_;
};

/// The routed experts of a PREFILL chunk as a grouped GEMM on the simdgroup matrix units, fp32 throughout
/// (hy4_flash_moe_mm.metal).  It reads the same plan as MoeF32 (groups of one expert's entries) plus a TILE LIST - one
/// (group, first entry) per TM entries of a group - and writes the same h / ye, so MoeF32::append_combine adds the
/// top-k in slot order exactly as for decode.  Not bitwise with decode (the summation order differs); the values
/// multiplied are the same fp32 weights and activations.  IQ2_XXS gate/up, Q4_K down.
struct Hy4FlashMmArgs {
    uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
    uint32_t gu_row, d_row, n_latent, n_ff, n_tile;
    uint32_t off_gexp, off_gstart, off_etok, off_ew;
    float limit;
};

/// TM entries x TN outputs a threadgroup, split over SGM x SGN simdgroups; half: f16 operands, f32 sums.
/// mpp: Metal 4 matmul2d (hy4_flash_mm2_*, SGM * SGN simdgroups as its execution scope) instead of simdgroup_matrix.
struct MoeMmTile { int tm = 64, tn = 32, sgm = 4, sgn = 1; bool half = true; bool mpp = false; int tk = 32; };

class MoeMmF32 {
public:
    using Tile = MoeMmTile;
    MoeMmF32() = default;
    MoeMmF32(mtl::Context& cx, int lib, uint32_t n_latent, uint32_t n_ff, Tile gu = {}, Tile dn = {},
             const mtl::Constants& probe = {})
        : n_latent_(n_latent), n_ff_(n_ff), gu_(gu), dn_(dn) {
        if (gu.tm != dn.tm) throw std::runtime_error("MoeMmF32: gate/up and down share one tile list, so one TM");
        // The K loop steps by TK and mm_stage_x reads whole float4s / whole quant blocks, so a width that is not a
        // multiple of TK would read past a row.  1536 over TK 32 / 64 is fine; a future width or tile is not
        // silently wrong.
        if (n_latent % (uint32_t) gu.tk != 0 || n_ff % (uint32_t) dn.tk != 0)
            throw std::runtime_error("MoeMmF32: TK must divide n_latent and n_ff");
        auto nm = [](const char* k, const Tile& t) {
            const std::string ty = t.half ? "_half" : "_float";
            if (t.mpp)
                return std::string("hy4_flash_mm2_") + k + "_" + std::to_string(t.tm) + "_" + std::to_string(t.tn) + "_" +
                       std::to_string(t.sgm * t.sgn) + ty + (t.tk == 32 ? "" : "_k" + std::to_string(t.tk));
            return std::string("hy4_flash_mm_") + k + "_" + std::to_string(t.tm) + "_" + std::to_string(t.tn) + "_" +
                   std::to_string(t.sgm) + "_" + std::to_string(t.sgn) + ty;
        };
        p_gu_ = cx.pipeline(lib, nm("gu", gu), probe);
        p_dn_ = cx.pipeline(lib, nm("down", dn), probe);
    }
    int tm() const { return gu_.tm; }
    const std::string& gu_name() const { return p_gu_.name; }
    const std::string& down_name() const { return p_dn_.name; }

    /// The tile list of a plan section: for each group g with ne entries, ceil(ne / TM) tiles (g, m0).  Appended
    /// to `out`; returns the count.  At most n_group + n_entry / TM tiles.
    static uint32_t tiles(const int32_t* gstart, uint32_t n_group, int tm, std::vector<int32_t>& out) {
        uint32_t n = 0;
        for (uint32_t g = 0; g < n_group; ++g)
            for (int32_t m0 = 0; m0 < gstart[g + 1] - gstart[g]; m0 += tm, ++n) {
                out.push_back((int32_t) g);
                out.push_back(m0);
            }
        return n;
    }

    /// The same list written straight into `out` (room for `cap` tiles), the tail filled with the padding entry
    /// (-1, 0) the kernels skip - so a grid longer than the real tile count cannot read a stale tile.  Returns the
    /// real count, or `cap + 1` if the section needs more than `cap` (the caller's size is then wrong).
    static uint32_t tiles_into(const int32_t* gstart, uint32_t n_group, int tm, int32_t* out, uint32_t cap) {
        uint32_t n = 0;
        for (uint32_t g = 0; g < n_group; ++g)
            for (int32_t m0 = 0; m0 < gstart[g + 1] - gstart[g]; m0 += tm, ++n) {
                if (n >= cap) return cap + 1;
                out[2 * n] = (int32_t) g;
                out[2 * n + 1] = m0;
            }
        for (uint32_t i = n; i < cap; ++i) {
            out[2 * i] = -1;
            out[2 * i + 1] = 0;
        }
        return n;
    }

    /// The worst case for one plan section: every group contributes a partial tile.
    static uint32_t tile_cap(uint32_t g_max, uint32_t n_entry, int tm) {
        return g_max + (n_entry + (uint32_t) tm - 1) / (uint32_t) tm;
    }

    /// x: the window's gate/up input [n_tok][n_latent] (raw f32, IQ2_XXS reads no prepared plane); h: [entry][n_ff].
    void append_gu(std::vector<mtl::Dispatch>& ds, const Hy4FlashMmArgs& a, const mtl::Buffer& W, const mtl::Buffer& plan,
                   const mtl::Buffer& tiles, const mtl::Buffer& x, size_t x_off, const mtl::Buffer& h,
                   size_t tiles_off = 0) const {
        if (a.n_tile == 0) return;
        mtl::Dispatch d;
        d.p = &p_gu_;
        d.args = {mtl::Arg::B(W), mtl::Arg::B(x, x_off), mtl::Arg::B(h), mtl::Arg::B(plan), mtl::Arg::V(a),
                  mtl::Arg::B(tiles, tiles_off)};
        d.grid[0] = (n_ff_ + (uint32_t) gu_.tn - 1) / (uint32_t) gu_.tn;
        d.grid[1] = a.n_tile;
        d.tg[0] = (uint32_t) (gu_.sgm * gu_.sgn) * 32;
        mtl::require_fit(d, "hy4_flash_mm_gu");
        ds.push_back(d);
    }
    /// h: [entry][n_ff] f32; ye: [entry][n_latent], multiplied by the entry's routing weight.
    void append_down(std::vector<mtl::Dispatch>& ds, const Hy4FlashMmArgs& a, const mtl::Buffer& W,
                     const mtl::Buffer& plan, const mtl::Buffer& tiles, const mtl::Buffer& h,
                     const mtl::Buffer& ye, size_t tiles_off = 0) const {
        if (a.n_tile == 0) return;
        mtl::Dispatch d;
        d.p = &p_dn_;
        d.args = {mtl::Arg::B(W), mtl::Arg::B(h), mtl::Arg::B(ye), mtl::Arg::B(plan), mtl::Arg::V(a),
                  mtl::Arg::B(tiles, tiles_off)};
        d.grid[0] = (n_latent_ + (uint32_t) dn_.tn - 1) / (uint32_t) dn_.tn;
        d.grid[1] = a.n_tile;
        d.tg[0] = (uint32_t) (dn_.sgm * dn_.sgn) * 32;
        mtl::require_fit(d, "hy4_flash_mm_down");
        ds.push_back(d);
    }

private:
    uint32_t n_latent_ = 0, n_ff_ = 0;
    Tile gu_, dn_;
    mtl::Pipeline p_gu_, p_dn_;
};

}  // namespace strata::hy4_flash
