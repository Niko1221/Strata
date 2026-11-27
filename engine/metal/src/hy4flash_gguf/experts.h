// engine/metal/src/hy4flash_gguf/experts.h - the routed experts' residency for the hy4-flash Metal engine.
//
// WHAT THIS LAYER IS.  `metal/src/engine/stream.h` already owns the arena: the slots, the LRU, the I/O threads,
// the smap the kernels read and the read-ahead.  What is hy4-flash's own is (a) the ordinal -> block mapping its
// `open` now takes, (b) how big the arena may be on this machine, which for this model is the single most
// important number there is, and (c) the host's turn at a cut, with the read-ahead switch and the counters.
//
// THE ARENA IS THE WHOLE GAME.  The trunk's routed experts are 76.5 GB (Q3_K) or 53.4 GB (IQ2_AH) and the
// machine has 64 GiB, so every token reads whatever is not resident from the SSD: the arena's size IS the byte
// hit rate, and nothing else in this engine moves that term as far.  So the arena is taken as LARGE AS THE
// HARDWARE ALLOWS rather than as large as the default rule allows: `arena_budget.h`'s `footprint_cap` leaves
// 40% of the RAM to macOS and its file cache, and that file-cache reserve does not apply here at all - dense,
// experts and the Engram table are all read with F_NOCACHE or O_DIRECT preads, so this engine puts nothing in
// the file cache.
//
// WHAT IS NOT SAFE ABOUT THAT, said plainly: a big arena leaves macOS little, and macOS answers a shortage by
// compressing and then swapping anonymous memory - which the arena is.  A run whose swap grows by a couple of
// gigabytes is to be stopped.  The size here is therefore not a proven safe value, and the engine prints what
// it chose so a run can be read against it.
#pragma once

#include "model.h"
#include "mtl_rt.h"

#include <cstdint>
#include <string>

namespace strata::hy4_flash {

/// How big the arena may be, as the three hard limits and which of them decided.
struct ArenaPlan {
    double phys = 0;           ///< bytes of RAM
    double working_set = 0;    ///< the Metal device's recommended working set
    double base = 0;           ///< what the process already holds (dense weights, buffers, host memory)
    double default_cap = 0;    ///< arena_budget.h's footprint_cap: the DEFAULT rule's ceiling on all of it
    double gpu_room = 0;       ///< working_set - base - a 1 GB margin
    double ram_room = 0;       ///< phys - base - what macOS keeps - the n-gram reserve
    double expert_total = 0;   ///< every routed expert of the ordinals that were opened
    double arena = 0;          ///< what will be asked for
    const char* bound = "";    ///< which limit decided
    double avail = 0;          ///< what macOS could give when the plan was made (arena_avail)
    bool from_env = false;     ///< STRATA_METAL_ARENA_GB said so
    bool over_default_cap = false;
};

/// The largest arena OBSERVED TO PASS on this machine - not a proven ceiling.
///
/// It was established by filling the arena from the profile with the dense weights resident and watching swap:
/// larger sizes drove the machine into swap growth or memory pressure, this one did not.
///
/// WHAT THAT DOES NOT ESTABLISH: nothing between this value and the next size up was tried, the runs were
/// loading-path runs rather than a service soak, and they carried NO KV cache, indexer cache or prefill
/// scratch - so the real headroom in a full run is smaller than it looks here.  Treat this as "the biggest
/// value seen to pass the loading path", re-check it as the forward pass grows, and keep watching swap.
///
/// The fill is the dangerous part: tens of gigabytes of anonymous memory written at full speed before `mlock`
/// wires it, which is when macOS compresses and then swaps it.  `STRATA_METAL_ARENA_GB` overrides this in
/// either direction and turns off the live `arena_shrink`/`arena_grow` guards, which is the user's choice
/// (arena_budget.h).
constexpr double kArenaObservedGB = 28.0;

/// Decides the arena's size.  Reads the machine (sysctl, host_statistics64, task_info), allocates nothing.
/// Call it AFTER `load_dense`, so `base` is the real resident footprint rather than a guess.
ArenaPlan plan_arena(const Model& m, const mtl::Context& cx);

/// The one line the engine prints at start: the arena it got, the bytes already resident, the default cap and
/// whether this goes over it.
std::string arena_line(const ArenaPlan& p);

/// `stream.open` with hy4-flash's ordinal -> block prefix mapping.  `with_mtp` includes the MTP block's ordinal,
/// which decode never reads.  Returns "" or the problem.
std::string open_experts(Model& m, bool with_mtp = false);

/// Lays out and fills the arena.  `profile` is an expert-profile.bin in ORDINAL numbering (its `n_layers` field
/// must be the ordinal count, not the block count); a missing one leaves the fill layer-interleaved.
///
/// `min_slots` is the floor on each size class's slot count.  64 is enough for decode (a window asks for at most
/// 8 x 16 experts of one layer); a PREFILL chunk asks for every expert of the layer at once, so a build that can
/// run one passes 576 - one per expert plus a margin, as metal/src/engine/window.cpp does for Qwen's prompt path.
/// Too low a floor is not a slow path but a dead one: `ExpertStream::victim` ends the process when a class has no
/// unpinned slot left.
std::string make_arena(Model& m, mtl::Context& cx, const ArenaPlan& p, const std::string& profile,
                       int min_slots = 64);

/// The host's turn at a cut, i.e. after expert ordinal `o`'s router.
///
/// `ids` are the window's T x topk routed expert ids.  `pred` is ordinal o + 1's router logits for the window's
/// tokens (T x n_experts, a 512-float stride), computed from THIS ordinal's FFN input: the experts the next
/// layer will want are then read while the GPU computes this one.  It changes WHEN bytes are read and never
/// WHICH, so it cannot move a number; `HY4_FLASH_PREFETCH=0` passes nullptr instead and the engine says so at start.
void at_cut(Model& m, int o, int T, int topk, const int32_t* ids, const float* pred, const float* pred2 = nullptr);

/// Whether `at_cut` will pass `pred` on.  `HY4_FLASH_PREFETCH=0` turns the cross-layer read-ahead off;
/// `STRATA_METAL_PREFETCH=0` turns it off inside the stream as well.
bool prefetch_enabled();

}  // namespace strata::hy4_flash
