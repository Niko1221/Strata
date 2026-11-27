// engine/metal/src/hy4flash_gguf/experts.cpp - see experts.h.
#include "experts.h"

#include "engine/arena_budget.h"
#include "engine/sysmem.h"

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace strata::hy4_flash {

using strata::metal_engine::ArenaMem;
using strata::metal_engine::ProcMem;

namespace {

constexpr double kGB = 1e9;
constexpr int kHy4FlashReadAhead = 16;   // the read-ahead's depth when STRATA_METAL_PREFETCH is unset (make_arena)
constexpr double kGpuMargin = 1.0 * kGB;   // left free of the Metal working set beside the arena and the weights

double env_gb(const char* name, double miss) {
    const char* e = std::getenv(name);
    return e && *e ? std::atof(e) : miss;
}

// "Off" in every spelling a script might write.  Accepting only the literal "0" would leave HY4_FLASH_PREFETCH=false
// silently ON, so a sweep that trusted the variable would run a configuration it did not ask for.
bool env_off(const char* name) {
    const char* e = std::getenv(name);
    if (e == nullptr || *e == '\0') return false;
    std::string v(e);
    for (char& c : v) c = (char) std::tolower((unsigned char) c);
    return v == "0" || v == "false" || v == "no" || v == "off";
}

}  // namespace

ArenaPlan plan_arena(const Model& m, const mtl::Context& cx) {
    ArenaPlan p;
    p.phys = metal_engine::sys_phys();
    p.working_set = (double) cx.recommended_working_set();
    const ProcMem pm = metal_engine::proc_mem();

    ArenaMem am;
    am.phys = p.phys;
    am.working_set = p.working_set;
    am.allocated = (double) cx.allocated();
    am.proc_footprint = pm.footprint;
    am.filecache_min = metal_engine::sys_filecache_min();
    am.available = metal_engine::sys_avail(p.phys - am.allocated);
    p.base = metal_engine::footprint_base(am);
    p.default_cap = metal_engine::footprint_cap(am);

    // The three hard limits.  The DEFAULT rule is deliberately not one of them: its `leave` is
    // max(arena_keep, 40% of the RAM, the file-cache floor + 12 GB), and the file-cache part of that does not
    // apply to this engine - it reads dense, experts and the Engram table with F_NOCACHE / O_DIRECT preads, so
    // nothing of its own is ever in the file cache.  What is left to macOS here is `arena_keep` (7 GB, or 15% of
    // the RAM when that is more) plus the n-gram row cache's reserve.
    p.gpu_room = p.working_set - p.base - kGpuMargin;
    // What macOS can give NOW, not what the machine has: `phys` ignores every other process, and a run started
    // beside one that holds 20 GB would plan an arena that cannot exist.  This is the same `ram` term
    // `arena_budget()` uses, and `arena_shrink` re-checks it just before the arena is committed.
    p.avail = am.available;
    p.ram_room = am.available - metal_engine::arena_keep(am) - metal_engine::arena_ple_reserve(am);
    p.expert_total = 0;
    for (int o = 0; o < m.stream.n_layers(); ++o)
        p.expert_total += (double) m.stream.expert_bytes(o) * (double) m.stream.n_experts();

    const double hard = std::min({p.gpu_room, p.ram_room, p.expert_total});
    p.arena = std::min(hard, kArenaObservedGB * kGB);
    p.bound = p.arena == p.gpu_room ? "the GPU working set"
              : p.arena == p.ram_room ? "the RAM macOS would be left without"
              : p.arena == p.expert_total ? "the experts themselves (a bigger arena would hold nothing more)"
                                          : "the largest value observed to pass (experts.h kArenaObservedGB)";
    if (const double g = env_gb("STRATA_METAL_ARENA_GB", -1); g >= 0) {
        p.arena = g * kGB;
        p.from_env = true;
        p.bound = "STRATA_METAL_ARENA_GB";
    }
    if (p.arena < 1.0 * kGB) p.arena = 1.0 * kGB;
    p.over_default_cap = p.arena + p.base + metal_engine::arena_ple_reserve(am) > p.default_cap + 1e6;
    return p;
}

std::string arena_line(const ArenaPlan& p) {
    char b[1024];
    std::snprintf(b, sizeof b,
                  "expert arena   %.2f GB (%s), %.2f GB already resident, footprint cap %.2f GB%s; "
                  "RAM %.2f GB (%.2f GB macOS could give now), GPU working set %.2f GB; "
                  "limits: GPU %.2f, RAM %.2f, experts %.2f GB",
                  p.arena / kGB, p.bound, p.base / kGB, p.default_cap / kGB,
                  p.over_default_cap ? " - OVER THE DEFAULT CAP, which is this engine's deliberate choice "
                                       "(ADR §10.1: the cap's file-cache reserve does not apply to F_NOCACHE "
                                       "reads) and NOT a measured-safe value: watch swap (rule R6.1)"
                                     : " (under the default cap)",
                  p.phys / kGB, p.avail / kGB, p.working_set / kGB, p.gpu_room / kGB, p.ram_room / kGB,
                  p.expert_total / kGB);
    return b;
}

std::string open_experts(Model& m, bool with_mtp) {
    // The stream's layer index is the EXPERT ORDINAL.  Ordinal o lives in block `m.experts[o].block_id`,
    // which for this model is o + 1 - block 0 is the dense FFN - and the last ordinal is the MTP block, whose
    // experts this engine does not open at all (they are BF16 and nothing reads them).
    const int n = (int) m.experts.size();
    std::vector<std::string> prefix;
    for (int o = 0; o < n; ++o) {
        const int32_t b = m.experts[(size_t) o].block_id;
        if (!with_mtp && b >= m.geo.trunk_blocks) break;
        prefix.push_back("blk." + std::to_string(b) + ".");
    }
    if (prefix.empty()) return "this model has no routed experts to stream";
    std::string err;
    if (!m.stream.open(m.tensors, prefix, err)) return err;
    if (m.stream.n_layers() != (int) prefix.size())
        return "the stream opened " + std::to_string(m.stream.n_layers()) + " ordinals, not " +
               std::to_string(prefix.size());
    // The ordinals the stream computed must be the byte ranges the model's own table has: an off-by-one here
    // reads another block's experts and nothing downstream would notice (model.h's warning).
    for (int o = 0; o < m.stream.n_layers(); ++o) {
        const ExpertSlices& e = m.experts[(size_t) o];
        const uint64_t want[3] = {e.off[kGate], e.off[kUp], e.off[kDown]};
        for (int k = 0; k < 3; ++k)
            if (m.stream.slice_offset(o, 0, k) != want[k] || m.stream.slice_bytes(o, k) != e.stride[k])
                return "ordinal " + std::to_string(o) + " (block " + std::to_string(e.block_id) +
                       "): the stream's slice " + std::to_string(k) + " is not the model table's";
    }
    return {};
}

std::string make_arena(Model& m, mtl::Context& cx, const ArenaPlan& p, const std::string& profile,
                       int min_slots) {
    // hy4-flash's own read-ahead settings.  Both change WHEN an expert is read, never which one a window uses, so
    // every token is the same.
    //   depth   kHy4FlashReadAhead per token of the next ordinal's predicted top (STRATA_METAL_PREFETCH overrides);
    //   order   HY4_FLASH_AHEAD_EARLY=1 issues it before the demand wait instead of after.  Off by default: issued
    //           early, the read-ahead competes with the demand reads for the SSD.
    m.stream.default_read_ahead(kHy4FlashReadAhead);
    {
        const char* e = std::getenv("HY4_FLASH_AHEAD_EARLY");
        m.stream.set_ahead_before_wait(e != nullptr && *e != '\0' && !env_off("HY4_FLASH_AHEAD_EARLY"));
    }
    // The live guards (arena_shrink / arena_grow) are built on an ArenaMem WITHOUT arena_gb set even when the
    // size was chosen here, so a machine that is short of memory at this moment still shrinks the layout and
    // stops the stepped fill.  An explicit STRATA_METAL_ARENA_GB switches them off, as it does for Qwen: that is
    // the user saying "fill it anyway" (arena_budget.h).
    ArenaMem am;
    am.phys = p.phys;
    am.working_set = p.working_set;
    am.allocated = (double) cx.allocated();
    am.proc_footprint = metal_engine::proc_mem().footprint;
    am.filecache_min = metal_engine::sys_filecache_min();
    am.available = metal_engine::sys_avail(p.phys - am.allocated);
    if (p.from_env) am.arena_gb = p.arena / kGB;

    double budget = p.arena;
    std::string err;
    bool minimal = false;
    for (int tries = 0;; ++tries) {
        const double planned = budget;
        std::function<uint64_t()> recheck = [&]() -> uint64_t {
            return (uint64_t) metal_engine::arena_shrink(am, planned, metal_engine::sys_avail(planned));
        };
        // min_slots: 64 for decode, 576 once a prompt chunk can run.  A chunk's layer wants one slot per
        // expert of that layer at once (512), and `victim()` exits the process when a class runs out - so the
        // floor has to cover a whole layer plus a margin, the way window.cpp's prompt path does it.
        if (m.stream.make_arena(cx, (uint64_t) budget, profile, min_slots, err,
                                minimal ? std::function<uint64_t()>() : recheck,
                                [&](uint64_t step) {
                                    return metal_engine::arena_grow(am, metal_engine::sys_avail(am.available),
                                                                    (double) step);
                                }))
            break;
        if (m.stream.shrink_to() == 0 || tries == 4) return err;
        if ((double) m.stream.shrink_to() + 64.0 * (1 << 20) >= budget) minimal = true;
        std::fprintf(stderr, "strata_hy4_flash_metal: less memory is free than when the arena was planned: %.2f -> %.2f GB\n",
                     budget / kGB, m.stream.shrink_to() / kGB);
        budget = (double) m.stream.shrink_to();
    }
    // The cross-layer read-ahead is the only remaining lever on the SSD wait once the arena is as big as it can
    // be, so it is ON unless told otherwise - and the engine prints which, so a run's configuration is never in
    // doubt.
    std::fprintf(stderr, "strata_hy4_flash_metal: cross-layer read-ahead %s\n",
                 prefetch_enabled() ? "on (the next ordinal's router runs on this ordinal's FFN input; it changes "
                                      "WHEN bytes are read, never which - HY4_FLASH_PREFETCH=0 turns it off)"
                                    : "OFF (HY4_FLASH_PREFETCH=0 or STRATA_METAL_PREFETCH=0)");
    // One layer of fence lag: the engine hands the GPU a whole window's command buffer and the host's turn at
    // ordinal o + 1 can begin while the GPU is still inside ordinal o, so a slot stays safe for one handshake
    // longer than the cut alone would guarantee (stream.h set_fence_lag).
    m.stream.set_fence_lag(1);
    return {};
}

bool prefetch_enabled() { return !env_off("HY4_FLASH_PREFETCH") && !env_off("STRATA_METAL_PREFETCH"); }

void at_cut(Model& m, int o, int T, int topk, const int32_t* ids, const float* pred, const float* pred2) {
    m.stream.at_layer(o, T, topk, ids, prefetch_enabled() ? pred : nullptr, prefetch_enabled() ? pred2 : nullptr);
}

}  // namespace strata::hy4_flash
