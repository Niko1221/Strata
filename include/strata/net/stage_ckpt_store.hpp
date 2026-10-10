// include/strata/net/stage_ckpt_store.hpp - a stage worker's part of the main process's conversation checkpoints
// (docs/remote-stage/CHECKPOINTS.md): its own layers' running state at a position, under the main process's id.
//
// The main process alone decides what is kept.  Every Reset / CkptSave / CkptRestore names the ids it will hold if
// the operation succeeds (`keep`), and the store drops the others only then; it never evicts on its own.  A part that
// does not fit (kStageCkptMax after pruning) is not stored and the store stays as it was.  At the hello a worker
// also refuses a main process whose checkpoints would not fit in half of the RAM available (stage_ckpt_ram_ok).
// Header-only, no GPU: stage_ckpt_store_test.cpp walks it by hand.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/net/stage_link.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace strata::net {

class StageCkptStore {
public:
    struct Part {
        int64_t L = 0;                         ///< the position it was saved at (its length: it carries no ids)
        core::ConversationCheckpoint state;    ///< this worker's layers' running state
    };

    /// `id` fits once the store is pruned to `keep` (the parts that stay, other than one of the same id, plus it)
    bool admits(int64_t id, const std::vector<int64_t>& keep) const {
        size_t n = 1;
        for (const auto& kv : parts_) n += kv.first != id && kept(kv.first, keep) ? 1 : 0;
        return n <= (size_t) kStageCkptMax;
    }
    /// stores `id` (in place of one of the same id), then drops every part not in `keep` - never `id` itself: a
    /// part reported stored is held (the main process's keep names it anyway).  A failed allocation
    /// (std::bad_alloc) leaves the store as it was.
    void put(int64_t id, Part part, const std::vector<int64_t>& keep) {
        parts_.insert_or_assign(id, std::move(part));
        for (auto it = parts_.begin(); it != parts_.end();)
            it = it->first == id || kept(it->first, keep) ? std::next(it) : parts_.erase(it);
    }
    const Part* find(int64_t id) const {
        const auto it = parts_.find(id);
        return it == parts_.end() ? nullptr : &it->second;
    }
    void prune(const std::vector<int64_t>& keep) {
        for (auto it = parts_.begin(); it != parts_.end();)
            it = kept(it->first, keep) ? std::next(it) : parts_.erase(it);
    }
    void clear() { parts_.clear(); }   ///< the main process went away: a new one is a new id space
    size_t size() const { return parts_.size(); }

private:
    static bool kept(int64_t id, const std::vector<int64_t>& keep) {
        return std::find(keep.begin(), keep.end(), id) != keep.end();
    }
    std::map<int64_t, Part> parts_;
};

/// The hello: `ckpt_max` parts of `part_bytes` each must fit in half of the RAM `available` here (Linux overcommits:
/// a store that outgrows the RAM meets the OOM killer, not std::bad_alloc).  A ckpt_max the link refuses anyway
/// (<= 0, > kStageCkptMax) or an unknown `available` passes.
inline bool stage_ckpt_ram_ok(int32_t ckpt_max, uint64_t part_bytes, std::optional<uint64_t> available,
                              std::string& err) {
    if (ckpt_max <= 0 || ckpt_max > kStageCkptMax || !available) return true;
    const uint64_t need = (uint64_t) ckpt_max * part_bytes, share = *available / 2;
    if (need <= share) return true;
    const uint64_t fit = part_bytes > 0 ? share / part_bytes : 0;
    err = "the main process may have this worker keep " + std::to_string(ckpt_max) + " conversation checkpoints of " +
          std::to_string(part_bytes >> 20) + " MiB (" + std::to_string(need >> 20) + " MiB), more than half of the " +
          std::to_string(*available >> 20) + " MiB of RAM available here: start it with --prompt-cache " +
          std::to_string(fit > 1 ? fit - 1 : 0) + " or less";
    return false;
}

}  // namespace strata::net
