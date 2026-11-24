#include "strata/hy4flash_gguf/conversation_cache.hpp"
#include "strata/hy4flash_gguf/kv_cache.hpp"
#include "strata/hy4flash_gguf/indexer.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/hy4flash_gguf/mtp.hpp"
#include "strata/hy4flash_gguf/serve.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace strata::hy4_flash {
namespace {
void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int shared_prefix(const std::vector<int32_t>& ids, const std::vector<int32_t>& prompt, int limit) {
    const int end = std::min({int(ids.size()), int(prompt.size()), limit});
    return int(std::mismatch(ids.begin(), ids.begin() + end, prompt.begin()).first - ids.begin());
}

struct Region {
    void* device;
    size_t bytes_per_group;
};

void append_regions(std::vector<Region>& out, KvCache& kv, IndexerCache* idx, int group) {
    // A token's rows are contiguous in every format, the Q8_0 latent's scales included, so a group of
    // tokens is `group` rows.
    const int format = kv_format(kv);
    for (void* p : kv.kv_cmpr) out.push_back({p, size_t(group) * kv_cache_lat_row_bytes(kv.kv_lora, format)});
    for (void* p : kv.k_pe) out.push_back({p, size_t(group) * kv_cache_pe_row_bytes(kv.n_rot, format)});
    if (idx) {
        if (group != idx->kpool) throw std::logic_error("incompatible hy4-flash indexer pool size");
        for (void* p : idx->pooled) out.push_back({p, size_t(idx->key_dim) * sizeof(float)});
    }
}

void set_position(KvCache& kv, IndexerCache* idx, int position) {
    kv.len = position;
    if (!idx) return;
    idx->valid_pos = position;
    idx->n_pool = position / idx->kpool;
    idx->carry = 0;
    idx->verify_base = -1;
    idx->verify_tokens = 0;
    const int counters[] = {0, idx->n_pool};
    checked(cudaMemcpy(idx->counters, counters, sizeof counters, cudaMemcpyHostToDevice));
}
} // namespace

ConversationCache::ConversationCache(int mib, int slots)
    : budget_(size_t(mib) << 20), slots_(slots) {}

void ConversationCache::event(const char* name, int tokens) const {
    std::fprintf(stderr, "strata serve: conversation cache: %s %d tokens; parked=%zu bytes=%zu evictions=%zu\n",
                 name, tokens, entries_.size(), used_, evictions_);
}

void ConversationCache::clear() {
    entries_.clear();
    used_ = 0;
}

int ConversationCache::resume(const ServeState& s, float* hidden, const std::vector<int32_t>& live,
                              const std::vector<int32_t>& prompt, bool keep_turn) {
    const int group = std::max(1, s.g->idx_key_dim > 0 ? s.g->idx_kpool : 1);
    auto aligned = [group](int n) { return n - n % group; };
    const int available = std::min(int(live.size()), s.cache->len);
    const int common = shared_prefix(live, prompt, available);
    const int last = int(prompt.size()) - 1; // replay last prompt token for logits
    int best = aligned(std::min(common, last));
    if (!slots_ || !budget_) return best;

    auto donor = entries_.end();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        const int n = aligned(shared_prefix(it->ids, prompt, last));
        if (n > best) {
            best = n;
            donor = it;
        }
    }
    int capture = aligned(available);
    if (keep_turn && donor == entries_.end() && common >= capture) capture = 0;
    if (!capture && donor == entries_.end()) return best;

    IndexerCache* target_idx = forward_indexer(*s.fo->dense);
    if (target_idx) capture = std::min(capture, aligned(target_idx->valid_pos));
    std::vector<Region> regions;
    append_regions(regions, *s.cache, target_idx, group);
    if (s.mtp) {
        capture = std::min(capture, aligned(s.mtp->position()));
        append_regions(regions, s.mtp->kv_cache(), &s.mtp->indexer_cache(), group);
        // Row zero is always zero. Hidden row n conditions the draft at n.
        regions.push_back({hidden + s.g->hidden_dim, size_t(group) * s.g->hidden_dim * sizeof(float)});
    }
    size_t group_bytes = 0;
    for (const auto& region : regions) group_bytes += region.bytes_per_group;
    const size_t groups = capture / group;
    const size_t need = groups * group_bytes + size_t(capture) * sizeof(int32_t);
    // Detach the incoming image so eviction cannot invalidate it. Charge its
    // bytes until the transfer finishes, including when there is only one slot.
    std::list<Image> incoming;
    size_t held = 0;
    if (donor != entries_.end()) {
        held = donor->bytes;
        used_ -= held;
        incoming.splice(incoming.end(), entries_, donor);
    }
    bool park = capture && need <= budget_ - held;
    if (park || !incoming.empty()) checked(cudaDeviceSynchronize());
    if (park) {
        while (entries_.size() >= slots_ || used_ > budget_ - held - need) {
            used_ -= entries_.front().bytes;
            entries_.pop_front();
            ++evictions_;
        }
        park = strata::core::conversation_memory_admit(strata::core::available_host_bytes(), need, 0);
    }
    if (park) {
        try {
            Image image;
            image.bytes = need;
            image.ids.assign(live.begin(), live.begin() + capture);
            image.rows.reset(new unsigned char[groups * group_bytes]);
            auto* dst = image.rows.get();
            for (const auto& region : regions) {
                const size_t n = groups * region.bytes_per_group;
                checked(cudaMemcpy(dst, region.device, n, cudaMemcpyDeviceToHost));
                dst += n;
            }
            entries_.push_back(std::move(image));
            used_ += need;
            event("parked", capture);
        } catch (const std::bad_alloc&) {
            event("skipped", capture);
        }
    } else if (capture) {
        event("skipped", capture);
    }

    if (!incoming.empty()) {
        const Image& image = incoming.front();
        const size_t stored_groups = image.ids.size() / group;
        const auto* src = image.rows.get();
        for (const auto& region : regions) {
            checked(cudaMemcpy(region.device, src, size_t(best / group) * region.bytes_per_group,
                               cudaMemcpyHostToDevice));
            src += stored_groups * region.bytes_per_group;
        }
        set_position(*s.cache, target_idx, best);
        if (s.mtp) {
            set_position(s.mtp->kv_cache(), &s.mtp->indexer_cache(), best);
            s.mtp->forget_speculation();
        }
        event("restored", best);
    }
    return best;
}
} // namespace strata::hy4_flash
