#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <vector>

namespace strata::hy4_flash {

struct ServeState;

// Host images of committed pool-aligned prefixes. A single serving thread owns
// this store. The device allocations remain in place for captured CUDA graphs.
class ConversationCache {
public:
    ConversationCache(int mib, int slots);
    int resume(const ServeState& state, float* hidden, const std::vector<int32_t>& live,
               const std::vector<int32_t>& prompt, bool keep_turn);
    void clear();

private:
    struct Image {
        std::vector<int32_t> ids;
        std::unique_ptr<unsigned char[]> rows;
        size_t bytes = 0;
    };
    void event(const char* name, int tokens) const;
    std::list<Image> entries_;
    size_t budget_, slots_, used_ = 0, evictions_ = 0;
};

} // namespace strata::hy4_flash
