#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::hy4_flash {

// The KV and causal positions stay sequential. Only the RoPE angles use these
// three device planes, with a constant position offset after the prompt.
struct ImageRope {
    const int32_t* axes = nullptr;
    int length = 0;
    int delta = 0;
};

struct VisionPrompt {
    virtual ~VisionPrompt() = default;
    ImageRope rope;
    std::vector<uint8_t> image_mask;
    virtual void embed(float* state, int base, int count, void* stream) = 0;
};

bool probe_vision(const std::string& shard, std::string& error);
std::unique_ptr<VisionPrompt> encode_pictures(const std::string& shard, const std::string& file,
    const std::vector<int32_t>& ids, std::string& error);

}  // namespace strata::hy4_flash
