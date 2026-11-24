#include "df2_bridge.hpp"

namespace fnx::df2 {
struct Bridge::Impl {};
Bridge::Bridge() = default;
Bridge::~Bridge() = default;
bool Bridge::load(const std::string&, int, int, const std::vector<int32_t>&,
                  const void*, size_t, int, std::function<void(int, float*)>,
                  std::string& error) {
    error = "SYCL split: DFlash2 is not supported";
    return false;
}
int Bridge::lanes() const { return 0; }
uint64_t Bridge::vram_bytes() const { return 0; }
}  // namespace fnx::df2
