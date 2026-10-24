#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::muse {
// Reference-engine limits, not model geometry. Mappings and library overhead are
// excluded; all live KV, decoded weight tiles and activation scratch are checked.
constexpr size_t tile_bytes = 4 * 1024 * 1024;
constexpr size_t host_budget_bytes = 1024ULL * 1024 * 1024;
constexpr size_t device_budget_bytes = 64 * 1024 * 1024;
size_t checked_add(size_t a, size_t b);
size_t checked_mul(size_t a, size_t b);
uint32_t parse_uint(const std::string& text, const char* name);
std::vector<uint32_t> parse_tokens(const std::string& text);

struct Options {
    std::string model, logits_file, backend = "cpu";
    std::vector<uint32_t> tokens;
    uint32_t context = 128, steps = 0, gpu = 0;
};
Options parse_options(int argc, const char* const* argv);

struct Geometry {
    size_t hidden = 0, vocab = 0, layers = 0, heads = 0, kv_heads = 0;
    size_t head_dim = 0, rotary_dim = 0, latent = 0, experts = 0, top_k = 0;
    size_t train_context = 0, window = 0;
    std::vector<size_t> ff;
    std::vector<bool> local;
    float epsilon = 0, post_epsilon = 0, rope_base = 0;
    float logit_scale = 0, softcap = 0, expert_scale = 0;
    size_t memory_bytes(size_t context) const;
};

class Model {
public:
    explicit Model(const std::string& path, size_t context, const std::string& backend = "cpu", int gpu = 0);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    const Geometry& geometry() const;
    // A sequential, causal token input. Returns full next-token logits.
    std::vector<float> forward(uint32_t token);
    bool is_end(uint32_t token) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Logits contain forced inputs only. Generation uses later logits internally.
void run(const Options& options);
} // namespace strata::muse
