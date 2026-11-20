#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::core {
// Configure once before allocating/capturing sessions. No environment selection.
bool spin8_attention_configure(const std::string& mode, const std::string& profile, std::string& error);
bool spin8_attention_enabled();
bool spin8_attention_check(std::string& error);
bool spin8_attention_step(const float* query, const uint16_t* key, const uint16_t* value,
    const int32_t* step, size_t capacity, size_t heads, size_t kv_heads, size_t dimension,
    float* scores, float* output, int32_t* status, int32_t* host_status, void* stream, std::string& error);
}
