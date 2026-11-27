#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#if defined(FNX_DISTRIBUTED)
#include <vector>
#endif

namespace fnx::distributed {
#if defined(FNX_DISTRIBUTED)
bool enabled();
int first_card();
int end_card();
bool coordinator();
void sum(float* values, size_t count, void* stream);
void gather(const float* input, float* output, size_t count, void* stream);
void exchange(const float* input, float* output, size_t stride, size_t offset, size_t count, void* stream);
uint64_t minimum(uint64_t local);
void gather_host(const void* local, size_t bytes, void* result, const std::vector<int>& sizes);
void gather_rows(void* values, size_t stride, size_t bytes);
unsigned results(float* values, int* ids, unsigned* errors, int stride, int rows, float& gpu_ms, double* card_ms = nullptr);
void pass_rows(int rows);
std::string command(const std::string& root);
#else
inline constexpr bool enabled() { return false; }
inline constexpr int first_card() { return 0; }
inline constexpr int end_card() { return FNX_NCARD; }
inline constexpr bool coordinator() { return true; }
#endif
}
