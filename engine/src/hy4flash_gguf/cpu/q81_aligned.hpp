#pragma once

#include <cstdint>

namespace strata::hy4_flash::cpu::q81_aligned {
constexpr unsigned mantissa_count = 1u << 23;
const uint32_t* device_tables();
void quantize(const uint32_t* table, const float* x, void* out, int n);
void gate_up(const uint32_t* table, const uint8_t* gate, const uint8_t* up, const uint8_t* activation,
             float* out, int rows);
void down(int type, const uint8_t* weight, const uint8_t* activation, float* out, int rows);
} // namespace strata::hy4_flash::cpu::q81_aligned
