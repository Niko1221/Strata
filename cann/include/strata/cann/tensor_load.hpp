// cann/include/strata/cann/tensor_load.hpp - a GGUF tensor as floats, whatever it is stored as.
#pragma once

#include <vector>

#include "strata/cann/gguf.hpp"

namespace strata::cann {

// Bytes the tensor occupies on disk; 0 when the type is not supported yet.
uint64_t tensor_bytes_on_disk(const GgufTensor & t);

// Dequantize / convert the whole tensor to float in file order.
// Throws std::runtime_error for a type the port cannot read yet.
std::vector<float> load_tensor(const Gguf & g, const GgufTensor & t);

} // namespace strata::cann
