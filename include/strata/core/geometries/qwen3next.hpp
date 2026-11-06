// include/strata/core/geometries/qwen3next.hpp - Qwen3-Next-80B-A3B (qwen3next): the same trunk pattern as
// flash_next - GDN with full attention every 4th layer - at different dimensions, and WITHOUT the parts
// Flash-Next added: no indexer (dense attention on the full-attention layers), no hyper-connections (plain
// pre-norm residual), no MTP. The shape is ModelGeometry::qwen3next(), the single source read_geometry seeds from.
#pragma once

namespace strata::core {

inline constexpr const char* kGeometryName = "qwen3next";
inline constexpr ModelGeometry kCompiledGeometry = ModelGeometry::qwen3next();

}  // namespace strata::core
