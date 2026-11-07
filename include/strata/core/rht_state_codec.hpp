#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace strata::core::rht {

constexpr uint64_t kVersion = 1;
constexpr uint64_t kQmax = 32767;
constexpr uint64_t kSeed = 0;

struct Layout {
    uint64_t layers;
    uint64_t rows;
    uint64_t dim;
    uint64_t conv_floats;
    uint64_t logical;
    uint64_t stored;
};

namespace detail {

constexpr uint64_t kDim = 128;
constexpr float kScaleFloor = 1.0e-12f;
constexpr uint64_t kConvChunkBytes = 16u * 1024u;

// Generated offline from SHA256("omlx-gdn-rht-v1:0:128:" + little-endian i),
// taking +1 when the digest low bit is 1 and -1 otherwise. Verified fingerprint:
// 68 positive / 60 negative; first 32: -++-+-+-+++---+---+-+-++++---+--;
// SHA256(packbits) = 862b7f24ac5c46c977c56eefa6571c83a3844398d6d2ff76d30b931a5bd97a75.
constexpr std::array<float, kDim> kSigns = {
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f,
    +1.0f, +1.0f, +1.0f, -1.0f, -1.0f, -1.0f, +1.0f, -1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f,
    +1.0f, +1.0f, -1.0f, -1.0f, -1.0f, +1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, +1.0f, +1.0f,
    +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, +1.0f, +1.0f, +1.0f, +1.0f, -1.0f,
    +1.0f, +1.0f, +1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f,
    +1.0f, +1.0f, +1.0f, -1.0f, -1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, +1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, -1.0f, -1.0f, +1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, +1.0f, +1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f,
    +1.0f, +1.0f, +1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f,
};

struct Metrics {
    uint64_t recurrent_floats;
    uint64_t per_layer_floats;
    uint64_t conv_bytes;
    uint64_t logical_bytes;
    uint64_t stored_bytes;
};

inline bool checked_add(uint64_t a, uint64_t b, uint64_t& out) noexcept {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    out = a + b;
    return true;
}

inline bool checked_mul(uint64_t a, uint64_t b, uint64_t& out) noexcept {
    if (a && b > std::numeric_limits<uint64_t>::max() / a) return false;
    out = a * b;
    return true;
}

inline bool calculate_metrics(const Layout& layout, Metrics& metrics) noexcept {
    if (!layout.layers || !layout.rows || layout.dim != kDim || !layout.conv_floats) return false;

    uint64_t row_values = 0;
    if (!checked_mul(layout.rows, layout.dim, row_values)) return false;
    if (!checked_add(row_values, layout.conv_floats, metrics.per_layer_floats)) return false;
    if (!checked_mul(layout.layers, metrics.per_layer_floats, metrics.logical_bytes)) return false;
    if (!checked_mul(metrics.logical_bytes, sizeof(float), metrics.logical_bytes)) return false;
    metrics.recurrent_floats = row_values;
    if (!checked_mul(layout.conv_floats, sizeof(float), metrics.conv_bytes)) return false;

    uint64_t quantized_row_bytes = 0;
    if (!checked_mul(layout.dim, sizeof(int16_t), quantized_row_bytes)) return false;
    if (!checked_add(quantized_row_bytes, sizeof(float), quantized_row_bytes)) return false;
    uint64_t recurrent_bytes = 0;
    if (!checked_mul(layout.rows, quantized_row_bytes, recurrent_bytes)) return false;
    uint64_t per_layer_stored = 0;
    if (!checked_add(recurrent_bytes, metrics.conv_bytes, per_layer_stored)) return false;
    if (!checked_mul(layout.layers, per_layer_stored, metrics.stored_bytes)) return false;

    if (metrics.logical_bytes > std::numeric_limits<std::size_t>::max()) return false;
    if (metrics.conv_bytes > std::numeric_limits<std::size_t>::max()) return false;
    return true;
}

inline bool metrics_for(const Layout& layout, Metrics& metrics) noexcept {
    return calculate_metrics(layout, metrics) && layout.logical == metrics.logical_bytes &&
           layout.stored == metrics.stored_bytes;
}

inline bool valid_layout(const Layout& layout) noexcept {
    Metrics metrics{};
    return metrics_for(layout, metrics);
}

inline void hadamard(float* values) noexcept {
    for (uint64_t width = 1; width < kDim; width <<= 1) {
        for (uint64_t base = 0; base < kDim; base += width << 1) {
            for (uint64_t i = 0; i < width; ++i) {
                const float a = values[base + i];
                const float b = values[base + width + i];
                values[base + i] = a + b;
                values[base + width + i] = a - b;
            }
        }
    }
}

template <class Sink>
inline bool safe_put(Sink& put, const void* data, std::size_t size) noexcept {
    try {
        return put(data, size);
    } catch (...) {
        return false;
    }
}

template <class Source>
inline bool safe_read(Source& read, void* data, std::size_t size) noexcept {
    try {
        return read(data, size);
    } catch (...) {
        return false;
    }
}

inline bool encode_row(const float* src, float* work, int16_t* quantized, float& scale) noexcept {
    for (uint64_t i = 0; i < kDim; ++i) work[i] = src[i] * kSigns[i];
    hadamard(work);

    constexpr float inv_sqrt_dim = 1.0f / 11.313708498984761f;
    float max_abs = 0.0f;
    for (uint64_t i = 0; i < kDim; ++i) {
        work[i] *= inv_sqrt_dim;
        if (!std::isfinite(work[i])) return false;
        max_abs = std::max(max_abs, std::fabs(work[i]));
    }
    if (!std::isfinite(max_abs)) return false;

    scale = std::max(max_abs / static_cast<float>(kQmax), kScaleFloor);
    if (!std::isfinite(scale) || scale < kScaleFloor) return false;
    for (uint64_t i = 0; i < kDim; ++i) {
        const float normalized = work[i] / scale;
        if (!std::isfinite(normalized)) return false;
        const float rounded = std::rint(normalized);
        if (!std::isfinite(rounded)) return false;
        const float clipped = std::max(-static_cast<float>(kQmax),
                                        std::min(static_cast<float>(kQmax), rounded));
        quantized[i] = static_cast<int16_t>(clipped);
    }
    return true;
}

} // namespace detail

inline bool layout_from(const std::array<int64_t, 18>& geometry,
                        int64_t layer_lo, int64_t layer_hi, Layout& out) noexcept {
    const int64_t n_layers = geometry[1];
    const int64_t qsa_interval = geometry[2];
    const int64_t dim = geometry[3];
    const int64_t value_heads = geometry[5];
    const int64_t d_conv = geometry[6];
    const int64_t conv_channels = geometry[7];
    if (n_layers <= 0 || qsa_interval <= 0 || dim != static_cast<int64_t>(detail::kDim) ||
        value_heads <= 0 || d_conv <= 1 || conv_channels <= 0) return false;
    if (layer_lo < 0 || layer_hi <= layer_lo || layer_hi > n_layers) return false;

    const uint64_t interval = static_cast<uint64_t>(qsa_interval);
    const uint64_t lo = static_cast<uint64_t>(layer_lo);
    const uint64_t hi = static_cast<uint64_t>(layer_hi);
    const uint64_t qsa_layers = hi / interval - lo / interval;
    const uint64_t owned_layers = (hi - lo) - qsa_layers;
    if (!owned_layers) return false;

    Layout candidate{};
    candidate.layers = owned_layers;
    candidate.dim = static_cast<uint64_t>(dim);
    candidate.conv_floats = 0;
    if (!detail::checked_mul(candidate.dim, static_cast<uint64_t>(value_heads), candidate.rows)) return false;
    if (!detail::checked_mul(static_cast<uint64_t>(conv_channels),
                             static_cast<uint64_t>(d_conv - 1), candidate.conv_floats)) return false;

    detail::Metrics metrics{};
    if (!detail::calculate_metrics(candidate, metrics)) return false;
    candidate.logical = metrics.logical_bytes;
    candidate.stored = metrics.stored_bytes;
    if (!detail::metrics_for(candidate, metrics)) return false;
    out = candidate;
    return true;
}

inline uint64_t scratch_bytes(const Layout& layout) noexcept {
    if (!detail::valid_layout(layout)) return 0;
    return sizeof(float) * detail::kDim + sizeof(int16_t) * detail::kDim;
}

inline bool all_finite(const void* src, uint64_t logical) noexcept {
    if (logical % sizeof(float) != 0 || logical > std::numeric_limits<std::size_t>::max()) return false;
    if (!logical) return true;
    if (!src) return false;
    const auto* values = static_cast<const float*>(src);
    const uint64_t count = logical / sizeof(float);
    for (uint64_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[static_cast<std::size_t>(i)])) return false;
    }
    return true;
}

template <class Sink>
inline bool encode(const void* src, const Layout& layout, Sink&& put) noexcept {
    detail::Metrics metrics{};
    if (!src || !detail::metrics_for(layout, metrics) || !all_finite(src, metrics.logical_bytes)) return false;

    auto&& sink = put;
    const auto* source = static_cast<const float*>(src);
    const std::size_t per_layer_floats = static_cast<std::size_t>(metrics.per_layer_floats);
    const std::size_t recurrent_floats = static_cast<std::size_t>(metrics.recurrent_floats);
    const std::size_t conv_bytes = static_cast<std::size_t>(metrics.conv_bytes);
    std::array<float, detail::kDim> work{};
    std::array<int16_t, detail::kDim> quantized{};

    for (uint64_t layer = 0; layer < layout.layers; ++layer) {
        const std::size_t layer_offset = static_cast<std::size_t>(layer) * per_layer_floats;
        const float* layer_src = source + layer_offset;
        for (uint64_t row = 0; row < layout.rows; ++row) {
            float scale = 0.0f;
            const float* row_src = layer_src + static_cast<std::size_t>(row) * static_cast<std::size_t>(layout.dim);
            if (!detail::encode_row(row_src, work.data(), quantized.data(), scale)) return false;
            if (!detail::safe_put(sink, &scale, sizeof(scale)) ||
                !detail::safe_put(sink, quantized.data(), sizeof(quantized))) return false;
        }

        const auto* conv = reinterpret_cast<const unsigned char*>(layer_src + recurrent_floats);
        for (std::size_t offset = 0; offset < conv_bytes;) {
            const std::size_t count = std::min<std::size_t>(static_cast<std::size_t>(detail::kConvChunkBytes), conv_bytes - offset);
            if (!detail::safe_put(sink, conv + offset, count)) return false;
            offset += count;
        }
    }
    return true;
}

template <class Source>
inline bool decode(Source&& read, void* dst, const Layout& layout) noexcept {
    detail::Metrics metrics{};
    if (!dst || !detail::metrics_for(layout, metrics)) return false;

    auto&& source = read;
    auto* destination = static_cast<float*>(dst);
    const std::size_t per_layer_floats = static_cast<std::size_t>(metrics.per_layer_floats);
    const std::size_t recurrent_floats = static_cast<std::size_t>(metrics.recurrent_floats);
    const std::size_t conv_bytes = static_cast<std::size_t>(metrics.conv_bytes);
    std::array<float, detail::kDim> work{};
    std::array<int16_t, detail::kDim> quantized{};
    constexpr float inv_sqrt_dim = 1.0f / 11.313708498984761f;

    for (uint64_t layer = 0; layer < layout.layers; ++layer) {
        const std::size_t layer_offset = static_cast<std::size_t>(layer) * per_layer_floats;
        float* layer_dst = destination + layer_offset;
        for (uint64_t row = 0; row < layout.rows; ++row) {
            float scale = 0.0f;
            if (!detail::safe_read(source, &scale, sizeof(scale)) ||
                !detail::safe_read(source, quantized.data(), sizeof(quantized))) return false;
            if (!std::isfinite(scale) || scale < detail::kScaleFloor) return false;
            bool zero_row = true;
            for (int16_t q : quantized) {
                if (q < -static_cast<int16_t>(kQmax) || q > static_cast<int16_t>(kQmax)) return false;
                zero_row = zero_row && q == 0;
            }

            float* row_dst = layer_dst + static_cast<std::size_t>(row) * static_cast<std::size_t>(layout.dim);
            if (zero_row) {
                std::fill(row_dst, row_dst + detail::kDim, 0.0f);
                continue;
            }
            for (uint64_t i = 0; i < detail::kDim; ++i) {
                work[i] = static_cast<float>(quantized[i]) * scale;
                if (!std::isfinite(work[i])) return false;
            }
            detail::hadamard(work.data());
            for (uint64_t i = 0; i < detail::kDim; ++i) {
                work[i] *= inv_sqrt_dim;
                work[i] *= detail::kSigns[i];
                if (!std::isfinite(work[i])) return false;
                row_dst[i] = work[i];
            }
        }

        auto* conv_dst = reinterpret_cast<unsigned char*>(layer_dst + recurrent_floats);
        for (std::size_t offset = 0; offset < conv_bytes;) {
            const std::size_t count = std::min<std::size_t>(static_cast<std::size_t>(detail::kConvChunkBytes), conv_bytes - offset);
            if (!detail::safe_read(source, conv_dst + offset, count)) return false;
            if (!all_finite(conv_dst + offset, count)) return false;
            offset += count;
        }
    }
    return true;
}

} // namespace strata::core::rht
