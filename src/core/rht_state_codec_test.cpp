#include "strata/core/rht_state_codec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

using strata::core::rht::Layout;
using strata::core::rht::all_finite;
using strata::core::rht::decode;
using strata::core::rht::encode;
using strata::core::rht::layout_from;
using strata::core::rht::scratch_bytes;

constexpr char kExpectedSigns[] =
    "-++-+-+-" "+++---+-" "--+-+-++" "++---+--"
    "----++-+" "+-+-++++" "++-+-+++" "--+++++-"
    "+++++-+-" "+++----+" "--+++--+" "--+---++"
    "--+--+--" "--+-++++" "----+-++" "+++++-+-";
static_assert(sizeof(kExpectedSigns) - 1 == 128, "sign fixture must have 128 entries");

static int checks_run = 0;
static void check(bool ok, const char* label) {
    ++checks_run;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        std::exit(1);
    }
}

static std::array<int64_t, 18> geometry(int64_t n_layers = 1, int64_t interval = 4,
                                         int64_t dim = 128, int64_t value_heads = 1,
                                         int64_t d_conv = 2, int64_t conv_channels = 3) {
    return {2560, n_layers, interval, dim, 16, value_heads, d_conv, conv_channels,
            6144, 24, 2, 256, 4, 128, 4, 320, 512, 640};
}

static void append(std::vector<uint8_t>& bytes, const void* data, std::size_t size) {
    const auto* first = static_cast<const uint8_t*>(data);
    bytes.insert(bytes.end(), first, first + size);
}

static std::vector<uint8_t> encode_to_bytes(const void* src, const Layout& layout, std::size_t sink_chunk) {
    std::vector<uint8_t> bytes;
    const bool ok = encode(src, layout, [&](const void* data, std::size_t size) {
        const auto* input = static_cast<const uint8_t*>(data);
        for (std::size_t offset = 0; offset < size;) {
            const std::size_t count = std::min(sink_chunk, size - offset);
            append(bytes, input + offset, count);
            offset += count;
        }
        return true;
    });
    check(ok, "stream encode succeeds");
    return bytes;
}

static bool read_fragmented(const std::vector<uint8_t>& bytes, std::size_t& offset,
                           void* destination, std::size_t size, std::size_t fragment) {
    if (size > bytes.size() - std::min(offset, bytes.size())) return false;
    auto* out = static_cast<uint8_t*>(destination);
    for (std::size_t copied = 0; copied < size;) {
        const std::size_t count = std::min(fragment, size - copied);
        std::memcpy(out + copied, bytes.data() + offset, count);
        copied += count;
        offset += count;
    }
    return true;
}

static unsigned parity(unsigned value) {
    unsigned bit = 0;
    while (value) {
        bit ^= value & 1u;
        value >>= 1;
    }
    return bit;
}

static void reference_row(const float* input, std::size_t dim, const char* signs,
                          std::vector<float>& transformed, float& scale,
                          std::vector<int16_t>& quantized) {
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(dim));
    transformed.resize(dim);
    quantized.resize(dim);
    for (std::size_t k = 0; k < dim; ++k) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < dim; ++j) {
            const float signed_value = input[j] * (signs[j] == '+' ? 1.0f : -1.0f);
            sum += parity(static_cast<unsigned>(k & j)) ? -signed_value : signed_value;
        }
        transformed[k] = sum * inv_sqrt;
    }
    float max_abs = 0.0f;
    for (float value : transformed) max_abs = std::max(max_abs, std::fabs(value));
    scale = std::max(max_abs / 32767.0f, 1.0e-12f);
    for (std::size_t i = 0; i < dim; ++i) {
        const float rounded = std::rint(transformed[i] / scale);
        const float clipped = std::max(-32767.0f, std::min(32767.0f, rounded));
        quantized[i] = static_cast<int16_t>(clipped);
    }
}

static void reference_decode_row(const std::vector<int16_t>& quantized, float scale,
                                 const char* signs, std::vector<float>& output) {
    const std::size_t dim = quantized.size();
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(dim));
    output.resize(dim);
    for (std::size_t j = 0; j < dim; ++j) {
        float sum = 0.0f;
        for (std::size_t k = 0; k < dim; ++k) {
            const float value = static_cast<float>(quantized[k]) * scale;
            sum += parity(static_cast<unsigned>(k & j)) ? -value : value;
        }
        output[j] = (sum * inv_sqrt) * (signs[j] == '+' ? 1.0f : -1.0f);
    }
}

static void test_dim8_golden_reference() {
    constexpr char signs[] = "-++---+-";
    constexpr int16_t expected_q[] = {-16383, 27306, -10922, -32767, 16383, -16383, 16383, 5461};
    const float input[] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> transformed, reconstructed;
    std::vector<int16_t> quantized;
    float scale = 0.0f;
    reference_row(input, 8, signs, transformed, scale, quantized);
    check(std::fabs(scale - 2.5895814e-4f) < 1.0e-10f, "dim-8 reference scale golden");
    check(quantized.size() == 8 && std::equal(quantized.begin(), quantized.end(), expected_q),
          "dim-8 reference q golden");
    reference_decode_row(quantized, scale, signs, reconstructed);
    float max_error = 0.0f;
    for (std::size_t i = 0; i < 8; ++i) max_error = std::max(max_error, std::fabs(reconstructed[i] - input[i]));
    check(max_error < 2.0e-4f, "dim-8 reference roundtrip golden");
}

static void test_geometry_and_sizes() {
    Layout layout{};
    check(layout_from(geometry(48, 4, 128, 48, 4, 10240), 0, 48, layout), "production geometry accepted");
    check(layout.layers == 36 && layout.rows == 6144 && layout.dim == 128 &&
          layout.conv_floats == 30720, "production layout fields");
    check(layout.logical == 117669888 && layout.stored == 61931520,
          "production logical and stored formulas");
    check(scratch_bytes(layout) <= 68u * 1024u && scratch_bytes(layout) > 0,
          "bounded codec scratch");

    Layout carved{};
    check(layout_from(geometry(12, 4), 2, 8, carved) && carved.layers == 4,
          "carve excludes QSA layers 3 and 7");
    check(!layout_from(geometry(12, 4), 3, 4, carved), "QSA-only carve rejected");
    check(!layout_from(geometry(0, 4), 0, 0, carved), "empty geometry rejected");
    check(!layout_from(geometry(12, 0), 0, 12, carved), "zero interval rejected");
    check(!layout_from(geometry(12, 4, 64), 0, 12, carved), "unsupported dimension rejected");
    check(!layout_from(geometry(12, 4, 127), 0, 12, carved), "non-power-of-two dimension rejected");
    check(!layout_from(geometry(12, 4), -1, 8, carved), "negative carve rejected");
    check(!layout_from(geometry(12, 4), 9, 8, carved), "reversed carve rejected");
    check(!layout_from(geometry(12, 4), 0, 13, carved), "out-of-range carve rejected");

    auto overflowing = geometry(2, 3, 128, 1, std::numeric_limits<int64_t>::max(),
                                std::numeric_limits<int64_t>::max());
    check(!layout_from(overflowing, 0, 2, carved), "geometry product overflow rejected");
}

static void test_finite_and_signs() {
    Layout layout{};
    check(layout_from(geometry(), 0, 1, layout), "small production layout available");
    check(layout.logical == (128u * 128u + 3u) * sizeof(float), "small logical formula");
    check(all_finite(nullptr, 0), "empty finite scan");
    std::vector<float> values(static_cast<std::size_t>(layout.logical / sizeof(float)), 0.0f);
    check(all_finite(values.data(), layout.logical), "finite scan accepts finite state");
    values[layout.logical / sizeof(float) - 1] = std::numeric_limits<float>::quiet_NaN();
    check(!all_finite(values.data(), layout.logical), "finite scan rejects NaN");
    values.back() = std::numeric_limits<float>::infinity();
    check(!all_finite(values.data(), layout.logical), "finite scan rejects infinity");
    check(!all_finite(values.data(), layout.logical - 1), "finite scan rejects partial float bytes");

    std::fill(values.begin(), values.end(), 0.0f);
    for (std::size_t row = 0; row < 128; ++row) values[row * 128 + row] = 1.0f;
    const auto encoded = encode_to_bytes(values.data(), layout, 7);
    const std::size_t row_bytes = sizeof(float) + 128u * sizeof(int16_t);
    const float expected_scale = (1.0f / std::sqrt(128.0f)) / 32767.0f;
    int positive = 0;
    for (std::size_t row = 0; row < 128; ++row) {
        float scale = 0.0f;
        std::memcpy(&scale, encoded.data() + row * row_bytes, sizeof(scale));
        check(std::fabs(scale - expected_scale) < 1.0e-10f, "dim-128 basis scale");
        int16_t q0 = 0;
        std::memcpy(&q0, encoded.data() + row * row_bytes + sizeof(float), sizeof(q0));
        const int16_t expected = kExpectedSigns[row] == '+' ? 32767 : -32767;
        check(q0 == expected, "dim-128 sign table entry");
        positive += kExpectedSigns[row] == '+';
    }
    check(positive == 68, "dim-128 sign table population");

    std::fill(values.begin(), values.end(), 0.0f);
    const auto zero_encoded = encode_to_bytes(values.data(), layout, 7);
    float zero_scale = 0.0f;
    std::memcpy(&zero_scale, zero_encoded.data(), sizeof(zero_scale));
    check(zero_scale == 1.0e-12f, "dim-128 zero-row scale floor");
    for (std::size_t i = 0; i < 128; ++i) {
        int16_t q = 1;
        std::memcpy(&q, zero_encoded.data() + sizeof(float) + i * sizeof(q), sizeof(q));
        check(q == 0, "dim-128 zero-row quantization");
    }
}

static void test_roundtrip_and_streaming() {
    Layout layout{};
    check(layout_from(geometry(1, 4, 128, 1, 3, 2), 0, 1, layout), "roundtrip layout accepted");
    const std::size_t float_count = static_cast<std::size_t>(layout.logical / sizeof(float));
    const std::size_t recurrent_count = static_cast<std::size_t>(layout.rows * layout.dim);
    std::vector<float> source(float_count, 0.0f);
    for (std::size_t row = 0; row < layout.rows; ++row) {
        if (row == 1) continue;
        for (std::size_t i = 0; i < layout.dim; ++i) {
            const int value = static_cast<int>((row * 19 + i * 7) % 31) - 15;
            source[row * static_cast<std::size_t>(layout.dim) + i] = static_cast<float>(value) * 0.125f;
        }
    }
    source[recurrent_count] = 3.25f;
    source[recurrent_count + 1] = -0.0f;
    source[recurrent_count + 2] = -17.5f;
    const auto original = source;

    const auto encoded = encode_to_bytes(source.data(), layout, 3);
    check(encoded.size() == layout.stored, "streamed stored byte count");
    check(std::memcmp(source.data(), original.data(), layout.logical) == 0, "encode leaves source unchanged");

    const std::size_t row_bytes = sizeof(float) + static_cast<std::size_t>(layout.dim) * sizeof(int16_t);
    float zero_scale = 0.0f;
    std::memcpy(&zero_scale, encoded.data() + row_bytes, sizeof(zero_scale));
    check(zero_scale == 1.0e-12f, "zero row uses exact scale floor");
    for (std::size_t i = 0; i < layout.dim; ++i) {
        int16_t q = 1;
        std::memcpy(&q, encoded.data() + row_bytes + sizeof(float) + i * sizeof(q), sizeof(q));
        check(q == 0, "zero row quantizes exactly to zero");
    }

    const std::size_t conv_offset = static_cast<std::size_t>(layout.rows) * row_bytes;
    const std::size_t conv_bytes = static_cast<std::size_t>(layout.conv_floats) * sizeof(float);
    check(std::memcmp(encoded.data() + conv_offset, source.data() + recurrent_count, conv_bytes) == 0,
          "conv payload is copied raw");

    std::vector<float> destination(float_count, -99.0f);
    std::size_t read_offset = 0;
    const bool decoded = decode([&](void* out, std::size_t size) {
        return read_fragmented(encoded, read_offset, out, size, 5);
    }, destination.data(), layout);
    check(decoded && read_offset == encoded.size(), "fragmented source decodes all chunks");
    check(destination.data() != source.data(), "decode writes independent destination");
    check(std::memcmp(source.data(), original.data(), layout.logical) == 0, "decode does not mutate source");
    check(std::memcmp(destination.data() + recurrent_count, source.data() + recurrent_count, conv_bytes) == 0,
          "decoded conv payload preserves raw bits");
    check(std::memcmp(destination.data() + static_cast<std::size_t>(layout.dim),
                      source.data() + static_cast<std::size_t>(layout.dim),
                      static_cast<std::size_t>(layout.dim) * sizeof(float)) == 0,
          "all-zero row restores exactly");

    std::vector<float> ref_source(layout.dim);
    for (std::size_t i = 0; i < layout.dim; ++i) ref_source[i] = source[i];
    std::vector<float> transformed, reference;
    std::vector<int16_t> q;
    float scale = 0.0f;
    reference_row(ref_source.data(), layout.dim, kExpectedSigns, transformed, scale, q);
    reference_decode_row(q, scale, kExpectedSigns, reference);
    float max_reference_delta = 0.0f;
    float max_roundtrip_error = 0.0f;
    for (std::size_t i = 0; i < layout.dim; ++i) {
        max_reference_delta = std::max(max_reference_delta, std::fabs(destination[i] - reference[i]));
        max_roundtrip_error = std::max(max_roundtrip_error, std::fabs(destination[i] - source[i]));
    }
    check(max_reference_delta < 3.0e-4f, "128-point roundtrip agrees with direct reference");
    check(max_roundtrip_error < 1.0e-3f, "128-point RHT roundtrip error bound");
}

static void test_failures() {
    Layout layout{};
    check(layout_from(geometry(), 0, 1, layout), "failure fixture layout accepted");
    std::vector<float> source(static_cast<std::size_t>(layout.logical / sizeof(float)), 0.25f);
    Layout malformed = layout;
    ++malformed.logical;
    check(scratch_bytes(malformed) == 0, "malformed layout has no scratch estimate");
    check(!encode(source.data(), malformed, [](const void*, std::size_t) { return true; }),
          "encode rejects malformed layout");
    std::vector<float> destination(source.size());
    check(!decode([](void*, std::size_t) { return true; }, destination.data(), malformed),
          "decode rejects malformed layout");

    bool wrote = false;
    check(!encode(source.data(), layout, [&](const void*, std::size_t) { wrote = true; return false; }) && wrote,
          "short write is reported");
    check(!encode(source.data(), layout, [](const void*, std::size_t) -> bool { throw 1; }),
          "throwing sink is converted to failure");
    check(!decode([](void*, std::size_t) { return false; }, destination.data(), layout),
          "short read is reported");
    check(!decode([](void*, std::size_t) -> bool { throw 1; }, destination.data(), layout),
          "throwing source is converted to failure");

    source[3] = std::numeric_limits<float>::infinity();
    std::size_t written_bytes = 0;
    check(!encode(source.data(), layout, [&](const void*, std::size_t size) {
        written_bytes += size;
        return true;
    }) && written_bytes == 0, "non-finite state rejected before output");

    std::fill(source.begin(), source.end(), 0.0f);
    const auto encoded = encode_to_bytes(source.data(), layout, 17);
    auto bad_scale = encoded;
    const float invalid_scale = 0.0f;
    std::memcpy(bad_scale.data(), &invalid_scale, sizeof(invalid_scale));
    std::size_t offset = 0;
    check(!decode([&](void* out, std::size_t size) {
        return read_fragmented(bad_scale, offset, out, size, 11);
    }, destination.data(), layout), "decode rejects zero scale");

    auto bad_q = encoded;
    const int16_t invalid_q = std::numeric_limits<int16_t>::min();
    std::memcpy(bad_q.data() + sizeof(float), &invalid_q, sizeof(invalid_q));
    offset = 0;
    check(!decode([&](void* out, std::size_t size) {
        return read_fragmented(bad_q, offset, out, size, 13);
    }, destination.data(), layout), "decode rejects out-of-range quantized value");
}

int main() {
    test_dim8_golden_reference();
    test_geometry_and_sizes();
    test_finite_and_signs();
    test_roundtrip_and_streaming();
    test_failures();
    std::printf("rht_state_codec: %d checks passed\n", checks_run);
}
