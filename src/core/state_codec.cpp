#include "strata/core/state_codec.hpp"

#include <algorithm>
#include <cstring>

namespace strata::core {
namespace {

bool fail(std::string& error, const char* message) {
    error = std::string("state codec: ") + message;
    return false;
}

char lower(char c) { return c >= 'A' && c <= 'Z' ? (char) (c - 'A' + 'a') : c; }

bool named(const std::string& name, const char* want) {
    size_t i = 0;
    for (; want[i]; ++i) {
        if (i >= name.size() || lower(name[i]) != want[i]) return false;
    }
    return i == name.size();
}

// One fp32 -> bf16, round to nearest even. A NaN stays a NaN (its payload is quieted, because truncating seven
// mantissa bits can empty it and a bf16 with a zero payload and a full exponent is an Inf, not a NaN). A finite
// value above the largest bf16 becomes an Inf, which is what round-to-nearest-even means at the top of the range:
// 0x7F7FFFFF is 65535/65536 of the way past the last finite bf16, so its nearest representable value is the next
// encoding, and the next encoding of that exponent is the infinity. Nothing else changes value by magnitude:
// bf16 carries fp32's exponent, so no finite value below that ceiling can reach it.
uint16_t bf16_of(float f) {
    uint32_t b = 0;
    std::memcpy(&b, &f, 4);
    if ((b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu)) return (uint16_t) ((b >> 16) | 0x0040u);
    return (uint16_t) ((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
}

// bf16 is the top half of a float32, so widening is a shift: no rounding, no table, and every bit pattern - including
// the infinities and the NaNs bf16_of produced - means the value it means as an fp32.
uint32_t bf16_bits_of(uint16_t h) { return ((uint32_t) h) << 16; }

float f32_of_bf16(uint16_t h) {
    const uint32_t b = bf16_bits_of(h);
    float f = 0;
    std::memcpy(&f, &b, 4);
    return f;
}

} // namespace

bool state_codec_by_name(const std::string& name, StateCodec& out) {
    if (named(name, "fp32") || named(name, "float32")) { out = StateCodec::Fp32; return true; }
    if (named(name, "bf16") || named(name, "bfloat16")) { out = StateCodec::Bf16; return true; }
    return false;
}

const char* state_codec_name(StateCodec codec) {
    switch (codec) {
        case StateCodec::Fp32: return "fp32";
        case StateCodec::Bf16: return "bf16";
    }
    return nullptr;
}

bool state_codec_by_code(uint64_t code, StateCodec& out) {
    if (code > UINT8_MAX) return false;
    const auto codec = (StateCodec) (uint8_t) code;
    if (state_codec_name(codec) == nullptr) return false;
    out = codec;
    return true;
}

size_t state_codec_size(size_t fp32_bytes, StateCodec codec) {
    switch (codec) {
        case StateCodec::Fp32: return fp32_bytes;
        case StateCodec::Bf16: return fp32_bytes / 2;
    }
    return fp32_bytes;
}

size_t state_codec_fp32_bytes(size_t coded_bytes, StateCodec codec) {
    switch (codec) {
        case StateCodec::Fp32: return coded_bytes;
        case StateCodec::Bf16:
            // half of a whole number of floats: even, and twice its coded size without overflow at these sizes
            if (coded_bytes > SIZE_MAX / 2) return 0;
            return coded_bytes * 2;
    }
    return coded_bytes;
}

bool state_encode(const float* src, size_t floats, StateCodec codec, uint8_t* dst, std::string& error) {
    if (!floats) return true;
    if (!src || !dst) return fail(error, "missing array");
    switch (codec) {
        case StateCodec::Fp32:
            std::memcpy(dst, src, floats * 4);
            return true;
        case StateCodec::Bf16: {
            for (size_t i = 0; i < floats; ++i) {
                const uint16_t h = bf16_of(src[i]);
                std::memcpy(dst + i * 2, &h, 2);   // the file and the host buffers are little-endian by contract
            }
            return true;
        }
    }
    return fail(error, "unknown codec");
}

bool state_decode(const uint8_t* src, size_t floats, StateCodec codec, float* dst, std::string& error) {
    if (!floats) return true;
    if (!src || !dst) return fail(error, "missing array");
    switch (codec) {
        case StateCodec::Fp32:
            std::memcpy(dst, src, floats * 4);
            return true;
        case StateCodec::Bf16:
            for (size_t i = 0; i < floats; ++i) {
                uint16_t h = 0;
                std::memcpy(&h, src + i * 2, 2);
                dst[i] = f32_of_bf16(h);
            }
            return true;
        }
    return fail(error, "unknown codec");
}

bool state_transcode_bytes(const uint8_t* src, size_t floats, StateCodec from, StateCodec to, uint8_t* dst,
                           std::string& error) {
    if (!floats) return true;
    if (!src || !dst) return fail(error, "missing array");
    if (state_codec_name(from) == nullptr || state_codec_name(to) == nullptr) return fail(error, "unknown codec");
    if (from == to) { std::memcpy(dst, src, state_codec_size(floats * 4, from)); return true; }
    // Values are loaded and stored through fixed-size memcpy: these are byte buffers whose alignment is the
    // allocator's, not float or uint16_t arrays, and a reinterpret_cast would promise more than that. The compiler
    // turns the loop into vector loads either way.
    if (from == StateCodec::Fp32 && to == StateCodec::Bf16) {
        for (size_t i = 0; i < floats; ++i) {
            float f = 0;
            std::memcpy(&f, src + i * 4, 4);   // the file and the host buffers are little-endian by contract
            const uint16_t h = bf16_of(f);
            std::memcpy(dst + i * 2, &h, 2);
        }
        return true;
    }
    if (from == StateCodec::Bf16 && to == StateCodec::Fp32) {
        for (size_t i = 0; i < floats; ++i) {
            uint16_t h = 0;
            std::memcpy(&h, src + i * 2, 2);
            const uint32_t b = bf16_bits_of(h);   // the bits f32_of_bf16's value is, written straight out
            std::memcpy(dst + i * 4, &b, 4);
        }
        return true;
    }
    // A codec added to the enum without a pair of paths here lands on the refusal rather than being treated as one
    // of the two conversions below.
    return fail(error, "unknown codec");
}

} // namespace strata::core