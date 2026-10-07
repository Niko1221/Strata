// Compact MLA cache. FP8 is scaled E4M3FN, with separate latent and RoPE scales.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::glm {
enum class KvFormat { F32, BF16, FP8 };
inline const char* kv_name(KvFormat f) { return f == KvFormat::F32 ? "f32" : f == KvFormat::BF16 ? "bf16" : "fp8"; }
inline KvFormat kv_format(const std::string& s) {
    if (s == "f32") return KvFormat::F32;
    if (s == "bf16") return KvFormat::BF16;
    if (s == "fp8") return KvFormat::FP8;
    throw std::invalid_argument("--kv must be f32, bf16 or fp8");
}
inline uint16_t to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x7fffffu) ? 0x40u : 0u));
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline float from_bf16(uint16_t u) { uint32_t b = (uint32_t)u << 16; float f; std::memcpy(&f, &b, 4); return f; }
inline float from_fp8(uint8_t u) {
    const int e = (u >> 3) & 15, m = u & 7;
    const float x = e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp((float)m, -9);
    return (u & 128) ? -x : x;
}
inline uint8_t to_fp8(float x) {
    const uint8_t sign = std::signbit(x) ? 128 : 0;
    x = std::min(448.0f, std::fabs(x));
    if (x < 0.015625f) return sign | (uint8_t)std::nearbyint(x * 512.0f);
    int e; std::frexp(x, &e); --e;
    int m = (int)std::nearbyint(std::ldexp(x, 3 - e));
    if (m == 16) { ++e; m = 8; }
    return sign | (uint8_t)std::min(126, ((e + 7) << 3) + m - 8);
}

class KvCache {
public:
    void reset(int layers, int context, int latent, int rope, KvFormat fmt) {
        fmt_ = fmt; ctx_ = context; latent_ = latent; width_ = latent + rope;
        stride_ = width_ * (fmt == KvFormat::F32 ? 4 : fmt == KvFormat::BF16 ? 2 : 1)
                + (fmt == KvFormat::FP8 ? 8 : 0);
        data_.assign((size_t)layers * context * stride_, 0);
        for (int i = 0; i < 256; ++i) lut_[i] = from_fp8((uint8_t)i);
    }
    uint64_t bytes() const { return data_.size(); }
    /// The f32 cache's rows of `layer` in place (nullptr for bf16/fp8, which must be read through read_layer).
    const float* f32_layer(int layer) const {
        return fmt_ == KvFormat::F32 ? (const float*) (data_.data() + (size_t) layer * ctx_ * stride_) : nullptr;
    }
    KvFormat format() const { return fmt_; }
    void write(int layer, int pos, const float* src) {
        uint8_t* p = data_.data() + ((size_t)layer * ctx_ + pos) * stride_;
        if (fmt_ == KvFormat::F32) { std::memcpy(p, src, width_ * 4); return; }
        if (fmt_ == KvFormat::BF16) {
            for (int i = 0; i < width_; ++i) { uint16_t b = to_bf16(src[i]); std::memcpy(p + i * 2, &b, 2); }
            return;
        }
        for (int part = 0; part < 2; ++part) {
            const int b = part ? latent_ : 0, e = part ? width_ : latent_;
            float scale = 0;
            for (int i = b; i < e; ++i) scale = std::max(scale, std::fabs(src[i]));
            scale = scale > 0 ? scale / 448.0f : 1.0f;
            std::memcpy(p + width_ + part * 4, &scale, 4);
            for (int i = b; i < e; ++i) p[i] = to_fp8(src[i] / scale);
        }
    }
    void read_layer(int layer, int count, float* dst) const {
        const uint8_t* p = data_.data() + (size_t)layer * ctx_ * stride_;
        if (fmt_ == KvFormat::F32) { std::memcpy(dst, p, (size_t)count * width_ * 4); return; }
        for (int s = 0; s < count; ++s, p += stride_, dst += width_) {
            if (fmt_ == KvFormat::BF16) {
                for (int i = 0; i < width_; ++i) { uint16_t b; std::memcpy(&b, p + i * 2, 2); dst[i] = from_bf16(b); }
            } else {
                float scales[2]; std::memcpy(scales, p + width_, 8);
                for (int i = 0; i < width_; ++i) dst[i] = lut_[p[i]] * scales[i >= latent_];
            }
        }
    }
private:
    KvFormat fmt_ = KvFormat::F32;
    int ctx_ = 0, latent_ = 0, width_ = 0, stride_ = 0;
    float lut_[256]{};
    std::vector<uint8_t> data_;
};
} // namespace strata::glm
