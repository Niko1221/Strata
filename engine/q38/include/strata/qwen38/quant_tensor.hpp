#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::qwen38 {

/// GGML / Strata-native quantization type IDs supported by strata-q38.
enum class GgmlType : int32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q8_0    = 8,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    BF16    = 30,
    PQ2_0   = 142,
};

/// Self-describing device/host-pinned weight tensor for per-tensor mixed quantization.
struct StrataQuantTensor {
    const void* d_data         = nullptr;
    GgmlType    type           = GgmlType::F16;
    int32_t     ne0            = 0;       ///< Input dimension K (contiguous row length)
    int32_t     ne1            = 0;       ///< Output dimension N (number of rows)
    size_t      row_bytes      = 0;       ///< Exact byte stride per row
    size_t      total_bytes    = 0;       ///< row_bytes * ne1
    bool        is_host_pinned = false;

    static constexpr size_t block_elements(GgmlType t) {
        switch (t) {
            case GgmlType::F32:
            case GgmlType::F16:
            case GgmlType::BF16:   return 1;
            case GgmlType::Q4_0:
            case GgmlType::Q8_0:
            case GgmlType::IQ4_NL: return 32;
            case GgmlType::PQ2_0:  return 128;
            default:               return 256;
        }
    }

    static constexpr size_t block_bytes(GgmlType t) {
        switch (t) {
            case GgmlType::F32:    return 4;
            case GgmlType::F16:
            case GgmlType::BF16:   return 2;
            case GgmlType::Q4_0:   return 18;
            case GgmlType::Q8_0:   return 34;
            case GgmlType::Q4_K:   return 144;
            case GgmlType::Q5_K:   return 176;
            case GgmlType::Q6_K:   return 210;
            case GgmlType::IQ3_S:  return 110;
            case GgmlType::IQ2_S:  return 82;
            case GgmlType::IQ4_XS: return 136;
            case GgmlType::PQ2_0:  return 34;
            default:               return 0;
        }
    }
};

inline StrataQuantTensor make_quant_tensor(const void* data, int32_t type_id, int32_t ne0, int32_t ne1,
                                           bool host_pinned = false) {
    const auto t = static_cast<GgmlType>(type_id);
    const size_t be = StrataQuantTensor::block_elements(t);
    const size_t bb = StrataQuantTensor::block_bytes(t);
    const size_t rb = (be > 0 && bb > 0 && ne0 > 0) ? (static_cast<size_t>(ne0) / be) * bb : 0;
    return StrataQuantTensor{
        data, t, ne0, ne1, rb, rb * static_cast<size_t>(ne1 > 0 ? ne1 : 0), host_pinned,
    };
}

} // namespace strata::qwen38
