// cann/tests/quant_parity.cpp - do our Q4_K/Q6_K decoders agree with ggml, and does a
// dequantize-then-matmul on the NPU reproduce the CPU result?
//
// The oracle is ggml's own `to_float` for the same type (linked from libggml-base), so a
// mismatch here means our block layout or the scale unpacking is wrong, not that the math
// differs in the last bit.
#include <acl/acl.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "strata/cann/device.hpp"
#include "strata/cann/ops.hpp"
#include "strata/cann/quants.hpp"

#ifdef STRATA_CANN_HAVE_GGML
#include <ggml.h>
#endif

using namespace strata::cann;

namespace {

constexpr int GGML_TYPE_Q4_K_ID = 12;
constexpr int GGML_TYPE_Q6_K_ID = 14;

std::vector<uint8_t> random_blocks(size_t bytes, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<uint8_t> v(bytes);
    for (auto & b : v) {
        b = (uint8_t) byte(rng);
    }
    return v;
}

// Put a sane fp16 scale/min into the block header so the values stay in a usable range.
void set_half(uint8_t * at, float value) {
    uint16_t h;
    f32_to_f16(&value, &h, 1);
    memcpy(at, &h, 2);
}

double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs((double) a[i] - (double) b[i]));
    }
    return m;
}

} // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int failures = 0;

    // ---- 1. dequant parity against ggml ----
    struct QuantCase {
        int type;
        size_t block_bytes;
        const char * name;
        size_t header_scale_offset;
        size_t header_min_offset; // SIZE_MAX for Q6_K (no min)
    };
    const QuantCase cases[] = {
        { GGML_TYPE_Q4_K_ID, 144, "Q4_K", 0, 2 },
        { GGML_TYPE_Q6_K_ID, 210, "Q6_K", 208, (size_t) -1 },
    };

    const int64_t n_values = QK_K * 8;
    for (const QuantCase & c : cases) {
        std::vector<uint8_t> bytes = random_blocks((size_t) (n_values / QK_K) * c.block_bytes, 100 + c.type);
        for (int64_t b = 0; b < n_values / QK_K; ++b) {
            uint8_t * blk = bytes.data() + b * c.block_bytes;
            set_half(blk + c.header_scale_offset, 0.0123f);
            if (c.header_min_offset != (size_t) -1) {
                set_half(blk + c.header_min_offset, 0.0071f);
            }
        }

        std::vector<float> ours(n_values);
        std::vector<float> ref(n_values);
        dequantize(c.type, bytes.data(), ours.data(), n_values);

#ifdef STRATA_CANN_HAVE_GGML
        const ggml_type_traits * traits = ggml_get_type_traits((enum ggml_type) c.type);
        if (!traits || !traits->to_float) {
            printf("%-6s FAIL: ggml has no to_float for this type\n", c.name);
            ++failures;
            continue;
        }
        traits->to_float(bytes.data(), ref.data(), n_values);
        const double d = max_abs_diff(ours, ref);
        const bool same = memcmp(ours.data(), ref.data(), ours.size() * sizeof(float)) == 0;
        printf("%-6s vs ggml: max|diff| = %.9g  %s\n", c.name, d, same ? "BIT-EXACT" : "DIFFERS");
        if (!same) {
            ++failures;
        }
#else
        (void) ref;
        printf("%-6s: built without the ggml oracle, decoder ran only\n", c.name);
#endif
    }

    // ---- 2. dequantize a weight matrix and multiply it on the NPU ----
    {
        const int64_t M = 64, K = 512, N = 256;
        std::vector<uint8_t> wbytes((size_t) (K * N / QK_K) * 144, 0);
        for (int64_t b = 0; b < K * N / QK_K; ++b) {
            uint8_t * blk = wbytes.data() + b * 144;
            // random payload, modest scales so the dot product stays in fp16 range
            std::vector<uint8_t> rnd = random_blocks(128 + 12, 7 + (uint32_t) b);
            memcpy(blk + 4, rnd.data(), 12 + 128);
            set_half(blk + 0, 0.005f);
            set_half(blk + 2, 0.004f);
        }
        std::vector<float> wf((size_t) K * N);
        dequantize(GGML_TYPE_Q4_K_ID, wbytes.data(), wf.data(), K * N);

        std::mt19937 rng(11);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        std::vector<float> af((size_t) M * K);
        for (auto & x : af) {
            x = dist(rng);
        }

        // fp16 rounding of both operands, as the device sees them
        std::vector<float> a16(af.size()), w16(wf.size());
        {
            std::vector<uint16_t> ha(af.size()), hw(wf.size());
            f32_to_f16(af.data(), ha.data(), af.size());
            f16_to_f32(ha.data(), a16.data(), a16.size());
            f32_to_f16(wf.data(), hw.data(), wf.size());
            f16_to_f32(hw.data(), w16.data(), w16.size());
        }

        const std::vector<float> ref = matmul_ref(a16, w16, M, K, N);

        init();
        Device device(0);
        Stream stream(device);

        std::vector<uint16_t> hw(w16.size()), ha(a16.size());
        f32_to_f16(w16.data(), hw.data(), hw.size());
        f32_to_f16(a16.data(), ha.data(), ha.size());

        Buffer bw(w16.size() * 2), ba(a16.size() * 2), bc((size_t) M * N * 2);
        bw.copy_from_host(hw.data(), hw.size() * 2);
        ba.copy_from_host(ha.data(), ha.size() * 2);

        Tensor tw{ bw.data(), DType::F16, N, K };
        Tensor ta{ ba.data(), DType::F16, K, M };
        Tensor tc{ bc.data(), DType::F16, N, M };
        matmul(stream, ta, tw, tc);

        std::vector<uint16_t> hc((size_t) M * N);
        bc.copy_to_host(hc.data(), hc.size() * 2);
        stream.synchronize();
        std::vector<float> got((size_t) M * N);
        f16_to_f32(hc.data(), got.data(), got.size());

        const double d = max_abs_diff(got, ref);
        const bool ok = d <= 0.5;
        printf("Q4_K dequant -> NPU matmul %lldx%lldx%lld: max|diff| vs CPU = %.6f  %s\n", (long long) M,
               (long long) K, (long long) N, d, ok ? "PASS" : "FAIL");
        if (!ok) {
            ++failures;
        }
    }

    shutdown();
    if (failures == 0) {
        printf("\nall quant checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
