// tests/test_dq_traits.cpp - the CUDA decode math, checked on the HOST.
//
// include/dsv4/dq_traits.hpp holds the per-type decode code that the matvec kernel in src/gpu_cuda.cu
// runs on the device. Compiled for the host (no DSV4_HOST_DEVICE, plain iq:: tables), the very same lines
// can be compared against dequant_row() and row_dot() from src/dequant.cpp, which the golden tests already
// trust. That is what keeps a wrong nibble shift in a kernel from becoming silent garbage in the logits:
// this test needs no GPU, no CUDA toolkit and no model file.
//
// For every supported type: fill a row with random bytes, decode it both ways, require bit-identical
// floats. Byte-exact is the right bar here - both sides read the same bytes with the same arithmetic, so
// any difference is a transcription bug, not rounding.
#include "dsv4/dequant.hpp"
#include "dsv4/dq_traits.hpp"
#include "dsv4/gguf_header.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;

namespace {

// xorshift64*: deterministic, so a failure is reproducible from the same source line.
uint64_t rs = 0x9E3779B97F4A7C15ULL;
uint64_t nxt() { rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27; return rs * 2685821657736338717ULL; }

bool bits_eq(float a, float b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    uint32_t ua, ub; std::memcpy(&ua, &a, 4); std::memcpy(&ub, &b, 4);
    return ua == ub;
}

/// Decode `rows` random rows of `type` twice and compare. Returns the number of mismatching elements.
template <class Tr>
size_t check_type(uint32_t type, int64_t in, int rows) {
    const size_t rb = dsv4::row_bytes(type, in);
    if (!rb) { std::fprintf(stderr, "FAIL row_bytes(%s, %lld) == 0\n", dsv4::ggml_type_str(type).c_str(), (long long) in); ++g_fail; return 1; }
    if ((int64_t)(in / Tr::BE) * (int64_t) Tr::BB != (int64_t) rb) {
        std::fprintf(stderr, "FAIL %s: trait block %dx%d != row_bytes %zu\n", dsv4::ggml_type_str(type).c_str(),
                     Tr::BE, Tr::BB, rb);
        ++g_fail;
        return 1;
    }
    std::vector<uint8_t> w((size_t) rb * rows);
    for (uint8_t& b : w) b = (uint8_t) (nxt() & 0xFF);
    std::vector<float> ref((size_t) in), got((size_t) in);
    size_t bad = 0;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row = w.data() + (size_t) r * rb;
        std::fill(got.begin(), got.end(), 0.f);
        dsv4::dequant_row(type, row, in, ref.data());
        for (int64_t e = 0; e < in; ++e) got[(size_t) e] = Tr::at(row, (int) e);
        for (int64_t e = 0; e < in; ++e) {
            if (!bits_eq(ref[(size_t) e], got[(size_t) e])) {
                if (bad < 3)
                    std::fprintf(stderr, "  %s row %d elem %lld: dequant=%+.8g trait=%+.8g\n",
                                 dsv4::ggml_type_str(type).c_str(), r, (long long) e,
                                 (double) ref[(size_t) e], (double) got[(size_t) e]);
                ++bad;
            }
        }
    }
    const size_t total = (size_t) in * rows;
    if (bad) {
        std::fprintf(stderr, "FAIL %-9s %2lldx%-6lld : %zu / %zu elements differ\n",
                     dsv4::ggml_type_str(type).c_str(), (long long) rows, (long long) in, bad, total);
        ++g_fail;
    } else {
        std::printf("ok  %-9s %2lld rows x %-6lld elements : bit-identical to dequant_row\n",
                    dsv4::ggml_type_str(type).c_str(), (long long) rows, (long long) in);
    }
    return bad;
}

/// The kernel multiplies the decoded row by x and sums; row_dot does the same on the host. Same check,
/// through the dot product, for the shapes the model really uses.
template <class Tr>
void check_dot(uint32_t type, int64_t in, int rows, double tol) {
    const size_t rb = dsv4::row_bytes(type, in);
    if (!rb) return;
    std::vector<uint8_t> w((size_t) rb * rows);
    for (uint8_t& b : w) b = (uint8_t) (nxt() & 0xFF);
    std::vector<float> x((size_t) in);
    for (int64_t i = 0; i < in; ++i) x[(size_t) i] = (float) (((nxt() >> 11) % 2000) - 1000) / 500.f;
    double worst = 0;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row = w.data() + (size_t) r * rb;
        const float want = dsv4::row_dot(type, row, x.data(), in);
        double acc = 0;
        for (int64_t e = 0; e < in; ++e) acc += (double) Tr::at(row, (int) e) * (double) x[(size_t) e];
        const double scale = std::max(1.0, std::fabs((double) want));
        worst = std::max(worst, std::fabs(acc - (double) want) / scale);
    }
    if (worst > tol) {
        std::fprintf(stderr, "FAIL %-9s dot %dx%-6lld : rel err %.3g > %.3g\n",
                     dsv4::ggml_type_str(type).c_str(), rows, (long long) in, worst, tol);
        ++g_fail;
    } else {
        std::printf("ok  %-9s dot %dx%-6lld : max rel err %.2g (float tree sum vs double left-to-right)\n",
                    dsv4::ggml_type_str(type).c_str(), rows, (long long) in, worst);
    }
}

}  // namespace

int main() {
    // The shapes DeepSeek-V4-Flash actually stores (shapes.txt): in = the row length the model uses.
    check_type<dqt::F32T>(dsv4::T_F32, 4096, 4);
    check_type<dqt::BF16T>(dsv4::T_BF16, 4096, 4);          // ffn_gate_inp
    check_type<dqt::Q8_0T>(dsv4::T_Q8_0, 4096, 4);          // attn_kv, attn_q_b, attn_output_*
    check_type<dqt::Q8_0T>(dsv4::T_Q8_0, 1024, 4);
    check_type<dqt::Q4_KT>(dsv4::T_Q4_K, 4096, 4);          // token_embd, output
    check_type<dqt::Q5_KT>(dsv4::T_Q5_K, 4096, 4);          // attn_q_a, ffn_*_shexp
    check_type<dqt::Q6_KT>(dsv4::T_Q6_K, 2048, 4);          // ffn_down_shexp
    check_type<dqt::IQ3_XXST>(dsv4::T_IQ3_XXS, 4096, 4);    // ffn_gate_exps
    check_type<dqt::IQ3_XXST>(dsv4::T_IQ3_XXS, 2048, 4);    // ffn_down_exps
    check_type<dqt::IQ2_XXST>(dsv4::T_IQ2_XXS, 4096, 4);
    check_type<dqt::IQ2_XXST>(dsv4::T_IQ2_XXS, 2048, 4);
    check_type<dqt::IQ1_MT>(dsv4::T_IQ1_M, 4096, 4);
    check_type<dqt::IQ1_MT>(dsv4::T_IQ1_M, 2048, 4);
    check_type<dqt::MXFP4T>(dsv4::T_MXFP4, 4096, 4);

    // And the same math through a dot product, which is what the kernel actually computes.
    check_dot<dqt::Q8_0T>(dsv4::T_Q8_0, 4096, 8, 1e-5);
    check_dot<dqt::Q4_KT>(dsv4::T_Q4_K, 4096, 8, 1e-5);
    check_dot<dqt::Q5_KT>(dsv4::T_Q5_K, 4096, 8, 1e-5);
    check_dot<dqt::Q6_KT>(dsv4::T_Q6_K, 2048, 8, 1e-5);
    check_dot<dqt::IQ3_XXST>(dsv4::T_IQ3_XXS, 4096, 8, 1e-5);
    check_dot<dqt::IQ2_XXST>(dsv4::T_IQ2_XXS, 4096, 8, 1e-5);
    check_dot<dqt::IQ1_MT>(dsv4::T_IQ1_M, 4096, 8, 1e-5);
    check_dot<dqt::MXFP4T>(dsv4::T_MXFP4, 4096, 8, 1e-5);

    std::printf("%s\n", g_fail ? "TRAIT TESTS FAILED" : "ALL TRAIT TESTS PASSED");
    return g_fail ? 1 : 0;
}
