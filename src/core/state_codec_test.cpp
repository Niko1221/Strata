#include "strata/core/state_codec.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>
#include <vector>

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

// every bf16 code, widened and coded again
bool every_code_round_trips() {
    std::vector<float> f(65536);
    std::vector<uint8_t> out(65536 * 2);
    std::string error;
    for (uint32_t i = 0; i < 65536; ++i) {
        const uint32_t widened = i << 16;   // bfloat16 is the top half of a float32
        std::memcpy(&f[i], &widened, 4);
    }
    for (uint32_t i = 0; i < 65536; ++i) {
        uint32_t b = 0;
        std::memcpy(&b, &f[i], 4);
        if ((b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu)) continue;   // widened NaNs differ by payload
        if (!state_encode(&f[i], 1, StateCodec::Bf16, out.data() + i * 2, error)) return false;
        uint16_t back = 0;
        std::memcpy(&back, out.data() + i * 2, 2);
        if (back != (uint16_t) i) return false;
    }
    return true;
}

// the worst relative error over a spread of magnitudes, and the sign/zero/extreme cases
void accuracy() {
    std::vector<float> in;
    in.push_back(0.0f); in.push_back(-0.0f); in.push_back(1.0f); in.push_back(-1.0f);
    in.push_back(0.5f); in.push_back(1.0f / 3.0f); in.push_back(-1.0f / 3.0f);
    in.push_back(std::numeric_limits<float>::min());   // the smallest normal, which is also bf16's smallest normal
    in.push_back(1e-30f); in.push_back(1e-8f); in.push_back(1e-3f); in.push_back(3.14159265f);
    in.push_back(1e6f); in.push_back(1e20f); in.push_back(-1e20f); in.push_back(1e38f);
    uint32_t seed = 12345;
    for (int i = 0; i < 4096; ++i) {
        seed = seed * 1664525u + 1013904223u;
        const float magnitude = (float) std::ldexp(1.0, (int) (seed % 80) - 40);
        seed = seed * 1664525u + 1013904223u;
        in.push_back(((seed % 2000) - 1000) / 1000.0f * magnitude);
    }
    std::vector<uint8_t> coded(in.size() * 2);
    std::vector<float> back(in.size());
    std::string error;
    check(state_encode(in.data(), in.size(), StateCodec::Bf16, coded.data(), error), "encode a spread of magnitudes");
    check(state_decode(coded.data(), in.size(), StateCodec::Bf16, back.data(), error), "decode it again");
    const double bound = 1.0 / 256;   // bfloat16 carries 8 significant bits: round-to-nearest-even is within 2^-8
    double worst = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        const float a = in[i], b = back[i];
        if (a == 0.0f) { check(b == 0.0f, "a zero stays a zero with its sign"); continue; }
        if (std::isfinite(a) && !std::isfinite(b)) { check(std::fabs(a) > 3.3e38f, "only a value past the ceiling inflates"); continue; }
        const double rel = std::fabs((b - a) / a);
        if (rel > worst) worst = rel;
    }
    check(worst <= bound, "the worst relative error is within 2^-8");
    check(std::signbit(back[1]), "negative zero keeps its sign");
}

// bf16 has fp32's exponent, so nothing reaches the ceiling by accident; what it gives up is precision, and below its
// smallest denormal a value becomes a zero of the same sign. No value becomes another value's magnitude.
void underflow() {
    const float values[] = {std::numeric_limits<float>::denorm_min(), 1e-40f, -1e-40f, 1e-39f, -1e-39f,
                            std::numeric_limits<float>::min(), -std::numeric_limits<float>::min()};
    uint8_t coded[sizeof values / sizeof *values * 2];
    float back[sizeof values / sizeof *values];
    std::string error;
    check(state_encode(values, 7, StateCodec::Bf16, coded, error) &&
          state_decode(coded, 7, StateCodec::Bf16, back, error), "the subnormal range converts");
    const double step = std::ldexp(1.0, -133);       // one bf16 denormal step
    for (size_t i = 0; i < 7; ++i) {
        check(std::isfinite(back[i]), "nothing in the subnormal range becomes an infinity or a NaN");
        check(std::signbit(back[i]) == std::signbit(values[i]), "and every value keeps its sign");
        // below bf16's smallest normal the grid is fixed, so the error is bounded in absolute terms (half a step),
        // not in relative ones; a value under half a step flushes to a zero of its sign
        check(std::fabs((double) back[i] - (double) values[i]) <= step / 2,
              "a subnormal moves by at most half a bf16 denormal step");
        if (std::fabs(values[i]) < std::ldexp(1.0, -134)) check(back[i] == 0.0f, "under half a step it flushes to zero");
    }
}

void specials() {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float max = std::numeric_limits<float>::max();
    const float values[] = {inf, -inf, nan, max, -max};
    uint8_t coded[sizeof values / sizeof *values * 2];
    float back[sizeof values / sizeof *values];
    std::string error;
    check(state_encode(values, 5, StateCodec::Bf16, coded, error), "encode the extremes");
    check(state_decode(coded, 5, StateCodec::Bf16, back, error), "decode the extremes");
    check(back[0] == inf && back[1] == -inf, "both infinities survive unchanged");
    check(std::isnan(back[2]), "a NaN stays a NaN (a truncated payload would have made it an Inf)");
    check(std::isinf(back[3]) && back[3] > 0 && std::isinf(back[4]) && back[4] < 0,
          "float max rounds to the infinity, which is what round-to-nearest-even means there");
}
// The byte-buffer conversion against the two-step it replaces: decode into an fp32 scratch, encode the scratch. Both
// paths get the same input bytes, and their output must be identical byte for byte - the point of the direct path is
// that it does less work, not that it does different work.
bool two_step(const std::vector<uint8_t>& src, size_t floats, StateCodec from, StateCodec to,
              std::vector<uint8_t>& dst, std::string& error) {
    std::vector<float> scratch(floats, 0.f);
    if (!state_decode(src.data(), floats, from, scratch.data(), error)) return false;
    dst.assign(state_codec_size(floats * 4, to), 0);
    return state_encode(scratch.data(), floats, to, dst.data(), error);
}

void transcode() {
    std::string error;
    std::vector<uint32_t> words;
    // ties in both directions of the even/odd rule, signed zeros, subnormals at both ends of their range, the
    // infinities, float max (which rounds past the ceiling), and NaNs whose payload truncates away to nothing
    for (const uint32_t b : {0x00000000u, 0x80000000u, 0x3F800001u, 0x3F807FFFu, 0x3F808000u, 0x3F808001u,
                             0x3F80FFFFu, 0x3F818000u, 0x3F7FFFFFu, 0x00000001u, 0x80000001u, 0x007FFFFFu,
                             0x00800000u, 0x00400000u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F800000u, 0xFF800000u,
                             0x7FC00000u, 0x7F800001u, 0xFF800001u, 0xFFC00001u, 0x7F807FFEu})
        words.push_back(b);
    uint32_t seed = 20251013u;
    for (int i = 0; i < 20000; ++i) {   // ordinary magnitudes and raw bit patterns alike, NaNs included
        seed = seed * 1664525u + 1013904223u;
        words.push_back(seed ^ (seed >> 11) * 2246822519u);
    }
    std::vector<uint8_t> fp32(words.size() * 4, 0);
    for (size_t i = 0; i < words.size(); ++i) std::memcpy(fp32.data() + i * 4, &words[i], 4);

    std::vector<uint8_t> direct(fp32.size() / 2), coded_ref, back(fp32.size()), widened_ref;
    check(state_transcode_bytes(fp32.data(), words.size(), StateCodec::Fp32, StateCodec::Bf16, direct.data(), error),
          "fp32 bytes to bf16 bytes in one pass");
    check(two_step(fp32, words.size(), StateCodec::Fp32, StateCodec::Bf16, coded_ref, error), "the scratch path too");
    check(direct == coded_ref, "and both write the same bf16 bytes");

    check(state_transcode_bytes(direct.data(), words.size(), StateCodec::Bf16, StateCodec::Fp32, back.data(), error),
          "bf16 bytes back to fp32 bytes in one pass");
    check(two_step(direct, words.size(), StateCodec::Bf16, StateCodec::Fp32, widened_ref, error), "and the scratch path");
    check(back == widened_ref, "both give the same fp32 bytes back");
    bool shift_only = true;
    for (size_t i = 0; i < words.size(); ++i) {
        uint16_t h = 0;
        uint32_t w = 0;
        std::memcpy(&h, direct.data() + i * 2, 2);
        std::memcpy(&w, back.data() + i * 4, 4);
        if (w != (((uint32_t) h) << 16)) shift_only = false;
    }
    check(shift_only, "widening is exactly the shift: every fp32 word is the bf16 code with sixteen zero bits under it");

    // a codec that is already the one asked for copies the bytes that array occupies, in both codecs
    std::vector<uint8_t> same(fp32.size());
    check(state_transcode_bytes(fp32.data(), words.size(), StateCodec::Fp32, StateCodec::Fp32, same.data(), error) &&
          same == fp32, "fp32 to fp32 is the bytes it started with");
    std::vector<uint8_t> same_coded(direct.size());
    check(state_transcode_bytes(direct.data(), words.size(), StateCodec::Bf16, StateCodec::Bf16, same_coded.data(), error) &&
          same_coded == direct, "bf16 to bf16 is the bytes it started with");

    // small counts, odd counts, and nothing to convert
    for (const size_t n : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(5), size_t(17), size_t(1023)}) {
        std::vector<uint8_t> d(n ? n * 2 : 1, 0xAA), r;
        check(state_transcode_bytes(fp32.data(), n, StateCodec::Fp32, StateCodec::Bf16, d.data(), error) &&
              two_step(fp32, n, StateCodec::Fp32, StateCodec::Bf16, r, error), "a small or odd count converts");
        check(std::equal(r.begin(), r.end(), d.begin()), "and matches the scratch path at that count");
        if (n == 0) check(d[0] == 0xAA, "nothing to convert writes nothing");
        std::vector<uint8_t> d2(n ? n * 4 : 1, 0xBB), r2;
        check(state_transcode_bytes(direct.data(), n, StateCodec::Bf16, StateCodec::Fp32, d2.data(), error) &&
              two_step(direct, n, StateCodec::Bf16, StateCodec::Fp32, r2, error) &&
              std::equal(r2.begin(), r2.end(), d2.begin()), "the same both ways round");
        if (n == 0) check(d2[0] == 0xBB, "and writes nothing when there is nothing to convert");
    }

    // refusals, the same shape as the other two entry points
    check(state_transcode_bytes(nullptr, 0, StateCodec::Fp32, StateCodec::Bf16, nullptr, error),
          "a null pair with nothing to convert is not a failure");
    check(!state_transcode_bytes(nullptr, 4, StateCodec::Fp32, StateCodec::Bf16, nullptr, error) && !error.empty(),
          "a null array with values to convert fails and says why");
    error.clear();
    const auto unknown = (StateCodec) 2;
    check(!state_transcode_bytes(fp32.data(), 4, unknown, StateCodec::Bf16, direct.data(), error) && !error.empty(),
          "a codec this build does not have is refused as the source");
    check(!state_transcode_bytes(fp32.data(), 4, StateCodec::Fp32, unknown, direct.data(), error) && !error.empty(),
          "and as the destination, not silently copied");
}
} // namespace

int main() {
    std::string error;

    // sizes
    check(state_codec_size(0, StateCodec::Fp32) == 0, "fp32: no bytes is no bytes");
    check(state_codec_size(118063152, StateCodec::Fp32) == 118063152, "fp32 is the identity");
    check(state_codec_size(118063152, StateCodec::Bf16) == 59031576, "bf16 is half");
    check(state_codec_fp32_bytes(59031576, StateCodec::Bf16) == 118063152, "and the inverse finds the fp32 bytes back");
    check(state_codec_fp32_bytes(118063152, StateCodec::Fp32) == 118063152, "fp32 inverse is the identity");
    check(state_codec_fp32_bytes(SIZE_MAX, StateCodec::Bf16) == 0, "a size that would overflow is refused");

    // names and file codes
    StateCodec codec = StateCodec::Bf16;
    check(state_codec_by_name("fp32", codec) && codec == StateCodec::Fp32, "the option name fp32");
    check(state_codec_by_name("BF16", codec) && codec == StateCodec::Bf16, "the option name is case-insensitive");
    check(state_codec_by_name("bfloat16", codec) && codec == StateCodec::Bf16, "the long name");
    check(!state_codec_by_name("", codec), "an empty name is refused");
    check(!state_codec_by_name("int8", codec), "a codec this build does not have is refused, not defaulted");
    check(!state_codec_by_name("fp32x", codec), "a name with trailing text is refused");
    check(std::strcmp(state_codec_name(StateCodec::Fp32), "fp32") == 0, "the name round-trips");
    check(state_codec_by_code(0, codec) && codec == StateCodec::Fp32, "file code 0 is fp32");
    check(state_codec_by_code(1, codec) && codec == StateCodec::Bf16, "file code 1 is bf16");
    check(!state_codec_by_code(2, codec), "an unknown file code is refused");
    check(!state_codec_by_code(256, codec), "a code that is not a byte is refused");

    // fp32 is today's bytes
    {
        const std::vector<float> in{1.0f / 3.0f, -2.5e-7f, 1e30f, 0.0f, -0.0f};
        std::vector<uint8_t> coded(in.size() * 4), today(in.size() * 4);
        std::vector<float> back(in.size());
        check(state_encode(in.data(), in.size(), StateCodec::Fp32, coded.data(), error), "fp32 encode");
        std::memcpy(today.data(), in.data(), in.size() * 4);
        check(coded == today, "fp32 encode writes the bytes the state already has");
        check(state_decode(coded.data(), in.size(), StateCodec::Fp32, back.data(), error), "fp32 decode");
        check(back == in, "fp32 round trip is bit-exact");
    }

    // refusals
    check(state_encode(nullptr, 0, StateCodec::Bf16, nullptr, error), "nothing to convert is not a failure");
    check(!state_encode(nullptr, 4, StateCodec::Bf16, nullptr, error) && !error.empty(),
          "a missing array with values to convert fails and says why");
    check(!state_decode(nullptr, 4, StateCodec::Bf16, nullptr, error), "the same on the way back");

    check(every_code_round_trips(), "every bf16 code widens and codes back to itself");
    accuracy();
    underflow();
    specials();
    transcode();

    std::printf("state_codec_test: %d checks passed\n", checks);
}