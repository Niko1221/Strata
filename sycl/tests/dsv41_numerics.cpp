// Independent numeric oracle for the current DeepSeek SYCL port.
#include "ds4_numerics.dp.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {
float decode(unsigned q) {
    const unsigned exponent = (q >> 3) & 15, fraction = q & 7;
    if (exponent == 15 && fraction == 7) return std::numeric_limits<float>::quiet_NaN();
    const float value = exponent == 0 ? std::ldexp(float(fraction), -9)
        : std::ldexp(1.0f + float(fraction) / 8, int(exponent) - 7);
    return std::copysign(value, (q & 128) ? -1.0f : 1.0f);
}

float round_reference(float x) {
    if (std::isnan(x)) return x;
    const double magnitude = std::fabs(double(x));
    if (magnitude >= 448) return std::copysign(448.0f, x);
    unsigned nearest = 0;
    double distance = std::numeric_limits<double>::infinity();
    // Exhaust the representable values rather than sharing the port's exponent
    // rounding algorithm. An even encoding wins a midpoint tie.
    for (unsigned q = 0; q < 127; ++q) {
        const double error = std::fabs(magnitude - double(decode(q)));
        if (error < distance || (error == distance && !(q & 1))) {
            nearest = q;
            distance = error;
        }
    }
    return std::copysign(decode(nearest), x);
}

bool same(float a, float b) {
    return (std::isnan(a) && std::isnan(b)) || std::bit_cast<unsigned>(a) == std::bit_cast<unsigned>(b);
}
}

int main() try {
    std::vector<float> input{0, -0.0f, 448, -448,
        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()};
    for (unsigned q = 0; q < 127; ++q) {
        input.push_back(decode(q));
        input.push_back(-decode(q));
        if (q < 126) {
            const float midpoint = (decode(q) + decode(q + 1)) / 2;
            for (float value : {std::nextafter(midpoint, 0.0f), midpoint,
                               std::nextafter(midpoint, std::numeric_limits<float>::infinity())}) {
                input.push_back(value);
                input.push_back(-value);
            }
        }
    }
    unsigned state = 42;
    for (int i = 0; i < 8192; ++i) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        input.push_back(std::bit_cast<float>(state));
    }
    sycl::queue queue(sycl::gpu_selector_v);
    float* values = sycl::malloc_shared<float>(input.size(), queue);
    float* rounded = sycl::malloc_shared<float>(input.size(), queue);
    float* decoded = sycl::malloc_shared<float>(256, queue);
    if (!values || !rounded || !decoded) return 2;
    std::copy(input.begin(), input.end(), values);
    const size_t size = input.size();
    queue.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
        rounded[i] = strata::ds4::e4m3_round(values[i]);
    });
    queue.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) {
        decoded[i] = strata::ds4::e4m3_value(uint8_t(i[0]));
    });
    queue.wait_and_throw();
    size_t failures = 0;
    for (size_t i = 0; i < size; ++i) {
        if (!same(rounded[i], round_reference(input[i]))) {
            if (failures++ < 8) std::fprintf(stderr, "round[%zu] input=%a got=%a expected=%a\n",
                i, input[i], rounded[i], round_reference(input[i]));
        }
    }
    for (unsigned q = 0; q < 256; ++q) if (!same(decoded[q], decode(q))) ++failures;
    sycl::free(decoded, queue); sycl::free(rounded, queue); sycl::free(values, queue);
    std::printf("DeepSeek E4M3: %zu rounding inputs, 256 codes, %zu failures\n", size, failures);
    return failures ? 1 : 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 2;
}
