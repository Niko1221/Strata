#include "dp4a_cases.hpp"
#include <cstdio>

int main() {
    size_t tested = 0;
#ifdef P100_SCALAR_REFERENCE
    size_t scalar_tested = 0;
#endif
    for (const auto &v : p100_test::cases()) {
        const p100_test::Case aliases[] = {v, {v.a, v.b, v.a}, {v.a, v.b, v.b},
                                          {v.a, v.a, v.c}, {v.a, v.a, v.a}};
        for (auto x : aliases) {
            if (p100_test::vmad_model(x.a, x.b, x.c) != p100_test::oracle(x.a, x.b, x.c)) {
                std::fprintf(stderr, "arithmetic model mismatch\n");
                return 1;
            }
            ++tested;
#ifdef P100_SCALAR_REFERENCE
            int64_t partial = x.c;
            bool safe = true;
            int8_t a[4], b[4];
            std::memcpy(a, &x.a, 4); std::memcpy(b, &x.b, 4);
            for (int j = 0; j < 4; ++j) {
                partial += int64_t(a[j]) * b[j];
                safe = safe && partial >= -2147483648LL && partial <= 2147483647LL;
            }
            if (safe) {
                if (strata_dp4a(x.a, x.b, x.c) != p100_test::oracle(x.a, x.b, x.c)) return 1;
                ++scalar_tested;
            }
#endif
        }
    }
    if (p100_test::oracle(p100_test::from_bits(0x80808080u),
                          p100_test::from_bits(0x80808080u), 0) != 65536 ||
        p100_test::oracle(0x01010101, -1, 0) != -4 ||
        p100_test::oracle(0x01010101, 0x01010101, 2147483647) != -2147483645) return 1;
    std::printf("PASS: %zu arithmetic-model cases; CPU-only check does not execute production CUDA assembly\n", tested);
#ifdef P100_SCALAR_REFERENCE
    std::printf("PASS: %zu defined pristine scalar cases under UBSan\n", scalar_tested);
#endif
}
