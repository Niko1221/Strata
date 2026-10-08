#include "glm_model_options.hpp"
#include <cstdio>
#include <cstdlib>
int main() {
    using namespace strata::core::glm_model_options;
    std::vector<int> values;
    int checks = 0;
    const auto check = [&](bool okay) { ++checks; if (!okay) std::exit(1); };
    check(integer_list(" 0 , 2 , 3 ", values) && values == std::vector<int>({0, 2, 3}));
    for (const char* text : {"", "1 2", "0,1,", "-1", "x", "999999999999999999999"}) check(!integer_list(text, values));
    check(devices_ok({0, 2}, 3, 45, 16));
    check(!devices_ok({4}, 4, 45, 16));
    check(!devices_ok({0, 0}, 2, 45, 16));
    values.clear(); for (int i = 0; i < 17; ++i) values.push_back(i);
    check(!devices_ok(values, 20, 45, 16));
    check(!devices_ok({0, 1, 2}, 3, 2, 16));
    std::printf("glm_model_options_test: PASS (%d checks)\n", checks);
}
