#include "strata/program/remote_helper_options.hpp"

#ifdef NDEBUG
#undef NDEBUG  // Keep assertions active in Release-mode CTest builds.
#endif
#include <cassert>
#include <cstddef>
#include <limits>

using strata::program::remote_helpers::OptionName;
using strata::program::remote_helpers::enabled_helper_count;
using strata::program::remote_helpers::has_enabled_helpers;
using strata::program::remote_helpers::has_helper_gap;
using strata::program::remote_helpers::parse_cache_size;
using strata::program::remote_helpers::parse_option_name;

int main() {
    size_t index = 0;
    assert(parse_option_name("--expert-cache-device1", index) == OptionName::Valid && index == 0);
    assert(parse_option_name("--expert-cache-device3", index) == OptionName::Valid && index == 2);
    assert(parse_option_name("--expert-cache-device12", index) == OptionName::Valid && index == 11);
    assert(parse_option_name("--expert-cache-device", index) == OptionName::InvalidHelperIndex);
    assert(parse_option_name("--expert-cache-device0", index) == OptionName::InvalidHelperIndex);
    assert(parse_option_name("--expert-cache-device2x", index) == OptionName::InvalidHelperIndex);
    assert(parse_option_name("--expert-cache-device18446744073709551616", index) == OptionName::InvalidHelperIndex);
    assert(parse_option_name("--expert-cache", index) == OptionName::NotHelperOption);

    int slots = -1;
    bool automatic = false;
    assert(parse_cache_size("auto", slots, automatic) && slots == std::numeric_limits<int>::max() && automatic);
    assert(parse_cache_size("0", slots, automatic) && slots == 0 && !automatic);
    assert(parse_cache_size("8192", slots, automatic) && slots == 8192 && !automatic);
    assert(!parse_cache_size("-1", slots, automatic));
    assert(!parse_cache_size("12x", slots, automatic));
    assert(!parse_cache_size("2147483648", slots, automatic));
    assert(!parse_cache_size("", slots, automatic));

    assert(!has_enabled_helpers({0, 0, 0}));
    assert(enabled_helper_count({0, 0, 0}) == 0);
    assert(!has_helper_gap({0, 0, 0}));
    assert(has_enabled_helpers({32, 64, 128, 256}));
    assert(enabled_helper_count({32, 64, 128, 256}) == 4);
    assert(!has_helper_gap({32, 64, 128, 256}));
    assert(has_helper_gap({32, 0, 128}));
}
