#pragma once

#include <charconv>
#include <cstddef>
#include <limits>
#include <string_view>
#include <system_error>
#include <vector>

namespace strata::program::remote_helpers {

enum class OptionName { NotHelperOption, InvalidHelperIndex, Valid };

/// Parse the backward-compatible `--expert-cache-deviceN` name into its zero-based helper position.
/// The number is a helper ordinal, not a physical CUDA device id; runtime mapping happens after all
/// options have been read. Returns InvalidHelperIndex for a malformed or overflowing suffix.
inline OptionName parse_option_name(std::string_view name, size_t& helper_index) {
    constexpr std::string_view prefix = "--expert-cache-device";
    if (name.substr(0, prefix.size()) != prefix) return OptionName::NotHelperOption;
    const std::string_view suffix = name.substr(prefix.size());
    unsigned long long ordinal = 0;
    const auto result = std::from_chars(suffix.data(), suffix.data() + suffix.size(), ordinal);
    if (suffix.empty() || result.ec != std::errc{} || result.ptr != suffix.data() + suffix.size() || ordinal == 0 ||
        ordinal > (unsigned long long) std::numeric_limits<size_t>::max())
        return OptionName::InvalidHelperIndex;
    helper_index = (size_t) (ordinal - 1);
    return OptionName::Valid;
}

/// Parse a helper cache size. `auto` is represented by INT_MAX, matching the existing startup sizing path.
inline bool parse_cache_size(std::string_view value, int& slots, bool& automatic) {
    if (value == "auto") {
        slots = std::numeric_limits<int>::max();
        automatic = true;
        return true;
    }
    int parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (value.empty() || result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed < 0) return false;
    slots = parsed;
    automatic = false;
    return true;
}

inline bool has_enabled_helpers(const std::vector<int>& slots) {
    for (const int value : slots) if (value > 0) return true;
    return false;
}

inline size_t enabled_helper_count(const std::vector<int>& slots) {
    size_t count = 0;
    for (const int value : slots) if (value > 0) ++count;
    return count;
}

inline bool has_helper_gap(const std::vector<int>& slots) {
    bool gap = false;
    for (const int value : slots) {
        if (value > 0 && gap) return true;
        if (value == 0) gap = true;
    }
    return false;
}

}  // namespace strata::program::remote_helpers
