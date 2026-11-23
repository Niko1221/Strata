#include "ink_shard_directory.h"
#include <algorithm>
#include <cstdio>

int main() {
    using strata::artifact::StrataExpertEntry;
    std::vector<StrataExpertEntry> entries(8);
    for (size_t i = 0; i < entries.size(); ++i) {
        entries[i].expert_id = (i + 1) % 4;
        entries[i].body_offset = i * 8;
        entries[i].body_bytes = 8;
    }
    const auto valid = [&] { return strata::ink::valid_shard_directory<4>(entries, 2, 64); };
    if (!valid()) return 1;
    std::swap(entries[4], entries[7]);
    if (!valid()) return 2;
    const auto saved = entries.back();
    entries.back().expert_id = entries[4].expert_id;
    if (valid()) return 3;
    entries.back().expert_id = 4;
    if (valid()) return 4;
    entries.back() = saved;
    entries.back().body_offset = UINT64_MAX;
    if (valid()) return 5;
    entries.back().body_offset = 57;
    if (valid()) return 6;
    entries.pop_back();
    if (valid()) return 7;
    std::puts("Inkling shard directory checks passed");
}
