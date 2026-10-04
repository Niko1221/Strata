#include "strata/program/helper_residency.hpp"

#include <cstdio>

int main() {
    using strata::program::helper_expert_reserved;
    const std::vector<uint8_t> owners{0, 1, 0, 1};
    if (helper_expert_reserved(owners, 1, false) ||
        helper_expert_reserved(owners, 0, true) ||
        !helper_expert_reserved(owners, 1, true) ||
        !helper_expert_reserved(owners, 3, true) ||
        helper_expert_reserved({}, 0, true) ||
        helper_expert_reserved(owners, owners.size(), true)) return 1;
    // Primary promotion must leave every helper-owned entry out of the candidate set.
    std::vector<std::size_t> candidates;
    for (std::size_t i = 0; i < owners.size(); ++i)
        if (!helper_expert_reserved(owners, i, true)) candidates.push_back(i);
    if (candidates != std::vector<std::size_t>{0, 2}) return 1;
    std::puts("helper_residency_test OK");
}
