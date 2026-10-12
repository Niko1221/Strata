// A CPU probe for independent hashlib checks of the shared SHA-256 primitive.
#include "strata/core/branch_sha256.hpp"
#include <cstdio>
#include <iostream>
#include <iterator>
#include <vector>

int main() {
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(std::cin)), {});
    strata::core::BranchDigest result;
    strata::core::branch_sha256(bytes.data(), bytes.size(), result);
    for (auto byte : result.bytes) std::printf("%02x", byte);
    std::puts("");
}
