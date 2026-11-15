// SPDX-License-Identifier: MIT
// Original deterministic acceptance fixture. candidate.hpp must contain only
// the generated unique_sorted function. No model code is downloaded or run here.
#include <algorithm>
#include <climits>
#include <random>
#include <vector>
#include "candidate.hpp"

int main() {
    std::vector<std::vector<int>> tests = {
        {}, {1}, {2, 1, 2, -3, -3}, {INT_MIN, INT_MAX, 0, INT_MIN}
    };
    std::mt19937 random(42);
    for (int t = 0; t < 100; ++t) {
        std::vector<int> values;
        for (int j = 0; j < t; ++j) values.push_back(int(random() % 101) - 50);
        tests.push_back(values);
    }
    for (auto values : tests) {
        auto expected = values;
        std::sort(expected.begin(), expected.end());
        expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
        if (unique_sorted(values) != expected) return 1;
    }
    return 0;
}
