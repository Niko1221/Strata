#include "strata/program/context_boundary.hpp"

#include <cassert>
#include <cstdint>
#include <limits>

int main() {
    using strata::program::serve_context_fits;
    using strata::program::serve_context_window;
    constexpr int64_t capacity = 131072;

    assert(serve_context_fits(130048, 1024, capacity, true));
    assert(serve_context_fits(131071, 1, capacity, true));
    assert(!serve_context_fits(130048, 1025, capacity, true));
    assert(!serve_context_fits(131072, 1, capacity, true));
    assert(!serve_context_fits(130048, 1024, capacity, false));
    assert(serve_context_fits(130040, 1024, capacity, false));
    assert(!serve_context_fits(1, std::numeric_limits<int64_t>::max(), capacity, true));

    assert(serve_context_window(4, 131071, capacity, true) == 1);
    assert(serve_context_window(4, 131070, capacity, true) == 2);
    assert(serve_context_window(4, 131068, capacity, true) == 4);
    assert(serve_context_window(4, 131072, capacity, true) == 0);
    assert(serve_context_window(4, 131071, capacity, false) == 4);
}
