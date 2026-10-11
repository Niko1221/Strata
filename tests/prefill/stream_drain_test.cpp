#include "strata/prefill/stream_drain.hpp"
#include <cstdio>
#include <initializer_list>

int main() {
    // A failed compute/copy drain must preserve the original error and must not
    // wait on later streams. Includes the optional KV-prefetch stream (#872).
    for (int count : {2, 3}) {
        for (int failed = -1; failed < count; ++failed) {
            int calls = 0;
            const auto result = strata::prefill::checked_stream_drain([&](int stream) {
                ++calls;
                return stream == failed ? 700 : 0;
            }, count);
            if (result.error != (failed < 0 ? 0 : 700) || result.stream != failed ||
                calls != (failed < 0 ? count : failed + 1)) {
                std::fprintf(stderr, "drain error handling failed: count=%d failure=%d\n", count, failed);
                return 1;
            }
        }
    }
    return 0;
}
