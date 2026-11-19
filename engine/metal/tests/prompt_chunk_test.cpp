// metal/tests/prompt_chunk_test.cpp - the call size --prefill auto hands the engine on Metal: four window row chunks,
// capped by the auto:N ceiling.
//
// It is a host test: prompt_chunk.h is plain arithmetic, so it needs no Metal device and no model.
#include "engine/prompt_chunk.h"

#include <cstdint>
#include <cstdio>

namespace {

int failures = 0;

void check(int64_t inner, int64_t max, int64_t want, const char* what) {
    const int64_t got = strata::metal_engine::auto_prompt_call_rows(inner, max);
    const bool ok = got == want;
    std::printf("%-58s inner=%lld max=%lld -> %lld (want %lld) %s\n", what, (long long) inner, (long long) max,
                (long long) got, (long long) want, ok ? "ok" : "FAILED");
    if (!ok) ++failures;
}

}  // namespace

int main() {
    constexpr int64_t kRows = strata::metal_engine::kAutoPromptChunkRows;
    check(kRows, 8192, 4096, "the default row chunk, four to a call");
    check(kRows, 32768, 4096, "auto:32768 keeps the default four");
    check(kRows, 4096, 4096, "a ceiling equal to four row chunks");
    check(kRows, 2048, 2048, "a ceiling under four row chunks");
    check(4096, 8192, 8192, "4096-row chunks stop at the default ceiling");
    check(4096, 16384, 16384, "auto:16384 allows four 4096-row chunks");
    check(8192, 32768, 32768, "8192-row chunks at the widest ceiling");
    check(64, 8192, 256, "the smallest row chunk");
    check(64, 200, 200, "the smallest row chunk under a smaller ceiling");
    return failures == 0 ? 0 : 1;
}
