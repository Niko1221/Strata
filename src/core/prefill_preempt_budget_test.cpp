#include "strata/core/prefill_preempt_budget.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace strata::core;
static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
int main() {
    PreemptByteCount n;
    n.add(1 << 20);
    check(n.fits_mib(1), "exact budget fits");
    n.add(1);
    check(!n.fits_mib(1), "one byte over budget rejected");
    check(n.fits_mib(0), "zero disables the cap");
    check(n.fits_mib(uint64_t(1) << 44), "large MiB cap cannot wrap to zero");
    n.bytes = std::numeric_limits<size_t>::max(); n.add(1);
    check(!n.fits_mib(0), "addition overflow fails even with no cap");
    n = {}; n.product(std::numeric_limits<size_t>::max(), 2);
    check(!n.valid, "product overflow fails closed");
    n = {}; n.product(std::numeric_limits<size_t>::max(), 0);
    check(n.valid && n.bytes == 0, "empty product is safe");
    std::vector<ConversationCheckpoint> checkpoints(1);
    checkpoints[0].gdn.resize(2 << 20);
    n = {}; preempt_prompt_bytes(n, 200000, 2048, checkpoints);
    check(!n.fits_mib(2), "checkpoint payload counted before the park copy");
    PreemptByteCount ids; preempt_prompt_bytes(ids, 200000, 2048, {});
    check(ids.bytes == 200000 * sizeof(int64_t) + 2048 * sizeof(int32_t), "full unread prompt and running ids counted");
    check(n.bytes >= ids.bytes + (2 << 20) + sizeof(ConversationCheckpoint), "checkpoint directory counted");
    checkpoints[0].stage_parts.resize(1);
    checkpoints[0].stage_parts[0].dead.resize(500);
    PreemptByteCount parts; preempt_prompt_bytes(parts, 200000, 2048, checkpoints);
    check(parts.bytes >= n.bytes + sizeof(ConversationCheckpoint) + 500, "nested checkpoint accounting");
    std::printf("%d checks passed\n", checks);
}
