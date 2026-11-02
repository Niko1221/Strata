#include "strata/program/pipeline_batch.hpp"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

using strata::program::pipeline_batch::Config;
using strata::program::pipeline_batch::request_windows;
using strata::program::pipeline_batch::supported;

static void check(bool ok, const char* why) {
    if (!ok) {
        std::fprintf(stderr, "pipeline_batch_test: %s\n", why);
        std::exit(1);
    }
}

int main() {
    Config c;
    c.enabled = true;
    c.slots = 4;
    c.groups = 2;
    c.groups_explicit = true;
    c.later_stages = 1;
    c.windows = 2;
    c.batch_mtp = true;
    c.piped_drafts = 2;
    check(supported(c), "explicit supported topology rejected");
    auto reject = [](Config other) { check(!supported(other), "unsupported topology admitted"); };
    auto other = c; other.enabled = false; reject(other);
    other = c; other.slots = 0; reject(other); // physical single GPU keeps its existing MTP path
    other = c; other.slots = 2; reject(other);
    other = c; other.groups = 1; reject(other);
    other = c; other.groups_explicit = false; reject(other);
    other = c; other.later_stages = 0; reject(other);
    other = c; other.later_stages = 2; reject(other);
    other = c; other.windows = -1; reject(other);
    other = c; other.windows = 1; reject(other);
    other = c; other.adapt_async = true; reject(other);
    other = c; other.batch_mtp = false; reject(other);
    other = c; other.piped_drafts = 1; reject(other);

    // The regression is the admission boundary, not merely the batch decoder:
    // a diagnostic pw=2 must not re-enable the solo verifiers during BGEN.
    check(request_windows(true, 2, -1, false) == 2, "solo GEN lost its pipeline");
    check(request_windows(true, 2, 0, false) == 0, "first admission used solo pipeline");
    check(request_windows(true, 2, 1, true) == 0, "second admission used solo pipeline");
    check(request_windows(true, 2, -1, true) == 0, "active batch used solo pipeline");
    check(request_windows(true, 2, -1, false) == 2, "return to solo lost pipeline");
    check(request_windows(true, 0, -1, false) == 0, "diagnostic disable ignored");
    for (int w : {0, 1, 2})
        for (int slot : {-1, 0, 1, 3})
            for (bool active : {false, true})
                check(request_windows(false, w, slot, active) == w, "opt-out changed request policy");
    std::puts("pipeline_batch_test: PASS (startup gates; GEN/admit/batch/GEN; opt-out)");
}
