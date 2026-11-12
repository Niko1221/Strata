// src/kernels/cpu/pool_sleep_diag.hpp - what `diag()` reports about the workers being parked and asleep in the kernel.
//
// Two tests ask this, both about issue #4624: `pool_spin_test`, which is registered in the CPU configuration CI builds,
// and `pool_tasks_test`, which needs the native expert kernels and is not.  At a zero spin budget every worker should
// have reached the kernel sleep between batches; at a large budget none of them should have slept yet.  This is state
// read from the pool's own report (`pool.cpp` prints ` wN=sleeping` for each worker and "%u of %d workers parked" for
// its own count), so no test needs a clock, a frequency or a cost threshold to tell parked from spinning.
//
// A tmpfile keeps the report out of the test's own output; failing to open or read it reads as -1, so the run fails
// loudly rather than counting a worker asleep that was never reported.
#pragma once

#include "strata/kernels/cpu/pool.hpp"

#include <cstdio>
#include <string>

namespace strata::kernels::cpu {

// The pool's report as one string; empty when the tmpfile could not be opened or read.
inline std::string diag_text(const ExpertPool& pool) {
    std::FILE* tmp = std::tmpfile();
    if (!tmp) return {};
    pool.diag(tmp);
    std::fflush(tmp);
    std::string text;
    char buf[256];
    std::rewind(tmp);
    size_t got;
    while ((got = std::fread(buf, 1, sizeof buf, tmp)) > 0) text.append(buf, got);
    std::fclose(tmp);
    return text;
}

// How many of the pool's workers the report says are asleep in the kernel, read from its own per-worker line.
inline int sleepers_in(const std::string& text, int workers) {
    int asleep = 0;
    for (int i = 0; i < workers; ++i)
        if (text.find(" w" + std::to_string(i) + "=sleeping") != std::string::npos) ++asleep;
    return asleep;
}

inline int sleepers_of(const ExpertPool& pool, int workers) {
    const std::string text = diag_text(pool);
    if (text.empty()) return -1;
    return sleepers_in(text, workers);
}

/// Two state counts from one report, for a test that has to wait for a state rather than guess when it happened.
///
/// `asleep` is the per-worker count above.  `at_park` is the number the pool itself keeps (`parked_` in `pool.cpp`: a
/// worker adds to it when it arrives at the park loop and takes away from it when it leaves), printed as "%u of %d
/// workers parked".  It starts at zero and only a worker thread adds to its own, so `at_park == workers` is evidence
/// that every worker *thread* got to the park loop - which the per-worker state alone cannot show, because the
/// constructor stores `kParked` before the threads run.  Read together with `asleep`, it says where the workers are:
/// at the park, not asleep, with a non-zero spin budget, and the pause stage has run - every worker is inside that
/// stage.
///
/// Either field is -1 when the report could not be read or the phrase it looks for is missing, so a test that waits on
/// them fails rather than concluding a worker was somewhere it was never reported.
struct park_state {
    int at_park;
    int asleep;
};

inline park_state park_state_of(const ExpertPool& pool, int workers) {
    const std::string text = diag_text(pool);
    if (text.empty()) return {-1, -1};
    const int asleep = sleepers_in(text, workers);
    const std::string needle = " workers parked";
    const size_t at = text.find(needle);
    if (at == std::string::npos) return {-1, asleep};
    size_t end = at;   // the digits immediately before the phrase: "4 of 4 workers parked"
    while (end > 0 && text[end - 1] == ' ') --end;
    size_t begin = end;
    while (begin > 0 && text[begin - 1] >= '0' && text[begin - 1] <= '9') --begin;
    if (begin == end) return {-1, asleep};
    return {std::stoi(text.substr(begin, end - begin)), asleep};
}

}   // namespace strata::kernels::cpu
