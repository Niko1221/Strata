#include "strata/core/request_stop.hpp"
#include <cstdio>
#include <thread>

int main() {
    strata::core::RequestStop stop;
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) { std::fprintf(stderr, "%s\n", what); ++failures; }
    };
    const auto first = stop.epoch();
    stop.begin(first);
    check(!stop.load(), "new request starts normally");
    stop.request();
    check(stop.load(), "active request STOP is visible");
    const auto next = stop.epoch();
    stop.begin(next);
    check(!stop.load(), "STOP before the next GEN is stale");
    // GEN is queued, BACKGROUND wait delays admission, then STOP arrives.
    const auto queued = stop.epoch();
    stop.request();
    stop.begin(queued);
    check(stop.load(), "STOP after queued GEN must survive admission");
    const auto old = stop.epoch();
    stop.request();
    stop.request();
    stop.begin(old);
    check(stop.load(), "multiple queued STOPs remain visible");
    // Either scheduling order is valid, but neither may lose cancellation.
    for (int i = 0; i < 256; ++i) {
        const auto tag = stop.epoch();
        std::thread reader([&] { stop.request(); });
        stop.begin(tag);
        reader.join();
        check(stop.load(), "concurrent STOP cannot be cleared by admission");
    }
    std::printf("request_stop_test: %s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
