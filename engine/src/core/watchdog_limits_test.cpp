// src/core/watchdog_limits_test.cpp - the serve watchdog's limits parse (#6861: negative "elapsed" seconds from atoi of an out-of-range value).
#include "strata/core/watchdog_limits.hpp"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    using strata::core::watchdog_seconds;
    bool bad = true;
    CHECK(watchdog_seconds(nullptr, 60, &bad) == 60 && !bad);
    CHECK(watchdog_seconds("", 60, &bad) == 60 && !bad);
    CHECK(watchdog_seconds("0", 60, &bad) == 0 && !bad);              // 0 = off, as documented
    CHECK(watchdog_seconds("180", 60, &bad) == 180 && !bad);
    CHECK(watchdog_seconds("604800", 60, &bad) == 604800 && !bad);    // a week: the largest accepted
    // the values atoi turned into garbage or negatives
    CHECK(watchdog_seconds("3099937436", 60, &bad) == 60 && bad);     // past int: atoi gave -1195029860 on the reporter's engine
    CHECK(watchdog_seconds("2387171783", 60, &bad) == 60 && bad);     // -1907795513
    CHECK(watchdog_seconds("99999999999999999999", 60, &bad) == 60 && bad);
    CHECK(watchdog_seconds("-5", 60, &bad) == 60 && bad);
    CHECK(watchdog_seconds("604801", 60, &bad) == 60 && bad);
    CHECK(watchdog_seconds("12s", 60, &bad) == 60 && bad);
    CHECK(watchdog_seconds("abc", 600, &bad) == 600 && bad);
    CHECK(watchdog_seconds(" 5", 60, &bad) == 5 && !bad);             // strtol skips leading blanks, as atoi did
    if (fails == 0) std::printf("watchdog_limits_test: ok\n");
    return fails == 0 ? 0 : 1;
}
