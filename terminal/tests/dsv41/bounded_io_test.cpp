// tests/dsv41/bounded_io_test.cpp - Test atomic CAS bounded IO and drafter lifecycle
#include "ds4_bounded_io.hpp"

#include <cstdio>
#include <thread>
#include <vector>
#include <atomic>

static int fails = 0;
#define CHECK(c)                                                           \
    do {                                                                   \
        if (!(c)) {                                                        \
            std::printf("FAIL line %d: %s\n", __LINE__, #c);               \
            ++fails;                                                       \
        }                                                                  \
    } while (0)

using namespace strata::dsv41;

static void test_drafter_lifecycle() {
    BoundedIoConfig cfg;
    cfg.drafter_mode = DrafterMode::DisabledUnloaded;
    BoundedIoManager mgr(cfg);

    CHECK(mgr.is_drafter_unloaded());
    CHECK(mgr.allocated_drafter_vram_bytes() == 0);

    mgr.load_drafter();
    CHECK(!mgr.is_drafter_unloaded());
    CHECK(mgr.allocated_drafter_vram_bytes() > 0);

    mgr.unload_drafter();
    CHECK(mgr.is_drafter_unloaded());
    CHECK(mgr.allocated_drafter_vram_bytes() == 0);
}

static void test_bounded_io_limit() {
    BoundedIoConfig cfg;
    cfg.max_pending_io_bytes = 1000;
    BoundedIoManager mgr(cfg);

    CHECK(mgr.reserve_io(600));
    CHECK(mgr.in_flight_bytes() == 600);

    // Exceeds limit
    CHECK(!mgr.reserve_io(500));
    CHECK(mgr.in_flight_bytes() == 600);

    // Fits limit
    CHECK(mgr.reserve_io(400));
    CHECK(mgr.in_flight_bytes() == 1000);

    // Release
    mgr.release_io(700);
    CHECK(mgr.in_flight_bytes() == 300);

    // Release more than in-flight clamps to 0
    mgr.release_io(1000);
    CHECK(mgr.in_flight_bytes() == 0);
}

static void test_generational_cancellation() {
    BoundedIoConfig cfg;
    BoundedIoManager mgr(cfg);

    CHECK(mgr.active_generation_id() == 1);
    CHECK(mgr.reserve_io(5000));
    CHECK(mgr.in_flight_bytes() == 5000);

    // Wrong generation id does not cancel
    mgr.cancel_pending(99);
    CHECK(mgr.active_generation_id() == 1);
    CHECK(mgr.in_flight_bytes() == 5000);

    // Correct generation id cancels
    mgr.cancel_pending(1);
    CHECK(mgr.active_generation_id() == 2);
    CHECK(mgr.in_flight_bytes() == 0);
}

int main() {
    test_drafter_lifecycle();
    test_bounded_io_limit();
    test_generational_cancellation();
    if (fails == 0) {
        std::printf("All bounded IO tests passed.\n");
        return 0;
    }
    return 1;
}
