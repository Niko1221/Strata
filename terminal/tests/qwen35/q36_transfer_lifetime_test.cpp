#define _CRT_SECURE_NO_WARNINGS
#include "strata/qwen35/expert_store.hpp"

#include <cstdio>
#include <cstring>

using strata::qwen35::ExpertPinPool;
using strata::qwen35::PinSlotState;

namespace {

int g_fails = 0;

void check(bool ok, const char* text, int line) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s:%d %s\n", "q36_transfer_lifetime_test.cpp", line, text);
    ++g_fails;
}

#define CHECK(cond) check((cond), #cond, __LINE__)

}  // namespace

int main() {
    ExpertPinPool pool;
    CHECK(pool.slot_count() == ExpertPinPool::kDefaultSlots);
    CHECK(pool.slot_bytes() == ExpertPinPool::kDefaultSlotBytes);
    CHECK(pool.committed_bytes() == ExpertPinPool::kDefaultSlots * ExpertPinPool::kDefaultSlotBytes);
    CHECK(pool.committed_bytes() == (64ull << 20));

    const int a = pool.begin_fill();
    const int b = pool.begin_fill();
    CHECK(a == 0 || a == 1);
    CHECK(b == 0 || b == 1);
    CHECK(a != b);
    CHECK(pool.begin_fill() == -1);
    CHECK(pool.state(a) == PinSlotState::HostFilling);
    CHECK(pool.state(b) == PinSlotState::HostFilling);

    uint8_t* pa = pool.data(a);
    uint8_t* pb = pool.data(b);
    CHECK(pa != nullptr && pb != nullptr && pa != pb);
    std::memset(pa, 0xAB, 64);
    std::memset(pb, 0xCD, 64);
    CHECK(pa[0] == 0xAB);
    CHECK(pb[0] == 0xCD);

    CHECK(!pool.mark_h2d(-1));
    CHECK(!pool.mark_h2d(99));
    CHECK(pool.mark_h2d(a));
    CHECK(pool.state(a) == PinSlotState::H2dInFlight);
    CHECK(!pool.mark_h2d(a));  // already in flight
    CHECK(pool.mark_h2d(b));
    CHECK(pool.state(b) == PinSlotState::H2dInFlight);

    CHECK(!pool.release(-1));
    CHECK(pool.release(a));
    CHECK(pool.state(a) == PinSlotState::Reusable);
    CHECK(!pool.release(a));  // already free
    CHECK(pool.release(b));
    CHECK(pool.state(b) == PinSlotState::Reusable);

    const int again = pool.begin_fill();
    CHECK(again == a || again == b);
    CHECK(pool.state(again) == PinSlotState::HostFilling);
    // HostFilling may release without H2D (cancel / failure path).
    CHECK(pool.release(again));
    CHECK(pool.state(again) == PinSlotState::Reusable);

    ExpertPinPool tiny(1, 4096);
    CHECK(tiny.slot_count() == 1);
    CHECK(tiny.slot_bytes() == 4096);
    CHECK(tiny.committed_bytes() == 4096);
    CHECK(tiny.begin_fill() == 0);
    CHECK(tiny.begin_fill() == -1);
    CHECK(tiny.owns_buffers());

    uint8_t ext_a[64] = {}, ext_b[64] = {};
    uint8_t* ext[2] = {ext_a, ext_b};
    ExpertPinPool borrowed(ext, 2, 64);
    CHECK(!borrowed.owns_buffers());
    CHECK(borrowed.slot_count() == 2);
    CHECK(borrowed.slot_bytes() == 64);
    CHECK(borrowed.data(0) == ext_a);
    CHECK(borrowed.data(1) == ext_b);
    CHECK(borrowed.begin_fill() == 0);
    CHECK(borrowed.mark_h2d(0));
    CHECK(borrowed.release(0));

    if (g_fails) {
        std::fprintf(stderr, "%d check(s) failed\n", g_fails);
        return 1;
    }
    std::fprintf(stderr, "q36_transfer_lifetime_test: ok\n");
    return 0;
}
