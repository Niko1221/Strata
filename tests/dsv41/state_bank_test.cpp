// Strata. The parked-state bank of the burst-interleaving path: put/get/drop, LRU eviction by bytes and count.
#include "ds4_state_bank.hpp"

#include <cstdio>

using strata::ds4::StateBank;

static int fails = 0;
#define CHECK(x)                                                                                    \
    do {                                                                                            \
        if (!(x)) {                                                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);                                \
            ++fails;                                                                                \
        }                                                                                           \
    } while (0)

static std::vector<uint8_t> blob(size_t n, uint8_t fill) { return std::vector<uint8_t>(n, fill); }

int main() {
    {   // put, get, drop
        StateBank b;
        CHECK(b.put(1, blob(100, 7)));
        CHECK(b.count() == 1 && b.bytes() == 100);
        const std::vector<uint8_t>* s = b.get(1);
        CHECK(s != nullptr && s->size() == 100 && (*s)[0] == 7);
        CHECK(b.get(2) == nullptr);
        CHECK(b.drop(1) && b.count() == 0 && b.bytes() == 0);
        CHECK(!b.drop(1));
    }
    {   // replacing a state updates the byte count
        StateBank b;
        CHECK(b.put(1, blob(100, 1)));
        CHECK(b.put(1, blob(40, 2)));
        CHECK(b.count() == 1 && b.bytes() == 40);
        CHECK((*b.get(1))[0] == 2);
    }
    {   // a get refreshes the LRU order: the untouched one is evicted first
        StateBank::Options o;
        o.max_count = 2;
        o.max_bytes = 1000;
        StateBank b(o);
        CHECK(b.put(1, blob(100, 1)));
        CHECK(b.put(2, blob(100, 2)));
        b.get(1);                              // 1 is now the newest; 2 is the oldest
        CHECK(b.put(3, blob(100, 3)));          // evicts 2
        CHECK(b.has(1) && !b.has(2) && b.has(3));
        CHECK(b.evictions() == 1 && b.evicted_bytes() == 100);
    }
    {   // the byte budget bounds the bank, oldest first
        StateBank::Options o;
        o.max_count = 8;
        o.max_bytes = 250;
        StateBank b(o);
        CHECK(b.put(1, blob(100, 1)));
        CHECK(b.put(2, blob(100, 2)));
        CHECK(b.put(3, blob(100, 3)));          // 300 > 250: 1 goes
        CHECK(!b.has(1) && b.has(2) && b.has(3));
        CHECK(b.bytes() == 200);
    }
    {   // a state larger than the whole budget is refused, and nothing is parked
        StateBank::Options o;
        o.max_bytes = 50;
        StateBank b(o);
        CHECK(!b.put(1, blob(100, 1)));
        CHECK(b.count() == 0 && b.bytes() == 0);
    }
    if (fails == 0) std::printf("ds4_state_bank_test: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
