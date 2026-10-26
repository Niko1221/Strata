// Strata. The batched-KV block tables (src/dsv41/ds4_batch_kv.hpp, M2 of docs/DSV41_BATCHING.md): slot arithmetic,
// arena exhaustion and the LRU eviction order. No GPU.
#include "ds4_batch_kv.hpp"

#include <cstdio>

using strata::ds4::BatchKv;
using strata::ds4::KvBatch;

static int fails = 0;
#define CHECK(x)                                                                                    \
    do {                                                                                            \
        if (!(x)) {                                                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);                                \
            ++fails;                                                                                \
        }                                                                                           \
    } while (0)

static BatchKv cfg(int32_t blocks, int32_t seqs) {
    BatchKv c;
    c.block_tokens = 4;          // small blocks keep the tests readable
    c.blocks = blocks;
    c.max_seqs = seqs;
    return c;
}

int main() {
    {   // slot arithmetic
        KvBatch b(cfg(8, 2));
        CHECK(b.open(1));
        int32_t ev = -1;
        const int64_t s0 = b.touch(1, 0, &ev);
        CHECK(s0 >= 0 && ev == -1);
        CHECK(b.touch(1, 3, &ev) == s0 + 3);          // same block, offset 3
        const int64_t s4 = b.touch(1, 4, &ev);        // next block
        CHECK(s4 != s0 && s4 >= 0);
        CHECK(b.free_blocks() == 6);
    }
    {   // two sequences do not share a slot
        KvBatch b(cfg(8, 3));
        CHECK(b.open(1) && b.open(2));
        int32_t ev = -1;
        const int64_t a = b.touch(1, 0, &ev);
        const int64_t c = b.touch(2, 0, &ev);
        CHECK(a >= 0 && c >= 0 && a != c);
    }
    {   // exhaustion: an empty arena cannot grow, and eviction keeps the touched sequence alive
        KvBatch b(cfg(1, 2));
        CHECK(b.open(1) && b.open(2));
        int32_t ev = -1;
        CHECK(b.touch(1, 0, &ev) >= 0);               // takes the only block
        CHECK(b.free_blocks() == 0);
        // seq 2 needs a block: the oldest other (seq 1) is evicted
        ev = -1;
        const int64_t s2 = b.touch(2, 0, &ev);
        CHECK(ev == 1);
        CHECK(s2 >= 0);
        CHECK(b.find(1) == nullptr && b.find(2) != nullptr);
    }
    {   // a sequence alone cannot evict itself
        KvBatch b(cfg(1, 1));
        CHECK(b.open(7));
        int32_t ev = -1;
        CHECK(b.touch(7, 0, &ev) >= 0);
        CHECK(b.touch(7, 4, &ev) < 0 && ev == -1);    // no other sequence to evict
    }
    {   // release returns the blocks; close and reopen works
        KvBatch b(cfg(4, 2));
        CHECK(b.open(1));
        int32_t ev = -1;
        CHECK(b.touch(1, 0, &ev) >= 0);
        CHECK(b.free_blocks() == 3);
        b.close(1);
        CHECK(b.free_blocks() == 4 && b.find(1) == nullptr);
        CHECK(b.open(1));
        CHECK(b.touch(1, 0, &ev) >= 0);
    }
    {   // LRU order: touched sequence moves to the back
        KvBatch b(cfg(8, 2));
        CHECK(b.open(1) && b.open(2));
        b.touch_order(1);
        CHECK(b.sequences()[0].seq() == 2 && b.sequences()[1].seq() == 1);
        b.touch_order(2);
        CHECK(b.sequences()[0].seq() == 1 && b.sequences()[1].seq() == 2);
    }
    if (fails == 0) std::printf("ds4_batch_kv_test: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
