// Strata. The experts-per-word rule of the DeepSeek V4.1 engine (src/dsv41/ds4_experts_rule.hpp): 5 on a card with 16 GiB or less, 6
// elsewhere, --experts overrides, DS4_TOPK over both.  No GPU needed.
#include "ds4_experts_rule.hpp"

#include <cstdio>
#include <cstdlib>

using strata::ds4::experts_per_word;
using strata::ds4::experts_test_override;
using strata::ds4::copy_all_fits;

static int fails = 0;
#define CHECK_EQ(a, b)                                                                              \
    do {                                                                                            \
        const long long x = (a), y = (b);                                                           \
        if (x != y) {                                                                               \
            std::printf("FAIL %s:%d: %s = %lld, expected %lld\n", __FILE__, __LINE__, #a, x, y);    \
            ++fails;                                                                                \
        }                                                                                           \
    } while (0)

int main() {
    const uint64_t GiB = 1ull << 30;
    // the owner's 12 GB-class cards (RTX 5070 12227 MiB, RTX 3060 12 GB), 16 GiB cards: faster mode by default
    CHECK_EQ(experts_per_word(-1, 12227ull << 20, 6), 5);
    CHECK_EQ(experts_per_word(-1, 12 * GiB, 6), 5);
    CHECK_EQ(experts_per_word(-1, 16 * GiB, 6), 5);
    CHECK_EQ(experts_per_word(-1, 8 * GiB, 6), 5);
    // 24 GB and bigger, and several cards: the model's own 6
    CHECK_EQ(experts_per_word(-1, 16 * GiB + 1, 6), 6);
    CHECK_EQ(experts_per_word(-1, 24 * GiB, 6), 6);
    CHECK_EQ(experts_per_word(-1, 4 * 32 * GiB, 6), 6);
    CHECK_EQ(experts_per_word(-1, 2 * 12 * GiB, 6), 6);   // two 12 GB cards hold 24 GiB in all
    // a GPU that shares the PC's RAM (Strix Halo, ~91 GiB visible): faster mode whatever the size; discrete cards by size only
    CHECK_EQ(experts_per_word(-1, 91 * GiB, 6, true), 5);
    CHECK_EQ(experts_per_word(-1, 24 * GiB, 6, false), 6);   // a discrete 24 GB card
    CHECK_EQ(experts_per_word(-1, 12 * GiB, 6, false), 5);   // a discrete 12 GB card
    CHECK_EQ(experts_per_word(6, 91 * GiB, 6, true), 6);     // --experts 6 wins on the unified-memory GPU too
    CHECK_EQ(experts_per_word(-1, 91 * GiB, 4, true), 4);    // never raised
    // unknown memory (0): the model's count, never a silent cut
    CHECK_EQ(experts_per_word(-1, 0, 6), 6);
    // the flag decides, both ways
    CHECK_EQ(experts_per_word(6, 12 * GiB, 6), 6);
    CHECK_EQ(experts_per_word(4, 12 * GiB, 6), 4);
    CHECK_EQ(experts_per_word(5, 96 * GiB, 6), 5);
    CHECK_EQ(experts_per_word(9, 12 * GiB, 6), 6);   // never more than the model has
    CHECK_EQ(experts_per_word(1, 12 * GiB, 6), 1);
    // a model that routes to 1 (or no) expert: left alone
    CHECK_EQ(experts_per_word(-1, 12 * GiB, 1), 1);
    CHECK_EQ(experts_per_word(-1, 12 * GiB, 0), 0);
    // a model that routes to fewer than 5: the rule never raises it
    CHECK_EQ(experts_per_word(-1, 12 * GiB, 4), 4);
    // the test override wins over the rule and the flag
#if defined(_WIN32)
    _putenv("DS4_TOPK=4");
#else
    setenv("DS4_TOPK", "4", 1);
#endif
    CHECK_EQ(experts_test_override(experts_per_word(-1, 96 * GiB, 6), 6), 4);
    CHECK_EQ(experts_test_override(experts_per_word(6, 12 * GiB, 6), 6), 4);
#if defined(_WIN32)
    _putenv("DS4_TOPK=9");
#else
    setenv("DS4_TOPK", "9", 1);
#endif
    CHECK_EQ(experts_test_override(5, 6), 5);   // out of range: ignored
#if defined(_WIN32)
    _putenv("DS4_TOPK=");
#else
    unsetenv("DS4_TOPK");
#endif
    CHECK_EQ(experts_test_override(5, 6), 5);
    // prompt chunk: prefill_chunk_default(own_default, gpu_share, gguf_copy, file_experts, hip, min_vram, big=8192).  What `strata launch` runs on one card
    // passes neither --tp nor --threads, but the engine's default tp is 2, so own_default is false there; a launch with several cards passes --tp 1.
    using strata::ds4::prefill_chunk_default;
    // RTX 5070 / RTX 3060 (12 GB), launch default: experts in place on a card under 16 GiB -> 4096 (their logs said 8192 before this rule)
    CHECK_EQ(prefill_chunk_default(false, 0.06, false, true, false, 12227ull << 20), 4096);
    CHECK_EQ(prefill_chunk_default(false, 0.06, false, true, false, 12 * GiB), 4096);
    // the same cards with --tp 1 on one card: share 0.06 < 0.5 -> the same 4096
    CHECK_EQ(prefill_chunk_default(true, 0.06, false, true, false, 12 * GiB), 4096);
    // 16 GB cards (15.9 GiB total) 4096; a card with exactly 16 GiB or more keeps the big chunk; short VRAM (8 GB) 4096; unknown VRAM the big one
    CHECK_EQ(prefill_chunk_default(false, 0.10, false, true, false, 16 * GiB - 1), 4096);
    CHECK_EQ(prefill_chunk_default(false, 0.10, false, true, false, 16 * GiB), 8192);
    CHECK_EQ(prefill_chunk_default(false, 0.10, false, true, false, 24 * GiB), 8192);
    CHECK_EQ(prefill_chunk_default(false, 0.0, false, true, false, 8 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(false, 0.0, false, true, false, 0), 8192);
    // a 12 GB HIP card is not measured: the big chunk
    CHECK_EQ(prefill_chunk_default(false, 0.06, false, true, true, 12 * GiB), 8192);
    // a 5090-class card (32 GB, share 0.19): in place 8192; with the experts copied into RAM 4096
    CHECK_EQ(prefill_chunk_default(false, 0.19, false, true, false, 32 * GiB), 8192);
    CHECK_EQ(prefill_chunk_default(false, 0.19, true, true, false, 32 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(true, 0.19, true, true, false, 32 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(false, 0.10, true, true, false, 16 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(false, 0.05, true, true, false, 24 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(false, 0.06, true, true, false, 12227ull << 20), 4096);
    // copied into RAM on a card under 10 GiB (short VRAM): not the copy rule; the in-place rule for a small card gives 4096, unknown VRAM 8192
    CHECK_EQ(prefill_chunk_default(false, 0.0, true, true, false, 8 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(false, 0.0, true, true, false, 0), 8192);
    // 1 x R9700 (HIP, 32 GB, share 0.19), single-card default run (tp 1 passed by a multi-card box): 8192, also when copied (HIP is not measured at 4096)
    CHECK_EQ(prefill_chunk_default(true, 0.19, false, true, true, 32 * GiB), 8192);
    CHECK_EQ(prefill_chunk_default(true, 0.19, true, true, true, 32 * GiB), 8192);
    // 4 x R9700 (share 0.86), single-card default run (--tp 1 --threads off): the small chunk
    CHECK_EQ(prefill_chunk_default(true, 0.86, false, true, true, 32 * GiB), 1024);
    CHECK_EQ(prefill_chunk_default(true, 0.50, false, true, true, 32 * GiB), 1024);
    CHECK_EQ(prefill_chunk_default(true, 0.49, false, true, true, 32 * GiB), 8192);
    // threads / tp runs (own_default false) keep their defaults whatever the share: 8192 streamed, 2048 for a pack that is not a GGUF
    CHECK_EQ(prefill_chunk_default(false, 0.86, false, true, true, 32 * GiB), 8192);
    CHECK_EQ(prefill_chunk_default(false, 0.30, false, false, false, 24 * GiB), 2048);
    // the rule takes no --max-context: a 128K session on a 12 GB card gets the same chunk, because the chunk's buffers and the KV are
    // allocated whole at load for that context and a chunk that does not fit there is halved before the first prompt (4096 -> 2048 -> 1024 ...)
    // copy_all_fits: the 251 GB R9700 box (108 GB lane) copies; a 64 GB PC, a 125 GB box with the 2.25 pack's lane, an unknown RAM do not
    {
        const double GB = 1e9, margin = 8.0 * (1ull << 30);
        CHECK_EQ(copy_all_fits(108 * GB, (uint64_t) (236 * GB), margin, 1e300), 1);
        CHECK_EQ(copy_all_fits(108 * GB, (uint64_t) (110 * GB), margin, 1e300), 0);   // inside the margin
        CHECK_EQ(copy_all_fits(108 * GB, (uint64_t) (116 * GB), margin, 1e300), 0);   // 108 + 8.6 > 116
        CHECK_EQ(copy_all_fits(108 * GB, (uint64_t) (118 * GB), margin, 1e300), 1);
        CHECK_EQ(copy_all_fits(108 * GB, (uint64_t) (101 * GB), margin, 1e300), 0);   // the 125 GB box, free RAM 101 GB
        CHECK_EQ(copy_all_fits(92 * GB, ~uint64_t{0}, margin, 1e300), 0);             // RAM not known
        CHECK_EQ(copy_all_fits(0, (uint64_t) (236 * GB), margin, 1e300), 0);          // nothing on the CPU lane
        // Windows: the commit room (RAM + page file beside its margin) must hold the lane too, whatever the free RAM says
        CHECK_EQ(copy_all_fits(108 * GB, (uint64_t) (236 * GB), margin, 90 * GB), 0);
        CHECK_EQ(copy_all_fits(108 * GB, (uint64_t) (236 * GB), margin, 108 * GB), 1);
        // the owner-PC class: 64 GB RAM, a 12 GB card (a lane of ~120 GB), 40 GB free, a 90 GB commit limit
        CHECK_EQ(copy_all_fits(120 * GB, (uint64_t) (40 * GB), margin, 60 * GB), 0);
        // a 192 GB Windows PC with a 4 GB page file: 150 GB free RAM but only 100 GB of commit room: no
        CHECK_EQ(copy_all_fits(108 * GB, (uint64_t) (150 * GB), margin, 100 * GB), 0);
    }
    if (fails == 0) std::printf("experts rule: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
