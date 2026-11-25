// strata/core/ink_expert_policy.hpp: the hot-expert tier rules of strata-inkling (Qwen-style adaptive tier over a pool of slots). No GPU.
#include "strata/core/ink_expert_policy.hpp"

#include <cstdio>
#include <set>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

InkExpertPolicy make(int n_keys, int n_pinned, int n_slots, const InkExpertPolicyCfg& cfg = {}) {
    std::vector<int> pinned, owner;
    for (int i = 0; i < n_pinned; ++i) pinned.push_back(i);
    for (int s = 0; s < n_slots; ++s) owner.push_back(n_pinned + s);
    InkExpertPolicy p;
    p.init(n_keys, pinned, owner, cfg);
    return p;
}
}  // namespace

int main() {
    // 1. a hot expert outside the card takes the coldest slot; pinned experts never move
    {
        auto p = make(100, 4, 4);                       // keys 0..3 pinned, 4..7 in slots, the rest out
        for (int i = 0; i < 10; ++i) p.add_use(50);     // hot, outside
        for (int i = 0; i < 6; ++i) p.add_use(5);
        p.add_use(6, 3.0f); p.add_use(7, 3.0f); p.add_use(4, 3.0f);
        for (int i = 0; i < 20; ++i) p.add_use(0);      // a pinned expert, also hot: irrelevant
        auto sw = p.round();
        CHECK(sw.size() == 1);
        CHECK(sw[0].in_key == 50);
        CHECK(sw[0].out_key >= 4 && sw[0].out_key <= 7);
        CHECK(p.resident(50));
        CHECK(!p.resident(sw[0].out_key));
        CHECK(p.resident(0) && p.resident(1) && p.resident(2) && p.resident(3));   // pinned stay
    }
    // 2. the usage floor: a candidate under it never gets in even when the slots are empty-handed
    {
        auto p = make(100, 0, 2);
        p.add_use(60, 1.0f);                             // under the floor 2.0
        CHECK(p.round().empty());
    }
    // 3. min gain: an expert only slightly hotter than the coldest resident does not swap
    {
        auto p = make(100, 0, 2);
        p.add_use(0, 5.0f); p.add_use(1, 5.0f);
        p.add_use(70, 6.0f);                             // 6.0 <= 5.0 + 1.5
        CHECK(p.round().empty());
        p.add_use(70, 4.0f);                             // decayed 4.2 + 4 = 8.2 > 3.5 + 1.5
        CHECK(p.round().size() == 1);
    }
    // 4. swap cap: a tiny card moves at most 16 experts per round, a big one up to swaps_cap
    {
        auto p = make(1000, 0, 50);                      // 5% resident -> cap 16
        for (int k = 500; k < 540; ++k) p.add_use(k, 10.0f);
        CHECK(p.round().size() == 16);
        auto q = make(1000, 0, 400);                     // 40% resident -> base cap 96
        for (int k = 500; k < 640; ++k) q.add_use(k, 10.0f);
        CHECK(q.round().size() == 96);
        CHECK(ink_policy_swaps_cap(100, 1000, 96) == 16);
        CHECK(ink_policy_swaps_cap(200, 1000, 96) == 32);
        CHECK(ink_policy_swaps_cap(300, 1000, 96) == 96);
        CHECK(ink_policy_swaps_cap(300, 1000, 8) == 8);
    }
    // 5. decay: a burst fades; after enough rounds an old favourite loses against a steady new one
    {
        auto p = make(100, 0, 1);
        p.add_use(0, 10.0f);
        for (int i = 0; i < 6; ++i) p.round();           // 10 * 0.7^6 = 1.2
        CHECK(p.usage(0) < 1.5f);
        p.add_use(80, 4.0f);
        auto sw = p.round();
        CHECK(sw.size() == 1 && sw[0].in_key == 80 && sw[0].out_key == 0);
    }
    // 6. an empty slot is filled by the hottest candidates first
    {
        std::vector<int> owner = {-1, -1, -1};
        InkExpertPolicy p;
        p.init(100, {}, owner);
        p.add_use(10, 3.0f); p.add_use(11, 9.0f); p.add_use(12, 5.0f); p.add_use(13, 4.0f);
        auto sw = p.round();
        CHECK(sw.size() == 3);
        std::set<int> in;
        for (auto& s : sw) { in.insert(s.in_key); CHECK(s.out_key == -1); }
        CHECK(in.count(11) && in.count(12) && in.count(13) && !in.count(10));
        CHECK(p.n_resident() == 3);
    }
    // 7. slots stay a bijection: every resident expert has exactly one slot, no slot is shared (a round over many swaps)
    {
        auto p = make(300, 10, 60);
        unsigned s = 12345u;
        for (int r = 0; r < 50; ++r) {
            for (int i = 0; i < 400; ++i) { s = s * 1664525u + 1013904223u; p.add_use((int)((s >> 8) % 300), 1.0f); }
            p.round();
            std::set<int> slots;
            int res = 0;
            for (int k = 0; k < 300; ++k) {
                const int sl = p.slot_of(k);
                if (sl >= 0) { CHECK(slots.insert(sl).second); ++res; }
            }
            CHECK(res == 60);
            CHECK(p.n_resident() == 70);
            for (int k = 0; k < 10; ++k) CHECK(p.resident(k) && p.slot_of(k) < 0);
        }
    }
    // 8. pin_count: a quarter of the card, at least one
    CHECK(InkExpertPolicy::pin_count(1000, 0.25f) == 250);
    CHECK(InkExpertPolicy::pin_count(2, 0.25f) == 1);
    CHECK(InkExpertPolicy::pin_count(0, 0.25f) == 0);
    CHECK(InkExpertPolicy::pin_count(1000, 5.0f) == 750);   // clamped
    if (failed) { std::fprintf(stderr, "%d checks failed\n", failed); return 1; }
    std::printf("ink_expert_policy_test OK\n");
    return 0;
}
